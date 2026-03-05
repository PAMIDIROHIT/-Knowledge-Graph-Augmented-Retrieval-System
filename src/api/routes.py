"""
FastAPI routes: /query, /graph/explore, /health
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from src.api.schemas import (
    GraphEdge,
    GraphExploreRequest,
    GraphExploreResponse,
    GraphNode,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    CitationOut,
)
from src.mlops.metrics import (
    ANSWER_QUALITY_GAUGE,
    GRAPH_TRAVERSAL_DEPTH_HISTOGRAM,
    QUERY_LATENCY_HISTOGRAM,
    RETRIEVAL_STRATEGY_COUNTER,
)

router = APIRouter()


def _get_retriever(request: Request):
    return request.app.state.retriever


def _get_neo4j(request: Request):
    return request.app.state.neo4j


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    body: QueryRequest,
    request: Request,
) -> QueryResponse | StreamingResponse:
    """
    Main query endpoint.
    Accepts a question and returns a structured answer with citations.
    Supports streaming when stream=True.
    """
    retriever = _get_retriever(request)

    if body.stream:
        async def _stream_generator() -> AsyncGenerator[str, None]:
            loop = asyncio.get_event_loop()
            gen = retriever.answer_streaming(body.question, mode=body.search_mode)
            for token in gen:
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(_stream_generator(), media_type="text/event-stream")

    # Non-streaming: run in threadpool to avoid blocking the event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: retriever.answer(
            body.question,
            mode=body.search_mode,
            top_k=body.top_k,
            graph_depth=body.graph_depth,
        ),
    )

    # Prometheus metrics
    QUERY_LATENCY_HISTOGRAM.labels(strategy=result.retrieval_strategy).observe(
        result.latency_ms / 1000
    )
    RETRIEVAL_STRATEGY_COUNTER.labels(strategy=result.retrieval_strategy).inc()
    GRAPH_TRAVERSAL_DEPTH_HISTOGRAM.observe(len(result.graph_nodes_traversed))

    return QueryResponse(
        answer=result.answer,
        reasoning_path=result.reasoning_path,
        citations=[CitationOut(**c.model_dump()) for c in result.citations],
        graph_nodes_traversed=result.graph_nodes_traversed,
        retrieval_strategy=result.retrieval_strategy,
        latency_ms=result.latency_ms,
    )


@router.get("/graph/explore", response_model=GraphExploreResponse)
async def graph_explore(
    entity_name: str,
    depth: int = 2,
    request: Request = None,
) -> GraphExploreResponse:
    """
    Explore the neighborhood of a named entity.
    Returns nodes and edges for the frontend D3.js explorer.
    """
    if depth < 1 or depth > 5:
        raise HTTPException(status_code=422, detail="depth must be between 1 and 5")

    neo4j = _get_neo4j(request)

    # Find entity by name
    entity = neo4j.get_entity_by_name(entity_name)
    if not entity:
        # Try fuzzy match
        candidates = neo4j.search_entities_by_name_fuzzy(entity_name, limit=1)
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail=f"Entity '{entity_name}' not found in the knowledge graph",
            )
        entity = candidates[0]

    neighborhood = neo4j.get_entity_neighborhood(entity["id"], depth=depth)

    nodes = [
        GraphNode(
            id=n.get("id", ""),
            name=n.get("name", ""),
            type=n.get("type", "OTHER"),
            description=n.get("description", ""),
        )
        for n in neighborhood.get("nodes", [])
        if n.get("id")
    ]

    edges = [
        GraphEdge(
            id=e.get("id", ""),
            source=e.get("source", ""),
            target=e.get("target", ""),
            relation_type=e.get("relation_type", "RELATED_TO"),
            description=e.get("description", ""),
            confidence=float(e.get("confidence", 0.0)),
        )
        for e in neighborhood.get("edges", [])
        if e.get("source") and e.get("target")
    ]

    return GraphExploreResponse(
        nodes=nodes,
        edges=edges,
        center_entity=entity.get("id"),
    )


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Service health check."""
    neo4j = _get_neo4j(request)
    retriever = _get_retriever(request)

    neo4j_ok = neo4j.verify_connectivity()
    index_loaded = retriever.local.indexer.is_loaded

    stats = {"nodes": 0, "edges": 0, "communities": 0}
    if neo4j_ok:
        try:
            stats = neo4j.get_graph_stats()
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if neo4j_ok else "degraded",
        neo4j_connected=neo4j_ok,
        index_loaded=index_loaded,
        graph_node_count=stats["nodes"],
        graph_edge_count=stats["edges"],
        community_count=stats["communities"],
    )
