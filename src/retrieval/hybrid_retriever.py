"""
Hybrid retriever: routes queries between local/global search strategies
and fuses their scores for a combined ranking. Uses Grok (xAI) for QA.
"""

from __future__ import annotations

import re
import time
from typing import Any, Optional

import openai
import structlog

from src.extraction.prompts import QA_SYSTEM_PROMPT, QA_USER_TEMPLATE
from src.extraction.schema import Citation, QueryResponse
from src.graph.indexer import GraphIndexer
from src.graph.neo4j_client import Neo4jClient
from src.retrieval.global_search import GlobalSearcher
from src.retrieval.local_search import LocalSearcher

logger = structlog.get_logger(__name__)

# Heuristic patterns for routing to global search
GLOBAL_SEARCH_PATTERNS = [
    r"\b(theme|trend|pattern|overview|summary|across|dominant|common|broad)\b",
    r"\b(industry|sector|market|regulatory|all|entire|whole)\b",
    r"\b(what are the|what is the overall|analyze the)\b",
]

COMPILED_GLOBAL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in GLOBAL_SEARCH_PATTERNS
]


def _route_query(question: str, mode: str) -> str:
    """Determine retrieval strategy based on question text and explicit mode."""
    if mode in ("local", "global", "hybrid"):
        return mode

    # Auto-routing heuristic
    for pattern in COMPILED_GLOBAL_PATTERNS:
        if pattern.search(question):
            return "global"
    return "local"


def _build_evidence_text(local_ctx: dict, global_ctx: dict) -> str:
    """Format retrieved context into a single evidence block for the QA prompt."""
    parts: list[str] = []

    if local_ctx.get("nodes"):
        parts.append("=== ENTITY GRAPH CONTEXT ===")
        for node in local_ctx["nodes"][:15]:
            parts.append(
                f"[{node.get('type', 'ENTITY')}] {node.get('name', '')} — {node.get('description', '')} (chunk: {node.get('source_chunk_id', 'N/A')})"
            )

    if local_ctx.get("edges"):
        parts.append("\n=== GRAPH RELATIONS ===")
        for edge in local_ctx["edges"][:20]:
            parts.append(
                f"{edge.get('source', '')} —[{edge.get('relation_type', '')}]→ {edge.get('target', '')}: {edge.get('description', '')}"
            )

    if global_ctx.get("communities"):
        parts.append("\n=== COMMUNITY SUMMARIES ===")
        for comm in global_ctx["communities"][:5]:
            parts.append(
                f"[Community: {comm.get('title', comm.get('id', ''))}]\n{comm.get('summary', '')}"
            )

    if global_ctx.get("supplemental_entities"):
        parts.append("\n=== SUPPLEMENTAL ENTITIES ===")
        for ent in global_ctx["supplemental_entities"][:10]:
            parts.append(f"- {ent.get('name', '')} ({ent.get('type', '')}): {ent.get('description', '')}")

    return "\n".join(parts) if parts else "No relevant evidence retrieved."


class HybridRetriever:
    """
    Routes queries to the appropriate retrieval strategy and generates answers
    using Grok-3 via xAI API.
    """

    def __init__(
        self,
        indexer: GraphIndexer,
        neo4j_client: Neo4jClient,
        top_k_vector: int = 10,
        top_k_graph: int = 5,
        traversal_depth: int = 2,
        vector_weight: float = 0.4,
        graph_weight: float = 0.6,
        qa_model: str = "llama-3.3-70b-versatile",
        qa_max_tokens: int = 16384,
        api_key: Optional[str] = None,
    ) -> None:
        self.local = LocalSearcher(
            indexer=indexer,
            neo4j_client=neo4j_client,
            top_k_vector=top_k_vector,
            top_k_graph=top_k_graph,
            traversal_depth=traversal_depth,
        )
        self.global_ = GlobalSearcher(
            indexer=indexer,
            neo4j_client=neo4j_client,
        )
        self.qa_model = qa_model
        self.qa_max_tokens = qa_max_tokens
        import os
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )

    def retrieve(self, question: str, mode: str = "auto") -> dict[str, Any]:
        """Run retrieval and return structured context."""
        strategy = _route_query(question, mode)
        local_ctx: dict[str, Any] = {"nodes": [], "edges": [], "source_chunks": [], "reasoning_path": [], "entity_ids": []}
        global_ctx: dict[str, Any] = {"communities": [], "supplemental_entities": [], "reasoning_path": [], "community_ids": []}

        if strategy in ("local", "hybrid"):
            local_ctx = self.local.search(question)

        if strategy in ("global", "hybrid"):
            global_ctx = self.global_.search(question)

        reasoning_path = local_ctx.get("reasoning_path", []) + global_ctx.get("reasoning_path", [])
        graph_nodes = local_ctx.get("entity_ids", []) + global_ctx.get("community_ids", [])

        return {
            "strategy": strategy,
            "local_ctx": local_ctx,
            "global_ctx": global_ctx,
            "reasoning_path": reasoning_path,
            "graph_nodes_traversed": graph_nodes,
        }

    def answer(
        self,
        question: str,
        mode: str = "auto",
        top_k: int = 10,
        graph_depth: int = 2,
    ) -> QueryResponse:
        """Full retrieval + generation pipeline."""
        t0 = time.monotonic()

        ctx = self.retrieve(question, mode)
        evidence_text = _build_evidence_text(ctx["local_ctx"], ctx["global_ctx"])
        reasoning_path = ctx["reasoning_path"]
        graph_nodes = ctx["graph_nodes_traversed"]

        qa_prompt = QA_USER_TEMPLATE.format(
            question=question,
            retrieval_strategy=ctx["strategy"],
            reasoning_path="\n".join(f"  {i+1}. {step}" for i, step in enumerate(reasoning_path)),
            evidence_text=evidence_text,
            graph_nodes=", ".join(graph_nodes[:20]) if graph_nodes else "none",
        )

        try:
            response = self._client.chat.completions.create(
                model=self.qa_model,
                max_tokens=self.qa_max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": QA_SYSTEM_PROMPT},
                    {"role": "user", "content": qa_prompt},
                ],
            )
            raw = (response.choices[0].message.content or "").strip()

            import json
            import re as _re
            raw = _re.sub(r"```(?:json)?", "", raw).strip()
            answer_data = json.loads(raw)
            answer_text = answer_data.get("answer", "")
            answer_reasoning = answer_data.get("reasoning_path", reasoning_path)
            raw_citations = answer_data.get("citations", [])
            citations = [Citation(**c) for c in raw_citations if isinstance(c, dict)]
        except Exception as exc:
            logger.error("qa_generation_failed", error=str(exc))
            answer_text = "I could not generate an answer due to an internal error."
            answer_reasoning = reasoning_path
            citations = []

        latency_ms = (time.monotonic() - t0) * 1000

        return QueryResponse(
            answer=answer_text,
            reasoning_path=answer_reasoning,
            citations=citations,
            graph_nodes_traversed=graph_nodes,
            retrieval_strategy=ctx["strategy"],
            latency_ms=round(latency_ms, 1),
        )

    def answer_streaming(self, question: str, mode: str = "auto"):
        """Generator that yields answer tokens via streaming."""
        ctx = self.retrieve(question, mode)
        evidence_text = _build_evidence_text(ctx["local_ctx"], ctx["global_ctx"])
        reasoning_path = ctx["reasoning_path"]

        qa_prompt = QA_USER_TEMPLATE.format(
            question=question,
            retrieval_strategy=ctx["strategy"],
            reasoning_path="\n".join(f"  {i+1}. {step}" for i, step in enumerate(reasoning_path)),
            evidence_text=evidence_text,
            graph_nodes=", ".join(ctx["graph_nodes_traversed"][:20]) or "none",
        )

        with self._client.chat.completions.stream(
            model=self.qa_model,
            max_tokens=self.qa_max_tokens,
            messages=[
                {"role": "system", "content": QA_SYSTEM_PROMPT},
                {"role": "user", "content": qa_prompt},
            ],
        ) as stream:
            for text in stream.text_stream:
                yield text
