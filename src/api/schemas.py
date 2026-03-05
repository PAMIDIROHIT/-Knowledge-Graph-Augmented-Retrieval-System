"""
Pydantic request/response models for the FastAPI API layer.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The user's question")
    search_mode: str = Field(
        default="auto",
        description="Search strategy: auto | local | global | hybrid",
        pattern="^(auto|local|global|hybrid)$",
    )
    top_k: int = Field(default=10, ge=1, le=50)
    graph_depth: int = Field(default=2, ge=1, le=5)
    stream: bool = Field(default=False, description="Enable streaming response")


class CitationOut(BaseModel):
    source: str
    chunk_id: str
    text: str
    entity_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    reasoning_path: list[str] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)
    graph_nodes_traversed: list[str] = Field(default_factory=list)
    retrieval_strategy: str = ""
    latency_ms: float = 0.0


class GraphExploreRequest(BaseModel):
    entity_name: str = Field(..., min_length=1)
    depth: int = Field(default=2, ge=1, le=5)


class GraphNode(BaseModel):
    id: str
    name: str
    type: str
    description: str = ""


class GraphEdge(BaseModel):
    id: str = ""
    source: str
    target: str
    relation_type: str
    description: str = ""
    confidence: float = 0.0


class GraphExploreResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    center_entity: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    neo4j_connected: bool
    index_loaded: bool
    graph_node_count: int
    graph_edge_count: int
    community_count: int
