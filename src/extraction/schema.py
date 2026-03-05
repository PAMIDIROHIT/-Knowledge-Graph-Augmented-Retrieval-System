"""
Pydantic v2 schemas for the GraphRAG extraction pipeline.
All entity IDs: ent_{chunk_id}_{sequential_number}
All relation IDs: rel_{source_ent_id}_{target_ent_id}_{type}
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class EntityType(str, Enum):
    ORGANIZATION = "ORGANIZATION"
    PERSON = "PERSON"
    LOCATION = "LOCATION"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    FINANCIAL_METRIC = "FINANCIAL_METRIC"
    DATE = "DATE"
    TECHNOLOGY = "TECHNOLOGY"
    OTHER = "OTHER"


class RelationType(str, Enum):
    ACQUIRED = "ACQUIRED"
    ACQUIRED_BY = "ACQUIRED_BY"
    HAS_CEO = "HAS_CEO"
    IS_CEO_OF = "IS_CEO_OF"
    PREVIOUSLY_LED = "PREVIOUSLY_LED"
    WORKS_FOR = "WORKS_FOR"
    LOCATED_IN = "LOCATED_IN"
    COMPETES_WITH = "COMPETES_WITH"
    PARTNERS_WITH = "PARTNERS_WITH"
    INVESTED_IN = "INVESTED_IN"
    FOUNDED = "FOUNDED"
    PRODUCES = "PRODUCES"
    REPORTED = "REPORTED"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    RELATED_TO = "RELATED_TO"


class Entity(BaseModel):
    id: str = Field(..., description="Unique entity identifier: ent_{chunk_id}_{seq}")
    name: str = Field(..., min_length=1, description="Canonical entity name")
    type: EntityType = Field(..., description="Entity type classification")
    description: str = Field(default="", description="Short entity description")
    source_chunk_id: str = Field(..., description="ID of the source document chunk")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    aliases: list[str] = Field(default_factory=list, description="Alternative names")
    attributes: dict[str, Any] = Field(default_factory=dict, description="Extra attributes")

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Entity name must not be blank")
        return v.strip()

    @field_validator("id")
    @classmethod
    def id_format(cls, v: str) -> str:
        if not v.startswith("ent_"):
            raise ValueError("Entity ID must start with 'ent_'")
        return v


class Relation(BaseModel):
    id: str = Field(..., description="Unique relation identifier: rel_{src}_{tgt}_{type}")
    source_entity_id: str = Field(..., description="Source entity ID")
    target_entity_id: str = Field(..., description="Target entity ID")
    relation_type: str = Field(..., description="Relation type string")
    description: str = Field(default="", description="Relation description / evidence")
    source_chunk_id: str = Field(..., description="ID of the source document chunk")
    confidence: float = Field(..., ge=0.0, le=1.0)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def id_format(cls, v: str) -> str:
        if not v.startswith("rel_"):
            raise ValueError("Relation ID must start with 'rel_'")
        return v

    @field_validator("source_entity_id", "target_entity_id")
    @classmethod
    def entity_id_format(cls, v: str) -> str:
        if not v.startswith("ent_"):
            raise ValueError("Entity reference ID must start with 'ent_'")
        return v


class ExtractionResult(BaseModel):
    chunk_id: str = Field(..., description="Document chunk ID this result came from")
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    extraction_latency_ms: float = Field(default=0.0, description="LLM call latency (ms)")
    model_used: str = Field(default="claude-sonnet-4-6")

    @model_validator(mode="after")
    def validate_relation_references(self) -> "ExtractionResult":
        entity_ids = {e.id for e in self.entities}
        for rel in self.relations:
            if rel.source_entity_id not in entity_ids:
                raise ValueError(
                    f"Relation {rel.id} references unknown source entity {rel.source_entity_id}"
                )
            if rel.target_entity_id not in entity_ids:
                raise ValueError(
                    f"Relation {rel.id} references unknown target entity {rel.target_entity_id}"
                )
        return self


class DocumentChunk(BaseModel):
    id: str = Field(..., description="Chunk identifier")
    text: str = Field(..., description="Raw chunk text")
    source_document: str = Field(..., description="Source document path or name")
    chunk_index: int = Field(..., description="Order within the source document")
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk text must not be blank")
        return v


class GraphNode(BaseModel):
    """Unified node representation for Neo4j and NetworkX."""
    id: str
    name: str
    type: str
    description: str = ""
    embedding: Optional[list[float]] = None
    community_id: Optional[str] = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class Community(BaseModel):
    id: str = Field(..., description="Community identifier: com_{level}_{number}")
    level: int = Field(..., ge=0, description="Leiden hierarchy level")
    entity_ids: list[str] = Field(default_factory=list, description="Member entity IDs")
    summary: str = Field(default="", description="LLM-generated community summary")
    title: str = Field(default="", description="Short community title")
    key_entities: list[str] = Field(default_factory=list, description="Top entity names")
    embedding: Optional[list[float]] = None
    size: int = Field(default=0)

    @model_validator(mode="after")
    def compute_size(self) -> "Community":
        self.size = len(self.entity_ids)
        return self


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The user question")
    search_mode: str = Field(
        default="auto",
        description="Search mode: auto | local | global | hybrid",
        pattern="^(auto|local|global|hybrid)$",
    )
    top_k: int = Field(default=10, ge=1, le=50)
    graph_depth: int = Field(default=2, ge=1, le=5)


class Citation(BaseModel):
    source: str
    chunk_id: str
    text: str
    entity_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    reasoning_path: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    graph_nodes_traversed: list[str] = Field(default_factory=list)
    retrieval_strategy: str = ""
    latency_ms: float = 0.0
