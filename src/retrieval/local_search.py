"""
LOCAL search: entity-anchored graph traversal.
Finds top-k entities via vector similarity, then traverses their Neo4j neighborhoods.
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

from src.graph.indexer import GraphIndexer
from src.graph.neo4j_client import Neo4jClient

logger = structlog.get_logger(__name__)


class LocalSearcher:
    """
    Entity-anchored retrieval strategy.
    1. Vector search → top-k entity IDs
    2. For each entity: traverse Neo4j neighborhood (configurable depth)
    3. Collect entity nodes, edges, and source chunks as context
    """

    def __init__(
        self,
        indexer: GraphIndexer,
        neo4j_client: Neo4jClient,
        top_k_vector: int = 10,
        top_k_graph: int = 5,
        traversal_depth: int = 2,
    ) -> None:
        self.indexer = indexer
        self.neo4j = neo4j_client
        self.top_k_vector = top_k_vector
        self.top_k_graph = top_k_graph
        self.traversal_depth = traversal_depth

    def search(
        self,
        query: str,
        top_k_vector: Optional[int] = None,
        traversal_depth: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Execute local search and return structured context.
        Returns: {nodes, edges, source_chunks, reasoning_path, entity_ids}
        """
        k_vec = top_k_vector or self.top_k_vector
        depth = traversal_depth or self.traversal_depth

        # Step 1: Vector similarity search → entity IDs
        vector_hits = self.indexer.search(query, top_k=k_vec)
        # Filter to ent_ ids only (exclude community ids)
        entity_hits = [(eid, score) for eid, score in vector_hits if eid.startswith("ent_")]

        reasoning_path = [
            f"VECTOR SEARCH: Found {len(entity_hits)} entity candidates for query '{query[:60]}...'"
        ]

        if not entity_hits:
            logger.info("local_search_no_entities", query=query[:60])
            return {
                "nodes": [],
                "edges": [],
                "source_chunks": [],
                "reasoning_path": reasoning_path,
                "entity_ids": [],
            }

        # Step 2: For each top entity, traverse neighborhood
        all_nodes: dict[str, dict] = {}
        all_edges: dict[str, dict] = {}
        source_chunk_ids: set[str] = set()

        for rank, (entity_id, score) in enumerate(entity_hits[:self.top_k_graph]):
            neighborhood = self.neo4j.get_entity_neighborhood(entity_id, depth=depth)

            for node in neighborhood.get("nodes", []):
                nid = node.get("id", "")
                if nid:
                    all_nodes[nid] = node
                    if node.get("source_chunk_id"):
                        source_chunk_ids.add(node["source_chunk_id"])

            for edge in neighborhood.get("edges", []):
                eid_key = edge.get("id", f"{edge.get('source')}_{edge.get('target')}")
                all_edges[eid_key] = edge

            reasoning_path.append(
                f"GRAPH TRAVERSAL (depth={depth}): Entity '{entity_id}' "
                f"[score={score:.3f}] → {len(neighborhood.get('nodes', []))} neighbors, "
                f"{len(neighborhood.get('edges', []))} edges"
            )

        logger.info(
            "local_search_complete",
            query_len=len(query),
            nodes=len(all_nodes),
            edges=len(all_edges),
            chunks=len(source_chunk_ids),
        )

        return {
            "nodes": list(all_nodes.values()),
            "edges": list(all_edges.values()),
            "source_chunks": list(source_chunk_ids),
            "reasoning_path": reasoning_path,
            "entity_ids": [eid for eid, _ in entity_hits[:self.top_k_graph]],
        }
