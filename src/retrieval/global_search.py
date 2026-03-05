"""
GLOBAL search: community summary retrieval for broad thematic questions.
Retrieves relevant Community nodes by vector similarity,
then supplements with high-confidence entity chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import structlog

from src.graph.indexer import GraphIndexer
from src.graph.neo4j_client import Neo4jClient

logger = structlog.get_logger(__name__)

_COMMUNITY_CACHE: dict[str, dict] = {}  # community_id → community data


def _load_community_cache(artifacts_dir: str = "artifacts") -> None:
    """Load communities.json into memory once for Neo4j-less fallback."""
    path = Path(artifacts_dir) / "communities.json"
    if not path.exists() or _COMMUNITY_CACHE:
        return
    for comm in json.loads(path.read_text()):
        _COMMUNITY_CACHE[comm["id"]] = comm


class GlobalSearcher:
    """
    Community-anchored retrieval strategy (global search).
    1. Vector search → top-k community IDs
    2. Retrieve community summaries from Neo4j
    3. Optionally fetch the key entity chunks for supplemental detail
    """

    def __init__(
        self,
        indexer: GraphIndexer,
        neo4j_client: Neo4jClient,
        top_k_communities: int = 5,
        supplement_with_entities: bool = True,
        top_k_entity_supplement: int = 10,
    ) -> None:
        self.indexer = indexer
        self.neo4j = neo4j_client
        self.top_k_communities = top_k_communities
        self.supplement_with_entities = supplement_with_entities
        self.top_k_entity_supplement = top_k_entity_supplement
        _load_community_cache()

    def search(
        self,
        query: str,
        top_k_communities: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Execute global search.
        Returns: {communities, supplemental_entities, reasoning_path, community_ids}
        """
        k = top_k_communities or self.top_k_communities

        # Step 1: Vector search for communities
        all_hits = self.indexer.search(query, top_k=k + 20)
        community_hits = [(cid, score) for cid, score in all_hits if cid.startswith("com_")][:k]

        reasoning_path = [
            f"COMMUNITY DETECTION: Searching for relevant community summaries for '{query[:60]}...'"
        ]

        if not community_hits:
            logger.info("global_search_no_communities", query=query[:60])
            return {
                "communities": [],
                "supplemental_entities": [],
                "reasoning_path": reasoning_path,
                "community_ids": [],
            }

        # Step 2: Fetch community data from Neo4j (fallback to communities.json)
        communities: list[dict] = []
        for comm_id, score in community_hits:
            try:
                results = self.neo4j.run_cypher(
                    "MATCH (c:Community {id: $id}) RETURN c LIMIT 1",
                    {"id": comm_id},
                )
                comm_data = dict(results[0]["c"]) if results and results[0].get("c") else None
            except Exception:
                comm_data = None
            if comm_data is None:
                comm_data = dict(_COMMUNITY_CACHE.get(comm_id, {}))
            if comm_data:
                comm_data["_search_score"] = score
                communities.append(comm_data)
                reasoning_path.append(
                    f"COMMUNITY SUMMARY: '{comm_data.get('title', comm_id)}' "
                    f"[score={score:.3f}] size={comm_data.get('size', '?')}"
                )

        # Step 3: Supplemental entity search if requested (skip Neo4j if unavailable)
        supplemental_entities: list[dict] = []
        if self.supplement_with_entities and communities:
            entity_hits = [
                (eid, score)
                for eid, score in all_hits
                if eid.startswith("ent_")
            ][:self.top_k_entity_supplement]

            for ent_id, _ in entity_hits:
                try:
                    ent_results = self.neo4j.run_cypher(
                        "MATCH (e:Entity {id: $id}) RETURN e LIMIT 1",
                        {"id": ent_id},
                    )
                    if ent_results and ent_results[0].get("e"):
                        supplemental_entities.append(dict(ent_results[0]["e"]))
                except Exception:
                    supplemental_entities.append({"id": ent_id})

            reasoning_path.append(
                f"VECTOR SEARCH: Supplemented with {len(supplemental_entities)} entity chunks"
            )

        logger.info(
            "global_search_complete",
            communities=len(communities),
            supplemental=len(supplemental_entities),
        )

        return {
            "communities": communities,
            "supplemental_entities": supplemental_entities,
            "reasoning_path": reasoning_path,
            "community_ids": [cid for cid, _ in community_hits],
        }
