"""
Leiden community detection + LLM-based community summary generation using Grok.
Stores communities back in Neo4j as Community nodes.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import openai
import networkx as nx
import structlog

from src.extraction.prompts import (
    COMMUNITY_SUMMARY_SYSTEM_PROMPT,
    COMMUNITY_SUMMARY_USER_TEMPLATE,
)
from src.extraction.schema import Community

logger = structlog.get_logger(__name__)


def run_leiden(
    graph: nx.Graph,
    resolution: float = 1.0,
    n_iterations: int = 10,
    random_seed: int = 42,
    min_community_size: int = 3,
) -> dict[str, int]:
    """
    Run Leiden algorithm using graspologic.
    Returns {node_id: community_label} mapping.
    """
    try:
        from graspologic.partition import leiden  # type: ignore[import]
    except ImportError:
        logger.warning("graspologic_not_available_using_louvain_fallback")
        return _louvain_fallback(graph, random_seed)

    if graph.number_of_nodes() == 0:
        return {}

    # graspologic leiden expects undirected graph
    undirected = graph.to_undirected() if nx.is_directed(graph) else graph
    partition = leiden(
        undirected,
        resolution=resolution,
        extra_forced_iterations=n_iterations,
        random_seed=random_seed,
    )

    # Filter out tiny communities by relabeling small ones to -1
    from collections import Counter
    counts = Counter(partition.values())
    result = {
        node: label if counts[label] >= min_community_size else -1
        for node, label in partition.items()
    }
    logger.info(
        "leiden_complete",
        communities=len({v for v in result.values() if v != -1}),
        noise_nodes=sum(1 for v in result.values() if v == -1),
    )
    return result


def _louvain_fallback(graph: nx.Graph, seed: int = 42) -> dict[str, int]:
    """Fallback to NetworkX Louvain when graspologic unavailable."""
    try:
        from networkx.algorithms.community import louvain_communities  # type: ignore
        communities = louvain_communities(graph.to_undirected(), seed=seed)
        partition: dict[str, int] = {}
        for i, comm in enumerate(communities):
            for node in comm:
                partition[node] = i
        logger.info("louvain_fallback_complete", communities=len(communities))
        return partition
    except Exception as exc:
        logger.error("louvain_fallback_failed", error=str(exc))
        # Last resort: each node is its own community
        return {node: i for i, node in enumerate(graph.nodes())}


class CommunityDetector:
    """
    Orchestrates Leiden community detection and LLM-based summary generation.
    """

    def __init__(
        self,
        summary_model: str = "grok-3",
        max_communities_to_summarize: int = 500,
        api_key: Optional[str] = None,
    ) -> None:
        self.summary_model = summary_model
        self.max_communities_to_summarize = max_communities_to_summarize
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("GROK_API_KEY"),
            base_url="https://api.x.ai/v1",
        )

    def detect_and_summarize(
        self,
        graph: nx.Graph,
        entities_by_id: dict[str, dict[str, Any]],
        relations: list[dict[str, Any]],
        resolution: float = 1.0,
        n_iterations: int = 10,
        random_seed: int = 42,
        min_community_size: int = 3,
        level: int = 0,
    ) -> list[Community]:
        """Run Leiden, group entities, generate summaries, return Community list."""
        partition = run_leiden(
            graph,
            resolution=resolution,
            n_iterations=n_iterations,
            random_seed=random_seed,
            min_community_size=min_community_size,
        )

        # Group entity IDs by community label
        from collections import defaultdict
        groups: dict[int, list[str]] = defaultdict(list)
        for node_id, label in partition.items():
            if label != -1:
                groups[label].append(node_id)

        communities: list[Community] = []
        # Sort by size descending, cap at max
        sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)
        sorted_groups = sorted_groups[:self.max_communities_to_summarize]

        logger.info("summarizing_communities", count=len(sorted_groups))
        for idx, (label, entity_ids) in enumerate(sorted_groups):
            community_id = f"com_{level}_{label:04d}"
            community = self._summarize_community(
                community_id=community_id,
                level=level,
                entity_ids=entity_ids,
                entities_by_id=entities_by_id,
                relations=relations,
            )
            communities.append(community)
            if idx % 50 == 0:
                logger.info("communities_summarized", progress=idx, total=len(sorted_groups))

        return communities

    def _summarize_community(
        self,
        community_id: str,
        level: int,
        entity_ids: list[str],
        entities_by_id: dict[str, dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> Community:
        """Generate an LLM summary for one community."""
        member_entities = [entities_by_id[eid] for eid in entity_ids if eid in entities_by_id]
        entity_id_set = set(entity_ids)

        # Relations where both endpoints are in this community
        intra_relations = [
            r for r in relations
            if r.get("source_entity_id") in entity_id_set
            and r.get("target_entity_id") in entity_id_set
        ]

        entities_text = "\n".join(
            f"- {e.get('name', '')} ({e.get('type', '')}): {e.get('description', '')}"
            for e in member_entities[:20]  # cap for prompt length
        )
        relations_text = "\n".join(
            f"- {r.get('relation_type', '')}: {r.get('description', '')}"
            for r in intra_relations[:20]
        )

        prompt = COMMUNITY_SUMMARY_USER_TEMPLATE.format(
            community_id=community_id,
            level=level,
            entities_text=entities_text or "No entities",
            relations_text=relations_text or "No intra-community relations",
        )

        try:
            response = self._client.chat.completions.create(
                model=self.summary_model,
                max_tokens=8192,
                timeout=15,
                messages=[
                    {"role": "system", "content": COMMUNITY_SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = (response.choices[0].message.content or "").strip()

            import re
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            summary_data = json.loads(raw)
            title = summary_data.get("title", f"Community {community_id}")
            summary = summary_data.get("summary", "")
            key_entities = summary_data.get("key_entities", [])
        except Exception as exc:
            logger.warning(
                "community_summary_failed",
                community_id=community_id,
                error=str(exc),
            )
            title = f"Community {community_id}"
            summary = f"Community of {len(entity_ids)} entities."
            key_entities = [e.get("name", "") for e in member_entities[:5]]

        return Community(
            id=community_id,
            level=level,
            entity_ids=entity_ids,
            summary=summary,
            title=title,
            key_entities=key_entities,
        )
