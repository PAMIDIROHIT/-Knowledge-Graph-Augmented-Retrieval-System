"""
Builds a NetworkX mirror of the Neo4j knowledge graph for Leiden community detection.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import networkx as nx
import structlog

logger = structlog.get_logger(__name__)


class NetworkXBuilder:
    """
    Maintains an in-memory NetworkX directed graph that mirrors Neo4j.
    Used for running Leiden community detection (graspologic).
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    def add_entity(self, entity: dict[str, Any]) -> None:
        self._graph.add_node(
            entity["id"],
            name=entity.get("name", ""),
            type=entity.get("type", "OTHER"),
            description=entity.get("description", ""),
            confidence=entity.get("confidence", 0.0),
            source_chunk_id=entity.get("source_chunk_id", ""),
        )

    def add_relation(self, relation: dict[str, Any]) -> None:
        src = relation["source_entity_id"]
        tgt = relation["target_entity_id"]
        if src not in self._graph or tgt not in self._graph:
            logger.debug(
                "networkx_skip_dangling_edge",
                src=src,
                tgt=tgt,
                relation_type=relation.get("relation_type"),
            )
            return
        self._graph.add_edge(
            src,
            tgt,
            id=relation["id"],
            relation_type=relation.get("relation_type", "RELATED_TO"),
            confidence=relation.get("confidence", 0.0),
            weight=relation.get("confidence", 1.0),
        )

    def build_from_lists(
        self,
        entities: list[dict[str, Any]],
        relations: list[dict[str, Any]],
    ) -> None:
        """Populate the graph from extracted entity and relation dicts."""
        for entity in entities:
            self.add_entity(entity)
        for relation in relations:
            self.add_relation(relation)
        logger.info(
            "networkx_graph_built",
            nodes=self._graph.number_of_nodes(),
            edges=self._graph.number_of_edges(),
        )

    def to_undirected(self) -> nx.Graph:
        """Return undirected version (required for Leiden algorithm)."""
        return self._graph.to_undirected()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._graph, f)
        logger.info("networkx_graph_saved", path=str(path))

    @classmethod
    def load(cls, path: str | Path) -> "NetworkXBuilder":
        instance = cls()
        with open(path, "rb") as f:
            instance._graph = pickle.load(f)
        logger.info(
            "networkx_graph_loaded",
            path=str(path),
            nodes=instance._graph.number_of_nodes(),
        )
        return instance

    def get_stats(self) -> dict[str, Any]:
        g = self._graph
        return {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
            "density": round(nx.density(g), 6),
            "is_connected": nx.is_weakly_connected(g) if g.number_of_nodes() > 0 else False,
            "components": nx.number_weakly_connected_components(g),
        }
