"""
Stage 3: Build Neo4j knowledge graph + NetworkX mirror + Leiden communities.
DVC stage: reads entities.json and relations.json, populates Neo4j, runs Leiden.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import click
import structlog

logger = structlog.get_logger(__name__)


def _load_params() -> dict[str, Any]:
    try:
        import yaml
        with open("params.yaml") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


@click.command()
@click.option("--entities", "entities_file", required=True, type=click.Path(exists=True))
@click.option("--relations", "relations_file", required=True, type=click.Path(exists=True))
def main(entities_file: str, relations_file: str) -> None:
    from src.graph.community import CommunityDetector
    from src.graph.neo4j_client import Neo4jClient
    from src.graph.networkx_builder import NetworkXBuilder

    params = _load_params()
    graph_cfg = params.get("graph", {})
    community_cfg = params.get("community", {})

    entities = json.loads(Path(entities_file).read_text())
    relations = json.loads(Path(relations_file).read_text())
    logger.info("data_loaded", entities=len(entities), relations=len(relations))

    # Connect to Neo4j
    client = Neo4jClient(
        uri=os.environ.get("NEO4J_URI", graph_cfg.get("neo4j_uri", "bolt://localhost:7687")),
        user=os.environ.get("NEO4J_USER", graph_cfg.get("neo4j_user", "neo4j")),
        password=os.environ.get("NEO4J_PASSWORD", graph_cfg.get("neo4j_password", "graphrag_password")),
        database=graph_cfg.get("neo4j_database", "neo4j"),
        max_connection_pool_size=graph_cfg.get("max_connection_pool_size", 50),
    )

    if not client.verify_connectivity():
        raise RuntimeError("Cannot connect to Neo4j. Check that the service is running.")

    t0 = time.monotonic()
    client.create_constraints_and_indexes()

    # Write entities
    for entity in entities:
        client.create_entity_node(entity)
    logger.info("entities_written", count=len(entities))

    # Write relations
    for relation in relations:
        client.create_relation_edge(relation)
    logger.info("relations_written", count=len(relations))

    neo4j_duration = time.monotonic() - t0

    # Build NetworkX mirror
    builder = NetworkXBuilder()
    builder.build_from_lists(entities, relations)
    nx_graph = builder.to_undirected()

    # Run Leiden community detection
    entities_by_id = {e["id"]: e for e in entities}
    detector = CommunityDetector(
        summary_model=community_cfg.get("summary_model", "claude-sonnet-4-6"),
        max_communities_to_summarize=community_cfg.get("max_communities_to_summarize", 500),
    )
    communities = detector.detect_and_summarize(
        graph=nx_graph,
        entities_by_id=entities_by_id,
        relations=relations,
        resolution=float(community_cfg.get("resolution", 1.0)),
        n_iterations=int(community_cfg.get("n_iterations", 10)),
        random_seed=int(community_cfg.get("random_seed", 42)),
        min_community_size=int(community_cfg.get("min_community_size", 3)),
    )

    # Write communities to Neo4j
    for comm in communities:
        client.create_community_node(comm.model_dump())
        for eid in comm.entity_ids:
            client.link_entity_to_community(eid, comm.id)

    logger.info("communities_written", count=len(communities))

    # Save artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    builder.save(artifacts_dir / "networkx_graph.pkl")
    (artifacts_dir / "communities.json").write_text(
        json.dumps([c.model_dump() for c in communities], indent=2)
    )

    stats = client.get_graph_stats()
    duration_s = time.monotonic() - t0
    metrics = {
        "graph_nodes": stats["nodes"],
        "graph_edges": stats["edges"],
        "communities": stats["communities"],
        "neo4j_write_duration_s": round(neo4j_duration, 2),
        "total_duration_s": round(duration_s, 2),
    }
    Path("data/processed/graph_metrics.json").write_text(json.dumps(metrics, indent=2))

    logger.info("graph_build_complete", **metrics)
    client.close()


if __name__ == "__main__":
    main()
