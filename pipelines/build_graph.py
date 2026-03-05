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

    # Load .env if present
    env_path = Path("../.env") if not Path(".env").exists() else Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() and k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip()

    entities = json.loads(Path(entities_file).read_text())
    relations = json.loads(Path(relations_file).read_text())
    logger.info("data_loaded", entities=len(entities), relations=len(relations))

    # Connect to Neo4j (optional — pipeline continues without it)
    client = Neo4jClient(
        uri=os.environ.get("NEO4J_URI", graph_cfg.get("neo4j_uri", "bolt://localhost:7687")),
        user=os.environ.get("NEO4J_USER", graph_cfg.get("neo4j_user", "neo4j")),
        password=os.environ.get("NEO4J_PASSWORD", graph_cfg.get("neo4j_password", "graphrag_password")),
        database=graph_cfg.get("neo4j_database", "neo4j"),
        max_connection_pool_size=graph_cfg.get("max_connection_pool_size", 50),
    )

    neo4j_available = client.verify_connectivity()
    if not neo4j_available:
        logger.warning("neo4j_not_available_skipping_graph_writes")

    t0 = time.monotonic()
    neo4j_duration = 0.0

    if neo4j_available:
        t_neo4j = time.monotonic()
        client.create_constraints_and_indexes()

        # Write entities
        for entity in entities:
            client.create_entity_node(entity)
        logger.info("entities_written", count=len(entities))

        # Write relations
        for relation in relations:
            client.create_relation_edge(relation)
        logger.info("relations_written", count=len(relations))
        neo4j_duration = time.monotonic() - t_neo4j

    # Build NetworkX mirror (always — needed for community detection)
    builder = NetworkXBuilder()
    builder.build_from_lists(entities, relations)
    nx_graph = builder.to_undirected()

    # Run Leiden community detection
    entities_by_id = {e["id"]: e for e in entities}
    detector = CommunityDetector(
        summary_model=community_cfg.get("summary_model", "grok-3"),
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

    # Write communities to Neo4j (only if connected)
    if neo4j_available:
        for comm in communities:
            client.create_community_node(comm.model_dump())
            for eid in comm.entity_ids:
                client.link_entity_to_community(eid, comm.id)
        logger.info("communities_written_to_neo4j", count=len(communities))

    # Save artifacts (always)
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    builder.save(artifacts_dir / "networkx_graph.pkl")
    (artifacts_dir / "communities.json").write_text(
        json.dumps([c.model_dump() for c in communities], indent=2)
    )

    duration_s = time.monotonic() - t0
    if neo4j_available:
        stats = client.get_graph_stats()
        graph_nodes = stats["nodes"]
        graph_edges = stats["edges"]
        graph_communities = stats["communities"]
        client.close()
    else:
        graph_nodes = nx_graph.number_of_nodes()
        graph_edges = nx_graph.number_of_edges()
        graph_communities = len(communities)

    metrics = {
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "communities": graph_communities,
        "neo4j_available": neo4j_available,
        "neo4j_write_duration_s": round(neo4j_duration, 2),
        "total_duration_s": round(duration_s, 2),
    }
    Path("data/processed/graph_metrics.json").write_text(json.dumps(metrics, indent=2))

    logger.info("graph_build_complete", **metrics)


if __name__ == "__main__":
    main()
