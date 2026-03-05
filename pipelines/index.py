"""
Stage 4: Embed entity/community nodes and build FAISS index.
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
@click.option("--output", "output_dir", required=True, type=click.Path())
def main(entities_file: str, output_dir: str) -> None:
    from src.graph.indexer import GraphIndexer

    params = _load_params()
    ret_cfg = params.get("retrieval", {})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    entities = json.loads(Path(entities_file).read_text())

    # Load communities if available
    communities_path = Path("artifacts/communities.json")
    communities = json.loads(communities_path.read_text()) if communities_path.exists() else []

    indexer = GraphIndexer(
        index_path=str(out / "faiss_index.bin"),
        id_map_path=str(out / "faiss_id_map.json"),
        embedding_model=ret_cfg.get("embedding_model", "text-embedding-3-small"),
        embedding_dim=int(ret_cfg.get("embedding_dim", 1536)),
    )

    t0 = time.monotonic()
    indexer.build(entities=entities, communities=communities)
    duration_s = time.monotonic() - t0

    metrics = {
        "entities_indexed": len(entities),
        "communities_indexed": len(communities),
        "total_vectors": len(entities) + len(communities),
        "indexing_duration_s": round(duration_s, 2),
    }
    Path("data/processed/index_metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("indexing_complete", **metrics)


if __name__ == "__main__":
    main()
