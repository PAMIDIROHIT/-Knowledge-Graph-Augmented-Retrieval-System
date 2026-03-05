"""
Stage 2: Entity and relation extraction DVC stage.
Reads chunked docs, runs LLM extraction, writes entities.json and relations.json.
Logs extraction stats to MLflow.
"""

from __future__ import annotations

import asyncio
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
        import yaml  # type: ignore[import]
        with open("params.yaml") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


@click.command()
@click.option("--input", "input_file", required=True, type=click.Path(exists=True))
@click.option("--output", "output_dir", required=True, type=click.Path())
def main(input_file: str, output_dir: str) -> None:
    from src.extraction.entity_extractor import EntityExtractor
    from src.extraction.schema import DocumentChunk

    params = _load_params()
    ext_cfg = params.get("extraction", {})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load .env if present
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            if k.strip() and k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip()

    # Load chunks
    raw_chunks = json.loads(Path(input_file).read_text())
    chunks = [DocumentChunk(**c) for c in raw_chunks]
    logger.info("chunks_loaded", count=len(chunks))

    extractor = EntityExtractor(
        model=ext_cfg.get("model", "llama-3.3-70b-versatile"),
        max_tokens=int(ext_cfg.get("max_tokens", 16384)),
        effort_level=ext_cfg.get("effort_level", "medium"),
        confidence_threshold=float(ext_cfg.get("confidence_threshold", 0.70)),
        max_concurrency=int(ext_cfg.get("max_concurrency", 4)),
    )

    t0 = time.monotonic()
    results = asyncio.run(
        extractor.extract_all(
            chunks,
            batch_size=int(ext_cfg.get("batch_size", 8)),
        )
    )
    duration_s = time.monotonic() - t0

    # Aggregate results
    all_entities = [e.model_dump() for r in results for e in r.entities]
    all_relations = [rel.model_dump() for r in results for rel in r.relations]
    latencies = [r.extraction_latency_ms for r in results if r.extraction_latency_ms > 0]

    # Write outputs
    (out / "entities.json").write_text(json.dumps(all_entities, indent=2))
    (out / "relations.json").write_text(json.dumps(all_relations, indent=2))

    import numpy as np  # type: ignore[import]
    p95_latency = float(np.percentile(latencies, 95)) if latencies else 0.0

    metrics = {
        "entities_extracted": len(all_entities),
        "relations_extracted": len(all_relations),
        "chunks_processed": len(results),
        "extraction_duration_s": round(duration_s, 2),
        "extraction_latency_p95_ms": round(p95_latency, 1),
    }
    (out / "extraction_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Log to MLflow if available
    try:
        import mlflow
        params_cfg = _load_params()
        mlflow_uri = params_cfg.get("mlflow", {}).get("tracking_uri", "http://localhost:5001")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(params_cfg.get("mlflow", {}).get("experiment_name", "graphrag_experiments"))
        with mlflow.start_run(run_name="extract"):
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            mlflow.log_params({
                "model": ext_cfg.get("model", "claude-sonnet-4-6"),
                "confidence_threshold": ext_cfg.get("confidence_threshold", 0.70),
                "chunk_size": params.get("extraction", {}).get("chunk_size", 1024),
            })
    except Exception as exc:
        logger.warning("mlflow_logging_failed", error=str(exc))

    logger.info("extraction_complete", **metrics)


if __name__ == "__main__":
    main()
