"""
MLflow logging helpers for GraphRAG experiment tracking.
"""

from __future__ import annotations

import functools
import os
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

import mlflow
import structlog

logger = structlog.get_logger(__name__)


def _load_params() -> dict[str, Any]:
    try:
        import yaml
        with open("params.yaml") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def setup_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> None:
    """Initialize MLflow tracking."""
    params = _load_params()
    uri = tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI",
        params.get("mlflow", {}).get("tracking_uri", "http://localhost:5001"),
    )
    exp = experiment_name or params.get("mlflow", {}).get(
        "experiment_name", "graphrag_experiments"
    )
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp)
    logger.info("mlflow_initialized", tracking_uri=uri, experiment=exp)


@contextmanager
def mlflow_run(
    run_name: str,
    tags: Optional[dict[str, str]] = None,
) -> Generator[mlflow.ActiveRun, None, None]:
    """Context manager for an MLflow run with automatic cleanup."""
    with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
        logger.info("mlflow_run_started", run_name=run_name, run_id=run.info.run_id)
        yield run
        logger.info("mlflow_run_completed", run_id=run.info.run_id)


def log_extraction_run(
    metrics: dict[str, Any],
    params: dict[str, Any],
) -> None:
    """Log entity extraction pipeline metrics to MLflow."""
    try:
        setup_mlflow()
        with mlflow_run("extract"):
            mlflow.log_params(params)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
    except Exception as exc:
        logger.warning("mlflow_extraction_log_failed", error=str(exc))


def log_graph_build_run(
    metrics: dict[str, Any],
    params: dict[str, Any],
) -> None:
    """Log graph build metrics to MLflow."""
    try:
        setup_mlflow()
        with mlflow_run("build_graph"):
            mlflow.log_params(params)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
    except Exception as exc:
        logger.warning("mlflow_graph_build_log_failed", error=str(exc))


def log_benchmark_run(
    run_name: str,
    ragas_scores: dict[str, float],
    multihop_accuracy: float,
    exact_match: float,
    f1_score: float,
    mrr: float,
    strategy: str,
    config: dict[str, Any],
) -> str:
    """
    Log a benchmark evaluation run to MLflow.
    Returns the MLflow run ID.
    """
    try:
        setup_mlflow()
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_params({
                "strategy": strategy,
                **{k: str(v) for k, v in config.items()},
            })
            mlflow.log_metric("multihop_accuracy", multihop_accuracy)
            mlflow.log_metric("exact_match", exact_match)
            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("mrr", mrr)
            for metric_name, score in ragas_scores.items():
                mlflow.log_metric(f"ragas_{metric_name}", score)
            return run.info.run_id
    except Exception as exc:
        logger.warning("mlflow_benchmark_log_failed", error=str(exc))
        return ""
