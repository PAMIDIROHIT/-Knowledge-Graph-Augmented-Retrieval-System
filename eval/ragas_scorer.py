"""
RAGAS metric computation for the GraphRAG system.
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


def compute_ragas_scores(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict[str, float]:
    """
    Compute RAGAS metrics: faithfulness, answer_relevancy,
    context_precision, context_recall.
    
    Returns averaged scores across all QA pairs.
    """
    try:
        from datasets import Dataset  # type: ignore[import]
        from ragas import evaluate  # type: ignore[import]
        from ragas.metrics import (  # type: ignore[import]
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        scores = {
            "faithfulness": float(result["faithfulness"]),
            "answer_relevancy": float(result["answer_relevancy"]),
            "context_precision": float(result["context_precision"]),
            "context_recall": float(result["context_recall"]),
        }
        logger.info("ragas_scores_computed", **scores)
        return scores
    except ImportError:
        logger.warning("ragas_not_available_using_mock_scores")
        return _mock_ragas_scores(questions, answers, ground_truths)
    except Exception as exc:
        logger.error("ragas_computation_failed", error=str(exc))
        return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0}


def _mock_ragas_scores(
    questions: list[str],
    answers: list[str],
    ground_truths: list[str],
) -> dict[str, float]:
    """Simple lexical overlap scores as fallback when RAGAS is unavailable."""
    def token_overlap(a: str, b: str) -> float:
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())
        if not a_tokens or not b_tokens:
            return 0.0
        intersection = a_tokens & b_tokens
        return len(intersection) / max(len(a_tokens), len(b_tokens))

    relevancy_scores = []
    overlap_scores = []
    for q, a, gt in zip(questions, answers, ground_truths):
        relevancy_scores.append(token_overlap(q, a))
        overlap_scores.append(token_overlap(gt, a))

    avg_rel = sum(relevancy_scores) / max(len(relevancy_scores), 1)
    avg_overlap = sum(overlap_scores) / max(len(overlap_scores), 1)

    return {
        "faithfulness": avg_overlap,
        "answer_relevancy": avg_rel,
        "context_precision": avg_overlap,
        "context_recall": avg_overlap,
    }
