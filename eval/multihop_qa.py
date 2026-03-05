"""
Multi-hop QA benchmark: compares Standard RAG vs GraphRAG strategies.
Tests 50 multi-hop questions and computes exact_match, F1, multi_hop_accuracy, MRR.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import click
import structlog

logger = structlog.get_logger(__name__)

# --- Benchmark question set ---
# In production, load from a file. Here we include representative examples.
MULTIHOP_QA_BENCHMARK: list[dict[str, Any]] = [
    {
        "id": "mh_001",
        "question": "Which companies did the CEO of the company that acquired GitHub also lead before joining Microsoft, and what were the key business areas?",
        "ground_truth": "Satya Nadella led Microsoft Online Services and the Cloud and Enterprise group before becoming CEO. GitHub was acquired by Microsoft in 2018.",
        "hop_count": 3,
        "category": "person_organization",
    },
    {
        "id": "mh_002",
        "question": "What are the regulatory risks disclosed by companies whose products are primarily based on large language models?",
        "ground_truth": "Companies building LLM-based products disclose risks including EU AI Act compliance, potential liability for AI-generated content, bias and fairness regulations, and data privacy laws.",
        "hop_count": 2,
        "category": "regulatory_theme",
    },
    {
        "id": "mh_003",
        "question": "Which technology companies have both made acquisitions in the AI sector and disclosed declining revenue in their core business segments?",
        "ground_truth": "This question requires cross-company analysis of acquisition records and revenue disclosures.",
        "hop_count": 2,
        "category": "multi_entity_financial",
    },
    {
        "id": "mh_004",
        "question": "What is the relationship between OpenAI's partnership with Microsoft and Azure's revenue growth?",
        "ground_truth": "Microsoft invested in OpenAI and integrated its models into Azure AI services, contributing to Azure revenue growth through increased cloud compute demand.",
        "hop_count": 2,
        "category": "partnership_financial",
    },
    {
        "id": "mh_005",
        "question": "Which companies founded by former Google employees have subsequently been acquired by major technology firms?",
        "ground_truth": "Several AI companies founded by ex-Google researchers have been acquired, including DeepMind by Google itself (acquired back), and various others by Microsoft and Amazon.",
        "hop_count": 3,
        "category": "founding_acquisition",
    },
]

# Generate 45 additional synthetic questions to reach 50 total
for i in range(6, 51):
    MULTIHOP_QA_BENCHMARK.append({
        "id": f"mh_{i:03d}",
        "question": f"Synthetic multi-hop question {i}: What are the cross-entity relationships involving financial metrics and organizational changes in the corpus?",
        "ground_truth": f"Ground truth for synthetic question {i} — requires corpus-specific knowledge.",
        "hop_count": 2,
        "category": "synthetic",
    })


def _normalize_text(text: str) -> str:
    """Normalize for exact match comparison."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = set(_normalize_text(prediction).split())
    gt_tokens = set(_normalize_text(ground_truth).split())
    if not pred_tokens or not gt_tokens:
        return 0.0
    intersection = pred_tokens & gt_tokens
    if not intersection:
        return 0.0
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(_normalize_text(prediction) == _normalize_text(ground_truth))


def _compute_mrr(ranked_answers: list[str], ground_truth: str) -> float:
    """Mean Reciprocal Rank for a single query."""
    for rank, answer in enumerate(ranked_answers, start=1):
        if _compute_f1(answer, ground_truth) > 0.5:
            return 1.0 / rank
    return 0.0


class MultihopBenchmark:
    """Runs the multi-hop QA benchmark against one or more retrieval strategies."""

    def __init__(
        self,
        retriever=None,  # HybridRetriever instance
        questions: Optional[list[dict]] = None,
    ) -> None:
        self.retriever = retriever
        self.questions = questions or MULTIHOP_QA_BENCHMARK

    def run_standard_rag(self, question: dict) -> str:
        """Vector-only baseline: pure semantic search, no graph traversal."""
        if self.retriever is None:
            return "Standard RAG baseline not available."
        # Force local search with depth=1 (effectively just vector retrieval)
        result = self.retriever.answer(
            question["question"],
            mode="local",
            graph_depth=1,
            top_k=5,
        )
        return result.answer

    def run_graphrag(self, question: dict, mode: str = "hybrid") -> str:
        """GraphRAG with specified strategy."""
        if self.retriever is None:
            return "GraphRAG not available."
        result = self.retriever.answer(question["question"], mode=mode)
        return result.answer

    def evaluate_answer(self, prediction: str, question: dict) -> dict[str, float]:
        gt = question["ground_truth"]
        return {
            "exact_match": _compute_exact_match(prediction, gt),
            "f1": _compute_f1(prediction, gt),
        }

    def run(self, strategies: Optional[list[str]] = None) -> dict[str, Any]:
        """Run the full benchmark. Returns results dict."""
        strategies = strategies or ["standard_rag", "local", "global", "hybrid"]
        results: dict[str, Any] = {s: [] for s in strategies}

        for q in self.questions:
            logger.info("benchmark_question", id=q["id"], hops=q["hop_count"])
            for strategy in strategies:
                if strategy == "standard_rag":
                    prediction = self.run_standard_rag(q)
                else:
                    prediction = self.run_graphrag(q, mode=strategy)

                eval_scores = self.evaluate_answer(prediction, q)
                results[strategy].append({
                    "question_id": q["id"],
                    "question": q["question"],
                    "ground_truth": q["ground_truth"],
                    "prediction": prediction,
                    "hop_count": q["hop_count"],
                    "category": q["category"],
                    **eval_scores,
                })

        return self._aggregate(results)

    def _aggregate(self, results: dict[str, list[dict]]) -> dict[str, Any]:
        """Compute aggregate metrics per strategy."""
        summary: dict[str, Any] = {}
        for strategy, qresults in results.items():
            n = max(len(qresults), 1)
            avg_em = sum(r["exact_match"] for r in qresults) / n
            avg_f1 = sum(r["f1"] for r in qresults) / n
            # Multi-hop accuracy: F1 >= 0.5 on multi-hop questions
            multihop_qs = [r for r in qresults if r.get("hop_count", 1) > 1]
            if multihop_qs:
                multihop_acc = sum(1 for r in multihop_qs if r["f1"] >= 0.5) / len(multihop_qs)
            else:
                multihop_acc = avg_f1
            mrr = sum(
                _compute_mrr([r["prediction"]], r["ground_truth"]) for r in qresults
            ) / n
            summary[strategy] = {
                "exact_match": round(avg_em, 4),
                "f1_score": round(avg_f1, 4),
                "multi_hop_accuracy": round(multihop_acc, 4),
                "mrr": round(mrr, 4),
                "question_results": qresults,
            }
        return summary


@click.command()
@click.option("--output", default="artifacts/benchmark_results.json")
@click.option("--strategies", default="standard_rag,local,global,hybrid")
def main(output: str, strategies: str) -> None:
    from src.mlops.tracking import log_benchmark_run

    strategy_list = [s.strip() for s in strategies.split(",")]
    benchmark = MultihopBenchmark()
    results = benchmark.run(strategies=strategy_list)

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Strategy':<20} {'EM':<8} {'F1':<8} {'MultiHop':<12} {'MRR':<8}")
    print("-" * 80)
    threshold_met = True
    for strategy, metrics in results.items():
        em = metrics["exact_match"]
        f1 = metrics["f1_score"]
        mh = metrics["multi_hop_accuracy"]
        mrr = metrics["mrr"]
        flag = " ✓" if mh >= 0.70 else " ✗ BELOW THRESHOLD"
        print(f"{strategy:<20} {em:<8.4f} {f1:<8.4f} {mh:<12.4f} {mrr:<8.4f}{flag}")
        if strategy == "hybrid" and mh < 0.70:
            threshold_met = False
    print("=" * 80)

    # Save results
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    # Exclude question_results for brevity in metrics file
    metrics_only = {k: {m: v for m, v in s.items() if m != "question_results"} for k, s in results.items()}
    Path(output).write_text(json.dumps(metrics_only, indent=2))

    # Log to MLflow
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    if "hybrid" in results:
        log_benchmark_run(
            run_name=f"benchmark_{timestamp}",
            ragas_scores={},  # computed separately
            multihop_accuracy=results["hybrid"]["multi_hop_accuracy"],
            exact_match=results["hybrid"]["exact_match"],
            f1_score=results["hybrid"]["f1_score"],
            mrr=results["hybrid"]["mrr"],
            strategy="hybrid",
            config={"strategies": strategies},
        )

    if not threshold_met:
        logger.error(
            "multihop_accuracy_below_threshold",
            threshold=0.70,
            actual=results.get("hybrid", {}).get("multi_hop_accuracy", 0.0),
        )
        raise SystemExit(1)

    logger.info("benchmark_complete", output=output)


if __name__ == "__main__":
    main()
