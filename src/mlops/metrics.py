"""
Prometheus metric definitions for the GraphRAG API.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

QUERY_LATENCY_HISTOGRAM = Histogram(
    "graphrag_query_latency_seconds",
    "End-to-end query latency in seconds",
    labelnames=["strategy"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)

RETRIEVAL_STRATEGY_COUNTER = Counter(
    "graphrag_retrieval_strategy_total",
    "Number of queries per retrieval strategy",
    labelnames=["strategy"],
)

GRAPH_TRAVERSAL_DEPTH_HISTOGRAM = Histogram(
    "graphrag_graph_traversal_depth",
    "Number of graph nodes traversed per query",
    buckets=[0, 1, 2, 5, 10, 20, 50, 100],
)

ANSWER_QUALITY_GAUGE = Gauge(
    "graphrag_answer_quality_score",
    "Latest RAGAS answer quality score (0-1)",
)

GRAPH_NODE_COUNT_GAUGE = Gauge(
    "graphrag_graph_node_count",
    "Total number of entity nodes in the knowledge graph",
)

GRAPH_EDGE_COUNT_GAUGE = Gauge(
    "graphrag_graph_edge_count",
    "Total number of relation edges in the knowledge graph",
)

COMMUNITY_COUNT_GAUGE = Gauge(
    "graphrag_community_count",
    "Total number of communities in the knowledge graph",
)

EXTRACTION_LATENCY_HISTOGRAM = Histogram(
    "graphrag_extraction_latency_seconds",
    "Entity extraction latency per chunk",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)
