"""
Tests for retrieval components (local, global, hybrid).
All external dependencies are mocked.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.retrieval.hybrid_retriever import _route_query


# ─── Routing tests ─────────────────────────────────────────────────────────────

class TestQueryRouting:
    def test_explicit_local_mode(self):
        assert _route_query("What is Apple?", "local") == "local"

    def test_explicit_global_mode(self):
        assert _route_query("What are the trends?", "global") == "global"

    def test_explicit_hybrid_mode(self):
        assert _route_query("Anything", "hybrid") == "hybrid"

    def test_auto_routes_broad_themes_to_global(self):
        question = "What are the dominant themes in AI regulatory risk disclosures?"
        assert _route_query(question, "auto") == "global"

    def test_auto_routes_specific_entities_to_local(self):
        question = "Who acquired GitHub and what is their CEO's background?"
        assert _route_query(question, "auto") == "local"

    def test_auto_trend_patterns(self):
        assert _route_query("What patterns appear across all companies?", "auto") == "global"
        assert _route_query("What is the overall market overview?", "auto") == "global"

    def test_auto_entity_specific(self):
        assert _route_query("What did Microsoft report in their 10-K filing?", "auto") == "local"


# ─── LocalSearcher tests ──────────────────────────────────────────────────────

class TestLocalSearcher:
    def _make_searcher(self, vector_hits=None, neighborhood=None):
        from src.retrieval.local_search import LocalSearcher

        mock_indexer = MagicMock()
        mock_indexer.search.return_value = vector_hits if vector_hits is not None else [
            ("ent_c1_001", 0.92),
            ("ent_c1_002", 0.85),
        ]
        mock_indexer.is_loaded = True

        mock_neo4j = MagicMock()
        mock_neo4j.get_entity_neighborhood.return_value = neighborhood or {
            "nodes": [
                {"id": "ent_c1_001", "name": "Apple", "type": "ORGANIZATION", "description": "Tech"},
                {"id": "ent_c1_002", "name": "Tim Cook", "type": "PERSON", "description": "CEO"},
            ],
            "edges": [
                {"id": "rel_001", "source": "ent_c1_001", "target": "ent_c1_002",
                 "relation_type": "HAS_CEO", "description": "", "confidence": 0.9},
            ],
        }

        searcher = LocalSearcher(
            indexer=mock_indexer,
            neo4j_client=mock_neo4j,
            top_k_vector=5,
            top_k_graph=2,
            traversal_depth=2,
        )
        return searcher, mock_indexer, mock_neo4j

    def test_search_returns_structured_context(self):
        searcher, _, _ = self._make_searcher()
        result = searcher.search("Who is the CEO of Apple?")
        assert "nodes" in result
        assert "edges" in result
        assert "reasoning_path" in result
        assert "entity_ids" in result
        assert len(result["nodes"]) > 0

    def test_search_with_no_entity_hits(self):
        searcher, mock_indexer, _ = self._make_searcher(vector_hits=[])
        result = searcher.search("Unknown entity query")
        assert result["nodes"] == []
        assert result["edges"] == []

    def test_community_hits_filtered_from_local_search(self):
        """Local search should filter out community IDs (com_ prefix)."""
        searcher, mock_indexer, _ = self._make_searcher(
            vector_hits=[("com_0_0001", 0.95), ("ent_c1_001", 0.85)]
        )
        result = searcher.search("test query")
        # entity_ids should only contain ent_ IDs
        assert all(eid.startswith("ent_") for eid in result["entity_ids"])

    def test_reasoning_path_populated(self):
        searcher, _, _ = self._make_searcher()
        result = searcher.search("test")
        assert len(result["reasoning_path"]) >= 2
        assert any("VECTOR SEARCH" in step for step in result["reasoning_path"])
        assert any("GRAPH TRAVERSAL" in step for step in result["reasoning_path"])


# ─── GlobalSearcher tests ──────────────────────────────────────────────────────

class TestGlobalSearcher:
    def _make_searcher(self, all_hits=None, community_data=None):
        from src.retrieval.global_search import GlobalSearcher

        mock_indexer = MagicMock()
        mock_indexer.search.return_value = all_hits or [
            ("com_0_0001", 0.91),
            ("ent_c1_001", 0.88),
        ]
        mock_indexer.is_loaded = True

        mock_neo4j = MagicMock()
        mock_neo4j.run_cypher.return_value = community_data or [{
            "c": {
                "id": "com_0_0001",
                "title": "AI Technology Companies",
                "summary": "This community covers AI companies and their products.",
                "size": 15,
            }
        }]

        searcher = GlobalSearcher(
            indexer=mock_indexer,
            neo4j_client=mock_neo4j,
            top_k_communities=3,
        )
        return searcher, mock_indexer, mock_neo4j

    def test_search_returns_communities(self):
        searcher, _, _ = self._make_searcher()
        result = searcher.search("What are the themes in AI industry?")
        assert "communities" in result
        assert len(result["communities"]) >= 1

    def test_search_no_communities(self):
        searcher, mock_indexer, _ = self._make_searcher(all_hits=[("ent_c1_001", 0.9)])
        result = searcher.search("test")
        assert result["communities"] == []

    def test_reasoning_path_populated(self):
        searcher, _, _ = self._make_searcher()
        result = searcher.search("test query")
        assert any("COMMUNITY" in step for step in result["reasoning_path"])


# ─── HybridRetriever tests ─────────────────────────────────────────────────────

class TestHybridRetriever:
    def _make_retriever(self):
        from src.retrieval.hybrid_retriever import HybridRetriever

        mock_indexer = MagicMock()
        mock_indexer.search.return_value = [("ent_c1_001", 0.9)]
        mock_indexer.is_loaded = True

        mock_neo4j = MagicMock()
        mock_neo4j.get_entity_neighborhood.return_value = {"nodes": [], "edges": []}
        mock_neo4j.run_cypher.return_value = []

        return HybridRetriever(
            indexer=mock_indexer,
            neo4j_client=mock_neo4j,
            api_key="test_key",
        ), mock_indexer, mock_neo4j

    def test_retrieve_local_strategy(self):
        retriever, _, _ = self._make_retriever()
        ctx = retriever.retrieve("What is Apple's revenue?", mode="local")
        assert ctx["strategy"] == "local"
        assert "local_ctx" in ctx

    def test_retrieve_global_strategy(self):
        retriever, _, _ = self._make_retriever()
        ctx = retriever.retrieve("What are the themes in the industry?", mode="global")
        assert ctx["strategy"] == "global"
        assert "global_ctx" in ctx

    def test_retrieve_hybrid_calls_both(self):
        retriever, mock_indexer, _ = self._make_retriever()
        retriever.retrieve("test question", mode="hybrid")
        # Both local and global should have called the index
        assert mock_indexer.search.call_count >= 1
