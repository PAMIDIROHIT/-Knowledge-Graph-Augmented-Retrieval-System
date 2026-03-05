"""
Tests for the FastAPI endpoints.
Uses TestClient with dependency overrides and mocked state.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient


def _make_mock_retriever(answer="Test answer.", strategy="local"):
    from src.extraction.schema import QueryResponse, Citation
    mock = MagicMock()
    mock.answer.return_value = QueryResponse(
        answer=answer,
        reasoning_path=["step 1", "step 2"],
        citations=[Citation(source="test.txt", chunk_id="chunk_001", text="supporting text")],
        graph_nodes_traversed=["ent_001", "ent_002"],
        retrieval_strategy=strategy,
        latency_ms=123.4,
    )
    mock.local = MagicMock()
    mock.local.indexer = MagicMock()
    mock.local.indexer.is_loaded = True
    return mock


def _make_mock_neo4j(node_count=100, edge_count=200):
    mock = MagicMock()
    mock.verify_connectivity.return_value = True
    mock.get_graph_stats.return_value = {"nodes": node_count, "edges": edge_count, "communities": 15}
    mock.get_entity_by_name.return_value = {"id": "ent_test_001", "name": "Microsoft", "type": "ORGANIZATION", "description": "Tech company"}
    mock.get_entity_neighborhood.return_value = {
        "nodes": [
            {"id": "ent_test_001", "name": "Microsoft", "type": "ORGANIZATION", "description": ""},
            {"id": "ent_test_002", "name": "Satya Nadella", "type": "PERSON", "description": "CEO"},
        ],
        "edges": [
            {"id": "rel_001", "source": "ent_test_001", "target": "ent_test_002",
             "relation_type": "HAS_CEO", "description": "", "confidence": 0.95},
        ],
    }
    return mock


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TESTING", "1")
    from src.api.main import app
    mock_retriever = _make_mock_retriever()
    mock_neo4j = _make_mock_neo4j()

    app.state.retriever = mock_retriever
    app.state.neo4j = mock_neo4j

    with TestClient(app) as c:
        yield c


class TestQueryEndpoint:
    def test_query_returns_answer(self, client):
        response = client.post("/query", json={"question": "What is Microsoft?", "search_mode": "auto"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "Test answer."
        assert "citations" in data
        assert "retrieval_strategy" in data

    def test_query_validation_short_question(self, client):
        response = client.post("/query", json={"question": "Hi"})
        assert response.status_code == 422

    def test_query_invalid_search_mode(self, client):
        response = client.post("/query", json={"question": "Test question", "search_mode": "invalid"})
        assert response.status_code == 422

    def test_query_valid_modes(self, client):
        for mode in ["auto", "local", "global", "hybrid"]:
            response = client.post("/query", json={"question": "What is Apple?", "search_mode": mode})
            assert response.status_code == 200


class TestGraphExploreEndpoint:
    def test_explore_returns_nodes_edges(self, client):
        response = client.get("/graph/explore?entity_name=Microsoft&depth=2")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) >= 1

    def test_explore_entity_not_found(self, client):
        from src.api.main import app
        mock_neo4j = app.state.neo4j
        mock_neo4j.get_entity_by_name.return_value = None
        mock_neo4j.search_entities_by_name_fuzzy.return_value = []

        response = client.get("/graph/explore?entity_name=NONEXISTENT_ENTITY_XYZ&depth=2")
        assert response.status_code == 404

    def test_explore_invalid_depth(self, client):
        response = client.get("/graph/explore?entity_name=Microsoft&depth=10")
        assert response.status_code == 422


class TestHealthEndpoint:
    def test_health_returns_status(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "neo4j_connected" in data
        assert "index_loaded" in data
        assert "graph_node_count" in data

    def test_health_when_neo4j_down(self, client):
        from src.api.main import app
        app.state.neo4j.verify_connectivity.return_value = False
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"
        assert data["neo4j_connected"] is False
