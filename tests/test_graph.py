"""
Tests for Neo4j client and NetworkX builder.
Uses mocking so tests can run without a live Neo4j instance.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, call

from src.graph.networkx_builder import NetworkXBuilder


# ─── NetworkXBuilder tests ────────────────────────────────────────────────────

class TestNetworkXBuilder:
    ENTITIES = [
        {"id": "ent_c1_001", "name": "Apple", "type": "ORGANIZATION", "description": "Tech", "confidence": 0.9, "source_chunk_id": "c1"},
        {"id": "ent_c1_002", "name": "Tim Cook", "type": "PERSON", "description": "CEO", "confidence": 0.95, "source_chunk_id": "c1"},
        {"id": "ent_c1_003", "name": "iPhone", "type": "PRODUCT", "description": "Phone", "confidence": 0.92, "source_chunk_id": "c1"},
    ]
    RELATIONS = [
        {"id": "rel_001", "source_entity_id": "ent_c1_001", "target_entity_id": "ent_c1_002", "relation_type": "HAS_CEO", "confidence": 0.9},
        {"id": "rel_002", "source_entity_id": "ent_c1_001", "target_entity_id": "ent_c1_003", "relation_type": "PRODUCES", "confidence": 0.88},
    ]

    def test_build_from_lists(self):
        builder = NetworkXBuilder()
        builder.build_from_lists(self.ENTITIES, self.RELATIONS)
        assert builder.graph.number_of_nodes() == 3
        assert builder.graph.number_of_edges() == 2

    def test_node_attributes_preserved(self):
        builder = NetworkXBuilder()
        builder.add_entity(self.ENTITIES[0])
        node = builder.graph.nodes["ent_c1_001"]
        assert node["name"] == "Apple"
        assert node["type"] == "ORGANIZATION"

    def test_dangling_edge_skipped(self):
        builder = NetworkXBuilder()
        builder.add_entity(self.ENTITIES[0])
        # Relation references a non-existent entity
        builder.add_relation({
            "id": "rel_bad",
            "source_entity_id": "ent_c1_001",
            "target_entity_id": "ent_NONEXISTENT_001",
            "relation_type": "ACQUIRED",
            "confidence": 0.9,
        })
        assert builder.graph.number_of_edges() == 0

    def test_to_undirected(self):
        import networkx as nx
        builder = NetworkXBuilder()
        builder.build_from_lists(self.ENTITIES, self.RELATIONS)
        undirected = builder.to_undirected()
        assert isinstance(undirected, nx.Graph)
        assert not undirected.is_directed()

    def test_stats(self):
        builder = NetworkXBuilder()
        builder.build_from_lists(self.ENTITIES, self.RELATIONS)
        stats = builder.get_stats()
        assert stats["nodes"] == 3
        assert stats["edges"] == 2
        assert "density" in stats

    def test_save_and_load(self, tmp_path):
        builder = NetworkXBuilder()
        builder.build_from_lists(self.ENTITIES, self.RELATIONS)
        save_path = tmp_path / "graph.pkl"
        builder.save(save_path)

        loaded = NetworkXBuilder.load(save_path)
        assert loaded.graph.number_of_nodes() == 3
        assert loaded.graph.number_of_edges() == 2

    def test_empty_graph_stats(self):
        builder = NetworkXBuilder()
        stats = builder.get_stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0


# ─── Neo4jClient tests (mocked) ───────────────────────────────────────────────

class TestNeo4jClientMocked:
    def _make_client(self):
        """Create a Neo4jClient with a mocked driver."""
        with patch("src.graph.neo4j_client.GraphDatabase") as mock_db:
            mock_driver = MagicMock()
            mock_db.driver.return_value = mock_driver
            from src.graph.neo4j_client import Neo4jClient
            client = Neo4jClient()
            client._driver = mock_driver
            return client, mock_driver

    def test_verify_connectivity_success(self):
        client, mock_driver = self._make_client()
        mock_driver.verify_connectivity.return_value = None
        assert client.verify_connectivity() is True

    def test_verify_connectivity_failure(self):
        client, mock_driver = self._make_client()
        mock_driver.verify_connectivity.side_effect = Exception("Connection refused")
        assert client.verify_connectivity() is False

    def test_cypher_query_uses_parameters(self):
        """Verify that run_cypher ALWAYS passes parameters — never string-formatted queries."""
        client, mock_driver = self._make_client()

        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        client.run_cypher("MATCH (e:Entity {id: $id}) RETURN e", {"id": "ent_test_001"})

        # The query must be called with a dict of parameters, never formatted
        call_args = mock_session.run.call_args
        assert call_args[0][1] == {"id": "ent_test_001"}, "Parameters must be passed as dict"

    def test_get_entity_neighborhood_uses_parameterized_query(self):
        """Entity neighborhood traversal must use $entity_id and $depth parameters."""
        client, mock_driver = self._make_client()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.run.return_value = mock_result
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        client.get_entity_neighborhood("ent_test_001", depth=2)

        # Use call_args_list[0] to check the FIRST (main neighbourhood) query,
        # not the fallback single-entity lookup that runs when the result is empty.
        first_call_args = mock_session.run.call_args_list[0]
        params = first_call_args[0][1]
        # Must use parameter dict with entity_id and depth — NEVER format them into the string
        assert "entity_id" in params, "entity_id must be a parameter, not interpolated"
        assert "depth" in params, "depth must be a parameter, not interpolated"
        # Verify the query string does NOT contain the literal entity ID (injection check)
        query_str = first_call_args[0][0]
        assert "ent_test_001" not in query_str, "Entity ID must NOT be interpolated into Cypher query"
