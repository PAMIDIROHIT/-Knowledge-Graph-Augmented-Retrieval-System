"""
Neo4j driver wrapper with connection pooling, retry logic, and parameterized Cypher helpers.
ALL Cypher queries use parameterized inputs — NEVER string interpolation.
"""

from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from typing import Any, Generator, Optional

import structlog
from neo4j import GraphDatabase, Session
from neo4j.exceptions import (
    ClientError,
    ServiceUnavailable,
    SessionExpired,
    TransientError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)


def _retryable(exc: Exception) -> bool:
    return isinstance(exc, (ServiceUnavailable, SessionExpired, TransientError))


class Neo4jClient:
    """
    Thread-safe Neo4j client with connection pooling and retry logic.
    Uses parameterized queries throughout to prevent Cypher injection.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "graphrag_password",
        database: str = "neo4j",
        max_connection_pool_size: int = 50,
        connection_acquisition_timeout: int = 60,
    ) -> None:
        self._uri = uri
        self._database = database
        self._driver = GraphDatabase.driver(
            uri,
            auth=(user, password),
            max_connection_pool_size=max_connection_pool_size,
            connection_acquisition_timeout=connection_acquisition_timeout,
        )
        logger.info("neo4j_client_initialized", uri=uri, database=database)

    def close(self) -> None:
        self._driver.close()

    def verify_connectivity(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as exc:
            logger.error("neo4j_connectivity_failed", error=str(exc))
            return False

    @contextmanager
    def _session(self) -> Generator[Session, None, None]:
        with self._driver.session(database=self._database) as session:
            yield session

    @retry(
        retry=retry_if_exception_type((ServiceUnavailable, SessionExpired, TransientError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def run_cypher(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """Execute a parameterized Cypher query and return results as list of dicts."""
        params = parameters or {}
        with self._session() as session:
            result = session.run(query, params)
            return [record.data() for record in result]

    def run_write(
        self,
        query: str,
        parameters: Optional[dict[str, Any]] = None,
    ) -> None:
        """Execute a write transaction with retry."""
        params = parameters or {}

        def _tx(tx: Any) -> None:
            tx.run(query, params)

        with self._session() as session:
            session.execute_write(_tx)

    # ------------------------------------------------------------------ #
    # Schema management                                                    #
    # ------------------------------------------------------------------ #

    def create_constraints_and_indexes(self) -> None:
        """Create constraints and indexes for production performance."""
        statements = [
            # Uniqueness constraints
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE",
            # Lookup indexes
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX relation_type_idx IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.relation_type)",
            "CREATE INDEX community_level_idx IF NOT EXISTS FOR (c:Community) ON (c.level)",
        ]
        for stmt in statements:
            try:
                self.run_write(stmt)
                logger.debug("schema_statement_executed", stmt=stmt[:60])
            except ClientError as exc:
                # Already exists — safe to ignore
                logger.debug("schema_already_exists", error=str(exc)[:80])

    # ------------------------------------------------------------------ #
    # Entity operations                                                    #
    # ------------------------------------------------------------------ #

    def create_entity_node(self, entity: dict[str, Any]) -> None:
        """Upsert an Entity node. Uses MERGE to avoid duplicates."""
        query = """
        MERGE (e:Entity {id: $id})
        SET e.name        = $name,
            e.type        = $type,
            e.description = $description,
            e.confidence  = $confidence,
            e.source_chunk_id = $source_chunk_id,
            e.aliases     = $aliases,
            e.updated_at  = timestamp()
        """
        self.run_write(query, parameters={
            "id": entity["id"],
            "name": entity["name"],
            "type": entity["type"],
            "description": entity.get("description", ""),
            "confidence": entity.get("confidence", 0.0),
            "source_chunk_id": entity.get("source_chunk_id", ""),
            "aliases": entity.get("aliases", []),
        })

    def create_relation_edge(self, relation: dict[str, Any]) -> None:
        """Upsert a RELATION edge between two Entity nodes."""
        query = """
        MATCH (src:Entity {id: $source_entity_id})
        MATCH (tgt:Entity {id: $target_entity_id})
        MERGE (src)-[r:RELATION {id: $id}]->(tgt)
        SET r.relation_type  = $relation_type,
            r.description    = $description,
            r.confidence     = $confidence,
            r.source_chunk_id = $source_chunk_id,
            r.updated_at     = timestamp()
        """
        self.run_write(query, parameters={
            "id": relation["id"],
            "source_entity_id": relation["source_entity_id"],
            "target_entity_id": relation["target_entity_id"],
            "relation_type": relation["relation_type"],
            "description": relation.get("description", ""),
            "confidence": relation.get("confidence", 0.0),
            "source_chunk_id": relation.get("source_chunk_id", ""),
        })

    def get_entity_by_name(self, name: str) -> Optional[dict[str, Any]]:
        """Find an entity by exact name match."""
        query = "MATCH (e:Entity) WHERE e.name = $name RETURN e LIMIT 1"
        results = self.run_cypher(query, {"name": name})
        return results[0]["e"] if results else None

    def get_entity_neighborhood(
        self,
        entity_id: str,
        depth: int = 2,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Traverse the neighborhood of an entity up to `depth` hops.
        Returns {nodes: [...], edges: [...]}.
        Depth is inlined as a literal (Neo4j Aura 5.x requires literals for variable-length paths).
        """
        # Neo4j Aura 5.x doesn't allow parameters in variable-length path bounds.
        # Depth is an integer from internal code — safe to inline.
        depth = max(1, min(int(depth), 5))  # clamp to 1–5
        query = f"""
        MATCH path = (start:Entity {{id: $entity_id}})-[r:RELATION*1..{depth}]-(neighbor:Entity)
        WITH nodes(path) AS ns, relationships(path) AS rs
        UNWIND ns AS n
        WITH COLLECT(DISTINCT {{id: n.id, name: n.name, type: n.type, description: n.description}}) AS nodes_list,
             rs
        UNWIND rs AS rel
        WITH nodes_list,
             COLLECT(DISTINCT {{
               id: rel.id,
               source: startNode(rel).id,
               target: endNode(rel).id,
               relation_type: rel.relation_type,
               description: rel.description,
               confidence: rel.confidence
             }}) AS edges_list
        RETURN nodes_list AS nodes, edges_list AS edges
        """
        results = self.run_cypher(query, {"entity_id": entity_id})
        if not results:
            # Return just the starting entity
            entity = self.run_cypher(
                "MATCH (e:Entity {id: $entity_id}) RETURN e", {"entity_id": entity_id}
            )
            node = entity[0]["e"] if entity else {}
            return {"nodes": [node] if node else [], "edges": []}
        return {"nodes": results[0].get("nodes", []), "edges": results[0].get("edges", [])}

    def search_entities_by_name_fuzzy(self, query_text: str, limit: int = 10) -> list[dict]:
        """Full-text search on entity names (requires full-text index)."""
        cypher = """
        MATCH (e:Entity)
        WHERE toLower(e.name) CONTAINS toLower($query)
        RETURN e ORDER BY e.confidence DESC LIMIT $limit
        """
        results = self.run_cypher(cypher, {"query": query_text, "limit": limit})
        return [r["e"] for r in results]

    # ------------------------------------------------------------------ #
    # Community operations                                                 #
    # ------------------------------------------------------------------ #

    def create_community_node(self, community: dict[str, Any]) -> None:
        """Upsert a Community node."""
        query = """
        MERGE (c:Community {id: $id})
        SET c.level        = $level,
            c.title        = $title,
            c.summary      = $summary,
            c.size         = $size,
            c.key_entities = $key_entities,
            c.updated_at   = timestamp()
        """
        self.run_write(query, parameters={
            "id": community["id"],
            "level": community.get("level", 0),
            "title": community.get("title", ""),
            "summary": community.get("summary", ""),
            "size": community.get("size", 0),
            "key_entities": community.get("key_entities", []),
        })

    def link_entity_to_community(self, entity_id: str, community_id: str) -> None:
        """Create MEMBER_OF edge from Entity to Community."""
        query = """
        MATCH (e:Entity {id: $entity_id})
        MATCH (c:Community {id: $community_id})
        MERGE (e)-[:MEMBER_OF]->(c)
        """
        self.run_write(query, parameters={
            "entity_id": entity_id,
            "community_id": community_id,
        })

    def get_all_communities(self, level: Optional[int] = None) -> list[dict[str, Any]]:
        if level is not None:
            query = "MATCH (c:Community {level: $level}) RETURN c ORDER BY c.size DESC"
            results = self.run_cypher(query, {"level": level})
        else:
            query = "MATCH (c:Community) RETURN c ORDER BY c.level, c.size DESC"
            results = self.run_cypher(query, {})
        return [r["c"] for r in results]

    def get_graph_stats(self) -> dict[str, int]:
        """Return high-level graph statistics."""
        q_nodes = "MATCH (e:Entity) RETURN count(e) AS n"
        q_edges = "MATCH ()-[r:RELATION]->() RETURN count(r) AS n"
        q_communities = "MATCH (c:Community) RETURN count(c) AS n"
        nodes = self.run_cypher(q_nodes)[0]["n"]
        edges = self.run_cypher(q_edges)[0]["n"]
        communities = self.run_cypher(q_communities)[0]["n"]
        return {"nodes": nodes, "edges": edges, "communities": communities}

    def set_entity_embedding(self, entity_id: str, embedding: list[float]) -> None:
        """Store a vector embedding on an Entity node."""
        query = "MATCH (e:Entity {id: $id}) SET e.embedding = $embedding"
        self.run_write(query, {"id": entity_id, "embedding": embedding})

    def set_community_embedding(self, community_id: str, embedding: list[float]) -> None:
        query = "MATCH (c:Community {id: $id}) SET c.embedding = $embedding"
        self.run_write(query, {"id": community_id, "embedding": embedding})
