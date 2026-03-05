"""
FastAPI application entry point.
Manages lifecycle: startup (load index, connect Neo4j), shutdown (close connections).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.routes import router
from src.graph.indexer import GraphIndexer
from src.graph.neo4j_client import Neo4jClient
from src.retrieval.hybrid_retriever import HybridRetriever

logger = structlog.get_logger(__name__)


def _load_params():
    try:
        import yaml
        with open("params.yaml") as f:
            return yaml.safe_load(f)
    except Exception:
        return {}


def _load_env():
    """Load .env file if present (python-dotenv style, without requiring the package)."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: initialize and tear down resources."""
    # In test mode, state is pre-populated by the test fixture — skip initialization.
    if os.environ.get("TESTING"):
        logger.info("graphrag_api_started")
        yield
        logger.info("graphrag_api_shutdown")
        return

    _load_env()
    params = _load_params()
    graph_cfg = params.get("graph", {})
    ret_cfg = params.get("retrieval", {})
    qa_cfg = params.get("qa_model", {})

    # Connect to Neo4j
    neo4j = Neo4jClient(
        uri=os.environ.get("NEO4J_URI", graph_cfg.get("neo4j_uri", "bolt://localhost:7687")),
        user=os.environ.get("NEO4J_USERNAME", os.environ.get("NEO4J_USER", graph_cfg.get("neo4j_user", "neo4j"))),
        password=os.environ.get("NEO4J_PASSWORD", graph_cfg.get("neo4j_password", "graphrag_password")),
        database=os.environ.get("NEO4J_DATABASE", graph_cfg.get("neo4j_database", "neo4j")),
        max_connection_pool_size=graph_cfg.get("max_connection_pool_size", 50),
    )

    # Load FAISS index
    index_path = ret_cfg.get("faiss_index_path", "artifacts/faiss_index.bin")
    id_map_path = ret_cfg.get("faiss_id_map_path", "artifacts/faiss_id_map.json")
    indexer = GraphIndexer(
        index_path=index_path,
        id_map_path=id_map_path,
        embedding_model=ret_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
        embedding_dim=int(ret_cfg.get("embedding_dim", 384)),
    )

    if Path(index_path).exists():
        try:
            indexer.load()
            logger.info("faiss_index_loaded_at_startup")
        except Exception as exc:
            logger.warning("faiss_index_load_failed", error=str(exc))
    else:
        logger.warning("faiss_index_not_found", path=index_path)

    groq_api_key = os.environ.get("GROQ_API_KEY")

    # Build retriever
    retriever = HybridRetriever(
        indexer=indexer,
        neo4j_client=neo4j,
        top_k_vector=int(ret_cfg.get("top_k_vector", 10)),
        top_k_graph=int(ret_cfg.get("top_k_graph", 5)),
        traversal_depth=int(ret_cfg.get("graph_traversal_depth", 2)),
        vector_weight=float(ret_cfg.get("hybrid_vector_weight", 0.4)),
        graph_weight=float(ret_cfg.get("hybrid_graph_weight", 0.6)),
        qa_model=qa_cfg.get("model", "llama-3.3-70b-versatile"),
        qa_max_tokens=int(qa_cfg.get("max_tokens", 8192)),
        api_key=groq_api_key,
    )

    app.state.neo4j = neo4j
    app.state.indexer = indexer
    app.state.retriever = retriever

    logger.info("graphrag_api_started")
    yield

    # Teardown
    neo4j.close()
    logger.info("graphrag_api_shutdown")


app = FastAPI(
    title="GraphRAG API",
    description="Production Knowledge Graph-Augmented Retrieval system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend (open frontend/index.html) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)

# Serve static files (D3.js explorer)
static_dir = Path("src/api/static")
if static_dir.exists():
    app.mount("/explorer", StaticFiles(directory=str(static_dir), html=True), name="static")

# Serve the main frontend app at /app
frontend_dir = Path("frontend")
if frontend_dir.exists():
    app.mount("/app", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

# Prometheus metrics instrumentation
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
