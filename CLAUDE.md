# GraphRAG Project — Claude Memory File

## Project Purpose
Knowledge Graph-Augmented Retrieval system answering multi-hop questions
by combining vector similarity with Neo4j graph traversal.

## Architecture (quick reference)
- Ingestion → Entity Extraction (claude-sonnet-4-6) → Neo4j + NetworkX
- Community Detection (Leiden) → Community Summaries
- Hybrid Retrieval: FAISS vector search + Cypher graph traversal
- FastAPI REST API with streaming responses and provenance citations
- MLflow tracking, DVC pipeline, GitHub Actions CI, Prometheus/Grafana

## Key Commands
```bash
# Activate environment
source .venv/bin/activate

# Run full pipeline
dvc repro

# Run tests
pytest tests/ -v

# Start services
docker-compose up -d

# MLflow UI
mlflow ui --port 5001

# Neo4j browser
open http://localhost:7474

# API server (dev)
uvicorn src.api.main:app --reload --port 8000
```

## Conventions
- All entity IDs follow: ent_{chunk_id}_{sequential_number}
- All relation IDs follow: rel_{source_id}_{target_id}_{type}
- Cypher queries use parameterized inputs — NEVER string interpolation
- All API responses include a citations list with source chunk IDs
- Progress tracked in progress.md; test status in tests_status.json
- All modules use structlog for structured logging
- Pydantic v2 models throughout; use model_validate not parse_obj

## Service Ports
- Neo4j bolt: 7687  | Neo4j browser: 7474
- FastAPI:    8000
- MLflow:     5001
- Prometheus: 9090
- Grafana:    3000

## Critical Constraints
- NEVER lower multi_hop_accuracy CI threshold below 0.70
- NEVER use string interpolation in Cypher queries (injection risk)
- NEVER delete Neo4j data without explicit user confirmation
- ALWAYS parameterize extraction model via params.yaml, not hardcoded
- ALWAYS use parameterized Cypher queries to prevent injection attacks

## Entity ID Schema
- Entity IDs:   ent_{chunk_id}_{sequential_number}   e.g. ent_chunk_042_001
- Relation IDs: rel_{source_ent_id}_{target_ent_id}_{type}
- Community IDs: com_{level}_{community_number}

## Current Phase
Phase 1 complete — Foundation & Schema

## Known Issues
[Claude: log blockers here as they are discovered]
