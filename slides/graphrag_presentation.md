# Knowledge Graph-Augmented Retrieval System (GraphRAG)
### A Production-Grade End-to-End AI Pipeline

---

## Slide 1 — The Problem with Standard RAG

### What is Standard RAG?
- **Retrieve-then-Generate**: chunk documents → embed → store in vector DB → nearest-neighbor search → pass to LLM
- Works well for **fact lookup** ("What did the report say about revenue?")

### Where It Breaks Down
| Weakness | Example |
|---|---|
| No understanding of *relationships* | "How are OpenAI and Microsoft connected?" |
| Misses global/thematic patterns | "What are the dominant themes in these 500 docs?" |
| Flat retrieval — loses context hierarchy | Finds the sentence, loses the paragraph's meaning |
| No multi-hop reasoning | "Who invested in the company that GPT-4 creator works at?" |

> **Standard RAG is a keyword/semantic match engine — it doesn't *understand* the world.**

---

## Slide 2 — The GraphRAG Solution

### Core Idea
Build a **knowledge graph** from documents, then use it as a structured reasoning layer on top of vector search.

```
Documents → Entity/Relation Extraction → Knowledge Graph
                                              ↓
Query → FAISS Vector Search + Graph Traversal → LLM Answer
```

### What this enables
- **Multi-hop reasoning**: Follow chains of relationships across entities
- **Community-level summaries**: Detect clusters (companies, technologies, people) and summarize them
- **Dual retrieval modes**: Local (entity neighborhood) + Global (community themes)
- **Source attribution**: Every answer cites graph paths and relations

---

## Slide 3 — System Architecture (End-to-End)

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE (DVC)                          │
│                                                                   │
│  raw docs → [INGEST] → chunks.json                               │
│            → [EXTRACT] → entities.json + relations.json          │
│            → [BUILD GRAPH] → Neo4j + NetworkX + communities.json │
│            → [INDEX] → FAISS index (384-dim, sentence-transformer)│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI SERVER (port 8000)                   │
│                                                                   │
│  POST /query → HybridRetriever → LocalSearch + GlobalSearch      │
│                               → Grok LLM (xAI) → JSON answer    │
│  GET  /graph/explore → neighborhood visualization                │
│  GET  /health → system status                                    │
│  GET  /metrics → Prometheus metrics                              │
│  GET  /app/ → Web UI frontend                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Slide 4 — The 4-Stage DVC Pipeline

### Stage 1: Ingest
- **Input**: Raw `.txt`/`.pdf`/`.md` files in `data/raw/`
- **What it does**: Splits documents into overlapping chunks (512 tokens, 64 overlap)
- **Output**: `data/processed/chunks.json` with metadata (file, position, char count)
- **Tech**: Pure Python, regex-based chunking

### Stage 2: Entity & Relation Extraction
- **Input**: `chunks.json`
- **What it does**: Sends each chunk to Grok-3 with a structured prompt → extracts entities (names, types, descriptions) and relations (triples: source → relation → target)
- **Output**: `entities.json` + `relations.json`
- **Tech**: Grok-3 (xAI), async OpenAI-compatible SDK, structured JSON output

### Stage 3: Graph Construction
- **Input**: `entities.json` + `relations.json`
- **What it does**: Builds Neo4j property graph + NetworkX mirror → runs **Leiden community detection** → generates LLM summaries per community
- **Output**: Neo4j database + `artifacts/communities.json` + `artifacts/networkx_graph.pkl`
- **Tech**: Neo4j, graspologic (Leiden), NetworkX, Grok-3

### Stage 4: Indexing
- **Input**: `entities.json` + `communities.json`
- **What it does**: Embeds all entities and community summaries → builds FAISS flat inner-product index
- **Output**: `artifacts/faiss_index.bin` + `artifacts/faiss_id_map.json`
- **Tech**: sentence-transformers (`all-MiniLM-L6-v2`, 384-dim), FAISS

---

## Slide 5 — Hybrid Retrieval: Local + Global Search

### Local Search (entity-anchored)
1. Embed query → FAISS top-K nearest entities
2. For each entity: traverse Neo4j neighborhood up to depth-2 (BFS)
3. Collect subgraph (nodes + edges), deduplicated
4. Feed subgraph context to LLM

**Best for**: Specific entity questions ("Tell me about Sam Altman", "What does OpenAI produce?")

### Global Search (community-anchored)
1. Embed query → FAISS top-K nearest **communities**
2. Fetch community summaries (pre-generated LLM summaries of clusters)
3. Supplement with entity hits from same search
4. Feed community + entity context to LLM

**Best for**: Thematic questions ("What are the key AI safety companies?", "Summarize competition in LLMs")

### Auto Routing
Query classifier routes to `local`, `global`, or `hybrid` based on:
- Keywords: "who/what specific..." → local
- Keywords: "all/every/theme/summary..." → global  
- Mixed → hybrid (both paths, weighted combination)

---

## Slide 6 — Key Technologies

| Layer | Technology | Why This Choice |
|---|---|---|
| **LLM (extraction + QA)** | Grok-3 (xAI) | OpenAI-compatible API, long context (131K), fast reasoning |
| **Embeddings** | sentence-transformers `all-MiniLM-L6-v2` | Free, local, 384-dim, no API cost |
| **Vector Store** | FAISS (IndexFlatIP) | Sub-millisecond search at 1M+ vectors, no server needed |
| **Graph DB** | Neo4j (optional) | Cypher queries, native graph traversal, depth-N neighborhood |
| **Graph Library** | NetworkX + graspologic | Leiden community detection, offline fallback |
| **API Framework** | FastAPI + Pydantic | Async, OpenAPI schema, streaming SSE |
| **Pipeline** | DVC | Reproducible ML pipelines, data versioning |
| **Observability** | Prometheus + structlog | Latency histograms, error rates, structured logs |
| **Testing** | pytest + pytest-asyncio | 61 tests, mocked LLM/DB dependencies |

---

## Slide 7 — GraphRAG vs Standard RAG

```
Query: "What's the relationship between OpenAI and Microsoft?"

Standard RAG:
  → Finds chunk: "Microsoft invested $10B in OpenAI"
  → Answer: "Microsoft invested in OpenAI"
  ✗ Misses: Sam Altman's board crisis, Azure hosting, GPT-4 exclusivity

GraphRAG:
  → FAISS: finds entity ent_openai, ent_microsoft
  → Graph traversal: OpenAI -INVESTED_BY-> Microsoft
                     OpenAI -HOSTED_ON-> Azure (Microsoft)
                     Sam Altman -IS_CEO_OF-> OpenAI
                     OpenAI -PRODUCES-> GPT-4
  → Community: [AI Companies cluster] with theme summary
  → Answer: multi-hop, relationship-aware, with citation graph
  ✓ Richer, grounded, traceable
```

### Performance vs Standard RAG
| Metric | Standard RAG | GraphRAG |
|---|---|---|
| Multi-hop questions | ❌ Poor | ✅ Strong |
| Entity disambiguation | ❌ Fails | ✅ By graph identity |
| Thematic questions | ❌ Inconsistent | ✅ Via community summaries |
| Latency per query | ~500ms | ~800-1500ms |
| Reasoning transparency | ❌ Black box | ✅ Full path shown |

---

## Slide 8 — Knowledge Graph Deep Dive

### What's in the Graph?
- **Entities**: Named things (people, organizations, products, technologies, locations)
- **Relations**: Typed edges with descriptions (IS_CEO_OF, PRODUCES, INVESTED_IN, COMPETES_WITH, RELATED_TO...)
- **Communities**: Auto-detected clusters of densely interconnected entities

### Entity Schema
```json
{
  "id": "ent_ai_companies_0001_002",
  "name": "Sam Altman",
  "type": "PERSON",
  "description": "CEO of OpenAI, leading AI safety and commercial deployment",
  "confidence": 0.95,
  "source_chunk_id": "chunk_001"
}
```

### Leiden Community Detection
- Uses **Leiden algorithm** (improvement over Louvain) via graspologic
- Detects natural clusters in the entity graph
- Each community gets an LLM-generated summary: title, key entities, thematic description
- **Result for our corpus**: 9 communities covering AI companies, researchers, products, and technologies

---

## Slide 9 — API Design & Frontend

### REST Endpoints
```
POST /query
  Body: {question, search_mode: "local"|"global"|"auto", top_k, graph_depth}
  Response: {answer, reasoning_path[], citations[], graph_nodes_traversed[], latency_ms}

GET /graph/explore?entity_name=OpenAI&depth=2
  Response: {entities[], relations[], community_id}

GET /health → {status, neo4j_connected, index_loaded, ...}
GET /metrics → Prometheus text format
```

### Streaming support
- `POST /query` with `stream: true` → Server-Sent Events (SSE)
- Token-by-token streaming from Grok LLM

### Web Frontend
- Located at `http://localhost:8000/app/index.html`
- Chat interface with reasoning path visualization
- Shows graph nodes traversed, citations per answer

---

## Slide 10 — Observability & Production Readiness

### Metrics (Prometheus)
- `query_latency_seconds` — histogram by retrieval strategy
- `retrieval_strategy_total` — counter (local/global/hybrid)
- `graph_traversal_depth` — histogram

### Structured Logging (structlog)
```json
{"event": "local_search_complete", "nodes": 15, "edges": 8, "chunks": 5, "query_len": 34}
{"event": "qa_generation_failed", "error": "..."}
{"event": "faiss_index_loaded", "vectors": 63}
```

### Resilience
- **FAISS without Neo4j**: Server starts, serves FAISS-only results ✅
- **No LLM credits**: Returns retrieval context with graceful error ✅
- **Neo4j unavailable**: Falls back to in-memory graph from JSON files ✅
- **Missing index**: Graceful startup warning, FAISS re-builds on next pipeline run ✅

### Test Coverage
- **61 tests** across 4 suites: API, extraction, graph, retrieval
- Fully mocked: no real API calls in tests
- pytest-asyncio for async extraction tests

---

## Slide 11 — Skills Demonstrated

### ML / AI Engineering
- **LLM Integration**: Async/sync client patterns, structured output parsing, error handling
- **Graph ML**: Entity extraction, Leiden community detection, multi-hop traversal
- **Embedding Pipeline**: sentence-transformers, FAISS index build/search, dimension management
- **RAG Architecture**: Hybrid retrieval design, evidence aggregation, QA prompt engineering

### Software Engineering
- **Clean Architecture**: Separation of extraction / graph / retrieval / API layers
- **Production patterns**: Async FastAPI, Pydantic schemas, Prometheus metrics, structured logs
- **Testing discipline**: 61 tests, mocked dependencies, asyncio test patterns
- **Data pipelines**: DVC for reproducible ML pipelines, artifact tracking

### DevOps / Infrastructure
- **Optional dependencies**: Neo4j graceful degradation, API fallbacks
- **Configuration management**: `params.yaml` for all hyperparameters
- **CI-friendly**: Fully containerizable, secrets via `.env`, no hardcoded credentials

---

## Slide 12 — Interview Q&A Guide

**Q: Why not just use a bigger vector store / more chunks?**
> Chunking splits context arbitrarily — a relation spanning two chunks is lost. Graph extraction captures *semantic relationships* explicitly, independent of chunk boundaries.

**Q: How does Leiden compare to Louvain community detection?**
> Leiden guarantees all communities are internally connected (Louvain doesn't), has better resolution control, and converges faster on large graphs.

**Q: Why sentence-transformers instead of OpenAI text-embedding-3-small?**
> Zero API cost, runs locally, no latency overhead, 384-dim vs 1536-dim means 4x smaller FAISS index. For domain-adapted embeddings, fine-tuning sentence-transformers is also straightforward.

**Q: What's the query routing logic?**
> Keywords like "who/what/where" → local (entity-specific). Keywords like "all/every/compare/themes/summarize" → global (community-level). Mixed → hybrid with weighted score combination.

**Q: How does streaming work?**
> The `/query` endpoint supports SSE. The `answer_streaming()` method uses `openai.stream()` context manager and `yield`s tokens. The FastAPI `StreamingResponse` wraps the generator.

**Q: What's DVC doing here?**
> DVC tracks pipeline stage dependencies (which files are inputs/outputs of each stage), caches intermediate artifacts to avoid re-running expensive LLM extraction, and enables `dvc repro` for full reproducibility.

---

## Summary Slide

### What I Built
A **production-ready GraphRAG system** that:
1. **Ingests** any text documents
2. **Extracts** a knowledge graph via LLM (54 entities, 37 relations, 9 communities)
3. **Retrieves** answers using dual local/global FAISS + graph traversal
4. **Generates** grounded, cited answers via Grok-3
5. **Serves** via FastAPI with streaming, metrics, and a web UI
6. **Tests** with 61 unit tests, all passing

### Tech Stack
`Grok-3 • sentence-transformers • FAISS • Neo4j • NetworkX • Leiden • FastAPI • DVC • Pydantic • Prometheus • Python 3.13`

### Key Differentiator vs Standard RAG
> GraphRAG reasons over *relationships* between entities — not just similarity of words. This enables multi-hop inference, community-level synthesis, and explainable reasoning paths.
