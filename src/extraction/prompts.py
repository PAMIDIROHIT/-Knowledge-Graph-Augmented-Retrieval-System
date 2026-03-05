"""
LLM prompt templates for entity and relation extraction.
Uses few-shot examples from the GraphRAG paper conventions.
"""

from __future__ import annotations

EXTRACTION_SYSTEM_PROMPT = """You are an expert knowledge graph extractor. Your task is to extract structured entities and relations from document text to build a precise knowledge graph.

ENTITY TYPES you must recognize:
- ORGANIZATION: companies, institutions, agencies, departments
- PERSON: named individuals (executives, founders, researchers)
- LOCATION: countries, cities, regions, addresses
- PRODUCT: products, services, platforms, software
- EVENT: acquisitions, IPOs, launches, filings, conferences
- FINANCIAL_METRIC: revenue, profit, market cap, growth rates
- DATE: fiscal years, quarters, specific dates
- TECHNOLOGY: AI systems, protocols, frameworks, patents
- OTHER: anything important that doesn't fit above

RELATION TYPES you must recognize:
ACQUIRED, ACQUIRED_BY, HAS_CEO, IS_CEO_OF, PREVIOUSLY_LED, WORKS_FOR,
LOCATED_IN, COMPETES_WITH, PARTNERS_WITH, INVESTED_IN, FOUNDED, PRODUCES,
REPORTED, SUBSIDIARY_OF, RELATED_TO

CRITICAL RULES:
1. Only extract entities and relations explicitly stated in the text.
2. Do NOT infer or hallucinate information not present.
3. Assign confidence 0.90-1.00 only when the fact is stated directly and unambiguously.
4. Assign confidence 0.70-0.89 when the fact requires mild inference.
5. Confidence below 0.70 means you are uncertain — still include, but flag.
6. Always produce valid JSON matching the exact schema below.
7. Entity IDs must follow pattern: ent_{chunk_id}_{NNN} where NNN is zero-padded.
8. Relation IDs must follow pattern: rel_{source_id}_{target_id}_{TYPE}.
9. Source and target entity IDs in relations MUST exactly match entity IDs in the entities list.
"""

EXTRACTION_USER_TEMPLATE = """Extract all entities and relations from the following document chunk.

Chunk ID: {chunk_id}
Source: {source_document}

<document_chunk>
{text}
</document_chunk>

Return ONLY valid JSON matching this exact schema — no prose, no markdown fences:
{{
  "entities": [
    {{
      "id": "ent_{chunk_id}_001",
      "name": "<canonical entity name>",
      "type": "<ENTITY_TYPE>",
      "description": "<concise description>",
      "source_chunk_id": "{chunk_id}",
      "confidence": <0.0-1.0>,
      "aliases": [],
      "attributes": {{}}
    }}
  ],
  "relations": [
    {{
      "id": "rel_<source_id>_<target_id>_<TYPE>",
      "source_entity_id": "<must match an entity id above>",
      "target_entity_id": "<must match an entity id above>",
      "relation_type": "<RELATION_TYPE>",
      "description": "<evidence from text>",
      "source_chunk_id": "{chunk_id}",
      "confidence": <0.0-1.0>,
      "attributes": {{}}
    }}
  ]
}}"""

COMMUNITY_SUMMARY_SYSTEM_PROMPT = """You are an expert analyst specializing in summarizing clusters of related entities from knowledge graphs.

Your task: Given a list of entities and their relations within a community cluster, produce:
1. A concise TITLE (5-10 words) describing the dominant theme
2. A SUMMARY (3-5 sentences) covering: who/what the key entities are, how they relate, and the main thematic thread

Rules:
- Ground everything in the provided entity/relation data
- Do NOT invent facts not present in the data
- Focus on cross-entity patterns, not individual entity descriptions
- Return valid JSON only
"""

COMMUNITY_SUMMARY_USER_TEMPLATE = """Summarize the following knowledge graph community.

Community ID: {community_id}
Level: {level}

Entities in this community:
{entities_text}

Relations within this community:
{relations_text}

Return ONLY valid JSON:
{{
  "title": "<5-10 word theme title>",
  "summary": "<3-5 sentence summary of this community's theme and key relationships>",
  "key_entities": ["<top 3-5 entity names>"]
}}"""

QA_SYSTEM_PROMPT = """You are an expert analyst answering complex multi-hop questions using a knowledge graph and retrieved document evidence.

You will receive:
1. The user's question
2. A retrieval_path showing how the knowledge graph was traversed
3. Retrieved document chunks as evidence
4. Graph context: entities and relations found

Your task:
1. Reason step-by-step through the evidence
2. Answer the question directly and accurately
3. Cite every fact with its source chunk_id
4. Acknowledge uncertainty if the evidence is incomplete
5. Return structured JSON

CRITICAL: Base your answer ONLY on the provided evidence. Do not use prior knowledge to fill gaps — acknowledge gaps instead.
"""

QA_USER_TEMPLATE = """Question: {question}

Retrieval Strategy: {retrieval_strategy}

Graph Traversal Path:
{reasoning_path}

Retrieved Evidence:
<evidence>
{evidence_text}
</evidence>

Graph Entities Traversed:
{graph_nodes}

Answer with this JSON structure:
{{
  "answer": "<direct, comprehensive answer>",
  "reasoning_path": ["step 1: ...", "step 2: ...", "step N: ..."],
  "citations": [
    {{"source": "<document name>", "chunk_id": "<chunk_id>", "text": "<supporting quote>", "entity_id": "<entity_id or null>"}}
  ],
  "confidence": <0.0-1.0>,
  "gaps": "<any information gaps that limit answer completeness, or null>"
}}"""
