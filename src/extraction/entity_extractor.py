"""
LLM-based entity and relation extractor using Grok (xAI) via OpenAI-compatible API.
Uses async batch processing for extraction reliability.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Optional

import openai
import structlog

from src.extraction.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
)
from src.extraction.schema import (
    DocumentChunk,
    Entity,
    EntityType,
    ExtractionResult,
    Relation,
)

logger = structlog.get_logger(__name__)


def _make_entity_id(chunk_id: str, seq: int) -> str:
    return f"ent_{chunk_id}_{seq:03d}"


def _make_relation_id(source_id: str, target_id: str, rel_type: str) -> str:
    return f"rel_{source_id}_{target_id}_{rel_type}"


def _sanitize_entity_type(raw: str) -> EntityType:
    try:
        return EntityType(raw.upper())
    except ValueError:
        return EntityType.OTHER


def _parse_extraction_response(
    raw_json: str,
    chunk_id: str,
    source_document: str,
    confidence_threshold: float,
) -> tuple[list[Entity], list[Relation]]:
    """Parse LLM JSON output into validated Entity and Relation objects."""
    # Strip markdown fences if the model adds them
    raw_json = re.sub(r"```(?:json)?", "", raw_json).strip()

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        logger.warning("json_parse_error", chunk_id=chunk_id, error=str(exc))
        return [], []

    raw_entities: list[dict] = data.get("entities", [])
    raw_relations: list[dict] = data.get("relations", [])

    entities: list[Entity] = []
    id_map: dict[str, str] = {}  # Maps LLM-generated ID → validated ID

    for seq, raw in enumerate(raw_entities, start=1):
        try:
            conf = float(raw.get("confidence", 0.0))
            if conf < confidence_threshold:
                logger.debug(
                    "entity_below_threshold",
                    name=raw.get("name"),
                    confidence=conf,
                    threshold=confidence_threshold,
                )
                continue

            canonical_id = _make_entity_id(chunk_id, seq)
            llm_id = raw.get("id", canonical_id)
            id_map[llm_id] = canonical_id

            entity = Entity(
                id=canonical_id,
                name=str(raw.get("name", "")).strip(),
                type=_sanitize_entity_type(str(raw.get("type", "OTHER"))),
                description=str(raw.get("description", "")),
                source_chunk_id=chunk_id,
                confidence=conf,
                aliases=raw.get("aliases", []),
                attributes=raw.get("attributes", {}),
            )
            entities.append(entity)
        except Exception as exc:  # noqa: BLE001
            logger.warning("entity_parse_error", chunk_id=chunk_id, error=str(exc), raw=raw)

    entity_ids = {e.id for e in entities}
    relations: list[Relation] = []

    for raw in raw_relations:
        try:
            conf = float(raw.get("confidence", 0.0))
            if conf < confidence_threshold:
                continue

            # Remap LLM IDs to canonical IDs
            src_llm = raw.get("source_entity_id", "")
            tgt_llm = raw.get("target_entity_id", "")
            src_id = id_map.get(src_llm, src_llm)
            tgt_id = id_map.get(tgt_llm, tgt_llm)

            if src_id not in entity_ids or tgt_id not in entity_ids:
                logger.debug(
                    "relation_dangling_reference",
                    src=src_id,
                    tgt=tgt_id,
                    chunk_id=chunk_id,
                )
                continue

            rel_type = str(raw.get("relation_type", "RELATED_TO")).upper()
            rel_id = _make_relation_id(src_id, tgt_id, rel_type)

            relation = Relation(
                id=rel_id,
                source_entity_id=src_id,
                target_entity_id=tgt_id,
                relation_type=rel_type,
                description=str(raw.get("description", "")),
                source_chunk_id=chunk_id,
                confidence=conf,
                attributes=raw.get("attributes", {}),
            )
            relations.append(relation)
        except Exception as exc:  # noqa: BLE001
            logger.warning("relation_parse_error", chunk_id=chunk_id, error=str(exc))

    return entities, relations


class EntityExtractor:
    """
    Async entity and relation extractor using Grok-3 via xAI API.
    Supports configurable concurrency.
    """

    def __init__(
        self,
        model: str = "grok-3",
        max_tokens: int = 16384,
        effort_level: str = "medium",
        confidence_threshold: float = 0.70,
        max_concurrency: int = 4,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort_level = effort_level
        self.confidence_threshold = confidence_threshold
        self.max_concurrency = max_concurrency
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = openai.AsyncOpenAI(
            api_key=api_key or os.environ.get("GROK_API_KEY"),
            base_url="https://api.x.ai/v1",
        )

    async def extract_chunk(self, chunk: DocumentChunk) -> ExtractionResult:
        """Extract entities and relations from a single document chunk."""
        prompt = EXTRACTION_USER_TEMPLATE.format(
            chunk_id=chunk.id,
            source_document=chunk.source_document,
            text=chunk.text,
        )

        start = time.monotonic()
        async with self._semaphore:
            try:
                response = await self._client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
            except openai.APIStatusError as exc:
                logger.error(
                    "grok_api_error",
                    chunk_id=chunk.id,
                    status=exc.status_code,
                    message=str(exc),
                )
                return ExtractionResult(
                    chunk_id=chunk.id,
                    model_used=self.model,
                    extraction_latency_ms=0.0,
                )
            except openai.APIConnectionError as exc:
                logger.error("grok_connection_error", chunk_id=chunk.id, error=str(exc))
                return ExtractionResult(
                    chunk_id=chunk.id,
                    model_used=self.model,
                    extraction_latency_ms=0.0,
                )

        latency_ms = (time.monotonic() - start) * 1000

        raw_text = response.choices[0].message.content or ""

        entities, relations = _parse_extraction_response(
            raw_json=raw_text,
            chunk_id=chunk.id,
            source_document=chunk.source_document,
            confidence_threshold=self.confidence_threshold,
        )

        logger.info(
            "chunk_extracted",
            chunk_id=chunk.id,
            entities=len(entities),
            relations=len(relations),
            latency_ms=round(latency_ms, 1),
        )

        return ExtractionResult(
            chunk_id=chunk.id,
            entities=entities,
            relations=relations,
            extraction_latency_ms=latency_ms,
            model_used=self.model,
        )

    async def extract_batch(
        self, chunks: list[DocumentChunk]
    ) -> list[ExtractionResult]:
        """Extract entities and relations from a batch of chunks concurrently."""
        tasks = [self.extract_chunk(chunk) for chunk in chunks]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return list(results)

    async def extract_all(
        self,
        chunks: list[DocumentChunk],
        batch_size: int = 8,
    ) -> list[ExtractionResult]:
        """Process all chunks in batches, respecting rate limits."""
        all_results: list[ExtractionResult] = []
        total = len(chunks)

        for batch_start in range(0, total, batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            logger.info(
                "processing_batch",
                batch_start=batch_start,
                batch_end=min(batch_start + batch_size, total),
                total=total,
            )
            batch_results = await self.extract_batch(batch)
            all_results.extend(batch_results)

        return all_results
