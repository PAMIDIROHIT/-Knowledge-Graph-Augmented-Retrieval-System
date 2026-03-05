"""
Tests for entity extraction schema and extractor.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.extraction.schema import (
    Community,
    DocumentChunk,
    Entity,
    EntityType,
    ExtractionResult,
    Relation,
)
from src.extraction.entity_extractor import (
    EntityExtractor,
    _make_entity_id,
    _make_relation_id,
    _parse_extraction_response,
    _sanitize_entity_type,
)


# ─── Schema tests ──────────────────────────────────────────────────────────────

class TestEntitySchema:
    def test_valid_entity(self):
        e = Entity(
            id="ent_chunk_001_001",
            name="Microsoft",
            type=EntityType.ORGANIZATION,
            description="Technology company",
            source_chunk_id="chunk_001",
            confidence=0.97,
        )
        assert e.name == "Microsoft"
        assert e.type == EntityType.ORGANIZATION

    def test_entity_id_must_start_with_ent(self):
        with pytest.raises(Exception):
            Entity(
                id="bad_id",
                name="Test",
                type=EntityType.ORGANIZATION,
                source_chunk_id="chunk_001",
                confidence=0.9,
            )

    def test_entity_name_stripped(self):
        e = Entity(
            id="ent_chunk_001_001",
            name="  Microsoft  ",
            type=EntityType.ORGANIZATION,
            source_chunk_id="chunk_001",
            confidence=0.9,
        )
        assert e.name == "Microsoft"

    def test_entity_blank_name_raises(self):
        with pytest.raises(Exception):
            Entity(
                id="ent_chunk_001_001",
                name="   ",
                type=EntityType.ORGANIZATION,
                source_chunk_id="chunk_001",
                confidence=0.9,
            )

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            Entity(
                id="ent_chunk_001_001",
                name="Test",
                type=EntityType.ORGANIZATION,
                source_chunk_id="chunk_001",
                confidence=1.5,
            )


class TestRelationSchema:
    def test_valid_relation(self):
        ent1 = Entity(id="ent_c1_001", name="A", type=EntityType.ORGANIZATION, source_chunk_id="c1", confidence=0.9)
        ent2 = Entity(id="ent_c1_002", name="B", type=EntityType.PERSON, source_chunk_id="c1", confidence=0.9)
        rel = Relation(
            id="rel_ent_c1_001_ent_c1_002_ACQUIRED",
            source_entity_id="ent_c1_001",
            target_entity_id="ent_c1_002",
            relation_type="ACQUIRED",
            source_chunk_id="c1",
            confidence=0.85,
        )
        assert rel.relation_type == "ACQUIRED"

    def test_relation_id_must_start_with_rel(self):
        with pytest.raises(Exception):
            Relation(
                id="invalid_id",
                source_entity_id="ent_c1_001",
                target_entity_id="ent_c1_002",
                relation_type="ACQUIRED",
                source_chunk_id="c1",
                confidence=0.85,
            )


class TestExtractionResult:
    def test_round_trip_valid(self):
        e = Entity(id="ent_c1_001", name="Apple", type=EntityType.ORGANIZATION, source_chunk_id="c1", confidence=0.9)
        r = ExtractionResult(chunk_id="c1", entities=[e], relations=[])
        assert r.chunk_id == "c1"
        assert len(r.entities) == 1

    def test_dangling_relation_raises(self):
        """Relations referencing non-existent entities should raise."""
        e = Entity(id="ent_c1_001", name="Apple", type=EntityType.ORGANIZATION, source_chunk_id="c1", confidence=0.9)
        bad_rel = Relation(
            id="rel_ent_c1_001_ent_c1_999_ACQUIRED",
            source_entity_id="ent_c1_001",
            target_entity_id="ent_c1_999",  # not in entities list
            relation_type="ACQUIRED",
            source_chunk_id="c1",
            confidence=0.8,
        )
        with pytest.raises(Exception, match="unknown"):
            ExtractionResult(chunk_id="c1", entities=[e], relations=[bad_rel])


class TestDocumentChunk:
    def test_valid_chunk(self):
        c = DocumentChunk(id="chunk_001", text="Hello world.", source_document="test.txt", chunk_index=0)
        assert c.id == "chunk_001"

    def test_blank_text_raises(self):
        with pytest.raises(Exception):
            DocumentChunk(id="chunk_001", text="   ", source_document="test.txt", chunk_index=0)


class TestCommunitySchema:
    def test_size_computed(self):
        comm = Community(id="com_0_0001", level=0, entity_ids=["ent_1", "ent_2", "ent_3"])
        assert comm.size == 3

    def test_id_format(self):
        comm = Community(id="com_0_0001", level=0)
        assert comm.id.startswith("com_")


# ─── Helper function tests ─────────────────────────────────────────────────────

class TestHelpers:
    def test_make_entity_id(self):
        assert _make_entity_id("chunk_001", 1) == "ent_chunk_001_001"
        assert _make_entity_id("chunk_001", 42) == "ent_chunk_001_042"

    def test_make_relation_id(self):
        rid = _make_relation_id("ent_c1_001", "ent_c1_002", "ACQUIRED")
        assert rid == "rel_ent_c1_001_ent_c1_002_ACQUIRED"

    def test_sanitize_entity_type_known(self):
        assert _sanitize_entity_type("ORGANIZATION") == EntityType.ORGANIZATION
        assert _sanitize_entity_type("person") == EntityType.PERSON

    def test_sanitize_entity_type_unknown(self):
        assert _sanitize_entity_type("ALIEN_ENTITY") == EntityType.OTHER


# ─── Parser tests ──────────────────────────────────────────────────────────────

class TestParseExtractionResponse:
    SAMPLE_JSON = json.dumps({
        "entities": [
            {
                "id": "ent_chunk_001_001",
                "name": "Apple Inc.",
                "type": "ORGANIZATION",
                "description": "Consumer technology company",
                "source_chunk_id": "chunk_001",
                "confidence": 0.97,
                "aliases": [],
                "attributes": {},
            },
            {
                "id": "ent_chunk_001_002",
                "name": "Tim Cook",
                "type": "PERSON",
                "description": "CEO of Apple Inc.",
                "source_chunk_id": "chunk_001",
                "confidence": 0.95,
                "aliases": [],
                "attributes": {},
            },
        ],
        "relations": [
            {
                "id": "rel_ent_chunk_001_001_ent_chunk_001_002_IS_CEO_OF",
                "source_entity_id": "ent_chunk_001_001",
                "target_entity_id": "ent_chunk_001_002",
                "relation_type": "IS_CEO_OF",
                "description": "Tim Cook is CEO",
                "source_chunk_id": "chunk_001",
                "confidence": 0.93,
                "attributes": {},
            }
        ],
    })

    def test_parse_valid_response(self):
        entities, relations = _parse_extraction_response(
            self.SAMPLE_JSON, "chunk_001", "test.txt", 0.70
        )
        assert len(entities) == 2
        assert len(relations) == 1
        assert entities[0].name == "Apple Inc."

    def test_confidence_filter(self):
        """With a high threshold, low-confidence entities are filtered."""
        entities, relations = _parse_extraction_response(
            self.SAMPLE_JSON, "chunk_001", "test.txt", 0.99
        )
        assert len(entities) == 0

    def test_markdown_fence_stripped(self):
        fenced = "```json\n" + self.SAMPLE_JSON + "\n```"
        entities, _ = _parse_extraction_response(fenced, "chunk_001", "test.txt", 0.70)
        assert len(entities) == 2

    def test_invalid_json_returns_empty(self):
        entities, relations = _parse_extraction_response(
            "not valid json {{{", "chunk_001", "test.txt", 0.70
        )
        assert entities == []
        assert relations == []

    def test_dangling_relation_filtered(self):
        """Relations where endpoint IDs don't match any entity are dropped."""
        data = {
            "entities": [{
                "id": "ent_chunk_001_001",
                "name": "Apple", "type": "ORGANIZATION",
                "description": "", "source_chunk_id": "chunk_001",
                "confidence": 0.9, "aliases": [], "attributes": {},
            }],
            "relations": [{
                "id": "rel_ent_chunk_001_001_ent_chunk_001_999_ACQUIRED",
                "source_entity_id": "ent_chunk_001_001",
                "target_entity_id": "ent_chunk_001_999",  # doesn't exist
                "relation_type": "ACQUIRED",
                "description": "", "source_chunk_id": "chunk_001",
                "confidence": 0.9, "attributes": {},
            }],
        }
        entities, relations = _parse_extraction_response(
            json.dumps(data), "chunk_001", "test.txt", 0.70
        )
        assert len(entities) == 1
        assert len(relations) == 0


# ─── Extractor async tests ─────────────────────────────────────────────────────

class TestEntityExtractor:
    @pytest.mark.asyncio
    async def test_extract_chunk_returns_result(self):
        """Test that extract_chunk returns an ExtractionResult even with mocked API."""
        extractor = EntityExtractor(api_key="test_key")

        mock_response = MagicMock()
        mock_block = MagicMock()
        mock_block.text = json.dumps({
            "entities": [{
                "id": "ent_chunk_001_001",
                "name": "Microsoft",
                "type": "ORGANIZATION",
                "description": "Tech company",
                "source_chunk_id": "chunk_001",
                "confidence": 0.95,
                "aliases": [],
                "attributes": {},
            }],
            "relations": [],
        })
        mock_response.content = [mock_block]

        with patch.object(extractor._client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            chunk = DocumentChunk(id="chunk_001", text="Microsoft is a tech company.", source_document="test.txt", chunk_index=0)
            result = await extractor.extract_chunk(chunk)

        assert isinstance(result, ExtractionResult)
        assert len(result.entities) == 1
        assert result.entities[0].name == "Microsoft"

    @pytest.mark.asyncio
    async def test_extract_chunk_api_error_returns_empty(self):
        """API errors should return empty ExtractionResult, not raise."""
        import anthropic
        extractor = EntityExtractor(api_key="test_key")

        with patch.object(extractor._client.messages, 'create', new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = anthropic.APIConnectionError(request=MagicMock())
            chunk = DocumentChunk(id="chunk_001", text="Test text.", source_document="test.txt", chunk_index=0)
            result = await extractor.extract_chunk(chunk)

        assert isinstance(result, ExtractionResult)
        assert result.entities == []
