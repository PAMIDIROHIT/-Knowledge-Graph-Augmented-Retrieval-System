"""
Node embedding and FAISS index builder.
Embeds all Entity nodes using OpenAI text-embedding-3-small,
builds a FAISS index, and saves to artifacts/.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
import structlog
from openai import OpenAI

logger = structlog.get_logger(__name__)

EMBEDDING_DIM = 1536  # text-embedding-3-small output dimension


def embed_texts(
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
    api_key: Optional[str] = None,
) -> np.ndarray:
    """Embed a list of texts using OpenAI embeddings API in batches."""
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [e.embedding for e in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embeddings)
        logger.debug("embeddings_batch_done", progress=i + len(batch), total=len(texts))

    return np.array(all_embeddings, dtype=np.float32)


class GraphIndexer:
    """
    Builds and manages a FAISS index over Entity and Community nodes.
    The ID map file maintains the mapping: FAISS int index → entity/community ID.
    """

    def __init__(
        self,
        index_path: str = "artifacts/faiss_index.bin",
        id_map_path: str = "artifacts/faiss_id_map.json",
        embedding_model: str = "text-embedding-3-small",
        embedding_dim: int = EMBEDDING_DIM,
        api_key: Optional[str] = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.id_map_path = Path(id_map_path)
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self._api_key = api_key
        self._index: Optional[faiss.IndexFlatIP] = None
        self._id_map: list[str] = []  # position → entity/community id

    def build(
        self,
        entities: list[dict[str, Any]],
        communities: Optional[list[dict[str, Any]]] = None,
    ) -> None:
        """Build FAISS index from entity (and optionally community) nodes."""
        items = list(entities)
        if communities:
            items.extend(communities)

        texts = []
        ids = []
        for item in items:
            text = f"{item.get('name', '')} {item.get('type', '')} {item.get('description', '')} {item.get('summary', '')}".strip()
            texts.append(text or item.get("id", ""))
            ids.append(item["id"])

        logger.info("building_embeddings", count=len(texts))
        embeddings = embed_texts(texts, model=self.embedding_model, api_key=self._api_key)

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._index.add(embeddings)
        self._id_map = ids

        self.save()
        logger.info("faiss_index_built", vectors=self._index.ntotal)

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        self.id_map_path.write_text(json.dumps(self._id_map, indent=2))
        logger.info("faiss_index_saved", path=str(self.index_path))

    def load(self) -> None:
        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{self.index_path}'. "
                "Run 'dvc repro' to build the pipeline first."
            )
        self._index = faiss.read_index(str(self.index_path))
        self._id_map = json.loads(self.id_map_path.read_text())
        logger.info("faiss_index_loaded", vectors=self._index.ntotal)

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        api_key: Optional[str] = None,
    ) -> list[tuple[str, float]]:
        """
        Search the FAISS index for the top-k most similar nodes.
        Returns list of (entity_id, score) tuples.
        """
        if self._index is None:
            if not self.index_path.exists():
                logger.warning("faiss_index_not_built", path=str(self.index_path))
                return []
            self.load()

        query_vec = embed_texts([query_text], model=self.embedding_model, api_key=api_key or self._api_key)
        faiss.normalize_L2(query_vec)

        scores, indices = self._index.search(query_vec, top_k)
        results: list[tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            results.append((self._id_map[idx], float(score)))
        return results

    @property
    def is_loaded(self) -> bool:
        return self._index is not None
