"""
Stage 1: Ingest and chunk documents.
Reads raw documents and splits them into overlapping text chunks.
Writes: data/processed/chunks.json, data/processed/ingest_metrics.json
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import click
import structlog

logger = structlog.get_logger(__name__)

# Lazy import to avoid requiring all dependencies for simple tasks
def _load_params() -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import]
        with open("params.yaml") as f:
            return yaml.safe_load(f)
    except Exception:
        return {"extraction": {"chunk_size": 1024, "chunk_overlap": 128}}


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    # Split on sentence endings
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current:
            chunks.append(" ".join(current))
            # Overlap: keep last ~overlap chars worth of sentences
            overlap_sents: list[str] = []
            overlap_len = 0
            for s in reversed(current):
                if overlap_len + len(s) <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current = overlap_sents
            current_len = overlap_len
        current.append(sent)
        current_len += sent_len

    if current:
        chunks.append(" ".join(current))
    return [c.strip() for c in chunks if c.strip()]


def make_chunk_id(source: str, index: int) -> str:
    base = f"{Path(source).stem}_{index:04d}"
    return f"chunk_{hashlib.md5(base.encode()).hexdigest()[:8]}"


def ingest_directory(input_dir: Path, chunk_size: int, overlap: int) -> tuple[list[dict], dict]:
    chunks: list[dict] = []
    file_count = 0
    total_chars = 0

    for ext in ["*.txt", "*.md", "*.json"]:
        for fpath in sorted(input_dir.rglob(ext)):
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
                total_chars += len(text)
                file_count += 1
                text_chunks = _chunk_text(text, chunk_size, overlap)
                for idx, chunk_text in enumerate(text_chunks):
                    cid = make_chunk_id(str(fpath), idx)
                    chunks.append({
                        "id": cid,
                        "text": chunk_text,
                        "source_document": str(fpath.name),
                        "chunk_index": idx,
                        "metadata": {"file": str(fpath), "size": len(chunk_text)},
                    })
                logger.info("file_ingested", file=fpath.name, chunks=len(text_chunks))
            except Exception as exc:
                logger.error("file_ingest_error", file=str(fpath), error=str(exc))

    metrics = {
        "files_ingested": file_count,
        "chunks_created": len(chunks),
        "total_chars": total_chars,
        "avg_chunk_len": round(total_chars / max(len(chunks), 1), 1),
    }
    return chunks, metrics


@click.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True))
@click.option("--output", "output_dir", required=True, type=click.Path())
def main(input_dir: str, output_dir: str) -> None:
    params = _load_params()
    chunk_size = params["extraction"]["chunk_size"]
    overlap = params["extraction"]["chunk_overlap"]

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    chunks, metrics = ingest_directory(Path(input_dir), chunk_size, overlap)
    metrics["ingest_duration_s"] = round(time.monotonic() - t0, 2)

    (out / "chunks.json").write_text(json.dumps(chunks, indent=2))
    (out / "ingest_metrics.json").write_text(json.dumps(metrics, indent=2))

    logger.info("ingest_complete", **metrics)


if __name__ == "__main__":
    main()
