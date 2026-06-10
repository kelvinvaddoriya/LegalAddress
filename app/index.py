"""Embed chunks and persist the search index.

Run after ingest:
    python -m app.index
"""
from __future__ import annotations

import json

import numpy as np

from . import config, embed


def embed_text_for(chunk: dict) -> str:
    """What we actually embed: citation + title + body, so the model can match
    on the section number and heading as well as the legal text."""
    title = f" – {chunk['title']}" if chunk.get("title") else ""
    return f"{chunk['citation']}{title}\n{chunk['context']}"


def main() -> None:
    if not config.CHUNKS_FILE.exists():
        raise SystemExit("No chunks.json yet. Run:  python -m app.ingest")
    chunks = json.loads(config.CHUNKS_FILE.read_text(encoding="utf-8"))
    print(f"Embedding {len(chunks)} chunks with {config.EMBED_MODEL} ...")

    matrix = embed.embed_texts([embed_text_for(c) for c in chunks])

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_FILE, matrix)
    config.INDEX_CHUNKS_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved index: {matrix.shape[0]} vectors x {matrix.shape[1]} dims")


if __name__ == "__main__":
    main()
