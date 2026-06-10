"""OpenAI embeddings (the only place we call OpenAI)."""
from __future__ import annotations

import numpy as np
from openai import OpenAI

from . import config

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.require("OPENAI_API_KEY", config.OPENAI_API_KEY))
    return _client


def embed_texts(texts: list[str], batch_size: int = 100) -> np.ndarray:
    """Return an L2-normalized (n, d) float32 matrix for cosine via dot product."""
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client().embeddings.create(model=config.EMBED_MODEL, input=batch)
        vectors.extend(d.embedding for d in resp.data)
    arr = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
