"""In-memory hybrid search index (semantic + BM25, fused with RRF).

For a personal-scale corpus (a few thousand statute chunks) a NumPy dot product
plus an in-memory BM25 index is instant and needs no vector database.
"""
from __future__ import annotations

import json
import re

import numpy as np
from rank_bm25 import BM25Okapi

from . import config

_TOKEN_RE = re.compile(r"[a-zäöüß0-9§]+", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Store:
    def __init__(self, embeddings: np.ndarray, chunks: list[dict]):
        self.embeddings = embeddings
        self.chunks = chunks
        corpus = [
            tokenize(f"{c['citation']} {c.get('title','')} {c['context']}")
            for c in chunks
        ]
        self.bm25 = BM25Okapi(corpus)

    @classmethod
    def load(cls) -> "Store":
        if not config.EMBEDDINGS_FILE.exists():
            raise RuntimeError("No index found. Run:  python -m app.ingest && python -m app.index")
        embeddings = np.load(config.EMBEDDINGS_FILE)
        chunks = json.loads(config.INDEX_CHUNKS_FILE.read_text(encoding="utf-8"))
        return cls(embeddings, chunks)

    def _rrf(self, ranked_lists: list[list[int]]) -> dict[int, float]:
        scores: dict[int, float] = {}
        for ranking in ranked_lists:
            for rank, idx in enumerate(ranking):
                scores[idx] = scores.get(idx, 0.0) + 1.0 / (config.RRF_K + rank + 1)
        return scores

    def search(self, query_vec: np.ndarray, query_text: str, k: int) -> list[dict]:
        # Semantic: cosine == dot product (vectors are normalized).
        sims = self.embeddings @ query_vec
        vec_rank = np.argsort(-sims)[: config.TOP_K_VECTOR].tolist()

        # Keyword: BM25 over the same chunks. Only keep docs it actually matched
        # (score > 0); otherwise an all-zero query (e.g. English words vs German
        # text) yields a degenerate ranking that pollutes the fusion.
        bm_scores = self.bm25.get_scores(tokenize(query_text))
        bm_rank = [int(i) for i in np.argsort(-bm_scores) if bm_scores[i] > 0][
            : config.TOP_K_BM25
        ]

        fused = self._rrf([vec_rank, bm_rank])
        top = sorted(fused, key=lambda i: -fused[i])[:k]

        results = []
        for idx in top:
            chunk = dict(self.chunks[idx])
            chunk["score"] = round(float(fused[idx]), 5)
            results.append(chunk)
        return results
