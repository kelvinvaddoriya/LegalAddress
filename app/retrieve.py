"""Glue between query text, embeddings, the hybrid store, and reranking."""
from __future__ import annotations

from . import config, embed, rerank
from .store import Store


def retrieve(store: Store, question: str) -> list[dict]:
    """Return the final context chunks: hybrid search -> fusion -> rerank."""
    query_vec = embed.embed_query(question)
    pool_size = config.RERANK_POOL if config.RERANK_ENABLED else config.TOP_N_CONTEXT
    candidates = store.search(query_vec, question, k=pool_size)
    return rerank.rerank(question, candidates, config.TOP_N_CONTEXT)
