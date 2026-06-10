"""LLM reranker: pick the most relevant sections from the candidate pool.

Hybrid (vector + BM25) fusion is good at recall but can mis-order the top few,
especially for English questions against German statutes. A fast model reads the
actual question and candidate texts and selects the best N. Falls back to the
fusion order on any error, so retrieval never hard-fails on the rerank step.
"""
from __future__ import annotations

import re

from anthropic import Anthropic

from . import config

_client: Anthropic | None = None


def _client_() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(
            api_key=config.require("ANTHROPIC_API_KEY", config.ANTHROPIC_API_KEY)
        )
    return _client


def _candidate_block(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        title = f" — {c['title']}" if c.get("title") else ""
        snippet = c["context"][:300].replace("\n", " ")
        lines.append(f"[{i}] {c['citation']}{title} ({c['law']}): {snippet}")
    return "\n".join(lines)


def rerank(question: str, candidates: list[dict], n: int) -> list[dict]:
    if not config.RERANK_ENABLED or len(candidates) <= n:
        return candidates[:n]

    prompt = (
        f"A user asked: \"{question}\"\n\n"
        f"Below are candidate German legal sections (numbered). Choose the {n} "
        f"that are MOST relevant and useful for answering the question, ordered "
        f"best first. Reply with ONLY a JSON array of the chosen index numbers, "
        f"e.g. [3,0,7]. No prose.\n\n{_candidate_block(candidates)}"
    )
    try:
        resp = _client_().messages.create(
            model=config.RERANK_MODEL,
            max_tokens=60,
            temperature=0,
            messages=[
                {"role": "user", "content": prompt},
                # Prefill forces the model to emit just the array, no rambling.
                {"role": "assistant", "content": "["},
            ],
            stop_sequences=["]"],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        # Defensive parse: take indices in order, drop dupes and out-of-range.
        order, seen = [], set()
        for tok in re.findall(r"\d+", text):
            i = int(tok)
            if 0 <= i < len(candidates) and i not in seen:
                seen.add(i)
                order.append(i)
        picked = [candidates[i] for i in order]
        if not picked:
            return candidates[:n]
        # Append any not chosen, so we always return up to n.
        seen = {id(c) for c in picked}
        for c in candidates:
            if len(picked) >= n:
                break
            if id(c) not in seen:
                picked.append(c)
        return picked[:n]
    except Exception:
        return candidates[:n]
