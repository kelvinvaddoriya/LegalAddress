"""Grounded answer generation with Claude (the only place we call Anthropic
for answers). Streams tokens, strips any stray emojis, and returns only the
sources the answer actually cites.
"""
from __future__ import annotations

import re
from collections.abc import Iterator

from anthropic import Anthropic

from . import config, prompts

_client: Anthropic | None = None

# Strip emojis / pictographs / variation selectors, just in case the model
# slips one in despite the no-emoji instruction.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # emoji & supplemental symbols
    "\U00002600-\U000027BF"   # misc symbols + dingbats
    "\U00002300-\U000023FF"   # technical (clock, watch, ...)
    "\U00002B00-\U00002BFF"   # stars / arrows
    "\U00002190-\U000021FF"   # arrows
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "️‍⃣"      # variation selector, ZWJ, keycap
    "]"
)


def strip_emoji(text: str) -> str:
    return _EMOJI_RE.sub("", text)


def client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(
            api_key=config.require("ANTHROPIC_API_KEY", config.ANTHROPIC_API_KEY)
        )
    return _client


def _system() -> list[dict]:
    return [
        {
            "type": "text",
            "text": prompts.SYSTEM,
            "cache_control": {"type": "ephemeral"},  # static -> cached
        }
    ]


def answer(question: str, chunks: list[dict], lang: str = "en") -> dict:
    """Non-streaming answer (used by tests/eval)."""
    resp = client().messages.create(
        model=config.GEN_MODEL,
        max_tokens=1200,
        system=_system(),
        messages=[{"role": "user", "content": prompts.build_user_message(question, chunks, lang)}],
    )
    text = strip_emoji("".join(b.text for b in resp.content if b.type == "text"))
    return {
        "answer": text,
        "sources": prompts.filter_cited_sources(text, chunks),
        "escalation": prompts.needs_escalation(question),
    }


def answer_stream(question: str, chunks: list[dict], lang: str = "en") -> Iterator[dict]:
    """Yield {'type':'delta','text':...} events, then one {'type':'done',...}."""
    raw: list[str] = []
    with client().messages.stream(
        model=config.GEN_MODEL,
        max_tokens=1200,
        system=_system(),
        messages=[{"role": "user", "content": prompts.build_user_message(question, chunks, lang)}],
    ) as stream:
        for delta in stream.text_stream:
            raw.append(delta)
            yield {"type": "delta", "text": strip_emoji(delta)}

    full = strip_emoji("".join(raw))
    yield {
        "type": "done",
        "sources": prompts.filter_cited_sources(full, chunks),
        "escalation": prompts.needs_escalation(question),
    }
