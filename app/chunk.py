"""Chunking for statute text.

The natural unit is one paragraph (§), which is also our citation unit, so by
default a section becomes a single chunk. Sections longer than MAX_CHUNK_CHARS
are split on Absatz boundaries ("(1)", "(2)", ...) with the section title and
citation carried into each piece.
"""
from __future__ import annotations

import re

from .config import MAX_CHUNK_CHARS

# Split *before* an Absatz marker like "(1) ", "(12) " at a word boundary.
_ABSATZ_RE = re.compile(r"(?=(?:^|\s)\(\d+\)\s)")


def _chunk_id(citation: str, suffix: str = "") -> str:
    base = citation.replace("§", "").replace(" ", "").strip()
    return f"{base}{suffix}"


def split_section(sec: dict) -> list[dict]:
    text = sec["text"]
    if len(text) <= MAX_CHUNK_CHARS:
        return [{**sec, "id": _chunk_id(sec["citation"]), "context": text}]

    # Long section: split into Absatz groups, packing up to the size limit.
    pieces = [p.strip() for p in _ABSATZ_RE.split(text) if p.strip()]
    chunks: list[dict] = []
    buf: list[str] = []
    part = 1

    def flush() -> None:
        nonlocal buf, part
        if not buf:
            return
        body = " ".join(buf)
        m = re.match(r"\((\d+)\)", buf[0])
        absatz = f" Abs. {m.group(1)}" if m else f" (Teil {part})"
        chunks.append(
            {
                **sec,
                "id": _chunk_id(sec["citation"], f"-{part}"),
                "citation": f"{sec['citation']}{absatz}",
                "context": body,
            }
        )
        buf = []
        part += 1

    for piece in pieces:
        if buf and sum(len(b) for b in buf) + len(piece) > MAX_CHUNK_CHARS:
            flush()
        buf.append(piece)
    flush()
    return chunks
