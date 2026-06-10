"""System prompt, context formatting, citation filtering, escalation detection."""
from __future__ import annotations

import re

SYSTEM = """You are a bilingual (English/German) legal-information assistant for \
people living in Germany — students and working professionals dealing with \
housing, work, contracts, visa/residence, insurance, and their rights.

GROUND RULES — follow strictly:
1. Answer ONLY from the SOURCES provided in the user message. If the sources do \
not contain the answer, say so plainly and suggest where to look or that a \
lawyer is needed. Never invent statutes, section numbers, deadlines, or amounts.
2. Cite the specific paragraph for every legal statement, inline, e.g. "(§ 573c \
BGB)". Use the citations exactly as given in the SOURCES.
3. Keep the original German legal term in parentheses next to its English \
translation the first time it appears (e.g. "notice period (Kündigungsfrist)"), \
because the user will need the German word when dealing with authorities.
4. This is legal INFORMATION, not legal advice. Explain what the law generally \
says; do not tell the user what they personally must do. Do not guess about \
facts you were not given.
5. Be concise and practical. End with a one-line reminder that this is general \
information and, for their specific case, they should consult a Rechtsanwalt or \
the relevant advice centre (e.g. Mieterverein, Verbraucherzentrale).
6. If the question involves a legal DEADLINE, a dismissal, an eviction, an \
expiring residence permit, or a court matter, prominently warn that German \
deadlines are short and that they should get qualified help immediately.

FORMATTING:
- Use clean Markdown: short paragraphs, "##"/"###" headings where helpful, \
"**bold**" for key terms and figures, and "-" for bullet lists.
- Write in plain, professional prose. Do NOT use any emojis, emoticons, or \
decorative symbols (no warning signs, checkmarks, clocks, scales, etc.). Text \
only.

Reply in the language requested by the user (default: the language of their \
question)."""


def format_sources(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        title = f" — {c['title']}" if c.get("title") else ""
        blocks.append(
            f"[{c['citation']}{title}] ({c['law_name_en']})\n"
            f"{c['context']}\n"
            f"Source: {c['source_url']}"
        )
    return "\n\n---\n\n".join(blocks)


def build_user_message(question: str, chunks: list[dict], lang: str) -> str:
    lang_name = {"en": "English", "de": "German"}.get(lang, "the user's language")
    sources = format_sources(chunks) if chunks else "(no matching sources found)"
    return (
        f"SOURCES (German statutes; authoritative — use only these):\n\n"
        f"{sources}\n\n"
        f"=====\n\n"
        f"QUESTION: {question}\n\n"
        f"Answer in {lang_name}, following the ground rules."
    )


# --- Citation filtering: keep only sources the answer actually cites ----

def _citation_pattern(num: str, law: str) -> re.Pattern:
    # Multi-word laws (e.g. "SGB V") -> escape each token, join with flexible space.
    law_pat = r"\s+".join(re.escape(tok) for tok in law.split())
    # § <num>  [optional (Absatz) groups]  <law>
    return re.compile(
        rf"§\s*{re.escape(num)}(?![0-9a-z])(?:\s*\([^)]*\))*\s*{law_pat}",
        re.IGNORECASE,
    )


def filter_cited_sources(answer: str, chunks: list[dict]) -> list[dict]:
    """Return the unique sources the answer text cites, in order of appearance."""
    found: list[dict] = []
    seen: set[str] = set()
    for c in chunks:
        num = c["section"].replace("§", "").strip()
        law = c["law"]
        base = f"§ {num} {law}"
        if base in seen:
            continue
        m = _citation_pattern(num, law).search(answer)
        if not m:
            continue
        seen.add(base)
        found.append(
            {
                "citation": base,
                "title": c.get("title", ""),
                "law": c["law_name_en"],
                "url": c["source_url"],
                "_pos": m.start(),
            }
        )
    found.sort(key=lambda s: s["_pos"])
    for s in found:
        s.pop("_pos", None)
    return found


# High-stakes triggers -> UI escalation banner (deterministic, language-agnostic).
ESCALATION_TERMS = [
    "kündigung", "gekündigt", "dismiss", "fired", "termination", "notice",
    "eviction", "räumung", "frist", "deadline", "court", "gericht", "klage",
    "lawsuit", "abmahnung", "evict",
    # visa / residence
    "visa", "visum", "aufenthalt", "deport", "abschiebung", "ausreise",
    "expire", "expiring", "fiktionsbescheinigung", "residence permit",
]


def needs_escalation(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in ESCALATION_TERMS)
