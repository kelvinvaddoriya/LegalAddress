"""Retrieval recall check against eval/questions.json.

For each question, did hybrid retrieval surface at least one of the expected
sections? This is the fast signal when you change chunking, embeddings, or the
retrieval knobs. Generation quality is still eyeballed in the UI.

Run:  python -m eval.run
"""
from __future__ import annotations

import json
from pathlib import Path

from app.retrieve import retrieve
from app.store import Store

QUESTIONS = Path(__file__).parent / "questions.json"


def section_of(citation: str) -> str:
    # "§ 573c BGB Abs. 1" -> "573c"
    return citation.replace("§", "").strip().split()[0].lower()


def main() -> None:
    store = Store.load()
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    passed = 0
    for item in data["questions"]:
        chunks = retrieve(store, item["q"])
        got = {section_of(c["citation"]) for c in chunks}
        want = {s.lower() for s in item["expect"]}
        ok = bool(got & want)
        passed += ok
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {item['q'][:60]:<60} want={sorted(want)} got={sorted(got)}")
    total = len(data["questions"])
    print(f"\nRecall@{len(chunks)}: {passed}/{total}")


if __name__ == "__main__":
    main()
