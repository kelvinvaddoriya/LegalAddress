# Architecture

This document explains LegalAddress from first principles: why it exists, the
decisions that shaped it, how the pieces fit together today, and where the
project is heading. The README is the operator's manual (how to run it); this is
the map of *how it came to be and how it works inside*.

---

## 1. What it is, and why it exists

LegalAddress is a **small, private, bilingual (EN/DE) web app** that answers
everyday legal questions for people living in Germany — students and working
professionals — about housing, work, contracts, residence, insurance, and their
rights.

The animating problem: German law is authoritative, public, and free to read at
`gesetze-im-internet.de` — but it is in dense legal German, organized by statute
rather than by life situation. Someone who just got a `Kündigung`, or whose
residence permit is expiring, doesn't know which paragraph governs their case,
and a general-purpose chatbot will happily *invent* a section number that sounds
right. The whole design exists to close that gap **without hallucinating law**.

Two hard commitments fall out of that and constrain everything downstream:

1. **Grounded, not generative.** Every legal statement is answered *only* from
   real statute text retrieved for the question, and cited to the exact
   paragraph (`§ 573c BGB`). If the sources don't cover it, the app says so.
2. **Information, not advice.** It explains what the law generally says, keeps
   the German legal term next to the English (so the user has the right word for
   authorities), and routes people to a `Rechtsanwalt` or advice centre —
   loudly, when the matter is time-sensitive (deadlines, dismissal, eviction,
   expiring visa).

It is built for **personal use, not the market**. That single fact justifies the
most important architectural choice: *stay small.*

---

## 2. The governing constraints

Almost every decision in the codebase traces back to four self-imposed limits:

- **No vector database.** The corpus is a few hundred statute sections. A NumPy
  matrix in RAM with a dot product is instant and needs no infrastructure.
- **No local ML models.** Only two external API providers are allowed: **OpenAI**
  (embeddings) and **Anthropic** (reranking + answer generation). Nothing else
  to host, fine-tune, or keep warm.
- **No emojis, no decoration in answers.** The output is legal information; it
  reads like prose, not chat. This is enforced in code, not just by prompt.
- **Citations are load-bearing.** The paragraph is simultaneously the *chunk
  unit*, the *citation unit*, and the *filter unit* for which sources get shown.

Keep these in mind — they explain why the code looks deliberately under-built in
places (e.g. an in-memory store, a synchronous request path). That is the point.

---

## 3. The shape of the system

There are two distinct lifecycles. They share a data model but run at different
times.

```
        ┌─────────────────────── OFFLINE: build the knowledge base ───────────────────────┐
        │                                                                                  │
 sources.json ──▶ ingest.py ──▶ chunk.py ──▶ data/chunks.json ──▶ index.py ──▶ data/index/ │
 (which laws    (download +    (1 § per      (section records)   (embed via    (embeddings.npy
  & §-ranges)    parse XML)     chunk)                            OpenAI)        + chunks.json) │
        │                                                                                  │
        └──────────────────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────── ONLINE: serve an answer (per request) ───────────────────┐
        │                                                                                  │
 question ──▶ embed query ──▶ store.search (vector + BM25, RRF) ──▶ rerank (Claude Haiku)  │
                                                                          │                │
            stream answer (Claude Sonnet) ◀── build grounded prompt ◀─────┘                │
                  │                                                                         │
                  └──▶ strip emoji ──▶ filter to only cited sources ──▶ NDJSON to browser   │
        │                                                                                  │
        └──────────────────────────────────────────────────────────────────────────────────┘
```

The offline path is run by hand (`python -m app.ingest && python -m app.index`)
whenever coverage changes. The online path runs inside FastAPI on every `/ask`.

---

## 4. The offline path — building the knowledge base

### 4.1 `data/sources.json` — the coverage manifest

This is the single source of truth for *what law the app knows*. Each entry is
one statute slice: a `gesetze-im-internet.de` slug, a human domain label
(`housing`, `work`, `visa`, …), and `include` ranges of paragraph numbers
(`"535-577a"`; empty means the whole law). The same law can appear under several
domains with different ranges — BGB shows up three times (housing, contracts,
work) because the German Civil Code spans all three.

Growing coverage is intentionally a *config edit*, not a code change: add an
entry, re-run ingest + index.

### 4.2 `app/ingest.py` — download and parse

For each entry it fetches the law's XML zip from `gesetze-im-internet.de`
(cached in `data/raw/<slug>.zip` so re-runs are offline), unzips it, and walks
every `<norm>`. It keeps only norms with a real paragraph label (`<enbez>` like
`§ 573c`) — skipping structural headings and preambles — flattens the body text
out of the XML markup, filters by the `include` ranges, and emits one **section
record** per paragraph. A deterministic `source_url` is built back to the
official HTML page for that section.

Section numbers are parsed as `(number, letter-suffix)` so that `§ 18a` sorts
and range-matches correctly against `§ 18`–`§ 21`.

### 4.3 `app/chunk.py` — one § per chunk

The natural retrieval unit *is* the citation unit: a single paragraph. So by
default a section becomes exactly one chunk. The only exception is length: a
section longer than `MAX_CHUNK_CHARS` (3500) is split on `Absatz` boundaries
(`(1)`, `(2)`, …), carrying the title and citation into each piece and labelling
it (`§ X Abs. 1`). This keeps chunks self-describing and citations honest even
after a split.

### 4.4 `app/embed.py` + `app/index.py` — embed and persist

`index.py` embeds `citation + title + body` for each chunk (so the section
number and heading are matchable, not just the legal prose) via OpenAI
`text-embedding-3-large`. Vectors are **L2-normalized** at write time, which is
what lets the online search use a plain dot product as cosine similarity. The
result is persisted as two files: `data/index/embeddings.npy` (the matrix) and
`data/index/chunks.json` (the aligned metadata). Together these *are* the index.

Current scale: ~611 chunks across nine domains.

---

## 5. The online path — answering a question

### 5.1 `app/main.py` — the FastAPI surface

Tiny by design. On startup (`lifespan`) it loads the index once into an
in-memory `Store`; if the index isn't built yet it records the error and
surfaces it per-request as a 503 rather than crashing. Three routes: `GET /`
serves the page, `/static` serves assets, and `POST /ask` does the work.

`/ask` runs **retrieval and rerank up front** (so failures there return a clean
JSON error *before* streaming starts), then streams the answer as
**NDJSON** — one JSON event per line: many `{"type":"delta","text":…}` events,
then a single `{"type":"done", sources, escalation}` (or `{"type":"error"}`).

### 5.2 `app/store.py` — hybrid search

`Store.search` runs two retrievers over the same chunks and fuses them:

- **Semantic:** `embeddings @ query_vec` (cosine, since everything is
  normalized), take the top `TOP_K_VECTOR` (30).
- **Keyword:** BM25 (`rank-bm25`) over the tokenized `citation + title + body`,
  take the top `TOP_K_BM25` (30). Critically, it **discards zero-score matches** —
  an English query against German text can otherwise produce a degenerate
  all-zero ranking that pollutes the fusion.
- **Fusion:** Reciprocal Rank Fusion (`RRF_K = 60`) combines the two rankings
  into one score, and the top pool is returned.

This is the heart of the bilingual story: embeddings catch cross-lingual meaning
("notice period" ≈ `Kündigungsfrist`), BM25 catches exact section numbers and
German terms the user typed verbatim, and RRF lets each cover the other's blind
spot.

### 5.3 `app/rerank.py` — the LLM cross-check

Fusion is good at *recall* but can mis-order the top few — again, worst for
English questions against German statutes. So the fused pool (`RERANK_POOL` = 14)
is handed to **Claude Haiku**, which reads the actual question and candidate
snippets and returns just a JSON array of the best indices. The call is
constrained hard (temperature 0, an assistant prefill of `[`, a `]` stop
sequence) so it emits *only* the array. It is defensive on both ends: a bad parse
or any exception **falls back to the fusion order**, so retrieval never
hard-fails on the rerank step. The top `TOP_N_CONTEXT` (6) chunks survive into
the answer.

This reuses the Anthropic key already needed for generation — no third provider,
honouring the two-provider rule.

### 5.4 `app/prompts.py` + `app/generate.py` — grounded generation

`prompts.SYSTEM` encodes the two hard commitments as explicit ground rules:
answer only from the supplied sources, cite the exact paragraph inline, keep the
German term beside the English, give information not advice, escalate
deadlines/dismissals/visa loudly, clean Markdown, **no emojis**. The user message
packs the retrieved sections as an authoritative `SOURCES` block followed by the
question and the target language.

`generate.py` is the only place answers are produced. It streams from Claude
(`claude-sonnet-4-6` by default), and the system prompt is sent with
`cache_control: ephemeral` so the static instructions are prompt-cached across
requests. Two safety nets run on the output:

- **`strip_emoji`** — a regex belt-and-braces removal of any pictograph the model
  slips in despite the instruction.
- **`filter_cited_sources`** — the answer text is scanned for each candidate's
  citation (a flexible regex that tolerates `Absatz` groups and multi-word laws
  like `SGB V`), and **only the sources actually cited are returned**, in order
  of appearance. The user never sees a "source" the answer didn't lean on.

Separately, **`needs_escalation`** does a deterministic, language-agnostic
keyword scan of the *question* (`kündigung`, `frist`, `visa`, `deadline`, …) to
raise the UI warning banner — independent of the model, so a time-sensitive
matter is flagged even if generation hiccups.

### 5.5 `static/index.html` — the page

A single self-contained file: textarea, EN/DE toggle, light/dark toggle, and a
streaming reader that parses the NDJSON, renders a safe subset of Markdown
(escape-first, then headings/bold/bullets), shows the escalation banner, and
lists the cited sources as links. As of the most recent work (June 2026) the UI
was redesigned into a warm, editorial "old book" style — paper/sepia light theme
and a candlelit dark theme, served by Fraunces + Spectral serif fonts, with the
theme choice persisted in `localStorage`. All streaming/citation logic was kept
intact; only presentation changed.

---

## 6. The data model (one record, end to end)

A single chunk dict carries everything from ingest to the browser, which is why
no database is needed:

| field | meaning | set by |
|-------|---------|--------|
| `law`, `law_name_de`, `law_name_en` | statute identity | sources.json |
| `domain` | life area (`housing`, `work`, …) | sources.json |
| `section` | `§ 573c` | ingest |
| `citation` | `§ 573c BGB` (+ `Abs. n` if split) | ingest / chunk |
| `title` | the section heading | ingest |
| `text` / `context` | full body / the chunk's body | ingest / chunk |
| `source_url` | link to the official page | ingest |
| `id` | dedupe key (`573cBGB`) | chunk |
| `score` | RRF fusion score | store (per query) |

The same shape is embedded, searched, reranked, formatted into the prompt, and
filtered for display — never reshaped, only annotated.

---

## 7. Quality: how we know it works

`eval/run.py` is a **retrieval-recall** harness, not an answer grader. For each
question in `eval/questions.json` (which pairs a question with the section a good
answer *should* cite), it runs the full retrieve→rerank path and checks whether
any expected section made the final context. That is the fast, objective signal
to run after any change to chunking, embeddings, or the retrieval knobs in
`config.py`. Generation quality itself is still eyeballed in the UI — deliberate,
given the personal scale.

---

## 8. The current arc

Where the project has been and where it points:

- **Foundation (done):** the offline pipeline, the in-memory hybrid store, and
  grounded streaming generation — the core "cite-real-law-or-say-you-can't" loop.
- **Retrieval hardening (done):** BM25 zero-score filtering and the Claude-Haiku
  reranker were added specifically to fix cross-lingual mis-ranking (English
  question → German statute), the system's sharpest failure mode.
- **Coverage growth (ongoing):** from a housing/work core out to nine domains
  (~611 chunks) including visa/residence, insurance, student finance, and
  parental leave. The latest batch (June 2026) added the "what happens after a
  job ends" path (unemployment benefit & job-seeking registration, SGB III),
  **citizenship** as a new domain (naturalisation & post-2024 dual nationality,
  StAG), and family reunification (AufenthG §§ 27–36) — each paired with new
  `eval/questions.json` recall checks. This is the dimension that scales by
  editing `sources.json`. **StAG coverage is considered complete:** the 2024
  reform (StARModG) repealed both § 12 and § 25 (`weggefallen` in the source
  XML), so dual-citizenship law is now fully carried by § 10 (naturalising
  foreigners need not renounce their prior nationality) — there are no further
  operative StAG sections to ingest.
- **Presentation (recent — June 2026):** the editorial "old book" UI redesign
  with a light/dark toggle; logic untouched.

The natural next steps, consistent with the constraints, are *more coverage*
(more statutes/ranges, more eval questions to guard recall) and *retrieval
tuning* (the knobs in `config.py`) — not new infrastructure. The architecture is
intentionally finished where it is finished: a personal tool that stays small,
cites real law, and refuses to pretend.

---

## 9. File map (the one-paragraph index)

| Path | Role |
|------|------|
| `data/sources.json` | coverage manifest — which laws & §-ranges to ingest |
| `app/ingest.py` | download + parse statute XML into section records |
| `app/chunk.py` | one § per chunk; split long sections by `Absatz` |
| `app/embed.py` | OpenAI embeddings (normalized) |
| `app/index.py` | embed all chunks, persist `data/index/` |
| `app/store.py` | in-memory hybrid search: vector + BM25 fused with RRF |
| `app/rerank.py` | Claude-Haiku reranker over the fused pool (fails safe) |
| `app/retrieve.py` | glue: query → embed → search → rerank |
| `app/prompts.py` | system prompt, source formatting, citation filter, escalation |
| `app/generate.py` | grounded streamed answer; emoji strip; cited-source filter |
| `app/config.py` | paths, model names, retrieval knobs, key loading |
| `app/main.py` | FastAPI: serve page, stream `/ask` as NDJSON |
| `static/index.html` | the bilingual editorial page (EN/DE, light/dark) |
| `eval/run.py` | retrieval-recall check against `eval/questions.json` |
