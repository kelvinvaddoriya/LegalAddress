# LegalAddress

A small, private, bilingual (EN/DE) web app that answers everyday legal
questions for people living in Germany — students and working professionals —
about **housing, work, contracts, and rights**.

It gives grounded legal **information** (with citations to the actual German
statute), **not** legal advice. Built for personal use, not for the market.

## How it works

```
gesetze-im-internet.de XML  →  parse + chunk per §  →  OpenAI embeddings
                                                              │
question  →  hybrid retrieval (vector + BM25, RRF fused)  ────┘
          →  LLM rerank (Claude Haiku picks the most relevant sections)
          →  Claude streams the answer, grounded in & citing those sections
          →  only the cited sources are shown, in a FastAPI custom page
```

- **Embeddings:** OpenAI (`text-embedding-3-large`)
- **Generation:** Claude (`claude-sonnet-4-6` by default), streamed token-by-token
- **Reranker:** Claude Haiku (`claude-haiku-4-5`) — no extra API provider
- **Vector store:** in-memory NumPy (the corpus is small) + `rank-bm25`
- **Answers:** plain text, no emojis; only the sections the answer cites are listed
- No vector database, no local ML models — only two API providers (OpenAI, Anthropic).

## Setup

```powershell
# 1. keys
copy .env.example .env        # then edit .env and add your two keys

# 2. install (already done if the venv exists)
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 3. build the knowledge base
.venv\Scripts\python.exe -m app.ingest   # download + parse statutes -> data/chunks.json
.venv\Scripts\python.exe -m app.index    # embed -> data/index/

# 4. run
.venv\Scripts\python.exe -m uvicorn app.main:app --reload
# open http://127.0.0.1:8000
```

## Adding more law

Edit `data/sources.json` — add an entry with the law's `gesetze-im-internet.de`
slug, a domain, and the paragraph ranges to include (empty = whole law). Then
re-run `app.ingest` and `app.index`.

## Checking quality

`eval/questions.json` holds test questions with the section a good answer should
cite. Run `.venv\Scripts\python.exe -m eval.run` to check retrieval recall after
any change to chunking, embeddings, or retrieval settings.

## Layout

| Path | Role |
|------|------|
| `app/ingest.py` | download + parse statutes from XML |
| `app/chunk.py` | one § per chunk (split long ones by Absatz) |
| `app/embed.py` | OpenAI embeddings |
| `app/index.py` | build the persisted index |
| `app/store.py` | in-memory hybrid search (vector + BM25 + RRF) |
| `app/rerank.py` | LLM reranker (Claude Haiku) over the fused candidates |
| `app/retrieve.py` | query → embed → hybrid search → rerank → chunks |
| `app/prompts.py` | system prompt, source formatting, escalation triggers |
| `app/generate.py` | grounded, streamed answer from Claude; emoji-stripped; cited-source filter |
| `app/main.py` | FastAPI app |
| `static/index.html` | the bilingual page |

## Scope & limits

- General information only; always points users to a Rechtsanwalt / advice centre
  for their specific case, and flags time-sensitive matters (deadlines, dismissal).
- Coverage is only what's listed in `data/sources.json`. Currently ~611 chunks
  across nine domains:
  - **housing** — tenancy law (BGB §§ 535–577a), housing benefit / Wohngeld (WoGG §§ 1–7)
  - **work** — dismissal protection (KSchG), leave (BUrlG), employment contracts &
    notice (BGB §§ 611–630), minimum wage (MiLoG), part-time/fixed-term (TzBfG),
    sick pay (EntgFG), written terms (NachwG), working hours (ArbZG), job reference
    (GewO § 109), parental allowance & leave (BEEG), unemployment benefit &
    job-seeking registration / Sperrzeit (SGB III §§ 38, 137–164)
  - **contracts** — standard terms/AGB (BGB §§ 305–310), distance/consumer
    contracts & 14-day withdrawal (BGB §§ 312–312k, 355–361)
  - **rights** — anti-discrimination (AGG), maternity protection (MuSchG)
  - **residence** — address registration / Anmeldung incl. landlord confirmation
    (BMG §§ 17–23a)
  - **education** — student finance / BAföG (§§ 1–17)
  - **visa** — residence titles, study permit (§ 16b), post-study job search
    (§ 20), skilled-worker permits & EU Blue Card (§§ 18a–18g), self-employment
    (§ 21), settlement permit (§ 9), family reunification — spouse/children
    (§§ 27–36) — AufenthG §§ 4–21, 27–36
  - **insurance** — statutory health insurance status: who must be insured incl.
    students (§ 5), exemptions, voluntary & family insurance (SGB V §§ 5–10)
  - **citizenship** — naturalisation requirements incl. no longer having to give
    up prior nationality after the 2024 reform (§ 10), naturalisation of
    spouses (§ 9) — StAG §§ 3–4, 8–16
