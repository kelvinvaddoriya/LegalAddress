"""Central configuration and filesystem paths.

Loads keys from a local .env file. Nothing here talks to the network.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# --- Paths -------------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # downloaded statute XML zips
INDEX_DIR = DATA_DIR / "index"        # embeddings + chunk metadata
CHUNKS_FILE = DATA_DIR / "chunks.json"
SOURCES_FILE = DATA_DIR / "sources.json"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
INDEX_CHUNKS_FILE = INDEX_DIR / "chunks.json"
STATIC_DIR = ROOT / "static"
AUTH_DB = DATA_DIR / "auth.db"        # users + sessions (SQLite, stdlib only)

# --- Models ------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-large")
GEN_MODEL = os.getenv("GEN_MODEL", "claude-sonnet-4-6")
RERANK_MODEL = os.getenv("RERANK_MODEL", "claude-haiku-4-5-20251001")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- Auth --------------------------------------------------------------
# Session cookie lifetime, in days.
SESSION_TTL_DAYS = int(os.getenv("SESSION_TTL_DAYS", "30"))
# Set AUTH_COOKIE_SECURE=1 when served over HTTPS (e.g. a Cloudflare/ngrok
# tunnel). Leave 0 for plain-http LAN access, or the cookie won't be sent.
AUTH_COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "0") == "1"
# Optional invite gate. If set, signup requires this code — strongly
# recommended when the app is reachable beyond localhost, so strangers
# can't register and burn your OpenAI/Anthropic quota.
SIGNUP_CODE = os.getenv("SIGNUP_CODE") or None

# --- Retrieval knobs ---------------------------------------------------
TOP_K_VECTOR = 30     # candidates from semantic search
TOP_K_BM25 = 30       # candidates from keyword search
RERANK_POOL = 14      # candidates kept after RRF fusion, handed to the reranker
TOP_N_CONTEXT = 6     # chunks fed to the answer model after reranking
RRF_K = 60            # Reciprocal Rank Fusion constant

# Reranking: an LLM cross-checks the fused candidates against the question.
# Uses the Anthropic key you already have (no extra provider). Fixes cases
# where semantic + keyword fusion ranks a less-relevant section too high
# (e.g. cross-lingual English question -> German statute).
RERANK_ENABLED = True

# Source-text length above which a section is split by Absatz.
MAX_CHUNK_CHARS = 3500


def require(name: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value
