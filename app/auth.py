"""User accounts and sessions — stdlib only (sqlite3 + hashlib + secrets).

Deliberately no extra dependencies (passlib/bcrypt/itsdangerous): keeps the
project pure-Python and safe on Python 3.14, matching the rest of the stack.

  - Passwords: PBKDF2-HMAC-SHA256 with a per-user random salt.
  - Sessions: opaque random tokens; only their SHA-256 hash is stored, so a
    leaked database can't be replayed as live sessions.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time

from . import config

ITERATIONS = 240_000
SESSION_TTL = config.SESSION_TTL_DAYS * 24 * 3600


# --- Password hashing --------------------------------------------------
def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _algo, iters, salt_hex, hash_hex = stored.split("$")
        dk = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iters)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk.hex(), hash_hex)


# --- Database ----------------------------------------------------------
def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(config.AUTH_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    config.AUTH_DB.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute(
            """CREATE TABLE IF NOT EXISTS users(
                   id          INTEGER PRIMARY KEY AUTOINCREMENT,
                   email       TEXT UNIQUE NOT NULL,
                   pw_hash     TEXT NOT NULL,
                   created_at  REAL NOT NULL)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS sessions(
                   token_hash  TEXT PRIMARY KEY,
                   user_id     INTEGER NOT NULL,
                   created_at  REAL NOT NULL,
                   expires_at  REAL NOT NULL)"""
        )


# --- Users -------------------------------------------------------------
def _normalize_email(email: str) -> str:
    return email.strip().lower()


def create_user(email: str, password: str) -> int:
    """Insert a new user, returning its id. Raises ValueError on bad input
    or a duplicate email."""
    email = _normalize_email(email)
    if "@" not in email or len(email) < 3:
        raise ValueError("Enter a valid email address.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    with _connect() as c:
        try:
            cur = c.execute(
                "INSERT INTO users(email, pw_hash, created_at) VALUES(?,?,?)",
                (email, hash_password(password), time.time()),
            )
        except sqlite3.IntegrityError:
            raise ValueError("An account with that email already exists.")
        return int(cur.lastrowid)


def verify_user(email: str, password: str) -> int | None:
    """Return the user id if credentials are valid, else None."""
    email = _normalize_email(email)
    with _connect() as c:
        row = c.execute(
            "SELECT id, pw_hash FROM users WHERE email=?", (email,)
        ).fetchone()
    if row and verify_password(password, row["pw_hash"]):
        return int(row["id"])
    return None


# --- Sessions ----------------------------------------------------------
def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    with _connect() as c:
        c.execute(
            "INSERT INTO sessions(token_hash, user_id, created_at, expires_at) "
            "VALUES(?,?,?,?)",
            (_hash_token(token), user_id, now, now + SESSION_TTL),
        )
    return token


def user_for_token(token: str | None) -> dict | None:
    """Resolve a session cookie to {'id', 'email'}, or None. Expired sessions
    are pruned on access."""
    if not token:
        return None
    with _connect() as c:
        row = c.execute(
            """SELECT u.id AS id, u.email AS email, s.expires_at AS expires_at
                   FROM sessions s JOIN users u ON u.id = s.user_id
                   WHERE s.token_hash = ?""",
            (_hash_token(token),),
        ).fetchone()
    if not row:
        return None
    if row["expires_at"] < time.time():
        delete_session(token)
        return None
    return {"id": int(row["id"]), "email": row["email"]}


def delete_session(token: str | None) -> None:
    if not token:
        return
    with _connect() as c:
        c.execute("DELETE FROM sessions WHERE token_hash=?", (_hash_token(token),))
