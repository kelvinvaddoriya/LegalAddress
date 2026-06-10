"""FastAPI app: serves the page and streams answers.

Run:
    .venv/Scripts/python.exe -m uvicorn app.main:app --reload
Then open http://127.0.0.1:8000
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    RedirectResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth, config, generate
from .retrieve import retrieve
from .store import Store

state: dict = {}

COOKIE = "la_session"


@asynccontextmanager
async def lifespan(app: FastAPI):
    auth.init_db()
    try:
        state["store"] = Store.load()
        print(f"Loaded {len(state['store'].chunks)} chunks.")
    except Exception as exc:  # index not built yet — surfaced per-request
        state["store"] = None
        state["error"] = str(exc)
        print(f"WARNING: {exc}")
    yield


app = FastAPI(title="LegalAddress", lifespan=lifespan)


# --- Auth helpers ------------------------------------------------------
def current_user(request: Request) -> dict | None:
    return auth.user_for_token(request.cookies.get(COOKIE))


def _set_session_cookie(resp: JSONResponse, token: str) -> None:
    resp.set_cookie(
        COOKIE,
        token,
        max_age=auth.SESSION_TTL,
        httponly=True,
        samesite="lax",
        secure=config.AUTH_COOKIE_SECURE,
        path="/",
    )


class Credentials(BaseModel):
    email: str
    password: str
    code: str | None = None


@app.post("/auth/signup")
def signup(creds: Credentials):
    if config.SIGNUP_CODE and (creds.code or "") != config.SIGNUP_CODE:
        return JSONResponse(
            status_code=403, content={"error": "Invalid or missing invite code."}
        )
    try:
        user_id = auth.create_user(creds.email, creds.password)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    resp = JSONResponse({"email": creds.email.strip().lower()})
    _set_session_cookie(resp, auth.create_session(user_id))
    return resp


@app.post("/auth/login")
def login(creds: Credentials):
    user_id = auth.verify_user(creds.email, creds.password)
    if not user_id:
        return JSONResponse(
            status_code=401, content={"error": "Wrong email or password."}
        )
    resp = JSONResponse({"email": creds.email.strip().lower()})
    _set_session_cookie(resp, auth.create_session(user_id))
    return resp


@app.post("/auth/logout")
def logout(request: Request):
    auth.delete_session(request.cookies.get(COOKIE))
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE, path="/")
    return resp


@app.get("/me")
def me(request: Request):
    user = current_user(request)
    if not user:
        return JSONResponse(status_code=401, content={"error": "Not authenticated."})
    return {"email": user["email"]}


class AskRequest(BaseModel):
    question: str
    lang: str = "en"


@app.post("/ask")
def ask(req: AskRequest, request: Request):
    if not current_user(request):
        return JSONResponse(status_code=401, content={"error": "Please sign in."})
    store = state.get("store")
    if store is None:
        return JSONResponse(status_code=503, content={"error": state.get("error", "Index not loaded.")})
    if not req.question.strip():
        return JSONResponse(status_code=400, content={"error": "Empty question."})

    # Retrieval + rerank happen up front (before the response starts streaming).
    try:
        chunks = retrieve(store, req.question)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})

    def ndjson():
        try:
            for event in generate.answer_stream(req.question, chunks, req.lang):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "error": str(exc)}) + "\n"

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")


@app.get("/")
def index(request: Request):
    if not current_user(request):
        return RedirectResponse("/login", status_code=302)
    return FileResponse(config.STATIC_DIR / "index.html")


@app.get("/login")
def login_page(request: Request):
    if current_user(request):
        return RedirectResponse("/", status_code=302)
    return FileResponse(config.STATIC_DIR / "auth.html")


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
