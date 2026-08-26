"""
api/server.py

GPI Knowledge Assistant — HTTP API.

A fully independent front end on top of web/session_manager.py (which
itself sits on top of the unmodified core/, retrieval/, memory/, llm/,
and storage/ packages). This does not import or alter main.py or
web/app.py — all three front ends (terminal, Streamlit, API) can run
concurrently against the same underlying engine and chat storage.

Run with:
    python api/server.py
    # or
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Once running, interactive docs are available at /docs.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.config import (
    ASSISTANT_NAME, COMPANY_FULL_NAME, CHARTS_OUTPUT_DIR, SIGNUP_DEPARTMENTS,
)
from retrieval.knowledge_index import index as knowledge_index
from web.session_manager import SessionRegistry
from auth import store as auth_store
from auth import permissions as auth_permissions

app = FastAPI(
    title=ASSISTANT_NAME,
    description=f"Knowledge assistant API for {COMPANY_FULL_NAME}.",
    version="1.0.0",
)

# Permissive CORS so a future app on a different origin can call this
# directly during development. Tighten allow_origins before exposing this
# publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One process-wide registry, mirroring how the Streamlit app keeps one
# per browser session — here there is one server process, so one registry
# is the equivalent shared state.
registry = SessionRegistry()

# Chart PNGs are written to a local directory (CHARTS_OUTPUT_DIR) by
# core/data_analyst.py -- a bare filesystem path like
# "charts_output/outliers_x.png" means nothing to a browser on another
# machine (e.g. reached through an ngrok tunnel from the company laptop).
# Mounting that directory here means a chart can be referenced as a plain
# fetchable URL ("/charts/outliers_x.png") instead -- see
# _chart_url_path() below, used everywhere a chart_path is returned.
os.makedirs(CHARTS_OUTPUT_DIR, exist_ok=True)
app.mount("/charts", StaticFiles(directory=CHARTS_OUTPUT_DIR), name="charts")

# The custom chat frontend (static/index.html + app.js) -- see
# RUNNING_REMOTELY.md. Not Streamlit: a single static page you can freely
# restyle, with no server-side script-rerun model to fight with, served
# directly by this same FastAPI process so there's nothing extra to run.
_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/app", StaticFiles(directory=_STATIC_DIR, html=True), name="frontend")


def _chart_url_path(chart_path: Optional[str]) -> Optional[str]:
    """Convert a local charts_output/<file>.png path into a URL this same
    server can actually serve to a remote browser. Returns None
    unchanged if there's no chart."""
    if not chart_path:
        return None
    return f"/charts/{os.path.basename(chart_path)}"


# ─── Auth ────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str
    designation: str = Field(..., description="Free-text job title/post.")
    department: str = Field(..., description=f"One of {SIGNUP_DEPARTMENTS}")


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    token: str
    user: Dict[str, Any]


def get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    FastAPI dependency: resolves the bearer token in the Authorization
    header to a user dict, or raises 401. Every endpoint that should
    require login (asking questions, uploading files, managing
    permissions) depends on this rather than reading the header itself,
    so there's exactly one place session validation happens.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")
    token = authorization.split(" ", 1)[1].strip()
    user = auth_store.resolve_session(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Session expired or invalid. Please log in again.")
    return user.public_dict()


def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[Dict[str, Any]]:
    """Same as get_current_user, but returns None instead of raising --
    for endpoints that work either way (e.g. /health)."""
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    user = auth_store.resolve_session(token)
    return user.public_dict() if user else None


@app.get("/auth/departments")
def get_departments() -> List[str]:
    """The signup department dropdown list -- see core/config.py's
    SIGNUP_DEPARTMENTS for where to edit it."""
    return SIGNUP_DEPARTMENTS


@app.post("/auth/register", response_model=AuthResponse)
def register(req: RegisterRequest) -> AuthResponse:
    try:
        user = auth_store.register(req.email, req.password, req.designation, req.department)
    except auth_store.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    token = auth_store.create_session(user.email)
    return AuthResponse(token=token, user=user.public_dict())


@app.post("/auth/login", response_model=AuthResponse)
def login(req: LoginRequest) -> AuthResponse:
    try:
        user = auth_store.login(req.email, req.password)
    except auth_store.AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    token = auth_store.create_session(user.email)
    return AuthResponse(token=token, user=user.public_dict())


@app.post("/auth/logout")
def logout(authorization: Optional[str] = Header(None)) -> Dict[str, str]:
    if authorization and authorization.lower().startswith("bearer "):
        auth_store.revoke_session(authorization.split(" ", 1)[1].strip())
    return {"status": "logged_out"}


@app.get("/auth/me")
def me(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    return user


# ─── File upload + access control ──────────────────────────────────────────

class SetPermissionRequest(BaseModel):
    allowed_departments: List[str] = Field(default_factory=list)
    allowed_users: List[str] = Field(default_factory=list)


class FileInfo(BaseModel):
    source_file: str
    uploaded_by: str
    allowed_departments: List[str]
    allowed_users: List[str]


_PDF_EXTS = {".pdf"}
_SHEET_EXTS = {".xlsx", ".xls", ".csv"}


def _ingest_uploaded_pdf() -> None:
    """Background task: incrementally (re-)index the PDF knowledge
    source after a new PDF is uploaded. Only new/changed files are
    actually re-processed (checksum-tracked) -- see
    retrieval/pdf_retriever.py -- so this is safe to call after every
    single upload without re-doing prior work."""
    try:
        from retrieval.pdf_retriever import pdf_retriever
        pdf_retriever.rebuild(force=False)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Background PDF ingestion failed")


@app.post("/files/upload", response_model=FileInfo)
def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    allowed_departments: str = Form("", description="Comma-separated department names, or empty for no restriction."),
    allowed_users: str = Form("", description="Comma-separated emails, or empty for no restriction."),
    user: Dict[str, Any] = Depends(get_current_user),
) -> FileInfo:
    """
    Upload a document (PDF) or spreadsheet (xlsx/xls/csv), and choose
    who besides the uploader and admins can see its content -- including
    through the chatbot (see auth/permissions.py, and
    core/engine.py's enforcement at retrieval time). Leaving both
    allowed_departments and allowed_users empty means the file is open
    to everyone, same as before this feature existed.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in _PDF_EXTS | _SHEET_EXTS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: PDF, XLSX, XLS, CSV.",
        )

    from core.config import PDF_KNOWLEDGE_RAW_DIR, DOCUMENTS_DIR
    target_dir = PDF_KNOWLEDGE_RAW_DIR if ext in _PDF_EXTS else os.path.join(DOCUMENTS_DIR, "raw")
    os.makedirs(target_dir, exist_ok=True)

    safe_name = os.path.basename(file.filename)
    dest_path = os.path.join(target_dir, safe_name)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    departments = [d.strip() for d in allowed_departments.split(",") if d.strip()]
    users = [u.strip() for u in allowed_users.split(",") if u.strip()]
    for d in departments:
        if d not in SIGNUP_DEPARTMENTS:
            raise HTTPException(status_code=400, detail=f"Unknown department: {d}")

    perm = auth_permissions.set_permission(
        source_file=safe_name,
        uploaded_by=user["email"],
        allowed_departments=departments,
        allowed_users=users,
    )

    if ext in _PDF_EXTS:
        # Runs after this request returns -- ingestion (chunking +
        # embedding) can take a while for a large PDF, no reason to
        # make the uploader's browser wait on it. Spreadsheets need no
        # separate ingestion step -- core/data_analyst.py's registry
        # picks up any file already sitting in documents/raw/ on its
        # next read, live.
        background_tasks.add_task(_ingest_uploaded_pdf)

    return FileInfo(**perm.__dict__)


@app.get("/files", response_model=List[FileInfo])
def list_files(user: Dict[str, Any] = Depends(get_current_user)) -> List[FileInfo]:
    """Every uploaded file's permission record. Any logged-in user can
    see WHO a file is restricted to (useful for figuring out who to ask
    for access) -- this lists metadata, not file content, so it doesn't
    leak anything the access-control check itself is meant to protect."""
    return [FileInfo(**p.__dict__) for p in auth_permissions.list_permissions()]


@app.post("/files/{source_file}/permissions", response_model=FileInfo)
def set_file_permissions(
    source_file: str, req: SetPermissionRequest, user: Dict[str, Any] = Depends(get_current_user)
) -> FileInfo:
    """Update a file's access list. Only the original uploader or an
    admin may change it."""
    existing = auth_permissions.get_permission(source_file)
    if existing and existing.uploaded_by != user["email"] and not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Only the uploader or an admin can change this file's access list.")
    for d in req.allowed_departments:
        if d not in SIGNUP_DEPARTMENTS:
            raise HTTPException(status_code=400, detail=f"Unknown department: {d}")
    perm = auth_permissions.set_permission(
        source_file=source_file,
        uploaded_by=existing.uploaded_by if existing else user["email"],
        allowed_departments=req.allowed_departments,
        allowed_users=req.allowed_users,
    )
    return FileInfo(**perm.__dict__)


# ─── Schemas ───────────────────────────────────────────────────────────────────

class ChatCreateRequest(BaseModel):
    name: Optional[str] = Field(None, description="Optional chat name; auto-generated if omitted.")


class ChatRenameRequest(BaseModel):
    new_name: str


class AskRequest(BaseModel):
    question: str
    chat_name: Optional[str] = Field(
        None, description="Chat to ask in. Defaults to the currently active chat."
    )


class AskResponse(BaseModel):
    answer: str
    route: str
    chat_name: str
    intent: Dict[str, Any]
    results: List[Dict[str, Any]]
    prompt_size: int
    retrieval_time: float
    llm_time: float
    total_time: float
    is_followup: bool
    error: Optional[str] = None
    chart_path: Optional[str] = None
    restricted: bool = False


class ChatInfo(BaseModel):
    name: str
    is_current: bool


class MessageItem(BaseModel):
    role: str
    content: str
    chart_path: Optional[str] = None


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_chat_name(chat_name: Optional[str]) -> str:
    name = chat_name or registry.current_chat()
    if name not in registry.list_chats():
        raise HTTPException(status_code=404, detail=f"Chat '{name}' does not exist.")
    return name


# ─── Health / status ───────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    """Liveness/readiness probe. Never touches the LLM, so this stays fast
    and reliable even if Ollama itself is down."""
    return {
        "status": "ok",
        "assistant": ASSISTANT_NAME,
        "indexed_sections": len(knowledge_index.items),
    }


# ─── Chat management ────────────────────────────────────────────────────────────

@app.get("/chats", response_model=List[ChatInfo])
def list_chats() -> List[ChatInfo]:
    current = registry.current_chat()
    return [ChatInfo(name=n, is_current=(n == current)) for n in registry.list_chats()]


@app.post("/chats", response_model=ChatInfo)
def create_chat(req: ChatCreateRequest) -> ChatInfo:
    name = registry.create_chat(req.name)
    return ChatInfo(name=name, is_current=True)


@app.post("/chats/{chat_name}/switch", response_model=ChatInfo)
def switch_chat(chat_name: str) -> ChatInfo:
    if not registry.switch_chat(chat_name):
        raise HTTPException(status_code=404, detail=f"Chat '{chat_name}' does not exist.")
    return ChatInfo(name=chat_name, is_current=True)


@app.post("/chats/{chat_name}/rename", response_model=ChatInfo)
def rename_chat(chat_name: str, req: ChatRenameRequest) -> ChatInfo:
    if not registry.rename_chat(chat_name, req.new_name):
        raise HTTPException(
            status_code=400,
            detail="Rename failed — source chat may not exist, or target name is invalid/taken.",
        )
    return ChatInfo(name=req.new_name, is_current=(registry.current_chat() == req.new_name))


@app.delete("/chats/{chat_name}")
def delete_chat(chat_name: str) -> Dict[str, str]:
    if not registry.delete_chat(chat_name):
        raise HTTPException(
            status_code=400,
            detail="Delete failed — chat may not exist, or it's the last remaining chat.",
        )
    return {"status": "deleted", "name": chat_name}


@app.post("/chats/{chat_name}/clear")
def clear_chat(chat_name: str) -> Dict[str, str]:
    _resolve_chat_name(chat_name)
    registry.clear_chat(chat_name)
    return {"status": "cleared", "name": chat_name}


@app.get("/chats/{chat_name}/messages", response_model=List[MessageItem])
def get_messages(chat_name: str) -> List[MessageItem]:
    name = _resolve_chat_name(chat_name)
    messages = []
    for m in registry.messages(name):
        m = dict(m)
        m["chart_path"] = _chart_url_path(m.get("chart_path"))
        messages.append(MessageItem(**m))
    return messages


# ─── Ask (blocking) ─────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, user: Dict[str, Any] = Depends(get_current_user)) -> AskResponse:
    """
    Run the full retrieval + answer pipeline for a single question and
    return the complete result. Use this for simple request/response
    integrations; use /ask/stream for token-by-token streaming.

    Requires login -- the logged-in user's department/email is what
    department-based file access control (auth/permissions.py) checks
    every retrieved result against.
    """
    name = _resolve_chat_name(req.chat_name)
    try:
        result = registry.ask(name, req.question, user=user)
    except Exception as exc:
        # The engine itself never raises for ordinary failures (it returns
        # an EngineResult with route="ERROR"); this catches only truly
        # unexpected failures so the API never 500s without explanation.
        raise HTTPException(status_code=500, detail=str(exc))

    return AskResponse(
        answer=result.answer,
        route=result.route,
        chat_name=name,
        intent=result.intent,
        results=result.results,
        prompt_size=result.prompt_size,
        retrieval_time=result.retrieval_time,
        llm_time=result.llm_time,
        total_time=result.total_time,
        is_followup=result.is_followup,
        error=result.error,
        chart_path=_chart_url_path(result.chart_path),
        restricted=result.restricted,
    )


# ─── Ask (streaming, Server-Sent Events) ───────────────────────────────────────

@app.post("/ask/stream")
def ask_stream(req: AskRequest, user: Dict[str, Any] = Depends(get_current_user)):
    """
    Stream the answer as Server-Sent Events. Event types mirror
    AssistantEngine.ask_stream(): "intent", "retrieval", "chunk", "done".
    The final "done" event's data is the full EngineResult, JSON-encoded.

    Requires login, same as /ask -- see get_current_user.
    """
    import json

    name = _resolve_chat_name(req.chat_name)
    engine = registry.engine_for(name)

    if name != registry.current_chat():
        registry.chat_manager.switch_chat(name)

    def _event_stream():
        accumulated_answer = ""
        for event_type, data in engine.ask_stream(req.question, user=user):
            if event_type == "chunk":
                accumulated_answer += data
                payload = json.dumps({"chunk": data})
            elif event_type == "intent":
                payload = json.dumps({"intent": data})
            elif event_type == "retrieval":
                payload = json.dumps({"result_count": len(data)})
            elif event_type == "done":
                result = data
                registry.chat_manager.add_message("user", req.question)
                registry.chat_manager.add_message("assistant", result.answer, chart_path=result.chart_path)
                payload = json.dumps({
                    "answer": result.answer,
                    "route": result.route,
                    "retrieval_time": result.retrieval_time,
                    "llm_time": result.llm_time,
                    "total_time": result.total_time,
                    "error": result.error,
                    "chart_path": _chart_url_path(result.chart_path),
                    "restricted": result.restricted,
                })
            elif event_type == "chart":
                payload = json.dumps({"chart_path": _chart_url_path(data)})
            else:
                continue
            yield f"event: {event_type}\ndata: {payload}\n\n"

    return StreamingResponse(_event_stream(), media_type="text/event-stream")


# ─── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)