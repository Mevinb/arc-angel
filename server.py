"""
FastAPI Server for Local Grok Chatbot.
Provides SSE streaming, session management, multi-tier memory inspection, and engine control.
"""
import re
import uuid
import json
import time
import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from typing import Optional, List, Dict
from urllib.parse import quote
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

from config import (
    WEB_HOST,
    WEB_PORT,
    DEFAULT_CTX_SIZE,
    MODELS,
    DEFAULT_MODEL_ID,
    ATTACHMENTS_DIR,
    MAX_ATTACHMENT_BYTES,
    MAX_ATTACHMENT_CHARS,
    PROVIDER_LOCAL,
    PROVIDER_OPENAI,
    OPENAI_MODELS,
    DEFAULT_OPENAI_MODEL_ID,
    get_active_model_id,
    set_active_model_id,
    get_active_model_config,
    get_active_provider,
    set_active_provider,
    get_active_openai_model_id,
    set_active_openai_model_id,
    get_active_engine_info,
    to_ui_model_id,
    parse_ui_model_id,
    describe_openai_model,
    get_openai_api_key,
    get_openai_key_source,
    get_openai_base_url,
    set_openai_credentials,
    clear_openai_credentials,
    mask_api_key,
    detect_recommended_gpu_layers
)
from memory import (
    db_session,
    UserDossier,
    EpisodicMemory,
    ConversationSummarizer,
    MemoryOrchestrator
)
from persona.prompts import GROK_MODES
from persona.chatml import format_chatml_prompt, estimate_tokens
from engine import (
    LlamaClient,
    OpenAIClient,
    ProviderRouter,
    LlamaProcessManager,
    WaylandController,
    DesktopAgent
)
from engine.vision import VisionEngine
from engine.task_orchestrator import TaskOrchestrator, detect_task_type
from engine.task_runner import TaskRunner
from engine.workspace_tools import WorkspaceManager
from services.db import init_platform_db
from services.agents import AgentService
from services.automations import AutomationService
from services.policies import PolicyService
from services.run_manager import RunManager
from services.mobile import MobileService
from services.safety import SafetyService
from api_v2 import router as api_v2_router

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize additive v2 schema and defaults before accepting requests.
    try:
        init_platform_db()
        PolicyService.ensure_defaults()
        AgentService.ensure_defaults()
    except Exception:
        logger.exception("Platform v2 initialization failed")
    # Auto-resume interrupted v2 runs and start the scheduler.  Both are
    # best-effort: a failure here must not prevent the legacy chat UI starting.
    try:
        await RunManager.get().recover()
    except Exception:
        logger.exception("Could not recover interrupted v2 runs")
    try:
        AutomationService.start_scheduler()
    except Exception:
        logger.exception("Could not start automation scheduler")
    # Auto-ensure the local llama-server is online in background (local provider only)
    if get_active_provider() == PROVIDER_LOCAL:
        asyncio.create_task(asyncio.to_thread(LlamaProcessManager.ensure_running))
    try:
        await TaskRunner.recover_on_startup()
    except Exception:
        pass
    yield
    try:
        await AutomationService.stop_scheduler()
    except Exception:
        pass
    try:
        await RunManager.get().shutdown()
    except Exception:
        pass
    try:
        runner = TaskRunner()
        if hasattr(runner, "shutdown"):
            await runner.shutdown()
    except Exception:
        pass

app = FastAPI(title="Local Grok Chatbot", version="2.0.0", lifespan=lifespan)
app.include_router(api_v2_router)

_PUBLIC_LOCK_PATHS = {
    "/unlock", "/favicon.ico", "/api/v2/health",
    "/api/v2/safety/unlock", "/api/v2/safety/lock-status",
}


@app.middleware("http")
async def app_lock_middleware(request: Request, call_next):
    """Require an unlock cookie for the local UI/API when the app lock is on.

    Mobile endpoints use their own per-device token and static assets must load
    for the unlock screen, so both remain reachable while locked.
    """
    path = request.url.path
    if (
        path in _PUBLIC_LOCK_PATHS
        or path.startswith("/static/")
        or path == "/mobile"
        or path == "/api/v2/mobile/pair"
    ):
        return await call_next(request)
    try:
        locked = SafetyService.lock_enabled()
    except Exception:
        logger.exception("Could not read app-lock state")
        locked = False
    if not locked:
        return await call_next(request)
    # Already-paired mobile devices keep working while the local UI is locked;
    # creating a new pairing or listing devices still requires an unlock cookie.
    if path.startswith("/api/v2/mobile/"):
        mobile_token = request.headers.get("X-Mobile-Token") or ""
        auth = request.headers.get("Authorization") or ""
        if not mobile_token and auth.lower().startswith("bearer "):
            mobile_token = auth[7:].strip()
        if mobile_token and MobileService.authenticate(mobile_token):
            return await call_next(request)
    token = request.cookies.get("grok_unlock") or request.headers.get("X-Unlock-Token")
    if SafetyService.check_unlock_token(token):
        return await call_next(request)
    if path.startswith("/api/"):
        return JSONResponse(
            status_code=423,
            content={"detail": "App locked. Open /unlock to enter your PIN."},
        )
    target = path + (f"?{request.url.query}" if request.url.query else "")
    return RedirectResponse(
        f"/unlock?next={quote(target, safe='/%?=&')}", status_code=303
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(status_code=404, content={"detail": str(exc)})

@app.exception_handler(PermissionError)
async def permission_error_handler(request: Request, exc: PermissionError):
    return JSONResponse(status_code=403, content={"detail": str(exc)})

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

UPLOADS_DIR = Path(__file__).resolve().parent / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

llama_client = LlamaClient()
openai_client = OpenAIClient()
llm_router = ProviderRouter(llama_client=llama_client, openai_client=openai_client)
desktop_agent = DesktopAgent(llm_client=llm_router)

# Cache for models reported live by the OpenAI API (avoids hammering /v1/models)
_live_openai_cache: Dict = {"ids": [], "ts": 0.0, "ok": False}


async def get_live_openai_model_ids(force: bool = False) -> List[str]:
    """Fetches chat-capable model ids from the OpenAI API with a short cache."""
    if not get_openai_api_key():
        return []
    now = time.time()
    ttl = 600 if _live_openai_cache["ok"] else 60
    if not force and (now - _live_openai_cache["ts"]) < ttl:
        return _live_openai_cache["ids"]

    result = await openai_client.fetch_models()
    ids: List[str] = []
    if result.get("ok"):
        for mid in result.get("models", []):
            low = mid.lower()
            if not re.match(r"^(gpt|chatgpt|o\d|codex)", low):
                continue
            if re.search(r"(embedding|tts|whisper|dall-e|image|audio|realtime|moderation|transcribe|sora)", low):
                continue
            ids.append(mid)
    _live_openai_cache.update(ids=ids, ts=now, ok=bool(result.get("ok")))
    return ids

class DesktopRunRequest(BaseModel):
    goal: str
    session_id: Optional[str] = None

# Request Models
class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Conversation"
    mode: Optional[str] = "classic"

class RenameSessionRequest(BaseModel):
    title: str

class ChatRequest(BaseModel):
    session_id: str
    message: str
    mode: Optional[str] = "classic"
    image_data: Optional[str] = None
    image_filename: Optional[str] = None
    attachment: Optional[Dict] = None
    regenerate: Optional[bool] = False
    edit_message_id: Optional[int] = None

class FactRequest(BaseModel):
    key: str
    value: str
    category: Optional[str] = "general"

class StartServerRequest(BaseModel):
    model_id: Optional[str] = None
    gpu_layers: Optional[int] = None
    ctx_size: Optional[int] = DEFAULT_CTX_SIZE

class SelectModelRequest(BaseModel):
    model_id: str
    gpu_layers: Optional[int] = None
    restart: bool = True

class OpenAIKeyRequest(BaseModel):
    api_key: str
    base_url: Optional[str] = None

class ManualActionRequest(BaseModel):
    type: str  # "click" | "key"
    x: Optional[int] = None
    y: Optional[int] = None
    key: Optional[str] = None

class DisplayModeRequest(BaseModel):
    isolated_display: bool

class TaskActionRequest(BaseModel):
    task_id: str



@app.api_route("/", methods=["GET", "HEAD"])
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.api_route("/platform", methods=["GET", "HEAD"])
async def platform():
    target = STATIC_DIR / "platform.html"
    if not target.exists():
        raise HTTPException(status_code=404, detail="Command Center is not installed")
    return FileResponse(target)


@app.api_route("/mobile", methods=["GET", "HEAD"])
async def mobile():
    target = STATIC_DIR / "mobile.html"
    if not target.exists():
        raise HTTPException(status_code=404, detail="Mobile companion is not installed")
    return FileResponse(target)


@app.api_route("/unlock", methods=["GET", "HEAD"])
async def unlock():
    target = STATIC_DIR / "unlock.html"
    if not target.exists():
        raise HTTPException(status_code=404, detail="Unlock screen is not installed")
    return FileResponse(target)

# System & Engine Status
@app.get("/api/models")
async def list_models():
    active_info = get_active_engine_info()
    provider = active_info["provider"]
    has_openai_key = bool(get_openai_api_key())

    res = []
    # Local llama.cpp models
    for m_id, cfg in MODELS.items():
        res.append({
            "id": m_id,
            "name": cfg["name"],
            "badge": cfg.get("badge", ""),
            "has_vision": cfg.get("has_vision", False),
            "exists": cfg["path"].exists(),
            "mmproj_exists": bool(cfg.get("mmproj") and cfg["mmproj"].exists()),
            "is_active": provider == PROVIDER_LOCAL and m_id == active_info["model_id"],
            "provider": PROVIDER_LOCAL,
            "locked": False,
            "reasoning": False,
            "default_ngl": cfg.get("default_ngl", 24),
            "total_layers": cfg.get("total_layers", 32),
            "recommended_ngl": detect_recommended_gpu_layers(m_id)
        })

    # OpenAI ChatGPT models (catalog + anything extra available on the account)
    live_ids = await get_live_openai_model_ids() if has_openai_key else []
    catalog_ids = list(OPENAI_MODELS.keys()) + [i for i in live_ids if i not in OPENAI_MODELS]
    for mid in catalog_ids:
        desc = describe_openai_model(mid)
        res.append({
            "id": to_ui_model_id(PROVIDER_OPENAI, mid),
            "name": desc["name"],
            "badge": desc.get("badge", "ChatGPT"),
            "has_vision": desc.get("has_vision", False),
            "exists": True,
            "mmproj_exists": False,
            "is_active": provider == PROVIDER_OPENAI and mid == active_info["model_id"],
            "provider": PROVIDER_OPENAI,
            "locked": not has_openai_key,
            "reasoning": desc.get("reasoning", False),
            "default_ngl": 0,
            "total_layers": 0,
            "recommended_ngl": 0
        })
    return res

@app.post("/api/models/select")
async def select_model(req: SelectModelRequest):
    provider, raw_id = parse_ui_model_id(req.model_id)

    if provider == PROVIDER_OPENAI:
        if not get_openai_api_key():
            raise HTTPException(
                status_code=400,
                detail="OpenAI API key not configured. Click the 🔑 key button to add your ChatGPT API key first."
            )
        if not raw_id:
            raise HTTPException(status_code=400, detail="Empty OpenAI model id")
        set_active_openai_model_id(raw_id)
        set_active_provider(PROVIDER_OPENAI)
        desc = describe_openai_model(raw_id)
        return {
            "status": "selected",
            "provider": PROVIDER_OPENAI,
            "model_id": to_ui_model_id(PROVIDER_OPENAI, raw_id),
            "model_name": desc["name"]
        }

    if req.model_id not in MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {req.model_id}")
    set_active_model_id(req.model_id)
    if req.restart:
        res = LlamaProcessManager.switch_model(model_id=req.model_id, gpu_layers=req.gpu_layers)
        return {"status": "switched", "provider": PROVIDER_LOCAL, "model_id": req.model_id, "process": res}
    return {"status": "selected", "provider": PROVIDER_LOCAL, "model_id": req.model_id}

@app.get("/api/status")
async def get_status():
    provider = get_active_provider()
    engine_info = get_active_engine_info()
    facts = UserDossier.get_all_facts()
    with db_session() as conn:
        cursor = conn.execute("SELECT COUNT(*) as cnt FROM episodic_memories")
        episodic_count = cursor.fetchone()["cnt"]

    if provider == PROVIDER_OPENAI:
        is_ready = bool(get_openai_api_key())
        pm_status = {
            "running": False,
            "pid": None,
            "active_model_id": engine_info["model_id"],
            "active_model_name": engine_info["name"],
            "model": engine_info["model_id"],
            "has_vision": engine_info["has_vision"],
            "log_file": None
        }
    else:
        is_ready = await llama_client.is_ready()
        pm_status = LlamaProcessManager.get_status()

    return {
        "engine_ready": is_ready,
        "provider": provider,
        "process_status": pm_status,
        "active_model_id": engine_info["ui_model_id"],
        "model_name": engine_info["name"],
        "has_vision": engine_info["has_vision"],
        "reasoning": engine_info.get("reasoning", False),
        "dossier_facts_count": len(facts),
        "episodic_memories_count": episodic_count,
        "has_openai_key": bool(get_openai_api_key()),
        "openai_key_source": get_openai_key_source(),
        "available_modes": list(GROK_MODES.keys())
    }

@app.post("/api/server/start")
async def start_server(req: StartServerRequest):
    if get_active_provider() == PROVIDER_OPENAI:
        return {
            "status": "info",
            "provider": PROVIDER_OPENAI,
            "message": "ChatGPT cloud provider is active — no local llama-server is needed."
        }
    res = LlamaProcessManager.start(model_id=req.model_id, gpu_layers=req.gpu_layers, ctx_size=req.ctx_size)
    return res

@app.post("/api/server/stop")
async def stop_server():
    if get_active_provider() == PROVIDER_OPENAI:
        return {
            "status": "info",
            "provider": PROVIDER_OPENAI,
            "message": "ChatGPT cloud provider is active — nothing to stop locally."
        }
    return LlamaProcessManager.stop()

# ---------------------------------------------------------------------------
# OpenAI (ChatGPT) API key management
# ---------------------------------------------------------------------------
@app.get("/api/openai/status")
async def openai_status():
    key = get_openai_api_key()
    active_model = get_active_openai_model_id()

    live_ids = await get_live_openai_model_ids() if key else []
    catalog_ids = list(OPENAI_MODELS.keys()) + [i for i in live_ids if i not in OPENAI_MODELS]

    models = []
    for mid in catalog_ids:
        desc = describe_openai_model(mid)
        models.append({
            "id": mid,
            "name": desc["name"],
            "badge": desc.get("badge", "ChatGPT"),
            "has_vision": desc.get("has_vision", False),
            "reasoning": desc.get("reasoning", False),
            "is_active": mid == active_model
        })

    return {
        "has_key": bool(key),
        "key_source": get_openai_key_source(),
        "key_hint": mask_api_key(key) if key else None,
        "base_url": get_openai_base_url(),
        "provider": get_active_provider(),
        "active_model": active_model,
        "default_model": DEFAULT_OPENAI_MODEL_ID,
        "models": models
    }

@app.post("/api/openai/key")
async def save_openai_key(req: OpenAIKeyRequest):
    api_key = (req.api_key or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="API key cannot be empty")

    base_url = (req.base_url or "").strip().rstrip("/") or None

    # Validate the key against the API before persisting it
    probe = OpenAIClient(base_url=base_url, api_key=api_key)
    result = await probe.fetch_models()
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "OpenAI API key validation failed")

    set_openai_credentials(api_key, base_url)

    # Reset live model cache and switch to the cloud provider immediately
    _live_openai_cache.update(ids=[], ts=0.0, ok=False)
    set_active_provider(PROVIDER_OPENAI)
    set_active_openai_model_id(get_active_openai_model_id())

    return {
        "status": "saved",
        "validated": True,
        "models_available": len(result.get("models", [])),
        "provider": PROVIDER_OPENAI,
        "active_model": get_active_openai_model_id()
    }

@app.delete("/api/openai/key")
async def delete_openai_key():
    clear_openai_credentials()
    _live_openai_cache.update(ids=[], ts=0.0, ok=False)

    still_has_key = bool(get_openai_api_key())
    reverts_to_local = False
    if not still_has_key and get_active_provider() == PROVIDER_OPENAI:
        set_active_provider(PROVIDER_LOCAL)
        reverts_to_local = True

    if still_has_key:
        message = "Key is still provided via the OPENAI_API_KEY environment variable; remove it there."
    else:
        message = "OpenAI key removed."
        if reverts_to_local:
            message += " Switched back to the local model."

    return {
        "status": "deleted",
        "has_key": still_has_key,
        "key_source": get_openai_key_source(),
        "provider": get_active_provider(),
        "message": message
    }

# Session Management
@app.get("/api/sessions")
async def list_sessions():
    with db_session() as conn:
        cursor = conn.execute("SELECT id, title, mode, created_at, updated_at FROM sessions ORDER BY updated_at DESC")
        return [dict(row) for row in cursor.fetchall()]

@app.post("/api/sessions")
async def create_session(req: CreateSessionRequest):
    sess_id = str(uuid.uuid4())
    with db_session() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, mode, created_at, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (sess_id, req.title, req.mode)
        )
    return {"id": sess_id, "title": req.title, "mode": req.mode}

@app.patch("/api/sessions/{session_id}")
async def rename_session(session_id: str, req: RenameSessionRequest):
    with db_session() as conn:
        cur = conn.execute(
            "UPDATE sessions SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (req.title, session_id)
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "renamed", "id": session_id, "title": req.title}

@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    with db_session() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM session_summaries WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    return {"status": "deleted", "id": session_id}

@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    with db_session() as conn:
        cursor = conn.execute(
            "SELECT id, role, content, tokens, image_url, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

@app.delete("/api/sessions/{session_id}/messages/{message_id}")
async def delete_message(session_id: str, message_id: int):
    """Deletes a single message (used by edit/regenerate flows)."""
    with db_session() as conn:
        cur = conn.execute(
            "DELETE FROM messages WHERE id = ? AND session_id = ?",
            (message_id, session_id)
        )
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="Message not found")
    return {"status": "deleted", "id": message_id}

# Memory Inspector Endpoints
@app.get("/api/memory/dossier")
async def get_dossier():
    return UserDossier.get_all_facts()

@app.post("/api/memory/dossier")
async def set_dossier_fact(req: FactRequest):
    UserDossier.set_fact(req.key, req.value, category=req.category)
    return {"status": "saved", "key": req.key, "value": req.value}

@app.delete("/api/memory/dossier/{key}")
async def delete_dossier_fact(key: str):
    UserDossier.delete_fact(key)
    return {"status": "deleted", "key": key}

@app.get("/api/memory/episodic")
async def search_episodic(q: Optional[str] = None):
    if q:
        return EpisodicMemory.search_memories(q, limit=10)
    with db_session() as conn:
        cursor = conn.execute("SELECT id, session_id, content, summary, tags, created_at FROM episodic_memories ORDER BY id DESC LIMIT 15")
        return [dict(row) for row in cursor.fetchall()]

# Desktop Automation Endpoints (Mutter Wayland GNOME 46)
@app.post("/api/desktop/run")
async def run_desktop(req: DesktopRunRequest):
    async def event_generator():
        async for ev in desktop_agent.execute_task(req.goal):
            yield f"data: {json.dumps(ev)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/desktop/stop")
async def stop_desktop():
    desktop_agent.abort()
    return {"status": "stopped"}

@app.get("/api/desktop/shot")
async def get_desktop_shot(path: str):
    p = Path(path)
    if not p.exists() or not path.startswith("/tmp/grok_shots"):
        raise HTTPException(status_code=404, detail="Screenshot not found")
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return FileResponse(p, media_type="image/png", headers=headers)

@app.get("/api/desktop/status")
async def get_desktop_status():
    from config import get_pilot_isolated_display
    from engine.virtual_display import VirtualDisplayManager
    avail, reason = VirtualDisplayManager.is_available()
    try:
        bounds = desktop_agent.controller.get_desktop_bounds()
        b_dict = {"x": bounds[0], "y": bounds[1], "width": bounds[2], "height": bounds[3]}
    except Exception:
        b_dict = None
    return {
        "is_running": desktop_agent.is_running,
        "current_task": desktop_agent.current_task,
        "bounds": b_dict,
        "isolated_display": get_pilot_isolated_display(),
        "sandbox_available": avail,
        "sandbox_reason": reason
    }

@app.post("/api/desktop/manual_action")
async def desktop_manual_action(req: ManualActionRequest):
    """Executes a direct manual click or key press from the live preview viewer."""
    try:
        ctrl = desktop_agent.controller
        if req.type == "click" and req.x is not None and req.y is not None:
            ctrl.click(req.x, req.y)
        elif req.type == "key" and req.key:
            ctrl.press_key(req.key)
        else:
            raise HTTPException(status_code=400, detail="Invalid action parameters")

        time.sleep(0.3)
        now_ms = int(time.time() * 1000)
        shot_path = ctrl.capture_screen()
        return {
            "status": "ok",
            "screenshot_url": f"/api/desktop/shot?path={shot_path}&t={now_ms}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/desktop/display_mode")
async def get_desktop_display_mode():
    from config import get_pilot_isolated_display
    from engine.virtual_display import VirtualDisplayManager
    avail, reason = VirtualDisplayManager.is_available()
    return {
        "isolated_display": get_pilot_isolated_display(),
        "available": avail,
        "reason": reason
    }

@app.post("/api/desktop/display_mode")
async def set_desktop_display_mode(req: DisplayModeRequest):
    from config import set_pilot_isolated_display
    from engine.virtual_display import VirtualDisplayManager
    set_pilot_isolated_display(req.isolated_display)
    avail, reason = VirtualDisplayManager.is_available()
    return {
        "isolated_display": req.isolated_display,
        "available": avail,
        "reason": reason
    }

# Attachments (chat document/code uploads)
def _extract_attachment_text(path: Path, filename: str) -> str:
    """Best-effort plain-text extraction from an uploaded attachment."""
    suffix = Path(filename).suffix.lower()
    try:
        if suffix in (".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".jsonl",
                      ".yaml", ".yml", ".toml", ".csv", ".ini", ".cfg", ".html", ".htm",
                      ".css", ".sh", ".c", ".cpp", ".h", ".java", ".go", ".rs", ".sql", ".log"):
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".pdf":
            try:
                import pypdf  # optional dependency
                reader = pypdf.PdfReader(str(path))
                return "\n".join((page.extract_text() or "") for page in reader.pages)
            except Exception as e:
                return f"[PDF text extraction unavailable: {e}]"
    except Exception as e:
        return f"[Could not read attachment: {e}]"
    return f"[Binary attachment '{filename}' stored; no text extraction available for this type.]"

@app.post("/api/attachments")
async def upload_attachment(request: Request):
    """Accepts raw file bytes. Filename is passed via the X-Filename header."""
    raw = await request.body()
    filename = request.headers.get("x-filename", "attachment")
    if len(raw) > MAX_ATTACHMENT_BYTES:
        raise HTTPException(status_code=413, detail=f"Attachment exceeds {MAX_ATTACHMENT_BYTES} bytes")
    safe_name = re.sub(r"[^\w.\-]+", "_", filename or "attachment")
    att_id = uuid.uuid4().hex
    dest = ATTACHMENTS_DIR / f"{att_id}_{safe_name}"
    dest.write_bytes(raw)
    text = _extract_attachment_text(dest, safe_name)
    chars = len(text)
    return {
        "id": att_id,
        "url": f"/api/attachments/{dest.name}",
        "name": safe_name,
        "size": len(raw),
        "extension": Path(safe_name).suffix.lower(),
        "chars": chars,
        "preview": text[:200],
    }

@app.get("/api/attachments/{stored_name}")
async def get_attachment(stored_name: str):
    target = (ATTACHMENTS_DIR / stored_name).resolve()
    if not str(target).startswith(str(ATTACHMENTS_DIR.resolve())) or not target.is_file():
        raise HTTPException(status_code=404, detail="Attachment not found")
    return FileResponse(str(target), filename=stored_name)

# Hierarchical Long-Horizon Orchestrator Endpoints
@app.get("/api/orchestrator/tasks")
async def list_orchestrator_tasks(limit: int = 20):
    return TaskOrchestrator.list_tasks(limit=limit)

@app.get("/api/orchestrator/tasks/{task_id}")
async def get_orchestrator_task(task_id: str):
    t = TaskOrchestrator.get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    return t

@app.post("/api/orchestrator/tasks/{task_id}/start")
async def start_orchestrator_task(task_id: str):
    runner = TaskRunner()
    ok = await runner.start(task_id)
    if not ok:
        raise HTTPException(status_code=409, detail="Task cannot be started (not found or already terminal)")
    return {"status": "started", "task_id": task_id}

@app.post("/api/orchestrator/tasks/{task_id}/retry")
async def retry_orchestrator_task(task_id: str):
    runner = TaskRunner()
    ok = await runner.retry(task_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"status": "retrying", "task_id": task_id}

@app.get("/api/orchestrator/tasks/{task_id}/events")
async def get_orchestrator_events(task_id: str, since_id: int = 0, limit: int = 500):
    return {"task_id": task_id, "events": TaskOrchestrator.get_events(task_id, since_id=since_id, limit=limit)}

@app.get("/api/orchestrator/tasks/{task_id}/events/stream")
async def stream_orchestrator_events(task_id: str, since_id: int = 0):
    t = TaskOrchestrator.get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        runner = TaskRunner()
        async for ev in runner.subscribe(task_id, since_id=since_id):
            yield f"data: {json.dumps(ev, default=str)}\n\n"
            if ev.get("type") == "task_update" and ev.get("status") in (
                "completed", "partially_completed", "failed", "aborted", "interrupted"
            ):
                break

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/orchestrator/tasks/{task_id}/files")
async def list_orchestrator_files(task_id: str, rel_path: str = ""):
    t = TaskOrchestrator.get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    ws = WorkspaceManager(t["workspace_dir"])
    try:
        entries = ws.list_files(rel_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unsafe or invalid path: {e}")
    return {"task_id": task_id, "rel_path": rel_path, "entries": entries}

@app.get("/api/orchestrator/tasks/{task_id}/file")
async def get_orchestrator_file(task_id: str, rel_path: str):
    t = TaskOrchestrator.get_task(task_id)
    if not t:
        raise HTTPException(status_code=404, detail="Task not found")
    ws_dir = Path(t["workspace_dir"]).resolve()
    target = (ws_dir / rel_path).resolve()
    if not str(target).startswith(str(ws_dir)) or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(target), filename=target.name)

@app.post("/api/orchestrator/pause")
async def pause_orchestrator_task(req: TaskActionRequest):
    await TaskRunner().pause(req.task_id)
    return {"status": "paused", "task_id": req.task_id}

@app.post("/api/orchestrator/resume")
async def resume_orchestrator_task(req: TaskActionRequest):
    await TaskRunner().resume(req.task_id)
    return {"status": "resumed", "task_id": req.task_id}

@app.post("/api/orchestrator/abort")
async def abort_orchestrator_task(req: TaskActionRequest):
    await TaskRunner().abort(req.task_id)
    return {"status": "aborted", "task_id": req.task_id}


@app.delete("/api/memory/episodic/{mem_id}")
async def delete_episodic(mem_id: int):
    with db_session() as conn:
        conn.execute("DELETE FROM episodic_memories WHERE id = ?", (mem_id,))
    return {"status": "deleted", "id": mem_id}

@app.get("/api/memory/summary/{session_id}")
async def get_summary(session_id: str):
    summary = ConversationSummarizer.get_summary(session_id)
    return {"session_id": session_id, "summary": summary or ""}

def is_desktop_intent(text: str) -> bool:
    """Detects whether a user message expresses desktop or app automation intent."""
    t = text.strip().lower()
    if t.startswith("/desktop"):
        return True
    patterns = [
        r"\b(?:open|launch|start|run)\s+(?:brave|chrome|google-chrome|browser|files|nautilus|terminal|calculator|calc|settings|code|vscode|antigravity|antigravity-ide|ide|spotify|discord)\b",
        r"\b(?:search|look up|google)\s+.*?\s+(?:in|on|with|using)\s+(?:brave|chrome|browser)\b",
        r"\b(?:in|on|using)\s+(?:brave|chrome|browser)\s+search\b",
        r"\bclick\s+(?:on\s+)?(?:the\s+)?[\w\s]+\b",
        r"\b(?:use|move)\s+(?:the\s+)?cursor\b",
    ]
    for p in patterns:
        if re.search(p, t):
            return True
    return False

# Chat Streaming Endpoint
@app.post("/api/chat/stream")
async def stream_chat(req: ChatRequest, background_tasks: BackgroundTasks):
    session_id = req.session_id
    user_msg = (req.message or "").strip()
    mode = req.mode or "classic"

    if not user_msg and not req.image_data and not req.regenerate and not req.edit_message_id:
        raise HTTPException(status_code=400, detail="Message or image cannot be empty")
    if not user_msg and not req.regenerate:
        user_msg = "Analyze and describe this image without filters."

    # Vision Analysis Pipeline
    image_url = None
    image_analysis_text = ""
    engine_info = get_active_engine_info()
    provider = engine_info["provider"]
    model_has_native_vision = engine_info["has_vision"]

    saved_path: Optional[Path] = None
    image_url = None

    if req.image_data:
        try:
            ext = "png"
            data_str = req.image_data
            if "," in data_str:
                header, encoded = data_str.split(",", 1)
                if "jpeg" in header or "jpg" in header:
                    ext = "jpg"
                elif "webp" in header:
                    ext = "webp"
            else:
                encoded = data_str

            raw_bytes = base64.b64decode(encoded)
            file_name = f"{uuid.uuid4().hex[:12]}.{ext}"
            saved_path = UPLOADS_DIR / file_name
            with open(saved_path, "wb") as f:
                f.write(raw_bytes)
            image_url = f"/uploads/{file_name}"

            # Only run CPU BLIP/OCR if model does not have native vision
            if not model_has_native_vision:
                analysis = VisionEngine.analyze(str(saved_path), user_prompt=user_msg)
                image_analysis_text = analysis.get("context_block", "")
        except Exception as img_err:
            print(f"[!] Vision analysis error: {img_err}")
            if not model_has_native_vision:
                image_analysis_text = f"[Attached image uploaded: {img_err}]"

    # Attachment (document/code) text extraction + prompt injection
    attachment_meta = None
    attachment_text = ""
    if req.attachment and req.attachment.get("id"):
        att_name = req.attachment.get("name", "attachment")
        stored = None
        att_id = str(req.attachment["id"])
        for cand in ATTACHMENTS_DIR.glob(f"{att_id}_*"):
            stored = cand
            break
        if stored and stored.is_file():
            raw_text = _extract_attachment_text(stored, att_name)
            attachment_text = raw_text[:MAX_ATTACHMENT_CHARS]
            attachment_meta = {
                "url": f"/api/attachments/{stored.name}",
                "name": att_name,
                "type": Path(att_name).suffix.lower().lstrip("."),
                "size": stored.stat().st_size,
            }

    # Regenerate / edit flows: drop the previous assistant turn (and the edited user msg)
    if req.regenerate or req.edit_message_id:
        with db_session() as conn:
            if req.edit_message_id:
                conn.execute(
                    "DELETE FROM messages WHERE session_id = ? AND id >= ?",
                    (session_id, req.edit_message_id)
                )
                user_message_id = None
            else:
                row = conn.execute(
                    "SELECT id, content, image_url FROM messages WHERE session_id = ? AND role = 'user' ORDER BY id DESC LIMIT 1",
                    (session_id,)
                ).fetchone()
                if row:
                    user_msg = row["content"]
                    if not image_url:
                        image_url = row["image_url"]
                    user_message_id = row["id"]
                    conn.execute(
                        "DELETE FROM messages WHERE session_id = ? AND role = 'assistant' AND id > ?",
                        (session_id, user_message_id)
                    )

    # Ensure session exists or update title if it's default
    clean_title = user_msg.split("\n")[0][:36].strip()
    with db_session() as conn:
        cursor = conn.execute("SELECT id, title FROM sessions WHERE id = ?", (session_id,))
        existing_sess = cursor.fetchone()
        if not existing_sess:
            conn.execute(
                "INSERT INTO sessions (id, title, mode) VALUES (?, ?, ?)",
                (session_id, clean_title, mode)
            )
        elif existing_sess["title"] in ["New Conversation", "New Chat"] and len(clean_title) > 0:
            conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (clean_title, session_id))

    # 1. Store user message in DB with image_url / attachment
    if req.edit_message_id:
        # Edited user message was already removed; re-insert the new version
        with db_session() as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, image_url, attachment_url, attachment_name, attachment_type) VALUES (?, 'user', ?, ?, ?, ?, ?)",
                (session_id, user_msg, image_url,
                 attachment_meta["url"] if attachment_meta else None,
                 attachment_meta["name"] if attachment_meta else None,
                 attachment_meta["type"] if attachment_meta else None)
            )
            user_message_id = cur.lastrowid
    elif not req.regenerate:
        user_message_id = None
        with db_session() as conn:
            cur = conn.execute(
                "INSERT INTO messages (session_id, role, content, image_url, attachment_url, attachment_name, attachment_type) VALUES (?, 'user', ?, ?, ?, ?, ?)",
                (session_id, user_msg, image_url,
                 attachment_meta["url"] if attachment_meta else None,
                 attachment_meta["name"] if attachment_meta else None,
                 attachment_meta["type"] if attachment_meta else None)
            )
            user_message_id = cur.lastrowid
            # Update session timestamp
            conn.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))

    # 2. Assemble prompt using Multi-Tier Memory Orchestrator
    prompt_extras = []
    if image_analysis_text:
        prompt_extras.append(image_analysis_text)
    if attachment_text:
        prompt_extras.append(f"[ATTACHED FILE: {attachment_meta['name'] if attachment_meta else 'attachment'}]\n```\n{attachment_text}\n```")
    effective_user_msg = ("\n\n".join(prompt_extras) + f"\n\nUser Question/Request: {user_msg}") if prompt_extras else user_msg

    # Structured chat messages (used by OpenAI) + ChatML string (llama.cpp raw endpoint)
    chat_messages, budget = MemoryOrchestrator.assemble_messages(
        session_id=session_id,
        current_user_message=effective_user_msg,
        mode=mode,
        exclude_message_id=user_message_id
    )
    if provider == PROVIDER_LOCAL:
        chatml_prompt = format_chatml_prompt(chat_messages, append_assistant_header=True)
        budget["total_estimated_tokens"] = estimate_tokens(chatml_prompt)
    else:
        chatml_prompt = None

    # Sampling parameters tuned to persona
    sampling_params = GROK_MODES.get(mode, {}).get("sampling", {
        "temperature": 0.80,
        "top_p": 0.90,
        "min_p": 0.05,
        "presence_penalty": 0.20,
        "frequency_penalty": 0.10,
        "repeat_penalty": 1.08
    })

    # Reasoning-effort hint for GPT-5.x / o-series models
    reasoning_effort = None
    if provider == PROVIDER_OPENAI:
        reasoning_effort = {"deep_truth": "high", "desktop": "low"}.get(mode, "medium")

    async def event_generator():
        # First send memory budget & telemetry info (including image_url)
        yield f"data: {json.dumps({'type': 'telemetry', 'budget': budget, 'image_url': image_url})}\n\n"

        # Check if user message requires Hierarchical Long-Horizon Task Orchestration
        is_lh, task_type = detect_task_type(user_msg, has_reference_image=bool(saved_path))
        if is_lh:
            task_type_label = {
                "dataset_builder": "Reference-Image Dataset Builder",
                "software_project": "End-to-End Software Project Builder",
                "general": "Hierarchical Task Orchestrator"
            }.get(task_type, "Autonomous Task Orchestrator")

            orchestrator = TaskOrchestrator(
                llm_client=openai_client if provider == PROVIDER_OPENAI else llama_client,
                controller=desktop_agent.controller
            )
            created_task = TaskOrchestrator.create_task(
                goal=user_msg,
                task_type=task_type,
                session_id=session_id,
                reference_image_path=str(saved_path) if saved_path else None
            )
            task_id = created_task["id"]
            ws_path = created_task["workspace_dir"]

            header_msg = f"🏗️ **{task_type_label} Active**\nTask ID: `{task_id}` • Workspace: `{ws_path}`\nObjective: *{user_msg}*\n\n"
            yield f"data: {json.dumps({'type': 'token', 'token': header_msg})}\n\n"
            summary_parts = [header_msg]

            last_shot_url = None
            try:
                runner = TaskRunner()
                await runner.start(task_id)
                async for ev in runner.subscribe(task_id, since_id=0):
                    ev_type = ev.get("type")
                    if ev_type == "task_update":
                        yield f"data: {json.dumps(ev)}\n\n"
                        if ev.get("message"):
                            msg_line = f"📋 *{ev['message']}*\n\n"
                            summary_parts.append(msg_line)
                            yield f"data: {json.dumps({'type': 'token', 'token': msg_line})}\n\n"
                    elif ev_type == "subgoal_update":
                        yield f"data: {json.dumps(ev)}\n\n"
                        sg_title = ev.get("title", "") or (ev.get("subgoal") or {}).get("title", "")
                        sg_idx = ev.get("subgoal_idx", 0)
                        sg_status = ev.get("status", "") or (ev.get("subgoal") or {}).get("status", "")
                        if sg_status == "running":
                            sg_line = f"🔄 **Sub-goal {sg_idx}**: *{sg_title}* [In Progress]\n"
                            summary_parts.append(sg_line)
                            yield f"data: {json.dumps({'type': 'token', 'token': sg_line})}\n\n"
                        elif sg_status == "completed":
                            sg_line = f"✅ **Sub-goal {sg_idx} Completed**: *{sg_title}*\n"
                            summary_parts.append(sg_line)
                            yield f"data: {json.dumps({'type': 'token', 'token': sg_line})}\n\n"
                    elif ev_type == "desktop_step":
                        if ev.get("screenshot_url"):
                            last_shot_url = ev["screenshot_url"]
                        yield f"data: {json.dumps(ev)}\n\n"
                    elif ev_type == "error":
                        yield f"data: {json.dumps(ev)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                return

            # Persist assistant summary if not already persisted by runner
            try:
                with db_session() as conn:
                    cur = conn.execute("SELECT id FROM messages WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,))
                    if not cur.fetchone():
                        assistant_text = "".join(summary_parts)
                        conn.execute(
                            "INSERT INTO messages (session_id, role, content, task_id) VALUES (?, 'assistant', ?, ?)",
                            (session_id, assistant_text, task_id),
                        )
            except Exception:
                pass

            final_payload = {"type": "done", "full_response": "".join(summary_parts)}
            if last_shot_url:
                final_payload["image_url"] = last_shot_url
            yield f"data: {json.dumps(final_payload)}\n\n"
            return

        # Desktop Agent execution mode (Mutter Wayland GNOME 46)
        if mode == "desktop" or is_desktop_intent(user_msg):
            goal_text = user_msg[8:].strip() if user_msg.lower().startswith("/desktop") else user_msg
            yield f"data: {json.dumps({'type': 'token', 'token': f'🖱️ **Desktop Pilot Active (Mutter Wayland)**\nTarget Goal: *{goal_text}*\n\n'})}\n\n"
            summary_parts = [f"🖱️ **Desktop Pilot Active (Mutter Wayland)**\nTarget Goal: *{goal_text}*\n\n"]
            last_shot_url = None
            async for ev in desktop_agent.execute_task(goal_text):
                step_num = ev.get("step", 0)
                thought = ev.get("thought", "")
                act = ev.get("action", "")
                det = ev.get("detail", "")
                shot_url = ev.get("screenshot_url")
                if shot_url:
                    last_shot_url = shot_url

                yield f"data: {json.dumps({'type': 'desktop_step', 'step': step_num, 'thought': thought, 'action': act, 'detail': det, 'screenshot_url': shot_url, 'status': ev.get('status', 'running')})}\n\n"

                if step_num == 0:
                    step_str = f"🚀 *{det}*\n\n"
                elif thought:
                    step_str = f"**Step {step_num}**\n- 💭 *{thought}*\n- ⚡ `{act}`: {det.split('→')[-1].strip()}\n\n"
                else:
                    step_str = f"**Step {step_num}**\n- ⚡ `{act}`: {det}\n\n"
                summary_parts.append(step_str)
                yield f"data: {json.dumps({'type': 'token', 'token': step_str})}\n\n"

                if ev.get("status") in ("completed", "aborted", "max_steps", "error"):
                    break

            if last_shot_url:
                final_link = f"\n[View Final Desktop Screen State]({last_shot_url})\n"
                summary_parts.append(final_link)
                yield f"data: {json.dumps({'type': 'token', 'token': final_link})}\n\n"

            assistant_text = "".join(summary_parts).strip()
            with db_session() as conn:
                conn.execute(
                    "INSERT INTO messages (session_id, role, content, tokens) VALUES (?, 'assistant', ?, ?)",
                    (session_id, assistant_text, len(summary_parts))
                )
            yield f"data: {json.dumps({'type': 'done', 'full_response': assistant_text, 'image_url': last_shot_url})}\n\n"
            return

        # Classic chat / ChatGPT mode
        full_response = []
        try:
            async for chunk in llm_router.stream_completion(
                prompt=chatml_prompt,
                messages=chat_messages,
                image_data=req.image_data,
                temperature=sampling_params.get("temperature", 0.80),
                top_p=sampling_params.get("top_p", 0.90),
                min_p=sampling_params.get("min_p", 0.05),
                repeat_penalty=sampling_params.get("repeat_penalty", 1.08),
                presence_penalty=sampling_params.get("presence_penalty", 0.20),
                frequency_penalty=sampling_params.get("frequency_penalty", 0.10),
                max_tokens=2500 if provider == PROVIDER_OPENAI else 1500,
                reasoning_effort=reasoning_effort
            ):
                if "error" in chunk:
                    yield f"data: {json.dumps({'type': 'error', 'message': chunk['error']})}\n\n"
                    return

                token = chunk.get("token", "")
                full_response.append(token)

                payload = {
                    "type": "token",
                    "token": token,
                    "tok_sec": chunk.get("tokens_per_sec", 0),
                    "total_tokens": chunk.get("total_tokens", 0)
                }
                yield f"data: {json.dumps(payload)}\n\n"

                if chunk.get("done", False):
                    break
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        assistant_text = "".join(full_response).strip()

        with db_session() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, tokens) VALUES (?, 'assistant', ?, ?)",
                (session_id, assistant_text, len(full_response))
            )

        try:
            MemoryOrchestrator.post_interaction_tasks(
                session_id=session_id,
                user_msg=effective_user_msg if image_analysis_text else user_msg,
                assistant_msg=assistant_text,
                user_message_id=user_message_id
            )
        except Exception as err:
            print(f"[!] Error in post_interaction_tasks: {err}")

        yield f"data: {json.dumps({'type': 'done', 'full_response': assistant_text})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str):
    with db_session() as conn:
        cursor = conn.execute("SELECT title, mode, created_at FROM sessions WHERE id = ?", (session_id,))
        sess = cursor.fetchone()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        msg_cursor = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        )
        messages = [dict(r) for r in msg_cursor.fetchall()]

    markdown_lines = [
        f"# {sess['title']}",
        f"*Mode: {sess['mode']} | Exported from Local Grok ({sess['created_at']})*\n",
        "---\n"
    ]
    for m in messages:
        header = f"### {'👤 User' if m['role'] == 'user' else '⚡ Grok'}"
        markdown_lines.append(f"{header}\n{m['content']}\n")

    return {"title": sess["title"], "markdown": "\n".join(markdown_lines)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=WEB_HOST, port=WEB_PORT, reload=True)
