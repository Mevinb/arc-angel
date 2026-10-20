"""Versioned API for the autonomous GROK platform.

The legacy /api routes remain untouched.  This router exposes the v2 service
layer with explicit, small endpoints that the command-center UI and mobile
companion consume.
"""
from __future__ import annotations

import asyncio
import base64
import html as html_lib
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Header, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse

from config import ATTACHMENTS_DIR, DATA_DIR, DB_PATH, WORKSPACES_DIR
from services.agents import AgentService
from services.audit import AuditService
from services.automations import AutomationService
from services.backup import BackupService
from services.code_index import CodeIndexService
from services.connectors import ConnectorService
from services.events import ActivityService, NotificationService
from services.files import FileService, _allowed_roots, _resolve_root
from services.knowledge import KnowledgeService
from services.mobile import MobileService
from services.models import ModelService
from services.personal import PersonalService
from services.plugins import PluginService
from services.policies import Decision, PolicyService
from services.research import ResearchService
from services.run_manager import RunManager
from services.safety import SafetyService
from services.tasks import TaskService, TERMINAL_RUN
from services.tools import ToolContext, ToolRegistry

router = APIRouter(prefix="/api/v2", tags=["v2"])


def _not_found(message: str = "Not found") -> HTTPException:
    return HTTPException(status_code=404, detail=message)


def _bad(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def _forbidden(message: str = "Forbidden") -> HTTPException:
    return HTTPException(status_code=403, detail=message)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_path(path: str, *, must_exist: bool = False) -> Path:
    p = Path(path).expanduser().resolve()
    allowed = any(p == root or root in p.parents for root in _allowed_roots())
    if not allowed:
        raise PermissionError("Path is outside the approved roots. Set GROK_WORKSPACE_ROOTS to allow it.")
    if must_exist and not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _run_active(run_id: str) -> bool:
    return any(item["run_id"] == run_id for item in RunManager.get().active())


# --------------------------------------------------------------------- meta
@router.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": True, "version": "2.0.0", "time": _now(),
        "database": str(DB_PATH),
        "safety": SafetyService.status(),
        "run_manager": RunManager.get().active(),
    }


@router.get("/overview")
async def overview() -> Dict[str, Any]:
    return {
        "dashboard": PersonalService.dashboard(),
        "agents": AgentService.list_agents(),
        "tools": ToolRegistry.get().list_tools(),
        "profiles": ModelService.list_profiles(),
        "stats": TaskService.stats(),
        "recent_runs": TaskService.recent_runs(limit=8),
    }


@router.get("/meta")
async def meta() -> Dict[str, Any]:
    return {
        "version": "2.0.0", "api_prefix": "/api/v2",
        "database": str(DB_PATH),
        "capabilities": [
            "agents", "tasks", "runs", "checkpoints", "approvals", "policies",
            "knowledge", "research", "connectors", "automations", "mobile",
            "files", "code", "backups", "audit",
        ],
        "allowed_roots": [str(p) for p in _allowed_roots()],
        "safety": SafetyService.status(),
    }


# -------------------------------------------------------------------- safety
@router.get("/safety")
async def safety_status() -> Dict[str, Any]:
    return SafetyService.status()


@router.post("/safety/kill")
async def safety_kill(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return SafetyService.engage_kill_switch(str(payload.get("reason") or "User requested emergency stop"))


@router.post("/safety/release")
async def safety_release() -> Dict[str, Any]:
    return SafetyService.release_kill_switch()


@router.post("/safety/offline")
async def safety_offline(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return SafetyService.set_offline_mode(bool(payload.get("enabled")))


@router.post("/safety/redaction")
async def safety_redaction(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    SafetyService.set_redaction(bool(payload.get("enabled", True)))
    return SafetyService.status()


@router.post("/safety/lock")
async def safety_lock(
    response: Response, payload: Dict[str, Any] = Body(default={})
) -> Dict[str, Any]:
    pin = str(payload.get("pin") or "")
    result = SafetyService.set_lock(pin or None)
    if result.get("enabled"):
        # Keep the session that configured the lock unlocked.
        token = SafetyService.create_unlock_token(pin)
        if token:
            response.set_cookie(
                "grok_unlock", token, max_age=SafetyService.UNLOCK_TTL,
                httponly=True, samesite="lax", path="/",
            )
    else:
        response.delete_cookie("grok_unlock", path="/")
    return result


@router.get("/safety/lock-status")
async def safety_lock_status() -> Dict[str, Any]:
    return SafetyService.lock_status()


@router.post("/safety/unlock")
async def safety_unlock(
    response: Response, payload: Dict[str, Any] = Body(...)
) -> Dict[str, Any]:
    if not SafetyService.lock_enabled():
        raise HTTPException(status_code=400, detail="App lock is not enabled")
    token = SafetyService.create_unlock_token(str(payload.get("pin") or ""))
    if not token:
        raise HTTPException(status_code=401, detail="Incorrect PIN")
    response.set_cookie(
        "grok_unlock", token, max_age=SafetyService.UNLOCK_TTL,
        httponly=True, samesite="lax", path="/",
    )
    return {"ok": True, "expires_in_seconds": SafetyService.UNLOCK_TTL}


# ------------------------------------------------------------------ policies
@router.get("/policies")
async def list_policies(enabled_only: bool = False) -> List[Dict[str, Any]]:
    return PolicyService.list_policies(enabled_only=enabled_only)


@router.post("/policies")
async def create_policy(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return PolicyService.create(payload)


@router.get("/policies/{policy_id}")
async def get_policy(policy_id: str) -> Dict[str, Any]:
    policy = PolicyService.get(policy_id)
    if not policy:
        raise _not_found("Policy not found")
    return policy


@router.patch("/policies/{policy_id}")
async def update_policy(policy_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    policy = PolicyService.update(policy_id, payload)
    if not policy:
        raise _not_found("Policy not found")
    return policy


@router.delete("/policies/{policy_id}")
async def delete_policy(policy_id: str) -> Dict[str, Any]:
    if not PolicyService.delete(policy_id):
        raise _not_found("Policy not found")
    return {"ok": True}


@router.post("/policies/evaluate")
async def evaluate_policy(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return PolicyService.evaluate(payload)


@router.get("/approvals")
async def list_approvals(status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    return PolicyService.list_approvals(status=status, limit=limit)


@router.get("/approvals/{approval_id}")
async def get_approval(approval_id: str) -> Dict[str, Any]:
    approval = PolicyService.get_approval(approval_id)
    if not approval:
        raise _not_found("Approval not found")
    return approval


@router.post("/approvals/{approval_id}/resolve")
async def resolve_approval(approval_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    approval = PolicyService.resolve_approval(
        approval_id, bool(payload.get("approved")), str(payload.get("note") or ""),
    )
    if not approval:
        raise _not_found("Approval not found")
    result: Dict[str, Any] = {"approval": approval}
    run_id = approval.get("run_id")
    connector_payload = approval.get("payload") or {}
    if (
        approval["status"] == "approved"
        and connector_payload.get("connection_id")
        and connector_payload.get("action")
    ):
        try:
            result["execution"] = await ConnectorService.execute_action(
                connector_payload["connection_id"], connector_payload["action"],
                connector_payload.get("payload") or {}, approval_id=approval_id,
            )
        except Exception as exc:
            result["execution_error"] = str(exc)
    if run_id:
        run = TaskService.get_run(run_id)
        if run and not _run_active(run_id):
            try:
                if approval["status"] == "approved" and run["status"] == "awaiting_approval":
                    TaskService.update_run_status(run_id, "running")
                    if run.get("task_id"):
                        TaskService.update_task(run["task_id"], {"status": "running"})
                    RunManager.get().start(run_id)
                elif approval["status"] == "denied" and run["status"] == "awaiting_approval":
                    TaskService.update_run_status(run_id, "failed", error="Approval denied")
            except Exception as exc:
                result["resume_error"] = str(exc)
    return result


# --------------------------------------------------------------------- tools
@router.get("/tools")
async def list_tools(include_hidden: bool = False) -> List[Dict[str, Any]]:
    return ToolRegistry.get().list_tools(include_hidden=include_hidden)


@router.get("/tools/{tool_name}")
async def get_tool(tool_name: str) -> Dict[str, Any]:
    spec = ToolRegistry.get().get_tool(tool_name)
    if not spec:
        raise _not_found("Tool not found")
    return spec.manifest()


@router.post("/tools/{tool_name}/execute")
async def execute_tool(tool_name: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    args = payload.get("arguments") or {}
    workspace = payload.get("workspace")
    if workspace:
        _check_path(str(workspace), must_exist=False)
    target = Path(workspace).expanduser().resolve() if workspace else (WORKSPACES_DIR / "manual").resolve()
    target.mkdir(parents=True, exist_ok=True)
    ctx = ToolContext(
        workspace_dir=str(target), simulate=bool(payload.get("simulate")),
        approval_id=payload.get("approval_id"), interactive=True,
    )
    return await ToolRegistry.get().execute(
        tool_name, args, ctx, approval_id=payload.get("approval_id"),
    )


# -------------------------------------------------------------------- agents
@router.get("/agents")
async def list_agents(enabled_only: bool = False) -> List[Dict[str, Any]]:
    return AgentService.list_agents(enabled_only=enabled_only)


@router.post("/agents")
async def create_agent(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return AgentService.create(payload)


@router.get("/agent-templates")
async def agent_templates() -> List[Dict[str, Any]]:
    return AgentService.templates()


@router.post("/agent-templates/{template_id}/run")
async def run_agent_template(
    template_id: str, payload: Dict[str, Any] = Body(default={}),
) -> Dict[str, Any]:
    return AgentService.create_from_template(
        template_id, goal_override=payload.get("goal"),
        agent_id=payload.get("agent_id"), auto_start=bool(payload.get("auto_start", True)),
    )


@router.get("/agents/{agent_id}")
async def get_agent(agent_id: str) -> Dict[str, Any]:
    agent = AgentService.get(agent_id)
    if not agent:
        raise _not_found("Agent not found")
    return agent


@router.patch("/agents/{agent_id}")
async def update_agent(agent_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    agent = AgentService.update(agent_id, payload)
    if not agent:
        raise _not_found("Agent not found")
    return agent


@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str) -> Dict[str, Any]:
    if not AgentService.delete(agent_id):
        raise _not_found("Agent not found")
    return {"ok": True}


@router.post("/agents/{agent_id}/duplicate")
async def duplicate_agent(agent_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return AgentService.duplicate(agent_id, payload.get("name"))


@router.get("/agents/{agent_id}/manifest")
async def agent_manifest(agent_id: str) -> Dict[str, Any]:
    try:
        return AgentService.manifest(agent_id)
    except ValueError as exc:
        raise _not_found(str(exc))


# --------------------------------------------------------------------- tasks
@router.get("/tasks")
async def list_tasks(
    status: Optional[str] = None, agent_id: Optional[str] = None,
    q: Optional[str] = None, limit: int = 50, offset: int = 0,
) -> Dict[str, Any]:
    return TaskService.list_tasks(
        status=status, agent_id=agent_id, q=q, limit=limit, offset=offset,
    )


@router.post("/tasks")
async def create_task(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    title = payload.get("title") or (payload.get("goal") or "Untitled task")[:200]
    goal = payload.get("goal") or title
    return TaskService.create_task(
        title=title, goal=goal,
        agent_id=payload.get("agent_id") or "orchestrator",
        priority=int(payload.get("priority") or 3),
        due_at=payload.get("due_at"), dependencies=payload.get("dependencies"),
        budget=payload.get("budget"), policy_id=payload.get("policy_id"),
        session_id=payload.get("session_id"), parent_id=payload.get("parent_id"),
        origin=payload.get("origin") or "user", tags=payload.get("tags"),
        workspace_dir=payload.get("workspace_dir"),
        auto_start=bool(payload.get("auto_start", False)),
    )


@router.get("/tasks/{task_id}")
async def get_task(task_id: str) -> Dict[str, Any]:
    task = TaskService.get_task(task_id)
    if not task:
        raise _not_found("Task not found")
    return task


@router.patch("/tasks/{task_id}")
async def update_task(task_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    task = TaskService.update_task(task_id, payload)
    if not task:
        raise _not_found("Task not found")
    return task


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str) -> Dict[str, Any]:
    if not TaskService.delete_task(task_id):
        raise _not_found("Task not found")
    return {"ok": True}


@router.post("/tasks/{task_id}/start")
async def start_task(task_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    run = TaskService.create_run(task_id, budget_override=payload.get("budget"))
    RunManager.get().start(run["id"])
    return {"ok": True, "run": TaskService.get_run(run["id"])}


@router.post("/tasks/{task_id}/retry")
async def retry_task(task_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    task = TaskService.get_task(task_id)
    if not task:
        raise _not_found("Task not found")
    if task.get("latest_run"):
        latest = task["latest_run"]
        if latest.get("status") not in TERMINAL_RUN and latest.get("status") not in ("paused", "interrupted"):
            raise _bad("Task already has an active run")
    run = TaskService.create_run(task_id, budget_override=payload.get("budget"), replay_of=payload.get("replay_of"))
    RunManager.get().start(run["id"])
    return {"ok": True, "run": TaskService.get_run(run["id"])}


@router.post("/tasks/{task_id}/pause")
async def pause_task(task_id: str) -> Dict[str, Any]:
    task = TaskService.get_task(task_id)
    if not task:
        raise _not_found("Task not found")
    active = next((r for r in TaskService.list_runs(task_id=task_id) if r["status"] in ("running", "queued")), None)
    if not active:
        raise _bad("Task has no active run")
    return await RunManager.get().pause(active["id"])


@router.post("/tasks/{task_id}/resume")
async def resume_task(task_id: str) -> Dict[str, Any]:
    runs = TaskService.list_runs(task_id=task_id)
    run = next((r for r in runs if r["status"] in ("paused", "interrupted")), None)
    if not run:
        raise _bad("Task has no paused or interrupted run")
    return await RunManager.get().resume(run["id"])


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> Dict[str, Any]:
    task = TaskService.get_task(task_id)
    if not task:
        raise _not_found("Task not found")
    active = next((r for r in TaskService.list_runs(task_id=task_id) if r["status"] not in TERMINAL_RUN), None)
    if active:
        return await RunManager.get().cancel(active["id"])
    TaskService.update_task(task_id, {"status": "cancelled"})
    return TaskService.get_task(task_id)


@router.get("/runs")
async def list_runs(task_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    return TaskService.list_runs(task_id=task_id, status=status, limit=limit)


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> Dict[str, Any]:
    run = TaskService.get_run(run_id)
    if not run:
        raise _not_found("Run not found")
    return run


@router.post("/runs/{run_id}/pause")
async def pause_run(run_id: str) -> Dict[str, Any]:
    return await RunManager.get().pause(run_id)


@router.post("/runs/{run_id}/resume")
async def resume_run(run_id: str) -> Dict[str, Any]:
    return await RunManager.get().resume(run_id)


@router.post("/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> Dict[str, Any]:
    return await RunManager.get().cancel(run_id)


@router.get("/runs/{run_id}/events")
async def run_events(run_id: str, since_id: int = 0, limit: int = 1000) -> List[Dict[str, Any]]:
    if not TaskService.get_run(run_id):
        raise _not_found("Run not found")
    return TaskService.get_events(run_id, since_id=since_id, limit=limit)


@router.get("/runs/{run_id}/timeline")
async def run_timeline(run_id: str) -> Dict[str, Any]:
    timeline = TaskService.timeline(run_id)
    if not timeline:
        raise _not_found("Run not found")
    return timeline


@router.get("/runs/{run_id}/events/stream")
async def stream_run_events(run_id: str, request: Request) -> StreamingResponse:
    if not TaskService.get_run(run_id):
        raise _not_found("Run not found")

    async def generate():
        last = 0
        idle = 0
        while True:
            if await request.is_disconnected():
                break
            events = TaskService.get_events(run_id, since_id=last, limit=500)
            if events:
                idle = 0
                for event in events:
                    last = max(last, event["id"])
                    yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"
            else:
                idle += 1
                if idle % 20 == 0:
                    yield ": keepalive\n\n"
            run = TaskService.get_run(run_id)
            if run and run["status"] in TERMINAL_RUN:
                yield f"event: stream.end\ndata: {json.dumps({'run_id': run_id, 'status': run['status']})}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/event-stream")


# ------------------------------------------------------------------ research
@router.get("/research")
async def list_research(limit: int = 50) -> List[Dict[str, Any]]:
    return ResearchService.list(limit=limit)


@router.post("/research")
async def create_research(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    project = ResearchService.create(
        payload.get("question") or payload.get("title") or "",
        title=payload.get("title"), max_sources=int(payload.get("max_sources") or 8),
        profile=payload.get("profile") or "quality",
    )
    if payload.get("auto_start", True):
        ResearchService.start(
            project["id"], max_sources=int(payload.get("max_sources") or 8),
            profile=payload.get("profile") or "quality",
        )
    return ResearchService.get(project["id"])  # type: ignore[return-value]


@router.get("/research/{project_id}")
async def get_research(project_id: str, wait: bool = False, timeout: float = 180.0) -> Dict[str, Any]:
    project = await ResearchService.wait(project_id, timeout=timeout) if wait else ResearchService.get(project_id)
    if not project:
        raise _not_found("Research project not found")
    return project


@router.delete("/research/{project_id}")
async def delete_research(project_id: str) -> Dict[str, Any]:
    if not ResearchService.delete(project_id):
        raise _not_found("Research project not found")
    return {"ok": True}


@router.post("/research/{project_id}/start")
async def start_research(project_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return ResearchService.start(
        project_id, max_sources=int(payload.get("max_sources") or 8),
        profile=payload.get("profile") or "quality",
    )


@router.post("/research/{project_id}/notes")
async def add_research_note(project_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return ResearchService.add_note(
        project_id, kind=payload.get("kind") or "note",
        title=payload.get("title") or "Note", content=payload.get("content") or "",
        url=payload.get("url"), credibility=float(payload.get("credibility") or 0.6),
    )


@router.post("/research/{project_id}/task")
async def research_to_task(project_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    project = ResearchService.get(project_id)
    if not project:
        raise _not_found("Research project not found")
    return TaskService.create_task(
        title=payload.get("title") or f"Deep research: {project['title']}",
        goal=payload.get("goal") or (
            f"Deeply research this question and produce a cited report: {project['question']}. "
            "Use web sources, verify conflicting claims, and save durable findings."
        ),
        agent_id="researcher", origin="research", tags=["research", project_id],
        auto_start=bool(payload.get("auto_start", True)),
    )


# ----------------------------------------------------------------- knowledge
@router.get("/knowledge/sources")
async def list_knowledge(
    q: Optional[str] = None, source_type: Optional[str] = None,
    profile: Optional[str] = None, limit: int = 100, offset: int = 0,
) -> Dict[str, Any]:
    return KnowledgeService.list_sources(
        q=q, source_type=source_type, limit=limit, offset=offset,
    )


@router.post("/knowledge/sources")
async def add_knowledge(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await asyncio.to_thread(
        KnowledgeService.add_text,
        title=payload.get("title") or "Untitled",
        content=payload.get("content") or "",
        source_type=payload.get("source_type") or "note",
        uri=payload.get("uri"), author=payload.get("author"),
        tags=payload.get("tags"), metadata=payload.get("metadata"),
        private=bool(payload.get("private")), profile=payload.get("profile") or "personal",
        credibility=float(payload.get("credibility") or 0.6),
    )


@router.post("/knowledge/file")
async def add_knowledge_file(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    path = _check_path(str(payload.get("path") or ""), must_exist=True)
    if not path.is_file():
        raise _bad("Path is not a file")
    return await asyncio.to_thread(
        KnowledgeService.add_file, str(path), payload.get("title"),
        payload.get("tags"), bool(payload.get("private")), payload.get("profile") or "personal",
    )


@router.post("/knowledge/folder")
async def add_knowledge_folder(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    path = _check_path(str(payload.get("path") or ""), must_exist=True)
    if not path.is_dir():
        raise _bad("Path is not a folder")
    return await asyncio.to_thread(
        KnowledgeService.add_folder, str(path), payload.get("tags"),
        int(payload.get("max_files") or 500), bool(payload.get("private")),
        payload.get("profile") or "personal",
    )


@router.post("/knowledge/url")
async def add_knowledge_url(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await KnowledgeService.add_url(
        payload.get("url") or "", tags=payload.get("tags"),
        private=bool(payload.get("private")), profile=payload.get("profile") or "personal",
    )


@router.post("/knowledge/transcript")
async def add_knowledge_transcript(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await asyncio.to_thread(
        KnowledgeService.add_transcript,
        payload.get("title") or "Transcript", payload.get("content") or "",
        payload.get("people"), payload.get("source_type") or "transcript",
        tags=payload.get("tags"), private=bool(payload.get("private")),
        profile=payload.get("profile") or "personal",
    )


@router.post("/knowledge/search")
async def search_knowledge(payload: Dict[str, Any] = Body(...)) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(
        KnowledgeService.search, payload.get("query") or "",
        limit=int(payload.get("limit") or 8),
        source_type=payload.get("source_type"), profile=payload.get("profile"),
        private=payload.get("private"), tags=payload.get("tags"), people=payload.get("people"),
    )


@router.post("/knowledge/ask")
async def ask_knowledge(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await asyncio.to_thread(
        KnowledgeService.ask, payload.get("question") or "",
        filters={k: v for k, v in payload.items() if k in ("source_type", "profile", "private", "tags", "people", "limit")},
        synthesize=bool(payload.get("synthesize")),
        answer_style=payload.get("answer_style") or "concise",
    )


@router.get("/knowledge/stats")
async def knowledge_stats() -> Dict[str, Any]:
    return KnowledgeService.stats()


@router.post("/knowledge/reindex")
async def reindex_knowledge() -> Dict[str, Any]:
    return await asyncio.to_thread(KnowledgeService.reindex)


@router.get("/knowledge/sources/{source_id}")
async def get_knowledge_source(source_id: str) -> Dict[str, Any]:
    source = KnowledgeService.get_source(source_id)
    if not source:
        raise _not_found("Knowledge source not found")
    return source


@router.delete("/knowledge/sources/{source_id}")
async def delete_knowledge_source(source_id: str) -> Dict[str, Any]:
    if not KnowledgeService.delete_source(source_id):
        raise _not_found("Knowledge source not found")
    return {"ok": True}


# --------------------------------------------------------------------- models
@router.get("/models/profiles")
async def model_profiles() -> Dict[str, Any]:
    return ModelService.list_profiles()


@router.post("/models/route")
async def route_model(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return ModelService.route(
        task_type=payload.get("task_type") or "general",
        profile=payload.get("profile"), optimize=payload.get("optimize"),
    )


@router.post("/models/compare")
async def compare_models(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await ModelService.compare(
        payload.get("prompt") or "", payload.get("models") or [],
        int(payload.get("max_tokens") or 500),
    )


@router.post("/models/vote")
async def vote_model(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return ModelService.vote(
        payload.get("prompt") or "", payload.get("model") or "",
        payload.get("provider") or "openai", payload.get("vote") or "up",
    )


@router.get("/models/performance")
async def model_performance() -> List[Dict[str, Any]]:
    return ModelService.performance()


# ------------------------------------------------------------------- activity
@router.get("/activity")
async def list_activity(
    kind: Optional[str] = None, severity: Optional[str] = None,
    q: Optional[str] = None, limit: int = 100, before_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    return ActivityService.list(kind=kind, severity=severity, q=q, limit=limit, before_id=before_id)


@router.get("/notifications")
async def list_notifications(
    unread_only: bool = False, include_dismissed: bool = False, limit: int = 100,
) -> List[Dict[str, Any]]:
    return NotificationService.list(
        unread_only=unread_only, include_dismissed=include_dismissed, limit=limit,
    )


@router.get("/notifications/unread-count")
async def unread_notifications() -> Dict[str, int]:
    return {"unread": NotificationService.unread_count()}


@router.post("/notifications/{notification_id}/read")
async def mark_notification_read(notification_id: int) -> Dict[str, Any]:
    count = NotificationService.mark_read(notification_id)
    return {"ok": bool(count), "updated": count}


@router.post("/notifications/read-all")
async def mark_all_notifications_read() -> Dict[str, Any]:
    return {"updated": NotificationService.mark_read(all_=True)}


@router.delete("/notifications/{notification_id}")
async def dismiss_notification(notification_id: int) -> Dict[str, Any]:
    return {"ok": NotificationService.dismiss(notification_id)}


# ---------------------------------------------------------------------- audit
@router.get("/audit")
async def list_audit(
    category: Optional[str] = None, status: Optional[str] = None,
    run_id: Optional[str] = None, task_id: Optional[str] = None,
    agent_id: Optional[str] = None, tool_name: Optional[str] = None,
    limit: int = 100, offset: int = 0,
) -> List[Dict[str, Any]]:
    return AuditService.list_events(
        category=category, status=status, run_id=run_id, task_id=task_id,
        agent_id=agent_id, tool_name=tool_name, limit=limit, offset=offset,
    )


@router.get("/audit/verify")
async def verify_audit() -> Dict[str, Any]:
    return AuditService.verify_chain()


@router.get("/audit/export")
async def export_audit() -> Dict[str, Any]:
    return AuditService.export()


@router.get("/audit/{event_id}")
async def get_audit(event_id: int) -> Dict[str, Any]:
    event = AuditService.get(event_id)
    if not event:
        raise _not_found("Audit event not found")
    return event


# ----------------------------------------------------------------- connectors
@router.get("/connectors")
async def connector_catalog() -> List[Dict[str, Any]]:
    return ConnectorService.catalog()


@router.get("/connectors/connections")
async def list_connections() -> List[Dict[str, Any]]:
    return ConnectorService.list_connections()


@router.post("/connectors/{connector_id}/oauth-url")
async def connector_oauth_url(connector_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return ConnectorService.oauth_url(
        connector_id, payload.get("redirect_uri") or "",
        account_label=payload.get("account_label") or "Primary",
    )


@router.get("/connectors/oauth/callback")
async def connector_oauth_callback(code: str, state: str) -> HTMLResponse:
    try:
        connection = await ConnectorService.complete_oauth(code, state)
        body = (
            f"<h2>Connected {html_lib.escape(str(connection['connector_id']))}</h2>"
            "<p>You can close this window.</p>"
        )
    except Exception as exc:
        body = f"<h2>Connection failed</h2><pre>{html_lib.escape(str(exc))}</pre>"
    return HTMLResponse(
        f"<html><body style='font-family:system-ui;padding:2rem;background:#111;color:#eee'>{body}"
        "<script>setTimeout(()=>window.close(),1800)</script></body></html>"
    )


@router.post("/connectors/{connector_id}/connect")
async def connect_connector(connector_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return ConnectorService.connect_credentials(
        connector_id, account_label=payload.get("account_label") or "Primary",
        credentials=payload.get("credentials") or {}, scopes=payload.get("scopes"),
        metadata=payload.get("metadata"),
    )


@router.post("/connections/{connection_id}/test")
async def test_connection(connection_id: str) -> Dict[str, Any]:
    return await ConnectorService.test(connection_id)


@router.post("/connections/{connection_id}/sync")
async def sync_connection(connection_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return await ConnectorService.sync(connection_id, int(payload.get("limit") or 25))


@router.get("/connections/{connection_id}/items")
async def connection_items(connection_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    return ConnectorService.items(connection_id, limit=limit)


@router.delete("/connections/{connection_id}")
async def disconnect_connection(connection_id: str) -> Dict[str, Any]:
    if not ConnectorService.disconnect(connection_id):
        raise _not_found("Connection not found")
    return {"ok": True}


@router.post("/connections/{connection_id}/actions")
async def connector_action(connection_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    action = payload.get("action") or ""
    action_payload = payload.get("payload") or {}
    prepared = ConnectorService.prepare_action(connection_id, action, action_payload)
    if prepared.get("status") == "ready":
        return await ConnectorService.execute_action(connection_id, action, action_payload)
    return prepared


@router.post("/connections/{connection_id}/actions/execute")
async def execute_connector_action(connection_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await ConnectorService.execute_action(
        connection_id, payload.get("action") or "", payload.get("payload") or {},
        approval_id=payload.get("approval_id"),
    )


# ---------------------------------------------------------------- automations
@router.get("/automations")
async def list_automations() -> List[Dict[str, Any]]:
    return AutomationService.list()


@router.post("/automations")
async def create_automation(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return AutomationService.create(payload)


@router.get("/automations/{automation_id}")
async def get_automation(automation_id: str) -> Dict[str, Any]:
    automation = AutomationService.get(automation_id)
    if not automation:
        raise _not_found("Automation not found")
    return automation


@router.patch("/automations/{automation_id}")
async def update_automation(automation_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    automation = AutomationService.update(automation_id, payload)
    if not automation:
        raise _not_found("Automation not found")
    return automation


@router.delete("/automations/{automation_id}")
async def delete_automation(automation_id: str) -> Dict[str, Any]:
    if not AutomationService.delete(automation_id):
        raise _not_found("Automation not found")
    return {"ok": True}


@router.post("/automations/{automation_id}/run")
async def run_automation(automation_id: str) -> Dict[str, Any]:
    return await AutomationService.run(automation_id, trigger="manual")


@router.post("/automations/{automation_id}/toggle")
async def toggle_automation(automation_id: str) -> Dict[str, Any]:
    automation = AutomationService.get(automation_id)
    if not automation:
        raise _not_found("Automation not found")
    return AutomationService.update(automation_id, {"enabled": not automation["enabled"]})


@router.post("/hooks/{automation_id}")
async def automation_webhook(automation_id: str, request: Request) -> Dict[str, Any]:
    raw = await request.body()
    signature = request.headers.get("X-Grok-Signature") or request.headers.get("X-Hub-Signature-256") or ""
    if not AutomationService.verify_webhook(automation_id, raw, signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except Exception:
        payload = {"raw": raw.decode("utf-8", errors="replace")[:10000]}
    return await AutomationService.run(automation_id, trigger="webhook", payload=payload)


# --------------------------------------------------------------------- files
@router.get("/files")
async def file_info(root: Optional[str] = None) -> Dict[str, Any]:
    return FileService.root_info(root)


@router.get("/files/tree")
async def file_tree(root: Optional[str] = None, max_entries: int = 1000, max_depth: int = 6) -> Dict[str, Any]:
    return FileService.tree(root, max_entries=max_entries, max_depth=max_depth)


@router.get("/files/read")
async def file_read(path: str, root: Optional[str] = None, max_chars: int = 100000) -> Dict[str, Any]:
    return FileService.read(path, root, max_chars=max_chars)


@router.put("/files/write")
async def file_write(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return FileService.write(payload.get("path") or "", payload.get("content") or "", payload.get("root"))


@router.get("/files/search")
async def file_search(q: str, root: Optional[str] = None, max_results: int = 50) -> Dict[str, Any]:
    return FileService.search(q, root, max_results=max_results)


@router.get("/files/checkpoints")
async def list_checkpoints() -> List[Dict[str, Any]]:
    return FileService.list_checkpoints()


@router.post("/files/checkpoints")
async def create_checkpoint(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await asyncio.to_thread(FileService.create_checkpoint, payload.get("label") or "manual", payload.get("root"))


@router.get("/files/checkpoints/{checkpoint_id}/diff")
async def diff_checkpoint(checkpoint_id: str) -> Dict[str, Any]:
    return await asyncio.to_thread(FileService.diff_checkpoint, checkpoint_id)


@router.post("/files/checkpoints/{checkpoint_id}/restore")
async def restore_checkpoint(checkpoint_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return await asyncio.to_thread(
        FileService.restore_checkpoint, checkpoint_id,
        confirm=bool(payload.get("confirm")), prune=bool(payload.get("prune")),
    )


@router.delete("/files/checkpoints/{checkpoint_id}")
async def delete_checkpoint(checkpoint_id: str) -> Dict[str, Any]:
    if not FileService.delete_checkpoint(checkpoint_id):
        raise _not_found("Checkpoint not found")
    return {"ok": True}


# ---------------------------------------------------------------------- code
@router.get("/code/repos")
async def code_repos() -> List[Dict[str, Any]]:
    return CodeIndexService.repos()


@router.post("/code/index")
async def code_index(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return await asyncio.to_thread(CodeIndexService.index, payload.get("root"))


@router.get("/code/search")
async def code_search(q: str, root: Optional[str] = None, limit: int = 30) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(CodeIndexService.search, q, root, limit)


@router.get("/code/symbols")
async def code_symbols(root: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(CodeIndexService.symbols, root, limit)


@router.get("/code/repo-info")
async def code_repo_info(root: Optional[str] = None) -> Dict[str, Any]:
    return await asyncio.to_thread(CodeIndexService.repo_info, root)


@router.get("/code/file")
async def code_file(path: str, root: Optional[str] = None) -> Dict[str, Any]:
    return await asyncio.to_thread(CodeIndexService.read_file, path, root)


# ------------------------------------------------------------------ personal
@router.get("/dashboard")
async def dashboard() -> Dict[str, Any]:
    return PersonalService.dashboard()


@router.get("/dashboard/brief")
async def daily_brief() -> Dict[str, Any]:
    return PersonalService.daily_brief()


@router.get("/documents")
async def list_documents(q: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    return PersonalService.list_documents(q=q, limit=limit)


@router.post("/documents")
async def create_document(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return PersonalService.create_document(payload)


@router.get("/documents/{doc_id}")
async def get_document(doc_id: str) -> Dict[str, Any]:
    doc = PersonalService.get_document(doc_id)
    if not doc:
        raise _not_found("Document not found")
    return doc


@router.patch("/documents/{doc_id}")
async def update_document(doc_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    doc = PersonalService.update_document(doc_id, payload)
    if not doc:
        raise _not_found("Document not found")
    return doc


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str) -> Dict[str, Any]:
    if not PersonalService.delete_document(doc_id):
        raise _not_found("Document not found")
    return {"ok": True}


@router.get("/finance/transactions")
async def list_transactions(
    month: Optional[str] = None, category: Optional[str] = None, limit: int = 200,
) -> List[Dict[str, Any]]:
    return PersonalService.list_transactions(month=month, category=category, limit=limit)


@router.post("/finance/transactions")
async def add_transaction(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return PersonalService.add_transaction(payload)


@router.delete("/finance/transactions/{tx_id}")
async def delete_transaction(tx_id: int) -> Dict[str, Any]:
    if not PersonalService.delete_transaction(tx_id):
        raise _not_found("Transaction not found")
    return {"ok": True}


@router.get("/finance/summary")
async def finance_summary(month: Optional[str] = None) -> Dict[str, Any]:
    return PersonalService.finance_summary(month)


@router.get("/bills")
async def list_bills(unpaid_only: bool = False) -> List[Dict[str, Any]]:
    return PersonalService.list_bills(unpaid_only=unpaid_only)


@router.post("/bills")
async def create_bill(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return PersonalService.create_bill(payload)


@router.post("/bills/{bill_id}/pay")
async def pay_bill(bill_id: str) -> Dict[str, Any]:
    return {"ok": PersonalService.pay_bill(bill_id)}


@router.get("/habits")
async def list_habits() -> List[Dict[str, Any]]:
    return PersonalService.list_habits()


@router.post("/habits")
async def create_habit(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return PersonalService.create_habit(payload)


@router.post("/habits/{habit_id}/log")
async def log_habit(habit_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return PersonalService.log_habit(
        habit_id, float(payload.get("value") or 1), payload.get("day"),
    )


@router.get("/prompts")
async def list_prompts(category: Optional[str] = None) -> List[Dict[str, Any]]:
    return PersonalService.list_prompts(category)


@router.post("/prompts")
async def create_prompt(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return PersonalService.create_prompt(payload)


@router.delete("/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str) -> Dict[str, Any]:
    return {"ok": PersonalService.delete_prompt(prompt_id)}


@router.get("/shortcuts")
async def list_shortcuts() -> List[Dict[str, Any]]:
    return PersonalService.list_shortcuts()


@router.post("/shortcuts")
async def upsert_shortcut(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return PersonalService.upsert_shortcut(payload)


@router.delete("/shortcuts/{shortcut_id}")
async def delete_shortcut(shortcut_id: str) -> Dict[str, Any]:
    return {"ok": PersonalService.delete_shortcut(shortcut_id)}


@router.get("/profiles")
async def list_profiles() -> List[Dict[str, Any]]:
    return PersonalService.list_profiles()


@router.post("/profiles")
async def create_profile(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return PersonalService.create_profile(payload)


@router.patch("/profiles/{profile_id}")
async def update_profile(profile_id: str, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    profile = PersonalService.update_profile(profile_id, payload)
    if not profile:
        raise _not_found("Profile not found")
    return profile


# ------------------------------------------------------------------- backups
@router.get("/backups")
async def list_backups() -> List[Dict[str, Any]]:
    return BackupService.list()


@router.post("/backups")
async def create_backup(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return await asyncio.to_thread(
        BackupService.create, payload.get("label") or "manual",
        payload.get("passphrase"),
    )


@router.post("/backups/{backup_id}/inspect")
async def inspect_backup(backup_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return await asyncio.to_thread(BackupService.inspect, backup_id, payload.get("passphrase"))


@router.post("/backups/{backup_id}/stage-restore")
async def stage_restore(backup_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return await asyncio.to_thread(
        BackupService.stage_restore, backup_id, payload.get("passphrase"), payload.get("dest"),
    )


@router.delete("/backups/{backup_id}")
async def delete_backup(backup_id: str) -> Dict[str, Any]:
    if not BackupService.delete(backup_id):
        raise _not_found("Backup not found")
    return {"ok": True}


@router.post("/privacy/export")
async def export_data(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return await asyncio.to_thread(BackupService.export_json, payload.get("passphrase"))


@router.post("/privacy/cleanup")
async def cleanup_data(payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    return await asyncio.to_thread(
        BackupService.cleanup,
        activity_days=int(payload.get("activity_days") or 180),
        backup_days=int(payload.get("backup_days") or 30),
    )


@router.post("/privacy/delete-all")
async def delete_all_data(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return await asyncio.to_thread(
        BackupService.delete_all_data, payload.get("confirm") or "",
        include_audit=bool(payload.get("include_audit")),
        delete_files=bool(payload.get("delete_files")),
        passphrase=payload.get("passphrase"),
    )


# ------------------------------------------------------------------- plugins
@router.get("/plugins")
async def list_plugins() -> List[Dict[str, Any]]:
    return PluginService.list_plugins()


@router.post("/plugins/install")
async def install_plugin(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return PluginService.install(payload.get("path") or "")


@router.post("/plugins/{plugin_id}/enable")
async def enable_plugin(plugin_id: str, payload: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
    plugin = PluginService.enable(plugin_id, bool(payload.get("enabled", True)))
    if not plugin:
        raise _not_found("Plugin not found")
    return plugin


@router.delete("/plugins/{plugin_id}")
async def uninstall_plugin(plugin_id: str) -> Dict[str, Any]:
    if not PluginService.uninstall(plugin_id):
        raise _not_found("Plugin not found")
    return {"ok": True}


# --------------------------------------------------------------------- mobile
def _mobile_device(request: Request) -> Dict[str, Any]:
    token = request.headers.get("X-Mobile-Token") or ""
    if not token:
        auth = request.headers.get("Authorization") or ""
        if auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()
    device = MobileService.authenticate(token)
    if not device:
        raise HTTPException(status_code=401, detail="Invalid or missing mobile token")
    return device


@router.post("/mobile/pairing")
async def create_mobile_pairing(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return MobileService.create_pairing(
        payload.get("name") or "My phone", payload.get("platform") or "mobile",
    )


@router.post("/mobile/pair")
async def pair_mobile(payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return MobileService.pair(payload.get("device_id") or "", payload.get("code") or "")


@router.get("/mobile/devices")
async def list_mobile_devices() -> List[Dict[str, Any]]:
    return MobileService.list_devices()


@router.delete("/mobile/devices/{device_id}")
async def revoke_mobile_device(device_id: str) -> Dict[str, Any]:
    if not MobileService.revoke(device_id):
        raise _not_found("Device not found")
    return {"ok": True}


@router.get("/mobile/me")
async def mobile_me(request: Request) -> Dict[str, Any]:
    return _mobile_device(request)


@router.post("/mobile/share")
async def mobile_share(request: Request, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return MobileService.share(_mobile_device(request), payload)


@router.post("/mobile/command")
async def mobile_command(request: Request, payload: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return MobileService.command(_mobile_device(request), payload.get("text") or "")


@router.get("/mobile/brief")
async def mobile_brief(request: Request) -> Dict[str, Any]:
    _mobile_device(request)
    return PersonalService.daily_brief()
