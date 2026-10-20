"""Built-in typed tools exposed to agents and the extension API."""
from __future__ import annotations

import asyncio
import base64
import difflib
import hashlib
import ipaddress
import json
import os
import re
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

from services.policies import RiskClass
from services.safety import SafetyService
from services.tools import ToolContext, ToolRegistry, ToolSpec

MAX_READ_CHARS = 50_000
MAX_OUTPUT_CHARS = 24_000

_DANGEROUS_SHELL = [
    r"rm\s+-[a-z]*r[a-z]*f[a-z]*\s+/(?:\s|$|\*)",
    r"\bmkfs(?:\.\w+)?\b",
    r"\bdd\b[^\n]*\bof=/dev/(?:sd|nvme|vd|hd)",
    r":\(\)\s*\{",
    r"\b(?:shutdown|reboot|poweroff|halt)\b",
    r">\s*/dev/(?:sd|nvme|vd|hd)",
    r"(?:curl|wget)[^\n|]*\|\s*(?:bash|sh|zsh)",
    r"\bchmod\s+-R\s+777\s+/(?:\s|$)",
    r"\b(?:useradd|userdel|passwd)\b",
]
_BLOCKED_HOSTS = {
    "localhost", "localhost.localdomain", "metadata.google.internal",
    "instance-data", "169.254.169.254",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(value: Any, limit: int = MAX_OUTPUT_CHARS) -> Any:
    if isinstance(value, str):
        if len(value) > limit:
            return value[:limit] + f"\n… [truncated {len(value) - limit} chars]"
        return value
    return value


def is_safe_public_url(url: str, allow_private: bool = False) -> bool:
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.strip().lower().rstrip(".")
    if allow_private:
        return True
    if host in _BLOCKED_HOSTS or host.endswith(".local") or host.endswith(".internal"):
        return False
    try:
        ip = ipaddress.ip_address(host)
        if (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved or ip.is_unspecified
        ):
            return False
        return True
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80), proto=socket.IPPROTO_TCP)
        for info in infos:
            addr = info[4][0]
            ip = ipaddress.ip_address(addr)
            if (
                ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_multicast or ip.is_reserved or ip.is_unspecified
            ):
                return False
    except socket.gaierror:
        # DNS may be temporarily unavailable (e.g. offline); the HTTP call will
        # produce a clear error. Block only when resolution yields a bad address.
        return True
    except Exception:
        return False
    return True


def _allow_private_fetch() -> bool:
    return os.getenv("GROK_ALLOW_PRIVATE_FETCH", "0") == "1"


def _clean_env() -> Dict[str, str]:
    blocked = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "OPENAI", "MASTER", "COOKIE")
    env = {}
    for key, value in os.environ.items():
        if any(part in key.upper() for part in blocked):
            continue
        env[key] = value
    return env


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
async def _files_list(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    target = ctx.safe_path(args.get("path", "."))
    if not target.exists():
        return {"success": False, "error": f"Not found: {args.get('path', '.')}"}
    limit = int(args.get("max_entries", 500))
    entries: List[Dict[str, Any]] = []
    if target.is_file():
        return {"success": True, "entries": [{"path": target.name, "is_dir": False, "size_bytes": target.stat().st_size}]}
    iterator = target.rglob("*") if args.get("recursive", False) else target.iterdir()
    for item in iterator:
        try:
            rel = str(item.relative_to(ctx.workspace_root()))
        except ValueError:
            rel = str(item)
        if any(part in (".git", "__pycache__", "node_modules", ".venv") for part in Path(rel).parts):
            continue
        try:
            entries.append({
                "path": rel,
                "is_dir": item.is_dir(),
                "size_bytes": item.stat().st_size if item.is_file() else 0,
            })
        except OSError:
            continue
        if len(entries) >= limit:
            break
    entries.sort(key=lambda e: (not e["is_dir"], e["path"].lower()))
    return {"success": True, "root": str(ctx.workspace_root()), "entries": entries, "truncated": len(entries) >= limit}


async def _files_read(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    target = ctx.safe_path(args["path"])
    if not target.is_file():
        return {"success": False, "error": f"File not found: {args['path']}"}
    limit = min(int(args.get("max_chars", 12000)), MAX_READ_CHARS)
    raw = target.read_bytes()
    if b"\x00" in raw[:4096]:
        return {"success": False, "error": "Binary file; use specialized tools to inspect it.", "size_bytes": len(raw)}
    text = raw.decode("utf-8", errors="replace")
    return {
        "success": True, "path": args["path"], "size_bytes": len(raw),
        "content": text[:limit], "truncated": len(text) > limit,
    }


async def _files_write(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    target = ctx.safe_path(args["path"])
    content = args.get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False, default=str)
    mode = str(args.get("mode") or "overwrite").lower()
    if mode not in ("overwrite", "append"):
        return {"success": False, "error": "mode must be 'overwrite' or 'append'"}

    previous_size = target.stat().st_size if target.exists() and target.is_file() else 0
    # Diff previews are bounded. The write itself is never rewritten from this
    # truncated copy: append streams to the file so large files stay intact.
    old = ""
    if previous_size <= MAX_OUTPUT_CHARS:
        try:
            old = target.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError):
            old = ""
    addition = content.encode("utf-8")
    if mode == "append":
        new_bytes = previous_size + len(addition)
        diff_source = old + content if old or previous_size == 0 else ""
        diff = "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True), diff_source.splitlines(keepends=True),
                fromfile=args["path"], tofile=args["path"], n=2,
            )
        )[:8000]
        if not diff and previous_size:
            diff = f"Appended {len(addition)} bytes to an existing file; full diff suppressed for a large file."
    else:
        new_bytes = len(addition)
        diff = "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True), content.splitlines(keepends=True),
                fromfile=args["path"], tofile=args["path"], n=2,
            )
        )[:8000]
    if ctx.simulate:
        return {
            "success": True, "simulated": True, "path": args["path"],
            "bytes": new_bytes, "diff": diff, "mode": mode,
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "append":
        with target.open("a", encoding="utf-8") as handle:
            handle.write(content)
        new_bytes = target.stat().st_size
    else:
        target.write_text(content, encoding="utf-8")
        new_bytes = target.stat().st_size
    return {
        "success": True, "path": args["path"], "bytes": new_bytes,
        "diff": diff, "mode": mode,
    }


async def _files_delete(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    target = ctx.safe_path(args["path"])
    if not target.exists():
        return {"success": False, "error": f"Not found: {args['path']}"}
    if target == ctx.workspace_root():
        return {"success": False, "error": "Refusing to delete the workspace root."}
    if target.is_dir():
        count = sum(1 for _ in target.rglob("*"))
        if not args.get("recursive", False):
            return {"success": False, "error": "Directory is not empty; pass recursive=true after approval."}
    else:
        count = 1
    if ctx.simulate:
        return {"success": True, "simulated": True, "would_delete": args["path"], "items": count}
    import shutil

    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"success": True, "deleted": args["path"], "items": count}


async def _files_move(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    src = ctx.safe_path(args["source"])
    dst = ctx.safe_path(args["target"])
    if not src.exists():
        return {"success": False, "error": f"Not found: {args['source']}"}
    if dst.exists() and not args.get("overwrite", False):
        return {"success": False, "error": f"Target already exists: {args['target']}"}
    if ctx.simulate:
        return {"success": True, "simulated": True, "source": args["source"], "target": args["target"]}
    dst.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    if args.get("overwrite", False) and dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))
    return {"success": True, "source": args["source"], "target": args["target"]}


async def _files_mkdir(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    target = ctx.safe_path(args["path"])
    if ctx.simulate:
        return {"success": True, "simulated": True, "would_create": args["path"]}
    target.mkdir(parents=True, exist_ok=True)
    return {"success": True, "created": args["path"]}


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------
async def _shell_run(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    command = str(args.get("command") or "").strip()
    for pattern in _DANGEROUS_SHELL:
        if re.search(pattern, command, re.IGNORECASE):
            return {
                "success": False, "blocked": True,
                "error": "Command blocked by hard safety guard (catastrophic/destructive pattern).",
            }
    if ctx.simulate:
        return {"success": True, "simulated": True, "would_run": command, "cwd": str(ctx.workspace_root())}

    timeout = min(float(args.get("timeout", 60)), 180.0)
    started = time.time()
    proc = await asyncio.create_subprocess_shell(
        command,
        cwd=str(ctx.workspace_root()),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_clean_env(),
        start_new_session=True,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        timed_out = False
    except asyncio.TimeoutError:
        timed_out = True
        try:
            os.killpg(os.getpgid(proc.pid), 15)
        except Exception:
            proc.kill()
        stdout, stderr = await proc.communicate()
    return {
        "success": (proc.returncode == 0 and not timed_out),
        "command": command,
        "exit_code": proc.returncode,
        "stdout": _truncate(stdout.decode("utf-8", errors="replace")),
        "stderr": _truncate(stderr.decode("utf-8", errors="replace")),
        "duration_seconds": round(time.time() - started, 2),
        "timed_out": timed_out,
    }


# ---------------------------------------------------------------------------
# Web / HTTP
# ---------------------------------------------------------------------------
async def _web_search(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    from engine.web_tools import web_search

    result = await web_search(args["query"], max_results=int(args.get("max_results", 5)))
    if result.get("results"):
        for item in result["results"]:
            item["url"] = str(item.get("url", ""))[:1000]
    return result


async def _web_fetch(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    from engine.web_tools import web_fetch

    url = args["url"]
    if not is_safe_public_url(url, allow_private=_allow_private_fetch()):
        return {"success": False, "error": "URL blocked: only public http(s) hosts are allowed.", "url": url}
    return await web_fetch(url, max_chars=int(args.get("max_chars", 8000)))


async def _http_request(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    import httpx

    url = args["url"]
    if not is_safe_public_url(url, allow_private=_allow_private_fetch()):
        return {"success": False, "error": "URL blocked: only public http(s) hosts are allowed."}
    if ctx.simulate:
        return {"success": True, "simulated": True, "method": args.get("method", "GET"), "url": url}
    headers = args.get("headers") or {}
    clean_headers = {str(k): str(v) for k, v in headers.items() if str(k).lower() not in ("host", "content-length")}
    async with httpx.AsyncClient(timeout=min(float(args.get("timeout", 30)), 60), follow_redirects=False) as client:
        res = await client.request(
            args.get("method", "GET").upper(), url, headers=clean_headers,
            content=args.get("body"),
        )
    return {
        "success": res.status_code < 400,
        "status_code": res.status_code,
        "headers": dict(list(res.headers.items())[:30]),
        "body": _truncate(res.text),
    }


# ---------------------------------------------------------------------------
# Knowledge / memory / notifications
# ---------------------------------------------------------------------------
async def _knowledge_add(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    from services.knowledge import KnowledgeService

    source = KnowledgeService.add_text(
        title=args["title"], content=args["content"],
        source_type=args.get("source_type", "note"),
        uri=args.get("uri"), tags=args.get("tags") or [],
        metadata={"added_by_agent": ctx.agent_id, "run_id": ctx.run_id},
        private=bool(args.get("private", False)),
        profile=args.get("profile", "personal"),
    )
    return {"success": True, "source_id": source["id"], "chunks": source["chunk_count"]}


async def _knowledge_search(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    from services.knowledge import KnowledgeService

    results = KnowledgeService.search(
        args["query"], limit=int(args.get("limit", 8)),
        source_type=args.get("source_type"), profile=args.get("profile"),
        private=args.get("private"),
    )
    return {"success": True, "results": results}


async def _memory_remember(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    value = str(args["value"])
    if SafetyService.offline_mode():
        pass
    from services.crypto import looks_like_secret

    if looks_like_secret(value):
        return {
            "success": False,
            "error": "Refused to store a value that looks like a credential. Secrets never enter memory.",
        }
    from memory.dossier import UserDossier

    key = re.sub(r"[^a-z0-9_\- ]", "", str(args["key"]).strip().lower())[:80]
    if not key:
        return {"success": False, "error": "Invalid memory key."}
    UserDossier.set_fact(key, value[:4000], category=args.get("category", "general"), confidence=float(args.get("confidence", 1.0)))
    return {"success": True, "key": key, "category": args.get("category", "general")}


async def _memory_forget(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    from memory.dossier import UserDossier

    key = str(args["key"]).strip().lower()
    UserDossier.delete_fact(key)
    return {"success": True, "forgotten": key}


async def _notifications_send(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    from services.events import NotificationService

    nid = NotificationService.create(
        title=args["title"], body=args.get("body", ""),
        severity=args.get("severity", "info"), kind="agent",
        action_url=args.get("action_url"),
        data={"agent_id": ctx.agent_id, "run_id": ctx.run_id},
    )
    return {"success": True, "notification_id": nid}


# ---------------------------------------------------------------------------
# Desktop (existing pilot, gated behind explicit approval)
# ---------------------------------------------------------------------------
async def _desktop_run(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    if ctx.simulate:
        return {"success": True, "simulated": True, "would_run_desktop_goal": args["goal"]}
    import server

    agent = server.desktop_agent
    steps = []
    async for event in agent.execute_task(args["goal"]):
        steps.append({
            "step": event.get("step"), "action": event.get("action"),
            "detail": event.get("detail"), "thought": event.get("thought"),
            "status": event.get("status"), "screenshot_url": event.get("screenshot_url"),
        })
        if event.get("status") in ("completed", "aborted", "max_steps", "error"):
            break
        if len(steps) >= int(args.get("max_steps", 25)):
            agent.abort()
            break
    return {"success": True, "steps": steps[-30:]}


async def _time_now(ctx: ToolContext, args: Dict[str, Any]) -> Any:
    now = datetime.now(timezone.utc)
    return {"iso": now.isoformat(), "utc": now.strftime("%Y-%m-%d %H:%M:%S UTC"), "timezone": args.get("timezone", "UTC")}


def register_builtin_tools(registry: ToolRegistry) -> None:
    object_schema = {"type": "object", "properties": {}}
    tools = [
        ToolSpec(
            "files.list", "List files", "List files in the approved workspace.",
            {"type": "object", "properties": {
                "path": {"type": "string", "default": "."},
                "recursive": {"type": "boolean", "default": False},
                "max_entries": {"type": "integer", "default": 500, "maximum": 5000},
            }},
            RiskClass.READ_ONLY, _files_list, requires_workspace=True, tags=["files"],
        ),
        ToolSpec(
            "files.read", "Read file", "Read a UTF-8 text file from the approved workspace.",
            {"type": "object", "required": ["path"], "properties": {
                "path": {"type": "string"}, "max_chars": {"type": "integer", "default": 12000},
            }},
            RiskClass.READ_ONLY, _files_read, requires_workspace=True, tags=["files"],
        ),
        ToolSpec(
            "files.write", "Write file", "Create or overwrite a file inside the workspace.",
            {"type": "object", "required": ["path", "content"], "properties": {
                "path": {"type": "string"}, "content": {"type": "string", "maxLength": 500000},
                "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"},
            }},
            RiskClass.REVERSIBLE_WRITE, _files_write, side_effects=True,
            idempotent=False, requires_workspace=True, tags=["files", "write"],
        ),
        ToolSpec(
            "files.delete", "Delete file", "Delete a file or directory after approval.",
            {"type": "object", "required": ["path"], "properties": {
                "path": {"type": "string"}, "recursive": {"type": "boolean", "default": False},
            }},
            RiskClass.DESTRUCTIVE_WRITE, _files_delete, side_effects=True,
            idempotent=False, requires_workspace=True, tags=["files", "destructive"],
        ),
        ToolSpec(
            "files.move", "Move file", "Move or rename a file within the workspace.",
            {"type": "object", "required": ["source", "target"], "properties": {
                "source": {"type": "string"}, "target": {"type": "string"},
                "overwrite": {"type": "boolean", "default": False},
            }},
            RiskClass.REVERSIBLE_WRITE, _files_move, side_effects=True,
            idempotent=False, requires_workspace=True, tags=["files", "write"],
        ),
        ToolSpec(
            "files.mkdir", "Create folder", "Create a directory inside the workspace.",
            {"type": "object", "required": ["path"], "properties": {"path": {"type": "string"}}},
            RiskClass.REVERSIBLE_WRITE, _files_mkdir, side_effects=True,
            idempotent=True, requires_workspace=True, tags=["files", "write"],
        ),
        ToolSpec(
            "shell.run", "Run shell command", "Run an approved shell command in the workspace sandbox.",
            {"type": "object", "required": ["command"], "properties": {
                "command": {"type": "string", "maxLength": 8000},
                "timeout": {"type": "number", "default": 60, "maximum": 180},
            }},
            RiskClass.RESTRICTED, _shell_run, side_effects=True,
            idempotent=False, requires_workspace=True, tags=["shell", "code"],
        ),
        ToolSpec(
            "web.search", "Web search", "Search the public web without an API key.",
            {"type": "object", "required": ["query"], "properties": {
                "query": {"type": "string", "maxLength": 500},
                "max_results": {"type": "integer", "default": 5, "maximum": 10},
            }},
            RiskClass.READ_ONLY, _web_search, network=True, tags=["web", "research"],
        ),
        ToolSpec(
            "web.fetch", "Fetch web page", "Fetch and extract readable text from a public URL.",
            {"type": "object", "required": ["url"], "properties": {
                "url": {"type": "string", "maxLength": 2000},
                "max_chars": {"type": "integer", "default": 8000, "maximum": 40000},
            }},
            RiskClass.READ_ONLY, _web_fetch, network=True, tags=["web", "research"],
        ),
        ToolSpec(
            "http.request", "HTTP request", "Make an approved HTTP request to a public host.",
            {"type": "object", "required": ["url"], "properties": {
                "url": {"type": "string", "maxLength": 2000},
                "method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"], "default": "GET"},
                "headers": object_schema, "body": {"type": "string", "maxLength": 200000},
                "timeout": {"type": "number", "default": 30, "maximum": 60},
            }},
            RiskClass.RESTRICTED, _http_request, network=True, side_effects=True,
            idempotent=False, tags=["web"],
        ),
        ToolSpec(
            "knowledge.add", "Add knowledge", "Add a note or source to the local knowledge vault.",
            {"type": "object", "required": ["title", "content"], "properties": {
                "title": {"type": "string", "maxLength": 300},
                "content": {"type": "string", "maxLength": 500000},
                "source_type": {"type": "string", "default": "note"},
                "uri": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}},
                "private": {"type": "boolean", "default": False},
                "profile": {"type": "string", "default": "personal"},
            }},
            RiskClass.REVERSIBLE_WRITE, _knowledge_add, side_effects=True,
            idempotent=False, tags=["knowledge"],
        ),
        ToolSpec(
            "knowledge.search", "Search knowledge", "Hybrid search over the user's local knowledge vault.",
            {"type": "object", "required": ["query"], "properties": {
                "query": {"type": "string", "maxLength": 1000},
                "limit": {"type": "integer", "default": 8, "maximum": 30},
                "source_type": {"type": "string"}, "profile": {"type": "string"},
                "private": {"type": "boolean"},
            }},
            RiskClass.READ_ONLY, _knowledge_search, tags=["knowledge", "research"],
        ),
        ToolSpec(
            "memory.remember", "Remember fact", "Store an explicit fact in the personal dossier.",
            {"type": "object", "required": ["key", "value"], "properties": {
                "key": {"type": "string", "maxLength": 80},
                "value": {"type": "string", "maxLength": 4000},
                "category": {"type": "string", "default": "general"},
                "confidence": {"type": "number", "default": 1.0},
            }},
            RiskClass.REVERSIBLE_WRITE, _memory_remember, side_effects=True, tags=["memory"],
        ),
        ToolSpec(
            "memory.forget", "Forget fact", "Delete a fact from the personal dossier.",
            {"type": "object", "required": ["key"], "properties": {"key": {"type": "string", "maxLength": 80}}},
            RiskClass.REVERSIBLE_WRITE, _memory_forget, side_effects=True, tags=["memory"],
        ),
        ToolSpec(
            "notifications.send", "Send local notification", "Create a local notification for the user.",
            {"type": "object", "required": ["title"], "properties": {
                "title": {"type": "string", "maxLength": 300},
                "body": {"type": "string", "maxLength": 4000},
                "severity": {"type": "string", "enum": ["info", "warning", "error", "critical"], "default": "info"},
                "action_url": {"type": "string"},
            }},
            RiskClass.REVERSIBLE_WRITE, _notifications_send, side_effects=True, tags=["notifications"],
        ),
        ToolSpec(
            "desktop.run", "Desktop pilot", "Operate the Linux desktop (Wayland) with visual verification.",
            {"type": "object", "required": ["goal"], "properties": {
                "goal": {"type": "string", "maxLength": 2000},
                "max_steps": {"type": "integer", "default": 15, "maximum": 40},
            }},
            RiskClass.RESTRICTED, _desktop_run, side_effects=True,
            idempotent=False, timeout_seconds=300, requires_workspace=False, tags=["desktop", "browser"],
        ),
        ToolSpec(
            "time.now", "Current time", "Return the current UTC time.",
            {"type": "object", "properties": {"timezone": {"type": "string"}}},
            RiskClass.READ_ONLY, _time_now, tags=["utility"],
        ),
    ]
    for spec in tools:
        registry.register(spec)
