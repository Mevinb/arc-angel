"""Agent gallery, custom agents, manifests and task templates."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from memory.database import db_session
from services.audit import AuditService
from services.events import ActivityService
from services.tools import ToolRegistry

DEFAULT_AGENTS = [
    {
        "id": "orchestrator",
        "name": "Atlas",
        "role": "Chief of Staff",
        "icon": "compass",
        "description": "Plans, delegates and drives multi-step work; own the outcome, not just the first step.",
        "system_prompt": (
            "You are Atlas, the user's chief of staff. Break the goal into concrete steps, use tools to do the work, "
            "keep a running checkpoint, and only stop for a genuine approval, missing credential, or hard blocker. "
            "Prefer reversible actions. Report what changed and where the artifacts are. Never invent tool results."
        ),
        "tools": [
            "files.list", "files.read", "files.write", "files.move", "files.mkdir",
            "shell.run", "web.search", "web.fetch", "knowledge.add", "knowledge.search",
            "memory.remember", "notifications.send", "time.now",
        ],
        "risk_profile": "balanced",
        "model_profile": "reasoning",
    },
    {
        "id": "researcher",
        "name": "Scout",
        "role": "Research Analyst",
        "icon": "search",
        "description": "Finds, verifies, compares and cites sources; produces evidence-backed briefs.",
        "system_prompt": (
            "You are Scout, a rigorous research analyst. Search broadly, open primary sources, compare conflicting claims, "
            "and distinguish fact from inference. Every non-obvious claim needs a citation URL or a local source reference. "
            "Save durable findings to the knowledge vault and write the final brief into the workspace."
        ),
        "tools": [
            "web.search", "web.fetch", "knowledge.add", "knowledge.search",
            "files.list", "files.read", "files.write", "memory.remember", "time.now",
        ],
        "risk_profile": "cautious",
        "model_profile": "quality",
    },
    {
        "id": "builder",
        "name": "Forge",
        "role": "Software Engineer",
        "icon": "terminal",
        "description": "Inspects repositories, implements changes, runs tests and fixes failures.",
        "system_prompt": (
            "You are Forge, a pragmatic senior engineer. Read the code before editing, make the smallest correct change, "
            "preserve project conventions, and verify with the project's own tests/build commands. Do not run destructive "
            "commands. If a command fails, diagnose the root cause instead of guessing. Summarize files changed and checks run."
        ),
        "tools": [
            "files.list", "files.read", "files.write", "files.move", "files.mkdir",
            "shell.run", "web.search", "web.fetch", "knowledge.search", "time.now",
        ],
        "risk_profile": "balanced",
        "model_profile": "code",
    },
    {
        "id": "analyst",
        "name": "Prism",
        "role": "Data Analyst",
        "icon": "chart",
        "description": "Turns files and datasets into clean analysis, tables and explainable conclusions.",
        "system_prompt": (
            "You are Prism, a careful data analyst. Inspect structure before computing, state assumptions, check totals and "
            "edge cases, and separate correlation from causation. Produce a reproducible report plus a small machine-readable "
            "artifact when useful. Never overstate certainty."
        ),
        "tools": [
            "files.list", "files.read", "files.write", "knowledge.search",
            "web.search", "web.fetch", "time.now",
        ],
        "risk_profile": "cautious",
        "model_profile": "reasoning",
    },
    {
        "id": "communicator",
        "name": "Quill",
        "role": "Communications",
        "icon": "pen",
        "description": "Drafts clear messages, briefs and documents in the user's voice.",
        "system_prompt": (
            "You are Quill, a precise editor and communications partner. Match the user's requested tone, keep the point "
            "up front, remove filler, and never send externally without explicit approval. Save drafts to the workspace and "
            "flag any factual claim that needs the user's confirmation."
        ),
        "tools": [
            "files.list", "files.read", "files.write", "knowledge.search",
            "web.fetch", "notifications.send", "time.now",
        ],
        "risk_profile": "cautious",
        "model_profile": "balanced",
    },
    {
        "id": "operator",
        "name": "Pilot",
        "role": "Desktop Operator",
        "icon": "monitor",
        "description": "Operates Linux apps and browsers with visual verification for approved GUI tasks.",
        "system_prompt": (
            "You are Pilot, a desktop automation operator. Work in small verified steps, inspect screenshots, and stop if the "
            "screen does not match expectations. Never enter payment details, change credentials, or publish/send anything "
            "without explicit approval. Prefer keyboard shortcuts and accessible controls over blind coordinate clicks."
        ),
        "tools": [
            "desktop.run", "web.search", "web.fetch", "files.read", "files.list",
            "notifications.send", "time.now",
        ],
        "risk_profile": "high",
        "model_profile": "quality",
    },
    {
        "id": "sentinel",
        "name": "Warden",
        "role": "Security Reviewer",
        "icon": "shield",
        "description": "Reviews code, policies and proposed actions for risk and least privilege.",
        "system_prompt": (
            "You are Warden, a defensive security reviewer. Identify concrete risks, attack paths, exposed secrets, and "
            "over-broad permissions. Rank findings by severity and propose the smallest safe remediation. Do not modify "
            "production state; this agent is read-only unless the user explicitly changes its tools."
        ),
        "tools": ["files.list", "files.read", "knowledge.search", "time.now"],
        "risk_profile": "read_only",
        "model_profile": "reasoning",
    },
    {
        "id": "librarian",
        "name": "Index",
        "role": "Knowledge Curator",
        "icon": "book",
        "description": "Organizes notes, sources and transcripts into a searchable, well-cited vault.",
        "system_prompt": (
            "You are Index, the user's knowledge curator. Deduplicate sources, preserve provenance, write concise summaries, "
            "and tag consistently. Never silently discard source material; note conflicts and stale information explicitly."
        ),
        "tools": [
            "knowledge.add", "knowledge.search", "files.list", "files.read",
            "files.write", "web.fetch", "time.now",
        ],
        "risk_profile": "cautious",
        "model_profile": "fast",
    },
]

TASK_TEMPLATES = [
    {
        "id": "daily-brief",
        "title": "Daily brief",
        "description": "Summarize today's priorities, calendar and important messages.",
        "agent_id": "orchestrator",
        "goal": "Create a concise daily brief: today's priorities, time-sensitive items, and the three most important next actions. Check the knowledge vault and available sources; mark anything uncertain.",
        "budget": {"max_steps": 12, "max_seconds": 900},
    },
    {
        "id": "research-brief",
        "title": "Research brief",
        "description": "Investigate a question and produce a cited brief.",
        "agent_id": "researcher",
        "goal": "Research the user's question thoroughly. Compare at least three independent sources where possible, save findings, and write a cited brief with a clear bottom line.",
        "budget": {"max_steps": 20, "max_seconds": 1800},
    },
    {
        "id": "code-review",
        "title": "Code review",
        "description": "Review the current workspace for correctness, security and regressions.",
        "agent_id": "sentinel",
        "goal": "Review the workspace code for actionable correctness, security and regression issues. Report file/line references, severity, and a minimal fix for each finding.",
        "budget": {"max_steps": 16, "max_seconds": 1200},
    },
    {
        "id": "build-feature",
        "title": "Build a feature",
        "description": "Implement, test and verify a scoped feature in the workspace.",
        "agent_id": "builder",
        "goal": "Implement the requested feature in the workspace, following project conventions, then run the relevant tests/build and report changed files and verification results.",
        "budget": {"max_steps": 30, "max_seconds": 2700},
    },
    {
        "id": "organize-notes",
        "title": "Organize notes",
        "description": "Curate the knowledge vault and produce a clean index.",
        "agent_id": "librarian",
        "goal": "Review the knowledge vault for duplicates, missing provenance and stale notes, organize tags, and write a concise index of the most important topics.",
        "budget": {"max_steps": 14, "max_seconds": 1200},
    },
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentService:
    @classmethod
    def ensure_defaults(cls) -> None:
        now = _now()
        with db_session() as conn:
            for agent in DEFAULT_AGENTS:
                exists = conn.execute("SELECT id FROM v2_agents WHERE id=?", (agent["id"],)).fetchone()
                if exists:
                    continue
                conn.execute(
                    """
                    INSERT INTO v2_agents (
                        id, name, description, role, system_prompt, tools, risk_profile,
                        model_profile, icon, is_builtin, enabled, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?)
                    """,
                    (
                        agent["id"], agent["name"], agent["description"], agent["role"],
                        agent["system_prompt"], json.dumps(agent["tools"]),
                        agent["risk_profile"], agent["model_profile"], agent["icon"], now, now,
                    ),
                )

    @classmethod
    def list_agents(cls, enabled_only: bool = False) -> List[Dict[str, Any]]:
        cls.ensure_defaults()
        clause = "WHERE enabled=1" if enabled_only else ""
        with db_session() as conn:
            rows = conn.execute(f"SELECT * FROM v2_agents {clause} ORDER BY is_builtin DESC, name ASC").fetchall()
            agents = []
            for row in rows:
                agent = cls._row(row)
                counts = conn.execute(
                    "SELECT status, COUNT(*) AS n FROM v2_tasks WHERE agent_id=? GROUP BY status",
                    (row["id"],),
                ).fetchall()
                agent["task_counts"] = {c["status"]: c["n"] for c in counts}
                agents.append(agent)
        return agents

    @classmethod
    def get(cls, agent_id: str) -> Optional[Dict[str, Any]]:
        cls.ensure_defaults()
        with db_session() as conn:
            row = conn.execute("SELECT * FROM v2_agents WHERE id=?", (agent_id,)).fetchone()
            if not row:
                return None
            agent = cls._row(row)
            latest = conn.execute(
                "SELECT id, task_id, status, created_at FROM v2_runs WHERE agent_id=? ORDER BY created_at DESC LIMIT 5",
                (agent_id,),
            ).fetchall()
        agent["recent_runs"] = [dict(r) for r in latest]
        return agent

    @classmethod
    def create(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = (payload.get("name") or "").strip()[:80]
        if not name:
            raise ValueError("Agent name is required")
        agent_id = payload.get("id") or re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or f"agent-{uuid.uuid4().hex[:6]}"
        agent_id = re.sub(r"[^a-z0-9_-]", "-", agent_id.lower())[:60]
        tools = cls._validate_tools(payload.get("tools") or [])
        now = _now()
        with db_session() as conn:
            if conn.execute("SELECT 1 FROM v2_agents WHERE id=?", (agent_id,)).fetchone():
                raise ValueError(f"Agent id already exists: {agent_id}")
            conn.execute(
                """
                INSERT INTO v2_agents (id, name, description, role, system_prompt, tools,
                    risk_profile, model_profile, icon, is_builtin, enabled, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    agent_id, name, (payload.get("description") or "")[:1000],
                    (payload.get("role") or "Custom agent")[:120],
                    (payload.get("system_prompt") or "You are a careful, tool-using assistant.")[:12000],
                    json.dumps(tools), payload.get("risk_profile") or "cautious",
                    payload.get("model_profile") or "balanced", payload.get("icon") or "bot",
                    1 if payload.get("enabled", True) else 0, now, now,
                ),
            )
        agent = cls.get(agent_id)
        AuditService.log(
            "agent", "agent.created", actor="user", agent_id=agent_id,
            summary=f"Created agent '{name}'", data={"tools": tools},
        )
        return agent  # type: ignore[return-value]

    @classmethod
    def update(cls, agent_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        agent = cls.get(agent_id)
        if not agent:
            return None
        allowed = {"name", "description", "role", "system_prompt", "tools", "risk_profile", "model_profile", "icon", "enabled"}
        sets, params = [], []
        for key, value in patch.items():
            if key not in allowed:
                continue
            if key == "tools":
                value = json.dumps(cls._validate_tools(value))
            if key == "enabled":
                value = 1 if value else 0
            sets.append(f"{key}=?")
            params.append(value)
        if not sets:
            return agent
        sets.append("updated_at=?")
        params.extend([_now(), agent_id])
        with db_session() as conn:
            conn.execute(f"UPDATE v2_agents SET {', '.join(sets)} WHERE id=?", params)
        AuditService.log(
            "agent", "agent.updated", actor="user", agent_id=agent_id,
            summary=f"Updated agent '{agent['name']}'", data={"fields": sorted(patch)},
        )
        return cls.get(agent_id)

    @classmethod
    def delete(cls, agent_id: str) -> bool:
        agent = cls.get(agent_id)
        if not agent:
            return False
        if agent["is_builtin"]:
            raise ValueError("Built-in agents cannot be deleted; disable or duplicate them instead")
        with db_session() as conn:
            active = conn.execute(
                "SELECT COUNT(*) AS n FROM v2_tasks WHERE agent_id=? AND status IN ('queued','running','paused','awaiting_approval','waiting')",
                (agent_id,),
            ).fetchone()["n"]
            if active:
                raise ValueError("Agent has active tasks; cancel or reassign them first")
            conn.execute("DELETE FROM v2_agents WHERE id=?", (agent_id,))
        AuditService.log("agent", "agent.deleted", actor="user", agent_id=agent_id, summary=f"Deleted agent '{agent['name']}'")
        return True

    @classmethod
    def duplicate(cls, agent_id: str, name: Optional[str] = None) -> Dict[str, Any]:
        agent = cls.get(agent_id)
        if not agent:
            raise ValueError("Agent not found")
        payload = {
            "name": name or f"{agent['name']} copy", "description": agent["description"],
            "role": agent["role"], "system_prompt": agent["system_prompt"],
            "tools": agent["tools"], "risk_profile": agent["risk_profile"],
            "model_profile": agent["model_profile"], "icon": agent["icon"],
        }
        return cls.create(payload)

    @classmethod
    def manifest(cls, agent_id: str) -> Dict[str, Any]:
        agent = cls.get(agent_id)
        if not agent:
            raise ValueError("Agent not found")
        registry = ToolRegistry.get()
        available = {t["name"]: t for t in registry.list_tools()}
        allowed = []
        for tool_name in agent["tools"]:
            if tool_name == "*":
                allowed = sorted(available)
                break
            if tool_name.endswith(".*"):
                prefix = tool_name[:-1]
                allowed.extend(sorted(n for n in available if n.startswith(prefix)))
            elif tool_name in available:
                allowed.append(tool_name)
        allowed = sorted(set(allowed))
        manifests = [available[n] for n in allowed]
        tool_text = "\n".join(
            f"- {t['name']}: {t['description']} Input schema: {json.dumps(t['input_schema'], ensure_ascii=False)}"
            for t in manifests
        ) or "(no tools available)"
        system = (
            f"{agent['system_prompt']}\n\n"
            f"## Runtime contract\n"
            f"- Current UTC time: {_now()}\n"
            f"- Agent: {agent['name']} ({agent['role']})\n"
            f"- Risk profile: {agent['risk_profile']}\n"
            f"- Tools you may call (only these):\n{tool_text}\n\n"
            "Call a tool by returning a single JSON object with keys \"tool\" and \"arguments\". "
            "To finish, return a JSON object with key \"final\" containing the answer. "
            "Never claim success without a tool result. Use relative workspace paths. "
            "If an action needs approval, the runtime pauses and asks the user; do not try to bypass it. "
            "If blocked, explain the blocker and what you tried."
        )
        return {
            "id": agent["id"], "name": agent["name"], "role": agent["role"],
            "system_prompt": system, "tools": manifests,
            "risk_profile": agent["risk_profile"], "model_profile": agent["model_profile"],
        }

    @classmethod
    def resolve_tool(cls, agent_id: str, tool_name: str) -> Optional[Dict[str, Any]]:
        manifest = cls.manifest(agent_id)
        for tool in manifest["tools"]:
            if tool["name"] == tool_name:
                return tool
        return None

    @classmethod
    def templates(cls) -> List[Dict[str, Any]]:
        return TASK_TEMPLATES

    @classmethod
    def create_from_template(
        cls, template_id: str, *, goal_override: Optional[str] = None,
        agent_id: Optional[str] = None, auto_start: bool = False,
    ) -> Dict[str, Any]:
        from services.tasks import TaskService

        template = next((t for t in TASK_TEMPLATES if t["id"] == template_id), None)
        if not template:
            raise ValueError("Unknown task template")
        return TaskService.create_task(
            title=template["title"],
            goal=goal_override or template["goal"],
            agent_id=agent_id or template["agent_id"],
            budget=template.get("budget"),
            origin="template",
            tags=[template_id],
            auto_start=auto_start,
        )

    @classmethod
    def _validate_tools(cls, tools: List[str]) -> List[str]:
        if not isinstance(tools, list):
            raise ValueError("tools must be a list")
        if "*" in tools:
            return ["*"]
        available = {t["name"] for t in ToolRegistry.get().list_tools()}
        cleaned = []
        for name in tools:
            name = str(name).strip()
            if name.endswith(".*"):
                prefix = name[:-1]
                if not any(n.startswith(prefix) for n in available):
                    raise ValueError(f"Unknown tool group: {name}")
                cleaned.append(name)
            elif name in available:
                cleaned.append(name)
            else:
                raise ValueError(f"Unknown tool: {name}")
        return sorted(set(cleaned))

    @classmethod
    def _row(cls, row) -> Dict[str, Any]:
        try:
            tools = json.loads(row["tools"] or "[]")
        except Exception:
            tools = []
        return {
            "id": row["id"], "name": row["name"], "description": row["description"],
            "role": row["role"], "system_prompt": row["system_prompt"], "tools": tools,
            "risk_profile": row["risk_profile"], "model_profile": row["model_profile"],
            "icon": row["icon"], "is_builtin": bool(row["is_builtin"]),
            "enabled": bool(row["enabled"]), "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
