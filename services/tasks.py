"""Goal-based task board and resumable run persistence."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import WORKSPACES_DIR
from memory.database import db_session
from services.audit import AuditService
from services.crypto import redact_obj, redact_text
from services.events import ActivityService, NotificationService

TASK_STATUSES = (
    "queued", "waiting", "running", "paused", "awaiting_approval",
    "completed", "partially_completed", "failed", "cancelled", "interrupted",
)
TERMINAL_TASK = {"completed", "partially_completed", "failed", "cancelled"}
RUN_STATUSES = (
    "queued", "running", "paused", "awaiting_approval",
    "completed", "partially_completed", "failed", "cancelled", "interrupted",
)
TERMINAL_RUN = {"completed", "partially_completed", "failed", "cancelled"}

TASK_TRANSITIONS = {
    "queued": {"running", "waiting", "paused", "cancelled", "interrupted", "failed"},
    "waiting": {"queued", "cancelled", "failed"},
    "running": {"running", "completed", "partially_completed", "failed", "paused", "awaiting_approval", "cancelled", "interrupted"},
    "paused": {"running", "queued", "cancelled", "interrupted"},
    "awaiting_approval": {"running", "cancelled", "failed", "interrupted"},
    "interrupted": {"queued", "running", "cancelled"},
    "failed": {"queued", "cancelled"},
    "completed": set(),
    "partially_completed": set(),
    "cancelled": set(),
}
RUN_TRANSITIONS = {
    "queued": {"running", "cancelled", "interrupted", "failed"},
    "running": {"running", "completed", "partially_completed", "failed", "paused", "awaiting_approval", "cancelled", "interrupted"},
    "paused": {"running", "cancelled", "interrupted"},
    "awaiting_approval": {"running", "cancelled", "failed", "interrupted"},
    "interrupted": {"running", "cancelled", "failed"},
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}

DEFAULT_BUDGET = {
    "max_steps": 24,
    "max_seconds": 1800,
    "max_tool_calls": 60,
    "max_cost_usd": 2.0,
    "auto_resume": True,
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(value, default):
    if not value:
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


class TaskService:
    # ------------------------------------------------------------------ tasks
    @classmethod
    def create_task(
        cls,
        *,
        title: str,
        goal: str,
        agent_id: Optional[str] = None,
        priority: int = 3,
        due_at: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        budget: Optional[Dict[str, Any]] = None,
        policy_id: Optional[str] = None,
        session_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        origin: str = "user",
        tags: Optional[List[str]] = None,
        workspace_dir: Optional[str] = None,
        auto_start: bool = False,
    ) -> Dict[str, Any]:
        title = (title or goal or "Untitled task").strip()[:300]
        goal = (goal or title).strip()
        if not goal:
            raise ValueError("Task goal cannot be empty")
        task_id = f"t_{uuid.uuid4().hex[:12]}"
        if workspace_dir:
            from services.files import _resolve_root

            ws = _resolve_root(workspace_dir)
            ws.mkdir(parents=True, exist_ok=True)
        else:
            ws = (WORKSPACES_DIR / f"v2_{task_id}").resolve()
            ws.mkdir(parents=True, exist_ok=True)
        deps = [d for d in (dependencies or []) if d]
        for dep in deps:
            if not cls.get_task(dep):
                raise ValueError(f"Unknown dependency task: {dep}")
        status = "waiting" if cls._has_incomplete_deps(deps) else "queued"
        merged_budget = {**DEFAULT_BUDGET, **(budget or {})}
        now = _now()
        with db_session() as conn:
            conn.execute(
                """
                INSERT INTO v2_tasks (
                    id, title, goal, agent_id, status, priority, due_at, parent_id,
                    dependencies, budget, policy_id, workspace_dir, session_id, origin,
                    tags, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id, title, goal, agent_id, status, int(priority), due_at, parent_id,
                    json.dumps(deps), json.dumps(merged_budget), policy_id, str(ws),
                    session_id, origin, json.dumps(tags or []), now, now,
                ),
            )
        task = cls.get_task(task_id)
        AuditService.log(
            "task", "task.created", actor="user" if origin == "user" else "agent",
            agent_id=agent_id, task_id=task_id, summary=f"Created task '{title}'",
            data={"goal": goal, "budget": merged_budget, "dependencies": deps},
            activity_title=f"Task created: {title}",
        )
        if auto_start and status == "queued":
            try:
                from services.run_manager import RunManager

                run = cls.create_run(task_id)
                RunManager.get().start(run["id"])
                task = cls.get_task(task_id) or task
            except Exception as exc:
                # Do not silently claim the task started when the worker could
                # not be launched.  Keep the task queued and surface the error.
                try:
                    with db_session() as conn:
                        conn.execute(
                            "UPDATE v2_tasks SET error=?, updated_at=? WHERE id=?",
                            (redact_text(str(exc))[:4000], _now(), task_id),
                        )
                except Exception:
                    pass
                NotificationService.create(
                    title="Task could not start",
                    body=f"{title}: {exc}", severity="error", kind="task",
                    action_url="/platform#tasks", data={"task_id": task_id},
                )
                task = cls.get_task(task_id) or task
        return task  # type: ignore[return-value]

    @classmethod
    def get_task(cls, task_id: str) -> Optional[Dict[str, Any]]:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM v2_tasks WHERE id = ?", (task_id,)).fetchone()
            if not row:
                return None
            run_rows = conn.execute(
                "SELECT * FROM v2_runs WHERE task_id = ? ORDER BY created_at DESC", (task_id,)
            ).fetchall()
        task = cls._task_row(row)
        task["runs"] = [cls._run_row(r) for r in run_rows]
        task["latest_run"] = task["runs"][0] if task["runs"] else None
        return task

    @classmethod
    def list_tasks(
        cls,
        *,
        status: Optional[str] = None,
        agent_id: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        include_archived: bool = True,
    ) -> Dict[str, Any]:
        clauses, params = [], []
        if status:
            if status == "active":
                clauses.append("status NOT IN ('completed','partially_completed','failed','cancelled')")
            else:
                clauses.append("status = ?")
                params.append(status)
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if q:
            clauses.append("(title LIKE ? OR goal LIKE ?)")
            params.extend([f"%{q}%", f"%{q}%"])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with db_session() as conn:
            total = conn.execute(f"SELECT COUNT(*) AS n FROM v2_tasks {where}", params).fetchone()["n"]
            rows = conn.execute(
                f"SELECT * FROM v2_tasks {where} ORDER BY priority DESC, created_at DESC LIMIT ? OFFSET ?",
                [*params, max(1, min(limit, 200)), max(0, offset)],
            ).fetchall()
            tasks = []
            for row in rows:
                task = cls._task_row(row)
                latest = conn.execute(
                    "SELECT id, status, started_at, finished_at FROM v2_runs WHERE task_id=? ORDER BY created_at DESC LIMIT 1",
                    (row["id"],),
                ).fetchone()
                task["latest_run"] = dict(latest) if latest else None
                tasks.append(task)
        return {"total": total, "tasks": tasks}

    @classmethod
    def update_task(cls, task_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed = {"title", "goal", "priority", "due_at", "status", "policy_id", "tags", "result", "error"}
        current = cls.get_task(task_id)
        if not current:
            return None
        sets, params = [], []
        for key, value in patch.items():
            if key not in allowed:
                continue
            if key == "status":
                cls._validate_task_transition(current["status"], value)
            if key in ("tags",):
                value = json.dumps(value)
            sets.append(f"{key} = ?")
            params.append(value)
        if not sets:
            return current
        sets.append("updated_at = ?")
        params.append(_now())
        params.append(task_id)
        with db_session() as conn:
            conn.execute(f"UPDATE v2_tasks SET {', '.join(sets)} WHERE id = ?", params)
        return cls.get_task(task_id)

    @classmethod
    def delete_task(cls, task_id: str) -> bool:
        task = cls.get_task(task_id)
        if not task:
            return False
        if task["status"] in ("running", "awaiting_approval", "paused"):
            raise ValueError("Pause or cancel the task before deleting it")
        with db_session() as conn:
            run_ids = [r["id"] for r in conn.execute("SELECT id FROM v2_runs WHERE task_id=?", (task_id,)).fetchall()]
            for rid in run_ids:
                conn.execute("DELETE FROM run_events WHERE run_id = ?", (rid,))
                conn.execute("DELETE FROM run_steps WHERE run_id = ?", (rid,))
            conn.execute("DELETE FROM v2_runs WHERE task_id = ?", (task_id,))
            conn.execute("DELETE FROM v2_tasks WHERE id = ?", (task_id,))
        ActivityService.record(
            kind="task", title=f"Task deleted: {task['title']}", ref_type="task", ref_id=task_id,
            severity="warning",
        )
        return True

    # ------------------------------------------------------------------- runs
    @classmethod
    def create_run(
        cls,
        task_id: str,
        *,
        budget_override: Optional[Dict[str, Any]] = None,
        replay_of: Optional[str] = None,
        attempt: Optional[int] = None,
    ) -> Dict[str, Any]:
        task = cls.get_task(task_id)
        if not task:
            raise ValueError("Task not found")
        if task["status"] in ("running", "awaiting_approval", "paused"):
            raise ValueError("Task already has an active run; resume it instead")
        active = [
            r for r in task.get("runs", [])
            if r["status"] not in TERMINAL_RUN and r["status"] != "interrupted"
        ]
        if active:
            raise ValueError("Task already has an unfinished run")
        if cls._has_incomplete_deps(task.get("dependencies") or []):
            raise ValueError("Task dependencies are not complete")
        run_id = f"r_{uuid.uuid4().hex[:12]}"
        budget = {**(task.get("budget") or DEFAULT_BUDGET), **(budget_override or {})}
        checkpoint = {
            "steps_done": 0, "plan": [], "scratchpad": {}, "artifacts": [],
            "usage": {"tool_calls": 0, "steps": 0, "tool_seconds": 0},
            "pending_approval": None, "history": [],
        }
        attempts = attempt
        if attempts is None:
            with db_session() as conn:
                row = conn.execute("SELECT COUNT(*) AS n FROM v2_runs WHERE task_id=?", (task_id,)).fetchone()
            attempts = int(row["n"]) + 1
        now = _now()
        with db_session() as conn:
            conn.execute(
                """
                INSERT INTO v2_runs (
                    id, task_id, agent_id, status, attempt, checkpoint, policy_id,
                    budget, replay_of, created_at
                ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id, task_id, task.get("agent_id"), attempts, json.dumps(checkpoint),
                    task.get("policy_id"), json.dumps(budget), replay_of, now,
                ),
            )
            conn.execute(
                "UPDATE v2_tasks SET status='queued', error=NULL, updated_at=? WHERE id=?",
                (now, task_id),
            )
        cls.add_event(run_id, "run.created", {
            "task_id": task_id, "agent_id": task.get("agent_id"),
            "budget": budget, "attempt": attempts, "replay_of": replay_of,
        })
        AuditService.log(
            "run", "run.created", actor="user", agent_id=task.get("agent_id"),
            task_id=task_id, run_id=run_id, summary=f"Run #{attempts} created for '{task['title']}'",
            data={"budget": budget, "replay_of": replay_of}, activity_title=f"Run queued: {task['title']}",
        )
        return cls.get_run(run_id)  # type: ignore[return-value]

    @classmethod
    def get_run(cls, run_id: str) -> Optional[Dict[str, Any]]:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM v2_runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return None
            steps = conn.execute(
                "SELECT * FROM run_steps WHERE run_id = ? ORDER BY idx", (run_id,)
            ).fetchall()
            task_row = conn.execute("SELECT * FROM v2_tasks WHERE id = ?", (row["task_id"],)).fetchone()
        run = cls._run_row(row)
        run["steps"] = [cls._step_row(s) for s in steps]
        run["task"] = cls._task_row(task_row) if task_row else None
        return run

    @classmethod
    def list_runs(
        cls, *, task_id: Optional[str] = None, status: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 200)))
        with db_session() as conn:
            rows = conn.execute(
                f"SELECT * FROM v2_runs {where} ORDER BY created_at DESC LIMIT ?", params
            ).fetchall()
        return [cls._run_row(r) for r in rows]

    @classmethod
    def recent_runs(cls, limit: int = 8) -> List[Dict[str, Any]]:
        with db_session() as conn:
            rows = conn.execute(
                """
                SELECT r.*, t.title AS task_title
                FROM v2_runs r LEFT JOIN v2_tasks t ON t.id = r.task_id
                ORDER BY r.created_at DESC LIMIT ?
                """,
                (max(1, min(limit, 50)),),
            ).fetchall()
        runs = []
        for row in rows:
            item = cls._run_row(row)
            item["title"] = row["task_title"] or item["task_id"]
            runs.append(item)
        return runs

    @classmethod
    def update_run_status(cls, run_id: str, status: str, *, error: Optional[str] = None, result: Optional[str] = None) -> Dict[str, Any]:
        run = cls.get_run(run_id)
        if not run:
            raise ValueError("Run not found")
        cls._validate_run_transition(run["status"], status)
        now = _now()
        sets = ["status = ?"]
        params: List[Any] = [status]
        if status == "running" and not run.get("started_at"):
            sets.append("started_at = ?")
            params.append(now)
        if status in TERMINAL_RUN or status == "interrupted":
            sets.append("finished_at = ?")
            params.append(now)
        if error is not None:
            sets.append("error = ?")
            params.append(redact_text(error)[:4000])
        if result is not None:
            sets.append("result = ?")
            params.append(redact_text(result)[:20000])
        sets.append("heartbeat_at = ?")
        params.append(now)
        params.append(run_id)
        with db_session() as conn:
            conn.execute(f"UPDATE v2_runs SET {', '.join(sets)} WHERE id = ?", params)

        task = cls.get_task(run["task_id"])
        if task and status in ("paused", "awaiting_approval", "completed", "interrupted"):
            try:
                cls._validate_task_transition(task["status"], status)
                with db_session() as conn:
                    conn.execute("UPDATE v2_tasks SET status=?, updated_at=? WHERE id=?", (status, now, task["id"]))
            except ValueError:
                pass
        if status in ("completed", "partially_completed"):
            cls._finish_task_from_run(run, status, result or "")
        if status == "failed":
            try:
                with db_session() as conn:
                    conn.execute(
                        "UPDATE v2_tasks SET status='failed', error=?, updated_at=? WHERE id=? AND status NOT IN ('completed','partially_completed','cancelled')",
                        (redact_text(error or "")[:4000], now, run["task_id"]),
                    )
            except Exception:
                pass
        cls.add_event(run_id, f"run.{status}", {
            "status": status, "error": error, "result": redact_obj(result) if result else None,
        })
        if status in ("completed", "partially_completed", "failed", "cancelled", "interrupted"):
            severity = "info" if status in ("completed", "partially_completed") else "error"
            ActivityService.record(
                kind="run", title=f"Run {status}: {task['title'] if task else run_id}",
                detail=(result or error or "")[:1000], ref_type="run", ref_id=run_id,
                agent_id=run.get("agent_id"), severity=severity,
                data={"task_id": run["task_id"]},
            )
        return cls.get_run(run_id)  # type: ignore[return-value]

    @classmethod
    def save_checkpoint(cls, run_id: str, checkpoint: Dict[str, Any]) -> None:
        with db_session() as conn:
            conn.execute(
                "UPDATE v2_runs SET checkpoint=?, heartbeat_at=? WHERE id=?",
                (json.dumps(redact_obj(checkpoint), default=str), _now(), run_id),
            )

    @classmethod
    def load_checkpoint(cls, run_id: str) -> Dict[str, Any]:
        run = cls.get_run(run_id)
        return (run or {}).get("checkpoint") or {}

    @classmethod
    def record_usage(
        cls, run_id: str, *, tokens_in: int = 0, tokens_out: int = 0,
        cost_usd: float = 0.0, model: Optional[str] = None, provider: Optional[str] = None,
    ) -> None:
        with db_session() as conn:
            conn.execute(
                """
                UPDATE v2_runs SET tokens_in=tokens_in+?, tokens_out=tokens_out+?,
                    cost_usd=cost_usd+?, model=COALESCE(?, model), provider=COALESCE(?, provider)
                WHERE id=?
                """,
                (tokens_in, tokens_out, cost_usd, model, provider, run_id),
            )

    # ----------------------------------------------------------------- steps
    @classmethod
    def add_step(cls, run_id: str, idx: int, title: str, *, input_data: Any = None, tool_name: Optional[str] = None) -> int:
        with db_session() as conn:
            cur = conn.execute(
                """
                INSERT INTO run_steps (run_id, idx, title, status, tool_name, input, started_at)
                VALUES (?, ?, ?, 'running', ?, ?, ?)
                ON CONFLICT(run_id, idx) DO UPDATE SET title=excluded.title, status='running',
                    started_at=excluded.started_at, tool_name=excluded.tool_name, input=excluded.input
                """,
                (run_id, idx, title[:300], tool_name, json.dumps(redact_obj(input_data or {}), default=str), _now()),
            )
            row = conn.execute("SELECT id FROM run_steps WHERE run_id=? AND idx=?", (run_id, idx)).fetchone()
        return int(row["id"])

    @classmethod
    def finish_step(
        cls, run_id: str, idx: int, *, status: str = "completed",
        output: Any = None, error: Optional[str] = None, checkpoint: Any = None,
    ) -> None:
        with db_session() as conn:
            conn.execute(
                """
                UPDATE run_steps SET status=?, output=?, error=?, checkpoint=?, finished_at=?
                WHERE run_id=? AND idx=?
                """,
                (
                    status, json.dumps(redact_obj(output), default=str),
                    redact_text(error or "")[:4000] or None,
                    json.dumps(redact_obj(checkpoint), default=str) if checkpoint is not None else None,
                    _now(), run_id, idx,
                ),
            )

    # ---------------------------------------------------------------- events
    @classmethod
    def add_event(cls, run_id: str, event_type: str, payload: Any = None) -> int:
        with db_session() as conn:
            cur = conn.execute(
                "INSERT INTO run_events (run_id, created_at, type, payload) VALUES (?, ?, ?, ?)",
                (run_id, _now(), event_type, json.dumps(redact_obj(payload or {}), default=str)),
            )
            conn.execute("UPDATE v2_runs SET heartbeat_at=? WHERE id=?", (_now(), run_id))
            return cur.lastrowid

    @classmethod
    def get_events(cls, run_id: str, since_id: int = 0, limit: int = 1000) -> List[Dict[str, Any]]:
        with db_session() as conn:
            rows = conn.execute(
                "SELECT * FROM run_events WHERE run_id=? AND id>? ORDER BY id ASC LIMIT ?",
                (run_id, since_id, max(1, min(limit, 2000))),
            ).fetchall()
        events = []
        for row in rows:
            events.append({
                "id": row["id"], "run_id": row["run_id"], "created_at": row["created_at"],
                "type": row["type"], "payload": _load(row["payload"], {}),
            })
        return events

    @classmethod
    def timeline(cls, run_id: str, limit: int = 1000) -> Dict[str, Any]:
        run = cls.get_run(run_id)
        if not run:
            return {}
        events = cls.get_events(run_id, limit=limit)
        audit = AuditService.timeline_for_run(run_id)
        return {"run": run, "events": events, "steps": run.get("steps", []), "audit": audit}

    # ------------------------------------------------------------- recovery
    @classmethod
    def recover_interrupted(cls) -> List[Dict[str, Any]]:
        with db_session() as conn:
            rows = conn.execute(
                "SELECT * FROM v2_runs WHERE status IN ('running','queued') ORDER BY created_at"
            ).fetchall()
            now = _now()
            conn.execute(
                "UPDATE v2_runs SET status='interrupted', finished_at=?, error='Server restarted during execution' WHERE status IN ('running','queued')",
                (now,),
            )
            conn.execute(
                "UPDATE v2_tasks SET status='interrupted', updated_at=? WHERE status IN ('running','queued','awaiting_approval')",
                (now,),
            )
        recovered = []
        for row in rows:
            run = cls._run_row(row)
            cls.add_event(run["id"], "run.interrupted", {"reason": "server_restart"})
            budget = run.get("budget") or {}
            if budget.get("auto_resume", True):
                recovered.append(run)
        return recovered

    # --------------------------------------------------------------- helpers
    @classmethod
    def _finish_task_from_run(cls, run: Dict[str, Any], status: str, result: str) -> None:
        task = cls.get_task(run["task_id"])
        if not task:
            return
        outcome = "completed" if status == "completed" else "partially_completed"
        desired = "completed" if outcome == "completed" else "partially_completed"
        try:
            cls._validate_task_transition(task["status"], desired)
        except ValueError:
            desired = task["status"]
        with db_session() as conn:
            conn.execute(
                "UPDATE v2_tasks SET status=?, result=?, completed_at=?, updated_at=? WHERE id=?",
                (desired, redact_text(result)[:20000], _now(), _now(), task["id"]),
            )
        NotificationService.create(
            title=f"Task {desired}",
            body=f"{task['title']}" + (f" — {result[:300]}" if result else ""),
            severity="info" if desired == "completed" else "warning",
            kind="task", action_url="/platform#tasks",
            data={"task_id": task["id"], "run_id": run["id"]},
        )
        cls._release_dependents(task["id"])

    @classmethod
    def _release_dependents(cls, completed_task_id: str) -> None:
        with db_session() as conn:
            waiting = conn.execute("SELECT id, dependencies FROM v2_tasks WHERE status='waiting'").fetchall()
            for row in waiting:
                deps = _load(row["dependencies"], [])
                if completed_task_id not in deps:
                    continue
                if cls._has_incomplete_deps(deps):
                    continue
                conn.execute(
                    "UPDATE v2_tasks SET status='queued', updated_at=? WHERE id=?",
                    (_now(), row["id"]),
                )
                cls.add_event_dependency(row["id"])

    @classmethod
    def add_event_dependency(cls, task_id: str) -> None:
        ActivityService.record(
            kind="task", title=f"Task ready: {task_id}",
            detail="All dependencies completed", ref_type="task", ref_id=task_id,
        )

    @classmethod
    def _has_incomplete_deps(cls, deps: List[str]) -> bool:
        if not deps:
            return False
        with db_session() as conn:
            rows = conn.execute(
                f"SELECT status FROM v2_tasks WHERE id IN ({','.join('?' * len(deps))})", deps
            ).fetchall()
        return any(row["status"] not in ("completed", "partially_completed") for row in rows) or len(rows) < len(deps)

    @classmethod
    def _validate_task_transition(cls, current: str, target: str) -> None:
        if current == target:
            return
        if target not in TASK_STATUSES:
            raise ValueError(f"Unknown task status: {target}")
        if target not in TASK_TRANSITIONS.get(current, set()):
            raise ValueError(f"Invalid task transition: {current} -> {target}")

    @classmethod
    def _validate_run_transition(cls, current: str, target: str) -> None:
        if current == target:
            return
        if target not in RUN_STATUSES:
            raise ValueError(f"Unknown run status: {target}")
        if target not in RUN_TRANSITIONS.get(current, set()):
            raise ValueError(f"Invalid run transition: {current} -> {target}")
        if current in TERMINAL_RUN:
            raise ValueError(f"Run is terminal ({current})")

    @classmethod
    def _task_row(cls, row) -> Dict[str, Any]:
        return {
            "id": row["id"], "title": row["title"], "goal": row["goal"],
            "agent_id": row["agent_id"], "status": row["status"],
            "priority": row["priority"], "due_at": row["due_at"],
            "parent_id": row["parent_id"],
            "dependencies": _load(row["dependencies"], []),
            "budget": _load(row["budget"], {}),
            "policy_id": row["policy_id"], "workspace_dir": row["workspace_dir"],
            "session_id": row["session_id"], "origin": row["origin"],
            "tags": _load(row["tags"], []), "result": row["result"],
            "error": row["error"], "created_at": row["created_at"],
            "updated_at": row["updated_at"], "completed_at": row["completed_at"],
        }

    @classmethod
    def _run_row(cls, row) -> Dict[str, Any]:
        return {
            "id": row["id"], "task_id": row["task_id"], "agent_id": row["agent_id"],
            "status": row["status"], "attempt": row["attempt"],
            "checkpoint": _load(row["checkpoint"], {}),
            "policy_id": row["policy_id"], "budget": _load(row["budget"], {}),
            "model": row["model"], "provider": row["provider"],
            "started_at": row["started_at"], "finished_at": row["finished_at"],
            "heartbeat_at": row["heartbeat_at"], "error": row["error"],
            "result": row["result"], "tokens_in": row["tokens_in"],
            "tokens_out": row["tokens_out"], "cost_usd": row["cost_usd"],
            "replay_of": row["replay_of"], "created_at": row["created_at"],
        }

    @classmethod
    def _step_row(cls, row) -> Dict[str, Any]:
        return {
            "id": row["id"], "run_id": row["run_id"], "idx": row["idx"],
            "title": row["title"], "status": row["status"], "tool_name": row["tool_name"],
            "input": _load(row["input"], {}), "output": _load(row["output"], {}),
            "error": row["error"], "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }

    @classmethod
    def stats(cls) -> Dict[str, Any]:
        with db_session() as conn:
            tasks = conn.execute("SELECT status, COUNT(*) AS n FROM v2_tasks GROUP BY status").fetchall()
            runs = conn.execute("SELECT status, COUNT(*) AS n FROM v2_runs GROUP BY status").fetchall()
            usage = conn.execute(
                "SELECT COALESCE(SUM(tokens_in+tokens_out),0) AS tokens, COALESCE(SUM(cost_usd),0) AS cost FROM v2_runs"
            ).fetchone()
        return {
            "tasks": {row["status"]: row["n"] for row in tasks},
            "runs": {row["status"]: row["n"] for row in runs},
            "tokens": usage["tokens"], "cost_usd": round(usage["cost"] or 0, 6),
        }
