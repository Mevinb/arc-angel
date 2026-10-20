"""Scheduled, webhook and event-driven automations."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from memory.database import db_session
from services.audit import AuditService
from services.crypto import redact_obj, redact_text
from services.events import ActivityService, NotificationService

logger = logging.getLogger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime] = None) -> str:
    return (value or _now()).isoformat()


class AutomationService:
    _scheduler: Optional[asyncio.Task] = None
    _stop = False
    _lock = asyncio.Lock()

    # ------------------------------------------------------------------- CRUD
    @classmethod
    def create(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        name = (payload.get("name") or "").strip()
        if not name:
            raise ValueError("Automation name is required")
        trigger_type = payload.get("trigger_type") or "schedule"
        if trigger_type not in ("schedule", "webhook", "event"):
            raise ValueError("trigger_type must be schedule, webhook or event")
        action_type = payload.get("action_type") or "create_task"
        if action_type not in ("create_task", "notify", "research", "backup", "run_existing_task"):
            raise ValueError("Unknown action_type")
        automation_id = f"au_{uuid.uuid4().hex[:12]}"
        now = _now()
        trigger_config = payload.get("trigger_config") or {}
        next_run = cls._next_run(trigger_type, trigger_config, now)
        with db_session() as conn:
            conn.execute(
                """
                INSERT INTO automations (
                    id, name, description, enabled, trigger_type, trigger_config, action_type,
                    action_config, agent_id, policy_id, next_run_at, run_count, webhook_secret,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    automation_id, name[:200], (payload.get("description") or "")[:1000],
                    1 if payload.get("enabled", True) else 0, trigger_type, json.dumps(trigger_config),
                    action_type, json.dumps(payload.get("action_config") or {}),
                    payload.get("agent_id") or "orchestrator", payload.get("policy_id"),
                    next_run, uuid.uuid4().hex, _iso(now), _iso(now),
                ),
            )
        AuditService.log(
            "automation", "automation.created", actor="user",
            summary=f"Automation created: {name}",
            data={"automation_id": automation_id, "trigger": trigger_type, "action": action_type},
        )
        return cls.get(automation_id)  # type: ignore[return-value]

    @classmethod
    def get(cls, automation_id: str) -> Optional[Dict[str, Any]]:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM automations WHERE id=?", (automation_id,)).fetchone()
            if not row:
                return None
            runs = conn.execute(
                "SELECT * FROM automation_runs WHERE automation_id=? ORDER BY created_at DESC LIMIT 10",
                (automation_id,),
            ).fetchall()
            states = conn.execute(
                "SELECT key, value FROM automation_states WHERE automation_id=?", (automation_id,)
            ).fetchall()
        result = cls._row(row)
        result["runs"] = [dict(r) for r in runs]
        result["state"] = {s["key"]: s["value"] for s in states}
        return result

    @classmethod
    def list(cls) -> List[Dict[str, Any]]:
        with db_session() as conn:
            rows = conn.execute("SELECT * FROM automations ORDER BY created_at DESC").fetchall()
        return [cls._row(r) for r in rows]

    @classmethod
    def update(cls, automation_id: str, patch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        current = cls.get(automation_id)
        if not current:
            return None
        allowed = {
            "name", "description", "enabled", "trigger_type", "trigger_config",
            "action_type", "action_config", "agent_id", "policy_id",
        }
        sets, params = [], []
        for key, value in patch.items():
            if key not in allowed:
                continue
            if key in ("trigger_config", "action_config"):
                value = json.dumps(value)
            if key == "enabled":
                value = 1 if value else 0
            sets.append(f"{key}=?")
            params.append(value)
        if not sets:
            return current
        trigger_type = patch.get("trigger_type", current["trigger_type"])
        trigger_config = patch.get("trigger_config", current["trigger_config"])
        next_run = cls._next_run(trigger_type, trigger_config, _now())
        sets.append("next_run_at=?")
        params.append(next_run)
        sets.append("updated_at=?")
        params.extend([_iso(), automation_id])
        with db_session() as conn:
            conn.execute(f"UPDATE automations SET {', '.join(sets)} WHERE id=?", params)
        AuditService.log(
            "automation", "automation.updated", actor="user",
            summary=f"Automation updated: {current['name']}", data={"fields": sorted(patch)},
        )
        return cls.get(automation_id)

    @classmethod
    def delete(cls, automation_id: str) -> bool:
        automation = cls.get(automation_id)
        if not automation:
            return False
        with db_session() as conn:
            conn.execute("DELETE FROM automation_states WHERE automation_id=?", (automation_id,))
            conn.execute("DELETE FROM automation_runs WHERE automation_id=?", (automation_id,))
            conn.execute("DELETE FROM automations WHERE id=?", (automation_id,))
        AuditService.log("automation", "automation.deleted", actor="user", summary=f"Automation deleted: {automation['name']}")
        return True

    # ------------------------------------------------------------------- runs
    @classmethod
    async def run(cls, automation_id: str, *, trigger: str = "schedule", payload: Any = None) -> Dict[str, Any]:
        automation = cls.get(automation_id)
        if not automation:
            raise ValueError("Automation not found")
        if not automation["enabled"] and trigger != "manual":
            raise ValueError("Automation is disabled")
        run_id = f"ar_{uuid.uuid4().hex[:12]}"
        with db_session() as conn:
            conn.execute(
                "INSERT INTO automation_runs (id, automation_id, created_at, status, detail) VALUES (?, ?, ?, 'running', '')",
                (run_id, automation_id, _iso()),
            )
        result: Dict[str, Any] = {"run_id": run_id, "automation_id": automation_id, "trigger": trigger}
        try:
            action = automation["action_type"]
            cfg = automation["action_config"]
            if action == "create_task":
                from services.tasks import TaskService

                title = cfg.get("title") or automation["name"]
                goal = cfg.get("goal") or automation["description"] or title
                if payload:
                    goal = f"{goal}\n\nTrigger payload:\n{json.dumps(payload, default=str)[:4000]}"
                task = TaskService.create_task(
                    title=title, goal=goal, agent_id=automation["agent_id"],
                    budget=cfg.get("budget"), policy_id=automation["policy_id"],
                    origin=f"automation:{automation_id}", tags=[automation_id],
                    auto_start=bool(cfg.get("auto_start", True)),
                )
                result.update({"task_id": task["id"], "detail": f"Created task {task['id']}"})
            elif action == "notify":
                NotificationService.create(
                    title=cfg.get("title") or automation["name"],
                    body=cfg.get("body") or automation["description"] or "Automation triggered",
                    severity=cfg.get("severity", "info"), kind="automation",
                    data={"automation_id": automation_id, "trigger": trigger},
                )
                result["detail"] = "Notification sent"
            elif action == "research":
                from services.research import ResearchService

                question = cfg.get("question") or automation["description"]
                if not question:
                    raise ValueError("Research automation needs a question")
                project = ResearchService.create(question, title=cfg.get("title"))
                ResearchService.start(project["id"], max_sources=int(cfg.get("max_sources", 8)))
                result.update({"project_id": project["id"], "detail": f"Research project {project['id']} started"})
            elif action == "backup":
                from services.backup import BackupService

                backup = BackupService.create(label=cfg.get("label") or automation["name"])
                result.update({"backup_id": backup["id"], "detail": f"Backup {backup['id']} created"})
            elif action == "run_existing_task":
                from services.tasks import TaskService
                from services.run_manager import RunManager

                task_id = cfg.get("task_id")
                if not task_id:
                    raise ValueError("run_existing_task needs task_id")
                run = TaskService.create_run(task_id)
                RunManager.get().start(run["id"])
                result.update({"task_id": task_id, "run_id": run["id"], "detail": f"Started run {run['id']}"})
            status = "completed"
        except Exception as exc:
            status = "failed"
            result["detail"] = f"{type(exc).__name__}: {exc}"
            logger.warning("Automation %s failed: %s", automation_id, exc)
        next_run = cls._next_run(
            automation["trigger_type"], automation["trigger_config"], _now(),
        ) if automation["trigger_type"] == "schedule" else automation["next_run_at"]
        with db_session() as conn:
            conn.execute(
                "UPDATE automation_runs SET status=?, detail=?, task_id=? WHERE id=?",
                (status, redact_text(result.get("detail", ""))[:4000], result.get("task_id"), run_id),
            )
            conn.execute(
                """
                UPDATE automations SET last_run_at=?, last_status=?, run_count=run_count+1,
                    next_run_at=?, updated_at=? WHERE id=?
                """,
                (_iso(), status, next_run, _iso(), automation_id),
            )
        ActivityService.record(
            kind="automation", title=f"Automation {status}: {automation['name']}",
            detail=result.get("detail", ""), ref_type="automation", ref_id=automation_id,
            severity="info" if status == "completed" else "error",
            data={"run_id": run_id, "trigger": trigger},
        )
        AuditService.log(
            "automation", f"automation.{status}", actor="system",
            status="ok" if status == "completed" else "error",
            summary=f"Automation {status}: {automation['name']}",
            data={**result, "trigger": trigger, "payload": redact_obj(payload)},
        )
        return {**result, "status": status, "detail": result.get("detail", "")}

    @classmethod
    async def tick(cls) -> List[Dict[str, Any]]:
        if _offline():
            return []
        async with cls._lock:
            now = _iso()
            with db_session() as conn:
                rows = conn.execute(
                    "SELECT id FROM automations WHERE enabled=1 AND trigger_type='schedule' AND next_run_at IS NOT NULL AND next_run_at<=?",
                    (now,),
                ).fetchall()
            results = []
            for row in rows:
                try:
                    results.append(await cls.run(row["id"], trigger="schedule"))
                except Exception:
                    logger.exception("Automation run failed")
            try:
                from services.policies import PolicyService

                PolicyService.expire_stale()
            except Exception:
                pass
            return results

    @classmethod
    def start_scheduler(cls, interval: float = 30.0) -> None:
        if cls._scheduler and not cls._scheduler.done():
            return
        cls._stop = False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        cls._scheduler = loop.create_task(cls._scheduler_loop(interval))

    @classmethod
    async def _scheduler_loop(cls, interval: float) -> None:
        while not cls._stop:
            try:
                await cls.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Automation scheduler tick failed")
            await asyncio.sleep(interval)

    @classmethod
    async def stop_scheduler(cls) -> None:
        cls._stop = True
        if cls._scheduler:
            cls._scheduler.cancel()
            try:
                await cls._scheduler
            except (asyncio.CancelledError, Exception):
                pass
            cls._scheduler = None

    @classmethod
    def verify_webhook(cls, automation_id: str, body: bytes, signature: str) -> bool:
        automation = cls.get(automation_id)
        if not automation or automation["trigger_type"] != "webhook":
            return False
        expected = hmac.new(
            automation["webhook_secret"].encode("utf-8"), body, hashlib.sha256
        ).hexdigest()
        supplied = (signature or "").removeprefix("sha256=").strip()
        return hmac.compare_digest(expected, supplied)

    @classmethod
    def set_state(cls, automation_id: str, key: str, value: Any) -> None:
        with db_session() as conn:
            conn.execute(
                """
                INSERT INTO automation_states (automation_id, key, value, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(automation_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (automation_id, key, json.dumps(value, default=str), _iso()),
            )

    # ---------------------------------------------------------------- helpers
    @classmethod
    def _next_run(cls, trigger_type: str, cfg: Dict[str, Any], now: datetime) -> Optional[str]:
        if trigger_type != "schedule":
            return None
        if cfg.get("at"):
            try:
                target = datetime.fromisoformat(str(cfg["at"]))
                if target.tzinfo is None:
                    target = target.replace(tzinfo=timezone.utc)
                if target > now:
                    return target.isoformat()
            except Exception:
                pass
        every = int(cfg.get("every_minutes") or cfg.get("interval_minutes") or 60)
        every = max(1, min(every, 60 * 24 * 30))
        if cfg.get("time_of_day"):
            try:
                hour, minute = [int(x) for x in str(cfg["time_of_day"]).split(":", 1)]
                target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if target <= now:
                    target += timedelta(days=1)
                return target.isoformat()
            except Exception:
                pass
        return (now + timedelta(minutes=every)).isoformat()

    @classmethod
    def _row(cls, row) -> Dict[str, Any]:
        try:
            trigger_config = json.loads(row["trigger_config"] or "{}")
        except Exception:
            trigger_config = {}
        try:
            action_config = json.loads(row["action_config"] or "{}")
        except Exception:
            action_config = {}
        return {
            "id": row["id"], "name": row["name"], "description": row["description"],
            "enabled": bool(row["enabled"]), "trigger_type": row["trigger_type"],
            "trigger_config": trigger_config, "action_type": row["action_type"],
            "action_config": action_config, "agent_id": row["agent_id"],
            "policy_id": row["policy_id"], "next_run_at": row["next_run_at"],
            "last_run_at": row["last_run_at"], "last_status": row["last_status"],
            "run_count": row["run_count"], "webhook_secret": row["webhook_secret"],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }


def _offline() -> bool:
    try:
        from services.safety import SafetyService

        return SafetyService.offline_mode()
    except Exception:
        return False
