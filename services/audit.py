"""
Append-only, hash-chained audit timeline.

Every prompt, tool call, policy decision, approval, connector action and agent
step can be recorded here.  Rows are protected by SQLite triggers and chained
with SHA-256 so tampering is detectable via ``verify_chain``.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from memory.database import db_session
from services.crypto import redact_obj, redact_text

_lock = threading.Lock()

_HASHED_FIELDS = (
    "created_at", "category", "action", "actor", "status", "risk_class",
    "agent_id", "task_id", "run_id", "tool_name", "connector_id",
    "approval_id", "summary", "data", "prev_hash",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical(record: Dict[str, Any]) -> str:
    payload = {k: record.get(k) for k in _HASHED_FIELDS}
    # On write, `data` is the canonical JSON string.  On read, the row helper
    # exposes it as a parsed object; serialize it identically before hashing.
    if not isinstance(payload.get("data"), str):
        payload["data"] = json.dumps(payload.get("data") or {}, sort_keys=True, default=str)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _compute_hash(record: Dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(record).encode("utf-8")).hexdigest()


class AuditService:
    @classmethod
    def log(
        cls,
        category: str,
        action: str,
        *,
        actor: str = "system",
        status: str = "ok",
        risk_class: Optional[str] = None,
        agent_id: Optional[str] = None,
        task_id: Optional[str] = None,
        run_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        connector_id: Optional[str] = None,
        approval_id: Optional[str] = None,
        summary: str = "",
        data: Optional[Any] = None,
        activity: bool = True,
        activity_title: Optional[str] = None,
        severity: str = "info",
    ) -> Dict[str, Any]:
        created_at = _now()
        safe_data = redact_obj(data if data is not None else {})
        data_str = json.dumps(safe_data, sort_keys=True, default=str)
        summary = redact_text(summary or "")[:1000]

        with _lock:
            with db_session() as conn:
                last = conn.execute(
                    "SELECT hash FROM audit_events ORDER BY id DESC LIMIT 1"
                ).fetchone()
                prev_hash = last["hash"] if last else "GENESIS"
                record = {
                    "created_at": created_at,
                    "category": category,
                    "action": action,
                    "actor": actor,
                    "status": status,
                    "risk_class": risk_class,
                    "agent_id": agent_id,
                    "task_id": task_id,
                    "run_id": run_id,
                    "tool_name": tool_name,
                    "connector_id": connector_id,
                    "approval_id": approval_id,
                    "summary": summary,
                    "data": data_str,
                    "prev_hash": prev_hash,
                }
                digest = _compute_hash(record)
                cur = conn.execute(
                    """
                    INSERT INTO audit_events (
                        created_at, category, action, actor, status, risk_class,
                        agent_id, task_id, run_id, tool_name, connector_id,
                        approval_id, summary, data, prev_hash, hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        created_at, category, action, actor, status, risk_class,
                        agent_id, task_id, run_id, tool_name, connector_id,
                        approval_id, summary, data_str, prev_hash, digest,
                    ),
                )
                event_id = cur.lastrowid

        event = {
            "id": event_id,
            "created_at": created_at,
            "category": category,
            "action": action,
            "actor": actor,
            "status": status,
            "risk_class": risk_class,
            "agent_id": agent_id,
            "task_id": task_id,
            "run_id": run_id,
            "tool_name": tool_name,
            "connector_id": connector_id,
            "approval_id": approval_id,
            "summary": summary,
            "data": safe_data,
            "prev_hash": prev_hash,
            "hash": digest,
        }

        if activity:
            try:
                from services.events import ActivityService

                ActivityService.record(
                    kind=category,
                    title=activity_title or f"{category}: {action}",
                    detail=summary,
                    ref_type="audit",
                    ref_id=str(event_id),
                    agent_id=agent_id,
                    severity=severity if status in ("ok", "approved") else ("warning" if status in ("denied", "approval_required") else "error"),
                    data={"run_id": run_id, "task_id": task_id, "tool": tool_name},
                )
            except Exception:
                pass
        return event

    @classmethod
    def list_events(
        cls,
        *,
        category: Optional[str] = None,
        status: Optional[str] = None,
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        for field, value in (
            ("category", category), ("status", status), ("run_id", run_id),
            ("task_id", task_id), ("agent_id", agent_id), ("tool_name", tool_name),
        ):
            if value:
                clauses.append(f"{field} = ?")
                params.append(value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([max(1, min(limit, 500)), max(0, offset)])
        with db_session() as conn:
            rows = conn.execute(
                f"SELECT * FROM audit_events {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                params,
            ).fetchall()
        return [cls._row_to_dict(row) for row in rows]

    @classmethod
    def get(cls, event_id: int) -> Optional[Dict[str, Any]]:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM audit_events WHERE id = ?", (event_id,)).fetchone()
        return cls._row_to_dict(row) if row else None

    @classmethod
    def timeline_for_run(cls, run_id: str) -> List[Dict[str, Any]]:
        with db_session() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_events WHERE run_id = ? ORDER BY id ASC", (run_id,)
            ).fetchall()
        return [cls._row_to_dict(row) for row in rows]

    @classmethod
    def verify_chain(cls) -> Dict[str, Any]:
        with db_session() as conn:
            rows = conn.execute("SELECT * FROM audit_events ORDER BY id ASC").fetchall()
        prev = "GENESIS"
        for row in rows:
            record = cls._row_to_dict(row)
            if record["prev_hash"] != prev:
                return {
                    "ok": False,
                    "events": len(rows),
                    "broken_at": record["id"],
                    "reason": "prev_hash mismatch",
                }
            expected = _compute_hash(record)
            if expected != record["hash"]:
                return {
                    "ok": False,
                    "events": len(rows),
                    "broken_at": record["id"],
                    "reason": "hash mismatch",
                }
            prev = record["hash"]
        return {"ok": True, "events": len(rows), "head": prev}

    @classmethod
    def export(cls) -> Dict[str, Any]:
        with db_session() as conn:
            rows = conn.execute("SELECT * FROM audit_events ORDER BY id ASC").fetchall()
        return {
            "version": 2,
            "exported_at": _now(),
            "verification": cls.verify_chain(),
            "events": [cls._row_to_dict(row) for row in rows],
        }

    @staticmethod
    def _row_to_dict(row) -> Dict[str, Any]:
        data = row["data"]
        try:
            parsed = json.loads(data) if data else {}
        except Exception:
            parsed = {"raw": data}
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "category": row["category"],
            "action": row["action"],
            "actor": row["actor"],
            "status": row["status"],
            "risk_class": row["risk_class"],
            "agent_id": row["agent_id"],
            "task_id": row["task_id"],
            "run_id": row["run_id"],
            "tool_name": row["tool_name"],
            "connector_id": row["connector_id"],
            "approval_id": row["approval_id"],
            "summary": row["summary"],
            "data": parsed,
            "prev_hash": row["prev_hash"],
            "hash": row["hash"],
        }
