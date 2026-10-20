"""Unified activity feed and notification service."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from memory.database import db_session
from services.crypto import redact_obj, redact_text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ActivityService:
    @classmethod
    def record(
        cls,
        *,
        kind: str,
        title: str,
        detail: str = "",
        ref_type: Optional[str] = None,
        ref_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        severity: str = "info",
        data: Optional[Any] = None,
    ) -> int:
        with db_session() as conn:
            cur = conn.execute(
                """
                INSERT INTO activity_events
                (created_at, kind, title, detail, ref_type, ref_id, agent_id, severity, data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now(), kind, redact_text(title or "")[:300],
                    redact_text(detail or "")[:2000], ref_type, ref_id, agent_id,
                    severity, json.dumps(redact_obj(data or {}), default=str),
                ),
            )
            return cur.lastrowid

    @classmethod
    def list(
        cls,
        *,
        kind: Optional[str] = None,
        agent_id: Optional[str] = None,
        severity: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 100,
        before_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        clauses, params = [], []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if severity:
            clauses.append("severity = ?")
            params.append(severity)
        if q:
            clauses.append("(title LIKE ? OR detail LIKE ?)")
            params.extend([f"%{q}%", f"%{q}%"])
        if before_id:
            clauses.append("id < ?")
            params.append(before_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 500)))
        with db_session() as conn:
            rows = conn.execute(
                f"SELECT * FROM activity_events {where} ORDER BY id DESC LIMIT ?", params
            ).fetchall()
        out = []
        for row in rows:
            try:
                data = json.loads(row["data"] or "{}")
            except Exception:
                data = {}
            out.append({
                "id": row["id"], "created_at": row["created_at"], "kind": row["kind"],
                "title": row["title"], "detail": row["detail"],
                "ref_type": row["ref_type"], "ref_id": row["ref_id"],
                "agent_id": row["agent_id"], "severity": row["severity"], "data": data,
            })
        return out

    @classmethod
    def counts(cls) -> Dict[str, int]:
        with db_session() as conn:
            rows = conn.execute(
                "SELECT kind, COUNT(*) AS n FROM activity_events GROUP BY kind"
            ).fetchall()
        return {row["kind"]: row["n"] for row in rows}

    @classmethod
    def clear(cls) -> None:
        with db_session() as conn:
            conn.execute("DELETE FROM activity_events")


class NotificationService:
    @classmethod
    def create(
        cls,
        *,
        title: str,
        body: str = "",
        severity: str = "info",
        kind: str = "system",
        action_url: Optional[str] = None,
        data: Optional[Any] = None,
        activity: bool = True,
    ) -> int:
        now = _now()
        with db_session() as conn:
            cur = conn.execute(
                """
                INSERT INTO notifications
                (created_at, title, body, severity, kind, read, dismissed, action_url, data)
                VALUES (?, ?, ?, ?, ?, 0, 0, ?, ?)
                """,
                (now, redact_text(title)[:300], redact_text(body)[:4000], severity,
                 kind, action_url, json.dumps(redact_obj(data or {}), default=str)),
            )
            nid = cur.lastrowid
        if activity:
            ActivityService.record(
                kind="notification",
                title=title,
                detail=body,
                ref_type="notification",
                ref_id=str(nid),
                severity=severity,
                data={"notification_id": nid, "kind": kind},
            )
        return nid

    @classmethod
    def list(
        cls,
        *,
        unread_only: bool = False,
        include_dismissed: bool = False,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses = []
        if unread_only:
            clauses.append("read = 0")
        if not include_dismissed:
            clauses.append("dismissed = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with db_session() as conn:
            rows = conn.execute(
                f"SELECT * FROM notifications {where} ORDER BY id DESC LIMIT ?",
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [cls._row(row) for row in rows]

    @classmethod
    def unread_count(cls) -> int:
        with db_session() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM notifications WHERE read = 0 AND dismissed = 0"
            ).fetchone()
        return int(row["n"])

    @classmethod
    def mark_read(cls, notification_id: Optional[int] = None, all_: bool = False) -> int:
        with db_session() as conn:
            if all_:
                cur = conn.execute("UPDATE notifications SET read = 1 WHERE read = 0")
            elif notification_id is not None:
                cur = conn.execute("UPDATE notifications SET read = 1 WHERE id = ?", (notification_id,))
            else:
                return 0
            return cur.rowcount

    @classmethod
    def dismiss(cls, notification_id: int) -> bool:
        with db_session() as conn:
            cur = conn.execute(
                "UPDATE notifications SET dismissed = 1, read = 1 WHERE id = ?", (notification_id,)
            )
            return cur.rowcount > 0

    @classmethod
    def clear(cls, *, only_read: bool = False) -> int:
        with db_session() as conn:
            if only_read:
                cur = conn.execute("DELETE FROM notifications WHERE read = 1")
            else:
                cur = conn.execute("DELETE FROM notifications")
            return cur.rowcount

    @staticmethod
    def _row(row) -> Dict[str, Any]:
        try:
            data = json.loads(row["data"] or "{}")
        except Exception:
            data = {}
        return {
            "id": row["id"], "created_at": row["created_at"], "title": row["title"],
            "body": row["body"], "severity": row["severity"], "kind": row["kind"],
            "read": bool(row["read"]), "dismissed": bool(row["dismissed"]),
            "action_url": row["action_url"], "data": data,
        }
