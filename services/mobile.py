"""Secure local mobile companion pairing and lightweight share/command ingress."""
from __future__ import annotations

import hashlib
import secrets
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from memory.database import db_session
from services.audit import AuditService
from services.events import NotificationService
from services.tasks import TaskService


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_PAIR_ATTEMPTS: Dict[str, list[float]] = {}
_PAIR_WINDOW_SECONDS = 600
_PAIR_MAX_ATTEMPTS = 10


class MobileService:
    @classmethod
    def create_pairing(cls, name: str, platform: str = "mobile") -> Dict[str, Any]:
        device_id = f"md_{uuid.uuid4().hex[:12]}"
        code = f"{secrets.randbelow(1_000_000):06d}"
        with db_session() as conn:
            conn.execute(
                """
                INSERT INTO mobile_devices (id, name, platform, pairing_code_hash, token_hash, created_at)
                VALUES (?, ?, ?, ?, NULL, ?)
                """,
                (device_id, (name or "My phone")[:100], (platform or "mobile")[:40], _hash(code), _now()),
            )
        AuditService.log(
            "mobile", "mobile.pairing_created", actor="user",
            summary=f"Mobile pairing created for {name}", data={"device_id": device_id},
        )
        return {
            "device_id": device_id, "pairing_code": code,
            "expires_in_seconds": 600,
            "instructions": "Open /mobile on the phone and enter this code within 10 minutes.",
        }

    @classmethod
    def pair(cls, device_id: str, code: str) -> Dict[str, Any]:
        now_ts = time.time()
        attempts = [ts for ts in _PAIR_ATTEMPTS.get(device_id, []) if now_ts - ts < _PAIR_WINDOW_SECONDS]
        _PAIR_ATTEMPTS[device_id] = attempts
        if len(attempts) >= _PAIR_MAX_ATTEMPTS:
            raise ValueError("Too many pairing attempts; create a new pairing code")
        with db_session() as conn:
            row = conn.execute("SELECT * FROM mobile_devices WHERE id=?", (device_id,)).fetchone()
            if not row:
                raise ValueError("Device pairing not found")
            if row["paired_at"]:
                raise ValueError("Device is already paired")
            created = datetime.fromisoformat(row["created_at"])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) - created > timedelta(minutes=10):
                raise ValueError("Pairing code expired; create a new pairing")
            if not secrets.compare_digest(row["pairing_code_hash"] or "", _hash(code or "")):
                attempts.append(now_ts)
                _PAIR_ATTEMPTS[device_id] = attempts
                raise ValueError("Incorrect pairing code")
            _PAIR_ATTEMPTS.pop(device_id, None)
            token = f"gmk_{secrets.token_urlsafe(36)}"
            conn.execute(
                "UPDATE mobile_devices SET token_hash=?, paired_at=?, last_seen=? WHERE id=?",
                (_hash(token), _now(), _now(), device_id),
            )
        AuditService.log(
            "mobile", "mobile.paired", actor="user",
            summary=f"Mobile device paired: {row['name']}", data={"device_id": device_id},
            activity_title="Mobile device paired",
        )
        return {"device_id": device_id, "name": row["name"], "token": token}

    @classmethod
    def authenticate(cls, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        with db_session() as conn:
            row = conn.execute(
                "SELECT * FROM mobile_devices WHERE token_hash=? AND paired_at IS NOT NULL",
                (_hash(token),),
            ).fetchone()
            if not row:
                return None
            conn.execute("UPDATE mobile_devices SET last_seen=? WHERE id=?", (_now(), row["id"]))
        return {
            "id": row["id"], "name": row["name"], "platform": row["platform"],
            "paired_at": row["paired_at"], "push_enabled": bool(row["push_enabled"]),
        }

    @classmethod
    def list_devices(cls) -> List[Dict[str, Any]]:
        with db_session() as conn:
            rows = conn.execute(
                "SELECT id, name, platform, paired_at, last_seen, push_enabled, created_at FROM mobile_devices ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    @classmethod
    def revoke(cls, device_id: str) -> bool:
        with db_session() as conn:
            count = conn.execute("DELETE FROM mobile_devices WHERE id=?", (device_id,)).rowcount
        _PAIR_ATTEMPTS.pop(device_id, None)
        if count:
            AuditService.log("mobile", "mobile.revoked", actor="user", summary=f"Mobile device revoked: {device_id}")
        return bool(count)

    @classmethod
    def share(cls, device: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        text = (payload.get("text") or payload.get("content") or "").strip()
        url = (payload.get("url") or "").strip()
        if not text and not url:
            raise ValueError("Nothing to share")
        title = (payload.get("title") or text.splitlines()[0] or url)[:200]
        goal = (
            f"Handle this item shared from {device['name']}:\n\n{text}"
            + (f"\n\nURL: {url}" if url else "")
            + "\n\nInspect it, decide whether it needs a task, note or reply, and report what you did."
        )
        task = TaskService.create_task(
            title=f"Shared: {title}", goal=goal, agent_id="orchestrator",
            origin=f"mobile:{device['id']}", tags=["mobile", "share"],
            auto_start=bool(payload.get("auto_start", True)),
        )
        NotificationService.create(
            title="Shared item received", body=title, kind="mobile",
            data={"task_id": task["id"], "device_id": device["id"]},
        )
        return {"ok": True, "task": task}

    @classmethod
    def command(cls, device: Dict[str, Any], text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            raise ValueError("Command text is required")
        lower = text.lower()
        if lower.startswith("task:") or lower.startswith("todo:"):
            task = TaskService.create_task(
                title=text.split(":", 1)[1].strip()[:200],
                goal=text.split(":", 1)[1].strip(),
                origin=f"mobile:{device['id']}", auto_start=True,
            )
            return {"kind": "task", "task": task}
        if lower.startswith("note:"):
            from services.knowledge import KnowledgeService

            note = KnowledgeService.add_note("Mobile note", text.split(":", 1)[1].strip(), tags=["mobile"])
            return {"kind": "note", "note": note}
        task = TaskService.create_task(
            title=text[:200], goal=text, origin=f"mobile:{device['id']}",
            agent_id="orchestrator", auto_start=True,
        )
        return {"kind": "task", "task": task}
