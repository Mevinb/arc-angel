"""Kill switch, offline/private mode, and optional app lock."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from memory.database import db_session

_LOCK_SETTING = "app_lock"
_KILL_SWITCH = "kill_switch"
_OFFLINE_MODE = "offline_mode"
_REDACTION = "redaction_enabled"

_unlock_tokens: Dict[str, float] = {}
_UNLOCK_TTL = 8 * 3600


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SafetyService:
    UNLOCK_TTL = _UNLOCK_TTL

    @classmethod
    def _get_meta(cls, key: str, default: Optional[str] = None) -> Optional[str]:
        with db_session() as conn:
            row = conn.execute("SELECT value FROM platform_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    @classmethod
    def _set_meta(cls, key: str, value: Optional[str]) -> None:
        with db_session() as conn:
            conn.execute(
                """
                INSERT INTO platform_meta (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, value, _now()),
            )

    # ------------------------------------------------------------ kill switch
    @classmethod
    def kill_switch_engaged(cls) -> bool:
        return (cls._get_meta(_KILL_SWITCH, "0") or "0") == "1"

    @classmethod
    def engage_kill_switch(cls, reason: str = "User requested emergency stop") -> Dict[str, Any]:
        cls._set_meta(_KILL_SWITCH, "1")
        cancelled = 0
        try:
            from services.run_manager import RunManager

            cancelled = RunManager.get().cancel_all(reason)
        except Exception:
            pass
        try:
            import asyncio

            from services.automations import AutomationService

            asyncio.get_running_loop().create_task(AutomationService.stop_scheduler())
        except Exception:
            pass
        try:
            import server

            server.desktop_agent.abort()
        except Exception:
            pass
        try:
            from services.events import NotificationService

            NotificationService.create(
                title="Emergency stop engaged",
                body=reason,
                severity="critical",
                kind="safety",
            )
        except Exception:
            pass
        try:
            from services.audit import AuditService

            AuditService.log(
                "safety", "kill_switch.engaged", actor="user", status="ok",
                summary=reason, data={"cancelled_runs": cancelled}, activity_title="Kill switch engaged",
            )
        except Exception:
            pass
        return {"kill_switch": True, "cancelled_runs": cancelled}

    @classmethod
    def release_kill_switch(cls) -> Dict[str, Any]:
        cls._set_meta(_KILL_SWITCH, "0")
        try:
            from services.audit import AuditService

            AuditService.log(
                "safety", "kill_switch.released", actor="user", status="ok",
                summary="Kill switch released", activity_title="Kill switch released",
            )
        except Exception:
            pass
        return {"kill_switch": False}

    # --------------------------------------------------------------- modes
    @classmethod
    def offline_mode(cls) -> bool:
        return (cls._get_meta(_OFFLINE_MODE, "0") or "0") == "1"

    @classmethod
    def set_offline_mode(cls, enabled: bool) -> Dict[str, Any]:
        cls._set_meta(_OFFLINE_MODE, "1" if enabled else "0")
        try:
            from services.audit import AuditService

            AuditService.log(
                "safety", "offline_mode.enabled" if enabled else "offline_mode.disabled",
                actor="user", summary="Offline mode changed", data={"enabled": enabled},
            )
        except Exception:
            pass
        return {"offline_mode": enabled}

    @classmethod
    def redaction_enabled(cls) -> bool:
        return (cls._get_meta(_REDACTION, "1") or "1") == "1"

    @classmethod
    def set_redaction(cls, enabled: bool) -> None:
        cls._set_meta(_REDACTION, "1" if enabled else "0")

    # ------------------------------------------------------------- app lock
    @classmethod
    def lock_enabled(cls) -> bool:
        raw = cls._get_meta(_LOCK_SETTING)
        if not raw:
            return False
        try:
            return bool(json.loads(raw).get("enabled"))
        except Exception:
            return False

    @classmethod
    def set_lock(cls, pin: Optional[str]) -> Dict[str, Any]:
        if not pin:
            cls._set_meta(_LOCK_SETTING, json.dumps({"enabled": False}))
            _unlock_tokens.clear()
            try:
                from services.audit import AuditService

                AuditService.log(
                    "safety", "app_lock.disabled", actor="user", summary="App lock disabled"
                )
            except Exception:
                pass
            return {"enabled": False}
        if len(pin) < 4:
            raise ValueError("PIN must be at least 4 characters")
        salt = secrets.token_bytes(16)
        iterations = 260_000
        digest = hashlib.pbkdf2_hmac("sha256", pin.encode("utf-8"), salt, iterations)
        payload = {
            "enabled": True,
            "salt": base64.b64encode(salt).decode("ascii"),
            "hash": base64.b64encode(digest).decode("ascii"),
            "iterations": iterations,
        }
        cls._set_meta(_LOCK_SETTING, json.dumps(payload))
        try:
            from services.audit import AuditService

            AuditService.log("safety", "app_lock.enabled", actor="user", summary="App lock enabled")
        except Exception:
            pass
        return {"enabled": True}

    @classmethod
    def verify_lock(cls, pin: str) -> bool:
        raw = cls._get_meta(_LOCK_SETTING)
        if not raw:
            return False
        try:
            data = json.loads(raw)
            if not data.get("enabled"):
                return True
            salt = base64.b64decode(data["salt"])
            expected = base64.b64decode(data["hash"])
            digest = hashlib.pbkdf2_hmac(
                "sha256", (pin or "").encode("utf-8"), salt, int(data["iterations"])
            )
            return hmac.compare_digest(digest, expected)
        except Exception:
            return False

    @classmethod
    def create_unlock_token(cls, pin: str) -> Optional[str]:
        if not cls.lock_enabled():
            return None
        if not cls.verify_lock(pin):
            return None
        token = secrets.token_urlsafe(32)
        _unlock_tokens[token] = time.time() + cls.UNLOCK_TTL
        return token

    @classmethod
    def check_unlock_token(cls, token: Optional[str]) -> bool:
        if not cls.lock_enabled():
            return True
        if not token:
            return False
        expiry = _unlock_tokens.get(token)
        if not expiry or expiry < time.time():
            _unlock_tokens.pop(token, None)
            return False
        return True

    @classmethod
    def lock_status(cls) -> Dict[str, Any]:
        return {
            "lock_enabled": cls.lock_enabled(),
            "kill_switch": cls.kill_switch_engaged(),
            "offline_mode": cls.offline_mode(),
            "redaction_enabled": cls.redaction_enabled(),
        }

    @classmethod
    def status(cls) -> Dict[str, Any]:
        from services.crypto import list_secrets

        return {
            **cls.lock_status(),
            "secret_count": len(list_secrets()),
        }
