"""
Local encryption + redaction utilities.

Credentials (OAuth tokens, API keys, pairing secrets) are encrypted at rest with
Fernet (AES-128-CBC + HMAC).  The master key is either provided through
``GROK_MASTER_KEY`` or generated once at ``data/secrets/master.key`` with 0600
permissions.  Ciphertext is stored in SQLite via :mod:`services.crypto` helpers;
plaintext is never written to the database, prompts, audit records or logs.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from cryptography.fernet import Fernet, InvalidToken

from config import DATA_DIR

SECRETS_DIR = DATA_DIR / "secrets"
SECRETS_DIR.mkdir(parents=True, exist_ok=True)
MASTER_KEY_FILE = SECRETS_DIR / "master.key"

_lock = threading.Lock()
_fernet: Optional[Fernet] = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_or_create_key() -> bytes:
    env_key = os.getenv("GROK_MASTER_KEY", "").strip()
    if env_key:
        return env_key.encode("utf-8")
    if MASTER_KEY_FILE.exists():
        return MASTER_KEY_FILE.read_bytes().strip()
    key = Fernet.generate_key()
    tmp = MASTER_KEY_FILE.with_suffix(".tmp")
    tmp.write_bytes(key)
    try:
        os.chmod(tmp, 0o600)
        os.replace(tmp, MASTER_KEY_FILE)
        os.chmod(MASTER_KEY_FILE, 0o600)
    except OSError:
        if not MASTER_KEY_FILE.exists():
            MASTER_KEY_FILE.write_bytes(key)
    return key


def get_fernet() -> Fernet:
    global _fernet
    if _fernet is None:
        with _lock:
            if _fernet is None:
                try:
                    _fernet = Fernet(_load_or_create_key())
                except Exception as exc:  # invalid GROK_MASTER_KEY
                    raise RuntimeError(
                        "Invalid GROK_MASTER_KEY. It must be a url-safe base64 Fernet key."
                    ) from exc
    return _fernet


def encrypt_text(plaintext: str) -> str:
    if plaintext is None:
        plaintext = ""
    return get_fernet().encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt_text(ciphertext: str) -> str:
    return get_fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")


def encrypt_json(data: Any) -> str:
    return encrypt_text(json.dumps(data, sort_keys=True, default=str))


def decrypt_json(ciphertext: str) -> Any:
    return json.loads(decrypt_text(ciphertext))


def set_secret(name: str, value: Any, meta: Optional[Dict[str, Any]] = None) -> None:
    """Encrypt and persist a secret under ``name``."""
    from memory.database import db_session

    payload = encrypt_json(value)
    now = _now()
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO secret_store (name, ciphertext, meta, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET ciphertext=excluded.ciphertext,
                meta=excluded.meta, updated_at=excluded.updated_at
            """,
            (name, payload, json.dumps(meta or {}, default=str), now, now),
        )


def get_secret(name: str, default: Any = None) -> Any:
    from memory.database import db_session

    with db_session() as conn:
        row = conn.execute("SELECT ciphertext FROM secret_store WHERE name = ?", (name,)).fetchone()
    if not row:
        return default
    try:
        return decrypt_json(row["ciphertext"])
    except (InvalidToken, ValueError, TypeError):
        return default


def delete_secret(name: str) -> bool:
    from memory.database import db_session

    with db_session() as conn:
        cur = conn.execute("DELETE FROM secret_store WHERE name = ?", (name,))
        return cur.rowcount > 0


def list_secrets() -> list[Dict[str, Any]]:
    from memory.database import db_session

    with db_session() as conn:
        rows = conn.execute(
            "SELECT name, meta, created_at, updated_at FROM secret_store ORDER BY name"
        ).fetchall()
    result = []
    for row in rows:
        try:
            meta = json.loads(row["meta"] or "{}")
        except Exception:
            meta = {}
        result.append({
            "name": row["name"],
            "meta": meta,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        })
    return result


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------
_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bsk-proj-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b"),
    re.compile(r"(?i)\b(api[_-]?key|secret|password|passwd|token|authorization)\b\s*[:=]\s*['\"]?([^\s'\"]{6,})"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{10,}"),
]

_SENSITIVE_KEY_NAMES = {
    "api_key", "apikey", "password", "passwd", "secret", "token",
    "access_token", "refresh_token", "authorization", "client_secret",
    "private_key", "pairing_code", "webhook_secret",
}


def redact_text(text: str, replacement: str = "[REDACTED]") -> str:
    if not text:
        return text
    out = str(text)
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            out = pattern.sub(lambda m: f"{m.group(1)}={replacement}", out)
        elif pattern.groups == 1:
            out = pattern.sub(lambda m: f"{m.group(1)} {replacement}", out)
        else:
            out = pattern.sub(replacement, out)
    return out


def looks_like_secret(text: str) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def redact_obj(value: Any, max_string: int = 4000, depth: int = 0) -> Any:
    """Recursively redact secrets and cap string lengths for safe persistence."""
    if depth > 8:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if str(k).lower() in _SENSITIVE_KEY_NAMES:
                out[k] = "[REDACTED]"
            else:
                out[k] = redact_obj(v, max_string=max_string, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [redact_obj(v, max_string=max_string, depth=depth + 1) for v in value[:200]]
    if isinstance(value, str):
        text = redact_text(value)
        if len(text) > max_string:
            return text[:max_string] + f"… [truncated {len(text) - max_string} chars]"
        return text
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    text = redact_text(str(value))
    if len(text) > max_string:
        return text[:max_string] + f"… [truncated {len(text) - max_string} chars]"
    return text


def mask_secret(value: str, keep: int = 4) -> str:
    value = value or ""
    if len(value) <= keep * 2:
        return "•" * len(value)
    return f"{value[:4]}…{value[-keep:]}"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def constant_time_equals(a: str, b: str) -> bool:
    import hmac

    return hmac.compare_digest((a or "").encode("utf-8"), (b or "").encode("utf-8"))


def token_fingerprint(token: str) -> str:
    return digest(token)[:16]


def base64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")
