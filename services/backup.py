"""Encrypted/consistent local backups, staging restore, retention and data wipe."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import uuid
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from config import DATA_DIR, DB_PATH, WORKSPACES_DIR
from memory.database import db_session
from services.audit import AuditService
from services.crypto import redact_text


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


BACKUP_DIR = DATA_DIR / "backups"
BACKUP_DIR.mkdir(parents=True, exist_ok=True)
MAGIC = b"GROKBAK1"


class BackupService:
    @classmethod
    def create(cls, label: str = "manual", passphrase: Optional[str] = None) -> Dict[str, Any]:
        backup_id = f"bk_{uuid.uuid4().hex[:12]}"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        plain_path = BACKUP_DIR / f"{stamp}-{backup_id}.zip"
        manifest = {
            "id": backup_id, "label": label[:200], "created_at": _now(),
            "app": "GROK Platform", "format": 2, "encrypted": bool(passphrase),
        }
        with tempfile.TemporaryDirectory(prefix="grokbackup-") as tmp:
            tmp_db = Path(tmp) / "grok_memory.db"
            cls._snapshot_db(tmp_db)
            with zipfile.ZipFile(plain_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                zf.write(tmp_db, "data/grok_memory.db")
                for folder_name in ("workspaces", "uploads", "attachments"):
                    folder = DATA_DIR / folder_name
                    if folder.exists():
                        for path in folder.rglob("*"):
                            if path.is_file() and ".git" not in path.parts:
                                zf.write(path, str(path.relative_to(DATA_DIR.parent)))
                secrets_dir = DATA_DIR / "secrets"
                if secrets_dir.exists():
                    for path in secrets_dir.rglob("*"):
                        # The master key must never travel in a backup: including it
                        # would make every encrypted credential in secret_store
                        # readable from an otherwise unencrypted archive.
                        if path.is_file() and path.name != "master.key":
                            zf.write(path, str(path.relative_to(DATA_DIR.parent)))
                manifest["database_bytes"] = tmp_db.stat().st_size
                zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        size = plain_path.stat().st_size
        final_path = plain_path
        if passphrase:
            final_path = plain_path.with_suffix(".grokbackup")
            cls._encrypt_file(plain_path, final_path, passphrase)
            plain_path.unlink(missing_ok=True)
            size = final_path.stat().st_size
        checksum = cls._sha256(final_path)
        manifest.update({"path": str(final_path), "size_bytes": size, "sha256": checksum})
        meta_path = final_path.with_suffix(final_path.suffix + ".json")
        meta_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        AuditService.log(
            "backup", "backup.created", actor="user",
            summary=f"Backup created: {label}", data={
                "backup_id": backup_id, "size_bytes": size, "sha256": checksum,
                "encrypted": bool(passphrase), "file": final_path.name,
            },
            activity_title=f"Backup created: {label}",
        )
        return cls._public(manifest)

    @classmethod
    def list(cls) -> List[Dict[str, Any]]:
        items = []
        for meta_path in sorted(BACKUP_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                if data.get("app") == "GROK Platform":
                    items.append(cls._public(data))
            except Exception:
                continue
        return items

    @classmethod
    def get(cls, backup_id: str) -> Optional[Dict[str, Any]]:
        for item in cls.list():
            if item["id"] == backup_id:
                return item
        return None

    @classmethod
    def delete(cls, backup_id: str) -> bool:
        item = cls.get(backup_id)
        if not item:
            return False
        path = Path(item["path"])
        if path.exists():
            path.unlink()
        meta = path.with_suffix(path.suffix + ".json")
        meta.unlink(missing_ok=True)
        AuditService.log("backup", "backup.deleted", actor="user", summary=f"Backup deleted: {backup_id}")
        return True

    @classmethod
    def inspect(cls, backup_id: str, passphrase: Optional[str] = None) -> Dict[str, Any]:
        item = cls.get(backup_id)
        if not item:
            raise ValueError("Backup not found")
        path = Path(item["path"])
        if not path.exists():
            raise FileNotFoundError(str(path))
        source = cls._decrypt_to_temp(path, passphrase) if item.get("encrypted") else path
        try:
            with zipfile.ZipFile(source) as zf:
                names = zf.namelist()
                manifest = json.loads(zf.read("manifest.json")) if "manifest.json" in names else {}
            return {"manifest": manifest, "files": names[:5000], "file_count": len(names)}
        finally:
            if source != path:
                source.unlink(missing_ok=True)

    @classmethod
    def stage_restore(
        cls, backup_id: str, passphrase: Optional[str] = None, dest: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extracts a backup into staging; never overwrites the live database."""
        item = cls.get(backup_id)
        if not item:
            raise ValueError("Backup not found")
        path = Path(item["path"])
        if not path.exists():
            raise FileNotFoundError(str(path))
        source = cls._decrypt_to_temp(path, passphrase) if item.get("encrypted") else path
        staging = Path(dest) if dest else DATA_DIR / "restore_staging" / item["id"]
        staging = staging.expanduser().resolve()
        staging.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(source) as zf:
                for member in zf.infolist():
                    target = (staging / member.filename).resolve()
                    if staging not in target.parents and target != staging:
                        raise ValueError("Unsafe path in backup archive")
                zf.extractall(staging)
        finally:
            if source != path:
                source.unlink(missing_ok=True)
        AuditService.log(
            "backup", "backup.staged", actor="user",
            summary=f"Backup staged for restore: {backup_id}",
            data={"backup_id": backup_id, "staging": str(staging)},
        )
        return {
            "ok": True, "backup_id": backup_id, "staging": str(staging),
            "note": "Staging only. Stop the app and swap data/grok_memory.db after verifying the staged copy.",
        }

    @classmethod
    def export_json(cls, passphrase: Optional[str] = None) -> Dict[str, Any]:
        tables = cls._all_tables()
        payload: Dict[str, Any] = {"version": 2, "exported_at": _now(), "tables": {}}
        with db_session() as conn:
            for table in tables:
                if table == "secret_store":
                    continue
                try:
                    rows = conn.execute(f"SELECT * FROM {table}").fetchall()
                    payload["tables"][table] = [dict(r) for r in rows]
                except Exception:
                    continue
        path = BACKUP_DIR / f"export-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.json"
        raw = json.dumps(payload, indent=2, default=str).encode("utf-8")
        if passphrase:
            raw = cls._encrypt_bytes(raw, passphrase)
            path = path.with_suffix(".json.enc")
        path.write_bytes(raw)
        AuditService.log(
            "privacy", "data.exported", actor="user",
            summary="User data exported as JSON", data={"size_bytes": len(raw), "file": path.name},
        )
        return {"path": str(path), "size_bytes": len(raw), "encrypted": bool(passphrase), "tables": len(payload["tables"])}

    @classmethod
    def cleanup(cls, *, activity_days: int = 180, backup_days: int = 30) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        activity_cutoff = (now - timedelta(days=max(1, activity_days))).isoformat()
        removed_activity = 0
        removed_backups = 0
        with db_session() as conn:
            cur = conn.execute("DELETE FROM activity_events WHERE created_at < ?", (activity_cutoff,))
            removed_activity += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            cur = conn.execute(
                "DELETE FROM notifications WHERE read=1 AND created_at < ?", (activity_cutoff,)
            )
            removed_activity += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
            cur = conn.execute(
                "DELETE FROM run_events WHERE created_at < ? AND run_id IN (SELECT id FROM v2_runs WHERE status IN ('completed','failed','cancelled'))",
                (activity_cutoff,),
            )
            removed_activity += cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        cutoff_ts = (now - timedelta(days=max(1, backup_days))).timestamp()
        for meta_path in BACKUP_DIR.glob("*.json"):
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                path = Path(data.get("path", ""))
                if path.exists() and path.stat().st_mtime < cutoff_ts:
                    path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    removed_backups += 1
            except Exception:
                continue
        AuditService.log(
            "privacy", "data.retention", actor="system",
            summary="Retention cleanup completed",
            data={"activity_removed": removed_activity, "backups_removed": removed_backups},
        )
        return {"activity_removed": removed_activity, "backups_removed": removed_backups}

    @classmethod
    def delete_all_data(
        cls, confirm: str, *, include_audit: bool = False, delete_files: bool = False,
        passphrase: Optional[str] = None,
    ) -> Dict[str, Any]:
        if confirm != "DELETE ALL DATA":
            raise ValueError('Type "DELETE ALL DATA" to confirm')
        safety_backup = cls.create("pre-delete safety backup", passphrase=passphrase)
        tables = cls._all_tables()
        # Delete child tables before parents; audit is handled separately.
        order = [
            # Legacy desktop/chat tables first (children before parents).
            "task_events", "task_subgoals", "autonomous_tasks",
            "messages", "session_summaries", "sessions",
            "episodic_memories", "user_dossier",
            # V2 tables.
            "run_events", "run_steps", "v2_runs", "v2_tasks", "research_claims",
            "research_notes", "research_projects", "automation_runs", "automation_states",
            "automations", "connector_items", "connector_states", "connector_connections",
            "knowledge_fts", "knowledge_chunks", "knowledge_sources", "model_votes",
            "model_stats", "file_checkpoints", "code_repos", "documents", "finance_transactions",
            "bills", "habit_logs", "habits", "mobile_devices", "plugins", "personal_shortcuts",
            "prompt_library", "profiles", "approvals", "policies", "v2_agents", "notifications",
            "activity_events", "platform_meta", "secret_store",
        ]
        deleted: Dict[str, int] = {}
        with db_session() as conn:
            for table in order:
                if table not in tables:
                    continue
                try:
                    cur = conn.execute(f"DELETE FROM {table}")
                    deleted[table] = max(0, cur.rowcount or 0)
                except sqlite3.OperationalError:
                    continue
            if include_audit:
                conn.execute("DROP TRIGGER IF EXISTS audit_events_no_update")
                conn.execute("DROP TRIGGER IF EXISTS audit_events_no_delete")
                cur = conn.execute("DELETE FROM audit_events")
                deleted["audit_events"] = max(0, cur.rowcount or 0)
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS audit_events_no_update
                    BEFORE UPDATE ON audit_events BEGIN
                        SELECT RAISE(ABORT, 'audit_events is append-only');
                    END
                    """
                )
                conn.execute(
                    """
                    CREATE TRIGGER IF NOT EXISTS audit_events_no_delete
                    BEFORE DELETE ON audit_events BEGIN
                        SELECT RAISE(ABORT, 'audit_events is append-only');
                    END
                    """
                )
            conn.execute(
                "INSERT OR REPLACE INTO platform_meta (key, value, updated_at) VALUES ('schema_version','2',?)",
                (_now(),),
            )
        try:
            from services.policies import PolicyService

            PolicyService.ensure_defaults()
        except Exception:
            pass
        try:
            from services.agents import AgentService

            AgentService.ensure_defaults()
        except Exception:
            pass
        removed_files = 0
        if delete_files:
            for folder_name in (
                "workspaces", "uploads", "attachments", "restore_staging",
                "file_checkpoints", "plugins", "sandbox",
            ):
                folder = DATA_DIR / folder_name
                if folder.exists():
                    for child in folder.iterdir():
                        try:
                            if child.is_dir():
                                shutil.rmtree(child)
                            else:
                                child.unlink()
                            removed_files += 1
                        except Exception:
                            continue
            # Runtime/config files that contain credentials or personal state.
            for file_name in (
                "openai_key.json", "active_model.json", "pilot_display_mode.json",
                "llama_server.log", "llama_server.pid", "running_process.json",
            ):
                path = DATA_DIR / file_name
                try:
                    if path.exists() and path.is_file():
                        path.unlink()
                        removed_files += 1
                except Exception:
                    continue
            for path in DATA_DIR.glob("long_convo_*"):
                try:
                    if path.is_file():
                        path.unlink()
                        removed_files += 1
                except Exception:
                    continue
        if not include_audit:
            AuditService.log(
                "privacy", "data.deleted_all", actor="user",
                summary="All user data deleted (audit log retained as tamper-evident record)",
                data={"tables": deleted, "files_removed": removed_files, "safety_backup": safety_backup["id"]},
                activity_title="All user data deleted",
            )
        return {
            "ok": True, "deleted": deleted, "files_removed": removed_files,
            "safety_backup": safety_backup, "audit_retained": not include_audit,
        }

    # ---------------------------------------------------------------- internals
    @classmethod
    def _snapshot_db(cls, target: Path) -> None:
        source = sqlite3.connect(str(DB_PATH))
        try:
            dest = sqlite3.connect(str(target))
            try:
                source.backup(dest)
            finally:
                dest.close()
        finally:
            source.close()

    @classmethod
    def _encrypt_file(cls, source: Path, target: Path, passphrase: str) -> None:
        salt = os.urandom(16)
        key = cls._derive_key(passphrase, salt)
        token = Fernet(key).encrypt(source.read_bytes())
        target.write_bytes(MAGIC + salt + token)

    @classmethod
    def _decrypt_to_temp(cls, path: Path, passphrase: Optional[str]) -> Path:
        if not passphrase:
            raise ValueError("This backup is encrypted; a passphrase is required")
        raw = path.read_bytes()
        if not raw.startswith(MAGIC):
            raise ValueError("Unsupported encrypted backup format")
        salt = raw[len(MAGIC):len(MAGIC) + 16]
        token = raw[len(MAGIC) + 16:]
        try:
            data = Fernet(cls._derive_key(passphrase, salt)).decrypt(token)
        except InvalidToken as exc:
            raise ValueError("Incorrect backup passphrase") from exc
        tmp = tempfile.NamedTemporaryFile(prefix="grokbackup-", suffix=".zip", delete=False)
        tmp.write(data)
        tmp.close()
        return Path(tmp.name)

    @classmethod
    def _encrypt_bytes(cls, data: bytes, passphrase: str) -> bytes:
        salt = os.urandom(16)
        return MAGIC + salt + Fernet(cls._derive_key(passphrase, salt)).encrypt(data)

    @staticmethod
    def _derive_key(passphrase: str, salt: bytes) -> bytes:
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=390000)
        return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _all_tables(cls) -> List[str]:
        with db_session() as conn:
            return [r["name"] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall()]

    @staticmethod
    def _public(data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in data.items() if k not in ("label",) or True}
