"""Workspace file inspection, checkpoints, diffs and safe restore."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import BASE_DIR, DATA_DIR, WORKSPACES_DIR
from memory.database import db_session
from services.audit import AuditService


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


CHECKPOINT_DIR = DATA_DIR / "file_checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".cache", "dist", "build", ".mypy_cache", ".pytest_cache"}
TEXT_EXT = {
    ".txt", ".md", ".markdown", ".py", ".js", ".ts", ".tsx", ".jsx", ".json", ".jsonl",
    ".yaml", ".yml", ".toml", ".csv", ".ini", ".cfg", ".html", ".htm", ".css", ".sh",
    ".bash", ".c", ".cpp", ".h", ".hpp", ".java", ".go", ".rs", ".sql", ".log", ".rst",
    ".tex", ".xml", ".gitignore", ".env.example",
}


def _allowed_roots() -> List[Path]:
    roots = [WORKSPACES_DIR.resolve(), BASE_DIR.resolve()]
    extra = os.getenv("GROK_WORKSPACE_ROOTS", "")
    for part in extra.split(os.pathsep):
        if part.strip():
            try:
                roots.append(Path(part).expanduser().resolve())
            except Exception:
                continue
    return list(dict.fromkeys(roots))


def _resolve_root(root: Optional[str]) -> Path:
    base = Path(root).expanduser().resolve() if root else WORKSPACES_DIR.resolve()
    for allowed in _allowed_roots():
        if base == allowed or allowed in base.parents:
            return base
    raise PermissionError("Workspace root is outside the approved roots. Set GROK_WORKSPACE_ROOTS to allow it.")


def _safe_rel(root: Path, rel: str) -> Path:
    target = (root / (rel or ".")).resolve()
    if target != root and root not in target.parents:
        raise PermissionError("Path escapes the workspace")
    return target


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FileService:
    @classmethod
    def root_info(cls, root: Optional[str] = None) -> Dict[str, Any]:
        base = _resolve_root(root)
        return {
            "root": str(base), "exists": base.exists(),
            "allowed_roots": [str(p) for p in _allowed_roots()],
        }

    @classmethod
    def tree(cls, root: Optional[str] = None, *, max_entries: int = 1000, max_depth: int = 6) -> Dict[str, Any]:
        base = _resolve_root(root)
        if not base.exists():
            raise FileNotFoundError(str(base))
        entries = []
        for path in sorted(base.rglob("*")):
            rel = path.relative_to(base)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if len(rel.parts) > max_depth:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append({
                "path": str(rel), "name": path.name,
                "is_dir": path.is_dir(), "size": stat.st_size if path.is_file() else 0,
                "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
            })
            if len(entries) >= max_entries:
                break
        return {"root": str(base), "entries": entries, "truncated": len(entries) >= max_entries}

    @classmethod
    def read(cls, path: str, root: Optional[str] = None, max_chars: int = 100000) -> Dict[str, Any]:
        base = _resolve_root(root)
        target = _safe_rel(base, path)
        if not target.is_file():
            raise FileNotFoundError(str(target))
        if target.stat().st_size > 5 * 1024 * 1024:
            raise ValueError("File is too large to display (>5 MB)")
        raw = target.read_bytes()
        try:
            content = raw.decode("utf-8")
            binary = False
        except UnicodeDecodeError:
            import base64

            content = base64.b64encode(raw[:20000]).decode("ascii")
            binary = True
        return {
            "path": str(target.relative_to(base)), "content": content[:max_chars],
            "truncated": len(content) > max_chars, "binary": binary,
            "size": len(raw), "sha256": _hash_file(target),
        }

    @classmethod
    def write(cls, path: str, content: str, root: Optional[str] = None) -> Dict[str, Any]:
        base = _resolve_root(root)
        target = _safe_rel(base, path)
        if target == base or base not in target.parents:
            raise PermissionError("Invalid target")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        AuditService.log(
            "files", "file.written", actor="user",
            summary=f"File written: {target.relative_to(base)}",
            data={"root": str(base), "path": str(target.relative_to(base)), "bytes": len(content.encode())},
        )
        return {"ok": True, "path": str(target.relative_to(base)), "size": target.stat().st_size}

    @classmethod
    def search(
        cls, query: str, root: Optional[str] = None, *, max_results: int = 50,
        max_file_bytes: int = 2 * 1024 * 1024,
    ) -> Dict[str, Any]:
        base = _resolve_root(root)
        needle = (query or "").lower().strip()
        if not needle:
            return {"results": [], "query": query}
        results = []
        for path in base.rglob("*"):
            if len(results) >= max_results:
                break
            rel = path.relative_to(base)
            if any(part in SKIP_DIRS for part in rel.parts) or not path.is_file():
                continue
            if needle in str(rel).lower():
                results.append({"path": str(rel), "line": 0, "preview": "filename match"})
            if path.suffix.lower() not in TEXT_EXT or path.stat().st_size > max_file_bytes:
                continue
            try:
                with path.open("r", encoding="utf-8", errors="replace") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if needle in line.lower():
                            results.append({"path": str(rel), "line": lineno, "preview": line.strip()[:300]})
                            if len(results) >= max_results:
                                break
            except Exception:
                continue
        return {"query": query, "results": results, "root": str(base)}

    # ------------------------------------------------------------- checkpoints
    @classmethod
    def create_checkpoint(cls, label: str, root: Optional[str] = None) -> Dict[str, Any]:
        base = _resolve_root(root)
        if not base.exists():
            raise FileNotFoundError(str(base))
        checkpoint_id = f"fc_{uuid.uuid4().hex[:12]}"
        storage = CHECKPOINT_DIR / checkpoint_id
        storage.mkdir(parents=True, exist_ok=True)
        archive = storage / "snapshot.zip"
        count = 0
        size = 0
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for path in base.rglob("*"):
                rel = path.relative_to(base)
                if any(part in SKIP_DIRS for part in rel.parts) or not path.is_file():
                    continue
                if path.stat().st_size > 20 * 1024 * 1024:
                    continue
                zf.write(path, str(rel))
                count += 1
                size += path.stat().st_size
        with db_session() as conn:
            conn.execute(
                """
                INSERT INTO file_checkpoints (id, label, root, storage_dir, files_count, size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (checkpoint_id, label[:200], str(base), str(archive), count, size, _now()),
            )
        AuditService.log(
            "files", "checkpoint.created", actor="user",
            summary=f"File checkpoint created: {label}",
            data={"checkpoint_id": checkpoint_id, "root": str(base), "files": count, "size": size},
        )
        return cls.get_checkpoint(checkpoint_id)  # type: ignore[return-value]

    @classmethod
    def list_checkpoints(cls) -> List[Dict[str, Any]]:
        with db_session() as conn:
            rows = conn.execute("SELECT * FROM file_checkpoints ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    @classmethod
    def get_checkpoint(cls, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM file_checkpoints WHERE id=?", (checkpoint_id,)).fetchone()
        return dict(row) if row else None

    @classmethod
    def diff_checkpoint(cls, checkpoint_id: str) -> Dict[str, Any]:
        checkpoint = cls.get_checkpoint(checkpoint_id)
        if not checkpoint:
            raise ValueError("Checkpoint not found")
        base = _resolve_root(checkpoint["root"])
        archive = Path(checkpoint["storage_dir"])
        old: Dict[str, str] = {}
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue
                old[name] = hashlib.sha256(zf.read(name)).hexdigest()
        current: Dict[str, str] = {}
        for path in base.rglob("*"):
            rel = path.relative_to(base)
            if any(part in SKIP_DIRS for part in rel.parts) or not path.is_file():
                continue
            current[str(rel)] = _hash_file(path)
        added = sorted(set(current) - set(old))
        removed = sorted(set(old) - set(current))
        changed = sorted(k for k in set(old) & set(current) if old[k] != current[k])
        return {
            "checkpoint_id": checkpoint_id, "root": str(base),
            "added": added[:1000], "removed": removed[:1000], "changed": changed[:1000],
            "counts": {"added": len(added), "removed": len(removed), "changed": len(changed)},
        }

    @classmethod
    def restore_checkpoint(
        cls, checkpoint_id: str, *, confirm: bool = False, prune: bool = False,
    ) -> Dict[str, Any]:
        if not confirm:
            raise ValueError("Set confirm=true to restore a checkpoint")
        checkpoint = cls.get_checkpoint(checkpoint_id)
        if not checkpoint:
            raise ValueError("Checkpoint not found")
        base = _resolve_root(checkpoint["root"])
        archive = Path(checkpoint["storage_dir"])
        safety = cls.create_checkpoint("pre-restore safety snapshot", str(base))
        with zipfile.ZipFile(archive) as zf:
            members = {
                info.filename.replace("\\", "/").rstrip("/")
                for info in zf.infolist() if not info.is_dir()
            }
            for member in zf.infolist():
                target = (base / member.filename).resolve()
                if target != base and base not in target.parents:
                    raise PermissionError("Unsafe path in checkpoint")
            zf.extractall(base)
        pruned: List[str] = []
        if prune:
            for path in sorted(base.rglob("*"), reverse=True):
                rel = path.relative_to(base)
                if any(part in SKIP_DIRS for part in rel.parts):
                    continue
                if path.is_dir() or path.is_symlink() or not path.is_file():
                    continue
                if rel.as_posix() not in members:
                    path.unlink()
                    pruned.append(rel.as_posix())
        AuditService.log(
            "files", "checkpoint.restored", actor="user",
            summary=f"Checkpoint restored: {checkpoint_id}",
            data={
                "checkpoint_id": checkpoint_id, "root": str(base),
                "safety_checkpoint": safety["id"], "prune": prune,
                "pruned_count": len(pruned), "pruned": pruned[:200],
            },
            activity_title="File checkpoint restored",
        )
        return {
            "ok": True, "checkpoint_id": checkpoint_id,
            "safety_checkpoint": safety, "pruned": pruned,
        }

    @classmethod
    def delete_checkpoint(cls, checkpoint_id: str) -> bool:
        checkpoint = cls.get_checkpoint(checkpoint_id)
        if not checkpoint:
            return False
        shutil.rmtree(Path(checkpoint["storage_dir"]).parent, ignore_errors=True)
        with db_session() as conn:
            conn.execute("DELETE FROM file_checkpoints WHERE id=?", (checkpoint_id,))
        AuditService.log("files", "checkpoint.deleted", actor="user", summary=f"Checkpoint deleted: {checkpoint_id}")
        return True
