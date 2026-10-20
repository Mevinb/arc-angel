"""Lightweight local code index and repository metadata (no execution)."""
from __future__ import annotations

import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from memory.database import db_session
from services.audit import AuditService
from services.files import SKIP_DIRS, TEXT_EXT, _resolve_root, _safe_rel


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


SYMBOL_PATTERNS = [
    ("python", re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\("), "function"),
    ("python", re.compile(r"^\s*class\s+([A-Za-z_]\w*)"), "class"),
    ("js", re.compile(r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$]\w*)\s*\("), "function"),
    ("js", re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_$]\w*)"), "class"),
    ("js", re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$]\w*)\s*=\s*(?:async\s*)?\("), "function"),
    ("go", re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\("), "function"),
    ("rust", re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)\s*\("), "function"),
    ("java", re.compile(r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:class|interface|record)\s+([A-Za-z_]\w*)"), "class"),
]


class CodeIndexService:
    @classmethod
    def index(cls, root: Optional[str] = None, max_files: int = 5000) -> Dict[str, Any]:
        base = _resolve_root(root)
        if not base.is_dir():
            raise NotADirectoryError(str(base))
        files = []
        symbols = []
        for path in base.rglob("*"):
            if len(files) >= max_files:
                break
            rel = path.relative_to(base)
            if any(part in SKIP_DIRS for part in rel.parts) or not path.is_file():
                continue
            if path.suffix.lower() not in TEXT_EXT and path.name not in (".gitignore", "Dockerfile", "Makefile"):
                continue
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
            files.append(str(rel))
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            suffix = path.suffix.lower()
            for lineno, line in enumerate(lines, 1):
                for _lang, pattern, kind in SYMBOL_PATTERNS:
                    if suffix in (".py",) and _lang != "python":
                        continue
                    if suffix in (".js", ".jsx", ".ts", ".tsx") and _lang != "js":
                        continue
                    match = pattern.match(line)
                    if match:
                        symbols.append({
                            "name": match.group(1), "kind": kind,
                            "path": str(rel), "line": lineno, "signature": line.strip()[:240],
                        })
                        break
        repo_id = f"cr_{uuid.uuid4().hex[:10]}"
        payload = {"symbols": symbols[:20000]}
        with db_session() as conn:
            conn.execute("DELETE FROM code_repos WHERE root=?", (str(base),))
            conn.execute(
                "INSERT INTO code_repos (id, root, file_count, symbol_count, index_data, indexed_at) VALUES (?, ?, ?, ?, ?, ?)",
                (repo_id, str(base), len(files), len(symbols), json.dumps(payload), _now()),
            )
        AuditService.log(
            "code", "code.indexed", actor="user",
            summary=f"Indexed {len(files)} files and {len(symbols)} symbols",
            data={"root": str(base), "files": len(files), "symbols": len(symbols)},
        )
        return {
            "id": repo_id, "root": str(base), "files": len(files),
            "symbols": len(symbols), "sample": symbols[:100],
        }

    @classmethod
    def repos(cls) -> List[Dict[str, Any]]:
        with db_session() as conn:
            rows = conn.execute(
                "SELECT id, root, file_count, symbol_count, indexed_at FROM code_repos ORDER BY indexed_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    @classmethod
    def search(cls, query: str, root: Optional[str] = None, limit: int = 30) -> List[Dict[str, Any]]:
        base = _resolve_root(root)
        with db_session() as conn:
            row = conn.execute("SELECT * FROM code_repos WHERE root=?", (str(base),)).fetchone()
        if row:
            try:
                symbols = json.loads(row["index_data"] or "{}").get("symbols", [])
            except Exception:
                symbols = []
        else:
            symbols = cls.index(str(base)).get("sample", [])
        q = (query or "").lower()
        matches = [s for s in symbols if q in s["name"].lower() or q in s["path"].lower()]
        return matches[:max(1, min(limit, 200))]

    @classmethod
    def symbols(cls, root: Optional[str] = None, limit: int = 500) -> List[Dict[str, Any]]:
        base = _resolve_root(root)
        with db_session() as conn:
            row = conn.execute("SELECT index_data FROM code_repos WHERE root=?", (str(base),)).fetchone()
        if not row:
            return []
        try:
            return json.loads(row["index_data"] or "{}").get("symbols", [])[:max(1, min(limit, 2000))]
        except Exception:
            return []

    @classmethod
    def repo_info(cls, root: Optional[str] = None) -> Dict[str, Any]:
        base = _resolve_root(root)
        info: Dict[str, Any] = {"root": str(base), "is_git": (base / ".git").exists()}
        if not info["is_git"]:
            return info

        def run(args: List[str]) -> str:
            try:
                proc = subprocess.run(
                    args, cwd=str(base), capture_output=True, text=True, timeout=8,
                    check=False,
                )
                return proc.stdout.strip()[:5000]
            except Exception as exc:
                return f"error: {exc}"

        info["branch"] = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
        info["status"] = run(["git", "status", "--short"])
        info["recent_commits"] = run(["git", "log", "--oneline", "-8"])
        info["diff_stat"] = run(["git", "diff", "--stat"])
        return info

    @classmethod
    def read_file(cls, path: str, root: Optional[str] = None) -> Dict[str, Any]:
        base = _resolve_root(root)
        target = _safe_rel(base, path)
        if not target.is_file():
            raise FileNotFoundError(str(target))
        return {
            "path": str(target.relative_to(base)),
            "content": target.read_text(encoding="utf-8", errors="replace")[:200000],
            "size": target.stat().st_size,
        }
