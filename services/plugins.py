"""Local plugin registry. Plugins expose typed HTTP tools, never in-process code."""
from __future__ import annotations

import json
import re
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import DATA_DIR
from memory.database import db_session
from services.audit import AuditService

PLUGIN_DIR = DATA_DIR / "plugins"
PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{1,60}$")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PluginService:
    @classmethod
    def list_plugins(cls, enabled_only: bool = False) -> List[Dict[str, Any]]:
        clause = "WHERE enabled=1" if enabled_only else ""
        with db_session() as conn:
            rows = conn.execute(f"SELECT * FROM plugins {clause} ORDER BY installed_at DESC").fetchall()
        out = []
        for row in rows:
            try:
                manifest = json.loads(row["manifest"] or "{}")
            except Exception:
                manifest = {}
            out.append({
                "id": row["id"], "name": row["name"], "version": row["version"],
                "description": row["description"], "path": row["path"],
                "manifest": manifest, "enabled": bool(row["enabled"]),
                "installed_at": row["installed_at"],
                "tool_count": len(manifest.get("tools") or []),
            })
        return out

    @classmethod
    def install(cls, source: str) -> Dict[str, Any]:
        src = Path(source).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(str(src))
        staging = PLUGIN_DIR / f".install-{uuid.uuid4().hex[:8]}"
        temp_dir: Optional[Path] = None
        try:
            if src.is_file() and src.suffix.lower() == ".zip":
                staging.mkdir(parents=True, exist_ok=True)
                temp_dir = staging
                with zipfile.ZipFile(src) as zf:
                    for member in zf.infolist():
                        target = (temp_dir / member.filename).resolve()
                        if temp_dir not in target.parents and target != temp_dir:
                            raise ValueError("Unsafe path in plugin archive")
                    zf.extractall(temp_dir)
                manifest_path = cls._find_manifest(temp_dir)
            elif src.is_dir():
                manifest_path = cls._find_manifest(src)
            else:
                raise ValueError("Plugin source must be a directory or .zip archive")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            plugin_id = cls._validate_manifest(manifest)
            target_dir = PLUGIN_DIR / plugin_id
            if target_dir.exists():
                raise ValueError(f"Plugin already installed: {plugin_id}")
            if temp_dir is not None:
                shutil.move(str(temp_dir), str(target_dir))
            else:
                shutil.copytree(src, target_dir)
            with db_session() as conn:
                conn.execute(
                    """
                    INSERT INTO plugins (id, name, version, description, path, manifest, enabled, installed_at)
                    VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        plugin_id, manifest.get("name") or plugin_id,
                        manifest.get("version") or "0.1.0",
                        manifest.get("description") or "",
                        str(target_dir), json.dumps(manifest), _now(),
                    ),
                )
        except Exception:
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        AuditService.log(
            "plugin", "plugin.installed", actor="user",
            summary=f"Plugin installed: {plugin_id}", data={"plugin_id": plugin_id},
            activity_title=f"Plugin installed: {plugin_id}",
        )
        return next(p for p in cls.list_plugins() if p["id"] == plugin_id)

    @classmethod
    def enable(cls, plugin_id: str, enabled: bool = True) -> Optional[Dict[str, Any]]:
        with db_session() as conn:
            count = conn.execute(
                "UPDATE plugins SET enabled=? WHERE id=?", (1 if enabled else 0, plugin_id)
            ).rowcount
        if not count:
            return None
        try:
            from services.tools import ToolRegistry

            ToolRegistry.get().reload_plugin_tools()
        except Exception:
            pass
        AuditService.log(
            "plugin", "plugin.enabled" if enabled else "plugin.disabled", actor="user",
            summary=f"Plugin {'enabled' if enabled else 'disabled'}: {plugin_id}",
        )
        return next((p for p in cls.list_plugins() if p["id"] == plugin_id), None)

    @classmethod
    def uninstall(cls, plugin_id: str) -> bool:
        plugin = next((p for p in cls.list_plugins() if p["id"] == plugin_id), None)
        if not plugin:
            return False
        path = Path(plugin["path"]).resolve()
        if PLUGIN_DIR not in path.parents:
            raise ValueError("Plugin path is outside the plugin directory")
        shutil.rmtree(path, ignore_errors=True)
        with db_session() as conn:
            conn.execute("DELETE FROM plugins WHERE id=?", (plugin_id,))
        try:
            from services.tools import ToolRegistry

            ToolRegistry.get().reload_plugin_tools()
        except Exception:
            pass
        AuditService.log("plugin", "plugin.uninstalled", actor="user", summary=f"Plugin uninstalled: {plugin_id}")
        return True

    @classmethod
    def _find_manifest(cls, folder: Path) -> Path:
        direct = folder / "manifest.json"
        if direct.is_file():
            return direct
        matches = list(folder.glob("*/manifest.json"))
        if len(matches) == 1:
            return matches[0]
        raise ValueError("Plugin manifest.json was not found")

    @classmethod
    def _validate_manifest(cls, manifest: Dict[str, Any]) -> str:
        if not isinstance(manifest, dict):
            raise ValueError("Plugin manifest must be a JSON object")
        plugin_id = str(manifest.get("id") or "").lower()
        if not SAFE_ID.match(plugin_id):
            raise ValueError("Plugin id must be 2-60 lowercase letters, numbers, _ or -")
        tools = manifest.get("tools") or []
        if not isinstance(tools, list):
            raise ValueError("manifest.tools must be a list")
        for tool in tools:
            if not isinstance(tool, dict) or not tool.get("name"):
                raise ValueError("Each plugin tool needs a name")
            endpoint = tool.get("endpoint") or manifest.get("endpoint")
            if not endpoint or not str(endpoint).startswith(("http://", "https://")):
                raise ValueError("Plugin tools need an http(s) endpoint")
            schema = tool.get("input_schema") or {"type": "object", "properties": {}}
            if not isinstance(schema, dict) or schema.get("type") != "object":
                raise ValueError("Plugin tool input_schema must be an object schema")
        return plugin_id
