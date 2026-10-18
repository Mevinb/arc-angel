"""
Workspace Tools: Secure filesystem, code editing, shell execution, and download bridge
for Grok's Hierarchical Long-Horizon Task Orchestrator.
All actions are scoped to the task's dedicated workspace directory: data/workspaces/<task_id>/
"""
import os
import shutil
import subprocess
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Union

logger = logging.getLogger("workspace_tools")


class WorkspaceManager:
    """Manages an isolated workspace directory for an autonomous long-horizon task."""

    def __init__(self, workspace_dir: Optional[Union[str, Path]] = None, task_id: Optional[str] = None):
        if workspace_dir is None:
            if task_id:
                from config import WORKSPACES_DIR
                workspace_dir = WORKSPACES_DIR / task_id
            else:
                raise ValueError("Either workspace_dir or task_id must be provided.")
        self.workspace_dir = Path(workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

    def cleanup(self):
        """Removes the workspace directory if needed."""
        if self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def _resolve_safe_path(self, rel_path: str) -> Path:
        """Resolves rel_path against workspace_dir and guards against path traversal."""
        clean = rel_path.strip().lstrip("/\\")
        target = (self.workspace_dir / clean).resolve()
        try:
            target.relative_to(self.workspace_dir)
        except ValueError:
            raise PermissionError(f"Access denied: path '{rel_path}' escapes workspace boundary '{self.workspace_dir}'")
        return target

    def write_file(self, rel_path: str, content: str) -> str:
        """
        Creates or overwrites a file inside the task workspace.
        Automatically creates missing parent directories.
        """
        target = self._resolve_safe_path(rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        logger.info("Workspace [%s] wrote file '%s' (%d bytes)", self.workspace_dir.name, rel_path, len(content))
        return f"Successfully wrote {len(content)} characters to '{rel_path}'."

    def edit_file(self, rel_path: str, old_str: str, new_str: str) -> str:
        """
        Performs targeted substring replacement inside an existing workspace file.
        Essential for self-healing code debugging loops.
        """
        target = self._resolve_safe_path(rel_path)
        if not target.exists():
            return f"Error: File '{rel_path}' does not exist in workspace."

        content = target.read_text(encoding="utf-8")
        if old_str not in content:
            return f"Error: Target text to replace was not found in '{rel_path}'."

        count = content.count(old_str)
        updated = content.replace(old_str, new_str, 1)
        target.write_text(updated, encoding="utf-8")
        logger.info("Workspace [%s] edited '%s' (replaced 1 of %d occurrences)", self.workspace_dir.name, rel_path, count)
        return f"Successfully updated '{rel_path}'."

    def read_file(self, rel_path: str, max_chars: int = 8000) -> str:
        """Reads contents of a file within the workspace, truncated to max_chars."""
        target = self._resolve_safe_path(rel_path)
        if not target.exists():
            return f"Error: File '{rel_path}' not found in workspace."

        content = target.read_text(encoding="utf-8", errors="replace")
        if len(content) > max_chars:
            return content[:max_chars] + f"\n... [Truncated: showing first {max_chars} of {len(content)} chars]"
        return content

    def run_shell(self, command: str, timeout: int = 60) -> Dict[str, Any]:
        """
        Executes a shell command directly within the workspace directory.
        Used for builds, npm/pip installations, pytest runs, or downloading models/scripts.
        """
        logger.info("Workspace [%s] executing shell command: %r", self.workspace_dir.name, command)
        start_t = time.time()
        try:
            res = subprocess.run(
                command,
                shell=True,
                cwd=str(self.workspace_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout
            )
            dur = round(time.time() - start_t, 2)
            return {
                "command": command,
                "exit_code": res.returncode,
                "stdout": res.stdout.strip(),
                "stderr": res.stderr.strip(),
                "duration_seconds": dur,
                "success": res.returncode == 0
            }
        except subprocess.TimeoutExpired:
            return {
                "command": command,
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds.",
                "duration_seconds": timeout,
                "success": False
            }
        except Exception as e:
            return {
                "command": command,
                "exit_code": -1,
                "stdout": "",
                "stderr": str(e),
                "duration_seconds": round(time.time() - start_t, 2),
                "success": False
            }

    def list_files(self, rel_path: str = "") -> List[Dict[str, Any]]:
        """Lists files and directories inside workspace or subfolder."""
        base = self._resolve_safe_path(rel_path) if rel_path else self.workspace_dir
        if not base.exists():
            return []

        items = []
        for p in sorted(base.rglob("*")):
            try:
                rel = str(p.relative_to(self.workspace_dir))
                items.append({
                    "path": rel,
                    "is_dir": p.is_dir(),
                    "size_bytes": p.stat().st_size if p.is_file() else 0
                })
            except Exception:
                pass
        return items

    def save_latest_download(
        self,
        target_rel_path: str,
        max_age_seconds: int = 120,
        downloads_dir: Optional[Path] = None
    ) -> Optional[str]:
        """
        Inspects the user's ~/Downloads directory for the newest file created
        within max_age_seconds, moves it into workspace at target_rel_path,
        and returns the destination path.
        """
        dl_dir = downloads_dir or (Path.home() / "Downloads")
        if not dl_dir.exists():
            logger.warning("Downloads directory '%s' does not exist", dl_dir)
            return None

        now = time.time()
        candidates = []
        for item in dl_dir.iterdir():
            if item.is_file() and not item.name.startswith("."):
                try:
                    mtime = item.stat().st_mtime
                    if now - mtime <= max_age_seconds:
                        candidates.append((mtime, item))
                except Exception:
                    pass

        if not candidates:
            logger.warning("No recent download found in '%s' (window: %ds)", dl_dir, max_age_seconds)
            return None

        candidates.sort(key=lambda c: c[0], reverse=True)
        newest_file = candidates[0][1]

        target = self._resolve_safe_path(target_rel_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(newest_file), str(target))
        logger.info("Moved download '%s' -> '%s'", newest_file.name, target)
        return str(target)
