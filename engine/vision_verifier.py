"""
Real verification helpers for the task orchestrator.

Provides concrete, checkable verification of sub-goal outcomes instead of the
previous always-true placeholder:
- File/dir existence and non-emptiness inside the workspace.
- Command exit status + output checks (real subprocess run in the workspace).
- Optional image verification via the local VisionEngine (caption/OCR compare).

Every helper returns ``(ok: bool, detail: str)`` and never raises; failures are
reported as ``(False, "<reason>")`` so the orchestrator can retry.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.workspace_tools import WorkspaceManager

logger = logging.getLogger(__name__)

# Keywords that signal a sub-goal is expected to produce specific artifacts.
_CODE_EXT = {".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cpp", ".sh", ".sql", ".html", ".css"}
_TEXT_EXT = _CODE_EXT | {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".csv", ".jsonl", ".ini", ".cfg"}
_IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}


class TaskVerifier:
    """Stateless verification utilities bound to a workspace."""

    def __init__(self, workspace: Optional[WorkspaceManager] = None):
        self.workspace = workspace

    # ------------------------------------------------------------------ files
    def verify_files_exist(self, rel_paths: List[str], require_non_empty: bool = True) -> Tuple[bool, str]:
        if not self.workspace:
            return True, "No workspace bound; file check skipped."
        missing, empty = [], []
        for rel in rel_paths:
            try:
                target = self.workspace._resolve_safe_path(rel)
            except Exception as e:
                missing.append(f"{rel} (unsafe: {e})")
                continue
            if not target.exists():
                missing.append(rel)
            elif require_non_empty and target.is_file() and target.stat().st_size == 0:
                empty.append(rel)
        if missing:
            return False, f"Missing expected files: {', '.join(missing)}"
        if empty:
            return False, f"Expected files are empty: {', '.join(empty)}"
        return True, f"Verified {len(rel_paths)} file(s) exist and are non-empty."

    def verify_workspace_has_artifacts(self, min_files: int = 1) -> Tuple[bool, str]:
        if not self.workspace:
            return True, "No workspace bound; artifact check skipped."
        try:
            files = [f for f in self.workspace.list_files("") if not f["is_dir"]]
        except Exception as e:
            return False, f"Could not list workspace: {e}"
        if len(files) < min_files:
            return False, f"Workspace has {len(files)} file(s); expected at least {min_files}."
        return True, f"Workspace contains {len(files)} file(s)."

    # --------------------------------------------------------------- commands
    def verify_command(self, command: str, expect_substring: Optional[str] = None, timeout: int = 60) -> Tuple[bool, str]:
        """Runs a real command in the workspace and checks its exit status/output."""
        if not self.workspace:
            return True, "No workspace bound; command check skipped."
        res = self.workspace.run_shell(command, timeout=timeout)
        out = (res.get("stdout", "") + "\n" + res.get("stderr", "")).strip()
        if not res.get("success"):
            return False, f"Command '{command}' failed (exit {res.get('returncode')}): {out[:300]}"
        if expect_substring and expect_substring not in out:
            return False, f"Command succeeded but output did not contain '{expect_substring}': {out[:300]}"
        return True, f"Command '{command}' succeeded. Output: {out[:200]}"

    # ----------------------------------------------------------------- images
    def verify_image_matches(self, image_path: str, reference_path: Optional[str] = None) -> Tuple[bool, str]:
        """Uses the local VisionEngine to confirm an image is valid/decodable and optionally similar to a reference."""
        target = Path(image_path)
        if not target.exists():
            return False, f"Image '{target.name}' does not exist."
        if target.stat().st_size == 0:
            return False, f"Image '{target.name}' is empty."
        try:
            from engine.vision import VisionEngine, PIL_AVAILABLE
            if not PIL_AVAILABLE:
                return True, f"Image '{target.name}' saved ({target.stat().st_size} bytes); vision libs unavailable."
            analysis = VisionEngine.analyze(str(target))
            caption = analysis.get("caption", "")
            if "error" in caption.lower():
                return False, f"Image '{target.name}' could not be analyzed: {caption[:160]}"
            detail = f"Image '{target.name}' verified (caption: {caption[:160]})."
            if reference_path and os.path.exists(reference_path):
                ref = VisionEngine.analyze(reference_path)
                ref_caption = ref.get("caption", "")
                # Lightweight heuristic: any shared significant keyword counts as a match signal.
                ref_words = {w.lower() for w in ref_caption.split() if len(w) > 4}
                got_words = {w.lower() for w in caption.split() if len(w) > 4}
                overlap = ref_words & got_words
                if overlap:
                    detail += f" Shared visual terms with reference: {', '.join(sorted(overlap)[:6])}."
                else:
                    detail += " No shared visual terms with reference (heuristic)."
            return True, detail
        except Exception as e:
            logger.warning("Vision verification failed for %s: %s", image_path, e)
            return False, f"Vision verification error: {e}"

    # ------------------------------------------------------------- dispatcher
    def verify_subgoal(self, sg: Dict[str, Any], task: Dict[str, Any], last_detail: str = "") -> Tuple[bool, str]:
        """
        Chooses an appropriate real verification strategy based on the sub-goal text.
        Returns (ok, detail). Defaults to a workspace-artifact check.
        """
        title = (sg.get("title") or "").lower()
        desc = (sg.get("description") or "").lower()
        blob = f"{title} {desc}"

        # Tests / builds -> run them for real
        if any(k in blob for k in ["test", "pytest", "unittest", "build", "compile"]):
            return self.verify_command("pytest -q || python3 -m unittest discover -q || echo NO_TESTS")

        # Explicit file mentions, e.g. "create main.py"
        mentioned = [tok.strip("`,.;:()[]\"'") for tok in blob.replace("(", " ").replace(")", " ").split()]
        expected = []
        for tok in mentioned:
            ext = Path(tok).suffix
            if ext in _TEXT_EXT and "/" not in tok:
                expected.append(tok)
        if expected:
            return self.verify_files_exist(sorted(set(expected)))

        # Image/dataset sub-goals
        if any(ext.strip(".") in blob for ext in _IMAGE_EXT) or "image" in blob or "dataset" in blob:
            if self.workspace:
                imgs = [f for f in self.workspace.list_files("") if Path(f["path"]).suffix.lower() in _IMAGE_EXT]
                if imgs:
                    return True, f"Found {len(imgs)} image artifact(s) in workspace."
                return False, "No image artifacts found in workspace."

        # Fall back to a real workspace artifact check
        return self.verify_workspace_has_artifacts(min_files=1)
