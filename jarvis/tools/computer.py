"""Phase 3 — Computer control engine.

Primary backend: Open Interpreter (https://github.com/OpenInterpreter/open-interpreter)
if installed. Fallback: a hardened local shell tool built on subprocess with
command risk classification from the safety module.

Both are exposed to the LLM as tools:
- ``computer.run_shell``  — run a shell command (YELLOW, escalated to RED for
  dangerous patterns; blocked patterns are refused outright).
- ``computer.run_python`` — run a short Python snippet in a subprocess.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..safety.permissions import RiskLevel, classify_command
from .base import Tool, ToolResult

BLOCKED_HINT = "This command matches a blocked pattern and will never be run."

# Commands we outright refuse regardless of approval.
_FORBIDDEN = (
    "rm -rf /", "mkfs", ":(){:|:&};:", "/dev/sda", "shutdown", "reboot",
)


class ShellTool(Tool):
    name = "computer.run_shell"
    description = ("Run a shell command on this machine and return stdout/stderr. "
                   "Read-only commands run automatically; mutating or dangerous "
                   "commands require approval.")
    risk = RiskLevel.YELLOW
    parameters = {
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout_seconds": {"type": "integer", "description": "Max seconds to wait (default 60)"},
        },
        "required": ["command"],
    }

    def __init__(self, workdir: Optional[Path] = None,
                 interpreter_factory: Optional[Any] = None) -> None:
        self.workdir = workdir or Path.cwd()
        # Open Interpreter is optional; factory injected for tests.
        self._interpreter_factory = interpreter_factory
        self._interpreter: Optional[Any] = None

    def availability(self) -> str:
        try:
            import interpreter  # noqa: F401
            return "open-interpreter available"
        except ImportError:
            return "open-interpreter not installed (using subprocess fallback) — pip install open-interpreter"

    def _get_interpreter(self) -> Optional[Any]:
        if self._interpreter is not None:
            return self._interpreter
        if self._interpreter_factory is not None:
            self._interpreter = self._interpreter_factory()
            return self._interpreter
        try:
            from interpreter import interpreter  # type: ignore
            interpreter.auto_run = False  # human approval handled by our guard
            interpreter.max_output = 2000
            self._interpreter = interpreter
        except ImportError:
            self._interpreter = None
        return self._interpreter

    def run(self, command: str = "", timeout_seconds: int = 60, **_: Any) -> ToolResult:
        command = (command or "").strip()
        if not command:
            return ToolResult.failure("No command provided")
        if classify_command(command) == RiskLevel.RED:
            return ToolResult.failure(f"{BLOCKED_HINT} Refused: `{command}`")

        try:
            completed = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=max(5, min(int(timeout_seconds), 600)),
                cwd=str(self.workdir),
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure(f"Command timed out after {timeout_seconds}s: {command}")
        except OSError as exc:
            return ToolResult.failure(f"Failed to execute: {exc}")

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        output = stdout
        if stderr:
            output = f"{output}\n[stderr]\n{stderr}" if output else stderr
        return ToolResult.success(
            output=output[:8000] or "(no output)",
            exit_code=completed.returncode,
        )


class PythonTool(Tool):
    name = "computer.run_python"
    description = ("Execute a short Python snippet in an isolated subprocess and "
                   "return stdout/stderr. Good for calculations and data wrangling. "
                   "Network and filesystem access are allowed; be careful.")
    risk = RiskLevel.YELLOW
    parameters = {
        "properties": {
            "code": {"type": "string", "description": "Python source code to run"},
        },
        "required": ["code"],
    }

    def run(self, code: str = "", **_: Any) -> ToolResult:
        code = (code or "").strip()
        if not code:
            return ToolResult.failure("No code provided")
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as handle:
            handle.write(code)
            path = handle.name
        try:
            completed = subprocess.run(
                ["python3", path],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.failure("Python execution timed out after 60s")
        finally:
            Path(path).unlink(missing_ok=True)

        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        output = stdout
        if stderr:
            output = f"{output}\n[stderr]\n{stderr}" if output else stderr
        return ToolResult.success(output[:8000] or "(no output)",
                                  exit_code=completed.returncode)


class OpenInterpreterTool(Tool):
    """Run natural-language computer tasks through Open Interpreter when present."""

    name = "computer.task"
    description = ("Give Open Interpreter a natural-language computer task "
                   "(e.g. 'install pandas and list installed packages'). Only "
                   "available when open-interpreter is installed.")
    risk = RiskLevel.YELLOW
    parameters = {
        "properties": {
            "task": {"type": "string", "description": "Natural-language task"},
        },
        "required": ["task"],
    }

    def __init__(self, llm_config: Optional[Any] = None) -> None:
        self.llm_config = llm_config

    def availability(self) -> str:
        try:
            import interpreter  # noqa: F401
            return "open-interpreter available"
        except ImportError:
            return "open-interpreter not installed — pip install open-interpreter"

    def run(self, task: str = "", **_: Any) -> ToolResult:
        task = (task or "").strip()
        if not task:
            return ToolResult.failure("No task provided")
        try:
            from interpreter import interpreter  # type: ignore
        except ImportError:
            return ToolResult.failure(
                "open-interpreter is not installed. Install with: "
                "pip install open-interpreter")

        if self.llm_config is not None:
            try:
                interpreter.llm.api_base = self.llm_config.base_url
                interpreter.llm.api_key = self.llm_config.api_key or "missing-key"
                interpreter.llm.model = self.llm_config.model("reasoning")
            except Exception:  # noqa: BLE001 - fall back to OI defaults
                pass
        try:
            chunks = interpreter.chat(task, stream=False, display=False)
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"Open Interpreter failed: {exc}")
        text = ""
        if isinstance(chunks, str):
            text = chunks
        else:
            text = " ".join(str(c) for c in chunks) if chunks else ""
        return ToolResult.success(text[:8000] or "(no output)")


def register_computer_tools(registry: Any, workdir: Optional[Path] = None) -> None:
    registry.register(ShellTool(workdir=workdir))
    registry.register(PythonTool())
    registry.register(OpenInterpreterTool())


__all__ = ["ShellTool", "PythonTool", "OpenInterpreterTool", "register_computer_tools",
           "classify_command"]
