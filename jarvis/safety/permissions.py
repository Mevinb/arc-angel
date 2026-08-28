"""Phase 9 — Safety & permission system.

Every JARVIS action is classified into one of three risk levels:

- GREEN  (automatic):        read emails, search the web, analyze jobs, read
                             files, summarize information.
- YELLOW (approval required): create email drafts, modify files, fill forms,
                             download files, change application info.
- RED    (explicit approval): send emails, submit applications, delete files,
                             run dangerous commands, purchase, publish.

Modes:
- ``auto``        green actions run; yellow/red are denied (used by automation).
- ``interactive`` green runs; yellow/red prompt the human (default).
- ``yolo``        everything runs without asking (explicit opt-in, discouraged).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional

logger = logging.getLogger("jarvis.safety")


class RiskLevel(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass
class Action:
    """A concrete action JARVIS wants to perform."""
    tool: str
    description: str
    details: str = ""
    risk: RiskLevel = RiskLevel.GREEN


@dataclass
class Decision:
    allowed: bool
    risk: RiskLevel
    reason: str = ""


# Approver signature: (action) -> True/False. Provided by the UI (rich prompt)
# or by automation (always False for yellow/red in auto mode).
Approver = Callable[[Action], bool]

ALWAYS_ASK_TOOLS: Dict[str, RiskLevel] = {
    "email.send": RiskLevel.RED,
    "email.create_draft": RiskLevel.YELLOW,
    "application.submit": RiskLevel.RED,
    "browser.task": RiskLevel.GREEN,  # browser automation: auto-run (user opted in)
    "browser.fill_form": RiskLevel.YELLOW,
    "browser.download": RiskLevel.YELLOW,
    "computer.run": RiskLevel.YELLOW,  # may be escalated to RED by command scan
    "computer.run_shell": RiskLevel.YELLOW,
    "computer.run_python": RiskLevel.YELLOW,
    "computer.task": RiskLevel.YELLOW,
    "jobs.update_status": RiskLevel.YELLOW,
    "file.write": RiskLevel.YELLOW,
    "file.delete": RiskLevel.RED,
    "publish": RiskLevel.RED,
    "purchase": RiskLevel.RED,
}

# Severity ordering — RiskLevel is a str Enum, so ``>`` between members would
# compare alphabetically ("green" < "red" < "yellow"). Always order through this.
RISK_SEVERITY = {RiskLevel.GREEN: 0, RiskLevel.YELLOW: 1, RiskLevel.RED: 2}


def highest_risk(*risks: RiskLevel) -> RiskLevel:
    """Return the most severe of the given risk levels."""
    return max(risks, key=lambda risk: RISK_SEVERITY[risk])

# Commands that are outright blocked (never run, regardless of approval).
BLOCKED_COMMAND_PATTERNS = [
    r"\brm\s+(-[a-zA-Z]*[rf][a-zA-Z]*\s+)+/*\s*$",  # rm -rf /
    r"\bmkfs\b",
    r"\bdd\s+if=.*of=/dev/(sd|nvme)",
    r">\s*/dev/sd[a-z]",
    r"\bshutdown\b|\breboot\b",
    r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;:",              # fork bomb
    r"\bchmod\s+-R\s+777\s+/\b",
]

# Commands that escalate computer.run to RED.
DANGEROUS_COMMAND_PATTERNS = [
    r"\bsudo\b",
    r"\brm\s+(-[a-zA-Z]*[rf])",
    r"\bcurl\b[^|]*\|\s*(ba)?sh",
    r"\bwget\b[^|]*\|\s*(ba)?sh",
    r"\bgit\s+push\b.*--force",
    r"\bpip\s+install\b.*--break-system-packages",
    r"\bapt(-get)?\s+(remove|purge)\b",
    r"\bkill(all)?\s+-9\b",
    r">\s*/dev/nvme",
]


def classify_command(command: str) -> RiskLevel:
    """Static analysis of a shell command for destructive patterns."""
    for pattern in BLOCKED_COMMAND_PATTERNS:
        if re.search(pattern, command):
            return RiskLevel.RED
    for pattern in DANGEROUS_COMMAND_PATTERNS:
        if re.search(pattern, command):
            return RiskLevel.RED
    # Mutating commands -> yellow
    if re.search(r"^\s*(pip|npm|apt|apt-get|cargo|uv)\s+(install|uninstall|remove)", command):
        return RiskLevel.YELLOW
    if re.search(r"[>|]{1,2}\s*\S+", command) and not command.strip().startswith("echo"):
        return RiskLevel.YELLOW  # writes output somewhere
    if re.search(r"^\s*(mv|cp|touch|mkdir|tee|sed\s+-i|chmod|chown)\b", command):
        return RiskLevel.YELLOW
    return RiskLevel.GREEN


class PermissionDenied(Exception):
    """Raised when an action is blocked by the permission system."""


class PermissionGuard:
    """Gate every action through its risk level and the configured mode."""

    def __init__(self, mode: str = "interactive", approver: Optional[Approver] = None) -> None:
        if mode not in ("auto", "interactive", "yolo"):
            raise ValueError(f"Unknown safety mode: {mode!r}")
        self.mode = mode
        self.approver = approver
        self._counts = {"approved": 0, "denied": 0, "auto_green": 0, "blocked": 0}

    # ------------------------------------------------------------------ api
    def risk_for(self, tool: str, details: str = "",
                 default: RiskLevel = RiskLevel.GREEN) -> RiskLevel:
        """Risk of a tool call. Starts from the ALWAYS_ASK_TOOLS table (or the
        tool's own classification passed as ``default``) and escalates when a
        ``computer.*`` command scan finds dangerous patterns."""
        risk = ALWAYS_ASK_TOOLS.get(tool, default)
        if tool.startswith("computer.") and details:
            command_risk = classify_command(details)
            risk = highest_risk(risk, command_risk)
        return risk

    def evaluate(self, action: Action) -> Decision:
        """Decide whether an action may proceed in the current mode."""
        risk = highest_risk(action.risk,
                            self.risk_for(action.tool, action.details))

        if risk == RiskLevel.GREEN:
            self._counts["auto_green"] += 1
            return Decision(True, risk, "read-only action")

        if self.mode == "yolo":
            logger.warning("YOLO mode: auto-approving %s action %r", risk.value, action.tool)
            return Decision(True, risk, "yolo mode")

        if self.mode == "auto":
            self._counts["denied"] += 1
            return Decision(False, risk, f"{risk.value} actions require human approval "
                                         f"and are denied in auto mode")

        # interactive: ask the human
        if self.approver is None:
            self._counts["denied"] += 1
            return Decision(False, risk, "no approver configured")
        try:
            approved = bool(self.approver(action))
        except Exception as exc:  # noqa: BLE001 - approver must never crash the agent
            logger.error("Approver failed: %s", exc)
            approved = False

        if approved:
            self._counts["approved"] += 1
            return Decision(True, risk, "human approved")
        self._counts["denied"] += 1
        return Decision(False, risk, "human denied")

    def check(self, action: Action) -> None:
        """Raise PermissionDenied if the action is not allowed."""
        decision = self.evaluate(action)
        if not decision.allowed:
            raise PermissionDenied(f"[{decision.risk.value.upper()}] {action.tool}: "
                                   f"{decision.reason}")
        logger.info("Allowed %s action %s (%s)", action.risk.value, action.tool, decision.reason)

    def stats(self) -> Dict[str, int]:
        return dict(self._counts)


@dataclass
class AuditEntry:
    tool: str
    risk: RiskLevel
    allowed: bool
    description: str = ""
    at: float = field(default_factory=lambda: __import__("time").time())
