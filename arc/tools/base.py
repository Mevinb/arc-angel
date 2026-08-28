"""Tool framework: every ARC capability is a ``Tool`` with a JSON schema,
a risk level, and a ``run()`` method. The registry gates calls through the
permission system and exposes OpenAI function-calling schemas for the LLM."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..safety.permissions import Action, PermissionGuard, RiskLevel

logger = logging.getLogger("arc.tools")

#: Some gateways front Anthropic/Gemini-style APIs that forbid "." in tool
#: names. They sanitize ``memory.remember`` to ``memory_remember`` (sometimes
#: appending a hex suffix like ``memory_remember_64ed5d80c28d...``) and fail to
#: restore the original name in tool-call responses. We match those back.
_SANITIZED_HASH = re.compile(r"_[0-9a-f]{8,}$")


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        if self.ok:
            return self.output
        return f"ERROR: {self.output}"

    @classmethod
    def success(cls, output: str = "", **data: Any) -> "ToolResult":
        return cls(ok=True, output=output, data=data)

    @classmethod
    def failure(cls, message: str, **data: Any) -> "ToolResult":
        return cls(ok=False, output=message, data=data)


class Tool(ABC):
    """Base class for all ARC tools."""

    #: unique name, e.g. "jobs.search"
    name: str = "tool"
    #: one-line description shown to the LLM
    description: str = ""
    #: default risk classification (see safety.permissions)
    risk: RiskLevel = RiskLevel.GREEN
    #: JSON-schema style parameters for function calling
    parameters: Dict[str, Any] = field(default_factory=dict)

    @abstractmethod
    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool. Must never raise — return ToolResult.failure."""

    # ---------------------------------------------------------------- schema
    def to_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters.get("properties", {}),
                    "required": self.parameters.get("required", []),
                },
            },
        }

    def describe_action(self, kwargs: Dict[str, Any]) -> str:
        """Human-readable one-liner used in approval prompts."""
        preview = ", ".join(f"{k}={str(v)[:60]!r}" for k, v in list(kwargs.items())[:3])
        return f"{self.name}({preview})"


class ToolRegistry:
    """Holds tools, validates calls, enforces permissions."""

    def __init__(self, guard: PermissionGuard) -> None:
        self.guard = guard
        self._tools: Dict[str, Tool] = {}
        self._hooks: List[Callable[[str, Dict[str, Any], ToolResult], None]] = []

    # ------------------------------------------------------------------ api
    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug("Registered tool %s", tool.name)
        return tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def resolve(self, name: str) -> Optional[Tool]:
        """Look up a tool, tolerating provider-sanitized names.

        Exact match first. If the name was sanitized by the gateway
        (dots -> underscores, optional hex suffix), match it against the
        sanitized form of every registered tool. Ambiguous mappings
        (two tools sanitize to the same string) resolve to None.
        """
        tool = self._tools.get(name)
        if tool is not None:
            return tool
        if "." not in name and "_" in name:
            base = _SANITIZED_HASH.sub("", name)
            matches = {t for t in self._tools.values()
                       if t.name.replace(".", "_") in {base, name}}
            if len(matches) == 1:
                resolved = matches.pop()
                logger.info("Resolved sanitized tool name %r -> %r", name, resolved.name)
                return resolved
        return None

    def names(self) -> List[str]:
        return sorted(self._tools)

    def all_tools(self) -> List[Tool]:
        return list(self._tools.values())

    def schemas(self) -> List[Dict[str, Any]]:
        return [tool.to_schema() for tool in self._tools.values()]

    def on_result(self, hook: Callable[[str, Dict[str, Any], ToolResult], None]) -> None:
        """Register a callback invoked after every tool call (for UI logging)."""
        self._hooks.append(hook)

    def call(self, name: str, kwargs: Optional[Dict[str, Any]] = None) -> ToolResult:
        """Run a tool after permission checks. Errors become failed results."""
        kwargs = dict(kwargs or {})
        tool = self.resolve(name)
        if tool is None:
            return ToolResult.failure(f"Unknown tool: {name}. Available: {', '.join(self.names())}")

        # Risk classification: the tool's own level, possibly escalated by the
        # permission guard (e.g. dangerous command patterns in details).
        details = str(kwargs.get("command", "") or kwargs.get("url", "") or "")
        risk = self.guard.risk_for(tool.name, details, default=tool.risk)

        action = Action(
            tool=tool.name,
            description=tool.describe_action(kwargs),
            details=details,
            risk=risk,
        )
        try:
            self.guard.check(action)
        except Exception as exc:  # PermissionDenied
            logger.info("Denied tool call %s: %s", tool.name, exc)
            result = ToolResult.failure(f"Permission denied: {exc}", denied=True)
            self._notify(tool.name, kwargs, result)
            return result

        try:
            result = tool.run(**kwargs)
        except Exception as exc:  # noqa: BLE001 - tools must not crash the agent
            logger.exception("Tool %s crashed", tool.name)
            result = ToolResult.failure(f"{type(exc).__name__}: {exc}")
        self._notify(tool.name, kwargs, result)
        return result

    def _notify(self, name: str, kwargs: Dict[str, Any], result: ToolResult) -> None:
        for hook in self._hooks:
            try:
                hook(name, kwargs, result)
            except Exception:  # noqa: BLE001 - hooks must not break execution
                logger.exception("Tool hook failed")

    def availability(self) -> Dict[str, str]:
        """Status of optional heavyweight engines for `arc doctor`."""
        statuses: Dict[str, str] = {}
        for tool in self._tools.values():
            if hasattr(tool, "availability"):
                statuses[tool.name] = getattr(tool, "availability")()
        return statuses


def simple_tool(name: str, description: str, fn: Callable[..., ToolResult],
                parameters: Optional[Dict[str, Any]] = None,
                risk: RiskLevel = RiskLevel.GREEN) -> Tool:
    """Wrap a plain function as a Tool (for lightweight integrations)."""

    class FunctionTool(Tool):
        pass

    FunctionTool.name = name
    FunctionTool.description = description
    FunctionTool.parameters = parameters or {"properties": {}, "required": []}
    FunctionTool.risk = risk

    def run(**kwargs: Any) -> ToolResult:
        return fn(**kwargs)

    FunctionTool.run = staticmethod(run)
    return FunctionTool()
