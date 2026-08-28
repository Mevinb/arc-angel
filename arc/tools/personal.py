"""Phase 7 tools — long-term memory and profile access for the LLM.

These give the agent durable memory across sessions (backed by SQLite) and
read access to the personal profile.
"""

from __future__ import annotations

from typing import Any

from ..core.memory import LongTermMemory
from ..profile.profile import Profile
from ..safety.permissions import RiskLevel
from .base import Tool, ToolResult


class MemoryRememberTool(Tool):
    name = "memory.remember"
    description = ("Save a durable fact about the user for future sessions "
                   "(e.g. key='user.timezone', value='Europe/Berlin'). Use for "
                   "preferences, ongoing situations and standing instructions.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "key": {"type": "string", "description": "Stable identifier, e.g. 'user.timezone'"},
            "value": {"type": "string", "description": "The fact to remember"},
        },
        "required": ["key", "value"],
    }

    def __init__(self, memory: LongTermMemory) -> None:
        self.memory = memory

    def run(self, key: str = "", value: str = "", **_: Any) -> ToolResult:
        if not key or not value:
            return ToolResult.failure("Both key and value are required")
        self.memory.remember(key, value)
        return ToolResult.success(f"Remembered {key!r}.")


class MemoryRecallTool(Tool):
    name = "memory.recall"
    description = ("Recall a remembered fact by key, or list all memories when "
                   "no key is given.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "key": {"type": "string", "description": "Fact key (optional — omit to list all)"},
        },
        "required": [],
    }

    def __init__(self, memory: LongTermMemory) -> None:
        self.memory = memory

    def run(self, key: str = "", **_: Any) -> ToolResult:
        if key:
            value = self.memory.recall(key)
            return (ToolResult.success(f"{key} = {value}") if value is not None
                    else ToolResult.failure(f"Nothing remembered for {key!r}"))
        facts = self.memory.recall_all()
        if not facts:
            return ToolResult.success("No long-term memories yet.")
        return ToolResult.success("\n".join(f"{k} = {v}" for k, v in facts.items()))


class MemoryForgetTool(Tool):
    name = "memory.forget"
    description = "Delete a remembered fact by key."
    risk = RiskLevel.YELLOW
    parameters = {
        "properties": {"key": {"type": "string"}},
        "required": ["key"],
    }

    def __init__(self, memory: LongTermMemory) -> None:
        self.memory = memory

    def run(self, key: str = "", **_: Any) -> ToolResult:
        if not key:
            return ToolResult.failure("No key provided")
        self.memory.forget(key)
        return ToolResult.success(f"Forgot {key!r}.")


class ProfileSearchTool(Tool):
    name = "profile.search"
    description = ("Search the user's saved profile: projects by query, or the "
                   "full summary when no query is given. Answers 'what projects "
                   "do I have with Python?' style questions.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "query": {"type": "string", "description": "Search projects for this text (optional)"},
        },
        "required": [],
    }

    def __init__(self, profile: Profile) -> None:
        self.profile = profile

    def run(self, query: str = "", **_: Any) -> ToolResult:
        if query:
            matches = self.profile.search_projects(query)
            if not matches:
                return ToolResult.failure(f"No projects match {query!r}")
            lines = []
            for project in matches:
                tech = ", ".join(str(t) for t in project.get("tech", []))
                lines.append(f"- {project.get('name', '?')} [{tech}]: "
                             f"{project.get('description', '')}")
            return ToolResult.success("\n".join(lines))
        return ToolResult.success(self.profile.summarize())


def register_personal_tools(registry: Any, memory: LongTermMemory,
                            profile: Profile) -> None:
    registry.register(MemoryRememberTool(memory))
    registry.register(MemoryRecallTool(memory))
    registry.register(MemoryForgetTool(memory))
    registry.register(ProfileSearchTool(profile))


__all__ = ["register_personal_tools", "MemoryRememberTool", "MemoryRecallTool",
           "MemoryForgetTool", "ProfileSearchTool"]
