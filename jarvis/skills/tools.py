"""JARVIS tools exposing the skill library to the agent.

These let the orchestrator *discover* (search/list) and *activate* (load) a
SKILL.md skill as context. Loading is a read-only, GREEN-risk operation that
injects a single skill's instructions into the conversation so the model can
follow its guidance.
"""

from __future__ import annotations

from typing import Any

from ..safety.permissions import RiskLevel
from ..tools.base import Tool, ToolResult
from .index import SkillLibrary, SkillEntry


class SkillsSearchTool(Tool):
    name = "skills.search"
    description = ("Search the installed skill & component library for a task "
                   "or domain (e.g. 'sql performance', 'react testing', 'gcp "
                   "deployment', 'filesystem mcp', 'git branch skill'). Returns "
                   "matching entries — expert SKILL.md skills plus catalog "
                   "components (MCPs, loops, subagents, hooks, plugins, prompts, "
                   "CLI tools) — with kind, category and description so you can "
                   "pick one to load.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "query": {"type": "string", "description": "Task or domain to search for"},
            "limit": {"type": "integer", "description": "Max results (default 10)"},
            "kind": {"type": "string", "description": "Optional filter: skill, mcp, loop, subagent, hook, plugin, prompt, tool"},
        },
        "required": ["query"],
    }

    def __init__(self, library: SkillLibrary) -> None:
        self.library = library

    def run(self, query: str = "", limit: int = 10, kind: str = "", **_: Any) -> ToolResult:
        if not self.library.count:
            return ToolResult.failure("No skills indexed. Run `jarvis skills "
                                      "install` to install the collection.")
        results = self.library.search(query, limit=max(1, int(limit)))
        if kind:
            kind = kind.strip().lower()
            results = [e for e in results if (e.kind or "skill") == kind]
        if not results:
            return ToolResult.failure(f"No skills match {query!r}. Try "
                                      "`skills.categories` or a broader term.")
        lines = [f"{i+1}. {_fmt_entry(e)}" for i, e in enumerate(results)]
        return ToolResult.success(
            "\n".join(lines),
            count=len(results),
            skills=[e.to_schema() for e in results],
        )


class SkillsListTool(Tool):
    name = "skills.list"
    description = ("List installed skills, optionally filtered by category "
                   "(use `skills.categories` to see valid names). Returns name, "
                   "category and description for each.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "category": {"type": "string", "description": "Filter by category (optional)"},
            "limit": {"type": "integer", "description": "Max results (default 50)"},
        },
        "required": [],
    }

    def __init__(self, library: SkillLibrary) -> None:
        self.library = library

    def run(self, category: str = "", limit: int = 50, **_: Any) -> ToolResult:
        if not self.library.count:
            return ToolResult.failure("No skills indexed.")
        entries = self.library.list_by_category(category)
        if category and not entries:
            return ToolResult.failure(f"No skills in category {category!r}. "
                                      "See `skills.categories`.")
        entries = entries[:max(1, int(limit))]
        lines = [f"{i+1}. {_fmt_entry(e)}" for i, e in enumerate(entries)]
        return ToolResult.success(
            "\n".join(lines) or "(empty)",
            count=len(entries),
            total=self.library.count,
            skills=[e.to_schema() for e in entries],
        )


class SkillsLoadTool(Tool):
    name = "skills.load"
    description = ("Load the full detail of a named skill or component (from "
                   "`skills.search`). For an expert SKILL.md skill returns its "
                   "complete instructions as context; for a catalog component "
                   "(MCP/loop/hook/plugin/tool...) returns its description, "
                   "source and the exact install command.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "name": {"type": "string", "description": "Exact name to load"},
        },
        "required": ["name"],
    }

    def __init__(self, library: SkillLibrary) -> None:
        self.library = library

    def run(self, name: str = "", **_: Any) -> ToolResult:
        entry = self.library.get(name)
        if entry is None:
            return ToolResult.failure(
                f"No skill named {name!r}. Use `skills.search` to find one.")
        content = self.library.load(name)
        if not content:
            return ToolResult.failure(
                f"Skill {name!r} is indexed but its content is not installed. "
                "Run `jarvis skills install`.")
        kind = entry.kind or "skill"
        header = f"# {kind.title()}: {entry.name}\nCategory: {entry.category}\n"
        return ToolResult.success(
            header + "\n" + content,
            name=entry.name,
            category=entry.category,
            kind=kind,
            characters=len(content),
        )


class SkillsCategoriesTool(Tool):
    name = "skills.categories"
    description = ("List the skill categories and how many skills each holds, "
                   "to browse the library by domain before searching.")
    risk = RiskLevel.GREEN
    parameters = {"properties": {}, "required": []}

    def __init__(self, library: SkillLibrary) -> None:
        self.library = library

    def run(self, **_: Any) -> ToolResult:
        if not self.library.count:
            return ToolResult.failure("No skills indexed.")
        counts = self.library.categories_with_counts()
        lines = [f"- {cat} ({count})" for cat, count in counts.items()]
        return ToolResult.success(
            "\n".join(lines), total=self.library.count, categories=counts)


def _fmt_entry(entry: SkillEntry) -> str:
    badge = f"({entry.kind})" if entry.kind else "(skill)"
    if entry.name.endswith(f" {badge}"):
        badge = ""  # name already carries the kind suffix (collision rename)
    head = f"{entry.name}  {badge}".rstrip()
    return f"{head} [{entry.category}] — {entry.description}"


def register_skills_tools(registry: Any, library: SkillLibrary) -> None:
    registry.register(SkillsSearchTool(library))
    registry.register(SkillsListTool(library))
    registry.register(SkillsLoadTool(library))
    registry.register(SkillsCategoriesTool(library))


__all__ = ["register_skills_tools", "SkillsSearchTool", "SkillsListTool",
           "SkillsLoadTool", "SkillsCategoriesTool"]
