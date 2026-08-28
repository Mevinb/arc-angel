"""SKILL.md library: index/search/load, tools, and app wiring.

Tests build a small hermetic content tree in ``tmp_path`` (index + a couple of
SKILL.md files) rather than depending on the full installed collection, so they
are fast and deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jarvis.app import JarvisApp
from jarvis.config import load_config
from jarvis.safety.permissions import PermissionGuard
from jarvis.skills.index import SkillLibrary
from jarvis.skills.tools import (SkillsCategoriesTool, SkillsListTool,
                                 SkillsLoadTool, SkillsSearchTool,
                                 register_skills_tools)
from jarvis.tools.base import ToolRegistry

QA_SKILL = """---
name: qa-expert
version: 1.0.0
description: Comprehensive quality assurance strategy and test planning.
author: luo-kai
tags: [qa-expert, testing]
---
# Qa Expert

When invoked:
1. Review test coverage
2. Plan test strategy
"""

SQL_SKILL = """---
name: mssql
version: 1.0.0
description: Microsoft SQL Server performance and tuning expertise.
author: luo-kai
tags: [sql, database]
---
# SQL Server

Optimize queries, indexes and execution plans.
"""


def _make_index(content_root: Path) -> SkillLibrary:
    """Build a SkillsLibrary root mirroring the real collection layout:
    ``skills-index.json`` in the root and SKILL.md files under a ``content/``
    subdir. Index paths use the upstream ``ai-agent-skills/`` prefix and
    URL-encoded segments (as the shipped index does)."""
    content_root.mkdir(parents=True, exist_ok=True)
    encoded_dir = "security (by Luo Kai)"  # matches the decoded index segment
    content = content_root / "content" / encoded_dir
    (content / "qa").mkdir(parents=True, exist_ok=True)
    (content / "qa" / "SKILL.md").write_text(QA_SKILL, encoding="utf-8")
    (content / "db").mkdir(parents=True, exist_ok=True)
    (content / "db" / "SKILL.md").write_text(SQL_SKILL, encoding="utf-8")
    (content / "db" / "notes.txt").write_text("ignore me", encoding="utf-8")

    # URL-encoded " (by Luo Kai)"-style segment, as the real index uses.
    encoded_dir = "security%20%28by%20Luo%20Kai%29"
    index = {
        "total": 2,
        "cats": {"01": "Testing", "03": "Databases"},
        "skills": [
            {"n": "qa-expert", "c": "01", "d": "Quality assurance strategy.",
             "p": f"ai-agent-skills/{encoded_dir}/qa/SKILL.md"},
            {"n": "mssql", "c": "03", "d": "SQL Server performance.",
             "p": f"ai-agent-skills/{encoded_dir}/db/SKILL.md"},
        ],
    }
    (content_root / "skills-index.json").write_text(
        json.dumps(index), encoding="utf-8")
    return SkillLibrary(content_root)


def _skilled_registry(tmp_path: Path) -> ToolRegistry:
    library = _make_index(tmp_path / "skills")
    registry = ToolRegistry(PermissionGuard(mode="auto"))
    register_skills_tools(registry, library)
    return registry


# ------------------------------------------------------------------ library
class TestSkillLibrary:
    def test_loads_index(self, tmp_path: Path):
        lib = _make_index(tmp_path / "skills")
        assert lib.count == 2
        assert lib.content_available is True
        assert "Testing" in lib.categories.values()
        assert lib.get("qa-expert").name == "qa-expert"

    def test_search_by_description(self, tmp_path: Path):
        lib = _make_index(tmp_path / "skills")
        results = lib.search("quality assurance")
        assert any(e.name == "qa-expert" for e in results)
        sql = lib.search("sql performance")
        assert any(e.name == "mssql" for e in sql)

    def test_search_empty_returns_limited(self, tmp_path: Path):
        lib = _make_index(tmp_path / "skills")
        assert lib.search("", limit=2) == lib.search("", limit=2)  # stable
        assert len(lib.search("", limit=1)) == 1

    def test_get_case_insensitive_and_missing(self, tmp_path: Path):
        lib = _make_index(tmp_path / "skills")
        assert lib.get("QA-EXPERT") is not None
        assert lib.get("does-not-exist") is None

    def test_load_returns_skill_content(self, tmp_path: Path):
        lib = _make_index(tmp_path / "skills")
        content = lib.load("qa-expert")
        assert "Review test coverage" in content
        # index paths use URL-encoded segments and are resolved to disk
        assert lib.get("mssql").path.endswith("db/SKILL.md")

    def test_load_missing_name_empty(self, tmp_path: Path):
        assert _make_index(tmp_path / "skills").load("nope") == ""

    def test_categories_with_counts(self, tmp_path: Path):
        lib = _make_index(tmp_path / "skills")
        assert lib.categories_with_counts() == {"Databases": 1, "Testing": 1}

    def test_frontmatter_scan_when_no_index(self, tmp_path: Path):
        content = tmp_path / "skills" / "content"
        (content / "qa").mkdir(parents=True)
        (content / "qa" / "SKILL.md").write_text(QA_SKILL, encoding="utf-8")
        lib = SkillLibrary(tmp_path / "skills")
        assert lib.count == 1
        assert lib.get("qa-expert") is not None
        content_back = lib.load("qa-expert")
        assert "Review test coverage" in content_back


# ------------------------------------------------------------------ tools
class TestSkillsTools:
    def test_search_tool_result(self, tmp_path: Path):
        reg = _skilled_registry(tmp_path)
        result = reg.call("skills.search", {"query": "quality assurance"})
        assert result.ok
        assert any("qa-expert" in line for line in result.output.splitlines())
        assert result.data["count"] >= 1

    def test_search_tool_no_match(self, tmp_path: Path):
        reg = _skilled_registry(tmp_path)
        result = reg.call("skills.search", {"query": "zz no such"})
        assert not result.ok and "No skills match" in result.output

    def test_list_tool_and_category_filter(self, tmp_path: Path):
        reg = _skilled_registry(tmp_path)
        all_ = reg.call("skills.list", {})
        assert all_.ok and all_.data["total"] == 2
        testing = reg.call("skills.list", {"category": "Testing"})
        assert testing.ok
        assert any("qa-expert" in line for line in testing.output.splitlines())
        bad = reg.call("skills.list", {"category": "Nope"})
        assert not bad.ok

    def test_load_tool_and_unknown(self, tmp_path: Path):
        reg = _skilled_registry(tmp_path)
        ok = reg.call("skills.load", {"name": "qa-expert"})
        assert ok.ok and "Review test coverage" in ok.output
        missing = reg.call("skills.load", {"name": "nope"})
        assert not missing.ok and "No skill named" in missing.output

    def test_categories_tool(self, tmp_path: Path):
        reg = _skilled_registry(tmp_path)
        result = reg.call("skills.categories", {})
        assert result.ok
        assert result.data["total"] == 2

    def test_tools_are_green(self, tmp_path: Path):
        reg = _skilled_registry(tmp_path)
        for name in ("skills.search", "skills.list", "skills.load",
                     "skills.categories"):
            assert reg.get(name) is not None
            assert reg.get(name).risk.name == "GREEN"


# ------------------------------------------------------------------- wiring
class TestSkillsWiring:
    def test_skills_tools_registered_and_report(self, tmp_path: Path):
        config = load_config(project_root=tmp_path)
        config.safety_mode = "auto"
        config.skills = {"root": str(tmp_path / "skills")}
        _make_index(tmp_path / "skills")
        app = JarvisApp(config=config, quiet=True)
        try:
            for name in ("skills.search", "skills.list", "skills.load",
                         "skills.categories"):
                assert name in app.registry.names()
            report = app.health_report()
            assert report["skills"]["available"] is True
            assert report["skills"]["count"] >= 1
        finally:
            app.close()

    def test_no_skills_root_still_works(self, tmp_path: Path):
        config = load_config(project_root=tmp_path)
        config.safety_mode = "auto"
        app = JarvisApp(config=config, quiet=True)
        try:
            # No skills dir -> no skill tools, but app still healthy.
            assert "skills.search" not in app.registry.names()
            assert app.health_report()["skills"]["available"] is False
        finally:
            app.close()
