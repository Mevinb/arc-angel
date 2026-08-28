"""SKILL.md library: index/search/load, tools, and app wiring.

Tests build a small hermetic content tree in ``tmp_path`` (index + a couple of
SKILL.md files) rather than depending on the full installed collection, so they
are fast and deterministic.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc.app import ArcApp
from arc.config import load_config
from arc.safety.permissions import PermissionGuard
from arc.skills.index import SkillLibrary
from arc.skills.tools import (SkillsCategoriesTool, SkillsListTool,
                                 SkillsLoadTool, SkillsSearchTool,
                                 register_skills_tools)
from arc.tools.base import ToolRegistry

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


# --------------------------------------------------------------- catalog source
def _make_catalog_root(root: Path) -> Path:
    """Build a minimal awesome-ai-agent-tools-style catalog root.

    One entry per category (skills, mcps, loops, plugins) exercising the varied
    field names, plus a bundled SKILL.md so load() has real content.
    """
    root.mkdir(parents=True, exist_ok=True)
    (root / "skills").mkdir(parents=True, exist_ok=True)
    (root / "skills" / "catalog.json").write_text(json.dumps({
        "name": "Skills Catalog", "skills": [
            {"id": "executing-plans", "name": "Executing Plans",
             "source": "obra/superpowers", "category": "Development",
             "description": "Executes implementation plans.",
             "install": "npx skills add obra/superpowers --skill executing-plans"},
        ]}), encoding="utf-8")
    (root / "mcps").mkdir(parents=True, exist_ok=True)
    (root / "mcps" / "catalog.json").write_text(json.dumps({
        "servers": [
            {"id": "filesystem", "name": "Filesystem MCP",
             "category": "Official Reference", "description": "File operations.",
             "github": "https://github.com/modelcontextprotocol/servers",
             "install": "npx -y @modelcontextprotocol/server-filesystem /tmp"},
        ]}), encoding="utf-8")
    (root / "loops").mkdir(parents=True, exist_ok=True)
    (root / "loops" / "catalog.json").write_text(json.dumps({
        "loops": [
            {"id": "docs-sweep", "title": "The docs sweep",
             "category": "engineering", "sourceRepo": "Forward-Future/loop-library",
             "description": "Keeps docs aligned.",
             "source": "https://sig.example/docs-sweep"},
        ]}), encoding="utf-8")
    (root / "plugins").mkdir(parents=True, exist_ok=True)
    (root / "plugins" / "catalog.json").write_text(json.dumps({
        "plugins": [
            {"id": "claude-plugins", "name": "Official Claude Plugins",
             "category": "Claude Code", "description": "Plugin marketplace.",
             "websiteUrl": "https://claude.ai/plugins",
             "installCommand": "claude install @anthropics/claude-plugins"},
        ]}), encoding="utf-8")
    bundled = root / "assets" / "skills" / "awesome-ai-agent-tools"
    bundled.mkdir(parents=True, exist_ok=True)
    (bundled / "SKILL.md").write_text(QA_SKILL, encoding="utf-8")
    return root


class TestCatalogSource:
    def test_parses_all_categories(self, tmp_path: Path):
        lib = SkillLibrary(tmp_path / "skills", extra_roots=[_make_catalog_root(tmp_path / "src")])
        kinds = {e.kind for e in lib._entries.values()}
        assert {"skill", "mcps", "loops", "plugins"} <= kinds
        # 4 catalog entries + 1 bundled SKILL.md
        assert lib.count == 5

    def test_kind_fields_populated(self, tmp_path: Path):
        lib = SkillLibrary(tmp_path / "skills", extra_roots=[_make_catalog_root(tmp_path / "src")])
        mcp = lib.get("Filesystem MCP")
        assert mcp.kind == "mcps"
        assert "modelcontextprotocol" in mcp.source_url
        assert "npx -y @modelcontextprotocol" in mcp.install
        plugin = lib.get("Official Claude Plugins")
        assert plugin.kind == "plugins"
        assert plugin.install == "claude install @anthropics/claude-plugins"

    def test_search_across_kinds(self, tmp_path: Path):
        lib = SkillLibrary(tmp_path / "skills", extra_roots=[_make_catalog_root(tmp_path / "src")])
        hits = {e.name: e.kind for e in lib.search("filesystem")}
        assert hits.get("Filesystem MCP") == "mcps"
        loop = lib.search("docs")
        assert any(e.kind == "loops" for e in loop)

    def test_load_catalog_entry_synthesizes(self, tmp_path: Path):
        lib = SkillLibrary(tmp_path / "skills", extra_roots=[_make_catalog_root(tmp_path / "src")])
        content = lib.load("Filesystem MCP")
        assert "Kind: mcps" in content
        assert "Install: npx -y" in content
        assert "Source:" in content

    def test_load_bundled_skill_returns_content(self, tmp_path: Path):
        lib = SkillLibrary(tmp_path / "skills", extra_roots=[_make_catalog_root(tmp_path / "src")])
        content = lib.load("qa-expert")
        assert "Review test coverage" in content

    def test_duplicate_name_disambiguated(self, tmp_path: Path):
        root = tmp_path / "skills"
        root.mkdir(parents=True, exist_ok=True)
        # primary index contains qa-expert; catalog also bundles a qa-expert
        _make_index(root)
        lib = SkillLibrary(root, extra_roots=[_make_catalog_root(tmp_path / "src")])
        assert lib.get("qa-expert") is not None  # primary kept
        assert lib.count >= 3

    def test_collision_entry_resolvable_by_renamed_key(self, tmp_path: Path):
        # A catalog entry whose name collides with a primary skill is renamed
        # with its kind suffix and stays resolvable + loadable.
        root = tmp_path / "skills"
        root.mkdir(parents=True, exist_ok=True)
        _make_index(root)  # primary: qa-expert, mssql
        src = tmp_path / "src"
        (src / "mcps").mkdir(parents=True)
        (src / "mcps" / "catalog.json").write_text(json.dumps({
            "servers": [{"id": "mssql", "name": "mssql",
                         "category": "Databases",
                         "description": "MSSQL MCP server.",
                         "github": "https://github.com/x/mssql",
                         "install": "npx -y mssql-mcp"}]}),
            encoding="utf-8")
        lib = SkillLibrary(root, extra_roots=[src])
        primary = lib.get("mssql")
        assert primary is not None and primary.kind == ""  # primary wins the plain name
        assert "Optimize queries" in lib.load("mssql")
        renamed = lib.get("mssql (mcps)")
        assert renamed is not None and renamed.kind == "mcps"
        assert "Install: npx -y mssql-mcp" in lib.load("mssql (mcps)")

    def test_skill_load_missing_kind_returns_empty(self, tmp_path: Path):
        # An indexed-but-missing non-catalog skill still returns ""
        lib = _make_index(tmp_path / "skills")
        assert lib.load("nope") == ""


class TestCatalogSearchToolKindFilter:
    def test_kind_filter(self, tmp_path: Path):
        root = tmp_path / "skills"
        root.mkdir(parents=True, exist_ok=True)
        _make_index(root)
        library = SkillLibrary(root, extra_roots=[_make_catalog_root(tmp_path / "src")])
        registry = ToolRegistry(PermissionGuard(mode="auto"))
        register_skills_tools(registry, library)
        mcp = registry.call("skills.search", {"query": "filesystem", "kind": "mcps"})
        assert mcp.ok
        assert "Filesystem MCP" in mcp.output
        # keyword will not match the filesystem mcp (it is described by 'file ops')
        skill = registry.call("skills.search", {"query": "filesystem", "kind": "skill"})
        assert not skill.ok or "Filesystem MCP" not in skill.output

    def test_load_tool_kind_header(self, tmp_path: Path):
        root = tmp_path / "skills"
        root.mkdir(parents=True, exist_ok=True)
        _make_index(root)
        library = SkillLibrary(root, extra_roots=[_make_catalog_root(tmp_path / "src")])
        registry = ToolRegistry(PermissionGuard(mode="auto"))
        register_skills_tools(registry, library)
        res = registry.call("skills.load", {"name": "Filesystem MCP"})
        assert res.ok and "Mcps: Filesystem MCP" in res.output
        assert res.data["kind"] == "mcps"


# ------------------------------------------------------------------- wiring
class TestSkillsWiring:
    def test_skills_tools_registered_and_report(self, tmp_path: Path):
        config = load_config(project_root=tmp_path)
        config.safety_mode = "auto"
        config.skills = {"root": str(tmp_path / "skills")}
        _make_index(tmp_path / "skills")
        app = ArcApp(config=config, quiet=True)
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
        app = ArcApp(config=config, quiet=True)
        try:
            # No skills dir -> no skill tools, but app still healthy.
            assert "skills.search" not in app.registry.names()
            assert app.health_report()["skills"]["available"] is False
        finally:
            app.close()

    def test_sources_auto_discovered_from_sources_dir(self, tmp_path: Path):
        config = load_config(project_root=tmp_path)
        config.safety_mode = "auto"
        config.skills = {"root": str(tmp_path / "skills")}
        _make_index(tmp_path / "skills")
        # add a catalog source under data/skills/sources/<name>/
        _make_catalog_root(tmp_path / "skills" / "sources" / "awesome")
        app = ArcApp(config=config, quiet=True)
        try:
            # the catalog entries (mcps/loops/...) are indexed alongside skills
            hits = app.skills.search("filesystem")
            assert any(e.kind == "mcps" for e in hits)
            report = app.health_report()
            assert report["skills"]["available"] is True
            assert report["skills"]["kinds"].get("mcps", 0) >= 1
            assert report["skills"]["count"] >= 4
        finally:
            app.close()


class TestLegacyEnvFallback:
    def test_jarvis_skills_root_env(self, tmp_path: Path, monkeypatch):
        from arc.skills.index import skill_library_from_env
        monkeypatch.delenv("ARC_SKILLS_ROOT", raising=False)
        monkeypatch.setenv("JARVIS_SKILLS_ROOT", str(tmp_path / "skills"))
        _make_index(tmp_path / "skills")
        lib = skill_library_from_env(project_root=tmp_path)
        assert lib.root == tmp_path / "skills"

    def test_arc_skills_root_env_beats_jarvis(self, tmp_path: Path, monkeypatch):
        from arc.skills.index import skill_library_from_env
        arc_root = tmp_path / "arc-skills"
        jarvis_root = tmp_path / "jarvis-skills"
        _make_index(arc_root); _make_index(jarvis_root)
        monkeypatch.setenv("ARC_SKILLS_ROOT", str(arc_root))
        monkeypatch.setenv("JARVIS_SKILLS_ROOT", str(jarvis_root))
        lib = skill_library_from_env(project_root=tmp_path)
        assert lib.root == arc_root

    def test_jarvis_skills_root_dotenv(self, tmp_path: Path, monkeypatch):
        from arc.skills.index import skill_library_from_env
        monkeypatch.delenv("ARC_SKILLS_ROOT", raising=False)
        monkeypatch.delenv("JARVIS_SKILLS_ROOT", raising=False)
        _make_index(tmp_path / "skills")
        (tmp_path / ".env").write_text(f"JARVIS_SKILLS_ROOT={tmp_path / 'skills'}\n")
        lib = skill_library_from_env(project_root=tmp_path)
        assert lib.root == tmp_path / "skills"


# ---------------------------------------------------------------- install
class TestInstallSource:
    def test_copy_local_source(self, tmp_path: Path):
        from arc.skills.install import install_source
        src = _make_catalog_root(tmp_path / "local-src")
        sources_root = tmp_path / "skills" / "sources"
        dest = install_source(sources_root, repo_url="https://github.com/x/y.git",
                              name="awesome", source=src)
        assert (dest / "skills" / "catalog.json").is_file()
        assert dest.name == "awesome"

    def test_derives_name_from_repo_url(self, tmp_path: Path):
        from arc.skills.install import install_source
        dest = install_source(tmp_path / "sources",
                              repo_url="https://github.com/me/awesome-x.git",
                              source=_make_catalog_root(tmp_path / "s2"))
        assert dest.name == "awesome-x"
