"""Discovery and loading of ``SKILL.md`` skills.

A :class:`SkillLibrary` points at a skills root and exposes a lightweight,
searchable index. It prefers the curated ``skills-index.json`` shipped with the
Luo-Kai collection (paths there use URL-encoded segments and an
``ai-agent-skills/`` prefix we strip when resolving against the content root).
If no index file exists, it falls back to scanning every ``**/SKILL.md`` and
parsing its YAML frontmatter.

The full instructions of a skill are *not* loaded eagerly — :meth:`load`
returns the file text only when the model asks for a specific skill, keeping
the hot index compact.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

logger = logging.getLogger("jarvis.skills")

#: Name of the prebuilt searchable index inside a skills root.
INDEX_FILENAME = "skills-index.json"
#: Directory (relative to the root) holding the content root.
CONTENT_DIRNAME = "content"
#: Prefix on index file paths pointing at the upstream repo layout. Stripped
#: when resolving against the *content* root, which is the ai-agent-skills dir.
UPSTREAM_PREFIX = "ai-agent-skills/"

#: Subdirectory (relative to a *primary* skills root) holding additional
#: catalog-style sources such as awesome-ai-agent-tools.
SOURCES_DIRNAME = "sources"

#: Component kinds understood from catalog-style sources. The empty string is
#: the implicit "skill" kind used by the primary (SKILL.md) collection.
KINDS = ("skill", "mcp", "loop", "subagent", "hook",
         "plugin", "prompt", "tool")

#: Catalog-style roots (e.g. awesome-ai-agent-tools) expose one ``catalog.json``
#: per directory/category. Map the directory name to the JSON key holding the
#: actual entry list (some catalogs also have a ``categories`` list we must not
#: mistake for entries).
CATALOG_LIST_KEY = {
    "skills": "skills",
    "mcps": "servers",
    "loops": "loops",
    "subagents": "subagents",
    "hooks": "hooks",
    "plugins": "plugins",
    "prompts": "prompts",
    "tools": "tools",
}


@dataclass
class SkillEntry:
    """One component's catalog metadata (no body text).

    Beyond the primary SKILL.md collection, an entry may come from a catalog
    source (awesome-ai-agent-tools) and describe an MCP server, loop, subagent,
    hook, plugin, prompt, or CLI tool rather than a SKILL.md. ``kind`` records
    that; ``source``/``source_url``/``install`` record provenance + how to add it.
    """

    name: str
    description: str
    category: str = ""
    path: str = ""  # filesystem path to SKILL.md (absolute when content available)
    kind: str = ""  # "" (skill) | one of KINDS
    source: str = ""      # provenance repo / author (e.g. "obra/superpowers")
    source_url: str = ""  # human-readable URL if any
    install: str = ""     # exact command to add/install this component

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "path": self.path,
            "kind": self.kind,
            "source": self.source,
            "source_url": self.source_url,
            "install": self.install,
        }

    def to_schema(self) -> Dict[str, Any]:
        """Compact form for the LLM/system prompt."""
        entry = {"name": self.name, "category": self.category,
                 "description": self.description}
        if self.kind:
            entry["kind"] = self.kind
        if self.install:
            entry["install"] = self.install
        return entry


class SkillLibrary:
    """Searchable catalog of ``SKILL.md`` skills with on-demand loading."""

    def __init__(self, root: Path | str,
                 extra_roots: Optional[List[Path | str]] = None) -> None:
        self.root = Path(root)
        self.extra_roots = [Path(r) for r in (extra_roots or [])]
        self._entries: Dict[str, SkillEntry] = {}
        self.categories: Dict[str, str] = {}
        self._content_root: Optional[Path] = None
        self._load()
        for extra in self.extra_roots:
            self._load_catalog_root(extra)

    # ------------------------------------------------------------ discovery
    def _load(self) -> None:
        index = self.root / INDEX_FILENAME
        self._content_root = self.root / CONTENT_DIRNAME
        if index.is_file():
            try:
                data = json.loads(index.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover
                logger.warning("Skills index unreadable, scanning instead: %s", exc)
                data = {}
            self.categories = {str(k): str(v) for k, v in
                               dict(data.get("cats") or {}).items()}
            self.categories[""] = "All"
            for raw in data.get("skills") or []:
                name = str(raw.get("n") or "")
                if not name:
                    continue
                category = self.categories.get(str(raw.get("c") or ""), "")
                self._entries[name] = SkillEntry(
                    name=name,
                    description=str(raw.get("d") or ""),
                    category=category,
                    path=self._resolve_index_path(str(raw.get("p") or "")),
                )
        else:
            self._scan_frontmatter()

    def _load_catalog_root(self, extra: Path) -> None:
        """Merge a catalog-style source (e.g. awesome-ai-agent-tools).

        Such a root ships one ``catalog.json`` per component category
        (``skills/``, ``mcps/``, ``loops/``, ...). Each entry becomes a searchable
        :class:`SkillEntry` with a ``kind`` tag, provenance and an install
        command. Any real ``**/SKILL.md`` files bundled in the root are also
        indexed so their full content can be loaded.
        """
        if not extra.is_dir():
            return
        for dirname, key in CATALOG_LIST_KEY.items():
            catalog = extra / dirname / "catalog.json"
            entries = _load_catalog_entries(catalog, key, kind=dirname)
            for entry in entries:
                self._add_entry(entry)
        # Also index any bundled SKILL.md content (the collection's own skills).
        for skill_md in sorted(extra.rglob("SKILL.md")):
            meta = _read_frontmatter(skill_md) or {}
            name = str(meta.get("name") or skill_md.parent.name)
            if name in self._entries:
                continue
            category = str(meta.get("category", "")) or \
                self._dir_category(skill_md)
            self._add_entry(SkillEntry(
                name=name,
                description=str(meta.get("description", "")),
                category=category,
                path=str(skill_md),
                kind="skill",
                source=str(extra.resolve()),
            ))

    def _add_entry(self, entry: SkillEntry) -> None:
        """Insert an entry, disambiguating duplicate names across sources.

        On a clash the entry *itself* is renamed (not just its key) so the name
        shown by search/list is always exactly resolvable via :meth:`get`.
        """
        name = entry.name or entry.category
        if not name:
            return
        if name in self._entries:
            # Name clash with the primary collection or another source: keep
            # both, qualifying this one with its kind.
            suffix = f" ({entry.kind})" if entry.kind else " (alt)"
            entry.name = name + suffix
            name = entry.name
        if self.categories.get(entry.category) is None:
            self.categories[entry.category] = entry.category
        self.categories[""] = "All"
        self._entries[name] = entry

    def _resolve_index_path(self, index_path: str) -> str:
        """Turn an index file path into an absolute path on disk, if possible."""
        rel = unquote(index_path)
        if rel.startswith(UPSTREAM_PREFIX):
            rel = rel[len(UPSTREAM_PREFIX):]
        if self._content_root is not None:
            candidate = self._content_root / rel
            if candidate.is_file():
                return str(candidate)
        return ""

    def _scan_frontmatter(self) -> None:  # pragma: no cover - fallback path
        if self._content_root is None or not self._content_root.is_dir():
            self.categories = {"": "All"}
            return
        for skill_md in sorted(self._content_root.rglob("SKILL.md")):
            meta = _read_frontmatter(skill_md) or {}
            name = str(meta.get("name") or "")
            if not name:
                name = skill_md.parent.name
            category = str(meta.get("category", "")) or \
                self._dir_category(skill_md)
            if category and category not in self.categories.values():
                self.categories[category] = category
            self._entries[name] = SkillEntry(
                name=name,
                description=str(meta.get("description", "")),
                category=category,
                path=str(skill_md),
            )

    @staticmethod
    def _dir_category(skill_md: Path) -> str:
        parts = skill_md.parts
        for i, part in enumerate(parts):
            if part and part[0:2].isdigit() and "-" in part:
                return " ".join(part.split("-", 1)[1].split()).title()
        return ""

    # ------------------------------------------------------------------ api
    @property
    def count(self) -> int:
        return len(self._entries)

    @property
    def content_available(self) -> bool:
        """Whether the actual SKILL.md files are reachable for loading."""
        if not self._content_root:
            return False
        if self._content_root.is_dir():
            return True
        # The content root may itself be a SKILL.md-bearing dir in some setups.
        return self._content_root.is_file()

    @property
    def content_root(self) -> Optional[Path]:
        return self._content_root if (self._content_root and
                                      self._content_root.exists()) else None

    def search(self, query: str, limit: int = 10) -> List[SkillEntry]:
        """Return skills whose name/description/category match the query,
        best matches first. Empty query returns the first ``limit`` entries."""
        query = (query or "").strip().lower()
        if not query:
            return sorted(self._entries.values(),
                          key=lambda e: e.name)[:limit]
        tokens = [t for t in query.split() if t]
        scored: List[tuple] = []
        for entry in self._entries.values():
            name = entry.name.lower()
            desc = (entry.description or "").lower()
            cat = (entry.category or "").lower()
            score = 0
            for token in tokens:
                if token in name:
                    score += 4
                if token in cat:
                    score += 2
                if token in desc:
                    score += 1
            if score:
                scored.append((score, entry))
        scored.sort(key=lambda t: (-t[0], t[1].name))
        return [entry for _, entry in scored[:limit]]

    def list_by_category(self, category: str = "") -> List[SkillEntry]:
        if not category:
            return sorted(self._entries.values(), key=lambda e: e.name)
        cat = category.strip().lower()
        return sorted((e for e in self._entries.values()
                       if (e.category or "").lower() == cat),
                      key=lambda e: e.name)

    def categories_with_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self._entries.values():
            cat = entry.category or "Uncategorized"
            counts[cat] = counts.get(cat, 0) + 1
        return dict(sorted(counts.items()))

    def get(self, name: str) -> Optional[SkillEntry]:
        entry = self._entries.get(name)
        if entry:
            return entry
        # lenient: case-insensitive / partial unique match
        lower = name.strip().lower()
        matched = [e for e in self._entries.values() if e.name.lower() == lower]
        if len(matched) == 1:
            return matched[0]
        return None

    def load(self, name: str) -> str:
        """Return the full SKILL.md text for a skill, or '' if unavailable.

        For catalog-style entries (kind != "") that have no bundled SKILL.md
        file, a synthesized, actionable block is returned — description,
        provenance and the exact command to install/add the component — so the
        agent can act on it rather than dead-end.
        """
        entry = self.get(name)
        if entry is None:
            return ""
        if entry.path:
            try:
                path = Path(entry.path)
                if path.is_file():
                    return path.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover
                logger.debug("Failed to read skill file %s", entry.path, exc_info=True)
        if entry.kind:
            # Catalog / remote component without bundled content: give the agent
            # everything it needs to use or install it.
            lines = [f"# {entry.name}", f"Kind: {entry.kind}",
                     f"Category: {entry.category or 'Uncategorized'}",
                     f"Description: {entry.description}"]
            if entry.source_url:
                lines.append(f"Source: {entry.source_url}")
            elif entry.source:
                lines.append(f"Source: {entry.source}")
            if entry.install:
                lines.append(f"Install: {entry.install}")
            return "\n".join(lines)
        return ""


def _read_frontmatter(path: Path) -> Dict[str, Any]:
    """Minimal YAML-frontmatter parser (no PyYAML dependency in this path)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:  # pragma: no cover
        return {}
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    meta: Dict[str, Any] = {}
    for line in text[3:end].splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value.startswith("[") and value.endswith("]"):
            value = ",".join(v.strip().strip("'\"") for v in value[1:-1].split(","))
        else:
            value = value.strip("'\"")
        meta[key] = value
    return meta


def _load_catalog_entries(catalog: Path, list_key: str,
                          kind: str) -> List[SkillEntry]:
    """Parse one category's ``catalog.json`` into unified :class:`SkillEntry`\\ s.

    Entry shapes vary by category (skills/servers/loops/subagents/hooks/
    plugins/prompts/tools), so field access is defensive and tolerates missing
    keys. Returns [] when the catalog is absent or unreadable.
    """
    try:
        data = json.loads(catalog.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    raw_list = data.get(list_key)
    if not isinstance(raw_list, list):
        # Some catalogs nest entries under a wrapper like {"items": [...]}.
        for candidate in ("items", "entries", "components"):
            if isinstance(data.get(candidate), list):
                raw_list = data[candidate]
                break
    if not isinstance(raw_list, list):
        return []

    from_field = {
        "skills": "name", "mcps": "name", "loops": "title",
        "subagents": "name", "hooks": "name", "plugins": "name",
        "prompts": "name", "tools": "name",
    }
    install_field = {
        "skills": "install", "mcps": "install", "plugins": "installCommand",
        "tools": "install",
    }
    # source_url falls back over these keys in order
    url_keys = {
        "skills": ["source_url", "github"],
        "mcps": ["github"],
        "loops": ["source", "sourceRepo"],
        "plugins": ["websiteUrl"],
        "prompts": ["source"],
        "tools": ["url", "homepage"],
        "subagents": ["tags", "github"],
        "hooks": ["source", "github"],
    }
    source_keys = {
        "skills": ["source"],
        "mcps": ["github", "source"],
        "loops": ["sourceRepo", "source"],
        "hooks": ["source"],
        "plugins": ["source", "platform"],
        "prompts": ["source"],
        "tools": ["url", "source"],
    }

    name_key = from_field.get(kind, "name")
    out: List[SkillEntry] = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get(name_key) or raw.get("name") or raw.get("title") or "")
        if not name:
            continue
        category = _catalog_category(kind, raw)
        install = ""
        if kind in install_field:
            install = str(raw.get(install_field[kind]) or "").strip()
        source_url = _first_str(raw, url_keys.get(kind, []) + ["url", "source_url"])
        # keep only plausible URLs; otherwise fall back to provenance string
        if source_url and not str(source_url).startswith(("http", "github.com")):
            source_url = "" if kind == "skills" else source_url
        source = _first_str(raw, source_keys.get(kind, []))
        out.append(SkillEntry(
            name=name,
            description=str(raw.get("description") or ""),
            category=category,
            kind=kind,
            source=source or "",
            source_url=source_url or "",
            install=install,
        ))
    return out


def _catalog_category(kind: str, raw: Dict[str, Any]) -> str:
    """Category label for a catalog entry, preferring a category-specific id."""
    cat = str(raw.get("category") or "").strip()
    if cat:
        return cat
    # loops store categories as numeric-ish ids in `category`; use tags fallback
    tags = raw.get("tags")
    if isinstance(tags, list) and tags:
        return str(tags[0])
    return "" if kind in ("skills",) else kind.title()


def _first_str(raw: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
        elif isinstance(value, list) and value:
            return str(value[0])
    return ""


def skill_library_from_env(project_root: Path | str | None = None,
                           **env: Any) -> SkillLibrary:
    """Build the library from the conventional root path.

    Respects an explicit ``skills_root`` argument or ``JARVIS_SKILLS_ROOT`` in
    the environment/env-file; otherwise defaults to ``<root>/data/skills``.
    """
    import os
    root = Path(project_root or Path(__file__).resolve().parent.parent.parent)
    skills_root = env.get("skills_root") or os.environ.get("JARVIS_SKILLS_ROOT")
    if not skills_root:
        # honour the same .env file the config loader uses
        env_file = root / ".env"
        if env_file.is_file():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("JARVIS_SKILLS_ROOT"):
                    skills_root = line.partition("=")[2].strip().strip("'\"")
    if skills_root:
        return SkillLibrary(root=Path(skills_root),
                             extra_roots=_discover_sources(Path(skills_root)))
    return SkillLibrary(root=root / "data" / "skills",
                        extra_roots=_discover_sources(root / "data" / "skills"))


def _discover_sources(skills_root: Path) -> List[Path]:
    """Each subdirectory of ``<root>/sources`` is an extra catalog source."""
    sources = skills_root / SOURCES_DIRNAME
    if not sources.is_dir():
        return []
    return [p for p in sorted(sources.iterdir()) if p.is_dir()]


__all__ = ["SkillLibrary", "SkillEntry", "skill_library_from_env",
           "CATALOG_LIST_KEY", "SOURCES_DIRNAME", "KINDS"]
