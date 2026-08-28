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


@dataclass
class SkillEntry:
    """One skill's catalog metadata (no body text)."""

    name: str
    description: str
    category: str = ""
    path: str = ""  # filesystem path to SKILL.md (absolute when content available)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "path": self.path,
        }

    def to_schema(self) -> Dict[str, Any]:
        """Compact form for the LLM/system prompt."""
        return {"name": self.name, "category": self.category,
                "description": self.description}


class SkillLibrary:
    """Searchable catalog of ``SKILL.md`` skills with on-demand loading."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._entries: Dict[str, SkillEntry] = {}
        self.categories: Dict[str, str] = {}
        self._content_root: Optional[Path] = None
        self._load()

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
        """Return the full SKILL.md text for a skill, or '' if unavailable."""
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
        return SkillLibrary(root=Path(skills_root))
    return SkillLibrary(root=root / "data" / "skills")


__all__ = ["SkillLibrary", "SkillEntry", "skill_library_from_env"]
