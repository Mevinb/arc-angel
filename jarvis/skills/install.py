"""Installation helpers for the SKILL.md collection.

``install_skills`` clones (or copies) the Luo-Kai aggregation repo into the
JARVIS ``data/skills/`` root, keeping the compact ``skills-index.json`` and a
``content`` pointer to the ``ai-agent-skills`` directory. A lightweight index
re-builder is included for use when a repo ships no prebuilt index.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .index import CONTENT_DIRNAME, INDEX_FILENAME

REPO_URL = "https://github.com/luokai0/ai-agent-skills-by-luo-kai.git"
CONTENT_SOURCE_DIR = "ai-agent-skills"


def install_skills(dest: Path, repo_url: str = REPO_URL,
                   source: Optional[Path] = None) -> Path:
    """Install the collection under ``dest``.

    ``dest`` becomes a valid :class:`SkillLibrary` root holding
    ``skills-index.json`` and a ``content`` subdir (or symlink). When ``source``
    is given the files are copied from there instead of cloning the network.

    Returns the root path.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if source is not None:
        _copy_collection(Path(source), dest)
    else:
        _clone_collection(dest, repo_url)
    _finalize(dest)
    return dest


def _clone_collection(dest: Path, repo_url: str) -> None:
    tmp = dest / ".tmp-src"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", repo_url, str(tmp)],
            check=True, capture_output=True, text=True, timeout=1200,
        )
        _copy_tree(tmp, dest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _copy_collection(source: Path, dest: Path) -> None:
    _copy_tree(source, dest)


def _copy_tree(source: Path, dest: Path) -> None:
    """Copy the interesting parts of a clone into the destination root."""
    for name in (INDEX_FILENAME,):
        src = source / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    content_src = source / CONTENT_SOURCE_DIR
    content_dest = dest / CONTENT_DIRNAME
    if content_src.is_dir():
        # Remove any existing content destination WITHOUT following symlinks —
        # rmtree on a symlink would delete its (possibly large) target.
        if content_dest.is_symlink() or content_dest.is_file():
            content_dest.unlink(missing_ok=True)
        elif content_dest.is_dir():
            shutil.rmtree(content_dest, ignore_errors=True)
        # symlink when possible to avoid duplicating large content trees
        try:
            content_dest.symlink_to(content_src, target_is_directory=True)
        except OSError:
            shutil.copytree(content_src, content_dest, dirs_exist_ok=True)


def _finalize(dest: Path) -> None:
    """Ensure an index exists; rebuild one via frontmatter if not shipped."""
    index = dest / INDEX_FILENAME
    content = dest / CONTENT_DIRNAME
    if not index.is_file() and content.exists():
        write_index(dest, content, index)


def write_index(root: Path, content: Path, index: Path) -> None:
    """Rebuild a ``skills-index.json`` by scanning ``**/SKILL.md`` frontmatter."""
    from .index import _read_frontmatter

    skills: List[Dict[str, Any]] = []
    if content.is_dir():
        for skill_md in sorted(content.rglob("SKILL.md")):
            meta = _read_frontmatter(skill_md) or {}
            name = str(meta.get("name") or skill_md.parent.name)
            skills.append({
                "n": name,
                "d": str(meta.get("description", "")),
                "p": str(skill_md.relative_to(root)),
            })
    root.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps({"total": len(skills), "cats": {},
                                 "skills": skills}, ensure_ascii=False),
                     encoding="utf-8")


__all__ = ["install_skills", "write_index", "REPO_URL",
           "CONTENT_SOURCE_DIR", "CONTENT_DIRNAME", "INDEX_FILENAME"]
