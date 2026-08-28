"""Phase 7 — Personal profile & memory.

The profile lives in ``data/profile.yaml`` (created from an embedded example on
first run). It stores the resume, education, skills, projects, experience,
preferences and achievements so JARVIS can answer questions like:

  "What projects have I done with Python?"
  "Does this internship match my skills?"
  "Answer this application question using my experience."
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("jarvis.profile")

PROFILE_EXAMPLE: Dict[str, Any] = {
    "name": "Your Name",
    "email": "you@example.com",
    "phone": "",
    "location": "Your City",
    "github": "https://github.com/yourname",
    "linkedin": "",
    "education": [
        {"school": "Your University", "degree": "B.Tech Computer Science",
         "start": "2024", "end": "2028", "highlights": []},
    ],
    "skills": {
        "languages": ["Python", "JavaScript", "C++"],
        "frameworks": ["FastAPI", "React", "PyTorch"],
        "tools": ["Git", "Docker", "Linux", "SQLite"],
    },
    "projects": [
        {"name": "JARVIS", "description": "Personal AI assistant orchestrating LLM "
         "routing, browser automation and an internship search engine",
         "tech": ["Python", "OpenAI API", "SQLite"], "url": ""},
    ],
    "experience": [
        {"company": "", "role": "", "period": "", "highlights": []},
    ],
    "achievements": [],
    "preferred_roles": ["Software Engineer Intern", "Backend Intern", "AI/ML Intern"],
    "preferred_locations": ["Remote", "Your City"],
    "resume_path": "data/resume.pdf",
}


class Profile:
    """Load / save / summarize the personal profile."""

    def __init__(self, path: Path, data: Dict[str, Any]) -> None:
        self.path = path
        self.data = data

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, path: Path, create_if_missing: bool = True) -> "Profile":
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Profile {path} must be a YAML mapping")
            # Fill in any missing top-level keys from the example
            for key, value in PROFILE_EXAMPLE.items():
                data.setdefault(key, value if not isinstance(value, (dict, list)) else
                                ({} if isinstance(value, dict) else []))
            return cls(path, data)
        if create_if_missing:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                "# JARVIS personal profile — edit me!\n" +
                yaml.safe_dump(PROFILE_EXAMPLE, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            logger.info("Created example profile at %s", path)
            return cls(path, dict(PROFILE_EXAMPLE))
        raise FileNotFoundError(f"Profile not found: {path}")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    # ------------------------------------------------------------- accessors
    @property
    def name(self) -> str:
        return str(self.data.get("name", ""))

    def skills_list(self) -> List[str]:
        """Flat list of every skill mentioned in the profile."""
        skills: List[str] = []
        for group in (self.data.get("skills") or {}).values():
            if isinstance(group, list):
                skills.extend(str(s) for s in group)
        # Include technologies from projects and experience
        for project in self.data.get("projects") or []:
            skills.extend(str(t) for t in project.get("tech", []))
        # De-duplicate, case-insensitive, preserving order
        seen: set = set()
        unique: List[str] = []
        for skill in skills:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                unique.append(skill)
        return unique

    def search_projects(self, query: str) -> List[Dict[str, Any]]:
        """Projects matching a free-text query (name, description or tech)."""
        needle = query.lower()
        matches = []
        for project in self.data.get("projects") or []:
            haystack = " ".join([
                str(project.get("name", "")),
                str(project.get("description", "")),
                " ".join(str(t) for t in project.get("tech", [])),
            ]).lower()
            if needle in haystack:
                matches.append(project)
        return matches

    # ------------------------------------------------------------ summarizing
    def summarize(self) -> str:
        """Compact profile block injected into the LLM system prompt."""
        lines: List[str] = [f"Name: {self.name}"]
        for key in ("email", "location", "github", "linkedin"):
            if self.data.get(key):
                lines.append(f"{key.title()}: {self.data[key]}")
        skills = self.skills_list()
        if skills:
            lines.append("Skills: " + ", ".join(skills))
        roles = self.data.get("preferred_roles") or []
        if roles:
            lines.append("Preferred roles: " + ", ".join(str(r) for r in roles))
        locations = self.data.get("preferred_locations") or []
        if locations:
            lines.append("Preferred locations: " + ", ".join(str(l) for l in locations))
        education = self.data.get("education") or []
        for edu in education[:3]:
            if edu.get("school"):
                lines.append(f"Education: {edu.get('degree', '')} at {edu['school']}"
                             f" ({edu.get('start', '')}-{edu.get('end', '')})")
        for project in (self.data.get("projects") or [])[:6]:
            if project.get("name"):
                tech = ", ".join(str(t) for t in project.get("tech", []))
                lines.append(f"Project: {project['name']} — {project.get('description', '')}"
                             f" [{tech}]")
        return "\n".join(lines)

    def to_text(self) -> str:
        """Full YAML dump of the profile (for deep questions)."""
        return yaml.safe_dump(self.data, sort_keys=False, allow_unicode=True)

    def is_placeholder(self) -> bool:
        """True if the user hasn't edited the example profile yet."""
        return self.name in ("", "Your Name")

    # Allow dict-like access for convenience
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.data.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value
