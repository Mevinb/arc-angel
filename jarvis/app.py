"""Application facade — wires every JARVIS subsystem together.

One ``JarvisApp`` instance owns the config, database, profile, LLM router,
permission guard, tool registry, engines and orchestrator. The CLI, the
terminal UI and the scheduler all operate on it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from .config import JarvisConfig, load_config
from .core.llm import LLMRouter
from .core.logs import setup_logging
from .core.memory import LongTermMemory
from .core.orchestrator import Orchestrator
from .db.database import Database
from .internships.engine import InternshipEngine, register_internship_tools
from .profile.profile import Profile
from .safety.permissions import Approver, PermissionGuard
from .skills.index import SkillLibrary, _discover_sources
from .skills.tools import register_skills_tools
from .tools.base import ToolRegistry
from .tools.browser import register_browser_tools
from .tools.computer import register_computer_tools
from .tools.email_engine import GmailEngine, GmailTools, register_email_tools
from .tools.personal import register_personal_tools

logger = logging.getLogger("jarvis.app")


class JarvisApp:
    """The whole assistant, assembled."""

    def __init__(self, config: Optional[JarvisConfig] = None,
                 approver: Optional[Approver] = None,
                 quiet: bool = False) -> None:
        self.config = config or load_config()
        if not quiet:
            setup_logging(self.config.data_dir)
        self.db = Database(self.config.db_path)
        self.profile = Profile.load(self.config.profile_path)
        self.router = LLMRouter(self.config.llm)
        self.guard = PermissionGuard(mode=self.config.safety_mode,
                                     approver=approver)
        self.registry = ToolRegistry(self.guard)

        # Skills (SKILL.md collection) — optional; degrades to no-op if absent.
        # Extra catalog sources (awesome-ai-agent-tools & co.) are auto-discovered
        # under <skills_root>/sources and merged in.
        self.skills_root = self._resolve_skills_root()
        self.skills = None
        if self.skills_root:
            try:
                self.skills = SkillLibrary(self.skills_root,
                                           extra_roots=_discover_sources(self.skills_root))
            except Exception as exc:  # noqa: BLE001 - never break startup
                logger.warning("Failed to load skill library: %s", exc)
                self.skills = None

        # Engines
        self.gmail_engine = GmailEngine(
            credentials_path=Path(self.config.gmail.get("credentials_path",
                                                        "data/gmail-credentials.json")),
            token_path=Path(self.config.gmail.get("token_path",
                                                  "data/gmail-token.json")),
        )
        self.gmail_tools = GmailTools(self.gmail_engine, self.router,
                                      self.db, self.profile)
        self.internships = InternshipEngine(
            db=self.db, profile=self.profile, router=self.router,
            config=self.config.internships)

        # Long-term memory + agent
        self.long_term = LongTermMemory(self.db)
        self.orchestrator = Orchestrator(
            router=self.router, registry=self.registry,
            profile=self.profile, long_term=self.long_term,
            skills=self.skills,
        )

        self._register_tools()

    # ------------------------------------------------------------------ setup
    def _resolve_skills_root(self) -> Optional[Path]:
        """Find the skills root.

        An explicit ``skills.root`` in config (or ``JARVIS_SKILLS_ROOT`` in the
        env) is honored; a relative value resolves against the working
        directory. Otherwise the conventional runtime location
        ``<data_dir>/skills`` is used when present, else ``None``.
        """
        configured = (self.config.skills.get("root") or "").strip()
        if configured:
            path = Path(configured)
            if not path.is_absolute():
                path = Path.cwd() / path
            return path
        default = self.config.data_dir / "skills"
        return default if default.exists() else None

    def _register_tools(self) -> None:
        register_personal_tools(self.registry, self.long_term, self.profile)
        register_browser_tools(self.registry, router=self.router,
                               screenshot_dir=self.config.data_dir / "screenshots")
        register_computer_tools(self.registry,
                                workdir=Path.cwd())
        register_email_tools(self.registry, self.gmail_engine, self.gmail_tools)
        register_internship_tools(self.registry, self.internships)
        # Register skill tools only when the library actually has content, so an
        # uninstalled collection doesn't clutter the tool list.
        if self.skills is not None and self.skills.count:
            register_skills_tools(self.registry, self.skills)
        logger.debug("Registered tools: %s", ", ".join(self.registry.names()))

    def set_approver(self, approver: Optional[Approver]) -> None:
        """Swap the human-approval callback (UI prompt, scheduler auto-deny)."""
        self.guard.approver = approver

    # ------------------------------------------------------------------- api
    def chat(self, message: str) -> str:
        """One agent turn; returns the assistant reply."""
        return self.orchestrator.handle(message).reply

    def health_report(self) -> Dict[str, Any]:
        """Everything `jarvis doctor` shows."""
        llm = self.router.health()
        report: Dict[str, Any] = {
            "llm": llm,
            "safety_mode": self.guard.mode,
            "tools": self.registry.names(),
            "optional": {},
            "db": {"path": str(self.db.path), "jobs": self.db.job_stats()},
            "profile": {
                "path": str(self.profile.path),
                "placeholder": self.profile.is_placeholder(),
            },
        }
        for tool in self.registry.all_tools():
            if hasattr(tool, "availability"):
                report["optional"][tool.name] = tool.availability()
        report["optional"]["gmail.oauth"] = self.gmail_engine.availability_message()
        report["skills"] = self._skills_report()
        return report

    def _skills_report(self) -> Dict[str, Any]:
        if self.skills is None or not self.skills.count:
            return {"available": False, "count": 0, "content": False,
                    "root": str(self.skills_root or ""), "categories": 0,
                    "kinds": {}}
        kinds = {}
        for entry in self.skills._entries.values():
            kind = entry.kind or "skill"
            kinds[kind] = kinds.get(kind, 0) + 1
        return {
            "available": True,
            "count": self.skills.count,
            "content": self.skills.content_available,
            "root": str(self.skills_root or ""),
            "categories": len(self.skills.categories),
            "kinds": kinds,
        }

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> "JarvisApp":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def create_app(approver: Optional[Approver] = None,
               quiet: bool = False) -> JarvisApp:
    """Convenience factory (loads config from the usual locations)."""
    return JarvisApp(approver=approver, quiet=quiet)


__all__ = ["JarvisApp", "create_app"]
