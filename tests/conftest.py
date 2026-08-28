"""Shared fixtures: temp dirs, fake LLM clients, fake profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from arc.config import LLMConfig, load_config
from arc.core.llm import LLMRouter
from arc.db.database import Database
from arc.profile.profile import Profile


# --------------------------------------------------------------- fake LLM API
class FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: Dict[str, Any]) -> None:
        self.id = call_id
        self.function = FakeFunction(name, __import__("json").dumps(arguments))


class FakeMessage:
    def __init__(self, content: str = "",
                 tool_calls: Optional[List[FakeToolCall]] = None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, message: FakeMessage, finish_reason: str = "stop") -> None:
        self.message = message
        self.finish_reason = finish_reason


class FakeResponse:
    def __init__(self, message: FakeMessage, finish_reason: str = "stop") -> None:
        self.choices = [FakeChoice(message, finish_reason)]
        self.usage = None


class ScriptedCompletions:
    """Stand-in for client.chat.completions with canned responses."""

    def __init__(self, responses: List[FakeResponse]) -> None:
        self.responses = list(responses)
        self.requests: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeResponse:
        self.requests.append(kwargs)
        if not self.responses:
            raise RuntimeError("no scripted responses left")
        return self.responses.pop(0)


class ScriptedClient:
    """Stand-in for the OpenAI client."""

    def __init__(self, responses: List[FakeResponse]) -> None:
        self.completions = ScriptedCompletions(responses)
        self.chat = self


class ExplodingClient:
    """Client that always fails — for fallback/error-path tests."""

    class _Boom:
        @staticmethod
        def create(**_: Any) -> Any:
            raise ConnectionError("gateway unreachable")

    chat = _Boom()


def make_router(responses: Optional[List[FakeResponse]] = None,
                client: Optional[Any] = None) -> LLMRouter:
    config = LLMConfig(
        base_url="http://fake.local/v1", api_key="test",
        models={"fast": "fake-fast", "reasoning": "fake-reasoning",
                "vision": "fake-vision"},
        fallbacks={"fast": [], "reasoning": [], "vision": []},
    )
    if client is None:
        client = ScriptedClient(responses or [])
    return LLMRouter(config, client_factory=lambda _cfg: client)


# -------------------------------------------------------------------- fixtures
@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard every test against env leakage from third-party imports.

    Optional engines (e.g. browser-use, open-interpreter) call ``load_dotenv()``
    at import time, which reads the project's ``.env`` and injects ARC LLM
    vars into ``os.environ``. That would otherwise leak into config-default
    assertions, so scrub the LLM-related vars before each test.
    """
    for name in ("ARC_LLM_BASE_URL", "ARC_LLM_API_KEY",
                 "ARC_MODEL_FAST", "ARC_MODEL_REASONING",
                 "ARC_MODEL_VISION", "ARC_DATA_DIR",
                 "ARC_SAFETY_MODE", "ARC_GMAIL_CREDENTIALS",
                 "ARC_GMAIL_TOKEN",
                 # Legacy pre-rename names — still scrubbed so live .env vars
                 # don't leak via third-party load_dotenv() calls.
                 "JARVIS_LLM_BASE_URL", "JARVIS_LLM_API_KEY",
                 "JARVIS_MODEL_FAST", "JARVIS_MODEL_REASONING",
                 "JARVIS_MODEL_VISION", "JARVIS_DATA_DIR",
                 "JARVIS_SAFETY_MODE", "JARVIS_GMAIL_CREDENTIALS",
                 "JARVIS_GMAIL_TOKEN", "JARVIS_SKILLS_ROOT",
                 "ARC_SKILLS_ROOT"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "test.db")
    yield database
    database.close()


@pytest.fixture()
def profile(tmp_path: Path) -> Profile:
    return Profile(tmp_path / "profile.yaml", data={
        "name": "Test User",
        "email": "test@example.com",
        "location": "Berlin",
        "skills": {
            "languages": ["Python", "C++"],
            "frameworks": ["FastAPI"],
            "tools": ["Git", "Docker"],
        },
        "projects": [
            {"name": "ARC", "description": "AI assistant", "tech": ["Python"],
             "url": ""},
        ],
        "preferred_roles": ["Software Engineer Intern", "Backend Intern"],
        "preferred_locations": ["Remote", "Berlin"],
    })


@pytest.fixture()
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for _name in ("ARC_DATA_DIR", "JARVIS_DATA_DIR",
                  "ARC_LLM_BASE_URL", "JARVIS_LLM_BASE_URL",
                  "ARC_LLM_API_KEY", "JARVIS_LLM_API_KEY",
                  "ARC_SKILLS_ROOT", "JARVIS_SKILLS_ROOT"):
        monkeypatch.delenv(_name, raising=False)
    return load_config(project_root=tmp_path)
