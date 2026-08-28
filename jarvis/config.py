"""Configuration loading for JARVIS.

Precedence (highest wins):
  1. Environment variables (JARVIS_*)
  2. config/config.yaml (user-editable, gitignored)
  3. Built-in defaults

A tiny .env loader is included so no python-dotenv dependency is needed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a hard dependency
    yaml = None

# Project root = parent of the jarvis package
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: Dict[str, Any] = {
    "llm": {
        "base_url": "http://localhost:20128/v1",
        "api_key": "",
        "models": {"fast": "auto/fast", "reasoning": "auto", "vision": "auto"},
        "fallbacks": {
            "fast": ["auto"],
            "reasoning": ["auto/coding"],
            "vision": ["auto"],
        },
        "request": {"timeout_seconds": 90, "max_retries": 2},
    },
    "safety": {"mode": "interactive"},
    "internships": {
        "sources": ["remoteok", "arbeitnow", "hackernews"],
        "max_results_per_source": 40,
        "llm_analysis": True,
        "min_score_to_save": 20,
    },
    "automation": {
        "email_check_minutes": 30,
        "job_search_minutes": 360,
        "deadline_check_minutes": 60,
    },
    "gmail": {
        "credentials_path": "data/gmail-credentials.json",
        "token_path": "data/gmail-token.json",
    },
}

_ENV_OVERRIDES = {
    ("llm", "base_url"): "JARVIS_LLM_BASE_URL",
    ("llm", "api_key"): "JARVIS_LLM_API_KEY",
    ("safety", "mode"): "JARVIS_SAFETY_MODE",
    ("skills", "root"): "JARVIS_SKILLS_ROOT",
}
_ENV_MODEL_OVERRIDES = {
    "fast": "JARVIS_MODEL_FAST",
    "reasoning": "JARVIS_MODEL_REASONING",
    "vision": "JARVIS_MODEL_VISION",
}


def load_env_file(path: Path) -> Dict[str, str]:
    """Load KEY=VALUE pairs from a .env file. Ignores comments/blank lines."""
    env: Dict[str, str] = {}
    if not path.is_file():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env.setdefault(key.strip(), value.strip().strip("'\""))
    return env


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    models: Dict[str, str]
    fallbacks: Dict[str, List[str]]
    timeout_seconds: int = 90
    max_retries: int = 2

    def model(self, role: str) -> str:
        return self.models.get(role, self.models.get("reasoning", "auto"))

    def chain(self, role: str) -> List[str]:
        """Primary model plus fallbacks, deduplicated, order preserved."""
        seen: set = set()
        chain: List[str] = []
        for model in [self.model(role)] + list(self.fallbacks.get(role, [])):
            if model and model not in seen:
                seen.add(model)
                chain.append(model)
        return chain


@dataclass
class JarvisConfig:
    llm: LLMConfig
    safety_mode: str
    internships: Dict[str, Any]
    automation: Dict[str, Any]
    gmail: Dict[str, str]
    skills: Dict[str, str] = field(default_factory=dict)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    config_path: Path = field(default_factory=lambda: PROJECT_ROOT / "config" / "config.yaml")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jarvis.db"

    @property
    def profile_path(self) -> Path:
        return self.data_dir / "profile.yaml"


def load_config(project_root: Path | None = None,
                env_file: Path | None = None) -> JarvisConfig:
    """Build the effective configuration from defaults, YAML and env vars."""
    root = project_root or PROJECT_ROOT
    env_path = env_file or root / ".env"
    file_env = load_env_file(env_path)

    def env_value(name: str) -> str | None:
        return os.environ.get(name) or file_env.get(name)

    raw = _deep_merge(DEFAULTS, {})
    config_yaml = root / "config" / "config.yaml"
    if yaml is not None and config_yaml.is_file():
        loaded = yaml.safe_load(config_yaml.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{config_yaml} must contain a YAML mapping")
        raw = _deep_merge(raw, loaded)

    # Environment overrides
    for (section, key), env_name in _ENV_OVERRIDES.items():
        value = env_value(env_name)
        if value:
            raw.setdefault(section, {})[key] = value
    for role, env_name in _ENV_MODEL_OVERRIDES.items():
        value = env_value(env_name)
        if value:
            raw["llm"].setdefault("models", {})[role] = value

    llm_raw = raw["llm"]
    request = llm_raw.get("request", {})
    llm = LLMConfig(
        base_url=str(llm_raw.get("base_url", DEFAULTS["llm"]["base_url"])),
        api_key=str(llm_raw.get("api_key", "") or ""),
        models={k: str(v) for k, v in llm_raw.get("models", {}).items()},
        fallbacks={k: [str(m) for m in v] for k, v in llm_raw.get("fallbacks", {}).items()},
        timeout_seconds=int(request.get("timeout_seconds", 90)),
        max_retries=int(request.get("max_retries", 2)),
    )

    data_dir = Path(str(os.environ.get("JARVIS_DATA_DIR") or (root / "data")))
    data_dir.mkdir(parents=True, exist_ok=True)

    return JarvisConfig(
        llm=llm,
        safety_mode=str(raw.get("safety", {}).get("mode", "interactive")).lower(),
        internships=dict(raw.get("internships", {})),
        automation=dict(raw.get("automation", {})),
        gmail={k: str(v) for k, v in raw.get("gmail", {}).items()},
        skills={k: str(v) for k, v in raw.get("skills", {}).items()},
        data_dir=data_dir,
        config_path=config_yaml,
    )
