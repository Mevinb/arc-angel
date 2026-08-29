"""Configuration loading for ARC.

Precedence (highest wins):
  1. Environment variables (ARC_*)
  2. config/config.yaml (user-editable, gitignored)
  3. Built-in defaults

Legacy JARVIS_* environment variables are honoured as fallbacks for
pre-rename setups (ARC_* always wins when both are set).

A tiny .env loader is included so no python-dotenv dependency is needed.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is a hard dependency
    yaml = None

# Project root = parent of the arc package
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
    "voice": {
        "enabled": False,
        "stt": "faster-whisper",
        "stt_model": "small",
        "stt_language": "en",
        "tts": "auto",
        "tts_voice": "af_heart",
        "device": "auto",
        "always_listening": True,
        "full_duplex": True,
        "allow_voice_approval": True,
        "vad_aggressiveness": 3,
        "confidence_threshold": 0.6,
        "wake_word": "hey arc",
        "wake_word_enabled": True,
    },
    "browser": {
        "user_data_dir": "~/.config/google-chrome",
        "cdp_port": 9222,
        "ozone": "auto",
    },
}

_ENV_OVERRIDES = {
    ("llm", "base_url"): "ARC_LLM_BASE_URL",
    ("llm", "api_key"): "ARC_LLM_API_KEY",
    ("safety", "mode"): "ARC_SAFETY_MODE",
    ("skills", "root"): "ARC_SKILLS_ROOT",
    ("voice", "stt"): "ARC_VOICE_STT",
    ("voice", "stt_model"): "ARC_VOICE_STT_MODEL",
    ("voice", "tts"): "ARC_VOICE_TTS",
    ("voice", "device"): "ARC_VOICE_DEVICE",
    ("browser", "user_data_dir"): "ARC_CHROME_USER_DATA_DIR",
    ("browser", "cdp_port"): "ARC_CHROME_CDP_PORT",
    ("browser", "ozone"): "ARC_CHROME_OZONE",
}
_ENV_MODEL_OVERRIDES = {
    "fast": "ARC_MODEL_FAST",
    "reasoning": "ARC_MODEL_REASONING",
    "vision": "ARC_MODEL_VISION",
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


def _migrate_legacy_db(data_dir: Path) -> None:
    """One-time migration: copy data/jarvis.db → data/arc.db (with WAL sidecars).

    A pre-rename installation keeps its data; the legacy file is left in
    place as a backup.  No-op when ``arc.db`` already exists or no legacy
    file is present.
    """
    legacy = data_dir / "jarvis.db"
    current = data_dir / "arc.db"
    if current.exists() or not legacy.is_file():
        return
    try:
        shutil.copy2(legacy, current)
    except OSError:
        return
    for suffix in ("-wal", "-shm"):
        side = data_dir / f"jarvis.db{suffix}"
        if side.is_file():
            try:
                shutil.copy2(side, data_dir / f"arc.db{suffix}")
            except OSError:
                pass


@dataclass
class ArcConfig:
    llm: LLMConfig
    safety_mode: str
    internships: Dict[str, Any]
    automation: Dict[str, Any]
    gmail: Dict[str, str]
    skills: Dict[str, str] = field(default_factory=dict)
    voice: Dict[str, Any] = field(default_factory=dict)
    browser: Dict[str, Any] = field(default_factory=dict)
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    config_path: Path = field(default_factory=lambda: PROJECT_ROOT / "config" / "config.yaml")

    @property
    def db_path(self) -> Path:
        _migrate_legacy_db(self.data_dir)
        return self.data_dir / "arc.db"

    @property
    def profile_path(self) -> Path:
        return self.data_dir / "profile.yaml"


def load_config(project_root: Path | None = None,
                env_file: Path | None = None) -> ArcConfig:
    """Build the effective configuration from defaults, YAML and env vars."""
    root = project_root or PROJECT_ROOT
    env_path = env_file or root / ".env"
    file_env = load_env_file(env_path)

    def env_value(name: str) -> str | None:
        value = os.environ.get(name) or file_env.get(name)
        if value:
            return value
        if name.startswith("ARC_"):
            legacy = "JARVIS_" + name[len("ARC_") :]
            return os.environ.get(legacy) or file_env.get(legacy)
        return None

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

    data_dir = Path(str(
        os.environ.get("ARC_DATA_DIR")
        or os.environ.get("JARVIS_DATA_DIR")
        or (root / "data")
    ))
    data_dir.mkdir(parents=True, exist_ok=True)

    # Voice config with typed defaults
    voice_raw = raw.get("voice", {})
    voice = {
        "enabled": bool(voice_raw.get("enabled", False)),
        "stt": str(voice_raw.get("stt", "faster-whisper")),
        "stt_model": str(voice_raw.get("stt_model", "small")),
        "stt_language": str(voice_raw.get("stt_language", "en")),
        "tts": str(voice_raw.get("tts", "auto")),
        "tts_voice": str(voice_raw.get("tts_voice", "af_heart")),
        "device": str(voice_raw.get("device", "auto")),
        "always_listening": bool(voice_raw.get("always_listening", True)),
        "full_duplex": bool(voice_raw.get("full_duplex", True)),
        "allow_voice_approval": bool(voice_raw.get("allow_voice_approval", True)),
        "vad_aggressiveness": int(voice_raw.get("vad_aggressiveness", 2)),
        "confidence_threshold": float(voice_raw.get("confidence_threshold", 0.6)),
    }
    # Browser — your live Chrome profile
    browser_raw = raw.get("browser", {})
    browser = {
        "user_data_dir": str(browser_raw.get("user_data_dir", "~/.config/google-chrome")),
        "cdp_port": int(browser_raw.get("cdp_port", 9222)),
        "ozone": str(browser_raw.get("ozone", "auto")),
    }
    # cdp_port may be str from env, coerce
    try:
        browser["cdp_port"] = int(browser["cdp_port"])
    except Exception:
        browser["cdp_port"] = 9222
    # Env overrides for voice (catch-all for ARC_VOICE_*)
    for key in list(voice.keys()):
        env_name = f"ARC_VOICE_{key.upper()}"
        val = env_value(env_name)
        if val is not None:
            # coerce to same type as default
            default = voice[key]
            if isinstance(default, bool):
                voice[key] = val.lower() in ("1", "true", "yes", "on")
            elif isinstance(default, int):
                try:
                    voice[key] = int(val)
                except ValueError:
                    pass
            elif isinstance(default, float):
                try:
                    voice[key] = float(val)
                except ValueError:
                    pass
            else:
                voice[key] = val

    return ArcConfig(
        llm=llm,
        safety_mode=str(raw.get("safety", {}).get("mode", "interactive")).lower(),
        internships=dict(raw.get("internships", {})),
        automation=dict(raw.get("automation", {})),
        gmail={k: str(v) for k, v in raw.get("gmail", {}).items()},
        skills={k: str(v) for k, v in raw.get("skills", {}).items()},
        voice=voice,
        browser=browser,
        data_dir=data_dir,
        config_path=config_yaml,
    )
