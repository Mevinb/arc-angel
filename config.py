import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACES_DIR = DATA_DIR / "workspaces"
WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = Path(os.getenv("GROK_DB_PATH", str(DATA_DIR / "grok_memory.db")))
ACTIVE_MODEL_FILE = Path(os.getenv("GROK_STATE_FILE", str(DATA_DIR / "active_model.json")))

# ---------------------------------------------------------------------------
# Providers: local llama.cpp engine or OpenAI (ChatGPT API)
# ---------------------------------------------------------------------------
PROVIDER_LOCAL = "local"
PROVIDER_OPENAI = "openai"

# LLM Engine Configuration & Model Catalog
LLAMA_SERVER_BIN = Path("/home/mevlec/Downloads/llama.cpp/build/bin/llama-server")

MODELS: Dict[str, Dict[str, Any]] = {
    "qwen": {
        "id": "qwen",
        "name": "Qwen 3.5 9B Vision",
        "path": Path("/home/mevlec/Downloads/Qwen3.5-9B-Q4_K_M.gguf"),
        "mmproj": Path("/home/mevlec/Downloads/mmproj-Qwen3.5-9B-f16.gguf"),
        "default_ngl": 24,  # Safe GPU offload leaving headroom for 0.9GB mmproj & KV cache on 6GB VRAM
        "total_layers": 32,
        "has_vision": True,
        "alias": "qwen-vision",
        "badge": "Vision 9B"
    },
    "dolphin": {
        "id": "dolphin",
        "name": "Dolphin 3.0 Llama 3.1 8B",
        "path": Path("/home/mevlec/Downloads/dolphin3.0-llama3.1-8b-q4_k_m.gguf"),
        "mmproj": None,
        "default_ngl": 33,  # Full GPU offload for 8B text-only
        "total_layers": 33,
        "has_vision": False,
        "alias": "dolphin-grok",
        "badge": "Uncensored 8B"
    }
}

DEFAULT_MODEL_ID = "qwen" if MODELS["qwen"]["path"].exists() else "dolphin"

# ---------------------------------------------------------------------------
# OpenAI (ChatGPT API) Configuration
# ---------------------------------------------------------------------------
OPENAI_BASE_URL_DEFAULT = "https://api.openai.com/v1"
OPENAI_KEY_FILE = Path(os.getenv("GROK_OPENAI_KEY_FILE", str(DATA_DIR / "openai_key.json")))
OPENAI_MODEL_PREFIX = "openai:"

# Catalog of ChatGPT models available on this account (mini/nano tier + flagships).
# Each entry is annotated with vision support, reasoning-model behavior and
# reasoning_effort support so payloads are built correctly per model family.
OPENAI_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-5.6-terra": {
        "id": "gpt-5.6-terra", "name": "GPT-5.6 Terra", "badge": "ChatGPT",
        "has_vision": True, "reasoning": True, "supports_reasoning_effort": True,
    },
    "gpt-5.6-luna": {
        "id": "gpt-5.6-luna", "name": "GPT-5.6 Luna", "badge": "ChatGPT",
        "has_vision": True, "reasoning": True, "supports_reasoning_effort": True,
    },
    "gpt-5.4-mini-2026-03-17": {
        "id": "gpt-5.4-mini-2026-03-17", "name": "GPT-5.4 Mini", "badge": "ChatGPT",
        "has_vision": True, "reasoning": True, "supports_reasoning_effort": True,
    },
    "gpt-5.4-nano-2026-03-17": {
        "id": "gpt-5.4-nano-2026-03-17", "name": "GPT-5.4 Nano", "badge": "ChatGPT",
        "has_vision": True, "reasoning": True, "supports_reasoning_effort": True,
    },
    "gpt-5.1-codex-mini": {
        "id": "gpt-5.1-codex-mini", "name": "GPT-5.1 Codex Mini", "badge": "Codex",
        "has_vision": False, "reasoning": True, "supports_reasoning_effort": True,
    },
    "gpt-5-mini-2025-08-07": {
        "id": "gpt-5-mini-2025-08-07", "name": "GPT-5 Mini", "badge": "ChatGPT",
        "has_vision": True, "reasoning": True, "supports_reasoning_effort": True,
    },
    "gpt-5-nano-2025-08-07": {
        "id": "gpt-5-nano-2025-08-07", "name": "GPT-5 Nano", "badge": "ChatGPT",
        "has_vision": True, "reasoning": True, "supports_reasoning_effort": True,
    },
    "gpt-4.1-mini-2025-04-14": {
        "id": "gpt-4.1-mini-2025-04-14", "name": "GPT-4.1 Mini", "badge": "ChatGPT",
        "has_vision": True, "reasoning": False, "supports_reasoning_effort": False,
    },
    "gpt-4.1-nano-2025-04-14": {
        "id": "gpt-4.1-nano-2025-04-14", "name": "GPT-4.1 Nano", "badge": "ChatGPT",
        "has_vision": True, "reasoning": False, "supports_reasoning_effort": False,
    },
    "gpt-4o-mini-2024-07-18": {
        "id": "gpt-4o-mini-2024-07-18", "name": "GPT-4o Mini", "badge": "ChatGPT",
        "has_vision": True, "reasoning": False, "supports_reasoning_effort": False,
    },
    "o4-mini-2025-04-16": {
        "id": "o4-mini-2025-04-16", "name": "o4-mini", "badge": "Reasoning",
        "has_vision": True, "reasoning": True, "supports_reasoning_effort": True,
    },
    "o1-mini-2024-09-12": {
        "id": "o1-mini-2024-09-12", "name": "o1-mini", "badge": "Reasoning",
        "has_vision": False, "reasoning": True, "supports_reasoning_effort": False,
    },
    "codex-mini-latest": {
        "id": "codex-mini-latest", "name": "Codex Mini", "badge": "Codex",
        "has_vision": False, "reasoning": True, "supports_reasoning_effort": True,
    },
}

DEFAULT_OPENAI_MODEL_ID = "gpt-5.6-terra"


def describe_openai_model(model_id: str) -> Dict[str, Any]:
    """Returns capability metadata for a model id (catalog entry or heuristic)."""
    if model_id in OPENAI_MODELS:
        return dict(OPENAI_MODELS[model_id])

    mid = (model_id or "").lower()
    vision = any(mid.startswith(p) for p in (
        "gpt-5", "gpt-4.1", "gpt-4o", "gpt-4-turbo", "gpt-4-vision", "o3", "o4", "chatgpt"
    ))
    if "codex" in mid:
        vision = False
    reasoning = any(mid.startswith(p) for p in ("o1", "o3", "o4", "gpt-5")) or "codex" in mid
    effort = any(mid.startswith(p) for p in ("gpt-5", "o3", "o4"))

    # Pretty display name: drop date suffix, prettify gpt/o tokens
    import re as _re
    base = _re.sub(r"-(20\d\d-\d\d-\d\d)$", "", model_id or "unknown")
    words = []
    for w in base.split("-"):
        if w == "gpt":
            words.append("GPT")
        elif w.startswith("gpt"):
            words.append("GPT" + w[3:])
        elif _re.match(r"^o\d$", w):
            words.append(w)
        else:
            words.append(w[:1].upper() + w[1:])
    name = " ".join(words)

    return {
        "id": model_id,
        "name": name,
        "badge": "ChatGPT",
        "has_vision": vision,
        "reasoning": reasoning,
        "supports_reasoning_effort": effort,
    }


# ---------------------------------------------------------------------------
# Persisted state (active provider, active local model, active OpenAI model)
# ---------------------------------------------------------------------------
def _read_state() -> Dict[str, Any]:
    if ACTIVE_MODEL_FILE.exists():
        try:
            data = json.loads(ACTIVE_MODEL_FILE.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _write_state(patch: Dict[str, Any]) -> None:
    state = _read_state()
    state.update(patch)
    ACTIVE_MODEL_FILE.write_text(json.dumps(state, indent=2))


def get_active_model_id() -> str:
    """Reads persisted active local model ID, falling back to default."""
    state = _read_state()
    m_id = state.get("active_model_id")
    if m_id in MODELS and MODELS[m_id]["path"].exists():
        return m_id
    return DEFAULT_MODEL_ID


def set_active_model_id(model_id: str) -> bool:
    """Persists chosen active local model ID and switches provider to local."""
    if model_id in MODELS:
        _write_state({"active_model_id": model_id, "provider": PROVIDER_LOCAL})
        return True
    return False


def get_active_model_config() -> Dict[str, Any]:
    """Retrieves full config dictionary for the currently active local model."""
    m_id = get_active_model_id()
    return MODELS.get(m_id, MODELS[DEFAULT_MODEL_ID])


def get_active_provider() -> str:
    """Returns the active inference provider ('local' or 'openai')."""
    provider = _read_state().get("provider")
    return provider if provider in (PROVIDER_LOCAL, PROVIDER_OPENAI) else PROVIDER_LOCAL


def set_active_provider(provider: str) -> bool:
    if provider not in (PROVIDER_LOCAL, PROVIDER_OPENAI):
        return False
    _write_state({"provider": provider})
    return True


def get_active_openai_model_id() -> str:
    """Returns the persisted active OpenAI model id (validated against catalog)."""
    mid = _read_state().get("openai_model")
    if mid and isinstance(mid, str):
        return mid
    return DEFAULT_OPENAI_MODEL_ID


def set_active_openai_model_id(model_id: str) -> bool:
    """Persists the active OpenAI model id (accepts raw or 'openai:' prefixed id)."""
    mid = (model_id or "").strip()
    if mid.startswith(OPENAI_MODEL_PREFIX):
        mid = mid[len(OPENAI_MODEL_PREFIX):]
    if not mid:
        return False
    _write_state({"openai_model": mid})
    return True


def to_ui_model_id(provider: str, model_id: str) -> str:
    """Combined id used by the frontend model dropdown."""
    if provider == PROVIDER_OPENAI:
        return f"{OPENAI_MODEL_PREFIX}{model_id}"
    return model_id


def parse_ui_model_id(ui_model_id: str) -> Tuple[str, str]:
    """Splits a UI model id into (provider, raw_model_id)."""
    ui_model_id = (ui_model_id or "").strip()
    if ui_model_id.startswith(OPENAI_MODEL_PREFIX):
        return PROVIDER_OPENAI, ui_model_id[len(OPENAI_MODEL_PREFIX):]
    return PROVIDER_LOCAL, ui_model_id


# ---------------------------------------------------------------------------
# HTTP/TLS helpers
# ---------------------------------------------------------------------------
def httpx_verify():
    """
    Returns the `verify=` value for httpx clients.

    httpx needs a CA bundle to build its default SSL context (normally certifi).
    On interpreters where certifi is not installed (e.g. fresh uv Python
    environments) fall back to the interpreter's default SSL context so the app
    still works for both local llama.cpp (plain HTTP) and HTTPS APIs.
    """
    try:
        import certifi  # noqa: F401
        return True
    except Exception:
        import ssl
        return ssl.create_default_context()


# ---------------------------------------------------------------------------
# OpenAI credentials (env var takes precedence, then local key file)
# ---------------------------------------------------------------------------
def get_openai_api_key() -> str:
    """Resolves the OpenAI API key from OPENAI_API_KEY env or the local key file."""
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    if OPENAI_KEY_FILE.exists():
        try:
            data = json.loads(OPENAI_KEY_FILE.read_text())
            return (data.get("api_key") or "").strip()
        except Exception:
            return ""
    return ""


def get_openai_key_source() -> Optional[str]:
    """'env' when from environment, 'file' when saved via UI, else None."""
    if os.getenv("OPENAI_API_KEY", "").strip():
        return "env"
    if OPENAI_KEY_FILE.exists():
        try:
            data = json.loads(OPENAI_KEY_FILE.read_text())
            if (data.get("api_key") or "").strip():
                return "file"
        except Exception:
            pass
    return None


def mask_api_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 12:
        return "•" * len(key)
    return f"{key[:7]}…{key[-4:]}"


def get_openai_base_url() -> str:
    """Base URL for OpenAI API calls (env > saved file > official default)."""
    env_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    if OPENAI_KEY_FILE.exists():
        try:
            data = json.loads(OPENAI_KEY_FILE.read_text())
            saved = (data.get("base_url") or "").strip()
            if saved:
                return saved.rstrip("/")
        except Exception:
            pass
    return OPENAI_BASE_URL_DEFAULT


def set_openai_credentials(api_key: str, base_url: Optional[str] = None) -> None:
    """Persists API key (and optional base URL) to a 0600 local file."""
    data: Dict[str, str] = {"api_key": api_key.strip()}
    if base_url and base_url.strip() and base_url.strip().rstrip("/") != OPENAI_BASE_URL_DEFAULT:
        data["base_url"] = base_url.strip().rstrip("/")
    OPENAI_KEY_FILE.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(OPENAI_KEY_FILE, 0o600)
    except OSError:
        pass


def clear_openai_credentials() -> None:
    """Removes the stored key file (env-provided keys remain active)."""
    if OPENAI_KEY_FILE.exists():
        try:
            OPENAI_KEY_FILE.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Unified active engine info (local llama.cpp or OpenAI ChatGPT)
# ---------------------------------------------------------------------------
def get_active_engine_info() -> Dict[str, Any]:
    """Provider-agnostic description of the active inference engine."""
    provider = get_active_provider()
    if provider == PROVIDER_OPENAI:
        mid = get_active_openai_model_id()
        desc = describe_openai_model(mid)
        return {
            "provider": PROVIDER_OPENAI,
            "model_id": mid,
            "ui_model_id": to_ui_model_id(provider, mid),
            "name": desc["name"],
            "badge": desc.get("badge", "ChatGPT"),
            "has_vision": bool(desc.get("has_vision", False)),
            "reasoning": bool(desc.get("reasoning", False)),
            "supports_reasoning_effort": bool(desc.get("supports_reasoning_effort", False)),
        }
    cfg = get_active_model_config()
    return {
        "provider": PROVIDER_LOCAL,
        "model_id": cfg["id"],
        "ui_model_id": cfg["id"],
        "name": cfg["name"],
        "badge": cfg.get("badge", ""),
        "has_vision": bool(cfg.get("has_vision", False)),
        "reasoning": False,
        "supports_reasoning_effort": False,
    }


def get_active_engine_label() -> str:
    """Short human label used in UI/agent messages."""
    info = get_active_engine_info()
    if info["provider"] == PROVIDER_OPENAI:
        return f"{info['name']} (ChatGPT API)"
    return info["name"]


LLAMA_HOST = os.getenv("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8080"))
LLAMA_URL = f"http://{LLAMA_HOST}:{LLAMA_PORT}"

# Web Server Configuration
WEB_HOST = os.getenv("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("WEB_PORT", "8000"))

# Context & Memory Limits (Designed for local models + vision tokens)
DEFAULT_CTX_SIZE = int(os.getenv("CTX_SIZE", "8192"))  # Expanded context to 8k for vision + tools
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "3000"))
MAX_DOSSIER_TOKENS = 250
MAX_EPISODIC_TOKENS = 350
MAX_SUMMARY_TOKENS = 200
MAX_SLIDING_WINDOW_TURNS = 6  # Last 3 user/assistant pairs verbatim

# Cloud models have huge context windows: allow richer memory injection tiers.
CLOUD_PROMPT_BUDGET_TOKENS = int(os.getenv("CLOUD_PROMPT_BUDGET_TOKENS", "12000"))
CLOUD_SLIDING_BUDGET_TOKENS = int(os.getenv("CLOUD_SLIDING_BUDGET_TOKENS", "6000"))
CLOUD_SLIDING_WINDOW_TURNS = 10


def get_memory_budgets() -> Dict[str, int]:
    """
    Provider-aware token budgets for the 4-tier memory orchestrator.
    Local mode keeps the original tight 4k-context budgets; cloud mode allows
    substantially larger memory injection since ChatGPT context is huge.
    """
    if get_active_provider() == PROVIDER_OPENAI:
        return {
            "max_prompt_tokens": CLOUD_PROMPT_BUDGET_TOKENS,
            "dossier_tokens": 700,
            "episodic_tokens": 1000,
            "summary_tokens": 600,
            "sliding_window_turns": CLOUD_SLIDING_WINDOW_TURNS,
            "sliding_budget": CLOUD_SLIDING_BUDGET_TOKENS,
        }
    return {
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "dossier_tokens": MAX_DOSSIER_TOKENS,
        "episodic_tokens": MAX_EPISODIC_TOKENS,
        "summary_tokens": MAX_SUMMARY_TOKENS,
        "sliding_window_turns": MAX_SLIDING_WINDOW_TURNS,
        "sliding_budget": 1600,
    }


# Hardware & GPU Offload Configuration
DEFAULT_NGL = int(os.getenv("GPU_LAYERS", "33"))
FLASH_ATTN = os.getenv("FLASH_ATTN", "on")         # Flash attention cuts KV cache VRAM in half

def detect_recommended_gpu_layers(model_id: Optional[str] = None) -> int:
    """
    Checks free NVIDIA VRAM and returns optimal layers for the model.
    """
    m_id = model_id or get_active_model_id()
    cfg = MODELS.get(m_id, MODELS[DEFAULT_MODEL_ID])
    default_ngl = cfg.get("default_ngl", 24)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
            check=True
        )
        free_mb = int(result.stdout.strip().split("\n")[0])
        if free_mb >= 5000:
            return default_ngl
        elif free_mb >= 3000:
            return max(10, int(default_ngl * 0.7))
        elif free_mb >= 1500:
            return max(5, int(default_ngl * 0.4))
        else:
            return 0   # CPU inference
    except Exception:
        return default_ngl

RECOMMENDED_NGL = DEFAULT_NGL


# For backwards compatibility with direct imports
MODEL_PATH = get_active_model_config()["path"]

# ---------------------------------------------------------------------------
# Desktop Pilot Display Mode (Physical Screen vs Isolated Virtual Display)
# ---------------------------------------------------------------------------
PILOT_DISPLAY_MODE_FILE = DATA_DIR / "pilot_display_mode.json"

def get_pilot_isolated_display() -> bool:
    """Returns True if Desktop Pilot is configured to run in an isolated virtual display (Xvfb :99)."""
    env_val = os.getenv("PILOT_ISOLATED_DISPLAY")
    if env_val is not None:
        return env_val.strip().lower() in ("1", "true", "yes")
    if PILOT_DISPLAY_MODE_FILE.exists():
        try:
            data = json.loads(PILOT_DISPLAY_MODE_FILE.read_text())
            return bool(data.get("isolated_display", False))
        except Exception:
            pass
    return False

def set_pilot_isolated_display(enabled: bool) -> None:
    """Sets Desktop Pilot display isolation mode (persisted to disk)."""
    PILOT_DISPLAY_MODE_FILE.write_text(json.dumps({"isolated_display": bool(enabled)}))

