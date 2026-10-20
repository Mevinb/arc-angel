"""
Model router with task profiles, local/cloud failover, comparison and voting.

The existing ProviderRouter remains the streaming path for chat.  This service
adds higher-level automatic routing for agents/research plus model comparison,
performance accounting and per-task preferences.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config import (
    PROVIDER_LOCAL,
    PROVIDER_OPENAI,
    describe_openai_model,
    get_active_engine_info,
    get_active_provider,
    get_openai_api_key,
)
from engine.llama_client import LlamaClient
from engine.openai_client import OpenAIClient
from memory.database import db_session
from services.audit import AuditService
from services.crypto import redact_text
from services.safety import SafetyService


class ModelUnavailable(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


PROFILES: Dict[str, Dict[str, Any]] = {
    "local_only": {"description": "Never leave this machine.", "provider": PROVIDER_LOCAL, "model": None},
    "fast": {"description": "Lowest latency.", "model": "gpt-5.4-nano-2026-03-17", "reasoning_effort": "low"},
    "cheap": {"description": "Cheapest cloud tokens.", "model": "gpt-5.4-nano-2026-03-17", "reasoning_effort": "low"},
    "balanced": {"description": "Good quality at moderate cost.", "model": "gpt-5.6-luna", "reasoning_effort": "medium"},
    "quality": {"description": "Best available quality.", "model": "gpt-5.6-terra", "reasoning_effort": "high"},
    "reasoning": {"description": "Deep reasoning tasks.", "model": "o4-mini-2025-04-16", "reasoning_effort": "high"},
    "code": {"description": "Code generation and repair.", "model": "gpt-5.1-codex-mini", "reasoning_effort": "medium"},
}

TASK_DEFAULTS = {
    "chat": "balanced", "research": "quality", "analysis": "reasoning",
    "code": "code", "writing": "balanced", "email": "fast",
    "planning": "reasoning", "council": "quality",
}

# Very rough public list-price estimates (USD / 1K tokens), used only for local
# budgeting/accounting, never billed by this app.
_COST_PER_1K = {
    "gpt-5.6-terra": 0.012, "gpt-5.6-luna": 0.006, "gpt-5.4-mini-2026-03-17": 0.003,
    "gpt-5.4-nano-2026-03-17": 0.001, "gpt-5.1-codex-mini": 0.003,
    "gpt-5-mini-2025-08-07": 0.002, "gpt-5-nano-2025-08-07": 0.001,
    "gpt-4.1-mini-2025-04-14": 0.0008, "gpt-4.1-nano-2025-04-14": 0.0003,
    "o4-mini-2025-04-16": 0.003, "o1-mini-2024-09-12": 0.003,
}


class ModelService:
    @classmethod
    def list_profiles(cls) -> Dict[str, Any]:
        active = get_active_engine_info()
        return {
            "active": {
                "provider": active["provider"], "model": active["model_id"],
                "name": active["name"], "has_vision": active["has_vision"],
            },
            "profiles": [
                {"id": pid, **meta, "effective_provider": cls._profile_provider(meta)}
                for pid, meta in PROFILES.items()
            ],
            "cloud_available": bool(get_openai_api_key()),
            "offline_mode": SafetyService.offline_mode(),
        }

    @classmethod
    def _profile_provider(cls, meta: Dict[str, Any]) -> str:
        if meta.get("provider") == PROVIDER_LOCAL or not get_openai_api_key() or SafetyService.offline_mode():
            return PROVIDER_LOCAL
        return PROVIDER_OPENAI

    @classmethod
    def route(
        cls, task_type: str = "general", profile: Optional[str] = None,
        optimize: Optional[str] = None,
    ) -> Dict[str, Any]:
        active = get_active_engine_info()
        profile = profile or TASK_DEFAULTS.get(task_type, "balanced")
        if profile not in PROFILES:
            profile = "balanced"

        if SafetyService.offline_mode():
            return {
                "provider": PROVIDER_LOCAL, "model": active["model_id"],
                "profile": "local_only", "reasoning_effort": None,
                "reason": "Offline mode forces the local model.",
            }
        if not get_openai_api_key() or active["provider"] == PROVIDER_LOCAL or profile == "local_only":
            return {
                "provider": PROVIDER_LOCAL, "model": active["model_id"],
                "profile": profile, "reasoning_effort": None,
                "reason": "Local provider selected.",
            }

        meta = PROFILES.get(profile, PROFILES["balanced"])
        model = meta.get("model") or active["model_id"]
        if optimize == "cost":
            model = PROFILES["cheap"]["model"]
        elif optimize == "speed":
            model = PROFILES["fast"]["model"]
        elif optimize == "quality":
            model = PROFILES["quality"]["model"]
        return {
            "provider": PROVIDER_OPENAI, "model": model, "profile": profile,
            "reasoning_effort": meta.get("reasoning_effort"),
            "reason": f"Cloud profile '{profile}' selected" + (f" (optimize={optimize})" if optimize else ""),
        }

    @classmethod
    def is_available(cls) -> bool:
        if SafetyService.offline_mode():
            info = get_active_engine_info()
            return info["provider"] == PROVIDER_LOCAL
        return bool(get_openai_api_key()) or get_active_provider() == PROVIDER_LOCAL

    @classmethod
    async def complete(
        cls,
        messages: List[Dict[str, Any]],
        *,
        profile: Optional[str] = None,
        task_type: str = "general",
        max_tokens: int = 1200,
        temperature: float = 0.4,
        timeout: float = 120.0,
        optimize: Optional[str] = None,
    ) -> str:
        route = cls.route(task_type=task_type, profile=profile, optimize=optimize)
        errors: List[str] = []
        attempts = [route]
        # Failover: if the selected provider fails, try the other configured one.
        if route["provider"] == PROVIDER_OPENAI:
            attempts.append({"provider": PROVIDER_LOCAL, "model": get_active_engine_info()["model_id"], "reasoning_effort": None})
        elif get_openai_api_key() and not SafetyService.offline_mode():
            attempts.append({
                "provider": PROVIDER_OPENAI, "model": PROFILES["fast"]["model"],
                "reasoning_effort": "low",
            })

        for attempt in attempts:
            started = time.time()
            try:
                text, usage = await cls._stream_once(
                    messages, attempt, max_tokens, temperature, timeout
                )
                latency = (time.time() - started) * 1000
                cls.record_run(
                    attempt["model"], attempt["provider"], task_type, True,
                    latency, usage.get("tokens", 0),
                    cls.estimate_cost(attempt["model"], usage.get("tokens", 0)),
                )
                return text
            except Exception as exc:
                latency = (time.time() - started) * 1000
                errors.append(f"{attempt['provider']}:{attempt['model']}: {exc}")
                cls.record_run(
                    attempt["model"], attempt["provider"], task_type, False,
                    latency, 0, 0.0,
                )
        raise ModelUnavailable("; ".join(errors) or "No model provider is available")

    @classmethod
    async def _stream_once(
        cls,
        messages: List[Dict[str, Any]],
        route: Dict[str, Any],
        max_tokens: int,
        temperature: float,
        timeout: float,
    ) -> tuple:
        tokens: List[str] = []
        usage = {"tokens": 0}
        client = OpenAIClient() if route["provider"] == PROVIDER_OPENAI else LlamaClient()
        if route["provider"] == PROVIDER_OPENAI and not get_openai_api_key():
            raise ModelUnavailable("No OpenAI API key configured")
        kwargs = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if route["provider"] == PROVIDER_OPENAI:
            kwargs["model"] = route["model"]
            if route.get("reasoning_effort"):
                kwargs["reasoning_effort"] = route["reasoning_effort"]
        try:
            import asyncio

            async with asyncio.timeout(timeout):
                async for chunk in client.stream_completion(**kwargs):
                    if chunk.get("error"):
                        raise ModelUnavailable(str(chunk["error"]))
                    if chunk.get("token"):
                        tokens.append(chunk["token"])
                    usage["tokens"] = int(chunk.get("total_tokens") or usage["tokens"])
                    if chunk.get("done"):
                        break
        except TimeoutError as exc:
            raise ModelUnavailable(f"timed out after {timeout:.0f}s") from exc
        text = "".join(tokens).strip()
        if not text:
            raise ModelUnavailable("model returned an empty response")
        AuditService.log(
            "model", "model.complete", actor="system", status="ok",
            summary=f"{route['provider']}:{route['model']} completed",
            data={
                "model": route["model"], "provider": route["provider"],
                "prompt": redact_text(json.dumps(messages, default=str))[:8000],
                "response": redact_text(text)[:8000],
                "tokens": usage["tokens"],
            },
            activity=False,
        )
        return text, usage

    @classmethod
    def estimate_cost(cls, model: str, tokens: int) -> float:
        return round((tokens / 1000.0) * _COST_PER_1K.get(model, 0.002), 6)

    # ------------------------------------------------------------- accounting
    @classmethod
    def record_run(
        cls, model: str, provider: str, task_type: str, success: bool,
        latency_ms: float, tokens: int, cost_usd: float,
    ) -> None:
        with db_session() as conn:
            row = conn.execute(
                "SELECT * FROM model_stats WHERE model=? AND provider=? AND task_type=?",
                (model, provider, task_type),
            ).fetchone()
            if row:
                conn.execute(
                    """
                    UPDATE model_stats SET runs=runs+1, successes=successes+?, total_latency_ms=total_latency_ms+?,
                        tokens=tokens+?, cost_usd=cost_usd+?, updated_at=? WHERE model=? AND provider=? AND task_type=?
                    """,
                    (
                        1 if success else 0, latency_ms, tokens, cost_usd, _now(),
                        model, provider, task_type,
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO model_stats (model, provider, task_type, runs, successes, total_latency_ms,
                        tokens, cost_usd, updated_at) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                    """,
                    (model, provider, task_type, 1 if success else 0, latency_ms, tokens, cost_usd, _now()),
                )

    @classmethod
    def performance(cls) -> List[Dict[str, Any]]:
        with db_session() as conn:
            rows = conn.execute(
                "SELECT * FROM model_stats ORDER BY runs DESC, successes DESC"
            ).fetchall()
        out = []
        for row in rows:
            runs = row["runs"] or 0
            successes = row["successes"] or 0
            out.append({
                "model": row["model"], "provider": row["provider"],
                "task_type": row["task_type"], "runs": runs,
                "successes": successes,
                "success_rate": round(successes / runs, 3) if runs else 0.0,
                "avg_latency_ms": round((row["total_latency_ms"] or 0) / runs, 1) if runs else 0,
                "tokens": row["tokens"], "cost_usd": round(row["cost_usd"] or 0, 6),
                "votes_up": row["votes_up"], "votes_down": row["votes_down"],
                "updated_at": row["updated_at"],
            })
        return out

    # -------------------------------------------------------------- comparison
    @classmethod
    async def compare(
        cls, prompt: str, models: List[str], max_tokens: int = 500,
    ) -> Dict[str, Any]:
        models = [m for m in dict.fromkeys(models or []) if m][:4]
        if not models:
            raise ValueError("Provide between 1 and 4 model ids")
        results = []
        for model_id in models:
            started = time.time()
            provider = PROVIDER_LOCAL if not model_id.startswith("openai:") else PROVIDER_OPENAI
            raw_model = model_id.split(":", 1)[1] if provider == PROVIDER_OPENAI else model_id
            try:
                if provider == PROVIDER_OPENAI:
                    if not get_openai_api_key():
                        raise ModelUnavailable("No OpenAI API key configured")
                    client = OpenAIClient()
                    kwargs = {"messages": [{"role": "user", "content": prompt}], "model": raw_model, "max_tokens": max_tokens}
                else:
                    info = get_active_engine_info()
                    if info["provider"] != PROVIDER_LOCAL:
                        raise ModelUnavailable("Switch to the local provider to compare local models")
                    raw_model = info["model_id"]
                    client = LlamaClient()
                    kwargs = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
                tokens: List[str] = []
                usage = 0
                import asyncio

                async with asyncio.timeout(90):
                    async for chunk in client.stream_completion(**kwargs):
                        if chunk.get("error"):
                            raise ModelUnavailable(str(chunk["error"]))
                        if chunk.get("token"):
                            tokens.append(chunk["token"])
                        usage = int(chunk.get("total_tokens") or usage)
                        if chunk.get("done"):
                            break
                text = "".join(tokens).strip()
                latency = round((time.time() - started) * 1000, 1)
                cls.record_run(raw_model, provider, "comparison", bool(text), latency, usage, cls.estimate_cost(raw_model, usage))
                results.append({
                    "model": model_id, "provider": provider, "raw_model": raw_model,
                    "ok": bool(text), "response": text, "latency_ms": latency,
                    "tokens": usage, "error": None,
                })
            except Exception as exc:
                latency = round((time.time() - started) * 1000, 1)
                cls.record_run(raw_model, provider, "comparison", False, latency, 0, 0.0)
                results.append({
                    "model": model_id, "provider": provider, "raw_model": raw_model,
                    "ok": False, "response": "", "latency_ms": latency,
                    "tokens": 0, "error": str(exc),
                })
        AuditService.log(
            "model", "model.compare", actor="user", status="ok",
            summary=f"Compared {len(models)} models",
            data={"prompt": redact_text(prompt)[:4000], "models": models,
                  "results": [{k: v for k, v in r.items() if k != "response"} for r in results]},
        )
        return {"prompt": prompt, "results": results}

    @classmethod
    def vote(cls, prompt: str, model: str, provider: str, vote: str) -> Dict[str, Any]:
        if vote not in ("up", "down"):
            raise ValueError("vote must be 'up' or 'down'")
        prompt_hash = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()[:24]
        with db_session() as conn:
            conn.execute(
                "INSERT INTO model_votes (prompt_hash, model, provider, vote, created_at) VALUES (?, ?, ?, ?, ?)",
                (prompt_hash, model, provider, vote, _now()),
            )
            row = conn.execute(
                "SELECT * FROM model_stats WHERE model=? AND provider=? AND task_type='comparison'",
                (model, provider),
            ).fetchone()
            if row:
                conn.execute(
                    f"UPDATE model_stats SET votes_{vote}=votes_{vote}+1, updated_at=? WHERE model=? AND provider=? AND task_type='comparison'",
                    (_now(), model, provider),
                )
            else:
                up = 1 if vote == "up" else 0
                down = 1 if vote == "down" else 0
                conn.execute(
                    """
                    INSERT INTO model_stats (model, provider, task_type, runs, successes,
                        total_latency_ms, tokens, cost_usd, votes_up, votes_down, updated_at)
                    VALUES (?, ?, 'comparison', 0, 0, 0, 0, 0, ?, ?, ?)
                    """,
                    (model, provider, up, down, _now()),
                )
        return {"ok": True, "model": model, "vote": vote}
