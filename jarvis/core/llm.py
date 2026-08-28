"""Phase 1 — OmniRoute: central LLM routing layer.

Talks to any OpenAI-compatible gateway. OmniRoute (https://github.com/diegosouzapw/OmniRoute)
is the default target: it serves http://localhost:20128/v1 and accepts model
names like ``auto`` (balanced), ``auto/fast`` (low latency), ``auto/coding``.

Features
--------
- Role-based routing: FAST (cheap classification), REASONING (planning/analysis),
  VISION (screenshot understanding).
- Automatic model fallback: if the primary model for a role fails, the next
  entry in the fallback chain is tried.
- Tool calling support for the orchestrator's agent loop.
- ``ask_json`` helper with robust JSON extraction for structured analysis.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from ..config import LLMConfig

logger = logging.getLogger("jarvis.llm")

Message = Dict[str, Any]  # {"role": ..., "content": ...}


class ModelRole(str, Enum):
    FAST = "fast"
    REASONING = "reasoning"
    VISION = "vision"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    model: str = ""
    finish_reason: str = ""
    usage: Dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ModelUnavailableError(RuntimeError):
    """All models in the chain failed."""


class LLMRouter:
    """Routes requests to fast/reasoning/vision models with fallbacks."""

    def __init__(
        self,
        config: LLMConfig,
        client_factory: Optional[Callable[[LLMConfig], Any]] = None,
    ) -> None:
        self.config = config
        self._client_factory = client_factory or (lambda cfg: OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key or "missing-key",
            timeout=cfg.timeout_seconds,
            max_retries=cfg.max_retries,
        ))
        self._client: Any = None
        # Simple failure cooldowns: model -> epoch seconds until which it is skipped
        self._cooldown: Dict[str, float] = {}
        self.total_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    # ------------------------------------------------------------------ client
    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._client_factory(self.config)
        return self._client

    def _mark_failure(self, model: str, seconds: float = 60.0) -> None:
        self._cooldown[model] = time.time() + seconds
        logger.warning("Model %r entering %.0fs cooldown after failure", model, seconds)

    def _available_chain(self, role: ModelRole | str) -> List[str]:
        now = time.time()
        chain = self.config.chain(str(role))
        fresh = [m for m in chain if self._cooldown.get(m, 0) <= now]
        return fresh or chain  # if everything is cooling down, still try in order

    # ------------------------------------------------------------------ core
    def complete(
        self,
        messages: List[Message],
        role: ModelRole | str = ModelRole.REASONING,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        force_tool: Optional[str] = None,
    ) -> LLMResponse:
        """Send a chat completion, walking the fallback chain on failure."""
        chain = self._available_chain(role)
        last_error: Exception | None = None

        for model in chain:
            try:
                kwargs: Dict[str, Any] = dict(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                if tools:
                    kwargs["tools"] = [
                        t if "type" in t else {"type": "function", "function": t}
                        for t in tools
                    ]
                    if force_tool:
                        kwargs["tool_choice"] = {"type": "function", "function": {"name": force_tool}}
                response = self.client.chat.completions.create(**kwargs)
                parsed = self._parse_response(response)
                parsed.model = model
                self._track_usage(parsed)
                return parsed
            except Exception as exc:  # noqa: BLE001 - fall through to next model
                last_error = exc
                logger.warning("Model %s failed: %s", model, exc)
                self._mark_failure(model)

        raise ModelUnavailableError(
            f"All models failed for role '{role}': {last_error}"
        ) from last_error

    # ------------------------------------------------------------- helpers
    def chat(self, prompt: str, system: str | None = None,
             role: ModelRole | str = ModelRole.REASONING) -> str:
        """One-shot text completion."""
        messages: List[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.complete(messages, role=role).content.strip()

    def ask_json(self, prompt: str, system: str | None = None,
                 role: ModelRole | str = ModelRole.REASONING) -> Any:
        """Ask the model for JSON and parse it. Raises ValueError on bad JSON."""
        sys_prompt = (system or "") + (
            "\nRespond with a single valid JSON value only. "
            "No markdown fences, no commentary."
        ).strip()
        raw = self.chat(prompt, system=sys_prompt, role=role)
        return extract_json(raw)

    def health(self) -> Dict[str, Any]:
        """Check gateway reachability; returns status dict for `jarvis doctor`.

        Also reports which configured role -> model IDs resolve to models the
        gateway actually offers, so JARVIS (and the user) can see exactly which
        model will be used for each role.
        """
        try:
            catalog = {m.id for m in self.client.models.list().data}
            configured = {}
            for role in ("fast", "reasoning", "vision"):
                mid = self.config.model(role)
                configured[role] = {"model": mid, "known": mid in catalog}
            return {
                "ok": True,
                "base_url": self.config.base_url,
                "models": sorted(catalog)[:15],
                "model_count": len(catalog),
                "configured": configured,
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "base_url": self.config.base_url, "error": str(exc)}

    # ------------------------------------------------------------------ parse
    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        choice = response.choices[0]
        message = choice.message
        tool_calls: List[ToolCall] = []
        for tc in getattr(message, "tool_calls", None) or []:
            args = tc.function.arguments
            try:
                parsed_args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                parsed_args = {"_raw": args}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=parsed_args))

        usage = getattr(response, "usage", None)
        usage_dict = {}
        if usage is not None:
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage_dict[key] = int(getattr(usage, key, 0) or 0)

        return LLMResponse(
            content=(message.content or "").strip(),
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "",
            usage=usage_dict,
        )

    def _track_usage(self, response: LLMResponse) -> None:
        for key in self.total_usage:
            self.total_usage[key] += response.usage.get(key, 0)


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Extract the first JSON object/array from an LLM reply, tolerating fences."""
    if not text:
        raise ValueError("Empty response")
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK.search(text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Fall back to the first {...} or [...] span
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        for index in range(start, len(text)):
            char = text[index]
            if char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:index + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"No valid JSON found in response: {text[:200]!r}")
