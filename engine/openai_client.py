"""
Async streaming client for the OpenAI (ChatGPT) Chat Completions API.

Drop-in sibling of LlamaClient: yields the same chunk contract used by the
FastAPI server and the desktop agent:

    { "token": str, "done": bool, "tokens_per_sec": float, "total_tokens": int }

or on failure:

    { "error": str, "done": True }

Handles per-model quirks automatically:
- Reasoning models (o-series, GPT-5.x, Codex) receive `max_completion_tokens`,
  no sampling params, and an optional `reasoning_effort`.
- A single auto-sanitize retry strips rejected parameters for older or
  OpenAI-compatible endpoints (e.g. `max_completion_tokens` -> `max_tokens`).
"""
import json
import time
import httpx
from typing import AsyncGenerator, Dict, Any, List, Optional

from config import (
    get_openai_api_key,
    get_openai_base_url,
    get_active_openai_model_id,
    describe_openai_model,
    httpx_verify,
)


class OpenAIAPIError(Exception):
    """Structured error from the OpenAI API (or an OpenAI-compatible endpoint)."""

    def __init__(self, status_code: int, message: str, code: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code

    def friendly(self) -> str:
        msg = (self.message or "").strip()
        if self.status_code == 401:
            return "OpenAI rejected the API key (401). Check your key under the 🔑 ChatGPT Key settings."
        if self.status_code == 403:
            return f"OpenAI access denied (403): {msg}"
        if self.status_code == 404:
            return f"OpenAI model or endpoint not found (404): {msg}. Check the model name and API base URL."
        if self.status_code == 429:
            return f"OpenAI rate limit or quota exceeded (429): {msg}"
        if self.status_code == 400:
            return f"OpenAI rejected the request (400): {msg}"
        if 500 <= self.status_code < 600:
            return f"OpenAI server error ({self.status_code}): {msg}"
        return f"OpenAI API error ({self.status_code}): {msg}"


class OpenAIClient:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        # Optional per-instance overrides (used to validate credentials before saving)
        self._base_url_override = (base_url or "").rstrip("/") or None
        self._api_key_override = (api_key or "").strip() or None

    # ------------------------------------------------------------------ utils
    @property
    def base_url(self) -> str:
        return self._base_url_override or get_openai_base_url()

    def _api_key(self) -> str:
        return self._api_key_override or get_openai_api_key()

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }

    async def is_ready(self) -> bool:
        """Cloud engine is ready when an API key is configured."""
        return bool(self._api_key())

    # ------------------------------------------------------------ validation
    async def fetch_models(self) -> Dict[str, Any]:
        """
        Calls GET /models to validate the API key.
        Returns { "ok": bool, "models": [ids], "error": str|None }.
        """
        if not self._api_key():
            return {"ok": False, "models": [], "error": "No OpenAI API key configured."}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=10.0), verify=httpx_verify()) as client:
                res = await client.get(f"{self.base_url}/models", headers=self._headers())
            if res.status_code == 200:
                data = res.json().get("data", []) or []
                ids = sorted({m.get("id") for m in data if isinstance(m, dict) and m.get("id")})
                return {"ok": True, "models": ids, "error": None}
            raise self._api_error_from_body(res.status_code, res.content)
        except OpenAIAPIError as e:
            return {"ok": False, "models": [], "error": e.friendly()}
        except Exception as e:
            return {"ok": False, "models": [], "error": f"Could not reach {self.base_url}: {e}"}

    def _api_error_from_body(self, status_code: int, body: bytes) -> OpenAIAPIError:
        text = body.decode("utf-8", errors="ignore") if isinstance(body, (bytes, bytearray)) else str(body)
        message, code = text[:500], None
        try:
            parsed = json.loads(text)
            err = parsed.get("error") if isinstance(parsed, dict) else None
            if isinstance(err, dict):
                message = err.get("message") or message
                code = err.get("code") or err.get("type")
            elif isinstance(err, str):
                message = err
        except (json.JSONDecodeError, AttributeError):
            pass
        return OpenAIAPIError(status_code, message, code)

    # --------------------------------------------------------------- payload
    def _build_messages(
        self,
        prompt: Optional[str],
        messages: Optional[List[Dict[str, Any]]],
        image_data: Optional[str],
    ) -> List[Dict[str, Any]]:
        if messages:
            msgs = [dict(m) for m in messages]
        else:
            msgs = [{"role": "user", "content": prompt or ""}]

        if image_data:
            data_uri = image_data if image_data.startswith("data:") else f"data:image/jpeg;base64,{image_data}"
            last = msgs[-1] if msgs else {"role": "user", "content": ""}
            if msgs and msgs[-1].get("role") != "user":
                msgs.append({"role": "user", "content": ""})
                last = msgs[-1]
            content = last.get("content")
            if isinstance(content, list):
                parts = list(content)
            else:
                parts = [{"type": "text", "text": content}] if content else []
            parts.append({"type": "image_url", "image_url": {"url": data_uri}})
            last["content"] = parts

        return msgs

    def _build_payload(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        temperature: float,
        top_p: float,
        presence_penalty: float,
        frequency_penalty: float,
        max_tokens: int,
        reasoning_effort: Optional[str],
    ) -> Dict[str, Any]:
        desc = describe_openai_model(model)
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if desc.get("reasoning"):
            # Reasoning tokens count against the completion budget; keep a floor
            # so the visible answer isn't starved (o-series / GPT-5.x / Codex).
            floor = 1024 if reasoning_effort == "low" else 2048
            payload["max_completion_tokens"] = max(int(max_tokens or 0), floor)
            if desc.get("supports_reasoning_effort") and reasoning_effort:
                payload["reasoning_effort"] = reasoning_effort
        else:
            payload["max_completion_tokens"] = int(max_tokens or 1500)
            payload["temperature"] = float(temperature)
            payload["top_p"] = float(top_p)
            payload["presence_penalty"] = float(presence_penalty)
            payload["frequency_penalty"] = float(frequency_penalty)
        return payload

    def _sanitize_payload(self, payload: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Strips parameters rejected by the API (compat with older endpoints)."""
        msg = (error_message or "").lower()
        out = dict(payload)
        changed = False

        def drop(key: str):
            nonlocal changed
            if key in out:
                out.pop(key)
                changed = True

        if "use 'max_completion_tokens'" in msg or 'use "max_completion_tokens"' in msg:
            # Endpoint wants the modern name but we somehow sent the legacy one
            if "max_tokens" in out:
                out["max_completion_tokens"] = out.pop("max_tokens")
                changed = True
        elif "max_completion_tokens" in msg and "max_completion_tokens" in out:
            # Older/compatible endpoint: rename to legacy max_tokens
            out["max_tokens"] = out.pop("max_completion_tokens")
            changed = True

        if any(k in msg for k in ("temperature", "top_p", "top p", "presence_penalty", "frequency_penalty")):
            for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
                drop(k)

        if "reasoning_effort" in msg:
            drop("reasoning_effort")
        if "stream_options" in msg:
            drop("stream_options")

        if not changed:
            # Unknown rejection: conservatively drop all optional knobs we added
            for k in ("reasoning_effort", "stream_options", "temperature", "top_p",
                      "presence_penalty", "frequency_penalty"):
                drop(k)

        return out

    # -------------------------------------------------------------- streaming
    async def _stream_payload(self, payload: Dict[str, Any]) -> AsyncGenerator[Dict[str, Any], None]:
        start_time = time.time()
        token_count = 0
        usage: Optional[Dict[str, Any]] = None
        finish_reason: Optional[str] = None
        saw_content = False

        timeout = httpx.Timeout(connect=15.0, read=300.0, write=30.0, pool=15.0)
        async with httpx.AsyncClient(timeout=timeout, verify=httpx_verify()) as client:
            async with client.stream(
                "POST", f"{self.base_url}/chat/completions", json=payload, headers=self._headers()
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise self._api_error_from_body(response.status_code, body)

                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                    elif line.startswith("{"):
                        data_str = line  # some compatible endpoints skip the SSE prefix
                    else:
                        continue

                    if data_str == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(chunk, dict):
                        continue

                    if chunk.get("error"):
                        err = chunk["error"]
                        msg = err.get("message") if isinstance(err, dict) else str(err)
                        raise OpenAIAPIError(200, msg or "Stream error", None)

                    if chunk.get("usage"):
                        usage = chunk["usage"]

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue

                    choice = choices[0] or {}
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or ""
                    if content:
                        token_count += 1
                        saw_content = True
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]

                    if content or finish_reason:
                        elapsed = max(0.001, time.time() - start_time)
                        total = (usage or {}).get("completion_tokens") or token_count
                        yield {
                            "token": content,
                            "done": bool(finish_reason),
                            "tokens_per_sec": round(token_count / elapsed, 1),
                            "total_tokens": total,
                        }

                    if finish_reason:
                        return

        # Stream closed without an explicit finish marker
        if saw_content:
            elapsed = max(0.001, time.time() - start_time)
            total = (usage or {}).get("completion_tokens") or token_count
            yield {
                "token": "",
                "done": True,
                "tokens_per_sec": round(token_count / elapsed, 1),
                "total_tokens": total,
            }
        elif finish_reason == "length":
            raise OpenAIAPIError(
                200,
                "The model hit its token budget before producing visible output. Try more max tokens.",
            )

    async def stream_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        image_data: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        min_p: float = 0.05,               # llama.cpp-only, ignored here
        repeat_penalty: float = 1.08,      # llama.cpp-only, ignored here
        presence_penalty: float = 0.20,
        frequency_penalty: float = 0.10,
        max_tokens: int = 1500,
        reasoning_effort: Optional[str] = None,
        model: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streams a chat completion from the OpenAI API.

        `messages` (structured) takes precedence; `prompt` is used as a single
        user message when no structured messages are supplied.
        """
        if not self._api_key():
            yield {
                "error": "No OpenAI API key configured. Click the 🔑 ChatGPT Key button in the sidebar to add one.",
                "done": True,
            }
            return

        model_id = model or get_active_openai_model_id()
        built_messages = self._build_messages(prompt, messages, image_data)
        payload = self._build_payload(
            model=model_id,
            messages=built_messages,
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

        attempts = 0
        while attempts < 3:
            attempts += 1
            emitted = False
            try:
                async for event in self._stream_payload(payload):
                    emitted = True
                    yield event
                return
            except OpenAIAPIError as e:
                # Retry once (twice max) with sanitized params if the failure is
                # a parameter rejection and nothing was streamed yet.
                if not emitted and e.status_code == 400 and attempts < 3:
                    sanitized = self._sanitize_payload(payload, e.message)
                    if sanitized != payload:
                        payload = sanitized
                        continue
                yield {"error": e.friendly(), "done": True}
                return
            except httpx.TimeoutException:
                yield {"error": f"OpenAI API request timed out ({self.base_url}).", "done": True}
                return
            except httpx.ConnectError as e:
                yield {"error": f"Could not connect to {self.base_url}: {e}", "done": True}
                return
            except Exception as e:
                yield {"error": f"OpenAI streaming error: {e}", "done": True}
                return
