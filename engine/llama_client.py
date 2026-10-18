"""
Async HTTP client for llama.cpp server with SSE streaming support.

Supports three call forms used across the app:
1. Raw prompt  -> /completion endpoint with explicit ChatML prompt (chat flow)
2. messages    -> /v1/chat/completions (structured chat, used by cloud parity)
3. image_data  -> /v1/chat/completions with multimodal content parts
"""
import json
import time
import httpx
from typing import AsyncGenerator, Dict, Any, List, Optional
from config import LLAMA_URL, httpx_verify

class LlamaClient:
    def __init__(self, base_url: str = LLAMA_URL):
        self.base_url = base_url.rstrip("/")

    async def is_ready(self) -> bool:
        """Checks if llama-server is alive and ready to process requests."""
        try:
            async with httpx.AsyncClient(timeout=2.0, verify=httpx_verify()) as client:
                res = await client.get(f"{self.base_url}/health")
                return res.status_code == 200
        except Exception:
            return False

    async def stream_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        image_data: Optional[str] = None,
        temperature: float = 0.8,
        top_p: float = 0.9,
        min_p: float = 0.05,
        repeat_penalty: float = 1.08,
        presence_penalty: float = 0.20,
        frequency_penalty: float = 0.10,
        max_tokens: int = 1500,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streams completion tokens from llama-server.
        Supports multimodal vision if image_data (base64) is provided.
        Yields dicts with:
        { "token": str, "done": bool, "tokens_per_sec": float, "total_tokens": int }
        """
        use_chat_api = bool(image_data) or (messages is not None)

        if use_chat_api:
            if messages:
                msgs: List[Dict[str, Any]] = [dict(m) for m in messages]
            else:
                msgs = [{"role": "user", "content": prompt or ""}]

            if image_data:
                data_uri = image_data if image_data.startswith("data:") else f"data:image/jpeg;base64,{image_data}"
                last = msgs[-1] if msgs else {"role": "user", "content": ""}
                if msgs and msgs[-1].get("role") != "user":
                    msgs.append({"role": "user", "content": ""})
                    last = msgs[-1]
                content = last.get("content")
                text_str = content if isinstance(content, str) else (prompt or "")
                # Standard OpenAI multimodal format: image_url part followed by text prompt
                parts = [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": text_str}
                ]
                last["content"] = parts

            chat_payload = {
                "messages": msgs,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": True,
                "chat_template_kwargs": {"enable_thinking": False}
            }
            if reasoning_effort:
                chat_payload["reasoning_effort"] = reasoning_effort

            start_time = time.time()
            token_count = 0

            try:
                async with httpx.AsyncClient(timeout=120.0, verify=httpx_verify()) as client:
                    async with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=chat_payload) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            yield {
                                "error": f"llama-server error {response.status_code}: {error_text.decode('utf-8', errors='ignore')}",
                                "done": True
                            }
                            return

                        async for line in response.aiter_lines():
                            line = line.strip()
                            if not line or not line.startswith("data: "):
                                continue

                            data_str = line[6:]
                            if data_str == "[DONE]":
                                break

                            try:
                                chunk = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue

                            choices = chunk.get("choices", [])
                            if not choices:
                                continue

                            delta = choices[0].get("delta", {})
                            content = delta.get("content") or delta.get("reasoning_content") or ""
                            finish = choices[0].get("finish_reason")
                            stop = finish is not None

                            if content:
                                token_count += 1
                            elapsed = max(0.001, time.time() - start_time)
                            tok_per_sec = round(token_count / elapsed, 1)

                            if content or stop:
                                yield {
                                    "token": content,
                                    "done": stop,
                                    "tokens_per_sec": tok_per_sec,
                                    "total_tokens": token_count
                                }

                            if stop:
                                break
            except Exception as e:
                yield {
                    "error": f"llama-server connection error: {e}",
                    "done": True
                }
            return

        payload = {
            "prompt": prompt or "",
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "repeat_penalty": repeat_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "n_predict": max_tokens,
            "stream": True,
            "stop": ["<|im_end|>", "<|eot_id|>", "<|im_start|>", "User:", "ASSISTANT:", "<|end_of_text|>", "<|endoftext|>"]
        }

        start_time = time.time()
        token_count = 0

        async with httpx.AsyncClient(timeout=120.0, verify=httpx_verify()) as client:
            async with client.stream("POST", f"{self.base_url}/completion", json=payload) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    yield {
                        "error": f"llama-server error {response.status_code}: {error_text.decode('utf-8', errors='ignore')}",
                        "done": True
                    }
                    return

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data: "):
                        continue
                    
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    content = chunk.get("content", "")
                    stop = chunk.get("stop", False)
                    token_count += 1
                    elapsed = max(0.001, time.time() - start_time)
                    tok_per_sec = round(token_count / elapsed, 1)

                    yield {
                        "token": content,
                        "done": stop,
                        "tokens_per_sec": tok_per_sec,
                        "total_tokens": token_count
                    }

                    if stop:
                        break
