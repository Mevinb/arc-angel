"""
Provider Router: dispatches inference calls to the active inference engine.

- "local"  -> llama.cpp llama-server via LlamaClient (unchanged behavior)
- "openai" -> ChatGPT API via OpenAIClient

Both clients expose the same stream_completion() async-generator contract, so
the FastAPI server and the desktop agent stay provider-agnostic.
"""
from typing import Any, AsyncGenerator, Dict
from config import get_active_provider, PROVIDER_OPENAI, get_active_engine_info
from .llama_client import LlamaClient
from .openai_client import OpenAIClient


class ProviderRouter:
    def __init__(self, llama_client: LlamaClient = None, openai_client: OpenAIClient = None):
        self.llama_client = llama_client or LlamaClient()
        self.openai_client = openai_client or OpenAIClient()

    @property
    def is_cloud(self) -> bool:
        """True when the OpenAI (ChatGPT) provider is active."""
        return get_active_provider() == PROVIDER_OPENAI

    @property
    def has_vision(self) -> bool:
        """True when the currently active model supports native multimodal vision."""
        return bool(get_active_engine_info().get("has_vision", False))

    def active_client(self):
        return self.openai_client if self.is_cloud else self.llama_client

    async def is_ready(self) -> bool:
        return await self.active_client().is_ready()

    async def stream_completion(self, **kwargs: Any) -> AsyncGenerator[Dict[str, Any], None]:
        async for chunk in self.active_client().stream_completion(**kwargs):
            yield chunk
