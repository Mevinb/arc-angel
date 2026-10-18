#!/usr/bin/env python3
"""
End-to-end integration test for GROK's ChatGPT (OpenAI) provider.

Runs the real FastAPI app against an offline mock OpenAI server:
- API key save + validation (valid, invalid)
- Provider/model listing and switching
- Streaming chat through the OpenAI client + 4-tier memory persistence
- Auto-sanitize retry for reasoning models rejecting sampling params
- Key removal reverting to the local provider

Run:
    python3.12 scripts/test_openai_integration.py
"""
import asyncio
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Isolated test state (do not touch the real DB, key file, or provider state)
TEST_DIR = Path("/tmp/grok_openai_test")
TEST_DIR.mkdir(parents=True, exist_ok=True)
for leftover in TEST_DIR.glob("*"):
    if leftover.is_file():
        leftover.unlink()

os.environ["GROK_DB_PATH"] = str(TEST_DIR / "test_memory.db")
os.environ["GROK_STATE_FILE"] = str(TEST_DIR / "active_model.json")
os.environ["GROK_OPENAI_KEY_FILE"] = str(TEST_DIR / "openai_key.json")
os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:8123/v1"
os.environ.pop("OPENAI_API_KEY", None)

MOCK_HOST, MOCK_PORT = "127.0.0.1", 8123


def start_mock_server():
    import uvicorn

    import mock_openai_server

    uvicorn.run(mock_openai_server.app, host=MOCK_HOST, port=MOCK_PORT, log_level="error")


def wait_for_port(host: str, port: int, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def check(condition: bool, label: str):
    if not condition:
        raise AssertionError(f"FAILED: {label}")
    print(f"  [ok] {label}")


async def main():
    print("[*] Starting mock OpenAI server on 127.0.0.1:8123 ...")
    threading.Thread(target=start_mock_server, daemon=True).start()
    check(wait_for_port(MOCK_HOST, MOCK_PORT), "mock OpenAI server is up")

    import httpx
    import server  # imported AFTER env isolation is in place

    transport = httpx.ASGITransport(app=server.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
        print("\n[1] Status before any key (should be local provider)")
        r = await client.get("/api/status")
        status = r.json()
        check(status["provider"] == "local", "provider defaults to local")
        check(isinstance(status["engine_ready"], bool), "status reports a boolean engine readiness flag")
        check(status["has_openai_key"] is False, "no OpenAI key is configured yet")

        print("\n[2] Rejecting an invalid API key")
        r = await client.post("/api/openai/key", json={"api_key": "sk-wrong"})
        check(r.status_code == 400, "invalid key rejected with HTTP 400")
        check("401" in r.json().get("detail", ""), "error detail mentions the 401 from OpenAI")

        print("\n[3] Saving a valid API key (auto-switches to ChatGPT)")
        r = await client.post("/api/openai/key", json={"api_key": "sk-mock-valid"})
        check(r.status_code == 200, "valid key accepted")
        payload = r.json()
        check(payload["provider"] == "openai", "provider switched to openai")
        check(payload["active_model"] == "gpt-5.6-terra", "default ChatGPT model selected")

        print("\n[4] Status after key (cloud engine ready, no local process needed)")
        r = await client.get("/api/status")
        status = r.json()
        check(status["provider"] == "openai", "provider is openai")
        check(status["engine_ready"] is True, "engine_ready is true from the API key alone")
        check(status["active_model_id"] == "openai:gpt-5.6-terra", "active model id is the cloud id")
        check(status["has_vision"] is True, "GPT-5.6 Terra reports native vision")

        r = await client.get("/api/openai/status")
        cloud_status = r.json()
        check(cloud_status["has_key"] is True, "openai status reports a saved key")
        check(cloud_status["key_source"] == "file", "key source is the local key file")
        check(cloud_status["key_hint"] and cloud_status["key_hint"] != "sk-mock-valid", "key hint is masked")
        check(cloud_status["base_url"] == os.environ["OPENAI_BASE_URL"], "custom base URL is honored")
        check(
            any(m["id"] == "gpt-mock-terra" for m in cloud_status["models"]),
            "live models appear in the ChatGPT model list",
        )

        print("\n[5] Model catalog includes local + ChatGPT models")
        r = await client.get("/api/models")
        models = r.json()
        ids = [m["id"] for m in models]
        check("qwen" in ids or "dolphin" in ids, "local GGUF models still listed")
        check("openai:gpt-5.6-terra" in ids, "catalog ChatGPT model listed")
        check("openai:gpt-mock-terra" in ids, "live models fetched from the API are merged")
        active = [m for m in models if m["is_active"]]
        check(len(active) == 1 and active[0]["provider"] == "openai", "exactly one active model (cloud)")
        cloud = [m for m in models if m["provider"] == "openai"]
        check(all(not m["locked"] for m in cloud), "cloud models unlocked with a key present")

        print("\n[6] Switching to a specific ChatGPT model")
        r = await client.post("/api/models/select", json={"model_id": "openai:gpt-mock-terra", "restart": False})
        check(r.status_code == 200, "cloud model selection accepted")
        check(r.json()["provider"] == "openai", "selection reports openai provider")

        print("\n[7] Streaming chat through the OpenAI client end-to-end")
        r = await client.post("/api/sessions", json={"title": "OpenAI Test", "mode": "classic"})
        session_id = r.json()["id"]
        tokens, done_payload, errors, telemetry = [], None, [], None
        async with client.stream(
            "POST", "/api/chat/stream",
            json={"session_id": session_id, "message": "Say hello", "mode": "classic"}
        ) as resp:
            check(resp.status_code == 200, "chat stream opened")
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if ev["type"] == "telemetry":
                    telemetry = ev
                elif ev["type"] == "token":
                    tokens.append(ev["token"])
                elif ev["type"] == "error":
                    errors.append(ev["message"])
                elif ev["type"] == "done":
                    done_payload = ev
        full = "".join(tokens)
        check(not errors, f"no stream errors (got: {errors})")
        check("Hello from the mock ChatGPT." in full, "mock completion streamed token-by-token")
        check(done_payload is not None and "Hello" in done_payload["full_response"], "done event carries full response")
        check(telemetry is not None and telemetry["budget"]["total_estimated_tokens"] > 0, "memory budget telemetry emitted")

        print("\n[8] Persistence + 4-tier memory maintenance")
        r = await client.get(f"/api/sessions/{session_id}/messages")
        msgs = r.json()
        check(len(msgs) == 2, "user + assistant messages persisted")
        check(msgs[0]["role"] == "user" and msgs[1]["role"] == "assistant", "roles correct")
        check(msgs[1]["content"].startswith("Hello from the mock"), "assistant content persisted")

        print("\n[9] Local engine controls are no-ops in cloud mode")
        r = await client.post("/api/server/start", json={})
        check(r.json()["status"] == "info", "server start is a no-op in cloud mode")
        r = await client.post("/api/server/stop")
        check(r.json()["status"] == "info", "server stop is a no-op in cloud mode")

        print("\n[10] Removing the saved key reverts to the local provider")
        r = await client.delete("/api/openai/key")
        data = r.json()
        check(data["has_key"] is False, "key removed")
        check(data["provider"] == "local", "provider reverted to local")
        r = await client.get("/api/models")
        lock_states = {m["id"]: m["locked"] for m in r.json() if m["provider"] == "openai"}
        check(all(lock_states.values()), "cloud models are locked again without a key")
        r = await client.get("/api/openai/status")
        check(r.json()["has_key"] is False, "openai status reflects the removed key")

    print("\n[11] Reasoning-model auto-sanitize retry (client level)")
    from engine import OpenAIClient

    oc = OpenAIClient(base_url=os.environ["OPENAI_BASE_URL"], api_key="sk-mock-valid")
    got, errors = [], []
    async for chunk in oc.stream_completion(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-mock-reasoning",
        temperature=0.5,
        top_p=0.9,
        presence_penalty=0.2,
        frequency_penalty=0.1,
    ):
        if "error" in chunk:
            errors.append(chunk["error"])
        else:
            got.append(chunk.get("token", ""))
    check(not errors, f"sanitize retry recovered from parameter rejection (errors: {errors})")
    check("Hello" in "".join(got), "reasoning-model stream produced content")

    print("\n[12] Missing-key client guard")
    oc_no_key = OpenAIClient(base_url=os.environ["OPENAI_BASE_URL"], api_key="")
    events = [ev async for ev in oc_no_key.stream_completion(prompt="hi", model="gpt-mock-terra")]
    check(events and "error" in events[0] and events[0]["done"], "client yields a friendly missing-key error")

    print("\n==============================================")
    print("  ALL OPENAI INTEGRATION TESTS PASSED ✔")
    print("==============================================")


if __name__ == "__main__":
    asyncio.run(main())
