"""
Offline mock of the OpenAI Chat Completions API for testing GROK's ChatGPT mode.

Simulates:
- Auth validation (only "sk-mock-valid" is accepted)
- GET /v1/models
- POST /v1/chat/completions with SSE streaming, usage accounting and [DONE]
- Strict parameter rejection for the "gpt-mock-reasoning" model (exercises the
  client's auto-sanitize retry path)

Run standalone:
    python3.12 scripts/mock_openai_server.py            # listens on 127.0.0.1:8123
"""
import asyncio
import json

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Mock OpenAI API")

VALID_KEY = "sk-mock-valid"

MOCK_MODELS = ["gpt-mock-terra", "gpt-mock-mini", "gpt-mock-reasoning"]

REPLY_CHUNKS = ["Hello", " from", " the", " mock", " ChatGPT."]


def _auth_error():
    return JSONResponse(
        {
            "error": {
                "message": "Incorrect API key provided: you must provide a valid API key.",
                "type": "invalid_request_error",
                "code": "invalid_api_key",
            }
        },
        status_code=401,
    )


def _check_auth(request: Request):
    return request.headers.get("authorization", "") == f"Bearer {VALID_KEY}"


@app.get("/v1/models")
async def list_models(request: Request):
    if not _check_auth(request):
        return _auth_error()
    return {
        "object": "list",
        "data": [{"id": mid, "object": "model", "owned_by": "mock"} for mid in MOCK_MODELS],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if not _check_auth(request):
        return _auth_error()

    body = await request.json()
    model = body.get("model", "")

    if "reasoning" in model:
        for bad in ("temperature", "top_p", "presence_penalty", "frequency_penalty"):
            if bad in body:
                return JSONResponse(
                    {
                        "error": {
                            "message": f"Unsupported parameter: '{bad}' is not supported with this model.",
                            "type": "invalid_request_error",
                            "param": bad,
                            "code": "unsupported_parameter",
                        }
                    },
                    status_code=400,
                )

    if not body.get("stream", False):
        return {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "".join(REPLY_CHUNKS)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": len(REPLY_CHUNKS), "total_tokens": 15},
        }

    async def event_stream():
        for part in REPLY_CHUNKS:
            chunk = {
                "id": "chatcmpl-mock",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {"content": part}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.005)

        final = {
            "id": "chatcmpl-mock",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": len(REPLY_CHUNKS), "total_tokens": 15},
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8123, log_level="warning")
