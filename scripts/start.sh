#!/usr/bin/env bash
set -e

# Base directory
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "=========================================================="
echo "           GROK // UNCENSORED LOCAL CHATBOT               "
echo "=========================================================="

# 0. Pick a Python interpreter that actually has the dependencies
#    Priority: PYTHON_BIN env > project .venv > system pythons > auto-provision
PYTHON_BIN="${PYTHON_BIN:-}"

# Project venv (created with --system-site-packages so the system PyGObject/GNOME
# desktop automation and system Pillow are visible to the app)
if [ -z "$PYTHON_BIN" ] && [ -x "$DIR/.venv/bin/python" ] && "$DIR/.venv/bin/python" -c "import fastapi, uvicorn, httpx, cryptography" >/dev/null 2>&1; then
    PYTHON_BIN="$DIR/.venv/bin/python"
fi

if [ -z "$PYTHON_BIN" ]; then
    for cand in python3.12 python3.13 python3; do
        if command -v "$cand" >/dev/null 2>&1 && "$cand" -c "import fastapi, uvicorn, httpx, cryptography" >/dev/null 2>&1; then
            PYTHON_BIN="$cand"
            break
        fi
    done
fi

# Auto-provision the venv with uv when no usable interpreter was found
if [ -z "$PYTHON_BIN" ] && command -v uv >/dev/null 2>&1; then
    echo "[*] No Python with fastapi/uvicorn/httpx/cryptography found — provisioning .venv with uv..."
    if uv venv --python python3 --system-site-packages "$DIR/.venv" >/dev/null 2>&1 \
        && uv pip install --python "$DIR/.venv/bin/python" --quiet fastapi uvicorn httpx cryptography pydantic pillow >/dev/null 2>&1; then
        PYTHON_BIN="$DIR/.venv/bin/python"
    fi
fi

if [ -z "$PYTHON_BIN" ]; then
    echo "[-] Error: no Python with fastapi/uvicorn/httpx/cryptography found."
    echo "    Option A (recommended, also enables the Desktop Pilot):"
    echo "        uv venv --system-site-packages .venv && uv pip install --python .venv/bin/python fastapi uvicorn httpx cryptography"
    echo "    Option B: use a Python 3.12/3.13 (pip install --user -r requirements.txt)"
    exit 1
fi
echo "[+] Using Python: $PYTHON_BIN"

# Desktop Pilot capability check (PyGObject comes from the system, not pip)
if "$PYTHON_BIN" -c "import gi" >/dev/null 2>&1; then
    echo "[+] Desktop Pilot ready (PyGObject available)."
else
    echo "[!] Desktop Pilot unavailable: PyGObject (gi) missing for this interpreter."
    echo "    Enable it with:  uv venv --system-site-packages .venv && uv pip install --python .venv/bin/python fastapi uvicorn httpx cryptography"
fi

# 1. Determine active provider (local llama.cpp vs OpenAI ChatGPT API)
PROVIDER="$($PYTHON_BIN -c "from config import get_active_provider; print(get_active_provider())" 2>/dev/null || echo "local")"
HAS_OPENAI_KEY="$($PYTHON_BIN -c "from config import get_openai_api_key; print('yes' if get_openai_api_key() else 'no')" 2>/dev/null || echo "no")"
echo "[*] Active provider: $PROVIDER   (OpenAI key configured: $HAS_OPENAI_KEY)"

# 2. Local engine checks (only when the local provider is active)
LLAMA_SERVER="${LLAMA_SERVER_BIN:-/home/mevlec/Downloads/llama.cpp/build/bin/llama-server}"
if [ "$PROVIDER" = "local" ]; then
    if [ ! -f "$LLAMA_SERVER" ]; then
        echo "[!] Warning: llama-server binary not found at $LLAMA_SERVER"
        echo "    Local GGUF models are unavailable, but ChatGPT API mode still works:"
        echo "    add your key in the web UI (🔑) or export OPENAI_API_KEY=sk-..."
    else
        echo "[+] llama-server binary found."
        FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo "0")
        echo "[*] Free GPU VRAM: ${FREE_VRAM} MiB"

        mkdir -p data
        if ! pgrep -f "llama-server" >/dev/null 2>&1; then
            echo "[*] Initializing LLM Engine via LlamaProcessManager..."
            "$PYTHON_BIN" -c "from engine.process_manager import LlamaProcessManager; LlamaProcessManager.start()"
            echo "[+] Engine start requested."
        else
            echo "[+] llama-server is already running."
        fi
    fi
else
    echo "[+] ChatGPT cloud mode active — skipping local llama-server startup."
    if [ "$HAS_OPENAI_KEY" = "no" ]; then
        echo "[!] No OpenAI API key found. Add it in the web UI (🔑 button) or export OPENAI_API_KEY=sk-..."
    fi
fi

# 3. Start Web Application & Manage Lifetime
PORT="${WEB_PORT:-8000}"

cleanup_port() {
    local target_port="$1"
    local pids=""
    if command -v lsof >/dev/null 2>&1; then
        pids=$(lsof -ti tcp:"$target_port" -sTCP:LISTEN 2>/dev/null || true)
    elif command -v fuser >/dev/null 2>&1; then
        pids=$(fuser "$target_port/tcp" 2>/dev/null || true)
    fi

    if [ -n "$pids" ]; then
        echo "[*] Port $target_port is currently held by listening process(es): $pids"
        echo "[*] Closing existing process on port $target_port..."
        kill -15 $pids 2>/dev/null || true
        for _ in {1..20}; do
            local remaining=""
            if command -v lsof >/dev/null 2>&1; then
                remaining=$(lsof -ti tcp:"$target_port" -sTCP:LISTEN 2>/dev/null || true)
            elif command -v fuser >/dev/null 2>&1; then
                remaining=$(fuser "$target_port/tcp" 2>/dev/null || true)
            fi
            if [ -z "$remaining" ]; then
                break
            fi
            sleep 0.1
        done
        local still_running=""
        if command -v lsof >/dev/null 2>&1; then
            still_running=$(lsof -ti tcp:"$target_port" -sTCP:LISTEN 2>/dev/null || true)
        elif command -v fuser >/dev/null 2>&1; then
            still_running=$(fuser "$target_port/tcp" 2>/dev/null || true)
        fi
        if [ -n "$still_running" ]; then
            echo "[!] Force-killing remaining processes on port $target_port..."
            kill -9 $still_running 2>/dev/null || true
            sleep 0.2
        fi
        echo "[+] Port $target_port successfully cleared."
    fi
}

# Clean any existing process occupying the port before launching
cleanup_port "$PORT"

SERVER_PID=""

shutdown_handler() {
    trap - EXIT INT TERM HUP
    echo ""
    echo "[*] Terminal closed or stop requested. Shutting down Grok..."
    if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
        # Terminate child workers spawned by uvicorn reload
        pkill -P "$SERVER_PID" 2>/dev/null || true
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        sleep 0.5
        kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
    # Guarantee listening port release without touching clients
    local leftover=""
    if command -v lsof >/dev/null 2>&1; then
        leftover=$(lsof -ti tcp:"$PORT" -sTCP:LISTEN 2>/dev/null || true)
    elif command -v fuser >/dev/null 2>&1; then
        leftover=$(fuser "$PORT/tcp" 2>/dev/null || true)
    fi
    if [ -n "$leftover" ]; then
        kill -9 $leftover 2>/dev/null || true
    fi
    exit 0
}

# Catch SIGHUP (terminal closed), SIGINT (Ctrl+C), SIGTERM, and EXIT
trap shutdown_handler EXIT INT TERM HUP

echo "[+] Starting Web Backend on http://localhost:$PORT ..."
"$PYTHON_BIN" server.py &
SERVER_PID=$!
wait "$SERVER_PID"
