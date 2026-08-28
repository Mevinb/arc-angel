#!/usr/bin/env bash
#
# start.sh — launch OmniRoute (if not already running) and JARVIS together.
#
#   ./start.sh                start OmniRoute then launch JARVIS chat
#   ./start.sh doctor         start OmniRoute then run `jarvis doctor`
#   ./start.sh ask "..."      start OmniRoute then run a one-shot question
#   ./start.sh --no-llm       launch JARVIS without starting OmniRoute
#
# OmniRoute runs in the background (detached) so chat stays in the foreground
# in this terminal. It binds to http://127.0.0.1:20128/v1, which JARVIS reads
# from .env / config.yaml.
#
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Locate the omniroute binary: $OMNIROUTE_BIN > PATH > nvm install dir.
if [[ -z "${OMNIROUTE_BIN:-}" ]]; then
    OMNIROUTE_BIN="$(command -v omniroute || true)"
fi
if [[ -z "${OMNIROUTE_BIN:-}" ]]; then
    # nvm binaries are often not on PATH in non-interactive shells.
    for candidate in "$HOME"/.nvm/versions/node/*/bin/omniroute; do
        if [[ -x "$candidate" ]]; then
            OMNIROUTE_BIN="$candidate"
            break
        fi
    done
fi
JARVIS_BIN="${JARVIS_BIN:-$PROJECT_DIR/.venv/bin/jarvis}"
OMNIROUTE_URL="http://127.0.0.1:20128"
OMNIROUTE_LOG="${OMNIROUTE_LOG:-$PROJECT_DIR/data/omniroute.log}"
OMNIROUTE_PID_FILE="${OMNIROUTE_PID_FILE:-$PROJECT_DIR/data/omniroute.pid}"

# ---------------------------------------------------------------------------
# Skip LLM handling if requested
# ---------------------------------------------------------------------------
SKIP_LLM=0
for arg in "$@"; do
    if [[ "$arg" == "--no-llm" ]]; then
        SKIP_LLM=1
    fi
done

# ---------------------------------------------------------------------------
# Health check: is OmniRoute already serving?
# ---------------------------------------------------------------------------
omniroute_is_up() {
    curl -fsS -m 3 "$OMNIROUTE_URL/v1/models" >/dev/null 2>&1
}

wait_for_omniroute() {
    local n="${1:-30}"
    for _ in $(seq 1 "$n"); do
        if omniroute_is_up; then
            return 0
        fi
        sleep 1
    done
    return 1
}

start_omniroute() {
    if omniroute_is_up; then
        echo "OmniRoute already running on $OMNIROUTE_URL"
        return 0
    fi

    if ! command -v "$OMNIROUTE_BIN" >/dev/null 2>&1 && [ ! -x "$OMNIROUTE_BIN" ]; then
        echo "ERROR: OmniRoute not found at $OMNIROUTE_BIN" >&2
        echo "Install it with:  npm install -g omniroute" >&2
        return 1
    fi

    mkdir -p "$(dirname "$OMNIROUTE_LOG")"

    echo "Starting OmniRoute on $OMNIROUTE_URL ..."
    # Fully detach: new session (setsid), stdio to log file. The --no-recovery
    # and --no-tray flags avoid a headless crash-loop.
    setsid "$OMNIROUTE_BIN" serve --no-open --no-tray --no-recovery \
        --port 20128 < /dev/null >> "$OMNIROUTE_LOG" 2>&1 &
    echo $! > "$OMNIROUTE_PID_FILE"
    disown 2>/dev/null || true

    if wait_for_omniroute 30; then
        echo "OmniRoute is up: $OMNIROUTE_URL/v1"
        return 0
    else
        echo "ERROR: OmniRoute did not become ready within 30s — see $OMNIROUTE_LOG" >&2
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Launch JARVIS
# ---------------------------------------------------------------------------
launch_jarvis() {
    if [ ! -x "$JARVIS_BIN" ]; then
        echo "ERROR: JARVIS not found at $JARVIS_BIN" >&2
        echo "Activate the venv / install with:  pip install -e ." >&2
        return 1
    fi
    # Pass all args except our --no-llm flag through to jarvis CLI.
    exec "$JARVIS_BIN" "$@"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
ARGS=()
for arg in "$@"; do
    [[ "$arg" == "--no-llm" ]] || ARGS+=("$arg")
done

if [[ "$SKIP_LLM" -eq 1 ]]; then
    echo "Skipping OmniRoute (--no-llm)."
else
    start_omniroute
fi

launch_jarvis "${ARGS[@]}"
