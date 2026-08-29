#!/usr/bin/env bash
#
# start.sh — launch OmniRoute (if not already running) and ARC together.
#
#   ./start.sh                start OmniRoute then launch ARC chat
#   ./start.sh doctor         start OmniRoute then run `arc doctor`
#   ./start.sh ask "..."      start OmniRoute then run a one-shot question
#   ./start.sh --no-llm       launch ARC without starting OmniRoute
#   ./start.sh stop           stop running OmniRoute gateway
#
# OmniRoute runs in the background while ARC runs in the foreground.
# When ARC exits or the session is interrupted (e.g. Ctrl+C), OmniRoute and any
# background processes spawned by this script are cleanly terminated.
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

if [[ -n "${OMNIROUTE_BIN:-}" && -x "${OMNIROUTE_BIN:-}" ]]; then
    export PATH="$(dirname "$OMNIROUTE_BIN"):$PATH"
fi

ARC_BIN="${ARC_BIN:-$PROJECT_DIR/.venv/bin/arc}"
OMNIROUTE_PORT="${OMNIROUTE_PORT:-20128}"
OMNIROUTE_URL="${OMNIROUTE_URL:-http://127.0.0.1:${OMNIROUTE_PORT}}"
OMNIROUTE_LOG="${OMNIROUTE_LOG:-$PROJECT_DIR/data/omniroute.log}"
OMNIROUTE_PID_FILE="${OMNIROUTE_PID_FILE:-$PROJECT_DIR/data/omniroute.pid}"

STARTED_OMNIROUTE=0
OMNIROUTE_PID=""

# ---------------------------------------------------------------------------
# Process cleanup handlers
# ---------------------------------------------------------------------------
stop_omniroute() {
    local pid="${OMNIROUTE_PID:-}"
    if [[ -z "$pid" && -f "$OMNIROUTE_PID_FILE" ]]; then
        pid="$(cat "$OMNIROUTE_PID_FILE" 2>/dev/null || true)"
    fi

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping OmniRoute (PID $pid)..."
        # Send SIGTERM to process group, fallback to PID
        kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true

        local count=0
        while kill -0 "$pid" 2>/dev/null && [ "$count" -lt 30 ]; do
            sleep 0.1
            count=$((count + 1))
        done

        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
    rm -f "$OMNIROUTE_PID_FILE"
}

cleanup() {
    local exit_code=$?
    # Prevent re-entry
    trap - EXIT INT TERM HUP

    # Stop OmniRoute if it was spawned by this script
    if [[ "$STARTED_OMNIROUTE" -eq 1 ]]; then
        stop_omniroute
    fi

    # Terminate any remaining background jobs started by this script
    local bg_pids
    bg_pids="$(jobs -p 2>/dev/null || true)"
    if [[ -n "$bg_pids" ]]; then
        echo "$bg_pids" | xargs -r kill -TERM 2>/dev/null || true
    fi

    exit "$exit_code"
}

trap cleanup EXIT INT TERM HUP

# ---------------------------------------------------------------------------
# Manual stop command: ./start.sh stop
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "stop" || "${1:-}" == "down" ]]; then
    stop_omniroute
    exit 0
fi

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
    # Start OmniRoute in background in its own process group (setsid)
    # Redirect stdio to log file.
    setsid "$OMNIROUTE_BIN" serve --no-open --no-tray --no-recovery \
        --port "$OMNIROUTE_PORT" < /dev/null >> "$OMNIROUTE_LOG" 2>&1 &
    OMNIROUTE_PID=$!
    STARTED_OMNIROUTE=1
    echo "$OMNIROUTE_PID" > "$OMNIROUTE_PID_FILE"

    if wait_for_omniroute 30; then
        echo "OmniRoute is up: $OMNIROUTE_URL/v1"
        return 0
    else
        echo "ERROR: OmniRoute did not become ready within 30s — see $OMNIROUTE_LOG" >&2
        stop_omniroute
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Launch ARC
# ---------------------------------------------------------------------------
launch_arc() {
    if [ ! -x "$ARC_BIN" ]; then
        echo "ERROR: ARC not found at $ARC_BIN" >&2
        echo "Activate the venv / install with:  pip install -e ." >&2
        return 1
    fi
    # Pass all args except our --no-llm flag through to arc CLI.
    # Note: Do not use exec so that bash stays alive to trap EXIT/INT/TERM
    "$ARC_BIN" "$@"
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

ARC_EXIT=0
launch_arc "${ARGS[@]}" || ARC_EXIT=$?
exit "$ARC_EXIT"
