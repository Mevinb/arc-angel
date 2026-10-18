#!/usr/bin/env bash
# GROK // Convenience entry point — just run:  ./start.sh
# All real logic lives in scripts/start.sh (Python detection, provider check,
# llama-server bootstrap, web backend launch). This wrapper only forwards args.
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$DIR/scripts/start.sh" "$@"
