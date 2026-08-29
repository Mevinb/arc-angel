#!/usr/bin/env bash
# Enable Chrome remote debugging for ARC so it can drive YOUR profile (Library/image-gen stays logged in)
# Usage: ./scripts/enable-chrome-cdp.sh [--port 9222] [--user-data-dir ~/.config/google-chrome]
set -euo pipefail
PORT="${1:-9222}"
# Parse --port / --user-data-dir
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2;;
    --user-data-dir) UDD="$2"; shift 2;;
    *) shift;;
  esac
done
UDD="${UDD:-${ARC_CHROME_USER_DATA_DIR:-$HOME/.config/google-chrome}}"
UDD_EXPANDED="${UDD/#\~/$HOME}"
echo "Chrome profile: $UDD_EXPANDED"
echo "CDP port: $PORT"
echo ""

# Patch .desktop so future clicks launch debuggable Chrome
DESKTOP_SRC="/usr/share/applications/google-chrome.desktop"
DESKTOP_DST="$HOME/.local/share/applications/google-chrome.desktop"
mkdir -p "$(dirname "$DESKTOP_DST")"
if [[ -f "$DESKTOP_SRC" ]]; then
  cp "$DESKTOP_SRC" "$DESKTOP_DST"
  # Add --remote-debugging-port if not present
  if ! grep -q "remote-debugging-port" "$DESKTOP_DST"; then
    sed -i "s|Exec=/usr/bin/google-chrome-stable|Exec=/usr/bin/google-chrome-stable --remote-debugging-port=$PORT --user-data-dir=$UDD_EXPANDED --ozone-platform-hint=auto|g" "$DESKTOP_DST"
    sed -i "s|Exec=/opt/google/chrome/google-chrome|Exec=/opt/google/chrome/google-chrome --remote-debugging-port=$PORT --user-data-dir=$UDD_EXPANDED --ozone-platform-hint=auto|g" "$DESKTOP_DST"
    echo "Patched $DESKTOP_DST to launch with --remote-debugging-port=$PORT"
  else
    echo "Already patched: $DESKTOP_DST"
  fi
  update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
fi

echo ""
echo "Now close ALL Chrome windows and relaunch:"
echo "  pkill chrome; google-chrome --remote-debugging-port=$PORT --user-data-dir=$UDD_EXPANDED --ozone-platform-hint=auto https://chatgpt.com &"
echo ""
echo "Verify:"
echo "  curl -s http://127.0.0.1:$PORT/json/version | head"
echo "  arc doctor  # should show chrome.profile cdp=:$PORT alive"
echo ""
echo "Or let ARC launch it once (no existing Chrome): it will run:"
echo "  google-chrome --user-data-dir=$UDD_EXPANDED --remote-debugging-port=$PORT --ozone-platform-hint=auto about:blank"
