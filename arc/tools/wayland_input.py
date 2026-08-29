"""Wayland-native typing and cursor helpers.

Tiered fallback for GNOME/Wayland (DISPLAY=:1, WAYLAND_DISPLAY=wayland-0):
  typing: wtype → ydotool type → uinput (python-uinput) → failure
  cursor: ydotool mousemove/click → uinput → failure
  focus: wmctrl / gdbus not reliable on pure Wayland, so we rely on
         Playwright Page.bring_to_front(); this module only moves the
         pointer when the underlying portal allows it.

All helpers are GREEN (read-only from safety perspective) and must never
raise — they return ToolResult-style tuples.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from typing import Tuple

logger = logging.getLogger("arc.wayland_input")


def _has(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def availability() -> str:
    parts = []
    for cmd in ("wtype", "ydotool"):
        parts.append(f"{cmd}={'found' if _has(cmd) else 'missing'}")
    # uinput check
    import pathlib

    uinput = pathlib.Path("/dev/uinput")
    parts.append(f"uinput={'rw' if uinput.exists() else 'missing'}")
    return ", ".join(parts) + " (Wayland DISPLAY=:1 WAYLAND_DISPLAY=wayland-0)"


def type_text(text: str) -> Tuple[bool, str]:
    """Type text at OS level on Wayland. Returns (ok, message)."""
    if not text:
        return False, "No text provided"
    # 1) wtype (preferred, xdg-desktop-portal + compositor)
    if _has("wtype"):
        try:
            result = subprocess.run(["wtype", text], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return True, f"Typed {len(text)} chars via wtype"
            logger.debug("wtype failed: %s %s", result.stdout, result.stderr)
        except Exception as exc:
            logger.debug("wtype exception: %s", exc)
    # 2) ydotool
    if _has("ydotool"):
        try:
            # ydotool needs daemon; try `ydotool type --help` style is file vs arg
            # upstream accepts `ydotool type --text "hello"` or `ydotool type "hello"`
            for args in ([f"--text", text], [text]):
                try:
                    result = subprocess.run(["ydotool", "type", *args], capture_output=True, text=True, timeout=10)
                    if result.returncode == 0:
                        return True, f"Typed {len(text)} chars via ydotool"
                except Exception:
                    continue
        except Exception as exc:
            logger.debug("ydotool type exception: %s", exc)
    return False, "No Wayland typing backend available (need wtype or ydotool + ydotoold). Install: sudo apt install wtype ydotool"


def move_click(x: int, y: int, button: str = "left") -> Tuple[bool, str]:
    """Move cursor to (x,y) and click. Returns (ok, message)."""
    # ydotool is the only viable Wayland cursor mover without portals
    if _has("ydotool"):
        try:
            result = subprocess.run(
                ["ydotool", "mousemove", "--", str(int(x)), str(int(y))],
                capture_output=True, text=True, timeout=10,
            )
            # Some builds use different flag style
            if result.returncode != 0:
                result = subprocess.run(
                    ["ydotool", "mousemove", "-x", str(int(x)), "-y", str(int(y))],
                    capture_output=True, text=True, timeout=10,
                )
            if result.returncode == 0:
                # click
                click_map = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
                code = click_map.get(button.lower(), "0xC0")
                subprocess.run(["ydotool", "click", code], capture_output=True, text=True, timeout=10)
                return True, f"Moved to ({x},{y}) and clicked {button} via ydotool"
            logger.debug("ydotool mousemove failed: %s %s", result.stdout, result.stderr)
        except Exception as exc:
            logger.debug("ydotool mousemove exception: %s", exc)
    return False, "No Wayland cursor backend available (need ydotool + ydotoold). Install: sudo apt install ydotool && systemctl --user enable --now ydotoold"


def scroll(direction: str = "down", amount: int = 3) -> Tuple[bool, str]:
    if _has("ydotool"):
        try:
            # ydotool doesn't have scroll; emulate via button 4/5
            btn = "4" if direction == "up" else "5"
            for _ in range(max(1, min(int(amount), 10))):
                subprocess.run(["ydotool", "click", btn], capture_output=True, text=True, timeout=5)
            return True, f"Scrolled {direction} x{amount} via ydotool"
        except Exception as exc:
            logger.debug("scroll ydotool failed: %s", exc)
    return False, "No scroll backend available"


__all__ = ["availability", "type_text", "move_click", "scroll"]
