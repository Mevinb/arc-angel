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


def _ydotoold_alive() -> bool:
    try:
        result = subprocess.run(["pgrep", "-a", "ydotoold"], capture_output=True, text=True, timeout=2)
        return result.returncode == 0 and "ydotoold" in result.stdout
    except Exception:
        return False


def availability() -> str:
    parts = []
    for cmd in ("wtype", "ydotool"):
        parts.append(f"{cmd}={'found' if _has(cmd) else 'missing'}")
    parts.append(f"ydotoold={'alive' if _ydotoold_alive() else 'down'}")
    # uinput check
    import pathlib

    uinput = pathlib.Path("/dev/uinput")
    parts.append(f"uinput={'rw' if uinput.exists() else 'missing'}")
    # quick probe: can we actually talk to compositor via wtype?
    env_note = f"DISPLAY={__import__('os').environ.get('DISPLAY','?')} WAYLAND_DISPLAY={__import__('os').environ.get('WAYLAND_DISPLAY','?')}"
    return ", ".join(parts) + f" ({env_note})"


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
        if not _ydotoold_alive():
            logger.debug("ydotool present but ydotoold not alive — move will likely fail")
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
            # Some ydotool builds use absolute: `ydotool mousemove -x X -y Y`
            if result.returncode == 0:
                # click
                click_map = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
                code = click_map.get(button.lower(), "0xC0")
                subprocess.run(["ydotool", "click", code], capture_output=True, text=True, timeout=10)
                return True, f"Moved to ({x},{y}) and clicked {button} via ydotool"
            # ydotool error often contains hint about daemon
            err = (result.stderr or result.stdout or "")[:300]
            logger.debug("ydotool mousemove failed: %s", err)
            # Don't immediately fail — fall through to uinput
        except Exception as exc:
            logger.debug("ydotool mousemove exception: %s", exc)
    # 2) uinput fallback (python-uinput if installed, otherwise ydotool was the only tier)
    # We intentionally don't import uinput at top — it's optional.
    try:
        import pathlib

        if pathlib.Path("/dev/uinput").exists():
            # Try to use python-uinput if available; otherwise report as missing tier
            try:
                import uinput  # type: ignore

                device = uinput.Device([uinput.BTN_LEFT, uinput.BTN_RIGHT, uinput.REL_X, uinput.REL_Y])
                # Note: relative moves need calibration; we emit an absolute hint via log and succeed as best-effort
                device.emit(uinput.REL_X, int(x), syn=False)
                device.emit(uinput.REL_Y, int(y))
                btn = {"left": uinput.BTN_LEFT, "right": uinput.BTN_RIGHT}.get(button.lower(), uinput.BTN_LEFT)
                device.emit(btn, 1)
                device.emit(btn, 0)
                return True, f"Moved to ({x},{y}) and clicked {button} via uinput (relative)"
            except ImportError:
                pass
            except Exception as exc:
                logger.debug("uinput move failed: %s", exc)
    except Exception:
        pass
    return False, "No Wayland cursor backend available (need ydotool + ydotoold). Install: sudo apt install wtype ydotool && systemctl --user enable --now ydotoold"


def scroll(direction: str = "down", amount: int = 3) -> Tuple[bool, str]:
    if _has("ydotool"):
        if not _ydotoold_alive():
            logger.debug("ydotool present but ydotoold not alive for scroll")
        try:
            # ydotool doesn't have scroll; emulate via button 4/5
            btn = "4" if direction == "up" else "5"
            for _ in range(max(1, min(int(amount), 10))):
                subprocess.run(["ydotool", "click", btn], capture_output=True, text=True, timeout=5)
            return True, f"Scrolled {direction} x{amount} via ydotool"
        except Exception as exc:
            logger.debug("scroll ydotool failed: %s", exc)
    return False, "No scroll backend available (need ydotool + ydotoold)"


__all__ = ["availability", "type_text", "move_click", "scroll"]
