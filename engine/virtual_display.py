"""
Isolated Virtual Display Controller (:99) using Xvfb, Openbox, and xdotool.
Runs Desktop Pilot in an isolated virtual screen without moving the user's
physical cursor or covering host desktop windows.
"""
from __future__ import annotations

import os
import re
import time
import shutil
import urllib.parse
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable
from engine.wayland_controller import WaylandController, copy_to_clipboard, read_clipboard

DEFAULT_VIRTUAL_DISPLAY = os.getenv("PILOT_VIRTUAL_DISPLAY", ":99")
DEFAULT_RESOLUTION = "1920x1080x24"


class VirtualDisplayManager:
    _instance: Optional["VirtualDisplayManager"] = None

    def __init__(self, display: str = DEFAULT_VIRTUAL_DISPLAY, resolution: str = DEFAULT_RESOLUTION):
        self.display = display
        self.resolution = resolution
        self.xvfb_proc: Optional[subprocess.Popen] = None
        self.wm_proc: Optional[subprocess.Popen] = None
        self._is_running = False

    @classmethod
    def get_instance(cls) -> "VirtualDisplayManager":
        if cls._instance is None:
            cls._instance = VirtualDisplayManager()
        return cls._instance

    @staticmethod
    def is_available() -> Tuple[bool, str]:
        """
        Checks whether all system dependencies required for Isolated Virtual Display mode
        are installed (Xvfb, xdotool, and an image capture tool).
        """
        xvfb = shutil.which("Xvfb")
        xdotool = shutil.which("xdotool")
        capture = shutil.which("import") or shutil.which("scrot") or shutil.which("ffmpeg")

        if not xvfb:
            return False, "Xvfb is not installed. Run: sudo apt install -y xvfb xdotool openbox"
        if not xdotool:
            return False, "xdotool is not installed. Run: sudo apt install -y xvfb xdotool openbox"
        if not capture:
            return False, "No capture tool found (imagemagick or scrot). Run: sudo apt install -y imagemagick scrot"

        return True, "Isolated Virtual Display dependencies are satisfied."

    def ensure_running(self) -> bool:
        """Starts Xvfb virtual display and window manager if not already running."""
        if self._is_running and self.xvfb_proc and self.xvfb_proc.poll() is None:
            return True

        avail, reason = self.is_available()
        if not avail:
            raise RuntimeError(reason)

        display_num = self.display.lstrip(":")
        sock_path = Path(f"/tmp/.X11-unix/X{display_num}")

        # If a previous stale socket or process remains, clean it up
        if sock_path.exists():
            try:
                subprocess.run(["pkill", "-f", f"Xvfb {self.display}"], timeout=2, check=False)
                time.sleep(0.3)
                if sock_path.exists():
                    sock_path.unlink()
            except Exception:
                pass

        # Spawn Xvfb
        xvfb_cmd = [
            "Xvfb", self.display,
            "-screen", "0", self.resolution,
            "-ac", "+extension", "GLX", "+render", "-noreset"
        ]
        self.xvfb_proc = subprocess.Popen(
            xvfb_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )

        # Wait for display server to come online
        for _ in range(20):
            time.sleep(0.1)
            if sock_path.exists():
                break

        # Launch Openbox / Fluxbox window manager if installed
        wm_bin = shutil.which("openbox") or shutil.which("fluxbox") or shutil.which("twm")
        if wm_bin:
            env = {**os.environ, "DISPLAY": self.display}
            cmd = [wm_bin]
            if "openbox" in wm_bin:
                cmd.append("--sm-disable")
            self.wm_proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
            time.sleep(0.5)

        self._is_running = True
        return True

    def stop(self) -> None:
        """Stops the virtual display server and window manager."""
        if self.wm_proc:
            try:
                self.wm_proc.terminate()
            except Exception:
                pass
            self.wm_proc = None

        if self.xvfb_proc:
            try:
                self.xvfb_proc.terminate()
            except Exception:
                pass
            self.xvfb_proc = None

        self._is_running = False

    def get_desktop_bounds(self) -> Tuple[int, int, int, int]:
        """Returns bounds for the virtual display resolution."""
        w, h = 1920, 1080
        parts = self.resolution.split("x")
        if len(parts) >= 2:
            try:
                w = int(parts[0])
                h = int(parts[1])
            except Exception:
                pass
        return (0, 0, w, h)

    def capture_screen(self, out_path: Optional[str] = None, include_cursor: bool = True) -> str:
        """Captures a screenshot of the virtual display root window."""
        self.ensure_running()
        if not out_path:
            shots_dir = Path("/tmp/grok_shots")
            shots_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(shots_dir / f"shot_{int(time.time() * 1000)}.png")

        # Try imagemagick import first
        if shutil.which("import"):
            cmd = ["import", "-window", "root", "-display", self.display, out_path]
            res = subprocess.run(cmd, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(out_path):
                return out_path

        # Try scrot
        if shutil.which("scrot"):
            env = {**os.environ, "DISPLAY": self.display}
            cmd = ["scrot", "-z", out_path]
            res = subprocess.run(cmd, env=env, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(out_path):
                return out_path

        # Try ffmpeg
        if shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "x11grab", "-video_size", "1920x1080",
                "-i", f"{self.display}.0", "-vframes", "1", out_path
            ]
            res = subprocess.run(cmd, timeout=5, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if res.returncode == 0 and os.path.exists(out_path):
                return out_path

        raise RuntimeError(f"Failed to capture virtual display frame on {self.display}")

    def click(self, x: int, y: int, button: int = 1) -> None:
        """Clicks at coordinate (x, y) on the virtual display using xdotool."""
        self.ensure_running()
        subprocess.run(
            ["xdotool", "mousemove", "--display", self.display, str(x), str(y), "click", str(button)],
            timeout=3,
            check=False
        )

    def move_cursor(self, x: int, y: int) -> None:
        """Moves cursor to coordinate (x, y) on the virtual display."""
        self.ensure_running()
        subprocess.run(
            ["xdotool", "mousemove", "--display", self.display, str(x), str(y)],
            timeout=3,
            check=False
        )

    def press_key(self, combo: str) -> None:
        """Presses key or combination on the virtual display."""
        self.ensure_running()
        # Translate keys if necessary
        key_map = {"super": "Super_L", "enter": "Return", "esc": "Escape"}
        parts = [key_map.get(p.strip().lower(), p.strip()) for p in combo.split("+")]
        combo_clean = "+".join(parts)
        subprocess.run(
            ["xdotool", "key", "--display", self.display, combo_clean],
            timeout=3,
            check=False
        )

    def paste_text(self, text: str) -> None:
        """Copies text and sends Ctrl+V on the virtual display."""
        self.ensure_running()
        copy_to_clipboard(text)
        time.sleep(0.05)
        self.press_key("ctrl+v")
        time.sleep(0.05)

    def type_text(self, text: str, delay: float = 0.012, auto_paste_threshold: int = 15) -> None:
        """Types text using xdotool, switching to paste_text for strings longer than threshold."""
        self.ensure_running()
        text_str = str(text)
        if len(text_str) > auto_paste_threshold:
            self.paste_text(text_str)
            return

        subprocess.run(
            ["xdotool", "type", "--display", self.display, "--delay", "12", text_str],
            timeout=5,
            check=False
        )

    def read_clipboard(self) -> str:
        """Reads system clipboard."""
        return read_clipboard()

    def open_url(self, url_or_query: str, browser: str = "brave", profile: str = "Default") -> None:
        """Opens URL or Google search query in an isolated browser window inside the virtual display."""
        self.ensure_running()
        target = url_or_query.strip()
        is_url = bool(re.match(r"^(https?://|www\.|file:///)", target, re.IGNORECASE))
        if not is_url and "." not in target.split("/")[0]:
            target = f"https://www.google.com/search?q={urllib.parse.quote_plus(target)}"
        elif target.startswith("www."):
            target = f"https://{target}"

        env = {
            **os.environ,
            "DISPLAY": self.display,
        }

        # Dedicated sandbox browser profile inside /tmp/grok_sandbox_browser
        sandbox_profile_dir = "/tmp/grok_sandbox_browser"
        flags = f"--ozone-platform=x11 --user-data-dir='{sandbox_profile_dir}' --no-first-run --new-window"

        browser_lower = browser.lower()
        if "chrome" in browser_lower:
            cmd = f"google-chrome {flags} \"{target}\""
        else:
            cmd = (
                f"brave-browser {flags} \"{target}\" || "
                f"/snap/bin/brave {flags} \"{target}\" || "
                f"brave {flags} \"{target}\""
            )

        subprocess.Popen(
            cmd,
            shell=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        time.sleep(2.5)

    def launch_app(self, command_or_app: str, wait_seconds: float = 2.5) -> None:
        """Launches an application inside the virtual display."""
        self.ensure_running()
        cmd = WaylandController.resolve_app_command(command_or_app)
        env = {**os.environ, "DISPLAY": self.display}
        subprocess.Popen(
            cmd,
            shell=True,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def tile_active_window(self, side: str = "right") -> None:
        """In virtual display, window managers handle sizing natively."""
        pass

    # Delegate OCR and shake detection to established WaylandController routines
    get_screen_ocr_elements = staticmethod(WaylandController.get_screen_ocr_elements)
    prioritize_ocr_elements = classmethod(WaylandController.prioritize_ocr_elements)

    def start_shake_detector(self, callback: Callable[[], None]) -> None:
        WaylandController.get_instance().start_shake_detector(callback)

    def stop_shake_detector(self) -> None:
        WaylandController.get_instance().stop_shake_detector()
