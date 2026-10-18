"""
Wayland Controller for Mutter GNOME 46.
Native, zero-prompt Wayland desktop automation using:
- org.gnome.Mutter.RemoteDesktop (pixel-exact absolute pointer motions, button clicks, keysym typing)
- org.gnome.Mutter.ScreenCast + PipeWire (zero-prompt full desktop screen captures)
- Subprocess app launcher & window utilities
"""
from __future__ import annotations

import os
import re
import time
import shutil
import select
import struct
import threading
import subprocess
import urllib.parse
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Callable

from .browser_profiles import resolve_browser_and_profile

try:  # PyGObject is only required by the Desktop Pilot, not by chat / ChatGPT mode
    import gi
    gi.require_version("Gio", "2.0")
    from gi.repository import Gio, GLib
    GI_AVAILABLE = True
    _GI_IMPORT_ERROR = ""
except Exception as _gi_exc:  # pragma: no cover - depends on host environment
    Gio = None  # type: ignore[assignment]
    GLib = None  # type: ignore[assignment]
    GI_AVAILABLE = False
    _GI_IMPORT_ERROR = str(_gi_exc)


def require_gi() -> None:
    """Raises a helpful error when desktop automation runs without PyGObject."""
    if not GI_AVAILABLE:
        raise RuntimeError(
            "Desktop Pilot requires PyGObject (python3-gi), which is unavailable for "
            f"this Python interpreter ({_GI_IMPORT_ERROR or 'import gi failed'}). "
            "Install it (e.g. `sudo apt install python3-gi`) or run the app with an "
            "interpreter that provides it. Chat and ChatGPT mode work without it."
        )

RD = "org.gnome.Mutter.RemoteDesktop"
RD_PATH = "/org/gnome/Mutter/RemoteDesktop"
RD_SESSION = f"{RD}.Session"
SC = "org.gnome.Mutter.ScreenCast"
SC_PATH = "/org/gnome/Mutter/ScreenCast"
SC_SESSION = f"{SC}.Session"
DISPLAY = "org.gnome.Mutter.DisplayConfig"
DISPLAY_PATH = "/org/gnome/Mutter/DisplayConfig"

# evdev mouse button codes
BUTTONS = {
    "left": 0x110,
    "right": 0x111,
    "middle": 0x112,
    "back": 0x116,
    "forward": 0x115
}

# X11 keysym mappings for layout-independent key injection
KEYSYMS: Dict[str, int] = {
    "ctrl": 0xFFE3, "control": 0xFFE3, "leftctrl": 0xFFE3, "rightctrl": 0xFFE4,
    "shift": 0xFFE1, "leftshift": 0xFFE1, "rightshift": 0xFFE2,
    "alt": 0xFFE9, "leftalt": 0xFFE9, "rightalt": 0xFFEA,
    "super": 0xFFEB, "meta": 0xFFEB, "win": 0xFFEB,
    "enter": 0xFF0D, "return": 0xFF0D,
    "esc": 0xFF1B, "escape": 0xFF1B,
    "tab": 0xFF09, "backspace": 0xFF08, "delete": 0xFFFF, "insert": 0xFF63,
    "space": 0x0020,
    "up": 0xFF52, "down": 0xFF54, "left": 0xFF51, "right": 0xFF53,
    "home": 0xFF50, "end": 0xFF57, "pageup": 0xFF55, "pagedown": 0xFF56,
}
KEYSYMS.update({f"f{n}": 0xFFBD + n for n in range(1, 13)})
KEYSYMS.update({c: ord(c) for c in "abcdefghijklmnopqrstuvwxyz0123456789"})
KEYSYMS.update({
    "minus": 0x002D, "equal": 0x003D, "comma": 0x002C, "period": 0x002E,
    "slash": 0x002F, "semicolon": 0x003B, "apostrophe": 0x0027,
    "bracketleft": 0x005B, "bracketright": 0x005D, "backslash": 0x005C,
    "grave": 0x0060, "plus": 0x002B,
})

def char_to_keysym(ch: str) -> int:
    """Converts a character to layout-independent X11 keysym."""
    code = ord(ch)
    if code in (0x0A, 0x0D):
        return 0xFF0D  # Return
    if code == 0x09:
        return 0xFF09  # Tab
    if 0x20 <= code <= 0xFF:
        return code
    return code | 0x01000000

def copy_to_clipboard(text: str) -> bool:
    """Copies text to the system clipboard using wl-copy, xclip, or pyperclip."""
    if shutil.which("wl-copy"):
        try:
            res = subprocess.run(["wl-copy"], input=text.encode("utf-8"), timeout=2, check=False)
            if res.returncode == 0:
                return True
        except Exception:
            pass
    if shutil.which("xclip"):
        try:
            res = subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), timeout=2, check=False)
            if res.returncode == 0:
                return True
        except Exception:
            pass
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        return False


def read_clipboard() -> str:
    """Reads current text from system clipboard using wl-paste, xclip, or pyperclip."""
    if shutil.which("wl-paste"):
        try:
            res = subprocess.run(
                ["wl-paste", "--no-newline"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2,
                text=True,
                check=False
            )
            if res.returncode == 0:
                return res.stdout
        except Exception:
            pass
    if shutil.which("xclip"):
        try:
            res = subprocess.run(
                ["xclip", "-selection", "clipboard", "-o"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2,
                text=True,
                check=False
            )
            if res.returncode == 0:
                return res.stdout
        except Exception:
            pass
    try:
        import pyperclip
        return pyperclip.paste() or ""
    except Exception:
        return ""


class WaylandController:
    _instance = None
    _lock = threading.RLock()

    def __init__(self, idle_timeout: float = 45.0):
        self.idle_timeout = idle_timeout
        self._bus: Optional[Gio.DBusConnection] = None
        self._rd_path: Optional[str] = None
        self._sc_path: Optional[str] = None
        self._stream_path: Optional[str] = None
        self._bounds: Optional[Tuple[int, int, int, int]] = None  # (left, top, width, height)
        self._timer: Optional[threading.Timer] = None
        self.last_x: float = 0.0
        self.last_y: float = 0.0
        self._shake_thread: Optional[threading.Thread] = None
        self._shake_stop_event = threading.Event()

    @classmethod
    def get_instance(cls) -> "WaylandController":
        with cls._lock:
            if cls._instance is None:
                cls._instance = WaylandController()
            return cls._instance

    def _get_bus(self) -> Gio.DBusConnection:
        require_gi()
        if self._bus is None:
            self._bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        return self._bus

    def _dbus_call(self, name: str, path: str, iface: str, method: str, args: Any = None, timeout: int = 10000) -> Any:
        bus = self._get_bus()
        return bus.call_sync(name, path, iface, method, args, None, Gio.DBusCallFlags.NONE, timeout, None)

    def get_desktop_bounds(self) -> Tuple[int, int, int, int]:
        """Calculates global desktop logical boundaries across monitors."""
        require_gi()
        if self._bounds:
            return self._bounds

        state = self._dbus_call(DISPLAY, DISPLAY_PATH, DISPLAY, "GetCurrentState").unpack()
        monitors, logical = state[1], state[2]
        current_mode: Dict[str, Tuple[int, int]] = {}
        for monitor in monitors:
            connector = monitor[0][0]
            for mode in monitor[1]:
                if mode[6].get("is-current"):
                    current_mode[connector] = (mode[1], mode[2])
                    break

        left = top = 10**9
        right = bottom = -10**9
        for lm in logical:
            x, y, scale, transform = lm[0], lm[1], lm[2], lm[3]
            width = height = 0
            for spec in lm[5]:
                size = current_mode.get(spec[0])
                if size:
                    width, height = size
                    break
            if transform in (1, 3, 5, 7):
                width, height = height, width
            width = round(width / scale)
            height = round(height / scale)
            left, top = min(left, x), min(top, y)
            right, bottom = max(right, x + width), max(bottom, y + height)

        self._bounds = (left, top, right - left, bottom - top)
        return self._bounds

    def _ensure_session(self) -> Tuple[str, str]:
        """Establishes active Mutter RemoteDesktop session with area recording stream."""
        with self._lock:
            if self._rd_path and self._stream_path:
                self._touch()
                return self._rd_path, self._stream_path

            bounds = self.get_desktop_bounds()

            rd_path = self._dbus_call(RD, RD_PATH, RD, "CreateSession").unpack()[0]
            session_id = self._dbus_call(
                RD, rd_path, "org.freedesktop.DBus.Properties", "Get",
                GLib.Variant("(ss)", (RD_SESSION, "SessionId"))
            ).unpack()[0]

            sc_path = self._dbus_call(
                SC, SC_PATH, SC, "CreateSession",
                GLib.Variant("(a{sv})", ({"remote-desktop-session-id": GLib.Variant("s", session_id)},))
            ).unpack()[0]

            stream = self._dbus_call(
                SC, sc_path, SC_SESSION, "RecordArea",
                GLib.Variant("(iiiia{sv})", (bounds[0], bounds[1], bounds[2], bounds[3], {
                    "cursor-mode": GLib.Variant("u", 0)
                }))
            ).unpack()[0]

            self._dbus_call(RD, rd_path, RD_SESSION, "Start")

            self._rd_path = rd_path
            self._sc_path = sc_path
            self._stream_path = stream
            self._touch()
            return rd_path, stream

    def _touch(self) -> None:
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.idle_timeout, self.stop)
        self._timer.daemon = True
        self._timer.start()

    def stop(self) -> None:
        """Closes RemoteDesktop session gracefully and halts shake detector."""
        self.stop_shake_detector()
        with self._lock:
            if self._timer:
                self._timer.cancel()
                self._timer = None
            if self._rd_path:
                try:
                    self._dbus_call(RD, self._rd_path, RD_SESSION, "Stop")
                except Exception:
                    pass
            self._rd_path = None
            self._sc_path = None
            self._stream_path = None

    def start_shake_detector(self, on_shake_callback: Callable[[], None]) -> None:
        """
        Monitors /dev/input/mice in a background thread for physical mouse shakes.
        If user shakes the physical mouse vigorously (>600px displacement in 0.5s),
        on_shake_callback is immediately invoked to abort desktop automation.
        """
        if not os.path.exists("/dev/input/mice") or not os.access("/dev/input/mice", os.R_OK):
            return

        self.stop_shake_detector()
        self._shake_stop_event.clear()

        def _shake_worker():
            try:
                with open("/dev/input/mice", "rb") as f:
                    accum_dist = 0
                    last_reset = time.time()
                    while not self._shake_stop_event.is_set():
                        r, _, _ = select.select([f], [], [], 0.2)
                        if not r:
                            if time.time() - last_reset > 0.5:
                                accum_dist = 0
                                last_reset = time.time()
                            continue

                        data = f.read(3)
                        if len(data) < 3:
                            continue
                        _, dx, dy = struct.unpack("bbb", data)
                        accum_dist += abs(dx) + abs(dy)
                        now = time.time()
                        if now - last_reset > 0.5:
                            accum_dist = abs(dx) + abs(dy)
                            last_reset = now
                        elif accum_dist > 600:
                            try:
                                on_shake_callback()
                            except Exception:
                                pass
                            break
            except Exception:
                pass

        self._shake_thread = threading.Thread(target=_shake_worker, daemon=True)
        self._shake_thread.start()

    def stop_shake_detector(self) -> None:
        """Stops the background mouse shake detector thread."""
        self._shake_stop_event.set()
        self._shake_thread = None

    def tile_active_window(self, side: str = "right") -> None:
        """Tiles active window to left or right half using Wayland Super+Arrow shortcut."""
        key = "super+right" if side.lower() == "right" else "super+left"
        self.press_key(key)

    # ========================== Pointer Actions ==========================

    def move_cursor(self, x: float, y: float) -> Tuple[float, float]:
        """Moves cursor to absolute desktop coordinates (x, y)."""
        rd_path, stream = self._ensure_session()
        bounds = self.get_desktop_bounds()
        bx, by, bw, bh = bounds
        sx, sy = x - bx, y - by
        sx = max(0.0, min(float(bw), sx))
        sy = max(0.0, min(float(bh), sy))

        self._dbus_call(
            RD, rd_path, RD_SESSION, "NotifyPointerMotionAbsolute",
            GLib.Variant("(sdd)", (stream, sx, sy))
        )
        self.last_x, self.last_y = x, y
        return x, y

    def button(self, name: str = "left", pressed: bool = True) -> None:
        """Sets mouse button state (pressed or released)."""
        code = BUTTONS.get(name.lower(), 0x110)
        rd_path, _ = self._ensure_session()
        self._dbus_call(
            RD, rd_path, RD_SESSION, "NotifyPointerButton",
            GLib.Variant("(ib)", (code, pressed))
        )

    def click(self, x: Optional[float] = None, y: Optional[float] = None, button: str = "left", count: int = 1) -> None:
        """Moves to (x, y) if given and clicks button count times."""
        if x is not None and y is not None:
            self.move_cursor(x, y)
            time.sleep(0.05)

        for i in range(max(1, int(count))):
            if i > 0:
                time.sleep(0.06)
            self.button(button, True)
            time.sleep(0.03)
            self.button(button, False)

    def drag(self, x1: float, y1: float, x2: float, y2: float, button: str = "left", steps: int = 20) -> None:
        """Smoothly drags cursor from (x1, y1) to (x2, y2)."""
        self.move_cursor(x1, y1)
        time.sleep(0.06)
        self.button(button, True)
        time.sleep(0.04)

        for step in range(1, steps + 1):
            curr_x = x1 + (x2 - x1) * (step / steps)
            curr_y = y1 + (y2 - y1) * (step / steps)
            self.move_cursor(curr_x, curr_y)
            time.sleep(0.012)

        time.sleep(0.06)
        self.button(button, False)

    # ========================== Keyboard Actions ==========================

    def keysym(self, sym: int, pressed: bool) -> None:
        """Emits layout-independent X11 keysym down/up event."""
        rd_path, _ = self._ensure_session()
        self._dbus_call(
            RD, rd_path, RD_SESSION, "NotifyKeyboardKeysym",
            GLib.Variant("(ub)", (int(sym), pressed))
        )

    def paste_text(self, text: str) -> None:
        """Copies text to clipboard and triggers Ctrl+V paste."""
        copy_to_clipboard(text)
        time.sleep(0.05)
        self.press_key("ctrl+v")
        time.sleep(0.05)

    def type_text(self, text: str, delay: float = 0.012, auto_paste_threshold: int = 15) -> None:
        """
        Types raw text character by character using layout-safe keysyms.
        If string length exceeds auto_paste_threshold, switches to clipboard paste for speed and stability.
        """
        text_str = str(text)
        if len(text_str) > auto_paste_threshold:
            self.paste_text(text_str)
            return

        for ch in text_str:
            sym = char_to_keysym(ch)
            self.keysym(sym, True)
            time.sleep(0.005)
            self.keysym(sym, False)
            time.sleep(delay)

    def press_key(self, combo: str) -> None:
        """
        Presses a key or key combination (e.g. 'ctrl+t', 'ctrl+l', 'enter', 'alt+tab').
        """
        parts = [p.strip().lower() for p in combo.split("+")]
        syms_to_release: List[int] = []

        try:
            for part in parts:
                sym = KEYSYMS.get(part)
                if sym is None:
                    if len(part) == 1:
                        sym = char_to_keysym(part)
                    else:
                        continue
                self.keysym(sym, True)
                syms_to_release.append(sym)
                time.sleep(0.02)
        finally:
            for sym in reversed(syms_to_release):
                self.keysym(sym, False)
                time.sleep(0.01)

    def type_into(self, x: float, y: float, text: str, submit: bool = True) -> None:
        """
        Combines focusing a field, clearing existing content, typing new text, and optionally submitting.
        1. Clicks (x, y) to establish keyboard focus.
        2. Waits 300ms for input focus state to settle.
        3. Sends Ctrl+A then Backspace to clear placeholder or stale text.
        4. Types/pastes text (uses paste_text clipboard bridge if text > 15 chars).
        5. If submit is True, waits 200ms and presses Enter.
        """
        self.click(x, y)
        time.sleep(0.3)
        self.press_key("ctrl+a")
        time.sleep(0.05)
        self.press_key("backspace")
        time.sleep(0.1)
        self.type_text(text)
        if submit:
            time.sleep(0.2)
            self.press_key("enter")

    def upload_file_to_dialog(self, file_path: str) -> bool:
        """
        When a browser file-picker dialog opens (after clicking an 'Upload' or 'Attach' button),
        sends Ctrl+L to reveal direct path input in GTK/GNOME file chooser, pastes the absolute file_path,
        and presses Enter to attach the file.
        """
        p = Path(file_path).resolve()
        if not p.exists():
            logger.warning("Upload target file does not exist: %s", file_path)
            return False

        abs_path = str(p)
        logger.info("Uploading file to GTK dialog: %s", abs_path)
        self.press_key("ctrl+l")
        time.sleep(0.35)
        self.paste_text(abs_path)
        time.sleep(0.25)
        self.press_key("enter")
        time.sleep(0.5)
        return True

    def wait_for_condition(self, description: str = "", max_wait_sec: int = 45, poll_interval: int = 5) -> bool:
        """
        Polls the system or pauses execution until an asynchronous operation
        (such as an AI image generation or a heavy asset compile) settles,
        avoiding wasting LLM cognitive steps.
        """
        logger.info("Waiting for condition: '%s' (max %ds, poll every %ds)", description, max_wait_sec, poll_interval)
        elapsed = 0
        while elapsed < max_wait_sec:
            time.sleep(poll_interval)
            elapsed += poll_interval
            self._touch()
        return True

    def save_latest_download(self, target_path: str, max_age_seconds: int = 120) -> Optional[str]:
        """
        Finds the newest file downloaded in ~/Downloads in the last max_age_seconds
        and moves it to target_path.
        """
        dl_dir = Path.home() / "Downloads"
        if not dl_dir.exists():
            return None

        now = time.time()
        candidates = []
        for item in dl_dir.iterdir():
            if item.is_file() and not item.name.startswith("."):
                try:
                    mtime = item.stat().st_mtime
                    if now - mtime <= max_age_seconds:
                        candidates.append((mtime, item))
                except Exception:
                    pass

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[0], reverse=True)
        newest = candidates[0][1]
        dest = Path(target_path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(newest), str(dest))
        logger.info("save_latest_download moved '%s' -> '%s'", newest.name, dest)
        return str(dest)

    # ========================== Screen Capture ==========================

    def capture_screen(self, out_path: Optional[str] = None, include_cursor: bool = True) -> str:
        """
        Captures a frame of the monitor directly via Mutter ScreenCast + PipeWire.
        Does NOT trigger user confirmation dialogs or require extensions.
        """
        if not out_path:
            shots_dir = Path("/tmp/grok_shots")
            shots_dir.mkdir(parents=True, exist_ok=True)
            out_path = str(shots_dir / f"shot_{int(time.time() * 1000)}.png")

        conn = self._get_bus()

        # 1. Primary connector
        state = self._dbus_call(DISPLAY, DISPLAY_PATH, DISPLAY, "GetCurrentState").unpack()
        connector = state[1][0][0][0]

        # 2. Dedicated capture screencast session
        sc_session = self._dbus_call(
            SC, SC_PATH, SC, "CreateSession",
            GLib.Variant("(a{sv})", ({},))
        ).unpack()[0]

        opts = {"cursor-mode": GLib.Variant("u", 1 if include_cursor else 0)}
        stream_path = self._dbus_call(
            SC, sc_session, f"{SC}.Session", "RecordMonitor",
            GLib.Variant("(sa{sv})", (connector, opts))
        ).unpack()[0]

        loop = GLib.MainLoop()
        capture_state: Dict[str, Any] = {}

        def on_stream_added(_conn, _sender, _path, _iface, _signal, params):
            capture_state["node_id"] = params[0]
            loop.quit()

        sub = conn.signal_subscribe(
            None, f"{SC}.Stream", "PipeWireStreamAdded", stream_path, None,
            Gio.DBusSignalFlags.NONE, on_stream_added
        )
        GLib.timeout_add_seconds(3, lambda: (loop.quit(), False)[1])

        self._dbus_call(SC, sc_session, f"{SC}.Session", "Start")
        loop.run()
        conn.signal_unsubscribe(sub)

        node_id = capture_state.get("node_id")
        if not node_id:
            try:
                self._dbus_call(SC, sc_session, f"{SC}.Session", "Stop")
            except Exception:
                pass
            raise RuntimeError("Failed to obtain PipeWire stream node from Mutter ScreenCast")

        # 3. Pull 1 frame via gst-launch
        cmd = [
            "gst-launch-1.0", "-q",
            "pipewiresrc", f"path={node_id}", "num-buffers=1",
            "!", "videoconvert",
            "!", "pngenc",
            "!", "filesink", f"location={out_path}"
        ]
        try:
            res = subprocess.run(cmd, timeout=5, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if res.returncode != 0:
                raise RuntimeError(f"GStreamer snapshot error: {res.stderr.decode()}")
        finally:
            try:
                self._dbus_call(SC, sc_session, f"{SC}.Session", "Stop")
            except Exception:
                pass

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(f"Screenshot file {out_path} was not created or is empty")

        return out_path

    # ========================== App & Window Utils ==========================

    @classmethod
    def resolve_app_command(cls, command_or_app: str, profile: Optional[str] = None) -> str:
        """Resolves common browser and application aliases to reliable CLI commands with exact profile directory."""
        app = command_or_app.strip().lower()
        if app in ["brave", "brave-browser"]:
            _, resolved_pdir, _ = resolve_browser_and_profile("brave", profile)
            return (
                f"brave-browser --new-window --profile-directory='{resolved_pdir}' || "
                f"/snap/bin/brave --new-window --profile-directory='{resolved_pdir}' || "
                f"brave --new-window --profile-directory='{resolved_pdir}'"
            )
        elif app in ["chrome", "google-chrome"]:
            bin_cmd, resolved_pdir, _ = resolve_browser_and_profile("chrome", profile)
            return f"{bin_cmd} --new-window --profile-directory='{resolved_pdir}'"
        elif app in ["chromium"]:
            bin_cmd, resolved_pdir, _ = resolve_browser_and_profile("chromium", profile)
            return f"{bin_cmd} --new-window --profile-directory='{resolved_pdir}'"
        elif app in ["files", "nautilus"]:
            return "nautilus"
        elif app in ["terminal", "gnome-terminal"]:
            return "gnome-terminal"
        elif app in ["calculator", "gnome-calculator"]:
            return "gnome-calculator"
        elif app in ["code", "vscode"]:
            return "code"
        elif app in ["settings"]:
            return "gnome-control-center"
        return command_or_app

    def focus_window(self, app_or_title: Optional[str] = None) -> bool:
        """
        Forces active window focus using GNOME Shell or AT-SPI, ensuring key/mouse events register.
        1. Tries GNOME Shell Eval D-Bus activation.
        2. Tries AT-SPI accessible window discovery and focus.
        3. Falls back to clicking top-center header bar of the foremost window.
        """
        # 1. Try GNOME Shell Eval
        try:
            res = subprocess.run([
                "gdbus", "call", "--session", "--dest", "org.gnome.Shell",
                "--object-path", "/org/gnome/Shell",
                "--method", "org.gnome.Shell.Eval",
                "global.display.get_focus_window().activate(global.get_current_time());"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1, text=True, check=False)
            if res.returncode == 0 and "true" in res.stdout.lower():
                return True
        except Exception:
            pass

        # 2. Try AT-SPI accessible window discovery and focus
        try:
            import gi
            gi.require_version("Atspi", "2.0")
            from gi.repository import Atspi
            desktop = Atspi.get_desktop(0)
            target_kw = (app_or_title or "").lower().strip()
            
            for i in range(desktop.get_child_count()):
                app = desktop.get_child_at_index(i)
                if not app:
                    continue
                app_name = (app.get_name() or "").lower()
                matches = not target_kw or (target_kw in app_name) or any(k in app_name for k in ["brave", "chrome", "chromium", "firefox", "calculator", "nautilus", "files"])
                if matches:
                    for j in range(app.get_child_count()):
                        win = app.get_child_at_index(j)
                        if not win:
                            continue
                        states = win.get_state_set()
                        if states.contains(Atspi.StateType.SHOWING) or states.contains(Atspi.StateType.VISIBLE):
                            comp = win.get_component_iface()
                            if comp:
                                try:
                                    if comp.grab_focus():
                                        return True
                                except Exception:
                                    pass
                                bbox = comp.get_extents(Atspi.CoordType.SCREEN)
                                if bbox.width > 100 and bbox.height > 100:
                                    # Click top bar / header to bring window to front
                                    hx = max(10, bbox.x + min(200, bbox.width // 2))
                                    hy = max(10, bbox.y + 15)
                                    self.click(hx, hy)
                                    time.sleep(0.15)
                                    return True
        except Exception as e:
            logger.debug("AT-SPI window focus attempt: %s", e)

        # 3. Safe fallback: click top-center of desktop bounds
        try:
            bounds = self.get_desktop_bounds()
            self.click(bounds[0] + bounds[2] // 2, bounds[1] + 25)
            time.sleep(0.1)
            return True
        except Exception:
            return False

    @classmethod
    def launch_app(cls, command_or_app: str, wait_seconds: float = 2.5, profile: Optional[str] = None) -> None:
        """Launches an application in the background and waits for the window to render on Wayland."""
        cmd = cls.resolve_app_command(command_or_app, profile=profile)
        subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        if wait_seconds > 0:
            time.sleep(wait_seconds)

        # Force active window focus
        try:
            cls.get_instance().focus_window(command_or_app)
        except Exception:
            pass

        # Optional split-screen tiling if requested
        if os.getenv("SPLIT_SCREEN_PILOT", "0") == "1":
            try:
                cls.get_instance().tile_active_window("right")
            except Exception:
                pass

    def read_clipboard(self) -> str:
        """Reads text from system clipboard."""
        return read_clipboard()

    @classmethod
    def open_url(cls, url_or_query: str, browser: Optional[str] = None, profile: Optional[str] = None) -> None:
        """
        Directly opens a URL or Google search query in the browser in 1 fast step,
        bypassing the fragile launch -> wait -> Ctrl+L -> type -> Enter sequence.
        """
        target = url_or_query.strip()
        is_url = bool(re.match(r"^(https?://|www\.|file:///)", target, re.IGNORECASE))
        if not is_url and "." not in target.split("/")[0]:
            target = f"https://www.google.com/search?q={urllib.parse.quote_plus(target)}"
        elif target.startswith("www."):
            target = f"https://{target}"

        bin_cmd, resolved_pdir, _ = resolve_browser_and_profile(browser, profile)
        if "chrome" in bin_cmd.lower():
            cmd = f"google-chrome --new-window --profile-directory='{resolved_pdir}' \"{target}\""
        elif "chromium" in bin_cmd.lower():
            cmd = f"chromium --new-window --profile-directory='{resolved_pdir}' \"{target}\""
        else:
            cmd = (
                f"brave-browser --new-window --profile-directory='{resolved_pdir}' \"{target}\" || "
                f"/snap/bin/brave --new-window --profile-directory='{resolved_pdir}' \"{target}\" || "
                f"brave --new-window --profile-directory='{resolved_pdir}' \"{target}\""
            )

        subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        time.sleep(2.0)

        # Force active window focus
        try:
            cls.get_instance().focus_window(browser or "browser")
        except Exception:
            pass

        if os.getenv("SPLIT_SCREEN_PILOT", "0") == "1":
            try:
                cls.get_instance().tile_active_window("right")
            except Exception:
                pass

    @classmethod
    def prioritize_ocr_elements(
        cls,
        elements: List[Dict[str, Any]],
        bounds: Optional[Tuple[int, int, int, int]] = None,
        max_elements: int = 80
    ) -> List[Dict[str, Any]]:
        """
        Prioritizes OCR elements so centered dialogs/modals are not dropped by top-bar clutter.
        Returns up to max_elements sorted in natural reading order (y // 16, x).
        """
        if len(elements) <= max_elements:
            return sorted(elements, key=lambda e: (e.get("y", 0) // 16, e.get("x", 0)))

        # Determine desktop or bounding box dimensions
        if bounds:
            bx, by, bw, bh = bounds
        else:
            bx, by, bw, bh = 0, 0, 1920, 1080

        # Center modal/dialog zone (middle 70% horizontally and vertically)
        cx_min = bx + bw * 0.15
        cx_max = bx + bw * 0.85
        cy_min = by + bh * 0.15
        cy_max = by + bh * 0.85

        center_elements = []
        outer_elements = []

        for el in elements:
            cx = el.get("cx", el.get("x", 0) + el.get("w", 0) // 2)
            cy = el.get("cy", el.get("y", 0) + el.get("h", 0) // 2)
            if cx_min <= cx <= cx_max and cy_min <= cy <= cy_max:
                center_elements.append(el)
            else:
                outer_elements.append(el)

        # Retain center elements first (up to max_elements * 0.7), fill remainder with outer elements
        max_center = int(max_elements * 0.70)
        chosen_center = center_elements[:max_center]
        remaining_slots = max_elements - len(chosen_center)
        chosen_outer = outer_elements[:remaining_slots]

        # Combine and sort in natural top-to-bottom, left-to-right reading order
        combined = chosen_center + chosen_outer
        combined.sort(key=lambda e: (e.get("y", 0) // 16, e.get("x", 0)))
        return combined

    @staticmethod
    def get_screen_ocr_elements(image_path: str) -> List[Dict[str, Any]]:
        """
        Extracts words and bounding boxes using Tesseract TSV output.
        Returns a list of dicts: {'text': str, 'x': int, 'y': int, 'w': int, 'h': int, 'cx': int, 'cy': int}
        """
        try:
            res = subprocess.run(
                ["tesseract", image_path, "stdout", "tsv"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=8
            )
            elements = []
            for line in res.stdout.strip().split("\n")[1:]:
                parts = line.split("\t")
                if len(parts) >= 12:
                    text = parts[11].strip()
                    conf = float(parts[10]) if parts[10] != "-1" else -1
                    if text and conf > 25:
                        left = int(parts[6])
                        top = int(parts[7])
                        w = int(parts[8])
                        h = int(parts[9])
                        elements.append({
                            "text": text,
                            "x": left,
                            "y": top,
                            "w": w,
                            "h": h,
                            "cx": left + w // 2,
                            "cy": top + h // 2,
                            "conf": conf
                        })
            return elements
        except Exception:
            return []
