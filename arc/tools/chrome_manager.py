"""Singleton manager for visible Chrome on Wayland — attaches to YOUR live profile.

Ensures ARC drives the exact Chrome you see (same google-chrome binary,
same user-data-dir, preserving chatgpt.com login for Library/image-gen)
and avoids per-call --new-window spam or /tmp profile copies.

Attach strategy:
1. Try CDP http://127.0.0.1:<port>/json/version → connect_over_cdp (reuses your logged-in context)
2. If no Chrome at all → launch debuggable Chrome with --user-data-dir=YOUR_PROFILE --remote-debugging-port=<port>
3. If Chrome running without CDP → don't spawn second --user-data-dir=Same (Singleton lock). Instead xdg-open URL in existing window; for DOM automation fall back to persistent context with same dir only if lock free, otherwise use wtype/wayland fallback and instruct user to restart Chrome with --remote-debugging-port.

All Playwright sync_api calls run on a single dedicated thread to avoid greenlet errors.
"""

from __future__ import annotations

import http.client
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("arc.chrome_manager")

# Configurable via env (see arc/app health report)
def _cdp_port() -> int:
    for name in ("ARC_CHROME_CDP_PORT", "ARC_BROWSER_CDP_PORT"):
        v = os.environ.get(name)
        if v and v.isdigit():
            return int(v)
    return 9222

def _user_data_dir() -> Path:
    # Explicit override via config file is handled by ChromeManager.configure() from ArcApp
    # Here we honor env and common locations
    for name in ("ARC_CHROME_USER_DATA_DIR", "ARC_BROWSER_USER_DATA_DIR"):
        v = os.environ.get(name)
        if v:
            p = Path(v).expanduser()
            if p.exists():
                return p
    # Common locations — pick first that exists and looks like Chrome profile
    candidates = [
        Path.home() / ".config/google-chrome",
        Path.home() / ".config/google-chrome-beta",
        Path.home() / ".config/google-chrome-unstable",
        Path.home() / ".config/chromium",
        Path.home() / "snap/chromium/common/chromium",
        Path.home() / ".config/BraveSoftware/Brave-Browser",
    ]
    for p in candidates:
        if (p / "Default").exists() or (p / "Local State").exists():
            return p
    # Default even if not exists yet
    return Path.home() / ".config/google-chrome"


def _resolve_user_data_dir_from_config() -> Path:
    """If ChromeManager was configured via App, use that; otherwise env/common."""
    # Check if a configured override was set via singleton
    try:
        if ChromeManager._configured_user_data_dir is not None:  # type: ignore[attr-defined]
            p = Path(ChromeManager._configured_user_data_dir).expanduser()  # type: ignore[attr-defined]
            return p
    except Exception:
        pass
    return _user_data_dir()

_DISPLAY = os.environ.get("DISPLAY") or ":1"
_WAYLAND = os.environ.get("WAYLAND_DISPLAY") or "wayland-0"

def _env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("DISPLAY", _DISPLAY)
    env.setdefault("WAYLAND_DISPLAY", _WAYLAND)
    if "XDG_SESSION_TYPE" not in env:
        env["XDG_SESSION_TYPE"] = os.environ.get("XDG_SESSION_TYPE", "wayland")
    return env

def _has_existing_chrome_process() -> bool:
    for bin_name in ("chrome", "chromium", "brave"):
        try:
            r = subprocess.run(["pgrep", "-a", bin_name], capture_output=True, text=True, timeout=2)
            if r.returncode == 0 and bin_name in r.stdout.lower():
                return True
        except Exception:
            pass
    return False

def _has_existing_chrome_with_cdp(port: int) -> bool:
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        conn.request("GET", "/json/version")
        return conn.getresponse().status == 200
    except Exception:
        return False

def _detect_chrome_cmd_user_data_dir() -> Optional[str]:
    """Try to read --user-data-dir from running chrome cmdline, if any."""
    try:
        r = subprocess.run(["pgrep", "-a", "chrome"], capture_output=True, text=True, timeout=2)
        if r.returncode != 0:
            return None
        for line in r.stdout.splitlines():
            if "--user-data-dir=" in line:
                for part in line.split():
                    if part.startswith("--user-data-dir="):
                        return part.split("=", 1)[1]
    except Exception:
        pass
    return None

def _xdg_open(url: str) -> bool:
    try:
        env = _env()
        # Try xdg-open, gio open, then google-chrome --new-window as last resort
        for cmd in (["xdg-open", url], ["gio", "open", url], ["google-chrome", "--new-window", url]):
            try:
                subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except FileNotFoundError:
                continue
    except Exception:
        pass
    return False


class _DummyPage:
    """Fallback when Chrome runs without CDP — represents your real window via xdg-open.
    Playwright DOM ops become no-ops; caller should fallback to wayland_input (wtype/ydotool).
    """

    def __init__(self, url: str = "https://chatgpt.com"):
        self.url = url
        self._url = url

    def title(self) -> str:
        return "ChatGPT — your Chrome (CDP disabled, use --remote-debugging-port)"

    def bring_to_front(self) -> None:
        pass

    def goto(self, url: str, **_: Any) -> None:
        _xdg_open(url)
        self.url = url
        self._url = url

    def wait_for_timeout(self, _ms: int) -> None:
        time.sleep(min(_ms, 1000) / 1000)

    def inner_text(self, _sel: str) -> str:
        return "Dummy page — real Chrome via xdg-open. Restart Chrome with --remote-debugging-port for full automation."

    def eval_on_selector_all(self, *_, **__) -> list:
        return []

    def screenshot(self, **_: Any) -> None:
        pass

    def click(self, *_, **__) -> None:
        raise RuntimeError("CDP disabled — use wayland_input move/click fallback")

    def fill(self, *_, **__) -> None:
        raise RuntimeError("CDP disabled — use wayland_input type fallback")

    @property
    def keyboard(self):  # type: ignore
        class _Kb:
            def type(self, *_a, **_kw): raise RuntimeError("CDP disabled")
            def press(self, *_a, **_kw): raise RuntimeError("CDP disabled")
        return _Kb()

    @property
    def mouse(self):  # type: ignore
        class _M:
            def click(self, *_a, **_kw): raise RuntimeError("CDP disabled")
            def move(self, *_a, **_kw): raise RuntimeError("CDP disabled")
            def wheel(self, *_a, **_kw): raise RuntimeError("CDP disabled")
        return _M()

    def focus(self, *_, **__) -> None:
        raise RuntimeError("CDP disabled")

class ChromeManager:
    _instance: Optional["ChromeManager"] = None
    _lock = threading.Lock()
    _configured_user_data_dir: Optional[str] = None
    _configured_cdp_port: Optional[int] = None

    @classmethod
    def configure(cls, user_data_dir: Optional[str] = None, cdp_port: Optional[int] = None) -> None:
        """Set profile/port from ArcConfig (called by ArcApp). Updates live instance if exists."""
        if user_data_dir:
            cls._configured_user_data_dir = str(Path(user_data_dir).expanduser())
        if cdp_port is not None:
            try:
                cls._configured_cdp_port = int(cdp_port)
            except Exception:
                pass
        # If singleton already exists, push config to it
        if cls._instance is not None:
            try:
                if user_data_dir:
                    cls._instance._user_data_dir = Path(cls._configured_user_data_dir).expanduser()  # type: ignore[arg-type]
                if cdp_port is not None and cls._configured_cdp_port is not None:
                    cls._instance._cdp_port = cls._configured_cdp_port
                    cls._instance._cdp_url = f"http://127.0.0.1:{cls._configured_cdp_port}"
            except Exception:
                pass

    @classmethod
    def instance(cls) -> "ChromeManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arc-playwright")
        self._pw = None
        self._page: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._last_url: str = ""
        # Respect configured values (from App) or env/common fallback
        if ChromeManager._configured_cdp_port is not None:
            self._cdp_port = ChromeManager._configured_cdp_port
        else:
            self._cdp_port = _cdp_port()
        self._cdp_url = f"http://127.0.0.1:{self._cdp_port}"
        if ChromeManager._configured_user_data_dir is not None:
            self._user_data_dir = Path(ChromeManager._configured_user_data_dir).expanduser()
        else:
            # Check config helper that also checks configured singleton
            try:
                self._user_data_dir = _resolve_user_data_dir_from_config()
            except Exception:
                self._user_data_dir = _user_data_dir()
        self._daemon_proc: Optional[subprocess.Popen] = None
        self._op_lock = threading.Lock()

    def _is_cdp_alive(self) -> bool:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", self._cdp_port, timeout=1)
            conn.request("GET", "/json/version")
            return conn.getresponse().status == 200
        except Exception:
            return False

    def _thread_init_playwright(self) -> None:
        if self._pw is not None:
            return
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()

    def availability(self) -> str:
        port = self._cdp_port
        cdp = "alive" if self._is_cdp_alive() else "down"
        existing = "running" if _has_existing_chrome_process() else "none"
        udd = str(self._user_data_dir)
        live_args = _detect_chrome_cmd_user_data_dir() or "default"
        base = f"profile={udd} (live_args={live_args}) cdp=:{port} {cdp} chrome={existing} DISPLAY={_env().get('DISPLAY')}"
        if cdp == "down" and existing == "running":
            base += f" — FIX: pkill chrome; google-chrome --remote-debugging-port={port} --user-data-dir={udd} --ozone-platform-hint=auto  (or ./scripts/enable-chrome-cdp.sh)"
        return base

    def _thread_connect_or_launch(self) -> Any:
        self._thread_init_playwright()
        assert self._pw is not None

        # 1) CDP already alive -> connect
        if self._is_cdp_alive():
            try:
                browser = self._pw.chromium.connect_over_cdp(self._cdp_url)
                ctx = browser.contexts[0] if browser.contexts else browser.new_context()
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                self._browser, self._context, self._page = browser, ctx, page
                logger.debug("ChromeManager: connected over CDP :%s", self._cdp_port)
                return page
            except Exception as exc:
                logger.debug("CDP connect failed: %s", exc)

        # 2) No Chrome at all -> launch debuggable helper with YOUR profile
        if not _has_existing_chrome_process():
            if self._daemon_proc is None or self._daemon_proc.poll() is not None:
                try:
                    # Ensure parent dir exists
                    self._user_data_dir.mkdir(parents=True, exist_ok=True)
                    self._daemon_proc = subprocess.Popen(
                        [
                            "google-chrome",
                            f"--user-data-dir={self._user_data_dir}",
                            f"--remote-debugging-port={self._cdp_port}",
                            "--no-first-run",
                            "--no-default-browser-check",
                            "--ozone-platform-hint=auto",
                            "about:blank",
                        ],
                        env=_env(),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    for _ in range(20):
                        if self._is_cdp_alive():
                            break
                        time.sleep(0.2)
                    if self._is_cdp_alive():
                        browser = self._pw.chromium.connect_over_cdp(self._cdp_url)
                        ctx = browser.contexts[0] if browser.contexts else browser.new_context()
                        page = ctx.pages[0] if ctx.pages else ctx.new_page()
                        self._browser, self._context, self._page = browser, ctx, page
                        logger.info("ChromeManager: launched your-profile Chrome :%s %s", self._cdp_port, self._user_data_dir)
                        return page
                except Exception as exc:
                    logger.debug("CDP daemon launch failed: %s", exc)
            else:
                # Daemon already launching, wait
                for _ in range(10):
                    if self._is_cdp_alive():
                        break
                    time.sleep(0.2)
                if self._is_cdp_alive():
                    try:
                        browser = self._pw.chromium.connect_over_cdp(self._cdp_url)
                        ctx = browser.contexts[0] if browser.contexts else browser.new_context()
                        page = ctx.pages[0] if ctx.pages else ctx.new_page()
                        self._browser, self._context, self._page = browser, ctx, page
                        return page
                    except Exception as exc:
                        logger.debug("CDP reuse failed: %s", exc)

        # 3) Chrome running without CDP -> don't spawn second --user-data-dir=Same (Singleton lock).
        # Use xdg-open to tell existing instance to handle URL; for automation we need CDP, so
        # try persistent context only if lock free, otherwise instruct user.
        if _has_existing_chrome_process() and not self._is_cdp_alive():
            logger.info("ChromeManager: existing Chrome without CDP on :%s — will use xdg-open and fallback persistent context", self._cdp_port)
            # Try persistent context with YOUR profile only if Playwright can acquire it (will fail if Singleton locked)
            try:
                # Use same user-data-dir explicitly — Playwright will error if locked, we catch
                browser = self._pw.chromium.launch_persistent_context(
                    user_data_dir=str(self._user_data_dir),
                    channel="chrome",
                    headless=False,
                    args=["--ozone-platform-hint=auto", "--disable-blink-features=AutomationControlled", "--no-first-run"],
                )
                # launch_persistent_context returns BrowserContext directly
                page = browser.pages[0] if browser.pages else browser.new_page()
                self._browser, self._context, self._page = None, browser, page  # type: ignore
                logger.info("ChromeManager: persistent context with your profile %s", self._user_data_dir)
                return page
            except Exception as exc:
                logger.debug("Persistent context with your profile failed (Singleton locked, expected): %s", exc)
                # Don't launch isolated guest when your real Chrome is running — it would be logged-out.
                # Return dummy that represents your real window (via xdg-open); automation will fallback to wtype/ydotool.
                dummy = _DummyPage(url="https://chatgpt.com")
                self._page = dummy  # type: ignore
                self._browser = None
                self._context = None
                logger.warning(
                    "ChromeManager: Chrome running without CDP — using your real window via xdg-open (no isolated guest). "
                    "For full DOM automation, restart your Chrome: google-chrome --remote-debugging-port=%s --user-data-dir=%s --ozone-platform-hint=auto",
                    self._cdp_port, self._user_data_dir,
                )
                return dummy  # type: ignore

        # 4) Isolated fallback (no existing chrome, CDP failed)
        try:
            args = ["--ozone-platform-hint=auto", "--disable-blink-features=AutomationControlled", "--no-first-run"]
            try:
                browser = self._pw.chromium.launch(headless=False, channel="chrome", args=args)
            except Exception:
                browser = self._pw.chromium.launch(headless=False, args=args)
            ctx = browser.new_context(viewport={"width": 1280, "height": 800})
            try:
                ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            except Exception:
                pass
            page = ctx.new_page()
            self._browser, self._context, self._page = browser, ctx, page
            logger.warning("ChromeManager: isolated fallback — not your profile. Fix: close other Chromes and set ARC_CHROME_USER_DATA_DIR=%s", self._user_data_dir)
            return page
        except Exception as exc:
            raise RuntimeError(f"Chrome launch failed: {exc}") from exc

    def _thread_ensure_visible(self, url: str) -> Any:
        page = self._page
        if page is not None:
            try:
                _ = page.url
                if url and url.rstrip("/") in (page.url or ""):
                    try:
                        page.bring_to_front()
                    except Exception:
                        pass
                    return page
                if url:
                    # If existing Chrome has no CDP but page is from persistent/isolated context, navigate there
                    # Otherwise try to tell existing Chrome to open via xdg-open then navigate our page
                    if _has_existing_chrome_process() and not self._is_cdp_alive():
                        # Best-effort: ask OS Chrome to open URL so user sees it even if our Page is isolated
                        try:
                            _xdg_open(url)
                        except Exception:
                            pass
                    page.goto(url, wait_until="domcontentloaded", timeout=40_000)
                    page.wait_for_timeout(2500)
                    self._last_url = url
                return page
            except Exception:
                pass
        page = self._thread_connect_or_launch()
        if url:
            try:
                if _has_existing_chrome_process() and not self._is_cdp_alive():
                    try:
                        _xdg_open(url)
                    except Exception:
                        pass
                if page.url == "about:blank" or "about:blank" in page.url:
                    page.goto(url, wait_until="domcontentloaded", timeout=40_000)
                else:
                    if url.rstrip("/") not in page.url:
                        page.goto(url, wait_until="domcontentloaded", timeout=40_000)
                page.wait_for_timeout(2500)
                try:
                    page.bring_to_front()
                except Exception:
                    pass
                self._last_url = url
            except Exception as exc:
                logger.debug("goto %s failed: %s", url, exc)
        return page

    def _submit(self, fn, *args, **kwargs) -> Any:
        fut: Future = self._executor.submit(fn, *args, **kwargs)
        return fut.result(timeout=60)

    def ensure_visible(self, url: str = "") -> Any:
        if url and not url.startswith(("http://", "https://")):
            url = "https://" + url
        with self._op_lock:
            # Only bootstrap --new-window if truly no Chrome at all
            if not self._is_cdp_alive() and self._page is None and url and not _has_existing_chrome_process():
                try:
                    subprocess.Popen(
                        ["google-chrome", f"--user-data-dir={self._user_data_dir}", "--new-window", url],
                        env=_env(),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    time.sleep(1.0)
                except Exception:
                    pass
            # If Chrome running without CDP, ensure the URL is at least opened in the live window via xdg-open
            if url and _has_existing_chrome_process() and not self._is_cdp_alive():
                try:
                    _xdg_open(url)
                    time.sleep(0.5)
                except Exception:
                    pass
            page = self._submit(self._thread_ensure_visible, url or "https://chatgpt.com")
            return page

    def do(self, fn) -> Any:
        def _wrap():
            page = self._page
            if page is None:
                page = self._thread_connect_or_launch()
            return fn(page)
        return self._submit(_wrap)

    def screenshot(self, path: Path) -> str:
        def _shot(page):
            path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(path))
            return str(path)
        return self.do(_shot)

    def title(self) -> str:
        return self.do(lambda p: p.title())

    def url(self) -> str:
        return self.do(lambda p: p.url)

    def close(self) -> None:
        def _close():
            try:
                if self._browser:
                    # Don't close user's live browser if it's persistent context
                    # Only close isolated browsers
                    if hasattr(self._browser, "close"):
                        self._browser.close()
                if self._context and hasattr(self._context, "close"):
                    try:
                        self._context.close()
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            self._submit(_close)
        except Exception:
            pass
