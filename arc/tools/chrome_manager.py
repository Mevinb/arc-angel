"""Singleton manager for visible Chrome on Wayland.

Ensures exactly one visible Chrome window/tab is reused across all ARC
tool calls. Attaches via CDP to the existing session (preserving login)
and avoids per-call ``google-chrome --new-window`` spam or /tmp profile
copies that duplicate memory and lock the main profile.

All Playwright sync_api calls run on a single dedicated thread to avoid
greenlet `cannot switch to a different thread` errors when the
Orchestrator invokes tools from inside an asyncio-adjacent ThreadPool.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("arc.chrome_manager")

_CDPPort = 9222
_DISPLAY = os.environ.get("DISPLAY", ":1")
_WAYLAND = os.environ.get("WAYLAND_DISPLAY", "wayland-0")


def _env() -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("DISPLAY", _DISPLAY)
    env.setdefault("WAYLAND_DISPLAY", _WAYLAND)
    env.setdefault("XDG_SESSION_TYPE", "wayland")
    return env


class ChromeManager:
    """Process-wide singleton that owns one Playwright thread."""

    _instance: Optional["ChromeManager"] = None
    _lock = threading.Lock()

    @classmethod
    def instance(cls) -> "ChromeManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arc-playwright")
        self._pw = None  # sync_playwright instance (started on thread)
        self._page: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._last_url: str = ""
        self._cdp_url = f"http://127.0.0.1:{_CDPPort}"
        # Serialize ensure_visible calls
        self._op_lock = threading.Lock()

    # ---------------------------------------------------------------- internal helpers running on the playwright thread

    def _is_cdp_alive(self) -> bool:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", _CDPPort, timeout=1)
            conn.request("GET", "/json/version")
            return conn.getresponse().status == 200
        except Exception:
            return False

    def _thread_init_playwright(self) -> None:
        if self._pw is not None:
            return
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()

    def _thread_connect_or_launch(self) -> Any:
        """Return a Page attached to the visible Chrome, running on this thread."""
        self._thread_init_playwright()
        assert self._pw is not None

        # 1) Try CDP to existing debuggable Chrome
        if self._is_cdp_alive():
            try:
                browser = self._pw.chromium.connect_over_cdp(self._cdp_url)
                ctx = browser.contexts[0] if browser.contexts else browser.new_context()
                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                self._browser, self._context, self._page = browser, ctx, page
                logger.debug("ChromeManager: connected over CDP")
                return page
            except Exception as exc:
                logger.debug("CDP connect failed: %s", exc)

        # 2) Ensure a debuggable Chrome daemon is running (once, no profile copy)
        if not self._is_cdp_alive():
            try:
                subprocess.Popen(
                    [
                        "google-chrome",
                        f"--remote-debugging-port={_CDPPort}",
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--ozone-platform-hint=auto",
                        "about:blank",
                    ],
                    env=_env(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                # Wait for CDP
                for _ in range(20):
                    if self._is_cdp_alive():
                        break
                    time.sleep(0.2)
                if self._is_cdp_alive():
                    browser = self._pw.chromium.connect_over_cdp(self._cdp_url)
                    ctx = browser.contexts[0] if browser.contexts else browser.new_context()
                    page = ctx.pages[0] if ctx.pages else ctx.new_page()
                    self._browser, self._context, self._page = browser, ctx, page
                    logger.debug("ChromeManager: launched CDP daemon and connected")
                    return page
            except Exception as exc:
                logger.debug("CDP daemon launch failed: %s", exc)

        # 3) Fallback: launch visible Chrome via Playwright using system channel
        # (last resort — still visible, but isolated). We keep browser open.
        try:
            args = ["--ozone-platform-hint=auto", "--disable-blink-features=AutomationControlled", "--no-first-run", "--no-default-browser-check"]
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
            logger.debug("ChromeManager: launched visible channel=chrome")
            return page
        except Exception as exc:
            raise RuntimeError(f"Chrome launch failed: {exc}") from exc

    def _thread_ensure_visible(self, url: str) -> Any:
        page = self._page
        # If we have a live page, reuse it
        if page is not None:
            try:
                # Lightweight liveness probe
                _ = page.url
                # Already on desired url? Just focus
                if url and url.rstrip("/") in (page.url or ""):
                    try:
                        page.bring_to_front()
                    except Exception:
                        pass
                    return page
                # Navigate existing page instead of new window
                if url:
                    page.goto(url, wait_until="domcontentloaded", timeout=40_000)
                    page.wait_for_timeout(2500)
                    self._last_url = url
                return page
            except Exception:
                # Page dead, fall through to reconnect
                pass

        # No live page — connect/launch
        page = self._thread_connect_or_launch()
        if url:
            try:
                # Prefer navigating existing page over spawning new window
                if page.url == "about:blank" or "about:blank" in page.url:
                    page.goto(url, wait_until="domcontentloaded", timeout=40_000)
                else:
                    # If CDP page is not on target, navigate it
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

    # ---------------------------------------------------------------- public API (thread-safe, idempotent)

    def ensure_visible(self, url: str = "") -> Any:
        """Idempotent: return a live Page on `url`, reusing the window if possible."""
        # Normalize url
        if url and not url.startswith(("http://", "https://")):
            url = "https://" + url
        with self._op_lock:
            # Also try OS-level pop via google-chrome --new-window only if we have
            # no live page AND CDP not alive (i.e., Chrome not running at all).
            # Otherwise rely on CDP navigation to avoid window spam.
            if not self._is_cdp_alive() and self._page is None and url:
                try:
                    # One-shot OS pop to bootstrap Chrome if nothing running
                    subprocess.Popen(
                        ["google-chrome", "--new-window", url],
                        env=_env(),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    time.sleep(1.0)
                except Exception:
                    pass
            page = self._submit(self._thread_ensure_visible, url or "https://chatgpt.com")
            return page

    def do(self, fn) -> Any:
        """Run an arbitrary callable `fn(page)` on the playwright thread."""
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
        """For tests only."""
        def _close():
            try:
                if self._browser:
                    self._browser.close()
            except Exception:
                pass
        try:
            self._submit(_close)
        except Exception:
            pass


__all__ = ["ChromeManager"]
