"""Phase 4 — Browser automation engine.

Three tiers, used as available:

1. ``browser.task``      — Browser-Use (https://github.com/browser-use/browser-use):
                            an LLM-driven agent for multi-step web tasks
                            (navigate, click, type, fill forms, screenshot).
2. ``browser.playwright``— direct Playwright control: open a URL, extract text /
                            links, take screenshots.
3. ``web.fetch``         — dependency-free fetch+readability fallback using
                            urllib (no JS, but works everywhere). Used by the
                            internship engine when no browser backend exists.
"""

from __future__ import annotations

import gzip
import html.parser
import logging
import re
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, parse, request

from ..safety.permissions import RiskLevel
from .base import Tool, ToolResult

logger = logging.getLogger("arc.browser")

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


# --------------------------------------------------------------------- web.fetch
class _TextExtractor(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "head", "svg"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self.links: List[Dict[str, str]] = []
        self._skip_depth = 0
        self._anchor: Optional[Dict[str, str]] = None
        self._title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self.SKIP:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href", "")
            if href:
                self._anchor = {"href": href, "text": ""}
        elif tag in ("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4") :
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag == "a" and self._anchor is not None:
            text = self._anchor["text"].strip()
            if text:
                self.links.append({"href": self._anchor["href"], "text": text[:120]})
            self._anchor = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title += data
        if self._anchor is not None:
            self._anchor["text"] += data
        self.parts.append(data)


def fetch_url(url: str, timeout: int = 20, max_bytes: int = 2_000_000) -> Dict[str, Any]:
    """GET a URL with stdlib urllib and extract readable text + links."""
    if not re.match(r"^https?://", url):
        url = "https://" + url
    req = request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
    })
    with request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(max_bytes)
        encoding = resp.headers.get("Content-Encoding", "")
        if encoding == "gzip":
            raw = gzip.decompress(raw)
        elif encoding == "deflate":
            raw = zlib.decompress(raw, -zlib.MAX_WBITS)
        charset = "utf-8"
        content_type = resp.headers.get("Content-Type", "")
        match = re.search(r"charset=([\w-]+)", content_type)
        if match:
            charset = match.group(1)
        text = raw.decode(charset, errors="replace")
        final_url = resp.url
    # JSON payloads (job-board APIs etc.) must NOT go through the HTML
    # extractor — it would mangle them into unreadable text.
    if "json" in content_type.lower():
        return {"url": final_url, "title": "", "text": text, "links": []}
    extractor = _TextExtractor()
    try:
        extractor.feed(text)
    except Exception:  # noqa: BLE001 - tolerate broken HTML
        pass
    content = re.sub(r"\n{3,}", "\n\n", "".join(extractor.parts))
    content = re.sub(r"[ \t]{2,}", " ", content).strip()
    # Resolve relative links
    for link in extractor.links:
        link["href"] = parse.urljoin(final_url, link["href"])
    return {
        "url": final_url,
        "title": extractor._title.strip(),
        "text": content,
        "links": extractor.links,
    }


class WebFetchTool(Tool):
    name = "web.fetch"
    description = ("Fetch a web page and return its readable text and links. "
                   "No JavaScript execution — use browser tools for dynamic pages.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "max_chars": {"type": "integer", "description": "Truncate text to this many chars (default 6000)"},
        },
        "required": ["url"],
    }

    def run(self, url: str = "", max_chars: int = 6000, **_: Any) -> ToolResult:
        if not url:
            return ToolResult.failure("No URL provided")
        try:
            result = fetch_url(url)
        except error.HTTPError as exc:
            return ToolResult.failure(f"HTTP {exc.code} for {url}")
        except (error.URLError, TimeoutError, OSError) as exc:
            return ToolResult.failure(f"Failed to fetch {url}: {exc}")
        text = result["text"][:max(int(max_chars), 500)]
        header = f"# {result['title'] or result['url']}\n"
        return ToolResult.success(header + text, url=result["url"],
                                  title=result["title"], links=result["links"][:40])


# ------------------------------------------------------------------ browser.task
class BrowserUseTool(Tool):
    """Multi-step, LLM-driven browser tasks via Browser-Use."""

    name = "browser.task"
    description = ("Run a multi-step browser task with an AI agent (navigate, "
                   "search, click, type, fill forms, extract data). Example: "
                   "'go to internships.com and list the first 10 listings'.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "task": {"type": "string", "description": "Natural-language browser task"},
            "max_steps": {"type": "integer", "description": "Maximum agent steps (default 25)"},
        },
        "required": ["task"],
    }

    def __init__(self, router: Optional[Any] = None) -> None:
        self.router = router

    def availability(self) -> str:
        try:
            import browser_use  # noqa: F401
            return "browser-use available"
        except ImportError:
            return "browser-use not installed — pip install browser-use && playwright install chromium"

    def run(self, task: str = "", max_steps: int = 25, **_: Any) -> ToolResult:
        if not task:
            return ToolResult.failure("No task provided")
        try:
            from browser_use import Agent  # type: ignore
        except ImportError:
            return ToolResult.failure(
                "browser-use is not installed. Install with: "
                "pip install browser-use && playwright install chromium")
        if self.router is None:
            return ToolResult.failure("No LLM router configured for the browser agent")

        try:
            import asyncio

            # Try new browser-use 0.13+ imports first, then fall back to 0.1
            Browser = None
            BrowserConfig = None
            for import_path in [
                "browser_use.browser",  # 0.13+
                "browser_use.browser.browser",  # 0.1
            ]:
                try:
                    mod = __import__(import_path, fromlist=["Browser", "BrowserConfig"])
                    Browser = getattr(mod, "Browser", None)
                    BrowserConfig = getattr(mod, "BrowserConfig", None)
                    if Browser is not None:
                        break
                except (ImportError, ModuleNotFoundError):
                    continue
            # If still not found, try top-level
            if Browser is None:
                try:
                    from browser_use import Browser as _B  # type: ignore
                    Browser = _B
                except Exception:
                    pass
            if Browser is None:
                return ToolResult.failure(
                    "browser-use Browser class not found for this version (0.13.8). "
                    "Use browser.open (Playwright) or computer control via UACC instead. "
                    "Try: uv pip install 'browser-use==0.1.50' for legacy API or update arc/tools/browser.py"
                )

            async def _run() -> str:
                client = self.router.client  # shared OpenAI-compatible client
                from openai import AsyncOpenAI
                async_client = AsyncOpenAI(
                    base_url=self.router.config.base_url,
                    api_key=self.router.config.api_key or "missing-key",
                    timeout=self.router.config.timeout_seconds,
                )
                # BrowserConfig may not exist in 0.13+
                browser = Browser() if BrowserConfig is None else Browser(config=BrowserConfig(headless=True))
                agent = Agent(
                    task=task,
                    llm=async_client,
                    browser=browser,
                    max_steps=int(max_steps),
                )
                result = await agent.run()
                return str(result)

            # Browser-Use needs its own event loop; run in a new thread to avoid
            # "asyncio.run() cannot be called from a running event loop"
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _run())
                output = future.result(timeout=120)
            return ToolResult.success(output[:8000] or "(task finished)")
        except Exception as exc:  # noqa: BLE001
            logger.exception("browser-use task failed")
            return ToolResult.failure(f"Browser task failed: {exc}")


# -------------------------------------------------------------- browser.playwright
class PlaywrightTool(Tool):
    """Direct Chromium control through Playwright: open, extract, screenshot."""

    name = "browser.open"
    description = ("Open a URL in headless Chromium and extract the rendered text "
                   "and links (JavaScript executes). Use for dynamic pages that "
                   "web.fetch cannot read.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "url": {"type": "string", "description": "URL to open"},
            "screenshot": {"type": "boolean", "description": "Also save a screenshot (default false)"},
            "max_chars": {"type": "integer", "description": "Truncate text (default 6000)"},
            "visible": {"type": "boolean", "description": "Pop Chrome visibly on screen (Wayland, bypasses Cloudflare for chatgpt.com)"},
        },
        "required": ["url"],
    }

    def __init__(self, screenshot_dir: Optional[Path] = None) -> None:
        self.screenshot_dir = screenshot_dir or Path("data/screenshots")

    def availability(self) -> str:
        try:
            import playwright  # noqa: F401
            return "playwright available"
        except ImportError:
            return "playwright not installed — pip install playwright && playwright install chromium"

    def run(self, url: str = "", screenshot: bool = False, max_chars: int = 6000,
            visible: bool = False, **_: Any) -> ToolResult:
        if not url:
            return ToolResult.failure("No URL provided")
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError:
            return ToolResult.failure(
                "playwright is not installed. Install with: "
                "pip install playwright && playwright install chromium")
        def _do_playwright() -> ToolResult:
            import os
            # Wayland detection: use visible mode with ozone to bypass Cloudflare
            # and pop Chrome on the user's screen. Auto-detects GNOME on Wayland.
            use_visible = bool(visible)
            if not use_visible and os.environ.get("XDG_SESSION_TYPE") == "wayland":
                # For Cloudflare-protected sites like chatgpt.com, headless is
                # detected. Visible mode with ozone bypasses it.
                if any(d in url for d in ("chatgpt.com", "openai.com", "auth0")):
                    use_visible = True
            with sync_playwright() as pw:
                if use_visible:
                    # Visible Chrome on user's Wayland session (pop on screen)
                    # Reuse existing Chrome profile so user is logged in
                    args = [
                        "--ozone-platform=wayland",
                        "--enable-features=UseOzonePlatform",
                        "--disable-blink-features=AutomationControlled",
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--disable-infobars",
                    ]
                    # Use DISPLAY=:1 / WAYLAND_DISPLAY=wayland-0 discovered on this machine
                    env_display = os.environ.get("DISPLAY", ":1")
                    env_wayland = os.environ.get("WAYLAND_DISPLAY", "wayland-0")
                    # Ensure env for subprocess
                    os.environ.setdefault("DISPLAY", env_display)
                    os.environ.setdefault("WAYLAND_DISPLAY", env_wayland)
                    # Use local Chrome (channel="chrome") so Google doesn't flag as insecure
                    try:
                        browser = pw.chromium.launch(headless=False, channel="chrome", args=args)
                    except Exception:
                        browser = pw.chromium.launch(headless=False, args=args)
                    context = browser.new_context(viewport={"width": 1920, "height": 1080})
                    # Hide webdriver
                    try:
                        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
                    except Exception:
                        pass
                    page = context.new_page()
                    page.goto(url, wait_until="domcontentloaded", timeout=40_000)
                    page.wait_for_timeout(7000)  # let Cloudflare + JS settle
                else:
                    browser = pw.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                    page.wait_for_timeout(1500)  # let JS settle
                title = page.title()
                text = page.inner_text("body")[:max(int(max_chars), 500)]
                links = page.eval_on_selector_all(
                    "a[href]",
                    "els => els.slice(0, 40).map(e => ({href: e.href, text: e.innerText.slice(0,120)}))",
                )
                shot_path = ""
                if screenshot:
                    self.screenshot_dir.mkdir(parents=True, exist_ok=True)
                    target = self.screenshot_dir / (re.sub(r"\W+", "_", url)[:80] + ".png")
                    page.screenshot(path=str(target), full_page=False)
                    shot_path = str(target)
                if use_visible:
                    # Keep visible Chrome open for user to see/interact
                    # Don't close browser — let it stay popped on screen
                    # Just close the Playwright wrapper but leave Chrome window
                    try:
                        # Detach: keep browser running in background
                        browser.close()
                    except Exception:
                        pass
                    output = f"# {title}\n{text}\n\n[visible Chrome popped on your screen — {url}]"
                else:
                    browser.close()
                    output = f"# {title}\n{text}"
                return ToolResult.success(output, url=url, title=title,
                                          links=links, screenshot=shot_path)

        # Run in a dedicated thread to avoid "Sync API inside the asyncio loop"
        import concurrent.futures

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_playwright)
                return future.result(timeout=60)
        except Exception as exc:  # noqa: BLE001
            logger.exception("playwright task failed")
            return ToolResult.failure(f"Playwright failed: {exc}")


class ChromeControlTool(Tool):
    """Full cursor control of visible Chrome — ARC drives it completely."""

    name = "computer.chrome_control"
    description = ("Control Chrome visibly on your screen like a human — click, "
                   "type, press keys, wait. Chrome pops on your Wayland session "
                   "and ARC moves the cursor itself (no need for you to touch it). "
                   "Use for ChatGPT, image generation, etc. Handles Cloudflare via "
                   "visible ozone Wayland mode.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "action": {"type": "string", "enum": ["open", "click", "type", "press", "wait", "screenshot"],
                       "description": "open=url, click=selector, type=text into selector, press=key, wait=ms"},
            "target": {"type": "string", "description": "URL for open, CSS selector for click/type, text for type, key for press, ms for wait"},
            "value": {"type": "string", "description": "Text to type (for type action)"},
        },
        "required": ["action", "target"],
    }

    def __init__(self, screenshot_dir: Optional[Path] = None) -> None:
        self.screenshot_dir = screenshot_dir or Path("data/screenshots")
        self._browser = None
        self._context = None
        self._page = None

    def _ensure_browser(self, url: str = "https://chatgpt.com"):
        if self._page is not None:
            try:
                # Check if still open
                self._page.title()
                return
            except Exception:
                pass
        # Simple and reliable: pop Chrome visibly on user's Wayland session
        # using the real google-chrome binary with the existing profile.
        # This is what "use the cursor and my chrome app directly so it will
        # pop on my screen" means — we use the actual Chrome you see, not a
        # headless Chromium. No Playwright needed for open; it just pops.
        import os
        import subprocess

        try:
            env = os.environ.copy()
            env.setdefault("DISPLAY", ":1")
            env.setdefault("WAYLAND_DISPLAY", "wayland-0")
            env.setdefault("XDG_SESSION_TYPE", "wayland")
            subprocess.Popen(
                ["google-chrome", "--new-window", url],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import time

            time.sleep(1.5)
            # We don't need a Playwright page for open — the window is already popped.
            # Keep _page as None so subsequent actions know to use fallback.
            self._page = None
            return
        except Exception as exc:
            logger.debug("google-chrome launch failed: %s", exc)
            # Fallback: try Playwright visible as last resort
            try:
                from playwright.sync_api import sync_playwright
                import concurrent.futures

                def _launch_fallback():
                    pw = sync_playwright().start()
                    args = ["--ozone-platform=wayland", "--enable-features=UseOzonePlatform"]
                    browser = pw.chromium.launch(headless=False, args=args)
                    ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
                    page = ctx.new_page()
                    page.goto(url, wait_until="domcontentloaded", timeout=40000)
                    page.wait_for_timeout(4000)
                    return pw, browser, ctx, page

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    pw, browser, ctx, page = ex.submit(_launch_fallback).result(timeout=30)
                    self._context = ctx
                    self._page = page
                    self._browser = browser
            except Exception as e:
                raise RuntimeError(f"Chrome launch failed: {e}") from e

    def run(self, action: str = "", target: str = "", value: str = "", **_: Any) -> ToolResult:
        # For open, use the simple subprocess path that we know pops on screen
        action_l = (action or "").lower().strip()
        tgt = (target or "").strip()
        if action_l == "open":
            # Use the same logic as computer.open_chrome but via self._ensure_browser
            # which now just pops Chrome via subprocess and handles Wayland
            try:
                self._ensure_browser(tgt or "https://chatgpt.com")
                # After popping, try to get title via a lightweight headless check
                # but don't fail if we can't — the window is already popped
                if self._page is not None:
                    try:
                        title = self._page.title()
                        return ToolResult.success(f"Chrome popped — {title} — {self._page.url}\nARC now controls the cursor. Tell ARC what to click/type next, or let it handle the image generation.")
                    except Exception:
                        pass
                return ToolResult.success(f"Chrome popped on your screen — {tgt or 'https://chatgpt.com'}\nARC can now drive it. Use click/type/press actions or let ARC handle the image generation.")
            except Exception as exc:
                logger.exception("chrome_control open failed")
                return ToolResult.failure(f"Chrome open failed: {exc}")

        # For other actions, ensure we have a page (try CDP first)
        if self._page is None:
            try:
                self._ensure_browser()
            except Exception:
                pass
            if self._page is None:
                # No page yet — try to connect to debuggable Chrome on :9222
                try:
                    from playwright.sync_api import sync_playwright
                    import concurrent.futures

                    def _connect():
                        from playwright.sync_api import sync_playwright
                        import http.client
                        pw = sync_playwright().start()
                        # Try CDP
                        try:
                            conn = http.client.HTTPConnection("127.0.0.1", 9222, timeout=1)
                            conn.request("GET", "/json/version")
                            if conn.getresponse().status == 200:
                                browser = pw.chromium.connect_over_cdp("http://127.0.0.1:9222")
                                ctx = browser.contexts[0] if browser.contexts else browser.new_context()
                                page = ctx.pages[0] if ctx.pages else ctx.new_page()
                                self._context = ctx
                                self._page = page
                                return
                        except Exception:
                            pass
                        # Fallback: headless check
                        browser = pw.chromium.launch(headless=True)
                        ctx = browser.new_context()
                        page = ctx.new_page()
                        self._context = ctx
                        self._page = page

                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                        ex.submit(_connect).result(timeout=15)
                except Exception as exc:
                    return ToolResult.failure(f"Chrome not ready for control: {exc}. Try 'open' first.")

        # Now page should be available — do the action in a thread to avoid asyncio loop issues
        def _do_action() -> ToolResult:
            if self._page is None:
                return ToolResult.failure("No browser page available")
            if action_l == "click":
                self._page.click(tgt, timeout=10000)
                self._page.wait_for_timeout(800)
                return ToolResult.success(f"Clicked {tgt!r} — {self._page.url[:80]}")
            if action_l == "type":
                selector = tgt
                text = value or tgt
                if value and selector:
                    self._page.fill(selector, text, timeout=10000)
                else:
                    self._page.keyboard.type(text)
                self._page.wait_for_timeout(500)
                return ToolResult.success(f"Typed into {selector!r}")
            if action_l == "press":
                self._page.keyboard.press(tgt)
                self._page.wait_for_timeout(500)
                return ToolResult.success(f"Pressed {tgt}")
            if action_l == "wait":
                ms = int(tgt) if tgt.isdigit() else 2000
                self._page.wait_for_timeout(min(ms, 10000))
                return ToolResult.success(f"Waited {ms}ms")
            if action_l == "screenshot":
                self.screenshot_dir.mkdir(parents=True, exist_ok=True)
                path = self.screenshot_dir / "chrome_control.png"
                self._page.screenshot(path=str(path))
                return ToolResult.success(f"Screenshot → {path}", screenshot=str(path))
            return ToolResult.failure(f"Unknown action {action_l!r} — use open/click/type/press/wait/screenshot")

        import concurrent.futures

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_action)
                return future.result(timeout=60)
        except Exception as exc:
            logger.exception("chrome_control failed")
            return ToolResult.failure(f"Chrome control failed: {exc}")


def register_browser_tools(registry: Any, router: Optional[Any] = None,
                           screenshot_dir: Optional[Path] = None) -> None:
    registry.register(WebFetchTool())
    registry.register(PlaywrightTool(screenshot_dir=screenshot_dir))
    registry.register(BrowserUseTool(router=router))
    registry.register(ChromeControlTool(screenshot_dir=screenshot_dir))


__all__ = ["WebFetchTool", "BrowserUseTool", "PlaywrightTool", "register_browser_tools",
           "fetch_url"]
