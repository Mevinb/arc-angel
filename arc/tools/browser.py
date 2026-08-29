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

            # Browser-Use needs its own event loop; create one off the main thread.
            output = asyncio.run(_run())
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
        try:
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
                    args = ["--ozone-platform=wayland", "--enable-features=UseOzonePlatform"]
                    # Use DISPLAY=:1 / WAYLAND_DISPLAY=wayland-0 discovered on this machine
                    env_display = os.environ.get("DISPLAY", ":1")
                    env_wayland = os.environ.get("WAYLAND_DISPLAY", "wayland-0")
                    # Ensure env for subprocess
                    os.environ.setdefault("DISPLAY", env_display)
                    os.environ.setdefault("WAYLAND_DISPLAY", env_wayland)
                    browser = pw.chromium.launch(headless=False, args=args)
                    context = browser.new_context(viewport={"width": 1920, "height": 1080})
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
        # Launch visible Chrome on Wayland — pops on user's screen
        from playwright.sync_api import sync_playwright
        import os
        pw = sync_playwright().start()
        args = ["--ozone-platform=wayland", "--enable-features=UseOzonePlatform"]
        # Use user's existing profile so login persists
        user_data = os.path.expanduser("~/.config/google-chrome")
        # Try persistent context first (keeps login), fallback to regular
        try:
            self._context = pw.chromium.launch_persistent_context(
                user_data_dir=user_data,
                headless=False,
                args=args,
                viewport={"width": 1920, "height": 1080},
            )
            self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        except Exception:
            browser = pw.chromium.launch(headless=False, args=args)
            self._context = browser.new_context(viewport={"width": 1920, "height": 1080})
            self._page = self._context.new_page()
        self._page.goto(url, wait_until="domcontentloaded", timeout=40000)
        self._page.wait_for_timeout(7000)

    def run(self, action: str = "", target: str = "", value: str = "", **_: Any) -> ToolResult:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return ToolResult.failure("playwright not installed")
        try:
            action = (action or "").lower().strip()
            target = (target or "").strip()
            if action == "open":
                self._ensure_browser(target or "https://chatgpt.com")
                title = self._page.title()
                return ToolResult.success(f"Chrome popped — {title} — {self._page.url}\nARC now controls the cursor. Tell ARC what to click/type next, or let it handle the image generation.")
            if self._page is None:
                self._ensure_browser()
            if action == "click":
                self._page.click(target, timeout=10000)
                self._page.wait_for_timeout(800)
                return ToolResult.success(f"Clicked {target!r} — {self._page.url[:80]}")
            if action == "type":
                # target is selector, value is text
                selector = target
                text = value or target
                if value:
                    self._page.fill(selector, text, timeout=10000)
                else:
                    self._page.keyboard.type(text)
                self._page.wait_for_timeout(500)
                return ToolResult.success(f"Typed into {selector!r}")
            if action == "press":
                self._page.keyboard.press(target)
                self._page.wait_for_timeout(500)
                return ToolResult.success(f"Pressed {target}")
            if action == "wait":
                ms = int(target) if target.isdigit() else 2000
                self._page.wait_for_timeout(min(ms, 10000))
                return ToolResult.success(f"Waited {ms}ms")
            if action == "screenshot":
                self.screenshot_dir.mkdir(parents=True, exist_ok=True)
                path = self.screenshot_dir / "chrome_control.png"
                self._page.screenshot(path=str(path))
                return ToolResult.success(f"Screenshot → {path}", screenshot=str(path))
            return ToolResult.failure(f"Unknown action {action!r} — use open/click/type/press/wait/screenshot")
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
