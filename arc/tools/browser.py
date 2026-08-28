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
            from browser_use.browser.browser import Browser, BrowserConfig  # type: ignore

            async def _run() -> str:
                client = self.router.client  # shared OpenAI-compatible client
                from openai import AsyncOpenAI
                async_client = AsyncOpenAI(
                    base_url=self.router.config.base_url,
                    api_key=self.router.config.api_key or "missing-key",
                    timeout=self.router.config.timeout_seconds,
                )
                agent = Agent(
                    task=task,
                    llm=async_client,
                    browser=Browser(config=BrowserConfig(headless=True)),
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
            **_: Any) -> ToolResult:
        if not url:
            return ToolResult.failure("No URL provided")
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError:
            return ToolResult.failure(
                "playwright is not installed. Install with: "
                "pip install playwright && playwright install chromium")
        try:
            with sync_playwright() as pw:
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
                browser.close()
            output = f"# {title}\n{text}"
            return ToolResult.success(output, url=url, title=title,
                                      links=links, screenshot=shot_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("playwright task failed")
            return ToolResult.failure(f"Playwright failed: {exc}")


def register_browser_tools(registry: Any, router: Optional[Any] = None,
                           screenshot_dir: Optional[Path] = None) -> None:
    registry.register(WebFetchTool())
    registry.register(PlaywrightTool(screenshot_dir=screenshot_dir))
    registry.register(BrowserUseTool(router=router))


__all__ = ["WebFetchTool", "BrowserUseTool", "PlaywrightTool", "register_browser_tools",
           "fetch_url"]
