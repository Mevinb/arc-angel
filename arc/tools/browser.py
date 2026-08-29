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
                   "web.fetch cannot read. For chatgpt.com/login use visible=true "
                   "to pop your logged-in Chrome via ChromeManager.")
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
        if not re.match(r"^https?://", url):
            url = "https://" + url
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
        except ImportError:
            return ToolResult.failure(
                "playwright is not installed. Install with: "
                "pip install playwright && playwright install chromium")

        # Visible ChatGPT/login: delegate to singleton ChromeManager (no duplicate windows)
        import os as _os
        use_visible = bool(visible)
        if not use_visible and _os.environ.get("XDG_SESSION_TYPE") == "wayland":
            if any(d in url for d in ("chatgpt.com", "openai.com", "auth0")):
                use_visible = True

        if use_visible and any(d in url for d in ("chatgpt.com", "openai.com")):
            try:
                from .chrome_manager import ChromeManager

                page = ChromeManager.instance().ensure_visible(url)
                # Extract text/links on the manager's dedicated Playwright thread
                def _extract(p):
                    title = p.title()
                    text = p.inner_text("body")[:max(int(max_chars), 500)]
                    links = p.eval_on_selector_all(
                        "a[href]",
                        "els => els.slice(0, 40).map(e => ({href: e.href, text: e.innerText.slice(0,120)}))",
                    )
                    # Also extract image URLs (for ChatGPT image gen, cdn.oaistatic.com etc.)
                    try:
                        images = p.eval_on_selector_all(
                            "img[src]",
                            "els => els.slice(0, 20).map(e => e.src).filter(src => src.includes('oai') || src.includes('cdn') || src.includes('files.') || src.includes('blob:') || src.match(/\\.(png|jpg|jpeg|webp)/i))",
                        )
                    except Exception:
                        images = []
                    shot_path = ""
                    if screenshot:
                        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
                        target = self.screenshot_dir / (re.sub(r"\W+", "_", url)[:80] + ".png")
                        p.screenshot(path=str(target), full_page=False)
                        shot_path = str(target)
                    return title, text, links, images, shot_path

                mgr = ChromeManager.instance()
                title, text, links, images, shot_path = mgr.do(_extract)  # type: ignore[arg-type]
                output = f"# {title}\n{text}\n\n[visible Chrome — {url} — same window reused, logged-in]"
                if images:
                    output += f"\n\n[images: {', '.join(images[:5])}]"
                return ToolResult.success(output, url=url, title=title, links=links, images=images, screenshot=shot_path)
            except Exception as exc:
                logger.debug("ChromeManager visible failed, falling back to headless: %s", exc)

        # Headless path — runs on a dedicated thread to avoid greenlet/asyncio clash
        def _do_headless() -> ToolResult:
            from playwright.sync_api import sync_playwright as _sp
            with _sp() as pw:
                browser = pw.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_timeout(1500)
                title = page.title()
                text = page.inner_text("body")[:max(int(max_chars), 500)]
                links = page.eval_on_selector_all(
                    "a[href]",
                    "els => els.slice(0, 40).map(e => ({href: e.href, text: e.innerText.slice(0,120)}))",
                )
                try:
                    images = page.eval_on_selector_all(
                        "img[src]",
                        "els => els.slice(0, 20).map(e => e.src).filter(src => src.includes('oai') || src.includes('cdn') || src.includes('files.') || src.match(/\\.(png|jpg|jpeg|webp)/i))",
                    )
                except Exception:
                    images = []
                shot_path = ""
                if screenshot:
                    self.screenshot_dir.mkdir(parents=True, exist_ok=True)
                    target = self.screenshot_dir / (re.sub(r"\W+", "_", url)[:80] + ".png")
                    page.screenshot(path=str(target), full_page=False)
                    shot_path = str(target)
                browser.close()
                header = f"# {title}\n{text}"
                if images:
                    header += f"\n\n[images: {', '.join(images[:5])}]"
                return ToolResult.success(header, url=url, title=title, links=links, images=images, screenshot=shot_path)

        import concurrent.futures

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_headless)
                return future.result(timeout=60)
        except Exception as exc:  # noqa: BLE001
            logger.exception("playwright task failed")
            return ToolResult.failure(f"Playwright failed: {exc}")


class ChromeControlTool(Tool):
    """Full cursor control of visible Chrome — ARC drives it completely.

    Uses ChromeManager singleton so all actions share one Playwright thread
    and one visible window (no per-call `google-chrome --new-window` spam).
    Supports both browser-DOM actions (click/type/press via Playwright) and
    Wayland-native OS actions (move/type_system/scroll via wtype/ydotool)
    so the system can use both typing and cursor movement.
    """

    name = "computer.chrome_control"
    description = ("Control Chrome visibly on your screen like a human — click, "
                   "type, press keys, move cursor, wait. Chrome pops on your "
                   "Wayland session and ARC moves the cursor itself (no need for "
                   "you to touch it). Uses ChromeManager singleton (one window). "
                   "Also supports OS-level typing/movement via wtype/ydotool on Wayland.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "action": {"type": "string", "enum": ["open", "click", "type", "press", "wait", "screenshot",
                                                  "move", "type_system", "scroll"],
                       "description": "open=url, click=selector, type=text into selector, press=key, wait=ms, move='x,y', type_system=text to type at OS level, scroll=direction"},
            "target": {"type": "string", "description": "URL for open, CSS selector for click/type, text for type, key for press, ms for wait, 'x,y' for move, text for type_system, direction for scroll"},
            "value": {"type": "string", "description": "Text to type (for type action)"},
        },
        "required": ["action", "target"],
    }

    def __init__(self, screenshot_dir: Optional[Path] = None) -> None:
        self.screenshot_dir = screenshot_dir or Path("data/screenshots")

    def availability(self) -> str:
        try:
            from .wayland_input import availability as _wa

            return f"playwright available; {_wa()}"
        except Exception:
            return "playwright available"

    def run(self, action: str = "", target: str = "", value: str = "", **_: Any) -> ToolResult:
        from .chrome_manager import ChromeManager

        action_l = (action or "").lower().strip()
        tgt = (target or "").strip()
        mgr = ChromeManager.instance()

        # ----- OS-level Wayland actions (no page needed) -----
        if action_l == "type_system":
            try:
                from .wayland_input import type_text as _type_text

                text = value or tgt
                ok, msg = _type_text(text)
                if ok:
                    return ToolResult.success(msg)
                # Fallback: try Playwright keyboard.type via manager page
                try:
                    mgr.do(lambda p: p.keyboard.type(text))
                    return ToolResult.success(f"Typed {len(text)} chars via browser fallback")
                except Exception:
                    return ToolResult.failure(msg)
            except Exception as exc:
                return ToolResult.failure(f"type_system failed: {exc}")

        if action_l == "move":
            try:
                from .wayland_input import move_click as _move_click

                # target "x,y" or "x,y:button"
                parts = tgt.replace(":", ",").split(",")
                if len(parts) < 2:
                    return ToolResult.failure("move requires 'x,y' e.g. '500,300'")
                x, y = int(parts[0].strip()), int(parts[1].strip())
                button = parts[2].strip() if len(parts) > 2 else "left"
                ok, msg = _move_click(x, y, button)
                if ok:
                    return ToolResult.success(msg + " (OS cursor)")
                # Fallback: browser synthetic mouse (visible in-page, not OS, but proves ChromeControl works)
                try:
                    mgr.do(lambda p: (p.mouse.move(x, y), p.mouse.click(x, y))[1])
                    return ToolResult.success(
                        f"Moved to ({x},{y}) and clicked {button} via browser fallback (ydotool not available: {msg}). Install ydotool for OS cursor."
                    )
                except Exception:
                    return ToolResult.failure(msg)
            except Exception as exc:
                return ToolResult.failure(f"move failed: {exc}")

        if action_l == "scroll":
            try:
                from .wayland_input import scroll as _scroll

                direction = tgt.lower() if tgt.lower() in ("up", "down", "left", "right") else "down"
                amount = int(value) if value and value.isdigit() else 3
                ok, msg = _scroll(direction, amount)
                if ok:
                    return ToolResult.success(msg)
                # Fallback: wheel via Playwright
                try:
                    mgr.do(lambda p: p.mouse.wheel(0, 300 if direction == "down" else -300))
                    return ToolResult.success(f"Scrolled {direction} via browser")
                except Exception:
                    return ToolResult.failure(msg)
            except Exception as exc:
                return ToolResult.failure(f"scroll failed: {exc}")

        # ----- Browser actions via ChromeManager (shared thread, no greenlet error) -----
        if action_l == "open":
            try:
                page = mgr.ensure_visible(tgt or "https://chatgpt.com")
                title = mgr.do(lambda p: p.title())
                cur = mgr.do(lambda p: p.url)
                return ToolResult.success(f"Chrome ready — {title} — {cur}\nSame window reused; ARC now controls cursor/typing. Use click/type/press/move/type_system as needed.",
                                          url=cur, title=title)
            except Exception as exc:
                logger.exception("chrome_control open failed")
                return ToolResult.failure(f"Chrome open failed: {exc}")

        # For non-open, ensure a page exists (lazy)
        try:
            # This also handles idempotent reuse
            mgr.ensure_visible()
        except Exception as exc:
            return ToolResult.failure(f"Chrome not ready: {exc}. Try 'open' first.")

        def _do_action(page) -> ToolResult:  # runs on manager thread
            # If we're on dummy (real Chrome without CDP), Playwright ops will raise "CDP disabled" — fallback to OS
            is_dummy = page.__class__.__name__ == "_DummyPage" or "CDP disabled" in str(getattr(page, "title", lambda: "")())
            if action_l == "click":
                # Support "selector" or "x,y" — for x,y also warp OS cursor when possible (dual)
                if "," in tgt and tgt.replace(",", "").replace(" ", "").isdigit():
                    x_str, y_str = [s.strip() for s in tgt.split(",")][:2]
                    x_i, y_i = int(x_str), int(y_str)
                    # Best-effort OS warp (don't fail if missing)
                    try:
                        from .wayland_input import move_click as _mc

                        _mc(x_i, y_i, "left")
                    except Exception:
                        pass
                    try:
                        page.mouse.click(x_i, y_i)
                    except Exception as exc:
                        if "CDP disabled" in str(exc):
                            # Already did OS click via _mc, so success even if browser click failed
                            return ToolResult.success(f"Clicked at {tgt!r} via OS (CDP disabled, your real Chrome)")
                        raise
                    try:
                        page.wait_for_timeout(800)
                    except Exception:
                        pass
                    return ToolResult.success(f"Clicked at {tgt!r} — {page.url[:80]} (browser + OS best-effort)")
                # Selector click — try Playwright, fallback to instruction if dummy
                try:
                    page.click(tgt, timeout=10000)
                    try:
                        page.wait_for_timeout(800)
                    except Exception:
                        pass
                    return ToolResult.success(f"Clicked {tgt!r} — {page.url[:80]}")
                except Exception as exc:
                    if is_dummy or "CDP disabled" in str(exc):
                        return ToolResult.failure(
                            f"Click {tgt!r} needs CDP. Your Chrome runs without --remote-debugging-port, so I used your real window via xdg-open but can't click selector without CDP. "
                            f"Fix: restart Chrome: google-chrome --remote-debugging-port={mgr._cdp_port} --user-data-dir={mgr._user_data_dir} --ozone-platform-hint=auto  — then retry. "
                            f"Workaround: use click with coordinates e.g. '800,400' via wayland."
                        )
                    raise
            if action_l == "type":
                selector = tgt
                text = value or tgt
                # If dummy/CDP disabled, directly use OS typing (your real window is focused via bring_to_front/xdg-open)
                if is_dummy:
                    try:
                        from .wayland_input import type_text as _tt
                        ok, msg = _tt(text)
                        if ok:
                            return ToolResult.success(f"Typed {len(text)} chars into {selector!r} via OS ({msg}) — your real Chrome")
                        return ToolResult.failure(f"OS typing failed: {msg}. Need wtype: sudo apt install wtype")
                    except Exception as exc:
                        return ToolResult.failure(f"Type fallback failed: {exc}")
                if value and selector and not selector.isdigit():
                    try:
                        page.fill(selector, text, timeout=8000)
                    except Exception as exc:
                        if "CDP disabled" in str(exc):
                            # Fallback to OS
                            try:
                                from .wayland_input import type_text as _tt
                                ok, msg = _tt(text)
                                if ok:
                                    return ToolResult.success(f"Typed {len(text)} chars via OS ({msg})")
                            except Exception:
                                pass
                            return ToolResult.failure(f"Fill {selector!r} needs CDP (your Chrome lacks --remote-debugging-port). Use type_system instead or restart Chrome with CDP.")
                        # Fallback: focus then type
                        try:
                            page.focus(selector, timeout=5000)
                        except Exception:
                            pass
                        try:
                            page.keyboard.type(text)
                        except Exception as exc2:
                            if "CDP disabled" in str(exc2):
                                from .wayland_input import type_text as _tt
                                ok, msg = _tt(text)
                                if ok:
                                    return ToolResult.success(f"Typed {len(text)} chars via OS ({msg})")
                                raise
                            raise
                else:
                    try:
                        page.keyboard.type(text)
                    except Exception as exc:
                        if "CDP disabled" in str(exc):
                            from .wayland_input import type_text as _tt
                            ok, msg = _tt(text)
                            if ok:
                                return ToolResult.success(f"Typed {len(text)} chars via OS ({msg})")
                            raise
                        raise
                try:
                    page.wait_for_timeout(500)
                except Exception:
                    pass
                return ToolResult.success(f"Typed {len(text)} chars into {selector!r}")
            if action_l == "press":
                try:
                    page.keyboard.press(tgt)
                    try:
                        page.wait_for_timeout(500)
                    except Exception:
                        pass
                    return ToolResult.success(f"Pressed {tgt}")
                except Exception as exc:
                    if is_dummy or "CDP disabled" in str(exc):
                        # Try OS-level via wtype -k if available, else instruct
                        try:
                            import subprocess, shutil
                            if shutil.which("wtype"):
                                # wtype handles Enter etc. via typing newline for Enter
                                if tgt.lower() == "enter":
                                    from .wayland_input import type_text as _tt
                                    ok, msg = _tt("\n")
                                    if ok:
                                        return ToolResult.success(f"Pressed {tgt} via OS ({msg})")
                                else:
                                    # Try wtype -k (best-effort)
                                    subprocess.run(["wtype", "-k", tgt], timeout=5)
                                    return ToolResult.success(f"Pressed {tgt} via wtype")
                        except Exception:
                            pass
                        return ToolResult.failure(
                            f"Press {tgt!r} needs CDP. Your Chrome lacks --remote-debugging-port. "
                            f"Fix: google-chrome --remote-debugging-port={mgr._cdp_port} --user-data-dir={mgr._user_data_dir} — then retry. Workaround: use ydotool key."
                        )
                    raise
            if action_l == "wait":
                ms = int(tgt) if tgt.isdigit() else 2000
                page.wait_for_timeout(min(ms, 10000))
                return ToolResult.success(f"Waited {ms}ms")
            if action_l == "screenshot":
                self.screenshot_dir.mkdir(parents=True, exist_ok=True)
                path = self.screenshot_dir / "chrome_control.png"
                page.screenshot(path=str(path))
                # Try to extract image URLs for download hint
                try:
                    images = page.eval_on_selector_all(
                        "img[src]",
                        "els => els.slice(0, 10).map(e => e.src).filter(src => src.includes('oai') || src.includes('cdn') || src.includes('files.') || src.match(/\\.(png|jpg|jpeg|webp)/i))",
                    )
                    if images:
                        return ToolResult.success(f"Screenshot → {path}\n[images: {', '.join(images[:3])}] Use file.download to save to data/downloads/images", screenshot=str(path), images=images)
                except Exception:
                    pass
                return ToolResult.success(f"Screenshot → {path}", screenshot=str(path))
            return ToolResult.failure(f"Unknown action {action_l!r} — use open/click/type/press/wait/screenshot/move/type_system/scroll")

        try:
            return mgr.do(_do_action)
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
