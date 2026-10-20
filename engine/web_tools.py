"""
Web tools for the autonomous agent: DuckDuckGo HTML search + page fetch/extract.

Uses only stdlib parsing (html.parser) plus httpx, so no API key or extra
dependency (beautifulsoup4/pypdf) is required. All network access is best-effort
and degrades gracefully to a structured error dict instead of raising.
"""
from __future__ import annotations

import html
import logging
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional

import httpx

from config import (
    WEB_SEARCH_ENABLED,
    WEB_SEARCH_MAX_RESULTS,
    WEB_FETCH_TIMEOUT,
    WEB_FETCH_MAX_CHARS,
)

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_DDG_HTML = "https://html.duckduckgo.com/html/"
_DDG_LITE = "https://lite.duckduckgo.com/lite/"
_DDG_IA = "https://api.duckduckgo.com/"
_BING = "https://www.bing.com/search"


class _TextExtractor(HTMLParser):
    """Collects visible text, skipping script/style/noscript and head content."""

    _SKIP = {"script", "style", "noscript", "head", "svg", "template"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._title_parts: List[str] = []
        self._in_title = False
        self.chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self._SKIP:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in self._BLOCK:
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_parts.append(data.strip())
        if self._skip_depth == 0 and data.strip():
            self.chunks.append(data)

    @property
    def title(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._title_parts)).strip()

    @property
    def text(self) -> str:
        raw = "".join(self.chunks)
        raw = re.sub(r"[ \t\u00a0]+", " ", raw)
        raw = re.sub(r"\n\s*\n\s*\n+", "\n\n", raw)
        return raw.strip()


def _clean_ddg_link(href: str) -> str:
    """DuckDuckGo wraps results as /l/?uddg=<encoded url>; unwrap to the real URL."""
    if not href:
        return ""
    m = re.search(r"[?&]uddg=([^&]+)", href)
    if m:
        from urllib.parse import unquote
        return unquote(m.group(1))
    if href.startswith("//"):
        return "https:" + href
    return href


class _DDGResultParser(HTMLParser):
    """Parses DuckDuckGo HTML/Lite result pages into (title, url, snippet) tuples."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: List[Dict[str, str]] = []
        self._in_link = False
        self._in_snippet = False
        self._link_href = ""
        self._title_parts: List[str] = []
        self._snippet_parts: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_d = {k.lower(): v for k, v in attrs}
        cls = (attrs_d.get("class") or "").lower()
        if tag == "a" and "result__a" in cls:
            self._in_link = True
            self._link_href = attrs_d.get("href", "")
            self._title_parts = []
        elif tag == "a" and "result-link" in cls:  # lite layout
            self._in_link = True
            self._link_href = attrs_d.get("href", "")
            self._title_parts = []
        if "result__snippet" in cls:
            self._in_snippet = True
            self._snippet_parts = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            self._in_link = False
            title = re.sub(r"\s+", " ", " ".join(self._title_parts)).strip()
            url = _clean_ddg_link(self._link_href)
            if title and url.startswith("http"):
                self.results.append({
                    "title": title,
                    "url": url,
                    "snippet": "",
                })
        if self._in_snippet and tag in ("a", "div", "td"):
            self._in_snippet = False
            snippet = re.sub(r"\s+", " ", " ".join(self._snippet_parts)).strip()
            if self.results and snippet and not self.results[-1]["snippet"]:
                self.results[-1]["snippet"] = snippet

    def handle_data(self, data: str) -> None:
        if self._in_link:
            self._title_parts.append(data)
        if self._in_snippet:
            self._snippet_parts.append(data)


def _decode_bing_url(href: str) -> str:
    """Bing wraps result links as /ck/a?...&u=a1<base64url>; decode the real target."""
    if not href:
        return ""
    m = re.search(r"[?&]u=a1([A-Za-z0-9_\-]+)", href)
    if m:
        import base64
        raw = m.group(1)
        pad = "=" * (-len(raw) % 4)
        try:
            return base64.urlsafe_b64decode(raw + pad).decode("utf-8", "ignore")
        except Exception:
            return href
    if href.startswith("//"):
        return "https:" + href
    return href


def _parse_bing(html_text: str, limit: int) -> List[Dict[str, str]]:
    """Extracts Bing organic results (b_algo blocks) via regex."""
    results: List[Dict[str, str]] = []
    blocks = re.findall(r'<li class="b_algo".*?(?=<li class="b_algo"|</ol>)', html_text, re.S)
    for block in blocks:
        m = re.search(r"<h2[^>]*>\s*<a[^>]*href=\"([^\"]+)\"[^>]*>(.*?)</a>", block, re.S)
        if not m:
            continue
        url = _decode_bing_url(html.unescape(m.group(1)))
        title = re.sub(r"<[^>]+>", "", m.group(2))
        title = re.sub(r"\s+", " ", html.unescape(title)).strip()
        snippet = ""
        sm = re.search(r'<p class="[^"]*b_lineclamp[^"]*"[^>]*>(.*?)</p>', block, re.S)
        if sm:
            snippet = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", sm.group(1)))).strip()
        if title and url.startswith("http"):
            results.append({"title": title, "url": url, "snippet": snippet})
        if len(results) >= limit:
            break
    return results


def _parse_ddg_instant_answer(data: Dict[str, Any], limit: int) -> List[Dict[str, str]]:
    """Flattens DDG Instant Answer API results into the common result shape."""
    results: List[Dict[str, str]] = []

    def add(text: str, url: str, snippet: str = "") -> None:
        if text and url:
            results.append({"title": text, "url": url, "snippet": snippet})

    if data.get("AbstractText") and data.get("AbstractURL"):
        add(data.get("Heading") or data.get("AbstractSource") or "Result", data["AbstractURL"], data["AbstractText"])

    def walk(topics) -> None:
        for t in topics or []:
            if len(results) >= limit:
                return
            if "Topics" in t:
                walk(t["Topics"])
            else:
                add(t.get("Text", "")[:120], t.get("FirstURL", ""), t.get("Text", "")[:200])

    walk(data.get("RelatedTopics"))
    if data.get("Results"):
        for r in data["Results"][:limit]:
            add(r.get("Text", "")[:120], r.get("FirstURL", ""), r.get("Text", "")[:200])
    return results[:limit]


async def _http_get(url: str, params: Dict[str, Any]) -> Optional[httpx.Response]:
    async with httpx.AsyncClient(
        headers=DEFAULT_HEADERS, follow_redirects=True, timeout=WEB_FETCH_TIMEOUT
    ) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp


async def web_search(query: str, max_results: Optional[int] = None) -> Dict[str, Any]:
    """Searches the web with graceful multi-backend fallback (no API key needed)."""
    if not WEB_SEARCH_ENABLED:
        return {"success": False, "error": "Web search disabled", "results": []}
    query = (query or "").strip()
    if not query:
        return {"success": False, "error": "Empty query", "results": []}
    limit = max_results or WEB_SEARCH_MAX_RESULTS
    errors: List[str] = []

    # 1) DuckDuckGo HTML / Lite endpoints
    for endpoint in (_DDG_HTML, _DDG_LITE):
        try:
            resp = await _http_get(endpoint, {"q": query})
            parser = _DDGResultParser()
            parser.feed(resp.text)
            results = parser.results[:limit]
            if results:
                return {"success": True, "query": query, "results": results, "source": "duckduckgo"}
        except Exception as e:
            errors.append(f"ddg:{e}")
        # POST form variant (some DDG deployments require it)
        try:
            async with httpx.AsyncClient(
                headers=DEFAULT_HEADERS, follow_redirects=True, timeout=WEB_FETCH_TIMEOUT
            ) as client:
                resp = await client.post(endpoint, data={"q": query})
                resp.raise_for_status()
            parser = _DDGResultParser()
            parser.feed(resp.text)
            if parser.results:
                return {"success": True, "query": query, "results": parser.results[:limit], "source": "duckduckgo"}
        except Exception as e:
            errors.append(f"ddg-post:{e}")

    # 2) Bing HTML (decode /ck/a redirect wrappers)
    try:
        resp = await _http_get(_BING, {"q": query})
        results = _parse_bing(resp.text, limit)
        if results:
            return {"success": True, "query": query, "results": results, "source": "bing"}
    except Exception as e:
        errors.append(f"bing:{e}")

    # 3) DuckDuckGo Instant Answer API (last resort, encyclopedic results)
    try:
        resp = await _http_get(_DDG_IA, {"q": query, "format": "json", "no_html": 1})
        results = _parse_ddg_instant_answer(resp.json(), limit)
        if results:
            return {"success": True, "query": query, "results": results, "source": "duckduckgo-ia"}
    except Exception as e:
        errors.append(f"ddg-ia:{e}")

    return {"success": False, "error": "; ".join(errors)[:400], "results": [], "query": query}


async def web_fetch(url: str, max_chars: Optional[int] = None) -> Dict[str, Any]:
    """Fetches a URL and extracts readable plain text.

    Redirects are followed manually so every hop is checked against the same
    public-host policy as the initial URL (prevents redirect-based SSRF).
    """
    url = (url or "").strip()
    if not url.startswith("http"):
        return {"success": False, "error": "Invalid URL", "url": url}
    limit = max_chars or WEB_FETCH_MAX_CHARS
    from urllib.parse import urljoin

    from services.builtin_tools import _allow_private_fetch, is_safe_public_url

    current = url
    try:
        async with httpx.AsyncClient(
            headers=DEFAULT_HEADERS, follow_redirects=False, timeout=WEB_FETCH_TIMEOUT
        ) as client:
            resp = None
            for _ in range(6):
                if not is_safe_public_url(current, allow_private=_allow_private_fetch()):
                    return {
                        "success": False,
                        "error": "URL blocked: only public http(s) hosts are allowed.",
                        "url": current,
                    }
                resp = await client.get(current)
                if resp.status_code not in (301, 302, 303, 307, 308):
                    break
                location = resp.headers.get("location")
                if not location:
                    break
                current = urljoin(current, location)
            else:
                return {"success": False, "error": "Too many redirects", "url": url}
            if resp is None:
                return {"success": False, "error": "No response", "url": url}
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            text = resp.text
        if "html" in content_type.lower() or "<html" in text[:2000].lower():
            parser = _TextExtractor()
            parser.feed(text)
            title = parser.title
            body = parser.text
        else:
            title = ""
            body = text
        truncated = len(body) > limit
        return {
            "success": True,
            "url": str(resp.url),
            "title": html.unescape(title)[:300],
            "content": body[:limit],
            "truncated": truncated,
            "length": len(body),
        }
    except Exception as e:
        logger.warning("Web fetch failed for %s: %s", url, e)
        return {"success": False, "error": str(e), "url": url}


async def web_research(query: str, fetch_top: int = 2) -> Dict[str, Any]:
    """Convenience: search then fetch the top N pages for richer context."""
    search = await web_search(query)
    if not search.get("success"):
        return search
    pages = []
    for res in search.get("results", [])[:fetch_top]:
        page = await web_fetch(res["url"])
        if page.get("success"):
            pages.append({"url": page["url"], "title": page.get("title") or res.get("title"), "content": page.get("content", "")[:2500]})
    return {**search, "pages": pages}
