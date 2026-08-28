"""Job sources for the internship engine (Phase 6).

Each source returns a normalized list of job dicts:
  {company, role, location, url, source, description, requirements, tags}

All sources use the stdlib fetcher (no API keys). They are best-effort: a
failing source logs a warning and returns [] so one bad API never kills a
search run.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from ..tools.browser import fetch_url

logger = logging.getLogger("jarvis.jobs")

Job = Dict[str, Any]

INTERNSHIP_KEYWORDS = [
    "intern", "internship", "trainee", "co-op", "coop", "fellowship",
    "junior", "graduate program", "graduate programme", "new grad", "entry level",
]

_CACHE: Dict[str, tuple[float, Any]] = {}
CACHE_TTL = 600.0  # 10 minutes — be polite to public APIs


def _cached(key: str, fetch: Callable[[], Any]) -> Any:
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < CACHE_TTL:
        return hit[1]
    value = fetch()
    _CACHE[key] = (now, value)
    return value


def _looks_like_internship(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in INTERNSHIP_KEYWORDS)


def _normalise(job: Job, source: str) -> Job:
    job.setdefault("company", "")
    job.setdefault("role", "")
    job.setdefault("location", "")
    job.setdefault("url", "")
    job.setdefault("description", "")
    job.setdefault("requirements", "")
    job.setdefault("tags", [])
    job["source"] = source
    return job


# --------------------------------------------------------------------- remoteok
def fetch_remoteok(max_results: int = 40, internship_only: bool = True) -> List[Job]:
    """RemoteOK public API — https://remoteok.com/api"""
    def _fetch() -> List[Job]:
        try:
            result = fetch_url("https://remoteok.com/api", timeout=20)
            data = json.loads(result["text"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("RemoteOK fetch failed: %s", exc)
            return []
        jobs: List[Job] = []
        for item in data:
            if not isinstance(item, dict) or "position" not in item:
                continue
            text = f"{item.get('position', '')} {item.get('description', '')[:500]}"
            if internship_only and not _looks_like_internship(text):
                continue
            jobs.append(_normalise({
                "company": item.get("company", ""),
                "role": item.get("position", ""),
                "location": item.get("location", "") or "Remote",
                "url": item.get("url", "") or item.get("apply_url", ""),
                "description": (item.get("description") or "")[:1200],
                "tags": item.get("tags", [])[:10],
            }, "remoteok"))
            if len(jobs) >= max_results:
                break
        return jobs

    return _cached(f"remoteok:{max_results}:{internship_only}", _fetch)


# -------------------------------------------------------------------- arbeitnow
def fetch_arbeitnow(max_results: int = 40, internship_only: bool = True) -> List[Job]:
    """Arbeitnow free job-board API — https://www.arbeitnow.com/api/job-board-api"""
    def _fetch() -> List[Job]:
        try:
            result = fetch_url("https://www.arbeitnow.com/api/job-board-api", timeout=20)
            data = json.loads(result["text"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Arbeitnow fetch failed: %s", exc)
            return []
        jobs: List[Job] = []
        for item in data.get("data", []):
            text = f"{item.get('title', '')} {item.get('description', '')[:400]}"
            if internship_only and not _looks_like_internship(text):
                continue
            remote = "Remote" if item.get("remote") else ""
            jobs.append(_normalise({
                "company": item.get("company_name", ""),
                "role": item.get("title", ""),
                "location": item.get("location", "") + (f" / {remote}" if remote else ""),
                "url": item.get("url", ""),
                "description": (item.get("description") or "")[:1200],
                "tags": item.get("tags", [])[:10],
            }, "arbeitnow"))
            if len(jobs) >= max_results:
                break
        return jobs

    return _cached(f"arbeitnow:{max_results}:{internship_only}", _fetch)


# ------------------------------------------------------------------- hackernews
def fetch_hackernews(max_results: int = 40, internship_only: bool = True) -> List[Job]:
    """Hacker News 'Who is hiring' threads via the Algolia API."""
    def _fetch() -> List[Job]:
        url = ("https://hn.algolia.com/api/v1/search_by_date?query="
               "%22who%20is%20hiring%22&tags=story&hitsPerPage=5")
        try:
            result = fetch_url(url, timeout=20)
            stories = json.loads(result["text"]).get("hits", [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("HN story search failed: %s", exc)
            return []
        jobs: List[Job] = []
        for story in stories[:2]:
            story_id = story.get("objectID")
            if not story_id:
                continue
            comment_url = (f"https://hn.algolia.com/api/v1/search?tags=comment,"
                           f"story_{story_id}&hitsPerPage=100")
            try:
                comments = json.loads(fetch_url(comment_url, timeout=25)["text"])
            except Exception as exc:  # noqa: BLE001
                logger.warning("HN comment fetch failed: %s", exc)
                continue
            for hit in comments.get("hits", []):
                text = hit.get("comment_text") or ""
                if internship_only and not _looks_like_internship(text):
                    continue
                # First line is typically "Company | Location | ROLE"
                first_line = text.split("<p>", 1)[0]
                clean = first_line.replace("\n", " ").strip()
                parts = [p.strip() for p in clean.split("|")]
                company = parts[0] if parts else ""
                role = parts[-1] if len(parts) > 1 else "Software Intern"
                jobs.append(_normalise({
                    "company": company[:80],
                    "role": role[:120],
                    "location": parts[1] if len(parts) > 2 else "Remote",
                    "url": f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                    "description": text[:1500],
                    "tags": ["hackernews"],
                }, "hackernews"))
                if len(jobs) >= max_results:
                    return jobs
        return jobs

    return _cached(f"hackernews:{max_results}:{internship_only}", _fetch)


# ----------------------------------------------------------------------- manual
def manual_job(company: str, role: str, url: str = "", location: str = "",
               description: str = "") -> Job:
    """Create a job record manually (e.g. pasted by the user or found by the
    browser agent)."""
    return _normalise({
        "company": company, "role": role, "url": url,
        "location": location, "description": description,
    }, "manual")


SOURCES: Dict[str, Callable[..., List[Job]]] = {
    "remoteok": fetch_remoteok,
    "arbeitnow": fetch_arbeitnow,
    "hackernews": fetch_hackernews,
}


def fetch_all(source_names: Optional[List[str]] = None, max_per_source: int = 40,
              internship_only: bool = True) -> List[Job]:
    """Aggregate jobs from all configured sources, de-duplicated."""
    names = source_names or list(SOURCES)
    collected: List[Job] = []
    seen: set = set()
    for name in names:
        fetcher = SOURCES.get(name)
        if fetcher is None:
            logger.warning("Unknown job source: %s", name)
            continue
        try:
            jobs = fetcher(max_results=max_per_source, internship_only=internship_only)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Source %s failed: %s", name, exc)
            continue
        for job in jobs:
            key = (job["company"].lower(), job["role"].lower(), job["url"])
            if key in seen:
                continue
            seen.add(key)
            collected.append(job)
    logger.info("Fetched %d unique jobs from %d sources", len(collected), len(names))
    return collected
