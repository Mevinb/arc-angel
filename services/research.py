"""Source-grounded research projects with citations and claim tracking."""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from engine.web_tools import web_fetch, web_research
from memory.database import db_session
from services.audit import AuditService
from services.crypto import redact_text
from services.knowledge import KnowledgeService
from services.models import ModelService, ModelUnavailable
from services.safety import SafetyService


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchService:
    _jobs: Dict[str, asyncio.Task] = {}
    _lock = asyncio.Lock()

    @classmethod
    def create(
        cls, question: str, *, title: Optional[str] = None, max_sources: int = 8,
        profile: str = "quality", task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        question = (question or "").strip()
        if not question:
            raise ValueError("A research question is required")
        project_id = f"rs_{uuid.uuid4().hex[:12]}"
        now = _now()
        with db_session() as conn:
            conn.execute(
                """
                INSERT INTO research_projects (id, title, question, status, task_id, report, bibliography, created_at, updated_at)
                VALUES (?, ?, ?, 'draft', ?, '', '[]', ?, ?)
                """,
                (project_id, (title or question)[:300], question, task_id, now, now),
            )
        AuditService.log(
            "research", "research.created", actor="user", task_id=task_id,
            summary=f"Research project created: {(title or question)[:160]}",
            data={"project_id": project_id, "question": question, "max_sources": max_sources},
        )
        project = cls.get(project_id) or {}
        project["max_sources"] = max_sources
        project["profile"] = profile
        return project

    @classmethod
    def get(cls, project_id: str) -> Optional[Dict[str, Any]]:
        with db_session() as conn:
            row = conn.execute("SELECT * FROM research_projects WHERE id=?", (project_id,)).fetchone()
            if not row:
                return None
            notes = conn.execute(
                "SELECT * FROM research_notes WHERE project_id=? ORDER BY id", (project_id,)
            ).fetchall()
            claims = conn.execute(
                "SELECT * FROM research_claims WHERE project_id=? ORDER BY id", (project_id,)
            ).fetchall()
        return {
            "id": row["id"], "title": row["title"], "question": row["question"],
            "status": row["status"], "task_id": row["task_id"],
            "report": row["report"], "bibliography": cls._load(row["bibliography"], []),
            "notes": [cls._note_row(n) for n in notes],
            "claims": [cls._claim_row(c) for c in claims],
            "created_at": row["created_at"], "updated_at": row["updated_at"],
        }

    @classmethod
    def list(cls, limit: int = 50) -> List[Dict[str, Any]]:
        with db_session() as conn:
            rows = conn.execute(
                "SELECT id, title, question, status, task_id, created_at, updated_at FROM research_projects ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 200)),),
            ).fetchall()
        return [dict(r) for r in rows]

    @classmethod
    def delete(cls, project_id: str) -> bool:
        if not cls.get(project_id):
            return False
        with db_session() as conn:
            conn.execute("DELETE FROM research_notes WHERE project_id=?", (project_id,))
            conn.execute("DELETE FROM research_claims WHERE project_id=?", (project_id,))
            conn.execute("DELETE FROM research_projects WHERE id=?", (project_id,))
        AuditService.log("research", "research.deleted", actor="user", summary=f"Research project deleted: {project_id}")
        return True

    @classmethod
    def add_note(
        cls, project_id: str, *, kind: str, title: str, content: str,
        url: Optional[str] = None, credibility: float = 0.6,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not cls.get(project_id):
            raise ValueError("Research project not found")
        with db_session() as conn:
            cur = conn.execute(
                """
                INSERT INTO research_notes (project_id, kind, title, content, url, credibility, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id, kind[:40], redact_text(title)[:300], content[:200000],
                    url, float(credibility), json.dumps(metadata or {}), _now(),
                ),
            )
            note_id = cur.lastrowid
            conn.execute("UPDATE research_projects SET updated_at=? WHERE id=?", (_now(), project_id))
            row = conn.execute("SELECT * FROM research_notes WHERE id=?", (note_id,)).fetchone()
        return cls._note_row(row)

    @classmethod
    def start(cls, project_id: str, *, max_sources: int = 8, profile: str = "quality") -> Dict[str, Any]:
        project = cls.get(project_id)
        if not project:
            raise ValueError("Research project not found")
        existing = cls._jobs.get(project_id)
        if existing and not existing.done():
            return {"status": "running", "project": project}
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError("Research.start must be called from an async context")
        task = loop.create_task(cls._run(project_id, max_sources=max_sources, profile=profile))
        cls._jobs[project_id] = task
        task.add_done_callback(lambda _t, pid=project_id: cls._jobs.pop(pid, None))
        return {"status": "started", "project": cls.get(project_id)}

    @classmethod
    async def wait(cls, project_id: str, timeout: float = 180.0) -> Optional[Dict[str, Any]]:
        task = cls._jobs.get(project_id)
        if task:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        return cls.get(project_id)

    @classmethod
    async def _run(cls, project_id: str, *, max_sources: int, profile: str) -> Dict[str, Any]:
        if SafetyService.offline_mode():
            return cls._fail(project_id, "Offline mode is enabled; research requires network access.")
        project = cls.get(project_id)
        if not project:
            return {}
        with db_session() as conn:
            conn.execute("UPDATE research_projects SET status='running', updated_at=? WHERE id=?", (_now(), project_id))
        AuditService.log(
            "research", "research.started", actor="agent",
            summary=f"Research started: {project['title']}", data={"project_id": project_id},
            activity_title=f"Research started: {project['title']}",
        )
        try:
            queries = await cls._plan_queries(project["question"], profile)
            sources: Dict[str, Dict[str, Any]] = {}
            for query in queries[:5]:
                result = await web_research(query, fetch_top=2)
                if not result.get("success"):
                    continue
                for item in result.get("results", []):
                    url = item.get("url")
                    if not url or url in sources:
                        continue
                    sources[url] = {
                        "title": item.get("title") or url, "url": url,
                        "snippet": item.get("snippet") or "", "content": "",
                    }
                for page in result.get("pages", []):
                    url = page.get("url")
                    if not url:
                        continue
                    entry = sources.setdefault(url, {"title": page.get("title") or url, "url": url, "snippet": ""})
                    entry["content"] = page.get("content", "")[:6000]
                    if page.get("title"):
                        entry["title"] = page["title"]
            selected = list(sources.values())[:max(1, min(max_sources, 20))]
            for idx, source in enumerate(selected, 1):
                if not source.get("content"):
                    page = await web_fetch(source["url"], max_chars=5000)
                    if page.get("success"):
                        source["content"] = page.get("content", "")
                        source["title"] = source.get("title") or page.get("title") or source["url"]
                cls.add_note(
                    project_id, kind="source", title=source["title"],
                    content=source.get("content") or source.get("snippet") or "",
                    url=source["url"],
                    credibility=cls._credibility(source["url"]),
                    metadata={"index": idx},
                )
            if not selected:
                return cls._finish(project_id, "No reliable sources were found. Try rephrasing the question or adding sources manually.", [])
            synthesis = await cls._synthesize(project["question"], selected, profile)
            report = synthesis.get("report") or cls._fallback_report(project["question"], selected)
            claims = synthesis.get("claims") or []
            with db_session() as conn:
                for claim in claims[:30]:
                    if not isinstance(claim, dict) or not claim.get("claim"):
                        continue
                    support = claim.get("support") or ""
                    conn.execute(
                        """
                        INSERT INTO research_claims (project_id, claim, status, support, source_url, conflict, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            project_id, str(claim["claim"])[:2000], claim.get("status") or "supported",
                            str(support)[:4000], claim.get("source_url"), claim.get("conflict"), _now(),
                        ),
                    )
                    if support:
                        cls.add_note(project_id, kind="claim", title=str(claim["claim"])[:300], content=str(support)[:8000])
            bibliography = [
                {"index": i, "title": s["title"], "url": s["url"], "credibility": cls._credibility(s["url"])}
                for i, s in enumerate(selected, 1)
            ]
            result = cls._finish(project_id, report, bibliography)
            try:
                KnowledgeService.add_text(
                    title=f"Research: {project['title']}",
                    content=report + "\n\n## Sources\n" + "\n".join(
                        f"[{b['index']}] {b['title']} — {b['url']}" for b in bibliography
                    ),
                    source_type="research", uri=f"research://{project_id}",
                    tags=["research"], metadata={"project_id": project_id, "bibliography": bibliography},
                    credibility=0.7,
                )
            except Exception:
                pass
            return result
        except Exception as exc:
            return cls._fail(project_id, f"{type(exc).__name__}: {exc}")

    @classmethod
    async def _plan_queries(cls, question: str, profile: str) -> List[str]:
        prompt = (
            "Create 2-4 focused web search queries to answer the research question. "
            'Return JSON only: {"queries": ["...", "..."]}. Include one query that seeks '
            "contrary evidence or limitations.\n\nQuestion: " + question
        )
        try:
            raw = await ModelService.complete(
                [{"role": "user", "content": prompt}], profile="fast", task_type="research",
                max_tokens=250, temperature=0.2, timeout=40,
            )
            parsed = cls._json(raw)
            queries = parsed.get("queries") if isinstance(parsed, dict) else None
            if isinstance(queries, list):
                cleaned = [str(q).strip()[:300] for q in queries if str(q).strip()][:4]
                if cleaned:
                    return cleaned
        except Exception:
            pass
        return [question, f"{question} evidence", f"{question} limitations criticism"]

    @classmethod
    async def _synthesize(cls, question: str, sources: List[Dict[str, Any]], profile: str) -> Dict[str, Any]:
        corpus = []
        for idx, source in enumerate(sources, 1):
            body = (source.get("content") or source.get("snippet") or "")[:4000]
            corpus.append(f"[{idx}] {source['title']}\nURL: {source['url']}\n{body}")
        prompt = (
            "Write a careful research brief that answers the question using ONLY the numbered sources below. "
            "Cite every non-obvious claim with [n]. Distinguish evidence from inference, note conflicts, and end with "
            "a short 'Bottom line'. Return JSON only: "
            '{"report": "markdown report", "claims": [{"claim": "...", "status": "supported|mixed|unsupported", '
            '"support": "...", "source_url": "...", "conflict": null}]}.\n\n'
            f"Question: {question}\n\nSources:\n" + "\n\n".join(corpus)[:50000]
        )
        try:
            raw = await ModelService.complete(
                [{"role": "user", "content": prompt}], profile=profile,
                task_type="research", max_tokens=2600, temperature=0.2, timeout=180,
            )
            parsed = cls._json(raw)
            if isinstance(parsed, dict) and parsed.get("report"):
                return parsed
            return {"report": raw.strip(), "claims": []}
        except (ModelUnavailable, Exception):
            return {"report": cls._fallback_report(question, sources), "claims": []}

    @classmethod
    def _finish(cls, project_id: str, report: str, bibliography: List[Dict[str, Any]]) -> Dict[str, Any]:
        with db_session() as conn:
            conn.execute(
                "UPDATE research_projects SET status='completed', report=?, bibliography=?, updated_at=? WHERE id=?",
                (report[:200000], json.dumps(bibliography), _now(), project_id),
            )
        AuditService.log(
            "research", "research.completed", actor="agent", summary=f"Research completed: {project_id}",
            data={"project_id": project_id, "sources": len(bibliography), "chars": len(report)},
            activity_title="Research completed",
        )
        return cls.get(project_id) or {}

    @classmethod
    def _fail(cls, project_id: str, error: str) -> Dict[str, Any]:
        with db_session() as conn:
            conn.execute(
                "UPDATE research_projects SET status='failed', report=?, updated_at=? WHERE id=?",
                (redact_text(error)[:4000], _now(), project_id),
            )
        AuditService.log(
            "research", "research.failed", actor="agent", status="error",
            summary=f"Research failed: {error[:300]}", data={"project_id": project_id},
            activity_title="Research failed",
        )
        return cls.get(project_id) or {}

    @classmethod
    def _fallback_report(cls, question: str, sources: List[Dict[str, Any]]) -> str:
        lines = [f"# Research brief: {question}", "", "## Sources reviewed", ""]
        for idx, source in enumerate(sources, 1):
            snippet = re.sub(r"\s+", " ", (source.get("snippet") or source.get("content") or ""))[:500]
            lines.append(f"- [{idx}] [{source['title']}]({source['url']}) — {snippet}")
        lines += ["", "## Note", "A synthesis model was unavailable, so this is an extractive brief. "
                  "Open the sources above to verify claims."]
        return "\n".join(lines)

    @classmethod
    def _credibility(cls, url: str) -> float:
        host = (url or "").lower()
        if any(h in host for h in (".gov", ".edu", "arxiv.org", "nature.com", "science.org", "who.int")):
            return 0.9
        if any(h in host for h in ("reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org")):
            return 0.8
        if any(h in host for h in ("wikipedia.org", "github.com", "docs.")):
            return 0.7
        if any(h in host for h in ("medium.com", "substack.com", "reddit.com", "x.com", "twitter.com")):
            return 0.35
        return 0.55

    @classmethod
    def _note_row(cls, row) -> Dict[str, Any]:
        try:
            meta = json.loads(row["metadata"] or "{}")
        except Exception:
            meta = {}
        return {
            "id": row["id"], "project_id": row["project_id"], "kind": row["kind"],
            "title": row["title"], "content": row["content"], "url": row["url"],
            "credibility": row["credibility"], "metadata": meta, "created_at": row["created_at"],
        }

    @classmethod
    def _claim_row(cls, row) -> Dict[str, Any]:
        return {
            "id": row["id"], "project_id": row["project_id"], "claim": row["claim"],
            "status": row["status"], "support": row["support"], "source_url": row["source_url"],
            "conflict": row["conflict"], "created_at": row["created_at"],
        }

    @staticmethod
    def _json(text: str) -> Optional[Dict[str, Any]]:
        raw = (text or "").strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 3:
                raw = "\n".join(lines[1:-1]).strip()
        start, end = raw.find("{"), raw.rfind("}")
        for candidate in (raw, raw[start:end + 1] if start >= 0 and end > start else ""):
            try:
                value = json.loads(candidate)
                return value if isinstance(value, dict) else None
            except Exception:
                continue
        return None

    @staticmethod
    def _load(value: Any, default: Any) -> Any:
        try:
            return json.loads(value) if isinstance(value, str) else (value or default)
        except Exception:
            return default
