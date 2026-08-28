"""Phase 6 — Internship engine.

Workflow:
    Job websites -> fetch listings -> LLM/heuristic analysis -> match against
    profile -> rank jobs -> save to database.

Also generates recruiter emails (DRAFTS only — YELLOW) and prepares
application answers from the profile.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..core.llm import LLMRouter
from ..db.database import Database, utcnow
from ..profile.profile import Profile
from ..safety.permissions import RiskLevel
from ..tools.base import Tool, ToolResult
from . import matcher, sources

logger = logging.getLogger("jarvis.engine")


class InternshipEngine:
    def __init__(self, db: Database, profile: Profile,
                 router: Optional[LLMRouter] = None,
                 config: Optional[Dict[str, Any]] = None) -> None:
        self.db = db
        self.profile = profile
        self.router = router
        config = config or {}
        self.source_names = list(config.get("sources", ["remoteok", "arbeitnow", "hackernews"]))
        self.max_per_source = int(config.get("max_results_per_source", 40))
        self.llm_analysis = bool(config.get("llm_analysis", True)) and router is not None
        self.min_score_to_save = int(config.get("min_score_to_save", 20))

    # --------------------------------------------------------------- search
    def search(self, internship_only: bool = True,
               source_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Fetch, score, rank and persist jobs. Returns the ranked list."""
        jobs = sources.fetch_all(source_names or self.source_names,
                                 max_per_source=self.max_per_source,
                                 internship_only=internship_only)
        if not jobs:
            return []
        for job in jobs:
            matcher.score_job(job, self.profile,
                              self.router if self.llm_analysis else None)
        ranked = matcher.rank(jobs)
        saved = 0
        for job in ranked:
            if job.get("match_score", 0) >= self.min_score_to_save:
                self.db.upsert_job(job)
                saved += 1
        logger.info("Search: %d scored, %d saved (min score %d)",
                    len(ranked), saved, self.min_score_to_save)
        return ranked

    # -------------------------------------------------------------- analysis
    def analyze_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Deep-dive a single job: requirements, gaps, tailored pitch."""
        if self.router is None:
            result = matcher.heuristic_score(job, self.profile)
            return {"analysis": result["match_reasons"], **result}
        prompt = (
            f"CANDIDATE:\n{self.profile.summarize()}\n\n"
            f"JOB POSTING:\n{job.get('role', '')} at {job.get('company', '')}\n"
            f"{str(job.get('description', ''))[:3000]}\n\n"
            "Return JSON with keys: requirements (list of hard requirements), "
            "candidate_gaps (list), candidate_strengths (list), "
            "tailored_pitch (2-3 sentences why this candidate fits), "
            "fit_score (0-100)."
        )
        from ..core.llm import extract_json
        try:
            return self.router.ask_json(prompt, role="reasoning")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Job analysis failed: %s", exc)
            return {"error": str(exc)}

    # ------------------------------------------------------------- drafting
    def draft_recruiter_email(self, job: Dict[str, Any]) -> Dict[str, str]:
        """Generate a recruiter outreach draft for a job (never sends)."""
        if self.router is None:
            return {
                "subject": f"Interest in {job.get('role', 'role')} — {self.profile.name}",
                "body": (f"Hello,\n\nI'm {self.profile.name}, a student interested in the "
                         f"{job.get('role', 'internship')} position at "
                         f"{job.get('company', 'your company')}. "
                         "My background matches several of the requirements and I would "
                         "love to discuss the opportunity.\n\nBest,\n"
                         f"{self.profile.name}"),
            }
        prompt = (
            f"MY PROFILE:\n{self.profile.summarize()}\n\n"
            f"JOB:\n{job.get('role', '')} at {job.get('company', '')} "
            f"({job.get('location', '')})\n{str(job.get('description', ''))[:2000]}\n\n"
            "Write a short, warm, specific recruiter outreach email. "
            "Return JSON: {\"subject\": str, \"body\": str}. Body under 150 words."
        )
        try:
            parsed = self.router.ask_json(prompt, role="reasoning")
            return {"subject": str(parsed.get("subject", "")),
                    "body": str(parsed.get("body", ""))}
        except Exception as exc:  # noqa: BLE001
            logger.warning("Recruiter email drafting failed: %s", exc)
            return {"subject": "", "body": "", "error": str(exc)}

    def answer_application_question(self, question: str) -> str:
        """Answer an application question using the profile."""
        if self.router is None:
            return ("I need an LLM connection to answer application questions. "
                    "Configure OmniRoute (see .env.example).")
        prompt = (f"MY FULL PROFILE:\n{self.profile.to_text()}\n\n"
                  f"APPLICATION QUESTION: {question}\n\n"
                  "Write a strong, honest, concrete answer (under 200 words) "
                  "drawing on the projects, skills and education above.")
        return self.router.chat(prompt, role="reasoning")

    # -------------------------------------------------------------- reporting
    def report(self, jobs: List[Dict[str, Any]], top: int = 10) -> str:
        lines = [f"{len(jobs)} jobs found"]
        strong = [j for j in jobs if j.get("match_score", 0) >= 70]
        lines.append(f"{len(strong)} strong matches (70+)\n")
        for job in jobs[:top]:
            lines.append(
                f"{job.get('role', '?')} — {job.get('company', '?')} "
                f"[{job.get('match_score', 0)}%] ({job.get('location', '?')})")
            for reason in (job.get("match_reasons") or [])[:2]:
                lines.append(f"    · {reason}")
            if job.get("url"):
                lines.append(f"    {job['url']}")
        return "\n".join(lines)


# ------------------------------------------------------------ LLM tool wrappers
class JobSearchTool(Tool):
    name = "jobs.search"
    description = ("Search internship/job boards (RemoteOK, Arbeitnow, Hacker News "
                   "Who-is-hiring), score every listing against the user's profile, "
                   "rank and save them to the database. Use for 'find internships'.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "include_non_internships": {
                "type": "boolean",
                "description": "Include non-internship listings (default false)"},
        },
        "required": [],
    }

    def __init__(self, engine: InternshipEngine) -> None:
        self.engine = engine

    def run(self, include_non_internships: bool = False, **_: Any) -> ToolResult:
        jobs = self.engine.search(internship_only=not include_non_internships)
        return ToolResult.success(self.engine.report(jobs), count=len(jobs))


class JobListTool(Tool):
    name = "jobs.list"
    description = ("List saved jobs from the database, optionally filtered by "
                   "status (new/saved/applied/interview/rejected/offer) and "
                   "minimum match score.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {
            "status": {"type": "string", "description": "Filter by status"},
            "min_score": {"type": "integer", "description": "Minimum match score (default 0)"},
        },
        "required": [],
    }

    def __init__(self, engine: InternshipEngine) -> None:
        self.engine = engine

    def run(self, status: str = "", min_score: int = 0, **_: Any) -> ToolResult:
        jobs = self.engine.db.list_jobs(status=status or None,
                                        min_score=int(min_score), limit=30)
        if not jobs:
            return ToolResult.success("No saved jobs match that filter.")
        lines = []
        for job in jobs:
            lines.append(f"#{job['id']} [{job['match_score']}%] {job['role']} — "
                         f"{job['company']} ({job['status']}) {job['location']}")
        return ToolResult.success("\n".join(lines), count=len(jobs))


class JobAnalyzeTool(Tool):
    name = "jobs.analyze"
    description = ("Deep-analyze one saved job: requirements, gaps, strengths and a "
                   "tailored pitch, using the user's profile. Pass the job id.")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {"job_id": {"type": "integer"}},
        "required": ["job_id"],
    }

    def __init__(self, engine: InternshipEngine) -> None:
        self.engine = engine

    def run(self, job_id: int = 0, **_: Any) -> ToolResult:
        job = self.engine.db.get_job(int(job_id))
        if job is None:
            return ToolResult.failure(f"No job with id {job_id}")
        analysis = self.engine.analyze_job(job)
        lines = [f"{job['role']} — {job['company']}"]
        for key in ("requirements", "candidate_strengths", "candidate_gaps"):
            items = analysis.get(key) or []
            if items:
                lines.append(f"\n{key.replace('_', ' ').title()}:")
                lines.extend(f"  - {i}" for i in items[:8])
        if analysis.get("tailored_pitch"):
            lines.append(f"\nPitch: {analysis['tailored_pitch']}")
        if analysis.get("fit_score") is not None:
            lines.append(f"LLM fit score: {analysis.get('fit_score')}")
        return ToolResult.success("\n".join(lines))


class JobDraftEmailTool(Tool):
    name = "jobs.draft_recruiter_email"
    description = ("Draft a recruiter outreach email for a saved job (creates text "
                   "only — it is NOT sent, and never sends without the user).")
    risk = RiskLevel.GREEN
    parameters = {
        "properties": {"job_id": {"type": "integer"}},
        "required": ["job_id"],
    }

    def __init__(self, engine: InternshipEngine) -> None:
        self.engine = engine

    def run(self, job_id: int = 0, **_: Any) -> ToolResult:
        job = self.engine.db.get_job(int(job_id))
        if job is None:
            return ToolResult.failure(f"No job with id {job_id}")
        draft = self.engine.draft_recruiter_email(job)
        if draft.get("error"):
            return ToolResult.failure(f"Drafting failed: {draft['error']}")
        return ToolResult.success(
            f"Subject: {draft['subject']}\n\n{draft['body']}\n\n"
            "(Draft only — nothing was sent.)", draft=draft)


class ApplicationStatusTool(Tool):
    name = "jobs.update_status"
    description = ("Update a job's application status (new, saved, applied, "
                   "interview, rejected, offer) and optionally add a note.")
    risk = RiskLevel.YELLOW
    parameters = {
        "properties": {
            "job_id": {"type": "integer"},
            "status": {"type": "string", "enum": ["new", "saved", "applied",
                                                  "interview", "rejected", "offer"]},
            "note": {"type": "string"},
        },
        "required": ["job_id", "status"],
    }

    def __init__(self, engine: InternshipEngine) -> None:
        self.engine = engine

    def run(self, job_id: int = 0, status: str = "", note: str = "", **_: Any) -> ToolResult:
        job_id = int(job_id)
        job = self.engine.db.get_job(job_id)
        if job is None:
            return ToolResult.failure(f"No job with id {job_id}")
        changes: Dict[str, Any] = {"status": status}
        if note:
            changes["notes"] = ((job.get("notes") or "") + ("\n" if job.get("notes") else "")
                                + f"{utcnow()}: {note}").strip()
        if status == "applied" and not job.get("date_applied"):
            changes["date_applied"] = utcnow()
        self.engine.db.update_job(job_id, **changes)
        return ToolResult.success(f"Job #{job_id} ({job['role']} @ {job['company']}) "
                                  f"→ status '{status}'.")


class DeadlinesTool(Tool):
    name = "jobs.deadlines"
    description = "Show upcoming deadlines, interviews and follow-ups from the tracker."
    risk = RiskLevel.GREEN
    parameters = {"properties": {}, "required": []}

    def __init__(self, engine: InternshipEngine) -> None:
        self.engine = engine

    def run(self, **_: Any) -> ToolResult:
        events = self.engine.db.upcoming_events(within_days=14)
        if not events:
            return ToolResult.success("No upcoming deadlines or interviews.")
        lines = [f"{e['kind']}: {e['description']} — due {e['due_at']}"
                 for e in events[:20]]
        return ToolResult.success("\n".join(lines))


def register_internship_tools(registry: Any, engine: InternshipEngine) -> None:
    registry.register(JobSearchTool(engine))
    registry.register(JobListTool(engine))
    registry.register(JobAnalyzeTool(engine))
    registry.register(JobDraftEmailTool(engine))
    registry.register(ApplicationStatusTool(engine))
    registry.register(DeadlinesTool(engine))


__all__ = ["InternshipEngine", "register_internship_tools"]
