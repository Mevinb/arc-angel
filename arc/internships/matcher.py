"""Match jobs against the personal profile and rank them (Phase 6).

Two scoring passes:
1. ``heuristic_score`` — fast, deterministic, offline: skill overlap, role
   preference, location preference, seniority signals and red flags.
2. ``llm_score`` — optional refinement through the reasoning model, producing
   a nuanced score with justifications.

The engine uses heuristic first; when ``internships.llm_analysis`` is enabled
the LLM re-scores the top candidates.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from ..core.llm import LLMRouter
from ..profile.profile import Profile

logger = logging.getLogger("arc.match")

# Phrases that suggest the job is not entry-level friendly.
SENIORITY_REDFLAGS = [
    r"\b(\d+)\+?\s*years?\b",           # "5+ years" — checked numerically below
    r"\bsenior\b", r"\bstaff\b", r"\bprincipal\b", r"\blead\b",
    r"\barchitect\b", r"\bmanager\b", r"\bhead of\b", r"\bdirector\b",
]

REMOTE_PATTERNS = [r"\bremote\b", r"\banywhere\b", r"\bwork from home\b", r"\bwfh\b"]


# Words too generic to earn role-preference points on their own ("Marketing
# Intern" must not score as a "Backend Intern" match just because both say intern).
GENERIC_ROLE_WORDS = {
    "intern", "internship", "junior", "senior", "co-op", "coop", "trainee",
    "fellow", "fellowship", "engineer", "developer", "role", "position", "job",
}


def heuristic_score(job: Dict[str, Any], profile: Profile) -> Dict[str, Any]:
    """Score 0-100 with reasons. Deterministic; safe offline."""
    text = " ".join(str(job.get(field, "")) for field in
                    ("role", "description", "requirements", "location")).lower()
    tags = " ".join(str(t) for t in job.get("tags", [])).lower()
    haystack = f"{text} {tags}"

    reasons: List[str] = []
    score = 0.0

    # --- skill overlap (up to 45 points)
    skills = profile.skills_list()
    matched: List[str] = []
    for skill in skills:
        if re.search(rf"\b{re.escape(skill.lower())}\b", haystack):
            matched.append(skill)
    if skills:
        ratio = len(matched) / len(skills)
        score += min(45.0, ratio * 150)
        if matched:
            reasons.append(f"Skills matched: {', '.join(matched[:8])}")

    # --- role preference (up to 25 points, substantive words only)
    preferred_roles = [str(r).lower() for r in (profile.get("preferred_roles") or [])]
    role_text = str(job.get("role", "")).lower()
    role_bonus = 0.0
    for preferred in preferred_roles:
        words = [w for w in re.findall(r"[a-z+#.]+", preferred)
                 if len(w) > 2 and w not in GENERIC_ROLE_WORDS]
        if not words:
            continue
        hits = sum(1 for w in words if w in role_text)
        role_bonus = max(role_bonus, hits / len(words))
    if role_bonus > 0:
        score += role_bonus * 25
        reasons.append("Role matches your preferences")
    if re.search(r"\bintern(ship)?\b|\bco-?op\b", role_text):
        score += 10
        reasons.append("Explicit internship/co-op role")

    # --- location (up to 15 points)
    location = str(job.get("location", "")).lower()
    if any(re.search(p, location) for p in REMOTE_PATTERNS):
        score += 10
        reasons.append("Remote friendly")
    preferred_locations = [str(l).lower() for l in (profile.get("preferred_locations") or [])]
    for preferred in preferred_locations:
        if preferred and preferred in location:
            score += 5
            reasons.append(f"Preferred location: {preferred}")
            break

    # --- seniority red flags (up to -30)
    for pattern in SENIORITY_REDFLAGS:
        match = re.search(pattern, haystack)
        if not match:
            continue
        if pattern.startswith(r"\b(\d+)"):  # years requirement
            years = int(match.group(1))
            if years >= 3:
                score -= min(20, years * 2)
                reasons.append(f"Asks for {years}+ years experience")
        else:
            score -= 8
            reasons.append(f"Seniority signal: {match.group(0)}")

    # --- internship signals in the body
    if re.search(r"\bintern(ship)?\b", haystack):
        score += 5
    # --- recency/quality signals
    if job.get("url", "").startswith("https"):
        score += 2

    final = int(max(0, min(100, round(score))))
    if not reasons:
        reasons.append("No strong signals either way")
    return {
        "match_score": final,
        "match_reasons": reasons,
        "matched_skills": matched,
    }


def llm_score(job: Dict[str, Any], profile: Profile,
              router: LLMRouter) -> Dict[str, Any]:
    """Ask the reasoning model to score fit. Falls back to heuristic on error."""
    prompt = (
        "You are a strict career coach scoring a job posting for a student.\n"
        f"CANDIDATE PROFILE:\n{profile.summarize()}\n\n"
        f"JOB:\nCompany: {job.get('company', '')}\n"
        f"Role: {job.get('role', '')}\n"
        f"Location: {job.get('location', '')}\n"
        f"Description: {str(job.get('description', ''))[:1500]}\n\n"
        "Score the fit from 0-100 (be honest; 50+ means worth applying). "
        "Return JSON: {\"score\": int, \"reasons\": [\"...\"], "
        "\"missing_skills\": [\"...\"], \"verdict\": \"apply\"|\"maybe\"|\"skip\"}"
    )
    try:
        parsed = router.ask_json(prompt, role="reasoning")
        return {
            "match_score": int(max(0, min(100, parsed.get("score", 0)))),
            "match_reasons": [str(r) for r in parsed.get("reasons", [])][:6],
            "missing_skills": [str(s) for s in parsed.get("missing_skills", [])][:8],
            "verdict": str(parsed.get("verdict", "maybe")),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM scoring failed (%s); keeping heuristic score", exc)
        return {}


def score_job(job: Dict[str, Any], profile: Profile,
              router: Optional[LLMRouter] = None) -> Dict[str, Any]:
    """Combine heuristic + optional LLM scoring into the job record."""
    base = heuristic_score(job, profile)
    job.update(base)
    if router is not None:
        refined = llm_score(job, profile, router)
        if refined:
            if refined["match_score"]:
                # Blend: weight LLM 60 / heuristic 40
                blended = int(round(0.6 * refined["match_score"] + 0.4 * base["match_score"]))
                job["match_score"] = blended
                job["match_reasons"] = refined["match_reasons"] or base["match_reasons"]
            job.update({k: v for k, v in refined.items()
                        if k not in ("match_score", "match_reasons")})
    return job


def rank(jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort jobs by score, most promising first."""
    return sorted(jobs, key=lambda job: job.get("match_score", 0), reverse=True)
