"""Job matching: heuristic scoring and ranking."""

from __future__ import annotations

from arc.internships.matcher import heuristic_score, rank, score_job
from tests.conftest import ExplodingClient, make_router


def _job(**overrides):
    base = {
        "company": "Acme", "role": "Software Engineer Intern",
        "location": "Remote", "url": "https://acme.com",
        "description": "Build backend services with Python and FastAPI.",
        "requirements": "", "tags": [],
    }
    base.update(overrides)
    return base


class TestHeuristicScore:
    def test_perfect_internship_scores_high(self, profile):
        result = heuristic_score(_job(), profile)
        assert result["match_score"] >= 60
        assert "Python" in result["matched_skills"]
        assert any("intern" in r.lower() for r in result["match_reasons"])

    def test_senior_job_penalised(self, profile):
        result = heuristic_score(
            _job(role="Principal Architect", description="Lead the platform team. "
                  "10+ years of experience required."), profile)
        assert result["match_score"] < 40
        assert any("years" in r for r in result["match_reasons"])

    def test_unrelated_job_scores_low(self, profile):
        result = heuristic_score(
            _job(role="Marketing Intern",
                 description="Social media campaigns and copywriting.",
                 tags=["marketing"]), profile)
        assert result["match_score"] < 40

    def test_location_preference_bonus(self, profile):
        remote = heuristic_score(_job(location="Remote"), profile)
        onsite = heuristic_score(_job(location="Onsite — Sydney"), profile)
        assert remote["match_score"] >= onsite["match_score"]

    def test_score_bounds(self, profile):
        result = heuristic_score(_job(role="x" * 200, description="y" * 5000), profile)
        assert 0 <= result["match_score"] <= 100


class TestRankAndScoreJob:
    def test_rank_orders_by_score(self, profile):
        jobs = [_job(role="Marketing Intern", description="copywriting"),
                _job(), _job(role="Senior Architect", description="10+ years, lead")]
        for job in jobs:
            score_job(job, profile, router=None)
        ranked = rank(jobs)
        assert ranked[0]["role"] == "Software Engineer Intern"

    def test_score_job_without_llm(self, profile):
        job = score_job(_job(), profile, router=None)
        assert "match_score" in job and "match_reasons" in job

    def test_score_job_with_failing_llm_keeps_heuristic(self, profile):
        router = make_router(client=ExplodingClient())
        job = score_job(_job(), profile, router=router)
        assert job["match_score"] > 0  # heuristic result survives
