"""Email engine: LLM classification with heuristic fallback, DB persistence."""

from __future__ import annotations

from arc.core.memory import LongTermMemory
from arc.db.database import Database
from arc.tools.email_engine import GmailEngine, GmailTools
from tests.conftest import ExplodingClient, FakeMessage, FakeResponse, make_router

MESSAGES = [
    {"id": "m1", "thread_id": "t1", "subject": "Interview invitation",
     "from": "Recruiter Jane <jane@acme.com>", "date": "Mon, 1 Jan 2026",
     "snippet": "We would like to schedule a technical interview with you."},
    {"id": "m2", "thread_id": "t2", "subject": "Your application",
     "from": "no-reply@jobs.com", "date": "Mon, 1 Jan 2026",
     "snippet": "We regret to inform you we moved forward with other candidates."},
    {"id": "m3", "thread_id": "t3", "subject": "Weekly newsletter",
     "from": "news@digest.io", "date": "Mon, 1 Jan 2026",
     "snippet": "Top 10 stories this week."},
]


def _gmail_tools(db: Database, router) -> GmailTools:
    engine = GmailEngine(credentials_path=db.path.parent / "creds.json",
                         token_path=db.path.parent / "token.json")
    return GmailTools(engine, router, db)


class TestClassification:
    def test_heuristic_fallback_when_llm_fails(self, db):
        router = make_router(client=ExplodingClient())
        tools = _gmail_tools(db, router)
        enriched = tools.classify_and_summarize(MESSAGES)
        by_subject = {record["subject"]: record for record in enriched}
        assert by_subject["Interview invitation"]["category"] == "interview"
        assert by_subject["Your application"]["category"] == "rejection"
        assert by_subject["Weekly newsletter"]["category"] == "other"

    def test_persists_threads_and_recruiters(self, db):
        router = make_router(client=ExplodingClient())
        tools = _gmail_tools(db, router)
        # Message 1 is a recruiter/interview mail — classify it as recruiter
        # to exercise the recruiter-persistence branch.
        recruiter_message = dict(MESSAGES[0])
        recruiter_message["subject"] = "Opportunity at Acme"
        recruiter_message["snippet"] = "I am a recruiter with an exciting role"
        tools.classify_and_summarize([recruiter_message])
        threads = db.list_email_threads()
        assert len(threads) == 1
        recruiters = db.list_recruiters()
        assert recruiters and recruiters[0]["email"] == "jane@acme.com"

    def test_llm_classification_used_when_available(self, db):
        payload = ('[{"index": 0, "category": "recruiter", "summary": "Jane wants to chat", '
                   '"actionable": true, "recruiter_email": "jane@acme.com"}]')
        router = make_router([FakeResponse(FakeMessage(payload))])
        tools = _gmail_tools(db, router)
        enriched = tools.classify_and_summarize([MESSAGES[0]])
        assert enriched[0]["category"] == "recruiter"
        assert enriched[0]["summary"] == "Jane wants to chat"
        assert db.list_email_threads(category="recruiter")

    def test_daily_digest_renders_counts(self, db):
        router = make_router(client=ExplodingClient())
        tools = _gmail_tools(db, router)
        original_search = tools.engine.search
        tools.engine.search = lambda query, max_results=25: MESSAGES  # type: ignore[method-assign]
        try:
            digest = tools.daily_digest()
        finally:
            tools.engine.search = original_search  # type: ignore[method-assign]
        assert "Recent mail" in digest
        assert "interview: 1" in digest
        assert "rejection: 1" in digest


class TestAvailability:
    def test_missing_credentials_reported(self, tmp_path):
        engine = GmailEngine(credentials_path=tmp_path / "nope.json",
                             token_path=tmp_path / "token.json")
        message = engine.availability_message()
        assert "not installed" in message or "missing" in message
