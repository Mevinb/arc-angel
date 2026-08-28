"""Database: jobs CRUD + dedupe, recruiters, threads, events, memory."""

from __future__ import annotations

from arc.db.database import Database, utcnow


class TestJobs:
    def test_upsert_and_get(self, db: Database):
        job_id = db.upsert_job({
            "company": "Acme", "role": "Backend Intern", "url": "https://acme.com/1",
            "match_score": 72, "match_reasons": ["Skills matched: Python"],
            "location": "Remote", "source": "remoteok",
        })
        job = db.get_job(job_id)
        assert job is not None
        assert job["company"] == "Acme"
        assert job["match_score"] == 72
        assert job["status"] == "new"
        assert job["match_reasons"] == ["Skills matched: Python"]

    def test_upsert_dedupes_on_company_role_url(self, db: Database):
        first = db.upsert_job({"company": "Acme", "role": "Intern",
                               "url": "https://acme.com/1", "match_score": 50})
        second = db.upsert_job({"company": "Acme", "role": "Intern",
                                "url": "https://acme.com/1", "match_score": 80})
        assert first == second
        job = db.get_job(first)
        assert job["match_score"] == 80  # score refreshed
        assert db.job_stats()["total"] == 1

    def test_upsert_keeps_longest_description(self, db: Database):
        job_id = db.upsert_job({"company": "A", "role": "R", "url": "u1",
                                "description": "short"})
        db.upsert_job({"company": "A", "role": "R", "url": "u1",
                       "description": "a much longer description"})
        assert db.get_job(job_id)["description"] == "a much longer description"

    def test_list_filters(self, db: Database):
        db.upsert_job({"company": "A", "role": "R1", "url": "u1", "match_score": 90})
        db.upsert_job({"company": "B", "role": "R2", "url": "u2", "match_score": 30})
        db.upsert_job({"company": "C", "role": "R3", "url": "u3", "match_score": 60,
                       "status": "applied"})
        assert [j["company"] for j in db.list_jobs(min_score=50)] == ["A", "C"]
        assert [j["company"] for j in db.list_jobs(status="applied")] == ["C"]

    def test_update_job(self, db: Database):
        job_id = db.upsert_job({"company": "A", "role": "R", "url": "u"})
        db.update_job(job_id, status="applied", date_applied=utcnow())
        job = db.get_job(job_id)
        assert job["status"] == "applied"
        assert job["date_applied"]

    def test_find_job(self, db: Database):
        db.upsert_job({"company": "Acme", "role": "Intern", "url": "u"})
        assert db.find_job("Acme", "Intern") is not None
        assert db.find_job("Nope", "Intern") is None


class TestRecruitersAndThreads:
    def test_recruiter_upsert(self, db: Database):
        first = db.upsert_recruiter("Jane", "jane@acme.com", "Acme")
        second = db.upsert_recruiter("Jane Doe", "jane@acme.com", "Acme Inc")
        assert first == second
        recruiters = db.list_recruiters()
        assert recruiters[0]["name"] == "Jane Doe"  # refined name kept
        assert recruiters[0]["company"] == "Acme Inc"

    def test_email_thread_upsert(self, db: Database):
        db.upsert_email_thread({"gmail_thread_id": "t1", "subject": "Hello",
                                "from_email": "a@b.c", "snippet": "hi",
                                "category": "recruiter", "summary": "A recruiter!",
                                "received_at": utcnow()})
        db.upsert_email_thread({"gmail_thread_id": "t1", "subject": "Hello",
                                "from_email": "a@b.c", "snippet": "hi",
                                "category": "other", "summary": "recategorized",
                                "received_at": utcnow()})
        threads = db.list_email_threads(category="other")
        assert len(threads) == 1
        assert threads[0]["summary"] == "recategorized"


class TestEvents:
    def test_add_and_upcoming(self, db: Database):
        db.add_event("deadline", "2026-09-10T12:00:00", "Acme application due")
        events = db.upcoming_events(within_days=30)
        assert len(events) == 1
        assert events[0]["kind"] == "deadline"

    def test_complete_event_hides_it(self, db: Database):
        event_id = db.add_event("interview", "2026-09-10T10:00:00", "Acme interview")
        db.complete_event(event_id)
        assert db.upcoming_events(within_days=30) == []


class TestMemory:
    def test_remember_recall_forget(self, db: Database):
        db.remember("user.timezone", "Europe/Berlin")
        assert db.recall("user.timezone") == "Europe/Berlin"
        db.remember("user.timezone", "UTC")  # overwrite
        assert db.recall("user.timezone") == "UTC"
        assert db.recall_all() == {"user.timezone": "UTC"}
        db.forget("user.timezone")
        assert db.recall("user.timezone") is None
