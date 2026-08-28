"""Application facade: wiring, health report, one-shot chat (fake LLM)."""

from __future__ import annotations

import json
from pathlib import Path

from arc.app import ArcApp
from arc.config import load_config
from tests.conftest import FakeMessage, FakeResponse, FakeToolCall


def _app(tmp_path: Path, responses=None, client=None) -> ArcApp:
    config = load_config(project_root=tmp_path)
    config.safety_mode = "auto"  # no interactive approvals in tests
    app = ArcApp(config=config, quiet=True)
    if responses is not None or client is not None:
        from tests.conftest import ScriptedClient
        app.router._client = client or ScriptedClient(responses or [])
    return app


class TestWiring:
    def test_all_tools_registered(self, tmp_path):
        app = _app(tmp_path)
        expected = {
            "web.fetch", "browser.open", "browser.task",
            "computer.run_shell", "computer.run_python", "computer.task",
            "email.search", "email.digest", "email.read_thread",
            "email.create_draft", "email.send",
            "jobs.search", "jobs.list", "jobs.analyze",
            "jobs.draft_recruiter_email", "jobs.update_status", "jobs.deadlines",
            "memory.remember", "memory.recall", "memory.forget",
            "profile.search",
        }
        assert expected.issubset(set(app.registry.names()))

    def test_profile_created_on_first_run(self, tmp_path):
        app = _app(tmp_path)
        assert app.config.profile_path.is_file()
        assert app.profile.is_placeholder()

    def test_health_report_shape(self, tmp_path):
        app = _app(tmp_path)
        report = app.health_report()
        assert report["safety_mode"] == "auto"
        assert "llm" in report and "ok" in report["llm"]
        assert "email.send" in report["optional"] or "gmail.oauth" in report["optional"]
        assert report["db"]["jobs"]["total"] == 0

    def test_chat_end_to_end_with_tools(self, tmp_path):
        app = _app(tmp_path, responses=[
            FakeResponse(FakeMessage("", tool_calls=[
                FakeToolCall("call-1", "memory.remember",
                             {"key": "user.name", "value": "Test"})])),
            FakeResponse(FakeMessage("Saved it!")),
        ])
        reply = app.chat("remember that my name is Test")
        assert reply == "Saved it!"
        assert app.db.recall("user.name") == "Test"


class TestJobFlow:
    def test_search_reports_and_saves(self, tmp_path, monkeypatch):
        import arc.internships.sources as sources

        def fake_source(**_):
            return [
                sources.manual_job("Acme", "Software Engineer Intern",
                                   url="https://acme.com",
                                   description="Python FastAPI backend intern work"),
                sources.manual_job("OldCorp", "Principal Architect",
                                   url="https://old.com",
                                   description="12+ years experience, lead team"),
            ]

        monkeypatch.setattr(sources, "SOURCES", {"fake": fake_source})
        app = _app(tmp_path)
        app.internships.source_names = ["fake"]
        app.internships.llm_analysis = False  # offline: heuristic scoring only
        jobs = app.internships.search()
        assert len(jobs) == 2
        assert jobs[0]["company"] == "Acme"  # intern job outranks the senior one
        # Only the promising listing is persisted (min_score_to_save = 20).
        assert app.db.job_stats()["total"] == 1
        assert app.db.list_jobs()[0]["company"] == "Acme"
        report = app.internships.report(jobs)
        assert "Acme" in report
