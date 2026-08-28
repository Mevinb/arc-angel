"""Automation scheduler: task selection, auto-mode safety, run loop pieces."""

from __future__ import annotations

import json

from jarvis.automation.scheduler import Scheduler
from jarvis.config import load_config
from tests.conftest import FakeMessage, FakeResponse, ScriptedClient


class TestScheduler:
    def test_unknown_task_fails_cleanly(self, tmp_path):
        from jarvis.app import JarvisApp
        config = load_config(project_root=tmp_path)
        config.safety_mode = "auto"
        app = JarvisApp(config=config, quiet=True)
        try:
            result = Scheduler(app).run_task("nope")
            assert not result.ok
            assert "Unknown task" in result.output
            assert "email_check" in result.output
        finally:
            app.close()

    def test_job_search_task_saves_jobs(self, tmp_path, monkeypatch):
        import jarvis.internships.sources as sources

        def fake_source(**_):
            return [sources.manual_job("Acme", "Backend Intern",
                                       url="https://acme.com",
                                       description="Python intern role")]

        monkeypatch.setattr(sources, "SOURCES", {"fake": fake_source})
        from jarvis.app import JarvisApp
        config = load_config(project_root=tmp_path)
        config.safety_mode = "auto"
        app = JarvisApp(config=config, quiet=True)
        app.internships.source_names = ["fake"]
        try:
            scheduler = Scheduler(app)
            result = scheduler.run_task("job_search")
            assert result.ok
            assert "Acme" in result.output
            assert app.db.job_stats()["total"] == 1
        finally:
            app.close()

    def test_email_check_skipped_without_gmail(self, tmp_path):
        from jarvis.app import JarvisApp
        config = load_config(project_root=tmp_path)
        config.safety_mode = "auto"
        app = JarvisApp(config=config, quiet=True)
        try:
            result = Scheduler(app).run_task("email_check")
            # Not installed / no credentials → graceful skip, not a crash.
            assert result.ok
            assert "skipped" in result.output
        finally:
            app.close()

    def test_deadline_check_reports_events(self, tmp_path):
        from datetime import datetime, timedelta

        from jarvis.app import JarvisApp
        config = load_config(project_root=tmp_path)
        config.safety_mode = "auto"
        app = JarvisApp(config=config, quiet=True)
        try:
            due = (datetime.now() + timedelta(days=2)).isoformat(timespec="seconds")
            app.db.add_event("deadline", due, "Acme application")
            result = Scheduler(app).run_task("deadline_check")
            assert result.ok
            assert "Acme application" in result.output
        finally:
            app.close()

    def test_task_intervals_respected(self, tmp_path):
        from jarvis.app import JarvisApp
        config = load_config(project_root=tmp_path)
        config.safety_mode = "auto"
        app = JarvisApp(config=config, quiet=True)
        try:
            scheduler = Scheduler(app)
            task = scheduler.tasks["deadline_check"]
            assert task.due()  # never run → due immediately
            assert task.seconds_until_due() == 0.0
            task.last_run = __import__("time").time()  # just ran
            assert not task.due()
            assert task.seconds_until_due() > 0
        finally:
            app.close()
