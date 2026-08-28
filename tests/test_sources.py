"""Job sources: normalization, internship detection, aggregation, dedupe."""

from __future__ import annotations

import json

import pytest

import arc.internships.sources as sources
from arc.internships.sources import _looks_like_internship, fetch_all, manual_job


@pytest.fixture(autouse=True)
def clear_cache():
    sources._CACHE.clear()
    yield
    sources._CACHE.clear()


class TestInternshipDetection:
    def test_positive_signals(self):
        for text in ("Software Internship", "junior developer", "New Grad role",
                     "6-month co-op", "AI fellowship", "trainee program"):
            assert _looks_like_internship(text), text

    def test_negative_signals(self):
        for text in ("Senior Backend Engineer", "Head of Design",
                     "Staff Software Engineer"):
            assert not _looks_like_internship(text), text


class TestManualJob:
    def test_normalises_fields(self):
        job = manual_job("Acme", "Intern", url="https://acme.com")
        assert job["source"] == "manual"
        assert job["location"] == ""
        assert job["tags"] == []


def _patch_source(monkeypatch, name: str, jobs):
    monkeypatch.setitem(sources.SOURCES, name, lambda **_: jobs)


class TestFetchAll:
    def test_aggregates_and_dedupes(self, monkeypatch):
        shared = manual_job("Acme", "Intern", url="u1")
        other = manual_job("Beta", "Intern", url="u2")
        _patch_source(monkeypatch, "remoteok", [shared, other])
        _patch_source(monkeypatch, "arbeitnow", [shared])  # duplicate
        jobs = fetch_all(["remoteok", "arbeitnow"])
        assert len(jobs) == 2

    def test_failing_source_skipped(self, monkeypatch):
        def broken(**_):
            raise RuntimeError("API down")

        monkeypatch.setitem(sources.SOURCES, "remoteok", broken)
        _patch_source(monkeypatch, "arbeitnow", [manual_job("Beta", "Intern")])
        jobs = fetch_all(["remoteok", "arbeitnow"])
        assert len(jobs) == 1

    def test_unknown_source_ignored(self):
        assert fetch_all(["nope"]) == []


class TestRemoteokParsing:
    def test_parses_api_payload(self, monkeypatch):
        payload = json.dumps([
            {"legal": "ignore me"},
            {"position": "Backend Intern", "company": "Acme",
             "location": "Remote", "url": "https://acme.com/job",
             "description": "Python and FastAPI", "tags": ["python"]},
            {"position": "Senior Architect", "company": "OldCorp",
             "url": "https://old.com"},  # filtered out (not an internship)
        ])

        monkeypatch.setattr(sources, "fetch_url",
                            lambda url, timeout=20: {"text": payload})
        jobs = sources.fetch_remoteok()
        assert len(jobs) == 1
        assert jobs[0]["company"] == "Acme"
        assert jobs[0]["source"] == "remoteok"
        assert jobs[0]["tags"] == ["python"]
