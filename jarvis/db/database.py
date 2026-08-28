"""Phase 8 — SQLite persistence for jobs, applications, recruiters,
email threads, calendar-ish events and long-term memory.

Uses the stdlib sqlite3 driver with WAL journaling. Schema is created
idempotently on first connect.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger("jarvis.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT NOT NULL,
    role TEXT NOT NULL,
    location TEXT DEFAULT '',
    url TEXT DEFAULT '',
    source TEXT DEFAULT '',
    description TEXT DEFAULT '',
    requirements TEXT DEFAULT '',
    match_score INTEGER DEFAULT 0,
    match_reasons TEXT DEFAULT '[]',
    deadline TEXT DEFAULT '',
    status TEXT DEFAULT 'new',
    recruiter_id INTEGER,
    date_found TEXT NOT NULL,
    date_applied TEXT DEFAULT '',
    notes TEXT DEFAULT '',
    raw_json TEXT DEFAULT '{}',
    UNIQUE(company, role, url)
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_score ON jobs(match_score);

CREATE TABLE IF NOT EXISTS recruiters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT DEFAULT '',
    email TEXT NOT NULL,
    company TEXT DEFAULT '',
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL,
    notes TEXT DEFAULT '',
    UNIQUE(email)
);

CREATE TABLE IF NOT EXISTS email_threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gmail_thread_id TEXT UNIQUE NOT NULL,
    subject TEXT DEFAULT '',
    from_email TEXT DEFAULT '',
    snippet TEXT DEFAULT '',
    category TEXT DEFAULT 'other',
    summary TEXT DEFAULT '',
    received_at TEXT NOT NULL,
    processed INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,               -- deadline | interview | followup | reminder
    job_id INTEGER REFERENCES jobs(id) ON DELETE CASCADE,
    due_at TEXT NOT NULL,
    description TEXT DEFAULT '',
    done INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_events_due ON events(due_at);

CREATE TABLE IF NOT EXISTS memory (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

JOB_COLUMNS = (
    "company", "role", "location", "url", "source", "description", "requirements",
    "match_score", "match_reasons", "deadline", "status", "recruiter_id",
    "date_found", "date_applied", "notes", "raw_json",
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    """Thread-safe wrapper around sqlite3 for JARVIS data."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        logger.info("Database ready at %s", self.path)

    # ------------------------------------------------------------------ jobs
    def upsert_job(self, job: Dict[str, Any]) -> Optional[int]:
        """Insert or update a job (dedupe on company+role+url). Returns job id,
        or None if nothing changed."""
        row = {col: job.get(col, "" if col not in ("match_score",) else 0)
               for col in JOB_COLUMNS}
        row["status"] = row.get("status") or "new"
        row["match_score"] = int(row.get("match_score") or 0)
        if isinstance(row.get("match_reasons"), (list, dict)):
            row["match_reasons"] = json.dumps(row["match_reasons"])
        if isinstance(row.get("raw_json"), (list, dict)):
            row["raw_json"] = json.dumps(row["raw_json"])
        row.setdefault("date_found", utcnow())
        if not row.get("date_applied"):
            row["date_applied"] = ""

        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO jobs ({cols}) VALUES ({marks})
                   ON CONFLICT(company, role, url) DO UPDATE SET
                     match_score=excluded.match_score,
                     match_reasons=excluded.match_reasons,
                     description=CASE WHEN LENGTH(excluded.description) > LENGTH(jobs.description)
                                      THEN excluded.description ELSE jobs.description END
                   RETURNING id""".format(cols=",".join(JOB_COLUMNS),
                                          marks=",".join("?" * len(JOB_COLUMNS))),
                [row[c] for c in JOB_COLUMNS],
            )
            job_id = cursor.fetchone()[0]
            self._conn.commit()
        return job_id

    def get_job(self, job_id: int) -> Optional[Dict[str, Any]]:
        return self._row_to_job(self._conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone())

    def find_job(self, company: str, role: str) -> Optional[Dict[str, Any]]:
        return self._row_to_job(self._conn.execute(
            "SELECT * FROM jobs WHERE company = ? AND role = ? ORDER BY id DESC LIMIT 1",
            (company, role)).fetchone())

    def list_jobs(self, status: Optional[str] = None, min_score: int = 0,
                  limit: int = 100) -> List[Dict[str, Any]]:
        query = "SELECT * FROM jobs WHERE match_score >= ?"
        params: Sequence[Any] = [min_score]
        if status:
            query += " AND status = ?"
            params = list(params) + [status]
        query += " ORDER BY match_score DESC, date_found DESC LIMIT ?"
        params = list(params) + [limit]
        return [self._row_to_job(r) for r in self._conn.execute(query, params).fetchall()]

    def update_job(self, job_id: int, **changes: Any) -> None:
        if not changes:
            return
        allowed = set(JOB_COLUMNS) | {"status", "notes", "date_applied", "deadline"}
        sets, params = [], []
        for key, value in changes.items():
            if key not in allowed:
                raise KeyError(f"Cannot update job column {key!r}")
            if isinstance(value, (list, dict)):
                value = json.dumps(value)
            sets.append(f"{key} = ?")
            params.append(value)
        params.append(job_id)
        with self._lock:
            self._conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)
            self._conn.commit()

    def delete_job(self, job_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._conn.commit()

    def job_stats(self) -> Dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        stats = {r["status"]: r["n"] for r in rows}
        stats["total"] = sum(stats.values())
        return stats

    # ------------------------------------------------------------- recruiters
    def upsert_recruiter(self, name: str, email: str, company: str = "",
                         notes: str = "") -> int:
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO recruiters (name, email, company, first_seen, last_seen, notes)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(email) DO UPDATE SET
                     last_seen=excluded.last_seen,
                     name=CASE WHEN excluded.name != '' THEN excluded.name ELSE recruiters.name END,
                     company=CASE WHEN excluded.company != '' THEN excluded.company
                                  ELSE recruiters.company END
                   RETURNING id""",
                (name, email, company, utcnow(), utcnow(), notes),
            )
            recruiter_id = cursor.fetchone()[0]
            self._conn.commit()
        return recruiter_id

    def list_recruiters(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._conn.execute(
            "SELECT * FROM recruiters ORDER BY last_seen DESC").fetchall()]

    # ---------------------------------------------------------- email threads
    def upsert_email_thread(self, thread: Dict[str, Any]) -> Optional[int]:
        with self._lock:
            cursor = self._conn.execute(
                """INSERT INTO email_threads
                     (gmail_thread_id, subject, from_email, snippet, category, summary, received_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(gmail_thread_id) DO UPDATE SET
                     category=excluded.category, summary=excluded.summary,
                     snippet=excluded.snippet
                   RETURNING id""",
                (thread.get("gmail_thread_id", ""), thread.get("subject", ""),
                 thread.get("from_email", ""), thread.get("snippet", ""),
                 thread.get("category", "other"), thread.get("summary", ""),
                 thread.get("received_at", utcnow())),
            )
            thread_id = cursor.fetchone()[0]
            self._conn.commit()
        return thread_id

    def list_email_threads(self, category: Optional[str] = None,
                           limit: int = 50) -> List[Dict[str, Any]]:
        if category:
            rows = self._conn.execute(
                "SELECT * FROM email_threads WHERE category = ? ORDER BY received_at DESC LIMIT ?",
                (category, limit)).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM email_threads ORDER BY received_at DESC LIMIT ?",
                (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ---------------------------------------------------------------- events
    def add_event(self, kind: str, due_at: str, description: str = "",
                  job_id: Optional[int] = None) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO events (kind, job_id, due_at, description) VALUES (?, ?, ?, ?)",
                (kind, job_id, due_at, description))
            self._conn.commit()
            return cursor.lastrowid or 0

    def upcoming_events(self, within_days: int = 7,
                        include_done: bool = False) -> List[Dict[str, Any]]:
        query = ("SELECT e.*, j.company, j.role FROM events e "
                 "LEFT JOIN jobs j ON j.id = e.job_id "
                 "WHERE e.due_at <= datetime('now', ?) ")
        if not include_done:
            query += "AND e.done = 0 "
        query += "ORDER BY e.due_at ASC"
        rows = self._conn.execute(query, (f"+{within_days} days",)).fetchall()
        return [dict(r) for r in rows]

    def complete_event(self, event_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE events SET done = 1 WHERE id = ?", (event_id,))
            self._conn.commit()

    # ---------------------------------------------------------------- memory
    def remember(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO memory (key, value, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(key) DO UPDATE SET value=excluded.value,
                     updated_at=excluded.updated_at""",
                (key, value, utcnow()))
            self._conn.commit()

    def recall(self, key: str) -> Optional[str]:
        row = self._conn.execute("SELECT value FROM memory WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def recall_all(self) -> Dict[str, str]:
        return {r["key"]: r["value"] for r in
                self._conn.execute("SELECT key, value FROM memory ORDER BY key").fetchall()}

    def forget(self, key: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memory WHERE key = ?", (key,))
            self._conn.commit()

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _row_to_job(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        job = dict(row)
        for key in ("match_reasons", "raw_json"):
            default = "{}" if key == "raw_json" else "[]"
            try:
                job[key] = json.loads(job.get(key) or default)
            except (json.JSONDecodeError, TypeError):
                job[key] = {} if key == "raw_json" else []
        return job

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # Context manager support
    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
