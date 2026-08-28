"""Phase 11 — Automation: scheduled background tasks.

Tasks (intervals configurable in ``config.yaml`` → ``automation``):
- ``email_check``    — fetch recent mail, classify, flag actionable threads.
- ``job_search``     — pull job boards, score against the profile, save.
- ``deadline_check`` — surface upcoming interviews/deadlines/follow-ups.

Safety: the scheduler runs in ``auto`` mode — GREEN actions only; anything
needing approval is skipped and reported. The CLI constructs the app with
``safety.mode=auto`` for `arc automate`, so a scheduled run can never send
mail or run commands on its own.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..app import ArcApp
from ..db.database import Database

logger = logging.getLogger("arc.automation")


@dataclass
class TaskResult:
    task: str
    ok: bool
    output: str
    ran_at: str


class AutomationTask:
    name: str = "task"
    description: str = ""

    def __init__(self, app: ArcApp, interval_minutes: int) -> None:
        self.app = app
        self.interval_minutes = max(1, int(interval_minutes))
        self.last_run: Optional[float] = None

    def due(self) -> bool:
        return self.last_run is None or \
            (time.time() - self.last_run) >= self.interval_minutes * 60

    def seconds_until_due(self) -> float:
        if self.last_run is None:
            return 0.0
        return max(0.0, self.interval_minutes * 60 - (time.time() - self.last_run))

    def run(self) -> str:
        raise NotImplementedError

    def execute(self) -> TaskResult:
        ran_at = datetime.now().isoformat(timespec="seconds")
        self.last_run = time.time()
        try:
            output = self.run()
            self.app.db.remember(f"automation.last_run.{self.name}", ran_at)
            return TaskResult(self.name, True, output, ran_at)
        except Exception as exc:  # noqa: BLE001 - scheduled tasks never crash the loop
            logger.exception("Automation task %s failed", self.name)
            return TaskResult(self.name, False, f"failed: {exc}", ran_at)


class EmailCheckTask(AutomationTask):
    name = "email_check"
    description = "Fetch recent Gmail, classify and summarize actionable mail."

    def run(self) -> str:
        problem = self.app.gmail_engine.availability_message()
        if problem != "ready":
            return f"skipped — {problem}"
        digest = self.app.gmail_tools.daily_digest()
        actionable = digest.count("[ACTION]")
        return f"{digest}\n\n({actionable} actionable thread(s))"


class JobSearchTask(AutomationTask):
    name = "job_search"
    description = "Search job boards, score listings, save to the database."

    def run(self) -> str:
        # Run without per-listing LLM scoring in automation to bound cost;
        # heuristic scoring still applies and the report shows the top matches.
        engine = self.app.internships
        saved_llm = engine.llm_analysis
        engine.llm_analysis = False
        try:
            jobs = engine.search(internship_only=True)
        finally:
            engine.llm_analysis = saved_llm
        return engine.report(jobs, top=5)


class DeadlineCheckTask(AutomationTask):
    name = "deadline_check"
    description = "Surface upcoming deadlines, interviews and follow-ups."

    def run(self) -> str:
        events = self.app.db.upcoming_events(within_days=7)
        if not events:
            return "No upcoming deadlines in the next 7 days."
        lines = [f"{e['kind']}: {e['description']} — due {e['due_at']}"
                 for e in events[:15]]
        return "\n".join(lines)


class Scheduler:
    """Runs the automation tasks on a loop."""

    def __init__(self, app: ArcApp) -> None:
        self.app = app
        automation = app.config.automation
        self.tasks: Dict[str, AutomationTask] = {
            task.name: task
            for task in (
                EmailCheckTask(app, automation.get("email_check_minutes", 30)),
                JobSearchTask(app, automation.get("job_search_minutes", 360)),
                DeadlineCheckTask(app, automation.get("deadline_check_minutes", 60)),
            )
        }

    # ------------------------------------------------------------------- api
    def run_task(self, name: str) -> TaskResult:
        task = self.tasks.get(name)
        if task is None:
            available = ", ".join(sorted(self.tasks))
            return TaskResult(name, False,
                              f"Unknown task {name!r}. Available: {available}",
                              datetime.now().isoformat(timespec="seconds"))
        return task.execute()

    def run_all(self) -> List[TaskResult]:
        return [task.execute() for task in self.tasks.values()]

    def run_forever(self, poll_seconds: int = 30) -> None:
        """Block, running tasks when their intervals elapse. Ctrl-C to stop."""
        logger.info("Scheduler started: %s",
                    {name: f"every {t.interval_minutes}m"
                     for name, t in self.tasks.items()})
        try:
            while True:
                for task in self.tasks.values():
                    if task.due():
                        result = task.execute()
                        self._report(result)
                sleep_for = min(
                    [task.seconds_until_due() for task in self.tasks.values()]
                    + [poll_seconds])
                time.sleep(min(max(sleep_for, 1), poll_seconds))
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")

    @staticmethod
    def _report(result: TaskResult) -> None:
        status = "ok" if result.ok else "FAILED"
        logger.info("[%s] %s at %s:\n%s", status, result.task, result.ran_at,
                    result.output)


__all__ = ["Scheduler", "AutomationTask", "TaskResult", "EmailCheckTask",
           "JobSearchTask", "DeadlineCheckTask"]
