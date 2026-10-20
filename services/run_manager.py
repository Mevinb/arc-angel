"""Background run supervision, pause/resume/cancel and live event fan-out."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from memory.database import db_session
from services.events import NotificationService
from services.tasks import TERMINAL_RUN, TaskService

logger = logging.getLogger(__name__)


class RunManager:
    _instance: Optional["RunManager"] = None

    def __init__(self) -> None:
        self._workers: Dict[str, asyncio.Task] = {}
        self._runtimes: Dict[str, Any] = {}
        self._subscribers: Dict[str, Set[Callable[[Dict[str, Any]], Any]]] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def get(cls) -> "RunManager":
        if cls._instance is None:
            cls._instance = RunManager()
        return cls._instance

    # --------------------------------------------------------------- lifecycle
    def start(self, run_id: str) -> Dict[str, Any]:
        """Start a queued/interrupted run. Must be called from the event loop."""
        loop = asyncio.get_running_loop()
        run = TaskService.get_run(run_id)
        if not run:
            raise ValueError("Run not found")
        if run["status"] in TERMINAL_RUN:
            raise ValueError(f"Run is already {run['status']}")
        if run_id in self._workers and not self._workers[run_id].done():
            raise ValueError("Run is already being supervised")
        from services.agent_runtime import AgentRuntime

        try:
            runtime = AgentRuntime(run, manager=self)
        except Exception:
            TaskService.update_run_status(run_id, "failed", error="Agent could not be loaded")
            raise
        self._runtimes[run_id] = runtime
        task = loop.create_task(self._supervise(run_id, runtime))
        self._workers[run_id] = task
        task.add_done_callback(lambda _t, rid=run_id: self._cleanup(rid))
        return {"run_id": run_id, "status": "starting"}

    async def _supervise(self, run_id: str, runtime: Any) -> None:
        try:
            await runtime.execute()
            run = TaskService.get_run(run_id)
            if run and run["status"] not in TERMINAL_RUN and run["status"] not in ("paused", "awaiting_approval", "interrupted"):
                TaskService.update_run_status(run_id, "failed", error="Runtime stopped without a final status")
        except asyncio.CancelledError:
            run = TaskService.get_run(run_id)
            if run and run["status"] not in TERMINAL_RUN:
                TaskService.update_run_status(run_id, "cancelled", error="Run worker was cancelled")
            raise
        except Exception as exc:
            logger.exception("Run %s crashed", run_id)
            run = TaskService.get_run(run_id)
            if run and run["status"] not in TERMINAL_RUN:
                TaskService.update_run_status(run_id, "failed", error=str(exc))
                NotificationService.create(
                    title="Agent run failed", body=str(exc)[:1000], severity="error",
                    kind="run", action_url="/platform#tasks",
                    data={"run_id": run_id, "task_id": run["task_id"]},
                )

    def _cleanup(self, run_id: str) -> None:
        self._workers.pop(run_id, None)
        self._runtimes.pop(run_id, None)

    async def wait(self, run_id: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        worker = self._workers.get(run_id)
        if worker:
            try:
                await asyncio.wait_for(asyncio.shield(worker), timeout=timeout)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
        return TaskService.get_run(run_id)

    async def pause(self, run_id: str) -> Dict[str, Any]:
        run = TaskService.get_run(run_id)
        if not run:
            raise ValueError("Run not found")
        if run["status"] in TERMINAL_RUN:
            raise ValueError(f"Run is already {run['status']}")
        if run["status"] == "awaiting_approval":
            raise ValueError("Resolve the pending approval before pausing")
        runtime = self._runtimes.get(run_id)
        if runtime:
            runtime.request_pause()
            # Wait briefly for the runtime to reach a checkpoint.
            for _ in range(20):
                await asyncio.sleep(0.15)
                current = TaskService.get_run(run_id)
                if current and current["status"] == "paused":
                    return current
        if run["status"] in ("queued", "interrupted"):
            TaskService.update_run_status(run_id, "paused" if run["status"] == "queued" else "interrupted")
            return TaskService.get_run(run_id)  # type: ignore[return-value]
        return TaskService.get_run(run_id)  # type: ignore[return-value]

    async def resume(self, run_id: str) -> Dict[str, Any]:
        run = TaskService.get_run(run_id)
        if not run:
            raise ValueError("Run not found")
        if run["status"] not in ("paused", "interrupted"):
            raise ValueError(f"Run is {run['status']}, not paused/interrupted")
        if run_id in self._workers and not self._workers[run_id].done():
            raise ValueError("Run worker is still active")
        TaskService.update_run_status(run_id, "running")
        TaskService.add_event(run_id, "run.resumed", {"resumed_at": time.time()})
        self.start(run_id)
        return TaskService.get_run(run_id)  # type: ignore[return-value]

    async def cancel(self, run_id: str, force_after: float = 10.0) -> Dict[str, Any]:
        run = TaskService.get_run(run_id)
        if not run:
            raise ValueError("Run not found")
        if run["status"] in TERMINAL_RUN:
            return run
        runtime = self._runtimes.get(run_id)
        if runtime:
            runtime.request_cancel()
        TaskService.update_run_status(run_id, "cancelled", error="Cancelled by user")
        try:
            from memory.database import db_session
            with db_session() as conn:
                conn.execute(
                    "UPDATE v2_tasks SET status='cancelled', updated_at=? WHERE id=? AND status NOT IN ('completed','partially_completed')",
                    (datetime.now(timezone.utc).isoformat(), run["task_id"]),
                )
        except Exception:
            pass
        worker = self._workers.get(run_id)
        if worker and not worker.done():
            async def force() -> None:
                await asyncio.sleep(force_after)
                if not worker.done():
                    worker.cancel()
            asyncio.create_task(force())
        return TaskService.get_run(run_id)  # type: ignore[return-value]

    def cancel_all(self, reason: str = "Cancelled by safety control") -> int:
        """Synchronously stop every supervised run (used by the kill switch)."""
        cancelled = 0
        for run_id, runtime in list(self._runtimes.items()):
            try:
                runtime.request_cancel()
            except Exception:
                pass
            run = TaskService.get_run(run_id)
            if not run or run["status"] in TERMINAL_RUN:
                continue
            try:
                TaskService.update_run_status(run_id, "cancelled", error=reason)
                with db_session() as conn:
                    conn.execute(
                        "UPDATE v2_tasks SET status='cancelled', updated_at=? "
                        "WHERE id=? AND status NOT IN ('completed','partially_completed')",
                        (datetime.now(timezone.utc).isoformat(), run["task_id"]),
                    )
                cancelled += 1
            except Exception:
                logger.debug("Could not cancel run %s", run_id, exc_info=True)
        return cancelled

    # ----------------------------------------------------------------- events
    def subscribe(self, run_id: str, callback: Callable[[Dict[str, Any]], Any]) -> None:
        self._subscribers.setdefault(run_id, set()).add(callback)

    def unsubscribe(self, run_id: str, callback: Callable[[Dict[str, Any]], Any]) -> None:
        subs = self._subscribers.get(run_id)
        if subs:
            subs.discard(callback)
            if not subs:
                self._subscribers.pop(run_id, None)

    def emit(self, run_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        event = {"run_id": run_id, "type": event_type, "payload": payload, "timestamp": time.time()}
        for callback in list(self._subscribers.get(run_id, set())):
            try:
                result = callback(event)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:
                logger.debug("Event subscriber failed", exc_info=True)

    # --------------------------------------------------------------- recovery
    async def recover(self) -> List[Dict[str, Any]]:
        recovered = TaskService.recover_interrupted()
        resumed = []
        for run in recovered:
            try:
                # AgentRuntime.execute() performs the interrupted -> running
                # transition.  Starting directly avoids a duplicate run.running
                # event and keeps recovery idempotent.
                TaskService.add_event(run["id"], "run.recovered", {"reason": "server_restart"})
                self.start(run["id"])
                resumed.append(run)
            except Exception as exc:
                logger.warning("Could not auto-resume run %s: %s", run["id"], exc)
        return resumed

    def active(self) -> List[Dict[str, Any]]:
        return [
            {"run_id": rid, "done": task.done()}
            for rid, task in self._workers.items() if not task.done()
        ]

    async def shutdown(self, timeout: float = 5.0) -> None:
        workers = list(self._workers.values())
        for worker in workers:
            worker.cancel()
        if workers:
            await asyncio.gather(*workers, return_exceptions=True)
        self._workers.clear()
        self._runtimes.clear()
        self._subscribers.clear()
