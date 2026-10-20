"""
TaskRunner: background execution, event persistence/replay, pause/resume, retries, recovery.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Dict, Any, AsyncGenerator, Optional, Set

from config import WORKSPACES_DIR
from memory.database import db_session
from engine.task_orchestrator import TaskOrchestrator
from engine.desktop_agent import DesktopAgent
from engine.wayland_controller import WaylandController
from engine.llama_client import LlamaClient
from engine.router import ProviderRouter


class TaskControl:
    def __init__(self):
        self._paused = asyncio.Event()
        self._paused.clear()
        self._aborted = asyncio.Event()

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def abort(self):
        self._aborted.set()
        self._paused.clear()

    def is_paused(self) -> bool:
        return self._paused.is_set()

    def is_aborted(self) -> bool:
        return self._aborted.is_set()

    async def checkpoint(self):
        while self._paused.is_set() and not self._aborted.is_set():
            await asyncio.sleep(0.5)
        if self._aborted.is_set():
            raise asyncio.CancelledError("Task aborted")


class TaskRunner:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskRunner, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._runners: Dict[str, asyncio.Task] = {}
        self._subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self._controls: Dict[str, TaskControl] = {}
        self._lock = asyncio.Lock()
        self._gui_lock = asyncio.Lock()

    @classmethod
    async def recover_on_startup(cls):
        runner = cls()
        with db_session() as conn:
            conn.execute(
                "UPDATE autonomous_tasks SET status='interrupted', last_error='Server restarted during execution' WHERE status IN ('running','queued')"
            )
            conn.execute(
                "UPDATE task_subgoals SET status='pending', updated_at=CURRENT_TIMESTAMP WHERE task_id IN (SELECT id FROM autonomous_tasks WHERE status='interrupted') AND status='running'"
            )

    async def start(self, task_id: str) -> bool:
        if task_id in self._runners and not self._runners[task_id].done():
            return True
        orch = TaskOrchestrator(llm_client=ProviderRouter())
        task = orch.get_task(task_id)
        if not task:
            return False
        if task.get("status") in ("completed", "aborted", "failed"):
            return False
        ctrl = TaskControl()
        self._controls[task_id] = ctrl
        self._subscribers.setdefault(task_id, set())
        t = asyncio.create_task(self._run(task_id, orch, ctrl))
        self._runners[task_id] = t
        return True

    async def retry(self, task_id: str) -> bool:
        orch = TaskOrchestrator(llm_client=ProviderRouter())
        try:
            orch.retry_task(task_id)
        except Exception:
            pass
        return await self.start(task_id)

    async def pause(self, task_id: str):
        if task_id in self._controls:
            self._controls[task_id].pause()
        TaskOrchestrator.pause_task(task_id)

    async def resume(self, task_id: str):
        if task_id in self._controls:
            self._controls[task_id].resume()
        TaskOrchestrator.resume_task(task_id)
        if task_id not in self._runners or self._runners[task_id].done():
            await self.start(task_id)

    async def abort(self, task_id: str):
        if task_id in self._controls:
            self._controls[task_id].abort()
        try:
            agent = getattr(TaskOrchestrator, "_current_agents", {}).get(task_id)
            if agent and hasattr(agent, "abort"):
                agent.abort()
        except Exception:
            pass
        TaskOrchestrator.abort_task(task_id)

    def _persist_event(self, task_id: str, ev: Dict[str, Any]) -> int:
        event_type = ev.get("type") or ev.get("action") or "event"
        payload = json.dumps(ev, default=str)
        with db_session() as conn:
            cur = conn.execute(
                "INSERT INTO task_events (task_id, event_type, payload) VALUES (?, ?, ?)",
                (task_id, event_type, payload)
            )
            return cur.lastrowid

    async def _broadcast(self, task_id: str, ev: Dict[str, Any]):
        subs = self._subscribers.get(task_id, set())
        dead = []
        for q in subs:
            try:
                q.put_nowait(ev)
            except Exception:
                dead.append(q)
        for q in dead:
            subs.discard(q)

    async def subscribe(self, task_id: str, since_id: int = 0) -> AsyncGenerator[Dict[str, Any], None]:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers.setdefault(task_id, set()).add(q)
        try:
            with db_session() as conn:
                cur = conn.execute(
                    "SELECT id, event_type, payload FROM task_events WHERE task_id = ? AND id > ? ORDER BY id ASC",
                    (task_id, since_id)
                )
                for row in cur.fetchall():
                    try:
                        payload = json.loads(row["payload"])
                        payload["id"] = row["id"]
                        yield payload
                    except Exception:
                        continue
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield ev
                    if ev.get("type") == "task_update" and ev.get("status") in (
                        "completed",
                        "partially_completed",
                        "failed",
                        "aborted",
                        "interrupted",
                    ):
                        break
                except asyncio.TimeoutError:
                    yield {"type": "heartbeat", "task_id": task_id, "ts": time.time()}
        finally:
            self._subscribers.get(task_id, set()).discard(q)

    async def shutdown(self):
        """Cancels all in-flight tasks and marks them interrupted."""
        tasks = [t for t in self._runners.values() if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._runners.clear()
        self._controls.clear()

    async def _run(self, task_id: str, orch: TaskOrchestrator, ctrl: TaskControl):
        try:
            orch.update_task_status(task_id, "running")
            async for ev in orch.run_task(task_id, control=ctrl):
                try:
                    eid = self._persist_event(task_id, ev)
                    ev["id"] = eid
                except Exception:
                    pass
                await self._broadcast(task_id, ev)
            task = orch.get_task(task_id)
            result_summary = (task or {}).get("result_summary") or ""
            if not result_summary:
                for sg in (task or {}).get("subgoals", []):
                    if sg.get("result_summary"):
                        result_summary += f"- {sg.get('title')}: {sg.get('result_summary')}\n"
            session_id = (task or {}).get("session_id")
            if session_id and result_summary.strip():
                try:
                    with db_session() as conn:
                        conn.execute(
                            "INSERT INTO messages (session_id, role, content, task_id, status) VALUES (?, 'assistant', ?, ?, 'complete')",
                            (session_id, result_summary.strip(), task_id),
                        )
                except Exception:
                    pass
            terminal_ev = {
                "type": "task_update",
                "task_id": task_id,
                "task": orch.get_task(task_id),
                "status": (orch.get_task(task_id) or {}).get("status"),
            }
            try:
                eid = self._persist_event(task_id, terminal_ev)
                terminal_ev["id"] = eid
            except Exception:
                pass
            await self._broadcast(task_id, terminal_ev)
        except asyncio.CancelledError:
            orch.update_task_status(task_id, "interrupted")
            ev = {"type": "task_update", "task_id": task_id, "status": "interrupted"}
            try:
                eid = self._persist_event(task_id, ev)
                ev["id"] = eid
            except Exception:
                pass
            await self._broadcast(task_id, ev)
        except Exception as e:
            orch.update_task_status(task_id, "failed")
            try:
                with db_session() as conn:
                    conn.execute("UPDATE autonomous_tasks SET last_error = ? WHERE id = ?", (str(e), task_id))
            except Exception:
                pass
            ev = {"type": "task_update", "task_id": task_id, "status": "failed", "message": str(e)}
            try:
                eid = self._persist_event(task_id, ev)
                ev["id"] = eid
            except Exception:
                pass
            await self._broadcast(task_id, ev)
        finally:
            self._runners.pop(task_id, None)
            self._controls.pop(task_id, None)


def get_task_runner() -> TaskRunner:
    return TaskRunner()
