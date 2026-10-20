"""
Automated Verification Suite for the restart-safe TaskRunner:
event persistence/replay, pause/resume/abort checkpoints, retries and recovery.
"""
import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

_TEMP = tempfile.TemporaryDirectory(prefix="grok-task-runner-tests-")
os.environ["GROK_DB_PATH"] = str(Path(_TEMP.name) / "grok-test.db")
os.environ["GROK_STATE_FILE"] = str(Path(_TEMP.name) / "active-model.json")

import engine.task_orchestrator as orchestrator_module
from memory.database import init_db
from engine.task_orchestrator import TaskOrchestrator
from engine.task_runner import TaskRunner, TaskControl

orchestrator_module.WORKSPACES_DIR = Path(_TEMP.name) / "workspaces"
orchestrator_module.WORKSPACES_DIR.mkdir(parents=True, exist_ok=True)


class TestTaskRunner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    @classmethod
    def tearDownClass(cls):
        _TEMP.cleanup()

    def _new_task(self, goal="unit-test task"):
        task = TaskOrchestrator.create_task(goal=goal, task_type="software_project")
        TaskOrchestrator.set_subgoals(task["id"], [
            {"title": "Scaffold project", "description": "Create project files.", "tool_preference": "workspace"},
        ])
        return task

    def test_control_pause_resume_abort(self):
        async def scenario():
            ctrl = TaskControl()
            self.assertFalse(ctrl.is_paused())
            ctrl.pause()
            self.assertTrue(ctrl.is_paused())
            waiter = asyncio.create_task(ctrl.checkpoint())
            await asyncio.sleep(0.1)
            self.assertFalse(waiter.done())
            ctrl.resume()
            await asyncio.wait_for(waiter, timeout=3.0)
            ctrl.abort()
            with self.assertRaises(asyncio.CancelledError):
                await ctrl.checkpoint()

        asyncio.run(scenario())

    def test_events_persist_and_replay(self):
        task = self._new_task()
        runner = TaskRunner()
        runner._persist_event(task["id"], {"type": "task_update", "status": "running", "message": "hello"})
        runner._persist_event(task["id"], {"type": "subgoal_update", "subgoal_idx": 1, "status": "running"})

        events = TaskOrchestrator.get_events(task["id"])
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["type"], "task_update")
        self.assertIn("id", events[0])
        # since_id replay filter
        first_id = events[0]["id"]
        later = TaskOrchestrator.get_events(task["id"], since_id=first_id)
        self.assertEqual(len(later), 1)
        self.assertEqual(later[0]["type"], "subgoal_update")

    def test_retry_resets_failed_subgoal(self):
        task = self._new_task()
        TaskOrchestrator.update_subgoal(task["id"], 1, "failed", result_summary="boom", increment_attempt=True)
        TaskOrchestrator.update_task_status(task["id"], "failed")

        ok = TaskOrchestrator.retry_task(task["id"])
        self.assertTrue(ok)
        refreshed = TaskOrchestrator.get_task(task["id"])
        self.assertEqual(refreshed["status"], "queued")
        self.assertEqual(refreshed["subgoals"][0]["status"], "pending")
        self.assertEqual(refreshed["subgoals"][0]["attempts"], 0)

    def test_recovery_marks_running_interrupted(self):
        task = self._new_task()
        TaskOrchestrator.update_task_status(task["id"], "running")
        TaskOrchestrator.update_subgoal(task["id"], 1, "running")

        asyncio.run(TaskRunner.recover_on_startup())

        refreshed = TaskOrchestrator.get_task(task["id"])
        self.assertEqual(refreshed["status"], "interrupted")
        self.assertEqual(refreshed["subgoals"][0]["status"], "pending")

    def test_full_run_workspace_mode(self):
        task = self._new_task(goal="build a tiny project")
        runner = TaskRunner()

        async def scenario():
            ok = await runner.start(task["id"])
            self.assertTrue(ok)
            collected = []
            async for ev in runner.subscribe(task["id"], since_id=0):
                collected.append(ev)
                if ev.get("type") == "task_update" and ev.get("status") in (
                    "completed", "partially_completed", "failed", "aborted", "interrupted"
                ):
                    break
            return collected

        events = asyncio.run(asyncio.wait_for(scenario(), timeout=60))
        types = {e.get("type") for e in events}
        self.assertIn("task_update", types)
        final = TaskOrchestrator.get_task(task["id"])
        self.assertIn(final["status"], ("completed", "partially_completed", "failed"))
        self.assertTrue(final.get("result_summary"))

    def test_runner_shutdown(self):
        task = self._new_task()
        runner = TaskRunner()

        async def scenario():
            await runner.start(task["id"])
            await runner.shutdown()
            self.assertFalse(runner._runners)

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main(verbosity=2)
