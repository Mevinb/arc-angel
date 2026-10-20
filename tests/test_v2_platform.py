"""End-to-end smoke tests for the v2 platform API, UI routes and mobile flow."""
from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import threading
import time
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import AsyncMock, patch

_TEMP = tempfile.TemporaryDirectory(prefix="grok-v2-tests-")
os.environ["GROK_DB_PATH"] = str(Path(_TEMP.name) / "grok-test.db")
os.environ["GROK_STATE_FILE"] = str(Path(_TEMP.name) / "active-model.json")
os.environ["GROK_WEB_SEARCH"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
from config import WORKSPACES_DIR  # noqa: E402
from services import backup as backup_module  # noqa: E402
from services import builtin_tools as builtin_tools_module  # noqa: E402
from services.backup import BackupService  # noqa: E402
from services.builtin_tools import _files_write  # noqa: E402
from services.files import FileService  # noqa: E402
from services.tools import ToolContext  # noqa: E402


class PlatformApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client_cm = TestClient(server.app)
        cls.client = cls.client_cm.__enter__()
        cls.workspaces: set[Path] = set()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client_cm.__exit__(None, None, None)
        for path in cls.workspaces:
            shutil.rmtree(path, ignore_errors=True)
        _TEMP.cleanup()

    def workspace(self, label: str) -> Path:
        path = WORKSPACES_DIR / f"test-{label}-{uuid.uuid4().hex[:8]}"
        path.mkdir(parents=True, exist_ok=True)
        self.workspaces.add(path)
        return path

    def test_app_lock_enforces_unlock(self) -> None:
        pairing = self.client.post(
            "/api/v2/mobile/pairing", json={"name": "Locked phone", "platform": "mobile"}
        )
        self.assertEqual(pairing.status_code, 200, pairing.text)
        pairing_data = pairing.json()
        paired = self.client.post(
            "/api/v2/mobile/pair",
            json={"device_id": pairing_data["device_id"], "code": pairing_data["pairing_code"]},
        )
        self.assertEqual(paired.status_code, 200, paired.text)
        mobile_token = paired.json()["token"]

        enabled = self.client.post("/api/v2/safety/lock", json={"pin": "2468"})
        self.assertEqual(enabled.status_code, 200, enabled.text)
        self.assertTrue(enabled.json()["enabled"])
        self.client.cookies.clear()

        blocked = self.client.get("/api/v2/overview")
        self.assertEqual(blocked.status_code, 423, blocked.text)
        page = self.client.get("/", follow_redirects=False)
        self.assertEqual(page.status_code, 303)
        self.assertIn("/unlock?next=", page.headers["location"])
        self.assertEqual(self.client.get("/unlock").status_code, 200)
        self.assertEqual(
            self.client.get("/api/v2/safety/lock-status").json()["lock_enabled"], True
        )
        wrong = self.client.post("/api/v2/safety/unlock", json={"pin": "0000"})
        self.assertEqual(wrong.status_code, 401)
        mobile_me = self.client.get(
            "/api/v2/mobile/me", headers={"X-Mobile-Token": mobile_token}
        )
        self.assertEqual(mobile_me.status_code, 200, mobile_me.text)
        blocked_pairing = self.client.post(
            "/api/v2/mobile/pairing", json={"name": "Intruder"}
        )
        self.assertEqual(blocked_pairing.status_code, 423, blocked_pairing.text)

        unlocked = self.client.post("/api/v2/safety/unlock", json={"pin": "2468"})
        self.assertEqual(unlocked.status_code, 200, unlocked.text)
        self.assertIn("grok_unlock", self.client.cookies)
        self.assertEqual(self.client.get("/api/v2/overview").status_code, 200)

        disabled = self.client.post("/api/v2/safety/lock", json={"pin": None})
        self.assertEqual(disabled.status_code, 200, disabled.text)
        self.assertFalse(disabled.json()["enabled"])
        self.client.cookies.clear()
        self.assertEqual(self.client.get("/api/v2/overview").status_code, 200)
        revoked = self.client.delete(f"/api/v2/mobile/devices/{pairing_data['device_id']}")
        self.assertEqual(revoked.status_code, 200, revoked.text)

    def test_task_workspace_outside_roots_rejected(self) -> None:
        outside = Path(tempfile.gettempdir()) / f"grok-outside-{uuid.uuid4().hex[:8]}"
        created = self.client.post(
            "/api/v2/tasks",
            json={"title": "Outside", "goal": "Should be rejected", "workspace_dir": str(outside)},
        )
        self.assertEqual(created.status_code, 403, created.text)
        self.assertFalse(outside.exists())
        self.assertIn("approved roots", created.text)

    def test_health_meta_and_ui_routes(self) -> None:
        health = self.client.get("/api/v2/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["ok"])
        self.assertIn("database", health.json())

        meta = self.client.get("/api/v2/meta")
        self.assertEqual(meta.status_code, 200)
        self.assertEqual(meta.json()["database"], os.environ["GROK_DB_PATH"])

        overview = self.client.get("/api/v2/overview")
        self.assertEqual(overview.status_code, 200)
        self.assertIsInstance(overview.json()["recent_runs"], list)

        platform = self.client.get("/platform")
        self.assertEqual(platform.status_code, 200)
        self.assertIn("Command Center", platform.text)
        mobile = self.client.get("/mobile")
        self.assertEqual(mobile.status_code, 200)
        self.assertIn("GROK", mobile.text)
        self.assertEqual(self.client.get("/static/js/platform.js").status_code, 200)
        self.assertEqual(self.client.get("/static/js/mobile.js").status_code, 200)

    def test_mobile_pair_command_brief_and_revoke(self) -> None:
        pairing = self.client.post(
            "/api/v2/mobile/pairing",
            json={"name": "Test phone", "platform": "mobile"},
        )
        self.assertEqual(pairing.status_code, 200)
        pairing_data = pairing.json()
        self.assertIn("pairing_code", pairing_data)

        paired = self.client.post(
            "/api/v2/mobile/pair",
            json={"device_id": pairing_data["device_id"], "code": pairing_data["pairing_code"]},
        )
        self.assertEqual(paired.status_code, 200, paired.text)
        token = paired.json()["token"]
        headers = {"X-Mobile-Token": token}

        me = self.client.get("/api/v2/mobile/me", headers=headers)
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["id"], pairing_data["device_id"])

        command = self.client.post(
            "/api/v2/mobile/command", headers=headers,
            json={"text": "note: a mobile-created test note"},
        )
        self.assertEqual(command.status_code, 200, command.text)
        self.assertEqual(command.json()["kind"], "note")

        brief = self.client.get("/api/v2/mobile/brief", headers=headers)
        self.assertEqual(brief.status_code, 200)
        self.assertIn("markdown", brief.json())
        self.assertIn("data", brief.json())

        devices = self.client.get("/api/v2/mobile/devices")
        self.assertEqual(devices.status_code, 200)
        self.assertTrue(any(d["id"] == pairing_data["device_id"] for d in devices.json()))

        revoked = self.client.delete(f"/api/v2/mobile/devices/{pairing_data['device_id']}")
        self.assertEqual(revoked.status_code, 200)
        self.assertEqual(self.client.get("/api/v2/mobile/me", headers=headers).status_code, 401)

        second = self.client.post("/api/v2/mobile/pairing", json={"name": "Brute force test"}).json()
        wrong = "000000" if second["pairing_code"] != "000000" else "111111"
        for _ in range(10):
            attempt = self.client.post(
                "/api/v2/mobile/pair",
                json={"device_id": second["device_id"], "code": wrong},
            )
            self.assertEqual(attempt.status_code, 400, attempt.text)
        blocked = self.client.post(
            "/api/v2/mobile/pair",
            json={"device_id": second["device_id"], "code": second["pairing_code"]},
        )
        self.assertEqual(blocked.status_code, 400, blocked.text)
        self.assertIn("Too many pairing attempts", blocked.text)
        self.client.delete(f"/api/v2/mobile/devices/{second['device_id']}")

    def test_task_completion_events_and_recent_runs(self) -> None:
        workspace = self.workspace("run")
        fake = AsyncMock(return_value='{"final": "Completed by the test model."}')
        with patch("services.agent_runtime.ModelService.complete", new=fake):
            created = self.client.post(
                "/api/v2/tasks",
                json={
                    "title": "API completion test",
                    "goal": "Finish this test task.",
                    "agent_id": "orchestrator",
                    "workspace_dir": str(workspace),
                    "auto_start": True,
                    "budget": {"max_steps": 4},
                },
            )
            self.assertEqual(created.status_code, 200, created.text)
            task_id = created.json()["id"]
            deadline = time.time() + 15
            task = created.json()
            while time.time() < deadline:
                task = self.client.get(f"/api/v2/tasks/{task_id}").json()
                if task["status"] in {"completed", "partially_completed", "failed", "cancelled"}:
                    break
                time.sleep(0.05)
        self.assertEqual(task["status"], "completed", task.get("error"))
        self.assertTrue(task["latest_run"])
        run_id = task["latest_run"]["id"]

        events = self.client.get(f"/api/v2/runs/{run_id}/events").json()
        types = [event["type"] for event in events]
        self.assertLessEqual(types.count("run.completed"), 1)
        self.assertLessEqual(types.count("run.failed"), 1)
        self.assertIn("run.started", types)
        audit = self.client.get("/api/v2/audit/verify").json()
        self.assertTrue(audit["ok"], audit)
        self.assertGreater(audit["events"], 0)

        overview = self.client.get("/api/v2/overview").json()
        recent = overview["recent_runs"]
        self.assertTrue(any(run["id"] == run_id for run in recent))
        match = next(run for run in recent if run["id"] == run_id)
        self.assertEqual(match["title"], "API completion test")

        knowledge = self.client.get("/api/v2/knowledge/stats").json()
        self.assertIn("embedded_chunks", knowledge)
        self.assertEqual(knowledge["embedding_dim"], 384)

    def test_approval_pauses_and_resumes_run(self) -> None:
        workspace = self.workspace("approval")
        prompts = [
            '{"plan": ["Run a gated command", "Finish"]}',
            '{"tool": "shell.run", "arguments": {"command": "printf approved > approval-result.txt"}}',
            '{"final": "Approved command completed."}',
        ]
        fake = AsyncMock(side_effect=prompts)
        with patch("services.agent_runtime.ModelService.complete", new=fake):
            created = self.client.post(
                "/api/v2/tasks",
                json={
                    "title": "Approval flow test",
                    "goal": "Run a command that requires approval.",
                    "agent_id": "orchestrator",
                    "workspace_dir": str(workspace),
                    "auto_start": True,
                    "budget": {"max_steps": 4},
                },
            )
            self.assertEqual(created.status_code, 200, created.text)
            task_id = created.json()["id"]
            run_id = created.json()["latest_run"]["id"]

            deadline = time.time() + 10
            task = created.json()
            while time.time() < deadline:
                task = self.client.get(f"/api/v2/tasks/{task_id}").json()
                if task["status"] == "awaiting_approval":
                    break
                time.sleep(0.05)
            self.assertEqual(task["status"], "awaiting_approval", task.get("error"))

            approvals = self.client.get("/api/v2/approvals?status=pending&limit=100").json()
            approval = next((a for a in approvals if a.get("run_id") == run_id), None)
            self.assertIsNotNone(approval)
            resolved = self.client.post(
                f"/api/v2/approvals/{approval['id']}/resolve",
                json={"approved": True, "note": "test approval"},
            )
            self.assertEqual(resolved.status_code, 200, resolved.text)
            self.assertEqual(resolved.json()["approval"]["status"], "approved")

            deadline = time.time() + 20
            while time.time() < deadline:
                task = self.client.get(f"/api/v2/tasks/{task_id}").json()
                if task["status"] in {"completed", "partially_completed", "failed", "cancelled"}:
                    break
                time.sleep(0.1)

        self.assertEqual(task["status"], "completed", task.get("error"))
        self.assertEqual((workspace / "approval-result.txt").read_text(encoding="utf-8"), "approved")
        events = self.client.get(f"/api/v2/runs/{run_id}/events").json()
        types = [event["type"] for event in events]
        self.assertEqual(types.count("run.awaiting_approval"), 1)
        self.assertEqual(types.count("run.completed"), 1)

        reuse = self.client.post(
            "/api/v2/tools/shell.run/execute",
            json={
                "workspace": str(workspace),
                "approval_id": approval["id"],
                "arguments": {"command": "printf bypassed > bypassed.txt"},
            },
        )
        self.assertEqual(reuse.status_code, 200, reuse.text)
        self.assertEqual(reuse.json()["status"], "denied")
        self.assertIn("arguments", reuse.json()["error"])
        self.assertFalse((workspace / "bypassed.txt").exists())

    def test_pause_and_resume_run(self) -> None:
        workspace = self.workspace("pause")
        model_started = threading.Event()
        calls = {"n": 0}

        async def fake_complete(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return '{"plan": ["Wait for a pause", "Finish"]}'
            if calls["n"] == 2:
                model_started.set()
                await asyncio.sleep(0.8)
                return '{"tool": "time.now", "arguments": {}}'
            return '{"final": "Resumed and finished."}'

        with patch("services.agent_runtime.ModelService.complete", new=fake_complete):
            created = self.client.post(
                "/api/v2/tasks",
                json={
                    "title": "Pause flow test",
                    "goal": "Pause and resume cleanly.",
                    "agent_id": "orchestrator",
                    "workspace_dir": str(workspace),
                    "auto_start": True,
                    "budget": {"max_steps": 5},
                },
            )
            self.assertEqual(created.status_code, 200, created.text)
            task_id = created.json()["id"]
            run_id = created.json()["latest_run"]["id"]
            self.assertTrue(model_started.wait(timeout=5))

            paused = self.client.post(f"/api/v2/tasks/{task_id}/pause")
            self.assertEqual(paused.status_code, 200, paused.text)
            self.assertEqual(paused.json()["status"], "paused")

            resumed = self.client.post(f"/api/v2/tasks/{task_id}/resume")
            self.assertEqual(resumed.status_code, 200, resumed.text)

            deadline = time.time() + 20
            task = resumed.json()
            while time.time() < deadline:
                task = self.client.get(f"/api/v2/tasks/{task_id}").json()
                if task["status"] in {"completed", "partially_completed", "failed", "cancelled"}:
                    break
                time.sleep(0.1)

        self.assertEqual(task["status"], "completed", task.get("error"))
        events = self.client.get(f"/api/v2/runs/{run_id}/events").json()
        types = [event["type"] for event in events]
        self.assertEqual(types.count("run.paused"), 1)
        self.assertEqual(types.count("run.resumed"), 1)

    def test_tool_budget_exhaustion_marks_partial(self) -> None:
        workspace = self.workspace("budget")
        prompts = [
            '{"plan": ["Use one tool", "Finish"]}',
            '{"tool": "time.now", "arguments": {}}',
            '{"final": "This should not be reached."}',
        ]
        fake = AsyncMock(side_effect=prompts)
        with patch("services.agent_runtime.ModelService.complete", new=fake):
            created = self.client.post(
                "/api/v2/tasks",
                json={
                    "title": "Budget flow test",
                    "goal": "Exercise the tool-call budget.",
                    "agent_id": "orchestrator",
                    "workspace_dir": str(workspace),
                    "auto_start": True,
                    "budget": {"max_steps": 5, "max_tool_calls": 1},
                },
            )
            self.assertEqual(created.status_code, 200, created.text)
            task_id = created.json()["id"]
            run_id = created.json()["latest_run"]["id"]
            deadline = time.time() + 20
            task = created.json()
            while time.time() < deadline:
                task = self.client.get(f"/api/v2/tasks/{task_id}").json()
                if task["status"] in {"completed", "partially_completed", "failed", "cancelled"}:
                    break
                time.sleep(0.1)

        self.assertEqual(task["status"], "partially_completed", task.get("error"))
        run = self.client.get(f"/api/v2/runs/{run_id}").json()
        self.assertIn("Tool-call budget exhausted", run.get("error") or "")
        self.assertEqual(fake.await_count, 2)

    def test_three_tool_failures_stop_without_extra_model_call(self) -> None:
        workspace = self.workspace("failures")
        prompts = ['{"plan": ["Try a missing file three times"]}'] + [
            '{"tool": "files.read", "arguments": {"path": "missing.txt"}}'
        ] * 4
        fake = AsyncMock(side_effect=prompts)
        with patch("services.agent_runtime.ModelService.complete", new=fake):
            created = self.client.post(
                "/api/v2/tasks",
                json={
                    "title": "Failure guard test",
                    "goal": "Stop after three tool failures.",
                    "agent_id": "orchestrator",
                    "workspace_dir": str(workspace),
                    "auto_start": True,
                    "budget": {"max_steps": 10},
                },
            )
            self.assertEqual(created.status_code, 200, created.text)
            task_id = created.json()["id"]
            run_id = created.json()["latest_run"]["id"]
            deadline = time.time() + 20
            task = created.json()
            while time.time() < deadline:
                task = self.client.get(f"/api/v2/tasks/{task_id}").json()
                if task["status"] in {"completed", "partially_completed", "failed", "cancelled"}:
                    break
                time.sleep(0.1)

        self.assertEqual(task["status"], "partially_completed", task.get("error"))
        self.assertEqual(fake.await_count, 4)
        run = self.client.get(f"/api/v2/runs/{run_id}").json()
        self.assertIn("three consecutive tool failures", run.get("error") or "")
        events = self.client.get(f"/api/v2/runs/{run_id}/events").json()
        types = [event["type"] for event in events]
        self.assertEqual(types.count("run.partially_completed"), 1)
        self.assertEqual(fake.await_count, 4)

    def test_web_fetch_checks_each_redirect_hop(self) -> None:
        class RedirectHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(302)
                self.send_header("Location", "http://169.254.169.254/latest/meta-data/")
                self.end_headers()

            def log_message(self, *args) -> None:
                pass

        server_instance = HTTPServer(("127.0.0.1", 0), RedirectHandler)
        thread = threading.Thread(target=server_instance.serve_forever, daemon=True)
        thread.start()
        try:
            from engine.web_tools import web_fetch

            def allow_local(url: str, allow_private: bool = False) -> bool:
                return "169.254.169.254" not in url

            with patch.object(builtin_tools_module, "is_safe_public_url", side_effect=allow_local):
                result = asyncio.run(
                    web_fetch(f"http://127.0.0.1:{server_instance.server_port}/")
                )
        finally:
            server_instance.shutdown()
            server_instance.server_close()
            thread.join(timeout=2)
        self.assertFalse(result["success"], result)
        self.assertIn("blocked", result["error"].lower())

    def test_backup_create_inspect_stage_and_delete(self) -> None:
        with tempfile.TemporaryDirectory(prefix="grok-backup-test-") as tmp:
            data_dir = Path(tmp) / "data"
            backup_dir = data_dir / "backups"
            backup_dir.mkdir(parents=True)
            secret_dir = data_dir / "secrets"
            secret_dir.mkdir()
            (secret_dir / "master.key").write_text("must-not-be-backed-up", encoding="utf-8")
            (secret_dir / "oauth_state.json").write_text("{}", encoding="utf-8")
            with patch.object(backup_module, "DATA_DIR", data_dir), patch.object(
                backup_module, "BACKUP_DIR", backup_dir
            ):
                created = BackupService.create("suite backup")
                self.assertTrue(Path(created["path"]).exists())
                self.assertTrue(created["sha256"])
                inspected = BackupService.inspect(created["id"])
                self.assertIn("data/grok_memory.db", inspected["files"])
                self.assertIn("data/secrets/oauth_state.json", inspected["files"])
                self.assertNotIn("data/secrets/master.key", inspected["files"])
                staged = BackupService.stage_restore(
                    created["id"], dest=str(Path(tmp) / "restore")
                )
                self.assertTrue((Path(staged["staging"]) / "data" / "grok_memory.db").exists())
                self.assertTrue(BackupService.delete(created["id"]))
                self.assertFalse(Path(created["path"]).exists())

    def test_append_write_preserves_large_file(self) -> None:
        workspace = self.workspace("append")
        target = workspace / "large.txt"
        original = "A" * 30000
        target.write_text(original, encoding="utf-8")
        ctx = ToolContext(workspace_dir=str(workspace))
        result = asyncio.run(_files_write(ctx, {
            "path": "large.txt", "content": "TAIL", "mode": "append",
        }))
        self.assertTrue(result["success"])
        self.assertEqual(target.read_text(encoding="utf-8"), original + "TAIL")
        self.assertEqual(result["bytes"], 30004)

    def test_checkpoint_prune_removes_added_files(self) -> None:
        workspace = self.workspace("checkpoint")
        original = workspace / "original.txt"
        original.write_text("original", encoding="utf-8")
        checkpoint_ids: list[str] = []
        checkpoint = FileService.create_checkpoint("restore test", str(workspace))
        checkpoint_ids.append(checkpoint["id"])
        added = workspace / "added-after.txt"
        added.write_text("temporary", encoding="utf-8")

        merged = FileService.restore_checkpoint(checkpoint["id"], confirm=True)
        checkpoint_ids.append(merged["safety_checkpoint"]["id"])
        self.assertTrue(added.exists())

        pruned = FileService.restore_checkpoint(checkpoint["id"], confirm=True, prune=True)
        checkpoint_ids.append(pruned["safety_checkpoint"]["id"])
        self.assertIn("added-after.txt", pruned["pruned"])
        self.assertFalse(added.exists())
        self.assertEqual(original.read_text(encoding="utf-8"), "original")

        for checkpoint_id in checkpoint_ids:
            FileService.delete_checkpoint(checkpoint_id)


if __name__ == "__main__":
    unittest.main()
