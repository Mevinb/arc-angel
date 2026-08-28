"""Tool framework: registry gating, ToolResult, shell/python tools."""

from __future__ import annotations

from typing import Any

from arc.safety.permissions import Action, PermissionGuard, RiskLevel
from arc.tools.base import Tool, ToolRegistry, ToolResult
from arc.tools.computer import PythonTool, ShellTool


class EchoTool(Tool):
    name = "test.echo"
    description = "Echo a message."
    risk = RiskLevel.GREEN
    parameters = {"properties": {"message": {"type": "string"}},
                  "required": ["message"]}

    def run(self, message: str = "", **_: Any) -> ToolResult:
        return ToolResult.success(f"echo: {message}")


class YellTool(Tool):
    name = "test.yell"
    description = "A YELLOW-level tool."
    risk = RiskLevel.YELLOW
    parameters = {"properties": {}, "required": []}

    def run(self, **_: Any) -> ToolResult:
        return ToolResult.success("yelled")


def _registry(mode: str = "interactive",
              approver=None) -> ToolRegistry:
    guard = PermissionGuard(mode=mode, approver=approver)
    return ToolRegistry(guard)


class TestToolResult:
    def test_success_and_failure(self):
        ok = ToolResult.success("done", extra=1)
        assert ok.ok and ok.render() == "done" and ok.data == {"extra": 1}
        bad = ToolResult.failure("nope")
        assert not bad.ok and bad.render() == "ERROR: nope"


class TestRegistry:
    def test_register_and_schemas(self):
        registry = _registry()
        registry.register(EchoTool())
        schema = registry.schemas()[0]
        assert schema["function"]["name"] == "test.echo"
        assert schema["function"]["parameters"]["required"] == ["message"]
        assert registry.names() == ["test.echo"]

    def test_duplicate_registration_rejected(self):
        registry = _registry()
        registry.register(EchoTool())
        import pytest
        with pytest.raises(ValueError):
            registry.register(EchoTool())

    def test_call_green_tool(self):
        registry = _registry()
        registry.register(EchoTool())
        result = registry.call("test.echo", {"message": "hi"})
        assert result.ok and result.output == "echo: hi"

    def test_unknown_tool(self):
        registry = _registry()
        result = registry.call("nope.nope", {})
        assert not result.ok and "Unknown tool" in result.output

    def test_unknown_dotted_tool_not_resolved(self):
        registry = _registry()
        registry.register(EchoTool())
        result = registry.call("test.absent", {})
        assert not result.ok and "Unknown tool" in result.output


class TestSanitizedNameResolution:
    """Gateways fronting Anthropic/Gemini APIs sanitize dots to underscores
    (sometimes with a hex suffix) and don't restore them. The registry maps
    those names back to the registered dotted tools."""

    def test_underscore_name_resolves(self):
        registry = _registry()
        registry.register(EchoTool())
        result = registry.call("test_echo", {"message": "hi"})
        assert result.ok and result.output == "echo: hi"

    def test_underscore_name_with_hash_suffix_resolves(self):
        registry = _registry()
        registry.register(EchoTool())
        result = registry.call("test_echo_64ed5d80c28d3d059d9ddeea9032",
                               {"message": "hi"})
        assert result.ok and result.output == "echo: hi"

    def test_multi_segment_name_resolves(self):
        class MultiWordTool(Tool):
            name = "test.multi_word"
            description = "Multi-segment name."
            parameters = {"properties": {}, "required": []}

            def run(self, **_: Any) -> ToolResult:
                return ToolResult.success("ok")

        registry = _registry()
        registry.register(MultiWordTool())
        assert registry.call("test_multi_word", {}).ok

    def test_ambiguous_sanitized_names_fail(self):
        class DottedTool(Tool):
            name = "amb.a.b"
            description = "Sanitizes to amb_a_b."
            parameters = {"properties": {}, "required": []}

            def run(self, **_: Any) -> ToolResult:
                return ToolResult.success("dotted")

        class UnderscoreTool(Tool):
            name = "amb.a_b"
            description = "Also sanitizes to amb_a_b."
            parameters = {"properties": {}, "required": []}

            def run(self, **_: Any) -> ToolResult:
                return ToolResult.success("underscore")

        registry = _registry()
        registry.register(DottedTool())
        registry.register(UnderscoreTool())
        result = registry.call("amb_a_b", {})
        assert not result.ok and "Unknown tool" in result.output

    def test_resolve_returns_registered_tool_object(self):
        registry = _registry()
        echo = registry.register(EchoTool())
        assert registry.resolve("test_echo") is echo
        assert registry.resolve("test.echo") is echo
        assert registry.resolve("test_echo_deadbeefcafe") is echo
        assert registry.resolve("unrelated_name") is None

    def test_hooks_receive_resolved_name(self):
        seen = []
        registry = _registry()
        registry.register(EchoTool())
        registry.on_result(lambda name, kwargs, result: seen.append((name, result.ok)))
        registry.call("test_echo_64ed5d80c28d3d059d9ddeea9032", {"message": "x"})
        assert seen == [("test.echo", True)]

    def test_yellow_denied_in_auto_mode(self):
        registry = _registry(mode="auto")
        registry.register(YellTool())
        result = registry.call("test.yell", {})
        assert not result.ok
        assert "Permission denied" in result.output

    def test_yellow_approved_via_approver(self):
        def approver(action: Action) -> bool:
            return action.description.startswith("test.yell")

        registry = _registry(mode="interactive", approver=approver)
        registry.register(YellTool())
        result = registry.call("test.yell", {})
        assert result.ok and result.output == "yelled"

    def test_tool_exceptions_become_failures(self):
        class BoomTool(Tool):
            name = "test.boom"
            description = "Always crashes."
            parameters = {"properties": {}, "required": []}

            def run(self, **_: Any) -> ToolResult:
                raise ValueError("kaboom")

        registry = _registry()
        registry.register(BoomTool())
        result = registry.call("test.boom", {})
        assert not result.ok and "kaboom" in result.output

    def test_result_hooks_fire(self):
        seen = []
        registry = _registry()
        registry.register(EchoTool())
        registry.on_result(lambda name, kwargs, result: seen.append((name, result.ok)))
        registry.call("test.echo", {"message": "x"})
        assert seen == [("test.echo", True)]


class TestShellTool:
    def test_simple_command(self):
        tool = ShellTool()
        result = tool.run(command="echo hello-world")
        assert result.ok and "hello-world" in result.output

    def test_blocked_command_refused(self):
        tool = ShellTool()
        result = tool.run(command="rm -rf /")
        assert not result.ok and "blocked" in result.output.lower()

    def test_exit_code_reported(self):
        tool = ShellTool()
        result = tool.run(command="exit 3")
        assert result.ok and result.data["exit_code"] == 3

    def test_timeout(self):
        tool = ShellTool()
        result = tool.run(command="sleep 5", timeout_seconds=1)
        assert not result.ok and "timed out" in result.output


class TestPythonTool:
    def test_runs_code(self):
        result = PythonTool().run(code="print(21 * 2)")
        assert result.ok and "42" in result.output

    def test_stderr_captured(self):
        result = PythonTool().run(code="import sys; print('bad', file=sys.stderr)")
        assert result.ok and "bad" in result.output

    def test_empty_code(self):
        assert not PythonTool().run(code="").ok
