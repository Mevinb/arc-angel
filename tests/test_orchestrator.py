"""Orchestrator: the tool-using agent loop."""

from __future__ import annotations

import json

from arc.core.orchestrator import Orchestrator, build_system_prompt
from arc.safety.permissions import PermissionGuard, RiskLevel
from arc.tools.base import Tool, ToolRegistry, ToolResult
from tests.conftest import (ExplodingClient, FakeMessage, FakeResponse,
                            FakeToolCall, make_router)


class EchoTool(Tool):
    name = "test.echo"
    description = "Echo a message."
    parameters = {"properties": {"message": {"type": "string"}},
                  "required": ["message"]}

    def run(self, message: str = "", **_) -> ToolResult:
        return ToolResult.success(f"echo: {message}")


class YellTool(Tool):
    name = "test.yell"
    description = "YELLOW-level tool."
    risk = RiskLevel.YELLOW
    parameters = {"properties": {}, "required": []}

    def run(self, **_) -> ToolResult:
        return ToolResult.success("yelled")


def _registry(mode: str = "interactive", approver=None) -> ToolRegistry:
    registry = ToolRegistry(PermissionGuard(mode=mode, approver=approver))
    registry.register(EchoTool())
    registry.register(YellTool())
    return registry


def _tool_call_response(call_id: str, name: str, arguments: dict) -> FakeResponse:
    return FakeResponse(FakeMessage(content="", tool_calls=[
        FakeToolCall(call_id, name, arguments)]))


class TestAgentLoop:
    def test_direct_answer_without_tools(self):
        router = make_router([FakeResponse(FakeMessage("Hello there!"))])
        agent = Orchestrator(router, _registry())
        turn = agent.handle("hi")
        assert turn.reply == "Hello there!"
        assert turn.tool_calls == 0
        assert turn.iterations == 1

    def test_tool_call_round_trip(self):
        router = make_router([
            _tool_call_response("call-1", "test.echo", {"message": "world"}),
            FakeResponse(FakeMessage("The tool said: echo: world")),
        ])
        agent = Orchestrator(router, _registry())
        turn = agent.handle("say world via the tool")
        assert turn.reply == "The tool said: echo: world"
        assert turn.tool_calls == 1
        assert turn.tool_log[0]["tool"] == "test.echo"
        assert turn.tool_log[0]["ok"] is True
        # The tool result must be visible to the second model call.
        second_request = router.client.completions.requests[1]
        tool_messages = [m for m in second_request["messages"] if m.get("role") == "tool"]
        assert tool_messages and "echo: world" in tool_messages[0]["content"]
        # The assistant tool-call message is replayed with valid JSON arguments.
        assistant_calls = [m for m in second_request["messages"]
                           if m.get("role") == "assistant" and m.get("tool_calls")]
        assert assistant_calls
        assert json.loads(assistant_calls[0]["tool_calls"][0]["function"]
                          ["arguments"]) == {"message": "world"}

    def test_denied_tool_surfaces_error(self):
        router = make_router([
            _tool_call_response("call-1", "test.yell", {}),
            FakeResponse(FakeMessage("I was blocked from yelling.")),
        ])
        agent = Orchestrator(router, _registry(mode="auto"))
        turn = agent.handle("yell please")
        assert turn.tool_log[0]["ok"] is False
        assert "Permission denied" in turn.tool_log[0]["output"]
        second_request = router.client.completions.requests[1]
        tool_messages = [m for m in second_request["messages"] if m.get("role") == "tool"]
        assert "Permission denied" in tool_messages[0]["content"]

    def test_unknown_tool_handled_gracefully(self):
        router = make_router([
            _tool_call_response("call-1", "no.such_tool", {}),
            FakeResponse(FakeMessage("That tool does not exist.")),
        ])
        agent = Orchestrator(router, _registry())
        turn = agent.handle("use a fake tool")
        assert turn.tool_log[0]["ok"] is False
        assert turn.reply == "That tool does not exist."

    def test_max_iterations_bounded(self):
        endless = [_tool_call_response(f"call-{i}", "test.echo", {"message": "x"})
                   for i in range(50)]
        router = make_router(endless)
        agent = Orchestrator(router, _registry(), max_iterations=3)
        turn = agent.handle("loop forever")
        assert turn.iterations == 3
        assert turn.tool_calls == 3
        assert "tool-use limit" in turn.reply

    def test_model_unavailable_message(self):
        router = make_router(client=ExplodingClient())
        agent = Orchestrator(router, _registry())
        turn = agent.handle("hello?")
        assert "could not reach any LLM backend" in turn.reply

    def test_conversation_memory_carries_context(self):
        router = make_router([FakeResponse(FakeMessage("first")),
                              FakeResponse(FakeMessage("second"))])
        agent = Orchestrator(router, _registry())
        agent.handle("first question")
        agent.handle("second question")
        second_request = router.client.completions.requests[1]
        roles = [m["role"] for m in second_request["messages"]]
        assert roles == ["system", "user", "assistant", "user"]

    def test_reset_clears_conversation(self):
        router = make_router([FakeResponse(FakeMessage("ok"))])
        agent = Orchestrator(router, _registry())
        agent.handle("remember this")
        agent.reset()
        assert agent.conversation.snapshot() == []


class TestSystemPrompt:
    def test_contains_profile_and_memory(self, profile):
        from arc.core.memory import LongTermMemory
        from arc.db.database import Database
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(Path(tmp) / "t.db")
            memory = LongTermMemory(db)
            memory.remember("user.timezone", "Europe/Berlin")
            prompt = build_system_prompt(profile, memory, ["test.echo"])
            assert "Test User" in prompt
            assert "Europe/Berlin" in prompt
            assert "test.echo" in prompt
            db.close()

    def test_placeholder_profile_hint(self, profile):
        profile.data["name"] = "Your Name"
        prompt = build_system_prompt(profile, None, [])
        assert "placeholder" in prompt

    def test_skill_library_section_when_present(self):
        class FakeSkills:
            count = 500
            def categories_with_counts(self):
                return {"Testing": 20, "DevOps & Cloud": 300}
        prompt = build_system_prompt(None, None, ["skills.search"], skills=FakeSkills())
        assert "# Skill & component library" in prompt
        assert 'skills.search("<topic>")' in prompt
        assert 'skills.load("<name>")' in prompt
        assert "Available categories:" in prompt
        assert "DevOps & Cloud (300)" in prompt

    def test_no_skill_section_without_library(self):
        prompt = build_system_prompt(None, None, ["x"])
        assert "# Skill" not in prompt

    def test_skill_section_absent_for_empty_library(self):
        class FakeSkills:
            count = 0
            def categories_with_counts(self):
                return {}
        prompt = build_system_prompt(None, None, ["skills.search"], skills=FakeSkills())
        assert "# Skill library" not in prompt
