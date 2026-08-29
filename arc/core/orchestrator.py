"""Phase 2 — Orchestrator: the ARC agent loop.

Wires the LLM router, tool registry and memory into a tool-using agent:

    user message -> LLM (with tool schemas) -> tool calls -> results ->
    LLM again -> ... -> final answer

The loop is bounded (``max_iterations``) and every tool call is gated by the
permission guard inside the registry, so the agent can never side-step
approval flows.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..core.llm import LLMRouter, ModelUnavailableError, ModelRole
from ..core.memory import ConversationMemory, LongTermMemory
from ..profile.profile import Profile
from ..tools.base import ToolRegistry, ToolResult

logger = logging.getLogger("arc.orchestrator")

MAX_TOOL_ITERATIONS = 25
TOOL_RESULT_CHAR_LIMIT = 6000


def build_system_prompt(profile: Optional[Profile],
                        long_term: Optional[LongTermMemory],
                        tool_names: List[str],
                        skills: Optional[Any] = None) -> str:
    """Compose the agent's identity, context and rules.

    ``skills`` (optional) is a :class:`arc.skills.index.SkillLibrary`. When
    present it adds a "# Skill library" section that lists the available
    categories and nudges the agent to discover and load a relevant SKILL.md
    before tackling domain-specific or complex tasks.
    """
    today = datetime.now().strftime("%A, %d %B %Y")
    parts: List[str] = [
        f"You are ARC, a personal AI assistant. Today is {today}.",
        "You help with email triage, internship hunting, web research, "
        "and computer tasks — proactively, but always safely.",
        "",
        "Operating rules:",
        "- Prefer tools over guessing; gather facts before answering.",
        "- Tool output marked ERROR means the action failed: adapt instead of retrying blindly.",
        "- NEVER send emails, submit applications, or run destructive commands unless the "
        "user explicitly asked for exactly that. Drafts are fine; sending needs the user.",
        "- If a tool reports that a dependency is missing, tell the user the install "
        "command instead of retrying.",
        "- Be concise. Use short paragraphs or bullet lists.",
        "- If denied permission for an action, explain what was blocked and ask the user "
        "how to proceed.",
    ]
    if profile is not None and not profile.is_placeholder():
        parts.append("\n# User profile\n" + profile.summarize())
    elif profile is not None:
        parts.append("\n(User profile is still the placeholder template — suggest "
                     "`arc init` to fill it in when relevant.)")
    if long_term is not None:
        context = long_term.context_block()
        if context:
            parts.append("\n# Long-term memory\n" + context)
    skills_block = _skills_prompt_block(skills)
    if skills_block:
        parts.append(skills_block)
    if tool_names:
        parts.append("\nAvailable tools: " + ", ".join(tool_names))
    return "\n".join(parts)


def _skills_prompt_block(skills: Optional[Any]) -> str:
    """Build the '# Skill library' prompt section, or '' when unavailable."""
    if skills is None or not getattr(skills, "count", 0):
        return ""
    categories = getattr(skills, "categories_with_counts", None)
    lines = ["\n# Skill & component library",
             "You have a library of expert SKILL.md skills and a catalog of AI "
             "agent components (MCP servers, loops, subagents, hooks, plugins, "
             "prompts, CLI tools). For domain-specific, complex, or "
             "infrastructure tasks, FIRST look for a relevant entry: "
             "call `skills.search(\"<topic>\")` to find candidates "
             "(optionally filtered by `kind`), then `skills.load(\"<name>\")` "
             "to bring a skill's instructions into scope, or an MCP/loop/hook/"
             "tool's description and exact install command, before answering. "
             "Use `skills.categories` to browse. "
             "Only use entries genuinely relevant to the task; do not load them "
             "for trivial questions."]
    if categories is not None:
        try:
            counts = categories()
            if counts:
                lines.append("Available categories:")
                for category, count in counts.items():
                    lines.append(f"- {category} ({count})")
        except Exception:  # noqa: BLE001 - prompt building must never crash
            logger.debug("Could not read skill categories for prompt",
                         exc_info=True)
    return "\n".join(lines)


@dataclass
class AgentTurn:
    """Result of one full agent run (possibly several LLM round-trips)."""
    reply: str
    tool_calls: int = 0
    iterations: int = 0
    model: str = ""
    tool_log: List[Dict[str, Any]] = field(default_factory=list)


class Orchestrator:
    """Tool-using agent loop around the LLM router and tool registry."""

    def __init__(
        self,
        router: LLMRouter,
        registry: ToolRegistry,
        profile: Optional[Profile] = None,
        long_term: Optional[LongTermMemory] = None,
        max_iterations: int = MAX_TOOL_ITERATIONS,
        on_event: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        skills: Optional[Any] = None,
    ) -> None:
        self.router = router
        self.registry = registry
        self.profile = profile
        self.long_term = long_term
        self.max_iterations = max_iterations
        self.on_event = on_event
        self.skills = skills
        self.conversation = ConversationMemory()
        self.system_prompt = build_system_prompt(
            profile, long_term, registry.names(), skills)

    # ------------------------------------------------------------------ loop
    def handle(self, user_message: str) -> AgentTurn:
        """Process one user message through the full agent loop."""
        self.conversation.add("user", user_message)
        turn = AgentTurn(reply="")
        messages = self._messages()

        for iteration in range(1, self.max_iterations + 1):
            turn.iterations = iteration
            try:
                response = self.router.complete(
                    messages,
                    role=ModelRole.REASONING,
                    tools=self.registry.schemas(),
                )
            except ModelUnavailableError as exc:
                turn.reply = (
                    "I could not reach any LLM backend. Check that OmniRoute (or "
                    f"your OpenAI-compatible gateway) is running.\nDetails: {exc}")
                self.conversation.add("assistant", turn.reply)
                return turn
            except Exception as exc:  # noqa: BLE001
                logger.exception("LLM call failed")
                turn.reply = f"The model request failed unexpectedly: {exc}"
                self.conversation.add("assistant", turn.reply)
                return turn

            turn.model = response.model
            if not response.has_tool_calls:
                turn.reply = response.content or "(no response)"
                self.conversation.add("assistant", turn.reply)
                return turn

            # Record the assistant's tool-call message verbatim, then execute.
            messages.append(self._assistant_message(response))
            self.conversation.messages.append(self._assistant_message(response))
            for call in response.tool_calls:
                result = self.registry.call(call.name, call.arguments)
                rendered = self._render_result(result)
                # Gateways may sanitize tool names (dots -> underscores); show
                # the resolved name in logs/UI, but keep the model's verbatim
                # name in API-facing history (providers expect it back).
                resolved = self.registry.resolve(call.name)
                display_name = resolved.name if resolved else call.name
                turn.tool_calls += 1
                turn.tool_log.append({
                    "tool": display_name,
                    "arguments": call.arguments,
                    "ok": result.ok,
                    "output": result.output[:500],
                })
                self._emit("tool_result", {"tool": display_name, "ok": result.ok,
                                           "output": result.output[:2000]})
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": rendered,
                }
                messages.append(tool_message)
                self.conversation.add_tool_result(call.id, rendered)

        turn.reply = ("I hit my tool-use limit for this request "
                      f"({self.max_iterations} steps). Here is what I gathered so far:\n"
                      + (turn.tool_log[-1]["output"] if turn.tool_log else ""))
        self.conversation.add("assistant", turn.reply)
        return turn

    def reset(self) -> None:
        self.conversation.clear()
        self.system_prompt = build_system_prompt(
            self.profile, self.long_term, self.registry.names(), self.skills)

    # ------------------------------------------------------------- internals
    def _messages(self) -> List[Dict[str, Any]]:
        """Fresh message list: system prompt + conversation snapshot."""
        return [{"role": "system", "content": self.system_prompt}] + \
            self.conversation.snapshot()

    def _emit(self, event: str, data: Dict[str, Any]) -> None:
        if self.on_event is not None:
            try:
                self.on_event(event, data)
            except Exception:  # noqa: BLE001 - events must not break the loop
                logger.exception("on_event handler failed")

    @staticmethod
    def _assistant_message(response: Any) -> Dict[str, Any]:
        """Rebuild the assistant message (including tool calls) for the API."""
        message: Dict[str, Any] = {"role": "assistant",
                                   "content": response.content or None}
        if response.has_tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in response.tool_calls
            ]
        return message

    @staticmethod
    def _render_result(result: ToolResult) -> str:
        rendered = result.render()
        if len(rendered) > TOOL_RESULT_CHAR_LIMIT:
            rendered = rendered[:TOOL_RESULT_CHAR_LIMIT] + "\n... (truncated)"
        return rendered


__all__ = ["Orchestrator", "AgentTurn", "build_system_prompt"]
