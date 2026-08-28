"""Agent memory: rolling conversation window + long-term facts in SQLite.

Two layers:
- ``ConversationMemory`` — recent turns, trimmed to an approximate character
  budget, so the agent loop stays inside the model context window.
- ``LongTermMemory`` — durable key/value facts backed by the database
  (e.g. "user.timezone": "IST", "recruiter.jane": "waiting on reply").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..db.database import Database


@dataclass
class ConversationMemory:
    max_chars: int = 24_000
    messages: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, role: str, content: str, **extra: Any) -> None:
        message: Dict[str, Any] = {"role": role, "content": content}
        message.update(extra)
        self.messages.append(message)
        self._trim()

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": content,
        })
        self._trim()

    def _trim(self) -> None:
        # Keep the system prompt (if first) and drop oldest turns once over budget.
        total = sum(len(str(m.get("content", ""))) for m in self.messages)
        while total > self.max_chars and len(self.messages) > 2:
            # Never remove index 0 if it is the system prompt
            start = 1 if self.messages[0].get("role") == "system" else 0
            removed = self.messages.pop(start)
            total -= len(str(removed.get("content", "")))

    def snapshot(self) -> List[Dict[str, Any]]:
        return [dict(m) for m in self.messages]

    def clear(self) -> None:
        self.messages.clear()

    def last_user_message(self) -> Optional[str]:
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return None


class LongTermMemory:
    """Durable facts stored in the database ``memory`` table."""

    def __init__(self, db: Database) -> None:
        self.db = db

    def remember(self, key: str, value: str) -> None:
        self.db.remember(key, value)

    def recall(self, key: str) -> Optional[str]:
        return self.db.recall(key)

    def recall_all(self) -> Dict[str, str]:
        return self.db.recall_all()

    def forget(self, key: str) -> None:
        self.db.forget(key)

    def context_block(self, limit: int = 40) -> str:
        """Render known facts as a compact system-prompt block."""
        facts = self.recall_all()
        if not facts:
            return ""
        items = list(facts.items())[:limit]
        return "Known long-term memories:\n" + "\n".join(
            f"- {key}: {value}" for key, value in items)
