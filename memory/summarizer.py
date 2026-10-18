"""
Rolling Conversation Compactor (Tier 3 Memory):
Condenses older conversation turns into high-density narrative summaries
so that context size remains strictly bounded while conversational narrative is preserved.
"""
from typing import Optional
from .database import db_session
from persona.chatml import estimate_tokens

class ConversationSummarizer:
    @staticmethod
    def get_summary(session_id: str) -> Optional[str]:
        """Fetches the rolling summary for a given session."""
        with db_session() as conn:
            cursor = conn.execute("SELECT summary FROM session_summaries WHERE session_id = ?", (session_id,))
            row = cursor.fetchone()
            return row["summary"] if row else None

    @staticmethod
    def update_summary(session_id: str, new_summary: str, last_message_id: int):
        """Updates or sets the rolling summary for a session."""
        with db_session() as conn:
            conn.execute("""
            INSERT INTO session_summaries (session_id, summary, last_message_id, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(session_id) DO UPDATE SET
                summary = excluded.summary,
                last_message_id = excluded.last_message_id,
                updated_at = CURRENT_TIMESTAMP;
            """, (session_id, new_summary.strip(), last_message_id))

    @classmethod
    def format_for_prompt(cls, session_id: str, max_tokens: int = 200) -> str:
        """Formats the rolling summary for injection into the prompt."""
        summary = cls.get_summary(session_id)
        if not summary:
            return ""

        text = f"[SUMMARY OF EARLIER DIALOGUE IN THIS SESSION]\n{summary.strip()}"
        if estimate_tokens(text) > max_tokens:
            # Clip safely
            text = text[:int(max_tokens * 3.5)] + "..."
        return text

    @classmethod
    def compact_session_if_needed(cls, session_id: str, max_window: int = 6):
        """
        Checks if the session has older messages that should be compacted into summary.
        Can run fast heuristic compaction immediately.
        """
        with db_session() as conn:
            cursor = conn.execute(
                "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,)
            )
            messages = [dict(row) for row in cursor.fetchall()]

        if len(messages) <= max_window:
            return

        older_messages = messages[:-max_window]
        last_older_id = older_messages[-1]["id"]

        # Formulate key points from older turns
        points = []
        for msg in older_messages:
            preview = msg["content"].replace("\n", " ").strip()
            if len(preview) > 100:
                preview = preview[:97] + "..."
            points.append(f"{msg['role'].capitalize()}: {preview}")

        compacted = " | ".join(points)
        cls.update_summary(session_id, compacted, last_older_id)
