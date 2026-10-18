"""
User Dossier (Tier 1 Memory): Permanent Entity & Biographical Fact Store.
Keeps facts about user's identity, dreams, struggles, and relationships.
"""
import re
from typing import List, Dict, Optional
from datetime import datetime
from .database import db_session
from persona.chatml import estimate_tokens

class UserDossier:
    @staticmethod
    def set_fact(key: str, value: str, category: str = "general", confidence: float = 1.0):
        """Saves or updates a fact in the user dossier."""
        with db_session() as conn:
            conn.execute("""
            INSERT INTO user_dossier (key, value, category, confidence, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                category = excluded.category,
                confidence = excluded.confidence,
                updated_at = CURRENT_TIMESTAMP;
            """, (key.strip().lower(), value.strip(), category, confidence))

    @staticmethod
    def delete_fact(key: str):
        """Deletes a fact from the user dossier."""
        with db_session() as conn:
            conn.execute("DELETE FROM user_dossier WHERE key = ?", (key.strip().lower(),))

    @staticmethod
    def get_all_facts() -> List[Dict]:
        """Returns all facts stored in the dossier."""
        with db_session() as conn:
            cursor = conn.execute("SELECT key, value, category, confidence, updated_at FROM user_dossier ORDER BY category, key")
            return [dict(row) for row in cursor.fetchall()]

    @classmethod
    def format_for_prompt(cls, max_tokens: int = 250) -> str:
        """
        Formats facts into a compact, high-density text block for prompt injection.
        """
        facts = cls.get_all_facts()
        if not facts:
            return ""

        lines = [
            "[BACKGROUND CONTEXT ON THE PERSON YOU'RE TALKING WITH]",
            "You know these things about them from past chats. Weave them in naturally when relevant, never recite them mechanically:"
        ]
        current_tokens = estimate_tokens(lines[0] + lines[1])

        for f in facts:
            line = f"- {f['key'].replace('_', ' ').capitalize()}: {f['value']}"
            line_tokens = estimate_tokens(line)
            if current_tokens + line_tokens > max_tokens:
                break
            lines.append(line)
            current_tokens += line_tokens

        if len(lines) <= 2:
            return ""
        return "\n".join(lines)

    @classmethod
    def auto_extract_facts(cls, user_msg: str):
        """
        Fast heuristic fact extraction from user input without requiring LLM roundtrip.
        Runs instantly on every message.
        """
        text = user_msg.strip()
        lower = text.lower()

        # 1. Name detection
        name_match = re.search(r"\b(?:my name is|i am|i'm|call me)\s+([A-Z][a-zA-Z]+)", text)
        if name_match:
            name = name_match.group(1).strip()
            if name.lower() not in ["here", "good", "fine", "just", "trying", "grok", "ready"]:
                cls.set_fact("name", name, category="identity")

        # 2. Age detection
        age_match = re.search(r"\b(?:i am|i'm|currently)\s+(\d{1,2})\s*(?:years old|yaers old|yo|y/o)?\b", lower)
        if age_match:
            cls.set_fact("age", f"{age_match.group(1)} years old", category="identity")

        # 3. Dream / Career / Ambition
        dream_match = re.search(r"\b(?:my dream was|my dream has always been|i wanted to become|i wanted to be|my goal is to become|dream is|wanted to be a|become a)\s+([a-zA-Z\s]+?)(?:,|;|\.|\bafter\b|\beven\b|$)", lower)
        if dream_match:
            career = dream_match.group(1).strip()
            if len(career.split()) <= 5 and career not in ["here", "honest", "real", "sure"]:
                cls.set_fact("dream_career", career, category="career")

        # 4. Struggles / Attempts
        struggle_match = re.search(r"\b(?:i tried|failed to get|failed|didn't get|didnt get|after \d+ (?:brutal )?attempts)\s+([^,.;]+)", lower)
        if struggle_match:
            cls.set_fact("struggle_or_attempts", struggle_match.group(0).strip(), category="experiences")

        # 5. Parents / Relationships
        parents_match = re.search(r"\b(?:my parents|my dad|my mom|my mother|my father|my partner|my ex)\s+([^,.;]+)", lower)
        if parents_match:
            cls.set_fact("family_dynamics", parents_match.group(0).strip(), category="relationships")

        # 6. Secrets & Code Names
        if "secret" in lower or "passphrase" in lower or "code" in lower:
            secret_match = re.search(r"\b(?:my secret project (?:name )?is|secret project is|secret project name:|secret is|my secret is|passphrase is|code name is|code is)\s+([a-zA-Z0-9_\-]+)", text, re.IGNORECASE)
            if secret_match:
                cls.set_fact("secret_project", secret_match.group(1).strip(), category="secrets")
            elif "secret" in lower:
                sec_general = re.search(r"\b(?:secret:)\s+([^,.;]+)", text, re.IGNORECASE)
                if sec_general:
                    cls.set_fact("secret", sec_general.group(1).strip(), category="secrets")

        # 7. Current skills & secret activities
        if "learning" in lower or "studying" in lower:
            skill_match = re.search(r"\b(?:been learning|learning|studying)\s+([a-zA-Z0-9_\+\#\s]+?)(?:at night|on the side|recently|lately|,|\.|$)", lower)
            if skill_match:
                skill = skill_match.group(1).strip()
                if len(skill.split()) <= 4:
                    cls.set_fact("learning_focus", skill, category="career")
