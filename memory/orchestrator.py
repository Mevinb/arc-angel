"""
Memory Orchestrator and Token Budget Controller.
Builds the context-optimized prompt injecting all memory tiers without overflowing context limits.

Exposes two entry points:
- assemble_messages(): structured chat messages (OpenAI / chat-completions APIs)
- assemble_prompt():   ChatML string for llama.cpp's raw /completion endpoint
"""
from typing import List, Dict, Tuple, Optional
from config import get_memory_budgets
from persona.prompts import get_system_prompt
from persona.chatml import format_chatml_prompt, estimate_tokens
from .dossier import UserDossier
from .episodic import EpisodicMemory
from .summarizer import ConversationSummarizer
from .database import db_session

class MemoryOrchestrator:
    @classmethod
    def assemble_messages(
        cls,
        session_id: str,
        current_user_message: str,
        mode: str = "classic",
        exclude_message_id: Optional[int] = None
    ) -> Tuple[List[Dict[str, str]], Dict]:
        """
        Assembles token-budgeted chat messages incorporating all 4 tiers of memory.
        Returns:
            (messages_list, token_breakdown_dict)
        """
        budgets = get_memory_budgets()

        # 1. Tier 1: User Dossier Facts
        dossier_text = UserDossier.format_for_prompt(max_tokens=budgets["dossier_tokens"])

        # 2. Tier 2: Episodic Memory Retrieval based on current user query
        episodic_text = EpisodicMemory.format_for_prompt(
            query=current_user_message,
            max_tokens=budgets["episodic_tokens"]
        )

        # 3. Tier 3: Rolling Conversation Summary
        summary_text = ConversationSummarizer.format_for_prompt(
            session_id=session_id,
            max_tokens=budgets["summary_tokens"]
        )

        # Combine memory tiers into a structured memory block
        memory_blocks = []
        if dossier_text:
            memory_blocks.append(dossier_text)
        if episodic_text:
            memory_blocks.append(episodic_text)
        if summary_text:
            memory_blocks.append(summary_text)

        combined_memory_str = "\n\n".join(memory_blocks)

        # 4. Construct System Prompt with Memory Context
        system_content = get_system_prompt(mode=mode, memory_context=combined_memory_str)

        # 5. Tier 4: Fetch sliding window of recent messages with token budget guard
        max_sliding_budget = budgets["sliding_budget"]
        with db_session() as conn:
            cursor = conn.execute(
                "SELECT id, role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, budgets["sliding_window_turns"])
            )
            raw_history = [dict(row) for row in cursor.fetchall()]

        # Keep messages that fit within sliding budget
        kept_messages = []
        accumulated_tokens = 0
        for m in raw_history:
            if exclude_message_id is not None and m.get("id") == exclude_message_id:
                continue
            m_tokens = estimate_tokens(m["content"])
            if accumulated_tokens + m_tokens > max_sliding_budget and kept_messages:
                break
            kept_messages.append(m)
            accumulated_tokens += m_tokens

        # Reverse to chronological order
        recent_history = list(reversed(kept_messages))

        # Assemble messages list
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
        for m in recent_history:
            messages.append({"role": m["role"], "content": m["content"]})

        # Add current user turn
        messages.append({"role": "user", "content": current_user_message})

        token_breakdown = {
            "system_and_persona_tokens": estimate_tokens(system_content),
            "dossier_tokens": estimate_tokens(dossier_text),
            "episodic_tokens": estimate_tokens(episodic_text),
            "summary_tokens": estimate_tokens(summary_text),
            "sliding_window_turns": len(recent_history),
            "total_estimated_tokens": sum(estimate_tokens(m["content"]) for m in messages)
        }

        return messages, token_breakdown

    @classmethod
    def assemble_prompt(
        cls,
        session_id: str,
        current_user_message: str,
        mode: str = "classic",
        exclude_message_id: Optional[int] = None
    ) -> Tuple[str, Dict]:
        """
        Assembles a token-budgeted ChatML prompt incorporating all 4 tiers of memory.
        Returns:
            (formatted_chatml_string, token_breakdown_dict)
        """
        messages, token_breakdown = cls.assemble_messages(
            session_id=session_id,
            current_user_message=current_user_message,
            mode=mode,
            exclude_message_id=exclude_message_id
        )
        final_prompt = format_chatml_prompt(messages, append_assistant_header=True)
        token_breakdown["total_estimated_tokens"] = estimate_tokens(final_prompt)
        return final_prompt, token_breakdown

    @classmethod
    def post_interaction_tasks(cls, session_id: str, user_msg: str, assistant_msg: str, user_message_id: Optional[int] = None):
        """
        Background maintenance after a message exchange:
        - Extracts facts into User Dossier
        - Compacts old messages into rolling summary if needed
        - Indexes into episodic memory
        """
        # 1. Auto-extract facts from user statement
        UserDossier.auto_extract_facts(user_msg)

        # 2. Check if this exchange is worth archiving into episodic memory
        if len(user_msg.split()) > 6 or len(assistant_msg.split()) > 20:
            summary_hint = f"User: {user_msg[:80]} -> Grok: {assistant_msg[:80]}"
            EpisodicMemory.add_memory(
                session_id=session_id,
                content=f"User: {user_msg}\nGrok: {assistant_msg}",
                summary=summary_hint,
                tags="dialogue",
                source_message_id=user_message_id
            )

        # 3. Compact older turns if window exceeded
        budgets = get_memory_budgets()
        ConversationSummarizer.compact_session_if_needed(session_id, max_window=budgets["sliding_window_turns"])
