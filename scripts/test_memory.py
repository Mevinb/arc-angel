"""
Automated Verification Suite for Grok Chatbot Multi-Tier Memory Engine.
Verifies memory retention across low context bounds, fact extraction, and FTS5 episodic retrieval.
"""
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from memory import (
    init_db,
    db_session,
    UserDossier,
    EpisodicMemory,
    ConversationSummarizer,
    MemoryOrchestrator
)
from persona.chatml import estimate_tokens

class TestGrokMemoryArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def test_01_user_dossier_fact_extraction_and_persistence(self):
        """Test auto-extraction of personal identity, goals, and struggles."""
        # Simulated user message revealing critical life details
        sample_input = "my name is Asin and im currently 21 yaers old , my dream was to become a doc , but i didnt get seats in medicine even tho i tried 3 timmes"
        UserDossier.auto_extract_facts(sample_input)

        facts = {f["key"]: f["value"] for f in UserDossier.get_all_facts()}
        
        self.assertIn("name", facts)
        self.assertEqual(facts["name"], "Asin")
        self.assertIn("age", facts)
        self.assertIn("21", facts["age"])
        self.assertIn("dream_career", facts)
        self.assertIn("doc", facts["dream_career"].lower())

        # Test prompt formatting
        dossier_prompt = UserDossier.format_for_prompt(max_tokens=250)
        self.assertIn("Asin", dossier_prompt)
        self.assertIn("PERMANENT USER DOSSIER", dossier_prompt)
        print("✓ Tier 1 (User Dossier) verified successfully.")

    def test_02_episodic_memory_fts5_retrieval(self):
        """Test FTS5 full-text and semantic retrieval across episodic chunks."""
        test_session = "test-session-xyz"
        EpisodicMemory.add_memory(
            session_id=test_session,
            content="User discussed the stress of medical entrance exams and feeling burnt out.",
            summary="User burnt out from entrance exams.",
            tags="career,medical"
        )
        EpisodicMemory.add_memory(
            session_id=test_session,
            content="User mentioned wanting to explore programming and machine learning instead.",
            summary="User looking into programming and ML.",
            tags="career,tech"
        )

        # Query relevant to medicine
        results_med = EpisodicMemory.search_memories("medical entrance exam", limit=2)
        self.assertGreaterEqual(len(results_med), 1)
        self.assertIn("entrance exams", results_med[0]["content"])

        # Formatted prompt injection test
        prompt_block = EpisodicMemory.format_for_prompt("exams and medicine", max_tokens=350)
        self.assertIn("RELEVANT PAST MEMORIES", prompt_block)
        print("✓ Tier 2 (Episodic Memory FTS5) verified successfully.")

    def test_03_rolling_summary_compactor(self):
        """Test session compaction for long dialogues."""
        test_sess = "compaction-sess-1"
        ConversationSummarizer.update_summary(
            session_id=test_sess,
            new_summary="User discussed career dilemmas and received candid feedback from Grok.",
            last_message_id=10
        )

        retrieved_summary = ConversationSummarizer.get_summary(test_sess)
        self.assertIsNotNone(retrieved_summary)
        self.assertIn("career dilemmas", retrieved_summary)

        formatted = ConversationSummarizer.format_for_prompt(test_sess, max_tokens=200)
        self.assertIn("SUMMARY OF EARLIER DIALOGUE", formatted)
        print("✓ Tier 3 (Rolling Summary Compactor) verified successfully.")

    def test_04_memory_orchestrator_token_budget_invariant(self):
        """
        Verify that assembled prompts strictly obey the token budget 
        even with long conversation histories and multiple memory tiers active.
        """
        test_sess = "orchestrator-test-session"
        with db_session() as conn:
            # Seed multiple messages
            for i in range(12):
                conn.execute(
                    "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                    (test_sess, "user" if i % 2 == 0 else "assistant", f"Message turn {i} with some long text to fill context.")
                )

        prompt, budget = MemoryOrchestrator.assemble_prompt(
            session_id=test_sess,
            current_user_message="Tell me what you remember about my situation.",
            mode="unhinged"
        )

        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("Asin", prompt)  # Dossier retained!
        self.assertIn("UNHINGED mode", prompt)
        self.assertIn("<|im_start|>assistant", prompt)

        # Invariant: Prompt must stay comfortably within bounded budget (< 3000 tokens)
        self.assertLess(budget["total_estimated_tokens"], 3000)
        print(f"✓ Tier 4 & Orchestrator verified. Total Prompt Tokens: {budget['total_estimated_tokens']} (Budget: <3000)")

if __name__ == "__main__":
    unittest.main()
