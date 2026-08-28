"""ARC — a personal AI assistant built on an OpenAI-compatible LLM gateway
(OmniRoute), with computer control, browser automation, email, and an
internship search engine.

Phases (see README):
  1. OmniRoute LLM routing      -> arc.core.llm
  2. Arc core orchestrator   -> arc.core.orchestrator
  3. Computer control           -> arc.tools.computer
  4. Browser automation         -> arc.tools.browser
  5. Email engine               -> arc.tools.email_engine
  6. Internship engine          -> arc.internships
  7. Profile & memory           -> arc.profile, arc.core.memory
  8. SQLite database            -> arc.db
  9. Safety & permissions       -> arc.safety
 10. Terminal interface         -> arc.ui
 11. Automation                 -> arc.automation
"""

__version__ = "0.1.0"
