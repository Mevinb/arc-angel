"""JARVIS — a personal AI assistant built on an OpenAI-compatible LLM gateway
(OmniRoute), with computer control, browser automation, email, and an
internship search engine.

Phases (see README):
  1. OmniRoute LLM routing      -> jarvis.core.llm
  2. Jarvis core orchestrator   -> jarvis.core.orchestrator
  3. Computer control           -> jarvis.tools.computer
  4. Browser automation         -> jarvis.tools.browser
  5. Email engine               -> jarvis.tools.email_engine
  6. Internship engine          -> jarvis.internships
  7. Profile & memory           -> jarvis.profile, jarvis.core.memory
  8. SQLite database            -> jarvis.db
  9. Safety & permissions       -> jarvis.safety
 10. Terminal interface         -> jarvis.ui
 11. Automation                 -> jarvis.automation
"""

__version__ = "0.1.0"
