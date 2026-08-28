# ARC — Agent & Developer Reference Guide

This document outlines the architecture, testing instructions, and operational workflows for AI agents and developers working on the ARC repository.

---

## 1. Project Overview

**ARC** is a modular, personal AI assistant built in Python. It coordinates LLM routing, computer control, browser automation, Gmail integration, and an automated internship search engine with multi-tier permission safety.

### Architecture Map (11 Core Phases)

| Phase | Subsystem | Key Files | Description |
|---|---|---|---|
| **1** | LLM Gateway & Routing | [`arc/core/llm.py`(arc/core/llm.py) | Role-based routing (`fast`, `reasoning`, `vision`) with fallback chains and failure cooldowns. |
| **2** | Python Orchestrator | [`arc/core/orchestrator.py`(arc/core/orchestrator.py) | Agent tool-calling loop, context management, iteration boundaries, audit logging. |
| **3** | Computer Control | [`arc/tools/computer.py`(arc/tools/computer.py) | Shell/Python execution with risk categorization; fallback to subprocess if `open-interpreter` is missing. |
| **4** | Browser Automation | [`arc/tools/browser.py`(arc/tools/browser.py) | HTTP fetching (`web.fetch`), Playwright page control (`browser.open`), and `browser-use` automation. |
| **5** | Email Engine | [`arc/tools/email_engine.py`(arc/tools/email_engine.py) | Gmail API integration for search, thread summarization, inbox digests, and recruiter draft generation. |
| **6** | Internship Search | [`arc/internships/`(arc/internships) | Aggregation from RemoteOK, Arbeitnow, and Hacker News. Heuristic + LLM profile scoring and tracking. |
| **7** | Profile & Memory | [`arc/profile/`(arc/profile), [`arc/core/memory.py`(arc/core/memory.py) | User resume/preferences YAML, rolling conversation context, and durable long-term memory. |
| **8** | SQLite Database | [`arc/db/database.py`(arc/db/database.py) | SQLite store with WAL mode for jobs, applications, recruiter emails, and memories. |
| **9** | Safety & Permissions | [`arc/safety/permissions.py`(arc/safety/permissions.py) | Multi-tier action risk analysis (**GREEN**, **YELLOW**, **RED**); blocked command patterns. |
| **10** | Terminal UI & CLI | [`arc/ui/terminal.py`(arc/ui/terminal.py), [`arc/cli.py`(arc/cli.py) | Rich REPL chat interface, slash commands (`/help`, `/doctor`, `/tools`), CLI subcommands. |
| **11** | Background Scheduler | [`arc/automation/scheduler.py`(arc/automation/scheduler.py) | Periodic scheduled tasks (email digest, job sweeps, deadline reminders) forced into `auto` safety mode. |

---

## 2. Environment & Testing

### Running Tests
The test suite is fully offline-capable and does not require active API keys or LLM connections:

```bash
# From workspace root:
source .venv/bin/activate
PYTHONPATH=. pytest
```
*Current test status: 114 passed (100% pass rate).*

### Compilation Check
```bash
python -m compileall -q arc
```

### Health Diagnostics
```bash
arc doctor
```
*Note: Returns exit code 0 when an LLM gateway is reachable; returns 1 if LLM gateway is offline (fallback heuristics are used automatically).*

---

## 3. Safety Guardrails

All tools executed by the orchestrator are gated by [`PermissionGuard`(arc/safety/permissions.py):

- **`GREEN` (Read-only)**: Auto-runs (`web.fetch`, `jobs.search`, `email.search`, `memory.recall`).
- **`YELLOW` (State Modification)**: Prompts user in `interactive` mode, denied in `auto` mode (`computer.run_shell`, `email.draft`, `jobs.update_status`).
- **`RED` (Irreversible/External)**: Strict double-confirmation in `interactive` mode (`email.send`, `jobs.apply_external`).
- **Blocked Patterns**: Dangerous commands (`rm -rf /`, `mkfs`, fork bombs) are refused unconditionally.

---

## 4. Operational Tips for Agents

- **Configuration**: Managed in `config/config.yaml` or `.env`.
- **Database**: Default SQLite file stored at `data/arc.db`.
- **User Profile**: Default YAML profile stored at `data/profile.yaml`.
- **Graceful Fallbacks**: If LLM endpoints are unreachable, internship matchers fall back to weighted keyword heuristics, and email classifiers fall back to rule-based heuristics.
