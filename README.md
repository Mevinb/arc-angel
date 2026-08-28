# JARVIS — Personal AI Assistant

A personal AI assistant built in Python: LLM routing, email triage, an
internship search engine, web/computer control, and a safety-first
permission system — all orchestrated through a tool-using agent loop.

```
      ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
      ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
      ██║███████║██████╔╝██║   ██║██║███████╗
 ██   ██║██╔══██╗██╔══██╗╚██╗ ██╔╝██║╚════██║
 ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
  ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
```

## Architecture (11 phases)

| Phase | Module | What it does |
|-------|--------|--------------|
| 1 | `jarvis/core/llm.py` | **OmniRoute LLM routing** — role-based model selection (fast / reasoning / vision) with automatic fallback chains and failure cooldowns. Works with any OpenAI-compatible gateway. |
| 2 | `jarvis/core/orchestrator.py` | **Python orchestrator** — tool-using agent loop: message → LLM → tool calls → results → final answer. Bounded iterations, full audit log. |
| 3 | `jarvis/tools/computer.py` | **Computer control** — hardened shell/Python tools with static command risk analysis; optional Open Interpreter backend. |
| 4 | `jarvis/tools/browser.py` | **Browser automation** — dependency-free `web.fetch`, optional Playwright (`browser.open`) and Browser-Use (`browser.task`). |
| 5 | `jarvis/tools/email_engine.py` | **Gmail engine** — OAuth, search, LLM classification (recruiter / interview / application update / rejection), digest, drafts. Sending is RED-gated. |
| 6 | `jarvis/internships/` | **Internship engine** — live sources (RemoteOK, Arbeitnow, Hacker News Who-is-hiring), heuristic + LLM scoring against your profile, ranking, SQLite tracker. |
| 7 | `jarvis/profile/`, `jarvis/tools/personal.py` | **Personal profile & memory** — resume in YAML, rolling conversation memory, durable long-term memory in SQLite. |
| 8 | `jarvis/db/database.py` | **SQLite persistence** — jobs, recruiters, email threads, events/deadlines, memory. WAL, thread-safe. |
| 9 | `jarvis/safety/permissions.py` | **Safety & permissions** — GREEN / YELLOW / RED risk levels, blocked + dangerous command patterns, approval prompts, `auto` / `interactive` / `yolo` modes. |
| 10 | `jarvis/ui/terminal.py`, `jarvis/cli.py` | **Terminal UI** — rich REPL with Markdown replies, tool activity, approval prompts, slash commands. |
| 11 | `jarvis/automation/scheduler.py` | **Automation** — scheduled email checks, job searches and deadline reminders, always in `auto` safety mode. |
| 12 | `jarvis/skills/` | **Skill library** — search the bundled ~4.7k `SKILL.md` skills (Anthropic format) and load one's instructions on demand as context for the agent. |

## Quick start

The fastest way to get the exact system the author runs — JARVIS routed through
an OmniRoute LLM gateway, with optional Gmail/browser/computer extras and the
SKILL.md library — is the one-command installer:

```bash
git clone https://github.com/Mevinb/arc-angel.git && cd arc-angel
./install.sh                 # venv + JARVIS + OmniRoute + skills library
./start.sh                   # start the gateway and launch JARVIS chat
```

`install.sh --help` lists flags (`--no-omniroute`, `--no-skills`, `--extras`,
`--all`). It creates your `.env`/`config` only if missing, never overwrites
them, needs no account, and commits nothing.

Manual setup (equivalent):

```bash
# Python 3.11+
uv v && source .venv/bin/activate   # or: python -m venv .venv
uv pip install -e .                 # or: pip install -e .
npm install -g omniroute            # LLM gateway (or use any OpenAI-compatible one)

jarvis init       # create config.yaml, .env and your profile
jarvis doctor     # verify the LLM gateway, tools and database
jarvis            # start chatting
```

### Connecting an LLM

JARVIS talks to any OpenAI-compatible endpoint. The default targets
[OmniRoute](https://github.com/diegosouzapw/OmniRoute) at
`http://localhost:20128/v1`:

```bash
# .env
JARVIS_LLM_BASE_URL=http://localhost:20128/v1
JARVIS_LLM_API_KEY=your-key
```

Everything degrades gracefully without an LLM: job matching falls back to
heuristic scoring, email classification to keyword rules, and the agent
answers with a clear "gateway unreachable" message.

## Commands

```
jarvis                      interactive chat (default)
jarvis ask "question"       one-shot question
jarvis jobs search          search boards, score, save
jarvis jobs list [--status applied] [--min-score 50]
jarvis jobs analyze ID      deep analysis: requirements, gaps, pitch
jarvis jobs email ID        draft a recruiter email (never sends)
jarvis email digest         classify recent mail
jarvis email search QUERY   Gmail search
jarvis automate run all     run all scheduled tasks once
jarvis automate start       run the scheduler loop
jarvis skills search Q      search the SKILL.md library
jarvis skills list          list skills (--category, --limit)
jarvis skills categories    list categories with counts
jarvis skills install       clone/install the collection
jarvis doctor               health check
jarvis init                 guided setup
```

In the chat REPL: `/help` `/tools` `/new` `/doctor` `/stats` `/mode` `/quit`.

## Safety model

Every tool call is classified before it runs:

- **GREEN** — read-only (fetch a page, search jobs, read mail): runs automatically.
- **YELLOW** — creates things (drafts, shell commands, job status changes): asks first.
- **RED** — irreversible (send email, submit applications): asks first, loudly.

Blocked shell patterns (`rm -rf /`, `mkfs`, fork bombs, …) are refused
outright, regardless of approval. Modes:

| Mode | Behaviour |
|------|-----------|
| `interactive` (default) | GREEN runs; YELLOW/RED prompt the human |
| `auto` | GREEN runs; YELLOW/RED denied — used by the scheduler |
| `yolo` | everything runs (explicit opt-in, discouraged) |

## Configuration

Precedence: environment (`JARVIS_*`) → `config/config.yaml` → built-in
defaults. See `config/config.example.yaml` and `.env.example` for all keys:
LLM endpoints and models, safety mode, internship sources and thresholds,
automation intervals, Gmail credential paths.

Your profile lives in `data/profile.yaml` — education, skills, projects,
preferred roles and locations. The matcher scores every listing against it,
and the agent reads it for application answers and recruiter emails.

## Skills (SKILL.md library)

JARVIS can discover and activate the `SKILL.md` skills from the bundled
[ai-agent-skills-by-luo-kai](https://github.com/luokai0/ai-agent-skills-by-luo-kai)
collection — ~4,700 indexed, 9,786 files total. These are progressive-disclosure
*instructions* (Anthropic format): JARVIS keeps only the compact 1.1 MB
searchable index hot and loads a single skill's full text on demand, so the
agent can follow expert guidance without bloating its context.

Install the collection (clones or copies it, keeping a `content` pointer):

```bash
jarvis skills install                 # clone from GitHub into data/skills
jarvis skills install --source /path # copy from a local copy (no network)
```

Discover and use skills from the REPL or one-shot via the agent tools:

```
skills.search(query)   # find skills by name/description/category
skills.list(category)  # enumerate a category / all
skills.load(name)      # pull a skill's full instructions into context
skills.categories      # browse categories with skill counts
```

`jarvis doctor` reports the count and whether content is reachable. Point
`JARVIS_SKILLS_ROOT` (or `skills.root` in `config.yaml`) at any directory
holding `skills-index.json` + a `content/` subdir to override the collection.

## Optional engines

```bash
pip install -e ".[gmail]"      # Gmail API (google-api-python-client, google-auth)
pip install -e ".[browser]"    # Playwright + Browser-Use
pip install -e ".[computer]"   # Open Interpreter
pip install -e ".[dev]"        # pytest, ruff
```

Gmail needs an OAuth client JSON from Google Cloud Console at
`data/gmail-credentials.json`; the token is cached after the first
browser-based consent flow.

## Development

```bash
pip install -e ".[dev]"
pytest              # 104 tests, no network or LLM required
```

Tests fake the LLM client and network sources, so the whole agent loop,
permission system and engines are covered offline.

## Project layout

```
jarvis/
├── app.py              # facade wiring everything together
├── cli.py              # argparse CLI
├── config.py           # env + YAML + defaults
├── core/               # llm router, orchestrator, memory, logging
├── tools/              # tool framework + computer/browser/email/personal tools
├── internships/        # sources, matcher, engine
├── profile/            # personal profile
├── db/                 # SQLite persistence
├── safety/             # permissions & risk classification
├── ui/                 # rich terminal UI
├── automation/         # scheduler
└── skills/             # SKILL.md library (index, tools, installer)
```
