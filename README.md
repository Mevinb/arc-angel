# ARC — Personal AI Assistant

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
| 1 | `arc/core/llm.py` | **OmniRoute LLM routing** — role-based model selection (fast / reasoning / vision) with automatic fallback chains and failure cooldowns. Works with any OpenAI-compatible gateway. |
| 2 | `arc/core/orchestrator.py` | **Python orchestrator** — tool-using agent loop: message → LLM → tool calls → results → final answer. Bounded iterations, full audit log. |
| 3 | `arc/tools/computer.py` | **Computer control** — hardened shell/Python tools with static command risk analysis; optional Open Interpreter backend. |
| 4 | `arc/tools/browser.py` | **Browser automation** — dependency-free `web.fetch`, optional Playwright (`browser.open`) and Browser-Use (`browser.task`). |
| 5 | `arc/tools/email_engine.py` | **Gmail engine** — OAuth, search, LLM classification (recruiter / interview / application update / rejection), digest, drafts. Sending is RED-gated. |
| 6 | `arc/internships/` | **Internship engine** — live sources (RemoteOK, Arbeitnow, Hacker News Who-is-hiring), heuristic + LLM scoring against your profile, ranking, SQLite tracker. |
| 7 | `arc/profile/`, `arc/tools/personal.py` | **Personal profile & memory** — resume in YAML, rolling conversation memory, durable long-term memory in SQLite. |
| 8 | `arc/db/database.py` | **SQLite persistence** — jobs, recruiters, email threads, events/deadlines, memory. WAL, thread-safe. |
| 9 | `arc/safety/permissions.py` | **Safety & permissions** — GREEN / YELLOW / RED risk levels, blocked + dangerous command patterns, approval prompts, `auto` / `interactive` / `yolo` modes. |
| 10 | `arc/ui/terminal.py`, `arc/cli.py` | **Terminal UI** — rich REPL with Markdown replies, tool activity, approval prompts, slash commands. |
| 11 | `arc/automation/scheduler.py` | **Automation** — scheduled email checks, job searches and deadline reminders, always in `auto` safety mode. |
| 12 | `arc/skills/` | **Skill library** — search the bundled ~4.7k `SKILL.md` skills (Anthropic format) and load one's instructions on demand as context for the agent. |

## Quick start

The fastest way to get the exact system the author runs — ARC routed through
an OmniRoute LLM gateway, with optional Gmail/browser/computer extras and the
SKILL.md library — is the one-command installer:

```bash
git clone https://github.com/Mevinb/arc-angel.git && cd arc-angel
./install.sh                 # venv + ARC + OmniRoute + skills library
./start.sh                   # start the gateway and launch ARC chat
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

arc init       # create config.yaml, .env and your profile
arc doctor     # verify the LLM gateway, tools and database
arc            # start chatting
```

### Connecting an LLM

ARC talks to any OpenAI-compatible endpoint. The default targets
[OmniRoute](https://github.com/diegosouzapw/OmniRoute) at
`http://localhost:20128/v1`:

```bash
# .env
ARC_LLM_BASE_URL=http://localhost:20128/v1
ARC_LLM_API_KEY=your-key
```

Everything degrades gracefully without an LLM: job matching falls back to
heuristic scoring, email classification to keyword rules, and the agent
answers with a clear "gateway unreachable" message.

## Commands

```
arc                      interactive chat (default)
arc ask "question"       one-shot question
arc jobs search          search boards, score, save
arc jobs list [--status applied] [--min-score 50]
arc jobs analyze ID      deep analysis: requirements, gaps, pitch
arc jobs email ID        draft a recruiter email (never sends)
arc email digest         classify recent mail
arc email search QUERY   Gmail search
arc automate run all     run all scheduled tasks once
arc automate start       run the scheduler loop
arc skills search Q      search the SKILL.md library
arc skills list          list skills (--category, --limit)
arc skills categories    list categories with counts
arc skills install       clone/install the collection
arc doctor               health check
arc init                 guided setup
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

Precedence: environment (`ARC_*`) → `config/config.yaml` → built-in
defaults. See `config/config.example.yaml` and `.env.example` for all keys:
LLM endpoints and models, safety mode, internship sources and thresholds,
automation intervals, Gmail credential paths.

Your profile lives in `data/profile.yaml` — education, skills, projects,
preferred roles and locations. The matcher scores every listing against it,
and the agent reads it for application answers and recruiter emails.

## Skills (SKILL.md library)

ARC can discover and activate the `SKILL.md` skills from the bundled
[ai-agent-skills-by-luo-kai](https://github.com/luokai0/ai-agent-skills-by-luo-kai)
collection — ~4,700 indexed, 9,786 files total. These are progressive-disclosure
*instructions* (Anthropic format): ARC keeps only the compact 1.1 MB
searchable index hot and loads a single skill's full text on demand, so the
agent can follow expert guidance without bloating its context.

Install the collection (clones or copies it, keeping a `content` pointer):

```bash
arc skills install                 # clone from GitHub into data/skills
arc skills install --source /path # copy from a local copy (no network)
```

Beyond the Luo-Kai `SKILL.md` collection, ARC also indexes catalog-style
sources such as
[awesome-ai-agent-tools](https://github.com/michielhdoteth/awesome-ai-agent-tools)
(a curated library of AI-agent components). One catalog source adds hundreds of
searchable entries — **MCP servers, agent loops, subagents, hooks, plugins,
prompts and CLI tools** — each with its source, description and exact install
command, discovered through the same `skills.search`/`skills.load` tools:

```bash
arc skills add-source                                    # clone awesome-ai-agent-tools
arc skills add-source --repo <url> --name <dir>          # any catalog repo
arc skills add-source --source /path/to/catalog --name x # copy from local (no network)
```

Sources land in `<skills_root>/sources/<name>/` and are auto-discovered on the
next run. `skills.search(..., kind="mcp")` filters by component type, and
`skills.load("Filesystem MCP")` returns the component's description, source and
install command.

Discover and use skills from the REPL or one-shot via the agent tools:

```
skills.search(query)   # find skills/components by name/description/category
skills.search(query, kind="mcp")  # filter by kind (skill|mcp|loop|subagent|hook|plugin|prompt|tool)
skills.list(category)  # enumerate a category / all
skills.load(name)      # pull a skill's instructions / a component's install info
skills.categories      # browse categories with counts
```

`arc doctor` reports the count and whether content is reachable. Point
`ARC_SKILLS_ROOT` (or `skills.root` in `config.yaml`) at any directory
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
arc/
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
