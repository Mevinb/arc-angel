# ARC — your personal AI angel

> **LLM routing • email triage • internship hunting • browser & computer control** — all behind a safety-first, tool-using agent loop that actually does work for you.

```
         █████╗ ██████╗  ██████╗
        ██╔══██╗██╔══██╗██╔════╝
        ███████║██████╔╝██║
        ██╔══██║██╔══██╗██║
        ██║  ██║██║  ██║╚██████╗
        ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝
         — personal AI assistant that doesn't ask you to do its job
```

<p>
  <a href="https://github.com/Mevinb/arc-angel"><img alt="python" src="https://img.shields.io/badge/python-3.11%2B-blue"></a>
  <a href="https://github.com/diegosouzapw/OmniRoute"><img alt="omniroute" src="https://img.shields.io/badge/LLM-OmniRoute%20%7C%20OpenAI--compatible-7c3aed"></a>
  <img alt="skills" src="https://img.shields.io/badge/skills-5415%20indexed-0ea5e9">
  <img alt="tests" src="https://img.shields.io/badge/tests-135%20passing-22c55e">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

**ARC** is not a chatbot wrapper. It's a local agent that lives on your machine, talks to whatever LLM you point it at, and uses real tools to handle real work: triage your inbox, hunt internships across 3 boards and rank them against your profile, control your browser/computer, remember who your recruiters are, and not nuke your system while doing it.

No cloud lock-in. No account required. Bring your own model.

---

## What it actually does

- **Talks through OmniRoute** — role-based routing (`fast` / `reasoning` / `vision`) with fallback chains and cooldowns. Any OpenAI-compatible gateway works (OmniRoute, OpenRouter, Ollama, LM Studio, vLLM).
- **Reads & triages Gmail** — OAuth, LLM classification (recruiter / interview / update / rejection), digests, and draft replies. Sending is `RED`-gated — it will never send without you confirming.
- **Hunts internships** — scrapes RemoteOK, Arbeitnow, Hacker News Who-is-hiring, scores every listing heuristically *and* with the LLM against your `data/profile.yaml`, tracks deadlines.
- **Controls your machine** — shell/Python with static risk analysis, `web.fetch`, Playwright + Browser-Use. Dangerous commands (`rm -rf /`, `mkfs`, fork bombs) are refused outright.
- **Remembers** — SQLite (WAL) for jobs/recruiters/threads/events + rolling context + durable long-term memory.
- **Knows 5k+ skills** — searchable `SKILL.md` library (Anthropic format) + catalog sources (MCPs, loops, subagents, hooks, plugins, prompts, tools) loaded on demand so context stays lean.

---

## Architecture

| # | Module | What it does |
|---|--------|-------------|
| 1 | `arc/core/llm.py` | OmniRoute routing, fallback chains, failure cooldowns |
| 2 | `arc/core/orchestrator.py` | Agent loop: message → LLM → tool calls → results → answer |
| 3 | `arc/tools/computer.py` | Hardened shell/Python (optional Open Interpreter) |
| 4 | `arc/tools/browser.py` | `web.fetch`, `browser.open` (Playwright), `browser.task` (Browser-Use) |
| 5 | `arc/tools/email_engine.py` | Gmail OAuth, search, classification, digest, draft generation |
| 6 | `arc/internships/` | Multi-source aggregation + heuristic/LLM scoring |
| 7 | `arc/profile/` + `arc/tools/personal.py` | Resume in YAML + memory (rolling + durable) |
| 8 | `arc/db/database.py` | SQLite, WAL, thread-safe |
| 9 | `arc/safety/permissions.py` | GREEN / YELLOW / RED + blocked patterns + `auto`/`interactive`/`yolo` |
| 10 | `arc/ui/terminal.py` + `arc/cli.py` | Rich REPL, markdown, tool activity, slash commands |
| 11 | `arc/automation/scheduler.py` | Scheduled email checks, job sweeps, deadline pings (forced `auto` mode) |
| 12 | `arc/skills/` | 5,415-component searchable library, progressive disclosure |

---

## Quick start — 30 seconds

**One-command installer** (venv + ARC + OmniRoute + skills):

```bash
git clone https://github.com/Mevinb/arc-angel.git && cd arc-angel
./install.sh              # needs: python 3.11+, node 18+
./start.sh                # starts gateway, launches ARC chat
```

`install.sh` never overwrites your `.env`/`config` — it only creates them if missing. Flags: `--no-omniroute`, `--no-skills`, `--extras all`, `--all`, `--help`.

**Manual:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                  # or: uv pip install -e .
npm install -g omniroute          # or point ARC at any OpenAI-compatible URL

arc init                          # creates config.yaml, .env, your profile
arc doctor                        # checks gateway, tools, DB, skills
arc                               # chat
```

### Plug in any LLM

ARC just needs an OpenAI-compatible endpoint. Default is OmniRoute on `localhost:20128`:

```bash
# .env
ARC_LLM_BASE_URL=http://localhost:20128/v1
ARC_LLM_API_KEY=sk-...
ARC_MODEL_FAST=auto/fast
ARC_MODEL_REASONING=auto/reasoning
ARC_MODEL_VISION=auto/vision
```

No LLM? It still works — matching falls back to heuristics, email to keyword rules, and the agent tells you the gateway is unreachable instead of hallucinating.

---

## Commands

```bash
arc                              # interactive chat (default)
arc ask "summarize my inbox"     # one-shot
arc jobs search                  # search boards, score vs profile, save
arc jobs list --min-score 50     # filter saved jobs
arc jobs analyze 42              # requirements, gaps, pitch
arc jobs email 42                # draft recruiter email (never auto-sends)
arc email digest                 # classify recent mail
arc email search "SDE intern"    # Gmail search
arc automate run all             # run scheduled tasks once
arc automate start               # scheduler loop
arc skills search "filesystem"   # search library
arc skills search "mcp" --kind mcp
arc skills categories            # browse 96 categories
arc skills add-source            # add awesome-ai-agent-tools catalog
arc doctor                       # health check
arc init                         # guided setup
```

Inside the REPL: `/help` `/tools` `/new` `/doctor` `/stats` `/mode` `/quit`

---

## Safety — it asks before it wrecks

Every tool call is classified *before* execution:

- **GREEN** — read-only (fetch page, search jobs, read mail) → runs
- **YELLOW** — creates things (drafts, shell commands, status changes) → asks
- **RED** — irreversible (send email, submit) → asks loudly

Blocked patterns are refused even in `yolo`:

| Mode | Behavior |
|------|----------|
| `interactive` *(default)* | GREEN runs, YELLOW/RED prompt |
| `auto` | GREEN runs, YELLOW/RED denied — used by the scheduler |
| `yolo` | everything runs (you asked for it) |

---

## Configuration

Precedence: **env `ARC_*` → `config/config.yaml` → defaults**. Legacy `JARVIS_*` vars still work as fallbacks (so old `.env` files don't break).

See `.env.example` and `config/config.example.yaml` for everything: endpoints, models, safety mode, internship sources/thresholds, automation intervals, Gmail paths.

Your profile lives in `data/profile.yaml` — education, skills, projects, preferred roles/locations. ARC scores every job against it and uses it for drafts.

---

## Skills library — 5,415 components, zero bloat

ARC keeps a compact 1.1MB index hot and loads one skill's full text on demand (Anthropic `SKILL.md` progressive disclosure).

- **Core:** ~4,700 skills from [ai-agent-skills-by-luo-kai](https://github.com/luokai0/ai-agent-skills-by-luo-kai) (9,786 files)
- **Catalogs:** [awesome-ai-agent-tools](https://github.com/michielhdoteth/awesome-ai-agent-tools) and any catalog repo — adds **MCP servers, loops, subagents, hooks, plugins, prompts, tools** with source + install command

```bash
arc skills install                                    # clone into data/skills
arc skills install --source /local/path --name x      # copy local, no network
arc skills add-source                                 # add awesome-ai-agent-tools
arc skills add-source --repo <url> --name <dir>       # any catalog
```

Sources land in `<skills_root>/sources/<name>/` and are auto-discovered. Filter by kind:

```bash
arc skills search "filesystem" --kind mcp
arc skills search "deploy" --kind plugin
```

In-agent tools: `skills.search(query, kind?)` · `skills.list(category)` · `skills.load(name)` · `skills.categories`

`arc doctor` reports `5415 indexed (96 categories) — content ✓`. Override via `ARC_SKILLS_ROOT` or `skills.root` in `config.yaml`.

---

## Optional engines

```bash
pip install -e ".[gmail]"     # google-api-python-client, google-auth
pip install -e ".[browser]"   # playwright + browser-use
pip install -e ".[computer]"  # open-interpreter
pip install -e ".[dev]"       # pytest + friends
pip install -e ".[all]"       # everything
```

**Gmail:** create an OAuth Desktop client in Google Cloud Console → download JSON → place at `data/gmail-credentials.json` (or set `ARC_GMAIL_CREDENTIALS`). First `arc email digest` opens a browser for consent and caches `data/gmail-token.json`.

---

## Development

```bash
pip install -e ".[dev]"
pytest                        # 135 tests, no network/LLM needed
python -m compileall -q arc
```

Tests fake the LLM and network sources — full agent loop, permissions, and engines are covered offline.

```
arc/
├── app.py              # facade wiring everything
├── cli.py              # argparse CLI
├── config.py           # env + YAML + defaults
├── core/               # llm router, orchestrator, memory, logging
├── tools/              # tool framework + computer/browser/email/personal
├── internships/        # sources, matcher, engine
├── profile/            # YAML profile
├── db/                 # SQLite
├── safety/             # permissions & risk classification
├── ui/                 # rich terminal UI
├── automation/         # scheduler
└── skills/             # index, tools, installer
```

---

## License

MIT — do whatever you want, just don't blame us when ARC roasts your resume.

<p align="center"><i>built for people who want an assistant that actually assists.</i></p>
