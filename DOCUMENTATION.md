# GROK — System Documentation

> Self-hosted AI chat system with a dual engine (local GGUF models **or** the OpenAI
> ChatGPT API), a 4-tier persistent memory, vision, and an autonomous **Desktop Pilot**
> that can drive a GNOME Wayland desktop with the mouse and keyboard.

---

## 1. What This System Is

GROK is a local web application (no build step, no cloud component except the optional
ChatGPT API) made of four parts:

1. **Web backend** — FastAPI (`server.py`) exposing a REST + SSE API and serving a
   single-page vanilla-JS frontend (`static/`).
2. **LLM engine layer** — a provider router that talks either to a local
   **llama.cpp `llama-server`** process (GGUF models, with automatic GPU offload) or to
   the **OpenAI Chat Completions API** (ChatGPT models, including reasoning models).
3. **Memory layer** — a SQLite database with a 4-tier memory system
   (dossier → episodic recall → session summary → sliding window) that is assembled
   into every prompt.
4. **Perception & actuation layer** — vision for image understanding plus the
   **Desktop Pilot** (`engine/desktop_agent.py` + `engine/wayland_controller.py`) which
   sees the screen (Mutter ScreenCast + PipeWire + OCR) and controls it
   (Mutter RemoteDesktop: absolute pointer motion, buttons, keysym typing).

Everything persistent lives in `data/` on your machine. Nothing is sent anywhere
unless you activate ChatGPT cloud mode, in which case only the chat prompts and
attached images go to the OpenAI API.

### High-level architecture

```
Browser (static HTML/CSS/JS, SSE streaming)
   │  HTTP  ─ 127.0.0.1:8000
   ▼
FastAPI server (server.py)
   ├── ProviderRouter ──► LlamaClient   ──► llama-server  (local GGUF, 127.0.0.1:8080)
   │                  └─► OpenAIClient  ──► api.openai.com/v1 (or custom base URL)
   ├── MemoryOrchestrator ──► SQLite data/grok_memory.db
   │      (dossier · episodic FTS5 · summaries · messages/sessions)
   ├── /api/v2 platform ──► services/*
   │      (agents · run manager · approvals · knowledge · connectors · audit)
   │      ├── Command Center UI  /platform
   │      └── Pocket companion   /mobile
   ├── VisionEngine (BLIP caption + Tesseract OCR — fallback for non-vision models)
   └── DesktopAgent ──► WaylandController ──► GNOME Mutter DBus
                              │                 ├── org.gnome.Mutter.RemoteDesktop (input)
                              │                 ├── org.gnome.Mutter.ScreenCast (capture)
                              │                 └── org.gnome.Mutter.DisplayConfig (geometry)
                              ├── gst-launch-1.0 + PipeWire  (single-frame screenshots)
                              └── tesseract TSV              (OCR elements with coordinates)
```

---

## 2. Feature Overview

**Engines & models**
- Dual provider: **Local** (llama.cpp GGUF) or **ChatGPT** (OpenAI API) — switchable live from the UI; the choice is persisted in `data/active_model.json`.
- **13 OpenAI models** preconfigured (GPT-5.6 Terra/Luna, GPT-5.4 mini/nano, GPT-5.1 Codex Mini, GPT-5 mini/nano, GPT-4.1 mini/nano, GPT-4o mini, o4-mini, o1-mini, Codex Mini) plus live merge of `/v1/models` from your account.
- **Local GGUF models** with per-model vision (e.g. Qwen 3.5 9B Vision via an `mmproj` projector) and automatic **GPU offload recommendation** based on free VRAM (`nvidia-smi`).
- Reasoning-model aware: GPT-5.x / o-series / Codex automatically receive `max_completion_tokens` + `reasoning_effort` instead of sampling parameters; rejected parameters are sanitized and retried automatically.

**Chat & UI**
- Streaming responses (SSE) with live telemetry: tokens/s, token count, and a context budget meter (dossier/episodic breakdown).
- **4 personas / modes** (see §3) selectable via pills in the top bar.
- Sessions sidebar: create, switch, delete; auto-titled from the first message; transcript **export to Markdown**.
- Markdown rendering with syntax highlighting, copy-message button, suggestion chips on the welcome screen.

**Memory**
- **4-tier memory** (see §4): permanent user dossier, episodic recall, rolling session summary, sliding dialogue window — all SQLite, all local.
- Heuristic fact extraction from your messages (no LLM needed) and a **Memory Inspector** drawer to view/add/edit/delete facts and search episodic memories.
- Provider-aware budgets: the same memory that fits a 4k local context expands automatically in cloud mode (12k prompt budget).

**Vision**
- Attach images via the 📎 button, **Ctrl+V paste**, or drag-and-drop; images are stored under `data/uploads/`.
- Models with **native vision** (ChatGPT vision models, local Qwen mmproj) receive the image directly; text-only models get an automatic **BLIP caption + OCR** context block. Optional refusal-free local pipeline (`torch` + `sentence-transformers` + Pillow + Tesseract).

**Desktop Pilot** (see §6 for the deep dive)
- Autonomous multi-step desktop control on **GNOME Wayland** through Mutter's RemoteDesktop/ScreenCast APIs — no xdotool hacks, no permission dialogs.
- Works with **either engine**: the pilot's "brain" can be a local model or a ChatGPT model.
- Live viewer in the UI: screenshot of the pilot's view, current thought, action, step counter, and a red **Stop Pilot** emergency button.

**Autonomous platform (v2)**
- **Command Center** at `/platform`: task/run board, approval inbox, knowledge, research, connectors, automations, files/code, audit, backup and safety controls.
- **Pocket companion** at `/mobile`: one-time-code pairing, daily brief, quick task/note commands, and share-to-Atlas.
- **Agents and resumable runs** with checkpoints, budgets, pause/resume/cancel/retry, restart recovery, and a live event stream.
- **Policy-gated tools**: risky actions pause the run for approval; the kill switch cancels supervised runs and stops automation scheduling.
- **Knowledge vault** with local hybrid retrieval and citations; **research projects** with notes, claims and reports.
- **Encrypted connector credentials**, scheduled tasks, HMAC webhooks, file checkpoints, code indexing, SQLite backups with optional passphrase encryption, privacy export and a hash-chain audit log.

**Operations & safety**
- One-command start (`./start.sh`) with automatic Python/venv detection, engine bootstrap, and provider-aware startup.
- API key handling: validated against the API before saving, stored with `chmod 600`, masked in the UI, can live in an env var instead.
- Offline test suite with a mock OpenAI server; no key or internet required to verify the whole cloud flow.
- Everything runs on `127.0.0.1` by default; no telemetry leaves the machine.

---

## 3. Chat Modes / Personas

| Mode key | UI name | Character |
|---|---|---|
| `classic` | **Grok Classic** | Natural, witty, candid, razor-sharp, authentically human, zero corporate fluff. |
| `unhinged` | **Unhinged Grok** | Completely unfiltered, raw banter, maximum swagger, zero guardrails, pure honesty. |
| `deep_truth` | **Deep Truth / Thinker** | First-principles analytical realism; deconstructs hype, bias and propaganda. |
| `desktop` | **Desktop Agent** | Turns the chat into the Desktop Pilot: interprets your message as a desktop task (see §6). |

Modes are defined in `persona/prompts.py` (`GROK_MODES`) with their own system prompt
and sampling parameters. Cloud reasoning effort is mapped per mode:
`deep_truth → high`, `desktop → low`, everything else `medium`.

The Desktop Pilot can also be triggered from any other mode by writing `/desktop <goal>`
or a message that looks like a desktop intent (e.g. *"open brave and search cat videos"*).

---

## 4. Memory System (4 tiers)

All memory lives in **one SQLite file: `data/grok_memory.db`**.

| Tier | What it is | Storage table | Local budget | Cloud budget |
|---|---|---|---|---|
| 1 — Dossier | Permanent facts about you (identity, career, struggles, relationships, secrets) | `user_dossier` (key/value/category/confidence) | 250 tok | 700 tok |
| 2 — Episodic | Recalls of past exchanges, retrieved by relevance (SQLite **FTS5/BM25**) | `episodic_memories` + `episodic_fts` (content, summary, tags) | 350 tok | 1000 tok |
| 3 — Summary | Rolling extractive summary of the older part of the current session | `session_summaries` | 200 tok | 600 tok |
| 4 — Sliding window | The most recent messages verbatim | `messages` | last 6 msgs, 1600 tok | last 10 msgs, 6000 tok |

**How a prompt is assembled** (`memory/orchestrator.py → assemble_messages`):

1. Dossier block (`[BACKGROUND CONTEXT ON THE PERSON YOU'RE TALKING WITH]`)
2. Episodic block (`[PAST CONVERSATIONS YOU RECALL WITH THEM]`, query = your message)
3. Summary block (`[SUMMARY OF EARLIER DIALOGUE IN THIS SESSION]`)
4. All non-empty blocks are appended to the mode's system prompt → first `system` message
5. Sliding window of recent history (newest-first token accumulation until the budget is hit)
6. The new user message

Local mode additionally renders the whole thing as **ChatML** for llama.cpp
(`<|im_start|>role … <|im_end|>`); cloud mode sends structured `messages` to the API.

**After every reply** (`post_interaction_tasks`, runs synchronously so the next turn
sees it immediately):
- **Dossier extraction** — regex heuristics pull identity/career/secret/relationship
  facts out of your message (no LLM call).
- **Episodic save** — meaningful exchanges (`>6` user words or `>20` assistant words)
  are stored with a short summary for future recall.
- **Summarization** — once the session exceeds the sliding window, older messages are
  compacted into the rolling summary (extractive previews, capped at 100 chars each).

Token estimation is deliberately conservative (`len(text) / 3.6`), so budgets are
approximations with safety margin. The invariant tested by `scripts/test_memory.py` is
that a local prompt stays under 3000 estimated tokens.

---

## 5. Vision Pipeline

```
attach (📎 / Ctrl+V / drag-drop) → base64 data URL → POST /api/chat/stream
   → saved to data/uploads/<id>.<png|jpg|webp>
   → if model has NATIVE vision:  image passed straight to the model
     else:                       VisionEngine.analyze() builds a context block:
                                   BLIP caption (local, CPU) + Tesseract OCR text
```

- **Native vision**: ChatGPT vision models (GPT-5.x/GPT-4.1/4o/o4-mini) and local
  models with an `mmproj` projector (Qwen 3.5 9B Vision) — no Python ML needed.
- **BLIP + OCR fallback** (for text-only local models): captioning via
  `transformers`/`torch` on CPU (loaded lazily, cached), OCR via the `tesseract`
  binary. If Pillow/torch are missing, the fallback reports a clear error and the
  chat continues.
- Images persist as files under `data/uploads/` and are linked in the message history
  (`image_url`), so they reappear when you reload a session.

---

## 6. Desktop Pilot — How It Works

The Desktop Pilot is an **observe → think → act** loop that uses the LLM as the
decision maker and GNOME's own Wayland APIs as the hands and eyes. It works entirely
without `xdotool`/`ydotool`, without X11, and without any confirmation dialogs.

### 6.1 Requirements

| Requirement | Why | Check |
|---|---|---|
| GNOME on **Wayland** (Mutter; built for GNOME 46) | RemoteDesktop/ScreenCast DBus APIs | `echo $XDG_SESSION_TYPE` → `wayland` |
| A running **session bus** (`DBUS_SESSION_BUS_ADDRESS`) | all controller DBus calls | env var set in a desktop terminal |
| **PyGObject** (`python3-gi`) in the app's interpreter | DBus bindings | app logs `Desktop Pilot ready (PyGObject available).` |
| `gst-launch-1.0` (GStreamer tools) | pull one PipeWire frame to PNG | `which gst-launch-1.0` |
| `tesseract` | OCR of the screenshot into clickable text elements | `which tesseract` |
| Brave or Chrome (optional) | target for `navigate_browser` / `launch_app` | alias map, see below |

Because most uv-managed Pythons lack `gi`, the recommended setup is a project venv
built **from the system Python with `--system-site-packages`** (this is what
`./start.sh` prefers and what the README's one-time setup creates).

### 6.2 Architecture

```
DesktopAgent (engine/desktop_agent.py)
  │  "brain": llm_router → local llama-server  OR  OpenAI API
  │
  ├── perception:  WaylandController.capture_screen()   → PNG in /tmp/grok_shots/
  │                WaylandController.get_screen_ocr_elements() → [{text,x,y,w,h,cx,cy,conf}]
  │
  └── actuation:   WaylandController.move_cursor/click/type_text/press_key/launch_app
                     │
                     ▼
        org.gnome.Mutter.RemoteDesktop  (a session object)
           • NotifyPointerMotionAbsolute  (pixel-exact absolute motion)
           • NotifyPointerButton          (evdev codes: left 0x110, right 0x111, …)
           • NotifyKeyboardKeysym         (layout-independent X11 keysyms)
        org.gnome.Mutter.ScreenCast
           • RecordMonitor / RecordArea   → PipeWire stream node
        org.gnome.Mutter.DisplayConfig
           • GetCurrentState              → monitor geometry & scale
```

### 6.3 The agent loop

`DesktopAgent.execute_task(goal, max_steps=15)` (an async generator; the UI consumes it
as SSE events):

1. **Initialize** — emit a `step 0` event, make sure the engine is ready
   (local: auto-start `llama-server` and wait up to 15 s for `/health`;
   cloud: just require a configured API key).
2. **Perceive (once)** — compute desktop bounds and take a first screenshot.
3. **Loop (max 15 steps)**:
   1. Take a fresh screenshot (`/tmp/grok_shots/shot_<ms>.png`).
   2. Run OCR over it → text elements with coordinates.
   3. Build the decision prompt (system prompt = agent rules; user message =):
      ```
      Goal: <your goal>

      Current Screen State:
      - "Address bar" at coordinates (500, 60)
      - "Search" at coordinates (600, 300)
      …                                (up to 35 merged text elements)

      Recent Steps Taken:
      Step 3: click_text("Search") -> Clicked "Search" at (600,300)
      …                                (last 4 steps)

      What is your next action?
      ```
   4. Ask the LLM (temperature 0.10, top_p 0.85, max_tokens 140,
      `reasoning_effort="low"`) which streams back strictly:
      ```
      Thought: <1–2 sentences of reasoning>
      Action: <exactly one tool call>
      ```
   5. Parse `Thought:` / `Action:` with regex and **execute** the action through
      `WaylandController` (with small human-like pauses: 0.3–2 s).
   6. Append `Step N: action -> result` to history, emit a step event, repeat — unless
      the action was `finish("…")` or the user pressed **⏹ Stop Pilot**.

Every event carries the step number, thought, action, human-readable detail and a
**screenshot URL** so the UI can show exactly what the pilot sees (`#desktop-live-viewer`).

### 6.4 Action vocabulary

The model must emit exactly one of these per step (single or double quotes accepted;
multiple actions can be chained with ` then ` / ` && `):

| Action | Effect |
|---|---|
| `launch_app("brave")` | Launch an application. Aliases: brave/brave-browser, chrome/google-chrome (`--profile-directory='Default'`), files/nautilus, terminal/gnome-terminal, calculator/gnome-calculator, code/vscode, settings/gnome-control-center; anything else is run verbatim as a shell command. |
| `navigate_browser("query or URL")` | Focus the browser address bar (`Ctrl+L`), type the text, press Enter, wait 2 s. |
| `click(x, y)` | Absolute click at desktop coordinates. |
| `click_text("Search")` | Finds the best-matching OCR element (exact match scores highest; substring and word-overlap heuristics follow), then clicks its center. |
| `type_text("string")` | Types text with layout-independent keysyms (character by character). |
| `press_key("ctrl+l")` | Press a key or combo (`enter`, `ctrl+t`, `alt+tab`, `f5`, …). |
| `wait(seconds)` | Pause (capped at 5 s). |
| `finish("summary")` | Declare the task complete and stop the loop. |

The system prompt enforces two safety habits: **launch the requested app first** before
clicking anything, and never click random taskbar/desktop icons.

### 6.5 Perception details

- **Screenshots** — `capture_screen()` asks DisplayConfig for the primary connector,
  opens a dedicated ScreenCast session, calls `RecordMonitor`, listens for the
  `PipeWireStreamAdded` signal (3 s timeout), then pulls exactly one frame with:
  ```
  gst-launch-1.0 -q pipewiresrc path=<node_id> num-buffers=1 ! videoconvert ! pngenc ! filesink location=<out.png>
  ```
  (5 s subprocess timeout; the ScreenCast session is stopped afterwards). No permission
  dialogs appear — Mutter grants this to the session's own process tree.
- **OCR elements** — `tesseract <img> stdout tsv`; words with confidence > 25 are kept,
  sorted by line (`y//16`), same-line fragments (Δy < 12 px, gap < 35 px) are merged,
  and only the first 35 elements are sent to the model (keeps prompts small and fast).
- **Multi-monitor** — `get_desktop_bounds()` unions all logical monitors from
  DisplayConfig (honoring scale/rotation) into `(x, y, width, height)`; all pointer
  coordinates are clamped into this space before being sent as absolute motions.

### 6.6 Input injection details

- Pointer motion uses `NotifyPointerMotionAbsolute` (stream + relative-to-bounds x/y) —
  pixel-exact, independent of pointer acceleration.
- Buttons use evdev codes (`left 0x110`, `right 0x111`, `middle 0x112`).
- Typing converts each character to an X11 keysym (`char_to_keysym`), so it works
  regardless of keyboard layout; `press_key` resolves names like `ctrl`, `shift`,
  `enter`, `f5`, `pageup` to their keysyms and presses modifiers in order, releasing in
  reverse.
- `launch_app` uses `subprocess.Popen(..., start_new_session=True)` so apps detach
  from the server.

### 6.7 Session lifecycle & safety

- One long-lived Mutter RemoteDesktop session per app process (singleton
  `WaylandController`), with an **idle auto-stop**: a 45 s timer calls `stop()`
  (RemoteDesktop `Stop`) after the last input. `_ensure_session()` transparently
  recreates it on the next action.
- **Emergency stop** — the red **⏹ Stop Pilot** button in the top bar (or
  `POST /api/desktop/stop`) sets an abort flag that is checked before every step.
  Note: an in-flight LLM call is not cancelled; the action computed from it may still
  execute before the loop exits.
- The pilot stops automatically after `finish(...)` or after 15 steps (`max_steps`).

### 6.8 Events & API

`POST /api/desktop/run` `{goal}` streams raw step events; the chat path
(`POST /api/chat/stream` with mode `desktop` or a desktop intent) wraps them as
`{"type": "desktop_step", "step", "thought", "action", "detail", "screenshot_url", "status"}`.
`GET /api/desktop/shot?path=…` serves the PNG (only paths under `/tmp/grok_shots`),
`GET /api/desktop/status` reports `is_running`, `current_task` and screen `bounds`.

### 6.9 Pilot troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Desktop Pilot requires PyGObject (python3-gi)…` | interpreter without `gi` | run with the project venv (`uv venv --system-site-packages .venv && uv pip install --python .venv/bin/python fastapi uvicorn httpx cryptography pydantic pillow`), then `./start.sh` |
| `Failed to initialize desktop capture: …` | no session bus / not a GNOME Wayland session | launch the app from a terminal **inside** your desktop session |
| Screenshot times out / `gst-launch` error | missing GStreamer tools or PipeWire | `sudo apt install gstreamer1.0-tools` |
| No text elements → pilot clicks blindly | `tesseract` missing | `sudo apt install tesseract-ocr` |
| Chrome doesn't open | profile/binary mismatch | check the alias map, or use `launch_app("brave")`; `google-chrome` must be on PATH |

---

## 7. Providers & Models

### 7.1 OpenAI (ChatGPT) catalog

Configured in `config.py → OPENAI_MODELS`; unknown ids from your account's `/v1/models`
are merged automatically in the UI.

| Model ID | Vision | Reasoning | Reasoning effort | Tag |
|---|---|---|---|---|
| `gpt-5.6-terra` *(default)* | ✅ | ✅ | ✅ | ChatGPT |
| `gpt-5.6-luna` | ✅ | ✅ | ✅ | ChatGPT |
| `gpt-5.4-mini-2026-03-17` | ✅ | ✅ | ✅ | ChatGPT |
| `gpt-5.4-nano-2026-03-17` | ✅ | ✅ | ✅ | ChatGPT |
| `gpt-5.1-codex-mini` | — | ✅ | ✅ | Codex |
| `gpt-5-mini-2025-08-07` | ✅ | ✅ | ✅ | ChatGPT |
| `gpt-5-nano-2025-08-07` | ✅ | ✅ | ✅ | ChatGPT |
| `gpt-4.1-mini-2025-04-14` | ✅ | — | — | ChatGPT |
| `gpt-4.1-nano-2025-04-14` | ✅ | — | — | ChatGPT |
| `gpt-4o-mini-2024-07-18` | ✅ | — | — | ChatGPT |
| `o4-mini-2025-04-16` | ✅ | ✅ | ✅ | Reasoning |
| `o1-mini-2024-09-12` | — | ✅ | — | Reasoning |
| `codex-mini-latest` | — | ✅ | ✅ | Codex |

Behavioral notes:
- Reasoning models receive `max_completion_tokens` (floor 1024 at low effort, 2048
  otherwise) and `reasoning_effort`; classic sampling parameters are omitted.
- If the API rejects a parameter, the client **sanitizes and retries once**
  (e.g. `max_completion_tokens → max_tokens`, dropping sampling params) — this also
  makes third-party OpenAI-compatible endpoints work.
- Cloud replies are capped at 2500 tokens (1500 local).

### 7.2 Credentials

- Resolution order: `OPENAI_API_KEY` env → `data/openai_key.json` (`chmod 600`).
- The key is validated against `/v1/models` **before** saving; the UI shows a masked
  hint (`sk-abc…wxyz`) and the source (env/file).
- `OPENAI_BASE_URL` (env or saved) lets you use compatible endpoints; the ChatGPT
  models then come from that endpoint's catalogue.
- Removing the key reverts the app to the local provider and locks cloud models in
  the dropdown (🔒).

### 7.3 Local models

`config.py → MODELS` defines local GGUF models:

| Key | Name | Layers | Default `-ngl` | Vision |
|---|---|---|---|---|
| `qwen` | Qwen 3.5 9B Vision (`mmproj` projector) | 32 | 24 | ✅ native |
| `dolphin` | Dolphin 3.0 Llama 3.1 8B (uncensored) | 33 | 33 | — |

`detect_recommended_gpu_layers()` reads free VRAM via `nvidia-smi` and recommends an
offload level (≥5 GB → full `ngl`, ≥3 GB → 70 %, ≥1.5 GB → 40 %, else CPU). The chosen
level is passed to `llama-server` on start. Engine lifecycle (start/stop/switch model)
is managed by `engine/process_manager.py` (pid file in `data/`, logs to
`data/llama_server.log`; a **Start/Stop Engine** button lives in the sidebar).

---

## 8. Autonomous Platform (v2)

The v2 platform is additive. Existing chats, memory and Desktop Pilot endpoints stay
unchanged; v2 tables are created idempotently by `services/db.py::init_platform_db()` on
startup, and no destructive migration runs automatically.

### 8.1 Services

| Module | Responsibility |
|---|---|
| `services/agents.py` | Built-in/custom agent manifests, tool allowlists, model profiles, templates. |
| `services/tasks.py` | Tasks, runs, steps, events, dependency checks, budgets, transitions, recovery. |
| `services/run_manager.py` | Background supervision, pause/resume/cancel, event fan-out, restart recovery, kill switch integration. |
| `services/agent_runtime.py` | JSON-action execution loop, checkpointing, budget enforcement, approval waits, finalization. |
| `services/tools.py` / `builtin_tools.py` | Typed tool registry and built-in files, shell, web, knowledge, connector and desktop tools. |
| `services/policies.py` | Risk classes, approval records, policy evaluation, approval resume/failure handling. |
| `services/knowledge.py` | Sources, chunks, local 384-dimension hashed embeddings, FTS5 + semantic retrieval, cited answers. |
| `services/research.py` | Research projects, notes, claims, bibliography and report assembly. |
| `services/connectors.py` | Catalog, encrypted credentials, OAuth helpers, sync, cached items and actions. |
| `services/automations.py` | Schedule/webhook triggers, HMAC verification, scheduler loop and run history. |
| `services/files.py` | Approved-root file tree/read/write helpers, checkpoints, diffs and restore. |
| `services/code_index.py` | Symbol indexing, code search and git repository status. |
| `services/backup.py` | SQLite/workspace backups (optional passphrase encryption), restore and privacy cleanup. |
| `services/audit.py` | Append-only, hash-chained audit events and verification/export. |
| `services/mobile.py` | One-time pairing codes, hashed mobile tokens, briefs, commands and shares. |
| `services/safety.py` | Kill switch, offline/redaction modes, and an enforced PIN app lock with expiring unlock cookies. |

### 8.2 Runs, statuses and approvals

- Task/run statuses: `queued`, `waiting`, `running`, `paused`, `awaiting_approval`,
  `completed`, `partially_completed`, `failed`, `cancelled`, `interrupted`.
- `completed`, `partially_completed`, `failed` and `cancelled` are terminal. A run that
  exhausts a step/tool/time/cost budget after doing useful work ends as
  `partially_completed`.
- The runtime checkpoints messages, plan, artifacts, usage and step count after every
  action. Startup recovery marks unfinished runs `interrupted` and resumes those allowed
  by their budget.
- Tool execution can return `awaiting_approval`. The run pauses, a pending approval is
  recorded, and the agent continues only after `/approvals/{id}/resolve` approves it.
  Denial fails the run without retrying an equivalent bypass.
- Run events are persisted once per transition and streamed over
  `GET /api/v2/runs/{id}/events/stream` (SSE).

### 8.3 Command Center (`/platform`)

The Command Center is a no-build vanilla-JS operations UI backed entirely by `/api/v2`:

- Overview: active work, pending approvals, spend, recent runs, activity and safety.
- Tasks & Runs: create/start/pause/resume/retry/cancel tasks and inspect steps/events.
- Approvals: approve or deny pending actions; approved connector actions run explicitly.
- Knowledge: add notes, ask cited questions, inspect sources and reindex.
- Research: create projects, launch agent tasks, inspect notes/claims/report.
- Connectors: connect credentials or OAuth, test, sync and disconnect.
- Automations: schedule tasks, create HMAC webhooks, run/toggle/delete.
- Files & Code: approved-root browsing, previews, checkpoint operations, code indexing.
- Audit: verify the hash chain and export JSON.
- Settings: kill switch, offline mode, backups, mobile pairing and privacy export.

### 8.4 Pocket companion (`/mobile`)

1. In **Command Center → Settings**, create a pairing for a phone.
2. Open the displayed `/mobile?device=...&code=...` link (or enter the device ID and
   six-digit code manually) within 10 minutes.
3. The phone receives a bearer token stored in `localStorage`; only its hash is stored in
   SQLite. Requests use `X-Mobile-Token` or `Authorization: Bearer`.
4. The companion can read `/mobile/me`, `/mobile/brief`, run `/mobile/command` and
   `/mobile/share`, and revoke its device. Tokens can be revoked from Settings at any time.

### 8.5 Safety, files and audit

- The kill switch cancels supervised runs, updates their tasks, and schedules the
  automation scheduler to stop. Offline mode disables network/provider/connector paths.
- File tools are constrained to approved roots (`data/workspaces`, the repository, and
  `GROK_WORKSPACE_ROOTS`). Checkpoint restore can merge (default) or `prune=true` files
  added after the snapshot; every restore first creates a safety checkpoint.
- Connector credentials and sensitive values are encrypted with the local master key and
  redacted from audit/log output. The audit table is append-only with update/delete
  triggers and a SHA-256 hash chain.
- Backups deliberately exclude `data/secrets/master.key`: an unencrypted archive must
  never contain both ciphertext and the key that decrypts it. To restore encrypted
  credentials on a different machine, provide the original `GROK_MASTER_KEY` (or copy
  `master.key` separately). "Delete all data" clears legacy chats, memory, desktop tasks
  and v2 tables; `delete_files=true` also removes workspaces, uploads, attachments,
  checkpoints, plugins, the OpenAI key and runtime state. The safety backup and audit
  trail are retained unless explicitly removed.

---

## 9. HTTP API Reference

All endpoints are on `http://127.0.0.1:8000`.

| Method & path | Purpose |
|---|---|
| `GET /` | Serves the web UI (`static/index.html`). |
| `GET /api/status` | Engine readiness, active provider/model, key presence, memory counts, available modes. |
| `GET /api/models` | Merged local + OpenAI model list with badges, vision/reasoning flags, lock state, recommended GPU layers. |
| `POST /api/models/select` | Switch model: `{model_id, gpu_layers?, restart?}` (cloud ids prefixed `openai:`). |
| `POST /api/server/start` / `POST /api/server/stop` | Start/stop local `llama-server` (no-ops in cloud mode). |
| `GET /api/openai/status` | Key state (source, masked hint), base URL, model list, active model. |
| `POST /api/openai/key` | Validate + save key: `{api_key, base_url?}`; auto-switches to ChatGPT. |
| `DELETE /api/openai/key` | Remove saved key; reverts to local if no env key. |
| `GET /api/sessions` · `POST /api/sessions` · `DELETE /api/sessions/{id}` | Session list / create `{title, mode}` / delete (cascades messages + summary). |
| `GET /api/sessions/{id}/messages` | Full message history (with `image_url`). |
| `GET /api/sessions/{id}/export` | Markdown transcript `{title, markdown}`. |
| `POST /api/chat/stream` | Main chat: `{session_id, message, mode, image_data?, image_filename?}` → SSE stream. |
| `GET/POST /api/memory/dossier` · `DELETE /api/memory/dossier/{key}` | Dossier facts CRUD. |
| `GET /api/memory/episodic?q=` · `DELETE /api/memory/episodic/{id}` | Search / delete episodic memories. |
| `GET /api/memory/summary/{session_id}` | Rolling summary for a session. |
| `POST /api/desktop/run` | Run the Desktop Pilot on a goal → SSE step events. |
| `POST /api/desktop/stop` | Emergency stop. |
| `GET /api/desktop/status` | Pilot running state + screen bounds. |
| `GET /api/desktop/shot?path=` | Serve a pilot screenshot PNG (must be under `/tmp/grok_shots`). |
| `GET /platform` | Command Center UI (`static/platform.html`). |
| `GET /mobile` | Pocket companion UI (`static/mobile.html`); uses its own pairing token and stays reachable while the app lock is engaged. |
| `GET /unlock` | PIN unlock screen used when the optional app lock is enabled. |
| `GET /api/v2/health` · `GET /api/v2/meta` | Platform health, version, database path, capabilities and safety state. |
| `GET /api/v2/overview` | Dashboard, agents, tools, model profiles, stats and recent runs. |
| `GET/POST /api/v2/tasks` · `GET/PATCH/DELETE /api/v2/tasks/{id}` | Task board CRUD. |
| `POST /api/v2/tasks/{id}/{start,pause,resume,retry,cancel}` | Run lifecycle controls. |
| `GET /api/v2/runs/{id}` · `GET /api/v2/runs/{id}/events[/stream]` | Run state, persisted events and SSE timeline. |
| `GET /api/v2/approvals` · `POST /api/v2/approvals/{id}/resolve` | Pending approvals and explicit resolution. |
| `GET/POST /api/v2/knowledge/*` · `POST /api/v2/knowledge/ask` | Vault sources, search, cited answers and reindex. |
| `GET/POST /api/v2/research/*` | Research projects, notes, claims, reports and task launch. |
| `GET/POST /api/v2/connectors/*` · `/api/v2/connections/*` | Connector catalog, credentials, OAuth, sync, items and actions. |
| `GET/POST /api/v2/automations/*` · `POST /api/v2/hooks/{id}` | Scheduled/webhook automations and signed hook delivery. |
| `GET/POST /api/v2/files/*` · `POST /api/v2/files/checkpoints/{id}/restore` | Approved-root files, checkpoints/diffs and merge/prune restore. |
| `GET/POST /api/v2/code/*` · `/api/v2/backups/*` · `/api/v2/audit/*` | Code index, encrypted backups and hash-chain audit/export. |
| `GET/POST /api/v2/safety/*` | Kill switch, offline mode, redaction and app lock; unlock sessions last 8 hours. |
| `POST /api/v2/mobile/pair` · `GET /api/v2/mobile/{me,brief}` · `POST /api/v2/mobile/{command,share}` | Mobile pairing and token-authorized companion API. |

Mobile endpoints read `X-Mobile-Token` or `Authorization: Bearer`; the rest of `/api/v2`
is intended for the localhost UI (do not expose the server to an untrusted network
without an authenticating reverse proxy). When the app lock is enabled, local UI/API
requests require the `grok_unlock` cookie issued by `POST /api/v2/safety/unlock`;
`/static/*`, `/mobile`, `/api/v2/health`, `/unlock` and the lock-status endpoint remain
reachable. Existing paired mobile devices authenticate with their own token; creating a
new pairing, listing devices, or revoking a device still requires the unlock cookie. To
reset a forgotten PIN, stop the server and remove the `app_lock` row from
`platform_meta` in the SQLite database.

**`/api/chat/stream` SSE event types**: `telemetry` (context budget + image URL),
`token` (streamed text + tok/s), `desktop_step` (pilot progress), `error`, `done`
(full response). The desktop-run stream emits raw step events (same keys as
`desktop_step` minus the `type` field).

---

## 10. Data, Environment & Privacy

**Files created at runtime (all under the repo):**

```
data/
├── grok_memory.db         # SQLite: sessions, messages, dossier, episodic (+FTS5), summaries
├── active_model.json      # active provider + model (+ gpu layers state)
├── openai_key.json        # optional saved API key (chmod 600)
├── llama_server.pid       # pid of the managed llama-server
├── llama_server.log       # engine log
├── uploads/               # images you attach to chats
├── workspaces/            # isolated agent/task workspaces
├── backups/               # encrypted SQLite + workspace backups
├── file_checkpoints/      # file snapshot archives
├── attachments/           # knowledge/file ingestion cache
├── long_convo_report.json # outputs of the stress-test script
└── long_convo_transcript.md
/tmp/grok_shots/           # Desktop Pilot screenshots (ephemeral, safe to delete)
```

**Environment variables:**

| Variable | Default | Meaning |
|---|---|---|
| `OPENAI_API_KEY` | — | ChatGPT key (takes precedence over the saved file) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint override |
| `GROK_DB_PATH` | `data/grok_memory.db` | SQLite location (used by tests to isolate) |
| `GROK_STATE_FILE` | `data/active_model.json` | provider/model state file |
| `GROK_OPENAI_KEY_FILE` | `data/openai_key.json` | saved key location |
| `GROK_WORKSPACE_ROOTS` | unset | extra approved file roots, separated by `os.pathsep` |
| `LLAMA_HOST` / `LLAMA_PORT` | `127.0.0.1` / `8080` | local engine address |
| `WEB_HOST` / `WEB_PORT` | `127.0.0.1` / `8000` | web server binding |
| `CTX_SIZE` | `4096` | llama-server context size |
| `GPU_LAYERS` | `33` | default `-ngl` (GPU offload) override |
| `FLASH_ATTN` | `on` | llama-server flash attention |
| `MAX_PROMPT_TOKENS` / `CLOUD_PROMPT_BUDGET_TOKENS` / `CLOUD_SLIDING_BUDGET_TOKENS` | 3000 / 12000 / 6000 | memory budgets |
| `PYTHON_BIN`, `LLAMA_SERVER_BIN` | auto | used by `scripts/start.sh` only |

**Privacy:** chat, memory, screenshots and keys stay on the machine. In local mode
nothing leaves it. In ChatGPT mode, only the assembled prompt (which includes memory
context) and attached images are sent to your configured OpenAI endpoint.

---

## 11. Running & Testing

### Start the app

```bash
cd /home/mevlec/Data/grok
./start.sh          # → wrapper around scripts/start.sh
```

`scripts/start.sh` picks the interpreter (env `PYTHON_BIN` → project `.venv` → system
pythons → auto-provisions `.venv` with `uv`), reports Desktop Pilot availability,
boots `llama-server` when the local provider is active, then serves the app on
http://localhost:8000.

One-time venv setup (enables the Desktop Pilot by exposing system PyGObject):

```bash
uv venv --system-site-packages .venv
uv pip install --python .venv/bin/python fastapi uvicorn httpx cryptography pydantic pillow
```

### Tests & tools (all offline unless noted)

| Script | What it does |
|---|---|
| `scripts/test_openai_integration.py` | End-to-end ChatGPT-provider test against the bundled mock API (42 checks: key validation, model listing/switching, streaming chat, memory persistence, sanitize-retry). |
| `scripts/mock_openai_server.py` | The mock OpenAI API (`127.0.0.1:8123`, key `sk-mock-valid`) used by the suite. |
| `scripts/test_memory.py` | Unit tests for the 4-tier memory (dossier extraction, FTS search, summaries, budget invariant). |
| `tests/test_v2_platform.py` | Offline v2 API/UI suite (13 tests): health/meta, platform/mobile/unlock routes, PIN app-lock enforcement, out-of-root workspace rejection, mobile pairing/command/brief/revoke with brute-force throttling, task completion/events/recent runs, approval pause/resume with argument binding, run pause/resume, tool-budget partial completion, three-failure stop guard, redirect-SSRF blocking, audit-chain verification, backup create/inspect/stage/delete with master-key exclusion, large append writes and checkpoint prune. Run with `.venv/bin/python -m unittest -v tests.test_v2_platform`. |
| `scripts/stress_test_long_convo.py` | 25-turn conversation stress test with adversarial recall probes; writes a transcript + JSON report into `data/`. |
| `scripts/test_vision_api.py` / `test_natural_conversation.py` | Scripted chat/vision checks against a running server. |
| `scripts/browser_*.py` | Headless-Chrome UI tests (require `websockets` + `google-chrome`). |
| `scripts/run_gpu.sh` | Standalone `llama-server` launcher (dolphin model, `-ngl 33`). |

---

## 12. Troubleshooting

| Symptom | Fix |
|---|---|
| Engine card shows **Server Offline** | app not running — start it with `./start.sh` |
| **API Key Missing** / cloud models locked (🔒) | click 🔑 and save a key, or `export OPENAI_API_KEY=sk-…` before start |
| Desktop Pilot error about `python3-gi` | use the `.venv` setup from §6.1/§11 (system PyGObject + pip deps) |
| Local model won't start | check `data/llama_server.log`; verify the GGUF path in `config.py → MODELS`; VRAM may be low — pick a lower GPU-offload level |
| `No module named 'certifi'` (older setups) | `pip install certifi`, or rely on the built-in fallback (`config.httpx_verify()`) |
| Port 8000 busy | another instance is running: `ss -tlnp \| grep 8000` |
| Screenshots in the pilot viewer go stale | the pilot finished or was stopped; press Stop Pilot and run a new goal |

---

## 13. File Map

```
server.py                     FastAPI app: chat endpoints, SSE, static serving, v2 router/lifespan
api_v2.py                     /api/v2 platform endpoints
services/                     v2 platform services (agents, runs, approvals, knowledge, connectors, audit...)
tests/test_v2_platform.py     offline v2 API/UI smoke tests
config.py                     paths, providers, model catalogs, key mgmt, budgets, VRAM logic
start.sh                      convenience wrapper → scripts/start.sh
requirements.txt              pip dependencies (+ notes about PyGObject)
engine/
  router.py                   ProviderRouter: local vs cloud dispatch
  llama_client.py             llama.cpp client (raw /completion + chat/vision API)
  openai_client.py            OpenAI streaming client (reasoning models, sanitize-retry)
  process_manager.py          llama-server lifecycle (start/stop/switch, health, pid/log)
  desktop_agent.py            Desktop Pilot loop (prompting, parsing, actions, abort)
  wayland_controller.py       Mutter DBus: input injection, screenshots, OCR, geometry
  vision.py                   BLIP caption + Tesseract OCR fallback pipeline
memory/
  database.py                 SQLite schema, connections, FTS5 triggers
  dossier.py                  Tier 1: permanent user facts (+ heuristic extraction)
  episodic.py                 Tier 2: episodic store + FTS5/BM25 retrieval
  summarizer.py               Tier 3: rolling extractive session summaries
  orchestrator.py             4-tier prompt assembly + post-interaction tasks
persona/
  prompts.py                  GROK_MODES: persona system prompts + sampling
  chatml.py                   token estimation + ChatML formatting
static/                       UI (no build step): chat, /platform Command Center, /mobile Pocket, CSS/JS
scripts/                      start/dev/test/utilities (see §11)
data/                         runtime state (DB, keys, uploads, logs) — see §10
```
