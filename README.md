# GROK // Local Uncensored Chatbot

> 📖 Full system documentation — architecture, all features, and how the Desktop Pilot
> works under the hood: **[DOCUMENTATION.md](DOCUMENTATION.md)**

An uncensored, truthful AI chatbot modeled after Grok, running either:
- **Locally** via CUDA-accelerated `llama.cpp` (`llama-server`) with **Dolphin 3.0 Llama 3.1 8B** / **Qwen 3.5 9B Vision**, or
- **In the cloud** via your **ChatGPT / OpenAI API key** (GPT-5.6, GPT-5.x, GPT-4.1, o-series, Codex).

Both engines share the same specialized **4-tier hierarchical memory architecture**, personas, vision pipeline, and desktop pilot.

---

## ⚡ Quick Start

```bash
cd /home/mevlec/Data/grok
./start.sh
```

Then open your browser at **[http://localhost:8000](http://localhost:8000)**.

(`./start.sh` is a thin wrapper around `scripts/start.sh`. It prefers the project `.venv`,
auto-detects another Python with the dependencies otherwise, boots `llama-server` when the
local provider is active, and starts the web backend.)

### Option A — ChatGPT API key (cloud)

1. Click the **🔑 button** in the bottom-left engine card.
2. Paste your OpenAI API key (`sk-...`) and press **Save & Validate**. The key is validated live, stored locally in `data/openai_key.json` (chmod 600), and the app switches to ChatGPT automatically.
3. Pick any **☁️ model** in the top model dropdown (GPT-5.6 Terra, GPT-5.6 Luna, GPT-5.4/5.x mini & nano, GPT-4.1 mini/nano, GPT-4o mini, o4-mini, o1-mini, Codex Mini...).

Alternatively export the key before starting — no UI setup needed:

```bash
export OPENAI_API_KEY="sk-..."
./start.sh
```

(Optional) Use any OpenAI-compatible endpoint with `OPENAI_BASE_URL="https://..."`.

### Option B — Local GGUF models (offline)

In the bottom-left corner of the UI, click **"Start Engine"** to launch `llama-server`. When the indicator turns green (**Engine Online**), you are ready to chat. Select a local model (`⚡`/`👁️` entries) in the top dropdown to switch back from cloud mode.

---

## 🧠 Why the Low-Context Memory System Works

Small context windows (e.g., 2,048 – 4,096 tokens) normally cause local LLMs to forget previous conversations or crash with out-of-memory errors after 5–8 messages.

To solve this, this chatbot uses a **4-Tier Memory Engine**:

1. **Tier 1: Permanent User Dossier (Entity Memory)**:
   - Learns who you are (name, age, goals, struggles, family dynamics, secrets, personal preferences).
   - Injected into every prompt as a compact, permanent bulleted dossier (~100 tokens).
   - **Never forgotten**, even across new sessions or weeks later.
2. **Tier 2: Episodic Semantic Archive (RAG via FTS5)**:
   - Automatically indexes conversation highlights and past discussions into an SQLite FTS5 database.
   - When you ask a question related to a past topic, relevant memories are recalled on-the-fly (~200 tokens).
3. **Tier 3: Rolling Conversation Compactor**:
   - Compresses long dialogues beyond the sliding window into an evolving narrative summary.
4. **Tier 4: Active Sliding Window**:
   - Keeps the last 4–6 messages verbatim for immediate flow, banter, and context.
5. **Strict Token Budget Invariant**:
   - Locally the total prompt stays under **3,000 tokens** (1,000+ reserved for generation inside the 4,096 context window).
   - In ChatGPT cloud mode the same tiers automatically scale up to a **12,000-token memory budget** since cloud context windows are far larger.

---

## 🔑 ChatGPT Cloud Mode (OpenAI API)

- **Key management**: paste the key in the UI (🔑) or use `OPENAI_API_KEY`. It is never exposed back to the browser (only a masked hint is shown). Remove it any time from the same modal.
- **Model routing**: selecting a `☁️` model sends the assembled 4-tier memory (system persona + dossier + episodic recall + rolling summary + sliding window) as structured chat messages. The local `llama-server` is not touched and can stay stopped.
- **Reasoning models**: GPT-5.x / o-series / Codex models automatically receive `max_completion_tokens` + `reasoning_effort` (mode-aware: Deep Truth → high, classic → medium, desktop pilot → low) instead of sampling parameters. Unsupported parameters are retried once automatically for OpenAI-compatible endpoints.
- **Vision**: GPT-5.x / GPT-4.1 / GPT-4o / o4-mini accept images natively. Text-only models (o1-mini, Codex) fall back to the local BLIP + OCR analysis block.
- **Desktop Pilot**: works in cloud mode too — same Mutter Wayland control loop, driven by the ChatGPT model.

---

---

## 🎭 Grok Personas & Desktop Pilot

Switch between modes using the top pills in the UI:
- **Grok Classic**: Witty, candid, perceptive, zero corporate disclaimers, authentic confidant.
- **Unhinged Grok**: Completely unfiltered, profanity-friendly, maximum banter, raw honesty.
- **Deep Truth**: First-principles analytical thinker, deconstructing spin, bias, and media narratives.
- **Desktop Pilot (Mutter Wayland GNOME 46)**: Autonomously operates Chrome and desktop apps using the cursor and keyboard via native `org.gnome.Mutter.RemoteDesktop` and zero-prompt PipeWire screen perception. Includes an Emergency Stop button in the top bar.

### Desktop Pilot prerequisites (GNOME / Wayland)

The pilot drives your desktop through `org.gnome.Mutter.RemoteDesktop`, which needs
PyGObject (`python3-gi`) **in the interpreter running the app**. Custom Python builds
(e.g. uv-managed 3.12/3.13) usually don't include it — the system Python does.

`./start.sh` prefers the project `.venv` when it exists. One-time setup:

```bash
cd /home/mevlec/Data/grok
uv venv --system-site-packages .venv
uv pip install --python .venv/bin/python fastapi uvicorn httpx
```

`--system-site-packages` is what makes the system PyGObject (and Pillow) visible inside
the venv. Without `gi` the pilot reports a clear error while chat, ChatGPT mode, local
models and memory all keep working.

---

## 🛠️ Memory Inspector Drawer

Click **"Memory Dossier"** in the sidebar to:
- View all facts Grok currently knows about you.
- Manually add, edit, or delete facts.
- Search episodic memories from past chats.

---

## 🚀 GPU & VRAM Performance

- Model: `dolphin3.0-llama3.1-8b-q4_k_m.gguf` (4.6 GB)
- Full GPU Offload: Requires ~4.8 GB VRAM (`-ngl 33`).
- If another heavy GPU workload (e.g., ComfyUI) is active, the engine automatically adjusts to hybrid or CPU mode.

---

## 🧪 Offline ChatGPT Provider Test

No API key or internet needed — the test spins up a local mock of the OpenAI API and
drives the real app against it (key save/reject, model list + switching, streaming chat,
memory persistence, key removal, and the reasoning-model parameter auto-retry):

```bash
python3.12 scripts/test_openai_integration.py
```

The mock server itself lives in `scripts/mock_openai_server.py` (port 8123, only accepts
`sk-mock-valid`).

