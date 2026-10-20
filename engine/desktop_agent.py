"""
LLM-Driven Autonomous Desktop Agent for Grok on GNOME 46 Wayland.
Powered by the active inference engine (local llama.cpp model or ChatGPT API), ReAct cognitive loop.
Perceives live screen state via Mutter ScreenCast + Tesseract OCR and dynamically reasons
to execute mouse and keyboard actions across Brave, Chrome, and any Linux desktop application.
"""
import os
import re
import time
import json
import base64
import asyncio
import io
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, AsyncGenerator
from PIL import Image

logger = logging.getLogger("desktop_agent")

from .wayland_controller import WaylandController
from .process_manager import LlamaProcessManager
from .llama_client import LlamaClient
from .browser_profiles import get_available_profiles_summary
from memory.dossier import UserDossier
from persona.chatml import format_chatml_prompt
from config import (
    get_active_provider,
    PROVIDER_OPENAI,
    get_openai_api_key,
    get_active_engine_label,
    get_active_engine_info,
)

def prepare_screenshot_for_llm(shot_path: str, max_dim: Optional[int] = None, quality: int = 85) -> Optional[str]:
    """
    Prepares screenshot for vision model. For cloud models (OpenAI / ChatGPT API),
    preserves full native resolution up to 2048px to prevent coordinate distortion.
    For local llama-server, downscales to 1024px if needed to save VRAM.
    """
    if not shot_path or not os.path.exists(shot_path):
        return None
    if max_dim is None:
        max_dim = 2048 if get_active_provider() == PROVIDER_OPENAI else 1024
    try:
        with Image.open(shot_path) as img:
            rgb_img = img.convert("RGB")
            if max(rgb_img.size) > max_dim:
                rgb_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            rgb_img.save(buf, format="JPEG", quality=quality)
            b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64_str}"
    except Exception as e:
        logger.warning(f"Error compressing screenshot for vision model: {e}")
        try:
            with open(shot_path, "rb") as f:
                b64_str = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{b64_str}"
        except Exception:
            return None


def split_chained_actions(action_str: str) -> List[str]:
    """Splits chained actions on ' then ' or ' && ' only when outside quoted string literals."""
    tokens = []
    current = []
    in_quote = None
    i = 0
    while i < len(action_str):
        ch = action_str[i]
        if ch in ('"', "'"):
            if in_quote == ch:
                in_quote = None
            elif in_quote is None:
                in_quote = ch
            current.append(ch)
            i += 1
        elif in_quote:
            current.append(ch)
            i += 1
        elif action_str[i:i+6].lower() == " then ":
            tokens.append("".join(current).strip())
            current = []
            i += 6
        elif action_str[i:i+4] == " && ":
            tokens.append("".join(current).strip())
            current = []
            i += 4
        else:
            current.append(ch)
            i += 1
    if current:
        tokens.append("".join(current).strip())
    return [t for t in tokens if t]


def compute_screen_delta(
    shot_a: Optional[str],
    shot_b: Optional[str],
    ocr_a: Optional[set],
    ocr_b: Optional[set]
) -> float:
    """
    Computes screen difference delta between 0.0 (identical) and 1.0 (completely changed).
    Combines OCR text Jaccard difference and downscaled pixel comparison.
    """
    ocr_delta = 1.0
    if ocr_a is not None and ocr_b is not None:
        if not ocr_a and not ocr_b:
            ocr_delta = 0.0
        else:
            intersection = len(ocr_a & ocr_b)
            union = len(ocr_a | ocr_b)
            ocr_delta = 1.0 - (intersection / max(union, 1))

    pixel_delta = 1.0
    if shot_a and shot_b and os.path.exists(shot_a) and os.path.exists(shot_b):
        try:
            with Image.open(shot_a) as img_a, Image.open(shot_b) as img_b:
                small_a = img_a.convert("L").resize((64, 64), Image.Resampling.BOX)
                small_b = img_b.convert("L").resize((64, 64), Image.Resampling.BOX)
                diff = sum(abs(p1 - p2) for p1, p2 in zip(small_a.getdata(), small_b.getdata()))
                pixel_delta = diff / (64 * 64 * 255.0)
        except Exception:
            pixel_delta = ocr_delta

    if ocr_a is not None and shot_a and os.path.exists(shot_a):
        return (ocr_delta + pixel_delta) / 2.0
    elif ocr_a is not None:
        return ocr_delta
    return pixel_delta


def get_action_signature(action_str: str) -> str:
    """Extracts a normalized signature for action target comparison (e.g. click coordinates or text)."""
    clean = action_str.strip().lower()
    # click(x, y), double_click(x, y), right_click(x, y)
    m = re.match(r"(?:click|double_click|right_click)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", clean)
    if m:
        return f"coord:{m.group(1)},{m.group(2)}"
    # type_into(x, y, ...)
    m = re.match(r"type_into\s*\(\s*(\d+)\s*,\s*(\d+)", clean)
    if m:
        return f"coord:{m.group(1)},{m.group(2)}"
    # click_text("...")
    m = re.match(r"click_text\s*\(\s*[\"'](.*?)[\"']", clean)
    if m:
        return f"text:{m.group(1).strip()}"
    # click_id(id)
    m = re.match(r"click_id\s*\(\s*(\d+)", clean)
    if m:
        return f"id:{m.group(1)}"
    # type_text("...")
    m = re.match(r"type_text\s*\(\s*[\"'](.*?)[\"']", clean)
    if m:
        return f"type:{m.group(1).strip()}"
    return clean


def parse_type_into_args(args_str: str) -> Optional[Dict[str, Any]]:
    """
    Parses arguments for type_into call. Supports:
    - (x, y, text, submit=True|False)
    - (x, y, text) [default submit=True]
    - (id, text, submit=True|False)
    - (text, submit=True|False)
    """
    raw = args_str.strip()
    submit = True
    if re.search(r"submit\s*=\s*(false|0)", raw, re.IGNORECASE):
        submit = False
    elif re.search(r"submit\s*=\s*(true|1)", raw, re.IGNORECASE):
        submit = True
    elif re.search(r",\s*false\s*$", raw, re.IGNORECASE):
        submit = False

    # Check for x, y coordinates
    m_xy = re.match(r"^(\d+)\s*,\s*(\d+)\s*,\s*[\"'](.*)", raw, re.DOTALL)
    if m_xy:
        x, y = int(m_xy.group(1)), int(m_xy.group(2))
        rest = m_xy.group(3)
        text_m = re.match(r"^(.*?)(?:[\"']\s*(?:,\s*(?:submit\s*=\s*)?(?:True|False))?\s*)$", rest, re.DOTALL | re.IGNORECASE)
        text = text_m.group(1) if text_m else rest.rstrip("\"'). ")
        return {"type": "xy", "x": x, "y": y, "text": text, "submit": submit}

    # Check for id, text
    m_id = re.match(r"^(\d+)\s*,\s*[\"'](.*)", raw, re.DOTALL)
    if m_id:
        idx = int(m_id.group(1))
        rest = m_id.group(2)
        text_m = re.match(r"^(.*?)(?:[\"']\s*(?:,\s*(?:submit\s*=\s*)?(?:True|False))?\s*)$", rest, re.DOTALL | re.IGNORECASE)
        text = text_m.group(1) if text_m else rest.rstrip("\"'). ")
        return {"type": "id", "id": idx, "text": text, "submit": submit}

    # Text only
    m_txt = re.match(r"^[\"'](.*)", raw, re.DOTALL)
    if m_txt:
        rest = m_txt.group(1)
        text_m = re.match(r"^(.*?)(?:[\"']\s*(?:,\s*(?:submit\s*=\s*)?(?:True|False))?\s*)$", rest, re.DOTALL | re.IGNORECASE)
        text = text_m.group(1) if text_m else rest.rstrip("\"'). ")
        return {"type": "text", "text": text, "submit": submit}

    return None


KNOWN_ACTIONS_REGEX = re.compile(
    r"\b(launch_app|open_url|navigate_browser|click_id|click_text|click|double_click|right_click|type_into|type_text|paste_text|copy_selection|press_key|scroll|wait|finish|write_file|edit_file|read_file|run_shell|upload_file_to_dialog|wait_for_condition|save_latest_download)\s*\([^)]*\)",
    re.IGNORECASE
)


def parse_pilot_response(response_text: str) -> Tuple[str, str]:
    """
    Parses Thought: and Action: from model output with resilience against:
    - <think>...</think> reasoning blocks
    - Markdown bolding (**Action:**, **Thought:**)
    - Code fences and backticks
    - Missing Action: labels when a known action function call is present
    """
    logger.info("Raw Pilot LLM output: %r", response_text)

    thought = ""
    action_str = ""
    extracted_think = ""

    # 1. Extract and strip <think>...</think> or unclosed <think>...
    if "</think>" in response_text.lower():
        think_match = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL | re.IGNORECASE)
        if think_match:
            extracted_think = think_match.group(1).strip()
        cleaned = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL | re.IGNORECASE).strip()
    elif "<think>" in response_text.lower():
        # Unclosed <think>: check if Action: or a known tool call is present later in the string
        action_split = re.search(
            r"(?:(?:\*\*|\*)?Action(?:\*\*|\*)?:?|\b(?:launch_app|open_url|navigate_browser|click|double_click|right_click|type_into|type_text|paste_text|press_key|scroll|wait|finish)\s*\()",
            response_text,
            re.IGNORECASE
        )
        if action_split:
            idx = action_split.start()
            extracted_think = re.sub(r"^<think>", "", response_text[:idx], flags=re.IGNORECASE).strip()
            cleaned = response_text[idx:].strip()
        else:
            extracted_think = re.sub(r"^<think>", "", response_text, flags=re.IGNORECASE).strip()
            cleaned = ""
    else:
        cleaned = response_text.strip()

    # 2. Normalize markdown bolding on labels and strip code fences
    cleaned = re.sub(r"\*\*(?:Action|Action:)\*\*:?", "Action:", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\*(?:Action|Action:)\*:?", "Action:", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\*\*(?:Thought|Thought:)\*\*:?", "Thought:", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\*(?:Thought|Thought:)\*:?", "Thought:", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"```(?:json|python|bash|sh)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    # 3. Match explicit Thought: and Action:
    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", cleaned, re.DOTALL | re.IGNORECASE)
    action_match = re.search(r"Action:\s*(.*)", cleaned, re.IGNORECASE)

    if thought_match:
        thought = thought_match.group(1).strip()
    if action_match:
        action_str = action_match.group(1).strip()
    elif "Action:" in cleaned:
        action_str = cleaned.split("Action:")[-1].strip()

    # 4. Fallback Action Matcher: scan entire cleaned text for known action patterns
    if not action_str or not KNOWN_ACTIONS_REGEX.search(action_str):
        fallback_action = KNOWN_ACTIONS_REGEX.search(cleaned)
        if fallback_action:
            action_str = fallback_action.group(0).strip()

    # Clean action string (remove backticks, trailing periods/semicolons)
    if action_str:
        action_str = action_str.strip().strip("`").rstrip(";.")
        if "\n" in action_str:
            lines = [ln.strip() for ln in action_str.split("\n") if ln.strip()]
            for ln in lines:
                if KNOWN_ACTIONS_REGEX.search(ln):
                    action_str = ln
                    break
            else:
                action_str = lines[0]
        # If action_str contains conversational words around a known single tool call, isolate the call
        if " then " not in action_str and " && " not in action_str:
            m_known = KNOWN_ACTIONS_REGEX.search(action_str)
            if m_known:
                action_str = m_known.group(0).strip()

    # 5. Fallback Thought
    if not thought:
        if extracted_think:
            first_line = extracted_think.split("\n")[0].strip()
            thought = first_line[:140] if first_line else "Analyzing screen state."
        elif cleaned and action_str and cleaned != action_str:
            pre_text = cleaned.split(action_str)[0].strip()
            if pre_text and len(pre_text) < 160:
                thought = pre_text
            else:
                thought = "Evaluating screen state and choosing next action."
        else:
            thought = "Evaluating screen state and choosing next action."

    return thought, action_str


SYSTEM_PROMPT = """You are Grok Desktop Agent, an intelligent AI piloting a Linux GNOME Wayland desktop.
You control the desktop using mouse and keyboard actions to achieve the user's objective.

Available Tools:
1. open_url("url_or_query", browser="...", profile="...") - Directly opens browser to any URL or Google search query in 1 fast step (e.g. open_url("https://github.com", profile="mevinbenty") or open_url("weather in tokyo")). PREFERRED for web browsing!
2. launch_app("app_name", profile="...") - Launch or focus an application (e.g. launch_app("antigravity"), launch_app("code"), launch_app("chrome", profile="mevinbenty"), launch_app("terminal"), launch_app("nautilus")).
3. navigate_browser("url_or_query") - In an already-open browser window, automatically focuses the address bar (Ctrl+L), types the URL/query, and presses Enter.
4. type_into(x, y, "text", submit=True) - Combined click, clear, type/paste, and submit tool. Moves cursor to (x, y), clicks to focus, clears field, types text, and presses Enter if submit is True (default True). ALWAYS use this for text inputs, prompt bars, and search fields!
5. click(x, y) - Move the mouse cursor to absolute coordinate (x, y) and left-click.
6. double_click(x, y) - Move to (x, y) and double-click.
7. right_click(x, y) - Move to (x, y) and right-click (open context menu).
8. click_text("text") - Search for matching text on screen and click it.
9. click_id(id) - Click element by its index ID from the screen state.
10. type_text("string") - Type text using keyboard into the currently focused window or field.
11. paste_text("string") - Paste text directly into the focused field via clipboard (faster for long strings).
12. copy_selection() - Copy current selection (Ctrl+C) and inspect clipboard text.
13. press_key("key_combo") - Press a key or shortcut (e.g. "enter", "ctrl+l", "ctrl+t", "ctrl+w", "alt+tab", "super", "backspace", "escape", "tab").
14. scroll("up" | "down") - Scroll page up or down.
15. wait(seconds) - Pause for seconds to let a page or app load (e.g. wait(2)).
16. write_file("rel_path", "content") - Write code, configs, or dataset metadata to workspace.
17. edit_file("rel_path", "old_str", "new_str") - Patch workspace files for code updates and self-healing debugging.
18. read_file("rel_path") - Read workspace file contents or logs.
19. run_shell("command") - Execute builds, installs (pip/npm), pytest, or shell commands inside workspace.
20. upload_file_to_dialog("file_path") - Auto-paste and attach reference image/file to browser file picker dialog.
21. wait_for_condition("description", max_wait_sec=45) - Polls until an async operation (like image generation) completes.
22. save_latest_download("target_rel_path") - Saves latest download from ~/Downloads to workspace (e.g. "images/img_001.png").
23. finish("summary") - The goal has been successfully accomplished.

CRITICAL OPERATIONAL RULES:
1. APP LAUNCHING & FOCUS RULE:
   If the user's goal specifies an application (e.g. Antigravity IDE, VS Code, Terminal, Chrome, Brave, Files, Calculator) and that app is NOT already focused on screen, your VERY FIRST ACTION MUST BE:
   Action: launch_app("<app_name>")
   - If the user asks for Antigravity (e.g. "antigravity", "antigravity tab", "antigravity project"), use: launch_app("antigravity").
   - If the user asks for VS Code / code editor, use: launch_app("code").
   - If the user asks for Terminal, use: launch_app("terminal").
   - If the user asks for web browsing, search, or a website URL, use: open_url(...) or launch_app("<browser>").
   - CRITICAL: NEVER launch or focus Brave, Chrome, or any web browser when the user requested an IDE, code editor, or desktop application! Do NOT mistake editor tabs or project windows for browser tabs!
   NEVER click random desktop icons, taskbars, or type text before the requested application is opened!

2. TEXT INPUT & CHATGPT RULES:
   When interacting with prompt bars, chat inputs, login fields, or search boxes, ALWAYS use type_into(x, y, 'text', submit=True) or type_into(x, y, 'text'). NEVER rely on bare click(x, y) or click_text() followed by separate typing steps.
   When interacting with ChatGPT (chatgpt.com) or other web chat apps:
   - ChatGPT's input is a prompt textarea near the bottom center (often labeled 'Ask anything' or 'Message ChatGPT'). ALWAYS use type_into(x, y, 'your message', submit=True).
   - If clicking coordinates does not activate or type into the input, DO NOT repeat the exact same click! Target visible button text (e.g. click_text('Ask anything')) or use keyboard shortcuts: press_key('tab') to cycle focus into the input box or press_key('enter').
   - If pressing Enter produces a newline instead of submitting, click the visible Send button (arrow icon next to the input) or send press_key('ctrl+enter').
   - NEVER repeat an identical click or action if the screen state has not changed!

3. INTERMEDIATE BLOCKERS & PROFILE SELECTORS:
   If an unexpected modal, "Who's using Brave?", "Who's using Chrome?" profile picker, cookie banner, or dialog appears after launching an app, you MUST interact with it first before attempting navigate_browser again. Click your profile name (e.g. click_text("Person 1") or click(x, y) on the avatar) or use press_key("enter") to enter the browser.

4. BROWSER NAVIGATION RULE (Brave, Chrome, Firefox):
   Once the browser window is open and unobstructed, to search or navigate, ALWAYS prefer:
   Action: navigate_browser("<query_or_url>")
   Alternatively: press_key("ctrl+l") to focus address bar, then type_text("<query>"), then press_key("enter").

5. NEVER TRUST ACTION HISTORY ALONE & VERIFICATION RULE:
   Do NOT call finish() just because a previous step says navigate_browser or click was executed. You MUST verify in the Current Screen State (and screenshot) that the final goal state is actually visible on screen (e.g. actual search results for the query or target page content are visible). If a modal blocked your action or the page is still loading, wait or resolve the blocker.

6. DESKTOP APPS RULE:
   For Calculator, Files, or Settings: look at the exact labels in "Current Screen State" and click the target with click_text("<label>") or click(x, y).

7. COMPLETION RULE:
   Only after visually/OCR verifying in Current Screen State that the goal is visibly accomplished, call finish("<summary of what was achieved>").

8. NEVER CLICK OR INTERACT WITH GROK CHAT WINDOW:
   Under NO circumstances should you click or interact with the GROK web application interface itself (identifiable by "GROK // Local Uncensored", "localhost:8000", "127.0.0.1:8000", or GROK control panels like "Stop Pilot"). If GROK is visible on screen, do NOT type in its address bar or click its buttons. Always call launch_app("<target_app>") to open or focus your destination application (e.g. launch_app("antigravity") if the goal mentions Antigravity, or the specific target requested by the user).

9. DIRECT ACTION FORMAT RULE (NO THINKING TAGS):
   Do NOT output <think>...</think> tags or conversational preamble. Immediately output:
   Thought: <1-2 sentences reasoning about what you see on screen and choosing the next step>
   Action: <single tool call, e.g. launch_app("antigravity") or type_into(450, 800, "hello", submit=True) or finish("Done")>

10. LOGGED-IN PROFILES RULE:
   ALWAYS prefer the user's logged-in browser and profile via open_url(url, browser="...", profile="...") or launch_app("...", profile="..."). If the user mentions an account, profile, or username (e.g. 'mevinbenty'), specify profile="mevinbenty". If no profile is specified, use the primary active logged-in profile listed in Available Logged-In Browser Profiles.

11. HYBRID WORKSPACE & CODING RULE:
   For software projects, coding, file creation, or testing, NEVER try to type code character-by-character into text editors using click or type_text. ALWAYS use write_file("path", "content") and run_shell("command"). Use edit_file("path", "old", "new") to fix bugs.

12. FALLBACK & ERROR RECOVERY RULES:
   - VERIFY TARGET BEFORE ADVANCING: After typing or clicking, inspect the next screen state. Did the text land in the intended input field (e.g. the AI Chat prompt bar) rather than the code editor or terminal?
   - IMMEDIATE UNDO: If text was accidentally typed into the code editor or wrong window, your IMMEDIATE NEXT ACTION MUST BE to undo or clear it (e.g. press_key("ctrl+z") or press_key("escape")) before trying anything else!
   - NO BLIND WAITING: NEVER call wait(5), wait(10), or wait(15) unless you see active visual proof of execution on screen (e.g. streaming tokens, spinner, active terminal build). If the prompt was not submitted, DO NOT wait!
   - ANTIGRAVITY & VS CODE NAVIGATION:
     * In Antigravity IDE, the Chat / Agent panel is located on the RIGHT side of the screen (x > 1400 on a 1920x1200 monitor).
     * To focus the Chat input in Antigravity or VS Code, use press_key("ctrl+l") or click the chat input prompt area on the lower right (around x=1600, y=1080).
     * NEVER click in the center-left area (x < 1300) when aiming for the Chat panel, as that is the code editor.

13. EXACT COORDINATE RESOLUTION:
   - Current Desktop Resolution is 1920x1200.
   - All coordinates for click(x, y), double_click(x, y), and type_into(x, y) MUST be specified in this native 1920x1200 coordinate space.
   - Always refer to the coordinates provided in the Current Screen State list or use click_text("exact phrase") to hit elements precisely.

Respond STRICTLY in this format:
Thought: <1-2 sentences reasoning about what you see on screen and choosing the next step>
Action: <single tool call, e.g. launch_app("antigravity") or type_into(450, 800, "cats", submit=True) or finish("Done")>"""


WORKSPACE_SYSTEM_PROMPT = """You are Grok Task Agent, an autonomous coding, research and file agent working INSIDE a sandboxed workspace.
You do NOT control a GUI. You operate only through workspace and web tools, and you must verify your work.

Available Tools:
1. write_file("rel_path", "content") - Create/overwrite a file in the workspace.
2. edit_file("rel_path", "old_str", "new_str") - Patch an existing workspace file.
3. read_file("rel_path") - Read a workspace file.
4. run_shell("command") - Run builds, installs, pytest, or shell commands inside the workspace.
5. web_search("query") - Search the internet and get ranked results (title/url/snippet).
6. web_fetch("url") - Fetch a web page and extract readable text.
7. web_research("query", fetch_top=2) - Search then fetch the top pages for deeper context.
8. save_web_result("rel_path") - Save the most recent web context into a workspace file.
9. finish("summary") - The goal has been accomplished.

CRITICAL RULES:
1. For code, always use write_file/edit_file. Never describe code without writing it.
2. Use run_shell to actually run tests/builds and read the output. Report real results, not assumptions.
3. Use web_search/web_fetch to gather facts you do not know. Cite the URLs you used.
4. Verify before finishing: after writing files or running commands, read/confirm the output in a subsequent step.
5. Keep going until the objective is truly complete, then call finish("<summary + verification evidence>").

Respond STRICTLY in this format:
Thought: <1-2 sentences reasoning about the next step>
Action: <single tool call, e.g. write_file("main.py", "...") or run_shell("pytest -q") or finish("Done")>"""


class DesktopAgent:
    def __init__(self, llm_client=None, llama_client=None, controller=None, workspace_manager=None, use_gui: bool = True):
        self._custom_controller = controller is not None
        self.controller = controller or WaylandController.get_instance()
        self.workspace_manager = workspace_manager
        self.use_gui = use_gui
        # Client may be a LlamaClient, an OpenAIClient, or a ProviderRouter
        self.llm_client = llm_client or llama_client or LlamaClient()
        self.is_running = False
        self._abort_requested = False
        self.current_task = ""
        self.step_history: List[Dict[str, Any]] = []
        self._last_web_context: str = ""

    def abort(self) -> None:
        """Emergency stop for active agent execution."""
        self._abort_requested = True
        self.is_running = False

    def _ensure_engine(self) -> bool:
        """Ensures the active inference engine is reachable (local llama-server or cloud API)."""
        if get_active_provider() == PROVIDER_OPENAI:
            return bool(get_openai_api_key())
        return LlamaProcessManager.ensure_running(wait_seconds=60.0)

    def _format_screen_elements(self, raw_elements: List[Dict[str, Any]], max_elements: int = 80) -> str:
        """Formats OCR detections into clean, high-density UI elements for LLM reasoning, prioritizing central modals."""
        if not raw_elements:
            return "No prominent text elements detected on screen (screen may be idle or blank)."

        # Blocklist GROK UI strings to prevent self-clicking
        grok_blocklist = [
            "grok", "desktop pilot", "local uncensored", "telemetry", "stop pilot",
            "context: bounded", "localhost:8000", "127.0.0.1:8000", "dlv-"
        ]

        filtered = []
        for el in raw_elements:
            t = el["text"].strip()
            t_low = t.lower()
            if any(kw in t_low for kw in grok_blocklist):
                continue
            if len(t) > 1 or t in ["?", "+", "-", "/", "=", "7", "8", "9", "4", "5", "6", "1", "2", "3", "0"]:
                filtered.append(el)

        # Merge words on the same horizontal line that are close together
        sorted_elements = sorted(filtered, key=lambda e: (e["y"] // 16, e["x"]))
        merged: List[Dict[str, Any]] = []
        
        for el in sorted_elements:
            if not merged:
                merged.append(dict(el))
                continue
            last = merged[-1]
            if abs(last["y"] - el["y"]) < 12 and (el["x"] - (last["x"] + last["w"])) < 35:
                last["text"] += f" {el['text']}"
                last["w"] = (el["x"] + el["w"]) - last["x"]
                last["cx"] = last["x"] + last["w"] // 2
            else:
                merged.append(dict(el))

        # Prioritize center elements (where popups/dialogs/pickers appear) over top-bar clutter
        try:
            bounds = self.controller.get_desktop_bounds()
        except Exception:
            bounds = (0, 0, 1920, 1080)

        prioritized = WaylandController.prioritize_ocr_elements(merged, bounds=bounds, max_elements=max_elements)
        self._current_step_elements = prioritized

        lines = []
        for idx, item in enumerate(prioritized):
            lines.append(f'[{idx}] "{item["text"]}" at coordinates ({item["cx"]}, {item["cy"]})')

        return "\n".join(lines)

    async def execute_workspace_task(self, goal: str, max_steps: int = 25) -> AsyncGenerator[Dict[str, Any], None]:
        """Headless ReAct loop using only workspace + web tools (no GUI control)."""
        self.is_running = True
        self._abort_requested = False
        self.current_task = goal
        self.step_history = []

        yield {
            "step": 0,
            "action": "initialize",
            "detail": f"Starting workspace-only agent for: '{goal}'",
            "screenshot_url": None,
            "status": "running"
        }

        engine_ok = await asyncio.to_thread(self._ensure_engine)
        if not engine_ok:
            self.is_running = False
            if get_active_provider() == PROVIDER_OPENAI:
                reason = "No OpenAI API key configured. Add your ChatGPT key in the sidebar."
            else:
                reason = "Failed to start or connect to the local llama-server."
            yield {"step": 0, "action": "error", "detail": reason, "screenshot_url": None, "status": "error"}
            return

        history_log: List[str] = []
        finished = False
        step = 1
        try:
            while step <= max_steps and not self._abort_requested and self.is_running:
                history_text = "\n".join(history_log[-6:]) if history_log else "None."

                workspace_listing = ""
                if self.workspace_manager:
                    try:
                        files = self.workspace_manager.list_files("")
                        workspace_listing = "\n".join(f"- {f['path']}" for f in files[:60]) or "(empty workspace)"
                    except Exception:
                        workspace_listing = "(unavailable)"

                user_prompt = f"""Goal: {goal}

Workspace files:
{workspace_listing}

Recent Steps Taken:
{history_text}

What is your next action?"""

                agent_messages = [
                    {"role": "system", "content": WORKSPACE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ]
                chatml_prompt = format_chatml_prompt(agent_messages)

                thought = ""
                action_str = ""
                full_response = []
                try:
                    async for chunk in self.llm_client.stream_completion(
                        prompt=chatml_prompt,
                        messages=agent_messages,
                        image_data=None,
                        temperature=0.15,
                        top_p=0.9,
                        max_tokens=600,
                        reasoning_effort="low"
                    ):
                        if "error" in chunk:
                            yield {"step": step, "action": "error", "detail": chunk["error"], "status": "error"}
                            self.is_running = False
                            return
                        tok = chunk.get("token", "")
                        full_response.append(tok)
                except Exception as e:
                    yield {"step": step, "action": "error", "detail": f"Inference failed: {e}", "status": "error"}
                    self.is_running = False
                    return

                thought, action_str = parse_pilot_response("".join(full_response))
                if not action_str:
                    history_log.append(f"Step {step}: no valid action parsed")
                    yield {"step": step, "action": "none", "thought": thought, "detail": "No valid action parsed; retrying.", "status": "running"}
                    step += 1
                    continue

                detail, is_finished = await self._execute_action(action_str, [])
                history_log.append(f"Step {step}: {action_str} -> {detail[:200]}")
                self.step_history.append({"step": step, "action": action_str, "detail": detail})
                yield {
                    "step": step,
                    "action": action_str,
                    "thought": thought,
                    "detail": detail,
                    "screenshot_url": None,
                    "status": "completed" if is_finished else "running"
                }
                if is_finished:
                    finished = True
                    break
                step += 1
        finally:
            self.is_running = False

        if not finished and not self._abort_requested:
            yield {"step": step, "action": "max_steps", "detail": f"Reached max steps ({max_steps}) without finishing.", "status": "max_steps"}

    async def execute_task(self, goal: str, max_steps: int = 15) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Executes a desktop automation task driven by the active inference engine.
        Yields step-by-step progress events with live screenshots.
        """
        self.is_running = True
        self._abort_requested = False
        self.current_task = goal
        self.step_history = []
        self.controller.start_shake_detector(self.abort)

        shots_dir = Path("/tmp/grok_shots")
        shots_dir.mkdir(parents=True, exist_ok=True)

        yield {
            "step": 0,
            "action": "initialize",
            "detail": f"Starting autonomous desktop pilot for: '{goal}'",
            "screenshot_url": None,
            "status": "running"
        }

        # Step 0: Ensure the active LLM engine is online
        engine_label = get_active_engine_label()
        engine_ok = await asyncio.to_thread(self._ensure_engine)
        if not engine_ok:
            self.is_running = False
            if get_active_provider() == PROVIDER_OPENAI:
                reason = "No OpenAI API key configured. Add your ChatGPT key in the sidebar (🔑)."
            else:
                reason = "Failed to start or connect to the local llama-server. Please check GPU/VRAM status."
            yield {
                "step": 0,
                "action": "error",
                "detail": reason,
                "screenshot_url": None,
                "status": "error"
            }
            return

        # Controller selection: Isolated Virtual Display (:99) vs Physical Screen (Mutter Wayland)
        from config import get_pilot_isolated_display
        from engine.virtual_display import VirtualDisplayManager

        session_type = "Wayland session"
        if not getattr(self, "_custom_controller", False) and "Mock" not in type(self.controller).__name__:
            if get_pilot_isolated_display():
                avail, reason = VirtualDisplayManager.is_available()
                if avail:
                    self.controller = VirtualDisplayManager.get_instance()
                    try:
                        self.controller.ensure_running()
                        session_type = "Isolated Virtual Sandbox (:99)"
                    except Exception as ve:
                        self.controller = WaylandController.get_instance()
                        session_type = f"Mutter Wayland (Sandbox init failed: {ve})"
                else:
                    self.controller = WaylandController.get_instance()
                    session_type = f"Mutter Wayland ({reason})"
            else:
                self.controller = WaylandController.get_instance()

        # Initial screen perception
        try:
            bounds = self.controller.get_desktop_bounds()
            initial_shot = self.controller.capture_screen()
            now_ms = int(time.time() * 1000)
            shot_rel_url = f"/api/desktop/shot?path={initial_shot}&t={now_ms}"
            yield {
                "step": 0,
                "action": "perceive_desktop",
                "detail": f"Desktop active ({bounds[2]}x{bounds[3]} {session_type}). {engine_label} cognitive pilot online.",
                "screenshot_url": shot_rel_url,
                "status": "running"
            }
        except Exception as e:
            self.is_running = False
            yield {
                "step": 0,
                "action": "error",
                "detail": f"Failed to initialize desktop capture: {e}",
                "screenshot_url": None,
                "status": "error"
            }
            return

        # Autonomous ReAct Decision Loop
        step = 1
        history_log: List[str] = []
        prev_ocr_set: Optional[set] = None
        prev_shot_path: Optional[str] = None
        prev_action_str: str = ""
        consecutive_stalls: int = 0
        last_stalled_signature: str = ""
        loop_breaker_interrupt: str = ""

        # Determine if active model supports native vision
        has_vision = getattr(self.llm_client, "has_vision", False)
        if not has_vision:
            try:
                engine_info = get_active_engine_info()
                has_vision = bool(engine_info.get("has_vision", False))
            except Exception:
                has_vision = False

        finished = False
        try:
            while step <= max_steps and not self._abort_requested and self.is_running:
                # 1. Capture current live screen state
                now_ms = int(time.time() * 1000)
                shot_path = str(shots_dir / f"shot_{now_ms}.png")
                current_shot = self.controller.capture_screen(out_path=shot_path)
                shot_rel_url = f"/api/desktop/shot?path={current_shot}&t={now_ms}"
                raw_elements = self.controller.get_screen_ocr_elements(current_shot)
                max_ocr = 40 if (get_active_provider() != PROVIDER_OPENAI) else 80
                elements_text = self._format_screen_elements(raw_elements, max_elements=max_ocr)

                # Track normalized OCR text set to measure screen delta
                current_ocr_set = {el["text"].strip().lower() for el in raw_elements if el.get("text", "").strip()}

                # Measure screen delta: warn if last action was an interactive command but screen delta < 2%
                stalled_warning = ""
                is_stalled = False
                if prev_ocr_set is not None and prev_action_str:
                    act_lower = prev_action_str.lower()
                    interactive = any(k in act_lower for k in [
                        "navigate_browser", "click", "click_text", "click_id", "type_into", "type_text", "paste_text", "press_key", "launch_app"
                    ])
                    if interactive:
                        screen_delta = compute_screen_delta(prev_shot_path, current_shot, prev_ocr_set, current_ocr_set)
                        if screen_delta < 0.02:
                            is_stalled = True
                            stalled_warning = (
                                "WARNING: The screen did not change after your last action / has not updated after clicking or typing at those coordinates. "
                                "The app window may lack focus or the exact coordinates did not trigger the field. "
                                "You must use a different coordinate, hit a broader target, scroll, or try an alternative approach.\n\n"
                            )

                # Prepare screenshot data URI for vision models (preserve up to 2048px for OpenAI, 1024px for local)
                image_data = None
                if has_vision and os.path.exists(current_shot):
                    image_data = prepare_screenshot_for_llm(current_shot, max_dim=None, quality=85)

                # 2. Build prompt for the active engine (local model or ChatGPT)
                history_text = "\n".join(history_log[-4:]) if history_log else "None."
                vision_hint = "[Live screenshot attached for visual inspection]\n\n" if (has_vision and image_data) else ""

                # Dynamic browser profiles & user dossier context
                try:
                    profiles_summary = get_available_profiles_summary()
                except Exception:
                    profiles_summary = ""

                dossier_hint = ""
                try:
                    facts = UserDossier.get_all_facts()
                    relevant = [
                        f"- {f['key'].replace('_', ' ').title()}: {f['value']}"
                        for f in facts
                        if any(k in f['key'].lower() for k in ['browser', 'profile', 'name', 'account', 'email', 'login', 'identity'])
                    ]
                    if relevant:
                        dossier_hint = "User Identity & Defaults:\n" + "\n".join(relevant)
                except Exception:
                    dossier_hint = ""

                system_content = SYSTEM_PROMPT
                if profiles_summary:
                    system_content += f"\n\n{profiles_summary}"
                if dossier_hint:
                    system_content += f"\n\n{dossier_hint}"

                interrupt_prefix = loop_breaker_interrupt
                loop_breaker_interrupt = ""  # Reset after consuming for prompt

                user_prompt = f"""Goal: {goal}

Desktop Display Coordinate Space: {bounds[2]}x{bounds[3]} (Native Resolution).
All mouse click/type coordinates MUST be specified in this exact {bounds[2]}x{bounds[3]} coordinate space.

{interrupt_prefix}{stalled_warning}{vision_hint}Current Screen State:
{elements_text}

Recent Steps Taken:
{history_text}

What is your next action?"""

                agent_messages = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_prompt}
                ]
                chatml_prompt = format_chatml_prompt(agent_messages)

                # 3. Model inference with max_tokens=400 to prevent truncated reasoning
                thought = ""
                action_str = ""
                full_response = []
                inference_error: Optional[str] = None

                try:
                    async for chunk in self.llm_client.stream_completion(
                        prompt=chatml_prompt,
                        messages=agent_messages,
                        image_data=image_data,
                        temperature=0.10,
                        top_p=0.85,
                        max_tokens=400,
                        reasoning_effort="low"
                    ):
                        if chunk.get("error"):
                            inference_error = chunk.get("error")
                        token = chunk.get("token", "")
                        if token:
                            full_response.append(token)
                except Exception as inf_err:
                    inference_error = str(inf_err)

                response_text = "".join(full_response).strip()
                thought, action_str = parse_pilot_response(response_text)

                # Multimodal Fallback: if vision inference errored or produced empty action, retry once text-only
                if (inference_error or not action_str) and image_data:
                    logger.warning(
                        "Step %d: Multimodal inference failed or produced empty action (err=%s). Retrying step text-only.",
                        step, inference_error
                    )
                    retry_user_prompt = f"""Goal: {goal}

Desktop Display Coordinate Space: {bounds[2]}x{bounds[3]} (Native Resolution).
All mouse click/type coordinates MUST be specified in this exact {bounds[2]}x{bounds[3]} coordinate space.

{interrupt_prefix}{stalled_warning}Current Screen State:
{elements_text}

Recent Steps Taken:
{history_text}

What is your next action?"""
                    retry_messages = [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": retry_user_prompt}
                    ]
                    retry_chatml = format_chatml_prompt(retry_messages)
                    retry_tokens = []
                    retry_error: Optional[str] = None
                    try:
                        async for chunk in self.llm_client.stream_completion(
                            prompt=retry_chatml,
                            messages=retry_messages,
                            image_data=None,
                            temperature=0.10,
                            top_p=0.85,
                            max_tokens=400,
                            reasoning_effort="low"
                        ):
                            if chunk.get("error"):
                                retry_error = chunk.get("error")
                            tok = chunk.get("token", "")
                            if tok:
                                retry_tokens.append(tok)
                    except Exception as r_err:
                        retry_error = str(r_err)

                    if retry_tokens:
                        response_text = "".join(retry_tokens).strip()
                        thought, action_str = parse_pilot_response(response_text)
                        inference_error = retry_error

                # Fast Failure: If action is still empty after retry, halt immediately instead of looping
                if not action_str or not action_str.strip():
                    err_detail = f"Pilot halted: Model emitted no valid action. Raw output: {response_text!r}"
                    if inference_error:
                        err_detail += f" (Inference error: {inference_error})"
                    logger.error(err_detail)
                    yield {
                        "step": step,
                        "thought": thought or "Failed to choose an action from screen state.",
                        "action": "error",
                        "detail": err_detail,
                        "screenshot_url": shot_rel_url,
                        "status": "error"
                    }
                    self.is_running = False
                    break

                # Loop Breaker Check: if the same coordinate/action is targeted 2 times consecutively without state modification
                curr_sig = get_action_signature(action_str)
                prev_sig = get_action_signature(prev_action_str) if prev_action_str else ""
                is_repeat = (curr_sig == prev_sig) or bool(last_stalled_signature and curr_sig == last_stalled_signature)

                if is_stalled and is_repeat:
                    consecutive_stalls += 1
                elif is_stalled:
                    consecutive_stalls = 1
                else:
                    consecutive_stalls = 0

                last_stalled_signature = curr_sig if is_stalled else ""

                if consecutive_stalls >= 2:
                    logger.warning(
                        "Step %d: Loop breaker triggered on signature %s (stalls=%d). Rejecting duplicate action and forcing pivot.",
                        step, curr_sig, consecutive_stalls
                    )
                    loop_breaker_interrupt = (
                        f"LOOP BREAKER INTERRUPT: You targeted the exact same coordinates or action ('{action_str}') 2 times consecutively without any screen update. "
                        "You are FORBIDDEN from repeating this action. Pivot now: use keyboard shortcuts (press_key('tab') or press_key('enter')), focus the window, use type_into with submit=True, or click a visible text element.\n\n"
                    )
                    # Attempt active window focus recovery
                    try:
                        self.controller.focus_window()
                    except Exception:
                        pass

                    detail_msg = f"Rejected duplicate action '{action_str}': Loop breaker triggered. Pivoting agent strategy."
                    history_log.append(f"Step {step}: {action_str} -> {detail_msg}")
                    yield {
                        "step": step,
                        "thought": f"Stuck click/focus loop detected on '{action_str}'. Intercepting and pivoting strategy.",
                        "action": "loop_breaker_pivot",
                        "detail": detail_msg,
                        "screenshot_url": shot_rel_url,
                        "status": "running"
                    }
                    prev_ocr_set = current_ocr_set
                    prev_shot_path = current_shot
                    prev_action_str = action_str
                    step += 1
                    await asyncio.sleep(0.5)
                    continue

                # 4. Execute the parsed tool action(s)
                detail_msg, is_finished = await self._execute_action(action_str, raw_elements)

                # Record state for delta comparison in next step
                prev_ocr_set = current_ocr_set
                prev_shot_path = current_shot
                prev_action_str = action_str

                history_log.append(f"Step {step}: {action_str} -> {detail_msg}")

                status = "completed" if is_finished else "running"
                yield {
                    "step": step,
                    "thought": thought,
                    "action": action_str,
                    "detail": f"{thought}\n→ **{action_str}**: {detail_msg}",
                    "screenshot_url": shot_rel_url,
                    "status": status
                }

                if is_finished:
                    finished = True
                if is_finished or self._abort_requested:
                    break

                step += 1
                await asyncio.sleep(0.5)

            if self._abort_requested:
                yield {
                    "step": step,
                    "action": "abort",
                    "detail": "Desktop pilot halted by emergency stop.",
                    "screenshot_url": None,
                    "status": "aborted"
                }
            elif not finished and step > max_steps:
                yield {
                    "step": step,
                    "action": "max_steps",
                    "detail": f"Reached maximum steps ({max_steps}) without explicit finish().",
                    "screenshot_url": None,
                    "status": "max_steps"
                }
        finally:
            self.controller.stop_shake_detector()
            self.is_running = False

    async def _execute_action(self, action_str: str, raw_elements: List[Dict[str, Any]]) -> Tuple[str, bool]:
        """Parses and executes tool call(s) using WaylandController."""
        action_clean = action_str.strip()

        # Handle chained actions separated by 'then' or '&&' outside quotes
        sub_actions = split_chained_actions(action_clean)
        if len(sub_actions) > 1:
            details = []
            is_done = False
            for sub_a in sub_actions:
                sub_detail, sub_done = await self._execute_single_action(sub_a, raw_elements)
                details.append(sub_detail)
                if sub_done:
                    is_done = True
                    break
                await asyncio.sleep(0.3)
            return " ; ".join(details), is_done

        return await self._execute_single_action(action_clean, raw_elements)

    async def _execute_single_action(self, action_clean: str, raw_elements: List[Dict[str, Any]]) -> Tuple[str, bool]:
        """Executes a single atomic tool call."""
        # finish(summary)
        m = re.match(r"finish\([\"']?(.*?)[\"']?\)", action_clean, re.IGNORECASE)
        if m:
            summary = m.group(1).strip() or "Goal achieved."
            return f"Task completed: {summary}", True

        # web_search("query") - headless internet search (no browser/GUI needed)
        m = re.match(r"web_search\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE | re.DOTALL)
        if m:
            from engine.web_tools import web_search
            query = m.group(1).strip()
            res = await web_search(query)
            if res.get("success"):
                lines = [f"- {r['title']} ({r['url']})" + (f": {r['snippet'][:120]}" if r.get("snippet") else "") for r in res["results"]]
                self._last_web_context = "\n".join(lines)
                return f"Web search '{query}' returned {len(res['results'])} results:\n" + "\n".join(lines), False
            return f"Web search '{query}' failed: {res.get('error')}", False

        # web_fetch("url") - fetch and extract page text
        m = re.match(r"web_fetch\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE | re.DOTALL)
        if m:
            from engine.web_tools import web_fetch
            url = m.group(1).strip()
            res = await web_fetch(url)
            if res.get("success"):
                snippet = res.get("content", "")[:2500]
                self._last_web_context = snippet
                return f"Fetched '{res.get('url')}' ({res.get('length')} chars). Extracted content:\n{snippet}", False
            return f"Web fetch '{url}' failed: {res.get('error')}", False

        # web_research("query", fetch_top=N) - search + fetch top pages
        m = re.match(r"web_research\((.*)\)", action_clean, re.IGNORECASE | re.DOTALL)
        if m:
            from engine.web_tools import web_research
            args_str = m.group(1).strip()
            query = ""
            fetch_top = 2
            qm = re.match(r"[\"'](.*?)[\"']", args_str)
            if qm:
                query = qm.group(1).strip()
                rem = args_str[qm.end():].strip()
                fm = re.search(r"fetch_top\s*=\s*(\d+)", rem)
                if fm:
                    fetch_top = int(fm.group(1))
            if query:
                res = await web_research(query, fetch_top=fetch_top)
                if res.get("success"):
                    parts = [f"Research '{query}' ({len(res.get('results', []))} results):"]
                    for r in res.get("results", []):
                        parts.append(f"- {r['title']} ({r['url']})")
                    for p in res.get("pages", []):
                        parts.append(f"\n### {p.get('title')} ({p.get('url')})\n{p.get('content', '')[:1500]}")
                    text = "\n".join(parts)
                    self._last_web_context = text
                    return text, False
                return f"Web research '{query}' failed: {res.get('error')}", False
            return "web_research requires a query string.", False

        # save_web_result("rel_path") - persist last web context into workspace
        m = re.match(r"save_web_result\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE | re.DOTALL)
        if m:
            rel_path = m.group(1).strip()
            if self.workspace_manager and getattr(self, "_last_web_context", ""):
                try:
                    self.workspace_manager.write_file(rel_path, self._last_web_context)
                    return f"Saved web research to '{rel_path}'.", False
                except Exception as e:
                    return f"Failed to save web result: {e}", False
            return "No web context available to save.", False

        # open_url(url_or_query, browser=..., profile=...)
        m = re.match(r"open_url\((.*)\)", action_clean, re.IGNORECASE | re.DOTALL)
        if m:
            args_str = m.group(1).strip()
            target = ""
            browser = None
            profile = None

            target_match = re.match(r"[\"'](.*?)[\"']", args_str)
            if target_match:
                target = target_match.group(1).strip()
                rem = args_str[target_match.end():].strip()

                prof_kw = re.search(r"profile\s*=\s*[\"'](.*?)[\"']", rem, re.IGNORECASE)
                if prof_kw:
                    profile = prof_kw.group(1).strip()
                brow_kw = re.search(r"browser\s*=\s*[\"'](.*?)[\"']", rem, re.IGNORECASE)
                if brow_kw:
                    browser = brow_kw.group(1).strip()

                if not profile and not browser:
                    pos_matches = re.findall(r"[\"'](.*?)[\"']", rem)
                    if len(pos_matches) == 1:
                        if pos_matches[0].lower() in ["brave", "chrome", "google-chrome", "chromium"]:
                            browser = pos_matches[0]
                        else:
                            profile = pos_matches[0]
                    elif len(pos_matches) >= 2:
                        browser = pos_matches[0]
                        profile = pos_matches[1]

            if target:
                if browser or profile:
                    self.controller.open_url(target, browser=browser, profile=profile)
                else:
                    self.controller.open_url(target)
                await asyncio.sleep(2.0)
                detail = f"Opened '{target}' in browser"
                if browser or profile:
                    detail += f" (browser={browser or 'auto'}, profile={profile or 'active'})"
                return f"{detail}.", False

        # navigate_browser(url_or_query)
        m = re.match(r"navigate_browser\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE)
        if m:
            query = m.group(1).strip()
            self.controller.press_key("ctrl+l")
            await asyncio.sleep(0.3)
            self.controller.type_text(query)
            await asyncio.sleep(0.3)
            self.controller.press_key("enter")
            await asyncio.sleep(2.0)
            return (
                f"Sent Ctrl+L, typed '{query}', and pressed Enter "
                "(verify on the next screen state that the browser actually navigated; "
                "if a profile picker or popup blocked it, dismiss/select it first).",
                False
            )

        # launch_app(app_name, profile=...)
        m = re.match(r"launch_app\((.*)\)", action_clean, re.IGNORECASE | re.DOTALL)
        if m:
            args_str = m.group(1).strip()
            app = ""
            profile = None

            app_match = re.match(r"[\"'](.*?)[\"']", args_str)
            if app_match:
                app = app_match.group(1).strip().lower()
                rem = args_str[app_match.end():].strip()

                prof_kw = re.search(r"profile\s*=\s*[\"'](.*?)[\"']", rem, re.IGNORECASE)
                if prof_kw:
                    profile = prof_kw.group(1).strip()
                else:
                    pos = re.findall(r"[\"'](.*?)[\"']", rem)
                    if pos:
                        profile = pos[0].strip()

            if app:
                if profile:
                    self.controller.launch_app(app, wait_seconds=2.5, profile=profile)
                else:
                    self.controller.launch_app(app, wait_seconds=2.5)
                await asyncio.sleep(0.5)
                detail = f"Launched application '{app}'"
                if profile:
                    detail += f" with profile '{profile}'"
                return f"{detail}.", False

        # click(x, y)
        m = re.match(r"click\((\d+),\s*(\d+)\)", action_clean, re.IGNORECASE)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            self.controller.click(x, y)
            await asyncio.sleep(0.5)
            return f"Moved cursor to ({x}, {y}) and clicked.", False

        # double_click(x, y)
        m = re.match(r"double_click\((\d+),\s*(\d+)\)", action_clean, re.IGNORECASE)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            self.controller.click(x, y, button="left", count=2)
            await asyncio.sleep(0.5)
            return f"Double-clicked at ({x}, {y}).", False

        # right_click(x, y)
        m = re.match(r"right_click\((\d+),\s*(\d+)\)", action_clean, re.IGNORECASE)
        if m:
            x, y = int(m.group(1)), int(m.group(2))
            self.controller.click(x, y, button="right", count=1)
            await asyncio.sleep(0.5)
            return f"Right-clicked at ({x}, {y}).", False

        # click_id(id)
        m = re.match(r"click_id\((\d+)\)", action_clean, re.IGNORECASE)
        if m:
            idx = int(m.group(1))
            elements = getattr(self, "_current_step_elements", None) or raw_elements
            if 0 <= idx < len(elements):
                target_el = elements[idx]
                cx, cy = target_el.get("cx", target_el.get("x", 0)), target_el.get("cy", target_el.get("y", 0))
                self.controller.click(cx, cy)
                await asyncio.sleep(0.5)
                return f"Clicked element [{idx}] '{target_el.get('text', '')}' at ({cx}, {cy}).", False
            return f"Element ID [{idx}] out of range (total elements: {len(elements)}).", False

        # click_text(text)
        m = re.match(r"click_text\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE)
        if m:
            target_text = m.group(1).strip().lower()
            target_words = set(target_text.split())
            elements = getattr(self, "_current_step_elements", None) or raw_elements
            best_el = None
            best_score = 0.0

            for el in elements:
                t = el.get("text", "").strip().lower()
                if not t:
                    continue
                # 1. Exact match gets highest score
                if target_text == t:
                    score = 1000.0
                # 2. Target text is entirely contained within the element phrase
                elif target_text in t:
                    score = 500.0 + (len(target_text) / len(t)) * 100.0
                # 3. Element text is contained in target text (only if substantial word match)
                elif t in target_text and len(t) >= 4:
                    if any(w == t for w in target_words):
                        score = 300.0 + (len(t) / len(target_text)) * 100.0
                    else:
                        score = 50.0 + (len(t) / len(target_text)) * 50.0
                else:
                    el_words = set(t.split())
                    overlap = target_words.intersection(el_words)
                    if overlap:
                        score = (len(overlap) / max(len(target_words), len(el_words))) * 200.0
                    else:
                        score = 0.0

                if score > best_score:
                    best_score = score
                    best_el = el

            if best_el and best_score >= 60.0:
                cx = best_el.get("cx", best_el.get("x", 0) + best_el.get("w", 0) // 2)
                cy = best_el.get("cy", best_el.get("y", 0) + best_el.get("h", 0) // 2)
                self.controller.click(cx, cy)
                await asyncio.sleep(0.5)
                return f"Found '{best_el.get('text', '')}' at ({cx}, {cy}). Moved cursor and clicked.", False

            cand_msg = f" (closest was '{best_el.get('text', '')}' with low score {best_score:.0f})" if best_el else ""
            return f"Element '{target_text}' not reliably found on screen{cand_msg}. Refused to click random coordinates.", False

        # type_into(...)
        if action_clean.lower().startswith("type_into("):
            m_all = re.match(r"type_into\s*\((.*)\)", action_clean, re.IGNORECASE | re.DOTALL)
            if m_all:
                parsed = parse_type_into_args(m_all.group(1))
                if parsed:
                    p_type = parsed["type"]
                    submit = parsed["submit"]
                    text = parsed["text"]
                    sub_str = " and pressed Enter" if submit else ""
                    if p_type == "xy":
                        x, y = parsed["x"], parsed["y"]
                        self.controller.type_into(x, y, text, submit=submit)
                        await asyncio.sleep(0.5)
                        return f"Focused ({x}, {y}), typed '{text}'{sub_str}.", False
                    elif p_type == "id":
                        idx = parsed["id"]
                        elements = getattr(self, "_current_step_elements", None) or raw_elements
                        if 0 <= idx < len(elements):
                            target_el = elements[idx]
                            cx, cy = target_el.get("cx", target_el.get("x", 0)), target_el.get("cy", target_el.get("y", 0))
                            self.controller.type_into(cx, cy, text, submit=submit)
                            await asyncio.sleep(0.5)
                            return f"Focused element [{idx}] '{target_el.get('text', '')}' at ({cx}, {cy}), typed '{text}'{sub_str}.", False
                        else:
                            self.controller.type_text(text)
                            if submit:
                                await asyncio.sleep(0.2)
                                self.controller.press_key("enter")
                            return f"Typed '{text}'{sub_str} (element [{idx}] out of range).", False
                    elif p_type == "text":
                        self.controller.type_text(text)
                        if submit:
                            await asyncio.sleep(0.2)
                            self.controller.press_key("enter")
                        return f"Typed '{text}'{sub_str}.", False

        # type_text(text)
        m = re.match(r"type_text\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE)
        if m:
            text = m.group(1)
            self.controller.type_text(text)
            await asyncio.sleep(0.3)
            return f"Typed '{text}'.", False

        # paste_text(text)
        m = re.match(r"paste_text\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE)
        if m:
            text = m.group(1)
            self.controller.paste_text(text)
            await asyncio.sleep(0.3)
            return f"Pasted '{text}' via clipboard.", False

        # copy_selection()
        if re.match(r"copy_selection\(\)", action_clean, re.IGNORECASE):
            self.controller.press_key("ctrl+c")
            await asyncio.sleep(0.3)
            copied = self.controller.read_clipboard()
            snippet = copied[:120] if copied else "(clipboard empty)"
            return f"Copied selection to clipboard: '{snippet}'", False

        # press_key(key)
        m = re.match(r"press_key\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE)
        if m:
            key = m.group(1).strip()
            self.controller.press_key(key)
            await asyncio.sleep(0.4)
            return f"Pressed key '{key}'.", False

        # scroll("up" | "down")
        m = re.match(r"scroll\([\"']?(up|down)[\"']?\)", action_clean, re.IGNORECASE)
        if m:
            direction = m.group(1).lower()
            key = "pageup" if direction == "up" else "pagedown"
            self.controller.press_key(key)
            await asyncio.sleep(0.4)
            return f"Scrolled {direction} (pressed {key}).", False

        # wait(seconds)
        m = re.match(r"wait\((\d+(?:\.\d+)?)\)", action_clean, re.IGNORECASE)
        if m:
            sec = float(m.group(1))
            await asyncio.sleep(min(sec, 5.0))
            return f"Waited {sec}s.", False

        # upload_file_to_dialog(file_path)
        m = re.match(r"upload_file_to_dialog\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE)
        if m:
            fpath = m.group(1).strip()
            if self.workspace_manager and not os.path.isabs(fpath):
                fpath = str(self.workspace_manager._resolve_safe_path(fpath))
            res = self.controller.upload_file_to_dialog(fpath)
            return f"Uploaded '{fpath}' to file dialog (success={res}).", False

        # wait_for_condition(desc, max_wait=45)
        m = re.match(r"wait_for_condition\([\"'](.*?)[\"'](?:,\s*(\d+))?\)", action_clean, re.IGNORECASE)
        if m:
            desc = m.group(1).strip()
            max_w = int(m.group(2)) if m.group(2) else 45
            self.controller.wait_for_condition(desc, max_wait_sec=max_w)
            return f"Waited for condition '{desc}' (up to {max_w}s).", False

        # save_latest_download(target_path)
        m = re.match(r"save_latest_download\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE)
        if m:
            target_p = m.group(1).strip()
            if self.workspace_manager:
                dest = self.workspace_manager.save_latest_download(target_p)
            else:
                dest = self.controller.save_latest_download(target_p)
            if dest:
                return f"Saved newest download to '{dest}'.", False
            return f"No recent download found to save as '{target_p}'.", False

        # write_file(rel_path, content)
        m = re.match(r"write_file\([\"'](.*?)[\"'],\s*[\"'](.*?)[\"']\)", action_clean, re.IGNORECASE | re.DOTALL)
        if m:
            rpath, content = m.group(1).strip(), m.group(2)
            if self.workspace_manager:
                res = self.workspace_manager.write_file(rpath, content)
                return res, False
            return "Error: write_file requires an active workspace manager.", False

        # edit_file(rel_path, old_str, new_str)
        m = re.match(r"edit_file\([\"'](.*?)[\"'],\s*[\"'](.*?)[\"'],\s*[\"'](.*?)[\"']\)", action_clean, re.IGNORECASE | re.DOTALL)
        if m:
            rpath, old_s, new_s = m.group(1).strip(), m.group(2), m.group(3)
            if self.workspace_manager:
                res = self.workspace_manager.edit_file(rpath, old_s, new_s)
                return res, False
            return "Error: edit_file requires an active workspace manager.", False

        # read_file(rel_path)
        m = re.match(r"read_file\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE)
        if m:
            rpath = m.group(1).strip()
            if self.workspace_manager:
                content = self.workspace_manager.read_file(rpath)
                return f"Read '{rpath}':\n{content}", False
            return "Error: read_file requires an active workspace manager.", False

        # run_shell(command)
        m = re.match(r"run_shell\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE | re.DOTALL)
        if m:
            cmd = m.group(1).strip()
            if self.workspace_manager:
                res = self.workspace_manager.run_shell(cmd)
                out = res["stdout"] or res["stderr"] or f"(exit {res['exit_code']})"
                return f"Executed `{cmd}` (code {res['exit_code']}):\n{out[:400]}", False
            return "Error: run_shell requires an active workspace manager.", False

        # Fallback keyword checks
        if "ctrl+l" in action_clean.lower():
            self.controller.press_key("ctrl+l")
            return "Pressed 'ctrl+l' to focus address bar.", False
        elif "enter" in action_clean.lower():
            self.controller.press_key("enter")
            return "Pressed 'enter'.", False

        return f"Unknown action: {action_clean}", False
