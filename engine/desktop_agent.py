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
from persona.chatml import format_chatml_prompt
from config import (
    get_active_provider,
    PROVIDER_OPENAI,
    get_openai_api_key,
    get_active_engine_label,
    get_active_engine_info,
)

def prepare_screenshot_for_llm(shot_path: str, max_dim: int = 1024, quality: int = 80) -> Optional[str]:
    """
    Downscales screenshot to a maximum dimension (1024px) and compresses to JPEG
    to keep vision token count within llama-server context limits while preserving clarity.
    """
    if not shot_path or not os.path.exists(shot_path):
        return None
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


KNOWN_ACTIONS_REGEX = re.compile(
    r"\b(launch_app|open_url|navigate_browser|click_id|click_text|click|double_click|right_click|type_into|type_text|paste_text|copy_selection|press_key|scroll|wait|finish)\s*\([^)]*\)",
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
1. open_url("url_or_query") - Directly opens browser to any URL or Google search query in 1 fast step (e.g. open_url("https://github.com") or open_url("weather in tokyo")). PREFERRED for web browsing!
2. launch_app("app_name") - Launch an application (e.g. "brave", "google-chrome", "nautilus", "gnome-terminal", "gnome-calculator", "code", "spotify").
3. navigate_browser("url_or_query") - In an already-open browser window, automatically focuses the address bar (Ctrl+L), types the URL/query, and presses Enter.
4. click(x, y) - Move the mouse cursor to absolute coordinate (x, y) and left-click.
5. double_click(x, y) - Move to (x, y) and double-click.
6. right_click(x, y) - Move to (x, y) and right-click (open context menu).
7. click_text("text") - Search for matching text on screen and click it.
8. click_id(id) - Click element by its index ID from the screen state.
9. type_text("string") - Type text using keyboard into the currently focused window or field.
10. paste_text("string") - Paste text directly into the focused field via clipboard (faster for long strings).
11. copy_selection() - Copy current selection (Ctrl+C) and inspect clipboard text.
12. press_key("key_combo") - Press a key or shortcut (e.g. "enter", "ctrl+l", "ctrl+t", "ctrl+w", "alt+tab", "super", "backspace", "escape").
13. scroll("up" | "down") - Scroll page up or down.
14. wait(seconds) - Pause for seconds to let a page or app load (e.g. wait(2)).
15. finish("summary") - The goal has been successfully accomplished.

CRITICAL OPERATIONAL RULES:
1. APP LAUNCHING RULE:
   If the user's goal specifies an application (e.g. Brave, Chrome, Files, Calculator, Terminal) and that app is NOT already open and active on screen, your VERY FIRST ACTION MUST BE:
   Action: launch_app("<app_name>")
   NEVER click random desktop icons, taskbars, or type text before the requested application is opened!

2. INTERMEDIATE BLOCKERS & PROFILE SELECTORS:
   If an unexpected modal, "Who's using Brave?", "Who's using Chrome?" profile picker, cookie banner, or dialog appears after launching an app, you MUST interact with it first before attempting navigate_browser again. Click your profile name (e.g. click_text("Person 1") or click(x, y) on the avatar) or use press_key("enter") to enter the browser.

3. BROWSER NAVIGATION RULE (Brave, Chrome, Firefox):
   Once the browser window is open and unobstructed, to search or navigate, ALWAYS prefer:
   Action: navigate_browser("<query_or_url>")
   Alternatively: press_key("ctrl+l") to focus address bar, then type_text("<query>"), then press_key("enter").

4. NEVER TRUST ACTION HISTORY ALONE & VERIFICATION RULE:
   Do NOT call finish() just because a previous step says navigate_browser or click was executed. You MUST verify in the Current Screen State (and screenshot) that the final goal state is actually visible on screen (e.g. actual search results for the query or target page content are visible). If a modal blocked your action or the page is still loading, wait or resolve the blocker.

5. DESKTOP APPS RULE:
   For Calculator, Files, or Settings: look at the exact labels in "Current Screen State" and click the target with click_text("<label>") or click(x, y).

6. COMPLETION RULE:
   Only after visually/OCR verifying in Current Screen State that the goal is visibly accomplished, call finish("<summary of what was achieved>").

7. NEVER CLICK OR INTERACT WITH GROK CHAT WINDOW:
   Under NO circumstances should you click or interact with the GROK web application interface itself (identifiable by "GROK // Local Uncensored", "localhost:8000", "127.0.0.1:8000", or GROK control panels like "Stop Pilot"). If GROK is visible on screen, do NOT type in its address bar or click its buttons. Always call launch_app("<target_app>") to open and focus your destination application.

8. DIRECT ACTION FORMAT RULE (NO THINKING TAGS):
   Do NOT output <think>...</think> tags or conversational preamble. Immediately output:
   Thought: <1-2 sentences reasoning about what you see on screen and choosing the next step>
   Action: <single tool call, e.g. launch_app("brave") or click(450, 80) or finish("Done")>

Respond STRICTLY in this format:
Thought: <1-2 sentences reasoning about what you see on screen and choosing the next step>
Action: <single tool call, e.g. launch_app("brave") or navigate_browser("cats") or click(450, 80) or finish("Done")>"""


class DesktopAgent:
    def __init__(self, llm_client=None, llama_client=None, controller=None):
        self._custom_controller = controller is not None
        self.controller = controller or WaylandController.get_instance()
        # Client may be a LlamaClient, an OpenAIClient, or a ProviderRouter
        self.llm_client = llm_client or llama_client or LlamaClient()
        self.is_running = False
        self._abort_requested = False
        self.current_task = ""
        self.step_history: List[Dict[str, Any]] = []

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
        prev_action_str: str = ""

        # Determine if active model supports native vision
        has_vision = getattr(self.llm_client, "has_vision", False)
        if not has_vision:
            try:
                engine_info = get_active_engine_info()
                has_vision = bool(engine_info.get("has_vision", False))
            except Exception:
                has_vision = False

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

                # Measure screen delta: warn if last action was an interactive command but screen didn't change (>90% identical)
                stalled_warning = ""
                if prev_ocr_set is not None and prev_action_str:
                    act_lower = prev_action_str.lower()
                    interactive = any(k in act_lower for k in [
                        "navigate_browser", "click", "click_text", "type_text", "press_key", "launch_app"
                    ])
                    if interactive and (current_ocr_set or prev_ocr_set):
                        overlap = len(prev_ocr_set & current_ocr_set) / max(len(prev_ocr_set | current_ocr_set), 1)
                        if overlap > 0.90:
                            stalled_warning = (
                                "WARNING: The screen did not change after your last action. "
                                "Your previous action likely had no effect because a modal, "
                                "profile picker, or unfocused window is blocking it.\n\n"
                            )

                # Prepare screenshot data URI for vision models (downscaled to max 1024px JPEG)
                image_data = None
                if has_vision and os.path.exists(current_shot):
                    image_data = prepare_screenshot_for_llm(current_shot, max_dim=1024, quality=80)

                # 2. Build prompt for the active engine (local model or ChatGPT)
                history_text = "\n".join(history_log[-4:]) if history_log else "None."
                vision_hint = "[Live screenshot attached for visual inspection]\n\n" if (has_vision and image_data) else ""
                user_prompt = f"""Goal: {goal}

{stalled_warning}{vision_hint}Current Screen State:
{elements_text}

Recent Steps Taken:
{history_text}

What is your next action?"""

                agent_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
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

{stalled_warning}Current Screen State:
{elements_text}

Recent Steps Taken:
{history_text}

What is your next action?"""
                    retry_messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
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

                # 4. Execute the parsed tool action(s)
                detail_msg, is_finished = await self._execute_action(action_str, raw_elements)

                # Record state for delta comparison in next step
                prev_ocr_set = current_ocr_set
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
        finally:
            self.controller.stop_shake_detector()
            self.is_running = False

    async def _execute_action(self, action_str: str, raw_elements: List[Dict[str, Any]]) -> Tuple[str, bool]:
        """Parses and executes tool call(s) using WaylandController."""
        action_clean = action_str.strip()

        # Handle chained actions separated by 'then' or '&&'
        if " then " in action_clean or " && " in action_clean:
            sub_actions = [a.strip() for a in re.split(r"\s+then\s+|\s*&&\s*", action_clean) if a.strip()]
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

        # open_url(url_or_query)
        m = re.match(r"open_url\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE)
        if m:
            target = m.group(1).strip()
            self.controller.open_url(target)
            await asyncio.sleep(2.0)
            return f"Opened '{target}' directly in browser.", False

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

        # launch_app(app_name)
        m = re.match(r"launch_app\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE)
        if m:
            app = m.group(1).strip().lower()
            if app in ["brave", "brave-browser"]:
                cmd = "brave-browser --new-window --profile-directory='Default' || /snap/bin/brave --new-window --profile-directory='Default' || brave --new-window --profile-directory='Default'"
            elif app in ["chrome", "google-chrome"]:
                cmd = "google-chrome --new-window --profile-directory='Default'"
            elif app in ["files", "nautilus"]:
                cmd = "nautilus"
            elif app in ["terminal", "gnome-terminal"]:
                cmd = "gnome-terminal"
            elif app in ["calculator", "gnome-calculator"]:
                cmd = "gnome-calculator"
            elif app in ["code", "vscode"]:
                cmd = "code"
            elif app in ["settings"]:
                cmd = "gnome-control-center"
            else:
                cmd = app

            self.controller.launch_app(cmd, wait_seconds=2.5)
            await asyncio.sleep(0.5)
            return f"Launched application '{app}'.", False

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
            target_words = set(re.findall(r"\w+", target_text))
            best_el = None
            best_score = 0.0

            for el in raw_elements:
                t = el["text"].strip().lower()
                if not t:
                    continue
                # 1. Exact match gets highest score
                if target_text == t:
                    score = 100.0 + len(t)
                # 2. Target text is entirely contained within the element
                elif target_text in t:
                    score = 50.0 + (len(target_text) / len(t)) * 10
                # 3. Element text is entirely contained in target (only if substantial length >= 3)
                elif len(t) >= 3 and t in target_text:
                    score = 20.0 + (len(t) / len(target_text)) * 10
                else:
                    # 4. Word-level overlap
                    el_words = set(re.findall(r"\w+", t))
                    overlap = target_words.intersection(el_words)
                    if overlap:
                        score = len(overlap) * 5.0
                    else:
                        score = 0.0

                if score > best_score:
                    best_score = score
                    best_el = el

            if best_el and best_score >= 10.0:
                cx, cy = best_el["cx"], best_el["cy"]
                self.controller.click(cx, cy)
                await asyncio.sleep(0.5)
                return f"Found '{best_el['text']}' at ({cx}, {cy}). Moved cursor and clicked.", False
            return f"Element '{target_text}' not found on screen.", False

        # type_into(x, y, "text") or type_into(id, "text") or type_into("text")
        m_xy = re.match(r"type_into\((\d+),\s*(\d+),\s*[\"'](.*?)[\"']\)", action_clean, re.IGNORECASE)
        if m_xy:
            x, y, text = int(m_xy.group(1)), int(m_xy.group(2)), m_xy.group(3)
            self.controller.click(x, y)
            await asyncio.sleep(0.2)
            self.controller.type_text(text)
            await asyncio.sleep(0.3)
            return f"Clicked ({x}, {y}) and typed '{text}'.", False

        m_id = re.match(r"type_into\((\d+),\s*[\"'](.*?)[\"']\)", action_clean, re.IGNORECASE)
        if m_id:
            idx, text = int(m_id.group(1)), m_id.group(2)
            elements = getattr(self, "_current_step_elements", None) or raw_elements
            if 0 <= idx < len(elements):
                target_el = elements[idx]
                cx, cy = target_el.get("cx", target_el.get("x", 0)), target_el.get("cy", target_el.get("y", 0))
                self.controller.click(cx, cy)
                await asyncio.sleep(0.2)
            self.controller.type_text(text)
            await asyncio.sleep(0.3)
            return f"Typed '{text}'.", False

        m_text = re.match(r"type_into\([\"'](.*?)[\"']\)", action_clean, re.IGNORECASE)
        if m_text:
            text = m_text.group(1)
            self.controller.type_text(text)
            await asyncio.sleep(0.3)
            return f"Typed '{text}'.", False

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

        # Fallback keyword checks
        if "ctrl+l" in action_clean.lower():
            self.controller.press_key("ctrl+l")
            return "Pressed 'ctrl+l' to focus address bar.", False
        elif "enter" in action_clean.lower():
            self.controller.press_key("enter")
            return "Pressed 'enter'.", False

        return f"Unknown action: {action_clean}", False
