"""VoiceSession — always-listening, full-duplex, voice approvals in human language.

RTX 4050 6GB: faster-whisper small (0.5GB) + Kokoro (~2GB) or piper (CPU)
fits comfortably. Full duplex is done with two logical tasks:
  listener thread — continuous VAD → STT → queue
  speaker thread — takes LLM sentences → TTS, but can be interrupted (barge-in)

Human language approval: the guard calls VoiceApprover which speaks the
request and listens for natural "yes / yeah go ahead / approve" vs "no / cancel".
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from typing import Optional

from rich.console import Console

from .stt import STTProvider, Transcription
from .tts import TTSProvider
from .vad import VADDetector

logger = logging.getLogger("arc.voice.session")

# Natural language approval patterns — human language, not just "yes"
AFFIRMATIVE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|okay|ok|approve|approved|allow|go ahead|do it|proceed|confirm|yea|yess|aye|affirmative|please do)\b",
    re.I,
)
NEGATIVE = re.compile(
    r"\b(no|nope|nah|cancel|stop|abort|deny|denied|don't|dont|never mind|nevermind|wait|hold on|negative)\b",
    re.I,
)
EXIT_PHRASES = re.compile(r"\b(exit|quit|goodbye|bye bye|see you|stop listening|arc stop)\b", re.I)


def is_exit_phrase(text: str) -> bool:
    return bool(EXIT_PHRASES.search(text or ""))


def is_echo(transcript: str, last_tts: str, max_age: float = 3.5, last_time: float = 0.0) -> bool:
    """Check if transcript is echo of recent TTS (to avoid self-trigger).

    Always-listening picks up the speaker's own TTS. We compare normalized
    lower-case and use SequenceMatcher for fuzzy match. If transcript is
    very similar to last TTS and within 3.5s, treat as echo.
    Also catches substring echo: if transcript is short and appears inside TTS.
    """
    if not transcript or not last_tts:
        return False
    if last_time and (time.time() - last_time) > max_age:
        return False
    try:
        import difflib

        # Normalize both
        a = re.sub(r"[^\w\s]", "", transcript.lower().strip())
        b = re.sub(r"[^\w\s]", "", last_tts.lower().strip())
        if not a or not b:
            return False
        # Direct substring — transcript inside TTS is definitely echo (e.g. "I can hear me" inside "I can hear you loud and clear")
        if len(a) >= 4 and (a in b or b in a):
            return True
        # Short transcript that is subset of TTS words
        a_words = set(a.split())
        b_words = set(b.split())
        if a_words and b_words:
            # If all words of transcript are in TTS, it's echo (e.g. "I can hear me" words subset of TTS)
            if a_words.issubset(b_words) and len(a_words) >= 2:
                return True
        # Fuzzy ratio first — very similar strings are echo (lowered threshold)
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        if ratio > 0.70:
            return True
        # Word overlap — be sensitive for short transcripts but not substring
        if a_words and b_words:
            overlap_transcript = len(a_words & b_words) / len(a_words)
            if overlap_transcript >= 0.55:
                return True
            if len(a_words) <= 2 and len(a_words & b_words) >= 1:
                shared = a_words & b_words
                if any(len(w) > 3 for w in shared):
                    return True
        return ratio > 0.60
    except Exception:
        return False


def parse_approval(text: str) -> Optional[bool]:
    """Natural language → True/False/None (None = unclear)."""
    if not text:
        return None
    # Normalize apostrophes so "don't" → "dont"
    t = re.sub(r"['’]", "", text.lower().strip())
    neg_m = NEGATIVE.search(t)
    aff_m = AFFIRMATIVE.search(t)
    if neg_m and aff_m:
        # Both present — e.g. "dont do it" (dont=neg, do it=aff) should be negative
        # Use first occurrence to decide, which handles "dont do it" → negative
        # and "yes, no problem" → affirmative (yes before no)
        if neg_m.start() < aff_m.start():
            return False
        elif aff_m.start() < neg_m.start():
            return True
        return None
    if neg_m:
        return False
    if aff_m:
        return True
    return None


class VoiceApprover:
    """Human-language approver for YELLOW/RED actions.

    Called by PermissionGuard via `app.set_approver`. Speaks the request,
    listens for natural language, falls back to text Confirm.ask on low
    confidence or unclear.

    Supports task/session approvals: if user says "yes for all" / "approve
    all for this task" / "yes for session", the guard is told to auto-approve
    subsequent similar actions so we don't ask per-cmd.
    """

    def __init__(
        self,
        stt: STTProvider,
        tts: TTSProvider,
        console: Optional[Console] = None,
        confidence_threshold: float = 0.6,
        fallback_to_text: bool = True,
        guard: Optional[Any] = None,
    ) -> None:
        self.stt = stt
        self.tts = tts
        self.console = console or Console()
        self.threshold = confidence_threshold
        self.fallback_to_text = fallback_to_text
        self.guard = guard

    def _is_approve_all_phrase(self, text: str) -> Optional[str]:
        t = (text or "").lower()
        if any(k in t for k in ("for all", "always", "approve all", "all task", "all for")):
            return "task"
        if "session" in t:
            return "session"
        return None

    def __call__(self, action) -> bool:
        # Speak the approval request in human language
        risk = getattr(action, "risk", None)
        risk_str = getattr(risk, "value", str(risk)) if risk else "unknown"
        desc = getattr(action, "description", str(action))[:500]
        # Hint for task/session approval so user knows they can say "yes for all"
        prompt = f"{risk_str} action: {desc}. Say yes to approve, no to deny, or yes for all to approve this task."
        logger.info("VoiceApprover: %s", prompt)
        try:
            self.tts.speak(prompt)
        except Exception as exc:
            logger.warning("TTS speak failed in approver: %s", exc)

        # Listen for human language approval
        try:
            tx: Transcription = self.stt.listen_once(timeout=8.0, silence_after=0.8)
            text = (tx.text or "").strip()
            conf = float(getattr(tx, "confidence", 0.0) or 0.0)
            logger.info("VoiceApprover heard %r (conf=%.2f)", text, conf)
            if text and conf >= self.threshold:
                decision = parse_approval(text)
                if decision is not None:
                    if decision and self.guard is not None:
                        scope = self._is_approve_all_phrase(text)
                        if scope is not None and risk is not None:
                            try:
                                # Approve all of this risk (and lower) for task/session
                                self.guard.approve_for_task(risk, scope=scope)  # type: ignore[arg-type]
                                self.tts.speak(f"Approved for {scope}. Won't ask again for similar actions.")
                                logger.info("Voice task approval: %r -> %s scope=%s", text, decision, scope)
                                return True
                            except Exception as exc:
                                logger.debug("guard.approve_for_task failed: %s", exc)
                    self.tts.speak("Approved." if decision else "Denied.")
                    logger.info("Voice approval: %r -> %s", text, decision)
                    return bool(decision)
                # Unclear — fall through to text or re-prompt
                self.tts.speak(f"I didn't catch that. You said {text}. Please say yes or no.")
                # One retry
                tx2 = self.stt.listen_once(timeout=6.0, silence_after=0.8)
                text2 = (tx2.text or "").strip()
                conf2 = float(getattr(tx2, "confidence", 0.0) or 0.0)
                decision2 = parse_approval(text2)
                if decision2 is not None and conf2 >= self.threshold:
                    if decision2 and self.guard is not None:
                        scope2 = self._is_approve_all_phrase(text2)
                        if scope2 is not None and risk is not None:
                            try:
                                self.guard.approve_for_task(risk, scope=scope2)  # type: ignore[arg-type]
                                self.tts.speak(f"Approved for {scope2}.")
                                return True
                            except Exception:
                                pass
                    self.tts.speak("Approved." if decision2 else "Denied.")
                    return bool(decision2)
            # Low confidence or empty — fallback to text (support all/session here too)
            if self.fallback_to_text:
                from rich.prompt import Prompt

                try:
                    # Need to stop any ongoing TTS so prompt is visible
                    self.tts.stop()
                except Exception:
                    pass
                self.console.print(f"[dim]Voice approval unclear ({text!r} conf={conf:.2f}) — falling back to text.[/dim]")
                try:
                    choice = Prompt.ask(
                        "Allow this? [y/n/all/session]",
                        choices=["y", "n", "a", "s", "yes", "no", "all", "session"],
                        default="n",
                        show_choices=False,
                        console=self.console,
                    ).strip().lower()
                except Exception:
                    return False
                if choice in ("y", "yes"):
                    return True
                if choice in ("a", "all", "always") and self.guard is not None and risk is not None:
                    try:
                        self.guard.approve_for_task(risk, scope="task")  # type: ignore[arg-type]
                    except Exception:
                        pass
                    return True
                if choice in ("s", "session") and self.guard is not None and risk is not None:
                    try:
                        self.guard.approve_for_task(risk, scope="session")  # type: ignore[arg-type]
                    except Exception:
                        pass
                    return True
                return False
            return False
        except Exception as exc:
            logger.warning("VoiceApprover failed: %s", exc)
            if self.fallback_to_text:
                from rich.prompt import Prompt

                try:
                    choice = Prompt.ask(
                        "Allow this? [y/n/all/session]",
                        choices=["y", "n", "a", "s", "yes", "no", "all", "session"],
                        default="n",
                        show_choices=False,
                        console=self.console,
                    ).strip().lower()
                    if choice in ("y", "yes"):
                        return True
                    if choice in ("a", "all", "always") and self.guard is not None and risk is not None:
                        try:
                            self.guard.approve_for_task(risk, scope="task")  # type: ignore[arg-type]
                        except Exception:
                            pass
                        return True
                    if choice in ("s", "session") and self.guard is not None and risk is not None:
                        try:
                            self.guard.approve_for_task(risk, scope="session")  # type: ignore[arg-type]
                        except Exception:
                            pass
                        return True
                except Exception:
                    return False
            return False


class VoiceSession:
    """Always-listening, full-duplex voice loop.

    Usage:
        app = ArcApp(...)
        stt = FasterWhisperSTT(model="small", device="cuda")
        tts = make_tts("auto")  # kokoro → piper → pyttsx3
        session = VoiceSession(app, stt, tts)
        session.run_forever()  # blocks until exit phrase
        # or one-shot:
        session.handle_once("what's the weather")
    """

    def __init__(
        self,
        app,
        stt: STTProvider,
        tts: TTSProvider,
        vad: Optional[VADDetector] = None,
        console: Optional[Console] = None,
        allow_voice_approval: bool = True,
        confidence_threshold: float = 0.6,
        wake_word: Optional[str] = None,
        wake_word_enabled: bool = True,
    ) -> None:
        self.app = app
        self.stt = stt
        self.tts = tts
        self.vad = vad or VADDetector()
        self.console = console or Console()
        self.allow_voice_approval = allow_voice_approval
        self.confidence_threshold = confidence_threshold
        # Wake word — when enabled, only respond after hearing "hey" (persists till session end)
        voice_cfg = getattr(app.config, "voice", {}) or {}
        self.wake_word = (wake_word or voice_cfg.get("wake_word", "hey") or "hey").lower().strip()
        self.wake_word_enabled = bool(wake_word_enabled and voice_cfg.get("wake_word_enabled", True))
        self._wake_active_until: float = 0.0
        self._wake_timeout: float = float("inf")  # persist till session end once woken
        self._wake_ever_triggered: bool = False
        self._typed_thread: Optional[threading.Thread] = None
        self._queue: queue.Queue[Transcription] = queue.Queue()
        self._stop = threading.Event()
        self._listener_thread: Optional[threading.Thread] = None
        self._approver: Optional[VoiceApprover] = None
        self._last_tts_text: str = ""
        self._last_tts_time: float = 0.0
        self._tts_muted_until: float = 0.0  # mute STT while TTS playing + 0.8s tail
        self._last_open_time: float = 0.0  # debounce for browser open spam

        # Install voice approver if allowed
        if allow_voice_approval:
            # Pass guard so "yes for all" can set task/session approvals
            try:
                guard_ref = getattr(self.app, "guard", None)
            except Exception:
                guard_ref = None
            self._approver = VoiceApprover(
                stt=self.stt,
                tts=self.tts,
                console=self.console,
                confidence_threshold=confidence_threshold,
                guard=guard_ref,
            )
            try:
                self.app.set_approver(self._approver)
            except Exception as exc:
                logger.debug("set_approver failed: %s", exc)
            # Keep guard in sync if app guard is swapped later
            if guard_ref is not None and hasattr(guard_ref, "approver"):
                # ensure approver's guard stays current (for tests that swap guard)
                self._approver.guard = guard_ref

    def _contains_wake_word(self, text: str) -> bool:
        if not self.wake_word_enabled or not self.wake_word:
            return True  # no wake word required
        t = (text or "").lower().strip()
        # Wake on just "hey" or "hey arc" variants — very permissive
        if t == "hey" or t.startswith("hey ") or " hey " in f" {t} ":
            return True
        return self.wake_word in t or "hey arc" in t or "hey ark" in t or "hi arc" in t or t.startswith("hey")

    def _extract_after_wake(self, text: str) -> str:
        if not self.wake_word_enabled:
            return text
        t_low = text.lower()
        # Try full phrases first, then just "hey"
        for phrase in (self.wake_word, "hey arc", "hey ark", "hi arc", "hey"):
            idx = t_low.find(phrase)
            if idx != -1:
                after = text[idx + len(phrase):].strip(" ,.!?")
                # If just "hey" with no command, keep awake and return empty to prompt
                if not after:
                    return ""
                return after
        return text

    # ------------------------------------------------------------------ internals

    def _listener(self) -> None:
        """Background thread: always listening, pushes transcripts to queue."""
        logger.info("Voice listener started (always listening, full duplex, wake_word=%r enabled=%s)", self.wake_word, self.wake_word_enabled)
        # Fragment joining buffer — handles VAD splitting "Who is the pr-" + "video"
        pending_text = ""
        pending_conf = 0.0
        pending_time = 0.0
        while not self._stop.is_set():
            # Full duplex but with echo suppression: if TTS is speaking, we still
            # listen for barge-in, but we suppress obvious echo of our own voice.
            # Keep a short mute window after TTS starts to avoid immediate echo.
            try:
                # Increased silence_after to join phrases, less fragmentation
                tx = self.stt.listen_once(timeout=1.5, silence_after=1.2)
                text = (tx.text or "").strip()
                if not text:
                    continue
                # Hard mute while TTS is speaking + tail (prevents hearing itself)
                now_mute = time.time()
                if now_mute < self._tts_muted_until:
                    # Allow wake word even during mute
                    if self._contains_wake_word(text):
                        logger.info("Wake word barge-in during mute: %r", text)
                    elif is_echo(text, self._last_tts_text, max_age=4.0, last_time=self._last_tts_time):
                        logger.debug("Dropping muted echo while TTS %r", text)
                        continue
                    elif len(text.split()) <= 3:
                        logger.debug("Dropping muted short %r (muted %.1fs left)", text, self._tts_muted_until - now_mute)
                        continue
                    else:
                        logger.info("Barge-in during mute window: %r", text)
                # Filter very low confidence
                if float(getattr(tx, "confidence", 0.0) or 0.0) < 0.25:
                    logger.debug("Dropping low-confidence transcript %r", text)
                    continue
                # If TTS is speaking (extra flag), be extra strict
                if self.tts.is_speaking:
                    if is_echo(text, self._last_tts_text, max_age=4.0, last_time=self._last_tts_time):
                        logger.debug("Dropping echo while speaking %r", text)
                        continue
                    if len(text.split()) <= 2 and (time.time() - self._last_tts_time) < 2.0:
                        logger.debug("Dropping short transcript while speaking %r", text)
                        continue
                # General echo suppression even when not speaking (within 3.5s)
                if is_echo(text, self._last_tts_text, last_time=self._last_tts_time, max_age=3.5):
                    logger.debug("Dropping echo transcript %r (last TTS %r)", text, self._last_tts_text[:40])
                    continue
                # Wake word handling — once "hey" heard, stay awake till session end
                if self.wake_word_enabled:
                    now = time.time()
                    has_wake = self._contains_wake_word(text)
                    # Once woken, stay awake forever (persist till session end)
                    is_active = self._wake_ever_triggered or (now < self._wake_active_until)
                    if has_wake:
                        self._wake_ever_triggered = True
                        # Extract command after wake word
                        cmd = self._extract_after_wake(text)
                        # If just "hey" with no command, wake and wait for next utterance
                        if not cmd.strip():
                            self._wake_active_until = float("inf")
                            logger.info("Wake word detected: %r — now awake till session end", text)
                            self._last_tts_text = "Yes?"
                            self._last_tts_time = time.time()
                            self._tts_muted_until = time.time() + 1.2
                            try:
                                self.tts.speak("Yes?")
                            except Exception:
                                pass
                            self.console.print("[dim]Wake word — now awake till you say exit[/dim]")
                            continue
                        else:
                            # Wake + command in same utterance
                            self._wake_active_until = float("inf")
                            self._wake_ever_triggered = True
                            logger.info("Wake word + command: %r -> %r (persisting awake)", text, cmd)
                            tx.text = cmd
                            text = cmd
                    elif is_active:
                        # Already woken — treat as command (no need to say hey again)
                        logger.info("Wake active (persisted) — Heard: %r (conf=%.2f)", text, tx.confidence)
                    else:
                        logger.debug("Ignoring (no wake word, not yet woken): %r", text)
                        self.console.print(f"[dim]Ignored (say 'hey' first): {text}[/dim]")
                        continue
                # Fragment joining: if text is very short and we got another fragment quickly, join them
                # e.g. "Who is the pr-" + "video" should be one command
                now = time.time()
                if pending_text and (now - pending_time) < 1.8 and len(pending_text.split()) <= 6:
                    # Check if pending looks incomplete (ends with - or short) or current is continuation
                    combined = f"{pending_text} {text}".strip()
                    # If combined is more complete, use it and clear pending
                    if len(combined.split()) > len(pending_text.split()):
                        text = combined
                        tx.text = combined
                        # Average confidence
                        try:
                            tx.confidence = (pending_conf + float(getattr(tx, "confidence", 0.0) or 0.0)) / 2
                        except Exception:
                            pass
                        pending_text = ""
                        pending_conf = 0.0
                    else:
                        pending_text = text
                        pending_conf = float(getattr(tx, "confidence", 0.0) or 0.0)
                        pending_time = now
                        # Wait for possible next fragment instead of pushing immediately
                        time.sleep(0.3)
                        continue
                # If text is very short (1-2 words) and may be fragment, buffer it briefly
                if len(text.split()) <= 2 and len(text) < 12 and not self._contains_wake_word(text):
                    # Don't push immediately — wait to see if next fragment completes it
                    pending_text = text
                    pending_conf = float(getattr(tx, "confidence", 0.0) or 0.0)
                    pending_time = now
                    logger.debug("Buffering short fragment %r for joining", text)
                    time.sleep(0.4)
                    # Check if another fragment arrived quickly — if not, push the short one
                    if time.time() - pending_time >= 0.4 and pending_text == text:
                        pending_text = ""
                        # fall through to push
                    else:
                        continue

                logger.info("Heard: %r (conf=%.2f)", text, tx.confidence)
                pending_text = ""
                pending_conf = 0.0
                self._queue.put(tx)
                # If ARC is currently speaking, this will trigger barge-in in the main loop
            except Exception as exc:
                logger.debug("listener error: %s", exc)
                time.sleep(0.2)

    def _typed_listener(self) -> None:
        """Background thread: also accept typed input via stdin so you can type even in voice mode."""
        logger.info("Typed listener started — you can type commands while voice is listening")
        import sys

        while not self._stop.is_set():
            try:
                # Use input() with prompt — Rich Console handles it, but we use plain input for simplicity
                # We do blocking read with timeout via select to allow stop check
                import select

                # Check if stdin has data (non-blocking) — if not, sleep and continue
                # Use sys.stdin directly for portability
                if sys.stdin in select.select([sys.stdin], [], [], 0.5)[0]:
                    line = sys.stdin.readline()
                    if not line:
                        # EOF
                        time.sleep(0.1)
                        continue
                    text = line.strip()
                    if not text:
                        continue
                    if is_exit_phrase(text):
                        self._queue.put(Transcription(text=text, confidence=1.0))
                        continue
                    # Typed input bypasses wake word — you already typed, so handle directly
                    # But still support "hey" prefix — strip it if present
                    if text.lower().startswith("hey "):
                        text = text[3:].strip()
                        if text.lower().startswith("arc "):
                            text = text[3:].strip()
                    logger.info("Typed: %r", text)
                    self._queue.put(Transcription(text=text, confidence=1.0))
                else:
                    time.sleep(0.1)
            except Exception as exc:
                logger.debug("typed listener error: %s", exc)
                time.sleep(0.5)

    def _speak_with_bargein(self, text: str) -> bool:
        """Speak text sentence-by-sentence, checking for barge-in between sentences.

        Returns True if barge-in happened (new transcript in queue), False otherwise.
        Uses a thread for TTS so we can poll the queue while speaking.
        """
        # Remember for echo suppression + hard mute window
        self._last_tts_text = text
        self._last_tts_time = time.time()
        # Estimate TTS duration: ~0.06s per char + 0.8s tail, mute STT during this
        est_duration = len(text) * 0.06 + 0.9
        self._tts_muted_until = time.time() + est_duration
        logger.debug("TTS mute until %.1fs (est %.1fs for %d chars)", self._tts_muted_until - time.time(), est_duration, len(text))
        # Split into sentences for more natural barge-in granularity
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return False
        for sent in sentences:
            if self._stop.is_set():
                return True
            # Check for pending barge-in before starting sentence
            if not self._queue.empty():
                # Peek without consuming - check if it's echo first
                try:
                    peek = self._queue.queue[0].text if hasattr(self._queue, "queue") else ""
                    if is_echo(peek, self._last_tts_text, last_time=self._last_tts_time):
                        # Drop echo and continue speaking
                        try:
                            self._queue.get_nowait()
                        except queue.Empty:
                            pass
                        logger.debug("Ignoring echo barge-in %r", peek[:40])
                    else:
                        logger.info("Barge-in detected before sentence %r", sent[:40])
                        self.tts.stop()
                        return True
                except Exception:
                    logger.info("Barge-in detected before sentence %r", sent[:40])
                    self.tts.stop()
                    return True
            # Speak in a thread so we can poll for barge-in while speaking
            done = threading.Event()

            def _speak():
                try:
                    self.tts.speak(sent)
                finally:
                    done.set()

            t = threading.Thread(target=_speak, daemon=True)
            t.start()
            # Poll queue while TTS is speaking
            while not done.is_set():
                if not self._queue.empty():
                    try:
                        peek = self._queue.queue[0].text if hasattr(self._queue, "queue") else ""
                        if is_echo(peek, self._last_tts_text, last_time=self._last_tts_time):
                            try:
                                self._queue.get_nowait()
                            except queue.Empty:
                                pass
                            logger.debug("Ignoring echo while speaking %r", peek[:40])
                            continue
                    except Exception:
                        pass
                    logger.info("Barge-in while speaking %r", sent[:40])
                    self.tts.stop()
                    # Wait a bit for stop to take effect
                    time.sleep(0.15)
                    return True
                if self._stop.is_set():
                    self.tts.stop()
                    return True
                time.sleep(0.08)
            t.join(timeout=0.1)
        return False

    def _handle_text(self, text: str) -> Optional[str]:
        """Send text to ARC and speak the reply (with barge-in)."""
        if not text or not text.strip():
            return None
        # Echo suppression: if this transcript is actually our own TTS, ignore
        if is_echo(text, self._last_tts_text, last_time=self._last_tts_time):
            logger.debug("Ignoring echo in handle_text %r", text[:60])
            return None
        if is_exit_phrase(text):
            self.tts.speak("Goodbye. Stopping voice session.")
            self._stop.set()
            return None
        # Debounce rapid duplicate browser open requests (voice hallucination: "open Chrome" repeated)
        low = text.lower()
        if any(k in low for k in ("open chrome", "open browser", "chatgpt")):
            now = time.time()
            if now - self._last_open_time < 5.0:
                logger.debug("Debouncing duplicate open request %r (%.1fs since last)", text, now - self._last_open_time)
                self.console.print("[dim]Debounced duplicate Chrome open — using existing window.[/dim]")
                return None
            self._last_open_time = now

        self.console.print(f"[bold cyan]you ›[/bold cyan] {text}")
        # Show thinking state
        try:
            turn = self.app.orchestrator.handle(text)
        except Exception as exc:
            logger.exception("orchestrator.handle failed")
            err = f"The agent crashed: {exc}"
            self.console.print(f"[red]{err}[/red]")
            try:
                self.tts.speak("Sorry, the agent crashed.")
            except Exception:
                pass
            return err

        reply = (getattr(turn, "reply", None) or "").strip() or "(empty)"
        self.console.print(f"[dim]model: {getattr(turn, 'model', '?')} · {getattr(turn, 'tool_calls', 0)} tool calls[/dim]")
        # Also print to console for accessibility
        self.console.print(f"[bold]arc ›[/bold] {reply[:500]}")

        # Speak with barge-in support
        barged = self._speak_with_bargein(reply)
        if barged:
            logger.info("Barge-in handled, will process next utterance")
        return reply

    # ------------------------------------------------------------------ public

    def handle_once(self, text: str) -> Optional[str]:
        """One-shot: handle a single utterance (used for `arc voice --once`)."""
        return self._handle_text(text)

    def run_forever(self) -> None:
        """Always-listening full-duplex loop. Blocks until exit phrase or Ctrl-C."""
        from rich.text import Text

        # Show ARC banner (same as TerminalUI but voice)
        try:
            from ..ui.terminal import BANNER

            self.console.print(Text(BANNER, style="cyan bold"))
        except Exception:
            pass
        self.console.print("  [bold]ARC voice[/bold] — always listening, full duplex. Say [cyan]exit[/cyan] or [cyan]goodbye[/cyan] to stop.\n")
        # Announce via TTS as well — set echo suppression + mute before speaking
        tts_text = "Arc voice is ready. I'm listening. Just talk, and I'll interrupt if you need me."
        self._last_tts_text = tts_text
        self._last_tts_time = time.time()
        self._tts_muted_until = time.time() + len(tts_text) * 0.06 + 1.0
        try:
            self.tts.speak(tts_text)
        except Exception:
            pass

        # Start listener thread (voice) + typed thread (keyboard) — you can both talk and type
        self._stop.clear()
        self._listener_thread = threading.Thread(target=self._listener, daemon=True, name="arc-voice-listener")
        self._listener_thread.start()
        self._typed_thread = threading.Thread(target=self._typed_listener, daemon=True, name="arc-typed-listener")
        self._typed_thread.start()

        self.console.print("[dim]Listening… (full duplex — you can interrupt me anytime)[/dim]")
        self.console.print("[dim]Voice: say [cyan]hey[/cyan] once to wake — stays awake till you say exit[/dim]")
        self.console.print("[dim]Type: just type your message and press Enter — you don't need to say hey[/dim]")

        try:
            while not self._stop.is_set():
                try:
                    tx: Transcription = self._queue.get(timeout=0.15)
                except queue.Empty:
                    continue
                text = (tx.text or "").strip()
                if not text:
                    continue
                self._handle_text(text)
                # Check for stop after handling (exit phrase sets _stop)
                if self._stop.is_set():
                    break
        except KeyboardInterrupt:
            self.console.print("\n[dim]Voice session interrupted (Ctrl-C).[/dim]")
            self._stop.set()
        finally:
            self._stop.set()
            # Interrupt STT if it's blocking in listen_once
            try:
                # Wake up listener by pushing dummy
                self._queue.put(Transcription(text="__stop__", confidence=0.0))
            except Exception:
                pass
            try:
                self.tts.stop()
            except Exception:
                pass
            if self._listener_thread:
                self._listener_thread.join(timeout=1.0)
            if getattr(self, "_typed_thread", None):
                try:
                    self._typed_thread.join(timeout=0.5)
                except Exception:
                    pass
            self.console.print("[dim]Voice session ended.[/dim]")
            try:
                self.tts.speak("Voice session ended.")
            except Exception:
                pass
