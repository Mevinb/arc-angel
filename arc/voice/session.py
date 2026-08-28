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
    """

    def __init__(
        self,
        stt: STTProvider,
        tts: TTSProvider,
        console: Optional[Console] = None,
        confidence_threshold: float = 0.6,
        fallback_to_text: bool = True,
    ) -> None:
        self.stt = stt
        self.tts = tts
        self.console = console or Console()
        self.threshold = confidence_threshold
        self.fallback_to_text = fallback_to_text

    def __call__(self, action) -> bool:
        # Speak the approval request in human language
        risk = getattr(action, "risk", None)
        risk_str = getattr(risk, "value", str(risk)) if risk else "unknown"
        desc = getattr(action, "description", str(action))[:500]
        prompt = f"{risk_str} action: {desc}. Say yes to approve, or no to deny."
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
                    self.tts.speak("Approved." if decision2 else "Denied.")
                    return bool(decision2)
            # Low confidence or empty — fallback to text
            if self.fallback_to_text:
                from rich.prompt import Confirm

                try:
                    # Need to stop any ongoing TTS so prompt is visible
                    self.tts.stop()
                except Exception:
                    pass
                self.console.print(f"[dim]Voice approval unclear ({text!r} conf={conf:.2f}) — falling back to text.[/dim]")
                return bool(Confirm.ask("Allow this?", default=False, console=self.console))
            return False
        except Exception as exc:
            logger.warning("VoiceApprover failed: %s", exc)
            if self.fallback_to_text:
                from rich.prompt import Confirm

                return bool(Confirm.ask("Allow this?", default=False, console=self.console))
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
    ) -> None:
        self.app = app
        self.stt = stt
        self.tts = tts
        self.vad = vad or VADDetector()
        self.console = console or Console()
        self.allow_voice_approval = allow_voice_approval
        self.confidence_threshold = confidence_threshold
        self._queue: queue.Queue[Transcription] = queue.Queue()
        self._stop = threading.Event()
        self._listener_thread: Optional[threading.Thread] = None
        self._approver: Optional[VoiceApprover] = None

        # Install voice approver if allowed
        if allow_voice_approval:
            self._approver = VoiceApprover(
                stt=self.stt,
                tts=self.tts,
                console=self.console,
                confidence_threshold=confidence_threshold,
            )
            try:
                self.app.set_approver(self._approver)
            except Exception as exc:
                logger.debug("set_approver failed: %s", exc)

    # ------------------------------------------------------------------ internals

    def _listener(self) -> None:
        """Background thread: always listening, pushes transcripts to queue."""
        logger.info("Voice listener started (always listening, full duplex)")
        while not self._stop.is_set():
            try:
                tx = self.stt.listen_once(timeout=1.2, silence_after=0.8)
                text = (tx.text or "").strip()
                if not text:
                    continue
                # Filter very low confidence
                if float(getattr(tx, "confidence", 0.0) or 0.0) < 0.25:
                    logger.debug("Dropping low-confidence transcript %r", text)
                    continue
                logger.info("Heard: %r (conf=%.2f)", text, tx.confidence)
                self._queue.put(tx)
                # If ARC is currently speaking, this will trigger barge-in in the main loop
            except Exception as exc:
                logger.debug("listener error: %s", exc)
                time.sleep(0.2)

    def _speak_with_bargein(self, text: str) -> bool:
        """Speak text sentence-by-sentence, checking for barge-in between sentences.

        Returns True if barge-in happened (new transcript in queue), False otherwise.
        Uses a thread for TTS so we can poll the queue while speaking.
        """
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
        if is_exit_phrase(text):
            self.tts.speak("Goodbye. Stopping voice session.")
            self._stop.set()
            return None

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
        # Announce via TTS as well
        try:
            self.tts.speak("Arc voice is ready. I'm listening. Just talk, and I'll interrupt if you need me.")
        except Exception:
            pass

        # Start listener thread
        self._stop.clear()
        self._listener_thread = threading.Thread(target=self._listener, daemon=True, name="arc-voice-listener")
        self._listener_thread.start()

        self.console.print("[dim]Listening… (full duplex — you can interrupt me anytime)[/dim]")

        try:
            while not self._stop.is_set():
                try:
                    tx: Transcription = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                text = (tx.text or "").strip()
                if not text:
                    continue
                self._handle_text(text)
        except KeyboardInterrupt:
            self.console.print("\n[dim]Voice session interrupted (Ctrl-C).[/dim]")
        finally:
            self._stop.set()
            try:
                self.tts.stop()
            except Exception:
                pass
            if self._listener_thread:
                self._listener_thread.join(timeout=1.0)
            self.console.print("[dim]Voice session ended.[/dim]")
            try:
                self.tts.speak("Voice session ended.")
            except Exception:
                pass
