"""TTS — local, interruptible. Kokoro on GPU if VRAM allows, else Piper/pyttsx3.

Always-listening full-duplex needs TTS that can be stopped mid-sentence
when the user barges in. All providers expose `speak(text)` + `stop()` +
`is_speaking`.

Priority for local RTX 4050 6GB:
  1. Kokoro (neural, ~2GB VRAM) — best quality, still fits with small whisper (0.5GB). Used if `kokoro` is installed and RAM check passes.
  2. Piper (piper-tts) — CPU, tiny, very fast, no VRAM.
  3. pyttsx3 (espeak) — fallback, always available after `pip install pyttsx3`, robotic but works.

The session picks the first available provider in that order unless config forces one.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("arc.voice.tts")

# ------------------------------------------------------------------ base


class TTSProvider(ABC):
    @abstractmethod
    def speak(self, text: str) -> None:
        """Block until text is spoken (or stopped via `stop()`)."""
        ...

    def stop(self) -> None:
        """Interrupt current speech if any. No-op if not speaking."""
        pass

    @property
    def is_speaking(self) -> bool:
        return False

    @property
    def available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return self.__class__.__name__


# ------------------------------------------------------------------ fakes


class FakeTTS(TTSProvider):
    """Deterministic fake for tests — records what was spoken."""

    def __init__(self) -> None:
        self.spoken: list[str] = []
        self._speaking = False
        self.stopped_count = 0

    def speak(self, text: str) -> None:
        self._speaking = True
        # simulate short delay
        time.sleep(0.01)
        self.spoken.append(text)
        self._speaking = False

    def stop(self) -> None:
        if self._speaking:
            self.stopped_count += 1
        self._speaking = False

    @property
    def is_speaking(self) -> bool:
        return self._speaking


# ------------------------------------------------------------------ pyttsx3 (espeak) — always local


class Pyttsx3TTS(TTSProvider):
    """Local espeak via pyttsx3. CPU, no VRAM, robotic but reliable.

    Interruptible via `stop()` which calls engine.stop().
    """

    def __init__(self, voice: str | None = None, rate: int = 180) -> None:
        self.voice = voice
        self.rate = rate
        self._engine = None
        self._speaking = False
        self._lock = threading.Lock()
        try:
            import pyttsx3  # type: ignore
            self._pyttsx3 = pyttsx3
            self._engine = pyttsx3.init()
            if voice:
                for v in self._engine.getProperty("voices"):
                    if voice.lower() in v.id.lower() or voice.lower() in v.name.lower():
                        self._engine.setProperty("voice", v.id)
                        break
            self._engine.setProperty("rate", rate)
            logger.info("pyttsx3 TTS ready (voice=%s, rate=%d)", voice or "default", rate)
        except Exception as exc:
            logger.warning("pyttsx3 init failed: %s — TTS will use print fallback", exc)
            self._engine = None
            self._pyttsx3 = None  # type: ignore

    @property
    def available(self) -> bool:
        return self._engine is not None

    def speak(self, text: str) -> None:
        text = self._clean(text)
        if not text:
            return
        if self._engine is None:
            # fallback: print so full-duplex still shows output
            print(f"[tts] {text}", flush=True)
            return
        with self._lock:
            self._speaking = True
        try:
            # pyttsx3 is not thread-safe for concurrent speak, so we serialize
            self._engine.say(text)
            self._engine.runAndWait()
        except RuntimeError as exc:
            # runAndWait loop already started in another thread
            logger.debug("pyttsx3 runAndWait collision: %s", exc)
            # fallback to print
            print(f"[tts] {text}", flush=True)
        except Exception as exc:
            logger.warning("pyttsx3 speak failed: %s", exc)
            print(f"[tts] {text}", flush=True)
        finally:
            with self._lock:
                self._speaking = False

    def stop(self) -> None:
        if self._engine is None:
            return
        try:
            self._engine.stop()
        except Exception:
            pass
        with self._lock:
            self._speaking = False

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._speaking

    @staticmethod
    def _clean(text: str) -> str:
        # strip markdown, code fences for more natural speech
        text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"[*_#>|]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        # limit length for TTS (avoid reading huge dumps)
        if len(text) > 800:
            text = text[:800] + "…"
        return text


# ------------------------------------------------------------------ Piper (CPU, tiny, fast)


class PiperTTS(TTSProvider):
    """Piper TTS — local, CPU, <100MB. Preferred over pyttsx3 when installed.

    Requires `pip install piper-tts` and a voice model. Falls back to Pyttsx3TTS if not available.
    """

    def __init__(self, voice: str = "en_US-amy-medium", model_path: str | None = None) -> None:
        self.voice = voice
        self.model_path = model_path
        self._piper = None
        self._voice = None
        self._fallback: Optional[Pyttsx3TTS] = None
        self._speaking = False
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        try:
            import piper  # type: ignore
            self._piper = piper
            # Try to load voice model immediately
            try:
                from pathlib import Path
                # Search common locations
                candidates = []
                if model_path:
                    candidates.append(Path(model_path))
                # Current dir (where download_voices puts it)
                candidates.append(Path(f"{voice}.onnx"))
                candidates.append(Path.cwd() / f"{voice}.onnx")
                # Home piper dir
                candidates.append(Path.home() / ".local/share/piper/voices" / voice / f"{voice}.onnx")
                candidates.append(Path.home() / f".local/share/piper/{voice}.onnx")
                # Data dir
                candidates.append(Path("data/piper") / f"{voice}.onnx")
                model_file = None
                for p in candidates:
                    if p.is_file():
                        model_file = p
                        break
                if model_file:
                    self._voice = piper.PiperVoice.load(str(model_file))
                    logger.info("piper TTS ready (voice=%s, model=%s)", voice, model_file)
                else:
                    logger.info("piper voice %s not found — will use pyttsx3 until downloaded (python -m piper.download_voices %s)", voice, voice)
                    self._fallback = Pyttsx3TTS()
            except Exception as exc:
                logger.warning("piper voice load failed: %s — fallback to pyttsx3", exc)
                self._fallback = Pyttsx3TTS()
        except ImportError:
            logger.info("piper-tts not installed — will use pyttsx3 (pip install piper-tts)")
            self._piper = None
            self._fallback = Pyttsx3TTS()

    @property
    def available(self) -> bool:
        return (self._voice is not None) or (self._piper is not None) or (self._fallback is not None and self._fallback.available)

    def speak(self, text: str) -> None:
        text = Pyttsx3TTS._clean(text)
        if not text:
            return
        if self._voice is not None and self._piper is not None:
            with self._lock:
                self._speaking = True
                self._stop_flag.clear()
            try:
                import sounddevice as sd
                import numpy as np

                # Piper synthesizes to AudioChunk(s) with audio_float_array
                audio_chunks = []
                for chunk in self._voice.synthesize(text):
                    if self._stop_flag.is_set():
                        break
                    # AudioChunk has audio_float_array (numpy) and sample_rate
                    arr = getattr(chunk, "audio_float_array", None)
                    if arr is not None:
                        audio_chunks.append(arr)
                    elif hasattr(chunk, "audio_int16_bytes"):
                        # Fallback for older API
                        import numpy as np
                        audio_chunks.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16).astype(np.float32) / 32768.0)

                if self._stop_flag.is_set() or not audio_chunks:
                    return
                audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
                samplerate = getattr(self._voice, "config", {}).sample_rate if hasattr(self._voice.config, "sample_rate") else 22050
                try:
                    samplerate = int(samplerate)
                except Exception:
                    samplerate = 22050

                if self._stop_flag.is_set():
                    return
                sd.play(audio, samplerate=samplerate)
                while sd.get_stream().active and not self._stop_flag.is_set():
                    sd.sleep(50)
                if self._stop_flag.is_set():
                    sd.stop()
            except Exception as exc:
                logger.warning("piper speak failed: %s — fallback", exc)
                if self._fallback:
                    self._fallback.speak(text)
                else:
                    print(f"[tts] {text}", flush=True)
            finally:
                with self._lock:
                    self._speaking = False
            return
        if self._piper is None:
            if self._fallback:
                self._fallback.speak(text)
            else:
                print(f"[tts] {text}", flush=True)
            return
        # piper installed but voice not loaded — fallback
        if self._fallback:
            self._fallback.speak(text)
        else:
            print(f"[tts] {text}", flush=True)

    def stop(self) -> None:
        self._stop_flag.set()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        if self._fallback:
            self._fallback.stop()
        with self._lock:
            self._speaking = False

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            if self._speaking:
                return True
        if self._fallback:
            return self._fallback.is_speaking
        return False


# ------------------------------------------------------------------ Kokoro (neural, ~2GB VRAM, best quality)


class KokoroTTS(TTSProvider):
    """Kokoro — neural TTS, ~82M params, in 32 languages. Best quality.

    Needs ~2GB VRAM on GPU (float16) or ~1GB RAM on CPU. On RTX 4050 6GB it
    fits comfortably alongside faster-whisper small (0.5GB). If VRAM is tight
    or `kokoro` is not installed, the session will fall back to Piper/pyttsx3.

    Install: `pip install -e \".[kokoro]\"` or `pip install kokoro soundfile`
    Voice: e.g. `af_heart` (American female), `bf_emma`, etc. See `kokoro` docs.
    """

    def __init__(
        self,
        voice: str = "af_heart",
        lang: str = "en-us",
        device: str = "auto",
        sample_rate: int = 24000,
    ) -> None:
        self.voice = voice
        self.lang = lang
        self.sample_rate = sample_rate
        self._speaking = False
        self._stop_flag = threading.Event()
        self._lock = threading.Lock()
        self._koko = None
        self._fallback: Optional[Pyttsx3TTS] = None

        # Device selection
        if device == "auto":
            try:
                import torch  # type: ignore
                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    # Need ~2GB free for kokoro
                    if free > 2.5 * 1024**3:
                        device = "cuda"
                    else:
                        logger.info("Kokoro: only %.1fGB free VRAM, using CPU", free / 1024**3)
                        device = "cpu"
                else:
                    device = "cpu"
            except Exception:
                device = "cpu"
        self.device = device

        try:
            from kokoro import KPipeline  # type: ignore
            import soundfile as sf  # type: ignore
            self._sf = sf
            # KPipeline loads model lazily; we init with lang
            try:
                self._koko = KPipeline(lang_code=lang[0] if lang else "a", repo_id="hexgrad/Kokoro-82M")
                logger.info("Kokoro TTS ready (voice=%s, lang=%s, device=%s)", voice, lang, device)
            except Exception as exc:
                logger.warning("Kokoro pipeline init failed: %s — falling back to pyttsx3", exc)
                self._koko = None
                self._fallback = Pyttsx3TTS()
        except ImportError:
            logger.info("kokoro not installed — will use piper/pyttsx3 (pip install -e \".[kokoro]\")")
            self._koko = None
            self._fallback = Pyttsx3TTS()

    @property
    def available(self) -> bool:
        return self._koko is not None or (self._fallback is not None and self._fallback.available)

    def speak(self, text: str) -> None:
        text = Pyttsx3TTS._clean(text)
        if not text:
            return
        if self._koko is None:
            if self._fallback:
                self._fallback.speak(text)
            else:
                print(f"[tts] {text}", flush=True)
            return
        with self._lock:
            self._speaking = True
            self._stop_flag.clear()
        try:
            # Kokoro streaming: generate audio in chunks and play via sounddevice
            # For now we generate whole utterance and play; streaming sentence-by-sentence
            # is handled by VoiceSession which calls speak() per sentence.
            import sounddevice as sd  # type: ignore

            # KPipeline returns generator of (graphemes, phonemes, audio)
            # We use the simple API: koko(text, voice)
            audio_chunks = []
            for _, _, audio in self._koko(text, voice=self.voice):
                if self._stop_flag.is_set():
                    break
                audio_chunks.append(audio)
            if self._stop_flag.is_set() or not audio_chunks:
                return
            import numpy as np  # type: ignore

            audio = np.concatenate(audio_chunks)
            # Play — interruptible via stop()
            sd.play(audio, samplerate=self.sample_rate)
            # Wait with stop polling (instead of sd.wait which blocks)
            while sd.get_stream().active and not self._stop_flag.is_set():
                sd.sleep(50)
            if self._stop_flag.is_set():
                sd.stop()
        except ImportError:
            # sounddevice not available
            if self._fallback:
                self._fallback.speak(text)
            else:
                print(f"[tts] {text}", flush=True)
        except Exception as exc:
            logger.warning("Kokoro speak failed: %s — falling back", exc)
            if self._fallback:
                self._fallback.speak(text)
            else:
                print(f"[tts] {text}", flush=True)
        finally:
            with self._lock:
                self._speaking = False

    def stop(self) -> None:
        self._stop_flag.set()
        try:
            import sounddevice as sd  # type: ignore

            sd.stop()
        except Exception:
            pass
        if self._fallback:
            self._fallback.stop()
        with self._lock:
            self._speaking = False

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._speaking


# ------------------------------------------------------------------ factory


def make_tts(kind: str = "auto", voice: str | None = None, device: str = "auto") -> TTSProvider:
    """Factory: picks the best available TTS for the machine.

    `kind` = "auto" | "kokoro" | "piper" | "pyttsx3" | "fake"
    `auto` tries Kokoro (if VRAM allows) → Piper → pyttsx3.
    """
    kind = (kind or "auto").lower()
    if kind == "fake":
        return FakeTTS()
    if kind == "kokoro":
        return KokoroTTS(voice=voice or "af_heart", device=device)
    if kind == "piper":
        return PiperTTS(voice=voice or "en_US-amy-medium")
    if kind == "pyttsx3":
        return Pyttsx3TTS(voice=voice)

    # auto: try kokoro first if we have RAM, but prefer piper when VRAM tight (llama running)
    if kind == "auto":
        # If VRAM low (<3GB free), piper (100MB CPU) is safer than kokoro (2GB)
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                try:
                    free, _ = torch.cuda.mem_get_info()
                    if free < 3 * 1024**3:
                        t = PiperTTS(voice=voice or "en_US-amy-medium")
                        if t.available and t._voice is not None:
                            return t
                except Exception:
                    pass
        except Exception:
            pass
        # Try kokoro if installed and we have VRAM
        try:
            t = KokoroTTS(voice=voice or "af_heart", device=device)
            if t.available and t._koko is not None:
                return t
        except Exception:
            pass
        # Then piper
        try:
            t = PiperTTS(voice=voice or "en_US-amy-medium")
            if t.available and (t._voice is not None or t._piper is not None):
                return t
        except Exception:
            pass
        return Pyttsx3TTS(voice=voice)
    return Pyttsx3TTS(voice=voice)
