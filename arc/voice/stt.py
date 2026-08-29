"""STT — Faster-Whisper local on RTX 4050 (6GB) + streaming.

Always-listening, full-duplex: sounddevice captures 16kHz mono, webrtcvad
gates silence, faster-whisper transcribes on CUDA (float16) with ~300ms.

Falls back to CPU if CUDA unavailable, and to FakeSTT for tests / when
faster-whisper is not installed.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("arc.voice.stt")

try:
    import numpy as np  # type: ignore
    _HAS_NP = True
except ImportError:
    np = None  # type: ignore
    _HAS_NP = False

try:
    from faster_whisper import WhisperModel  # type: ignore
    _HAS_FASTER_WHISPER = True
except ImportError:
    WhisperModel = None  # type: ignore
    _HAS_FASTER_WHISPER = False

try:
    import sounddevice as sd  # type: ignore
    _HAS_SD = True
except ImportError:
    sd = None  # type: ignore
    _HAS_SD = False


@dataclass
class Transcription:
    text: str
    confidence: float  # 0.0-1.0, mapped from no_speech_prob
    language: str = "en"
    is_final: bool = True
    duration: float = 0.0


class STTProvider(ABC):
    @abstractmethod
    def transcribe(self, audio: "np.ndarray", sample_rate: int = 16000) -> Transcription:
        ...

    @abstractmethod
    def listen_once(self, timeout: float = 8.0, silence_after: float = 0.8) -> Transcription:
        """Block until an utterance is captured or timeout.

        Always-listening loop calls this repeatedly. Uses VAD to detect
        speech start and silence_after to detect end.
        """
        ...

    @property
    def available(self) -> bool:
        return True


class FakeSTT(STTProvider):
    """Deterministic fake for tests / CI — no mic, no model."""

    def __init__(self, script: list[str] | None = None) -> None:
        self.script = list(script or ["yes"])
        self.calls = 0

    def transcribe(self, audio, sample_rate=16000) -> Transcription:
        text = self.script[min(self.calls, len(self.script) - 1)] if self.script else "yes"
        self.calls += 1
        return Transcription(text=text, confidence=0.95, is_final=True)

    def listen_once(self, timeout=8.0, silence_after=0.8) -> Transcription:
        if not self.script:
            return Transcription(text="", confidence=0.0, is_final=True)
        text = self.script.pop(0) if self.script else ""
        self.calls += 1
        # simulate confidence; empty script => silence
        conf = 0.92 if text else 0.0
        return Transcription(text=text, confidence=conf, is_final=True)


class FasterWhisperSTT(STTProvider):
    """Local Faster-Whisper on CUDA (RTX 4050 6GB).

    Defaults to `small` (500MB VRAM) for real-time <300ms. `medium` (1.5GB)
    also fits 6GB if user wants higher accuracy via config.

    Uses ctranslate2 with float16 on CUDA, int8 on CPU.
    """

    def __init__(
        self,
        model: str = "small",
        device: str = "auto",
        compute_type: str = "auto",
        language: str = "en",
        vad_filter: bool = True,
    ) -> None:
        self.model_name = model
        self.language = language
        self.vad_filter = vad_filter
        self._model: Optional[WhisperModel] = None
        self._lock = threading.Lock()

        # Device selection: auto → try cuda first, fallback to cpu
        # On RTX 4050 6GB, cuda float16 is preferred (500MB for small). We try
        # cuda without requiring torch — ctranslate2 will tell us if it works.
        if device == "auto":
            # Try to detect CUDA via ctranslate2 or torch if available
            try:
                import ctranslate2  # type: ignore

                # ctranslate2 4.x exposes get_cuda_device_count
                if hasattr(ctranslate2, "get_cuda_device_count"):
                    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
                else:
                    # Fallback: try torch if available, else assume cuda (let WhisperModel decide)
                    try:
                        import torch  # type: ignore

                        device = "cuda" if torch.cuda.is_available() else "cpu"
                    except Exception:
                        device = "cuda"  # try cuda, will fallback on load failure
            except Exception:
                device = "cpu"
        self.device = device

        if compute_type == "auto":
            compute_type = "float16" if self.device == "cuda" else "int8"
        self.compute_type = compute_type

        if not _HAS_FASTER_WHISPER:
            logger.warning("faster-whisper not installed — STT will fail until `pip install -e .[voice]`")
            return

        try:
            logger.info("Loading faster-whisper %s on %s (%s)…", model, device, compute_type)
            self._model = WhisperModel(model, device=device, compute_type=compute_type)
            logger.info("faster-whisper %s ready on %s", model, device)
        except Exception as exc:
            logger.warning("faster-whisper load failed (%s) — trying CPU int8", exc)
            try:
                self._model = WhisperModel(model, device="cpu", compute_type="int8")
                self.device = "cpu"
                self.compute_type = "int8"
                logger.info("faster-whisper %s ready on CPU fallback", model)
            except Exception as exc2:
                logger.error("faster-whisper failed completely: %s", exc2)
                self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def transcribe(self, audio: "np.ndarray", sample_rate: int = 16000) -> Transcription:
        if self._model is None:
            return Transcription(text="", confidence=0.0, language=self.language)
        if audio is None or len(audio) == 0:
            return Transcription(text="", confidence=0.0)
        # Energy check: if audio is mostly silence, don't hallucinate
        if _HAS_NP and hasattr(audio, "size"):
            try:
                rms = float((audio.astype(float) ** 2).mean() ** 0.5) if audio.size > 0 else 0.0
                if rms < 0.008:  # very quiet
                    logger.debug("Dropping low-energy audio (rms=%.4f)", rms)
                    return Transcription(text="", confidence=0.0, duration=0.0)
            except Exception:
                pass
        # faster-whisper expects float32 [-1, 1] at 16kHz
        if sample_rate != 16000 and _HAS_NP:
            # naive — assume caller already resampled; we don't resample here
            pass
        start = time.time()
        try:
            with self._lock:
                segments, info = self._model.transcribe(
                    audio,
                    language=self.language if self.language != "auto" else None,
                    vad_filter=self.vad_filter,
                    beam_size=1,  # fastest for real-time
                    best_of=1,
                    temperature=0.0,
                )
                # Collect segments and check for hallucination
                segments_list = list(segments)
                text = " ".join(seg.text.strip() for seg in segments_list).strip()
                # Use no_speech_prob and avg_logprob to detect hallucination
                # If all segments have high no_speech_prob, it's likely silence
                if segments_list:
                    avg_no_speech = sum(getattr(s, "no_speech_prob", 0.0) or 0.0 for s in segments_list) / len(segments_list)
                    avg_logprob = sum(getattr(s, "avg_logprob", -1.0) or -1.0 for s in segments_list) / len(segments_list)
                    # Hallucination check: high no_speech or very low logprob
                    if avg_no_speech > 0.6:
                        logger.debug("Dropping hallucination (no_speech=%.2f) %r", avg_no_speech, text)
                        return Transcription(text="", confidence=0.0, duration=time.time() - start)
                    if avg_logprob < -1.5 and len(text) < 20:
                        logger.debug("Dropping low logprob (%.2f) %r", avg_logprob, text)
                        return Transcription(text="", confidence=0.0, duration=time.time() - start)
                # info.language_probability is a proxy for confidence
                conf = float(getattr(info, "language_probability", 0.85) or 0.85)
                # Penalize very short / empty
                if not text:
                    conf = 0.0
                elif len(text) < 3:
                    conf *= 0.5
                # Common whisper hallucinations on silence
                hallucinations = {
                    "i'm always afraid of it",
                    "im always afraid of it",
                    "i'm not voice is ready",
                    "im not voice is ready",
                    "voice is ready",
                    "voice is ready.",
                    "moises ready",
                    "moises",
                    "thank you",
                    "thanks for watching",
                    "you",
                    "so",
                    "the",
                }
                # Normalize punctuation for hallucination check
                norm = re.sub(r"[^\w\s]", "", text.lower().strip())
                if norm in hallucinations or text.lower().strip() in hallucinations:
                    logger.debug("Dropping known hallucination %r", text)
                    return Transcription(text="", confidence=0.0, duration=time.time() - start)
                dur = time.time() - start
                logger.debug("transcribed %d samples -> %r (conf=%.2f, %.2fs)", len(audio), text, conf, dur)
                return Transcription(text=text, confidence=conf, language=getattr(info, "language", self.language), duration=dur)
        except Exception as exc:
            # If CUDA libs missing (libcublas), fallback to CPU and retry once
            if "libcublas" in str(exc).lower() or "cuda" in str(exc).lower():
                logger.warning("CUDA transcribe failed (%s) — falling back to CPU", exc)
                try:
                    with self._lock:
                        # Reload on CPU if not already
                        if self.device != "cpu":
                            logger.info("Reloading faster-whisper %s on CPU int8", self.model_name)
                            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
                            self.device = "cpu"
                            self.compute_type = "int8"
                            # Retry once on CPU
                            segments, info = self._model.transcribe(
                                audio,
                                language=self.language if self.language != "auto" else None,
                                vad_filter=self.vad_filter,
                                beam_size=1,
                                best_of=1,
                                temperature=0.0,
                            )
                            segments_list = list(segments)
                            text = " ".join(seg.text.strip() for seg in segments_list).strip()
                            if segments_list:
                                avg_no_speech = sum(getattr(s, "no_speech_prob", 0.0) or 0.0 for s in segments_list) / len(segments_list)
                                if avg_no_speech > 0.6:
                                    logger.debug("CPU fallback dropping hallucination (no_speech=%.2f) %r", avg_no_speech, text)
                                    return Transcription(text="", confidence=0.0, duration=time.time() - start)
                                norm2 = re.sub(r"[^\w\s]", "", text.lower().strip())
                                hallucinations2 = {
                                    "im always afraid of it",
                                    "im not voice is ready",
                                    "voice is ready",
                                    "moises ready",
                                    "moises",
                                    "thank you",
                                    "thanks for watching",
                                    "you",
                                    "so",
                                    "the",
                                }
                                if norm2 in hallucinations2 or text.lower().strip() in hallucinations2:
                                    logger.debug("CPU fallback dropping hallucination %r", text)
                                    return Transcription(text="", confidence=0.0, duration=time.time() - start)
                            conf = float(getattr(info, "language_probability", 0.85) or 0.85)
                            if not text:
                                conf = 0.0
                            elif len(text) < 3:
                                conf *= 0.5
                            dur = time.time() - start
                            logger.info("CPU fallback transcribed %r (conf=%.2f)", text, conf)
                            return Transcription(text=text, confidence=conf, language=getattr(info, "language", self.language), duration=dur)
                except Exception as exc2:
                    logger.warning("CPU fallback also failed: %s", exc2)
            logger.warning("transcribe failed: %s", exc)
            return Transcription(text="", confidence=0.0, duration=time.time() - start)

    def listen_once(self, timeout: float = 12.0, silence_after: float = 0.9) -> Transcription:
        """Capture from mic until utterance ends (VAD-gated).

        Always-listening: blocks, but returns quickly on silence.
        Uses sounddevice + webrtcvad if available, else records fixed window.
        """
        if not _HAS_SD:
            logger.warning("sounddevice not installed — cannot listen (pip install sounddevice)")
            time.sleep(min(timeout, 0.5))
            return Transcription(text="", confidence=0.0)

        from .vad import VADDetector

        vad = VADDetector(aggressiveness=3)
        sample_rate = 16000
        chunk_ms = 30
        chunk_samples = int(sample_rate * chunk_ms / 1000)
        # collect with silence detection
        audio_chunks: list = []
        q: queue.Queue = queue.Queue()
        silence_start: float | None = None
        speech_started = False
        start_time = time.time()
        result: list[Transcription] = []

        def callback(indata, frames, time_info, status):
            if status:
                logger.debug("sounddevice status: %s", status)
            q.put(indata.copy())

        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", blocksize=chunk_samples, callback=callback):
                while time.time() - start_time < timeout:
                    try:
                        data = q.get(timeout=0.1)
                    except queue.Empty:
                        # check if we have speech and now silence_after elapsed
                        if speech_started and silence_start is not None and (time.time() - silence_start) > silence_after:
                            break
                        continue
                    # data is int16
                    is_speech = vad.is_speech(data.tobytes(), sample_rate)
                    # Convert to float32 for whisper later
                    if _HAS_NP:
                        f32 = (data.flatten().astype("float32") / 32768.0)
                    else:
                        f32 = data

                    if is_speech:
                        speech_started = True
                        silence_start = None
                        audio_chunks.append(f32)
                    else:
                        if speech_started:
                            # still collect a bit of trailing silence for natural cutoff
                            audio_chunks.append(f32)
                            if silence_start is None:
                                silence_start = time.time()
                            elif (time.time() - silence_start) > silence_after:
                                break
                        else:
                            # no speech yet — keep listening, but don't accumulate endless silence
                            # keep only last 0.5s of silence as preamble
                            if len(audio_chunks) > 16:  # ~0.5s at 30ms
                                audio_chunks.pop(0)
                            audio_chunks.append(f32)
                            # if we've waited timeout without speech, return empty
                            if (time.time() - start_time) > timeout and not speech_started:
                                break

                    # Early exit if we have long utterance (>15s) — avoid OOM
                    total_samples = sum(len(c) for c in audio_chunks)
                    if total_samples > sample_rate * 20:
                        break

        except Exception as exc:
            logger.warning("mic listen failed: %s", exc)
            return Transcription(text="", confidence=0.0)

        if not speech_started or not audio_chunks:
            return Transcription(text="", confidence=0.0)

        if _HAS_NP:
            audio = np.concatenate(audio_chunks).astype("float32")
        else:
            # fallback
            import array
            audio = b"".join(c.tobytes() if hasattr(c, "tobytes") else bytes(c) for c in audio_chunks)  # type: ignore

        # Trim leading/trailing silence (we already have VAD, but whisper's filter helps)
        return self.transcribe(audio, sample_rate)
