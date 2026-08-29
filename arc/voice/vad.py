"""VAD — webrtcvad wrapper for always-listening.

Used to gate the always-listening loop so we don't feed silence to Whisper.
Falls back to a no-op (always True) when webrtcvad is not installed.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("arc.voice.vad")

try:
    import webrtcvad  # type: ignore
    _HAS_WEBRTCVAD = True
except ImportError:
    webrtcvad = None  # type: ignore
    _HAS_WEBRTCVAD = False

try:
    import numpy as np  # type: ignore
    _HAS_NP = True
except ImportError:
    np = None  # type: ignore
    _HAS_NP = False


class VADDetector:
    """Voice Activity Detection for always-listening.

    Uses webrtcvad when available, otherwise a simple energy detector.
    All audio is expected as 16kHz mono int16 or float32.
    """

    def __init__(self, aggressiveness: int = 3, sample_rate: int = 16000) -> None:
        self.aggressiveness = max(0, min(3, aggressiveness))
        self.sample_rate = sample_rate
        self._vad = None
        if _HAS_WEBRTCVAD:
            try:
                self._vad = webrtcvad.Vad(self.aggressiveness)
                logger.info("webrtcvad VAD aggressiveness=%d", self.aggressiveness)
            except Exception as exc:
                logger.warning("webrtcvad init failed: %s — falling back to energy", exc)
                self._vad = None
        else:
            logger.info("webrtcvad not installed — using energy VAD (pip install webrtcvad)")

        # Energy threshold for fallback (tuned for 16-bit PCM)
        self.energy_threshold = 500

    def is_speech(self, frame: bytes, sample_rate: int | None = None) -> bool:
        """Return True if frame contains speech.

        Args:
            frame: 10/20/30ms of 16kHz mono PCM (int16 bytes). webrtcvad
                   requires 10, 20 or 30 ms frames. For other sizes we fall
                   back to energy detection.
            sample_rate: overrides instance sample_rate if given
        """
        sr = sample_rate or self.sample_rate
        if self._vad is not None and sr in (8000, 16000, 32000, 48000):
            # webrtcvad requires 10/20/30 ms
            frame_len_ms = len(frame) / 2 / sr * 1000  # int16 = 2 bytes
            if frame_len_ms in (10, 20, 30):
                try:
                    return bool(self._vad.is_speech(frame, sr))
                except Exception:
                    pass
        # Fallback: energy detector
        return self._energy_is_speech(frame)

    def _energy_is_speech(self, frame: bytes) -> bool:
        if not frame:
            return False
        if _HAS_NP:
            try:
                arr = np.frombuffer(frame, dtype=np.int16).astype(float)
                if len(arr) == 0:
                    return False
                rms = float((arr ** 2).mean() ** 0.5)
                return rms > self.energy_threshold
            except Exception:
                pass
        # Last resort: check if any non-zero bytes
        return any(b != 0 for b in frame[:100])

    def is_speech_array(self, audio: "np.ndarray", threshold: float = 0.008) -> bool:
        """Convenience for float32 [-1,1] arrays (faster-whisper style)."""
        if not _HAS_NP or audio is None or len(audio) == 0:
            return False
        try:
            rms = float((audio.astype(float) ** 2).mean() ** 0.5)
            return rms > threshold
        except Exception:
            return False

    @property
    def available(self) -> bool:
        return _HAS_WEBRTCVAD
