"""ARC voice package — always-listening, full-duplex, local on RTX 4050.

Exposes STT/TTS/VAD/session for `arc voice`.
"""

from .stt import STTProvider, FasterWhisperSTT, FakeSTT
from .tts import TTSProvider, KokoroTTS, PiperTTS, Pyttsx3TTS, FakeTTS
from .vad import VADDetector
from .session import VoiceSession, VoiceApprover

__all__ = [
    "STTProvider", "FasterWhisperSTT", "FakeSTT",
    "TTSProvider", "KokoroTTS", "PiperTTS", "Pyttsx3TTS", "FakeTTS",
    "VADDetector",
    "VoiceSession", "VoiceApprover",
]
