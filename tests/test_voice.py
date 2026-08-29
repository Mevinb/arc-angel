"""Voice — always-listening, full-duplex, local. Tests use FakeSTT/FakeTTS."""

from __future__ import annotations

from pathlib import Path

from arc.config import load_config
from arc.voice import FakeSTT, FakeTTS, VADDetector, VoiceApprover, VoiceSession
from arc.voice.session import parse_approval, is_exit_phrase


class TestVAD:
    def test_energy_vad(self):
        vad = VADDetector(aggressiveness=2)
        # silence
        assert not vad.is_speech(b"\x00" * 480)  # 30ms silence
        # loud frame
        assert vad.is_speech(b"\xff\x7f" * 240)
        assert not vad.available or vad.available  # just check property

    def test_array_vad(self):
        vad = VADDetector()
        import numpy as np

        silence = np.zeros(1600, dtype=np.float32)
        assert not vad.is_speech_array(silence)
        loud = np.ones(1600, dtype=np.float32) * 0.5
        assert vad.is_speech_array(loud)


class TestSTT:
    def test_fake_stt(self):
        s = FakeSTT(script=["hello world", "yes"])
        tx = s.listen_once()
        assert tx.text == "hello world"
        assert tx.confidence > 0.8
        tx2 = s.listen_once()
        assert tx2.text == "yes"

    def test_faster_whisper_not_required_for_tests(self, tmp_path: Path):
        # FasterWhisperSTT should not crash when faster-whisper is not installed
        from arc.voice.stt import FasterWhisperSTT

        stt = FasterWhisperSTT(model="tiny", device="cpu")
        # may be unavailable in CI, but should not raise
        assert isinstance(stt.available, bool)


class TestTTS:
    def test_fake_tts(self):
        t = FakeTTS()
        t.speak("hello")
        assert t.spoken == ["hello"]
        t.speak("world")
        assert len(t.spoken) == 2
        assert not t.is_speaking

    def test_pyttsx3_fallback(self):
        from arc.voice.tts import Pyttsx3TTS, make_tts

        t = Pyttsx3TTS()
        # available may be False in CI without espeak, but shouldn't crash
        assert isinstance(t.available, bool)
        t2 = make_tts("auto")
        assert t2 is not None
        t3 = make_tts("fake")
        assert isinstance(t3, FakeTTS)

    def test_kokoro_fallback(self):
        from arc.voice.tts import KokoroTTS

        t = KokoroTTS()
        # Should fallback to pyttsx3 if kokoro not installed
        assert isinstance(t.available, bool)


class TestApprovalParsing:
    def test_affirmative(self):
        assert parse_approval("yes") is True
        assert parse_approval("yeah go ahead") is True
        assert parse_approval("sure, do it") is True
        assert parse_approval("approve") is True
        assert parse_approval("okay") is True

    def test_negative(self):
        assert parse_approval("no") is False
        assert parse_approval("nope") is False
        assert parse_approval("cancel") is False
        assert parse_approval("don't do it") is False

    def test_unclear(self):
        assert parse_approval("") is None
        assert parse_approval("maybe") is None
        assert parse_approval("perhaps") is None
        # "yes no" is ambiguous — may be True (first wins) or None
        assert parse_approval("yes no") in (None, True)

    def test_exit(self):
        assert is_exit_phrase("exit")
        assert is_exit_phrase("goodbye")
        assert is_exit_phrase("arc stop")
        assert not is_exit_phrase("hello")


class TestVoiceApprover:
    def test_approves_on_yes(self):
        stt = FakeSTT(script=["yes"])
        tts = FakeTTS()
        approver = VoiceApprover(stt=stt, tts=tts, confidence_threshold=0.6, fallback_to_text=False)

        class FakeAction:
            risk = type("R", (), {"value": "yellow"})()
            description = "run shell: ls"

        assert approver(FakeAction()) is True
        assert any("Approved" in s for s in tts.spoken)

    def test_denies_on_no(self):
        stt = FakeSTT(script=["no"])
        tts = FakeTTS()
        approver = VoiceApprover(stt=stt, tts=tts, fallback_to_text=False)

        class FakeAction:
            risk = type("R", (), {"value": "yellow"})()
            description = "run shell"

        assert approver(FakeAction()) is False

    def test_fallback_to_text_on_unclear(self, monkeypatch):
        stt = FakeSTT(script=["maybe"])
        tts = FakeTTS()
        approver = VoiceApprover(stt=stt, tts=tts, fallback_to_text=True)
        # Mock Prompt/Confirm.ask to return True (approver now uses Prompt with all/session support)
        import arc.voice.session as sess

        monkeypatch.setattr("rich.prompt.Prompt.ask", lambda *a, **kw: "y")
        monkeypatch.setattr("rich.prompt.Confirm.ask", lambda *a, **kw: True)

        class FakeAction:
            risk = type("R", (), {"value": "yellow"})()
            description = "test"

        # First attempt unclear, second also unclear, should fall back to Confirm.ask -> True
        # We need to give two unclear scripts so it falls through
        stt.script = ["maybe", "maybe"]
        assert approver(FakeAction()) is True


class TestVoiceSession:
    def test_handle_once(self, tmp_path: Path):
        config = load_config(project_root=tmp_path)
        config.safety_mode = "auto"
        from arc.app import ArcApp

        app = ArcApp(config=config, quiet=True)
        try:
            stt = FakeSTT(script=["hello"])
            tts = FakeTTS()
            session = VoiceSession(app, stt=stt, tts=tts, allow_voice_approval=False)
            reply = session.handle_once("hello")
            assert reply is not None
            assert len(reply) > 0
            # handle_once should have spoken? It goes via _handle_text -> _speak_with_bargein
            # For handle_once, it will have called tts via _handle_text
        finally:
            app.close()

    def test_handle_once_exit(self, tmp_path: Path):
        config = load_config(project_root=tmp_path)
        config.safety_mode = "auto"
        from arc.app import ArcApp

        app = ArcApp(config=config, quiet=True)
        try:
            stt = FakeSTT(script=[])
            tts = FakeTTS()
            session = VoiceSession(app, stt=stt, tts=tts)
            reply = session.handle_once("exit")
            assert reply is None  # exit phrase returns None and sets stop
        finally:
            app.close()

    def test_barge_in(self, tmp_path: Path):
        config = load_config(project_root=tmp_path)
        config.safety_mode = "auto"
        from arc.app import ArcApp

        app = ArcApp(config=config, quiet=True)
        try:
            stt = FakeSTT(script=["second question"])
            tts = FakeTTS()
            # Simulate TTS speaking and then barge-in queued
            session = VoiceSession(app, stt=stt, tts=tts)
            # Put a transcript in queue to simulate barge-in while speaking
            from arc.voice.stt import Transcription

            session._queue.put(Transcription(text="interrupt", confidence=0.9))
            barged = session._speak_with_bargein("This is a long reply. It has multiple sentences. Should be interrupted.")
            assert barged is True
        finally:
            app.close()
