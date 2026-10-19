import json

import numpy as np
import pytest

from src.core.events import SpeechSegment
from src.emotion_checkpoint import PreprocessingConfig
from src.emotion_fusion_infer import EmotionPrediction
from src.live_demo import SCENARIOS, ComparisonProcessor, Robot, Study

TRUST_DIMENSIONS = {"competence", "integrity", "benevolence", "transparency"}


class FakeModel:
    def __init__(self, probabilities):
        self.preprocessing = PreprocessingConfig()
        self.provenance = {"kind": "test"}
        self.probabilities = probabilities
        self.seen = []

    def predict(self, text, audio, sample_rate):
        self.seen.append(text)
        label = max(self.probabilities, key=self.probabilities.get)
        return EmotionPrediction(label, list(self.probabilities).index(label), dict(self.probabilities))


class FakeRobot:
    """Speaks instantly and never overlaps, unless told to."""

    def __init__(self, echo=False):
        self.voice, self.speaking, self.line, self.echo, self.spoken = "test", False, None, echo, []

    def speak(self, text, on_done=None):
        self.line = text
        self.spoken.append(text)
        if on_done:
            on_done(1.0, 2.0)

    def overlaps(self, start, end):
        return self.echo


class FakeDashboard:
    def __init__(self):
        self.state, self.actions = {}, {}

    def update(self, **values):
        self.state.update(values)


def models():
    return (FakeModel({"anger": 0.1, "happiness": 0.6, "sadness": 0.1, "neutral": 0.2}),
            FakeModel({"anger": 0.7, "happiness": 0.1, "sadness": 0.1, "neutral": 0.1}))


def segment():
    return SpeechSegment(0.0, 1.0, 16000, np.zeros(16000, np.float32))


def run(processor, text):
    """Drive the real processor with a supplied transcript instead of Whisper."""
    original = processor.a.process
    processor.a.process = lambda seg: original(seg, text)
    try:
        return processor.process(segment())
    finally:
        processor.a.process = original


def scenarios():
    return json.loads(SCENARIOS.read_text())


def test_comparison_context_smoothing_and_reset():
    a, b = models()
    processor = ComparisonProcessor(None, a, b)
    first = run(processor, "Where is my bag?")
    second = run(processor, "I want a supervisor.")
    assert a.seen == ["Where is my bag?", "I want a supervisor."]
    assert b.seen == ["Where is my bag?", "Where is my bag? [SEP] I want a supervisor."]
    assert first["comparison"]["previous_text"] is None
    assert second["comparison"]["previous_text"] == "Where is my bag?"
    assert set(second["comparison"]["smoothed"]) == {"balanced", "max_f1"}
    processor.reset()
    assert run(processor, "Thank you.")["comparison"]["previous_text"] is None


def test_empty_transcript_keeps_context_and_skips_b():
    a, b = models()
    processor = ComparisonProcessor(None, a, b)
    run(processor, "One.")
    assert "comparison" not in run(processor, "")
    run(processor, "Two.")
    assert b.seen == ["One.", "One. [SEP] Two."]


def test_speech_overlapping_the_robot_is_never_scored():
    a, b = models()
    processor = ComparisonProcessor(None, a, b, robot=FakeRobot(echo=True))
    result = run(processor, "Excuse me. You are blocking my route.")
    assert result["status"] == "ignored_robot_speech" and result["emotion"] is None
    assert a.seen == [] and b.seen == []


def test_robot_echo_window():
    robot = Robot(silent=True)
    robot.windows = [(10.0, 12.0)]
    assert robot.overlaps(11.0, 11.5) and robot.overlaps(9.0, 10.5) and robot.overlaps(12.2, 13.0)
    assert not robot.overlaps(12.6, 14.0) and not robot.overlaps(5.0, 9.5)


def test_hri_scenarios_are_well_formed():
    items = scenarios()
    assert len({s["id"] for s in items}) == len(items) >= 10
    for s in items:
        assert {"id", "title", "robot", "setting", "turns", "probe", "trust"} <= s.keys()
        assert s["turns"] and all(isinstance(t, str) and t for t in s["turns"])
        assert s["trust"] in TRUST_DIMENSIONS


def test_consented_trial_is_recorded_end_to_end(tmp_path):
    a, b = models()
    processor = ComparisonProcessor(None, a, b)
    robot = FakeRobot()
    study = Study(FakeDashboard(), processor, robot, scenarios(), {"note": "test"}, root=tmp_path)
    with pytest.raises(ValueError):
        study.start_session({"participant": "P01", "consent": False})
    study.start_session({"participant": "P01", "consent": True})
    first = scenarios()[0]
    study.start_trial({"scenario": first["id"]})
    study.robot_next({})
    result = run(processor, "Oh, sorry, I'll move.")
    result["event"] = "prediction"
    study.on_result(result)
    study.end_trial({})
    with pytest.raises(ValueError):
        study.report({"feeling": "joy", "trust": 9})
    study.report({"feeling": "anger", "trust": 3, "note": "pushy robot"})
    study.end_session({})
    folder = next(tmp_path.iterdir())
    events = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == ["session_start", "trial_start", "robot_turn", "utterance",
                                            "trial_end", "self_report", "session_end"]
    assert robot.spoken == [first["turns"][0]]
    utterance = events[3]
    assert utterance["text"] == "Oh, sorry, I'll move." and utterance["comparison"]["B"]["probabilities"]
    assert (folder / utterance["audio"]).exists()
    assert events[5]["feeling"] == "anger" and events[5]["trust"] == 3
    assert json.loads((folder / "session.json").read_text())["participant"] == "P01"


def test_without_a_session_nothing_is_written(tmp_path):
    a, b = models()
    processor = ComparisonProcessor(None, a, b)
    study = Study(FakeDashboard(), processor, FakeRobot(), scenarios(), {}, root=tmp_path)
    study.start_trial({"scenario": scenarios()[0]["id"]})
    study.robot_next({})
    result = run(processor, "Hello.")
    result["event"] = "prediction"
    study.on_result(result)
    study.end_trial({})
    assert "recording" in result and result["recording"] is None
    assert study.state["awaiting_report"] is None
    assert list(tmp_path.iterdir()) == []


class FakeShared:
    """Stands in for SharedEncoders: both models plus the single-signal readings."""

    def __init__(self, a, b):
        self.a, self.b, self.controls_seen = a, b, []

    def predict(self, text_a, text_b, audio, sample_rate, controls=None):
        self.controls_seen.append(controls)
        out = {"A": (self.a.predict(text_a, audio, sample_rate), 5.0), "B": (self.b.predict(text_b, audio, sample_rate), 6.0)}
        if controls is not None:
            out["modalities"] = (controls.predict(None), 0.3)
        return out


class FakeControls:
    def predict(self, pooled):
        return {"text": {"anger": 0.1, "happiness": 0.1, "sadness": 0.1, "neutral": 0.7},
                "audio": {"anger": 0.8, "happiness": 0.1, "sadness": 0.05, "neutral": 0.05}}


def test_each_signal_is_reported_beside_the_combined_models():
    a, b = models()
    processor = ComparisonProcessor(None, a, b, shared=FakeShared(a, b), controls=FakeControls())
    comparison = run(processor, "Fine.")["comparison"]
    assert comparison["modalities"]["text"]["neutral"] == 0.7 and comparison["modalities"]["audio"]["anger"] == 0.8
    assert comparison["modalities"]["ms"] == 0.3 and set(comparison) >= {"A", "B", "smoothed"}
    without = ComparisonProcessor(None, a, b, shared=FakeShared(a, b))  # controls not installed
    assert "modalities" not in run(without, "Fine.")["comparison"]


def test_missing_control_files_hide_the_rows_instead_of_failing(tmp_path):
    from src.modality_controls import ModalityControls, files

    assert ModalityControls.load(tmp_path) is None
    assert len(files(tmp_path)) == 6


def test_robot_voice_follows_the_platform():
    from src.live_demo import SAPI_SCRIPT, speech_engine

    assert (speech_engine("darwin"), speech_engine("win32"), speech_engine("linux")) == ("say", "sapi", None)
    args, stdin, env = Robot(platform="darwin").command("Hello.")
    assert args == ["say", "-v", "Samantha", "-r", "170", "Hello."] and stdin is None
    args, stdin, env = Robot("Microsoft Zira Desktop", 220, platform="win32").command("Café, s'il vous plaît.", "out.wav")
    assert args[0] == "powershell" and args[-1] == SAPI_SCRIPT
    assert stdin == "Café, s'il vous plaît." and env == {"ROBOT_VOICE": "Microsoft Zira Desktop", "ROBOT_RATE": "2", "ROBOT_WAV": "out.wav"}
    assert Robot(platform="win32").command("Hi.")[2]["ROBOT_VOICE"] == ""  # system default voice


def test_robot_without_a_speech_engine_is_timed_silently(monkeypatch):
    import time as clock

    robot = Robot(platform="linux")
    assert robot.engine is None and robot.silent
    assert robot.duration("one two three four five six seven eight nine ten") == pytest.approx(60 * 10 / 170)
    monkeypatch.setattr(robot, "duration", lambda text: 0.05)
    done = []
    robot.speak("Hello there.", on_done=lambda start, end: done.append(end - start))
    for _ in range(100):
        if done:
            break
        clock.sleep(0.01)
    assert done and done[0] >= 0.05 and robot.overlaps(*robot.windows[0])
