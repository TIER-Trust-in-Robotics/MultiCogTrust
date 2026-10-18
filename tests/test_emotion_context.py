import numpy as np
import pytest

from src.emotion_checkpoint import EMOTION_LABELS
from src.emotion_context import SpeakerContext


def probs(*values):
    return dict(zip(EMOTION_LABELS, values))


def test_first_utterance_passes_through():
    context = SpeakerContext.preset("balanced")
    out = context.update(probs(0.1, 0.2, 0.3, 0.4))
    assert np.allclose(list(out.values()), [0.1, 0.2, 0.3, 0.4])


def test_context_pulls_an_uncertain_reading_toward_the_speaker_state():
    context = SpeakerContext.preset("max_f1")
    context.update(probs(0.9, 0.04, 0.03, 0.03))
    out = context.update(probs(0.3, 0.2, 0.1, 0.4))
    assert max(out, key=out.get) == "anger"
    assert abs(sum(out.values()) - 1) < 1e-9


def test_gate_lets_a_confident_reading_override_context():
    gated, plain = SpeakerContext(3.0, 2.0), SpeakerContext(3.0, 0.0)
    for context in (gated, plain):
        context.update(probs(0.97, 0.01, 0.01, 0.01))
    confident = probs(0.02, 0.02, 0.94, 0.02)
    assert max(gated.update(confident).items(), key=lambda kv: kv[1])[0] == "sadness"
    assert max(plain.update(confident).items(), key=lambda kv: kv[1])[0] == "anger"


def test_speakers_are_independent_and_reset_clears_state():
    context = SpeakerContext.preset("max_f1")
    context.update(probs(0.9, 0.04, 0.03, 0.03), speaker="F")
    out = context.update(probs(0.1, 0.2, 0.3, 0.4), speaker="M")
    assert np.allclose(list(out.values()), [0.1, 0.2, 0.3, 0.4])
    context.reset("F")
    assert "F" not in context.states and "M" in context.states


def test_rejects_invalid_probabilities():
    with pytest.raises(ValueError):
        SpeakerContext().update(probs(0, 0, 0, 0))


def test_live_pipeline_context_is_opt_in():
    from src.pipeline_demo import parse_args

    assert parse_args([]).context == "off"
    assert parse_args(["--context", "balanced"]).context == "balanced"
    with pytest.raises(SystemExit):
        parse_args(["--context", "unknown"])
