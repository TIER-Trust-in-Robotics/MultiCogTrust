"""Causal per-speaker smoothing of utterance emotion probabilities.

IEMOCAP speakers tend to hold an emotion across consecutive lines, so each new
prediction p_t is combined with the speaker's previous smoothed state:

    q_t ∝ p_t · q_{t-1} ** (strength · (1 - max p_t) ** gate)

gate = 0 is plain log-linear smoothing. gate > 0 lets a confident new reading
override the context, which keeps most of the gain while still catching
emotion changes. Presets were chosen on out-of-fold Sessions 1-3 (train on two
sessions, predict the third); Session 4 was evaluated once per preset.

Session 4, deployed checkpoint, Whisper tiny.en text (abstentions count wrong):
    none      macro F1 0.723, accuracy on emotion changes 0.648
    balanced  macro F1 0.735, accuracy on emotion changes 0.618
    max_f1    macro F1 0.761, accuracy on emotion changes 0.539

Gains are larger on improvised dialogues (S4 0.726 -> 0.781 with max_f1) than
scripted ones (0.691 -> 0.716). IEMOCAP improvisations target one emotion per
scenario, so real conversations may shift more often than this data does.
"""

from __future__ import annotations

import numpy as np

from src.emotion_checkpoint import EMOTION_LABELS

PRESETS = {
    "balanced": {"strength": 3.0, "gate": 2.0},
    "max_f1": {"strength": 0.6, "gate": 0.0},
}


class SpeakerContext:
    """Keeps one smoothed state per speaker; call reset() when a conversation ends."""

    def __init__(self, strength: float = 3.0, gate: float = 2.0):
        if strength < 0 or gate < 0:
            raise ValueError("strength and gate must be non-negative")
        self.strength, self.gate = strength, gate
        self.states: dict[str, np.ndarray] = {}

    @classmethod
    def preset(cls, name: str) -> "SpeakerContext":
        return cls(**PRESETS[name])

    def reset(self, speaker: str | None = None):
        if speaker is None:
            self.states.clear()
        else:
            self.states.pop(speaker, None)

    def update(self, probabilities: dict[str, float], speaker: str = "default") -> dict[str, float]:
        p = np.array([probabilities[name] for name in EMOTION_LABELS], dtype=np.float64)
        if not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
            raise ValueError("Probabilities must be finite, non-negative and not all zero")
        p /= p.sum()
        state = self.states.get(speaker)
        if state is None:
            q = p
        else:
            weight = self.strength * (1 - p.max()) ** self.gate
            q = np.exp(np.log(p + 1e-8) + weight * np.log(state + 1e-8))
            q /= q.sum()
        self.states[speaker] = q
        return dict(zip(EMOTION_LABELS, q.tolist()))
