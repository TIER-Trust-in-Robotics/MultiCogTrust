"""Text-only and voice-only comparison models: what each signal says on its own.

Small heads on the mean-pooled frozen features of the deployed encoders, trained on
Sessions 1-3 by src/benchmark_controls.py; probabilities are averaged over three
seeds. They show each modality's reading next to the fused models and never feed
a deployed prediction.
"""

from __future__ import annotations

from pathlib import Path

from src.emotion_checkpoint import EMOTION_LABELS

ROOT = Path(__file__).resolve().parents[1]
CONTROLS_DIR = ROOT / "models/mvp/controls"
MODES = ("text", "audio")
SEEDS = (42, 43, 44)


def files(directory=CONTROLS_DIR):
    return [Path(directory) / f"{mode}_{seed}.pt" for mode in MODES for seed in SEEDS]


class ModalityControls:
    def __init__(self, directory=CONTROLS_DIR):
        import torch
        from torch import nn

        self.heads = {mode: [] for mode in MODES}
        for mode in MODES:
            for seed in SEEDS:
                checkpoint = torch.load(Path(directory) / f"{mode}_{seed}.pt", map_location="cpu", weights_only=False)
                if checkpoint.get("mode") != mode:
                    raise ValueError(f"{mode}_{seed}.pt holds a {checkpoint.get('mode')!r} control")
                head = nn.Sequential(
                    nn.Linear(checkpoint["input_dim"], checkpoint["hidden_dim"]), nn.LeakyReLU(),
                    nn.Dropout(checkpoint["dropout"]), nn.Linear(checkpoint["hidden_dim"], len(EMOTION_LABELS)),
                )
                head.load_state_dict(checkpoint["state_dict"])
                head.eval()
                self.heads[mode].append((head, checkpoint["mean"].float(), checkpoint["scale"].float()))

    @classmethod
    def load(cls, directory=CONTROLS_DIR):
        """The controls, or None when their files are not installed (the demo then hides those rows)."""
        return cls(directory) if all(path.exists() for path in files(directory)) else None

    def predict(self, pooled):
        """{"text": probabilities, "audio": probabilities} from mean-pooled encoder features (one vector each)."""
        import torch

        with torch.inference_mode():
            out = {}
            for mode, heads in self.heads.items():
                features = pooled[mode].float().cpu()
                p = torch.stack([head(((features - mean) / scale)[None]).softmax(-1)[0]
                                 for head, mean, scale in heads]).mean(0)
                out[mode] = {label: float(p[i]) for i, label in enumerate(EMOTION_LABELS)}
            return out
