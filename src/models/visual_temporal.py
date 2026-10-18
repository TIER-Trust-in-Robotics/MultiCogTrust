"""Small temporal visual classifier and explicit audio/text late fusion."""

from __future__ import annotations

import torch
from torch import nn

MODES = ("blendshapes", "image", "combined")


def fit_normalization(rows, std_floor=0.001):
    """Caller supplies training rows only; unavailable frames never enter statistics."""
    statistics = {}
    for field, mask in (
        ("blendshapes", "blendshape_mask"),
        ("image_embeddings", "image_mask"),
    ):
        values = torch.cat([r[field][r[mask]] for r in rows]).double()
        if not len(values):
            raise ValueError(f"No available training frames for {field}")
        statistics[field] = {
            "mean": values.mean(0).float(),
            "std": values.std(0, unbiased=False).clamp_min(std_floor).float(),
            "frames": len(values),
        }
    return statistics


class VisualTemporalClassifier(nn.Module):
    def __init__(self, mode, hidden_dim=64, dropout=0.2):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown visual mode: {mode}")
        self.mode = mode
        self.config = {"mode": mode, "hidden_dim": hidden_dim, "dropout": dropout}
        for field, width in (("blendshapes", 52), ("image_embeddings", 576)):
            self.register_buffer(field + "_mean", torch.zeros(width))
            self.register_buffer(field + "_std", torch.ones(width))
        width = {"blendshapes": 52, "image": 576, "combined": 628}[mode]
        # Availability and relative time are explicit, including gaps inside a sequence.
        self.project = nn.Sequential(
            nn.Linear(width + 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )

    def set_normalization(self, statistics):
        for field in ("blendshapes", "image_embeddings"):
            for kind in ("mean", "std"):
                getattr(self, field + "_" + kind).copy_(statistics[field][kind])

    def forward(self, batch):
        sequence = batch["sequence_mask"]
        bm = batch["blendshape_mask"] & sequence
        im = batch["image_mask"] & sequence
        if self.mode == "blendshapes":
            im = torch.zeros_like(im)
        elif self.mode == "image":
            bm = torch.zeros_like(bm)
        valid = bm | im
        values = []
        for field, mask, enabled in (
            ("blendshapes", bm, self.mode != "image"),
            ("image_embeddings", im, self.mode != "blendshapes"),
        ):
            if enabled:
                normalized = (batch[field] - getattr(self, field + "_mean")) / getattr(
                    self, field + "_std"
                )
                # Mask after normalization; arbitrary missing/padding values cannot leak.
                values.append(torch.where(mask[..., None], normalized, 0.0))
        times = torch.where(sequence, batch["times_sec"] / 6.0, 0.0)
        values.append(torch.stack([bm.float(), im.float(), times], -1))
        projected = self.project(torch.cat(values, -1))
        lengths = sequence.sum(1)
        if bool((lengths < 1).any()):
            raise ValueError("Visual sequences require at least one masked placeholder")
        packed = nn.utils.rnn.pack_padded_sequence(
            projected, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoded, hidden = self.temporal(packed)
        encoded, _ = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, total_length=sequence.shape[1]
        )
        pooled = (encoded * valid[..., None]).sum(1) / valid.sum(
            1, keepdim=True
        ).clamp_min(1)
        logits = self.head(torch.cat([pooled, hidden[-1]], -1))
        return logits, valid.any(1)


def fuse_probabilities(base, visual, available, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("Fusion weight must be between zero and one")
    mixed = (1 - alpha) * base + alpha * visual
    return torch.where(available[:, None], mixed, base)


def augment_visual(batch, mode, frame_dropout, stream_dropout):
    result = dict(batch)
    bm, im = batch["blendshape_mask"].clone(), batch["image_mask"].clone()
    keep = torch.rand_like(bm, dtype=torch.float32) >= frame_dropout
    bm &= keep
    im &= keep
    if mode == "combined":
        bm &= torch.rand(len(bm), 1, device=bm.device) >= stream_dropout
        im &= torch.rand(len(im), 1, device=im.device) >= stream_dropout
    result.update(blendshape_mask=bm, image_mask=im)
    return result
