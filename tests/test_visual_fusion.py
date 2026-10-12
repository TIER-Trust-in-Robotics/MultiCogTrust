import numpy as np
import torch

from src.models.visual_temporal import (
    VisualTemporalClassifier,
    fit_normalization,
    fuse_probabilities,
)
from src.visual_fusion import visual_row, save_visual_checkpoint, load_visual_checkpoint
from src.visual_feature_dataset import collate_visual


def row(length=5):
    rng = np.random.default_rng(length)
    return visual_row(
        rng.normal(size=(length, 52)).astype(np.float32),
        rng.normal(size=(length, 576)).astype(np.float32),
        np.arange(length) * 0.2,
        [True] * length,
        [True] * length,
    )


def test_missing_visual_and_padding_cannot_change_available_predictions():
    torch.manual_seed(2)
    model = VisualTemporalClassifier("combined").eval()
    sample = row(3)
    single = collate_visual([sample])
    batch = collate_visual([sample, row(8)])
    with torch.no_grad():
        alone, _ = model(single)
        padded, _ = model(batch)
    torch.testing.assert_close(alone[0], padded[0], atol=1e-6, rtol=1e-6)
    batch["image_mask"].zero_()
    batch["blendshape_mask"].zero_()
    batch["image_embeddings"].fill_(100000)
    logits, available = model(batch)
    assert torch.isfinite(logits).all() and not available.any()
    base = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
    assert torch.equal(
        fuse_probabilities(base, logits.softmax(-1), available, 0.75), base
    )


def test_normalization_ignores_missing_values_and_validation_rows():
    training = row(3)
    training["image_mask"][0] = False
    training["image_embeddings"][0].fill_(10000)
    stats = fit_normalization([training])
    assert stats["image_embeddings"]["frames"] == 2
    torch.testing.assert_close(
        stats["image_embeddings"]["mean"], training["image_embeddings"][1:].mean(0)
    )


def test_checkpoint_roundtrip_and_empty_sequence_fallback(tmp_path):
    model = VisualTemporalClassifier("image").eval()
    path = tmp_path / "visual.pt"
    save_visual_checkpoint(path, model, 0.25, {"test_evaluated": False}, {})
    restored, payload = load_visual_checkpoint(path)
    batch = collate_visual([row()])
    with torch.no_grad():
        torch.testing.assert_close(model(batch)[0], restored(batch)[0])
    empty = visual_row([], [], [], [], [])
    logits, available = restored(collate_visual([empty]))
    assert not available.any() and torch.isfinite(logits).all()
    assert payload["alpha"] == 0.25


def test_combined_head_can_use_either_stream_without_the_other():
    model = VisualTemporalClassifier("combined").eval()
    batch = collate_visual([row()])
    batch["blendshape_mask"].zero_()
    with torch.no_grad():
        expected, available = model(batch)
        batch["blendshapes"].fill_(100000)
        actual, _ = model(batch)
    assert available.all()
    torch.testing.assert_close(actual, expected)
