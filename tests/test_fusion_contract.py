import pytest
import torch


@pytest.mark.parametrize("pool", ["min", "max", "mean", "cls"])
def test_predictions_ignore_masked_padding(tiny_model, pool):
    model = tiny_model(pool)
    text, audio = torch.randn(1, 3, 8), torch.randn(1, 5, 8)
    with torch.inference_mode():
        single = model.forward_from_embeddings(text, audio).logits
        padded = model.forward_from_embeddings(
            torch.cat([text, torch.randn(1, 4, 8) * 100], dim=1),
            torch.cat([audio, torch.randn(1, 6, 8) * 100], dim=1),
            torch.tensor([[1, 1, 1, 0, 0, 0, 0]]),
            torch.tensor([[1] * 5 + [0] * 6]),
        ).logits
    torch.testing.assert_close(single, padded, atol=2e-6, rtol=2e-6)


def test_raw_waveform_batch_matches_single_clip_with_time_normalization(tiny_model):
    model = tiny_model()
    waveform = torch.randn(1, 64)
    ids, text_mask = torch.tensor([[1, 2, 3]]), torch.ones(1, 3, dtype=torch.long)
    padded_audio = torch.cat(
        [torch.cat([waveform, torch.full((1, 32), 999.0)], dim=1), torch.randn(1, 96)]
    )
    mask = torch.tensor([[1] * 64 + [0] * 32, [1] * 96])
    with torch.inference_mode():
        single = model(ids, text_mask, waveform).logits
        batch = model(
            ids.repeat(2, 1), text_mask.repeat(2, 1), padded_audio, mask
        ).logits
    torch.testing.assert_close(single[0], batch[0], atol=2e-6, rtol=2e-6)


def test_reject_empty_modality_instead_of_nan(tiny_model):
    model = tiny_model()
    with pytest.raises(ValueError, match="all-masked"):
        model.forward_from_embeddings(
            torch.randn(1, 2, 8),
            torch.randn(1, 2, 8),
            audio_attention_mask=torch.zeros(1, 2),
        )
    with pytest.raises(ValueError, match="mask shape"):
        model(
            torch.ones(1, 3, dtype=torch.long),
            torch.ones(1, 3),
            torch.ones(1, 64),
            torch.ones(1, 16),
        )


def test_frozen_encoders_stay_deterministic_and_trainable_encoder_gets_gradients(
    tiny_model,
):
    frozen = tiny_model().train()
    assert not frozen.text_encoder.training and not frozen.audio_encoder.training
    model = tiny_model(freeze=False).train()
    mask = torch.tensor([[1] * 64 + [0] * 32, [1] * 96])
    output = model(
        torch.ones(2, 3, dtype=torch.long), torch.ones(2, 3), torch.randn(2, 96), mask
    )
    output.logits.square().sum().backward()
    assert model.audio_encoder.conv.weight.grad is not None
    assert torch.isfinite(model.audio_encoder.conv.weight.grad).all()
