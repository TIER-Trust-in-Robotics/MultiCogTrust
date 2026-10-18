from types import SimpleNamespace

import numpy as np
import pytest
import torch

from conftest import TinyTokenizer
from src.core.events import SpeechSegment
from src.emotion_checkpoint import (
    PreprocessingConfig,
    prepare_audio,
    save_checkpoint,
    read_checkpoint,
    load_checkpoint_weights,
    validate_checkpoint,
)
from src.inference_pipeline import SpeechProcessor


@pytest.mark.parametrize("full", [True, False])
def test_checkpoint_roundtrip_predictions(tiny_model, tmp_path, full):
    model = tiny_model()
    path = tmp_path / "emotion.pt"
    save_checkpoint(
        path,
        model,
        preprocessing=PreprocessingConfig(),
        provenance={"kind": "test"},
        include_encoders=full,
    )
    loaded = tiny_model()
    load_checkpoint_weights(loaded, read_checkpoint(path, "iemocap_emotion_4"))
    text, audio = torch.randn(1, 3, 8), torch.randn(1, 8, 8)
    torch.testing.assert_close(
        model.forward_from_embeddings(text, audio).logits,
        loaded.forward_from_embeddings(text, audio).logits,
    )


def test_incompatible_or_incomplete_checkpoint_fails(tiny_model, tmp_path):
    with pytest.raises(ValueError, match="MOSEI"):
        validate_checkpoint({"num_labels": 7, "trainable_state_dict": {}})
    path = tmp_path / "emotion.pt"
    model = tiny_model()
    save_checkpoint(
        path,
        model,
        preprocessing=PreprocessingConfig(),
        provenance={},
        include_encoders=False,
    )
    checkpoint = read_checkpoint(path)
    checkpoint["model_state_dict"].pop("classifier.bias")
    with pytest.raises(ValueError, match="exactly match"):
        load_checkpoint_weights(model, checkpoint)
    checkpoint["label_names"] = ["neutral", "anger", "sadness", "happiness"]
    with pytest.raises(ValueError, match="mapping"):
        validate_checkpoint(checkpoint)


def test_audio_preparation_matches_rate_and_handles_invalid_input():
    config = PreprocessingConfig(max_audio_sec=0.1)
    waveform = prepare_audio(np.ones((22050, 2), dtype=np.float32), 22050, config)
    assert len(waveform) == 1600 and waveform.dtype == torch.float32
    assert len(prepare_audio(np.zeros(5, dtype=np.float32), 16000, config)) == 1024
    for value in [
        np.array([], dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([1], dtype=np.int16),
    ]:
        with pytest.raises(ValueError):
            prepare_audio(value, 16000, config)


def test_predictor_replay_and_live_segment_share_processing(
    tiny_model, tmp_path, monkeypatch
):
    from src.emotion_fusion_infer import CrossModalEmotionPredictor

    path = tmp_path / "emotion.pt"
    save_checkpoint(
        path,
        tiny_model(),
        preprocessing=PreprocessingConfig(max_audio_sec=0.1),
        provenance={"kind": "test"},
    )
    monkeypatch.setattr(
        "src.emotion_fusion_infer.AutoTokenizer.from_pretrained",
        lambda *a, **k: TinyTokenizer(),
    )
    predictor = CrossModalEmotionPredictor(path, local_files_only=True)
    audio = np.sin(np.arange(1600) / 11).astype(np.float32)

    class ASR:
        def transcribe(self, samples, **kwargs):
            assert len(samples) == 1600
            return iter([SimpleNamespace(text="hello there")]), None

    processor = SpeechProcessor(ASR(), predictor)
    live = processor.process(SpeechSegment(123.0, 123.1, 16000, audio))
    replay = processor.process(SpeechSegment(0.0, 0.1, 16000, audio), "hello there")
    assert live["emotion"] == replay["emotion"]
    assert (
        live["transcript_source"] == "asr" and replay["transcript_source"] == "provided"
    )
    assert "trust" not in live["emotion"]
    assert sum(live["emotion"]["probabilities"].values()) == pytest.approx(1.0)
    empty = processor.process(SpeechSegment(0.0, 0.1, 16000, audio), " ")
    assert empty["status"] == "empty_transcript" and empty["emotion"] is None
