"""Shared four-class emotion predictor for microphone and recorded audio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoTokenizer

from src.emotion_checkpoint import (
    PreprocessingConfig,
    load_checkpoint_weights,
    prepare_audio,
    read_checkpoint,
)
from src.models import CrossModalTransformerFusionClassifier


@dataclass
class EmotionPrediction:
    label: str
    label_index: int
    probabilities: dict[str, float]


class CrossModalEmotionPredictor:
    def __init__(
        self,
        checkpoint_path="models/cross_modal_finetune_iemocap.pt",
        device="cpu",
        local_files_only=False,
    ):
        self.device = device
        checkpoint = read_checkpoint(checkpoint_path, expected_task="iemocap_emotion_4")
        self.label_names = checkpoint["label_names"]
        self.provenance = checkpoint["provenance"]
        self.preprocessing = PreprocessingConfig(**checkpoint["preprocessing"])
        config = checkpoint["model_config"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["text_model_name"],
            revision=config.get("text_revision"),
            local_files_only=local_files_only,
        )
        self.model = CrossModalTransformerFusionClassifier(
            **config,
            freeze_text_encoder=True,
            freeze_audio_encoder=True,
            local_files_only=local_files_only,
        )
        load_checkpoint_weights(self.model, checkpoint)
        self.model.to(device).eval()

    @torch.inference_mode()
    def predict(
        self, text: str, audio: np.ndarray, sample_rate=16000
    ) -> EmotionPrediction:
        if not text.strip():
            raise ValueError("Emotion prediction requires a nonempty transcript")
        waveform = (
            prepare_audio(audio, sample_rate, self.preprocessing)
            .unsqueeze(0)
            .to(self.device)
        )
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.preprocessing.max_tokens,
        )
        output = self.model(
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
            waveform,
        )
        probabilities = output.logits.softmax(dim=-1)[0].cpu()
        index = int(probabilities.argmax())
        return EmotionPrediction(
            self.label_names[index],
            index,
            {name: float(probabilities[i]) for i, name in enumerate(self.label_names)},
        )
