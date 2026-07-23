"""Live inference wrapper for the cross-modal emotion model.

Loads a checkpoint trained by ``train_cross_modal_finetune_iemocap.py`` and
exposes a single ``predict(text, audio)`` call returning the 4-class emotion
distribution plus a coarse trust proxy for the HRI downstream.

The audio input is a 16 kHz mono float32 waveform (exactly what the VAD pushes
as ``speachSegment.samples``), so this drops straight into the live pipeline.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import CrossModalTransformerFusionClassifier  # noqa: E402

SAMPLE_RATE = 16000
# DistilHuBERT's conv stack downsamples ~320x; keep a small floor so very short
# clips still yield at least a couple of frames.
MIN_SAMPLES = 1024

# Coarse mapping from emotion -> trust contribution in [0, 1], a placeholder for
# the HRI trust head. Neutral/happiness read as more trust-conducive than
# anger/sadness. Tune or replace with a learned trust head downstream.
TRUST_WEIGHTS = {"anger": 0.0, "happiness": 1.0, "sadness": 0.3, "neutral": 0.7}


@dataclass
class EmotionPrediction:
    label: str
    label_index: int
    probabilities: dict[str, float]
    trust_proxy: float


class CrossModalEmotionPredictor:
    def __init__(
        self,
        checkpoint_path: str = "models/cross_modal_finetune_iemocap.pt",
        device: str = "cpu",
        text_model_name: str = "distilbert-base-uncased",
        audio_model_name: str = "ntu-spml/distilhubert",
    ) -> None:
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.label_names: list[str] = checkpoint["label_names"]

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.model = CrossModalTransformerFusionClassifier(
            num_labels=len(self.label_names),
            text_model_name=text_model_name,
            audio_model_name=audio_model_name,
            fusion_head_output_type=checkpoint.get("fusion_head", "min"),
            freeze_text_encoder=True,
            freeze_audio_encoder=True,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device).eval()

    @torch.no_grad()
    def predict(self, text: str, audio: np.ndarray) -> EmotionPrediction:
        waveform = torch.as_tensor(np.asarray(audio, dtype=np.float32))
        if waveform.numel() < MIN_SAMPLES:
            waveform = F.pad(waveform, (0, MIN_SAMPLES - waveform.numel()))
        waveform = waveform.unsqueeze(0).to(self.device)

        encoded = self.tokenizer(
            text or "",
            return_tensors="pt",
            truncation=True,
            max_length=297,
        )
        output = self.model(
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device),
            waveform,
        )
        probabilities = F.softmax(output.logits, dim=-1).squeeze(0).cpu()
        index = int(probabilities.argmax())
        prob_map = {
            name: float(probabilities[i]) for i, name in enumerate(self.label_names)
        }
        trust_proxy = sum(
            prob_map[name] * TRUST_WEIGHTS.get(name, 0.0) for name in self.label_names
        )
        return EmotionPrediction(
            label=self.label_names[index],
            label_index=index,
            probabilities=prob_map,
            trust_proxy=float(trust_proxy),
        )
