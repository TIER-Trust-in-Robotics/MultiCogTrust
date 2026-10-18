from types import SimpleNamespace

import pytest
import torch
from torch import nn


class TinyEncoder(nn.Module):
    """Offline encoder with time-dependent normalization, like HuBERT's front end."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, _commit_hash="test-revision")
        self.embedding = nn.Embedding(32, 8)
        self.conv = nn.Conv1d(1, 8, kernel_size=4, stride=4)
        self.norm = nn.GroupNorm(8, 8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, audio=None, input_ids=None, attention_mask=None):
        if input_ids is not None:
            x = self.embedding(input_ids)
        else:
            x = self.norm(self.conv(audio.unsqueeze(1))).transpose(1, 2)
        return SimpleNamespace(last_hidden_state=self.dropout(x))


class TinyTokenizer:
    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        lengths = [
            min(len(text.split()) + 2, kwargs.get("max_length", 297)) for text in texts
        ]
        ids = torch.zeros(len(texts), max(lengths), dtype=torch.long)
        for i, length in enumerate(lengths):
            ids[i, :length] = torch.arange(1, length + 1) % 31
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}


@pytest.fixture
def tiny_model(monkeypatch):
    from src.models.cross_modal_transformer import CrossModalTransformerFusionClassifier

    monkeypatch.setattr(
        "src.models.cross_modal_transformer.AutoModel.from_pretrained",
        lambda *a, **k: TinyEncoder(),
    )

    def make(pool="min", freeze=True):
        torch.manual_seed(17)
        return CrossModalTransformerFusionClassifier(
            4,
            fusion_dim=8,
            num_attention_heads=2,
            dropout=0,
            linear_layer_output=(8,),
            fusion_head_output_type=pool,
            freeze_text_encoder=freeze,
            freeze_audio_encoder=freeze,
        ).eval()

    return make
