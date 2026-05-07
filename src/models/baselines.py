"""Text-only and audio-only baseline classifiers for IEMOCAP emotion."""

import torch
import torch.nn as nn
from transformers import AutoModel


class TextOnlyClassifier(nn.Module):
    """DistilBERT + linear head. Encoder frozen by default."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 11,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_labels)

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: (batch, seq_len, hidden_size) — per-token embeddings
        token_embeddings = outputs.last_hidden_state
        # CLS token as sentence representation
        pooled = token_embeddings[:, 0, :]
        return self.classifier(pooled)

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True


class AudioOnlyGRUClassifier(nn.Module):
    """Linear projection -> GRU -> masked mean pooling -> classifier."""

    def __init__(
        self,
        input_dim: int = 88,
        hidden_dim: int = 128,
        num_labels: int = 11,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, audio_features: torch.Tensor, audio_mask: torch.Tensor) -> torch.Tensor:
        # audio_features: (batch, max_frames, 88)
        x = self.input_proj(audio_features)
        x, _ = self.gru(x)
        x = self.dropout(x)
        # masked mean over valid frames only
        mask = audio_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)
