from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool sequence values using a 0/1 mask."""
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    summed = (values * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    return summed / counts


@dataclass(frozen=True)
class SimpleFusionOutput:
    logits: torch.Tensor
    text_embedding: torch.Tensor
    audio_embedding: torch.Tensor
    fused_embedding: torch.Tensor


class SimpleTextProsodyFusionClassifier(nn.Module):
    """
    Baseline multimodal classifier with no token-level attention.

    Text is encoded by DistilBERT and mean-pooled. Prosody is projected from
    token-aligned 88-dim OpenSMILE features and mean-pooled. The pooled vectors
    are concatenated and passed through a small MLP classifier.
    """

    def __init__(
        self,
        num_labels: int,
        text_model_name: str = "distilbert-base-uncased",
        audio_feature_dim: int = 88,
        audio_hidden_dim: int = 128,
        fusion_hidden_dim: int = 256,
        dropout: float = 0.2,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        text_hidden_dim = int(self.text_model.config.hidden_size)

        if freeze_text_encoder:
            for parameter in self.text_model.parameters():
                parameter.requires_grad = False

        self.audio_projection = nn.Sequential(
            nn.Linear(audio_feature_dim, audio_hidden_dim),
            nn.LayerNorm(audio_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_dim + audio_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_features: torch.Tensor,
        audio_token_mask: torch.Tensor,
    ) -> SimpleFusionOutput:
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return self.forward_from_embeddings(
            text_hidden_state=text_outputs.last_hidden_state,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_token_mask=audio_token_mask,
        )

    def forward_from_embeddings(
        self,
        text_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_features: torch.Tensor,
        audio_token_mask: torch.Tensor,
    ) -> SimpleFusionOutput:
        text_embedding = masked_mean(
            text_hidden_state,
            attention_mask,
        )

        projected_audio = self.audio_projection(audio_features)
        combined_audio_mask = audio_token_mask * attention_mask
        audio_embedding = masked_mean(projected_audio, combined_audio_mask)

        fused_embedding = torch.cat([text_embedding, audio_embedding], dim=-1)
        logits = self.classifier(fused_embedding)
        return SimpleFusionOutput(
            logits=logits,
            text_embedding=text_embedding,
            audio_embedding=audio_embedding,
            fused_embedding=fused_embedding,
        )
