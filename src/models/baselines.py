from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoModel

from src.models.simple_fusion import masked_mean


@dataclass(frozen=True)
class BaselineOutput:
    logits: torch.Tensor
    embedding: torch.Tensor


class TextOnlyClassifier(nn.Module):
    """Frozen-transformer text baseline with masked mean pooling."""

    def __init__(
        self,
        num_labels: int,
        text_model_name: str = "distilbert-base-uncased",
        dropout: float = 0.2,
        freeze_text_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        text_hidden_dim = int(self.text_model.config.hidden_size)

        if freeze_text_encoder:
            for parameter in self.text_model.parameters():
                parameter.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(text_hidden_dim, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> BaselineOutput:
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        embedding = masked_mean(text_outputs.last_hidden_state, attention_mask)
        logits = self.classifier(embedding)
        return BaselineOutput(logits=logits, embedding=embedding)


class AudioOnlyGRUClassifier(nn.Module):
    """Temporal prosody baseline over aligned eGeMAPS frame sequences."""

    def __init__(
        self,
        num_labels: int,
        audio_feature_dim: int = 88,
        audio_hidden_dim: int = 128,
        gru_hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_feature_dim, audio_hidden_dim),
            nn.LayerNorm(audio_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=audio_hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(gru_hidden_dim, num_labels),
        )

    def forward(
        self,
        audio_frame_features: torch.Tensor,
        audio_frame_mask: torch.Tensor,
    ) -> BaselineOutput:
        projected = self.audio_projection(audio_frame_features)
        encoded, _ = self.gru(projected)
        embedding = masked_mean(encoded, audio_frame_mask)
        logits = self.classifier(embedding)
        return BaselineOutput(logits=logits, embedding=embedding)


class SlidingFrameTextFusionClassifier(nn.Module):
    """Frozen-text plus temporal sliding-frame prosody fusion baseline."""

    def __init__(
        self,
        num_labels: int,
        text_model_name: str = "distilbert-base-uncased",
        audio_feature_dim: int = 88,
        audio_hidden_dim: int = 128,
        audio_gru_hidden_dim: int = 128,
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
        self.audio_gru = nn.GRU(
            input_size=audio_hidden_dim,
            hidden_size=audio_gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_dim + audio_gru_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_frame_features: torch.Tensor,
        audio_frame_mask: torch.Tensor,
    ) -> BaselineOutput:
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embedding = masked_mean(text_outputs.last_hidden_state, attention_mask)

        projected_audio = self.audio_projection(audio_frame_features)
        audio_encoded, _ = self.audio_gru(projected_audio)
        audio_embedding = masked_mean(audio_encoded, audio_frame_mask)

        embedding = torch.cat([text_embedding, audio_embedding], dim=-1)
        logits = self.classifier(embedding)
        return BaselineOutput(logits=logits, embedding=embedding)


class TextAudioCrossAttentionClassifier(nn.Module):
    """Text-token queries over sliding-frame audio states."""

    def __init__(
        self,
        num_labels: int,
        text_model_name: str = "distilbert-base-uncased",
        audio_feature_dim: int = 88,
        num_attention_heads: int = 8,
        classifier_hidden_dim: int = 256,
        dropout: float = 0.2,
        freeze_text_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        text_hidden_dim = int(self.text_model.config.hidden_size)

        if freeze_text_encoder:
            for parameter in self.text_model.parameters():
                parameter.requires_grad = False

        self.audio_projection = nn.Sequential(
            nn.Linear(audio_feature_dim, text_hidden_dim),
            nn.LayerNorm(text_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=text_hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_norm = nn.LayerNorm(text_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(text_hidden_dim, classifier_hidden_dim),
            nn.LayerNorm(classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_frame_features: torch.Tensor,
        audio_frame_mask: torch.Tensor,
    ) -> BaselineOutput:
        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_hidden = text_outputs.last_hidden_state
        audio_states = self.audio_projection(audio_frame_features)

        effective_audio_mask = audio_frame_mask.clone()
        empty_audio_rows = effective_audio_mask.sum(dim=1) <= 0
        if empty_audio_rows.any():
            effective_audio_mask[empty_audio_rows, 0] = 1.0

        attended_audio, _ = self.cross_attention(
            query=text_hidden,
            key=audio_states,
            value=audio_states,
            key_padding_mask=effective_audio_mask <= 0,
            need_weights=False,
        )
        fused_tokens = self.fusion_norm(text_hidden + self.dropout(attended_audio))
        embedding = masked_mean(fused_tokens, attention_mask)
        logits = self.classifier(embedding)
        return BaselineOutput(logits=logits, embedding=embedding)
