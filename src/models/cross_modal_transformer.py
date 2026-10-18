"""Lightweight cross-modal transformer fusion classifier.

A port of MemoCMT (Khan et al., *Scientific Reports*, 2025;
https://github.com/tpnam0901/MemoCMT) adapted for live, lightweight inference.

The bidirectional cross-attention fusion and the cls/mean/max/min pooling are
kept faithful to the original. The heavy ``bert-base-uncased`` /
``HuBERT-base`` encoders are swapped for distilled variants
(``distilbert-base-uncased`` + ``ntu-spml/distilhubert``), taking the encoder
stack to ~90M parameters (~96M including the fusion head).

Original fusion (per ``src/models/networks.py`` in the MemoCMT repo):
    text branch : MultiheadAttention(Q=audio, K=text, V=text) -> Linear -> LN
    audio branch: MultiheadAttention(Q=text, K=audio, V=audio) -> Linear -> LN
    fused       : cat([text_norm, audio_norm], dim=1) -> pool -> MLP head

(The repo also instantiates a third ``fusion_attention`` block that is never
called in its ``forward``; it is intentionally omitted here.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn
from transformers import AutoModel


@dataclass(frozen=True)
class CrossModalFusionOutput:
    logits: torch.Tensor
    pooled_embedding: torch.Tensor
    text_norm: torch.Tensor
    audio_norm: torch.Tensor
    text_attn_weights: Optional[torch.Tensor] = None
    audio_attn_weights: Optional[torch.Tensor] = None


class CrossModalTransformerFusionClassifier(nn.Module):
    """MemoCMT-style cross-modal transformer with distilled encoders.

    Text is encoded by DistilBERT and audio by DistilHuBERT (both 16 kHz,
    768-dim). Each modality cross-attends to the other, the two attended
    sequences are concatenated and pooled, and a small MLP produces class
    logits. Both encoders share a hidden size (``encoder_dim``); the cross
    attention runs at that dim and projects to ``fusion_dim`` afterwards, as in
    the original.
    """

    def __init__(
        self,
        num_labels: int,
        text_model_name: str = "distilbert-base-uncased",
        audio_model_name: str = "ntu-spml/distilhubert",
        fusion_dim: int = 768,
        num_attention_heads: int = 8,
        dropout: float = 0.05,
        linear_layer_output: Sequence[int] = (128,),
        fusion_head_output_type: str = "min",
        freeze_text_encoder: bool = True,
        freeze_audio_encoder: bool = False,
        text_revision: str | None = None,
        audio_revision: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if fusion_head_output_type not in {"cls", "mean", "max", "min"}:
            raise ValueError(
                f"Invalid fusion_head_output_type: {fusion_head_output_type!r}"
            )

        self.text_encoder = AutoModel.from_pretrained(
            text_model_name, revision=text_revision, local_files_only=local_files_only
        )
        self.audio_encoder = AutoModel.from_pretrained(
            audio_model_name, revision=audio_revision, local_files_only=local_files_only
        )
        self.model_config = dict(
            num_labels=num_labels,
            text_model_name=text_model_name,
            audio_model_name=audio_model_name,
            fusion_dim=fusion_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            linear_layer_output=list(linear_layer_output),
            fusion_head_output_type=fusion_head_output_type,
            text_revision=getattr(self.text_encoder.config, "_commit_hash", None)
            or text_revision,
            audio_revision=getattr(self.audio_encoder.config, "_commit_hash", None)
            or audio_revision,
        )
        self.freeze_text_encoder = freeze_text_encoder
        self.freeze_audio_encoder = freeze_audio_encoder

        text_dim = int(self.text_encoder.config.hidden_size)
        audio_dim = int(self.audio_encoder.config.hidden_size)
        if text_dim != audio_dim:
            raise ValueError(
                "Cross attention requires matching encoder dims, got "
                f"text={text_dim}, audio={audio_dim}. Pick encoders with the "
                "same hidden size (DistilBERT and DistilHuBERT are both 768)."
            )
        self.encoder_dim = text_dim

        if freeze_text_encoder:
            for parameter in self.text_encoder.parameters():
                parameter.requires_grad = False
        if freeze_audio_encoder:
            for parameter in self.audio_encoder.parameters():
                parameter.requires_grad = False

        # Text branch: audio queries attend over text keys/values.
        self.text_attention = nn.MultiheadAttention(
            embed_dim=self.encoder_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.text_linear = nn.Linear(self.encoder_dim, fusion_dim)
        self.text_layer_norm = nn.LayerNorm(fusion_dim)

        # Audio branch: text queries attend over audio keys/values.
        self.audio_attention = nn.MultiheadAttention(
            embed_dim=self.encoder_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.audio_linear = nn.Linear(self.encoder_dim, fusion_dim)
        self.audio_layer_norm = nn.LayerNorm(fusion_dim)

        self.dropout = nn.Dropout(dropout)

        self.linear_layer_output = list(linear_layer_output)
        head_layers: list[nn.Module] = []
        previous_dim = fusion_dim
        for hidden_dim in self.linear_layer_output:
            head_layers.append(nn.Linear(previous_dim, hidden_dim))
            head_layers.append(nn.LeakyReLU())
            previous_dim = hidden_dim
        self.head = nn.Sequential(*head_layers)
        self.classifier = nn.Linear(previous_dim, num_labels)

        self.fusion_head_output_type = fusion_head_output_type

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen encoders must produce the same features as the cached/live path.
        if self.freeze_text_encoder:
            self.text_encoder.eval()
        if self.freeze_audio_encoder:
            self.audio_encoder.eval()
        return self

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return DistilBERT token embeddings ``[B, T_text, dim]``."""
        return self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

    def encode_audio(
        self,
        audio_input: torch.Tensor,
        audio_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return DistilHuBERT frame embeddings ``[B, T_audio, dim]``.

        ``audio_input`` is a raw 16 kHz waveform batch ``[B, num_samples]``.
        """
        return self._encode_audio_with_mask(audio_input, audio_attention_mask)[0]

    def _encode_audio_with_mask(self, audio_input, audio_attention_mask):
        if audio_input.ndim != 2 or audio_input.shape[1] == 0:
            raise ValueError("Audio must have shape [batch, nonempty samples]")
        mask = _validated_mask(audio_attention_mask, audio_input, "audio samples")
        lengths = mask.sum(1)
        expected = (
            torch.arange(audio_input.shape[1], device=audio_input.device)[None]
            < lengths[:, None]
        )
        if not torch.equal(mask, expected):
            raise ValueError("Audio sample masks must be contiguous and right padded")
        # HuBERT's group norm includes time. Masking attention alone cannot undo
        # padding in that convolutional encoder. Encode each real length first.
        if bool(mask.all()):
            embeddings = self.audio_encoder(audio_input).last_hidden_state
            return embeddings, torch.ones(
                embeddings.shape[:2], device=embeddings.device, dtype=torch.bool
            )
        rows = [
            self.audio_encoder(audio_input[i : i + 1, : int(length)]).last_hidden_state[
                0
            ]
            for i, length in enumerate(lengths)
        ]
        embeddings = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        frame_lengths = torch.tensor(
            [len(row) for row in rows], device=embeddings.device
        )
        frame_mask = (
            torch.arange(embeddings.shape[1], device=embeddings.device)[None]
            < frame_lengths[:, None]
        )
        return embeddings, frame_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        audio_input: torch.Tensor,
        audio_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> CrossModalFusionOutput:
        text_embeddings = self.encode_text(input_ids, attention_mask)
        # forward accepts a waveform-sample mask; forward_from_embeddings
        # accepts an encoder-frame mask. They are deliberately different sizes.
        audio_embeddings, frame_mask = self._encode_audio_with_mask(
            audio_input, audio_attention_mask
        )
        return self.forward_from_embeddings(
            text_embeddings=text_embeddings,
            audio_embeddings=audio_embeddings,
            text_attention_mask=attention_mask,
            audio_attention_mask=frame_mask,
            output_attentions=output_attentions,
        )

    def forward_from_embeddings(
        self,
        text_embeddings: torch.Tensor,
        audio_embeddings: torch.Tensor,
        text_attention_mask: Optional[torch.Tensor] = None,
        audio_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> CrossModalFusionOutput:
        """Run fusion on pre-computed encoder embeddings.

        Lets the live pipeline cache/stream encoder outputs (e.g. reuse a text
        embedding across audio frames) instead of re-running the encoders.
        Attention masks use the HF convention (1 = keep, 0 = pad) and are
        converted to ``key_padding_mask`` internally.
        """
        text_mask = _validated_mask(text_attention_mask, text_embeddings, "text")
        audio_mask = _validated_mask(
            audio_attention_mask, audio_embeddings, "audio frames"
        )
        text_key_padding = ~text_mask
        audio_key_padding = ~audio_mask

        # Text branch: audio queries, text keys/values.
        text_attention, text_attn_weights = self.text_attention(
            audio_embeddings,
            text_embeddings,
            text_embeddings,
            key_padding_mask=text_key_padding,
            need_weights=output_attentions,
            average_attn_weights=False,
        )
        text_norm = self.dropout(self.text_layer_norm(self.text_linear(text_attention)))

        # Audio branch: text queries, audio keys/values.
        audio_attention, audio_attn_weights = self.audio_attention(
            text_embeddings,
            audio_embeddings,
            audio_embeddings,
            key_padding_mask=audio_key_padding,
            need_weights=output_attentions,
            average_attn_weights=False,
        )
        audio_norm = self.dropout(
            self.audio_layer_norm(self.audio_linear(audio_attention))
        )

        fusion_norm = self.dropout(torch.cat((text_norm, audio_norm), dim=1))
        # The first branch has audio queries; the second has text queries.
        pooled = self._pool(fusion_norm, torch.cat((audio_mask, text_mask), dim=1))

        logits = self.classifier(self.head(self.dropout(pooled)))
        return CrossModalFusionOutput(
            logits=logits,
            pooled_embedding=pooled,
            text_norm=text_norm,
            audio_norm=audio_norm,
            text_attn_weights=text_attn_weights if output_attentions else None,
            audio_attn_weights=audio_attn_weights if output_attentions else None,
        )

    def _pool(self, fusion_norm: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.fusion_head_output_type == "cls":
            first = mask.long().argmax(dim=1)
            return fusion_norm[torch.arange(len(first), device=first.device), first]
        if self.fusion_head_output_type == "mean":
            return fusion_norm.masked_fill(~mask.unsqueeze(-1), 0).sum(
                dim=1
            ) / mask.sum(dim=1, keepdim=True)
        if self.fusion_head_output_type == "max":
            return fusion_norm.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(dim=1)[
                0
            ]
        return fusion_norm.masked_fill(~mask.unsqueeze(-1), torch.inf).min(dim=1)[0]


def _validated_mask(mask, values, name):
    if values.shape[1] == 0:
        raise ValueError(f"{name} sequence is empty")
    if mask is None:
        return torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
    if mask.shape != values.shape[:2]:
        raise ValueError(
            f"{name} mask shape {tuple(mask.shape)} must match {tuple(values.shape[:2])}"
        )
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError(f"{name} mask must contain only 0 and 1")
    mask = mask.to(device=values.device, dtype=torch.bool)
    if not bool(mask.any(dim=1).all()):
        raise ValueError(f"{name} contains an all-masked example")
    return mask


def _to_key_padding_mask(
    attention_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Convert an HF 1=keep/0=pad mask to a True=ignore key_padding_mask."""
    if attention_mask is None:
        return None
    return attention_mask == 0
