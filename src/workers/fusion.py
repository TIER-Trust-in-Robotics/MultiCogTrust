from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import torch

from src.core.events import (
    FusionPredictionEvent,
    ProsodyFeatureEvent,
    TextEmbeddingEvent,
)
from src.models.simple_fusion import SimpleTextProsodyFusionClassifier


@dataclass(frozen=True)
class SimpleFusionWorkerConfig:
    device: str = "cpu"
    queue_get_timeout_sec: float = 0.25
    queue_put_timeout_sec: float = 0.25
    stale_event_sec: float = 30.0


def intervals_overlap(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> bool:
    return left_start < right_end and right_start < left_end


def prosody_to_token_features(
    text_event: TextEmbeddingEvent,
    prosody_event: ProsodyFeatureEvent,
    feature_width: int = 88,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_count = int(text_event.input_ids.shape[0])
    token_audio = np.zeros((token_count, feature_width), dtype=np.float32)
    token_mask = np.zeros((token_count,), dtype=np.float32)

    word_features = np.asarray(prosody_event.word_features, dtype=np.float32)
    word_mask = np.asarray(prosody_event.word_feature_mask, dtype=np.float32)
    for token_idx, word_idx in enumerate(text_event.token_word_indices):
        if word_idx is None or word_idx >= len(word_features):
            continue
        token_audio[token_idx] = word_features[word_idx]
        token_mask[token_idx] = (
            word_mask[word_idx] if word_idx < len(word_mask) else 1.0
        )

    return torch.from_numpy(token_audio), torch.from_numpy(token_mask)


def find_matching_prosody(
    text_event: TextEmbeddingEvent,
    prosody_buffer: list[ProsodyFeatureEvent],
) -> ProsodyFeatureEvent | None:
    overlapping = [
        event
        for event in prosody_buffer
        if intervals_overlap(
            text_event.start_time,
            text_event.end_time,
            event.start_time,
            event.end_time,
        )
    ]
    if not overlapping:
        return None
    return min(
        overlapping,
        key=lambda event: abs(event.start_time - text_event.start_time)
        + abs(event.end_time - text_event.end_time),
    )


async def _put_until_stopped(
    output_queue: asyncio.Queue[FusionPredictionEvent],
    stop_event: asyncio.Event,
    event: FusionPredictionEvent,
    timeout_sec: float,
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(output_queue.put(event), timeout=timeout_sec)
            return
        except TimeoutError:
            continue


async def run_simple_fusion_worker(
    text_queue: asyncio.Queue[TextEmbeddingEvent],
    prosody_queue: asyncio.Queue[ProsodyFeatureEvent],
    output_queue: asyncio.Queue[FusionPredictionEvent],
    stop_event: asyncio.Event,
    model: SimpleTextProsodyFusionClassifier,
    config: SimpleFusionWorkerConfig | None = None,
) -> None:
    config = config or SimpleFusionWorkerConfig()
    device = torch.device(config.device)
    model.to(device)
    model.eval()
    prosody_buffer: list[ProsodyFeatureEvent] = []

    while not stop_event.is_set():
        try:
            while True:
                prosody_buffer.append(prosody_queue.get_nowait())
        except asyncio.QueueEmpty:
            pass

        try:
            text_event = await asyncio.wait_for(
                text_queue.get(),
                timeout=config.queue_get_timeout_sec,
            )
        except TimeoutError:
            continue

        min_start = text_event.start_time - config.stale_event_sec
        prosody_buffer = [
            event for event in prosody_buffer if event.end_time >= min_start
        ]
        prosody_event = find_matching_prosody(text_event, prosody_buffer)
        if prosody_event is None:
            continue

        token_audio, audio_token_mask = prosody_to_token_features(
            text_event,
            prosody_event,
        )
        with torch.inference_mode():
            outputs = model.forward_from_embeddings(
                text_hidden_state=text_event.last_hidden_state.unsqueeze(0).to(
                    device
                ),
                attention_mask=text_event.attention_mask.unsqueeze(0).to(device),
                audio_features=token_audio.unsqueeze(0).to(device),
                audio_token_mask=audio_token_mask.unsqueeze(0).to(device),
            )
            probabilities = torch.softmax(outputs.logits, dim=-1)[0].detach().cpu()
            prediction = FusionPredictionEvent(
                start_time=text_event.start_time,
                end_time=text_event.end_time,
                utterance_id=prosody_event.utterance_id,
                text=text_event.text,
                logits=outputs.logits[0].detach().cpu(),
                probabilities=probabilities,
                predicted_label=int(probabilities.argmax().item()),
            )

        await _put_until_stopped(
            output_queue,
            stop_event,
            prediction,
            config.queue_put_timeout_sec,
        )
