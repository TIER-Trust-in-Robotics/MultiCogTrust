"""Async NLP worker for transcript embeddings."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src import config
from src.core.events import TextEmbeddingEvent, TranscriptEvent
from src.text.encoder import TextHiddenStateEncoder


@dataclass(frozen=True)
class NLPConfig:
    model_name: str = config.MODEL_NAME
    device: str = "cpu"
    queue_get_timeout_sec: float = 0.5
    queue_put_timeout_sec: float = 0.25


def _to_text_embedding_event(
    transcript: TranscriptEvent,
    encoder: TextHiddenStateEncoder,
) -> TextEmbeddingEvent:
    hidden_states = encoder.encode(transcript.text)
    return TextEmbeddingEvent(
        start_time=transcript.start_time,
        end_time=transcript.end_time,
        text=transcript.text,
        tokens=hidden_states.tokens,
        words=[word.word for word in hidden_states.words],
        input_ids=hidden_states.input_ids,
        attention_mask=hidden_states.attention_mask,
        last_hidden_state=hidden_states.last_hidden_state,
        word_hidden_states=hidden_states.word_hidden_states,
        token_word_indices=hidden_states.token_word_indices,
        word_token_indices=hidden_states.word_token_indices,
    )


async def _put_until_stopped(
    output_queue: asyncio.Queue[TextEmbeddingEvent],
    stop_event: asyncio.Event,
    event: TextEmbeddingEvent,
    timeout_sec: float,
) -> None:
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(output_queue.put(event), timeout=timeout_sec)
            return
        except TimeoutError:
            continue


async def run_nlp_worker(
    input_queue: asyncio.Queue[TranscriptEvent],
    output_queue: asyncio.Queue[TextEmbeddingEvent],
    stop_event: asyncio.Event,
    config_: NLPConfig,
    encoder: TextHiddenStateEncoder | None = None,
) -> None:
    """Encode transcript events and publish token/word hidden states."""

    config_ = config_ or NLPConfig()
    encoder = encoder or TextHiddenStateEncoder(
        model_name=config_.model_name,
        device=config_.device,
    )

    while not stop_event.is_set():
        try:
            transcript = await asyncio.wait_for(
                input_queue.get(),
                timeout=config_.queue_get_timeout_sec,
            )
        except TimeoutError:
            continue

        if not transcript.text.strip():
            continue

        event = await asyncio.to_thread(
            _to_text_embedding_event,
            transcript,
            encoder,
        )
        await _put_until_stopped(
            output_queue,
            stop_event,
            event,
            config_.queue_put_timeout_sec,
        )
