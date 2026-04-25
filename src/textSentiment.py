"""Text embedding utilities for transcribed utterances.

The real-time pipeline needs token-level transformer states while keeping a
stable mapping back to the original transcript words. This module exposes a
small encoder and a queue worker that can be used from ``pipeline_demo.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
import queue
import re
import threading
from typing import Any

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")


@dataclass(frozen=True)
class WordSpan:
    word: str
    start_char: int
    end_char: int


@dataclass
class TextHiddenStates:
    text: str
    tokens: list[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    last_hidden_state: torch.Tensor
    token_offsets: list[tuple[int, int]]
    token_word_indices: list[int | None]
    words: list[WordSpan]
    word_token_indices: list[list[int]]
    word_hidden_states: torch.Tensor

    @property
    def hidden_size(self) -> int:
        return int(self.last_hidden_state.shape[-1])


def _extract_words(text: str) -> list[WordSpan]:
    return [
        WordSpan(match.group(0), match.start(), match.end())
        for match in WORD_RE.finditer(text)
    ]


def _overlaps(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return start_a < end_b and start_b < end_a


def _map_tokens_to_words(
    token_offsets: list[tuple[int, int]],
    special_tokens_mask: list[int],
    words: list[WordSpan],
) -> tuple[list[int | None], list[list[int]]]:
    token_word_indices: list[int | None] = []
    word_token_indices: list[list[int]] = [[] for _ in words]

    word_idx = 0
    for token_idx, (token_start, token_end) in enumerate(token_offsets):
        if special_tokens_mask[token_idx] or token_start == token_end:
            token_word_indices.append(None)
            continue

        while word_idx < len(words) and words[word_idx].end_char <= token_start:
            word_idx += 1

        matched_idx: int | None = None
        for candidate_idx in range(word_idx, len(words)):
            word = words[candidate_idx]
            if word.start_char >= token_end:
                break
            if _overlaps(token_start, token_end, word.start_char, word.end_char):
                matched_idx = candidate_idx
                break

        token_word_indices.append(matched_idx)
        if matched_idx is not None:
            word_token_indices[matched_idx].append(token_idx)

    return token_word_indices, word_token_indices


class TextHiddenStateEncoder:
    """Encode text and aggregate subword states to transcript words."""

    def __init__(
        self, model_name: str = "distilbert-base-uncased", device: str | None = None
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )
        if not self.tokenizer.is_fast:
            raise ValueError(
                f"{model_name!r} must use a fast tokenizer for offset alignment"
            )

        self.model: PreTrainedModel = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, text: str) -> TextHiddenStates:
        text = text.strip()
        words = _extract_words(text)

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            truncation=True,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        special_tokens_mask = encoded.pop("special_tokens_mask")[0].tolist()
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = self.model(**model_inputs)

        last_hidden_state = outputs.last_hidden_state[0].detach().cpu()
        token_offsets = [(int(start), int(end)) for start, end in offsets]
        token_word_indices, word_token_indices = _map_tokens_to_words(
            token_offsets,
            [int(value) for value in special_tokens_mask],
            words,
        )

        word_hidden_rows: list[torch.Tensor] = []
        for token_indices in word_token_indices:
            if token_indices:
                word_hidden_rows.append(last_hidden_state[token_indices].mean(dim=0))
            else:
                word_hidden_rows.append(torch.zeros(last_hidden_state.shape[-1]))

        word_hidden_states = (
            torch.stack(word_hidden_rows)
            if word_hidden_rows
            else torch.empty((0, last_hidden_state.shape[-1]))
        )

        return TextHiddenStates(
            text=text,
            tokens=self.tokenizer.convert_ids_to_tokens(
                model_inputs["input_ids"][0].detach().cpu()
            ),
            input_ids=model_inputs["input_ids"][0].detach().cpu(),
            attention_mask=model_inputs["attention_mask"][0].detach().cpu(),
            last_hidden_state=last_hidden_state,
            token_offsets=token_offsets,
            token_word_indices=token_word_indices,
            words=words,
            word_token_indices=word_token_indices,
            word_hidden_states=word_hidden_states,
        )


def _queue_item_text(item: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(item, str):
        return item, {}
    if isinstance(item, dict):
        text = item.get("text", "")
        metadata = {key: value for key, value in item.items() if key != "text"}
        return str(text), metadata
    return str(item), {}


def nlp_worker(
    text_queue: queue.Queue,
    stop_event: threading.Event,
    result_queue: queue.Queue | None = None,
    model_name: str = "distilbert-base-uncased",
    device: str | None = None,
    print_summary: bool = True,
    encoder: TextHiddenStateEncoder | None = None,
):
    if encoder is None:
        encoder = TextHiddenStateEncoder(model_name=model_name, device=device)

    while not stop_event.is_set():
        try:
            item = text_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            break

        text, metadata = _queue_item_text(item)
        if not text.strip():
            continue

        hidden_states = encoder.encode(text)
        result = {
            "text": text,
            "metadata": metadata,
            "hidden_states": hidden_states,
        }
        if result_queue is not None:
            result_queue.put(result)

        if print_summary:
            print(
                "  Text hidden states: "
                f"{len(hidden_states.tokens)} tokens, "
                f"{len(hidden_states.words)} words, "
                f"token tensor={tuple(hidden_states.last_hidden_state.shape)}, "
                f"word tensor={tuple(hidden_states.word_hidden_states.shape)}"
            )

    if result_queue is not None:
        result_queue.put(None)


if __name__ == "__main__":
    sample = "I am so frustrated right now."
    encoder = TextHiddenStateEncoder()
    states = encoder.encode(sample)
    print(f"tokens: {states.tokens}")
    print(f"token hidden states: {tuple(states.last_hidden_state.shape)}")
    print(f"words: {[word.word for word in states.words]}")
    print(f"word hidden states: {tuple(states.word_hidden_states.shape)}")
