"""Causal dialogue text construction, without model/device side effects."""

from __future__ import annotations

from collections import defaultdict
import hashlib


def transcript(row, mode, speaker, asr):
    if mode == "ref" or (mode == "hri" and row["speaker"] != speaker):
        return row["text"].strip()
    return asr.get(row["utterance_id"], "").strip()


def context_records(rows, variant, mode, asr, label_map):
    if variant not in {"other", "ownother", "own2"} or mode not in {"ref", "asr", "hri"}:
        raise ValueError("Unknown context condition")
    if any(row["session"] not in (1, 2, 3, 4) for row in rows):
        raise ValueError("Only development sessions are allowed")
    dialogs = defaultdict(list)
    for row in rows:
        dialogs[row["dialog_id"]].append(row)
    for sequence in dialogs.values():
        sequence.sort(key=lambda row: (row["dialog_start_sec"], row["utterance_id"]))
        for target in sequence:
            if target["label_code"] not in label_map or (mode != "ref" and target["session"] != 4):
                continue
            current = transcript(target, mode, target["speaker"], asr)
            if not current:
                continue
            # Complete before the participant starts; no transcript from an
            # overlapping/future utterance can enter this strict HRI condition.
            earlier = [row for row in sequence if row["dialog_end_sec"] <= target["dialog_start_sec"]
                       and transcript(row, mode, target["speaker"], asr)]
            earlier.sort(key=lambda row: (row["dialog_end_sec"], row["dialog_start_sec"], row["utterance_id"]))
            own = [row for row in earlier if row["speaker"] == target["speaker"]]
            other = [row for row in earlier if row["speaker"] != target["speaker"]]
            if variant == "other":
                selected = other[-1:]
            elif variant == "ownother":
                selected = sorted(own[-1:] + other[-1:], key=lambda row: row["dialog_start_sec"])
            else:
                selected = own[-2:]
            parts = [transcript(row, mode, target["speaker"], asr) for row in selected] + [current]
            yield {"utterance_id": target["utterance_id"], "session": target["session"],
                   "dialog_id": target["dialog_id"], "speaker": target["speaker"],
                   "start_sec": target["dialog_start_sec"], "context_ids": [row["utterance_id"] for row in selected],
                   "context_end_sec": [row["dialog_end_sec"] for row in selected], "parts": parts,
                   "text_sha256": hashlib.sha256(" [SEP] ".join(parts).encode()).hexdigest()}


def fit_context(record, tokenizer, max_tokens):
    parts, ids = list(record["parts"]), list(record["context_ids"])
    dropped = 0
    while len(parts) > 1 and len(tokenizer(" [SEP] ".join(parts))["input_ids"]) > max_tokens:
        parts.pop(0)
        ids.pop(0)
        dropped += 1
    text = " [SEP] ".join(parts)
    length = len(tokenizer(text)["input_ids"])
    return {**record, "encoded_text": text, "encoded_context_ids": ids, "dropped_context_lines": dropped,
            "token_count_before_truncation": length, "current_truncated": length > max_tokens,
            "encoded_text_sha256": hashlib.sha256(text.encode()).hexdigest()}
