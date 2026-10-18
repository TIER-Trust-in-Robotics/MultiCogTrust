"""Provenance-checked, float32 frozen encoder sequences for development sessions."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.emotion_checkpoint import prepare_audio, tokenize_text


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def feature_path(directory, utterance_id):
    if Path(utterance_id).name != utterance_id or not utterance_id:
        raise ValueError("Invalid utterance id")
    return Path(directory) / f"{utterance_id}.npz"


def validate_arrays(text, audio):
    if (
        text.ndim != 2
        or audio.ndim != 2
        or min(len(text), len(audio)) == 0
        or text.shape[1] != audio.shape[1]
        or not np.isfinite(text).all()
        or not np.isfinite(audio).all()
        or text.dtype != np.float32
        or audio.dtype != np.float32
    ):
        raise ValueError("Invalid cached encoder sequences")


def build_cache(
    samples, model, tokenizer, preprocessing, directory, metadata_hash, device
):
    if not (model.freeze_text_encoder and model.freeze_audio_encoder):
        raise ValueError("Embedding caches require both encoders to be frozen")
    if any(s.session == 5 for s in samples):
        raise ValueError("Development cache must not encode Session 5")
    ids = [s.utterance_id for s in samples]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate utterance IDs")
    config = model.model_config
    contract = {
        "schema_version": 1,
        "metadata_sha256": metadata_hash,
        "encoders": {
            k: config[k]
            for k in (
                "text_model_name",
                "text_revision",
                "audio_model_name",
                "audio_revision",
            )
        },
        "preprocessing": asdict(preprocessing),
        "dtype": "float32",
        "utterance_ids": ids,
    }
    if (
        not contract["encoders"]["text_revision"]
        or not contract["encoders"]["audio_revision"]
    ):
        raise ValueError("Encoder revisions must be pinned")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    index_path = directory / "index.json"
    if index_path.exists():
        previous = json.loads(index_path.read_text())
        if previous["contract"] != contract:
            raise ValueError(
                "Cache contract changed; choose a separate cache directory"
            )
        if previous.get("complete"):
            if [r["utterance_id"] for r in previous["entries"]] != ids:
                raise ValueError("Completed cache has inconsistent entries")
            for sample, entry in zip(samples, previous["entries"]):
                source_hash = hashlib.sha256(
                    (sample.text + "\n" + sha256(sample.audio_path)).encode()
                ).hexdigest()
                if source_hash != entry["source_sha256"]:
                    raise ValueError(f"Cache source changed: {sample.utterance_id}")
                if (
                    sha256(feature_path(directory, sample.utterance_id))
                    != entry["feature_sha256"]
                ):
                    raise ValueError(f"Cache checksum mismatch: {sample.utterance_id}")
            # Keep the index bytes stable: checkpoints cite its checksum.
            print(f"Reusing verified cache for {len(samples)} utterances", flush=True)
            return previous
    write_json(index_path, {"contract": contract, "complete": False})
    model.eval()
    entries = []
    started = time.perf_counter()
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            source_hash = hashlib.sha256(
                (sample.text + "\n" + sha256(sample.audio_path)).encode()
            ).hexdigest()
            path = feature_path(directory, sample.utterance_id)
            if path.exists():
                with np.load(path, allow_pickle=False) as cached:
                    if str(cached["source_sha256"]) != source_hash:
                        raise ValueError(f"Cache source changed: {sample.utterance_id}")
                    text, audio = cached["text"], cached["audio"]
                    validate_arrays(text, audio)
            else:
                values, sample_rate = sf.read(sample.audio_path, dtype="float32")
                waveform = prepare_audio(values, sample_rate, preprocessing)[None].to(
                    device
                )
                tokens = tokenize_text(tokenizer, sample.text, preprocessing)
                text = (
                    model.encode_text(
                        tokens["input_ids"].to(device),
                        tokens["attention_mask"].to(device),
                    )[0]
                    .cpu()
                    .numpy()
                    .copy()
                )
                audio = model.encode_audio(waveform)[0].cpu().numpy().copy()
                validate_arrays(text, audio)
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, text=text, audio=audio, source_sha256=source_hash)
                temporary.replace(path)
            entries.append(
                {
                    "utterance_id": sample.utterance_id,
                    "session": sample.session,
                    "label": sample.label,
                    "text_tokens": len(text),
                    "audio_frames": len(audio),
                    "source_sha256": source_hash,
                    "feature_sha256": sha256(path),
                }
            )
            if (i + 1) % 100 == 0 or i + 1 == len(samples):
                print(
                    f"Cached {i + 1}/{len(samples)} ({(i + 1) / (time.perf_counter() - started):.1f} utterances/s)",
                    flush=True,
                )
    index = {
        "contract": contract,
        "complete": True,
        "entries": entries,
        "seconds": time.perf_counter() - started,
    }
    write_json(index_path, index)
    return index


class CachedDataset(Dataset):
    def __init__(self, samples, directory):
        self.samples, self.directory = samples, Path(directory)
        index = json.loads((self.directory / "index.json").read_text())
        if not index["complete"]:
            raise ValueError("Cache is incomplete")
        available = {r["utterance_id"]: r for r in index["entries"]}
        for sample in samples:
            entry = available.get(sample.utterance_id)
            if (
                entry is None
                or entry["label"] != sample.label
                or entry["session"] != sample.session
            ):
                raise ValueError("Cache/sample identity mismatch")
            if (
                sha256(feature_path(directory, sample.utterance_id))
                != entry["feature_sha256"]
            ):
                raise ValueError("Cache checksum mismatch")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        with np.load(
            feature_path(self.directory, sample.utterance_id), allow_pickle=False
        ) as item:
            return (
                torch.from_numpy(item["text"]),
                torch.from_numpy(item["audio"]),
                sample.label,
            )


def collate_embeddings(batch):
    texts, audios, labels = zip(*batch)

    def pad(rows):
        values = torch.nn.utils.rnn.pad_sequence(rows, batch_first=True)
        mask = (
            torch.arange(values.shape[1])[None]
            < torch.tensor([len(r) for r in rows])[:, None]
        )
        return values, mask

    text, text_mask = pad(texts)
    audio, audio_mask = pad(audios)
    return text, text_mask, audio, audio_mask, torch.tensor(labels)
