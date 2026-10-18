"""Load timestamped visual sequences with explicit detection/padding masks."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src.iemocap_embedding_cache import sha256


class VisualFeatureDataset(Dataset):
    def __init__(self, directory, partition, verify_checksums=True):
        self.directory = Path(directory)
        if partition not in ("train", "validation"):
            raise ValueError(
                "Development visual store exposes only train and validation"
            )
        report = json.loads((self.directory / "coverage_report.json").read_text())
        if not report["complete_development_set"] or report["session_5_processed"]:
            raise ValueError("Expected a complete development-only visual store")
        if (
            sha256(self.directory / "metadata.jsonl") != report["metadata_sha256"]
            or sha256(self.directory / "config.json") != report["config_sha256"]
        ):
            raise ValueError("Visual store manifest/config checksum mismatch")
        self.config = json.loads((self.directory / "config.json").read_text())
        records = [
            json.loads(line)
            for line in (self.directory / "metadata.jsonl").read_text().splitlines()
        ]
        if len({r["utterance_id"] for r in records}) != len(records):
            raise ValueError("Duplicate visual utterance IDs")
        for row in records:
            if row["session"] not in (1, 2, 3, 4) or row["partition"] != (
                "validation" if row["session"] == 4 else "train"
            ):
                raise ValueError("Visual session/partition mismatch")
        self.records = [r for r in records if r["partition"] == partition]
        if verify_checksums:
            for row in self.records:
                if (
                    sha256(self.directory / row["feature_path"])
                    != row["feature_sha256"]
                ):
                    raise ValueError(
                        f"Visual feature checksum mismatch: {row['utterance_id']}"
                    )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        with np.load(self.directory / row["feature_path"], allow_pickle=False) as data:
            times = data["timestamps_sec"] - row["dialog_start_sec"]
            # Unavailable timestamps are placeholders and remain masked.
            times = np.where(data["decoded"], times, 0).astype(np.float32)
            return {
                "utterance_id": row["utterance_id"],
                "label": row["label"],
                "blendshapes": torch.from_numpy(data["blendshapes"]),
                "image_embeddings": torch.from_numpy(data["image_embeddings"]),
                "times_sec": torch.from_numpy(times),
                "visual_mask": torch.from_numpy(data["valid"]),
                "image_mask": torch.from_numpy(data["image_valid"]),
                "blendshape_mask": torch.from_numpy(data["blendshape_valid"]),
                "decoded_mask": torch.from_numpy(data["decoded"]),
            }


def collate_visual(batch):
    result = {
        "utterance_ids": [row["utterance_id"] for row in batch],
        "labels": torch.tensor([row["label"] for row in batch]),
    }
    for key in (
        "blendshapes",
        "image_embeddings",
        "times_sec",
        "visual_mask",
        "image_mask",
        "blendshape_mask",
        "decoded_mask",
    ):
        result[key] = torch.nn.utils.rnn.pad_sequence(
            [r[key] for r in batch], batch_first=True
        )
    lengths = torch.tensor([len(row["times_sec"]) for row in batch])
    result["sequence_mask"] = torch.arange(int(lengths.max()))[None] < lengths[:, None]
    return result
