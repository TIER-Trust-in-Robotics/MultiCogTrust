"""Checksum-verified MMSA MOSEI development features, stored as memory maps.

The original pickle is restricted to NumPy array reconstruction. Test arrays
are necessarily deserialized with the release, then discarded without model
inference. Only train/valid arrays are exported for the reference trainers.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit_benchmark_data import read_mosei_csv
from src.iemocap_embedding_cache import sha256, write_json

RELEASE_SHA256 = "ad8b23d50557045e7d47959ce6c5b955d8d983f2979c7d9b7b9226f6dd6fec1f"
COUNTS = {"train": 16326, "valid": 1871}


class NumpyReleaseUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unapproved pickle global: {module}.{name}")
        return allowed[(module, name)]


def prepare(source: Path, output: Path, csv_path: Path):
    if (output / "index.json").exists():
        raise FileExistsError("Preserving prepared features; choose a new output")
    if sha256(source) != RELEASE_SHA256:
        raise ValueError("Feature file differs from the official MMSA checksum")
    print("Verified original release; loading restricted NumPy pickle", flush=True)
    with source.open("rb") as handle:
        data = NumpyReleaseUnpickler(handle).load()
    test = data.pop("test")
    test_ids = [str(uid).replace("$_$", "/") for uid in test["id"]]
    test_count = len(test_ids)
    del test
    gc.collect()
    if set(data) != {"train", "valid"} or test_count != 4659:
        raise ValueError("Unexpected original release partitions")
    csv_rows = {row["id"]: row for row in read_mosei_csv(csv_path)}
    index = {
        "schema_version": 1,
        "release_sha256": RELEASE_SHA256,
        "release_path": str(source.resolve()),
        "csv_sha256": sha256(csv_path),
        "protocol": "mmsa_original_unaligned_50_release",
        "test_count_deserialized_then_discarded": test_count,
        "test_ids": test_ids,
        "test_model_inference": False,
        "splits": {},
        "conversion": "Float32 as upstream; no temporal truncation, normalization or quantization. Audio -inf becomes zero in the dataset reader as upstream.",
    }
    for split, expected in COUNTS.items():
        subset = data.pop(split)
        ids = [str(uid).replace("$_$", "/") for uid in subset["id"]]
        if len(ids) != expected or len(set(ids)) != expected or set(ids) & set(test_ids):
            raise ValueError("Unexpected/overlapping development sample IDs")
        directory = output / split
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {}
        labels = np.asarray(subset["regression_labels"], dtype=np.float32).reshape(-1)
        for uid, value in zip(ids, labels, strict=True):
            if csv_rows[uid]["split"] != split or not np.isclose(csv_rows[uid]["sentiment"], value, atol=1e-6):
                raise ValueError(f"Release/CSV label or split mismatch: {uid}")
        for name in ("text", "audio", "vision", "regression_labels"):
            values = subset[name]
            shape = np.shape(values)
            if shape[0] != expected:
                raise ValueError(f"Invalid feature count: {split}/{name}")
            path = directory / f"{name}.npy"
            if path.exists():
                raise FileExistsError(path)
            mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
            nonfinite = 0
            for start in range(0, expected, 128):
                chunk = np.asarray(values[start:start + 128], dtype=np.float32)
                nonfinite += int((~np.isfinite(chunk)).sum())
                mapped[start:start + 128] = chunk
            mapped.flush()
            del mapped
            arrays[name] = {"path": str(path.resolve()), "shape": list(shape), "dtype": "float32", "sha256": sha256(path), "nonfinite_values": nonfinite}
            print(f"Exported {split}/{name}: {shape}", flush=True)
        rows = [{"id": uid, "text": str(text)} for uid, text in zip(ids, subset["raw_text"], strict=True)]
        manifest = directory / "samples.jsonl"
        with manifest.open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        index["splits"][split] = {"count": expected, "arrays": arrays, "samples_path": str(manifest.resolve()), "samples_sha256": sha256(manifest)}
        del subset
        gc.collect()
    train_ids = {r["id"] for r in map(json.loads, (output / "train/samples.jsonl").read_text().splitlines())}
    valid_ids = {r["id"] for r in map(json.loads, (output / "valid/samples.jsonl").read_text().splitlines())}
    if train_ids & valid_ids:
        raise ValueError("Overlapping train/valid IDs")
    write_json(output / "index.json", index)
    print("Development feature export complete; test features discarded", flush=True)


class ReferenceFeatures(Dataset):
    def __init__(self, root: Path, split: str, normalize: bool, verify: bool = True):
        if split not in COUNTS:
            raise ValueError("Reference loader admits only train/valid")
        self.index = json.loads((root / "index.json").read_text())
        if self.index["release_sha256"] != RELEASE_SHA256 or self.index["test_model_inference"]:
            raise ValueError("Unverified release contract")
        info = self.index["splits"][split]
        if sha256(info["samples_path"]) != info["samples_sha256"]:
            raise ValueError("Feature sample manifest changed")
        self.ids = [r["id"] for r in map(json.loads, Path(info["samples_path"]).read_text().splitlines())]
        self.arrays = {}
        for name, spec in info["arrays"].items():
            path = Path(spec["path"])
            if verify and sha256(path) != spec["sha256"]:
                raise ValueError(f"Feature bytes changed: {path}")
            self.arrays[name] = np.load(path, mmap_mode="r", allow_pickle=False)
        self.normalize = normalize
        if len(self.ids) != COUNTS[split] or any(len(a) != len(self.ids) for a in self.arrays.values()):
            raise ValueError("Feature/ID count mismatch")
        if len(set(self.ids)) != len(self.ids) or set(self.ids) & set(self.index["test_ids"]):
            raise ValueError("Duplicate or test sample in development features")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        text = np.array(self.arrays["text"][i], dtype=np.float32, copy=True)
        audio = np.array(self.arrays["audio"][i], dtype=np.float32, copy=True)
        vision = np.array(self.arrays["vision"][i], dtype=np.float32, copy=True)
        audio[audio == -np.inf] = 0
        if self.normalize:
            # Same mean over all padded timesteps as upstream __normalize.
            audio = np.mean(audio, axis=0, keepdims=True)
            vision = np.mean(vision, axis=0, keepdims=True)
            audio[np.isnan(audio)] = 0
            vision[np.isnan(vision)] = 0
        if not all(np.isfinite(x).all() for x in (text, audio, vision)):
            raise ValueError(f"Nonfinite reference features: {self.ids[i]}")
        label = float(np.asarray(self.arrays["regression_labels"][i]).reshape(-1)[0])
        return torch.from_numpy(text), torch.from_numpy(audio), torch.from_numpy(vision), torch.tensor([label], dtype=torch.float32), i


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/MOSEI/Processed/unaligned_50.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/benchmarks/week1_v1/mmsa_features")
    parser.add_argument("--csv", type=Path, default=ROOT / "data/MOSEI/label.csv")
    args = parser.parse_args()
    prepare(args.source, args.output, args.csv)
