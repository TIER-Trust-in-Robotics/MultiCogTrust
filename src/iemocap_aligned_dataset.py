"""Load pre-processed IEMOCAP aligned features from data/processed/."""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from datasets import Dataset

_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "iemocap_aligned_features"

LABEL_NAMES = [
    "anger", "happiness", "excitement", "neutral", "sadness",
    "frustration", "fear", "surprise", "disgust", "undefined", "other",
]


def load_iemocap_aligned(data_dir: Path = _DATA_DIR) -> Tuple[Dataset, Dataset, int, List[str]]:
    """Load aligned IEMOCAP features. Returns (train_ds, test_ds, num_labels, label_names)."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Aligned features not found at {data_dir}. "
            "Run the alignment pipeline first."
        )

    records = [
        json.loads(line)
        for line in (data_dir / "metadata.jsonl").read_text().splitlines()
        if line.strip()
    ]
    audio_npz = np.load(data_dir / "audio_frames.npz", allow_pickle=True)
    word_npz = np.load(data_dir / "word_features.npz", allow_pickle=True)

    train_rows = _empty_rows()
    test_rows = _empty_rows()

    for r in records:
        uid = r["utterance_id"]
        rows = train_rows if r["split"] == "train" else test_rows
        rows["text"].append(r["text"])
        rows["label"].append(r["label"])
        rows["audio_frame_features"].append(audio_npz[uid].tolist())
        rows["audio_frame_start_sec"].append(audio_npz[f"{uid}__start_sec"].tolist())
        rows["audio_frame_end_sec"].append(audio_npz[f"{uid}__end_sec"].tolist())
        rows["word_audio_features"].append(word_npz[uid].tolist())
        rows["word_audio_feature_mask"].append(r["word_audio_feature_mask"])
        rows["words"].append(r["words"])

    return Dataset.from_dict(train_rows), Dataset.from_dict(test_rows), 11, LABEL_NAMES


def _empty_rows() -> dict:
    return {
        "text": [],
        "label": [],
        "audio_frame_features": [],
        "audio_frame_start_sec": [],
        "audio_frame_end_sec": [],
        "word_audio_features": [],
        "word_audio_feature_mask": [],
        "words": [],
    }
