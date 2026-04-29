import csv
import json
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, Audio
import ast


DATA_ROOT = Path(__name__).resolve().parent / "data"
IEMOCAP_ROOT = DATA_ROOT / "IEMOCAP_full_release"
MSP_ROOT = DATA_ROOT / "MSP-PODCAST"
IEMOCAP_FEATURES_PATH = DATA_ROOT / "iemocap_features.npz"
MSP_FEATURES_PATH = DATA_ROOT / "msp_features.npz"

FEATURE_SET_KEYS = {
    "all": ("all_features", "feature_names"),
    "trust": ("trust_features", "trust_feature_names:"),
}


def _unpack_feature_records(raw_array: np.ndarray) -> list[dict[str, Any]]:
    """Unpack the pickled feature list stored as a scalar bytes entry in the npz files."""
    payload = raw_array.item()
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"Expected pickled bytes, got {type(payload)!r}")
    records = pickle.loads(payload)
    if not isinstance(records, list):
        raise TypeError(f"Expected a list of feature records, got {type(records)!r}")
    return records


def _feature_names(records: list[dict[str, Any]], feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SET_KEYS:
        valid = ", ".join(sorted(FEATURE_SET_KEYS))
        raise ValueError(f"feature_set must be one of: {valid}")
    _, names_key = FEATURE_SET_KEYS[feature_set]
    names = records[0][names_key]
    return [str(name) for name in names]


def _feature_records_to_dataset(
    records: list[dict[str, Any]],
    label_codes: np.ndarray,
    label2idx: dict[str, int],
    feature_set: str,
) -> Dataset:
    feature_key, _ = FEATURE_SET_KEYS[feature_set]
    if len(records) != len(label_codes):
        raise ValueError(
            f"Feature/label length mismatch: {len(records)} features, "
            f"{len(label_codes)} labels"
        )

    names = _feature_names(records, feature_set)
    expected_width = len(names)
    features = []
    for i, record in enumerate(records):
        vector = np.asarray(record[feature_key], dtype=np.float32)
        if vector.shape != (expected_width,):
            raise ValueError(
                f"Feature row {i} has shape {vector.shape}; "
                f"expected ({expected_width},)"
            )
        features.append(vector.tolist())

    labels = [str(label) for label in label_codes]
    return Dataset.from_dict(
        {
            "features": features,
            "label": [label2idx[label] for label in labels],
            "label_code": labels,
        }
    )


def _load_feature_npz(
    npz_path: Path,
    label2idx: dict[str, int],
    label_names: list[str],
    feature_set: str = "trust",
) -> tuple[Dataset, Dataset, int, list[str]]:
    """
    Load one of the local acoustic/prosodic feature archives.

    feature_set:
        "trust" -> 19 curated/prosody-focused features.
        "all" -> all 88 eGeMAPS-style utterance functionals.
    """
    with np.load(npz_path, allow_pickle=True) as archive:
        train_records = _unpack_feature_records(archive["train_features"])
        test_records = _unpack_feature_records(archive["test_features"])
        train_ds = _feature_records_to_dataset(
            train_records, archive["train_labels"], label2idx, feature_set
        )
        test_ds = _feature_records_to_dataset(
            test_records, archive["test_labels"], label2idx, feature_set
        )
    return train_ds, test_ds, len(label2idx), label_names


# -------------------------
# IEMOCAP
# -------------------------

# renaming t
IEMOCAP_LABELS = [
    "ang",  # anger
    "hap",  # happiness
    "exc",  # excitement
    "neu",  # neutral
    "sad",  # sadness
    "fru",  # frustration
    "fea",  # fear
    "sur",  # surprise
    "dis",  # disgust
    "xxx",  # other/undefined
    "oth",  # other
]

IEMOCAP_LABEL_NAMES = [
    "anger",
    "happiness",
    "excitement",
    "neutral",
    "sadness",
    "frustration",
    "fear",
    "surprise",
    "disgust",
    "undefined",
    "other",
]

IEMOCAP_LABEL2IDX = {l: i for i, l in enumerate(IEMOCAP_LABELS)}


def _parse_iemocap_emo_file(
    emo_path: Path,
) -> dict[str, tuple[str, float, float, float]]:
    """Parse an IEMOCAP EmoEvaluation file. Returns {utterance_id: (emotion, V, A, D)}."""
    utt2emo = {}
    with open(emo_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("["):
                continue
            # Format: [start - end]\t<UTTERANCE_ID>\t<EMOTION>\t[V(alence), A(ctivation), D(ominance)]
            # \t = tab
            parts = line.split("\t")
            if len(parts) >= 3:
                utt_id = parts[1].strip()
                emo = parts[2].strip()
                VAD = ast.literal_eval(parts[3])
                utt2emo[utt_id] = (emo, *VAD)
    return utt2emo


def _parse_iemocap_transcript(trans_path: Path) -> dict[str, str]:
    """Parse an IEMOCAP transcription file. Returns {utterance_id: text}."""
    utt2text = {}
    with open(trans_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: UTT_ID [start-end]: text
            match = re.match(
                r"^(\S+)\s+\[[\d.\-]+\]:\s*(.+)$", line
            )  # it works i guess, thanks claude Opus 4.6 wt Extended Thinking™    :^).
            # The regex grabs two things, the first non-white space (UTT_ID), ignores anything within brackets (i.e. the time stamp), as well as a colon and white space before the next block of text that is grabbed until the end of line symbol ($)
            if match:
                utt_id = match.group(1)
                text = match.group(2).strip()
                utt2text[utt_id] = text
    return utt2text


def load_iemocap() -> tuple[Dataset, Dataset, int, list[str]]:
    """
    Load IEMOCAP dataset from local files.

    Sessions 1-4 are used for training, Session 5 for testing.

    Returns: (train_ds, test_ds, num_labels, label_names)
    """
    train_texts, train_labels, train_audio = [], [], []
    test_texts, test_labels, test_audio = [], [], []

    for session_num in range(1, 6):
        session_dir = IEMOCAP_ROOT / f"Session{session_num}"
        emo_dir = session_dir / "dialog" / "EmoEvaluation"
        trans_dir = session_dir / "dialog" / "transcriptions"
        wav_dir = session_dir / "sentences" / "wav"

        # Skip Attribute / Categorical / Self-evaluation subdirs
        emo_files = [
            f
            for f in emo_dir.glob("*.txt")
            if f.stem not in ("Attribute", "Categorical", "Self-evaluation")
        ]

        for emo_file in sorted(emo_files):
            dialog_name = emo_file.stem
            trans_file = trans_dir / f"{dialog_name}.txt"
            if not trans_file.exists():
                continue

            utt2emo = _parse_iemocap_emo_file(emo_file)
            utt2text = _parse_iemocap_transcript(trans_file)

            for utt_id, emo_record in utt2emo.items():
                emo = emo_record[0]
                if utt_id not in utt2text:
                    continue
                if emo not in IEMOCAP_LABEL2IDX:
                    continue

                audio_path = wav_dir / dialog_name / f"{utt_id}.wav"
                if not audio_path.exists():
                    continue

                text = utt2text[utt_id]
                label = IEMOCAP_LABEL2IDX[emo]

                if session_num <= 4:
                    train_texts.append(text)
                    train_labels.append(label)
                    train_audio.append(str(audio_path))
                else:
                    test_texts.append(text)
                    test_labels.append(label)
                    test_audio.append(str(audio_path))

    train_ds = Dataset.from_dict(
        {"text": train_texts, "label": train_labels, "audio": train_audio}
    ).cast_column("audio", Audio())
    test_ds = Dataset.from_dict(
        {"text": test_texts, "label": test_labels, "audio": test_audio}
    ).cast_column("audio", Audio())

    return train_ds, test_ds, len(IEMOCAP_LABELS), IEMOCAP_LABEL_NAMES


def load_iemocap_features(
    feature_set: str = "trust",
) -> tuple[Dataset, Dataset, int, list[str]]:
    """
    Load IEMOCAP acoustic/prosodic features from data/iemocap_features.npz.

    The archive follows the same split convention as load_iemocap:
    Sessions 1-4 are train and Session 5 is test.
    """
    if _is_aligned_manifest(IEMOCAP_FEATURES_PATH):
        return load_iemocap_aligned()
    return _load_feature_npz(
        IEMOCAP_FEATURES_PATH, IEMOCAP_LABEL2IDX, IEMOCAP_LABEL_NAMES, feature_set
    )


def load_iemocap_aligned(
    output_dir: Path | None = None,
) -> tuple[Dataset, Dataset, int, list[str]]:
    """
    Load generated IEMOCAP word-aligned text/prosody artifacts.
    """
    from src.iemocap_aligned_dataset import (
        DEFAULT_OUTPUT_DIR,
        load_iemocap_aligned as _load_iemocap_aligned,
    )

    return _load_iemocap_aligned(output_dir or DEFAULT_OUTPUT_DIR)


# -------------------------
# MSP-PODCAST
# -------------------------

MSP_LABELS = [
    "A",  # Anger
    "S",  # Sadness
    "H",  # Happiness
    "U",  # Surprise
    "F",  # Fear
    "D",  # Disgust
    "C",  # Contempt
    "N",  # Neutral
]

MSP_LABEL_NAMES = [
    "anger",
    "sadness",
    "happiness",
    "surprise",
    "fear",
    "disgust",
    "contempt",
    "neutral",
]

MSP_LABEL2IDX = {l: i for i, l in enumerate(MSP_LABELS)}


def load_msp_podcast() -> tuple[Dataset, Dataset, int, list[str]]:
    """
    Load MSP-PODCAST dataset from local files.

    Uses Train split for training and Test1 for evaluation.
    Excludes 'X' (no agreement) and 'O' (other) labels from the labels

    Returns: (train_ds, test_ds, num_labels, label_names)
    """
    labels_path = MSP_ROOT / "Labels" / "labels_consensus.csv"
    transcripts_dir = MSP_ROOT / "Transcripts"
    audios_dir = MSP_ROOT / "Audios"

    train_texts, train_labels, train_audio = [], [], []
    test_texts, test_labels, test_audio = [], [], []

    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emo = row["EmoClass"]
            split = row["Split_Set"]

            # Skip excluded labels
            if emo not in MSP_LABEL2IDX:
                continue

            # Only use Train and Test1 (Test 2 could work, but maybe later)
            if split not in ("Train", "Test1"):
                continue

            filename = row["FileName"]
            transcript_file = transcripts_dir / filename.replace(".wav", ".txt")
            audio_file = audios_dir / filename
            if not transcript_file.exists() or not audio_file.exists():
                continue

            text = transcript_file.read_text().strip()
            if not text:
                continue

            label = MSP_LABEL2IDX[emo]

            if split == "Train":
                train_texts.append(text)
                train_labels.append(label)
                train_audio.append(str(audio_file))
            else:
                test_texts.append(text)
                test_labels.append(label)
                test_audio.append(str(audio_file))

    train_ds = Dataset.from_dict(
        {"text": train_texts, "label": train_labels, "audio": train_audio}
    ).cast_column("audio", Audio())
    test_ds = Dataset.from_dict(
        {"text": test_texts, "label": test_labels, "audio": test_audio}
    ).cast_column("audio", Audio())

    return train_ds, test_ds, len(MSP_LABELS), MSP_LABEL_NAMES


def load_msp_podcast_features(
    feature_set: str = "trust",
) -> tuple[Dataset, Dataset, int, list[str]]:
    """
    Load MSP-PODCAST acoustic/prosodic features from data/msp_features.npz.

    The archive follows labels_consensus.csv for Train and Test1 examples with
    one of the eight MSP primary emotion labels.
    """
    if _is_aligned_manifest(MSP_FEATURES_PATH):
        return load_msp_aligned()
    return _load_feature_npz(
        MSP_FEATURES_PATH, MSP_LABEL2IDX, MSP_LABEL_NAMES, feature_set
    )


def load_msp_aligned(
    output_dir: Path | None = None,
) -> tuple[Dataset, Dataset, int, list[str]]:
    """
    Load generated MSP-PODCAST word-aligned text/prosody artifacts.

    Build them first with:
        uv run python src/msp_aligned_dataset.py
    """
    from src.msp_aligned_dataset import (
        DEFAULT_OUTPUT_DIR,
        load_msp_aligned as _load_msp_aligned,
    )

    return _load_msp_aligned(output_dir or DEFAULT_OUTPUT_DIR)


def get_feature_names(
    dataset_name: str = "iemocap", feature_set: str = "trust"
) -> list[str]:
    """Return the acoustic feature names stored in one of the local npz archives."""
    npz_path = {
        "iemocap": IEMOCAP_FEATURES_PATH,
        "msp": MSP_FEATURES_PATH,
        "msp_podcast": MSP_FEATURES_PATH,
    }[dataset_name]
    if _is_aligned_manifest(npz_path):
        aligned_dir = (
            DATA_ROOT / "iemocap_aligned_features"
            if dataset_name == "iemocap"
            else DATA_ROOT / "msp_aligned_features"
        )
        return json.loads((aligned_dir / "feature_names.json").read_text())
    with np.load(npz_path, allow_pickle=True) as archive:
        records = _unpack_feature_records(archive["train_features"])
    return _feature_names(records, feature_set)


def _is_aligned_manifest(npz_path: Path) -> bool:
    if not npz_path.exists():
        return False
    with np.load(npz_path, allow_pickle=False) as archive:
        if "artifact_type" not in archive.files:
            return False
        return str(archive["artifact_type"]) == "aligned_word_prosody_manifest"


def _raw_iemocap_label_splits() -> tuple[list[str], list[str]]:
    train_labels, test_labels = [], []

    for session_num in range(1, 6):
        session_dir = IEMOCAP_ROOT / f"Session{session_num}"
        emo_dir = session_dir / "dialog" / "EmoEvaluation"
        trans_dir = session_dir / "dialog" / "transcriptions"
        wav_dir = session_dir / "sentences" / "wav"
        emo_files = [
            f
            for f in emo_dir.glob("*.txt")
            if f.stem not in ("Attribute", "Categorical", "Self-evaluation")
        ]

        for emo_file in sorted(emo_files):
            dialog_name = emo_file.stem
            trans_file = trans_dir / f"{dialog_name}.txt"
            if not trans_file.exists():
                continue

            utt2emo = _parse_iemocap_emo_file(emo_file)
            utt2text = _parse_iemocap_transcript(trans_file)

            for utt_id, emo_record in utt2emo.items():
                emo = emo_record[0]
                audio_path = wav_dir / dialog_name / f"{utt_id}.wav"
                if (
                    utt_id not in utt2text
                    or emo not in IEMOCAP_LABEL2IDX
                    or not audio_path.exists()
                ):
                    continue
                if session_num <= 4:
                    train_labels.append(emo)
                else:
                    test_labels.append(emo)

    return train_labels, test_labels


def _raw_msp_feature_label_splits() -> tuple[list[str], list[str]]:
    labels_path = MSP_ROOT / "Labels" / "labels_consensus.csv"
    train_labels, test_labels = [], []

    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emo = row["EmoClass"]
            split = row["Split_Set"]
            if emo not in MSP_LABEL2IDX:
                continue
            if split == "Train":
                train_labels.append(emo)
            elif split == "Test1":
                test_labels.append(emo)

    return train_labels, test_labels


def _compare_label_split(
    raw_labels: list[str], npz_labels: np.ndarray
) -> dict[str, Any]:
    archived_labels = [str(label) for label in npz_labels]
    return {
        "raw_count": len(raw_labels),
        "archive_count": len(archived_labels),
        "same_count": len(raw_labels) == len(archived_labels),
        "same_order": raw_labels == archived_labels,
        "raw_label_counts": dict(sorted(Counter(raw_labels).items())),
        "archive_label_counts": dict(sorted(Counter(archived_labels).items())),
    }


def verify_feature_archives() -> dict[str, dict[str, dict[str, Any]]]:
    """
    Check that the feature npz labels follow the original local dataset order.

    IEMOCAP is checked against the Session 1-4/Session 5 raw files. MSP-PODCAST
    is checked against labels_consensus.csv Train/Test1 rows with valid primary
    emotion labels, which is the form used by the feature archive.
    """
    iemocap_train, iemocap_test = _raw_iemocap_label_splits()
    msp_train, msp_test = _raw_msp_feature_label_splits()

    with np.load(IEMOCAP_FEATURES_PATH, allow_pickle=True) as iemocap_archive:
        iemocap_result = {
            "train": _compare_label_split(
                iemocap_train, iemocap_archive["train_labels"]
            ),
            "test": _compare_label_split(iemocap_test, iemocap_archive["test_labels"]),
        }

    with np.load(MSP_FEATURES_PATH, allow_pickle=True) as msp_archive:
        msp_result = {
            "train": _compare_label_split(msp_train, msp_archive["train_labels"]),
            "test": _compare_label_split(msp_test, msp_archive["test_labels"]),
        }

    return {"iemocap": iemocap_result, "msp_podcast": msp_result}
