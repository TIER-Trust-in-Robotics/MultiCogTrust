"""
Loader for the datasets into Dataset objects, current only usex the openSMILE features from the audio

[openSMILE feature] :
"""

from pathlib import Path
from datasets import Dataset
import csv
import pickle
import sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from speechProsodic import ProsodicFeatureExtractor


DATA_ROOT = Path(__file__).resolve().parent  # path to datasets folder
# DATA_ROOT = "__name__".
IEMOCAP_ROOT = DATA_ROOT / "IEMOCAP_full_release"
MSP_ROOT = DATA_ROOT / "MSP-PODCAST"

# -------------------------
# IEMOCAP
# -------------------------

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

BINARY_IEMOCAP_LABELS = {
    "ang": 0,
    "hap": 0,
    "exc": 0,
    "neu": 1,
    "sad": 0,
    "fru": 0,
    "fea": 0,
    "sur": 0,
    "dis": 0,
    "xxx": 1,
    "oth": 0,
}

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


def _parse_iemocap_emo_file(emo_path: Path) -> dict:
    """Parse an IEMOCAP EmoEvaluation file. Returns {utterance_id: emotion}."""
    utt2emo = {}
    with open(emo_path) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("["):
                continue
            # Format: [start - end]\t<UTTERANCE_ID>\t<EMOTION>\t[V(alence), A(ctivation), D(ominance)]
            # \t = tabx
            parts = line.split("\t")
            if len(parts) >= 3:
                utt_id = parts[1].strip()
                emo = parts[2].strip()
                utt2emo[utt_id] = emo
    return utt2emo


def _extract_iemocap_raw():
    """Extract features and collect original string labels (no index conversion)."""
    osExtractor = ProsodicFeatureExtractor()

    train_features, train_labels = [], []
    test_features, test_labels = [], []

    for session_num in range(1, 6):
        session_dir = IEMOCAP_ROOT / f"Session{session_num}"
        emo_dir = session_dir / "dialog" / "EmoEvaluation"
        wav_dir = session_dir / "sentences" / "wav"

        emo_files = [
            f
            for f in emo_dir.glob("*.txt")
            if f.stem not in ("Attribute", "Categorical", "Self-evaluation")
        ]

        utterances = [
            (utt_id, emo, wav_dir / emo_file.stem / f"{utt_id}.wav")
            for emo_file in sorted(emo_files)
            for utt_id, emo in _parse_iemocap_emo_file(emo_file).items()
        ]

        for utt_id, emo, audio_path in tqdm(
            utterances, desc=f"IEMOCAP Session{session_num}", leave=False
        ):
            if not audio_path.exists():
                continue

            features = osExtractor.extract(str(audio_path))

            if session_num <= 4:
                train_features.append(features)
                train_labels.append(emo)
            else:
                test_features.append(features)
                test_labels.append(emo)

    return train_features, train_labels, test_features, test_labels


def load_iemocap(binary: bool = False):
    train_features, train_str, test_features, test_str = _extract_iemocap_raw()

    if binary:
        train_labels = [BINARY_IEMOCAP_LABELS[l] for l in train_str]
        test_labels = [BINARY_IEMOCAP_LABELS[l] for l in test_str]
        n_labels = 2
        label_names = ["non-neutral", "neutral"]
    else:
        train_labels = [IEMOCAP_LABEL2IDX[l] for l in train_str]
        test_labels = [IEMOCAP_LABEL2IDX[l] for l in test_str]
        n_labels = len(IEMOCAP_LABELS)
        label_names = IEMOCAP_LABEL_NAMES

    train_ds = Dataset.from_dict(
        {"audio_features": train_features, "labels": train_labels}
    )
    test_ds = Dataset.from_dict(
        {"audio_features": test_features, "labels": test_labels}
    )

    return train_ds, test_ds, n_labels, label_names


# -------------------------
# MSP_PODCAST
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

BINARY_MSP_LABEL = {
    "A": 0,
    "S": 0,
    "H": 0,
    "U": 0,
    "F": 0,
    "D": 0,
    "C": 0,
    "N": 1,
}

MSP_LABEL2IDX = {l: i for i, l in enumerate(MSP_LABELS)}


def _extract_msp_raw():
    """Extract features and collect original string labels (no index conversion)."""
    osExtractor = ProsodicFeatureExtractor()
    labels_path = MSP_ROOT / "Labels" / "labels_consensus.csv"
    audios_dir = MSP_ROOT / "Audios"

    train_features, train_labels = [], []
    test_features, test_labels = [], []

    with open(labels_path, newline="") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["EmoClass"] in MSP_LABEL2IDX and r["Split_Set"] in ("Train", "Test1")
        ]

    for row in tqdm(rows, desc="MSP-PODCAST"):
        file_name = row["FileName"]
        emo = row["EmoClass"]
        split = row["Split_Set"]

        audio_path = audios_dir / file_name
        feature = osExtractor.extract(str(audio_path))

        if split == "Train":
            train_features.append(feature)
            train_labels.append(emo)
        else:
            test_features.append(feature)
            test_labels.append(emo)

    return train_features, train_labels, test_features, test_labels


def load_msp_podcast(binary: bool = False):
    train_features, train_str, test_features, test_str = _extract_msp_raw()

    if binary:
        train_labels = [BINARY_MSP_LABEL[l] for l in train_str]
        test_labels = [BINARY_MSP_LABEL[l] for l in test_str]
        n_labels = 2
        label_names = ["non-neutral", "neutral"]
    else:
        train_labels = [MSP_LABEL2IDX[l] for l in train_str]
        test_labels = [MSP_LABEL2IDX[l] for l in test_str]
        n_labels = len(MSP_LABELS)
        label_names = MSP_LABEL_NAMES

    train_ds = Dataset.from_dict(
        {"audio_features": train_features, "labels": train_labels}
    )
    test_ds = Dataset.from_dict(
        {"audio_features": test_features, "labels": test_labels}
    )

    return train_ds, test_ds, n_labels, label_names


# -------------------------
# Save / Load
# -------------------------

DATASET_BINARY_LABELS = {
    "iemocap": BINARY_IEMOCAP_LABELS,
    "msp": BINARY_MSP_LABEL,
}

DATASET_LABEL2IDX = {
    "iemocap": IEMOCAP_LABEL2IDX,
    "msp": MSP_LABEL2IDX,
}


def save_dataset_npz(
    path: str,
    train_features: list,
    train_labels: list[str],
    test_features: list,
    test_labels: list[str],
    dataset_name: str,
):
    """Save extracted features and original string labels to a .npz file.

    Args:
        path: Output file path (e.g. "iemocap_features.npz").
        train_features: List of feature dicts from ProsodicFeatureExtractor.
        train_labels: List of original string labels (e.g. "ang", "neu", "N").
        test_features: List of feature dicts for the test split.
        test_labels: List of original string labels for the test split.
        dataset_name: One of "iemocap" or "msp" — stored so the loader knows
                      which label mapping to use.
    """
    np.savez(
        path,
        train_features=np.array(pickle.dumps(train_features)),
        train_labels=np.array(train_labels),
        test_features=np.array(pickle.dumps(test_features)),
        test_labels=np.array(test_labels),
        dataset_name=np.array(dataset_name),
    )


def load_dataset_npz(path: str, binary: bool = False):
    """Load a .npz dataset saved by save_dataset_npz.

    Args:
        path: Path to the .npz file.
        binary: If True, map labels to neutral (1) vs non-neutral (0).
                If False, use original multi-class label indices.

    Returns:
        (train_ds, test_ds, n_labels, label_names)
    """
    data = np.load(path, allow_pickle=True)

    train_features = pickle.loads(data["train_features"].item())
    test_features = pickle.loads(data["test_features"].item())
    train_str_labels = data["train_labels"].tolist()
    test_str_labels = data["test_labels"].tolist()
    dataset_name = str(data["dataset_name"])

    label2idx = DATASET_LABEL2IDX[dataset_name]
    binary_map = DATASET_BINARY_LABELS[dataset_name]

    if binary:
        train_labels = [binary_map[l] for l in train_str_labels]
        test_labels = [binary_map[l] for l in test_str_labels]
        n_labels = 2
        label_names = ["non-neutral", "neutral"]
    else:
        train_labels = [label2idx[l] for l in train_str_labels]
        test_labels = [label2idx[l] for l in test_str_labels]
        if dataset_name == "iemocap":
            n_labels = len(IEMOCAP_LABELS)
            label_names = IEMOCAP_LABEL_NAMES
        else:
            n_labels = len(MSP_LABELS)
            label_names = MSP_LABEL_NAMES

    train_ds = Dataset.from_dict(
        {"audio_features": train_features, "labels": train_labels}
    )
    test_ds = Dataset.from_dict(
        {"audio_features": test_features, "labels": test_labels}
    )

    return train_ds, test_ds, n_labels, label_names


if __name__ == "__main__":
    """
    Extract features from both datasets and save to .npz files.
    Should only need to run once.
    """
    # print("Extracting IEMOCAP features...")
    # ie_train_feat, ie_train_lbl, ie_test_feat, ie_test_lbl = _extract_iemocap_raw()
    # save_dataset_npz(
    #     str(DATA_ROOT / "iemocap_features.npz"),
    #     ie_train_feat,
    #     ie_train_lbl,
    #     ie_test_feat,
    #     ie_test_lbl,
    #     dataset_name="iemocap",
    # )
    # print(
    #     f"  Saved iemocap_features.npz  (train={len(ie_train_lbl)}, test={len(ie_test_lbl)})"
    # )

    print("Extracting MSP-PODCAST features...")
    msp_train_feat, msp_train_lbl, msp_test_feat, msp_test_lbl = _extract_msp_raw()
    save_dataset_npz(
        str(DATA_ROOT / "msp_features.npz"),
        msp_train_feat,
        msp_train_lbl,
        msp_test_feat,
        msp_test_lbl,
        dataset_name="msp",
    )
    print(
        f"  Saved msp_features.npz  (train={len(msp_train_lbl)}, test={len(msp_test_lbl)})"
    )

    print("Done.")
