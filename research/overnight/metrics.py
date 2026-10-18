"""Shared metrics for the overnight analyses (Sessions 1-4 only)."""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import f1_score

from src.emotion_checkpoint import EMOTION_LABEL_MAP
from src.prepare_iemocap import DEFAULT_METADATA

RUNS = ROOT / "data/processed/research/runs"
META = {r["utterance_id"]: r for r in map(json.loads, Path(DEFAULT_METADATA).read_text().splitlines())
        if r["session"] in (1, 2, 3, 4)}
LABEL = {u: EMOTION_LABEL_MAP.get(r["label_code"], -1) for u, r in META.items()}


def scored(sessions):
    if any(session not in (1, 2, 3, 4) for session in sessions):
        raise ValueError("Only development sessions are allowed")
    return [u for u, r in META.items() if r["session"] in sessions and LABEL[u] >= 0]


def f1(uids, probs):
    """Macro F1 over all scored lines; a missing prediction (empty transcript) counts as wrong."""
    return float(f1_score([LABEL[u] for u in uids], [int(np.argmax(probs[u])) if u in probs else -1 for u in uids],
                          labels=[0, 1, 2, 3], average="macro", zero_division=0))


def change_accuracy(uids, probs):
    """Accuracy on lines whose label differs from the speaker's previous scored line."""
    chains = defaultdict(list)
    for u in uids:
        r = META[u]
        chains[(r["dialog_id"], r["speaker"])].append((r["dialog_start_sec"], u))
    hits = []
    for seq in chains.values():
        seq = [u for _, u in sorted(seq)]
        for prev, u in zip(seq, seq[1:]):
            if LABEL[u] != LABEL[prev]:
                hits.append(u in probs and int(np.argmax(probs[u])) == LABEL[u])
    return float(np.mean(hits))


def load(tag, name, kind="swa3"):
    session = int(name.split(":", 1)[0])
    if session not in (1, 2, 3, 4):
        raise ValueError("Only development sessions are allowed")
    with np.load(RUNS / f"{tag}.npz", allow_pickle=False) as data:
        ids = data[f"ids::{name}"].copy()
        probs = data[f"{kind}::{name}"].copy() if kind == "swa3" else data[f"probs::{name}"][[e - 1 for e in kind]].mean(0)
    if len(set(ids)) != len(ids) or probs.shape != (len(ids), 4) or not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1, atol=1e-5):
        raise ValueError("Invalid prediction array or repeated IDs")
    if any(uid not in META or META[uid]["session"] != session for uid in ids):
        raise ValueError("Prediction IDs do not belong to the declared development session")
    return dict(zip(ids, probs))


def oof(recipe, seed, text, kind="swa3"):
    merged = {}
    for h in (1, 2, 3):
        fold = load(f"loso{h}_{recipe}_s{seed}", f"{h}:{text}", kind)
        if set(merged) & set(fold):
            raise ValueError("OOF folds overlap")
        merged.update(fold)
    return merged
