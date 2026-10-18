"""Recompute the five-seed A/B table and an auditable structured report.

Reads original predictions only: no inference or Session 4 selection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

from metrics import RUNS, change_accuracy, f1, load, oof, scored
from src.iemocap_embedding_cache import sha256, write_json

SEEDS = (42, 43, 44, 45, 46)
s123, s4 = scored((1, 2, 3)), scored((4,))
rows, hashes = {}, {}
for recipe, ref, asr in (("ref", "reference", "asr"), ("prev", "prevref", "prevasr")):
    for seed in SEEDS:
        o = oof(recipe, seed, ref)
        w = load(f"full_{recipe}_s{seed}", f"4:{asr}")
        r = load(f"full_{recipe}_s{seed}", f"4:{ref}")
        if set(o) != set(s123) or set(r) != set(s4) or len(w) != 1027 or not set(w) <= set(s4):
            raise ValueError("Original prediction coverage changed")
        rows[recipe, seed] = (f1(s123, o), change_accuracy(s123, o), f1(s4, w), change_accuracy(s4, w), f1(s4, r), change_accuracy(s4, r))
        for tag in [f"full_{recipe}_s{seed}", *[f"loso{held}_{recipe}_s{seed}" for held in (1, 2, 3)]]:
            hashes[tag] = sha256(RUNS / f"{tag}.npz")
names = ("OOF F1", "OOF change", "S4 Whisper F1", "S4 Whisper change", "S4 ref F1", "S4 ref change")
print("| metric | A recipe mean (sd) | B recipe mean (sd) | B-A paired mean | 95% CI | B wins |")
print("|---|---|---|---|---|---|")
summary = {}
for j, name in enumerate(names):
    a = np.array([rows["ref", s][j] for s in SEEDS]); b = np.array([rows["prev", s][j] for s in SEEDS])
    d = b - a
    half = 2.776 * d.std(ddof=1) / np.sqrt(len(d))  # t, 4 df
    summary[name] = {"a": a.tolist(), "b": b.tolist(), "a_mean": float(a.mean()), "b_mean": float(b.mean()),
                     "a_sd": float(a.std(ddof=1)), "b_sd": float(b.std(ddof=1)), "difference": float(d.mean()),
                     "ci95_seed_only": [float(d.mean() - half), float(d.mean() + half)], "b_wins": int((d > 0).sum())}
    print(f"| {name} | {a.mean():.4f} ({a.std(ddof=1):.4f}) | {b.mean():.4f} ({b.std(ddof=1):.4f}) | {d.mean():+.4f} | "
          f"[{d.mean() - half:+.4f}, {d.mean() + half:+.4f}] | {(d > 0).sum()}/5 |")
write_json(RUNS.parent / "phase1_summary.json", {"finished": True, "seeds": SEEDS, "runs": len(hashes), "metrics": summary,
           "prediction_sha256": hashes, "analyzer_sha256": sha256(Path(__file__)), "test_model_inference": False,
           "counts": {"oof": len(s123), "session4": len(s4), "session4_asr_available": 1027},
           "limitations": ["Intervals describe seed variation on fixed data, not uncertainty across new people or domains.",
                           "The original 40 runs retained predictions but not checkpoint weights or per-run source manifests.",
                           "A is the retrained current-line recipe, not the deployed checkpoint selected on Session 4.",
                           "Missing ASR predictions count as errors in macro F1 and change accuracy."]})
