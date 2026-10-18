"""Check long cross-split transcript matches for near-identical audio.

This label-blind audit decodes candidate WAVs, including test WAVs, solely to
check split integrity. It performs no model inference or automatic exclusions.
Candidates above 0.98 correlation need source-level review before final testing.
This targeted check is not an exhaustive near-duplicate or speaker-ID audit.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import correlate
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit_benchmark_data import read_mosei_csv
from src.iemocap_embedding_cache import sha256, write_json


def audio_similarity(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or not len(x) or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Expected finite, nonempty mono waveforms")
    x, y = x - x.mean(), y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    if denominator == 0:
        return None, None
    values = correlate(x, y, method="fft")
    index = int(np.argmax(np.abs(values)))
    return float(abs(values[index]) / denominator), index - len(y) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/MOSEI")
    parser.add_argument("--audit", type=Path, default=ROOT / "data/processed/benchmarks/mosei_clean_v2/audit.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/benchmarks/mosei_clean_v2/near_duplicate_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserving existing duplicate audit; choose a new --output")
    audit = json.loads(args.audit.read_text())
    if audit["label_csv_sha256"] != sha256(args.root / "label.csv"):
        raise ValueError("Label file differs from the media audit")
    rows = {r["id"]: r for r in read_mosei_csv(args.root / "label.csv")}
    reports = []
    for group in audit["cross_split_normalized_transcript_groups"]:
        if len(rows[group[0][0]]["text"].split()) < 10:
            continue
        for (a, split_a), (b, split_b) in itertools.combinations(group, 2):
            if split_a == split_b:
                continue
            pa, pb = args.root / "wav" / f"{a}.wav", args.root / "wav" / f"{b}.wav"
            x, rate_x = sf.read(pa, dtype="float32")
            y, rate_y = sf.read(pb, dtype="float32")
            if rate_x != 16000 or rate_y != rate_x:
                raise ValueError("Duplicate audit requires matching 16 kHz WAVs")
            score, lag = audio_similarity(x, y)
            reports.append({
                "a": a, "split_a": split_a, "b": b, "split_b": split_b,
                "audio_sha256_a": sha256(pa), "audio_sha256_b": sha256(pb),
                "normalized_peak_correlation": score, "lag_sec": lag / rate_x if lag is not None else None,
                "duration_a_sec": len(x) / rate_x, "duration_b_sec": len(y) / rate_y,
            })
    candidates = [r for r in reports if r["normalized_peak_correlation"] is not None and r["normalized_peak_correlation"] >= .98]
    report = {
        "method": "Full-waveform normalized cross-correlation; exact normalized transcript matches with >=10 whitespace words only",
        "threshold": .98, "pairs_checked": len(reports), "pairs": reports,
        "strong_duplicate_candidates": candidates,
        "validation_test_candidates": [r for r in candidates if {r["split_a"], r["split_b"]} == {"valid", "test"}],
        "test_model_inference": False, "source_sha256": sha256(Path(__file__)),
        "label_csv_sha256": audit["label_csv_sha256"],
        "limitation": "Not exhaustive; missed matches, common scripts and shared speaker identities require further review. No exclusions based on this diagnostic alone.",
    }
    write_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
