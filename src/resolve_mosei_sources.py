"""Resolve the previously flagged source groups and freeze a raw-data view.

Decodes test media solely for label-blind source integrity checks. Exclusions
are predetermined from exact/near-duplicate recordings, never sentiment.
This targeted review does not establish exhaustive speaker-disjoint splits.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit_benchmark_data import read_mosei_csv, development_records
from src.audit_mosei_duplicates import audio_similarity
from src.iemocap_embedding_cache import sha256, write_json

GROUPS = [
    {"videos": ["VN1IPN1TL3Q", "wO8fUOC4OSE"], "pairs": [["VN1IPN1TL3Q/0", "wO8fUOC4OSE/0"], ["VN1IPN1TL3Q/5", "wO8fUOC4OSE/5"]], "exclude_video": "VN1IPN1TL3Q", "exclude_split": "train", "review": "Duplicated recording clips: matching presenter, background and motion; exact WAV duplicates."},
    {"videos": ["i5bmKlHv-SA", "svsbGQn389o"], "pairs": [["i5bmKlHv-SA/4", "svsbGQn389o/6"]], "exclude_video": "i5bmKlHv-SA", "exclude_split": "train", "review": "Duplicated recording clip: matching presenter, office background and motion; exact WAV duplicate."},
    {"videos": ["jrDqduyQrfo", "WbtsuXkaGeg"], "pairs": [["jrDqduyQrfo/1", "WbtsuXkaGeg/1"]], "exclude_video": "jrDqduyQrfo", "exclude_split": "valid", "review": "Shared prerecorded Expert Village introduction confirmed by audio and video. Other clips cover different topics with the same visually matching presenter. Conservatively quarantine the whole validation video group."},
]


def frame(path, second):
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_MSEC, 1000 * second)
    ok, value = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"Could not decode source frame: {path}")
    return cv2.resize(cv2.cvtColor(value, cv2.COLOR_BGR2GRAY), (160, 120)).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/benchmarks/week1_v1/source_resolution")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserving existing source review")
    raw = ROOT / "data/MOSEI"
    rows = {r["id"]: r for r in read_mosei_csv(raw / "label.csv")}
    prior = ROOT / "data/processed/benchmarks/mosei_clean_v2"
    audit = json.loads((prior / "audit.json").read_text())
    if sha256(raw / "label.csv") != audit["label_csv_sha256"]:
        raise ValueError("Raw source labels changed since the audit")
    groups = []
    for group in GROUPS:
        evidence = []
        for a, b in group["pairs"]:
            wa, wb = raw / "wav" / f"{a}.wav", raw / "wav" / f"{b}.wav"
            x, rx = sf.read(wa, dtype="float32")
            y, ry = sf.read(wb, dtype="float32")
            if rx != 16000 or ry != rx:
                raise ValueError("Unexpected source audio format")
            score, lag = audio_similarity(x, y)
            if score is None or score < .98:
                raise ValueError("Flagged source pair no longer meets the frozen duplicate threshold")
            va, vb = raw / "Raw" / f"{a}.mp4", raw / "Raw" / f"{b}.mp4"
            frames = []
            for second in (.5, 1., 2.):
                fa, fb = frame(va, second), frame(vb, max(0, second - lag / rx))
                frames.append({"time_a_sec": second, "time_b_sec": max(0, second - lag / rx), "grayscale_correlation": float(np.corrcoef(fa.ravel(), fb.ravel())[0, 1]), "pixel_mae": float(np.abs(fa - fb).mean())})
            evidence.append({"a": a, "b": b, "split_a": rows[a]["split"], "split_b": rows[b]["split"], "wav_sha256_a": sha256(wa), "wav_sha256_b": sha256(wb), "video_sha256_a": sha256(va), "video_sha256_b": sha256(vb), "audio_correlation": score, "audio_lag_sec": lag / rx, "video_samples": frames})
        groups.append({**group, "evidence": evidence})
    existing = development_records(prior / "development.jsonl")
    exclusions = [r for r in existing if r["split"] == "valid" and r["video_id"] == "jrDqduyQrfo"]
    retained = [r for r in existing if r not in exclusions]
    if Counter(r["split"] for r in retained) != {"train": 16317, "valid": 1867}:
        raise ValueError("Unexpected reviewed development counts")
    args.output.mkdir(parents=True)
    manifest = args.output / "development.jsonl"
    with manifest.open("x") as handle:
        for row in retained:
            handle.write(json.dumps(row) + "\n")
    report = {
        "protocol": "mosei_sdk_intersection_source_deduplicated_v3",
        "label_csv_sha256": audit["label_csv_sha256"],
        "source_sha256": sha256(Path(__file__)),
        "prior_development_contract_sha256": sha256(prior / "development_contract.json"),
        "groups": groups, "known_recording_groups_resolved": True,
        "exclusion_decisions_use_sentiment_labels": False,
        "new_exclusions": [{"id": r["id"], "split": "valid", "reason": "whole_validation_source_video_with_shared_test_introduction"} for r in exclusions],
        "prior_training_exclusions": audit["development_view"]["exclusions"],
        "development_counts": dict(Counter(r["split"] for r in retained)),
        "test_count_retained": sum(r["split"] == "test" for r in rows.values()),
        "test_model_inference": False,
        "visual_review_artifact": str(ROOT / "data/processed/benchmarks/week1_v1/source_review/flagged_clips.png"),
        "visual_review_sha256": sha256(ROOT / "data/processed/benchmarks/week1_v1/source_review/flagged_clips.png"),
        "remaining_semantic_warnings": [
            "QxU625Hn370/1: reference/ASR disagreement unresolved; retain pending authoritative transcript review, no exclusion based on ASR alone.",
            "268836/5: short reference/ASR disagreement; plausible name-transcription error, retain.",
            "-NFrJFQijFE: all four clips already excluded as official-fold outsiders. Their WAVs match video audio, but sampled ASR/transcript disagreement does not establish replacement labels or the original intended source.",
        ],
        "limitations": ["Targeted review of flagged groups only; no exhaustive speaker-identity or near-duplicate audit.", "Matching introduction does not mean every surrounding clip is the same utterance; whole-video exclusion is conservative.", "Historical validation/test model exposure remains disclosed; cleaning does not restore a pristine test set.", "Original MMSA feature references retain their original split; their results have the known source-overlap caveat."],
    }
    write_json(args.output / "report.json", report)
    write_json(args.output / "development_contract.json", {"protocol": report["protocol"], "manifest_sha256": sha256(manifest), "report_sha256": sha256(args.output / "report.json"), "label_csv_sha256": report["label_csv_sha256"], "test_model_inference": False})
    print(json.dumps({"known_groups_resolved": True, "development_counts": report["development_counts"], "new_exclusions": report["new_exclusions"], "test_model_inference": False}, indent=2))


if __name__ == "__main__":
    main()
