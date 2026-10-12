"""Audit local MOSEI media/folds and MSP metadata without model test inference.

Only train/valid records enter development.jsonl. Hashing/header checks do not
establish semantic audio/transcript/video alignment; that is a separate gate.
The supplied CMU fold file is parsed as literals, never executed.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iemocap_embedding_cache import sha256, write_json


def read_mosei_csv(path):
    records, seen = [], set()
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            video, clip = row["video_id"], row["clip_id"]
            if any(not v or v in {".", ".."} or Path(v).name != v for v in (video, clip)):
                raise ValueError("Invalid media identifier")
            uid = f"{video}/{clip}"
            if uid in seen:
                raise ValueError(f"Duplicate MOSEI segment: {uid}")
            seen.add(uid)
            sentiment = float(row["label"])
            if not np.isfinite(sentiment) or abs(sentiment) > 3:
                raise ValueError(f"Invalid sentiment: {uid}")
            if row["mode"] not in {"train", "valid", "test"}:
                raise ValueError(f"Unknown split: {uid}")
            records.append({
                "id": uid, "video_id": video, "clip_id": clip,
                "text": row["text"], "sentiment": sentiment, "split": row["mode"],
            })
    if not records:
        raise ValueError("Empty MOSEI label file")
    return records


def read_official_folds(path):
    names = {f"standard_{s}_fold": s for s in ("train", "valid", "test")}
    folds = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                values = ast.literal_eval(node.value)
                if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
                    raise ValueError("Invalid official fold literal")
                if len(values) != len(set(values)):
                    raise ValueError("Duplicate video in official fold")
                folds[names[name]] = set(values)
    if set(folds) != {"train", "valid", "test"}:
        raise ValueError("Incomplete official folds")
    if any(folds[a] & folds[b] for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("Official folds overlap")
    return folds


def development_records(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if not rows or any(r["split"] not in {"train", "valid"} for r in rows):
        raise ValueError("Development manifest must exclude all test records")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate development record")
    if {r["split"] for r in rows} != {"train", "valid"}:
        raise ValueError("Both development splits are required")
    return rows


def audit_mosei(root, folds_path):
    rows = read_mosei_csv(root / "label.csv")
    folds = read_official_folds(folds_path)
    local = {s: {r["video_id"] for r in rows if r["split"] == s} for s in folds}
    overlaps = {f"{a}/{b}": sorted(local[a] & local[b]) for a, b in
                (("train", "valid"), ("train", "test"), ("valid", "test"))}
    errors, audio_hashes, texts = [], defaultdict(list), defaultdict(list)
    started = time.perf_counter()
    for i, row in enumerate(rows):
        uid = row["id"]
        audio, video = root / "wav" / f"{uid}.wav", root / "Raw" / f"{uid}.mp4"
        row.update(audio_path=str(audio.resolve()), video_path=str(video.resolve()))
        if not video.is_file() or not video.stat().st_size:
            errors.append({"id": uid, "error": "missing_or_empty_video"})
        else:
            row["video_bytes"] = video.stat().st_size
        try:
            info = sf.info(audio)
            if info.samplerate != 16000 or info.channels != 1 or info.frames <= 0:
                raise ValueError(f"Unexpected WAV format: {info.samplerate}/{info.channels}/{info.frames}")
            row.update(duration_sec=info.duration, audio_sha256=sha256(audio))
            audio_hashes[row["audio_sha256"]].append((uid, row["split"]))
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append({"id": uid, "error": str(exc)})
        normalized = " ".join(row["text"].lower().split())
        if not normalized:
            errors.append({"id": uid, "error": "empty_transcript"})
        texts[hashlib.sha256(normalized.encode()).hexdigest()].append((uid, row["split"]))
        row["official_fold_member"] = row["video_id"] in folds[row["split"]]
        if (i + 1) % 5000 == 0:
            print(f"Audited {i + 1}/{len(rows)} media pairs", flush=True)

    def cross_split(groups):
        return [items for items in groups.values() if len({s for _, s in items}) > 1]

    splits = {}
    for split in folds:
        subset = [r for r in rows if r["split"] == split]
        duration = [r["duration_sec"] for r in subset if "duration_sec" in r]
        splits[split] = {
            "segments": len(subset), "videos": len(local[split]),
            "zero_sentiment": sum(r["sentiment"] == 0 for r in subset),
            "rounded_class_counts": dict(sorted(Counter(int(np.rint(r["sentiment"])) for r in subset).items())),
            "outside_official_fold": sorted(local[split] - folds[split]),
            "official_videos_absent_locally": sorted(folds[split] - local[split]),
            "duration_percentiles_sec": dict(zip(("min", "p50", "p95", "max"), np.percentile(duration, [0, 50, 95, 100]).tolist())) if duration else {},
            "longer_than_8_seconds": sum(t > 8 for t in duration),
        }
    duplicate_audio = cross_split(audio_hashes)
    report = {
        "dataset": "local_MMSA_mirror_MOSEI", "label_csv_sha256": sha256(root / "label.csv"),
        "official_folds_sha256": sha256(folds_path), "splits": splits,
        "video_split_overlaps": overlaps, "cross_split_identical_wav_groups": duplicate_audio,
        "cross_split_normalized_transcript_groups": cross_split(texts),
        "errors": errors,
        "media_integrity_passed": not errors,
        "development_integrity_passed": not errors and not any(overlaps.values()) and not duplicate_audio,
        "exact_official_video_sets": all(local[s] == folds[s] for s in folds),
        "semantic_alignment_verified": False,
        "test_model_inference": False,
        "checks": "All WAV headers and file hashes; all video existence/sizes. No full video decode or semantic alignment check.",
        "elapsed_sec": time.perf_counter() - started,
    }
    return rows, report


def audit_msp(root):
    with zipfile.ZipFile(root / "Labels.zip") as archive:
        with archive.open("Labels/labels_consensus.csv") as handle:
            rows = list(csv.DictReader(io.TextIOWrapper(handle)))
    counts = defaultdict(Counter)
    for row in rows:
        counts[row["Split_Set"]][row["EmoClass"]] += 1
    return {
        "release": "2.0 according to local readme", "consensus_rows": len(rows),
        "split_emotion_counts": dict(counts),
        "labels_archive_sha256": sha256(root / "Labels.zip"),
        "partitions_sha256": sha256(root / "Partitions.txt"),
        "audio_available_in_this_folder": any(root.rglob("*.wav")),
        "test3_labels": "withheld; excluded from consensus file",
        "access_url": "https://www.lab-msp.com/MSP/MSP-Podcast.html",
        "status": "Metadata available; audio access must be restored before acoustic benchmarking",
    }


def quarantine_training_conflicts(rows, report):
    """Quarantine training source videos; validation/test records stay intact.

One identical clip can reveal a reuploaded source recording with differently
encoded neighboring clips. Exclude that training video's entire group.
"""
    reasons = defaultdict(list)
    for row in rows:
        if row["split"] == "train" and not row["official_fold_member"]:
            reasons[row["id"]].append("training_video_absent_from_official_fold")
    duplicate_source_videos = set()
    for group in report["cross_split_identical_wav_groups"]:
        for uid, split in group:
            if split == "train":
                reasons[uid].append("training_wav_identical_to_reserved_split")
                duplicate_source_videos.add(uid.rsplit("/", 1)[0])
    conflicted_videos = set(report["video_split_overlaps"]["train/valid"]) | set(report["video_split_overlaps"]["train/test"])
    for row in rows:
        if row["split"] == "train" and row["video_id"] in conflicted_videos:
            reasons[row["id"]].append("training_video_in_reserved_split")
        if row["split"] == "train" and row["video_id"] in duplicate_source_videos:
            reasons[row["id"]].append("training_source_video_contains_cross_split_duplicate_audio")
    retained = [r for r in rows if r["id"] not in reasons]
    remaining_hashes = defaultdict(set)
    for row in retained:
        if "audio_sha256" in row:
            remaining_hashes[row["audio_sha256"]].add(row["split"])
    passed = not report["errors"] and not report["video_split_overlaps"]["valid/test"] and not any(len(s) > 1 for s in remaining_hashes.values())
    report["raw_development_integrity_passed"] = report["development_integrity_passed"]
    report["development_integrity_passed"] = passed
    report["development_view"] = {
        "name": "mosei_sdk_intersection_source_deduplicated_v2",
        "selection_uses_emotion_or_sentiment_labels": False,
        "exclusions": [{"id": uid, "split": "train", "reasons": why} for uid, why in sorted(reasons.items())],
        "retained_counts": dict(Counter(r["split"] for r in retained)),
        "interpretation": "Modified training set; not an exact standard-fold benchmark. No raw files changed. Validation/test examples retained.",
    }
    return retained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mosei-root", type=Path, default=ROOT / "data/MOSEI")
    parser.add_argument("--folds", type=Path, default=ROOT / "data/processed/benchmarks/protocol_sources/cmu_mosei_folds.py")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/benchmarks/mosei_native")
    parser.add_argument("--quarantine-training-conflicts", action="store_true",
                        help="Create an explicit modified view excluding training fold outsiders and cross-split duplicates")
    args = parser.parse_args()
    if (args.output / "audit.json").exists():
        raise FileExistsError("Preserving prior audit; choose a new --output")
    rows, report = audit_mosei(args.mosei_root, args.folds)
    if args.quarantine_training_conflicts:
        rows = quarantine_training_conflicts(rows, report)
    write_json(args.output / "audit.json", report)
    write_json(args.output / "msp_inventory.json", audit_msp(ROOT / "docs/MSP-Podcast"))
    if not report["development_integrity_passed"]:
        raise ValueError("Media/split integrity failed; inspect audit.json before training")
    manifest = args.output / "development.jsonl"
    with manifest.open("x") as handle:
        for row in rows:
            if row["split"] != "test":
                handle.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(args.output / "development_contract.json", {
        "manifest_sha256": sha256(manifest), "audit_sha256": sha256(args.output / "audit.json"),
        "label_csv_sha256": report["label_csv_sha256"],
        "protocol": report.get("development_view", {}).get("name", "local_mirror_folds"),
        "test_model_inference": False,
    })
    print(json.dumps({k: v for k, v in report.items() if k not in {"cross_split_normalized_transcript_groups"}}, indent=2))


if __name__ == "__main__":
    main()
