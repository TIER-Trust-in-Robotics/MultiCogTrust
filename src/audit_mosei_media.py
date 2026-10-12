"""Seeded train/validation media checks and ASR/transcript disagreement audit.

This sample is a data diagnostic, not an emotion benchmark. ASR disagreement
flags review candidates; it does not prove an annotation is wrong. No test
media is decoded and no records are automatically relabeled or removed.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit_benchmark_data import read_mosei_csv, read_official_folds
from src.iemocap_embedding_cache import sha256, write_json


def words(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", html.unescape(text).lower())


def word_error_rate(reference, hypothesis):
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return None
    previous = list(range(len(hyp) + 1))
    for i, actual in enumerate(ref, 1):
        current = [i]
        for j, guess in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (actual != guess)))
        previous = current
    return previous[-1] / len(ref)


def main():
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/MOSEI")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/benchmarks/mosei_media_sample_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-split", type=int, default=12)
    args = parser.parse_args()
    if args.samples_per_split < 1:
        raise ValueError("Need at least one media example per development split")
    rows = read_mosei_csv(args.root / "label.csv")
    folds = read_official_folds(ROOT / "data/processed/benchmarks/protocol_sources/cmu_mosei_folds.py")
    rng = random.Random(args.seed)
    sampled = []
    for split in ("train", "valid"):
        pool = sorted((r for r in rows if r["split"] == split), key=lambda r: r["id"])
        sampled.extend([{**r, "selection": "seeded_sample"} for r in rng.sample(pool, min(args.samples_per_split, len(pool)))])
    sampled.extend([{**r, "selection": "training_fold_mismatch"} for r in rows
                    if r["split"] == "train" and r["video_id"] not in folds["train"]
                    and r["id"] not in {s["id"] for s in sampled}])
    if any(r["split"] == "test" for r in sampled):
        raise ValueError("Test examples are excluded from media development checks")
    model_path = Path(snapshot_download("Systran/faster-whisper-tiny.en", local_files_only=True))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {
        "seed": args.seed, "samples_per_split": args.samples_per_split,
        "ids": [r["id"] for r in sampled], "source_sha256": sha256(Path(__file__)),
        "label_csv_sha256": sha256(args.root / "label.csv"),
        "asr_model": "Systran/faster-whisper-tiny.en", "asr_revision": model_path.name,
        "asr_weights_sha256": sha256(model_path / "model.bin"), "beam_size": 1,
        "language": "en", "vad_filter": False, "audio_crop": "none",
        "test_decoding_or_inference": False, "sentiment_labels_used_for_selection": False,
    })
    model = WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=2)
    reports = []
    with (args.output / "samples.jsonl").open("x") as handle:
        for row in sampled:
            started = time.perf_counter()
            uid = row["id"]
            audio, video = args.root / "wav" / f"{uid}.wav", args.root / "Raw" / f"{uid}.mp4"
            report = {"id": uid, "split": row["split"], "selection": row["selection"], "reference": row["text"]}
            try:
                info = json.loads(subprocess.run([
                    "ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,duration", "-of", "json", str(video),
                ], capture_output=True, text=True, check=True, timeout=30).stdout)
                report["video_metadata"] = info
                raw = subprocess.run([
                    "ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1",
                ], capture_output=True, check=True, timeout=60).stdout
                decoded = np.frombuffer(raw, dtype="<i2")
                wav, rate = sf.read(audio, dtype="int16")
                report["wav_matches_decoded_video_audio_exactly"] = rate == 16000 and np.array_equal(decoded, wav)
                report["audio_sha256"], report["video_sha256"] = sha256(audio), sha256(video)
                report["duration_sec"] = len(wav) / rate
                segments, _ = model.transcribe(str(audio), beam_size=1, language="en", vad_filter=False)
                hypothesis = " ".join(segment.text.strip() for segment in segments).strip()
                report["asr_text"] = hypothesis
                report["word_error_rate"] = word_error_rate(row["text"], hypothesis)
            except (subprocess.SubprocessError, OSError, RuntimeError, ValueError) as exc:
                report["error"] = str(exc)
            report["elapsed_sec"] = time.perf_counter() - started
            reports.append(report)
            handle.write(json.dumps(report, allow_nan=False) + "\n")
            handle.flush()
            print(f"Media {len(reports)}/{len(sampled)} {uid}: WER={report.get('word_error_rate')} paired_audio={report.get('wav_matches_decoded_video_audio_exactly')}", flush=True)
    groups = {}
    for selection in ("seeded_sample", "training_fold_mismatch"):
        group = [r for r in reports if r["selection"] == selection]
        wer = [r["word_error_rate"] for r in group if r.get("word_error_rate") is not None]
        groups[selection] = {
            "count": len(group), "completed": sum("error" not in r for r in group),
            "median_wer": float(np.median(wer)) if wer else None,
            "wer_above_0_75": [r["id"] for r in group if r.get("word_error_rate", 0) > .75],
        }
    write_json(args.output / "summary.json", {
        "groups": groups, "count": len(reports),
        "exact_audio_video_pairs": sum(r.get("wav_matches_decoded_video_audio_exactly", False) for r in reports),
        "errors": [r for r in reports if "error" in r], "test_decoding_or_inference": False,
        "interpretation": "Small seeded audit plus separately selected fold-mismatch cases. WER is a review signal, not ground truth about dataset quality; human listening/reading is still needed. No automatic exclusions based on ASR.",
    })


if __name__ == "__main__":
    main()
