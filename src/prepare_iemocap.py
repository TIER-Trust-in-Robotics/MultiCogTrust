"""Audit restored IEMOCAP audio/video and write metadata without extracting features.

Raw 11-class labels and Session 5 are preserved. The four-class experiment is
an additional view, not a mutation of the original labels. Video timestamps
refer to full dialogs; speaker crops and learned visual features are later work.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.emotion_checkpoint import EMOTION_LABEL_MAP, EMOTION_LABELS
from src.iemocap_aligned_dataset import (
    parse_wdseg,
    transcript_match_stats,
    validate_words,
)
from to_ds import IEMOCAP_LABEL2IDX, _parse_iemocap_emo_file, _parse_iemocap_transcript

DEFAULT_OUTPUT = ROOT / "data/processed/iemocap_manifest"
DEFAULT_METADATA = DEFAULT_OUTPUT / "metadata.jsonl"


def alignment_end(path):
    ends = []
    for line in path.read_text().splitlines():
        match = re.match(r"^\s*(\d+)\s+(\d+)\s+[-\d]+\s+\S+\s*$", line)
        if match:
            ends.append((int(match[2]) + 1) * 0.01)
    return max(ends) if ends else None


def summary(values):
    if not values:
        return {"count": 0}
    values = np.asarray(values)
    return {
        "count": len(values),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def probe_video(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    if not info.get("streams"):
        raise ValueError(f"No video stream: {path}")
    return {
        "path": str(path),
        "duration_sec": float(info["format"]["duration"]),
        **info["streams"][0],
    }


def prepare(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not all((root / f"Session{s}").is_dir() for s in range(1, 6)):
        raise FileNotFoundError("All five IEMOCAP sessions must be restored")
    video_paths = sorted(root.glob("Session*/dialog/avi/DivX/*.avi"))
    with ThreadPoolExecutor(max_workers=4) as pool:
        videos = {
            Path(item["path"]).stem: item for item in pool.map(probe_video, video_paths)
        }
    records, errors, warnings = [], [], []
    counts = Counter()
    duration_deltas, lexical_deltas, word_counts = [], [], []
    seen = set()
    for session in range(1, 6):
        session_dir = root / f"Session{session}"
        for emo_path in sorted((session_dir / "dialog/EmoEvaluation").glob("*.txt")):
            dialog = emo_path.stem
            transcript_path = session_dir / "dialog/transcriptions" / f"{dialog}.txt"
            emotions = _parse_iemocap_emo_file(emo_path)
            transcripts = (
                _parse_iemocap_transcript(transcript_path)
                if transcript_path.exists()
                else {}
            )
            spans = {}
            for line in emo_path.read_text().splitlines():
                match = re.match(r"\[([\d.]+) - ([\d.]+)\]\s+(\S+)", line)
                if match:
                    spans[match[3]] = [float(match[1]), float(match[2])]
            counts["labeled_utterances"] += len(emotions)
            for uid, annotation in emotions.items():
                code, *vad = annotation
                wav = session_dir / "sentences/wav" / dialog / f"{uid}.wav"
                wdseg = (
                    session_dir / "sentences/ForcedAlignment" / dialog / f"{uid}.wdseg"
                )
                if uid in seen:
                    errors.append({"utterance_id": uid, "error": "duplicate id"})
                    continue
                seen.add(uid)
                if (
                    code not in IEMOCAP_LABEL2IDX
                    or uid not in transcripts
                    or not wav.exists()
                ):
                    errors.append(
                        {
                            "utterance_id": uid,
                            "error": "missing transcript/WAV or unknown label",
                        }
                    )
                    continue
                info = sf.info(wav)
                duration = info.frames / info.samplerate
                if duration <= 0 or info.samplerate != 16000:
                    errors.append(
                        {"utterance_id": uid, "error": "invalid WAV duration/rate"}
                    )
                    continue
                counts["with_labels_transcripts_wavs"] += 1
                words, word_errors, match_stats = [], [], None
                final_end = None
                if wdseg.exists():
                    counts["with_wdseg"] += 1
                    words = parse_wdseg(wdseg)
                    word_errors = validate_words(words, duration, 0.05)
                    final_end = alignment_end(wdseg)
                    if final_end is not None:
                        duration_deltas.append(final_end - duration)
                    if words:
                        lexical_deltas.append(words[-1].end_sec - duration)
                        match_stats = transcript_match_stats(transcripts[uid], words)
                    if not words:
                        word_errors.append("no lexical words")
                    if word_errors:
                        warnings.append(
                            {"utterance_id": uid, "alignment_errors": word_errors}
                        )
                        if any("nonmonotonic" in issue for issue in word_errors):
                            errors.append(
                                {
                                    "utterance_id": uid,
                                    "error": "nonmonotonic word timing",
                                }
                            )
                        words = []  # Never include invalid word spans as usable alignment.
                    else:
                        counts["with_valid_word_spans"] += 1
                else:
                    warnings.append({"utterance_id": uid, "warning": "missing wdseg"})
                start, end = spans[uid]
                video = videos.get(dialog)
                video_span_valid = False
                if video is None:
                    errors.append(
                        {"utterance_id": uid, "error": "missing dialog video"}
                    )
                elif end > video["duration_sec"] + 0.1:
                    warnings.append(
                        {
                            "utterance_id": uid,
                            "warning": "video ends before utterance",
                            "missing_tail_sec": end - video["duration_sec"],
                        }
                    )
                else:
                    video_span_valid = True
                    counts["with_video_span"] += 1
                if start < 0 or end <= start or abs((end - start) - duration) > 0.1:
                    errors.append(
                        {"utterance_id": uid, "error": "dialog/WAV timing mismatch"}
                    )
                word_counts.append(len(words))
                record = {
                    "utterance_id": uid,
                    "dialog_id": dialog,
                    "session": session,
                    "split": "train" if session <= 4 else "test",
                    "text": transcripts[uid],
                    "label_code": code,
                    "label": IEMOCAP_LABEL2IDX[code],
                    "emotion4_label": EMOTION_LABEL_MAP.get(code),
                    "vad": vad,
                    "audio_path": str(wav),
                    "duration_sec": duration,
                    "sample_rate": info.samplerate,
                    "video_path": video["path"] if video else None,
                    "video_span_valid": video_span_valid,
                    "video_duration_sec": video["duration_sec"] if video else None,
                    "speaker": uid.rsplit("_", 1)[-1][0],
                    "dialog_start_sec": start,
                    "dialog_end_sec": end,
                    "wdseg_path": str(wdseg) if wdseg.exists() else None,
                    "alignment_valid": bool(words),
                    "alignment_errors": word_errors,
                    "alignment_end_sec_including_silence": final_end,
                    "words": [vars(word) for word in words],
                    "transcript_alignment": match_stats,
                }
                records.append(record)
        print(f"Session {session}: {len(records)} metadata records audited", flush=True)
    wav_count = sum(1 for _ in root.glob("Session*/sentences/wav/*/*.wav"))
    if len(records) != counts["labeled_utterances"]:
        errors.append({"error": "metadata/label count mismatch"})
    if duration_deltas and (
        abs(float(np.median(duration_deltas))) > 0.05
        or np.mean(np.abs(duration_deltas) > 0.1) > 0.05
    ):
        errors.append({"error": "systematic forced alignment/WAV duration mismatch"})
    split_counts = Counter(r["split"] for r in records)
    label_counts = {
        split: dict(Counter(r["label_code"] for r in records if r["split"] == split))
        for split in ("train", "test")
    }
    emotion4_counts = {
        split: dict(
            Counter(
                EMOTION_LABELS[r["emotion4_label"]]
                for r in records
                if r["split"] == split and r["emotion4_label"] is not None
            )
        )
        for split in ("train", "test")
    }
    text_matches = [
        r["transcript_alignment"] for r in records if r["transcript_alignment"]
    ]
    report = {
        "artifact_type": "metadata_only",
        "raw_root": str(root),
        "counts": dict(counts),
        "wav_files_discovered": wav_count,
        "metadata_records": len(records),
        "dialog_videos": len(videos),
        "videos": videos,
        "split_counts": dict(split_counts),
        "label_counts": label_counts,
        "emotion4_counts": emotion4_counts,
        "alignment_duration_delta_sec_including_silence": summary(duration_deltas),
        "lexical_end_minus_wav_sec": summary(lexical_deltas),
        "word_counts": summary(word_counts),
        "transcript_alignment": {
            "compared_utterances": len(text_matches),
            "same_normalized_text_count": sum(
                m["same_normalized_text"] for m in text_matches
            ),
            "prefix_match_ratio": summary(
                [m["exact_prefix_match_ratio"] for m in text_matches]
            ),
        },
        "temporal_acoustic_features_extracted": False,
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }
    output.mkdir(parents=True, exist_ok=True)
    metadata_text = "".join(json.dumps(r) + "\n" for r in records)
    report["metadata_sha256"] = hashlib.sha256(metadata_text.encode()).hexdigest()
    (output / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if errors:
        raise RuntimeError(
            f"Audit failed with {len(errors)} errors; see {output / 'validation_report.json'}"
        )
    (output / "metadata.jsonl").write_text(metadata_text)
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "passed",
                    "metadata_records",
                    "dialog_videos",
                    "split_counts",
                    "emotion4_counts",
                )
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/IEMOCAP_full_release")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.root, args.output)
