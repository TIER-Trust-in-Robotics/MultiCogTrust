"""Cross-check utterance/video time using the audio muxed into each source AVI."""

from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.signal import correlate
import soundfile as sf
from src.iemocap_embedding_cache import sha256, write_json
from src.prepare_iemocap import DEFAULT_METADATA


def match_audio(reference, source, start_sec, sample_rate=16000, search_sec=0.5):
    """Find normalized correlation against each video audio channel near label time."""
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim > 1:
        reference = reference.mean(1)
    # Keep enough waveform for an unambiguous match, avoiding huge FFTs.
    reference = reference[: 4 * sample_rate]
    reference = reference - reference.mean()
    norm = np.linalg.norm(reference)
    if norm < 1e-6:
        return {"status": "silent_reference"}
    low = max(0, int((start_sec - search_sec) * sample_rate))
    high = min(
        len(source), int((start_sec + search_sec) * sample_rate) + len(reference)
    )
    best = None
    for channel in range(source.shape[1]):
        values = np.asarray(source[low:high, channel], dtype=np.float64)
        if len(values) < len(reference):
            continue
        dot = correlate(values, reference, mode="valid", method="fft")
        cumulative = np.r_[0.0, np.cumsum(values)]
        squares = np.r_[0.0, np.cumsum(values * values)]
        size = len(reference)
        sums = cumulative[size:] - cumulative[:-size]
        energy = squares[size:] - squares[:-size] - sums * sums / size
        scores = dot / (norm * np.sqrt(np.maximum(energy, 1e-20)))
        index = int(np.argmax(scores))
        result = {
            "status": "matched",
            "correlation": float(scores[index]),
            "channel": channel,
            "offset_sec": (low + index) / sample_rate - start_sec,
        }
        if best is None or result["correlation"] > best["correlation"]:
            best = result
    return best or {"status": "video_audio_too_short"}


def audit(metadata, output, limit=None):
    groups = defaultdict(list)
    for row in map(json.loads, Path(metadata).read_text().splitlines()):
        if row["session"] != 5 and row["emotion4_label"] is not None:
            groups[row["dialog_id"]].append(row)
    reports = []
    for index, (dialog, rows) in enumerate(sorted(groups.items())):
        if limit is not None and index >= limit:
            break
        video = rows[0]["video_path"]
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                video,
                "-map",
                "0:a:0",
                "-ar",
                "16000",
                "-ac",
                "2",
                "-f",
                "f32le",
                "pipe:1",
            ],
            check=True,
            capture_output=True,
        ).stdout
        source = np.frombuffer(raw, np.float32).reshape(-1, 2)
        matches = []
        for speaker in ("F", "M"):
            candidates = sorted(
                [
                    r
                    for r in rows
                    if r["speaker"] == speaker
                    and r["duration_sec"] >= 0.75
                    and r["dialog_end_sec"] <= len(source) / 16000
                ],
                key=lambda r: r["dialog_start_sec"],
            )
            for position in sorted({0, len(candidates) // 2, len(candidates) - 1}):
                if not candidates:
                    break
                row = candidates[position]
                audio, sr = sf.read(row["audio_path"], dtype="float32")
                if sr != 16000:
                    raise ValueError("Unexpected waveform rate")
                match = match_audio(audio, source, row["dialog_start_sec"])
                matches.append(
                    {
                        "utterance_id": row["utterance_id"],
                        "speaker": speaker,
                        "label_start_sec": row["dialog_start_sec"],
                        **match,
                    }
                )
        reports.append({"dialog_id": dialog, "video_path": video, "matches": matches})
        if (index + 1) % 10 == 0:
            print(
                f"Audio/video timing audited {index + 1}/{len(groups)} dialogs",
                flush=True,
            )
    matches = [m for r in reports for m in r["matches"]]
    trusted = [m for m in matches if m.get("correlation", 0) >= 0.8]
    offsets = np.array([m["offset_sec"] for m in trusted])
    report = {
        "scope": "development_sessions_1_to_4",
        "metadata_sha256": sha256(metadata),
        "dialog_count": len(reports),
        "sample_count": len(matches),
        "trusted_matches": len(trusted),
        "min_correlation": min((m.get("correlation", 0) for m in matches), default=0),
        "median_offset_sec": float(np.median(offsets)) if len(offsets) else None,
        "max_abs_offset_sec": float(np.max(np.abs(offsets))) if len(offsets) else None,
        "passed": bool(
            len(matches)
            and len(trusted) == len(matches)
            and np.max(np.abs(offsets)) < 0.05
        ),
        "dialogs": reports,
        "interpretation": "Waveform-to-muxed-audio timing; speaker panel mapping and visual lip timing need image inspection separately.",
    }
    write_json(output, report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "dialogs"}, indent=2),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/mvp/day4_alignment/audio_video_audit.json",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    audit(args.metadata, args.output, args.limit)
