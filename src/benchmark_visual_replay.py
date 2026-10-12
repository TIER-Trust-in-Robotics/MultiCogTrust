"""Paced, uncached visual replay including decode, detection and image encoding."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from src.extract_iemocap_visual import decoded_video
from src.iemocap_embedding_cache import sha256, write_json
from src.visual_features import (
    FaceFeatureExtractor,
    FaceImageEmbedder,
    DETECTOR_PATH,
    panel_crops,
)


def percentiles(values):
    return (
        {
            "count": len(values),
            "p50_ms": float(np.median(values)),
            "p95_ms": float(np.quantile(values, 0.95)),
            "max_ms": float(max(values)),
        }
        if values
        else {"count": 0}
    )


def benchmark(args):
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    embedder = FaceImageEmbedder(args.device)
    face = FaceFeatureExtractor(args.model)
    frames = decoded_video(args.video)
    rows = []
    decode_ms = 0.0
    deadline = None
    target = args.start_sec
    crops = None
    try:
        while target < args.start_sec + args.seconds:
            start = time.perf_counter()
            try:
                index, seconds, bgr = next(frames)
            except StopIteration:
                break
            decode_ms += 1000 * (time.perf_counter() - start)
            if crops is None:
                crops = panel_crops(bgr)
            if seconds < target:
                continue
            if deadline is None:
                wall_start = time.perf_counter()
                source_start = seconds
            deadline = wall_start + (seconds - source_start)
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            started = time.perf_counter()
            x0, y0, x1, y1 = crops[args.side]
            rgb = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            found = face.extract(rgb, round(seconds * 1000))
            if found.image_valid:
                embedder.encode([found.crop])
            finished = time.perf_counter()
            rows.append(
                {
                    "frame_index": index,
                    "source_timestamp_sec": seconds,
                    "valid_face": found.image_valid,
                    "valid_blendshapes": found.blendshape_valid,
                    "processing_ms": 1000 * (finished - started) + decode_ms,
                    "finish_lateness_ms": max(0, 1000 * (finished - deadline)),
                    "start_lateness_ms": max(0, 1000 * (started - deadline)),
                }
            )
            decode_ms = 0.0
            target += 1 / args.sample_hz
            if len(rows) % 100 == 0:
                print(f"Paced replay {len(rows)} frames", flush=True)
    finally:
        face.close()
        frames.close()
    measured = rows[args.warmup_frames :]
    timing = percentiles([r["processing_ms"] for r in measured])
    valid_timing = percentiles(
        [r["processing_ms"] for r in measured if r["valid_face"]]
    )
    late = percentiles([r["start_lateness_ms"] for r in measured])
    if not measured:
        raise ValueError("Not enough replay frames after warm-up")
    report = {
        "scope": "paced_source_video_visual_only_uncached",
        "context": args.context,
        "video": str(args.video),
        "video_sha256": sha256(args.video),
        "side": args.side,
        "sample_hz": args.sample_hz,
        "requested_seconds": args.seconds,
        "frames": len(rows),
        "warmup_frames": args.warmup_frames,
        "device": args.device,
        "processing": timing,
        "valid_face_processing": valid_timing,
        "start_lateness": late,
        "final_start_lateness_ms": rows[-1]["start_lateness_ms"],
        "final_finish_lateness_ms": rows[-1]["finish_lateness_ms"],
        "valid_face_rate": float(np.mean([r["valid_face"] for r in measured])),
        "target_p95_under_100_ms": timing["p95_ms"] < 100,
        "no_accumulated_backlog": rows[-1]["start_lateness_ms"] < 1000 / args.sample_hz,
        "hardware": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip(),
        "platform": platform.platform(),
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1024**2),
        "image_encoder": embedder.provenance,
        "landmarker_sha256": sha256(args.model),
        "face_detector_sha256": sha256(DETECTOR_PATH),
        "samples": rows,
        "limitations": [
            "No webcam or audio/ASR/fusion in this benchmark",
            "Replay is one development speaker track",
            "Throughput does not establish emotion accuracy",
        ],
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("samples", "image_encoder")},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        type=Path,
        default=ROOT
        / "data/IEMOCAP_full_release/Session1/dialog/avi/DivX/Ses01F_impro01.avi",
    )
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument(
        "--model", type=Path, default=ROOT / "src/weights/face_landmarker.task"
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--start-sec", type=float, default=6.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--sample-hz", type=float, default=10.0)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--context", default="standalone")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/mvp/day4_visual_latency.json",
    )
    benchmark(parser.parse_args())
