"""Extract aligned visual sequences for all four-class development utterances.

Default: Sessions 1-4, first six seconds per utterance, 5 Hz. Session 5 remains
unprocessed until final evaluation. Face failures retain zero rows plus masks.
"""

from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import closing
from concurrent.futures import ProcessPoolExecutor
import heapq
import json
import multiprocessing
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import mediapipe
import numpy as np
import torch
import torchvision

from src.iemocap_embedding_cache import sha256, write_json
from src.prepare_iemocap import DEFAULT_METADATA
from src.visual_features import (
    BLENDSHAPE_NAMES,
    DETECTOR_PATH,
    STATUS_NAMES,
    FaceFeatureExtractor,
    FaceImageEmbedder,
    panel_crops,
    speaker_side,
)

_WORKER_EMBEDDER = None
_WORKER_MODEL = None


def initialize_worker(device, model_path):
    global _WORKER_EMBEDDER, _WORKER_MODEL
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    _WORKER_EMBEDDER = FaceImageEmbedder(device)
    _WORKER_MODEL = model_path


def extract_worker(task):
    rows, output, contract = task
    return extract_dialog(rows, output, contract, _WORKER_EMBEDDER, _WORKER_MODEL)


def sample_times(start, end, rate, max_seconds):
    if rate <= 0 or max_seconds <= 0 or end <= start:
        raise ValueError("Invalid visual sampling interval")
    stop = min(end, start + max_seconds)
    return start + np.arange(int(np.ceil((stop - start) * rate - 1e-9))) / rate


def empty_features(times):
    size = len(times)
    return {
        "requested_sec": np.array(times, dtype=np.float64),
        "timestamps_sec": np.full(size, np.nan),
        "frame_indices": np.full(size, -1, np.int64),
        "blendshapes": np.zeros((size, 52), np.float32),
        "image_embeddings": np.zeros((size, 576), np.float32),
        "face_boxes": np.zeros((size, 4), np.float32),
        "valid": np.zeros(size, bool),
        "image_valid": np.zeros(size, bool),
        "blendshape_valid": np.zeros(size, bool),
        "detection_score": np.zeros(size, np.float32),
        "decoded": np.zeros(size, bool),
        "status": np.full(size, 4, np.uint8),
        "face_count": np.zeros(size, np.uint8),
    }


def decoded_video(path, timing_stats=None):
    """Use FFmpeg best-effort display times exposed by OpenCV.

    These DivX AVIs contain nonmonotonic raw PTS on B-frames. PyAV frame.pts
    therefore cannot be used as the presentation timeline. CAP_PROP_POS_MSEC
    was checked against ffprobe best_effort_timestamp_time on the source.
    """
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode video: {path}")
    previous, index = -1.0, 0
    missing_tail = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if seconds == 0 and previous > 0:
                # Some AVI drain frames have no PTS or DTS (ffprobe confirms
                # N/A). Preserve their absence; do not invent display times.
                missing_tail.append(index)
                index += 1
                continue
            if missing_tail:
                raise ValueError(f"Missing display time inside video: {path}")
            if not np.isfinite(seconds) or seconds <= previous:
                raise ValueError(f"Nonmonotonic display time in {path}: {seconds}")
            previous = seconds
            yield index, seconds, frame
            index += 1
    finally:
        if timing_stats is not None:
            timing_stats["untimed_tail_frame_indices"] = missing_tail
        capture.release()


def extract_dialog(rows, output, contract, embedder, model_path):
    dialog = rows[0]["dialog_id"]
    video = Path(rows[0]["video_path"])
    fingerprint = {"contract": contract, "video_sha256": sha256(video)}
    record_path = output / "dialogs" / f"{dialog}.json"
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError(
                f"Visual source/config changed for {dialog}; use a separate output"
            )
        for row in previous["utterances"]:
            if sha256(output / row["feature_path"]) != row["feature_sha256"]:
                raise ValueError(
                    f"Visual cache checksum mismatch: {row['utterance_id']}"
                )
        return previous
    buffers, requests = {}, []
    by_id = {row["utterance_id"]: row for row in rows}
    for row in rows:
        uid = row["utterance_id"]
        times = sample_times(
            row["dialog_start_sec"],
            row["dialog_end_sec"],
            contract["sample_hz"],
            contract["max_seconds"],
        )
        buffers[uid] = empty_features(times)
        for index, seconds in enumerate(times):
            heapq.heappush(requests, (seconds, uid, index))
    trackers, pending = {}, []
    stats = {
        "decode_sec": 0.0,
        "face_sec": 0.0,
        "embedding_sec": 0.0,
        "selected_frames": 0,
        "face_calls": 0,
    }
    started = time.perf_counter()

    def flush():
        if not pending:
            return
        before = time.perf_counter()
        embeddings = embedder.encode([p[0] for p in pending])
        stats["embedding_sec"] += time.perf_counter() - before
        for (_, refs), embedding in zip(pending, embeddings):
            for uid, index in refs:
                buffers[uid]["image_embeddings"][index] = embedding
        pending.clear()

    try:
        with closing(decoded_video(video, stats)) as frames:
            crops = None
            while requests:
                before = time.perf_counter()
                try:
                    frame_index, seconds, frame = next(frames)
                except StopIteration:
                    break
                stats["decode_sec"] += time.perf_counter() - before
                if crops is None:
                    crops = panel_crops(frame)
                if seconds + 1e-9 < requests[0][0]:
                    continue
                selected = defaultdict(list)
                while requests and requests[0][0] <= seconds + 1e-9:
                    target, uid, index = heapq.heappop(requests)
                    row = by_id[uid]
                    if seconds >= min(
                        row["dialog_end_sec"],
                        row["dialog_start_sec"] + contract["max_seconds"],
                    ):
                        continue
                    if seconds - target > 0.05:
                        continue  # Do not bridge video timestamp gaps.
                    selected[speaker_side(dialog, row["speaker"])].append((uid, index))
                if not selected:
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stats["selected_frames"] += 1
                for side, refs in selected.items():
                    if side not in trackers:
                        trackers[side] = FaceFeatureExtractor(model_path)
                    x0, y0, x1, y1 = crops[side]
                    face = trackers[side].extract(
                        rgb[y0:y1, x0:x1], int(round(seconds * 1000))
                    )
                    stats["face_sec"] += face.milliseconds / 1000
                    stats["face_calls"] += 1
                    for uid, index in refs:
                        values = buffers[uid]
                        values["timestamps_sec"][index] = seconds
                        values["frame_indices"][index] = frame_index
                        values["decoded"][index] = True
                        values["face_count"][index] = face.face_count
                        values["status"][index] = face.status
                        values["valid"][index] = face.status == 0
                        values["image_valid"][index] = face.image_valid
                        values["blendshape_valid"][index] = face.blendshape_valid
                        values["detection_score"][index] = face.detection_score
                        values["blendshapes"][index] = face.blendshapes
                        if face.image_valid:
                            values["face_boxes"][index] = face.box + np.array(
                                [x0, y0, x0, y0]
                            )
                    if face.image_valid:
                        pending.append((face.crop, refs))
                    if len(pending) >= 32:
                        flush()
            flush()
    finally:
        for tracker in trackers.values():
            tracker.close()
    stats["wall_sec"] = time.perf_counter() - started
    records = []
    for row in rows:
        uid = row["utterance_id"]
        features = buffers[uid]
        path = output / "features" / f"{uid}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **features)
        temporary.replace(path)
        records.append(
            {
                "utterance_id": uid,
                "dialog_id": dialog,
                "session": row["session"],
                "partition": "validation" if row["session"] == 4 else "train",
                "label": row["emotion4_label"],
                "speaker": row["speaker"],
                "side": speaker_side(dialog, row["speaker"]),
                "dialog_start_sec": row["dialog_start_sec"],
                "dialog_end_sec": row["dialog_end_sec"],
                "window_end_sec": min(
                    row["dialog_end_sec"],
                    row["dialog_start_sec"] + contract["max_seconds"],
                ),
                "video_span_valid": row["video_span_valid"],
                "frame_count": len(features["valid"]),
                "valid_frames": int(features["valid"].sum()),
                "image_valid_frames": int(features["image_valid"].sum()),
                "decoded_frames": int(features["decoded"].sum()),
                "coverage": float(features["valid"].mean()),
                "status_counts": {
                    STATUS_NAMES[int(k)]: int(v)
                    for k, v in zip(*np.unique(features["status"], return_counts=True))
                },
                "feature_path": str(path.relative_to(output)),
                "feature_sha256": sha256(path),
            }
        )
    report = {
        "dialog_id": dialog,
        "fingerprint": fingerprint,
        "crops": crops,
        "stats": stats,
        "utterances": records,
    }
    write_json(record_path, report)
    return report


def coverage_report(records):
    groups = defaultdict(list)
    for row in records:
        groups[f"{row['partition']}/{row['label']}"].append(row)

    def summarize(rows):
        frames = sum(r["frame_count"] for r in rows)
        valid = sum(r["valid_frames"] for r in rows)
        images = sum(r["image_valid_frames"] for r in rows)
        return {
            "utterances": len(rows),
            "requested_frames": frames,
            "valid_frames": valid,
            "frame_coverage": valid / max(1, frames),
            "image_valid_frames": images,
            "image_frame_coverage": images / max(1, frames),
            "utterances_with_any_image": sum(r["image_valid_frames"] > 0 for r in rows),
            "utterances_without_image": sum(r["image_valid_frames"] == 0 for r in rows),
            "utterances_with_any_face": sum(r["valid_frames"] > 0 for r in rows),
            "utterances_without_face": sum(r["valid_frames"] == 0 for r in rows),
            "incomplete_source_video_spans": sum(
                not r["video_span_valid"] for r in rows
            ),
        }

    return {
        "overall": summarize(records),
        "by_partition_label": {k: summarize(v) for k, v in sorted(groups.items())},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/iemocap_visual_features_v2",
    )
    parser.add_argument(
        "--model", type=Path, default=ROOT / "src/weights/face_landmarker.task"
    )
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=float, default=6.0)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--limit-dialogs", type=int)
    parser.add_argument("--dialog-id", action="append")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent dialog processes; use 2 on the target Mac",
    )
    args = parser.parse_args()
    if args.workers < 1 or (args.workers > 1 and args.device != "cpu"):
        parser.error(
            "workers must be positive; parallel extraction uses CPU image encoders"
        )
    audit = json.loads(args.metadata.with_name("validation_report.json").read_text())
    if not audit["passed"] or audit["metadata_sha256"] != sha256(args.metadata):
        raise ValueError("Metadata must match a passing audit")
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    embedder = FaceImageEmbedder(args.device)
    contract = {
        "schema_version": 2,
        "metadata_sha256": sha256(args.metadata),
        "sample_hz": args.sample_hz,
        "max_seconds": args.max_seconds,
        "sessions": [1, 2, 3, 4],
        "landmarker_sha256": sha256(args.model),
        "face_detector": {
            "name": "YuNet_2023mar",
            "sha256": sha256(DETECTOR_PATH),
            "score_threshold": 0.7,
            "nms_threshold": 0.3,
        },
        "landmarker_mode": "IMAGE_on_YuNet_crop",
        "mediapipe_version": mediapipe.__version__,
        "torchvision_version": torchvision.__version__,
        "image_encoder": embedder.provenance,
        "blendshape_names": BLENDSHAPE_NAMES,
        "status_names": STATUS_NAMES,
        "selection": "largest_face_min_area_0.0125_ambiguity_ratio_0.6",
        "face_crop_margin": 0.25,
        "panel_geometry": "first_frame_black_border_trim",
        "frame_timestamps": "ffmpeg_best_effort_via_opencv_pos_msec",
        "speaker_mapping": "dialog_marker_on_left_other_participant_on_right",
    }
    # Round-trip JSON keys to compare exactly when resuming.
    contract = json.loads(json.dumps(contract))
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != contract:
        raise ValueError("Visual cache configuration changed; use another --output")
    write_json(config_path, contract)
    groups = defaultdict(list)
    all_rows = [json.loads(line) for line in args.metadata.read_text().splitlines()]
    for row in all_rows:
        if row["session"] != 5 and row["emotion4_label"] is not None:
            groups[row["dialog_id"]].append(row)
    selected = sorted(groups)[: args.limit_dialogs]
    if args.dialog_id:
        missing = set(args.dialog_id) - groups.keys()
        if missing:
            raise ValueError(f"Unknown development dialogs: {sorted(missing)}")
        selected = [dialog for dialog in selected if dialog in args.dialog_id]
    records, dialogs = [], []

    def collect(i, dialog, report):
        records.extend(report["utterances"])
        dialogs.append({"dialog_id": dialog, **report["stats"]})
        info = coverage_report(records)["overall"]
        print(
            f"Visual {i + 1}/{len(selected)} {dialog}: {info['utterances']} utterances, {info['frame_coverage']:.1%} blendshape / {info['image_frame_coverage']:.1%} image coverage",
            flush=True,
        )

    if args.workers == 1:
        for i, dialog in enumerate(selected):
            collect(
                i,
                dialog,
                extract_dialog(
                    groups[dialog], args.output, contract, embedder, args.model
                ),
            )
    else:
        tasks = [(groups[dialog], args.output, contract) for dialog in selected]
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initialize_worker,
            initargs=(args.device, args.model),
        ) as executor:
            for i, report in enumerate(executor.map(extract_worker, tasks)):
                collect(i, selected[i], report)
    metadata_path = args.output / "metadata.jsonl"
    metadata_path.write_text("".join(json.dumps(r) + "\n" for r in records))
    report = {
        "complete_development_set": len(selected) == len(groups),
        "session_5_processed": False,
        "metadata_sha256": sha256(metadata_path),
        "source_metadata_sha256": sha256(args.metadata),
        "config_sha256": sha256(config_path),
        "dialog_count": len(dialogs),
        **coverage_report(records),
        "dialogs": dialogs,
        "unavailable_au_comparison": "pending: no local OpenFace executable; blendshapes are not AUs",
    }
    write_json(args.output / "coverage_report.json", report)
    print(
        json.dumps({k: v for k, v in report.items() if k != "dialogs"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
