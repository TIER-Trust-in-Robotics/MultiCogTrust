"""Render speaker crop contact sheets and word/audio/visual alignment timelines."""

from __future__ import annotations
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import soundfile as sf
from src.extract_iemocap_visual import decoded_video
from src.iemocap_embedding_cache import sha256, write_json
from src.prepare_iemocap import DEFAULT_METADATA
from src.visual_features import STATUS_NAMES


def render(feature_root, metadata, output, utterance_ids=None):
    source = {
        r["utterance_id"]: r
        for r in map(json.loads, Path(metadata).read_text().splitlines())
    }
    features = {
        r["utterance_id"]: r
        for r in map(
            json.loads, (feature_root / "metadata.jsonl").read_text().splitlines()
        )
    }
    if utterance_ids:
        chosen = [features[uid] for uid in utterance_ids]
    else:
        chosen = []
        for session in range(1, 5):
            for marker in ("F", "M"):
                for side in ("left", "right"):
                    candidates = [
                        r
                        for r in features.values()
                        if r["session"] == session
                        and r["dialog_id"][5] == marker
                        and r["side"] == side
                        and r["valid_frames"]
                    ]
                    if candidates:
                        chosen.append(candidates[0])
    output.mkdir(parents=True, exist_ok=True)
    requests = defaultdict(list)
    for row in chosen:
        with np.load(feature_root / row["feature_path"], allow_pickle=False) as data:
            valid = np.flatnonzero(data["valid"])
            missing = np.flatnonzero(data["decoded"] & ~data["valid"])
            for index in ([int(valid[len(valid) // 2])] if len(valid) else []) + (
                [int(missing[0])] if len(missing) else []
            ):
                requests[row["dialog_id"]].append(
                    {
                        "utterance_id": row["utterance_id"],
                        "side": row["side"],
                        "frame_index": int(data["frame_indices"][index]),
                        "timestamp_sec": float(data["timestamps_sec"][index]),
                        "status": STATUS_NAMES[int(data["status"][index])],
                        "box": data["face_boxes"][index].tolist(),
                    }
                )
    panels, inspection = [], []
    for dialog, requests_here in requests.items():
        wanted = defaultdict(list)
        for request in requests_here:
            wanted[request["frame_index"]].append(request)
        video = source[requests_here[0]["utterance_id"]]["video_path"]
        for index, seconds, frame in decoded_video(video):
            if index not in wanted:
                continue
            for request in wanted.pop(index):
                if abs(seconds - request["timestamp_sec"]) > 1e-6:
                    raise AssertionError(
                        "Contact sheet frame time differs from extracted feature"
                    )
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image)
                width, height = image.size
                left = 0 if request["side"] == "left" else width // 2
                draw.rectangle(
                    (left + 2, 2, left + width // 2 - 3, height - 3),
                    outline="yellow",
                    width=3,
                )
                if request["status"] in ("valid", "landmarks_unavailable"):
                    draw.rectangle(
                        request["box"],
                        outline="lime" if request["status"] == "valid" else "orange",
                        width=3,
                    )
                tile = Image.new("RGB", (width, height + 35), "white")
                tile.paste(image, (0, 35))
                ImageDraw.Draw(tile).text(
                    (8, 8),
                    f"{request['utterance_id']} {seconds:.3f}s {request['side']} {request['status']}",
                    fill="black",
                )
                panels.append(tile)
                inspection.append(request)
            if not wanted:
                break
    # Split contact sheets to keep text and boxes readable at native resolution.
    contact_paths = []
    for start in range(0, len(panels), 8):
        batch = panels[start : start + 8]
        canvas = Image.new("RGB", (1440, 515 * int(np.ceil(len(batch) / 2))), "white")
        for i, tile in enumerate(batch):
            canvas.paste(tile, ((i % 2) * 720, (i // 2) * 515))
        path = output / f"contacts_{start // 8 + 1}.jpg"
        canvas.save(path, quality=94)
        contact_paths.append(str(path))
    timeline_paths = []
    for row in chosen:
        uid = row["utterance_id"]
        original = source[uid]
        data = np.load(feature_root / row["feature_path"], allow_pickle=False)
        waveform, sr = sf.read(original["audio_path"], dtype="float32")
        duration = row["window_end_sec"] - row["dialog_start_sec"]
        waveform = waveform[: int(duration * sr)]
        step = 320
        rms = [
            float(np.sqrt(np.mean(waveform[i : i + step] ** 2)))
            for i in range(0, len(waveform), step)
        ]
        t = np.array(data["timestamps_sec"]) - row["dialog_start_sec"]
        valid = data["valid"]
        fig, axes = plt.subplots(
            4,
            1,
            figsize=(11, 7),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1.3, 0.65, 0.8]},
        )
        axes[0].plot(np.arange(len(rms)) * step / sr, rms, color="#34495e")
        axes[0].set_ylabel("Audio RMS")
        for index, name in [(25, "jaw open"), (44, "smile left"), (45, "smile right")]:
            values = data["blendshapes"][:, index].copy()
            values[~valid] = np.nan
            axes[1].plot(t, values, ".-", label=name)
        axes[1].set_ylabel("Blendshape\ncoefficient")
        axes[1].legend(loc="upper right", fontsize=8)
        available = data["image_valid"].astype(int) + data["blendshape_valid"].astype(
            int
        )
        axes[2].scatter(
            data["requested_sec"] - row["dialog_start_sec"],
            available,
            c=np.where(
                available == 2,
                "#16813e",
                np.where(available == 1, "#c78716", "#c43b36"),
            ),
            s=12,
        )
        axes[2].set_yticks([0, 1, 2], labels=["missing", "image", "both"])
        for i, word in enumerate(original["words"]):
            if word["start_sec"] >= duration:
                continue
            axes[3].plot(
                [word["start_sec"], min(duration, word["end_sec"])],
                [i % 2, i % 2],
                linewidth=3,
            )
            axes[3].text(
                word["start_sec"], i % 2 + 0.12, word["word"], fontsize=7, rotation=20
            )
        axes[3].set_ylim(-0.3, 2)
        axes[3].set_yticks([])
        axes[3].set_ylabel("Aligned words")
        axes[3].set_xlabel("Seconds from utterance start")
        axes[3].set_xlim(0, duration)
        fig.suptitle(
            f"{uid} | {row['side']} panel | coverage {row['coverage']:.0%}\n{original['text'][:115]}",
            fontsize=10,
        )
        fig.tight_layout()
        path = output / f"{uid}_timeline.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        timeline_paths.append(str(path))
    report = {
        "scope": "development sample inspection, not full-corpus manual verification",
        "samples": inspection,
        "contacts": contact_paths,
        "timelines": timeline_paths,
        "feature_metadata_sha256": sha256(feature_root / "metadata.jsonl"),
        "source_metadata_sha256": sha256(metadata),
        "review_status": "awaiting_visual_inspection",
    }
    write_json(output / "inspection_manifest.json", report)
    print(
        json.dumps(
            {
                "frames": len(inspection),
                "contacts": contact_paths,
                "timelines": timeline_paths,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data/processed/iemocap_visual_features_v2",
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/mvp/day4_alignment/visual_review",
    )
    parser.add_argument("--utterance-id", action="append")
    parser.add_argument(
        "--record-review",
        help="Record inspection notes for existing images; does not render or inspect them automatically",
    )
    parser.add_argument("--reviewer", default="user")
    args = parser.parse_args()
    if args.record_review:
        path = args.output / "inspection_manifest.json"
        report = json.loads(path.read_text())
        if report["feature_metadata_sha256"] != sha256(
            args.features / "metadata.jsonl"
        ):
            raise ValueError("Feature store changed since these images were rendered")
        report.update(
            review_status="inspected",
            reviewer=args.reviewer,
            review_notes=args.record_review,
            reviewed_artifact_sha256={
                name: sha256(name) for name in report["contacts"] + report["timelines"]
            },
        )
        write_json(path, report)
        print(f"Review recorded in {path}")
    else:
        render(args.features, args.metadata, args.output, args.utterance_id)
