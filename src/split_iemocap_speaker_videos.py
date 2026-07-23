"""Split IEMOCAP dialog videos into speaker-specific crops.

The IEMOCAP dialog videos under ``Session*/dialog/avi/DivX`` are side-by-side
composites. In the checked samples, the speaker encoded after the session id
(``Ses01F_*`` or ``Ses01M_*``) appears on the left panel, and the other speaker
appears on the right panel.

Examples:
    uv run python src/split_iemocap_speaker_videos.py --dialog-id Ses01F_impro01
    uv run python src/split_iemocap_speaker_videos.py --limit 5 --dry-run
    uv run python src/split_iemocap_speaker_videos.py --crop-mode half

Outputs are video-only by default because the dialog audio is not speaker
separated. Use ``--keep-audio`` only if you explicitly want the original mixed
dialog audio copied into each crop.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IEMOCAP_ROOT = REPO_ROOT / "data" / "IEMOCAP_full_release"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "processed" / "iemocap_speaker_videos"
DIALOG_RE = re.compile(r"^Ses(?P<session>\d{2})(?P<left_speaker>[FM])_.+")


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: str | None
    duration: float | None
    nb_frames: int | None


@dataclass(frozen=True)
class Crop:
    x: int
    y: int
    width: int
    height: int

    def ffmpeg_expr(self) -> str:
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"

    def to_json(self) -> dict[str, int]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split IEMOCAP dialog AVI files into female and male speaker crops."
    )
    parser.add_argument(
        "--iemocap-root",
        type=Path,
        default=DEFAULT_IEMOCAP_ROOT,
        help="Path to data/IEMOCAP_full_release.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory for cropped speaker videos and manifest.jsonl.",
    )
    parser.add_argument(
        "--dialog-id",
        action="append",
        default=[],
        help="Exact dialog id to process, e.g. Ses01F_impro01. Repeatable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many dialog videos after filtering.",
    )
    parser.add_argument(
        "--speaker",
        choices=["both", "F", "M"],
        default="both",
        help="Which speaker crop to write.",
    )
    parser.add_argument(
        "--crop-mode",
        choices=["auto-panel", "half"],
        default="auto-panel",
        help=(
            "auto-panel trims black borders using sampled frames; half keeps the "
            "full left/right half of the 720x480 composite."
        ),
    )
    parser.add_argument(
        "--auto-threshold",
        type=int,
        default=12,
        help="Pixel threshold used by --crop-mode auto-panel to find non-black panels.",
    )
    parser.add_argument(
        "--auto-padding",
        type=int,
        default=4,
        help="Padding in pixels added around auto-detected panels.",
    )
    parser.add_argument(
        "--output-ext",
        default=".mp4",
        help="Output video extension. Defaults to .mp4.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=18,
        help="x264 CRF for cropped outputs; lower is higher quality.",
    )
    parser.add_argument(
        "--preset",
        default="veryfast",
        help="x264 preset for cropped outputs.",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="Include original mixed dialog audio in each cropped speaker video.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cropped videos and manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ffmpeg commands and write no files.",
    )
    return parser.parse_args()


def discover_dialog_videos(iemocap_root: Path) -> list[Path]:
    return sorted(
        path
        for path in iemocap_root.glob("Session*/dialog/avi/DivX/*.avi")
        if path.is_file() and not path.name.startswith(".")
    )


def parse_dialog_id(video_path: Path) -> tuple[int, str, str]:
    dialog_id = video_path.stem
    match = DIALOG_RE.match(dialog_id)
    if not match:
        raise ValueError(f"Cannot parse IEMOCAP dialog id from {video_path}")
    session = int(match.group("session"))
    left_speaker = match.group("left_speaker")
    return session, left_speaker, dialog_id


def other_speaker(speaker: str) -> str:
    return "M" if speaker == "F" else "F"


def speaker_lab_path(iemocap_root: Path, session: int, speaker: str, dialog_id: str) -> Path:
    return (
        iemocap_root
        / f"Session{session}"
        / "dialog"
        / "lab"
        / f"Ses{session:02d}_{speaker}"
        / f"{dialog_id}.lab"
    )


def run_json_command(command: list[str]) -> dict:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def probe_video(video_path: Path) -> VideoInfo:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe was not found on PATH")

    data = run_json_command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(video_path),
        ]
    )
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in {video_path}")

    stream = streams[0]
    return VideoInfo(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=stream.get("r_frame_rate"),
        duration=float(stream["duration"]) if stream.get("duration") else None,
        nb_frames=int(stream["nb_frames"]) if stream.get("nb_frames") else None,
    )


def make_even_crop(x: int, y: int, width: int, height: int, frame_width: int, frame_height: int) -> Crop:
    x = max(0, min(x, frame_width - 2))
    y = max(0, min(y, frame_height - 2))
    if x % 2:
        x -= 1
    if y % 2:
        y -= 1

    right = max(x + 2, min(x + width, frame_width))
    bottom = max(y + 2, min(y + height, frame_height))
    width = right - x
    height = bottom - y
    if width % 2:
        width -= 1
    if height % 2:
        height -= 1
    if width <= 0 or height <= 0:
        raise ValueError("Computed an empty crop")
    return Crop(x=x, y=y, width=width, height=height)


def half_crops(video_info: VideoInfo) -> dict[str, Crop]:
    half_width = video_info.width // 2
    left = make_even_crop(0, 0, half_width, video_info.height, video_info.width, video_info.height)
    right = make_even_crop(
        half_width,
        0,
        video_info.width - half_width,
        video_info.height,
        video_info.width,
        video_info.height,
    )
    return {"left": left, "right": right}


def sample_seconds(duration: float | None) -> list[float]:
    if duration is None or duration <= 0:
        return [1.0, 5.0, 10.0]
    candidates = [1.0, 5.0, duration * 0.25, duration * 0.50, max(0.0, duration - 1.0)]
    unique = sorted({round(sec, 3) for sec in candidates if 0.0 <= sec < duration})
    return unique or [0.0]


def detect_auto_panel_crops(
    video_path: Path,
    video_info: VideoInfo,
    threshold: int,
    padding: int,
) -> dict[str, Crop]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required for --crop-mode auto-panel") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    half_width = video_info.width // 2
    boxes: dict[str, list[int] | None] = {"left": None, "right": None}

    for sec in sample_seconds(video_info.duration):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for side, x0, x1 in (
            ("left", 0, half_width),
            ("right", half_width, video_info.width),
        ):
            mask = gray[:, x0:x1] > threshold
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            current = [
                int(xs.min()) + x0,
                int(ys.min()),
                int(xs.max()) + x0 + 1,
                int(ys.max()) + 1,
            ]
            if boxes[side] is None:
                boxes[side] = current
            else:
                previous = boxes[side]
                assert previous is not None
                boxes[side] = [
                    min(previous[0], current[0]),
                    min(previous[1], current[1]),
                    max(previous[2], current[2]),
                    max(previous[3], current[3]),
                ]

    cap.release()

    fallback = half_crops(video_info)
    crops: dict[str, Crop] = {}
    for side, box in boxes.items():
        if box is None:
            crops[side] = fallback[side]
            continue
        x0, y0, x1, y1 = box
        crops[side] = make_even_crop(
            x0 - padding,
            y0 - padding,
            (x1 - x0) + 2 * padding,
            (y1 - y0) + 2 * padding,
            video_info.width,
            video_info.height,
        )
    return crops


def build_ffmpeg_command(
    input_path: Path,
    output_path: Path,
    crop: Crop,
    crf: int,
    preset: str,
    keep_audio: bool,
    overwrite: bool,
) -> list[str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found on PATH")

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-vf",
        crop.ffmpeg_expr(),
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
    ]
    if keep_audio:
        command.extend(["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"])
    else:
        command.append("-an")
    command.append(str(output_path))
    return command


def selected_speaker_sides(left_speaker: str, speaker_filter: str) -> Iterable[tuple[str, str]]:
    right_speaker = other_speaker(left_speaker)
    side_speakers = (("left", left_speaker), ("right", right_speaker))
    for side, speaker in side_speakers:
        if speaker_filter == "both" or speaker == speaker_filter:
            yield side, speaker


def output_path_for(output_root: Path, session: int, dialog_id: str, speaker: str, extension: str) -> Path:
    extension = extension if extension.startswith(".") else f".{extension}"
    return output_root / f"Session{session}" / dialog_id / f"{dialog_id}_{speaker}{extension}"


def process_video(
    video_path: Path,
    args: argparse.Namespace,
) -> list[dict]:
    session, left_speaker, dialog_id = parse_dialog_id(video_path)
    video_info = probe_video(video_path)
    if args.crop_mode == "half":
        crops = half_crops(video_info)
    else:
        crops = detect_auto_panel_crops(
            video_path,
            video_info,
            threshold=args.auto_threshold,
            padding=args.auto_padding,
        )

    records = []
    for side, speaker in selected_speaker_sides(left_speaker, args.speaker):
        output_path = output_path_for(args.output_root, session, dialog_id, speaker, args.output_ext)
        lab_path = speaker_lab_path(args.iemocap_root, session, speaker, dialog_id)
        crop = crops[side]
        command = build_ffmpeg_command(
            input_path=video_path,
            output_path=output_path,
            crop=crop,
            crf=args.crf,
            preset=args.preset,
            keep_audio=args.keep_audio,
            overwrite=args.overwrite,
        )

        status = "dry_run"
        if args.dry_run:
            print(" ".join(command))
        elif output_path.exists() and not args.overwrite:
            status = "skipped_exists"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, check=True)
            status = "written"

        records.append(
            {
                "status": status,
                "dialog_id": dialog_id,
                "session": session,
                "marker_speaker": left_speaker,
                "speaker": speaker,
                "side": side,
                "input_path": str(video_path),
                "output_path": str(output_path),
                "lab_path": str(lab_path),
                "lab_exists": lab_path.exists(),
                "crop": crop.to_json(),
                "source_video": {
                    "width": video_info.width,
                    "height": video_info.height,
                    "fps": video_info.fps,
                    "duration": video_info.duration,
                    "nb_frames": video_info.nb_frames,
                },
            }
        )
    return records


def main() -> None:
    args = parse_args()
    args.iemocap_root = args.iemocap_root.resolve()
    args.output_root = args.output_root.resolve()

    dialog_filter = set(args.dialog_id)
    videos = discover_dialog_videos(args.iemocap_root)
    if dialog_filter:
        videos = [path for path in videos if path.stem in dialog_filter]
        missing = sorted(dialog_filter - {path.stem for path in videos})
        if missing:
            raise FileNotFoundError(f"Requested dialog ids were not found: {', '.join(missing)}")
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        raise FileNotFoundError(f"No IEMOCAP dialog videos found under {args.iemocap_root}")

    manifest_records = []
    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] {video_path.name}")
        manifest_records.extend(process_video(video_path, args))

    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output_root / "manifest.jsonl"
        mode = "w" if args.overwrite else "a"
        with manifest_path.open(mode, encoding="utf-8") as handle:
            for record in manifest_records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"Wrote manifest: {manifest_path}")

    counts: dict[str, int] = {}
    for record in manifest_records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    print("Summary:", ", ".join(f"{status}={count}" for status, count in sorted(counts.items())))


if __name__ == "__main__":
    main()
