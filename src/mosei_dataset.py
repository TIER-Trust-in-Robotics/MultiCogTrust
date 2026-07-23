"""CMU-MOSEI dataset preparation for the cross-modal (audio+text) model.

Works from the MMSA raw distribution (mirrored at
``huggingface.co/datasets/tamb2203579/CMU-MOSEI``):

    data/MOSEI/label.csv   video_id, clip_id, text, label (sentiment -3..3),
                           annotation, mode (train/valid/test standard folds)
    data/MOSEI/Raw/        {video_id}/{clip_id}.mp4 segment clips

Stages (each idempotent, resumable):

    uv run python src/mosei_dataset.py --extract-audio   # mp4 -> 16k mono wav
    uv run python src/mosei_dataset.py --write-metadata  # -> metadata.jsonl

Outputs ``data/MOSEI/wav/{video_id}/{clip_id}.wav`` and
``data/mosei_aligned_features/metadata.jsonl`` with records shaped like the
IEMOCAP aligned metadata (text + audio_path + label + split) so the training
scripts share one loading pattern.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iemocap_aligned_dataset import resolve_data_path  # noqa: E402

MOSEI_ROOT = ROOT / "data" / "MOSEI"
RAW_DIR = MOSEI_ROOT / "Raw"
WAV_DIR = MOSEI_ROOT / "wav"
LABEL_CSV = MOSEI_ROOT / "label.csv"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "mosei_aligned_features"
SAMPLE_RATE = 16000
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv")


@dataclass(frozen=True)
class MoseiClip:
    video_id: str
    clip_id: str
    text: str
    sentiment: float
    annotation: str
    split: str


def read_label_csv(label_csv: Path = LABEL_CSV) -> list[MoseiClip]:
    clips: list[MoseiClip] = []
    with open(label_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            clips.append(
                MoseiClip(
                    video_id=row["video_id"],
                    clip_id=row["clip_id"],
                    text=row["text"].strip(),
                    sentiment=float(row["label"]),
                    annotation=row["annotation"],
                    split=row["mode"],
                )
            )
    return clips


def find_clip_video(clip: MoseiClip, raw_dir: Path = RAW_DIR) -> Path | None:
    for ext in VIDEO_EXTENSIONS:
        candidate = raw_dir / clip.video_id / f"{clip.clip_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def extract_wav(video_path: Path, wav_path: Path) -> tuple[Path, str | None]:
    """ffmpeg mp4 -> 16 kHz mono wav; returns (wav_path, error or None)."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(video_path),
            "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            str(wav_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not wav_path.exists():
        return wav_path, result.stderr.strip() or "ffmpeg failed"
    return wav_path, None


def extract_audio(clips: list[MoseiClip], workers: int) -> None:
    jobs: list[tuple[Path, Path]] = []
    missing_video = 0
    for clip in clips:
        wav_path = WAV_DIR / clip.video_id / f"{clip.clip_id}.wav"
        if wav_path.exists():
            continue
        video_path = find_clip_video(clip)
        if video_path is None:
            missing_video += 1
            continue
        jobs.append((video_path, wav_path))

    print(f"{len(jobs)} clips to extract ({missing_video} missing videos)")
    failures: list[str] = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(extract_wav, video, wav): (video, wav)
            for video, wav in jobs
        }
        for future in as_completed(futures):
            wav_path, error = future.result()
            done += 1
            if error:
                failures.append(f"{wav_path}: {error}")
            if done % 500 == 0 or done == len(jobs):
                print(f"  extracted {done}/{len(jobs)} ({len(failures)} failures)")
    if failures:
        print(f"{len(failures)} extraction failures, first few:")
        for line in failures[:5]:
            print(f"  {line}")


def write_metadata(clips: list[MoseiClip], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.jsonl"
    written = 0
    missing = 0
    with open(metadata_path, "w") as handle:
        for clip in clips:
            wav_path = WAV_DIR / clip.video_id / f"{clip.clip_id}.wav"
            if not wav_path.exists():
                missing += 1
                continue
            handle.write(
                json.dumps(
                    {
                        "video_id": clip.video_id,
                        "clip_id": clip.clip_id,
                        "text": clip.text,
                        "sentiment": clip.sentiment,
                        "annotation": clip.annotation,
                        "split": clip.split,
                        "audio_path": str(wav_path),
                    }
                )
                + "\n"
            )
            written += 1
    print(f"Wrote {written} records to {metadata_path} ({missing} missing wavs)")


@dataclass
class MoseiSample:
    text: str
    audio_path: str
    sentiment: float
    split: str


def load_mosei_samples(
    metadata_path: str | Path = DEFAULT_OUTPUT_DIR / "metadata.jsonl",
    data_root: Path = ROOT / "data",
) -> list[MoseiSample]:
    samples: list[MoseiSample] = []
    missing = 0
    with open(metadata_path) as handle:
        for line in handle:
            record = json.loads(line)
            local_path = resolve_data_path(record["audio_path"], data_root)
            if not local_path.exists():
                missing += 1
                continue
            samples.append(
                MoseiSample(
                    text=record["text"],
                    audio_path=str(local_path),
                    sentiment=float(record["sentiment"]),
                    split=record["split"],
                )
            )
    if not samples:
        raise FileNotFoundError(
            f"No audio from {metadata_path} resolved under {data_root} "
            f"({missing} paths missing); run src/mosei_dataset.py first"
        )
    if missing:
        print(f"Warning: skipped {missing} clips with missing audio")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-audio", action="store_true")
    parser.add_argument("--write-metadata", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not LABEL_CSV.exists():
        raise FileNotFoundError(
            f"{LABEL_CSV} not found; download label.csv and Raw.zip from "
            "huggingface.co/datasets/tamb2203579/CMU-MOSEI into data/MOSEI/"
        )
    clips = read_label_csv()
    splits = {}
    for clip in clips:
        splits[clip.split] = splits.get(clip.split, 0) + 1
    print(f"label.csv: {len(clips)} clips, splits={splits}")

    if args.extract_audio:
        extract_audio(clips, workers=args.workers)
    if args.write_metadata:
        write_metadata(clips, args.output_dir)
    if not args.extract_audio and not args.write_metadata:
        print("Nothing to do; pass --extract-audio and/or --write-metadata")


if __name__ == "__main__":
    main()
