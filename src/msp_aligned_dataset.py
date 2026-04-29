from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iemocap_aligned_dataset import (  # noqa: E402
    SAMPLE_RATE,
    SlidingOpenSmileExtractor,
    WordSpan,
    aggregate_word_features,
    get_wav_duration,
    normalize_text_tokens,
    read_audio_mono,
    summarize_numeric,
)
from to_ds import MSP_LABEL2IDX, MSP_LABELS, MSP_LABEL_NAMES  # noqa: E402

DATA_ROOT = ROOT / "data"
MSP_ROOT = DATA_ROOT / "MSP-PODCAST"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "msp_aligned_features"
TEXTGRID_EMPTY_WORDS = {"", "<eps>", "<sil>", "sil", "sp", "spn"}


def parse_textgrid_words(path: Path) -> list[WordSpan]:
    words: list[WordSpan] = []
    in_words_tier = False
    current_start: float | None = None
    current_end: float | None = None

    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if line.startswith("name = "):
                tier_name = line.split("=", 1)[1].strip().strip('"')
                in_words_tier = tier_name == "words"
                continue
            if not in_words_tier:
                continue
            if line.startswith("item [") and words:
                break
            if line.startswith("xmin = "):
                current_start = float(line.split("=", 1)[1].strip())
            elif line.startswith("xmax = "):
                current_end = float(line.split("=", 1)[1].strip())
            elif line.startswith("text = "):
                raw_word = line.split("=", 1)[1].strip().strip('"')
                word = raw_word.strip().lower()
                if (
                    current_start is not None
                    and current_end is not None
                    and word not in TEXTGRID_EMPTY_WORDS
                ):
                    words.append(
                        WordSpan(
                            word=word,
                            raw_word=raw_word,
                            start_sec=current_start,
                            end_sec=current_end,
                        )
                    )
                current_start = None
                current_end = None

    return words


def transcript_match_stats(text: str, words: list[WordSpan]) -> dict[str, Any]:
    transcript_tokens = normalize_text_tokens(text)
    aligned_tokens = [word.word for word in words]
    compared = min(len(transcript_tokens), len(aligned_tokens))
    exact_prefix_matches = sum(
        1 for idx in range(compared) if transcript_tokens[idx] == aligned_tokens[idx]
    )
    return {
        "transcript_token_count": len(transcript_tokens),
        "aligned_word_count": len(aligned_tokens),
        "exact_prefix_matches": exact_prefix_matches,
        "exact_prefix_match_ratio": exact_prefix_matches / compared
        if compared
        else 0.0,
        "same_normalized_text": transcript_tokens == aligned_tokens,
    }


def validate_words(
    words: list[WordSpan],
    duration_sec: float,
    tolerance_sec: float,
) -> list[str]:
    errors: list[str] = []
    previous_end = 0.0
    for idx, word in enumerate(words):
        if word.start_sec < 0:
            errors.append(f"word {idx} starts before zero")
        if word.end_sec <= word.start_sec:
            errors.append(f"word {idx} has non-positive duration")
        if word.end_sec > duration_sec + tolerance_sec:
            errors.append(
                f"word {idx} ends at {word.end_sec:.3f}s past duration {duration_sec:.3f}s"
            )
        if word.start_sec < previous_end - 1e-6:
            errors.append(f"word {idx} is nonmonotonic")
        previous_end = word.end_sec
    return errors


def iter_msp_utterances(msp_root: Path = MSP_ROOT):
    labels_path = msp_root / "Labels" / "labels_consensus.csv"
    transcripts_dir = msp_root / "Transcripts"
    audios_dir = msp_root / "Audios"
    align_dir = msp_root / "ForceAligned"

    with labels_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_code = row["EmoClass"]
            split_set = row["Split_Set"]
            if label_code not in MSP_LABEL2IDX or split_set not in ("Train", "Test1"):
                continue

            filename = row["FileName"]
            utterance_id = Path(filename).stem
            transcript_path = transcripts_dir / f"{utterance_id}.txt"
            audio_path = audios_dir / filename
            textgrid_path = align_dir / f"{utterance_id}.TextGrid"

            if not transcript_path.exists() or not audio_path.exists():
                continue

            text = transcript_path.read_text().strip()
            if not text:
                continue

            yield {
                "utterance_id": utterance_id,
                "split": "train" if split_set == "Train" else "test",
                "text": text,
                "label_code": label_code,
                "label": MSP_LABEL2IDX[label_code],
                "vad": [
                    float(row["EmoVal"]),
                    float(row["EmoAct"]),
                    float(row["EmoDom"]),
                ],
                "activation": float(row["EmoAct"]),
                "valence": float(row["EmoVal"]),
                "dominance": float(row["EmoDom"]),
                "speaker_id": row["SpkrID"],
                "gender": row["Gender"],
                "audio_path": str(audio_path),
                "textgrid_path": str(textgrid_path),
            }


def count_msp_candidates(msp_root: Path = MSP_ROOT) -> int:
    labels_path = msp_root / "Labels" / "labels_consensus.csv"
    total = 0
    with labels_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label_code = row["EmoClass"]
            split_set = row["Split_Set"]
            if label_code not in MSP_LABEL2IDX or split_set not in ("Train", "Test1"):
                continue
            total += 1
    return total


def flush_shard(
    output_dir: Path,
    shard_idx: int,
    audio_arrays: dict[str, np.ndarray],
    word_arrays: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    audio_path = output_dir / f"audio_frames_{shard_idx:05d}.npz"
    word_path = output_dir / f"word_features_{shard_idx:05d}.npz"
    np.savez_compressed(audio_path, **audio_arrays)
    np.savez_compressed(word_path, **word_arrays)
    audio_arrays.clear()
    word_arrays.clear()
    return audio_path, word_path


def build_msp_aligned_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    msp_root: Path = MSP_ROOT,
    sample_rate: int = SAMPLE_RATE,
    window_sec: float = 1.0,
    hop_sec: float = 0.5,
    limit: int | None = None,
    tolerance_sec: float = 0.25,
    progress_every: int = 1000,
    shard_size: int = 5000,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = SlidingOpenSmileExtractor(
        sample_rate=sample_rate,
        window_sec=window_sec,
        hop_sec=hop_sec,
    )
    total_to_process = count_msp_candidates(msp_root)
    started_at = time.monotonic()

    metadata_path = output_dir / "metadata.jsonl"
    audio_arrays: dict[str, np.ndarray] = {}
    word_arrays: dict[str, np.ndarray] = {}
    shard_manifest: list[dict[str, Any]] = []
    shard_idx = 0
    shard_records = 0

    report: dict[str, Any] = {
        "utterances_discovered": 0,
        "included_utterances": 0,
        "missing_textgrid": 0,
        "invalid_word_spans": 0,
        "feature_extraction_failures": 0,
        "words_without_audio_overlap": 0,
        "duration_mismatch_sec": [],
        "frame_counts": [],
        "word_counts": [],
        "transcript_alignment": [],
        "label_counts": {"train": Counter(), "test": Counter()},
        "split_counts": Counter(),
        "errors": [],
        "config": {
            "sample_rate": sample_rate,
            "window_sec": window_sec,
            "hop_sec": hop_sec,
            "tolerance_sec": tolerance_sec,
            "progress_every": progress_every,
            "shard_size": shard_size,
        },
    }

    def print_progress(processed: int, *, force: bool = False) -> None:
        if progress_every <= 0:
            return
        if not force and processed % progress_every != 0:
            return
        target = f" included_target={limit}" if limit is not None else ""
        elapsed = time.monotonic() - started_at
        print(
            "msp-aligned "
            f"processed={processed}/{total_to_process} "
            f"included={report['included_utterances']} "
            f"missing_textgrid={report['missing_textgrid']} "
            f"invalid_spans={report['invalid_word_spans']} "
            f"feature_failures={report['feature_extraction_failures']} "
            f"elapsed={elapsed:.1f}s"
            f"{target}",
            file=sys.stderr,
            flush=True,
        )

    print_progress(0, force=True)
    with metadata_path.open("w") as metadata_file:
        for base_record in iter_msp_utterances(msp_root):
            if limit is not None and report["included_utterances"] >= limit:
                break
            report["utterances_discovered"] += 1
            utterance_id = base_record["utterance_id"]
            textgrid_path = Path(base_record["textgrid_path"])
            if not textgrid_path.exists():
                report["missing_textgrid"] += 1
                print_progress(report["utterances_discovered"])
                continue

            audio_path = Path(base_record["audio_path"])
            duration_sec = get_wav_duration(audio_path)
            words = parse_textgrid_words(textgrid_path)
            word_errors = validate_words(words, duration_sec, tolerance_sec)
            if word_errors:
                report["invalid_word_spans"] += 1
                report["errors"].append(
                    {"utterance_id": utterance_id, "errors": word_errors[:5]}
                )
                print_progress(report["utterances_discovered"])
                continue

            try:
                audio, sr = read_audio_mono(audio_path, sample_rate=sample_rate)
                frames = extractor.extract(audio)
            except Exception as exc:  # pragma: no cover - keeps long batch jobs alive
                report["feature_extraction_failures"] += 1
                report["errors"].append(
                    {"utterance_id": utterance_id, "errors": [repr(exc)]}
                )
                print_progress(report["utterances_discovered"])
                continue

            word_features, word_mask = aggregate_word_features(words, frames)
            no_overlap = int((word_mask == 0).sum())
            report["words_without_audio_overlap"] += no_overlap

            final_alignment_end = words[-1].end_sec if words else 0.0
            mismatch = final_alignment_end - duration_sec
            match_stats = transcript_match_stats(base_record["text"], words)

            record = {
                **{k: v for k, v in base_record.items() if k != "textgrid_path"},
                "duration_sec": duration_sec,
                "sample_rate": sr,
                "words": [
                    {
                        "word": word.word,
                        "raw_word": word.raw_word,
                        "start_sec": word.start_sec,
                        "end_sec": word.end_sec,
                    }
                    for word in words
                ],
                "audio_frame_count": int(frames.features.shape[0]),
                "word_count": len(words),
                "word_audio_feature_mask": word_mask.astype(int).tolist(),
                "transcript_alignment": match_stats,
                "feature_shard": shard_idx,
            }
            metadata_file.write(json.dumps(record) + "\n")

            audio_arrays[utterance_id] = frames.features
            audio_arrays[f"{utterance_id}__start_sec"] = frames.start_sec
            audio_arrays[f"{utterance_id}__end_sec"] = frames.end_sec
            word_arrays[utterance_id] = word_features
            word_arrays[f"{utterance_id}__mask"] = word_mask
            shard_records += 1

            split = base_record["split"]
            report["included_utterances"] += 1
            report["split_counts"][split] += 1
            report["label_counts"][split][base_record["label_code"]] += 1
            report["duration_mismatch_sec"].append(mismatch)
            report["frame_counts"].append(int(frames.features.shape[0]))
            report["word_counts"].append(len(words))
            report["transcript_alignment"].append(match_stats)
            print_progress(report["utterances_discovered"])

            if shard_records >= shard_size:
                audio_path, word_path = flush_shard(
                    output_dir, shard_idx, audio_arrays, word_arrays
                )
                shard_manifest.append(
                    {
                        "shard": shard_idx,
                        "records": shard_records,
                        "audio_frames": audio_path.name,
                        "word_features": word_path.name,
                    }
                )
                shard_idx += 1
                shard_records = 0

    if shard_records > 0:
        audio_path, word_path = flush_shard(
            output_dir, shard_idx, audio_arrays, word_arrays
        )
        shard_manifest.append(
            {
                "shard": shard_idx,
                "records": shard_records,
                "audio_frames": audio_path.name,
                "word_features": word_path.name,
            }
        )

    feature_names = extractor.feature_names or []
    if report["included_utterances"] and len(feature_names) != 88:
        report["errors"].append(
            {
                "utterance_id": None,
                "errors": [f"expected 88 OpenSMILE features, got {len(feature_names)}"],
            }
        )

    (output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (output_dir / "shards.json").write_text(json.dumps(shard_manifest, indent=2))
    serializable_report = summarize_report(report, shard_manifest)
    print_progress(report["utterances_discovered"], force=True)
    (output_dir / "alignment_report.json").write_text(
        json.dumps(serializable_report, indent=2)
    )
    return serializable_report


def summarize_report(
    report: dict[str, Any],
    shard_manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    transcript_stats = report["transcript_alignment"]
    exact_matches = sum(1 for item in transcript_stats if item["same_normalized_text"])
    prefix_ratios = [item["exact_prefix_match_ratio"] for item in transcript_stats]
    return {
        "utterances_discovered": report["utterances_discovered"],
        "included_utterances": report["included_utterances"],
        "missing_textgrid": report["missing_textgrid"],
        "invalid_word_spans": report["invalid_word_spans"],
        "feature_extraction_failures": report["feature_extraction_failures"],
        "words_without_audio_overlap": report["words_without_audio_overlap"],
        "duration_mismatch_sec": summarize_numeric(report["duration_mismatch_sec"]),
        "frame_counts": summarize_numeric(report["frame_counts"]),
        "word_counts": summarize_numeric(report["word_counts"]),
        "transcript_alignment": {
            "count": len(transcript_stats),
            "same_normalized_text_count": exact_matches,
            "same_normalized_text_ratio": exact_matches / len(transcript_stats)
            if transcript_stats
            else 0.0,
            "exact_prefix_match_ratio": summarize_numeric(prefix_ratios),
        },
        "split_counts": dict(sorted(report["split_counts"].items())),
        "label_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in report["label_counts"].items()
        },
        "label_names": MSP_LABEL_NAMES,
        "label_codes": MSP_LABELS,
        "shards": shard_manifest,
        "config": report["config"],
        "errors": report["errors"][:100],
    }


def load_msp_aligned(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Dataset, Dataset, int, list[str]]:
    metadata_path = output_dir / "metadata.jsonl"
    shards_path = output_dir / "shards.json"
    records = [
        json.loads(line)
        for line in metadata_path.read_text().splitlines()
        if line.strip()
    ]
    shards = json.loads(shards_path.read_text())

    for shard in shards:
        shard_records = [
            record for record in records if record["feature_shard"] == shard["shard"]
        ]
        with np.load(output_dir / shard["audio_frames"], allow_pickle=False) as audio_archive:
            with np.load(
                output_dir / shard["word_features"], allow_pickle=False
            ) as word_archive:
                for record in shard_records:
                    utterance_id = record["utterance_id"]
                    record["audio_frame_features"] = audio_archive[utterance_id].tolist()
                    record["audio_frame_start_sec"] = audio_archive[
                        f"{utterance_id}__start_sec"
                    ].tolist()
                    record["audio_frame_end_sec"] = audio_archive[
                        f"{utterance_id}__end_sec"
                    ].tolist()
                    record["word_audio_features"] = word_archive[utterance_id].tolist()

    train_records = [record for record in records if record["split"] == "train"]
    test_records = [record for record in records if record["split"] == "test"]
    return (
        Dataset.from_list(train_records),
        Dataset.from_list(test_records),
        len(MSP_LABELS),
        MSP_LABEL_NAMES,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build temporally aligned MSP-PODCAST text/prosody artifacts."
    )
    parser.add_argument("--msp-root", type=Path, default=MSP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tolerance-sec", type=float, default=0.25)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="Print a stderr progress line every N discovered utterances; use 0 to disable.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=5000,
        help="Number of included utterances per compressed NPZ shard.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_msp_aligned_dataset(
        output_dir=args.output_dir,
        msp_root=args.msp_root,
        sample_rate=args.sample_rate,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        limit=args.limit,
        tolerance_sec=args.tolerance_sec,
        progress_every=args.progress_every,
        shard_size=args.shard_size,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
