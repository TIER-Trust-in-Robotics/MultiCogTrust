from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer

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


class LazyMSPSlidingFrameFusionDataset(TorchDataset):
    """MSP sliding-frame fusion dataset with lazy audio shard loading."""

    def __init__(
        self,
        output_dir: Path,
        split: str,
        tokenizer_name: str,
        max_length: int,
        limit: int | None = None,
        shard_cache_size: int = 2,
    ) -> None:
        self.output_dir = output_dir
        self.split = split
        self.shard_cache_size = max(shard_cache_size, 1)
        self._shard_cache: OrderedDict[int, np.lib.npyio.NpzFile] = OrderedDict()

        metadata_path = output_dir / "metadata.jsonl"
        shards_path = output_dir / "shards.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing MSP metadata file: {metadata_path}")
        if not shards_path.exists():
            raise FileNotFoundError(f"Missing MSP shard manifest: {shards_path}")

        shards = json.loads(shards_path.read_text())
        self.audio_shards = {
            int(shard["shard"]): output_dir / shard["audio_frames"] for shard in shards
        }
        self.records: list[dict[str, Any]] = []
        with metadata_path.open() as metadata_file:
            for line in metadata_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record["split"] != split:
                    continue
                self.records.append(
                    {
                        "utterance_id": record["utterance_id"],
                        "text": record["text"],
                        "label": int(record["label"]),
                        "feature_shard": int(record["feature_shard"]),
                    }
                )
                if limit is not None and len(self.records) >= limit:
                    break

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        tokenized = tokenizer(
            [record["text"] for record in self.records],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        self.input_ids = tokenized["input_ids"]
        self.attention_mask = tokenized["attention_mask"]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        utterance_id = record["utterance_id"]
        shard = self._get_audio_shard(record["feature_shard"])
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "audio_frame_features": np.asarray(shard[utterance_id], dtype=np.float32),
            "label": record["label"],
        }

    def _get_audio_shard(self, shard_idx: int) -> np.lib.npyio.NpzFile:
        if shard_idx in self._shard_cache:
            archive = self._shard_cache.pop(shard_idx)
            self._shard_cache[shard_idx] = archive
            return archive

        shard_path = self.audio_shards[shard_idx]
        archive = np.load(shard_path, allow_pickle=False)
        self._shard_cache[shard_idx] = archive
        while len(self._shard_cache) > self.shard_cache_size:
            _, evicted = self._shard_cache.popitem(last=False)
            evicted.close()
        return archive

    def close(self) -> None:
        for archive in self._shard_cache.values():
            archive.close()
        self._shard_cache.clear()

    def __del__(self) -> None:
        self.close()


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
    resume: bool = False,
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
    existing_ids: set[str] = set()
    existing_records: list[dict[str, Any]] = []
    existing_report: dict[str, Any] = {}
    shard_manifest: list[dict[str, Any]] = []
    shard_idx = 0
    shard_records = 0

    if resume:
        existing_ids, existing_records, shard_manifest, existing_report = load_resume_state(
            output_dir
        )
        if shard_manifest:
            shard_idx = max(int(shard["shard"]) for shard in shard_manifest) + 1

    report: dict[str, Any] = {
        "utterances_discovered": 0,
        "included_utterances": 0,
        "skipped_existing_utterances": 0,
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
            "resume": resume,
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
            f"skipped_existing={report['skipped_existing_utterances']} "
            f"missing_textgrid={report['missing_textgrid']} "
            f"invalid_spans={report['invalid_word_spans']} "
            f"feature_failures={report['feature_extraction_failures']} "
            f"elapsed={elapsed:.1f}s"
            f"{target}",
            file=sys.stderr,
            flush=True,
        )

    print_progress(0, force=True)
    metadata_mode = "a" if resume else "w"
    with metadata_path.open(metadata_mode) as metadata_file:
        for base_record in iter_msp_utterances(msp_root):
            if limit is not None and report["included_utterances"] >= limit:
                break
            report["utterances_discovered"] += 1
            utterance_id = base_record["utterance_id"]
            if utterance_id in existing_ids:
                report["skipped_existing_utterances"] += 1
                print_progress(report["utterances_discovered"])
                continue

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
    if resume and not feature_names:
        feature_names_path = output_dir / "feature_names.json"
        if feature_names_path.exists():
            feature_names = json.loads(feature_names_path.read_text())
    if report["included_utterances"] and len(feature_names) != 88:
        report["errors"].append(
            {
                "utterance_id": None,
                "errors": [f"expected 88 OpenSMILE features, got {len(feature_names)}"],
            }
        )

    if feature_names:
        (output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))
    (output_dir / "shards.json").write_text(json.dumps(shard_manifest, indent=2))
    serializable_report = summarize_report(
        report,
        shard_manifest,
        existing_records=existing_records,
        existing_report=existing_report,
    )
    print_progress(report["utterances_discovered"], force=True)
    (output_dir / "alignment_report.json").write_text(
        json.dumps(serializable_report, indent=2)
    )
    return serializable_report


def load_resume_state(
    output_dir: Path,
) -> tuple[set[str], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    metadata_path = output_dir / "metadata.jsonl"
    shards_path = output_dir / "shards.json"
    report_path = output_dir / "alignment_report.json"

    records: list[dict[str, Any]] = []
    if metadata_path.exists():
        records = [
            json.loads(line)
            for line in metadata_path.read_text().splitlines()
            if line.strip()
        ]

    shards: list[dict[str, Any]] = []
    if shards_path.exists():
        shards = json.loads(shards_path.read_text())
        for shard in shards:
            audio_path = output_dir / shard["audio_frames"]
            word_path = output_dir / shard["word_features"]
            if not audio_path.exists() or not word_path.exists():
                raise FileNotFoundError(
                    f"Cannot resume: missing shard files for shard {shard['shard']}"
                )

    existing_report: dict[str, Any] = {}
    if report_path.exists():
        existing_report = json.loads(report_path.read_text())

    existing_ids = {record["utterance_id"] for record in records}
    return existing_ids, records, shards, existing_report


def extend_report_from_records(
    report: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    for record in records:
        split = record["split"]
        report["included_utterances"] += 1
        report["split_counts"][split] += 1
        report["label_counts"][split][record["label_code"]] += 1
        report["frame_counts"].append(record["audio_frame_count"])
        report["word_counts"].append(record["word_count"])
        report["transcript_alignment"].append(record["transcript_alignment"])
        if record["words"]:
            mismatch = record["words"][-1]["end_sec"] - record["duration_sec"]
        else:
            mismatch = -record["duration_sec"]
        report["duration_mismatch_sec"].append(mismatch)
        report["words_without_audio_overlap"] += sum(
            1 for value in record["word_audio_feature_mask"] if value == 0
        )


def summarize_report(
    report: dict[str, Any],
    shard_manifest: list[dict[str, Any]],
    existing_records: list[dict[str, Any]] | None = None,
    existing_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if existing_records:
        report = {
            **report,
            "duration_mismatch_sec": list(report["duration_mismatch_sec"]),
            "frame_counts": list(report["frame_counts"]),
            "word_counts": list(report["word_counts"]),
            "transcript_alignment": list(report["transcript_alignment"]),
            "label_counts": {
                split: Counter(counts)
                for split, counts in report["label_counts"].items()
            },
            "split_counts": Counter(report["split_counts"]),
        }
        extend_report_from_records(report, existing_records)
    existing_report = existing_report or {}
    transcript_stats = report["transcript_alignment"]
    exact_matches = sum(1 for item in transcript_stats if item["same_normalized_text"])
    prefix_ratios = [item["exact_prefix_match_ratio"] for item in transcript_stats]
    return {
        "utterances_discovered": max(
            report["utterances_discovered"],
            existing_report.get("utterances_discovered", 0),
        ),
        "included_utterances": report["included_utterances"],
        "skipped_existing_utterances": report.get("skipped_existing_utterances", 0),
        "missing_textgrid": report["missing_textgrid"],
        "invalid_word_spans": max(
            report["invalid_word_spans"],
            existing_report.get("invalid_word_spans", 0),
        ),
        "feature_extraction_failures": max(
            report["feature_extraction_failures"],
            existing_report.get("feature_extraction_failures", 0),
        ),
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
        "errors": (existing_report.get("errors", []) + report["errors"])[:100],
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


def load_msp_sliding_frame_fusion_datasets(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tokenizer_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    limit_train: int | None = None,
    limit_test: int | None = None,
    shard_cache_size: int = 2,
) -> tuple[
    LazyMSPSlidingFrameFusionDataset,
    LazyMSPSlidingFrameFusionDataset,
    int,
    list[str],
]:
    train_dataset = LazyMSPSlidingFrameFusionDataset(
        output_dir=output_dir,
        split="train",
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        limit=limit_train,
        shard_cache_size=shard_cache_size,
    )
    test_dataset = LazyMSPSlidingFrameFusionDataset(
        output_dir=output_dir,
        split="test",
        tokenizer_name=tokenizer_name,
        max_length=max_length,
        limit=limit_test,
        shard_cache_size=shard_cache_size,
    )
    return train_dataset, test_dataset, len(MSP_LABELS), MSP_LABEL_NAMES


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append missing records without rebuilding existing metadata/shard entries.",
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
        resume=args.resume,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
