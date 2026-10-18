"""
IEMOCAP embedding datasets generator

Current two version:
(1) Sliding Window. Prosdic features are extract of every audio frame, averaging those that overlaps with the alignment of words spoken.

(2) Text Centered. The prosodic features are captured from audio of exactly the duration of each word with a small buffer before and after.


See /docs/embeddings.md for more details
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import opensmile
import soundfile as sf
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from to_ds import (  # noqa: E402
    IEMOCAP_LABEL2IDX,
    IEMOCAP_LABELS,
    IEMOCAP_LABEL_NAMES,
    _parse_iemocap_emo_file,
    _parse_iemocap_transcript,
)

DATA_ROOT = ROOT / "data"
IEMOCAP_ROOT = DATA_ROOT / "IEMOCAP_full_release"
DEFAULT_OUTPUT_DIR = DATA_ROOT / "iemocap_aligned_features"
WORD_FEATURE_MODES = ("overlap", "centered")
CONTROL_WORDS = {"<s>", "</s>", "<sil>"}
FRAME_SECONDS = 0.01
SAMPLE_RATE = 16000


def resolve_data_path(stored_path: str | Path, data_root: Path = DATA_ROOT) -> Path:
    """Resolve a path recorded in a cached artifact against this checkout.

    Cached metadata (e.g. ``metadata.jsonl``) stores absolute paths from
    whichever machine generated it. Rebase everything after the last ``data``
    component onto ``data_root`` so artifacts survive repo moves. Returns the
    original path untouched if it exists or has no ``data`` component; callers
    must still check ``exists()``.
    """
    path = Path(stored_path)
    if path.exists():
        return path
    parts = path.parts
    if "data" not in parts:
        return path
    tail_index = len(parts) - 1 - tuple(reversed(parts)).index("data")
    return data_root.joinpath(*parts[tail_index + 1 :])


@dataclass(frozen=True)
class WordSpan:
    word: str
    raw_word: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class AudioFrames:
    features: np.ndarray
    start_sec: np.ndarray
    end_sec: np.ndarray


def normalize_alignment_word(word: str) -> str:
    """Normalize only enough for matching/debugging while preserving timing."""
    word = re.sub(r"\(\d+\)$", "", word)
    return word.lower()


def normalize_text_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?", text.lower())
    return tokens


def parse_wdseg(path: Path) -> list[WordSpan]:
    words: list[WordSpan] = []
    with path.open() as f:
        for line in f:
            match = re.match(r"^\s*(\d+)\s+(\d+)\s+[-\d]+\s+(\S+)\s*$", line)
            if not match:
                continue

            start_frame = int(match.group(1))
            end_frame = int(match.group(2))
            raw_word = match.group(3)
            if raw_word in CONTROL_WORDS:
                continue

            word = normalize_alignment_word(raw_word)
            words.append(
                WordSpan(
                    word=word,
                    raw_word=raw_word,
                    start_sec=start_frame * FRAME_SECONDS,
                    end_sec=(end_frame + 1) * FRAME_SECONDS,
                )
            )
    return words


def read_audio_mono(
    path: Path, sample_rate: int = SAMPLE_RATE
) -> tuple[np.ndarray, int]:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    if source_rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=source_rate, target_sr=sample_rate)
        source_rate = sample_rate
    return np.asarray(audio, dtype=np.float32), source_rate


def get_wav_duration(path: Path) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def window_starts(duration: float, window_sec: float, hop_sec: float) -> list[float]:
    if duration <= 0:
        return []
    if duration <= window_sec:
        return [0.0]

    starts = list(np.arange(0.0, duration - window_sec + 1e-9, hop_sec))
    tail_start = max(0.0, duration - window_sec)
    if not starts or not math.isclose(starts[-1], tail_start, abs_tol=1e-4):
        starts.append(tail_start)
    return [float(start) for start in starts]


class SlidingOpenSmileExtractor:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_sec: float = 1.0,
        hop_sec: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self.feature_names: list[str] | None = None

    def extract(self, audio: np.ndarray) -> AudioFrames:
        duration = len(audio) / self.sample_rate
        starts = window_starts(duration, self.window_sec, self.hop_sec)
        rows: list[np.ndarray] = []
        ends: list[float] = []

        for start_sec in starts:
            end_sec = min(duration, start_sec + self.window_sec)
            start_idx = int(round(start_sec * self.sample_rate))
            end_idx = max(start_idx + 1, int(round(end_sec * self.sample_rate)))
            segment = audio[start_idx:end_idx]
            feature_df = self.smile.process_signal(segment, self.sample_rate)
            if self.feature_names is None:
                self.feature_names = [str(name) for name in feature_df.columns]
            vector = feature_df.values.astype(np.float32).reshape(-1)
            rows.append(np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0))
            ends.append(float(end_sec))

        if not rows:
            width = len(self.feature_names or [])
            return AudioFrames(
                features=np.zeros((0, width), dtype=np.float32),
                start_sec=np.zeros((0,), dtype=np.float32),
                end_sec=np.zeros((0,), dtype=np.float32),
            )

        return AudioFrames(
            features=np.vstack(rows).astype(np.float32),
            start_sec=np.asarray(starts, dtype=np.float32),
            end_sec=np.asarray(ends, dtype=np.float32),
        )


def centered_window_bounds(
    word: WordSpan,
    duration_sec: float,
    context_sec: float,
) -> tuple[float, float]:
    if duration_sec <= 0:
        return 0.0, 0.0
    if duration_sec <= context_sec:
        return 0.0, duration_sec

    midpoint = (word.start_sec + word.end_sec) / 2.0
    start_sec = midpoint - (context_sec / 2.0)
    end_sec = midpoint + (context_sec / 2.0)
    if start_sec < 0.0:
        start_sec = 0.0
        end_sec = context_sec
    if end_sec > duration_sec:
        end_sec = duration_sec
        start_sec = max(0.0, duration_sec - context_sec)
    return float(start_sec), float(end_sec)


class WordCenteredOpenSmileExtractor:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        context_sec: float = 1.0,
    ):
        self.sample_rate = sample_rate
        self.context_sec = context_sec
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        self.feature_names: list[str] | None = None

    def extract(self, audio: np.ndarray, words: list[WordSpan]) -> AudioFrames:
        duration = len(audio) / self.sample_rate
        rows: list[np.ndarray] = []
        starts: list[float] = []
        ends: list[float] = []

        for word in words:
            start_sec, end_sec = centered_window_bounds(
                word, duration, self.context_sec
            )
            start_idx = int(round(start_sec * self.sample_rate))
            end_idx = max(start_idx + 1, int(round(end_sec * self.sample_rate)))
            segment = audio[start_idx:end_idx]
            feature_df = self.smile.process_signal(segment, self.sample_rate)
            if self.feature_names is None:
                self.feature_names = [str(name) for name in feature_df.columns]
            vector = feature_df.values.astype(np.float32).reshape(-1)
            rows.append(np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0))
            starts.append(start_sec)
            ends.append(end_sec)

        if not rows:
            width = len(self.feature_names or [])
            return AudioFrames(
                features=np.zeros((0, width), dtype=np.float32),
                start_sec=np.zeros((0,), dtype=np.float32),
                end_sec=np.zeros((0,), dtype=np.float32),
            )

        return AudioFrames(
            features=np.vstack(rows).astype(np.float32),
            start_sec=np.asarray(starts, dtype=np.float32),
            end_sec=np.asarray(ends, dtype=np.float32),
        )


def aggregate_word_features(
    words: list[WordSpan],
    frames: AudioFrames,
) -> tuple[np.ndarray, np.ndarray]:
    width = frames.features.shape[1] if frames.features.ndim == 2 else 0
    word_features = np.zeros((len(words), width), dtype=np.float32)
    mask = np.zeros((len(words),), dtype=np.float32)

    for word_idx, word in enumerate(words):
        overlap_start = np.maximum(frames.start_sec, word.start_sec)
        overlap_end = np.minimum(frames.end_sec, word.end_sec)
        overlaps = np.maximum(0.0, overlap_end - overlap_start)
        total_overlap = float(overlaps.sum())
        if total_overlap <= 0:
            continue

        weights = overlaps / total_overlap
        word_features[word_idx] = weights.astype(np.float32) @ frames.features
        mask[word_idx] = 1.0

    return word_features, mask


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


def iter_iemocap_utterances(iemocap_root: Path = IEMOCAP_ROOT):
    for session_num in range(1, 6):
        session_dir = iemocap_root / f"Session{session_num}"
        emo_dir = session_dir / "dialog" / "EmoEvaluation"
        trans_dir = session_dir / "dialog" / "transcriptions"
        wav_dir = session_dir / "sentences" / "wav"
        align_dir = session_dir / "sentences" / "ForcedAlignment"

        emo_files = [
            f
            for f in emo_dir.glob("*.txt")
            if f.stem not in ("Attribute", "Categorical", "Self-evaluation")
        ]

        for emo_file in sorted(emo_files):
            dialog_id = emo_file.stem
            trans_file = trans_dir / f"{dialog_id}.txt"
            if not trans_file.exists():
                continue

            utt2emo = _parse_iemocap_emo_file(emo_file)
            utt2text = _parse_iemocap_transcript(trans_file)

            for utterance_id, emo_record in utt2emo.items():
                label_code = emo_record[0]
                if utterance_id not in utt2text or label_code not in IEMOCAP_LABEL2IDX:
                    continue

                audio_path = wav_dir / dialog_id / f"{utterance_id}.wav"
                wdseg_path = align_dir / dialog_id / f"{utterance_id}.wdseg"
                if not audio_path.exists():
                    continue

                yield {
                    "utterance_id": utterance_id,
                    "dialog_id": dialog_id,
                    "session": session_num,
                    "split": "train" if session_num <= 4 else "test",
                    "text": utt2text[utterance_id],
                    "label_code": label_code,
                    "label": IEMOCAP_LABEL2IDX[label_code],
                    "vad": [float(value) for value in emo_record[1:]],
                    "audio_path": str(audio_path),
                    "wdseg_path": str(wdseg_path),
                }


def build_iemocap_aligned_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    iemocap_root: Path = IEMOCAP_ROOT,
    sample_rate: int = SAMPLE_RATE,
    window_sec: float = 1.0,
    hop_sec: float = 0.5,
    word_feature_mode: str = "overlap",
    word_context_sec: float = 1.0,
    limit: int | None = None,
    tolerance_sec: float = 0.25,
    progress_every: int = 100,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if word_feature_mode not in WORD_FEATURE_MODES:
        raise ValueError(
            f"word_feature_mode must be one of {WORD_FEATURE_MODES}, "
            f"got {word_feature_mode!r}"
        )
    if word_feature_mode == "centered":
        extractor = WordCenteredOpenSmileExtractor(
            sample_rate=sample_rate,
            context_sec=word_context_sec,
        )
    else:
        extractor = SlidingOpenSmileExtractor(
            sample_rate=sample_rate,
            window_sec=window_sec,
            hop_sec=hop_sec,
        )
    utterances = list(iter_iemocap_utterances(iemocap_root))
    total_to_process = len(utterances)
    started_at = time.monotonic()

    metadata_records: list[dict[str, Any]] = []
    audio_arrays: dict[str, np.ndarray] = {}
    word_arrays: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {
        "utterances_discovered": 0,
        "included_utterances": 0,
        "missing_wdseg": 0,
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
            "word_feature_mode": word_feature_mode,
            "word_context_sec": word_context_sec,
            "alignment_frame_seconds": FRAME_SECONDS,
            "tolerance_sec": tolerance_sec,
            "progress_every": progress_every,
        },
    }

    def print_progress(processed: int, *, force: bool = False) -> None:
        if progress_every <= 0:
            return
        if not force and processed % progress_every != 0:
            return
        elapsed = time.monotonic() - started_at
        target = f" included_target={limit}" if limit is not None else ""
        print(
            "iemocap-aligned "
            f"processed={processed}/{total_to_process} "
            f"included={report['included_utterances']} "
            f"missing_wdseg={report['missing_wdseg']} "
            f"invalid_spans={report['invalid_word_spans']} "
            f"feature_failures={report['feature_extraction_failures']} "
            f"elapsed={elapsed:.1f}s"
            f"{target}",
            file=sys.stderr,
            flush=True,
        )

    print_progress(0, force=True)
    for base_record in utterances:
        if limit is not None and report["included_utterances"] >= limit:
            break
        report["utterances_discovered"] += 1

        utterance_id = base_record["utterance_id"]
        wdseg_path = Path(base_record["wdseg_path"])
        if not wdseg_path.exists():
            report["missing_wdseg"] += 1
            print_progress(report["utterances_discovered"])
            continue

        audio_path = Path(base_record["audio_path"])
        duration_sec = get_wav_duration(audio_path)
        words = parse_wdseg(wdseg_path)
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
            if word_feature_mode == "centered":
                frames = extractor.extract(audio, words)
                word_features = frames.features
                word_mask = np.ones((len(words),), dtype=np.float32)
            else:
                frames = extractor.extract(audio)
                word_features, word_mask = aggregate_word_features(words, frames)
        except Exception as exc:  # pragma: no cover - keeps long batch jobs alive
            report["feature_extraction_failures"] += 1
            report["errors"].append(
                {"utterance_id": utterance_id, "errors": [repr(exc)]}
            )
            print_progress(report["utterances_discovered"])
            continue

        no_overlap = int((word_mask == 0).sum())
        report["words_without_audio_overlap"] += no_overlap

        final_alignment_end = words[-1].end_sec if words else 0.0
        mismatch = final_alignment_end - duration_sec
        match_stats = transcript_match_stats(base_record["text"], words)

        record = {
            **{k: v for k, v in base_record.items() if k != "wdseg_path"},
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
            "word_audio_context_start_sec": frames.start_sec.tolist()
            if word_feature_mode == "centered"
            else None,
            "word_audio_context_end_sec": frames.end_sec.tolist()
            if word_feature_mode == "centered"
            else None,
            "word_audio_feature_mask": word_mask.astype(int).tolist(),
            "transcript_alignment": match_stats,
        }
        metadata_records.append(record)

        audio_arrays[utterance_id] = frames.features
        audio_arrays[f"{utterance_id}__start_sec"] = frames.start_sec
        audio_arrays[f"{utterance_id}__end_sec"] = frames.end_sec
        word_arrays[utterance_id] = word_features
        word_arrays[f"{utterance_id}__mask"] = word_mask

        split = base_record["split"]
        report["included_utterances"] += 1
        report["split_counts"][split] += 1
        report["label_counts"][split][base_record["label_code"]] += 1
        report["duration_mismatch_sec"].append(mismatch)
        report["frame_counts"].append(int(frames.features.shape[0]))
        report["word_counts"].append(len(words))
        report["transcript_alignment"].append(match_stats)
        print_progress(report["utterances_discovered"])

    feature_names = extractor.feature_names or []
    if metadata_records and len(feature_names) != 88:
        report["errors"].append(
            {
                "utterance_id": None,
                "errors": [f"expected 88 OpenSMILE features, got {len(feature_names)}"],
            }
        )

    metadata_path = output_dir / "metadata.jsonl"
    with metadata_path.open("w") as f:
        for record in metadata_records:
            f.write(json.dumps(record) + "\n")

    np.savez_compressed(output_dir / "audio_frames.npz", **audio_arrays)
    np.savez_compressed(output_dir / "word_features.npz", **word_arrays)
    (output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    serializable_report = summarize_report(report)
    print_progress(report["utterances_discovered"], force=True)
    (output_dir / "alignment_report.json").write_text(
        json.dumps(serializable_report, indent=2)
    )
    return serializable_report


def summarize_numeric(values: list[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    transcript_stats = report["transcript_alignment"]
    exact_matches = sum(1 for item in transcript_stats if item["same_normalized_text"])
    prefix_ratios = [item["exact_prefix_match_ratio"] for item in transcript_stats]
    label_counts = {
        split: dict(sorted(counts.items()))
        for split, counts in report["label_counts"].items()
    }
    return {
        "utterances_discovered": report["utterances_discovered"],
        "included_utterances": report["included_utterances"],
        "missing_wdseg": report["missing_wdseg"],
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
        "label_counts": label_counts,
        "label_names": IEMOCAP_LABEL_NAMES,
        "label_codes": IEMOCAP_LABELS,
        "config": report["config"],
        "errors": report["errors"][:100],
    }


def load_iemocap_aligned(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Dataset, Dataset, int, list[str]]:
    metadata_path = output_dir / "metadata.jsonl"
    audio_npz_path = output_dir / "audio_frames.npz"
    word_npz_path = output_dir / "word_features.npz"

    records = [
        json.loads(line)
        for line in metadata_path.read_text().splitlines()
        if line.strip()
    ]
    with np.load(audio_npz_path, allow_pickle=False) as audio_archive:
        with np.load(word_npz_path, allow_pickle=False) as word_archive:
            for record in records:
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
        len(IEMOCAP_LABELS),
        IEMOCAP_LABEL_NAMES,
    )


def tokenize_record_with_word_audio(
    record: dict[str, Any],
    tokenizer,
    max_length: int = 128,
    feature_width: int | None = None,
) -> dict[str, Any]:
    """
    Tokenize one aligned record and copy each word-level audio vector to its subwords.

    The tokenizer must support Hugging Face fast-tokenizer word_ids(), which is
    the normal path for BERT/RoBERTa-style token-level fusion.
    """
    words = [word["word"] for word in record["words"]]
    word_features = np.asarray(record["word_audio_features"], dtype=np.float32)
    if feature_width is None:
        feature_width = int(word_features.shape[1]) if word_features.ndim == 2 else 0

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    word_ids = encoding.word_ids()
    token_audio = np.zeros((max_length, feature_width), dtype=np.float32)
    audio_token_mask = np.zeros((max_length,), dtype=np.int64)

    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx >= len(word_features):
            continue
        token_audio[token_idx] = word_features[word_idx]
        audio_token_mask[token_idx] = int(record["word_audio_feature_mask"][word_idx])

    return {
        **encoding,
        "audio_features": token_audio.tolist(),
        "audio_token_mask": audio_token_mask.tolist(),
        "label": record["label"],
        "utterance_id": record["utterance_id"],
    }


def tokenize_dataset_with_word_audio(
    dataset: Dataset,
    tokenizer,
    max_length: int = 128,
    feature_width: int = 88,
) -> Dataset:
    records = [
        tokenize_record_with_word_audio(
            record,
            tokenizer=tokenizer,
            max_length=max_length,
            feature_width=feature_width,
        )
        for record in dataset
    ]
    return Dataset.from_list(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build temporally aligned IEMOCAP text/prosody artifacts."
    )
    parser.add_argument("--iemocap-root", type=Path, default=IEMOCAP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    parser.add_argument("--window-sec", type=float, default=1.0)
    parser.add_argument("--hop-sec", type=float, default=0.5)
    parser.add_argument(
        "--word-feature-mode",
        choices=WORD_FEATURE_MODES,
        default="overlap",
        help=(
            "overlap pools fixed sliding windows into words; centered extracts "
            "one OpenSMILE context window centered on each word."
        ),
    )
    parser.add_argument(
        "--word-context-sec",
        type=float,
        default=1.0,
        help="Context-window length for --word-feature-mode centered.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tolerance-sec", type=float, default=0.25)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print a stderr progress line every N discovered utterances; use 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_iemocap_aligned_dataset(
        output_dir=args.output_dir,
        iemocap_root=args.iemocap_root,
        sample_rate=args.sample_rate,
        window_sec=args.window_sec,
        hop_sec=args.hop_sec,
        word_feature_mode=args.word_feature_mode,
        word_context_sec=args.word_context_sec,
        limit=args.limit,
        tolerance_sec=args.tolerance_sec,
        progress_every=args.progress_every,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
