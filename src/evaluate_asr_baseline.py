"""Session 4 diagnostic with the actual tiny.en ASR path; no model selection."""

from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from sklearn.metrics import accuracy_score, f1_score, recall_score
import torch

from src.emotion_checkpoint import EMOTION_LABEL_MAP, PreprocessingConfig, prepare_audio
from src.emotion_fusion_infer import CrossModalEmotionPredictor
from src.iemocap_embedding_cache import sha256, write_json
from src.inference_pipeline import transcribe_samples
from src.prepare_iemocap import DEFAULT_METADATA


def diagnostic_metrics(rows):
    if not rows:
        return {
            "count": 0,
            "abstentions": 0,
            "accuracy": None,
            "macro_f1": None,
            "uar": None,
            "confusion_with_abstention_column": np.zeros((4, 5), dtype=int).tolist(),
        }
    truth = [r["label"] for r in rows]
    predicted = [r["prediction"] for r in rows]
    matrix = np.zeros((4, 5), dtype=int)
    for a, b in zip(truth, predicted):
        matrix[a, b if b >= 0 else 4] += 1
    return {
        "count": len(rows),
        "abstentions": sum(p < 0 for p in predicted),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(
            f1_score(
                truth, predicted, labels=[0, 1, 2, 3], average="macro", zero_division=0
            )
        ),
        "uar": float(
            recall_score(
                truth, predicted, labels=[0, 1, 2, 3], average="macro", zero_division=0
            )
        ),
        "confusion_with_abstention_column": matrix.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/mvp/audio_text_baseline.pt"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/processed/mvp/asr_validation"
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--transcribe-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    preprocessing = PreprocessingConfig()
    audit = json.loads(args.metadata.with_name("validation_report.json").read_text())
    if not audit["passed"] or audit["metadata_sha256"] != sha256(args.metadata):
        raise ValueError("ASR evaluation requires a passing metadata audit")
    rows = [
        r
        for r in map(json.loads, args.metadata.read_text().splitlines())
        if r["session"] == 4 and r["label_code"] in EMOTION_LABEL_MAP
    ]
    if not rows:
        raise ValueError("No Session 4 validation examples")
    model_path = Path(
        snapshot_download("Systran/faster-whisper-tiny.en", local_files_only=True)
    )
    config = {
        "metadata_sha256": sha256(args.metadata),
        "session": 4,
        "preprocessing": asdict(preprocessing),
        "asr_model": "Systran/faster-whisper-tiny.en",
        "asr_revision": model_path.name,
        "asr_weights_sha256": sha256(model_path / "model.bin"),
        "beam_size": 1,
        "language": "en",
        "vad_filter": False,
        "compute_type": "int8",
        "audio_mode": "already_segmented_utterance",
    }
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("ASR cache protocol changed; choose another output")
    write_json(config_path, config)
    transcript_path = args.output / "transcripts.jsonl"
    cached = {}
    if transcript_path.exists():
        for row in map(json.loads, transcript_path.read_text().splitlines()):
            if row["utterance_id"] in cached:
                raise ValueError("Duplicate ASR cache record")
            cached[row["utterance_id"]] = row
    expected = {r["utterance_id"] for r in rows}
    if not set(cached) <= expected:
        raise ValueError("ASR cache includes an unexpected sample")
    transcriber = None
    with transcript_path.open("a") as handle:
        for i, row in enumerate(rows):
            uid = row["utterance_id"]
            audio_hash = sha256(row["audio_path"])
            if uid in cached:
                if cached[uid]["audio_sha256"] != audio_hash:
                    raise ValueError("ASR audio source changed")
                continue
            if transcriber is None:
                transcriber = WhisperModel(
                    str(model_path), device="cpu", compute_type="int8", cpu_threads=2
                )
            audio, sr = sf.read(row["audio_path"], dtype="float32")
            started = time.perf_counter()
            text = transcribe_samples(
                transcriber, prepare_audio(audio, sr, preprocessing).numpy()
            )
            record = {
                "utterance_id": uid,
                "audio_sha256": audio_hash,
                "text": text,
                "reference_text": row["text"],
                "duration_sec": row["duration_sec"],
                "asr_elapsed_ms": 1000 * (time.perf_counter() - started),
            }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            cached[uid] = record
            if (i + 1) % 100 == 0 or i + 1 == len(rows):
                print(f"ASR validation {i + 1}/{len(rows)}", flush=True)
    del transcriber
    if args.transcribe_only:
        print(f"Cached all {len(rows)} validation transcripts", flush=True)
        return
    predictor = CrossModalEmotionPredictor(
        args.checkpoint, args.device, local_files_only=True
    )
    if predictor.provenance["metadata_sha256"] != config["metadata_sha256"]:
        raise ValueError("ASR/checkpoint metadata mismatch")
    if predictor.preprocessing != preprocessing:
        raise ValueError("Checkpoint/ASR preprocessing mismatch")
    predictions = []
    for i, row in enumerate(rows):
        uid = row["utterance_id"]
        text = cached[uid]["text"]
        prediction = -1
        probabilities = None
        if text:
            audio, sr = sf.read(row["audio_path"], dtype="float32")
            result = predictor.predict(text, audio, sr)
            prediction = result.label_index
            probabilities = result.probabilities
        predictions.append(
            {
                "utterance_id": uid,
                "label": EMOTION_LABEL_MAP[row["label_code"]],
                "prediction": prediction,
                "probabilities": probabilities,
                "duration_sec": row["duration_sec"],
                "asr_text": text,
            }
        )
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f"ASR emotion evaluation {i + 1}/{len(rows)}", flush=True)
    report = {
        "scope": "full_session4_asr_diagnostic_no_selection",
        "test_evaluated": False,
        "checkpoint_sha256": sha256(args.checkpoint),
        "transcripts_sha256": sha256(transcript_path),
        "config": config,
        "metrics": diagnostic_metrics(predictions),
        "by_audio_truncation": {
            name: diagnostic_metrics(
                [p for p in predictions if (p["duration_sec"] > 6) == truncated]
            )
            for name, truncated in (
                ("at_most_six_seconds", False),
                ("longer_than_six_seconds", True),
            )
        },
        "interpretation": "Already segmented utterances through actual Whisper and public raw-audio predictor. Difference from reference-text results includes ASR errors and six-second text/audio truncation. Does not include VAD segmentation or establish live-device accuracy.",
    }
    write_json(args.output / "metrics.json", report)
    (args.output / "predictions.jsonl").write_text(
        "".join(json.dumps(p) + "\n" for p in predictions)
    )
    print(
        json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
