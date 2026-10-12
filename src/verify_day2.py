"""Verify the day-2 data, checkpoint and recorded-inference gate using local assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
from faster_whisper import WhisperModel

from src.core.events import AudioChunk, SpeechSegment
from src.emotion_checkpoint import EMOTION_LABELS, read_checkpoint
from src.emotion_fusion_infer import CrossModalEmotionPredictor
from src.inference_pipeline import SpeechProcessor
from src.prepare_iemocap import DEFAULT_METADATA
from src.segmentAudioTorch import SileroVAD
from to_ds import load_iemocap


def verify(args):
    metadata_bytes = args.metadata.read_bytes()
    records = [json.loads(line) for line in metadata_bytes.splitlines()]
    by_id = {r["utterance_id"]: r for r in records}
    audit = json.loads(args.metadata.with_name("validation_report.json").read_text())
    metadata_hash = hashlib.sha256(metadata_bytes).hexdigest()
    assert audit["passed"] and audit["metadata_sha256"] == metadata_hash
    assert len(by_id) == len(records) == audit["metadata_records"]
    assert all(
        r["split"] == ("test" if r["session"] == 5 else "train") for r in records
    )
    # Compare against the existing loader, including its order and raw labels.
    original_train, original_test, num_labels, _ = load_iemocap()
    assert num_labels == 11
    for split, original in (("train", original_train), ("test", original_test)):
        subset = [r for r in records if r["split"] == split]
        assert len(subset) == len(original)
        for row, source in zip(subset, original.select_columns(["text", "label"])):
            assert row["text"] == source["text"] and row["label"] == source["label"]
    checkpoint = read_checkpoint(args.checkpoint, "iemocap_emotion_4")
    provenance = checkpoint["provenance"]
    assert provenance["metadata_sha256"] == metadata_hash
    train_ids, validation_ids = map(
        set, (provenance["train_utterance_ids"], provenance["validation_utterance_ids"])
    )
    assert train_ids and validation_ids and train_ids.isdisjoint(validation_ids)
    assert all(
        by_id[uid]["session"] not in (provenance["validation_session"], 5)
        for uid in train_ids
    )
    assert all(
        by_id[uid]["session"] == provenance["validation_session"]
        for uid in validation_ids
    )
    metrics = json.loads(args.checkpoint.with_suffix(".metrics.json").read_text())
    assert not metrics["test_evaluated"]
    assert provenance["epoch"] == metrics["best_epoch"]
    sample = by_id[args.utterance_id]
    assert sample["session"] != 5, "Use a development utterance for this gate"
    audio, sr = sf.read(sample["audio_path"], dtype="float32")
    predictor = CrossModalEmotionPredictor(
        args.checkpoint, args.device, local_files_only=True
    )
    asr = WhisperModel(
        "tiny.en", device="cpu", compute_type="int8", local_files_only=True
    )
    processor = SpeechProcessor(asr, predictor)
    result = processor.process(SpeechSegment(0, len(audio) / sr, sr, audio))
    assert (
        result["status"] == "ok"
        and list(result["emotion"]["probabilities"]) == EMOTION_LABELS
    )
    assert np.isclose(sum(result["emotion"]["probabilities"].values()), 1.0)
    # Feed the same WAV through the real VAD in 512-sample chunks as a second,
    # microphone-shaped replay. Tail silence lets the segment close naturally.
    vad = SileroVAD(threshold=0.4)
    padded_audio = np.pad(audio, (0, 16000))
    detected = []
    for start in range(0, len(padded_audio), 512):
        chunk = np.pad(
            padded_audio[start : start + 512],
            (0, max(0, start + 512 - len(padded_audio))),
        )
        _, speech = vad.process_chunk(AudioChunk(start / sr, sr, chunk))
        if speech is not None:
            detected.append(processor.process(speech))
    assert detected and any(
        r["status"] == "ok" and r["emotion"] is not None for r in detected
    )
    # Real pretrained encoders: adding a longer batch member must not change
    # the short clip's prediction. No synthetic encoder is used here.
    waveform = torch.from_numpy(audio[:8000]).unsqueeze(0)
    second_length = max(waveform.shape[1] + 1600, 16000)
    waves = torch.zeros(2, second_length)
    waves[0, : waveform.shape[1]] = waveform[0]
    waves[1] = torch.linspace(-0.1, 0.1, second_length)
    mask = torch.zeros_like(waves, dtype=torch.long)
    mask[0, : waveform.shape[1]] = 1
    mask[1] = 1
    single_tokens = predictor.tokenizer(result["text"], return_tensors="pt")
    batch_tokens = predictor.tokenizer(
        [result["text"], "A longer text for the padding check."],
        padding=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        single = predictor.model(
            single_tokens["input_ids"].to(args.device),
            single_tokens["attention_mask"].to(args.device),
            waveform.to(args.device),
        ).logits[0]
        batch = predictor.model(
            batch_tokens["input_ids"].to(args.device),
            batch_tokens["attention_mask"].to(args.device),
            waves.to(args.device),
            mask.to(args.device),
        ).logits[0]
    difference = float((single - batch).abs().max())
    torch.testing.assert_close(single, batch, atol=2e-5, rtol=2e-5)
    report = {
        "passed": True,
        "scope": "day_1_2_integration_not_benchmark",
        "device": args.device,
        "metadata_records": len(records),
        "original_loader_order_and_labels_match": True,
        "metadata_sha256": metadata_hash,
        "dialog_videos": audit["dialog_videos"],
        "data_warning_count": len(audit["warnings"]),
        "session_5_evaluated": False,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_kind": provenance["kind"],
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "best_epoch": provenance["epoch"],
        "utterance_id": args.utterance_id,
        "recorded_result": result,
        "vad_replay_results": detected,
        "real_encoder_padding_max_abs_difference": difference,
        "not_validated": [
            "live microphone hardware",
            "live webcam",
            "trained visual fusion",
            "full-dataset emotion accuracy",
            "p95 latency",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "passed",
                    "metadata_records",
                    "train_count",
                    "validation_count",
                    "real_encoder_padding_max_abs_difference",
                )
            },
            indent=2,
        )
    )
    print(f"Verification saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/mvp/day2_smoke.pt"
    )
    parser.add_argument("--utterance-id", default="Ses01F_impro01_F000")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/mvp/day2_verification.json",
    )
    verify(parser.parse_args())
