"""Replay the legacy seven-class MOSEI checkpoint on full validation only.

Keep argmax and expected sentiment separate. This is an audit of the existing
artifact under pinned current encoder revisions; historical revisions were not
recorded, so agreement with the archived metrics is checked rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit_benchmark_data import development_records
from src.benchmark_metrics import class_scores, mosei_metrics
from src.iemocap_embedding_cache import sha256, write_json
from src.models import CrossModalTransformerFusionClassifier
from src.train_cross_modal_transformer_iemocap import load_audio


def load_legacy_head(model, checkpoint):
    if checkpoint.get("num_labels") != 7:
        raise ValueError("Expected the legacy seven-class MOSEI checkpoint")
    if checkpoint.get("fusion_head_output_type") != model.fusion_head_output_type:
        raise ValueError("Checkpoint pooling mismatch")
    missing, unexpected = model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    if unexpected or any(not k.startswith(("text_encoder.", "audio_encoder.")) for k in missing):
        raise ValueError(f"Incompatible legacy head: {missing=}, {unexpected=}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark_quality_v1.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/benchmarks/mosei_clean_v1/development.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/benchmarks/mosei_legacy_validation_v1")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cfg = config["mosei"]["legacy_replay"]
    if config["test_model_inference"] or cfg["split"] != "valid":
        raise ValueError("Legacy replay is restricted to validation")
    contract = json.loads(args.manifest.with_name("development_contract.json").read_text())
    if (contract["manifest_sha256"] != sha256(args.manifest)
            or contract["label_csv_sha256"] != config["mosei"]["label_csv_sha256"]):
        raise ValueError("Manifest does not match the audited label file")
    rows = [r for r in development_records(args.manifest) if r["split"] == "valid"]
    checkpoint_path = ROOT / cfg["checkpoint"]
    protocol = {
        "task": "mosei_legacy_7class_validation_replay",
        "config": cfg, "checkpoint_sha256": sha256(checkpoint_path),
        "manifest_sha256": sha256(args.manifest),
        "source_sha256": sha256(Path(__file__)),
        "fusion_source_sha256": sha256(ROOT / "src/models/cross_modal_transformer.py"),
        "device": args.device, "torch_version": str(torch.__version__),
        "test_model_inference": False,
        "historical_encoder_revisions_known": False,
        "validation_ids": [r["id"] for r in rows],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "protocol.json"
    if config_path.exists() and json.loads(config_path.read_text()) != protocol:
        raise ValueError("Replay protocol changed; choose a new output directory")
    write_json(config_path, protocol)
    prediction_path = args.output / "predictions.jsonl"
    predictions = []
    if prediction_path.exists():
        predictions = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    if [p["id"] for p in predictions] != [r["id"] for r in rows[:len(predictions)]]:
        raise ValueError("Replay cache is not an exact prefix of validation IDs")
    if len(predictions) > len(rows):
        raise ValueError("Replay cache has extra records")
    for row in rows:
        if sha256(row["audio_path"]) != row["audio_sha256"]:
            raise ValueError(f"Audio changed since audit: {row['id']}")
    for prior, row in zip(predictions, rows):
        if prior["label"] != row["sentiment"] or prior["audio_sha256"] != row["audio_sha256"]:
            raise ValueError("Replay cache source/label mismatch")
        class_scores([prior["probabilities"]])

    started = time.perf_counter()
    if len(predictions) < len(rows):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = CrossModalTransformerFusionClassifier(
            num_labels=7, fusion_head_output_type=checkpoint["fusion_head_output_type"],
            freeze_text_encoder=True, freeze_audio_encoder=True, local_files_only=True,
            **{k: cfg[k] for k in ("text_model_name", "text_revision", "audio_model_name", "audio_revision")},
        ).to(args.device).eval()
        load_legacy_head(model, checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(cfg["text_model_name"], revision=cfg["text_revision"], local_files_only=True)
        with prediction_path.open("a") as handle, torch.inference_mode():
            for i in range(len(predictions), len(rows)):
                row = rows[i]
                tokens = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=cfg["max_tokens"])
                text = model.encode_text(tokens["input_ids"].to(args.device), tokens["attention_mask"].to(args.device))
                waveform = load_audio(row["audio_path"], cfg["max_audio_sec"])[None].to(args.device)
                audio = model.encode_audio(waveform)
                # Match the precision of the historical saved embedding shards.
                probabilities = model.forward_from_embeddings(text.half().float(), audio.half().float()).logits.softmax(-1)[0].cpu().tolist()
                prediction = {
                    "id": row["id"], "label": row["sentiment"], "duration_sec": row["duration_sec"],
                    "audio_sha256": row["audio_sha256"], "probabilities": probabilities,
                }
                handle.write(json.dumps(prediction, allow_nan=False) + "\n")
                handle.flush()
                predictions.append(prediction)
                if (i + 1) % 100 == 0 or i + 1 == len(rows):
                    print(f"Replayed {i + 1}/{len(rows)} validation clips ({time.perf_counter() - started:.1f}s)", flush=True)

    truth = np.array([p["label"] for p in predictions])
    scores = class_scores([p["probabilities"] for p in predictions])
    historical = json.loads(checkpoint_path.with_name("mosei_cross_modal_metrics.json").read_text())["valid"]
    results = {name: mosei_metrics(truth, values) for name, values in scores.items()}
    strata = {}
    durations = np.array([p["duration_sec"] for p in predictions])
    for name, mask in (("audio_complete_at_8s", durations <= 8), ("audio_truncated_at_8s", durations > 8)):
        strata[name] = {key: mosei_metrics(truth[mask], values[mask]) for key, values in scores.items() if mask.any()}
    write_json(args.output / "summary.json", {
        "finished": True, "test_model_inference": False, "count": len(rows),
        "projections": results, "duration_strata": strata,
        "historical_validation": historical,
        "argmax_delta_from_historical": {k: results["argmax"][k] - v for k, v in historical.items() if k != "n"},
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "predictions_sha256": sha256(prediction_path),
        "this_invocation_elapsed_sec": time.perf_counter() - started,
        "interpretation": "Full validation audit, not a new test result. Expected-score projection is a diagnostic of the existing CE-trained model, not a retrained regression model. The legacy model used the unfiltered mirror training set.",
    })
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
