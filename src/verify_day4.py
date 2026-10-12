"""Verify the full development baseline, visual feature store and alignment evidence."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader
from src.emotion_checkpoint import EMOTION_LABEL_MAP, read_checkpoint
from src.emotion_fusion_infer import CrossModalEmotionPredictor
from src.iemocap_embedding_cache import (
    CachedDataset,
    collate_embeddings,
    sha256,
    write_json,
)
from src.prepare_iemocap import DEFAULT_METADATA
from src.train_cross_modal_finetune_iemocap import load_samples
from src.train_mvp_baseline import evaluate
from src.extract_iemocap_visual import coverage_report, extract_dialog
from src.visual_feature_dataset import VisualFeatureDataset, collate_visual
from src.visual_features import FaceImageEmbedder, DETECTOR_PATH, speaker_side


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_asr(directory, validation, source, checkpoint_path, metadata_path):
    report = json.loads((directory / "metrics.json").read_text())
    rows = [
        json.loads(line)
        for line in (directory / "predictions.jsonl").read_text().splitlines()
    ]
    transcripts = {
        r["utterance_id"]: r
        for r in map(
            json.loads, (directory / "transcripts.jsonl").read_text().splitlines()
        )
    }
    require(
        not report["test_evaluated"]
        and report["checkpoint_sha256"] == sha256(checkpoint_path)
        and report["transcripts_sha256"] == sha256(directory / "transcripts.jsonl")
        and report["config"]["metadata_sha256"] == sha256(metadata_path),
        "ASR diagnostic provenance mismatch",
    )
    expected = [s.utterance_id for s in validation]
    require(
        [r["utterance_id"] for r in rows] == expected
        and set(transcripts) == set(expected),
        "ASR diagnostic must cover every validation utterance exactly once",
    )
    for row in rows:
        original = source[row["utterance_id"]]
        transcript = transcripts[row["utterance_id"]]
        require(
            row["label"] == EMOTION_LABEL_MAP[original["label_code"]]
            and row["duration_sec"] == original["duration_sec"]
            and row["asr_text"] == transcript["text"]
            and transcript["audio_sha256"] == sha256(original["audio_path"]),
            "ASR prediction/source mismatch",
        )
        if row["asr_text"]:
            values = np.array(list(row["probabilities"].values()))
            require(
                values.shape == (4,)
                and np.isfinite(values).all()
                and np.all(values >= 0)
                and abs(values.sum() - 1) < 1e-5
                and row["prediction"] == int(values.argmax()),
                "Invalid ASR diagnostic probabilities",
            )
        else:
            require(
                row["prediction"] == -1 and row["probabilities"] is None,
                "An empty transcript must be an abstention",
            )

    def verify_metrics(subset, metrics):
        counts = np.zeros((4, 5), dtype=int)
        for row in subset:
            counts[
                row["label"], row["prediction"] if row["prediction"] >= 0 else 4
            ] += 1
        correct = np.diag(counts[:, :4])
        support = counts.sum(1)
        predicted = counts[:, :4].sum(0)
        f1 = np.divide(
            2 * correct,
            support + predicted,
            out=np.zeros(4, float),
            where=(support + predicted) > 0,
        ).mean()
        uar = np.divide(
            correct, support, out=np.zeros(4, float), where=support > 0
        ).mean()
        require(
            metrics["count"] == len(subset)
            and metrics["abstentions"] == int(counts[:, 4].sum())
            and metrics["confusion_with_abstention_column"] == counts.tolist(),
            "ASR diagnostic counts differ from predictions",
        )
        for key, value in (
            ("accuracy", correct.sum() / len(subset)),
            ("macro_f1", f1),
            ("uar", uar),
        ):
            require(abs(metrics[key] - value) < 1e-10, f"ASR {key} does not reproduce")

    verify_metrics(rows, report["metrics"])
    for name, truncated in (
        ("at_most_six_seconds", False),
        ("longer_than_six_seconds", True),
    ):
        verify_metrics(
            [r for r in rows if (r["duration_sec"] > 6) == truncated],
            report["by_audio_truncation"][name],
        )
    return {
        "metrics": report["metrics"],
        "by_audio_truncation": report["by_audio_truncation"],
    }


def verify_visual_records(directory, source):
    report = json.loads((directory / "coverage_report.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    require(
        config["schema_version"] == 2,
        "Expected visual schema v2 with separate feature masks",
    )
    require(
        config["face_detector"]["sha256"] == sha256(DETECTOR_PATH),
        "Face detector weights changed",
    )
    rows = [
        json.loads(line)
        for line in (directory / "metadata.jsonl").read_text().splitlines()
    ]
    expected = {
        uid
        for uid, r in source.items()
        if r["session"] != 5 and r["label_code"] in EMOTION_LABEL_MAP
    }
    require(
        {r["utterance_id"] for r in rows} == expected and len(rows) == len(expected),
        "Visual coverage does not match the full development sample set",
    )
    require(
        report["complete_development_set"] and not report["session_5_processed"],
        "Visual store is incomplete or includes Session 5",
    )
    all_lags = []
    for row in rows:
        original = source[row["utterance_id"]]
        require(
            row["session"] == original["session"]
            and row["label"] == EMOTION_LABEL_MAP[original["label_code"]],
            "Visual identity/label mismatch",
        )
        require(
            row["side"] == speaker_side(original["dialog_id"], original["speaker"]),
            "Wrong speaker panel",
        )
        path = directory / row["feature_path"]
        require(
            sha256(path) == row["feature_sha256"], "Visual feature checksum mismatch"
        )
        with np.load(path, allow_pickle=False) as data:
            count = row["frame_count"]
            valid = data["valid"]
            decoded = data["decoded"]
            require(
                data["blendshapes"].shape == (count, 52)
                and data["image_embeddings"].shape == (count, 576),
                "Visual feature shape mismatch",
            )
            require(
                valid.dtype == np.bool_ and decoded.dtype == np.bool_,
                "Visual masks must be boolean",
            )
            require(
                np.array_equal(valid, data["status"] == 0) and np.all(decoded[valid]),
                "Visual status/mask mismatch",
            )
            require(
                np.array_equal(valid, data["image_valid"] & data["blendshape_valid"])
                and np.all(decoded[data["image_valid"]]),
                "Image/blendshape availability mismatch",
            )
            for field, mask in (
                ("blendshapes", data["blendshape_valid"]),
                ("image_embeddings", data["image_valid"]),
            ):
                require(
                    np.isfinite(data[field]).all() and not data[field][~mask].any(),
                    "Invalid or unmasked missing visual features",
                )
            times = data["timestamps_sec"][decoded]
            require(np.all(np.diff(times) > 0), "Visual timestamps not increasing")
            require(
                np.all(times >= original["dialog_start_sec"])
                and np.all(times < row["window_end_sec"]),
                "Visual frame outside audio window",
            )
            require(
                np.array_equal(data["frame_indices"] >= 0, decoded),
                "Frame index/mask mismatch",
            )
            lags = times - data["requested_sec"][decoded]
            require(
                np.all(lags >= -1e-8) and np.all(lags <= 0.05),
                "Visual sampling timing mismatch",
            )
            all_lags.extend(lags.tolist())
            require(
                int(valid.sum()) == row["valid_frames"]
                and int(data["image_valid"].sum()) == row["image_valid_frames"]
                and int(decoded.sum()) == row["decoded_frames"],
                "Visual coverage counts differ from arrays",
            )
    calculated = coverage_report(rows)
    require(
        calculated["overall"] == report["overall"]
        and calculated["by_partition_label"] == report["by_partition_label"],
        "Coverage report does not reproduce",
    )
    # Exercise the downstream sequence loader, including distinct missing/padding masks.
    for partition in ("train", "validation"):
        dataset = VisualFeatureDataset(directory, partition, verify_checksums=False)
        batch = collate_visual([dataset[0], dataset[-1]])
        require(
            bool((batch["visual_mask"] <= batch["sequence_mask"]).all()),
            "Invalid visual padding mask",
        )
    return {
        "count": len(rows),
        **calculated,
        "max_sampling_delay_sec": float(max(all_lags)),
        "config": config,
    }


def verify(args):
    torch.set_num_threads(4)
    original = [json.loads(line) for line in args.metadata.read_text().splitlines()]
    source = {r["utterance_id"]: r for r in original}
    samples = load_samples(str(args.metadata), str(ROOT))
    train = [s for s in samples if s.session in (1, 2, 3)]
    validation = [s for s in samples if s.session == 4]
    checkpoint = read_checkpoint(args.checkpoint, "iemocap_emotion_4")
    provenance = checkpoint["provenance"]
    metrics = json.loads(args.checkpoint.with_suffix(".metrics.json").read_text())
    require(
        metrics["finished"] and not metrics["test_evaluated"],
        "Training incomplete or Session 5 evaluated",
    )
    require(
        provenance["train_utterance_ids"] == [s.utterance_id for s in train]
        and provenance["validation_utterance_ids"]
        == [s.utterance_id for s in validation],
        "Baseline does not use every development utterance with the correct split",
    )
    require(
        provenance["metadata_sha256"] == sha256(args.metadata),
        "Baseline data hash mismatch",
    )
    require(
        provenance["cache_index_sha256"] == sha256(args.cache / "index.json"),
        "Encoder cache index changed",
    )
    require(
        metrics["checkpoint_sha256"] == sha256(args.checkpoint),
        "Checkpoint hash mismatch",
    )
    history = metrics["history"]
    require(
        len(history) == provenance["config"]["epochs"], "Training budget incomplete"
    )
    best = max(history, key=lambda r: r["validation"]["macro_f1"])
    require(
        best["epoch"] == provenance["epoch"] == metrics["best_epoch"],
        "Best checkpoint selection mismatch",
    )
    predictor = CrossModalEmotionPredictor(
        args.checkpoint, args.device, local_files_only=True
    )
    dataset = CachedDataset(validation, args.cache)
    loader = DataLoader(
        dataset,
        batch_size=provenance["config"]["batch_size"],
        collate_fn=collate_embeddings,
    )
    reproduced, probabilities = evaluate(predictor.model, loader, args.device)
    require(
        reproduced == metrics["validation"],
        "Full validation metrics do not reproduce from the saved checkpoint",
    )
    saved = [
        json.loads(line)
        for line in args.checkpoint.with_suffix(".validation.jsonl")
        .read_text()
        .splitlines()
    ]
    require(
        [r["utterance_id"] for r in saved] == [s.utterance_id for s in validation],
        "Validation prediction order mismatch",
    )
    require(
        [r["label"] for r in saved] == [s.label for s in validation],
        "Validation labels mismatch",
    )
    require(
        np.allclose(
            [r["probabilities"] for r in saved], probabilities, atol=2e-5, rtol=2e-5
        ),
        "Saved validation probabilities differ from checkpoint",
    )
    # Check raw-waveform/tokenizer inference against cached inference on short,
    # long and all-class examples. This uses the public live predictor.
    selected = set()
    for label in range(4):
        candidates = [i for i, s in enumerate(validation) if s.label == label]
        selected.update(
            [
                candidates[0],
                candidates[-1],
                min(
                    candidates,
                    key=lambda i: source[validation[i].utterance_id]["duration_sec"],
                ),
                max(
                    candidates,
                    key=lambda i: source[validation[i].utterance_id]["duration_sec"],
                ),
            ]
        )
    differences = []
    for index in sorted(selected):
        sample = validation[index]
        audio, sr = sf.read(sample.audio_path, dtype="float32")
        raw = predictor.predict(sample.text, audio, sr)
        difference = float(
            np.max(
                np.abs(
                    np.array(list(raw.probabilities.values())) - probabilities[index]
                )
            )
        )
        require(difference < 2e-5, "Raw/cached baseline inference mismatch")
        differences.append(
            {
                "utterance_id": sample.utterance_id,
                "max_probability_difference": difference,
            }
        )
    asr = verify_asr(args.asr, validation, source, args.checkpoint, args.metadata)
    visual = verify_visual_records(args.features, source)
    require(
        visual["config"]["metadata_sha256"] == sha256(args.metadata),
        "Visual source manifest mismatch",
    )
    require(
        visual["config"]["max_seconds"] == predictor.preprocessing.max_audio_sec,
        "Audio/visual window mismatch",
    )
    alignment = json.loads(args.alignment.read_text())
    expected_dialogs = {
        r["dialog_id"]
        for r in original
        if r["session"] != 5 and r["emotion4_label"] is not None
    }
    require(
        alignment["passed"]
        and {r["dialog_id"] for r in alignment["dialogs"]} == expected_dialogs,
        "Audio/video timing audit incomplete",
    )
    inspection = json.loads(args.inspection.read_text())
    require(
        inspection["review_status"] == "inspected"
        and inspection["source_metadata_sha256"] == sha256(args.metadata)
        and inspection["feature_metadata_sha256"]
        == sha256(args.features / "metadata.jsonl"),
        "Current visual contact sheets have not been inspected",
    )
    artifacts = inspection.get("reviewed_artifact_sha256", {})
    require(bool(artifacts), "Visual review is missing artifact fingerprints")
    for path, digest in artifacts.items():
        require(sha256(path) == digest, "A reviewed visual artifact changed")
    inspected_sides = {
        (
            source[r["utterance_id"]]["session"],
            source[r["utterance_id"]]["dialog_id"][5],
            r["side"],
        )
        for r in inspection["samples"]
    }
    require(
        inspected_sides
        == {
            (s, marker, side)
            for s in (1, 2, 3, 4)
            for marker in ("F", "M")
            for side in ("left", "right")
        },
        "Inspect both speaker panels and both marker participants in every development session",
    )
    # Re-extract a real dialog with fresh model instances to validate the cache.
    dialog = sorted(expected_dialogs)[0]
    rows = [
        r
        for r in original
        if r["dialog_id"] == dialog and r["emotion4_label"] is not None
    ]
    embedder = FaceImageEmbedder(visual["config"]["image_encoder"]["device"])
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="mct_visual_recheck_") as temporary:
        replay = extract_dialog(
            rows,
            Path(temporary),
            visual["config"],
            embedder,
            ROOT / "src/weights/face_landmarker.task",
        )
        for row in replay["utterances"]:
            with (
                np.load(Path(temporary) / row["feature_path"]) as new,
                np.load(args.features / row["feature_path"]) as old,
            ):
                for key in new.files:
                    require(
                        np.allclose(
                            new[key], old[key], atol=2e-5, rtol=2e-5, equal_nan=True
                        ),
                        f"Visual replay differs: {row['utterance_id']} {key}",
                    )
    report = {
        "passed": True,
        "scope": "days_3_4_development_baseline_and_visual_features",
        "source_metadata_sha256": sha256(args.metadata),
        "checkpoint_sha256": sha256(args.checkpoint),
        "visual_feature_metadata_sha256": sha256(args.features / "metadata.jsonl"),
        "visual_config_sha256": sha256(args.features / "config.json"),
        "alignment_audit_sha256": sha256(args.alignment),
        "inspection_manifest_sha256": sha256(args.inspection),
        "asr_metrics_sha256": sha256(args.asr / "metrics.json"),
        "train_count": len(train),
        "validation_count": len(validation),
        "session_5_evaluated": False,
        "best_epoch": metrics["best_epoch"],
        "validation": reproduced,
        "asr_diagnostic": asr,
        "majority_baseline": metrics["majority_baseline"],
        "raw_cached_parity": differences,
        "visual": {k: v for k, v in visual.items() if k != "config"},
        "visual_reextraction_dialog": dialog,
        "alignment_dialogs": alignment["dialog_count"],
        "alignment_samples": alignment["sample_count"],
        "alignment_max_abs_offset_sec": alignment["max_abs_offset_sec"],
        "visual_inspection_frames": len(inspection["samples"]),
        "not_claimed": [
            "Session 5 accuracy",
            "multimodal gain from vision",
            "FACS AU measurements",
            "trust prediction",
            "live microphone/webcam integration",
        ],
    }
    write_json(args.output, report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("raw_cached_parity", "visual")
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument(
        "--checkpoint", type=Path, default=ROOT / "models/mvp/audio_text_baseline.pt"
    )
    parser.add_argument(
        "--cache", type=Path, default=ROOT / "data/processed/iemocap_encoder_cache"
    )
    parser.add_argument(
        "--asr",
        type=Path,
        default=ROOT / "data/processed/mvp/asr_validation",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data/processed/iemocap_visual_features_v2",
    )
    parser.add_argument(
        "--alignment",
        type=Path,
        default=ROOT / "data/processed/mvp/day4_alignment/audio_video_audit.json",
    )
    parser.add_argument(
        "--inspection",
        type=Path,
        default=ROOT
        / "data/processed/mvp/day4_alignment/visual_review/inspection_manifest.json",
    )
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/mvp/day4_verification.json",
    )
    verify(parser.parse_args())
