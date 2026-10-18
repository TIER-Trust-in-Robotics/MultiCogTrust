"""Verify visual deployment selection, uncached parity and recorded runtime evidence.

Run the README replay and live capture commands first. No Session 5 inference,
camera access or silent checkpoint changes are performed by this verifier.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from src.emotion_checkpoint import EMOTION_LABELS
from src.evaluate_asr_baseline import diagnostic_metrics
from src.iemocap_embedding_cache import sha256, write_json
from src.models.visual_temporal import fuse_probabilities
from src.train_mvp_visual import visual_predictions
from src.visual_feature_dataset import VisualFeatureDataset
from src.visual_features import speaker_side
from src.visual_fusion import load_visual_checkpoint, VisualFusionPredictor
from src.visual_stream import VisualExtractor, VisualBuffer, extract_interval


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(args):
    torch.set_num_threads(2)
    folder = ROOT / "data/processed/mvp/day5_7"
    models = ROOT / "models/mvp/visual_day5"
    base_path = ROOT / "models/mvp/audio_text_baseline.pt"
    training = json.loads((folder / "visual_verification.json").read_text())
    require(
        training["passed"] and not training["test_evaluated"],
        "Visual training gate missing",
    )
    require(
        training["comparison_sha256"] == sha256(models / "comparison.json"),
        "Comparison changed",
    )
    model, reference = load_visual_checkpoint(models / "selected.pt", base_path)
    deployed, deployment = load_visual_checkpoint(models / "deployment.pt", base_path)
    require(
        all(
            torch.equal(v, deployed.state_dict()[k])
            for k, v in model.state_dict().items()
        ),
        "ASR deployment altered the reference-selected model weights",
    )
    selection = json.loads((models / "deployment.selection.json").read_text())
    require(
        selection["checkpoint_sha256"] == sha256(models / "deployment.pt"),
        "Deployment checksum differs",
    )
    require(
        selection["reference_selected_checkpoint_sha256"]
        == sha256(models / "selected.pt"),
        "Reference changed",
    )
    require(
        selection["config_sha256"] == sha256(ROOT / "configs/mvp_day7_deployment.json"),
        "Selection config changed",
    )
    dataset = VisualFeatureDataset(
        ROOT / "data/processed/iemocap_visual_features_v2", "validation"
    )
    rows = [dataset[i] for i in range(len(dataset))]
    visual, available = visual_predictions(model, rows)
    asr_path = ROOT / "data/processed/mvp/asr_validation/predictions.jsonl"
    require(
        selection["asr_predictions_sha256"] == sha256(asr_path), "ASR evidence changed"
    )
    asr = {
        r["utterance_id"]: r for r in map(json.loads, asr_path.read_text().splitlines())
    }
    base = torch.tensor(
        [
            [asr[r["utterance_id"]]["probabilities"][n] for n in EMOTION_LABELS]
            if asr[r["utterance_id"]]["probabilities"]
            else [0.25] * 4
            for r in rows
        ]
    )
    candidates = []
    for alpha in selection["config"]["alphas"]:
        probabilities = fuse_probabilities(base, visual, available, alpha)
        metrics = diagnostic_metrics(
            [
                {
                    "label": r["label"],
                    "prediction": int(probabilities[i].argmax())
                    if asr[r["utterance_id"]]["probabilities"]
                    else -1,
                }
                for i, r in enumerate(rows)
            ]
        )
        candidates.append({"alpha": alpha, "metrics": metrics})
    require(candidates == selection["candidates"], "ASR selection does not reproduce")
    chosen = max(candidates, key=lambda c: (c["metrics"]["macro_f1"], -c["alpha"]))
    require(
        chosen == selection["selected"] and chosen["alpha"] == deployment["alpha"],
        "Deployment selection differs",
    )
    require(
        torch.equal(
            fuse_probabilities(base, visual, torch.zeros_like(available), 0.25), base
        ),
        "Missing vision changes the base prediction",
    )

    # Four explicit short development clips, covering all classes and both speakers.
    ids = [
        "Ses04F_impro01_F004",
        "Ses04F_impro03_F012",
        "Ses04F_impro02_F000",
        "Ses04F_impro03_M000",
    ]
    raw_by_id = {
        r["utterance_id"]: r
        for r in map(
            json.loads,
            (ROOT / "data/processed/iemocap_manifest/metadata.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    index_by_id = {r["utterance_id"]: i for i, r in enumerate(rows)}
    predictor = VisualFusionPredictor(models / "selected.pt", base_path)
    parity = []
    for uid in ids:
        source, index = raw_by_id[uid], index_by_id[uid]
        require(source["session"] != 5, "Test session must remain reserved")
        extractor = VisualExtractor(reference["feature_config"], args.device)
        try:
            raw, evidence = extract_interval(
                extractor,
                source["video_path"],
                speaker_side(source["dialog_id"], source["speaker"]),
                source["dialog_start_sec"],
                source["dialog_end_sec"],
            )
        finally:
            extractor.close()
        cached = rows[index]
        for key in ("image_mask", "blendshape_mask"):
            require(
                torch.equal(raw[key], cached[key]), f"Raw mask differs: {uid}/{key}"
            )
        differences = {}
        for key in ("image_embeddings", "blendshapes", "times_sec"):
            differences[key] = float((raw[key] - cached[key]).abs().max())
            require(
                differences[key] < 1e-4,
                f"Raw features differ: {uid}/{key}: {differences[key]}",
            )
        probabilities = dict(zip(EMOTION_LABELS, base[index].tolist()))
        raw_prediction, cached_prediction = (
            predictor.predict(probabilities, raw),
            predictor.predict(probabilities, cached),
        )
        delta = max(
            abs(
                raw_prediction["probabilities"][n]
                - cached_prediction["probabilities"][n]
            )
            for n in EMOTION_LABELS
        )
        require(delta < 1e-6, f"Raw/cached fusion differs: {uid}")
        missing, _ = VisualBuffer().select(0, 2)
        fallback = predictor.predict(probabilities, missing)
        require(
            fallback["probabilities"] == probabilities
            and fallback["route"] == "audio_text_missing_vision",
            "Public fallback differs",
        )
        parity.append(
            {
                "utterance_id": uid,
                "feature_max_difference": differences,
                "probability_max_difference": delta,
                "observed_frames": evidence["observed_frames"],
            }
        )

    runtime = {}
    for name in ("dialogue_replay_final", "live_capture_final", "fallback_replay"):
        path = folder / (name + ".jsonl")
        events = list(map(json.loads, path.read_text().splitlines()))
        config = events[0]
        require(config["event"] == "run_config", f"Configuration missing: {name}")
        require(
            config["base_checkpoint_sha256"] == sha256(base_path),
            f"Runtime base differs: {name}",
        )
        for filename, checksum in config["runtime_sha256"].items():
            require(
                sha256(ROOT / "src" / filename) == checksum,
                f"Runtime changed after capture: {filename}",
            )
        predictions = [e for e in events if e.get("event") == "prediction"]
        report = events[-1]
        require(report["event"] == "stream_summary", f"Shutdown did not finish: {name}")
        require(
            report["predictions"] == len(predictions),
            f"Prediction count differs: {name}",
        )
        for key, cap in (
            ("audio", 32),
            ("frame", 2),
            ("segment", 4),
            ("visual_history", 450),
        ):
            require(
                report.get(
                    key + "_queue_high_watermark",
                    report.get(key + "_high_watermark", 0),
                )
                <= cap,
                f"Unbounded queue: {name}/{key}",
            )
        require(report["audio_chunks"] > 30, f"No useful audio capture: {name}")
        if name == "dialogue_replay_final":
            require(
                len(predictions) >= 3 and report["video_frames"] >= 100,
                "Insufficient synchronized replay",
            )
            require(
                config["replay_source"]["audio_channel_match"]["correlation"] >= 0.8,
                "Speaker channel unverified",
            )
            require(
                sum(
                    report[k]
                    for k in (
                        "dropped_audio_chunks",
                        "dropped_video_frames",
                        "dropped_segments",
                        "stale_segments",
                    )
                )
                == 0,
                "Replay dropped input or predictions",
            )
        if name == "live_capture_final":
            require(
                report["source"] == "live_devices"
                and report["camera_error"] is None
                and report["video_frames"] >= 50,
                "Live camera capture incomplete",
            )
            require(
                report["elapsed_sec"] < config["arguments"]["duration"] + 10,
                "Live shutdown exceeded drain allowance",
            )
            require(
                "adc_time" in report["audio_clock"], "Device timestamp mapping missing"
            )
        if name == "fallback_replay":
            require(
                predictions and report["video_frames"] == 0,
                "Fallback replay incomplete",
            )
        for p in predictions:
            require(p["source_clock"] == "monotonic_seconds", "Mixed time bases")
            for timestamp in p.get("visual", {}).get("source_timestamps", []):
                require(
                    p["start_time"]
                    <= timestamp
                    < min(p["end_time"], p["start_time"] + 6),
                    "Future visual information leaked",
                )
            if p["emotion"]:
                if name == "fallback_replay" or chosen["alpha"] == 0:
                    require(
                        p["emotion"]["probabilities"]
                        == p["audio_text_emotion"]["probabilities"],
                        "Expected exact audio/text output",
                    )
        runtime[name] = {"sha256": sha256(path), "summary": report}

    inspector = json.loads((folder / "inspector_verification.json").read_text())
    require(inspector["http_contract_passed"], "Inspector HTTP checks incomplete")
    result = {
        "engineering_passed": True,
        "test_evaluated": False,
        "visual_training_gate_sha256": sha256(folder / "visual_verification.json"),
        "deployment_selection_reproduced": True,
        "deployment_alpha": chosen["alpha"],
        "deployment_metrics": chosen["metrics"],
        "raw_cached_parity": parity,
        "runtime": runtime,
        "inspector": inspector,
        "limits": [
            "Browser appearance review is pending because computer-use access was denied.",
            "Short device capture and development replay do not establish live emotion accuracy or a long-session latency guarantee.",
            "Camera timestamps use frame receipt time; physical sensor offset remains uncalibrated.",
            "ASR validation selected the deployment weight and is not independent evaluation.",
        ],
    }
    write_json(folder / "day7_verification.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "engineering_passed",
                    "deployment_alpha",
                    "test_evaluated",
                    "limits",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    verify(parser.parse_args())
