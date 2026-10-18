"""Reproduce all visual comparisons and evaluate controlled missing-input behavior."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from src.emotion_checkpoint import EMOTION_LABELS
from src.evaluate_asr_baseline import diagnostic_metrics
from src.iemocap_embedding_cache import sha256, write_json
from src.models.visual_temporal import fit_normalization, fuse_probabilities
from src.train_mvp_baseline import metrics_from_predictions
from src.train_mvp_visual import visual_predictions, select_alpha
from src.visual_feature_dataset import VisualFeatureDataset
from src.visual_fusion import load_visual_checkpoint, VisualFusionPredictor


def require(condition, message):
    if not condition:
        raise ValueError(message)


def intervention(name):
    generator = torch.Generator().manual_seed(20261010)

    def apply(batch):
        batch = dict(batch)
        bm, im = batch["blendshape_mask"].clone(), batch["image_mask"].clone()
        if name in ("no_vision", "no_blendshapes"):
            bm.zero_()
        if name in ("no_vision", "no_image"):
            im.zero_()
        if name == "half_frames_missing":
            keep = torch.rand(bm.shape, generator=generator) >= 0.5
            bm &= keep
            im &= keep
        if name == "time_shuffled":
            for key in ("blendshapes", "image_embeddings"):
                batch[key] = batch[key].clone()
            for i, length in enumerate(batch["sequence_mask"].sum(1)):
                order = torch.randperm(int(length), generator=generator)
                for key in ("blendshapes", "image_embeddings"):
                    batch[key][i, :length] = batch[key][i, order]
                bm[i, :length] = bm[i, order]
                im[i, :length] = im[i, order]
        batch.update(blendshape_mask=bm, image_mask=im)
        return batch

    return apply


def verify(args):
    torch.set_num_threads(2)
    comparison = json.loads((args.models / "comparison.json").read_text())
    config = comparison["protocol"]["config"]
    require(
        comparison["finished"] and not comparison["test_evaluated"],
        "Incomplete comparison or test exposure",
    )
    require(
        sha256(args.base_checkpoint)
        == comparison["protocol"]["base_checkpoint_sha256"],
        "Baseline changed",
    )
    require(
        sha256(args.features / "metadata.jsonl")
        == comparison["protocol"]["visual_metadata_sha256"],
        "Features changed",
    )
    datasets = {
        p: VisualFeatureDataset(args.features, p) for p in ("train", "validation")
    }
    rows = {p: [d[i] for i in range(len(d))] for p, d in datasets.items()}
    for p, key in (
        ("train", "train_utterance_ids"),
        ("validation", "validation_utterance_ids"),
    ):
        require(
            [r["utterance_id"] for r in rows[p]] == comparison["protocol"][key],
            "Comparison sample set changed",
        )
    statistics = fit_normalization(rows["train"], config["normalization_std_floor"])
    validation = rows["validation"]
    labels = [r["label"] for r in validation]
    base_cache = json.loads(args.base_predictions.read_text())
    require(
        sha256(args.base_predictions)
        == comparison["protocol"]["base_predictions_sha256"],
        "Base predictions changed",
    )
    base_by_id = {r["utterance_id"]: r for r in base_cache["predictions"]}
    base = torch.tensor(
        [base_by_id[r["utterance_id"]]["probabilities"] for r in validation]
    )
    require(
        metrics_from_predictions(labels, base.numpy()) == comparison["baseline"],
        "Baseline metrics differ",
    )
    common = torch.tensor(
        [bool(r["image_mask"].any() and r["blendshape_mask"].any()) for r in validation]
    )
    verified, group = [], {}
    for mode in config["modes"]:
        group[mode] = []
        for seed in config["seeds"]:
            path = args.models / f"{mode}_seed{seed}.pt"
            report = json.loads(path.with_suffix(".metrics.json").read_text())
            require(
                report["finished"] and len(report["history"]) == config["epochs"],
                "Training budget incomplete",
            )
            require(
                sha256(path) == report["checkpoint_sha256"], "Visual checkpoint changed"
            )
            model, payload = load_visual_checkpoint(path, args.base_checkpoint)
            require(
                payload["provenance"]["mode"] == mode
                and payload["provenance"]["seed"] == seed,
                "Run identity differs",
            )
            require(
                payload["provenance"]["normalization_partition"] == "train",
                "Normalization partition differs",
            )
            for field in statistics:
                for kind in ("mean", "std"):
                    require(
                        torch.equal(
                            getattr(model, field + "_" + kind), statistics[field][kind]
                        ),
                        "Normalization does not reproduce from training only",
                    )
            best = max(
                report["history"],
                key=lambda r: (r["validation"]["macro_f1"], -r["alpha"], -r["epoch"]),
            )
            require(
                best["epoch"] == report["best_epoch"] == payload["provenance"]["epoch"]
                and best["alpha"] == payload["alpha"],
                "Epoch/weight selection differs",
            )
            visual, available = visual_predictions(model, validation)
            alpha, metrics, mixed = select_alpha(
                labels, base, visual, available, config["fusion_alphas"]
            )
            require(
                alpha == payload["alpha"] and metrics == report["validation"],
                "Full validation does not reproduce",
            )
            saved = [
                json.loads(line)
                for line in path.with_suffix(".validation.jsonl")
                .read_text()
                .splitlines()
            ]
            require(
                [r["utterance_id"] for r in saved]
                == [r["utterance_id"] for r in validation],
                "Saved prediction identities differ",
            )
            require(
                np.allclose(
                    [r["probabilities"] for r in saved], mixed.numpy(), atol=1e-7
                ),
                "Saved predictions differ",
            )
            no_visual, none_available = visual_predictions(
                model, validation, intervention=intervention("no_vision")
            )
            fallback = fuse_probabilities(base, no_visual, none_available, alpha)
            require(
                torch.equal(fallback, base) and not none_available.any(),
                "Missing vision does not return exact baseline",
            )
            common_metrics = metrics_from_predictions(
                np.asarray(labels)[common], mixed[common].numpy()
            )
            verified.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "macro_f1": metrics["macro_f1"],
                    "epoch": best["epoch"],
                    "alpha": alpha,
                    "common_valid": common_metrics,
                    "exact_fallback": True,
                }
            )
            group[mode].append(metrics["macro_f1"])
    for mode, values in group.items():
        require(
            abs(
                np.mean(values) - comparison["by_representation"][mode]["mean_macro_f1"]
            )
            < 1e-12,
            "Representation mean differs",
        )
    selected_mode = max(config["modes"], key=lambda m: np.mean(group[m]))
    candidates = [r for r in verified if r["mode"] == selected_mode]
    chosen = min(
        candidates,
        key=lambda r: (abs(r["macro_f1"] - np.mean(group[selected_mode])), r["seed"]),
    )
    require(
        (selected_mode, chosen["seed"])
        == (comparison["selected_mode"], comparison["selected_seed"]),
        "Demo selection rule differs",
    )
    selected_path = args.models / "selected.pt"
    require(
        sha256(selected_path)
        == sha256(args.models / f"{selected_mode}_seed{chosen['seed']}.pt"),
        "Selected model differs from chosen run",
    )
    model, payload = load_visual_checkpoint(selected_path, args.base_checkpoint)
    robustness = {}
    for name in (
        "no_vision",
        "no_image",
        "no_blendshapes",
        "half_frames_missing",
        "time_shuffled",
    ):
        visual, available = visual_predictions(
            model, validation, intervention=intervention(name)
        )
        mixed = fuse_probabilities(base, visual, available, payload["alpha"])
        robustness[name] = {
            "metrics": metrics_from_predictions(labels, mixed.numpy()),
            "fallback_utterances": int((~available).sum()),
        }
    visual, available = visual_predictions(model, validation)
    runtime = VisualFusionPredictor(selected_path, args.base_checkpoint)
    parity = []
    expected = fuse_probabilities(base, visual, available, payload["alpha"])
    for index in sorted({0, len(validation) - 1, *[labels.index(i) for i in range(4)]}):
        prediction = runtime.predict(
            dict(zip(EMOTION_LABELS, base[index].tolist())), validation[index]
        )
        difference = float(
            np.max(
                np.abs(
                    list(prediction["probabilities"].values()) - expected[index].numpy()
                )
            )
        )
        require(difference < 1e-6, "Public visual predictor/batch parity differs")
        parity.append(
            {
                "utterance_id": validation[index]["utterance_id"],
                "max_probability_difference": difference,
            }
        )
    asr_path = args.asr / "predictions.jsonl"
    asr = {
        r["utterance_id"]: r for r in map(json.loads, asr_path.read_text().splitlines())
    }
    asr_report = json.loads((args.asr / "metrics.json").read_text())
    require(
        asr_report["checkpoint_sha256"] == sha256(args.base_checkpoint),
        "ASR baseline provenance differs",
    )
    asr_rows = []
    for i, row in enumerate(validation):
        before = asr[row["utterance_id"]]
        prediction, probabilities = -1, None
        if before["probabilities"] is not None:
            asr_base = torch.tensor(
                [[before["probabilities"][name] for name in EMOTION_LABELS]]
            )
            fused = fuse_probabilities(
                asr_base, visual[i : i + 1], available[i : i + 1], payload["alpha"]
            )[0]
            prediction, probabilities = int(fused.argmax()), fused.tolist()
        asr_rows.append(
            {
                "utterance_id": row["utterance_id"],
                "label": row["label"],
                "prediction": prediction,
                "probabilities": probabilities,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_name("selected_asr_predictions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in asr_rows)
    )
    result = {
        "passed": True,
        "test_evaluated": False,
        "training_examples": len(rows["train"]),
        "validation_examples": len(validation),
        "comparison_sha256": sha256(args.models / "comparison.json"),
        "selected_checkpoint_sha256": sha256(selected_path),
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "visual_metadata_sha256": sha256(args.features / "metadata.jsonl"),
        "runs_verified": verified,
        "common_valid_count": int(common.sum()),
        "baseline_common_valid": metrics_from_predictions(
            np.asarray(labels)[common], base[common].numpy()
        ),
        "selected_robustness": robustness,
        "public_predictor_parity": parity,
        "asr_diagnostic": {
            "baseline": asr_report["metrics"],
            "selected": diagnostic_metrics(asr_rows),
            "asr_predictions_sha256": sha256(asr_path),
        },
        "interpretation": "Missing-input interventions and time shuffling are validation diagnostics, not independent evidence. ASR retains four empty-transcript abstentions. All configurations and comparisons exclude Session 5.",
    }
    write_json(args.output, result)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in (
                    "runs_verified",
                    "selected_robustness",
                    "public_predictor_parity",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=ROOT / "models/mvp/visual_day5")
    parser.add_argument(
        "--features",
        type=Path,
        default=ROOT / "data/processed/iemocap_visual_features_v2",
    )
    parser.add_argument(
        "--base-checkpoint",
        type=Path,
        default=ROOT / "models/mvp/audio_text_baseline.pt",
    )
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=ROOT / "data/processed/mvp/day5_base_predictions.json",
    )
    parser.add_argument(
        "--asr", type=Path, default=ROOT / "data/processed/mvp/asr_validation"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/processed/mvp/day5_7/visual_verification.json",
    )
    verify(parser.parse_args())
