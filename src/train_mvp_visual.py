"""Matched three-seed visual comparisons with a frozen audio/text reference."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from src.emotion_fusion_infer import CrossModalEmotionPredictor
from src.iemocap_embedding_cache import (
    CachedDataset,
    collate_embeddings,
    sha256,
    write_json,
)
from src.models.visual_temporal import (
    VisualTemporalClassifier,
    augment_visual,
    fit_normalization,
    fuse_probabilities,
)
from src.prepare_iemocap import DEFAULT_METADATA
from src.train_cross_modal_finetune_iemocap import load_samples
from src.train_mvp_baseline import evaluate as evaluate_base, metrics_from_predictions
from src.visual_feature_dataset import VisualFeatureDataset, collate_visual
from src.visual_fusion import save_visual_checkpoint, load_visual_checkpoint


def base_predictions(args):
    path = args.base_predictions
    contract = {
        "base_checkpoint_sha256": sha256(args.base_checkpoint),
        "metadata_sha256": sha256(args.metadata),
        "encoder_cache_index_sha256": sha256(args.encoder_cache / "index.json"),
    }
    samples = [
        s
        for s in load_samples(str(args.metadata), str(ROOT))
        if s.session in (1, 2, 3, 4)
    ]
    if path.exists():
        result = json.loads(path.read_text())
        if result["contract"] != contract or [
            r["utterance_id"] for r in result["predictions"]
        ] != [s.utterance_id for s in samples]:
            raise ValueError("Base prediction cache changed")
        return result
    predictor = CrossModalEmotionPredictor(
        args.base_checkpoint, args.base_device, local_files_only=True
    )
    loader = DataLoader(
        CachedDataset(samples, args.encoder_cache),
        batch_size=16,
        collate_fn=collate_embeddings,
    )
    _, probabilities = evaluate_base(predictor.model, loader, args.base_device)
    result = {
        "contract": contract,
        "predictions": [
            {
                "utterance_id": s.utterance_id,
                "session": s.session,
                "label": s.label,
                "probabilities": p,
            }
            for s, p in zip(samples, probabilities)
        ],
    }
    existing = {
        r["utterance_id"]: r["probabilities"]
        for r in map(
            json.loads,
            args.base_checkpoint.with_suffix(".validation.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    for row in result["predictions"]:
        if row["session"] == 4 and not np.allclose(
            row["probabilities"], existing[row["utterance_id"]], atol=2e-5, rtol=2e-5
        ):
            raise ValueError("Frozen baseline validation predictions changed")
    write_json(path, result)
    del predictor
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return result


@torch.inference_mode()
def visual_predictions(model, rows, batch_size=64, intervention=None):
    probabilities, available = [], []
    for batch in DataLoader(rows, batch_size=batch_size, collate_fn=collate_visual):
        if intervention:
            batch = intervention(batch)
        logits, valid = model(batch)
        probabilities.append(logits.softmax(-1))
        available.append(valid)
    return torch.cat(probabilities), torch.cat(available)


def select_alpha(labels, base, visual, available, alphas):
    candidates = []
    for alpha in alphas:
        mixed = fuse_probabilities(base, visual, available, alpha)
        metrics = metrics_from_predictions(labels, mixed.numpy())
        candidates.append((metrics["macro_f1"], -alpha, alpha, metrics, mixed))
    return max(candidates, key=lambda r: (r[0], r[1]))[2:]


def fit_one(
    mode,
    seed,
    rows,
    validation,
    base,
    stats,
    config,
    provenance,
    feature_config,
    output,
):
    path = output / f"{mode}_seed{seed}.pt"
    metrics_path = path.with_suffix(".metrics.json")
    identity = {**provenance, "mode": mode, "seed": seed}
    if metrics_path.exists():
        report = json.loads(metrics_path.read_text())
        if (
            report["identity"] != identity
            or not report["finished"]
            or report["checkpoint_sha256"] != sha256(path)
        ):
            raise ValueError(f"Completed run changed: {path}")
        return report
    if path.exists():
        raise FileExistsError(f"Preserving incomplete run {path}; use another --output")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = VisualTemporalClassifier(mode, config["hidden_dim"], config["dropout"])
    model.set_normalization(stats)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    loader = DataLoader(
        rows, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_visual
    )
    labels = [r["label"] for r in validation]
    best_score, best_alpha, history = -1.0, 1.0, []
    started = time.perf_counter()
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        loss_sum, available_count = 0.0, 0
        for batch in loader:
            batch = augment_visual(
                batch, mode, config["frame_dropout"], config["combined_stream_dropout"]
            )
            logits, available = model(batch)
            if not bool(available.any()):
                continue
            loss = torch.nn.functional.cross_entropy(
                logits[available], batch["labels"][available]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip_norm"]
            )
            optimizer.step()
            count = int(available.sum())
            loss_sum += float(loss.detach()) * count
            available_count += count
        model.eval()
        visual, available = visual_predictions(model, validation, config["batch_size"])
        alpha, metrics, _ = select_alpha(
            labels, base, visual, available, config["fusion_alphas"]
        )
        available_labels = np.asarray(labels)[available.numpy()]
        visual_metrics = metrics_from_predictions(
            available_labels, visual[available].numpy()
        )
        row = {
            "epoch": epoch,
            "train_visual_loss": loss_sum / max(available_count, 1),
            "augmented_available_training_examples": available_count,
            "alpha": alpha,
            "validation": metrics,
            "visual_only_available": visual_metrics,
        }
        history.append(row)
        if (metrics["macro_f1"], -alpha) > (best_score, -best_alpha):
            best_score, best_alpha = metrics["macro_f1"], alpha
            save_visual_checkpoint(
                path,
                model,
                alpha,
                {**identity, "epoch": epoch, "validation": metrics},
                feature_config,
            )
        write_json(
            path.with_suffix(".progress.json"),
            {"finished": False, "identity": identity, "history": history},
        )
        print(
            f"{mode} seed={seed} epoch={epoch:02d}: visual F1={visual_metrics['macro_f1']:.4f}, fused F1={metrics['macro_f1']:.4f}, alpha={alpha}",
            flush=True,
        )
    model, payload = load_visual_checkpoint(path)
    visual, available = visual_predictions(model, validation, config["batch_size"])
    mixed = fuse_probabilities(base, visual, available, payload["alpha"])
    metrics = metrics_from_predictions(labels, mixed.numpy())
    if metrics != payload["provenance"]["validation"]:
        raise ValueError("Selected visual checkpoint does not reproduce")
    report = {
        "finished": True,
        "identity": identity,
        "checkpoint": str(path),
        "checkpoint_sha256": sha256(path),
        "best_epoch": payload["provenance"]["epoch"],
        "alpha": payload["alpha"],
        "validation": metrics,
        "visual_only_available": metrics_from_predictions(
            np.asarray(labels)[available.numpy()], visual[available].numpy()
        ),
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "history": history,
        "elapsed_sec": time.perf_counter() - started,
    }
    write_json(metrics_path, report)
    predictions = [
        {
            "utterance_id": r["utterance_id"],
            "label": r["label"],
            "visual_available": bool(a),
            "base_probabilities": bp.tolist(),
            "visual_probabilities": vp.tolist(),
            "probabilities": fp.tolist(),
        }
        for r, a, bp, vp, fp in zip(validation, available, base, visual, mixed)
    ]
    path.with_suffix(".validation.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in predictions)
    )
    write_json(
        path.with_suffix(".progress.json"),
        {"finished": True, "identity": identity, "history": history},
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/mvp_day5_visual.json"
    )
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
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
        "--encoder-cache",
        type=Path,
        default=ROOT / "data/processed/iemocap_encoder_cache",
    )
    parser.add_argument(
        "--base-predictions",
        type=Path,
        default=ROOT / "data/processed/mvp/day5_base_predictions.json",
    )
    parser.add_argument("--base-device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--output", type=Path, default=ROOT / "models/mvp/visual_day5")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if (
        config["train_sessions"] != [1, 2, 3]
        or config["validation_session"] != 4
        or config["test_session"] != 5
        or config["test_evaluated"]
    ):
        raise ValueError("Development split is fixed; Session 5 must remain excluded")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    reference = base_predictions(args)
    base_by_id = {r["utterance_id"]: r for r in reference["predictions"]}
    datasets = {
        p: VisualFeatureDataset(args.features, p) for p in ("train", "validation")
    }
    feature_config = datasets["train"].config
    if (
        feature_config["sample_hz"] != config["sample_hz"]
        or feature_config["max_seconds"] != config["max_seconds"]
    ):
        raise ValueError("Feature sampling differs from the experiment protocol")
    rows = {
        p: [dataset[i] for i in range(len(dataset))] for p, dataset in datasets.items()
    }
    for partition, data in rows.items():
        expected_sessions = (1, 2, 3) if partition == "train" else (4,)
        expected = {
            uid for uid, r in base_by_id.items() if r["session"] in expected_sessions
        }
        if {r["utterance_id"] for r in data} != expected:
            raise ValueError("Audio/text and visual sample sets differ")
        for row in data:
            if row["label"] != base_by_id[row["utterance_id"]]["label"]:
                raise ValueError("Joined labels differ")
    validation = rows["validation"]
    base = torch.tensor(
        [base_by_id[r["utterance_id"]]["probabilities"] for r in validation],
        dtype=torch.float32,
    )
    base_metrics = metrics_from_predictions(
        [r["label"] for r in validation], base.numpy()
    )
    stats = fit_normalization(rows["train"], config["normalization_std_floor"])
    provenance = {
        **reference["contract"],
        "config": config,
        "config_sha256": sha256(args.config),
        "visual_metadata_sha256": sha256(args.features / "metadata.jsonl"),
        "visual_config_sha256": sha256(args.features / "config.json"),
        "base_predictions_sha256": sha256(args.base_predictions),
        "train_utterance_ids": [r["utterance_id"] for r in rows["train"]],
        "validation_utterance_ids": [r["utterance_id"] for r in validation],
        "normalization_partition": "train",
        "test_evaluated": False,
        "normalization_frame_counts": {k: v["frames"] for k, v in stats.items()},
    }
    runs = []
    for mode in config["modes"]:
        for seed in config["seeds"]:
            runs.append(
                fit_one(
                    mode,
                    seed,
                    rows["train"],
                    validation,
                    base,
                    stats,
                    config,
                    provenance,
                    feature_config,
                    args.output,
                )
            )
    summary = {}
    for mode in config["modes"]:
        group = [r for r in runs if r["identity"]["mode"] == mode]
        values = np.array([r["validation"]["macro_f1"] for r in group])
        summary[mode] = {
            "mean_macro_f1": float(values.mean()),
            "std_macro_f1": float(values.std(ddof=1)),
            "min_macro_f1": float(values.min()),
            "max_macro_f1": float(values.max()),
            "trainable_parameters": group[0]["trainable_parameters"],
            "seeds": [r["identity"]["seed"] for r in group],
        }
    selected_mode = max(config["modes"], key=lambda m: summary[m]["mean_macro_f1"])
    selected = min(
        [r for r in runs if r["identity"]["mode"] == selected_mode],
        key=lambda r: (
            abs(r["validation"]["macro_f1"] - summary[selected_mode]["mean_macro_f1"]),
            r["identity"]["seed"],
        ),
    )
    # Alpha zero is already in every identical selection grid. A null result is retained.
    selected_path = args.output / "selected.pt"
    shutil.copyfile(selected["checkpoint"], selected_path)
    report = {
        "finished": True,
        "test_evaluated": False,
        "protocol": provenance,
        "baseline": base_metrics,
        "by_representation": summary,
        "selected": {
            k: selected[k]
            for k in (
                "checkpoint",
                "checkpoint_sha256",
                "best_epoch",
                "alpha",
                "validation",
                "trainable_parameters",
            )
        },
        "selected_mode": selected_mode,
        "selected_seed": selected["identity"]["seed"],
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_sha256": sha256(selected_path),
        "runs": [
            {
                k: r[k]
                for k in (
                    "checkpoint",
                    "checkpoint_sha256",
                    "best_epoch",
                    "alpha",
                    "validation",
                    "visual_only_available",
                    "trainable_parameters",
                    "elapsed_sec",
                )
            }
            for r in runs
        ],
        "interpretation": "Three matched seeds per representation with validation epoch/alpha selection. Validation is not an unbiased final estimate; Session 5 is excluded. ImageNet features and blendshapes have different pretraining, so this is a practical system comparison.",
    }
    write_json(args.output / "comparison.json", report)
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("protocol", "runs")}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
