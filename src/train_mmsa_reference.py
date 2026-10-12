"""Validation-only MMSA LMF/MulT reference with pinned release semantics.

Preserves upstream model, optimizer quirks, shuffled validation batch loss,
rounded selection and eight stale epochs. Ordered post-reload predictions and
native metrics are saved separately. No test dataset/forward pass is created.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_metrics import mosei_metrics
from src.iemocap_embedding_cache import sha256, write_json
from src.mmsa_features import ReferenceFeatures
from src.models.mmsa_reference.lmf import LMF
from src.models.mmsa_reference.mult import MULT


def optimizer_for(model, config):
    if config["model_name"] == "lmf":
        params = list(model.parameters())
        # Upstream omits fusion_weights/fusion_bias, parameters 3 and 4.
        return torch.optim.Adam([
            {"params": params[:3], "lr": config["factor_lr"]},
            {"params": params[5:], "lr": config["learning_rate"]},
        ], weight_decay=config["weight_decay"])
    # Released MulT trainer ignores its configured weight_decay.
    return torch.optim.Adam(model.parameters(), lr=config["learning_rate"])


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    scores = np.empty(len(loader.dataset), dtype=np.float64)
    truth = np.empty_like(scores)
    batch_losses = []
    for text, audio, vision, labels, indices in loader:
        prediction = model(text.to(device), audio.to(device), vision.to(device))["M"]
        batch_losses.append(float(torch.nn.functional.l1_loss(prediction, labels.to(device))))
        scores[indices.numpy()] = prediction.cpu().numpy().reshape(-1)
        truth[indices.numpy()] = labels.numpy().reshape(-1)
    return round(float(np.mean(batch_losses)), 4), mosei_metrics(truth, scores), scores, truth


def save_atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=ROOT / "data/processed/benchmarks/week1_v1/mmsa_features")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="mps")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["test_predictions_allowed"] or config["selection_metric"] != "upstream_validation_Loss":
        raise ValueError("Requires frozen validation-only protocol")
    args.output.mkdir(parents=True, exist_ok=True)
    last_path, best_path = args.output / "last.pt", args.output / "best.pt"
    if (args.output / "metrics.json").exists() or (best_path.exists() and not args.resume):
        raise FileExistsError("Preserving reference run; use --resume only for unfinished runs")
    torch.set_num_threads(4)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(args.device)
    train = ReferenceFeatures(args.features, "train", config.get("need_normalized", False))
    valid = ReferenceFeatures(args.features, "valid", config.get("need_normalized", False))
    loaders = {name: DataLoader(ds, batch_size=config["batch_size"], shuffle=True) for name, ds in (("train", train), ("valid", valid))}
    ordered = DataLoader(valid, batch_size=config["batch_size"], shuffle=False)
    model = (LMF if config["model_name"] == "lmf" else MULT)(SimpleNamespace(**config)).to(device)
    optimizer = optimizer_for(model, config)
    scheduler = None
    if config["model_name"] == "mult":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=.1, patience=config["patience"])
    provenance = {
        "config_sha256": sha256(args.config), "feature_index_sha256": sha256(args.features / "index.json"),
        "trainer_sha256": sha256(Path(__file__)), "feature_reader_sha256": sha256(ROOT / "src/mmsa_features.py"),
        "model_provenance_sha256": sha256(ROOT / "src/models/mmsa_reference/provenance.json"),
        "device": str(device), "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "train_count": len(train), "validation_count": len(valid), "test_model_inference": False,
    }
    epoch, best_epoch, best_loss, history = 0, 0, float("inf"), []
    elapsed_before = 0.0
    if args.resume:
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        if saved["provenance"] != provenance:
            raise ValueError("Resume provenance changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"])
        if args.device == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        random.setstate(saved["python_rng"])
        epoch, best_epoch, best_loss, history = saved["epoch"], saved["best_epoch"], saved["best_loss"], saved["history"]
        elapsed_before = saved["elapsed_sec"]
    started = time.perf_counter()
    print(f"{config['model_name']} seed {config['seed']}: {len(train)} train/{len(valid)} valid, {sum(p.numel() for p in model.parameters())} parameters; test excluded", flush=True)
    while epoch - best_epoch < config["early_stop"]:
        epoch += 1
        model.train()
        total, epoch_start = 0.0, time.perf_counter()
        update = config.get("update_epochs", 1)
        left = update
        for step, (text, audio, vision, labels, _) in enumerate(loaders["train"], 1):
            if left == update:
                optimizer.zero_grad()
            left -= 1
            prediction = model(text.to(device), audio.to(device), vision.to(device))["M"]
            loss = torch.nn.functional.l1_loss(prediction, labels.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            if config.get("grad_clip", -1) != -1:
                torch.nn.utils.clip_grad_value_([p for p in model.parameters() if p.requires_grad], config["grad_clip"])
            total += float(loss.detach())
            if not left:
                optimizer.step()
                left = update
            if step % 500 == 0:
                print(f"epoch {epoch} batch {step}/{len(loaders['train'])}", flush=True)
        # Upstream discards an incomplete accumulation group at the epoch end.
        selection_loss, metrics, scores, truth = evaluate(model, loaders["valid"], device)
        if scheduler is not None:
            scheduler.step(selection_loss)
        if selection_loss <= best_loss - 1e-6:
            best_loss, best_epoch = selection_loss, epoch
            save_atomic(best_path, {"model": cpu_state(model), "provenance": provenance, "epoch": epoch, "selection_loss": selection_loss, "validation_scores": torch.tensor(scores), "validation_truth": torch.tensor(truth)})
        history.append({"epoch": epoch, "train_batch_mae": total / len(loaders["train"]), "upstream_validation_Loss": selection_loss, "validation": metrics, "seconds": time.perf_counter() - epoch_start})
        elapsed = elapsed_before + time.perf_counter() - started
        save_atomic(last_path, {
            "model": cpu_state(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict() if scheduler else None,
            "provenance": provenance, "epoch": epoch, "best_epoch": best_epoch, "best_loss": best_loss, "history": history,
            "torch_rng": torch.get_rng_state(), "mps_rng": torch.mps.get_rng_state() if args.device == "mps" else None,
            "python_rng": random.getstate(), "elapsed_sec": elapsed,
        })
        write_json(args.output / "progress.json", {"finished": False, "best_epoch": best_epoch, "history": history, "provenance": provenance})
        print(f"epoch {epoch}: val MAE={metrics['mae']:.4f}, r={metrics['pearson_r']:.4f}, upstream Loss={selection_loss:.4f}, best={best_epoch} ({history[-1]['seconds']:.1f}s)", flush=True)
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    _, metrics, scores, truth = evaluate(model, ordered, device)
    if not np.allclose(scores, best["validation_scores"].numpy(), rtol=1e-5, atol=1e-5):
        raise AssertionError("Reloaded checkpoint failed validation prediction reproduction")
    rows = [{"id": uid, "truth": float(y), "score": float(p)} for uid, y, p in zip(valid.ids, truth, scores, strict=True)]
    with (args.output / "validation.jsonl").open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    report = {"finished": True, "best_epoch": best_epoch, "epochs_run": epoch, "selection_metric": "upstream_validation_Loss", "best_selection_loss": best_loss, "validation": metrics, "history": history, "provenance": provenance, "config": config, "test_model_inference": False, "checkpoint_sha256": sha256(best_path), "predictions_sha256": sha256(args.output / "validation.jsonl"), "reload_max_abs_difference": float(np.max(np.abs(scores - best["validation_scores"].numpy()))), "total_parameters": sum(p.numel() for p in model.parameters()), "elapsed_sec": elapsed_before + time.perf_counter() - started}
    write_json(args.output / "metrics.json", report)
    write_json(args.output / "progress.json", {"finished": True, "best_epoch": best_epoch, "epochs_run": epoch})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
