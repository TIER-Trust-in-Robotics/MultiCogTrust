"""Train bounded development controls on IEMOCAP or native MOSEI sentiment.

IEMOCAP uses the exact frozen encoder cache of the current audio/text baseline.
MOSEI fits a lexical TF-IDF ridge regressor to continuous sentiment. No test
prediction path is provided. Outputs never replace deployment checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.audit_benchmark_data import development_records
from src.benchmark_metrics import mosei_metrics
from src.iemocap_embedding_cache import CachedDataset, sha256, write_json
from src.prepare_iemocap import DEFAULT_METADATA
from src.train_cross_modal_finetune_iemocap import load_samples
from src.train_mvp_baseline import metrics_from_predictions


def standardize_from_training(train, validation):
    mean, scale = train.mean(0), train.std(0)
    scale = np.maximum(scale, 1e-5)
    return (train - mean) / scale, (validation - mean) / scale, mean, scale


def save_predictions(path, ids, labels, predictions, probabilities=None):
    with path.open("x") as handle:
        for i, uid in enumerate(ids):
            row = {"id": uid, "label": float(labels[i]), "prediction": float(predictions[i])}
            if probabilities is not None:
                row["probabilities"] = probabilities[i].tolist()
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def run_iemocap(config, output):
    cfg = config["iemocap"]
    if cfg["train_sessions"] != [1, 2, 3] or cfg["validation_session"] != 4 or cfg["reserved_session"] != 5:
        raise ValueError("This development comparison must exclude Session 5")
    index_path = ROOT / cfg["cache_index"]
    index = json.loads(index_path.read_text())
    if index["contract"]["metadata_sha256"] != sha256(DEFAULT_METADATA):
        raise ValueError("Metadata differs from the encoder cache")
    samples = [s for s in load_samples(str(DEFAULT_METADATA), str(ROOT)) if s.session in [1, 2, 3, 4]]
    if [s.utterance_id for s in samples] != index["contract"]["utterance_ids"]:
        raise ValueError("Development/cache IDs or ordering differ")
    dataset = CachedDataset(samples, index_path.parent)
    pooled_text, pooled_audio = [], []
    for text, audio, _ in dataset:
        pooled_text.append(text.mean(0).numpy())
        pooled_audio.append(audio.mean(0).numpy())
    pooled_text, pooled_audio = np.stack(pooled_text), np.stack(pooled_audio)
    train_mask = np.array([s.session in [1, 2, 3] for s in samples])
    labels = np.array([s.label for s in samples])
    validation_ids = [s.utterance_id for s in samples if s.session == 4]
    y_train, y_val = labels[train_mask], labels[~train_mask]
    provenance = {
        "cache_index_sha256": sha256(index_path), "encoders": index["contract"]["encoders"],
        "preprocessing": index["contract"]["preprocessing"],
        "train_count": len(y_train), "validation_count": len(y_val),
        "train_utterance_ids": [s.utterance_id for s in samples if s.session != 4],
        "validation_utterance_ids": validation_ids,
        "test_model_inference": False,
        "comparison": "Same representations, sessions, 20-epoch budget, optimizer, batch size and learning rate; heads differ in capacity and pooling.",
    }
    write_json(output / "provenance.json", provenance)
    runs = []
    for mode in cfg["modes"]:
        features = {"text": pooled_text, "audio": pooled_audio,
                    "concatenation": np.concatenate([pooled_text, pooled_audio], axis=1)}[mode]
        train, valid, mean, scale = standardize_from_training(features[train_mask], features[~train_mask])
        train_x, val_x = torch.from_numpy(train), torch.from_numpy(valid)
        for seed in config["seeds"]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = nn.Sequential(
                nn.Linear(features.shape[1], cfg["hidden_dim"]), nn.LeakyReLU(),
                nn.Dropout(cfg["dropout"]), nn.Linear(cfg["hidden_dim"], 4),
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
            loader = DataLoader(TensorDataset(train_x, torch.from_numpy(y_train)),
                                batch_size=cfg["batch_size"], shuffle=True,
                                generator=torch.Generator().manual_seed(seed))
            best, history, selected = -1, [], None
            started = time.perf_counter()
            for epoch in range(1, cfg["epochs"] + 1):
                model.train()
                total_loss = 0
                for x, y in loader:
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.cross_entropy(model(x), y)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * len(y)
                model.eval()
                with torch.inference_mode():
                    probabilities = model(val_x).softmax(-1).numpy()
                metrics = metrics_from_predictions(y_val, probabilities)
                history.append({"epoch": epoch, "train_loss": total_loss / len(y_train), "validation": metrics})
                if metrics["macro_f1"] > best:
                    best, selected = metrics["macro_f1"], {
                        "mode": mode, "seed": seed, "best_epoch": epoch, "validation": metrics,
                        "trainable_parameters": sum(p.numel() for p in model.parameters()),
                    }
                    selected_probabilities = probabilities.copy()
                    torch.save({
                        "schema": "development_pooled_control_v1", "task": "iemocap_emotion_4",
                        "mode": mode, "state_dict": model.state_dict(),
                        "input_dim": features.shape[1], "hidden_dim": cfg["hidden_dim"],
                        "dropout": cfg["dropout"], "mean": torch.from_numpy(mean),
                        "scale": torch.from_numpy(scale), "selection": selected, "provenance": provenance,
                    }, output / f"{mode}_{seed}.pt")
            selected.update(history=history, elapsed_sec=time.perf_counter() - started)
            write_json(output / f"{mode}_{seed}.json", selected)
            save_predictions(output / f"{mode}_{seed}.validation.jsonl", validation_ids,
                             y_val, selected_probabilities.argmax(1), selected_probabilities)
            runs.append(selected)
            print(f"{mode} seed {seed}: F1={best:.4f} UAR={selected['validation']['uar']:.4f} epoch={selected['best_epoch']} ({selected['elapsed_sec']:.1f}s)", flush=True)
    summary = {}
    for mode in cfg["modes"]:
        summary[mode] = {
            key: {"mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}
            for key in ("macro_f1", "uar", "accuracy")
            for values in [[r["validation"][key] for r in runs if r["mode"] == mode]]
        }
    baseline_path = ROOT / cfg["reference_checkpoint"]
    baseline = json.loads(baseline_path.with_suffix(".metrics.json").read_text())
    if baseline["checkpoint_sha256"] != sha256(baseline_path):
        raise ValueError("Reference checkpoint changed")
    if baseline["provenance"]["validation_utterance_ids"] != validation_ids:
        raise ValueError("Reference baseline validation sample set differs")
    return {
        "task": cfg["task"], "test_model_inference": False, "controls": summary,
        "fixed_cmt_seed42": baseline["validation"],
        "reference_checkpoint_sha256": sha256(baseline_path),
        "note": "Controls have three seeds; fixed reference has one seed. This is not a published-model reproduction or a held-out test comparison.",
    }


def run_mosei(config, manifest, output):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    import joblib

    cfg = config["mosei"]
    contract = json.loads(manifest.with_name("development_contract.json").read_text())
    if (contract["manifest_sha256"] != sha256(manifest)
            or contract["label_csv_sha256"] != cfg["label_csv_sha256"]
            or contract["protocol"] != cfg["split_source"]):
        raise ValueError("MOSEI development contract changed")
    rows = development_records(manifest)
    train, valid = ([r for r in rows if r["split"] == s] for s in ("train", "valid"))
    vectorizer = TfidfVectorizer(
        max_features=cfg["tfidf_max_features"], ngram_range=tuple(cfg["tfidf_ngram_range"]),
        min_df=cfg["tfidf_min_df"], sublinear_tf=True, strip_accents="unicode",
    )
    x_train = vectorizer.fit_transform([r["text"] for r in train])
    x_val = vectorizer.transform([r["text"] for r in valid])
    y_train, y_val = (np.array([r["sentiment"] for r in group]) for group in (train, valid))
    history, best, selected = [], float("inf"), None
    for alpha in cfg["ridge_alphas"]:
        regressor = Ridge(alpha=alpha, solver="lsqr", tol=1e-6, max_iter=5000)
        regressor.fit(x_train, y_train)
        predictions = regressor.predict(x_val)
        metrics = mosei_metrics(y_val, predictions)
        history.append({"alpha": alpha, "validation": metrics})
        print(f"MOSEI lexical ridge alpha={alpha}: MAE={metrics['mae']:.4f} r={metrics['pearson_r']:.4f} Acc2(non0)={metrics['acc2_non0']:.4f}", flush=True)
        if metrics["mae"] < best:
            best, selected, best_predictions = metrics["mae"], history[-1], predictions.copy()
            joblib.dump(Pipeline([("tfidf", vectorizer), ("regression", regressor)]), output / "tfidf_ridge.joblib")
    save_predictions(output / "tfidf_ridge.validation.jsonl", [r["id"] for r in valid], y_val, best_predictions)
    return {
        "task": cfg["task"], "test_model_inference": False,
        "train_count": len(train), "validation_count": len(valid), "vocabulary_size": len(vectorizer.vocabulary_),
        "manifest_sha256": sha256(manifest), "selected": selected, "history": history,
        "training_mean_constant": mosei_metrics(y_val, np.full(len(y_val), y_train.mean())),
        "selection": "Minimum validation MAE over the four preregistered alphas; vocabulary fit only on training.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("iemocap", "mosei"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/benchmark_quality_v1.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/benchmarks/mosei_clean_v2/development.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["test_model_inference"]:
        raise ValueError("This runner is for development only")
    args.output.mkdir(parents=True, exist_ok=False)
    import sklearn
    write_json(args.output / "run_config.json", {
        "config": config, "config_sha256": sha256(args.config),
        "runner_sha256": sha256(Path(__file__)),
        "metrics_sha256": sha256(ROOT / "src/benchmark_metrics.py"),
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
    })
    torch.set_num_threads(4)
    started = time.perf_counter()
    report = run_iemocap(config, args.output) if args.dataset == "iemocap" else run_mosei(config, args.manifest, args.output)
    report["elapsed_sec"] = time.perf_counter() - started
    report["finished"] = True
    write_json(args.output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
