"""Phase 3: native affect dimensions from local frozen audio/text features.

Uses all finite IEMOCAP V/A/D labels in Sessions1-4, including emotions outside
the four-class view. MSP uses native Train/Development only, all emotion codes.
MSP regularization transfers the IEMOCAP OOF text-valence choice, so no MSP
Development tuning is performed. Test audio, transcripts and feature rows are
never encoded or evaluated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from threadpoolctl import threadpool_limits

from harness import CACHE, CONFIG, CONTEXT, OUT
from src.emotion_checkpoint import EMOTION_LABEL_MAP
from src.iemocap_embedding_cache import sha256, write_json
from src.prepare_iemocap import DEFAULT_METADATA

LAMBDAS = np.array([.0001, .001, .01, .1, 1.])


def resume_encoding(previous, contract, matrix_exists):
    """A progress marker cannot substitute for a removed feature matrix."""
    if previous is None:
        return 0, False
    if previous["contract"] != contract:
        raise ValueError("MSP feature source contract changed")
    if not matrix_exists:
        return 0, False
    return previous["encoded"], previous.get("complete", False)


def prepare_iemocap():
    path = OUT / "iemocap_dimension_features.npz"
    if path.exists():
        return path
    rows = [row for row in map(json.loads, Path(DEFAULT_METADATA).read_text().splitlines()) if row["session"] in (1, 2, 3, 4)]
    text, audio, accepted, outside_nominal_range = [], [], [], []
    for row in rows:
        target = row.get("vad")
        if target is None or len(target) != 3 or not np.isfinite(target).all():
            continue
        if not all(1 <= value <= 5 for value in target):
            # Seven local release annotations exceed the nominal 1-5 scale.
            # Preserve those source values rather than clipping or dropping them.
            outside_nominal_range.append({"utterance_id": row["utterance_id"], "vad": target})
        directory = CACHE if row["label_code"] in EMOTION_LABEL_MAP else CONTEXT
        with np.load(directory / f"{row['utterance_id']}.npz", allow_pickle=False) as values:
            text.append(np.concatenate([values["text"].mean(0), values["text"].std(0)]))
            audio.append(np.concatenate([values["audio"].mean(0), values["audio"].std(0)]))
        accepted.append(row)
    np.savez(path, text=np.array(text, np.float32), audio=np.array(audio, np.float32),
             target=np.array([row["vad"] for row in accepted], np.float64), session=np.array([row["session"] for row in accepted]),
             ids=np.array([row["utterance_id"] for row in accepted]), label=np.array([row["label_code"] for row in accepted]))
    write_json(OUT / "iemocap_dimension_features.json", {"metadata_sha256": sha256(DEFAULT_METADATA), "features_sha256": sha256(path),
               "count": len(accepted), "order": ["valence", "arousal", "dominance"], "pooling": "Unpadded sequence mean and population standard deviation, including text special tokens",
               "excluded_missing_vad": len(rows) - len(accepted), "test_encoding": False,
               "outside_nominal_range_preserved": outside_nominal_range,
               "range_policy": "Preserve all finite native annotations, including seven release values outside the nominal 1-5 range; normalize error by nominal range 4."})
    print(f"Prepared IEMOCAP VAD features for {len(accepted)} utterances", flush=True)
    return path


def prepare_msp():
    import torch
    from transformers import AutoModel, AutoTokenizer
    csv_path = ROOT / "data/MSP-Podcast/Labels/labels_consensus.csv"
    transcript_dir = ROOT / "data/MSP-Podcast/Transcripts"
    groups = {"Train": [], "Development": []}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["Split_Set"] not in groups:
                continue
            filename = row["FileName"]
            if Path(filename).name != filename:
                raise ValueError("Invalid MSP identifier")
            valence = float(row["EmoVal"])
            if not np.isfinite(valence) or not 1 <= valence <= 7:
                raise ValueError("Invalid native MSP valence label")
            path = transcript_dir / Path(filename).with_suffix(".txt")
            raw = path.read_bytes()
            groups[row["Split_Set"]].append({"id": filename, "split": row["Split_Set"], "valence": valence,
                "text": raw.decode("utf-8").strip(), "text_sha256": hashlib.sha256(raw).hexdigest()})
    if {name: len(rows) for name, rows in groups.items()} != {"Train": 169190, "Development": 34399}:
        raise ValueError("Unexpected native MSP Train/Development counts")
    records = groups["Train"] + groups["Development"]
    contract = {"labels_csv_sha256": sha256(csv_path), "rows": len(records), "train": len(groups["Train"]),
                "development": len(groups["Development"]), "text_model": CONFIG["text_model_name"], "revision": CONFIG["text_revision"],
                "max_tokens": CONFIG["max_tokens"], "pooling": "mean+population std of unpadded tokens including special tokens", "dtype": "float16",
                "transcript_manifest_hash": hashlib.sha256(json.dumps([(row["id"], row["split"], row["text_sha256"]) for row in records]).encode()).hexdigest()}
    progress_path = OUT / "msp_dimension_encoding.json"
    matrix_path = OUT / "msp_dimension_features.npy"
    previous = json.loads(progress_path.read_text()) if progress_path.exists() else None
    matrix_exists = matrix_path.exists()
    start, complete = resume_encoding(previous, contract, matrix_exists)
    if not matrix_exists:
        # Reset the marker before allocating/loading models: an interruption
        # must not leave a new empty matrix with an old completed marker.
        write_json(progress_path, {"contract": contract, "encoded": 0, "complete": False, "pid": os.getpid()})
    matrix = np.lib.format.open_memmap(matrix_path, mode="r+" if matrix_exists else "w+", dtype=np.float16, shape=(len(records), 1536))
    if not complete:
        torch.set_num_threads(2)
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model_name"], revision=CONFIG["text_revision"], local_files_only=True)
        model = AutoModel.from_pretrained(CONFIG["text_model_name"], revision=CONFIG["text_revision"], local_files_only=True).to("mps").eval()
        started = time.time()
        for i in range(start, len(records), 32):
            batch = records[i:i + 32]
            tokens = tokenizer([row["text"] for row in batch], padding=True, truncation=True, max_length=CONFIG["max_tokens"], return_tensors="pt")
            with torch.inference_mode():
                embeddings = model(**{key: value.to("mps") for key, value in tokens.items()}).last_hidden_state.cpu().numpy()
            for j, row in enumerate(batch):
                value = embeddings[j, :int(tokens["attention_mask"][j].sum())]
                matrix[i + j] = np.concatenate([value.mean(0), value.std(0)]).astype(np.float16)
            if i % 1024 == 0:
                matrix.flush()
                write_json(progress_path, {"contract": contract, "encoded": i + len(batch), "complete": False, "pid": os.getpid()})
            if i % 8192 == 0:
                print(f"MSP valence encoding {i}/{len(records)} ({time.time() - started:.0f}s)", flush=True)
        matrix.flush()
        write_json(progress_path, {"contract": contract, "encoded": len(records), "complete": True, "features_sha256": sha256(matrix_path),
                   "empty_transcripts": sum(not row["text"] for row in records), "test_encoding": False})
        del model
        torch.mps.empty_cache()
    return matrix, records, contract


def covariance_fit(features, targets):
    """Sufficient statistics for normalized squared loss + lambda L2."""
    n, dimension = features.shape
    total, square = np.zeros(dimension), np.zeros(dimension)
    cross, xy = np.zeros((dimension, dimension)), np.zeros((dimension, targets.shape[1]))
    for start in range(0, n, 2048):
        values = np.asarray(features[start:start + 2048], dtype=np.float64)
        total += values.sum(0)
        square += np.square(values).sum(0)
        cross += values.T @ values
        xy += values.T @ targets[start:start + len(values)]
    mean = total / n
    scale = np.sqrt(np.maximum(square / n - mean * mean, 0.))
    scale[scale < 1e-6] = 1.
    target_mean = targets.mean(0)
    covariance = (cross / n - np.outer(mean, mean)) / np.outer(scale, scale)
    covariance = (covariance + covariance.T) / 2
    rhs = (xy / n - mean[:, None] * target_mean) / scale[:, None]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return {"mean": mean, "scale": scale, "target_mean": target_mean, "eigenvalues": np.maximum(eigenvalues, 0.),
            "eigenvectors": eigenvectors, "projected_rhs": eigenvectors.T @ rhs}


def weights(fit, lambdas):
    return fit["eigenvectors"] @ (fit["projected_rhs"] / (fit["eigenvalues"][:, None] + np.asarray(lambdas)))


def prediction(features, fit, coefficient):
    result = []
    for start in range(0, len(features), 2048):
        normalized = (np.asarray(features[start:start + 2048], np.float64) - fit["mean"]) / fit["scale"]
        result.append(normalized @ coefficient + fit["target_mean"])
    return np.concatenate(result)


def regression_metrics(truth, predicted, native_range):
    return {"n": len(truth), "mae": float(np.abs(truth - predicted).mean()), "rmse": float(np.sqrt(np.square(truth - predicted).mean())),
            "pearson_r": float(np.corrcoef(truth, predicted)[0, 1]) if np.ptp(predicted) > 0 else None,
            "r2": float(1 - np.square(truth - predicted).sum() / np.square(truth - truth.mean()).sum()),
            "normalized_mae": float(np.abs(truth - predicted).mean() / native_range)}


def main():
    if (OUT / "phase3_summary.json").exists():
        raise FileExistsError("Preserving completed dimension analysis")
    path = prepare_iemocap()
    msp_features, msp_records, msp_contract = prepare_msp()
    # Four deep-learning workers own the training slots until phase 2 ends.
    while True:
        job = json.loads((OUT / "phase2_job.json").read_text())
        if job["status"] in ("finished", "failed"):
            break
        os.kill(job["controller_pid"], 0)
        time.sleep(5)
    data = np.load(path, allow_pickle=False)
    sessions, native_y = data["session"], data["target"]
    train, valid = sessions < 4, sessions == 4
    y = (native_y - 1.) / 4.
    summary, selected_lambdas, prediction_arrays = {}, {}, {}
    with threadpool_limits(limits=2):
        for modality in ("text", "audio", "both"):
            x = np.concatenate([data["text"], data["audio"]], 1) if modality == "both" else data[modality]
            oof_predictions = np.empty((len(LAMBDAS), int(train.sum()), 3))
            train_sessions = sessions[train]
            for held in (1, 2, 3):
                fit_mask, hold_mask = train & (sessions != held), sessions == held
                fit = covariance_fit(x[fit_mask], y[fit_mask])
                for i, regularization in enumerate(LAMBDAS):
                    oof_predictions[i, train_sessions == held] = prediction(x[hold_mask], fit, weights(fit, regularization))
            loss = np.square(oof_predictions - y[train][None]).mean(1)
            chosen = loss.argmin(0)
            lambdas = LAMBDAS[chosen]
            selected_lambdas[modality] = lambdas.tolist()
            write_json(OUT / f"phase3_selection_{modality}.json", {"lambda": lambdas.tolist(), "oof_mse": loss.tolist(), "grid": LAMBDAS.tolist(),
                       "order": ["valence", "arousal", "dominance"], "selection_sessions": [1, 2, 3]})
            fit = covariance_fit(x[train], y[train])
            coefficient = weights(fit, lambdas)
            predicted = prediction(x[valid], fit, coefficient) * 4. + 1.
            selected_oof = np.column_stack([oof_predictions[chosen[j], :, j] for j in range(3)]) * 4. + 1.
            summary[modality] = {name: {"session4": regression_metrics(native_y[valid, j], predicted[:, j], 4.),
                               "oof": regression_metrics(native_y[train, j], selected_oof[:, j], 4.)} for j, name in enumerate(("valence", "arousal", "dominance"))}
            prediction_arrays[f"{modality}_s4"] = predicted
            prediction_arrays[f"{modality}_oof"] = selected_oof
            np.savez(OUT / f"dimension_ridge_{modality}.npz", mean=fit["mean"], scale=fit["scale"], target_mean=fit["target_mean"], weights=coefficient)
            print(modality, json.dumps(summary[modality]), flush=True)
        train_n = msp_contract["train"]
        msp_y = (np.array([row["valence"] for row in msp_records])[:, None] - 1.) / 6.
        regularization = selected_lambdas["text"][0]
        fit = covariance_fit(msp_features[:train_n], msp_y[:train_n])
        coefficient = weights(fit, regularization)
        predicted = prediction(msp_features[train_n:], fit, coefficient).ravel() * 6. + 1.
        msp_result = regression_metrics(msp_y[train_n:, 0] * 6. + 1., predicted, 6.)
        msp_mean = np.full(len(predicted), msp_y[:train_n, 0].mean() * 6. + 1.)
        msp_result = {"text": msp_result, "train_mean_baseline": regression_metrics(msp_y[train_n:, 0] * 6. + 1., msp_mean, 6.),
                      "train_n": train_n, "development_n": len(predicted), "lambda": regularization,
                      "lambda_source": "IEMOCAP OOF text-valence selection, fixed before MSP Development prediction", "all_emotion_codes": True,
                      "encoding_contract": msp_contract}
        np.savez(OUT / "msp_valence_predictions.npz", ids=np.array([row["id"] for row in msp_records[train_n:]]),
                 truth=msp_y[train_n:, 0] * 6. + 1., prediction=predicted)
        np.savez(OUT / "dimension_ridge_msp_text.npz", mean=fit["mean"], scale=fit["scale"], target_mean=fit["target_mean"], weights=coefficient)
    np.savez(OUT / "iemocap_dimension_predictions.npz", ids=data["ids"], session=sessions, truth=native_y, **prediction_arrays)
    write_json(OUT / "phase3_summary.json", {"finished": True, "iemocap": summary, "selected_lambdas": selected_lambdas,
               "iemocap_counts": {"train": int(train.sum()), "session4": int(valid.sum())}, "msp": msp_result,
               "source_sha256": sha256(Path(__file__)), "test_model_inference": False,
               "limitations": ["IEMOCAP dimensions use all finite annotations, not only the four-class subset.", "Ridge pooling may discard temporal and lexical structure.", "IEMOCAP and MSP have different native dimension scales; compare normalized error and correlation with domain caveats.", "These are affect labels; no trust target was trained or validated."]})
    print("Phase 3 complete", flush=True)


if __name__ == "__main__":
    main()
