"""Optional OOF-trained face heads and a happiness-only correction policy.

Uses the verified existing development-only visual feature store. It never
loads the S4-selected visual head. Fixed head training and gate grid are saved
before any fit; the gate is frozen using OOF reference predictions before
loading Session 4 audio/text predictions.
"""

import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from happiness_gate import gated_labels
from harness import OUT, ROOT
from metrics import LABEL, load, oof, scored
from src.iemocap_embedding_cache import sha256, write_json
from src.models.visual_temporal import VisualTemporalClassifier, augment_visual, fit_normalization
from src.train_mvp_visual import visual_predictions
from src.visual_feature_dataset import VisualFeatureDataset, collate_visual
from src.visual_fusion import save_visual_checkpoint, load_visual_checkpoint

SEEDS = (42, 43)
DIRECTORY = ROOT / "data/processed/iemocap_visual_features_v2"
DEST = OUT / "stretch_face"
CONFIG_PATH = ROOT / "configs/mvp_day5_visual.json"


def fit(rows, held, config, seed, tag, feature_config, source_hashes):
    path = DEST / f"{tag}.pt"
    if path.exists():
        raise FileExistsError(f"Preserving existing face head {tag}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = VisualTemporalClassifier("combined", config["hidden_dim"], config["dropout"])
    model.set_normalization(fit_normalization(rows, config["normalization_std_floor"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    loader = DataLoader(rows, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_visual,
                        generator=torch.Generator().manual_seed(seed))
    snapshots = []
    for epoch in range(1, 21):
        model.train()
        for batch in loader:
            batch = augment_visual(batch, "combined", config["frame_dropout"], config["combined_stream_dropout"])
            logits, available = model(batch)
            if not available.any():
                continue
            loss = torch.nn.functional.cross_entropy(logits[available], batch["labels"][available])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"])
            optimizer.step()
        if epoch >= 11:
            snapshots.append({key: value.detach().clone() for key, value in model.state_dict().items()})
    model.load_state_dict({key: torch.stack([state[key] for state in snapshots]).mean(0) for key in snapshots[0]})
    model.eval()
    p, available = visual_predictions(model, held)
    provenance = {"seed": seed, "train_ids": [row["utterance_id"] for row in rows], "eval_ids": [row["utterance_id"] for row in held],
                  "weight_average_epochs": [11, 20], "source_sha256": sha256(Path(__file__)), "sources": source_hashes,
                  "test_evaluated": False, "selection": "Fixed twenty epochs; no intermediate evaluation or checkpoint selection"}
    save_visual_checkpoint(path, model, 0., provenance, feature_config)
    reloaded, _ = load_visual_checkpoint(path)
    replay, replay_available = visual_predictions(reloaded, held)
    torch.testing.assert_close(replay, p, rtol=0, atol=0)
    assert torch.equal(replay_available, available)
    np.savez(path.with_suffix(".npz"), ids=np.array(provenance["eval_ids"]), probabilities=p.numpy(), available=available.numpy())
    print(f"Finished and replayed {tag}", flush=True)
    return {row["utterance_id"]: (probability.numpy(), bool(valid)) for row, probability, valid in zip(held, p, available)}


def score(ids, base, visual, policy):
    available_ids = [uid for uid in ids if uid in base]
    probabilities = np.array([base[uid] for uid in available_ids])
    face = np.array([visual[uid][0] for uid in available_ids])
    available = np.array([visual[uid][1] for uid in available_ids])
    labels = probabilities.argmax(1)
    modified, changed = gated_labels(probabilities, face, available, policy[0], policy[1])
    truth = np.array([LABEL[uid] for uid in ids])
    original = {uid: label for uid, label in zip(available_ids, labels)}
    corrected = {uid: label for uid, label in zip(available_ids, modified)}
    before = np.array([original.get(uid, -1) for uid in ids])
    after = np.array([corrected.get(uid, -1) for uid in ids])
    changed_truth = np.array([LABEL[uid] for uid, flag in zip(available_ids, changed) if flag])
    return {"base_f1": float(f1_score(truth, before, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
            "gated_f1": float(f1_score(truth, after, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
            "changed": int(changed.sum()), "fixed": int((changed_truth == 1).sum()),
            "harmed": int(sum(flag and before_label == LABEL[uid] for uid, before_label, flag in zip(available_ids, labels, changed))),
            "n": len(ids), "face_available": int(available.sum()), "base_available": len(available_ids)}


def main():
    if (OUT / "stretch_face_summary.json").exists():
        raise FileExistsError("Preserving finished optional face experiment")
    DEST.mkdir(exist_ok=True)
    contract = {"seeds": list(SEEDS), "mode": "combined", "epochs": 20, "weight_average_epochs": [11, 20],
                "config_sha256": sha256(CONFIG_PATH), "visual_thresholds": [0.5, 0.6, 0.7, 0.8, 0.9],
                "base_confidence_limits": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], "baseline": "B previous-own-line recipe, matched seed",
                "gate_selection": "Maximum mean OOF macro F1; ties prefer fewer changed utterances, then no override. Include no-op baseline.",
                "restriction": "Only change a non-happiness base label to happiness; missing vision and empty ASR preserve the base route",
                "test_evaluated": False, "deploy": False, "source_sha256": sha256(Path(__file__))}
    write_json(OUT / "stretch_face_protocol.json", contract)
    print("Face stretch contract frozen; waiting for core fits", flush=True)
    while not (OUT / "phase3_summary.json").exists() or json.loads((OUT / "phase2_job.json").read_text())["status"] != "finished":
        job = json.loads((OUT / "phase2_job.json").read_text())
        if job["status"] == "running":
            os.kill(job["controller_pid"], 0)
        time.sleep(5)
    torch.set_num_threads(2)
    train_dataset = VisualFeatureDataset(DIRECTORY, "train")
    held_dataset = VisualFeatureDataset(DIRECTORY, "validation")
    train = [(train_dataset.records[i]["session"], train_dataset[i]) for i in range(len(train_dataset))]
    held_rows = [held_dataset[i] for i in range(len(held_dataset))]
    source_hashes = {name: sha256(DIRECTORY / name) for name in ("config.json", "metadata.jsonl", "coverage_report.json")}
    config = json.loads(CONFIG_PATH.read_text())
    face_oof = {}
    for seed in SEEDS:
        face_oof[seed] = {}
        for held in (1, 2, 3):
            trained = [row for session, row in train if session != held]
            validation = [row for session, row in train if session == held]
            face_oof[seed].update(fit(trained, validation, config, seed, f"loso{held}_s{seed}", train_dataset.config, source_hashes))
    train_ids = scored((1, 2, 3))
    bases = {seed: oof("prev", seed, "prevref") for seed in SEEDS}
    policies = [(2., 0.), *[(v, b) for v in contract["visual_thresholds"] for b in contract["base_confidence_limits"]]]
    candidates = []
    for policy in policies:
        results = [score(train_ids, bases[seed], face_oof[seed], policy) for seed in SEEDS]
        candidates.append({"policy": list(policy), "f1_mean": float(np.mean([row["gated_f1"] for row in results])),
                           "changed": sum(row["changed"] for row in results), "runs": results})
    selected = max(candidates, key=lambda row: (row["f1_mean"], -row["changed"]))
    write_json(OUT / "stretch_face_selection.json", {"selected": selected, "candidates": candidates, "fit_sessions": [1, 2, 3],
               "frozen_unix": time.time(), "selection": contract["gate_selection"]})
    rows = []
    for seed in SEEDS:
        face = fit([row for _, row in train], held_rows, config, seed, f"full_s{seed}", train_dataset.config, source_hashes)
        result = {"seed": seed}
        for condition, text in (("reference", "prevref"), ("asr", "prevasr")):
            result[condition] = score(scored((4,)), load(f"full_prev_s{seed}", f"4:{text}"), face, selected["policy"])
        rows.append(result)
    hashes = {path.name: sha256(path) for path in DEST.glob("*")}
    write_json(OUT / "stretch_face_summary.json", {"finished": True, "protocol_sha256": sha256(OUT / "stretch_face_protocol.json"),
               "selection_sha256": sha256(OUT / "stretch_face_selection.json"), "oof_selected": selected, "session4": rows,
               "training_runs": 8, "saved_artifact_hashes": hashes, "test_model_inference": False,
               "limitations": ["Two seeds and one development session; the gate is not deployed.", "The visual architecture is inherited; no claim of an independent architecture comparison.", "Labels are changed only to happiness. The gate does not output calibrated fused probabilities."]})
    print("Face stretch complete", json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
