"""Select an ASR deployment weight without retraining or changing the primary comparison."""

from __future__ import annotations
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
from src.visual_fusion import load_visual_checkpoint, save_visual_checkpoint


def main():
    torch.set_num_threads(2)
    config_path = ROOT / "configs/mvp_day7_deployment.json"
    config = json.loads(config_path.read_text())
    reference = ROOT / "models/mvp/visual_day5/selected.pt"
    base_checkpoint = ROOT / "models/mvp/audio_text_baseline.pt"
    model, payload = load_visual_checkpoint(reference, base_checkpoint)
    dataset = VisualFeatureDataset(
        ROOT / "data/processed/iemocap_visual_features_v2", "validation"
    )
    rows = [dataset[i] for i in range(len(dataset))]
    visual, available = visual_predictions(model, rows)
    asr_path = ROOT / "data/processed/mvp/asr_validation/predictions.jsonl"
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
    for alpha in config["alphas"]:
        fused = fuse_probabilities(base, visual, available, alpha)
        predictions = [
            {
                "label": r["label"],
                "prediction": int(fused[i].argmax())
                if asr[r["utterance_id"]]["probabilities"]
                else -1,
            }
            for i, r in enumerate(rows)
        ]
        candidates.append({"alpha": alpha, "metrics": diagnostic_metrics(predictions)})
    chosen = max(candidates, key=lambda r: (r["metrics"]["macro_f1"], -r["alpha"]))
    selection = {
        "config": config,
        "config_sha256": sha256(config_path),
        "reference_selected_checkpoint_sha256": sha256(reference),
        "asr_predictions_sha256": sha256(asr_path),
        "asr_config": json.loads(asr_path.with_name("config.json").read_text()),
        "candidates": candidates,
        "selected": chosen,
        "interpretation": "This ASR validation set now participates in deployment weight selection. Its result is a development selection score, not independent evaluation. The head and reference comparison are unchanged; Session 5 is excluded.",
    }
    output = reference.with_name("deployment.pt")
    if output.exists():
        raise FileExistsError(f"Preserving existing deployment selection: {output}")
    save_visual_checkpoint(
        output,
        model,
        chosen["alpha"],
        {
            **payload["provenance"],
            "reference_selection": {
                "alpha": payload["alpha"],
                "validation": payload["provenance"]["validation"],
            },
            "deployment_selection": selection,
            "validation": chosen["metrics"],
        },
        payload["feature_config"],
    )
    write_json(
        output.with_suffix(".selection.json"),
        {**selection, "checkpoint_sha256": sha256(output)},
    )
    print(
        json.dumps(
            {"chosen": chosen, "candidates": candidates, "checkpoint": str(output)},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
