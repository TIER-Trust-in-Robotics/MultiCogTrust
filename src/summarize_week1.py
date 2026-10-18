"""Check completed week-one artifacts and summarize matched development results.

Missing runs are explicit. A partial summary never passes the completion gate.
Use --require-complete after all groups finish. This command runs no models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark_metrics import mosei_metrics
from src.iemocap_embedding_cache import sha256, write_json
from src.train_mvp_baseline import metrics_from_predictions


def aggregate(rows, keys):
    return {key: {"mean": float(np.mean([r[key] for r in rows])), "sample_sd": float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else None} for key in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = ROOT / "data/processed/benchmarks/week1_v1"
    missing, completed, errors = [], {}, []
    initial = json.loads((root / "input_provenance.json").read_text())
    if sha256(ROOT / "models/mvp/audio_text_baseline.pt") != initial["baseline_checkpoint_sha256"]:
        errors.append("Original deployed baseline changed")
    matrix = json.loads((ROOT / "configs/week1/protocol_matrix.json").read_text())
    source_report = root / "source_resolution/report.json"
    if sha256(source_report) != matrix["provenance"]["source_resolution_report_sha256"]:
        errors.append("Source resolution report changed")
    if sha256(ROOT / "data/MOSEI/label.csv") != json.loads(source_report.read_text())["label_csv_sha256"]:
        errors.append("Raw MOSEI label CSV changed")
    reference_ids = None
    for group in ("iemocap_base", "iemocap_smoothing"):
        rows = []
        for seed in (42, 43, 44):
            output = root / group / f"seed{seed}.pt"
            metrics_path = output.with_suffix(".metrics.json")
            if not metrics_path.exists():
                missing.append(f"{group}/seed{seed}")
                continue
            report = json.loads(metrics_path.read_text())
            config_path = ROOT / "configs/week1" / f"{group}_seed{seed}.json"
            provenance = report["provenance"]
            if not report["finished"] or report["test_evaluated"] or provenance["config_sha256"] != sha256(config_path) or report["checkpoint_sha256"] != sha256(output) or len(report["history"]) != 20:
                errors.append(f"Run contract failed: {group}/{seed}")
            predictions = [json.loads(line) for line in output.with_suffix(".validation.jsonl").read_text().splitlines()]
            ids = [r["utterance_id"] for r in predictions]
            if len(ids) != 1031 or len(set(ids)) != 1031 or any(not uid.startswith("Ses04") for uid in ids):
                errors.append(f"Validation coverage failed: {group}/{seed}")
            if reference_ids is None:
                reference_ids = ids
            elif reference_ids != ids:
                errors.append(f"Comparator IDs differ: {group}/{seed}")
            computed = metrics_from_predictions([r["label"] for r in predictions], [r["probabilities"] for r in predictions])
            if computed != report["validation"]:
                errors.append(f"Saved prediction metrics differ: {group}/{seed}")
            rows.append({"seed": seed, "metrics": computed, "best_epoch": report["best_epoch"], "total_parameters": report["total_parameters"], "trainable_parameters": report["trainable_parameters"], "checkpoint_sha256": report["checkpoint_sha256"], "metrics_sha256": sha256(metrics_path)})
        completed[group] = {"runs": rows, "aggregate": aggregate([r["metrics"] for r in rows], ("macro_f1", "uar", "accuracy")) if rows else None}
    for group in ("mosei_lmf", "mosei_mult"):
        rows = []
        for seed in (42, 43, 44):
            output = root / group / f"seed{seed}"
            metrics_path = output / "metrics.json"
            if not metrics_path.exists():
                missing.append(f"{group}/seed{seed}")
                continue
            report = json.loads(metrics_path.read_text())
            config_path = ROOT / "configs/week1" / f"{group}_seed{seed}.json"
            if not report["finished"] or report["test_model_inference"] or report["provenance"]["config_sha256"] != sha256(config_path) or report["checkpoint_sha256"] != sha256(output / "best.pt") or report["reload_max_abs_difference"] > 1e-4:
                errors.append(f"Run contract/reload failed: {group}/{seed}")
            predictions = [json.loads(line) for line in (output / "validation.jsonl").read_text().splitlines()]
            if len(predictions) != 1871 or len({r["id"] for r in predictions}) != 1871:
                errors.append(f"MOSEI coverage failed: {group}/{seed}")
            computed = mosei_metrics([r["truth"] for r in predictions], [r["score"] for r in predictions])
            if computed != report["validation"]:
                errors.append(f"Saved native metrics differ: {group}/{seed}")
            rows.append({"seed": seed, "metrics": computed, "best_epoch": report["best_epoch"], "epochs_run": report["epochs_run"], "total_parameters": report["total_parameters"], "checkpoint_sha256": report["checkpoint_sha256"], "metrics_sha256": sha256(metrics_path)})
        completed[group] = {"runs": rows, "aggregate": aggregate([r["metrics"] for r in rows], ("mae", "pearson_r", "acc7", "acc2_non0", "f1_non0", "acc2_has0", "f1_has0")) if rows else None}
    asr_path = root / "iemocap_asr/summary.json"
    improvement = None
    if not asr_path.exists():
        missing.append("iemocap_asr/full_nine_head_replay")
    else:
        asr = json.loads(asr_path.read_text())
        if asr["test_model_inference"] or len(asr["runs"]) != 9 or any(r["metrics"]["count"] != 1031 or r["metrics"]["abstentions"] != 4 or r["reference_replay_max_abs_difference"] > 1e-4 for r in asr["runs"].values()):
            errors.append("ASR/reference replay completeness failed")
        completed["asr"] = {}
        for group in ("baseline", "smoothing", "base"):
            runs = [asr["runs"][f"{group}_seed{s}"] for s in (42, 43, 44)]
            completed["asr"][group] = {"metrics": aggregate([r["metrics"] for r in runs], ("macro_f1", "uar", "accuracy")), "reference": aggregate([r["reference_replay"] for r in runs], ("macro_f1", "uar", "accuracy"))}
        reference_delta = completed["asr"]["smoothing"]["reference"]["macro_f1"]["mean"] - completed["asr"]["baseline"]["reference"]["macro_f1"]["mean"]
        asr_delta = completed["asr"]["smoothing"]["metrics"]["macro_f1"]["mean"] - completed["asr"]["baseline"]["metrics"]["macro_f1"]["mean"]
        improvement = {"reference_macro_f1_delta": reference_delta, "asr_macro_f1_delta": asr_delta, "candidate_passes_preregistered_mean_quality_gate": reference_delta > 0 and asr_delta > 0, "deployment_checkpoint_replaced": False, "paired_reference_deltas": [asr["runs"][f"smoothing_seed{s}"]["reference_replay"]["macro_f1"] - asr["runs"][f"baseline_seed{s}"]["reference_replay"]["macro_f1"] for s in (42, 43, 44)], "paired_asr_deltas": [asr["runs"][f"smoothing_seed{s}"]["metrics"]["macro_f1"] - asr["runs"][f"baseline_seed{s}"]["metrics"]["macro_f1"] for s in (42, 43, 44)]}
        completed["encoder_comparison"] = {"larger_minus_distilled_reference": {key: completed["asr"]["base"]["reference"][key]["mean"] - completed["asr"]["baseline"]["reference"][key]["mean"] for key in ("macro_f1", "uar")}, "interpretation": "Development comparison only; engineering tolerance is two percentage points, not statistical equivalence or a paper-protocol reproduction."}
    report = {"complete": not missing and not errors, "missing": missing, "errors": errors, "test_model_inference": False, "protocol_matrix_sha256": sha256(ROOT / "configs/week1/protocol_matrix.json"), "source_sha256": sha256(Path(__file__)), "completed": completed, "improvement": improvement, "raw_view_rescore": json.loads((root / "raw_view_rescore.json").read_text()), "interpretation": "Three-seed development means and sample SD; no population-generalization, trust-validation or published-test superiority claim."}
    write_json(root / "summary.json", report)
    print(json.dumps({"complete": report["complete"], "missing": missing, "errors": errors, "aggregates": {k: v.get("aggregate") for k, v in completed.items()}, "improvement": improvement}, indent=2))
    if errors or (args.require_complete and missing):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
