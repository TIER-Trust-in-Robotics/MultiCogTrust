"""Finalize the week-one evidence after all fixed experiment groups succeed.

Keeps completion conditional on artifact checks and the full test suite. Saves
verification evidence and updates the existing repository docs with computed
development results. Never edits a deployment checkpoint or runs test inference.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.iemocap_embedding_cache import sha256, write_json
from src.run_week1_experiments import wait_dependency


def statistic(value, percent=False):
    scale = 100 if percent else 1
    result = f"{value['mean'] * scale:.2f}" if percent else f"{value['mean']:.4f}"
    if value["sample_sd"] is not None:
        result += f" ± {value['sample_sd'] * scale:.2f}" if percent else f" ± {value['sample_sd']:.4f}"
    return result + ("%" if percent else "")


def update_block(path, text):
    start, end = "<!-- WEEK1_RESULTS_START -->", "<!-- WEEK1_RESULTS_END -->"
    current = path.read_text()
    block = f"{start}\n{text}\n{end}"
    if start in current:
        before, remaining = current.split(start, 1)
        _, after = remaining.split(end, 1)
        current = before + block + after
    else:
        current = current.rstrip() + "\n\n" + block + "\n"
    path.write_text(current)


def publish_evidence(summary):
    completed = summary["completed"]
    lines = ["## Completed week-one development comparisons", "", "Computed from the checked three-seed runs (42/43/44). Values are mean ± sample SD; all IEMOCAP rows use the same 1,031 Session 4 examples.", "", "| IEMOCAP model | Reference macro F1 | Reference UAR | Actual tiny.en ASR macro F1 |", "|---|---:|---:|---:|"]
    for group, name in (("baseline", "Distilled CMT baseline"), ("base", "BERT/HuBERT-base CMT"), ("smoothing", "Distilled CMT, smoothing 0.1")):
        row = completed["asr"][group]
        lines.append(f"| {name} | {statistic(row['reference']['macro_f1'])} | {statistic(row['reference']['uar'])} | {statistic(row['metrics']['macro_f1'])} |")
    comparison = completed["encoder_comparison"]["larger_minus_distilled_reference"]
    improvement = summary["improvement"]
    lines += ["", f"The larger-minus-distilled mean gaps are **{comparison['macro_f1'] * 100:.2f} macro-F1 points** and **{comparison['uar'] * 100:.2f} UAR points**. " + ("The baseline meets the proposed two-point engineering tolerance on both metrics." if all(v <= .02 for v in comparison.values()) else "The baseline exceeds the proposed two-point engineering tolerance on at least one metric."), "", f"Smoothing changes mean reference macro F1 by **{improvement['reference_macro_f1_delta'] * 100:+.2f} points** and actual-ASR macro F1 by **{improvement['asr_macro_f1_delta'] * 100:+.2f} points**. " + ("It passes the preregistered mean-quality gate; per-class errors and full-system cost still require review before deployment." if improvement['candidate_passes_preregistered_mean_quality_gate'] else "It does not pass the preregistered mean-quality gate; retain the existing baseline."), "", "All nine selected IEMOCAP heads pass independent reference replay and actual-ASR diagnostics. Four empty-ASR utterances remain as abstentions in every 1,031-example ASR score. No deployment checkpoint was replaced.", "", "These are development comparisons, with the existing session split and six-second crop. They are not MemoCMT author-protocol reproduction, statistical equivalence, population-generalization evidence or trust validation. See `data/processed/benchmarks/week1_v1/summary.json` for seed results, paired deltas and provenance."]
    update_block(ROOT / "docs/async_multimodal_pipeline_roadmap.md", "\n".join(lines))
    lines = ["## Completed week-one original-feature references", "", "All rows use the original released 16,326-example training / 1,871-example validation split and seeds 42/43/44. Values are mean ± sample SD. The known recording-overlap caveat remains attached; no new test predictions were made.", "", "| Reference | MAE | Pearson r | Acc-7 | Acc-2 non0 | F1 non0 | Acc-2 has0 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for group, name in (("mosei_lmf", "MMSA LMF"), ("mosei_mult", "MMSA MulT")):
        row = completed[group]["aggregate"]
        cells = [name, statistic(row["mae"]), statistic(row["pearson_r"]), statistic(row["acc7"], True), statistic(row["acc2_non0"], True), statistic(row["f1_non0"]), statistic(row["acc2_has0"], True)]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "Every selected reference checkpoint reproduces its saved validation predictions after reload. Validation-only selection follows the pinned release behavior; original-feature scores are separate from the modified raw-data results and published test scores. These runs establish reproducible references, without a latest-SOTA or live-camera quality claim.", "", "Checked results: `data/processed/benchmarks/week1_v1/summary.json`. Final test-suite evidence and source hashes: `week1_v1/finalization/verification.json`."]
    update_block(ROOT / "docs/mosei_benchmark.md", "\n".join(lines))
    p = ROOT / "docs/mosei_benchmark.md"
    text = p.read_text().replace("Training is **in progress**. LMF precedes MulT; no three-seed reference result\nor superiority claim is established yet.", "Training is **complete**; the checked three-seed results are appended below.\nNo superiority claim follows from validation versus published test numbers.")
    p.write_text(text)
    p = ROOT / "docs/async_multimodal_pipeline_roadmap.md"
    text = p.read_text().replace("The larger BERT/HuBERT-base comparison and LMF references are training.", "The larger BERT/HuBERT-base, LMF and MulT reference runs are complete.").replace("Training underway\nis not a completed comparison.", "The checked comparisons are appended below.")
    p.write_text(text)


def main():
    root = ROOT / "data/processed/benchmarks/week1_v1"
    output = root / "finalization"
    output.mkdir(exist_ok=True)
    job = output / "job.json"
    if job.exists():
        state = json.loads(job.read_text())
        if state["status"] == "finished":
            raise FileExistsError("Preserving completed finalization")
        try:
            os.kill(state["controller_pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Finalization controller already live")
    state = {"status": "waiting", "controller_pid": os.getpid(), "started_unix": time.time(), "test_model_inference": False}
    write_json(job, state)
    try:
        for group in ("iemocap_base", "iemocap_smoothing", "iemocap_asr", "mosei_lmf", "mosei_mult"):
            wait_dependency(root / group / "job.json")
        commands = [[sys.executable, str(ROOT / "src/summarize_week1.py"), "--require-complete"], [sys.executable, "-m", "pytest", "-q"]]
        checks = []
        for i, command in enumerate(commands):
            log = output / f"check{i}.{int(time.time())}.log"
            with log.open("x") as handle:
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT, env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
                state.update(status="running", child_pid=process.pid, argv=command, log=str(log))
                write_json(job, state)
                code = process.wait()
            checks.append({"argv": command, "exit_code": code, "log": str(log), "log_sha256": sha256(log)})
            if code:
                raise RuntimeError(f"Final check failed; inspect {log}")
        summary = json.loads((root / "summary.json").read_text())
        if not summary["complete"]:
            raise AssertionError("Summary completion gate failed")
        source_paths = [ROOT / "src" / name for name in ("train_mvp_baseline.py", "mmsa_features.py", "train_mmsa_reference.py", "evaluate_week1_asr.py", "resolve_mosei_sources.py", "summarize_week1.py", "finalize_week1.py", "run_week1_experiments.py")]
        verification = {"passed": True, "checks": checks, "test_model_inference": False, "summary_sha256": sha256(root / "summary.json"), "source_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in source_paths}, "completed_unix": time.time()}
        write_json(output / "verification.json", verification)
        publish_evidence(summary)
        state.update(status="finished", finished_unix=time.time(), verification_sha256=sha256(output / "verification.json"))
        write_json(job, state)
        print("Week-one evidence finalized; deployment baseline preserved", flush=True)
    except BaseException as error:
        state.update(status="failed", error=str(error), failed_unix=time.time())
        write_json(job, state)
        raise


if __name__ == "__main__":
    main()
