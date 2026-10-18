"""Bounded context experiment: OOF selection first, then frozen S4 reports."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
from harness import OUT, ROOT
from metrics import change_accuracy, f1, load, oof, scored
from src.iemocap_embedding_cache import sha256, write_json
import numpy as np

SEEDS = (42, 43)
VARIANTS = ("causalother", "causalownother", "causalown2")
JOB_PATH = OUT / "phase2_job.json"


def run_jobs(jobs):
    pending, live = list(jobs), []
    while pending or live:
        while pending and len(live) < 4:
            tag, arguments = pending.pop(0)
            result = OUT / "runs" / f"{tag}.npz"
            provenance = result.with_suffix(".json")
            if result.exists() and provenance.exists():
                saved = json.loads(provenance.read_text())
                if sha256(result) != saved["predictions_sha256"]:
                    raise ValueError("Completed run changed")
                continue
            log = OUT / "logs" / f"{tag}.log"
            handle = log.open("a")
            cmd = [sys.executable, str(Path(__file__).with_name("harness.py")), "run", "--tag", tag,
                   "--final-only-eval", "--save-weights", *arguments]
            process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT,
                                       env={**os.environ, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
            live.append((process, tag, handle))
            print(f"Started {tag}, PID {process.pid}", flush=True)
        for item in list(live):
            process, tag, handle = item
            code = process.poll()
            if code is not None:
                handle.close()
                live.remove(item)
                if code:
                    for remaining, _, stream in live:
                        remaining.terminate()
                        remaining.wait()
                        stream.close()
                    raise RuntimeError(f"Training failed: {tag} ({code})")
                print(f"Completed {tag}", flush=True)
        write_json(JOB_PATH, {"status": "running", "controller_pid": os.getpid(), "pending": len(pending),
                             "live": [{"pid": process.pid, "tag": tag} for process, tag, _ in live], "updated_unix": time.time()})
        if live:
            time.sleep(2)


def summarize_oof(recipe, text):
    ids = scored((1, 2, 3))
    values = []
    for seed in SEEDS:
        predictions = oof(recipe, seed, text)
        if set(predictions) != set(ids):
            raise ValueError("Incomplete OOF coverage")
        values.append({"seed": seed, "f1": f1(ids, predictions), "change_accuracy": change_accuracy(ids, predictions)})
    return {"runs": values, "f1_mean": float(np.mean([row["f1"] for row in values])), "f1_sd": float(np.std([row["f1"] for row in values], ddof=1)),
            "change_mean": float(np.mean([row["change_accuracy"] for row in values]))}


def main():
    started = time.time()
    audit = json.loads((OUT / "context_source_audit.json").read_text())
    if not audit["passed"]:
        raise ValueError("Context audit not passed")
    first = []
    for variant in VARIANTS:
        for seed in SEEDS:
            for held in (1, 2, 3):
                first.append((f"loso{held}_{variant}_s{seed}", ["--train", *[str(k) for k in (1, 2, 3) if k != held],
                              "--train-text", variant + "ref", "--eval", f"{held}:{variant}ref", "--seed", str(seed)]))
    run_jobs(first)
    oof_results = {variant: summarize_oof(variant, variant + "ref") for variant in VARIANTS}
    oof_results["previous_own"] = summarize_oof("prev", "prevref")
    oof_results["current_only"] = summarize_oof("ref", "reference")
    selected = max(VARIANTS, key=lambda variant: (oof_results[variant]["f1_mean"], -VARIANTS.index(variant)))
    selection = {"selected_variant": selected, "oof": oof_results, "rule": "Maximum mean OOF macro F1 across seeds 42/43; no S4 scores inspected",
                 "frozen_unix": time.time(), "seeds": list(SEEDS), "context_audit_sha256": sha256(OUT / "context_source_audit.json")}
    selection_path = OUT / "phase2_selection.json"
    if selection_path.exists() and json.loads(selection_path.read_text())["selected_variant"] != selected:
        raise ValueError("Frozen OOF selection changed")
    if not selection_path.exists():
        write_json(selection_path, selection)
    print(f"OOF selection frozen: {selected}", flush=True)
    final_jobs = []
    for variant in VARIANTS:
        for seed in SEEDS:
            evaluations = [f"4:{variant}ref", f"4:{variant}asr"]
            if variant != "causalown2":
                evaluations.append(f"4:{variant}hri")
            final_jobs.append((f"full_{variant}_s{seed}", ["--train-text", variant + "ref", "--eval", *evaluations, "--seed", str(seed)]))
    run_jobs(final_jobs)
    results = {}
    for variant in VARIANTS:
        results[variant] = {"oof": oof_results[variant], "session4": {}}
        for condition in ("ref", "asr", "hri"):
            if condition == "hri" and variant == "causalown2":
                continue
            rows = []
            for seed in SEEDS:
                predictions = load(f"full_{variant}_s{seed}", f"4:{variant}{condition}")
                ids = scored((4,))
                rows.append({"seed": seed, "f1": f1(ids, predictions), "change_accuracy": change_accuracy(ids, predictions),
                             "predicted_count": len(predictions), "scored_count": len(ids)})
            results[variant]["session4"][condition] = {"runs": rows, "f1_mean": float(np.mean([row["f1"] for row in rows])),
                         "f1_sd": float(np.std([row["f1"] for row in rows], ddof=1)), "change_mean": float(np.mean([row["change_accuracy"] for row in rows]))}
    write_json(OUT / "phase2_summary.json", {"finished": True, "selected_by_oof": selected, "selection_sha256": sha256(selection_path),
               "results": results, "comparators": {key: oof_results[key] for key in ("previous_own", "current_only")},
               "test_model_inference": False, "training_runs": 24, "elapsed_sec": time.time() - started,
               "limitation": "Two seeds for context variants; HRI exact-other text is a proxy using the interlocutor's human reference transcript, not robot-deployment validation."})
    write_json(JOB_PATH, {"status": "finished", "controller_pid": os.getpid(), "finished_unix": time.time(), "summary_sha256": sha256(OUT / "phase2_summary.json")})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write_json(JOB_PATH, {"status": "failed", "controller_pid": os.getpid(), "error": str(error), "failed_unix": time.time()})
        raise
