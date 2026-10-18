"""Matched previous-own-line control for phase-two float16 cache storage.

Frozen before inspecting phase-two S4 results. No new hyperparameter search:
quantize the existing development-only text features, then use the same seeds,
fixed eight epochs and epoch 3-8 average as the new context variants.
"""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from harness import IMPROVE, OUT, ROOT
from metrics import change_accuracy, f1, load, oof, scored
from src.iemocap_embedding_cache import sha256, write_json


def main():
    if (OUT / "precision_control_summary.json").exists():
        raise FileExistsError("Preserving completed precision control")
    write_json(OUT / "precision_control_protocol.json", {"seeds": [42, 43], "epochs": 8, "weight_average_epochs": [3, 8],
               "train_sessions": [1, 2, 3], "oof_folds": [1, 2, 3], "storage": "float16 roundtrip of original previous-own-line float32 text features",
               "question": "Matched precision comparator for phase2 variants; no seed, epoch or hyperparameter selection on S4",
               "max_workers": 3, "test_evaluated": False, "source_sha256": sha256(Path(__file__)), "frozen_unix": time.time()})
    for condition, old_source in (("ref", "prevref"), ("asr", "prevasr")):
        directory = OUT / f"text_prev16{condition}"
        directory.mkdir(exist_ok=True)
        rows = []
        for uid in scored((1, 2, 3, 4) if condition == "ref" else (4,)):
            original = IMPROVE / f"text_{old_source}" / f"{uid}.npy"
            if not original.exists():
                assert condition == "asr"
                continue
            value = np.load(original, allow_pickle=False)
            assert value.dtype == np.float32 and value.ndim == 2 and value.shape[1] == 768 and np.isfinite(value).all()
            path = directory / f"{uid}.npy"
            if path.exists():
                np.testing.assert_array_equal(np.load(path, allow_pickle=False), value.astype(np.float16))
            else:
                np.save(path, value.astype(np.float16), allow_pickle=False)
            rows.append({"utterance_id": uid, "source_sha256": sha256(original), "feature_sha256": sha256(path)})
        assert len(rows) == (4290 if condition == "ref" else 1027)
        manifest = directory / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        write_json(directory / "contract.json", {"count": len(rows), "dtype": "float16", "source": str(IMPROVE / f"text_{old_source}"),
                   "manifest_sha256": sha256(manifest), "conversion": "float32 to float16, loaded as float32 by the cached-head trainer", "test_encoding": False})
    print("Precision control frozen; waiting for primary context fits", flush=True)
    while True:
        job = json.loads((OUT / "phase2_job.json").read_text())
        if job["status"] == "finished":
            break
        if job["status"] == "failed":
            raise RuntimeError("Primary context experiment failed")
        os.kill(job["controller_pid"], 0)
        time.sleep(5)
    # Three fusion workers leave one training slot for the optional CPU face run.
    jobs = []
    for seed in (42, 43):
        for held in (1, 2, 3, None):
            tag = f"loso{held}_prev16_s{seed}" if held else f"full_prev16_s{seed}"
            train = [k for k in (1, 2, 3) if k != held]
            evaluations = [f"{held}:prev16ref"] if held else ["4:prev16ref", "4:prev16asr"]
            jobs.append((tag, ["--train", *map(str, train), "--train-text", "prev16ref", "--eval", *evaluations, "--seed", str(seed)]))
    pending, live = list(jobs), []
    while pending or live:
        while pending and len(live) < 3:
            tag, arguments = pending.pop(0)
            handle = (OUT / "logs" / f"{tag}.log").open("a")
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("harness.py")), "run", "--tag", tag,
                         "--final-only-eval", "--save-weights", *arguments], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                         env={**os.environ, "HF_HUB_OFFLINE": "1"})
            live.append((process, tag, handle))
            print(f"Started {tag}: {process.pid}", flush=True)
        for item in list(live):
            process, tag, handle = item
            code = process.poll()
            if code is not None:
                handle.close()
                live.remove(item)
                if code:
                    for other, _, stream in live:
                        other.terminate()
                        other.wait()
                        stream.close()
                    raise RuntimeError(f"Precision control failed: {tag}")
                print(f"Completed {tag}", flush=True)
        write_json(OUT / "precision_control_job.json", {"status": "running", "controller_pid": os.getpid(),
                   "pending": len(pending), "live": [{"pid": process.pid, "tag": tag} for process, tag, _ in live]})
        if live:
            time.sleep(2)
    results = []
    for seed in (42, 43):
        row = {"seed": seed}
        for condition, ids, predictions in [("oof", scored((1, 2, 3)), oof("prev16", seed, "prev16ref")),
                ("reference", scored((4,)), load(f"full_prev16_s{seed}", "4:prev16ref")),
                ("asr", scored((4,)), load(f"full_prev16_s{seed}", "4:prev16asr"))]:
            row[condition] = {"f1": f1(ids, predictions), "change_accuracy": change_accuracy(ids, predictions), "n": len(ids)}
        results.append(row)
    write_json(OUT / "precision_control_summary.json", {"finished": True, "training_runs": 8, "runs": results,
               "protocol_sha256": sha256(OUT / "precision_control_protocol.json"), "test_model_inference": False})
    write_json(OUT / "precision_control_job.json", {"status": "finished", "controller_pid": os.getpid()})
    print("Precision control complete", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
