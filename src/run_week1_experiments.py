"""Run fixed three-seed experiment groups with durable logs and job state.

Dependencies must finish successfully before the next group starts. Existing
completed runs are verified, never overwritten. MMSA runs can resume from an
epoch boundary; partial IEMOCAP runs are preserved and require manual recovery.
"""

from __future__ import annotations

import argparse
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


def wait_dependency(path):
    print(f"Waiting for dependency {path}", flush=True)
    while True:
        if path.exists():
            state = json.loads(path.read_text())
            if state["status"] == "finished":
                return
            if state["status"] == "failed":
                raise RuntimeError(f"Dependency failed: {path}")
            pid = state.get("controller_pid")
            if pid:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError as error:
                    raise RuntimeError(f"Dependency controller is no longer live: {path}") from error
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("iemocap_smoothing", "iemocap_asr", "mosei_lmf", "mosei_mult"), required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), required=True)
    parser.add_argument("--after", type=Path, action="append", default=[])
    args = parser.parse_args()
    root = ROOT / "data/processed/benchmarks/week1_v1" / args.group
    root.mkdir(parents=True, exist_ok=True)
    job = root / "job.json"
    if job.exists():
        old = json.loads(job.read_text())
        if old.get("status") in {"running", "waiting"}:
            try:
                os.kill(old["controller_pid"], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError("Existing controller is live")
    state = {"status": "waiting", "controller_pid": os.getpid(), "group": args.group, "device": args.device, "dependencies": [str(p) for p in args.after], "started_unix": time.time(), "test_model_inference": False}
    write_json(job, state)
    try:
        for path in args.after:
            wait_dependency(path)
        if args.group == "iemocap_asr":
            summary_path = root / "summary.json"
            if summary_path.exists():
                summary = json.loads(summary_path.read_text())
                if summary["test_model_inference"] or len(summary["runs"]) != 9:
                    raise ValueError("Existing ASR summary is incomplete")
                for report in summary["runs"].values():
                    if sha256(report["checkpoint"]) != report["checkpoint_sha256"]:
                        raise ValueError("ASR comparator checkpoint changed")
                state.update(status="finished", finished_unix=time.time())
                write_json(job, state)
                print("Verified completed ASR comparison", flush=True)
                return
            log = root / f"diagnostic.{int(time.time())}.log"
            cmd = [sys.executable, str(ROOT / "src/evaluate_week1_asr.py"), "--output", str(root), "--device", args.device]
            with log.open("x") as handle:
                process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
                state.update(status="running", child_pid=process.pid, argv=cmd, log=str(log))
                write_json(job, state)
                code = process.wait()
            if code or not (root / "summary.json").exists():
                raise RuntimeError(f"ASR diagnostic failed; inspect {log}")
            state.update(status="finished", finished_unix=time.time())
            write_json(job, state)
            return
        for seed in (42, 43, 44):
            config = ROOT / "configs/week1" / f"{args.group}_seed{seed}.json"
            if args.group == "iemocap_smoothing":
                output = root / f"seed{seed}.pt"
                metrics = output.with_suffix(".metrics.json")
                checkpoint = output
                cmd = [sys.executable, str(ROOT / "src/train_mvp_baseline.py"), "--config", str(config), "--cache", str(ROOT / "data/processed/iemocap_encoder_cache"), "--output", str(output), "--device", args.device, "--offline"]
            else:
                output = root / f"seed{seed}"
                metrics, checkpoint = output / "metrics.json", output / "best.pt"
                cmd = [sys.executable, str(ROOT / "src/train_mmsa_reference.py"), "--config", str(config), "--output", str(output), "--device", args.device]
                if (output / "last.pt").exists():
                    cmd.append("--resume")
            if metrics.exists():
                report = json.loads(metrics.read_text())
                if not report["finished"] or report["provenance"]["config_sha256"] != sha256(config) or report["checkpoint_sha256"] != sha256(checkpoint):
                    raise ValueError("Completed result provenance failed")
                print(f"Verified completed seed {seed}", flush=True)
                continue
            log = root / f"seed{seed}.{int(time.time())}.log"
            with log.open("x") as handle:
                process = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})
                state.update(status="running", seed=seed, child_pid=process.pid, argv=cmd, log=str(log), config_sha256=sha256(config))
                write_json(job, state)
                print(f"Started {args.group} seed {seed}, PID {process.pid}; log {log}", flush=True)
                code = process.wait()
            write_json(root / f"seed{seed}.exit.json", {"exit_code": code, "finished_unix": time.time(), "log": str(log)})
            if code or not metrics.exists():
                raise RuntimeError(f"Seed {seed} did not finish successfully; inspect {log}")
        state.update(status="finished", finished_unix=time.time())
        write_json(job, state)
    except BaseException as error:
        state.update(status="failed", error=str(error), failed_unix=time.time())
        write_json(job, state)
        raise


if __name__ == "__main__":
    main()
