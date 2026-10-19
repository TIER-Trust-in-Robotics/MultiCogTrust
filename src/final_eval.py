"""Frozen final evaluation of the live audio/text models on IEMOCAP Session 5, run once.

    uv run python src/final_eval.py --session 4                      # dry run on the development session
    uv run python src/final_eval.py --session 5 --confirm-held-out   # the single held-out run

Every line takes the deployed path: Whisper with the live demo's settings on the
first 6 s, then SpeechProcessor for A (current line) and B (the same speaker's
previous Whisper line [SEP] current line). Smoothing presets are replayed per
speaker. Settings come from configs/final_eval_session5.json.

Session 5 runs only when that config is "frozen", every pinned hash matches,
tracked files are committed, a Session 4 dry run of the same commit reproduced
dry_run_reference, and no earlier Session 5 run exists. Nothing is selected or
tuned here; a failed gate is fixed on Session 4, never by looking at Session 5.

Freezing: run the Session 4 dry run, copy its macro F1 into dry_run_reference,
set "status": "frozen", commit, run the dry run again on that commit (it must
pass), then run Session 5.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.emotion_checkpoint import EMOTION_LABEL_MAP, EMOTION_LABELS
from src.iemocap_embedding_cache import sha256, write_json

DEFAULT_CONFIG = ROOT / "configs/final_eval_session5.json"
METADATA_FIELDS = ("utterance_id", "dialog_id", "session", "split", "speaker", "label_code", "text",
                   "dialog_start_sec", "dialog_end_sec")
SUBSETS = {
    "improvised": lambda r: "_impro" in r["dialog_id"],
    "scripted": lambda r: "_script" in r["dialog_id"],
    "female_speaker": lambda r: r["speaker"] == "F",
    "male_speaker": lambda r: r["speaker"] == "M",
}


class GateError(RuntimeError):
    """A precondition is not met; nothing was evaluated."""


def git_state(root=ROOT):
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"),
            "tracked_changes": bool(git("status", "--porcelain", "--untracked-files=no"))}


def whisper_dir(config):
    from huggingface_hub import snapshot_download

    asr = config["asr"]
    return Path(snapshot_download(asr["model"], revision=asr["snapshot"], local_files_only=True))


def metadata_fingerprint(path):
    """SHA-256 of the metadata fields this evaluation uses, wherever IEMOCAP is stored.

    The manifest records absolute file paths, so its file hash differs on every machine.
    Audio is identified by its path inside the release instead (Session5/sentences/wav/...).
    """
    rows = []
    for row in map(json.loads, Path(path).read_text().splitlines()):
        parts = Path(row["audio_path"]).parts
        start = max(i for i, part in enumerate(parts) if re.fullmatch(r"Session\d+", part))
        rows.append({**{key: row[key] for key in METADATA_FIELDS}, "audio": "/".join(parts[start:])})
    rows.sort(key=lambda r: r["utterance_id"])
    return hashlib.sha256("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()).hexdigest()


def check_inputs(config, root=ROOT):
    """Every pinned input must match its hash, and the ASR settings must be the live demo's."""
    from src.inference_pipeline import FAST_ASR

    metadata = config["metadata"]
    pinned = {spec["checkpoint"]: spec["sha256"] for spec in config["models"].values()}
    actual = {path: sha256(root / path) if (root / path).exists() else None for path in pinned}
    pinned[metadata["path"]] = metadata["fingerprint_sha256"]
    actual[metadata["path"]] = metadata_fingerprint(root / metadata["path"]) if (root / metadata["path"]).exists() else None
    try:
        actual["whisper model.bin"] = sha256(whisper_dir(config) / "model.bin")
    except Exception:  # snapshot not downloaded
        actual["whisper model.bin"] = None
    pinned["whisper model.bin"] = config["asr"]["model_bin_sha256"]
    problems = [f"{path}: expected {pinned[path][:12]}…, found {(actual[path] or 'missing')[:12]}"
                for path in pinned if actual[path] != pinned[path]]
    if config["asr"]["options"] != FAST_ASR or config["asr"]["cpu_threads"] != 4:
        problems.append("ASR settings differ from the live demo's (FAST_ASR, 4 threads)")
    if {config["primary_model"], config["comparator_model"]} - set(config["models"]):
        problems.append("primary_model and comparator_model must name entries of models")
    if problems:
        raise GateError("Pinned inputs do not match:\n- " + "\n- ".join(problems))
    return actual


def check_gate(config, session, confirm, git, root=ROOT):
    """Session 4 is the development set and always runs; Session 5 must pass every check."""
    if session == 4:
        return
    if session != config["session"]:
        raise GateError(f"The config is for Session {config['session']}, not {session}")
    problems = []
    if config["status"] != "frozen":
        problems.append('the config status is not "frozen"')
    if not confirm:
        problems.append("--confirm-held-out was not passed")
    if git["tracked_changes"]:
        problems.append("tracked files have uncommitted changes")
    if any(value is None for value in config["dry_run_reference"]["macro_f1"].values()):
        problems.append("dry_run_reference is not filled in")
    dry = root / config["dry_run_dir"] / "manifest.json"
    if not dry.exists():
        problems.append("no Session 4 dry run")
    else:
        manifest = json.loads(dry.read_text())
        if manifest["git"] != {"commit": git["commit"], "tracked_changes": False}:
            problems.append("the Session 4 dry run is not of this clean commit")
        if manifest.get("reference_check") != "pass":
            problems.append("the Session 4 dry run did not reproduce dry_run_reference")
    out = root / config["output_dir"]
    if out.exists() and any(out.iterdir()):
        problems.append(f"{config['output_dir']} already holds a run; Session 5 is evaluated once")
    if problems:
        raise GateError("Session 5 was not run:\n- " + "\n- ".join(problems))


def session_dialogs(metadata_path, session):
    dialogs = defaultdict(list)
    for row in map(json.loads, Path(metadata_path).read_text().splitlines()):
        if row["session"] == session:
            dialogs[row["dialog_id"]].append(row)
    return {d: sorted(rows, key=lambda r: r["dialog_start_sec"]) for d, rows in sorted(dialogs.items())}


def run(config, session, root=ROOT):
    """All lines of the session in dialogue time order, each with its transcript and every model's output."""
    import soundfile as sf
    from faster_whisper import WhisperModel

    from src.core.events import SpeechSegment
    from src.emotion_context import SpeakerContext
    from src.emotion_fusion_infer import CrossModalEmotionPredictor
    from src.inference_pipeline import SpeechProcessor

    asr = config["asr"]
    models = {name: CrossModalEmotionPredictor(root / spec["checkpoint"], config["device"], local_files_only=True)
              for name, spec in config["models"].items()}
    transcriber = WhisperModel(str(whisper_dir(config)), device=asr["device"], compute_type=asr["compute_type"],
                               cpu_threads=asr["cpu_threads"])
    heard = SpeechProcessor(transcriber, None, {"language": asr["language"], "beam_size": asr["beam_size"], **asr["options"]})
    heard.preprocessing = models[config["comparator_model"]].preprocessing
    if any(model.preprocessing != heard.preprocessing for model in models.values()):
        raise GateError("The models disagree on audio/text preprocessing")
    silence = SpeechSegment(0.0, 1.0, 16000, np.zeros(16000, np.float32))
    for name, model in models.items():  # warm-up; timings are recorded, not claimed
        SpeechProcessor(None, model).process(silence, "Ready.")
    heard.process(silence)
    processors, smoothers = {}, {(name, preset): SpeakerContext.preset(preset)
                                 for name in models for preset in config["smoothing"]["secondary"]}
    records = []
    for dialog, rows in session_dialogs(root / config["metadata"]["path"], session).items():
        for row in rows:
            audio, sample_rate = sf.read(row["audio_path"], dtype="float32")
            segment = SpeechSegment(row["dialog_start_sec"], row["dialog_end_sec"], sample_rate, audio)
            asr_result = heard.process(segment)
            record = {key: row[key] for key in ("utterance_id", "dialog_id", "speaker", "label_code")}
            record.update(label=EMOTION_LABEL_MAP.get(row["label_code"], -1), asr_text=asr_result["text"],
                          asr_ms=asr_result["timings_ms"]["asr"], models={})
            for name, model in models.items():
                processor = processors.setdefault((name, dialog, row["speaker"]), SpeechProcessor(None, model))
                result = processor.process(segment, asr_result["text"])
                emotion = result["emotion"]
                probabilities = [emotion["probabilities"][label] for label in EMOTION_LABELS] if emotion else None
                smoothed = {preset: None if emotion is None else
                            [smoothers[name, preset].update(emotion["probabilities"], f"{dialog}/{row['speaker']}")[label]
                             for label in EMOTION_LABELS]
                            for preset in config["smoothing"]["secondary"]}
                record["models"][name] = {"probabilities": probabilities, "smoothed": smoothed,
                                          "context_text": result.get("context_text"),
                                          "emotion_ms": result["timings_ms"]["emotion"]}
            records.append(record)
    return records


def prediction(record, model, smoothing):
    output = record["models"][model]
    probabilities = output["probabilities"] if smoothing == "none" else output["smoothed"][smoothing]
    return -1 if probabilities is None else int(np.argmax(probabilities))


def macro_f1(truth, predicted):
    """Mean F1 over the four classes; an abstention (-1) is a miss for its true class."""
    scores = []
    for c in range(len(EMOTION_LABELS)):
        tp = np.sum((truth == c) & (predicted == c))
        wrong = np.sum((truth != c) & (predicted == c)) + np.sum((truth == c) & (predicted != c))
        scores.append(2 * tp / (2 * tp + wrong) if tp + wrong else 0.0)
    return float(np.mean(scores))


def change_mask(records):
    """Scored lines whose label differs from the same speaker's previous scored line (records in time order)."""
    previous, mask = {}, []
    for r in records:
        if r["label"] < 0:
            continue
        key = (r["dialog_id"], r["speaker"])
        mask.append(key in previous and previous[key] != r["label"])
        previous[key] = r["label"]
    return np.array(mask, bool)


def classification(truth, predicted, changes):
    matrix = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS) + 1), int)
    for t, p in zip(truth, predicted):
        matrix[t, p if p >= 0 else -1] += 1
    per_class = {}
    for c, label in enumerate(EMOTION_LABELS):
        tp, predicted_c, support = matrix[c, c], matrix[:, c].sum(), matrix[c].sum()
        precision, recall = (tp / predicted_c if predicted_c else 0.0), (tp / support if support else 0.0)
        per_class[label] = {"precision": float(precision), "recall": float(recall), "support": int(support),
                            "f1": float(2 * precision * recall / (precision + recall) if precision + recall else 0.0)}
    return {"count": int(len(truth)), "abstentions": int((predicted < 0).sum()),
            "macro_f1": macro_f1(truth, predicted),
            "uar": float(np.mean([per_class[label]["recall"] for label in EMOTION_LABELS])),
            "accuracy": float((truth == predicted).mean()) if len(truth) else None,
            "change_accuracy": float((truth == predicted)[changes].mean()) if changes.any() else None,
            "change_count": int(changes.sum()), "per_class": per_class,
            "confusion_with_abstention_column": matrix.tolist()}


def bootstrap(dialogs, statistic, resamples, seed, interval):
    """Percentile interval of statistic(sampled dialogue ids) over dialogue resamples."""
    rng = np.random.default_rng(seed)
    values = [statistic(rng.choice(dialogs, size=len(dialogs), replace=True)) for _ in range(resamples)]
    values = [v for v in values if v is not None]
    low, high = (1 - interval) / 2, 1 - (1 - interval) / 2
    return [float(np.quantile(values, low)), float(np.quantile(values, high))]


def evaluate(records, config):
    smoothings = ["none", *config["smoothing"]["secondary"]]
    scored = [r for r in records if r["label"] >= 0]
    truth, changes = np.array([r["label"] for r in scored]), change_mask(records)
    dialog_of = np.array([r["dialog_id"] for r in scored])
    dialogs = sorted(set(dialog_of))
    rows_of = {d: np.flatnonzero(dialog_of == d) for d in dialogs}
    predicted = {(m, s): np.array([prediction(r, m, s) for r in scored]) for m in config["models"] for s in smoothings}
    boot = config["metrics"]["bootstrap"]

    def sample(ids):
        return np.concatenate([rows_of[d] for d in ids])

    def ci(statistic):
        return bootstrap(dialogs, statistic, boot["resamples"], boot["seed"], boot["interval"])

    def change_acc(p, rows):
        hits = (truth[rows] == p[rows])[changes[rows]]
        return float(hits.mean()) if hits.size else None

    results = {"models": {}, "paired": {}}
    for (model, smoothing), p in predicted.items():
        cell = {"all": classification(truth, p, changes)}
        for subset, keep in config_subsets(config).items():
            rows = np.array([keep(r) for r in scored], bool)
            cell[subset] = classification(truth[rows], p[rows], changes[rows])
        cell["all"]["macro_f1_ci"] = ci(lambda ids: macro_f1(truth[sample(ids)], p[sample(ids)]))
        cell["all"]["change_accuracy_ci"] = ci(lambda ids: change_acc(p, sample(ids)))
        results["models"].setdefault(model, {})[smoothing] = cell
    primary, comparator = config["primary_model"], config["comparator_model"]
    for smoothing in smoothings:
        a, b = predicted[comparator, smoothing], predicted[primary, smoothing]
        results["paired"][smoothing] = {
            "difference": f"{primary} - {comparator}",
            "macro_f1": macro_f1(truth, b) - macro_f1(truth, a),
            "macro_f1_ci": ci(lambda ids: macro_f1(truth[sample(ids)], b[sample(ids)]) - macro_f1(truth[sample(ids)], a[sample(ids)])),
        }
    results["counts"] = {"lines": len(records), "scored": len(scored), "dialogues": len(dialogs),
                         "empty_transcripts": sum(not r["asr_text"] for r in records)}
    return results


def config_subsets(config):
    return {name: SUBSETS[name] for name in config["subsets"]}


def report(results, config, session):
    smoothings = ["none", *config["smoothing"]["secondary"]]
    title = "Final evaluation, IEMOCAP Session 5 (held out)" if session == 5 else "Dry run, IEMOCAP Session 4 (development)"
    counts = results["counts"]
    lines = [f"# {title}", "", f"Config `{config['name']}` ({config['status']}). {counts['scored']} scored of {counts['lines']} lines "
             f"in {counts['dialogues']} dialogues; {counts['empty_transcripts']} empty transcripts count as errors.", ""]
    if session == 5:
        lines += [f"**Disclosure.** {config['disclosure']}", ""]
    lines += ["| Model | Smoothing | Macro F1 [95% CI] | UAR | Accuracy | Change accuracy [95% CI] |", "|---|---|---|---|---|---|"]
    for model, cells in results["models"].items():
        for smoothing in smoothings:
            m = cells[smoothing]["all"]
            change = "n/a" if m["change_accuracy"] is None else f"{m['change_accuracy']:.3f} [{m['change_accuracy_ci'][0]:.3f}, {m['change_accuracy_ci'][1]:.3f}]"
            lines.append(f"| {model} | {smoothing} | {m['macro_f1']:.3f} [{m['macro_f1_ci'][0]:.3f}, {m['macro_f1_ci'][1]:.3f}] "
                         f"| {m['uar']:.3f} | {m['accuracy']:.3f} | {change} |")
    lines += ["", "| Paired difference | Smoothing | Macro F1 [95% CI] |", "|---|---|---|"]
    for smoothing, d in results["paired"].items():
        lines.append(f"| {d['difference']} | {smoothing} | {d['macro_f1']:+.3f} [{d['macro_f1_ci'][0]:+.3f}, {d['macro_f1_ci'][1]:+.3f}] |")
    lines += ["", "Per-class F1, no smoothing:", "", "| Model | " + " | ".join(EMOTION_LABELS) + " |", "|---" * (len(EMOTION_LABELS) + 1) + "|"]
    for model, cells in results["models"].items():
        per = cells["none"]["all"]["per_class"]
        lines.append(f"| {model} | " + " | ".join(f"{per[label]['f1']:.3f} (n={per[label]['support']})" for label in EMOTION_LABELS) + " |")
    lines += ["", "Macro F1 by subset, no smoothing:", "", "| Model | " + " | ".join(config["subsets"]) + " |", "|---" * (len(config["subsets"]) + 1) + "|"]
    for model, cells in results["models"].items():
        lines.append(f"| {model} | " + " | ".join(f"{cells['none'][s]['macro_f1']:.3f} (n={cells['none'][s]['count']})" for s in config["subsets"]) + " |")
    return "\n".join(lines) + "\n"


def reference_check(results, config):
    reference = config["dry_run_reference"]
    if any(value is None for value in reference["macro_f1"].values()):
        return "unset"
    observed = {m: results["models"][m]["none"]["all"]["macro_f1"] for m in reference["macro_f1"]}
    return "pass" if all(abs(observed[m] - v) <= reference["tolerance"] for m, v in reference["macro_f1"].items()) else "fail"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", type=int, choices=(4, 5), required=True)
    parser.add_argument("--confirm-held-out", action="store_true",
                        help="Required for Session 5: spends the single held-out evaluation")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    git = git_state()
    try:
        inputs = check_inputs(config)
        check_gate(config, args.session, args.confirm_held_out, git)
    except GateError as error:
        parser.exit(2, f"{error}\n")
    out = ROOT / (config["output_dir"] if args.session == 5 else config["dry_run_dir"])
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config["name"], "status": config["status"], "config_sha256": sha256(args.config),
                "session": args.session, "git": git, "inputs": inputs, "host": platform.platform(),
                "python": platform.python_version(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if args.session == 5:
        with (out / "started.json").open("x") as handle:  # a second Session 5 run can never overwrite this
            json.dump(manifest, handle, indent=2)
    records = run(config, args.session)
    results = evaluate(records, config)
    (out / "predictions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    write_json(out / "metrics.json", results)
    (out / "report.md").write_text(report(results, config, args.session))
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if args.session == 4:
        manifest["reference_check"] = reference_check(results, config)
        manifest["observed_macro_f1"] = {m: results["models"][m]["none"]["all"]["macro_f1"] for m in config["models"]}
    write_json(out / "manifest.json", manifest)
    print((out / "report.md").read_text())
    if args.session == 4:
        print(f"Reference check: {manifest['reference_check']} (observed macro F1 {manifest['observed_macro_f1']})")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
