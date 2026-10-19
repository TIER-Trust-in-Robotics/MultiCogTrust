import copy
import json
from pathlib import Path

import numpy as np
import pytest

from src.final_eval import GateError, change_mask, check_gate, evaluate, macro_f1, reference_check

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/final_eval_session5.json").read_text())
CLEAN = {"commit": "abc123", "tracked_changes": False}


def frozen(tmp_path, commit="abc123", check="pass"):
    config = copy.deepcopy(CONFIG)
    config.update(status="frozen", output_dir="out/s5", dry_run_dir="out/s4")
    config["dry_run_reference"]["macro_f1"] = {"A": 0.73, "B": 0.725}
    (tmp_path / "out/s4").mkdir(parents=True)
    (tmp_path / "out/s4/manifest.json").write_text(json.dumps(
        {"git": {"commit": commit, "tracked_changes": False}, "reference_check": check}))
    return config


def test_session5_is_refused_while_the_config_is_a_draft(tmp_path):
    config = copy.deepcopy(CONFIG)
    config.update(status="draft")
    config["dry_run_reference"]["macro_f1"] = {"A": None, "B": None}
    with pytest.raises(GateError) as error:
        check_gate(config, 5, confirm=False, git=CLEAN, root=tmp_path)
    for reason in ('not "frozen"', "--confirm-held-out", "dry_run_reference", "no Session 4 dry run"):
        assert reason in str(error.value)


def test_session5_runs_only_after_a_passing_dry_run_of_the_same_clean_commit(tmp_path):
    check_gate(frozen(tmp_path), 5, confirm=True, git=CLEAN, root=tmp_path)  # every condition holds
    with pytest.raises(GateError, match="not of this clean commit"):
        check_gate(frozen(tmp_path / "old", commit="old"), 5, True, CLEAN, tmp_path / "old")
    with pytest.raises(GateError, match="did not reproduce"):
        check_gate(frozen(tmp_path / "fail", check="fail"), 5, True, CLEAN, tmp_path / "fail")
    with pytest.raises(GateError, match="uncommitted changes"):
        check_gate(frozen(tmp_path / "dirty"), 5, True, {**CLEAN, "tracked_changes": True}, tmp_path / "dirty")
    with pytest.raises(GateError, match="--confirm-held-out"):
        check_gate(frozen(tmp_path / "noconfirm"), 5, False, CLEAN, tmp_path / "noconfirm")


def test_an_existing_session5_run_blocks_a_second_one(tmp_path):
    config = frozen(tmp_path)
    (tmp_path / "out/s5").mkdir(parents=True)
    (tmp_path / "out/s5/started.json").write_text("{}")
    with pytest.raises(GateError, match="evaluated once"):
        check_gate(config, 5, True, CLEAN, tmp_path)


def test_the_session4_dry_run_is_never_gated(tmp_path):
    check_gate(CONFIG, 4, confirm=False, git={**CLEAN, "tracked_changes": True}, root=tmp_path)


def test_shipped_config_pins_the_live_demo_settings():
    from src.inference_pipeline import FAST_ASR

    assert CONFIG["status"] in ("draft", "frozen") and CONFIG["session"] == 5
    assert CONFIG["asr"]["options"] == FAST_ASR and CONFIG["asr"]["cpu_threads"] == 4
    assert {CONFIG["primary_model"], CONFIG["comparator_model"]} == set(CONFIG["models"])
    assert CONFIG["smoothing"]["primary"] == "none"


def record(dialog, speaker, label, a, b=None):
    probs = lambda k: None if k is None else np.eye(4)[k].tolist()
    return {"dialog_id": dialog, "speaker": speaker, "label": label, "asr_text": "" if a is None else "x",
            "models": {"A": {"probabilities": probs(a), "smoothed": {"balanced": probs(a), "max_f1": probs(a)}},
                       "B": {"probabilities": probs(b if b is not None else a),
                             "smoothed": {"balanced": probs(b if b is not None else a), "max_f1": probs(b if b is not None else a)}}}}


def test_abstentions_count_as_errors_and_changes_follow_each_speaker():
    records = [record("Ses04F_impro01", "F", 0, 0), record("Ses04F_impro01", "M", 3, 3),
               record("Ses04F_impro01", "F", -1, 1),  # unscored: context only
               record("Ses04F_impro01", "F", 1, None),  # empty transcript, label changed 0 -> 1
               record("Ses04F_script01_1", "F", 1, 1, b=0)]
    assert change_mask(records).tolist() == [False, False, True, False]
    truth, predicted = np.array([0, 3, 1, 1]), np.array([0, 3, -1, 1])
    assert macro_f1(truth, predicted) == pytest.approx((1 + 1 + 2 / 3 + 0) / 4)
    results = evaluate(records, CONFIG)
    a = results["models"]["A"]["none"]
    assert a["all"]["abstentions"] == 1 and a["all"]["change_accuracy"] == 0.0
    assert a["improvised"]["count"] == 3 and a["scripted"]["count"] == 1
    assert results["paired"]["none"]["macro_f1"] < 0  # B misses the scripted line A gets
    low, high = a["all"]["macro_f1_ci"]
    assert low <= a["all"]["macro_f1"] <= high
    assert evaluate(records, CONFIG)["models"]["A"]["none"]["all"]["macro_f1_ci"] == a["all"]["macro_f1_ci"]  # seeded


def test_reference_check_needs_every_model_within_tolerance():
    config = copy.deepcopy(CONFIG)
    config["dry_run_reference"]["macro_f1"] = {"A": None, "B": None}
    results = {"models": {m: {"none": {"all": {"macro_f1": v}}} for m, v in {"A": 0.730, "B": 0.725}.items()}}
    assert reference_check(results, config) == "unset"
    config["dry_run_reference"]["macro_f1"] = {"A": 0.732, "B": 0.725}
    assert reference_check(results, config) == "pass"
    config["dry_run_reference"]["macro_f1"]["A"] = 0.740
    assert reference_check(results, config) == "fail"


def test_metadata_fingerprint_ignores_where_iemocap_is_stored(tmp_path):
    from src.final_eval import metadata_fingerprint

    def write(name, root, label="ang"):
        rows = [{"utterance_id": "Ses05F_impro01_F000", "dialog_id": "Ses05F_impro01", "session": 5, "split": "test",
                 "speaker": "F", "label_code": label, "text": "Hi.", "dialog_start_sec": 1.0, "dialog_end_sec": 2.0,
                 "audio_path": f"{root}/IEMOCAP_full_release/Session5/sentences/wav/Ses05F_impro01/Ses05F_impro01_F000.wav",
                 "video_path": f"{root}/elsewhere.avi"}]
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
        return metadata_fingerprint(tmp_path / name)

    here = write("a.jsonl", "/Users/someone/Session9/project/data")  # an unrelated "Session" folder higher up
    assert here == write("b.jsonl", "/home/lab/datasets")
    assert here != write("c.jsonl", "/home/lab/datasets", label="neu")
