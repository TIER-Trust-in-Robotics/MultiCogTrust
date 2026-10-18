import json

import numpy as np
import pytest

from src.audit_benchmark_data import (
    development_records, quarantine_training_conflicts, read_mosei_csv, read_official_folds,
)
from src.benchmark_metrics import class_scores, mosei_metrics
from src.benchmark_controls import standardize_from_training
from src.audit_mosei_duplicates import audio_similarity
from src.audit_mosei_media import word_error_rate


def test_native_mosei_zero_thresholds_rounding_and_continuous_error():
    # Zero is positive only for has0; bankers' rounding maps +/-0.5 to zero.
    metrics = mosei_metrics([-1, 0, 1, 0.5, -0.5], [0, 0, 1, 0.5, -0.5])
    assert metrics["n_nonzero"] == 4
    assert metrics["acc2_non0"] == 1
    assert metrics["acc2_has0"] == 0.8
    assert metrics["acc7"] == 0.8
    assert metrics["mae"] == 0.2
    # Clipping belongs only to discrete accuracy, not MAE or correlation.
    clipped = mosei_metrics([-3, 3], [-9, 9])
    assert clipped["acc7"] == 1
    assert clipped["mae"] == 6
    assert clipped["pearson_r"] == pytest.approx(1)


def test_mosei_degenerate_cases_remain_valid_json():
    metrics = mosei_metrics([0, 0], [0, 0])
    assert metrics["pearson_r"] is None
    assert metrics["acc2_non0"] is metrics["f1_non0"] is None
    assert metrics["acc2_has0"] == 1
    assert mosei_metrics([-1, 0, 1] * 100, [0.7777614750478565] * 300)["pearson_r"] is None
    json.dumps(metrics, allow_nan=False)
    for truth, prediction in (([], []), ([0], [0, 1]), ([np.nan], [0]), ([4], [0])):
        with pytest.raises(ValueError):
            mosei_metrics(truth, prediction)


def test_seven_way_expected_score_is_not_argmax_score():
    p = [[0, 0, 0.4, 0.5, 0, 0, 0.1]]
    scores = class_scores(p)
    assert scores["argmax"][0] == 0
    assert scores["expected"][0] == pytest.approx(-0.1)
    with pytest.raises(ValueError):
        class_scores([[0.1] * 7])


def test_training_scaler_cannot_see_validation_distribution():
    train, valid, mean, scale = standardize_from_training(
        np.array([[1, 2], [3, 2]], dtype=np.float32),
        np.array([[100, 2]], dtype=np.float32),
    )
    np.testing.assert_equal(mean, [2, 2])
    np.testing.assert_equal(train[:, 0], [-1, 1])
    assert valid[0, 0] == 98
    assert np.isfinite(valid).all()


def test_development_manifest_rejects_test_and_duplicate_ids(tmp_path):
    path = tmp_path / "development.jsonl"
    for rows in (
        [{"id": "a", "split": "train"}, {"id": "b", "split": "test"}],
        [{"id": "a", "split": "train"}, {"id": "a", "split": "valid"}],
    ):
        path.write_text("\n".join(map(json.dumps, rows)))
        with pytest.raises(ValueError):
            development_records(path)


def test_quarantine_only_removes_training_without_label_selection():
    rows = [
        {"id": "a/0", "video_id": "a", "split": "train", "official_fold_member": True, "audio_sha256": "shared"},
        {"id": "a/1", "video_id": "a", "split": "train", "official_fold_member": True, "audio_sha256": "differently_encoded_neighbor"},
        {"id": "b/0", "video_id": "b", "split": "test", "official_fold_member": True, "audio_sha256": "shared"},
        {"id": "c/0", "video_id": "c", "split": "train", "official_fold_member": False, "audio_sha256": "other"},
        {"id": "d/0", "video_id": "d", "split": "valid", "official_fold_member": True, "audio_sha256": "valid"},
    ]
    report = {
        "cross_split_identical_wav_groups": [[("a/0", "train"), ("b/0", "test")]],
        "video_split_overlaps": {"train/valid": [], "train/test": [], "valid/test": []},
        "errors": [], "development_integrity_passed": False,
    }
    kept = quarantine_training_conflicts(rows, report)
    assert [r["id"] for r in kept] == ["b/0", "d/0"]
    assert report["development_integrity_passed"]
    assert len(rows) == 5
    assert not report["development_view"]["selection_uses_emotion_or_sentiment_labels"]


def test_csv_and_official_fold_validation(tmp_path):
    csv_path = tmp_path / "label.csv"
    csv_path.write_text("video_id,clip_id,text,label,mode\na,0,hello,1,train\na,0,hello,1,train\n")
    with pytest.raises(ValueError, match="Duplicate"):
        read_mosei_csv(csv_path)
    folds = tmp_path / "folds.py"
    folds.write_text("standard_train_fold=['a']\nstandard_valid_fold=['a']\nstandard_test_fold=['b']\n")
    with pytest.raises(ValueError, match="overlap"):
        read_official_folds(folds)
    folds.write_text("standard_train_fold=print('must not execute')\n")
    with pytest.raises(ValueError):
        read_official_folds(folds)


def test_duplicate_check_tolerates_gain_and_detects_silence():
    waveform = np.random.default_rng(42).normal(size=1000)
    score, lag = audio_similarity(waveform, waveform * 0.3)
    assert score == pytest.approx(1)
    assert lag == 0
    assert audio_similarity(np.zeros(100), np.zeros(100)) == (None, None)


def test_media_wer_handles_punctuation_and_insertions():
    assert word_error_rate("&gt;&gt; Hello, world!", "hello world") == 0
    assert word_error_rate("hello world", "hello") == .5
    assert word_error_rate("", "hello") is None


def test_legacy_mosei_loader_rejects_wrong_task_and_incomplete_head(tiny_model):
    from src.evaluate_mosei_checkpoint import load_legacy_head
    model = tiny_model()
    with pytest.raises(ValueError, match="seven-class"):
        load_legacy_head(model, {"num_labels": 4})
    # Only absent encoder keys are allowed; silently missing fusion keys is not.
    with pytest.raises(ValueError, match="Incompatible"):
        load_legacy_head(model, {"num_labels": 7, "fusion_head_output_type": "min", "trainable_state_dict": {}})
