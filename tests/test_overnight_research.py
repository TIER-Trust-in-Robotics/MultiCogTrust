"""Scientific boundary checks; no dataset, model download or device capture."""

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from research.overnight.context_sources import context_records, fit_context
from research.overnight.dimensions import covariance_fit, prediction, resume_encoding, weights
from research.overnight.happiness_gate import gated_labels


def row(uid, speaker, start, end, text, code="neu", dialog="d", session=4):
    return {"utterance_id": uid, "speaker": speaker, "dialog_start_sec": start,
            "dialog_end_sec": end, "text": text, "label_code": code,
            "dialog_id": dialog, "session": session}


def test_other_context_is_complete_and_in_the_same_dialog():
    rows = [row("safe", "M", 0, 1, "finished"),
            row("overlap", "M", 2, 7, "not finished"),
            row("different", "M", 0, 4, "wrong dialog", dialog="elsewhere"),
            row("target", "F", 5, 6, "current")]
    record = next(item for item in context_records(rows, "other", "ref", {}, {"neu": 3})
                  if item["utterance_id"] == "target")
    assert record["context_ids"] == ["safe"]
    assert record["parts"] == ["finished", "current"]


def test_hri_uses_exact_other_text_and_asr_for_own_lines_including_unscored():
    rows = [row("own", "F", 0, 1, "own reference", code="xxx"),
            row("other", "M", 1, 2, "exact other", code="fru"),
            row("target", "F", 3, 4, "target reference")]
    records = list(context_records(rows, "ownother", "hri",
                   {"own": "own ASR", "other": "other ASR", "target": "current ASR"}, {"neu": 3}))
    assert len(records) == 1
    assert records[0]["parts"] == ["own ASR", "exact other", "current ASR"]
    assert records[0]["context_ids"] == ["own", "other"]


def test_empty_current_asr_abstains_and_test_session_is_rejected():
    assert list(context_records([row("x", "F", 0, 1, "reference")], "other", "hri", {}, {"neu": 3})) == []
    with pytest.raises(ValueError, match="development"):
        list(context_records([row("x", "F", 0, 1, "held out", session=5)], "other", "ref", {}, {"neu": 3}))


def test_own2_keeps_two_latest_completed_own_lines():
    rows = [row("first", "F", 0, 1, "one"), row("second", "F", 1, 2, "two"),
            row("other", "M", 2, 3, "other"), row("third", "F", 3, 4, "three"),
            row("target", "F", 5, 6, "four")]
    record = next(item for item in context_records(rows, "own2", "ref", {}, {"neu": 3})
                  if item["utterance_id"] == "target")
    assert record["context_ids"] == ["second", "third"]


def test_long_context_drops_oldest_lines_and_preserves_current():
    tokenizer = lambda text: {"input_ids": list(range(len(text.split()) + 2))}
    record = {"parts": ["very old context", "prior", "current words"], "context_ids": ["old", "prior"]}
    fitted = fit_context(record, tokenizer, max_tokens=6)
    assert fitted["encoded_context_ids"] == ["prior"]
    assert fitted["encoded_text"] == "prior [SEP] current words"
    assert not fitted["current_truncated"]
    assert record["context_ids"] == ["old", "prior"]
    assert fit_context({"parts": ["one two three four five"], "context_ids": []}, tokenizer, 3)["current_truncated"]


@pytest.mark.parametrize("regularization", [0.0001, 0.1, 1.0])
def test_covariance_ridge_matches_independent_sklearn_fit(regularization):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(83, 7)) * np.arange(1, 8) + 10
    x[:, 2] = 4
    y = rng.normal(size=(83, 3)) + x[:, :3] @ rng.normal(size=(3, 3))
    held = rng.normal(size=(17, 7))
    fit = covariance_fit(x, y)
    actual = prediction(held, fit, weights(fit, regularization))
    scaler = StandardScaler().fit(x)
    reference = Ridge(alpha=len(x) * regularization).fit(scaler.transform(x), y)
    np.testing.assert_allclose(actual, reference.predict(scaler.transform(held)), atol=1e-9)


def test_each_dimension_can_use_a_different_regularization():
    rng = np.random.default_rng(8)
    x, y = rng.normal(size=(50, 4)), rng.normal(size=(50, 3))
    fit = covariance_fit(x, y)
    combined = weights(fit, np.array([.001, .1, 1.]))
    for j, value in enumerate([.001, .1, 1.]):
        np.testing.assert_allclose(combined[:, j], weights(fit, value)[:, j])


def test_happiness_gate_preserves_missing_vision_and_confident_base_labels():
    base = np.array([[.4, .1, .2, .3], [.9, .01, .04, .05], [.4, .1, .2, .3]])
    visual = np.tile([.01, .95, .02, .02], (3, 1))
    labels, changed = gated_labels(base, visual, [False, True, True], .8, .6)
    np.testing.assert_array_equal(labels, [0, 0, 1])
    np.testing.assert_array_equal(changed, [False, False, True])


def test_face_gate_cannot_override_to_other_emotions():
    base = np.array([[.1, .1, .4, .4], [.1, .8, .05, .05]])
    visual = np.array([[.95, .03, .01, .01], [.01, .96, .01, .02]])
    labels, changed = gated_labels(base, visual, [True, True], .1, 1.)
    np.testing.assert_array_equal(labels, base.argmax(1))
    assert not changed.any()


def test_face_gate_requires_aligned_four_class_inputs():
    with pytest.raises(ValueError, match="aligned"):
        gated_labels(np.ones((2, 4)), np.ones((1, 4)), [True, True], .8, .6)


def test_deleted_feature_matrix_restarts_even_with_completed_marker():
    contract = {"rows": 100, "revision": "pinned"}
    completed = {"contract": contract, "encoded": 100, "complete": True}
    assert resume_encoding(completed, contract, matrix_exists=False) == (0, False)
    assert resume_encoding(completed, contract, matrix_exists=True) == (100, True)


def test_partial_encoding_resumes_only_if_matrix_still_exists():
    contract = {"rows": 100}
    partial = {"contract": contract, "encoded": 32, "complete": False}
    assert resume_encoding(partial, contract, matrix_exists=True) == (32, False)
    assert resume_encoding(partial, contract, matrix_exists=False) == (0, False)
    assert resume_encoding(None, contract, matrix_exists=False) == (0, False)


def test_missing_matrix_does_not_erase_a_changed_source_contract():
    with pytest.raises(ValueError, match="source contract changed"):
        resume_encoding({"contract": {"revision": "old"}, "encoded": 100}, {"revision": "new"}, False)
