"""Protect reference fidelity and evaluation boundaries."""

import io
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.mmsa_features import NumpyReleaseUnpickler, ReferenceFeatures
from src.models.mmsa_reference.lmf import LMF
from src.models.mmsa_reference.transformer import buffered_future_mask
from src.train_mmsa_reference import evaluate, optimizer_for

ROOT = Path(__file__).resolve().parents[1]


def test_release_unpickler_rejects_executable_globals():
    payload = pickle.dumps(Path("should_not_be_constructed"))
    with pytest.raises(pickle.UnpicklingError, match="Unapproved pickle global"):
        NumpyReleaseUnpickler(io.BytesIO(payload)).load()


def test_release_numpy_arrays_survive_restricted_loader():
    array = np.array([[1., 2.]], dtype=np.float32)
    restored = NumpyReleaseUnpickler(io.BytesIO(pickle.dumps(array, protocol=4))).load()
    np.testing.assert_array_equal(restored, array)


def test_test_feature_loader_is_rejected_before_filesystem_read(tmp_path):
    with pytest.raises(ValueError, match="only train/valid"):
        ReferenceFeatures(tmp_path / "absent", "test", False)


def test_lmf_reference_preserves_upstream_optimizer_omissions():
    config = json.loads((ROOT / "configs/week1/mosei_lmf_seed42.json").read_text())
    model = LMF(SimpleNamespace(**config))
    optimizer = optimizer_for(model, config)
    included = {id(p) for group in optimizer.param_groups for p in group["params"]}
    omitted = {name for name, p in model.named_parameters() if id(p) not in included}
    assert omitted == {"fusion_weights", "fusion_bias"}
    assert optimizer.param_groups[0]["lr"] == .001
    assert optimizer.param_groups[0]["weight_decay"] == .001


def test_validation_predictions_are_in_dataset_order_when_loader_shuffles():
    class Echo(torch.nn.Module):
        def forward(self, text, audio, vision):
            return {"M": text[:, :1]}

    scores = torch.tensor([[2.], [-1.], [0.], [1.], [-2.]])
    truth = scores + .2
    dataset = TensorDataset(scores, scores, scores, truth, torch.arange(5))
    loss, metrics, predicted, actual = evaluate(Echo(), DataLoader(dataset, batch_size=2, shuffle=True), "cpu")
    np.testing.assert_array_equal(predicted, scores.numpy().ravel())
    np.testing.assert_array_equal(actual, truth.numpy().ravel())
    assert metrics["n"] == 5
    assert loss == .2


def test_future_mask_values_and_device_match_reference():
    query, key = torch.zeros(2, 1, 4), torch.zeros(4, 1, 4)
    mask = buffered_future_mask(query, key)
    assert mask.device == query.device
    np.testing.assert_array_equal(torch.isneginf(mask).numpy(), [[False, False, False, True], [False, False, False, False]])
