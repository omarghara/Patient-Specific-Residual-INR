"""Tests for acquisition-normalized loss terms."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.losses import data_consistency


def test_data_consistency_averages_only_measured_samples():
    pred = torch.zeros(2, 2, 2, dtype=torch.complex64)
    meas = torch.full_like(pred, 10.0)
    mask = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    meas[:, 0, 0] = 1.0

    assert data_consistency(pred, meas, mask) == pytest.approx(1.0)


def test_data_consistency_scale_is_invariant_to_mask_density_and_coils():
    for coils in (1, 3):
        pred = torch.zeros(coils, 4, 4, dtype=torch.complex64)
        for measured in (1, 8, 16):
            mask = torch.zeros(4, 4)
            mask.reshape(-1)[:measured] = 1.0
            meas = mask.expand(coils, -1, -1).to(torch.complex64)
            assert data_consistency(pred, meas, mask) == pytest.approx(1.0)


def test_data_consistency_rejects_empty_mask():
    y = torch.zeros(1, 2, 2, dtype=torch.complex64)
    assert torch.isnan(data_consistency(y, y, torch.zeros(2, 2)))
