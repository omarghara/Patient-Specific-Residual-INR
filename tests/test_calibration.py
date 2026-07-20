"""Tests for reference-free prior/k-space scale calibration."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.calibration import prior_scale_from_kspace, real_least_squares_scale
from presinr.forward import CartesianSense


def test_real_least_squares_scale_recovers_complex_data_scale():
    torch.manual_seed(4)
    prediction = torch.randn(3, 5, dtype=torch.complex64)
    target = 2.75 * prediction

    scale = real_least_squares_scale(prediction, target)

    assert scale == pytest.approx(2.75, abs=1e-6)


def test_real_scale_respects_mask_and_nonnegative_constraint():
    prediction = torch.tensor([1 + 1j, 2 - 1j, 3 + 0j])
    target = torch.tensor([4 + 4j, -100 + 50j, 12 + 0j])
    weights = torch.tensor([1.0, 0.0, 1.0])

    assert real_least_squares_scale(
        prediction, target, weights=weights
    ) == pytest.approx(4.0, abs=1e-6)
    assert real_least_squares_scale(
        prediction, -target, weights=weights, nonnegative=True
    ) == pytest.approx(0.0, abs=1e-7)


def test_prior_scale_uses_acquired_kspace_only():
    shape = (6, 8)
    prior = torch.linspace(0.1, 1.0, shape[0] * shape[1]).reshape(shape)
    phase = torch.linspace(-0.5, 0.5, shape[0] * shape[1]).reshape(shape)
    mask = torch.zeros(shape)
    mask[:, ::2] = 1
    operator = CartesianSense(
        torch.ones(2, *shape, dtype=torch.complex64), mask
    )
    candidate = torch.polar(prior, phase)
    measured = 1.8 * operator(candidate)
    measured[:, :, 1::2] = 1000  # ignored because these positions are unmeasured

    scale = prior_scale_from_kspace(prior, phase, operator, measured)

    assert scale == pytest.approx(1.8, abs=1e-5)


def test_zero_energy_calibration_is_rejected():
    with pytest.raises(ValueError, match="zero-energy"):
        real_least_squares_scale(torch.zeros(4), torch.ones(4))

