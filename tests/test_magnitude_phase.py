"""Tests for phase-aware k-space INR variants."""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.forward import CartesianSense
from presinr.losses import phase_tv_2d
from presinr.models import (
    CurrentMagnitudePhaseINR,
    PriorMagnitudePhaseINR,
    build_inr,
    make_coord_grid,
)
from presinr.recon import (
    KspaceFitConfig,
    MagnitudePhaseFitConfig,
    PhaseFitConfig,
    fit_current_only,
    fit_magnitude_phase_residual,
    fit_phase_inr,
)


class ConstantINR(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.as_tensor(values, dtype=torch.float32))

    def forward(self, coords):
        return self.values.expand(coords.shape[0], -1)


def test_magnitude_phase_composition_separates_change_and_phase():
    model = PriorMagnitudePhaseINR(
        ConstantINR([0.5]),
        ConstantINR([0.1]),
        ConstantINR([torch.pi / 2]),
    )
    coords = make_coord_grid(2, 3)
    prior, delta, magnitude, phase = model.components(coords)
    image = model(coords, (2, 3))

    assert torch.allclose(prior, torch.full_like(prior, 0.5))
    assert torch.allclose(delta, torch.full_like(delta, 0.1))
    assert torch.allclose(magnitude, torch.full_like(magnitude, 0.6))
    assert torch.allclose(phase, torch.full_like(phase, torch.pi / 2))
    assert torch.allclose(image.real, torch.zeros_like(image.real), atol=1e-6)
    assert torch.allclose(image.imag, torch.full_like(image.imag, 0.6), atol=1e-6)


def test_magnitude_phase_composition_applies_fixed_prior_scale():
    model = PriorMagnitudePhaseINR(
        ConstantINR([0.5]),
        ConstantINR([0.1]),
        ConstantINR([0.0]),
        prior_scale=2.0,
    )
    scaled_prior, delta, magnitude, _ = model.components(make_coord_grid(2, 2))

    assert model.prior_scale == pytest.approx(2.0, abs=1e-6)
    assert torch.allclose(scaled_prior, torch.ones_like(scaled_prior))
    assert torch.allclose(delta, torch.full_like(delta, 0.1))
    assert torch.allclose(magnitude, torch.full_like(magnitude, 1.1))
    assert not model.log_prior_scale.requires_grad


def test_invalid_prior_scale_is_rejected():
    with pytest.raises(ValueError, match="prior_scale"):
        PriorMagnitudePhaseINR(
            ConstantINR([0.5]),
            ConstantINR([0.0]),
            ConstantINR([0.0]),
            prior_scale=0.0,
        )


def test_current_magnitude_phase_has_matching_nonnegative_composition():
    model = CurrentMagnitudePhaseINR(
        ConstantINR([0.7]), ConstantINR([torch.pi / 2])
    )
    coords = make_coord_grid(2, 3)
    raw, magnitude, phase = model.components(coords)
    image = model(coords, (2, 3))

    assert torch.allclose(raw, torch.full_like(raw, 0.7))
    assert torch.allclose(magnitude, torch.full_like(magnitude, 0.7))
    assert torch.allclose(phase, torch.full_like(phase, torch.pi / 2))
    assert torch.allclose(image.real, torch.zeros_like(image.real), atol=1e-6)
    assert torch.allclose(image.imag, torch.full_like(image.imag, 0.7), atol=1e-6)


def test_magnitude_is_nonnegative_and_bound_is_validated():
    model = PriorMagnitudePhaseINR(
        ConstantINR([0.1]), ConstantINR([-0.5]), ConstantINR([0.0])
    )
    _, _, magnitude, _ = model.components(make_coord_grid(2, 2))
    assert torch.count_nonzero(magnitude) == 0

    with pytest.raises(ValueError, match="magnitude_residual_bound"):
        PriorMagnitudePhaseINR(
            ConstantINR([0.0]),
            ConstantINR([0.0]),
            ConstantINR([0.0]),
            magnitude_residual_bound=0.0,
        )


def test_phase_tv_is_invariant_to_full_wraps():
    phase = torch.tensor([[0.0, 2 * torch.pi], [-2 * torch.pi, 0.0]])
    assert phase_tv_2d(phase) == pytest.approx(0.0, abs=1e-7)


def test_phase_initializer_uses_circular_difference():
    phase_inr = ConstantINR([torch.pi])
    target = torch.full((2, 2), -torch.pi)
    history = fit_phase_inr(
        phase_inr,
        target,
        cfg=PhaseFitConfig(iters=1, lr=0.0),
        verbose=False,
    )
    assert history["loss"][0] == pytest.approx(0.0, abs=1e-7)


def test_current_only_and_magnitude_phase_fit_return_expected_maps():
    shape = (4, 4)
    op = CartesianSense(
        torch.ones(1, *shape, dtype=torch.complex64), torch.ones(shape)
    )
    ksp = torch.zeros(1, *shape, dtype=torch.complex64)

    current = build_inr(
        "siren", out_features=2, hidden_features=8, hidden_layers=1
    )
    current_result = fit_current_only(
        current,
        op,
        ksp,
        shape,
        cfg=KspaceFitConfig(iters=1, lr=0.0),
        verbose=False,
    )
    assert current_result.recon.shape == shape
    assert len(current_result.history["dc"]) == 1

    model = PriorMagnitudePhaseINR(
        build_inr("siren", out_features=1, hidden_features=8, hidden_layers=1),
        build_inr("siren", out_features=1, hidden_features=8, hidden_layers=1),
        build_inr("siren", out_features=1, hidden_features=8, hidden_layers=1),
    )
    result = fit_magnitude_phase_residual(
        model,
        op,
        ksp,
        shape,
        cfg=MagnitudePhaseFitConfig(iters=1, lr=0.0),
        verbose=False,
    )
    assert result.recon.shape == shape
    assert result.magnitude_residual.shape == shape
    assert result.phase.shape == shape
    assert result.prior_scale == pytest.approx(1.0)
    assert result.history["magnitude_map"].shape == shape
    assert not any(parameter.requires_grad for parameter in model.prior_inr.parameters())
