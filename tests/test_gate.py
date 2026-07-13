"""Tests for bounded, regularized residual gates."""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.models import PriorResidualINR, build_inr, make_coord_grid
from presinr.forward import CartesianSense
from presinr.recon import ImageFitConfig, ResidualFitConfig, fit_residual, fit_residual_image


class ConstantINR(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.as_tensor(values, dtype=torch.float32))

    def forward(self, coords):
        return self.values.expand(coords.shape[0], -1)


def test_bounded_gate_components_match_forward_composition():
    prior = ConstantINR([0.25])
    residual = ConstantINR([2.0, -2.0])
    gate = ConstantINR([0.0])  # sigmoid -> 0.5
    model = PriorResidualINR(prior, residual, gate, residual_bound=0.4)
    coords = make_coord_grid(2, 3)

    raw, effective, g = model.residual_components(coords)
    expected_raw = 0.4 * torch.tanh(torch.tensor([2.0, -2.0]))
    assert torch.allclose(raw, expected_raw.expand_as(raw))
    assert torch.allclose(g, torch.full_like(g, 0.5))
    assert torch.allclose(effective, 0.5 * raw)

    image = model(coords, (2, 3))
    assert torch.allclose(image.real.reshape(-1), 0.25 + effective[:, 0])
    assert torch.allclose(image.imag.reshape(-1), effective[:, 1])


@pytest.mark.parametrize("bound", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_residual_bound_is_rejected(bound):
    with pytest.raises(ValueError, match="residual_bound"):
        PriorResidualINR(ConstantINR([0.0]), ConstantINR([0.0, 0.0]), residual_bound=bound)


def test_regularized_image_gate_requires_bound_and_raw_residual_penalty():
    prior = build_inr("siren", out_features=1, hidden_features=8, hidden_layers=1)
    residual = build_inr("siren", out_features=1, hidden_features=8, hidden_layers=1)
    gate = build_inr("siren", out_features=1, hidden_features=8, hidden_layers=1)
    target = torch.zeros(4, 4)

    with pytest.raises(ValueError, match="scale degeneracy"):
        fit_residual_image(
            prior,
            residual,
            target,
            (4, 4),
            cfg=ImageFitConfig(iters=0, lambda_gate=1e-3),
            gate_inr=gate,
            verbose=False,
        )


def test_constrained_image_gate_reports_raw_effective_and_gate_maps():
    prior = build_inr("siren", out_features=1, hidden_features=8, hidden_layers=1)
    residual = build_inr("siren", out_features=1, hidden_features=8, hidden_layers=1)
    gate = build_inr("siren", out_features=1, hidden_features=8, hidden_layers=1)
    target = torch.zeros(4, 4)
    _, effective, hist = fit_residual_image(
        prior,
        residual,
        target,
        (4, 4),
        cfg=ImageFitConfig(
            iters=1,
            lr=0.0,
            lambda_gate=1e-3,
            lambda_raw_res=1e-3,
            lambda_gate_tv=1e-3,
            residual_bound=1.0,
        ),
        gate_inr=gate,
        verbose=False,
    )

    raw = hist["raw_residual_map"]
    g = hist["gate_map"]
    assert torch.allclose(effective, raw * g, atol=1e-7)
    assert len(hist["raw_res_l1"]) == 1
    assert len(hist["gate_tv"]) == 1


def test_kspace_gate_enforces_constraint_and_returns_maps():
    shape = (4, 4)
    op = CartesianSense(torch.ones(1, *shape, dtype=torch.complex64), torch.ones(shape))
    ksp = torch.zeros(1, *shape, dtype=torch.complex64)
    prior = build_inr("siren", out_features=1, hidden_features=8, hidden_layers=1)
    residual = build_inr("siren", out_features=2, hidden_features=8, hidden_layers=1)
    gate = build_inr("siren", out_features=1, hidden_features=8, hidden_layers=1)

    unconstrained = PriorResidualINR(prior, residual, gate)
    with pytest.raises(ValueError, match="scale degeneracy"):
        fit_residual(
            unconstrained,
            op,
            ksp,
            shape,
            cfg=ResidualFitConfig(iters=0),
            verbose=False,
        )

    constrained = PriorResidualINR(prior, residual, gate, residual_bound=1.0)
    result = fit_residual(
        constrained,
        op,
        ksp,
        shape,
        cfg=ResidualFitConfig(iters=1, lr=0.0, lambda_raw_res=1e-3),
        verbose=False,
    )
    assert result.residual.shape == (*shape, 2)
    assert result.gate.shape == shape
    assert result.history["effective_residual_map"].shape == (*shape, 2)
