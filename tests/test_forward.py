"""Correctness tests for the forward model and INR plumbing."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.forward import CartesianSense
from presinr.models import PriorResidualINR, build_inr, make_coord_grid


def _op(n=32, nc=4, seed=0):
    torch.manual_seed(seed)
    mps = torch.randn(nc, n, n, dtype=torch.complex64)
    mask = (torch.rand(n, n) > 0.5).float()
    return CartesianSense(mps, mask), n, nc


def test_shapes():
    op, n, nc = _op()
    x = torch.randn(n, n, dtype=torch.complex64)
    y = op(x)
    assert y.shape == (nc, n, n)
    assert op.adjoint(y).shape == (n, n)


def test_adjoint_dot_product():
    """<A x, y> == <x, A^H y> for the linear SENSE operator."""
    op, n, nc = _op()
    x = torch.randn(n, n, dtype=torch.complex64)
    y = torch.randn(nc, n, n, dtype=torch.complex64)
    lhs = torch.vdot(op(x).reshape(-1), y.reshape(-1))
    rhs = torch.vdot(x.reshape(-1), op.adjoint(y).reshape(-1))
    assert torch.allclose(lhs, rhs, atol=1e-4), (lhs, rhs)


def test_composition_forward_is_complex():
    n = 16
    prior = build_inr("siren", out_features=1, hidden_features=32, hidden_layers=2)
    resid = build_inr("siren", out_features=2, hidden_features=32, hidden_layers=2)
    model = PriorResidualINR(prior, resid)
    coords = make_coord_grid(n, n)
    x = model(coords, (n, n))
    assert x.shape == (n, n)
    assert x.is_complex()


def test_gradients_flow_to_residual_only():
    n = 16
    prior = build_inr("siren", out_features=1, hidden_features=32, hidden_layers=2)
    resid = build_inr("siren", out_features=2, hidden_features=32, hidden_layers=2)
    model = PriorResidualINR(prior, resid)
    model.freeze_prior()
    coords = make_coord_grid(n, n)
    x = model(coords, (n, n))
    x.abs().sum().backward()
    assert all(p.grad is None for p in model.prior_inr.parameters())
    assert any(p.grad is not None for p in model.residual_inr.parameters())


if __name__ == "__main__":
    test_shapes()
    test_adjoint_dot_product()
    test_composition_forward_is_complex()
    test_gradients_flow_to_residual_only()
    print("all tests passed")
