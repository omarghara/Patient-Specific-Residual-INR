"""Tests for coordinate-network representations."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.models import FourierSiren, build_inr, make_coord_grid


def test_fourier_siren_is_deterministic_for_a_fixed_seed():
    first = FourierSiren(
        hidden_features=8, hidden_layers=1, mapping_size=4, sigma=1.5, seed=7
    )
    second = FourierSiren(
        hidden_features=8, hidden_layers=1, mapping_size=4, sigma=1.5, seed=7
    )

    assert torch.equal(first.B, second.B)
    assert first.encode(make_coord_grid(2, 3)).shape == (6, 8)


def test_build_inr_accepts_fourier_siren_alias():
    model = build_inr(
        "fourier_siren",
        out_features=1,
        hidden_features=8,
        hidden_layers=1,
        mapping_size=4,
        sigma=1.0,
    )
    assert model(make_coord_grid(2, 2)).shape == (4, 1)


@pytest.mark.parametrize("mapping_size,sigma", [(0, 1.0), (4, 0.0)])
def test_fourier_siren_validates_encoding(mapping_size, sigma):
    with pytest.raises(ValueError):
        FourierSiren(mapping_size=mapping_size, sigma=sigma)

