"""Tests for the LAPS acceleration-experiment helpers."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.baselines.laps_nerp import (
    CenterPaddedSense,
    LapsNerpConfig,
    LapsNerpStageConfig,
    LapsSiren,
    fit_laps_nerp,
)
from presinr.sampling import (
    cartesian_line_holdout,
    laps_retrospective_1d_mask,
    laps_retrospective_2d_mask,
)


def test_cartesian_line_holdout_is_deterministic_disjoint_and_protects_center():
    acquired = torch.zeros(12, 32)
    acquired[:, [1, 4, 7, 13, 14, 15, 16, 17, 18, 23, 27, 30]] = 1
    train, validation, info = cartesian_line_holdout(
        acquired,
        validation_fraction=0.25,
        seed=9,
        phase_encode_dim=1,
        protected_center_lines=6,
    )
    again_train, again_validation, _ = cartesian_line_holdout(
        acquired,
        validation_fraction=0.25,
        seed=9,
        phase_encode_dim=1,
        protected_center_lines=6,
    )

    assert torch.equal(train, again_train)
    assert torch.equal(validation, again_validation)
    assert not torch.any(train.bool() & validation.bool())
    assert torch.equal(train.bool() | validation.bool(), acquired.bool())
    assert info.acquired_lines == 12
    assert info.training_lines == 9
    assert info.validation_lines == 3
    assert info.protected_center_lines == 6
    assert info.validation_fraction == 0.25
    assert torch.all(train[:, 13:19] == acquired[:, 13:19])


def test_cartesian_line_holdout_supports_row_encoding_and_missing_samples():
    acquired = torch.zeros(24, 10, dtype=torch.bool)
    acquired[[1, 5, 9, 10, 11, 12, 13, 17, 21], :] = True
    acquired[5, 3] = False
    train, validation, info = cartesian_line_holdout(
        acquired,
        validation_fraction=0.2,
        seed=4,
        phase_encode_dim=0,
        protected_center_lines=5,
    )

    assert info.phase_encode_dim == 0
    assert not torch.any(train & validation)
    assert torch.equal(train | validation, acquired)
    assert not train[5, 3] and not validation[5, 3]
    selected_rows = validation.any(dim=1)
    assert torch.equal(validation[selected_rows], acquired[selected_rows])


def test_cartesian_line_holdout_rejects_impossible_single_line_split():
    acquired = torch.zeros(8, 8)
    acquired[:, 4] = 1
    with pytest.raises(ValueError, match="at least two"):
        cartesian_line_holdout(acquired, phase_encode_dim=1)


def test_laps_mask_is_deterministic_subset_with_reported_acceleration():
    acquired = torch.ones(32, 64)
    first, info = laps_retrospective_1d_mask(
        acquired, 4, seed=17, phase_encode_dim=1
    )
    second, _ = laps_retrospective_1d_mask(
        acquired, 4, seed=17, phase_encode_dim=1
    )

    assert torch.equal(first, second)
    assert not torch.any(first.bool() & ~acquired.bool())
    # The released LAPS symmetric ACS trim keeps one extra physical center line
    # when the requested trim is odd (16 output lines here, bookkeeping target 16).
    assert int(first.any(dim=0).sum()) == info.output_lines
    assert info.center_lines in (15, 16)
    assert info.effective_acceleration == 64 / info.output_lines


def test_laps_mask_default_axis_accepts_boolean_mask():
    acquired = torch.ones(32, 64, dtype=torch.bool)
    acquired[:, ::5] = False
    output, info = laps_retrospective_1d_mask(acquired, 4, seed=5)
    assert output.dtype == torch.bool
    assert info.phase_encode_dim == 1
    assert not torch.any(output & ~acquired)


def test_laps_mask_respects_already_missing_lines_and_axis():
    acquired = torch.ones(64, 24)
    acquired[::3, :] = 0
    output, info = laps_retrospective_1d_mask(
        acquired, 4, seed=2, phase_encode_dim=0
    )

    assert not torch.any(output.bool() & ~acquired.bool())
    assert info.phase_encode_dim == 0
    assert int(output.any(dim=1).sum()) == info.output_lines


def test_laps_2d_mask_is_deterministic_subset_with_bounded_acceleration():
    acquired = torch.ones(128, 128)
    first, info = laps_retrospective_2d_mask(acquired, 10, seed=23)
    second, _ = laps_retrospective_2d_mask(acquired, 10, seed=23)

    assert torch.equal(first, second)
    assert not torch.any(first.bool() & ~acquired.bool())
    assert info.phase_encode_dim == -1
    assert info.output_lines == int(first.sum())
    assert info.effective_acceleration == info.bounded_points / info.output_lines
    assert info.output_lines < info.input_lines


def test_center_padded_sense_has_matching_adjoint():
    torch.manual_seed(3)
    native_shape = (8, 6)
    stored_shape = (8, 8)
    mps = torch.randn(2, *native_shape, dtype=torch.complex64)
    mask = (torch.rand(native_shape) > 0.3).float()
    operator = CenterPaddedSense(mps, mask, stored_shape)
    image = torch.randn(*stored_shape, dtype=torch.complex64)
    kspace = torch.randn(2, *native_shape, dtype=torch.complex64)

    lhs = torch.vdot(operator(image).reshape(-1), kspace.reshape(-1))
    rhs = torch.vdot(image.reshape(-1), operator.adjoint(kspace).reshape(-1))
    assert operator(image).shape == (2, *native_shape)
    assert operator.adjoint(kspace).shape == stored_shape
    assert torch.allclose(lhs, rhs, atol=1e-4)


def test_release_default_nerp_parameter_count():
    model = LapsSiren(LapsNerpConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_839_618


def test_tiny_laps_nerp_fit_returns_both_output_conventions():
    torch.manual_seed(4)
    native_shape = (6, 4)
    stored_shape = (6, 6)
    mps = torch.ones(1, *native_shape, dtype=torch.complex64)
    operator = CenterPaddedSense(mps, torch.ones(native_shape), stored_shape)
    target = torch.randn(*stored_shape, dtype=torch.complex64)
    kspace = operator(target)
    prior = target.abs()
    short = LapsNerpStageConfig(
        max_iter=2,
        lr=1e-3,
        min_iterations=10,
        patience=2,
    )
    config = LapsNerpConfig(
        embedding_size=4,
        network_depth=3,
        network_width=8,
        cg_iters=2,
        prior_stage=short,
        kspace_stage=short,
    )

    result = fit_laps_nerp(
        prior,
        operator,
        kspace,
        config=config,
        device=torch.device("cpu"),
        verbose=False,
    )

    assert result.recon_released.shape == stored_shape
    assert result.recon_scaled.shape == stored_shape
    assert result.recon_released.is_complex()
    assert result.recon_scaled.is_complex()
    assert result.scale_matrix.shape == (2, 2)
    assert len(result.prior_history["loss"]) == 2
    assert len(result.kspace_history["loss"]) == 2
    assert torch.isfinite(result.recon_scaled).all()
