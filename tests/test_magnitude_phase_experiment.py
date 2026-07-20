"""Focused tests for reusable magnitude/phase experiment utilities."""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.experiments.magnitude_phase import (
    MagnitudePhaseTrainConfig,
    build_scalar_inr,
    relative_kspace_error,
    train_magnitude_phase,
    zero_last_linear_,
)
from presinr.forward import CartesianSense
from presinr.models import CurrentMagnitudePhaseINR, PriorMagnitudePhaseINR
from presinr.models.inr import make_coord_grid


class ConstantINR(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = nn.Parameter(torch.tensor([value], dtype=torch.float32))

    def forward(self, coordinates):
        return self.value.expand(coordinates.shape[0], 1)


def _full_operator(shape=(4, 4)):
    return CartesianSense(
        torch.ones(1, *shape, dtype=torch.complex64), torch.ones(shape)
    )


@pytest.mark.parametrize("kind", ["siren", "fourier_siren"])
def test_scalar_builder_is_locally_deterministic_and_can_start_at_zero(kind):
    kwargs = {"hidden_features": 8, "hidden_layers": 1}
    if kind == "fourier_siren":
        kwargs.update(mapping_size=4, sigma=1.5)

    torch.manual_seed(91)
    rng_before = torch.random.get_rng_state().clone()
    first = build_scalar_inr(kind, seed=7, zero_last=True, **kwargs)
    rng_after = torch.random.get_rng_state()
    second = build_scalar_inr(kind, seed=7, zero_last=True, **kwargs)
    coordinates = make_coord_grid(3, 2)

    assert torch.equal(rng_before, rng_after)
    assert all(
        torch.equal(first.state_dict()[name], second.state_dict()[name])
        for name in first.state_dict()
    )
    assert torch.count_nonzero(first(coordinates)) == 0
    assert first(coordinates).shape == (6, 1)


def test_zero_last_linear_rejects_modules_without_a_linear_layer():
    with pytest.raises(ValueError, match="no nn.Linear"):
        zero_last_linear_(nn.ReLU())


def test_relative_kspace_error_is_reference_free_and_masked():
    shape = (4, 4)
    mask = torch.zeros(shape)
    mask[:, ::2] = 1
    operator = CartesianSense(
        torch.ones(1, *shape, dtype=torch.complex64), mask
    )
    target_image = torch.randn(*shape, dtype=torch.complex64)
    measured = operator(target_image)
    measured_corrupted = measured.clone()
    measured_corrupted[..., :, 1::2] = 1000

    assert relative_kspace_error(target_image, operator, measured_corrupted) \
        == pytest.approx(0.0, abs=1e-7)
    assert relative_kspace_error(
        torch.zeros_like(target_image), operator, measured_corrupted
    ) == pytest.approx(1.0, abs=1e-7)


def test_current_only_training_uses_post_update_validation_and_no_delta_penalty():
    shape = (4, 4)
    operator = _full_operator(shape)
    target = torch.ones(shape, dtype=torch.complex64)
    measured = operator(target)
    magnitude = ConstantINR(0.25)
    phase = ConstantINR(0.0)
    model = CurrentMagnitudePhaseINR(magnitude, phase)
    initial_error = relative_kspace_error(model(make_coord_grid(*shape), shape), operator, measured)

    result = train_magnitude_phase(
        model,
        operator,
        measured,
        shape,
        MagnitudePhaseTrainConfig(
            iterations=3,
            magnitude_lr=0.1,
            phase_lr=0.1,
            lambda_delta_l1=100.0,
            lambda_delta_tv=100.0,
            grad_clip_norm=1.0,
            eval_every=1,
            fixed_phase=True,
        ),
        validation_operator=operator,
        validation_kspace=measured,
    )

    validation_errors = [row["validation_error"] for row in result.history]
    assert result.mode == "current_only"
    assert result.delta is None and result.final_delta is None
    assert all(row["delta_l1_before_update"] == 0.0 for row in result.history)
    assert all(row["delta_tv_before_update"] == 0.0 for row in result.history)
    assert result.history[0]["validation_error"] < initial_error
    assert result.best_validation_error == pytest.approx(min(validation_errors))
    assert result.best_iteration == result.history[validation_errors.index(min(validation_errors))]["iteration"]
    assert result.history[-1]["magnitude_lr"] < result.history[0]["magnitude_lr"]
    assert all(row["gradient_norm_before_clip"] is not None for row in result.history)
    assert result.recon.shape == shape
    assert result.magnitude.shape == shape
    assert result.phase.shape == shape
    assert result.iterations_completed == 3
    assert result.runtime_seconds >= 0
    assert 0 < result.trainable_parameters < result.total_parameters
    assert phase.value.item() == pytest.approx(0.0)
    assert not phase.value.requires_grad


def test_residual_training_saves_delta_and_final_checkpoint_without_validation():
    shape = (4, 4)
    operator = _full_operator(shape)
    target = torch.full(shape, 0.8, dtype=torch.complex64)
    measured = operator(target)
    prior = ConstantINR(0.5)
    delta = ConstantINR(0.1)
    phase = ConstantINR(0.0)
    model = PriorMagnitudePhaseINR(prior, delta, phase)

    result = train_magnitude_phase(
        model,
        operator,
        measured,
        shape,
        MagnitudePhaseTrainConfig(
            iterations=2,
            magnitude_lr=0.05,
            phase_lr=0.05,
            lambda_delta_l1=0.1,
            lambda_delta_tv=0.1,
            eval_every=1,
            fixed_phase=True,
        ),
    )

    assert result.mode == "prior_residual"
    assert result.delta is not None and result.delta.shape == shape
    assert result.final_delta is not None
    assert result.best_iteration == 2
    assert result.best_validation_error is None
    assert torch.equal(result.recon, result.final_recon)
    assert torch.equal(result.magnitude, result.final_magnitude)
    assert torch.equal(result.phase, result.final_phase)
    assert torch.equal(result.delta, result.final_delta)
    assert any(row["delta_l1_before_update"] > 0 for row in result.history)
    assert all(parameter.requires_grad is False for parameter in prior.parameters())


def test_best_validation_state_is_restored_when_the_final_iterate_is_worse():
    shape = (4, 4)
    operator = _full_operator(shape)
    train_measurements = operator(torch.ones(shape, dtype=torch.complex64))
    validation_measurements = operator(
        torch.full(shape, 0.2, dtype=torch.complex64)
    )
    model = CurrentMagnitudePhaseINR(ConstantINR(0.25), ConstantINR(0.0))

    result = train_magnitude_phase(
        model,
        operator,
        train_measurements,
        shape,
        MagnitudePhaseTrainConfig(
            iterations=4,
            magnitude_lr=0.1,
            phase_lr=0.1,
            min_lr_ratio=1.0,
            grad_clip_norm=None,
            eval_every=1,
            fixed_phase=True,
        ),
        validation_operator=operator,
        validation_kspace=validation_measurements,
    )

    errors = [row["validation_error"] for row in result.history]
    assert result.best_iteration == 1
    assert errors[0] == pytest.approx(min(errors))
    assert not torch.equal(result.recon, result.final_recon)
    restored = model(make_coord_grid(*shape), shape).detach().cpu()
    assert torch.allclose(restored, result.recon)


def test_validation_arguments_must_be_paired():
    shape = (4, 4)
    operator = _full_operator(shape)
    model = CurrentMagnitudePhaseINR(ConstantINR(0.5), ConstantINR(0.0))
    measured = operator(torch.ones(shape, dtype=torch.complex64))

    with pytest.raises(ValueError, match="must be supplied together"):
        train_magnitude_phase(
            model,
            operator,
            measured,
            shape,
            MagnitudePhaseTrainConfig(iterations=1),
            validation_operator=operator,
        )
