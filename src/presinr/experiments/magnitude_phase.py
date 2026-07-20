"""Reusable training utilities for magnitude/phase INR experiments.

The functions in this module deliberately contain no reference-image metrics.
Model selection is driven by acquired validation k-space, which makes the same
trainer suitable for development sweeps and reference-free deployment.  Both
the patient-prior residual model and a parameter-matched current-only baseline
share the optimization path.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..losses import data_consistency, phase_tv_2d, tv_2d
from ..models.composition import CurrentMagnitudePhaseINR, PriorMagnitudePhaseINR
from ..models.inr import build_inr, make_coord_grid


def zero_last_linear_(module: nn.Module) -> nn.Module:
    """Zero the terminal :class:`~torch.nn.Linear` layer in place.

    The traversal works for both a raw SIREN and the nested SIREN inside a
    Fourier-feature encoder.  It consumes no random numbers, so using it after
    deterministic construction preserves reproducibility.  A zero-output
    residual starts exactly from the scaled prior rather than adding a random
    anatomy-shaped perturbation.
    """

    last_linear = None
    for child in module.modules():
        if isinstance(child, nn.Linear):
            last_linear = child
    if last_linear is None:
        raise ValueError("module contains no nn.Linear layer to zero")
    with torch.no_grad():
        last_linear.weight.zero_()
        if last_linear.bias is not None:
            last_linear.bias.zero_()
    return module


def build_scalar_inr(
    kind: str = "siren",
    *,
    seed: Optional[int] = 0,
    zero_last: bool = False,
    **inr_kwargs,
) -> nn.Module:
    """Build a deterministic scalar SIREN or Fourier-SIREN.

    Args:
        kind: ``"siren"`` (also ``"raw"``) or ``"fourier_siren"`` (also
            ``"ff_siren"``).  The ReLU Fourier MLP is intentionally excluded
            so this is a controlled coordinate-encoding ablation.
        seed: Local construction seed.  The caller's global CPU RNG state is
            restored afterwards.  ``None`` uses the ambient RNG state.
        zero_last: If true, zero the output layer with
            :func:`zero_last_linear_`.
        **inr_kwargs: Architecture arguments accepted by
            :func:`presinr.models.build_inr`; ``out_features`` is fixed to one.

    Returns:
        A scalar-output coordinate network.
    """

    aliases = {
        "siren": "siren",
        "raw": "siren",
        "raw_siren": "siren",
        "fourier_siren": "fourier_siren",
        "ff_siren": "fourier_siren",
    }
    normalized = aliases.get(str(kind).lower())
    if normalized is None:
        raise ValueError(
            "scalar INR kind must be 'siren' or 'fourier_siren', "
            f"got {kind!r}"
        )
    if "out_features" in inr_kwargs and int(inr_kwargs["out_features"]) != 1:
        raise ValueError("build_scalar_inr always uses out_features=1")
    inr_kwargs = dict(inr_kwargs)
    inr_kwargs["out_features"] = 1
    if normalized == "fourier_siren":
        inr_kwargs["seed"] = seed

    def construct() -> nn.Module:
        return build_inr(normalized, **inr_kwargs)

    if seed is None:
        model = construct()
    else:
        # INRs are constructed on CPU. fork_rng makes their trainable weights
        # deterministic without perturbing the experiment's ambient RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            model = construct()
    if zero_last:
        zero_last_linear_(model)
    return model


def _expanded_mask(operator, target: torch.Tensor) -> Optional[torch.Tensor]:
    mask = getattr(operator, "mask", None)
    if mask is None:
        return None
    mask = torch.as_tensor(mask, device=target.device, dtype=target.real.dtype)
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(0)
    try:
        return mask.expand_as(target.real)
    except RuntimeError as error:
        raise ValueError(
            f"operator mask shape {tuple(mask.shape)} cannot broadcast to "
            f"k-space shape {tuple(target.shape)}"
        ) from error


@torch.no_grad()
def relative_kspace_error(
    reconstruction: torch.Tensor,
    operator,
    measured_kspace: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> float:
    """Return measured-sample relative k-space error without a reference image.

    The metric is ``||M(A x - y)||_2 / ||M y||_2``.  Values outside the
    operator's mask are ignored even when ``measured_kspace`` contains nonzero
    data there, which is important for explicit train/validation operators.
    """

    if not math.isfinite(float(eps)) or eps <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    prediction = operator(reconstruction)
    measured = torch.as_tensor(
        measured_kspace, device=prediction.device, dtype=prediction.dtype
    )
    if prediction.shape != measured.shape:
        raise ValueError(
            f"predicted and measured k-space must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(measured.shape)}"
        )
    mask = _expanded_mask(operator, measured)
    residual = prediction - measured
    target = measured
    if mask is not None:
        residual = residual * mask
        target = target * mask
    denominator = torch.linalg.vector_norm(target)
    if float(denominator) <= eps:
        raise ValueError("relative k-space error requires nonzero measured energy")
    return float(torch.linalg.vector_norm(residual) / denominator)


@dataclass(frozen=True)
class MagnitudePhaseTrainConfig:
    """Optimization settings shared by residual and current-only models."""

    iterations: int = 2000
    magnitude_lr: float = 3e-4
    phase_lr: float = 1e-4
    prior_scale_lr: Optional[float] = None
    lambda_delta_l1: float = 1e-3
    lambda_delta_tv: float = 0.0
    lambda_phase_tv: float = 0.0
    min_lr_ratio: float = 0.05
    grad_clip_norm: Optional[float] = 1.0
    eval_every: int = 50
    fixed_phase: bool = False


@dataclass
class MagnitudePhaseTrainResult:
    """Selected and final checkpoints from :func:`train_magnitude_phase`.

    ``recon``, ``magnitude``, ``phase``, ``delta``, and ``state_dict`` describe
    the lowest-validation-error post-update checkpoint.  When no validation
    operator is supplied they describe the fixed-stop final checkpoint instead.
    The ``final_*`` maps always preserve the raw last optimization iterate.
    All saved tensors and state-dict values reside on CPU.
    """

    mode: str
    recon: torch.Tensor
    magnitude: torch.Tensor
    phase: torch.Tensor
    delta: Optional[torch.Tensor]
    final_recon: torch.Tensor
    final_magnitude: torch.Tensor
    final_phase: torch.Tensor
    final_delta: Optional[torch.Tensor]
    state_dict: Dict[str, torch.Tensor]
    history: List[Dict[str, Any]]
    best_iteration: int
    best_validation_error: Optional[float]
    iterations_completed: int
    runtime_seconds: float
    trainable_parameters: int
    total_parameters: int


def _validate_config(config: MagnitudePhaseTrainConfig) -> None:
    if not isinstance(config.iterations, int) or config.iterations < 1:
        raise ValueError(f"iterations must be a positive integer, got {config.iterations}")
    if not isinstance(config.eval_every, int) or config.eval_every < 1:
        raise ValueError(f"eval_every must be a positive integer, got {config.eval_every}")
    for name in ("magnitude_lr", "phase_lr"):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
    if config.prior_scale_lr is not None and (
        not math.isfinite(float(config.prior_scale_lr))
        or config.prior_scale_lr <= 0
    ):
        raise ValueError(
            "prior_scale_lr must be finite and positive when provided, "
            f"got {config.prior_scale_lr}"
        )
    for name in ("lambda_delta_l1", "lambda_delta_tv", "lambda_phase_tv"):
        value = float(getattr(config, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative, got {value}")
    if (
        not math.isfinite(float(config.min_lr_ratio))
        or not 0 <= config.min_lr_ratio <= 1
    ):
        raise ValueError(
            f"min_lr_ratio must lie in [0, 1], got {config.min_lr_ratio}"
        )
    if config.grad_clip_norm is not None and (
        not math.isfinite(float(config.grad_clip_norm))
        or config.grad_clip_norm <= 0
    ):
        raise ValueError(
            "grad_clip_norm must be finite and positive when provided, "
            f"got {config.grad_clip_norm}"
        )


def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _copy_optional(value: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    return None if value is None else value.detach().cpu().clone()


def _render(
    model: nn.Module,
    coords: torch.Tensor,
    shape: Tuple[int, int],
    cached_prior: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(model, PriorMagnitudePhaseINR):
        _, delta, magnitude, phase = model.components(
            coords, prior_mag=cached_prior
        )
    else:
        _, magnitude, phase = model.components(coords)
        delta = None
    reconstruction = torch.polar(magnitude, phase).reshape(*shape)
    magnitude_map = magnitude.reshape(*shape)
    phase_map = phase.reshape(*shape)
    delta_map = None if delta is None else delta.reshape(*shape)
    return reconstruction, magnitude_map, phase_map, delta_map


def _snapshot_maps(maps):
    reconstruction, magnitude, phase, delta = maps
    return (
        reconstruction.detach().cpu().clone(),
        magnitude.detach().cpu().clone(),
        phase.detach().cpu().clone(),
        _copy_optional(delta),
    )


def train_magnitude_phase(
    model: nn.Module,
    train_operator,
    train_kspace: torch.Tensor,
    shape: Tuple[int, int],
    cfg: MagnitudePhaseTrainConfig = MagnitudePhaseTrainConfig(),
    *,
    validation_operator=None,
    validation_kspace: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    verbose: bool = False,
) -> MagnitudePhaseTrainResult:
    """Train a residual or current-only magnitude/phase INR.

    Training minimizes measured-sample complex k-space MSE.  A cosine learning
    rate schedule and optional gradient clipping are applied to every trainable
    branch.  Delta L1/TV terms are used only for
    :class:`PriorMagnitudePhaseINR`; they are exactly zero for the matched
    current-only baseline.  Phase TV is shared by both modes unless phase is
    fixed.

    Validation is optional but its operator and measurements must be supplied
    together.  Checkpoints are evaluated *after* each selected optimizer update,
    and the returned model is restored to the best validation state.  With no
    validation data, the fixed-stop final state is returned.
    """

    _validate_config(cfg)
    if not isinstance(model, (PriorMagnitudePhaseINR, CurrentMagnitudePhaseINR)):
        raise TypeError(
            "model must be PriorMagnitudePhaseINR or CurrentMagnitudePhaseINR"
        )
    if len(shape) != 2 or any(int(size) <= 0 for size in shape):
        raise ValueError(f"shape must contain two positive dimensions, got {shape}")
    shape = (int(shape[0]), int(shape[1]))
    has_validation = validation_operator is not None or validation_kspace is not None
    if has_validation and (
        validation_operator is None or validation_kspace is None
    ):
        raise ValueError(
            "validation_operator and validation_kspace must be supplied together"
        )

    device = torch.device(device or train_kspace.device)
    model = model.to(device)
    train_operator = train_operator.to(device)
    train_kspace = train_kspace.to(device)
    if has_validation:
        validation_operator = validation_operator.to(device)
        validation_kspace = validation_kspace.to(device)

    is_residual = isinstance(model, PriorMagnitudePhaseINR)
    mode = "prior_residual" if is_residual else "current_only"
    if is_residual:
        model.freeze_prior()
        magnitude_branch = model.magnitude_residual_inr
    else:
        magnitude_branch = model.magnitude_inr

    if cfg.fixed_phase:
        for parameter in model.phase_inr.parameters():
            parameter.requires_grad_(False)
        model.phase_inr.eval()

    magnitude_parameters = [
        parameter for parameter in magnitude_branch.parameters()
        if parameter.requires_grad
    ]
    if not magnitude_parameters:
        raise ValueError("magnitude branch has no trainable parameters")
    parameter_groups = [
        {"params": magnitude_parameters, "lr": cfg.magnitude_lr, "name": "magnitude"}
    ]
    phase_parameters = [
        parameter for parameter in model.phase_inr.parameters()
        if parameter.requires_grad
    ]
    if phase_parameters:
        parameter_groups.append(
            {"params": phase_parameters, "lr": cfg.phase_lr, "name": "phase"}
        )
    if is_residual and model.log_prior_scale.requires_grad:
        parameter_groups.append(
            {
                "params": [model.log_prior_scale],
                "lr": (
                    cfg.magnitude_lr
                    if cfg.prior_scale_lr is None
                    else cfg.prior_scale_lr
                ),
                "name": "prior_scale",
            }
        )

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(parameter_groups)

    def cosine_factor(step: int) -> float:
        fraction = min(float(step) / cfg.iterations, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * fraction))
        return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_factor)
    coords = make_coord_grid(*shape, device=device)
    cached_prior = None
    if is_residual:
        with torch.no_grad():
            cached_prior = model.prior_magnitude(coords)

    history: List[Dict[str, Any]] = []
    best_validation_error = float("inf")
    best_iteration = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_maps = None

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    for zero_based_iteration in range(cfg.iterations):
        iteration = zero_based_iteration + 1
        learning_rates = {
            group["name"]: float(group["lr"]) for group in optimizer.param_groups
        }
        optimizer.zero_grad(set_to_none=True)
        reconstruction, magnitude, phase, delta = _render(
            model, coords, shape, cached_prior
        )
        dc = data_consistency(
            train_operator(reconstruction),
            train_kspace,
            mask=getattr(train_operator, "mask", None),
        )
        if is_residual:
            delta_l1 = cfg.lambda_delta_l1 * delta.abs().mean()
            delta_tv = (
                cfg.lambda_delta_tv * tv_2d(delta)
                if cfg.lambda_delta_tv > 0
                else torch.zeros((), device=device)
            )
        else:
            # Keep these identically zero even if a residual configuration is
            # reused for the matched current-only baseline.
            delta_l1 = torch.zeros((), device=device)
            delta_tv = torch.zeros((), device=device)
        phase_tv = (
            cfg.lambda_phase_tv * phase_tv_2d(phase)
            if cfg.lambda_phase_tv > 0 and phase_parameters
            else torch.zeros((), device=device)
        )
        regularization = delta_l1 + delta_tv + phase_tv
        loss = dc + regularization
        loss.backward()
        gradient_norm = None
        if cfg.grad_clip_norm is not None:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                ],
                cfg.grad_clip_norm,
            )
        optimizer.step()
        scheduler.step()

        should_evaluate = (
            iteration == 1
            or iteration % cfg.eval_every == 0
            or iteration == cfg.iterations
        )
        if should_evaluate:
            # Rendering here, after optimizer.step(), is intentional: checkpoint
            # iteration numbers correspond to weights that have received that
            # many updates.
            with torch.no_grad():
                post_update_maps = _render(model, coords, shape, cached_prior)
                train_error = relative_kspace_error(
                    post_update_maps[0], train_operator, train_kspace
                )
                validation_error = (
                    relative_kspace_error(
                        post_update_maps[0],
                        validation_operator,
                        validation_kspace,
                    )
                    if has_validation
                    else None
                )
            row = {
                "iteration": iteration,
                "loss_before_update": float(loss.detach()),
                "dc_before_update": float(dc.detach()),
                "regularization_before_update": float(regularization.detach()),
                "delta_l1_before_update": float(delta_l1.detach()),
                "delta_tv_before_update": float(delta_tv.detach()),
                "phase_tv_before_update": float(phase_tv.detach()),
                "train_error": train_error,
                "validation_error": validation_error,
                "gradient_norm_before_clip": (
                    None if gradient_norm is None else float(gradient_norm)
                ),
                **{f"{name}_lr": value for name, value in learning_rates.items()},
            }
            history.append(row)
            if has_validation and validation_error < best_validation_error:
                best_validation_error = validation_error
                best_iteration = iteration
                best_state = _cpu_state_dict(model)
                best_maps = _snapshot_maps(post_update_maps)
            if verbose:
                validation_text = (
                    ""
                    if validation_error is None
                    else f"  val-rel={validation_error:.6f}"
                )
                print(
                    f"[magnitude-phase/{mode}] iter {iteration:5d}  "
                    f"loss={float(loss.detach()):.6f}  "
                    f"train-rel={train_error:.6f}{validation_text}"
                )

    with torch.no_grad():
        final_maps_device = _render(model, coords, shape, cached_prior)
    final_maps = _snapshot_maps(final_maps_device)
    final_state = _cpu_state_dict(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime_seconds = time.perf_counter() - started

    if not has_validation:
        best_iteration = cfg.iterations
        best_validation_value = None
        best_state = final_state
        best_maps = final_maps
    else:
        best_validation_value = float(best_validation_error)
        if best_state is None or best_maps is None:
            raise RuntimeError("validation was requested but no checkpoint was evaluated")
        model.load_state_dict(best_state)

    recon, magnitude, phase, delta = best_maps
    final_recon, final_magnitude, final_phase, final_delta = final_maps
    return MagnitudePhaseTrainResult(
        mode=mode,
        recon=recon,
        magnitude=magnitude,
        phase=phase,
        delta=delta,
        final_recon=final_recon,
        final_magnitude=final_magnitude,
        final_phase=final_phase,
        final_delta=final_delta,
        state_dict=best_state,
        history=history,
        best_iteration=best_iteration,
        best_validation_error=best_validation_value,
        iterations_completed=cfg.iterations,
        runtime_seconds=runtime_seconds,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
    )


__all__ = [
    "MagnitudePhaseTrainConfig",
    "MagnitudePhaseTrainResult",
    "build_scalar_inr",
    "relative_kspace_error",
    "train_magnitude_phase",
    "zero_last_linear_",
]
