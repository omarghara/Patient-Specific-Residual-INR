"""Reference-free intensity calibration from acquired complex k-space.

Longitudinal DICOM magnitudes and follow-up k-space generally arrive in
unrelated intensity units.  A residual model should not have to encode this
global nuisance scale as anatomical change.  The helpers here estimate the
real scalar that best maps a candidate complex image into the measured
k-space, using acquired samples only.
"""

from typing import Optional

import torch


def _broadcast_weights(weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weights = torch.as_tensor(weights, device=target.device, dtype=target.real.dtype)
    while weights.ndim < target.ndim:
        weights = weights.unsqueeze(0)
    try:
        return weights.expand_as(target.real)
    except RuntimeError as error:
        raise ValueError(
            f"weights with shape {tuple(weights.shape)} cannot broadcast to "
            f"data shape {tuple(target.shape)}"
        ) from error


def real_least_squares_scale(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: Optional[torch.Tensor] = None,
    nonnegative: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return the real scalar minimizing ``||a * prediction - target||_2``.

    Complex inputs use the real part of their Hermitian inner product.  An
    optional mask or set of nonnegative weights may omit unmeasured samples.
    The returned scalar stays on the input device and remains differentiable
    with respect to both tensors, although reconstruction code normally calls
    this function under ``torch.no_grad()``.
    """
    prediction = torch.as_tensor(prediction)
    target = torch.as_tensor(target, device=prediction.device)
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if prediction.numel() == 0:
        raise ValueError("prediction and target must be nonempty")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must contain only finite values")
    if not torch.isfinite(torch.as_tensor(eps)) or eps <= 0:
        raise ValueError(f"eps must be finite and positive, got {eps}")

    product = prediction.conj() * target
    energy = prediction.abs().square()
    if weights is not None:
        weight = _broadcast_weights(weights, target)
        if not torch.isfinite(weight).all() or torch.any(weight < 0):
            raise ValueError("weights must be finite and nonnegative")
        product = product * weight
        energy = energy * weight

    denominator = energy.sum().real
    if float(denominator.detach()) <= eps:
        raise ValueError("cannot calibrate scale from a zero-energy prediction")
    scale = product.sum().real / denominator
    return scale.clamp_min(0.0) if nonnegative else scale


def prior_scale_from_kspace(
    prior_magnitude: torch.Tensor,
    phase: torch.Tensor,
    operator,
    measured_kspace: torch.Tensor,
    *,
    nonnegative: bool = True,
) -> torch.Tensor:
    """Calibrate a prior magnitude to measured k-space without a reference.

    ``phase`` is typically initialized from a zero-filled or CG-SENSE image.
    Only locations selected by ``operator.mask`` contribute to the fit.
    """
    prior_magnitude = torch.as_tensor(prior_magnitude)
    phase = torch.as_tensor(
        phase,
        device=prior_magnitude.device,
        dtype=prior_magnitude.real.dtype,
    )
    if prior_magnitude.shape != phase.shape:
        raise ValueError(
            f"prior magnitude and phase must have the same shape, got "
            f"{tuple(prior_magnitude.shape)} and {tuple(phase.shape)}"
        )
    if torch.is_complex(prior_magnitude):
        prior_magnitude = prior_magnitude.abs()
    if torch.any(prior_magnitude < 0):
        raise ValueError("prior magnitude must be nonnegative")
    candidate = torch.polar(prior_magnitude.float(), phase.float())
    prediction = operator(candidate)
    measured_kspace = measured_kspace.to(
        device=prediction.device,
        dtype=prediction.dtype,
    )
    mask = getattr(operator, "mask", None)
    return real_least_squares_scale(
        prediction,
        measured_kspace,
        weights=mask,
        nonnegative=nonnegative,
    )

