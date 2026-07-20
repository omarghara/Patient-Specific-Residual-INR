"""Experiment-level utilities shared by notebooks and command-line studies."""

from .magnitude_phase import (
    MagnitudePhaseTrainConfig,
    MagnitudePhaseTrainResult,
    build_scalar_inr,
    relative_kspace_error,
    train_magnitude_phase,
    zero_last_linear_,
)

__all__ = [
    "MagnitudePhaseTrainConfig",
    "MagnitudePhaseTrainResult",
    "build_scalar_inr",
    "relative_kspace_error",
    "train_magnitude_phase",
    "zero_last_linear_",
]
