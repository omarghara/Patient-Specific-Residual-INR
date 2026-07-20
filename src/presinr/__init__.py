"""Patient-Specific Residual INR for accelerated follow-up MRI reconstruction."""

from .forward import CartesianSense
from .calibration import prior_scale_from_kspace, real_least_squares_scale
from .metrics import (
    acquisition_calibrated_longitudinal_metrics,
    change_cosine,
    change_gain,
    mutual_information,
    prior_followup_mutual_information,
)
from .models.composition import (
    CurrentMagnitudePhaseINR,
    PriorMagnitudePhaseINR,
    PriorResidualINR,
)
from .models.inr import build_inr, make_coord_grid
from .recon import (
    ImageFitConfig,
    KspaceFitConfig,
    MagnitudePhaseFitConfig,
    PhaseFitConfig,
    PriorFitConfig,
    ReconResult,
    ResidualFitConfig,
    fit_image_inr,
    fit_current_only,
    fit_magnitude_phase_residual,
    fit_phase_inr,
    fit_prior,
    fit_residual,
    fit_residual_image,
)

__all__ = [
    "CartesianSense",
    "real_least_squares_scale",
    "prior_scale_from_kspace",
    "change_cosine",
    "change_gain",
    "acquisition_calibrated_longitudinal_metrics",
    "mutual_information",
    "prior_followup_mutual_information",
    "PriorResidualINR",
    "PriorMagnitudePhaseINR",
    "CurrentMagnitudePhaseINR",
    "build_inr",
    "make_coord_grid",
    "fit_prior",
    "fit_phase_inr",
    "fit_current_only",
    "fit_magnitude_phase_residual",
    "fit_residual",
    "fit_residual_image",
    "fit_image_inr",
    "PriorFitConfig",
    "ResidualFitConfig",
    "KspaceFitConfig",
    "PhaseFitConfig",
    "MagnitudePhaseFitConfig",
    "ImageFitConfig",
    "ReconResult",
]
