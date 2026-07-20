"""Prior + residual image composition.

Reconstructs the current (complex) slice as a frozen magnitude *prior* plus a
learned complex *residual*::

    x_hat = (m_prior + r_real) + i * r_imag

matching the proposal's ``x_current(c) = f_prior(c) + r_current(c)``. The prior
INR carries stable patient anatomy (magnitude); the residual INR carries the
interval change and the phase needed for k-space data consistency.

An optional spatial support ``gate`` g(c) in [0, 1] multiplies the residual
(``x_hat = m_prior + g * r``). For an interpretable, numerically identifiable
gated objective, use a finite ``residual_bound`` and regularize the pre-gate
residual as well as the effective product.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PriorResidualINR(nn.Module):
    def __init__(
        self,
        prior_inr: nn.Module,
        residual_inr: nn.Module,
        gate_inr: Optional[nn.Module] = None,
        residual_bound: Optional[float] = None,
    ):
        super().__init__()
        self.prior_inr = prior_inr
        self.residual_inr = residual_inr
        self.gate_inr = gate_inr
        if residual_bound is not None and (
            not math.isfinite(residual_bound) or residual_bound <= 0
        ):
            raise ValueError(f"residual_bound must be finite and positive, got {residual_bound}")
        self.residual_bound = residual_bound

    def freeze_prior(self):
        for p in self.prior_inr.parameters():
            p.requires_grad_(False)
        self.prior_inr.eval()

    def prior_magnitude(self, coords: torch.Tensor) -> torch.Tensor:
        """Prior magnitude at ``coords`` -> ``(N,)``."""
        return self.prior_inr(coords)[..., 0]

    def residual(self, coords: torch.Tensor) -> torch.Tensor:
        """Complex residual at ``coords`` as real/imag channels -> ``(N, 2)``."""
        r = self.residual_inr(coords)
        if self.residual_bound is not None:
            r = self.residual_bound * torch.tanh(r)
        return r

    def gate(self, coords: torch.Tensor) -> Optional[torch.Tensor]:
        if self.gate_inr is None:
            return None
        return torch.sigmoid(self.gate_inr(coords)[..., 0])

    def residual_components(
        self, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Return ``(residual, effective_residual, gate)`` at ``coords``.

        ``residual`` is post-bound but pre-gate. Keeping it explicit is important:
        gated objectives must regularize this component, rather than only the
        product ``gate * residual``, to avoid a scale degeneracy.
        """
        residual = self.residual(coords)
        gate = self.gate(coords)
        effective = residual if gate is None else gate[..., None] * residual
        return residual, effective, gate

    def forward(
        self,
        coords: torch.Tensor,
        shape: Tuple[int, int],
        prior_mag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the complex reconstruction reshaped to ``shape``.

        ``prior_mag`` may be precomputed (prior is frozen in stage 2) to avoid
        re-evaluating the prior INR each iteration.
        """
        m = prior_mag if prior_mag is not None else self.prior_magnitude(coords)
        _, effective, _ = self.residual_components(coords)
        real = m + effective[..., 0]
        imag = effective[..., 1]
        x = torch.complex(real, imag)
        return x.reshape(*shape)


class PriorMagnitudePhaseINR(nn.Module):
    """Frozen prior magnitude + scalar magnitude change + independent phase.

    The released longitudinal prior is a magnitude image, whereas measured MRI
    k-space constrains a complex follow-up image.  Keeping phase outside the
    residual makes it meaningful to impose sparsity on longitudinal magnitude
    change::

        magnitude(c) = clamp_min(prior(c) + delta_m(c), 0)
        x(c) = magnitude(c) * exp(1j * phase(c))

    Without this separation, a complex residual must encode the entire object
    phase and therefore cannot generally be sparse even when anatomy is stable.
    """

    def __init__(
        self,
        prior_inr: nn.Module,
        magnitude_residual_inr: nn.Module,
        phase_inr: nn.Module,
        magnitude_residual_bound: Optional[float] = None,
        prior_scale: float = 1.0,
        learn_prior_scale: bool = False,
    ):
        super().__init__()
        self.prior_inr = prior_inr
        self.magnitude_residual_inr = magnitude_residual_inr
        self.phase_inr = phase_inr
        if magnitude_residual_bound is not None and (
            not math.isfinite(magnitude_residual_bound)
            or magnitude_residual_bound <= 0
        ):
            raise ValueError(
                "magnitude_residual_bound must be finite and positive, "
                f"got {magnitude_residual_bound}"
            )
        self.magnitude_residual_bound = magnitude_residual_bound
        if not math.isfinite(float(prior_scale)) or prior_scale <= 0:
            raise ValueError(
                f"prior_scale must be finite and positive, got {prior_scale}"
            )
        self.log_prior_scale = nn.Parameter(
            torch.tensor(math.log(float(prior_scale)), dtype=torch.float32),
            requires_grad=bool(learn_prior_scale),
        )

    def freeze_prior(self):
        for parameter in self.prior_inr.parameters():
            parameter.requires_grad_(False)
        self.prior_inr.eval()

    def prior_magnitude(self, coords: torch.Tensor) -> torch.Tensor:
        return self.prior_inr(coords)[..., 0]

    def scaled_prior_magnitude(self, coords: torch.Tensor) -> torch.Tensor:
        """Prior contribution after acquisition-derived intensity scaling."""
        return torch.exp(self.log_prior_scale) * self.prior_magnitude(coords)

    @property
    def prior_scale(self) -> torch.Tensor:
        return torch.exp(self.log_prior_scale)

    def set_prior_scale(self, value: float) -> None:
        """Set a positive acquisition-unit prior scale in place."""
        if not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"prior scale must be finite and positive, got {value}")
        with torch.no_grad():
            self.log_prior_scale.copy_(
                torch.as_tensor(
                    math.log(float(value)),
                    device=self.log_prior_scale.device,
                    dtype=self.log_prior_scale.dtype,
                )
            )

    def magnitude_residual(self, coords: torch.Tensor) -> torch.Tensor:
        delta = self.magnitude_residual_inr(coords)[..., 0]
        if self.magnitude_residual_bound is not None:
            delta = self.magnitude_residual_bound * torch.tanh(delta)
        return delta

    def phase(self, coords: torch.Tensor) -> torch.Tensor:
        return self.phase_inr(coords)[..., 0]

    def components(
        self,
        coords: torch.Tensor,
        prior_mag: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(scaled_prior, delta_m, nonnegative_magnitude, phase)``.

        ``prior_mag`` is the cached *unscaled* prior-INR output.  Scaling it
        inside the composition keeps the residual in current-acquisition units.
        """
        raw_prior = prior_mag if prior_mag is not None else self.prior_magnitude(coords)
        raw_prior = torch.clamp_min(raw_prior, 0.0)
        prior = self.prior_scale * raw_prior
        delta = self.magnitude_residual(coords)
        magnitude = torch.clamp_min(prior + delta, 0.0)
        phase = self.phase(coords)
        return prior, delta, magnitude, phase

    def forward(
        self,
        coords: torch.Tensor,
        shape: Tuple[int, int],
        prior_mag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        _, _, magnitude, phase = self.components(coords, prior_mag=prior_mag)
        image = torch.polar(magnitude, phase)
        return image.reshape(*shape)


class CurrentMagnitudePhaseINR(nn.Module):
    """Prior-free current magnitude and phase with the same branch structure.

    Matching the magnitude and phase branch sizes to
    :class:`PriorMagnitudePhaseINR` gives a controlled baseline for measuring
    the value of the longitudinal prior rather than INR capacity alone.
    """

    def __init__(self, magnitude_inr: nn.Module, phase_inr: nn.Module):
        super().__init__()
        self.magnitude_inr = magnitude_inr
        self.phase_inr = phase_inr

    def components(
        self, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_magnitude = self.magnitude_inr(coords)[..., 0]
        magnitude = torch.clamp_min(raw_magnitude, 0.0)
        phase = self.phase_inr(coords)[..., 0]
        return raw_magnitude, magnitude, phase

    def forward(self, coords: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
        _, magnitude, phase = self.components(coords)
        return torch.polar(magnitude, phase).reshape(*shape)
