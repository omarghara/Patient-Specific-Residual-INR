"""Prior + residual image composition.

Reconstructs the current (complex) slice as a frozen magnitude *prior* plus a
learned complex *residual*::

    x_hat = (m_prior + r_real) + i * r_imag

matching the proposal's ``x_current(c) = f_prior(c) + r_current(c)``. The prior
INR carries stable patient anatomy (magnitude); the residual INR carries the
interval change and the phase needed for k-space data consistency.

An optional spatial ``gate`` g(c) in [0, 1] multiplies the residual
(``x_hat = m_prior + g * r``). It is disabled by default -- the first target is
the plain prior+residual model; the gate is a later upgrade.
"""

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
        r = self.residual(coords)
        g = self.gate(coords)
        if g is not None:
            r = g[..., None] * r
        real = m + r[..., 0]
        imag = r[..., 1]
        x = torch.complex(real, imag)
        return x.reshape(*shape)
