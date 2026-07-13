"""Multi-coil Cartesian SENSE forward model: y = M F S x.

``S`` applies coil sensitivities, ``F`` is a centered 2-D FFT, ``M`` is the
undersampling mask. Adjoint is the standard SENSE combine. Mirrors
``laps.recon.linops.CartesianSenseLinop`` but trimmed to what the residual-INR
reconstruction needs (single 2-D slice, autograd through the image).
"""

from typing import Optional

import torch
import torch.nn as nn

from .fft import fftc, ifftc


class CartesianSense(nn.Module):
    """Forward operator for a single 2-D slice.

    Args:
        mps: complex coil sensitivity maps, shape ``(Nc, Nx, Ny)``.
        mask: sampling mask, shape ``(Nx, Ny)`` (0/1). ``None`` -> fully sampled.
    """

    def __init__(self, mps: torch.Tensor, mask: Optional[torch.Tensor] = None):
        super().__init__()
        assert mps.ndim == 3, "mps must be (Nc, Nx, Ny)"
        self.n_coils, self.nx, self.ny = mps.shape
        if mask is None:
            mask = torch.ones(self.nx, self.ny, device=mps.device)
        mask = mask.to(torch.float32)
        if tuple(mask.shape) != (self.nx, self.ny):
            raise ValueError(
                f"mask must have shape {(self.nx, self.ny)}, got {tuple(mask.shape)}"
            )
        if not torch.any(mask != 0):
            raise ValueError("sampling mask contains no measured samples")
        self.register_buffer("mps", mps.to(torch.complex64))
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Image ``(Nx, Ny)`` complex -> coil k-space ``(Nc, Nx, Ny)`` complex."""
        xc = x[None] * self.mps               # (Nc, Nx, Ny)
        k = fftc(xc, dim=(-2, -1))
        return k * self.mask                  # broadcast over coils

    def adjoint(self, y: torch.Tensor) -> torch.Tensor:
        """Coil k-space ``(Nc, Nx, Ny)`` -> combined image ``(Nx, Ny)`` complex."""
        img = ifftc(y * self.mask, dim=(-2, -1))
        return torch.sum(img * torch.conj(self.mps), dim=0)

    def zero_filled(self, y: torch.Tensor) -> torch.Tensor:
        """SENSE-combined zero-filled reconstruction (adjoint of measurements)."""
        return self.adjoint(y)
