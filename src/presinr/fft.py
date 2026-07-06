"""Centered FFT utilities.

Convention matches the LAPS reference implementation
(``ifftshift -> fftn(norm="ortho") -> fftshift``) so that our data-consistency
term and reconstructions are directly comparable to their baselines.
"""

from typing import Sequence

import torch


def fftc(x: torch.Tensor, dim: Sequence[int] = (-2, -1)) -> torch.Tensor:
    """Centered orthonormal n-D FFT along ``dim``."""
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.fftn(x, dim=dim, norm="ortho")
    return torch.fft.fftshift(x, dim=dim)


def ifftc(x: torch.Tensor, dim: Sequence[int] = (-2, -1)) -> torch.Tensor:
    """Centered orthonormal n-D inverse FFT along ``dim``."""
    x = torch.fft.ifftshift(x, dim=dim)
    x = torch.fft.ifftn(x, dim=dim, norm="ortho")
    return torch.fft.fftshift(x, dim=dim)
