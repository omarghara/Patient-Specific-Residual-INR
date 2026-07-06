"""Training objectives for residual-INR reconstruction.

Total stage-2 loss (proposal Eq. for the plain model)::

    L = ||M F S x_hat - y||_2^2  +  lambda_res * ||r||_1  [ + lambda_tv * TV(r) ]

with optional gate terms reserved for the later gated variant.
"""

import torch


def data_consistency(y_pred: torch.Tensor, y_meas: torch.Tensor) -> torch.Tensor:
    """Mean squared k-space error over measured points (complex)."""
    return (y_pred - y_meas).abs().pow(2).mean()


def residual_l1(r: torch.Tensor) -> torch.Tensor:
    """L1 penalty on the complex residual magnitude.

    ``r`` has real/imag in the last dimension, shape ``(..., 2)``.
    """
    mag = torch.sqrt(r[..., 0] ** 2 + r[..., 1] ** 2 + 1e-12)
    return mag.mean()


def tv_2d(img: torch.Tensor) -> torch.Tensor:
    """Anisotropic total variation of a real 2-D map ``(H, W)``."""
    dx = (img[1:, :] - img[:-1, :]).abs().mean()
    dy = (img[:, 1:] - img[:, :-1]).abs().mean()
    return dx + dy


def gate_l1(g: torch.Tensor) -> torch.Tensor:
    """Sparsity penalty encouraging the gate to stay closed (trust the prior)."""
    return g.abs().mean()
