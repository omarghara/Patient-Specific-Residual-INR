"""Training objectives for residual-INR reconstruction.

Total stage-2 loss (proposal Eq. for the plain model)::

    L = ||M F S x_hat - y||_2^2 + lambda_eff ||g r||_1
        + lambda_raw ||r||_1 + lambda_gate ||g||_1 + lambda_gate_tv TV(g)

The plain residual model is the special case ``g = 1``.
"""

import torch


def data_consistency(
    y_pred: torch.Tensor,
    y_meas: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Mean squared complex k-space error over measured samples only.

    Dividing by the full zero-filled grid makes regularization strength depend on
    acceleration. Supplying the acquisition mask keeps the data-term scale tied
    to the number of actual measurements instead.
    """
    error = (y_pred - y_meas).abs().pow(2)
    if mask is None:
        return error.mean()
    weights = mask.to(device=error.device, dtype=error.dtype)
    while weights.ndim < error.ndim:
        weights = weights.unsqueeze(0)
    weights = weights.expand_as(error)
    # The forward operator validates nonempty acquisition masks once at
    # construction time. Avoid a GPU-to-CPU ``item()`` synchronization here,
    # because this function runs on every optimization iteration.
    return (error * weights).sum() / weights.sum()


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


def phase_tv_2d(phase: torch.Tensor) -> torch.Tensor:
    """Wrap-invariant smoothness penalty for a phase map in radians.

    ``1 - cos(delta)`` treats phase values separated by a multiple of ``2*pi``
    as identical, unlike ordinary TV applied directly to wrapped angles.
    """
    dx = (1.0 - torch.cos(phase[1:, :] - phase[:-1, :])).mean()
    dy = (1.0 - torch.cos(phase[:, 1:] - phase[:, :-1])).mean()
    return dx + dy


def gate_l1(g: torch.Tensor) -> torch.Tensor:
    """Sparsity penalty encouraging the gate to stay closed (trust the prior)."""
    return g.abs().mean()
