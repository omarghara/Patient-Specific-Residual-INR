"""Small shared helpers: seeding, device, numpy conversion, image dumps."""

import os
import random
from typing import Optional, Sequence

import numpy as np
import torch


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: Optional[str] = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _spatial_shape(shape: Sequence[int]) -> tuple[int, int]:
    """Validate and return a positive 2-D spatial shape."""
    if len(shape) != 2:
        raise ValueError(f"shape must contain exactly two dimensions, got {tuple(shape)}")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"shape dimensions must be positive, got {(height, width)}")
    return height, width


def center_crop_to(x: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
    """Exactly center-crop the last two dimensions of ``x`` to ``shape``.

    Unlike :func:`resize_to`, this operation never interpolates values.  It is
    the inverse spatial operation for SLAM references, which are reconstructed
    on the native k-space matrix and then center-zero-padded to ``256 x 256``.
    Leading dimensions and real/complex dtypes are preserved.
    """
    out_h, out_w = _spatial_shape(shape)
    in_h, in_w = x.shape[-2:]
    if out_h > in_h or out_w > in_w:
        raise ValueError(
            f"cannot crop spatial shape {(in_h, in_w)} to larger shape {(out_h, out_w)}"
        )
    top = (in_h - out_h) // 2
    left = (in_w - out_w) // 2
    return x[..., top : top + out_h, left : left + out_w]


def center_pad_to(
    x: torch.Tensor,
    shape: Sequence[int],
    value: float = 0.0,
) -> torch.Tensor:
    """Exactly center-pad the last two dimensions of ``x`` to ``shape``.

    When a size difference is odd, the extra sample is placed on the bottom or
    right, matching NumPy's conventional ``delta // 2`` center-padding rule and
    the SLAM dataset example. Leading dimensions and gradients are preserved.
    """
    import torch.nn.functional as F

    out_h, out_w = _spatial_shape(shape)
    in_h, in_w = x.shape[-2:]
    if out_h < in_h or out_w < in_w:
        raise ValueError(
            f"cannot pad spatial shape {(in_h, in_w)} to smaller shape {(out_h, out_w)}"
        )
    dh, dw = out_h - in_h, out_w - in_w
    left, right = dw // 2, dw - dw // 2
    top, bottom = dh // 2, dh - dh // 2
    return F.pad(x, (left, right, top, bottom), mode="constant", value=value)


def resize_to(x: torch.Tensor, shape) -> torch.Tensor:
    """Bilinearly resize a 2-D image ``(H, W)`` to ``shape``.

    This is a geometric interpolation and must not be used to undo known
    center-padding (use :func:`center_crop_to` for that). Complex tensors are
    resized channel-wise on real/imag.
    """
    import torch.nn.functional as F

    if tuple(x.shape[-2:]) == tuple(shape):
        return x
    if torch.is_complex(x):
        stacked = torch.stack([x.real, x.imag], dim=0)[None]  # (1,2,H,W)
        out = F.interpolate(stacked, size=tuple(shape), mode="bilinear", align_corners=False)
        return torch.complex(out[0, 0], out[0, 1])
    out = F.interpolate(x[None, None].float(), size=tuple(shape), mode="bilinear", align_corners=False)
    return out[0, 0]


def save_magnitude_panel(
    images,
    titles,
    path,
    cmap="gray",
    signed=None,
    value_ranges=None,
):
    """Save a row of images with explicit signed/fixed-range visualization.

    By default panels show magnitude with a per-image robust maximum. Set an
    entry in ``signed`` to use the real-valued map with a symmetric diverging
    scale, and/or provide ``(vmin, vmax)`` in ``value_ranges`` (for example,
    ``(0, 1)`` for a gate). This prevents sign loss and misleading gate autoscale.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(images)
    signed = [False] * n if signed is None else list(signed)
    value_ranges = [None] * n if value_ranges is None else list(value_ranges)
    if len(titles) != n or len(signed) != n or len(value_ranges) != n:
        raise ValueError("images, titles, signed, and value_ranges must have equal lengths")
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, img, title, is_signed, fixed_range in zip(
        axes, images, titles, signed, value_ranges
    ):
        raw = to_numpy(img)
        arr = np.real(raw) if is_signed else np.abs(raw)
        if fixed_range is not None:
            vmin, vmax = fixed_range
        elif is_signed:
            vmax = np.quantile(np.abs(arr), 0.999) + 1e-8
            vmin = -vmax
        else:
            vmin, vmax = 0, np.quantile(arr, 0.999) + 1e-8
        ax.imshow(arr, cmap="coolwarm" if is_signed else cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def misregister(img, shift=(0.0, 0.0), angle=0.0):
    """Apply a rigid misregistration (rotation then translation) to a 2-D image.

    Simulates imperfect registration between the prior and follow-up scans.
    Accepts/returns a torch tensor; ``shift`` is ``(dy, dx)`` in pixels.
    """
    from scipy.ndimage import rotate, shift as ndi_shift

    is_torch = isinstance(img, torch.Tensor)
    dev = img.device if is_torch else None
    a = to_numpy(img).astype(np.float32)
    if angle != 0.0:
        a = rotate(a, angle, reshape=False, order=1, mode="nearest")
    if shift != (0.0, 0.0):
        a = ndi_shift(a, list(shift), order=1, mode="nearest")
    out = torch.from_numpy(a)
    return out.to(dev) if is_torch else out


def save_loss_plot(hist, path, title="residual training loss"):
    """Plot each loss component (and the total) on a log-y axis vs iteration.

    ``hist`` maps component name -> list of per-iteration values. Components that
    are identically zero (e.g. an unused TV term) are skipped.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    label = {
        "total": "total",
        "data": "data fit  ||x̂-y||₁",
        "res_l1": "λ_res · ||g·r||₁",
        "raw_res_l1": "λ_raw · ||r||₁",
        "tv": "λ_tv · TV(g·r)",
        "gate": "λ_gate · mean(g)",
        "gate_tv": "λ_gate_tv · TV(g)",
    }
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for k, v in hist.items():
        if not isinstance(v, (list, tuple)):
            continue
        if len(v) == 0 or not any(abs(x) > 0 for x in v):
            continue
        y = [max(float(x), 1e-12) for x in v]
        ax.plot(range(len(y)), y, label=label.get(k, k),
                linewidth=2.0 if k == "total" else 1.3)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
