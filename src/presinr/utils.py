"""Small shared helpers: seeding, device, numpy conversion, image dumps."""

import os
import random
from typing import Optional

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


def save_magnitude_panel(images, titles, path, cmap="gray"):
    """Save a row of magnitude images to ``path`` for quick qualitative checks."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        arr = np.abs(to_numpy(img))
        vmax = np.quantile(arr, 0.999) + 1e-8
        ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path
