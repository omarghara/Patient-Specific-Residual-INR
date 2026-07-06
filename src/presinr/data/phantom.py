"""Synthetic longitudinal multi-coil phantom for pipeline validation.

Builds a *prior* magnitude image and a *current* complex image that differs from
the prior by a localized interval change (a simulated lesion) plus a smooth
phase, then simulates multi-coil undersampled k-space with the same forward
model used for real SLAM data. Ground truth is known, so we can check both
fidelity and change preservation.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import sigpy.mri as mr
import torch

from ..forward import CartesianSense


@dataclass
class PhantomSample:
    prior: torch.Tensor      # (H, W) real magnitude
    current: torch.Tensor    # (H, W) complex ground-truth follow-up
    mps: torch.Tensor        # (Nc, H, W) complex coil maps
    mask: torch.Tensor       # (H, W) sampling mask
    ksp: torch.Tensor        # (Nc, H, W) undersampled measurements


def _base_anatomy(n: int) -> np.ndarray:
    from skimage.data import shepp_logan_phantom
    from skimage.transform import resize

    ph = shepp_logan_phantom().astype(np.float32)
    ph = resize(ph, (n, n), order=1, mode="reflect", anti_aliasing=True)
    return ph / (ph.max() + 1e-8)


def _disk(n: int, cx: float, cy: float, r: float) -> np.ndarray:
    ys, xs = np.mgrid[0:n, 0:n]
    return (((xs - cx) ** 2 + (ys - cy) ** 2) <= r ** 2).astype(np.float32)


def make_phantom(
    n: int = 192,
    n_coils: int = 8,
    accel: float = 4.0,
    lesion_intensity: float = 0.6,
    lesion_radius: float = 9.0,
    noise_std: float = 1e-3,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> PhantomSample:
    rng = np.random.default_rng(seed)
    device = device or torch.device("cpu")

    base = _base_anatomy(n)
    prior = base.copy()

    # Interval change: a focal lesion appears in the follow-up scan.
    lesion = _disk(n, cx=0.60 * n, cy=0.42 * n, r=lesion_radius)
    current_mag = np.clip(base + lesion_intensity * lesion, 0.0, None)

    # Smooth phase so the follow-up is genuinely complex.
    ys, xs = np.mgrid[0:n, 0:n] / n - 0.5
    phase = 0.8 * np.pi * (xs + 0.5 * ys) + 0.3 * np.pi * (xs ** 2 - ys ** 2)
    current = (current_mag * np.exp(1j * phase)).astype(np.complex64)

    mps = mr.birdcage_maps((n_coils, n, n), r=1.25).astype(np.complex64)
    if accel and accel > 1.0:
        mask = mr.poisson((n, n), accel, calib=(24, 24), dtype=float, seed=seed)
    else:
        mask = np.ones((n, n), dtype=float)

    mps_t = torch.from_numpy(mps).to(device)
    mask_t = torch.from_numpy(np.asarray(mask)).to(torch.float32).to(device)
    current_t = torch.from_numpy(current).to(device)
    prior_t = torch.from_numpy(prior.astype(np.float32)).to(device)

    op = CartesianSense(mps_t, mask_t).to(device)
    with torch.no_grad():
        ksp = op(current_t)
        noise = noise_std * (
            torch.randn_like(ksp.real) + 1j * torch.randn_like(ksp.real)
        )
        ksp = (ksp + noise) * mask_t

    return PhantomSample(
        prior=prior_t, current=current_t, mps=mps_t, mask=mask_t, ksp=ksp
    )
