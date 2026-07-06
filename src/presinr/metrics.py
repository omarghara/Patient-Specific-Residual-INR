"""Evaluation metrics.

Standard reconstruction fidelity (PSNR / SSIM / NMSE) on magnitude images, plus
the two change-aware metrics from the proposal:

* Change-Preservation Error (CPE): L1 between the recon-vs-prior difference and
  the reference-vs-prior difference. Low = interval change is preserved.
* Prior-Bias Score (PBS): ratio of recon-vs-prior change to reference-vs-prior
  change. ~1 is good; <<1 flags over-regression toward the prior.

Magnitudes are normalized by the reference's robust max so ``data_range=1``,
consistent with the LAPS metric convention.
"""

from typing import Dict

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _mag(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return np.abs(x).astype(np.float32)


def _norm(x: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(x / (scale + 1e-12), 0.0, None)


def robust_scale(x_ref) -> float:
    """Robust intensity scale (99.9th percentile of the reference magnitude)."""
    m = _mag(x_ref)
    return float(np.quantile(m, 0.999))


def psnr(recon, ref, scale: float = None) -> float:
    r, t = _mag(recon), _mag(ref)
    if scale is None:
        scale = robust_scale(t)
    r, t = _norm(r, scale), _norm(t, scale)
    return float(peak_signal_noise_ratio(t, r, data_range=1.0))


def ssim(recon, ref, scale: float = None) -> float:
    r, t = _mag(recon), _mag(ref)
    if scale is None:
        scale = robust_scale(t)
    r, t = _norm(r, scale), _norm(t, scale)
    return float(structural_similarity(t, r, data_range=1.0))


def nmse(recon, ref) -> float:
    r, t = _mag(recon), _mag(ref)
    return float(np.sum((r - t) ** 2) / (np.sum(t ** 2) + 1e-12))


def cpe(recon, ref, prior) -> float:
    """Change-Preservation Error: || (|recon|-prior) - (|ref|-prior) ||_1."""
    r, t, p = _mag(recon), _mag(ref), _mag(prior)
    return float(np.mean(np.abs((r - p) - (t - p))))


def pbs(recon, ref, prior, eps: float = 1e-6) -> float:
    """Prior-Bias Score: ||(|recon|-prior)||_1 / ||(|ref|-prior)||_1."""
    r, t, p = _mag(recon), _mag(ref), _mag(prior)
    num = np.sum(np.abs(r - p))
    den = np.sum(np.abs(t - p))
    return float(num / (den + eps))


def all_metrics(recon, ref, prior=None) -> Dict[str, float]:
    scale = robust_scale(ref)
    out = {
        "psnr": psnr(recon, ref, scale),
        "ssim": ssim(recon, ref, scale),
        "nmse": nmse(recon, ref),
    }
    if prior is not None:
        out["cpe"] = cpe(recon, ref, prior)
        out["pbs"] = pbs(recon, ref, prior)
    return out
