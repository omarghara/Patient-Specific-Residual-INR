"""Magnitude-image reconstruction and longitudinal evaluation metrics.

Standard PSNR/SSIM/NMSE are complemented by metrics on the signed magnitude
change relative to the prior:

* ``change_cosine`` measures spatial/sign agreement of reconstructed and true
  change (ideal: 1).
* ``change_gain`` measures recovered change amplitude along the true-change
  direction (ideal: 1; 0 is complete regression to the prior).
* mutual information (MI, in bits) is reported for prior/reference and
  prior/reconstruction. Their difference should approach zero; MI itself is a
  similarity diagnostic and is not monotonically a reconstruction-quality score.

The proposal's original CPE and PBS remain available as legacy functions, but
are intentionally excluded from :func:`all_metrics`: CPE algebraically reduces
to magnitude MAE, while PBS ignores the location and sign of change.
"""

from typing import Dict, Optional

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


def _unit_magnitude(x, scale: Optional[float] = None) -> np.ndarray:
    """Magnitude robustly mapped to ``[0, 1]`` for longitudinal histograms."""
    m = _mag(x)
    if scale is None:
        scale = robust_scale(m)
    return np.clip(m / (float(scale) + 1e-12), 0.0, 1.0)


def _check_same_shape(*arrays: np.ndarray):
    shapes = {tuple(a.shape) for a in arrays}
    if len(shapes) != 1:
        raise ValueError(f"metric inputs must have the same shape, got {sorted(shapes)}")


def _longitudinal_components(recon, ref, prior, foreground_threshold: float = 0.05):
    """Return normalized signed changes and a method-independent foreground.

    All three change maps use the reference scale, so a literal prior copy has
    exactly zero reconstructed change. Prior and reference must therefore be
    registered and intensity-harmonized before evaluation (the SLAM loader does
    this). The foreground mask uses independently scaled prior/reference maps and
    never depends on the reconstruction method.
    """
    ref_scale = robust_scale(ref)
    prior_scale = robust_scale(prior)
    r = _unit_magnitude(recon, ref_scale)
    t = _unit_magnitude(ref, ref_scale)
    p = _unit_magnitude(prior, ref_scale)
    p_for_mask = _unit_magnitude(prior, prior_scale)
    _check_same_shape(r, t, p)
    mask = (p_for_mask > foreground_threshold) | (t > foreground_threshold)
    if not np.any(mask):
        mask = np.ones_like(t, dtype=bool)
    return (
        (r - p)[mask].astype(np.float64),
        (t - p)[mask].astype(np.float64),
        mask,
        ref_scale,
        prior_scale,
    )


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


def nmse(recon, ref, scale: float = None) -> float:
    r, t = _mag(recon), _mag(ref)
    if scale is None:
        scale = robust_scale(t)
    r, t = _norm(r, scale), _norm(t, scale)
    return float(np.sum((r - t) ** 2) / (np.sum(t ** 2) + 1e-12))


def cpe(recon, ref, prior) -> float:
    """Legacy proposal CPE; algebraically identical to magnitude MAE."""
    r, t, p = _mag(recon), _mag(ref), _mag(prior)
    return float(np.mean(np.abs((r - p) - (t - p))))


def pbs(recon, ref, prior, eps: float = 1e-6) -> float:
    """Legacy change-magnitude ratio; does not measure spatial agreement."""
    r, t, p = _mag(recon), _mag(ref), _mag(prior)
    num = np.sum(np.abs(r - p))
    den = np.sum(np.abs(t - p))
    return float(num / (den + eps))


def change_cosine(
    recon,
    ref,
    prior,
    foreground_threshold: float = 0.05,
    eps: float = 1e-12,
) -> float:
    """Cosine agreement between reconstructed and reference signed change.

    Returns ``NaN`` when the reference contains no measurable change and ``0``
    when true change exists but the reconstruction exactly copies the prior.
    """
    d_recon, d_ref, _, _, _ = _longitudinal_components(
        recon, ref, prior, foreground_threshold
    )
    ref_norm = float(np.linalg.norm(d_ref))
    if ref_norm <= eps:
        return float("nan")
    recon_norm = float(np.linalg.norm(d_recon))
    if recon_norm <= eps:
        return 0.0
    return float(np.dot(d_recon, d_ref) / (recon_norm * ref_norm))


def change_gain(
    recon,
    ref,
    prior,
    foreground_threshold: float = 0.05,
    eps: float = 1e-12,
) -> float:
    """Recovered signed-change amplitude along the true-change direction."""
    d_recon, d_ref, _, _, _ = _longitudinal_components(
        recon, ref, prior, foreground_threshold
    )
    ref_energy = float(np.dot(d_ref, d_ref))
    if ref_energy <= eps:
        return float("nan")
    return float(np.dot(d_recon, d_ref) / ref_energy)


def mutual_information(
    x,
    y,
    bins: int = 32,
    mask=None,
    x_scale: Optional[float] = None,
    y_scale: Optional[float] = None,
) -> float:
    """Histogram plug-in mutual information in bits.

    Values use fixed bins on ``[0,1]`` after robust magnitude scaling. Optional
    scales and a fixed mask let callers compare several reconstructions with the
    exact same histogram definition.
    """
    if bins < 2:
        raise ValueError(f"bins must be at least 2, got {bins}")
    a = _unit_magnitude(x, x_scale)
    b = _unit_magnitude(y, y_scale)
    _check_same_shape(a, b)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        if m.shape != a.shape:
            raise ValueError(f"MI mask shape {m.shape} does not match image shape {a.shape}")
        a, b = a[m], b[m]
    else:
        a, b = a.reshape(-1), b.reshape(-1)
    if a.size == 0:
        return float("nan")

    counts, _, _ = np.histogram2d(a, b, bins=bins, range=((0.0, 1.0), (0.0, 1.0)))
    total = float(counts.sum())
    if total <= 0:
        return float("nan")
    pxy = counts / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    independent = px @ py
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log2(pxy[nz] / independent[nz])))


def prior_followup_mutual_information(
    prior,
    followup,
    bins: int = 32,
    foreground_threshold: float = 0.05,
) -> float:
    """MI in bits between prior and follow-up on their fixed joint foreground."""
    prior_scale = robust_scale(prior)
    followup_scale = robust_scale(followup)
    p = _unit_magnitude(prior, prior_scale)
    f = _unit_magnitude(followup, followup_scale)
    _check_same_shape(p, f)
    mask = (p > foreground_threshold) | (f > foreground_threshold)
    if not np.any(mask):
        mask = np.ones_like(f, dtype=bool)
    return mutual_information(
        prior,
        followup,
        bins=bins,
        mask=mask,
        x_scale=prior_scale,
        y_scale=followup_scale,
    )


def all_metrics(
    recon,
    ref,
    prior=None,
    mi_bins: int = 32,
    foreground_threshold: float = 0.05,
) -> Dict[str, float]:
    scale = robust_scale(ref)
    out = {
        "psnr": psnr(recon, ref, scale),
        "ssim": ssim(recon, ref, scale),
        "nmse": nmse(recon, ref, scale),
    }
    if prior is not None:
        d_recon, d_ref, mask, ref_scale, prior_scale = _longitudinal_components(
            recon, ref, prior, foreground_threshold
        )
        ref_energy = float(np.dot(d_ref, d_ref))
        recon_norm = float(np.linalg.norm(d_recon))
        ref_norm = float(np.sqrt(ref_energy))
        out["change_cosine"] = (
            float("nan")
            if ref_norm <= 1e-12
            else 0.0
            if recon_norm <= 1e-12
            else float(np.dot(d_recon, d_ref) / (recon_norm * ref_norm))
        )
        out["change_gain"] = (
            float("nan") if ref_energy <= 1e-12 else float(np.dot(d_recon, d_ref) / ref_energy)
        )
        mi_prior_ref = mutual_information(
            prior,
            ref,
            bins=mi_bins,
            mask=mask,
            x_scale=prior_scale,
            y_scale=ref_scale,
        )
        mi_prior_recon = mutual_information(
            prior,
            recon,
            bins=mi_bins,
            mask=mask,
            x_scale=prior_scale,
            y_scale=ref_scale,
        )
        out["mi_prior_ref"] = mi_prior_ref
        out["mi_prior_recon"] = mi_prior_recon
        out["mi_prior_delta"] = mi_prior_recon - mi_prior_ref
    return out
