"""Two-stage patient-specific residual-INR reconstruction.

Stage 1 (:func:`fit_prior`): fit an INR to the registered prior magnitude image
and freeze it.

Stage 2 (:func:`fit_residual`): with the prior frozen, learn a complex residual
INR from the current undersampled k-space under data consistency, plus residual
sparsity (and optional TV / gate) regularization.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

from .forward import CartesianSense
from .losses import data_consistency, gate_l1, phase_tv_2d, residual_l1, tv_2d
from .models.composition import PriorMagnitudePhaseINR, PriorResidualINR
from .models.inr import make_coord_grid


@dataclass
class PriorFitConfig:
    iters: int = 2000
    lr: float = 1e-4
    log_every: int = 250


@dataclass
class ResidualFitConfig:
    iters: int = 2000
    lr: float = 1e-3
    lambda_res: float = 1e-3
    lambda_tv: float = 0.0
    lambda_gate: float = 0.0
    lambda_raw_res: float = 0.0
    lambda_gate_tv: float = 0.0
    log_every: int = 250


@dataclass
class KspaceFitConfig:
    """Configuration for an unregularized current-only complex INR."""

    iters: int = 2000
    lr: float = 1e-3
    log_every: int = 250


@dataclass
class PhaseFitConfig:
    """Image-space initialization of a phase INR using circular loss."""

    iters: int = 1000
    lr: float = 1e-4
    log_every: int = 250


@dataclass
class MagnitudePhaseFitConfig:
    """K-space fitting of magnitude change and a separate phase field."""

    iters: int = 2000
    lr: float = 1e-3
    phase_lr: Optional[float] = None
    prior_scale_lr: Optional[float] = None
    lambda_change: float = 1e-3
    lambda_change_tv: float = 0.0
    lambda_phase_tv: float = 0.0
    log_every: int = 250


@dataclass
class ImageFitConfig:
    iters: int = 2000
    lr: float = 1e-3
    lambda_res: float = 0.0
    lambda_tv: float = 0.0
    lambda_gate: float = 0.0
    lambda_raw_res: float = 0.0
    lambda_gate_tv: float = 0.0
    residual_bound: Optional[float] = None
    log_every: int = 250


@dataclass
class ReconResult:
    recon: torch.Tensor                       # (H, W) complex
    history: Dict[str, Any] = field(default_factory=dict)
    residual: Optional[torch.Tensor] = None    # (H, W, 2), post-bound/pre-gate
    gate: Optional[torch.Tensor] = None        # (H, W), if enabled
    magnitude_residual: Optional[torch.Tensor] = None  # (H, W), if separated
    phase: Optional[torch.Tensor] = None        # (H, W), radians, if separated
    prior_scale: Optional[float] = None         # acquisition-unit prior multiplier


def fit_prior(
    prior_inr: torch.nn.Module,
    prior_img: torch.Tensor,
    cfg: PriorFitConfig = PriorFitConfig(),
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """Fit ``prior_inr`` (scalar output) to a real magnitude image via L1."""
    device = device or prior_img.device
    prior_inr = prior_inr.to(device)
    H, W = prior_img.shape
    coords = make_coord_grid(H, W, device=device)
    target = prior_img.reshape(-1).to(device).float()

    opt = torch.optim.Adam(prior_inr.parameters(), lr=cfg.lr)
    hist = {"loss": []}
    for it in range(cfg.iters):
        opt.zero_grad(set_to_none=True)
        pred = prior_inr(coords)[..., 0]
        loss = (pred - target).abs().mean()
        loss.backward()
        opt.step()
        hist["loss"].append(loss.item())
        if verbose and (it % cfg.log_every == 0 or it == cfg.iters - 1):
            print(f"[prior] iter {it:5d}  L1={loss.item():.5f}")
    return hist


def fit_phase_inr(
    phase_inr: torch.nn.Module,
    target_phase: torch.Tensor,
    cfg: PhaseFitConfig = PhaseFitConfig(),
    weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, list]:
    """Initialize a scalar phase INR with a wrap-invariant circular loss.

    ``target_phase`` is in radians. Optional nonnegative ``weights`` are useful
    for suppressing unreliable background phase in a zero-filled image.
    """
    device = device or target_phase.device
    phase_inr = phase_inr.to(device)
    height, width = target_phase.shape
    coords = make_coord_grid(height, width, device=device)
    target = target_phase.reshape(-1).to(device).float()
    if weights is None:
        weight = torch.ones_like(target)
    else:
        if tuple(weights.shape) != (height, width):
            raise ValueError(
                f"phase weights must have shape {(height, width)}, got {tuple(weights.shape)}"
            )
        weight = weights.reshape(-1).to(device).float().clamp_min(0.0)
    denominator = weight.sum().clamp_min(1e-8)

    optimizer = torch.optim.Adam(phase_inr.parameters(), lr=cfg.lr)
    history = {"loss": []}
    for iteration in range(cfg.iters):
        optimizer.zero_grad(set_to_none=True)
        prediction = phase_inr(coords)[..., 0]
        loss = ((1.0 - torch.cos(prediction - target)) * weight).sum() / denominator
        loss.backward()
        optimizer.step()
        history["loss"].append(loss.item())
        if verbose and (
            iteration % cfg.log_every == 0 or iteration == cfg.iters - 1
        ):
            print(f"[phase-init] iter {iteration:5d}  circular={loss.item():.6f}")
    return history


def _masked_l1(pred_flat, target_flat, mask_flat):
    """L1 over observed pixels only (mask_flat may be None -> all pixels)."""
    if mask_flat is None:
        return (pred_flat - target_flat).abs().mean()
    return ((pred_flat - target_flat).abs() * mask_flat).sum() / (mask_flat.sum() + 1e-8)


def fit_image_inr(
    inr: torch.nn.Module,
    target: torch.Tensor,
    shape: Tuple[int, int],
    mask: Optional[torch.Tensor] = None,
    cfg: ImageFitConfig = ImageFitConfig(),
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    """Fit a single scalar-output INR to a magnitude image on observed pixels.

    Used both for prior fitting (``mask=None`` -> all pixels) and for the
    NeRP-style baseline where an INR initialized on the prior is fine-tuned on a
    subset of the follow-up pixels (the prior "looks at" the image it
    reconstructs). Returns ``(recon, history)``.
    """
    device = device or target.device
    inr = inr.to(device)
    H, W = shape
    coords = make_coord_grid(H, W, device=device)
    target_flat = target.reshape(-1).to(device).float()
    mask_flat = None if mask is None else mask.reshape(-1).to(device).float()

    opt = torch.optim.Adam(inr.parameters(), lr=cfg.lr)
    hist = {"loss": []}
    for it in range(cfg.iters):
        opt.zero_grad(set_to_none=True)
        pred = inr(coords)[..., 0]
        loss = _masked_l1(pred, target_flat, mask_flat)
        loss.backward()
        opt.step()
        hist["loss"].append(loss.item())
        if verbose and (it % cfg.log_every == 0 or it == cfg.iters - 1):
            print(f"[img-inr]  iter {it:5d}  L1(obs)={loss.item():.5f}")
    with torch.no_grad():
        recon = inr(coords)[..., 0].reshape(H, W)
    return recon.detach(), hist


def fit_residual_image(
    prior_inr: torch.nn.Module,
    residual_inr: torch.nn.Module,
    target: torch.Tensor,
    shape: Tuple[int, int],
    cfg: ImageFitConfig = ImageFitConfig(),
    device: Optional[torch.device] = None,
    verbose: bool = True,
    mask: Optional[torch.Tensor] = None,
    gate_inr: Optional[torch.nn.Module] = None,
):
    """Image-space proof of concept (no forward model).

    With the prior INR frozen, learn a real magnitude residual so that
    ``prior + residual`` matches the reference magnitude image directly. Tests
    whether the residual branch captures the interval change before introducing
    the ill-posed k-space inverse problem.

    ``mask`` (optional, ``(H, W)`` 0/1) restricts supervision to a subset of the
    follow-up pixels; the reconstruction is still produced for the full grid.

    ``gate_inr`` (optional) enables the gated variant ``x = prior + g * r`` with
    ``g = sigmoid(gate_inr(c)) in [0, 1]`` and a ``lambda_gate`` sparsity penalty
    (gate stays closed -> trust the prior -> opens where data demand it). The
    robust gated objective uses a finite ``residual_bound`` and positive
    ``lambda_raw_res`` so shrinking ``g`` cannot be offset by an unbounded ``r``.
    The final gate and raw/effective residual maps are returned in ``history``.

    Returns ``(recon, residual_map, history)`` where both maps are real ``(H, W)``.
    """
    device = device or target.device
    prior_inr = prior_inr.to(device).eval()
    for p in prior_inr.parameters():
        p.requires_grad_(False)
    residual_inr = residual_inr.to(device)

    H, W = shape
    coords = make_coord_grid(H, W, device=device)
    target_flat = target.reshape(-1).to(device).float()
    mask_flat = None if mask is None else mask.reshape(-1).to(device).float()
    with torch.no_grad():
        prior_mag = prior_inr(coords)[..., 0]

    params = list(residual_inr.parameters())
    if gate_inr is not None:
        gate_inr = gate_inr.to(device)
        params += list(gate_inr.parameters())
        if cfg.residual_bound is None or cfg.lambda_raw_res <= 0:
            raise ValueError(
                "a gate requires residual_bound and lambda_raw_res > 0 "
                "to prevent gate/residual scale degeneracy"
            )
    if cfg.residual_bound is not None and (
        not math.isfinite(cfg.residual_bound) or cfg.residual_bound <= 0
    ):
        raise ValueError(f"residual_bound must be finite and positive, got {cfg.residual_bound}")
    opt = torch.optim.Adam(params, lr=cfg.lr)

    def residual_components():
        r = residual_inr(coords)[..., 0]
        if cfg.residual_bound is not None:
            r = cfg.residual_bound * torch.tanh(r)
        if gate_inr is None:
            return r, r, None
        g = torch.sigmoid(gate_inr(coords)[..., 0])
        return r, g * r, g

    # Per-component history (weighted terms sum to "total") for convergence plots.
    hist = {
        "total": [], "data": [], "res_l1": [], "raw_res_l1": [],
        "tv": [], "gate": [], "gate_tv": [],
    }
    for it in range(cfg.iters):
        opt.zero_grad(set_to_none=True)
        r, r_eff, g = residual_components()
        x = prior_mag + r_eff
        fit = _masked_l1(x, target_flat, mask_flat)
        res_l1 = cfg.lambda_res * r_eff.abs().mean()
        raw_res_l1 = cfg.lambda_raw_res * r.abs().mean()
        tv = cfg.lambda_tv * tv_2d(r_eff.reshape(H, W)) if cfg.lambda_tv > 0 else torch.zeros((), device=device)
        gate_pen = cfg.lambda_gate * g.mean() if g is not None else torch.zeros((), device=device)
        gate_tv = (
            cfg.lambda_gate_tv * tv_2d(g.reshape(H, W))
            if g is not None and cfg.lambda_gate_tv > 0
            else torch.zeros((), device=device)
        )
        loss = fit + res_l1 + raw_res_l1 + tv + gate_pen + gate_tv
        loss.backward()
        opt.step()
        hist["total"].append(loss.item())
        hist["data"].append(fit.item())
        hist["res_l1"].append(float(res_l1))
        hist["raw_res_l1"].append(float(raw_res_l1))
        hist["tv"].append(float(tv))
        hist["gate"].append(float(gate_pen))
        hist["gate_tv"].append(float(gate_tv))
        if verbose and (it % cfg.log_every == 0 or it == cfg.iters - 1):
            print(f"[img-resid] iter {it:5d}  total={loss.item():.5f}  data(L1)={fit.item():.5f}")

    with torch.no_grad():
        r, r_eff, g = residual_components()
        recon = (prior_mag + r_eff).reshape(H, W)
        residual_map = r_eff.reshape(H, W)
        hist["gate_map"] = None if g is None else g.reshape(H, W).detach()
        hist["raw_residual_map"] = r.reshape(H, W).detach()
    return recon.detach(), residual_map.detach(), hist


def fit_residual(
    model: PriorResidualINR,
    op: CartesianSense,
    ksp: torch.Tensor,
    shape: Tuple[int, int],
    cfg: ResidualFitConfig = ResidualFitConfig(),
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> ReconResult:
    """Learn the residual (prior frozen) from undersampled k-space."""
    device = device or ksp.device
    model = model.to(device)
    op = op.to(device)
    ksp = ksp.to(device)
    model.freeze_prior()

    if model.gate_inr is not None:
        if model.residual_bound is None or cfg.lambda_raw_res <= 0:
            raise ValueError(
                "a gate requires model.residual_bound and "
                "lambda_raw_res > 0 to prevent gate/residual scale degeneracy"
            )

    H, W = shape
    coords = make_coord_grid(H, W, device=device)
    # Prior is frozen -> evaluate once.
    with torch.no_grad():
        prior_mag = model.prior_magnitude(coords)

    params = list(model.residual_inr.parameters())
    if model.gate_inr is not None:
        params += list(model.gate_inr.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)

    hist = {
        "loss": [], "dc": [], "reg": [], "res_l1": [],
        "raw_res_l1": [], "tv": [], "gate": [], "gate_tv": [],
    }
    for it in range(cfg.iters):
        opt.zero_grad(set_to_none=True)
        r, r_eff, g = model.residual_components(coords)
        x = torch.complex(prior_mag + r_eff[..., 0], r_eff[..., 1]).reshape(H, W)

        y_pred = op(x)
        dc = data_consistency(y_pred, ksp, mask=op.mask)
        res_l1 = cfg.lambda_res * residual_l1(r_eff)
        raw_res_l1 = cfg.lambda_raw_res * residual_l1(r)
        tv = torch.zeros((), device=device)
        gate_pen = torch.zeros((), device=device)
        gate_tv = torch.zeros((), device=device)
        reg = res_l1 + raw_res_l1
        if cfg.lambda_tv > 0:
            rmag = torch.sqrt(r_eff[..., 0] ** 2 + r_eff[..., 1] ** 2 + 1e-12).reshape(H, W)
            tv = cfg.lambda_tv * tv_2d(rmag)
            reg = reg + tv
        if g is not None and cfg.lambda_gate > 0:
            gate_pen = cfg.lambda_gate * gate_l1(g)
            reg = reg + gate_pen
        if g is not None and cfg.lambda_gate_tv > 0:
            gate_tv = cfg.lambda_gate_tv * tv_2d(g.reshape(H, W))
            reg = reg + gate_tv
        loss = dc + reg
        loss.backward()
        opt.step()

        hist["loss"].append(loss.item())
        hist["dc"].append(dc.item())
        hist["reg"].append(float(reg))
        hist["res_l1"].append(float(res_l1))
        hist["raw_res_l1"].append(float(raw_res_l1))
        hist["tv"].append(float(tv))
        hist["gate"].append(float(gate_pen))
        hist["gate_tv"].append(float(gate_tv))
        if verbose and (it % cfg.log_every == 0 or it == cfg.iters - 1):
            print(f"[resid] iter {it:5d}  loss={loss.item():.6f}  dc={dc.item():.6f}")

    with torch.no_grad():
        r, r_eff, g = model.residual_components(coords)
        recon = torch.complex(prior_mag + r_eff[..., 0], r_eff[..., 1]).reshape(H, W)
        residual = r.reshape(H, W, 2).detach()
        gate = None if g is None else g.reshape(H, W).detach()
        hist["effective_residual_map"] = r_eff.reshape(H, W, 2).detach()
        hist["raw_residual_map"] = residual
        hist["gate_map"] = gate
    return ReconResult(recon=recon.detach(), history=hist, residual=residual, gate=gate)


def fit_current_only(
    inr: torch.nn.Module,
    op: CartesianSense,
    ksp: torch.Tensor,
    shape: Tuple[int, int],
    cfg: KspaceFitConfig = KspaceFitConfig(),
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> ReconResult:
    """Fit a random complex INR using only measured follow-up k-space.

    This is the prior-free baseline needed to determine whether a longitudinal
    prior adds value beyond the implicit bias of the coordinate network itself.
    ``inr`` must return real/imaginary channels in its final dimension.
    """
    device = device or ksp.device
    inr = inr.to(device)
    op = op.to(device)
    ksp = ksp.to(device)
    height, width = shape
    coords = make_coord_grid(height, width, device=device)

    with torch.no_grad():
        probe = inr(coords[:1])
    if probe.shape[-1] != 2:
        raise ValueError(
            f"current-only INR must return 2 real/imag channels, got {probe.shape[-1]}"
        )

    optimizer = torch.optim.Adam(inr.parameters(), lr=cfg.lr)
    history = {"loss": [], "dc": []}
    for iteration in range(cfg.iters):
        optimizer.zero_grad(set_to_none=True)
        prediction = inr(coords)
        image = torch.complex(prediction[..., 0], prediction[..., 1]).reshape(
            height, width
        )
        dc = data_consistency(op(image), ksp, mask=op.mask)
        dc.backward()
        optimizer.step()
        history["loss"].append(dc.item())
        history["dc"].append(dc.item())
        if verbose and (
            iteration % cfg.log_every == 0 or iteration == cfg.iters - 1
        ):
            print(f"[current-only] iter {iteration:5d}  dc={dc.item():.6f}")

    with torch.no_grad():
        prediction = inr(coords)
        recon = torch.complex(prediction[..., 0], prediction[..., 1]).reshape(
            height, width
        )
    return ReconResult(recon=recon.detach(), history=history)


def fit_magnitude_phase_residual(
    model: PriorMagnitudePhaseINR,
    op: CartesianSense,
    ksp: torch.Tensor,
    shape: Tuple[int, int],
    cfg: MagnitudePhaseFitConfig = MagnitudePhaseFitConfig(),
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> ReconResult:
    """Fit sparse magnitude change and independent phase from measured k-space."""
    device = device or ksp.device
    model = model.to(device)
    op = op.to(device)
    ksp = ksp.to(device)
    model.freeze_prior()

    height, width = shape
    coords = make_coord_grid(height, width, device=device)
    with torch.no_grad():
        prior_mag = model.prior_magnitude(coords)

    phase_lr = cfg.lr if cfg.phase_lr is None else cfg.phase_lr
    parameter_groups = [
        {"params": model.magnitude_residual_inr.parameters(), "lr": cfg.lr},
        {"params": model.phase_inr.parameters(), "lr": phase_lr},
    ]
    if model.log_prior_scale.requires_grad:
        scale_lr = cfg.lr if cfg.prior_scale_lr is None else cfg.prior_scale_lr
        parameter_groups.append({"params": [model.log_prior_scale], "lr": scale_lr})
    optimizer = torch.optim.Adam(parameter_groups)
    history = {
        "loss": [],
        "dc": [],
        "reg": [],
        "change_l1": [],
        "change_tv": [],
        "phase_tv": [],
        "prior_scale": [],
    }
    for iteration in range(cfg.iters):
        optimizer.zero_grad(set_to_none=True)
        _, delta, magnitude, phase = model.components(coords, prior_mag=prior_mag)
        image = torch.polar(magnitude, phase).reshape(height, width)
        dc = data_consistency(op(image), ksp, mask=op.mask)
        change_l1 = cfg.lambda_change * delta.abs().mean()
        change_tv = (
            cfg.lambda_change_tv * tv_2d(delta.reshape(height, width))
            if cfg.lambda_change_tv > 0
            else torch.zeros((), device=device)
        )
        phase_tv = (
            cfg.lambda_phase_tv * phase_tv_2d(phase.reshape(height, width))
            if cfg.lambda_phase_tv > 0
            else torch.zeros((), device=device)
        )
        regularization = change_l1 + change_tv + phase_tv
        loss = dc + regularization
        loss.backward()
        optimizer.step()

        history["loss"].append(loss.item())
        history["dc"].append(dc.item())
        history["reg"].append(float(regularization))
        history["change_l1"].append(float(change_l1))
        history["change_tv"].append(float(change_tv))
        history["phase_tv"].append(float(phase_tv))
        history["prior_scale"].append(float(model.prior_scale.detach()))
        if verbose and (
            iteration % cfg.log_every == 0 or iteration == cfg.iters - 1
        ):
            print(
                f"[mag-phase] iter {iteration:5d}  "
                f"loss={loss.item():.6f}  dc={dc.item():.6f}"
            )

    with torch.no_grad():
        _, delta, magnitude, phase = model.components(coords, prior_mag=prior_mag)
        recon = torch.polar(magnitude, phase).reshape(height, width)
        magnitude_residual = delta.reshape(height, width).detach()
        phase_map = phase.reshape(height, width).detach()
        history["magnitude_map"] = magnitude.reshape(height, width).detach()
        history["magnitude_residual_map"] = magnitude_residual
        history["phase_map"] = phase_map
    return ReconResult(
        recon=recon.detach(),
        history=history,
        magnitude_residual=magnitude_residual,
        phase=phase_map,
        prior_scale=float(model.prior_scale.detach()),
    )
