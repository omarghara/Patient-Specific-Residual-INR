"""Two-stage patient-specific residual-INR reconstruction.

Stage 1 (:func:`fit_prior`): fit an INR to the registered prior magnitude image
and freeze it.

Stage 2 (:func:`fit_residual`): with the prior frozen, learn a complex residual
INR from the current undersampled k-space under data consistency, plus residual
sparsity (and optional TV / gate) regularization.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from .forward import CartesianSense
from .losses import data_consistency, gate_l1, residual_l1, tv_2d
from .models.composition import PriorResidualINR
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
    log_every: int = 250


@dataclass
class ReconResult:
    recon: torch.Tensor                       # (H, W) complex
    history: Dict[str, list] = field(default_factory=dict)


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

    H, W = shape
    coords = make_coord_grid(H, W, device=device)
    # Prior is frozen -> evaluate once.
    with torch.no_grad():
        prior_mag = model.prior_magnitude(coords)

    params = list(model.residual_inr.parameters())
    if model.gate_inr is not None:
        params += list(model.gate_inr.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)

    hist = {"loss": [], "dc": [], "reg": []}
    for it in range(cfg.iters):
        opt.zero_grad(set_to_none=True)
        r = model.residual(coords)                     # (N, 2)
        g = model.gate(coords)
        r_eff = r if g is None else g[..., None] * r
        x = torch.complex(prior_mag + r_eff[..., 0], r_eff[..., 1]).reshape(H, W)

        y_pred = op(x)
        dc = data_consistency(y_pred, ksp)
        reg = cfg.lambda_res * residual_l1(r_eff)
        if cfg.lambda_tv > 0:
            rmag = torch.sqrt(r_eff[..., 0] ** 2 + r_eff[..., 1] ** 2 + 1e-12).reshape(H, W)
            reg = reg + cfg.lambda_tv * tv_2d(rmag)
        if g is not None and cfg.lambda_gate > 0:
            reg = reg + cfg.lambda_gate * gate_l1(g)
        loss = dc + reg
        loss.backward()
        opt.step()

        hist["loss"].append(loss.item())
        hist["dc"].append(dc.item())
        hist["reg"].append(float(reg))
        if verbose and (it % cfg.log_every == 0 or it == cfg.iters - 1):
            print(f"[resid] iter {it:5d}  loss={loss.item():.6f}  dc={dc.item():.6f}")

    with torch.no_grad():
        r = model.residual(coords)
        g = model.gate(coords)
        r_eff = r if g is None else g[..., None] * r
        recon = torch.complex(prior_mag + r_eff[..., 0], r_eff[..., 1]).reshape(H, W)
    return ReconResult(recon=recon.detach(), history=hist)
