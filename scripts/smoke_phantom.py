"""End-to-end smoke test on the synthetic longitudinal phantom.

Runs stage-1 prior fitting + stage-2 residual reconstruction and compares
against zero-filled and prior-copy baselines. Validates that the full pipeline
(forward model, INRs, losses, metrics) is wired correctly before touching real
SLAM data.

    python scripts/smoke_phantom.py --prior-iters 1500 --resid-iters 1500
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.data import make_phantom
from presinr.forward import CartesianSense
from presinr.metrics import all_metrics
from presinr.models import PriorResidualINR, build_inr
from presinr.recon import PriorFitConfig, ResidualFitConfig, fit_prior, fit_residual
from presinr.utils import get_device, save_magnitude_panel, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--coils", type=int, default=8)
    ap.add_argument("--accel", type=float, default=4.0)
    ap.add_argument("--prior-iters", type=int, default=1500)
    ap.add_argument("--resid-iters", type=int, default=1500)
    ap.add_argument("--lambda-res", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="reports/smoke_phantom.png")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device={device}")

    sample = make_phantom(
        n=args.n, n_coils=args.coils, accel=args.accel, seed=args.seed, device=device
    )
    H = W = args.n
    op = CartesianSense(sample.mps, sample.mask).to(device)

    # Baselines.
    zero_filled = op.adjoint(sample.ksp)
    prior_copy = sample.prior.to(torch.complex64)

    # Stage 1: fit + freeze the prior INR.
    prior_inr = build_inr("siren", in_features=2, out_features=1, hidden_features=256, hidden_layers=4)
    fit_prior(prior_inr, sample.prior, PriorFitConfig(iters=args.prior_iters), device=device)

    # Stage 2: residual INR (slightly smaller) from k-space.
    residual_inr = build_inr("siren", in_features=2, out_features=2, hidden_features=128, hidden_layers=4)
    model = PriorResidualINR(prior_inr, residual_inr).to(device)
    result = fit_residual(
        model, op, sample.ksp, (H, W),
        ResidualFitConfig(iters=args.resid_iters, lambda_res=args.lambda_res),
        device=device,
    )
    recon = result.recon

    gt = sample.current
    prior = sample.prior
    rows = [
        ("zero-filled", zero_filled),
        ("prior-copy", prior_copy),
        ("prior+residual (ours)", recon),
    ]
    print("\n=== metrics vs. ground-truth current (magnitude) ===")
    print(f"{'method':24s} {'PSNR':>7s} {'SSIM':>7s} {'NMSE':>8s} {'CPE':>8s} {'PBS':>7s}")
    for name, img in rows:
        m = all_metrics(img, gt, prior)
        print(f"{name:24s} {m['psnr']:7.2f} {m['ssim']:7.3f} {m['nmse']:8.4f} "
              f"{m['cpe']:8.4f} {m['pbs']:7.3f}")

    out = save_magnitude_panel(
        [prior, gt, zero_filled, recon, (recon - gt)],
        ["prior", "current (GT)", "zero-filled", "ours", "|ours - GT|"],
        args.out,
    )
    print(f"\nsaved qualitative panel -> {out}")


if __name__ == "__main__":
    main()
