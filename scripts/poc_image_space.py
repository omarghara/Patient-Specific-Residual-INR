"""Image-space proof of concept (Stage-0, before k-space).

Fit the prior INR to the DICOM/magnitude prior, then learn a residual INR so
that ``prior + residual`` matches the current reference image *directly in image
space* (no forward model). This tests the core claim -- that the residual branch
captures the interval change -- in isolation from the ill-posed k-space inverse
problem. The k-space data-consistency stage replaces the image-space supervision
later.

    python scripts/poc_image_space.py --source phantom
    python scripts/poc_image_space.py --source slam --index 0 --middle-only
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.metrics import all_metrics
from presinr.models import build_inr
from presinr.recon import ImageFitConfig, PriorFitConfig, fit_prior, fit_residual_image
from presinr.utils import get_device, save_magnitude_panel, set_seed, to_numpy


def load_source(args, device):
    """Return (prior_mag, target_mag) real tensors on the same grid."""
    if args.source == "phantom":
        from presinr.data import make_phantom

        s = make_phantom(n=args.n, n_coils=8, accel=4.0, seed=args.seed, device=device)
        prior = s.prior.float()
        target = s.current.abs().float()
        return prior, target
    # slam
    from presinr.data.slam import SlamTestSlices

    ds = SlamTestSlices(middle_only=args.middle_only, normalize=True)
    print(f"loaded {len(ds)} test slices")
    sample = ds[args.index]
    print(f"change_extent={sample['change_extent']}")
    prior = sample["prior"].abs().float().to(device)
    target = sample["recon"].abs().float().to(device)
    return prior, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["phantom", "slam"], default="phantom")
    ap.add_argument("--n", type=int, default=192, help="phantom size")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--middle-only", action="store_true")
    ap.add_argument("--prior-iters", type=int, default=2000)
    ap.add_argument("--resid-iters", type=int, default=2000)
    ap.add_argument("--lambda-res", type=float, default=1e-3)
    ap.add_argument("--lambda-tv", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device={device} source={args.source}")

    prior, target = load_source(args, device)
    H, W = target.shape
    print(f"grid={(H, W)}")
    true_change = target - prior  # what the residual should recover

    # Stage 1: fit + freeze prior INR to the magnitude prior.
    prior_inr = build_inr("siren", out_features=1, hidden_features=256, hidden_layers=4)
    fit_prior(prior_inr, prior, PriorFitConfig(iters=args.prior_iters), device=device)

    # Stage 2 (image space): residual so prior + residual == reference.
    residual_inr = build_inr("siren", out_features=1, hidden_features=128, hidden_layers=4)
    recon, residual_map, _ = fit_residual_image(
        prior_inr, residual_inr, target, (H, W),
        ImageFitConfig(iters=args.resid_iters, lambda_res=args.lambda_res, lambda_tv=args.lambda_tv),
        device=device,
    )

    # Fidelity + change-aware metrics (target is the "reference", prior is DICOM).
    print("\n=== metrics vs. reference (magnitude, image space) ===")
    print(
        f"{'method':24s} {'PSNR':>7s} {'SSIM':>7s} {'NMSE':>8s} "
        f"{'D-cos':>7s} {'D-gain':>8s} {'MI(P,F)':>8s} {'MI-d':>8s}"
    )
    for name, img in [("prior-copy", prior), ("prior+residual (ours)", recon)]:
        m = all_metrics(img, target, prior)
        print(
            f"{name:24s} {m['psnr']:7.2f} {m['ssim']:7.3f} {m['nmse']:8.4f} "
            f"{m['change_cosine']:7.3f} {m['change_gain']:8.3f} "
            f"{m['mi_prior_ref']:8.3f} {m['mi_prior_delta']:8.3f}"
        )

    # How well did the learned residual recover the true interval change?
    a = to_numpy(residual_map).ravel()
    b = to_numpy(true_change).ravel()
    corr = float(np.corrcoef(a, b)[0, 1])
    rel_l1 = float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + 1e-8))
    print(f"\nresidual vs. true change:  corr={corr:.3f}  rel-L1={rel_l1:.3f}")

    out = args.out or f"reports/poc_image_{args.source}.png"
    save_magnitude_panel(
        [prior, target, recon, residual_map, true_change],
        ["prior (DICOM)", "reference", "prior+residual", "residual (learned)", "true change"],
        out,
        signed=[False, False, False, True, True],
    )
    print(f"saved panel -> {out}")


if __name__ == "__main__":
    main()
