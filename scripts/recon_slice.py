"""Reconstruct one real SLAM test slice with the prior+residual INR.

Compares against zero-filled and prior-copy baselines and reports fidelity +
change-aware metrics against the reference recon.

    python scripts/recon_slice.py --index 0 --middle-only

Requires the SLAM data to be downloaded first (scripts/fetch_slam.py).
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.data.slam import SlamTestSlices
from presinr.forward import CartesianSense
from presinr.metrics import all_metrics
from presinr.models import PriorResidualINR, build_inr
from presinr.recon import PriorFitConfig, ResidualFitConfig, fit_prior, fit_residual
from presinr.utils import center_crop_to, center_pad_to, get_device, save_magnitude_panel, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--middle-only", action="store_true")
    ap.add_argument("--prior-iters", type=int, default=3000)
    ap.add_argument("--resid-iters", type=int, default=3000)
    ap.add_argument("--lambda-res", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="reports/recon_slice.png")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()

    ds = SlamTestSlices(middle_only=args.middle_only, normalize=True)
    print(f"loaded {len(ds)} test slices")
    sample = ds[args.index]
    ksp = sample["ksp"].to(device)
    mps = sample["mps"].to(device)
    mask = sample["mask"].to(device)
    assert ksp.shape[-2:] == mps.shape[-2:] == mask.shape, (
        f"shape mismatch ksp{tuple(ksp.shape)} mps{tuple(mps.shape)} mask{tuple(mask.shape)}"
    )
    # SLAM reconstructs on the native k-space grid and then *center-pads* the
    # reference/prior to 256x256. Undo that padding exactly for the prior used by
    # the forward model. Bilinear resizing would squeeze and misalign anatomy.
    H, W = mps.shape[-2:]
    ref_full = sample["recon"].to(device)
    prior_full = sample["prior"].to(device)
    # The loader exposes these exact views as a guard against accidental
    # interpolation; retain a crop fallback for older prepared datasets.
    prior_native = sample.get("prior_native", center_crop_to(prior_full, (H, W))).to(device)
    print(f"native grid={(H, W)}  ref/prior stored={tuple(ref_full.shape)}  "
          f"Nc={mps.shape[0]}  change_extent={sample['change_extent']}")

    op = CartesianSense(mps, mask).to(device)
    zero_filled = op.adjoint(ksp)
    prior_copy = prior_full.to(torch.complex64)

    prior_inr = build_inr("siren", out_features=1, hidden_features=256, hidden_layers=4)
    fit_prior(prior_inr, prior_native, PriorFitConfig(iters=args.prior_iters), device=device)

    residual_inr = build_inr("siren", out_features=2, hidden_features=128, hidden_layers=4)
    model = PriorResidualINR(prior_inr, residual_inr).to(device)
    result = fit_residual(
        model, op, ksp, (H, W),
        ResidualFitConfig(iters=args.resid_iters, lambda_res=args.lambda_res),
        device=device,
    )
    # Return to the released 256x256 reference grid using the dataset's exact
    # padding convention before computing metrics or saving panels.
    zero_filled_full = center_pad_to(zero_filled, ref_full.shape)
    recon_full = center_pad_to(result.recon, ref_full.shape)

    print("\n=== metrics vs. reference recon (magnitude) ===")
    print(
        f"{'method':24s} {'PSNR':>7s} {'SSIM':>7s} {'NMSE':>8s} "
        f"{'D-cos':>7s} {'D-gain':>8s} {'MI(P,F)':>8s} {'MI-d':>8s}"
    )
    for name, img in [
        ("zero-filled", zero_filled_full),
        ("prior-copy", prior_copy),
        ("prior+residual (ours)", recon_full),
    ]:
        m = all_metrics(img, ref_full, prior_full)
        print(
            f"{name:24s} {m['psnr']:7.2f} {m['ssim']:7.3f} {m['nmse']:8.4f} "
            f"{m['change_cosine']:7.3f} {m['change_gain']:8.3f} "
            f"{m['mi_prior_ref']:8.3f} {m['mi_prior_delta']:8.3f}"
        )

    out = save_magnitude_panel(
        [prior_full, ref_full, zero_filled_full, recon_full, (recon_full - ref_full)],
        ["prior", "reference", "zero-filled", "ours", "|ours - ref|"],
        args.out,
    )
    print(f"\nsaved qualitative panel -> {out}")


if __name__ == "__main__":
    main()
