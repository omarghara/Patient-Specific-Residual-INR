"""Image-space supervision sweep: how many follow-up pixels does the residual need?

For each fraction p of *observed* follow-up pixels, train three reconstructions
(supervised only on those p pixels, evaluated on the full image):

  * prior-copy        -- basic baseline: use the DICOM prior as-is (0% follow-up)
  * prior-finetune    -- NeRP-style: INR pretrained on the prior, then fine-tuned
                         on the observed follow-up pixels (prior "looks at" the
                         image it reconstructs); no residual decomposition
  * prior+residual    -- ours: frozen prior INR + residual INR on observed pixels

Writes per-fraction panels (including residual - true change), a prior-INR
reconstruction figure, and a standalone LaTeX report with the results table.

    python scripts/sweep_supervision.py --source slam --middle-only
    python scripts/sweep_supervision.py --source phantom
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.metrics import all_metrics
from presinr.models import build_inr
from presinr.recon import (
    ImageFitConfig,
    PriorFitConfig,
    fit_image_inr,
    fit_prior,
    fit_residual_image,
)
from presinr.utils import get_device, save_magnitude_panel, set_seed, to_numpy


def load_source(args, device):
    if args.source == "phantom":
        from presinr.data import make_phantom

        s = make_phantom(n=args.n, n_coils=8, accel=4.0, seed=args.seed, device=device)
        return s.prior.float(), s.current.abs().float(), "phantom"
    from presinr.data.slam import SlamTestSlices

    ds = SlamTestSlices(middle_only=args.middle_only, normalize=True)
    sample = ds[args.index]
    tag = f"slam (change_extent={sample['change_extent']})"
    return sample["prior"].abs().float().to(device), sample["recon"].abs().float().to(device), tag


def pixel_mask(H, W, frac, seed, device):
    # Reuse one random ranking so masks are nested as supervision increases.
    g = torch.Generator().manual_seed(seed)
    m = (torch.rand(H * W, generator=g) < frac).float().reshape(H, W)
    return m.to(device)


def latex_escape(s):
    return s.replace("_", r"\_").replace("%", r"\%")


def write_latex(out_dir, source_tag, fractions, rows, corr_by_frac):
    """Write fidelity, signed-change, and prior/follow-up MI metrics."""
    lines = []
    A = lines.append
    A(r"\documentclass[11pt]{article}")
    A(r"\usepackage[margin=1in]{geometry}")
    A(r"\usepackage{graphicx,booktabs,amsmath,float}")
    A(r"\usepackage[table]{xcolor}")
    A(r"\title{Image-Space Residual INR: Follow-up Pixel-Supervision Sweep}")
    A(r"\author{Patient-Specific Residual INR --- " + latex_escape(source_tag) + "}")
    A(r"\date{\today}")
    A(r"\begin{document}\maketitle")

    A(r"\section*{Setup}")
    A(r"The current (follow-up) image is reconstructed in image space as "
      r"$\hat{x} = f_\theta^{\mathrm{prior}} + r_\phi$, with the prior INR "
      r"$f_\theta^{\mathrm{prior}}$ fit to the registered DICOM prior and frozen. "
      r"The residual $r_\phi$ is trained using only a fraction $p$ of the follow-up "
      r"pixels (chosen uniformly at random) and evaluated on the full image. "
      r"Baselines: \textbf{prior-copy} (DICOM prior, uses $0\%$ of the follow-up) "
      r"and \textbf{prior-finetune} (NeRP-style: the prior INR itself, initialized "
      r"on the DICOM, fine-tuned on the same observed follow-up pixels).")

    # Metrics table
    A(r"\begin{table}[H]\centering")
    A(r"\caption{Reconstruction vs.\ supervision fraction $p$. Change cosine "
      r"and gain are ideally 1; $\Delta MI=MI(P,\hat F)-MI(P,F)$ is ideally 0.}")
    A(r"\begin{tabular}{llrrrrrrr}")
    A(r"\toprule")
    A(r"$p$ & method & PSNR & SSIM & NMSE & $\Delta$cos & $\Delta$gain & $MI(P,F)$ & $\Delta MI$ \\")
    A(r"\midrule")
    for i, frac in enumerate(fractions):
        block = [r for r in rows if r["frac"] == frac]
        for j, r in enumerate(block):
            head = f"{int(frac*100)}\\%" if j == 0 else ""
            name = latex_escape(r["method"])
            A(f"{head} & {name} & {r['psnr']:.2f} & {r['ssim']:.3f} & "
              f"{r['nmse']:.4f} & {r['change_cosine']:.3f} & {r['change_gain']:.3f} & "
              f"{r['mi_prior_ref']:.3f} & {r['mi_prior_delta']:.3f} \\\\")
        A(r"\midrule" if i < len(fractions) - 1 else "")
    A(r"\bottomrule")
    A(r"\end{tabular}\end{table}")

    # Residual-vs-true-change correlation
    A(r"\begin{table}[H]\centering")
    A(r"\caption{How well the learned residual recovers the true interval change "
      r"$(|x_{\mathrm{ref}}|-|x_{\mathrm{prior}}|)$ for the ours model.}")
    A(r"\begin{tabular}{lrr}\toprule")
    A(r"$p$ (pixels) & corr & rel-L1 \\ \midrule")
    for frac in fractions:
        c = corr_by_frac[frac]
        A(f"{int(frac*100)}\\% & {c['corr']:.3f} & {c['rel_l1']:.3f} \\\\")
    A(r"\bottomrule\end{tabular}\end{table}")

    # Prior INR reconstruction figure
    A(r"\section*{Prior INR reconstruction}")
    A(r"\begin{figure}[H]\centering")
    A(r"\includegraphics[width=\linewidth]{prior_recon.png}")
    A(r"\caption{The frozen prior INR and the DICOM image it reconstructs.}")
    A(r"\end{figure}")

    # Per-fraction panels
    A(r"\section*{Qualitative results (ours) per supervision fraction}")
    for frac in fractions:
        A(r"\begin{figure}[H]\centering")
        A(rf"\includegraphics[width=\linewidth]{{panel_p{int(frac*100):03d}.png}}")
        A(rf"\caption{{$p={int(frac*100)}\%$ of follow-up pixels observed. "
          r"Left to right: DICOM prior, reference follow-up, our reconstruction "
          r"($f^{\mathrm{prior}}+r$), learned residual $r$, true change, and the "
          r"error $|r-\text{true change}|$.}")
        A(r"\end{figure}")

    A(r"\end{document}")
    tex_path = os.path.join(out_dir, "results.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    return tex_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["phantom", "slam"], default="slam")
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--middle-only", action="store_true")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--prior-iters", type=int, default=2500)
    ap.add_argument("--fit-iters", type=int, default=2000)
    ap.add_argument("--lambda-res", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    prior, target, source_tag = load_source(args, device)
    H, W = target.shape
    true_change = target - prior
    out_dir = args.out_dir or f"reports/sweep_{args.source}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"device={device}  source={source_tag}  grid={(H, W)}  out={out_dir}")

    # Prior INR: fit once to the full DICOM prior, then freeze.
    prior_inr = build_inr("siren", out_features=1, hidden_features=256, hidden_layers=4)
    fit_prior(prior_inr, prior, PriorFitConfig(iters=args.prior_iters), device=device)
    with torch.no_grad():
        from presinr.models import make_coord_grid
        prior_recon = prior_inr(make_coord_grid(H, W, device=device))[..., 0].reshape(H, W)
    save_magnitude_panel(
        [prior, prior_recon, (prior_recon - prior)],
        ["DICOM prior", "prior INR recon", "|diff|"],
        os.path.join(out_dir, "prior_recon.png"),
    )

    rows, corr_by_frac = [], {}
    print(
        f"\n{'frac':>5s} {'method':22s} {'PSNR':>7s} {'SSIM':>7s} {'NMSE':>8s} "
        f"{'D-cos':>7s} {'D-gain':>8s} {'MI(P,F)':>8s} {'MI-d':>8s}"
    )
    for frac in args.fractions:
        mask = pixel_mask(H, W, frac, args.seed, device)

        # basic baseline: prior-copy (uses 0% of follow-up)
        m = all_metrics(prior, target, prior)
        rows.append({"frac": frac, "method": "prior-copy", **m})
        print(f"{frac:5.2f} {'prior-copy':22s} {m['psnr']:7.2f} {m['ssim']:7.3f} {m['nmse']:8.4f} "
              f"{m['change_cosine']:7.3f} {m['change_gain']:8.3f} {m['mi_prior_ref']:8.3f} {m['mi_prior_delta']:8.3f}")

        # NeRP-style: prior INR fine-tuned on observed follow-up pixels
        nerp_inr = copy.deepcopy(prior_inr)
        for p in nerp_inr.parameters():
            p.requires_grad_(True)
        nerp_recon, _ = fit_image_inr(
            nerp_inr, target, (H, W), mask=mask,
            cfg=ImageFitConfig(iters=args.fit_iters, lr=1e-4), device=device, verbose=False,
        )
        m = all_metrics(nerp_recon, target, prior)
        rows.append({"frac": frac, "method": "prior-finetune (NeRP)", **m})
        print(f"{frac:5.2f} {'prior-finetune (NeRP)':22s} {m['psnr']:7.2f} {m['ssim']:7.3f} {m['nmse']:8.4f} "
              f"{m['change_cosine']:7.3f} {m['change_gain']:8.3f} {m['mi_prior_ref']:8.3f} {m['mi_prior_delta']:8.3f}")

        # ours: frozen prior + residual on observed pixels
        residual_inr = build_inr("siren", out_features=1, hidden_features=128, hidden_layers=4)
        recon, residual_map, _ = fit_residual_image(
            prior_inr, residual_inr, target, (H, W),
            cfg=ImageFitConfig(iters=args.fit_iters, lambda_res=args.lambda_res),
            device=device, verbose=False, mask=mask,
        )
        m = all_metrics(recon, target, prior)
        rows.append({"frac": frac, "method": "prior+residual (ours)", **m})
        print(f"{frac:5.2f} {'prior+residual (ours)':22s} {m['psnr']:7.2f} {m['ssim']:7.3f} {m['nmse']:8.4f} "
              f"{m['change_cosine']:7.3f} {m['change_gain']:8.3f} {m['mi_prior_ref']:8.3f} {m['mi_prior_delta']:8.3f}")

        a, b = to_numpy(residual_map).ravel(), to_numpy(true_change).ravel()
        corr_by_frac[frac] = {
            "corr": float(np.corrcoef(a, b)[0, 1]),
            "rel_l1": float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + 1e-8)),
        }

        save_magnitude_panel(
            [prior, target, recon, residual_map, true_change, (residual_map - true_change)],
            ["DICOM prior", "reference", "prior+residual", "residual (learned)",
             "true change", "residual - true change"],
            os.path.join(out_dir, f"panel_p{int(frac*100):03d}.png"),
            signed=[False, False, False, True, True, False],
        )

    tex = write_latex(out_dir, source_tag, args.fractions, rows, corr_by_frac)
    print(f"\nsaved LaTeX report -> {tex}")
    print(f"panels + figures    -> {out_dir}/")


if __name__ == "__main__":
    main()
