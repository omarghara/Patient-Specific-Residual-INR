"""Misregistration stress test (image space).

Takes a well-registered ``change_extent=0`` slice, shifts the prior by increasing
amounts, and
reconstructs the follow-up from a partial (default 50%) random pixel set. This
exposes prior bias: methods that over-trust a wrong prior ghost the shifted
anatomy into unobserved regions. Natural between-visit intensity/noise differences
remain; the injected shift is the controlled perturbation, not the only difference.

Methods:
  * prior-copy       -- the (misregistered) DICOM prior
  * current-only INR -- fit an INR to the observed pixels, NO prior (robust floor)
  * prior-finetune   -- NeRP-style: prior INR fine-tuned on observed pixels
  * prior+residual   -- ours: frozen prior + residual
  * prior+gated res. -- ours + spatial trust gate g(c): x = prior + g * r

    python scripts/stress_misreg.py

Writes panels, gate maps, a degradation curve, and a compiled-ready LaTeX report.
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.data.slam import SlamTestImageSlices
from presinr.metrics import all_metrics
from presinr.models import build_inr, make_coord_grid
from presinr.recon import ImageFitConfig, PriorFitConfig, fit_image_inr, fit_prior, fit_residual_image
from presinr.utils import get_device, misregister, save_loss_plot, save_magnitude_panel, set_seed, to_numpy

METHODS = ["prior-copy", "current-only INR", "prior-finetune (NeRP)",
           "prior+residual (ours)", "prior+gated residual (ours+gate)"]


def make_gate():
    g = build_inr("siren", out_features=1, hidden_features=64, hidden_layers=3)
    with torch.no_grad():
        g.net[-1].bias.fill_(-2.0)   # sigmoid(-2) ~= 0.12: closed, but trainable
    return g


def pixel_mask(H, W, frac, seed, device):
    gen = torch.Generator().manual_seed(seed)
    return (torch.rand(H * W, generator=gen) < frac).float().reshape(H, W).to(device)


def pick_clean_slice(device):
    """A middle, anatomy-rich slice with radiologist ``change_extent=0``."""
    ds = SlamTestImageSlices(change_extent=0, normalize=True)
    best, best_e = None, -1
    for i in range(len(ds)):
        s = ds[i]
        e = float((s["prior"] > 0.1).float().mean())   # fraction of brain pixels
        if s["is_middle_slice"] or e > best_e:
            if s["is_middle_slice"]:
                best = s
                break
            best, best_e = s, e
    return best["prior"].to(device), best["recon"].to(device)


def esc(s):
    return s.replace("_", r"\_").replace("%", r"\%")


def write_latex(out_dir, shifts, frac, rows, gate_open):
    def cell(r):
        return (
            f"{r['psnr']:.2f} & {r['ssim']:.3f} & {r['nmse']:.4f} & "
            f"{r['change_cosine']:.3f} & {r['change_gain']:.3f} & "
            f"{r['mi_prior_ref']:.3f}"
        )

    L, A = [], None
    lines = []
    A = lines.append
    A(r"\documentclass[11pt]{article}")
    A(r"\usepackage[margin=0.9in]{geometry}\usepackage{graphicx,booktabs,amsmath,float}")
    A(r"\title{Robustness to Prior Misregistration: a Residual Support Gate}")
    A(r"\author{Patient-Specific Residual INR --- SLAM Ax\_T2\_2D, change\_extent=0}")
    A(r"\date{\today}\begin{document}\maketitle")
    A(r"\section*{Setup}")
    A(rf"A radiologist-labeled change-0 follow-up slice is reconstructed from {int(frac*100)}\% of its "
      r"pixels (random). The prior is deliberately shifted by $s$ pixels before use. Natural "
      r"between-visit differences remain, while shift is the controlled perturbation. A method that over-trusts the "
      r"prior ghosts the shifted anatomy into unobserved regions. \textbf{current-only INR} "
      r"uses no prior (robust floor); \textbf{prior-finetune} is NeRP-style; \textbf{ours} adds "
      r"a frozen prior; \textbf{ours+gate} adds a bounded, raw-residual-regularized spatial "
      r"support gate $g(c)\in[0,1]$. The gate is diagnostic, not a calibrated probability.")

    A(r"\begin{table}[H]\centering")
    A(rf"\caption{{Reconstruction vs.\ prior shift $s$ (px) at {int(frac*100)}\% supervision. "
      r"Change cosine/gain are ideally 1; $MI(P_s,F)$ is descriptive and should decrease "
      r"as the shifted prior becomes less aligned.}")
    A(r"\begin{tabular}{llrrrrrr}\toprule")
    A(r"shift & method & PSNR & SSIM & NMSE & $\Delta$cos & $\Delta$gain & $MI(P_s,F)$ \\ \midrule")
    for s in shifts:
        blk = sorted([r for r in rows if r["shift"] == s], key=lambda r: METHODS.index(r["method"]))
        for j, r in enumerate(blk):
            head = f"{s}" if j == 0 else ""
            nm = esc(r["method"])
            A(f"{head} & {nm} & {cell(r)} \\\\")
        if s != shifts[-1]:
            A(r"\midrule")
    A(r"\bottomrule\end{tabular}\end{table}")

    A(r"\begin{table}[H]\centering")
    A(r"\caption{Mean support-gate opening $\overline{g}$ vs.\ prior shift. Its scale is "
      r"set by the residual bound and regularization and is not a calibrated trust probability.}")
    A(r"\begin{tabular}{l" + "r" * len(shifts) + r"}\toprule")
    A(r"shift (px) & " + " & ".join(str(s) for s in shifts) + r" \\")
    A(r"$\overline{g}$ & " + " & ".join(f"{gate_open[s]:.3f}" for s in shifts) + r" \\ \bottomrule")
    A(r"\end{tabular}\end{table}")

    A(r"\begin{figure}[H]\centering\includegraphics[width=0.7\linewidth]{degradation.png}")
    A(r"\caption{PSNR vs.\ prior shift. The current-only curve is a seeded, prior-independent "
      r"control; the remaining curves quantify sensitivity to the shifted prior.}\end{figure}")

    A(r"\section*{Qualitative panels}")
    for s in shifts:
        A(r"\begin{figure}[H]\centering")
        A(rf"\includegraphics[width=\linewidth]{{panel_s{s:02d}.png}}")
        A(rf"\caption{{Prior shift $s={s}$ px. Left to right: shifted prior, reference, "
          r"NeRP, ours (frozen+residual), ours+gate, and the gate map $g$ "
          r"(fixed display range $[0,1]$; white permits more residual correction).}")
        A(r"\end{figure}")
    A(r"\end{document}")
    path = os.path.join(out_dir, "results.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shifts", type=int, nargs="+", default=[0, 4, 8, 12])
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--angle-per-shift", type=float, default=0.0, help="deg rotation per px shift")
    ap.add_argument("--prior-iters", type=int, default=2000)
    ap.add_argument("--fit-iters", type=int, default=2000)
    ap.add_argument("--lambda-res", type=float, default=1e-3)
    ap.add_argument("--lambda-raw-res", type=float, default=1e-4)
    ap.add_argument("--lambda-gate", type=float, default=5e-3)
    ap.add_argument("--lambda-gate-tv", type=float, default=1e-4)
    ap.add_argument("--residual-bound", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="reports/stress_misreg")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    clean_prior, target = pick_clean_slice(device)
    H, W = target.shape
    mask = pixel_mask(H, W, args.frac, args.seed, device)
    coords = make_coord_grid(H, W, device=device)
    print(f"device={device} slice={(H,W)} frac={args.frac}")

    rows, gate_open, psnr_curve = [], {}, {m: [] for m in METHODS}
    for s in args.shifts:
        print(f"\n### prior shift = {s}px ###")
        angle = args.angle_per_shift * s
        prior_mis = misregister(clean_prior, shift=(s, s), angle=angle)

        # Reset before each method so the same architecture starts from the same
        # weights at every shift. In particular, current-only must be shift-invariant.
        set_seed(args.seed + 10)
        prior_inr = build_inr("siren", out_features=1, hidden_features=256, hidden_layers=4)
        fit_prior(prior_inr, prior_mis, PriorFitConfig(iters=args.prior_iters), device=device, verbose=False)

        recons = {}
        recons["prior-copy"] = prior_mis

        set_seed(args.seed + 20)
        cur = build_inr("siren", out_features=1, hidden_features=256, hidden_layers=4)
        recons["current-only INR"], _ = fit_image_inr(cur, target, (H, W), mask=mask,
                                                      cfg=ImageFitConfig(iters=args.fit_iters, lr=1e-4),
                                                      device=device, verbose=False)
        nerp = copy.deepcopy(prior_inr)
        for p in nerp.parameters():
            p.requires_grad_(True)
        recons["prior-finetune (NeRP)"], _ = fit_image_inr(nerp, target, (H, W), mask=mask,
                                                           cfg=ImageFitConfig(iters=args.fit_iters, lr=1e-4),
                                                           device=device, verbose=False)
        set_seed(args.seed + 30)
        res = build_inr("siren", out_features=1, hidden_features=128, hidden_layers=4)
        recons["prior+residual (ours)"], _, _ = fit_residual_image(
            prior_inr, res, target, (H, W),
            cfg=ImageFitConfig(
                iters=args.fit_iters,
                lambda_res=args.lambda_res,
                lambda_raw_res=args.lambda_raw_res,
                residual_bound=args.residual_bound,
            ),
            device=device, verbose=False, mask=mask)
        set_seed(args.seed + 30)
        res2 = build_inr("siren", out_features=1, hidden_features=128, hidden_layers=4)
        gated, _, ghist = fit_residual_image(
            prior_inr, res2, target, (H, W),
            cfg=ImageFitConfig(
                iters=args.fit_iters,
                lambda_res=args.lambda_res,
                lambda_raw_res=args.lambda_raw_res,
                lambda_gate=args.lambda_gate,
                lambda_gate_tv=args.lambda_gate_tv,
                residual_bound=args.residual_bound,
            ),
            device=device, verbose=False, mask=mask, gate_inr=make_gate())
        recons["prior+gated residual (ours+gate)"] = gated
        gate_map = ghist["gate_map"]
        gate_open[s] = float(gate_map.mean())

        for m in METHODS:
            mt = all_metrics(recons[m], target, prior_mis)
            rows.append({"shift": s, "method": m, **mt})
            psnr_curve[m].append(mt["psnr"])
            print(
                f"  {m:34s} PSNR={mt['psnr']:6.2f} SSIM={mt['ssim']:.3f} "
                f"D-cos={mt['change_cosine']:.3f} D-gain={mt['change_gain']:.3f} "
                f"MI(Ps,F)={mt['mi_prior_ref']:.3f}"
            )
        print(f"  mean gate opening = {gate_open[s]:.3f}")

        save_magnitude_panel(
            [prior_mis, target, recons["prior-finetune (NeRP)"], recons["prior+residual (ours)"],
             recons["prior+gated residual (ours+gate)"], gate_map],
            [f"shifted prior (s={s})", "reference", "NeRP", "ours", "ours+gate", "gate g"],
            os.path.join(args.out_dir, f"panel_s{s:02d}.png"),
            value_ranges=[None, None, None, None, None, (0.0, 1.0)],
        )

    # Degradation curve
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in METHODS:
        ax.plot(args.shifts, psnr_curve[m], marker="o", label=m)
    ax.set_xlabel("prior misregistration shift (px)")
    ax.set_ylabel("PSNR vs. true follow-up (dB)")
    ax.set_title(f"Robustness to prior misregistration ({int(args.frac*100)}% supervision)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "degradation.png"), dpi=130)
    plt.close(fig)

    tex = write_latex(args.out_dir, args.shifts, args.frac, rows, gate_open)
    print(f"\nsaved LaTeX report -> {tex}")


if __name__ == "__main__":
    main()
