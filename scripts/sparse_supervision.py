"""Ill-posed image-space supervision: does the prior help when data is scarce?

Two experiments on a single follow-up slice, reconstructing the FULL image while
supervising the residual on only part of it:

  A. Very sparse random -- observe a small random fraction p of pixels
     (p = 50% down to 2%). Tests interpolation under growing gaps.
  B. Structured hole    -- observe everything EXCEPT a central square hole of
     side h. The hole has no interior samples, so it can only be filled from the
     prior. Tests genuine extrapolation/inpainting.

Methods (all supervised on the same observed pixels):
  * current-only INR -- no prior (the robust floor)
  * prior-finetune   -- NeRP-style, fine-tune the prior INR on observed pixels
  * prior+residual   -- ours: frozen prior + residual
  * prior+gated res. -- ours + spatial trust gate

The hypothesis: as supervision shrinks, current-only collapses while the
prior-anchored methods hold -- i.e. the prior is worth its keep precisely when
data is insufficient (the k-space regime, previewed cheaply in image space).

    python scripts/sparse_supervision.py
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.data.slam import SlamTestImageSlices
from presinr.metrics import all_metrics, psnr, robust_scale
from presinr.models import build_inr
from presinr.recon import ImageFitConfig, PriorFitConfig, fit_image_inr, fit_prior, fit_residual_image
from presinr.utils import get_device, save_magnitude_panel, set_seed, to_numpy

METHODS = ["current-only INR", "prior-finetune (NeRP)",
           "prior+residual (ours)", "prior+gated residual (ours+gate)"]
GATE_KW = dict(residual_bound=1.5, lambda_raw_res=1e-3, lambda_gate=5e-3)


def make_gate():
    g = build_inr("siren", out_features=1, hidden_features=64, hidden_layers=3)
    with torch.no_grad():
        g.net[-1].bias.fill_(-2.0)   # start mostly closed -> trust prior
    return g


def random_mask(H, W, frac, seed, device):
    gen = torch.Generator().manual_seed(seed + int(frac * 1000))
    return (torch.rand(H * W, generator=gen) < frac).float().reshape(H, W).to(device)


def hole_mask(H, W, side, device):
    """Observe everything except a central ``side x side`` square."""
    m = torch.ones(H, W, device=device)
    if side > 0:
        cy, cx = H // 2, W // 2
        h = side // 2
        m[cy - h:cy + h, cx - h:cx + h] = 0.0
    return m


def hole_slice(H, W, side):
    cy, cx = H // 2, W // 2
    h = side // 2
    return slice(cy - h, cy + h), slice(cx - h, cx + h)


def pick_slice(device, change_extent=1):
    """Middle slice of the requested change_extent scan (anatomy-rich)."""
    ds = SlamTestImageSlices(change_extent=change_extent, normalize=True)
    idx = 0
    for i in range(len(ds)):
        if ds[i]["is_middle_slice"]:
            idx = i
            break
    s = ds[idx]
    return s["prior"].to(device), s["recon"].to(device)


def run_methods(prior, target, mask, prior_inr, shape, device, fit_iters):
    out, gate_map = {}, None
    cur = build_inr("siren", out_features=1, hidden_features=256, hidden_layers=4)
    out["current-only INR"], _ = fit_image_inr(cur, target, shape, mask=mask,
                                               cfg=ImageFitConfig(iters=fit_iters, lr=1e-4),
                                               device=device, verbose=False)
    nerp = copy.deepcopy(prior_inr)
    for p in nerp.parameters():
        p.requires_grad_(True)
    out["prior-finetune (NeRP)"], _ = fit_image_inr(nerp, target, shape, mask=mask,
                                                    cfg=ImageFitConfig(iters=fit_iters, lr=1e-4),
                                                    device=device, verbose=False)
    res = build_inr("siren", out_features=1, hidden_features=128, hidden_layers=4)
    out["prior+residual (ours)"], _, _ = fit_residual_image(
        prior_inr, res, target, shape,
        cfg=ImageFitConfig(iters=fit_iters, lambda_res=1e-3), device=device, verbose=False, mask=mask)
    res2 = build_inr("siren", out_features=1, hidden_features=128, hidden_layers=4)
    gated, _, ghist = fit_residual_image(
        prior_inr, res2, target, shape,
        cfg=ImageFitConfig(iters=fit_iters, lambda_res=1e-3, **GATE_KW),
        device=device, verbose=False, mask=mask, gate_inr=make_gate())
    out["prior+gated residual (ours+gate)"] = gated
    gate_map = ghist["gate_map"]
    return out, gate_map


def esc(s):
    return s.replace("_", r"\_").replace("%", r"\%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--change-extent", type=int, default=1)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.5, 0.25, 0.1, 0.05, 0.02])
    ap.add_argument("--holes", type=int, nargs="+", default=[0, 48, 96, 144])
    ap.add_argument("--panel-frac", type=float, default=0.05)
    ap.add_argument("--panel-hole", type=int, default=96)
    ap.add_argument("--prior-iters", type=int, default=2000)
    ap.add_argument("--fit-iters", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="reports/sparse")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    prior, target = pick_slice(device, args.change_extent)
    H, W = target.shape
    scale = robust_scale(target)
    print(f"device={device} slice={(H, W)} change_extent={args.change_extent}")

    prior_inr = build_inr("siren", out_features=1, hidden_features=256, hidden_layers=4)
    fit_prior(prior_inr, prior, PriorFitConfig(iters=args.prior_iters), device=device, verbose=False)

    rand_rows, hole_rows = [], []
    rand_psnr = {m: [] for m in METHODS}
    hole_full_psnr = {m: [] for m in METHODS}
    hole_hole_psnr = {m: [] for m in METHODS}

    # ---- Experiment A: very sparse random ----
    print("\n=== Experiment A: very sparse random ===")
    for fr in args.fractions:
        mask = random_mask(H, W, fr, args.seed, device)
        recons, gate_map = run_methods(prior, target, mask, prior_inr, (H, W), device, args.fit_iters)
        for m in METHODS:
            mt = all_metrics(recons[m], target, prior)
            rand_rows.append({"frac": fr, "method": m, **mt})
            rand_psnr[m].append(mt["psnr"])
        print(f"  p={fr:.2f}: " + "  ".join(f"{m.split()[0]}={all_metrics(recons[m],target,prior)['psnr']:.1f}" for m in METHODS))
        if abs(fr - args.panel_frac) < 1e-9:
            save_magnitude_panel(
                [target, target * mask, recons[METHODS[0]], recons[METHODS[1]],
                 recons[METHODS[2]], recons[METHODS[3]], gate_map],
                ["reference", f"observed ({int(fr*100)}%)", "current-only", "NeRP", "ours", "ours+gate", "gate g"],
                os.path.join(args.out_dir, f"panelA_p{int(fr*100):03d}.png"),
                value_ranges=[None, None, None, None, None, None, (0, 1)],
            )

    # ---- Experiment B: structured hole ----
    print("\n=== Experiment B: structured hole ===")
    for hs in args.holes:
        mask = hole_mask(H, W, hs, device)
        recons, gate_map = run_methods(prior, target, mask, prior_inr, (H, W), device, args.fit_iters)
        hy, hx = hole_slice(H, W, hs) if hs > 0 else (slice(0, H), slice(0, W))
        for m in METHODS:
            mt = all_metrics(recons[m], target, prior)
            hp = psnr(recons[m][hy, hx], target[hy, hx], scale) if hs > 0 else mt["psnr"]
            hole_rows.append({"hole": hs, "method": m, "psnr_hole": hp, **mt})
            hole_full_psnr[m].append(mt["psnr"])
            hole_hole_psnr[m].append(hp)
        print(f"  hole={hs}: " + "  ".join(f"{m.split()[0]}(hole)={hole_hole_psnr[m][-1]:.1f}" for m in METHODS))
        if hs == args.panel_hole:
            save_magnitude_panel(
                [target, target * mask, recons[METHODS[0]], recons[METHODS[1]],
                 recons[METHODS[2]], recons[METHODS[3]], gate_map],
                ["reference", f"observed (hole {hs})", "current-only", "NeRP", "ours", "ours+gate", "gate g"],
                os.path.join(args.out_dir, f"panelB_h{hs:03d}.png"),
                value_ranges=[None, None, None, None, None, None, (0, 1)],
            )

    # ---- curves ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in METHODS:
        ax.plot([f * 100 for f in args.fractions], rand_psnr[m], marker="o", label=m)
    ax.set_xscale("log")
    ax.set_xlabel("observed pixels (%, log)")
    ax.set_ylabel("PSNR vs. full follow-up (dB)")
    ax.set_title("A. Very sparse random supervision")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "curveA.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in METHODS:
        ax.plot(args.holes, hole_hole_psnr[m], marker="o", label=m)
    ax.set_xlabel("hole side (px)")
    ax.set_ylabel("PSNR inside the hole (dB)")
    ax.set_title("B. Structured hole (fill from prior)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "curveB.png"), dpi=130)
    plt.close(fig)

    write_latex(args.out_dir, args, rand_rows, hole_rows)
    print(f"\nsaved report -> {os.path.join(args.out_dir, 'results.tex')}")


def write_latex(out_dir, args, rand_rows, hole_rows):
    L = []
    A = L.append
    A(r"\documentclass[11pt]{article}")
    A(r"\usepackage[margin=0.9in]{geometry}\usepackage{graphicx,booktabs,amsmath,float}")
    A(r"\title{Does the Prior Help when Data is Scarce? Ill-Posed Image-Space Supervision}")
    A(rf"\author{{Patient-Specific Residual INR --- SLAM Ax\_T2\_2D, change\_extent={args.change_extent}}}")
    A(r"\date{\today}\begin{document}\maketitle")

    A(r"\section*{Setup}")
    A(r"We reconstruct a single follow-up slice from only a subset of its pixels, to "
      r"probe the regime that actually matters for accelerated MRI: \emph{insufficient} "
      r"data. The residual (or, for baselines, the reconstruction) is supervised only on "
      r"the observed pixels and evaluated on the \emph{full} image. Four methods share the "
      r"same observed pixels: \textbf{current-only INR} (no prior, the robust floor); "
      r"\textbf{prior-finetune} (NeRP-style, the prior INR fine-tuned on observed pixels); "
      r"\textbf{ours} (frozen prior + residual); and \textbf{ours+gate} (adds a spatial "
      r"trust gate). If the prior is worth its keep, the prior-anchored methods should hold "
      r"up as supervision shrinks while current-only collapses.")

    A(r"\subsection*{A. Very sparse random supervision}")
    A(r"A random fraction $p$ of pixels is observed (scattered across the image), swept from "
      r"50\% down to 2\%. As $p$ falls the gaps between observed pixels grow, so pure "
      r"interpolation (current-only) degrades; a prior can fill the gaps with plausible "
      r"anatomy.")
    A(r"\begin{table}[H]\centering\caption{Very sparse random: full-image metrics vs.\ observed fraction $p$.}")
    A(r"\begin{tabular}{llrrr}\toprule")
    A(r"$p$ & method & PSNR & SSIM & change-gain \\ \midrule")
    for i, fr in enumerate(args.fractions):
        blk = sorted([r for r in rand_rows if r["frac"] == fr], key=lambda r: METHODS.index(r["method"]))
        for j, r in enumerate(blk):
            head = f"{int(fr*100)}\\%" if j == 0 else ""
            nm = esc(r["method"])
            if "gate" in r["method"] or r["method"] == "prior+residual (ours)":
                nm = r"\textbf{" + nm + "}"
            cg = "n/a" if not np.isfinite(r.get("change_gain", float("nan"))) else f"{r['change_gain']:.3f}"
            A(f"{head} & {nm} & {r['psnr']:.2f} & {r['ssim']:.3f} & {cg} \\\\")
        if i < len(args.fractions) - 1:
            A(r"\midrule")
    A(r"\bottomrule\end{tabular}\end{table}")
    A(r"\begin{figure}[H]\centering\includegraphics[width=0.62\linewidth]{curveA.png}")
    A(r"\caption{PSNR vs.\ observed fraction (log axis). A flat curve = robust to data scarcity.}\end{figure}")
    A(r"\begin{figure}[H]\centering")
    A(rf"\includegraphics[width=\linewidth]{{panelA_p{int(args.panel_frac*100):03d}.png}}")
    A(rf"\caption{{Reconstructions at $p={int(args.panel_frac*100)}\%$: reference, observed pixels, then each method, and the gate map (white = distrust prior).}}")
    A(r"\end{figure}")

    A(r"\subsection*{B. Structured hole}")
    A(r"Everything is observed \emph{except} a central square hole of side $h$; the hole has "
      r"no interior samples, so it can only be filled by extrapolation or from the prior. As "
      r"$h$ grows the reconstruction inside the hole depends entirely on the prior.")
    A(r"\begin{table}[H]\centering\caption{Structured hole: PSNR inside the hole (the filled region) and over the full image.}")
    A(r"\begin{tabular}{llrrr}\toprule")
    A(r"hole $h$ & method & PSNR(hole) & PSNR(full) & SSIM(full) \\ \midrule")
    for i, hs in enumerate(args.holes):
        blk = sorted([r for r in hole_rows if r["hole"] == hs], key=lambda r: METHODS.index(r["method"]))
        for j, r in enumerate(blk):
            head = f"{hs}" if j == 0 else ""
            nm = esc(r["method"])
            if "gate" in r["method"] or r["method"] == "prior+residual (ours)":
                nm = r"\textbf{" + nm + "}"
            A(f"{head} & {nm} & {r['psnr_hole']:.2f} & {r['psnr']:.2f} & {r['ssim']:.3f} \\\\")
        if i < len(args.holes) - 1:
            A(r"\midrule")
    A(r"\bottomrule\end{tabular}\end{table}")
    A(r"\begin{figure}[H]\centering\includegraphics[width=0.62\linewidth]{curveB.png}")
    A(r"\caption{PSNR inside the hole vs.\ hole size. current-only cannot fill a hole it never saw.}\end{figure}")
    A(r"\begin{figure}[H]\centering")
    A(rf"\includegraphics[width=\linewidth]{{panelB_h{args.panel_hole:03d}.png}}")
    A(rf"\caption{{Reconstructions with a {args.panel_hole}px hole: reference, observed (hole blanked), then each method, and the gate map.}}")
    A(r"\end{figure}")
    A(r"\end{document}")
    with open(os.path.join(out_dir, "results.tex"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
