"""Supervision sweep across change_extent = 0, 1, 2 (image space).

For each change level, picks the slice with the most actual interval change,
then runs the 25/50/75/100% follow-up-pixel sweep for three methods
(prior-copy, prior-finetune/NeRP, prior+residual/ours) and writes a combined
LaTeX report comparing them. Uses the image-only SLAM path (no k-space).

    python scripts/sweep_change.py
"""

import argparse
import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from presinr.data.slam import SlamTestImageSlices
from presinr.metrics import all_metrics, psnr
from presinr.models import build_inr, make_coord_grid
from presinr.recon import ImageFitConfig, PriorFitConfig, fit_image_inr, fit_prior, fit_residual_image
from presinr.utils import get_device, save_loss_plot, save_magnitude_panel, set_seed, to_numpy


def pick_max_change_slice(change_extent, device):
    """Return (prior, target) for the slice with the largest anatomical change."""
    ds = SlamTestImageSlices(change_extent=change_extent, normalize=True)
    best, best_e = None, -1.0
    for i in range(len(ds)):
        s = ds[i]
        prior, target = s["prior"], s["recon"]
        anat = (prior > 0.05).float()
        e = float(((target - prior).abs() * anat).mean())
        if e > best_e:
            best_e, best = e, (prior, target, s["slice_index"], s["scan_index"])
    prior, target, sl, scan = best
    print(f"  change={change_extent}: scan {scan}, slice {sl}, change-energy={best_e:.4f}")
    return prior.to(device), target.to(device), {
        "scan_index": int(scan),
        "slice_index": int(sl),
        "change_energy": float(best_e),
    }


def pixel_mask(H, W, frac, seed, device):
    # Reuse one random ranking so masks are nested as supervision increases.
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(H * W, generator=g) < frac).float().reshape(H, W).to(device)


def run_methods(prior, target, mask, prior_inr, shape, device, fit_iters, lambda_res):
    out = {}
    out["prior-copy"] = (prior, None)

    nerp = copy.deepcopy(prior_inr)
    for p in nerp.parameters():
        p.requires_grad_(True)
    nerp_recon, _ = fit_image_inr(nerp, target, shape, mask=mask,
                                  cfg=ImageFitConfig(iters=fit_iters, lr=1e-4),
                                  device=device, verbose=False)
    out["prior-finetune (NeRP)"] = (nerp_recon, None)

    residual = build_inr("siren", out_features=1, hidden_features=128, hidden_layers=4)
    recon, res_map, hist = fit_residual_image(prior_inr, residual, target, shape,
                                              cfg=ImageFitConfig(iters=fit_iters, lambda_res=lambda_res),
                                              device=device, verbose=False, mask=mask)
    out["prior+residual (ours)"] = (recon, res_map)
    return out, hist


METHODS = ["prior-copy", "prior-finetune (NeRP)", "prior+residual (ours)"]


def esc(s):
    return s.replace("_", r"\_").replace("%", r"\%")


def write_latex(out_dir, changes, fractions, rows, corr, panel_fracs, prior_psnr, selected):
    def cell(r):
        return (
            f"{r['psnr']:.2f} & {r['ssim']:.3f} & {r['nmse']:.4f} & "
            f"{r['change_cosine']:.3f} & {r['change_gain']:.3f} & "
            f"{r['mi_prior_ref']:.3f} & {r['mi_prior_delta']:.3f}"
        )

    L = []
    A = L.append
    A(r"\documentclass[11pt]{article}")
    A(r"\usepackage[margin=0.9in]{geometry}\usepackage{graphicx,booktabs,amsmath,float}")
    A(r"\title{Image-Space Residual INR: Pixel-Supervision Sweep across Change Extent}")
    A(r"\author{Patient-Specific Residual INR --- SLAM Ax\_T2\_2D, change\_extent $\in\{0,1,2\}$}")
    A(r"\date{\today}\begin{document}\maketitle")
    A(r"\section*{Setup}")
    A(r"Image-space reconstruction $\hat{x}=f^{\mathrm{prior}}_\theta+r_\phi$ with the "
      r"prior INR frozen. The residual is trained on a random fraction $p$ of the "
      r"follow-up pixels and evaluated on the full image. For each change level the "
      r"slice with the largest anatomical change is shown. Baselines: \textbf{prior-copy} "
      r"(DICOM prior) and \textbf{prior-finetune} (NeRP-style: prior INR fine-tuned on the "
      r"observed pixels). This is a deliberately selected stress example per scan-level label, "
      r"not a population estimate. Change cosine/gain are ideally 1; $\Delta MI$ is ideally 0.")
    A(r"\begin{table}[H]\centering\begin{tabular}{lrrr}\toprule")
    A(r"change & scan index & slice index & selection energy \\ \midrule")
    for ce in changes:
        s = selected[ce]
        A(f"{ce} & {s['scan_index']} & {s['slice_index']} & {s['change_energy']:.5f} \\\\")
    A(r"\bottomrule\end{tabular}\end{table}")

    # Prior INR fidelity (how well the frozen prior INR captures the DICOM)
    A(r"\section*{Frozen prior INR vs.\ DICOM}")
    A(r"How faithfully the prior INR $f^{\mathrm{prior}}_\theta$ reproduces the "
      r"registered DICOM prior it is frozen from:")
    A(r"\begin{table}[H]\centering\begin{tabular}{lr}\toprule")
    A(r"change\_extent & prior INR PSNR (dB) \\ \midrule")
    for ce in changes:
        A(f"{ce} & {prior_psnr[ce]:.2f} \\\\")
    A(r"\bottomrule\end{tabular}\end{table}")
    for ce in changes:
        A(r"\begin{figure}[H]\centering")
        A(rf"\includegraphics[width=0.8\linewidth]{{prior_c{ce}.png}}")
        A(rf"\caption{{change\_extent={ce}: DICOM prior, the frozen prior INR "
          rf"reconstruction (PSNR {prior_psnr[ce]:.1f} dB), and their difference.}}")
        A(r"\end{figure}")

    # Headline comparison at a fixed fraction
    for pf in panel_fracs:
        A(r"\begin{table}[H]\centering")
        A(rf"\caption{{Comparison across change extent at $p={int(pf*100)}\%$ observed follow-up pixels.}}")
        A(r"\begin{tabular}{llrrrrrrr}\toprule")
        A(r"change & method & PSNR & SSIM & NMSE & $\Delta$cos & $\Delta$gain & $MI(P,F)$ & $\Delta MI$ \\ \midrule")
        for ce in changes:
            blk = [r for r in rows if r["change"] == ce and abs(r["frac"] - pf) < 1e-9]
            blk = sorted(blk, key=lambda r: METHODS.index(r["method"]))
            for j, r in enumerate(blk):
                head = str(ce) if j == 0 else ""
                nm = esc(r["method"])
                A(f"{head} & {nm} & {cell(r)} \\\\")
            A(r"\midrule" if ce != changes[-1] else "")
        A(r"\bottomrule\end{tabular}\end{table}")

    # Full sweep per change extent
    for ce in changes:
        A(rf"\subsection*{{change\_extent = {ce}: full supervision sweep}}")
        A(r"\begin{table}[H]\centering\begin{tabular}{llrrrrrrr}\toprule")
        A(r"$p$ & method & PSNR & SSIM & NMSE & $\Delta$cos & $\Delta$gain & $MI(P,F)$ & $\Delta MI$ \\ \midrule")
        for i, fr in enumerate(fractions):
            blk = sorted([r for r in rows if r["change"] == ce and abs(r["frac"] - fr) < 1e-9],
                         key=lambda r: METHODS.index(r["method"]))
            for j, r in enumerate(blk):
                head = f"{int(fr*100)}\\%" if j == 0 else ""
                nm = esc(r["method"])
                A(f"{head} & {nm} & {cell(r)} \\\\")
            A(r"\midrule" if i < len(fractions) - 1 else "")
        A(r"\bottomrule\end{tabular}")
        A(r"\end{table}")

    # Correlation of learned residual with true change
    A(r"\subsection*{Learned residual vs.\ true change (ours)}")
    A(r"\begin{table}[H]\centering\begin{tabular}{l" + "rr" * len(changes) + r"}\toprule")
    A(" & " + " & ".join([rf"\multicolumn{{2}}{{c}}{{change {ce}}}" for ce in changes]) + r" \\")
    A(r"$p$ & " + " & ".join(["corr & rel-L1"] * len(changes)) + r" \\ \midrule")
    for fr in fractions:
        cells = []
        for ce in changes:
            c = corr[(ce, fr)]
            cells.append(f"{c['corr']:.3f} & {c['rel_l1']:.3f}")
        A(f"{int(fr*100)}\\% & " + " & ".join(cells) + r" \\")
    A(r"\bottomrule\end{tabular}\end{table}")

    # Panels + residual training convergence
    A(r"\section*{Qualitative panels and residual training convergence (ours)}")
    for ce in changes:
        for pf in panel_fracs:
            A(r"\begin{figure}[H]\centering")
            A(rf"\includegraphics[width=\linewidth]{{panel_c{ce}_p{int(pf*100):03d}.png}}")
            A(rf"\caption{{change\_extent={ce}, $p={int(pf*100)}\%$. Left to right: prior, "
              r"reference, prior+residual (ours), learned residual, true change, "
              r"$|$residual $-$ true change$|$.}")
            A(r"\end{figure}")
            A(r"\begin{figure}[H]\centering")
            A(rf"\includegraphics[width=0.62\linewidth]{{loss_c{ce}_p{int(pf*100):03d}.png}}")
            A(rf"\caption{{Residual training loss (change\_extent={ce}, $p={int(pf*100)}\%$): "
              r"total and its components (data fit, $\lambda_{\mathrm{res}}\|r\|_1$) on a log axis.}")
            A(r"\end{figure}")
    A(r"\end{document}")
    path = os.path.join(out_dir, "results.tex")
    with open(path, "w") as f:
        f.write("\n".join(L))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--changes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--panel-fracs", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--prior-iters", type=int, default=2000)
    ap.add_argument("--fit-iters", type=int, default=1500)
    ap.add_argument("--lambda-res", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="reports/sweep_change")
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"device={device} out={args.out_dir}")

    rows, corr, prior_psnr, selected = [], {}, {}, {}
    for ce in args.changes:
        print(f"\n### change_extent = {ce} ###")
        prior, target, selected[ce] = pick_max_change_slice(ce, device)
        H, W = target.shape
        true_change = target - prior
        coords = make_coord_grid(H, W, device=device)

        prior_inr = build_inr("siren", out_features=1, hidden_features=256, hidden_layers=4)
        fit_prior(prior_inr, prior, PriorFitConfig(iters=args.prior_iters), device=device, verbose=False)

        # Frozen prior INR vs. the real DICOM it reconstructs.
        with torch.no_grad():
            prior_recon = prior_inr(coords)[..., 0].reshape(H, W)
        pp = psnr(prior_recon, prior)
        prior_psnr[ce] = pp
        print(f"  frozen prior INR vs DICOM: PSNR={pp:.2f} dB")
        save_magnitude_panel(
            [prior, prior_recon, (prior_recon - prior)],
            ["DICOM prior", f"frozen prior INR (PSNR {pp:.1f} dB)", "|INR - DICOM|"],
            os.path.join(args.out_dir, f"prior_c{ce}.png"),
        )

        for fr in args.fractions:
            mask = pixel_mask(H, W, fr, args.seed, device)
            res, ours_hist = run_methods(prior, target, mask, prior_inr, (H, W), device, args.fit_iters, args.lambda_res)
            for name in METHODS:
                recon, _ = res[name]
                m = all_metrics(recon, target, prior)
                rows.append({"change": ce, "frac": fr, "method": name, **m})
            recon, res_map = res["prior+residual (ours)"]
            a, b = to_numpy(res_map).ravel(), to_numpy(true_change).ravel()
            corr[(ce, fr)] = {"corr": float(np.corrcoef(a, b)[0, 1]),
                              "rel_l1": float(np.mean(np.abs(a - b)) / (np.mean(np.abs(b)) + 1e-8))}
            om = all_metrics(recon, target, prior)
            print(
                f"  p={fr:.2f}  ours PSNR={om['psnr']:.2f} "
                f"D-cos={om['change_cosine']:.3f} D-gain={om['change_gain']:.3f} "
                f"MI(P,F)={om['mi_prior_ref']:.3f} MI-d={om['mi_prior_delta']:.3f} "
                f"resid-corr={corr[(ce,fr)]['corr']:.3f}"
            )
            if fr in args.panel_fracs:
                save_magnitude_panel(
                    [prior, target, recon, res_map, true_change, (res_map - true_change)],
                    ["prior", "reference", "prior+residual", "residual", "true change", "residual - true change"],
                    os.path.join(args.out_dir, f"panel_c{ce}_p{int(fr*100):03d}.png"),
                    signed=[False, False, False, True, True, False],
                )
                save_loss_plot(
                    ours_hist,
                    os.path.join(args.out_dir, f"loss_c{ce}_p{int(fr*100):03d}.png"),
                    title=f"residual training loss (change={ce}, p={int(fr*100)}%)",
                )

    tex = write_latex(
        args.out_dir,
        args.changes,
        args.fractions,
        rows,
        corr,
        args.panel_fracs,
        prior_psnr,
        selected,
    )
    print(f"\nsaved LaTeX report -> {tex}")


if __name__ == "__main__":
    main()
