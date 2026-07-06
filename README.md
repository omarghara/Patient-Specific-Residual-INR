# Patient-Specific Residual INR for Accelerated Follow-up Brain MRI

Reconstruct an accelerated follow-up brain MRI from undersampled k-space using
the patient's **previous** scan as a patient-specific prior, while preserving
real interval change.

The current slice is modeled as a **frozen prior INR** plus a **learned complex
residual INR**:

```
x_hat(c) = f_prior(c) + r(c)          # (gated variant: f_prior(c) + g(c)·r(c))
```

trained against the MRI forward model `y ≈ M F S x_hat` (mask · FFT · coil
sensitivities) with data-consistency + residual-sparsity losses. See
[context/project-proposal.tex](context/project-proposal.tex) for the full plan
and [context/ramp-up.md](context/ramp-up.md) for the literature review.

## Status

- [x] Environment (`presinr` conda env: torch 2.5.1+cu121, sigpy, MRI stack)
- [x] MRI forward model (Cartesian SENSE, centered FFT — matches LAPS convention)
- [x] SIREN / Fourier-feature INRs, prior+residual composition (+ optional gate)
- [x] Losses (data consistency, residual L1, TV) and metrics (PSNR/SSIM/NMSE/CPE/PBS)
- [x] Synthetic phantom smoke test — validated end-to-end on GPU (PSNR ~44.7)
- [x] SLAM downloader + per-slice loader (no credentials needed)
- [ ] Real-slice reconstruction + baselines (pending full test-split download)
- [ ] Baselines: current-only INR, NeRP-style, LACS; change-aware evaluation
- [ ] Adaptive residual **gate** (headline novelty upgrade)

## Setup

```bash
# environment (already created as `presinr`)
mamba activate presinr
pip install -e .        # optional: makes `presinr` importable without sys.path

pytest tests/           # forward-model adjoint test etc.
```

## Quickstart

```bash
# 1. Validate the whole pipeline on a synthetic longitudinal phantom (no data needed)
python scripts/smoke_phantom.py

# 2. Get data. Our method is scan-specific, so only the TEST split is needed.
python scripts/fetch_slam.py             # minimal: 1 test scan (fast sanity)
python scripts/fetch_slam.py --test-only # full test split with k-space (recommended)

# 3. Reconstruct one real SLAM test slice with baselines + metrics
python scripts/recon_slice.py --index 0 --middle-only
```

## Layout

```
src/presinr/
  fft.py            centered FFT (ifftshift → fftn(ortho) → fftshift)
  forward.py        CartesianSense: y = M F S x  (+ adjoint / zero-filled)
  losses.py         data_consistency, residual_l1, tv_2d, gate_l1
  metrics.py        psnr, ssim, nmse, cpe (change-preservation), pbs (prior-bias)
  recon.py          fit_prior (stage 1), fit_residual (stage 2)
  models/
    inr.py          Siren, FourierMLP, coordinate grid
    composition.py  PriorResidualINR (prior + residual, optional gate)
  data/
    phantom.py      synthetic multi-coil longitudinal phantom
    slam.py         SLAM downloader (mirrors laps.slam) + per-slice dataset
scripts/            fetch_slam, smoke_phantom, recon_slice
configs/            slice_default.yaml
tests/              forward-model / composition correctness
```

## Data

SLAM (Stanford Longitudinally Accelerated MRI Dataset), Stanford Digital
Repository [PURL rq296rb2765](https://purl.stanford.edu/rq296rb2765); introduced
by the LAPS paper ([arXiv:2407.00537](https://arxiv.org/abs/2407.00537),
[code](https://github.com/SetsompopLab/LAPS)). Each test scan provides
multi-coil k-space + coil maps + sampling mask + reference recon + a registered
magnitude prior, with a radiologist `change_extent` label (0/1/2) for
change-stratified evaluation. Downloads over plain HTTP — no credentials.
