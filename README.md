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

Staged plan: **image space first**, then k-space data consistency.

- [x] Environment (`presinr` conda env: torch 2.5.1+cu121, sigpy, MRI stack)
- [x] MRI forward model (Cartesian SENSE, centered FFT — matches LAPS convention)
- [x] SIREN / Fourier-feature INRs, prior+residual composition (+ optional gate)
- [x] Losses (measured-sample data consistency, residual/gate regularization)
- [x] Metrics (PSNR/SSIM/NMSE, signed change cosine/gain, prior-follow-up MI)
- [x] Synthetic phantom smoke test — validated end-to-end on GPU (PSNR ~44.7)
- [x] SLAM downloader + per-slice loader (no credentials needed)
- [x] **Stage 0 — image-space POC**: `prior + residual` fit to the reference
      directly. Phantom: residual↔true-change corr **0.99**, PSNR 44. Real SLAM
      slice: PSNR **37.5**, corr **0.97**. See
      `scripts/poc_image_space.py`.
- [x] Stage 1 — k-space data-consistency reconstruction wired + normalization
      validated (`scripts/recon_slice.py`); exact SLAM center-crop/pad geometry
      is covered by regression tests. Needs regularization tuning.
- [ ] Change-stratified evaluation on the full test split (`--test-only`)
- [ ] Baselines: current-only INR, NeRP-style, LACS
- [x] Image-space residual-support gate with a bounded pre-gate residual,
      pre-gate sparsity, and gate TV (removes the gate/residual scale degeneracy)
- [ ] Gated k-space evaluation and gate calibration/ablation

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

# 3. Stage 0: image-space proof of concept (start here — no forward model)
python scripts/poc_image_space.py --source phantom
python scripts/poc_image_space.py --source slam --middle-only

# 4. Stage 1: k-space data-consistency reconstruction with baselines + metrics
python scripts/recon_slice.py --index 0 --middle-only
```

## Layout

```
src/presinr/
  fft.py            centered FFT (ifftshift → fftn(ortho) → fftshift)
  forward.py        CartesianSense: y = M F S x  (+ adjoint / zero-filled)
  losses.py         data_consistency, residual_l1, tv_2d, gate_l1
  metrics.py        PSNR/SSIM/NMSE, signed change cosine/gain, mutual information
  recon.py          fit_prior (stage 1), fit_residual (stage 2)
  models/
    inr.py          Siren, FourierMLP, coordinate grid
    composition.py  PriorResidualINR (prior + residual, optional gate)
  data/
    phantom.py      synthetic multi-coil longitudinal phantom
    slam.py         SLAM downloader (mirrors laps.slam) + per-slice dataset
scripts/            fetch_slam, smoke_phantom, recon_slice
configs/            slice_default.yaml
tests/              forward, geometry, metrics/MI, losses, and gate constraints
```

Longitudinal reporting uses change cosine (spatial/sign agreement, ideal 1),
change gain (recovered amplitude, ideal 1), and histogram mutual information in
bits. `MI(prior, reference)` describes the case; the reconstruction diagnostic is
`MI(prior, recon) - MI(prior, reference)`, ideally 0. MI is not treated as a
monotonic quality score because copying the prior can increase it.

The generated files currently under `reports/` predate the corrected SLAM
geometry and longitudinal metrics. Regenerate them before using their numbers in
a final comparison.

## Data

SLAM (Stanford Longitudinally Accelerated MRI Dataset), Stanford Digital
Repository [PURL rq296rb2765](https://purl.stanford.edu/rq296rb2765); introduced
by the LAPS paper ([arXiv:2407.00537](https://arxiv.org/abs/2407.00537),
[code](https://github.com/SetsompopLab/LAPS)). Each test scan provides
multi-coil k-space + coil maps + sampling mask + reference recon + a registered
magnitude prior, with a radiologist `change_extent` label (0/1/2) for
change-stratified evaluation. Downloads over plain HTTP — no credentials.
