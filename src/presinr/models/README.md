# Model guide

This directory contains the coordinate-based implicit neural representations
(INRs) and the compositions used to form a follow-up MRI image. All networks are
scan-specific: their parameters are optimized for one patient slice rather than
learned from a population training set.

There are two levels to keep separate:

1. **INR backbones** in [`inr.py`](inr.py) map coordinates to one or more
   numbers.
2. **Image compositions** in [`composition.py`](composition.py) assign meanings
   such as prior magnitude, longitudinal change, phase, or gate to those
   outputs and combine them into one complex MRI image.

The current-only and NeRP-style baselines reuse the same INR backbones but do
not need separate composition classes.

## Common input representation

Every INR receives flattened coordinates

\[
c=(x,y), \qquad c\in[-1,1]^2,
\]

with tensor shape `(Nx * Ny, 2)`. The last network dimension determines the
meaning of its output:

| Output width | Typical interpretation |
|---:|---|
| 1 | magnitude, scalar magnitude change, phase, or gate logit |
| 2 | real and imaginary channels of a complex image or residual |

The networks do not directly receive pixels, k-space samples, or coil maps as
input. MRI physics enters through the differentiable forward operator after the
network has generated an image on the full coordinate grid.

## Backbone models

### `SineLayer`

`SineLayer` applies

\[
h=\sin(\omega_0(Wc+b)).
\]

It uses the initialization proposed for sinusoidal representation networks:
the first layer has a coordinate-appropriate uniform initialization, while
hidden layers scale their initialization by `omega_0`. This is a building block
rather than a complete reconstruction model.

### `Siren`

`Siren` is the default INR backbone. It consists of:

1. a first `SineLayer`;
2. a configurable number of hidden `SineLayer` modules;
3. a final linear layer producing the requested channels.

Why it is used here:

- periodic activations represent fine spatial detail better than a plain ReLU
  coordinate MLP;
- it requires no external training dataset;
- the same implementation can represent magnitude, phase, residuals, and
  gates by changing `out_features`.

Common constructions are:

```python
prior_inr = build_inr("siren", out_features=1, hidden_features=256, hidden_layers=4)
residual_inr = build_inr("siren", out_features=2, hidden_features=128, hidden_layers=4)
phase_inr = build_inr("siren", out_features=1, hidden_features=64, hidden_layers=3)
```

### `FourierMLP`

`FourierMLP` maps coordinates through fixed random Fourier features:

\[
\gamma(c)=\left[\sin(2\pi Bc),\cos(2\pi Bc)\right],
\]

then passes them through a ReLU MLP. Matrix `B` is sampled once and stored as a
non-trainable buffer. `mapping_size`, `sigma`, and `seed` control the feature
distribution.

This is an alternative backbone for ablations. It is not the current default
and changing from SIREN also changes optimization behavior, so learning rates
and iteration counts should be retuned.

### `build_inr`

Factory function selecting a backbone:

```python
model = build_inr("siren", ...)
model = build_inr("fourier", ...)
```

Accepted Fourier aliases are `fourier`, `ffn`, and `fourier_mlp`.

## Longitudinal composition models

### Prior magnitude INR

The prior INR is a scalar SIREN fitted to the registered, DICOM-derived prior
magnitude:

\[
f_{\theta_p}(c)\approx m_{prior}(c).
\]

It is not a separate Python class: it is a `Siren(out_features=1)` with a
specific training role. `fit_prior` in `recon.py` trains it using image-space L1
loss. Longitudinal models freeze its parameters before current k-space fitting.

The prior contains anatomy and magnitude contrast but no raw MRI phase.

### `PriorResidualINR`: original complex-residual model

The original formulation combines the frozen scalar prior with a two-output
complex residual:

\[
\hat{x}(c)=\left[f_{\theta_p}(c)+r_R(c)\right]+i\,r_I(c).
\]

Roles:

| Component | Output | Training state during follow-up fitting |
|---|---:|---|
| Prior INR | 1 magnitude value | Frozen |
| Residual INR | 2 real/imaginary values | Trainable |

`fit_residual` minimizes measured k-space error plus selected residual, TV, and
gate regularizers.

Main advantage:

- simple and flexible; the residual can correct any complex discrepancy
  between the prior and measured follow-up.

Main limitation:

- because the prior is magnitude-only, the imaginary residual must represent
  the entire follow-up phase. The complex residual therefore need not be sparse
  even when longitudinal anatomy is unchanged, weakening its interpretation as
  a pure change map.

Construction:

```python
model = PriorResidualINR(prior_inr, residual_inr)
```

### Gated `PriorResidualINR`

The same class optionally accepts a scalar gate INR:

\[
g(c)=\sigma(g_{\psi}(c)), \qquad
\hat{x}(c)=f_{\theta_p}(c)+g(c)r(c).
\]

The gate is intended as a residual support map: closed regions use the prior,
while open regions allow correction.

The product `g * r` has a scale ambiguity: the gate can shrink while an
unbounded residual grows. For that reason, the implementation requires both:

- a finite `residual_bound`, applied through `bound * tanh(r)`;
- positive regularization of the pre-gate residual during gated fitting.

Typical construction:

```python
model = PriorResidualINR(
    prior_inr,
    residual_inr,
    gate_inr=gate_inr,
    residual_bound=1.0,
)
```

The gate is a learned support variable, not automatically a calibrated
probability that the prior is trustworthy.

### `PriorMagnitudePhaseINR`: phase-aware magnitude-change model

This variation separates longitudinal magnitude change from acquisition phase:

\[
\hat{m}(c)=\max\left(f_{\theta_p}(c)+\Delta m_{\eta}(c),0\right),
\]

\[
\hat{x}(c)=\hat{m}(c)e^{i\phi_{\psi}(c)}.
\]

Roles:

| Component | Output | Training state during follow-up fitting |
|---|---:|---|
| Prior INR | 1 magnitude value | Frozen |
| Magnitude-change INR | 1 scalar `delta_m` | Trainable |
| Phase INR | 1 angle in radians | Trainable |

The complex image is produced with `torch.polar(magnitude, phase)`. Only
`delta_m` receives longitudinal-change sparsity. Phase can receive
wrap-invariant circular TV and can be initialized from the phase of a smoothed
zero-filled follow-up reconstruction.

This separation matters even when magnitude is unchanged. If the true image is

\[
x=m_{prior}e^{i\phi},
\]

the original complex model requires a dense residual

\[
r_R=m_{prior}(\cos\phi-1), \qquad
r_I=m_{prior}\sin\phi.
\]

The magnitude-phase model can instead use `delta_m = 0` and place the entire
acquisition phase in `phi`.

Construction:

```python
model = PriorMagnitudePhaseINR(
    prior_inr,
    magnitude_change_inr,
    phase_inr,
)
```

`fit_magnitude_phase_residual` performs joint k-space fitting after optional
`fit_phase_inr` initialization.

Main advantages:

- the learned scalar magnitude residual is more interpretable as longitudinal
  change;
- phase is not penalized as if it were anatomical change;
- magnitude nonnegativity is explicit.

Current limitations:

- zero-filled phase can be unreliable in low-signal or strongly aliased areas;
- magnitude and phase remain a nonconvex joint inverse problem;
- sparsity and phase-TV strengths require validation across subjects and
  acceleration levels.

## Baseline models built from the same backbones

### Current-only complex INR

This baseline is a random `Siren(out_features=2)` interpreted directly as real
and imaginary image channels:

\[
\hat{x}(c)=f_{\theta,R}(c)+i f_{\theta,I}(c).
\]

`fit_current_only` trains it only from measured current k-space. It never sees
the patient prior or evaluation reference.

Purpose:

- determines whether a prior-informed method improves on the implicit bias of
  an INR alone;
- prevents attributing all INR reconstruction gains to longitudinal data.

It has no dedicated composition class because one two-output network already
produces the complete complex image.

### NeRP-style complex fine-tuning baseline

The notebook contains a real-k-space adaptation of NeRP:

1. create one two-output INR;
2. fit it to `[prior_magnitude, 0]`, embedding the DICOM-derived prior and zero
   phase in its weights;
3. fine-tune all the same weights using measured current k-space.

This is intentionally called **NeRP-style**, not an exact reproduction. The
original NeRP MRI setting used a real-valued image with radial NUFFT, whereas
SLAM provides actual complex multi-coil Cartesian measurements. Here the
fine-tuned network must learn phase that was absent during prior embedding,
which can cause slow optimization and destructive modification of its embedded
magnitude representation.

The implementation lives in
[`notebooks/kspace_inr_pipeline.ipynb`](../../../notebooks/kspace_inr_pipeline.ipynb),
not in a standalone model class.

## Model comparison

| Variant | Uses patient prior? | Follow-up output | Where phase lives | Sparse quantity |
|---|---:|---|---|---|
| Current-only INR | No | Complete real/imag image | Full complex INR | None by default |
| NeRP-style | Initialization | Complete real/imag image | Fine-tuned complex INR | None by default |
| Original residual | Yes, frozen | Complex residual | Complex residual | Complex residual magnitude |
| Gated residual | Yes, frozen | Gate and complex residual | Complex residual | Bounded raw/effective residual and gate |
| Magnitude-phase residual | Yes, frozen | Scalar change and phase | Independent phase INR | Scalar magnitude change |

## Which model should be used?

- Use **current-only INR** as the minimum prior-free scientific baseline.
- Use **original complex residual** to reproduce the initial project proposal
  and measure the cost of placing phase in the residual.
- Use **magnitude-phase residual** when the goal is an interpretable magnitude
  change map from a magnitude-only DICOM prior.
- Use the **gated residual** only for support/gating ablations with the bounded,
  raw-residual-regularized objective.
- Use **NeRP-style** as a conceptual prior-initialization comparator, while
  clearly reporting that it is adapted to complex multi-coil SLAM data.
- Use **FourierMLP** in controlled backbone ablations rather than mixing it into
  the main comparison without retuning.

All final comparisons should use the same slice, measured mask, normalization,
random-seed policy, optimization budget, and evaluation code. Inspect signed
change and magnitude-error maps in addition to PSNR, SSIM, and NMSE.

