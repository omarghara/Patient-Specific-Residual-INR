# Source package guide

The `src` directory contains the installable `presinr` Python package. Its
purpose is to reconstruct an accelerated follow-up MRI slice from undersampled
multi-coil k-space while optionally using a registered magnitude image from an
earlier examination of the same patient.

The package uses the `src` layout: when the project is installed with
`pip install -e .`, imports resolve as `import presinr` even though the package
source lives under `src/presinr`.

## Reconstruction flow

The real k-space pipeline follows this sequence:

1. [`data/slam.py`](presinr/data/slam.py) loads one slice of measured k-space,
   coil sensitivity maps, its sampling mask, the registered prior magnitude,
   and a reference reconstruction used only for evaluation.
2. [`models/inr.py`](presinr/models/inr.py) creates a coordinate network that
   maps normalized 2-D coordinates to image values.
3. [`recon.py`](presinr/recon.py) fits a prior INR to the previous examination
   and then fits one of the follow-up reconstruction variants.
4. [`forward.py`](presinr/forward.py) maps every candidate complex image to
   predicted multi-coil k-space through the differentiable operator
   `A(x) = M F S x`.
5. [`losses.py`](presinr/losses.py) compares predicted and measured k-space and
   supplies the selected regularizers.
6. [`metrics.py`](presinr/metrics.py) evaluates magnitude fidelity and
   longitudinal change after optimization. The reference image is never passed
   to a k-space optimizer.

## Package structure

```text
src/
├── README.md                 this guide
└── presinr/
    ├── __init__.py           public package API
    ├── fft.py                centered orthonormal FFT operations
    ├── forward.py            multi-coil Cartesian SENSE operator
    ├── losses.py             data-consistency and regularization losses
    ├── metrics.py            fidelity, change, and MI metrics
    ├── recon.py              fitting loops and configuration dataclasses
    ├── utils.py              geometry, reproducibility, and plotting helpers
    ├── data/
    │   ├── __init__.py       public data API
    │   ├── phantom.py        synthetic longitudinal MRI example
    │   └── slam.py           SLAM download, preparation, and datasets
    └── models/
        ├── README.md         detailed model guide
        ├── __init__.py       public model API
        ├── inr.py            SIREN and Fourier-feature coordinate networks
        └── composition.py    longitudinal image compositions
```

## Module reference

### `presinr/__init__.py`

Defines the public top-level API. It re-exports the forward operator, metrics,
model compositions, fitting functions, and configuration dataclasses so common
objects can be imported directly:

```python
from presinr import CartesianSense, PriorMagnitudePhaseINR
from presinr import fit_prior, fit_magnitude_phase_residual
```

This file contains no reconstruction logic; it controls which names are part of
the supported package interface.

### `presinr/fft.py`

Provides centered orthonormal Fourier transforms:

- `fftc`: `ifftshift -> fftn(norm="ortho") -> fftshift`.
- `ifftc`: the corresponding centered inverse transform.

The convention matches the LAPS/SLAM reconstruction convention. Changing the
shift order or normalization changes the forward model and invalidates direct
k-space comparison.

### `presinr/forward.py`

Defines `CartesianSense`, the differentiable single-slice multi-coil MRI
operator:

\[
A(x)=MFSx.
\]

- `S` multiplies the image by each complex coil sensitivity map.
- `F` applies the centered 2-D FFT.
- `M` retains measured k-space positions and zeros unmeasured positions.

Important methods:

- `forward(x)`: complex image `(Nx, Ny)` to coil k-space `(Nc, Nx, Ny)`.
- `adjoint(y)`: masked inverse FFT followed by sensitivity-weighted coil
  combination.
- `zero_filled(y)`: the adjoint reconstruction used as a non-learned baseline
  and as a source of approximate follow-up phase.

The sampling mask uses `1` for acquired samples and `0` for missing samples.

### `presinr/losses.py`

Contains objectives shared by the fitting loops:

- `data_consistency`: mean squared complex error over measured k-space samples
  only. Averaging over measured entries keeps its scale comparable across
  acceleration factors.
- `residual_l1`: mean magnitude of a two-channel real/imaginary residual.
- `tv_2d`: anisotropic total variation for scalar 2-D maps.
- `phase_tv_2d`: wrap-invariant phase smoothness using
  `1 - cos(phase_difference)`.
- `gate_l1`: mean absolute gate opening used to encourage compact support.

### `presinr/metrics.py`

Evaluates magnitude reconstructions against the released reference:

- `psnr`, `ssim`, and `nmse`: standard magnitude fidelity metrics.
- `change_cosine`: spatial and sign agreement between reconstructed and true
  signed change relative to the prior; ideal value is `1`.
- `change_gain`: recovered amplitude along the true-change direction; ideal
  value is `1`, while prior copy gives `0` when true change exists.
- `mutual_information`: fixed-bin histogram MI in bits.
- `prior_followup_mutual_information`: MI on the prior/follow-up foreground.
- `all_metrics`: returns the standard, change-aware, and MI diagnostics in one
  dictionary.

`cpe` and `pbs` remain for historical compatibility but are not included in
`all_metrics`: CPE collapses to magnitude MAE, and PBS ignores the location and
sign of change.

MI is descriptive rather than higher-is-better. In particular,
`MI(prior, reconstruction) - MI(prior, reference)` should approach zero; a large
positive value can indicate excessive prior copying.

### `presinr/recon.py`

Contains all optimization loops. It does not define the MRI physics or network
layers; it connects models, coordinates, losses, and the forward operator.

Configuration dataclasses:

- `PriorFitConfig`: prior-image fitting iterations and learning rate.
- `ResidualFitConfig`: complex-residual k-space loss and regularizers.
- `KspaceFitConfig`: prior-free current-only complex INR.
- `PhaseFitConfig`: circular image-space phase initialization.
- `MagnitudePhaseFitConfig`: scalar magnitude-change and independent-phase
  k-space fitting.
- `ImageFitConfig`: image-space proof-of-concept experiments.
- `ReconResult`: complex reconstruction, histories, and optional residual,
  gate, magnitude-change, and phase maps.

Fitting functions:

- `fit_prior`: fits a scalar INR to the registered prior magnitude with L1.
- `fit_phase_inr`: fits a scalar phase INR with a weighted circular loss.
- `fit_current_only`: fits a random two-output complex INR only from current
  k-space; this is the essential prior-free baseline.
- `fit_residual`: freezes the prior and fits the original complex residual,
  optionally with a support gate.
- `fit_magnitude_phase_residual`: freezes the prior, fits scalar magnitude
  change, and fits phase independently.
- `fit_image_inr`: scalar image fitting, including the image-space NeRP-style
  baseline.
- `fit_residual_image`: image-space residual proof of concept with optional
  sparse support gate.

The k-space functions receive only measured k-space, coil maps through the
operator, and the prior where applicable. The reference is not an optimizer
argument.

### `presinr/utils.py`

Shared engineering helpers:

- `set_seed`: seeds Python, NumPy, CPU Torch, and CUDA Torch.
- `get_device`: chooses CUDA when available unless a device is requested.
- `to_numpy`: safely detaches Torch tensors for plotting and metrics.
- `center_crop_to` and `center_pad_to`: exact non-interpolating SLAM geometry
  conversion between the native k-space matrix and stored 256-by-256 grid.
- `resize_to`: interpolation for cases that genuinely require resizing; it must
  not be used to undo known SLAM padding.
- `save_magnitude_panel`: saves magnitude or signed image panels with explicit
  display ranges.
- `misregister`: introduces controlled rigid prior perturbations.
- `save_loss_plot`: plots nonzero optimization-history components on a log
  scale.

### `presinr/data/__init__.py`

Defines the small public data API. It currently re-exports `PhantomSample` and
`make_phantom`. SLAM classes are imported explicitly from `presinr.data.slam`
because downloading and preparing clinical data is a separate workflow.

### `presinr/data/phantom.py`

Builds a synthetic longitudinal multi-coil problem with known ground truth. It
creates:

- a prior magnitude image;
- a current complex image containing localized simulated change and smooth
  phase;
- synthetic coil sensitivity maps;
- an undersampling mask;
- measured k-space generated with the same `CartesianSense` operator used for
  real data.

`PhantomSample` stores these tensors, and `make_phantom` constructs the complete
sample. This module is intended for deterministic pipeline validation, not for
clinical conclusions.

### `presinr/data/slam.py`

Handles the Stanford longitudinal MRI data:

- `pull_metadata` and `pull_volumes`: download metadata and selected raw
  volumes.
- `download_minimal`: obtains the small one-scan test configuration.
- `prepare_test`: converts raw volumes into per-slice k-space examples.
- `fetch_test_image_scans` and `prepare_test_images`: prepare image-only
  prior/reference pairs without downloading k-space.
- `SlamTestSlices`: loads prepared k-space, coil maps, mask, prior, and
  evaluation reference.
- `SlamTestImageSlices`: loads image-only examples used by Stage-0 studies.

For `SlamTestSlices`, the principal shapes are:

```text
ksp, mps             (Nc, Kx, Ky), complex
mask                  (Kx, Ky), real 0/1
prior_native          (Kx, Ky), real magnitude
recon_native          (Kx, Ky), complex reference
prior, recon          (Nx, Ny), stored center-padded images
```

The loader normalizes measured k-space from the robust zero-filled-adjoint
scale and exposes exact native-grid crops to prevent accidental interpolation.

### `presinr/models/__init__.py`

Defines the public model API. It re-exports coordinate grids, the SIREN and
Fourier-feature backbones, the model factory, and both longitudinal composition
classes. Detailed descriptions are in [`models/README.md`](presinr/models/README.md).

### `presinr/models/inr.py`

Defines the coordinate-network primitives:

- `make_coord_grid`: creates flattened `(x, y)` coordinates in `[-1, 1]^2`.
- `SineLayer`: linear transformation followed by a frequency-scaled sine.
- `Siren`: the default periodic MLP used for prior, residual, phase, gate, and
  current-only networks.
- `FourierMLP`: a ReLU MLP operating on fixed random Fourier features, kept as
  an architectural ablation.
- `build_inr`: constructs a requested INR backbone from a short string name.

### `presinr/models/composition.py`

Defines how multiple coordinate networks form one complex follow-up image:

- `PriorResidualINR`: original frozen-prior plus complex-residual formulation,
  with an optional bounded spatial support gate.
- `PriorMagnitudePhaseINR`: phase-aware variation with a scalar magnitude
  change and independent phase field.

See [`presinr/models/README.md`](presinr/models/README.md) for equations,
training roles, limitations, and construction examples.

## Important conventions

- Spatial coordinates are flattened in row-major order and normalized to
  `[-1, 1]`; reshaping restores `(Nx, Ny)` layout.
- Images optimized against k-space are complex tensors.
- The longitudinal DICOM-derived prior is magnitude-only.
- Coil sensitivities and k-space use `torch.complex64`.
- Model fitting occurs on the native k-space grid. Stored-grid padding is only
  restored for reference-aligned visualization and evaluation.
- White/`1` mask entries are measured; black/`0` entries are missing.
- The released reference reconstruction is evaluation-only in every k-space
  experiment.

