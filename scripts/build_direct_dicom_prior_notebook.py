"""Build the direct-DICOM prior ablation notebook.

The notebook keeps the experiment local and explicit: it compares a fitted
prior INR with direct bilinear lookup of the registered DICOM-derived pixels,
crossed with unit versus acquisition-calibrated intensity scale.  A
parameter-matched current-only complex INR and classical reconstructions are
included as controls.
"""

import json
import textwrap
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks" / "direct_dicom_prior_ablation.ipynb"


def markdown(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        r"""
        # Direct registered DICOM prior versus a fitted prior INR

        This notebook tests whether the registered prior magnitude must first be
        represented by an INR.  On the fixed reconstruction grid we can instead
        use the DICOM-derived pixels directly:

        \[
        \hat m(c_i)=\left[\alpha P(c_i)+\Delta m_\psi(c_i)\right]_+,
        \qquad
        \hat x(c_i)=\hat m(c_i)e^{i\phi_\omega(c_i)}.
        \]

        Here `P` is a fixed image field with **zero trainable parameters**.  The
        magnitude residual and phase remain INRs trained only from measured
        follow-up k-space.

        The experiment is deliberately factorial:

        1. fitted prior INR, scale 1 (current formulation);
        2. direct DICOM, scale 1 (isolates removal of prior fitting);
        3. fitted prior INR, k-space-calibrated scale;
        4. direct DICOM, k-space-calibrated scale (proposed formulation).

        A parameter-matched, current-only complex INR tests whether any gain
        really comes from the longitudinal prior.  CG-SENSE and zero-filled
        reconstructions provide classical context.

        This remains a development experiment on scan 16, slice 23.  Exclude
        the entire scan from held-out reporting.
        """
    ),
    markdown(
        r"""
        ## What is being isolated

        Removing the prior INR changes only the prior representation.  The
        residual network is already a separate network; it never inherits the
        prior weights.  Therefore direct pixels can improve accuracy only by
        avoiding prior-fit approximation/smoothing, and can improve efficiency
        by eliminating the prior fit.

        Intensity scale is a separate question.  DICOM magnitude and follow-up
        k-space have unrelated units, so the reference-free scale is estimated
        from acquired samples and the initialized current phase:

        \[
        \alpha_0 =
        \frac{\operatorname{Re}\langle A(Pe^{i\phi_0}),y\rangle}
        {\|A(Pe^{i\phi_0})\|_2^2}.
        \]

        `alpha_0` is kept fixed in this first test.  This prevents a learnable
        scale and the residual from trading the same global intensity change.
        """
    ),
    code(
        r"""
        from __future__ import annotations

        import copy
        import gc
        import hashlib
        import json
        import math
        import os
        import sys
        import time
        from dataclasses import asdict, dataclass
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from IPython.display import display
        from torchmetrics.functional.image import (
            peak_signal_noise_ratio,
            structural_similarity_index_measure,
        )

        REPO = Path.cwd().resolve()
        if not (REPO / "src" / "presinr").exists():
            REPO = Path("/home/omarg/Patient-Specific-Residual-INR")
        if not (REPO / "src" / "presinr").exists():
            raise RuntimeError(
                "Open this notebook from the Patient-Specific-Residual-INR checkout."
            )
        if str(REPO / "src") not in sys.path:
            sys.path.insert(0, str(REPO / "src"))

        from presinr.baselines.laps_nerp import (
            CenterPaddedSense,
            conjugate_gradient_sense,
        )
        from presinr.calibration import (
            prior_scale_from_kspace,
            real_least_squares_scale,
        )
        from presinr.data.slam import SlamTestSlices
        from presinr.losses import data_consistency, phase_tv_2d
        from presinr.metrics import (
            acquisition_calibrated_longitudinal_metrics,
            all_metrics,
        )
        from presinr.models import PriorMagnitudePhaseINR, build_inr
        from presinr.models.inr import make_coord_grid
        from presinr.recon import PhaseFitConfig, PriorFitConfig, fit_phase_inr, fit_prior
        from presinr.sampling import laps_retrospective_1d_mask
        from presinr.utils import center_pad_to, get_device, set_seed, to_numpy

        DEVICE = get_device()
        print("repository :", REPO)
        print("python     :", sys.executable)
        print("torch      :", torch.__version__)
        print("device     :", DEVICE)
        if DEVICE.type == "cuda":
            print("GPU        :", torch.cuda.get_device_name(DEVICE))
        """
    ),
    markdown(
        r"""
        ## Configuration

        The default run uses three paired initializations at requested
        `R = 6, 9, 13`.  This matters because the fitted-versus-direct difference
        may be smaller than seed variation.  Expected runtime is roughly
        25--35 minutes on the RTX 2080 Ti, depending on cache state.

        For a fast software check, execute with environment variable
        `PRESINR_DIRECT_PRIOR_SMOKE=1`.
        """
    ),
    code(
        r"""
        SMOKE = os.environ.get("PRESINR_DIRECT_PRIOR_SMOKE", "0") == "1"
        BASE_SEED = 42
        CACHE_VERSION = "direct-dicom-prior-v1"

        TARGET_SCAN_INDEX = 16
        TARGET_SLICE_INDEX = 23
        TARGET_CHANGE_EXTENT = 2
        PHASE_ENCODE_DIM = 1

        # Generate the full chain even though the primary experiment uses a
        # subset.  This exactly reproduces the nested masks/common scale from
        # magnitude_phase_acceleration_tuning.ipynb.
        MASK_ACCELERATIONS = (3, 5, 6, 7, 9, 11, 13)
        PRIMARY_RS = (6, 9, 13)
        PAIRED_SEEDS = (0, 1, 2)

        PRIOR_ITERS = 3000
        PHASE_ITERS = 1000
        CURRENT_INIT_ITERS = 1000
        JOINT_ITERS = 1200
        SCHEDULE_HORIZON = 3000
        EVAL_EVERY = 100

        DELTA_WIDTH = 128
        DELTA_LAYERS = 4
        DELTA_LR = 1e-4
        PHASE_LR = 3e-5
        CURRENT_WIDTH = 140  # 79,662 params: close to our 79,298 trainable params
        CURRENT_LR = 1e-4
        LAMBDA_CHANGE = 1e-3
        LAMBDA_PHASE_TV = 1e-5
        GRAD_CLIP = 1.0

        RESUME_CACHE = True
        RUN_FULL_SWEEP = False
        FULL_SWEEP_VARIANT = "direct_calibrated"
        FULL_SWEEP_SEED = 0

        if SMOKE:
            MASK_ACCELERATIONS = (3,)
            PRIMARY_RS = (3,)
            PAIRED_SEEDS = (0,)
            # Calibration needs a recognizable positive magnitude/phase fit;
            # joint reconstruction itself remains a two-step software check.
            PRIOR_ITERS = 100
            PHASE_ITERS = 100
            CURRENT_INIT_ITERS = 20
            JOINT_ITERS = 2
            SCHEDULE_HORIZON = 2
            EVAL_EVERY = 1
            RUN_FULL_SWEEP = False

        OUTPUT_DIR = (
            Path("/tmp/presinr-direct-dicom-prior-smoke")
            if SMOKE
            else REPO / "reports" / "direct_dicom_prior"
        )
        CACHE_DIR = OUTPUT_DIR / "cache"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        print("smoke mode       :", SMOKE)
        print("primary R        :", PRIMARY_RS)
        print("paired seeds     :", PAIRED_SEEDS)
        print("joint iterations :", JOINT_ITERS)
        print("output            :", OUTPUT_DIR)
        """
    ),
    markdown(
        r"""
        ## Load the fixed development slice

        The follow-up reference is available only for evaluation.  It never
        enters model fitting or the acquisition-derived prior scale.
        """
    ),
    code(
        r"""
        dataset = SlamTestSlices(data_dir=REPO / "data", middle_only=False, normalize=False)
        manifest = dataset.df.copy().reset_index(drop=True)
        manifest["dataset_position"] = np.arange(len(manifest))
        match = manifest[
            (manifest["scan_index"] == TARGET_SCAN_INDEX)
            & (manifest["slice_index"] == TARGET_SLICE_INDEX)
        ]
        if len(match) != 1:
            raise RuntimeError(
                f"Expected one sample for scan={TARGET_SCAN_INDEX}, "
                f"slice={TARGET_SLICE_INDEX}; found {len(match)}."
            )
        sample_row = match.iloc[0]
        if int(sample_row["change_extent"]) != TARGET_CHANGE_EXTENT:
            raise RuntimeError("Selected sample no longer has the expected change extent.")
        if int(sample_row["AccelNumDim"]) != 0:
            raise RuntimeError("This ablation requires an originally fully sampled case.")

        sample = dataset[int(sample_row["dataset_position"])]

        def quantile_scale(tensor, q=0.999):
            values = tensor.detach().abs().reshape(-1).float()
            return float(torch.quantile(values, q)) + 1e-8

        reference = sample["recon"].to(torch.complex64)
        prior = sample["prior"].float()
        reference = reference / quantile_scale(reference, 0.999)
        prior = prior / quantile_scale(prior, 0.999)
        stored_shape = tuple(sample["stored_shape"])

        display(
            sample_row[
                [
                    "dataset_position",
                    "index",
                    "scan_index",
                    "slice_index",
                    "change_extent",
                    "AccelNumDim",
                    "scan_plane",
                    "scan_type",
                ]
            ].to_frame("value")
        )
        print("stored shape :", stored_shape)
        print("native shape :", sample["native_shape"])
        print("coils        :", sample["ksp"].shape[0])

        prior_np = np.abs(to_numpy(prior))
        reference_np = np.abs(to_numpy(reference))
        normalized_change = reference_np - prior_np
        change_limit = float(np.quantile(np.abs(normalized_change), 0.995))

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
        axes[0].imshow(prior_np, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("registered prior DICOM magnitude")
        axes[1].imshow(reference_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("follow-up reference (evaluation only)")
        axes[2].imshow(
            normalized_change,
            cmap="coolwarm",
            vmin=-change_limit,
            vmax=change_limit,
        )
        axes[2].set_title("normalized signed change")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Reproduce the controlled nested k-space measurements

        Every higher acceleration mask is a subset of the preceding one.  All
        accelerations use the same phase-encoding direction and one scale
        computed from samples common to the complete mask chain.
        """
    ),
    code(
        r"""
        def stable_seed(*parts, base=BASE_SEED):
            payload = "|".join(map(str, (base,) + parts)).encode("utf-8")
            return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)

        def tensor_hash(tensor):
            array = tensor.detach().cpu().contiguous().numpy()
            return hashlib.sha256(array.tobytes()).hexdigest()[:16]

        raw_kspace = sample["ksp"].to(torch.complex64)
        mps = sample["mps"].to(torch.complex64)

        masks = {}
        mask_info = {}
        source_mask = sample["mask"].float()
        for requested_r in sorted(MASK_ACCELERATIONS):
            current_mask, info = laps_retrospective_1d_mask(
                source_mask,
                requested_r,
                seed=stable_seed(
                    "nested-mask", int(sample_row["index"]), requested_r
                ),
                phase_encode_dim=PHASE_ENCODE_DIM,
                vd_factor=0.8,
                n_candidates=100,
            )
            masks[requested_r] = current_mask
            mask_info[requested_r] = info
            source_mask = current_mask

        mask_stack = torch.stack([masks[r].bool() for r in MASK_ACCELERATIONS])
        common_mask = mask_stack.all(dim=0).float()
        common_operator = CenterPaddedSense(mps, common_mask, stored_shape)
        common_scale = quantile_scale(
            common_operator.adjoint(raw_kspace * common_mask), 0.999
        )

        support_native = torch.linalg.vector_norm(mps, dim=0) > 0.5
        support = center_pad_to(support_native.float(), stored_shape).cpu()

        measurements = {}
        measurement_rows = []
        for requested_r in MASK_ACCELERATIONS:
            operator = CenterPaddedSense(mps, masks[requested_r], stored_shape)
            kspace = raw_kspace * masks[requested_r] / common_scale
            zero_filled = operator.adjoint(kspace)
            measurements[requested_r] = {
                "operator": operator,
                "kspace": kspace,
                "zero_filled": zero_filled,
                "mask": masks[requested_r],
                "info": mask_info[requested_r],
            }
            measurement_rows.append(
                {
                    "requested_r": requested_r,
                    "effective_r": mask_info[requested_r].effective_acceleration,
                    "retained_lines": mask_info[requested_r].output_lines,
                    "center_lines": mask_info[requested_r].center_lines,
                    "phase_encode_dim": mask_info[requested_r].phase_encode_dim,
                }
            )

        measurement_table = pd.DataFrame(measurement_rows)
        display(measurement_table.round(4))
        print("common k-space scale:", common_scale)

        fig, axes = plt.subplots(
            2,
            len(MASK_ACCELERATIONS),
            figsize=(3.0 * len(MASK_ACCELERATIONS), 5.8),
            squeeze=False,
        )
        for column, requested_r in enumerate(MASK_ACCELERATIONS):
            item = measurements[requested_r]
            axes[0, column].imshow(to_numpy(item["mask"]), cmap="gray", vmin=0, vmax=1)
            axes[0, column].set_title(
                f"requested R={requested_r}\n"
                f"effective R={item['info'].effective_acceleration:.2f}"
            )
            axes[1, column].imshow(
                np.abs(to_numpy(item["zero_filled"] * support)), cmap="gray"
            )
            axes[1, column].set_title("zero-filled")
        for axis in axes.flat:
            axis.axis("off")
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Cache, evaluation, and timing helpers

        LAPS-style PSNR/SSIM retain their scalar reference alignment for
        comparability.  Longitudinal metrics are different: they are computed
        in fixed acquisition units, so a calibrated copy of the prior has
        exactly zero reconstructed change.
        """
    ),
    code(
        r"""
        def cache_fingerprint(payload):
            encoded = json.dumps(
                {"version": CACHE_VERSION, **payload},
                sort_keys=True,
                default=str,
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()[:16]

        def cache_path(kind, payload):
            return CACHE_DIR / f"{kind}_{cache_fingerprint(payload)}.pt"

        def cleanup_cuda():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        def count_parameters(module, trainable_only=False):
            return sum(
                parameter.numel()
                for parameter in module.parameters()
                if not trainable_only or parameter.requires_grad
            )

        def align_magnitude_laps(reconstruction, target_reference=reference):
            target = np.abs(to_numpy(target_reference)).astype(np.float64)
            target = target / (np.quantile(target, 0.99) + 1e-12)
            estimate = np.abs(to_numpy(reconstruction)).astype(np.float64)
            denominator = float(np.sum(estimate * estimate))
            gain = 1.0 if denominator <= 1e-20 else float(
                np.sum(estimate * target) / denominator
            )
            return gain * estimate, target, gain

        def laps_metrics(reconstruction, target_reference=reference):
            estimate, target, gain = align_magnitude_laps(
                reconstruction, target_reference
            )
            estimate_t = torch.as_tensor(estimate, dtype=torch.float32)
            target_t = torch.as_tensor(target, dtype=torch.float32)
            return {
                "laps_psnr": float(
                    peak_signal_noise_ratio(
                        estimate_t[None],
                        target_t[None],
                        data_range=1.0,
                        reduction="none",
                        dim=(-2, -1),
                    )
                ),
                "laps_ssim": float(
                    structural_similarity_index_measure(
                        estimate_t[None, None],
                        target_t[None, None],
                        data_range=1.0,
                        reduction="none",
                    )
                ),
                "laps_gain": gain,
            }

        def relative_data_error(reconstruction, operator, kspace):
            with torch.no_grad():
                prediction = operator.to(DEVICE)(reconstruction.to(DEVICE))
                residual = prediction - kspace.to(DEVICE)
                return float(
                    torch.linalg.vector_norm(residual)
                    / (torch.linalg.vector_norm(kspace.to(DEVICE)) + 1e-12)
                )

        def zero_last_linear(module):
            linears = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
            if not linears:
                raise ValueError("Cannot zero-initialize a model without a Linear layer.")
            with torch.no_grad():
                linears[-1].weight.zero_()
                if linears[-1].bias is not None:
                    linears[-1].bias.zero_()

        def cosine_scheduler(optimizer, horizon=SCHEDULE_HORIZON):
            def decay(step):
                fraction = min(float(step) / max(1, horizon), 1.0)
                return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * fraction))
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=decay)
        """
    ),
    markdown(
        r"""
        ## Fit identical phase initialization for every prior method

        Phase comes only from the current acquisition.  It is initialized by a
        small phase INR fitted to the zero-filled phase with magnitude-weighted,
        wrap-invariant circular loss.  Every prior arm at the same `R` loads the
        same phase state.
        """
    ),
    code(
        r"""
        def make_phase_inr():
            return build_inr(
                "siren", out_features=1, hidden_features=64, hidden_layers=3
            )

        phase_memory = {}

        def get_phase_artifact(requested_r):
            if requested_r in phase_memory:
                return phase_memory[requested_r]

            payload = {
                "kind": "phase",
                "sample_index": int(sample_row["index"]),
                "requested_r": requested_r,
                "iters": PHASE_ITERS,
                "common_scale": common_scale,
                "mask_hash": tensor_hash(masks[requested_r]),
            }
            path = cache_path("phase", payload)
            if RESUME_CACHE and path.exists():
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                phase_memory[requested_r] = artifact
                return artifact

            item = measurements[requested_r]
            operator = item["operator"].to(DEVICE)
            kspace = item["kspace"].to(DEVICE)
            zero_filled = operator.adjoint(kspace)
            weights = (
                zero_filled.abs() / quantile_scale(zero_filled, 0.99)
            ).clamp(0.0, 1.0)

            set_seed(stable_seed("phase", int(sample_row["index"]), requested_r))
            network = make_phase_inr().to(DEVICE)
            started = time.perf_counter()
            history = fit_phase_inr(
                network,
                torch.angle(zero_filled),
                cfg=PhaseFitConfig(
                    iters=PHASE_ITERS,
                    lr=1e-4,
                    log_every=max(1, PHASE_ITERS // 10),
                ),
                weights=weights,
                device=DEVICE,
                verbose=False,
            )
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            coords = make_coord_grid(*stored_shape, device=DEVICE)
            with torch.no_grad():
                fitted_phase = network(coords)[..., 0].reshape(stored_shape)
            artifact = {
                "state": {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                },
                "history": history,
                "runtime_seconds": time.perf_counter() - started,
                "fitted_phase": fitted_phase.detach().cpu(),
                "source_phase": torch.angle(zero_filled).detach().cpu(),
            }
            torch.save(artifact, path)
            phase_memory[requested_r] = artifact
            del network, zero_filled
            cleanup_cuda()
            return artifact

        for requested_r in PRIMARY_RS:
            get_phase_artifact(requested_r)

        preview_r = max(PRIMARY_RS)
        preview = get_phase_artifact(preview_r)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        axes[0].imshow(
            to_numpy(preview["source_phase"]),
            cmap="twilight",
            vmin=-math.pi,
            vmax=math.pi,
        )
        axes[0].set_title(f"zero-filled phase, R={preview_r}")
        axes[1].imshow(
            to_numpy(preview["fitted_phase"]),
            cmap="twilight",
            vmin=-math.pi,
            vmax=math.pi,
        )
        axes[1].set_title("shared phase INR initialization")
        axes[2].plot(preview["history"]["loss"])
        axes[2].set_yscale("log")
        axes[2].set_title("circular-loss history")
        for axis in axes[:2]:
            axis.axis("off")
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Construct the two prior representations

        `FixedImageField` performs bilinear lookup of the registered pixels.  At
        the native coordinate grid it must reproduce the input exactly.  It is
        not an INR and has no trainable parameters.

        The fitted arm uses the same 264,193-parameter prior SIREN as the
        previous notebook.  It is fitted once and frozen.
        """
    ),
    code(
        r"""
        class FixedImageField(nn.Module):
            # Zero-parameter continuous lookup of a fixed 2-D image.

            def __init__(self, image):
                super().__init__()
                image = torch.as_tensor(image, dtype=torch.float32)
                if image.ndim != 2:
                    raise ValueError(f"Expected a 2-D image, got {tuple(image.shape)}")
                self.register_buffer("image", image[None, None].clone())

            def forward(self, coords):
                # make_coord_grid stores (row, column); grid_sample expects
                # horizontal/column first and vertical/row second.
                sampling_grid = coords[..., [1, 0]].reshape(1, -1, 1, 2)
                sampled = F.grid_sample(
                    self.image,
                    sampling_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                return sampled[0, 0, :, 0, None]

        def make_prior_inr():
            return build_inr(
                "siren", out_features=1, hidden_features=256, hidden_layers=4
            )

        prior_payload = {
            "kind": "fitted-prior",
            "sample_index": int(sample_row["index"]),
            "iters": PRIOR_ITERS,
            "lr": 1e-4,
            "prior_hash": tensor_hash(prior),
        }
        prior_cache = cache_path("prior", prior_payload)
        if RESUME_CACHE and prior_cache.exists():
            prior_artifact = torch.load(
                prior_cache, map_location="cpu", weights_only=False
            )
        else:
            set_seed(stable_seed("fitted-prior", int(sample_row["index"])))
            prior_network = make_prior_inr().to(DEVICE)
            started = time.perf_counter()
            prior_history = fit_prior(
                prior_network,
                prior.to(DEVICE),
                cfg=PriorFitConfig(
                    iters=PRIOR_ITERS,
                    lr=1e-4,
                    log_every=max(1, PRIOR_ITERS // 10),
                ),
                device=DEVICE,
                verbose=False,
            )
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            prior_artifact = {
                "state": {
                    name: value.detach().cpu().clone()
                    for name, value in prior_network.state_dict().items()
                },
                "history": prior_history,
                "runtime_seconds": time.perf_counter() - started,
            }
            torch.save(prior_artifact, prior_cache)
            del prior_network
            cleanup_cuda()

        coords_cpu = make_coord_grid(*stored_shape)
        direct_field = FixedImageField(prior)
        with torch.no_grad():
            direct_prior = direct_field(coords_cpu)[..., 0].reshape(stored_shape)
        # grid_sample performs float32 coordinate interpolation; on a 256x256
        # align-corners grid the maximum roundoff is about 8e-6.
        if not torch.allclose(direct_prior, prior, atol=1e-5, rtol=1e-5):
            raise RuntimeError("Direct image lookup does not reproduce the prior grid.")
        if count_parameters(direct_field) != 0:
            raise RuntimeError("Direct image field unexpectedly has trainable parameters.")

        fitted_probe = make_prior_inr()
        fitted_probe.load_state_dict(prior_artifact["state"])
        fitted_probe.eval()
        with torch.no_grad():
            fitted_prior = fitted_probe(coords_cpu)[..., 0].reshape(stored_shape)

        fit_error = fitted_prior - prior
        grad_x = torch.zeros_like(prior)
        grad_y = torch.zeros_like(prior)
        grad_x[1:] = prior[1:] - prior[:-1]
        grad_y[:, 1:] = prior[:, 1:] - prior[:, :-1]
        gradient = torch.sqrt(grad_x.square() + grad_y.square())
        foreground = prior > 0.05
        edge_threshold = torch.quantile(gradient[foreground], 0.8)
        edge_mask = foreground & (gradient >= edge_threshold)
        nonedge_mask = foreground & ~edge_mask

        fit_metric = all_metrics(fitted_prior, prior)
        prior_fit_table = pd.DataFrame(
            [
                {
                    "prior_representation": "fitted prior INR",
                    "parameters": count_parameters(fitted_probe),
                    "preparation_seconds": prior_artifact["runtime_seconds"],
                    "mae": float(fit_error.abs().mean()),
                    "rmse": float(torch.sqrt(fit_error.square().mean())),
                    "maximum_error": float(fit_error.abs().max()),
                    "edge_mae": float(fit_error.abs()[edge_mask].mean()),
                    "nonedge_mae": float(fit_error.abs()[nonedge_mask].mean()),
                    "psnr": fit_metric["psnr"],
                    "ssim": fit_metric["ssim"],
                },
                {
                    "prior_representation": "direct DICOM field",
                    "parameters": 0,
                    "preparation_seconds": 0.0,
                    "mae": float((direct_prior - prior).abs().mean()),
                    "rmse": float(torch.sqrt((direct_prior - prior).square().mean())),
                    "maximum_error": float((direct_prior - prior).abs().max()),
                    "edge_mae": float((direct_prior - prior).abs()[edge_mask].mean()),
                    "nonedge_mae": float((direct_prior - prior).abs()[nonedge_mask].mean()),
                    "psnr": float("inf"),
                    "ssim": 1.0,
                },
            ]
        )
        display(prior_fit_table.round(6))

        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
        axes[0].imshow(to_numpy(prior), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("direct registered prior")
        axes[1].imshow(to_numpy(fitted_prior), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("fitted prior INR")
        axes[2].imshow(np.abs(to_numpy(fit_error)), cmap="magma", vmin=0, vmax=0.05)
        axes[2].set_title("absolute fit error")
        axes[3].imshow(to_numpy(edge_mask), cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("edge diagnostic region")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        plt.show()
        del fitted_probe
        cleanup_cuda()
        """
    ),
    markdown(
        r"""
        ## Calibrate each prior representation from measured k-space

        The phase used here is the exact fitted phase initialization that the
        reconstruction model receives.  Scale is calculated independently for
        the fitted and direct prior fields.  The reference-to-acquisition factor
        is evaluation-only and never enters reconstruction.
        """
    ),
    code(
        r"""
        prior_scales = {"direct": {}, "fitted": {}}
        reference_scales = {}
        calibration_rows = []

        for requested_r in PRIMARY_RS:
            item = measurements[requested_r]
            operator = item["operator"].to(DEVICE)
            kspace = item["kspace"].to(DEVICE)
            phase0 = get_phase_artifact(requested_r)["fitted_phase"].to(DEVICE)

            direct_alpha = float(
                prior_scale_from_kspace(
                    prior.to(DEVICE), phase0, operator, kspace
                ).detach()
            )
            fitted_alpha = float(
                prior_scale_from_kspace(
                    fitted_prior.clamp_min(0).to(DEVICE), phase0, operator, kspace
                ).detach()
            )
            reference_alpha = float(
                real_least_squares_scale(
                    operator(reference.to(DEVICE)),
                    kspace,
                    weights=operator.mask,
                ).detach()
            )
            if min(direct_alpha, fitted_alpha, reference_alpha) <= 0:
                raise RuntimeError("Calibration returned a nonpositive scale.")

            prior_scales["direct"][requested_r] = direct_alpha
            prior_scales["fitted"][requested_r] = fitted_alpha
            reference_scales[requested_r] = reference_alpha

            direct_candidate = torch.polar(prior.to(DEVICE), phase0)
            fitted_candidate = torch.polar(
                fitted_prior.clamp_min(0).to(DEVICE), phase0
            )
            unit_direct_error = relative_data_error(
                direct_candidate.detach().cpu(), item["operator"], item["kspace"]
            )
            calibrated_direct_error = relative_data_error(
                (direct_alpha * direct_candidate).detach().cpu(),
                item["operator"],
                item["kspace"],
            )
            unit_fitted_error = relative_data_error(
                fitted_candidate.detach().cpu(), item["operator"], item["kspace"]
            )
            calibrated_fitted_error = relative_data_error(
                (fitted_alpha * fitted_candidate).detach().cpu(),
                item["operator"],
                item["kspace"],
            )
            calibration_rows.append(
                {
                    "requested_r": requested_r,
                    "effective_r": item["info"].effective_acceleration,
                    "reference_to_acquisition": reference_alpha,
                    "direct_prior_alpha": direct_alpha,
                    "fitted_prior_alpha": fitted_alpha,
                    "direct_unit_data_error": unit_direct_error,
                    "direct_calibrated_data_error": calibrated_direct_error,
                    "fitted_unit_data_error": unit_fitted_error,
                    "fitted_calibrated_data_error": calibrated_fitted_error,
                }
            )
            del operator, kspace, phase0
            cleanup_cuda()

        calibration_table = pd.DataFrame(calibration_rows)
        calibration_table.to_csv(OUTPUT_DIR / "calibration.csv", index=False)
        display(calibration_table.round(5))

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
        axes[0].plot(
            calibration_table["effective_r"],
            calibration_table["reference_to_acquisition"],
            marker="o",
            label="reference (evaluation only)",
        )
        axes[0].plot(
            calibration_table["effective_r"],
            calibration_table["direct_prior_alpha"],
            marker="o",
            label="direct prior",
        )
        axes[0].plot(
            calibration_table["effective_r"],
            calibration_table["fitted_prior_alpha"],
            marker="o",
            label="fitted prior",
        )
        axes[0].axhline(1.0, color="black", linewidth=1, linestyle="--")
        axes[0].set_title("Acquisition-derived intensity scale")
        axes[0].set_xlabel("effective acceleration")
        axes[0].set_ylabel("multiplicative scale")
        axes[0].legend(frameon=False, fontsize=8)

        axes[1].plot(
            calibration_table["effective_r"],
            calibration_table["direct_unit_data_error"],
            marker="o",
            label="direct, scale 1",
        )
        axes[1].plot(
            calibration_table["effective_r"],
            calibration_table["direct_calibrated_data_error"],
            marker="o",
            label="direct, calibrated",
        )
        axes[1].set_title("Prior initialization data error")
        axes[1].set_xlabel("effective acceleration")
        axes[1].set_ylabel("relative acquired-k-space error")
        axes[1].legend(frameon=False, fontsize=8)
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Acquisition-calibrated metrics and classical controls

        `change_cosine`, `change_gain`, and MI are measured relative to the
        direct registered DICOM after mapping both prior and reference into
        acquisition units.  The definition is method-independent.

        A high-change ROI is defined only for evaluation from the calibrated
        reference/prior difference.  It never enters training.
        """
    ),
    code(
        r"""
        def full_metrics(raw_reconstruction, requested_r):
            item = measurements[requested_r]
            evaluated = raw_reconstruction.detach().cpu() * support
            reference_acquisition = reference * reference_scales[requested_r]
            prior_acquisition = prior * prior_scales["direct"][requested_r]

            aligned, target, alignment_gain = align_magnitude_laps(evaluated, reference)
            aligned_error = aligned - target
            acquisition_error = (
                evaluated.abs().numpy() - reference_acquisition.abs().numpy()
            )
            foreground = (
                (reference_acquisition.abs() > 0.05)
                | (prior_acquisition.abs() > 0.05)
            ).numpy()
            true_change = (
                reference_acquisition.abs() - prior_acquisition.abs()
            ).numpy()
            threshold = np.quantile(np.abs(true_change)[foreground], 0.90)
            change_roi = foreground & (np.abs(true_change) >= threshold)
            stable_roi = foreground & ~change_roi

            longitudinal = acquisition_calibrated_longitudinal_metrics(
                evaluated,
                reference,
                prior,
                reference_to_acquisition=reference_scales[requested_r],
                prior_to_acquisition=prior_scales["direct"][requested_r],
            )
            acquisition_global = all_metrics(evaluated, reference_acquisition)
            output = {
                **laps_metrics(evaluated, reference),
                "alignment_gain": alignment_gain,
                "aligned_rmse": float(np.sqrt(np.mean(aligned_error**2))),
                "acquisition_psnr": acquisition_global["psnr"],
                "acquisition_ssim": acquisition_global["ssim"],
                "acquisition_nmse": acquisition_global["nmse"],
                "magnitude_mae": float(np.mean(np.abs(acquisition_error))),
                "magnitude_rmse": float(np.sqrt(np.mean(acquisition_error**2))),
                "magnitude_error_p95": float(
                    np.quantile(np.abs(acquisition_error), 0.95)
                ),
                "magnitude_error_max": float(np.max(np.abs(acquisition_error))),
                "change_roi_mae": float(
                    np.mean(np.abs(acquisition_error[change_roi]))
                ),
                "stable_roi_mae": float(
                    np.mean(np.abs(acquisition_error[stable_roi]))
                ),
                "data_error": relative_data_error(
                    evaluated, item["operator"], item["kspace"]
                ),
                **longitudinal,
            }
            return output

        baseline_rows = []
        baseline_recons = {}
        for requested_r in PRIMARY_RS:
            item = measurements[requested_r]
            operator = item["operator"].to(DEVICE)
            kspace = item["kspace"].to(DEVICE)
            started = time.perf_counter()
            cg = conjugate_gradient_sense(
                operator,
                kspace,
                num_iters=25,
                lambda_l2=1e-3,
                tolerance=1e-10,
            ).detach().cpu()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            cg_seconds = time.perf_counter() - started
            baseline_recons[requested_r] = {
                "Zero-filled": item["zero_filled"].detach().cpu(),
                "CG-SENSE": cg,
            }
            for method, reconstruction, runtime in (
                ("Zero-filled", item["zero_filled"], 0.0),
                ("CG-SENSE", cg, cg_seconds),
            ):
                baseline_rows.append(
                    {
                        "method": method,
                        "requested_r": requested_r,
                        "effective_r": item["info"].effective_acceleration,
                        "joint_runtime_seconds": runtime,
                        **full_metrics(reconstruction, requested_r),
                    }
                )
            del operator, kspace
            cleanup_cuda()

        baseline_table = pd.DataFrame(baseline_rows)
        display(
            baseline_table[
                [
                    "method",
                    "requested_r",
                    "effective_r",
                    "laps_psnr",
                    "laps_ssim",
                    "change_cosine",
                    "change_gain",
                    "data_error",
                ]
            ].round(4)
        )
        """
    ),
    markdown(
        r"""
        ## Models and paired checkpoint-aware trainer

        All four prior arms share the exact same zero-initialized residual state
        and phase state for each `(R, seed)` pair.  Thus their first output is
        precisely `alpha * prior * exp(i*phase0)`.

        The current-only control is a two-output complex SIREN with 79,662
        parameters, close to our 79,298 trainable parameters.  It receives no
        DICOM information and is initialized by fitting the zero-filled complex
        image, which is derived only from current k-space.
        """
    ),
    code(
        r"""
        @dataclass(frozen=True)
        class VariantSpec:
            name: str
            prior_mode: str  # fitted, direct, or none
            scale_mode: str  # unit, calibrated, or none

        VARIANTS = (
            VariantSpec("fitted_unit", "fitted", "unit"),
            VariantSpec("direct_unit", "direct", "unit"),
            VariantSpec("fitted_calibrated", "fitted", "calibrated"),
            VariantSpec("direct_calibrated", "direct", "calibrated"),
            VariantSpec("current_only_complex", "none", "none"),
        )

        def make_delta_inr():
            return build_inr(
                "siren",
                out_features=1,
                hidden_features=DELTA_WIDTH,
                hidden_layers=DELTA_LAYERS,
            )

        def make_current_complex_inr():
            return build_inr(
                "siren",
                out_features=2,
                hidden_features=CURRENT_WIDTH,
                hidden_layers=DELTA_LAYERS,
            )

        shared_delta_states = {}

        def get_shared_delta_state(requested_r, paired_seed):
            key = (requested_r, paired_seed)
            if key not in shared_delta_states:
                set_seed(
                    stable_seed(
                        "paired-delta",
                        int(sample_row["index"]),
                        requested_r,
                        paired_seed,
                    )
                )
                network = make_delta_inr()
                zero_last_linear(network)
                shared_delta_states[key] = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                }
            return copy.deepcopy(shared_delta_states[key])

        current_init_memory = {}

        def get_current_init_artifact(requested_r, paired_seed):
            key = (requested_r, paired_seed)
            if key in current_init_memory:
                return current_init_memory[key]
            payload = {
                "kind": "current-only-complex-init",
                "sample_index": int(sample_row["index"]),
                "requested_r": requested_r,
                "paired_seed": paired_seed,
                "iters": CURRENT_INIT_ITERS,
                "width": CURRENT_WIDTH,
                "layers": DELTA_LAYERS,
                "mask_hash": tensor_hash(masks[requested_r]),
            }
            path = cache_path("current_init", payload)
            if RESUME_CACHE and path.exists():
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                current_init_memory[key] = artifact
                return artifact

            set_seed(
                stable_seed(
                    "current-only-init",
                    int(sample_row["index"]),
                    requested_r,
                    paired_seed,
                )
            )
            network = make_current_complex_inr().to(DEVICE)
            coords = make_coord_grid(*stored_shape, device=DEVICE)
            target = measurements[requested_r]["zero_filled"].to(DEVICE)
            target_channels = torch.stack([target.real, target.imag], dim=-1).reshape(-1, 2)
            optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
            history = []
            started = time.perf_counter()
            for iteration in range(CURRENT_INIT_ITERS):
                optimizer.zero_grad(set_to_none=True)
                prediction = network(coords)
                loss = F.mse_loss(prediction, target_channels)
                loss.backward()
                optimizer.step()
                history.append(float(loss.detach()))
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            artifact = {
                "state": {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                },
                "history": history,
                "runtime_seconds": time.perf_counter() - started,
            }
            torch.save(artifact, path)
            current_init_memory[key] = artifact
            del network
            cleanup_cuda()
            return artifact

        def build_prior_model(spec, requested_r, paired_seed):
            if spec.prior_mode == "direct":
                prior_field = FixedImageField(prior)
                base_prior = direct_prior
            elif spec.prior_mode == "fitted":
                prior_field = make_prior_inr()
                prior_field.load_state_dict(prior_artifact["state"])
                base_prior = fitted_prior
            else:
                raise ValueError(f"Not a prior model: {spec.prior_mode}")

            alpha = (
                1.0
                if spec.scale_mode == "unit"
                else prior_scales[spec.prior_mode][requested_r]
            )
            delta = make_delta_inr()
            delta.load_state_dict(get_shared_delta_state(requested_r, paired_seed))
            phase = make_phase_inr()
            phase.load_state_dict(get_phase_artifact(requested_r)["state"])
            model = PriorMagnitudePhaseINR(
                prior_field,
                delta,
                phase,
                prior_scale=alpha,
                learn_prior_scale=False,
            )
            return model, base_prior, alpha

        # Static parameter audit before expensive fitting.
        audit_delta = make_delta_inr()
        audit_phase = make_phase_inr()
        audit_current = make_current_complex_inr()
        parameter_table = pd.DataFrame(
            [
                {
                    "model": "ours follow-up branches",
                    "trainable_parameters": count_parameters(audit_delta)
                    + count_parameters(audit_phase),
                    "frozen_prior_parameters": count_parameters(make_prior_inr()),
                    "fixed_prior_buffer_values": 0,
                },
                {
                    "model": "direct DICOM follow-up branches",
                    "trainable_parameters": count_parameters(audit_delta)
                    + count_parameters(audit_phase),
                    "frozen_prior_parameters": 0,
                    "fixed_prior_buffer_values": prior.numel(),
                },
                {
                    "model": "current-only complex INR",
                    "trainable_parameters": count_parameters(audit_current),
                    "frozen_prior_parameters": 0,
                    "fixed_prior_buffer_values": 0,
                },
            ]
        )
        display(parameter_table)
        del audit_delta, audit_phase, audit_current
        """
    ),
    code(
        r"""
        def run_prior_trial(spec, requested_r, paired_seed):
            alpha = (
                1.0
                if spec.scale_mode == "unit"
                else prior_scales[spec.prior_mode][requested_r]
            )
            payload = {
                "kind": "prior-trial",
                "sample_index": int(sample_row["index"]),
                "variant": asdict(spec),
                "requested_r": requested_r,
                "paired_seed": paired_seed,
                "alpha": alpha,
                "prior_hash": tensor_hash(
                    prior if spec.prior_mode == "direct" else fitted_prior
                ),
                "mask_hash": tensor_hash(masks[requested_r]),
                "phase_hash": tensor_hash(
                    get_phase_artifact(requested_r)["fitted_phase"]
                ),
                "iters": JOINT_ITERS,
                "schedule_horizon": SCHEDULE_HORIZON,
                "delta_lr": DELTA_LR,
                "phase_lr": PHASE_LR,
                "lambda_change": LAMBDA_CHANGE,
                "lambda_phase_tv": LAMBDA_PHASE_TV,
                "zero_delta": True,
            }
            path = cache_path("trial", payload)
            if RESUME_CACHE and path.exists():
                print(
                    f"loaded {spec.name:24s} R={requested_r:g} seed={paired_seed}"
                )
                return torch.load(path, map_location="cpu", weights_only=False)

            item = measurements[requested_r]
            operator = item["operator"].to(DEVICE)
            kspace = item["kspace"].to(DEVICE)
            coords = make_coord_grid(*stored_shape, device=DEVICE)
            model, _, alpha = build_prior_model(
                spec, requested_r, paired_seed
            )
            model = model.to(DEVICE)
            model.freeze_prior()
            with torch.no_grad():
                prior_values = model.prior_magnitude(coords)
                _, initial_delta, initial_magnitude, initial_phase = model.components(
                    coords, prior_mag=prior_values
                )
            if float(initial_delta.abs().max()) > 1e-7:
                raise RuntimeError("Magnitude residual is not zero at initialization.")

            optimizer = torch.optim.Adam(
                [
                    {"params": model.magnitude_residual_inr.parameters(), "lr": DELTA_LR},
                    {"params": model.phase_inr.parameters(), "lr": PHASE_LR},
                ]
            )
            scheduler = cosine_scheduler(optimizer)

            history = []
            best_psnr = -float("inf")
            best_psnr_recon = None
            best_psnr_iteration = None

            def forward_terms():
                scaled_prior, delta, magnitude, phase = model.components(
                    coords, prior_mag=prior_values
                )
                image = torch.polar(magnitude, phase).reshape(stored_shape)
                dc = data_consistency(operator(image), kspace, mask=operator.mask)
                change_l1 = LAMBDA_CHANGE * delta.abs().mean()
                phase_tv = LAMBDA_PHASE_TV * phase_tv_2d(
                    phase.reshape(stored_shape)
                )
                regularization = change_l1 + phase_tv
                return (
                    image,
                    dc + regularization,
                    dc,
                    regularization,
                    scaled_prior,
                    delta,
                    magnitude,
                    phase,
                )

            started = time.perf_counter()
            for iteration in range(JOINT_ITERS):
                optimizer.zero_grad(set_to_none=True)
                terms = forward_terms()
                image, total, dc, regularization = terms[:4]
                should_evaluate = (
                    iteration == 0
                    or (iteration + 1) % EVAL_EVERY == 0
                    or iteration == JOINT_ITERS - 1
                )
                if should_evaluate:
                    metrics = full_metrics(image.detach().cpu(), requested_r)
                    history.append(
                        {
                            "iteration": iteration,
                            "total": float(total.detach()),
                            "dc": float(dc.detach()),
                            "reg": float(regularization.detach()),
                            "laps_psnr": metrics["laps_psnr"],
                            "change_cosine": metrics["change_cosine"],
                            "change_gain": metrics["change_gain"],
                        }
                    )
                    if metrics["laps_psnr"] > best_psnr:
                        best_psnr = metrics["laps_psnr"]
                        best_psnr_recon = image.detach().cpu().clone()
                        best_psnr_iteration = iteration

                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
                )
                optimizer.step()
                scheduler.step()

            with torch.no_grad():
                terms = forward_terms()
                final_recon = terms[0].detach().cpu()
                final_scaled_prior = terms[4].reshape(stored_shape).detach().cpu()
                final_delta = terms[5].reshape(stored_shape).detach().cpu()
                final_magnitude = terms[6].reshape(stored_shape).detach().cpu()
                final_phase = terms[7].reshape(stored_shape).detach().cpu()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            joint_seconds = time.perf_counter() - started

            phase_seconds = get_phase_artifact(requested_r)["runtime_seconds"]
            prior_seconds = (
                prior_artifact["runtime_seconds"]
                if spec.prior_mode == "fitted"
                else 0.0
            )
            result = {
                "variant": asdict(spec),
                "requested_r": requested_r,
                "paired_seed": paired_seed,
                "alpha": alpha,
                "trainable_parameters": count_parameters(model, trainable_only=True),
                "total_parameters": count_parameters(model),
                "prior_buffer_values": prior.numel()
                if spec.prior_mode == "direct"
                else 0,
                "prior_preparation_seconds": prior_seconds,
                "phase_init_seconds": phase_seconds,
                "current_init_seconds": 0.0,
                "joint_runtime_seconds": joint_seconds,
                "online_seconds": phase_seconds + joint_seconds,
                "cold_seconds": prior_seconds + phase_seconds + joint_seconds,
                "history": history,
                "best_psnr_iteration": best_psnr_iteration,
                "recons": {
                    "final": final_recon,
                    "best_psnr": best_psnr_recon,
                },
                "metrics": {
                    "final": full_metrics(final_recon, requested_r),
                    "best_psnr": full_metrics(best_psnr_recon, requested_r),
                },
                "components": {
                    "scaled_prior": final_scaled_prior,
                    "delta": final_delta,
                    "magnitude": final_magnitude,
                    "phase": final_phase,
                    "initial_magnitude": initial_magnitude.reshape(stored_shape).cpu(),
                    "initial_phase": initial_phase.reshape(stored_shape).cpu(),
                },
            }
            torch.save(result, path)
            print(
                f"finished {spec.name:24s} R={requested_r:g} seed={paired_seed} "
                f"PSNR={result['metrics']['final']['laps_psnr']:.2f} dB"
            )
            del model
            cleanup_cuda()
            return result

        def run_current_only_trial(spec, requested_r, paired_seed):
            init = get_current_init_artifact(requested_r, paired_seed)
            payload = {
                "kind": "current-only-trial",
                "sample_index": int(sample_row["index"]),
                "variant": asdict(spec),
                "requested_r": requested_r,
                "paired_seed": paired_seed,
                "init_iters": CURRENT_INIT_ITERS,
                "iters": JOINT_ITERS,
                "width": CURRENT_WIDTH,
                "layers": DELTA_LAYERS,
                "lr": CURRENT_LR,
                "mask_hash": tensor_hash(masks[requested_r]),
            }
            path = cache_path("trial", payload)
            if RESUME_CACHE and path.exists():
                print(
                    f"loaded {spec.name:24s} R={requested_r:g} seed={paired_seed}"
                )
                return torch.load(path, map_location="cpu", weights_only=False)

            item = measurements[requested_r]
            operator = item["operator"].to(DEVICE)
            kspace = item["kspace"].to(DEVICE)
            coords = make_coord_grid(*stored_shape, device=DEVICE)
            model = make_current_complex_inr().to(DEVICE)
            model.load_state_dict(init["state"])
            optimizer = torch.optim.Adam(model.parameters(), lr=CURRENT_LR)
            scheduler = cosine_scheduler(optimizer)

            history = []
            best_psnr = -float("inf")
            best_psnr_recon = None
            best_psnr_iteration = None

            def forward_terms():
                channels = model(coords)
                image = torch.complex(channels[..., 0], channels[..., 1]).reshape(
                    stored_shape
                )
                dc = data_consistency(operator(image), kspace, mask=operator.mask)
                return image, dc

            started = time.perf_counter()
            for iteration in range(JOINT_ITERS):
                optimizer.zero_grad(set_to_none=True)
                image, dc = forward_terms()
                should_evaluate = (
                    iteration == 0
                    or (iteration + 1) % EVAL_EVERY == 0
                    or iteration == JOINT_ITERS - 1
                )
                if should_evaluate:
                    metrics = full_metrics(image.detach().cpu(), requested_r)
                    history.append(
                        {
                            "iteration": iteration,
                            "total": float(dc.detach()),
                            "dc": float(dc.detach()),
                            "reg": 0.0,
                            "laps_psnr": metrics["laps_psnr"],
                            "change_cosine": metrics["change_cosine"],
                            "change_gain": metrics["change_gain"],
                        }
                    )
                    if metrics["laps_psnr"] > best_psnr:
                        best_psnr = metrics["laps_psnr"]
                        best_psnr_recon = image.detach().cpu().clone()
                        best_psnr_iteration = iteration
                dc.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()

            with torch.no_grad():
                final_recon, _ = forward_terms()
                final_recon = final_recon.detach().cpu()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            joint_seconds = time.perf_counter() - started
            result = {
                "variant": asdict(spec),
                "requested_r": requested_r,
                "paired_seed": paired_seed,
                "alpha": float("nan"),
                "trainable_parameters": count_parameters(model, trainable_only=True),
                "total_parameters": count_parameters(model),
                "prior_buffer_values": 0,
                "prior_preparation_seconds": 0.0,
                "phase_init_seconds": 0.0,
                "current_init_seconds": init["runtime_seconds"],
                "joint_runtime_seconds": joint_seconds,
                "online_seconds": init["runtime_seconds"] + joint_seconds,
                "cold_seconds": init["runtime_seconds"] + joint_seconds,
                "history": history,
                "best_psnr_iteration": best_psnr_iteration,
                "recons": {
                    "final": final_recon,
                    "best_psnr": best_psnr_recon,
                },
                "metrics": {
                    "final": full_metrics(final_recon, requested_r),
                    "best_psnr": full_metrics(best_psnr_recon, requested_r),
                },
                "components": {
                    "scaled_prior": None,
                    "delta": None,
                    "magnitude": final_recon.abs(),
                    "phase": torch.angle(final_recon),
                    "initial_magnitude": None,
                    "initial_phase": None,
                },
            }
            torch.save(result, path)
            print(
                f"finished {spec.name:24s} R={requested_r:g} seed={paired_seed} "
                f"PSNR={result['metrics']['final']['laps_psnr']:.2f} dB"
            )
            del model
            cleanup_cuda()
            return result

        def run_trial(spec, requested_r, paired_seed):
            if spec.prior_mode == "none":
                return run_current_only_trial(spec, requested_r, paired_seed)
            return run_prior_trial(spec, requested_r, paired_seed)
        """
    ),
    markdown(
        r"""
        ## Run the paired direct-versus-fitted ablation

        Final fixed-stop checkpoints are primary.  Oracle-best checkpoints are
        saved only to diagnose late optimization damage and are not used to
        select a different stopping time per method.
        """
    ),
    code(
        r"""
        trial_results = {}
        result_rows = []
        for paired_seed in PAIRED_SEEDS:
            for requested_r in PRIMARY_RS:
                for spec in VARIANTS:
                    result = run_trial(spec, requested_r, paired_seed)
                    trial_results[(spec.name, requested_r, paired_seed)] = result
                    for checkpoint in ("final", "best_psnr"):
                        metrics = result["metrics"][checkpoint]
                        result_rows.append(
                            {
                                "variant": spec.name,
                                "prior_mode": spec.prior_mode,
                                "scale_mode": spec.scale_mode,
                                "checkpoint": checkpoint,
                                "requested_r": requested_r,
                                "effective_r": measurements[requested_r][
                                    "info"
                                ].effective_acceleration,
                                "paired_seed": paired_seed,
                                "alpha": result["alpha"],
                                "best_psnr_iteration": result["best_psnr_iteration"],
                                "trainable_parameters": result["trainable_parameters"],
                                "total_parameters": result["total_parameters"],
                                "prior_buffer_values": result["prior_buffer_values"],
                                "prior_preparation_seconds": result[
                                    "prior_preparation_seconds"
                                ],
                                "phase_init_seconds": result["phase_init_seconds"],
                                "current_init_seconds": result["current_init_seconds"],
                                "joint_runtime_seconds": result[
                                    "joint_runtime_seconds"
                                ],
                                "online_seconds": result["online_seconds"],
                                "cold_seconds": result["cold_seconds"],
                                **metrics,
                            }
                        )

        results_table = pd.DataFrame(result_rows)
        results_table.to_csv(OUTPUT_DIR / "paired_ablation_all_checkpoints.csv", index=False)
        final_table = results_table[results_table["checkpoint"] == "final"].copy()
        final_table.to_csv(OUTPUT_DIR / "paired_ablation_final.csv", index=False)

        display(
            final_table[
                [
                    "variant",
                    "requested_r",
                    "paired_seed",
                    "alpha",
                    "laps_psnr",
                    "laps_ssim",
                    "change_cosine",
                    "change_gain",
                    "mi_prior_delta",
                    "change_roi_mae",
                    "data_error",
                    "joint_runtime_seconds",
                ]
            ].round(4)
        )
        """
    ),
    markdown(
        r"""
        ## Aggregate results and paired effects

        The most important comparisons are paired within the same `(R, seed)`:

        - `direct_unit - fitted_unit`: effect of removing prior fitting alone;
        - `direct_calibrated - fitted_calibrated`: same effect after proper scale;
        - calibrated minus unit within a representation: effect of scale.
        """
    ),
    code(
        r"""
        summary_table = (
            final_table.groupby(
                ["variant", "requested_r", "effective_r"], as_index=False
            )
            .agg(
                psnr_mean=("laps_psnr", "mean"),
                psnr_std=("laps_psnr", "std"),
                ssim_mean=("laps_ssim", "mean"),
                ssim_std=("laps_ssim", "std"),
                change_cosine_mean=("change_cosine", "mean"),
                change_gain_mean=("change_gain", "mean"),
                mi_delta_mean=("mi_prior_delta", "mean"),
                change_roi_mae_mean=("change_roi_mae", "mean"),
                magnitude_rmse_mean=("magnitude_rmse", "mean"),
                data_error_mean=("data_error", "mean"),
                joint_seconds_mean=("joint_runtime_seconds", "mean"),
                online_seconds_mean=("online_seconds", "mean"),
                cold_seconds_mean=("cold_seconds", "mean"),
            )
            .sort_values(["requested_r", "psnr_mean"], ascending=[True, False])
        )
        summary_table.to_csv(OUTPUT_DIR / "paired_ablation_summary.csv", index=False)
        display(summary_table.round(4))

        metric_columns = [
            "laps_psnr",
            "laps_ssim",
            "change_cosine",
            "change_gain",
            "change_roi_mae",
            "magnitude_rmse",
            "data_error",
        ]
        pivot = final_table.pivot(
            index=["requested_r", "effective_r", "paired_seed"],
            columns="variant",
            values=metric_columns,
        )

        effect_rows = []
        comparisons = {
            "direct_minus_fitted_unit": ("direct_unit", "fitted_unit"),
            "direct_minus_fitted_calibrated": (
                "direct_calibrated",
                "fitted_calibrated",
            ),
            "scale_effect_direct": ("direct_calibrated", "direct_unit"),
            "scale_effect_fitted": ("fitted_calibrated", "fitted_unit"),
            "direct_calibrated_minus_current": (
                "direct_calibrated",
                "current_only_complex",
            ),
        }
        for index in pivot.index:
            requested_r, effective_r, paired_seed = index
            for label, (left, right) in comparisons.items():
                row = {
                    "comparison": label,
                    "requested_r": requested_r,
                    "effective_r": effective_r,
                    "paired_seed": paired_seed,
                }
                for metric in metric_columns:
                    row[f"delta_{metric}"] = (
                        pivot.loc[index, (metric, left)]
                        - pivot.loc[index, (metric, right)]
                    )
                effect_rows.append(row)

        paired_effects = pd.DataFrame(effect_rows)
        paired_effects.to_csv(OUTPUT_DIR / "paired_effects.csv", index=False)
        paired_effect_summary = (
            paired_effects.groupby(
                ["comparison", "requested_r", "effective_r"], as_index=False
            )
            .agg(
                delta_psnr_mean=("delta_laps_psnr", "mean"),
                delta_psnr_std=("delta_laps_psnr", "std"),
                delta_ssim_mean=("delta_laps_ssim", "mean"),
                delta_change_cosine_mean=("delta_change_cosine", "mean"),
                delta_change_gain_mean=("delta_change_gain", "mean"),
                delta_change_roi_mae_mean=("delta_change_roi_mae", "mean"),
                delta_data_error_mean=("delta_data_error", "mean"),
            )
        )
        display(paired_effect_summary.round(5))

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
        for variant in [spec.name for spec in VARIANTS]:
            part = summary_table[summary_table["variant"] == variant].sort_values(
                "effective_r"
            )
            axes[0].errorbar(
                part["effective_r"],
                part["psnr_mean"],
                yerr=part["psnr_std"].fillna(0),
                marker="o",
                capsize=3,
                label=variant,
            )
            axes[1].plot(
                part["effective_r"],
                part["change_gain_mean"],
                marker="o",
                label=variant,
            )
            axes[2].plot(
                part["effective_r"],
                part["mi_delta_mean"],
                marker="o",
                label=variant,
            )
        axes[0].set_ylabel("LAPS PSNR (dB)")
        axes[1].set_ylabel("acquisition-calibrated change gain")
        axes[2].set_ylabel("MI prior bias (bits)")
        for axis in axes:
            axis.set_xlabel("effective acceleration")
            axis.grid(alpha=0.25)
        axes[2].legend(frameon=False, fontsize=7, loc="best")
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Convergence and qualitative diagnostics

        Qualitative panels use seed 0 by declaration, not the best seed.  Error
        and signed-change panels are in acquisition units without per-method
        reference alignment.  The `actual delta branch` panel is the network's
        real magnitude residual, not reconstruction-minus-prior.
        """
    ),
    code(
        r"""
        qualitative_seed = PAIRED_SEEDS[0]
        qualitative_variants = [
            "fitted_unit",
            "direct_unit",
            "fitted_calibrated",
            "direct_calibrated",
            "current_only_complex",
        ]

        for requested_r in PRIMARY_RS:
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
            for variant in qualitative_variants:
                result = trial_results[(variant, requested_r, qualitative_seed)]
                history = pd.DataFrame(result["history"])
                axes[0].plot(history["iteration"], history["total"], label=variant)
                axes[1].plot(
                    history["iteration"], history["laps_psnr"], label=variant
                )
                axes[2].plot(
                    history["iteration"], history["change_gain"], label=variant
                )
            axes[0].set_yscale("log")
            axes[0].set_ylabel("training objective")
            axes[1].set_ylabel("LAPS PSNR (diagnostic)")
            axes[2].set_ylabel("calibrated change gain")
            for axis in axes:
                axis.set_xlabel("iteration")
                axis.grid(alpha=0.25)
            axes[2].legend(frameon=False, fontsize=7)
            fig.suptitle(f"Convergence at requested R={requested_r}", y=1.02)
            fig.tight_layout()
            plt.show()

            reference_acq = reference.abs() * reference_scales[requested_r]
            prior_acq = prior * prior_scales["direct"][requested_r]
            true_change = reference_acq - prior_acq
            methods = {
                "reference": reference_acq,
                "CG-SENSE": baseline_recons[requested_r]["CG-SENSE"].abs(),
            }
            for variant in qualitative_variants:
                methods[variant] = trial_results[
                    (variant, requested_r, qualitative_seed)
                ]["recons"]["final"].abs()

            errors = {
                name: (image.cpu() * support - reference_acq.cpu()).abs()
                for name, image in methods.items()
                if name != "reference"
            }
            error_vmax = float(
                torch.quantile(
                    torch.cat([value.reshape(-1) for value in errors.values()]),
                    0.99,
                )
            )
            display_vmax = float(torch.quantile(reference_acq.reshape(-1), 0.995))
            fig, axes = plt.subplots(
                2, len(methods), figsize=(3.0 * len(methods), 6), squeeze=False
            )
            for column, (name, image) in enumerate(methods.items()):
                shown = image.cpu() * support
                axes[0, column].imshow(
                    to_numpy(shown), cmap="gray", vmin=0, vmax=display_vmax
                )
                if name == "reference":
                    axes[0, column].set_title(name)
                    axes[1, column].imshow(
                        np.zeros(stored_shape), cmap="magma", vmin=0, vmax=error_vmax
                    )
                else:
                    metric = full_metrics(
                        trial_results[(name, requested_r, qualitative_seed)][
                            "recons"
                        ]["final"]
                        if name in qualitative_variants
                        else baseline_recons[requested_r][name],
                        requested_r,
                    )
                    axes[0, column].set_title(
                        f"{name}\nPSNR {metric['laps_psnr']:.2f}, "
                        f"gain {metric['change_gain']:.2f}",
                        fontsize=8,
                    )
                    axes[1, column].imshow(
                        to_numpy(errors[name]),
                        cmap="magma",
                        vmin=0,
                        vmax=error_vmax,
                    )
                axes[1, column].set_title("raw magnitude error", fontsize=9)
            for axis in axes.flat:
                axis.axis("off")
            fig.suptitle(
                f"Acquisition-unit comparison: requested R={requested_r}", y=1.02
            )
            fig.tight_layout()
            plt.show()

            diagnostic_variants = [
                "fitted_calibrated",
                "direct_calibrated",
            ]
            change_limit = float(torch.quantile(true_change.abs(), 0.995))
            fig, axes = plt.subplots(
                len(diagnostic_variants), 5, figsize=(15, 6), squeeze=False
            )
            for row, variant in enumerate(diagnostic_variants):
                result = trial_results[(variant, requested_r, qualitative_seed)]
                reconstruction_change = (
                    result["recons"]["final"].abs() * support - prior_acq
                )
                change_error = reconstruction_change - true_change
                panels = [
                    (true_change, "true calibrated change"),
                    (reconstruction_change, "reconstructed change"),
                    (change_error, "change error"),
                    (result["components"]["delta"], "actual delta branch"),
                    (result["components"]["phase"], "reconstructed phase"),
                ]
                for column, (image, title) in enumerate(panels):
                    if column == 4:
                        axes[row, column].imshow(
                            to_numpy(image),
                            cmap="twilight",
                            vmin=-math.pi,
                            vmax=math.pi,
                        )
                    else:
                        axes[row, column].imshow(
                            to_numpy(image),
                            cmap="coolwarm",
                            vmin=-change_limit,
                            vmax=change_limit,
                        )
                    axes[row, column].set_title(f"{variant}\n{title}", fontsize=8)
                    axes[row, column].axis("off")
            fig.tight_layout()
            plt.show()
        """
    ),
    markdown(
        r"""
        ## Decision summary

        Interpret the paired differences as follows:

        - direct better than fitted: prior-INR fitting removed useful detail;
        - nearly identical: remove the prior fit because it adds cost without gain;
        - fitted better: the prior INR supplied useful smoothing/spectral bias;
        - calibration helps both: scale mismatch, not prior representation, was
          the major issue;
        - current-only matches direct calibrated: the longitudinal prior is not
          adding measurable value under this setup;
        - all prior methods fail together at high R: change the way prior trust
          is expressed rather than increasing network width.
        """
    ),
    code(
        r"""
        decision_rows = []
        for requested_r in PRIMARY_RS:
            effects_r = paired_effects[paired_effects["requested_r"] == requested_r]
            for comparison in comparisons:
                part = effects_r[effects_r["comparison"] == comparison]
                decision_rows.append(
                    {
                        "requested_r": requested_r,
                        "comparison": comparison,
                        "mean_delta_psnr_db": part["delta_laps_psnr"].mean(),
                        "std_delta_psnr_db": part["delta_laps_psnr"].std(),
                        "mean_delta_change_gain": part["delta_change_gain"].mean(),
                        "mean_delta_change_roi_mae": part[
                            "delta_change_roi_mae"
                        ].mean(),
                    }
                )
        decision_table = pd.DataFrame(decision_rows)
        display(decision_table.round(5))

        direct_fit_effect = decision_table[
            decision_table["comparison"] == "direct_minus_fitted_calibrated"
        ]
        scale_effect = decision_table[
            decision_table["comparison"] == "scale_effect_direct"
        ]
        prior_value = decision_table[
            decision_table["comparison"] == "direct_calibrated_minus_current"
        ]

        print(
            "Mean direct-vs-fitted calibrated PSNR effect:",
            f"{direct_fit_effect['mean_delta_psnr_db'].mean():+.3f} dB",
        )
        print(
            "Mean calibration PSNR effect for direct prior:",
            f"{scale_effect['mean_delta_psnr_db'].mean():+.3f} dB",
        )
        print(
            "Mean direct-prior value over current-only:",
            f"{prior_value['mean_delta_psnr_db'].mean():+.3f} dB",
        )

        configuration = {
            "sample_index": int(sample_row["index"]),
            "scan_index": TARGET_SCAN_INDEX,
            "slice_index": TARGET_SLICE_INDEX,
            "primary_accelerations": list(PRIMARY_RS),
            "paired_seeds": list(PAIRED_SEEDS),
            "joint_iterations": JOINT_ITERS,
            "delta_width": DELTA_WIDTH,
            "delta_layers": DELTA_LAYERS,
            "delta_lr": DELTA_LR,
            "phase_lr": PHASE_LR,
            "lambda_change": LAMBDA_CHANGE,
            "lambda_phase_tv": LAMBDA_PHASE_TV,
            "warning": "Development-only: exclude all of scan 16 from held-out reporting.",
        }
        (OUTPUT_DIR / "configuration.json").write_text(
            json.dumps(configuration, indent=2) + "\n"
        )
        """
    ),
    markdown(
        r"""
        ## Optional full acceleration sweep

        Leave this off until the paired experiment identifies a formulation.
        Set `RUN_FULL_SWEEP=True` and choose `FULL_SWEEP_VARIANT` in the
        configuration cell.  One declared seed is then used at every R; no
        per-acceleration model selection occurs.
        """
    ),
    code(
        r"""
        full_sweep_rows = []
        if RUN_FULL_SWEEP:
            selected_spec = next(
                spec for spec in VARIANTS if spec.name == FULL_SWEEP_VARIANT
            )
            if selected_spec.prior_mode == "none":
                raise ValueError("The optional sweep is intended for a prior formulation.")

            # Phase/scales for R values not in PRIMARY_RS are created only here.
            for requested_r in MASK_ACCELERATIONS:
                get_phase_artifact(requested_r)
                if requested_r not in reference_scales:
                    item = measurements[requested_r]
                    operator = item["operator"].to(DEVICE)
                    kspace = item["kspace"].to(DEVICE)
                    phase0 = get_phase_artifact(requested_r)["fitted_phase"].to(DEVICE)
                    prior_scales["direct"][requested_r] = float(
                        prior_scale_from_kspace(
                            prior.to(DEVICE), phase0, operator, kspace
                        ).detach()
                    )
                    prior_scales["fitted"][requested_r] = float(
                        prior_scale_from_kspace(
                            fitted_prior.clamp_min(0).to(DEVICE),
                            phase0,
                            operator,
                            kspace,
                        ).detach()
                    )
                    reference_scales[requested_r] = float(
                        real_least_squares_scale(
                            operator(reference.to(DEVICE)),
                            kspace,
                            weights=operator.mask,
                        ).detach()
                    )
                    del operator, kspace, phase0
                    cleanup_cuda()

                result = run_trial(
                    selected_spec, requested_r, FULL_SWEEP_SEED
                )
                full_sweep_rows.append(
                    {
                        "variant": selected_spec.name,
                        "requested_r": requested_r,
                        "effective_r": measurements[requested_r][
                            "info"
                        ].effective_acceleration,
                        "paired_seed": FULL_SWEEP_SEED,
                        **result["metrics"]["final"],
                    }
                )
            full_sweep_table = pd.DataFrame(full_sweep_rows)
            full_sweep_table.to_csv(
                OUTPUT_DIR / "selected_full_acceleration_sweep.csv", index=False
            )
            display(full_sweep_table.round(4))
        else:
            print("RUN_FULL_SWEEP=False; no formulation was selected automatically.")
        """
    ),
    markdown(
        r"""
        # Stronger ways to use the prior after this ablation

        Direct addition is only the simplest use of the registered DICOM.  The
        most promising next formulations are:

        1. **Residual-measurement reconstruction.** Form
           `y_delta = y - A(alpha * P * exp(i*phi0))` and reconstruct the
           innovation explicitly.  This makes prior disagreement visible in
           both image and k-space domains.

        2. **Smooth intensity harmonization plus localized change.** Use
           \(\hat m=e^{b_\gamma(c)}P(c)+\Delta m_\psi(c)\), where `b` is a
           very low-bandwidth bias field.  Global/protocol contrast no longer
           consumes the change residual.

        3. **Soft, measurement-driven prior trust.** Reconstruct a current-only
           image and use its disagreement with the prior to build a bounded
           confidence map.  The prior becomes a robust regularizer rather than
           a mandatory copy, reducing high-R prior leakage.

        4. **Registration-aware prior.** Use
           \(P(c+u_\eta(c))\) with a small, smooth deformation before adding the
           residual.  This directly targets the double-edge errors.  The
           deformation must be tightly constrained so it cannot warp away real
           pathology.  Continuous off-grid querying is the clearest reason to
           retain a prior INR.

        5. **Use prior structure, not prior intensity.** Condition the residual
           on `P`, its gradients, and coarse multiscale features, or add a robust
           edge-consistency loss.  This is less sensitive to DICOM intensity
           mismatch.

        6. **Data-consistency projection.** After the INR returns `z`, solve a
           short quadratic update
           \(\min_x\|Ax-y\|^2+\mu\|x-z\|^2\).  This preserves prior-guided
           detail while enforcing measured k-space more strongly.

        7. **Joint multi-slice/3-D residual.** Use the complete registered DICOM
           volume and share a residual field across slices while retaining each
           slice's 2-D acquisition operator.  Neighboring slices add evidence
           that the current one-slice experiment cannot exploit.

        Recommended sequence: direct/calibration ablation -> residual-measurement
        formulation -> smooth harmonization -> data-consistency projection ->
        measurement-driven trust or tightly constrained deformation.
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python (presinr-notebook)",
            "language": "python",
            "name": "presinr-notebook",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(OUTPUT)
