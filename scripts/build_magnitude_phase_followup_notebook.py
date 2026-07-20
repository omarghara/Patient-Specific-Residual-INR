"""Build the scale-calibrated magnitude-phase formulation follow-up notebook.

This creates a new notebook and report namespace.  It intentionally does not
modify the executed single-sample acceleration-tuning notebook or its outputs.
"""

import json
import textwrap
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks" / "magnitude_phase_formulation_followup.ipynb"
REPORT_DIR = REPO / "reports" / "magnitude_phase_followup"


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
        # Magnitude-phase INR formulation follow-up

        This notebook implements the improvements motivated by the executed
        single-slice tuning study, while preserving that study as an immutable
        historical baseline.

        The central questions are:

        1. Does acquisition-derived scaling prevent the residual branch from
           spending capacity on a global intensity mismatch?
        2. Which residual width and sparsity weight survive reference-free
           validation on withheld acquired k-space lines?
        3. Is zero-filled or CG-SENSE phase initialization better when every
           other setting is held fixed?
        4. Do fixed Fourier features extend the useful acceleration range?
        5. Does the longitudinal prior improve on a parameter-matched,
           current-only magnitude-phase INR?

        This remains a **development experiment on scan 16, slice 23**. It
        freezes one configuration and stopping iteration, but it does not make
        held-out cohort claims. Scan 16 must be excluded from later testing.
        """
    ),
    markdown(
        r"""
        ## Selection protocol

        Hyperparameters and stopping are selected only by normalized error on
        acquired phase-encoding lines withheld from optimization. Reference
        PSNR, SSIM, change metrics, and images are diagnostic and never enter
        selection.

        After selection, each reconstruction is restarted and trained on every
        acquired line for one globally fixed number of iterations. The final
        acceleration sweep never chooses a different checkpoint at each R.

        The older executed notebook and `reports/magnitude_phase_tuning/` are
        read-only inputs. New artifacts are written under
        `reports/magnitude_phase_followup/`.
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
        from dataclasses import asdict, dataclass, replace
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        import torch.nn as nn
        from IPython.display import display
        from skimage.metrics import structural_similarity

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
        from presinr.experiments.magnitude_phase import (
            MagnitudePhaseTrainConfig,
            build_scalar_inr,
            relative_kspace_error,
            train_magnitude_phase,
            zero_last_linear_,
        )
        from presinr.metrics import (
            acquisition_calibrated_longitudinal_metrics,
            mutual_information,
        )
        from presinr.models import CurrentMagnitudePhaseINR, PriorMagnitudePhaseINR
        from presinr.models.inr import make_coord_grid
        from presinr.recon import PhaseFitConfig, PriorFitConfig, fit_phase_inr, fit_prior
        from presinr.sampling import cartesian_line_holdout, laps_retrospective_1d_mask
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

        Set `PRESINR_FOLLOWUP_SMOKE=1` before launching the kernel for a
        two-iteration, one-acceleration software check. The default is the full
        staged development experiment and can take substantial GPU time.
        """
    ),
    code(
        r"""
        SMOKE = os.environ.get("PRESINR_FOLLOWUP_SMOKE", "0") == "1"
        BASE_SEED = 42
        CACHE_VERSION = "magnitude-phase-followup-v2"

        TARGET_SCAN_INDEX = 16
        TARGET_SLICE_INDEX = 23
        TARGET_CHANGE_EXTENT = 2
        PHASE_ENCODE_DIM = 1
        ACCELERATIONS = (3, 5, 6, 7, 9, 11, 13)
        TUNE_RS = (6, 9)
        FOURIER_RS = (6, 9, 13)
        QUALITATIVE_RS = (3, 6, 9, 13)

        PRIOR_ITERS = 3000
        PHASE_ITERS = 1000
        TUNE_ITERS = 1800
        MAX_FINAL_ITERS = 3000
        EVAL_EVERY = 100
        VALIDATION_FRACTION = 0.10

        WIDTHS = (64, 128)
        CHANGE_LAMBDAS = (0.0, 1e-4, 3e-4, 1e-3)
        RUN_FOURIER = True
        RUN_CURRENT_ONLY = True
        RUN_CG_BASELINE = True
        RESUME_CACHE = True

        SELECTED_SCALE_NAME = None
        SELECTED_WIDTH = None
        SELECTED_LAMBDA = None
        SELECTED_PHASE_SOURCE = None
        SELECTED_ENCODING_NAME = None
        SELECTED_ITERATIONS = None

        if SMOKE:
            ACCELERATIONS = (6,)
            TUNE_RS = (6,)
            FOURIER_RS = (6,)
            QUALITATIVE_RS = (6,)
            PRIOR_ITERS = 2
            PHASE_ITERS = 2
            TUNE_ITERS = 2
            MAX_FINAL_ITERS = 2
            EVAL_EVERY = 1
            WIDTHS = (128,)
            CHANGE_LAMBDAS = (3e-4,)

        OUTPUT_DIR = (
            Path("/tmp/presinr-magnitude-phase-followup-smoke")
            if SMOKE
            else REPO / "reports" / "magnitude_phase_followup"
        )
        CACHE_DIR = OUTPUT_DIR / "cache"
        LEGACY_DIR = REPO / "reports" / "magnitude_phase_tuning"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        print("smoke mode       :", SMOKE)
        print("accelerations    :", ACCELERATIONS)
        print("selection R      :", TUNE_RS)
        print("Fourier R        :", FOURIER_RS)
        print("output            :", OUTPUT_DIR)
        """
    ),
    markdown(
        r"""
        ## Historical baseline (read-only)

        These tables document the result that motivated this follow-up. Old
        longitudinal metrics are not reused because they were calculated after
        a method-specific oracle magnitude alignment. Only paper-style
        PSNR/SSIM and provenance are carried forward.
        """
    ),
    code(
        r"""
        legacy_selected = None
        legacy_ours_table = pd.DataFrame()
        legacy_nerp_table = pd.DataFrame()

        selected_path = LEGACY_DIR / "selected_configuration.json"
        if selected_path.exists():
            legacy_selected = json.loads(selected_path.read_text())
            print("historical selected configuration:")
            display(pd.DataFrame([legacy_selected["selected_spec"]]))

        ours_path = LEGACY_DIR / "ours_acceleration_sweep.csv"
        if ours_path.exists():
            legacy_ours_table = pd.read_csv(ours_path)
            legacy_ours_table = legacy_ours_table[
                legacy_ours_table["checkpoint"] == "final"
            ].copy()
            display(
                legacy_ours_table[
                    ["requested_r", "effective_r", "laps_psnr", "laps_ssim"]
                ].round(4)
            )

        nerp_path = LEGACY_DIR / "laps_nerp_reference.csv"
        if nerp_path.exists():
            legacy_nerp_table = pd.read_csv(nerp_path)
            print("loaded historical LAPS-NeRP table:", nerp_path)
        """
    ),
    markdown(
        r"""
        ## Development slice

        Metadata fixes the sample before reconstruction scores are inspected:
        scan 16, zero-based slice 23, large radiologist-rated change, and an
        originally fully sampled two-dimensional acquisition.
        """
    ),
    code(
        r"""
        dataset = SlamTestSlices(
            data_dir=REPO / "data", middle_only=False, normalize=False
        )
        manifest = dataset.df.copy().reset_index(drop=True)
        manifest["dataset_position"] = np.arange(len(manifest))
        match = manifest[
            (manifest["scan_index"] == TARGET_SCAN_INDEX)
            & (manifest["slice_index"] == TARGET_SLICE_INDEX)
        ]
        if len(match) != 1:
            raise RuntimeError(f"Expected one development row, found {len(match)}")
        sample_row = match.iloc[0]
        if int(sample_row["change_extent"]) != TARGET_CHANGE_EXTENT:
            raise RuntimeError("Development case change extent has changed.")
        if int(sample_row["AccelNumDim"]) != 0:
            raise RuntimeError("Development case must be originally fully sampled.")

        sample = dataset[int(sample_row["dataset_position"])]

        def quantile_scale(tensor, q=0.999):
            values = tensor.detach().abs().reshape(-1).float()
            return float(torch.quantile(values, q)) + 1e-8

        reference = sample["recon"].to(torch.complex64)
        prior = sample["prior"].float()
        reference = reference / quantile_scale(reference)
        prior = prior / quantile_scale(prior)
        raw_kspace = sample["ksp"].to(torch.complex64)
        mps = sample["mps"].to(torch.complex64)
        stored_shape = tuple(sample["stored_shape"])

        display(
            sample_row[
                [
                    "dataset_position", "index", "scan_index", "slice_index",
                    "change_extent", "AccelNumDim", "scan_plane", "scan_type",
                ]
            ].to_frame("value")
        )
        print("stored shape:", stored_shape, "coils:", raw_kspace.shape[0])

        prior_np = np.abs(to_numpy(prior))
        reference_np = np.abs(to_numpy(reference))
        true_change_unit = reference_np - prior_np
        change_limit_unit = float(np.quantile(np.abs(true_change_unit), 0.995))

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        axes[0].imshow(prior_np, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("registered prior")
        axes[1].imshow(reference_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("follow-up reference\n(evaluation only)")
        axes[2].imshow(
            true_change_unit,
            cmap="coolwarm",
            vmin=-change_limit_unit,
            vmax=change_limit_unit,
        )
        axes[2].set_title("signed change in normalized image units")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Nested masks and reference-free line holdouts

        Full masks match the previous controlled experiment. For development
        fitting, 10% of acquired phase-encoding lines are withheld while all
        central calibration lines stay in the training split. Train and
        validation masks are disjoint and exactly reconstruct the full mask.
        """
    ),
    code(
        r"""
        def stable_seed(*parts, base=BASE_SEED):
            payload = "|".join(map(str, (base,) + parts)).encode("utf-8")
            return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)

        def tensor_hash(tensor):
            value = tensor.detach().cpu().contiguous().numpy().tobytes()
            return hashlib.sha256(value).hexdigest()[:16]

        def cache_fingerprint(payload):
            encoded = json.dumps(
                {"version": CACHE_VERSION, **payload}, sort_keys=True, default=str
            ).encode("utf-8")
            return hashlib.sha256(encoded).hexdigest()[:16]

        def cache_path(kind, payload):
            return CACHE_DIR / f"{kind}_{cache_fingerprint(payload)}.pt"

        def cleanup_cuda():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        masks = {}
        mask_info = {}
        source_mask = sample["mask"].float()
        for requested_r in sorted(ACCELERATIONS):
            current_mask, info = laps_retrospective_1d_mask(
                source_mask,
                requested_r,
                seed=stable_seed("nested-mask", int(sample_row["index"]), requested_r),
                phase_encode_dim=PHASE_ENCODE_DIM,
                vd_factor=0.8,
                n_candidates=100,
            )
            masks[requested_r] = current_mask
            mask_info[requested_r] = info
            source_mask = current_mask

        common_mask = torch.stack(
            [masks[r].bool() for r in ACCELERATIONS]
        ).all(dim=0).float()
        common_operator = CenterPaddedSense(mps, common_mask, stored_shape)
        common_scale = quantile_scale(
            common_operator.adjoint(raw_kspace * common_mask)
        )
        support_native = torch.linalg.vector_norm(mps, dim=0) > 0.5
        support = center_pad_to(support_native.float(), stored_shape)

        measurements = {}
        measurement_rows = []
        for requested_r in ACCELERATIONS:
            full_mask = masks[requested_r]
            full_operator = CenterPaddedSense(mps, full_mask, stored_shape)
            full_kspace = raw_kspace * full_mask / common_scale
            if requested_r in set(TUNE_RS) | set(FOURIER_RS):
                train_mask, validation_mask, holdout_info = cartesian_line_holdout(
                    full_mask,
                    validation_fraction=VALIDATION_FRACTION,
                    seed=stable_seed("line-holdout", int(sample_row["index"]), requested_r),
                    phase_encode_dim=PHASE_ENCODE_DIM,
                    protected_center_lines=mask_info[requested_r].center_lines,
                    min_validation_lines=1,
                )
            else:
                train_mask = full_mask.clone()
                validation_mask = torch.zeros_like(full_mask)
                holdout_info = None
            if torch.any(train_mask.bool() & validation_mask.bool()):
                raise RuntimeError("Training and validation masks overlap.")
            if not torch.equal(
                train_mask.bool() | validation_mask.bool(), full_mask.bool()
            ):
                raise RuntimeError("Training and validation do not reconstruct full mask.")

            measurements[requested_r] = {
                "full_mask": full_mask,
                "full_operator": full_operator,
                "full_kspace": full_kspace,
                "zero_filled": full_operator.adjoint(full_kspace),
                "train_mask": train_mask,
                "train_operator": CenterPaddedSense(mps, train_mask, stored_shape),
                "train_kspace": raw_kspace * train_mask / common_scale,
                "validation_mask": validation_mask,
                "validation_operator": (
                    CenterPaddedSense(mps, validation_mask, stored_shape)
                    if validation_mask.any()
                    else None
                ),
                "validation_kspace": raw_kspace * validation_mask / common_scale,
            }
            measurement_rows.append(
                {
                    "requested_r": requested_r,
                    "effective_r": mask_info[requested_r].effective_acceleration,
                    "full_lines": mask_info[requested_r].output_lines,
                    "center_lines": mask_info[requested_r].center_lines,
                    "training_lines": (
                        holdout_info.training_lines if holdout_info else mask_info[requested_r].output_lines
                    ),
                    "validation_lines": holdout_info.validation_lines if holdout_info else 0,
                }
            )

        measurement_table = pd.DataFrame(measurement_rows)
        display(measurement_table)
        print("common acquisition scale:", common_scale)

        fig, axes = plt.subplots(
            2, len(ACCELERATIONS), figsize=(3.0 * len(ACCELERATIONS), 5.8), squeeze=False
        )
        for column, requested_r in enumerate(ACCELERATIONS):
            item = measurements[requested_r]
            full = to_numpy(item["full_mask"])
            split = np.zeros((*full.shape, 3), dtype=np.float32)
            split[..., 0] = to_numpy(item["validation_mask"])
            split[..., 1] = to_numpy(item["train_mask"])
            split[..., 2] = to_numpy(item["train_mask"])
            axes[0, column].imshow(split)
            axes[0, column].set_title(
                f"R={requested_r} / {mask_info[requested_r].effective_acceleration:.2f}\n"
                "cyan=train, red=validation"
            )
            axes[1, column].imshow(
                np.abs(to_numpy(item["zero_filled"] * support)), cmap="gray"
            )
            axes[1, column].set_title("full-mask zero-filled")
        for axis in axes.flat:
            axis.axis("off")
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Fixed-unit evaluation

        Two metric families are intentionally kept separate:

        - **LAPS metrics** optimally align each reconstructed magnitude to the
          reference. They are retained for paper comparison.
        - **Acquisition-calibrated metrics** never align an individual method.
          A single scalar maps the normalized reference into k-space units, and
          a reference-free scalar maps the normalized DICOM prior into those
          units. Change amplitude, MI, MAE, and NRMSE use these fixed units.

        The reference calibration is evaluation-only and never enters training
        or model selection.
        """
    ),
    code(
        r"""
        reference_to_acquisition = {}
        for requested_r in ACCELERATIONS:
            item = measurements[requested_r]
            prediction = item["full_operator"](reference)
            reference_to_acquisition[requested_r] = float(
                real_least_squares_scale(
                    prediction,
                    item["full_kspace"],
                    weights=item["full_mask"],
                )
            )

        prior_to_acquisition = {}

        def laps_alignment(reconstruction, target_reference=reference):
            estimate = np.abs(to_numpy(reconstruction)).astype(np.float64)
            target = np.abs(to_numpy(target_reference)).astype(np.float64)
            target = target / (np.quantile(target, 0.99) + 1e-12)
            denominator = float(np.sum(estimate * estimate))
            gain = 1.0 if denominator <= 1e-20 else float(
                np.sum(estimate * target) / denominator
            )
            return gain * estimate, target, gain

        def laps_metrics(reconstruction, target_reference=reference):
            estimate, target, gain = laps_alignment(reconstruction, target_reference)
            mse = float(np.mean((estimate - target) ** 2))
            return {
                "laps_psnr": float(-10.0 * np.log10(max(mse, 1e-12))),
                "laps_ssim": float(
                    structural_similarity(target, estimate, data_range=1.0)
                ),
                "laps_gain": gain,
            }

        def global_phase_align(estimate, target, mask):
            selected_estimate = estimate[mask]
            selected_target = target[mask]
            cross = torch.vdot(selected_estimate.reshape(-1), selected_target.reshape(-1))
            if float(cross.abs()) <= 1e-12:
                return estimate, torch.ones((), dtype=estimate.dtype)
            factor = cross / cross.abs()
            return factor * estimate, factor

        def fixed_unit_metrics(raw_reconstruction, requested_r):
            if requested_r not in prior_to_acquisition:
                raise RuntimeError("Prior acquisition calibration has not been computed.")
            item = measurements[requested_r]
            reconstruction = raw_reconstruction.detach().cpu() * support
            reference_factor = reference_to_acquisition[requested_r]
            prior_factor = prior_to_acquisition[requested_r]
            target_complex = reference_factor * reference.detach().cpu() * support
            target_magnitude = target_complex.abs()
            estimate_magnitude = reconstruction.abs()
            normalized_target = to_numpy(target_magnitude / reference_factor)
            normalized_estimate = to_numpy(estimate_magnitude / reference_factor)
            magnitude_error = normalized_estimate - normalized_target
            mse = float(np.mean(magnitude_error**2))

            longitudinal = acquisition_calibrated_longitudinal_metrics(
                reconstruction,
                reference,
                prior,
                reference_to_acquisition=reference_factor,
                prior_to_acquisition=prior_factor,
            )

            foreground = (target_magnitude > 0.05 * reference_factor) & support.bool()
            phase_aligned, phase_factor = global_phase_align(
                reconstruction, target_complex, foreground
            )
            complex_nrmse = float(
                torch.linalg.vector_norm(phase_aligned - target_complex)
                / (torch.linalg.vector_norm(target_complex) + 1e-12)
            )
            phase_difference = torch.angle(phase_aligned * target_complex.conj())
            phase_weights = target_magnitude.square() * foreground
            phase_rmse = float(
                torch.sqrt(
                    (phase_difference.square() * phase_weights).sum()
                    / (phase_weights.sum() + 1e-12)
                )
            )

            prior_acq = prior_factor * prior.detach().cpu()
            true_change = target_magnitude - prior_acq
            reconstructed_change = estimate_magnitude - prior_acq
            foreground_np = to_numpy(foreground).astype(bool)
            abs_change_np = np.abs(to_numpy(true_change))
            threshold = float(np.quantile(abs_change_np[foreground_np], 0.90))
            change_roi = foreground_np & (abs_change_np >= threshold)
            roi_error = to_numpy(reconstructed_change - true_change)[change_roi]

            output = {
                **laps_metrics(reconstruction),
                "fixed_psnr": float(-10.0 * np.log10(max(mse, 1e-12))),
                "fixed_ssim": float(
                    structural_similarity(
                        normalized_target, normalized_estimate, data_range=1.0
                    )
                ),
                "fixed_mae": float(np.mean(np.abs(magnitude_error))),
                "fixed_rmse": float(np.sqrt(mse)),
                "fixed_nrmse": float(
                    np.linalg.norm(magnitude_error)
                    / (np.linalg.norm(normalized_target) + 1e-12)
                ),
                "complex_nrmse_phase_aligned": complex_nrmse,
                "phase_rmse_radians": phase_rmse,
                "phase_alignment_angle": float(torch.angle(phase_factor)),
                "high_change_roi_mae": float(np.mean(np.abs(roi_error))),
                "data_error_full": float(
                    relative_kspace_error(
                        reconstruction.to(DEVICE),
                        item["full_operator"].to(DEVICE),
                        item["full_kspace"].to(DEVICE),
                    )
                ),
            }
            output.update(longitudinal)
            return output
        """
    ),
    markdown(
        r"""
        ## Fit the historical prior once

        The registered DICOM magnitude is fitted independently of follow-up
        k-space. During reconstruction the prior network is frozen. A complete
        cache fingerprint prevents architecture or optimizer changes from
        silently reusing an incompatible state.
        """
    ),
    code(
        r"""
        PRIOR_ARCHITECTURE = {
            "kind": "siren",
            "hidden_features": 256,
            "hidden_layers": 4,
        }

        def make_prior_inr():
            return build_scalar_inr(
                PRIOR_ARCHITECTURE["kind"],
                seed=stable_seed("prior-network", int(sample_row["index"])),
                hidden_features=PRIOR_ARCHITECTURE["hidden_features"],
                hidden_layers=PRIOR_ARCHITECTURE["hidden_layers"],
            )

        prior_payload = {
            "kind": "prior-fit",
            "sample_index": int(sample_row["index"]),
            "architecture": PRIOR_ARCHITECTURE,
            "iterations": PRIOR_ITERS,
            "lr": 1e-4,
            "normalization": "independent-q0.999",
        }
        prior_cache = cache_path("prior", prior_payload)
        if RESUME_CACHE and prior_cache.exists():
            prior_artifact = torch.load(
                prior_cache, map_location="cpu", weights_only=False
            )
            if prior_artifact.get("payload") != prior_payload:
                raise RuntimeError("Prior cache payload mismatch.")
            print("loaded prior cache:", prior_cache.name)
        else:
            set_seed(stable_seed("prior-fit", int(sample_row["index"])))
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
                "payload": prior_payload,
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

        prior_state = prior_artifact["state"]
        prior_probe = make_prior_inr().to(DEVICE)
        prior_probe.load_state_dict(prior_state)
        coordinates = make_coord_grid(*stored_shape, device=DEVICE)
        with torch.no_grad():
            fitted_prior = prior_probe(coordinates)[..., 0].reshape(stored_shape).cpu()
            fitted_prior_nonnegative = fitted_prior.clamp_min(0.0)
        prior_fit_mae = float(torch.mean(torch.abs(fitted_prior - prior)))
        print("prior fit MAE:", prior_fit_mae)
        print("offline prior fit seconds:", prior_artifact["runtime_seconds"])

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        axes[0].imshow(to_numpy(prior), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("registered prior")
        axes[1].imshow(to_numpy(fitted_prior_nonnegative), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("fitted nonnegative prior")
        axes[2].imshow(
            np.abs(to_numpy(fitted_prior_nonnegative - prior)),
            cmap="magma", vmin=0, vmax=0.1,
        )
        axes[2].set_title("absolute prior-fit error")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        plt.show()
        del prior_probe
        cleanup_cuda()
        """
    ),
    markdown(
        r"""
        ## Phase initialization and acquisition-derived prior scale

        Phase INRs are cached by acceleration, source, and data scope. Tuning
        initializers see training lines only. Full-mask initializers are used
        only after configuration and stopping have been frozen.

        The prior scale is the nonnegative real least-squares coefficient

        \[
        \alpha_0 =
        \frac{\operatorname{Re}\langle A(p e^{i\phi_0}),y\rangle}
             {\|A(p e^{i\phi_0})\|_2^2},
        \]

        computed entirely from acquired data. The fitted residual begins at
        zero, so the initial reconstruction is exactly the calibrated prior.
        """
    ),
    code(
        r"""
        PHASE_ARCHITECTURE = {
            "kind": "siren", "hidden_features": 64, "hidden_layers": 3
        }
        PHASE_LR = 1e-4
        PHASE_CG = {"iterations": 25, "lambda_l2": 1e-3, "tolerance": 1e-10}
        phase_memory = {}

        def make_phase_inr(seed):
            return build_scalar_inr(
                PHASE_ARCHITECTURE["kind"],
                seed=seed,
                hidden_features=PHASE_ARCHITECTURE["hidden_features"],
                hidden_layers=PHASE_ARCHITECTURE["hidden_layers"],
            )

        def scope_data(requested_r, scope):
            item = measurements[requested_r]
            if scope == "full":
                return item["full_operator"], item["full_kspace"], item["full_mask"]
            if scope == "train":
                return item["train_operator"], item["train_kspace"], item["train_mask"]
            raise ValueError(f"Unknown data scope: {scope}")

        def get_phase_artifact(requested_r, source="zf", scope="train"):
            key = (requested_r, source, scope)
            if key in phase_memory:
                return phase_memory[key]
            operator, kspace, phase_mask = scope_data(requested_r, scope)
            seed = stable_seed(
                "phase-network", int(sample_row["index"]), requested_r, scope
            )
            payload = {
                "kind": "phase-fit",
                "sample_index": int(sample_row["index"]),
                "requested_r": requested_r,
                "source": source,
                "scope": scope,
                "architecture": PHASE_ARCHITECTURE,
                "iterations": PHASE_ITERS,
                "lr": PHASE_LR,
                "weight_quantile": 0.99,
                "cg": PHASE_CG if source == "cg" else None,
                "seed": seed,
                "mask_hash": tensor_hash(phase_mask),
                "common_scale": float(common_scale),
            }
            path = cache_path("phase", payload)
            if RESUME_CACHE and path.exists():
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                if artifact.get("payload") != payload:
                    raise RuntimeError("Phase cache payload mismatch.")
                phase_memory[key] = artifact
                return artifact

            started = time.perf_counter()
            operator = operator.to(DEVICE)
            kspace = kspace.to(DEVICE)
            if source == "zf":
                phase_image = operator.adjoint(kspace)
            elif source == "cg":
                phase_image = conjugate_gradient_sense(
                    operator,
                    kspace,
                    num_iters=PHASE_CG["iterations"],
                    lambda_l2=PHASE_CG["lambda_l2"],
                    tolerance=PHASE_CG["tolerance"],
                )
            else:
                raise ValueError(f"Unknown phase source: {source}")
            weights = (
                phase_image.abs() / quantile_scale(phase_image, 0.99)
            ).clamp(0.0, 1.0)
            set_seed(seed)
            phase_network = make_phase_inr(seed).to(DEVICE)
            history = fit_phase_inr(
                phase_network,
                torch.angle(phase_image),
                cfg=PhaseFitConfig(
                    iters=PHASE_ITERS,
                    lr=PHASE_LR,
                    log_every=max(1, PHASE_ITERS // 10),
                ),
                weights=weights,
                device=DEVICE,
                verbose=False,
            )
            with torch.no_grad():
                phase_map = phase_network(coordinates)[..., 0].reshape(stored_shape)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            artifact = {
                "payload": payload,
                "state": {
                    name: value.detach().cpu().clone()
                    for name, value in phase_network.state_dict().items()
                },
                "phase": phase_map.detach().cpu(),
                "source_phase": torch.angle(phase_image).detach().cpu(),
                "history": history,
                "runtime_seconds": time.perf_counter() - started,
            }
            torch.save(artifact, path)
            phase_memory[key] = artifact
            del phase_network, phase_image
            cleanup_cuda()
            return artifact

        def calibrated_prior_scale(requested_r, scope="train", phase_source="zf"):
            phase_artifact = get_phase_artifact(requested_r, phase_source, scope)
            operator, kspace, _ = scope_data(requested_r, scope)
            with torch.no_grad():
                scale = float(
                    prior_scale_from_kspace(
                        fitted_prior_nonnegative.to(DEVICE),
                        phase_artifact["phase"].to(DEVICE),
                        operator.to(DEVICE),
                        kspace.to(DEVICE),
                    )
                )
                # Two-iteration smoke phase fits can remain nearly random and
                # yield a nonpositive real projection. The acquired source phase
                # is a safe reference-free fallback; converged full runs should
                # not need it.
                if not np.isfinite(scale) or scale <= 1e-8:
                    scale = float(
                        prior_scale_from_kspace(
                            fitted_prior_nonnegative.to(DEVICE),
                            phase_artifact["source_phase"].to(DEVICE),
                            operator.to(DEVICE),
                            kspace.to(DEVICE),
                        )
                    )
                if not np.isfinite(scale) or scale <= 1e-8:
                    raise RuntimeError(
                        f"Could not obtain a positive prior scale at R={requested_r}, "
                        f"scope={scope}, phase_source={phase_source}."
                    )
                return scale

        calibration_rows = []
        for requested_r in ACCELERATIONS:
            full_alpha = calibrated_prior_scale(requested_r, "full", "zf")
            prior_to_acquisition[requested_r] = full_alpha
            calibration_rows.append(
                {
                    "requested_r": requested_r,
                    "effective_r": mask_info[requested_r].effective_acceleration,
                    "reference_to_acquisition_eval_only": reference_to_acquisition[requested_r],
                    "prior_to_acquisition_reference_free": full_alpha,
                    "prior_over_reference_scale": (
                        full_alpha / reference_to_acquisition[requested_r]
                    ),
                }
            )
        calibration_table = pd.DataFrame(calibration_rows)
        calibration_table.to_csv(OUTPUT_DIR / "acquisition_calibration.csv", index=False)
        display(calibration_table.round(5))

        check_r = ACCELERATIONS[0]
        check_phase = get_phase_artifact(check_r, "zf", "full")["phase"]
        calibrated_prior_copy = torch.polar(
            prior_to_acquisition[check_r] * prior,
            check_phase,
        )
        copied = fixed_unit_metrics(calibrated_prior_copy, check_r)
        perfect = fixed_unit_metrics(
            reference_to_acquisition[check_r] * reference, check_r
        )
        assert abs(copied["change_gain"]) < 1e-6
        assert abs(copied["change_cosine"]) < 1e-6
        assert abs(perfect["change_gain"] - 1.0) < 1e-5
        assert abs(perfect["change_cosine"] - 1.0) < 1e-5
        print("metric sanity checks passed")
        """
    ),
    markdown(
        r"""
        ## Shared trial definition and trainer

        Every trial uses the same reusable post-update checkpoint trainer.
        `best_validation_error` is the sole selection signal. Reference metrics
        are attached afterward as diagnostics.

        The scale calibration always uses zero-filled phase during the clean
        phase-source ablation, so changing `phase_source` changes only the phase
        network initialization.
        """
    ),
    code(
        r"""
        @dataclass(frozen=True)
        class TrialSpec:
            name: str
            delta_width: int = 128
            delta_layers: int = 4
            delta_kind: str = "siren"
            mapping_size: int | None = None
            mapping_sigma: float | None = None
            fourier_seed: int = 31415
            zero_initialize_delta: bool = True
            prior_scale_mode: str = "acquisition_fixed"
            phase_source: str = "zf"
            scale_phase_source: str = "zf"
            magnitude_lr: float = 1e-4
            phase_lr: float = 3e-5
            prior_scale_lr: float = 1e-5
            lambda_delta_l1: float = 3e-4
            lambda_delta_tv: float = 0.0
            lambda_phase_tv: float = 1e-5
            min_lr_ratio: float = 0.05
            grad_clip_norm: float | None = 1.0

        def make_delta_inr(spec, requested_r):
            kwargs = {
                "hidden_features": spec.delta_width,
                "hidden_layers": spec.delta_layers,
            }
            if spec.delta_kind == "fourier_siren":
                kwargs.update(
                    mapping_size=spec.mapping_size,
                    sigma=spec.mapping_sigma,
                )
            return build_scalar_inr(
                spec.delta_kind,
                seed=stable_seed(
                    "delta-network",
                    int(sample_row["index"]),
                    requested_r,
                    spec.delta_kind,
                    spec.delta_width,
                    spec.delta_layers,
                    spec.mapping_size,
                    spec.mapping_sigma,
                    spec.fourier_seed,
                ),
                zero_last=spec.zero_initialize_delta,
                **kwargs,
            )

        def build_prior_trial_model(spec, requested_r, scope):
            prior_network = make_prior_inr()
            prior_network.load_state_dict(prior_state)
            delta_network = make_delta_inr(spec, requested_r)
            phase_seed = stable_seed(
                "phase-network", int(sample_row["index"]), requested_r, scope
            )
            phase_network = make_phase_inr(phase_seed)
            phase_artifact = get_phase_artifact(requested_r, spec.phase_source, scope)
            phase_network.load_state_dict(phase_artifact["state"])

            if spec.prior_scale_mode == "unit":
                alpha = 1.0
                learn_alpha = False
            elif spec.prior_scale_mode == "acquisition_fixed":
                alpha = calibrated_prior_scale(
                    requested_r, scope, spec.scale_phase_source
                )
                learn_alpha = False
            elif spec.prior_scale_mode == "acquisition_learned":
                alpha = calibrated_prior_scale(
                    requested_r, scope, spec.scale_phase_source
                )
                learn_alpha = True
            else:
                raise ValueError(f"Unknown prior scale mode: {spec.prior_scale_mode}")

            model = PriorMagnitudePhaseINR(
                prior_network,
                delta_network,
                phase_network,
                prior_scale=alpha,
                learn_prior_scale=learn_alpha,
            )
            return model, alpha, phase_artifact

        def train_config_for(spec, iterations):
            return MagnitudePhaseTrainConfig(
                iterations=iterations,
                magnitude_lr=spec.magnitude_lr,
                phase_lr=spec.phase_lr,
                prior_scale_lr=spec.prior_scale_lr,
                lambda_delta_l1=spec.lambda_delta_l1,
                lambda_delta_tv=spec.lambda_delta_tv,
                lambda_phase_tv=spec.lambda_phase_tv,
                min_lr_ratio=spec.min_lr_ratio,
                grad_clip_norm=spec.grad_clip_norm,
                eval_every=EVAL_EVERY,
            )

        def run_prior_trial(spec, requested_r, *, scope="train", iterations=TUNE_ITERS):
            item = measurements[requested_r]
            train_operator, train_kspace, train_mask = scope_data(requested_r, scope)
            if scope == "train":
                validation_operator = item["validation_operator"]
                validation_kspace = item["validation_kspace"]
            else:
                validation_operator = None
                validation_kspace = None
            cfg = train_config_for(spec, iterations)
            payload = {
                "kind": "prior-trial",
                "sample_index": int(sample_row["index"]),
                "requested_r": requested_r,
                "scope": scope,
                "spec": asdict(spec),
                "train_config": asdict(cfg),
                "train_mask_hash": tensor_hash(train_mask),
                "validation_mask_hash": (
                    tensor_hash(item["validation_mask"]) if scope == "train" else None
                ),
                "common_scale": float(common_scale),
            }
            path = cache_path("trial", payload)
            if RESUME_CACHE and path.exists():
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                if artifact.get("payload") != payload:
                    raise RuntimeError("Trial cache payload mismatch.")
                print(f"loaded {spec.name:28s} R={requested_r:g} {scope}")
                return artifact

            model, alpha_initial, phase_artifact = build_prior_trial_model(
                spec, requested_r, scope
            )
            set_seed(
                stable_seed(
                    "trial-fit", int(sample_row["index"]), requested_r, scope, spec.name
                )
            )
            result = train_magnitude_phase(
                model,
                train_operator,
                train_kspace,
                stored_shape,
                cfg=cfg,
                validation_operator=validation_operator,
                validation_kspace=validation_kspace,
                device=DEVICE,
                verbose=False,
            )
            diagnostic = fixed_unit_metrics(result.recon, requested_r)
            final_diagnostic = fixed_unit_metrics(result.final_recon, requested_r)
            artifact = {
                "payload": payload,
                "result": result,
                "alpha_initial": alpha_initial,
                "phase_init_runtime_seconds": phase_artifact["runtime_seconds"],
                "metrics": diagnostic,
                "final_metrics": final_diagnostic,
            }
            torch.save(artifact, path)
            print(
                f"finished {spec.name:28s} R={requested_r:g} {scope} "
                f"val={result.best_validation_error if result.best_validation_error is not None else 'fixed-stop'} "
                f"PSNR={diagnostic['fixed_psnr']:.2f}"
            )
            del model, result
            cleanup_cuda()
            return torch.load(path, map_location="cpu", weights_only=False)

        def trial_row(stage, spec, requested_r, artifact):
            result = artifact["result"]
            metrics = artifact["metrics"]
            final = artifact["final_metrics"]
            return {
                "stage": stage,
                "trial": spec.name,
                "requested_r": requested_r,
                "effective_r": mask_info[requested_r].effective_acceleration,
                "best_validation_nrmse": result.best_validation_error,
                "best_validation_iteration": result.best_iteration,
                "fixed_psnr_diagnostic": metrics["fixed_psnr"],
                "laps_psnr_diagnostic": metrics["laps_psnr"],
                "fixed_ssim_diagnostic": metrics["fixed_ssim"],
                "change_cosine_diagnostic": metrics["change_cosine"],
                "change_gain_diagnostic": metrics["change_gain"],
                "mi_prior_delta_diagnostic": metrics["mi_prior_delta"],
                "final_minus_selected_psnr_db": (
                    final["fixed_psnr"] - metrics["fixed_psnr"]
                ),
                "alpha_initial": artifact["alpha_initial"],
                "alpha_selected": float(torch.exp(
                    artifact["result"].state_dict["log_prior_scale"]
                )) if "log_prior_scale" in artifact["result"].state_dict else artifact["alpha_initial"],
                "joint_runtime_seconds": result.runtime_seconds,
                "phase_init_runtime_seconds": artifact["phase_init_runtime_seconds"],
                "online_runtime_seconds": (
                    result.runtime_seconds + artifact["phase_init_runtime_seconds"]
                ),
                "trainable_parameters": result.trainable_parameters,
                "total_parameters": result.total_parameters,
            }

        def select_trial_name(table, override=None):
            summary = (
                table.groupby("trial", as_index=False)
                .agg(
                    mean_validation_nrmse=("best_validation_nrmse", "mean"),
                    worst_validation_nrmse=("best_validation_nrmse", "max"),
                    mean_diagnostic_psnr=("fixed_psnr_diagnostic", "mean"),
                    mean_runtime_seconds=("online_runtime_seconds", "mean"),
                )
                .sort_values(
                    ["mean_validation_nrmse", "worst_validation_nrmse"],
                    ascending=True,
                )
            )
            display(summary.round(5))
            return str(summary.iloc[0]["trial"]) if override is None else str(override)
        """
    ),
    markdown(
        r"""
        ## Stage H — isolate residual initialization and prior scale

        This stage changes one ingredient at a time:

        1. unit prior scale with the historical random residual output;
        2. unit prior scale with an exactly zero residual;
        3. fixed acquisition-derived prior scale with zero residual;
        4. the same calibrated start with a slowly learnable scale.

        No residual bound, phase-source change, or learning-rate change is
        bundled into this comparison.
        """
    ),
    code(
        r"""
        scale_specs = [
            TrialSpec(
                name="unit_random_delta",
                prior_scale_mode="unit",
                zero_initialize_delta=False,
                lambda_delta_l1=1e-3,
            ),
            TrialSpec(
                name="unit_zero_delta",
                prior_scale_mode="unit",
                zero_initialize_delta=True,
                lambda_delta_l1=1e-3,
            ),
            TrialSpec(
                name="calibrated_fixed_zero",
                prior_scale_mode="acquisition_fixed",
                zero_initialize_delta=True,
                lambda_delta_l1=1e-3,
            ),
            TrialSpec(
                name="calibrated_learned_zero",
                prior_scale_mode="acquisition_learned",
                zero_initialize_delta=True,
                prior_scale_lr=1e-5,
                lambda_delta_l1=1e-3,
            ),
        ]
        if SMOKE:
            scale_specs = [scale_specs[1], scale_specs[2]]

        scale_artifacts = {}
        scale_rows = []
        for spec in scale_specs:
            for requested_r in TUNE_RS:
                artifact = run_prior_trial(spec, requested_r)
                scale_artifacts[(spec.name, requested_r)] = artifact
                scale_rows.append(trial_row("scale_init", spec, requested_r, artifact))

        scale_table = pd.DataFrame(scale_rows)
        scale_table.to_csv(OUTPUT_DIR / "scale_initialization_sweep.csv", index=False)
        display(scale_table.round(5))
        # The learnable-alpha row is diagnostic. The deployable formulation
        # deliberately locks acquisition-derived alpha so scale cannot absorb
        # longitudinal change on held-out cases.
        selected_scale_name = (
            "calibrated_fixed_zero"
            if SELECTED_SCALE_NAME is None else str(SELECTED_SCALE_NAME)
        )
        selected_scale_spec = next(
            spec for spec in scale_specs if spec.name == selected_scale_name
        )
        if selected_scale_spec.prior_scale_mode != "acquisition_fixed":
            raise ValueError(
                "The locked follow-up formulation must use acquisition_fixed prior scale; "
                "unit and learned scales are diagnostics only."
            )
        print("selected scale/init formulation:", selected_scale_name)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for name, part in scale_table.groupby("trial"):
            part = part.sort_values("effective_r")
            axes[0].plot(
                part["effective_r"], part["best_validation_nrmse"], marker="o", label=name
            )
            axes[1].plot(
                part["effective_r"], part["fixed_psnr_diagnostic"], marker="o", label=name
            )
            axes[2].plot(
                part["effective_r"], part["alpha_selected"], marker="o", label=name
            )
        axes[0].set_ylabel("held-out k-space NRMSE ↓")
        axes[1].set_ylabel("fixed-unit PSNR, diagnostic (dB)")
        axes[2].set_ylabel("selected prior scale")
        for axis in axes:
            axis.set_xlabel("effective acceleration")
            axis.grid(alpha=0.25)
        axes[2].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Stage I — residual width × sparsity penalty

        With scale handling fixed, this is the only Cartesian grid in the
        notebook. Widths 64 and 128 are crossed with weaker change penalties
        \(\lambda_{\Delta}\in\{0,10^{-4},3\times10^{-4},10^{-3}\}\)
        at R=6 and R=9. The winner minimizes mean held-out k-space NRMSE.
        """
    ),
    code(
        r"""
        width_lambda_specs = []
        for width in WIDTHS:
            for lambda_delta in CHANGE_LAMBDAS:
                label = f"w{width}_lambda_{lambda_delta:.0e}"
                width_lambda_specs.append(
                    replace(
                        selected_scale_spec,
                        name=label,
                        delta_width=width,
                        lambda_delta_l1=lambda_delta,
                    )
                )

        width_lambda_artifacts = {}
        width_lambda_rows = []
        for spec in width_lambda_specs:
            for requested_r in TUNE_RS:
                artifact = run_prior_trial(spec, requested_r)
                width_lambda_artifacts[(spec.name, requested_r)] = artifact
                row = trial_row("width_lambda", spec, requested_r, artifact)
                row.update(
                    delta_width=spec.delta_width,
                    lambda_delta_l1=spec.lambda_delta_l1,
                )
                width_lambda_rows.append(row)

        width_lambda_table = pd.DataFrame(width_lambda_rows)
        width_lambda_table.to_csv(OUTPUT_DIR / "width_lambda_sweep.csv", index=False)
        display(width_lambda_table.round(5))

        width_lambda_summary = (
            width_lambda_table.groupby(
                ["trial", "delta_width", "lambda_delta_l1"], as_index=False
            )
            .agg(
                mean_validation_nrmse=("best_validation_nrmse", "mean"),
                worst_validation_nrmse=("best_validation_nrmse", "max"),
                mean_diagnostic_psnr=("fixed_psnr_diagnostic", "mean"),
                worst_change_gain=("change_gain_diagnostic", "min"),
            )
            .sort_values(
                ["mean_validation_nrmse", "worst_validation_nrmse"], ascending=True
            )
        )
        display(width_lambda_summary.round(5))
        automatic_width_row = width_lambda_summary.iloc[0]
        selected_width = (
            int(automatic_width_row["delta_width"])
            if SELECTED_WIDTH is None else int(SELECTED_WIDTH)
        )
        selected_lambda = (
            float(automatic_width_row["lambda_delta_l1"])
            if SELECTED_LAMBDA is None else float(SELECTED_LAMBDA)
        )
        selected_width_lambda_spec = next(
            spec for spec in width_lambda_specs
            if spec.delta_width == selected_width
            and math.isclose(spec.lambda_delta_l1, selected_lambda)
        )
        print("selected width:", selected_width)
        print("selected lambda_delta_l1:", selected_lambda)

        figure, axes = plt.subplots(1, len(TUNE_RS), figsize=(6 * len(TUNE_RS), 4.5))
        axes = np.atleast_1d(axes)
        for axis, requested_r in zip(axes, TUNE_RS):
            pivot = width_lambda_table[
                width_lambda_table["requested_r"] == requested_r
            ].pivot(
                index="delta_width",
                columns="lambda_delta_l1",
                values="best_validation_nrmse",
            )
            image_handle = axis.imshow(pivot.values, cmap="viridis_r", aspect="auto")
            axis.set_xticks(range(len(pivot.columns)), [f"{v:.0e}" for v in pivot.columns])
            axis.set_yticks(range(len(pivot.index)), [str(v) for v in pivot.index])
            axis.set_xlabel("lambda delta L1")
            axis.set_ylabel("residual width")
            axis.set_title(f"Held-out NRMSE, R={requested_r}")
            for row_index in range(pivot.shape[0]):
                for column_index in range(pivot.shape[1]):
                    axis.text(
                        column_index, row_index,
                        f"{pivot.values[row_index, column_index]:.3f}",
                        ha="center", va="center", color="white", fontsize=8,
                    )
            figure.colorbar(image_handle, ax=axis)
        figure.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Stage J — clean phase initialization ablation

        Zero-filled and 25-iteration CG-SENSE phase initializations now differ
        in exactly one field: `phase_source`. Both use the scale estimated from
        zero-filled phase, the selected residual model, identical initial
        residual weights, and identical optimization.
        """
    ),
    code(
        r"""
        phase_specs = [
            replace(
                selected_width_lambda_spec,
                name="phase_zf",
                phase_source="zf",
                scale_phase_source="zf",
            ),
            replace(
                selected_width_lambda_spec,
                name="phase_cg",
                phase_source="cg",
                scale_phase_source="zf",
            ),
        ]
        if SMOKE:
            phase_specs = phase_specs[:1]

        phase_artifacts = {}
        phase_rows = []
        for spec in phase_specs:
            for requested_r in TUNE_RS:
                artifact = run_prior_trial(spec, requested_r)
                phase_artifacts[(spec.name, requested_r)] = artifact
                row = trial_row("phase", spec, requested_r, artifact)
                row["phase_source"] = spec.phase_source
                phase_rows.append(row)

        phase_table = pd.DataFrame(phase_rows)
        phase_table.to_csv(OUTPUT_DIR / "phase_source_sweep.csv", index=False)
        display(phase_table.round(5))
        selected_phase_name = select_trial_name(phase_table, SELECTED_PHASE_SOURCE)
        if SELECTED_PHASE_SOURCE is None:
            selected_phase_spec = next(
                spec for spec in phase_specs if spec.name == selected_phase_name
            )
        else:
            selected_phase_spec = next(
                spec for spec in phase_specs if spec.phase_source == SELECTED_PHASE_SOURCE
            )
        print("selected phase source:", selected_phase_spec.phase_source)
        """
    ),
    markdown(
        r"""
        ## Stage K — parameter-matched Fourier-feature SIRENs

        Fourier features are applied only to the magnitude residual. The phase
        and prior remain raw-coordinate SIRENs. Fourier widths are selected to
        keep residual trainable parameters close to the raw selected residual,
        preventing extra capacity from masquerading as an encoding benefit.

        The controlled variants are raw coordinates, 32 mappings at bandwidth
        1 or 3, and 64 mappings at bandwidth 3. R=13 is included explicitly to
        test whether extra high-frequency access helps or merely fits the
        k-space null space.
        """
    ),
    code(
        r"""
        def parameter_count(module):
            return sum(parameter.numel() for parameter in module.parameters())

        raw_probe_spec = replace(
            selected_phase_spec,
            name="raw_siren",
            delta_kind="siren",
            mapping_size=None,
            mapping_sigma=None,
        )
        raw_delta_parameters = parameter_count(make_delta_inr(raw_probe_spec, TUNE_RS[0]))

        def closest_fourier_width(mapping_size):
            candidates = []
            for width in range(32, 257):
                probe = build_scalar_inr(
                    "fourier_siren",
                    seed=BASE_SEED,
                    hidden_features=width,
                    hidden_layers=selected_phase_spec.delta_layers,
                    mapping_size=mapping_size,
                    sigma=1.0,
                )
                candidates.append((abs(parameter_count(probe) - raw_delta_parameters), width))
            return min(candidates)[1]

        ff32_width = closest_fourier_width(32)
        ff64_width = closest_fourier_width(64)
        encoding_specs = [raw_probe_spec]
        if RUN_FOURIER:
            encoding_specs.extend(
                [
                    replace(
                        selected_phase_spec,
                        name="ff32_sigma1",
                        delta_kind="fourier_siren",
                        delta_width=ff32_width,
                        mapping_size=32,
                        mapping_sigma=1.0,
                    ),
                    replace(
                        selected_phase_spec,
                        name="ff32_sigma3",
                        delta_kind="fourier_siren",
                        delta_width=ff32_width,
                        mapping_size=32,
                        mapping_sigma=3.0,
                    ),
                    replace(
                        selected_phase_spec,
                        name="ff64_sigma3",
                        delta_kind="fourier_siren",
                        delta_width=ff64_width,
                        mapping_size=64,
                        mapping_sigma=3.0,
                    ),
                ]
            )
        if SMOKE:
            encoding_specs = encoding_specs[:2]

        print("raw residual parameters:", raw_delta_parameters)
        print("matched FF widths: mapping32 ->", ff32_width, ", mapping64 ->", ff64_width)

        encoding_artifacts = {}
        encoding_rows = []
        for spec in encoding_specs:
            for requested_r in FOURIER_RS:
                artifact = run_prior_trial(spec, requested_r)
                encoding_artifacts[(spec.name, requested_r)] = artifact
                row = trial_row("encoding", spec, requested_r, artifact)
                row.update(
                    delta_kind=spec.delta_kind,
                    delta_width=spec.delta_width,
                    mapping_size=spec.mapping_size,
                    mapping_sigma=spec.mapping_sigma,
                )
                encoding_rows.append(row)

        encoding_table = pd.DataFrame(encoding_rows)
        encoding_table.to_csv(OUTPUT_DIR / "fourier_encoding_sweep.csv", index=False)
        display(encoding_table.round(5))
        selected_encoding_name = select_trial_name(
            encoding_table, SELECTED_ENCODING_NAME
        )
        selected_spec = next(
            spec for spec in encoding_specs if spec.name == selected_encoding_name
        )
        print("selected coordinate encoding:", selected_encoding_name)

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        for name, part in encoding_table.groupby("trial"):
            part = part.sort_values("effective_r")
            axes[0].plot(
                part["effective_r"], part["best_validation_nrmse"], marker="o", label=name
            )
            axes[1].plot(
                part["effective_r"], part["fixed_psnr_diagnostic"], marker="o", label=name
            )
            axes[2].plot(
                part["effective_r"], part["change_gain_diagnostic"], marker="o", label=name
            )
        axes[0].set_ylabel("held-out k-space NRMSE ↓")
        axes[1].set_ylabel("fixed-unit PSNR, diagnostic")
        axes[2].set_ylabel("change gain, diagnostic")
        for axis in axes:
            axis.set_xlabel("effective acceleration")
            axis.grid(alpha=0.25)
        axes[2].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Freeze one configuration and stopping iteration

        The selected encoding's best validation iterations are reduced to one
        median stopping time, rounded to the evaluation interval. All later
        reconstructions use this exact model and iteration count on their full
        acquired masks.
        """
    ),
    code(
        r"""
        selected_validation_iterations = [
            encoding_artifacts[(selected_spec.name, requested_r)][
                "result"
            ].best_iteration
            for requested_r in FOURIER_RS
        ]
        if SELECTED_ITERATIONS is None:
            selected_iterations = int(
                round(np.median(selected_validation_iterations) / EVAL_EVERY)
                * EVAL_EVERY
            )
            selected_iterations = max(1, min(selected_iterations, MAX_FINAL_ITERS))
        else:
            selected_iterations = int(SELECTED_ITERATIONS)

        locked_configuration = {
            "schema_version": "presinr-magnitude-phase-v1",
            "status": "locked-development-selection",
            "development_case": {
                "dataset_position": int(sample_row["dataset_position"]),
                "sample_index": int(sample_row["index"]),
                "scan_index": TARGET_SCAN_INDEX,
                "slice_index": TARGET_SLICE_INDEX,
                "change_extent": TARGET_CHANGE_EXTENT,
            },
            "model": {
                "magnitude_kind": selected_spec.delta_kind,
                "magnitude_kwargs": {
                    "hidden_features": selected_spec.delta_width,
                    "hidden_layers": selected_spec.delta_layers,
                    **(
                        {
                            "mapping_size": selected_spec.mapping_size,
                            "sigma": selected_spec.mapping_sigma,
                            "seed": selected_spec.fourier_seed,
                        }
                        if selected_spec.delta_kind == "fourier_siren" else {}
                    ),
                },
                "phase_kind": PHASE_ARCHITECTURE["kind"],
                "phase_kwargs": {
                    "hidden_features": PHASE_ARCHITECTURE["hidden_features"],
                    "hidden_layers": PHASE_ARCHITECTURE["hidden_layers"],
                },
                "magnitude_residual_bound": None,
                "prior_scale_mode": selected_spec.prior_scale_mode,
                "zero_initialize_residual": selected_spec.zero_initialize_delta,
            },
            "initialization": {
                "prior_iterations": PRIOR_ITERS,
                "prior_lr": 1e-4,
                "phase_iterations": PHASE_ITERS,
                "phase_lr": PHASE_LR,
                "current_magnitude_iterations": PHASE_ITERS,
                "current_magnitude_lr": 1e-4,
                "phase_source": selected_spec.phase_source,
                "scale_phase_source": selected_spec.scale_phase_source,
            },
            "training": {
                "iterations": selected_iterations,
                "magnitude_lr": selected_spec.magnitude_lr,
                "phase_lr": selected_spec.phase_lr,
                "prior_scale_lr": selected_spec.prior_scale_lr,
                "lambda_delta_l1": selected_spec.lambda_delta_l1,
                "lambda_delta_tv": selected_spec.lambda_delta_tv,
                "lambda_phase_tv": selected_spec.lambda_phase_tv,
                "min_lr_ratio": selected_spec.min_lr_ratio,
                "grad_clip_norm": selected_spec.grad_clip_norm,
                "eval_every": EVAL_EVERY,
            },
            "sampling": {
                "phase_encode_dim": PHASE_ENCODE_DIM,
                "vd_factor": 0.8,
                "n_candidates": 100,
            },
            "selection": {
                "accelerations": list(TUNE_RS),
                "fourier_accelerations": list(FOURIER_RS),
                "criterion": "heldout_kspace_nrmse",
                "validation_fraction": VALIDATION_FRACTION,
                "selected_stage_rows": {
                    "scale_init": selected_scale_name,
                    "width": selected_width,
                    "lambda_delta_l1": selected_lambda,
                    "phase_source": selected_spec.phase_source,
                    "encoding": selected_spec.name,
                    "validation_iterations": selected_validation_iterations,
                },
            },
            "evaluation": {
                "excluded_scan_indices": [TARGET_SCAN_INDEX],
                "reference_metrics_diagnostic_only": True,
            },
        }
        (OUTPUT_DIR / "locked_configuration.json").write_text(
            json.dumps(locked_configuration, indent=2) + "\n"
        )
        print("fixed stopping iteration:", selected_iterations)
        display(pd.DataFrame([locked_configuration["training"]]))
        """
    ),
    markdown(
        r"""
        ## Full-mask acceleration sweep

        The selected model is restarted at every R and trained on all acquired
        lines for the fixed stopping time. There is no validation checkpoint and
        no reference-dependent selection in this sweep.
        """
    ),
    code(
        r"""
        full_ours_artifacts = {}
        ours_rows = []
        frozen_spec = replace(selected_spec, name=f"locked_{selected_spec.name}")

        for requested_r in ACCELERATIONS:
            artifact = run_prior_trial(
                frozen_spec,
                requested_r,
                scope="full",
                iterations=selected_iterations,
            )
            full_ours_artifacts[requested_r] = artifact
            result = artifact["result"]
            metrics = artifact["metrics"]
            ours_rows.append(
                {
                    "method": "Ours improved (locked)",
                    "requested_r": requested_r,
                    "effective_r": mask_info[requested_r].effective_acceleration,
                    "joint_runtime_seconds": result.runtime_seconds,
                    "phase_init_runtime_seconds": artifact["phase_init_runtime_seconds"],
                    "online_runtime_seconds": (
                        result.runtime_seconds + artifact["phase_init_runtime_seconds"]
                    ),
                    "offline_prior_runtime_seconds": prior_artifact["runtime_seconds"],
                    "trainable_parameters": result.trainable_parameters,
                    "total_parameters": result.total_parameters,
                    "prior_scale_initial": artifact["alpha_initial"],
                    **metrics,
                }
            )

        ours_full_table = pd.DataFrame(ours_rows)
        ours_full_table.to_csv(OUTPUT_DIR / "ours_locked_acceleration.csv", index=False)
        display(
            ours_full_table[
                [
                    "requested_r", "effective_r", "fixed_psnr", "fixed_ssim",
                    "laps_psnr", "laps_ssim", "change_cosine", "change_gain",
                    "mi_prior_delta", "fixed_nrmse", "data_error_full",
                    "online_runtime_seconds", "trainable_parameters",
                ]
            ].round(5)
        )
        """
    ),
    markdown(
        r"""
        ## Stage M — parameter-matched current-only magnitude-phase INR

        This is the control needed to attribute any improvement to the patient
        prior rather than to coordinate-network regularization alone. It uses
        the exact selected magnitude and phase branch architectures. The only
        structural difference is removal of the prior term and residual
        penalty.

        To avoid a dead negative magnitude under `clamp_min`, the current-only
        magnitude branch is initialized by fitting the zero-filled magnitude,
        which is derived solely from its allowed acquired lines. Its stopping
        iteration is selected independently using the same held-out-line rule,
        then frozen before its full-mask acceleration sweep.
        """
    ),
    code(
        r"""
        CURRENT_MAGNITUDE_LR = 1e-4
        current_magnitude_memory = {}

        def get_current_magnitude_artifact(requested_r, scope="train"):
            key = (requested_r, scope, selected_spec.name)
            if key in current_magnitude_memory:
                return current_magnitude_memory[key]
            operator, kspace, magnitude_mask = scope_data(requested_r, scope)
            payload = {
                "kind": "current-magnitude-init",
                "sample_index": int(sample_row["index"]),
                "requested_r": requested_r,
                "scope": scope,
                "selected_spec": asdict(selected_spec),
                "iterations": PHASE_ITERS,
                "lr": CURRENT_MAGNITUDE_LR,
                "mask_hash": tensor_hash(magnitude_mask),
                "common_scale": float(common_scale),
            }
            path = cache_path("current_magnitude", payload)
            if RESUME_CACHE and path.exists():
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                if artifact.get("payload") != payload:
                    raise RuntimeError("Current magnitude cache payload mismatch.")
                current_magnitude_memory[key] = artifact
                return artifact

            magnitude_network = make_delta_inr(
                replace(selected_spec, zero_initialize_delta=False), requested_r
            ).to(DEVICE)
            target = operator.to(DEVICE).adjoint(kspace.to(DEVICE)).abs().reshape(-1)
            optimizer = torch.optim.Adam(
                magnitude_network.parameters(), lr=CURRENT_MAGNITUDE_LR
            )
            started = time.perf_counter()
            history = []
            for iteration in range(PHASE_ITERS):
                optimizer.zero_grad(set_to_none=True)
                prediction = magnitude_network(coordinates)[..., 0]
                loss = (prediction - target).abs().mean()
                loss.backward()
                optimizer.step()
                if (
                    iteration == 0
                    or (iteration + 1) % max(1, PHASE_ITERS // 10) == 0
                    or iteration == PHASE_ITERS - 1
                ):
                    history.append({"iteration": iteration + 1, "l1": float(loss.detach())})
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            artifact = {
                "payload": payload,
                "state": {
                    name: value.detach().cpu().clone()
                    for name, value in magnitude_network.state_dict().items()
                },
                "history": history,
                "runtime_seconds": time.perf_counter() - started,
            }
            torch.save(artifact, path)
            current_magnitude_memory[key] = artifact
            del magnitude_network
            cleanup_cuda()
            return artifact

        def build_current_model(requested_r, scope):
            magnitude_artifact = get_current_magnitude_artifact(requested_r, scope)
            magnitude_network = make_delta_inr(
                replace(selected_spec, zero_initialize_delta=False), requested_r
            )
            magnitude_network.load_state_dict(magnitude_artifact["state"])
            phase_seed = stable_seed(
                "phase-network", int(sample_row["index"]), requested_r, scope
            )
            phase_network = make_phase_inr(phase_seed)
            phase_artifact = get_phase_artifact(
                requested_r, selected_spec.phase_source, scope
            )
            phase_network.load_state_dict(phase_artifact["state"])
            return (
                CurrentMagnitudePhaseINR(magnitude_network, phase_network),
                magnitude_artifact,
                phase_artifact,
            )

        def run_current_trial(requested_r, *, scope="train", iterations=TUNE_ITERS):
            item = measurements[requested_r]
            train_operator, train_kspace, train_mask = scope_data(requested_r, scope)
            validation_operator = item["validation_operator"] if scope == "train" else None
            validation_kspace = item["validation_kspace"] if scope == "train" else None
            current_cfg = MagnitudePhaseTrainConfig(
                iterations=iterations,
                magnitude_lr=selected_spec.magnitude_lr,
                phase_lr=selected_spec.phase_lr,
                lambda_delta_l1=0.0,
                lambda_delta_tv=0.0,
                lambda_phase_tv=selected_spec.lambda_phase_tv,
                min_lr_ratio=selected_spec.min_lr_ratio,
                grad_clip_norm=selected_spec.grad_clip_norm,
                eval_every=EVAL_EVERY,
            )
            payload = {
                "kind": "current-only-trial",
                "sample_index": int(sample_row["index"]),
                "requested_r": requested_r,
                "scope": scope,
                "selected_spec": asdict(selected_spec),
                "train_config": asdict(current_cfg),
                "train_mask_hash": tensor_hash(train_mask),
                "validation_mask_hash": (
                    tensor_hash(item["validation_mask"]) if scope == "train" else None
                ),
                "common_scale": float(common_scale),
            }
            path = cache_path("current_trial", payload)
            if RESUME_CACHE and path.exists():
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                if artifact.get("payload") != payload:
                    raise RuntimeError("Current-only trial cache payload mismatch.")
                print(f"loaded current-only R={requested_r:g} {scope}")
                return artifact

            model, magnitude_artifact, phase_artifact = build_current_model(
                requested_r, scope
            )
            set_seed(
                stable_seed(
                    "current-only-fit", int(sample_row["index"]), requested_r, scope
                )
            )
            result = train_magnitude_phase(
                model,
                train_operator,
                train_kspace,
                stored_shape,
                cfg=current_cfg,
                validation_operator=validation_operator,
                validation_kspace=validation_kspace,
                device=DEVICE,
                verbose=False,
            )
            artifact = {
                "payload": payload,
                "result": result,
                "metrics": fixed_unit_metrics(result.recon, requested_r),
                "magnitude_init_runtime_seconds": magnitude_artifact["runtime_seconds"],
                "phase_init_runtime_seconds": phase_artifact["runtime_seconds"],
            }
            torch.save(artifact, path)
            print(
                f"finished current-only R={requested_r:g} {scope} "
                f"val={result.best_validation_error if result.best_validation_error is not None else 'fixed-stop'}"
            )
            del model, result
            cleanup_cuda()
            return torch.load(path, map_location="cpu", weights_only=False)

        current_validation_artifacts = {}
        current_validation_rows = []
        if RUN_CURRENT_ONLY:
            for requested_r in TUNE_RS:
                artifact = run_current_trial(requested_r, scope="train", iterations=TUNE_ITERS)
                current_validation_artifacts[requested_r] = artifact
                result = artifact["result"]
                current_validation_rows.append(
                    {
                        "requested_r": requested_r,
                        "effective_r": mask_info[requested_r].effective_acceleration,
                        "best_validation_nrmse": result.best_validation_error,
                        "best_validation_iteration": result.best_iteration,
                        "fixed_psnr_diagnostic": artifact["metrics"]["fixed_psnr"],
                        "trainable_parameters": result.trainable_parameters,
                    }
                )
            current_validation_table = pd.DataFrame(current_validation_rows)
            display(current_validation_table.round(5))
            current_selected_iterations = int(
                round(
                    np.median(current_validation_table["best_validation_iteration"])
                    / EVAL_EVERY
                ) * EVAL_EVERY
            )
            current_selected_iterations = max(
                1, min(current_selected_iterations, MAX_FINAL_ITERS)
            )
        else:
            current_validation_table = pd.DataFrame()
            current_selected_iterations = selected_iterations
        print("current-only fixed stopping iteration:", current_selected_iterations)

        current_full_artifacts = {}
        current_rows = []
        if RUN_CURRENT_ONLY:
            for requested_r in ACCELERATIONS:
                artifact = run_current_trial(
                    requested_r,
                    scope="full",
                    iterations=current_selected_iterations,
                )
                current_full_artifacts[requested_r] = artifact
                result = artifact["result"]
                current_rows.append(
                    {
                        "method": "Current-only matched",
                        "requested_r": requested_r,
                        "effective_r": mask_info[requested_r].effective_acceleration,
                        "joint_runtime_seconds": result.runtime_seconds,
                        "phase_init_runtime_seconds": artifact["phase_init_runtime_seconds"],
                        "magnitude_init_runtime_seconds": artifact[
                            "magnitude_init_runtime_seconds"
                        ],
                        "online_runtime_seconds": (
                            result.runtime_seconds
                            + artifact["phase_init_runtime_seconds"]
                            + artifact["magnitude_init_runtime_seconds"]
                        ),
                        "trainable_parameters": result.trainable_parameters,
                        "total_parameters": result.total_parameters,
                        **artifact["metrics"],
                    }
                )
        current_full_table = pd.DataFrame(current_rows)
        if not current_full_table.empty:
            current_validation_table.to_csv(
                OUTPUT_DIR / "current_only_validation.csv", index=False
            )
            current_full_table.to_csv(
                OUTPUT_DIR / "current_only_acceleration.csv", index=False
            )
            display(
                current_full_table[
                    [
                        "requested_r", "effective_r", "fixed_psnr", "fixed_ssim",
                        "change_cosine", "change_gain", "data_error_full",
                        "online_runtime_seconds", "trainable_parameters",
                    ]
                ].round(5)
            )
        """
    ),
    markdown(
        r"""
        ## Stage N — baselines and scale-applied LAPS-NeRP comparison

        Registered prior, zero-filled, and CG-SENSE are recomputed in the same
        acquisition units. The LAPS-NeRP rows are loaded from the immutable
        executed notebook because they used these exact masks and common scale.

        **LAPS-NeRP (+scale applied)** is the primary NeRP reference: its learned
        scale transform participated in the k-space loss. The released output,
        which omits that transform, remains an implementation diagnostic only.
        Historical LAPS rows supply paper-aligned metrics; their old change
        metrics are deliberately discarded.
        """
    ),
    code(
        r"""
        baseline_recons = {}
        baseline_rows = []
        for requested_r in ACCELERATIONS:
            item = measurements[requested_r]
            phase_map = get_phase_artifact(
                requested_r, selected_spec.phase_source, "full"
            )["phase"]
            calibrated_prior_recon = torch.polar(
                prior_to_acquisition[requested_r] * prior,
                phase_map,
            )
            zero_filled = item["zero_filled"].detach().cpu()
            if RUN_CG_BASELINE:
                started = time.perf_counter()
                cg_recon = conjugate_gradient_sense(
                    item["full_operator"].to(DEVICE),
                    item["full_kspace"].to(DEVICE),
                    num_iters=15,
                    lambda_l2=1e-4,
                    tolerance=1e-10,
                ).detach().cpu()
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
                cg_runtime = time.perf_counter() - started
            else:
                cg_recon = zero_filled.clone()
                cg_runtime = 0.0
            baseline_recons[requested_r] = {
                "Registered prior (calibrated)": calibrated_prior_recon,
                "Zero-filled": zero_filled,
                "CG-SENSE": cg_recon,
            }
            for method, reconstruction, runtime in (
                ("Registered prior (calibrated)", calibrated_prior_recon, 0.0),
                ("Zero-filled", zero_filled, 0.0),
                ("CG-SENSE", cg_recon, cg_runtime),
            ):
                baseline_rows.append(
                    {
                        "method": method,
                        "requested_r": requested_r,
                        "effective_r": mask_info[requested_r].effective_acceleration,
                        "online_runtime_seconds": runtime,
                        **fixed_unit_metrics(reconstruction, requested_r),
                    }
                )
        baseline_table = pd.DataFrame(baseline_rows)
        baseline_table.to_csv(OUTPUT_DIR / "context_baselines.csv", index=False)

        legacy_rows = []
        if not legacy_ours_table.empty:
            for _, row in legacy_ours_table.iterrows():
                if int(row["requested_r"]) not in ACCELERATIONS:
                    continue
                legacy_rows.append(
                    {
                        "method": "Historical ours (executed)",
                        "requested_r": int(row["requested_r"]),
                        "effective_r": float(row["effective_r"]),
                        "laps_psnr": float(row["laps_psnr"]),
                        "laps_ssim": float(row["laps_ssim"]),
                        "laps_gain": float(row["laps_gain"]),
                        "online_runtime_seconds": float(row["runtime_seconds"]),
                        "trainable_parameters": float(row["trainable_parameters"]),
                        "metric_note": "Only LAPS metrics reused; old change metrics discarded",
                    }
                )

        if not legacy_nerp_table.empty:
            method_map = {
                "LAPS-NeRP (+scale diagnostic)": "LAPS-NeRP (+scale applied)",
                "LAPS-NeRP (released)": "LAPS-NeRP (released diagnostic)",
            }
            for _, row in legacy_nerp_table.iterrows():
                requested_r = int(row["requested_r"])
                if requested_r not in ACCELERATIONS or row["method"] not in method_map:
                    continue
                legacy_rows.append(
                    {
                        "method": method_map[row["method"]],
                        "requested_r": requested_r,
                        "effective_r": float(row["effective_r"]),
                        "laps_psnr": float(row["laps_psnr"]),
                        "laps_ssim": float(row["laps_ssim"]),
                        "laps_gain": float(row["laps_gain"]),
                        "data_error_full": float(row["data_error_raw"]),
                        "online_runtime_seconds": float(row["runtime_seconds"]),
                        "trainable_parameters": float(row["trainable_parameters"]),
                        "metric_note": "Legacy raw tensor not loaded; fixed-unit fields unavailable",
                    }
                )
        legacy_comparison_table = pd.DataFrame(legacy_rows)

        final_comparison = pd.concat(
            [
                baseline_table,
                ours_full_table,
                current_full_table,
                legacy_comparison_table,
            ],
            ignore_index=True,
            sort=False,
        )
        final_comparison.to_csv(
            OUTPUT_DIR / "final_development_comparison.csv", index=False
        )

        primary_methods = [
            "CG-SENSE",
            "Ours improved (locked)",
            "Current-only matched",
            "LAPS-NeRP (+scale applied)",
        ]
        primary_table = final_comparison[
            final_comparison["method"].isin(primary_methods)
        ].copy()
        display(
            primary_table[
                [
                    "method", "requested_r", "effective_r", "fixed_psnr",
                    "fixed_ssim", "laps_psnr", "laps_ssim", "change_cosine",
                    "change_gain", "mi_prior_delta", "data_error_full",
                    "online_runtime_seconds", "trainable_parameters",
                ]
            ].sort_values(["requested_r", "method"]).round(5)
        )

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
        for method in primary_methods:
            part = primary_table[primary_table["method"] == method].sort_values(
                "effective_r"
            )
            if part.empty:
                continue
            axes[0].plot(part["effective_r"], part["laps_psnr"], marker="o", label=method)
            axes[1].plot(part["effective_r"], part["laps_ssim"], marker="o", label=method)
            if part["change_gain"].notna().any():
                axes[2].plot(
                    part["effective_r"], part["change_gain"], marker="o", label=method
                )
        axes[0].set_ylabel("LAPS PSNR (dB)")
        axes[1].set_ylabel("LAPS SSIM")
        axes[2].set_ylabel("fixed-unit change gain")
        for axis in axes:
            axis.set_xlabel("effective acceleration")
            axis.grid(alpha=0.25)
        axes[2].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Stage O — qualitative magnitude error, change decomposition, and phase

        The first error row is the literal absolute magnitude error in normalized
        reference units; it is not multiplied by five and it uses one common
        display range across methods. The second figure separates our total
        reconstructed change into the calibrated prior-scale component and the
        network's actual learned residual.
        """
    ),
    code(
        r"""
        for requested_r in [r for r in QUALITATIVE_RS if r in ACCELERATIONS]:
            reference_factor = reference_to_acquisition[requested_r]
            methods = {
                "reference": reference_factor * reference,
                "registered prior": baseline_recons[requested_r][
                    "Registered prior (calibrated)"
                ],
                "zero-filled": baseline_recons[requested_r]["Zero-filled"],
                "CG-SENSE": baseline_recons[requested_r]["CG-SENSE"],
                "ours improved": full_ours_artifacts[requested_r]["result"].recon,
            }
            if requested_r in current_full_artifacts:
                methods["current-only"] = current_full_artifacts[requested_r][
                    "result"
                ].recon

            target_normalized = np.abs(to_numpy(reference * support))
            normalized_images = {
                name: np.abs(to_numpy(image.detach().cpu() * support)) / reference_factor
                for name, image in methods.items()
            }
            errors = {
                name: np.abs(image - target_normalized)
                for name, image in normalized_images.items()
            }
            error_vmax = max(
                0.05,
                float(
                    np.quantile(
                        np.concatenate(
                            [value.reshape(-1) for key, value in errors.items() if key != "reference"]
                        ),
                        0.995,
                    )
                ),
            )
            fig, axes = plt.subplots(
                2, len(methods), figsize=(3.0 * len(methods), 6.0), squeeze=False
            )
            for column, (name, image) in enumerate(normalized_images.items()):
                values = laps_metrics(methods[name] * support)
                axes[0, column].imshow(image, cmap="gray", vmin=0, vmax=1)
                axes[0, column].set_title(
                    f"{name}\nPSNR {values['laps_psnr']:.2f}, "
                    f"SSIM {values['laps_ssim']:.3f}", fontsize=9
                )
                axes[1, column].imshow(
                    errors[name], cmap="magma", vmin=0, vmax=error_vmax
                )
                axes[1, column].set_title(
                    f"absolute magnitude error\nrange 0–{error_vmax:.3f}", fontsize=9
                )
            for axis in axes.flat:
                axis.axis("off")
            fig.suptitle(
                f"requested R={requested_r}, effective R="
                f"{mask_info[requested_r].effective_acceleration:.2f}", y=1.02
            )
            fig.tight_layout()
            plt.show()

            ours_artifact = full_ours_artifacts[requested_r]
            ours_result = ours_artifact["result"]
            selected_alpha = float(
                torch.exp(ours_result.state_dict["log_prior_scale"])
            )
            scaled_prior_component = (
                selected_alpha / reference_factor * to_numpy(fitted_prior_nonnegative)
                - prior_np
            )
            learned_delta_component = (
                to_numpy(ours_result.delta) / reference_factor
            )
            reconstructed_change = (
                to_numpy(ours_result.magnitude) / reference_factor - prior_np
            )
            true_change = reference_np - prior_np
            change_error = reconstructed_change - true_change
            limit = max(
                0.05, float(np.quantile(np.abs(true_change), 0.995))
            )

            target_complex = reference_factor * reference.detach().cpu() * support
            foreground = target_complex.abs() > 0.05 * reference_factor
            ours_complex = ours_result.recon * support
            ours_phase_aligned, _ = global_phase_align(
                ours_complex, target_complex, foreground
            )
            phase_error = torch.angle(ours_phase_aligned * target_complex.conj())
            phase_error = torch.where(
                foreground, phase_error, torch.full_like(phase_error, float("nan"))
            )

            fig, axes = plt.subplots(2, 4, figsize=(15, 7.0))
            change_maps = [
                (true_change, "true normalized change"),
                (reconstructed_change, "total reconstructed change"),
                (scaled_prior_component, "global scale component"),
                (learned_delta_component, "actual network delta"),
                (change_error, "change error"),
            ]
            for axis, (values, title) in zip(axes.flat[:5], change_maps):
                axis.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit)
                axis.set_title(title)
                axis.axis("off")
            axes[1, 1].imshow(
                to_numpy(ours_result.phase), cmap="twilight", vmin=-math.pi, vmax=math.pi
            )
            axes[1, 1].set_title("learned phase")
            axes[1, 2].imshow(
                to_numpy(phase_error), cmap="twilight", vmin=-math.pi, vmax=math.pi
            )
            axes[1, 2].set_title("global-phase-aligned phase error")
            raw_error = np.abs(
                to_numpy(ours_result.magnitude) / reference_factor - reference_np
            )
            axes[1, 3].imshow(raw_error, cmap="magma", vmin=0, vmax=error_vmax)
            axes[1, 3].set_title("ours absolute magnitude error")
            for axis in axes.flat[5:]:
                axis.axis("off")
            fig.tight_layout()
            plt.show()
        """
    ),
    markdown(
        r"""
        ## Reference-free convergence and descriptive operating limits

        The convergence plots show why each checkpoint was selected without
        reference images. The operating-limit table remains descriptive for
        this one slice: improved ours must exceed both the calibrated prior and
        zero-filled by 0.5 dB in fixed-unit PSNR, recover positive signed change,
        and is separately compared with CG-SENSE and the matched current-only
        control.
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(1, len(FOURIER_RS), figsize=(5.2 * len(FOURIER_RS), 4))
        axes = np.atleast_1d(axes)
        for axis, requested_r in zip(axes, FOURIER_RS):
            for spec in encoding_specs:
                artifact = encoding_artifacts[(spec.name, requested_r)]
                history = pd.DataFrame(artifact["result"].history)
                axis.plot(
                    history["iteration"], history["validation_error"], label=spec.name
                )
            axis.set_title(f"Held-out convergence, R={requested_r}")
            axis.set_xlabel("post-update iteration")
            axis.set_ylabel("validation k-space NRMSE")
            axis.grid(alpha=0.25)
        axes[-1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        plt.show()

        limit_rows = []
        for requested_r in ACCELERATIONS:
            ours_row = ours_full_table[
                ours_full_table["requested_r"] == requested_r
            ].iloc[0]
            context = baseline_table[
                baseline_table["requested_r"] == requested_r
            ].set_index("method")
            prior_or_zf = max(
                context.loc["Registered prior (calibrated)", "fixed_psnr"],
                context.loc["Zero-filled", "fixed_psnr"],
            )
            current_psnr = (
                float(
                    current_full_table[
                        current_full_table["requested_r"] == requested_r
                    ].iloc[0]["fixed_psnr"]
                )
                if not current_full_table.empty else float("nan")
            )
            limit_rows.append(
                {
                    "requested_r": requested_r,
                    "effective_r": ours_row["effective_r"],
                    "fixed_psnr": ours_row["fixed_psnr"],
                    "gain_over_prior_or_zf_db": ours_row["fixed_psnr"] - prior_or_zf,
                    "gain_over_cg_db": (
                        ours_row["fixed_psnr"] - context.loc["CG-SENSE", "fixed_psnr"]
                    ),
                    "gain_over_current_only_db": ours_row["fixed_psnr"] - current_psnr,
                    "change_cosine": ours_row["change_cosine"],
                    "change_gain": ours_row["change_gain"],
                    "useful": bool(
                        ours_row["fixed_psnr"] >= prior_or_zf + 0.5
                        and ours_row["change_cosine"] > 0
                    ),
                    "beats_cg": bool(
                        ours_row["fixed_psnr"] >= context.loc["CG-SENSE", "fixed_psnr"]
                    ),
                    "beats_current_only": bool(
                        np.isfinite(current_psnr)
                        and ours_row["fixed_psnr"] >= current_psnr
                    ),
                }
            )
        limit_table = pd.DataFrame(limit_rows)
        limit_table.to_csv(OUTPUT_DIR / "descriptive_operating_limits.csv", index=False)
        display(limit_table.round(5))

        def largest_contiguous_effective_r(table, column):
            last = None
            for _, row in table.sort_values("effective_r").iterrows():
                if not bool(row[column]):
                    break
                last = float(row["effective_r"])
            return last

        print("useful contiguous Rmax:", largest_contiguous_effective_r(limit_table, "useful"))
        print("CG-beating contiguous Rmax:", largest_contiguous_effective_r(limit_table, "beats_cg"))
        print(
            "current-only-beating contiguous Rmax:",
            largest_contiguous_effective_r(limit_table, "beats_current_only"),
        )
        """
    ),
    markdown(
        r"""
        ## Export and automatic development-slice conclusions

        All stage tables, the locked consumer configuration, final comparison,
        and operating-limit table have now been written. These conclusions are
        descriptive checks for this development slice only. The next notebook
        must consume `locked_configuration.json` without changing it and must
        exclude scan 16.
        """
    ),
    code(
        r"""
        locked_configuration["initialization"][
            "current_magnitude_iterations"
        ] = PHASE_ITERS
        locked_configuration["training"][
            "current_only_iterations"
        ] = current_selected_iterations
        locked_configuration["outputs"] = {
            "directory": str(OUTPUT_DIR),
            "final_comparison": "final_development_comparison.csv",
            "operating_limits": "descriptive_operating_limits.csv",
        }
        (OUTPUT_DIR / "locked_configuration.json").write_text(
            json.dumps(locked_configuration, indent=2) + "\n"
        )

        best_gain = limit_table.loc[
            limit_table["gain_over_current_only_db"].idxmax()
        ]
        hardest = limit_table.sort_values("effective_r").iloc[-1]
        conclusions = {
            "selected_scale_formulation": selected_scale_name,
            "selected_width": selected_width,
            "selected_lambda_delta_l1": selected_lambda,
            "selected_phase_source": selected_spec.phase_source,
            "selected_encoding": selected_spec.name,
            "fixed_iterations": selected_iterations,
            "current_only_fixed_iterations": current_selected_iterations,
            "largest_useful_effective_r": largest_contiguous_effective_r(
                limit_table, "useful"
            ),
            "largest_cg_beating_effective_r": largest_contiguous_effective_r(
                limit_table, "beats_cg"
            ),
            "largest_current_only_beating_effective_r": largest_contiguous_effective_r(
                limit_table, "beats_current_only"
            ),
            "largest_gain_over_current_only_db": float(
                best_gain["gain_over_current_only_db"]
            ),
            "hardest_r_change_gain": float(hardest["change_gain"]),
            "warning": (
                "Single development slice only. Freeze this configuration and "
                "exclude scan 16 from held-out evaluation."
            ),
        }
        (OUTPUT_DIR / "development_conclusions.json").write_text(
            json.dumps(conclusions, indent=2) + "\n"
        )
        display(pd.DataFrame([conclusions]).T.rename(columns={0: "value"}))
        print("wrote artifacts to", OUTPUT_DIR)
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
REPORT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
print(OUTPUT)
