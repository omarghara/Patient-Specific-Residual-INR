"""Build the single-sample magnitude-phase acceleration tuning notebook.

The generated notebook is intentionally self-contained.  It keeps experimental
code local to the notebook until a winning formulation is identified.
"""

import json
import textwrap
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks" / "magnitude_phase_acceleration_tuning.ipynb"


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
        # Magnitude-phase INR: single-sample acceleration tuning

        This notebook tunes **our patient-specific magnitude-phase residual INR
        first**.  LAPS-NeRP is deliberately postponed to the final section and
        cannot influence model selection.

        The development case is a fully sampled SLAM follow-up with a registered
        prior and radiologist-rated large change.  We retrospectively increase
        one-dimensional acceleration, determine where our formulation fails, and
        separate four possible bottlenecks:

        1. optimizer instability and stopping time;
        2. magnitude-residual network capacity;
        3. residual regularization and prior/current scale handling;
        4. phase initialization.

        This is a **development-sample experiment**, not test-set evidence.  Any
        configuration selected here must be frozen and evaluated on different
        subjects.  Exclude the entire selected scan from the later test cohort.
        """
    ),
    markdown(
        r"""
        ## Controlled setup

        Compared with the paper-style notebook, this notebook removes two
        confounders before attributing failure to the model:

        - the phase-encoding direction is fixed for every acceleration;
        - masks are nested, so every higher-R measurement is a subset of the
          lower-R measurement;
        - one scale computed from k-space common to every mask is used for all R.

        The reference follow-up is never used in the training loss.  PSNR and
        SSIM are evaluated at checkpoints because this is an explicitly
        designated tuning case.  The notebook keeps the final iterate, the
        lowest-training-objective checkpoint, and an oracle best-PSNR checkpoint.
        The oracle checkpoint is diagnostic only.
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
            LapsNerpConfig,
            LapsNerpStageConfig,
            conjugate_gradient_sense,
            fit_laps_nerp,
        )
        from presinr.data.slam import SlamTestSlices
        from presinr.losses import data_consistency, phase_tv_2d, tv_2d
        from presinr.metrics import all_metrics
        from presinr.models import build_inr
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

        The default run is the complete development experiment.  For a fast
        end-to-end software check from a terminal, execute the notebook with
        environment variable PRESINR_TUNING_SMOKE=1.
        """
    ),
    code(
        r"""
        SMOKE = os.environ.get("PRESINR_TUNING_SMOKE", "0") == "1"
        BASE_SEED = 42
        CACHE_VERSION = "magphase-tuning-v1"

        # Robust metadata identity for the default fully sampled, large-change case.
        TARGET_SCAN_INDEX = 16
        TARGET_SLICE_INDEX = 23
        TARGET_CHANGE_EXTENT = 2

        PHASE_ENCODE_DIM = 1
        NESTED_MASKS = True
        ACCELERATIONS = (3, 5, 6, 7, 9, 11, 13)
        CAPACITY_R = 9
        TUNE_RS = (6, 9)

        PRIOR_ITERS = 3000
        PHASE_ITERS = 1000
        JOINT_ITERS = 3000
        EVAL_EVERY = 100
        CAPACITY_WIDTHS = (64, 128, 256, 512)

        SELECTED_DELTA_WIDTH = None
        SELECTED_TRIAL_NAME = None
        SELECTED_ITERATIONS = None

        RESUME_CACHE = True
        RUN_LAPS_NERP = True
        NERP_ACCELERATIONS = ACCELERATIONS
        QUALITATIVE_RS = (3, 6, 9, 13)

        if SMOKE:
            ACCELERATIONS = (3,)
            CAPACITY_R = 3
            TUNE_RS = (3,)
            PRIOR_ITERS = 2
            PHASE_ITERS = 2
            JOINT_ITERS = 2
            EVAL_EVERY = 1
            CAPACITY_WIDTHS = (128,)
            NERP_ACCELERATIONS = (3,)
            QUALITATIVE_RS = (3,)

        OUTPUT_DIR = (
            Path("/tmp/presinr-magnitude-phase-tuning-smoke")
            if SMOKE
            else REPO / "reports" / "magnitude_phase_tuning"
        )
        CACHE_DIR = OUTPUT_DIR / "cache"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        print("smoke mode    :", SMOKE)
        print("accelerations :", ACCELERATIONS)
        print("capacity R    :", CAPACITY_R)
        print("tuning R      :", TUNE_RS)
        print("output         :", OUTPUT_DIR)
        """
    ),
    markdown(
        r"""
        ## Select the development sample

        The default is scan 16, middle slice 23:

        - fully sampled original acquisition, avoiding pre-existing undersampling;
        - change_extent = 2, the large-change stress category;
        - registered prior and current multi-coil k-space are both available.

        Selection is fixed by metadata, not by looking for the slice with the best
        or worst reconstruction score.
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
            raise RuntimeError(
                f"Expected one row for scan={TARGET_SCAN_INDEX}, "
                f"slice={TARGET_SLICE_INDEX}; found {len(match)}."
            )
        sample_row = match.iloc[0]
        if int(sample_row["change_extent"]) != TARGET_CHANGE_EXTENT:
            raise RuntimeError("Selected sample no longer has the expected change extent.")
        if int(sample_row["AccelNumDim"]) != 0:
            raise RuntimeError("The tuning case must be originally fully sampled.")

        dataset_position = int(sample_row["dataset_position"])
        sample = dataset[dataset_position]

        def quantile_scale(tensor, q=0.999):
            values = tensor.detach().abs().reshape(-1).float()
            return float(torch.quantile(values, q)) + 1e-8

        reference = sample["recon"].to(torch.complex64)
        prior = sample["prior"].float()
        reference = reference / quantile_scale(reference, 0.999)
        prior = prior / quantile_scale(prior, 0.999)

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
        print("stored shape :", sample["stored_shape"])
        print("native shape :", sample["native_shape"])
        print("coils        :", sample["ksp"].shape[0])

        prior_np = np.abs(to_numpy(prior))
        reference_np = np.abs(to_numpy(reference))
        true_change = reference_np - prior_np
        change_limit = float(np.quantile(np.abs(true_change), 0.995))

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
        axes[0].imshow(prior_np, cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("registered prior magnitude")
        axes[1].imshow(reference_np, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("follow-up reference (evaluation only)")
        axes[2].imshow(
            true_change,
            cmap="coolwarm",
            vmin=-change_limit,
            vmax=change_limit,
        )
        axes[2].set_title("true signed magnitude change")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Build a controlled acceleration bank

        Masks are generated in increasing R and each new mask is sampled from the
        preceding mask.  Therefore the masks are nested.  One fixed
        phase-encoding direction is used.

        K-space for every R is divided by the same scale computed from samples
        present in all masks.  This prevents an R-specific zero-filled percentile
        from changing the reconstruction units as acceleration changes.
        """
    ),
    code(
        r"""
        def stable_seed(*parts, base=BASE_SEED):
            payload = "|".join(map(str, (base,) + parts)).encode("utf-8")
            return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)

        raw_kspace = sample["ksp"].to(torch.complex64)
        mps = sample["mps"].to(torch.complex64)
        stored_shape = tuple(sample["stored_shape"])

        masks = {}
        mask_info = {}
        source_mask = sample["mask"].float()
        for requested_r in sorted(ACCELERATIONS):
            current_mask, info = laps_retrospective_1d_mask(
                source_mask if NESTED_MASKS else sample["mask"].float(),
                requested_r,
                seed=stable_seed("nested-mask", int(sample_row["index"]), requested_r),
                phase_encode_dim=PHASE_ENCODE_DIM,
                vd_factor=0.8,
                n_candidates=100,
            )
            masks[requested_r] = current_mask
            mask_info[requested_r] = info
            if NESTED_MASKS:
                source_mask = current_mask

        mask_stack = torch.stack([masks[r].bool() for r in ACCELERATIONS])
        common_mask = mask_stack.all(dim=0).float()
        common_operator = CenterPaddedSense(mps, common_mask, stored_shape)
        common_scale = quantile_scale(
            common_operator.adjoint(raw_kspace * common_mask), 0.999
        )

        support_native = torch.linalg.vector_norm(mps, dim=0) > 0.5
        support = center_pad_to(support_native.float(), stored_shape)

        measurements = {}
        measurement_rows = []
        for requested_r in ACCELERATIONS:
            operator = CenterPaddedSense(mps, masks[requested_r], stored_shape)
            kspace = raw_kspace * masks[requested_r] / common_scale
            zero_filled = operator.adjoint(kspace)
            per_r_scale = quantile_scale(
                operator.adjoint(raw_kspace * masks[requested_r]), 0.999
            )
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
                    "old_per_r_scale": per_r_scale,
                    "old_scale_over_common": per_r_scale / common_scale,
                }
            )

        measurement_table = pd.DataFrame(measurement_rows)
        display(measurement_table.round(4))
        print("common k-space scale:", common_scale)

        fig, axes = plt.subplots(
            2,
            len(ACCELERATIONS),
            figsize=(3.0 * len(ACCELERATIONS), 5.8),
            squeeze=False,
        )
        for column, requested_r in enumerate(ACCELERATIONS):
            item = measurements[requested_r]
            axes[0, column].imshow(to_numpy(item["mask"]), cmap="gray", vmin=0, vmax=1)
            axes[0, column].set_title(
                f"requested R={requested_r}\n"
                f"effective R={item['info'].effective_acceleration:.2f}"
            )
            axes[1, column].imshow(
                np.abs(to_numpy(item["zero_filled"] * support)),
                cmap="gray",
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
        ## Evaluation and cache helpers

        LAPS-style PSNR and SSIM use an optimal scalar magnitude alignment.  The
        qualitative error maps use the corresponding reference-scale alignment,
        so brightness mismatch is not confused with structural error.  The
        required gain is retained as a separate scale-stability diagnostic.
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
            gain = (
                1.0
                if denominator <= 1e-20
                else float(np.sum(estimate * target) / denominator)
            )
            return gain * estimate, target, gain

        def align_to_reference_scale(reconstruction, target_reference=reference):
            target = np.abs(to_numpy(target_reference)).astype(np.float64)
            estimate = np.abs(to_numpy(reconstruction)).astype(np.float64)
            denominator = float(np.sum(estimate * estimate))
            gain = (
                1.0
                if denominator <= 1e-20
                else float(np.sum(estimate * target) / denominator)
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

        def full_metrics(raw_reconstruction, requested_r):
            item = measurements[requested_r]
            raw = raw_reconstruction.detach().cpu()
            evaluated = raw * support
            aligned, target, reference_gain = align_to_reference_scale(evaluated)
            aligned_complex = evaluated * reference_gain
            longitudinal = all_metrics(aligned_complex, reference, prior)
            error = aligned - target
            output = {
                **laps_metrics(evaluated, reference),
                "reference_gain": reference_gain,
                "aligned_mae": float(np.mean(np.abs(error))),
                "aligned_rmse": float(np.sqrt(np.mean(error**2))),
                "aligned_nrmse": float(
                    np.linalg.norm(error) / (np.linalg.norm(target) + 1e-12)
                ),
                "data_error_raw": relative_data_error(
                    raw, item["operator"], item["kspace"]
                ),
                "data_error_supported": relative_data_error(
                    evaluated, item["operator"], item["kspace"]
                ),
            }
            output.update(
                {
                    key: longitudinal[key]
                    for key in (
                        "change_cosine",
                        "change_gain",
                        "mi_prior_ref",
                        "mi_prior_recon",
                        "mi_prior_delta",
                    )
                }
            )
            return output
        """
    ),
    markdown(
        r"""
        ## Minimal reconstruction context

        Registered prior, zero-filled, and CG-SENSE are context for deciding
        whether our tuned model remains useful.  They do not participate in
        hyperparameter selection.
        """
    ),
    code(
        r"""
        baseline_rows = []
        baseline_recons = {}

        for requested_r in ACCELERATIONS:
            item = measurements[requested_r]
            operator = item["operator"].to(DEVICE)
            kspace = item["kspace"].to(DEVICE)

            prior_raw = prior.clone()
            zf_raw = item["zero_filled"].clone()
            started = time.perf_counter()
            cg_raw = conjugate_gradient_sense(
                operator,
                kspace,
                num_iters=15,
                lambda_l2=1e-4,
                tolerance=1e-10,
            ).detach().cpu()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            cg_runtime = time.perf_counter() - started

            baseline_recons[requested_r] = {
                "Registered prior": prior_raw,
                "Zero-filled": zf_raw,
                "CG-SENSE": cg_raw,
            }
            for method, reconstruction, runtime in (
                ("Registered prior", prior_raw, 0.0),
                ("Zero-filled", zf_raw, 0.0),
                ("CG-SENSE", cg_raw, cg_runtime),
            ):
                baseline_rows.append(
                    {
                        "method": method,
                        "requested_r": requested_r,
                        "effective_r": item["info"].effective_acceleration,
                        "runtime_seconds": runtime,
                        **full_metrics(reconstruction, requested_r),
                    }
                )
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
                    "data_error_raw",
                ]
            ].round(4)
        )
        """
    ),
    markdown(
        r"""
        ## Fit the prior once

        The prior SIREN has 264,193 parameters, but it is frozen during
        reconstruction.  Every capacity and optimizer trial below receives the
        exact same fitted prior state.
        """
    ),
    code(
        r"""
        def make_prior_inr():
            return build_inr(
                "siren",
                out_features=1,
                hidden_features=256,
                hidden_layers=4,
            )

        prior_payload = {
            "kind": "prior",
            "sample_index": int(sample_row["index"]),
            "iters": PRIOR_ITERS,
            "lr": 1e-4,
            "seed": stable_seed("prior", int(sample_row["index"])),
        }
        prior_cache = cache_path("prior", prior_payload)

        if RESUME_CACHE and prior_cache.exists():
            prior_artifact = torch.load(prior_cache, map_location="cpu", weights_only=False)
            print("loaded prior cache:", prior_cache.name)
        else:
            set_seed(prior_payload["seed"])
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
                    key: value.detach().cpu().clone()
                    for key, value in prior_network.state_dict().items()
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
        prior_fit_mae = float(torch.mean(torch.abs(fitted_prior - prior)))

        print("prior parameters :", count_parameters(prior_probe))
        print("prior fit MAE    :", prior_fit_mae)
        print("prior runtime s  :", prior_artifact["runtime_seconds"])

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        axes[0].imshow(to_numpy(prior), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("registered prior")
        axes[1].imshow(to_numpy(fitted_prior), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("fitted prior INR")
        axes[2].imshow(
            np.abs(to_numpy(fitted_prior - prior)),
            cmap="magma",
            vmin=0,
            vmax=0.2,
        )
        axes[2].set_title("absolute fit error")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 3.2))
        plt.plot(prior_artifact["history"]["loss"])
        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel("prior L1")
        plt.title("Prior fit convergence")
        plt.grid(alpha=0.25)
        plt.show()
        del prior_probe
        cleanup_cuda()
        """
    ),
    markdown(
        r"""
        ## Deterministic phase initialization

        Phase initialization is cached per acceleration and source.  Every trial
        at the same R begins from exactly the same phase-network weights.

        The default source is the zero-filled phase.  A later trial also tests
        phase from a 25-iteration CG-SENSE reconstruction.
        """
    ),
    code(
        r"""
        def make_phase_inr():
            return build_inr(
                "siren",
                out_features=1,
                hidden_features=64,
                hidden_layers=3,
            )

        phase_memory = {}

        def get_phase_artifact(requested_r, source="zf"):
            key = (requested_r, source)
            if key in phase_memory:
                return phase_memory[key]

            payload = {
                "kind": "phase",
                "sample_index": int(sample_row["index"]),
                "requested_r": requested_r,
                "source": source,
                "iters": PHASE_ITERS,
                "common_scale": common_scale,
                "mask_hash": cache_fingerprint(
                    {"mask": masks[requested_r].cpu().numpy().tobytes().hex()}
                ),
            }
            path = cache_path("phase", payload)
            if RESUME_CACHE and path.exists():
                artifact = torch.load(path, map_location="cpu", weights_only=False)
                phase_memory[key] = artifact
                return artifact

            item = measurements[requested_r]
            operator = item["operator"].to(DEVICE)
            kspace = item["kspace"].to(DEVICE)
            if source == "zf":
                phase_image = operator.adjoint(kspace)
            elif source == "cg":
                phase_image = conjugate_gradient_sense(
                    operator,
                    kspace,
                    num_iters=25,
                    lambda_l2=1e-3,
                    tolerance=1e-10,
                )
            else:
                raise ValueError(f"Unknown phase source: {source}")

            weights = (
                phase_image.abs() / quantile_scale(phase_image, 0.99)
            ).clamp(0.0, 1.0)
            set_seed(stable_seed("phase", int(sample_row["index"]), requested_r, source))
            network = make_phase_inr().to(DEVICE)
            started = time.perf_counter()
            history = fit_phase_inr(
                network,
                torch.angle(phase_image),
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
            artifact = {
                "state": {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items()
                },
                "history": history,
                "runtime_seconds": time.perf_counter() - started,
                "source_phase": torch.angle(phase_image).detach().cpu(),
            }
            torch.save(artifact, path)
            phase_memory[key] = artifact
            del network, phase_image
            cleanup_cuda()
            return artifact

        phase_preview = get_phase_artifact(CAPACITY_R, "zf")
        phase_probe = make_phase_inr().to(DEVICE)
        phase_probe.load_state_dict(phase_preview["state"])
        with torch.no_grad():
            fitted_phase = phase_probe(coordinates)[..., 0].reshape(stored_shape).cpu()

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        axes[0].imshow(
            to_numpy(phase_preview["source_phase"]),
            cmap="twilight",
            vmin=-math.pi,
            vmax=math.pi,
        )
        axes[0].set_title(f"zero-filled phase, R={CAPACITY_R}")
        axes[1].imshow(
            to_numpy(fitted_phase),
            cmap="twilight",
            vmin=-math.pi,
            vmax=math.pi,
        )
        axes[1].set_title("phase INR initialization")
        axes[2].plot(phase_preview["history"]["loss"])
        axes[2].set_yscale("log")
        axes[2].set_title("circular-loss history")
        axes[2].set_xlabel("iteration")
        for axis in axes[:2]:
            axis.axis("off")
        fig.tight_layout()
        plt.show()
        del phase_probe
        cleanup_cuda()
        """
    ),
    markdown(
        r"""
        ## Tunable model and checkpoint-aware trainer

        The prior network stays frozen.  The trainer changes only the
        magnitude-residual architecture, optimization, regularization, optional
        residual bound, phase initialization, and an optional scalar multiplying
        the prior magnitude.

        Reference metrics are logged but never enter the loss or gradients.
        """
    ),
    code(
        r"""
        @dataclass(frozen=True)
        class TrialSpec:
            name: str
            delta_width: int = 128
            delta_layers: int = 4
            delta_lr: float = 3e-4
            phase_lr: float = 1e-4
            scale_lr: float = 1e-3
            lambda_change: float = 1e-3
            lambda_change_tv: float = 0.0
            lambda_phase_tv: float = 1e-5
            lambda_scale: float = 1e-4
            magnitude_bound: float | None = None
            learn_prior_scale: bool = False
            phase_source: str = "zf"
            schedule: str = "cosine"
            grad_clip: float | None = 1.0


        class TunableMagnitudePhaseINR(nn.Module):
            def __init__(
                self,
                prior_inr,
                delta_inr,
                phase_inr,
                *,
                magnitude_bound=None,
                learn_prior_scale=False,
            ):
                super().__init__()
                self.prior_inr = prior_inr
                self.delta_inr = delta_inr
                self.phase_inr = phase_inr
                self.magnitude_bound = magnitude_bound
                self.log_prior_scale = nn.Parameter(
                    torch.zeros(()), requires_grad=learn_prior_scale
                )

            def freeze_prior(self):
                for parameter in self.prior_inr.parameters():
                    parameter.requires_grad_(False)
                self.prior_inr.eval()

            def components(self, coords, prior_magnitude=None):
                prior_value = (
                    self.prior_inr(coords)[..., 0]
                    if prior_magnitude is None
                    else prior_magnitude
                )
                scale = torch.exp(self.log_prior_scale)
                raw_delta = self.delta_inr(coords)[..., 0]
                delta = (
                    raw_delta
                    if self.magnitude_bound is None
                    else self.magnitude_bound * torch.tanh(raw_delta)
                )
                magnitude = torch.clamp_min(scale * prior_value + delta, 0.0)
                phase = self.phase_inr(coords)[..., 0]
                return prior_value, raw_delta, delta, magnitude, phase, scale


        def build_trial_model(spec):
            prior_network = make_prior_inr()
            prior_network.load_state_dict(prior_state)
            delta_network = build_inr(
                "siren",
                out_features=1,
                hidden_features=spec.delta_width,
                hidden_layers=spec.delta_layers,
            )
            phase_network = make_phase_inr()
            model = TunableMagnitudePhaseINR(
                prior_network,
                delta_network,
                phase_network,
                magnitude_bound=spec.magnitude_bound,
                learn_prior_scale=spec.learn_prior_scale,
            )
            return model


        def run_trial(spec, requested_r, *, iters=JOINT_ITERS, schedule_horizon=None):
            schedule_horizon = JOINT_ITERS if schedule_horizon is None else schedule_horizon
            phase_artifact = get_phase_artifact(requested_r, spec.phase_source)
            payload = {
                "kind": "joint-trial",
                "sample_index": int(sample_row["index"]),
                "requested_r": requested_r,
                "spec": asdict(spec),
                "iters": iters,
                "schedule_horizon": schedule_horizon,
                "eval_every": EVAL_EVERY,
                "common_scale": common_scale,
                "mask_hash": cache_fingerprint(
                    {"mask": masks[requested_r].cpu().numpy().tobytes().hex()}
                ),
            }
            path = cache_path("trial", payload)
            if RESUME_CACHE and path.exists():
                print(f"loaded {spec.name:24s} R={requested_r:g}")
                return torch.load(path, map_location="cpu", weights_only=False)

            item = measurements[requested_r]
            operator = item["operator"].to(DEVICE)
            kspace = item["kspace"].to(DEVICE)
            coords = make_coord_grid(*stored_shape, device=DEVICE)

            # The seed excludes optimizer settings so optimizer variants with the
            # same architecture start from the same delta weights.
            set_seed(
                stable_seed(
                    "joint-init",
                    int(sample_row["index"]),
                    requested_r,
                    spec.delta_width,
                    spec.delta_layers,
                    spec.phase_source,
                )
            )
            model = build_trial_model(spec).to(DEVICE)
            model.phase_inr.load_state_dict(phase_artifact["state"])
            model.freeze_prior()
            with torch.no_grad():
                prior_magnitude = model.prior_inr(coords)[..., 0]

            parameter_groups = [
                {"params": model.delta_inr.parameters(), "lr": spec.delta_lr},
                {"params": model.phase_inr.parameters(), "lr": spec.phase_lr},
            ]
            if spec.learn_prior_scale:
                parameter_groups.append(
                    {"params": [model.log_prior_scale], "lr": spec.scale_lr}
                )
            optimizer = torch.optim.Adam(parameter_groups)

            scheduler = None
            if spec.schedule == "cosine":
                def decay(step):
                    fraction = min(float(step) / max(1, schedule_horizon), 1.0)
                    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * fraction))
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lr_lambda=decay
                )
            elif spec.schedule != "constant":
                raise ValueError(f"Unknown schedule: {spec.schedule}")

            history = []
            best_objective = float("inf")
            best_psnr = -float("inf")
            best_objective_recon = None
            best_psnr_recon = None
            best_objective_iteration = None
            best_psnr_iteration = None

            def forward_terms():
                (
                    _,
                    raw_delta,
                    delta,
                    magnitude,
                    phase,
                    prior_scale,
                ) = model.components(coords, prior_magnitude=prior_magnitude)
                image = torch.polar(magnitude, phase).reshape(stored_shape)
                dc = data_consistency(operator(image), kspace, mask=operator.mask)
                change_l1 = spec.lambda_change * delta.abs().mean()
                change_tv = (
                    spec.lambda_change_tv * tv_2d(delta.reshape(stored_shape))
                    if spec.lambda_change_tv > 0
                    else torch.zeros((), device=DEVICE)
                )
                phase_tv = (
                    spec.lambda_phase_tv * phase_tv_2d(phase.reshape(stored_shape))
                    if spec.lambda_phase_tv > 0
                    else torch.zeros((), device=DEVICE)
                )
                scale_penalty = (
                    spec.lambda_scale * model.log_prior_scale.square()
                    if spec.learn_prior_scale
                    else torch.zeros((), device=DEVICE)
                )
                regularization = change_l1 + change_tv + phase_tv + scale_penalty
                total = dc + regularization
                saturation = (
                    0.0
                    if spec.magnitude_bound is None
                    else float(
                        (delta.abs() >= 0.95 * spec.magnitude_bound)
                        .float()
                        .mean()
                        .detach()
                    )
                )
                return (
                    image,
                    total,
                    dc,
                    regularization,
                    prior_scale,
                    saturation,
                )

            started = time.perf_counter()
            for iteration in range(iters):
                optimizer.zero_grad(set_to_none=True)
                image, total, dc, regularization, prior_scale_value, saturation = (
                    forward_terms()
                )

                should_evaluate = (
                    iteration == 0
                    or (iteration + 1) % EVAL_EVERY == 0
                    or iteration == iters - 1
                )
                if should_evaluate:
                    checkpoint = image.detach().cpu()
                    quality = laps_metrics(checkpoint * support, reference)
                    row = {
                        "iteration": iteration,
                        "total": float(total.detach()),
                        "dc": float(dc.detach()),
                        "reg": float(regularization.detach()),
                        "laps_psnr": quality["laps_psnr"],
                        "laps_ssim": quality["laps_ssim"],
                        "laps_gain": quality["laps_gain"],
                        "prior_scale": float(prior_scale_value.detach()),
                        "bound_saturation": saturation,
                        "delta_lr": optimizer.param_groups[0]["lr"],
                        "phase_lr": optimizer.param_groups[1]["lr"],
                    }
                    history.append(row)
                    if row["total"] < best_objective:
                        best_objective = row["total"]
                        best_objective_recon = checkpoint.clone()
                        best_objective_iteration = iteration
                    if row["laps_psnr"] > best_psnr:
                        best_psnr = row["laps_psnr"]
                        best_psnr_recon = checkpoint.clone()
                        best_psnr_iteration = iteration

                total.backward()
                if spec.grad_clip is not None:
                    trainable = [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ]
                    torch.nn.utils.clip_grad_norm_(trainable, spec.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            with torch.no_grad():
                final_recon, final_total, _, _, final_scale, final_saturation = (
                    forward_terms()
                )
                final_recon = final_recon.detach().cpu()
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            runtime = time.perf_counter() - started

            if best_objective_recon is None:
                best_objective_recon = final_recon.clone()
                best_objective_iteration = iters
            if best_psnr_recon is None:
                best_psnr_recon = final_recon.clone()
                best_psnr_iteration = iters

            result = {
                "spec": asdict(spec),
                "requested_r": requested_r,
                "runtime_seconds": runtime,
                "trainable_parameters": count_parameters(model, trainable_only=True),
                "total_parameters": count_parameters(model),
                "history": history,
                "best_objective_iteration": best_objective_iteration,
                "best_psnr_iteration": best_psnr_iteration,
                "final_prior_scale": float(final_scale),
                "final_bound_saturation": final_saturation,
                "recons": {
                    "best_objective": best_objective_recon,
                    "best_psnr": best_psnr_recon,
                    "final": final_recon,
                },
                "metrics": {
                    "best_objective": full_metrics(
                        best_objective_recon, requested_r
                    ),
                    "best_psnr": full_metrics(best_psnr_recon, requested_r),
                    "final": full_metrics(final_recon, requested_r),
                },
            }
            torch.save(result, path)
            print(
                f"finished {spec.name:24s} R={requested_r:g} "
                f"best={result['metrics']['best_psnr']['laps_psnr']:.2f} dB "
                f"final={result['metrics']['final']['laps_psnr']:.2f} dB"
            )
            del model
            cleanup_cuda()
            return result
        """
    ),
    markdown(
        r"""
        ## Stage A — magnitude-network capacity at high acceleration

        The optimizer is held fixed while magnitude width changes.  This asks
        whether the current 66,561-parameter magnitude residual is genuinely
        capacity limited.

        Selection uses oracle PSNR only because this is a declared development
        case.  Parameter count and final-versus-best gaps must also be inspected:
        a larger model can fit the k-space null space more aggressively and become
        worse.
        """
    ),
    code(
        r"""
        capacity_results = {}
        capacity_base = TrialSpec(
            name="capacity",
            delta_lr=3e-4,
            phase_lr=1e-4,
            schedule="cosine",
            grad_clip=1.0,
        )

        for width in CAPACITY_WIDTHS:
            spec = replace(
                capacity_base,
                name=f"width_{width}",
                delta_width=width,
            )
            capacity_results[width] = run_trial(spec, CAPACITY_R)

        capacity_rows = []
        for width, result in capacity_results.items():
            best = result["metrics"]["best_psnr"]
            objective = result["metrics"]["best_objective"]
            final = result["metrics"]["final"]
            capacity_rows.append(
                {
                    "delta_width": width,
                    "trainable_parameters": result["trainable_parameters"],
                    "total_parameters_including_frozen_prior": result["total_parameters"],
                    "best_psnr": best["laps_psnr"],
                    "best_ssim": best["laps_ssim"],
                    "best_iteration": result["best_psnr_iteration"],
                    "objective_checkpoint_psnr": objective["laps_psnr"],
                    "final_psnr": final["laps_psnr"],
                    "final_minus_best_db": final["laps_psnr"] - best["laps_psnr"],
                    "change_cosine": best["change_cosine"],
                    "mi_prior_delta": best["mi_prior_delta"],
                    "runtime_seconds": result["runtime_seconds"],
                }
            )

        capacity_table = pd.DataFrame(capacity_rows).sort_values("delta_width")
        capacity_table.to_csv(OUTPUT_DIR / "capacity_sweep.csv", index=False)
        display(capacity_table.round(4))

        if SELECTED_DELTA_WIDTH is None:
            selected_delta_width = int(
                capacity_table.loc[capacity_table["best_psnr"].idxmax(), "delta_width"]
            )
        else:
            selected_delta_width = int(SELECTED_DELTA_WIDTH)
        print("selected magnitude width:", selected_delta_width)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(
            capacity_table["trainable_parameters"],
            capacity_table["best_psnr"],
            marker="o",
        )
        for _, row in capacity_table.iterrows():
            axes[0].annotate(
                f"W={int(row['delta_width'])}",
                (row["trainable_parameters"], row["best_psnr"]),
            )
        axes[0].set_xscale("log")
        axes[0].set_xlabel("trainable parameters")
        axes[0].set_ylabel("oracle-best LAPS PSNR (dB)")
        axes[0].set_title(f"Capacity frontier at R={CAPACITY_R}")

        axes[1].plot(
            capacity_table["delta_width"],
            capacity_table["final_minus_best_db"],
            marker="o",
        )
        axes[1].axhline(0, color="black", linewidth=1)
        axes[1].set_xlabel("magnitude width")
        axes[1].set_ylabel("final minus best PSNR (dB)")
        axes[1].set_title("Late-optimization damage")

        axes[2].plot(
            capacity_table["runtime_seconds"],
            capacity_table["best_psnr"],
            marker="o",
        )
        axes[2].set_xlabel("runtime (s)")
        axes[2].set_ylabel("oracle-best LAPS PSNR (dB)")
        axes[2].set_title("Performance/runtime frontier")
        for axis in axes:
            axis.grid(alpha=0.25)
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Stage B — optimization, regularization, scale, and phase

        These are staged hypotheses, not a Cartesian product:

        - current_default reproduces the present optimizer;
        - stable_cosine lowers learning rates, clips gradients, and decays them;
        - low_lr tests whether high-R instability is mainly step size;
        - stronger_change tests more prior/change regularization;
        - scale_and_bound adds a learned global prior scale and bounds delta m;
        - cg_phase_scale tests whether phase initialization is the bottleneck.

        Each candidate is scored at R=6 and R=9 with the selected capacity.
        """
    ),
    code(
        r"""
        optimization_specs = [
            TrialSpec(
                name="current_default",
                delta_width=selected_delta_width,
                delta_lr=1e-3,
                phase_lr=3e-4,
                schedule="constant",
                grad_clip=None,
            ),
            TrialSpec(
                name="stable_cosine",
                delta_width=selected_delta_width,
                delta_lr=3e-4,
                phase_lr=1e-4,
                schedule="cosine",
                grad_clip=1.0,
            ),
            TrialSpec(
                name="low_lr",
                delta_width=selected_delta_width,
                delta_lr=1e-4,
                phase_lr=3e-5,
                schedule="cosine",
                grad_clip=1.0,
            ),
            TrialSpec(
                name="stronger_change",
                delta_width=selected_delta_width,
                delta_lr=3e-4,
                phase_lr=1e-4,
                lambda_change=3e-3,
                lambda_change_tv=1e-5,
                schedule="cosine",
                grad_clip=1.0,
            ),
            TrialSpec(
                name="scale_and_bound",
                delta_width=selected_delta_width,
                delta_lr=3e-4,
                phase_lr=1e-4,
                magnitude_bound=0.5,
                learn_prior_scale=True,
                schedule="cosine",
                grad_clip=1.0,
            ),
            TrialSpec(
                name="cg_phase_scale",
                delta_width=selected_delta_width,
                delta_lr=3e-4,
                phase_lr=1e-4,
                magnitude_bound=0.5,
                learn_prior_scale=True,
                phase_source="cg",
                schedule="cosine",
                grad_clip=1.0,
            ),
        ]
        if SMOKE:
            optimization_specs = [
                replace(
                    optimization_specs[1],
                    name="smoke_stable",
                    delta_width=selected_delta_width,
                )
            ]

        optimization_results = {}
        optimization_rows = []
        for spec in optimization_specs:
            for requested_r in TUNE_RS:
                result = run_trial(spec, requested_r)
                optimization_results[(spec.name, requested_r)] = result
                best = result["metrics"]["best_psnr"]
                objective = result["metrics"]["best_objective"]
                final = result["metrics"]["final"]
                optimization_rows.append(
                    {
                        "trial": spec.name,
                        "requested_r": requested_r,
                        "effective_r": measurements[requested_r][
                            "info"
                        ].effective_acceleration,
                        "trainable_parameters": result["trainable_parameters"],
                        "best_psnr": best["laps_psnr"],
                        "best_ssim": best["laps_ssim"],
                        "best_iteration": result["best_psnr_iteration"],
                        "objective_psnr": objective["laps_psnr"],
                        "final_psnr": final["laps_psnr"],
                        "final_minus_best_db": final["laps_psnr"] - best["laps_psnr"],
                        "change_cosine": best["change_cosine"],
                        "change_gain": best["change_gain"],
                        "mi_prior_delta": best["mi_prior_delta"],
                        "aligned_rmse": best["aligned_rmse"],
                        "data_error_raw": best["data_error_raw"],
                        "prior_scale": result["final_prior_scale"],
                        "bound_saturation": result["final_bound_saturation"],
                        "runtime_seconds": result["runtime_seconds"],
                    }
                )

        optimization_table = pd.DataFrame(optimization_rows)
        optimization_table.to_csv(OUTPUT_DIR / "optimization_sweep.csv", index=False)
        display(optimization_table.round(4))

        optimization_summary = (
            optimization_table.groupby("trial", as_index=False)
            .agg(
                mean_best_psnr=("best_psnr", "mean"),
                mean_best_ssim=("best_ssim", "mean"),
                worst_change_cosine=("change_cosine", "min"),
                mean_abs_mi_delta=("mi_prior_delta", lambda x: np.mean(np.abs(x))),
                mean_final_minus_best_db=("final_minus_best_db", "mean"),
                median_best_iteration=("best_iteration", "median"),
                mean_runtime_seconds=("runtime_seconds", "mean"),
            )
            .sort_values("mean_best_psnr", ascending=False)
        )
        display(optimization_summary.round(4))

        if SELECTED_TRIAL_NAME is None:
            selected_trial_name = str(optimization_summary.iloc[0]["trial"])
        else:
            selected_trial_name = SELECTED_TRIAL_NAME
        selected_spec = next(
            spec for spec in optimization_specs if spec.name == selected_trial_name
        )

        selected_iterations_from_tuning = [
            optimization_results[(selected_trial_name, requested_r)][
                "best_psnr_iteration"
            ]
            + 1
            for requested_r in TUNE_RS
        ]
        if SELECTED_ITERATIONS is None:
            selected_iterations = int(
                round(
                    float(np.median(selected_iterations_from_tuning))
                    / max(1, EVAL_EVERY)
                )
                * max(1, EVAL_EVERY)
            )
            selected_iterations = max(1, min(selected_iterations, JOINT_ITERS))
        else:
            selected_iterations = int(SELECTED_ITERATIONS)

        selected_configuration = {
            "sample_index": int(sample_row["index"]),
            "selected_spec": asdict(selected_spec),
            "selected_iterations": selected_iterations,
            "selection_accelerations": list(TUNE_RS),
            "warning": (
                "Selected on one development slice using reference metrics; "
                "freeze before held-out evaluation."
            ),
        }
        (OUTPUT_DIR / "selected_configuration.json").write_text(
            json.dumps(selected_configuration, indent=2) + "\n"
        )
        print("selected trial     :", selected_trial_name)
        print("fixed stopping iter:", selected_iterations)
        display(pd.DataFrame([selected_configuration["selected_spec"]]))
        """
    ),
    markdown(
        r"""
        ## Convergence diagnosis

        A large gap between oracle best and final PSNR means the model reached a
        useful solution but the optimizer subsequently damaged it.  A poor best
        PSNR means the current parameterization or data are the limiting factor,
        not merely stopping time.
        """
    ),
    code(
        r"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
        colors = plt.cm.tab10(np.linspace(0, 1, len(optimization_specs)))

        plot_r = max(TUNE_RS)
        for color, spec in zip(colors, optimization_specs):
            result = optimization_results[(spec.name, plot_r)]
            history = pd.DataFrame(result["history"])
            axes[0].plot(
                history["iteration"],
                history["total"],
                label=spec.name,
                color=color,
            )
            axes[1].plot(
                history["iteration"],
                history["laps_psnr"],
                label=spec.name,
                color=color,
            )
            axes[2].plot(
                history["iteration"],
                history["laps_gain"],
                label=spec.name,
                color=color,
            )

        axes[0].set_yscale("log")
        axes[0].set_title(f"Training objective, R={plot_r}")
        axes[0].set_ylabel("loss")
        axes[1].set_title("Reference PSNR diagnostic")
        axes[1].set_ylabel("LAPS PSNR (dB)")
        axes[2].set_title("Scale drift")
        axes[2].set_ylabel("LAPS alignment gain")
        for axis in axes:
            axis.set_xlabel("iteration")
            axis.grid(alpha=0.25)
        axes[2].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        r"""
        ## Stage C — freeze the winner and sweep acceleration

        One architecture, optimizer, regularization choice, and stopping
        iteration is now fixed globally.  We do not choose a different
        configuration or stopping point for each R.
        """
    ),
    code(
        r"""
        sweep_results = {}
        sweep_rows = []
        frozen_spec = replace(selected_spec, name=f"selected_{selected_spec.name}")

        for requested_r in ACCELERATIONS:
            result = run_trial(
                frozen_spec,
                requested_r,
                iters=selected_iterations,
                schedule_horizon=JOINT_ITERS,
            )
            sweep_results[requested_r] = result
            for checkpoint in ("final", "best_objective", "best_psnr"):
                metrics = result["metrics"][checkpoint]
                sweep_rows.append(
                    {
                        "method": (
                            "Ours tuned (fixed stop)"
                            if checkpoint == "final"
                            else f"Ours diagnostic ({checkpoint})"
                        ),
                        "checkpoint": checkpoint,
                        "requested_r": requested_r,
                        "effective_r": measurements[requested_r][
                            "info"
                        ].effective_acceleration,
                        "runtime_seconds": result["runtime_seconds"],
                        "trainable_parameters": result["trainable_parameters"],
                        **metrics,
                    }
                )

        sweep_table = pd.DataFrame(sweep_rows)
        sweep_table.to_csv(OUTPUT_DIR / "ours_acceleration_sweep.csv", index=False)
        ours_final_table = sweep_table[sweep_table["checkpoint"] == "final"].copy()
        display(
            ours_final_table[
                [
                    "requested_r",
                    "effective_r",
                    "laps_psnr",
                    "laps_ssim",
                    "laps_gain",
                    "change_cosine",
                    "change_gain",
                    "mi_prior_delta",
                    "aligned_rmse",
                    "data_error_raw",
                ]
            ].round(4)
        )

        comparison_before_nerp = pd.concat(
            [
                baseline_table,
                ours_final_table.drop(columns=["checkpoint"], errors="ignore"),
            ],
            ignore_index=True,
            sort=False,
        )

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
        for method in (
            "Registered prior",
            "Zero-filled",
            "CG-SENSE",
            "Ours tuned (fixed stop)",
        ):
            part = comparison_before_nerp[
                comparison_before_nerp["method"] == method
            ].sort_values("effective_r")
            axes[0].plot(
                part["effective_r"], part["laps_psnr"], marker="o", label=method
            )
            axes[1].plot(
                part["effective_r"], part["laps_ssim"], marker="o", label=method
            )
            axes[2].plot(
                part["effective_r"], part["change_cosine"], marker="o", label=method
            )
        axes[0].set_ylabel("LAPS PSNR (dB)")
        axes[1].set_ylabel("LAPS SSIM")
        axes[2].set_ylabel("change cosine")
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
        ## Qualitative reconstruction, aligned error, learned change, and phase

        Displayed reconstructions and error maps use one least-squares gain to
        map each reconstruction into the reference magnitude scale.  The raw gain
        is printed in the title and remains a scale-stability diagnostic.
        """
    ),
    code(
        r"""
        for requested_r in [r for r in QUALITATIVE_RS if r in ACCELERATIONS]:
            ours_raw = sweep_results[requested_r]["recons"]["final"]
            methods = {
                "registered prior": prior,
                "zero-filled": baseline_recons[requested_r]["Zero-filled"],
                "CG-SENSE": baseline_recons[requested_r]["CG-SENSE"],
                "ours tuned": ours_raw,
            }
            fig, axes = plt.subplots(2, len(methods) + 1, figsize=(3.1 * (len(methods) + 1), 6))

            axes[0, 0].imshow(reference_np, cmap="gray", vmin=0, vmax=1)
            axes[0, 0].set_title("reference")
            axes[1, 0].imshow(
                np.zeros_like(reference_np), cmap="magma", vmin=0, vmax=1
            )
            axes[1, 0].set_title("aligned error")

            for column, (method, reconstruction) in enumerate(methods.items(), start=1):
                evaluated = reconstruction.detach().cpu() * support
                aligned, target, gain = align_to_reference_scale(evaluated)
                error = 5.0 * np.abs(aligned - target)
                values = laps_metrics(evaluated)
                axes[0, column].imshow(aligned, cmap="gray", vmin=0, vmax=1)
                axes[0, column].set_title(
                    f"{method}\nPSNR {values['laps_psnr']:.2f}, "
                    f"SSIM {values['laps_ssim']:.3f}\ngain {gain:.3f}",
                    fontsize=9,
                )
                axes[1, column].imshow(error, cmap="magma", vmin=0, vmax=1)
                axes[1, column].set_title("5x aligned magnitude error")
            for axis in axes.flat:
                axis.axis("off")
            fig.suptitle(
                f"requested R={requested_r}, "
                f"effective R={measurements[requested_r]['info'].effective_acceleration:.2f}",
                y=1.02,
            )
            fig.tight_layout()
            plt.show()

            ours_aligned, _, ours_gain = align_to_reference_scale(ours_raw * support)
            learned_change = ours_aligned - prior_np
            true_change = reference_np - prior_np
            delta = sweep_results[requested_r]["recons"]["final"]
            phase = torch.angle(delta).numpy()
            limit = float(np.quantile(np.abs(true_change), 0.995))

            fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
            axes[0].imshow(
                true_change, cmap="coolwarm", vmin=-limit, vmax=limit
            )
            axes[0].set_title("true magnitude change")
            axes[1].imshow(
                learned_change, cmap="coolwarm", vmin=-limit, vmax=limit
            )
            axes[1].set_title("reconstructed change")
            axes[2].imshow(
                learned_change - true_change,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            axes[2].set_title("change error")
            axes[3].imshow(
                phase, cmap="twilight", vmin=-math.pi, vmax=math.pi
            )
            axes[3].set_title("reconstructed phase")
            for axis in axes:
                axis.axis("off")
            fig.tight_layout()
            plt.show()
        """
    ),
    markdown(
        r"""
        ## Descriptive operating limits

        These thresholds are deliberately transparent and descriptive:

        - useful: at least 0.5 dB above both registered-prior and zero-filled,
          with positive change cosine;
        - competitive: matches or beats CG-SENSE.

        The largest contiguous R satisfying a condition is reported.  This is
        still one slice and one seed, not a universal acceleration limit.
        """
    ),
    code(
        r"""
        limit_rows = []
        for requested_r in ACCELERATIONS:
            ours_row = ours_final_table[
                ours_final_table["requested_r"] == requested_r
            ].iloc[0]
            baselines_r = baseline_table[
                baseline_table["requested_r"] == requested_r
            ].set_index("method")
            context_psnr = max(
                baselines_r.loc["Registered prior", "laps_psnr"],
                baselines_r.loc["Zero-filled", "laps_psnr"],
            )
            cg_psnr = baselines_r.loc["CG-SENSE", "laps_psnr"]
            useful = (
                ours_row["laps_psnr"] >= context_psnr + 0.5
                and ours_row["change_cosine"] > 0
            )
            competitive = ours_row["laps_psnr"] >= cg_psnr
            limit_rows.append(
                {
                    "requested_r": requested_r,
                    "effective_r": ours_row["effective_r"],
                    "ours_psnr": ours_row["laps_psnr"],
                    "gain_over_prior_or_zf_db": ours_row["laps_psnr"] - context_psnr,
                    "gain_over_cg_db": ours_row["laps_psnr"] - cg_psnr,
                    "change_cosine": ours_row["change_cosine"],
                    "useful": useful,
                    "competitive_with_cg": competitive,
                }
            )

        limit_table = pd.DataFrame(limit_rows)
        display(limit_table.round(4))

        def largest_contiguous_effective_r(frame, column):
            last = None
            for _, row in frame.sort_values("effective_r").iterrows():
                if not bool(row[column]):
                    break
                last = float(row["effective_r"])
            return last

        print(
            "useful contiguous Rmax     :",
            largest_contiguous_effective_r(limit_table, "useful"),
        )
        print(
            "CG-competitive contiguous Rmax:",
            largest_contiguous_effective_r(
                limit_table, "competitive_with_cg"
            ),
        )
        """
    ),
    markdown(
        r"""
        # Final reference only — LAPS-NeRP

        Our configuration is now frozen.  This final cell trains the release
        LAPS-NeRP implementation on the exact same sample, nested masks, coil
        maps, and common k-space scale.  NeRP therefore cannot influence tuning.

        The release output is the primary reference.  The scale-applied output is
        retained only as a diagnostic because both come from the same fit.
        """
    ),
    code(
        r"""
        nerp_results = {}
        nerp_rows = []

        if SMOKE:
            nerp_config = LapsNerpConfig(
                prior_stage=LapsNerpStageConfig(
                    max_iter=2,
                    lr=1e-4,
                    weight_decay=1e-4,
                    min_iterations=1,
                    patience=2,
                ),
                kspace_stage=LapsNerpStageConfig(
                    max_iter=2,
                    lr=1e-5,
                    weight_decay=0.0,
                    min_iterations=1,
                    patience=2,
                ),
            )
        else:
            nerp_config = LapsNerpConfig()

        fourier_generator = torch.Generator(device="cpu").manual_seed(BASE_SEED)
        nerp_fourier_matrix = (
            torch.randn(
                nerp_config.embedding_size,
                nerp_config.coordinate_size,
                generator=fourier_generator,
            )
            * nerp_config.embedding_scale
        )

        if RUN_LAPS_NERP:
            for requested_r in NERP_ACCELERATIONS:
                payload = {
                    "kind": "laps-nerp",
                    "sample_index": int(sample_row["index"]),
                    "requested_r": requested_r,
                    "config": asdict(nerp_config),
                    "common_scale": common_scale,
                    "mask_hash": cache_fingerprint(
                        {"mask": masks[requested_r].cpu().numpy().tobytes().hex()}
                    ),
                }
                path = cache_path("nerp", payload)
                if RESUME_CACHE and path.exists():
                    artifact = torch.load(
                        path, map_location="cpu", weights_only=False
                    )
                    print(f"loaded LAPS-NeRP R={requested_r:g}")
                else:
                    item = measurements[requested_r]
                    set_seed(
                        stable_seed(
                            "laps-nerp", int(sample_row["index"]), requested_r
                        )
                    )
                    started = time.perf_counter()
                    result = fit_laps_nerp(
                        prior,
                        item["operator"],
                        item["kspace"],
                        config=nerp_config,
                        device=DEVICE,
                        fourier_matrix=nerp_fourier_matrix,
                        verbose=False,
                    )
                    if DEVICE.type == "cuda":
                        torch.cuda.synchronize()
                    artifact = {
                        "runtime_seconds": time.perf_counter() - started,
                        "released": result.recon_released.detach().cpu(),
                        "scaled": result.recon_scaled.detach().cpu(),
                        "prior_history": result.prior_history,
                        "kspace_history": result.kspace_history,
                        "stage1_iterations": len(result.prior_history["loss"]),
                        "stage2_iterations": len(result.kspace_history["loss"]),
                    }
                    torch.save(artifact, path)
                    del result
                    cleanup_cuda()

                nerp_results[requested_r] = artifact
                for method, reconstruction in (
                    ("LAPS-NeRP (released)", artifact["released"]),
                    ("LAPS-NeRP (+scale diagnostic)", artifact["scaled"]),
                ):
                    nerp_rows.append(
                        {
                            "method": method,
                            "requested_r": requested_r,
                            "effective_r": measurements[requested_r][
                                "info"
                            ].effective_acceleration,
                            "runtime_seconds": artifact["runtime_seconds"],
                            "trainable_parameters": 1_839_618,
                            "stage1_iterations": artifact["stage1_iterations"],
                            "stage2_iterations": artifact["stage2_iterations"],
                            **full_metrics(reconstruction, requested_r),
                        }
                    )
        else:
            print("RUN_LAPS_NERP=False; reference fit skipped.")

        nerp_table = pd.DataFrame(nerp_rows)
        if not nerp_table.empty:
            nerp_table.to_csv(OUTPUT_DIR / "laps_nerp_reference.csv", index=False)
            display(
                nerp_table[
                    [
                        "method",
                        "requested_r",
                        "effective_r",
                        "laps_psnr",
                        "laps_ssim",
                        "laps_gain",
                        "change_cosine",
                        "mi_prior_delta",
                        "data_error_raw",
                        "runtime_seconds",
                    ]
                ].round(4)
            )
        """
    ),
    markdown(
        r"""
        ## Frozen ours versus NeRP reference

        This comparison remains a one-slice development result.  The next
        scientific step is to freeze the selected configuration and run it on
        held-out scans, including separate small-, large-, and no-change groups.
        """
    ),
    code(
        r"""
        if nerp_table.empty:
            print("Run the LAPS-NeRP cell above to create the final comparison.")
        else:
            final_comparison = pd.concat(
                [
                    baseline_table,
                    ours_final_table.drop(columns=["checkpoint"], errors="ignore"),
                    nerp_table,
                ],
                ignore_index=True,
                sort=False,
            )
            final_comparison.to_csv(
                OUTPUT_DIR / "final_development_comparison.csv", index=False
            )
            primary_methods = [
                "CG-SENSE",
                "Ours tuned (fixed stop)",
                "LAPS-NeRP (released)",
            ]

            display(
                final_comparison[
                    final_comparison["method"].isin(primary_methods)
                ][
                    [
                        "method",
                        "requested_r",
                        "effective_r",
                        "laps_psnr",
                        "laps_ssim",
                        "change_cosine",
                        "change_gain",
                        "mi_prior_delta",
                        "aligned_rmse",
                        "data_error_raw",
                        "runtime_seconds",
                        "trainable_parameters",
                    ]
                ].sort_values(["requested_r", "method"]).round(4)
            )

            fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
            for method in primary_methods:
                part = final_comparison[
                    final_comparison["method"] == method
                ].sort_values("effective_r")
                axes[0].plot(
                    part["effective_r"], part["laps_psnr"], marker="o", label=method
                )
                axes[1].plot(
                    part["effective_r"], part["laps_ssim"], marker="o", label=method
                )
                axes[2].plot(
                    part["effective_r"], part["change_cosine"], marker="o", label=method
                )
            axes[0].set_ylabel("LAPS PSNR (dB)")
            axes[1].set_ylabel("LAPS SSIM")
            axes[2].set_ylabel("change cosine")
            for axis in axes:
                axis.set_xlabel("effective acceleration")
                axis.grid(alpha=0.25)
            axes[2].legend(frameon=False, fontsize=8)
            fig.tight_layout()
            plt.show()

            for requested_r in [
                r
                for r in QUALITATIVE_RS
                if r in nerp_results and r in sweep_results
            ]:
                methods = {
                    "reference": reference,
                    "CG-SENSE": baseline_recons[requested_r]["CG-SENSE"],
                    "ours tuned": sweep_results[requested_r]["recons"]["final"],
                    "LAPS-NeRP": nerp_results[requested_r]["released"],
                }
                fig, axes = plt.subplots(2, len(methods), figsize=(3.2 * len(methods), 6))
                for column, (method, reconstruction) in enumerate(methods.items()):
                    evaluated = reconstruction.detach().cpu() * support
                    aligned, target, gain = align_to_reference_scale(evaluated)
                    error = 5.0 * np.abs(aligned - target)
                    values = laps_metrics(evaluated)
                    axes[0, column].imshow(aligned, cmap="gray", vmin=0, vmax=1)
                    axes[0, column].set_title(
                        f"{method}\nPSNR {values['laps_psnr']:.2f}, "
                        f"SSIM {values['laps_ssim']:.3f}\ngain {gain:.3f}",
                        fontsize=9,
                    )
                    axes[1, column].imshow(error, cmap="magma", vmin=0, vmax=1)
                    axes[1, column].set_title("5x aligned magnitude error")
                for axis in axes.flat:
                    axis.axis("off")
                fig.suptitle(
                    f"Frozen comparison: requested R={requested_r}, "
                    f"effective R={measurements[requested_r]['info'].effective_acceleration:.2f}",
                    y=1.02,
                )
                fig.tight_layout()
                plt.show()
        """
    ),
    markdown(
        r"""
        ## What to decide from this notebook

        1. If a larger magnitude network materially improves oracle-best and
           fixed-stop PSNR, capacity is a real bottleneck.
        2. If oracle-best is good but final is poor, stopping or learning rate is
           the bottleneck.
        3. If CG phase improves the same magnitude model, phase initialization is
           the bottleneck.
        4. If learned prior scale stabilizes gain across R, the prior/current
           normalization mismatch was contributing.
        5. If no variant improves high-R oracle-best quality, change the
           formulation before spending time on broad model comparisons.

        After choosing a formulation, lock the configuration JSON written by this
        notebook and evaluate it on different subjects.  Do not include scan 16
        in that held-out analysis.
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
