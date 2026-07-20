"""Build the locked held-out magnitude/phase evaluation notebook.

The generated notebook is deliberately separate from development/tuning.  A
normal run refuses to start unless the follow-up notebook has written a locked
configuration.  Smoke mode is the sole exception and exists only to exercise
the software path with tiny networks and two optimization steps.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUT = REPO / "notebooks" / "magnitude_phase_heldout_evaluation.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": textwrap.dedent(source).strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
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
        # Locked held-out evaluation: improved residual INR versus current-only INR

        This notebook evaluates the formulation selected in
        `magnitude_phase_formulation_followup.ipynb` without performing any further tuning.
        Its two neural methods have exactly the same trainable magnitude and
        phase branches:

        \[
        \widehat x_{\mathrm{ours}}(c)
          = \max\!\left(\alpha p_\theta(c)+\Delta m_\psi(c),0\right)
            e^{i\phi_\omega(c)},
        \qquad
        \widehat x_{\mathrm{current}}(c)
          = \max\!\left(g_\psi(c),0\right)e^{i\phi_\omega(c)}.
        \]

        Here \(\alpha\) is estimated only from retained k-space. The current-only
        magnitude branch is initialized from the zero-filled follow-up and never
        sees the prior. Both methods then receive the same full retrospective
        mask, phase initialization, optimizer schedule, and fixed iteration
        count.

        **Separation rule.** A normal run requires
        `reports/magnitude_phase_followup/locked_configuration.json`. This
        notebook never writes or modifies that file and never chooses a
        checkpoint or hyperparameter using a held-out reference.
        """
    ),
    markdown(
        r"""
        ## Study design fixed before reconstruction

        - Development scan 16 is prohibited.
        - The panel contains one metadata-selected middle 2-D T2 slice per SLAM
          subject: scans 3, 6, 9, 11, 15, and 18.
        - Replicate 0 is evaluated at requested \(R=3,6,9,13\).
        - Independent mask replicates 1 and 2 are evaluated only at \(R=9\).
        - Masks within a replicate are nested and are always subsets of the
          acquired mask.
        - Final fixed-stop reconstructions are reported; there is no oracle-best
          checkpoint in this notebook.

        SLAM contains only two `change_extent=2` scans, both from subject 5.
        Scan 16 was the development case, so scan 15 is the only remaining
        large-change scan. It is scan-held-out but **not subject-held-out**.
        Therefore the large-change stratum has \(n=1\) and cannot support a
        population-level claim.
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
        from dataclasses import asdict
        from pathlib import Path

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import torch
        from IPython.display import display
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity


        def find_repository(start: Path) -> Path:
            for candidate in (start.resolve(), *start.resolve().parents):
                if (candidate / "src" / "presinr").is_dir():
                    return candidate
            fallback = Path("/home/omarg/Patient-Specific-Residual-INR")
            if (fallback / "src" / "presinr").is_dir():
                return fallback
            raise RuntimeError("Could not locate the Patient-Specific-Residual-INR repository")


        REPO = find_repository(Path.cwd())
        if str(REPO / "src") not in sys.path:
            sys.path.insert(0, str(REPO / "src"))

        from presinr.baselines.laps_nerp import (
            CenterPaddedSense,
            conjugate_gradient_sense,
        )
        from presinr.calibration import prior_scale_from_kspace, real_least_squares_scale
        from presinr.data.slam import SlamTestSlices
        from presinr.experiments.magnitude_phase import (
            MagnitudePhaseTrainConfig,
            build_scalar_inr,
            relative_kspace_error,
            train_magnitude_phase,
            zero_last_linear_,
        )
        from presinr.metrics import acquisition_calibrated_longitudinal_metrics
        from presinr.models import CurrentMagnitudePhaseINR, PriorMagnitudePhaseINR
        from presinr.models.inr import make_coord_grid
        from presinr.recon import PhaseFitConfig, PriorFitConfig, fit_phase_inr, fit_prior
        from presinr.sampling import laps_retrospective_1d_mask
        from presinr.utils import center_pad_to, get_device, set_seed, to_numpy

        DEVICE = get_device()
        print("repository:", REPO)
        print("python    :", sys.executable)
        print("torch     :", torch.__version__)
        print("device    :", DEVICE)
        if DEVICE.type == "cuda":
            print("GPU       :", torch.cuda.get_device_name(DEVICE))
        """
    ),
    markdown(
        r"""
        ## Locked configuration and smoke exception

        Full mode validates the lock file before loading any image. The small
        normalization adapter below accepts both the explicit follow-up schema
        and the earlier `selected_spec` field names, but the resulting canonical
        configuration is printed and hashed into every cache key.

        Set `PRESINR_HELDOUT_SMOKE=1` for a two-iteration software test. Smoke
        output goes to `/tmp` and is visibly labelled; it is not scientific data.
        """
    ),
    code(
        r"""
        SMOKE = os.environ.get("PRESINR_HELDOUT_SMOKE", "0") == "1"
        BASE_SEED = 20260720
        CACHE_VERSION = "magnitude-phase-heldout-v1"
        LOCK_PATH = REPO / "reports" / "magnitude_phase_followup" / "locked_configuration.json"


        def nested_get(mapping, *paths, default=None):
            for path in paths:
                value = mapping
                found = True
                for key in path.split("."):
                    if not isinstance(value, dict) or key not in value:
                        found = False
                        break
                    value = value[key]
                if found and value is not None:
                    return value
            return default


        def canonicalize_lock(raw):
            selected = raw.get("selected_spec", {}) if isinstance(raw, dict) else {}
            model = raw.get("model", {}) if isinstance(raw, dict) else {}
            initialization = raw.get("initialization", {}) if isinstance(raw, dict) else {}
            training = raw.get("training", {}) if isinstance(raw, dict) else {}
            sampling = raw.get("sampling", {}) if isinstance(raw, dict) else {}

            magnitude_kind = nested_get(
                raw,
                "model.magnitude_kind",
                "model.delta_kind",
                "selected_spec.delta_kind",
                default="siren",
            )
            magnitude_kwargs = nested_get(raw, "model.magnitude_kwargs", default=None)
            if magnitude_kwargs is None:
                magnitude_kwargs = {
                    "hidden_features": int(nested_get(
                        raw, "model.delta_width", "selected_spec.delta_width", default=128
                    )),
                    "hidden_layers": int(nested_get(
                        raw, "model.delta_layers", "selected_spec.delta_layers", default=4
                    )),
                }
                mapping_size = nested_get(
                    raw, "model.mapping_size", "selected_spec.mapping_size", default=None
                )
                sigma = nested_get(raw, "model.sigma", "selected_spec.sigma", default=None)
                if mapping_size is not None:
                    magnitude_kwargs["mapping_size"] = int(mapping_size)
                if sigma is not None:
                    magnitude_kwargs["sigma"] = float(sigma)

            phase_kwargs = nested_get(raw, "model.phase_kwargs", default=None)
            if phase_kwargs is None:
                phase_kwargs = {
                    "hidden_features": int(nested_get(
                        raw, "model.phase_width", "selected_spec.phase_width", default=64
                    )),
                    "hidden_layers": int(nested_get(
                        raw, "model.phase_layers", "selected_spec.phase_layers", default=3
                    )),
                }

            canonical = {
                "schema_version": str(raw.get("schema_version", "presinr-magnitude-phase-v1")),
                "source_lock": str(LOCK_PATH),
                "magnitude_kind": str(magnitude_kind),
                "magnitude_kwargs": dict(magnitude_kwargs),
                "phase_kind": str(nested_get(raw, "model.phase_kind", default="siren")),
                "phase_kwargs": dict(phase_kwargs),
                "prior_kind": str(nested_get(raw, "model.prior_kind", default="siren")),
                "prior_kwargs": dict(nested_get(
                    raw,
                    "model.prior_kwargs",
                    default={"hidden_features": 256, "hidden_layers": 4},
                )),
                "magnitude_residual_bound": nested_get(
                    raw,
                    "model.magnitude_residual_bound",
                    "selected_spec.magnitude_bound",
                    default=None,
                ),
                "prior_scale_mode": str(nested_get(
                    raw, "model.prior_scale_mode", default="acquisition_fixed"
                )),
                "prior_iterations": int(nested_get(
                    raw, "initialization.prior_iterations", default=3000
                )),
                "prior_lr": float(nested_get(
                    raw, "initialization.prior_lr", default=1e-4
                )),
                "phase_iterations": int(nested_get(
                    raw, "initialization.phase_iterations", default=1000
                )),
                "phase_init_lr": float(nested_get(
                    raw, "initialization.phase_lr", default=1e-4
                )),
                "current_magnitude_iterations": int(nested_get(
                    raw, "initialization.current_magnitude_iterations", default=1000
                )),
                "current_magnitude_lr": float(nested_get(
                    raw, "initialization.current_magnitude_lr", default=1e-4
                )),
                "phase_source": str(nested_get(
                    raw,
                    "initialization.phase_source",
                    "selected_spec.phase_source",
                    default="zf",
                )),
                "iterations": int(nested_get(
                    raw, "training.iterations", "selected_iterations", default=1200
                )),
                "magnitude_lr": float(nested_get(
                    raw,
                    "training.magnitude_lr",
                    "selected_spec.delta_lr",
                    default=1e-4,
                )),
                "phase_lr": float(nested_get(
                    raw, "training.phase_lr", "selected_spec.phase_lr", default=3e-5
                )),
                "prior_scale_lr": float(nested_get(
                    raw,
                    "training.prior_scale_lr",
                    "selected_spec.scale_lr",
                    default=1e-3,
                )),
                "lambda_delta_l1": float(nested_get(
                    raw,
                    "training.lambda_delta_l1",
                    "selected_spec.lambda_change",
                    default=1e-3,
                )),
                "lambda_delta_tv": float(nested_get(
                    raw,
                    "training.lambda_delta_tv",
                    "selected_spec.lambda_change_tv",
                    default=0.0,
                )),
                "lambda_phase_tv": float(nested_get(
                    raw,
                    "training.lambda_phase_tv",
                    "selected_spec.lambda_phase_tv",
                    default=1e-5,
                )),
                "min_lr_ratio": float(nested_get(
                    raw, "training.min_lr_ratio", default=0.05
                )),
                "grad_clip_norm": nested_get(
                    raw,
                    "training.grad_clip_norm",
                    "selected_spec.grad_clip",
                    default=1.0,
                ),
                "eval_every": int(nested_get(
                    raw, "training.eval_every", default=100
                )),
                "phase_encode_dim": int(nested_get(
                    raw, "sampling.phase_encode_dim", default=1
                )),
                "vd_factor": float(nested_get(
                    raw, "sampling.vd_factor", default=0.8
                )),
                "n_candidates": int(nested_get(
                    raw, "sampling.n_candidates", default=100
                )),
            }
            if canonical["magnitude_residual_bound"] is not None:
                canonical["magnitude_residual_bound"] = float(
                    canonical["magnitude_residual_bound"]
                )
            if canonical["grad_clip_norm"] is not None:
                canonical["grad_clip_norm"] = float(canonical["grad_clip_norm"])
            return canonical


        if SMOKE:
            raw_lock = {
                "schema_version": 1,
                "model": {
                    "magnitude_kind": "siren",
                    "magnitude_kwargs": {"hidden_features": 8, "hidden_layers": 1},
                    "phase_kind": "siren",
                    "phase_kwargs": {"hidden_features": 8, "hidden_layers": 1},
                    "prior_kind": "siren",
                    "prior_kwargs": {"hidden_features": 8, "hidden_layers": 1},
                },
                "initialization": {
                    "prior_iterations": 2,
                    "phase_iterations": 2,
                    "current_magnitude_iterations": 2,
                },
                "training": {"iterations": 2, "eval_every": 1},
            }
            print("SMOKE MODE: using an internal two-iteration configuration")
        else:
            if not LOCK_PATH.exists():
                raise FileNotFoundError(
                    "Held-out evaluation is locked. Run the follow-up tuning notebook "
                    f"and create {LOCK_PATH} before executing this notebook."
                )
            raw_lock = json.loads(LOCK_PATH.read_text())
            if not isinstance(raw_lock, dict):
                raise ValueError("locked_configuration.json must contain a JSON object")
            if raw_lock.get("status") != "locked-development-selection":
                raise ValueError(
                    "locked_configuration.json is not marked "
                    "status='locked-development-selection'"
                )
            if nested_get(raw_lock, "model.prior_scale_mode") not in {
                "acquisition_fixed", "acquisition_learned"
            }:
                raise ValueError(
                    "model.prior_scale_mode must be acquisition_fixed or "
                    "acquisition_learned"
                )
            if nested_get(raw_lock, "model.zero_initialize_residual") is not True:
                raise ValueError("Held-out evaluation requires a zero-initialized residual")
            excluded = set(nested_get(
                raw_lock, "evaluation.excluded_scan_indices", default=[]
            ))
            if 16 not in excluded:
                raise ValueError("The locked evaluation protocol must explicitly exclude scan 16")

        LOCKED = canonicalize_lock(raw_lock)
        # Bind caches and reports to the complete lock, including selection and
        # evaluation provenance rather than only the normalized training fields.
        LOCK_HASH = hashlib.sha256(
            json.dumps(raw_lock, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]
        print("lock hash:", LOCK_HASH)
        display(pd.Series(LOCKED, name="locked value").to_frame())
        """
    ),
]

cells.extend(
    [
        markdown(
            r"""
            ## Fixed cases, accelerations, and mask replicates

            The row identities below were fixed from the SLAM manifest. Assertions
            make accidental replacement, inclusion of scan 16, or a changed
            `change_extent` fail loudly. Full mode evaluates the primary curve
            with replicate 0 and adds two independent masks at the stress point
            \(R=9\). Smoke mode uses scan 18 at \(R=6\) only.
            """
        ),
        code(
            r"""
            PANEL = pd.DataFrame(
                [
                    {"scan_index": 3, "slice_index": 23, "subject": 1, "change_extent": 1},
                    {"scan_index": 6, "slice_index": 30, "subject": 2, "change_extent": 0},
                    {"scan_index": 9, "slice_index": 22, "subject": 3, "change_extent": 0},
                    {"scan_index": 11, "slice_index": 24, "subject": 4, "change_extent": 1},
                    {"scan_index": 15, "slice_index": 31, "subject": 5, "change_extent": 2},
                    {"scan_index": 18, "slice_index": 22, "subject": 6, "change_extent": 1},
                ]
            )
            PROHIBITED_SCANS = {16}
            PRIMARY_ACCELERATIONS = (3, 6, 9, 13)
            MASK_STRESS_R = 9
            MASK_REPLICATES = (0, 1, 2)

            if SMOKE:
                PANEL = PANEL[PANEL["scan_index"] == 18].reset_index(drop=True)
                PRIMARY_ACCELERATIONS = (6,)
                MASK_STRESS_R = 6
                MASK_REPLICATES = (0,)

            SETTINGS = [
                {"requested_r": requested_r, "mask_replicate": 0}
                for requested_r in PRIMARY_ACCELERATIONS
            ]
            SETTINGS += [
                {"requested_r": MASK_STRESS_R, "mask_replicate": replicate}
                for replicate in MASK_REPLICATES
                if replicate != 0
            ]

            OUTPUT_DIR = (
                Path("/tmp/presinr-magnitude-phase-heldout-smoke")
                if SMOKE
                else REPO / "reports" / "magnitude_phase_heldout"
            )
            CACHE_DIR = OUTPUT_DIR / "cache"
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

            assert not (set(PANEL.scan_index) & PROHIBITED_SCANS)
            assert PANEL.subject.is_unique
            assert set(PANEL.change_extent) == ({1} if SMOKE else {0, 1, 2})
            print("output  :", OUTPUT_DIR)
            print("settings:", SETTINGS)
            display(PANEL)
            """
        ),
        markdown(
            r"""
            ## Determinism, cache, and unit helpers

            Every artifact key contains the locked configuration, case, mask
            fingerprint, and software cache version. Initial neural weights are
            kept identical across mask replicates for a case so the replicate
            stress test changes sampling rather than initialization.
            """
        ),
        code(
            r"""
            def stable_seed(*parts, base=BASE_SEED):
                payload = "|".join(map(str, (base,) + parts)).encode("utf-8")
                return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


            def json_ready(value):
                if isinstance(value, dict):
                    return {str(key): json_ready(item) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [json_ready(item) for item in value]
                if isinstance(value, np.generic):
                    return value.item()
                if isinstance(value, Path):
                    return str(value)
                return value


            def fingerprint(payload):
                body = {"cache_version": CACHE_VERSION, "lock_hash": LOCK_HASH, **payload}
                encoded = json.dumps(json_ready(body), sort_keys=True, default=str).encode("utf-8")
                return hashlib.sha256(encoded).hexdigest()[:16]


            def tensor_hash(tensor):
                array = tensor.detach().cpu().contiguous().numpy()
                return hashlib.sha256(array.tobytes()).hexdigest()[:16]


            def cache_path(kind, payload):
                return CACHE_DIR / f"{kind}_{fingerprint(payload)}.pt"


            def clone_state(module):
                return {
                    name: value.detach().cpu().clone()
                    for name, value in module.state_dict().items()
                }


            def count_parameters(module, trainable_only=False):
                return sum(
                    parameter.numel()
                    for parameter in module.parameters()
                    if not trainable_only or parameter.requires_grad
                )


            def quantile_scale(tensor, q=0.999):
                values = tensor.detach().abs().reshape(-1).float()
                return float(torch.quantile(values, q)) + 1e-8


            def cleanup_cuda():
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


            def synchronize():
                if DEVICE.type == "cuda":
                    torch.cuda.synchronize()
            """
        ),
        markdown(
            r"""
            ## Validate the manifest panel

            Selection uses only scan identity, slice identity, acquisition type,
            and the dataset's scan-level change category. The reconstruction
            reference is not loaded to rank or replace a case.
            """
        ),
        code(
            r"""
            dataset = SlamTestSlices(
                data_dir=REPO / "data", middle_only=False, normalize=False
            )
            manifest = dataset.df.copy().reset_index(drop=True)
            manifest["dataset_position"] = np.arange(len(manifest))

            selected_rows = []
            for expected in PANEL.to_dict("records"):
                match = manifest[
                    (manifest["scan_index"] == expected["scan_index"])
                    & (manifest["slice_index"] == expected["slice_index"])
                ]
                if len(match) != 1:
                    raise RuntimeError(
                        f"Expected exactly one SLAM row for {expected}; found {len(match)}"
                    )
                row = match.iloc[0]
                checks = {
                    "subj_index": expected["subject"],
                    "change_extent": expected["change_extent"],
                    "is_middle_slice": True,
                }
                for key, value in checks.items():
                    if row[key] != value:
                        raise RuntimeError(
                            f"Panel metadata changed for scan {expected['scan_index']}: "
                            f"{key}={row[key]!r}, expected {value!r}"
                        )
                if not str(row["scan_type"]).endswith("T2_2D"):
                    raise RuntimeError("Held-out panel must remain restricted to 2-D T2 scans")
                selected_rows.append(row)

            selected_manifest = pd.DataFrame(selected_rows).reset_index(drop=True)
            if 16 in set(selected_manifest.scan_index):
                raise RuntimeError("Development scan 16 must never enter held-out evaluation")

            columns = [
                "dataset_position", "index", "scan_index", "slice_index",
                "subj_index", "change_extent", "scan_plane", "scan_type",
                "AccelNumDim", "Nc", "Kx", "Ky",
            ]
            display(selected_manifest[columns])
            """
        ),
    ]
)

cells.extend(
    [
        markdown(
            r"""
            ## Reproducibility record and interpretation guardrails

            The protocol record binds the output tables to the exact lock hash,
            cases, masks, and smoke/full status. Interpret the matched comparison
            using paired rows:

            - a residual advantage in acquisition-unit NRMSE and ROI/change
              metrics supports useful patient-prior information beyond INR bias;
            - a PSNR-only advantage accompanied by worse change gain or
              false-change error suggests prior copying rather than faithful
              longitudinal reconstruction;
            - strong mask-seed variance at \(R=9\) means the apparent acceleration
              limit is sampling-pattern dependent;
            - `change_extent=0` should be judged primarily by false-change error;
            - the single extent-2 result is descriptive and shares subject 5 with
              development scan 16.

            This notebook reconstructs one predetermined 2-D slice per scan. It
            does not establish full-volume or subject-population performance.
            """
        ),
        code(
            r"""
            protocol = {
                "schema_version": "presinr-heldout-v1",
                "smoke": SMOKE,
                "cache_version": CACHE_VERSION,
                "lock_path": str(LOCK_PATH),
                "lock_hash": LOCK_HASH,
                "locked_configuration": LOCKED,
                "prohibited_scan_indices": sorted(PROHIBITED_SCANS),
                "cases": PANEL.to_dict("records"),
                "settings": SETTINGS,
                "primary_accelerations": list(PRIMARY_ACCELERATIONS),
                "mask_stress_r": MASK_STRESS_R,
                "mask_replicates": list(MASK_REPLICATES),
                "selection_uses_reference": False,
                "checkpoint": "fixed-stop final full-mask iterate",
                "roi_definition": (
                    "top decile of absolute acquisition-calibrated true change "
                    "inside method-independent foreground; evaluation only"
                ),
                "limitations": [
                    "scan 16 excluded because it was used for development",
                    "scan 15 is the only remaining extent-2 scan and shares subject 5 with scan 16",
                    "extent-2 stratum n=1",
                    "one predetermined middle 2-D slice per scan",
                    "mask replicates are not independent subjects",
                ],
                "outputs": {
                    "per_case": "per_case_metrics.csv",
                    "measurement_manifest": "measurement_manifest.csv",
                    "aggregate_primary": "aggregate_primary_curve.csv",
                    "aggregate_by_extent": "aggregate_by_change_extent.csv",
                    "mask_sensitivity": "mask_sensitivity_per_case.csv",
                    "paired_rows": "paired_neural_differences.csv",
                    "paired_summary": "paired_primary_summary.csv",
                    "runtime": "runtime_summary.csv",
                },
            }
            (OUTPUT_DIR / "protocol.json").write_text(
                json.dumps(json_ready(protocol), indent=2, sort_keys=True) + "\n"
            )

            print("Held-out evaluation complete.")
            print("Results:", OUTPUT_DIR)
            print(
                "Large-change limitation: scan 15 is scan-held-out but shares "
                "subject 5 with development scan 16; extent-2 n=1."
            )
            """
        ),
        markdown(
            r"""
            ## Output files

            - `per_case_metrics.csv`: every method/case/acceleration/mask row;
            - `paired_neural_differences.csv`: exact matched residual-minus-current
              differences;
            - `aggregate_primary_curve.csv`: replicate-0 acceleration summary;
            - `aggregate_by_change_extent.csv`: descriptive stratified summary;
            - `mask_sensitivity_per_case.csv`: within-case \(R=9\) seed variation;
            - `runtime_summary.csv`: offline prior, online initialization, joint,
              and total runtime components;
            - `measurement_manifest.csv`: effective acceleration, retained/ACS
              lines, normalization, and mask hashes;
            - `protocol.json`: immutable interpretation and provenance record.
            """
        ),
    ]
)


cells.extend(
    [
        markdown(
            r"""
            ## Aggregate without treating mask seeds as patients

            The primary acceleration curve uses replicate 0 only. Case means and
            standard deviations are reported overall and by the scan-level
            `change_extent` label. The independent \(R=9\) masks are first
            summarized within each case; they are not counted as additional
            subjects. Paired method differences retain exact case/mask matching.
            """
        ),
        code(
            r"""
            REPORT_METRICS = [
                "laps_psnr", "laps_ssim", "acq_mae", "acq_rmse", "acq_nrmse",
                "roi_mae", "roi_rmse", "roi_nrmse", "change_cosine",
                "change_gain", "roi_change_cosine", "roi_change_gain",
                "false_change_l1", "false_change_rms", "mi_prior_delta",
                "data_error", "online_seconds", "joint_fit_seconds",
            ]


            def aggregate_frame(frame, group_columns):
                grouped = frame.groupby(group_columns, dropna=False)
                counts = grouped.size().rename("n_cases")
                statistics = grouped[REPORT_METRICS].agg(["mean", "std"])
                statistics.columns = [
                    f"{metric}_{statistic}"
                    for metric, statistic in statistics.columns
                ]
                return counts.to_frame().join(statistics).reset_index()


            primary = per_case[per_case["mask_replicate"] == 0].copy()
            aggregate_overall = aggregate_frame(primary, ["method", "requested_r"])
            aggregate_by_extent = aggregate_frame(
                primary, ["method", "change_extent", "requested_r"]
            )
            aggregate_overall.to_csv(
                OUTPUT_DIR / "aggregate_primary_curve.csv", index=False
            )
            aggregate_by_extent.to_csv(
                OUTPUT_DIR / "aggregate_by_change_extent.csv", index=False
            )

            r9_replicates = per_case[
                (per_case["requested_r"] == MASK_STRESS_R)
                & per_case["method"].isin(
                    ["Improved residual INR", "Current-only INR (matched)"]
                )
            ].copy()
            mask_sensitivity_per_case = (
                r9_replicates.groupby(
                    [
                        "method", "case_id", "scan_index", "subject",
                        "change_extent", "requested_r",
                    ],
                    as_index=False,
                )[REPORT_METRICS]
                .agg(["mean", "std"])
                .reset_index()
            )
            mask_sensitivity_per_case.columns = [
                "_".join(str(part) for part in column if str(part))
                if isinstance(column, tuple)
                else str(column)
                for column in mask_sensitivity_per_case.columns
            ]
            mask_sensitivity_per_case.to_csv(
                OUTPUT_DIR / "mask_sensitivity_per_case.csv", index=False
            )

            pair_keys = [
                "case_id", "scan_index", "slice_index", "subject",
                "change_extent", "requested_r", "effective_r", "mask_replicate",
            ]
            paired = neural.pivot(
                index=pair_keys,
                columns="method",
                values=REPORT_METRICS,
            )
            paired.columns = [
                f"{metric}__{method}" for metric, method in paired.columns
            ]
            paired = paired.reset_index()
            for metric in REPORT_METRICS:
                ours = f"{metric}__Improved residual INR"
                current = f"{metric}__Current-only INR (matched)"
                paired[f"{metric}__ours_minus_current"] = paired[ours] - paired[current]
            paired.to_csv(OUTPUT_DIR / "paired_neural_differences.csv", index=False)

            runtime_summary = (
                neural.groupby("method")
                [[
                    "prior_fit_seconds", "phase_init_seconds",
                    "magnitude_init_seconds", "joint_fit_seconds",
                    "online_seconds", "offline_seconds",
                ]]
                .agg(["mean", "std", "median"])
            )
            runtime_summary.columns = [
                f"{component}_{statistic}"
                for component, statistic in runtime_summary.columns
            ]
            runtime_summary = runtime_summary.reset_index()
            runtime_summary.to_csv(OUTPUT_DIR / "runtime_summary.csv", index=False)

            display(
                aggregate_overall[
                    [
                        "method", "requested_r", "n_cases",
                        "laps_psnr_mean", "laps_psnr_std",
                        "laps_ssim_mean", "acq_nrmse_mean",
                        "change_cosine_mean", "false_change_rms_mean",
                    ]
                ].round(4)
            )
            display(runtime_summary.round(3))
            """
        ),
        markdown(
            r"""
            ## Paired method differences

            Positive PSNR/SSIM differences favor the improved residual model;
            negative NRMSE, ROI error, false-change, and data-error differences
            favor it. Rows remain visible rather than being reduced to one score,
            particularly because the large-change group contains one scan.
            """
        ),
        code(
            r"""
            paired_columns = [
                "case_id", "change_extent", "requested_r", "mask_replicate",
                "laps_psnr__ours_minus_current",
                "laps_ssim__ours_minus_current",
                "acq_nrmse__ours_minus_current",
                "roi_rmse__ours_minus_current",
                "change_cosine__ours_minus_current",
                "false_change_rms__ours_minus_current",
                "data_error__ours_minus_current",
            ]
            display(paired[paired_columns].round(4))

            primary_paired = paired[paired["mask_replicate"] == 0]
            paired_summary = (
                primary_paired.groupby("requested_r")
                [[column for column in paired_columns if column.endswith("ours_minus_current")]]
                .agg(["mean", "std"])
            )
            paired_summary.columns = [
                f"{metric}_{statistic}"
                for metric, statistic in paired_summary.columns
            ]
            paired_summary = paired_summary.reset_index()
            paired_summary.to_csv(
                OUTPUT_DIR / "paired_primary_summary.csv", index=False
            )
            display(paired_summary.round(4))
            """
        ),
        markdown(
            r"""
            ## Acceleration curves

            Error bars show between-case standard deviation for the fixed
            replicate-0 panel, not uncertainty over a larger patient population.
            """
        ),
        code(
            r"""
            curve_methods = [
                "Improved residual INR",
                "Current-only INR (matched)",
                "CG-SENSE",
            ]
            curve = aggregate_overall[
                aggregate_overall["method"].isin(curve_methods)
            ]
            fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))
            panels = [
                ("laps_psnr", "LAPS PSNR (dB)"),
                ("laps_ssim", "LAPS SSIM"),
                ("acq_nrmse", "Acquisition-unit NRMSE"),
                ("change_cosine", "Change cosine"),
            ]
            for method in curve_methods:
                part = curve[curve["method"] == method].sort_values("requested_r")
                for axis, (metric, ylabel) in zip(axes, panels):
                    axis.errorbar(
                        part["requested_r"],
                        part[f"{metric}_mean"],
                        yerr=part[f"{metric}_std"].fillna(0.0),
                        marker="o",
                        capsize=3,
                        label=method,
                    )
                    axis.set_xlabel("requested acceleration")
                    axis.set_ylabel(ylabel)
                    axis.grid(alpha=0.25)
            axes[-1].legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "primary_acceleration_curves.png", dpi=160)
            plt.show()
            """
        ),
        markdown(
            r"""
            ## Qualitative acquisition-unit reconstructions and error magnitude

            Images in a row share display limits. The error row is the literal
            unscaled magnitude error \(|\,|\widehat x|-|x_{ref}|\,|\) in calibrated
            acquisition units—not a clipped `5x` visualization. The signed-change
            row uses the same calibrated prior for every method.
            """
        ),
        code(
            r"""
            for (case_id, requested_r, replicate), item in qualitative.items():
                maps = item["maps"]
                exemplar = maps["Improved residual INR"]
                display_methods = [
                    "reference",
                    "calibrated prior",
                    "CG-SENSE",
                    "Current-only INR (matched)",
                    "Improved residual INR",
                ]
                display_maps = {
                    "reference": {
                        "reconstruction": exemplar["reference_acq"],
                        "error_magnitude": np.zeros_like(exemplar["reference_acq"]),
                        "reconstructed_change": exemplar["true_change"],
                    },
                    "calibrated prior": {
                        "reconstruction": exemplar["prior_acq"],
                        "error_magnitude": np.abs(
                            exemplar["prior_acq"] - exemplar["reference_acq"]
                        ),
                        "reconstructed_change": np.zeros_like(exemplar["true_change"]),
                    },
                    "CG-SENSE": maps["CG-SENSE"],
                    "Current-only INR (matched)": maps[
                        "Current-only INR (matched)"
                    ],
                    "Improved residual INR": maps["Improved residual INR"],
                }
                image_limit = float(np.quantile(exemplar["reference_acq"], 0.995))
                error_limit = max(
                    float(np.quantile(exemplar["error_magnitude"], 0.995)), 1e-8
                )
                change_limit = max(
                    float(np.quantile(np.abs(exemplar["true_change"]), 0.995)), 1e-8
                )

                fig, axes = plt.subplots(
                    3, len(display_methods), figsize=(3.15 * len(display_methods), 9.0)
                )
                for column, method in enumerate(display_methods):
                    current = display_maps[method]
                    axes[0, column].imshow(
                        current["reconstruction"], cmap="gray", vmin=0, vmax=image_limit
                    )
                    axes[0, column].set_title(method, fontsize=9)
                    axes[1, column].imshow(
                        current["error_magnitude"],
                        cmap="magma",
                        vmin=0,
                        vmax=error_limit,
                    )
                    axes[2, column].imshow(
                        current["reconstructed_change"],
                        cmap="coolwarm",
                        vmin=-change_limit,
                        vmax=change_limit,
                    )
                axes[0, 0].set_ylabel("magnitude")
                axes[1, 0].set_ylabel("absolute error")
                axes[2, 0].set_ylabel("signed change")
                for axis in axes.flat:
                    axis.set_xticks([])
                    axis.set_yticks([])
                fig.suptitle(
                    f"{case_id}, change_extent={int(item['metadata']['change_extent'])}, "
                    f"requested R={requested_r}, effective R={item['effective_r']:.2f}",
                    y=1.01,
                )
                fig.tight_layout()
                fig.savefig(
                    OUTPUT_DIR / f"qualitative_{case_id}_R{requested_r}_rep{replicate}.png",
                    dpi=160,
                    bbox_inches="tight",
                )
                plt.show()
            """
        ),
        markdown(
            r"""
            ## Learned residual, change error, phase, and evaluation ROI

            `learned delta` is the network's actual \(\Delta m_\psi\), distinct
            from reconstructed magnitude minus the DICOM prior. This distinction
            exposes any scale or nonnegativity effect hidden by a derived change
            panel.
            """
        ),
        code(
            r"""
            for (case_id, requested_r, replicate), item in qualitative.items():
                ours = item["maps"]["Improved residual INR"]
                delta = to_numpy(item["delta"])
                phase = to_numpy(item["phase"])
                true_change = ours["true_change"]
                change_error = ours["reconstructed_change"] - true_change
                limit = max(float(np.quantile(np.abs(true_change), 0.995)), 1e-8)
                error_limit = max(
                    float(np.quantile(np.abs(change_error), 0.995)), 1e-8
                )
                fig, axes = plt.subplots(1, 6, figsize=(18, 3.2))
                panels = [
                    (true_change, "true change", "coolwarm", -limit, limit),
                    (
                        ours["reconstructed_change"],
                        "reconstructed change",
                        "coolwarm",
                        -limit,
                        limit,
                    ),
                    (delta, "learned delta", "coolwarm", -limit, limit),
                    (
                        np.abs(change_error),
                        "change error magnitude",
                        "magma",
                        0,
                        error_limit,
                    ),
                    (phase, "reconstructed phase", "twilight", -math.pi, math.pi),
                    (
                        ours["change_roi"].astype(float),
                        "evaluation-only ROI",
                        "gray",
                        0,
                        1,
                    ),
                ]
                for axis, (image, title, cmap, low, high) in zip(axes, panels):
                    axis.imshow(image, cmap=cmap, vmin=low, vmax=high)
                    axis.set_title(title, fontsize=9)
                    axis.axis("off")
                fig.suptitle(f"{case_id}, requested R={requested_r}", y=1.02)
                fig.tight_layout()
                fig.savefig(
                    OUTPUT_DIR / f"diagnostics_{case_id}_R{requested_r}_rep{replicate}.png",
                    dpi=160,
                    bbox_inches="tight",
                )
                plt.show()
            """
        ),
    ]
)

cells.extend(
    [
        markdown(
            r"""
            ## Execute the frozen held-out protocol

            This is the expensive cell. It is resumable through content-addressed
            artifacts, but every reported neural row is a final full-mask fit at
            the locked iteration count. The calibrated-prior, zero-filled, and
            CG-SENSE rows provide non-neural context and do not participate in
            the main matched-branch claim.
            """
        ),
        code(
            r"""
            metric_rows = []
            measurement_rows = []
            qualitative = {}

            for case_number, case_row in selected_manifest.iterrows():
                case = load_case(case_row)
                print(
                    f"[{case_number + 1}/{len(selected_manifest)}] {case['case_id']} "
                    f"subject={int(case['row']['subj_index'])} "
                    f"change_extent={int(case['row']['change_extent'])}"
                )
                prior_network, prior_artifact = get_prior_artifact(case)
                bank_by_replicate = {
                    int(replicate): make_mask_bank(case, int(replicate))
                    for replicate in sorted({item["mask_replicate"] for item in SETTINGS})
                }

                for setting in SETTINGS:
                    requested_r = int(setting["requested_r"])
                    replicate = int(setting["mask_replicate"])
                    measurement = make_measurement(
                        case,
                        requested_r,
                        replicate,
                        bank_by_replicate[replicate],
                    )
                    measurement_rows.append(
                        {
                            "case_id": case["case_id"],
                            "scan_index": int(case["row"]["scan_index"]),
                            "slice_index": int(case["row"]["slice_index"]),
                            "subject": int(case["row"]["subj_index"]),
                            "change_extent": int(case["row"]["change_extent"]),
                            "requested_r": requested_r,
                            "effective_r": measurement["info"].effective_acceleration,
                            "mask_replicate": replicate,
                            "retained_lines": measurement["info"].output_lines,
                            "center_lines": measurement["info"].center_lines,
                            "input_lines": measurement["info"].input_lines,
                            "mask_hash": measurement["mask_hash"],
                            "common_kspace_scale": measurement["common_scale"],
                            "reference_calibration_method": measurement[
                                "reference_calibration_method"
                            ],
                        }
                    )

                    residual, current, phase_artifact = train_pair(
                        case, measurement, prior_network, prior_artifact
                    )
                    prior_scale = float(residual["prior_scale"])

                    neural_items = (
                        ("Improved residual INR", residual),
                        ("Current-only INR (matched)", current),
                    )
                    maps_for_setting = {}
                    for method, artifact in neural_items:
                        row, maps = evaluate_reconstruction(
                            method,
                            artifact["recon"],
                            case,
                            measurement,
                            prior_to_acquisition=prior_scale,
                            artifact=artifact,
                        )
                        metric_rows.append(row)
                        maps_for_setting[method] = maps

                    # Context baselines. The registered prior uses the shared
                    # initialized phase solely to make its data error meaningful.
                    phase_init = phase_artifact["source_phase"]
                    calibrated_prior = torch.polar(
                        prior_scale * case["prior"], phase_init
                    )
                    context_items = [
                        ("Calibrated registered prior", calibrated_prior, 0.0),
                        ("Zero-filled", measurement["zero_filled"], 0.0),
                    ]
                    synchronize()
                    cg_started = time.perf_counter()
                    cg_recon = conjugate_gradient_sense(
                        measurement["operator"].to(DEVICE),
                        measurement["kspace"].to(DEVICE),
                        num_iters=15,
                        lambda_l2=1e-4,
                        tolerance=1e-10,
                    ).detach().cpu()
                    synchronize()
                    context_items.append(
                        ("CG-SENSE", cg_recon, time.perf_counter() - cg_started)
                    )
                    for method, reconstruction, runtime in context_items:
                        row, maps = evaluate_reconstruction(
                            method,
                            reconstruction,
                            case,
                            measurement,
                            prior_to_acquisition=prior_scale,
                            runtime=runtime,
                        )
                        metric_rows.append(row)
                        maps_for_setting[method] = maps

                    keep_qualitative = (
                        replicate == 0
                        and requested_r == (6 if SMOKE else 9)
                        and (
                            SMOKE
                            or int(case["row"]["scan_index"]) in {9, 15, 18}
                        )
                    )
                    if keep_qualitative:
                        qualitative[(case["case_id"], requested_r, replicate)] = {
                            "metadata": copy.deepcopy(case["row"]),
                            "effective_r": measurement["info"].effective_acceleration,
                            "maps": maps_for_setting,
                            "delta": residual["delta"],
                            "phase": residual["phase"],
                        }

                    print(
                        f"  R={requested_r:2d} rep={replicate} "
                        f"effective={measurement['info'].effective_acceleration:.2f} "
                        f"params={residual['trainable_parameters']:,} each"
                    )
                    del residual, current, cg_recon, maps_for_setting
                    cleanup_cuda()

                # Incremental checkpoint in case a long run is interrupted.
                pd.DataFrame(metric_rows).to_csv(
                    OUTPUT_DIR / "per_case_metrics.partial.csv", index=False
                )
                pd.DataFrame(measurement_rows).to_csv(
                    OUTPUT_DIR / "measurement_manifest.partial.csv", index=False
                )
                del prior_network, prior_artifact, bank_by_replicate, case
                cleanup_cuda()

            per_case = pd.DataFrame(metric_rows)
            measurement_manifest = pd.DataFrame(measurement_rows)
            per_case.to_csv(OUTPUT_DIR / "per_case_metrics.csv", index=False)
            measurement_manifest.to_csv(
                OUTPUT_DIR / "measurement_manifest.csv", index=False
            )
            print(f"wrote {len(per_case)} metric rows")
            display(
                per_case[
                    [
                        "method", "case_id", "change_extent", "requested_r",
                        "mask_replicate", "effective_r", "laps_psnr", "laps_ssim",
                        "acq_nrmse", "change_cosine", "change_gain", "roi_rmse",
                        "data_error", "online_seconds",
                    ]
                ].round(4)
            )
            """
        ),
        markdown(
            r"""
            ## Verify matching and acquisition-unit invariants

            These checks turn the central comparison claims into executable
            invariants. They do not assert that either reconstruction is better.
            """
        ),
        code(
            r"""
            neural = per_case[
                per_case["method"].isin(
                    ["Improved residual INR", "Current-only INR (matched)"]
                )
            ].copy()
            parameter_counts = neural.groupby("method")["trainable_parameters"].unique()
            if any(len(values) != 1 for values in parameter_counts):
                raise RuntimeError("Trainable parameter count changed across cases")
            residual_count = int(
                parameter_counts.loc["Improved residual INR"][0]
            )
            current_count = int(
                parameter_counts.loc["Current-only INR (matched)"][0]
            )
            optional_scale_count = int(
                LOCKED["prior_scale_mode"] == "acquisition_learned"
            )
            if residual_count != current_count + optional_scale_count:
                raise RuntimeError("The magnitude and phase branches are not matched")

            prior_rows = per_case[per_case["method"] == "Calibrated registered prior"]
            if not np.allclose(prior_rows["change_gain"], 0.0, atol=1e-6):
                raise RuntimeError(
                    "A calibrated prior copy must have exactly zero reconstructed-change gain"
                )
            if not np.allclose(prior_rows["false_change_rms"], 0.0, atol=1e-6):
                raise RuntimeError(
                    "A calibrated prior copy must have exactly zero false-change RMS"
                )
            print(
                "matched magnitude+phase branch parameters:", current_count,
            )
            if optional_scale_count:
                print("residual additionally has one locked learned prior-scale scalar")
            print("calibrated-prior zero-change invariant: passed")
            """
        ),
    ]
)

cells.extend(
    [
        markdown(
            r"""
            ## One shared fixed-stop trainer

            The reusable experiment trainer has no reference-image input. It
            applies the same optimizer and cosine schedule to both model classes.
            Delta regularization is active only for the residual formulation;
            applying it to the current-only anatomy would not be a matched or
            meaningful objective.
            """
        ),
        code(
            r"""
            TRAIN_CONFIG = MagnitudePhaseTrainConfig(
                iterations=LOCKED["iterations"],
                magnitude_lr=LOCKED["magnitude_lr"],
                phase_lr=LOCKED["phase_lr"],
                prior_scale_lr=(
                    LOCKED["prior_scale_lr"]
                    if LOCKED["prior_scale_mode"] == "acquisition_learned"
                    else None
                ),
                lambda_delta_l1=LOCKED["lambda_delta_l1"],
                lambda_delta_tv=LOCKED["lambda_delta_tv"],
                lambda_phase_tv=LOCKED["lambda_phase_tv"],
                min_lr_ratio=LOCKED["min_lr_ratio"],
                grad_clip_norm=LOCKED["grad_clip_norm"],
                eval_every=LOCKED["eval_every"],
                fixed_phase=False,
            )
            print(TRAIN_CONFIG)


            def result_to_artifact(result, *, extra):
                return {
                    "mode": result.mode,
                    "recon": result.final_recon.detach().cpu(),
                    "magnitude": result.final_magnitude.detach().cpu(),
                    "phase": result.final_phase.detach().cpu(),
                    "delta": (
                        None
                        if result.final_delta is None
                        else result.final_delta.detach().cpu()
                    ),
                    "history": result.history,
                    "state_dict": result.state_dict,
                    "iterations_completed": result.iterations_completed,
                    "joint_runtime_seconds": result.runtime_seconds,
                    "trainable_parameters": result.trainable_parameters,
                    "total_parameters": result.total_parameters,
                    **extra,
                }


            def train_pair(case, measurement, prior_network, prior_artifact):
                phase_network, phase_artifact = get_phase_artifact(case, measurement)
                phase_state = clone_state(phase_network)
                common_payload = {
                    "case_id": case["case_id"],
                    "requested_r": measurement["requested_r"],
                    "replicate": measurement["mask_replicate"],
                    "mask_hash": measurement["mask_hash"],
                    "train_config": asdict(TRAIN_CONFIG),
                }

                # Improved residual: zero output starts exactly at calibrated prior.
                residual_path = cache_path("heldout_residual", common_payload)
                if residual_path.exists():
                    residual_artifact = torch.load(
                        residual_path, map_location="cpu", weights_only=False
                    )
                else:
                    prior_branch = make_scalar(
                        LOCKED["prior_kind"],
                        LOCKED["prior_kwargs"],
                        seed=stable_seed("prior", case["case_id"]),
                    )
                    prior_branch.load_state_dict(prior_artifact["state"])
                    delta_branch = make_scalar(
                        LOCKED["magnitude_kind"],
                        LOCKED["magnitude_kwargs"],
                        seed=stable_seed(
                            "magnitude-branch",
                            case["case_id"],
                            measurement["requested_r"],
                        ),
                        zero_last=True,
                    )
                    residual_phase = make_scalar(
                        LOCKED["phase_kind"],
                        LOCKED["phase_kwargs"],
                        seed=stable_seed(
                            "phase-network",
                            case["case_id"],
                            measurement["requested_r"],
                        ),
                    )
                    residual_phase.load_state_dict(phase_state)

                    coords = make_coord_grid(*case["stored_shape"], device=DEVICE)
                    prior_branch = prior_branch.to(DEVICE)
                    residual_phase = residual_phase.to(DEVICE)
                    with torch.no_grad():
                        fitted_prior = prior_branch(coords)[..., 0].reshape(
                            case["stored_shape"]
                        ).clamp_min(0.0)
                        fitted_phase = residual_phase(coords)[..., 0].reshape(
                            case["stored_shape"]
                        )
                        prior_scale = prior_scale_from_kspace(
                            fitted_prior,
                            fitted_phase,
                            measurement["operator"].to(DEVICE),
                            measurement["kspace"].to(DEVICE),
                        )
                        calibration_method = "complex_kspace_phase"
                    if float(prior_scale) <= 1e-8:
                        # A poor phase initializer can make the constrained real
                        # complex inner product nonpositive. Fall back to the
                        # same reference-free least-squares fit in zero-filled
                        # magnitude space rather than inventing a unit scale.
                        prior_scale = real_least_squares_scale(
                            fitted_prior,
                            measurement["zero_filled"].to(DEVICE).abs(),
                            weights=case["support"].to(DEVICE),
                        )
                        calibration_method = "zero_filled_magnitude_fallback"
                    if float(prior_scale) <= 1e-8:
                        raise RuntimeError(
                            f"Both reference-free prior calibrations failed for "
                            f"{case['case_id']}"
                        )

                    residual_model = PriorMagnitudePhaseINR(
                        prior_branch.cpu(),
                        delta_branch,
                        residual_phase.cpu(),
                        magnitude_residual_bound=LOCKED["magnitude_residual_bound"],
                        prior_scale=float(prior_scale),
                        learn_prior_scale=(
                            LOCKED["prior_scale_mode"] == "acquisition_learned"
                        ),
                    )
                    residual_model.freeze_prior()
                    result = train_magnitude_phase(
                        residual_model,
                        measurement["operator"],
                        measurement["kspace"],
                        case["stored_shape"],
                        cfg=TRAIN_CONFIG,
                        device=DEVICE,
                        verbose=False,
                    )
                    final_prior_scale = float(residual_model.prior_scale.detach())
                    residual_artifact = result_to_artifact(
                        result,
                        extra={
                            # The acquisition-only initial scale is the fixed,
                            # method-independent evaluation unit. A learned final
                            # scale is a model diagnostic, not a metric alignment.
                            "prior_scale": float(prior_scale),
                            "initial_prior_scale": float(prior_scale),
                            "final_prior_scale": final_prior_scale,
                            "prior_scale_mode": LOCKED["prior_scale_mode"],
                            "prior_calibration_method": calibration_method,
                            "prior_fit_runtime_seconds": prior_artifact["runtime_seconds"],
                            "phase_init_runtime_seconds": phase_artifact["runtime_seconds"],
                            "magnitude_init_runtime_seconds": 0.0,
                            "online_runtime_seconds": (
                                phase_artifact["runtime_seconds"] + result.runtime_seconds
                            ),
                            "offline_runtime_seconds": prior_artifact["runtime_seconds"],
                        },
                    )
                    torch.save(residual_artifact, residual_path)
                    del result, residual_model, prior_branch, delta_branch, residual_phase
                    cleanup_cuda()

                # Current-only: exact same branch definitions and phase state.
                current_path = cache_path("heldout_current", common_payload)
                if current_path.exists():
                    current_artifact = torch.load(
                        current_path, map_location="cpu", weights_only=False
                    )
                else:
                    current_magnitude, magnitude_artifact = (
                        get_current_magnitude_artifact(case, measurement)
                    )
                    current_phase = make_scalar(
                        LOCKED["phase_kind"],
                        LOCKED["phase_kwargs"],
                        seed=stable_seed(
                            "phase-network",
                            case["case_id"],
                            measurement["requested_r"],
                        ),
                    )
                    current_phase.load_state_dict(phase_state)
                    current_model = CurrentMagnitudePhaseINR(
                        current_magnitude, current_phase
                    )
                    result = train_magnitude_phase(
                        current_model,
                        measurement["operator"],
                        measurement["kspace"],
                        case["stored_shape"],
                        cfg=TRAIN_CONFIG,
                        device=DEVICE,
                        verbose=False,
                    )
                    current_artifact = result_to_artifact(
                        result,
                        extra={
                            "prior_scale": None,
                            "initial_prior_scale": None,
                            "final_prior_scale": None,
                            "prior_scale_mode": "not_applicable",
                            "prior_calibration_method": "not_applicable",
                            "prior_fit_runtime_seconds": 0.0,
                            "phase_init_runtime_seconds": phase_artifact["runtime_seconds"],
                            "magnitude_init_runtime_seconds": magnitude_artifact[
                                "runtime_seconds"
                            ],
                            "online_runtime_seconds": (
                                phase_artifact["runtime_seconds"]
                                + magnitude_artifact["runtime_seconds"]
                                + result.runtime_seconds
                            ),
                            "offline_runtime_seconds": 0.0,
                        },
                    )
                    torch.save(current_artifact, current_path)
                    del result, current_model, current_magnitude, current_phase
                    cleanup_cuda()

                expected_scale_parameter = int(
                    LOCKED["prior_scale_mode"] == "acquisition_learned"
                )
                if (
                    residual_artifact["trainable_parameters"]
                    != current_artifact["trainable_parameters"]
                    + expected_scale_parameter
                ):
                    raise RuntimeError(
                        "Branch-matched comparison failed: residual has "
                        f"{residual_artifact['trainable_parameters']:,} trainable "
                        "parameters but current-only has "
                        f"{current_artifact['trainable_parameters']:,}; expected only "
                        f"{expected_scale_parameter} optional prior-scale parameter."
                    )
                return residual_artifact, current_artifact, phase_artifact
            """
        ),
        markdown(
            r"""
            ## Evaluation metrics with an explicit scale contract

            Two families are intentionally kept separate:

            1. `laps_psnr` and `laps_ssim` apply the release-style oracle scalar
               magnitude alignment solely for literature comparison.
            2. Every `acq_*`, change, ROI, MI, and error-map value uses fixed
               acquisition-derived units. No reconstruction-specific reference
               gain is allowed.

            The evaluation-only change ROI is the largest-change decile inside a
            method-independent prior/reference foreground. It is never used for
            training or stopping. For `change_extent=0`, false-change RMS and L1
            are the primary longitudinal diagnostics; change direction can be
            unstable when true change energy is small.
            """
        ),
        code(
            r"""
            def laps_metrics(reconstruction, reference, support):
                estimate = np.abs(to_numpy(reconstruction)).astype(np.float64)
                target = np.abs(to_numpy(reference)).astype(np.float64)
                support_np = to_numpy(support).astype(bool)
                estimate = estimate * support_np
                target = target * support_np
                target = target / (np.quantile(target[support_np], 0.99) + 1e-12)
                denominator = float(np.sum(estimate[support_np] ** 2))
                gain = (
                    1.0
                    if denominator <= 1e-20
                    else float(
                        np.sum(estimate[support_np] * target[support_np]) / denominator
                    )
                )
                aligned = gain * estimate
                return {
                    "laps_psnr": float(
                        peak_signal_noise_ratio(target, aligned, data_range=1.0)
                    ),
                    "laps_ssim": float(
                        structural_similarity(target, aligned, data_range=1.0)
                    ),
                    "laps_gain": gain,
                }


            def masked_cosine_gain(predicted, target, mask):
                p = predicted[mask].astype(np.float64)
                t = target[mask].astype(np.float64)
                target_energy = float(np.dot(t, t))
                target_norm = math.sqrt(target_energy)
                predicted_norm = float(np.linalg.norm(p))
                cosine = (
                    float("nan")
                    if target_norm <= 1e-12
                    else 0.0
                    if predicted_norm <= 1e-12
                    else float(np.dot(p, t) / (predicted_norm * target_norm))
                )
                gain = (
                    float("nan")
                    if target_energy <= 1e-12
                    else float(np.dot(p, t) / target_energy)
                )
                return cosine, gain


            def evaluate_reconstruction(
                method,
                reconstruction,
                case,
                measurement,
                *,
                prior_to_acquisition,
                artifact=None,
                runtime=None,
            ):
                reconstruction = reconstruction.detach().cpu()
                support = case["support"].cpu()
                support_np = to_numpy(support).astype(bool)
                recon_mag = np.abs(to_numpy(reconstruction)).astype(np.float64)
                reference_acq = (
                    measurement["reference_to_acquisition"]
                    * np.abs(to_numpy(case["reference"])).astype(np.float64)
                )
                prior_acq = (
                    float(prior_to_acquisition)
                    * np.abs(to_numpy(case["prior"])).astype(np.float64)
                )
                recon_mag[~support_np] = 0.0
                reference_acq[~support_np] = 0.0
                prior_acq[~support_np] = 0.0

                foreground_scale = np.quantile(reference_acq[support_np], 0.999) + 1e-12
                foreground = support_np & (
                    (reference_acq > 0.05 * foreground_scale)
                    | (prior_acq > 0.05 * foreground_scale)
                )
                if not np.any(foreground):
                    foreground = support_np.copy()
                true_change = reference_acq - prior_acq
                reconstructed_change = recon_mag - prior_acq
                change_threshold = np.quantile(np.abs(true_change[foreground]), 0.90)
                change_roi = foreground & (np.abs(true_change) >= change_threshold)
                if not np.any(change_roi):
                    change_roi = foreground.copy()

                error = recon_mag - reference_acq
                raw_mae = float(np.mean(np.abs(error[foreground])))
                raw_rmse = float(np.sqrt(np.mean(error[foreground] ** 2)))
                raw_nrmse = float(
                    np.linalg.norm(error[foreground])
                    / (np.linalg.norm(reference_acq[foreground]) + 1e-12)
                )
                roi_mae = float(np.mean(np.abs(error[change_roi])))
                roi_rmse = float(np.sqrt(np.mean(error[change_roi] ** 2)))
                roi_nrmse = float(
                    np.linalg.norm(error[change_roi])
                    / (np.linalg.norm(reference_acq[change_roi]) + 1e-12)
                )
                roi_cosine, roi_gain = masked_cosine_gain(
                    reconstructed_change, true_change, change_roi
                )

                longitudinal = acquisition_calibrated_longitudinal_metrics(
                    reconstruction * support,
                    case["reference"] * support,
                    case["prior"] * support,
                    reference_to_acquisition=measurement["reference_to_acquisition"],
                    prior_to_acquisition=float(prior_to_acquisition),
                )
                data_error = relative_kspace_error(
                    reconstruction.to(DEVICE),
                    measurement["operator"].to(DEVICE),
                    measurement["kspace"].to(DEVICE),
                )
                row = {
                    "method": method,
                    "case_id": case["case_id"],
                    "scan_index": int(case["row"]["scan_index"]),
                    "slice_index": int(case["row"]["slice_index"]),
                    "subject": int(case["row"]["subj_index"]),
                    "change_extent": int(case["row"]["change_extent"]),
                    "scan_type": case["row"]["scan_type"],
                    "requested_r": measurement["requested_r"],
                    "effective_r": measurement["info"].effective_acceleration,
                    "mask_replicate": measurement["mask_replicate"],
                    "retained_lines": measurement["info"].output_lines,
                    "center_lines": measurement["info"].center_lines,
                    "common_kspace_scale": measurement["common_scale"],
                    "reference_to_acquisition": measurement[
                        "reference_to_acquisition"
                    ],
                    "reference_calibration_method": measurement[
                        "reference_calibration_method"
                    ],
                    "prior_to_acquisition": float(prior_to_acquisition),
                    **laps_metrics(reconstruction, case["reference"], support),
                    "acq_mae": raw_mae,
                    "acq_rmse": raw_rmse,
                    "acq_nrmse": raw_nrmse,
                    "roi_fraction": float(change_roi.mean()),
                    "roi_mae": roi_mae,
                    "roi_rmse": roi_rmse,
                    "roi_nrmse": roi_nrmse,
                    "roi_change_cosine": roi_cosine,
                    "roi_change_gain": roi_gain,
                    "false_change_l1": float(
                        np.mean(np.abs(reconstructed_change[foreground]))
                    ),
                    "false_change_rms": float(
                        np.sqrt(np.mean(reconstructed_change[foreground] ** 2))
                    ),
                    "true_change_l1": float(np.mean(np.abs(true_change[foreground]))),
                    "true_change_rms": float(
                        np.sqrt(np.mean(true_change[foreground] ** 2))
                    ),
                    "data_error": data_error,
                    **longitudinal,
                }
                if artifact is not None:
                    row.update(
                        {
                            "trainable_parameters": artifact["trainable_parameters"],
                            "total_parameters": artifact["total_parameters"],
                            "prior_fit_seconds": artifact[
                                "prior_fit_runtime_seconds"
                            ],
                            "phase_init_seconds": artifact[
                                "phase_init_runtime_seconds"
                            ],
                            "magnitude_init_seconds": artifact[
                                "magnitude_init_runtime_seconds"
                            ],
                            "joint_fit_seconds": artifact[
                                "joint_runtime_seconds"
                            ],
                            "online_seconds": artifact["online_runtime_seconds"],
                            "offline_seconds": artifact["offline_runtime_seconds"],
                            "learned_delta_l1": (
                                float("nan")
                                if artifact["delta"] is None
                                else float(artifact["delta"].abs().mean())
                            ),
                            "initial_prior_scale": artifact[
                                "initial_prior_scale"
                            ],
                            "final_prior_scale": artifact["final_prior_scale"],
                            "prior_scale_mode": artifact["prior_scale_mode"],
                            "prior_calibration_method": artifact[
                                "prior_calibration_method"
                            ],
                        }
                    )
                else:
                    row.update(
                        {
                            "trainable_parameters": 0,
                            "total_parameters": 0,
                            "prior_fit_seconds": 0.0,
                            "phase_init_seconds": 0.0,
                            "magnitude_init_seconds": 0.0,
                            "joint_fit_seconds": float(runtime or 0.0),
                            "online_seconds": float(runtime or 0.0),
                            "offline_seconds": 0.0,
                            "learned_delta_l1": float("nan"),
                            "initial_prior_scale": float(prior_to_acquisition),
                            "final_prior_scale": float(prior_to_acquisition),
                            "prior_scale_mode": "context_fixed",
                            "prior_calibration_method": "shared_from_residual",
                        }
                    )
                maps = {
                    "reconstruction": recon_mag,
                    "reference_acq": reference_acq,
                    "prior_acq": prior_acq,
                    "error_magnitude": np.abs(error),
                    "true_change": true_change,
                    "reconstructed_change": reconstructed_change,
                    "change_roi": change_roi,
                }
                return row, maps
            """
        ),
    ]
)

cells.extend(
    [
        markdown(
            r"""
            ## Case and measurement construction

            All requested masks for one replicate are generated in ascending
            acceleration and nested. The common k-space normalization for that
            replicate is computed from its \(R=13\) subset, so it never uses a
            sample unavailable at a reported lower acceleration. The acquired
            mask remains the immutable upper bound.

            Reference-to-acquisition calibration below is evaluation-only. It
            maps the supplied complex reference to the normalized k-space units
            with one method-independent least-squares scalar and cannot affect
            training, checkpoint selection, or prior calibration.
            """
        ),
        code(
            r"""
            BANK_ACCELERATIONS = (3, 6, 9, 13)


            def make_scalar(kind, kwargs, *, seed, zero_last=False):
                options = dict(kwargs)
                configured_seed = int(options.pop("seed", seed))
                return build_scalar_inr(
                    kind,
                    seed=configured_seed,
                    zero_last=zero_last,
                    **options,
                )


            def load_case(case_row):
                sample = dataset[int(case_row["dataset_position"])]
                stored_shape = tuple(sample["stored_shape"])
                raw_kspace = sample["ksp"].to(torch.complex64)
                mps = sample["mps"].to(torch.complex64)
                acquired_mask = sample["mask"].float()

                reference = sample["recon"].to(torch.complex64)
                prior = sample["prior"].float()
                reference = reference / quantile_scale(reference, 0.999)
                prior = prior / quantile_scale(prior, 0.999)

                support_native = torch.linalg.vector_norm(mps, dim=0) > 0.5
                support = center_pad_to(support_native.float(), stored_shape).bool()
                case_id = (
                    f"scan{int(case_row['scan_index']):02d}_"
                    f"slice{int(case_row['slice_index']):03d}"
                )
                return {
                    "case_id": case_id,
                    "row": case_row.to_dict(),
                    "sample": sample,
                    "stored_shape": stored_shape,
                    "raw_kspace": raw_kspace,
                    "mps": mps,
                    "acquired_mask": acquired_mask,
                    "reference": reference,
                    "prior": prior,
                    "support": support,
                }


            def make_mask_bank(case, replicate):
                masks, infos = {}, {}
                source = case["acquired_mask"].clone()
                for requested_r in BANK_ACCELERATIONS:
                    mask, info = laps_retrospective_1d_mask(
                        source,
                        requested_r,
                        seed=stable_seed(
                            "heldout-mask",
                            int(case["row"]["scan_index"]),
                            int(case["row"]["slice_index"]),
                            int(replicate),
                            int(requested_r),
                        ),
                        phase_encode_dim=LOCKED["phase_encode_dim"],
                        vd_factor=LOCKED["vd_factor"],
                        n_candidates=LOCKED["n_candidates"],
                    )
                    if torch.any(mask.bool() & ~case["acquired_mask"].bool()):
                        raise RuntimeError("Retrospective mask escaped the acquired support")
                    masks[requested_r], infos[requested_r] = mask, info
                    source = mask

                # R=13 is a subset of every reported mask in this replicate.
                common_mask = masks[max(BANK_ACCELERATIONS)]
                common_operator = CenterPaddedSense(
                    case["mps"], common_mask, case["stored_shape"]
                )
                common_scale = quantile_scale(
                    common_operator.adjoint(case["raw_kspace"] * common_mask),
                    0.999,
                )
                return masks, infos, common_scale


            def make_measurement(case, requested_r, replicate, bank):
                masks, infos, common_scale = bank
                mask = masks[requested_r]
                operator = CenterPaddedSense(case["mps"], mask, case["stored_shape"])
                kspace = case["raw_kspace"] * mask / common_scale
                zero_filled = operator.adjoint(kspace).detach().cpu()

                # Evaluation-only fixed mapping of the supplied reference to the
                # same acquisition units. Use all originally acquired samples for
                # a stable, method-independent scalar.
                acquired_operator = CenterPaddedSense(
                    case["mps"], case["acquired_mask"], case["stored_shape"]
                )
                acquired_kspace = (
                    case["raw_kspace"] * case["acquired_mask"] / common_scale
                )
                with torch.no_grad():
                    reference_prediction = acquired_operator(
                        case["reference"].to(acquired_kspace.device)
                    )
                    reference_scale = real_least_squares_scale(
                        reference_prediction,
                        acquired_kspace,
                        weights=case["acquired_mask"],
                    )
                    reference_calibration_method = "complex_acquired_kspace"
                if float(reference_scale) <= 1e-8:
                    acquired_zero_filled = acquired_operator.adjoint(acquired_kspace)
                    reference_scale = real_least_squares_scale(
                        case["reference"].abs().to(acquired_zero_filled.device),
                        acquired_zero_filled.abs(),
                        weights=case["support"].to(acquired_zero_filled.device),
                    )
                    reference_calibration_method = "acquired_adjoint_magnitude_fallback"
                if float(reference_scale) <= 1e-8:
                    raise RuntimeError(
                        f"Both evaluation-only reference calibrations failed for "
                        f"{case['case_id']}"
                    )

                return {
                    "requested_r": int(requested_r),
                    "mask_replicate": int(replicate),
                    "mask": mask,
                    "mask_hash": tensor_hash(mask),
                    "info": infos[requested_r],
                    "operator": operator,
                    "kspace": kspace,
                    "zero_filled": zero_filled,
                    "common_scale": float(common_scale),
                    "reference_to_acquisition": float(reference_scale),
                    "reference_calibration_method": reference_calibration_method,
                }
            """
        ),
        markdown(
            r"""
            ## Initialization artifacts

            The DICOM prior is fitted once per case and treated as offline work.
            Phase is initialized from the selected current-only source for every
            case/mask. The same phase state is copied into both neural methods.

            The matched current-only magnitude branch is fitted to the
            zero-filled current magnitude before joint k-space optimization. Its
            initialization time is online and is reported separately. No prior
            value or follow-up reference enters that fit.
            """
        ),
        code(
            r"""
            def get_prior_artifact(case):
                payload = {
                    "case_id": case["case_id"],
                    "kind": LOCKED["prior_kind"],
                    "kwargs": LOCKED["prior_kwargs"],
                    "iterations": LOCKED["prior_iterations"],
                    "lr": LOCKED["prior_lr"],
                }
                path = cache_path("prior", payload)
                network = make_scalar(
                    LOCKED["prior_kind"],
                    LOCKED["prior_kwargs"],
                    seed=stable_seed("prior", case["case_id"]),
                )
                if path.exists():
                    artifact = torch.load(path, map_location="cpu", weights_only=False)
                    network.load_state_dict(artifact["state"])
                    return network, artifact

                set_seed(stable_seed("prior", case["case_id"]))
                network = network.to(DEVICE)
                synchronize()
                started = time.perf_counter()
                history = fit_prior(
                    network,
                    case["prior"],
                    cfg=PriorFitConfig(
                        iters=LOCKED["prior_iterations"],
                        lr=LOCKED["prior_lr"],
                        log_every=max(1, LOCKED["prior_iterations"] // 10),
                    ),
                    device=DEVICE,
                    verbose=False,
                )
                synchronize()
                artifact = {
                    "state": clone_state(network),
                    "history": history,
                    "runtime_seconds": time.perf_counter() - started,
                    "parameters": count_parameters(network),
                }
                torch.save(artifact, path)
                network = network.cpu()
                cleanup_cuda()
                return network, artifact


            def phase_source_image(measurement):
                if LOCKED["phase_source"] == "zf":
                    return measurement["zero_filled"].to(DEVICE)
                if LOCKED["phase_source"] == "cg":
                    return conjugate_gradient_sense(
                        measurement["operator"].to(DEVICE),
                        measurement["kspace"].to(DEVICE),
                        num_iters=25,
                        lambda_l2=1e-3,
                        tolerance=1e-10,
                    )
                raise ValueError(f"Unsupported locked phase source: {LOCKED['phase_source']}")


            def get_phase_artifact(case, measurement):
                payload = {
                    "case_id": case["case_id"],
                    "requested_r": measurement["requested_r"],
                    "replicate": measurement["mask_replicate"],
                    "mask_hash": measurement["mask_hash"],
                    "source": LOCKED["phase_source"],
                    "kind": LOCKED["phase_kind"],
                    "kwargs": LOCKED["phase_kwargs"],
                    "iterations": LOCKED["phase_iterations"],
                    "lr": LOCKED["phase_init_lr"],
                }
                path = cache_path("phase", payload)
                network = make_scalar(
                    LOCKED["phase_kind"],
                    LOCKED["phase_kwargs"],
                    seed=stable_seed(
                        "phase-network",
                        case["case_id"],
                        measurement["requested_r"],
                    ),
                )
                if path.exists():
                    artifact = torch.load(path, map_location="cpu", weights_only=False)
                    network.load_state_dict(artifact["state"])
                    return network, artifact

                source_image = phase_source_image(measurement)
                weights = (
                    source_image.abs() / quantile_scale(source_image, 0.99)
                ).clamp(0.0, 1.0)
                network = network.to(DEVICE)
                synchronize()
                started = time.perf_counter()
                history = fit_phase_inr(
                    network,
                    torch.angle(source_image),
                    cfg=PhaseFitConfig(
                        iters=LOCKED["phase_iterations"],
                        lr=LOCKED["phase_init_lr"],
                        log_every=max(1, LOCKED["phase_iterations"] // 10),
                    ),
                    weights=weights,
                    device=DEVICE,
                    verbose=False,
                )
                synchronize()
                artifact = {
                    "state": clone_state(network),
                    "history": history,
                    "runtime_seconds": time.perf_counter() - started,
                    "source_phase": torch.angle(source_image).detach().cpu(),
                    "parameters": count_parameters(network),
                }
                torch.save(artifact, path)
                network = network.cpu()
                del source_image, weights
                cleanup_cuda()
                return network, artifact


            def get_current_magnitude_artifact(case, measurement):
                payload = {
                    "case_id": case["case_id"],
                    "requested_r": measurement["requested_r"],
                    "replicate": measurement["mask_replicate"],
                    "mask_hash": measurement["mask_hash"],
                    "kind": LOCKED["magnitude_kind"],
                    "kwargs": LOCKED["magnitude_kwargs"],
                    "iterations": LOCKED["current_magnitude_iterations"],
                    "lr": LOCKED["current_magnitude_lr"],
                }
                path = cache_path("current_magnitude", payload)
                network = make_scalar(
                    LOCKED["magnitude_kind"],
                    LOCKED["magnitude_kwargs"],
                    seed=stable_seed(
                        "magnitude-branch",
                        case["case_id"],
                        measurement["requested_r"],
                    ),
                )
                if path.exists():
                    artifact = torch.load(path, map_location="cpu", weights_only=False)
                    network.load_state_dict(artifact["state"])
                    return network, artifact

                network = network.to(DEVICE)
                synchronize()
                started = time.perf_counter()
                history = fit_prior(
                    network,
                    measurement["zero_filled"].abs(),
                    cfg=PriorFitConfig(
                        iters=LOCKED["current_magnitude_iterations"],
                        lr=LOCKED["current_magnitude_lr"],
                        log_every=max(1, LOCKED["current_magnitude_iterations"] // 10),
                    ),
                    device=DEVICE,
                    verbose=False,
                )
                synchronize()
                artifact = {
                    "state": clone_state(network),
                    "history": history,
                    "runtime_seconds": time.perf_counter() - started,
                    "parameters": count_parameters(network),
                }
                torch.save(artifact, path)
                network = network.cpu()
                cleanup_cuda()
                return network, artifact
            """
        ),
    ]
)

# The large notebook sections are declared in independently reviewable blocks
# above. Put them into dependency order here before serialization.
if len(cells) != 36:
    raise RuntimeError(f"builder expected 36 cells, found {len(cells)}")
ordered_indices = (
    list(range(0, 11))       # title, imports, lock, panel, manifest
    + list(range(32, 36))    # measurement and initialization helpers
    + list(range(28, 32))    # shared trainer and metric helpers
    + list(range(24, 28))    # execute and verify invariants
    + list(range(14, 24))    # aggregation and qualitative analysis
    + list(range(11, 14))    # protocol record and output index
)
cells = [cells[index] for index in ordered_indices]

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
OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
print(f"wrote {OUTPUT} ({len(cells)} cells)")
