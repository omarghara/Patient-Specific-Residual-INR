"""Retrospective sampling helpers used by the LAPS comparison.

The SLAM test scans can already be undersampled.  A retrospective mask must
therefore be a *subset* of the acquired mask: selecting a location at which no
measurement exists would silently turn a missing sample into a measured zero.

The 1-D routine below mirrors the released LAPS variable-density experiment:
its ACS bookkeeping cap is 21 central lines at R <= 3 (15 otherwise), it draws
exterior lines with the paper's variable-density law, tries 100 candidate
masks, and keeps the candidate with the smallest maximum gap. The release's
odd-trim ACS quirk is preserved, so the physical mask can contain one extra
center line. The 2-D routine mirrors its radial variable-density subset rule
and acquisition-bounds definition.
"""

import math
from dataclasses import dataclass
from math import ceil
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CartesianMaskInfo:
    """Metadata for a retrospectively generated Cartesian mask."""

    requested_acceleration: float
    effective_acceleration: float
    phase_encode_dim: int
    input_lines: int
    output_lines: int
    center_lines: int


@dataclass(frozen=True)
class RadialMaskInfo:
    """Metadata for a retrospectively generated 2-D LAPS mask.

    The ``*_lines`` aliases intentionally match :class:`CartesianMaskInfo` so
    experiment code can handle either sampling dimension uniformly; in this
    class they count k-space points rather than Cartesian lines.
    """

    requested_acceleration: float
    effective_acceleration: float
    phase_encode_dim: int
    input_lines: int
    output_lines: int
    center_lines: int
    bounded_points: int


@dataclass(frozen=True)
class KspaceHoldoutInfo:
    """Metadata for a deterministic Cartesian acquired-line holdout."""

    phase_encode_dim: int
    acquired_lines: int
    training_lines: int
    validation_lines: int
    protected_center_lines: int
    validation_fraction: float


def _longest_consecutive(indices: torch.Tensor) -> torch.Tensor:
    """Return the longest consecutive run, matching the LAPS release helper."""
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if indices.numel() == 0:
        return indices
    indices = torch.unique(indices, sorted=True)
    best_start = 0
    best_length = 0
    current_start = 0
    current_length = 1
    for index in range(1, indices.numel()):
        if int(indices[index]) == int(indices[index - 1]) + 1:
            current_length += 1
        else:
            if current_length > best_length:
                best_start, best_length = current_start, current_length
            current_start = index
            current_length = 1
    if current_length > best_length:
        best_start, best_length = current_start, current_length
    if best_length < 2:
        return indices.new_empty(0)
    return indices[best_start : best_start + best_length]


def infer_phase_encode_dim(mask: torch.Tensor, seed: int = 0) -> int:
    """Infer which spatial axis contains the Cartesian line selection.

    For an originally fully sampled mask, LAPS randomly chooses one of the two
    axes.  A local generator makes that choice deterministic without changing
    global PyTorch or NumPy RNG state.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {tuple(mask.shape)}")
    acquired = mask != 0
    row_fraction = float(acquired.any(dim=1).float().mean())
    col_fraction = float(acquired.any(dim=0).float().mean())

    if row_fraction < col_fraction:
        return 0
    if col_fraction < row_fraction:
        return 1

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return int(torch.randint(0, 2, (1,), generator=generator).item())


def cartesian_line_holdout(
    mask: torch.Tensor,
    validation_fraction: float = 0.1,
    *,
    seed: int = 0,
    phase_encode_dim: Optional[int] = None,
    protected_center_lines: int = 0,
    min_validation_lines: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, KspaceHoldoutInfo]:
    """Split acquired Cartesian lines into disjoint train/validation masks.

    The validation mask is always a subset of the acquired mask.  Central ACS
    lines can be protected so phase initialization and sensitivity-weighted
    fitting retain their low-frequency anchor.  The split is intended for
    reference-free checkpoint selection; after selecting an iteration, callers
    may refit on the full acquired mask for that fixed number of steps.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {tuple(mask.shape)}")
    if not math.isfinite(float(validation_fraction)) or not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be finite and strictly between 0 and 1, "
            f"got {validation_fraction}"
        )
    if protected_center_lines < 0:
        raise ValueError("protected_center_lines must be nonnegative")
    if min_validation_lines < 1:
        raise ValueError("min_validation_lines must be positive")

    input_device, input_dtype = mask.device, mask.dtype
    acquired = mask.detach().cpu() != 0
    if phase_encode_dim is None:
        phase_encode_dim = infer_phase_encode_dim(acquired, seed=seed)
    if phase_encode_dim not in (0, 1):
        raise ValueError(f"phase_encode_dim must be 0 or 1, got {phase_encode_dim}")

    oriented = acquired.T if phase_encode_dim == 0 else acquired
    acquired_lines = oriented.any(dim=0)
    indices = torch.where(acquired_lines)[0]
    line_count = int(indices.numel())
    if line_count < 2:
        raise ValueError("at least two acquired lines are required for a holdout")

    protected_count = min(int(protected_center_lines), line_count - 1)
    if protected_count:
        center = (oriented.shape[1] - 1) / 2.0
        order = torch.argsort((indices.float() - center).abs())
        protected = indices[order[:protected_count]]
    else:
        protected = indices.new_empty(0)
    candidates = indices[~torch.isin(indices, protected)]
    if candidates.numel() == 0:
        raise ValueError("no acquired exterior lines remain for validation")

    requested = max(min_validation_lines, int(round(line_count * validation_fraction)))
    validation_count = min(requested, int(candidates.numel()), line_count - 1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    chosen = candidates[
        torch.randperm(int(candidates.numel()), generator=generator)[:validation_count]
    ]

    validation_oriented = torch.zeros_like(oriented)
    validation_oriented[:, chosen] = oriented[:, chosen]
    training_oriented = oriented & ~validation_oriented
    validation = validation_oriented.T if phase_encode_dim == 0 else validation_oriented
    training = training_oriented.T if phase_encode_dim == 0 else training_oriented

    if torch.any(training & validation) or not torch.equal(training | validation, acquired):
        raise RuntimeError("invalid k-space holdout partition")

    info = KspaceHoldoutInfo(
        phase_encode_dim=int(phase_encode_dim),
        acquired_lines=line_count,
        training_lines=int(training_oriented.any(dim=0).sum()),
        validation_lines=int(validation_oriented.any(dim=0).sum()),
        protected_center_lines=protected_count,
        validation_fraction=float(validation_count / line_count),
    )
    return (
        training.to(device=input_device, dtype=input_dtype),
        validation.to(device=input_device, dtype=input_dtype),
        info,
    )


def laps_retrospective_1d_mask(
    mask: torch.Tensor,
    acceleration: float,
    *,
    seed: int = 0,
    phase_encode_dim: Optional[int] = None,
    vd_factor: float = 0.8,
    n_candidates: int = 100,
) -> Tuple[torch.Tensor, CartesianMaskInfo]:
    """Further undersample an acquired 2-D mask with the LAPS 1-D rule.

    Args:
        mask: Acquired sampling mask with shape ``(H, W)``.
        acceleration: Requested total 1-D acceleration, including ACS lines.
        seed: Deterministic seed for orientation (if needed) and sampling.
        phase_encode_dim: Axis along which phase-encoding lines are selected.
            If omitted, it is inferred from the acquired mask.
        vd_factor: LAPS variable-density exponent (0.8 in Experiment 1).
        n_candidates: Number of random masks from which the smallest-gap mask
            is selected (100 in Experiment 1).

    Returns:
        ``(mask_out, info)``. ``mask_out`` is guaranteed to be a subset of the
        input mask and has the same dtype/device.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {tuple(mask.shape)}")
    if not math.isfinite(float(acceleration)) or acceleration <= 1:
        raise ValueError(f"acceleration must be finite and greater than 1, got {acceleration}")
    if not math.isfinite(float(vd_factor)) or vd_factor <= 0:
        raise ValueError(f"vd_factor must be finite and positive, got {vd_factor}")
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be positive, got {n_candidates}")

    input_device = mask.device
    input_dtype = mask.dtype
    acquired = mask.detach().cpu() != 0
    if phase_encode_dim is None:
        phase_encode_dim = infer_phase_encode_dim(acquired, seed=seed)
    if phase_encode_dim not in (0, 1):
        raise ValueError(f"phase_encode_dim must be 0 or 1, got {phase_encode_dim}")

    # Work in the LAPS orientation: phase-encoding lines are columns.
    oriented = acquired.T if phase_encode_dim == 0 else acquired
    sampled_lines = oriented.any(dim=0)
    sampled_indices = torch.where(sampled_lines)[0]
    if sampled_indices.numel() == 0:
        raise ValueError("input mask contains no acquired lines")

    n_lines_total = sampled_lines.numel()
    input_line_count = int(sampled_indices.numel())
    target_line_count = int(ceil(n_lines_total / float(acceleration)))

    # Preserve the released LAPS center-line detection and trimming behavior.
    # It first removes the first point of every run before finding the longest
    # continuous region. When the requested trim is odd, its symmetric integer
    # trim retains one more physical ACS line than the bookkeeping count. This
    # small quirk is kept so masks match the released experiment code.
    gaps = torch.diff(
        sampled_indices,
        prepend=sampled_indices[0:1] * 100,
    )
    center_run = torch.sort(
        _longest_consecutive(sampled_indices[gaps == 1])
    ).values
    max_center = 21 if acceleration <= 3 else 15
    center_count_bookkeeping = min(int(center_run.numel()), max_center)
    if center_count_bookkeeping:
        trim = (int(center_run.numel()) - center_count_bookkeeping) // 2
        if trim > 0:
            center_indices = center_run[trim:-trim]
        else:
            center_indices = center_run
    else:
        center_indices = sampled_indices.new_empty(0)

    exterior_mask = torch.ones_like(sampled_indices, dtype=torch.bool)
    if center_indices.numel():
        exterior_mask &= ~torch.isin(sampled_indices, center_indices)
    exterior_indices = sampled_indices[exterior_mask]
    exterior_needed = max(0, target_line_count - center_count_bookkeeping)

    # A requested R below the input R cannot invent lines.  Likewise, retain
    # every available exterior line if the target asks for more than exist.
    exterior_needed = min(exterior_needed, int(exterior_indices.numel()))
    if target_line_count >= input_line_count:
        selected = sampled_indices
    elif exterior_needed == 0:
        # This is the release behavior: if ACS alone exceeds the target line
        # count, further undersampling is abandoned rather than trimming ACS.
        selected = sampled_indices
    else:
        distances = (exterior_indices.float() - n_lines_total // 2).abs()
        base = 1.0 - (1.8 / n_lines_total) * distances
        probabilities = base.clamp_min(torch.finfo(torch.float32).eps).pow(vd_factor)
        probabilities /= probabilities.sum()

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        probs = probabilities[None].expand(n_candidates, -1)
        draws = torch.multinomial(
            probs,
            num_samples=exterior_needed,
            replacement=False,
            generator=generator,
        )
        candidates = exterior_indices[None].expand(n_candidates, -1).gather(1, draws)
        fixed = center_indices[None].expand(n_candidates, -1)
        bounds = torch.tensor([-1, n_lines_total], dtype=sampled_indices.dtype)
        bounds = bounds[None].expand(n_candidates, -1)
        complete = torch.sort(torch.cat([candidates, fixed, bounds], dim=1), dim=1).values
        max_gaps = torch.diff(complete, dim=1).amax(dim=1)
        best = int(torch.argmin(max_gaps))
        selected = torch.sort(torch.cat([center_indices, candidates[best]])).values

    output_oriented = oriented.clone()
    keep_lines = torch.zeros(n_lines_total, dtype=torch.bool)
    keep_lines[selected] = True
    output_oriented[:, ~keep_lines] = False
    output = output_oriented.T if phase_encode_dim == 0 else output_oriented

    if torch.any(output & ~acquired):
        raise RuntimeError("retrospective mask is not a subset of the acquired mask")

    output_line_count = int(selected.numel())
    info = CartesianMaskInfo(
        requested_acceleration=float(acceleration),
        effective_acceleration=float(n_lines_total / output_line_count),
        phase_encode_dim=int(phase_encode_dim),
        input_lines=input_line_count,
        output_lines=output_line_count,
        center_lines=int(center_indices.numel()),
    )
    return output.to(device=input_device, dtype=input_dtype), info


def _radial_center_mask(shape: Tuple[int, int], radius: int) -> torch.Tensor:
    height, width = shape
    row = torch.arange(height)[:, None]
    column = torch.arange(width)[None, :]
    return (row - height // 2).square() + (column - width // 2).square() <= radius**2


def _mask_bounds(mask: torch.Tensor) -> torch.Tensor:
    """Match LAPS's one-pixel 3x3 dilation used for acquisition bounds."""
    kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32)
    expanded = F.conv2d(mask[None, None].float(), kernel, padding=1)[0, 0]
    return expanded > 0


def laps_retrospective_2d_mask(
    mask: torch.Tensor,
    acceleration: float,
    *,
    seed: int = 0,
    vd_factor: float = 1.5,
) -> Tuple[torch.Tensor, RadialMaskInfo]:
    """Further undersample an acquired mask with the released LAPS 2-D rule.

    The method finds the largest fully acquired central circle, caps its radius
    at 14 pixels for ``R <= 15`` (10 otherwise), and samples acquired exterior
    points without replacement with radially decaying probability. Effective R
    is computed inside the released code's dilated acquisition bounds.
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {tuple(mask.shape)}")
    if not math.isfinite(float(acceleration)) or acceleration <= 1:
        raise ValueError(f"acceleration must be finite and greater than 1, got {acceleration}")
    if not math.isfinite(float(vd_factor)) or vd_factor <= 0:
        raise ValueError(f"vd_factor must be finite and positive, got {vd_factor}")

    input_device = mask.device
    input_dtype = mask.dtype
    acquired = mask.detach().cpu() != 0
    if not acquired.any():
        raise ValueError("input mask contains no acquired samples")
    height, width = acquired.shape

    low, high = 0, min(height, width) // 2
    while low < high:
        middle = (low + high + 1) // 2
        candidate = _radial_center_mask((height, width), middle)
        if torch.all(acquired[candidate]):
            low = middle
        else:
            high = middle - 1
    input_center_radius = low
    center_radius = min(input_center_radius, 14 if acceleration <= 15 else 10)
    center = _radial_center_mask((height, width), center_radius)
    center_points = int(center.sum())

    input_points = int(acquired.sum())
    bounded_points = int(_mask_bounds(acquired).sum())
    target_points = int(ceil(bounded_points / float(acceleration)))
    exterior_needed = max(0, target_points - center_points)

    if target_points >= input_points or exterior_needed == 0:
        output = acquired.clone()
    else:
        sampleable = acquired & ~center
        coordinates = torch.argwhere(sampleable).float()
        if coordinates.numel() == 0:
            output = acquired.clone()
        else:
            exterior_needed = min(exterior_needed, coordinates.shape[0])
            center_coordinate = torch.tensor([[height / 2, width / 2]])
            distances = torch.linalg.vector_norm(coordinates - center_coordinate, dim=1)
            distances = distances / distances.max().clamp_min(1e-12)
            probabilities = (1.0 - 0.8 * distances).pow(vd_factor)
            probabilities /= probabilities.sum()
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
            chosen = torch.multinomial(
                probabilities,
                num_samples=exterior_needed,
                replacement=False,
                generator=generator,
            )
            chosen_coordinates = coordinates[chosen].long()
            output = center.clone()
            output[chosen_coordinates[:, 0], chosen_coordinates[:, 1]] = True

    if torch.any(output & ~acquired):
        raise RuntimeError("retrospective mask is not a subset of the acquired mask")
    output_points = int(output.sum())
    info = RadialMaskInfo(
        requested_acceleration=float(acceleration),
        effective_acceleration=float(bounded_points / output_points),
        phase_encode_dim=-1,
        input_lines=input_points,
        output_lines=output_points,
        center_lines=center_points,
        bounded_points=bounded_points,
    )
    return output.to(device=input_device, dtype=input_dtype), info
