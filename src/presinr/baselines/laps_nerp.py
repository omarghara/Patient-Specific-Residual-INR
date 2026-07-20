"""Standalone port of the NeRP baseline released with LAPS.

This module intentionally does not import :mod:`laps`: importing the sibling
repository pulls in diffusion, ANTs, and CUDA-specific dependencies unrelated
to NeRP.  The implementation follows ``SetsompopLab/LAPS`` commit
``ca1b5cc8d0d24b164a848c6fbd06b3fc5ec7d99b`` while using presinr's compact
Cartesian SENSE operator.

LAPS adapted the original magnitude-only NeRP method to complex clinical MRI:

1. obtain phase from an R-specific CG-SENSE reconstruction;
2. fit a Fourier-feature SIREN to ``prior_magnitude * exp(i * cg_phase)``;
3. initialize a trainable 2x2 real/imaginary scale transform; and
4. fine-tune the network and transform against measured k-space.

The release trains with the scale transform but accidentally omits it from the
returned final image. :class:`LapsNerpResult` exposes both the release-faithful
and scale-applied outputs instead of silently choosing one.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..forward import CartesianSense
from ..utils import center_crop_to, center_pad_to


@dataclass
class LapsNerpStageConfig:
    """Optimizer and early-stopping settings for one NeRP stage."""

    max_iter: int = 1000
    lr: float = 1e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    improvement_threshold: float = 0.98
    patience: int = 50
    min_iterations: int = 300


def _default_prior_stage() -> LapsNerpStageConfig:
    return LapsNerpStageConfig(lr=1e-4, weight_decay=1e-4, patience=50)


def _default_kspace_stage() -> LapsNerpStageConfig:
    return LapsNerpStageConfig(lr=1e-5, weight_decay=0.0, patience=100)


@dataclass
class LapsNerpConfig:
    """Release defaults for the LAPS complex-NeRP baseline."""

    embedding_size: int = 256
    coordinate_size: int = 2
    embedding_scale: float = 3.0
    network_depth: int = 8
    network_width: int = 512
    output_size: int = 2
    omega_0: float = 30.0
    cg_iters: int = 25
    cg_lambda: float = 1e-3
    cg_tolerance: float = 1e-10
    add_scale_fix: bool = True
    prior_stage: LapsNerpStageConfig = field(default_factory=_default_prior_stage)
    kspace_stage: LapsNerpStageConfig = field(default_factory=_default_kspace_stage)


@dataclass
class LapsNerpResult:
    """Outputs and diagnostics from one LAPS-NeRP reconstruction."""

    recon_released: torch.Tensor
    recon_scaled: torch.Tensor
    cg_recon: torch.Tensor
    complex_prior: torch.Tensor
    scale_matrix: torch.Tensor
    prior_history: Dict[str, list]
    kspace_history: Dict[str, list]

    @property
    def recon(self) -> torch.Tensor:
        """Release-faithful output (the final scale matrix is omitted)."""
        return self.recon_released


class CenterPaddedSense(nn.Module):
    """SENSE on a stored image grid larger than the native k-space grid.

    SLAM often stores 256x256 target/prior images while its native k-space has a
    smaller phase-encode dimension. LAPS reconstructs on the stored grid and
    center-crops inside the forward model; its adjoint applies the inverse
    center-padding operation. This wrapper gives all notebook methods identical
    geometry.
    """

    def __init__(
        self,
        mps: torch.Tensor,
        mask: torch.Tensor,
        image_shape: Sequence[int],
    ):
        super().__init__()
        if len(image_shape) != 2:
            raise ValueError(f"image_shape must have length 2, got {tuple(image_shape)}")
        self.native = CartesianSense(mps, mask)
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        native_shape = (self.native.nx, self.native.ny)
        if any(full < native for full, native in zip(self.image_shape, native_shape)):
            raise ValueError(
                f"stored image shape {self.image_shape} cannot be smaller than "
                f"native shape {native_shape}"
            )

    @property
    def mask(self) -> torch.Tensor:
        return self.native.mask

    @property
    def mps(self) -> torch.Tensor:
        return self.native.mps

    @property
    def native_shape(self) -> Tuple[int, int]:
        return self.native.nx, self.native.ny

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if tuple(image.shape[-2:]) != self.image_shape:
            raise ValueError(
                f"image must have stored shape {self.image_shape}, "
                f"got {tuple(image.shape[-2:])}"
            )
        return self.native(center_crop_to(image, self.native_shape))

    def adjoint(self, kspace: torch.Tensor) -> torch.Tensor:
        return center_pad_to(self.native.adjoint(kspace), self.image_shape)

    def zero_filled(self, kspace: torch.Tensor) -> torch.Tensor:
        return self.adjoint(kspace)

    def normal(self, image: torch.Tensor) -> torch.Tensor:
        return self.adjoint(self.forward(image))


class GaussianPositionalEncoder(nn.Module):
    """Fixed random Fourier features used by the LAPS NeRP release."""

    def __init__(
        self,
        embedding_size: int,
        coordinate_size: int,
        scale: float,
        matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        expected_shape = (embedding_size, coordinate_size)
        if matrix is None:
            matrix = torch.randn(*expected_shape) * scale
        elif tuple(matrix.shape) != expected_shape:
            raise ValueError(
                f"Fourier matrix must have shape {expected_shape}, got {tuple(matrix.shape)}"
            )
        self.register_buffer("B", matrix.detach().float().clone())

    @torch.no_grad()
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        projection = (2.0 * math.pi * coordinates) @ self.B.T
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class _LapsSirenLayer(nn.Module):
    """SIREN layer preserving the release's weight-only initialization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        omega_0: float,
        is_first: bool = False,
        is_last: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_last = is_last
        self.linear = nn.Linear(in_features, out_features)
        bound = (
            1.0 / in_features
            if is_first
            else math.sqrt(6.0 / in_features) / omega_0
        )
        # Deliberately leave nn.Linear's default bias initialization untouched.
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = self.linear(values)
        return values if self.is_last else torch.sin(self.omega_0 * values)


class LapsSiren(nn.Module):
    """Eight-layer, width-512 SIREN under the release defaults."""

    def __init__(self, config: LapsNerpConfig):
        super().__init__()
        if config.network_depth < 2:
            raise ValueError("network_depth must include at least first and final layers")
        input_size = 2 * config.embedding_size
        layers = [
            _LapsSirenLayer(
                input_size,
                config.network_width,
                omega_0=config.omega_0,
                is_first=True,
            )
        ]
        for _ in range(1, config.network_depth - 1):
            layers.append(
                _LapsSirenLayer(
                    config.network_width,
                    config.network_width,
                    omega_0=config.omega_0,
                )
            )
        layers.append(
            _LapsSirenLayer(
                config.network_width,
                config.output_size,
                omega_0=config.omega_0,
                is_last=True,
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.model(embedding)


def _coordinate_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    row, column = torch.meshgrid(
        torch.linspace(0.0, 1.0, height, device=device),
        torch.linspace(0.0, 1.0, width, device=device),
        indexing="ij",
    )
    return torch.stack([row, column], dim=-1)


def conjugate_gradient_sense(
    operator: CenterPaddedSense,
    kspace: torch.Tensor,
    *,
    num_iters: int = 25,
    lambda_l2: float = 1e-3,
    tolerance: float = 1e-10,
) -> torch.Tensor:
    """Complex CG with the same initialization and updates as LAPS."""
    rhs = operator.adjoint(kspace)
    estimate = rhs.clone()
    if num_iters <= 0:
        return estimate

    def normal(image: torch.Tensor) -> torch.Tensor:
        return operator.normal(image) + lambda_l2 * image

    residual = rhs - normal(estimate)
    direction = residual.clone()
    residual_inner = torch.real(torch.sum(residual.conj() * residual))
    for _ in range(num_iters):
        applied = normal(direction)
        denominator = torch.real(torch.sum(direction.conj() * applied))
        if not torch.isfinite(denominator) or denominator.abs() <= 1e-20:
            break
        alpha = residual_inner / denominator
        estimate = estimate + alpha * direction
        residual = residual - alpha * applied
        if torch.linalg.vector_norm(residual) < tolerance:
            break
        next_inner = torch.real(torch.sum(residual.conj() * residual))
        if residual_inner.abs() <= 1e-20:
            break
        direction = residual + (next_inner / residual_inner) * direction
        residual_inner = next_inner
    return estimate


def _complex_from_channels(channels: torch.Tensor) -> torch.Tensor:
    return torch.complex(channels[..., 0], channels[..., 1])


def _early_stopping_update(
    loss: float,
    best_loss: float,
    patience_counter: int,
    threshold: float,
) -> Tuple[float, int]:
    if loss < best_loss * threshold:
        return loss, 0
    return best_loss, patience_counter + 1


def fit_laps_nerp(
    prior_magnitude: torch.Tensor,
    operator: CenterPaddedSense,
    kspace: torch.Tensor,
    config: Optional[LapsNerpConfig] = None,
    *,
    device: Optional[torch.device] = None,
    fourier_matrix: Optional[torch.Tensor] = None,
    verbose: bool = True,
) -> LapsNerpResult:
    """Run the two-stage complex NeRP baseline released with LAPS.

    ``prior_magnitude`` must be on the operator's stored image grid. ``kspace``
    should already have the per-acceleration normalization used by LAPS
    (divide by the 99.9th percentile of ``abs(operator.adjoint(kspace))``); it is
    defensively re-masked here. An optional fixed ``fourier_matrix`` lets a
    sweep reuse the release's one encoder across reconstructions. The follow-up
    reference is intentionally not accepted, making evaluation leakage
    impossible.
    """
    config = config or LapsNerpConfig()
    if config.coordinate_size != 2:
        raise ValueError("LAPS-NeRP requires coordinate_size=2")
    if config.output_size != 2:
        raise ValueError("LAPS-NeRP requires output_size=2 real/imaginary channels")
    if config.embedding_size <= 0 or config.network_width <= 0:
        raise ValueError("embedding_size and network_width must be positive")
    device = device or prior_magnitude.device
    operator = operator.to(device)
    kspace = kspace.to(device)
    expected_kspace_shape = (
        operator.native.n_coils,
        operator.native.nx,
        operator.native.ny,
    )
    if tuple(kspace.shape) != expected_kspace_shape:
        raise ValueError(
            f"kspace must have shape {expected_kspace_shape}, got {tuple(kspace.shape)}"
        )
    if not torch.isfinite(kspace).all():
        raise ValueError("kspace contains non-finite values")
    kspace = kspace * operator.mask
    prior_magnitude = prior_magnitude.to(device).abs().float()
    if tuple(prior_magnitude.shape) != operator.image_shape:
        raise ValueError(
            f"prior shape {tuple(prior_magnitude.shape)} does not match "
            f"operator image shape {operator.image_shape}"
        )
    if not torch.isfinite(prior_magnitude).all():
        raise ValueError("prior magnitude contains non-finite values")
    max_prior = prior_magnitude.max()
    if max_prior <= 0:
        raise ValueError("prior magnitude is identically zero")
    prior_magnitude = prior_magnitude / max_prior

    with torch.no_grad():
        cg_recon = conjugate_gradient_sense(
            operator,
            kspace,
            num_iters=config.cg_iters,
            lambda_l2=config.cg_lambda,
            tolerance=config.cg_tolerance,
        )
        complex_prior = torch.polar(prior_magnitude, torch.angle(cg_recon))

    encoder = GaussianPositionalEncoder(
        config.embedding_size,
        config.coordinate_size,
        config.embedding_scale,
        matrix=fourier_matrix,
    ).to(device)
    model = LapsSiren(config).to(device).train()
    coordinates = _coordinate_grid(*operator.image_shape, device=device)
    with torch.no_grad():
        embedding = encoder(coordinates)
    prior_channels = torch.stack([complex_prior.real, complex_prior.imag], dim=-1)

    prior_cfg = config.prior_stage
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=prior_cfg.lr,
        betas=(prior_cfg.beta1, prior_cfg.beta2),
        weight_decay=prior_cfg.weight_decay,
    )
    prior_history: Dict[str, list] = {"loss": [], "psnr_proxy": []}
    best_loss = float("inf")
    patience_counter = 0
    for iteration in range(prior_cfg.max_iter):
        optimizer.zero_grad(set_to_none=True)
        output = model(embedding)
        loss = 0.5 * torch.nn.functional.mse_loss(output, prior_channels)
        loss.backward()
        optimizer.step()

        value = float(loss.detach())
        prior_history["loss"].append(value)
        prior_history["psnr_proxy"].append(-10.0 * math.log10(max(2.0 * value, 1e-30)))
        best_loss, patience_counter = _early_stopping_update(
            value,
            best_loss,
            patience_counter,
            prior_cfg.improvement_threshold,
        )
        if verbose and (iteration % 100 == 0 or iteration == prior_cfg.max_iter - 1):
            print(
                f"[LAPS-NeRP prior] iter {iteration:4d}  "
                f"loss={value:.3e}  patience={patience_counter}"
            )
        if (
            iteration >= prior_cfg.min_iterations
            and patience_counter >= prior_cfg.patience
        ):
            break

    with torch.no_grad():
        fitted_prior = model(embedding)
        cg_channels = torch.stack([cg_recon.real, cg_recon.imag], dim=-1)
        if config.add_scale_fix:
            scale_matrix = torch.linalg.lstsq(
                fitted_prior.reshape(-1, 2),
                cg_channels.reshape(-1, 2),
            ).solution
        else:
            scale_matrix = torch.eye(2, device=device)

    scale_parameter = nn.Parameter(scale_matrix.detach().clone())
    kspace_cfg = config.kspace_stage
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=kspace_cfg.lr,
        betas=(kspace_cfg.beta1, kspace_cfg.beta2),
        weight_decay=kspace_cfg.weight_decay,
    )
    if config.add_scale_fix:
        optimizer.add_param_group({"params": [scale_parameter]})

    kspace_history: Dict[str, list] = {"loss": []}
    best_loss = float("inf")
    patience_counter = 0
    for iteration in range(kspace_cfg.max_iter):
        optimizer.zero_grad(set_to_none=True)
        channels = model(embedding)
        if config.add_scale_fix:
            channels = channels @ scale_parameter
        prediction = _complex_from_channels(channels)
        loss = torch.mean(torch.abs(operator(prediction) - kspace) ** 2)
        loss.backward()
        optimizer.step()

        value = float(loss.detach())
        kspace_history["loss"].append(value)
        best_loss, patience_counter = _early_stopping_update(
            value,
            best_loss,
            patience_counter,
            kspace_cfg.improvement_threshold,
        )
        if verbose and (iteration % 100 == 0 or iteration == kspace_cfg.max_iter - 1):
            print(
                f"[LAPS-NeRP kspace] iter {iteration:4d}  "
                f"loss={value:.3e}  patience={patience_counter}"
            )
        if (
            iteration >= kspace_cfg.min_iterations
            and patience_counter >= kspace_cfg.patience
        ):
            break

    with torch.no_grad():
        final_channels = model(embedding)
        released = _complex_from_channels(final_channels).detach()
        scaled = _complex_from_channels(final_channels @ scale_parameter).detach()

    return LapsNerpResult(
        recon_released=released,
        recon_scaled=scaled,
        cg_recon=cg_recon.detach(),
        complex_prior=complex_prior.detach(),
        scale_matrix=scale_parameter.detach(),
        prior_history=prior_history,
        kspace_history=kspace_history,
    )
