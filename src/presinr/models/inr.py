"""Coordinate-based implicit neural representations (INRs).

Two parameterizations are provided:

* :class:`Siren` -- sine-activated MLP (Sitzmann et al., 2020). Default; its
  periodic activations fit high-frequency image structure without an explicit
  positional encoding.
* :class:`FourierMLP` -- ReLU MLP on random Fourier features (Tancik et al.,
  2020). Kept as an ablation alternative.

Both map coordinates ``(N, 2)`` in ``[-1, 1]^2`` to ``(N, out_features)``.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def make_coord_grid(nx: int, ny: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return a ``(nx*ny, 2)`` grid of normalized coordinates in ``[-1, 1]``.

    Row-major over ``(x, y)`` so ``reshape(nx, ny)`` recovers image layout.
    """
    ys = torch.linspace(-1.0, 1.0, ny, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, nx, device=device, dtype=dtype)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)


class SineLayer(nn.Module):
    def __init__(self, in_f, out_f, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_f, out_f)
        self._init_weights(in_f)

    def _init_weights(self, in_f):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / in_f
            else:
                bound = math.sqrt(6.0 / in_f) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class Siren(nn.Module):
    """Sine-activated coordinate MLP."""

    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 1,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        final_scale: float = 1.0,
    ):
        super().__init__()
        self.final_scale = final_scale
        layers = [SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0)]
        for _ in range(hidden_layers):
            layers.append(SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0))
        final = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
            final.weight.uniform_(-bound, bound)
            final.bias.zero_()
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.final_scale * self.net(coords)


class FourierMLP(nn.Module):
    """ReLU MLP on fixed random Fourier features."""

    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 1,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        mapping_size: int = 128,
        sigma: float = 10.0,
        final_scale: float = 1.0,
        seed: Optional[int] = 0,
    ):
        super().__init__()
        self.final_scale = final_scale
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        B = torch.randn(mapping_size, in_features, generator=g) * sigma
        self.register_buffer("B", B)
        layers, dim = [], 2 * mapping_size
        for _ in range(hidden_layers + 1):
            layers += [nn.Linear(dim, hidden_features), nn.ReLU(inplace=True)]
            dim = hidden_features
        layers.append(nn.Linear(dim, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * coords @ self.B.t()
        feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        return self.final_scale * self.net(feats)


def build_inr(kind: str = "siren", **kwargs) -> nn.Module:
    kind = kind.lower()
    if kind == "siren":
        return Siren(**kwargs)
    if kind in ("fourier", "ffn", "fourier_mlp"):
        return FourierMLP(**kwargs)
    raise ValueError(f"unknown INR kind: {kind}")
