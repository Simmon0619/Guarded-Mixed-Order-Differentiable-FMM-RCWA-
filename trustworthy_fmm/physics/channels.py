"""Diffraction channels, propagation masks, and branch conventions."""

import math
import numpy as np
from typing import Iterable, List, Sequence, Tuple

import torch

from trustworthy_fmm.core.types import Order


def order_grid(mx: int, my: int = 0) -> List[Order]:
    if mx < 0 or my < 0:
        raise ValueError("order half-widths must be non-negative")
    return [(m, n) for m in range(-mx, mx + 1) for n in range(-my, my + 1)]


def circular_order_set(
    lattice_nm: Tuple[float, float], approximate_terms: int
) -> List[Order]:
    """Physical circular truncation matching the FMMAX expansion definition."""
    if approximate_terms <= 0 or any(value <= 0 for value in lattice_nm):
        raise ValueError("approximate_terms and lattice periods must be positive")
    values = np.arange(-approximate_terms // 2, approximate_terms // 2 + 1)
    candidates = []
    g1 = 2 * math.pi / lattice_nm[0]
    g2 = 2 * math.pi / lattice_nm[1]
    cutoff = math.sqrt(approximate_terms * abs(g1 * g2) / math.pi)
    for m in values:
        for n in values:
            magnitude = math.sqrt((m * g1) ** 2 + (n * g2) ** 2)
            if magnitude < cutoff:
                candidates.append((magnitude, int(m), int(n)))
    candidates.sort(key=lambda item: item[0])
    return [(m, n) for _, m, n in candidates]


def outgoing_sqrt(value: torch.Tensor, atol: float = 1e-12) -> torch.Tensor:
    """Square root with positive-real propagating and positive-imaginary decay."""
    root = torch.sqrt(value)
    flip_decay = root.imag < -atol
    flip_propagating = (torch.abs(root.imag) <= atol) & (root.real < 0)
    return torch.where(flip_decay | flip_propagating, -root, root)


def normalized_in_plane_wavevectors(
    orders: Sequence[Order],
    wavelength_nm: float,
    lattice_nm: Tuple[float, float],
    incident_index: float,
    theta_deg: float = 0.0,
    phi_deg: float = 0.0,
    *,
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    base = incident_index * math.sin(theta)
    kx0 = base * math.cos(phi)
    ky0 = base * math.sin(phi)
    kx = torch.tensor(
        [kx0 + m * wavelength_nm / lattice_nm[0] for m, _ in orders],
        dtype=dtype,
        device=device,
    )
    ky = torch.tensor(
        [ky0 + n * wavelength_nm / lattice_nm[1] for _, n in orders],
        dtype=dtype,
        device=device,
    )
    return kx, ky


def longitudinal_wavevectors(
    kx_normalized: torch.Tensor,
    ky_normalized: torch.Tensor,
    refractive_index: complex,
) -> torch.Tensor:
    complex_dtype = torch.complex128 if kx_normalized.dtype == torch.float64 else torch.complex64
    kx = kx_normalized.to(complex_dtype)
    ky = ky_normalized.to(complex_dtype)
    n = torch.as_tensor(refractive_index, dtype=complex_dtype, device=kx.device)
    return outgoing_sqrt(n * n - kx * kx - ky * ky)


def propagating_mask(kz_normalized: torch.Tensor, tolerance: float = 1e-9) -> torch.Tensor:
    return (kz_normalized.real > 0) & (torch.abs(kz_normalized.imag) < tolerance)


def diffraction_angles_deg(
    kx_normalized: torch.Tensor,
    ky_normalized: torch.Tensor,
    refractive_index: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    radial = torch.sqrt(kx_normalized ** 2 + ky_normalized ** 2)
    valid = radial <= abs(refractive_index) + 1e-12
    ratio = torch.clamp(radial / abs(refractive_index), 0.0, 1.0)
    theta = torch.rad2deg(torch.asin(ratio))
    phi = torch.rad2deg(torch.atan2(ky_normalized, kx_normalized))
    nan = torch.full_like(theta, float("nan"))
    return torch.where(valid, theta, nan), torch.where(valid, phi, nan), valid
