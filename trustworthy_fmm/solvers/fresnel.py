"""Differentiable analytic Fresnel and homogeneous-film references."""

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from trustworthy_fmm.physics.channels import outgoing_sqrt


@dataclass
class FresnelResult:
    r: torch.Tensor
    t: torch.Tensor
    R: torch.Tensor
    T: torch.Tensor
    energy_residual: torch.Tensor


def _as_complex(value: complex, dtype: torch.dtype, device: str) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device)


def _modal_admittance(
    refractive_index: torch.Tensor,
    kx_normalized: torch.Tensor,
    polarization: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    kz = outgoing_sqrt(refractive_index * refractive_index - kx_normalized * kx_normalized)
    if polarization == "s":
        admittance = kz
    elif polarization == "p":
        admittance = refractive_index * refractive_index / kz
    else:
        raise ValueError("polarization must be 's' or 'p'")
    return kz, admittance


def fresnel_interface(
    incident_index: complex,
    transmitted_index: complex,
    theta_deg: float = 0.0,
    polarization: str = "s",
    *,
    dtype: torch.dtype = torch.complex128,
    device: str = "cpu",
) -> FresnelResult:
    n0 = _as_complex(incident_index, dtype, device)
    n1 = _as_complex(transmitted_index, dtype, device)
    kx = n0 * math.sin(math.radians(theta_deg))
    _, y0 = _modal_admittance(n0, kx, polarization)
    _, y1 = _modal_admittance(n1, kx, polarization)
    r = (y0 - y1) / (y0 + y1)
    t = 2 * y0 / (y0 + y1)
    R = torch.abs(r) ** 2
    T = torch.real(y1 / y0) * torch.abs(t) ** 2
    residual = torch.abs(R + T - 1)
    return FresnelResult(r, t, R.real, T.real, residual.real)


def homogeneous_stack(
    incident_index: complex,
    layer_indices: Sequence[complex],
    thicknesses_nm: Sequence[float],
    substrate_index: complex,
    wavelength_nm: float,
    theta_deg: float = 0.0,
    polarization: str = "s",
    *,
    dtype: torch.dtype = torch.complex128,
    device: str = "cpu",
) -> FresnelResult:
    if len(layer_indices) != len(thicknesses_nm):
        raise ValueError("each layer index requires one thickness")
    if wavelength_nm <= 0 or any(value < 0 for value in thicknesses_nm):
        raise ValueError("wavelength must be positive and thicknesses non-negative")
    n0 = _as_complex(incident_index, dtype, device)
    ns = _as_complex(substrate_index, dtype, device)
    kx = n0 * math.sin(math.radians(theta_deg))
    _, y0 = _modal_admittance(n0, kx, polarization)
    _, ys = _modal_admittance(ns, kx, polarization)
    matrix = torch.eye(2, dtype=dtype, device=device)
    k0 = 2 * math.pi / wavelength_nm
    for index, thickness in zip(layer_indices, thicknesses_nm):
        n = _as_complex(index, dtype, device)
        kz, y = _modal_admittance(n, kx, polarization)
        delta = k0 * kz * thickness
        cosine = torch.cos(delta)
        sine = torch.sin(delta)
        layer = torch.stack(
            (
                torch.stack((cosine, 1j * sine / y)),
                torch.stack((1j * y * sine, cosine)),
            )
        )
        matrix = matrix @ layer
    a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    denominator = y0 * a + y0 * ys * b + c + ys * d
    r = (y0 * a + y0 * ys * b - c - ys * d) / denominator
    t = 2 * y0 / denominator
    R = torch.abs(r) ** 2
    T = torch.real(ys / y0) * torch.abs(t) ** 2
    residual = torch.abs(R + T - 1)
    return FresnelResult(r, t, R.real, T.real, residual.real)

