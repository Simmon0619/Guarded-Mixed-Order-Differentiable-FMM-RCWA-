"""Differentiable scalar 1D TE RCWA baseline.

This is intentionally a small, auditable baseline: one sampled isotropic
patterned layer between homogeneous half spaces.  It establishes convolution,
modal, S-matrix, order-normalization, and autograd behavior before the vector 2D
formulation is added.
"""

import math
from typing import Dict, Sequence

import torch

from trustworthy_fmm.core.types import ScatteringResult
from trustworthy_fmm.fourier.convolution import convolution_matrix_1d
from trustworthy_fmm.fourier.factorization import (
    FFT_BASELINE,
    canonical_factorization,
    effective_permittivity_1d,
)
from trustworthy_fmm.physics.channels import outgoing_sqrt
from trustworthy_fmm.scattering.smatrix import (
    interface_smatrix,
    redheffer_star,
    transparent_propagation,
)


def _minimum_eigen_gap(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return torch.as_tensor(float("inf"), dtype=values.real.dtype, device=values.device)
    distances = torch.abs(values[:, None] - values[None, :])
    diagonal = torch.eye(values.numel(), dtype=torch.bool, device=values.device)
    distances = distances.masked_fill(diagonal, float("inf"))
    return torch.min(distances)


def _solve_1d_scalar(
    eps_profile: torch.Tensor,
    thickness_nm: float,
    wavelength_nm: float,
    period_nm: float,
    superstrate_index: complex,
    substrate_index: complex,
    half_order: int,
    theta_deg: float = 0.0,
    polarization: str = "te",
    factorization: str = FFT_BASELINE,
) -> ScatteringResult:
    if eps_profile.ndim != 1:
        raise ValueError("eps_profile must be a one-dimensional sampled unit cell")
    if not eps_profile.is_complex():
        target_dtype = torch.complex128 if eps_profile.dtype == torch.float64 else torch.complex64
        eps_profile = eps_profile.to(target_dtype)
    if thickness_nm < 0 or wavelength_nm <= 0 or period_nm <= 0:
        raise ValueError("physical lengths must be non-negative, with positive wavelength/period")
    if half_order < 0:
        raise ValueError("half_order must be non-negative")

    device = eps_profile.device
    dtype = eps_profile.dtype
    real_dtype = eps_profile.real.dtype
    orders_1d = torch.arange(-half_order, half_order + 1, dtype=real_dtype, device=device)
    order_tuples = [(int(m), 0) for m in range(-half_order, half_order + 1)]
    n_super = torch.as_tensor(superstrate_index, dtype=dtype, device=device)
    n_sub = torch.as_tensor(substrate_index, dtype=dtype, device=device)
    incident_kx = n_super * math.sin(math.radians(theta_deg))
    kx = incident_kx + orders_1d.to(dtype) * wavelength_nm / period_nm
    q_super = outgoing_sqrt(n_super * n_super - kx * kx)
    q_sub = outgoing_sqrt(n_sub * n_sub - kx * kx)

    count = orders_1d.numel()
    identity = torch.eye(count, dtype=dtype, device=device)
    formulation = canonical_factorization(factorization, dimensions=1)
    if polarization == "te" and formulation != FFT_BASELINE:
        raise ValueError("Li inverse factorization is only applicable to 1D TM polarization")
    eps_conv = convolution_matrix_1d(eps_profile, half_order)
    kx_matrix = torch.diag(kx)
    if polarization == "te":
        operator = eps_conv - kx_matrix @ kx_matrix
    elif polarization == "tm":
        eps_conv, transverse_eps = effective_permittivity_1d(
            eps_profile, half_order, formulation
        )
        p_tm = identity - kx_matrix @ torch.linalg.solve(eps_conv, kx_matrix)
        operator = p_tm @ transverse_eps
    else:
        raise ValueError("polarization must be 'te' or 'tm'")
    eigenvalues, w_layer = torch.linalg.eig(operator)
    q_layer = outgoing_sqrt(eigenvalues)
    if polarization == "te":
        v_layer = w_layer @ torch.diag(q_layer)
        admittance_super = q_super
        admittance_sub = q_sub
    else:
        v_layer = torch.linalg.solve(p_tm, w_layer @ torch.diag(q_layer))
        admittance_super = (n_super * n_super) / q_super
        admittance_sub = (n_sub * n_sub) / q_sub
    w_super = identity
    v_super = torch.diag(admittance_super)
    w_sub = identity
    v_sub = torch.diag(admittance_sub)
    entry = interface_smatrix(w_super, v_super, w_layer, v_layer)
    phase = torch.exp(1j * (2 * math.pi / wavelength_nm) * q_layer * thickness_nm)
    propagation = transparent_propagation(phase)
    exit_interface = interface_smatrix(w_layer, v_layer, w_sub, v_sub)
    global_s = redheffer_star(redheffer_star(entry, propagation), exit_interface)

    incident = torch.zeros(count, dtype=dtype, device=device)
    incident[half_order] = 1
    reflected = global_s.s11 @ incident
    transmitted = global_s.s21 @ incident
    incident_admittance = admittance_super[half_order].real
    propagating_super = (q_super.real > 0) & (torch.abs(q_super.imag) < 1e-9)
    propagating_sub = (q_sub.real > 0) & (torch.abs(q_sub.imag) < 1e-9)
    power_r = torch.where(
        propagating_super,
        torch.abs(reflected) ** 2 * admittance_super.real / incident_admittance,
        torch.zeros_like(q_super.real),
    )
    power_t = torch.where(
        propagating_sub,
        torch.abs(transmitted) ** 2 * admittance_sub.real / incident_admittance,
        torch.zeros_like(q_sub.real),
    )
    total_r = torch.sum(power_r).real
    total_t = torch.sum(power_t).real
    absorption = 1.0 - total_r - total_t
    lossless = (
        torch.max(torch.abs(eps_profile.imag)) < 1e-12
        and torch.abs(n_super.imag) < 1e-12
        and torch.abs(n_sub.imag) < 1e-12
    )
    energy_residual = torch.abs(total_r + total_t - 1.0) if bool(lossless) else torch.zeros_like(total_r)
    diagnostics: Dict[str, object] = {
        "half_order": half_order,
        "dtype": str(dtype),
        "min_eigen_gap": _minimum_eigen_gap(eigenvalues),
        "modal_condition": torch.linalg.cond(w_layer),
        "formulation": "scalar_1d_{}".format(polarization),
        "factorization": formulation,
        "propagating_reflection_orders": int(torch.sum(propagating_super).item()),
        "propagating_transmission_orders": int(torch.sum(propagating_sub).item()),
    }
    return ScatteringResult(
        orders=order_tuples,
        complex_t=transmitted,
        complex_r=reflected,
        power_t=power_t.real,
        power_r=power_r.real,
        total_T=total_t,
        total_R=total_r,
        absorption=absorption.real,
        energy_residual=energy_residual.real,
        diagnostics=diagnostics,
    )


def solve_1d_te(
    eps_profile: torch.Tensor,
    thickness_nm: float,
    wavelength_nm: float,
    period_nm: float,
    superstrate_index: complex,
    substrate_index: complex,
    half_order: int,
    theta_deg: float = 0.0,
) -> ScatteringResult:
    return _solve_1d_scalar(
        eps_profile,
        thickness_nm,
        wavelength_nm,
        period_nm,
        superstrate_index,
        substrate_index,
        half_order,
        theta_deg,
        "te",
    )


def solve_1d_tm(
    eps_profile: torch.Tensor,
    thickness_nm: float,
    wavelength_nm: float,
    period_nm: float,
    superstrate_index: complex,
    substrate_index: complex,
    half_order: int,
    theta_deg: float = 0.0,
    factorization: str = FFT_BASELINE,
) -> ScatteringResult:
    return _solve_1d_scalar(
        eps_profile,
        thickness_nm,
        wavelength_nm,
        period_nm,
        superstrate_index,
        substrate_index,
        half_order,
        theta_deg,
        "tm",
        factorization,
    )


def solve_multilayer_1d(
    eps_profiles: Sequence[torch.Tensor],
    thicknesses_nm: Sequence[float],
    wavelength_nm: float,
    period_nm: float,
    superstrate_index: complex,
    substrate_index: complex,
    half_order: int,
    theta_deg: float = 0.0,
    polarization: str = "te",
    factorization: str = FFT_BASELINE,
) -> ScatteringResult:
    """Cascade multiple sampled 1D patterned layers with stable S matrices."""
    if len(eps_profiles) == 0 or len(eps_profiles) != len(thicknesses_nm):
        raise ValueError("provide one positive-thickness entry per patterned layer")
    if any(thickness < 0 for thickness in thicknesses_nm):
        raise ValueError("layer thicknesses must be non-negative")
    first = eps_profiles[0]
    if first.ndim != 1:
        raise ValueError("each epsilon profile must be one-dimensional")
    if not first.is_complex():
        first = first.to(torch.complex128 if first.dtype == torch.float64 else torch.complex64)
    profiles = [profile.to(dtype=first.dtype, device=first.device) for profile in eps_profiles]
    if any(profile.shape != first.shape for profile in profiles):
        raise ValueError("all profiles must use the same unit-cell sampling")
    device, dtype, real_dtype = first.device, first.dtype, first.real.dtype
    orders_1d = torch.arange(-half_order, half_order + 1, dtype=real_dtype, device=device)
    order_tuples = [(int(m), 0) for m in range(-half_order, half_order + 1)]
    count = orders_1d.numel()
    identity = torch.eye(count, dtype=dtype, device=device)
    n_super = torch.as_tensor(superstrate_index, dtype=dtype, device=device)
    n_sub = torch.as_tensor(substrate_index, dtype=dtype, device=device)
    incident_kx = n_super * math.sin(math.radians(theta_deg))
    kx = incident_kx + orders_1d.to(dtype) * wavelength_nm / period_nm
    kx_matrix = torch.diag(kx)
    q_super = outgoing_sqrt(n_super * n_super - kx * kx)
    q_sub = outgoing_sqrt(n_sub * n_sub - kx * kx)
    formulation = canonical_factorization(factorization, dimensions=1)
    if polarization == "te" and formulation != FFT_BASELINE:
        raise ValueError("Li inverse factorization is only applicable to 1D TM polarization")
    if polarization == "te":
        admittance_super = q_super
        admittance_sub = q_sub
    elif polarization == "tm":
        admittance_super = (n_super * n_super) / q_super
        admittance_sub = (n_sub * n_sub) / q_sub
    else:
        raise ValueError("polarization must be 'te' or 'tm'")
    current_w = identity
    current_v = torch.diag(admittance_super)
    global_s = None
    eigen_gaps = []
    conditions = []
    for profile, thickness in zip(profiles, thicknesses_nm):
        eps_conv = convolution_matrix_1d(profile, half_order)
        if polarization == "te":
            operator = eps_conv - kx_matrix @ kx_matrix
            eigenvalues, w_layer = torch.linalg.eig(operator)
            q_layer = outgoing_sqrt(eigenvalues)
            v_layer = w_layer @ torch.diag(q_layer)
        else:
            eps_conv, transverse_eps = effective_permittivity_1d(
                profile, half_order, formulation
            )
            p_tm = identity - kx_matrix @ torch.linalg.solve(eps_conv, kx_matrix)
            operator = p_tm @ transverse_eps
            eigenvalues, w_layer = torch.linalg.eig(operator)
            q_layer = outgoing_sqrt(eigenvalues)
            v_layer = torch.linalg.solve(p_tm, w_layer @ torch.diag(q_layer))
        interface = interface_smatrix(current_w, current_v, w_layer, v_layer)
        phase = torch.exp(1j * (2 * math.pi / wavelength_nm) * q_layer * thickness)
        propagation = transparent_propagation(phase)
        global_s = interface if global_s is None else redheffer_star(global_s, interface)
        global_s = redheffer_star(global_s, propagation)
        current_w, current_v = w_layer, v_layer
        eigen_gaps.append(_minimum_eigen_gap(eigenvalues))
        conditions.append(torch.linalg.cond(w_layer))
    exit_interface = interface_smatrix(current_w, current_v, identity, torch.diag(admittance_sub))
    global_s = redheffer_star(global_s, exit_interface)
    incident = torch.zeros(count, dtype=dtype, device=device)
    incident[half_order] = 1
    reflected = global_s.s11 @ incident
    transmitted = global_s.s21 @ incident
    incident_admittance = admittance_super[half_order].real
    propagating_super = (q_super.real > 0) & (torch.abs(q_super.imag) < 1e-9)
    propagating_sub = (q_sub.real > 0) & (torch.abs(q_sub.imag) < 1e-9)
    power_r = torch.where(
        propagating_super,
        torch.abs(reflected) ** 2 * admittance_super.real / incident_admittance,
        torch.zeros_like(q_super.real),
    )
    power_t = torch.where(
        propagating_sub,
        torch.abs(transmitted) ** 2 * admittance_sub.real / incident_admittance,
        torch.zeros_like(q_sub.real),
    )
    total_r, total_t = torch.sum(power_r).real, torch.sum(power_t).real
    absorption = 1.0 - total_r - total_t
    lossless = all(bool(torch.max(torch.abs(profile.imag)) < 1e-12) for profile in profiles)
    lossless = lossless and bool(torch.abs(n_super.imag) < 1e-12) and bool(torch.abs(n_sub.imag) < 1e-12)
    energy_residual = torch.abs(total_r + total_t - 1.0) if lossless else torch.zeros_like(total_r)
    diagnostics: Dict[str, object] = {
        "half_order": half_order,
        "dtype": str(dtype),
        "layer_count": len(profiles),
        "min_eigen_gap": torch.min(torch.stack(eigen_gaps)),
        "max_modal_condition": torch.max(torch.stack(conditions)),
        "formulation": "scalar_1d_{}_multilayer".format(polarization),
        "factorization": formulation,
    }
    return ScatteringResult(
        orders=order_tuples,
        complex_t=transmitted,
        complex_r=reflected,
        power_t=power_t.real,
        power_r=power_r.real,
        total_T=total_t,
        total_R=total_r,
        absorption=absorption.real,
        energy_residual=energy_residual.real,
        diagnostics=diagnostics,
    )
