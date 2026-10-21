"""Differentiable isotropic 2D vector FMM for stratified patterned layers.

The formulation uses transverse electric-field eigenvectors of ``P @ Q`` and
stable interface/propagation S-matrix composition.  It is written independently
but follows the explicit vector equations frozen in the D1 mathematics notes and
cross-checked against TORCWA conventions.  Tangential electric and magnetic
fields are matched at every finite-layer boundary, and the resulting local
scattering blocks are cascaded with Redheffer star products.
"""

import math
import os
from typing import Dict, Optional, Sequence, Tuple

import torch

# Opt-in fast path (E17 stage 0, 2026-08-31).  ``min_eigen_gap`` (a 2N x 2N
# pairwise-distance build) and ``modal_condition`` (a full SVD of W) are
# diagnostics only -- no quantity that reaches ``power_t`` or its gradient
# depends on them.  Skipping them changes NOTHING numerically; it only writes
# placeholder values (inf / 0) into the diagnostics dict.  Default OFF so the
# frozen protocol path is untouched.  Enable via TFMM_FAST_DIAGNOSTICS=1 or by
# setting FAST_DIAGNOSTICS = True; keep it OFF for any run that consumes the
# eigen-gap or modal-condition diagnostics.
FAST_DIAGNOSTICS = os.environ.get("TFMM_FAST_DIAGNOSTICS", "0") == "1"

# Opt-in gap-only diagnostics (S8/E3, 2026-09-16): with FAST_DIAGNOSTICS on,
# still compute ``min_eigen_gap`` (the cheap pairwise-distance build on the
# DETACHED eigenvalues; no autograd node, no effect on any gradient) and skip
# only the SVD of W (``modal_condition`` stays 0).  Needed by the eigen-gap
# guard, which must read the gap between the forward and the backward.
GAP_ONLY_DIAGNOSTICS = os.environ.get("TFMM_GAP_ONLY_DIAGNOSTICS", "0") == "1"

# Opt-in export of the layer eigenmodes (2026-09-16, field maps inside the
# structure): with EXPORT_LAYER_MODES on, ``fields`` also carries references
# to the per-layer (q, W, V), the exterior V matrices and the wavevectors.
# Nothing is computed in addition and nothing in the solve changes; the
# tensors are the ones the cascade used.  Off by default (they are 2N x 2N).
EXPORT_LAYER_MODES = os.environ.get("TFMM_EXPORT_LAYER_MODES", "0") == "1"

# Opt-in A2 memory/scale path (2026-09-05).  Same mathematics, different
# floating-point operation order -- NOT bitwise identical to the frozen
# protocol, so both default OFF (acceptance: E17 tolerances on GPU/c64,
# untouched bitwise baseline with the flags off).
#
# ``MEMORY_EFFICIENT`` removes the dense GEMMs against diagonal / permutation
# matrices when assembling P, Q and V (they become broadcasted scalings and a
# signed block swap), reduces the transient footprint of the 4Nx4N interface
# solve, folds the propagation star product into elementwise phase scalings,
# and factorizes each repeated Redheffer left-hand side once.  This shrinks
# both the forward peak and what autograd saves (the removed GEMMs no longer
# save their 2Nx2N operands).
#
# ``CHECKPOINT_SMATRIX`` wraps every interface/star S-matrix step in
# ``torch.utils.checkpoint`` so backward recomputes the step instead of
# storing its LU factors and solutions (~90% of all autograd-saved bytes at
# (40,10)@2x, A1 measurement).  The eigendecomposition is never recomputed.
# Enable via TFMM_MEMORY_EFFICIENT=1 / TFMM_CHECKPOINT_SMATRIX=1 or by
# setting the module attributes.
MEMORY_EFFICIENT = os.environ.get("TFMM_MEMORY_EFFICIENT", "0") == "1"
CHECKPOINT_SMATRIX = os.environ.get("TFMM_CHECKPOINT_SMATRIX", "0") == "1"

from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from trustworthy_fmm.core.types import ScatteringResult
from trustworthy_fmm.fourier.factorization import (
    FFT_BASELINE,
    canonical_factorization,
    transverse_permittivity_2d,
)
from trustworthy_fmm.physics.channels import outgoing_sqrt
from trustworthy_fmm.scattering.smatrix import (
    SMatrix,
    interface_smatrix,
    interface_smatrix_direct,
    propagation_star,
    redheffer_star,
    redheffer_star_fused,
    transparent_propagation,
)
from trustworthy_fmm.solvers.eigendecomposition import eigendecomposition


def _block2(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.cat((a, b), dim=1), torch.cat((c, d), dim=1)), dim=0)


def _external_v(kx: torch.Tensor, ky: torch.Tensor, epsilon: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    kz = outgoing_sqrt(epsilon - kx * kx - ky * ky)
    safe_kz = torch.where(torch.abs(kz) < 1e-14, kz + 1e-14, kz)
    d = torch.diag
    tmp1 = torch.cat(
        (
            d(-ky * kx / safe_kz),
            d(kz + kx * kx / safe_kz),
        ),
        dim=0,
    )
    tmp2 = torch.cat(
        (
            d(-kz - ky * ky / safe_kz),
            d(kx * ky / safe_kz),
        ),
        dim=0,
    )
    return kz, torch.cat((tmp1, tmp2), dim=1)


def _minimum_gap(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return torch.as_tensor(float("inf"), dtype=values.real.dtype, device=values.device)
    distance = torch.abs(values[:, None] - values[None, :])
    distance = distance.masked_fill(
        torch.eye(values.numel(), dtype=torch.bool, device=values.device), float("inf")
    )
    return torch.min(distance)


def _order_power(
    electric: torch.Tensor,
    v_matrix: torch.Tensor,
    order_count: int,
    incident_power: torch.Tensor,
    propagating: torch.Tensor,
) -> torch.Tensor:
    magnetic = v_matrix @ electric
    ex, ey = electric[:order_count], electric[order_count:]
    hx, hy = magnetic[:order_count], magnetic[order_count:]
    flux = torch.real(ex * torch.conj(hy) - ey * torch.conj(hx))
    return torch.where(propagating, flux / incident_power, torch.zeros_like(flux))


def _layer_modes(
    eps_yx: torch.Tensor,
    half_orders: Tuple[int, int],
    basis_orders: Optional[Sequence[Tuple[int, int]]],
    lattice_nm: Tuple[float, float],
    kx: torch.Tensor,
    ky: torch.Tensor,
    kx_matrix: torch.Tensor,
    ky_matrix: torch.Tensor,
    identity: torch.Tensor,
    zeros: torch.Tensor,
    factorization: str,
    normal_smoothing_sigma_px: float,
    detach_tangent_field: bool,
    tangent_field: Optional[Tuple[torch.Tensor, torch.Tensor]],
    eig_backward: str,
    eig_eps_relative: float,
    eig_eps_absolute: float,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Solve the transverse modes for one z-invariant isotropic layer."""
    eps_conv, transverse_eps, generated_tangent = transverse_permittivity_2d(
        eps_yx,
        half_orders,
        factorization,
        lattice_nm=lattice_nm,
        basis_orders=basis_orders,
        tangent_field=tangent_field,
        smoothing_sigma_px=normal_smoothing_sigma_px,
        detach_tangent_field=detach_tangent_field,
    )
    if MEMORY_EFFICIENT:
        count = kx.shape[0]
        # P = p_base + [[Kx],[Ky]] eps_conv^-1 [[Ky, -Kx]] assembled from the
        # kx/ky VECTORS: the diagonal-matrix GEMMs of the baseline path become
        # broadcasted row/column scalings (identical mathematics).
        eps_inverse = torch.linalg.solve(eps_conv, identity)
        a_block = torch.cat(
            (eps_inverse * ky[None, :], eps_inverse * (-kx[None, :])), dim=1
        )
        del eps_inverse
        p_matrix = torch.cat((kx[:, None] * a_block, ky[:, None] * a_block), dim=0)
        del a_block
        p_matrix.diagonal(offset=count).add_(1.0)
        p_matrix.diagonal(offset=-count).sub_(1.0)
        # q_base = -transverse_eps @ p_base is a signed column-block swap of
        # ``transverse_eps``; left @ q_right contributes four diagonal blocks.
        q_matrix = torch.cat(
            (transverse_eps[:, count:], -transverse_eps[:, :count]), dim=1
        )
        del transverse_eps
        q_matrix.diagonal()[:count].sub_(kx * ky)
        q_matrix.diagonal()[count:].add_(ky * kx)
        q_matrix.diagonal(offset=count).add_(kx * kx)
        q_matrix.diagonal(offset=-count).sub_(ky * ky)
        product = p_matrix @ q_matrix
        del q_matrix
        eigenvalues, w_layer = eigendecomposition(
            product,
            eig_backward,
            eps_relative=eig_eps_relative,
            eps_absolute=eig_eps_absolute,
        )
        del product
        q_layer = outgoing_sqrt(eigenvalues)
        # W @ diag(q) is a column scaling.
        v_layer = torch.linalg.solve(p_matrix, w_layer * q_layer[None, :])
    else:
        left = torch.cat((kx_matrix, ky_matrix), dim=0)
        p_right = torch.cat((ky_matrix, -kx_matrix), dim=1)
        q_right = torch.cat((-ky_matrix, kx_matrix), dim=1)
        p_base = _block2(zeros, identity, -identity, zeros)
        # ``transverse_eps`` acts on [-Ey, Ex].  Right multiplication by
        # ``-p_base`` rotates [Ex, Ey] into that ordering.
        q_base = -transverse_eps @ p_base
        p_matrix = p_base + left @ torch.linalg.solve(eps_conv, p_right)
        q_matrix = q_base + left @ q_right
        eigenvalues, w_layer = eigendecomposition(
            p_matrix @ q_matrix,
            eig_backward,
            eps_relative=eig_eps_relative,
            eps_absolute=eig_eps_absolute,
        )
        q_layer = outgoing_sqrt(eigenvalues)
        v_layer = torch.linalg.solve(p_matrix, w_layer @ torch.diag(q_layer))
    tangent_coverage = torch.as_tensor(
        0.0, dtype=eps_yx.real.dtype, device=eps_yx.device
    )
    if generated_tangent is not None:
        tx, ty = generated_tangent
        resolved = (torch.abs(tx) ** 2 + torch.abs(ty) ** 2) > torch.finfo(tx.real.dtype).eps
        tangent_coverage = torch.mean(resolved.to(dtype=eps_yx.real.dtype))
    if FAST_DIAGNOSTICS:
        real_dtype = eps_yx.real.dtype
        gap = (_minimum_gap(eigenvalues.detach()) if GAP_ONLY_DIAGNOSTICS
               else torch.as_tensor(float("inf"), dtype=real_dtype, device=eps_yx.device))
        return (
            q_layer,
            w_layer,
            v_layer,
            gap,
            torch.as_tensor(0.0, dtype=real_dtype, device=eps_yx.device),
            tangent_coverage,
        )
    return (
        q_layer,
        w_layer,
        v_layer,
        _minimum_gap(eigenvalues),
        torch.linalg.cond(w_layer),
        tangent_coverage,
    )


def solve_vector_2d(
    eps_yx: torch.Tensor,
    thickness_nm: float,
    wavelength_nm: float,
    lattice_nm: Tuple[float, float],
    incident_index: complex,
    output_index: complex,
    half_orders: Tuple[int, int],
    theta_deg: float = 0.0,
    phi_deg: float = 0.0,
    incident_jones: Tuple[complex, complex] = (1.0 + 0j, 0.0 + 0j),
    basis_orders: Optional[Sequence[Tuple[int, int]]] = None,
    factorization: str = FFT_BASELINE,
    normal_smoothing_sigma_px: float = 1.0,
    detach_tangent_field: bool = False,
    tangent_field: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    eig_backward: str = "default",
    eig_eps_relative: float = 1e-12,
    eig_eps_absolute: float = 1e-24,
) -> ScatteringResult:
    """Solve one finite patterned layer between two homogeneous half spaces.

    This compatibility wrapper delegates to :func:`solve_multilayer_vector_2d`.
    The incident and output media are already included as semi-infinite layers;
    ``eps_yx`` describes the single finite layer between them.
    """
    return solve_multilayer_vector_2d(
        [eps_yx],
        [thickness_nm],
        wavelength_nm,
        lattice_nm,
        incident_index,
        output_index,
        half_orders,
        theta_deg=theta_deg,
        phi_deg=phi_deg,
        incident_jones=incident_jones,
        basis_orders=basis_orders,
        factorization=factorization,
        normal_smoothing_sigma_px=normal_smoothing_sigma_px,
        detach_tangent_field=detach_tangent_field,
        tangent_fields=None if tangent_field is None else [tangent_field],
        eig_backward=eig_backward,
        eig_eps_relative=eig_eps_relative,
        eig_eps_absolute=eig_eps_absolute,
    )


def solve_multilayer_vector_2d(
    eps_layers_yx: Sequence[torch.Tensor],
    thicknesses_nm: Sequence[float],
    wavelength_nm: float,
    lattice_nm: Tuple[float, float],
    incident_index: complex,
    output_index: complex,
    half_orders: Tuple[int, int],
    theta_deg: float = 0.0,
    phi_deg: float = 0.0,
    incident_jones: Tuple[complex, complex] = (1.0 + 0j, 0.0 + 0j),
    basis_orders: Optional[Sequence[Tuple[int, int]]] = None,
    factorization: str = FFT_BASELINE,
    normal_smoothing_sigma_px: float = 1.0,
    detach_tangent_field: bool = False,
    tangent_fields: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
    eig_backward: str = "default",
    eig_eps_relative: float = 1e-12,
    eig_eps_absolute: float = 1e-24,
) -> ScatteringResult:
    """Solve a stack of finite isotropic 2D layers.

    ``eps_layers_yx`` is ordered from the incident side toward the output side.
    Every entry is a sampled relative-permittivity unit cell with shape
    ``[Ny, Nx]``.  A uniform film is represented by a constant 2D tensor.  The
    two refractive-index arguments describe semi-infinite exterior media and
    must not be repeated in ``eps_layers_yx``.
    """
    if not eps_layers_yx or len(eps_layers_yx) != len(thicknesses_nm):
        raise ValueError("provide one thickness for every finite 2D layer")
    if any(layer.ndim != 2 for layer in eps_layers_yx):
        raise ValueError("every epsilon layer must have shape [Ny, Nx]")
    if any(float(thickness) < 0 for thickness in thicknesses_nm):
        raise ValueError("layer thicknesses must be non-negative")
    if wavelength_nm <= 0 or any(value <= 0 for value in lattice_nm):
        raise ValueError("wavelength and lattice periods must be positive")
    formulation = canonical_factorization(factorization, dimensions=2)
    if normal_smoothing_sigma_px < 0:
        raise ValueError("normal_smoothing_sigma_px must be non-negative")
    if tangent_fields is not None and len(tangent_fields) != len(eps_layers_yx):
        raise ValueError("provide one tangent field for every finite layer")
    if tangent_fields is not None and formulation == FFT_BASELINE:
        raise ValueError("tangent fields are only valid for vector factorization")
    mx, my = half_orders
    if mx < 0 or my < 0:
        raise ValueError("Fourier half-orders must be non-negative")
    orders = (
        [(m, n) for m in range(-mx, mx + 1) for n in range(-my, my + 1)]
        if basis_orders is None
        else [tuple(order) for order in basis_orders]
    )
    if (0, 0) not in orders or len(set(orders)) != len(orders):
        raise ValueError("basis_orders must be unique and contain (0, 0)")

    device = eps_layers_yx[0].device
    if any(layer.device != device for layer in eps_layers_yx):
        raise ValueError("all epsilon layers must be on the same device")
    use_complex128 = any(
        layer.dtype in (torch.float64, torch.complex128) for layer in eps_layers_yx
    )
    dtype = torch.complex128 if use_complex128 else torch.complex64
    layers = [layer.to(dtype=dtype, device=device) for layer in eps_layers_yx]
    count = len(orders)
    n_in = torch.as_tensor(incident_index, dtype=dtype, device=device)
    n_out = torch.as_tensor(output_index, dtype=dtype, device=device)
    theta, phi = math.radians(theta_deg), math.radians(phi_deg)
    base = n_in.real * math.sin(theta)
    kx = torch.as_tensor(
        [base * math.cos(phi) + m * wavelength_nm / lattice_nm[0] for m, _ in orders],
        dtype=dtype,
        device=device,
    )
    ky = torch.as_tensor(
        [base * math.sin(phi) + n * wavelength_nm / lattice_nm[1] for _, n in orders],
        dtype=dtype,
        device=device,
    )
    kx_matrix, ky_matrix = torch.diag(kx), torch.diag(ky)
    identity = torch.eye(count, dtype=dtype, device=device)
    zeros = torch.zeros_like(identity)
    kz_in, v_in = _external_v(kx, ky, n_in * n_in)
    kz_out, v_out = _external_v(kx, ky, n_out * n_out)
    external_w = torch.eye(2 * count, dtype=dtype, device=device)

    # A2 opt-in dispatch (see module flags).  ``_iface``/``_star``/``_prop``
    # select the memory-efficient variants and optionally checkpoint each
    # S-matrix step so backward recomputes it instead of storing LU factors
    # and solutions.  The eigendecomposition is never inside a checkpoint.
    iface_fn = interface_smatrix_direct if MEMORY_EFFICIENT else interface_smatrix
    star_fn = redheffer_star_fused if MEMORY_EFFICIENT else redheffer_star
    use_checkpoint = (
        CHECKPOINT_SMATRIX
        and torch.is_grad_enabled()
        and any(layer.requires_grad for layer in layers)
    )

    def _iface_blocks(w_left, v_left, w_right, v_right):
        s = iface_fn(w_left, v_left, w_right, v_right)
        return s.s11, s.s12, s.s21, s.s22

    def _star_blocks(l11, l12, l21, l22, r11, r12, r21, r22):
        s = star_fn(SMatrix(l11, l12, l21, l22), SMatrix(r11, r12, r21, r22))
        return s.s11, s.s12, s.s21, s.s22

    def _prop_blocks(s11, s12, s21, s22, propagator):
        s = propagation_star(SMatrix(s11, s12, s21, s22), propagator)
        return s.s11, s.s12, s.s21, s.s22

    def _iface(w_left, v_left, w_right, v_right):
        if use_checkpoint:
            return SMatrix(*_torch_checkpoint(
                _iface_blocks, w_left, v_left, w_right, v_right,
                preserve_rng_state=False))
        return iface_fn(w_left, v_left, w_right, v_right)

    def _star(left_s, right_s):
        if use_checkpoint:
            return SMatrix(*_torch_checkpoint(
                _star_blocks,
                left_s.s11, left_s.s12, left_s.s21, left_s.s22,
                right_s.s11, right_s.s12, right_s.s21, right_s.s22,
                preserve_rng_state=False))
        return star_fn(left_s, right_s)

    def _prop(s, propagator):
        if not MEMORY_EFFICIENT:
            return _star(s, transparent_propagation(propagator))
        if use_checkpoint:
            return SMatrix(*_torch_checkpoint(
                _prop_blocks, s.s11, s.s12, s.s21, s.s22, propagator,
                preserve_rng_state=False))
        return propagation_star(s, propagator)

    current_w, current_v = external_w, v_in
    global_s = None
    layer_modes = []
    eigen_gaps = []
    modal_conditions = []
    tangent_coverages = []
    if tangent_fields is None:
        per_layer_tangent_fields = [None] * len(layers)
    else:
        per_layer_tangent_fields = list(tangent_fields)
    for eps_yx, thickness_nm, tangent_field in zip(
        layers, thicknesses_nm, per_layer_tangent_fields
    ):
        (
            q_layer,
            w_layer,
            v_layer,
            eigen_gap,
            modal_condition,
            tangent_coverage,
        ) = _layer_modes(
            eps_yx,
            half_orders,
            None if basis_orders is None else orders,
            lattice_nm,
            kx,
            ky,
            kx_matrix,
            ky_matrix,
            identity,
            zeros,
            formulation,
            normal_smoothing_sigma_px,
            detach_tangent_field,
            tangent_field,
            eig_backward,
            eig_eps_relative,
            eig_eps_absolute,
        )
        boundary = _iface(current_w, current_v, w_layer, v_layer)
        global_s = boundary if global_s is None else _star(global_s, boundary)
        phase = torch.exp(1j * (2 * math.pi / wavelength_nm) * q_layer * thickness_nm)
        global_s = _prop(global_s, phase)
        current_w, current_v = w_layer, v_layer
        if EXPORT_LAYER_MODES:
            layer_modes.append((q_layer, w_layer, v_layer))
        eigen_gaps.append(eigen_gap)
        modal_conditions.append(modal_condition)
        tangent_coverages.append(tangent_coverage)
    exit_interface = _iface(current_w, current_v, external_w, v_out)
    global_s = _star(global_s, exit_interface)
    incident = torch.zeros(2 * count, dtype=dtype, device=device)
    zero_index = orders.index((0, 0))
    jones = torch.as_tensor(incident_jones, dtype=dtype, device=device)
    jones = jones / torch.clamp(torch.linalg.vector_norm(jones), min=1e-15)
    incident[zero_index] = jones[0]
    incident[count + zero_index] = jones[1]
    reflected = global_s.s11 @ incident
    transmitted = global_s.s21 @ incident
    incident_h = v_in @ incident
    incident_power = torch.real(
        incident[zero_index] * torch.conj(incident_h[count + zero_index])
        - incident[count + zero_index] * torch.conj(incident_h[zero_index])
    )
    propagating_in = (kz_in.real > 0) & (torch.abs(kz_in.imag) < 1e-9)
    propagating_out = (kz_out.real > 0) & (torch.abs(kz_out.imag) < 1e-9)
    power_r = _order_power(reflected, v_in, count, incident_power, propagating_in)
    power_t = _order_power(transmitted, v_out, count, incident_power, propagating_out)
    total_r, total_t = torch.sum(power_r), torch.sum(power_t)
    absorption = 1.0 - total_r - total_t
    lossless = (
        all(bool(torch.max(torch.abs(layer.imag)) < 1e-12) for layer in layers)
        and bool(torch.abs(n_in.imag) < 1e-12)
        and bool(torch.abs(n_out.imag) < 1e-12)
    )
    energy_residual = torch.abs(total_r + total_t - 1.0) if lossless else torch.zeros_like(total_r)
    ex_t, ey_t = transmitted[:count], transmitted[count:]
    ex_r, ey_r = reflected[:count], reflected[count:]
    co_t = ex_t * torch.conj(jones[0]) + ey_t * torch.conj(jones[1])
    co_r = ex_r * torch.conj(jones[0]) + ey_r * torch.conj(jones[1])
    diagnostics: Dict[str, object] = {
        "half_orders": half_orders,
        "basis_kind": "rectangular" if basis_orders is None else "custom",
        "basis_terms": count,
        "dtype": str(dtype),
        "formulation": "isotropic_multilayer_vector_pq",
        "factorization": formulation,
        "normal_smoothing_sigma_px": normal_smoothing_sigma_px,
        "detach_tangent_field": detach_tangent_field,
        "eig_backward": eig_backward,
        "eig_eps_relative": eig_eps_relative,
        "eig_eps_absolute": eig_eps_absolute,
        "layer_tangent_coverages": torch.stack(tangent_coverages),
        "min_eigen_gap": torch.min(torch.stack(eigen_gaps)),
        "modal_condition": torch.max(torch.stack(modal_conditions)),
        "layer_min_eigen_gaps": torch.stack(eigen_gaps),
        "layer_modal_conditions": torch.stack(modal_conditions),
        "layer_count": len(layers),
        "interface_count": len(layers) + 1,
        "boundary_conditions": "tangential_E_and_H_continuity",
        "incident_power": incident_power,
        "propagating_input_orders": int(torch.sum(propagating_in).item()),
        "propagating_output_orders": int(torch.sum(propagating_out).item()),
    }
    return ScatteringResult(
        orders=orders,
        complex_t=co_t,
        complex_r=co_r,
        power_t=power_t,
        power_r=power_r,
        total_T=total_t,
        total_R=total_r,
        absorption=absorption,
        energy_residual=energy_residual,
        diagnostics=diagnostics,
        fields=dict(
            {"tangential_t": torch.stack((ex_t, ey_t), dim=-1),
             "tangential_r": torch.stack((ex_r, ey_r), dim=-1)},
            **({"layer_modes": layer_modes, "v_in": v_in, "v_out": v_out,
                "kx": kx, "ky": ky, "kz_in": kz_in, "kz_out": kz_out,
                "incident": incident, "thicknesses_nm": list(thicknesses_nm)}
               if EXPORT_LAYER_MODES else {}),
        ),
    )
