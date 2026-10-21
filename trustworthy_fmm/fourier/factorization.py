"""Fourier-factorization helpers for discontinuous isotropic media.

The baseline formulation uses Laurent products everywhere.  ``LI_INVERSE``
applies Li's inverse rule to the electric-field component normal to a 1D
interface.  ``NORMAL`` builds the corresponding 2D transverse permittivity
matrix from a periodic tangent vector field.

The vector implementations follow the E2 construction summarized in Liu and
Fan (2012) and used by FMMAX.  ``NORMAL`` uses a real unit tangent field, while
``JONES`` maps the same boundary-aligned field to a complex Jones field.
``JONES_DIRECT_FOURIER`` additionally optimizes the retained complex Fourier
coefficients of that field.  Its quadratic weighted-alignment problem is solved
directly here; this is algebraically equivalent to the single Newton step used
by FMMAX, without constructing an autograd Hessian.
"""

from typing import Optional, Sequence, Tuple

import torch

from trustworthy_fmm.fourier.convolution import (
    convolution_matrix_1d,
    convolution_matrix_2d,
    convolution_matrix_orders_2d,
)


FFT_BASELINE = "fft_baseline"
LI_INVERSE = "li_inverse"
NORMAL = "normal"
JONES = "jones"
JONES_DIRECT_FOURIER = "jones_direct_fourier"

FOURIER_LOSS_WEIGHT_JONES_DIRECT = 0.05


def canonical_factorization(name: str, *, dimensions: int) -> str:
    """Return a validated canonical factorization name.

    Short aliases are accepted at the API boundary, while diagnostics always
    store one of the canonical constants above.
    """
    aliases = {
        "fft": FFT_BASELINE,
        "baseline": FFT_BASELINE,
        FFT_BASELINE: FFT_BASELINE,
        "li": LI_INVERSE,
        "inverse": LI_INVERSE,
        LI_INVERSE: LI_INVERSE,
        "normal_vector": NORMAL,
        NORMAL: NORMAL,
        "jones_vector": JONES,
        JONES: JONES,
        "jones_fourier": JONES_DIRECT_FOURIER,
        JONES_DIRECT_FOURIER: JONES_DIRECT_FOURIER,
    }
    try:
        result = aliases[str(name).lower()]
    except KeyError as error:
        raise ValueError("unknown Fourier factorization {!r}".format(name)) from error
    allowed = (
        {FFT_BASELINE, LI_INVERSE}
        if dimensions == 1
        else {FFT_BASELINE, NORMAL, JONES, JONES_DIRECT_FOURIER}
    )
    if result not in allowed:
        raise ValueError(
            "factorization {!r} is not available for {}D media".format(result, dimensions)
        )
    return result


def effective_permittivity_1d(
    eps_profile: torch.Tensor,
    half_order: int,
    factorization: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(eps_z, eps_transverse)`` matrices for scalar 1D RCWA.

    ``eps_z`` is the ordinary Fourier convolution matrix used to eliminate the
    longitudinal field.  For TM polarization, the inverse-rule transverse
    matrix is ``inv(conv(1 / eps))``.  In uniform media both formulations are
    exactly identical up to floating-point roundoff.
    """
    formulation = canonical_factorization(factorization, dimensions=1)
    eps_z = convolution_matrix_1d(eps_profile, half_order)
    if formulation == FFT_BASELINE:
        return eps_z, eps_z
    inverse_eps = convolution_matrix_1d(1.0 / eps_profile, half_order)
    identity = torch.eye(
        inverse_eps.shape[-1], dtype=inverse_eps.dtype, device=inverse_eps.device
    )
    return eps_z, torch.linalg.solve(inverse_eps, identity)


def _periodic_gaussian_smooth_2d(values: torch.Tensor, sigma_px: float) -> torch.Tensor:
    if sigma_px < 0:
        raise ValueError("normal smoothing sigma must be non-negative")
    if sigma_px == 0:
        return values
    ny, nx = values.shape
    fy = torch.fft.fftfreq(ny, dtype=values.dtype, device=values.device)
    fx = torch.fft.fftfreq(nx, dtype=values.dtype, device=values.device)
    squared_frequency = fy[:, None] ** 2 + fx[None, :] ** 2
    gaussian = torch.exp(-2.0 * torch.pi ** 2 * sigma_px ** 2 * squared_frequency)
    return torch.fft.ifftn(torch.fft.fftn(values) * gaussian).real


def periodic_tangent_field(
    eps_yx: torch.Tensor,
    lattice_nm: Tuple[float, float],
    *,
    smoothing_sigma_px: float = 1.0,
    relative_floor: float = 1e-12,
    detach: bool = False,
    representation: str = NORMAL,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a periodic tangent field from a sampled permittivity map.

    The material gradient is computed with centered periodic differences in
    physical coordinates.  A Fourier-domain Gaussian makes the direction field
    continuous across the unit-cell boundary.  The tangent is perpendicular to
    the smoothed gradient.  ``representation='normal'`` normalizes the real
    field elementwise.  ``representation='jones'`` retains the boundary-strength
    information and continuously becomes circularly polarized where a real
    tangent direction is undefined.  This removes the forced singularity of a
    real unit vector field around a closed curve.

    By default the field remains differentiable with respect to ``eps_yx``.
    Setting ``detach=True`` freezes only the geometry-derived direction field,
    which can be useful for controlled ablations but must be reported because it
    changes the gradient convention.
    """
    if eps_yx.ndim != 2:
        raise ValueError("eps_yx must have shape [Ny, Nx]")
    if any(float(length) <= 0 for length in lattice_nm):
        raise ValueError("lattice periods must be positive")
    if relative_floor <= 0:
        raise ValueError("relative_floor must be positive")
    representation = str(representation).lower()
    if representation not in (NORMAL, JONES):
        raise ValueError("representation must be 'normal' or 'jones'")
    source = eps_yx.real
    if detach:
        source = source.detach()
    ny, nx = source.shape
    # A truly 1D grating has one globally defined tangent direction.  Extending
    # that direction through the homogeneous parts of the cell is essential for
    # recovering Li's 1D inverse rule exactly (and avoids arbitrary zero-field
    # regions away from interfaces).
    variation_x = torch.amax(torch.abs(source - torch.roll(source, 1, dims=1))).detach()
    variation_y = torch.amax(torch.abs(source - torch.roll(source, 1, dims=0))).detach()
    variation_scale = max(float(variation_x), float(variation_y), torch.finfo(source.dtype).eps)
    if float(variation_y) <= relative_floor * variation_scale:
        return torch.zeros_like(source), torch.ones_like(source)
    if float(variation_x) <= relative_floor * variation_scale:
        return torch.ones_like(source), torch.zeros_like(source)
    dx = float(lattice_nm[0]) / nx
    dy = float(lattice_nm[1]) / ny
    grad_x = (torch.roll(source, -1, dims=1) - torch.roll(source, 1, dims=1)) / (2.0 * dx)
    grad_y = (torch.roll(source, -1, dims=0) - torch.roll(source, 1, dims=0)) / (2.0 * dy)
    grad_x = _periodic_gaussian_smooth_2d(grad_x, smoothing_sigma_px)
    grad_y = _periodic_gaussian_smooth_2d(grad_y, smoothing_sigma_px)
    tangent_x = -grad_y
    tangent_y = grad_x
    magnitude = torch.sqrt(tangent_x ** 2 + tangent_y ** 2)
    scale = torch.amax(magnitude).detach()
    floor = torch.clamp(scale * relative_floor, min=torch.finfo(source.dtype).eps)
    if representation == JONES:
        # Jones normalization from Liu and Fan (2012), also used by FMMAX.  The
        # raw field is first normalized globally, so its magnitude records how
        # strongly the local material gradient resolves an interface direction.
        # At unit magnitude the result is the real tangent.  At zero magnitude
        # it becomes circularly polarized and therefore has no preferred real
        # direction or unavoidable point singularity.
        scale_safe = torch.clamp(scale, min=torch.finfo(source.dtype).eps)
        raw_x = tangent_x / scale_safe
        raw_y = tangent_y / scale_safe
        local_magnitude = torch.sqrt(raw_x ** 2 + raw_y ** 2)
        resolved = local_magnitude > floor / scale_safe
        local_safe = torch.where(
            resolved, local_magnitude, torch.ones_like(local_magnitude)
        )
        inverse_sqrt_two = torch.rsqrt(
            torch.tensor(2.0, dtype=source.dtype, device=source.device)
        )
        unit_x = torch.where(
            resolved, raw_x / local_safe, torch.ones_like(raw_x) * inverse_sqrt_two
        )
        unit_y = torch.where(
            resolved, raw_y / local_safe, torch.ones_like(raw_y) * inverse_sqrt_two
        )
        phi = torch.pi / 8.0 * (1.0 + torch.cos(torch.pi * local_magnitude))
        direction = torch.complex(unit_x, unit_y)
        direction = torch.where(
            torch.abs(direction) > relative_floor,
            direction,
            torch.ones_like(direction),
        )
        phase = torch.exp(1j * torch.angle(direction))
        jones_x = phase * (
            unit_x * torch.cos(phi) - 1j * unit_y * torch.sin(phi)
        )
        jones_y = phase * (
            unit_y * torch.cos(phi) + 1j * unit_x * torch.sin(phi)
        )
        return jones_x, jones_y
    resolved = magnitude > floor
    safe = torch.where(resolved, magnitude, torch.ones_like(magnitude))
    tangent_x = torch.where(resolved, tangent_x / safe, torch.zeros_like(tangent_x))
    tangent_y = torch.where(resolved, tangent_y / safe, torch.zeros_like(tangent_y))
    return tangent_x, tangent_y


def _orders_2d(
    half_orders: Tuple[int, int],
    basis_orders: Optional[Sequence[Tuple[int, int]]],
) -> Tuple[Tuple[int, int], ...]:
    if basis_orders is not None:
        orders = tuple((int(m), int(n)) for m, n in basis_orders)
    else:
        mx, my = half_orders
        orders = tuple(
            (m, n) for m in range(-mx, mx + 1) for n in range(-my, my + 1)
        )
    if not orders or (0, 0) not in orders or len(set(orders)) != len(orders):
        raise ValueError("Fourier orders must be unique and contain (0, 0)")
    return orders


def _fourier_reconstruct_2d(
    coefficients: torch.Tensor,
    orders: Sequence[Tuple[int, int]],
    shape_yx: Tuple[int, int],
) -> torch.Tensor:
    """Reconstruct selected Fourier coefficients with ``ifft(norm='forward')`` scaling."""
    ny, nx = shape_yx
    real_dtype = coefficients.real.dtype
    y = torch.arange(ny, dtype=real_dtype, device=coefficients.device) / ny
    x = torch.arange(nx, dtype=real_dtype, device=coefficients.device) / nx
    order_tensor = torch.as_tensor(orders, dtype=real_dtype, device=coefficients.device)
    phase = 2.0 * torch.pi * (
        y[:, None, None] * order_tensor[None, :, 1]
        + x[None, :, None] * order_tensor[None, :, 0]
    )
    basis = torch.exp(1j * phase).reshape(ny * nx, len(orders))
    trailing_shape = coefficients.shape[1:]
    field = basis @ coefficients.reshape(len(orders), -1)
    return field.reshape((ny, nx) + trailing_shape)


def _jones_normalize(field: torch.Tensor, *, near_zero: float = 1e-8) -> torch.Tensor:
    """Jones normalization of Liu and Fan (2012), including the zero-field limit."""
    magnitude = torch.sqrt(torch.sum(torch.abs(field) ** 2, dim=-1, keepdim=True))
    maximum = torch.amax(magnitude)
    maximum_safe = torch.clamp(maximum, min=torch.finfo(magnitude.dtype).eps)
    field = field / maximum_safe
    magnitude = torch.sqrt(torch.sum(torch.abs(field) ** 2, dim=-1, keepdim=True))
    near = magnitude <= near_zero
    magnitude_safe = torch.where(near, torch.ones_like(magnitude), magnitude)
    inverse_sqrt_two = torch.rsqrt(
        torch.tensor(2.0, dtype=magnitude.dtype, device=magnitude.device)
    )
    tx = torch.where(near, inverse_sqrt_two, field[..., 0:1] / magnitude_safe)
    ty = torch.where(near, inverse_sqrt_two, field[..., 1:2] / magnitude_safe)
    phi = torch.pi / 8.0 * (1.0 + torch.cos(torch.pi * magnitude))
    direction = tx + 1j * ty
    direction = torch.where(
        torch.abs(direction) <= near_zero, torch.ones_like(direction), direction
    )
    phase = torch.exp(1j * torch.angle(direction))
    jx = phase * (tx * torch.cos(phi) - 1j * ty * torch.sin(phi))
    jy = phase * (ty * torch.cos(phi) + 1j * tx * torch.sin(phi))
    return torch.cat((jx, jy), dim=-1)


def jones_direct_fourier_tangent_field(
    eps_yx: torch.Tensor,
    lattice_nm: Tuple[float, float],
    half_orders: Tuple[int, int],
    *,
    basis_orders: Optional[Sequence[Tuple[int, int]]] = None,
    fourier_loss_weight: float = FOURIER_LOSS_WEIGHT_JONES_DIRECT,
    relative_floor: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute the complex Jones-direct field in the retained Fourier basis.

    This is a PyTorch implementation of FMMAX's ``JONES_DIRECT_FOURIER``
    construction.  The material gradient is first projected into the active
    expansion.  The retained coefficients ``c`` then minimize

    ``mean(w * |B c - target|^2) + (lambda / M) * sum(|k|^2 |c|^2)``.

    The target construction is deliberately detached from ``eps_yx``.  FMMAX
    applies the same stop-gradient convention: the optimized direction field is
    an auxiliary factorization choice, not a geometry-dependent term in the
    physical derivative.
    """
    if eps_yx.ndim != 2:
        raise ValueError("eps_yx must have shape [Ny, Nx]")
    if any(float(length) <= 0 for length in lattice_nm):
        raise ValueError("lattice periods must be positive")
    if fourier_loss_weight <= 0:
        raise ValueError("fourier_loss_weight must be positive")
    if relative_floor <= 0:
        raise ValueError("relative_floor must be positive")
    orders = _orders_2d(half_orders, basis_orders)
    source = eps_yx.real.detach()
    ny, nx = source.shape
    max_m = max(abs(m) for m, _ in orders)
    max_n = max(abs(n) for _, n in orders)
    if nx < 2 * max_m + 1 or ny < 2 * max_n + 1:
        raise ValueError("unit-cell grid is too small for the Jones Fourier expansion")

    variation_x = torch.amax(torch.abs(source - torch.roll(source, 1, dims=1)))
    variation_y = torch.amax(torch.abs(source - torch.roll(source, 1, dims=0)))
    variation_scale = max(
        float(variation_x), float(variation_y), torch.finfo(source.dtype).eps
    )
    complex_dtype = torch.complex128 if source.dtype == torch.float64 else torch.complex64
    if float(variation_y) <= relative_floor * variation_scale:
        return (
            torch.zeros_like(source, dtype=complex_dtype),
            torch.ones_like(source, dtype=complex_dtype),
        )
    if float(variation_x) <= relative_floor * variation_scale:
        return (
            torch.ones_like(source, dtype=complex_dtype),
            torch.zeros_like(source, dtype=complex_dtype),
        )

    lx, ly = (float(value) for value in lattice_nm)
    grad_x = 0.5 * nx * (torch.roll(source, -1, 1) - torch.roll(source, 1, 1))
    grad_y = 0.5 * ny * (torch.roll(source, -1, 0) - torch.roll(source, 1, 0))
    grad_x = grad_x * (ly / lx) ** 0.5
    grad_y = grad_y * (lx / ly) ** 0.5
    gradient = torch.stack((grad_x, grad_y), dim=-1).to(complex_dtype)

    gradient_fft = torch.fft.fftn(gradient, dim=(0, 1), norm="forward")
    filtered_coefficients = torch.stack(
        [gradient_fft[n % ny, m % nx] for m, n in orders], dim=0
    )
    optimize_shape = (
        1 if ny == 1 else 2 * (2 * max_n + 1),
        1 if nx == 1 else 2 * (2 * max_m + 1),
    )
    filtered_gradient = _fourier_reconstruct_2d(
        filtered_coefficients, orders, optimize_shape
    )
    gradient_magnitude = torch.sqrt(
        torch.sum(torch.abs(filtered_gradient) ** 2, dim=-1, keepdim=True)
    )
    max_gradient = torch.amax(gradient_magnitude)
    max_gradient_safe = torch.clamp(
        max_gradient, min=torch.finfo(gradient_magnitude.dtype).eps
    )
    filtered_gradient = filtered_gradient / max_gradient_safe
    alignment_weight = gradient_magnitude / max_gradient_safe

    target = torch.stack(
        (filtered_gradient[..., 1], -filtered_gradient[..., 0]), dim=-1
    )
    target_magnitude = torch.sqrt(torch.sum(torch.abs(target) ** 2, dim=-1, keepdim=True))
    target = torch.where(
        target_magnitude <= 1e-8,
        target,
        target / torch.where(
            target_magnitude <= 1e-8,
            torch.ones_like(target_magnitude),
            target_magnitude,
        ),
    )
    target = _jones_normalize(target)

    opt_ny, opt_nx = optimize_shape
    real_dtype = source.dtype
    y = torch.arange(opt_ny, dtype=real_dtype, device=source.device) / opt_ny
    x = torch.arange(opt_nx, dtype=real_dtype, device=source.device) / opt_nx
    order_tensor = torch.as_tensor(orders, dtype=real_dtype, device=source.device)
    phase = 2.0 * torch.pi * (
        y[:, None, None] * order_tensor[None, :, 1]
        + x[None, :, None] * order_tensor[None, :, 0]
    )
    basis = torch.exp(1j * phase).reshape(opt_ny * opt_nx, len(orders))
    weight = alignment_weight.reshape(-1, 1).to(complex_dtype)
    target_flat = target.reshape(-1, 2).to(complex_dtype)
    point_count = opt_ny * opt_nx
    normal_matrix = torch.conj(basis.T) @ (weight * basis) / point_count
    right_hand_side = torch.conj(basis.T) @ (weight * target_flat) / point_count

    reciprocal_norm_squared = (2.0 * torch.pi) ** 2 * (
        order_tensor[:, 0] ** 2 * ly / lx
        + order_tensor[:, 1] ** 2 * lx / ly
    )
    regularizer = fourier_loss_weight / len(orders) * reciprocal_norm_squared
    normal_matrix = normal_matrix + torch.diag(regularizer.to(complex_dtype))
    try:
        optimized_coefficients = torch.linalg.solve(normal_matrix, right_hand_side)
    except RuntimeError:
        optimized_coefficients = torch.linalg.lstsq(
            normal_matrix, right_hand_side
        ).solution

    field = _fourier_reconstruct_2d(optimized_coefficients, orders, (ny, nx))
    field_magnitude = torch.sqrt(torch.sum(torch.abs(field) ** 2, dim=-1, keepdim=True))
    maximum = torch.amax(field_magnitude)
    field = field / torch.clamp(maximum, min=torch.finfo(field_magnitude.dtype).eps)
    return field[..., 0], field[..., 1]


def _convolution_2d(
    values: torch.Tensor,
    half_orders: Tuple[int, int],
    basis_orders: Optional[Sequence[Tuple[int, int]]],
) -> torch.Tensor:
    if basis_orders is None:
        return convolution_matrix_2d(values, half_orders)
    return convolution_matrix_orders_2d(values, basis_orders)


def _block2(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor
) -> torch.Tensor:
    return torch.cat((torch.cat((a, b), dim=1), torch.cat((c, d), dim=1)), dim=0)


def transverse_permittivity_2d(
    eps_yx: torch.Tensor,
    half_orders: Tuple[int, int],
    factorization: str,
    *,
    lattice_nm: Tuple[float, float],
    basis_orders: Optional[Sequence[Tuple[int, int]]] = None,
    tangent_field: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    smoothing_sigma_px: float = 1.0,
    detach_tangent_field: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
    """Build longitudinal and transverse 2D permittivity matrices.

    Returns ``(eps_z, eps_transverse, tangent_field)``.  ``eps_transverse`` acts
    on the rotated transverse field ``[-Ey, Ex]``.  The vector solver performs
    the corresponding basis rotation when inserting it into the Q matrix.
    """
    formulation = canonical_factorization(factorization, dimensions=2)
    eps_z = _convolution_2d(eps_yx, half_orders, basis_orders)
    zeros = torch.zeros_like(eps_z)
    eps_matrix = _block2(eps_z, zeros, zeros, eps_z)
    if formulation == FFT_BASELINE:
        if tangent_field is not None:
            raise ValueError("tangent_field is only valid for vector factorization")
        return eps_z, eps_matrix, None

    if tangent_field is None:
        if formulation == JONES_DIRECT_FOURIER:
            tangent_field = jones_direct_fourier_tangent_field(
                eps_yx,
                lattice_nm,
                half_orders,
                basis_orders=basis_orders,
            )
        else:
            tangent_field = periodic_tangent_field(
                eps_yx,
                lattice_nm,
                smoothing_sigma_px=smoothing_sigma_px,
                detach=detach_tangent_field,
                representation=formulation,
            )
    tx, ty = tangent_field
    if tx.shape != eps_yx.shape or ty.shape != eps_yx.shape:
        raise ValueError("tangent field components must match eps_yx shape")
    tx = tx.to(dtype=eps_yx.dtype, device=eps_yx.device)
    ty = ty.to(dtype=eps_yx.dtype, device=eps_yx.device)

    eta_hat = _convolution_2d(1.0 / eps_yx, half_orders, basis_orders)
    identity = torch.eye(eta_hat.shape[-1], dtype=eta_hat.dtype, device=eta_hat.device)
    inverse_rule_eps = torch.linalg.solve(eta_hat, identity)
    delta_hat = eps_z - inverse_rule_eps
    delta_matrix = _block2(delta_hat, zeros, zeros, delta_hat)

    denom = torch.abs(tx) ** 2 + torch.abs(ty) ** 2
    resolved = denom > torch.finfo(denom.dtype).eps
    denom_safe = torch.where(resolved, denom, torch.ones_like(denom))
    p_yy = torch.where(resolved, torch.abs(tx) ** 2 / denom_safe, torch.zeros_like(denom))
    p_yx = torch.where(resolved, torch.conj(tx) * ty / denom_safe, torch.zeros_like(tx))
    p_xy = torch.where(resolved, tx * torch.conj(ty) / denom_safe, torch.zeros_like(tx))
    p_xx = torch.where(resolved, torch.abs(ty) ** 2 / denom_safe, torch.zeros_like(denom))
    projector = _block2(
        _convolution_2d(p_yy, half_orders, basis_orders),
        _convolution_2d(p_yx, half_orders, basis_orders),
        _convolution_2d(p_xy, half_orders, basis_orders),
        _convolution_2d(p_xx, half_orders, basis_orders),
    )
    transverse = eps_matrix - delta_matrix @ projector
    # With a numerically generated vector field the finite Fourier projection
    # need not preserve Hermiticity algebraically, even when the underlying
    # material is real.  The Galerkin operator for a lossless dielectric must be
    # Hermitian; enforcing that invariant prevents the factorization itself from
    # introducing artificial gain or absorption.  Lossy media are intentionally
    # left unsymmetrized.
    if bool(torch.max(torch.abs(eps_yx.imag)).detach() < 1e-12):
        transverse = 0.5 * (transverse + torch.conj(transverse.T))
    return eps_z, transverse, (tx, ty)
