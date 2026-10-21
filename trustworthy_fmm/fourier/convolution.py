"""Fourier coefficients and convolution matrices on sampled unit cells."""

import os
from typing import Sequence, Tuple

import torch

# Opt-in fast path (E17 stage 0, 2026-08-31).  The loop-based 2D builders below
# create one autograd node per matrix ELEMENT (30,625 at (12,3)), which makes
# the backward pass dominate the whole solve (~1.2 s of a 1.45 s value+grad).
# The vectorized branch gathers the IDENTICAL elements in one advanced-indexing
# op, so forward values are bitwise unchanged; gradient values agree to
# accumulation-order (last-ulp) differences only, because backward becomes one
# scatter-add instead of a 30k-node reduction tree.  Default OFF: the frozen
# E10/E16 protocol keeps the original graph and its bitwise baseline.  Enable
# per-process via TFMM_VECTORIZED_CONV=1 or by setting VECTORIZED_CONV = True.
VECTORIZED_CONV = os.environ.get("TFMM_VECTORIZED_CONV", "0") == "1"


def centered_fourier_coefficients(samples: torch.Tensor) -> torch.Tensor:
    """Return normalized, fft-shifted coefficients over the final spatial axes.

    One-dimensional input transforms its final axis.  Two-dimensional input
    transforms its final two axes.  Any leading axes are treated as batches.
    """
    if samples.ndim < 1:
        raise ValueError("samples must have at least one spatial axis")
    dims = (-1,) if samples.ndim == 1 else (-2, -1)
    coeff = torch.fft.fftn(samples, dim=dims, norm="forward")
    return torch.fft.fftshift(coeff, dim=dims)


def convolution_matrix_1d(samples: torch.Tensor, half_order: int) -> torch.Tensor:
    if samples.ndim != 1:
        raise ValueError("the initial 1D implementation accepts a single profile")
    if half_order < 0:
        raise ValueError("half_order must be non-negative")
    count = 2 * half_order + 1
    if samples.shape[0] < 2 * count - 1:
        raise ValueError("profile grid is too small for the requested convolution differences")
    coeff = centered_fourier_coefficients(samples)
    center = samples.shape[0] // 2
    orders = torch.arange(-half_order, half_order + 1, device=samples.device)
    differences = orders[:, None] - orders[None, :]
    return coeff[center + differences]


def convolution_matrix_2d(
    samples: torch.Tensor, half_orders: Tuple[int, int]
) -> torch.Tensor:
    if samples.ndim != 2:
        raise ValueError("the initial 2D implementation accepts a single unit cell")
    mx, my = half_orders
    if mx < 0 or my < 0:
        raise ValueError("half-orders must be non-negative")
    ny, nx = samples.shape
    if nx < 4 * mx + 1 or ny < 4 * my + 1:
        raise ValueError("unit-cell grid is too small for requested order differences")
    coeff = centered_fourier_coefficients(samples)
    cx, cy = nx // 2, ny // 2
    if VECTORIZED_CONV:
        mm = torch.arange(-mx, mx + 1, device=samples.device)
        nn = torch.arange(-my, my + 1, device=samples.device)
        m_of = torch.repeat_interleave(mm, 2 * my + 1)
        n_of = nn.repeat(2 * mx + 1)
        return coeff[cy + n_of[:, None] - n_of[None, :],
                     cx + m_of[:, None] - m_of[None, :]]
    orders = [(m, n) for m in range(-mx, mx + 1) for n in range(-my, my + 1)]
    rows = []
    for mi, ni in orders:
        rows.append(
            torch.stack([coeff[cy + ni - nj, cx + mi - mj] for mj, nj in orders])
        )
    return torch.stack(rows)


def convolution_matrix_orders_2d(
    samples: torch.Tensor, orders: Sequence[Tuple[int, int]]
) -> torch.Tensor:
    """Convolution matrix for an arbitrary symmetric set of `(m, n)` orders.

    Samples are interpreted at pixel centres, consistent with FMMAX's default
    ``centered_coordinates=True`` convention.  The half-pixel phase is
    immaterial when all requested coefficient differences lie in the principal
    FFT interval, but it must be retained once periodic FFT indexing aliases a
    difference at the sampling boundary.
    """
    if samples.ndim != 2 or not orders:
        raise ValueError("samples must be 2D and orders must be non-empty")
    if len(set(orders)) != len(orders):
        raise ValueError("orders must be unique")
    ny, nx = samples.shape
    # Use periodic FFT indexing.  This matches FMMAX and makes aliasing explicit
    # when a requested expansion exceeds the independent sampled coefficients.
    coeff = torch.fft.fftn(samples, dim=(-2, -1), norm="forward")
    ky = torch.fft.fftfreq(ny, dtype=samples.real.dtype, device=samples.device)
    kx = torch.fft.fftfreq(nx, dtype=samples.real.dtype, device=samples.device)
    phase = torch.exp(-1j * torch.pi * (ky[:, None] + kx[None, :]))
    coeff = coeff * phase
    if VECTORIZED_CONV:
        m_of = torch.as_tensor([m for m, _ in orders], device=samples.device)
        n_of = torch.as_tensor([n for _, n in orders], device=samples.device)
        return coeff[(n_of[:, None] - n_of[None, :]) % ny,
                     (m_of[:, None] - m_of[None, :]) % nx]
    return torch.stack(
        [
            torch.stack([coeff[(ni - nj) % ny, (mi - mj) % nx] for mj, nj in orders])
            for mi, ni in orders
        ]
    )
