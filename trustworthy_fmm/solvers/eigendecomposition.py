"""Optional, explicitly selected eigendecomposition backward stabilizers.

The forward solve is always ``torch.linalg.eig``.  The custom backward follows
TORCWA's Lorentzian replacement of inverse eigenvalue gaps.  ``lorentzian_scaled``
uses FMMAX's scale-aware broadening magnitude, avoiding a units-dependent fixed
epsilon.  Neither approximation is enabled silently: Gate C must demonstrate a
systematic, eig-risk-correlated failure before a stabilized method is selected.
"""

import os
from typing import Tuple

import torch


DEFAULT_EIG_BACKWARD = "default"
LORENTZIAN_FIXED = "lorentzian_fixed"
LORENTZIAN_SCALED = "lorentzian_scaled"

# Where ``torch.linalg.eig`` runs.  On this stack (torch 1.13.0 / CUDA 11.7 /
# Windows) the CUDA path (MAGMA geev) leaks host memory on every call -- about
# 0.4x the matrix size, i.e. +14.7 MB per (32,8) solve and ~12 GB over an
# 800-step descent (S6-1a, 2026-09-11) -- and neither gc nor empty_cache
# returns it.  The CPU path (LAPACK) does not leak and costs ~1.3x the eig wall.
# "cpu": move the matrix to the host for the decomposition and the eigenpairs
# back (autograd tracks both copies, so gradients are unchanged); "native":
# decompose where the matrix lives (the pre-S6 behaviour).
EIG_DEVICE = os.environ.get("TFMM_EIG_DEVICE", "cpu")


def _eig(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if EIG_DEVICE == "cpu" and matrix.is_cuda:
        eigenvalues, eigenvectors = torch.linalg.eig(matrix.cpu())
        return eigenvalues.to(matrix.device), eigenvectors.to(matrix.device)
    return torch.linalg.eig(matrix)


class _LorentzianEig(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        matrix: torch.Tensor,
        eps_relative: float,
        eps_absolute: float,
        scaled: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eigenvalues, eigenvectors = _eig(matrix)
        ctx.save_for_backward(eigenvalues, eigenvectors)
        ctx.input_was_complex = torch.is_complex(matrix)
        ctx.eps_relative = float(eps_relative)
        ctx.eps_absolute = float(eps_absolute)
        ctx.scaled = bool(scaled)
        return eigenvalues, eigenvectors

    @staticmethod
    def backward(ctx, grad_eigenvalues, grad_eigenvectors):
        eigenvalues, eigenvectors = ctx.saved_tensors
        if grad_eigenvalues is None:
            grad_eigenvalues = torch.zeros_like(eigenvalues)
        if grad_eigenvectors is None:
            grad_eigenvectors = torch.zeros_like(eigenvectors)
        delta = eigenvalues.unsqueeze(-2) - eigenvalues.unsqueeze(-1)
        if ctx.scaled:
            range_squared = torch.amax(torch.abs(delta) ** 2, dim=(-2, -1), keepdim=True)
            epsilon = torch.clamp(
                ctx.eps_relative * range_squared, min=ctx.eps_absolute
            )
        else:
            epsilon = torch.as_tensor(
                ctx.eps_absolute,
                dtype=eigenvalues.real.dtype,
                device=eigenvalues.device,
            )
        inverse_gap = torch.conj(delta) / (torch.abs(delta) ** 2 + epsilon)
        indices = torch.arange(eigenvalues.shape[-1], device=eigenvalues.device)
        inverse_gap[..., indices, indices] = 0.0
        eigenvectors_h = torch.conj(eigenvectors.transpose(-2, -1))
        vector_term = torch.conj(inverse_gap) * (
            eigenvectors_h @ grad_eigenvectors
        )
        right_hand_side = torch.diag_embed(grad_eigenvalues) + vector_term
        gradient = torch.linalg.solve(eigenvectors_h, right_hand_side) @ eigenvectors_h
        if not ctx.input_was_complex:
            gradient = gradient.real
        return gradient, None, None, None


def eigendecomposition(
    matrix: torch.Tensor,
    method: str = DEFAULT_EIG_BACKWARD,
    *,
    eps_relative: float = 1e-12,
    eps_absolute: float = 1e-24,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return eigenpairs with an auditable choice of backward convention."""
    method = str(method).lower()
    if method == DEFAULT_EIG_BACKWARD:
        return _eig(matrix)
    if eps_relative <= 0 or eps_absolute <= 0:
        raise ValueError("eigendecomposition broadening values must be positive")
    if method == LORENTZIAN_FIXED:
        return _LorentzianEig.apply(matrix, eps_relative, eps_absolute, False)
    if method == LORENTZIAN_SCALED:
        return _LorentzianEig.apply(matrix, eps_relative, eps_absolute, True)
    raise ValueError("unknown eig backward method {!r}".format(method))

