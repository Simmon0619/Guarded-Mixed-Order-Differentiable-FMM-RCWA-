"""Stable scattering blocks and Redheffer composition."""

from dataclasses import dataclass

import torch


@dataclass
class SMatrix:
    s11: torch.Tensor
    s12: torch.Tensor
    s21: torch.Tensor
    s22: torch.Tensor

    def __post_init__(self) -> None:
        shape = self.s11.shape
        if self.s11.ndim < 2 or shape[-1] != shape[-2]:
            raise ValueError("S blocks must be square matrices")
        if any(block.shape != shape for block in (self.s12, self.s21, self.s22)):
            raise ValueError("all S blocks must have identical shapes")


def redheffer_star(left: SMatrix, right: SMatrix) -> SMatrix:
    """Cascade `left` followed by `right` without explicit inverses."""
    n = left.s11.shape[-1]
    eye = torch.eye(n, dtype=left.s11.dtype, device=left.s11.device)
    d_right = torch.linalg.solve(eye - right.s11 @ left.s22, right.s11 @ left.s21)
    t_right = torch.linalg.solve(eye - right.s11 @ left.s22, right.s12)
    d_left = torch.linalg.solve(eye - left.s22 @ right.s11, left.s21)
    t_left = torch.linalg.solve(eye - left.s22 @ right.s11, left.s22 @ right.s12)
    return SMatrix(
        s11=left.s11 + left.s12 @ d_right,
        s12=left.s12 @ t_right,
        s21=right.s21 @ d_left,
        s22=right.s22 + right.s21 @ t_left,
    )


def transparent_propagation(propagator: torch.Tensor) -> SMatrix:
    if propagator.ndim != 1:
        raise ValueError("propagator must be a vector of modal phase factors")
    diagonal = torch.diag(propagator)
    zeros = torch.zeros_like(diagonal)
    return SMatrix(zeros, diagonal, diagonal, zeros)


def propagation_star(s: SMatrix, propagator: torch.Tensor) -> SMatrix:
    """Elementwise equivalent of ``redheffer_star(s, transparent_propagation(p))``.

    With the right factor ``(0, D, D, 0)`` where ``D = diag(propagator)`` the
    generic star product reduces exactly to ``(s11, s12 D, D s21, D s22 D)``:
    the four linear solves all have identity left-hand sides.  This variant
    performs the reduction as broadcasted row/column scalings, so it neither
    factorizes any matrix nor materializes ``D``.  Same mathematics, different
    floating-point operation order -- opt-in only (A2, 2026-09-05).
    """
    if propagator.ndim != 1:
        raise ValueError("propagator must be a vector of modal phase factors")
    return SMatrix(
        s11=s.s11,
        s12=s.s12 * propagator[None, :],
        s21=propagator[:, None] * s.s21,
        s22=propagator[:, None] * s.s22 * propagator[None, :],
    )


def redheffer_star_fused(left: SMatrix, right: SMatrix) -> SMatrix:
    """``redheffer_star`` with one LU factorization per repeated left-hand side.

    The generic version solves twice against ``I - right.s11 @ left.s22`` and
    twice against ``I - left.s22 @ right.s11``, factorizing each matrix twice.
    Here the two right-hand sides are concatenated so each left-hand side is
    factorized once.  Same mathematics, different floating-point operation order -- opt-in
    only (A2, 2026-09-05).
    """
    n = left.s11.shape[-1]
    eye = torch.eye(n, dtype=left.s11.dtype, device=left.s11.device)
    lhs = eye - right.s11 @ left.s22
    rhs = torch.cat((right.s11 @ left.s21, right.s12), dim=-1)
    solution = torch.linalg.solve(lhs, rhs)
    del lhs, rhs
    d_right, t_right = solution[..., :n], solution[..., n:]
    lhs = eye - left.s22 @ right.s11
    rhs = torch.cat((left.s21, left.s22 @ right.s12), dim=-1)
    solution2 = torch.linalg.solve(lhs, rhs)
    del lhs, rhs
    d_left, t_left = solution2[..., :n], solution2[..., n:]
    return SMatrix(
        s11=left.s11 + left.s12 @ d_right,
        s12=left.s12 @ t_right,
        s21=right.s21 @ d_left,
        s22=right.s22 + right.s21 @ t_left,
    )


def interface_smatrix_direct(
    w_left: torch.Tensor,
    v_left: torch.Tensor,
    w_right: torch.Tensor,
    v_right: torch.Tensor,
) -> SMatrix:
    """``interface_smatrix`` with the right-hand side sliced from the system.

    The generic right-hand side ``[[-W_l, W_r], [-V_l, -V_r]]`` equals the
    system matrix with its top block row negated, so it is assembled from the
    already-built system instead of from four fresh concatenations.  This
    halves the transient assembly footprint of the 4Nx4N solve; the solve
    itself is unchanged.  Same mathematics, different floating-point operation
    order -- opt-in only (A2, 2026-09-05).
    """
    n = w_left.shape[-1]
    if any(matrix.shape != (n, n) for matrix in (v_left, w_right, v_right)):
        raise ValueError("modal basis matrices must share an NxN shape")
    system = torch.cat(
        (
            torch.cat((w_left, -w_right), dim=-1),
            torch.cat((-v_left, -v_right), dim=-1),
        ),
        dim=-2,
    )
    rhs = torch.cat((-system[:n, :], system[n:, :]), dim=-2)
    solution = torch.linalg.solve(system, rhs)
    del system, rhs
    return SMatrix(
        s11=solution[:n, :n],
        s12=solution[:n, n:],
        s21=solution[n:, :n],
        s22=solution[n:, n:],
    )


def interface_smatrix(
    w_left: torch.Tensor,
    v_left: torch.Tensor,
    w_right: torch.Tensor,
    v_right: torch.Tensor,
) -> SMatrix:
    """Match tangential fields between two modal bases.

    Fields are represented as `E=W(a+ + a-)` and `F=V(a+ - a-)`.
    """
    n = w_left.shape[-1]
    if any(matrix.shape != (n, n) for matrix in (v_left, w_right, v_right)):
        raise ValueError("modal basis matrices must share an NxN shape")
    top = torch.cat((w_left, -w_right), dim=-1)
    bottom = torch.cat((-v_left, -v_right), dim=-1)
    system = torch.cat((top, bottom), dim=-2)
    rhs_left = torch.cat((-w_left, -v_left), dim=-2)
    rhs_right = torch.cat((w_right, -v_right), dim=-2)
    solution = torch.linalg.solve(system, torch.cat((rhs_left, rhs_right), dim=-1))
    return SMatrix(
        s11=solution[:n, :n],
        s12=solution[:n, n:],
        s21=solution[n:, :n],
        s22=solution[n:, n:],
    )

