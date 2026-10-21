"""Problem configuration, optimizer schedules and the CD-bias operator.

This is the small core the workbench needs: an immutable ``ProblemConfig``
(physics + design map + optimizer hyper-parameters), the truncation-order
helper, the beta / learning-rate schedules and the periodic critical-dimension
bias used for robustness checks.
"""
import math
from dataclasses import dataclass, replace
from typing import Sequence, Tuple

import numpy as np
import torch

from trustworthy_fmm.fabrication.process import periodic_signed_distance_nm


@dataclass(frozen=True)
class ProblemConfig:
    """One design problem.  Immutable; derive variants with ``variant``."""

    # -- physics
    wavelength_nm: float = 660.0
    lattice_nm: Tuple[float, float] = (2400.0, 700.0)   # (Lx, Ly)
    ny: int = 35
    nx: int = 120
    thickness_nm: float = 800.0
    n_incident: float = 1.46
    n_output: float = 1.0
    eps_background: float = 1.0
    eps_feature: float = 3.2 ** 2
    incident_jones: Tuple[complex, complex] = (1.0 + 0j, 0.0 + 0j)

    # -- design map: latent -> filter -> tanh projection -> hard threshold
    filter_nm: float = 60.0
    threshold: float = 0.5
    beta_ladder: Tuple[float, ...] = (1.0, 64.0)
    init_mean: float = 0.5
    init_scale: float = 0.08
    init_clamp_lo: float = 0.05
    init_clamp_hi: float = 0.95

    # -- optimizer (Adam)
    adam_lr0: float = 0.02
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps: float = 1e-8

    # -- numerics
    latent_dtype: torch.dtype = torch.float64

    # -- robustness audit
    cd_bias_nm: Tuple[float, ...] = (-20.0, 20.0)

    def __post_init__(self):
        if min(self.lattice_nm) <= 0 or self.wavelength_nm <= 0 or self.thickness_nm < 0:
            raise ValueError("lattice periods and wavelength must be positive")
        if self.latent_dtype.is_complex or not self.latent_dtype.is_floating_point:
            raise ValueError("latent_dtype must be a real floating type")

    @property
    def pixel_yx(self) -> Tuple[float, float]:
        """(dy, dx) in nm."""
        return (self.lattice_nm[1] / self.ny, self.lattice_nm[0] / self.nx)

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.ny, self.nx)

    def variant(self, **changes) -> "ProblemConfig":
        return replace(self, **changes)

    def min_grid_for(self, level: Sequence[int]) -> Tuple[int, int]:
        """Smallest (ny, nx) satisfying the convolution constraint n >= 4m+1."""
        return (4 * int(level[1]) + 1, 4 * int(level[0]) + 1)

    def supports(self, level: Sequence[int]) -> bool:
        ny_min, nx_min = self.min_grid_for(level)
        return self.ny >= ny_min and self.nx >= nx_min


DEFAULT = ProblemConfig()


def n_terms(level) -> int:
    """Number of retained Fourier orders (2Mx+1)(2My+1); matrices are 2N x 2N."""
    return (2 * int(level[0]) + 1) * (2 * int(level[1]) + 1)


def cosine_lr(progress: float, cfg: ProblemConfig = DEFAULT) -> float:
    return cfg.adam_lr0 * 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def beta_at(progress: float, ladder=None, cfg: ProblemConfig = DEFAULT) -> float:
    """Exponential beta continuation from ladder[0] to ladder[-1]."""
    ladder = cfg.beta_ladder if ladder is None else ladder
    lo, hi = ladder[0], ladder[-1]
    return lo * (hi / lo) ** min(max(progress, 0.0), 1.0)


def cd_bias_mask(mask_yx, bias_nm: float, cfg: ProblemConfig = DEFAULT) -> np.ndarray:
    """Uniform periodic edge bias in nm; positive grows the feature.

    ``periodic_signed_distance_nm`` measures pixel CENTRE to the nearest
    opposite-class pixel centre, so a boundary pixel sits at |sd| = one pixel
    while its edge is half a pixel away.  Growing by b keeps every pixel with
    sd >= -b; shrinking by b must drop the pixels with sd <= b, i.e. keep
    sd > b (the old ``sd >= -b`` for negative b left the boundary pixel in
    place: -20 nm was a no-op on a 20 nm grid and one pixel short on 5 nm).
    A bias smaller than one pixel changes nothing in either direction."""
    m = np.asarray(mask_yx, dtype=bool)
    if not m.any() or m.all():
        return m.astype(np.float64)
    sd = periodic_signed_distance_nm(m, cfg.pixel_yx)
    b = float(bias_nm)
    keep = (sd >= -b) if b >= 0 else (sd > -b)
    return keep.astype(np.float64)


# ============================================================ final scheme v2
# Defaults of the guarded mixed-order ticket protocol (docs/algorithm.md §5).
# Values calibrated on the P1/P2 problems, GPU/complex64, eig on the CPU
# (experiments S6-S10); see the "how to set them for a new problem" note there.
DESCENT = {
    "steps": 219,              # README ramp length
    "beta0": 2.0,              # beta(t) = beta0 (beta_end/beta0)^(t/steps)
    "beta_end": 2000.0,
    "mix_k": 4,                # every k-th gradient at the gate order ...
    "mix_from_beta": 16.0,     # ... from the first step with beta >= this
}
GUARDS = {
    "enabled": True,
    "norm_factor": 3.0,        # trigger: |g| > factor x median of the last `window` same-order norms
    "window": 8,
    "warmup": 4,               # no check before this many same-order evaluations
    "beta_min": 8.0,           # no check below this beta
    "jitters": (0.01, -0.01),  # norm trigger: recompute at beta (1 + 1 %), then (1 - 1 %) only if that raised
    "jitters_raise": (0.01, -0.01, 0.03, -0.03),   # raised / non-finite gradient: first finite recompute wins
    "stab_cos": 0.9,           # the original is kept if cos(orig, recompute) >= this ...
    "stab_ratio": (0.5, 2.0),  # ... and |orig| / |recompute| lies inside this
    "c128_cap": 6,             # CPU complex128 fallbacks per ticket (descent order only)
    "tau_default": 1e-4,       # eigen-gap "small" flag (logged only; the gap does not predict wrong gradients)
    "tau": {"28,4": 8.3e-6, "44,6": 3.9e-6, "24,3": 6.8e-6, "40,6": 4.4e-6, "32,5": 6.0e-6, "36,5": 5.0e-6},
}
REPAIR = {
    "enabled": False,
    "trigger": "manual",       # always | numeric | manual (start it later from the Summary / History tab)
    "bar": None,               # numeric: chain only if the delivered rung-1 value is below this ...
    "optimism_max": 0.05,      # ... and the gate optimism (gate - rung 1) is below this
    "beta_mode": "fixed",      # fixed (beta_fixed) | star (max(beta*, beta_min), beta* = beta of the delivered checkpoint)
    "beta_fixed": 64.0,
    "beta_min": 64.0,
    "hops": 15,
    "stop_after": 12,          # consecutive rejections that end the chain
    "alphas": (0.1, 0.25, 0.5),
    "re_steps": 30,
    "re_k": 4,
}
CADENCE = {"enabled": False, "k_set": (2, 4, 8), "up": 0.95, "down": 0.85}   # D1: adaptive mixing cadence
RISE = {"enabled": False}                                                    # D2: online rung rise (needs D1)


def beta_geometric(b0: float, b_end: float):
    """B(t) = b0 * (b_end/b0)^(t/T): the README ramp is beta_geometric(2, 2000)."""
    def fn(step: int, steps: int) -> float:
        return float(b0) * (float(b_end) / float(b0)) ** (step / steps)
    return fn


def ckpt_schedule(steps: int, early_every: float = 0.25, late_from: float = 0.6,
                  late_every: float = 0.05):
    """'Sparse early, dense late' checkpoint steps, always incl. 0 and steps."""
    s = {0, int(steps)}
    t = early_every
    while t < late_from - 1e-9:
        s.add(int(round(t * steps)))
        t += early_every
    t = late_from
    while t < 1.0 - 1e-9:
        s.add(int(round(t * steps)))
        t += late_every
    return sorted(s)


def o_ckpts(steps: int, every: int = 12):
    """The ticket's checkpoint set: every 12 steps plus sparse-early / dense-late
    (30 points on the 219-step ramp)."""
    return sorted(set(range(0, int(steps) + 1, int(every))) | set(ckpt_schedule(int(steps))))


def first_step_at(beta_fn, steps: int, beta_target: float) -> int:
    """First step whose beta reaches ``beta_target`` (steps + 1 if never)."""
    return next((t for t in range(int(steps) + 1) if beta_fn(t, steps) >= beta_target), int(steps) + 1)


def rise_ladder(level_d, level_g, cfg: ProblemConfig = DEFAULT):
    """D2 rungs strictly between the descent and the gate order: Mx += 4 per
    rung, My = round(aspect * Mx) with the order rule's aspect coefficient, each
    rung inside the grid's convolution constraint; the gate order closes the
    ladder.  P2 m=4: (28,4) -> (32,5) -> (36,5) -> (40,6) -> (44,6)."""
    aspect = 0.857 * cfg.lattice_nm[1] / cfg.lattice_nm[0]
    d, g = tuple(int(v) for v in level_d), tuple(int(v) for v in level_g)
    out = []
    mx = d[0] + 4
    while mx < g[0]:
        lv = (mx, max(1, int(round(mx * aspect))))
        if cfg.supports(lv) and n_terms(lv) > n_terms(d) and n_terms(lv) < n_terms(g) and lv not in out:
            out.append(lv)
        mx += 4
    if n_terms(g) > n_terms(d):
        out.append(g)
    return out
