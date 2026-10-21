"""Public, solver-independent data contracts.

The dataclasses deliberately contain tensors without assuming a particular
solver implementation.  Tasks consume these objects and therefore cannot
hard-code a diffraction order or reach into an eigensolver.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch


Order = Tuple[int, int]


class Polarization(str, Enum):
    S = "s"
    P = "p"
    JONES = "jones"


@dataclass(frozen=True)
class SimulationConfig:
    wavelength_nm: float
    lattice_nm: Tuple[float, float]
    fourier_order: Tuple[int, int]
    dtype: torch.dtype = torch.complex128
    device: str = "cpu"
    time_convention: str = "exp_minus_i_omega_t"
    forward_direction: str = "plus_z"
    order_indexing: str = "lexicographic_mn"
    formulation: str = "scalar_1d_te"

    def __post_init__(self) -> None:
        if self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive")
        if any(period <= 0 for period in self.lattice_nm):
            raise ValueError("lattice periods must be positive")
        if any(order < 0 for order in self.fourier_order):
            raise ValueError("Fourier half-orders must be non-negative")
        if self.dtype not in (torch.complex64, torch.complex128):
            raise ValueError("dtype must be torch.complex64 or torch.complex128")


@dataclass(frozen=True)
class InputChannel:
    wavelength_nm: float
    theta_deg: float = 0.0
    phi_deg: float = 0.0
    polarization: Polarization = Polarization.S
    jones: Optional[Tuple[complex, complex]] = None

    def __post_init__(self) -> None:
        if self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive")
        if not 0.0 <= self.theta_deg < 90.0:
            raise ValueError("theta_deg must be in [0, 90)")
        if self.polarization == Polarization.JONES and self.jones is None:
            raise ValueError("Jones polarization requires a two-component vector")


@dataclass(frozen=True)
class OutputChannel:
    order: Order
    polarization: Polarization = Polarization.S
    side: str = "transmission"

    def __post_init__(self) -> None:
        if self.side not in ("transmission", "reflection"):
            raise ValueError("side must be transmission or reflection")


@dataclass(frozen=True)
class DeviceState:
    name: str
    controls: Mapping[str, float] = field(default_factory=dict)
    material_dataset: Optional[str] = None


@dataclass
class ScatteringResult:
    orders: Sequence[Order]
    complex_t: torch.Tensor
    complex_r: torch.Tensor
    power_t: torch.Tensor
    power_r: torch.Tensor
    total_T: torch.Tensor
    total_R: torch.Tensor
    absorption: torch.Tensor
    energy_residual: torch.Tensor
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    fields: Optional[Mapping[str, torch.Tensor]] = None

    def order_index(self, order: Order) -> int:
        try:
            return list(self.orders).index(tuple(order))
        except ValueError as exc:
            raise KeyError("order {} is absent from the result".format(order)) from exc

    def transmitted_power(self, order: Order) -> torch.Tensor:
        return self.power_t[..., self.order_index(order)]

    def transmitted_amplitude(self, order: Order) -> torch.Tensor:
        return self.complex_t[..., self.order_index(order)]


@dataclass
class AuditResult:
    ad_directional: float
    finite_differences: Mapping[float, float]
    best_step: float
    best_fd: float
    relative_error: float
    gradient_norm: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FidelityDecision:
    action: str
    accepted: bool
    fourier_order: int
    trust_radius: float
    reason: str
    diagnostics: Mapping[str, float] = field(default_factory=dict)

