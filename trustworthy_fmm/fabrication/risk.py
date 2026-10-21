"""Finite-catalog manufacturing distributions and lower-tail reward risk.

The functions here deliberately use *lower-tail* CVaR because diffraction
efficiency is a reward: larger values are better.  A finite catalog can be
enumerated exactly, but this module never upgrades that statement to a
continuous-process or foundry-level guarantee.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ProcessScenario:
    name: str
    split: str
    weight: float = 1.0
    cd_bias_nm: float = 0.0
    thickness_delta_nm: float = 0.0
    feature_index_delta: float = 0.0
    ler_rms_nm: float = 0.0
    ler_correlation_nm: float = 0.0
    ler_seed: int = 0
    morphology: str = "none"
    morphology_radius_nm: float = 0.0
    sidewall_top_bias_nm: float = 0.0
    sidewall_bottom_bias_nm: float = 0.0
    sidewall_layers: int = 1
    category: str = "unspecified"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scenario name must be non-empty")
        if self.split not in ("train", "held_out"):
            raise ValueError("scenario split must be train or held_out")
        if self.weight <= 0:
            raise ValueError("scenario weight must be positive")
        if self.ler_rms_nm < 0 or self.ler_correlation_nm < 0:
            raise ValueError("LER magnitudes must be non-negative")
        if self.ler_rms_nm > 0 and self.ler_correlation_nm <= 0:
            raise ValueError("positive LER RMS requires a positive correlation length")
        if self.morphology not in ("none", "opening", "closing"):
            raise ValueError("unsupported morphology operation")
        if self.morphology_radius_nm < 0:
            raise ValueError("morphology radius must be non-negative")
        if self.sidewall_layers < 1:
            raise ValueError("sidewall_layers must be positive")

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], split: str
    ) -> "ProcessScenario":
        values = dict(payload)
        values["split"] = split
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_scenario_catalog(
    entries: Sequence[Mapping[str, Any]], split: str
) -> List[ProcessScenario]:
    scenarios = [ProcessScenario.from_mapping(entry, split) for entry in entries]
    names = [scenario.name for scenario in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("scenario names must be unique within a catalog")
    if not scenarios:
        raise ValueError("scenario catalog must not be empty")
    return scenarios


def validate_blind_split(
    training: Sequence[ProcessScenario], held_out: Sequence[ProcessScenario]
) -> None:
    train_names = {scenario.name for scenario in training}
    held_names = {scenario.name for scenario in held_out}
    overlap = train_names.intersection(held_names)
    if overlap:
        raise ValueError(
            "training and held-out scenario names overlap: {}".format(
                sorted(overlap)
            )
        )
    if any(scenario.split != "train" for scenario in training):
        raise ValueError("training catalog contains a non-training scenario")
    if any(scenario.split != "held_out" for scenario in held_out):
        raise ValueError("held-out catalog contains a non-held-out scenario")
    if any(scenario.morphology != "none" for scenario in training):
        raise ValueError("hard morphology must remain outside differentiable training")


def normalized_weights(
    values: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("values must be a non-empty vector")
    if weights is None:
        return torch.full_like(values, 1.0 / values.numel())
    result = torch.as_tensor(weights, dtype=values.dtype, device=values.device)
    if result.shape != values.shape:
        raise ValueError("weights must match values")
    if bool(torch.any(result <= 0)):
        raise ValueError("weights must be positive")
    return result / result.sum()


def weighted_mean_reward(
    values: torch.Tensor, weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    probabilities = normalized_weights(values, weights)
    return torch.sum(probabilities * values)


def lower_tail_cvar(
    values: torch.Tensor,
    beta: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Exact lower-tail CVaR of a finite weighted reward catalog.

    This is equivalent to ``max_t t - E[(t-values)_+]/beta``.  Sorting provides
    the exact empirical value and a valid subgradient through the selected tail.
    Fractional probability mass at the beta quantile is handled explicitly.
    """
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    probabilities = normalized_weights(values, weights)
    sorted_values, indices = torch.sort(values)
    sorted_probabilities = probabilities[indices]
    remaining = torch.as_tensor(beta, dtype=values.dtype, device=values.device)
    total = torch.zeros((), dtype=values.dtype, device=values.device)
    for value, probability in zip(sorted_values, sorted_probabilities):
        used = torch.minimum(probability, torch.clamp(remaining, min=0.0))
        total = total + used * value
        remaining = remaining - used
    return total / beta


def mean_lower_tail_reward(
    values: torch.Tensor,
    alpha: float,
    beta: float,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Bounded mean–LCVaR reward with alpha in [0,1]."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    mean = weighted_mean_reward(values, weights)
    tail = lower_tail_cvar(values, beta, weights)
    return (1.0 - alpha) * mean + alpha * tail


def weighted_quantile(
    values: Sequence[float], weights: Sequence[float], probability: float
) -> float:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be in [0, 1]")
    array = np.asarray(values, dtype=np.float64)
    mass = np.asarray(weights, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or mass.shape != array.shape:
        raise ValueError("values and weights must be matching non-empty vectors")
    if np.any(mass <= 0):
        raise ValueError("weights must be positive")
    order = np.argsort(array)
    array = array[order]
    mass = mass[order] / mass.sum()
    cumulative = np.cumsum(mass)
    index = min(int(np.searchsorted(cumulative, probability, side="left")), array.size - 1)
    return float(array[index])


def reward_statistics(
    values: Sequence[float],
    beta: float,
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty vector")
    mass = (
        np.ones(array.size, dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    tensor_values = torch.as_tensor(array, dtype=torch.float64)
    tensor_weights = torch.as_tensor(mass, dtype=torch.float64)
    return {
        "mean": float(weighted_mean_reward(tensor_values, tensor_weights)),
        "lower_tail_cvar": float(
            lower_tail_cvar(tensor_values, beta, tensor_weights)
        ),
        "p10": weighted_quantile(array, mass, 0.10),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def scenario_weights(
    scenarios: Sequence[ProcessScenario],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(
        [scenario.weight for scenario in scenarios], dtype=dtype, device=device
    )
