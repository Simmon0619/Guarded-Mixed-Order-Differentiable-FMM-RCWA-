"""Versioned complex-index datasets for multi-condition tasks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import yaml


@dataclass(frozen=True)
class MaterialStateData:
    name: str
    wavelength_nm: np.ndarray
    refractive_index: np.ndarray

    @property
    def permittivity(self) -> np.ndarray:
        return self.refractive_index ** 2


def load_material_states(path: Path, *, require_claim_allowed: bool = False) -> Dict[str, MaterialStateData]:
    with Path(path).open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if require_claim_allowed and not payload.get("claim_allowed", False):
        raise ValueError("dataset is explicitly marked numerical-only")
    wavelengths = np.asarray(payload["wavelength_nm"], dtype=np.float64)
    states = {}
    for name, values in payload["states"].items():
        n = np.asarray(values["n"], dtype=np.float64)
        k = np.asarray(values["k"], dtype=np.float64)
        if n.shape != wavelengths.shape or k.shape != wavelengths.shape:
            raise ValueError("material arrays must match wavelength_nm")
        states[name] = MaterialStateData(name, wavelengths, n + 1j * k)
    return states

