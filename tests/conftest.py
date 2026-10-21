from pathlib import Path

import pytest

from workbench.configuration import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def slab_config():
    return load_config(ROOT / "examples" / "slab.json")


@pytest.fixture
def optimization_config():
    return load_config(ROOT / "examples" / "optimize-small.json")
