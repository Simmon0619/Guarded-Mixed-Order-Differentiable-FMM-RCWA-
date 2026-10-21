import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from workbench.batch import run_experiment
from workbench.configuration import validate_config
from workbench.provenance import sha256_file
from workbench.reproduction import check_reference, execute_manifest, summarize

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("key,value", [("steps", 0), ("nx", 0), ("level_descent", [-1, 0]),
                                       ("seeds", []), ("gm_guards", "false"), ("filter_nm", float("nan")),
                                       ("levl_gate", [2, 1])])
def test_invalid_configuration_is_rejected(optimization_config, key, value):
    optimization_config["params"][key] = value
    with pytest.raises(ValueError):
        validate_config(optimization_config)


def test_batch_exports_exact_arrays_provenance_and_refuses_overwrite(tmp_path, slab_config):
    out = tmp_path / "slab"
    result = run_experiment(slab_config, out)
    assert result["eta_target"] == pytest.approx(64 / 91, abs=1e-6)
    assert result["fields"] == {}
    arrays = np.load(out / "arrays.npz", allow_pickle=False)
    assert arrays["density"].shape == (24, 32)
    assert arrays["power_t"].sum() == pytest.approx(64 / 91, abs=1e-9)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["source"]["files"]["trustworthy_fmm/solvers/vector_2d.py"]
    assert manifest["environment"]["dependencies"]["torch"]
    resolved = json.loads((out / "resolved.json").read_text())
    assert resolved["problem"]["wavelength_nm"] == 600
    assert resolved["engine"]["device"] == "cpu"
    for name, digest in json.loads((out / "checksums.json").read_text()).items():
        assert sha256_file(out / name) == digest
    with pytest.raises(FileExistsError):
        run_experiment(slab_config, out)


def test_small_optimization_exercises_mixing_and_binary_delivery(tmp_path, optimization_config):
    result = run_experiment(optimization_config, tmp_path / "opt")
    ticket = result["tickets"][0]
    assert ticket["n_high_order_steps"] > 0
    assert len(ticket["level_series"]) == optimization_config["params"]["steps"]
    assert result["selection"]["rule"] == "best gate reading"
    assert result["best"]["eta_gate"] == pytest.approx(max(c["eta_gate"] for c in ticket["checkpoints"]), abs=1e-6)
    assert result["best"]["eta_top"] == result["certification"][-1]["eta"]
    assert set(np.asarray(result["patterns"]["binary"]).ravel()).issubset({0, 1})


def test_cli_works_from_another_directory(tmp_path):
    process = subprocess.run([sys.executable, "-m", "workbench", "solve", "--config", str(ROOT / "examples/slab.json"),
                              "--output", str(tmp_path / "run")], cwd=tmp_path, capture_output=True, text=True, timeout=90)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["status"] == "complete"
    process = subprocess.run([sys.executable, "-m", "workbench", "optimize", "--config", str(ROOT / "examples/slab.json"),
                              "--output", str(tmp_path / "wrong")], cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
    assert not (tmp_path / "wrong").exists()


def test_reproduction_checks_reference_and_detects_tampering(tmp_path):
    out = tmp_path / "suite"
    index = execute_manifest(ROOT / "reproduce/smoke-manifest.json", out, only=["slab"])
    assert index["status"] == "complete"
    assert index["runs"][0]["reference"]["passed"]
    assert (out / "summary.csv").is_file()
    with (out / "slab/result.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        summarize(out)


def test_empty_paper_manifest_cannot_claim_reproduction(tmp_path):
    with pytest.raises(ValueError, match="no runnable experiments"):
        execute_manifest(ROOT / "reproduce/paper-manifest.json", tmp_path / "paper")
    assert not (tmp_path / "paper").exists()


def test_failed_reference_is_not_reported_as_pass():
    outcome = check_reference({"eta": 0.1}, {"checks": [{"path": ["eta"], "value": 0.9, "atol": 1e-5}]})
    assert not outcome["passed"]


def test_solver_failure_is_recorded_and_propagated(tmp_path, slab_config, monkeypatch):
    from workbench import app
    def fail(*args, **kwargs):
        raise RuntimeError("injected solver failure")
    monkeypatch.setattr(app, "run_solve", fail)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected solver failure"):
        run_experiment(slab_config, out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert not (out / "checksums.json").exists()


def test_nonfinite_physical_output_is_not_serialized_as_success(tmp_path, slab_config, monkeypatch):
    from workbench import app
    def invalid(job, **kwargs):
        job["arrays"] = {"power_t": np.asarray([float("nan")]), "power_r": np.asarray([0.0])}
        return {"eta_target": float("nan")}
    monkeypatch.setattr(app, "run_solve", invalid)
    out = tmp_path / "nonfinite"
    with pytest.raises(RuntimeError, match="non-finite diffraction"):
        run_experiment(slab_config, out)
    assert json.loads((out / "manifest.json").read_text())["status"] == "failed"
