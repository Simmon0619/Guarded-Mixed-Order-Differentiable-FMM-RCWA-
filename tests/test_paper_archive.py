"""The paper archive must detect corruption and preserve paired data."""
import importlib.util
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_runner", ROOT / "reproduce/paper.py")
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


def test_archive_integrity_and_tamper_detection(tmp_path):
    report = paper.verify()
    assert report["archived_files"] > 500
    assert report["missing_array_references"] > 0
    assert not report["full_numerical_reproduction"]
    inventory = paper.read_json(paper.PAPER / "inventory.json")
    first = inventory["bundles"][0]
    inventory["bundles"] = [first]
    paper.write_json(tmp_path / "inventory.json", inventory)
    target = tmp_path / first["path"]
    target.parent.mkdir(parents=True)
    shutil.copyfile(paper.PAPER / first["path"], target)
    with target.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="bundle checksum mismatch"):
        paper.verify(tmp_path)


def test_paired_statistics_from_raw_records():
    prefix = "stage_e1_metasurface_platform/experiments/"
    with zipfile.ZipFile(paper.PAPER / "bundles/F33_section33_results.zip") as archive:
        analysis = json.loads(archive.read(prefix + "F33_section33_results/data/f33_analysis.json"))
        for m, n in zip(range(1, 5), (24, 24, 12, 12)):
            result = analysis["m"][str(m)]
            tickets = json.loads(archive.read(prefix + "F33_section33_results/data/s10_B_m%d.json" % m))["tickets"]
            assert len(tickets) == len(result["rows"]) == n
            assert [tickets["s%02d" % i]["seed"] for i in range(n)] == [150000 + 1000 * m + i for i in range(n)]
            gains = np.array([r["B_Qr"] - r["T_Qr"] for r in result["rows"]])
            assert np.isfinite(gains).all()
            assert np.median(gains) == pytest.approx(result["paired_B_minus_T_Mr"]["median"], abs=1e-14)
            assert int((gains > 0).sum()) == result["paired_pos"]


def test_reference_comparison_rejects_changed_statistics():
    with pytest.raises(ValueError, match="reference value differs"):
        paper.compare_values({"gain": 0.2}, {"gain": 0.1})
    with pytest.raises(ValueError, match="reference keys differ"):
        paper.compare_values({}, {"gain": 0.1})
