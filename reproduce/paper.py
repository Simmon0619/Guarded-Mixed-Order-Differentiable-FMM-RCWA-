"""Verify and reanalyse the archived manuscript evidence without physics solves."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
STAGE = "stage_e1_metasurface_platform"
F33 = "F33_section33_results"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify(paper=PAPER):
    """Validate bundle bytes and every member, including the original file hashes."""
    paper = Path(paper)
    inventory = read_json(paper / "inventory.json")
    expected = {r["path"].removeprefix("archive/"): r for r in inventory["files"]}
    seen = set()
    for bundle in inventory["bundles"]:
        name = bundle["path"]
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts:
            raise ValueError("unsafe bundle path")
        path = paper / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != bundle["sha256"]:
            raise ValueError("bundle checksum mismatch: " + name)
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                member = info.filename
                if member not in expected or member in seen:
                    raise ValueError("unexpected or duplicate archive member: " + member)
                if PurePosixPath(member).is_absolute() or ".." in PurePosixPath(member).parts or "\\" in member or ":" in member:
                    raise ValueError("unsafe archive member: " + member)
                data = archive.read(info)
                row = expected[member]
                if len(data) != row["bytes"] or hashlib.sha256(data).hexdigest() != row["sha256"]:
                    raise ValueError("member checksum mismatch: " + member)
                seen.add(member)
    if seen != set(expected):
        raise ValueError("archive is missing inventoried files")
    missing = read_json(paper / "missing-array-references.json")["missing"]
    return {"status": "verified", "archived_files": len(seen), "bundles": len(inventory["bundles"]),
            "missing_array_references": len(missing), "full_numerical_reproduction": False}


def compare_values(actual, expected, path="root"):
    """Compare all stored statistics; allow only floating point roundoff."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("reference keys differ at " + path)
        for key in expected:
            compare_values(actual[key], expected[key], path + "." + key)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError("reference length differs at " + path)
        for i, (a, b) in enumerate(zip(actual, expected)):
            compare_values(a, b, path + "[%d]" % i)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("reference value differs at " + path)
    elif actual != expected:
        raise ValueError("reference value differs at " + path)


def prepare(output, *, paper=PAPER):
    report = verify(paper)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    # Deep research paths can exceed the legacy Windows MAX_PATH limit.
    if os.name == "nt":
        workspace = Path("\\\\?\\" + str(workspace))
    for bundle in read_json(Path(paper) / "inventory.json")["bundles"]:
        with zipfile.ZipFile(Path(paper) / bundle["path"]) as archive:
            # verify() checked every path and byte before creating the workspace.
            archive.extractall(workspace)
    checkout = workspace / STAGE / "fmm_workbench"
    for package in ("trustworthy_fmm", "workbench"):
        shutil.copytree(ROOT / package, checkout / package,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "data"))
    report["workspace"] = "workspace"
    write_json(output / "archive-verification.json", report)
    return workspace


def analyze(output, *, figures=False):
    output = Path(output).resolve()
    workspace = prepare(output)
    exp = workspace / STAGE / "experiments"
    data = exp / F33 / "data"
    reference = read_json(data / "f33_analysis.json")
    logs = output / "logs"
    logs.mkdir()
    (data / "logs").mkdir(exist_ok=True)
    report = {"status": "running", "kind": "analysis_of_archived_results", "physics_solves": 0,
              "python": sys.version, "reference_tolerance": {"atol": 1e-12, "rtol": 1e-12}, "commands": []}
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

    def run(name, script, *args):
        command = [sys.executable, str(script), *args]
        with (logs / (name + ".txt")).open("w", encoding="utf-8") as stream:
            subprocess.run(command, cwd=script.parent, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        report["commands"].append({"id": name, "script": script.relative_to(workspace).as_posix(), "args": list(args)})

    try:
        run("f33-analysis", exp / F33 / "scripts/f33_analyze.py")
        actual = read_json(data / "f33_analysis.json")
        compare_values(actual, reference)
        counts = [actual["m"][str(m)]["n"] for m in range(1, 5)]
        if counts != [24, 24, 12, 12]:
            raise ValueError("incomplete F33 paired dataset")
        report.update(reference_comparison="passed", pairs_per_channel=counts,
                      improved_pairs=sum(actual["m"][str(m)]["paired_pos"] for m in range(1, 5)))
        shutil.copyfile(data / "f33_analysis.json", output / "f33_analysis.json")
        run("si-tables", exp / F33 / "scripts/f33_si_tables.py")
        shutil.copyfile(logs / "si-tables.txt", output / "f33-tables.md")
        if figures:
            run("fig3-parts", exp / F33 / "scripts/f33_fig3_parts.py")
            run("si-figures", exp / "FS_si_figures/scripts/fs_composites.py", "C1", "C3", "C4", "C5")
            report["regenerated_figures"] = ["Fig. 3 panels", "SI S1", "SI S3", "SI S4", "SI S5"]
            report["figures_not_regenerated"] = {"SI S2": "requires missing S8 frozen-state latent array",
                                                 "Figs. 2 and 4 insets": "require missing delivered binary masks"}
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(output / "analysis-report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="verify bundles; report missing historical arrays")
    prep = sub.add_parser("prepare", help="extract a separate research workspace; launch no simulations")
    prep.add_argument("--output", required=True)
    analysis = sub.add_parser("analyze", help="recompute F33 statistics and compare with archived reference")
    analysis.add_argument("--output", required=True)
    analysis.add_argument("--figures", action="store_true", help="also redraw Fig. 3 panels and SI Figs. S1/S3/S4/S5")
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            result = verify()
        elif args.command == "prepare":
            result = {"status": "prepared", "workspace": str(prepare(args.output))}
        else:
            result = analyze(args.output, figures=args.figures)
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        print("error: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
