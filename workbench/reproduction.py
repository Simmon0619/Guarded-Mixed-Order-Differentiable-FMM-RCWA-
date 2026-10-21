"""Execute explicit experiment manifests and summarize their exported results."""
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

from .batch import run_experiment
from .configuration import load_config
from .provenance import sha256_file, write_json


def load_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or not isinstance(manifest.get("experiments"), list):
        raise ValueError("manifest must have schema_version=1 and an experiments list")
    seen = set()
    for entry in manifest["experiments"]:
        if not isinstance(entry, dict) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", entry.get("id", "")):
            raise ValueError("each experiment needs a simple, nonempty id")
        if entry["id"] in seen:
            raise ValueError("duplicate experiment id: " + entry["id"])
        seen.add(entry["id"])
        if not isinstance(entry.get("config"), str):
            raise ValueError("each experiment needs a config path")
        load_config(path.parent / entry["config"])
    return manifest


def check_reference(result, reference):
    checks = reference.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("reference must contain at least one numerical check")
    rows = []
    for check in checks:
        value = result
        for key in check["path"]:
            value = value[key]
        actual, expected = float(value), float(check["value"])
        atol, rtol = float(check["atol"]), float(check.get("rtol", 0.0))
        if not all(math.isfinite(v) for v in (actual, expected, atol, rtol)) or min(atol, rtol) < 0:
            raise ValueError("reference values and nonnegative tolerances must be finite")
        rows.append({"path": check["path"], "actual": actual, "expected": expected,
                     "atol": atol, "rtol": rtol,
                     "passed": abs(actual - expected) <= atol + rtol * abs(expected)})
    return {"passed": all(row["passed"] for row in rows), "checks": rows}


def execute_manifest(manifest_path, output, *, only=None, fields=False):
    path = Path(manifest_path).resolve()
    manifest = load_manifest(path)
    selected = set(only or [])
    entries = [e for e in manifest["experiments"] if not selected or e["id"] in selected]
    unknown = selected - {e["id"] for e in entries}
    if unknown:
        raise ValueError("unknown experiment ids: " + ", ".join(sorted(unknown)))
    if not entries:
        raise ValueError("manifest has no runnable experiments; populate the paper mapping or use smoke-manifest.json")
    # Preflight all inputs before launching any expensive run.
    configs = {e["id"]: load_config(path.parent / e["config"]) for e in entries}
    refs = {e["id"]: json.loads((path.parent / e["reference"]).read_text(encoding="utf-8"))
            for e in entries if e.get("reference")}
    root = Path(output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "experiment-manifest.json", manifest)
    index = {"schema_version": 1, "manifest_sha256": sha256_file(path), "status": "running", "runs": []}
    write_json(root / "index.json", index)
    try:
        for entry in entries:
            run_dir = root / entry["id"]
            result = run_experiment(configs[entry["id"]], run_dir, fields=fields)
            row = {"id": entry["id"], "directory": entry["id"], "type": configs[entry["id"]]["type"],
                   "paper_items": entry.get("paper_items", []), "purpose": entry.get("purpose", ""),
                   "config_sha256": sha256_file(path.parent / entry["config"])}
            if entry["id"] in refs:
                row["reference_sha256"] = sha256_file(path.parent / entry["reference"])
                row["reference"] = check_reference(result, refs[entry["id"]])
                write_json(root / (entry["id"] + "-reference.json"), refs[entry["id"]])
            index["runs"].append(row)
            write_json(root / "index.json", index)
            if "reference" in row and not row["reference"]["passed"]:
                raise ValueError("reference check failed for " + entry["id"])
        index["status"] = "complete"
        write_json(root / "index.json", index)
        summarize(root)
    except BaseException as exc:
        index.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        write_json(root / "index.json", index)
        raise
    return index


def summarize(root, *, plots=False):
    root = Path(root).resolve()
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    rows = []
    plot_data = []
    for run in index["runs"]:
        directory = root / run["directory"]
        checksums = json.loads((directory / "checksums.json").read_text(encoding="utf-8"))
        for name, expected in checksums.items():
            if Path(name).name != name or sha256_file(directory / name) != expected:
                raise ValueError("run checksum mismatch: " + run["id"] + "/" + name)
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        curve = result.get("convergence", result.get("certification", []))
        highest = curve[-1]
        rows.append({"id": run["id"], "type": run["type"], "paper_items": "; ".join(run["paper_items"]),
                     "eta_highest_order": highest["eta"], "Mx": highest["level"][0], "My": highest["level"][1],
                     "elapsed_s": manifest["elapsed_s"], "source_sha256": manifest["source"]["sha256"]})
        plot_data.append((run["id"], curve))
    if not rows:
        raise ValueError("no completed runs to summarize")
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        for name, curve in plot_data:
            fig, ax = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
            ax.plot([r["two_n"] for r in curve], [r["eta"] for r in curve], "o-")
            ax.set(xlabel="Matrix dimension, 2N", ylabel="Target-order efficiency", title=name)
            ax.grid(alpha=0.25)
            fig.savefig(root / (name + "-convergence.png"), dpi=180)
            fig.savefig(root / (name + "-convergence.pdf"))
            plt.close(fig)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a versioned experiment manifest; no manuscript results are assumed.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--fields", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.list:
            print(json.dumps(load_manifest(args.manifest), indent=2))
        else:
            if not args.output:
                parser.error("--output is required unless --list is used")
            execute_manifest(args.manifest, args.output, only=args.only, fields=args.fields)
            print("Completed: " + args.output)
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print("reproduce: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
