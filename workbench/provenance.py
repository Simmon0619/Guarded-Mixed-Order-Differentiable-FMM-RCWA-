"""Portable run metadata and source fingerprints; no network access."""
import hashlib
import json
import math
import platform
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

from . import __version__


def json_safe(value):
    """Serialize diagnostics as strict JSON (non-finite diagnostics become null)."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, complex):
        return {"real": json_safe(value.real), "imag": json_safe(value.imag)}
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)  # torch.dtype in the resolved dataclass


def write_json(path, value):
    Path(path).write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    import torch
    from trustworthy_fmm.solvers.eigendecomposition import EIG_DEVICE
    from trustworthy_fmm.solvers import vector_2d
    from trustworthy_fmm.fourier import convolution

    versions = {}
    for name in ("torch", "numpy", "scipy", "torcwa"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "package_version": __version__, "dependencies": versions,
            "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "threads": torch.get_num_threads(), "eig_device": EIG_DEVICE,
            "solver_flags": {"vectorized_convolution": convolution.VECTORIZED_CONV,
                             "fast_diagnostics": vector_2d.FAST_DIAGNOSTICS,
                             "gap_only_diagnostics": vector_2d.GAP_ONLY_DIAGNOSTICS,
                             "checkpoint_smatrix": vector_2d.CHECKPOINT_SMATRIX}}


def source_snapshot():
    root = Path(__file__).resolve().parent.parent
    paths = sorted(p for package in ("workbench", "trustworthy_fmm")
                   for p in (root / package).rglob("*")
                   if p.is_file() and p.suffix in (".py", ".html", ".js") and "__pycache__" not in p.parts)
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in paths}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    commit, dirty = None, None
    if (root / ".git").exists() and shutil.which("git"):
        try:
            def git(*args):
                return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL,
                                               text=True, timeout=5).strip()
            commit = git("rev-parse", "HEAD")
            dirty = bool(git("status", "--porcelain"))
        except (subprocess.SubprocessError, OSError):
            pass
    return {"sha256": digest, "files": hashes, "git_commit": commit, "git_dirty": dirty}
