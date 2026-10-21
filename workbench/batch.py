"""Synchronous batch API. Uses the same numerical workflow as the web worker."""
import threading
import time
import math
from datetime import datetime, timezone
from pathlib import Path

from .configuration import validate_config
from .provenance import environment, sha256_file, source_snapshot, write_json

_RUN_LOCK = threading.Lock()  # the existing numerical backend uses module flags


def run_experiment(config, output, *, fields=False):
    """Run a schema-v1 solve/optimization and write a new, self-describing directory.

    Output must not exist. Optional external engines are disabled unless explicitly
    requested. This API starts neither an HTTP server nor the web worker thread.
    Concurrent batch callers are serialized; do not embed a web worker in the same
    process. Existing algorithm defaults are otherwise preserved.
    """
    config = validate_config(config)
    from . import app
    import numpy as np
    import torch

    params = dict(config["params"])
    params.setdefault("compare_torcwa", False)
    params.setdefault("compare_lumerical", False)
    if params.get("engine") == "gpu_c64" and not torch.cuda.is_available():
        raise ValueError("gpu_c64 was requested but CUDA is unavailable; select cpu_c128 explicitly")
    cfg = app.build_cfg(params)
    target = tuple(params.get("target", [1, 0]))
    rule = app.order_rule(target, cfg)
    level = params.get("level_descent") or (rule["descent"] if config["type"] == "optimize" else [12, 3])
    levels = [level] + (params.get("levels_certify") or [])
    if config["type"] == "optimize":
        levels += [params.get("level_gate") or rule["gate"]]
    for lv in levels:
        if not cfg.supports(lv) or any(abs(t) > v for t, v in zip(target, lv)):
            raise ValueError("grid or target is incompatible with truncation order " + str(lv))
    output = Path(output).expanduser().resolve()
    with _RUN_LOCK:
        started = time.time()
        job = {"id": "batch", "type": config["type"], "params": params, "created": started,
               "trajectories": {}, "checkpoints": {}, "chain_jumps": [], "cancel": False}
        manifest = {"schema_version": 1, "status": "running", "type": config["type"],
                    "name": config.get("name"), "started_utc": datetime.now(timezone.utc).isoformat(),
                    "environment": environment(), "source": source_snapshot(),
                    "fields_requested": bool(fields),
                    "serialization": "Non-finite diagnostic numbers are encoded as null in JSON."}
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "input.json", config)
        write_json(output / "manifest.json", manifest)
        try:
            runner = app.run_solve if config["type"] == "solve" else app.run_optimize
            result = runner(job, include_fields=fields, capture_arrays=True)
            efficiency = result["eta_target"] if config["type"] == "solve" else result["best"]["eta_top"]
            if not math.isfinite(efficiency) or any(not np.isfinite(job["arrays"][key]).all() for key in ("power_t", "power_r")):
                raise RuntimeError("non-finite diffraction powers; this run cannot be marked complete")
            resolved = {"requested_params_with_batch_defaults": params, **job["resolved"], "engine": result["engine"]}
            write_json(output / "resolved.json", resolved)
            write_json(output / "result.json", result)
            write_json(output / "trajectories.json", {"trajectories": job["trajectories"],
                                                       "checkpoints": job["checkpoints"],
                                                       "chain_jumps": job["chain_jumps"]})
            np.savez_compressed(output / "arrays.npz", **job["arrays"])
            manifest.update(status="complete", elapsed_s=time.time() - started,
                            finished_utc=datetime.now(timezone.utc).isoformat(),
                            array_evaluation_level=result.get("level_descent") if config["type"] == "solve" else result["certification"][-1]["level"])
            write_json(output / "manifest.json", manifest)
            write_json(output / "checksums.json", {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()})
            return result
        except BaseException as exc:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                            elapsed_s=time.time() - started, error_type=type(exc).__name__, error=str(exc))
            write_json(output / "manifest.json", manifest)
            raise
