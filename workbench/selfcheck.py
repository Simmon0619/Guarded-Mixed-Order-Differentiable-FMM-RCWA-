"""Self-check for a fresh machine.

    python selfcheck.py            # environment report + correctness + GPU/CPU agreement
    python selfcheck.py --bench    # ... plus timing of one optimization step at two orders

What it verifies (each line is [PASS] / [FAIL] / [SKIP] / [INFO]):
  1. the solver library imports from this folder (nothing else is needed);
  2. a uniform slab reproduces the analytic Airy transmission (CPU, complex128);
  3. energy is conserved on a patterned structure (CPU);
  4. if a CUDA GPU is present: the GPU/complex64 value and gradient agree with
     the CPU/complex128 reference on the same pattern;
  5. optional engines: TORCWA (with a parity check when installed) and the
     Ansys Lumerical API (import only; no license is consumed).
Exit code 1 if anything FAILs.
"""
import argparse
import math
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FAILS = []


def report(status, text):
    print("[%s] %s" % (status, text), flush=True)
    if status == "FAIL":
        FAILS.append(text)


def slab_transmission(n1, n2, n3, thickness_nm, wavelength_nm):
    """Airy formula: power transmission of a uniform slab at normal incidence."""
    r12, t12 = (n1 - n2) / (n1 + n2), 2 * n1 / (n1 + n2)
    r23, t23 = (n2 - n3) / (n2 + n3), 2 * n2 / (n2 + n3)
    delta = 2 * math.pi / wavelength_nm * n2 * thickness_nm
    e = complex(math.cos(delta), math.sin(delta))
    t = t12 * t23 * e / (1 + r12 * r23 * e * e)
    return (n3 / n1) * abs(t) ** 2


def main(argv=None):
    FAILS.clear()
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="time value+gradient at (12,3) and (24,6)")
    args = ap.parse_args(argv)

    # ---- 1. environment
    report("INFO", "python %s on %s" % (platform.python_version(), platform.platform()))
    try:
        import numpy, scipy, torch  # noqa: E401
    except Exception as error:
        report("FAIL", "core dependency missing: %s  (pip install -r requirements.txt)" % error)
        return finish()
    report("INFO", "torch %s (built for CUDA %s), numpy %s, scipy %s"
           % (torch.__version__, torch.version.cuda, numpy.__version__, scipy.__version__))
    cuda = torch.cuda.is_available()
    if cuda:
        prop = torch.cuda.get_device_properties(0)
        report("INFO", "GPU: %s, %.1f GB" % (prop.name, prop.total_memory / 2 ** 30))
    else:
        report("INFO", "no CUDA GPU visible to torch: the workbench will run on the CPU "
                       "(install the CUDA build of torch from https://pytorch.org if you have an NVIDIA GPU)")
    try:
        from workbench import app
        import trustworthy_fmm
        lib = os.path.dirname(os.path.abspath(trustworthy_fmm.__file__))
        if os.path.abspath(os.path.dirname(lib)) != os.path.abspath(HERE):
            report("FAIL", "trustworthy_fmm was imported from %s, not from this folder" % lib)
        else:
            report("PASS", "workbench and solver library import from this folder")
    except Exception as error:
        report("FAIL", "import failed: %r" % error)
        return finish()

    cfg = app.build_cfg({})           # the workbench's default problem, 120x35 grid
    c128, c64 = torch.complex128, torch.complex64

    # ---- 2. uniform slab vs analytic Airy transmission (CPU, double)
    t0 = time.time()
    ones = torch.ones((cfg.ny, cfg.nx), dtype=cfg.latent_dtype)
    with torch.no_grad():
        r = app.solve_density(ones, (4, 1), cfg, 0.0, 0.0, c128, "cpu")
    ref = slab_transmission(cfg.n_incident, math.sqrt(cfg.eps_feature), cfg.n_output,
                            cfg.thickness_nm, cfg.wavelength_nm)
    err = abs(float(r.total_T) - ref)
    report("PASS" if err < 1e-8 else "FAIL",
           "uniform slab: RCWA T=%.10f vs Airy %.10f (|diff|=%.1e, %.2f s)" % (float(r.total_T), ref, err, time.time() - t0))

    # ---- 3. patterned structure: energy conservation (CPU, double)
    latent = app.app_initial_latent(2029, cfg, "y", "cpu")
    binary = app.app_hard_binary(latent, cfg, "y")
    level, target = (6, 2), (1, 0)
    t0 = time.time()
    with torch.no_grad():
        r_cpu = app.solve_density(binary, level, cfg, 0.0, 0.0, c128, "cpu")
    eta_cpu = app.eta_of(r_cpu, target)
    res = abs(float(r_cpu.energy_residual))
    report("PASS" if res < 1e-8 else "FAIL",
           "patterned (6,2) CPU/c128: eta(1,0)=%.6f, T+R-1=%.1e (%.2f s)" % (eta_cpu, res, time.time() - t0))
    t0 = time.time()
    v_cpu, g_cpu = app.value_grad(latent, level, target, 8.0, cfg, "y", 0.0, 0.0, c128, "cpu", False)
    report("PASS" if torch.isfinite(g_cpu).all() else "FAIL",
           "gradient CPU/c128: value=%.6f, |grad|=%.3e (%.2f s)" % (v_cpu, float(g_cpu.norm()), time.time() - t0))

    # ---- 4. GPU agreement
    if cuda:
        try:
            t0 = time.time()
            with torch.no_grad():
                r_gpu = app.solve_density(binary, level, cfg, 0.0, 0.0, c64, "cuda")
            eta_gpu = app.eta_of(r_gpu, target)
            # single precision: mixed tolerance |a-b| <= 1e-5 + 1e-3 |b|
            d_eta = abs(eta_gpu - eta_cpu)
            d_T = abs(float(r_gpu.total_T) - float(r_cpu.total_T))
            ok = d_eta <= 1e-5 + 1e-3 * abs(eta_cpu) and d_T <= 1e-5 + 1e-3 * float(r_cpu.total_T)
            report("PASS" if ok else "FAIL",
                   "GPU/c64 vs CPU/c128 value: eta=%.6f vs %.6f (|diff| %.1e), T=%.6f vs %.6f (|diff| %.1e), %.2f s incl. CUDA init"
                   % (eta_gpu, eta_cpu, d_eta, float(r_gpu.total_T), float(r_cpu.total_T), d_T, time.time() - t0))
            v_gpu, g_gpu = app.value_grad(latent.cuda(), level, target, 8.0, cfg, "y", 0.0, 0.0, c64, "cuda", False)
            g_gpu = g_gpu.cpu().to(g_cpu.dtype)
            cos = float((g_gpu * g_cpu).sum() / (g_gpu.norm() * g_cpu.norm() + 1e-30))
            d_v = abs(v_gpu - v_cpu)
            report("PASS" if (cos > 0.999 and d_v <= 1e-5 + 1e-3 * abs(v_cpu)) else "FAIL",
                   "GPU/c64 vs CPU/c128 gradient: cosine=%.6f, value %.6f vs %.6f (|diff| %.1e)" % (cos, v_gpu, v_cpu, d_v))
            report("INFO", "peak GPU memory during the checks: %.2f GB" % (torch.cuda.max_memory_allocated() / 2 ** 30))
        except Exception as error:
            report("FAIL", "GPU path raised: %r" % error)
    else:
        report("SKIP", "GPU agreement (no CUDA)")

    # ---- 5. optional engines
    try:
        import torcwa  # noqa: F401
        dtype, device = (c64, "cuda") if cuda else (c128, "cpu")
        eta_t = app.torcwa_eta_density(binary, level, target, cfg, 0.0, 0.0, dtype, device)
        d = abs(eta_t - eta_cpu)
        report("PASS" if d <= 1e-5 + 1e-3 * abs(eta_cpu) else "FAIL",
               "TORCWA parity at (6,2): %.6f vs ours %.6f (|diff| %.1e)" % (eta_t, eta_cpu, d))
    except ImportError:
        report("SKIP", "torcwa not installed (optional: pip install torcwa)")
    except Exception as error:
        report("FAIL", "torcwa raised: %r" % error)
    lum = app.lumerical_env()
    if lum["importable"]:
        report("INFO", "Lumerical API importable from %s (a session is only started when you tick the cross-check)" % lum["path"])
    else:
        report("SKIP", "Lumerical API not available: %s" % lum["error"])

    # ---- 6. optional benchmark
    if args.bench:
        dtype, device = (c64, "cuda") if cuda else (c128, "cpu")
        bcfg = app.build_cfg({"nx": 480, "ny": 140})     # the workbench's default grid
        lat = app.app_initial_latent(2029, bcfg, "y", device)
        for lv, reps in (((12, 3), 3), ((24, 6), 2)):
            app.value_grad(lat, lv, target, 8.0, bcfg, "y", 0.0, 0.0, dtype, device, False)   # warm-up
            if cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(reps):
                app.value_grad(lat, lv, target, 8.0, bcfg, "y", 0.0, 0.0, dtype, device, False)
            if cuda:
                torch.cuda.synchronize()
            per = (time.time() - t0) / reps
            report("INFO", "one optimization step (value+gradient) at %s, 2N=%d, %s: %.3f s"
                   % (lv, 2 * app.C.n_terms(lv), device, per))
        report("INFO", "the panel's cost estimate assumes 0.106 s per step at (12,3) on the reference GPU; "
                       "scale your expectations by the ratio you see above")
    return finish()


def finish():
    if FAILS:
        print("\n%d check(s) FAILED:" % len(FAILS))
        for f in FAILS:
            print("  - " + f)
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
