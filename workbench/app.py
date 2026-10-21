"""FMM workbench: a local web app that runs differentiable RCWA solves and
inverse-design optimizations on this machine and shows patterns, fields,
engine comparisons and timings in the browser.

Backend: Python standard-library HTTP server + a single worker thread (one
GPU job at a time).  Frontend: workbench/index.html (vanilla JS + Plotly).
Runs are persisted under workbench/data/runs/.

Run:  python run.py [--port 8321]      (or: python workbench/app.py)
"""
import argparse
import gc
import json
import math
import os
import queue
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))   # release root (contains trustworthy_fmm/)
# Retain direct-script startup, while installed and module imports use packages.
if __package__ in (None, "") and _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

THREADS = int(os.environ.get("FMM_THREADS", "3"))
torch.set_num_threads(THREADS)

from workbench import protocol as C  # noqa: E402
from workbench import descent as D  # noqa: E402
import trustworthy_fmm.fourier.convolution as _conv  # noqa: E402
import trustworthy_fmm.solvers.vector_2d as _v2d  # noqa: E402
from trustworthy_fmm.fabrication.process import (  # noqa: E402
    periodic_gaussian_filter,
    tanh_projection,
)
from trustworthy_fmm.physics.channels import outgoing_sqrt  # noqa: E402
from trustworthy_fmm.solvers.vector_2d import solve_multilayer_vector_2d  # noqa: E402

_conv.VECTORIZED_CONV = True      # vectorized convolution-matrix build (same values, far fewer autograd nodes)
_v2d.FAST_DIAGNOSTICS = True      # skip eigen-gap / condition diagnostics (numerically neutral)
_v2d.GAP_ONLY_DIAGNOSTICS = True  # ... but keep the detached minimum eigen gap (logged by the guards; no autograd node)

_SOURCE_CHECKOUT = os.path.isfile(os.path.join(_ROOT, "run.py")) and os.path.isfile(os.path.join(_ROOT, "pyproject.toml"))
RUNS_DIR = os.path.abspath(os.path.expanduser(os.environ.get(
    "FMM_RUNS_DIR", os.path.join(_HERE, "data", "runs") if _SOURCE_CHECKOUT
    else str(Path.home() / ".fmm-workbench" / "runs"))))


# ----------------------------------------------------------------- messages
# User-visible strings that travel through the JSON API are bilingual objects
# {"code", "zh", "en"}; the frontend shows the active language.  Exceptions
# (job errors) are English only.
_MSG = {
    "not_selected": ("未勾选", "not selected"),
    "fields_note": ("场由模态解重构：z<0 入射平面波 + 反射级次，z>h 透射级次，0≤z≤h 用层内本征模（振幅由两界面的切向连续性定出，"
                    "Ez 由安培定律 Ez = −(kx Hy − ky Hx)/ε 得到）；含倏逝阶与倏逝模。x 画两个周期，y = y0（对称面）。"
                    "入射 |E| = 1。",
                    "Fields reconstructed from the modal solution: incident plane wave + reflected orders for z<0, transmitted orders for z>h, "
                    "and the layer eigenmodes for 0≤z≤h (amplitudes from tangential continuity at both interfaces, Ez from Ampere's law "
                    "Ez = −(kx Hy − ky Hx)/ε); evanescent orders and modes included. x spans two periods at y = y0 (the symmetry plane); "
                    "incident |E| = 1."),
    "torcwa_missing": ("torcwa 不可用: {err}", "torcwa is not available: {err}"),
    "torcwa_note": ("TORCWA 口径：|t_xx|^2+|t_yx|^2（x 偏振约定）",
                    "TORCWA convention: |t_xx|^2+|t_yx|^2 (x-polarized source)"),
    "lum_pol": ("Lumerical 臂仅支持 x 偏振（P 源约定）",
                "the Lumerical arm supports x polarization only (P-source convention)"),
    "lum_theta": ("Lumerical 臂仅支持正入射（θ=0）",
                  "the Lumerical arm supports normal incidence only (θ=0)"),
    "lum_missing": ("Lumerical 不可用: {err}", "Lumerical is not available: {err}"),
    "lum_polygons": ("多边形导出不精确: {err}", "polygon export is not exact: {err}"),
    "lum_note": ("Lumerical RCWA（自带求解器，非 FDTD）；P 源 = x 偏振正入射；"
                 "Tp_power(m,0) 与本引擎 power_t 直接可比。",
                 "Lumerical RCWA (its built-in solver, not FDTD); P source = x-polarized normal "
                 "incidence; Tp_power(m,0) is directly comparable to this engine's power_t."),
    "lum_failed": ("Lumerical 运行失败: {err}", "Lumerical run failed: {err}"),
    "origin_saved": ("历史运行 {run}", "saved run {run}"),
    "origin_restart": ("起点重启 seed {seed} @step {step}",
                       "start restart seed {seed} @ step {step}"),
    "warn_descent_low": ("下降阶 {lv} 低于阶规则建议 {rule}（目标级次 {m}）",
                         "descent order {lv} is below the order rule's suggestion {rule} "
                         "(target order {m})"),
    "warn_gate_low": ("闸门阶 {gate} 不足下降阶 {lv} 的 2×",
                      "gate order {gate} is less than 2× the descent order {lv}"),
    "warn_ladder_low": ("认证阶梯最高阶 {top} 低于规则建议 {rule}",
                        "the top certification order {top} is below the rule's suggestion {rule}"),
    "warn_clipped": ("网格把规则阶截断了：完整规则需要 nx≥{nx}, ny≥{ny}",
                     "the grid clips the rule orders: the full rule needs nx ≥ {nx}, ny ≥ {ny}"),
    "origin_ticket": ("票 seed {seed} 的最优闸门检查点 @step {step}",
                      "best gate checkpoint of ticket seed {seed} @ step {step}"),
    "origin_repair": ("修复链（父本 seed {seed}，{acc} 次接受）",
                      "repair chain (parent seed {seed}, {acc} accepts)"),
    "warn_beta_star": ("保存的运行没有 β⋆，修复链改用固定 β={b}",
                       "the saved run has no β*; the repair chain uses the fixed β={b}"),
    "warn_rise_needs_cadence": ("在线升档需要自适应节奏，已自动开启 D1",
                                "the online rung rise needs the adaptive cadence; D1 was switched on"),
    "torcwa_opt_chain": ("链式协议下 TORCWA 头对头未实现（基线协议可用）",
                         "the TORCWA head-to-head is not implemented for the chain protocol "
                         "(use the baseline protocol)"),
    "torcwa_opt_pol": ("TORCWA 优化臂仅支持 x 偏振（xx+yx 约定）",
                       "the TORCWA optimizer arm supports x polarization only (xx+yx convention)"),
    "torcwa_opt_missing": ("torcwa 未安装，TORCWA 优化臂已跳过",
                           "torcwa is not installed; the TORCWA optimizer arm was skipped"),
}


def M(code, **kw):
    zh, en = _MSG[code]
    return {"code": code, "zh": zh.format(**kw), "en": en.format(**kw)}


def _lumerical_api_path():
    """LUMERICAL_API_PATH if set, else the newest Lumerical install found in the
    usual locations (None when there is none)."""
    env = os.environ.get("LUMERICAL_API_PATH")
    if env:
        return env
    import glob
    hits = []
    for pattern in (r"C:\Program Files\Lumerical\v*\api\python",
                    "/opt/lumerical/v*/api/python",
                    "/Applications/Lumerical/v*/api/python"):
        hits.extend(glob.glob(pattern))
    return sorted(hits)[-1] if hits else None


_LUM_STATE = {}


def lumerical_env():
    """{"path", "importable", "error"} — import is attempted once and cached."""
    if not _LUM_STATE:
        path = _lumerical_api_path()
        info = {"path": path, "importable": False, "error": None}
        if path is None:
            info["error"] = "no Lumerical install found; set LUMERICAL_API_PATH"
        else:
            try:
                if path not in sys.path:
                    sys.path.append(path)
                import lumapi  # noqa: F401
                info["importable"] = True
            except Exception as error:
                info["error"] = str(error)
        _LUM_STATE.update(info)
    return dict(_LUM_STATE)


# ------------------------------------------------------------------ helpers
def _round_list(array, digits=6):
    return np.round(np.asarray(array, dtype=np.float64), digits).tolist()


def _sym(rho, mode):
    if mode == "y":
        return 0.5 * (rho + torch.flip(rho, dims=(0,)))
    if mode == "x":
        return 0.5 * (rho + torch.flip(rho, dims=(1,)))
    if mode == "xy":
        rho = 0.5 * (rho + torch.flip(rho, dims=(0,)))
        return 0.5 * (rho + torch.flip(rho, dims=(1,)))
    return rho


def app_filtered(rho, cfg, mode):
    return _sym(periodic_gaussian_filter(_sym(rho, mode), cfg.lattice_nm,
                                         cfg.filter_nm), mode)


def app_projected(rho, beta, cfg, mode):
    return tanh_projection(app_filtered(rho, cfg, mode), float(beta), cfg.threshold)


def app_hard_binary(rho, cfg, mode):
    with torch.no_grad():
        return (app_filtered(rho, cfg, mode) >= cfg.threshold).to(cfg.latent_dtype)


def app_initial_latent(seed, cfg, mode, device):
    g = torch.Generator().manual_seed(int(seed))
    noise = torch.randn((cfg.ny, cfg.nx), generator=g, dtype=cfg.latent_dtype)
    smooth = periodic_gaussian_filter(noise, cfg.lattice_nm, cfg.filter_nm)
    smooth = smooth / smooth.std()
    latent = _sym((cfg.init_mean + cfg.init_scale * smooth)
                  .clamp(cfg.init_clamp_lo, cfg.init_clamp_hi), mode)
    return latent if device == "cpu" else latent.to(device)


class AppAdam:
    """e10_common.Adam with the symmetry mode passed in instead of baked in."""

    def __init__(self, shape, cfg, mode, device):
        self.cfg, self.mode = cfg, mode
        dev = None if device == "cpu" else device
        self.m = torch.zeros(shape, dtype=cfg.latent_dtype, device=dev)
        self.v = torch.zeros(shape, dtype=cfg.latent_dtype, device=dev)
        self.t = 0

    def step(self, rho, grad, lr):
        c = self.cfg
        self.t += 1
        self.m = c.adam_b1 * self.m + (1 - c.adam_b1) * grad
        self.v = c.adam_b2 * self.v + (1 - c.adam_b2) * grad.square()
        mh = self.m / (1 - c.adam_b1 ** self.t)
        vh = self.v / (1 - c.adam_b2 ** self.t)
        return _sym((rho + lr * mh / (vh.sqrt() + c.adam_eps)).clamp(0.0, 1.0),
                    self.mode)


def build_cfg(p):
    jones = {
        "x": (1.0 + 0j, 0.0 + 0j),
        "y": (0.0 + 0j, 1.0 + 0j),
    }.get(p.get("polarization", "x"))
    if jones is None:  # custom
        jones = (complex(float(p.get("jx_re", 1.0)), float(p.get("jx_im", 0.0))),
                 complex(float(p.get("jy_re", 0.0)), float(p.get("jy_im", 0.0))))
    beta = p.get("beta_ladder") or [1.0, 64.0]
    return C.DEFAULT.variant(
        wavelength_nm=float(p.get("wavelength_nm", 660.0)),
        lattice_nm=(float(p.get("lx_nm", 2400.0)), float(p.get("ly_nm", 700.0))),
        ny=int(p.get("ny", 35)), nx=int(p.get("nx", 120)),
        thickness_nm=float(p.get("thickness_nm", 800.0)),
        n_incident=float(p.get("n_incident", 1.46)),
        n_output=float(p.get("n_output", 1.0)),
        eps_background=float(p.get("n_background", 1.0)) ** 2,
        eps_feature=float(p.get("n_feature", 3.2)) ** 2,
        incident_jones=jones,
        filter_nm=float(p.get("filter_nm", 60.0)),
        threshold=float(p.get("threshold", 0.5)),
        beta_ladder=tuple(float(b) for b in beta),
        adam_lr0=float(p.get("lr0", 0.02)),
    )


def offload_threshold_2n():
    """save_on_cpu 'auto' threshold: 2N = 3000 on a 12 GB card (the measured
    comfort limit of the reference machine), scaled with the square root of
    the GPU memory (peak memory grows as (2N)^2)."""
    if not torch.cuda.is_available():
        return None
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    return int(round(3000.0 * math.sqrt(max(mem_gb, 1.0) / 12.0)))


def engine_of(p):
    """(dtype, device, memory_efficient, offload) from the params."""
    engine = p.get("engine", "gpu_c64")
    if engine == "gpu_c64" and torch.cuda.is_available():
        dtype, device = torch.complex64, "cuda"
    else:
        dtype, device = torch.complex128, "cpu"
    levels = [p.get("level_descent") or [12, 3]]
    if p.get("level_gate"):
        levels.append(p["level_gate"])   # mixed-order gradients: the gate-order backward is the memory peak
    two_n = max(2 * C.n_terms(lv) for lv in levels)
    me = bool(p.get("memory_efficient", device == "cuda"))
    off = p.get("save_on_cpu", "auto")
    thr = offload_threshold_2n() or 3000
    offload = (two_n >= thr) if off == "auto" else bool(off is True or off == "on")
    return dtype, device, me and device == "cuda", offload and device == "cuda"


def solve_density(density, level, cfg, theta, phi, dtype, device):
    eps = (cfg.eps_background
           + (cfg.eps_feature - cfg.eps_background) * density).to(dtype).to(device)
    return solve_multilayer_vector_2d(
        [eps], [cfg.thickness_nm], cfg.wavelength_nm, cfg.lattice_nm,
        cfg.n_incident, cfg.n_output, tuple(level),
        theta_deg=float(theta), phi_deg=float(phi),
        incident_jones=cfg.incident_jones)


class _export_modes:
    """Context: make the solver hand back the layer eigenmodes (field maps)."""

    def __enter__(self):
        self.prev = _v2d.EXPORT_LAYER_MODES
        _v2d.EXPORT_LAYER_MODES = True

    def __exit__(self, *exc):
        _v2d.EXPORT_LAYER_MODES = self.prev
        return False


def eta_of(result, target):
    return float(result.power_t[result.orders.index(tuple(target))])


def value_grad(rho, level, target, beta, cfg, mode, theta, phi, dtype, device,
               offload):
    x = rho.detach().clone().requires_grad_(True)
    ctx = (torch.autograd.graph.save_on_cpu(pin_memory=True)
           if offload else nullcontext())
    with ctx:
        r = solve_density(app_projected(x, beta, cfg, mode), level, cfg,
                          theta, phi, dtype, device)
        j = r.power_t[r.orders.index(tuple(target))]
        j.backward()
    return float(j.detach()), x.grad.detach()


# ------------------------------------------------------------ result blocks
def k_vectors(cfg, level, theta, phi):
    mx, my = level
    orders = [(m, n) for m in range(-mx, mx + 1) for n in range(-my, my + 1)]
    base = cfg.n_incident * math.sin(math.radians(theta))
    kx = np.array([base * math.cos(math.radians(phi))
                   + m * cfg.wavelength_nm / cfg.lattice_nm[0] for m, _ in orders])
    ky = np.array([base * math.sin(math.radians(phi))
                   + n * cfg.wavelength_nm / cfg.lattice_nm[1] for _, n in orders])
    return orders, kx, ky


def orders_block(result, cfg, level, theta, phi, target, max_evanescent=10):
    orders, kx, ky = k_vectors(cfg, level, theta, phi)
    kz_out = outgoing_sqrt(torch.as_tensor(
        cfg.n_output ** 2 - kx ** 2 - ky ** 2, dtype=torch.complex128)).numpy()
    power_t = result.power_t.detach().cpu().numpy().astype(np.float64)
    power_r = result.power_r.detach().cpu().numpy().astype(np.float64)
    t_co = result.complex_t.detach().cpu().numpy().astype(np.complex128)
    prop = kz_out.real > 1e-9
    abs_t = np.abs(t_co)
    keep = set(np.nonzero(prop)[0].tolist())
    evanescent = [i for i in np.argsort(-abs_t) if not prop[i]][:max_evanescent]
    keep.update(evanescent)

    def row(i):
        return {
            "m": orders[i][0], "n": orders[i][1],
            "kx": round(float(kx[i]), 6), "ky": round(float(ky[i]), 6),
            "kz_out": round(float(kz_out[i].real), 6),
            "propagating": bool(prop[i]),
            "power_t": float(power_t[i]), "power_r": float(power_r[i]),
            "abs_t": round(float(abs_t[i]), 6),
            "phase_t_deg": round(float(np.degrees(np.angle(t_co[i]))), 3),
            "is_target": list(orders[i]) == list(target),
        }

    rows = sorted((row(i) for i in keep),
                  key=lambda r: (not r["propagating"], r["m"], r["n"]))
    return {
        "rows": rows[:200],
        "total_T": float(result.total_T), "total_R": float(result.total_R),
        "energy_residual": float(result.energy_residual),
    }


FIELD_VIEWS_ALL = ("re_xz", "int_xz", "phase_xz", "xy_out", "xy_layer")


def fields_block(result, cfg, level, theta, phi, z_max_nm=None, binary=None, target=None,
                 z_in_nm=None, views=None, nx_pts=360, nz_in=None, nz_layer=40, nz_out=None, ny_xy=48, nx_xy=300):
    """Field maps of the delivered pattern from the modal solution.

    The two half spaces come from the reflected / transmitted order
    amplitudes (the incident plane wave added on the incident side).  The
    layer interior comes from the layer eigenmodes (q, W, V): the forward
    amplitudes are fixed by tangential E/H continuity at the incident
    interface and the backward ones at the output interface (each referenced
    to its own launching plane, so evanescent modes never grow), and Ez follows
    from Ampere's law, Ez = -(kx Hy - ky Hx) / eps(x, y).  z = 0 is the incident
    interface and z = h the output interface; x spans two periods at y = y0
    (the y symmetry plane).  Evanescent orders and modes are kept."""
    views = set(views) if views else {"re_xz"}
    views.add("re_xz")                      # Re(E_dominant) x-z is always drawn; the others are optional
    F = result.fields or {}
    if "layer_modes" not in F or len(F["layer_modes"]) != 1:
        raise ValueError("field maps need the layer modes of a single-layer solve (EXPORT_LAYER_MODES)")
    with torch.no_grad():
        kx, ky, kz_in, kz_out = F["kx"], F["ky"], F["kz_in"], F["kz_out"]
        q, W, Vl = F["layer_modes"][0]
        v_in, v_out, inc = F["v_in"], F["v_out"], F["incident"]
        cdt, dev = W.dtype, W.device
        rdt = torch.float64 if cdt == torch.complex128 else torch.float32
        N = kx.shape[0]
        k0 = 2.0 * math.pi / cfg.wavelength_nm
        h = float(cfg.thickness_nm)
        lx, ly = cfg.lattice_nm
        tt, tr = F["tangential_t"], F["tangential_r"]
        t = torch.cat((tt[:, 0], tt[:, 1])).to(cdt)
        r = torch.cat((tr[:, 0], tr[:, 1])).to(cdt)
        # layer amplitudes: [W W; V -V] [c+ ; c- X] = [inc + r ; V_in (inc - r)] at z = 0,
        #                   [W W; V -V] [c+ X ; c-] = [t ; V_out t]             at z = h
        A = torch.cat((torch.cat((W, W), 1), torch.cat((Vl, -Vl), 1)), 0)
        rhs = torch.stack((torch.cat((inc + r, v_in @ (inc - r))), torch.cat((t, v_out @ t))), 1)
        sol = torch.linalg.solve(A, rhs)
        c_plus, c_minus = sol[:2 * N, 0], sol[2 * N:, 1]
        del A, sol
        # grids
        z_max = float(z_max_nm) if z_max_nm else 2500.0
        z_in = float(z_in_nm) if z_in_nm else 1000.0
        dz = max(min(z_in, h, z_max) / 25.0, 8.0)              # common z resolution (nm), ~25 rows in the thinnest band
        nz_in = int(nz_in or max(20, min(160, round(z_in / dz))))
        nz_out = int(nz_out or max(30, min(260, round(z_max / dz))))
        z_in_v = np.linspace(-z_in, 0.0, nz_in, endpoint=False)
        z_lay_v = np.linspace(0.0, h, nz_layer, endpoint=False)
        z_out_v = np.linspace(h, h + z_max, nz_out)
        x_v = np.linspace(0.0, 2.0 * lx, nx_pts, endpoint=False)
        y0 = ly / 2.0
        xt = torch.as_tensor(x_v, dtype=rdt, device=dev)
        P = torch.exp(1j * k0 * (kx[None, :] * xt[:, None] + ky[None, :] * y0))      # (nx, N)
        Pt = P.transpose(0, 1)
        i0 = result.orders.index((0, 0))
        jones = inc[[i0, N + i0]]
        dominant = 0 if abs(complex(jones[0])) >= abs(complex(jones[1])) else 1

        def safe(kz):
            return torch.where(kz.abs() < 1e-9, kz + 1e-9, kz)

        def halfspace(amp, kz, sgn, z_rel):
            """E(z, x) of a set of orders with z-dependence exp(i k0 sgn kz z_rel)."""
            z = torch.as_tensor(z_rel, dtype=rdt, device=dev)
            ph = torch.exp(1j * k0 * sgn * kz[None, :] * z[:, None])                # (nz, N)
            ex, ey = amp[:N], amp[N:]
            ez = -(kx * ex + ky * ey) / (sgn * safe(kz))
            return (ph * ex[None, :]) @ Pt, (ph * ey[None, :]) @ Pt, (ph * ez[None, :]) @ Pt

        # incident side: incident plane wave (order (0,0) only; the other orders must not enter
        # the sum, their evanescent phase overflows) + reflected orders
        z_i = torch.as_tensor(z_in_v, dtype=rdt, device=dev)
        ph_i = torch.exp(1j * k0 * kz_in[i0] * z_i)[:, None]
        ez_i0 = -(kx[i0] * inc[i0] + ky[i0] * inc[N + i0]) / safe(kz_in[i0:i0 + 1])[0]
        Ex_i, Ey_i, Ez_i = [ph_i * (c * P[:, i0])[None, :] for c in (inc[i0], inc[N + i0], ez_i0)]
        Ex_r, Ey_r, Ez_r = halfspace(r, kz_in, -1.0, z_in_v)
        Ex_in, Ey_in, Ez_in = Ex_i + Ex_r, Ey_i + Ey_r, Ez_i + Ez_r
        # output side
        Ex_o, Ey_o, Ez_o = halfspace(t, kz_out, +1.0, z_out_v - h)
        # layer interior
        zl = torch.as_tensor(z_lay_v, dtype=rdt, device=dev)
        ph_p = torch.exp(1j * k0 * q[None, :] * zl[:, None])                          # (nz, 2N)
        ph_m = torch.exp(1j * k0 * q[None, :] * (h - zl)[:, None])
        Ecoef = (c_plus[None, :] * ph_p + c_minus[None, :] * ph_m) @ W.transpose(0, 1)   # (nz, 2N)
        Hcoef = (c_plus[None, :] * ph_p - c_minus[None, :] * ph_m) @ Vl.transpose(0, 1)
        Ex_l = Ecoef[:, :N] @ Pt
        Ey_l = Ecoef[:, N:] @ Pt
        Dz_l = -(Hcoef[:, N:] * kx[None, :] - Hcoef[:, :N] * ky[None, :]) @ Pt
        # eps along the y0 row (two periods) from the delivered binary
        dy, dx = cfg.pixel_yx
        dens = binary if binary is not None else None
        if dens is None:
            eps_x = torch.full((nx_pts,), float(cfg.eps_feature), dtype=rdt, device=dev)
            row = np.ones(cfg.nx)
        else:
            row = np.asarray(dens.detach().cpu().numpy() if torch.is_tensor(dens) else dens, dtype=np.float64)
            row = row[min(int(y0 / dy), row.shape[0] - 1), :]
            idx = np.minimum((x_v / dx).astype(int) % cfg.nx, cfg.nx - 1)
            eps_x = torch.as_tensor(cfg.eps_background + (cfg.eps_feature - cfg.eps_background) * row[idx],
                                    dtype=rdt, device=dev)
        Ez_l = Dz_l / eps_x[None, :]
        Ex = torch.cat((Ex_in, Ex_l, Ex_o), 0)
        Ey = torch.cat((Ey_in, Ey_l, Ey_o), 0)
        Ez = torch.cat((Ez_in, Ez_l, Ez_o), 0)
        E_dom = Ex if dominant == 0 else Ey
        re_dom = E_dom.real.cpu().numpy()
        intensity = (Ex.abs() ** 2 + Ey.abs() ** 2 + Ez.abs() ** 2).cpu().numpy() if "int_xz" in views else None
        phase = torch.angle(E_dom).cpu().numpy() if "phase_xz" in views else None
        # material runs along the y0 row over two periods (for the outline)
        runs = []
        on = None
        for i, v in enumerate(list(row >= 0.5) + [False]):
            if v and on is None:
                on = i
            elif not v and on is not None:
                runs.append([on * dx, i * dx])
                on = None
        runs = runs + [[a + lx, b + lx] for a, b in runs]
        # x-y planes (optional): two periods in x, at z = h + lambda (output) and z = h/2 (inside the layer)
        I_out = ph_out = I_lay = None
        xy_x = np.linspace(0.0, 2.0 * lx, nx_xy, endpoint=False)
        xy_y = np.linspace(0.0, ly, ny_xy, endpoint=False)
        z_plane = cfg.wavelength_nm
        zm = 0.5 * h
        if views & {"xy_out", "xy_layer"}:
            Xg, Yg = np.meshgrid(xy_x, xy_y)
            Xt = torch.as_tensor(Xg.ravel(), dtype=rdt, device=dev)
            Yt = torch.as_tensor(Yg.ravel(), dtype=rdt, device=dev)
            Pxy = torch.exp(1j * k0 * (kx[None, :] * Xt[:, None] + ky[None, :] * Yt[:, None]))   # (npts, N)
            if "xy_out" in views:
                ph_o = torch.exp(1j * k0 * kz_out * z_plane)
                ex_o, ey_o = t[:N] * ph_o, t[N:] * ph_o
                ez_o = -(kx * ex_o + ky * ey_o) / safe(kz_out)
                Exy = [(Pxy @ c).reshape(ny_xy, nx_xy) for c in (ex_o, ey_o, ez_o)]
                I_out = (Exy[0].abs() ** 2 + Exy[1].abs() ** 2 + Exy[2].abs() ** 2).cpu().numpy()
                ph_out = torch.angle(Exy[dominant]).cpu().numpy()
            if "xy_layer" in views:
                ph_pm = torch.exp(1j * k0 * q * zm)
                ph_mm = torch.exp(1j * k0 * q * (h - zm))
                ec = W @ (c_plus * ph_pm + c_minus * ph_mm)
                hc = Vl @ (c_plus * ph_pm - c_minus * ph_mm)
                Exl = (Pxy @ ec[:N]).reshape(ny_xy, nx_xy)
                Eyl = (Pxy @ ec[N:]).reshape(ny_xy, nx_xy)
                Dzl = -(Pxy @ (hc[N:] * kx - hc[:N] * ky)).reshape(ny_xy, nx_xy)
                if dens is not None:
                    full = np.asarray(dens.detach().cpu().numpy() if torch.is_tensor(dens) else dens, dtype=np.float64)
                    iy = np.minimum((Yg / dy).astype(int) % cfg.ny, cfg.ny - 1)
                    ix = np.minimum((Xg / dx).astype(int) % cfg.nx, cfg.nx - 1)
                    eps_xy = torch.as_tensor(cfg.eps_background + (cfg.eps_feature - cfg.eps_background) * full[iy, ix],
                                             dtype=rdt, device=dev)
                else:
                    eps_xy = torch.full((ny_xy, nx_xy), float(cfg.eps_feature), dtype=rdt, device=dev)
                Ezl = Dzl / eps_xy
                I_lay = (Exl.abs() ** 2 + Eyl.abs() ** 2 + Ezl.abs() ** 2).cpu().numpy()
        # target direction in the output medium
        tgt = None
        if target is not None and tuple(target) in result.orders:
            it = result.orders.index(tuple(target))
            kxt = float(kx[it].real)
            kyt = float(ky[it].real)
            kt = math.hypot(kxt, kyt)
            if kt <= cfg.n_output:
                tgt = {"order": list(target), "angle_deg": math.degrees(math.asin(kt / cfg.n_output)),
                       "sign_x": 1.0 if kxt >= 0 else -1.0,
                       "power_t": float(result.power_t[it])}
    z_all = np.concatenate((z_in_v, z_lay_v, z_out_v))
    out = {
        "views": sorted(views),
        "xz": {"x_nm": _round_list(x_v, 2), "z_nm": _round_list(z_all, 2), "y0_nm": round(y0, 2),
               "interfaces_nm": [0.0, h], "material_runs_nm": [[round(a, 2), round(b, 2)] for a, b in runs],
               "re_dom": _round_list(re_dom, 4), "dominant": "Ex" if dominant == 0 else "Ey"},
        "target": tgt,
    }
    if intensity is not None:
        out["xz"]["intensity"] = _round_list(intensity, 4)
    if phase is not None:
        out["xz"]["phase"] = _round_list(phase, 4)
    if I_out is not None:
        out["xy"] = {"x_nm": _round_list(xy_x, 2), "y_nm": _round_list(xy_y, 2), "z0_nm": round(h + z_plane, 2),
                     "intensity": _round_list(I_out, 4), "phase": _round_list(ph_out, 4)}
    if I_lay is not None:
        out["xy_layer"] = {"x_nm": _round_list(xy_x, 2), "y_nm": _round_list(xy_y, 2), "z0_nm": round(zm, 2),
                           "intensity": _round_list(I_lay, 4)}
    return dict(out, **{
        "note": M("fields_note"),
    })


def torcwa_block(density, levels, cfg, theta, phi, target, dtype, device):
    try:
        import torcwa
    except Exception as error:
        return {"available": False, "error": M("torcwa_missing", err=error)}
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    eps_xy = (cfg.eps_background + (cfg.eps_feature - cfg.eps_background)
              * density.transpose(0, 1).to(device=device, dtype=real_dtype))
    out = {"available": True, "rows": [],
           "note": M("torcwa_note")
                   if cfg.incident_jones != (1.0 + 0j, 0.0 + 0j) else ""}
    for level in levels:
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.time()
        sim = torcwa.rcwa(freq=1.0 / cfg.wavelength_nm,
                          order=[int(level[0]), int(level[1])],
                          L=list(cfg.lattice_nm), dtype=dtype, device=device,
                          stable_eig_grad=False)
        sim.add_input_layer(eps=cfg.n_incident ** 2)
        sim.add_output_layer(eps=cfg.n_output ** 2)
        sim.set_incident_angle(inc_ang=math.radians(theta),
                               azi_ang=math.radians(phi))
        sim.add_layer(cfg.thickness_nm, eps_xy)
        sim.solve_global_smatrix()

        def power(order):
            txx = sim.S_parameters(orders=list(order), direction="forward",
                                   port="transmission", polarization="xx",
                                   ref_order=[0, 0])
            tyx = sim.S_parameters(orders=list(order), direction="forward",
                                   port="transmission", polarization="yx",
                                   ref_order=[0, 0])
            return float(torch.sum(torch.abs(txx) ** 2 + torch.abs(tyx) ** 2))

        eta = power(target)
        torch.cuda.synchronize() if device == "cuda" else None
        row = {"level": list(level), "eta_target": eta, "s": time.time() - t0}
        # per-propagating-order comparison at the last (highest) level only
        if list(level) == list(levels[-1]):
            orders, kx, ky = k_vectors(cfg, level, theta, phi)
            kz = outgoing_sqrt(torch.as_tensor(
                cfg.n_output ** 2 - kx ** 2 - ky ** 2,
                dtype=torch.complex128)).numpy()
            per = []
            for i, order in enumerate(orders):
                if kz[i].real > 1e-9:
                    per.append({"m": order[0], "n": order[1],
                                "power_t": power(order)})
            row["per_order"] = per
        out["rows"].append(row)
        del sim
        if device == "cuda":
            torch.cuda.empty_cache()
    return out


# ------------------------------------------------- TORCWA as an OPTIMIZER
# The fair head-to-head the user asked for: same design map, same Adam, same
# schedule, same seeds -- only the physics value+gradient engine differs.
# TORCWA gives d(eta)/d(density); we chain it through OUR (autograd) design map
# to get d(eta)/d(latent), so both optimizers steer the identical parameters.
# x-polarized convention (xx+yx), the same one used for the engine parity check.
def _torcwa_sim_app(density, level, cfg, theta, phi, dtype, device):
    import torcwa
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    eps_xy = (cfg.eps_background + (cfg.eps_feature - cfg.eps_background)
              * density.transpose(0, 1).to(device=device, dtype=real_dtype))
    sim = torcwa.rcwa(freq=1.0 / cfg.wavelength_nm,
                      order=[int(level[0]), int(level[1])],
                      L=list(cfg.lattice_nm), dtype=dtype, device=device,
                      stable_eig_grad=True)
    sim.add_input_layer(eps=cfg.n_incident ** 2)
    sim.add_output_layer(eps=cfg.n_output ** 2)
    sim.set_incident_angle(inc_ang=math.radians(theta), azi_ang=math.radians(phi))
    sim.add_layer(cfg.thickness_nm, eps_xy)
    sim.solve_global_smatrix()
    return sim


def _torcwa_power(sim, target):
    txx = sim.S_parameters(orders=list(target), direction="forward",
                           port="transmission", polarization="xx", ref_order=[0, 0])
    tyx = sim.S_parameters(orders=list(target), direction="forward",
                           port="transmission", polarization="yx", ref_order=[0, 0])
    return torch.sum(torch.abs(txx) ** 2 + torch.abs(tyx) ** 2)


def torcwa_value_grad_latent(rho, level, target, beta, cfg, mode, theta, phi,
                             dtype, device):
    x = rho.detach().clone().requires_grad_(True)
    dens = app_projected(x, beta, cfg, mode)          # our differentiable map
    d = dens.detach().clone().requires_grad_(True)
    sim = _torcwa_sim_app(d, level, cfg, theta, phi, dtype, device)
    j = _torcwa_power(sim, target)
    g_dens = torch.autograd.grad(j, d)[0]
    dens.backward(g_dens.to(dens.dtype))
    return float(j.detach()), x.grad.detach()


def torcwa_eta_density(density, level, target, cfg, theta, phi, dtype, device):
    with torch.no_grad():
        sim = _torcwa_sim_app(density, level, cfg, theta, phi, dtype, device)
        return float(_torcwa_power(sim, target))


# ------------------------------------------------------ Ansys Lumerical RCWA
def lumerical_supported(cfg, theta, polarization):
    """The P-source -> x-pol mapping is only valid at normal incidence with an
    x-polarized source (the regime in which the P-source mapping was validated)."""
    if polarization != "x":
        return False, M("lum_pol")
    if abs(theta) > 1e-9:
        return False, M("lum_theta")
    return True, ""


def lumerical_block(binary, cfg, theta, phi, target, kvecs):
    """Live Lumerical RCWA on the delivered binary: eta(target) + per-order,
    at a ladder of 'max number k vectors'.  Returns availability + reason."""
    try:
        import numpy as _np
        info = lumerical_env()
        if not info["importable"]:
            raise ImportError(info["error"])
        import lumapi
        from workbench.polygons import export
    except Exception as error:
        return {"available": False, "error": M("lum_missing", err=error)}

    lam = cfg.wavelength_nm * 1e-9
    lx, ly = cfg.lattice_nm[0] * 1e-9, cfg.lattice_nm[1] * 1e-9
    height = cfg.thickness_nm * 1e-9
    mask = _np.asarray(binary) >= 0.5
    ny, nx = mask.shape
    try:
        outers, holes = export(mask, pixel_nm=cfg.pixel_yx, lattice_nm=cfg.lattice_nm)
    except AssertionError as error:
        return {"available": False, "error": M("lum_polygons", err=error)}
    out = {"available": True, "rows": [], "kvecs": list(kvecs),
           "note": M("lum_note")}
    fdtd = None
    try:
        fdtd = lumapi.FDTD(hide=True)
        for n_k in kvecs:
            fdtd.switchtolayout()
            fdtd.deleteall()
            fdtd.addrcwa()
            fdtd.set("name", "RCWA")
            for k, v in [("propagation axis", "z"), ("propagation direction", "forward"),
                         ("simulation region", "3D"), ("incident angle", "single"),
                         ("angle theta", 0.0), ("angle phi", 0.0),
                         ("x span", lx), ("y span", ly),
                         ("z min", -0.8 * lam), ("z max", height + lam),
                         ("background material", "<Object defined dielectric>"),
                         # simulation ambient = output half space (air side); the
                         # incidence substrate is added as an explicit rect below
                         ("index", cfg.n_output), ("interface position", "absolute"),
                         ("minimum wavelength", lam), ("maximum wavelength", lam),
                         ("frequency points", 1), ("max number k vectors", float(n_k)),
                         ("report grating orders", 1), ("report grating power", 1)]:
                fdtd.setnamed("RCWA", k, v)
            fdtd.setnamed("RCWA", "interface absolute positions",
                          _np.array([0.0, height]))
            # incidence-side substrate half space
            fdtd.addrect()
            fdtd.set("name", "substrate")
            for k, v in [("x", 0.0), ("y", 0.0), ("x span", lx), ("y span", ly),
                         ("z min", -0.8 * lam), ("z max", 0.0),
                         ("index", cfg.n_incident)]:
                fdtd.setnamed("substrate", k, v)
            n_feat = float(cfg.eps_feature) ** 0.5
            n_bg = float(cfg.eps_background) ** 0.5
            for i, poly in enumerate(outers):
                fdtd.addpoly(); name = "mat_%02d" % i
                fdtd.set("name", name)
                fdtd.setnamed(name, "vertices", poly * 1e-9)
                fdtd.setnamed(name, "z min", 0.0)
                fdtd.setnamed(name, "z max", height)
                fdtd.setnamed(name, "index", n_feat)
            for i, poly in enumerate(holes):
                fdtd.addpoly(); name = "hole_%02d" % i
                fdtd.set("name", name)
                fdtd.setnamed(name, "vertices", poly * 1e-9)
                fdtd.setnamed(name, "z min", 0.0)
                fdtd.setnamed(name, "z max", height)
                fdtd.setnamed(name, "index", n_bg)
                fdtd.setnamed(name, "override mesh order from material database", 1)
                fdtd.setnamed(name, "mesh order", 2)
            t0 = time.time()
            fdtd.run()
            gp = fdtd.getresult("RCWA", "grating_power")
            te = fdtd.getresult("RCWA", "total_energy")
            n = _np.asarray(gp["n"]).ravel().astype(int)
            m = _np.asarray(gp["m"]).ravel().astype(int)
            tp = _np.asarray(gp["Tp_power"])[0, 0]
            i_m0 = int(_np.where(m == 0)[0][0])
            row = {int(nn): float(tp[i, i_m0]) for i, nn in enumerate(n)}
            out["rows"].append({
                "kvec": int(n_k),
                "eta_target": row.get(int(target[0]), float("nan")),
                "Tp_total": float(_np.asarray(te["Tp"]).ravel()[0]),
                "Rp_total": float(_np.asarray(te["Rp"]).ravel()[0]),
                "s": round(time.time() - t0, 2),
                "per_order": [{"m": int(nn), "n": 0, "power_t": row[int(nn)]}
                              for nn in sorted(row) if abs(row[int(nn)]) > 1e-9],
                "n_polygons": [len(outers), len(holes)],
            })
    except Exception as error:
        out = {"available": False, "error": M("lum_failed", err=error)}
    finally:
        if fdtd is not None:
            try:
                fdtd.close()
            except Exception:
                pass
    return out


def pattern_block(latent, cfg, mode):
    with torch.no_grad():
        filt = app_filtered(latent.cpu(), cfg, mode)
        binary = (filt >= cfg.threshold).to(torch.float64)
    return {"latent": _round_list(latent.detach().cpu().numpy(), 4),
            "filtered": _round_list(filt.numpy(), 4),
            "binary": binary.numpy().astype(int).tolist()}


# ----------------------------------------------------------------- job types
# Default optimizer = guarded mixed-order tickets (final scheme v2, see the
# block before run_optimize and workbench/descent.py).  The pre-S6 chain
# protocol (start restarts -> 40 basin-hopping jumps at beta = 64, monotone
# acceptance of the hard binary at the GATE order) is kept as
# "legacy_chain"; e10_descent / redescend / chain_protocol / restart_arm
# below belong to it and are unchanged.
from types import SimpleNamespace


def order_rule(target, cfg):
    """Order-scaling rule: descent / gate / certification orders from the target
    order and the in-medium band n_feat*Lambda/lambda, clipped to the grid.

    Reproduces the reference protocol at 2400x700 nm, n=3.2, 660 nm:
      m<=2: descent (12,3), gate (24,6), ladder (32,8),(40,10)
      m=3 : descent (20,5), gate (32,8), ladder (40,10),(48,12)
    """
    m_eff = max(abs(int(target[0])), abs(int(target[1])))
    n_feat = float(cfg.eps_feature) ** 0.5
    base_x = n_feat * cfg.lattice_nm[0] / cfg.wavelength_nm
    aspect = 0.857 * cfg.lattice_nm[1] / cfg.lattice_nm[0]   # protocol: My = Mx/4
    if m_eff <= 2:
        k_d, k_g = 1.0, 2.0
    else:
        k_d, k_g = (m_eff + 2) / 3.0, (m_eff + 5) / 3.0

    def level(k):
        mx = max(4, 4 * int(round(base_x * k / 4.0)))
        my = max(1, int(round(mx * aspect)))
        return (mx, my)

    descent, gate = level(k_d), level(k_g)
    ladder = [level(k_g + 2.0 / 3.0), level(k_g + 4.0 / 3.0)]
    max_mx, max_my = (cfg.nx - 1) // 4, (cfg.ny - 1) // 4

    def clip(lv):
        return (min(lv[0], max_mx), min(lv[1], max_my))

    unclipped = {"descent": descent, "gate": gate, "ladder": ladder}
    out = {"m_eff": m_eff, "descent": clip(descent), "gate": clip(gate),
           "ladder": [], "unclipped": {k: (list(v) if k != "ladder" else [list(x) for x in v])
                                       for k, v in unclipped.items()},
           "grid_needed": {"nx": 4 * ladder[-1][0] + 1, "ny": 4 * ladder[-1][1] + 1}}
    seen = {out["descent"], out["gate"]}
    for lv in ladder:
        c = clip(lv)
        if c not in seen and C.n_terms(c) > C.n_terms(out["gate"]):
            out["ladder"].append(c)
            seen.add(c)
    out["clipped"] = (clip(descent) != descent or clip(gate) != gate
                      or any(clip(lv) != lv for lv in ladder))
    out["gate_ok"] = C.n_terms(out["gate"]) > C.n_terms(out["descent"])
    return out


def filtered_noise(seed, cfg):
    """Jump noise: unit-std periodic Gaussian-filtered field."""
    g = torch.Generator().manual_seed(int(seed))
    noise = torch.randn((cfg.ny, cfg.nx), generator=g, dtype=cfg.latent_dtype)
    smooth = periodic_gaussian_filter(noise, cfg.lattice_nm, cfg.filter_nm)
    return smooth / smooth.std()


def gate_eval(binary, level, ctx):
    with torch.no_grad():
        r = solve_density(binary, level, ctx.cfg, ctx.theta, ctx.phi, ctx.dtype, ctx.device)
    return eta_of(r, ctx.target)


def gate_score(binary, ctx):
    """Acceptance score of a hard binary at the gate order.  'nominal' = eta;
    'maximin_cd' = min{eta, eta(cd+b), eta(cd-b)} (three-scenario rule)."""
    detail = {"nominal": round(gate_eval(binary, ctx.gate, ctx), 6)}
    if ctx.accept_rule == "maximin_cd":
        for b in ctx.cd_biases:
            biased = torch.from_numpy(C.cd_bias_mask(binary.numpy() >= 0.5, b, ctx.cfg))
            detail["cd%+g" % b] = round(gate_eval(biased, ctx.gate, ctx), 6)
        score = min(detail.values())
    else:
        score = detail["nominal"]
    return float(score), detail


def certify_ladder(binary, ladder, ctx, job=None, tag=""):
    rows, r_top = [], None
    for lv in ladder:
        if job is not None:
            if job.get("cancel"):
                raise KeyboardInterrupt
            job["progress"] = {"phase": "certify", "tag": tag, "level": list(lv)}
        t0 = time.time()
        with torch.no_grad(), _export_modes():
            r = solve_density(binary, lv, ctx.cfg, ctx.theta, ctx.phi, ctx.dtype, ctx.device)
        rows.append({"level": list(lv), "two_n": 2 * C.n_terms(lv),
                     "eta": round(eta_of(r, ctx.target), 6), "s": round(time.time() - t0, 2)})
        r_top = r
    rows[-1]["total_T"] = round(float(r_top.total_T), 6)
    return rows, r_top


def e10_descent(job, ctx, seed, steps, engine, key, ckpt_every):
    """Beta-continuation descent from a random seed, with hard-binary
    checkpoints scored at the GATE order (our engine, neutral).  The delivered
    latent of a descent is its best gate checkpoint, never the relaxed end value."""
    cfg, mode, device = ctx.cfg, ctx.mode, ctx.device
    latent = app_initial_latent(seed, cfg, mode, device)
    adam = AppAdam(latent.shape, cfg, mode, device)
    traj, ckpts = [], []
    best = None
    t0 = time.time()
    for t in range(steps):
        if job.get("cancel"):
            raise KeyboardInterrupt
        progress = t / max(steps - 1, 1)
        beta = C.beta_at(progress, cfg=cfg)
        lr = C.cosine_lr(progress, cfg)
        if engine == "ours":
            eta, grad = value_grad(latent, ctx.level_d, ctx.target, beta, cfg, mode,
                                   ctx.theta, ctx.phi, ctx.dtype, device, ctx.offload)
        else:
            eta, grad = torcwa_value_grad_latent(latent, ctx.level_d, ctx.target, beta,
                                                 cfg, mode, ctx.theta, ctx.phi,
                                                 ctx.dtype, device)
        latent = adam.step(latent, grad, lr)
        traj.append(round(eta, 6))
        if (t + 1) % ckpt_every == 0 or t == steps - 1:
            binary = app_hard_binary(latent.detach().cpu(), cfg, mode)
            eg = gate_eval(binary, ctx.gate, ctx)
            ckpts.append({"step": t + 1, "eta_gate": round(eg, 6), "beta": round(beta, 2)})
            if best is None or eg > best["eta_gate"]:
                best = {"eta_gate": eg, "step": t + 1,
                        "latent": latent.detach().cpu().clone(), "binary": binary}
        job["progress"] = {"phase": "descent", "opt_engine": engine, "key": key,
                           "seed": seed, "step": t + 1, "steps": steps,
                           "eta": round(eta, 6),
                           "eta_gate_best": round(best["eta_gate"], 6) if best else None}
        job["trajectories"][key] = traj
        job["checkpoints"][key] = ckpts
    return {"seed": seed, "engine": engine, "key": key, "trajectory": traj,
            "checkpoints": ckpts, "best": best,
            "eta_final_relaxed": traj[-1] if traj else None,
            "eta_gate_best": round(best["eta_gate"], 6), "best_step": best["step"],
            "wall_s": round(time.time() - t0, 2)}


def redescend(rho0, ctx, steps, beta):
    """Chain re-descent: fixed beta (64 by protocol), fresh Adam, cosine lr."""
    rho = rho0.to(ctx.device)
    adam = AppAdam(rho.shape, ctx.cfg, ctx.mode, ctx.device)
    for t in range(steps):
        lr = C.cosine_lr(t / steps, ctx.cfg)
        _, grad = value_grad(rho, ctx.level_d, ctx.target, beta, ctx.cfg, ctx.mode,
                             ctx.theta, ctx.phi, ctx.dtype, ctx.device, ctx.offload)
        rho = adam.step(rho, grad, lr)
    return rho.detach().cpu()


def _load_saved_latent(run_id, cfg):
    name = _safe_run_name(run_id)
    npy = os.path.join(RUNS_DIR, name[:-5] + "_latent.npy")
    if os.path.exists(npy):   # full-precision incumbent, written at delivery
        latent = torch.from_numpy(np.load(npy).astype("float64")).to(cfg.latent_dtype)
    else:                     # older runs: the JSON pattern (4-decimal rounded)
        with open(os.path.join(RUNS_DIR, name), encoding="utf-8") as f:
            saved = json.load(f)
        latent = torch.tensor(saved["result"]["patterns"]["latent"], dtype=cfg.latent_dtype)
    if latent.shape != (cfg.ny, cfg.nx):
        raise ValueError("saved latent grid %s does not match the current grid (ny=%d, nx=%d)"
                         % (tuple(latent.shape), cfg.ny, cfg.nx))
    return latent


def chain_protocol(job, ctx, p):
    cfg, mode = ctx.cfg, ctx.mode
    n_jumps = int(p.get("chain_jumps", 40))
    alphas = [float(a) for a in (p.get("chain_alphas") or [0.1, 0.25, 0.5])]
    re_steps = int(p.get("chain_redescent_steps", 30))
    re_beta = float(p.get("chain_beta", 64.0))
    steps = int(p.get("steps", 60))
    ckpt_every = int(p.get("ckpt_every") or max(5, steps // 10))
    seeds = [int(s) for s in p.get("seeds", [2029, 2030])]
    start = p.get("chain_start") or {"kind": "restarts"}
    noise_base = int(p.get("chain_noise_seed", 9100))

    # ---- start: honest selection at the gate
    t_start = time.time()
    start_runs = []
    if start.get("kind") == "run_best":
        latent = _load_saved_latent(start["run_id"], cfg)
        origin = M("origin_saved", run=start["run_id"])
    else:
        use = seeds[:1] if start.get("kind") == "seed" else seeds
        for seed in use:
            start_runs.append(e10_descent(job, ctx, seed, steps, "ours",
                                          "start:%d" % seed, ckpt_every))
        pick = max(start_runs, key=lambda r: r["eta_gate_best"])
        latent = pick["best"]["latent"]
        origin = M("origin_restart", seed=pick["seed"], step=pick["best_step"])
    inc_latent = latent
    inc_binary = app_hard_binary(inc_latent, cfg, mode)
    inc_score, inc_detail = gate_score(inc_binary, ctx)
    start_wall = time.time() - t_start

    # ---- basin hopping
    jumps = []
    job["chain_jumps"] = jumps
    job["trajectories"]["chain:incumbent"] = [round(inc_score, 6)]
    job["trajectories"]["chain:candidate"] = [round(inc_score, 6)]
    accepts = 0
    t_chain = time.time()
    for j in range(n_jumps):
        if job.get("cancel"):
            raise KeyboardInterrupt
        alpha = alphas[j % len(alphas)]
        noise = filtered_noise(noise_base + j, cfg)
        pert = _sym((inc_latent + alpha * cfg.init_scale * noise).clamp(0.0, 1.0), mode)
        t_j = time.time()
        rho = redescend(pert, ctx, re_steps, re_beta)
        b = app_hard_binary(rho, cfg, mode)
        score, detail = gate_score(b, ctx)
        acc = score > inc_score
        if acc:
            inc_latent, inc_binary, inc_score, inc_detail = rho, b, score, detail
            accepts += 1
        jumps.append({"jump": j + 1, "alpha": alpha, "score": round(score, 6),
                      "detail": detail, "accepted": bool(acc),
                      "incumbent": round(inc_score, 6), "s": round(time.time() - t_j, 2)})
        job["trajectories"]["chain:incumbent"].append(round(inc_score, 6))
        job["trajectories"]["chain:candidate"].append(round(score, 6))
        job["progress"] = {"phase": "chain", "jump": j + 1, "n_jumps": n_jumps,
                           "alpha": alpha, "score": round(score, 6),
                           "incumbent": round(inc_score, 6), "accepts": accepts}
    chain_wall = time.time() - t_chain
    return {
        "origin": origin, "start_runs": start_runs,
        "start_score": round(job["trajectories"]["chain:incumbent"][0], 6),
        "jumps": jumps, "accepts": accepts, "n_jumps": n_jumps,
        "alphas": alphas, "redescent_steps": re_steps, "beta": re_beta,
        "incumbent": {"latent": inc_latent, "binary": inc_binary,
                      "score": round(inc_score, 6), "detail": inc_detail},
        "start_wall_s": round(start_wall, 2), "chain_wall_s": round(chain_wall, 2),
        "total_wall_s": round(start_wall + chain_wall, 2),
    }


def restart_arm(job, ctx, p, budget_s, seed_base, max_restarts=64):
    """Equal-wall-clock restart lottery: keep drawing fresh seeds until the chain's
    budget is spent; each ticket delivers its best gate checkpoint."""
    steps = int(p.get("steps", 60))
    ckpt_every = int(p.get("ckpt_every") or max(5, steps // 10))
    runs = []
    t0 = time.time()
    i = 0
    while i < max_restarts:
        if i >= 1 and (time.time() - t0) + sum(r["wall_s"] for r in runs) / len(runs) > budget_s:
            break          # a new ticket is drawn only if it is expected to finish inside the budget
        seed = seed_base + i
        job["progress"] = {"phase": "restart_arm", "seed": seed, "n_done": i,
                           "budget_s": round(budget_s, 1),
                           "spent_s": round(time.time() - t0, 1)}
        runs.append(e10_descent(job, ctx, seed, steps, "ours", "rs:%d" % seed, ckpt_every))
        i += 1
    best = max(runs, key=lambda r: r["eta_gate_best"])
    gates = sorted(r["eta_gate_best"] for r in runs)
    return {"runs": [{k: r[k] for k in ("seed", "eta_gate_best", "best_step",
                                        "eta_final_relaxed", "wall_s")} for r in runs],
            "n": len(runs), "best_seed": best["seed"],
            "best": best["best"], "best_gate": best["eta_gate_best"],
            "median_gate": gates[len(gates) // 2], "min_gate": gates[0],
            "wall_s": round(time.time() - t0, 2), "budget_s": round(budget_s, 2)}


# ------------------------------------------------ guarded mixed-order protocol
# The default protocol (final scheme v2, docs/algorithm.md section 5): n
# tickets = guarded mixed-order descents on the README ramp, one per seed;
# selection by the gate reading of the best hard-binary checkpoint; the winner
# certified on the ladder; an OPTIONAL repair chain (basin hopping at a fixed
# low beta) on the delivered ticket, started always / on a numeric criterion /
# by hand from a saved run; an equal-wall-clock control arm only when the
# chain ran.  The algorithm lives in workbench/descent.py; this block is the
# job plumbing (parameters, progress, records).
def _raise_if_cancelled(job):
    if job.get("cancel"):
        raise KeyboardInterrupt


def make_ops():
    """The solver / design-map primitives the descent module works with."""
    return SimpleNamespace(solve_density=solve_density, app_projected=app_projected,
                           app_hard_binary=app_hard_binary, app_initial_latent=app_initial_latent,
                           AppAdam=AppAdam, filtered_noise=filtered_noise, sym=_sym)


def _fnum(p, key, default):
    v = p.get(key)
    return float(default) if v in (None, "") else float(v)


def _inum(p, key, default):
    v = p.get(key)
    return int(default) if v in (None, "") else int(v)


def gm_settings(p, ctx, levels_c, warnings):
    """Guarded-mixed protocol settings from the job params (UI ids gm_*, repair_*)."""
    steps = _inum(p, "steps", C.DESCENT["steps"])
    beta0 = _fnum(p, "gm_beta0", C.DESCENT["beta0"])
    beta_end = _fnum(p, "gm_beta_end", C.DESCENT["beta_end"])
    mix_k = max(1, _inum(p, "gm_mix_k", C.DESCENT["mix_k"]))
    mix_from_beta = _fnum(p, "gm_mix_from_beta", C.DESCENT["mix_from_beta"])
    guards = dict(C.GUARDS)
    guards["enabled"] = bool(p.get("gm_guards", True))
    cadence = dict(C.CADENCE, enabled=bool(p.get("gm_cadence", False)))
    rise_on = bool(p.get("gm_rise", False))
    if rise_on and not cadence["enabled"]:
        cadence["enabled"] = True
        warnings.append(M("warn_rise_needs_cadence"))
    rise = dict(C.RISE, enabled=rise_on)
    beta_fn = C.beta_geometric(beta0, beta_end)
    mix_start_step = C.first_step_at(beta_fn, steps, mix_from_beta)
    high_budget = len(range(mix_start_step, steps, mix_k)) if cadence["enabled"] else None
    ckpts = C.o_ckpts(steps)
    rung1 = tuple(levels_c[0]) if levels_c else tuple(ctx.gate)
    top = tuple(levels_c[-1]) if levels_c else tuple(ctx.gate)
    repair = dict(C.REPAIR)
    repair.update({
        "enabled": bool(p.get("repair_enabled", False)),
        "trigger": str(p.get("repair_trigger") or C.REPAIR["trigger"]),
        "bar": None if p.get("repair_bar") in (None, "") else float(p.get("repair_bar")),
        "optimism_max": _fnum(p, "repair_optimism_max", C.REPAIR["optimism_max"]),
        "beta_mode": str(p.get("repair_beta_mode") or C.REPAIR["beta_mode"]),
        "beta_fixed": _fnum(p, "repair_beta_fixed", C.REPAIR["beta_fixed"]),
        "beta_min": _fnum(p, "repair_beta_min", C.REPAIR["beta_min"]),
        "hops": _inum(p, "repair_hops", C.REPAIR["hops"]),
        "stop_after": _inum(p, "repair_stop_after", C.REPAIR["stop_after"]),
        "alphas": tuple(float(a) for a in (p.get("chain_alphas") or C.REPAIR["alphas"])),
        "re_steps": _inum(p, "chain_redescent_steps", C.REPAIR["re_steps"]),
        "re_k": _inum(p, "repair_re_k", C.REPAIR["re_k"]),
    })
    return SimpleNamespace(steps=steps, beta0=beta0, beta_end=beta_end, beta_fn=beta_fn, mix_k=mix_k,
                           mix_from_beta=mix_from_beta, mix_start_step=mix_start_step, high_budget=high_budget,
                           ckpts=ckpts, guards=guards, cadence=cadence, rise=rise, rung1=rung1, top=top,
                           repair=repair, noise_seed=_inum(p, "chain_noise_seed", 9100),
                           adaptive=cadence["enabled"])


def gm_summary(gm):
    return {"steps": gm.steps, "beta0": gm.beta0, "beta_end": gm.beta_end, "mix_k": gm.mix_k,
            "mix_from_beta": gm.mix_from_beta, "mix_start_step": gm.mix_start_step,
            "high_budget": gm.high_budget, "checkpoints": gm.ckpts, "n_ckpt": len(gm.ckpts),
            "guards": {k: (list(v) if isinstance(v, tuple) else v) for k, v in gm.guards.items()},
            "cadence": {k: (list(v) if isinstance(v, tuple) else v) for k, v in gm.cadence.items()},
            "rise": dict(gm.rise), "rung1": list(gm.rung1), "top": list(gm.top),
            "repair": {k: (list(v) if isinstance(v, tuple) else v) for k, v in gm.repair.items()},
            "noise_seed": gm.noise_seed}


def run_ticket(job, ctx, gm, seed, key, index, n_tickets, role="ticket"):
    """One guarded mixed-order ticket.  Returns (record, tensors); the record is
    JSON-safe (both deliveries read at rung 1), the tensors hold the best
    checkpoint's latent / binary and the final binary."""
    t_start = time.perf_counter()
    pol = D.GradPolicy(ctx, ctx.level_d, ctx.gate, gm.guards, gm.cadence, gm.rise, k=gm.mix_k,
                       mix_from_beta=gm.mix_from_beta, steps=gm.steps,
                       beta_fn=lambda t: gm.beta_fn(t, gm.steps), high_budget=gm.high_budget)

    def on_step(d):
        last = pol.rows[-1] if pol.rows else {}
        job["progress"] = {"phase": "descent", "opt_engine": "ours", "key": key, "seed": seed, "role": role,
                           "step": d.t, "steps": d.steps,
                           "eta": d.traj_descent[-1] if d.traj_descent and d.traj_descent[-1] is not None else d.traj[-1],
                           "eta_gate_best": round(d.best["eta_gate"], 6),
                           "ticket": index + 1, "n_tickets": n_tickets, "level": list(pol.level_d), "k": pol.k,
                           "high": bool(d.level_series[-1]) if d.level_series else False,
                           "last_guard_action": last.get("action"),
                           "n_true_positive": int(sum(pol.stats["n_true_positive"].values())),
                           "n_trigger": int(sum(pol.stats["n_trigger"].values()))}
        job["trajectories"][key] = d.traj_descent
        job["checkpoints"][key] = [{"step": c["step"], "eta_gate": round(c["eta_gate"], 6), "beta": round(c["beta"], 2),
                                    "eta_binary_descent": round(c["eta_binary_descent"], 6)} for c in d.ckpts]

    d = D.TicketDescent(ctx, seed, gm.steps, gm.beta_fn, gm.beta_end, gm.ckpts, pol, on_step=on_step)
    r = d.finish()
    on_step(d)
    deliv = {}
    for name in ("best", "final"):
        dd = r[name]
        if name == "final" and torch.equal(dd["binary"], r["best"]["binary"]):
            e1 = deliv["best"]["eta_rung1"]
        else:
            e1 = D.eta_binary(dd["binary"], gm.rung1, ctx)
        deliv[name] = {"step": dd["step"], "eta_gate": round(dd["eta_gate"], 6), "eta_rung1": round(e1, 6),
                       "gate_optimism": round(dd["eta_gate"] - e1, 6),
                       "beta": round(gm.beta_fn(dd["step"], gm.steps), 3)}
    cliff = max(c["eta_gate"] for c in r["checkpoints"]) - r["final"]["eta_gate"]
    gate_read = float(r["gate_read_s"] or 0.0)
    probe_wall = float(pol.stats["wall_probe_s"])
    st = pol.stats
    rec = {
        "seed": int(seed), "key": key, "index": index, "role": role, "engine": "ours",
        "deliver_best": deliv["best"], "deliver_final": deliv["final"],
        "best_step": deliv["best"]["step"], "beta_star": deliv["best"]["beta"],
        "eta_gate_best": deliv["best"]["eta_gate"], "eta_rung1": deliv["best"]["eta_rung1"],
        "gate_optimism": deliv["best"]["gate_optimism"],
        "eta_final_relaxed": round(r["reported_continuous"], 6), "reported_continuous": round(r["reported_continuous"], 6),
        "collapse_step": r["collapse_step"],
        "cliff_gate_best_minus_final": round(cliff, 6),
        "terminal_gap_descent_minus_gate": round(r["checkpoints"][-1]["gap_descent_minus_gate"], 6),
        "n_high_order_steps": r["n_high_order_steps"], "high_budget": gm.high_budget,
        "checkpoints": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items()} for c in r["checkpoints"]],
        "trajectory": r["trajectory"], "trajectory_descent": r["trajectory_descent"],
        "level_series": r["level_series"], "k_history": pol.k_history, "rise_step": pol.rise_step,
        "level_d_final": list(pol.level_d),
        "guard_stats": st, "events": pol.events, "steps_log": pol.rows,
        "wall_descent_s": round(r["wall_descent_s"], 2), "wall_gate_s": round(r["wall_gate_s"], 2),
        "wall_high_order_s": round(r["wall_high_order_s"], 2), "wall_probe_s": round(probe_wall, 2),
        "wall_recompute_s": round(st["wall_recompute_s"], 2), "wall_c128_s": round(st["wall_c128_s"], 2),
        "gate_read_s": round(gate_read, 3),
        # S7 convention: descent wall (probe excluded unless the cadence uses it) + 12 gate reads
        "ticket_cost_s": round(r["wall_descent_s"] - (0.0 if gm.adaptive else probe_wall) + 12 * gate_read, 2),
        "unit_wall_s": round(time.perf_counter() - t_start, 2),
    }
    tensors = {"latent": r["best"]["latent"], "binary": r["best"]["binary"], "final_binary": r["final"]["binary"]}
    return rec, tensors


def _ticket_row(t):
    st = t.get("guard_stats") or {}
    return {"seed": t["seed"], "engine": "ours", "role": t.get("role", "ticket"), "key": t.get("key"),
            "trajectory": t.get("trajectory_descent") or t.get("trajectory"), "checkpoints": t.get("checkpoints"),
            "eta_final_relaxed": t.get("eta_final_relaxed"), "eta_gate_best": t["eta_gate_best"],
            "best_step": t["best_step"], "beta_star": t.get("beta_star"), "eta_rung1": t.get("eta_rung1"),
            "gate_optimism": t.get("gate_optimism"), "final_rung1": (t.get("deliver_final") or {}).get("eta_rung1"),
            "cliff": t.get("cliff_gate_best_minus_final"), "terminal_gap": t.get("terminal_gap_descent_minus_gate"),
            "n_high": t.get("n_high_order_steps"), "collapse_step": t.get("collapse_step"),
            "guard": {"n_eval": st.get("n_eval"), "n_trigger": st.get("n_trigger"),
                      "n_true_positive": st.get("n_true_positive"), "n_c128": st.get("n_c128"),
                      "n_skip_high": st.get("n_skip_high"), "n_kept_original": st.get("n_kept_original")},
            "k_final": (t.get("k_history") or [{}])[-1].get("k") if t.get("k_history") else None,
            "rise_step": t.get("rise_step"), "level_d_final": t.get("level_d_final"),
            "wall_s": t.get("unit_wall_s"), "ticket_cost_s": t.get("ticket_cost_s"),
            "wall_descent_s": t.get("wall_descent_s"), "wall_gate_s": t.get("wall_gate_s")}


def _load_saved_run(run_id):
    name = _safe_run_name(run_id)
    with open(os.path.join(RUNS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def run_repair(job, ctx, gm, parent, tensors):
    """The repair chain on the delivered ticket; parent = its record."""
    R = gm.repair
    lat = tensors["latent"]
    binary = tensors["binary"]
    # re-read the parent at the gate and at rung 1 (never trust a saved number: orders / engine may differ)
    job["progress"] = {"phase": "certify", "tag": "repair parent", "level": list(ctx.gate)}
    p_gate = D.eta_binary(binary, ctx.gate, ctx)
    p_r1 = D.eta_binary(binary, gm.rung1, ctx)
    beta_star = parent.get("beta_star")
    if R["beta_mode"] == "star" and beta_star is not None:
        beta_re = max(float(beta_star), float(R["beta_min"]))
    else:
        beta_re = float(R["beta_fixed"])
    job["trajectories"]["chain:incumbent"] = [round(p_gate, 6)]
    job["trajectories"]["chain:candidate"] = [round(p_gate, 6)]
    hops = int(R["hops"])

    def on_hop(rec, inc_gate, n_acc, jumps):
        job["chain_jumps"] = jumps
        job["trajectories"]["chain:incumbent"].append(round(inc_gate, 6))
        job["trajectories"]["chain:candidate"].append(rec["score"])
        job["progress"] = {"phase": "chain", "jump": rec["jump"], "n_jumps": hops, "alpha": rec["alpha"],
                           "score": rec["score"], "incumbent": round(inc_gate, 6), "accepts": n_acc}

    def on_step(j, t):
        pr = job.get("progress") or {}
        job["progress"] = {"phase": "chain", "jump": j + 1, "n_jumps": hops,
                           "alpha": R["alphas"][j % len(R["alphas"])],
                           "score": pr.get("score", round(p_gate, 6)), "incumbent": pr.get("incumbent", round(p_gate, 6)),
                           "accepts": pr.get("accepts", 0), "re_step": t + 1, "re_steps": int(R["re_steps"])}

    record, inc = D.repair_chain(ctx, lat, binary, p_gate, p_r1, beta_re, ctx.level_d, ctx.gate, gm.rung1,
                                 R, gm.guards, gm.noise_seed, on_hop=on_hop, on_step=on_step)
    record.update({"parent_seed": parent.get("seed"), "parent_key": parent.get("key"),
                   "parent_beta_star": beta_star, "beta_mode": R["beta_mode"],
                   "parent_eta_gate_saved": parent.get("eta_gate_best"), "parent_eta_rung1_saved": parent.get("eta_rung1")})
    return record, inc


def control_arm_gm(job, ctx, gm, budget_s, seed_base, base_tickets, max_tickets=64):
    """Equal-wall-clock control: fresh tickets until the chain's wall clock is
    spent (a new ticket starts only if it is expected to finish inside the
    budget), then 'the original tickets + the extra ones, best by gate'."""
    runs, tensors = [], {}
    t0 = time.time()
    i = 0
    while i < max_tickets:
        elapsed = time.time() - t0
        if i >= 1:
            mean_wall = sum(r["unit_wall_s"] for r in runs) / len(runs)
            if elapsed + mean_wall > budget_s:
                break
        seed = seed_base + i
        job["progress"] = {"phase": "restart_arm", "seed": seed, "n_done": i,
                           "budget_s": round(budget_s, 1), "spent_s": round(elapsed, 1)}
        rec, ten = run_ticket(job, ctx, gm, seed, "rs:%d" % seed, i, None, role="control")
        runs.append(rec)
        tensors[rec["key"]] = ten
        i += 1
    pool = list(base_tickets) + runs
    best = max(pool, key=lambda r: r["eta_gate_best"])
    gates = sorted(r["eta_gate_best"] for r in runs) or [best["eta_gate_best"]]
    out = {"runs": [_ticket_row(r) for r in runs], "n": len(runs), "n_base": len(base_tickets),
           "best_seed": best["seed"], "best_key": best.get("key"), "best_from_original": best not in runs,
           "best_gate": best["eta_gate_best"], "best_rung1": best.get("eta_rung1"),
           "median_gate": gates[len(gates) // 2], "min_gate": gates[0],
           "wall_s": round(time.time() - t0, 2), "budget_s": round(budget_s, 2)}
    return out, best, tensors


def guarded_mixed_protocol(job, ctx, p, gm, seeds, warnings, from_run=None):
    """Tickets -> selection by gate -> optional repair.  Returns everything the
    common tail of run_optimize needs (deliverable tensors, records)."""
    tickets, tensors = [], {}
    if from_run is None:
        for i, seed in enumerate(seeds):
            key = "ours:%d" % seed
            rec, ten = run_ticket(job, ctx, gm, seed, key, i, len(seeds))
            tickets.append(rec)
            tensors[key] = ten
        win = max(tickets, key=lambda t: t["eta_gate_best"])
        win_t = tensors[win["key"]]
        origin_msg = M("origin_ticket", seed=win["seed"], step=win["best_step"])
    else:
        saved = from_run
        res = saved.get("result") or {}
        for t in res.get("tickets") or []:
            t = dict(t, role="saved")
            tickets.append(t)
        best = res.get("best") or {}
        latent = _load_saved_latent(saved["run_id"], ctx.cfg)
        binary = app_hard_binary(latent, ctx.cfg, ctx.mode)
        win = {"seed": best.get("seed"), "key": "saved:%s" % saved["run_id"], "index": None, "role": "saved",
               "best_step": best.get("best_step"), "beta_star": best.get("beta_star"),
               "eta_gate_best": best.get("eta_gate"), "eta_rung1": best.get("eta_rung1"),
               "gate_optimism": best.get("gate_optimism")}
        win_t = {"latent": latent, "binary": binary, "final_binary": binary}
        origin_msg = M("origin_saved", run=saved["run_id"])
        if gm.repair["beta_mode"] == "star" and best.get("beta_star") is None:
            warnings.append(M("warn_beta_star", b=gm.repair["beta_fixed"]))
    selection = {"rule": "best gate reading", "n_tickets": len(tickets) if from_run is None else None,
                 "winner_seed": win["seed"], "winner_key": win["key"], "eta_gate": win["eta_gate_best"],
                 "eta_rung1": win.get("eta_rung1"), "gate_optimism": win.get("gate_optimism"),
                 "beta_star": win.get("beta_star"), "best_step": win.get("best_step"),
                 "from_run": None if from_run is None else from_run["run_id"]}
    # ---- repair decision
    R = gm.repair
    r1, opt = win.get("eta_rung1"), win.get("gate_optimism")
    decision = {"enabled": bool(R["enabled"]) or from_run is not None, "trigger": "always" if from_run is not None else R["trigger"],
                "eta_rung1": r1, "bar": R["bar"], "gate_optimism": opt, "optimism_max": R["optimism_max"],
                "beta_mode": R["beta_mode"], "triggered": False, "reason": "disabled"}
    if from_run is not None:
        decision.update(triggered=True, reason="from_run")
    elif not R["enabled"]:
        decision["reason"] = "disabled"
    elif R["trigger"] == "always":
        decision.update(triggered=True, reason="always")
    elif R["trigger"] == "numeric":
        if R["bar"] is None:
            decision["reason"] = "numeric_no_bar"
        elif r1 is not None and r1 >= R["bar"]:
            decision["reason"] = "numeric_rung1_not_below_bar"
        elif opt is not None and opt >= R["optimism_max"]:
            decision["reason"] = "numeric_optimism_too_large"
        else:
            decision.update(triggered=True, reason="numeric_ok")
    else:
        decision["reason"] = "manual"
    repair = None
    deliver_t = win_t
    deliver = {"latent": win_t["latent"], "binary": win_t["binary"], "eta_gate": win["eta_gate_best"],
               "gate_detail": {"nominal": win["eta_gate_best"]}, "origin": "ticket" if from_run is None else "saved",
               "seed": win["seed"], "engine": "ours", "eta_rung1": win.get("eta_rung1"),
               "gate_optimism": win.get("gate_optimism"), "beta_star": win.get("beta_star"),
               "best_step": win.get("best_step"), "origin_msg": origin_msg}
    if decision["triggered"]:
        repair, inc = run_repair(job, ctx, gm, win, win_t)
        # the parent was re-read: report the re-read numbers as the parent's
        decision.update(eta_rung1=repair["parent_eta_rung1"], gate_optimism=repair["parent_eta_gate"] - repair["parent_eta_rung1"])
        if repair["n_accept"] > 0:
            deliver_t = inc
            deliver.update({"latent": inc["latent"], "binary": inc["binary"], "eta_gate": repair["final_eta_gate"],
                            "gate_detail": {"nominal": repair["final_eta_gate"]}, "origin": "repair",
                            "eta_rung1": repair["final_eta_rung1"],
                            "gate_optimism": repair["final_eta_gate"] - repair["final_eta_rung1"],
                            "origin_msg": M("origin_repair", seed=win["seed"], acc=repair["n_accept"])})
        else:
            deliver.update({"eta_gate": repair["parent_eta_gate"], "gate_detail": {"nominal": repair["parent_eta_gate"]},
                            "eta_rung1": repair["parent_eta_rung1"],
                            "gate_optimism": repair["parent_eta_gate"] - repair["parent_eta_rung1"]})
    return {"tickets": tickets, "tensors": tensors, "winner": win, "selection": selection,
            "decision": decision, "repair": repair, "deliver": deliver}


def _merge_saved_params(p, saved):
    """Manual chain from a saved run: the problem comes from the saved run, the
    chain / comparison settings from the current panel."""
    sp = dict(saved.get("params") or {})
    keep = ("protocol", "chain_start", "repair_enabled", "repair_trigger", "repair_bar", "repair_optimism_max",
            "repair_beta_mode", "repair_beta_fixed", "repair_beta_min", "repair_hops", "repair_stop_after",
            "chain_alphas", "chain_redescent_steps", "repair_re_k", "chain_noise_seed", "compare_restart_budget",
            "compare_torcwa", "compare_lumerical", "lumerical_kvecs", "field_zmax_nm", "cd_bias_nm",
            "memory_efficient", "save_on_cpu")
    for k in keep:
        if k in p:
            sp[k] = p[k]
    return sp


def run_optimize(job, *, include_fields=True, capture_arrays=False):
    p = job["params"]
    protocol = p.get("protocol", "guarded_mixed")
    if protocol == "chain":
        protocol = "legacy_chain"
    if protocol == "restart":
        raise ValueError("the 'restart' protocol was removed: several seeds under guarded_mixed are the restart lottery")
    from_run = None
    start = p.get("chain_start") or {}
    if protocol == "guarded_mixed" and start.get("kind") == "run_best":
        from_run = _load_saved_run(start["run_id"])
        if (from_run.get("result") or {}).get("protocol") != "guarded_mixed":
            raise ValueError("a chain can only be started from a guarded_mixed run")
        p = _merge_saved_params(p, from_run)
        job["params"] = p
    cfg = build_cfg(p)
    dtype, device, me, offload = engine_of(p)
    mode = p.get("symmetry", "y")
    theta, phi = float(p.get("theta_deg", 0.0)), float(p.get("phi_deg", 0.0))
    target = tuple(int(v) for v in p.get("target", [1, 0]))
    rule = order_rule(target, cfg)
    level_d = tuple(int(v) for v in (p.get("level_descent") or rule["descent"]))
    gate = tuple(int(v) for v in (p.get("level_gate") or rule["gate"]))
    levels_c = [tuple(int(v) for v in lv) for lv in (p.get("levels_certify") or rule["ladder"])]
    for lv in [level_d, gate] + levels_c:
        if abs(target[0]) > lv[0] or abs(target[1]) > lv[1]:
            raise ValueError("target order %s exceeds truncation order %s" % (target, lv))
        if not cfg.supports(lv):
            raise ValueError("grid (ny=%d, nx=%d) violates the convolution constraint of order %s; "
                             "minimum (ny, nx) = %s" % (cfg.ny, cfg.nx, lv, cfg.min_grid_for(lv)))
    warnings = []
    if C.n_terms(level_d) < C.n_terms(rule["descent"]):
        warnings.append(M("warn_descent_low", lv=level_d, rule=rule["descent"], m=rule["m_eff"]))
    if C.n_terms(gate) < 2 * C.n_terms(level_d) * 0.9:
        warnings.append(M("warn_gate_low", gate=gate, lv=level_d))
    rule_top = rule["ladder"][-1] if rule["ladder"] else rule["gate"]
    if levels_c and C.n_terms(levels_c[-1]) < C.n_terms(rule_top):
        warnings.append(M("warn_ladder_low", top=levels_c[-1], rule=rule_top))
    if rule["clipped"]:
        warnings.append(M("warn_clipped", nx=rule["grid_needed"]["nx"], ny=rule["grid_needed"]["ny"]))
    ladder = sorted({level_d, gate, *levels_c}, key=C.n_terms)
    cd_biases = [float(b) for b in p.get("cd_bias_nm", [])]
    ctx = SimpleNamespace(cfg=cfg, mode=mode, theta=theta, phi=phi, target=target,
                          level_d=level_d, gate=gate, dtype=dtype, device=device,
                          offload=offload, accept_rule=p.get("accept_rule", "nominal"),
                          cd_biases=cd_biases or [-20.0, 20.0],
                          ops=make_ops(), check_cancel=lambda: _raise_if_cancelled(job))
    steps = int(p.get("steps", 60))
    ckpt_every = int(p.get("ckpt_every") or max(5, steps // 10))
    seeds = [int(s) for s in p.get("seeds", [2029, 2030, 2031])] or [2029]
    job["resolved"] = {"problem": asdict(cfg), "protocol": protocol,
                       "symmetry": mode, "theta_deg": theta, "phi_deg": phi,
                       "target": target, "seeds": seeds,
                       "level_descent": level_d, "level_gate": gate,
                       "levels_certify": levels_c, "evaluation_ladder": ladder,
                       "cd_bias_nm": cd_biases, "accept_rule": ctx.accept_rule}
    _v2d.MEMORY_EFFICIENT = me
    job["checkpoints"] = {}
    t_total = time.time()
    try:
        chain = None          # legacy chain record
        gmr = None            # guarded-mixed pipeline record
        gm = None
        budget = None
        runs = []
        torcwa_opt_reason = ""
        if protocol == "legacy_chain":
            if p.get("compare_torcwa_opt"):
                torcwa_opt_reason = M("torcwa_opt_chain")
            chain = chain_protocol(job, ctx, p)
            deliver = {"latent": chain["incumbent"]["latent"],
                       "binary": chain["incumbent"]["binary"],
                       "eta_gate": chain["incumbent"]["score"],
                       "gate_detail": chain["incumbent"]["detail"],
                       "origin": "chain", "seed": None, "engine": "ours"}
            runs = [{"seed": r["seed"], "engine": "ours", "role": "start",
                     "trajectory": r["trajectory"], "checkpoints": r["checkpoints"],
                     "eta_final_relaxed": r["eta_final_relaxed"],
                     "eta_gate_best": r["eta_gate_best"], "best_step": r["best_step"],
                     "wall_s": r["wall_s"]} for r in chain["start_runs"]]
        else:
            gm = gm_settings(p, ctx, levels_c, warnings)
            job["resolved"]["guarded_mixed"] = gm_summary(gm)
            gmr = guarded_mixed_protocol(job, ctx, p, gm, seeds, warnings, from_run=from_run)
            deliver = gmr["deliver"]
            runs = [_ticket_row(t) for t in gmr["tickets"]]

        # ---- certification ladder of the delivered binary
        certification, result_high = certify_ladder(deliver["binary"], ladder, ctx, job, "deliver")
        # ---- robustness / fields / orders / torcwa / lumerical on the delivered binary
        robustness = []
        for bias in cd_biases:
            job["progress"] = {"phase": "robustness", "bias_nm": bias}
            biased = torch.from_numpy(C.cd_bias_mask(deliver["binary"].numpy() >= 0.5, bias, cfg))
            with torch.no_grad():
                r = solve_density(biased, ladder[-1], cfg, theta, phi, dtype, device)
            robustness.append({"bias_nm": bias, "eta": round(eta_of(r, target), 6)})
        job["progress"] = {"phase": "fields"}
        orders_result = orders_block(result_high, cfg, ladder[-1], theta, phi, target)
        fields = fields_block(result_high, cfg, ladder[-1], theta, phi,
                              z_max_nm=p.get("field_zmax_nm"), z_in_nm=p.get("field_zin_nm"),
                              binary=deliver["binary"], target=target, views=p.get("field_views")) if include_fields else {}
        if capture_arrays:
            job["arrays"] = _capture_arrays(deliver["latent"], deliver["binary"], result_high)
        torcwa = {"available": False, "error": M("not_selected")}
        if p.get("compare_torcwa", True):
            job["progress"] = {"phase": "torcwa"}
            torcwa = torcwa_block(deliver["binary"], ladder, cfg, theta, phi, target, dtype, device)
        lumerical = {"available": False, "error": M("not_selected")}
        if p.get("compare_lumerical", False):
            ok, reason = lumerical_supported(cfg, theta, p.get("polarization", "x"))
            if not ok:
                lumerical = {"available": False, "error": reason}
            else:
                job["progress"] = {"phase": "lumerical"}
                lumerical = lumerical_block(deliver["binary"], cfg, theta, phi, target,
                                            p.get("lumerical_kvecs", [201, 401]))
        patterns = pattern_block(deliver["latent"], cfg, mode)
        patterns_by_engine = {"ours": dict(patterns, seed=deliver["seed"],
                                           eta_top=certification[-1]["eta"])}
        chain_out = None
        if chain is not None:
            chain_out = {k: chain[k] for k in ("origin", "start_score", "jumps", "accepts",
                                                "n_jumps", "alphas", "redescent_steps", "beta",
                                                "start_wall_s", "chain_wall_s", "total_wall_s")}
            chain_out["incumbent_score"] = chain["incumbent"]["score"]
            chain_out["incumbent_detail"] = chain["incumbent"]["detail"]
        top_eta = certification[-1]["eta"]
        best = {"seed": deliver["seed"], "engine": deliver["engine"], "origin": deliver["origin"],
                "eta_gate": deliver["eta_gate"], "gate_detail": deliver["gate_detail"],
                "eta_top": top_eta, "drift_gate_to_top": round(top_eta - deliver["eta_gate"], 6)}
        if gmr is not None:
            rung1_row = next((c for c in certification if tuple(c["level"]) == tuple(gm.rung1)), None)
            e1 = rung1_row["eta"] if rung1_row is not None else deliver.get("eta_rung1")
            best.update({"eta_rung1": e1, "level_rung1": list(gm.rung1),
                         "gate_optimism": round(deliver["eta_gate"] - e1, 6) if e1 is not None else None,
                         "rung1_minus_top": round(e1 - top_eta, 6) if e1 is not None else None,
                         "beta_star": deliver.get("beta_star"), "best_step": deliver.get("best_step"),
                         "origin_msg": deliver.get("origin_msg")})
        result = {
            "kind": "optimize",
            "protocol": protocol,
            "order_rule": {k: v for k, v in rule.items()},
            "warnings": warnings,
            "runs": runs,
            "opt_compare": [],
            "opt_engines": ["ours"],
            "torcwa_opt_reason": torcwa_opt_reason,
            "level_descent": list(level_d), "level_gate": list(gate),
            "ladder": [list(lv) for lv in ladder],
            "chain": chain_out,
            "budget_compare": None,
            "restart_arm": None,
            "best": best,
            "patterns_by_engine": patterns_by_engine,
            "certification": certification,
            "robustness": robustness,
            "orders": orders_result,
            "fields": fields,
            "torcwa": torcwa,
            "lumerical": lumerical,
            "patterns": patterns,
            "accept_rule": ctx.accept_rule,
            "timing": {"total_s": round(time.time() - t_total, 2)},
            "engine": {"dtype": str(dtype), "device": device,
                       "memory_efficient": me, "save_on_cpu": offload,
                       "threads": THREADS},
        }
        if gmr is not None:
            result.update({
                "gm": gm_summary(gm),
                "tickets": gmr["tickets"],
                "selection": gmr["selection"],
                "repair_decision": gmr["decision"],
                "repair": gmr["repair"],
            })
        # The deliverable is complete here.  Stash it so a cancel during the
        # (optional, long) equal-budget arm still returns and persists it, and
        # keep the incumbent latent on disk (a cancel used to lose it).
        job["partial"] = dict(result, partial=True)
        job["partial_latent"] = deliver["latent"].numpy()
        if chain is not None and p.get("compare_restart_budget", True):
            budget = restart_arm(job, ctx, p, chain["total_wall_s"],
                                 seed_base=max(seeds) + 1000)
            rows_rs, _ = certify_ladder(budget["best"]["binary"], ladder, ctx, job, "restart_arm")
            result["budget_compare"] = {
                "chain": {"eta_gate": deliver["eta_gate"], "eta_top": certification[-1]["eta"],
                          "wall_s": chain["total_wall_s"], "n_jumps": chain["n_jumps"],
                          "accepts": chain["accepts"], "n_start": len(chain["start_runs"])},
                "restart": {"eta_gate": budget["best_gate"], "eta_top": rows_rs[-1]["eta"],
                            "wall_s": budget["wall_s"], "n": budget["n"],
                            "median_gate": budget["median_gate"], "min_gate": budget["min_gate"],
                            "best_seed": budget["best_seed"], "convergence": rows_rs},
                "ladder_top": list(ladder[-1]),
            }
            budget.pop("best", None)   # tensors are not serializable
            result["restart_arm"] = budget
            result["timing"]["total_s"] = round(time.time() - t_total, 2)
        elif gmr is not None and gmr["repair"] is not None and p.get("compare_restart_budget", False):
            rep = gmr["repair"]
            base = [t for t in gmr["tickets"] if t.get("eta_gate_best") is not None]
            budget, rs_best, rs_tensors = control_arm_gm(job, ctx, gm, rep["wall_s"],
                                                         seed_base=max(seeds) + 1000, base_tickets=base)
            rs_bin = (rs_tensors.get(rs_best.get("key")) or gmr["tensors"].get(rs_best.get("key")) or {}).get("binary")
            rows_rs = None
            if rs_bin is not None:
                rows_rs, _ = certify_ladder(rs_bin, ladder, ctx, job, "restart_arm")
            result["budget_compare"] = {
                "chain": {"eta_gate": deliver["eta_gate"], "eta_rung1": best.get("eta_rung1"),
                          "eta_top": certification[-1]["eta"], "wall_s": round(rep["wall_s"], 2),
                          "n_jumps": rep["hops_run"], "accepts": rep["n_accept"], "n_start": len(base)},
                "restart": {"eta_gate": budget["best_gate"], "eta_rung1": budget["best_rung1"],
                            "eta_top": rows_rs[-1]["eta"] if rows_rs else None,
                            "wall_s": budget["wall_s"], "n": budget["n"], "n_base": budget["n_base"],
                            "best_from_original": budget["best_from_original"],
                            "median_gate": budget["median_gate"], "min_gate": budget["min_gate"],
                            "best_seed": budget["best_seed"], "convergence": rows_rs},
                "ladder_top": list(ladder[-1]),
            }
            result["restart_arm"] = budget
            result["timing"]["total_s"] = round(time.time() - t_total, 2)
        job.pop("partial", None)
        return result
    finally:
        _v2d.MEMORY_EFFICIENT = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_solve(job, *, include_fields=True, capture_arrays=False):
    p = job["params"]
    cfg = build_cfg(p)
    dtype, device, me, _ = engine_of(p)
    mode = p.get("symmetry", "y")
    theta, phi = float(p.get("theta_deg", 0.0)), float(p.get("phi_deg", 0.0))
    target = tuple(int(v) for v in p.get("target", [1, 0]))
    level = tuple(int(v) for v in p.get("level_descent", [12, 3]))
    if not cfg.supports(level):
        raise ValueError("grid (ny=%d, nx=%d) violates the convolution constraint of order %s; "
                         "minimum (ny, nx) = %s" % (cfg.ny, cfg.nx, level, cfg.min_grid_for(level)))
    source = p.get("pattern_source", {"kind": "seed", "seed": 2029})
    if source.get("kind") == "uniform":
        density = torch.full((cfg.ny, cfg.nx), float(source.get("fill", 1.0)),
                             dtype=cfg.latent_dtype)
        latent = density
    elif source.get("kind") == "run_best":
        with open(os.path.join(RUNS_DIR, _safe_run_name(source["run_id"])),
                  encoding="utf-8") as f:
            saved = json.load(f)
        density = torch.tensor(saved["result"]["patterns"]["binary"],
                               dtype=cfg.latent_dtype)
        if density.shape != (cfg.ny, cfg.nx):
            raise ValueError("saved pattern grid %s does not match the current grid (ny=%d, nx=%d)"
                             % (tuple(density.shape), cfg.ny, cfg.nx))
        latent = density
    else:
        latent = app_initial_latent(int(source.get("seed", 2029)), cfg, mode, "cpu")
        density = app_hard_binary(latent, cfg, mode)
    # convergence ladder: evaluate the SAME fixed structure at a ladder of
    # truncation orders so the user sees whether the reported value is
    # converged (point-2).  Descent level first, then any certify levels.
    ladder_levels = [level]
    for lv in p.get("levels_certify", []):
        lvt = tuple(int(v) for v in lv)
        if lvt not in ladder_levels and cfg.supports(lvt) \
                and abs(target[0]) <= lvt[0] and abs(target[1]) <= lvt[1]:
            ladder_levels.append(lvt)
    ladder_levels.sort(key=lambda lv: C.n_terms(lv))
    job["resolved"] = {"problem": asdict(cfg), "symmetry": mode,
                       "theta_deg": theta, "phi_deg": phi, "target": target,
                       "level_descent": level, "evaluation_ladder": ladder_levels,
                       "pattern_source": source}
    _v2d.MEMORY_EFFICIENT = me
    t_solve_start = time.time()
    try:
        job["progress"] = {"phase": "solve", "level": list(level)}
        convergence = []
        r_at = {}
        for lv in ladder_levels:
            t0 = time.time()
            with torch.no_grad(), _export_modes():
                rr = solve_density(density, lv, cfg, theta, phi, dtype, device)
            r_at[lv] = rr
            convergence.append({"level": list(lv), "two_n": 2 * C.n_terms(lv),
                                "eta": round(eta_of(rr, target), 6),
                                "total_T": round(float(rr.total_T), 6),
                                "s": round(time.time() - t0, 2)})
        r = r_at[level]
        ours_s = convergence[[tuple(c["level"]) for c in convergence].index(level)]["s"]
        orders_result = orders_block(r, cfg, level, theta, phi, target)
        job["progress"] = {"phase": "fields"}
        fields = fields_block(r, cfg, level, theta, phi,
                              z_max_nm=p.get("field_zmax_nm"), z_in_nm=p.get("field_zin_nm"),
                              binary=density, target=target, views=p.get("field_views")) if include_fields else {}
        if capture_arrays:
            job["arrays"] = _capture_arrays(latent, density, r)
        torcwa = {"available": False, "error": M("not_selected")}
        if p.get("compare_torcwa", True):
            job["progress"] = {"phase": "torcwa"}
            torcwa = torcwa_block(density, ladder_levels, cfg, theta, phi, target,
                                  dtype, device)
        lumerical = {"available": False, "error": M("not_selected")}
        if p.get("compare_lumerical", False):
            ok, reason = lumerical_supported(cfg, theta, p.get("polarization", "x"))
            if not ok:
                lumerical = {"available": False, "error": reason}
            else:
                job["progress"] = {"phase": "lumerical"}
                lumerical = lumerical_block(density, cfg, theta, phi, target,
                                            p.get("lumerical_kvecs", [201, 401]))
        return {
            "kind": "solve",
            "eta_target": round(eta_of(r, target), 6),
            "level_descent": list(level),
            "convergence": convergence,
            "orders": orders_result,
            "fields": fields,
            "torcwa": torcwa,
            "lumerical": lumerical,
            "patterns": {"latent": _round_list(latent.detach().cpu().numpy(), 4),
                         "filtered": _round_list(latent.detach().cpu().numpy(), 4)
                         if source.get("kind") != "seed" else
                         _round_list(app_filtered(latent, cfg, mode).numpy(), 4),
                         "binary": density.cpu().numpy().round().astype(int).tolist()},
            "timing": {"ours_solve_s": round(ours_s, 2),
                       "total_s": round(time.time() - t_solve_start, 2)},
            "engine": {"dtype": str(dtype), "device": device,
                       "memory_efficient": me, "threads": THREADS},
        }
    finally:
        _v2d.MEMORY_EFFICIENT = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _capture_arrays(latent, density, result):
    """Unrounded arrays for batch exports; display JSON keeps its legacy format."""
    return {"latent": latent.detach().cpu().numpy(),
            "density": density.detach().cpu().numpy(),
            "orders": np.asarray(result.orders, dtype=np.int64),
            "power_t": result.power_t.detach().cpu().numpy(),
            "power_r": result.power_r.detach().cpu().numpy()}


# ------------------------------------------------------------------ job queue
JOBS = {}
JOBS_LOCK = threading.Lock()
JOB_QUEUE = queue.Queue()
_JOB_COUNTER = [0]


def submit_job(kind, params):
    with JOBS_LOCK:
        _JOB_COUNTER[0] += 1
        job_id = "j%04d" % _JOB_COUNTER[0]
        job = {"id": job_id, "type": kind, "params": params, "status": "queued",
               "created": time.time(), "progress": {}, "trajectories": {},
               "checkpoints": {}, "chain_jumps": [],
               "result": None, "error": None, "cancel": False}
        JOBS[job_id] = job
    JOB_QUEUE.put(job_id)
    return job_id


def worker_loop():
    while True:
        job_id = JOB_QUEUE.get()
        with JOBS_LOCK:
            job = JOBS.get(job_id)
        if job is None or job.get("cancel"):
            if job is not None:
                job["status"] = "cancelled"
            continue
        job["status"] = "running"
        job["started"] = time.time()
        try:
            if job["type"] == "optimize":
                job["result"] = run_optimize(job)
            elif job["type"] == "solve":
                job["result"] = run_solve(job)
            else:
                raise ValueError("unknown job type %r" % job["type"])
            job["status"] = "done"
            job["finished"] = time.time()
            _persist_run(job)
        except KeyboardInterrupt:
            job["status"] = "cancelled"
            if job.get("partial"):
                # the chain deliverable was complete; keep it (comparison arm skipped)
                job["result"] = job["partial"]
                job["finished"] = time.time()
                _persist_run(job)
        except Exception:
            job["status"] = "error"
            job["error"] = traceback.format_exc()
        finally:
            job.setdefault("finished", time.time())


def _persist_run(job):
    os.makedirs(RUNS_DIR, exist_ok=True)
    run_id = time.strftime("run_%Y%m%d_%H%M%S") + "_" + job["id"]
    payload = {"run_id": run_id, "type": job["type"], "params": job["params"],
               "result": job["result"],
               "trajectories": job["trajectories"],
               "checkpoints": job.get("checkpoints", {}),
               "chain_jumps": job.get("chain_jumps", []),
               "created": job["created"], "finished": job["finished"]}
    with open(os.path.join(RUNS_DIR, run_id + ".json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f)
    if job.get("partial_latent") is not None:
        np.save(os.path.join(RUNS_DIR, run_id + "_latent.npy"), job["partial_latent"])
    job["run_id"] = run_id


def _safe_run_name(run_id):
    name = os.path.basename(str(run_id))
    if not name.endswith(".json"):
        name += ".json"
    return name


def job_view(job, full=True):
    view = {k: job.get(k) for k in ("id", "type", "status", "progress",
                                    "created", "started", "finished",
                                    "run_id", "error")}
    view["trajectories"] = job.get("trajectories", {})
    view["checkpoints"] = job.get("checkpoints", {})
    view["chain_jumps"] = job.get("chain_jumps", [])
    if full:
        view["result"] = job.get("result")
        view["params"] = job.get("params")
    else:
        result = job.get("result")
        if result is not None:
            view["summary"] = {"kind": result.get("kind"),
                               "best": result.get("best"),
                               "eta_target": result.get("eta_target")}
    return view


# --------------------------------------------------------------- HTTP server
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[http] %s\n" % (fmt % args))

    def _send(self, code, body, content_type="application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            with open(os.path.join(_HERE, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if path.startswith("/vendor/"):
            name = os.path.basename(path)
            full = os.path.join(_HERE, "vendor", name)
            if os.path.isfile(full) and name.endswith(".js"):
                with open(full, "rb") as f:
                    return self._send(200, f.read(),
                                      "application/javascript; charset=utf-8")
            return self._send(404, {"error": "not found"})
        if path == "/api/env":
            return self._send(200, environment_info())
        if path == "/api/jobs":
            with JOBS_LOCK:
                items = [job_view(j, full=False)
                         for j in sorted(JOBS.values(), key=lambda x: x["created"],
                                         reverse=True)]
            return self._send(200, {"jobs": items})
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                return self._send(404, {"error": "no such job"})
            return self._send(200, job_view(job))
        if path == "/api/runs":
            rows = []
            for name in sorted(os.listdir(RUNS_DIR) if os.path.isdir(RUNS_DIR) else [], reverse=True):
                if not name.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(RUNS_DIR, name), encoding="utf-8") as f:
                        head = json.load(f)
                    result = head.get("result") or {}
                    rows.append({
                        "run_id": head.get("run_id", name[:-5]),
                        "type": head.get("type"),
                        "finished": head.get("finished"),
                        "best": result.get("best"),
                        "eta_target": result.get("eta_target"),
                        "certification": result.get("certification"),
                        "protocol": result.get("protocol"),
                        "target": (head.get("params") or {}).get("target"),
                    })
                except Exception:
                    continue
            return self._send(200, {"runs": rows})
        if path.startswith("/api/runs/"):
            name = _safe_run_name(path.rsplit("/", 1)[-1])
            full = os.path.join(RUNS_DIR, name)
            if not os.path.exists(full):
                return self._send(404, {"error": "no such run"})
            with open(full, "rb") as f:
                return self._send(200, f.read())
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length") or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return self._send(400, {"error": "invalid JSON"})
        if path == "/api/order_rule":
            try:
                cfg = build_cfg(payload)
                target = tuple(int(v) for v in payload.get("target", [1, 0]))
                rule = order_rule(target, cfg)
                return self._send(200, {k: (list(v) if isinstance(v, tuple) else v)
                                        for k, v in rule.items()}
                                  | {"ladder": [list(lv) for lv in rule["ladder"]]})
            except Exception as error:
                return self._send(400, {"error": str(error)})
        if path == "/api/jobs":
            kind = payload.get("type")
            if kind not in ("optimize", "solve"):
                return self._send(400, {"error": "type must be optimize|solve"})
            try:
                build_cfg(payload.get("params", {}))  # early validation
            except Exception as error:
                return self._send(400, {"error": str(error)})
            job_id = submit_job(kind, payload.get("params", {}))
            return self._send(200, {"job_id": job_id})
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job_id = path.split("/")[-2]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                return self._send(404, {"error": "no such job"})
            job["cancel"] = True
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "not found"})


def _has_torcwa():
    try:
        import torcwa  # noqa: F401
        return True
    except Exception:
        return False


def environment_info():
    """What this machine offers: interpreter, torch/CUDA, GPU, optional engines."""
    import platform
    cuda = torch.cuda.is_available()
    gpu_mem = None
    if cuda:
        gpu_mem = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 30, 1)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_mem_gb": gpu_mem,
        "torcwa": _has_torcwa(),
        "lumerical": lumerical_env(),
        "threads": THREADS,
        "offload_2n": offload_threshold_2n(),
        "eig_device": os.environ.get("TFMM_EIG_DEVICE", "cpu"),
    }


def main(argv=None):
    global RUNS_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8321)
    ap.add_argument("--runs-dir", help="run history directory (or set FMM_RUNS_DIR)")
    args = ap.parse_args(argv)
    if args.runs_dir:
        RUNS_DIR = os.path.abspath(os.path.expanduser(args.runs_dir))
    os.makedirs(RUNS_DIR, exist_ok=True)
    threading.Thread(target=worker_loop, daemon=True).start()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    env = environment_info()
    print("FMM workbench: http://127.0.0.1:%d  (python %s, torch %s, GPU: %s)"
          % (args.port, env["python"], env["torch"], env["gpu"] or "none, CPU mode"), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
