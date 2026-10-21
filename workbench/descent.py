"""Guarded mixed-order descent tickets and the repair chain (final scheme v2).

Ported from ``experiments/S10_e3_v2/scripts/s10_e3.py`` (``GradPolicy``,
``PolicyDescent``, ``run_chain``) with the evaluation order kept, so that a
ticket reproduces the S10 reference bitwise on the same machine and runtime.

What a ticket is (see docs/algorithm.md section 5):
  * Adam on the latent, cosine learning rate, geometric beta ramp
    beta(t) = b0 (b_end/b0)^(t/T) (README ramp 2 -> 2000 over 219 steps);
  * mixed-order gradients: from the first step with beta >= mix_from_beta,
    every k-th gradient is taken at the GATE order, the others at the descent
    order, all in the same Adam state; at every gate-order step a descent-order
    "probe" gradient of the same latent is computed as well (used for the
    update only when the gate-order gradient was refused, otherwise logged);
  * the norm guard: a gradient whose norm exceeds ``norm_factor`` x the median
    of the last ``window`` same-order norms is recomputed at beta (1 + 1 %);
    the original is kept when it is stable under that change (cosine >=
    stab_cos and norm ratio inside stab_ratio), otherwise the recompute
    replaces it (a complex64 artifact, S8/S10);
  * raised / non-finite gradients: the jitter ladder, then one CPU complex128
    gradient at the descent order (capped per ticket), at the gate order a
    descent step instead;
  * hard-binary checkpoints read at the gate order (and at the descent order,
    for the truncation-gap curve); the delivery is the best gate checkpoint.

This module holds the algorithm only.  The solver primitives are injected
through ``ctx.ops`` (solve_density, app_projected, app_hard_binary,
app_initial_latent, AppAdam, filtered_noise, sym) so that the public functions
of app.py remain the single definition of the physics and the design map.
Optional hook ``ops.jitter_density(x, beta, frac, cfg, mode)``: the density the
guard's recompute uses; absent, it is ``app_projected(x, beta (1 + frac))``.
"""
import math
import time
from contextlib import nullcontext
from typing import Dict, List, Optional

import numpy as np
import torch

from . import protocol as C


# ------------------------------------------------------------------ helpers
def cosine(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    return float(a @ b / (a.norm() * b.norm()).clamp_min(1e-300))


def eta_binary(density, level, ctx):
    """One forward solve (no gradient) of a density / hard binary at ``level``."""
    with torch.no_grad():
        r = ctx.ops.solve_density(density, tuple(level), ctx.cfg, ctx.theta, ctx.phi,
                                  ctx.dtype, ctx.device)
    return float(r.power_t[r.orders.index(tuple(ctx.target))])


def check_cancel(ctx):
    fn = getattr(ctx, "check_cancel", None)
    if fn is not None:
        fn()


def _level_key(level):
    return "%d,%d" % (int(level[0]), int(level[1]))


def _tau_table(guards):
    out = {}
    for k, v in (guards.get("tau") or {}).items():
        if isinstance(k, str):
            parts = k.replace("-", ",").split(",")
            k = (int(parts[0]), int(parts[1]))
        out[tuple(int(x) for x in k)] = float(v)
    return out


# ------------------------------------------------------------- the gradient policy
class GradPolicy:
    """Order choice, cadence, guards and logging shared by tickets and chain re-descents."""

    def __init__(self, ctx, level_d, level_g, guards, cadence=None, rise=None, k=4,
                 mix_from_beta=16.0, mix_from_step=None, steps=None, beta_fn=None,
                 high_budget=None):
        self.ctx = ctx
        self.cfg, self.mode, self.device = ctx.cfg, ctx.mode, ctx.device
        self.guards = dict(C.GUARDS, **(guards or {}))
        self.cadence = dict(C.CADENCE, **(cadence or {}))
        self.rise = dict(C.RISE, **(rise or {}))
        self.tau = _tau_table(self.guards)
        self.level_d0 = tuple(int(v) for v in level_d)
        self.level_d = self.level_d0
        self.level_g = tuple(int(v) for v in level_g)
        self.rise_ladder = (C.rise_ladder(self.level_d0, self.level_g, self.cfg)
                            if self.rise.get("enabled") else [])
        self.k = int(k)
        self.mix_from_beta = None if mix_from_beta is None else float(mix_from_beta)
        self.mix_from_step = mix_from_step
        self.steps = steps
        self.beta_fn = beta_fn                       # beta(t): only used to locate the mixing start
        self.high_budget = int(high_budget) if high_budget is not None else 10 ** 9
        self.hop = None
        self.high_used = 0
        self.next_high = None
        self.probe_hist: List[float] = []
        self.norms: Dict[tuple, List[float]] = {}
        self.prev_g: Dict[tuple, torch.Tensor] = {}
        self.prev_low_g: Optional[torch.Tensor] = None
        self.c128_used = 0
        self.rise_step = None
        self.k_history: List[Dict] = []
        self.rows: List[Dict] = []
        self.events: List[Dict] = []
        self.stats = {"n_eval": {}, "n_trigger": {"norm": 0, "raise": 0, "nonfinite": 0},
                      "n_true_positive": {"norm": 0, "raise": 0, "nonfinite": 0},
                      "n_would_trigger": {"norm": 0}, "n_small_gap": 0, "n_c128": 0,
                      "n_skip_high": 0, "n_kept_original": 0, "n_rise_complete": 0,
                      "wall_recompute_s": 0.0, "wall_probe_s": 0.0, "wall_c128_s": 0.0}
        self._t_start_mix = None

    # ---- one forward + backward, eigen gap read in between
    def _fwd_bwd(self, latent, level, beta, dtype=None, device=None, jitter=None):
        ops = self.ctx.ops
        dtype = dtype or self.ctx.dtype
        device = device or self.device
        offload = bool(getattr(self.ctx, "offload", False)) and device == self.device and dtype == self.ctx.dtype
        x = latent.detach().clone().to(device).requires_grad_(True)
        try:
            with (torch.autograd.graph.save_on_cpu(pin_memory=True) if offload else nullcontext()):
                if jitter is None:
                    dens = ops.app_projected(x, float(beta), self.cfg, self.mode)
                elif hasattr(ops, "jitter_density"):      # the guard's recompute: an arm-defined density jitter
                    dens = ops.jitter_density(x, float(beta), float(jitter), self.cfg, self.mode)
                else:                                      # default = the beta jitter (density arms, bitwise)
                    dens = ops.app_projected(x, float(beta) * (1.0 + float(jitter)), self.cfg, self.mode)
                r = ops.solve_density(dens, tuple(level), self.cfg, self.ctx.theta, self.ctx.phi, dtype, device)
                gap = float(torch.as_tensor(r.diagnostics.get("min_eigen_gap", float("inf"))).detach().real)
                j = r.power_t[r.orders.index(tuple(self.ctx.target))]
                j.backward()
            v = float(j.detach())
            g = x.grad.detach()
            del r, j, dens
        except RuntimeError as exc:
            return None, None, float("nan"), "raise: " + str(exc)[:120]
        finally:
            del x
        if not (math.isfinite(v) and bool(torch.isfinite(g).all())):
            return None, None, gap, "nonfinite"
        if device != self.device:
            g = g.to(self.device)
        return v, g.to(self.cfg.latent_dtype), gap, None

    def _norm_check(self, level, norm):
        G = self.guards
        w = self.norms.get(level, [])
        if len(w) < int(G["warmup"]):
            return False, None
        med = float(np.median(w[-int(G["window"]):]))
        return norm > float(G["norm_factor"]) * med, med

    def gradient(self, latent, level, beta, t, is_high, adam=None):
        """Guarded value + gradient at ``level``.  Returns (v, g, row); g is None only
        when a gate-order gradient is refused (the caller takes a descent step)."""
        G = self.guards
        active = bool(G.get("enabled", True))
        level = tuple(level)
        tau = self.tau.get(level, float(G.get("tau_default", 1e-4)))
        key = _level_key(level)
        self.stats["n_eval"][key] = self.stats["n_eval"].get(key, 0) + 1
        t0 = time.perf_counter()
        v, g, gap, err = self._fwd_bwd(latent, level, beta)
        row = {"t": t, "hop": self.hop, "beta": round(float(beta), 3), "level": list(level),
               "high": bool(is_high), "value": v,
               "norm": float(g.norm()) if g is not None else None,
               "gap": gap if gap == gap else None, "action": "ok", "n_recompute": 0}
        if gap == gap and gap < tau:
            self.stats["n_small_gap"] += 1
            row["small_gap"] = True
        pg = self.prev_g.get(level)
        cos_prev = cosine(pg, g) if (pg is not None and g is not None) else None
        row["cos_prev_same_order"] = cos_prev
        reason = None
        if err is not None:
            reason = "raise" if err.startswith("raise") else "nonfinite"
            row["error"] = err
        elif beta >= float(G["beta_min"]):
            trip, med = self._norm_check(level, row["norm"])
            if trip:
                reason = "norm"
                row["norm_median"] = med
        if reason == "norm" and not active:
            self.stats["n_would_trigger"]["norm"] += 1
            row["action"] = "log_only_norm"
            reason = None
        if reason is not None:
            self.stats["n_trigger"][reason] += 1
            g_orig, v_orig = g, v
            t1 = time.perf_counter()
            decided = False
            row["recomputes"] = []
            ladder = G["jitters"] if reason == "norm" else G["jitters_raise"]
            for jit in ladder:
                b2 = float(beta) * (1.0 + float(jit))            # logged as beta_used only
                v2, g2, gap2, err2 = self._fwd_bwd(latent, level, beta, jitter=jit)
                row["n_recompute"] += 1
                rr = {"beta_used": round(b2, 4), "err": err2}
                if err2 is not None:
                    row["recomputes"].append(rr)
                    continue
                if g_orig is None:                                   # raise / non-finite: first finite recompute wins
                    v, g, gap, decided = v2, g2, gap2, True
                    row.update({"action": "jitter", "beta_used": round(b2, 4)})
                    row["recomputes"].append(rr)
                    break
                cs = cosine(g_orig, g2)
                ratio = float(g_orig.norm() / g2.norm().clamp_min(1e-300))
                rr.update({"cos_orig": cs, "ratio_orig_over_recomputed": ratio})
                row["recomputes"].append(rr)
                stable = (cs >= float(G["stab_cos"])) and (float(G["stab_ratio"][0]) <= ratio <= float(G["stab_ratio"][1]))
                if stable:                                           # survives a 1 % beta change: natural growth, keep it
                    v, g = v_orig, g_orig
                    row["action"] = "kept_original"
                    self.stats["n_kept_original"] += 1
                else:                                                # unstable: a c64 artifact; the recompute replaces it
                    v, g, gap = v2, g2, gap2
                    row.update({"action": "jitter", "beta_used": round(b2, 4), "cos_orig_recomputed": cs,
                                "norm_ratio_orig_over_recomputed": ratio, "true_positive": True})
                    self.stats["n_true_positive"][reason] += 1
                decided = True
                break
            if not decided:
                if g_orig is not None:                               # every recompute raised: keep what we have
                    v, g = v_orig, g_orig
                    row["action"] = "kept_original_no_recompute"
                elif is_high:
                    self.stats["n_skip_high"] += 1
                    row["action"] = "skip_high"
                    v, g = None, None
                elif self.c128_used < int(G["c128_cap"]):
                    t2 = time.perf_counter()
                    v3, g3, gap3, err3 = self._fwd_bwd(latent, level, beta, dtype=torch.complex128, device="cpu")
                    self.stats["wall_c128_s"] += time.perf_counter() - t2
                    self.c128_used += 1
                    self.stats["n_c128"] += 1
                    if err3 is None:
                        v, g, gap = v3, g3, gap3
                        row["action"] = "c128"
                    else:
                        row["action"] = "skipped_update"
                        v, g = None, None
                else:
                    row["action"] = "skipped_update"
                    v, g = None, None
            self.stats["wall_recompute_s"] += time.perf_counter() - t1
            self.events.append({"t": t, "hop": self.hop, "beta": float(beta), "level": list(level),
                                "high": bool(is_high), "reason": reason, "gap": gap if gap == gap else None,
                                "action": row["action"], "cos_prev_orig": cos_prev,
                                "cos": row.get("cos_orig_recomputed"),
                                "ratio": row.get("norm_ratio_orig_over_recomputed"),
                                "recomputes": row["recomputes"]})
        if g is not None:
            row["norm"] = float(g.norm())
            self.prev_g[level] = g
            self.norms.setdefault(level, []).append(row["norm"])
            if adam is not None and adam.t > 0:
                row["cos_m_g"] = cosine(adam.m, g)
        row["wall_s"] = time.perf_counter() - t0
        self.rows.append(row)
        return v, g, row

    # ---- cadence
    def _first_step_at(self, beta_target):
        if self._t_start_mix is None:
            steps = int(self.steps or 0)
            self._t_start_mix = next((t for t in range(steps + 1) if self.beta_fn(t) >= beta_target), steps + 1)
        return self._t_start_mix

    def is_high_step(self, t, beta):
        if self.mix_from_beta is None:
            start_ok = t >= int(self.mix_from_step or 0)
        else:
            start_ok = beta >= self.mix_from_beta
        if not start_ok:
            return False
        if not self.cadence.get("enabled"):
            t0 = int(self.mix_from_step or 0) if self.mix_from_beta is None else self._first_step_at(self.mix_from_beta)
            return (t - t0) % self.k == 0
        if self.high_used >= self.high_budget or (self.rise.get("enabled") and self.level_d == self.level_g):
            return False
        if self.next_high is None:
            self.next_high = t
        return t >= self.next_high

    def after_high(self, t, cos_probe, cos_proxy, checked):
        """Update k on two agreeing probe checks (only checks where the gate-order
        gradient was used count), the rung rise, and the next high step.  The
        high-order budget is a CAP: is_high_step stops at the budget, k is never
        stretched to pace it."""
        self.high_used += 1
        if not self.cadence.get("enabled"):
            return
        k_set = sorted(int(x) for x in self.cadence["k_set"])
        k_min, k_max = k_set[0], k_set[-1]
        if checked and cos_probe is not None:
            self.probe_hist.append(cos_probe)
        h = self.probe_hist[-2:]
        changed = None
        if checked and cos_probe is not None and len(h) == 2:
            if min(h) > float(self.cadence["up"]) and self.k < k_max:
                self.k, changed = min(self.k * 2, k_max), "up"
            elif max(h) < float(self.cadence["down"]):
                if self.k > k_min:
                    self.k, changed = max(self.k // 2, k_min), "down"
                if self.rise.get("enabled") and self.rise_ladder and self.level_d != self.level_g:
                    nxt = self.rise_ladder.pop(0)
                    self.level_d, self.rise_step = nxt, t
                    self.k_history.append({"t": t, "rise_to": list(nxt)})
                    changed = (changed or "") + "+rise"
                    if self.level_d == self.level_g:
                        self.stats["n_rise_complete"] += 1
                        self.events.append({"t": t, "reason": "rise_complete", "level": list(nxt)})
            if changed:
                self.probe_hist = []
        self.next_high = t + self.k
        self.k_history.append({"t": t, "k": self.k, "cos_probe": cos_probe, "cos_proxy": cos_proxy,
                               "checked": bool(checked), "changed": changed, "high_used": self.high_used})


# ------------------------------------------------------------- the ticket
class TicketDescent:
    """One resumable descent with hard-binary gate checkpoints; the gradient,
    order and cadence come from a GradPolicy.  ``on_step(self)`` (optional) is
    called after every optimizer step for progress reporting."""

    def __init__(self, ctx, seed, steps, beta_fn, beta_final, ckpt_steps, policy, on_step=None):
        self.ctx, self.seed, self.steps = ctx, int(seed), int(steps)
        self.beta_fn = beta_fn                          # beta(step, steps)
        self.beta_final = float(beta_final)
        self.ck = set(int(s) for s in ckpt_steps)
        self.policy = policy
        self.on_step = on_step
        cfg, mode = ctx.cfg, ctx.mode
        self.latent = ctx.ops.app_initial_latent(seed, cfg, mode, ctx.device)
        self.adam = ctx.ops.AppAdam(self.latent.shape, cfg, mode, ctx.device)
        self.t = 0
        self.traj: List[Optional[float]] = []
        self.traj_descent: List[Optional[float]] = []
        self.level_series: List[int] = []
        self.ckpts: List[Dict] = []
        self.best = self.last = None
        self.t_desc = self.t_gate = 0.0
        self.n_high, self.t_high = 0, 0.0
        self.gate_read_s: List[float] = []
        self._checkpoint(0)

    def _checkpoint(self, step):
        ctx, P = self.ctx, self.policy
        cfg, mode = ctx.cfg, ctx.mode
        lat = self.latent.detach().cpu().clone()
        b = ctx.ops.app_hard_binary(lat, cfg, mode)
        t0 = time.perf_counter()
        eg = eta_binary(b, P.level_g, ctx)
        t1 = time.perf_counter()
        self.gate_read_s.append(t1 - t0)
        ed0 = eta_binary(b, P.level_d0, ctx)
        ed = ed0 if P.level_d == P.level_d0 else eta_binary(b, P.level_d, ctx)
        self.t_gate += time.perf_counter() - t0
        ck = {"step": step, "eta_gate": eg, "beta": self.beta_fn(step, self.steps),
              "eta_binary_descent": ed0, "gap_descent_minus_gate": ed0 - eg}
        if P.level_d != P.level_d0:
            ck["eta_binary_descent_current"] = ed
            ck["level_d_current"] = list(P.level_d)
        self.ckpts.append(ck)
        if self.best is None or eg > self.best["eta_gate"]:
            self.best = {"eta_gate": eg, "step": step, "latent": lat, "binary": b}
        self.last = {"eta_gate": eg, "step": step, "latent": lat, "binary": b}
        return eg

    def run_to(self, step):
        ctx, P = self.ctx, self.policy
        end = min(int(step), self.steps)
        while self.t < end:
            check_cancel(ctx)
            t = self.t
            beta = self.beta_fn(t, self.steps)
            lr = C.cosine_lr(t / max(self.steps - 1, 1), ctx.cfg)
            high = P.is_high_step(t, beta)
            level = P.level_g if high else P.level_d
            t0 = time.perf_counter()
            v, g, row = P.gradient(self.latent, level, beta, t, high, adam=self.adam)
            dt_high = time.perf_counter() - t0
            v_desc = v if not high else None
            if high:
                t1 = time.perf_counter()
                vl, gl, rowl = P.gradient(self.latent, P.level_d, beta, t, False, adam=None)   # the descent-order probe
                rowl["probe"] = True
                P.stats["wall_probe_s"] += time.perf_counter() - t1
                v_desc = vl
                cos_probe = cosine(gl, g) if (g is not None and gl is not None) else None
                cos_proxy = cosine(P.prev_low_g, g) if (g is not None and P.prev_low_g is not None) else None
                row.update({"cos_probe": cos_probe, "cos_proxy": cos_proxy, "k": P.k})
                checked = g is not None
                if g is None:
                    row["fallback"] = "descent_step"
                    v, g = vl, gl
                P.after_high(t, cos_probe, cos_proxy, checked)
            elif g is not None:
                P.prev_low_g = g
            self.t_desc += time.perf_counter() - t0
            if high:
                self.n_high += 1
                self.t_high += dt_high
            if g is not None:
                self.latent = self.adam.step(self.latent, g, lr)
                self.traj.append(round(float(v), 6) if v is not None else None)
            else:
                self.traj.append(None)
                P.events.append({"t": t, "beta": float(beta), "level": list(level), "reason": "skipped_update"})
            self.traj_descent.append(round(float(v_desc), 6) if v_desc is not None else None)
            self.level_series.append(1 if high else 0)
            self.t = t + 1
            if self.t in self.ck:
                self._checkpoint(self.t)
            if self.on_step is not None:
                self.on_step(self)
        return self

    def collapse_step(self):
        if not self.traj:
            return None
        arr = np.array([x if x is not None else np.nan for x in self.traj], float)
        below = np.where(arr < 0.05)[0]
        below = below[below > 20]
        if below.size and np.nanmax(arr[below[0]:]) < 0.05:
            return int(below[0])
        return None

    def finish(self):
        ctx = self.ctx
        self.run_to(self.steps)
        if self.ckpts[-1]["step"] != self.steps:
            self._checkpoint(self.steps)
        cfg, mode = ctx.cfg, ctx.mode
        dens = ctx.ops.app_projected(self.last["latent"], self.beta_final, cfg, mode)
        reported = eta_binary(dens, self.policy.level_d, ctx)
        return {"best": self.best, "final": self.last, "checkpoints": self.ckpts,
                "trajectory": self.traj, "trajectory_descent": self.traj_descent,
                "level_series": self.level_series, "reported_continuous": reported,
                "collapse_step": self.collapse_step(), "wall_descent_s": self.t_desc,
                "wall_gate_s": self.t_gate, "n_ckpt": len(self.ckpts), "beta_final": self.beta_final,
                "n_high_order_steps": self.n_high, "wall_high_order_s": self.t_high,
                "gate_read_s": float(np.median(self.gate_read_s)) if self.gate_read_s else None,
                "steps_run": self.t}


# ------------------------------------------------------------- the repair chain
def repair_chain(ctx, parent_latent, parent_binary, parent_gate, parent_rung1, beta_re, level_d, level_g,
                 rung1, repair, guards, noise_seed, on_hop=None, on_step=None):
    """Basin hopping on the delivered ticket: perturb the latent, re-descend 30
    steps at fixed beta with mixed-order gradients under the same guards, hard
    threshold, score at the gate order, accept only improvements.  A rung-1
    read follows every accept (false-accept tracking).  Returns the record and
    the incumbent tensors."""
    cfg, mode, device = ctx.cfg, ctx.mode, ctx.device
    R = dict(C.REPAIR, **(repair or {}))
    hops, stop_after = int(R["hops"]), int(R["stop_after"])
    re_steps, re_k = int(R["re_steps"]), int(R["re_k"])
    alphas = [float(a) for a in R["alphas"]]
    pol = GradPolicy(ctx, level_d, level_g, guards, cadence={"enabled": False}, rise={"enabled": False},
                     k=re_k, mix_from_beta=None, mix_from_step=0, steps=re_steps,
                     beta_fn=lambda t: float(beta_re))
    inc_lat = parent_latent.to(device) if device != "cpu" else parent_latent.clone()
    inc_bin, inc_gate, inc_r1 = parent_binary, float(parent_gate), float(parent_rung1)
    jumps, curve = [], [{"jump": 0, "eta_gate": inc_gate, "eta_rung1": inc_r1}]
    n_acc, rejects = 0, 0
    t0 = time.perf_counter()
    for j in range(hops):
        check_cancel(ctx)
        alpha = alphas[j % len(alphas)]
        noise = ctx.ops.filtered_noise(int(noise_seed) + j, cfg)
        if device != "cpu":
            noise = noise.to(device)
        rho = ctx.ops.sym((inc_lat + alpha * cfg.init_scale * noise).clamp(0.0, 1.0), mode)
        adam = ctx.ops.AppAdam(rho.shape, cfg, mode, device)
        pol.hop, pol.prev_g, pol.norms, pol.prev_low_g = j, {}, {}, None      # per-latent guard state restarts at every hop
        t_j = time.perf_counter()
        for t in range(re_steps):
            check_cancel(ctx)
            lr = C.cosine_lr(t / re_steps, cfg)
            high = (re_k > 0) and (t % re_k == 0)
            v, g, row = pol.gradient(rho, pol.level_g if high else pol.level_d, float(beta_re), t, high, adam=adam)
            if g is None and high:
                v, g, row = pol.gradient(rho, pol.level_d, float(beta_re), t, False, adam=adam)
            if g is not None:
                rho = adam.step(rho, g, lr)
            if on_step is not None:
                on_step(j, t)
        cb = ctx.ops.app_hard_binary(rho.detach().cpu(), cfg, mode)
        sc = eta_binary(cb, level_g, ctx)
        acc = bool(sc > inc_gate)
        rec = {"jump": j + 1, "alpha": alpha, "score": round(sc, 6), "cand_gate": sc,
               "cand_minus_inc": sc - inc_gate, "n_flipped": int((cb != inc_bin).sum()), "accepted": acc,
               "s": round(time.perf_counter() - t_j, 2)}
        if acc:
            r1 = eta_binary(cb, rung1, ctx)
            rec.update({"cand_rung1": r1, "false_accept": bool(r1 < inc_r1)})
            inc_lat, inc_bin, inc_gate, inc_r1, n_acc, rejects = rho, cb, sc, r1, n_acc + 1, 0
        else:
            rejects += 1
        rec["incumbent"] = round(inc_gate, 6)
        jumps.append(rec)
        curve.append({"jump": j + 1, "eta_gate": inc_gate, "eta_rung1": inc_r1})
        if on_hop is not None:
            on_hop(rec, inc_gate, n_acc, jumps)
        if rejects >= stop_after:
            break
    record = {"beta_re": float(beta_re), "hops": hops, "stop_after": stop_after, "hops_run": len(jumps),
              "n_accept": n_acc, "n_false_accept": int(sum(1 for x in jumps if x.get("false_accept"))),
              "stopped_early": bool(rejects >= stop_after and len(jumps) < hops),
              "parent_eta_gate": float(parent_gate), "parent_eta_rung1": float(parent_rung1),
              "final_eta_gate": inc_gate, "final_eta_rung1": inc_r1, "gain_rung1": inc_r1 - float(parent_rung1),
              "alphas": alphas, "re_steps": re_steps, "re_k": re_k, "noise_seed": int(noise_seed),
              "jumps": jumps, "incumbent_curve": curve, "guard_stats": pol.stats, "events": pol.events,
              "steps_log": pol.rows, "wall_s": time.perf_counter() - t0}
    return record, {"latent": inc_lat.detach().cpu(), "binary": inc_bin}
