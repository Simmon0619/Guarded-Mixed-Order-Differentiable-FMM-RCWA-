# Algorithm notes: differentiable RCWA, GPU acceleration and the guarded mixed-order design protocol

> 中文版 / Chinese version: [algorithm.zh-CN.md](algorithm.zh-CN.md)

These notes are for readers who want to understand what the workbench is actually computing underneath. No RCWA background is assumed, but you should know what a Fourier series and a matrix eigenvalue are.
Formulas are kept to a minimum and stated precisely; each section ends with the location of the corresponding code.

---

## 1. The problem: diffraction efficiency of a periodic structure

Consider a thin-film structure that is periodic in x and y (periods Λx and Λy, thickness h), whose permittivity ε(x, y) does not depend on z within the layer. A plane wave arrives from the semi-infinite medium on the incident side
(refractive index n_in) and exits into a medium of refractive index n_out. Periodicity splits the transmitted light into discrete **diffraction orders** (m, n), and each order carries away a share of the power, η_{m,n}.
Inverse design asks: which binary pattern ε(x, y) ("material / no material") sends as much power as possible into one specified order?

---

## 2. How RCWA, the Fourier modal method (FMM), solves the problem

### 2.1 Expansion
Expand the periodic function ε(x, y) and the fields as Fourier series, keeping only the **N = (2Mx+1)(2My+1)** terms with |m| ≤ Mx and |n| ≤ My.
This truncation is the only approximation in RCWA. The transverse wavevector of each order is

k_{x,m} = k_{x,0} + m·2π/Λx, k_{y,n} = k_{y,0} + n·2π/Λy.

The larger the period, the smaller the reciprocal-lattice spacing 2π/Λ, and the more orders it takes to cover the same span of physical k-space -- so **large-period problems need large truncation orders**.
A workable empirical lower bound is M ≈ n_max·Λ/λ (which captures every propagating direction inside the film); convergence usually needs a further factor of 2 on top of that.

### 2.2 Convolution matrix
The Fourier coefficients of ε(x, y) are arranged into an N × N Toeplitz-type **convolution matrix** ⟦ε⟧ with entries ε_{m−m', n−n'}. Building it needs the Fourier coefficients of ε up to 2Mx and 2My,
so the pixel grid must satisfy nx ≥ 4Mx+1 and ny ≥ 4My+1 (otherwise the coefficients alias). That is where the constraint shown in the workbench panel comes from.

### 2.3 One eigenproblem per layer
Inside a layer, Maxwell's equations reduce to a second-order ordinary differential equation for the Fourier coefficients of the transverse electric field:

d²E/dz² = −k₀² (P Q) E,

where P and Q are 2N × 2N matrices assembled from ⟦ε⟧, ⟦ε⟧⁻¹ and the diagonal wavevector matrices. The **eigendecomposition of PQ**, PQ = W Λ W⁻¹, gives the modes W and the propagation constants q = √Λ;
the magnetic-field modes are V = P⁻¹ W diag(q). This step (`torch.linalg.eig`) is the dominant cost of the whole solve.

### 2.4 S-matrix cascading
At each interface, continuity of the transverse E and H fields gives a 4N × 4N linear system whose solution is the scattering matrix of that interface (S11, S12, S21, S22);
propagation inside a layer is just a diagonal phase factor; the Redheffer star product cascades the pieces into the global S-matrix. Unlike the transfer-matrix approach, the S-matrix is immune to the overflow caused by the exponential growth of evanescent waves.

### 2.5 Output
Applying the global S21 to the incident vector gives the complex transmitted amplitude t_{m,n} of every order; the power η_{m,n} is the Poynting flux (E × H*) divided by the incident power,
and is non-zero only for propagating orders (real k_z). The energy-conservation residual |R + T − 1| is the most basic self-check.

**Code**: `trustworthy_fmm/fourier/convolution.py` (convolution matrix), `solvers/vector_2d.py` (P, Q, eigenproblem, cascading, power),
`scattering/smatrix.py` (interface S-matrix, star product), `solvers/eigendecomposition.py`.

---

## 3. What "differentiable" means, and how the GPU is used

### 3.1 Automatic differentiation
The entire forward solve is written in PyTorch tensor operations -- FFT, matrix multiplication, linear solves and eigendecomposition all have backpropagation rules.
So d η / d ε(x, y) (the derivative of the efficiency with respect to the permittivity of every pixel) comes out of a single backward pass, at roughly the cost of one forward solve.
Without this gradient, a design problem with tens of thousands of pixels could only be attacked by random search.

The backward pass through the eigendecomposition contains a 1/(λ_i − λ_j)-type amplification when two eigenvalues are nearly degenerate. In single precision this occasionally produces a gradient that is
**finite but wrong** (norm 10–500× too large, direction nearly orthogonal to the true gradient); double precision gives the correct gradient at the same point, the size of the eigenvalue gap does *not* predict
where it happens, and the Lorentzian broadening used by other solvers does not repair it (it only bends healthy gradients by 20–30° once ε ≥ 1e−6). The workbench therefore keeps the exact backward and catches
these events with a runtime guard (§5.2). The library still ships the optional Lorentzian stabilizer, off by default.

### 3.2 Design map (turning "pixels" into a "manufacturable pattern")
Optimizing pixels directly produces unmanufacturable pixel noise. The workbench uses the standard three-stage map:

latent ρ ∈ [0,1]^{ny×nx} → symmetrization (optional) → periodic Gaussian filter (radius r_f) → tanh projection (sharpness β) → continuous density ρ̃ ∈ [0,1]
→ ε = ε_bg + (ε_feat − ε_bg)·ρ̃.

The delivered pattern does not use ρ̃; it uses a **hard threshold**: after filtering, pixels ≥ 0.5 are material -- this is the binary, manufacturable pattern.
The filter radius sets the minimum feature size; β continuation from small to large lets the optimization move through a smooth landscape first and then approach a binary pattern gradually.

Key point: the η(ρ̃) that the optimizer sees is the **relaxed value** (gray pixels amount to unmanufacturable intermediate permittivities, which add degrees of freedom and bias the number systematically upward);
the efficiency of the delivered pattern must be recomputed from the hard-binary pattern. The workbench plots the two separately (thin line vs ◆) and never reports the relaxed value.

**Code**: `trustworthy_fmm/fabrication/process.py` (filter, projection), `app_filtered / app_projected / app_hard_binary` in `workbench/app.py`.

### 3.3 GPU, precision and where the eigendecomposition runs
- All tensors live on CUDA in `complex64` (single-precision complex). Single precision makes the matrix algebra several times faster and halves GPU memory;
  for an O(1) quantity such as efficiency, the difference from double precision is at the 1e-4 level. Use `CPU / complex128` for high-precision checks (and remember §3.1: the *gradient* can be wrong in single precision at isolated points; the guard handles that).
- **The eigendecomposition itself runs on the CPU by default** (`TFMM_EIG_DEVICE=cpu`): on the reference stack (torch 1.13 / CUDA 11.7 / Windows) the CUDA eigensolver leaks host memory on every call (about 0.4× the matrix size, ~12 GB over an 800-step descent),
  which ends a long run with an out-of-memory error. The matrix is moved to the host for `torch.linalg.eig` and the eigenpairs are moved back; autograd tracks both copies, so the gradient is unchanged. The price is about +30 % on the eigendecomposition wall; the number of BLAS threads (`FMM_THREADS`, default 3) then matters for the wall clock.
  Set `TFMM_EIG_DEVICE=native` to decompose on the GPU if your torch build does not leak.
- GPU memory: the matrices are 2N × 2N, 8 bytes per complex64 entry; one solve keeps a few dozen such matrices alive at once (modes, S blocks, temporaries of the 4N×4N interface system),
  and the backward pass has to save about as many intermediates again. **Peak memory ∝ (2N)²**. With mixed-order gradients (§5.2) the gate-order backward is the memory peak, not the descent order.
- Two memory-saving switches (neither changes the mathematics):
  - **"Memory-efficient path"**: P, Q and V are assembled by vector scaling rather than dense multiplication with diagonal matrices; the right-hand side of the interface system is sliced from the system matrix rather than rebuilt;
    the star product for in-layer propagation reduces to element-wise phase scaling; a left-hand-side matrix that is reused is LU-factorized only once. Both the forward peak and the amount saved for the backward pass drop significantly.
  - **"save_on_cpu"**: the tensors saved for the backward pass are moved to (pinned) host memory, so peak GPU memory falls back to the forward peak, at a cost of +10–20% in time. The automatic setting turns it on when the largest gradient order has 2N ≥ 3000.
- Time: the forward solve is dominated by the eigendecomposition (slower on matrices from real structures than on random matrices, because the QR iteration converges more slowly), followed by the 4N×4N interface solve;
  the backward pass is mostly matrix multiplication and is usually cheaper than the forward. One optimization step ≈ 1.2–1.5 × a forward solve.

**Code**: the `MEMORY_EFFICIENT` / `CHECKPOINT_SMATRIX` / `GAP_ONLY_DIAGNOSTICS` switches and their comments at the top of `solvers/vector_2d.py`; `EIG_DEVICE` in `solvers/eigendecomposition.py`;
`propagation_star / redheffer_star_fused / interface_smatrix_direct` in `scattering/smatrix.py`.

---

## 4. Three tiers of truncation order: descent, gate, certification

The same pattern gives a different η at different truncation orders, and more orders bring it closer to the true value (material boundaries are discontinuous, so the Fourier series converges slowly).
This sets a trap: an optimizer that makes its choices at a low order will pick out the structures that **happen to be overestimated at that low order** (selection bias, the optimizer's curse),
and those collapse once the delivered pattern is evaluated at a high order. The workbench uses three tiers of order to keep "computing gradients", "making selections" and "reporting numbers" apart:

| Tier | Purpose | Number of calls | Requirement |
|---|---|---|---|
| Descent order | Value + gradient at most steps | steps × tickets (hundreds to thousands) | Cheap; the gradient direction just has to be right; **its value enters no decision** |
| Gate order | Every 4th gradient of a ticket (§5.2); scores every candidate binary pattern; decides which ticket to pick / whether a chain candidate is accepted | ≈ steps/4 gradients + one forward per candidate | ≥ 2 × descent order |
| Certification ladder | Convergence ladder for the delivered pattern: **rung 1** is the value every ticket is read at (the statistical tier), the **top rung** is the reported value | Once per rung | The higher the better; the top rung may still not be converged for high target orders (adjacent rungs differ by +0.004…+0.02 at m = 3, 4), so both numbers are reported |

What the measurements say about the tiers (experiments S5, S6, S8): the gate order buys **honest reporting**, not better structures (2× is enough; selecting at the descent order over-states the delivered value by 0.1–0.3);
the descent order must scale with the target order at the start and in every re-descent (one tier lower caps the reachable efficiency, one tier higher costs 7–10× per step without a per-ticket gain);
the truncation error of the descent order switches on with the projection sharpness (|error| ≤ 0.02 for β ≤ 16, saturating at the binary gap by β ≈ 128–1000).

### 4.1 Order rule (**"Auto-fill orders from target"**)
The base is the in-film propagation cutoff M₀ = n_feat·Λx/λ: for target order m ≤ 2, descent ≈ M₀ and gate ≈ 2M₀; for m ≥ 3, descent ≈ (m+2)/3·M₀ and gate ≈ (m+5)/3·M₀;
certification adds two more rungs above the gate; My is scaled to match the period ratio. The higher the target order and the finer the pattern (the smaller the filter radius), the higher the orders required.
When the rule is clipped by the grid constraint, the interface shows the nx/ny that would be needed -- refine the grid before trusting the numbers.

**Code**: `order_rule`, `certify_ladder`, `gate_eval` in `workbench/app.py`; `rise_ladder` in `workbench/protocol.py`.

---

## 5. The optimization protocol: guarded mixed-order tickets, gate selection, certification, optional repair

### 5.1 Why a plain descent is not enough
Pixelated topology inverse design is highly non-convex: the gradient only points the way locally, and which local optimum a run ends in is decided by its random initial value. The classical remedy is
**multi-start** (one descent per seed, keep the best), and the workbench keeps that structure: one seed = one **ticket**.

Two things go wrong inside a plain descent, and the protocol below is built around them:

1. **The optimizer climbs the truncation error.** Once the projection is sharp (β ≳ 16…128) the cheap descent order over-states the objective of the current design by exactly the truncation gap of its hard binary
   (0.05–0.2 on the reference problems, +0.1…+0.4 when the continuous value at the end of a run is compared with the certified value). The optimizer follows that error as if it were physics: the continuous value keeps rising while the certified value of the
   same trajectory stalls or collapses ("cliffs" of 0.1–0.5 between adjacent checkpoints). The published single-order workflow reports the inflated number.
2. **Single-precision gradients are occasionally wrong** (§3.1): a finite gradient 10–500× too large and pointing sideways, at isolated β values (about 1–2 per 100 evaluations). Adam's per-pixel normalization turns it into a full-size step in the wrong direction and the momentum carries it for ~10 steps;
   in the reference pool one such event at a gate-order step preceded a collapse from 0.89 to 0.14.

### 5.2 One ticket
| Item | Value | Why |
|---|---|---|
| Schedule | 219 Adam steps (lr₀ = 0.02, cosine to 0), β(t) = 2·(2000/2)^{t/219}: the geometric "README ramp" | The β end point is the main schedule factor (+0.1–0.25 certified vs β → 64); 219 steps reach the 800-step medians at 60 % of the wall, so more tickets can be drawn for the same money |
| Mixed-order gradients | From the first step with β ≥ 16, every 4th gradient is taken at the **gate order** (same Adam state); the other steps use the descent order | The direct remedy for climbing the truncation error: the truncation gap along the run drops from 0.14–0.18 to 0.02–0.03 and the certified value rises by +0.09 (m = 1, 2); the gain tracks the size of the single-order gap (+0.03 / +0.01 at m = 3 / 4). Below β = 16 the descent-order gradient already agrees with the high order, so mixing earlier only costs |
| Probe | At every gate-order step a descent-order gradient of the same latent is also computed (about 7 % of the descent wall) | It gives the continuous value shown in the Progress chart, the cosine between the two orders (used by the D1/D2 variants), and the fallback update when the gate-order gradient is refused |
| Norm guard | Every gradient (descent, gate, probe): if ‖g‖ > 3 × the median of the last 8 same-order norms (after 4 warm-up evaluations, β ≥ 8), recompute at β(1 + 1 %); keep the original if it is **stable** under that change (cosine ≥ 0.9 and norm ratio in [0.5, 2] = natural growth), otherwise use the recompute (an artifact). Raised / non-finite gradients: the ladder ±1 %, ±3 %, then one CPU complex128 gradient (at most 6 per ticket); at the gate order a descent step instead | Calibrated on the reference pool: replaced gradients had norm ratios ×16–289; true positives 0.5–2 per 100 evaluations; the guard removed every cliff of the pool (0/6 vs 1/6) at no cost |
| Checkpoints | 30 per ticket (every 12 steps plus a sparse-early / dense-late set): the current latent is hard-thresholded and read at the gate order (and at the descent order, for the truncation-gap curve) | Mid-run gate reads catch cliffs and collapses: the best checkpoint is often not the last step |
| Delivery | The checkpoint with the **best gate reading** (never the final field, never the continuous value); the final field is kept for the record | The continuous value is systematically high (+0.1…+0.4); the final field can be the collapsed one |
| Price | About 13 min per ticket for the second reference problem at m = 4 (≈ 70 % of the descent wall goes to the gate-order gradients), 7 min at m = 3, 1 min at m ≤ 2 on an RTX 3060 | Measured; the panel's hint counts the gate-order gradients and checkpoint reads of your settings |

### 5.3 Selection and certification
- Draw one ticket per seed (three by default); **select by the gate reading** of the delivered checkpoint (the honest value; the continuous value would pick the most inflated ticket and lose 0.01–0.05).
- Read every ticket at rung 1 of the certification ladder; certify the winner on the whole ladder. The Summary reports **gate, rung 1, top rung, rung 1 − top** and the **gate optimism** = gate − rung 1.
  A gate optimism above ~0.03 means the gate order still flatters the design; a rung 1 − top of more than ~0.01 means the top rung is not converged -- extend the ladder (and the grid) before quoting the number.
- A small optimism and a certified value that keeps rising along the ladder is what a healthy delivery looks like; a delivery whose best checkpoint is early (step < 100) with a collapsed final field is a **repair candidate** (§5.4).

### 5.4 The repair chain (optional, off by default)
Basin hopping on the delivered ticket: each hop = incumbent latent + α × 0.08 × filtered unit-variance noise (α cycling 0.1 / 0.25 / 0.5), 30 Adam steps of re-descent at a **fixed β** with mixed-order gradients (k = 4 from step 0, fresh Adam, cosine lr, the same guards, guard windows reset per hop),
hard threshold, one forward at the gate order, **monotone acceptance** (only a higher gate reading replaces the incumbent), a rung-1 read on every accept (false-accept tracking). 15 hops by default, stopping after 12 consecutive rejections; the cost is about 2.4 tickets.

It is a tool for *repairing weak tickets*, not for improving good ones. What 93 chains on the reference problems showed:
- on the best of three tickets (certified 0.73–0.91) the chain accepted nothing in 24 of 26 cases;
- early-peaked or collapsed tickets (best checkpoint before step ~100) with a small gate optimism (< 0.03) were repaired 10 times out of 11 (+0.2…+0.4); tickets with optimism ≥ 0.03 5 out of 17; strong parents 1 out of 55;
- the re-descent β decides whether the hops can work: β ≤ 200 repaired 15/28, β ≥ 500 2/8; on the same parent and the same noise, β = 64 never lost against the delivery β of the ticket (β* ≈ 10³) -- 4 wins, 6 ties. Hence the **default β = 64**; `max(β*, 64)` is offered as an option.

The panel offers three ways to start it: **always** (right after certification), **numeric** (only if the delivered rung-1 value is below a bar you enter *and* the gate optimism is below 0.05) and **manual** (run the tickets first, look at the Summary, and start the chain on the saved delivery from the Summary or History tab).
When the chain ran, the delivered pattern is its incumbent (which is never below the parent at the gate).

### 5.5 Equal-wall-clock control
When the chain ran and the control is ticked, the chain's wall clock is spent on extra tickets (a new one is started only if it is expected to finish inside the budget), and "original tickets + extra tickets, best by gate" is compared with the chain incumbent at rung 1. This is the honest question about the chain -- "would the same money buy more as tickets?" -- and on strong parents the answer was usually yes.

### 5.6 Optional variants (switches, off by default)
- **D1 adaptive cadence**: the mixing interval k ∈ {2, 4, 8} follows the cosine between the probe and the gate-order gradient (two consecutive probes > 0.95 → k doubles; < 0.85 → k halves), under a cap on the total number of gate-order steps equal to what k = 4 would spend. Observed: k = 8 early, k = 2 late.
- **D2 online rung rise** (needs D1): on the same "< 0.85 twice" event the descent order also climbs one rung towards the gate order (Mx + 4 per rung) and stays there; by step ~200 every ticket descends at the gate order. About +30 % cost.
- Status: on six tickets each they were 0.02–0.03 above the default at the median and at E[max of 3] (0.899 / 0.907 vs 0.880 certified), with one cliff each -- "promising, unconfirmed" by the adoption rule written before the data (not lower on both statistics *and* fewer cliffs or cheaper).

### 5.7 Things that were tried and do not help
| Not done | Counter-evidence |
|---|---|
| Eigen-gap guard | The wrong gradients sit at ordinary gaps; 5 % of healthy steps have smaller gaps |
| Direction guard (cosine to the previous gradient) | On a moving latent the cosine between consecutive gradients has median 0.88 and a 25 % quantile of 0.37 |
| Rejecting the gate-order gradient when it disagrees with the probe | Late in the ramp the two orders genuinely point in opposite directions (cosine −0.6…−0.9): that *is* the truncation error being climbed |
| Lorentzian eigen-backward broadening | Does not change the wrong gradients; bends healthy ones |
| A separate Adam state for the gate-order steps | The bias-corrected first step is a full-size sign step |
| Reallocating the ramp density (more steps at β 16–128) | Not better than the parent; two more early-peaked tickets |
| Ending the ramp at β = 64, or re-descending at the ticket's delivery β | Tickets want β → 2000, the chain wants a low β; the chain only judges hard binaries, so the two do not contradict |
| Reporting the continuous value, or a robust min{nominal, CD±} score as the chain's acceptance | Inflated by +0.1…+0.4; the acceptance is the nominal gate reading |

### 5.8 Setting the parameters for a new problem
1. Orders: **Auto-fill from target**; if the rule is clipped, refine the grid first.
2. Price ladder: time one gradient at each order before choosing the ladder (the Summary's per-ticket walls give it after the first run).
3. Ramp: 219 steps, β 2 → 2000 are the calibrated values for the two reference problems; do not move them first.
4. Guards: 3× / window 8 / warm-up 4 / β ≥ 8 / stability 0.9 & [0.5, 2] were calibrated on GPU complex64; after changing precision or solver, run once with the guard unticked and look at the trigger counts in the Summary (1–3 % "would trigger" is the calibrated regime).
5. Repair: start it by hand on tickets that peaked early or collapsed, with a small gate optimism; β = 64.

### 5.9 The legacy chain protocol
Kept under "Legacy chain protocol" for comparison with the pre-2026-09 shipped method: a few start descents (β 1 → 64, best gate checkpoint) → 40 basin-hopping jumps with a 30-step single-order re-descent at β = 64 → monotone gate acceptance → certification, plus the equal-wall-clock restart lottery. Its tickets are the old β → 64 ones and it has no guards; use it only to reproduce old runs.

**Code**: `GradPolicy`, `TicketDescent`, `repair_chain` in `workbench/descent.py`; `run_ticket`, `guarded_mixed_protocol`, `run_repair`, `control_arm_gm`, `run_optimize` in `workbench/app.py`;
the parameter blocks `DESCENT / GUARDS / REPAIR / CADENCE / RISE`, `beta_geometric`, `o_ckpts`, `rise_ladder` in `workbench/protocol.py`; legacy: `e10_descent`, `chain_protocol`, `restart_arm`.

---

## 6. Robustness: accounting for fabrication deviations

### 6.1 Why
The optimizer will exploit degrees of freedom that the solver allows but the process does not: sub-resolution thin lines, boundary pixels sitting right at the threshold. Such structures are highly efficient at the nominal geometry and collapse as soon as the boundary moves.

### 6.2 How fabrication defects are simulated
There is no physical lithography simulation; instead, the delivered binary pattern is subjected to **geometric perturbations**. The basis is the periodic **signed distance field** d(p): the distance from each pixel centre to the nearest pixel centre of the other class, positive inside material (a 3×3 tiling enforces periodicity; units of nm).

| Defect | Operator | Meaning |
|---|---|---|
| CD bias | grow by b: keep {p : d(p) ≥ −b}; shrink by b: keep {p : d(p) > b} | Every boundary grows (b>0) or shrinks (b<0) by b nm; models a global dose/etch deviation |
| Opening / closing | Morphological opening/closing with a disk of radius r | Erases lines thinner than 2r / fills gaps narrower than 2r; models the resolution limit |
| Line-edge roughness | d(p) ≥ −(b + σ·g(p)), with g a correlated Gaussian field | Random fluctuation of the boundary (a deterministic sample, not claimed as a worst case) |
| Sidewall tilt | Different biases at top and bottom, stacked as multiple layers | Non-vertical etch |

Note the pixel quantization: a bias smaller than one pixel changes no pixel at all, and grow / shrink are symmetric (a bias of one pixel moves every edge by one pixel in either direction).

### 6.3 Robustness testing and robust acceptance
- **Audit** (held-out): after delivery, the perturbed patterns are re-solved at the certification order, and the worst-case efficiency and the "worst / nominal" ratio are reported. The audit scenarios **never** take part in the optimization,
  so what you get is generalization to unseen perturbations rather than overfitting to the audit. The workbench audits CD ±20 nm by default.
- **Robust acceptance rule** (legacy chain protocol only): the chain's acceptance score is replaced by min{nominal, CD−b, CD+b} (all computed at the gate order), at a cost of two extra forward solves per jump.

**Code**: `trustworthy_fmm/fabrication/process.py` (signed distance, morphology, correlated random field, soft surrogate),
`cd_bias_mask` in `workbench/protocol.py`, `gate_score` in `workbench/app.py`.

---

## 7. Engine cross-checks

- **TORCWA** (an open-source differentiable RCWA solver): for the same pattern at the same truncation order, the two engines agree on η to the 1e-5 level -- this validates the implementation, not truncation convergence.
  The published TORCWA workflow (800 steps, one truncation order, the continuous value reported) was the comparison target of the protocol in §5: at equal wall clock the tickets of §5.2 delivered higher certified values on the reference problems, and the numbers the single-order workflow reports were inflated by +0.1…+0.4.
- **Ansys Lumerical RCWA**: the binary pattern is first converted exactly into pixel-boundary polygons (rasterizing them back must reproduce the pattern bit for bit), then handed to Lumerical's built-in RCWA solver,
  solved on its k-vector ladder, and the result is compared with this engine's highest-order value. Only x-polarized normal incidence is supported (P-source convention).

**Code**: `torcwa_block / lumerical_block` in `workbench/app.py`, `workbench/polygons.py`.

---

## 8. Glossary

| Term | In one sentence |
|---|---|
| Truncation order (Mx, My) | Half-width of the Fourier orders kept in each direction; matrix dimension 2N, N = (2Mx+1)(2My+1) |
| Relaxed value | Efficiency of the continuous (gray-scale) design, the optimizer's objective; unmanufacturable and biased high |
| Binary@gate | Efficiency of the manufacturable pattern after hard thresholding, at the gate order; the basis for every selection |
| Certification ladder / rung 1 / reported value | Sequence of efficiencies of the delivered pattern at increasing truncation orders / the first rung above the gate (every ticket is read there) / the value at the highest rung |
| Gate optimism | Gate reading minus rung-1 value of the same pattern; how much the gate order still flatters the design |
| Drift | Change from the gate value to the reported value; a strongly negative drift means the design was being "flattered" by the low order |
| Ticket | One guarded mixed-order descent from one seed, delivering its best gate checkpoint |
| Mixed-order gradients | Every k-th gradient of a ticket taken at the gate order in the same Adam state; the remedy for climbing the truncation error |
| Probe | The descent-order gradient computed alongside every gate-order gradient (same latent) |
| Norm guard / true positive | The trigger on a gradient norm 3× above its recent median / a triggered gradient that changed under a 1 % β change and was replaced |
| Cliff | Best gate checkpoint minus the final field's gate reading; a large cliff means the run collapsed after its best point |
| β* | The β of the delivered checkpoint (used by the optional `max(β*, 64)` re-descent rule) |
| Repair chain | Basin hopping on the delivered ticket at a fixed low β; a repair tool for weak tickets, not an improvement tool |
| Equal-wall-clock control | The control arm that spends the chain's wall clock on extra tickets |
| Held-out audit | Re-evaluation after delivery on perturbation scenarios never used during optimization |
