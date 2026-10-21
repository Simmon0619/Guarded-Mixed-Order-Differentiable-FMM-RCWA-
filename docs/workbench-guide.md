# Browser workbench guide

> A local, browser-based workbench for differentiable RCWA (Fourier modal method):
> set up a periodic grating, run GPU-accelerated forward solves or inverse design,
> and inspect patterns, near fields, engine cross-checks and timings. Python + one HTML file, no build chain.

> Installation, command-line use and validation: [project README](../README.md).
> Chinese guide: [README.zh-CN.md](../README.zh-CN.md).
> Historical experiment counts and timings below describe the development study; their original datasets are not included in this distribution. They are not independently reproduced by the smoke suite.

This is a web-based workbench that **runs on your own machine**: you enter parameters in the browser, the backend runs forward solves or inverse design with a differentiable RCWA solver,
and the results (patterns, field maps, engine comparisons, convergence curves, timings) appear on the page in real time. It is written for people encountering RCWA inverse design for the first time:
this manual assumes you can install Python packages from the command line, but it does not assume you know RCWA. Algorithm details are in [`docs/algorithm.md`](algorithm.md).

---

## 1. Installation and startup (5 minutes)

**Requirements**: Python 3.9+. An NVIDIA GPU is best (8 GB or more of GPU memory handles the usual truncation orders); without one you can still run small problems on the CPU.

```bash
# 1) Install dependencies. If you have an NVIDIA GPU, install the CUDA build of torch following the instructions at https://pytorch.org
pip install -r requirements.txt

# 2) Start the server
python run.py --port 8321

# 3) Open in your browser
http://127.0.0.1:8321
```

The top of the page shows the backend environment (Python and torch versions, GPU model and memory, whether TORCWA and the Lumerical API are available). If a GPU model is listed, the GPU has been detected. The **中文 / English** buttons at the top right switch the interface language; the browser remembers your choice.

Optional: `pip install torcwa` enables the comparison with the TORCWA engine. If Ansys Lumerical is installed in its default location, the Lumerical RCWA cross-check is enabled automatically; otherwise set the environment variable
`LUMERICAL_API_PATH` to its `api/python` directory (see §9).

Two environment variables matter for speed and stability: `TFMM_EIG_DEVICE` (default `cpu`: the eigendecomposition runs on the host because the CUDA eigensolver of torch 1.13 leaks host memory; set `native` on a torch build that does not) and `FMM_THREADS` (BLAS threads for that eigendecomposition, default 3).

On a machine you have not used before, run `python selfcheck.py` first (§9).

Directory layout:

```
fmm_workbench/
  run.py                 launcher script
  selfcheck.py           environment report + solver self-test for a new machine
  requirements.txt
  README.md              this manual (English)
  README.zh-CN.md        this manual (Chinese)
  docs/algorithm.md      algorithm documentation (English)
  docs/algorithm.zh-CN.md algorithm documentation (Chinese)
  workbench/
    app.py               backend: HTTP server + job queue + solve/optimization flow
    descent.py           the ticket (guarded mixed-order descent) and the repair chain
    index.html           frontend: single-file web page (all JS/CSS inlined)
    protocol.py          problem configuration, protocol defaults, schedules, CD-bias operator
    polygons.py          binary pattern → exact polygons (for Lumerical)
    vendor/plotly.min.js plotting library (offline)
    data/runs/           results of every run (generated automatically)
  trustworthy_fmm/       differentiable RCWA solver library
```

---

## 2. Three-minute quick start: run the default optimization

1. After startup, change nothing and click **"Run optimization"** at the bottom left. (The panel opens with the truncation orders already filled in by the order rule for the default problem.)
2. Switch to the **"Progress"** tab: you will see status cards and curves updating in real time -- which ticket is running, the current step and truncation order, the best binary checkpoint so far, and the guard counters.
3. When the status changes to `done` the page jumps to **"Summary"** automatically. With the default settings the run takes about half an hour on the reference GPU (three tickets of roughly ten minutes each, then the certification ladder).

The default problem is a two-dimensional grating: 660 nm wavelength, x-polarized normal incidence, a 2400 × 700 nm period, an 800 nm-thick pattern layer with n=3.2,
light entering from an n=1.46 substrate and exiting into air; the goal is to send as much of the transmitted light as possible into the (3,0) diffraction order. The default pattern grid of 480×140 corresponds to 5 nm pixels.

**The numbers to read** (the cards at the top of the Summary tab):

| Card | Meaning | Which one to trust |
|---|---|---|
| Reported value: η @ certification order | Target-order efficiency of the delivered binary pattern at the **highest truncation order** | **This is the final number** |
| Rung-1 certified η | The same pattern at the first certification rung; every ticket is read there, so it is the number tickets are compared on | Statistical tier |
| Gate value @ gate order | The pattern at the intermediate order used for every selection during the run | Intermediate quantity |
| Gate→certification drift | Reported value minus gate value; strongly negative means the design exploits the low-order truncation error | Diagnostic; the closer to 0 the better |
| Gate optimism (gate − rung 1) | How much the gate order still flatters the design | Below ~0.03 is healthy |

If the drift is more negative than −5 % or the optimism is large, see §5.

---

## 3. Left panel: what each parameter means

### Incidence
- **Wavelength λ (nm)**, **Incidence θ (°)** / **φ (°)** (in degrees; θ is the angle from the surface normal, φ is the azimuth).
- **Polarization**: x or y linear, or a custom Jones vector (jx, jy); its components are complex numbers, so enter the real and imaginary parts separately.
- **Incident-side index** / **Exit-side index**: the refractive indices of the semi-infinite media on the two sides. For example, glass into air: 1.46 / 1.0.

### Materials & thickness
- **Feature index n** / **Background index n**: the refractive index where the pattern layer "has material" and where it "has none" (the solver works with n²).
- **Thickness (nm)**: the thickness of the pattern layer.

### Period & grid
- **Period Λx (nm)**, **Period Λy (nm)**: the unit-cell size.
- **Grid nx × ny**: the number of pixels in the pattern; the panel shows the corresponding pixel size. Smaller pixels give finer boundaries but also a larger optimization space.
- Constraint: a truncation order (Mx, My) requires **nx ≥ 4Mx+1 and ny ≥ 4My+1** (the panel warns you); otherwise the solver cannot build the convolution matrix.

### Truncation orders & target (the most important group)
RCWA represents the fields with a finite number of Fourier orders: more orders means higher accuracy, longer run time and more GPU memory. The workbench splits this into three tiers (see §4 of the algorithm doc):

- **Descent order (Mx, My)**: the order used for most gradients of a ticket. Cheap; it only needs to get the gradient direction right.
- **Gate order (Mx, My)**: the order of every 4th gradient of a ticket and of the one forward solve performed on each candidate binary pattern (recommended ≥ 2 × the descent order). **Every selection and accept/reject decision is made here.**
- **Certification ladder**: an increasing sequence of orders used to score the delivered pattern after the optimization ends; the first rung is where every ticket is read, the highest order gives the reported value.
- **Target order (m, n)**: the transmitted diffraction order to maximize.
- **"Auto-fill orders from target"**: fills in all three tiers from the target order and the physical parameters (the rule is in §4.1 of the algorithm doc): M₀ = n_feat·Λx/λ, descent ≈ M₀·k_d and gate ≈ M₀·k_g rounded to multiples of 4 (k_d, k_g = 1, 2 for m ≤ 2 and (m+2)/3, (m+5)/3 for m ≥ 3), My from the period ratio, two certification rungs above the gate; orders the grid cannot hold are clipped and the needed nx/ny are shown. The panel does this once on load.
- **"Extend ladder to grid limit"**: replaces the certification ladder by rungs above the gate order (its aspect ratio, My + 1 per rung) up to the highest order the current grid allows.
- The line under the orders reads "descent 2N = …; the top order needs nx ≥ …, ny ≥ …": 2N = 2(2Mx+1)(2My+1) is the matrix dimension of the descent order (cost), and the grid requirement nx ≥ 4Mx+1, ny ≥ 4My+1 comes from the largest order among descent, gate and ladder (§2.2 of the algorithm doc).

### Fabrication constraints (design map)
- **Gaussian filter radius (nm)**: the latent first passes through a periodic Gaussian filter of this radius, which smooths the density before thresholding; the delivered binary pattern still needs a separate minimum-linewidth and spacing check. This is the main manufacturability constraint.
- **Binarization threshold**: after filtering, ≥ threshold = material.
- **Mirror symmetry**: forces the pattern to be mirror-symmetric about y (or x, or both x and y); symmetry shrinks the search space and stabilizes the optimization.
- **CD-bias audit (nm)**: after delivery, shrinks/grows the boundary by these distances and re-solves, to check how much efficiency survives fabrication errors (see §6 of the algorithm doc).

### Optimization protocol
- **Guarded mixed-order tickets (default)**: one ticket per seed = one 219-step descent on the geometric β ramp 2 → 2000, with every 4th gradient at the gate order from β ≥ 16 and a norm guard on every gradient; 30 hard-binary checkpoints read at the gate order; the delivery of a ticket is its best gate checkpoint; the best ticket by gate reading is certified. Adjustable: the ramp (β₀, β_end), the mixing interval k and its starting β, the guard on/off, and the two optional variants **D1** (adaptive cadence) and **D2** (online rung rise; needs D1). See §5 of the algorithm doc.
- **Repair chain** (checkbox, off by default): basin hopping on the delivered ticket -- perturb, re-descend 30 steps at a fixed β with mixed-order gradients, score the hard binary at the gate order, accept only improvements. **Trigger**: *manual* (run the tickets first; the Summary and History tabs offer a "start the chain" button on the saved delivery), *always*, or *numeric* (only if the delivered rung-1 value is below a bar you enter and the gate optimism is below the max). **Re-descent β**: fixed 64 (default; it never lost against the alternative on the same parent) or max(β*, 64). Also: hops (15) and the stop rule (12 consecutive rejections), the α rotation, re-descent steps / mixing k, the noise seed, and the **equal-wall-clock control** (the chain's wall clock spent on extra tickets, as evidence).
- **Legacy chain protocol**: the pre-2026-09 method (start restarts with β 1 → 64, 40 jumps at β = 64, robust acceptance option, equal-wall-clock lottery). Kept only to reproduce old runs.

### Optimization settings
- **Descent steps (per ticket)**: 219 by default (the mixing start, the checkpoint set and the D1 budget all follow from it).
- **Random seeds**: one seed = one independent random initialization = one ticket; the number of seeds is the number of tickets (one seed = no restart). The numbers themselves are arbitrary; they exist only for reproducibility.
- **Adam initial lr**: the learning rate (it follows a cosine decay over the ticket).

### Engine & comparisons
- **Solver engine**: `GPU / complex64` (recommended, fast) or `CPU / complex128` (double precision, slow, for high-precision checks).
- **Compare with TORCWA**: re-solves the delivered pattern with the TORCWA engine and compares efficiency and time order by order (requires torcwa).
- **Ansys Lumerical RCWA cross-check**: sends the pattern to the RCWA solver bundled with Ansys Lumerical (requires a local installation and a license).
- **Memory-efficient path**, **save_on_cpu**: two switches for when GPU memory runs short; see §7. The "auto" setting of save_on_cpu is scaled to the GPU found on this machine (the panel shows its threshold).
- **Field-map z range**: depth drawn on the incident (substrate) side and distance drawn on the exit side.
- **Field maps (optional)**: which extra maps to compute besides the Re(E) x–z cut (see §4).

### Pattern source (single solve)
When you only want a single solve without optimization, the pattern can come from: a random-seed initial pattern, a uniform fill (for example an all-material slab, useful for verification), or the best pattern of a saved run.

---

## 4. The six tabs on the right

**"Progress"**: status cards (ticket i/n and seed, step and current truncation order, the descent-order value / best binary@gate, the guard counter "replaced / triggered"); the descent curve, where the thin line is the "relaxed value" (the efficiency of the gray-scale design the optimizer sees internally, which is **not manufacturable and systematically too high**) and ◆ marks "binary@gate" (the efficiency of the actual candidate pattern at the gate order); when a repair chain runs, a hop-by-hop "candidate vs incumbent" plot and, with the control ticked, the equal-wall-clock plot; the job list lets you cancel jobs (a cancelled job keeps the delivery if it was already certified).

**"Patterns"**: three heat maps, latent → filtered density → delivered binary, in physical coordinates (nm).

**"Fields / phase"**: total transmission, total reflection and energy-conservation residual; a bar chart of the transmitted power in every propagating diffraction order (the target order in a darker color); an order table (wave vector, power, complex amplitude, phase); field maps reconstructed from the modal solution. Always drawn: an x–z cut through the symmetry plane over two periods of Re(E) for the dominant component (Ex for x polarization, Ey for y; the wavefronts) on the incident side, **inside the pattern layer** (from the layer eigenmodes, with the pattern outline drawn on top) and on the exit side, with the direction of the target order marked. Optional (checkboxes in the panel, computed only when ticked): |E|² x–z (the layer band drawn as the structure's cross-section, switchable to the interior field), phase x–z, and |E|² x–y planes one wavelength beyond the exit interface and at mid-layer. Incident |E| = 1; z = 0 is the incident interface. Note that |E|² is dominated by the evanescent near field at the pattern edges within ~100 nm of the interfaces and is nearly featureless beyond λ/2, which is why Re(E) is the default.

**"Engine comparison"**: the **truncation convergence curve** of the delivered pattern (η versus the total number of orders 2N, with the descent order and gate order marked) and its numeric table; efficiency and time for this engine vs TORCWA, order by order; the Lumerical cross-check table.

**"Summary"**: the cards of §2; warnings (for example a descent order below what the rule recommends, or a clipped grid); the **repair chain** block (trigger, decision with the three criterion values, β, hops / accepts / false accepts, rung 1 before and after, and the "start the chain" button when none ran); the **equal-wall-clock control** table; the **tickets** table (best step, β*, gate, rung 1, optimism, final rung 1, cliff, high-order steps, guard triggers / replacements, complex128 fallbacks, price and measured wall); the **guard statistics** per truncation order; the D1/D2 table when those switches are on; the certification ladder; the robustness audit (CD bias); engine & timing.

**"History"**: every job you have run is stored in `workbench/data/runs/`; click **"Load"** to load it back into the tabs, or **"chain"** to start the repair chain on that run's delivery (the problem definition comes from the saved run, the chain settings from the panel). A saved run can also serve as the pattern source for a single solve.

---

## 5. How to read "convergence", "drift" and "optimism"

The same pattern **legitimately gives different efficiencies** at different truncation orders: more orders represent sharp material boundaries better, and the efficiency converges gradually with 2N.
Changing the truncation order changes the finite-dimensional approximation. Agreement with another solver at matched settings checks the implementation; convergence must be assessed separately. Historical TORCWA comparisons reported differences around 1e-5 on the tested cases, which should be rerun for the configurations used in a manuscript.
Therefore:

- The reported value is always taken at the **highest order** of the certification ladder, and the rung-1 value is reported next to it;
- The convergence table on the **"Engine comparison"** tab gives the relative change between adjacent orders; when the change between the two highest orders is < 1% you can consider it converged, otherwise extend the ladder upward (and enlarge nx/ny if the grid is too small). For high target orders the top rung is often not converged yet: quote both numbers;
- A large (strongly negative) **"Gate→certification drift"** or a gate optimism above ~0.03 means the optimizer found a structure at the cheap truncation orders that exploits the truncation error. The remedy is to raise the descent order or the gate order (use the rule), not to trust the gate value;
- A ticket whose best checkpoint is early (step < 100) with a collapsed final field is what the repair chain is for.

---

## 6. Frequently asked questions

**The optimized efficiency is low / different seeds differ a lot.** This is the nature of inverse design (it is non-convex). Add seeds: each is one more ticket, and the best by gate reading is delivered. Mixed-order gradients and the guard remove most of the collapses a plain descent shows; what remains is the lottery over initial values.

**When should I start the repair chain?** On a delivery that peaked early (best checkpoint before step ~100) or collapsed, with a small gate optimism (< 0.03): those were repaired 10 times out of 11 in the reference experiments (+0.2…+0.4). On a healthy delivery (certified ≥ 0.8, late best checkpoint) the chain accepted nothing in 24 of 26 cases -- the same time buys more as extra tickets. Keep the re-descent β at 64; 15 hops cost about 2.4 tickets.

**The guard triggers many times -- is the run broken?** No. 1–3 % of the gradient evaluations trigger on the reference problems and about a third of those are replaced (0.5–2 per 100 evaluations); the Summary's guard table shows the counts per order. Zero triggers on a new problem or precision is worth a look, and so is a rate far above 5 % (rerun with the guard unticked to compare).

**Out of GPU memory (CUDA out of memory), or the run suddenly becomes extremely slow.** See §7.

**Error "grid ... violates the convolution constraint of order ...".** Increase nx/ny or lower the order; the hint line in the panel gives the minimum grid.

**In the CD-bias audit, −10 nm is identical to nominal.** A bias smaller than one pixel does not change any pixel: on a 20 nm pixel grid ±10 nm is a no-op, whereas on a 5 nm grid ±20 nm is a real 4-pixel perturbation in either direction.

**The TORCWA comparison is very slow at high orders.** That is TORCWA's own high-order cost; running the comparison at an intermediate order (for example the gate order) is enough to demonstrate that the engines agree.

**Can I change the physical model (multiple layers, dispersive materials, other shapes)?** The solver library supports multilayer stacks and arbitrarily sampled permittivity distributions, but the current workbench only exposes "one pattern layer + semi-infinite media on both sides"; the extension points are `build_cfg` and `solve_density` in `workbench/app.py`.

---

## 7. GPU memory and speed: a few rules of thumb

- All the large matrices are 2N × 2N with N = (2Mx+1)(2My+1); **both GPU memory and time grow as N squared to N cubed**. Doubling N (for example doubling Mx alone) costs about ×4 in GPU memory and about ×8 in time.
- Most of the time in a single solve goes to the matrix eigendecomposition, followed by the S-matrix algebra; each optimization step = one forward + one backward pass. In a ticket about 70 % of the descent wall goes to the gate-order gradients (every 4th step), so the gate order sets the price.
- The eigendecomposition runs on the CPU by default (`TFMM_EIG_DEVICE=cpu`, +30 % on that step, no host-memory leak); `FMM_THREADS` sets its BLAS threads (default 3). The reference timings in the documentation were measured with these defaults.
- **Memory-efficient path** (on by default): assembles the matrices in a mathematically equivalent way that needs fewer temporaries; peak forward GPU memory drops noticeably, and the numbers differ only at the level of floating-point rounding.
- **save_on_cpu** ("auto" turns it on when the largest gradient order has 2N ≥ 3000·√(GPU memory / 12 GB), i.e. 2450 on an 8 GB card, 3000 on 12 GB, 4240 on 24 GB): moves the intermediates saved for backpropagation to host memory, at the cost of 10–20% more time.
- Empirical ceiling for a 12 GB-class GPU: in complex64, value + gradient at 2N ≈ 5000 completes cleanly; beyond that you enter GPU memory paging and speed drops several-fold. In that case lower the gate order and use the high orders only for certification.
- Certification (no gradient) is much cheaper than an optimization step, so the certification order can be much higher than the descent order.
- The CPU double-precision mode is slow; use it only for high-precision checks on small problems.

---

## 8. Data and licensing status

- Local browser runs are stored in `workbench/data/runs/` in a source checkout and excluded from version control. Installed distributions use `~/.fmm-workbench/runs/`; use `FMM_RUNS_DIR` or `--runs-dir` to override it. Small validation inputs and an analytic reference are included under `examples/` and `reproduce/`; the original manuscript datasets are not bundled.
- The project is licensed under MIT; see the root LICENSE and THIRD_PARTY_NOTICES.md.
- When an optional dependency (TORCWA, Lumerical) is missing, the corresponding feature is disabled automatically and the UI says so; the main workflow is unaffected.

---

## 9. Deploying on another machine

Copy this one directory, `fmm_workbench/` (including its `trustworthy_fmm/`); nothing outside it is needed. On the new machine:

1. **Python ≥ 3.9**, preferably in a fresh virtual environment (conda or venv).
2. **Install torch for that machine's GPU.** Different GPUs and drivers need different torch builds: at https://pytorch.org/get-started/locally/ pick the CUDA version that matches the driver (`nvidia-smi` prints the highest CUDA version the driver supports); without an NVIDIA GPU install the CPU build. Then `pip install numpy scipy` (or `pip install -r requirements.txt`).
3. **Run the self-check**: `python selfcheck.py`. It prints the Python / torch / CUDA / GPU configuration and runs four checks: the solver library imports from this directory; a uniform slab reproduces the analytic Airy transmission (acceptance threshold: absolute transmission error < 1e-8); energy is conserved on a patterned structure; and, when a GPU is present, the GPU/complex64 value and gradient agree with the CPU/complex128 reference. Add `--bench` to time one optimization step on your hardware.
4. **Optional engines**: after `pip install torcwa` the self-check also runs a TORCWA parity check. Ansys Lumerical must be installed locally with a license; the workbench looks for its API under `C:\Program Files\Lumerical\v*\api\python` (Windows) or `/opt/lumerical/v*/api/python` (Linux) and takes the newest version; if it lives elsewhere, set `LUMERICAL_API_PATH` to that `api/python` directory. When an engine is missing, its options are disabled in the panel with a note at the top; nothing else is affected.
5. **Different GPU memory**: the "2N ≈ 5000" ceiling in §7 is for a 12 GB card; with less memory, lower the certification orders accordingly. A job that runs out of memory ends with a `CUDA out of memory` error in the job list and does not affect later jobs.
6. **Guard calibration**: the guard thresholds were calibrated on GPU complex64; on another precision or a torch build whose eigensolver behaves differently, run once with the guard unticked and compare the trigger counts (§6).
7. Start with `python run.py --port 8321`; the server listens on 127.0.0.1 only.

The historical GPU environment was Windows 10 with torch 1.13 and CUDA 11.7. The current CPU validation profile and verification scope are documented in the [project README](../README.md) and [validation notes](validation.md). AMD (ROCm) and Apple Silicon (MPS) have not been validated. The browser workbench falls back to the CPU when CUDA is unavailable; an explicit gpu_c64 request in the batch interface instead fails with a diagnostic.
