# Guarded Mixed-Order Differentiable FMM/RCWA

A PyTorch-based Fourier modal method (FMM), also known as rigorous coupled-wave
analysis (RCWA), with guarded mixed-order inverse design, a local browser
workbench, and a command-line interface for reproducible experiments.

The optimizer uses inexpensive truncation orders for most gradient evaluations,
interleaves higher-order gradients, selects hard-binary checkpoints at a common
gate order, and evaluates the delivered design on a higher-order ladder. Optional
gradient guards and a repair chain support the existing research workflow.

[Chinese guide](README.zh-CN.md) · [Browser guide](docs/workbench-guide.md) ·
[Algorithm notes](docs/algorithm.md) · [Validation](docs/validation.md) ·
[Reproduction framework](reproduce/README.md)

## Scope and status

The solver library contains one- and two-dimensional isotropic RCWA routines,
multilayer scattering-matrix composition, differentiable Fourier operators, and
geometric fabrication-perturbation utilities. The browser and batch workflows
currently expose a single patterned layer with real, constant refractive indices.
The browser provides field maps, diffraction-order powers, convergence curves,
optimization history, and optional comparisons with TORCWA and Ansys Lumerical.

This distribution includes executable examples, an analytic reference, and a
curated archive of the manuscript experiments. The F33 statistics can be
recomputed from 72 paired records. Some original design arrays are missing;
full numerical reproduction and pattern-inset regeneration remain incomplete.
See the [paper archive](paper/README.md) for the available evidence and gaps.

## Installation

Python 3.9 or newer is required by the package metadata. Use an isolated
environment. Python 3.12 is the reference version for the current CPU tests.

```bash
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` in Windows PowerShell,
or `source .venv/bin/activate` on Linux/macOS. If PowerShell activation is
unavailable, invoke `.venv\Scripts\python.exe` directly in the commands below.

For the pinned CPU validation profile, run these commands from the repository
root:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/cpu-reference.txt
python -m pip install -e ".[test,plots]"
```

For an NVIDIA GPU, install a compatible CUDA build using the
[official PyTorch instructions](https://pytorch.org/get-started/locally/), then
run `python -m pip install -e ".[test]"`. Do not apply the CPU reference profile
on top of a separately configured research GPU environment.

Use `python -m pip install .` for a regular installation, or an editable install
(`-e`) while developing. The console command and browser assets are included in
the package. The original source-checkout route remains available:

```bash
python -m pip install -r requirements.txt
python selfcheck.py
python run.py --port 8321
```

Optional extras are `test` (pytest), `plots` (Matplotlib), `compare` (TORCWA), and
`dev` (testing and build tools). Lumerical is supplied by Ansys and requires a
separate installation and license; it is not distributed with this project.

## Browser workbench

```bash
fmm-workbench serve --port 8321
```

Open [http://127.0.0.1:8321](http://127.0.0.1:8321). The server binds to the local
machine. `python run.py --port 8321` and `python -m workbench serve --port 8321`
provide equivalent entry points. The English/Chinese interface and existing
solve, optimize, history, field-map, and comparison features are retained.

A source checkout stores browser history in `workbench/data/runs/`. A regular
package installation uses `~/.fmm-workbench/runs/` so that results are not written
into the installed package. Override the location with `--runs-dir PATH` or the
`FMM_RUNS_DIR` environment variable.

The browser's default optimization is a research-sized calculation, not a quick
installation test. Start with the CPU examples below to verify a new machine.
See the [browser guide](docs/workbench-guide.md) for the individual controls,
protocol options, field conventions and optional solver comparisons.

## Command-line experiments

The batch interface runs the same numerical routines as the browser worker,
without starting an HTTP server:

```bash
fmm-workbench solve --config examples/slab.json --output outputs/slab
fmm-workbench optimize --config examples/optimize-small.json --output outputs/optimize-small
```

`python -m workbench` can replace `fmm-workbench` in each command. Configuration
paths are explicit, so installed commands can be used from another directory.
Output directories must be new: existing results are never overwritten.

Inputs are versioned JSON objects containing `schema_version`, `type`, `params`,
and optional `name` and `description` fields. Parameters use the browser
backend's names. Unknown parameter names and common invalid values are rejected.
The supplied configurations show complete small forward and optimization cases.

Batch runs omit field maps and external solver comparisons by default. Add
`--fields` for field maps, or explicitly set `compare_torcwa` / `compare_lumerical`
in the input for comparisons. Numerical solver and optimization defaults are
otherwise shared with the browser. An explicit `gpu_c64` batch request fails
when CUDA is unavailable, making an unintended CPU run visible; select
`cpu_c128` to run on the CPU.

Each successful run exports its input and resolved settings, result JSON,
trajectories, unrounded NumPy arrays, environment and source metadata, and file
checksums. Numerical arrays are separate from the existing display JSON, which
rounds some values. Non-finite diagnostic values are represented as JSON `null`.
See the [output format](reproduce/README.md#outputs-and-verification) for details.

The synchronous Python API uses the same configuration format:

```python
from workbench.configuration import load_config
from workbench.batch import run_experiment

config = load_config("examples/slab.json")
result = run_experiment(config, "outputs/slab-api")
print(result["eta_target"])
```

Batch calls are serialized because the inherited backend uses process-wide
numerical flags. Run the web server and batch jobs in separate processes.

## Validation

```bash
fmm-workbench env
fmm-workbench selfcheck
python -m pytest -q
```

`python selfcheck.py` remains supported. Add `--bench` to the self-check for
larger value-and-gradient timing cases.

The automated suite checks analytic slab transmission, lossless energy balance,
a finite-difference directional derivative, alternative numerical paths,
gradient-guard behavior, mixed-order cadence, gate-based checkpoint selection,
batch exports, CLI invocation and browser endpoints. The CUDA comparison is
skipped explicitly when a CUDA device is unavailable.

The current local CPU validation uses Python 3.12.14, PyTorch 2.5.1+cpu, NumPy
1.26.4 and SciPy 1.13.1 on Windows. The historical research environment was
Windows 10 with PyTorch 1.13 and CUDA 11.7; it has not been rerun as part of this
packaging update. Linux/Windows CPU CI is configured, but a workflow file alone
does not establish a successful CI run. ROCm and MPS have not been validated.

The dependency lower bounds are inherited from the original project and do not
imply that every newer version combination has been tested. Use the reference
profile for CPU checks and archive the actual GPU environment with paper results.
Read the [validation notes](docs/validation.md) for coverage and limitations.

## Reproducing manuscript results

Run the supplied infrastructure smoke suite and regenerate its summary and plots:

```bash
python reproduce/run_experiments.py --manifest reproduce/smoke-manifest.json --output outputs/smoke
python reproduce/make_figures.py --input outputs/smoke --plots
```

The slab case has an independent analytic reference. The six-step optimization
exercises the software pipeline at deliberately low truncation orders. Neither
is a substitute for a manuscript-scale experiment.

The manuscript archive includes F31/F33, S4–S10, supporting validation data,
original research scripts and saved figure assets. Verify it and regenerate the
available statistics and figures with:

```bash
python reproduce/paper.py verify
python reproduce/paper.py analyze --output outputs/paper-analysis --figures
```

This command recomputes the F33 analysis, checks all statistics against the
archived reference, generates the SI tables, and redraws the Fig. 3 panels and
SI Figs. S1/S3/S4/S5. It performs no new physics solves. The experiment mapping,
seeds and original commands are recorded in `reproduce/paper-manifest.json`.
The generic batch runner still rejects that manifest's empty `experiments` list:
historical research scripts have a separate entry point and must not be treated
as verified batch configurations.

Original `.npy`/`.npz` design arrays were absent from the supplied project. Fresh
high-order reads, SI Fig. S2, and pattern insets in Figs. 2 and 4 require these
arrays or a separately validated new campaign. See the
[archive guide](paper/README.md) and [reproduction guide](reproduce/README.md).

For a CP submission, associate the manuscript with a fixed software version and
source revision, and archive the exact code, inputs and relevant result data.
The local package version is currently `0.1.0`; it is not a published release.

## Numerical interpretation

- A **descent order** controls most gradients; a **gate order** provides a common
  basis for selecting binary checkpoints. The final **evaluation ladder** checks
  the delivered structure at increasing orders.
- The interface's term **certification** means higher-order numerical evaluation.
  It is not a rigorous error bound. Inspect the order-convergence curve and
  report the actual evaluation orders and remaining changes.
- Energy conservation and agreement between two solvers are useful checks, but
  neither alone proves truncation convergence or gradient correctness.
- Gaussian filtering controls smoothness before projection. Check minimum
  linewidth and spacing on the final binary pattern separately. CD-bias audits
  are geometric perturbation tests, not physical lithography simulations.
- Gradient guards are numerical heuristics. Their thresholds were calibrated on
  particular research cases; validate their behavior on a new problem or precision.
- Cross-machine optimization trajectories need not be bitwise identical. Report
  numerical tolerances, seeds and statistical comparisons where appropriate.

`TFMM_EIG_DEVICE=cpu` is the inherited default: eigendecomposition is performed
on the CPU even when other operations use CUDA. The original environment used
this setting to avoid observed host-memory growth. `TFMM_EIG_DEVICE=native`
uses the tensor's device; validate it on your stack before benchmarking.
`FMM_THREADS` sets the backend's PyTorch thread count (default: 3). Additional
numerical switches are documented in the [algorithm notes](docs/algorithm.md).

## Repository layout

```text
trustworthy_fmm/       RCWA solvers, Fourier operators and physical utilities
workbench/            Shared workflow, command-line API and local web application
examples/             Small, executable JSON configurations
reproduce/            Experiment manifests, analytic reference and analysis entry points
paper/                Checksummed research-data bundles and manuscript evidence inventory
tests/                Numerical, policy, batch and web regression tests
requirements/         Pinned direct-dependency CPU validation profile
docs/                 Algorithm, browser and validation documentation
run.py                Backward-compatible browser launcher
selfcheck.py          Backward-compatible self-check launcher
pyproject.toml        Package metadata, dependencies and console entry point
```

Build local wheel and source archives with `python -m pip install -e ".[dev]"`
and `python -m build`. Source archives include the examples, tests and reproduction
framework; wheels include the runtime code and browser assets.

## License and citation status

The project's original code and associated documentation are licensed under the
[MIT License](LICENSE). Bundled and external third-party components retain their
own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).

The repository is currently private. Applying an MIT license does not change
its visibility or create a public software release. Software citation metadata
is provided in [CITATION.cff](CITATION.cff); cite the exact commit used.

Manuscript bibliographic information and an archival software DOI are not yet
available. Add the verified citation metadata when they are finalized; no
publication or DOI is implied by this development version.
