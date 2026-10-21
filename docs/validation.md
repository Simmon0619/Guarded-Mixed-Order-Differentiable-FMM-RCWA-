# Validation and supported environments

The test suite is intended to catch regressions in the software used for a CP
manuscript. Passing it is not evidence that every design is converged, that every
gradient near a degenerate eigenproblem is accurate, or that all historical
performance claims have been reproduced.

## Checks

- Homogeneous-slab reflection and transmission against an independent transfer
  matrix, at normal and oblique incidence and both linear polarizations.
- Energy balance for a lossless patterned layer and the zero-thickness limit.
- An automatic-differentiation directional derivative against central finite
  differences at three step sizes, on a smooth non-normal-incidence case.
- Agreement between the memory-efficient and standard calculation paths, and
  between loop-based and vectorized convolution assembly, including gradients.
- Guard behavior under injected stable/unstable gradients and failed evaluations,
  bounded CPU fallback, mixed-order cadence and checkpoint selection by gate.
- A small end-to-end optimization and forward solve, portable CLI invocation,
  input validation, metadata/array exports and output overwrite protection.
- Manifest execution, analytic reference checks, result checksum verification,
  and refusal to treat an empty paper manifest as completed reproduction.
- Local HTTP resources, environment reporting, run-history listing and job
  submission. The Plotly bundle is included in the installed package.
- An optional CUDA value/gradient comparison, explicitly skipped without CUDA.

Run `python -m pytest -q` after `python -m pip install -e ".[test]"`. The original
machine-level checks remain available through `python selfcheck.py` and
`fmm-workbench selfcheck`; `--bench` adds larger timing cases.

## Environment records

Local verification on 2026-10-21 used Windows (build 26100), Python 3.12.14,
PyTorch 2.5.1+cpu, NumPy 1.26.4 and SciPy 1.13.1:

- 35 pytest cases passed; one CUDA case was skipped because CUDA was unavailable.
- The original numerical self-check passed its CPU checks. TORCWA and CUDA
  comparisons were skipped; Lumerical was only imported, not executed.
- Both supplied smoke experiments completed; CSV summaries and PNG/PDF plots
  were generated successfully, including the independent slab reference check.
- The slab and six-step optimization were also run against a saved copy of the
  pre-packaging source in separate processes. Their numerical result records,
  patterns, field maps and algorithm decisions matched exactly after timing
  fields were excluded. This comparison covers those cases, not all inputs.
- Wheel and source archives built successfully. The installed wheel was checked
  outside the source directory for package imports, a forward solve, browser
  HTML/Plotly resources and the installed-package history location. `pip check`
  reported no dependency conflicts.

The solver library, protocol parameters, polygon exporter and browser HTML/JS
were checked against their original hashes. Numerical formulas and optimizer
update rules were not changed by this packaging work.

`requirements/cpu-reference.txt` pins direct dependencies for a Python 3.12 CPU
validation environment. It is distinct from the original Windows/torch 1.13/
CUDA 11.7 research environment. A successful CPU test does not validate a CUDA
build or reproduce its stochastic optimization trajectories.

Every batch run records its actual dependency versions, device, precision,
thread count, eigendecomposition device, relevant solver flags and source-file
hashes. The declared lower bounds in `pyproject.toml` are inherited from the
original project; they are not a claim that every combination above those bounds
has been tested. Use the pinned profile for repeatable CPU checks.

Before submitting the paper, run the full manuscript suite on the intended
reference GPU, archive the complete environment and confirm the reported
truncation convergence and multi-seed statistics. Mark optional comparisons
that were skipped rather than counting them as passes.

## Manuscript archive verification (2026-10-21)

- Curated 643 original research files from the supplied parent project. Bundle
  and member SHA-256 checks pass; source files are preserved byte for byte.
- Executed the original F33 analysis in a separate extracted workspace. All
  statistics match the recorded reference with `atol=rtol=1e-12`; the four
  channels contain 24/24/12/12 paired runs, with 68 positive paired differences.
- Regenerated the F33 SI tables, three Fig. 3 panels and SI Figs. S1/S3/S4/S5
  from recorded data. This validation involved no new physics solves.
- Original design arrays are absent. SI Fig. S2, delivered-pattern insets and
  fresh reference evaluations remain incomplete. See `paper/README.md` and
  the machine-readable missing-array inventory; saved figures alone do not
  establish numerical reproduction.
