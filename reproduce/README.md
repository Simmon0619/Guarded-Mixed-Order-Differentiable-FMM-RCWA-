# Reproducing experiments

This directory provides executable infrastructure for a Computational Physics
(CP) manuscript. The small smoke cases test installation and numerical behavior.
Historical manuscript records and scripts are now bundled in `../paper/`.

For the historical data, use `python reproduce/paper.py verify` and
`python reproduce/paper.py analyze --output outputs/paper-analysis --figures`.
These commands verify the archive and reanalyse existing records; they do not
rerun the GPU optimization campaign. See [the archive guide](../paper/README.md)
for figure coverage and missing original arrays. `paper-manifest.json` records
the recovered experiment mapping, original commands, seeds and configuration
metadata. Its generic `experiments` list remains empty because historical
scripts use a different interface; use `paper.py` for this archive.

## Run the supplied suite

Install the project from the repository root, then run:

```bash
python -m pip install -e ".[test,plots]"
python reproduce/run_experiments.py --manifest reproduce/smoke-manifest.json --output outputs/smoke
python reproduce/make_figures.py --input outputs/smoke --plots
```

`outputs/smoke` must not already exist. Use a new directory for another run.
`--only slab` runs one selected case; `--list` lists the manifest without running
it. `--fields` adds the browser's field maps, which are omitted by default.
The same runner is available as `python -m workbench.reproduction` after a wheel
installation. Copy the experiment inputs with the scripts when working outside
the source checkout.

The slab reference is derived from the analytic Airy formula, not from a saved
solver output. Its JSON tolerance accounts for the legacy six-decimal display
rounding. Numerical tests compare unrounded tensors at tighter tolerances.
The six-step optimization only exercises the pipeline: its small truncation
orders are deliberately unsuitable for a convergence or performance claim.

## Connect the manuscript to executable cases

To port additional historical cases to the shared batch interface, populate
the `experiments` list with validated schema-v1 configurations. An empty manifest raises an error rather than reporting successful
reproduction. Each entry has this form:

```json
{
  "id": "figure-2-case-a",
  "config": "configs/figure-2-case-a.json",
  "purpose": "Describe the physical case and comparison being reproduced.",
  "paper_items": ["Figure 2a"],
  "reference": "reference_results/figure-2-case-a.json"
}
```

The example above describes the schema; those two files are not supplied.
Paths are relative to the manifest, independent of the shell's working directory.
`reference` is optional. Reference files contain a nonempty `checks` list, with
each entry specifying a result `path`, `value`, `atol`, and optionally `rtol`.
The runner accepts a value when `abs(actual - value) <= atol + rtol*abs(value)`.
Do not infer expected values or tolerances from the result being checked.

For each manuscript figure or table, record the physics, material model, grid,
orders, all seeds, beta schedule, guard settings, optimization budget, precision,
hardware and external solver versions. Separate exact forward checks from
statistical optimization claims. Equal-wall-clock experiments require the actual
elapsed time and completed work; results may differ between machines.

Include raw results or a versioned archive for historical aggregate claims.
Document expected runtime and peak memory after measuring the full cases. The
repository's historical GPU timings are not estimates for these CPU smoke cases.

## Outputs and verification

Each experiment creates a subdirectory containing:

| File | Contents |
|---|---|
| `input.json` | Original schema-v1 experiment input |
| `resolved.json` | Effective physical settings, orders, protocol settings and execution engine |
| `result.json` | The browser-compatible result, including its existing rounding |
| `trajectories.json` | Optimization trajectories, checkpoints and chain records |
| `arrays.npz` | Unrounded latent/density arrays and reflected/transmitted order powers |
| `manifest.json` | Status, runtime, dependency versions, numerical flags and source hashes |
| `checksums.json` | SHA-256 hashes of the completed run's files |

The array evaluation order is recorded in `manifest.json`: the requested solve
order for a forward run, and the highest evaluation order for optimization.
`density` may be continuous for a uniform-fill solve. Load arrays with
`numpy.load(path, allow_pickle=False)`. The archive contains the delivered state;
it is not a full Adam-state checkpoint for resuming an interrupted optimization.

The suite also saves an experiment index, the input manifest, numerical reference
copies where applicable, and `summary.csv`. Summarization verifies run checksums
before reading results. `--plots` writes convergence PNG and PDF files and needs
the `plots` extra. It does not generate manuscript-specific statistical figures;
add their analysis scripts when the corresponding datasets are supplied.

Failures produce a nonzero exit status. A run that fails after startup records
its failure in the run manifest; a failed reference marks the suite as failed.
Existing output directories are never overwritten. Non-finite diagnostic values
are serialized as JSON `null`; the raw numerical arrays retain their NumPy types.
No network connection is needed after dependencies and inputs are available.
