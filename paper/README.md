# Manuscript evidence archive

This archive was curated on 21 October 2026 from the supplied
`SY-CMT-paper-main` project. It accompanies the Computational Physics manuscript
and the installable workbench in the repository root. The research files are
preserved byte for byte; the original directories have not been modified.

## What is included

`bundles/` contains independent ZIP archives with the original relative paths.
`inventory.json` records each file's original path, size and SHA-256 hash, plus
the bundle hashes. Extract them with the command below to recover the original
research layout. The ZIP format keeps recorded data and historical scripts
together without placing a second solver package on the installation path.

| Manuscript material | Source experiments | Available evidence |
|---|---|---|
| Fig. 1 schematic and spectral illustrations | F1 | Scripts, numeric checks and source panels; final assembled Fig. 1 was not found |
| Fig. 2, conventional workflow and frozen-state diagnostics | F31, S7, S8 | JSON trajectories, reference reads, diagnostic scans, plotting scripts and saved panels |
| Fig. 3, guarded mixed-order method | F33, S10 | Protocol scripts, guard calibration, checkpoints and three independently redrawable panels |
| Fig. 4 and Sec. 3.3, paired performance and refinement | F33, S7 | 72 paired runs (24/24/12/12), reference-order reads, timing records, analysis and refinement records |
| Fig. 5, browser demonstration | F5 | Figure assets and screenshot scripts/metadata |
| Supplementary Figs. S1–S5 and supporting tables | FS, S4–S10, A1/A2, E16–E28 | Recorded validation/calibration data, dependencies, scripts and existing figures |

The E-series contains inherited supporting experiments, not additional evidence
for the final guarded protocol. `legacy-src.zip` preserves their historical
solver dependencies. The active installable solver remains in the repository
root. Manuscript drafts, internal writing reviews, assistant memory, unrelated
H/N/P research projects, external literature and local environments are omitted.
The manuscript directory identifies a later integrated manuscript outside the
supplied project; this archive does not designate the older drafts as final.

## Verified, inexpensive reproduction

From an installed source checkout:

```bash
python reproduce/paper.py verify
python reproduce/paper.py analyze --output outputs/paper-analysis --figures
```

The second command extracts into a **new** output directory, executes the
original F33 statistical analysis, compares every statistic with the archived
reference (absolute/relative tolerance `1e-12`), writes SI tables, and redraws
the three Fig. 3 panels and supplementary Figs. S1/S3/S4/S5. It performs **zero physics
solves**. Install the `plots` extra for figures. It never overwrites the bundled
data or an existing output directory. Fonts can change the rendered appearance
across operating systems; the numeric comparison is independent of fonts.

Outputs include `f33_analysis.json`, `f33-tables.md`, `analysis-report.json`,
command logs, and a separate `workspace/` holding regenerated figures. See
`reproduce/paper-manifest.json` for the experiment and figure mapping.

## Remaining gaps in full numerical reproduction

SI Fig. S2 also requires the missing S8 frozen state.

The supplied project contains no original `.npy`/`.npz` design arrays for these
experiments. In particular, the F31/F33 delivered masks, S7/F33 per-run binary
masks and latent fields, and the S8 frozen-state arrays are missing. Their hashes
and efficiency records do not reconstruct those arrays. Original plotting
commands requiring pattern insets and fresh high-order reads therefore cannot
all run from this archive alone.

`missing-array-references.json` lists absent array paths mentioned in archived
JSON. This broad inventory includes historical outputs and supporting studies;
it is not a count of files needed for one paper figure, nor an exhaustive list
of script-only dependencies. Saved PNG/SVG/PDF figures are retained as historical
assets, not presented as proof of fresh numerical reproduction.

For an isolated workspace without starting expensive calculations:

```bash
python reproduce/paper.py prepare --output outputs/paper-workspace
```

In the extracted `stage_e1_metasurface_platform/experiments/` tree, the original
entry points include `S7_headtohead/scripts/s7_headtohead.py` (conventional runs),
`F33_section33_results/scripts/f33_tickets.py` (final method), `f33_reads.py`
(reference evaluations), `f33_traj.py` (representative trajectory) and
`f33_refine.py` (refinement). Full commands and seeds are in the manifest.
These are historical scripts with resumable result files. For a fresh campaign,
use a separate empty results area: an existing ticket JSON may cause a script to
skip a run even when its corresponding design file is missing. Restoring the
original arrays is preferable to silently mixing old records with new arrays.

The recorded environment used Windows 10, an RTX 3060 12 GB, PyTorch
1.13.0+cu117, complex64 solves, CPU eigendecomposition and three PyTorch threads.
The original full dependency lockfile was not found. New hardware, PyTorch,
BLAS or precision may change optimization trajectories. The recorded timings
are historical measurements, not performance promises for the current machine.

Before a complete numerical reproduction claim, recover the referenced arrays
or run a separately versioned campaign, validate its results, and associate the
final manuscript with a fixed commit and an archival DOI. No DOI or public
release is claimed here.
