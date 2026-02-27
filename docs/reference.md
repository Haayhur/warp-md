# warp-md reference

This reference describes modules, APIs, parameters, and outputs.
All units are Angstrom unless noted.

## Modules (Rust crates)

- `traj-core`: system topology, selection engine, boxes, errors
- `traj-io`: PDB/GRO topology, DCD/XTC trajectories
- `traj-engine`: plans, executor, CPU fallback, device dispatch
- `traj-gpu`: CUDA context + buffers (optional feature)
- `traj-kernels`: CUDA kernels (compiled with nvrtc)
- `traj-py`: Python bindings (PyO3)
- `warp-pack`: CPU packing engine + IO for initial coordinates

## Core concepts

### System

- Holds topology + optional reference coordinates
- Python: `System.from_pdb(path)` or `System.from_gro(path)`
- Methods:
  - `system.select(expr)` returns a `Selection`
  - `system.n_atoms()` returns atom count
  - `system.atom_table()` returns a dict with `name`, `resname`, `resid`, `chain_id`, `mass`

### Selection language

Predicates:
- `name`, `resname`, `resid`, `chain`, `protein`, `backbone`

Boolean ops:
- `and`, `or`, `not`, parentheses

Examples:
- `resname SOL and name OW`
- `resid 10-50 and chain A`
- `not protein`

### Grouping (group_by)

Plans that aggregate per-group accept `group_by`:
- `"resid"` (group atoms by residue id)
- `"chain"` (group by chain id)
- `"resid_chain"` (unique resid within each chain)

Some plans accept `group_types`:
- list length = number of groups
- values are type ids (0..n_types-1)
- used to split outputs by species (with an extra total column)

### Trajectory

- `Trajectory.open_dcd(path, system, length_scale=1.0)`
- `Trajectory.open_xtc(path, system)`
- The atom count must match the `System`.
- DCD `length_scale` lets you convert nm to Angstrom (use 10.0 if needed).

### Feature store (chunked binary + JSON)

The `traj-engine` feature store can persist per-frame features for long runs:

- `FeatureStoreWriter::new("path_prefix", schema)` writes `path_prefix.bin` + `path_prefix.json`
- `FeatureStoreReader::open("path_prefix")` reads the index + data

This is useful for offline FFT analysis or inspection without re-reading trajectories.

### Device selection

All plan `run(...)` functions accept `device`:

- `"auto"`: CUDA if available, else CPU
- `"cpu"`
- `"cuda"` or `"cuda:0"`

## CLI (agent-friendly)

Single-command CLI for agents plus a config runner.

Tool definition (copy/paste into prompts):

```
TOOL warp-md
DESCRIPTION: Run a single analysis from CLI. Writes an output file and prints a JSON envelope by default.
COMMAND: warp-md <analysis> --topology PATH --traj PATH [analysis flags]
ANALYSES: rg, rmsd, msd, rdf, rotacf, conductivity, dielectric, dipole-alignment, ion-pair-correlation,
          structure-factor, water-count, equipartition, hbond, end-to-end, contour-length, chain-rg,
          bond-length-distribution, bond-angle-distribution, persistence-length
OUTPUT: .npz by default; use --out for .npy/.csv/.json; single-analysis emits JSON envelope by default
```

Examples:

```bash
warp-md rg --topology top.pdb --traj traj.xtc --selection "protein"
warp-md rdf --topology top.pdb --traj traj.xtc --sel-a "resname SOL and name OW" --sel-b "resname SOL and name OW" --bins 200 --r-max 10
warp-md msd --topology top.pdb --traj traj.xtc --selection "resname SOL" --group-by resid --axis 0,0,1
```

Charges (conductivity/dielectric/dipole):

```bash
warp-md conductivity --charges '[1.0,-1.0,1.0]' ...
warp-md conductivity --charges table:charges.csv ...
warp-md conductivity --charges 'selections:[{\"selection\":\"resname NA\",\"charge\":1.0}]' ...
```

Group types:

```bash
warp-md msd --group-types '[0,1,1,0]' ...
warp-md msd --group-types 'selections:resname NA,resname CL' ...
```

Config runner:

```bash
warp-md list-plans --format json
warp-md list-plans --format json --details
warp-md water-models --format json
warp-md run config.json
warp-md run config.json --stream ndjson
warp-md example > config.json
warp-md schema --format json
warp-md schema --kind result --format json
warp-md schema --kind event --format json
```

`warp-md run` emits a JSON contract with stable `status` + `exit_code`.
Success and failure are both machine-readable; with `--stream ndjson`, events are emitted line-by-line and the terminal event contains `final_envelope`.
Single-analysis commands emit the same envelope contract with `analysis_count=1`.

### Time decimation and binning

Some plans accept optional time controls:
- `frame_decimation=(start, stride)` to skip origin frames
- `dt_decimation=(cut1, stride1, cut2, stride2)` to thin long time lags
- `time_binning=(eps_num, eps_add)` to stabilize dt binning from floating errors

### Lag modes (auto / multi-tau / ring / FFT)

Correlator-based plans support `lag_mode`:

- `auto` (default): if the full series fits the host memory budget and dt is uniform, use FFT; otherwise use multi-tau.
- `multi_tau`: log-spaced lags, bounded memory, streaming-friendly (default for long trajectories).
- `ring`: exact up to `max_lag`, linear memory/runtime (best for short-time exactness).
- `fft`: exact all lags (requires uniform dt and holding the series in memory).

Tuning knobs:
- `max_lag`: only meaningful for `ring` (default 100k, capped by `memory_budget_bytes`).
- `memory_budget_bytes`: used to cap `max_lag` and decide `auto` FFT viability.
- `multi_tau_m`, `multi_tau_levels`: multi-tau resolution (default m=16, levels=20).

### CUDA coverage (summary)

GPU is optional and used for per-frame and time-lag kernels when available.

GPU-accelerated paths include:
- Rg, RMSD, MSD, RDF
- Polymer: end-to-end, contour length, chain Rg, bond length and angle histograms
- RotAcf (orientation extraction)
- Conductivity (group COM)
- Dielectric + Dipole alignment (group dipoles)
- Ion-pair correlation (group COM)
- Structure factor (RDF kernel)
- Water occupancy grid
- Equipartition (group kinetic energy)
- H-bond counts (distance + angle)

## Python API reference

### Packing (warp-pack)

Python entry points:
- `warp_md.pack.config.PackConfig`
- `warp_md.pack.runner.run`
- `warp_md.pack.export.export`

Key config fields:
- `structures`: list of `Structure(path, count, ...)`
- `box`: `Box((Lx, Ly, Lz))`
- `min_distance`: hard minimum distance
- `use_short_tol`, `short_tol_dist`, `short_tol_scale`: soft penalty region
- `writeout`, `writebad`: periodic snapshots and failure dump
- `restart_from`, `restart_to`: read/write Packmol-style restart files
- `relax_steps`, `relax_step`: post-pack overlap relaxation
- `add_box_sides`, `add_amber_ter`, `amber_ter_preserve`

Input formats:
- `pdb`, `xyz`, `mol2`, `tinker/txyz`, `amber/inpcrd/rst/rst7`

Output formats:
- `pdb`, `xyz`, `pdbx/mmcif`, `gro`, `lammps-data`, `mol2`, `crd`

### RgPlan

```python
RgPlan(selection, mass_weighted=False)
rg = plan.run(traj, system, chunk_frames=None, device="auto")
```

Output: `rg` is a 1D array, length = frames.

### RmsdPlan

```python
RmsdPlan(selection, reference="topology", align=True)
values = plan.run(traj, system)
```

Output: 1D array, length = frames.

### MsdPlan (multi-origin)

```python
MsdPlan(
    selection,
    group_by="resid",
    axis=None,
    length_scale=None,
    frame_decimation=None,
    dt_decimation=None,
    time_binning=None,
    group_types=None,
    lag_mode=None,
    max_lag=None,
    memory_budget_bytes=None,
    multi_tau_m=None,
    multi_tau_levels=None,
)

time, data = plan.run(traj, system)
```

Output:
- `time`: 1D array (dt bins)
- `data`: 2D array with shape `(rows, cols)`
- `cols = components * (n_types + 1)`
- `components = 5` if axis is provided else `4`
- component order: `x`, `y`, `z`, `axis` (optional), `total`
- for each component, columns are: type0, type1, ..., typeN, total
- `lag_mode="multi_tau"` returns log-spaced lags; `ring` and `fft` return contiguous lags.

### RdfPlan

```python
RdfPlan(sel_a, sel_b, bins, r_max, pbc="orthorhombic")

r, g, counts = plan.run(traj, system)
```

Output:
- `r`: bin centers
- `g`: g(r)
- `counts`: raw pair counts

### Polymer plans

```python
EndToEndPlan(selection)
ContourLengthPlan(selection)
ChainRgPlan(selection)
BondLengthDistributionPlan(selection, bins, r_max)
BondAngleDistributionPlan(selection, bins)
PersistenceLengthPlan(selection, max_bonds)
```

Outputs:
- `EndToEndPlan`, `ContourLengthPlan`, `ChainRgPlan`: 2D arrays (frames x chains)
- `BondLengthDistributionPlan`, `BondAngleDistributionPlan`: `(centers, counts)` histogram
- `PersistenceLengthPlan`: dict with keys `bond_autocorrelation`, `lb`, `lp`, `fit`, `kuhn_length`

### RotAcfPlan

```python
RotAcfPlan(
    selection,
    group_by="resid",
    orientation=[0, 1] or [0, 1, 2],
    p2_legendre=True,
    length_scale=None,
    frame_decimation=None,
    dt_decimation=None,
    time_binning=None,
    group_types=None,
    lag_mode=None,
    max_lag=None,
    memory_budget_bytes=None,
    multi_tau_m=None,
    multi_tau_levels=None,
)

time, data = plan.run(traj, system)
```

Output:
- `time`: 1D array
- `data`: 2D array, `cols = 2 * (n_types + 1)`
- first block is P1 for each type + total, second block is P2
- `time[0]` is always 0.0 with P1/P2 = 1.0

### ConductivityPlan

```python
ConductivityPlan(selection, group_by="resid", charges=charges, temperature=300.0,
                 transference=False, length_scale=None,
                 frame_decimation=None, dt_decimation=None, time_binning=None,
                 group_types=None,
                 lag_mode=None, max_lag=None, memory_budget_bytes=None,
                 multi_tau_m=None, multi_tau_levels=None)

time, data = plan.run(traj, system)
```

Output:
- If `transference=False`: `cols = 1`, total conductivity time series
- If `transference=True`: `cols = n_types * n_types + 1`
  - first `n_types * n_types` entries are a row-major matrix
  - last column is total

### DielectricPlan

```python
DielectricPlan(selection, group_by="resid", charges=charges, length_scale=None, group_types=None)

out = plan.run(traj, system)
```

Output dict keys:
- `time`, `rot_sq`, `trans_sq`, `rot_trans`, `dielectric_rot`, `dielectric_total`, `mu_avg`

### DipoleAlignmentPlan

```python
DipoleAlignmentPlan(selection, group_by="resid", charges=charges, length_scale=None, group_types=None)

time, data = plan.run(traj, system)
```

Output:
- `cols = 6 * (n_types + 1)`
- first half: cos(x), cos(y), cos(z) for each type + total
- second half: cos^2(x), cos^2(y), cos^2(z) for each type + total

### IonPairCorrelationPlan

```python
IonPairCorrelationPlan(
    selection,
    rclust_cat,
    rclust_ani,
    group_by="resid",
    cation_type=0,
    anion_type=1,
    max_cluster=10,
    length_scale=None,
    group_types=None,
    lag_mode=None,
    max_lag=None,
    memory_budget_bytes=None,
    multi_tau_m=None,
    multi_tau_levels=None,
)

time, data = plan.run(traj, system)
```

Output:
- `cols = 6` with order:
  - `ip_total`, `ip_cation`, `ip_anion`, `cp_total`, `cp_cation`, `cp_anion`
- `time[0]` is always 0.0 with all values = 1.0

### StructureFactorPlan

```python
StructureFactorPlan(selection, r_bins, r_max, q_bins, q_max, pbc="orthorhombic")

r, g, q, s = plan.run(traj, system)
```

Output:
- `r`, `g` (g(r))
- `q`, `s` (S(q))

### WaterCountPlan

```python
WaterCountPlan(water_selection, center_selection, box_unit, region_size, shift=None, length_scale=None)

out = plan.run(traj, system)
```

Output dict keys:
- `dims`: [nx, ny, nz]
- `mean`, `std`, `first`, `last`, `min`, `max`

### EquipartitionPlan

```python
EquipartitionPlan(selection, group_by="resid", velocity_scale=None, length_scale=None, group_types=None)

time, data = plan.run(traj, system)
```

Output:
- `cols = n_types + 1` (per-type + total temperature)

### HbondPlan

```python
HbondPlan(donors, acceptors, dist_cutoff=3.5, hydrogens=None, angle_cutoff=None)

time, data = plan.run(traj, system)
```

Output:
- `cols = 1`, hydrogen-bond count per time point
- If `angle` is provided, hydrogens must be supplied and matched 1:1 with donors

## Builder helpers

```python
from warp_md import charges_from_selections, charges_from_table, group_types_from_selections
```

- `charges_from_selections(system, entries, default=0.0)`
- `charges_from_table(system, path, delimiter=None, default=0.0)`
  - requires columns `resname`, `name` (or `atom`), `charge`
- `group_types_from_selections(system, selection, group_by, type_selections)`

## Python analysis wrappers (selected)

These wrappers live under `python/warp_md/analysis/` and expose convenience APIs on top of plans and CPU fallbacks.

- `analysis.nmr.ired_vector_and_matrix(..., return_corr=True, corr_mode="tensor"|"timecorr")`
- `analysis.nmr.nh_order_parameters(..., method="tensor"|"timecorr_fit")`
- `analysis.nmr.jcoupling(..., return_dihedral=False)`
- NMR supported-mode contract:
  - `ired_vector_and_matrix`: `order in {1,2}`; `corr_mode in {"tensor","timecorr"}`; `pbc in {"none","orthorhombic"}`.
  - `nh_order_parameters(method="tensor")`: supports `order in {1,2}`.
  - `nh_order_parameters(method="timecorr_fit")`: supports `order=2` only, and requires `tstep > 0`, `tcorr > 0` when provided.
  - Unsupported combinations fail with explicit `ValueError` (no silent fallback).
- `analysis.gist.gist(..., energy_method="direct"|"pme"|"none", pme_totals_source="openmm"|"native"|"direct_approx")`
- GIST supported-mode contract:
  - `energy_method` supports only `"direct"`, `"pme"`, or `"none"`.
  - `pme_totals_source` supports `"openmm"`, `"native"`, or `"direct_approx"` when `energy_method="pme"`.
  - `"native"` uses Rust/CUDA frame totals from the GIST direct path (OpenMM optional).
  - `"direct_approx"` reuses Rust direct frame totals for approximate scaling.
  - Unsupported combinations fail with explicit `ValueError`/`RuntimeError`.
- `analysis.rotdif.rotdif(..., return_fit=False, fit_component="p2", fit_window=None)`
- Rotdif supported-mode contract:
  - `orientation` is required and must have length 2 (bond vector) or 3 (plane normal).
  - `fit_component` supports `"p1"` or `"p2"` only when `return_fit=True`.
  - `fit_window`, when set, must be a finite 2-item tuple/list.
  - `p2_legendre` is a strict boolean.
  - Unsupported combinations fail with explicit `ValueError`.
- `analysis.diffusion.tordiff(..., return_transitions=False, transition_lag=1)` (`toroidal_diffusion` aliases `tordiff`)
- `analysis.surf.surf(..., algorithm="sasa"|"bbox"|"auto", probe_radius=1.4, n_sphere_points=64, radii=None)`
- `analysis.surf.molsurf(..., algorithm="sasa"|"bbox"|"auto", probe_radius=0.0, n_sphere_points=64, radii=None)`
- Surf/Molsurf supported-mode contract:
  - `algorithm` supports `"sasa"`, `"bbox"`, or `"auto"` (case-insensitive).
  - `probe_radius` must be finite and `>= 0`.
  - `n_sphere_points` must be a positive integer.
  - `radii`, when provided, must contain finite values `> 0`.
- `analysis.pucker.pucker(..., metric="amplitude"|"max_radius", return_phase=False)`
- Pucker supported-mode contract:
  - `metric` supports `"amplitude"` or `"max_radius"` (case-insensitive).
  - `return_phase` is a strict boolean.
  - Legacy bindings without `(metric, return_phase)` constructor support are rejected with explicit `RuntimeError`.
- `analysis.multipucker.multipucker(..., mode="histogram"|"legacy", range_max=None, normalize=True)`
- Multipucker supported-mode contract:
  - `bins` must be a positive integer.
  - `mode` supports `"histogram"` or `"legacy"` (case-insensitive).
  - `normalize` is a strict boolean.
  - `range_max`, when provided, must be finite and `> 0`.
- `analysis.symmrmsd.symmrmsd(..., remap=False, symmetry_groups=None, max_permutations=4096)`
- `analysis.check_chirality.check_chirality(..., planar_tolerance=1e-8, return_labels=False)`
- Check chirality supported-mode contract:
  - Each `group` must contain exactly 4 entries `(A, B, C, D)` for signed-volume evaluation.
  - `mass_weighted` and `return_labels` are strict booleans.
  - `planar_tolerance` must be finite and `>= 0`.
  - Labels use signed-volume thresholding: `+1` above tolerance, `-1` below `-tolerance`, `0` otherwise.
- `analysis.xtalsymm.xtalsymm(..., symmetry_ops=None)`
- Xtalsymm supported-mode contract:
  - `repeats` must be a 3-item tuple/list of positive integers.
  - `symmetry_ops`, when provided, must be non-empty and each op must be:
    - shape `(3,3)`, `(3,4)`, `(4,4)`, or flat length `9/12/16`
    - finite numeric values
    - affine for 4x4 form (last row `[0,0,0,1]`)
  - Legacy bindings without `symmetry_ops` constructor support are rejected with explicit `RuntimeError`.
- `analysis.dihedral_tools.rotate_dihedral(..., frame_indices=None)`
- `analysis.dihedral_tools.set_dihedral(..., frame_indices=None)`
- Dihedral tools supported-mode contract:
  - `atoms` must contain exactly 4 entries.
  - `pbc` for `set_dihedral` supports only `"none"` or `"orthorhombic"`.
  - `mass`, `degrees`, and `range360` are strict booleans.
  - Unsupported combinations fail with explicit `ValueError` (no implicit fallback).

## Notes on units

- Coordinates and lengths are Angstrom internally.
- GRO/XTC are converted from nm to Angstrom.
- DCD is assumed Angstrom; use `length_scale` if needed.
