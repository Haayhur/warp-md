# warp-md

CPU-first MVP scaffold for MD trajectory IO + analysis, with optional CUDA path.

## Status

- Rust workspace scaffolded with core, IO, engine, GPU stub, kernels placeholder, and PyO3 bindings.
- Implemented: PDB/GRO topology readers, DCD reader (32/64-bit markers, best-effort unit cell), XTC reader (xdrfile), selection engine, CPU Rg/RMSD/MSD/RDF, polymer analyses (end-to-end, contour length, chain Rg, bond length/angle distributions, persistence length), and analysis suite (rotational ACF, conductivity, dielectric, dipole alignment, ion-pair correlation, structure factor, water grid, equipartition temperature, H-bond counts).
- Correlator plans use streaming multi-tau by default (bounded memory); ring-buffer and FFT modes are available for short-time exactness or small streams.
- Feature store support: chunked binary + JSON index for offline/long-run feature capture (`feature_store` module).
- CUDA: optional GPU path (per-frame feature extraction and kernels) behind `--features cuda` and requires CUDA driver + nvrtc at runtime. Persistence length and parts of the analysis suite remain CPU reductions for now.

## Build and test

Rust:

```bash
cargo test
```

Pack module (Rust):

```bash
cargo test -p warp-pack
```

Pack module (CLI):

```bash
warp-pack --config pack.json --output packed.pdb --format pdb
```

CUDA build (requires CUDA runtime libraries present):

```bash
cargo test -p traj-engine --features cuda
```

If the CUDA toolkit isn't detected, set `CUDA_HOME` (or `CUDA_PATH`) so `cudarc` can resolve the version.

Python (requires maturin):

```bash
maturin develop
python -c "import warp_md; print(warp_md.System)"
```

Pack module (Python):

```bash
python3 -m pytest python/warp_md/tests/test_pack.py
```

Pack module (Python usage):

```python
from warp_md.pack import Box, Structure, PackConfig, water_pdb
from warp_md.pack.runner import run
from warp_md.pack.export import export

cfg = PackConfig(
    structures=[Structure(water_pdb("tip3p"), count=100)],
    box=Box((30.0, 30.0, 30.0)),
    restart_to="packed.restart",
    relax_steps=10,
    relax_step=0.5,
)
result = run(cfg)
export(result, "pdb", "packed.pdb")
```

## Python usage (CPU)

```python
from warp_md import System, Trajectory, RgPlan, EndToEndPlan, MsdPlan

system = System.from_pdb("example.pdb")
selection = system.select("name CA")
traj = Trajectory.open_dcd("traj.dcd", system)
plan = RgPlan(selection, mass_weighted=False)
rg = plan.run(traj, system, device="auto")
poly = EndToEndPlan(selection)
end_to_end = poly.run(traj, system, device="auto")
msd = MsdPlan(selection, group_by="resid")
time, data = msd.run(traj, system)
```

## CLI (Python)

Single-command CLI for agents, plus config-driven runs.

```bash
# one-command analysis
warp-md rg --topology top.pdb --traj traj.xtc --selection "protein"
warp-md rdf --topology top.pdb --traj traj.xtc --sel-a "resname SOL and name OW" --sel-b "resname SOL and name OW" --bins 200 --r-max 10

# config runner
warp-md example > config.json
warp-md run config.json
warp-md list-plans
```

By default, the CLI prints a JSON summary to stdout. Use `--no-summary` to disable.
YAML configs require `pip install warp-md[cli]`.

## Documentation

- Getting started and full walkthrough: `docs/tutorial.md`
- Full API + module reference: `docs/reference.md`
- Validation + accuracy comparison: `docs/validation.md`
- Docs index: `docs/README.md`

## Publication Benchmarks

Use the publication workflow entrypoint to produce report-ready benchmark
artifacts (manifest plan, standardized CSV tables, summary plots, and coverage
inventory) from one command:

```bash
.agent/bench-venv/bin/python scripts/bench/run_publication_bundle.py
```

Useful options:

```bash
# also execute manifest benchmark jobs (heavy)
.agent/bench-venv/bin/python scripts/bench/run_publication_bundle.py --execute-manifest

# include full-trajectory triplet figures
.agent/bench-venv/bin/python scripts/bench/run_publication_bundle.py --include-fulltraj

# plan commands only (no execution)
.agent/bench-venv/bin/python scripts/bench/run_publication_bundle.py --dry-run
```

## Python builder helpers

```python
from warp_md import charges_from_table, group_types_from_selections

charges = charges_from_table(system, "charges.tsv")
group_types = group_types_from_selections(
    system,
    system.select("resname BMIM or resname BF4"),
    "resid",
    ["resname BMIM", "resname BF4"],
)
```
