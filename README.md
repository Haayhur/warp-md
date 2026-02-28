# warp-md

Rust/CUDA/Python toolkit for molecular trajectory IO, analysis, packing, and peptide workflows.

## Installation

### Install from PyPI (recommended)

```bash
pip install warp-md
python -c "import warp_md; print(warp_md.System)"
```

Optional CLI dependency for YAML configs:

```bash
pip install "warp-md[cli]"
```

Installed CLI entry points:
- `warp-md` (analysis/config runner)
- `warp-pack` (packing utility)

`warp-pep` is currently distributed as a Rust crate CLI:

```bash
cargo install --path crates/warp-pep --force
```

## Build from Source

```bash
git clone https://github.com/Haayhur/warp-md.git
cd warp-md
pip install maturin
maturin develop
python -c "import warp_md; print(warp_md.System)"
```

## Quickstart

### CLI (analysis)

```bash
warp-md rg --topology top.pdb --traj traj.xtc --selection "protein"
warp-md rdf --topology top.pdb --traj traj.xtc --sel-a "resname SOL and name OW" --sel-b "resname SOL and name OW" --bins 200 --r-max 10

warp-md example > config.json
warp-md run config.json
warp-md list-plans
```

### Python (analysis)

```python
from warp_md import System, Trajectory, RgPlan

system = System.from_pdb("topology.pdb")
selection = system.select("protein")
traj = Trajectory.open_xtc("trajectory.xtc", system)
rg = RgPlan(selection).run(traj, system, device="auto")
print(rg[:5])
```

### Python (packing)

```python
from warp_md.pack import Box, Structure, PackConfig, water_pdb
from warp_md.pack.runner import run
from warp_md.pack.export import export

cfg = PackConfig(
    structures=[Structure(water_pdb("tip3p"), count=200)],
    box=Box((40.0, 40.0, 40.0)),
    min_distance=2.0,
)
result = run(cfg)
export(result, "pdb", "packed.pdb")
```

## Build and Test

```bash
# Rust
cargo test

# Focused crates
cargo test -p warp-pack
cargo test -p warp-pep

# CUDA engine tests (CUDA runtime required)
cargo test -p traj-engine --features cuda

# Python tests
python -m pytest python/warp_md/tests
```

## Documentation

- GitBook: [https://warp-md.gitbook.io](https://warp-md.gitbook.io)
- Local docs index: `docs/README.md`
- CLI reference: `docs/reference/cli.md`
- Packing guide: `docs/guides/packing.md`
- Peptide builder guide: `docs/guides/peptide-builder.md`
