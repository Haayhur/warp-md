# warp-md

Rust/CUDA/Python toolkit for molecular trajectory IO and analysis.

## Public repo scope

This public repo contains:
- Rust workspace and crates (`crates/`)
- Python package and tests (`python/warp_md/`)
- User docs (`docs/`)

Internal benchmark/manuscript/validation pipelines are intentionally excluded from this public snapshot.

## Modules

- `warp-md` (analysis module): trajectory IO + analysis plans with Python API/CLI (`python/warp_md/`, `crates/traj-*`, `crates/traj-py`)
- `warp-pack`: molecular packing engine and interfaces (`crates/warp-pack`, `python/warp_md/pack/`)
- `warp-pep`: peptide builder and mutation engine (`crates/warp-pep`)

## Installation

### Path A: Install from PyPI (recommended)

```bash
pip install warp-md
python -c "import warp_md; print(warp_md.System)"
```

Optional CUDA wheel:

```bash
pip install warp-md-cuda
```

### Path B: Build from Source (git clone)

```bash
git clone https://github.com/Haayhur/warp-md.git
cd warp-md
pip install maturin
maturin develop
python -c "import warp_md; print(warp_md.System)"
```

## Build and test

Rust:

```bash
cargo test
```

Pack crate:

```bash
cargo test -p warp-pack
```

Peptide crate:

```bash
cargo test -p warp-pep
```

CUDA-enabled engine tests (CUDA runtime required):

```bash
cargo test -p traj-engine --features cuda
```

Python bindings (maturin):

```bash
maturin develop
python -c "import warp_md; print(warp_md.System)"
```

Python tests:

```bash
python -m pytest python/warp_md/tests
```

## CLI quickstart

```bash
warp-md rg --topology top.pdb --traj traj.xtc --selection "protein"
warp-md rdf --topology top.pdb --traj traj.xtc --sel-a "resname SOL and name OW" --sel-b "resname SOL and name OW" --bins 200 --r-max 10

warp-md example > config.json
warp-md run config.json
warp-md list-plans
```

## Python quickstart

```python
from warp_md import System, Trajectory, RgPlan

system = System.from_pdb("topology.pdb")
selection = system.select("protein")
traj = Trajectory.open_xtc("trajectory.xtc", system)
rg = RgPlan(selection).run(traj, system, device="auto")
print(rg[:5])
```

## Pack quickstart (Python)

```python
from warp_md.pack import Box, Structure, PackConfig
from warp_md.pack.runner import run
from warp_md.pack.export import export

cfg = PackConfig(
    structures=[Structure("water.pdb", count=100)],
    box=Box((30.0, 30.0, 30.0)),
)
result = run(cfg)
export(result, "pdb", "packed.pdb")
```

## Documentation

- GitBook: [warp-md.gitbook.io](https://warp-md.gitbook.io)
- Local docs index: `docs/README.md`
- Installation paths (`pip` and source build)
- Quickstart and tutorial track
- Guides for packing and CUDA/GPU usage
- Full API reference and agent schema contract
