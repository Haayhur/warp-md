---
description: Solvate molecules, build simulation boxes, add ions — the packing tutorial
icon: cube
---

# Molecule Packing (warp-pack)

Solvate a protein, pack a polymer system into solvent, add counterions — all from Python or CLI. This tutorial covers single-structure packing, restart workflows, and polymer handoff.

---

## What You'll Learn

- Pack a protein into a water box
- Add ions for neutralization
- Use restart files for resumable packing
- Accept a polymer build manifest and solvate it
- Export in multiple formats

---

## Prerequisites

```bash
warp-pack --help                      # Verify the CLI is available
python -c "from warp_md.pack import PackConfig, Structure, water_pdb"  # Python API
```

---

## Step 1: Solvate a Protein

The simplest case: pack a protein and water into a cubic box.

### Python

```python
from warp_md.pack import Box, Structure, PackConfig, water_pdb
from warp_md.pack.runner import run
from warp_md.pack.export import export

cfg = PackConfig(
    structures=[
        Structure("protein.pdb", count=1),
        Structure(water_pdb("tip3p"), count=1000),
    ],
    box=Box((50.0, 50.0, 50.0)),
    min_distance=2.0,
    add_box_sides=True,
)
result = run(cfg)
export(result, "pdb", "solvated.pdb")
```

### CLI

```json
# pack.json
{
  "box": {"size": [50.0, 50.0, 50.0], "shape": "orthorhombic"},
  "min_distance": 2.0,
  "add_box_sides": true,
  "structures": [
    {"path": "protein.pdb", "count": 1},
    {"path": "water.pdb", "count": 1000, "rotate": true}
  ],
  "output": {"path": "solvated.pdb", "format": "pdb"}
}
```

```bash
warp-pack --config pack.json --output solvated.pdb --format pdb
```

---

## Step 2: Add Ions

Add counterions to neutralize the system:

```python
cfg = PackConfig(
    structures=[
        Structure("protein.pdb", count=1),
        Structure("ion_cl.pdb", count=10, filetype="mol2"),
        Structure("ion_na.pdb", count=10),
    ],
    box=Box((50.0, 50.0, 50.0)),
    min_distance=2.0,
)
result = run(cfg)
export(result, "pdb", "neutralized.pdb")
```

---

## Step 3: Use Restart Files

Packmol-compatible restart files let you resume packing or reuse placements:

```json
{
  "restart_from": "all.restart",
  "restart_to": "all.out.restart",
  "structures": [
    {"path": "water.pdb", "count": 200, "restart_from": "water.restart"}
  ]
}
```

### Key Parameters

| Parameter | What It Does |
|-----------|--------------|
| `restart_from` | Read previous placement state for continuation |
| `restart_to` | Write placement state for future resume |
| `seed` | Random seed for deterministic packing |
| `min_distance` | Hard minimum atom-atom distance (Å) |
| `add_box_sides` | Write CRYST1/box metadata to outputs |
| `relax_steps` / `relax_step` | Post-pack overlap relaxation |

---

## Step 4: Accept a Polymer Build

When `warp-build` emits a build manifest, `warp-pack` can consume it directly:

```json
{
  "schema_version": "warp-pack.agent.v1",
  "run_id": "warp-build-handoff-001",
  "polymer_build": {
    "build_manifest": "outputs/polymer_50mer.build.json",
    "topology_graph": "outputs/polymer_50mer.topology.json"
  },
  "environment": {
    "box": {"mode": "padding", "padding_angstrom": 12.0, "shape": "cubic"},
    "solvent": {"mode": "explicit", "model": "tip3p"},
    "ions": {
      "neutralize": {"enabled": true},
      "salt": {"name": "nacl", "molar": 0.15}
    },
    "morphology": {"mode": "single_chain_solution"}
  },
  "outputs": {
    "coordinates": "outputs/system.pdb",
    "format": "pdb-strict",
    "manifest": "outputs/system_manifest.json",
    "preserve_topology_graph": true,
    "write_conect": true
  }
}
```

```bash
warp-pack run pack_request.json --stream ndjson
```

---

## Output Formats

| Format | CLI Flag | Use Case |
|--------|----------|----------|
| `pdb` | `--format pdb` | Universal — works everywhere |
| `gro` | `--format gro` | GROMACS input |
| `xyz` | `--format xyz` | Lightweight, no topology |
| `mol2` | `--format mol2` | AMBER / charge-fitting |
| `pdbx` / `cif` | `--format pdbx` | mmCIF for deposition |

### Input Formats

`pdb`, `xyz`, `mol2`, `pdbx`/`cif`/`mmcif`, `gro`, `lammps`/`lammps-data`, `crd`, `tinker`/`txyz`, `amber`/`inpcrd`/`rst`/`rst7`

---

## Python API Summary

```python
from warp_md.pack import (
    PackConfig, Structure, Box, water_pdb,
    run, export,
)

# Build config
cfg = PackConfig(
    structures=[...],
    box=Box((size_x, size_y, size_z)),
    min_distance=2.0,
)

# Run packing
result = run(cfg)

# Export
export(result, "pdb", "output.pdb")
export(result, "gro", "output.gro")
export(result, "xyz", "output.xyz")
```

---

## What's Next

- [warp-build tutorial](warp-build.md) — build polymers for packing
- [warp-cg build tutorial](warp-cg-build.md) — build CG systems with membranes
- [Full pipeline example](../examples/agent-workflow.md) — build → pack → analyze
