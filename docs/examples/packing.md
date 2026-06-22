---
description: Solvate a protein, pack water, add ions — copy-paste ready
icon: cube
---

# Packing

Solvate a protein, pack water molecules, add counterions — all from Python or CLI.

---

## Solvate a Protein

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

```bash
warp-pack --config pack.json --output solvated.pdb --format pdb
```

### JSON Config

```json
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

---

## Add Ions

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
```

---

## Use Restart Files

Packmol-compatible restart files for resumable packing:

```json
{
  "restart_from": "all.restart",
  "restart_to": "all.out.restart",
  "structures": [
    {"path": "water.pdb", "count": 200, "restart_from": "water.restart"}
  ]
}
```

---

## Supported Input Formats

`pdb`, `xyz`, `mol2`, `pdbx`/`cif`/`mmcif`, `gro`, `lammps`/`lammps-data`, `crd`, `tinker`/`txyz`, `amber`/`inpcrd`/`rst`/`rst7`

## Supported Output Formats

`pdb`, `xyz`, `pdbx`/`cif`/`mmcif`, `gro`, `lammps`/`lammps-data`, `mol2`, `crd`
