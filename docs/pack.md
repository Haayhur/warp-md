# Molecule Packing (warp-pack)

CPU-first packer for building initial coordinates from small molecule templates. Rust implementation with Packmol-compatible configuration.

## Features

- 11+ input format support (PDB, XYZ, MOL2, CIF, GRO, LAMMPS, etc.)
- 7 output formats with full CONECT, TER, CRYST1 support in PDB
- Multi-molecule packing with spatial constraints
- Post-packing overlap relaxation
- Restart file support (Packmol-compatible)
- Random rotation control
- Disulfide bond handling
- Bundled water templates (TIP3P, TIP4P, etc.)

## CLI

## CLI

```bash
warp-pack --config pack.json --output out.pdb --format pdb
```

Packmol-style inputs are supported:

```bash
warp-pack --config packmol.inp
```

## Python

```python
from warp_md.pack import Box, Constraint, PackConfig, Structure, water_pdb
from warp_md.pack.runner import run
from warp_md.pack.export import export

cfg = PackConfig(
    structures=[
        Structure(water_pdb("tip3p"), count=200),
        Structure("ethanol.pdb", count=20),
    ],
    box=Box((40.0, 40.0, 40.0)),
    min_distance=2.0,
    add_box_sides=True,
    output=None,
)
result = run(cfg)
export(result, "pdb", "packed.pdb")
```

Note: bundled `water_pdb(...)` templates are **single-molecule** PDBs.

## JSON config example

```json
{
  "box": { "size": [40.0, 40.0, 40.0], "shape": "orthorhombic" },
  "seed": 0,
  "min_distance": 2.0,
  "add_box_sides": true,
  "writeout": 30.0,
  "writebad": true,
  "restart_to": "packed.restart",
  "relax_steps": 10,
  "relax_step": 0.5,
  "output": { "path": "packed.pdb", "format": "pdb" },
  "structures": [
    {
      "path": "water.pdb",
      "count": 200,
      "rotate": true,
      "constraints": [
        { "mode": "inside", "shape": "sphere", "center": [20.0, 20.0, 20.0], "radius": 18.0 }
      ]
    },
    {
      "path": "ethanol.mol2",
      "count": 20,
      "filetype": "mol2",
      "restart_from": "ethanol.restart",
      "changechains": true
    }
  ]
}
```

Template file: `docs/templates/pack.json`

## Supported input formats

- `pdb`
- `xyz`
- `mol2`
- `pdbx` / `cif` / `mmcif`
- `gro`
- `lammps` / `lammps-data` / `lmp`
- `crd`
- `tinker` / `txyz`
- `amber` / `inpcrd` / `rst` / `rst7` (coordinates; optional `topology` for metadata)

## Supported output formats

- `pdb` (CONECT, TER, CRYST1 supported)
- `xyz`
- `pdbx` / `cif` / `mmcif`
- `gro`
- `lammps` / `lammps-data` / `lmp`
- `mol2`
- `crd`

## Notes

- `writeout` writes periodic snapshots using `output` format.
- `writebad` writes the last partial structure on failure and forces `writeout` snapshots even without improvement.
- `restart_from` / `restart_to` read/write Packmol-style restart files (center + Euler angles).
- `relax_steps` and `relax_step` run a post-pack overlap relaxation pass.
- `gencan_maxit` and `gencan_step` control the optimizer iterations and step size.
- When the initial placement already satisfies overlap/constraint precision, GENCAN is skipped ("initial approximation is a solution" behavior).
- `nloop` / `nloop0` control the outer GENCAN loop counts (Packmol defaults: `200*ntype`, `20*ntype`).
- `discale` sets the initial GENCAN radius scale (decays toward 1.0).
- `add_box_sides_fix` adds a fixed offset to each box side in outputs (Packmol `add_box_sides`).
- `pbc_min` / `pbc_max` set explicit PBC bounds (Packmol `pbc` with min/max).
- `fbins` scales the spatial hashing cell size (`cell = min_distance * fbins`).
- `use_short_tol` enables a soft penalty zone between `short_tol_dist` and `min_distance`.
- Amber `topology` expects a simple `prmtop` file with `ATOM_NAME`, `RESIDUE_LABEL`, `RESIDUE_POINTER`,
  `ATOMIC_NUMBER`, and `CHARGE` sections.
- LAMMPS input supports `Atoms` styles `full`, `atomic`, `charge`, and `molecular`, optional `Masses`
  comments for element names, and `Bonds` section to build connectivity.

Structure-level radius overrides (Packmol parity):
- `radius` / `fscale` set per-atom defaults for a structure.
- `short_radius` / `short_radius_scale` enable per-atom short-distance penalties.
- `atom_overrides` allows per-atom overrides by index.
- `rot_bounds` constrains Euler rotation ranges per axis.
- `fixed_eulers` pairs with `positions` to fix orientation.
- `nloop` / `nloop0` override the global loop counts per structure.

Restart file format: one line per molecule with six floats: `x y z beta gamma teta`.
