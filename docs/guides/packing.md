---
description: Your agent builds molecular worlds
icon: cube
---

# Molecule Packing

Before you can simulate anything, you need atoms arranged in a box. warp-pack is your agent's world-building tool.

{% hint style="info" %}
warp-pack is a CPU-first packing engine that speaks Packmol's language but runs in pure Rust.
{% endhint %}

---

## Quick Start

{% tabs %}
{% tab title="CLI" %}
```bash
# From JSON config
warp-pack --config pack.json --output packed.pdb --format pdb

# Packmol-style inputs also work
warp-pack --config packmol.inp
```
{% endtab %}

{% tab title="Python" %}
```python
from warp_md.pack import Box, Structure, PackConfig
from warp_md.pack.runner import run
from warp_md.pack.export import export

cfg = PackConfig(
    structures=[
        Structure("water.pdb", count=200),
        Structure("ethanol.pdb", count=20),
    ],
    box=Box((40.0, 40.0, 40.0)),
    min_distance=2.0,
    add_box_sides=True,
)

result = run(cfg)
export(result, "pdb", "packed.pdb")
```
{% endtab %}
{% endtabs %}

---

## JSON Configuration

{% code title="pack.json" %}
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
      "changechains": true
    }
  ]
}
```
{% endcode %}

---

## Bundled Water Templates

warp-pack ships with single-molecule water templates. No hunting for PDB files:

```python
from warp_md.pack import water_pdb, Structure, PackConfig, Box
from warp_md.pack.runner import run

cfg = PackConfig(
    structures=[Structure(water_pdb("tip3p"), count=1000)],
    box=Box((40.0, 40.0, 40.0)),
    min_distance=2.0,
)
result = run(cfg)
```

{% hint style="warning" %}
Bundled `water_pdb(...)` templates are **single-molecule** PDBs. Perfect for packing.
{% endhint %}

---

## Restart Files

Resume packing or reuse placements with Packmol-style restart files:

```json
{
  "restart_from": "all.restart",
  "restart_to": "all.out.restart",
  "structures": [
    {
      "path": "water.pdb",
      "count": 200,
      "restart_from": "water.restart"
    }
  ]
}
```

**Format**: One line per molecule with six floats: `x y z beta gamma theta`

---

## Supported Formats

{% columns %}
{% column %}
### Input Formats

- `pdb`
- `xyz`
- `mol2`
- `pdbx` / `cif` / `mmcif`
- `gro`
- `lammps` / `lammps-data` / `lmp`
- `crd`
- `tinker` / `txyz`
- `amber` / `inpcrd` / `rst` / `rst7`
{% endcolumn %}

{% column %}
### Output Formats

- `pdb` (CONECT, TER, CRYST1)
- `xyz`
- `pdbx` / `cif` / `mmcif`
- `gro`
- `lammps` / `lammps-data` / `lmp`
- `mol2`
- `crd`
{% endcolumn %}
{% endcolumns %}

---

## Configuration Reference

<details>
<summary>Global Options</summary>

| Option | What It Does |
|--------|--------------|
| `box` | Box dimensions and shape |
| `seed` | Random seed for reproducibility |
| `min_distance` | Hard minimum distance between atoms |
| `writeout` | Write periodic snapshots (seconds) |
| `writebad` | Write partial structure on failure |
| `restart_from` / `restart_to` | Packmol restart files |
| `relax_steps` / `relax_step` | Post-pack overlap relaxation |
| `add_box_sides` | Add box dimensions to outputs |
| `pbc_min` / `pbc_max` | Explicit PBC bounds |

</details>

<details>
<summary>Structure Options</summary>

| Option | What It Does |
|--------|--------------|
| `path` | Path to structure file |
| `count` | Number of copies to place |
| `rotate` | Allow random rotation |
| `filetype` | Override file type detection |
| `constraints` | Placement constraints |
| `radius` / `fscale` | Per-atom radius defaults |
| `changechains` | Assign unique chain IDs |
| `rot_bounds` | Constrain Euler rotation ranges |
| `fixed_eulers` | Fix orientation (with positions) |

</details>

<details>
<summary>Constraint Types</summary>

```json
// Inside a sphere
{ "mode": "inside", "shape": "sphere", "center": [20, 20, 20], "radius": 18 }

// Outside a box
{ "mode": "outside", "shape": "box", "min": [0, 0, 0], "max": [10, 10, 10] }

// Fixed position
{ "mode": "fixed", "position": [20, 20, 20] }
```

</details>

---

## Example: Solvated Protein

```python
from warp_md.pack import Box, Structure, PackConfig, water_pdb
from warp_md.pack.runner import run
from warp_md.pack.export import export

cfg = PackConfig(
    structures=[
        # Protein at center (fixed)
        Structure("protein.pdb", count=1, constraints=[
            {"mode": "fixed", "position": [25.0, 25.0, 25.0]}
        ]),
        # Water molecules around protein
        Structure(water_pdb("tip3p"), count=5000, constraints=[
            {"mode": "outside", "shape": "sphere", "center": [25, 25, 25], "radius": 15}
        ]),
    ],
    box=Box((50.0, 50.0, 50.0)),
    min_distance=2.5,
    add_box_sides=True,
)

result = run(cfg)
export(result, "pdb", "solvated_protein.pdb")
print("World built successfully 🌍")
```

---

## Integration with peptide builders

Pack a peptide structure generated by an external builder (for example, `warp-pep` CLI):

```python
from warp_md.pack import Box, Structure, PackConfig, water_pdb
from warp_md.pack.runner import run
from warp_md.pack.export import export

# Assume peptide.pdb was generated upstream (e.g., warp-pep build ... -o peptide.pdb)

# Pack with water
cfg = PackConfig(
    structures=[
        Structure("peptide.pdb", count=1),
        Structure(water_pdb("tip3p"), count=1000),
    ],
    box=Box((40.0, 40.0, 40.0)),
    min_distance=2.0,
)

result = run(cfg)
export(result, "pdb", "solvated.pdb")
```

For peptide building documentation, see [Peptide Builder Guide](peptide-builder.md).

---

For a complete walkthrough, see [Zika VLP Packing Tutorial](../tutorial/zika_vlp_packmol_tutorial.md).
