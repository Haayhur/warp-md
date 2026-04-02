---
description: Your agent builds molecular worlds
icon: cube
---

# Molecule Packing

Before you can simulate anything, you need atoms arranged in a box. warp-pack is your agent's world-building tool.

{% hint style="info" %}
warp-pack is a CPU-first world-building engine that speaks Packmol's language but runs in pure Rust.
{% endhint %}

{% hint style="warning" %}
For polymer systems, build the chain with `warp-build` first. `warp-pack` no longer owns inline polymer construction.
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
)

result = run(cfg)
export(result, "pdb", "packed.pdb")
```
{% endtab %}
{% tab title="Polymer Handoff" %}
```bash
# Build the chain first
warp-build run polymer_build_request.json

# Then assemble the world around the build manifest
warp-pack example --mode polymer_build_handoff > pack_request.json
warp-pack run pack_request.json
```
{% endtab %}
{% endtabs %}

---

## Polymer Workflow

The agent-safe flow for polymer systems is:

1. `warp-build` compiles the source bundle into a target chain
2. `warp-build` emits coordinates, topology, charge handoff, and build manifest
3. `warp-pack` consumes the build manifest and assembles solvent, ions, box, and morphology

Minimal `warp-pack` handoff:

```json
{
  "version": "warp-pack.agent.v1",
  "polymer_build": {
    "build_manifest": "outputs/pmma_50mer.build.json"
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
    "manifest": "outputs/system_manifest.json"
  }
}
```

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

Bundled ions include `Na+`, `K+`, `Li+`, `Cl-`, `Br-`, `I-`, `Ca2+`, `Mg2+`, `SO4^2-`,
`HSO4-`, `NO3-`, and `OAc-`.
Bundled salts currently include:

| `salt.name` | Formula | Species |
|---|---|---|
| `nacl` | `NaCl` | `{"Na+": 1, "Cl-": 1}` |
| `kcl` | `KCl` | `{"K+": 1, "Cl-": 1}` |
| `licl` | `LiCl` | `{"Li+": 1, "Cl-": 1}` |
| `nabr` | `NaBr` | `{"Na+": 1, "Br-": 1}` |
| `nai` | `NaI` | `{"Na+": 1, "I-": 1}` |
| `libr` | `LiBr` | `{"Li+": 1, "Br-": 1}` |
| `cacl2` | `CaCl2` | `{"Ca2+": 1, "Cl-": 2}` |
| `mgcl2` | `MgCl2` | `{"Mg2+": 1, "Cl-": 2}` |
| `mgbr2` | `MgBr2` | `{"Mg2+": 1, "Br-": 2}` |
| `mgso4` | `MgSO4` | `{"Mg2+": 1, "SO4^2-": 1}` |
| `na2so4` | `Na2SO4` | `{"Na+": 2, "SO4^2-": 1}` |
| `kno3` | `KNO3` | `{"K+": 1, "NO3-": 1}` |
| `naoac` | `NaOAc` | `{"Na+": 1, "OAc-": 1}` |

Agents should prefer `salt.name` for these built-ins; `salt.formula` and `salt.species`
remain available for advanced cases.

For bundled polyatomic systems, the built-in templates work the same way as bundled water models:
no manual ion PDB paths needed. `salt.formula` remains a convenience path, not a universal
chemistry parser. For custom or ambiguous polyatomic systems, prefer `salt.name` or explicit
`salt.species`.

Ion metadata is now registry-backed from `python/warp_md/pack/data/ions.json`. To add more ions
without patching code, point `WARP_MD_ION_REGISTRY` at a JSON file with extra entries; relative
template paths resolve from that registry file's directory.

Salt recipes are registry-backed from `python/warp_md/pack/data/salts.json`. To add more built-in
salt names without patching code, point `WARP_MD_SALT_REGISTRY` at a JSON file with extra entries.

Python also exposes a low-level chemistry helper path for agent scripts:

```python
from warp_md.pack import resolve_chemistry, solution_pack_config

recipe = resolve_chemistry(
    box_size=50.0,
    solvent_model="tip3p",
    salt="mgso4",
    salt_molar=0.15,
    neutralize=True,
    solute_net_charge_e=-2.0,
)

cfg = solution_pack_config(
    solute_path="ligand.pdb",
    box_size=50.0,
    output_path="system_50A.pdb",
    solvent_model="tip3p",
    salt="cacl2",
    salt_molar=0.15,
)
```

`resolve_chemistry(...)` is the preview API for agents and scripts. It resolves salt stoichiometry,
rounded counts, chosen templates, estimated waters, achieved molarity, and neutralization choices
before the packer runs.

CLI equivalent:

```bash
warp-pack solution \
  --solute ligand.pdb \
  --box 50 \
  --solvent tip3p \
  --salt cacl2 \
  --salt-molar 0.15 \
  --output system_50A.pdb
```

Top-level `warp-pack --help` now advertises the legacy `--config` path, the chemistry-intent
`solution` path, and the contract commands so users and agents can discover the right entrypoint
without reading the source first.

Bundled ion entries also carry parameterization hints such as `recommended_families` and
`preferred_water_models`. These are exposed through Python `ion_metadata(...)` and
`ion_parameterization(...)` helpers so agents can keep force-field selection coupled to ion choice.

For agent-first custom chemistry, you can define request-scoped ions and salts inline:

```json
{
  "environment": {
    "ions": {
      "catalog": {
        "ions": [
          {
            "species": "Rb+",
            "template": "/abs/path/rb.pdb",
            "formula_symbol": "Rb",
            "charge_e": 1,
            "mass_amu": 85.4678
          },
          {
            "species": "F-",
            "template": "/abs/path/f.pdb",
            "formula_symbol": "F",
            "charge_e": -1,
            "mass_amu": 18.998403
          }
        ],
        "salts": [
          {
            "name": "rbf",
            "species": {"Rb+": 1, "F-": 1}
          }
        ]
      },
      "salt": {"name": "rbf", "molar": 0.15}
    }
  }
}
```

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
| `add_box_sides` | Add box dimensions to outputs (default: on) |
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
| `resnumbers` | Residue numbering mode (`3` for unique residue id per packed molecule); repeated structures auto-offset by default |
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
