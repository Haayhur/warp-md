---
description: Build, mutate, and export peptides — copy-paste ready
icon: dna
---

# Peptide Examples

Build, mutate, and export peptides — all from CLI or Python.

---

## Build an Alpha-Helix

```bash
# 5-residue alanine helix
warp-pep build -s AAAAA --preset alpha-helix --oxt -o helix.pdb

# Extended peptide
warp-pep build -s ACDEFGHIKLM -o extended.pdb

# Three-letter codes with Amber variants
warp-pep build -t ALA-CYX-HID-GLU --oxt --detect-ss -o out.pdb
```

---

## Mutate a Residue

```bash
# Mutate residue 2 from Ala to Gly
warp-pep mutate -i input.pdb -m A2G -o mutated.pdb

# Build and mutate in one shot
warp-pep mutate -s ACDEF -m C2G,D3W --oxt -o out.pdb
```

---

## Python (CLI Wrapper)

```python
import subprocess

# Build
subprocess.run([
    "warp-pep", "build", "-s", "ACDEFGHIKLM",
    "--preset", "alpha-helix", "-o", "helix.pdb"
], check=True)

# Mutate
subprocess.run([
    "warp-pep", "mutate", "-i", "helix.pdb",
    "-m", "A5G", "-o", "mutated.pdb"
], check=True)
```

---

## Build + Pack Pipeline

```python
import subprocess
from warp_md.pack import Box, Structure, PackConfig, water_pdb
from warp_md.pack.runner import run
from warp_md.pack.export import export

# 1. Build peptide
subprocess.run([
    "warp-pep", "build", "-s", "ACDEFGHIKLM",
    "--preset", "alpha-helix", "-o", "peptide.pdb"
], check=True)

# 2. Pack with water
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

---

## Supported Amino Acids

| Category | Codes |
|----------|-------|
| Standard 20 | `A C D E F G H I K L M N P Q R S T V W Y` |
| Amber variants | `CYX`, `HID`, `HIE`, `HIP`, `ASH`, `GLH`, `LYN` |
| Non-standard | `MSE` (selenomethionine), `PCA` (pyroglutamic) |
| D-amino acids | Lowercase: `a c d e f g h i k l m n p q r s t v w y` |

## Ramachandran Presets

| Preset | φ | ψ | 
|--------|---|---|
| `extended` | 180° | 180° |
| `alpha-helix` | -57° | -47° |
| `beta-sheet` | -120° | +130° |
| `polyproline` | -75° | +145° |

## Output Formats

`pdb`, `cif`/`pdbx`/`mmcif`, `xyz`, `gro`, `mol2`, `crd`, `lammps`/`lmp`
