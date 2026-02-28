---
description: Build and mutate peptide structures
icon: dna
---

# Peptide Building with warp-pep

warp-pep is your agent's peptide construction toolkit — build all-atom structures from sequences, apply mutations, and export to any format.

{% hint style="info" %}
warp-pep builds peptides from internal coordinate geometry (bond lengths, angles, dihedrals) for all 20 standard amino acids plus Amber force field variants.
{% endhint %}

{% hint style="warning" %}
The `warp-md` wheel does not currently expose a `warp_md.pep` Python module.
Python snippets in this guide drive the `warp-pep` CLI via `subprocess`.
{% endhint %}

```bash
# From repo root
cargo install --path crates/warp-pep --force
```

---

## Quick Start

{% tabs %}
{% tab title="CLI" %}
```bash
# Build a 5-residue alpha-helix
warp-pep build -s AAAAA --preset alpha-helix --oxt -o helix.pdb

# Build with Amber naming and disulfide detection
warp-pep build -t ALA-CYX-HID-GLU --oxt --detect-ss -o out.pdb

# Mutate residue 2 from Ala to Gly
warp-pep mutate -i input.pdb -m A2G -o mutated.pdb
```
{% endtab %}

{% tab title="Python" %}
```python
import subprocess

subprocess.run(
    ["warp-pep", "build", "-s", "AAAA", "--preset", "alpha-helix", "--oxt", "-o", "peptide.pdb"],
    check=True,
)
subprocess.run(["warp-pep", "mutate", "-i", "peptide.pdb", "-m", "A2G", "-o", "peptide_mutated.pdb"], check=True)
```
{% endtab %}
{% endtabs %}

---

## Input Modes

### One-Letter Codes

The simplest way to build peptides:

```bash
warp-pep build -s ACDEFGHIKLMNPQRSTVWY -o all20.pdb
```

{% hint style="info" %}
**D-Amino acids**: Use lowercase for D-form variants (e.g., `a` = D-Ala). Mixed sequences work: `"aGlA"` = D-Ala, Gly, Leu, Ala.
{% endhint %}

### Three-Letter Codes with Amber Variants

For force field precision:

```bash
warp-pep build -t ALA-CYX-HID-HIP-ASH-GLH-LYN -o amber.pdb
```

| Code | Residue | When to Use |
|------|---------|-------------|
| CYX | Cysteine | Disulfide-bonded CYS |
| HID | Histidine | Nδ protonated (default in many FF) |
| HIE | Histidine | Nε protonated |
| HIP | Histidine | Double protonated (charged) |
| ASH | Aspartate | Protonated (neutral) |
| GLH | Glutamate | Protonated (neutral) |
| LYN | Lysine | Neutral (deprotonated) |

### JSON Specification

For full control:

```json
{
  "residues": ["ALA", "CYX", "HID", "GLU"],
  "preset": "alpha-helix",
  "oxt": true,
  "detect_ss": true,
  "mutations": ["A1G"],
  "output": "out.pdb"
}
```

```bash
warp-pep build -j spec.json
```

---

## Ramachandran Presets

Choose your secondary structure:

```bash
# Alpha-helix
warp-pep build -s AAAAA --preset alpha-helix

# Beta-sheet
warp-pep build -s AAAAA --preset beta-sheet

# Polyproline-II
warp-pep build -s AAAAA --preset polyproline

# Extended (default)
warp-pep build -s AAAAA --preset extended
```

| Preset | φ | ψ | Structure |
|--------|---|---|------------|
| `extended` | 180° | 180° | Extended strand |
| `alpha-helix` | -57° | -47° | Right-handed α-helix |
| `beta-sheet` | -120° | +130° | Anti-parallel β-sheet |
| `polyproline` | -75° | +145° | Polyproline-II helix |

---

## Custom Backbone Angles

For precise control, specify per-junction angles:

```bash
# 3 residues = 2 junctions
warp-pep build -s AAA \
  --phi=-60,-60 \
  --psi=-45,-45 \
  --omega=180,180 \
  --oxt
```

{% hint style="warning" %}
Angle arrays must have length `num_residues - 1`. Use `--omega ~0` for cis peptide bonds (rare, usually preceding proline).
{% endhint %}

---

## Mutations

### CLI

```bash
# Single mutation
warp-pep mutate -i input.pdb -m A5G -o out.pdb

# Multiple mutations
warp-pep mutate -i input.pdb -m A5G,L10W,C15M -o out.pdb

# Build and mutate in one shot
warp-pep mutate -s ACDEF -m C2G,D3W --oxt -o out.pdb
```

**Mutation format**: `<from><position><to>`
- `A5G` = Ala at position 5 → Gly
- `W123A` = Trp at position 123 → Ala

### Python

```python
import subprocess

# Mutate position 5 Ala -> Gly
subprocess.run(["warp-pep", "mutate", "-i", "input.pdb", "-m", "A5G", "-o", "out.pdb"], check=True)

# Multiple mutations
subprocess.run(["warp-pep", "mutate", "-i", "input.pdb", "-m", "A5G,L10W", "-o", "out_multi.pdb"], check=True)
```

---

## Multi-Chain Structures

Build complexes with multiple chains:

```json
{
  "chains": [
    {
      "id": "A",
      "residues": ["ALA", "CYS", "GLU"],
      "preset": "alpha-helix"
    },
    {
      "id": "B",
      "residues": ["GLY", "VAL", "TRP"],
      "preset": "beta-sheet"
    }
  ],
  "oxt": true,
  "detect_ss": true
}
```

```bash
warp-pep build -j multi.json -o complex.pdb
```

```python
import json
import subprocess

spec = {
    "chains": [
        {"id": "A", "residues": ["ALA", "GLU", "VAL"], "preset": "alpha-helix"},
        {"id": "B", "residues": ["GLY", "TRP", "SER"], "preset": "beta-sheet"},
    ],
    "oxt": True,
}
with open("multi.json", "w", encoding="utf-8") as f:
    json.dump(spec, f)

subprocess.run(["warp-pep", "build", "-j", "multi.json", "-o", "complex.pdb"], check=True)
```

---

## Terminal Capping

Add ACE/NME caps for neutral termini:

`warp-pep` CLI in this repo currently does not expose a direct cap-toggle flag.
Use the Rust `warp-pep` library API for capping workflows.

| Cap | Structure | Position |
|-----|-----------|----------|
| ACE | CH3-CO- | N-terminal |
| NME | -NH-CH3 | C-terminal |

{% hint style="warning" %}
Adding NME cap removes any existing OXT atoms to avoid invalid mixed terminal chemistry.
{% endhint %}

---

## Disulfide Bonds

Auto-detect and label disulfide bonds:

```bash
warp-pep build -t ALA-CYS-CYS-ALA --detect-ss -o out.pdb
```

```python
import subprocess

subprocess.run(
    ["warp-pep", "build", "-t", "ALA-CYS-CYS-ALA", "--detect-ss", "-o", "out.pdb"],
    check=True,
)
```

Scans for CYS pairs with Sγ–Sγ distance < 2.5 Å and:
- Relabels as CYX (Amber convention)
- Writes SSBOND records to PDB

---

## Terminal OXT

Add the carboxyl terminal oxygen:

```bash
warp-pep build -s AAA --oxt -o out.pdb
```

```python
import subprocess

subprocess.run(["warp-pep", "build", "-s", "AAA", "--oxt", "-o", "out.pdb"], check=True)
```

{% hint style="info" %}
OXT is added to the last residue of each chain, excluding C-terminal caps (NME).
{% endhint %}

---

## Non-Standard Residues

warp-pep supports common non-standard amino acids:

| Code | Canonical | Special Atoms |
|------|-----------|---------------|
| MSE | MET | SE instead of SD (selenium) |
| PCA | GLU | Pyroglutamic acid (cyclic, no OE2) |

```bash
warp-pep build -t ALA-MSE-PCA -o nonstd.pdb
```

---

## Output Formats

Export to any major format:

```bash
warp-pep build -s AAA -o peptide.pdb    # PDB
warp-pep build -s AAA -o peptide.cif    # PDBx/mmCIF
warp-pep build -s AAA -o peptide.gro    # GROMACS
warp-pep build -s AAA -o peptide.mol2   # Tripos MOL2
warp-pep build -s AAA -o peptide.lmp    # LAMMPS
```

```python
import subprocess

subprocess.run(["warp-pep", "build", "-s", "AAA", "-o", "peptide.pdb"], check=True)
```

---

## Complete Workflow Example

Build a disulfide-bonded peptide with caps and analyze:

```python
import subprocess
from warp_md.pack import Box, PackConfig, Structure, water_pdb
from warp_md.pack.export import export
from warp_md.pack.runner import run

# Build with disulfide detection
subprocess.run(
    ["warp-pep", "build", "-t", "ALA-CYS-CYS-GLU", "--preset", "alpha-helix", "--detect-ss", "-o", "peptide.pdb"],
    check=True,
)

# Solvate with warp-pack
cfg = PackConfig(
    structures=[
        Structure("peptide.pdb", count=1),
        Structure(water_pdb("tip3p"), count=800),
    ],
    box=Box((36.0, 36.0, 36.0)),
    min_distance=2.0,
)
result = run(cfg)
export(result, "pdb", "solvated_peptide.pdb")
```

---

For CLI reference, see [pack.md](../pack.md) for packing integration.
