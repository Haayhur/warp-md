# Peptide Builder (warp-pep)

All-atom peptide structure construction and mutation from amino acid sequences using internal coordinate geometry.

> Note: the `warp-md` wheel currently does not expose a `warp_md.pep` Python module.
> Python examples below call the `warp-pep` CLI via `subprocess`.

Install `warp-pep` CLI (from repo root):

```bash
cargo install --path crates/warp-pep --force
```

## CLI

```bash
# Build from one-letter codes
warp-pep build -s AAAAA --preset alpha-helix --oxt -o helix.pdb

# Build from three-letter codes with Amber naming
warp-pep build -t ALA-CYX-HID-GLU --oxt --detect-ss -o out.pdb

# Build from JSON spec
warp-pep build -j spec.json

# Mutate residue 2 from Ala to Gly
warp-pep mutate -i input.pdb -m A2G -o mutated.pdb

# Build and mutate in one shot
warp-pep mutate -s ACDEF -m C2G,D3W --oxt -o out.pdb
```

## Python (CLI Wrapper)

```python
import subprocess

# Build extended peptide
subprocess.run(["warp-pep", "build", "-s", "ACDEFGHIKLM", "-o", "extended.pdb"], check=True)

# Build with preset
subprocess.run(
    ["warp-pep", "build", "-s", "AAAA", "--preset", "alpha-helix", "-o", "helix.pdb"],
    check=True,
)

# Mutate a residue in-place
subprocess.run(["warp-pep", "mutate", "-i", "helix.pdb", "-m", "A2G", "-o", "helix_mutated.pdb"], check=True)
```

## JSON Configuration

### Single-Chain Spec

```json
{
  "residues": ["ALA", "CYX", "HID", "GLU"],
  "preset": "alpha-helix",
  "oxt": true,
  "detect_ss": true,
  "mutations": ["A1G"],
  "output": "out.pdb",
  "format": "pdb"
}
```

### Multi-Chain Spec

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
      "residues": ["GLY", "VAL", "TRP"]
    }
  ],
  "oxt": true,
  "detect_ss": true,
  "output": "multi.pdb"
}
```

## Supported Amino Acids

### Standard 20

One-letter codes: `A C D E F G H I K L M N P Q R S T V W Y`

### Amber Force Field Variants

| Code | Residue | Description |
|------|---------|-------------|
| CYX | Cysteine | Disulfide-bonded |
| HID | Histidine | Nδ-H tautomer |
| HIE | Histidine | Nε-H tautomer |
| HIP | Histidine | Doubly protonated |
| ASH | Aspartate | Protonated (ASH) |
| GLH | Glutamate | Protonated (GLH) |
| LYN | Lysine | Neutral |

### Non-Standard Residues

| Code | Canonical | Description |
|------|-----------|-------------|
| MSE | MET | Selenomethionine (SE instead of SD) |
| PCA | GLU | Pyroglutamic acid (cyclic) |

### D-Amino Acids

Use lowercase one-letter codes: `a c d e f g h i k l m n p q r s t v w y`

Example: `"aGlA"` = D-Ala, Gly, Leu, Ala

## Ramachandran Presets

| Preset | φ (phi) | ψ (psi) | Description |
|--------|---------|---------|-------------|
| `extended` / `ext` | 180° | 180° | Extended conformation (default) |
| `alpha-helix` / `alpha` / `helix` | -57° | -47° | α-helix |
| `beta-sheet` / `beta` / `sheet` | -120° | +130° | Anti-parallel β-sheet |
| `polyproline` / `ppii` / `polyproline` | -75° | +145° | Polyproline-II helix |

## Custom Backbone Angles

Specify per-junction phi/psi/omega angles:

```bash
warp-pep build -s AAA --phi=-60,-60 --psi=-45,-45 --omega=180,180 --oxt
```

Note: Angle arrays must have length `num_residues - 1`.

## Mutation Syntax

Format: `<from><position><to>`

- `A5G` → Mutate residue 5 from Ala to Gly
- `W123A` → Mutate residue 123 from Trp to Ala
- Multiple: `A5G,L10W,C15M` (comma-separated)

## Supported Output Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| PDB | `.pdb` | CONECT, TER supported |
| PDBx/CIF | `.cif`, `.mmcif`, `.pdbx` | mmCIF format |
| XYZ | `.xyz` | Simple coordinate format |
| GRO | `.gro` | GROMACS (nm → Å conversion) |
| MOL2 | `.mol2` | Tripos format |
| CRD | `.crd` | Amber coordinates |
| LAMMPS | `.lmp`, `.lammps` | LAMMPS data format |

## Terminal Capping

Add ACE (N-terminal) and NME (C-terminal) caps:

`warp-pep` CLI in this repo currently does not expose a direct cap-toggle flag.
Use the Rust `warp-pep` library API for capping workflows.

Caps use Amber naming:
- **ACE**: Acetyl group (CH3-CO-) at N-terminus
- **NME**: N-methylamide (-NH-CH3) at C-terminus

## Disulfide Detection

Auto-detect Sγ–Sγ distance < 2.5 Å and relabel CYS → CYX:

```bash
warp-pep build -t ALA-CYS-CYS-ALA --detect-ss -o with_ssbond.pdb
```

Also writes SSBOND records to PDB output.

## OXT Terminal Oxygen

Add the terminal carboxyl oxygen:

```bash
warp-pep build -s AAA --oxt -o out.pdb
```

## Notes

- All bond lengths, angles, and dihedrals from standard peptide geometry
- Peptide bonds default to trans (ω = 180°)
- Use `--omega ~0` for cis bonds (e.g., preceding proline)
- Internal units are Angstrom
- Multi-chain structures renumber residues globally (seq_id)
- JSON spec is self-contained (may include output path and format)

## Integration with warp-pack

Build peptides and solvate them with water:

```python
import subprocess
from warp_md.pack import Box, Structure, PackConfig, water_pdb
from warp_md.pack.runner import run
from warp_md.pack.export import export

# Build peptide with warp-pep CLI
subprocess.run(
    ["warp-pep", "build", "-s", "ACDEFGHIKLM", "--preset", "alpha-helix", "-o", "peptide.pdb"],
    check=True,
)

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

For packing documentation, see `docs/guides/packing.md`.
