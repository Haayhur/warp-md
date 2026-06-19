---
description: Comprehensive structural and topology IO across 14+ molecular formats
icon: cube
---

# warp-structure

`warp-structure` is the core IO and representation crate for loading molecular structures, emitting modified conformations, and bridging native representations to computational engines. It provides extensive coverage over 14 base structure formats with format-aware coordinate and topological mapping.

## Supported Formats

`warp-structure` supports both reading and writing for the vast majority of standard computational chemistry formats. The table below lists the recognized file types:

| Format | Extensions | Capabilities |
| --- | --- | --- |
| **PDB / PDB-Strict** | `.pdb`, `.brk`, `.ent` | Read, Write, Connectivity Support, Hexadecimal Index Parsing |
| **PDBx / mmCIF** | `.cif`, `.pdbx`, `.mmcif` | Read, Write (Fallback for large PDBs) |
| **PDBQT** | `.pdbqt` | Read (AutoDock Vina support) |
| **PQR** | `.pqr` | Read, Write |
| **XYZ** | `.xyz` | Read, Write |
| **GROMACS** | `.gro` | Read, Write (PBC Box Vectors) |
| **GROMOS96** | `.g96`, `.gromos96` | Read, Write |
| **Mol2** | `.mol2` | Read, Write |
| **Tinker** | `.tinker`, `.txyz` | Read |
| **AMBER / Inpcrd** | `.amber`, `.inpcrd`, `.rst`, `.rst7`| Read (Requires `.prmtop`), Write |
| **LAMMPS Data** | `.lammps`, `.lammps-data`, `.lmp` | Read, Write |
| **CHARMM CRD** | `.crd` | Read, Write |

## Advanced Output Options

`warp-structure` automatically handles format specifics during writing. Advanced constraints and fallbacks include:

* **Automatic mmCIF Fallback**: If writing to standard `.pdb` but the system exceeds 99,999 atoms or 9,999 residues, `warp-structure` can automatically fallback to writing `.cif` (mmCIF format) if standard PDB strictness is configured to fail on overflow.
* **Connectivity Rules**: PDB/PDBx parsing offers flags (`ignore_conect`, `non_standard_conect`) to explicitly ignore standard `CONECT` records or force them when non-standard topologies arise.
* **Box Vector Enforcement**: Handles orthorhombic minimum-image formatting natively (`add_box_sides`, `box_sides_fix`), useful when converting from unit-less XYZ into GRO/G96.

## Structural Data Model

Internally, `warp-structure` converts parsed data into `MoleculeData`, maintaining unified standard properties:

* **Atoms**: `AtomRecord` with positions (`Vec3`), atomic number, charge, radius, mass, occupancy, B-factor, segment, and chain identities.
* **Bonds**: Topological links as pairs of 0-based indices.
* **Box Vectors**: Triclinic/orthorhombic cell matrices.
* **TER markers**: Explicit chain separations for PDB writing.

## Agent Orchestration

Agent workflows automatically interface with `warp-structure` during system handoffs, parameter tuning (`warp-qm`), or system packing (`warp-pack`). You can explicitly dictate the output format in most `warp-md` ecosystem tools simply by appending the corresponding extension (e.g., specifying `outputs.coordinates: system.lammps` in `warp-cg build`).
