---
description: PCA, DSSP, RMSD variants, fluctuations, dihedrals, covariance — the deep structure toolkit
icon: dna
---

# Structural Analysis

When your agent needs to go beyond "how compact is this thing?" and into "what is this thing actually doing?" — welcome to structural analysis.

{% hint style="info" %}
These Plans cover everything from PCA to DSSP to per-residue RMSD. Collective motions, secondary structure, conformational analysis — it's all here.
{% endhint %}

---

## RMSD Variants

Not all deviation is created equal. Pick the right flavor.

### SymmRmsdPlan

*RMSD that accounts for molecular symmetry — when atoms are interchangeable.*

```python
from warp_md import SymmRmsdPlan

plan = SymmRmsdPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.symmrmsd()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames,)`

---

### DistanceRmsdPlan

*No superposition needed — compares internal distance matrices.*

```python
from warp_md import DistanceRmsdPlan

plan = DistanceRmsdPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.distance_rmsd()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |

#### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`

---

### PairwiseRmsdPlan

*All-vs-all RMSD matrix. Perfect for clustering or identifying conformational states.*

```python
from warp_md import PairwiseRmsdPlan

plan = PairwiseRmsdPlan(selection)
matrix = plan.run(traj, system)  # (n_frames, n_frames)
```

Also available as: `warp_md.analysis.pairwise_rmsd()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to compare |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, n_frames)`
- **Unit**: Angstrom

---

### RmsdPerResPlan

*Which residues are moving the most? Per-residue RMSD answers that.*

```python
from warp_md import RmsdPerResPlan

plan = RmsdPerResPlan(selection)
result = plan.run(traj, system)  # (n_frames, n_residues)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, n_residues)`
- **Unit**: Angstrom

---

## Fluctuations

Three ways to measure how much atoms wiggle.

### RmsfPlan

*Root mean square fluctuation — the classic flexibility metric.*

```python
from warp_md import RmsfPlan

plan = RmsfPlan(selection)
rmsf = plan.run(traj, system)  # (n_atoms,) in Å
```

Also available as: `warp_md.analysis.rmsf()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |
| `byres` | `bool` | `False` | Per-residue rather than per-atom |
| `bymask` | `bool` | `False` | Compute RMSF per mask group |
| `calcadp` | `bool` | `False` | Compute anisotropic ADPs alongside |

#### Output

- **Type**: 1D array
- **Shape**: `(n_atoms,)` or `(n_residues,)`
- **Unit**: Angstrom

---

### BfactorsPlan

*Crystallographic B-factors from dynamics — compare simulation to experiment.*

```python
from warp_md import BfactorsPlan

plan = BfactorsPlan(selection)
bfactors = plan.run(traj, system)  # (n_atoms,) in Å²
```

Also available as: `warp_md.analysis.bfactors()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |

#### Output

- **Type**: 1D array
- **Shape**: `(n_atoms,)`
- **Unit**: Angstrom²

---

### AtomicFluctPlan

*Raw atomic fluctuations — the foundation RMSF and B-factors are built on.*

```python
from warp_md import AtomicFluctPlan

plan = AtomicFluctPlan(selection)
fluct = plan.run(traj, system)
```

Also available as: `warp_md.analysis.atomicfluct()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |

#### Output

- **Type**: 1D array
- **Shape**: `(n_atoms,)`

---

### AtomicAdpPlan

*Atomic displacement parameters (ADP) — anisotropic temperature factors per atom from trajectory fluctuations.*

```python
from warp_md import AtomicAdpPlan

plan = AtomicAdpPlan(selection)
adp = plan.run(traj, system)
```

Also available as: `warp_md.analysis.atomicfluct()` (ADP output mode)

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |

#### Output

- **Type**: 3D array
- **Shape**: `(n_atoms, 3, 3)` anisotropic displacement tensors

---

## PCA & Normal Modes

Reduce dimensionality. Find the big motions. Ignore the noise.

### PcaPlan

*Principal component analysis — the collective motions that matter.*

```python
from warp_md import PcaPlan

plan = PcaPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.pca()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |
| `n_vecs` | `int` | `20` | Number of eigenvectors to retain |
| `fit` | `bool` | `True` | Superpose frames before PCA |
| `ref` | `str` | `None` | Reference structure (`"topology"` or `"frame0"`) |
| `ref_mask` | `str` | `None` | Reference atom mask for alignment |

#### Output

- **Type**: dict with keys `eigenvectors`, `eigenvalues`, `mean`, `projections`, `pca`

---

### AnalyzeModesPlan

*Analyze normal modes or PCA modes — eigenvalues, eigenvectors, the works.*

```python
from warp_md import AnalyzeModesPlan

plan = AnalyzeModesPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.analyze_modes()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |

#### Output

- **Type**: dict with eigen decomposition results

---

### ProjectionPlan

*Project trajectory onto PCA or normal mode eigenvectors.*

```python
from warp_md import ProjectionPlan

plan = ProjectionPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.projection()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |
| `eigenvectors` | `array` | required | Eigenvectors from PCA/modes |
| `scalar_type` | `str` | `"scalar"` | Type of projection |
| `eigenvalues` | `array` | `None` | Optional eigenvalues for scaling |
| `average_coords` | `array` | `None` | Optional average coordinates |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, n_modes)`

---

## Covariance & Correlation

### MatrixPlan

*Build covariance, correlation, or distance matrices.*

```python
from warp_md import MatrixPlan

plan = MatrixPlan(selection)
matrix = plan.run(traj, system)
```

Also available as: `warp_md.analysis.covar()`, `warp_md.analysis.mwcovar()`, `warp_md.analysis.correl()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |

#### Output

- **Type**: 2D array
- **Shape**: `(n_atoms * 3, n_atoms * 3)` covariance/correlation matrix

---

### RadgyrTensorPlan

*The full Rg tensor — not just the scalar, but the shape.*

```python
from warp_md import RadgyrTensorPlan

plan = RadgyrTensorPlan(selection)
tensor = plan.run(traj, system)
```

Also available as: `warp_md.analysis.radgyr_tensor()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, 3)` — diagonalized Rg tensor components

---

## Dihedrals & Ring Pucker

Conformational analysis from backbone phi/psi to sugar pucker.

### DihedralPlan

*Measure a single dihedral angle across frames.*

```python
from warp_md import DihedralPlan

plan = DihedralPlan(selection)
angles = plan.run(traj, system)  # degrees
```

Also available as: `warp_md.analysis.dihedral()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | 4-atom selection defining dihedral |

#### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`
- **Unit**: degrees

---

### MultiDihedralPlan

*Multiple dihedrals at once — phi, psi, chi, you name it.*

```python
from warp_md import MultiDihedralPlan

plan = MultiDihedralPlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms defining multiple dihedrals |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, n_dihedrals)`
- **Unit**: degrees

---

### PuckerPlan / MultiPuckerPlan

*Ring pucker analysis — essential for nucleic acids and sugars.*

```python
from warp_md import PuckerPlan, MultiPuckerPlan

pucker = PuckerPlan(selection).run(traj, system)
multi = MultiPuckerPlan(selection).run(traj, system)
```

Also available as: `warp_md.analysis.pucker()`, `warp_md.analysis.multipucker()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Ring atom selection (5+ atoms) |

#### Output

- **Type**: 2D array (pucker amplitudes/phases per frame)
- **Shape**: `(n_frames, n_pucker_params)`

---

### DihedralRmsPlan

*RMSD in dihedral space — conformational deviation without superposition artifacts.*

```python
from warp_md import DihedralRmsPlan

plan = DihedralRmsPlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Dihedral atom selection |

#### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`

---

### RotateDihedralPlan / SetDihedralPlan

*Manipulate dihedral angles — rotate or set to specific values.*

```python
from warp_md import RotateDihedralPlan, SetDihedralPlan

RotateDihedralPlan(selection).run(traj, system)
SetDihedralPlan(selection).run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Dihedral atom selection |

#### Output

- Modifies coordinates in-place

---

### PermuteDihedralsPlan

*Permute dihedral angles to explore conformational space.*

```python
from warp_md import PermuteDihedralsPlan

plan = PermuteDihedralsPlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Dihedral atom selection |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, n_dihedrals)`

---

## DSSP (Secondary Structure)

*Helix, sheet, or coil — per residue, per frame.*

```python
from warp_md.analysis import dssp, dssp_allatoms, dssp_allresidues

# Per-residue assignment
ss = dssp(traj, system, selection)

# Per-atom DSSP labels
ss_atoms = dssp_allatoms(traj, system, selection)

# All residues at once
ss_all = dssp_allresidues(traj, system)
```

{% hint style="info" %}
DSSP uses phi/psi angle classification (H=helix, E=sheet, C=coil). CPU-only for now — it's already fast.
{% endhint %}

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` / `mask` | `Selection` | required | Backbone atoms for secondary structure assignment |
| `simplified` | `bool` | `False` | Collapse to 3-class (H/E/C) output |
| `dtype` | `str` | `"float"` | Output data type |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, n_residues)` — DSSP codes per residue per frame
- **Codes**: 0=none, 1=extended, 2=bridge, 3=3-10, 4=alpha, 5=pi, 6=turn, 7=bend

---

### KabschSanderPlan

*Kabsch-Sander secondary structure assignment — an alternative to DSSP using a different turn/h-bond classification.*

```python
from warp_md import KabschSanderPlan

plan = KabschSanderPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.kabsch_sander()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Backbone atoms |
| `energy_cutoff` | `float` | `-0.5` | H-bond energy cutoff (kcal/mol) |

#### Output

- **Type**: dict
- **Keys**: per-residue secondary structure assignment strings

---

## Shape & Order

### ShapeDescriptorsPlan

*Shape descriptors — asphericity, shape parameter, and relative shape anisotropy from the inertia tensor.*

```python
from warp_md import ShapeDescriptorsPlan

plan = ShapeDescriptorsPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.shape()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |
| `mass` | `bool` | `False` | Use mass-weighted inertia tensor |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, 3)` — asphericity, shape parameter, relative shape anisotropy

---

### NematicOrderPlan

*Nematic order parameter — orientational order of molecules or groups relative to a reference axis.*

```python
from warp_md import NematicOrderPlan

plan = NematicOrderPlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Tail/head indices for each molecule |
| `pbc` | `str` | `"orthorhombic"` | PBC handling mode |
| `reference_axis` | `list[float]` | `None` | Custom reference axis (3 floats) |

#### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`

---

### HelixOrientationPlan

*Helix orientation analysis — tilt, rotation, rise, radius, twist, and bending per frame and per residue.*

```python
from warp_md import HelixOrientationPlan

plan = HelixOrientationPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.helixorient()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | CA atoms of helical residues |

#### Output

- **Type**: dict with per-frame and per-residue helix geometry parameters

---

## Validation Plans

### CheckChiralityPlan

*Verify chirality hasn't flipped — catches simulation artifacts.*

```python
from warp_md import CheckChiralityPlan

result = CheckChiralityPlan(selection).run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Backbone atoms to check |

#### Output

- **Type**: dict with per-residue chirality flags

---

### CheckStructurePlan

*Structural sanity check — clashes, missing atoms, suspicious geometry.*

```python
from warp_md import CheckStructurePlan

result = CheckStructurePlan(selection).run(traj, system)
```

Also available as: `warp_md.analysis.check_structure()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to check |

#### Output

- **Type**: dict with clash counts, missing atom warnings, geometry outliers

---

## Diffusion in Dihedral Space

### TorsionDiffusionPlan / ToroidalDiffusionPlan

*Conformational diffusion — how fast do dihedrals explore their landscape?*

```python
from warp_md import TorsionDiffusionPlan, ToroidalDiffusionPlan

torsion_diff = TorsionDiffusionPlan(selection).run(traj, system)
toroidal_diff = ToroidalDiffusionPlan(selection).run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Dihedral atom selection |

#### Output

- **Type**: dict with diffusion coefficients and lag-time analysis

---

## Complete Structural Analysis Example

```python
from warp_md import (
    System, Trajectory,
    RmsfPlan, PcaPlan, PairwiseRmsdPlan, RmsdPerResPlan
)
from warp_md.analysis import dssp

system = System.from_file("protein.pdb")
traj = Trajectory.open_xtc("protein.xtc", system)
ca = system.select("name CA")
backbone = system.select("backbone")

# Flexibility map
rmsf = RmsfPlan(ca).run(traj, system)

# PCA — dominant motions
pca_result = PcaPlan(ca).run(traj, system)

# Conformational clustering
rmsd_matrix = PairwiseRmsdPlan(ca).run(traj, system)

# Per-residue deviation
per_res = RmsdPerResPlan(backbone).run(traj, system)

# Secondary structure timeline
ss = dssp(traj, system, backbone)

print(f"Max RMSF: {rmsf.max():.2f} Å — there's your flexible loop")
print("Your agent just profiled a protein's structural dynamics 🧬")
```
