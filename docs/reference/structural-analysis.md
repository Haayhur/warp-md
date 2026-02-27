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

---

### DistanceRmsdPlan

*No superposition needed — compares internal distance matrices.*

```python
from warp_md import DistanceRmsdPlan

plan = DistanceRmsdPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.distance_rmsd()`

---

### PairwiseRmsdPlan

*All-vs-all RMSD matrix. Perfect for clustering or identifying conformational states.*

```python
from warp_md import PairwiseRmsdPlan

plan = PairwiseRmsdPlan(selection)
matrix = plan.run(traj, system)  # (n_frames, n_frames)
```

Also available as: `warp_md.analysis.pairwise_rmsd()`

---

### RmsdPerResPlan

*Which residues are moving the most? Per-residue RMSD answers that.*

```python
from warp_md import RmsdPerResPlan

plan = RmsdPerResPlan(selection)
result = plan.run(traj, system)  # (n_frames, n_residues)
```

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

---

### BfactorsPlan

*Crystallographic B-factors from dynamics — compare simulation to experiment.*

```python
from warp_md import BfactorsPlan

plan = BfactorsPlan(selection)
bfactors = plan.run(traj, system)  # (n_atoms,) in Å²
```

Also available as: `warp_md.analysis.bfactors()`

---

### AtomicFluctPlan

*Raw atomic fluctuations — the foundation RMSF and B-factors are built on.*

```python
from warp_md import AtomicFluctPlan

plan = AtomicFluctPlan(selection)
fluct = plan.run(traj, system)
```

Also available as: `warp_md.analysis.atomicfluct()`

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

---

### AnalyzeModesPlan

*Analyze normal modes or PCA modes — eigenvalues, eigenvectors, the works.*

```python
from warp_md import AnalyzeModesPlan

plan = AnalyzeModesPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.analyze_modes()`

---

### ProjectionPlan

*Project trajectory onto PCA or normal mode eigenvectors.*

```python
from warp_md import ProjectionPlan

plan = ProjectionPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.projection()`

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

---

### RadgyrTensorPlan

*The full Rg tensor — not just the scalar, but the shape.*

```python
from warp_md import RadgyrTensorPlan

plan = RadgyrTensorPlan(selection)
tensor = plan.run(traj, system)
```

Also available as: `warp_md.analysis.radgyr_tensor()`

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

---

### MultiDihedralPlan

*Multiple dihedrals at once — phi, psi, chi, you name it.*

```python
from warp_md import MultiDihedralPlan

plan = MultiDihedralPlan(selection)
result = plan.run(traj, system)
```

---

### PuckerPlan / MultiPuckerPlan

*Ring pucker analysis — essential for nucleic acids and sugars.*

```python
from warp_md import PuckerPlan, MultiPuckerPlan

pucker = PuckerPlan(selection).run(traj, system)
multi = MultiPuckerPlan(selection).run(traj, system)
```

Also available as: `warp_md.analysis.pucker()`, `warp_md.analysis.multipucker()`

---

### DihedralRmsPlan

*RMSD in dihedral space — conformational deviation without superposition artifacts.*

```python
from warp_md import DihedralRmsPlan

plan = DihedralRmsPlan(selection)
result = plan.run(traj, system)
```

---

### RotateDihedralPlan / SetDihedralPlan

*Manipulate dihedral angles — rotate or set to specific values.*

```python
from warp_md import RotateDihedralPlan, SetDihedralPlan

RotateDihedralPlan(selection).run(traj, system)
SetDihedralPlan(selection).run(traj, system)
```

---

### PermuteDihedralsPlan

*Permute dihedral angles to explore conformational space.*

```python
from warp_md import PermuteDihedralsPlan

plan = PermuteDihedralsPlan(selection)
result = plan.run(traj, system)
```

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

---

## Validation Plans

### CheckChiralityPlan

*Verify chirality hasn't flipped — catches simulation artifacts.*

```python
from warp_md import CheckChiralityPlan

result = CheckChiralityPlan(selection).run(traj, system)
```

---

### CheckStructurePlan

*Structural sanity check — clashes, missing atoms, suspicious geometry.*

```python
from warp_md import CheckStructurePlan

result = CheckStructurePlan(selection).run(traj, system)
```

Also available as: `warp_md.analysis.check_structure()`

---

## Diffusion in Dihedral Space

### TorsionDiffusionPlan / ToroidalDiffusionPlan

*Conformational diffusion — how fast do dihedrals explore their landscape?*

```python
from warp_md import TorsionDiffusionPlan, ToroidalDiffusionPlan

torsion_diff = TorsionDiffusionPlan(selection).run(traj, system)
toroidal_diff = ToroidalDiffusionPlan(selection).run(traj, system)
```

---

## Complete Structural Analysis Example

```python
from warp_md import (
    System, Trajectory,
    RmsfPlan, PcaPlan, PairwiseRmsdPlan, RmsdPerResPlan
)
from warp_md.analysis import dssp

system = System.from_pdb("protein.pdb")
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
