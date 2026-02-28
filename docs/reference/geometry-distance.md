---
description: Distances, angles, vectors, COM, principal axes — measure anything between atoms
icon: ruler-combined
---

# Geometry & Distance

Your agent's precision measurement toolkit. If it exists in 3D space, warp-md can measure it.

{% hint style="info" %}
All distances in Angstrom. All angles in degrees (unless noted). No exceptions.
{% endhint %}

---

## Distance Plans

### DistancePlan

*Distance between atom pairs across frames.*

```python
from warp_md import DistancePlan

plan = DistancePlan(selection)
distances = plan.run(traj, system)
```

Also available as: `warp_md.analysis.dist()`

---

### PairwiseDistancePlan

*All-vs-all pairwise distances — the distance matrix.*

```python
from warp_md import PairwiseDistancePlan

plan = PairwiseDistancePlan(selection)
matrix = plan.run(traj, system)
```

---

### PairDistPlan

*Pair distribution — distance histogram between atom groups.*

```python
from warp_md import PairDistPlan

plan = PairDistPlan(selection)
result = plan.run(traj, system)
```

---

### DistanceToPointPlan

*How far is each atom from a fixed point in space?*

```python
from warp_md import DistanceToPointPlan

plan = DistanceToPointPlan(selection)
result = plan.run(traj, system)
```

---

### DistanceToReferencePlan

*Distance from each frame to a reference structure.*

```python
from warp_md import DistanceToReferencePlan

plan = DistanceToReferencePlan(selection)
result = plan.run(traj, system)
```

---

### LowestCurvePlan

*Find the lowest energy path — useful for PMF/free energy analysis.*

```python
from warp_md import LowestCurvePlan

plan = LowestCurvePlan(selection)
result = plan.run(traj, system)
```

---

## Angle Plans

### AnglePlan

*Measure bond angles (3-atom angle) across frames.*

```python
from warp_md import AnglePlan

plan = AnglePlan(selection)
angles = plan.run(traj, system)  # degrees
```

Also available as: `warp_md.analysis.angle()`

---

## Vector Plans

### VectorPlan

*Compute vectors between atom pairs.*

```python
from warp_md import VectorPlan

plan = VectorPlan(selection)
vectors = plan.run(traj, system)
```

Also available as: `warp_md.analysis.vector()`, `warp_md.analysis.vector_mask()`

---

## Centers of Mass / Geometry

### CenterOfMassPlan

*Mass-weighted center of a selection.*

```python
from warp_md import CenterOfMassPlan

plan = CenterOfMassPlan(selection)
com = plan.run(traj, system)  # (n_frames, 3) in Å
```

---

### CenterOfGeometryPlan

*Geometric center (unweighted) of a selection.*

```python
from warp_md import CenterOfGeometryPlan

plan = CenterOfGeometryPlan(selection)
cog = plan.run(traj, system)  # (n_frames, 3) in Å
```

---

## Principal Axes

### PrincipalAxesPlan

*Moments of inertia and principal axes — the shape of your selection.*

```python
from warp_md import PrincipalAxesPlan

plan = PrincipalAxesPlan(selection)
result = plan.run(traj, system)
```

---

## Neighbor Search & Contacts

### MindistPlan

*Minimum distance between two groups of atoms.*

```python
from warp_md import MindistPlan

plan = MindistPlan(selection)
result = plan.run(traj, system)
```

---

### HausdorffPlan

*Hausdorff distance — the maximum of the minimum distances. Measures how "different" two shapes are.*

```python
from warp_md import HausdorffPlan

plan = HausdorffPlan(selection)
result = plan.run(traj, system)
```

---

### ClosestAtomPlan

*Find the closest atom from one group to another.*

```python
from warp_md import ClosestAtomPlan

plan = ClosestAtomPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.closest_atom()`

---

### ClosestPlan

*Find closest N waters/molecules to a reference.*

```python
from warp_md import ClosestPlan

plan = ClosestPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.closest()`

---

### SearchNeighborsPlan

*Find all neighbors within a cutoff distance.*

```python
from warp_md import SearchNeighborsPlan

plan = SearchNeighborsPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.search_neighbors()`

---

### NativeContactsPlan

*Fraction of native contacts preserved — protein folding metric.*

```python
from warp_md import NativeContactsPlan

plan = NativeContactsPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.native_contacts()`

---

## Mapping & Misc

### AtomMapPlan

*Map atoms between different topologies or selections.*

```python
from warp_md import AtomMapPlan

plan = AtomMapPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.atom_map()`

---

### RandomizeIonsPlan

*Randomize ion positions — useful for initial configuration setup.*

```python
from warp_md import RandomizeIonsPlan

plan = RandomizeIonsPlan(selection)
result = plan.run(traj, system)
```

---

## Volume

### VolumePlan

*Box volume over time — tracks compression/expansion.*

```python
from warp_md import VolumePlan

plan = VolumePlan(selection)
volumes = plan.run(traj, system)  # Å³
```

---

## Complete Geometry Example

```python
from warp_md import (
    System, Trajectory,
    DistancePlan, AnglePlan, CenterOfMassPlan,
    MindistPlan, NativeContactsPlan
)

system = System.from_pdb("complex.pdb")
traj = Trajectory.open_xtc("complex.xtc", system)

protein = system.select("protein")
ligand = system.select("resname LIG")

# Track protein-ligand minimum distance
mindist = MindistPlan(protein).run(traj, system)

# Center of mass trajectory
com = CenterOfMassPlan(ligand).run(traj, system)

# Native contacts (folding analysis)
qnative = NativeContactsPlan(protein).run(traj, system)

print("Your agent just mapped the geometry of a protein-ligand complex 🎯")
```
