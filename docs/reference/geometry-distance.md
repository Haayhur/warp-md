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

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atom pairs defining distances |
| `pbc` | `str` | `"orthorhombic"` | PBC handling mode |
| `mass_weighted` | `bool` | `False` | Weight distances by atomic mass |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, n_pairs)`
- **Unit**: Angstrom

---

### PairwiseDistancePlan

*All-vs-all pairwise distances — the distance matrix.*

```python
from warp_md import PairwiseDistancePlan

plan = PairwiseDistancePlan(selection)
matrix = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to compute pairwise distances for |

#### Output

- **Type**: 3D array
- **Shape**: `(n_frames, n_atoms, n_atoms)`
- **Unit**: Angstrom

---

### PairDistPlan

*Pair distribution — distance histogram between atom groups.*

```python
from warp_md import PairDistPlan

plan = PairDistPlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `sel_a` | `Selection` | required | First atom group |
| `sel_b` | `Selection` | required | Second atom group |
| `bins` | `int` | `200` | Number of histogram bins |
| `r_max` | `float` | `10.0` | Maximum distance (Å) |
| `pbc` | `str` | `"orthorhombic"` | PBC handling |

#### Output

- **Type**: tuple `(r, g, counts)` — bin centers, distribution, raw counts

---

### DistanceToPointPlan

*How far is each atom from a fixed point in space?*

```python
from warp_md import DistanceToPointPlan

plan = DistanceToPointPlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to measure |
| `point` | `list[float]` | required | Fixed point coordinates `[x, y, z]` |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, n_atoms)`
- **Unit**: Angstrom

---

### DistanceToReferencePlan

*Distance from each frame to a reference structure.*

```python
from warp_md import DistanceToReferencePlan

plan = DistanceToReferencePlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to compare |
| `reference` | `str` | `"topology"` | Reference (`"topology"` or `"frame0"`) |

#### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`
- **Unit**: Angstrom

---

### LowestCurvePlan

*Find the lowest energy path — useful for PMF/free energy analysis.*

```python
from warp_md import LowestCurvePlan

plan = LowestCurvePlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |

#### Output

- **Type**: dict with path coordinates and energies

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

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | 3-atom selection defining the angle |

#### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`
- **Unit**: degrees

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

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atom pairs defining vectors |

#### Output

- **Type**: 3D array
- **Shape**: `(n_frames, n_pairs, 3)`
- **Unit**: Angstrom (vector components)

---

## Centers of Mass / Geometry

### CenterOfMassPlan

*Mass-weighted center of a selection.*

```python
from warp_md import CenterOfMassPlan

plan = CenterOfMassPlan(selection)
com = plan.run(traj, system)  # (n_frames, 3) in Å
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to compute COM for |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, 3)`
- **Unit**: Angstrom

---

### CenterOfGeometryPlan

*Geometric center (unweighted) of a selection.*

```python
from warp_md import CenterOfGeometryPlan

plan = CenterOfGeometryPlan(selection)
cog = plan.run(traj, system)  # (n_frames, 3) in Å
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to compute COG for |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, 3)`
- **Unit**: Angstrom

---

## Principal Axes

### PrincipalAxesPlan

*Moments of inertia and principal axes — the shape of your selection.*

```python
from warp_md import PrincipalAxesPlan

plan = PrincipalAxesPlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |
| `mass_weighted` | `bool` | `False` | Use mass-weighted inertia tensor |

#### Output

- **Type**: dict with eigenvalues and eigenvectors of inertia tensor

---

## Neighbor Search & Contacts

### MindistPlan

*Minimum distance between two groups of atoms.*

```python
from warp_md import MindistPlan

plan = MindistPlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `sel_a` | `Selection` | required | First atom group |
| `sel_b` | `Selection` | required | Second atom group |
| `pbc` | `str` | `"orthorhombic"` | PBC handling mode |

#### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`
- **Unit**: Angstrom

---

### HausdorffPlan

*Hausdorff distance — the maximum of the minimum distances. Measures how "different" two shapes are.*

```python
from warp_md import HausdorffPlan

plan = HausdorffPlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `sel_a` | `Selection` | required | First atom group |
| `sel_b` | `Selection` | required | Second atom group |
| `pbc` | `str` | `"orthorhombic"` | PBC handling mode |

#### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`
- **Unit**: Angstrom

---

### MaxdistPlan

*Maximum distance between two groups of atoms — the farthest pair per frame.*

```python
from warp_md import MaxdistPlan

plan = MaxdistPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.maxdist()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `sel_a` | `Selection` | required | First atom group |
| `sel_b` | `Selection` | required | Second atom group |
| `pbc` | `str` | `"orthorhombic"` | PBC handling mode |

#### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`
- **Unit**: Angstrom

---

### ClosestAtomPlan

*Find the closest atom from one group to another.*

```python
from warp_md import ClosestAtomPlan

plan = ClosestAtomPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.closest_atom()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atom groups (target + probe) |
| `pbc` | `str` | `"orthorhombic"` | PBC handling mode |

#### Output

- **Type**: tuple of (atom indices, distances)

---

### ClosestPlan

*Find closest N waters/molecules to a reference.*

```python
from warp_md import ClosestPlan

plan = ClosestPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.closest()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `target` | `Selection` | required | Reference selection |
| `probe` | `Selection` | required | Probe selection to search |
| `n_solvents` | `int` | required | Number of closest molecules to find |
| `pbc` | `str` | `"orthorhombic"` | PBC handling mode |

#### Output

- **Type**: dict with closest atom indices, distances, and coordinates

---

### SearchNeighborsPlan

*Find all neighbors within a cutoff distance.*

```python
from warp_md import SearchNeighborsPlan

plan = SearchNeighborsPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.search_neighbors()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `target` | `Selection` | required | Reference selection |
| `probe` | `Selection` | required | Probe selection to search |
| `cutoff` | `float` | required | Distance cutoff (Å) |
| `pbc` | `str` | `"orthorhombic"` | PBC handling mode |

#### Output

- **Type**: list of neighbor indices per frame

---

### NativeContactsPlan

*Fraction of native contacts preserved — protein folding metric.*

```python
from warp_md import NativeContactsPlan

plan = NativeContactsPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.native_contacts()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `mask` | `Selection` | required | Atom selection |
| `ref` | `str` or `int` | `"topology"` | Reference frame for native contacts |
| `distance` | `float` | `5.0` | Contact cutoff (Å) |
| `image` | `bool` | `True` | Use PBC imaging |
| `mask2` | `Selection` | `None` | Optional second selection |
| `mindist` | `float` | `None` | Minimum distance cutoff |
| `maxdist` | `float` | `None` | Maximum distance cutoff |

#### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`
- **Unit**: fraction of native contacts preserved

---

### SaltBridgePlan

*Salt bridge analysis — detect ion pairs between charged residues (e.g., Arg-COO⁻, Lys-COO⁻) within a distance cutoff.*

```python
from warp_md import SaltBridgePlan

plan = SaltBridgePlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.saltbr()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atom selection including charged groups |

#### Output

- **Type**: dict with salt bridge persistence and distances

---

## Running Average

### RunningAveragePlan

*Running average of coordinates over a window — smooths trajectory noise.*

```python
from warp_md import RunningAveragePlan

plan = RunningAveragePlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.runningavg()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to average |
| `window` | `int` | `None` | Fixed window size; omit for cumulative average |

#### Output

- **Type**: 2D array — smoothed coordinates `(n_frames, n_atoms * 3)`

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

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to map |

#### Output

- **Type**: dict mapping atom indices between topologies

---

### RandomizeIonsPlan

*Randomize ion positions — useful for initial configuration setup.*

```python
from warp_md import RandomizeIonsPlan

plan = RandomizeIonsPlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Ion selection to randomize |

#### Output

- Modifies coordinates in-place

---

## Volume

### VolumePlan

*Box volume over time — tracks compression/expansion.*

```python
from warp_md import VolumePlan

plan = VolumePlan(selection)
volumes = plan.run(traj, system)  # Å³
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms (uses box vectors from trajectory) |

#### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`
- **Unit**: Angstrom³

---

## Complete Geometry Example

```python
from warp_md import (
    System, Trajectory,
    DistancePlan, AnglePlan, CenterOfMassPlan,
    MindistPlan, NativeContactsPlan
)

system = System.from_file("complex.pdb")
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
