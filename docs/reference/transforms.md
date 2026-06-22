---
description: Align, center, image, strip, superpose — trajectory surgery without re-running simulations
icon: wand-magic-sparkles
---

# Transforms & Imaging

Sometimes your trajectory needs surgery before analysis. Centering, imaging, alignment, stripping — warp-md can reshape coordinates without touching the original simulation.

{% hint style="info" %}
Transform Plans modify coordinates in-place or produce new coordinate sets. They're essential preprocessing for many analyses.
{% endhint %}

---

## Alignment & Superposition

### AlignPlan

*Align all frames to a reference structure. The workhorse of trajectory preprocessing.*

```python
from warp_md import AlignPlan

plan = AlignPlan(selection)
plan.run(traj, system)
```

Also available as: `warp_md.analysis.align()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms used for alignment |

#### Output

- Modifies coordinates in-place

---

### SuperposePlan

*Superpose frames with fine-grained control over reference and target.*

```python
from warp_md import SuperposePlan

plan = SuperposePlan(selection)
plan.run(traj, system)
```

Also available as: `warp_md.analysis.superpose()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms for superposition |

#### Output

- Modifies coordinates in-place

---

### RotationMatrixPlan

*Extract the optimal rotation matrix per frame — useful if you need the transformation itself.*

```python
from warp_md import RotationMatrixPlan

plan = RotationMatrixPlan(selection)
result = plan.run(traj, system)  # (n_frames, 3, 3)
```

Also available as: `warp_md.analysis.rotation_matrix()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms for rotation fitting |
| `reference_mode` | `str` | `"topology"` | Reference structure mode |
| `mass_weighted` | `bool` | `False` | Mass-weight the fit |

#### Output

- **Type**: 3D array
- **Shape**: `(n_frames, 3, 3)`
- **Unit**: rotation matrix (orthonormal)

---

### AlignPrincipalAxisPlan

*Align the principal axis of a selection to a lab axis. Great for elongated molecules.*

```python
from warp_md import AlignPrincipalAxisPlan

plan = AlignPrincipalAxisPlan(selection)
plan.run(traj, system)
```

Also available as: `warp_md.analysis.align_principal_axis()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to compute principal axis from |
| `mass_weighted` | `bool` | `False` | Mass-weight the inertia tensor |

#### Output

- Modifies coordinates in-place

---

## Centering & Translation

### CenterTrajectoryPlan

*Center a selection at the box center (or origin).*

```python
from warp_md import CenterTrajectoryPlan

plan = CenterTrajectoryPlan(selection)
plan.run(traj, system)
```

Also available as: `warp_md.analysis.center()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to center |
| `center_at` | `str` | `"box"` | Centering target (`"box"` or `"origin"`) |
| `mass_weighted` | `bool` | `False` | Use COM rather than COG |

#### Output

- Modifies coordinates in-place

---

### TranslatePlan

*Translate all coordinates by a vector.*

```python
from warp_md import TranslatePlan

plan = TranslatePlan(selection)
plan.run(traj, system)
```

Also available as: `warp_md.analysis.translate()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to translate |

#### Output

- Modifies coordinates in-place

---

## Rotation & Scaling

### RotatePlan

*Rotate coordinates by a rotation matrix or angle.*

```python
from warp_md import RotatePlan

plan = RotatePlan(selection)
plan.run(traj, system)
```

Also available as: `warp_md.analysis.rotate()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to rotate |

#### Output

- Modifies coordinates in-place

---

### ScalePlan

*Scale coordinates — useful for unit conversion or box rescaling.*

```python
from warp_md import ScalePlan

plan = ScalePlan(selection)
plan.run(traj, system)
```

Also available as: `warp_md.analysis.scale()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to scale |
| `scale_factor` | `float` | required | Scaling factor |

#### Output

- Modifies coordinates in-place

---

### TransformPlan

*Apply a general 4×4 transformation matrix to coordinates.*

```python
from warp_md import TransformPlan

plan = TransformPlan(selection)
plan.run(traj, system)
```

Also available as: `warp_md.analysis.transform()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to transform |

#### Output

- Modifies coordinates in-place

---

## PBC Imaging

### ImagePlan

*Apply periodic boundary conditions — wrap atoms back into the box.*

```python
from warp_md import ImagePlan

plan = ImagePlan(selection)
plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to image |
| `pbc` | `str` | `"orthorhombic"` | PBC handling mode |

#### Output

- Modifies coordinates in-place

---

### AutoImagePlan

*Automatic PBC imaging — intelligently handles molecules split across boundaries.*

```python
from warp_md import AutoImagePlan

plan = AutoImagePlan(selection)
plan.run(traj, system)
```

Also available as: `warp_md.analysis.autoimage()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to auto-image |

#### Output

- Modifies coordinates in-place

{% hint style="warning" %}
**Order matters**: For many analyses, the recommended preprocessing pipeline is: `AutoImage → Center → Align`. Running RMSD on unimaged trajectories invites heartbreak.
{% endhint %}

---

### FixImageBondsPlan

*Fix bonds broken by PBC imaging — reconnects molecule fragments.*

```python
from warp_md import FixImageBondsPlan

plan = FixImageBondsPlan(selection)
plan.run(traj, system)
```

Also available as: `warp_md.analysis.fiximagedbonds()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to fix broken bonds for |

#### Output

- Modifies coordinates in-place

---

## Cell Operations

### ReplicateCellPlan

*Replicate the unit cell — build supercells for visualization or analysis.*

```python
from warp_md import ReplicateCellPlan

plan = ReplicateCellPlan(selection)
plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to replicate |

#### Output

- Produces new coordinates with replicated cell

---

### XtalSymmPlan

*Apply crystallographic symmetry operations.*

```python
from warp_md import XtalSymmPlan

plan = XtalSymmPlan(selection)
plan.run(traj, system)
```

Also available as: `warp_md.analysis.xtalsymm()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to transform |

#### Output

- Produces new coordinates with symmetry copies

---

## Structure Manipulation

### StripPlan

*Remove atoms from the trajectory — strip water, ions, whatever you don't need.*

```python
from warp_md import StripPlan

plan = StripPlan(selection)
plan.run(traj, system)
```

Also available as: `warp_md.analysis.strip()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to keep (inverse strips) |

#### Output

- Modifies trajectory atom count in-place

---

### MeanStructurePlan

*Average structure across all frames — the "consensus" conformation.*

```python
from warp_md import MeanStructurePlan

plan = MeanStructurePlan(selection)
mean = plan.run(traj, system)
```

Also available as: `warp_md.analysis.mean_structure()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to average |

#### Output

- **Type**: 2D array — average coordinates `(n_atoms, 3)`

---

### AverageFramePlan

*Average coordinates for a specific frame range.*

```python
from warp_md import AverageFramePlan

plan = AverageFramePlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.get_average_frame()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to average |

#### Output

- **Type**: 2D array — averaged coordinates `(n_atoms, 3)`

---

### MakeStructurePlan

*Create a new structure from coordinates — build topology from scratch.*

```python
from warp_md import MakeStructurePlan

plan = MakeStructurePlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.make_structure()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms for new structure |

#### Output

- **Type**: `System` object

---

## Velocity Operations

### GetVelocityPlan

*Extract velocities from frames that have them.*

```python
from warp_md import GetVelocityPlan

plan = GetVelocityPlan(selection)
velocities = plan.run(traj, system)
```

Also available as: `warp_md.analysis.get_velocity()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to extract velocities for |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, n_atoms * 3)` — velocity components

---

### SetVelocityPlan

*Set velocities — useful for NVE simulations or thermodynamic integration.*

```python
from warp_md import SetVelocityPlan

plan = SetVelocityPlan(selection)
plan.run(traj, system)
```

Also available as: `warp_md.analysis.set_velocity()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to set velocities for |

#### Output

- Modifies trajectory velocities in-place

---

## Complete Preprocessing Pipeline

```python
from warp_md import (
    System, Trajectory,
    AutoImagePlan, CenterTrajectoryPlan, AlignPlan,
    StripPlan, RmsdPlan
)

system = System.from_file("raw_simulation.pdb")
traj = Trajectory.open_xtc("raw_simulation.xtc", system)
protein = system.select("protein")

# The canonical preprocessing pipeline
AutoImagePlan(protein).run(traj, system)       # Fix PBC artifacts
CenterTrajectoryPlan(protein).run(traj, system) # Center protein in box
AlignPlan(protein).run(traj, system)            # Align to reference

# Now your analysis is clean
rmsd = RmsdPlan(protein).run(traj, system)

print("Raw trajectory → analysis-ready in 4 lines 🔧")
```
