---
description: GIST, density profiles, water shells, volumetric maps — where molecules go and why
icon: droplet
---

# Solvation & Density

Where does water like to hang out? How does solvent organize around your solute? What's the density profile across your membrane? This is the chapter that answers those questions.

{% hint style="info" %}
GIST (Grid Inhomogeneous Solvation Theory) is the crown jewel here — full solvation thermodynamics from trajectory data. But sometimes you just want a density profile, and that's here too.
{% endhint %}

---

## GIST — Solvation Thermodynamics

### GistGridPlan

*Grid-based GIST — computes solvation entropy and enthalpy on a 3D grid.*

```python
from warp_md import GistGridPlan

plan = GistGridPlan(selection)
result = plan.run(traj, system)
```

---

### GistDirectPlan

*Direct GIST — computes solvation properties without a grid intermediate.*

```python
from warp_md import GistDirectPlan

plan = GistDirectPlan(selection)
result = plan.run(traj, system)
```

---

### Functional GIST API

The functional API gives you more control over GIST configuration:

```python
from warp_md.analysis import gist, GistConfig, GistResult

config = GistConfig(
    # Configure grid, cutoffs, etc.
)
result: GistResult = gist(traj, system, config)
```

---

### PME Scaling

For production GIST with PME electrostatics:

```python
from warp_md import gist_apply_pme_scaling

# Apply PME correction to raw GIST energies
corrected = gist_apply_pme_scaling(raw_energies, system)
```

{% hint style="warning" %}
`gist_apply_pme_scaling` requires a build with GIST bindings. Rebuild with `maturin develop` if unavailable.
{% endhint %}

---

## Water Analysis

### WaterCountPlan

*Where does water like to hang out? 3D occupancy grid.*

```python
from warp_md import WaterCountPlan

plan = WaterCountPlan(
    water_selection=system.select("resname SOL and name OW"),
    center_selection=system.select("protein"),
    box_unit=(1.0, 1.0, 1.0),
    region_size=(30.0, 30.0, 30.0),
)
result = plan.run(traj, system)
```

#### Output (dict)

| Key | Description |
|-----|-------------|
| `dims` | Grid dimensions [nx, ny, nz] |
| `mean` | Mean occupancy per voxel |
| `std` | Standard deviation |
| `min`, `max` | Min/max occupancy |
| `first`, `last` | First/last frame snapshots |

---

### WatershellPlan

*Count waters in solvation shells — first shell, second shell, beyond.*

```python
from warp_md import WatershellPlan

plan = WatershellPlan(selection)
result = plan.run(traj, system)
```

---

### CountInVoxelPlan

*Count atoms in a voxel grid — not just water, anything.*

```python
from warp_md import CountInVoxelPlan

plan = CountInVoxelPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.count_in_voxel()`

---

### FreeVolumePlan

*Fraction of free volume (FFV) on a voxel grid.*

```python
from warp_md import FreeVolumePlan

# With auto-detection (recommended)
plan = FreeVolumePlan(
    selection=system.select("protein"),
    center_selection=system.select("protein"),
)
result = plan.run(traj, system)

# With explicit parameters
plan = FreeVolumePlan(
    selection=system.select("protein"),
    center_selection=system.select("protein"),
    box_unit=(1.0, 1.0, 1.0),      # Voxel size (Å), defaults to (1.0, 1.0, 1.0)
    region_size=(30.0, 30.0, 30.0), # Region extents (Å), auto-detected if None
    probe_radius=0.5,               # optional
)
result = plan.run(traj, system)
```

**Auto-detection:**
- `region_size` is computed from `center_selection` bounding box with 10% padding
- `box_unit` defaults to 1.0 Å if not specified
- Requires initial coordinates in the system (from PDB/GRO topology)

FFV is estimated per voxel as:

`FFV = free_frames / total_frames = 1 - occupied_frames / total_frames`

#### Output (dict)

| Key | Description |
|-----|-------------|
| `dims` | Grid dimensions `[nx, ny, nz]` |
| `mean` | Mean FFV per voxel (`0..1`) |
| `std` | FFV standard deviation |
| `first`, `last` | First/last frame free mask (0/1) |
| `min`, `max` | Per-voxel min/max over frames |

---

## Density Profiles

### DensityPlan

*Density profile along an axis — essential for membrane and interface systems.*

```python
from warp_md import DensityPlan

plan = DensityPlan(selection)
result = plan.run(traj, system)
```

---

### VolmapPlan

*Volumetric density map — 3D density grid for visualization.*

```python
from warp_md import VolmapPlan

plan = VolmapPlan(selection)
result = plan.run(traj, system)
```

---

## Surface Area

### SurfPlan

*LCPO surface area — fast analytical accessible surface area.*

```python
from warp_md import SurfPlan

plan = SurfPlan(selection)
area = plan.run(traj, system)  # Å²
```

Also available as: `warp_md.analysis.surf()`

---

### MolSurfPlan

*Molecular surface area — solvent-excluded surface.*

```python
from warp_md import MolSurfPlan

plan = MolSurfPlan(selection)
area = plan.run(traj, system)  # Å²
```

Also available as: `warp_md.analysis.molsurf()`

---

## Complete Solvation Analysis Example

```python
from warp_md import (
    System, Trajectory,
    WaterCountPlan, WatershellPlan,
    SurfPlan, DensityPlan
)

system = System.from_pdb("solvated_protein.pdb")
traj = Trajectory.open_xtc("solvated_protein.xtc", system)

protein = system.select("protein")
water_o = system.select("resname SOL and name OW")

# 3D water occupancy
water_grid = WaterCountPlan(
    water_selection=water_o,
    center_selection=protein,
    box_unit=(0.5, 0.5, 0.5),
    region_size=(40.0, 40.0, 40.0),
).run(traj, system)

# Solvation shells
shells = WatershellPlan(protein).run(traj, system)

# Surface area over time
sasa = SurfPlan(protein).run(traj, system)

# Density profile (e.g., membrane normal)
density = DensityPlan(system.select("all")).run(traj, system)

print(f"Mean SASA: {sasa.mean():.0f} Å²")
print("Your agent just mapped the solvation landscape 💧")
```
