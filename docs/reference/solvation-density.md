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

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms defining the grid region |

#### Output

- **Type**: dict with solvation thermodynamics (entropy, enthalpy, energy per voxel)

---

### GistDirectPlan

*Direct GIST — computes solvation properties without a grid intermediate.*

```python
from warp_md import GistDirectPlan

plan = GistDirectPlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Solute atoms for direct GIST |

#### Output

- **Type**: dict with direct solvation thermodynamics

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

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `water_selection` | `Selection` | required | Water atoms to count |
| `center_selection` | `Selection` | required | Reference region center |
| `box_unit` | `tuple[float]` | `(1.0, 1.0, 1.0)` | Voxel size (Å) |
| `region_size` | `tuple[float]` | required | Grid extents (Å) |

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

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `target` | `Selection` | required | Solute atoms |
| `probe` | `Selection` | required | Solvent atoms (water OW) |
| `cutoff` | `float` | `3.5` | Shell cutoff (Å) |
| `pbc` | `str` | `"orthorhombic"` | PBC handling mode |

#### Output

- **Type**: dict with shell populations (first shell, second shell, bulk)

---

### HydrationOrderPlan

*Hydration order parameter — tetrahedral order of water around a solute based on angle/distance distributions.*

```python
from warp_md import HydrationOrderPlan

plan = HydrationOrderPlan(selection, axis="z", bin=1.0, tblock=1)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.hydorder()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Solute atoms |
| `axis` | `str` | `"z"` | Profile axis |
| `bin` | `float` | `1.0` | Bin width (Å) |

#### Output

- **Type**: 2D array — hydration order profile along axis

---

### WaterOrderPlan

*Water ordering analysis — orientational order of water molecules relative to an interface or axis.*

```python
from warp_md import WaterOrderPlan

plan = WaterOrderPlan(oxygen_indices, hydrogen1_indices, hydrogen2_indices, charges, axis="z", bin=0.25)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.h2order()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `oxygen_indices` | `list[int]` | required | Water oxygen atom indices |
| `hydrogen1_indices` | `list[int]` | required | Water H1 atom indices |
| `hydrogen2_indices` | `list[int]` | required | Water H2 atom indices |
| `charges` | `list[float]` | required | Partial charges per atom |
| `axis` | `str` | `"z"` | Profile axis |
| `bin` | `float` | `0.25` | Bin width (Å) |

#### Output

- **Type**: dict with orientational order profiles

---

### CountInVoxelPlan

*Count atoms in a voxel grid — not just water, anything.*

```python
from warp_md import CountInVoxelPlan

plan = CountInVoxelPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.count_in_voxel()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to count |

#### Output

- **Type**: dict with voxel count grid

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

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms defining occupied volume |
| `center_selection` | `Selection` | required | Reference for grid centering |
| `box_unit` | `tuple[float]` | `(1.0, 1.0, 1.0)` | Voxel size (Å) |
| `region_size` | `tuple[float]` | auto | Grid extents (Å) |
| `probe_radius` | `float` | `0.5` | Probe radius for free volume (Å) |

#### Output (dict)

| Key | Description |
|-----|-------------|
| `dims` | Grid dimensions `[nx, ny, nz]` |
| `mean` | Mean FFV per voxel (`0..1`) |
| `std` | FFV standard deviation |
| `first`, `last` | First/last frame free mask (0/1) |
| `min`, `max` | Per-voxel min/max over frames |

---

### BondiFfvPlan

*Bondi free volume fraction — uses Bondi's van der Waals radii to estimate free volume on a grid.*

```python
from warp_md import BondiFfvPlan

plan = BondiFfvPlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |

#### Output

- **Type**: dict with Bondi free volume fraction grid

---

## Density Profiles

### DensityPlan

*Density profile along an axis — essential for membrane and interface systems.*

```python
from warp_md import DensityPlan

plan = DensityPlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to profile |
| `axis` | `str` | `"z"` | Profile axis |

#### Output

- **Type**: 2D array — density profile (axis bin centers, density values)

---

### LinearDensityPlan

*Linear density profile — mass or number density along a single axis with per-species decomposition.*

```python
from warp_md import LinearDensityPlan

plan = LinearDensityPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.lineardensity()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to profile |
| `average_axis` | `int` | `2` | Axis for averaging (0=x, 1=y, 2=z) |
| `bin` | `float` | `1.0` | Bin width (Å) |

#### Output

- **Type**: dict with per-species density profiles

---

### VolmapPlan

*Volumetric density map — 3D density grid for visualization.*

```python
from warp_md import VolmapPlan

plan = VolmapPlan(selection)
result = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to map |
| `resolution` | `float` | `1.0` | Grid resolution (Å) |

#### Output

- **Type**: 3D density grid array

---

## Solvent Organization

### SolventOrientationPlan

*Radial solvent orientation profiles — how solvent molecules orient around a solute, measured via cos(θ) distributions.*

```python
from warp_md import SolventOrientationPlan

plan = SolventOrientationPlan(
    solute_selection,
    atom1_indices, atom2_indices, atom3_indices,
    r_min=0.0, r_max=0.5, cbin=0.02, rbin=0.02,
)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.sorient()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `solute_selection` | `Selection` | required | Solute atoms |
| `atom1_indices` | `list[int]` | required | Solvent O indices |
| `atom2_indices` | `list[int]` | required | Solvent H1 indices |
| `atom3_indices` | `list[int]` | required | Solvent H2 indices |
| `r_min` | `float` | `0.0` | Minimum radial bin (Å) |
| `r_max` | `float` | `0.5` | Maximum radial bin (Å) |
| `cbin` | `float` | `0.02` | cos(θ) bin width |
| `rbin` | `float` | `0.02` | Radial bin width (Å) |

#### Output

- **Type**: dict with 2D orientation distribution (radial × cos(θ))

---

### SolventPolarizationPlan

*Solvent polarization analysis — radial dipole density, shell counts, and polarization profiles around a solute.*

```python
from warp_md import SolventPolarizationPlan

plan = SolventPolarizationPlan(
    solute_selection,
    atom1_indices, atom2_indices, atom3_indices,
    charges, r_min=0.0, r_max=0.32, bin=0.01,
)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.spol()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `solute_selection` | `Selection` | required | Solute atoms |
| `atom1_indices` | `list[int]` | required | Solvent O indices |
| `atom2_indices` | `list[int]` | required | Solvent H1 indices |
| `atom3_indices` | `list[int]` | required | Solvent H2 indices |
| `charges` | `list[float]` | required | Partial charges |
| `r_min` | `float` | `0.0` | Minimum radial bin (Å) |
| `r_max` | `float` | `0.32` | Maximum radial bin (Å) |
| `bin` | `float` | `0.01` | Radial bin width (Å) |

#### Output

- **Type**: dict with radial dipole density and shell polarization profiles

---

### HydrophobicDefectPlan

*Hydrophobic defect analysis — computes the hydrophobic defect (tilted area) of lipid leaflets for membrane systems.*

```python
from warp_md import HydrophobicDefectPlan

plan = HydrophobicDefectPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.hydrophobic_defect()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Lipid tail atoms |

#### Output

- **Type**: dict with hydrophobic defect area per frame

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

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to compute SASA for |

#### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`
- **Unit**: Angstrom²

---

### MolSurfPlan

*Molecular surface area — solvent-excluded surface.*

```python
from warp_md import MolSurfPlan

plan = MolSurfPlan(selection)
area = plan.run(traj, system)  # Å²
```

Also available as: `warp_md.analysis.molsurf()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to compute molecular surface for |

#### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`
- **Unit**: Angstrom²

---

## Complete Solvation Analysis Example

```python
from warp_md import (
    System, Trajectory,
    WaterCountPlan, WatershellPlan,
    SurfPlan, DensityPlan
)

system = System.from_file("solvated_protein.pdb")
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
