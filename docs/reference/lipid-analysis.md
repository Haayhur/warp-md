---
description: Membrane properties, lipid lateral diffusion, order parameters, and leaflets
icon: layers
---

# Lipid Analysis

A specialized suite of plans for analyzing lipid membranes, bilayers, and monolayers. These plans handle leaflet assignment, lateral diffusion, area per lipid, order parameters, and inter-leaflet dynamics.

---

## Area & Density

### LipidAreaPlan

*Calculates the Area Per Lipid (APL) across a membrane patch.*

```python
from warp_md import LipidAreaPlan

plan = LipidAreaPlan(
    selection,
    group_by="resid"
)
area = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Lipid atoms to analyze |
| `group_by` | `str` | `"resid"` | Grouping mode for per-lipid areas |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, n_lipids)`
- **Unit**: Angstrom² per lipid

---

### LipidMembraneThicknessPlan

*Computes the overall thickness of the membrane.*

```python
from warp_md import LipidMembraneThicknessPlan

plan = LipidMembraneThicknessPlan(
    selection,
    group_by="resid"
)
thickness = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Membrane atoms |
| `group_by` | `str` | `"resid"` | Grouping mode |

#### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`
- **Unit**: Angstrom

---

### LipidZThicknessPlan

*Measures the thickness of individual lipid leaflets along the bilayer normal (Z-axis).*

```python
from warp_md import LipidZThicknessPlan

plan = LipidZThicknessPlan(selection)
thickness = plan.run(traj, system)
```

Also available as: `warp_md.analysis.lipid_z_thickness()`

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Lipid atoms for thickness measurement |

#### Output

- **Type**: 2D array — per-lipid Z-thickness per frame
- **Shape**: `(n_frames, n_lipids)`
- **Unit**: Angstrom

---

## Leaflet Assignment

### LipidLeafletPlan

*Automatically assigns lipids to upper and lower leaflets.*

```python
from warp_md import LipidLeafletPlan

plan = LipidLeafletPlan(
    selection,
    group_by="resid"
)
leaflets = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Lipid atoms for leaflet assignment |
| `group_by` | `str` | `"resid"` | Grouping mode |

#### Output

- **Type**: dict with leaflet assignments per lipid per frame

---

### LipidCurvedLeafletPlan

*Assigns lipids to leaflets in curved membranes (vesicles, tubules).*

```python
from warp_md import LipidCurvedLeafletPlan

plan = LipidCurvedLeafletPlan(selection, cutoff=5.0)
leaflets = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Lipid atoms |
| `cutoff` | `float` | required | Midplane assignment cutoff (Å) |

#### Output

- **Type**: dict with curved leaflet assignments

---

## Dynamics & Transport

### LipidMsdPlan

*Mean Squared Displacement calculated purely in the lateral (XY) plane of the membrane, accounting for leaflet drift.*

```python
from warp_md import LipidMsdPlan

plan = LipidMsdPlan(
    selection,
    group_by="resid",
    lag_mode="auto"
)
time, msd = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Lipid atoms |
| `group_by` | `str` | `"resid"` | Grouping mode |
| `lag_mode` | `str` | `"auto"` | Lag mode (`"auto"`, `"multi_tau"`, `"ring"`, `"fft"`) |

#### Output

- `time`: 1D array — lag times
- `msd`: 2D array — lateral MSD per lipid group

---

### LipidFlipFlopPlan

*Detects and tracks lipids flipping between the upper and lower leaflets across the trajectory.*

```python
from warp_md import LipidFlipFlopPlan

plan = LipidFlipFlopPlan(
    selection,
    group_by="resid"
)
flips = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Lipid atoms |
| `group_by` | `str` | `"resid"` | Grouping mode |

#### Output

- **Type**: dict with flip-flop events, rates, and residence times

---

## Structural Order

### LipidSccPlan

*Calculates the Segmental Order Parameter ($S_{CD}$) for lipid tails.*

```python
from warp_md import LipidSccPlan

plan = LipidSccPlan(
    selection,
    group_by="resid"
)
scc = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Lipid tail atoms |
| `group_by` | `str` | `"resid"` | Grouping mode |

#### Output

- **Type**: 2D array
- **Shape**: `(n_frames, n_tail_segments)`
- **Unit**: S_{CD} order parameter

---

### LipidZAnglePlan & LipidZPositionPlan

*Measures the tilt angles and absolute Z positions of lipid molecules or specific tail segments relative to the bilayer normal.*

```python
from warp_md import LipidZAnglePlan, LipidZPositionPlan

angle_plan = LipidZAnglePlan(selection, group_by="resid")
angles = angle_plan.run(traj, system)

z_plan = LipidZPositionPlan(selection, group_by="resid")
z_pos = z_plan.run(traj, system)
```

#### Parameters (LipidZAnglePlan)

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Lipid atoms |
| `group_by` | `str` | `"resid"` | Grouping mode |

#### Output

- **Type**: 2D array — tilt angles per lipid per frame (degrees)

#### Parameters (LipidZPositionPlan)

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `membrane_selection` | `Selection` | required | Membrane atoms for height reference |
| `height_selection` | `Selection` | required | Atoms whose Z position is measured |

#### Output

- **Type**: 2D array — Z positions per group per frame (Å)

---

## Membrane Organization

### LipidRegistrationPlan

*Measures inter-leaflet registration (how lipid domains align across the two leaflets).*

```python
from warp_md import LipidRegistrationPlan

plan = LipidRegistrationPlan(
    selection,
    group_by="resid"
)
registration = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Lipid atoms |
| `group_by` | `str` | `"resid"` | Grouping mode |

#### Output

- **Type**: 1D array — registration index per frame

---

### LipidNeighbourPlan & LipidNeighbourMatrixPlan

*Analyzes lateral lipid-lipid neighborhood organization and interaction networks.*

```python
from warp_md import LipidNeighbourPlan, LipidNeighbourMatrixPlan

neighbors = LipidNeighbourPlan(selection, group_by="resid").run(traj, system)
matrix = LipidNeighbourMatrixPlan(selection, group_by="resid").run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Lipid atoms |
| `group_by` | `str` | `"resid"` | Grouping mode |

#### Output (LipidNeighbourPlan)

- **Type**: list of neighbor lipid indices per frame

#### Output (LipidNeighbourMatrixPlan)

- **Type**: 2D array — neighbor contact matrix per frame
- **Shape**: `(n_frames, n_lipids, n_lipids)`

---

## Lipid Curved Leaflets

### LipidCurvedLeafletPlan

*Assigns lipids to leaflets in curved membrane geometries.*

```python
from warp_md import LipidCurvedLeafletPlan

plan = LipidCurvedLeafletPlan(selection, cutoff=5.0)
leaflets = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Lipid atoms |
| `cutoff` | `float` | required | Distance cutoff for midplane assignment (Å) |

#### Output

- **Type**: dict with curved leaflet assignments per lipid per frame

---

## Lipid Largest Cluster

### LipidLargestClusterPlan

*Identifies the largest lipid cluster in each leaflet.*

```python
from warp_md import LipidLargestClusterPlan

plan = LipidLargestClusterPlan(selection, group_by="resid")
clusters = plan.run(traj, system)
```

#### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Lipid atoms |
| `group_by` | `str` | `"resid"` | Grouping mode |

#### Output

- **Type**: dict with cluster sizes and compositions per frame
