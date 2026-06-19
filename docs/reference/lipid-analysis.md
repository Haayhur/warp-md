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

### LipidZAnglePlan & LipidZPositionPlan

*Measures the tilt angles and absolute Z positions of lipid molecules or specific tail segments relative to the bilayer normal.*

```python
from warp_md import LipidZAnglePlan, LipidZPositionPlan

angle_plan = LipidZAnglePlan(selection, group_by="resid")
angles = angle_plan.run(traj, system)

z_plan = LipidZPositionPlan(selection, group_by="resid")
z_pos = z_plan.run(traj, system)
```

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

### LipidNeighbourPlan & LipidNeighbourMatrixPlan

*Analyzes lateral lipid-lipid neighborhood organization and interaction networks.*

```python
from warp_md import LipidNeighbourPlan, LipidNeighbourMatrixPlan

neighbors = LipidNeighbourPlan(selection, group_by="resid").run(traj, system)
matrix = LipidNeighbourMatrixPlan(selection, group_by="resid").run(traj, system)
```
