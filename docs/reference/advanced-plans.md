---
description: For when your agent needs to show off — transport, structure, and more
icon: microscope
---

# Advanced Plans

The advanced analysis suite for complex systems. Transport properties, structural correlations, and beyond.

---

## RotAcfPlan

*How fast are molecules reorienting?*

```python
from warp_md import RotAcfPlan

plan = RotAcfPlan(
    selection,
    group_by="resid",
    orientation=[0, 1],  # or [0, 1, 2] for plane normal
    p2_legendre=True,
)
time, data = plan.run(traj, system)
```

### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `orientation` | `list[int]` | required | 2 indices = vector; 3 = plane normal |
| `p2_legendre` | `bool` | `True` | Compute P2 Legendre polynomial |

{% hint style="info" %}
Orientation indices are **within each group**, not global atom indices.
{% endhint %}

### Output

- `time`: 1D array
- `data`: 2D array, `cols = 2 × (n_types + 1)`
  - First block: P1 per type + total
  - Second block: P2 per type + total

---

## ConductivityPlan

*How well does this ionic liquid conduct?*

```python
from warp_md import ConductivityPlan, charges_from_table

charges = charges_from_table(system, "charges.csv")

plan = ConductivityPlan(
    selection,
    group_by="resid",
    charges=charges,
    temperature=300.0,
    transference=False,
)
time, data = plan.run(traj, system)
```

### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `charges` | `list[float]` | required | Per-atom charges |
| `temperature` | `float` | required | Temperature (K) |
| `transference` | `bool` | `False` | Compute transference matrix |

### Output

- If `transference=False`: `cols = 1` (total conductivity)
- If `transference=True`: `cols = n_types² + 1` (matrix + total)

---

## DielectricPlan

*The dielectric response of your system*

```python
from warp_md import DielectricPlan

plan = DielectricPlan(selection, group_by="resid", charges=charges)
result = plan.run(traj, system)
```

### Output (dict)

| Key | Description |
|-----|-------------|
| `time` | Time array |
| `rot_sq` | Rotational component² |
| `trans_sq` | Translational component² |
| `rot_trans` | Cross term |
| `dielectric_rot` | Rotational dielectric |
| `dielectric_total` | Total dielectric |
| `mu_avg` | Average dipole |

---

## DipoleAlignmentPlan

*How aligned are the molecular dipoles with lab axes?*

```python
from warp_md import DipoleAlignmentPlan

plan = DipoleAlignmentPlan(selection, group_by="resid", charges=charges)
time, data = plan.run(traj, system)
```

### Output

- `cols = 6 × (n_types + 1)`
- First half: cos(x), cos(y), cos(z) per type + total
- Second half: cos²(x), cos²(y), cos²(z) per type + total

---

## IonPairCorrelationPlan

*Who's clustering with whom?*

```python
from warp_md import IonPairCorrelationPlan, group_types_from_selections

types = group_types_from_selections(system, selection, "resid",
    ["resname BMIM", "resname BF4"])

plan = IonPairCorrelationPlan(
    selection,
    rclust_cat=6.0,
    rclust_ani=6.0,
    group_by="resid",
    cation_type=0,
    anion_type=1,
    max_cluster=10,
    group_types=types,
)
time, data = plan.run(traj, system)
```

### Output

- `cols = 6`: ip_total, ip_cation, ip_anion, cp_total, cp_cation, cp_anion

---

## StructureFactorPlan

*S(q) from your RDF*

```python
from warp_md import StructureFactorPlan

plan = StructureFactorPlan(
    selection, bins=200, r_max=6.0, q_bins=100, q_max=20.0
)
r, g, q, s = plan.run(traj, system)
```

### Output

- `r`, `g`: g(r) from RDF
- `q`, `s`: S(q) structure factor

---

{% hint style="info" %}
**WaterCountPlan** is documented in [Solvation & Density](solvation-density.md#watercountplan).
{% endhint %}

---

## EquipartitionPlan

*Temperature from kinetic energy (requires velocities)*

```python
from warp_md import EquipartitionPlan

plan = EquipartitionPlan(selection, group_by="resid")
time, data = plan.run(traj, system)
```

### Output

- `cols = n_types + 1` (per-type + total temperature)

---

## HbondPlan

*How many hydrogen bonds per frame?*

```python
from warp_md import HbondPlan

# Distance-only filter
plan = HbondPlan(
    donors=system.select("name N"),
    acceptors=system.select("name O"),
    dist_cutoff=3.5,
)

# With angle filter
plan = HbondPlan(
    donors=system.select("name N"),
    acceptors=system.select("name O"),
    hydrogens=system.select("name H"),
    dist_cutoff=3.5,
    angle_cutoff=150.0,
)
time, data = plan.run(traj, system)
```

{% hint style="warning" %}
When using `angle_cutoff`, hydrogens must be 1:1 with donors.
{% endhint %}

### Output

- `cols = 1`: H-bond count per frame
