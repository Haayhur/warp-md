---
description: Charge and group type utilities for your agent
icon: wrench
---

# Builder Helpers

Utility functions to prepare charges and group types. Think of these as agent helpers — they do the tedious work so your agent can focus on science.

---

## Charges

Many analyses (conductivity, dielectric, dipole) need per-atom charges.

### charges_from_table

*Load charges from a CSV/TSV file*

```python
from warp_md import charges_from_table

charges = charges_from_table(system, "charges.csv", delimiter=None, default=0.0)
```

**File format** (columns: `resname`, `name`/`atom`, `charge`):

```csv
resname,name,charge
NA,NA,1.0
CL,CL,-1.0
SOL,OW,-0.834
SOL,HW1,0.417
SOL,HW2,0.417
```

### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `system` | `System` | required | System object |
| `path` | `str` | required | Path to CSV/TSV |
| `delimiter` | `str` | `None` | Auto-detect if None |
| `default` | `float` | `0.0` | Charge for unmatched atoms |

---

### charges_from_selections

*Define charges using selection rules*

```python
from warp_md import charges_from_selections

charges = charges_from_selections(system, [
    {"selection": "resname NA", "charge": 1.0},
    {"selection": "resname CL", "charge": -1.0},
    {"selection": "resname SOL and name OW", "charge": -0.834},
    {"selection": "resname SOL and (name HW1 or name HW2)", "charge": 0.417},
], default=0.0)
```

### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `system` | `System` | required | System object |
| `entries` | `list[dict]` | required | Selection/charge pairs |
| `default` | `float` | `0.0` | Charge for unmatched atoms |

---

## Group Types

For multi-species analyses (MSD by ion type, conductivity), tag groups with types.

### group_types_from_selections

*Map groups to type IDs*

```python
from warp_md import group_types_from_selections

types = group_types_from_selections(
    system,
    system.select("resname NA or resname CL"),
    "resid",
    ["resname NA", "resname CL"],  # type 0, type 1
)
```

### Parameters

| Parameter | Type | What It Does |
|-----------|------|--------------|
| `system` | `System` | System object |
| `selection` | `Selection` | Atoms to group |
| `group_by` | `str` | Grouping mode (`"resid"`, `"chain"`) |
| `type_selections` | `list[str]` | Selection for each type |

### Returns

- `list[int]`: Type ID for each group (0, 1, ..., n_types-1)

---

## CLI Usage

From the command line, these map to flexible input formats:

{% tabs %}
{% tab title="Charges" %}
```bash
# Inline array
warp-md conductivity --charges '[1.0,-1.0,1.0]' ...

# From CSV
warp-md conductivity --charges table:charges.csv ...

# From selections
warp-md conductivity --charges 'selections:[{"selection":"resname NA","charge":1.0}]' ...
```
{% endtab %}

{% tab title="Group Types" %}
```bash
# Inline array
warp-md msd --group-types '[0,1,1,0]' ...

# From selections
warp-md msd --group-types 'selections:resname NA,resname CL' ...
```
{% endtab %}
{% endtabs %}

---

## Complete Setup Example

```python
from warp_md import (
    System, Trajectory,
    ConductivityPlan,
    charges_from_table, group_types_from_selections
)

# Load system
system = System.from_pdb("ionic_liquid.pdb")
traj = Trajectory.open_xtc("ionic_liquid.xtc", system)

# Define selection
ions = system.select("resname BMIM or resname BF4")

# Load charges
charges = charges_from_table(system, "charges.csv")

# Define group types
types = group_types_from_selections(
    system, ions, "resid",
    ["resname BMIM", "resname BF4"]
)

# Run conductivity
plan = ConductivityPlan(
    ions, group_by="resid",
    charges=charges,
    temperature=300.0,
    group_types=types,
)
time, data = plan.run(traj, system)
print("Conductivity analysis complete 🔋")
```
