---
description: Power moves for when your agent needs to show off
icon: atom
---

# Advanced Analyses

You've mastered the basics. Now let's unlock the full arsenal — polymer metrics, rotational dynamics, transport properties, and more.

---

## Polymer Analyses

For when you're working with chains, polymers, or anything with sequential connectivity.

### End-to-End Distance

*How stretched out is this polymer?*

```python
from warp_md import EndToEndPlan

plan = EndToEndPlan(system.select("chain A"))
distances = plan.run(traj, system)

# Output: 2D array (frames × chains)
print(f"Mean end-to-end: {distances.mean():.2f} Å")
```

### Contour Length

*The total backbone length if you straightened it out*

```python
from warp_md import ContourLengthPlan

contour = ContourLengthPlan(system.select("chain A")).run(traj, system)
```

### Chain Rg

*Compactness, per-chain*

```python
from warp_md import ChainRgPlan

chain_rg = ChainRgPlan(system.select("chain A")).run(traj, system)
```

### Bond Length Distribution

*What's the typical bond length?*

```python
from warp_md import BondLengthDistributionPlan

centers, counts = BondLengthDistributionPlan(
    system.select("chain A"),
    bins=200,
    r_max=10.0
).run(traj, system)
```

### Bond Angle Distribution

*How bendy is this polymer?*

```python
from warp_md import BondAngleDistributionPlan

centers, counts = BondAngleDistributionPlan(
    system.select("chain A"),
    bins=180
).run(traj, system)
```

### Persistence Length

*The statistical stiffness of the chain*

```python
from warp_md import PersistenceLengthPlan

result = PersistenceLengthPlan(
    system.select("chain A"),
    max_bonds=20
).run(traj, system)

print(result.keys())
# ['bond_autocorrelation', 'lb', 'lp', 'fit', 'kuhn_length']
```

---

## Transport Properties

For ionic liquids, electrolytes, and anything that conducts.

### Rotational ACF

*How fast are molecules reorienting?*

```python
from warp_md import RotAcfPlan

plan = RotAcfPlan(
    selection=system.select("resname BMIM"),
    group_by="resid",
    orientation=[0, 1],  # vector from atom 0 to atom 1 within each group
    p2_legendre=True,    # compute P2 Legendre polynomial
)

time, data = plan.run(traj, system)
```

{% hint style="info" %}
**Orientation indices**:
- 2 indices = vector (atom0 → atom1)
- 3 indices = plane normal (atoms 0, 1, 2)

Indices are **within each group**, not global atom indices.
{% endhint %}

---

### Conductivity

*How well does this ionic liquid conduct?*

Requires per-atom charges:

```python
from warp_md import ConductivityPlan, charges_from_table

# Load charges from CSV (columns: resname, name, charge)
charges = charges_from_table(system, "charges.csv")

plan = ConductivityPlan(
    selection=system.select("resname BMIM or resname BF4"),
    group_by="resid",
    charges=charges,
    temperature=300.0,
)

time, conductivity = plan.run(traj, system)
```

<details>
<summary>Charge Loading Options</summary>

```python
from warp_md import charges_from_selections, charges_from_table

# From selection rules
charges = charges_from_selections(system, [
    {"selection": "resname NA", "charge": 1.0},
    {"selection": "resname CL", "charge": -1.0},
])

# From CSV/TSV file
charges = charges_from_table(system, "charges.csv")
```

</details>

---

### Dielectric

*The dielectric response of your system*

```python
from warp_md import DielectricPlan

plan = DielectricPlan(
    selection=system.select("resname BMIM or resname BF4"),
    group_by="resid",
    charges=charges,
)

result = plan.run(traj, system)
print(result.keys())
# ['time', 'rot_sq', 'trans_sq', 'rot_trans', 'dielectric_rot', 'dielectric_total', 'mu_avg']
```

---

### Dipole Alignment

*How aligned are the molecular dipoles?*

```python
from warp_md import DipoleAlignmentPlan

time, data = DipoleAlignmentPlan(
    system.select("resname BMIM"),
    group_by="resid",
    charges=charges,
).run(traj, system)
```

---

## Structural Analyses

### Ion-Pair Correlation

*Who's clustering with whom?*

```python
from warp_md import IonPairCorrelationPlan, group_types_from_selections

# Define group types
sel_ions = system.select("resname BMIM or resname BF4")
types = group_types_from_selections(
    system,
    sel_ions,
    "resid",
    ["resname BMIM", "resname BF4"],  # type 0, type 1
)

plan = IonPairCorrelationPlan(
    sel_ions,
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

---

### Structure Factor

*S(q) from your RDF*

```python
from warp_md import StructureFactorPlan

r, g, q, s = StructureFactorPlan(
    system.select("resname SOL and name OW"),
    bins=200,
    r_max=6.0,
    q_bins=100,
    q_max=20.0,
    pbc="orthorhombic"
).run(traj, system)
```

---

### Water Occupancy Grid

*Where does water like to hang out?*

```python
from warp_md import WaterCountPlan

result = WaterCountPlan(
    water_selection=system.select("resname SOL and name OW"),
    center_selection=system.select("protein"),
    box_unit=(1.0, 1.0, 1.0),       # grid cell size (Å)
    region_size=(30.0, 30.0, 30.0), # total grid size (Å)
).run(traj, system)

print(result['dims'], result['mean'].shape)
```

---

### Hydrogen Bond Counts

*How many H-bonds per frame?*

```python
from warp_md import HbondPlan

# Distance-only filter
hbond = HbondPlan(
    donors=system.select("name N"),
    acceptors=system.select("name O"),
    dist_cutoff=3.5,
)

# With angle filter (hydrogens must be 1:1 with donors)
hbond = HbondPlan(
    donors=system.select("name N"),
    acceptors=system.select("name O"),
    hydrogens=system.select("name H"),
    dist_cutoff=3.5,
    angle_cutoff=150.0,  # degrees
)

time, counts = hbond.run(traj, system)
```

---

<a href="cli-usage.md" class="button primary">Next: CLI Usage →</a>
