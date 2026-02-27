---
description: For when your agent is working with chains and polymers
icon: link
---

# Polymer Plans

Analyses designed for polymer chains. Your agent's toolkit for polymeric systems.

{% hint style="info" %}
These analyses assume your selection contains chains in sequential order.
{% endhint %}

---

## EndToEndPlan

*How stretched out is this polymer?*

```python
from warp_md import EndToEndPlan

plan = EndToEndPlan(selection)
distances = plan.run(traj, system)
```

### Output

- **Type**: 2D array
- **Shape**: `(n_frames, n_chains)`
- **Unit**: Angstrom

---

## ContourLengthPlan

*Total backbone length if you straightened it out*

```python
from warp_md import ContourLengthPlan

contour = ContourLengthPlan(selection).run(traj, system)
```

### Output

- **Type**: 2D array
- **Shape**: `(n_frames, n_chains)`
- **Unit**: Angstrom

---

## ChainRgPlan

*Compactness, measured per-chain*

```python
from warp_md import ChainRgPlan

chain_rg = ChainRgPlan(selection).run(traj, system)
```

### Output

- **Type**: 2D array
- **Shape**: `(n_frames, n_chains)`
- **Unit**: Angstrom

---

## BondLengthDistributionPlan

*What's the typical bond length?*

```python
from warp_md import BondLengthDistributionPlan

plan = BondLengthDistributionPlan(selection, bins=200, r_max=10.0)
centers, counts = plan.run(traj, system)
```

### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Chain atoms |
| `bins` | `int` | required | Number of bins |
| `r_max` | `float` | required | Maximum distance (Å) |

### Output

- `centers`: bin centers (Å)
- `counts`: histogram counts

---

## BondAngleDistributionPlan

*How bendy is this polymer?*

```python
from warp_md import BondAngleDistributionPlan

plan = BondAngleDistributionPlan(selection, bins=180)
centers, counts = plan.run(traj, system)
```

### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Chain atoms |
| `bins` | `int` | required | Number of bins |

### Output

- `centers`: bin centers (degrees)
- `counts`: histogram counts

---

## PersistenceLengthPlan

*Statistical stiffness from bond vector autocorrelation*

```python
from warp_md import PersistenceLengthPlan

plan = PersistenceLengthPlan(selection, max_bonds=20)
result = plan.run(traj, system)
```

### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Chain atoms |
| `max_bonds` | `int` | required | Max bond separation |

### Output (dict)

| Key | Description |
|-----|-------------|
| `bond_autocorrelation` | ⟨cos θ⟩ vs bond separation |
| `lb` | Mean bond length (Å) |
| `lp` | Persistence length (Å) |
| `fit` | Exponential fit parameters |
| `kuhn_length` | Kuhn length (2 × lp) |

---

## Complete Polymer Analysis Example

```python
from warp_md import (
    System, Trajectory,
    EndToEndPlan, ContourLengthPlan, ChainRgPlan,
    BondLengthDistributionPlan, PersistenceLengthPlan
)

system = System.from_pdb("polymer.pdb")
traj = Trajectory.open_xtc("polymer.xtc", system)
chain = system.select("chain A")

# Run the full polymer suite
end_to_end = EndToEndPlan(chain).run(traj, system)
contour = ContourLengthPlan(chain).run(traj, system)
chain_rg = ChainRgPlan(chain).run(traj, system)
bond_dist = BondLengthDistributionPlan(chain, bins=200, r_max=5.0).run(traj, system)
persistence = PersistenceLengthPlan(chain, max_bonds=50).run(traj, system)

print(f"End-to-end: {end_to_end.mean():.2f} Å")
print(f"Contour: {contour.mean():.2f} Å")
print(f"Chain Rg: {chain_rg.mean():.2f} Å")
print(f"Persistence length: {persistence['lp']:.2f} Å")
print("Your agent just analyzed a whole polymer 🧬")
```
