---
description: The building blocks every agent needs to understand
icon: cubes
---

# Core Concepts

The fundamental classes that make warp-md tick. Master these, and everything else clicks into place.

---

## System

Your molecular topology — the cast of characters your agent will analyze.

### Creation

```python
from warp_md import System

# From PDB (Angstrom)
system = System.from_pdb("topology.pdb")

# From GRO (nm → auto-converted to Angstrom)
system = System.from_gro("topology.gro")
```

### Methods

| Method | Returns | What It Does |
|--------|---------|--------------|
| `n_atoms()` | `int` | Total atom count |
| `select(expr)` | `Selection` | Create atom selection |
| `atom_table()` | `dict` | All atom properties |

### Atom Table

```python
table = system.atom_table()
# Keys: 'name', 'resname', 'resid', 'chain_id', 'mass'
```

---

## Selection

A subset of atoms your agent wants to analyze.

### Creation

```python
selection = system.select("backbone and chain A")
```

### Selection Language

**Predicates:**

| Predicate | Description | Example |
|-----------|-------------|---------|
| `name` | Atom name | `name CA` |
| `resname` | Residue name | `resname ALA` |
| `resid` | Residue ID | `resid 10`, `resid 10-50` |
| `chain` | Chain ID | `chain A` |
| `protein` | All protein atoms | `protein` |
| `backbone` | N, CA, C, O atoms | `backbone` |

**Boolean Operators:**

| Operator | Example |
|----------|---------|
| `and` | `backbone and chain A` |
| `or` | `resname ALA or resname GLY` |
| `not` | `not resname SOL` |
| `()` | `(resname ALA or resname GLY) and backbone` |

---

## Trajectory

The movie of your simulation — coordinate frames over time.

### Creation

```python
from warp_md import Trajectory

# DCD format
traj = Trajectory.open_dcd("traj.dcd", system)
traj = Trajectory.open_dcd("traj_nm.dcd", system, length_scale=10.0)

# XTC format (GROMACS)
traj = Trajectory.open_xtc("traj.xtc", system)
```

### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `length_scale` | `float` | `1.0` | Multiply coordinates (DCD only) |

{% hint style="warning" %}
Trajectory atom count must match `system.n_atoms()`. They need to be talking about the same molecules.
{% endhint %}

---

## Grouping (group_by)

Many plans aggregate atoms into groups. The `group_by` parameter controls how:

| Value | What It Does |
|-------|--------------|
| `"resid"` | Group by residue ID |
| `"chain"` | Group by chain ID |
| `"resid_chain"` | Unique resid within each chain |

### Example

```python
from warp_md import MsdPlan

# Group atoms by residue
msd = MsdPlan(selection, group_by="resid")

# Group by chain
msd = MsdPlan(selection, group_by="chain")
```

---

## Group Types

For multi-species analyses (MSD by ion type, conductivity), tag groups with types:

```python
from warp_md import group_types_from_selections

types = group_types_from_selections(
    system,
    system.select("resname NA or resname CL"),
    "resid",
    ["resname NA", "resname CL"],  # type 0, type 1
)

msd = MsdPlan(selection, group_by="resid", group_types=types)
```

Output columns will include per-type and total values.

---

## Device Selection

All `plan.run()` methods accept `device`:

| Value | Behavior |
|-------|----------|
| `"auto"` | CUDA if available, else CPU *(default)* |
| `"cpu"` | Force CPU |
| `"cuda"` | Force CUDA (first GPU) |
| `"cuda:0"` | Specific GPU index |

```python
result = plan.run(traj, system, device="auto")
result = plan.run(traj, system, device="cuda:0")
```

---

## Feature Store

For very long trajectories, persist per-frame features to disk:

```python
# Rust API
writer = FeatureStoreWriter("path_prefix", schema)
# Writes: path_prefix.bin + path_prefix.json

reader = FeatureStoreReader("path_prefix")
```

Useful for offline FFT analysis without re-reading the entire trajectory.
