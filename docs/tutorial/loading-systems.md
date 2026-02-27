---
description: Your agent meets its first atoms
icon: file-import
---

# Loading Systems & Trajectories

Before your agent can analyze anything, it needs to meet the molecules. Let's make introductions.

---

## Loading a System (Topology)

A `System` is your molecular cast of characters — atom names, residues, chains, masses. Load it from PDB or GRO:

{% tabs %}
{% tab title="PDB Format" %}
```python
from warp_md import System

system = System.from_pdb("example.pdb")
print(f"Loaded {system.n_atoms()} atoms — let's goooo")
```

{% hint style="info" %}
PDB files are in **Angstrom**. No conversion needed.
{% endhint %}
{% endtab %}

{% tab title="PDBQT Format" %}
```python
from warp_md import System

system = System.from_pdbqt("docking_poses.pdbqt")
print(f"Loaded {system.n_atoms()} atoms")
```

{% hint style="info" %}
PDBQT is treated like PDB with extra columns (charges/atom types). warp-md ignores the extra fields.
{% endhint %}
{% endtab %}

{% tab title="GRO Format" %}
```python
from warp_md import System

system = System.from_gro("example.gro")
print(f"Loaded {system.n_atoms()} atoms")
```

{% hint style="info" %}
GRO files are in **nm** — warp-md auto-converts to Angstrom so your agent doesn't have to think about it.
{% endhint %}
{% endtab %}
{% endtabs %}

---

## Peeking Under the Hood

Curious about what's in your system? Inspect the atom table:

```python
# Get atom information as a dictionary
table = system.atom_table()

print(table.keys())
# dict_keys(['name', 'resname', 'resid', 'chain_id', 'mass'])

# Access specific columns
atom_names = table['name']      # Every atom's name
residue_names = table['resname']  # Amino acid? Water? Ligand?
```

Your agent now knows everyone at the molecular party.

---

## Opening Trajectories

Trajectories are movies of your simulation — coordinate frames over time. Open them like this:

{% tabs %}
{% tab title="DCD Format" %}
```python
from warp_md import Trajectory

# Standard DCD (Angstrom)
traj = Trajectory.open_dcd("trajectory.dcd", system)

# DCD in nm? Multiply by 10 to convert
traj = Trajectory.open_dcd("trajectory_nm.dcd", system, length_scale=10.0)
```
{% endtab %}

{% tab title="XTC Format" %}
```python
from warp_md import Trajectory

# XTC (GROMACS format, nm → auto-converted to Angstrom)
traj = Trajectory.open_xtc("trajectory.xtc", system)
```
{% endtab %}
{% endtabs %}

{% hint style="warning" %}
**"Atom count does not match system"** — The most common error.

Your trajectory and topology are having a disagreement about how many atoms exist. Make sure you're using matching files.
{% endhint %}

---

## Unit Conventions (The Cheat Sheet)

| Format | Native Unit | warp-md Internal |
|--------|-------------|------------------|
| PDB | Angstrom | Angstrom |
| GRO | nm | Angstrom (auto-converted) |
| DCD | Angstrom* | Angstrom |
| XTC | nm | Angstrom (auto-converted) |

\* Some DCD files use nm — use `length_scale=10.0` to convert.

---

## Complete Loading Workflow

```python
from warp_md import System, Trajectory

# 1. Load topology
system = System.from_pdb("protein_solvated.pdb")
print(f"System: {system.n_atoms()} atoms")

# 2. Open trajectory
traj = Trajectory.open_xtc("production.xtc", system)

# 3. Ready for analysis!
print("Your agent is armed and ready 🚀")
```

---

<a href="selections.md" class="button primary">Next: Atom Selections →</a>
