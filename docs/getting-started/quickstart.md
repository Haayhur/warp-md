---
description: Your agent's first analysis in under 5 minutes
icon: bolt
---

# Quick Start

Let's get your agent from "freshly installed" to "analyzing molecules like a pro" in record time.

{% hint style="info" %}
This guide assumes you've completed [Installation](installation.md). If not, go do that — we'll wait.
{% endhint %}

---

## The 4-Step Simulation Dance

{% stepper %}
{% step %}
## Meet Your Molecules

```python
from warp_md import System

# Load topology from PDB or GRO
system = System.from_pdb("example.pdb")
# Or: system = System.from_gro("example.gro")

print(f"Loaded {system.n_atoms()} atoms")
```

{% hint style="info" %}
PDB files are in Angstrom. GRO files (in nanometers) get auto-converted. Your agent never has to think about it.
{% endhint %}
{% endstep %}

{% step %}
## Tell Your Agent What to Look At

```python
# The classics
backbone = system.select("backbone")
water = system.select("resname SOL")

# Get fancy with boolean logic
ca_chain_a = system.select("name CA and chain A")
```

The selection language is your agent's way of pointing at specific atoms. More on this in the [Selections tutorial](../tutorial/selections.md).
{% endstep %}

{% step %}
## Open the Time Machine

```python
from warp_md import Trajectory

# GROMACS format
traj = Trajectory.open_xtc("trajectory.xtc", system)

# Or CHARMM/NAMD format
traj = Trajectory.open_dcd("trajectory.dcd", system)
```

Trajectories are like movies of your simulation. Now your agent can watch.
{% endstep %}

{% step %}
## Run Analysis — The Payoff

```python
from warp_md import RgPlan

# Create an analysis plan
plan = RgPlan(backbone, mass_weighted=False)

# Execute (device="auto" uses GPU if available)
rg = plan.run(traj, system, device="auto")

print(f"Rg shape: {rg.shape}")
print(f"Mean Rg: {rg.mean():.2f} Å")
```

That's it. Your agent just computed the radius of gyration across an entire trajectory.
{% endstep %}
{% endstepper %}

---

## Complete Example (Copy-Paste Ready)

```python
from warp_md import System, Trajectory, RgPlan, RmsdPlan

# 1. Load the system
system = System.from_pdb("protein.pdb")

# 2. Select what matters
backbone = system.select("backbone")
ca_atoms = system.select("name CA")

# 3. Open trajectory
traj = Trajectory.open_xtc("trajectory.xtc", system)

# 4. Run analyses
rg = RgPlan(ca_atoms).run(traj, system)
rmsd = RmsdPlan(backbone, reference="topology", align=True).run(traj, system)

# 5. Harvest results
print(f"Frames: {len(rg)}")
print(f"Rg: {rg.mean():.2f} ± {rg.std():.2f} Å")
print(f"RMSD: {rmsd.mean():.2f} ± {rmsd.std():.2f} Å")
```

---

## CLI: For Agents Who Prefer Commands

Sometimes your agent just wants to run a quick analysis without writing Python:

```bash
# Radius of gyration
warp-md rg --topology protein.pdb --traj trajectory.xtc --selection "protein"

# RDF between water oxygens
warp-md rdf --topology system.pdb --traj trajectory.xtc \
    --sel-a "resname SOL and name OW" \
    --sel-b "resname SOL and name OW" \
    --bins 200 --r-max 10
```

{% hint style="success" %}
**Agent pro-tip**: Analysis and `run` commands output structured JSON envelopes by default. Discovery commands support `--json`/`--format json`.
{% endhint %}

---

## What's Next?

<table data-view="cards">
    <thead>
        <tr>
            <th>Topic</th>
            <th data-card-target data-type="content-ref">Link</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>📚 Full Tutorial</strong><br>The complete learning path for agents and humans</td>
            <td><a href="../tutorial/README.md">Tutorial</a></td>
        </tr>
        <tr>
            <td><strong>📊 Analysis Arsenal</strong><br>All 96 Plans — from Rg to GIST</td>
            <td><a href="../reference/README.md">API Reference</a></td>
        </tr>
        <tr>
            <td><strong>📦 World Building</strong><br>Pack molecules into simulation boxes</td>
            <td><a href="../guides/packing.md">Packing Guide</a></td>
        </tr>
    </tbody>
</table>
