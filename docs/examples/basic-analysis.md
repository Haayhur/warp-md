---
description: Rg, RMSD, RDF, MSD — the core 4 analyses in under 15 lines
icon: chart-simple
---

# Basic Analysis

The four most common trajectory analyses — copy-paste ready.

---

## Radius of Gyration

```python
from warp_md import System, Trajectory, RgPlan

system = System.from_file("protein.pdb")
traj = Trajectory.open_xtc("traj.xtc", system)
rg = RgPlan(system.select("protein")).run(traj, system)

print(f"Mean Rg: {rg.mean():.2f} Å")
```

**CLI:**
```bash
warp-md rg --topology protein.pdb --traj traj.xtc --selection "protein"
```

---

## RMSD

```python
from warp_md import RmsdPlan

rmsd = RmsdPlan(
    system.select("backbone"),
    reference="topology",
    align=True,
).run(traj, system)

print(f"Mean RMSD: {rmsd.mean():.2f} Å")
```

**CLI:**
```bash
warp-md rmsd --topology protein.pdb --traj traj.xtc --selection "backbone" --align
```

---

## RDF

```python
from warp_md import RdfPlan

water_o = system.select("resname SOL and name OW")
rdf = RdfPlan(water_o, water_o, bins=200, r_max=10.0)
r, g, counts = rdf.run(traj, system)
```

**CLI:**
```bash
warp-md rdf --topology system.pdb --traj traj.xtc \
  --sel-a "resname SOL and name OW" \
  --sel-b "resname SOL and name OW" \
  --bins 200 --r-max 10
```

---

## MSD (Multi-Origin)

```python
from warp_md import MsdPlan

msd = MsdPlan(
    system.select("resname SOL"),
    group_by="resid",
    lag_mode="multi_tau",
)
time, data = msd.run(traj, system)
```

**CLI:**
```bash
warp-md msd --topology system.pdb --traj traj.xtc \
  --selection "resname SOL" --group-by resid --lag-mode multi_tau
```

---

## Batch Config (All 4 at Once)

```json
{
  "version": "warp-md.agent.v1",
  "system": "protein.pdb",
  "trajectory": "traj.xtc",
  "analyses": [
    {"name": "rg", "selection": "protein"},
    {"name": "rmsd", "selection": "backbone", "align": true},
    {"name": "rdf", "sel_a": "resname SOL and name OW", "sel_b": "resname SOL and name OW", "bins": 200, "r_max": 10.0},
    {"name": "msd", "selection": "resname SOL", "group_by": "resid", "lag_mode": "multi_tau"}
  ]
}
```

```bash
warp-md run config.json --stream ndjson
```
