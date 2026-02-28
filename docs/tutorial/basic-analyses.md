---
description: Your agent's bread and butter — Rg, RMSD, MSD, RDF
icon: chart-line
---

# Basic Analyses

The four horsemen of trajectory analysis. Master these, and your agent can handle 80% of what computational scientists need.

---

## Radius of Gyration (Rg)

*How compact is this thing?*

```python
from warp_md import RgPlan

plan = RgPlan(
    selection=system.select("name CA"),
    mass_weighted=False  # or True for mass-weighted
)

rg = plan.run(traj, system, device="auto")

print(f"Shape: {rg.shape}")        # (n_frames,)
print(f"Mean Rg: {rg.mean():.2f} Å")
```

{% hint style="info" %}
**Output**: 1D array with one Rg value per frame. Simple. Clean. Useful.
{% endhint %}

---

## Root Mean Square Deviation (RMSD)

*How different is this from the starting structure?*

```python
from warp_md import RmsdPlan

plan = RmsdPlan(
    selection=system.select("backbone"),
    reference="topology",  # compare against initial structure
    align=True             # superimpose before measuring
)

rmsd = plan.run(traj, system, device="auto")

print(f"Final RMSD: {rmsd[-1]:.2f} Å")
```

{% tabs %}
{% tab title="Reference Options" %}
| Reference | What It Compares Against |
|-----------|-------------------------|
| `"topology"` | Initial PDB/GRO coordinates |
| `"frame0"` | First trajectory frame |
{% endtab %}

{% tab title="Alignment" %}
| `align` | Behavior |
|---------|----------|
| `True` | Superimpose before RMSD *(recommended)* |
| `False` | Raw RMSD without fitting |
{% endtab %}
{% endtabs %}

---

## Mean Square Displacement (MSD)

*How far are things diffusing?*

```python
from warp_md import MsdPlan

plan = MsdPlan(
    selection=system.select("resname SOL and name OW"),
    group_by="resid",  # treat each water molecule as a unit
)

time, msd = plan.run(traj, system, device="auto")

print(f"Time points: {len(time)}")
print(f"MSD shape: {msd.shape}")
```

### Power User Options

```python
plan = MsdPlan(
    selection=system.select("resname SOL"),
    group_by="resid",
    axis=[0.0, 0.0, 1.0],              # project onto z-axis
    frame_decimation=(100, 100),       # start at frame 100, stride 100
    lag_mode="multi_tau",              # bounded memory for long trajectories
)
```

<details>
<summary>Lag Mode Reference</summary>

| Mode | Description | When to Use |
|------|-------------|-------------|
| `"auto"` | FFT if fits memory, else multi-tau | Default (let warp-md decide) |
| `"multi_tau"` | Log-spaced lags, bounded memory | Long trajectories |
| `"ring"` | Exact up to max_lag | Short-time validation |
| `"fft"` | Exact all lags | Short trajectories |

</details>

---

## Radial Distribution Function (RDF)

*What's the local structure around each atom?*

```python
from warp_md import RdfPlan

plan = RdfPlan(
    sel_a=system.select("resname SOL and name OW"),
    sel_b=system.select("resname SOL and name OW"),
    bins=200,
    r_max=10.0,
    pbc="orthorhombic"
)

r, g, counts = plan.run(traj, system)

print(f"First peak at r = {r[g.argmax()]:.2f} Å")
```

{% hint style="info" %}
**Output**:
- `r`: bin centers (Å)
- `g`: g(r) values (the good stuff)
- `counts`: raw pair counts (for the curious)
{% endhint %}

---

## Complete Example: All Four at Once

```python
from warp_md import System, Trajectory
from warp_md import RgPlan, RmsdPlan, MsdPlan, RdfPlan

# Load
system = System.from_pdb("solvated_protein.pdb")
traj = Trajectory.open_xtc("production.xtc", system)

# Selections
protein = system.select("protein")
backbone = system.select("backbone")
water_o = system.select("resname SOL and name OW")

# Run the big four
rg = RgPlan(protein).run(traj, system)
rmsd = RmsdPlan(backbone, reference="topology", align=True).run(traj, system)
time, msd = MsdPlan(water_o, group_by="resid").run(traj, system)
r, g, _ = RdfPlan(water_o, water_o, bins=200, r_max=10.0).run(traj, system)

# Victory lap
print(f"Rg: {rg.mean():.2f} ± {rg.std():.2f} Å")
print(f"RMSD: {rmsd.mean():.2f} ± {rmsd.std():.2f} Å")
print(f"RDF first peak: {r[g.argmax()]:.2f} Å")
print("Your agent just did four analyses in 15 lines 🎉")
```

---

<a href="advanced-analyses.md" class="button primary">Next: Advanced Analyses →</a>
