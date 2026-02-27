# Tutorial: warp-md for beginners

This tutorial walks from zero to running every analysis module.
All examples use the Python API so you can move fast.

## 0) Install and build

warp-md is a Rust workspace with Python bindings.

```bash
# build + install python bindings
maturin develop

# quick smoke test
python -c "import warp_md; print(warp_md.System)"
```

Optional CUDA build (if you have a CUDA driver + nvrtc):

```bash
cargo test -p traj-engine --features cuda
```

## 0.5) Agent quickstart (CLI)

One-command CLI with JSON summary by default.

```bash
warp-md rg --topology top.pdb --traj traj.xtc --selection "protein"
warp-md rdf --topology top.pdb --traj traj.xtc --sel-a "resname SOL and name OW" --sel-b "resname SOL and name OW" --bins 200 --r-max 10
```

Use `--no-summary` to silence stdout, or `--summary-format text` for human output.

## 0.6) Pack initial coordinates (optional)

Use the packer to build initial coordinates from small templates.

CLI:

```bash
warp-pack --config pack.json --output packed.pdb --format pdb
```

Python:

```python
from warp_md.pack import Box, Structure, PackConfig, water_pdb
from warp_md.pack.runner import run
from warp_md.pack.export import export

cfg = PackConfig(
    structures=[Structure(water_pdb("tip3p"), count=200)],
    box=Box((40.0, 40.0, 40.0)),
    min_distance=2.0,
    add_box_sides=True,
)
result = run(cfg)
export(result, "pdb", "packed.pdb")
```

Bundled `water_pdb(...)` templates are **single-molecule** PDBs.

### Restart files

Use Packmol-style restart files (one line per molecule: `x y z beta gamma teta`) to
reuse placements or resume work.

```json
{
  "restart_from": "all.restart",
  "restart_to": "all.out.restart",
  "structures": [
    {
      "path": "water.pdb",
      "count": 200,
      "restart_from": "water.restart"
    }
  ]
}
```

## 1) Load a system (topology)

Use PDB or GRO to load topology and a reference frame.

```python
from warp_md import System

system = System.from_pdb("example.pdb")
# or
system = System.from_gro("example.gro")
```

Notes:
- PDB is assumed to be in Angstrom.
- GRO is in nm and is converted to Angstrom internally.

## 2) Build selections

Selections are simple and fast. Supported predicates:
- `name`, `resname`, `resid`, `chain`, `protein`, `backbone`
- boolean logic: `and`, `or`, `not`, parentheses

```python
sel_ca = system.select("name CA")
sel_bb = system.select("backbone")
sel_chain_a = system.select("chain A")
sel_range = system.select("resid 10-50")
sel_combo = system.select("(resname ALA or resname GLY) and backbone")
```

## 3) Open a trajectory

DCD and XTC are supported. The trajectory must match the system atom count.

```python
from warp_md import Trajectory

traj = Trajectory.open_dcd("traj.dcd", system)
# If your DCD coordinates are in nm:
traj = Trajectory.open_dcd("traj_nm.dcd", system, length_scale=10.0)

# Or XTC (always nm, converted to Angstrom):
traj = Trajectory.open_xtc("traj.xtc", system)
```

## 4) First analysis: Rg

```python
from warp_md import RgPlan

plan = RgPlan(sel_ca, mass_weighted=False)
rg = plan.run(traj, system, device="auto")
print(rg.shape)
```

## 5) RMSD

Reference options: `"topology"` (initial PDB/GRO), or `"frame0"`.
`align=True` means fit before measuring RMSD.

```python
from warp_md import RmsdPlan

rmsd = RmsdPlan(sel_bb, reference="topology", align=True)
values = rmsd.run(traj, system, device="auto")
```

## 6) MSD (multi-origin)

MSD groups atoms (by residue, chain, or resid+chain). You can also request
axis-projected MSD and time decimation.

```python
from warp_md import MsdPlan

msd = MsdPlan(
    sel_chain_a,
    group_by="resid",
    axis=[0.0, 0.0, 1.0],
    length_scale=1.0,
    frame_decimation=(100, 100),
    dt_decimation=(1000, 10, 10000, 100),
)

time, data = msd.run(traj, system, device="auto")
print(time.shape, data.shape)
```

Lag modes for correlators (MSD/RotAcf/Conductivity/Ion-pair):

```python
# auto = FFT if it fits memory and dt is uniform, else multi_tau
msd_auto = MsdPlan(sel_chain_a, group_by="resid", lag_mode="auto")

# multi_tau = bounded memory, log-spaced lags (best default for long runs)
msd_mt = MsdPlan(sel_chain_a, group_by="resid", lag_mode="multi_tau", multi_tau_m=16, multi_tau_levels=20)

# ring = exact up to max_lag (short-time validation)
msd_ring = MsdPlan(sel_chain_a, group_by="resid", lag_mode="ring", max_lag=10000)
```

## 7) RDF

```python
from warp_md import RdfPlan

rdf = RdfPlan(sel_chain_a, sel_chain_a, bins=200, r_max=10.0, pbc="orthorhombic")
r, g, counts = rdf.run(traj, system)
```

## 8) Polymer analyses

These assume that your selection contains chains in order.

```python
from warp_md import (
    EndToEndPlan,
    ContourLengthPlan,
    ChainRgPlan,
    BondLengthDistributionPlan,
    BondAngleDistributionPlan,
    PersistenceLengthPlan,
)

end_to_end = EndToEndPlan(sel_chain_a).run(traj, system)
contour = ContourLengthPlan(sel_chain_a).run(traj, system)
chain_rg = ChainRgPlan(sel_chain_a).run(traj, system)

bond_lengths = BondLengthDistributionPlan(sel_chain_a, bins=200, r_max=10.0).run(traj, system)
angles = BondAngleDistributionPlan(sel_chain_a, bins=180).run(traj, system)

persistence = PersistenceLengthPlan(sel_chain_a, max_bonds=20).run(traj, system)
```

## 9) Advanced analysis suite

### Rotational ACF

Provide orientation as 2 atom indices (vector) or 3 atom indices (plane).
Indices are within each group, not global atom indices.

```python
from warp_md import RotAcfPlan

rotacf = RotAcfPlan(
    sel_chain_a,
    group_by="resid",
    orientation=[0, 1],
    p2_legendre=True,
)

time, data = rotacf.run(traj, system)
```

### Conductivity

Requires per-atom charges. See builder helpers below.

```python
from warp_md import ConductivityPlan, charges_from_table

charges = charges_from_table(system, "charges.csv")
cond = ConductivityPlan(sel_chain_a, group_by="resid", charges=charges, temperature=300.0)

time, data = cond.run(traj, system)
```

### Dielectric

```python
from warp_md import DielectricPlan

charges = charges_from_table(system, "charges.csv")
diel = DielectricPlan(sel_chain_a, group_by="resid", charges=charges)

out = diel.run(traj, system)
print(out.keys())
```

### Dipole alignment

```python
from warp_md import DipoleAlignmentPlan

charges = charges_from_table(system, "charges.csv")
dip = DipoleAlignmentPlan(sel_chain_a, group_by="resid", charges=charges)

time, data = dip.run(traj, system)
```

### Ion-pair correlation

Requires cation/anion group types.

```python
from warp_md import IonPairCorrelationPlan, group_types_from_selections

sel_ions = system.select("resname BMIM or resname BF4")
types = group_types_from_selections(
    system,
    sel_ions,
    "resid",
    ["resname BMIM", "resname BF4"],
)

ip = IonPairCorrelationPlan(
    sel_ions,
    rclust_cat=6.0,
    rclust_ani=6.0,
    group_by="resid",
    cation_type=0,
    anion_type=1,
    max_cluster=10,
    group_types=types,
    lag_mode="multi_tau",
)

time, data = ip.run(traj, system)
```

### Structure factor

```python
from warp_md import StructureFactorPlan

sf = StructureFactorPlan(sel_chain_a, bins=200, r_max=6.0, q_bins=100, q_max=20.0, pbc="orthorhombic")

r, g, q, s = sf.run(traj, system)
```

### Water occupancy grid

```python
from warp_md import WaterCountPlan

water_sel = system.select("resname SOL and name OW")
center_sel = system.select("resname SOL")

# box_unit is the grid cell size (Angstrom)
# region_size is the full grid size (Angstrom)
water = WaterCountPlan(
    water_sel,
    center_sel,
    box_unit=(1.0, 1.0, 1.0),
    region_size=(30.0, 30.0, 30.0),
)

out = water.run(traj, system)
print(out["dims"], out["mean"].shape)
```

### Equipartition temperature

Requires velocities in the trajectory (if missing, result will be 0).

```python
from warp_md import EquipartitionPlan

equip = EquipartitionPlan(sel_chain_a, group_by="resid")

time, data = equip.run(traj, system)
```

### Hydrogen bond counts

Distance and angle filters are GPU-accelerated when CUDA is available.

```python
from warp_md import HbondPlan

hbond = HbondPlan(
    donors=system.select("name N"),
    acceptors=system.select("name O"),
    dist_cutoff=3.5,
    angle_cutoff=None,
)

time, data = hbond.run(traj, system)
```

Angle-filtered example (hydrogens must be 1:1 with donors):

```python
hbond = HbondPlan(
    donors=system.select("name N"),
    acceptors=system.select("name O"),
    hydrogens=system.select("name H"),
    dist_cutoff=3.5,
    angle_cutoff=150.0,
)
```

## 10) Builder helpers (charges + group types)

Use helpers to avoid hand-editing topology files.

```python
from warp_md import charges_from_selections, charges_from_table, group_types_from_selections

# Charges from selection rules
charges = charges_from_selections(system, [
    {"selection": "resname NA", "charge": 1.0},
    {"selection": "resname CL", "charge": -1.0},
])

# Charges from CSV/TSV (columns: resname, name, charge)
charges = charges_from_table(system, "charges.csv")

# Group types for multi-species analyses
sel = system.select("resname NA or resname CL")
group_types = group_types_from_selections(system, sel, "resid", ["resname NA", "resname CL"])
```

## 11) CPU vs CUDA

All plans accept `device`:

- `"auto"` (default): use CUDA if available, else CPU
- `"cpu"`
- `"cuda"` or `"cuda:0"`

```python
rg = RgPlan(sel_ca).run(traj, system, device="cpu")
rg = RgPlan(sel_ca).run(traj, system, device="cuda:0")
```

## 12) Tips and troubleshooting

- If you see "atom count does not match system", ensure the same topology is used for trajectory.
- If your DCD is in nm, pass `length_scale=10.0` to `Trajectory.open_dcd`.
- For large trajectories, increase `chunk_frames` to reduce overhead.
- If CUDA fails to initialize, fall back to `device="cpu"`.
