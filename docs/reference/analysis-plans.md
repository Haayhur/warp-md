---
description: The core 4 — Rg, RMSD, MSD, RDF — your agent's bread and butter
icon: chart-simple
---

# Analysis Plans

The core 4 trajectory analyses. These are your agent's bread and butter.

{% hint style="info" %}
This page covers the **foundational 4 Plans**. warp-md ships 96 Plans total — see [Structural Analysis](structural-analysis.md), [Geometry & Distance](geometry-distance.md), [Solvation & Density](solvation-density.md), [Transforms](transforms.md), and [NMR & Spectroscopy](nmr-spectroscopy.md) for the rest.
{% endhint %}

---

## RgPlan

*Radius of gyration over time — how compact is this thing?*

```python
from warp_md import RgPlan

plan = RgPlan(selection, mass_weighted=False)
rg = plan.run(traj, system, chunk_frames=None, device="auto")
```

### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |
| `mass_weighted` | `bool` | `False` | Weight by atomic mass |

### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`
- **Unit**: Angstrom

---

## RmsdPlan

*Root mean square deviation from reference — how different is this from the start?*

```python
from warp_md import RmsdPlan

plan = RmsdPlan(selection, reference="topology", align=True)
rmsd = plan.run(traj, system)
```

### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |
| `reference` | `str` | `"topology"` | `"topology"` or `"frame0"` |
| `align` | `bool` | `True` | Superimpose before RMSD |

### Output

- **Type**: 1D array
- **Shape**: `(n_frames,)`
- **Unit**: Angstrom

---

## MsdPlan

*Mean square displacement — how far is stuff diffusing?*

```python
from warp_md import MsdPlan

plan = MsdPlan(
    selection,
    group_by="resid",
    axis=None,
    length_scale=None,
    frame_decimation=None,
    dt_decimation=None,
    time_binning=None,
    group_types=None,
    lag_mode=None,
    max_lag=None,
    memory_budget_bytes=None,
    multi_tau_m=None,
    multi_tau_levels=None,
)
time, data = plan.run(traj, system)
```

### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `selection` | `Selection` | required | Atoms to analyze |
| `group_by` | `str` | `"resid"` | Grouping mode |
| `axis` | `list[float]` | `None` | Project onto axis [x, y, z] |
| `frame_decimation` | `tuple` | `None` | `(start, stride)` |
| `dt_decimation` | `tuple` | `None` | `(cut1, stride1, cut2, stride2)` |
| `lag_mode` | `str` | `None` | `"auto"`, `"multi_tau"`, `"ring"`, `"fft"` |
| `group_types` | `list[int]` | `None` | Per-group type IDs |

### Output

- `time`: 1D array (dt bins)
- `data`: 2D array `(rows, cols)`
  - `cols = components × (n_types + 1)`
  - Components: x, y, z, [axis], total
  - Per component: type0, type1, ..., total

---

## RdfPlan

*Radial distribution function — what's the local structure?*

```python
from warp_md import RdfPlan

plan = RdfPlan(sel_a, sel_b, bins=200, r_max=10.0, pbc="orthorhombic")
r, g, counts = plan.run(traj, system)
```

### Parameters

| Parameter | Type | Default | What It Does |
|-----------|------|---------|--------------|
| `sel_a` | `Selection` | required | First selection |
| `sel_b` | `Selection` | required | Second selection |
| `bins` | `int` | required | Number of bins |
| `r_max` | `float` | required | Maximum distance (Å) |
| `pbc` | `str` | `"orthorhombic"` | PBC handling |

### Output

- `r`: bin centers (Å)
- `g`: g(r) values — the actual RDF
- `counts`: raw pair counts (for the curious)

---

## Lag Mode Reference

For MSD and other correlator-based plans:

| Mode | Description | Memory | When to Use |
|------|-------------|--------|-------------|
| `"auto"` | FFT if fits, else multi-tau | Variable | Default — let warp-md decide |
| `"multi_tau"` | Log-spaced lags | Bounded | Long trajectories |
| `"ring"` | Exact up to max_lag | Linear | Short-time validation |
| `"fft"` | Exact all lags | Full series | Short trajectories |

### Tuning Parameters

| Parameter | Default | What It Does |
|-----------|---------|--------------|
| `max_lag` | 100000 | Ring mode max lag |
| `memory_budget_bytes` | - | Cap max_lag and FFT decision |
| `multi_tau_m` | 16 | Multi-tau resolution |
| `multi_tau_levels` | 20 | Multi-tau levels |
