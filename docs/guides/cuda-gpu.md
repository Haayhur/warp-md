---
description: Turbo mode — GPU acceleration for your agent
icon: bolt
---

# CUDA & GPU Acceleration

When CPU isn't fast enough, warp-md has turbo mode. Optional CUDA acceleration for when your agent needs speed.

{% hint style="info" %}
**No GPU? No problem.** Without CUDA, warp-md gracefully falls back to CPU. Your agent's code runs everywhere.
{% endhint %}

---

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit (driver + nvrtc)
- `CUDA_HOME` or `CUDA_PATH` environment variable set

---

## Setup

{% stepper %}
{% step %}
## Verify CUDA Installation

```bash
nvcc --version
nvidia-smi
```

If these work, you're halfway there.
{% endstep %}

{% step %}
## Set Environment Variable

{% tabs %}
{% tab title="Linux/macOS" %}
```bash
export CUDA_HOME=/usr/local/cuda
```
{% endtab %}

{% tab title="Windows" %}
```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0"
```
{% endtab %}
{% endtabs %}
{% endstep %}

{% step %}
## Build with CUDA Feature

```bash
cargo test -p traj-engine --features cuda
```
{% endstep %}

{% step %}
## Verify GPU Detection

```python
from warp_md import RgPlan
# device="auto" will use CUDA if available
# device="cuda" or "cuda:0" to force GPU
```

Your agent is now turbocharged.
{% endstep %}
{% endstepper %}

---

## Device Selection

All analysis plans accept a `device` parameter:

```python
from warp_md import RgPlan

# Automatic (CUDA if available, else CPU)
rg = plan.run(traj, system, device="auto")

# Force CPU
rg = plan.run(traj, system, device="cpu")

# Force CUDA (first GPU)
rg = plan.run(traj, system, device="cuda")
rg = plan.run(traj, system, device="cuda:0")
```

---

## What Gets Accelerated?

| Category | GPU-Accelerated Analyses |
|----------|-------------------------|
| **Core** | Rg, RMSD, MSD, RDF |
| **Polymer** | End-to-end, contour length, chain Rg, bond histograms |
| **Orientation** | RotAcf (orientation extraction) |
| **Transport** | Conductivity (group COM), Dielectric, Dipole alignment |
| **Structure** | Ion-pair correlation, Structure factor (RDF kernel) |
| **Spatial** | Water occupancy grid |
| **Thermodynamic** | Equipartition (group kinetic energy) |
| **Bonding** | H-bond counts (distance + angle) |

{% hint style="warning" %}
**CPU-only**: Persistence length and some reductions remain CPU-based. Not everything parallelizes well.
{% endhint %}

---

## Performance Tips

{% hint style="success" %}
**Chunk frames**: For large trajectories, adjust `chunk_frames` to optimize memory transfer:

```python
rg = plan.run(traj, system, device="cuda", chunk_frames=1000)
```
{% endhint %}

{% hint style="success" %}
**Multi-tau for long trajectories**: Use bounded-memory mode for MSD/ACF:

```python
msd = MsdPlan(selection, group_by="resid", lag_mode="multi_tau")
```
{% endhint %}

---

## Troubleshooting

<details>
<summary>CUDA not detected</summary>

Set `CUDA_HOME` or `CUDA_PATH` to your CUDA installation:

```bash
export CUDA_HOME=/usr/local/cuda
```

The `cudarc` crate needs this breadcrumb to find the runtime.
</details>

<details>
<summary>GPU out of memory</summary>

Reduce `chunk_frames` or switch to CPU:

```python
rg = plan.run(traj, system, device="cpu")
```

Sometimes discretion is the better part of valor.
</details>

<details>
<summary>Kernel compilation fails</summary>

Ensure `nvrtc` is available. This ships with the CUDA Toolkit.
</details>

---

## Benchmarks

Typical speedups on modern GPUs (RTX 3080):

| Analysis | CPU Time | GPU Time | Speedup |
|----------|----------|----------|---------|
| Rg (10k frames) | 2.5s | 0.3s | ~8x |
| RDF (10k frames) | 45s | 3s | ~15x |
| MSD (100k frames) | 120s | 8s | ~15x |

{% hint style="info" %}
Actual speedup depends on system size, trajectory length, and hardware. Your mileage may vary — but it's usually impressive.
{% endhint %}
