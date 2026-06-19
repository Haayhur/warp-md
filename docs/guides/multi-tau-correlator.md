---
description: High-performance correlators for computing time-series metrics over infinite trajectories
icon: waveform
---

# Multi-Tau & Time-series Correlators

When computing time-dependent properties (like Mean Squared Displacement, Velocity Autocorrelation, or Current), `traj-engine` provides highly optimized correlators that balance memory, precision, and performance.

## Correlation Modes (`LagMode`)

You can control how the trajectory engine buffers and correlates data over time frames using `LagSettings`. The default is `Auto`, which intelligently selects the best underlying correlator based on the available memory budget.

### 1. `Auto` (Default)
`Auto` mode estimates the required memory footprint for your metric (e.g., measuring Conductivity). 
- If the full lag buffer fits within `memory_budget_bytes` (default 512 MB), it selects **Ring** or **FFT** based on algorithmic complexity constraints.
- If it exceeds the memory budget (often true for massive trajectories with long lag times), it automatically decays into **MultiTau** correlation.

### 2. `MultiTau`
The **Multiple-Tau (Multi-Tau) Correlator** computes correlations on a logarithmic scale. 
- It maintains exact data for early, short-time lags.
- As the lag time $t$ increases, the correlator hierarchically decimates (averages) adjacent frames into coarser "levels".
- This provides deep, long-time correlation bounds across millions of frames using $O(\log N)$ memory, avoiding the massive memory overhead of standard linear correlation.
- **Parameters**: `multi_tau_m` (points per level, default 16) and `multi_tau_max_levels` (depth of hierarchy, default 20).

### 3. `Ring` (Ring-Buffer)
The **Ring-Buffer Correlator** maintains a sliding window of exactly `max_lag` frames in a circular queue.
- Computes exact linear correlations.
- Ideal for short `max_lag` queries, but scales memory linearly $O(M)$ where $M$ is `max_lag`.
- If your requested `max_lag` exceeds the `memory_budget_bytes`, the buffer is capped to prevent OOM errors unless forced.

### 4. `FFT`
The **Fast Fourier Transform Correlator** uses the convolution theorem to compute full auto-correlations in $O(N \log N)$ time rather than $O(N^2)$ direct summation.
- Computes exact correlations over the entire trajectory.
- Requires loading the target metric (not the whole trajectory, but the measured scalars/vectors) entirely into memory.
- Strictly bounds memory; if the target arrays exceed `memory_budget_bytes`, `Auto` mode will aggressively fall back to `MultiTau`.

## Usage in Plans

Most time-dependent execution plans (e.g., `ConductivityPlan`, `MsdPlan`, `VelocityAutoCorrPlan`) accept `LagSettings` natively.

```python
# Conceptual Agent Payload Example for a time-dependent plan:
{
  "max_lag": 50000,
  "lag_mode": "auto",
  "memory_budget_bytes": 1073741824 # 1 GB
}
```
