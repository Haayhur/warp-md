---
description: iRED, order parameters, J-coupling, IR spectra, lipid SCD — experimental observables from simulation
icon: wave-square
---

# NMR & Spectroscopy

Bridge the gap between simulation and experiment. Compute NMR observables, IR spectra, and lipid order parameters directly from your trajectories.

{% hint style="info" %}
These tools compute **experimental observables** from MD — the quantities that let you validate your force field against real data.
{% endhint %}

---

## NMR Order Parameters

### NmrIredPlan

*iRED (isotropic Reorientational Eigenmode Dynamics) — compute NMR order parameters without rotational diffusion assumptions.*

```python
from warp_md import NmrIredPlan

plan = NmrIredPlan(selection)
result = plan.run(traj, system)
```

---

### Functional NMR API

```python
from warp_md.analysis import (
    ired_vector_and_matrix,
    calc_ired_vector_and_matrix,
    nh_order_parameters,
    calc_nh_order_parameters,
    jcoupling,
)

# Compute iRED vectors and matrix
vectors, matrix = ired_vector_and_matrix(traj, system, selection)
# Or the calc_ variant
vectors, matrix = calc_ired_vector_and_matrix(traj, system, selection)

# N-H order parameters (S²)
s2 = nh_order_parameters(traj, system, selection)
# Or the calc_ variant
s2 = calc_nh_order_parameters(traj, system, selection)

# J-coupling constants
j = jcoupling(traj, system, selection)
```

{% hint style="info" %}
**N-H order parameters (S²)** range from 0 (fully flexible) to 1 (rigid). Values below 0.8 typically indicate loop regions or disordered segments.
{% endhint %}

---

## Correlation Functions

### AtomicCorrPlan

*Atomic correlation functions — cross-correlations between atom motions.*

```python
from warp_md import AtomicCorrPlan

plan = AtomicCorrPlan(selection)
result = plan.run(traj, system)
```

---

### VelocityAutoCorrPlan

*Velocity autocorrelation function — the starting point for vibrational spectra.*

```python
from warp_md import VelocityAutoCorrPlan

plan = VelocityAutoCorrPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.velocity_autocorrelation()`

---

### XcorrPlan

*Cross-correlation between two time series.*

```python
from warp_md import XcorrPlan

plan = XcorrPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.xcorr()`

---

### Functional Correlation API

```python
from warp_md.analysis import acorr, xcorr, timecorr

# Autocorrelation
acf = acorr(data)

# Cross-correlation
ccf = xcorr(data1, data2)

# Time correlation (general)
tcf = timecorr(data)
```

---

## Spectroscopy

### Infrared Spectra

*IR spectra from dipole moment autocorrelation — vibrational analysis from dynamics.*

```python
from warp_md.analysis import infraredspec

ir_spectrum = infraredspec(traj, system, selection)
```

---

### Lipid Order Parameters (SCD)

*Deuterium order parameters for lipid tails — membrane biophysics essential.*

```python
from warp_md.analysis import lipidscd

scd = lipidscd(traj, system, selection)
```

{% hint style="info" %}
Lipid SCD values typically range from -0.5 to 0. More negative = more ordered. Values near 0 indicate disordered tail ends.
{% endhint %}

---

## Signal Processing

### WaveletPlan

*Wavelet analysis — time-frequency decomposition of trajectory signals.*

```python
from warp_md import WaveletPlan

plan = WaveletPlan(selection)
result = plan.run(traj, system)
```

Also available as: `warp_md.analysis.wavelet()`

---

## Rotational Diffusion

```python
from warp_md.analysis import rotdif

# Compute rotational diffusion tensor
D_rot = rotdif(traj, system, selection)
```

---

## Energy Analysis

### Energy Analysis & Decomposition

*Parse and analyze energy outputs from simulations.*

```python
from warp_md.analysis import energy_analysis, ene_decomp, esander, lie, ti

# General energy analysis
energies = energy_analysis(traj, system)

# Energy decomposition
decomp = ene_decomp(traj, system)

# Linear Interaction Energy
lie_result = lie(traj, system)

# Thermodynamic Integration
ti_result = ti(traj, system)
```

---

## Complete NMR Validation Example

```python
from warp_md import System, Trajectory, NmrIredPlan
from warp_md.analysis import nh_order_parameters, jcoupling

system = System.from_pdb("protein.pdb")
traj = Trajectory.open_xtc("protein.xtc", system)
backbone = system.select("backbone")

# iRED order parameters
ired = NmrIredPlan(backbone).run(traj, system)

# N-H S² values
s2 = nh_order_parameters(traj, system, backbone)

# J-coupling constants
j = jcoupling(traj, system, backbone)

print("Experimental observables computed — ready for force field validation 🧪")
```
