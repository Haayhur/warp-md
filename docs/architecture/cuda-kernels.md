---
description: NVRTC runtime compilation, kernel architecture, and GPU dispatch
icon: bolt
---

# CUDA Kernels

warp-md compiles its bundled CUDA source at runtime using NVIDIA NVRTC. The
repository ships source fragments rather than precompiled `.cubin` or `.ptx`
artifacts.

---

## Runtime Compilation Flow

```
traj-kernels/*.cu (source)
    ↓ NVRTC compile (first GPU run)
    ↓ Cached in memory
    ↓ Reused for subsequent runs
    ↓ Dispatch through traj-gpu
```

### Why NVRTC?

1. **Runtime targeting** — compiles for the active CUDA environment
2. **No binary bloat** — no shipping `.cubin`/`.ptx` for every architecture
3. **Single source bundle** — `part1.cu` through `part6.cu` are concatenated by `traj-kernels`

---

## Kernel Inventory

### Core Analysis Kernels

| Kernel | Plans | Input Size | Output |
|--------|-------|-----------|--------|
| `rg_accum`, `rg_sumsq`, `rg_finalize` | RgPlan | Coordinates + masses | Rg per frame |
| `rmsd_centroid`, `rmsd_cov`, `rmsd_finalize` | RmsdPlan | Coordinates + reference | RMSD stages |
| `msd_accum`, `msd_time_lag` | MsdPlan | Coordinates or group COM series | MSD values |
| `rdf_hist` | RdfPlan | Pair coordinates | Histogram bins |

### Polymer Kernels

| Kernel | Plans | Description |
|--------|-------|-------------|
| `polymer_end_to_end` | EndToEndPlan | First-to-last atom distance per chain |
| `polymer_contour_length` | ContourLengthPlan | Summed bond lengths per chain |
| `polymer_chain_rg` | ChainRgPlan | Per-chain radius of gyration |
| `polymer_bond_hist` | BondLengthDistributionPlan | Bond length histogram |
| `polymer_angle_hist` | BondAngleDistributionPlan | Bond angle histogram |

### Transport Kernels

| Kernel | Plans | Description |
|--------|-------|-------------|
| `group_com_accum`, `group_com_finalize` | ConductivityPlan and other grouped plans | Center of mass stages |
| `group_dipole_accum`, `group_dipole_finalize` | DielectricPlan, DipoleAlignmentPlan | Group dipole stages |
| `orientation_plane`, `orientation_vector` | RotAcfPlan | Orientation extraction |

### Spatial Kernels

| Kernel | Plans | Description |
|--------|-------|-------------|
| `water_count` | WaterCountPlan | 3D occupancy grid update |
| `hbond_count`, `hbond_count_angle` | HbondPlan | Distance and angle filters |

---

## Kernel Design Patterns

### Per-Frame Reduction

```cuda
// Typical pattern: one thread per atom, one block per frame
__global__ void rg_accum(
    float3* coords,    // (n_frames * n_atoms)
    float*  out,       // (n_frames,)
    float3  center,
    int     n_atoms
) {
    int frame = blockIdx.x;
    int tid   = threadIdx.x;
    // ... compute per-atom contribution, warp-reduce, atom add
}
```

### Pairwise Distance

```cuda
// RDF: one block per frame, grid-stride over atom pairs
__global__ void rdf_hist(
    float3* coords_a, float3* coords_b,
    float*  hist,
    int     n_a, int n_b,
    float   r_max, int bins
) {
    // ... shared memory blocking for atom pairs
}
```

### Histogram

```cuda
// Bond length histogram: one block per frame
__global__ void polymer_bond_hist(
    float3* coords,    // (n_frames, n_atoms)
    int*    bonds,     // (n_bonds, 2) atom index pairs
    float*  hist,      // (bins,) output per frame
    int     n_bonds, int bins, float r_max
) {
    // shared memory atom add for histogram bins
}
```

---

## CPU-Only Plans

These remain on CPU — not every problem parallelizes well:

- Persistence length (serial correlation chain)
- DSSP (phi/psi classification per residue)
- PCA / covariance matrix construction
- Wavelet transform
- Energy decomposition
- Some reduction steps

---

## Device Selection

```python
# Automatic — CUDA if available
plan.run(traj, system, device="auto")

# Force CPU
plan.run(traj, system, device="cpu")

# Force specific GPU
plan.run(traj, system, device="cuda:0")
```

### Auto Mode Logic

```
device="auto":
1. Check if CUDA feature is compiled into binary
2. Query driver for available devices
3. If device found → initialize CUDA context
4. If CUDA is unavailable → select the CPU path
```

---

## Troubleshooting

```bash
# Hide GPU for CPU testing
export CUDA_VISIBLE_DEVICES=""

# Verify CUDA toolkit
nvcc --version

# Check NVRTC availability
ldd /path/to/warp-md | grep nvrtc
```

| Issue | Cause | Fix |
|-------|-------|-----|
| CUDA not detected | `CUDA_HOME` not set | `export CUDA_HOME=/usr/local/cuda` |
| OOM on GPU | Too many atoms per chunk | Reduce `chunk_frames` |
| Kernel compilation fails | NVRTC not on `LD_LIBRARY_PATH` | Ensure CUDA toolkit in library path |
