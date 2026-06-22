---
description: Deep dive into each Rust crate — purpose, API, and safety
icon: cube
---

# Rust Crates

Every crate in the warp-md workspace, with its purpose, public API surface, and safety policy.

---

## `traj-core`

**Purpose:** Foundational types — system topology, selection engine, box geometry, error handling.

**Safety:** `#![forbid(unsafe_code)]`

### Key Types

| Type | Description |
|------|-------------|
| `System` | Topology container — atoms, residues, chains, box |
| `Selection` | Boolean mask over atoms — created by `System::select()` |
| `Box` | Orthorhombic/triclinic simulation cell |
| `UnitCell` | Box vectors + angles |
| `AtomRecord` | Per-atom properties: name, element, resid, chain, mass, charge |
| `Error` / `Result` | Typed error handling |

### Selection Language

Predicates: `name`, `resname`, `resid`, `chain`, `protein`, `backbone`
Boolean: `and`, `or`, `not`, `()` — evaluated into a bitmask over atoms.

---

## `traj-io`

**Purpose:** Molecular IO — read/write PDB, GRO, DCD, XTC, TRR, and 10+ formats.

**Safety:** Isolated `unsafe` for low-level binary parsing.

### Reader Architecture

```
File → mmap (optional) → chunk reader → frame iterator
```

All formats stream frames without loading the full trajectory into memory.

### Supported Formats

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| PDB | ✓ | ✓ | CONECT, TER, CRYST1 |
| GRO | ✓ | ✓ | nm → Angstrom auto-conversion |
| DCD | ✓ | — | CHARMM/NAMD format |
| XTC | ✓ | — | GROMACS compressed |
| TRR | ✓ | — | High-precision + velocities |
| PDBx/mmCIF | ✓ | ✓ | Large-structure fallback |
| XYZ | ✓ | ✓ | Simple coordinate format |
| MOL2 | ✓ | ✓ | Tripos format |
| LAMMPS | ✓ | ✓ | Data format |
| CRD | ✓ | ✓ | Amber coordinates |
| G96 | ✓ | ✓ | GROMOS96 |
| H5MD | ✓ | — | HDF5-based MD |
| TNG | ✓ | — | Next-gen GROMACS |
| CPT | ✓ | — | Checkpoint |
| PDBQT | ✓ | — | AutoDock Vina |

---

## `traj-engine`

**Purpose:** Plan execution, CPU/GPU dispatch, correlator system, feature store.

### Plan Trait

Every analysis implements `Plan`:
```rust
pub trait Plan {
    fn name(&self) -> &'static str;
    fn run(&self, traj: &dyn Trajectory, system: &System, device: Device) -> Result<PlanOutput>;
}
```

### Correlator System

Time-series plans use pluggable correlators:
- `RingCorrelator` — exact up to max_lag, O(M) memory
- `MultiTauCorrelator` — log-spaced lags, O(log N) memory
- `FftCorrelator` — exact all lags via convolution, O(N log N)

### Feature Store

Persist per-frame features for offline analysis:
- `FeatureStoreWriter` — writes `.bin` + `.json` index
- `FeatureStoreReader` — reads index + data

---

## `traj-gpu`

**Purpose:** CUDA context management, buffer allocation, kernel dispatch.

**Safety:** Isolated `unsafe` for CUDA runtime calls.

### Architecture

```
traj-gpu
├── CudaContext       — device handle, stream, nvrtc
├── CudaBuffer<T>     — device memory with pinned host staging
├── DeviceMemoryPool  — reusable allocation
└── KernelCache       — NVRTC-compiled kernels (per-architecture)
```

### Device Dispatch

```rust
pub enum Device {
    Auto,          // CUDA if available, else CPU
    Cpu,           // Force CPU
    Cuda(u32),     // Specific GPU index
}
```

---

## `traj-kernels`

**Purpose:** CUDA kernel source for NVRTC runtime compilation.

Kernels are compiled at **first GPU run**, not pre-packaged as `.cubin`/`.ptx`. This ensures optimal machine code for the specific GPU architecture.

### Kernel Categories

| Kernel | Plans Using It |
|--------|----------------|
| `rg_accum` / `rg_sumsq` / `rg_finalize` | RgPlan |
| `rmsd_centroid` / `rmsd_cov` / `rmsd_finalize` | RmsdPlan |
| `msd_accum` / `msd_time_lag` | MsdPlan |
| `rdf_hist` | RdfPlan |
| `pairwise_distance` | PairwiseDistancePlan |
| `polymer_bond_hist` | BondLengthDistributionPlan |
| `group_com_accum` / `group_com_finalize` | Group COM stages used by transport plans |

---

## `traj-py`

**Purpose:** Python bindings via PyO3 — currently exposes 160 Plan classes plus functional helpers.

### Key Classes

| Python Class | Rust Backend | GPU |
|-------------|--------------|-----|
| `System` | `traj-core::System` | — |
| `Trajectory` | `traj-io` readers | — |
| `RgPlan` | `traj-engine::RgPlan` | ✓ |
| `RmsdPlan` | `traj-engine::RmsdPlan` | ✓ |
| `MsdPlan` | `traj-engine::MsdPlan` | ✓ |
| `RdfPlan` | `traj-engine::RdfPlan` | ✓ |
| `ConductivityPlan` | `traj-engine::ConductivityPlan` | ✓ |
| `DsspPlan` | `traj-engine::DsspPlan` | — |

---

## `warp-build`

**Purpose:** Polymer construction — compile source bundles into chains, emit topology graphs and handoff manifests.

### Build Pipeline

```
Source Bundle → Validate → Coordinate Build → Topology Graph → Handoff Manifest
```

### Handoff Tiers

| Tier | Meaning |
|------|---------|
| `md_ready` | Transferable topology + charges |
| `forcefield_backed` | Validated ffxml fallback |
| `minimizable_synthetic` | Clean geometry, synthetic topology |
| `graph_bonded_only` | Coordinates + graph only |

---

## `warp-pack`

**Purpose:** World-build engine — pack molecules into boxes, solvate, add ions.

Packmol-compatible configuration but pure Rust. Supports 11+ input formats, 7 output formats, GENCAN optimization, restart files.

---

## `warp-cg`

**Purpose:** Martini coarse-graining — map atomistic molecules to CG beads, tune bonded parameters via BO/PSO, emit ITP/TOP artifacts.

### Workflows

- SMILES → xTB reference → mapping → tuning
- External trajectory → mapping → BO/PSO tuning
- Source manifest → replay mapping → topology
- CG system building (membranes, bilayers, solvation zones)

---

## `warp-pep`

**Purpose:** Peptide construction — build all-atom peptides from sequences, apply mutations, export to any format.

Internal coordinate geometry (bond lengths, angles, dihedrals). Supports all 20 amino acids, Amber variants, D-amino acids, Ramachandran presets, multi-chain structures, disulfide detection.
