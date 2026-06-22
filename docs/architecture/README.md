---
description: Rust crate design, GPU architecture, and agent-first contract principles
icon: sitemap
---

# Architecture

warp-md is a **Rust workspace** with **optional CUDA acceleration** and **Python bindings** — designed agent-first from the ground up.

---

## Crate Map

```
warp-md workspace
├── traj-core       # System topology, selection engine, boxes, errors
├── traj-io         # Streaming molecular IO with bounded frame buffers
├── traj-engine     # Plans, executor, CPU/GPU dispatch, correlators
├── traj-gpu        # CUDA context + buffer management (optional)
├── traj-kernels    # NVRTC-compiled CUDA kernels at runtime
├── traj-py         # PyO3 bindings — 160 exposed Plan classes
├── warp-build      # Polymer construction stage + topology graphs
├── warp-pack      # World-build engine — solvent, ions, box, morphology
├── warp-cg         # Martini coarse-graining with xTB + BO/PSO tuning
└── warp-pep        # Peptide builder + mutation engine
```

### Safety Boundary

| Crate | `unsafe` Policy |
|-------|----------------|
| `traj-core` | `#![forbid(unsafe_code)]` |
| `traj-io` | Isolated `unsafe` for low-level IO |
| `traj-gpu` | Isolated `unsafe` for CUDA runtime calls |

---

## Agent-First Design

The agent-facing CLIs expose typed JSON contracts for requests, discovery,
streaming events, and results.

### Contract Layers

| Layer | Schema | Consumer |
|-------|--------|----------|
| Request | `RunRequest` (Pydantic) | Agent → CLI |
| Streaming | `StreamEvent` (NDJSON) | Real-time progress |
| Result | `RunEnvelope` (JSON) | CLI → Agent |
| Discovery | `ContractCatalog` (JSON) | Agent → runtime |
| MCP | Tool definitions (JSON) | LLM → tools |

See the [full agent contract](agent-contract.md) for complete schemas, error codes, determinism guarantees, and cross-tool contracts.

### Determinism Guarantees

- Same config → same results (given deterministic mode)
- Exit codes are machine-parseable (0=ok, 2=config, 3=spec, 4=runtime, 5=internal)
- Agent result artifacts include SHA-256 metadata when emitted through the run contract
- Plot recommendations are Rust-native deterministic UI hints

---

## GPU Architecture

CUDA is **optional** and **runtime-compiled**:

```
traj-kernels (CUDA source)
    ↓ NVRTC compilation at first GPU run
    ↓ cached in memory for subsequent runs
    ↓ graceful CPU fallback on failure
```

### Accelerated Paths

| Category | GPU Plans |
|----------|-----------|
| Core | Rg, RMSD, MSD, RDF |
| Polymer | End-to-end, contour length, chain Rg, bond histograms |
| Transport | Conductivity, Dielectric, Dipole alignment |
| Spatial | Water occupancy, H-bond counts |

### Device Selection

```python
plan.run(traj, system, device="auto")   # CUDA if available, else CPU
plan.run(traj, system, device="cpu")    # Force CPU
plan.run(traj, system, device="cuda")   # Force GPU
```

---

## Correlator Architecture

Time-series plans (MSD, RotAcf, Conductivity) use `traj-engine`'s correlator system:

| Mode | Memory | Precision | Use Case |
|------|--------|-----------|----------|
| `fft` | Full series | Exact all lags | Short trajectories, uniform dt |
| `ring` | O(max_lag) | Exact up to max_lag | Validation, short lags |
| `multi_tau` | O(log N) | Log-spaced lags | Long trajectories, bounded memory |
| `auto` | Variable | Best-effort | Default — FFT if fits, else multi-tau |

---

## IO Pipeline

```
File → mmap/chunk reader → frame buffer → plan executor → output
        ↓                        ↓
    traj-io parses            traj-engine dispatches
    (streaming/chunked)       (CPU thread or CUDA kernel)
```

All coordinates are normalized to **Angstrom** internally:
- PDB: native Angstrom
- GRO: nm × 10
- DCD: native, optional `length_scale`
- XTC: nm × 10

---

## Key Design Decisions

1. **Rust core, Python UX** — heavy lifting in safe Rust, thin PyO3 wrappers
2. **GPU-optional** — CUDA is a build feature, never a runtime requirement
3. **Streaming IO** — bounded frame processing is available for large trajectories
4. **Contract-first** — every response is a validated JSON schema
5. **Composable Plans** — Plans are independent, combinable, GPU-aware
6. **Deterministic by default** — same inputs → same outputs
