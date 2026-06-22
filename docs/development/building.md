---
description: Full build guide — Rust, Python bindings, CUDA, and packaging
icon: hammer
---

# Building

---

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | 1.75+ | Compile workspace crates (`Cargo.toml` MSRV) |
| Python | 3.9+ | Bindings and tests |
| pip | latest | Install `maturin` |
| CUDA Toolkit | 12.x (optional) | GPU acceleration |

---

## Rust Build

```bash
# Debug build (fast iteration)
cargo build

# Release build (optimized)
cargo build --release

# Build all features
cargo build --release --all-features

# Test all crates
cargo test

# Test a specific crate
cargo test -p warp-pack

# Test with CUDA (requires CUDA_HOME)
cargo test -p traj-engine --features cuda
```

---

## Python Bindings (maturin)

```bash
# Install maturin
pip install maturin

# Develop mode (editable install, fast iteration)
maturin develop

# Release mode
maturin develop --release

# Clean build
maturin develop --release --force

# Verify
python -c "import warp_md; print(warp_md.System)"
```

---

## CUDA Build

```bash
# Set CUDA home
export CUDA_HOME=/usr/local/cuda

# Build with CUDA features
cargo build --release --features cuda

# Test GPU paths
cargo test -p traj-engine --features cuda
```

### CUDA Troubleshooting

| Problem | Check |
|---------|-------|
| CUDA not detected | `nvcc --version` and `echo $CUDA_HOME` |
| nvrtc not found | Ensure CUDA toolkit is in `LD_LIBRARY_PATH` |
| GPU OOM | Reduce `chunk_frames` or switch to CPU |

---

## Python Tests

```bash
# Run all Python tests
python -m pytest python/warp_md/tests

# Run a specific test file
python -m pytest python/warp_md/tests/test_align.py

# Run with verbose output
python -m pytest python/warp_md/tests -v
```

---

## Packaging

warp-md uses `pyproject.toml` with maturin:

```bash
# Build wheel
maturin build --release

# Build with CUDA features
maturin build --release --features cuda

# Output location
# target/wheels/warp_md-*.whl
```

---

## Full CI Workflow

```bash
# 1. Clean build
cargo clean
maturin clean

# 2. Build and test Rust
cargo test --release

# 3. Build Python bindings
maturin develop --release

# 4. Run Python tests
python -m pytest python/warp_md/tests

# 5. Run benchmarks from the dedicated environment, if configured
.agent/bench-venv/bin/python scripts/bench/benchmark_readers.py --help
```
