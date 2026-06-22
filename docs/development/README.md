---
description: Contributing, building, testing, and validation guidelines
icon: code
---

# Development

Guidelines for contributing to warp-md — from building locally to running the full validation suite.

---

## Quick Start

```bash
git clone https://github.com/haayhur/warp-md.git
cd warp-md

# Build everything
cargo build --release

# Build Python bindings
maturin develop

# Verify
python -c "import warp_md; print(warp_md.System)"
```

---

## What's Here

| Page | What It Covers |
|------|----------------|
| [Contributing](contributing.md) | PR workflow, coding style, commit conventions |
| [Building](building.md) | Full build guide — Rust, Python, CUDA |
| [Testing](testing.md) | Rust tests, Python tests, property tests, golden data |
| [Validation](validation.md) | Benchmark manifests, scientific parity, completion audits |
