---
description: Testing guidelines — Rust tests, Python tests, property tests, and golden data
icon: vial
---

# Testing

---

## Rust Tests

```bash
# Run all workspace tests
cargo test

# Test a specific crate
cargo test -p traj-core
cargo test -p traj-io
cargo test -p traj-engine

# Test with CUDA
cargo test -p traj-engine --features cuda

# Test a specific test name
cargo test test_rmsd
```

### Test Organization

- Unit tests live in `src/tests/` or inline `#[cfg(test)] mod tests`
- Integration tests in `tests/` at crate root
- IO fixtures in `crates/*/tests/data/`

### What to Test

- Golden correctness vs reference implementation (MDAnalysis, MDTraj)
- IO round-trips (write → read → compare)
- Property-based invariants (RMSD(ref, ref) = 0, RDF normalization)
- Edge cases: empty selections, single atom, single frame
- Error paths: missing files, atom count mismatch

---

## Python Tests

```bash
# Run all
python -m pytest python/warp_md/tests

# Run specific file
python -m pytest python/warp_md/tests/test_align.py

# Verbose
python -m pytest python/warp_md/tests -v
```

### Test Names

Use clear, domain-specific names such as `test_align.py`, `test_rdf.py`, and
`test_system.py`.

---

## Benchmark Tests

Benchmarks live in `scripts/bench/`. The repository convention is a dedicated
environment at `.agent/bench-venv`; create it locally if it is not present:

```bash
python -m venv .agent/bench-venv
.agent/bench-venv/bin/python -m pip install -e .
source .agent/bench-venv/bin/activate

# Reader benchmark (XTC)
.agent/bench-venv/bin/python scripts/bench/benchmark_readers.py \
  --top results/alanine/new/peptide_solvated.pdb \
  --traj results/alanine/new/peptide_sim.xtc \
  --frames 5000 --repeats 3

# Transport benchmark
.agent/bench-venv/bin/python scripts/bench/benchmark_transport_metrics.py \
  --top results/water/water_box.pdb \
  --traj results/water/water_sim.xtc \
  --repeats 3 --max-lag 256 --device cpu
```

---

## Validation Suite

Full validation is defined in `internal/validation/`:

```bash
# Publication bundle
.agent/bench-venv/bin/python scripts/bench/run_publication_bundle.py

# Build report
.agent/bench-venv/bin/python scripts/bench/build_benchmark_report.py
```

### Validation Categories

| Category | What's Checked |
|----------|----------------|
| Scientific parity | Accuracy vs MDAnalysis, MDTraj, GROMACS |
| Performance | Speedups vs baselines, memory scaling |
| IO correctness | Round-trip parity for all formats |
| Agent contract | Schema validation, envelope format |
| GPU parity | CPU vs GPU numerical agreement |
