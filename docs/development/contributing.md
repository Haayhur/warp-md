---
description: How to contribute — PR workflow, coding style, and conventions
icon: git-merge
---

# Contributing

---

## PR Workflow

1. **Open an issue** — discuss the change before implementing
2. **Fork and branch** — `git checkout -b feature/your-feature`
3. **Make changes** — keep Rust files ~600-700 LOC, split modules when they grow
4. **Run tests** — `cargo test` and `python -m pytest python/warp_md/tests`
5. **Submit PR** — include brief summary, scope notes, test evidence

### What Makes a Good PR

- Clear description of the problem and solution
- Test evidence or rationale if tests are not yet available
- Architecture updates in `architect.md` if you add or rename modules
- No generated runtime artifacts in the review diff

---

## Coding Style

### Rust

- Follow `rustfmt` defaults (4-space indentation)
- `traj-core` must remain `#![forbid(unsafe_code)]`
- `unsafe` is isolated to `traj-io` and `traj-gpu` as specified in `architect.md`
- Prefer descriptive, domain-specific names (`Trajectory`, `FrameChunk`, `RgPlan`)
- Keep files ~600–700 LOC when possible

### Python

- PEP 8 (4-space indentation)
- Thin wrappers over Rust — heavy lifting is in Rust
- Clear test names like `test_align.py`

### Naming

- Rust: PascalCase for types, snake_case for functions
- Python: snake_case for functions and methods
- CLI commands: kebab-case (`list-plans`, `water-models`)

---

## Commit Conventions

- Prefer concise, imperative subjects: "Add DCD reader skeleton"
- Scope in parentheses when helpful: `traj-io(dcd): add atom count validation`
- No established convention for the single-commit history — use your judgment

---

## Architecture Alignment

- `architect.md` is the source of truth for MVP scope, crate boundaries, and performance constraints
- Align code changes with `architect.md`, or update it when making intentional deviations
- Contract-first for agent consumers: stable schemas, explicit errors, deterministic serialization

---

## Testing Expectations

- Golden correctness tests vs a reference implementation
- IO fixtures for PDB/GRO/DCD/XTC
- Property tests for invariants (e.g., RMSD(ref, ref) = 0)
- Keep tests close to code (`crates/*/tests` for Rust, `python/warp_md/tests` for Python)
