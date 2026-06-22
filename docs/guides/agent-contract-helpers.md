---
description: Contract helpers for agent workflows — validation, normalization, suggestion, inspection, and capabilities
icon: terminal
---

# Agent Contract Helpers

Warp-md ships a set of contract-first CLI commands that help agents validate, normalize, inspect, and discover analyses without running full trajectories.

---

## CLI Overview

| Command | Purpose |
|---------|---------|
| `validate` | Validate a run request against the agent schema |
| `normalize` | Canonicalize request aliases and fill optional defaults |
| `inspect-inputs` | Verify all referenced input files exist and are readable |
| `lint-selection` | Validate selection expression syntax and evaluate against a topology |
| `capabilities` | Print build fingerprint for contract verification |
| `suggest` | Map natural-language goals to analysis recommendations |
| `bundle-plan` | Expand an advertised analysis bundle into individual analysis list |

---

## Request Validation (`validate`)

Validate a run request JSON against the agent schema before submitting it for execution.

```bash
warp-md validate request.json
warp-md validate request.json --stdin          # Read from stdin
warp-md validate request.json --kind result    # Validate result envelope instead
warp-md validate request.json --kind event     # Validate stream event
```

Returns a structured validation result with error codes, field-level warnings, and schema compliance status. Exit code 0 means valid, 2 means schema violations found.

---

## Request Normalization (`normalize`)

Canonicalize request aliases and fill optional defaults before execution.

```bash
warp-md normalize request.json
warp-md normalize request.json --strip-unknown  # Remove unrecognized fields
```

Alias mappings:

| Input Alias | Canonical Field |
|-------------|----------------|
| `topology` | `system` |
| `traj` | `trajectory` |
| `top` | `system` |

Missing optional fields are filled with their schema defaults. Use `--strip-unknown` to remove fields not recognized by the current schema version — useful when forwarding a request to a stricter consumer.

---

## Input Inspection (`inspect-inputs`)

Validate all referenced input files without running any analyses.

```bash
warp-md inspect-inputs request.json
warp-md inspect-inputs request.json --stdin
```

Checks:

- Topology file exists and is readable
- Trajectory file exists and is readable
- Any reference file paths (reference PDB, charge tables, etc.) exist
- Reports missing or unreadable paths with clear error messages

Use this before `run` to catch missing inputs early, especially in automated pipelines where a missing file would cause a runtime error.

---

## Selection Linting (`lint-selection`)

Validate a selection expression's syntax and optionally evaluate it against a topology to check atom counts.

```bash
warp-md lint-selection "name CA and protein"
warp-md lint-selection "name CA and protein" --topology system.pdb
```

Without `--topology`, checks only syntax validity (balanced parentheses, valid selector keywords, proper negation). With `--topology`, also evaluates the selection and reports the matched atom count. Useful for catching invalid selection expressions before submitting a long batch run.

---

## Capabilities Fingerprint (`capabilities`)

Print a deterministic build fingerprint for contract verification.

```bash
warp-md capabilities
warp-md capabilities --format json
```

Emits JSON or YAML containing the schema version, CLI version, advertised
analysis plans and bundles, error codes, catalog hash, and supported contract
features.

Agents can compare this fingerprint against a known-good baseline to verify that the installed binary matches the expected build configuration.

---

## Goal-Based Analysis Suggestion (`suggest`)

Map natural-language goals to concrete analysis recommendations without LLM inference.

```bash
warp-md suggest "protein stability"
warp-md suggest "protein stability" --top-n 3
warp-md suggest "protein stability" --format json
```

Uses deterministic keyword matching against `_GOAL_PHRASES` in `contract.py`. No LLM, no fuzzy matching — pure scoring rules.

### Built-in Goal Phrases

| Goal Phrase | Suggested Analyses |
|------------|-------------------|
| `radius of gyration` | `rg` |
| `secondary structure` | `dssp`, `helix` |
| `alpha helix` | `helix`, `helixorient`, `dssp` |
| `helix geometry` | `helix`, `helixorient` |
| `mean square displacement` | `msd` |
| `diffusion coefficient` | `diffusion`, `msd` |
| `pair distribution` / `radial distribution` | `rdf` |
| `hydrogen bond` / `hydrogen bonds` | `hbond` |
| `free volume` | `free_volume`, `bondi_ffv` |
| `fractional free volume` | `bondi_ffv` |
| `water shell` / `solvation shell` | `watershell` |
| `count water` / `water count` | `water_count` |
| `molecular docking` / `binding pose` / `ligand contacts` | `docking` |

The `docking` analysis is only suggested when the goal contains trigger words (`binding`, `dock`, `ligand`, `pose`, `receptor`).

---

## Analysis Bundle Expansion (`bundle-plan`)

Expand an advertised analysis bundle into an individual analysis list.

```bash
warp-md bundle-plan standard_md_report
warp-md bundle-plan polymer_report
warp-md bundle-plan standard_md_report --format json
```

### Bundled Reports

| Bundle Name | Analyses Included |
|------------|-------------------|
| `standard_md_report` | rg, rmsd, rdf, msd, density |
| `protein_md_report` | rg, rmsd, dssp, hbond, native_contacts |
| `solvent_ion_report` | rdf, msd, conductivity, dielectric, water_count, watershell |
| `polymer_report` | rg, chain_rg, end_to_end, contour_length, persistence_length, bondi_ffv |

Bundles are defined in `contract.py` (`ANALYSIS_BUNDLES`) and `cli_analysis_registry.py`. The `bundle-plan` command expands them into individual analysis config entries, making it easy to compose multi-analysis workflows without manually listing every command.
