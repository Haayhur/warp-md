---
description: The definitive agent contract — schemas, envelopes, error codes, determinism, cross-tool, and MCP
icon: file-contract
---

# Agent Contract

Every warp-md tool speaks one contract. This is the **single source of truth** for agent integrations across all tools.

---

## Contract Layers

| Layer | Schema | Transport | Consumer |
|-------|--------|-----------|----------|
| Request | `RunRequest` (JSON) | stdin / file | CLI |
| Streaming | `StreamEvent` (NDJSON) | stderr | Agent (real-time) |
| Result | `RunEnvelope` (JSON) | stdout | Agent (parsed) |
| Schema Discovery | `ContractCatalog` (JSON) | stdout | Agent (runtime) |
| MCP | Tool definitions (JSON) | stdio | LLM |

---

## 1. Request Contract (`RunRequest`)

### Schema Version

All requests must set `"version": "warp-md.agent.v1"`.

### Minimal Request

```json
{
  "version": "warp-md.agent.v1",
  "system": "protein.pdb",
  "trajectory": "traj.xtc",
  "analyses": [
    {"name": "rg", "selection": "protein"}
  ]
}
```

### Full Request

```json
{
  "version": "warp-md.agent.v1",
  "run_id": "agent-run-42",
  "system": {"path": "system.gro"},
  "trajectory": {"path": "traj.xtc"},
  "inputs": {
    "ref_traj": {"path": "ref.xtc"}
  },
  "device": "auto",
  "stream": "ndjson",
  "chunk_frames": 512,
  "output_dir": "results",
  "fail_fast": true,
  "checkpoint": {
    "enabled": true,
    "interval_frames": 1000
  },
  "analyses": [
    {"name": "rg", "selection": "protein", "out": "results/rg.npz"},
    {"name": "rmsd", "selection": "backbone", "reference": 0, "align": true}
  ]
}
```

### Request Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `version` | `string` | required | Must be `"warp-md.agent.v1"` |
| `run_id` | `string?` | `null` | Agent-assigned correlation ID |
| `system` | `IoSpec` | — | Topology file (PDB/GRO/PDBQT). Alias: `topology` |
| `trajectory` | `IoSpec` | — | Trajectory file (DCD/XTC/TRR/PDBQT). Alias: `traj` |
| `inputs` | `map<string,IoSpec>?` | `null` | Additional input files for multi-source analyses |
| `device` | `string` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"`, `"cuda:N"` |
| `stream` | `StreamMode` | `"none"` | `"none"` or `"ndjson"` |
| `chunk_frames` | `int?` | `null` | Override chunk size (default: engine-determined) |
| `output_dir` | `string` | `"."` | Base directory for outputs |
| `fail_fast` | `bool` | `true` | Stop on first analysis failure |
| `checkpoint` | `CheckpointConfig?` | `null` | Optional checkpoint stream config |
| `analyses` | `list<AnalysisRequest>` | required | At least 1 analysis |

### IoSpec Format

```jsonc
// Bare string (auto-wrapped to {path: ...})
"protein.pdb"

// Explicit object
{"path": "traj.xtc"}

// With format hint
{"path": "poses.pdbqt", "format": "pdbqt"}
```

### StreamMode Enum

```text
"none"    # No streaming (default)
"ndjson"  # NDJSON events on stderr
```

### CheckpointConfig

```json
{
  "enabled": true,
  "interval_frames": 1000
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `false` | Enable checkpoint streaming |
| `interval_frames` | `int` | `1000` | Frames between checkpoint events |

### AnalysisRequest

```json
{
  "name": "rg",
  "selection": "protein",
  "out": "results/rg.npz"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `string` | Analysis name from the plan catalog |
| `out` | `string?` | Output file path (format inferred from extension) |
| `...` | varies | Analysis-specific parameters (see plan catalog) |

---

## 2. Result Envelope (`RunEnvelope`)

### Success

```json
{
  "schema_version": "warp-md.agent.v1",
  "status": "ok",
  "exit_code": 0,
  "run_id": "agent-run-42",
  "output_dir": "results",
  "system": {"path": "system.gro", "n_atoms": 12345},
  "trajectory": {"path": "traj.xtc", "n_frames": 5000},
  "analysis_count": 3,
  "started_at": "2026-02-06T10:00:00Z",
  "finished_at": "2026-02-06T10:00:12Z",
  "elapsed_ms": 12340,
  "warnings": [],
  "results": [
    {
      "analysis": "rg",
      "status": "ok",
      "out": "results/rg.npz",
      "artifact": {
        "path": "results/rg.npz",
        "format": "npz",
        "bytes": 40960,
        "sha256": "a1b2c3d4...",
        "plot_recommendations": [...],
        "companions": [...]
      }
    }
  ]
}
```

### Error

```json
{
  "schema_version": "warp-md.agent.v1",
  "status": "error",
  "exit_code": 3,
  "run_id": "agent-run-42",
  "analysis_count": 3,
  "started_at": "2026-02-06T10:00:00Z",
  "finished_at": "2026-02-06T10:00:01Z",
  "elapsed_ms": 850,
  "results": [],
  "error": {
    "code": "E_ANALYSIS_SPEC",
    "message": "rdf missing required fields: sel_a, sel_b",
    "context": {"analysis": "rdf", "index": 2}
  }
}
```

### Single-Analysis Envelope

`warp-md rg --topology top.pdb --traj traj.xtc --selection protein` emits the same envelope shape but with `analysis_count: 1` and no `run_id`.

### Artifact Metadata

Every result includes:

```json
{
  "path": "results/rg.npz",
  "format": "npz",
  "bytes": 40960,
  "sha256": "a1b2c3d4...",
  "plot_recommendations": [
    {
      "artifact": "results/rg.npz",
      "plot_type": "line",
      "x": {"field": "time_ps", "units": "ps"},
      "y": {"field": "rg_nm", "units": "nm"},
      "title": "Time series of radius of gyration values"
    }
  ],
  "companions": [
    {"path": "results/rg/manifest.json", "format": "json", "role": "npz_companion_manifest"},
    {"path": "results/rg/rg_nm.csv", "format": "csv", "role": "array_table", "source_key": "rg_nm"}
  ]
}
```

| Field | Description |
|-------|-------------|
| `path` | Filesystem path to output |
| `format` | `npz`, `npy`, `csv`, `json` |
| `bytes` | File size |
| `sha256` | Content hash for verification |
| `plot_recommendations` | Rust-native deterministic plot hints |
| `companions` | Non-Python companion artifacts (manifest + CSV for NPZ) |

### PlotRecommendation Schema

| Field | Type | Description |
|-------|------|-------------|
| `plot_type` | `string` | `"line"`, `"scatter"`, `"histogram"`, `"heatmap"`, `"bar"` |
| `x` | `PlotAxisSpec?` | X-axis field, units, source |
| `y` | `PlotAxisSpec?` | Y-axis field, units, source |
| `z` | `PlotAxisSpec?` | Z-axis field (for heatmaps) |
| `title` | `string` | Plot title |
| `artifact` | `string?` | Source artifact path |
| `shape` | `[u64]?` | Array shape hint |

---

## 3. Error Contract

### Exit Codes

| Code | Status | Meaning | Agent Action |
|------|--------|---------|--------------|
| `0` | `"ok"` | Success | Parse `results` array |
| `2` | `"error"` | Config validation error | Fix request JSON |
| `3` | `"error"` | Analysis specification error | Check required fields |
| `4` | `"error"` | Runtime error (IO/compute) | Verify file paths, system compatibility |
| `5` | `"error"` | Internal error | Report bug |

### Error Codes (machine-parseable string codes)

| Code | Meaning | Recovery |
|------|---------|----------|
| `E_CONFIG_LOAD` | Cannot read config file | Check file path |
| `E_CONFIG_VALIDATION` | JSON schema validation failed | Fix request structure |
| `E_CONFIG_VERSION` | Unknown schema version | Set `version` to `"warp-md.agent.v1"` |
| `E_CONFIG_MISSING_FIELD` | Required field not provided | Add missing field |
| `E_ANALYSIS_UNKNOWN` | Unknown analysis name | Use `list-plans` to discover available analyses |
| `E_ANALYSIS_SPEC` | Missing required analysis fields | Check analysis-specific required fields |
| `E_SELECTION_EMPTY` | Selection matched 0 atoms | Check selection syntax |
| `E_SELECTION_INVALID` | Selection syntax error | Fix selection expression |
| `E_SYSTEM_LOAD` | Cannot read topology file | Verify file exists and format is supported |
| `E_TRAJECTORY_LOAD` | Cannot read trajectory file | Verify file path and format |
| `E_TRAJECTORY_EOF` | Trajectory ended unexpectedly | Check file integrity |
| `E_RUNTIME_EXEC` | Computation error during analysis | Check device compatibility, memory |
| `E_OUTPUT_DIR` | Cannot create output directory | Check filesystem permissions |
| `E_OUTPUT_WRITE` | Cannot write output file | Check disk space, permissions |
| `E_DEVICE_UNAVAILABLE` | Requested device not available | Fall back to `device="cpu"` |
| `E_INPUT_MISSING` | Required input file not found | Check file path |
| `E_UNSUPPORTED_FORMAT` | File format not supported | Convert to supported format (PDB/GRO/DCD/XTC) |
| `E_TOPOLOGY_TRAJECTORY_MISMATCH` | Atom counts differ between topology and trajectory | Use matching files |
| `E_TOPOLOGY_ATOM_MISSING` | Atom not found in topology | Check selection/topology consistency |
| `E_NO_FRAMES` | Trajectory contains 0 frames | Check trajectory file |
| `E_EXTERNAL_TABLE_LOAD` | Cannot read charges/group types table | Check CSV/TSV format |
| `E_EXTERNAL_TABLE_COLUMN` | Missing required column in table | Add `resname`, `name`/`atom`, `charge` columns |
| `E_PLOT_RENDER` | Plot rendering failed | Check data shape compatibility |
| `E_BUNDLE_PARTIAL` | Some analyses in bundle failed | Check individual results |
| `E_INTERNAL` | Unexpected internal error | Report bug with reproduction steps |

---

## 4. Streaming Events (NDJSON)

Enable with `"stream": "ndjson"`. Events are written to **stderr** as newline-delimited JSON. stdout remains clean for the final envelope.

### Event Catalog

| Event | Schema | Trigger |
|-------|--------|---------|
| `run_started` | `RunStartedEvent` | Batch accepted for execution |
| `analysis_started` | `AnalysisStartedEvent` | Single analysis begins |
| `checkpoint` | `CheckpointEvent` | Mid-analysis progress tick (optional) |
| `analysis_completed` | `AnalysisCompletedEvent` | Analysis succeeded |
| `analysis_failed` | `AnalysisFailedEvent` | Analysis failed (batch continues if `fail_fast=false`) |
| `run_completed` | `RunCompletedEvent` | Batch finished (carries `final_envelope`) |
| `run_failed` | `RunFailedEvent` | Fatal batch failure (carries `final_envelope`) |

### Event Schemas

**RunStartedEvent:**
```json
{
  "event": "run_started",
  "analysis_count": 3,
  "completed": 0,
  "total": 3,
  "progress_pct": 0.0
}
```

**AnalysisStartedEvent:**
```json
{
  "event": "analysis_started",
  "index": 0,
  "analysis": "rg",
  "out": "results/rg.npz",
  "completed": 0,
  "total": 3,
  "progress_pct": 0.0
}
```

**CheckpointEvent:**
```json
{
  "event": "checkpoint",
  "analysis_index": 0,
  "analysis_name": "rg",
  "frames_processed": 500,
  "frames_total": 5000,
  "completed": 0,
  "total": 3,
  "progress_pct": 10.0
}
```

**AnalysisCompletedEvent:**
```json
{
  "event": "analysis_completed",
  "index": 0,
  "analysis": "rg",
  "status": "ok",
  "out": "results/rg.npz",
  "timing_ms": 1200,
  "completed": 1,
  "total": 3,
  "progress_pct": 33.3,
  "eta_ms": 2400
}
```

**AnalysisFailedEvent:**
```json
{
  "event": "analysis_failed",
  "index": 1,
  "analysis": "rmsd",
  "error": {
    "code": "E_ANALYSIS_SPEC",
    "message": "rmsd missing required fields: selection"
  },
  "completed": 0,
  "total": 3,
  "progress_pct": 0.0
}
```

**RunCompletedEvent:**
```json
{
  "event": "run_completed",
  "final_envelope": { ... full success envelope ... }
}
```

**RunFailedEvent:**
```json
{
  "event": "run_failed",
  "final_envelope": { ... full error envelope ... }
}
```

### Common Progress Fields

| Field | Type | Description |
|-------|------|-------------|
| `completed` | `int` | Analyses finished (including failures) |
| `total` | `int` | Total analyses in batch |
| `progress_pct` | `float` | 0–100 completion percentage |
| `eta_ms` | `int?` | Estimated milliseconds remaining |

---

## 5. Schema Discovery

Agents discover available analyses and their parameters at runtime:

```bash
# Full catalog with all analyses, required/optional fields, types, defaults
warp-md list-plans --json --details

# Request schema
warp-md schema --kind request --format json

# Result envelope schema
warp-md schema --kind result --format json

# Event schema
warp-md schema --kind event --format json

# Water models
warp-md water-models --format json
```

```python
from warp_md.agent_schema import (
    run_request_json_schema,
    run_result_json_schema,
    run_event_json_schema,
)

request_schema = run_request_json_schema()
result_schema = run_result_json_schema()
event_schema = run_event_json_schema()
```

### Plan Catalog Structure

Each plan entry includes:

```json
{
  "name": "rg",
  "required_fields": ["selection"],
  "optional_fields": ["mass_weighted", "out"],
  "fields": {
    "selection": {
      "type": "string",
      "semantic_type": "atom_selection",
      "description": "Atoms to analyze"
    },
    "mass_weighted": {
      "type": "bool",
      "default": false,
      "description": "Weight by atomic mass"
    },
    "out": {
      "type": "string",
      "description": "Output file path"
    }
  },
  "output": {
    "kind": "array",
    "format": "npz",
    "shape": ["n_frames"],
    "unit": "angstrom",
    "description": "Radius of gyration per frame"
  }
}
```

### Analysis Bundles

Predefined analysis bundles for common workflows:

| Bundle Name | Analyses | Use Case |
|-------------|----------|----------|
| `standard_md_report` | rg, rmsd, rdf, msd, hbond, end_to_end, contour_length | Standard MD trajectory report |
| `polymer_report` | end_to_end, contour_length, chain_rg, bond_length_distribution, bond_angle_distribution, persistence_length | Polymer characterization |

---

## 6. Determinism Contract

Same inputs → same outputs **when all of these hold:**

1. Same topology file (byte-identical)
2. Same trajectory file (byte-identical)
3. Same request JSON (byte-identical, including `seed` for stochastic analyses)
4. Same warp-md build (same Rust binary, same plan implementations)
5. Same device (`"cpu"` is fully deterministic; `"cuda"` may vary between GPU architectures)

**What breaks determinism:**
- Different trajectory chunk boundaries (only if using `chunk_frames`)
- CUDA cross-architecture differences
- Floating-point reductions on different thread counts (CPU only)
- Stochastic analyses without explicit `seed`

**For deterministic results, agents should:**
- Always use `device="cpu"` for verification runs
- Set `seed` explicitly for stochastic analyses
- Use the same warp-md build version

---

## 7. Plot Contract

Plots are generated from result envelopes:

```bash
warp-md plot result_envelope.json --out-dir plots
```

### PlotManifest

```json
{
  "schema_version": "warp-md.plot.v1",
  "run_id": "agent-run-42",
  "status": "ok",
  "plots": [
    {
      "path": "plots/rg.svg",
      "format": "svg",
      "plot_type": "line",
      "title": "Rg time series",
      "artifact": "results/rg.npz",
      "series": [
        {"field": "rg_nm", "label": "Rg (nm)", "color": "#1f77b4"}
      ]
    }
  ],
  "warnings": []
}
```

Each `PlotArtifact` carries the SVG file path and a plot manifest JSON for programmatic consumption.

---

## 8. Python Validation API

```python
from warp_md.agent_schema import validate_run_request, RunRequest

# Validate a dict before sending to CLI
payload = {
    "version": "warp-md.agent.v1",
    "system": "protein.pdb",
    "trajectory": "traj.xtc",
    "analyses": [{"name": "rg", "selection": "protein"}]
}
request: RunRequest = validate_run_request(payload)

# Access typed properties
print(request.system_spec)      # {"path": "protein.pdb"}
print(request.trajectory_spec)  # {"path": "traj.xtc"}
print(request.analyses[0].name) # "rg"
```

---

## 9. Cross-Tool Contracts

### warp-pack

**Schema version:** `"warp-pack.agent.v1"`

**Request:** `PackConfig` — JSON or YAML config file with structures, box, constraints.

**Result envelope:**
```json
{
  "schema_version": "warp-pack.agent.v1",
  "status": "ok",
  "run_id": "pack-run-01",
  "output_dir": "outputs",
  "artifacts": {
    "coordinates": "outputs/system.pdb",
    "manifest": "outputs/system_manifest.json",
    "md_package": null
  },
  "summary": {
    "component_count": 2,
    "total_atoms": 3200,
    "water_count": 1000,
    "ion_counts": {"Na+": 10, "Cl-": 10}
  },
  "manifest_path": "outputs/system_manifest.json",
  "warnings": []
}
```

**Error envelope:**
```json
{
  "schema_version": "warp-pack.agent.v1",
  "status": "error",
  "run_id": "pack-run-01",
  "exit_code": 4,
  "error": {"code": "E_PLACEMENT_FAILED", "message": "cannot place structure: water.pdb", "severity": "error"},
  "errors": [],
  "warnings": []
}
```

**Streaming events:** `pack_started`, `phase_started`, `molecule_placed`, `gencan_iteration`, `phase_complete`, `pack_complete`, `error`

**Error codes:** `E_CONFIG_VALIDATION`, `E_PLACEMENT_FAILED`, `E_FILE_NOT_FOUND`, `E_INTERNAL`

---

### warp-build

**Schema version:** `"warp-build.agent.v1"`

**Request:** `BuildRequest` — source bundle ref, target mode, realization, artifacts.

**Result envelope:** The success envelope includes the artifact request, full
run summary, QC report, relaxation/solver reports, phase timings, warnings, and
provenance fields. Inspect the exact installed schema with:

```bash
warp-build schema --kind result
```

**Error envelope:**
```json
{
  "schema_version": "warp-build.agent.v1",
  "status": "error",
  "request_id": "polymer-build-001",
  "errors": [{"code": "E_SOURCE_BUNDLE", "message": "bundle path not found", "severity": "error"}],
  "warnings": []
}
```

**Error codes:** `E_SOURCE_BUNDLE`, `E_VALIDATION`, `E_BUILD`, `E_MANIFEST`, `E_TOPOLOGY`, `E_CHARGE`, `E_REALIZATION`

---

### warp-cg

**Schema version:** `"warp-cg.agent.v1"`

**Request:** `CgRequest` — SMILES/repeat_smiles/source, mapping, trajectory_source, optimization, output.

**Result envelope:** Emits artifact paths for mapping JSON, CG PDB, CG trajectory, ITP, TOP, bonded stats, optimization report.

**Error codes:** `E_MAPPING`, `E_XTB`, `E_OPTIMIZATION`, `E_OUTPUT`, `E_VALIDATION`

---

### warp-pep

**Schema version:** CLI flags (no JSON request envelope).

**Result envelope:** Emitted on stderr in streaming mode:
```json
{"event":"operation_complete","total_atoms":167,"elapsed_ms":45}
```

**Error codes:** `E_BUILD`, `E_MUTATION`, `E_SEQUENCE`, `E_OUTPUT`

---

## 10. MCP Contract

### Tool Definitions

| Tool | Description | Arguments |
|------|-------------|-----------|
| `run_analysis` | Run MD trajectory analyses | `system_path`, `trajectory_path`, `analyses`, `output_dir`, `device`, `fail_fast` |
| `list_analyses` | List all analysis types | (none) |
| `get_analysis_schema` | Get schema for one analysis | `name` |
| `validate_config` | Dry-run validate a config | `config` |
| `pack_molecules` | Pack molecules into boxes | `config_path`, `output`, `format`, `stream` |
| `build_peptide` | Build peptide from sequence | `sequence`, `three_letter`, `preset`, `oxt`, `detect_ss` |
| `mutate_peptide` | Mutate residues in-place | `input`, `mutations`, `output` |

### Agent Discovery Loop

```
Agent → list_analyses → get_analysis_schema(name) → validate_config(config) → run_analysis(...)
```

---

## 11. Agent Tool Definition Template

```
TOOL warp-md
DESCRIPTION: Run MD trajectory analysis. Validates input via Pydantic schema.
             Outputs file artifacts + JSON envelope contract.
COMMAND: warp-md run <config.json> [--stream ndjson] [--debug-errors]
SCHEMA_DISCOVERY: warp-md schema --kind request --format json
RESULT_SCHEMA: warp-md schema --kind result --format json
EVENT_SCHEMA: warp-md schema --kind event --format json
PLAN_DISCOVERY: warp-md list-plans --json --details
ANALYSIS_BUNDLES: standard_md_report, polymer_report
EXIT_CODES: 0=ok, 2=config_error, 3=spec_error, 4=runtime_error, 5=internal
ERROR_CODES: E_CONFIG_LOAD, E_CONFIG_VALIDATION, E_CONFIG_VERSION, E_ANALYSIS_UNKNOWN, E_ANALYSIS_SPEC, E_SELECTION_EMPTY, E_SELECTION_INVALID, E_SYSTEM_LOAD, E_TRAJECTORY_LOAD, E_RUNTIME_EXEC, E_OUTPUT_WRITE, E_DEVICE_UNAVAILABLE, E_TOPOLOGY_TRAJECTORY_MISMATCH, E_NO_FRAMES, E_EXTERNAL_TABLE_LOAD, E_INTERNAL
STREAMING: --stream ndjson for real-time progress events
OUTPUT_FORMATS: npz, npy, csv, json
IDEMPOTENT: Yes (same config → same results on CPU with same seed)
```

---

## See Also

- [Agent Schema & Contract](../reference/agent-schema.md) — full reference with all request fields and result envelope shapes
- [Streaming Progress API](../guides/streaming-progress.md) — event types for all tools
- [Agent Framework Integrations](../guides/agent-frameworks.md) — LangChain, CrewAI, OpenAI, AutoGen
- [CLI Reference](../reference/cli.md) — all CLI commands and options
