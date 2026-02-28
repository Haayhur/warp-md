---
description: The contract your agent signs — Pydantic schemas, JSON envelopes, streaming events
icon: file-contract
---

# Agent Schema & Contract

This is the page your agent should read first. Every request validated. Every response predictable. Zero surprises.

{% hint style="info" %}
**Agent-first means contract-first.** warp-md uses Pydantic models for request validation, structured JSON envelopes for results, and NDJSON streaming for real-time progress. Your agent never has to guess what happened.
{% endhint %}

---

## Schema Discovery

Your agent can inspect the full contract at runtime:

```bash
# Request schema — what your agent sends
warp-md schema --kind request --format json

# Result schema — what comes back
warp-md schema --kind result --format json

# Streaming event schema — real-time progress
warp-md schema --kind event --format json
```

```python
from warp_md.agent_schema import (
    run_request_json_schema,
    run_result_json_schema,
    run_event_json_schema,
)

# Get full JSON Schema for programmatic consumption
request_schema = run_request_json_schema()
result_schema = run_result_json_schema()
event_schema = run_event_json_schema()
```

---

## The Request Contract

Every batch run starts with a `RunRequest` — a Pydantic-validated JSON document.

### Minimal Example

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

### Full Example

```json
{
  "version": "warp-md.agent.v1",
  "run_id": "agent-run-42",
  "system": {"path": "system.gro"},
  "trajectory": {"path": "traj.xtc"},
  "device": "auto",
  "stream": "ndjson",
  "chunk_frames": 512,
  "output_dir": "results",
  "analyses": [
    {"name": "rg", "selection": "protein", "out": "results/rg.npz"},
    {"name": "rmsd", "selection": "backbone", "reference": "topology", "align": true},
    {"name": "rdf", "sel_a": "resname SOL and name OW", "sel_b": "resname SOL and name OW", "bins": 200, "r_max": 10.0},
    {"name": "conductivity", "selection": "resname NA or resname CL", "charges": [1.0, -1.0], "temperature": 300.0}
  ]
}
```

### Docking Poses Example (PDBQT)

```json
{
  "version": "warp-md.agent.v1",
  "system": {"path": "complex.pdbqt"},
  "trajectory": {"path": "poses.pdbqt", "format": "pdbqt"},
  "analyses": [
    {
      "name": "docking",
      "receptor_mask": "protein and not resname LIG",
      "ligand_mask": "resname LIG"
    }
  ]
}
```

### Request Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `version` | `str` | required | Must be `"warp-md.agent.v1"` |
| `run_id` | `str?` | `null` | Agent-assigned correlation ID |
| `system` | `str` or `{path}` | required* | Topology file (PDB/GRO/PDBQT) |
| `topology` | `str` or `{path}` | required* | Alias for `system` |
| `trajectory` | `str` or `{path}` | required* | Trajectory file (DCD/XTC or PDB/PDBQT for docking poses) |
| `traj` | `str` or `{path}` | required* | Alias for `trajectory` |
| `device` | `str` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"`, `"cuda:N"` |
| `stream` | `str` | `"none"` | `"none"` or `"ndjson"` |
| `chunk_frames` | `int?` | `null` | Override chunk size |
| `output_dir` | `str` | `"."` | Base directory for outputs |
| `analyses` | `list` | required | At least one analysis |

\* Specify either `system` or `topology` (not both). Same for `trajectory`/`traj`.

{% hint style="warning" %}
**Convenience shortcuts**: Both `system` and `topology` accept a bare string (`"protein.pdb"`) which auto-wraps to `{"path": "protein.pdb"}`. Same for trajectory fields.
{% endhint %}

---

## Available Analyses (Config Runner)

The `name` field in each analysis maps to these 19 CLI-registered analyses:

| Name | Required Fields |
|------|----------------|
| `rg` | `selection` |
| `rmsd` | `selection` |
| `msd` | `selection` |
| `rotacf` | `selection`, `orientation` |
| `conductivity` | `selection`, `charges`, `temperature` |
| `dielectric` | `selection`, `charges` |
| `dipole_alignment` | `selection`, `charges` |
| `ion_pair_correlation` | `selection`, `rclust_cat`, `rclust_ani` |
| `structure_factor` | `selection`, `bins`, `r_max`, `q_bins`, `q_max` |
| `water_count` | `water_selection`, `center_selection`, `box_unit`, `region_size` |
| `equipartition` | `selection` |
| `hbond` | `donors`, `acceptors`, `dist_cutoff` |
| `rdf` | `sel_a`, `sel_b`, `bins`, `r_max` |
| `end_to_end` | `selection` |
| `contour_length` | `selection` |
| `chain_rg` | `selection` |
| `bond_length_distribution` | `selection`, `bins`, `r_max` |
| `bond_angle_distribution` | `selection`, `bins` |
| `persistence_length` | `selection` |
| `docking` | `receptor_mask`, `ligand_mask` |

{% hint style="info" %}
**Beyond the CLI**: The Python API exposes **96 Plan classes** total. The config runner covers the 19 most common analyses. For the rest, use the Python API directly.
{% endhint %}

---

## The Result Envelope

Every run produces a JSON envelope — success or failure. Your agent always knows what happened.

### Success Envelope

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
      "out": "results/rg.npz",
      "status": "ok",
      "artifact": {
        "path": "results/rg.npz",
        "format": "npz",
        "bytes": 40960,
        "sha256": "a1b2c3d4..."
      }
    }
  ]
}
```

### Error Envelope

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

### Exit Codes

| Code | Meaning | Agent Action |
|------|---------|--------------|
| `0` | Success | Parse `results` array |
| `2` | Config validation error | Fix the request JSON |
| `3` | Analysis specification error | Check required fields |
| `4` | Runtime error (IO/compute) | Check file paths, system compatibility |
| `5` | Internal error | Report bug |

### Artifact Metadata

Every successfully saved result includes:

| Field | Description |
|-------|-------------|
| `path` | Filesystem path to the output file |
| `format` | File format (`npz`, `npy`, `csv`, `json`) |
| `bytes` | File size |
| `sha256` | Content hash for verification |

---

## Streaming Events (NDJSON)

For long-running batches, enable real-time progress with `"stream": "ndjson"`:

```bash
warp-md run config.json --stream ndjson
```

Each line is a self-contained JSON event:

### Event Types

| Event | When | Key Fields |
|-------|------|------------|
| `run_started` | Batch begins | `analysis_count`, `progress_pct` |
| `analysis_started` | Analysis N begins | `index`, `analysis`, `out` |
| `analysis_completed` | Analysis N succeeds | `timing_ms`, `progress_pct`, `eta_ms` |
| `analysis_failed` | Analysis N fails | `error.code`, `error.message` |
| `run_completed` | All done (success) | `final_envelope` (full success envelope) |
| `run_failed` | Fatal failure | `final_envelope` (full error envelope) |

### Progress Fields (all events)

| Field | Description |
|-------|-------------|
| `completed` | Analyses finished so far |
| `total` | Total analyses in batch |
| `progress_pct` | Completion percentage (0–100) |
| `eta_ms` | Estimated time remaining (milliseconds) |

### Example Stream

```jsonl
{"event":"run_started","analysis_count":3,"completed":0,"total":3,"progress_pct":0.0}
{"event":"analysis_started","index":0,"analysis":"rg","out":"results/rg.npz","completed":0,"total":3,"progress_pct":0.0}
{"event":"analysis_completed","index":0,"analysis":"rg","status":"ok","out":"results/rg.npz","timing_ms":1200,"completed":1,"total":3,"progress_pct":33.3,"eta_ms":2400}
{"event":"analysis_started","index":1,"analysis":"rmsd","out":"results/rmsd.npz","completed":1,"total":3,"progress_pct":33.3}
{"event":"analysis_completed","index":1,"analysis":"rmsd","status":"ok","out":"results/rmsd.npz","timing_ms":800,"completed":2,"total":3,"progress_pct":66.7,"eta_ms":800}
{"event":"run_completed","final_envelope":{...}}
```

---

## Python Validation API

For agents that build requests programmatically:

```python
from warp_md.agent_schema import validate_run_request, RunRequest

# Validate a dict
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

{% hint style="success" %}
**Pro tip**: Use `validate_run_request()` before sending to the CLI. Catch schema errors in Python instead of parsing error envelopes.
{% endhint %}

---

## Agent Tool Definition

Copy this into your agent's tool registry:

```
TOOL warp-md
DESCRIPTION: Run MD trajectory analysis. Validates input via Pydantic schema.
             Outputs file artifacts + JSON envelope contract.
COMMAND: warp-md run <config.json> [--stream ndjson] [--debug-errors]
SCHEMA_DISCOVERY: warp-md schema --kind request --format json
RESULT_SCHEMA: warp-md schema --kind result --format json
EVENT_SCHEMA: warp-md schema --kind event --format json
PLAN_DISCOVERY: warp-md list-plans --json --details
EXIT_CODES: 0=ok, 2=config_error, 3=spec_error, 4=runtime_error, 5=internal
STREAMING: --stream ndjson for real-time progress events
IDEMPOTENT: Yes (same config → same results, given deterministic mode)
```
