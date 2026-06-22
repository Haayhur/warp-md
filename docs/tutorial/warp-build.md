---
description: Build polymers from monomers — homopolymers, copolymers, and handoff to packing
icon: dna
---

# Polymer Building (warp-build)

Build a polymer chain from monomer building blocks, validate geometry, and hand off to `warp-pack` for solvation. This tutorial covers the full agent workflow.

---

## What You'll Learn

- Generate a source bundle and example request
- Build a linear homopolymer
- Validate the build request
- Hand off the build manifest to warp-pack for solvation

---

## Prerequisites

```bash
warp-build capabilities          # Verify the build CLI is available
warp-build schema --kind request # Inspect the build contract schema
```

---

## Step 1: Generate a Source Bundle

A source bundle defines the monomer residue templates and junction rules:

```bash
warp-build example-bundle --out source.bundle.json
```

This writes `source.bundle.json` with example residue templates for polymer building.

---

## Step 2: Generate a Build Request

Create a build request from the bundle, specifying the polymer mode and alignment:

```bash
warp-build example --mode aligned --bundle-path source.bundle.json > request.json
```

This produces `request.json` — a full build contract ready for validation.

### What's Inside

```json
{
  "schema_version": "warp-build.agent.v1",
  "request_id": "polymer-build-50mer-001",
  "source_ref": {
    "bundle_id": "example_polymer_bundle_v1",
    "bundle_path": "source.bundle.json"
  },
  "target": {
    "mode": "linear_homopolymer",
    "repeat_unit": "A",
    "n_repeat": 50,
    "termini": {"head": "default", "tail": "default"}
  },
  "realization": {
    "conformation_mode": "aligned",
    "alignment_axis": "z"
  },
  "artifacts": {
    "coordinates": "outputs/polymer_50mer.pdb",
    "build_manifest": "outputs/polymer_50mer.build.json",
    "charge_manifest": "outputs/polymer_50mer.charge.json",
    "topology_graph": "outputs/polymer_50mer.topology.json"
  }
}
```

Key fields:
- `target.mode`: `linear_homopolymer`, `linear_copolymer`, or `branched`
- `target.n_repeat`: chain length (number of repeat units)
- `realization.conformation_mode`: `aligned` (zigzag along axis) or `extended` (fully stretched)
- `artifacts`: where the outputs land

---

## Step 3: Validate the Build Request

Deep validation checks geometry, QC readiness, and source artifact integrity:

```bash
warp-build validate request.json
```

Exit code 0 means valid. The response includes a `preflight_cache` with `cache_key` and `input_digest` so agents can correlate validate/run decisions.

```bash
warp-build validate request.json --deep   # Full geometry/QC preflight (default)
warp-build validate request.json --shallow # Quick schema-only check
```

---

## Step 4: Run the Build

```bash
warp-build run request.json --stream
```

With `--stream`, you get NDJSON progress events on stderr: bundle loading, geometry construction, topology emission, artifact writing.

### Outputs

| Artifact | What It Contains |
|----------|-----------------|
| `outputs/polymer_50mer.pdb` | Built polymer coordinates |
| `outputs/polymer_50mer.build.json` | Build manifest (residue graph, topology, termini) |
| `outputs/polymer_50mer.charge.json` | Per-atom charge manifest |

---

## Step 5: Hand Off to warp-pack

The build manifest feeds directly into warp-pack for solvation:

```json
{
  "schema_version": "warp-pack.agent.v1",
  "run_id": "warp-build-handoff-001",
  "polymer_build": {
    "build_manifest": "outputs/polymer_50mer.build.json",
    "topology_graph": "outputs/polymer_50mer.topology.json"
  },
  "environment": {
    "box": {"mode": "padding", "padding_angstrom": 12.0, "shape": "cubic"},
    "solvent": {"mode": "explicit", "model": "tip3p"},
    "ions": {
      "neutralize": {"enabled": true},
      "salt": {"name": "nacl", "molar": 0.15}
    },
    "morphology": {"mode": "single_chain_solution"}
  },
  "outputs": {
    "coordinates": "outputs/system.pdb",
    "format": "pdb-strict",
    "manifest": "outputs/system_manifest.json",
    "preserve_topology_graph": true,
    "write_conect": true
  }
}
```

```bash
warp-pack run pack_request.json --stream ndjson
```

---

## Python API

```python
from warp_md import build as wb

# Generate example bundle
bundle = wb.example_bundle()

# Create a build request
request = wb.example_request("aligned")

# Validate
result = wb.validate(request)

# Run
exit_code, envelope = wb.run(request, stream=False)

# Check status
print(f"Status: {envelope['status']}, exit code: {exit_code}")
```

---

## What's Next

- [warp-pack tutorial](warp-pack.md) — solvate your built polymer
- [warp-cg tutorial](warp-cg.md) — coarse-grain the polymer for Martini simulations
- [Agent workflow example](../examples/agent-workflow.md) — full build → pack → analyze pipeline
