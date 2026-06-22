---
description: Build a 50-mer polymer, emit handoff manifest, pack into solvent
icon: dna
---

# Polymer Build

Build a 50-mer homopolymer, validate, and hand off to warp-pack for solvation.

---

## Build a Linear Homopolymer

### CLI

```bash
# Generate example bundle
warp-build example-bundle --out source.bundle.json

# Generate request from bundle
warp-build example --mode aligned --bundle-path source.bundle.json > request.json

# Validate
warp-build validate request.json

# Run
warp-build run request.json --stream
```

### Python

```python
from warp_md import build as wb

request = wb.example_request("aligned")
result = wb.validate(request)
exit_code, envelope = wb.run(request, stream=False)
print(envelope["status"])
```

### Request JSON

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

---

## Hand Off to warp-pack

Once the chain is built, hand the manifest into warp-pack for solvation:

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

## Full Pipeline

```bash
# 1. Build polymer
warp-build run build_request.json --stream

# 2. Pack into solvent
warp-pack run pack_request.json --stream ndjson

# 3. Analyze trajectory
warp-md run analysis.json --stream ndjson
```
