---
description: Build → Pack → Analyze — a complete agent workflow
icon: workflow
---

# Agent Workflow

A complete pipeline: build a polymer, solvate it, run trajectory analysis, and interpret results.

---

## Pipeline: Build → Pack → Analyze

```bash
# 1. Build the polymer
warp-build run build_request.json --stream

# 2. Assemble the world
warp-pack run pack_request.json --stream ndjson

# 3. Analyze the trajectory
warp-md run analysis.json --stream ndjson
```

---

## Step 1: Build

```json
{
  "schema_version": "warp-build.agent.v1",
  "request_id": "polymer-build-001",
  "source_ref": {
    "bundle_id": "example_polymer_bundle_v1",
    "bundle_path": "source.bundle.json"
  },
  "target": {
    "mode": "linear_homopolymer",
    "repeat_unit": "A",
    "n_repeat": 50
  },
  "realization": {
    "conformation_mode": "aligned",
    "alignment_axis": "z"
  },
  "artifacts": {
    "coordinates": "outputs/polymer.pdb",
    "build_manifest": "outputs/polymer.build.json",
    "charge_manifest": "outputs/polymer.charge.json",
    "topology_graph": "outputs/polymer.topology.json"
  }
}
```

---

## Step 2: Pack

```json
{
  "schema_version": "warp-pack.agent.v1",
  "run_id": "warp-build-handoff-001",
  "polymer_build": {
    "build_manifest": "outputs/polymer.build.json",
    "topology_graph": "outputs/polymer.topology.json"
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

---

## Step 3: Analyze

```json
{
  "version": "warp-md.agent.v1",
  "system": "outputs/system.pdb",
  "trajectory": "outputs/traj.xtc",
  "analyses": [
    {"name": "rg", "selection": "protein"},
    {"name": "rmsd", "selection": "backbone", "align": true},
    {"name": "rdf", "sel_a": "resname SOL and name OW", "sel_b": "resname SOL and name OW", "bins": 200, "r_max": 10.0}
  ]
}
```

---

## Agent Script (Python)

```python
import json
import subprocess
from pathlib import Path

def run_tool(command, request_path):
    """Run a warp-md ecosystem tool and return the result envelope."""
    result = subprocess.run(
        [*command, str(request_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)

requests = {
    "build": Path("build_request.json"),
    "pack": Path("pack_request.json"),
    "analysis": Path("analysis.json"),
}

build_envelope = run_tool(["warp-build", "run"], requests["build"])
pack_envelope = run_tool(["warp-pack", "run"], requests["pack"])
analysis_envelope = run_tool(["warp-md", "run"], requests["analysis"])

print(f"Build: {build_envelope['status']}")
print(f"Pack: {pack_envelope['status']}")
print(f"Analysis: {analysis_envelope['status']}")
```
