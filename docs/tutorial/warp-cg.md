---
description: Map atomistic molecules to Martini coarse-grained beads — SMILES, trajectory, and polymer sources
icon: atom
---

# Coarse-Graining (warp-cg)

Map small molecules, polymers, or full trajectories to Martini-style coarse-grained beads. This tutorial covers three source modes: SMILES with xTB, external trajectories, and polymer build manifests.

---

## What You'll Learn

- Map a SMILES molecule to Martini beads with xTB reference
- Map from an external atomistic trajectory
- Map a polymer build manifest to CG beads
- Inspect and install the bundled Martini3 forcefield
- Understand the output artifacts

---

## Prerequisites

```bash
warp-cg capabilities          # Verify CLI is available (lists available engines)
warp-cg schema --kind request # Inspect the CG contract schema
```

---

## Step 1: SMILES → Martini (xTB Reference)

The simplest path: provide a SMILES string, `warp-cg` generates an xTB trajectory and maps to Martini beads.

### CLI

```bash
# Generate the built-in benzene request
warp-cg example > request.json

# Edit reference_source/optimization as needed, then validate and run
warp-cg validate request.json
warp-cg run request.json --stream ndjson
```

### Request JSON

```json
{
  "schema_version": "warp-cg.agent.v1",
  "name": "benzene",
  "smiles": "c1ccccc1",
  "reference_source": {
    "kind": "xtb",
    "xtb": {
      "temperature_k": 298.15,
      "time_ps": 2.0,
      "gfn": "gfnff",
      "seed": 42
    }
  },
  "output": {
    "out_dir": "results/cg/benzene",
    "write_mapping_json": true,
    "write_cg_pdb": true,
    "write_topology_itp": true,
    "write_topology_top": true,
    "write_bonded_parameter_map": true
  },
  "optimization": {
    "enabled": true,
    "source": "xtb",
    "method": "pso",
    "swarm_size": 12,
    "max_evaluations": 48,
    "objective": "bonded_parameter_parity"
  }
}
```

### Output Artifacts

| Artifact | What It Contains |
|----------|-----------------|
| `benzene_cg.pdb` | Bead-level CG coordinates |
| `benzene_martini.itp` | Martini ITP with bonds, angles, dihedrals |
| `benzene_martini.top` | GROMACS topology wrapper |
| `benzene_bonded_parameter_map.json` | Crosswalk from tuned params to ITP rows |
| `benzene_bonded_stats.json` | Bond/angle/dihedral statistics |
| `benzene_tuning_report.json` | PSO trace and best parameters |

---

## Step 2: External Trajectory → Martini

If you already have an atomistic trajectory (e.g., from a solvated MD simulation), `warp-cg` can map it directly:

```json
{
  "schema_version": "warp-cg.agent.v1",
  "name": "benzene_solvated",
  "smiles": "c1ccccc1",
  "trajectory_source": {
    "kind": "external",
    "path": "results/benzene/solvated.xtc",
    "topology": "results/benzene/solvated.pdb",
    "target_selection": "resname BENZ",
    "mass_weighted": true
  },
  "output": {
    "out_dir": "results/cg/benzene_solvated",
    "mapped_trajectory": "benzene_cg.xtc"
  },
  "optimization": {
    "enabled": true,
    "source": "external_trajectory",
    "method": "bayesian_optimization",
    "max_evaluations": 64,
    "objective": "bonded_parameter_parity"
  }
}
```

```bash
warp-cg run request.json --stream ndjson
```

This produces a CG trajectory (`benzene_cg.xtc`) alongside the topology artifacts.

---

## Step 3: Polymer Build Manifest → Martini

When `warp-build` emits a build manifest, `warp-cg` can map the polymer to Martini beads using residue-graph strategies:

```json
{
  "schema_version": "warp-cg.agent.v1",
  "name": "paa_50mer",
  "source": {
    "kind": "polymer_pack_manifest",
    "path": "polymer_pack_manifest.json"
  },
  "mapping": {
    "mode": "auto",
    "strategy": "polymer_residue_graph",
    "target_bead_size": 4,
    "repeat_unit_hint": "PAA"
  },
  "output": {
    "out_dir": "cg/paa_50mer"
  }
}
```

```bash
warp-cg run request.json --stream ndjson
```

---

## Step 4: Forcefield Management

`warp-cg` ships a pinned Martini3 forcefield snapshot for deterministic agent runs:

```bash
# Inspect bundled forcefield (SHA-256 manifest)
warp-cg forcefield inspect --kind martini3

# Install to project-local directory
warp-cg forcefield install --kind martini3 --dest forcefields/martini3
```

Runtime never fetches forcefield files from the network.

---

## Step 5: Run Martini OpenMM Refinement

For managed refinement of CG systems with OpenMM:

```json
{
  "schema_version": "warp-cg.agent.v1",
  "name": "benzene",
  "smiles": "c1ccccc1",
  "reference_source": {"kind": "xtb", ...},
  "output": {...},
  "optimization": {
    "enabled": true,
    "source": "xtb",
    "method": "pso",
    "runner": {
      "kind": "martini_openmm",
      "candidate_dir": "candidates/benzene",
      "template": {
        "cg_pdb": "benzene_cg.pdb",
        "cg_top": "benzene_martini.top",
        "forcefield_dir": "forcefields/martini3"
      }
    }
  }
}
```

Install runner dependencies:

```bash
pip install "warp-md[martini]"
```

---

## Python API

```python
from warp_md.cg_contract import (
    example_request,
    run_cg_request,
    validate_request_payload,
)

# Generate an example request
request = example_request()
request["smiles"] = "c1ccccc1"

# Validate
validation = validate_request_payload(request)

# Run
exit_code, envelope = run_cg_request(request, stream="none")

# Check outputs
print(f"Valid: {validation['valid']}")
print(f"Status: {envelope['status']}")
```

---

## What's Next

- [warp-cg build tutorial](warp-cg-build.md) — assemble CG systems with membranes and solvent
- [warp-build tutorial](warp-build.md) — build polymers for CG mapping
- [Coarse-graining example](../examples/coarse-graining.md) — copy-paste ready
