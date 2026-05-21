---
description: Quick start for AI agents using warp-md tools
icon: robot
---

# For AI Agents

warp-md is **built agent-first** — every tool speaks JSON, emits structured progress, and returns deterministic results.

{% hint style="success" %}
**Get started in 3 lines:**

```bash
pip install warp-md
warp-md rg --topology protein.pdb --traj traj.xtc --selection "protein"
# Returns JSON envelope with result file path
```
{% endhint %}

---

## What Agents Can Do

| Capability | Tool | Example |
|------------|------|---------|
| **Trajectory Analysis** | `warp-md` | Calculate Rg, RMSD, RDF, conductivity (96+ analyses) |
| **Polymer Construction** | `warp-build` | Build chains, sequence polymers, and emit manifests for world-build |
| **Molecular Packing** | `warp-pack` | Solvate proteins, build simulation boxes |
| **Coarse Graining** | `warp-cg` | Map SMILES and trajectories to Martini beads |
| **Peptide Building** | `warp-pep` | Construct peptides, introduce mutations |

---

## Quick Reference

### CLI Usage (Recommended for Agents)

```bash
# Analysis
warp-md run config.json --stream ndjson

# Polymer build
warp-build run build_request.json --stream

# Packing
warp-pack --config pack.yaml --stream --output packed.pdb

# Martini coarse graining
warp-cg run cg_request.json --stream ndjson

# Peptide building
warp-pep build -s ACDEFG --preset alpha-helix --output helix.pdb --stream
```

Minimal `warp-cg` request with a solvated external trajectory source and native
bonded-parameter tuning:

```json
{
  "schema_version": "warp-cg.agent.v1",
  "name": "benzene",
  "smiles": "c1ccccc1",
  "trajectory_source": {
    "path": "traj.xtc",
    "topology": "topology.pdb",
    "kind": "external",
    "target_selection": "resname BENZ",
    "stride": 1,
    "mass_weighted": true
  },
  "output": {
    "out_dir": "results/cg/benzene",
    "mapped_trajectory": "benzene_cg.xtc",
    "write_mapping_json": true,
    "write_cg_pdb": true,
    "write_topology_itp": true,
    "write_topology_top": true,
    "write_bonded_parameter_map": true
  },
  "parameter_tuning": {
    "enabled": true,
    "source": "external_trajectory",
    "method": "bayesian_optimization",
    "objective": "bonded_parameter_parity"
  }
}
```

`parameter_tuning` supports native `bayesian_optimization` and `pso` methods for
bonded parameter fitting against mapped reference bond, angle, and dihedral
statistics. Set `reference_source.kind` to `xtb` when the agent should initiate
an xTB reference optimization/MD run instead of using an external trajectory.
If no bonded reference statistics are available, tuning returns a structured
`skipped` report rather than fabricating parameters.
When provided, `parameter_tuning.max_evaluations` and `swarm_size` must be
positive integers.

For solvated systems, keep the full topology in `trajectory_source.topology` and
identify the molecule to coarse-grain with either `target_selection` or
`atom_indices`, not both. `mass_weighted: true` requires a topology and uses
atomic masses from the selected target atoms.
When `trajectory_source.kind` is present it must be `external`.
`environment_selection` is accepted as forward-compatible metadata for solvent
or environment context, but mapping currently uses `target_selection` or
`atom_indices`.
Path and selection fields must be non-empty strings.

Supported native trajectory inputs include `xtc`, `dcd`, `trr`, `gro`, `g96`,
`h5md`, `tng`, `cpt`, `pdb`, and `pdbqt`. Supported mapped trajectory outputs
include `xtc`, `dcd`, `trr`, `gro`, `g96`, `h5md`, and single-frame `cpt`.

Agent-visible `warp-cg` artifacts:

If `output` is omitted, warp-cg writes mapping JSON, CG PDB, Martini ITP,
Martini TOP, and the bonded parameter map by default. If `output` is partially
specified, set `write_topology_top` explicitly when a top-level Gromacs topology
wrapper is required.

| Kind | Meaning |
|------|---------|
| `martini_mapping_json` | Deterministic bead-to-atom mapping and bead graph |
| `coarse_grained_pdb` | Bead-level PDB for quick downstream setup |
| `coarse_grained_trajectory` | Native mapped trajectory in the requested output format |
| `bond_stats_json` | Bond distribution statistics from mapped reference frames |
| `bonded_stats_json` | Bond, angle, and dihedral distribution statistics |
| `bonded_parameter_map_json` | Crosswalk from stats/tuning parameter names to ITP rows |
| `bonded_parameter_tuning_report` | BO/PSO trace, bounds, and best bonded parameters |
| `martini_topology_itp` | Martini-style atoms, bonds, angles, and dihedrals |
| `martini_topology_top` | Top-level Gromacs topology wrapper including the generated ITP |
| `xtb_optimized_xyz` | xTB optimized atomistic reference structure |
| `xtb_reference_trajectory` | xTB reference trajectory when xTB MD succeeds |

xTB-initiated reference example:

```json
{
  "schema_version": "warp-cg.agent.v1",
  "name": "ethanol",
  "smiles": "CCO",
  "reference_source": {
    "kind": "xtb",
    "xtb": {
      "temperature_k": 298.15,
      "time_ps": 2.0,
      "timestep_fs": 1.0,
      "dump_fs": 100.0,
      "gfn": "gfnff",
      "seed": 42,
      "work_dir": "results/cg/ethanol/xtb"
    }
  },
  "output": {
    "out_dir": "results/cg/ethanol",
    "mapped_trajectory": "ethanol_cg.xtc",
    "write_mapping_json": true,
    "write_cg_pdb": true,
    "write_topology_itp": true,
    "write_topology_top": true,
    "write_bonded_parameter_map": true
  },
  "parameter_tuning": {
    "enabled": true,
    "source": "xtb",
    "method": "pso",
    "swarm_size": 12,
    "max_evaluations": 48,
    "objective": "bonded_parameter_parity"
  }
}
```

Runnable request templates live in `examples/warp_cg/`:

| File | Use |
|------|-----|
| `solvated_external_bo_request.json` | Solvated external trajectory with `target_selection`, mass-weighted mapping, BO tuning, and Martini ITP output |
| `xtb_pso_request.json` | xTB-initiated reference workflow with PSO tuning and mapped `gro` output |

### Python API

```python
from warp_md import System, Trajectory, RgPlan

system = System.from_file("protein.pdb")
traj = Trajectory.open_xtc("traj.xtc", system)
rg = RgPlan(system.select("protein")).run(traj)
print(f"Mean Rg: {rg.mean():.2f} Å")
```

### MCP Server (Claude Desktop)

```json
{
  "mcpServers": {
    "warp-md": {
      "command": "warp-md",
      "args": ["mcp"]
    }
  }
}
```

---

## Integration Guides

| Framework | Guide | Example Files |
|-----------|-------|---------------|
| **LangChain** | [Agent Framework Integrations](agent-frameworks.md#1-langchain) | `examples/agents/langchain/` |
| **CrewAI** | [Agent Framework Integrations](agent-frameworks.md#2-crewai) | `examples/agents/crewai/` |
| **AutoGen** | [Agent Framework Integrations](agent-frameworks.md#4-autogen) | `examples/agents/autogen/` |
| **OpenAI Agents** | [Agent Framework Integrations](agent-frameworks.md#3-openai-agents-sdk) | `examples/agents/openai/` |

---

## Streaming Progress

Enable real-time progress updates with `--stream`:

```bash
warp-pack --config pack.yaml --stream
```

**Output:**
```
📦 Packing 150 molecules...
  → Placing molecules...
    Placed 50/150 (33.3%)
  → Optimizing...
    Iter 100: f=2.1e-03 (10.0%)
  ✓ Complete: 4500 atoms in 52s
```

See [Streaming Progress API](streaming-progress.md) for complete event reference.

---

## Agent Contract

All tools follow the same contract:

1. **Structured input** - JSON schemas with validation
2. **Structured output** - JSON envelope with status
3. **Error codes** - Machine-parseable error types
4. **Streaming** - NDJSON events for long operations

### Success Response

```json
{
    "schema_version": "warp-md.agent.v1",
    "status": "ok",
    "results": [
        {"analysis": "rg", "out": "rg.npz", "status": "ok"}
    ]
}
```

### Error Response

```json
{
    "schema_version": "warp-md.agent.v1",
    "status": "error",
    "exit_code": 3,
    "error": {
        "code": "E_ANALYSIS_SPEC",
        "message": "rdf missing required fields: sel_a, sel_b"
    }
}
```

---

## Common Agent Patterns

### Pattern 1: Batch Analysis

```json
{
    "version": "warp-md.agent.v1",
    "system": "protein.pdb",
    "trajectory": "traj.xtc",
    "analyses": [
        {"name": "rg", "selection": "protein"},
        {"name": "rmsd", "selection": "backbone", "align": true},
        {"name": "rdf", "sel_a": "resname SOL and name OW", "sel_b": "resname SOL and name OW", "bins": 200, "r_max": 10.0}
    ]
}
```

### Pattern 2: Pipeline (Build → Pack → Analyze)

```bash
# 1. Build the polymer
warp-build run build_request.json --stream

# 2. Assemble the world
warp-pack run pack_request.json --stream ndjson

# 3. Analyze the trajectory
warp-md run analysis.json --stream ndjson
```

### Pattern 3: Iterative Design

```bash
# 1. Build initial peptide
warp-pep build -s ACDEFG --output peptide.pdb --stream

# 2. Mutate and test
warp-pep mutate -i peptide.pdb -m A5G -o mutant.pdb --stream

# 3. Compare stability
warp-md rg --topology peptide.pdb --traj traj.xtc --selection protein
warp-md rg --topology mutant.pdb --traj traj.xtc --selection protein
```

---

## See Also

* [Agent Schema & Contract](../reference/agent-schema.md) - Full API contract
* [Streaming Progress API](streaming-progress.md) - Event reference
* [Sample Agent Conversations](agent-transcripts.md) - Example transcripts
* [Agent Framework Integrations](agent-frameworks.md) - Framework guides
