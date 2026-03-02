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
| **Molecular Packing** | `warp-pack` | Solvate proteins, build simulation boxes |
| **Peptide Building** | `warp-pep` | Construct peptides, introduce mutations |

---

## Quick Reference

### CLI Usage (Recommended for Agents)

```bash
# Analysis
warp-md run config.json --stream ndjson

# Packing
warp-pack --config pack.yaml --stream --output packed.pdb

# Peptide building
warp-pep build -s ACDEFG --preset alpha-helix --output helix.pdb --stream
```

### Python API

```python
from warp_md import System, Trajectory, RgPlan

system = System.from_pdb("protein.pdb")
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

### Pattern 2: Pipeline (Pack → Analyze)

```bash
# 1. Pack the system
warp-pack --config pack.yaml --output packed.pdb --stream

# 2. Analyze the trajectory
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
