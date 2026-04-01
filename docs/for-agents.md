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
warp-pack run pack_request.json --stream ndjson

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
warp-pack run pack_request.json --stream ndjson
```

**Output (NDJSON on stderr):**
```
{"event":"run_started","schema_version":"warp-pack.agent.v1","run_id":"pack-001","elapsed_ms":0}
{"event":"phase_progress","schema_version":"warp-pack.agent.v1","run_id":"pack-001","phase":"solvation","progress_pct":33.3,"elapsed_ms":1200}
{"event":"run_completed","schema_version":"warp-pack.agent.v1","run_id":"pack-001","elapsed_ms":52000,"final_envelope":{"schema_version":"warp-pack.agent.v1","status":"ok"}}
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
