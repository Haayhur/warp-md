# Agent Framework Integration Examples

Examples demonstrating how to integrate warp-md tools with popular AI agent frameworks.

## Prerequisites

1. **Install warp-md:**
   ```bash
   pip install warp-md
   # or for GPU support
   pip install warp-md-cuda
   ```

2. **Install framework dependencies:**
   ```bash
   cd examples/agents
   pip install -r requirements.txt
   ```

3. **Set API key for your LLM:**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   # or create a .env file
   ```

## Why warp-md is Agent-Friendly

* **Contract-first** - Pydantic-validated schemas, deterministic outputs
* **Machine-readable** - JSON envelopes with structured error codes
* **CLI-accessible** - Any agent can invoke via subprocess
* **Streaming support** - NDJSON events for real-time progress
* **Zero Python lock-in** - Use from Rust, Python, or any language

## Framework Examples

### LangChain
* **Files:** `langchain/warp_md_tool.py`, `langchain/example_agent.py`, `langchain/warp_pack_streaming.py`
* **Install:** `pip install langchain langchain-core langchain-openai`
* **Pattern:** StructuredTool with Pydantic validation

### CrewAI
* **Files:** `crewai/warp_md_tool.py`, `crewai/example_crew.py`, `crewai/warp_pack_streaming.py`
* **Install:** `pip install crewai crewai-tools`
* **Pattern:** ToolBase with multi-agent crews

### OpenAI Agents SDK
* **Files:** `openai/warp_md_tool.py`, `openai/example_agent.py`
* **Install:** `pip install openai openai-agents`
* **Pattern:** Function calling with async support

### AutoGen
* **Files:** `autogen/warp_md_tool.py`, `autogen/example_agent.py`
* **Install:** `pip install pyautogen`
* **Pattern:** function_map for multi-agent conversations

## Utilities

### `warp_utils.py`
Common utilities for all frameworks:
- `parse_stream_events()` - Generic NDJSON stream parser
- `ProgressTracker` - Human-readable progress output
- Event handler base classes for custom implementations

## Available Tools

| Tool | CLI | Description |
|------|-----|-------------|
| **warp-md** | `warp-md` | MD trajectory analysis (96+ analyses) |
| **warp-pack** | `warp-pack` | Molecular packing for simulation setup |
| **warp-pep** | `warp-pep` | Peptide building and mutation |

## Quickstart

### Running the Examples

```bash
# LangChain example
cd langchain
python example_agent.py --topology ../../results/alanine/new/peptide_solvated.pdb \
                       --trajectory ../../results/alanine/new/peptide_sim.xtc

# CrewAI example
cd crewai
python example_crew.py --topology ../../results/alanine/new/peptide_solvated.pdb \
                      --trajectory ../../results/alanine/new/peptide_sim.xtc

# Streaming example
python langchain/warp_pack_streaming.py --config pack.yaml --output packed.pdb
```

### Notebook Quickstart

See `agent_quickstart.ipynb` for a notebook-style quickstart covering all frameworks.

### Minimal Python Usage

```python
from examples.agents.warp_utils import run_with_progress

# Run warp-pack with progress tracking
result = run_with_progress([
    "warp-pack", "--config", "pack.yaml", "--stream"
])
print(f"Packed {result['total_atoms']} atoms")
```

## Documentation

| Document | Description |
|----------|-------------|
| [Agent Framework Integrations](../../docs/guides/agent-frameworks.md) | Framework-specific guides |
| [Streaming Progress API](../../docs/guides/streaming-progress.md) | Streaming events reference |
| [Agent Schema & Contract](../../docs/reference/agent-schema.md) | Request/response contracts |
| [Analysis Plans Reference](../../docs/reference/analysis-plans.md) | Available analyses |

## Usage Pattern

All tools follow the same pattern:

1. **Define request** as JSON or parameters
2. **Specify files** (input paths)
3. **Execute** via CLI or Python API
4. **Parse** structured result envelope

### Example: Trajectory Analysis

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain import hub

from examples.agents.langchain.warp_md_tool import create_warp_md_tool

# Create tool
tool = create_warp_md_tool()

# Create agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
agent = create_tool_calling_agent(llm, [tool], hub.pull("hwchase17/openai-tools"))
executor = AgentExecutor(agent=agent, tools=[tool], verbose=True)

# Run
result = executor.invoke({
    "input": "Calculate Rg and RMSD for the protein in my_trajectory.xtc"
})
```

### Example: Molecular Packing with Streaming

```python
from examples.agents.warp_utils import run_with_progress

result = run_with_progress([
    "warp-pack", "--config", "pack.yaml", "--stream"
])

# Output:
# 📦 Packing 150 molecules...
#   → Placing molecules...
#     Placed 10/150 (6.7%)
#   → Optimizing (GenCan)...
#     Iter 100: f=2.10e-03 (10.0%)
#   ✓ gencan complete in 45.00s
# ✅ Pack complete: 4500 atoms in 52.00s
```

## Available Analyses (warp-md)

| Name | Description | Required Fields |
|------|-------------|-----------------|
| `rg` | Radius of gyration | `selection` |
| `rmsd` | RMS deviation | `selection` |
| `msd` | Mean squared displacement | `selection` |
| `rdf` | Radial distribution | `sel_a`, `sel_b`, `bins`, `r_max` |
| `conductivity` | Ionic conductivity | `selection`, `charges`, `temperature` |
| `hbond` | Hydrogen bonds | `donors`, `acceptors`, `dist_cutoff` |
| `end_to_end` | End-to-end distance | `selection` |
| `persistence_length` | Polymer persistence | `selection` |

Full list: [Analysis Plans Reference](../../docs/reference/analysis-plans.md)

## Error Handling

All tools return structured error envelopes:

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

| Code | Meaning | Recovery |
|------|---------|----------|
| `E_CONFIG_VALIDATION` | Invalid JSON | Fix request schema |
| `E_ANALYSIS_SPEC` | Missing required fields | Add required parameters |
| `E_SELECTION_EMPTY` | No atoms matched | Check selection syntax |
| `E_TRAJECTORY_LOAD` | File I/O error | Verify file paths |
