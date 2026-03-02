---
description: Integrate warp-md with LangChain, CrewAI, OpenAI Agents, and AutoGen
icon: robot
---

# Agent Framework Integrations

warp-md is built agent-first. Our Pydantic schemas, structured error codes, and JSON envelopes make it drop-in compatible with all major agentic frameworks.

{% hint style="success" %}
**Why warp-md is agent-friendly:**

* **Contract-first** - Pydantic-validated schemas, deterministic outputs
* **Machine-readable** - JSON envelopes with structured error codes
* **CLI-accessible** - Any agent can invoke via subprocess
* **Streaming support** - NDJSON events for real-time progress
* **Zero Python lock-in** - Use from Rust, Python, or any language with subprocess

See [Agent Schema & Contract](reference/agent-schema.md) for the full contract specification.
{% endhint %}

---

## Quick Comparison

| Framework | Integration Pattern | Complexity | Best For |
|-----------|---------------------|------------|----------|
| **LangChain** | StructuredTool | Simple | Tool-calling agents, LangGraph workflows |
| **CrewAI** | ToolBase | Simple | Multi-agent crews, role-based collaboration |
| **OpenAI Agents SDK** | Function calling | Simple | Native OpenAI agents, function-calling |
| **AutoGen** | function_map | Simple | Multi-agent conversations, code generation |

---

## Framework Guides

### 1. LangChain

**Install:**
```bash
pip install langchain langchain-core langchain-openai warp-md
```

**Quick Start:**
```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain import hub

from examples.agents.langchain.warp_md_tool import create_warp_md_tool

# Create the tool
warp_tool = create_warp_md_tool()

# Set up agent
llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = hub.pull("hwchase17/openai-tools")
agent = create_tool_calling_agent(llm, [warp_tool], prompt)
executor = AgentExecutor(agent=agent, tools=[warp_tool], verbose=True)

# Run analysis
result = executor.invoke({
    "input": "Calculate radius of gyration for the protein in protein.pdb and traj.xtc"
})
```

**Files:**
* `examples/agents/langchain/warp_md_tool.py` - Tool implementation
* `examples/agents/langchain/example_agent.py` - Complete agent example

---

### 2. CrewAI

**Install:**
```bash
pip install crewai crewai-tools warp-md
```

**Quick Start:**
```python
from crewai import Agent, Task, Crew
from examples.agents.crewai.warp_md_tool import WarpMDAnalysisTool

# Create specialist agents
analyst = Agent(
    role="MD Analyst",
    goal="Execute trajectory analyses",
    backstory="Expert computational chemist",
    tools=[WarpMDAnalysisTool()],
)

interpreter = Agent(
    role="Data Interpreter",
    goal="Interpret analysis results",
    backstory="Biophysicist specializing in MD data",
)

# Define workflow
analysis_task = Task(
    description="Calculate Rg and RMSD for the protein",
    expected_output="Result file paths",
    agent=analyst,
)

interpretation_task = Task(
    description="Interpret the results",
    expected_output="Scientific summary",
    agent=interpreter,
)

# Execute
crew = Crew(agents=[analyst, interpreter], tasks=[analysis_task, interpretation_task])
result = crew.kickoff()
```

**Files:**
* `examples/agents/crewai/warp_md_tool.py` - Tool implementation
* `examples/agents/crewai/example_crew.py` - Multi-agent crew example

---

### 3. OpenAI Agents SDK

**Install:**
```bash
pip install openai openai-agents warp-md
```

**Quick Start:**
```python
import asyncio
from openai import AsyncOpenAI
from examples.agents.openai.warp_md_tool import WarpMDTool

async def analyze():
    client = AsyncOpenAI()
    tool = WarpMDTool()

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": "Calculate Rg for protein.pdb and traj.xtc"
        }],
        tools=[tool.to_function_definition()],
    )

    # Handle tool call...
```

**Files:**
* `examples/agents/openai/warp_md_tool.py` - Tool implementation
* `examples/agents/openai/example_agent.py` - Async agent example

---

### 4. AutoGen

**Install:**
```bash
pip install pyautogen warp-md
```

**Quick Start:**
```python
from autogen import AssistantAgent, UserProxyAgent
from examples.agents.autogen.warp_md_tool import (
    warp_md_function,
    create_warp_md_tool_decorator,
)

# Create pre-configured tool
md_tool = create_warp_md_tool_decorator("protein.pdb", "traj.xtc")

# Set up agents
user_proxy = UserProxyAgent(
    name="user",
    function_map={"warp_md_analysis": md_tool},
)

analyst = AssistantAgent(
    name="analyst",
    llm_config={"config_list": [{"model": "gpt-4o", ...}]},
)

# Start conversation
user_proxy.initiate_chat(
    analyst,
    message="Analyze the protein structure stability"
)
```

**Files:**
* `examples/agents/autogen/warp_md_tool.py` - Function implementation
* `examples/agents/autogen/example_agent.py` - Multi-agent example

---

## Pattern: Custom Agent Wrapper

For custom frameworks or direct integration:

```python
import json
import subprocess
from pathlib import Path

class WarpMDWrapper:
    """Generic wrapper for any agent framework."""

    def analyze(self, topology: str, trajectory: str, analyses: list) -> dict:
        """Run analysis and return parsed result envelope."""
        request = {
            "version": "warp-md.agent.v1",
            "system": topology,
            "trajectory": trajectory,
            "analyses": analyses,
        }

        config_path = Path("_temp_warp_request.json")
        with open(config_path, "w") as f:
            json.dump(request, f)

        result = subprocess.run(
            ["warp-md", "run", str(config_path)],
            capture_output=True,
            text=True,
        )

        config_path.unlink()
        return json.loads(result.stdout)

# Usage
wrapper = WarpMDWrapper()
result = wrapper.analyze(
    "protein.pdb",
    "traj.xtc",
    [{"name": "rg", "selection": "protein"}]
)
```

---

## Common Analysis Patterns

### Single Analysis
```json
[{"name": "rg", "selection": "protein"}]
```

### Multiple Analyses
```json
[
    {"name": "rg", "selection": "protein"},
    {"name": "rmsd", "selection": "backbone", "align": true},
    {"name": "end_to_end", "selection": "protein"}
]
```

### Water Structure
```json
[
    {
        "name": "rdf",
        "sel_a": "resname SOL and name OW",
        "sel_b": "resname SOL and name OW",
        "bins": 200,
        "r_max": 10.0
    }
]
```

### Transport Properties
```json
[
    {
        "name": "conductivity",
        "selection": "resname NA or resname CL",
        "charges": [1.0, -1.0],
        "temperature": 300.0
    }
]
```

---

## Error Handling

All frameworks receive structured error envelopes:

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

Error codes are machine-parseable:
| Code | Meaning | Recovery |
|------|---------|----------|
| `E_CONFIG_VALIDATION` | Invalid JSON | Fix request schema |
| `E_ANALYSIS_SPEC` | Missing required fields | Add required parameters |
| `E_SELECTION_EMPTY` | No atoms matched | Check selection syntax |
| `E_TRAJECTORY_LOAD` | File I/O error | Verify file paths |

---

## Streaming Progress

warp-md tools support NDJSON streaming for real-time progress monitoring:

### warp-md Analysis
```python
request = {
    ...
    "stream": "ndjson",
    ...
}
```

### warp-pack Packing
```bash
warp-pack --config pack.yaml --stream
```

### warp-pep Building
```bash
warp-pep build -s ACDEFG --stream
```

See [Streaming Progress API](streaming-progress.md) for complete event reference and framework integration examples.

---

## Next Steps

* [Streaming Progress API](streaming-progress.md) - Complete streaming guide
* [Agent Schema & Contract](reference/agent-schema.md) - Full API contract
* [Analysis Plans Reference](reference/analysis-plans.md) - Available analyses
* [Examples](../examples/) - More agent examples
