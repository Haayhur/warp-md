---
description: Real-time progress monitoring via NDJSON streaming
icon: activity
---

# Streaming Progress API

warp-md tools support NDJSON (Newline-Delimited JSON) streaming for real-time progress monitoring. Agents can parse events from stderr to track operation status, report progress to users, and estimate completion times.

{% hint style="info" %}
**Why streaming?**

- **Long-running operations** - Molecular packing can take minutes
- **User feedback** - Agents can report "45% complete" instead of just spinning
- **Timeout detection** - Detect stuck processes via lack of events
- **Resource monitoring** - Track iteration counts, convergence rates
{% endhint %}

---

## Enabling Streaming

Add the `--stream` flag to any warp-md CLI command:

```bash
warp-pack --config pack.yaml --stream
warp-pep build -s ACDEFG --stream
```

Events are emitted to **stderr** (one JSON object per line), leaving stdout clean for the actual output.

---

## Event Reference

### warp-pack Events

| Event | Fields | Description |
|-------|--------|-------------|
| `pack_started` | `total_molecules`, `box_size`, `output_path` | Initial configuration |
| `phase_started` | `phase`, `total_molecules`, `max_iterations` | Phase begins |
| `molecule_placed` | `molecule_index`, `total_molecules`, `progress_pct` | Core placement progress |
| `gencan_iteration` | `iteration`, `obj_value`, `pg_sup`, `progress_pct`, `eta_ms` | Optimization iteration |
| `phase_complete` | `phase`, `elapsed_ms`, `final_obj_value` | Phase finished |
| `pack_complete` | `total_atoms`, `elapsed_ms`, `profile_ms` | Final result |
| `error` | `code`, `message` | Error occurred |

#### Phases

- `template_load` - Loading input structures
- `core_placement` - Initial molecule placement
- `movebad` - Stochastic refinement passes
- `gencan` - Gradient-based optimization
- `relax` - Final overlap relaxation

### warp-pep Events

| Event | Fields | Description |
|-------|--------|-------------|
| `operation_started` | `operation`, `total_residues`, `total_mutations` | Build/mutate starts |
| `mutation_complete` | `mutation_index`, `mutation_spec`, `progress_pct` | Each mutation |
| `operation_complete` | `total_atoms`, `elapsed_ms` | Final result |
| `error` | `code`, `message` | Error occurred |

---

## Example Stream Output

```jsonl
{"event":"pack_started","total_molecules":150,"box_size":[50,50,50],"output_path":"out.pdb"}
{"event":"phase_started","phase":"template_load","total_molecules":150}
{"event":"phase_complete","phase":"template_load","elapsed_ms":125}
{"event":"phase_started","phase":"core_placement","total_molecules":150}
{"event":"molecule_placed","molecule_index":10,"total_molecules":150,"progress_pct":6.7}
{"event":"molecule_placed","molecule_index":20,"total_molecules":150,"progress_pct":13.3}
{"event":"phase_complete","phase":"core_placement","elapsed_ms":5420}
{"event":"phase_started","phase":"gencan","max_iterations":1000}
{"event":"gencan_iteration","iteration":10,"max_iterations":1000,"obj_value":1.234e-2,"pg_sup":0.15,"progress_pct":1.0,"eta_ms":495000}
{"event":"gencan_iteration","iteration":100,"max_iterations":1000,"obj_value":2.1e-3,"pg_sup":0.02,"progress_pct":10.0,"eta_ms":45000}
{"event":"phase_complete","phase":"gencan","elapsed_ms":45000,"final_obj_value":1.5e-4}
{"event":"pack_complete","total_atoms":4500,"total_molecules":150,"elapsed_ms":52000,"profile_ms":{"templates":125,"place_core":5420,"gencan":45000,"relax":0}}
```

---

## Python Integration

### Basic Event Parsing

```python
import subprocess
import json

proc = subprocess.Popen(
    ["warp-pack", "--config", "pack.yaml", "--stream"],
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE,
)

for line in proc.stderr:
    event = json.loads(line)
    event_type = event["event"]

    if event_type == "gencan_iteration":
        pct = event["progress_pct"]
        print(f"Optimization: {pct:.1f}%")
    elif event_type == "pack_complete":
        atoms = event["total_atoms"]
        print(f"Done! {atoms} atoms")

proc.wait()
```

### Using the Progress Tracker

```python
from examples.agents.warp_utils import run_with_progress

result = run_with_progress([
    "warp-pack", "--config", "pack.yaml", "--stream"
])
# Output:
# 📦 Packing 150 molecules...
#   → Placing molecules...
#     Placed 10/150 (6.7%)
#     Placed 20/150 (13.3%)
#   → Optimizing (GenCan)...
#     Iter 10: f=1.23e-02, pg=1.50e-01 (1.0%)
#     Iter 100: f=2.10e-03, pg=2.00e-02 (10.0%)
#   ✓ gencan complete in 45.00s (f=1.50e-04)
# ✅ Pack complete: 4500 atoms, 150 molecules in 52.00s
```

### Custom Event Handler

```python
from examples.agents.warp_utils import WarpPackEventHandler, parse_stream_events

class MyHandler(WarpPackEventHandler):
    def on_gencan_iteration(self, event):
        # Log every 10 iterations
        if event.iteration % 10 == 0:
            print(f"Iter {event.iteration}: f={event.obj_value:.2e}")

    def on_pack_complete(self, event):
        # Send notification
        notify_user(f"Packing complete: {event.total_atoms} atoms")

handler = MyHandler()
proc = subprocess.Popen(["warp-pack", "--stream", ...], ...)
parse_stream_events(proc, handler)
```

---

## Agent Framework Integration

### LangChain

```python
from langchain_core.tools import StructuredTool
from examples.agents.warp_utils import run_with_progress

def pack_molecules(config: str, output: str) -> str:
    """Pack molecules with progress tracking."""
    result = run_with_progress([
        "warp-pack", "--config", config, "--output", output, "--stream"
    ])
    return f"Packed to {output}: {result['total_atoms']} atoms"

tool = StructuredTool.from_function(
    func=pack_molecules,
    name="warp_pack_streaming",
    description="Molecular packing with real-time progress",
)
```

### CrewAI

```python
from crewai import Agent
from examples.agents.crewai.warp_pack_streaming import StreamingWarpPackTool

packer = Agent(
    role="Molecular Packer",
    goal="Pack molecules into simulation boxes",
    backstory="Expert in solvation and system preparation",
    tools=[StreamingWarpPackTool()],
)

# Agent receives progress updates automatically
```

---

## Error Handling

All errors emit structured events:

```json
{"event":"error","code":"E_CONFIG_VALIDATION","message":"invalid YAML: ..."}
{"event":"error","code":"E_PLACEMENT_FAILED","message":"failed to place structure water.pdb"}
```

Error codes:
| Code | Meaning | Recovery |
|------|---------|----------|
| `E_CONFIG_VALIDATION` | Invalid config file | Fix YAML/JSON syntax |
| `E_PLACEMENT_FAILED` | Cannot place molecule | Increase box size or reduce molecule count |
| `E_FILE_NOT_FOUND` | Missing input file | Check file paths |
| `E_INTERNAL` | Unexpected error | Report bug |

---

## Best Practices

1. **Always use `--stream`** in agent contexts - enables monitoring
2. **Parse stderr line-by-line** - don't buffer entire output
3. **Handle incomplete lines** - JSON may span multiple lines on error
4. **Monitor `eta_ms`** - estimate completion time for users
5. **Check final event** - `pack_complete`/`operation_complete` confirms success
6. **Log iteration history** - useful for debugging convergence issues

---

## Performance Considerations

- Streaming overhead: <1% CPU, minimal memory
- Event frequency: ~1-10 events/second during optimization
- Use `ProgressTracker(verbose=False)` in production to reduce output

---

## See Also

- [Agent Framework Integrations](agent-frameworks.md) - Framework-specific examples
- [warp-pack Reference](../reference/pack.md) - Full packing documentation
- [Agent Schema & Contract](../reference/agent-schema.md) - Request/response schemas
