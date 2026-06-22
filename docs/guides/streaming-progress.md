---
description: Real-time progress monitoring via NDJSON streaming
icon: activity
---

# Streaming Progress API

Agent-facing run commands emit one JSON event per line on stderr. Stdout remains
available for the non-streaming result envelope or other command output.

## Enable Streaming

```bash
warp-md run analysis.json --stream ndjson
warp-build run build_request.json --stream
warp-pack run pack_request.json --stream ndjson
warp-cg run cg_request.json --stream ndjson
warp-cg build run cg_build_request.json --stream ndjson
warp-pep build -s ACDEFG --stream
```

The flag shape is tool-specific. `warp-build` and `warp-pep` use a boolean
`--stream`; the other commands accept `none|ndjson`.

## Contract Discovery

Do not hard-code event fields when runtime schema discovery is available:

```bash
warp-md schema --kind event
warp-build schema --kind event
warp-pack schema --kind event
warp-cg schema --kind event
warp-cg build schema --kind event
```

## Common Run Lifecycle

`warp-build`, contract-mode `warp-pack`, and `warp-cg build` use the same
high-level lifecycle:

| Event | Required core fields |
|-------|----------------------|
| `run_started` | `schema_version`, `run_id`, `elapsed_ms` |
| `phase_started` | `run_id`, `phase`, `elapsed_ms` |
| `phase_progress` | `run_id`, `phase`, `progress_pct`, `elapsed_ms` |
| `phase_completed` | `run_id`, `phase`, `elapsed_ms` |
| `run_completed` | `run_id`, `elapsed_ms`, `final_envelope` |
| `run_failed` | `run_id`, `elapsed_ms`, `final_envelope` |

Optional fields such as `eta_ms` and `artifact` are declared by each tool's
event schema.

### Example

```jsonl
{"schema_version":"warp-pack.agent.v1","event":"run_started","run_id":"pack-001","elapsed_ms":0}
{"schema_version":"warp-pack.agent.v1","event":"phase_started","run_id":"pack-001","phase":"resolve_inputs","elapsed_ms":2}
{"schema_version":"warp-pack.agent.v1","event":"phase_progress","run_id":"pack-001","phase":"pack","progress_pct":50.0,"elapsed_ms":1200,"eta_ms":1200}
{"schema_version":"warp-pack.agent.v1","event":"phase_completed","run_id":"pack-001","phase":"pack","elapsed_ms":2400}
```

The terminal `run_completed` or `run_failed` event contains the full final
envelope and is the authoritative completion signal.

## `warp-md` Events

| Event | Purpose |
|-------|---------|
| `run_started` | Batch accepted |
| `analysis_started` | One analysis begins |
| `checkpoint` | Optional frame-level progress |
| `analysis_completed` | Analysis succeeded |
| `analysis_failed` | Analysis failed |
| `run_completed` | Batch succeeded; includes final envelope |
| `run_failed` | Fatal batch failure; includes final envelope |

`checkpoint` only appears when the selected backend exposes mid-run progress.

## `warp-pep` Events

`warp-pep` emits lightweight operation events for build and mutation workflows.
Its CLI does not currently expose a schema command, so consumers should accept
unknown fields and key primarily on the `event` discriminator.

## Python Parsing

```python
import json
import subprocess

proc = subprocess.Popen(
    ["warp-pack", "run", "pack_request.json", "--stream", "ndjson"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)

final_envelope = None
for line in proc.stderr:
    event = json.loads(line)
    if event["event"] == "phase_progress":
        print(f"{event['phase']}: {event['progress_pct']:.1f}%")
    elif event["event"] in {"run_completed", "run_failed"}:
        final_envelope = event["final_envelope"]

return_code = proc.wait()
if final_envelope is None:
    raise RuntimeError(f"stream ended without a terminal event ({return_code=})")
```

## Agent Rules

1. Parse stderr incrementally; do not wait for process completion first.
2. Treat each line as one complete JSON object.
3. Correlate events by `run_id`.
4. Allow new optional fields for backwards-safe contract evolution.
5. Use `run_completed` or `run_failed`, not the last progress percentage, as
   the completion signal.
6. Validate events against the installed schema when strict handling is needed.

## See Also

- [Agent Framework Integrations](agent-frameworks.md)
- [warp-pack Guide](packing.md)
- [Agent Schema & Contract](../reference/agent-schema.md)
