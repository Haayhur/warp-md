---
description: The complete CLI reference — your agent's command center
icon: terminal
---

# CLI Reference

Everything the warp-md command line can do. Designed for agents, loved by everyone.

---

## Synopsis

```bash
warp-md <command> [options]
warp-md <analysis> --topology PATH --traj PATH [analysis options]
```

---

## Commands

| Command | What It Does |
|---------|--------------|
| `list-plans` | List available analyses (`--format text|json`, `--json`, `--details`) |
| `water-models` | List bundled water templates (`--format text|json`, `--json`) |
| `example` | Print example config JSON |
| `schema` | Print agent schema (`--kind request|result|event`, JSON/YAML, `--json` alias) |
| `run <config>` | Run analyses from config file |
| `<analysis>` | Run single analysis |

---

## Global Options

| Option | What It Does |
|--------|--------------|
| `--help` | Show help |
| `--version` | Show version |

---

## Single Analysis

```bash
warp-md <analysis> --topology PATH --traj PATH [options]
```

### Common Options

| Option | What It Does |
|--------|--------------|
| `--topology PATH` | Topology file (PDB/GRO) |
| `--traj PATH` | Trajectory file (DCD/XTC) |
| `--selection EXPR` | Atom selection |
| `--out PATH` | Output file (.npz/.npy/.csv/.json) |
| `--device DEV` | `auto`, `cpu`, `cuda` |

---

## Available Analyses

### Core

| Analysis | Required Options | What It Computes |
|----------|-----------------|------------------|
| `rg` | `--selection` | Radius of gyration |
| `rmsd` | `--selection` | RMSD from reference |
| `msd` | `--selection` | Mean square displacement |
| `rdf` | `--sel-a`, `--sel-b`, `--bins`, `--r-max` | Radial distribution |

### Polymer

| Analysis | Required Options | What It Computes |
|----------|-----------------|------------------|
| `end-to-end` | `--selection` | End-to-end distance |
| `contour-length` | `--selection` | Contour length |
| `chain-rg` | `--selection` | Per-chain Rg |
| `bond-length-distribution` | `--selection`, `--bins`, `--r-max` | Bond length histogram |
| `bond-angle-distribution` | `--selection`, `--bins` | Bond angle histogram |
| `persistence-length` | `--selection` | Persistence length |

### Advanced

| Analysis | Required Options | What It Computes |
|----------|-----------------|------------------|
| `rotacf` | `--selection`, `--orientation` | Rotational ACF |
| `conductivity` | `--selection`, `--charges`, `--temperature` | Ionic conductivity |
| `dielectric` | `--selection`, `--charges` | Dielectric properties |
| `dipole-alignment` | `--selection`, `--charges` | Dipole alignment |
| `ion-pair-correlation` | `--selection`, `--rclust-cat`, `--rclust-ani` | Ion-pair correlations |
| `structure-factor` | `--selection`, `--bins`, `--r-max`, `--q-bins`, `--q-max` | S(q) |
| `water-count` | `--water-selection`, `--center-selection`, `--box-unit`, `--region-size` | Water grid |
| `equipartition` | `--selection` | Temperature from velocities |
| `hbond` | `--donors`, `--acceptors`, `--dist-cutoff` | H-bond counts |

---

## Examples

{% tabs %}
{% tab title="Rg" %}
```bash
warp-md rg \
  --topology protein.pdb \
  --traj traj.xtc \
  --selection "name CA" \
  --out rg.npz
```
{% endtab %}

{% tab title="RDF" %}
```bash
warp-md rdf \
  --topology system.pdb \
  --traj traj.xtc \
  --sel-a "resname SOL and name OW" \
  --sel-b "resname SOL and name OW" \
  --bins 200 \
  --r-max 10.0 \
  --out rdf.npz
```
{% endtab %}

{% tab title="MSD" %}
```bash
warp-md msd \
  --topology system.pdb \
  --traj traj.xtc \
  --selection "resname SOL" \
  --group-by resid \
  --axis 0,0,1 \
  --lag-mode multi_tau
```
{% endtab %}

{% tab title="Conductivity" %}
```bash
warp-md conductivity \
  --topology il.pdb \
  --traj il.xtc \
  --selection "resname BMIM or resname BF4" \
  --group-by resid \
  --charges table:charges.csv \
  --temperature 300.0
```
{% endtab %}
{% endtabs %}

---

## Config Runner

For batch workflows with multiple analyses.

### Generate Example

```bash
warp-md example > config.json
```

### Run Config

```bash
warp-md run config.json
warp-md run config.json --stream ndjson    # Real-time progress
warp-md run config.json --debug-errors     # Include tracebacks
```

### Discovery Commands

```bash
warp-md schema --format json               # Full request schema
warp-md schema --kind result --format json # Result envelope schema
warp-md schema --kind event --format json  # Stream event schema
warp-md list-plans --format json --details # All analyses + parameter metadata
warp-md water-models --format json         # Bundled water templates
```

### Config Format

```json
{
  "version": "warp-md.agent.v1",
  "run_id": "prod-run-001",
  "system": {"path": "system.pdb"},
  "trajectory": {"path": "traj.xtc"},
  "device": "auto",
  "stream": "none",
  "output_dir": "outputs",
  "analyses": [
    {"name": "rg", "selection": "protein", "out": "outputs/rg.npz"},
    {"name": "rmsd", "selection": "backbone", "reference": "topology", "align": true},
    {"name": "rdf", "sel_a": "resname SOL", "sel_b": "resname SOL", "bins": 200, "r_max": 10.0}
  ]
}
```

---

{% hint style="info" %}
**Output Contract, Streaming Mode & Agent Tool Definition** are documented in the [Agent Schema](agent-schema.md) reference — the single source of truth for envelopes, exit codes, NDJSON events, and tool registration.
{% endhint %}
