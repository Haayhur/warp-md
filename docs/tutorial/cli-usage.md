---
description: Your agent's native language — JSON output, structured commands
icon: terminal
---

# CLI Usage

The command-line interface is where warp-md really shines for AI agents. Structured output, no parsing gymnastics, just run and get JSON.

{% hint style="info" %}
The CLI was **designed for agents first**. Analysis and `run` commands emit JSON envelopes by default; discovery commands support `--json`/`--format json`.
{% endhint %}

---

## Basic Usage

```bash
warp-md <analysis> --topology PATH --traj PATH [options]
```

### What's Available?

```bash
warp-md list-plans                    # Human-readable list
warp-md list-plans --json             # Discover the full schema + all analysis names
warp-md list-plans --format json --details  # Full parameter introspection
```

| Analysis | What It Computes |
|----------|------------------|
| `rg` | Radius of gyration |
| `rmsd` | Root mean square deviation |
| `msd` | Mean square displacement |
| `rdf` | Radial distribution function |
| `rotacf` | Rotational ACF |
| `conductivity` | Ionic conductivity |
| `dielectric` | Dielectric properties |
| `dipole-alignment` | Dipole alignment |
| `ion-pair-correlation` | Ion-pair correlations |
| `structure-factor` | S(q) |
| `water-count` | Water occupancy grid |
| `equipartition` | Temperature from velocities |
| `hbond` | Hydrogen bond counts |
| `end-to-end` | Polymer end-to-end distance |
| `contour-length` | Polymer contour length |
| `chain-rg` | Per-chain Rg |
| `bond-length-distribution` | Bond length histogram |
| `bond-angle-distribution` | Bond angle histogram |
| `persistence-length` | Polymer persistence length |

---

## Examples (Copy-Paste Ready)

{% tabs %}
{% tab title="Rg" %}
```bash
warp-md rg \
  --topology protein.pdb \
  --traj trajectory.xtc \
  --selection "protein"
```
{% endtab %}

{% tab title="RDF" %}
```bash
warp-md rdf \
  --topology system.pdb \
  --traj trajectory.xtc \
  --sel-a "resname SOL and name OW" \
  --sel-b "resname SOL and name OW" \
  --bins 200 \
  --r-max 10
```
{% endtab %}

{% tab title="MSD" %}
```bash
warp-md msd \
  --topology system.pdb \
  --traj trajectory.xtc \
  --selection "resname SOL" \
  --group-by resid \
  --axis 0,0,1
```
{% endtab %}
{% endtabs %}

---

## Output: Agent-Friendly by Default

Every command prints a JSON envelope with standardized fields:

```bash
# Default: writes output.npz, prints JSON envelope
warp-md rg --topology top.pdb --traj traj.xtc --selection "protein"

# Specify output file
warp-md rg --topology top.pdb --traj traj.xtc --selection "protein" --out rg.npy

# Include traceback on errors (for debugging)
warp-md rg --topology top.pdb --traj traj.xtc --selection "protein" --debug-errors
```

{% hint style="success" %}
**The envelope contract**: Every output includes `status`, `exit_code`, and results. Even errors return valid JSON with `status="error"` and a stable `error.code`. Your agent always knows what happened.
{% endhint %}

---

## Config Runner: Batch Workflows

For complex multi-analysis workflows, use a JSON config:

{% stepper %}
{% step %}
## Generate Example Config

```bash
warp-md example > config.json
```
{% endstep %}

{% step %}
## Customize It

```json
{
  "version": "warp-md.agent.v1",
  "run_id": "my-workflow",
  "system": {"path": "system.pdb"},
  "trajectory": {"path": "production.xtc"},
  "output_dir": "outputs",
  "analyses": [
    {"name": "rg", "selection": "protein"},
    {"name": "rmsd", "selection": "backbone", "reference": "topology"},
    {"name": "rdf", "sel_a": "resname SOL", "sel_b": "resname SOL", "bins": 200}
  ]
}
```
{% endstep %}

{% step %}
## Run It

```bash
warp-md run config.json

# Streaming mode for real-time progress
warp-md run config.json --stream ndjson
```
{% endstep %}
{% endstepper %}

{% hint style="info" %}
**Streaming events** include `completed`, `total`, `progress_pct`, and `eta_ms`. Your agent can show progress bars or estimate completion time.
{% endhint %}

---

## Charges and Group Types

Some analyses need extra metadata:

### Charges (for conductivity, dielectric, dipole)

```bash
# Inline array
warp-md conductivity --charges '[1.0,-1.0,1.0]' ...

# From CSV file
warp-md conductivity --charges table:charges.csv ...

# From selection rules
warp-md conductivity --charges 'selections:[{"selection":"resname NA","charge":1.0}]' ...
```

### Group Types (for ion-pair, MSD by type)

```bash
# Inline array
warp-md msd --group-types '[0,1,1,0]' ...

# From selections
warp-md msd --group-types 'selections:resname NA,resname CL' ...
```

---

## Agent Tool Definition

Copy this when prompting AI agents to use warp-md:

```
TOOL warp-md
DESCRIPTION: Run molecular dynamics trajectory analysis. Writes output file and prints JSON envelope.
COMMAND: warp-md <analysis> --topology PATH --traj PATH [analysis flags]
ANALYSES: rg, rmsd, msd, rdf, rotacf, conductivity, dielectric, dipole-alignment, 
          ion-pair-correlation, structure-factor, water-count, equipartition, hbond,
          end-to-end, contour-length, chain-rg, bond-length-distribution, 
          bond-angle-distribution, persistence-length
OUTPUT: .npz by default; --out for .npy/.csv/.json; envelope includes status/exit_code/results
DISCOVERY: warp-md list-plans --json (for full parameter schema)
```

---

<a href="../guides/packing.md" class="button primary">Next: Molecule Packing Guide →</a>
