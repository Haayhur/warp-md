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
| `frames` | Extract a single frame or stride a trajectory into `.pdb`/`.gro`/`.dcd`/`.xtc`/`.trr` |
| `example` | Print example config JSON |
| `schema` | Print agent schema (`--kind request|result|event|plot-manifest`, JSON/YAML, `--json` alias) |
| `validate` | Validate a run request (`request`, `--stdin`, `--kind request|result|event`) |
| `plan-schema` | Print one analysis contract schema by name or alias |
| `contract-template` | Print an analysis request template (`--fill-defaults`) |
| `normalize` | Canonicalize request aliases and optional defaults (`--strip-unknown`) |
| `capabilities` | Print the warp-md capabilities fingerprint |
| `bundle-plan` | Expand an advertised analysis bundle |
| `inspect-inputs` | Inspect referenced inputs without running analyses |
| `lint-selection` | Validate a selection expression, optionally against a topology |
| `suggest` | Suggest analyses from a natural-language goal description |
| `plot` | Render deterministic SVG plots from a result envelope |
| `build` | Forward a subcommand to the `warp-build` CLI |
| `mcp` | Start the MCP server over stdio |
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
| `--traj PATH` | Trajectory file (DCD/XTC/TRR) |
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

## Meta Commands

Commands for agent workflow — validation, schema inspection, request manipulation, and discovery.

| Command | Flags | What It Does |
|---------|-------|-------------|
| `validate <request>` | `--stdin`, `--kind request|result|event` | Validate a run request JSON against the agent schema |
| `plan-schema <name>` | — | Print the contract schema for one analysis plan by name or alias |
| `contract-template <name>` | `--fill-defaults` | Generate a minimal analysis request template with required fields |
| `normalize <request>` | `--stdin`, `--strip-unknown` | Canonicalize aliases (`topology`→`system`, `traj`→`trajectory`) and fill optional defaults |
| `capabilities` | `--format json|yaml`, `--json` | Print the contract capabilities fingerprint |
| `bundle-plan <name>` | `--format json|yaml`, `--json` | Expand an advertised analysis bundle into individual analysis list |
| `inspect-inputs <request>` | `--stdin` | Validate all referenced input files exist and are readable without running analyses |
| `lint-selection <expr>` | `--topology PATH` | Validate a selection expression syntax and optionally evaluate atom count against topology |
| `suggest <goal>` | `--top-n N`, `--format json|yaml` | Suggest analyses from a natural-language goal description (e.g. "protein stability") |

---

## Complete Analysis Index

Every CLI analysis command, mapped to its plan name and category reference page. Listed here so agents can discover the right command without scanning every category page.

| CLI Command | Plan Name | Category |
|------------|-----------|----------|
| `rg`, `gyrate` | `rg` | Core |
| `rmsd` | `rmsd` | Core |
| `msd` | `msd` | Core |
| `rdf` | `rdf` | Core |
| `mindist` | `mindist` | Geometry |
| `maxdist` | `maxdist` | Geometry |
| `pairdist` | `pairdist` | Geometry |
| `drid` | `drid` | Geometry |
| `watershell` | `watershell` | Solvation |
| `density` | `density` | Solvation |
| `volmap` | `volmap` | Solvation |
| `surf` | `surf` | Solvation |
| `molsurf` | `molsurf` | Solvation |
| `native-contacts` | `native_contacts` | Geometry |
| `shape-descriptors` | `shape_descriptors` | Structural |
| `runningavg` | `runningavg` | Geometry |
| `lineardensity` | `lineardensity` | Solvation |
| `nematic-order` | `nematic_order` | Structural |
| `dipole-moments` | `dipole_moments` | Advanced |
| `free-volume` | `free_volume` | Solvation |
| `hydrophobic-defects` | `hydrophobic_defects` | Solvation |
| `bondi-ffv`, `ffv` | `bondi_ffv` | Solvation |
| `saltbr`, `salt-bridge` | `saltbr` | Geometry |
| `current` | `current` | Advanced |
| `bundle` | `bundle` | Advanced |
| `h2order`, `water-order` | `h2order` | Solvation |
| `hydorder`, `hydration-order` | `hydorder` | Solvation |
| `sorient`, `solvent-orientation` | `sorient` | Solvation |
| `spol`, `solvent-polarization` | `spol` | Solvation |
| `gist` | `gist` | Solvation |
| `dssp` | `dssp` | Structural |
| `kabsch-sander` | `kabsch_sander` | Structural |
| `rama`, `ramachandran` | `rama` | Structural |
| `helix` | `helix` | Structural |
| `helixorient`, `helix-orientation` | `helixorient` | Structural |
| `mdmat`, `distance-matrix` | `mdmat` | Structural |
| `nmr` | `nmr` | NMR/Spectroscopy |
| `jcoupling` | `jcoupling` | NMR/Spectroscopy |
| `diffusion` | `diffusion` | Advanced |
| `pca` | `pca` | Structural |
| `rmsf` | `rmsf` | Structural |
| `projection` | `projection` | Structural |
| `tordiff` | `tordiff` | Structural |
| `docking` | `docking` | Geometry |
| `rotacf` | `rotacf` | Advanced |
| `conductivity` | `conductivity` | Advanced |
| `dielectric` | `dielectric` | Advanced |
| `dipole-alignment` | `dipole_alignment` | Advanced |
| `ion-pair-correlation` | `ion_pair_correlation` | Advanced |
| `structure-factor` | `structure_factor` | Advanced |
| `water-count` | `water_count` | Advanced |
| `equipartition` | `equipartition` | Advanced |
| `hbond` | `hbond` | Advanced |
| `end-to-end` | `end_to_end` | Polymer |
| `contour-length` | `contour_length` | Polymer |
| `chain-rg` | `chain_rg` | Polymer |
| `bond-length-distribution` | `bond_length_distribution` | Polymer |
| `bond-angle-distribution` | `bond_angle_distribution` | Polymer |
| `persistence-length` | `persistence_length` | Polymer |
| `lipid-leaflets` | `lipid_leaflets` | Lipid |
| `lipid-curved-leaflets` | `lipid_curved_leaflets` | Lipid |
| `lipid-z-positions` | `lipid_z_positions` | Lipid |
| `lipid-z-thickness` | `lipid_z_thickness` | Lipid |
| `lipid-z-angles` | `lipid_z_angles` | Lipid |
| `lipid-area` | `lipid_area` | Lipid |
| `lipid-flip-flop` | `lipid_flip_flop` | Lipid |
| `lipid-neighbours` | `lipid_neighbours` | Lipid |
| `lipid-neighbour-matrix` | `lipid_neighbour_matrix` | Lipid |
| `lipid-largest-cluster` | `lipid_largest_cluster` | Lipid |
| `lipid-membrane-thickness` | `lipid_membrane_thickness` | Lipid |
| `lipid-registration` | `lipid_registration` | Lipid |
| `lipid-msd` | `lipid_msd` | Lipid |
| `lipid-scc` | `lipid_scc` | Lipid |

### Category Reference Pages

| Category | Reference |
|----------|-----------|
| Core / Geometry | [Geometry & Distance](geometry-distance.md) |
| Structural / bio | [Structural Analysis](structural-analysis.md) |
| Solvation / density | [Solvation & Density](solvation-density.md) |
| Advanced transport / electrostatics | [Advanced Plans](advanced-plans.md) |
| Polymer | [Polymer Plans](polymer-plans.md) |
| Lipid | [Lipid Analysis](lipid-analysis.md) |
| Transforms | [Transforms](transforms.md) |
| NMR / spectroscopy | [NMR Spectroscopy](nmr-spectroscopy.md) |

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

### Edit Frames

```bash
# Every 10th frame from frames 100..499 into a new DCD
warp-md frames -p min.pdb -t eq_npt.dcd -o md_new.dcd -b 100 -e 500 -s 10

# Single frame extraction
warp-md frames -p min.pdb -t eq_npt.dcd -o frame_250.pdb -i 250

# TRR subset preserving time/velocities/forces/lambda
warp-md frames -p min.pdb -t eq_npt.trr -o subset.trr -b 100 -e 200 -s 5
```

### Run Config

```bash
warp-md run config.json
warp-md run config.json --stream ndjson    # Real-time progress
warp-md run config.json --debug-errors     # Include tracebacks
```

### Plot Results

```bash
warp-md plot warp_md_result.json --out-dir plots
```

Reads `artifact.plot_recommendations` and companion CSV metadata from the result envelope, then emits deterministic SVG plots plus a JSON report.

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
    {"name": "rmsd", "selection": "backbone", "reference": 0, "align": true},
    {"name": "rdf", "sel_a": "resname SOL", "sel_b": "resname SOL", "bins": 200, "r_max": 10.0}
  ]
}
```

---

{% hint style="info" %}
**Output Contract, Streaming Mode & Agent Tool Definition** are documented in the [Agent Schema](agent-schema.md) reference — the single source of truth for envelopes, exit codes, NDJSON events, and tool registration.
{% endhint %}
