---
description: Agent-first Martini coarse graining with external trajectories, xTB references, and bonded tuning
icon: atom
---

# warp-cg

`warp-cg` maps atomistic molecules to Martini-style coarse-grained beads and emits deterministic artifacts for agents.

## CLI Reference

| Subcommand | Flags | What It Does |
|-----------|-------|-------------|
| `run <request>` | `--stdin`, `--stream none|ndjson` | Run a first-class agent request JSON with streaming events |
| `validate <request>` | `--stdin` | Validate a first-class agent request JSON (source files, counts, template preconditions, xTB availability) |
| `schema` | `--kind request|result|event` | Print JSON schema for the agent contract |
| `example` | — | Print an example agent request |
| `capabilities` | — | Print agent capabilities fingerprint |
| `build run <request>` | `--stdin`, `--stream none|ndjson` | Execute a CG system build request (membranes, lipids, solvent, ions) |
| `build validate <request>` | `--stdin` | Validate a CG system build request |
| `build schema` | `--kind request|result|event` | Print CG build contract schema |
| `build example` | — | Print an example CG build request |
| `build capabilities` | — | Print CG build capabilities |
| `simulate schema` | `--kind request|plan|result|status|manifest` | Print CG simulation contract schema |
| `simulate example` | `--engine gromacs|openmm` | Print an example simulation request |
| `simulate capabilities` | — | Print CG simulation capabilities |
| `simulate validate <request>` | `--stdin` | Validate a CG simulation request |
| `simulate plan <request>` | `--stdin`, `--engine gromacs|openmm` | Emit an execution handoff plan with MDP templates or runner scripts |
| `simulate status <run_dir>` | — | Inspect a run directory for artifacts, checkpoints, and status |
| `forcefield inspect` | `--kind martini3` | Print the bundled forcefield manifest with SHA-256 hashes |
| `forcefield install` | `--kind martini3`, `--dest PATH`, `--overwrite` | Copy a bundled forcefield snapshot into a project-local directory |
| `runner martini-openmm` | Runner-specific OpenMM flags | Execute one prepared Martini/OpenMM system directly |

---

## Agent contract

Run a request file with streaming events:

```bash
warp-cg run cg_request.json --stream ndjson
```

Core inputs:

| Field | Purpose |
|-------|---------|
| `smiles` | Small-molecule template input |
| `repeat_smiles` | Polymer repeat-unit template input |
| `source` | Built-system or APS handoff input; accepts polymer build manifests, polymer pack manifests, coordinates/topology, coordinates/topology/charge manifest, or source manifests |
| `mapping.mode` | Mapping strategy: `auto` for source graph partitioning, `template` for replaying a `warp-cg.mapping_template.v1` file, or repeat/small-molecule mapping |
| `mapping.strategy` | Auto-mapping strategy, currently `polymer_residue_graph` |
| `mapping.target_bead_size` | Target heavy-atom count per auto-generated bead |
| `mapping.preserve_functional_groups` | Hint for auto mapping to keep graph-detected functional groups together where possible |
| `mapping.template` | Mapping-template path for `mapping.mode=template` |
| `mapping.repeat_unit_hint` | Polymer repeat label such as `PAA` or `PES` |
| `trajectory_source` | External trajectory/topology to map with warp-md native IO |
| `trajectory_source.kind` | Must be `external` when provided |
| `trajectory_source.target_selection` | Selection for the target molecule inside a solvated topology |
| `trajectory_source.atom_indices` | Explicit target atom indices when selection is not enough |
| `trajectory_source.environment_selection` | Forward-compatible solvent/environment metadata; mapping currently uses target selection or atom indices |
| `trajectory_source.mass_weighted` | Use topology masses for bead centers |
| `trajectory_source.sasa` | Optional Shrake-Rupley SASA settings: `probe_radius_nm`, `n_sphere_points`, `radii_nm`, `fallback_radius_nm` |
| `reference_source.kind=xtb` | Initiate xTB optimization/MD as the reference source |
| `forcefield` | Optional Martini3 forcefield resolver for generated `.top` files; supports bundled or project-local path sources |
| `optimization` | Preferred explicit optimization object; supports `source=aa_trajectory` and `source=xtb` |
| `optimization.method` | Search algorithm: `bayesian_optimization`/`bo` or `pso` |
| `optimization.fitting_mode` | Objective/evaluator mode: `direct_statistics`, `distribution_fit`, `external_evaluator`, or `simulation_fit` |
| `optimization.target_terms` | Bonded terms to optimize: `bonds`, `angles`, and/or `dihedrals` |
| `optimization.max_evaluations` | Positive evaluation budget for BO/PSO |
| `optimization.swarm_size` | Positive PSO swarm size when provided |
| `optimization.initial_parameters` | Optional parameter-name map used as the first BO/PSO initial guess; missing names use midpoint defaults |
| `output.write_topology_itp` | Write the molecule-level Martini-style ITP |
| `output.write_topology_top` | Write a top-level Gromacs topology wrapper; requires `write_topology_itp=true` and should be set explicitly in partial `output` objects |
| `output.write_cg_pdb` | Write a bead-level CG PDB coordinate artifact for quick downstream setup |
| `output.cg_pdb` | Optional CG PDB filename/path; defaults to `<name>_cg.pdb` |
| `output.write_bonded_parameter_map` | Write JSON explaining how bonded stats/tuned parameters map into ITP rows |

If no bonded reference statistics are available, parameter tuning returns a
structured `skipped` report rather than inventing fitted parameters.

`optimization.method` only selects the search algorithm. `distribution_fit`
uses mapped reference distributions inside warp-cg. `simulation_fit` is the
production refinement loop for candidate force fields: warp-cg writes candidate
parameters to a JSON-file evaluator or managed runner, the runner executes the
CG simulation, and warp-cg scores the returned candidate trajectory or targets
against the reference. Without `optimization.evaluator` or
`optimization.runner`, BO/PSO does not run candidate CG MD; it tunes against
extracted statistics or grouped reference targets.

## Martini3 forcefield files

`warp-cg` ships a pinned Martini3 forcefield snapshot for deterministic agent
runs. Runtime never fetches forcefield files from the network. Use the bundled
snapshot directly in a request:

```jsonc
"forcefield": {
  "kind": "martini3",
  "source": "bundled",
  "materialize": "copy"
}
```

When `output.write_topology_top=true`, this copies the snapshot under
`output.out_dir/forcefields/martini3`, writes
`warp_cg_forcefield_manifest.json` with SHA-256 hashes, and inserts the
requested Martini include before the generated molecule include.

For a project-local snapshot, install once:

```bash
warp-cg forcefield install --kind martini3 --dest forcefields/martini3
```

Then point requests at it:

```jsonc
"forcefield": {
  "kind": "martini3",
  "source": "path",
  "path": "forcefields/martini3",
  "materialize": "copy"
}
```

Use `materialize="none"` only when the generated `.top` should include files
from a stable user-managed path. Inspect the bundled manifest with:

```bash
warp-cg forcefield inspect --kind martini3
```

## Managed Martini/OpenMM refinement

`optimization.runner.kind="martini_openmm"` is the managed path for real
candidate-simulation refinement. It is a shortcut over the existing JSON-file
evaluator contract: warp-cg writes each BO/PSO candidate to
`candidate.json`, creates `martini_openmm_runner_spec.json`, runs
`python -m warp_md.cg_martini_openmm_evaluator`, then scores the returned
candidate trajectory against the AA/reference target set.

Install the optional runner dependencies in the execution environment:

```bash
pip install "warp-md[martini]"
```

The runner expects a candidate template directory that already contains the
solvated CG `.gro/.top` scaffold and any molecule `.itp` files. The evaluator
copies that template into each `evaluation_000000/` directory, replaces
candidate parameter placeholders such as `{{bond.group_1_length_nm}}`, copies
the materialized Martini forcefield to `evaluation_000000/forcefields/martini3`
when a root `forcefield` request is present, and runs OpenMM under
`evaluation_000000/run/`.

Minimal managed-runner shape:

```jsonc
"optimization": {
  "enabled": true,
  "method": "bo",
  "fitting_mode": "simulation_fit",
  "runner": {
    "kind": "martini_openmm",
    "work_dir": "candidate_evaluations",
    "template_dir": "candidate_template",
    "gro": "system.gro",
    "top": "system.top",
    "replacements": [
      {"path": "molecule.itp", "parameter": "bond.group_1_length_nm", "format": ".5f"}
    ],
    "protocol": {
      "eq_ns": 50.0,
      "prod_ns": 1000.0,
      "platform": "CUDA",
      "device": "0",
      "trajectory_format": "xtc"
    },
    "candidate_extraction": {
      "mapping": {"bead_names": ["B0", "B1"], "atom_indices": [[0], [1]]},
      "connections": [[0, 1]],
      "format": "xtc"
    }
  }
}
```

Use `method="pso"` plus `swarm_size` for PSO; use `method="bo"` or
`method="bayesian_optimization"` plus `bo` options for Bayesian optimization.
The simulation protocol is independent of the optimizer. For smoke tests set
`protocol.dry_run=true`; production runs should leave it false.

The direct runner CLI is available for checking a single prepared Martini
system:

```bash
warp-cg runner martini-openmm \
  --gro system.gro \
  --top system.top \
  --outdir run01 \
  --eq-ns 50 \
  --prod-ns 1000 \
  --platform CUDA \
  --device 0
```

Requests must provide at least one identity/handoff field: `smiles`,
`repeat_smiles`, or `source`. Template replay is source-driven: use
`source` plus `mapping.mode="template"` and `mapping.template`. Source-manifest
requests can be validated and executed without SMILES. When `source` is present,
the run path supports two modes. `mapping.mode="auto"` converts SMILES inputs or source
topology/coordinates into the same internal molecular graph mapper. Source runs
apply that mapper per residue, write terminal-aware residue bead templates, emit
residue-to-bead and AA-to-CG provenance, and build a full-chain CG PDB/ITP/TOP
when requested. They also write `<name>_mapping_template.json` so future runs can
replay the discovered mapping. `mapping.mode="template"` loads a
`warp-cg.mapping_template.v1` file and matches bead atom names against each
source residue role, failing on missing or duplicated required atoms.

For source-driven polymer handoffs, omit `source.target_selection` to map every
atom and residue in the resolved source coordinates. If `source.target_selection`
is present, it must be a normal warp-md topology selection expression such as
`resname PAA` or `chain A`; `polymer` is not a selector token.

Core artifacts:

| Artifact kind | Contents |
|---------------|----------|
| `martini_mapping_json` | Bead names, atom groups, and bead graph |
| `mapping_template_json` | Reusable generated or replayed `warp-cg.mapping_template.v1` template |
| `coarse_grained_pdb` | Bead-level PDB with first mapped-frame coordinates when available, otherwise deterministic scaffold coordinates |
| `coarse_grained_trajectory` | Mapped CG trajectory |
| `aa_to_cg_mapping_provenance` | Source coordinates/topology, selection policy, selected atom/residue counts, terminal roles, residue names/counts, repeat hint, residue-to-bead, and atom-to-bead provenance |
| `bond_stats_json` | Bond means/stds from mapped frames |
| `bonded_stats_json` | Bond, angle, and dihedral statistics |
| `bonded_parameter_map_json` | Machine-readable mapping from bonded stats/tuning names to `[ bonds ]`, `[ angles ]`, and `[ dihedrals ]` ITP rows |
| `bonded_optimization_report` | BO/PSO bounds, trace, and best parameters |
| `martini_topology_itp` | Martini-style atoms, bonds, angles, and dihedrals |
| `martini_topology_top` | Top-level Gromacs topology wrapper including the generated ITP |
| `xtb_optimized_xyz` | xTB optimized reference structure |
| `xtb_reference_trajectory` | xTB MD trajectory when produced |

Result manifests include normalized agent-facing fields:
`summary.input_mode`, `summary.mapping_mode`, `summary.aa_atom_count`,
`summary.cg_bead_count`, `summary.mapped_residue_count`,
`summary.optimized_terms`, `summary.optimization_source`, and
`artifact_paths`.

`warp-cg validate` reports source file existence, coordinate/topology atom
counts when readable, coordinate/topology count mismatches, target-selection
evaluation when a topology is available, template-mode replay preconditions,
bonded-stat preconditions, xTB executable availability, and a
runtime/cost estimate for requested optimization.

## Source handoff request

```json
{
  "schema_version": "warp-cg.agent.v1",
  "name": "paa_50mer",
  "source": {
    "kind": "polymer_pack_manifest",
    "path": "polymer_pack_manifest.json"
  },
  "mapping": {
    "mode": "auto",
    "strategy": "polymer_residue_graph",
    "target_bead_size": 4,
    "preserve_functional_groups": true,
    "repeat_unit_hint": "PAA",
    "terminal_aware": true
  },
  "output": {
    "out_dir": "cg/paa_50mer"
  }
}
```

Supported source-kind examples are shipped in `examples/warp_cg/`:
`polymer_build_manifest_to_cg_request.json`,
`polymer_pack_manifest_to_cg_request.json`, `source_manifest_to_cg_request.json`,
`coordinates_topology_to_cg_request.json`, and
`coordinates_topology_charge_manifest_to_cg_request.json`.

## Solvated external trajectory

Use `target_selection` to map only the solute while retaining topology context for selections and masses.
Use either `target_selection` or `atom_indices`; supplying both is rejected to
keep bead mapping deterministic.
Path and selection fields must be non-empty strings.
For `optimization.source=external_trajectory` or `aa_trajectory`, provide
`trajectory_source.path` and `trajectory_source.topology`. Use
`trajectory_source.target_selection` or `trajectory_source.atom_indices` for
solvated/multi-molecule systems. `length_scale` multiplies input coordinates
before CG output/statistics; use `10.0` when converting nm coordinates to
Angstrom-like PDB coordinates.

```json
{
  "schema_version": "warp-cg.agent.v1",
  "name": "benzene_solvated",
  "smiles": "c1ccccc1",
  "trajectory_source": {
    "kind": "external",
    "path": "results/benzene/solvated.xtc",
    "topology": "results/benzene/solvated.pdb",
    "target_selection": "resname BENZ",
    "length_scale": 10.0,
    "mass_weighted": true
  },
  "output": {
    "out_dir": "results/cg/benzene_solvated",
    "mapped_trajectory": "benzene_cg.xtc",
    "write_mapping_json": true,
    "write_cg_pdb": true,
    "write_topology_itp": true,
    "write_topology_top": true,
    "write_bonded_parameter_map": true
  },
  "optimization": {
    "enabled": true,
    "source": "external_trajectory",
    "method": "bayesian_optimization",
    "fitting_mode": "distribution_fit",
    "max_evaluations": 64,
    "bo": {
      "n_startup_trials": 8,
      "n_candidates": 1024
    },
    "objective": "bonded_parameter_parity"
  }
}
```

## xTB reference workflow

Use `reference_source.kind=xtb` when the agent should generate a reference from SMILES.
The xTB fields use `temperature_k` in K, `time_ps` in ps, and `timestep_fs` /
`dump_fs` in fs. xTB runs emit optimized/reference artifacts when produced and
`optimization.source=xtb` emits `bonded_optimization_report`.

```json
{
  "schema_version": "warp-cg.agent.v1",
  "name": "ethanol_xtb",
  "smiles": "CCO",
  "reference_source": {
    "kind": "xtb",
    "xtb": {
      "temperature_k": 298.15,
      "time_ps": 2.0,
      "timestep_fs": 1.0,
      "dump_fs": 100.0,
      "gfn": "gfnff",
      "seed": 42
    }
  },
  "output": {
    "out_dir": "results/cg/ethanol_xtb",
    "mapped_trajectory": "ethanol_cg.gro",
    "cg_pdb": "ethanol_cg.pdb",
    "write_cg_pdb": true,
    "write_topology_itp": true,
    "write_topology_top": true,
    "write_bonded_parameter_map": true
  },
  "optimization": {
    "enabled": true,
    "source": "xtb",
    "method": "pso",
    "fitting_mode": "distribution_fit",
    "swarm_size": 12,
    "max_evaluations": 48,
    "objective": "bonded_parameter_parity"
  }
}
```

If xTB MD does not produce a usable trajectory, the workflow can fall back to the optimized XYZ structure and still emit mapping/topology artifacts.

## Simulation-backed refinement

Use `optimization.fitting_mode="simulation_fit"` when the objective must come
from real candidate CG simulations instead of native distribution fitting. The
request must provide `reference_source.kind="precomputed"` or another reference
target source plus `optimization.evaluator.json_file.candidate_extraction`.
The evaluator receives `candidate.json`, writes `result.json`, and returns a
candidate trajectory path or candidate target distributions. See
`examples/warp_cg/simulation_fit_bo_request.json` for a BO template; switch
`optimization.method` to `pso` and set `swarm_size` to use PSO on the same
evaluator contract.

## Downstream simulation setup

A complete downstream setup bundle should include:

```text
<name>_cg.pdb
<name>_cg.gro or <name>_cg.xtc
<name>_martini.itp
<name>_martini.top
<name>_martini_mapping.json
<name>_bonded_stats.json
<name>_bonded_parameter_map.json
<name>_tuning_report.json
```

`<name>_bonded_parameter_map.json` is the agent-facing explanation layer.
For every term, it records:

```json
{
  "itp_section": "bonds",
  "beads_zero_based": [0, 1],
  "itp_atoms_one_based": [1, 2],
  "parameter_names": {
    "length_angstrom": "bond_0_1_length_angstrom",
    "force": "bond_0_1_force"
  },
  "source_stat": {
    "bead_i": 0,
    "bead_j": 1,
    "mean": 3.05,
    "std": 0.18,
    "samples": 1000
  },
  "itp_values": {
    "funct": 1,
    "length_nm": 0.305,
    "force": 30.864
  }
}
```

For grouped template or ITP-derived targets, parameter names use class labels
instead of bead indices, for example
`bond.middle.M0_AR1__M0_SO2_length_nm` and
`bond.middle.M0_AR1__M0_SO2_force`. The same names can be supplied in
`optimization.initial_parameters` to seed BO/PSO from previous runs, direct
statistics, or a curated force-field guess. Unknown names fail before
optimization starts; finite values are clamped to generated bounds.

The same structure is emitted for angles and dihedrals, including the exact BO/PSO parameter names used in the tuning report. This keeps the ITP human-readable while giving agents a deterministic crosswalk from fitted parameters back to source statistics.

## Validation gates

xTB lanes require an external `xtb` executable:

```bash
python scripts/validation/run_warp_cg_xtb_manifest.py \
  --manifest internal/validation/warp_cg_xtb_manifest.json \
  --status-out results/cg/warp_cg_xtb_validation_status.json
```

Scientific parity requires external Martini/OpenMM/Gromacs reference bonded statistics:

```bash
python scripts/validation/extract_warp_cg_reference_bonded_stats.py \
  --topology results/reference/gromacs_benzene.gro \
  --trajectory results/reference/gromacs_benzene.xtc \
  --mapping results/cg/benzene/benzene_martini_mapping.json \
  --selection "resname BENZ" \
  --length-scale 10.0 \
  --json-out results/reference/gromacs_benzene_bonded_stats.json

python scripts/validation/validate_warp_cg_bonded_parity.py \
  --warp results/cg/benzene/benzene_bonded_stats.json \
  --reference results/reference/gromacs_benzene_bonded_stats.json \
  --reference-engine gromacs \
  --json-out results/cg/benzene/bonded_parity_validation.json
```

The completion audit stays incomplete until external xTB and scientific parity lanes have real passing status.

---

## Coarse-Grained System Building (`warp-cg build`)

`warp-cg build` is a first-class system building environment for assembling coarse-grained molecular worlds. It supports complex membranes, stacked lipid bilayers, protein boundary foot-prints, mixed solvent/ion registries, phase-separated solvation zones, and outputs Gromacs-compatible topologies and coordinates.

### CLI Workflows

```bash
# Verify system building capabilities
warp-cg build capabilities

# Print the build request schema
warp-cg build schema --kind request

# Dry-run validate a build request
warp-cg build validate request.json

# Execute a build request
warp-cg build run request.json --stream ndjson
```

### Build Request Contract

The build request drives exact-proof deterministic geometry placement, lipid optimization, and solvent/ion filling.

**System Level (`system`)**
* `force_field`: Target force field for topology emission (e.g., `martini3`).
* `box_size_angstrom`: Array of `[X, Y, Z]` specifying the simulation box size.
* `pbc`: Periodicity mode string. Omit it to use the schema default.
* `placement`: Global placement settings.
  * `relaxation`: Enables/disables deterministic coordinate relaxation (`bool`).
  * `max_steps`: Maximum number of push/pull steps during relaxation.
  * `push_tolerance_angstrom`: Distance tolerance for resolving overlaps.

**Membranes (`membranes[]`)**
Each membrane object defines a distinct bilayer or monolayer stack in the Z-axis.
* `name`: Identifier for the membrane block.
* `center_z_angstrom`: Absolute Z-coordinate for the bilayer midplane.
* `protein_boundary`: Controls how lipids adapt around inserted proteins (supports boundary types like convex hulls, concave hulls, alpha shapes, and buffer distances).
* `solvate_voids`: Policy to flood interior membrane voids with solvent (`bool`).
* `leaflets[]`: Array of leaflet configurations.
  * `side`: `"upper"` or `"lower"`.
  * `apl_angstrom2`: Area per lipid. Directly impacts automated count planning.
  * `exclusions[]`: Array of `{"name", "center_angstrom", "radius_angstrom"}` defining manual circular holes.
  * `regions[]`: Defines exact geometrical regions (`circle`, `ellipse`, `rectangle`, `polygon`).
    * `role`: Determines how the region interacts with placement (`hole` or constrained `patch`).
    * `geometry`: Includes shape parameters, `scale_xy`, and `rotate_degrees`.
  * `composition[]`: Array of `{"lipid": <name>, "count": <int>}` defining stoichiometry.

Region objects do not have their own `composition`. A `patch` constrains the
placement union for the containing leaflet; composition remains a leaflet-level
field.

**Environment (`environment`)**
* `ions`: Controls bulk ion neutralization.
  * `neutralize`: Toggles automated counterion insertion (`bool`).
  * `cation` / `anion`: Standard names (e.g., `"Na+"`, `"Cl-"`).
  * `salt_molarity_mol_l`: Target bulk salt concentration.
* `solvent`: Controls solvent flooding.
  * `enabled`: Toggles solvent placement (`bool`).
  * `species`: Supports mixed species ratios with coarse-grained mapping limits.
  * `zones[]`: Allows phase-separated multi-zone solvation (e.g., distinct molarities or solvent types separated by the membrane).

**Outputs (`outputs`)**
* `coordinates`: Emitted system coordinates (e.g., `.gro`, `.pdb`, `.cif`).
* `topology`: Generated top-level topology (e.g., `.top`).
* `manifest`: Output metadata JSON.

### Example Build Request

```json
{
  "schema_version": "warp-cg.build.v1",
  "run_id": "membrane-bilayer-01",
  "mode": "membrane",
  "system": {
    "force_field": "martini3",
    "box_size_angstrom": [120.0, 120.0, 140.0],
    "placement": {
      "relaxation": true,
      "max_steps": 100,
      "push_tolerance_angstrom": 0.01
    }
  },
  "membranes": [
    {
      "name": "bilayer",
      "center_z_angstrom": 0.0,
      "leaflets": [
        {
          "name": "upper",
          "side": "upper",
          "apl_angstrom2": 64.0,
          "exclusions": [
            {
              "name": "protein-footprint",
              "center_angstrom": [0.0, 0.0],
              "radius_angstrom": 10.0
            }
          ],
          "regions": [
            {
              "name": "inspection-hole",
              "role": "hole",
              "geometry": {
                "shape": "circle",
                "center_angstrom": [24.0, 0.0],
                "radius_angstrom": 8.0
              }
            }
          ],
          "composition": [
            {"lipid": "POPC", "count": 64},
            {"lipid": "POPG", "count": 16}
          ]
        },
        {
          "name": "lower",
          "side": "lower",
          "apl_angstrom2": 64.0,
          "composition": [
            {"lipid": "POPC", "count": 80}
          ]
        }
      ]
    }
  ],
  "environment": {
    "ions": {
      "neutralize": true,
      "cation": "Na+",
      "anion": "Cl-"
    },
    "solvent": {
      "enabled": true
    }
  },
  "outputs": {
    "coordinates": "outputs/membrane.gro",
    "topology": "outputs/topol.top",
    "manifest": "outputs/membrane_manifest.json"
  }
}
```

---

## Simulation Handoff Planning (`warp-cg simulate`)

Once your coarse-grained system is built, `warp-cg simulate` handles MD simulation planning without directly executing the computation. It generates structured step execution directories, MDP templates (Gromacs), or runner scripts (OpenMM), and validates execution handoffs.

### CLI Workflows

```bash
# Print simulation schema
warp-cg simulate schema --kind request

# Validate a simulation request
warp-cg simulate validate request.json

# Generate execution shell scripts and plans
warp-cg simulate plan request.json --engine gromacs

# Inspect simulation directory checkpoints and status
warp-cg simulate status runs/cg-gromacs-001
```

### Simulation Request Contract

The simulate command builds standard artifacts (`SimulationPlan`) containing required inputs, execution steps, expected outputs, and static validation warnings without directly executing the engine.

**Top-Level Configuration**
* `run_id`: Unique string identifying the run.
* `engine`: The backend executor. Supported values: `"gromacs"`, `"openmm"`.

**System Integration (`system`)**
* `coordinates`: Path to input coordinates (`.gro`, `.pdb`).
* `topology`: Path to input topology (`.top`).
* `index`: Optional path to an index file (`.ndx`).
* `parameters[]`: Array of included parameter files (e.g., `"martini_v3.0.0.itp"`, `"martini_v3_openmm.xml"`).
* `build_manifest`: Path to the upstream `warp-cg build` manifest.
* `fitting_report`: Path to an upstream parameter tuning report.

**Execution Protocol (`protocol.stages[]`)**
A protocol consists of sequential stages mapping to specific simulation phases (minimization, equilibration, production).
* `name`: Identifier for the stage (e.g., `"minimize"`, `"nvt"`).
* `type`: Stage category (`"energy_minimization"`, `"md"`).
* `ensemble`: Target statistical ensemble (e.g., `"nvt"`, `"npt"`).
* `files`: Key-value map of stage-specific control files.
  * For Gromacs: requires `"mdp"` file paths.
  * For OpenMM: requires `"runner"` python script paths.
* `parameters`: Key-value map of direct protocol overrides (e.g., `"dt_ps": 0.02`, `"nsteps": 50000`).

**Execution Orchestration (`execution`)**
* `mode`: Defines the orchestrator. Supported values: `"external"`, `"local"`, `"slurm"`.
* `work_dir`: The root directory for generating run scripts and staging files.
* `resources`: Resource allocation hints (e.g., `"gpu": true`, `"mpi_ranks": 1`, `"omp_threads": 8`).

### Example GROMACS Simulation Request

```json
{
  "schema_version": "warp-cg.simulate.v1",
  "run_id": "cg-gromacs-001",
  "engine": "gromacs",
  "system": {
    "coordinates": "outputs/membrane.gro",
    "topology": "outputs/topol.top",
    "index": "outputs/index.ndx",
    "parameters": ["martini_v3.0.0.itp"],
    "build_manifest": "outputs/membrane_manifest.json",
    "fitting_report": "outputs/tuning_report.json"
  },
  "protocol": {
    "stages": [
      {
        "name": "minimize",
        "type": "energy_minimization",
        "files": {"mdp": "protocol/minimize.mdp"},
        "parameters": {"integrator": "steep"}
      },
      {
        "name": "nvt",
        "type": "md",
        "ensemble": "nvt",
        "files": {"mdp": "protocol/nvt.mdp"},
        "parameters": {"integrator": "md", "dt_ps": 0.02, "nsteps": 50000}
      }
    ]
  },
  "execution": {
    "mode": "external",
    "work_dir": "runs/cg-gromacs-001",
    "resources": {"gpu": true, "mpi_ranks": 1, "omp_threads": 8}
  }
}
```

---

## Completion evidence

The implementation surface supports native trajectory mapping, xTB initiation,
BO/PSO bonded fitting, mass-weighted centers, GRO/G96/CPT output, and ITP/TOP
generation.

| Gate | Evidence |
|------|----------|
| xTB end-to-end validation | Passing `internal/validation/warp_cg_xtb_manifest.json` lanes with `xtb` installed |
| BO/PSO scientific parity | Passing `internal/validation/warp_cg_bonded_parity_manifest.json`, including the concrete OpenMM bonded parity lane |
| Completion audit | `results/cg/warp_cg_completion_status.json` reports `complete` |

## Agent-native backmapping

Source-driven mappings emit `<name>_backmap_plan.json` with schema
`warp-cg.backmap-plan.v1`. The plan retains the atomistic residue templates,
the exact atom groups used to calculate each CG bead, and inter-residue
covalent links. Keep this artifact with the generated CG model; it is the
machine-readable inverse mapping contract.

Backmapping aligns each residue template against the current CG bead centers.
Placed neighbor atoms and unplaced neighbor bead centers provide orientation
constraints. Placement supports disconnected, branched, cyclic, and
crosslinked residue graphs. The result is an initialization geometry and
should still undergo force-field minimization before production simulation.

```rust
use warp_cg::backmap::BackmapPlan;

let plan: BackmapPlan = serde_json::from_value(backmap_artifact["plan"].clone())?;
let all_atom_coords = plan.execute(&cg_coords)?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

Forward mappings with overlapping atom groups are accepted for analysis but
are rejected by backmapping because they do not define a unique inverse.

Backmapping uses the versioned `warp-cg.backmap.v1` agent contract. Requests
accept inline frames or a trajectory path. Trajectory input is read in bounded
chunks and can stream directly to XTC without returning coordinates. Rebuilt
coordinates are omitted from results by default; enable `include_coordinates`
only for deliberately small jobs.

```bash
warp-cg backmap validate backmap_request.json
warp-cg backmap run backmap_request.json
```

The `warp-cg.backmap-result.v1` result restores source topology atom order and
reports mapped-bead, internal-bond, linked-bond, finite-coordinate, and
chirality diagnostics per frame. Quality thresholds can fail the request or
emit warnings. Outputs include JSON, PDB, GRO, XTC, DCD, and a versioned
minimization handoff.

```json
{
  "schema_version": "warp-cg.backmap.v1",
  "plan_path": "model_backmap_plan.json",
  "trajectory_path": "cg.xtc",
  "chunk_frames": 64,
  "include_coordinates": false,
  "make_whole": true,
  "box_vectors": [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
  "quality": {
    "max_bead_error": 0.00001,
    "max_link_bond_error": 0.25,
    "max_internal_bond_error": 0.000001,
    "max_chirality_inversions": 0,
    "max_steric_clashes": 0,
    "on_violation": "error"
  },
  "output": {
    "out_dir": "backmapped",
    "prefix": "model_aa",
    "formats": ["pdb", "xtc", "json"],
    "minimization_handoff": true
  }
}
```
