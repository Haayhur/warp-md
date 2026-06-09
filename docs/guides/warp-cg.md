---
description: Agent-first Martini coarse graining with external trajectories, xTB references, and bonded tuning
icon: atom
---

# warp-cg

`warp-cg` maps atomistic molecules to Martini-style coarse-grained beads and emits deterministic artifacts for agents.

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
| `reference_source.kind=xtb` | Initiate xTB optimization/MD as the reference source |
| `optimization` | Preferred explicit optimization object; supports `source=aa_trajectory` and `source=xtb` |
| `optimization.target_terms` | Bonded terms to optimize: `bonds`, `angles`, and/or `dihedrals` |
| `optimization.max_evaluations` | Positive evaluation budget for BO/PSO |
| `optimization.swarm_size` | Positive PSO swarm size when provided |
| `output.write_topology_itp` | Write the molecule-level Martini-style ITP |
| `output.write_topology_top` | Write a top-level Gromacs topology wrapper; requires `write_topology_itp=true` and should be set explicitly in partial `output` objects |
| `output.write_cg_pdb` | Write a bead-level CG PDB coordinate artifact for quick downstream setup |
| `output.cg_pdb` | Optional CG PDB filename/path; defaults to `<name>_cg.pdb` |
| `output.write_bonded_parameter_map` | Write JSON explaining how bonded stats/tuned parameters map into ITP rows |

If no bonded reference statistics are available, parameter tuning returns a
structured `skipped` report rather than inventing fitted parameters.

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
  "schema_version": "warp-cg.agent.v2",
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
  "schema_version": "warp-cg.agent.v2",
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
    "max_evaluations": 64,
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
  "schema_version": "warp-cg.agent.v2",
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
    "swarm_size": 12,
    "max_evaluations": 48,
    "objective": "bonded_parameter_parity"
  }
}
```

If xTB MD does not produce a usable trajectory, the workflow can fall back to the optimized XYZ structure and still emit mapping/topology artifacts.

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

## Completion evidence

The implementation surface supports native trajectory mapping, xTB initiation,
BO/PSO bonded fitting, mass-weighted centers, GRO/G96/CPT output, and ITP/TOP
generation.

| Gate | Evidence |
|------|----------|
| xTB end-to-end validation | Passing `internal/validation/warp_cg_xtb_manifest.json` lanes with `xtb` installed |
| BO/PSO scientific parity | Passing `internal/validation/warp_cg_bonded_parity_manifest.json`, including the concrete OpenMM bonded parity lane |
| Completion audit | `results/cg/warp_cg_completion_status.json` reports `complete` |
