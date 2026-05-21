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
| `smiles` | Atomistic molecule source for bead mapping |
| `trajectory_source` | External trajectory/topology to map with warp-md native IO |
| `trajectory_source.kind` | Must be `external` when provided |
| `trajectory_source.target_selection` | Selection for the target molecule inside a solvated topology |
| `trajectory_source.atom_indices` | Explicit target atom indices when selection is not enough |
| `trajectory_source.environment_selection` | Forward-compatible solvent/environment metadata; mapping currently uses target selection or atom indices |
| `trajectory_source.mass_weighted` | Use topology masses for bead centers |
| `reference_source.kind=xtb` | Initiate xTB optimization/MD as the reference source |
| `parameter_tuning` | Fit bonded parameters with BO or PSO |
| `parameter_tuning.max_evaluations` | Positive evaluation budget for BO/PSO |
| `parameter_tuning.swarm_size` | Positive PSO swarm size when provided |
| `output.write_topology_itp` | Write the molecule-level Martini-style ITP |
| `output.write_topology_top` | Write a top-level Gromacs topology wrapper; requires `write_topology_itp=true` and should be set explicitly in partial `output` objects |
| `output.write_cg_pdb` | Write a bead-level CG PDB coordinate artifact for quick downstream setup |
| `output.cg_pdb` | Optional CG PDB filename/path; defaults to `<name>_cg.pdb` |
| `output.write_bonded_parameter_map` | Write JSON explaining how bonded stats/tuned parameters map into ITP rows |

If no bonded reference statistics are available, parameter tuning returns a
structured `skipped` report rather than inventing fitted parameters.

Core artifacts:

| Artifact kind | Contents |
|---------------|----------|
| `martini_mapping_json` | Bead names, atom groups, and bead graph |
| `coarse_grained_pdb` | Bead-level PDB with first mapped-frame coordinates when available, otherwise deterministic scaffold coordinates |
| `coarse_grained_trajectory` | Mapped CG trajectory |
| `bond_stats_json` | Bond means/stds from mapped frames |
| `bonded_stats_json` | Bond, angle, and dihedral statistics |
| `bonded_parameter_map_json` | Machine-readable mapping from bonded stats/tuning names to `[ bonds ]`, `[ angles ]`, and `[ dihedrals ]` ITP rows |
| `bonded_parameter_tuning_report` | BO/PSO bounds, trace, and best parameters |
| `martini_topology_itp` | Martini-style atoms, bonds, angles, and dihedrals |
| `martini_topology_top` | Top-level Gromacs topology wrapper including the generated ITP |
| `xtb_optimized_xyz` | xTB optimized reference structure |
| `xtb_reference_trajectory` | xTB MD trajectory when produced |

## Solvated external trajectory

Use `target_selection` to map only the solute while retaining topology context for selections and masses.
Use either `target_selection` or `atom_indices`; supplying both is rejected to
keep bead mapping deterministic.
Path and selection fields must be non-empty strings.

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
  "parameter_tuning": {
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
  "parameter_tuning": {
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
