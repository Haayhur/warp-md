---
description: Map benzene to Martini beads with xTB reference and BO tuning
icon: atom
---

# Coarse-Graining

Map a small molecule to Martini beads, with xTB reference generation and Bayesian optimization tuning.

---

## SMILES → Martini (xTB Reference)

```bash
# Generate request
warp-cg example > request.json

# Validate, then run with streaming
warp-cg validate request.json
warp-cg run request.json --stream ndjson
```

### Request JSON

```json
{
  "schema_version": "warp-cg.agent.v1",
  "name": "benzene",
  "smiles": "c1ccccc1",
  "reference_source": {
    "kind": "xtb",
    "xtb": {
      "temperature_k": 298.15,
      "time_ps": 2.0,
      "gfn": "gfnff",
      "seed": 42
    }
  },
  "output": {
    "out_dir": "results/cg/benzene",
    "write_mapping_json": true,
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

---

## External Trajectory → Martini

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
    "mass_weighted": true
  },
  "output": {
    "out_dir": "results/cg/benzene_solvated",
    "mapped_trajectory": "benzene_cg.xtc"
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

```bash
warp-cg run request.json --stream ndjson
```

---

## Source Manifest → Martini

For polymer systems already built with `warp-build`:

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
    "repeat_unit_hint": "PAA"
  },
  "output": {
    "out_dir": "cg/paa_50mer"
  }
}
```

---

## Artifacts

| Artifact | Contents |
|----------|----------|
| `<name>_cg.pdb` | Bead-level CG coordinates |
| `<name>_martini.itp` | Martini ITP with bonds, angles, dihedrals |
| `<name>_martini.top` | Gromacs topology wrapper |
| `<name>_bonded_parameter_map.json` | Crosswalk from tuned params to ITP rows |
| `<name>_bonded_stats.json` | Bond/angle/dihedral statistics |
| `<name>_tuning_report.json` | BO/PSO trace and best parameters |
