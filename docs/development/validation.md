---
description: Benchmark manifests, scientific parity, and completion audits
icon: check-circle
---

# Validation

---

## Manifest-Driven Validation

warp-md uses JSON manifests to define validation lanes:

| Manifest | Purpose |
|----------|---------|
| `internal/benchmark/paper_manifest.json` | Publication benchmark definitions |
| `internal/benchmark/paper_manifest_status.json` | Lane pass/fail status |
| `internal/validation/*_manifest.json` | Scientific parity tests |

### Running Validation

```bash
# Build standardized report
.agent/bench-venv/bin/python scripts/bench/build_benchmark_report.py \
  --manifest internal/benchmark/paper_manifest.json

# With timeseries plots
.agent/bench-venv/bin/python scripts/bench/build_benchmark_report.py \
  --manifest internal/benchmark/paper_manifest.json \
  --timeseries-plots \
  --results-dir results/full_trajectory

# Refresh coverage tracker
python scripts/bench/generate_benchmark_coverage.py --write
```

---

## Scientific Parity

warp-md is validated against:

| Reference | Plans Verified |
|-----------|----------------|
| MDAnalysis | Rg, RMSD, MSD, RDF, SASA, RMSF, PCA |
| MDTraj | RMSD, pairwise RMSD, dihedrals |
| GROMACS (gmx) | RDF, conductivity, dielectric, MSD, SASA |
| pytraj/cpptraj | RMSD, distance, dihedral, PCA |

### Validation Reports

Reports live in `internal/analysis_verification/`:
- `master_audit.md` — full plan coverage audit
- `msd_report.md` — MSD accuracy analysis
- `rdf_pairdist_report.md` — RDF and pair distance accuracy
- `sasa_freevolume_report.md` — surface area accuracy
- `hbond_dssp_report.md` — hydrogen bond and DSSP accuracy

---

## Benchmark Datasets

| Dataset | Topology | Trajectory | Source |
|---------|----------|-----------|--------|
| Alanine dipeptide | `results/alanine/new/peptide_solvated.pdb` | `peptide_sim.xtc` | GROMACS |
| Water box | `results/water/water_box.pdb` | `water_sim.xtc` / `water_sim.dcd` | GROMACS |
| Ubiquitin ensemble | `results/ubiquitin-md-generated-ensemble/topology.pdb` | `Q75.dcd` | MD generated |

---

## Efficiency Review Gate

Before moving to a new module, answer:
> How can we improve computational efficiency for this benchmark lane while preserving or improving accuracy parity vs MDAnalysis and MDTraj?

Track prompts and actions in: `manuscript/tables/benchmark_efficiency_questions.md`

---

## Completion Audits

| Audit | File |
|-------|------|
| Master plan audit | `internal/analysis_verification/master_audit.md` |
| Cross-tool gap audit | `internal/analysis_verification/simple_cross_tool_gap_audit.md` |
| Detailed missing module audit | `internal/analysis_verification/detailed_missing_module_audit.md` |
| warp-cg completion | `results/cg/warp_cg_completion_status.json` |
| warp-build e2e verifier | `scripts/validation/verify_warp_build_e2e.py` |
