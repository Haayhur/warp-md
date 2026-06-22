Bundled Martini 3 Forcefield Snapshot
=====================================

These files are bundled to make warp-cg Martini/OpenMM and Gromacs handoffs
deterministic for agents and offline runs.

Upstream project: https://github.com/marrink-lab/martini-forcefields
License: Apache-2.0 for Martini force-field parameter files. See LICENSE.

Snapshot contents:
- martini_v3.0.0.itp
- martini_v3.0.0_ions_v1.itp
- martini_v3.0.0_nucleobases_v1.itp
- martini_v3.0.0_phospholipids_v1.itp
- martini_v3.0.0_small_molecules_v1.itp
- martini_v3.0.0_solvents_v1.itp
- martini_v3.0.0_sugars_v1.itp
- martini_v3.0_sterols_v1.0.itp

Update policy: replace this directory only through an explicit forcefield
snapshot refresh, then regenerate and review the SHA-256 manifest emitted by
`warp-cg forcefield inspect --kind martini3`.
