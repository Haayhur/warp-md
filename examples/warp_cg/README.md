# warp-cg agent request examples

These files are machine-readable request templates for `warp-cg run <request.json> --stream ndjson`.

- `smiles_small_molecule_request.json`: SMILES-only small molecule mapping.
- `paa_repeat_request.json`: PAA repeat-unit polymer mapping request.
- `pes_repeat_request.json`: PES repeat-unit polymer mapping request.
- `aa_trajectory_tuning_request.json`: AA trajectory bonded-parameter tuning using the `optimization` object.
- `xtb_tuning_request.json`: xTB optimize/MD reference plus bonded-parameter tuning using the `optimization` object.
- `polymer_build_manifest_to_cg_request.json`: warp-build handoff request using `source.kind=polymer_build_manifest`.
- `polymer_pack_manifest_to_cg_request.json`: warp-pack handoff request using `source.kind=polymer_pack_manifest`.
- `source_manifest_to_cg_request.json`: generic APS/source handoff request using `source.kind=source_manifest`.
- `coordinates_topology_to_cg_request.json`: direct coordinates plus topology request.
- `coordinates_topology_charge_manifest_to_cg_request.json`: direct coordinates plus topology plus charge manifest request.
- `solvated_external_bo_request.json`: map a target molecule from a solvated external trajectory, write CG coordinate/topology artifacts, and tune bonded parameters with native Bayesian optimization.
- `xtb_pso_request.json`: initiate an xTB reference workflow from SMILES, map the resulting reference, write CG coordinate/topology artifacts, and tune bonded parameters with PSO.

Paths are placeholders. Replace topology, trajectory, and output directories with real project paths before running.

Manifest/source requests are accepted by `warp-cg validate` for handoff checks
and by `warp-cg run` for source-driven polymer mapping. Use `mapping.mode=auto`
to feed the source topology graph into the shared SMILES/source mapper and emit a reusable
`<name>_mapping_template.json`. Use `mapping.mode=template` with that generated
template, or a curated `warp-cg.mapping_template.v1` file, to replay the mapping
against residue atom names. Both paths emit residue-to-bead maps, AA-to-CG
provenance, a full-chain CG PDB, and Martini-style ITP/TOP artifacts.

For source-driven polymer handoffs, omit `source.target_selection` to map every
atom and residue in the resolved source coordinates. If `source.target_selection`
is provided, it must be a normal warp-md topology selection expression such as
`resname PAA` or `chain A`; `polymer` is not a selector token.

Both request templates set `write_topology_top: true` explicitly so the run emits
a top-level Gromacs `.top` wrapper next to the generated Martini-style `.itp`.
They also request a CG PDB and bonded parameter map so downstream setup can see
both bead coordinates and the exact stats-to-ITP parameter crosswalk.

## Bonded parity validation contract

`bonded_stats_warp_example.json` and `bonded_stats_reference_example.json` show the JSON shape expected by `scripts/validation/validate_warp_cg_bonded_parity.py`.

Example command:

```bash
python scripts/validation/validate_warp_cg_bonded_parity.py \
  --warp examples/warp_cg/bonded_stats_warp_example.json \
  --reference examples/warp_cg/bonded_stats_reference_example.json \
  --reference-engine gromacs \
  --json-out results/cg/example_bonded_parity_validation.json
```

A real scientific-validation lane should replace the reference example with bonded statistics extracted from Martini/OpenMM/Gromacs simulation output using identical bead indices and units.
