use anyhow::{anyhow, Result};
use schemars::schema_for;
use serde_json::{json, Value};

use super::{CgEvent, CgRequest, CgResult, AGENT_SCHEMA_VERSION};

pub fn schema_json(kind: &str) -> Result<String> {
    let value = match kind {
        "request" => serde_json::to_value(schema_for!(CgRequest))?,
        "result" => serde_json::to_value(schema_for!(CgResult))?,
        "event" => serde_json::to_value(schema_for!(CgEvent))?,
        other => return Err(anyhow!("unknown warp-cg schema kind: {other}")),
    };
    Ok(serde_json::to_string_pretty(&value)?)
}

pub fn example_request() -> Value {
    json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "output": {
            "out_dir": "results/cg/benzene",
            "write_mapping_json": true,
            "write_topology_itp": true,
            "write_topology_top": true,
            "write_cg_pdb": true,
            "write_bonded_parameter_map": true,
            "exclusions": {"mode": "explicit_all_intra"},
            "dihedrals": {"enabled": false},
            "coordinates": {"unwrap_polymer": true}
        },
        "optimization": {
            "enabled": false,
            "source": "external_trajectory",
            "method": "bayesian_optimization",
            "objective": "bonded_parameter_parity"
        }
    })
}

pub fn example_requests() -> Value {
    json!({
        "smiles_small_molecule": example_request(),
        "paa_repeat_model": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "paa_repeat",
            "repeat_smiles": "C(C(=O)O)",
            "mapping": {"mode": "auto", "repeat_unit_hint": "PAA"},
            "output": {"out_dir": "cg/paa_repeat"}
        },
        "pes_repeat_model": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "pes_repeat",
            "repeat_smiles": "O=S(=O)(c1ccc(Oc2ccc(*)cc2)cc1)",
            "mapping": {"mode": "auto", "repeat_unit_hint": "PES"},
            "output": {"out_dir": "cg/pes_repeat"}
        },
        "source_with_gromacs_ndx_mapping": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "pamam_g1_ndx_reference",
            "source": {
                "kind": "coordinates_topology",
                "coordinates": "G1_DATA/aa_start.pdb",
                "topology": "G1_DATA/aa_topol.tpr",
                "trajectory": "G1_DATA/aa_traj.xtc"
            },
            "mapping": {
                "mode": "ndx",
                "ndx": "G1_DATA/cg_map.ndx"
            },
            "reference_source": {
                "kind": "external",
                "bonded_terms": {
                    "kind": "gromacs_itp",
                    "path": "G1_DATA/cg_model.itp",
                    "molecule_type": "MOL"
                }
            },
            "output": {"out_dir": "cg/pamam_g1_ndx_reference"}
        },
        "grouped_bonded_reference_only": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "pamam_g1_reference_only",
            "trajectory_source": {
                "kind": "external",
                "path": "G1_DATA/aa_traj.xtc",
                "topology": "G1_DATA/aa_topol.tpr"
            },
            "mapping": {
                "mode": "ndx",
                "ndx": "G1_DATA/cg_map.ndx"
            },
            "reference_source": {
                "kind": "external",
                "bonded_terms": {
                    "kind": "gromacs_itp",
                    "path": "G1_DATA/cg_model.itp",
                    "molecule_type": "MOL"
                }
            },
            "output": {
                "out_dir": "cg/pamam_g1_reference_only",
                "mapped_trajectory": "aa_mapped_cg.xtc"
            }
        },
        "aa_trajectory_tuning": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene_traj_tuned",
            "smiles": "c1ccccc1",
            "trajectory_source": {
                "kind": "external",
                "path": "benzene.xtc",
                "topology": "benzene.pdb",
                "target_selection": "resname BENZ"
            },
            "optimization": {
                "enabled": true,
                "source": "aa_trajectory",
                "method": "bayesian_optimization",
                "fitting_mode": "distribution_fit",
                "target_terms": ["constraints", "bonds", "angles", "dihedrals"],
                "initial_parameters": {
                    "bond.group_1_length_nm": 0.47,
                    "bond.group_1_force": 1250.0
                },
                "max_evaluations": 64
            },
            "output": {"out_dir": "cg/benzene_traj_tuned"}
        },
        "xtb_tuning": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "ethanol_xtb",
            "smiles": "CCO",
            "reference_source": {
                "kind": "xtb",
                "xtb": {
                    "mode": "optimize_and_md",
                    "gfn": "gfn2",
                    "temperature_k": 300.0,
                    "time_ps": 10.0
                }
            },
            "optimization": {
                "enabled": true,
                "source": "xtb",
                "method": "bayesian_optimization",
                "fitting_mode": "distribution_fit",
                "target_terms": ["bonds", "angles"],
                "max_evaluations": 64
            },
            "output": {"out_dir": "cg/ethanol_xtb"}
        },
        "polymer_build_manifest_to_cg": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "paa_50mer",
            "source": {
                "kind": "polymer_build_manifest",
                "path": "paa_50mer.build.json"
            },
            "mapping": {
                "mode": "auto",
                "strategy": "polymer_residue_graph",
                "repeat_unit_hint": "PAA",
                "terminal_aware": true
            },
            "output": {"out_dir": "cg/paa_50mer"}
        },
        "polymer_pack_manifest_to_cg": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "paa_50mer_box",
            "source": {
                "kind": "polymer_pack_manifest",
                "path": "polymer_pack_manifest.json"
            },
            "mapping": {
                "mode": "auto",
                "strategy": "polymer_residue_graph",
                "repeat_unit_hint": "PAA",
                "terminal_aware": true
            },
            "output": {"out_dir": "cg/paa_50mer_box"}
        },
        "source_manifest_to_cg": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "paa_source_manifest",
            "source": {
                "kind": "source_manifest",
                "path": "source_manifest.json"
            },
            "mapping": {
                "mode": "auto",
                "strategy": "polymer_residue_graph",
                "repeat_unit_hint": "PAA",
                "terminal_aware": true
            },
            "output": {"out_dir": "cg/paa_source_manifest"}
        },
        "coordinates_topology_to_cg": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "paa_coordinates_topology",
            "source": {
                "kind": "coordinates_topology",
                "coordinates": "paa_50mer.pdb",
                "topology": "paa_50mer.pdb",
                "format": "pdb",
                "topology_format": "pdb"
            },
            "mapping": {
                "mode": "auto",
                "strategy": "polymer_residue_graph",
                "repeat_unit_hint": "PAA",
                "terminal_aware": true
            },
            "output": {"out_dir": "cg/paa_coordinates_topology"}
        },
        "structure_to_cg": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene_structure",
            "source": {
                "kind": "structure",
                "coordinates": "benzene.pdb",
                "format": "pdb",
                "selection": "chain A"
            },
            "bonding": {
                "source": "infer_from_coordinates",
                "infer_bonds": true,
                "on_ambiguous": "warn"
            },
            "chemistry_hints": [
                {
                    "kind": "smiles",
                    "scope": "molecule",
                    "value": "c1ccccc1"
                }
            ],
            "chemistry_policy": {
                "hint_mode": "validate",
                "on_conflict": "warn"
            },
            "polymer": {
                "enabled": false,
                "role_mode": "infer",
                "terminal_aware": false,
                "end_group_policy": "preserve"
            },
            "mapping": {
                "mode": "auto"
            },
            "output": {"out_dir": "cg/benzene_structure"}
        },
        "template_assignment_only_to_cg": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "pes_chain_from_trimer_template",
            "source": {
                "kind": "structure",
                "coordinates": "pes_chain.pdb",
                "format": "pdb"
            },
            "mapping": {
                "mode": "template",
                "template": "pes_trimer_mapping_template.json",
                "template_policy": "assignment_only",
                "bonded_classing": {
                    "mode": "auto",
                    "source": "template_role_order"
                },
                "expected_beads_per_role": {
                    "head": 8,
                    "middle": 8,
                    "tail": 8
                },
                "on_bead_count_mismatch": "error"
            },
            "output": {"out_dir": "cg/pes_chain"}
        },
        "coordinates_topology_charge_manifest_to_cg": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "paa_coordinates_topology_charge",
            "source": {
                "kind": "coordinates_topology_charge_manifest",
                "coordinates": "paa_50mer.pdb",
                "topology": "paa_50mer.pdb",
                "charge_manifest": "paa_50mer_charge_manifest.json",
                "format": "pdb",
                "topology_format": "pdb"
            },
            "mapping": {
                "mode": "auto",
                "strategy": "polymer_residue_graph",
                "repeat_unit_hint": "PAA",
                "terminal_aware": true
            },
            "output": {"out_dir": "cg/paa_coordinates_topology_charge"}
        }
    })
}

pub fn capabilities() -> Value {
    json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "tool": "warp-cg",
        "contracts": {
            "request": AGENT_SCHEMA_VERSION,
            "result": AGENT_SCHEMA_VERSION,
            "event": AGENT_SCHEMA_VERSION
        },
        "inputs": {
            "identity_fields": {
                "requires_one_of": ["smiles", "repeat_smiles", "source", "trajectory_source with mapping.mode=ndx"],
                "smiles": "small molecule template input",
                "repeat_smiles": "polymer repeat-unit template input",
                "source": "built-system or APS handoff input",
                "mapping.template": "known or generated warp-cg.mapping_template.v1 path for mapping.mode=template",
                "mapping.ndx": "Gromacs NDX bead mapping for source-driven or reference-only grouped bonded workflows"
            },
            "accepted_source_kinds": {
                "structure": {"required": ["source.coordinates"], "optional": ["source.format", "source.selection", "bonding", "chemistry_hints", "chemistry_policy", "polymer"], "example": "examples().structure_to_cg"},
                "polymer_build_manifest": {"required": ["source.path"], "optional": ["source.target_selection"], "example": "examples/warp_cg/polymer_build_manifest_to_cg_request.json"},
                "polymer_pack_manifest": {"required": ["source.path"], "optional": ["source.target_selection"], "example": "examples/warp_cg/polymer_pack_manifest_to_cg_request.json"},
                "coordinates_topology": {"required": ["source.coordinates", "source.topology"], "optional": ["source.target_selection"], "example": "examples/warp_cg/coordinates_topology_to_cg_request.json"},
                "coordinates_topology_charge_manifest": {"required": ["source.coordinates", "source.topology", "source.charge_manifest"], "optional": ["source.target_selection"], "example": "examples/warp_cg/coordinates_topology_charge_manifest_to_cg_request.json"},
                "source_manifest": {"required": ["source.path"], "optional": ["source.target_selection"], "example": "examples/warp_cg/source_manifest_to_cg_request.json"}
            },
            "source_selection_semantics": {
                "default": "when source.selection/source.target_selection is omitted, source-driven mapping uses every atom and residue in the resolved source coordinates",
                "selection": "source.selection is the preferred agent-facing field; source.target_selection remains accepted for compatibility. It must be a valid warp-md topology selection expression, for example 'resname PAA' or 'chain A'; the literal string 'polymer' is not a selector",
                "execution_scope": "source-driven auto/template mapping currently builds residue-aware CG topology from the selected/default coordinate scope; manifest examples omit target_selection to select the full polymer handoff",
                "provenance": "aa_to_cg_mapping_provenance records selection policy, selected atom/residue counts, terminal roles, residue names/counts, repeat hint, chemistry hints, and atom-to-bead links"
            },
            "bonding": {
                "structure_default": "source.kind=structure may infer bonds from coordinates when no explicit bonds/topology are present; inference is reported in result.mapping_summary and warnings",
                "policy": "bonding.source supports explicit_topology, infer_from_coordinates, or template; bonding.on_ambiguous supports warn or error"
            },
            "chemistry_hints": {
                "status": "accepted_reported_and_validated",
                "supported_kinds": ["smiles", "template", "inline_graph"],
                "supported_scopes": ["molecule", "repeat_unit", "residue", "residue_role"],
                "policy": "chemistry_policy.hint_mode accepts validate, fill_missing, prefer_hint, or prefer_geometry; current auto mode records hints/provenance and uses geometry unless explicit template/ndx mapping is requested. Prefer a curated mapping.template with mapping.template_policy=assignment_only when the hint/template should be the mapping authority.",
                "smiles_validation": "SMILES hints are parsed and compared against source geometry aromatic six-ring perception; hint/geometry conflicts are emitted as warp_cg.chemistry_hint_geometry_conflict warnings or errors according to chemistry_policy.on_conflict"
            },
            "polymer_policy": {
                "default": "source.kind=structure is standalone unless polymer.enabled=true or mapping.strategy=polymer_residue_graph",
                "role_mode": ["infer", "explicit"],
                "end_group_policy": ["preserve", "map_as_repeat"],
                "result": "result.mapping_summary and provenance report role assignment and per-residue bead counts"
            },
            "trajectory_mapping": {
                "preferred_field": "trajectory_source",
                "formats_tested": ["dcd", "xtc", "gro", "g96", "cpt", "h5md", "tng", "trr", "pdb", "pdbqt"],
                "target_description": "topology plus warp-md selection expressions or explicit atom indices",
                "environment_selection": "accepted for future solvent/environment metadata; mapping currently uses target_selection or atom_indices",
                "make_whole": "optional boolean; when true, mapped CG coordinates are repaired across periodic boundaries using constraint/bond connectivity before reference bonded distributions are accumulated",
                "sasa": "optional Shrake-Rupley SASA config: probe_radius_nm, n_sphere_points, radii_nm, and fallback_radius_nm. Defaults use the reference-compatible 0.26 nm probe and topology-derived vdW radii when topology is readable and atom-count compatible."
            },
            "reference_targets": {
                "preferred_field": "reference_source.bonded_terms",
                "accepted_kinds": ["gromacs_topology", "gromacs_itp"],
                "required": ["reference_source.bonded_terms.path", "reference_source.bonded_terms.molecule_type"],
                "semantics": "derive grouped constraint/bond/angle/dihedral reference distributions from the mapped reference trajectory using the supplied CG topology/ITP",
                "reference_only_ndx": "trajectory_source + mapping.mode=ndx + reference_source.bonded_terms can map an AA/fine-grained trajectory into CG reference targets without smiles/repeat_smiles/source",
                "precomputed": "reference_source.kind=precomputed + reference_source.precomputed.target_set loads an existing ReferenceTargetSet JSON without rerunning trajectory mapping",
                "transforms": "reference_source.transform supports bonded reference bond_scaling, min_bond_length_nm, specific_bond_lengths_nm, and rg_offset_nm"
            },
            "reference_metrics": {
                "preferred_field": "reference_source.metrics",
                "accepted_kinds": ["json"],
                "semantics": "consumer-owned metric sidecars are merged into result.reference.metrics; use this for exact Gromacs/OpenMM/xTB analyses when those should be authoritative. Native trajectory extraction emits rg_mean_nm and standard Shrake-Rupley sasa_mean_nm2/sasa_std_nm2 metrics. External candidate scoring compares rg_mean_nm and sasa*_mean_nm2 when both reference and candidate metrics are available, and applies a large finite penalty when a declared reference metric is missing from candidate results.",
                "json_shape": {"metrics": {"metric_name": "number"}, "artifacts": [{"path": "optional relative/absolute path", "kind": "artifact kind"}]},
                "namespace": "optional prefix applied as namespace.metric_name"
            },
            "forcefield": {
                "preferred_field": "forcefield",
                "accepted_kinds": ["martini3"],
                "accepted_sources": ["bundled", "path"],
                "bundled": "forcefield.source=bundled uses the pinned Martini3 snapshot shipped under warp-cg assets; no network fetch occurs during validate or run",
                "path": "forcefield.source=path + forcefield.path uses a user/project-local Martini3 directory with the expected files",
                "materialize": "copy writes a deterministic forcefields/martini3 bundle plus warp_cg_forcefield_manifest.json under output.out_dir; none is accepted only with source=path",
                "include_files": "optional list of relative files to include before the generated molecule ITP; defaults to martini_v3.0.0.itp",
                "cli": "warp-cg forcefield inspect --kind martini3; warp-cg forcefield install --kind martini3 --dest forcefields/martini3",
                "artifacts": ["forcefield_manifest_json", "forcefield_directory"]
            },
            "topology": "used to validate solvated external trajectories and resolve target_selection",
            "xtb": {
                "reference_source": "reference_source.kind=xtb can initiate xTB optimization/MD",
                "executable_detection": "PATH lookup for xtb",
                "env": ["PATH", "OMP_NUM_THREADS"],
                "gfn": ["gfnff", "gfn0", "gfn1", "gfn2"]
            }
        },
        "optimization": {
            "status": "implemented",
            "methods": ["bayesian_optimization", "pso"],
            "method_aliases": {"bo": "bayesian_optimization"},
            "sources": {
                "external_trajectory": {
                    "required": ["smiles/repeat_smiles or mapping.mode=ndx", "trajectory_source.path"],
                    "selection": "trajectory_source.target_selection or trajectory_source.atom_indices selects the mapped solute; omit only for single-molecule trajectories",
                    "reference_targets": "optional reference_source.bonded_terms selects topology-defined grouped terms instead of graph-derived bonds only",
                    "units": {"length_scale": "multiply input coordinates by this value before writing CG output; use 10.0 for nm-to-Angstrom input when needed"},
                    "pbc": "set trajectory_source.make_whole=true for connectivity-based periodic repair of CG reference bonded targets",
                    "artifacts": ["coarse_grained_trajectory", "reference_targets_json", "bond_stats_json", "bonded_stats_json", "bonded_optimization_report"],
                    "example": "examples/warp_cg/solvated_external_bo_request.json"
                },
                "aa_trajectory": {
                    "required": ["smiles/repeat_smiles or mapping.mode=ndx", "trajectory_source.path", "optimization.enabled=true", "optimization.source=aa_trajectory"],
                    "selection": "same as external_trajectory",
                    "reference_targets": "optional reference_source.bonded_terms.path + molecule_type derives grouped bonded targets from the CG topology/ITP",
                    "artifacts": ["coarse_grained_trajectory", "reference_targets_json", "bonded_parameter_map_json", "bonded_optimization_report"],
                    "example": "examples/warp_cg/aa_trajectory_tuning_request.json"
                },
                "xtb": {
                    "required": ["smiles or repeat_smiles", "reference_source.kind=xtb", "optimization.enabled=true", "optimization.source=xtb"],
                    "units": {"temperature_k": "K", "time_ps": "ps", "timestep_fs": "fs", "dump_fs": "fs"},
                    "artifacts": ["xtb_optimized_xyz", "xtb_reference_trajectory", "reference_targets_json", "bonded_optimization_report"],
                    "example": "examples/warp_cg/xtb_tuning_request.json"
                },
                "precomputed_reference_with_runner": {
                    "required": ["smiles/repeat_smiles or source mapping", "reference_source.kind=precomputed", "reference_source.precomputed.target_set", "optimization.evaluator.kind=json_file or optimization.runner.kind=martini_openmm"],
                    "semantics": "use existing ReferenceTargetSet JSON as the reference and delegate candidate simulation to a consumer-owned JSON-file runner or the managed Martini/OpenMM runner",
                    "runner_result": "runner may return objective directly, candidate_targets for immediate warp-cg EMD scoring, or candidate_trajectory when evaluator-side extraction context is configured; auto mode uses simulation-backed scoring when candidate_extraction is present"
                }
            },
            "objective": "bonded_parameter_parity",
            "fitting_modes": {
                "auto": "default behavior: if a json_file evaluator or managed runner has candidate_extraction, require candidate_trajectory and score extracted candidate targets/Rg/SASA in warp-cg; otherwise use external evaluator/runner when provided, distribution_fit when reference targets exist, or stats-proxy BO/PSO",
                "direct_statistics": "do not run BO/PSO; assign equilibrium values from mapped AA bonded statistics and estimate force constants from inverse fluctuations",
                "distribution_fit": "optimize equilibrium value/phase and force constant parameters per selected grouped reference target against mapped AA reference distributions with EMD scoring",
                "external_evaluator": "delegate candidate CG simulation/scoring to optimization.evaluator or optimization.runner and use returned objective/candidate_targets/candidate_trajectory",
                "simulation_fit": "strict simulation-backed mode; requires evaluator.json_file.candidate_extraction or runner.candidate_extraction and candidate_trajectory results, ignores runner objective, and scores extracted candidate bonded targets plus Rg/SASA metrics in warp-cg"
            },
            "managed_runner": {
                "field": "optimization.runner",
                "kind": "martini_openmm",
                "dependency": "optional Python runtime dependency: martini-openmm plus openmm; core warp-md imports do not require it",
                "inputs": ["gro", "top", "optional template_dir", "optional replacements", "optional protocol"],
                "template_replacements": "candidate named parameters replace placeholders like {{bond.group_1_length_nm}} in copied template files; explicit replacements can restrict parameter/file/format",
                "protocol": "minimize, NPT equilibration, and optional NPT/NVT production with platform, precision, device, timestep, cutoff, reporting, defines, epsilon_r, and dry_run controls",
                "command": "warp-cg writes martini_openmm_runner_spec.json under output.out_dir/<runner.work_dir> and invokes python -m warp_md.cg_martini_openmm_evaluator for each candidate",
                "direct_cli": "warp-cg runner martini-openmm --gro system.gro --top system.top --outdir run01 --eq-ns 50 --prod-ns 1000 --platform CUDA --device 0"
            },
            "metric_scoring": {
                "field": "optimization.metric_scoring",
                "rg_weight": "non-negative multiplier for normalized rg_mean_nm residual; default 1.0",
                "sasa_weight": "non-negative multiplier for normalized sasa*_mean_nm2 residual; default 1.0",
                "missing_metric_penalty": "positive finite penalty applied per required or reference-declared metric missing from candidate results; default 1e6",
                "require_rg": "when true, missing reference or candidate Rg is penalized",
                "require_sasa": "when true, missing reference or candidate SASA is penalized"
            },
            "sample_policy": {
                "min_samples_per_term": "default 2",
                "allow_single_frame": "set true only for smoke tests or geometry-center checks",
                "on_insufficient_samples": "warn by default; error fails production fitting when any selected term has fewer than min_samples_per_term samples"
            },
            "initial_parameters": {
                "field": "optimization.initial_parameters",
                "semantics": "optional map from generated parameter name to initial value; used as the first BO/PSO initial guess before midpoint/random proposals",
                "partial": "partial maps are allowed; missing parameters are filled from the parameter-space midpoint",
                "validation": "unknown parameter names fail before optimization starts; finite values are clamped to generated bounds",
                "examples": ["bond.middle.M0_AR1__M0_SO2_length_nm", "bond.middle.M0_AR1__M0_SO2_force", "angle.middle.M0_A__M0_B__M1_A_angle_deg", "dihedral.middle.M0_A__M0_B__M1_A__M1_B_force"]
            },
            "unit_policy": {
                "generated_bond_reference_units": "nm",
                "gromacs_rendering": "bond equilibrium values are converted to nm from the declared reference target units when writing topology files"
            },
            "runner_contract": {
                "request_schema": "warp-cg.objective-request.v1",
                "result_schema": "warp-cg.objective-result.v1",
                "request_fields": ["candidate", "reference_targets when available"],
                "result_fields": {
                    "objective": "optional finite scalar objective; used directly when supplied except in simulation_fit or auto+candidate_extraction scoring",
                    "candidate_targets": "optional ReferenceTargetSet JSON for candidate CG simulation distributions; when objective is omitted, warp-cg scores this against reference_targets with bonded EMD plus available Rg/SASA metric residuals",
                    "candidate_trajectory": "optional object with path and optional mapped_trajectory_name; when evaluator candidate_extraction is configured, warp-cg maps/extracts targets and trajectory metrics from this trajectory using the same TargetExtractor path as ReferenceProvider",
                    "metrics": "optional finite numeric map copied into optimizer evaluation records",
                    "status": "completed, failed_simulation, failed_extraction, timed_out, or invalid_parameters"
                },
                "candidate_extraction_config": {
                    "field": "optimization.evaluator.json_file.candidate_extraction or optimization.runner.candidate_extraction",
                    "required_for_candidate_trajectory": ["mapping.bead_names", "mapping.atom_indices", "connections or bonded_terms"],
                    "optional_reader_fields": ["format", "topology", "topology_format", "start", "stop", "stride", "length_scale", "target_selection", "atom_indices", "mass_weighted", "make_whole", "chunk_frames", "sasa"],
                    "bonded_terms": "optional Gromacs topology/ITP source for grouped constraint/bond/angle/dihedral candidate target extraction"
                },
                "engine_boundary": "OpenMM, Gromacs, xTB, or user scripts own simulation; warp-cg exchanges candidate parameters, reference targets, candidate targets or candidate trajectory paths, metrics, and status"
            },
            "defaults": {"method": "bayesian_optimization", "max_evaluations": 32, "seed": 42},
            "bo": {
                "method": "set optimization.method to bayesian_optimization or bo",
                "algorithm": "gp_expected_improvement",
                "acquisition": "log_expected_improvement by default; expected_improvement is also accepted",
                "n_startup_trials": "optional positive Latin-hypercube warmup count",
                "n_candidates": "optional positive acquisition candidate count",
                "checkpoint_path": "optional JSON checkpoint path for long BO runs",
                "checkpoint_interval_evaluations": "save checkpoint after this many additional objective evaluations",
                "resume_from_checkpoint": "resume matching checkpoint state, including RNG state and evaluator signature"
            },
            "pso": {
                "swarm_size": "optional particle count override; default uses int(10 + 2*sqrt(dimensions))",
                "discrete_choices": "Rust optimizer API supports fst-pso-style categorical search by expanding each choice set into a probability segment and sampling choice indices before objective evaluation",
                "advanced": {
                    "fuzzy_self_tuning": "enable FST-style fuzzy adaptation of inertia/cognitive/social/min/max velocity controls",
                    "fuzzy_adapt_inertia": "allow fuzzy rules to update inertia",
                    "fuzzy_adapt_cognitive": "allow fuzzy rules to update cognitive factor",
                    "fuzzy_adapt_social": "allow fuzzy rules to update social factor",
                    "fuzzy_adapt_min_velocity": "allow fuzzy rules to update minimum velocity multiplier",
                    "fuzzy_adapt_max_velocity": "allow fuzzy rules to update maximum velocity multiplier",
                    "reboot_stalled_particles": "restart particles that fail to improve their local best for the configured threshold",
                    "reboot_after_local_stall_iterations": "local stall threshold before restart",
                    "restart_strategy": "random or recombine; recombine restarts stalled particles from elite personal-best positions using an FST-style combine operator while preserving the max_evaluations budget",
                    "linear_population_decrease": "experimental FST-style population decrease across the evaluation budget",
                    "max_iterations_without_global_best": "stop after this many iterations without global best improvement",
                    "checkpoint_path": "optional JSON checkpoint path for long PSO runs",
                    "checkpoint_interval_evaluations": "save checkpoint after this many additional objective evaluations",
                    "resume_from_checkpoint": "resume matching checkpoint state, including swarm and RNG state",
                    "discrete_probability_dilation": "enable fst-pso smooth-ramp sharpening before categorical sampling",
                    "discrete_probability_dilation_alpha": "smooth-ramp exponent; fst-pso commonly uses 8"
                }
            },
            "terms": ["bonds", "angles", "dihedrals"],
            "empty_stats_behavior": "returns skipped report when no bonded reference statistics are available"
        },
        "polymer_mapping": {
            "status": "implemented_auto_template_and_ndx_modes",
            "execution_status": "built-system source handoffs execute in auto mode and emit reusable mapping templates; template mode replays generated/curated residue-role atom-name mappings; ndx mode consumes Gromacs NDX bead sections",
            "modes": {
                "auto": "shared graph mapper fed by either SMILES or source coordinates/topology; source mode adds residue roles, templates, and provenance",
                "template": "replay a warp-cg.mapping_template.v1 file against source residue atom names; mapping.template_policy=strict_graph validates elements/local_bonds/connected groups, while assignment_only replays atom-name assignments, bead types, features, charges, and role templates without graph-derived local-bond failures",
                "ndx": "read Gromacs NDX bead sections as explicit AA-to-CG atom groups, preserving split mappings"
            },
            "template_policy": {
                "default": "strict_graph",
                "strict_graph": "validate template elements, local_bonds, and connected bead groups against target source geometry",
                "assignment_only": "treat the template as the mapping authority and skip graph-derived local_bonds/connected checks after atom names resolve"
            },
            "bead_count_guards": {
                "field": "mapping.expected_beads_per_role",
                "roles": ["head", "middle", "tail", "standalone"],
                "mismatch_policy": "mapping.on_bead_count_mismatch supports error or warn and emits warp_cg.bead_count_mismatch with residue role/count details"
            },
            "bonded_classing": {
                "field": "mapping.bonded_classing",
                "default": {"mode": "auto", "source": "template_role_order"},
                "modes": {
                    "auto": "derive grouped BondedTermSet classes from head/middle/tail role, template bead names, relative residue offset, and canonical forward/reverse ordering",
                    "explicit": "use user/agent supplied zero-based CG bead member classes and apply on_unclassified auto/singleton/drop/error",
                    "patch": "start from auto classes, then merge, split, or rename selected class labels"
                },
                "validation": ["member index range", "term tuple length", "duplicate member policy", "class label existence in patch mode", "non-empty classes", "explicit members present in generated CG bonded graph", "on_unclassified=error coverage"]
            },
            "supported_handoff_schemas": ["warp-build.manifest.v1", "warp-pack manifest", "coordinates_topology"],
            "terminal_aware_polymers": true,
            "multi_residue_templates": true,
            "chemistry_generalization_limits": {
                "tested_auto_classes": ["PAA-like vinyl polymers with carboxylic acid/carboxylate groups", "PES-like aromatic sulfone/ether repeat units", "small organic SMILES examples"],
                "auto_mapping_strengths": ["connected residue graph partitioning", "functional-group preservation for carboxylate/carboxylic acid, amide-like, sulfone/sulfonate-like, and phosphate-like groups", "terminal-aware head/middle/tail residue roles"],
                "requires_template_or_manual_review": ["ambiguous residue atom names", "crosslinked or network polymers", "branched polymers without clear residue order", "metals/coordination chemistry", "charged groups outside implemented functional-group detectors", "multi-component residues where one residue is not one repeat unit"],
                "template_required_when": "auto mapping does not preserve intended chemistry, source residue atom names must match a curated mapping, or reproducibility across related polymers is required"
            },
            "implemented_outputs": [
                "mapping_template_json",
                "backmap_plan_json",
                "repeat_unit_bead_template",
                "head_middle_tail_bead_templates",
                "residue_to_bead_map",
                "bead_connectivity_across_residues",
                "cg_topology_for_chain",
                "aa_to_cg_mapping_provenance",
                "simulation_readiness_json"
            ],
            "simulation_readiness": {
                "default_exclusions_mode": "explicit_all_intra",
                "dihedrals_default_enabled": false,
                "coordinate_unwrap_default": true,
                "reports": ["explicit exclusion policy", "dihedral emission policy", "max bonded distance before/after CG PDB unwrap"]
            }
        },
        "validation_checks": [
            "source files exist",
            "coordinate atom count can be read",
            "topology atom count can be read when supported",
            "coordinate/topology atom counts match",
            "target_selection is evaluated when a topology is available",
            "template-mode replay preconditions are reported",
            "bonded stats path preconditions are reported",
            "xtb executable availability is reported when requested",
            "optimization runtime/cost estimate is reported"
        ],
        "examples": example_requests(),
        "outputs": [
            "martini_bead_mapping_json",
            "mapping_template_json",
            "backmap_plan_json",
            "coarse_grained_pdb",
            "optional_coarse_grained_trajectory_xtc_dcd_gro_g96_cpt_trr_h5md",
            "reference_targets_json",
            "bond_stats_json",
            "bonded_stats_json",
            "bonded_parameter_map_json",
            "bonded_optimization_report",
            "aa_to_cg_mapping_provenance",
            "simulation_readiness_json",
            "martini_topology_itp",
            "martini_topology_top"
        ],
        "force_field": "Martini 3 coarse-grained bead assignment",
        "backmapping": crate::backmap_contract::capabilities()
    })
}
