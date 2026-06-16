use super::*;

#[test]
fn precomputed_reference_targets_run_without_trajectory_source() {
    let tmp = tempfile::tempdir().unwrap();
    let target_path = tmp.path().join("precomputed_targets.json");
    let metric_path = tmp.path().join("precomputed_metrics.json");
    std::fs::write(
        &target_path,
        serde_json::to_vec_pretty(&json!({
            "version": 1,
            "bin_config": {
                "bond_bin_width_nm": 0.01,
                "angle_bin_width_deg": 1.0,
                "dihedral_bin_width_deg": 1.0,
                "bonded_max_range_nm": 3.0
            },
            "constraints": [],
            "bonds": [{
                "kind": "bond",
                "label": "bond group 1",
                "beads": [0, 1],
                "members": [[0, 1]],
                "units": "nm",
                "periodic": false,
                "mean": 0.47,
                "std": 0.02,
                "samples": 8,
                "domain": [0.0, 3.0],
                "bin_edges": [0.0, 1.0],
                "probabilities": [1.0]
            }],
            "angles": [],
            "dihedrals": []
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &metric_path,
        serde_json::json!({"metrics": {"rg_mean_nm": 1.3}}).to_string(),
    )
    .unwrap();
    let request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "precomputed_reference".to_string(),
        smiles: Some("CC".to_string()),
        repeat_smiles: None,
        source: None,
        bonding: None,
        chemistry_hints: Vec::new(),
        chemistry_policy: None,
        polymer: None,
        mapping: None,
        topology: None,
        trajectory_source: None,
        reference_source: Some(ReferenceSource {
            kind: "precomputed".to_string(),
            xtb: None,
            precomputed: Some(PrecomputedReferenceRequest {
                source_kind: Some("cached_grouped_bonded".to_string()),
                target_set: target_path.to_string_lossy().to_string(),
            }),
            bonded_terms: None,
            metrics: vec![ReferenceMetricSourceRequest {
                kind: "json".to_string(),
                path: metric_path.to_string_lossy().to_string(),
                namespace: None,
                artifact_kind: None,
            }],
            transform: None,
        }),
        forcefield: None,
        optimization: None,
        output: CgOutputRequest {
            out_dir: tmp.path().to_string_lossy().to_string(),
            mapped_trajectory: None,
            write_mapping_json: false,
            write_topology_itp: false,
            write_topology_top: false,
            write_cg_pdb: false,
            cg_pdb: None,
            write_bonded_parameter_map: false,
        },
    };

    let result = run_request(&request, Instant::now()).unwrap();

    let reference = result.reference.as_ref().unwrap();
    assert_eq!(reference.source_kind, "cached_grouped_bonded");
    assert!(reference.target_set_available);
    assert_eq!(reference.metadata.mapped_by, "precomputed_stats");
    assert_eq!(reference.metrics["rg_mean_nm"], 1.3);
    assert!(result.artifact_paths.contains_key("reference_targets_json"));
    assert!(result.artifact_paths.contains_key("reference_metrics_json"));
    assert!(result.artifact_paths.contains_key("bond_stats_json"));
}

#[test]
fn optimization_rejects_single_sample_reference_when_strict() {
    let tmp = tempfile::tempdir().unwrap();
    let target_path = tmp.path().join("single_sample_targets.json");
    std::fs::write(
        &target_path,
        serde_json::to_vec_pretty(&json!({
            "version": 1,
            "bin_config": {
                "bond_bin_width_nm": 0.01,
                "angle_bin_width_deg": 1.0,
                "dihedral_bin_width_deg": 1.0,
                "bonded_max_range_nm": 3.0
            },
            "constraints": [],
            "bonds": [{
                "kind": "bond",
                "label": "single frame bond",
                "beads": [0, 1],
                "members": [[0, 1]],
                "units": "angstrom",
                "periodic": false,
                "mean": 4.7,
                "std": 0.0,
                "samples": 1,
                "domain": [0.0, 30.0],
                "bin_edges": [0.0, 1.0],
                "probabilities": [1.0]
            }],
            "angles": [],
            "dihedrals": []
        }))
        .unwrap(),
    )
    .unwrap();
    let request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "strict_single_sample".to_string(),
        smiles: Some("CC".to_string()),
        repeat_smiles: None,
        source: None,
        bonding: None,
        chemistry_hints: Vec::new(),
        chemistry_policy: None,
        polymer: None,
        mapping: None,
        topology: None,
        trajectory_source: None,
        reference_source: Some(ReferenceSource {
            kind: "precomputed".to_string(),
            xtb: None,
            precomputed: Some(PrecomputedReferenceRequest {
                source_kind: Some("cached_grouped_bonded".to_string()),
                target_set: target_path.to_string_lossy().to_string(),
            }),
            bonded_terms: None,
            metrics: Vec::new(),
            transform: None,
        }),
        forcefield: None,
        optimization: Some(ParameterTuningRequest {
            enabled: true,
            source: "external_trajectory".to_string(),
            method: "pso".to_string(),
            fitting_mode: Some("distribution_fit".to_string()),
            allow_single_frame: Some(false),
            min_samples_per_term: Some(2),
            on_insufficient_samples: Some("error".to_string()),
            max_evaluations: Some(2),
            seed: Some(7),
            initial_parameters: std::collections::BTreeMap::new(),
            swarm_size: Some(2),
            pso: None,
            bo: None,
            objective: "bonded_parameter_parity".to_string(),
            target_terms: None,
            xtb: None,
            metric_scoring: None,
            evaluator: None,
            runner: None,
        }),
        output: CgOutputRequest {
            out_dir: tmp.path().to_string_lossy().to_string(),
            mapped_trajectory: None,
            write_mapping_json: false,
            write_topology_itp: false,
            write_topology_top: false,
            write_cg_pdb: false,
            cg_pdb: None,
            write_bonded_parameter_map: false,
        },
    };

    let err = run_request(&request, Instant::now()).unwrap_err();
    assert!(err.to_string().contains("fewer than 2 samples"));
}

#[test]
fn precomputed_reference_targets_can_use_json_file_runner_evaluator() {
    let tmp = tempfile::tempdir().unwrap();
    let target_path = tmp.path().join("precomputed_targets.json");
    let runner = tmp.path().join("runner.sh");
    std::fs::write(
        &target_path,
        serde_json::to_vec_pretty(&json!({
            "version": 1,
            "bin_config": {
                "bond_bin_width_nm": 0.01,
                "angle_bin_width_deg": 1.0,
                "dihedral_bin_width_deg": 1.0,
                "bonded_max_range_nm": 3.0
            },
            "constraints": [],
            "bonds": [{
                "kind": "bond",
                "label": "bond group 1",
                "beads": [0, 1],
                "members": [[0, 1]],
                "units": "nm",
                "periodic": false,
                "mean": 0.47,
                "std": 0.02,
                "samples": 8,
                "domain": [0.0, 1.0],
                "bin_edges": [0.0, 0.5, 1.0],
                "probabilities": [1.0, 0.0]
            }],
            "angles": [],
            "dihedrals": []
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &runner,
        concat!(
            "#!/usr/bin/env bash\n",
            "set -euo pipefail\n",
            "python3 - <<'PY'\n",
            "import json, os\n",
            "with open(os.environ['WARP_CG_CANDIDATE_JSON']) as fp:\n",
            "    req = json.load(fp)\n",
            "candidate = req['reference_targets']\n",
            "candidate['bonds'][0]['mean'] = req['candidate']['parameters'][0]\n",
            "candidate['bonds'][0]['probabilities'] = [0.0, 1.0]\n",
            "with open(os.environ['WARP_CG_RESULT_JSON'], 'w') as fp:\n",
            "    json.dump({\n",
            "        'schema_version': 'warp-cg.objective-result.v1',\n",
            "        'status': 'completed',\n",
            "        'metrics': {'runner.frames': 4},\n",
            "        'candidate_targets': candidate\n",
            "    }, fp)\n",
            "PY\n",
        ),
    )
    .unwrap();
    make_executable(&runner);
    let request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "runner_reference".to_string(),
        smiles: Some("CC".to_string()),
        repeat_smiles: None,
        source: None,
        bonding: None,
        chemistry_hints: Vec::new(),
        chemistry_policy: None,
        polymer: None,
        mapping: None,
        topology: None,
        trajectory_source: None,
        reference_source: Some(ReferenceSource {
            kind: "precomputed".to_string(),
            xtb: None,
            precomputed: Some(PrecomputedReferenceRequest {
                source_kind: Some("cached_grouped_bonded".to_string()),
                target_set: target_path.to_string_lossy().to_string(),
            }),
            bonded_terms: None,
            metrics: Vec::new(),
            transform: None,
        }),
        forcefield: None,
        optimization: Some(ParameterTuningRequest {
            enabled: true,
            source: "external_trajectory".to_string(),
            method: "pso".to_string(),
            fitting_mode: Some("external_evaluator".to_string()),
            allow_single_frame: None,
            min_samples_per_term: None,
            on_insufficient_samples: None,
            max_evaluations: Some(2),
            seed: Some(7),
            initial_parameters: std::collections::BTreeMap::new(),
            swarm_size: Some(2),
            pso: None,
            bo: None,
            objective: "external_candidate_targets".to_string(),
            target_terms: None,
            xtb: None,
            metric_scoring: None,
            evaluator: Some(ObjectiveEvaluatorRequest {
                kind: "json_file".to_string(),
                json_file: Some(JsonFileEvaluatorRequest {
                    work_dir: "runner_evaluations".to_string(),
                    request_filename: None,
                    result_filename: None,
                    command: Some(JsonFileEvaluatorCommandRequest {
                        program: runner.to_string_lossy().to_string(),
                        args: Vec::new(),
                    }),
                    candidate_extraction: None,
                }),
            }),
            runner: None,
        }),
        output: CgOutputRequest {
            out_dir: tmp.path().to_string_lossy().to_string(),
            mapped_trajectory: None,
            write_mapping_json: false,
            write_topology_itp: false,
            write_topology_top: false,
            write_cg_pdb: false,
            cg_pdb: None,
            write_bonded_parameter_map: false,
        },
    };

    let result = run_request(&request, Instant::now()).unwrap();
    let report = result.optimization.unwrap().report.unwrap();

    assert_eq!(report.objective, "external_candidate_targets");
    assert_eq!(report.evaluations.len(), 2);
    assert!(report
        .evaluations
        .iter()
        .all(|record| record.metrics["runner.frames"] == 4.0));
    assert!(report
        .evaluations
        .iter()
        .all(|record| record.metrics.contains_key("bonds_emd")));
    assert!(tmp
        .path()
        .join("runner_evaluations/evaluation_000000/candidate.json")
        .exists());
}

#[test]
fn json_file_runner_can_return_candidate_trajectory_for_agent_extraction() {
    let tmp = tempfile::tempdir().unwrap();
    let target_path = tmp.path().join("precomputed_targets.json");
    let runner = tmp.path().join("runner.sh");
    let bin_edges = (0..=300).map(|idx| idx as f64 * 0.01).collect::<Vec<_>>();
    let mut probabilities = vec![0.0; 300];
    probabilities[25] = 1.0;
    std::fs::write(
        &target_path,
        serde_json::to_vec_pretty(&json!({
            "version": 1,
            "bin_config": {
                "bond_bin_width_nm": 0.01,
                "angle_bin_width_deg": 1.0,
                "dihedral_bin_width_deg": 1.0,
                "bonded_max_range_nm": 3.0
            },
            "constraints": [],
            "bonds": [{
                "kind": "bond",
                "label": "bond group 1",
                "beads": [0, 1],
                "members": [[0, 1]],
                "units": "nm",
                "periodic": false,
                "mean": 0.25,
                "std": 0.0,
                "samples": 2,
                "domain": [0.25, 0.25],
                "bin_edges": bin_edges,
                "probabilities": probabilities
            }],
            "angles": [],
            "dihedrals": []
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &runner,
        concat!(
            "#!/usr/bin/env bash\n",
            "set -euo pipefail\n",
            "cat > candidate.xyz <<'XYZ'\n",
            "2\n",
            "frame 0\n",
            "C 0.0 0.0 0.0\n",
            "C 0.75 0.0 0.0\n",
            "2\n",
            "frame 1\n",
            "C 0.0 0.0 0.0\n",
            "C 0.75 0.0 0.0\n",
            "XYZ\n",
            "cat > \"$WARP_CG_RESULT_JSON\" <<'JSON'\n",
            "{\"schema_version\":\"warp-cg.objective-result.v1\",",
            "\"status\":\"completed\",",
            "\"metrics\":{\"runner.frames\":2},",
            "\"candidate_trajectory\":{\"path\":\"candidate.xyz\"}}\n",
            "JSON\n",
        ),
    )
    .unwrap();
    make_executable(&runner);
    let request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "runner_candidate_traj".to_string(),
        smiles: Some("CC".to_string()),
        repeat_smiles: None,
        source: None,
        bonding: None,
        chemistry_hints: Vec::new(),
        chemistry_policy: None,
        polymer: None,
        mapping: None,
        topology: None,
        trajectory_source: None,
        reference_source: Some(ReferenceSource {
            kind: "precomputed".to_string(),
            xtb: None,
            precomputed: Some(PrecomputedReferenceRequest {
                source_kind: Some("cached_grouped_bonded".to_string()),
                target_set: target_path.to_string_lossy().to_string(),
            }),
            bonded_terms: None,
            metrics: Vec::new(),
            transform: None,
        }),
        forcefield: None,
        optimization: Some(ParameterTuningRequest {
            enabled: true,
            source: "external_trajectory".to_string(),
            method: "pso".to_string(),
            fitting_mode: Some("simulation_fit".to_string()),
            allow_single_frame: None,
            min_samples_per_term: None,
            on_insufficient_samples: None,
            max_evaluations: Some(2),
            seed: Some(9),
            initial_parameters: std::collections::BTreeMap::new(),
            swarm_size: Some(2),
            pso: None,
            bo: None,
            objective: "candidate_trajectory_targets".to_string(),
            target_terms: None,
            xtb: None,
            metric_scoring: None,
            evaluator: Some(ObjectiveEvaluatorRequest {
                kind: "json_file".to_string(),
                json_file: Some(JsonFileEvaluatorRequest {
                    work_dir: "runner_trajectory_evaluations".to_string(),
                    request_filename: None,
                    result_filename: None,
                    command: Some(JsonFileEvaluatorCommandRequest {
                        program: runner.to_string_lossy().to_string(),
                        args: Vec::new(),
                    }),
                    candidate_extraction: Some(CandidateTrajectoryExtractionRequest {
                        mapping: CandidateTrajectoryMappingRequest {
                            bead_names: vec!["B0".to_string(), "B1".to_string()],
                            atom_indices: vec![vec![0], vec![1]],
                        },
                        connections: vec![[0, 1]],
                        bonded_terms: None,
                        mapped_trajectory_name: Some("candidate_mapped.gro".to_string()),
                        format: None,
                        topology: None,
                        topology_format: None,
                        start: None,
                        stop: None,
                        stride: None,
                        length_scale: None,
                        target_selection: None,
                        atom_indices: None,
                        mass_weighted: None,
                        make_whole: None,
                        chunk_frames: None,
                        sasa: None,
                    }),
                }),
            }),
            runner: None,
        }),
        output: CgOutputRequest {
            out_dir: tmp.path().to_string_lossy().to_string(),
            mapped_trajectory: None,
            write_mapping_json: false,
            write_topology_itp: false,
            write_topology_top: false,
            write_cg_pdb: false,
            cg_pdb: None,
            write_bonded_parameter_map: false,
        },
    };

    let result = run_request(&request, Instant::now()).unwrap();
    let report = result.optimization.unwrap().report.unwrap();

    assert_eq!(report.objective, "candidate_trajectory_targets");
    assert!(report.message.contains("Simulation-backed scoring"));
    assert_eq!(report.evaluations.len(), 2);
    assert!(report.evaluations.iter().all(|record| {
        record.metrics["runner.frames"] == 2.0
            && record.metrics["candidate_trajectory.rg_samples"] == 2.0
            && record.metrics.contains_key("bonds_emd")
    }));
    assert!(tmp
        .path()
        .join("runner_trajectory_evaluations/evaluation_000000/candidate_mapped.gro")
        .exists());
}

#[test]
fn candidate_trajectory_extraction_uses_gromacs_bonded_term_groups() {
    let tmp = tempfile::tempdir().unwrap();
    let target_path = tmp.path().join("grouped_targets.json");
    let terms_path = tmp.path().join("candidate.itp");
    let runner = tmp.path().join("runner.sh");
    std::fs::write(
        &target_path,
        serde_json::to_vec_pretty(&json!({
            "version": 1,
            "bin_config": {
                "bond_bin_width_nm": 0.01,
                "angle_bin_width_deg": 1.0,
                "dihedral_bin_width_deg": 1.0,
                "bonded_max_range_nm": 3.0
            },
            "constraints": [],
            "bonds": [
                grouped_bond_target("bond group 1", [[0, 1], [2, 3]]),
                grouped_bond_target("bond group 2", [[1, 2]])
            ],
            "angles": [],
            "dihedrals": []
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        &terms_path,
        r#"
[ moleculetype ]
  MOL 1

[ atoms ]
  1 P1 1 MOL A 1 0
  2 P2 1 MOL B 2 0
  3 P3 1 MOL C 3 0
  4 P4 1 MOL D 4 0

[ bonds ]
; bond group 1
  1 2 1 0.25 1000
  3 4 1 0.25 1000

; bond group 2
  2 3 1 0.25 1000
"#,
    )
    .unwrap();
    std::fs::write(
        &runner,
        concat!(
            "#!/usr/bin/env bash\n",
            "set -euo pipefail\n",
            "cat > candidate.xyz <<'XYZ'\n",
            "4\n",
            "frame 0\n",
            "C 0.00 0.0 0.0\n",
            "C 0.75 0.0 0.0\n",
            "C 1.50 0.0 0.0\n",
            "C 2.25 0.0 0.0\n",
            "4\n",
            "frame 1\n",
            "C 0.00 0.0 0.0\n",
            "C 0.75 0.0 0.0\n",
            "C 1.50 0.0 0.0\n",
            "C 2.25 0.0 0.0\n",
            "XYZ\n",
            "cat > \"$WARP_CG_RESULT_JSON\" <<'JSON'\n",
            "{\"schema_version\":\"warp-cg.objective-result.v1\",",
            "\"status\":\"completed\",",
            "\"metrics\":{\"runner.frames\":2},",
            "\"candidate_trajectory\":{\"path\":\"candidate.xyz\"}}\n",
            "JSON\n",
        ),
    )
    .unwrap();
    make_executable(&runner);
    let request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "runner_grouped_candidate".to_string(),
        smiles: Some("CCCC".to_string()),
        repeat_smiles: None,
        source: None,
        bonding: None,
        chemistry_hints: Vec::new(),
        chemistry_policy: None,
        polymer: None,
        mapping: None,
        topology: None,
        trajectory_source: None,
        reference_source: Some(ReferenceSource {
            kind: "precomputed".to_string(),
            xtb: None,
            precomputed: Some(PrecomputedReferenceRequest {
                source_kind: Some("cached_grouped_bonded".to_string()),
                target_set: target_path.to_string_lossy().to_string(),
            }),
            bonded_terms: None,
            metrics: Vec::new(),
            transform: None,
        }),
        forcefield: None,
        optimization: Some(ParameterTuningRequest {
            enabled: true,
            source: "external_trajectory".to_string(),
            method: "pso".to_string(),
            fitting_mode: None,
            allow_single_frame: None,
            min_samples_per_term: None,
            on_insufficient_samples: None,
            max_evaluations: Some(2),
            seed: Some(11),
            initial_parameters: std::collections::BTreeMap::new(),
            swarm_size: Some(2),
            pso: None,
            bo: None,
            objective: "grouped_candidate_trajectory_targets".to_string(),
            target_terms: None,
            xtb: None,
            metric_scoring: None,
            evaluator: Some(ObjectiveEvaluatorRequest {
                kind: "json_file".to_string(),
                json_file: Some(JsonFileEvaluatorRequest {
                    work_dir: "runner_grouped_evaluations".to_string(),
                    request_filename: None,
                    result_filename: None,
                    command: Some(JsonFileEvaluatorCommandRequest {
                        program: runner.to_string_lossy().to_string(),
                        args: Vec::new(),
                    }),
                    candidate_extraction: Some(CandidateTrajectoryExtractionRequest {
                        mapping: CandidateTrajectoryMappingRequest {
                            bead_names: vec![
                                "B0".to_string(),
                                "B1".to_string(),
                                "B2".to_string(),
                                "B3".to_string(),
                            ],
                            atom_indices: vec![vec![0], vec![1], vec![2], vec![3]],
                        },
                        connections: Vec::new(),
                        bonded_terms: Some(BondedTermSource {
                            kind: "gromacs_itp".to_string(),
                            path: terms_path.to_string_lossy().to_string(),
                            molecule_type: "MOL".to_string(),
                        }),
                        mapped_trajectory_name: Some("candidate_grouped_mapped.gro".to_string()),
                        format: None,
                        topology: None,
                        topology_format: None,
                        start: None,
                        stop: None,
                        stride: None,
                        length_scale: None,
                        target_selection: None,
                        atom_indices: None,
                        mass_weighted: None,
                        make_whole: None,
                        chunk_frames: None,
                        sasa: None,
                    }),
                }),
            }),
            runner: None,
        }),
        output: CgOutputRequest {
            out_dir: tmp.path().to_string_lossy().to_string(),
            mapped_trajectory: None,
            write_mapping_json: false,
            write_topology_itp: false,
            write_topology_top: false,
            write_cg_pdb: false,
            cg_pdb: None,
            write_bonded_parameter_map: false,
        },
    };

    let result = run_request(&request, Instant::now()).unwrap();
    let report = result.optimization.unwrap().report.unwrap();

    assert_eq!(report.objective, "grouped_candidate_trajectory_targets");
    assert_eq!(report.evaluations.len(), 2);
    assert!(report.evaluations.iter().all(|record| {
        record.metrics["runner.frames"] == 2.0
            && record.metrics["candidate_trajectory.rg_samples"] == 2.0
            && record.metrics["bonds_emd"] > 0.0
    }));
    assert!(tmp
        .path()
        .join("runner_grouped_evaluations/evaluation_000000/candidate_reference_targets.json")
        .exists());
}

fn grouped_bond_target<const N: usize>(label: &str, members: [[usize; 2]; N]) -> serde_json::Value {
    let bin_edges = (0..=300).map(|idx| idx as f64 * 0.01).collect::<Vec<_>>();
    let mut probabilities = vec![0.0; 300];
    probabilities[25] = 1.0;
    json!({
        "kind": "bond",
        "label": label,
        "beads": members[0],
        "members": members.iter().map(|member| vec![member[0], member[1]]).collect::<Vec<_>>(),
        "units": "nm",
        "periodic": false,
        "mean": 0.25,
        "std": 0.0,
        "samples": 2 * N,
        "domain": [0.25, 0.25],
        "bin_edges": bin_edges,
        "probabilities": probabilities
    })
}

#[cfg(unix)]
fn make_executable(path: &std::path::Path) {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = std::fs::metadata(path).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(path, permissions).unwrap();
}

#[cfg(not(unix))]
fn make_executable(_path: &std::path::Path) {}
