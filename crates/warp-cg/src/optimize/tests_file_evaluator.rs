use super::*;
use crate::reference::{
    ReferenceBinConfig, ReferenceDistributionTarget, ReferenceTargetSet, ReferenceTermKind,
};
use crate::trajectory::{BeadMapping, NativeTrajectoryOptions};

#[test]
fn json_file_evaluator_runs_command_and_reads_result() {
    let tempdir = tempfile::tempdir().unwrap();
    let runner = tempdir.path().join("runner.sh");
    std::fs::write(
        &runner,
        concat!(
            "#!/usr/bin/env bash\n",
            "set -euo pipefail\n",
            "python3 - <<'PY'\n",
            "import json, os\n",
            "with open(os.environ['WARP_CG_CANDIDATE_JSON']) as fp:\n",
            "    req = json.load(fp)\n",
            "params = req['candidate']['named_parameters']\n",
            "objective = sum(p['value'] * p['value'] for p in params)\n",
            "with open(os.environ['WARP_CG_RESULT_JSON'], 'w') as fp:\n",
            "    json.dump({\n",
            "        'schema_version': 'warp-cg.objective-result.v1',\n",
            "        'status': 'completed',\n",
            "        'objective': objective,\n",
            "        'metrics': {'runner.parameter_count': len(params)}\n",
            "    }, fp)\n",
            "PY\n",
        ),
    )
    .unwrap();
    make_executable(&runner);

    let mut evaluator = JsonFileObjectiveEvaluator::new(JsonFileEvaluatorConfig {
        work_dir: tempdir.path().join("runs"),
        request_filename: "candidate.json".to_string(),
        result_filename: "result.json".to_string(),
        command: Some(JsonFileEvaluatorCommand {
            program: runner.to_string_lossy().to_string(),
            args: Vec::new(),
        }),
        reference_targets: None,
        reference_metrics: Default::default(),
        metric_scoring: Default::default(),
        force_reference_scoring: false,
        require_candidate_trajectory: false,
        candidate_extraction: None,
    });
    let bounds = vec![ParameterBound {
        name: "sigma".to_string(),
        min: 0.0,
        max: 1.0,
    }];
    let candidate = OptimizationCandidate::from_parameters(&[0.25], &bounds);

    let evaluation = evaluator.evaluate_candidate(&candidate);

    assert_eq!(evaluation.status, EvaluationStatus::Completed);
    assert_eq!(evaluation.objective, Some(0.0625));
    assert_eq!(evaluation.metrics["runner.parameter_count"], 1.0);
    let request_json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(tempdir.path().join("runs/evaluation_000000/candidate.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        request_json["schema_version"],
        JSON_OBJECTIVE_REQUEST_SCHEMA
    );
    assert_eq!(
        request_json["candidate"]["named_parameters"][0]["name"],
        "sigma"
    );
    assert_eq!(
        request_json["candidate"]["normalized_parameters"][0],
        serde_json::json!(0.25)
    );
}

#[test]
fn json_file_evaluator_returns_failed_status_from_result() {
    let tempdir = tempfile::tempdir().unwrap();
    let runner = tempdir.path().join("runner.sh");
    std::fs::write(
        &runner,
        concat!(
            "#!/usr/bin/env bash\n",
            "set -euo pipefail\n",
            "cat > \"$WARP_CG_RESULT_JSON\" <<'JSON'\n",
            "{\"schema_version\":\"warp-cg.objective-result.v1\",",
            "\"status\":\"invalid_parameters\",",
            "\"reason\":\"sigma out of supported range\"}\n",
            "JSON\n",
        ),
    )
    .unwrap();
    make_executable(&runner);

    let mut evaluator = JsonFileObjectiveEvaluator::new(JsonFileEvaluatorConfig {
        work_dir: tempdir.path().join("runs"),
        request_filename: "candidate.json".to_string(),
        result_filename: "result.json".to_string(),
        command: Some(JsonFileEvaluatorCommand {
            program: runner.to_string_lossy().to_string(),
            args: Vec::new(),
        }),
        reference_targets: None,
        reference_metrics: Default::default(),
        metric_scoring: Default::default(),
        force_reference_scoring: false,
        require_candidate_trajectory: false,
        candidate_extraction: None,
    });
    let candidate = OptimizationCandidate::from_parameters(
        &[2.0],
        &[ParameterBound {
            name: "sigma".to_string(),
            min: 0.0,
            max: 1.0,
        }],
    );

    let evaluation = evaluator.evaluate_candidate(&candidate);

    assert!(matches!(
        evaluation.status,
        EvaluationStatus::InvalidParameters { .. }
    ));
    assert_eq!(evaluation.objective, None);
}

#[test]
fn json_file_evaluator_can_drive_shared_optimizer_entrypoint() {
    let tempdir = tempfile::tempdir().unwrap();
    let runner = tempdir.path().join("runner.sh");
    std::fs::write(
        &runner,
        concat!(
            "#!/usr/bin/env bash\n",
            "set -euo pipefail\n",
            "python3 - <<'PY'\n",
            "import json, os\n",
            "with open(os.environ['WARP_CG_CANDIDATE_JSON']) as fp:\n",
            "    req = json.load(fp)\n",
            "x = req['candidate']['named_parameters'][0]['value']\n",
            "with open(os.environ['WARP_CG_RESULT_JSON'], 'w') as fp:\n",
            "    json.dump({'objective': (x - 0.2) ** 2, 'metrics': {'x': x}}, fp)\n",
            "PY\n",
        ),
    )
    .unwrap();
    make_executable(&runner);

    let mut evaluator = JsonFileObjectiveEvaluator::new(JsonFileEvaluatorConfig {
        work_dir: tempdir.path().join("runs"),
        request_filename: "candidate.json".to_string(),
        result_filename: "result.json".to_string(),
        command: Some(JsonFileEvaluatorCommand {
            program: runner.to_string_lossy().to_string(),
            args: Vec::new(),
        }),
        reference_targets: None,
        reference_metrics: Default::default(),
        metric_scoring: Default::default(),
        force_reference_scoring: false,
        require_candidate_trajectory: false,
        candidate_extraction: None,
    });
    let report = optimize_with_named_evaluator(
        &[ParameterBound {
            name: "epsilon".to_string(),
            min: 0.0,
            max: 1.0,
        }],
        &mut evaluator,
        &OptimizationConfig {
            method: "pso".to_string(),
            objective: "json_file_runner".to_string(),
            max_evaluations: 4,
            seed: 3,
            swarm_size: Some(2),
            pso: Some(PsoConfig {
                fuzzy_self_tuning: Some(false),
                max_iterations_without_global_best: Some(100),
                ..PsoConfig::default()
            }),
            bo: None,
            initial_parameters: std::collections::BTreeMap::new(),
        },
        &[],
    );

    assert_eq!(report.status, "ok");
    assert_eq!(report.evaluations.len(), 4);
    assert!(tempdir
        .path()
        .join("runs/evaluation_000003/result.json")
        .exists());
}

#[test]
fn json_file_evaluator_scores_candidate_targets_against_reference() {
    let tempdir = tempfile::tempdir().unwrap();
    let runner = tempdir.path().join("runner.sh");
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
            "candidate['bonds'][0]['mean'] = 0.55\n",
            "candidate['bonds'][0]['probabilities'] = [0.0, 1.0]\n",
            "with open(os.environ['WARP_CG_RESULT_JSON'], 'w') as fp:\n",
            "    json.dump({\n",
            "        'schema_version': 'warp-cg.objective-result.v1',\n",
            "        'status': 'completed',\n",
            "        'metrics': {'runner.frames': 10},\n",
            "        'candidate_targets': candidate\n",
            "    }, fp)\n",
            "PY\n",
        ),
    )
    .unwrap();
    make_executable(&runner);
    let reference_targets = single_bond_target_set();
    let mut evaluator = JsonFileObjectiveEvaluator::new(JsonFileEvaluatorConfig {
        work_dir: tempdir.path().join("runs"),
        request_filename: "candidate.json".to_string(),
        result_filename: "result.json".to_string(),
        command: Some(JsonFileEvaluatorCommand {
            program: runner.to_string_lossy().to_string(),
            args: Vec::new(),
        }),
        reference_targets: Some(reference_targets),
        reference_metrics: Default::default(),
        metric_scoring: Default::default(),
        force_reference_scoring: false,
        require_candidate_trajectory: false,
        candidate_extraction: None,
    });
    let candidate = OptimizationCandidate::from_parameters(
        &[0.55],
        &[ParameterBound {
            name: "bond:B1".to_string(),
            min: 0.0,
            max: 1.0,
        }],
    );

    let evaluation = evaluator.evaluate_candidate(&candidate);

    assert_eq!(evaluation.status, EvaluationStatus::Completed);
    assert!(evaluation.objective.unwrap() > 0.0);
    assert_eq!(evaluation.metrics["runner.frames"], 10.0);
    assert!(evaluation.metrics.contains_key("bonds_emd"));
    let request_json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(tempdir.path().join("runs/evaluation_000000/candidate.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(request_json["reference_targets"]["bonds"][0]["mean"], 0.47);
}

#[test]
fn json_file_evaluator_adds_rg_sasa_metric_objective() {
    let tempdir = tempfile::tempdir().unwrap();
    let runner = tempdir.path().join("runner.sh");
    std::fs::write(
        &runner,
        concat!(
            "#!/usr/bin/env bash\n",
            "set -euo pipefail\n",
            "python3 - <<'PY'\n",
            "import json, os\n",
            "with open(os.environ['WARP_CG_CANDIDATE_JSON']) as fp:\n",
            "    req = json.load(fp)\n",
            "with open(os.environ['WARP_CG_RESULT_JSON'], 'w') as fp:\n",
            "    json.dump({\n",
            "        'schema_version': 'warp-cg.objective-result.v1',\n",
            "        'status': 'completed',\n",
            "        'metrics': {'rg_mean_nm': 1.2, 'sasa_approx_mean_nm2': 11.0},\n",
            "        'candidate_targets': req['reference_targets']\n",
            "    }, fp)\n",
            "PY\n",
        ),
    )
    .unwrap();
    make_executable(&runner);
    let mut evaluator = JsonFileObjectiveEvaluator::new(JsonFileEvaluatorConfig {
        work_dir: tempdir.path().join("runs"),
        request_filename: "candidate.json".to_string(),
        result_filename: "result.json".to_string(),
        command: Some(JsonFileEvaluatorCommand {
            program: runner.to_string_lossy().to_string(),
            args: Vec::new(),
        }),
        reference_targets: Some(single_bond_target_set()),
        reference_metrics: std::collections::BTreeMap::from([
            ("rg_mean_nm".to_string(), 1.0),
            ("rg_std_nm".to_string(), 0.1),
            ("sasa_approx_mean_nm2".to_string(), 10.0),
            ("sasa_approx_std_nm2".to_string(), 2.0),
        ]),
        metric_scoring: StructuralMetricScoringConfig {
            rg_weight: 2.0,
            sasa_weight: 0.5,
            ..StructuralMetricScoringConfig::default()
        },
        force_reference_scoring: false,
        require_candidate_trajectory: false,
        candidate_extraction: None,
    });
    let candidate = OptimizationCandidate::from_parameters(
        &[0.47],
        &[ParameterBound {
            name: "bond:B1".to_string(),
            min: 0.0,
            max: 1.0,
        }],
    );

    let evaluation = evaluator.evaluate_candidate(&candidate);

    assert_eq!(evaluation.status, EvaluationStatus::Completed);
    assert!((evaluation.objective.unwrap() - 8.125).abs() < 1.0e-12);
    assert!((evaluation.metrics["rg_objective"] - 8.0).abs() < 1.0e-12);
    assert!((evaluation.metrics["sasa_objective"] - 0.125).abs() < 1.0e-12);
    assert!((evaluation.metrics["structural_metric_objective"] - 8.125).abs() < 1.0e-12);
    assert!(evaluation.metrics["bonded_emd_objective"].abs() < 1.0e-12);
}

#[test]
fn json_file_evaluator_penalizes_missing_sasa_metric() {
    let tempdir = tempfile::tempdir().unwrap();
    let runner = tempdir.path().join("runner.sh");
    std::fs::write(
        &runner,
        concat!(
            "#!/usr/bin/env bash\n",
            "set -euo pipefail\n",
            "python3 - <<'PY'\n",
            "import json, os\n",
            "with open(os.environ['WARP_CG_CANDIDATE_JSON']) as fp:\n",
            "    req = json.load(fp)\n",
            "with open(os.environ['WARP_CG_RESULT_JSON'], 'w') as fp:\n",
            "    json.dump({\n",
            "        'schema_version': 'warp-cg.objective-result.v1',\n",
            "        'status': 'completed',\n",
            "        'candidate_targets': req['reference_targets']\n",
            "    }, fp)\n",
            "PY\n",
        ),
    )
    .unwrap();
    make_executable(&runner);
    let mut evaluator = JsonFileObjectiveEvaluator::new(JsonFileEvaluatorConfig {
        work_dir: tempdir.path().join("runs"),
        request_filename: "candidate.json".to_string(),
        result_filename: "result.json".to_string(),
        command: Some(JsonFileEvaluatorCommand {
            program: runner.to_string_lossy().to_string(),
            args: Vec::new(),
        }),
        reference_targets: Some(single_bond_target_set()),
        reference_metrics: std::collections::BTreeMap::from([(
            "sasa_approx_mean_nm2".to_string(),
            10.0,
        )]),
        metric_scoring: Default::default(),
        force_reference_scoring: false,
        require_candidate_trajectory: false,
        candidate_extraction: None,
    });
    let candidate = OptimizationCandidate::from_parameters(
        &[0.47],
        &[ParameterBound {
            name: "bond:B1".to_string(),
            min: 0.0,
            max: 1.0,
        }],
    );

    let evaluation = evaluator.evaluate_candidate(&candidate);

    assert_eq!(evaluation.status, EvaluationStatus::Completed);
    assert!(evaluation.objective.unwrap() >= 1.0e6);
    assert_eq!(evaluation.metrics["sasa_missing_penalty"], 1.0e6);
}

#[test]
fn json_file_evaluator_extracts_candidate_targets_from_runner_trajectory() {
    let tempdir = tempfile::tempdir().unwrap();
    let runner = tempdir.path().join("runner.sh");
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
    let mapping = BeadMapping {
        bead_names: vec!["B0".to_string(), "B1".to_string()],
        atom_indices: vec![vec![0], vec![1]],
    };
    let mut evaluator = JsonFileObjectiveEvaluator::new(JsonFileEvaluatorConfig {
        work_dir: tempdir.path().join("runs"),
        request_filename: "candidate.json".to_string(),
        result_filename: "result.json".to_string(),
        command: Some(JsonFileEvaluatorCommand {
            program: runner.to_string_lossy().to_string(),
            args: Vec::new(),
        }),
        reference_targets: Some(default_binned_single_bond_target_set()),
        reference_metrics: Default::default(),
        metric_scoring: Default::default(),
        force_reference_scoring: false,
        require_candidate_trajectory: false,
        candidate_extraction: Some(CandidateTrajectoryExtractionConfig {
            mapping,
            connections: vec![(0, 1)],
            term_set: None,
            options: NativeTrajectoryOptions::default(),
            transform: None,
            mapped_trajectory_name: Some("candidate_mapped.gro".to_string()),
        }),
    });
    let candidate = OptimizationCandidate::from_parameters(
        &[0.75],
        &[ParameterBound {
            name: "bond:B1".to_string(),
            min: 0.0,
            max: 1.0,
        }],
    );

    let evaluation = evaluator.evaluate_candidate(&candidate);

    assert_eq!(evaluation.status, EvaluationStatus::Completed);
    assert!(evaluation.objective.unwrap() > 0.0);
    assert_eq!(evaluation.metrics["runner.frames"], 2.0);
    assert_eq!(evaluation.metrics["candidate_trajectory.rg_samples"], 2.0);
    assert!(evaluation.metrics.contains_key("bonds_emd"));
    assert!(tempdir
        .path()
        .join("runs/evaluation_000000/candidate_mapped.gro")
        .is_file());
    assert!(tempdir
        .path()
        .join("runs/evaluation_000000/candidate_reference_targets.json")
        .is_file());
}

#[test]
fn json_file_evaluator_force_scores_candidate_trajectory() {
    let tempdir = tempfile::tempdir().unwrap();
    let runner = tempdir.path().join("runner.sh");
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
            "\"objective\":999.0,",
            "\"candidate_trajectory\":{\"path\":\"candidate.xyz\"}}\n",
            "JSON\n",
        ),
    )
    .unwrap();
    make_executable(&runner);
    let mapping = BeadMapping {
        bead_names: vec!["B0".to_string(), "B1".to_string()],
        atom_indices: vec![vec![0], vec![1]],
    };
    let mut evaluator = JsonFileObjectiveEvaluator::new(JsonFileEvaluatorConfig {
        work_dir: tempdir.path().join("runs"),
        request_filename: "candidate.json".to_string(),
        result_filename: "result.json".to_string(),
        command: Some(JsonFileEvaluatorCommand {
            program: runner.to_string_lossy().to_string(),
            args: Vec::new(),
        }),
        reference_targets: Some(default_binned_single_bond_target_set()),
        reference_metrics: Default::default(),
        metric_scoring: Default::default(),
        force_reference_scoring: true,
        require_candidate_trajectory: true,
        candidate_extraction: Some(CandidateTrajectoryExtractionConfig {
            mapping,
            connections: vec![(0, 1)],
            term_set: None,
            options: NativeTrajectoryOptions::default(),
            transform: None,
            mapped_trajectory_name: None,
        }),
    });
    let candidate = OptimizationCandidate::from_parameters(
        &[0.75],
        &[ParameterBound {
            name: "bond:B1".to_string(),
            min: 0.0,
            max: 1.0,
        }],
    );

    let evaluation = evaluator.evaluate_candidate(&candidate);

    assert_eq!(evaluation.status, EvaluationStatus::Completed);
    assert_ne!(evaluation.objective, Some(999.0));
    assert!(evaluation
        .metrics
        .contains_key("candidate_trajectory.rg_samples"));
    assert!(evaluation.metrics.contains_key("bonds_emd"));
}

#[test]
fn json_file_evaluator_rejects_candidate_targets_only_when_trajectory_required() {
    let tempdir = tempfile::tempdir().unwrap();
    let runner = tempdir.path().join("runner.sh");
    std::fs::write(
        &runner,
        concat!(
            "#!/usr/bin/env bash\n",
            "set -euo pipefail\n",
            "python3 - <<'PY'\n",
            "import json, os\n",
            "with open(os.environ['WARP_CG_CANDIDATE_JSON']) as fp:\n",
            "    req = json.load(fp)\n",
            "with open(os.environ['WARP_CG_RESULT_JSON'], 'w') as fp:\n",
            "    json.dump({\n",
            "        'schema_version': 'warp-cg.objective-result.v1',\n",
            "        'status': 'completed',\n",
            "        'candidate_targets': req['reference_targets']\n",
            "    }, fp)\n",
            "PY\n",
        ),
    )
    .unwrap();
    make_executable(&runner);
    let mut evaluator = JsonFileObjectiveEvaluator::new(JsonFileEvaluatorConfig {
        work_dir: tempdir.path().join("runs"),
        request_filename: "candidate.json".to_string(),
        result_filename: "result.json".to_string(),
        command: Some(JsonFileEvaluatorCommand {
            program: runner.to_string_lossy().to_string(),
            args: Vec::new(),
        }),
        reference_targets: Some(single_bond_target_set()),
        reference_metrics: Default::default(),
        metric_scoring: Default::default(),
        force_reference_scoring: true,
        require_candidate_trajectory: true,
        candidate_extraction: None,
    });
    let candidate = OptimizationCandidate::from_parameters(
        &[0.47],
        &[ParameterBound {
            name: "bond:B1".to_string(),
            min: 0.0,
            max: 1.0,
        }],
    );

    let evaluation = evaluator.evaluate_candidate(&candidate);

    assert!(matches!(
        evaluation.status,
        EvaluationStatus::FailedExtraction { .. }
    ));
    let EvaluationStatus::FailedExtraction { reason } = evaluation.status else {
        unreachable!();
    };
    assert!(reason.contains("candidate_trajectory"));
}

#[test]
fn json_file_evaluator_rejects_missing_candidate_target_terms() {
    let tempdir = tempfile::tempdir().unwrap();
    let runner = tempdir.path().join("runner.sh");
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
            "candidate['bonds'] = []\n",
            "with open(os.environ['WARP_CG_RESULT_JSON'], 'w') as fp:\n",
            "    json.dump({\n",
            "        'schema_version': 'warp-cg.objective-result.v1',\n",
            "        'status': 'completed',\n",
            "        'candidate_targets': candidate\n",
            "    }, fp)\n",
            "PY\n",
        ),
    )
    .unwrap();
    make_executable(&runner);

    let evaluation = evaluate_single_bond_runner(tempdir.path(), &runner);

    assert!(matches!(
        evaluation.status,
        EvaluationStatus::FailedExtraction { .. }
    ));
    assert_eq!(evaluation.objective, None);
    let EvaluationStatus::FailedExtraction { reason } = evaluation.status else {
        unreachable!();
    };
    assert!(reason.contains("candidate_targets.bonds length mismatch"));
}

#[test]
fn json_file_evaluator_rejects_malformed_candidate_target_bins() {
    let tempdir = tempfile::tempdir().unwrap();
    let runner = tempdir.path().join("runner.sh");
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
            "candidate['bonds'][0]['bin_edges'] = [0.0, 0.25, 0.5, 1.0]\n",
            "candidate['bonds'][0]['probabilities'] = [0.5, 0.5, 0.0]\n",
            "with open(os.environ['WARP_CG_RESULT_JSON'], 'w') as fp:\n",
            "    json.dump({\n",
            "        'schema_version': 'warp-cg.objective-result.v1',\n",
            "        'status': 'completed',\n",
            "        'objective': 1.23,\n",
            "        'candidate_targets': candidate\n",
            "    }, fp)\n",
            "PY\n",
        ),
    )
    .unwrap();
    make_executable(&runner);

    let evaluation = evaluate_single_bond_runner(tempdir.path(), &runner);

    assert!(matches!(
        evaluation.status,
        EvaluationStatus::FailedExtraction { .. }
    ));
    assert_eq!(evaluation.objective, None);
    let EvaluationStatus::FailedExtraction { reason } = evaluation.status else {
        unreachable!();
    };
    assert!(reason.contains("candidate_targets.bonds[0].bin_edges length mismatch"));
}

fn evaluate_single_bond_runner(
    base_dir: &std::path::Path,
    runner: &std::path::Path,
) -> ObjectiveEvaluation {
    let mut evaluator = JsonFileObjectiveEvaluator::new(JsonFileEvaluatorConfig {
        work_dir: base_dir.join("runs"),
        request_filename: "candidate.json".to_string(),
        result_filename: "result.json".to_string(),
        command: Some(JsonFileEvaluatorCommand {
            program: runner.to_string_lossy().to_string(),
            args: Vec::new(),
        }),
        reference_targets: Some(single_bond_target_set()),
        reference_metrics: Default::default(),
        metric_scoring: Default::default(),
        force_reference_scoring: false,
        require_candidate_trajectory: false,
        candidate_extraction: None,
    });
    let candidate = OptimizationCandidate::from_parameters(
        &[0.55],
        &[ParameterBound {
            name: "bond:B1".to_string(),
            min: 0.0,
            max: 1.0,
        }],
    );
    evaluator.evaluate_candidate(&candidate)
}

fn default_binned_single_bond_target_set() -> ReferenceTargetSet {
    let bin_config = ReferenceBinConfig::default();
    ReferenceTargetSet {
        version: 1,
        constraints: Vec::new(),
        bonds: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Bond,
            Some("bond group 1".to_string()),
            vec![0, 1],
            vec![vec![0, 1]],
            &[0.25, 0.25],
            "nm",
            false,
            0.0,
            bin_config.bonded_max_range_nm,
            bin_config.bond_bin_width_nm,
        )],
        angles: Vec::new(),
        dihedrals: Vec::new(),
        bin_config,
    }
}

fn single_bond_target_set() -> ReferenceTargetSet {
    ReferenceTargetSet {
        version: 1,
        bin_config: ReferenceBinConfig::default(),
        constraints: Vec::new(),
        bonds: vec![ReferenceDistributionTarget {
            kind: ReferenceTermKind::Bond,
            label: Some("bond group 1".to_string()),
            beads: vec![0, 1],
            members: vec![vec![0, 1]],
            units: "nm".to_string(),
            periodic: false,
            mean: 0.47,
            std: 0.02,
            samples: 8,
            domain: [0.0, 1.0],
            bin_edges: vec![0.0, 0.5, 1.0],
            probabilities: vec![1.0, 0.0],
        }],
        angles: Vec::new(),
        dihedrals: Vec::new(),
    }
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
