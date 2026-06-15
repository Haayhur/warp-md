use super::*;
use crate::parameters::{AngleStats, DihedralStats};

fn sample_stats() -> Vec<BondStats> {
    vec![BondStats {
        bead_i: 0,
        bead_j: 1,
        mean: 3.0,
        std: 0.2,
        samples: 16,
    }]
}

fn sample_bonded_stats() -> BondedStats {
    BondedStats {
        bonds: sample_stats(),
        angles: vec![AngleStats {
            bead_i: 0,
            bead_j: 1,
            bead_k: 2,
            mean_deg: 120.0,
            std_deg: 8.0,
            samples: 16,
        }],
        dihedrals: vec![DihedralStats {
            bead_i: 0,
            bead_j: 1,
            bead_k: 2,
            bead_l: 3,
            mean_deg: 180.0,
            std_deg: 20.0,
            samples: 16,
        }],
    }
}

#[test]
fn bayesian_optimization_uses_expected_improvement_trace() {
    let report = optimize_bonded_parameters(
        &sample_stats(),
        &OptimizationConfig {
            method: "bayesian_optimization".to_string(),
            objective: "bonded_parameter_parity".to_string(),
            max_evaluations: 20,
            seed: 7,
            swarm_size: None,
            pso: None,
            bo: None,
            initial_parameters: std::collections::BTreeMap::new(),
        },
    );

    assert_eq!(report.status, "ok");
    assert_eq!(report.evaluations.len(), 20);
    assert!(report.objective_value.is_finite());
    assert!(
        report.objective_value < report.evaluations[0].objective.unwrap(),
        "BO should improve beyond the first warmup evaluation"
    );
    assert!(report
        .evaluations
        .iter()
        .any(|record| record.phase.as_deref() == Some("gp_log_ei")));
}

#[test]
fn pso_uses_same_bonded_objective() {
    let report = optimize_bonded_parameters(
        &sample_stats(),
        &OptimizationConfig {
            method: "pso".to_string(),
            objective: "bonded_parameter_parity".to_string(),
            max_evaluations: 20,
            seed: 11,
            swarm_size: Some(6),
            pso: None,
            bo: None,
            initial_parameters: std::collections::BTreeMap::new(),
        },
    );

    assert_eq!(report.status, "ok");
    assert!(!report.evaluations.is_empty());
    assert_eq!(report.best_parameters.len(), report.bounds.len());
}

#[test]
fn pso_counts_initial_swarm_against_evaluation_budget() {
    let report = optimize_bonded_parameters(
        &sample_stats(),
        &OptimizationConfig {
            method: "pso".to_string(),
            objective: "bonded_parameter_parity".to_string(),
            max_evaluations: 4,
            seed: 13,
            swarm_size: Some(12),
            pso: None,
            bo: None,
            initial_parameters: std::collections::BTreeMap::new(),
        },
    );

    assert_eq!(report.evaluations.len(), 4);
    assert_eq!(
        report
            .evaluations
            .iter()
            .map(|record| record.id)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
}

#[test]
fn optimizer_trait_accepts_custom_objective_evaluator() {
    struct SphereEvaluator;

    impl ObjectiveEvaluator for SphereEvaluator {
        fn evaluate(&mut self, parameters: &[f64]) -> ObjectiveEvaluation {
            ObjectiveEvaluation::completed(parameters.iter().map(|value| value * value).sum())
        }
    }

    let bounds = vec![
        ParameterBound {
            name: "x".to_string(),
            min: -1.0,
            max: 1.0,
        },
        ParameterBound {
            name: "y".to_string(),
            min: -1.0,
            max: 1.0,
        },
    ];
    let mut optimizer: Box<dyn Optimizer> =
        Box::new(super::pso::PsoOptimizer::with_config(21, Some(5), None));
    let mut evaluator = SphereEvaluator;
    let (_best, value, evaluations) = optimizer.optimize(&bounds, &mut evaluator, &[], 10);

    assert!(!evaluations.is_empty());
    assert!(value.is_finite());
}

#[test]
fn pso_evaluates_external_initial_guesses_before_random_particles() {
    struct SphereEvaluator;

    impl ObjectiveEvaluator for SphereEvaluator {
        fn evaluate(&mut self, parameters: &[f64]) -> ObjectiveEvaluation {
            ObjectiveEvaluation::completed(parameters.iter().map(|value| value * value).sum())
        }
    }

    let bounds = vec![
        ParameterBound {
            name: "x".to_string(),
            min: -1.0,
            max: 1.0,
        },
        ParameterBound {
            name: "y".to_string(),
            min: -1.0,
            max: 1.0,
        },
    ];
    let initial_guesses = vec![InitialGuess {
        source: "test_seed".to_string(),
        parameters: vec![0.0, 0.0],
    }];
    let mut optimizer: Box<dyn Optimizer> =
        Box::new(super::pso::PsoOptimizer::with_config(22, Some(4), None));
    let mut evaluator = SphereEvaluator;
    let (best, value, evaluations) =
        optimizer.optimize(&bounds, &mut evaluator, &initial_guesses, 8);

    assert_eq!(evaluations[0].parameters, vec![0.0, 0.0]);
    assert_eq!(best, vec![0.0, 0.0]);
    assert_eq!(value, 0.0);
}

#[test]
fn pso_uses_batch_objective_evaluation_for_population_steps() {
    struct BatchEvaluator {
        single_calls: usize,
        batch_sizes: Vec<usize>,
    }

    impl ObjectiveEvaluator for BatchEvaluator {
        fn evaluate(&mut self, parameters: &[f64]) -> ObjectiveEvaluation {
            self.single_calls += 1;
            ObjectiveEvaluation::completed(parameters.iter().map(|value| value * value).sum())
        }

        fn evaluate_batch(&mut self, parameter_sets: &[Vec<f64>]) -> Vec<ObjectiveEvaluation> {
            self.batch_sizes.push(parameter_sets.len());
            parameter_sets
                .iter()
                .map(|parameters| {
                    ObjectiveEvaluation::completed(
                        parameters.iter().map(|value| value * value).sum(),
                    )
                })
                .collect()
        }
    }

    let bounds = vec![
        ParameterBound {
            name: "x".to_string(),
            min: -1.0,
            max: 1.0,
        },
        ParameterBound {
            name: "y".to_string(),
            min: -1.0,
            max: 1.0,
        },
    ];
    let mut optimizer: Box<dyn Optimizer> =
        Box::new(super::pso::PsoOptimizer::with_config(23, Some(3), None));
    let mut evaluator = BatchEvaluator {
        single_calls: 0,
        batch_sizes: Vec::new(),
    };
    let (_best, _value, evaluations) = optimizer.optimize(&bounds, &mut evaluator, &[], 6);

    assert_eq!(evaluations.len(), 6);
    assert_eq!(evaluator.single_calls, 0);
    assert_eq!(evaluator.batch_sizes, vec![3, 3]);
}

#[test]
fn pso_config_overrides_stall_termination() {
    let permissive = optimize_bonded_parameters(
        &sample_stats(),
        &OptimizationConfig {
            method: "pso".to_string(),
            objective: "bonded_parameter_parity".to_string(),
            max_evaluations: 20,
            seed: 11,
            swarm_size: Some(6),
            pso: Some(PsoConfig {
                fuzzy_self_tuning: Some(false),
                reboot_stalled_particles: Some(false),
                reboot_after_local_stall_iterations: None,
                linear_population_decrease: Some(false),
                max_iterations_without_global_best: Some(100),
                ..PsoConfig::default()
            }),
            bo: None,
            initial_parameters: std::collections::BTreeMap::new(),
        },
    );

    assert_eq!(permissive.evaluations.len(), 20);
}

#[test]
fn pso_recombination_restart_respects_evaluation_budget() {
    struct FlatEvaluator;

    impl ObjectiveEvaluator for FlatEvaluator {
        fn evaluate(&mut self, _parameters: &[f64]) -> ObjectiveEvaluation {
            ObjectiveEvaluation::completed(1.0)
        }
    }

    let bounds = vec![
        ParameterBound {
            name: "x".to_string(),
            min: -1.0,
            max: 1.0,
        },
        ParameterBound {
            name: "y".to_string(),
            min: -1.0,
            max: 1.0,
        },
    ];
    let config = PsoConfig {
        fuzzy_self_tuning: Some(false),
        reboot_stalled_particles: Some(true),
        reboot_after_local_stall_iterations: Some(1),
        restart_strategy: Some("recombine".to_string()),
        max_iterations_without_global_best: Some(100),
        ..PsoConfig::default()
    };
    let mut optimizer = super::pso::PsoOptimizer::with_config(47, Some(4), Some(&config));
    let mut evaluator = FlatEvaluator;
    let (_best, value, evaluations) = optimizer.optimize(&bounds, &mut evaluator, &[], 12);

    assert_eq!(value, 1.0);
    assert_eq!(evaluations.len(), 12);
}

#[test]
fn pso_writes_and_resumes_checkpoint_state() {
    struct SphereEvaluator;

    impl ObjectiveEvaluator for SphereEvaluator {
        fn evaluate(&mut self, parameters: &[f64]) -> ObjectiveEvaluation {
            ObjectiveEvaluation::completed(parameters.iter().map(|value| value * value).sum())
        }
    }

    let tempdir = tempfile::tempdir().unwrap();
    let checkpoint_path = tempdir.path().join("pso_checkpoint.json");
    let checkpoint_path = checkpoint_path.to_string_lossy().to_string();
    let bounds = vec![
        ParameterBound {
            name: "x".to_string(),
            min: -1.0,
            max: 1.0,
        },
        ParameterBound {
            name: "y".to_string(),
            min: -1.0,
            max: 1.0,
        },
    ];
    let pso = PsoConfig {
        fuzzy_self_tuning: Some(false),
        checkpoint_path: Some(checkpoint_path.clone()),
        checkpoint_interval_evaluations: Some(1),
        resume_from_checkpoint: Some(false),
        max_iterations_without_global_best: Some(100),
        ..PsoConfig::default()
    };
    let mut first = super::pso::PsoOptimizer::with_config(31, Some(4), Some(&pso));
    let mut evaluator = SphereEvaluator;
    let (_best, _value, first_evaluations) = first.optimize(&bounds, &mut evaluator, &[], 4);

    assert_eq!(first_evaluations.len(), 4);
    let checkpoint_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&checkpoint_path).unwrap()).unwrap();
    assert_eq!(
        checkpoint_json["schema_version"],
        "warp-cg.pso-checkpoint.v1"
    );
    assert_eq!(checkpoint_json["evaluations"].as_array().unwrap().len(), 4);

    let resumed_pso = PsoConfig {
        resume_from_checkpoint: Some(true),
        ..pso
    };
    let mut resumed = super::pso::PsoOptimizer::with_config(31, Some(4), Some(&resumed_pso));
    let mut evaluator = SphereEvaluator;
    let (_best, _value, resumed_evaluations) = resumed.optimize(&bounds, &mut evaluator, &[], 8);

    assert_eq!(resumed_evaluations.len(), 8);
    for (resumed, original) in resumed_evaluations.iter().zip(first_evaluations.iter()) {
        assert_eq!(resumed.id, original.id);
        assert!((resumed.objective.unwrap() - original.objective.unwrap()).abs() < 1.0e-12);
        assert_eq!(resumed.parameters.len(), original.parameters.len());
        for (resumed_parameter, original_parameter) in
            resumed.parameters.iter().zip(original.parameters.iter())
        {
            assert!((resumed_parameter - original_parameter).abs() < 1.0e-12);
        }
    }
}

#[test]
fn bo_writes_and_resumes_checkpoint_history() {
    struct SphereEvaluator;

    impl ObjectiveEvaluator for SphereEvaluator {
        fn evaluate(&mut self, parameters: &[f64]) -> ObjectiveEvaluation {
            ObjectiveEvaluation::completed(parameters.iter().map(|value| value * value).sum())
        }
    }

    let tempdir = tempfile::tempdir().unwrap();
    let checkpoint_path = tempdir.path().join("bo_checkpoint.json");
    let checkpoint_path = checkpoint_path.to_string_lossy().to_string();
    let bounds = vec![
        ParameterBound {
            name: "x".to_string(),
            min: -1.0,
            max: 1.0,
        },
        ParameterBound {
            name: "y".to_string(),
            min: -1.0,
            max: 1.0,
        },
    ];
    let bo = BoConfig {
        checkpoint_path: Some(checkpoint_path.clone()),
        checkpoint_interval_evaluations: Some(1),
        resume_from_checkpoint: Some(false),
        n_startup_trials: Some(4),
        n_candidates: Some(128),
        ..BoConfig::default()
    };
    let mut first = super::bo::BayesianOptimizer::with_config(71, "sphere".to_string(), Some(&bo));
    let mut evaluator = SphereEvaluator;
    let (_best, _value, first_evaluations) = first.optimize(&bounds, &mut evaluator, &[], 5);

    assert_eq!(first_evaluations.len(), 5);
    let checkpoint_json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&checkpoint_path).unwrap()).unwrap();
    assert_eq!(
        checkpoint_json["schema_version"],
        "warp-cg.bo-checkpoint.v1"
    );
    assert_eq!(checkpoint_json["evaluations"].as_array().unwrap().len(), 5);

    let resumed_bo = BoConfig {
        resume_from_checkpoint: Some(true),
        ..bo
    };
    let mut resumed =
        super::bo::BayesianOptimizer::with_config(71, "sphere".to_string(), Some(&resumed_bo));
    let mut evaluator = SphereEvaluator;
    let (_best, _value, resumed_evaluations) = resumed.optimize(&bounds, &mut evaluator, &[], 9);

    assert_eq!(resumed_evaluations.len(), 9);
    for (resumed, original) in resumed_evaluations.iter().zip(first_evaluations.iter()) {
        assert_eq!(resumed.id, original.id);
        assert!((resumed.objective.unwrap() - original.objective.unwrap()).abs() < 1.0e-12);
        for (resumed_parameter, original_parameter) in
            resumed.parameters.iter().zip(original.parameters.iter())
        {
            assert!((resumed_parameter - original_parameter).abs() < 1.0e-12);
        }
    }
}

#[test]
fn bo_records_failed_evaluations_as_penalized_history() {
    struct FailingEvaluator;

    impl ObjectiveEvaluator for FailingEvaluator {
        fn evaluate(&mut self, parameters: &[f64]) -> ObjectiveEvaluation {
            if parameters[0] > 0.0 {
                ObjectiveEvaluation::failed(EvaluationStatus::FailedSimulation {
                    reason: "positive x unstable".to_string(),
                })
            } else {
                ObjectiveEvaluation::completed(parameters.iter().map(|value| value * value).sum())
            }
        }
    }

    let bounds = vec![
        ParameterBound {
            name: "x".to_string(),
            min: -1.0,
            max: 1.0,
        },
        ParameterBound {
            name: "y".to_string(),
            min: -1.0,
            max: 1.0,
        },
    ];
    let bo = BoConfig {
        n_startup_trials: Some(4),
        n_candidates: Some(128),
        failure_handling: Some(FailureHandling::Penalize { value: Some(99.0) }),
        ..BoConfig::default()
    };
    let mut optimizer =
        super::bo::BayesianOptimizer::with_config(73, "failing_sphere".to_string(), Some(&bo));
    let mut evaluator = FailingEvaluator;
    let (_best, value, evaluations) = optimizer.optimize(&bounds, &mut evaluator, &[], 10);

    assert!(value.is_finite());
    assert!(evaluations
        .iter()
        .any(|record| matches!(record.status, EvaluationStatus::FailedSimulation { .. })));
    assert!(evaluations
        .iter()
        .filter(|record| !record.status.is_completed())
        .all(|record| record.objective == Some(99.0)));
}

#[test]
fn pso_optimizes_discrete_choice_probabilities() {
    struct ChoiceEvaluator;

    impl DiscreteObjectiveEvaluator for ChoiceEvaluator {
        fn evaluate_discrete(&mut self, choices: &[usize]) -> ObjectiveEvaluation {
            ObjectiveEvaluation::completed(if choices == [2, 0] { 0.0 } else { 1.0 })
        }
    }

    let config = OptimizationConfig {
        method: "pso".to_string(),
        objective: "discrete_choice".to_string(),
        max_evaluations: 32,
        seed: 43,
        swarm_size: Some(8),
        pso: Some(PsoConfig {
            fuzzy_self_tuning: Some(false),
            max_iterations_without_global_best: Some(100),
            ..PsoConfig::default()
        }),
        bo: None,
        initial_parameters: std::collections::BTreeMap::new(),
    };
    let mut evaluator = ChoiceEvaluator;
    let (best_choices, best_value, evaluations) =
        optimize_discrete_choices(&[3, 2], &mut evaluator, &config);

    assert_eq!(best_choices, vec![2, 0]);
    assert_eq!(best_value, 0.0);
    assert_eq!(evaluations.len(), 32);
    assert!(evaluations
        .iter()
        .all(|record| record.probabilities.len() == 5));
}

#[test]
fn pso_discrete_choice_probability_dilation_is_configurable() {
    struct BatchChoiceEvaluator {
        seen_choices: Vec<Vec<usize>>,
    }

    impl DiscreteObjectiveEvaluator for BatchChoiceEvaluator {
        fn evaluate_discrete(&mut self, choices: &[usize]) -> ObjectiveEvaluation {
            self.seen_choices.push(choices.to_vec());
            ObjectiveEvaluation::completed(0.0)
        }
    }

    let config = OptimizationConfig {
        method: "pso".to_string(),
        objective: "discrete_choice".to_string(),
        max_evaluations: 8,
        seed: 5,
        swarm_size: Some(4),
        pso: Some(PsoConfig {
            fuzzy_self_tuning: Some(false),
            discrete_probability_dilation: Some(true),
            discrete_probability_dilation_alpha: Some(8.0),
            max_iterations_without_global_best: Some(100),
            ..PsoConfig::default()
        }),
        bo: None,
        initial_parameters: std::collections::BTreeMap::new(),
    };
    let mut evaluator = BatchChoiceEvaluator {
        seen_choices: Vec::new(),
    };
    let (_best_choices, value, evaluations) =
        optimize_discrete_choices(&[2], &mut evaluator, &config);

    assert_eq!(value, 0.0);
    assert_eq!(evaluations.len(), 8);
    assert_eq!(evaluator.seen_choices.len(), 8);
}

#[test]
fn bonded_term_optimization_includes_angles_and_dihedrals() {
    let report = optimize_bonded_terms(
        &sample_bonded_stats(),
        &OptimizationConfig {
            method: "bayesian_optimization".to_string(),
            objective: "bonded_parameter_parity".to_string(),
            max_evaluations: 24,
            seed: 17,
            swarm_size: None,
            pso: None,
            bo: None,
            initial_parameters: std::collections::BTreeMap::new(),
        },
    );
    let names: Vec<&str> = report
        .best_parameters
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();

    assert!(names.contains(&"bond_0_1_length_angstrom"));
    assert!(names.contains(&"angle_0_1_2_angle_deg"));
    assert!(names.contains(&"dihedral_0_1_2_3_phase_deg"));
    assert_eq!(report.best_parameters.len(), 6);
}

#[test]
fn direct_statistics_assigns_centers_and_force_constants_without_optimizer() {
    let report = direct_statistics_report(
        &sample_bonded_stats(),
        &OptimizationConfig {
            method: "bayesian_optimization".to_string(),
            objective: "bonded_parameter_parity".to_string(),
            max_evaluations: 24,
            seed: 17,
            swarm_size: None,
            pso: None,
            bo: None,
            initial_parameters: std::collections::BTreeMap::new(),
        },
    );
    let params = report
        .best_parameters
        .iter()
        .cloned()
        .collect::<std::collections::BTreeMap<_, _>>();

    assert_eq!(report.method, "direct_statistics");
    assert!(report.evaluations.is_empty());
    assert_eq!(params["bond_0_1_length_angstrom"], 3.0);
    assert_eq!(params["angle_0_1_2_angle_deg"], 120.0);
    assert_eq!(params["dihedral_0_1_2_3_phase_deg"], 180.0);
    assert!(params.contains_key("bond_0_1_force"));
    assert!(params.contains_key("angle_0_1_2_force"));
    assert!(params.contains_key("dihedral_0_1_2_3_force"));
}
