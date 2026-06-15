use super::*;

#[test]
fn optimize_with_named_evaluator_supplies_named_and_normalized_parameters() {
    struct EngineEvaluator {
        seen_names: Vec<Vec<String>>,
        seen_normalized: Vec<Vec<f64>>,
        batch_sizes: Vec<usize>,
    }

    impl NamedObjectiveEvaluator for EngineEvaluator {
        fn evaluate_candidate(&mut self, candidate: &OptimizationCandidate) -> ObjectiveEvaluation {
            self.seen_names.push(
                candidate
                    .named_parameters
                    .iter()
                    .map(|parameter| parameter.name.clone())
                    .collect(),
            );
            self.seen_normalized
                .push(candidate.normalized_parameters.clone());
            let objective = candidate
                .named_parameters
                .iter()
                .map(|parameter| parameter.value * parameter.value)
                .sum();
            let mut evaluation = ObjectiveEvaluation::completed(objective);
            evaluation
                .metrics
                .insert("runner.calls".to_string(), self.seen_names.len() as f64);
            evaluation
        }

        fn evaluate_candidate_batch(
            &mut self,
            candidates: &[OptimizationCandidate],
        ) -> Vec<ObjectiveEvaluation> {
            self.batch_sizes.push(candidates.len());
            candidates
                .iter()
                .map(|candidate| self.evaluate_candidate(candidate))
                .collect()
        }
    }

    let bounds = vec![
        ParameterBound {
            name: "bond_0_1_length_nm".to_string(),
            min: 0.0,
            max: 1.0,
        },
        ParameterBound {
            name: "angle_0_1_2_force".to_string(),
            min: 10.0,
            max: 20.0,
        },
    ];
    let mut evaluator = EngineEvaluator {
        seen_names: Vec::new(),
        seen_normalized: Vec::new(),
        batch_sizes: Vec::new(),
    };
    let report = optimize_with_named_evaluator(
        &bounds,
        &mut evaluator,
        &OptimizationConfig {
            method: "pso".to_string(),
            objective: "external_runner".to_string(),
            max_evaluations: 4,
            seed: 101,
            swarm_size: Some(2),
            pso: Some(PsoConfig {
                fuzzy_self_tuning: Some(false),
                max_iterations_without_global_best: Some(100),
                ..PsoConfig::default()
            }),
            bo: None,
            initial_parameters: std::collections::BTreeMap::new(),
        },
        &[InitialGuess {
            source: "user_seed".to_string(),
            parameters: vec![0.5, 15.0],
        }],
    );

    assert_eq!(report.status, "ok");
    assert_eq!(report.evaluations.len(), 4);
    assert_eq!(
        evaluator.seen_names[0],
        vec!["bond_0_1_length_nm", "angle_0_1_2_force"]
    );
    assert_eq!(evaluator.seen_normalized[0], vec![0.5, 0.5]);
    assert_eq!(evaluator.batch_sizes, vec![2, 2]);
    assert!(report.evaluations[0].metrics.contains_key("runner.calls"));
}

#[test]
fn named_evaluator_entrypoint_works_for_bayesian_optimization() {
    struct FailingEngineEvaluator;

    impl NamedObjectiveEvaluator for FailingEngineEvaluator {
        fn evaluate_candidate(&mut self, candidate: &OptimizationCandidate) -> ObjectiveEvaluation {
            if candidate.named_parameters[0].value > 0.5 {
                ObjectiveEvaluation::failed(EvaluationStatus::FailedSimulation {
                    reason: "external engine rejected candidate".to_string(),
                })
            } else {
                let objective = candidate.parameters.iter().map(|value| value * value).sum();
                ObjectiveEvaluation::completed(objective)
            }
        }
    }

    let bounds = vec![ParameterBound {
        name: "epsilon".to_string(),
        min: 0.0,
        max: 1.0,
    }];
    let mut evaluator = FailingEngineEvaluator;
    let report = optimize_with_named_evaluator(
        &bounds,
        &mut evaluator,
        &OptimizationConfig {
            method: "bayesian_optimization".to_string(),
            objective: "external_runner".to_string(),
            max_evaluations: 6,
            seed: 102,
            swarm_size: None,
            pso: None,
            bo: Some(BoConfig {
                n_startup_trials: Some(3),
                n_candidates: Some(32),
                failure_handling: Some(FailureHandling::Penalize { value: Some(77.0) }),
                ..BoConfig::default()
            }),
            initial_parameters: std::collections::BTreeMap::new(),
        },
        &[],
    );

    assert_eq!(report.status, "ok");
    assert_eq!(report.evaluations.len(), 6);
    assert!(report
        .evaluations
        .iter()
        .any(|record| matches!(record.status, EvaluationStatus::FailedSimulation { .. })));
    assert!(report
        .evaluations
        .iter()
        .filter(|record| !record.status.is_completed())
        .all(|record| record.objective == Some(77.0)));
}
