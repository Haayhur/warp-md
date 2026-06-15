mod bo;
mod bounds;
mod file_evaluator;
mod initial_guess;
mod objective;
mod pso;
mod types;
mod utils;

use crate::parameters::{BondStats, BondedStats};
use crate::reference::{ReferenceTargetSet, ReferenceTermKind};

use bo::BayesianOptimizer;
use bounds::{
    default_bounds, reference_target_bounds, reference_target_force_name,
    reference_target_value_name,
};
use initial_guess::{named_initial_guess, MidpointInitialGuessProvider};
use objective::{ParameterProxyEvaluator, ReferenceTargetEvaluator};
use pso::PsoOptimizer;

pub use file_evaluator::{
    CandidateTrajectoryExtractionConfig, JsonCandidateTrajectoryResult, JsonFileEvaluatorCommand,
    JsonFileEvaluatorConfig, JsonFileObjectiveEvaluator, JsonObjectiveRequest, JsonObjectiveResult,
    JsonObjectiveStatus, JSON_OBJECTIVE_REQUEST_SCHEMA, JSON_OBJECTIVE_RESULT_SCHEMA,
};
pub use types::{
    BoConfig, DiscreteEvaluationRecord, DiscreteObjectiveEvaluator, EvaluationRecord,
    EvaluationStatus, FailureHandling, InitialGuess, InitialGuessProvider, NamedObjectiveEvaluator,
    NamedObjectiveEvaluatorAdapter, NamedParameter, ObjectiveEvaluation, ObjectiveEvaluator,
    OptimizationCandidate, OptimizationConfig, OptimizationReport, Optimizer, ParameterBound,
    PsoConfig, StructuralMetricScoringConfig, TrainingSetPolicy,
};

pub fn optimize_bonded_parameters(
    stats: &[BondStats],
    config: &OptimizationConfig,
) -> OptimizationReport {
    optimize_bonded_terms(
        &BondedStats {
            bonds: stats.to_vec(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
        },
        config,
    )
}

pub fn optimize_bonded_terms(
    stats: &BondedStats,
    config: &OptimizationConfig,
) -> OptimizationReport {
    let bounds = default_bounds(stats);
    if bounds.is_empty() {
        return OptimizationReport {
            status: "skipped".to_string(),
            method: config.method.clone(),
            objective: config.objective.clone(),
            objective_value: 0.0,
            converged: true,
            bounds,
            best_parameters: Vec::new(),
            evaluations: Vec::new(),
            message: "No bonded reference statistics were available for parameter tuning."
                .to_string(),
        };
    }

    let max_evaluations = config.max_evaluations.max(1);
    let mut evaluator = ParameterProxyEvaluator::new(stats, &bounds);
    let initial_guesses = match initial_guesses_for_config(&bounds, config) {
        Ok(guesses) => guesses,
        Err(message) => return invalid_parameter_report(config, bounds, message),
    };
    let mut optimizer: Box<dyn Optimizer> = match config.method.as_str() {
        "pso" => Box::new(PsoOptimizer::with_config(
            config.seed,
            config.swarm_size,
            config.pso.as_ref(),
        )),
        _ => Box::new(BayesianOptimizer::with_config(
            config.seed,
            config.objective.clone(),
            config.bo.as_ref(),
        )),
    };
    let (best, best_value, evaluations) =
        optimizer.optimize(&bounds, &mut evaluator, &initial_guesses, max_evaluations);
    let best_parameters = bounds
        .iter()
        .zip(best.iter())
        .map(|(bound, value)| (bound.name.clone(), *value))
        .collect();

    OptimizationReport {
        status: "ok".to_string(),
        method: config.method.clone(),
        objective: config.objective.clone(),
        objective_value: best_value,
        converged: best_value <= 1.0e-8,
        bounds,
        best_parameters,
        evaluations,
        message: "Optimized bonded parameters against mapped reference bond, angle, and dihedral statistics."
            .to_string(),
    }
}

pub fn direct_statistics_report(
    stats: &BondedStats,
    config: &OptimizationConfig,
) -> OptimizationReport {
    let bounds = default_bounds(stats);
    if bounds.is_empty() {
        return OptimizationReport {
            status: "skipped".to_string(),
            method: "direct_statistics".to_string(),
            objective: config.objective.clone(),
            objective_value: 0.0,
            converged: true,
            bounds,
            best_parameters: Vec::new(),
            evaluations: Vec::new(),
            message:
                "No bonded reference statistics were available for direct parameter assignment."
                    .to_string(),
        };
    }
    let mut best_parameters = Vec::new();
    for stat in &stats.bonds {
        best_parameters.push((
            format!("bond_{}_{}_length_angstrom", stat.bead_i, stat.bead_j),
            stat.mean,
        ));
        best_parameters.push((
            format!("bond_{}_{}_force", stat.bead_i, stat.bead_j),
            (1.0 / stat.std.max(0.02).powi(2)).clamp(1.0, 5000.0),
        ));
    }
    for stat in &stats.angles {
        best_parameters.push((
            format!(
                "angle_{}_{}_{}_angle_deg",
                stat.bead_i, stat.bead_j, stat.bead_k
            ),
            stat.mean_deg.clamp(0.0, 180.0),
        ));
        best_parameters.push((
            format!(
                "angle_{}_{}_{}_force",
                stat.bead_i, stat.bead_j, stat.bead_k
            ),
            (1.0 / stat.std_deg.max(1.0).powi(2) * 10_000.0).clamp(1.0, 500.0),
        ));
    }
    for stat in &stats.dihedrals {
        best_parameters.push((
            format!(
                "dihedral_{}_{}_{}_{}_phase_deg",
                stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
            ),
            stat.mean_deg.clamp(-180.0, 180.0),
        ));
        best_parameters.push((
            format!(
                "dihedral_{}_{}_{}_{}_force",
                stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
            ),
            (1.0 / stat.std_deg.max(1.0).powi(2) * 1_000.0).clamp(0.1, 100.0),
        ));
    }
    OptimizationReport {
        status: "ok".to_string(),
        method: "direct_statistics".to_string(),
        objective: config.objective.clone(),
        objective_value: 0.0,
        converged: true,
        bounds,
        best_parameters,
        evaluations: Vec::new(),
        message: "Assigned bonded parameters directly from mapped AA statistics; no BO/PSO optimization was run."
            .to_string(),
    }
}

fn initial_guesses_for_config(
    bounds: &[ParameterBound],
    config: &OptimizationConfig,
) -> Result<Vec<InitialGuess>, String> {
    let initial_guess_provider = MidpointInitialGuessProvider;
    let mut guesses = Vec::new();
    if let Some(guess) = named_initial_guess(
        "optimization.initial_parameters",
        &config.initial_parameters,
        bounds,
    )? {
        guesses.push(guess);
    }
    guesses.extend(initial_guess_provider.initial_guesses(bounds));
    Ok(guesses)
}

fn invalid_parameter_report(
    config: &OptimizationConfig,
    bounds: Vec<ParameterBound>,
    message: String,
) -> OptimizationReport {
    OptimizationReport {
        status: "error".to_string(),
        method: config.method.clone(),
        objective: config.objective.clone(),
        objective_value: f64::INFINITY,
        converged: false,
        bounds,
        best_parameters: Vec::new(),
        evaluations: Vec::new(),
        message,
    }
}

pub fn direct_statistics_report_from_targets(
    targets: &ReferenceTargetSet,
    config: &OptimizationConfig,
) -> OptimizationReport {
    let mut bounds = reference_target_bounds(targets);
    let mut best_parameters = Vec::new();
    for target in targets
        .constraints
        .iter()
        .chain(targets.bonds.iter())
        .chain(targets.angles.iter())
        .chain(targets.dihedrals.iter())
    {
        best_parameters.push((reference_target_value_name(target), target.mean));
        if let Some(force_name) = reference_target_force_name(target) {
            best_parameters.push((force_name, direct_force_from_target(target)));
        }
    }
    for (name, value) in &best_parameters {
        if !bounds.iter().any(|bound| bound.name == *name) {
            bounds.push(ParameterBound {
                name: name.clone(),
                min: *value,
                max: *value,
            });
        }
    }
    if best_parameters.is_empty() {
        return OptimizationReport {
            status: "skipped".to_string(),
            method: "direct_statistics".to_string(),
            objective: config.objective.clone(),
            objective_value: 0.0,
            converged: true,
            bounds,
            best_parameters,
            evaluations: Vec::new(),
            message: "No grouped reference targets were available for direct parameter assignment."
                .to_string(),
        };
    }
    OptimizationReport {
        status: "ok".to_string(),
        method: "direct_statistics".to_string(),
        objective: config.objective.clone(),
        objective_value: 0.0,
        converged: true,
        bounds,
        best_parameters,
        evaluations: Vec::new(),
        message: "Assigned bonded parameters directly from grouped reference target statistics; no BO/PSO optimization was run."
            .to_string(),
    }
}

fn direct_force_from_target(target: &crate::reference::ReferenceDistributionTarget) -> f64 {
    match target.kind {
        ReferenceTermKind::Constraint => 0.0,
        ReferenceTermKind::Bond => (1.0 / target.std.max(0.02).powi(2)).clamp(1.0, 5000.0),
        ReferenceTermKind::Angle => {
            (1.0 / target.std.max(1.0).powi(2) * 10_000.0).clamp(1.0, 500.0)
        }
        ReferenceTermKind::Dihedral => {
            (1.0 / target.std.max(1.0).powi(2) * 1_000.0).clamp(0.1, 100.0)
        }
    }
}

pub fn optimize_reference_targets(
    targets: &ReferenceTargetSet,
    config: &OptimizationConfig,
) -> OptimizationReport {
    let bounds = reference_target_bounds(targets);
    if bounds.is_empty() {
        return OptimizationReport {
            status: "skipped".to_string(),
            method: config.method.clone(),
            objective: config.objective.clone(),
            objective_value: 0.0,
            converged: true,
            bounds,
            best_parameters: Vec::new(),
            evaluations: Vec::new(),
            message: "No reference target distributions were available for parameter tuning."
                .to_string(),
        };
    }

    let max_evaluations = config.max_evaluations.max(1);
    let mut evaluator = ReferenceTargetEvaluator::new(targets);
    let initial_guesses = match initial_guesses_for_config(&bounds, config) {
        Ok(guesses) => guesses,
        Err(message) => return invalid_parameter_report(config, bounds, message),
    };
    let mut optimizer: Box<dyn Optimizer> = match config.method.as_str() {
        "pso" => Box::new(PsoOptimizer::with_config(
            config.seed,
            config.swarm_size,
            config.pso.as_ref(),
        )),
        _ => Box::new(BayesianOptimizer::with_config(
            config.seed,
            config.objective.clone(),
            config.bo.as_ref(),
        )),
    };
    let (best, best_value, evaluations) =
        optimizer.optimize(&bounds, &mut evaluator, &initial_guesses, max_evaluations);
    let best_parameters = bounds
        .iter()
        .zip(best.iter())
        .map(|(bound, value)| (bound.name.clone(), *value))
        .collect();

    OptimizationReport {
        status: "ok".to_string(),
        method: config.method.clone(),
        objective: config.objective.clone(),
        objective_value: best_value,
        converged: best_value <= 1.0e-8,
        bounds,
        best_parameters,
        evaluations,
        message: "Optimized bonded parameters against grouped reference target distributions with EMD scoring."
            .to_string(),
    }
}

pub fn optimize_with_named_evaluator(
    bounds: &[ParameterBound],
    evaluator: &mut dyn NamedObjectiveEvaluator,
    config: &OptimizationConfig,
    initial_guesses: &[InitialGuess],
) -> OptimizationReport {
    if bounds.is_empty() {
        return OptimizationReport {
            status: "skipped".to_string(),
            method: config.method.clone(),
            objective: config.objective.clone(),
            objective_value: 0.0,
            converged: true,
            bounds: Vec::new(),
            best_parameters: Vec::new(),
            evaluations: Vec::new(),
            message: "No parameter bounds were provided for optimization.".to_string(),
        };
    }

    let mut adapter = NamedObjectiveEvaluatorAdapter::new(bounds, evaluator);
    let mut initial_guesses = initial_guesses.to_vec();
    if !config.initial_parameters.is_empty() {
        match named_initial_guess(
            "optimization.initial_parameters",
            &config.initial_parameters,
            bounds,
        ) {
            Ok(Some(guess)) => initial_guesses.insert(0, guess),
            Ok(None) => {}
            Err(message) => {
                return invalid_parameter_report(config, bounds.to_vec(), message);
            }
        }
    }
    let mut optimizer: Box<dyn Optimizer> = match config.method.as_str() {
        "pso" => Box::new(PsoOptimizer::with_config(
            config.seed,
            config.swarm_size,
            config.pso.as_ref(),
        )),
        _ => Box::new(BayesianOptimizer::with_config(
            config.seed,
            config.objective.clone(),
            config.bo.as_ref(),
        )),
    };
    let (best, best_value, evaluations) = optimizer.optimize(
        bounds,
        &mut adapter,
        &initial_guesses,
        config.max_evaluations.max(1),
    );
    let best_parameters = bounds
        .iter()
        .zip(best.iter())
        .map(|(bound, value)| (bound.name.clone(), *value))
        .collect();

    OptimizationReport {
        status: "ok".to_string(),
        method: config.method.clone(),
        objective: config.objective.clone(),
        objective_value: best_value,
        converged: best_value <= 1.0e-8,
        bounds: bounds.to_vec(),
        best_parameters,
        evaluations,
        message: "Optimized parameters with a consumer-provided named evaluator.".to_string(),
    }
}

pub fn optimize_reference_targets_with_named_evaluator(
    targets: &ReferenceTargetSet,
    evaluator: &mut dyn NamedObjectiveEvaluator,
    config: &OptimizationConfig,
    initial_guesses: &[InitialGuess],
) -> OptimizationReport {
    let bounds = reference_target_bounds(targets);
    optimize_with_named_evaluator(&bounds, evaluator, config, initial_guesses)
}

pub fn optimize_discrete_choices(
    choice_counts: &[usize],
    evaluator: &mut dyn DiscreteObjectiveEvaluator,
    config: &OptimizationConfig,
) -> (Vec<usize>, f64, Vec<DiscreteEvaluationRecord>) {
    assert!(
        !choice_counts.is_empty() && choice_counts.iter().all(|count| *count > 0),
        "discrete PSO choice counts must be non-empty and positive"
    );
    let mut optimizer =
        PsoOptimizer::with_config(config.seed, config.swarm_size, config.pso.as_ref());
    optimizer.optimize_discrete(choice_counts, evaluator, config.max_evaluations.max(1))
}

#[cfg(test)]
mod reference_target_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_file_evaluator;
#[cfg(test)]
mod tests_named_evaluator;
