mod bo;
mod bounds;
mod file_evaluator;
mod initial_guess;
mod objective;
mod pso;
mod types;
mod utils;

use crate::parameters::{BondStats, BondedStats};
use crate::reference::ReferenceTargetSet;

use bo::BayesianOptimizer;
use bounds::{default_bounds, reference_target_bounds};
use initial_guess::MidpointInitialGuessProvider;
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
    PsoConfig, TrainingSetPolicy,
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
    let initial_guess_provider = MidpointInitialGuessProvider;
    let initial_guesses = initial_guess_provider.initial_guesses(&bounds);
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
    let initial_guess_provider = MidpointInitialGuessProvider;
    let initial_guesses = initial_guess_provider.initial_guesses(&bounds);
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
        initial_guesses,
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
