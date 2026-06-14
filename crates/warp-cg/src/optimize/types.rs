use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ParameterBound {
    pub name: String,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum EvaluationStatus {
    Completed,
    FailedSimulation { reason: String },
    FailedExtraction { reason: String },
    TimedOut,
    InvalidParameters { reason: String },
}

impl EvaluationStatus {
    pub fn is_completed(&self) -> bool {
        matches!(self, Self::Completed)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EvaluationRecord {
    pub id: usize,
    pub parameters: Vec<f64>,
    pub normalized_parameters: Vec<f64>,
    pub objective: Option<f64>,
    pub metrics: BTreeMap<String, f64>,
    pub status: EvaluationStatus,
    pub seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acquisition_value: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_mean: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicted_sigma: Option<f64>,
}

impl EvaluationRecord {
    pub fn completed(
        id: usize,
        parameters: Vec<f64>,
        bounds: &[ParameterBound],
        objective: f64,
        seed: u64,
        phase: impl Into<String>,
    ) -> Self {
        let mut metrics = BTreeMap::new();
        metrics.insert("objective".to_string(), objective);
        Self {
            id,
            normalized_parameters: normalize_parameters(&parameters, bounds),
            parameters,
            objective: Some(objective),
            metrics,
            status: EvaluationStatus::Completed,
            seed,
            phase: Some(phase.into()),
            acquisition_value: None,
            predicted_mean: None,
            predicted_sigma: None,
        }
    }

    pub fn failed(
        id: usize,
        parameters: Vec<f64>,
        bounds: &[ParameterBound],
        status: EvaluationStatus,
        seed: u64,
        phase: impl Into<String>,
    ) -> Self {
        Self {
            id,
            normalized_parameters: normalize_parameters(&parameters, bounds),
            parameters,
            objective: None,
            metrics: BTreeMap::new(),
            status,
            seed,
            phase: Some(phase.into()),
            acquisition_value: None,
            predicted_mean: None,
            predicted_sigma: None,
        }
    }

    pub fn training_objective(&self) -> Option<f64> {
        self.objective.filter(|value| value.is_finite())
    }
}

pub fn normalize_parameters(parameters: &[f64], bounds: &[ParameterBound]) -> Vec<f64> {
    parameters
        .iter()
        .zip(bounds.iter())
        .map(|(value, bound)| {
            let scale = (bound.max - bound.min).max(1.0e-12);
            ((value - bound.min) / scale).clamp(0.0, 1.0)
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NamedParameter {
    pub name: String,
    pub value: f64,
    pub normalized_value: f64,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OptimizationCandidate {
    pub parameters: Vec<f64>,
    pub normalized_parameters: Vec<f64>,
    pub named_parameters: Vec<NamedParameter>,
}

impl OptimizationCandidate {
    pub fn from_parameters(parameters: &[f64], bounds: &[ParameterBound]) -> Self {
        let normalized_parameters = normalize_parameters(parameters, bounds);
        let named_parameters = parameters
            .iter()
            .zip(bounds.iter())
            .zip(normalized_parameters.iter())
            .map(|((value, bound), normalized_value)| NamedParameter {
                name: bound.name.clone(),
                value: *value,
                normalized_value: *normalized_value,
                min: bound.min,
                max: bound.max,
            })
            .collect();
        Self {
            parameters: parameters.to_vec(),
            normalized_parameters,
            named_parameters,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DiscreteEvaluationRecord {
    pub iteration: usize,
    pub objective: f64,
    pub choices: Vec<usize>,
    pub probabilities: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OptimizationReport {
    pub status: String,
    pub method: String,
    pub objective: String,
    pub objective_value: f64,
    pub converged: bool,
    pub bounds: Vec<ParameterBound>,
    pub best_parameters: Vec<(String, f64)>,
    pub evaluations: Vec<EvaluationRecord>,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub method: String,
    pub objective: String,
    pub max_evaluations: usize,
    pub seed: u64,
    pub swarm_size: Option<usize>,
    pub pso: Option<PsoConfig>,
    pub bo: Option<BoConfig>,
}

#[derive(Debug, Clone, Default)]
pub struct BoConfig {
    pub algorithm: Option<String>,
    pub acquisition: Option<String>,
    pub n_startup_trials: Option<usize>,
    pub n_candidates: Option<usize>,
    pub noise_variance: Option<f64>,
    pub training_set_policy: Option<TrainingSetPolicy>,
    pub failure_handling: Option<FailureHandling>,
    pub checkpoint_path: Option<String>,
    pub checkpoint_interval_evaluations: Option<usize>,
    pub resume_from_checkpoint: Option<bool>,
    pub evaluator_signature: Option<String>,
}

#[derive(Debug, Clone)]
pub struct TrainingSetPolicy {
    pub max_points: usize,
    pub keep_best: usize,
    pub keep_recent: usize,
    pub keep_diverse: usize,
}

#[derive(Debug, Clone)]
pub enum FailureHandling {
    Penalize { value: Option<f64> },
    ExcludeFromGpButKeepInHistory,
    ModelAsConstraintLater,
}

#[derive(Debug, Clone, Default)]
pub struct PsoConfig {
    pub fuzzy_self_tuning: Option<bool>,
    pub fuzzy_adapt_inertia: Option<bool>,
    pub fuzzy_adapt_cognitive: Option<bool>,
    pub fuzzy_adapt_social: Option<bool>,
    pub fuzzy_adapt_min_velocity: Option<bool>,
    pub fuzzy_adapt_max_velocity: Option<bool>,
    pub reboot_stalled_particles: Option<bool>,
    pub reboot_after_local_stall_iterations: Option<usize>,
    pub restart_strategy: Option<String>,
    pub linear_population_decrease: Option<bool>,
    pub max_iterations_without_global_best: Option<usize>,
    pub checkpoint_path: Option<String>,
    pub checkpoint_interval_evaluations: Option<usize>,
    pub resume_from_checkpoint: Option<bool>,
    pub discrete_probability_dilation: Option<bool>,
    pub discrete_probability_dilation_alpha: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct InitialGuess {
    pub source: String,
    pub parameters: Vec<f64>,
}

pub trait InitialGuessProvider {
    fn initial_guesses(&self, bounds: &[ParameterBound]) -> Vec<InitialGuess>;
}

#[derive(Debug, Clone)]
pub struct ObjectiveEvaluation {
    pub objective: Option<f64>,
    pub metrics: BTreeMap<String, f64>,
    pub status: EvaluationStatus,
}

impl ObjectiveEvaluation {
    pub fn completed(objective: f64) -> Self {
        let mut metrics = BTreeMap::new();
        metrics.insert("objective".to_string(), objective);
        Self {
            objective: Some(objective),
            metrics,
            status: EvaluationStatus::Completed,
        }
    }

    pub fn failed(status: EvaluationStatus) -> Self {
        Self {
            objective: None,
            metrics: BTreeMap::new(),
            status,
        }
    }

    pub fn objective_or_penalty(&self, penalty: f64) -> f64 {
        self.objective
            .filter(|value| value.is_finite())
            .unwrap_or(penalty)
    }
}

pub trait ObjectiveEvaluator {
    fn evaluate(&mut self, parameters: &[f64]) -> ObjectiveEvaluation;

    fn evaluate_batch(&mut self, parameter_sets: &[Vec<f64>]) -> Vec<ObjectiveEvaluation> {
        parameter_sets
            .iter()
            .map(|parameters| self.evaluate(parameters))
            .collect()
    }
}

pub trait NamedObjectiveEvaluator {
    fn evaluate_candidate(&mut self, candidate: &OptimizationCandidate) -> ObjectiveEvaluation;

    fn evaluate_candidate_batch(
        &mut self,
        candidates: &[OptimizationCandidate],
    ) -> Vec<ObjectiveEvaluation> {
        candidates
            .iter()
            .map(|candidate| self.evaluate_candidate(candidate))
            .collect()
    }
}

pub struct NamedObjectiveEvaluatorAdapter<'a> {
    bounds: Vec<ParameterBound>,
    evaluator: &'a mut dyn NamedObjectiveEvaluator,
}

impl<'a> NamedObjectiveEvaluatorAdapter<'a> {
    pub fn new(bounds: &[ParameterBound], evaluator: &'a mut dyn NamedObjectiveEvaluator) -> Self {
        Self {
            bounds: bounds.to_vec(),
            evaluator,
        }
    }
}

impl ObjectiveEvaluator for NamedObjectiveEvaluatorAdapter<'_> {
    fn evaluate(&mut self, parameters: &[f64]) -> ObjectiveEvaluation {
        let candidate = OptimizationCandidate::from_parameters(parameters, &self.bounds);
        self.evaluator.evaluate_candidate(&candidate)
    }

    fn evaluate_batch(&mut self, parameter_sets: &[Vec<f64>]) -> Vec<ObjectiveEvaluation> {
        let candidates = parameter_sets
            .iter()
            .map(|parameters| OptimizationCandidate::from_parameters(parameters, &self.bounds))
            .collect::<Vec<_>>();
        self.evaluator.evaluate_candidate_batch(&candidates)
    }
}

pub trait DiscreteObjectiveEvaluator {
    fn evaluate_discrete(&mut self, choices: &[usize]) -> ObjectiveEvaluation;

    fn evaluate_discrete_batch(&mut self, choice_sets: &[Vec<usize>]) -> Vec<ObjectiveEvaluation> {
        choice_sets
            .iter()
            .map(|choices| self.evaluate_discrete(choices))
            .collect()
    }
}

pub trait Optimizer {
    fn optimize(
        &mut self,
        bounds: &[ParameterBound],
        evaluator: &mut dyn ObjectiveEvaluator,
        initial_guesses: &[InitialGuess],
        max_evaluations: usize,
    ) -> (Vec<f64>, f64, Vec<EvaluationRecord>);
}
