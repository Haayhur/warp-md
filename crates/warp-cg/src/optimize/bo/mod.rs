mod acquisition;
mod candidate;
mod checkpoint;
mod settings;
mod surrogate;
mod training;

use std::path::Path;

use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;

use super::initial_guess::sanitize_initial_guesses;
use super::utils::{midpoint, random_position};
use super::{
    BoConfig, EvaluationRecord, InitialGuess, ObjectiveEvaluator, Optimizer, ParameterBound,
};
use acquisition::acquisition_score;
use candidate::{
    denormalize_parameters, latin_hypercube_point, local_normalized, midpoint_normalized,
    random_normalized,
};
use checkpoint::{load_checkpoint, save_checkpoint, BoCheckpoint};
use settings::BoSettings;
use surrogate::GaussianProcess;
use training::select_training_data;

pub(super) struct BayesianOptimizer {
    seed: u64,
    objective_signature: String,
    settings: BoSettings,
}

impl BayesianOptimizer {
    pub(super) fn with_config(
        seed: u64,
        objective_signature: String,
        config: Option<&BoConfig>,
    ) -> Self {
        Self {
            seed,
            objective_signature,
            settings: BoSettings::from_config(config),
        }
    }
}

impl Optimizer for BayesianOptimizer {
    fn optimize(
        &mut self,
        bounds: &[ParameterBound],
        evaluator: &mut dyn ObjectiveEvaluator,
        initial_guesses: &[InitialGuess],
        max_evaluations: usize,
    ) -> (Vec<f64>, f64, Vec<EvaluationRecord>) {
        run_bayesian_optimization(
            bounds,
            evaluator,
            initial_guesses,
            max_evaluations,
            self.seed,
            &self.objective_signature,
            &self.settings,
        )
    }
}

fn run_bayesian_optimization(
    bounds: &[ParameterBound],
    evaluator: &mut dyn ObjectiveEvaluator,
    initial_guesses: &[InitialGuess],
    max_evaluations: usize,
    seed: u64,
    objective_signature: &str,
    settings: &BoSettings,
) -> (Vec<f64>, f64, Vec<EvaluationRecord>) {
    let max_evaluations = max_evaluations.max(1);
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let checkpoint_path = settings.checkpoint_path.as_deref().map(Path::new);
    let loaded_checkpoint = checkpoint_path
        .filter(|_| settings.resume_from_checkpoint)
        .and_then(|path| {
            load_checkpoint(
                path,
                settings.algorithm,
                settings.acquisition,
                seed,
                bounds,
                objective_signature,
                settings.evaluator_signature.as_deref(),
            )
        });
    let (mut best, mut best_value, mut evaluations) = if let Some(checkpoint) = loaded_checkpoint {
        rng = checkpoint.rng;
        (
            checkpoint.best_parameters,
            checkpoint.best_value,
            checkpoint.evaluations,
        )
    } else {
        (
            midpoint(bounds),
            f64::INFINITY,
            Vec::with_capacity(max_evaluations),
        )
    };
    let mut last_checkpoint_at = evaluations.len();
    let startup_trials = settings.startup_trials(bounds.len(), max_evaluations);
    let initial_guesses = sanitize_initial_guesses(initial_guesses, bounds, startup_trials);

    while evaluations.len() < max_evaluations {
        let proposal = propose_parameters(
            bounds,
            &evaluations,
            &best,
            &initial_guesses,
            startup_trials,
            settings,
            &mut rng,
        );
        let evaluation = evaluator.evaluate(&proposal.parameters);
        let penalty = failure_penalty(settings, best_value, &evaluations);
        let value = evaluation.objective_or_penalty(penalty);
        if value < best_value {
            best_value = value;
            best = proposal.parameters.clone();
        }
        let mut record = EvaluationRecord::completed(
            evaluations.len(),
            proposal.parameters,
            bounds,
            value,
            seed,
            proposal.phase,
        );
        record.acquisition_value = proposal.acquisition_value;
        record.predicted_mean = proposal.predicted_mean;
        record.predicted_sigma = proposal.predicted_sigma;
        if !evaluation.metrics.is_empty() {
            record.metrics = evaluation.metrics;
            record.metrics.insert("objective".to_string(), value);
        }
        if !evaluation.status.is_completed() {
            record.status = evaluation.status;
            record.objective = Some(value);
        }
        evaluations.push(record);

        if should_save_checkpoint(settings, last_checkpoint_at, evaluations.len()) {
            save_current_checkpoint(
                checkpoint_path,
                bounds,
                objective_signature,
                seed,
                settings,
                &evaluations,
                &best,
                best_value,
                &rng,
            );
            last_checkpoint_at = evaluations.len();
        }
    }

    save_current_checkpoint(
        checkpoint_path,
        bounds,
        objective_signature,
        seed,
        settings,
        &evaluations,
        &best,
        best_value,
        &rng,
    );
    (best, best_value, evaluations)
}

struct Proposal {
    parameters: Vec<f64>,
    phase: &'static str,
    acquisition_value: Option<f64>,
    predicted_mean: Option<f64>,
    predicted_sigma: Option<f64>,
}

fn propose_parameters(
    bounds: &[ParameterBound],
    evaluations: &[EvaluationRecord],
    best: &[f64],
    initial_guesses: &[InitialGuess],
    startup_trials: usize,
    settings: &BoSettings,
    rng: &mut ChaCha12Rng,
) -> Proposal {
    let iteration = evaluations.len();
    if let Some(guess) = initial_guesses.get(iteration) {
        return Proposal {
            parameters: guess.parameters.clone(),
            phase: "initial_guess",
            acquisition_value: None,
            predicted_mean: None,
            predicted_sigma: None,
        };
    }
    if iteration < startup_trials {
        return Proposal {
            parameters: latin_hypercube_point(bounds, iteration, startup_trials, rng),
            phase: "startup_lhs",
            acquisition_value: None,
            predicted_mean: None,
            predicted_sigma: None,
        };
    }
    propose_gp_ei(bounds, evaluations, best, settings, rng).unwrap_or_else(|| Proposal {
        parameters: random_position(bounds, rng),
        phase: "fallback_random",
        acquisition_value: None,
        predicted_mean: None,
        predicted_sigma: None,
    })
}

fn propose_gp_ei(
    bounds: &[ParameterBound],
    evaluations: &[EvaluationRecord],
    best: &[f64],
    settings: &BoSettings,
    rng: &mut ChaCha12Rng,
) -> Option<Proposal> {
    let (x_train, y_train) = select_training_data(
        evaluations,
        &settings.training_set_policy,
        settings.failure_policy,
        settings.failure_penalty,
    );
    let model = GaussianProcess::fit(&x_train, &y_train, settings.noise_variance)?;
    let best_value = y_train.iter().copied().fold(f64::INFINITY, f64::min);
    let best_normalized = super::types::normalize_parameters(best, bounds);
    let candidate_count = settings.candidate_count(bounds.len());
    let mut best_candidate = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_mean = None;
    let mut best_sigma = None;

    for candidate_index in 0..candidate_count {
        let normalized = if candidate_index == 0 {
            midpoint_normalized(bounds.len())
        } else if candidate_index % 4 == 0 {
            local_normalized(&best_normalized, rng)
        } else {
            random_normalized(bounds.len(), rng)
        };
        let (mean, sigma) = model.predict(&normalized);
        let score = acquisition_score(settings.acquisition, best_value, mean, sigma);
        if score > best_score {
            best_score = score;
            best_mean = Some(mean);
            best_sigma = Some(sigma);
            best_candidate = Some(normalized);
        }
    }

    best_candidate.map(|normalized| Proposal {
        parameters: denormalize_parameters(&normalized, bounds),
        phase: "gp_log_ei",
        acquisition_value: Some(best_score),
        predicted_mean: best_mean,
        predicted_sigma: best_sigma,
    })
}

fn failure_penalty(
    settings: &BoSettings,
    best_value: f64,
    evaluations: &[EvaluationRecord],
) -> f64 {
    if let Some(value) = settings.failure_penalty {
        return value;
    }
    if best_value.is_finite() {
        return best_value.abs().max(best_value) * 10.0 + 1.0;
    }
    evaluations
        .iter()
        .filter_map(EvaluationRecord::training_objective)
        .max_by(|a, b| a.total_cmp(b))
        .map_or(1.0e12, |value| value.abs().max(value) * 10.0 + 1.0)
}

fn should_save_checkpoint(
    settings: &BoSettings,
    last_checkpoint_at: usize,
    completed_evaluations: usize,
) -> bool {
    settings.checkpoint_path.is_some()
        && settings.checkpoint_interval_evaluations > 0
        && completed_evaluations.saturating_sub(last_checkpoint_at)
            >= settings.checkpoint_interval_evaluations
}

#[allow(clippy::too_many_arguments)]
fn save_current_checkpoint(
    checkpoint_path: Option<&Path>,
    bounds: &[ParameterBound],
    objective_signature: &str,
    seed: u64,
    settings: &BoSettings,
    evaluations: &[EvaluationRecord],
    best: &[f64],
    best_value: f64,
    rng: &ChaCha12Rng,
) {
    if let Some(path) = checkpoint_path {
        let checkpoint = BoCheckpoint::new(
            settings.algorithm,
            settings.acquisition,
            seed,
            bounds,
            objective_signature.to_string(),
            settings.evaluator_signature.clone(),
            evaluations.to_vec(),
            best.to_vec(),
            best_value,
            rng.clone(),
        );
        save_checkpoint(path, &checkpoint);
    }
}
