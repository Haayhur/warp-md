use serde::{Deserialize, Serialize};

use crate::optimize::{BoConfig, FailureHandling, TrainingSetPolicy};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum BoAlgorithm {
    GpExpectedImprovement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum AcquisitionKind {
    ExpectedImprovement,
    LogExpectedImprovement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FailurePolicy {
    Penalize,
    ExcludeFromGpButKeepInHistory,
    ModelAsConstraintLater,
}

#[derive(Debug, Clone)]
pub(super) struct BoSettings {
    pub(super) algorithm: BoAlgorithm,
    pub(super) acquisition: AcquisitionKind,
    pub(super) n_startup_trials: Option<usize>,
    pub(super) n_candidates: Option<usize>,
    pub(super) noise_variance: f64,
    pub(super) training_set_policy: TrainingSetPolicy,
    pub(super) failure_policy: FailurePolicy,
    pub(super) failure_penalty: Option<f64>,
    pub(super) checkpoint_path: Option<String>,
    pub(super) checkpoint_interval_evaluations: usize,
    pub(super) resume_from_checkpoint: bool,
    pub(super) evaluator_signature: Option<String>,
}

impl BoSettings {
    pub(super) fn from_config(config: Option<&BoConfig>) -> Self {
        let mut settings = Self::default();
        if let Some(config) = config {
            settings.algorithm = parse_algorithm(config.algorithm.as_deref());
            settings.acquisition = parse_acquisition(config.acquisition.as_deref());
            settings.n_startup_trials = config.n_startup_trials;
            settings.n_candidates = config.n_candidates;
            if let Some(noise_variance) = config.noise_variance {
                settings.noise_variance = noise_variance.max(1.0e-12);
            }
            if let Some(policy) = &config.training_set_policy {
                settings.training_set_policy = sanitize_training_policy(policy.clone());
            }
            if let Some(failure_handling) = &config.failure_handling {
                let (policy, penalty) = parse_failure_handling(failure_handling);
                settings.failure_policy = policy;
                settings.failure_penalty = penalty;
            }
            settings.checkpoint_path = config.checkpoint_path.clone();
            if let Some(interval) = config.checkpoint_interval_evaluations {
                settings.checkpoint_interval_evaluations = interval;
            }
            if let Some(resume) = config.resume_from_checkpoint {
                settings.resume_from_checkpoint = resume;
            }
            settings.evaluator_signature = config.evaluator_signature.clone();
        }
        settings
    }

    pub(super) fn startup_trials(&self, dimensions: usize, max_evaluations: usize) -> usize {
        self.n_startup_trials
            .unwrap_or_else(|| (dimensions + 1).clamp(6, 16))
            .max(1)
            .min(max_evaluations)
    }

    pub(super) fn candidate_count(&self, dimensions: usize) -> usize {
        self.n_candidates
            .unwrap_or_else(|| (dimensions * 128).clamp(512, 8192))
            .max(1)
    }
}

impl Default for BoSettings {
    fn default() -> Self {
        Self {
            algorithm: BoAlgorithm::GpExpectedImprovement,
            acquisition: AcquisitionKind::LogExpectedImprovement,
            n_startup_trials: None,
            n_candidates: None,
            noise_variance: 1.0e-6,
            training_set_policy: TrainingSetPolicy {
                max_points: 128,
                keep_best: 32,
                keep_recent: 48,
                keep_diverse: 48,
            },
            failure_policy: FailurePolicy::Penalize,
            failure_penalty: None,
            checkpoint_path: None,
            checkpoint_interval_evaluations: 1,
            resume_from_checkpoint: false,
            evaluator_signature: None,
        }
    }
}

fn parse_algorithm(value: Option<&str>) -> BoAlgorithm {
    match value.unwrap_or("gp_expected_improvement") {
        "gp_expected_improvement" | "gp_ei" | "gp_log_ei" | "bayesian_optimization" | "bo" => {
            BoAlgorithm::GpExpectedImprovement
        }
        _ => BoAlgorithm::GpExpectedImprovement,
    }
}

fn parse_acquisition(value: Option<&str>) -> AcquisitionKind {
    match value.unwrap_or("log_expected_improvement") {
        "expected_improvement" | "ei" => AcquisitionKind::ExpectedImprovement,
        "log_expected_improvement" | "log_ei" | "logei" => AcquisitionKind::LogExpectedImprovement,
        _ => AcquisitionKind::LogExpectedImprovement,
    }
}

fn parse_failure_handling(value: &FailureHandling) -> (FailurePolicy, Option<f64>) {
    match value {
        FailureHandling::Penalize { value } => (FailurePolicy::Penalize, *value),
        FailureHandling::ExcludeFromGpButKeepInHistory => {
            (FailurePolicy::ExcludeFromGpButKeepInHistory, None)
        }
        FailureHandling::ModelAsConstraintLater => (FailurePolicy::ModelAsConstraintLater, None),
    }
}

fn sanitize_training_policy(policy: TrainingSetPolicy) -> TrainingSetPolicy {
    let max_points = policy.max_points.max(1);
    TrainingSetPolicy {
        max_points,
        keep_best: policy.keep_best.min(max_points),
        keep_recent: policy.keep_recent.min(max_points),
        keep_diverse: policy.keep_diverse.min(max_points),
    }
}
