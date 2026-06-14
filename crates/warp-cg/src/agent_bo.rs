use crate::optimize::{BoConfig, FailureHandling, TrainingSetPolicy};

use super::BoTuningRequest;

impl From<&BoTuningRequest> for BoConfig {
    fn from(request: &BoTuningRequest) -> Self {
        Self {
            algorithm: request.algorithm.clone(),
            acquisition: request.acquisition.clone(),
            n_startup_trials: request.n_startup_trials,
            n_candidates: request.n_candidates,
            noise_variance: request.noise_variance,
            training_set_policy: request.training_set_policy.as_ref().map(|policy| {
                TrainingSetPolicy {
                    max_points: policy.max_points,
                    keep_best: policy.keep_best,
                    keep_recent: policy.keep_recent,
                    keep_diverse: policy.keep_diverse,
                }
            }),
            failure_handling: Some(failure_handling(request)),
            checkpoint_path: request.checkpoint_path.clone(),
            checkpoint_interval_evaluations: request.checkpoint_interval_evaluations,
            resume_from_checkpoint: request.resume_from_checkpoint,
            evaluator_signature: request.evaluator_signature.clone(),
        }
    }
}

fn failure_handling(request: &BoTuningRequest) -> FailureHandling {
    match request.failure_handling.as_deref().unwrap_or("penalize") {
        "exclude_from_gp_but_keep_in_history" | "exclude" => {
            FailureHandling::ExcludeFromGpButKeepInHistory
        }
        "model_as_constraint_later" | "constraint" => FailureHandling::ModelAsConstraintLater,
        _ => FailureHandling::Penalize {
            value: request.failure_penalty,
        },
    }
}
