use crate::optimize::PsoConfig;

use super::PsoTuningRequest;

impl From<&PsoTuningRequest> for PsoConfig {
    fn from(request: &PsoTuningRequest) -> Self {
        Self {
            fuzzy_self_tuning: request.fuzzy_self_tuning,
            fuzzy_adapt_inertia: request.fuzzy_adapt_inertia,
            fuzzy_adapt_cognitive: request.fuzzy_adapt_cognitive,
            fuzzy_adapt_social: request.fuzzy_adapt_social,
            fuzzy_adapt_min_velocity: request.fuzzy_adapt_min_velocity,
            fuzzy_adapt_max_velocity: request.fuzzy_adapt_max_velocity,
            reboot_stalled_particles: request.reboot_stalled_particles,
            reboot_after_local_stall_iterations: request.reboot_after_local_stall_iterations,
            restart_strategy: request.restart_strategy.clone(),
            linear_population_decrease: request.linear_population_decrease,
            max_iterations_without_global_best: request.max_iterations_without_global_best,
            checkpoint_path: request.checkpoint_path.clone(),
            checkpoint_interval_evaluations: request.checkpoint_interval_evaluations,
            resume_from_checkpoint: request.resume_from_checkpoint,
            discrete_probability_dilation: request.discrete_probability_dilation,
            discrete_probability_dilation_alpha: request.discrete_probability_dilation_alpha,
        }
    }
}
