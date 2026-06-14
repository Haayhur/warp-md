use crate::optimize::PsoConfig;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub(super) struct PsoSettings {
    pub(super) inertia: f64,
    pub(super) cognitive: f64,
    pub(super) social: f64,
    pub(super) max_velocity_fraction: f64,
    pub(super) min_velocity_fraction: f64,
    pub(super) max_iterations_without_global_best: usize,
    pub(super) fuzzy_self_tuning: bool,
    pub(super) reboot_stalled_particles: bool,
    pub(super) reboot_after_local_stall_iterations: usize,
    pub(super) restart_strategy: RestartStrategy,
    pub(super) linear_population_decrease: bool,
    pub(super) checkpoint_path: Option<String>,
    pub(super) checkpoint_interval_evaluations: usize,
    pub(super) resume_from_checkpoint: bool,
    pub(super) discrete_probability_dilation: bool,
    pub(super) discrete_probability_dilation_alpha: f64,
    fuzzy_adapt_inertia: bool,
    fuzzy_adapt_cognitive: bool,
    fuzzy_adapt_social: bool,
    fuzzy_adapt_min_velocity: bool,
    fuzzy_adapt_max_velocity: bool,
}

impl Default for PsoSettings {
    fn default() -> Self {
        Self {
            inertia: 0.65,
            cognitive: 1.4,
            social: 1.4,
            max_velocity_fraction: 1.0,
            min_velocity_fraction: 0.0,
            max_iterations_without_global_best: 6,
            fuzzy_self_tuning: true,
            reboot_stalled_particles: false,
            reboot_after_local_stall_iterations: 30,
            restart_strategy: RestartStrategy::Random,
            linear_population_decrease: false,
            checkpoint_path: None,
            checkpoint_interval_evaluations: 0,
            resume_from_checkpoint: false,
            discrete_probability_dilation: false,
            discrete_probability_dilation_alpha: 8.0,
            fuzzy_adapt_inertia: true,
            fuzzy_adapt_cognitive: true,
            fuzzy_adapt_social: true,
            fuzzy_adapt_min_velocity: true,
            fuzzy_adapt_max_velocity: true,
        }
    }
}

impl PsoSettings {
    pub(super) fn from_config(config: Option<&PsoConfig>) -> Self {
        let mut settings = Self::default();
        if let Some(config) = config {
            if let Some(value) = config.fuzzy_self_tuning {
                settings.fuzzy_self_tuning = value;
            }
            if let Some(value) = config.fuzzy_adapt_inertia {
                settings.fuzzy_adapt_inertia = value;
            }
            if let Some(value) = config.fuzzy_adapt_cognitive {
                settings.fuzzy_adapt_cognitive = value;
            }
            if let Some(value) = config.fuzzy_adapt_social {
                settings.fuzzy_adapt_social = value;
            }
            if let Some(value) = config.fuzzy_adapt_min_velocity {
                settings.fuzzy_adapt_min_velocity = value;
            }
            if let Some(value) = config.fuzzy_adapt_max_velocity {
                settings.fuzzy_adapt_max_velocity = value;
            }
            if let Some(value) = config.reboot_stalled_particles {
                settings.reboot_stalled_particles = value;
            }
            if let Some(value) = config.reboot_after_local_stall_iterations {
                settings.reboot_after_local_stall_iterations = value;
            }
            if let Some(value) = &config.restart_strategy {
                settings.restart_strategy = RestartStrategy::from_config(value);
            }
            if let Some(value) = config.linear_population_decrease {
                settings.linear_population_decrease = value;
            }
            if let Some(value) = config.max_iterations_without_global_best {
                settings.max_iterations_without_global_best = value;
            }
            if let Some(value) = &config.checkpoint_path {
                settings.checkpoint_path = Some(value.clone());
            }
            if let Some(value) = config.checkpoint_interval_evaluations {
                settings.checkpoint_interval_evaluations = value;
            }
            if let Some(value) = config.resume_from_checkpoint {
                settings.resume_from_checkpoint = value;
            }
            if let Some(value) = config.discrete_probability_dilation {
                settings.discrete_probability_dilation = value;
            }
            if let Some(value) = config.discrete_probability_dilation_alpha {
                settings.discrete_probability_dilation_alpha = value;
            }
        }
        settings
    }

    pub(super) fn apply_fuzzy_control_mask(
        &self,
        tuned: ParticleControls,
        fallback: ParticleControls,
    ) -> ParticleControls {
        ParticleControls {
            inertia: if self.fuzzy_adapt_inertia {
                tuned.inertia
            } else {
                fallback.inertia
            },
            cognitive: if self.fuzzy_adapt_cognitive {
                tuned.cognitive
            } else {
                fallback.cognitive
            },
            social: if self.fuzzy_adapt_social {
                tuned.social
            } else {
                fallback.social
            },
            min_velocity_fraction: if self.fuzzy_adapt_min_velocity {
                tuned.min_velocity_fraction
            } else {
                fallback.min_velocity_fraction
            },
            max_velocity_fraction: if self.fuzzy_adapt_max_velocity {
                tuned.max_velocity_fraction
            } else {
                fallback.max_velocity_fraction
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RestartStrategy {
    Random,
    Recombine,
}

impl RestartStrategy {
    fn from_config(value: &str) -> Self {
        match value {
            "recombine" => Self::Recombine,
            _ => Self::Random,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(super) struct ParticleControls {
    pub(super) inertia: f64,
    pub(super) cognitive: f64,
    pub(super) social: f64,
    pub(super) max_velocity_fraction: f64,
    pub(super) min_velocity_fraction: f64,
}

impl From<&PsoSettings> for ParticleControls {
    fn from(settings: &PsoSettings) -> Self {
        Self {
            inertia: settings.inertia,
            cognitive: settings.cognitive,
            social: settings.social,
            max_velocity_fraction: settings.max_velocity_fraction,
            min_velocity_fraction: settings.min_velocity_fraction,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_control_mask_can_freeze_individual_coefficients() {
        let settings = PsoSettings::from_config(Some(&PsoConfig {
            fuzzy_adapt_inertia: Some(false),
            fuzzy_adapt_social: Some(false),
            fuzzy_adapt_max_velocity: Some(false),
            ..PsoConfig::default()
        }));
        let fallback = ParticleControls::from(&PsoSettings::default());
        let tuned = ParticleControls {
            inertia: 1.0,
            cognitive: 3.0,
            social: 3.0,
            max_velocity_fraction: 0.2,
            min_velocity_fraction: 0.01,
        };
        let masked = settings.apply_fuzzy_control_mask(tuned, fallback);

        assert_eq!(masked.inertia, fallback.inertia);
        assert_eq!(masked.social, fallback.social);
        assert_eq!(masked.max_velocity_fraction, fallback.max_velocity_fraction);
        assert_eq!(masked.cognitive, tuned.cognitive);
        assert_eq!(masked.min_velocity_fraction, tuned.min_velocity_fraction);
    }
}
