mod checkpoint;
mod discrete;
mod fuzzy;
mod movement;
mod particle;
mod settings;

use std::path::Path;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;

use super::initial_guess::sanitize_initial_guesses;
use super::utils::{midpoint, random_position};
use super::{
    DiscreteEvaluationRecord, DiscreteObjectiveEvaluator, EvaluationRecord, InitialGuess,
    ObjectiveEvaluator, Optimizer, ParameterBound, PsoConfig,
};
use checkpoint::{load_checkpoint, save_checkpoint, PsoCheckpoint};
use discrete::{discrete_probability_bounds, DiscreteProbabilityEvaluator};
use fuzzy::fuzzy_controls;
use movement::{
    clamp_velocity, default_swarm_size, euclidean_distance, euclidean_norm, reflective_position,
    velocity_caps,
};
use particle::{
    elite_personal_bests, maybe_decrease_population, reboot_particle, should_reboot_particle,
    Particle,
};
use settings::{ParticleControls, PsoSettings};

pub(super) struct PsoOptimizer {
    seed: u64,
    swarm_size: Option<usize>,
    settings: PsoSettings,
}

impl PsoOptimizer {
    pub(super) fn with_config(
        seed: u64,
        swarm_size: Option<usize>,
        config: Option<&PsoConfig>,
    ) -> Self {
        Self {
            seed,
            swarm_size,
            settings: PsoSettings::from_config(config),
        }
    }

    pub(super) fn optimize_discrete(
        &mut self,
        choice_counts: &[usize],
        evaluator: &mut dyn DiscreteObjectiveEvaluator,
        max_evaluations: usize,
    ) -> (Vec<usize>, f64, Vec<DiscreteEvaluationRecord>) {
        let bounds = discrete_probability_bounds(choice_counts);
        let mut probability_evaluator = DiscreteProbabilityEvaluator::new(
            choice_counts,
            evaluator,
            self.seed,
            self.settings.discrete_probability_dilation,
            self.settings.discrete_probability_dilation_alpha,
        );
        let (_best_probabilities, _best_value, _trace) = run_pso(
            &bounds,
            &mut probability_evaluator,
            &[],
            max_evaluations.max(1),
            self.seed,
            self.swarm_size,
            &self.settings,
        );

        (
            probability_evaluator.best_choices,
            probability_evaluator.best_value,
            probability_evaluator.evaluations,
        )
    }
}

impl Optimizer for PsoOptimizer {
    fn optimize(
        &mut self,
        bounds: &[ParameterBound],
        evaluator: &mut dyn ObjectiveEvaluator,
        initial_guesses: &[InitialGuess],
        max_evaluations: usize,
    ) -> (Vec<f64>, f64, Vec<EvaluationRecord>) {
        run_pso(
            bounds,
            evaluator,
            initial_guesses,
            max_evaluations,
            self.seed,
            self.swarm_size,
            &self.settings,
        )
    }
}

fn run_pso(
    bounds: &[ParameterBound],
    evaluator: &mut dyn ObjectiveEvaluator,
    initial_guesses: &[InitialGuess],
    max_evaluations: usize,
    seed: u64,
    swarm_size: Option<usize>,
    settings: &PsoSettings,
) -> (Vec<f64>, f64, Vec<EvaluationRecord>) {
    let mut rng = ChaCha12Rng::seed_from_u64(seed);
    let dims = bounds.len();
    let swarm_size = swarm_size
        .unwrap_or_else(|| default_swarm_size(dims))
        .max(1)
        .min(max_evaluations);
    let initial_guesses = sanitize_initial_guesses(initial_guesses, bounds, swarm_size);
    let search_widths = velocity_caps(bounds, 1.0);
    let max_distance = euclidean_norm(&search_widths).max(1.0e-12);
    let checkpoint_path = settings.checkpoint_path.as_deref().map(Path::new);
    let loaded_checkpoint = checkpoint_path
        .filter(|_| settings.resume_from_checkpoint)
        .and_then(|path| load_checkpoint(path, seed, bounds));
    let (
        mut particles,
        mut global_best,
        mut global_value,
        mut worst_value,
        mut evaluations,
        mut iterations_without_global_best,
    ) = if let Some(checkpoint) = loaded_checkpoint {
        rng = checkpoint.rng;
        (
            checkpoint.particles,
            checkpoint.global_best,
            checkpoint.global_value,
            checkpoint.worst_value,
            checkpoint.evaluations,
            checkpoint.iterations_without_global_best,
        )
    } else {
        initialize_swarm(
            bounds,
            evaluator,
            &initial_guesses,
            max_evaluations,
            swarm_size,
            dims,
            settings,
            seed,
            &mut rng,
        )
    };
    let mut last_checkpoint_at = evaluations.len();

    if evaluations.len() >= max_evaluations {
        save_current_checkpoint(
            checkpoint_path,
            seed,
            bounds,
            &particles,
            &global_best,
            global_value,
            worst_value,
            &evaluations,
            iterations_without_global_best,
            &rng,
        );
        return (global_best, global_value, evaluations);
    }

    while evaluations.len() < max_evaluations {
        let mut improved_this_iteration = false;
        let particles_this_step = particles.len().min(max_evaluations - evaluations.len());
        let elite_positions = elite_personal_bests(&particles, 4);
        let mut previous_positions = Vec::with_capacity(particles_this_step);
        let mut previous_values = Vec::with_capacity(particles_this_step);
        let mut trial_positions = Vec::with_capacity(particles_this_step);

        for particle in particles.iter_mut().take(particles_this_step) {
            if should_reboot_particle(particle, settings, max_evaluations, evaluations.len()) {
                reboot_particle(particle, bounds, &elite_positions, &mut rng, settings);
            }
            previous_positions.push(particle.position.clone());
            previous_values.push(particle.current_value);
            for dim in 0..dims {
                let rp: f64 = rng.gen();
                let rg: f64 = rng.gen();
                particle.velocity[dim] = particle.controls.inertia * particle.velocity[dim]
                    + particle.controls.cognitive
                        * rp
                        * (particle.personal_best[dim] - particle.position[dim])
                    + particle.controls.social * rg * (global_best[dim] - particle.position[dim]);
                let max_velocity = search_widths[dim] * particle.controls.max_velocity_fraction;
                let min_velocity = search_widths[dim] * particle.controls.min_velocity_fraction;
                particle.velocity[dim] =
                    clamp_velocity(particle.velocity[dim], max_velocity, min_velocity);
                particle.position[dim] = reflective_position(
                    particle.position[dim],
                    particle.velocity[dim],
                    bounds[dim].min,
                    bounds[dim].max,
                    &mut rng,
                );
            }
            trial_positions.push(particle.position.clone());
        }

        let trial_values = evaluator.evaluate_batch(&trial_positions);
        for (((particle, previous_position), previous_value), evaluation) in particles
            .iter_mut()
            .take(particles_this_step)
            .zip(previous_positions.into_iter())
            .zip(previous_values.into_iter())
            .zip(trial_values.into_iter())
        {
            let penalty = if worst_value.is_finite() {
                worst_value.max(global_value) * 10.0
            } else {
                1.0e12
            };
            let value = evaluation.objective_or_penalty(penalty);
            particle.current_value = value;
            worst_value = worst_value.max(value);
            let movement = euclidean_distance(&previous_position, &particle.position);
            let distance_from_best = euclidean_distance(&particle.position, &global_best);
            if settings.fuzzy_self_tuning {
                let fallback = ParticleControls::from(settings);
                let tuned = fuzzy_controls(
                    worst_value,
                    global_value,
                    previous_value,
                    value,
                    movement,
                    distance_from_best,
                    max_distance,
                    fallback,
                );
                particle.controls = settings.apply_fuzzy_control_mask(tuned, fallback);
            }
            if value < particle.personal_value {
                particle.personal_value = value;
                particle.personal_best = particle.position.clone();
                particle.local_stall_iterations = 0;
            } else {
                particle.local_stall_iterations += 1;
            }
            if value < global_value {
                global_value = value;
                global_best = particle.position.clone();
                improved_this_iteration = true;
            }
            let mut record = EvaluationRecord::completed(
                evaluations.len(),
                particle.position.clone(),
                bounds,
                value,
                seed,
                "pso",
            );
            if !evaluation.metrics.is_empty() {
                record.metrics = evaluation.metrics;
                record.metrics.insert("objective".to_string(), value);
            }
            if !evaluation.status.is_completed() {
                record.status = evaluation.status;
                record.objective = Some(value);
            }
            evaluations.push(record);
            if evaluations.len() >= max_evaluations {
                break;
            }
        }
        if evaluations.len() >= max_evaluations {
            save_current_checkpoint(
                checkpoint_path,
                seed,
                bounds,
                &particles,
                &global_best,
                global_value,
                worst_value,
                &evaluations,
                iterations_without_global_best,
                &rng,
            );
            return (global_best, global_value, evaluations);
        }
        maybe_decrease_population(
            &mut particles,
            dims,
            max_evaluations,
            evaluations.len(),
            settings,
        );
        if improved_this_iteration {
            iterations_without_global_best = 0;
        } else {
            iterations_without_global_best += 1;
            if iterations_without_global_best >= settings.max_iterations_without_global_best {
                save_current_checkpoint(
                    checkpoint_path,
                    seed,
                    bounds,
                    &particles,
                    &global_best,
                    global_value,
                    worst_value,
                    &evaluations,
                    iterations_without_global_best,
                    &rng,
                );
                return (global_best, global_value, evaluations);
            }
        }
        if should_save_checkpoint(settings, last_checkpoint_at, evaluations.len()) {
            save_current_checkpoint(
                checkpoint_path,
                seed,
                bounds,
                &particles,
                &global_best,
                global_value,
                worst_value,
                &evaluations,
                iterations_without_global_best,
                &rng,
            );
            last_checkpoint_at = evaluations.len();
        }
    }

    save_current_checkpoint(
        checkpoint_path,
        seed,
        bounds,
        &particles,
        &global_best,
        global_value,
        worst_value,
        &evaluations,
        iterations_without_global_best,
        &rng,
    );
    (global_best, global_value, evaluations)
}

#[allow(clippy::type_complexity)]
fn initialize_swarm(
    bounds: &[ParameterBound],
    evaluator: &mut dyn ObjectiveEvaluator,
    initial_guesses: &[InitialGuess],
    max_evaluations: usize,
    swarm_size: usize,
    dims: usize,
    settings: &PsoSettings,
    seed: u64,
    rng: &mut ChaCha12Rng,
) -> (
    Vec<Particle>,
    Vec<f64>,
    f64,
    f64,
    Vec<EvaluationRecord>,
    usize,
) {
    let mut initial_positions = Vec::with_capacity(swarm_size);
    for particle_index in 0..swarm_size {
        initial_positions.push(
            initial_guesses
                .get(particle_index)
                .map(|guess| guess.parameters.clone())
                .unwrap_or_else(|| random_position(bounds, rng)),
        );
    }

    let initial_values = evaluator.evaluate_batch(&initial_positions);
    let mut particles = Vec::with_capacity(swarm_size);
    let mut global_best = midpoint(bounds);
    let mut global_value = f64::INFINITY;
    let mut worst_value = f64::NEG_INFINITY;
    let mut evaluations = Vec::with_capacity(max_evaluations);

    for (position, evaluation) in initial_positions
        .into_iter()
        .zip(initial_values.into_iter())
    {
        let penalty = if worst_value.is_finite() {
            worst_value.max(global_value) * 10.0
        } else {
            1.0e12
        };
        let value = evaluation.objective_or_penalty(penalty);
        if value < global_value {
            global_value = value;
            global_best = position.clone();
        }
        worst_value = worst_value.max(value);
        let mut record = EvaluationRecord::completed(
            evaluations.len(),
            position.clone(),
            bounds,
            value,
            seed,
            "pso",
        );
        if !evaluation.metrics.is_empty() {
            record.metrics = evaluation.metrics;
            record.metrics.insert("objective".to_string(), value);
        }
        if !evaluation.status.is_completed() {
            record.status = evaluation.status;
            record.objective = Some(value);
        }
        evaluations.push(record);
        particles.push(Particle {
            position: position.clone(),
            velocity: vec![0.0; dims],
            personal_best: position,
            current_value: value,
            personal_value: value,
            local_stall_iterations: 0,
            controls: ParticleControls::from(settings),
        });
        if evaluations.len() >= max_evaluations {
            break;
        }
    }

    (
        particles,
        global_best,
        global_value,
        worst_value,
        evaluations,
        0,
    )
}

fn should_save_checkpoint(
    settings: &PsoSettings,
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
    seed: u64,
    bounds: &[ParameterBound],
    particles: &[Particle],
    global_best: &[f64],
    global_value: f64,
    worst_value: f64,
    evaluations: &[EvaluationRecord],
    iterations_without_global_best: usize,
    rng: &ChaCha12Rng,
) {
    if let Some(path) = checkpoint_path {
        let checkpoint = PsoCheckpoint::new(
            seed,
            bounds,
            particles.to_vec(),
            global_best.to_vec(),
            global_value,
            worst_value,
            evaluations.to_vec(),
            iterations_without_global_best,
            rng.clone(),
        );
        save_checkpoint(path, &checkpoint);
    }
}
