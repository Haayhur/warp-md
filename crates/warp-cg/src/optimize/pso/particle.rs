use rand::Rng;
use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use super::movement::default_swarm_size;
use super::settings::{ParticleControls, PsoSettings, RestartStrategy};
use crate::optimize::utils::random_position;
use crate::optimize::ParameterBound;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct Particle {
    pub(super) position: Vec<f64>,
    pub(super) velocity: Vec<f64>,
    pub(super) personal_best: Vec<f64>,
    pub(super) current_value: f64,
    pub(super) personal_value: f64,
    pub(super) local_stall_iterations: usize,
    pub(super) controls: ParticleControls,
}

pub(super) fn should_reboot_particle(
    particle: &Particle,
    settings: &PsoSettings,
    max_evaluations: usize,
    completed_evaluations: usize,
) -> bool {
    settings.reboot_stalled_particles
        && completed_evaluations + settings.reboot_after_local_stall_iterations < max_evaluations
        && particle.local_stall_iterations >= settings.reboot_after_local_stall_iterations
}

pub(super) fn reboot_particle(
    particle: &mut Particle,
    bounds: &[ParameterBound],
    elite_positions: &[Vec<f64>],
    rng: &mut ChaCha12Rng,
    settings: &PsoSettings,
) {
    particle.position = match settings.restart_strategy {
        RestartStrategy::Random => random_position(bounds, rng),
        RestartStrategy::Recombine => recombined_position(particle, elite_positions, bounds, rng)
            .unwrap_or_else(|| random_position(bounds, rng)),
    };
    particle.velocity.fill(0.0);
    particle.personal_best = particle.position.clone();
    particle.current_value = f64::INFINITY;
    particle.personal_value = f64::INFINITY;
    particle.local_stall_iterations = 0;
    particle.controls = ParticleControls::from(settings);
}

pub(super) fn elite_personal_bests(particles: &[Particle], limit: usize) -> Vec<Vec<f64>> {
    let mut elite = particles
        .iter()
        .filter(|particle| particle.personal_value.is_finite())
        .map(|particle| (particle.personal_value, particle.personal_best.clone()))
        .collect::<Vec<_>>();
    elite.sort_by(|a, b| a.0.total_cmp(&b.0));
    elite
        .into_iter()
        .take(limit)
        .map(|(_, position)| position)
        .collect()
}

fn recombined_position(
    particle: &Particle,
    elite_positions: &[Vec<f64>],
    bounds: &[ParameterBound],
    rng: &mut ChaCha12Rng,
) -> Option<Vec<f64>> {
    let elite_position = elite_positions
        .iter()
        .find(|position| position.as_slice() != particle.personal_best.as_slice())
        .or_else(|| elite_positions.first())?;
    Some(combine_positions(
        &particle.position,
        elite_position,
        0,
        bounds,
        rng,
    ))
}

pub(super) fn combine_positions(
    source: &[f64],
    elite: &[f64],
    rank: usize,
    bounds: &[ParameterBound],
    rng: &mut ChaCha12Rng,
) -> Vec<f64> {
    let beta = ((rank as f64 - 5.0).abs() - 1.0) / 3.0;
    source
        .iter()
        .zip(elite.iter())
        .zip(bounds.iter())
        .map(|((source, elite), bound)| {
            let half_delta = (elite - source) / 2.0;
            let lower = source - half_delta * (1.0 - beta);
            let upper = source + half_delta * (1.0 + beta);
            (lower + (upper - lower) * rng.gen::<f64>()).clamp(bound.min, bound.max)
        })
        .collect()
}

pub(super) fn maybe_decrease_population(
    particles: &mut Vec<Particle>,
    dims: usize,
    max_evaluations: usize,
    completed_evaluations: usize,
    settings: &PsoSettings,
) {
    if !settings.linear_population_decrease || particles.len() <= 1 {
        return;
    }
    let target = target_population_size(dims, max_evaluations, completed_evaluations)
        .max(1)
        .min(particles.len());
    if target >= particles.len() {
        return;
    }
    particles.sort_by(|a, b| a.personal_value.total_cmp(&b.personal_value));
    particles.truncate(target);
}

fn target_population_size(
    dims: usize,
    max_evaluations: usize,
    completed_evaluations: usize,
) -> usize {
    let heuristic_min = default_swarm_size(dims);
    let heuristic_max = ((dims as f64).sqrt() * (dims.max(2) as f64).ln()).round() as usize;
    let population_max = heuristic_max.max(heuristic_min);
    let population_min = heuristic_min.min(population_max).max(1);
    if max_evaluations <= population_max || completed_evaluations <= population_max {
        return population_max.min(max_evaluations.max(1));
    }
    let remaining_span = (max_evaluations - population_max).max(1) as f64;
    let progress = (completed_evaluations - population_max) as f64 / remaining_span;
    let target = population_max as f64 + (population_min as f64 - population_max as f64) * progress;
    let target = target.round().max(population_min as f64) as usize;
    target.min(max_evaluations.saturating_sub(completed_evaluations).max(1))
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn combine_positions_stays_inside_bounds() {
        let bounds = vec![
            ParameterBound {
                name: "x".to_string(),
                min: -1.0,
                max: 1.0,
            },
            ParameterBound {
                name: "y".to_string(),
                min: -2.0,
                max: 2.0,
            },
        ];
        let mut rng = ChaCha12Rng::seed_from_u64(11);
        let combined = combine_positions(&[-0.5, 0.0], &[0.75, 1.5], 0, &bounds, &mut rng);

        assert_eq!(combined.len(), 2);
        assert!(combined[0] >= -1.0 && combined[0] <= 1.0);
        assert!(combined[1] >= -2.0 && combined[1] <= 2.0);
    }

    #[test]
    fn elite_personal_bests_returns_lowest_objective_positions() {
        let particles = vec![
            particle_with_best(vec![1.0], 3.0),
            particle_with_best(vec![2.0], 1.0),
            particle_with_best(vec![3.0], 2.0),
        ];

        assert_eq!(
            elite_personal_bests(&particles, 2),
            vec![vec![2.0], vec![3.0]]
        );
    }

    fn particle_with_best(personal_best: Vec<f64>, personal_value: f64) -> Particle {
        Particle {
            position: personal_best.clone(),
            velocity: vec![0.0; personal_best.len()],
            personal_best,
            current_value: personal_value,
            personal_value,
            local_stall_iterations: 0,
            controls: ParticleControls {
                inertia: 0.0,
                cognitive: 0.0,
                social: 0.0,
                max_velocity_fraction: 0.0,
                min_velocity_fraction: 0.0,
            },
        }
    }
}
