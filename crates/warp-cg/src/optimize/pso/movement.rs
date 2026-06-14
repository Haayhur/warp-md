use rand::Rng;
use rand_chacha::ChaCha12Rng;

use crate::optimize::ParameterBound;

pub(super) fn default_swarm_size(dims: usize) -> usize {
    (10.0 + 2.0 * (dims as f64).sqrt()) as usize
}

pub(super) fn velocity_caps(bounds: &[ParameterBound], fraction: f64) -> Vec<f64> {
    bounds
        .iter()
        .map(|bound| ((bound.max - bound.min).abs() * fraction).max(0.0))
        .collect()
}

pub(super) fn reflective_position(
    value: f64,
    velocity: f64,
    min: f64,
    max: f64,
    rng: &mut ChaCha12Rng,
) -> f64 {
    let candidate = value + velocity;
    if candidate > max {
        max - rng.gen::<f64>() * velocity
    } else if candidate < min {
        min - rng.gen::<f64>() * velocity
    } else {
        candidate
    }
    .clamp(min, max)
}

pub(super) fn clamp_velocity(velocity: f64, max_velocity: f64, min_velocity: f64) -> f64 {
    if max_velocity <= 0.0 {
        return 0.0;
    }
    let capped = velocity.clamp(-max_velocity, max_velocity);
    if min_velocity <= 0.0 || capped.abs() >= min_velocity {
        capped
    } else if capped == 0.0 {
        min_velocity.min(max_velocity)
    } else {
        capped.signum() * min_velocity.min(max_velocity)
    }
}

pub(super) fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub(super) fn euclidean_norm(values: &[f64]) -> f64 {
    values.iter().map(|value| value * value).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_velocity_applies_positive_minimum_to_zero_velocity_like_fst_pso() {
        assert_eq!(clamp_velocity(0.0, 1.0, 0.2), 0.2);
        assert_eq!(clamp_velocity(0.1, 1.0, 0.2), 0.2);
        assert_eq!(clamp_velocity(-0.1, 1.0, 0.2), -0.2);
    }
}
