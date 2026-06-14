use rand::Rng;
use rand_chacha::ChaCha12Rng;

use crate::optimize::ParameterBound;

pub(super) fn latin_hypercube_point(
    bounds: &[ParameterBound],
    iteration: usize,
    total: usize,
    rng: &mut ChaCha12Rng,
) -> Vec<f64> {
    bounds
        .iter()
        .enumerate()
        .map(|(dim, bound)| {
            let offset: f64 = rng.gen();
            let rank = (iteration + dim * 3) % total.max(1);
            let unit = (rank as f64 + offset) / total.max(1) as f64;
            bound.min + unit * (bound.max - bound.min)
        })
        .collect()
}

pub(super) fn random_normalized(dimensions: usize, rng: &mut ChaCha12Rng) -> Vec<f64> {
    (0..dimensions).map(|_| rng.gen::<f64>()).collect()
}

pub(super) fn midpoint_normalized(dimensions: usize) -> Vec<f64> {
    vec![0.5; dimensions]
}

pub(super) fn local_normalized(center: &[f64], rng: &mut ChaCha12Rng) -> Vec<f64> {
    center
        .iter()
        .map(|value| (value + rng.gen_range(-0.15..=0.15)).clamp(0.0, 1.0))
        .collect()
}

pub(super) fn denormalize_parameters(normalized: &[f64], bounds: &[ParameterBound]) -> Vec<f64> {
    normalized
        .iter()
        .zip(bounds.iter())
        .map(|(value, bound)| bound.min + value.clamp(0.0, 1.0) * (bound.max - bound.min))
        .collect()
}
