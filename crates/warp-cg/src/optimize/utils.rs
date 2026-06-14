use rand::Rng;

use super::ParameterBound;

pub(super) fn midpoint(bounds: &[ParameterBound]) -> Vec<f64> {
    bounds
        .iter()
        .map(|bound| 0.5 * (bound.min + bound.max))
        .collect()
}

pub(super) fn random_position<R: Rng + ?Sized>(bounds: &[ParameterBound], rng: &mut R) -> Vec<f64> {
    bounds
        .iter()
        .map(|bound| rng.gen_range(bound.min..=bound.max))
        .collect()
}
