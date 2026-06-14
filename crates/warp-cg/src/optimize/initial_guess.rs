use super::utils::midpoint;
use super::{InitialGuess, InitialGuessProvider, ParameterBound};

pub(super) struct MidpointInitialGuessProvider;

impl InitialGuessProvider for MidpointInitialGuessProvider {
    fn initial_guesses(&self, bounds: &[ParameterBound]) -> Vec<InitialGuess> {
        vec![InitialGuess {
            source: "parameter_space_midpoint".to_string(),
            parameters: midpoint(bounds),
        }]
    }
}

pub(super) fn sanitize_initial_guesses(
    guesses: &[InitialGuess],
    bounds: &[ParameterBound],
    limit: usize,
) -> Vec<InitialGuess> {
    guesses
        .iter()
        .filter(|guess| guess.parameters.len() == bounds.len())
        .take(limit)
        .map(|guess| InitialGuess {
            source: guess.source.clone(),
            parameters: clamp_parameters(&guess.parameters, bounds),
        })
        .collect()
}

fn clamp_parameters(parameters: &[f64], bounds: &[ParameterBound]) -> Vec<f64> {
    parameters
        .iter()
        .zip(bounds.iter())
        .map(|(value, bound)| value.clamp(bound.min, bound.max))
        .collect()
}
