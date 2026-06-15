use super::utils::midpoint;
use super::{InitialGuess, InitialGuessProvider, ParameterBound};
use std::collections::{BTreeMap, BTreeSet};

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

pub(super) fn named_initial_guess(
    source: &str,
    parameters: &BTreeMap<String, f64>,
    bounds: &[ParameterBound],
) -> Result<Option<InitialGuess>, String> {
    if parameters.is_empty() {
        return Ok(None);
    }
    let bound_names = bounds
        .iter()
        .map(|bound| bound.name.as_str())
        .collect::<BTreeSet<_>>();
    let unknown = parameters
        .keys()
        .filter(|name| !bound_names.contains(name.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        return Err(format!(
            "optimization.initial_parameters contains unknown parameter name(s): {}",
            unknown.join(", ")
        ));
    }
    let mut guess = midpoint(bounds);
    for (idx, bound) in bounds.iter().enumerate() {
        if let Some(value) = parameters.get(&bound.name) {
            if !value.is_finite() {
                return Err(format!(
                    "optimization.initial_parameters.{} must be finite",
                    bound.name
                ));
            }
            guess[idx] = value.clamp(bound.min, bound.max);
        }
    }
    Ok(Some(InitialGuess {
        source: source.to_string(),
        parameters: guess,
    }))
}

fn clamp_parameters(parameters: &[f64], bounds: &[ParameterBound]) -> Vec<f64> {
    parameters
        .iter()
        .zip(bounds.iter())
        .map(|(value, bound)| value.clamp(bound.min, bound.max))
        .collect()
}
