use std::collections::BTreeSet;

use crate::optimize::{EvaluationRecord, TrainingSetPolicy};

use super::settings::FailurePolicy;

pub(super) fn select_training_data(
    evaluations: &[EvaluationRecord],
    policy: &TrainingSetPolicy,
    failure_policy: FailurePolicy,
    configured_penalty: Option<f64>,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let selected = selected_indices(evaluations, policy);
    let penalty = failure_penalty(evaluations, configured_penalty);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for index in selected {
        let record = &evaluations[index];
        if let Some(objective) = record.training_objective() {
            x.push(record.normalized_parameters.clone());
            y.push(objective);
            continue;
        }
        if matches!(failure_policy, FailurePolicy::Penalize) {
            x.push(record.normalized_parameters.clone());
            y.push(penalty);
        }
    }
    (x, y)
}

fn selected_indices(evaluations: &[EvaluationRecord], policy: &TrainingSetPolicy) -> Vec<usize> {
    if evaluations.len() <= policy.max_points {
        return (0..evaluations.len()).collect();
    }
    let mut selected = BTreeSet::new();
    add_best(evaluations, policy.keep_best, &mut selected);
    add_recent(evaluations.len(), policy.keep_recent, &mut selected);
    add_diverse(evaluations, policy.keep_diverse, &mut selected);
    let mut ordered = selected.into_iter().collect::<Vec<_>>();
    if ordered.len() > policy.max_points {
        ordered.truncate(policy.max_points);
    }
    ordered
}

fn add_best(evaluations: &[EvaluationRecord], count: usize, selected: &mut BTreeSet<usize>) {
    let mut ranked = evaluations
        .iter()
        .enumerate()
        .filter_map(|(index, record)| record.training_objective().map(|value| (index, value)))
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1));
    for (index, _) in ranked.into_iter().take(count) {
        selected.insert(index);
    }
}

fn add_recent(total: usize, count: usize, selected: &mut BTreeSet<usize>) {
    for index in total.saturating_sub(count)..total {
        selected.insert(index);
    }
}

fn add_diverse(evaluations: &[EvaluationRecord], count: usize, selected: &mut BTreeSet<usize>) {
    if count == 0 || evaluations.is_empty() {
        return;
    }
    selected.insert(0);
    while selected.len() < evaluations.len() && selected.len() < count {
        let candidate = (0..evaluations.len())
            .filter(|index| !selected.contains(index))
            .max_by(|a, b| {
                min_distance(&evaluations[*a], evaluations, selected).total_cmp(&min_distance(
                    &evaluations[*b],
                    evaluations,
                    selected,
                ))
            });
        if let Some(index) = candidate {
            selected.insert(index);
        } else {
            break;
        }
    }
}

fn min_distance(
    candidate: &EvaluationRecord,
    evaluations: &[EvaluationRecord],
    selected: &BTreeSet<usize>,
) -> f64 {
    selected
        .iter()
        .map(|index| {
            distance(
                &candidate.normalized_parameters,
                &evaluations[*index].normalized_parameters,
            )
        })
        .fold(f64::INFINITY, f64::min)
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn failure_penalty(evaluations: &[EvaluationRecord], configured: Option<f64>) -> f64 {
    if let Some(value) = configured {
        return value;
    }
    let worst = evaluations
        .iter()
        .filter_map(EvaluationRecord::training_objective)
        .max_by(|a, b| a.total_cmp(b));
    worst.map_or(1.0e12, |value| value.abs().max(value) * 10.0 + 1.0)
}
