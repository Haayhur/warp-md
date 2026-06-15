use std::collections::BTreeMap;

use crate::parameters::BondedStats;
use crate::reference::{
    ReferenceDistributionTarget, ReferenceScore, ReferenceTargetSet, ReferenceTermKind,
};

use super::{
    ObjectiveEvaluation, ObjectiveEvaluator, ParameterBound, StructuralMetricScoringConfig,
};

pub(super) struct ParameterProxyEvaluator {
    target: Vec<f64>,
    bounds: Vec<ParameterBound>,
}

impl ParameterProxyEvaluator {
    pub(super) fn new(stats: &BondedStats, bounds: &[ParameterBound]) -> Self {
        Self {
            target: target_vector(stats),
            bounds: bounds.to_vec(),
        }
    }
}

impl ObjectiveEvaluator for ParameterProxyEvaluator {
    fn evaluate(&mut self, parameters: &[f64]) -> ObjectiveEvaluation {
        ObjectiveEvaluation::completed(parameter_proxy_objective(
            parameters,
            &self.target,
            &self.bounds,
        ))
    }
}

pub(super) struct ReferenceTargetEvaluator {
    reference: ReferenceTargetSet,
}

impl ReferenceTargetEvaluator {
    pub(super) fn new(reference: &ReferenceTargetSet) -> Self {
        Self {
            reference: reference.clone(),
        }
    }
}

impl ObjectiveEvaluator for ReferenceTargetEvaluator {
    fn evaluate(&mut self, parameters: &[f64]) -> ObjectiveEvaluation {
        let candidate = candidate_target_set(&self.reference, parameters);
        let score = self.reference.bonded_emd(&candidate);
        reference_score_evaluation(score)
    }
}

fn target_vector(stats: &BondedStats) -> Vec<f64> {
    let mut target =
        Vec::with_capacity((stats.bonds.len() + stats.angles.len() + stats.dihedrals.len()) * 2);
    for stat in &stats.bonds {
        target.push(stat.mean);
        let variance = stat.std.max(0.02).powi(2);
        target.push((1.0 / variance).clamp(1.0, 5000.0));
    }
    for stat in &stats.angles {
        target.push(stat.mean_deg.clamp(0.0, 180.0));
        let variance = stat.std_deg.max(1.0).powi(2);
        target.push((10_000.0 / variance).clamp(1.0, 500.0));
    }
    for stat in &stats.dihedrals {
        target.push(stat.mean_deg.clamp(-180.0, 180.0));
        let variance = stat.std_deg.max(1.0).powi(2);
        target.push((1_000.0 / variance).clamp(0.1, 100.0));
    }
    target
}

fn parameter_proxy_objective(params: &[f64], target: &[f64], bounds: &[ParameterBound]) -> f64 {
    params
        .iter()
        .zip(target.iter())
        .zip(bounds.iter())
        .map(|((value, target), bound)| {
            let scale = (bound.max - bound.min).max(1.0e-9);
            ((value - target) / scale).powi(2)
        })
        .sum::<f64>()
        / params.len().max(1) as f64
}

fn candidate_target_set(reference: &ReferenceTargetSet, parameters: &[f64]) -> ReferenceTargetSet {
    let mut values = parameters.iter().copied();
    ReferenceTargetSet {
        version: reference.version,
        bin_config: reference.bin_config.clone(),
        constraints: reference
            .constraints
            .iter()
            .map(|target| candidate_term(target, values.next().unwrap_or(target.mean)))
            .collect(),
        bonds: reference
            .bonds
            .iter()
            .map(|target| {
                candidate_term_with_force(
                    target,
                    values.next().unwrap_or(target.mean),
                    values.next(),
                )
            })
            .collect(),
        angles: reference
            .angles
            .iter()
            .map(|target| {
                candidate_term_with_force(
                    target,
                    values.next().unwrap_or(target.mean),
                    values.next(),
                )
            })
            .collect(),
        dihedrals: reference
            .dihedrals
            .iter()
            .map(|target| {
                candidate_term_with_force(
                    target,
                    values.next().unwrap_or(target.mean),
                    values.next(),
                )
            })
            .collect(),
    }
}

fn candidate_term(
    reference: &ReferenceDistributionTarget,
    value: f64,
) -> ReferenceDistributionTarget {
    candidate_term_with_force(reference, value, None)
}

fn candidate_term_with_force(
    reference: &ReferenceDistributionTarget,
    value: f64,
    force: Option<f64>,
) -> ReferenceDistributionTarget {
    let samples = synthetic_samples(reference, value, force);
    let (range_min, range_max, bin_width) = match reference.kind {
        ReferenceTermKind::Constraint | ReferenceTermKind::Bond => {
            let max = reference.bin_edges.last().copied().unwrap_or(3.0);
            (0.0, max, reference.bin_width())
        }
        ReferenceTermKind::Angle => (0.0, 180.0, reference.bin_width()),
        ReferenceTermKind::Dihedral => (-180.0, 180.0, reference.bin_width()),
    };
    ReferenceDistributionTarget::from_samples(
        reference.kind,
        reference.label.clone(),
        reference.beads.clone(),
        reference.members.clone(),
        &samples,
        &reference.units,
        reference.periodic,
        range_min,
        range_max,
        bin_width,
    )
}

fn synthetic_samples(
    reference: &ReferenceDistributionTarget,
    value: f64,
    force: Option<f64>,
) -> Vec<f64> {
    let count = reference.samples.max(1);
    let std = force
        .and_then(|force| synthetic_std_from_force(reference.kind, force))
        .unwrap_or(reference.std)
        .max(reference.bin_width());
    if count == 1 {
        return vec![value];
    }
    (0..count)
        .map(|idx| {
            let centered = idx as f64 - (count - 1) as f64 * 0.5;
            let scaled = centered / count.max(2) as f64;
            value + scaled * std
        })
        .collect()
}

fn synthetic_std_from_force(kind: ReferenceTermKind, force: f64) -> Option<f64> {
    if !force.is_finite() || force <= 0.0 {
        return None;
    }
    Some(match kind {
        ReferenceTermKind::Constraint => 0.0,
        ReferenceTermKind::Bond => (1.0 / force).sqrt().clamp(0.001, 10.0),
        ReferenceTermKind::Angle => (10_000.0 / force).sqrt().clamp(0.1, 180.0),
        ReferenceTermKind::Dihedral => (1_000.0 / force).sqrt().clamp(0.1, 360.0),
    })
}

pub(super) fn reference_score_evaluation(score: ReferenceScore) -> ObjectiveEvaluation {
    let mut evaluation = ObjectiveEvaluation::completed(score.total);
    evaluation
        .metrics
        .insert("constraints_bonds_emd".to_string(), score.constraints_bonds);
    evaluation
        .metrics
        .insert("constraints_emd".to_string(), score.constraints);
    evaluation
        .metrics
        .insert("bonds_emd".to_string(), score.bonds);
    evaluation
        .metrics
        .insert("angles_emd".to_string(), score.angles);
    evaluation
        .metrics
        .insert("dihedrals_emd".to_string(), score.dihedrals);
    evaluation
        .metrics
        .insert("raw_constraints_emd".to_string(), score.raw_constraints);
    evaluation
        .metrics
        .insert("raw_bonds_emd".to_string(), score.raw_bonds);
    evaluation
        .metrics
        .insert("raw_angles_emd".to_string(), score.raw_angles);
    evaluation
        .metrics
        .insert("raw_dihedrals_emd".to_string(), score.raw_dihedrals);
    evaluation.metrics.insert(
        "bonds_to_angles_factor".to_string(),
        score.scoring.bonds_to_angles_factor,
    );
    evaluation
}

pub(super) fn reference_score_evaluation_with_metrics(
    score: ReferenceScore,
    reference_metrics: &BTreeMap<String, f64>,
    candidate_metrics: &BTreeMap<String, f64>,
    metric_config: &StructuralMetricScoringConfig,
) -> ObjectiveEvaluation {
    let mut evaluation = reference_score_evaluation(score);
    let metric_objective = structural_metric_objective(
        reference_metrics,
        candidate_metrics,
        metric_config,
        &mut evaluation.metrics,
    );
    if metric_objective > 0.0 {
        let bonded_objective = evaluation.objective.unwrap_or(0.0);
        let objective = bonded_objective + metric_objective;
        evaluation.objective = Some(objective);
        evaluation
            .metrics
            .insert("objective".to_string(), objective);
        evaluation
            .metrics
            .insert("bonded_emd_objective".to_string(), bonded_objective);
        evaluation
            .metrics
            .insert("structural_metric_objective".to_string(), metric_objective);
    }
    evaluation
}

fn structural_metric_objective(
    reference_metrics: &BTreeMap<String, f64>,
    candidate_metrics: &BTreeMap<String, f64>,
    config: &StructuralMetricScoringConfig,
    output: &mut BTreeMap<String, f64>,
) -> f64 {
    let rg = structural_metric_term(
        StructuralMetricKind::Rg,
        reference_metrics,
        candidate_metrics,
        config.rg_weight,
        config.require_rg,
        config.missing_metric_penalty,
        output,
    );
    let sasa = structural_metric_term(
        StructuralMetricKind::Sasa,
        reference_metrics,
        candidate_metrics,
        config.sasa_weight,
        config.require_sasa,
        config.missing_metric_penalty,
        output,
    );
    rg + sasa
}

#[derive(Debug, Clone, Copy)]
enum StructuralMetricKind {
    Rg,
    Sasa,
}

impl StructuralMetricKind {
    fn prefix(self) -> &'static str {
        match self {
            Self::Rg => "rg",
            Self::Sasa => "sasa",
        }
    }

    fn mean_units(self) -> &'static str {
        match self {
            Self::Rg => "nm",
            Self::Sasa => "nm2",
        }
    }
}

fn structural_metric_term(
    kind: StructuralMetricKind,
    reference_metrics: &BTreeMap<String, f64>,
    candidate_metrics: &BTreeMap<String, f64>,
    weight: f64,
    required: bool,
    missing_metric_penalty: f64,
    output: &mut BTreeMap<String, f64>,
) -> f64 {
    if weight <= 0.0 {
        return 0.0;
    }
    let weighted_missing_penalty = missing_metric_penalty * weight;
    let Some(reference) = mean_metric(reference_metrics, kind) else {
        if required {
            let prefix = kind.prefix();
            output.insert(
                format!("{prefix}_missing_reference_penalty"),
                weighted_missing_penalty,
            );
            return weighted_missing_penalty;
        }
        return 0.0;
    };
    let prefix = kind.prefix();
    let units = kind.mean_units();
    output.insert(format!("{prefix}_reference_mean_{units}"), reference);
    let Some(candidate) = mean_metric(candidate_metrics, kind) else {
        output.insert(
            format!("{prefix}_missing_penalty"),
            weighted_missing_penalty,
        );
        return weighted_missing_penalty;
    };
    let scale = std_metric(reference_metrics, kind)
        .filter(|value| *value > 1.0e-12)
        .unwrap_or_else(|| fallback_metric_scale(reference, kind));
    let error = candidate - reference;
    let scaled = error / scale;
    let objective = scaled.powi(2) * weight;
    output.insert(format!("{prefix}_candidate_mean_{units}"), candidate);
    output.insert(format!("{prefix}_error_{units}"), error);
    output.insert(format!("{prefix}_scale_{units}"), scale);
    output.insert(format!("{prefix}_scaled_error"), scaled);
    output.insert(format!("{prefix}_weight"), weight);
    output.insert(format!("{prefix}_objective"), objective);
    objective
}

fn fallback_metric_scale(reference: f64, kind: StructuralMetricKind) -> f64 {
    match kind {
        StructuralMetricKind::Rg => reference.abs().mul_add(0.05, 0.0).max(0.01),
        StructuralMetricKind::Sasa => reference.abs().mul_add(0.05, 0.0).max(0.01),
    }
}

fn mean_metric(metrics: &BTreeMap<String, f64>, kind: StructuralMetricKind) -> Option<f64> {
    find_metric(metrics, |key| match kind {
        StructuralMetricKind::Rg => key == "rg_mean_nm" || key.ends_with("_rg_mean_nm"),
        StructuralMetricKind::Sasa => {
            key.ends_with("sasa_approx_mean_nm2")
                || key.ends_with("sasa_mean_nm2")
                || (key.contains("sasa") && key.ends_with("mean_nm2"))
        }
    })
}

fn std_metric(metrics: &BTreeMap<String, f64>, kind: StructuralMetricKind) -> Option<f64> {
    find_metric(metrics, |key| match kind {
        StructuralMetricKind::Rg => key == "rg_std_nm" || key.ends_with("_rg_std_nm"),
        StructuralMetricKind::Sasa => {
            key.ends_with("sasa_approx_std_nm2")
                || key.ends_with("sasa_std_nm2")
                || (key.contains("sasa") && key.ends_with("std_nm2"))
        }
    })
}

fn find_metric(metrics: &BTreeMap<String, f64>, matches: impl Fn(&str) -> bool) -> Option<f64> {
    metrics.iter().find_map(|(key, value)| {
        let normalized = normalize_metric_key(key);
        if matches(&normalized) && value.is_finite() {
            Some(*value)
        } else {
            None
        }
    })
}

fn normalize_metric_key(key: &str) -> String {
    key.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect()
}
