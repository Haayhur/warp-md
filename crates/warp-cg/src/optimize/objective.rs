use crate::parameters::BondedStats;
use crate::reference::{
    ReferenceDistributionTarget, ReferenceScore, ReferenceTargetSet, ReferenceTermKind,
};

use super::{ObjectiveEvaluation, ObjectiveEvaluator, ParameterBound};

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
        let score = self.reference.compare_swarm_cg(&candidate);
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
            .map(|target| candidate_term(target, values.next().unwrap_or(target.mean)))
            .collect(),
        angles: reference
            .angles
            .iter()
            .map(|target| candidate_term(target, values.next().unwrap_or(target.mean)))
            .collect(),
        dihedrals: reference
            .dihedrals
            .iter()
            .map(|target| candidate_term(target, values.next().unwrap_or(target.mean)))
            .collect(),
    }
}

fn candidate_term(
    reference: &ReferenceDistributionTarget,
    value: f64,
) -> ReferenceDistributionTarget {
    let samples = synthetic_samples(reference, value);
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

fn synthetic_samples(reference: &ReferenceDistributionTarget, value: f64) -> Vec<f64> {
    let count = reference.samples.max(1);
    let std = reference.std.max(reference.bin_width());
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

pub(super) fn reference_score_evaluation(score: ReferenceScore) -> ObjectiveEvaluation {
    let mut evaluation = ObjectiveEvaluation::completed(score.total);
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
