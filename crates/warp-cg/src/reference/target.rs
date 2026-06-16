use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::parameters::{BondValueSeries, BondedValueSeries};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceTargetSet {
    pub version: u32,
    pub bin_config: ReferenceBinConfig,
    pub constraints: Vec<ReferenceDistributionTarget>,
    pub bonds: Vec<ReferenceDistributionTarget>,
    pub angles: Vec<ReferenceDistributionTarget>,
    pub dihedrals: Vec<ReferenceDistributionTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceBinConfig {
    pub bond_bin_width_nm: f64,
    pub angle_bin_width_deg: f64,
    pub dihedral_bin_width_deg: f64,
    pub bonded_max_range_nm: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReferenceTransformConfig {
    pub bond_scaling: Option<f64>,
    pub min_bond_length_nm: Option<f64>,
    pub specific_bond_lengths_nm: BTreeMap<String, f64>,
    pub rg_offset_nm: Option<f64>,
}

impl ReferenceTransformConfig {
    pub fn is_empty(&self) -> bool {
        self.bond_scaling.is_none()
            && self.min_bond_length_nm.is_none()
            && self.specific_bond_lengths_nm.is_empty()
            && self.rg_offset_nm.is_none()
    }

    pub fn apply(&self, values: &BondedValueSeries) -> BondedValueSeries {
        if self.is_empty() {
            return values.clone();
        }
        BondedValueSeries {
            constraints: transform_bond_groups(
                &values.constraints,
                "C",
                self.bond_scaling,
                self.min_bond_length_nm,
                &self.specific_bond_lengths_nm,
            ),
            bonds: transform_bond_groups(
                &values.bonds,
                "B",
                self.bond_scaling,
                self.min_bond_length_nm,
                &self.specific_bond_lengths_nm,
            ),
            angles: values.angles.clone(),
            dihedrals: values.dihedrals.clone(),
        }
    }
}

impl Default for ReferenceBinConfig {
    fn default() -> Self {
        Self {
            bond_bin_width_nm: 0.01,
            angle_bin_width_deg: 1.0,
            dihedral_bin_width_deg: 1.0,
            bonded_max_range_nm: 3.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceDistributionTarget {
    pub kind: ReferenceTermKind,
    pub label: Option<String>,
    pub beads: Vec<usize>,
    pub members: Vec<Vec<usize>>,
    pub units: String,
    pub periodic: bool,
    pub mean: f64,
    pub std: f64,
    pub samples: usize,
    pub domain: [f64; 2],
    pub bin_edges: Vec<f64>,
    pub probabilities: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceTermKind {
    Constraint,
    Bond,
    Angle,
    Dihedral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceScore {
    pub total: f64,
    pub constraints_bonds: f64,
    pub constraints: f64,
    pub bonds: f64,
    pub angles: f64,
    pub dihedrals: f64,
    pub raw_total: f64,
    pub raw_constraints: f64,
    pub raw_bonds: f64,
    pub raw_angles: f64,
    pub raw_dihedrals: f64,
    pub scoring: ReferenceScoringConfig,
    pub terms: Vec<ReferenceTermScore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceTermScore {
    pub kind: ReferenceTermKind,
    pub beads: Vec<usize>,
    pub members: Vec<Vec<usize>>,
    pub score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceScoringConfig {
    pub bonds_to_angles_factor: f64,
    pub root_sum_square: bool,
}

impl Default for ReferenceScoringConfig {
    fn default() -> Self {
        Self {
            bonds_to_angles_factor: 1.0,
            root_sum_square: false,
        }
    }
}

impl ReferenceScoringConfig {
    pub fn bonded_emd() -> Self {
        Self {
            bonds_to_angles_factor: 500.0,
            root_sum_square: true,
        }
    }
}

impl ReferenceTargetSet {
    pub fn from_values(values: &BondedValueSeries, bin_config: ReferenceBinConfig) -> Self {
        Self::from_values_with_transform(values, bin_config, &ReferenceTransformConfig::default())
    }

    pub fn from_values_with_transform(
        values: &BondedValueSeries,
        bin_config: ReferenceBinConfig,
        transform: &ReferenceTransformConfig,
    ) -> Self {
        let values = bonded_lengths_angstrom_to_nm(values);
        let values = transform.apply(&values);
        let constraints = values
            .constraints
            .iter()
            .map(|series| {
                ReferenceDistributionTarget::from_samples(
                    ReferenceTermKind::Constraint,
                    series.label.clone(),
                    vec![series.bead_i, series.bead_j],
                    bond_members(&series.members),
                    &series.values,
                    "nm",
                    false,
                    0.0,
                    bin_config.bonded_max_range_nm,
                    bin_config.bond_bin_width_nm,
                )
            })
            .collect();
        let bonds = values
            .bonds
            .iter()
            .map(|series| {
                ReferenceDistributionTarget::from_samples(
                    ReferenceTermKind::Bond,
                    series.label.clone(),
                    vec![series.bead_i, series.bead_j],
                    bond_members(&series.members),
                    &series.values,
                    "nm",
                    false,
                    0.0,
                    bin_config.bonded_max_range_nm,
                    bin_config.bond_bin_width_nm,
                )
            })
            .collect();
        let angles = values
            .angles
            .iter()
            .map(|series| {
                ReferenceDistributionTarget::from_samples(
                    ReferenceTermKind::Angle,
                    series.label.clone(),
                    vec![series.bead_i, series.bead_j, series.bead_k],
                    angle_members(&series.members),
                    &series.values_deg,
                    "deg",
                    false,
                    0.0,
                    180.0,
                    bin_config.angle_bin_width_deg,
                )
            })
            .collect();
        let dihedrals = values
            .dihedrals
            .iter()
            .map(|series| {
                ReferenceDistributionTarget::from_samples(
                    ReferenceTermKind::Dihedral,
                    series.label.clone(),
                    vec![series.bead_i, series.bead_j, series.bead_k, series.bead_l],
                    dihedral_members(&series.members),
                    &series.values_deg,
                    "deg",
                    true,
                    -180.0,
                    180.0,
                    bin_config.dihedral_bin_width_deg,
                )
            })
            .collect();
        Self {
            version: 1,
            bin_config,
            constraints,
            bonds,
            angles,
            dihedrals,
        }
    }

    pub fn compare(&self, candidate: &ReferenceTargetSet) -> ReferenceScore {
        self.compare_with_config(candidate, &ReferenceScoringConfig::default())
    }

    pub fn filter_terms(&self, terms: &[String]) -> Self {
        if terms.is_empty() {
            return self.clone();
        }
        Self {
            version: self.version,
            bin_config: self.bin_config.clone(),
            constraints: if includes_term(terms, "constraints") {
                self.constraints.clone()
            } else {
                Vec::new()
            },
            bonds: if includes_term(terms, "bonds") {
                self.bonds.clone()
            } else {
                Vec::new()
            },
            angles: if includes_term(terms, "angles") {
                self.angles.clone()
            } else {
                Vec::new()
            },
            dihedrals: if includes_term(terms, "dihedrals") {
                self.dihedrals.clone()
            } else {
                Vec::new()
            },
        }
    }

    pub fn bonded_emd(&self, candidate: &ReferenceTargetSet) -> ReferenceScore {
        self.compare_with_config(candidate, &ReferenceScoringConfig::bonded_emd())
    }

    pub fn compare_with_config(
        &self,
        candidate: &ReferenceTargetSet,
        config: &ReferenceScoringConfig,
    ) -> ReferenceScore {
        let mut terms = Vec::new();
        let raw_constraints = score_group(&self.constraints, &candidate.constraints, &mut terms);
        let raw_bonds = score_group(&self.bonds, &candidate.bonds, &mut terms);
        let raw_angles = score_group(&self.angles, &candidate.angles, &mut terms);
        let raw_dihedrals = score_group(&self.dihedrals, &candidate.dihedrals, &mut terms);
        let constraints = aggregate_category(
            &terms,
            ReferenceTermKind::Constraint,
            config.bonds_to_angles_factor,
            config.root_sum_square,
        );
        let bonds = aggregate_category(
            &terms,
            ReferenceTermKind::Bond,
            config.bonds_to_angles_factor,
            config.root_sum_square,
        );
        let angles = aggregate_category(
            &terms,
            ReferenceTermKind::Angle,
            1.0,
            config.root_sum_square,
        );
        let dihedrals = aggregate_category(
            &terms,
            ReferenceTermKind::Dihedral,
            1.0,
            config.root_sum_square,
        );
        let constraints_bonds = aggregate_bonded_category(&terms, config);
        ReferenceScore {
            total: constraints_bonds + angles + dihedrals,
            constraints_bonds,
            constraints,
            bonds,
            angles,
            dihedrals,
            raw_total: raw_constraints + raw_bonds + raw_angles + raw_dihedrals,
            raw_constraints,
            raw_bonds,
            raw_angles,
            raw_dihedrals,
            scoring: config.clone(),
            terms,
        }
    }
}

pub(crate) fn bonded_lengths_angstrom_to_nm(values: &BondedValueSeries) -> BondedValueSeries {
    BondedValueSeries {
        constraints: bond_group_lengths_angstrom_to_nm(&values.constraints),
        bonds: bond_group_lengths_angstrom_to_nm(&values.bonds),
        angles: values.angles.clone(),
        dihedrals: values.dihedrals.clone(),
    }
}

fn bond_group_lengths_angstrom_to_nm(groups: &[BondValueSeries]) -> Vec<BondValueSeries> {
    groups
        .iter()
        .map(|group| {
            let mut group = group.clone();
            group.values = group.values.iter().map(|value| value / 10.0).collect();
            group
        })
        .collect()
}

fn transform_bond_groups(
    groups: &[BondValueSeries],
    prefix: &str,
    global_scaling: Option<f64>,
    min_bond_length_nm: Option<f64>,
    specific_lengths_nm: &BTreeMap<String, f64>,
) -> Vec<BondValueSeries> {
    groups
        .iter()
        .enumerate()
        .map(|(idx, group)| {
            let scale = bond_group_scale(
                group,
                idx,
                prefix,
                global_scaling,
                min_bond_length_nm,
                specific_lengths_nm,
            );
            if (scale - 1.0).abs() <= f64::EPSILON {
                return group.clone();
            }
            let mut transformed = group.clone();
            transformed.values = group.values.iter().map(|value| value * scale).collect();
            transformed
        })
        .collect()
}

fn bond_group_scale(
    group: &BondValueSeries,
    idx: usize,
    prefix: &str,
    global_scaling: Option<f64>,
    min_bond_length_nm: Option<f64>,
    specific_lengths_nm: &BTreeMap<String, f64>,
) -> f64 {
    let mean = finite_mean(&group.values).unwrap_or(0.0);
    if let Some(scale) = global_scaling.filter(|value| value.is_finite() && *value > 0.0) {
        return scale;
    }
    if let Some(min_length) = min_bond_length_nm.filter(|value| value.is_finite() && *value > 0.0) {
        if mean > 0.0 && mean < min_length {
            return min_length / mean;
        }
    }
    let indexed_key = format!("{prefix}{}", idx + 1);
    let target = specific_lengths_nm.get(&indexed_key).or_else(|| {
        group
            .label
            .as_ref()
            .and_then(|label| specific_lengths_nm.get(label))
    });
    if let Some(target) = target.filter(|value| value.is_finite() && **value > 0.0) {
        if mean > 0.0 {
            return *target / mean;
        }
    }
    1.0
}

fn finite_mean(values: &[f64]) -> Option<f64> {
    let mut count = 0usize;
    let mut sum = 0.0;
    for value in values.iter().copied().filter(|value| value.is_finite()) {
        count += 1;
        sum += value;
    }
    (count > 0).then(|| sum / count as f64)
}

fn includes_term(terms: &[String], target: &str) -> bool {
    terms.iter().any(|term| term == target)
}

impl ReferenceDistributionTarget {
    pub fn from_samples(
        kind: ReferenceTermKind,
        label: Option<String>,
        beads: Vec<usize>,
        members: Vec<Vec<usize>>,
        samples: &[f64],
        units: &str,
        periodic: bool,
        range_min: f64,
        range_max: f64,
        bin_width: f64,
    ) -> Self {
        let bin_edges = bin_edges(range_min, range_max, bin_width);
        let probabilities = normalized_histogram(samples, &bin_edges);
        let domain = sample_domain(samples).unwrap_or([range_min, range_max]);
        let (mean, std) = if periodic {
            circular_mean_std_deg(samples)
        } else {
            linear_mean_std(samples)
        };
        Self {
            kind,
            label,
            beads,
            members,
            units: units.to_string(),
            periodic,
            mean,
            std,
            samples: samples.len(),
            domain,
            bin_edges,
            probabilities,
        }
    }

    pub fn earth_mover_distance(&self, candidate: &ReferenceDistributionTarget) -> f64 {
        if self.probabilities.is_empty() || candidate.probabilities.is_empty() {
            return f64::INFINITY;
        }
        if self.periodic {
            circular_emd_1d(
                &self.probabilities,
                &candidate.probabilities,
                self.bin_width(),
            )
        } else {
            emd_1d(
                &self.probabilities,
                &candidate.probabilities,
                self.bin_width(),
            )
        }
    }

    pub(crate) fn bin_width(&self) -> f64 {
        self.bin_edges
            .windows(2)
            .next()
            .map(|pair| pair[1] - pair[0])
            .unwrap_or(1.0)
    }
}

fn score_group(
    reference: &[ReferenceDistributionTarget],
    candidate: &[ReferenceDistributionTarget],
    terms: &mut Vec<ReferenceTermScore>,
) -> f64 {
    let mut total = 0.0;
    for target in reference {
        let score = candidate
            .iter()
            .find(|item| item.kind == target.kind && item.members == target.members)
            .map(|item| target.earth_mover_distance(item))
            .unwrap_or(f64::INFINITY);
        terms.push(ReferenceTermScore {
            kind: target.kind,
            beads: target.beads.clone(),
            members: target.members.clone(),
            score,
        });
        total += score;
    }
    total
}

fn aggregate_category(
    terms: &[ReferenceTermScore],
    kind: ReferenceTermKind,
    factor: f64,
    root_sum_square: bool,
) -> f64 {
    let scores = terms
        .iter()
        .filter(|term| term.kind == kind)
        .map(|term| term.score * factor)
        .collect::<Vec<_>>();
    if root_sum_square {
        scores.iter().map(|score| score.powi(2)).sum::<f64>().sqrt()
    } else {
        scores.iter().sum()
    }
}

fn aggregate_bonded_category(terms: &[ReferenceTermScore], config: &ReferenceScoringConfig) -> f64 {
    if config.root_sum_square {
        terms
            .iter()
            .filter(|term| {
                term.kind == ReferenceTermKind::Constraint || term.kind == ReferenceTermKind::Bond
            })
            .map(|term| (term.score * config.bonds_to_angles_factor).powi(2))
            .sum::<f64>()
            .sqrt()
    } else {
        aggregate_category(
            terms,
            ReferenceTermKind::Constraint,
            config.bonds_to_angles_factor,
            false,
        ) + aggregate_category(
            terms,
            ReferenceTermKind::Bond,
            config.bonds_to_angles_factor,
            false,
        )
    }
}

fn bond_members(members: &[[usize; 2]]) -> Vec<Vec<usize>> {
    members.iter().map(|member| member.to_vec()).collect()
}

fn angle_members(members: &[[usize; 3]]) -> Vec<Vec<usize>> {
    members.iter().map(|member| member.to_vec()).collect()
}

fn dihedral_members(members: &[[usize; 4]]) -> Vec<Vec<usize>> {
    members.iter().map(|member| member.to_vec()).collect()
}

fn bin_edges(min: f64, max: f64, width: f64) -> Vec<f64> {
    let width = if width.is_finite() && width > 0.0 {
        width
    } else {
        1.0
    };
    let bins = (((max - min) / width).ceil() as usize).max(1);
    (0..=bins).map(|idx| min + idx as f64 * width).collect()
}

fn normalized_histogram(samples: &[f64], edges: &[f64]) -> Vec<f64> {
    let bins = edges.len().saturating_sub(1);
    let mut counts = vec![0.0; bins];
    if bins == 0 {
        return counts;
    }
    for &sample in samples {
        if !sample.is_finite() || sample < edges[0] || sample > edges[bins] {
            continue;
        }
        let mut idx = edges
            .partition_point(|edge| *edge <= sample)
            .saturating_sub(1);
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1.0;
    }
    let total: f64 = counts.iter().sum();
    if total > 0.0 {
        for count in &mut counts {
            *count /= total;
        }
    }
    counts
}

fn sample_domain(samples: &[f64]) -> Option<[f64; 2]> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &sample in samples {
        if sample.is_finite() {
            min = min.min(sample);
            max = max.max(sample);
        }
    }
    min.is_finite().then_some([min, max])
}

fn linear_mean_std(samples: &[f64]) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|sample| (sample - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    (mean, variance.sqrt())
}

fn circular_mean_std_deg(samples: &[f64]) -> (f64, f64) {
    if samples.is_empty() {
        return (0.0, 0.0);
    }
    let (sin_sum, cos_sum) = samples
        .iter()
        .fold((0.0, 0.0), |(sin_sum, cos_sum), value| {
            let radians = value.to_radians();
            (sin_sum + radians.sin(), cos_sum + radians.cos())
        });
    let mean = sin_sum.atan2(cos_sum).to_degrees();
    let resultant = (sin_sum.powi(2) + cos_sum.powi(2)).sqrt() / samples.len() as f64;
    let std = (-2.0 * resultant.max(1.0e-12).ln()).sqrt().to_degrees();
    (mean, std)
}

fn emd_1d(reference: &[f64], candidate: &[f64], bin_width: f64) -> f64 {
    let bins = reference.len().min(candidate.len());
    let mut cumulative = 0.0;
    let mut distance = 0.0;
    for idx in 0..bins {
        cumulative += reference[idx] - candidate[idx];
        distance += cumulative.abs() * bin_width;
    }
    distance
}

fn circular_emd_1d(reference: &[f64], candidate: &[f64], bin_width: f64) -> f64 {
    let bins = reference.len().min(candidate.len());
    if bins == 0 {
        return f64::INFINITY;
    }
    (0..bins)
        .map(|shift| {
            let shifted = (0..bins)
                .map(|idx| candidate[(idx + shift) % bins])
                .collect::<Vec<_>>();
            emd_1d(&reference[..bins], &shifted, bin_width)
        })
        .fold(f64::INFINITY, f64::min)
}
