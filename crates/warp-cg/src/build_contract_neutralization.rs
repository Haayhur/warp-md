use warp_common::charge::neutralizer_count;

use super::{IonPolicy, NeutralizationSummary, ResolvedIonSpecies};

pub(super) fn resolve_neutralization(
    policy: &IonPolicy,
    net_charge: Option<f32>,
) -> NeutralizationSummary {
    let salt_method = policy.salt_method.clone();
    if !policy.neutralize {
        return NeutralizationSummary {
            enabled: false,
            salt_method,
            counterion: None,
            counterion_count: 0,
            counterion_charge_e: None,
            cation_delta: 0,
            anion_delta: 0,
            residual_charge_e: net_charge,
        };
    }
    let Some(net_charge) = net_charge else {
        return NeutralizationSummary {
            enabled: true,
            salt_method,
            counterion: None,
            counterion_count: 0,
            counterion_charge_e: None,
            cation_delta: 0,
            anion_delta: 0,
            residual_charge_e: None,
        };
    };
    if net_charge < 0.0 {
        let count = neutralizer_count(net_charge, policy.cation_charge_e.unsigned_abs() as f32)
            .unwrap_or(0);
        let (cation_delta, anion_delta) = neutralization_deltas(&salt_method, true, count);
        let residual = net_charge
            + cation_delta as f32 * policy.cation_charge_e as f32
            + anion_delta as f32 * policy.anion_charge_e as f32;
        NeutralizationSummary {
            enabled: true,
            salt_method,
            counterion: Some(policy.cation.clone()),
            counterion_count: count,
            counterion_charge_e: Some(policy.cation_charge_e),
            cation_delta,
            anion_delta,
            residual_charge_e: Some(residual),
        }
    } else if net_charge > 0.0 {
        let count =
            neutralizer_count(net_charge, policy.anion_charge_e.unsigned_abs() as f32).unwrap_or(0);
        let (cation_delta, anion_delta) = neutralization_deltas(&salt_method, false, count);
        let residual = net_charge
            + cation_delta as f32 * policy.cation_charge_e as f32
            + anion_delta as f32 * policy.anion_charge_e as f32;
        NeutralizationSummary {
            enabled: true,
            salt_method,
            counterion: Some(policy.anion.clone()),
            counterion_count: count,
            counterion_charge_e: Some(policy.anion_charge_e),
            cation_delta,
            anion_delta,
            residual_charge_e: Some(residual),
        }
    } else {
        NeutralizationSummary {
            enabled: true,
            salt_method,
            counterion: None,
            counterion_count: 0,
            counterion_charge_e: None,
            cation_delta: 0,
            anion_delta: 0,
            residual_charge_e: Some(0.0),
        }
    }
}

fn neutralization_deltas(
    salt_method: &str,
    needs_positive_charge: bool,
    count: usize,
) -> (isize, isize) {
    let count = count as isize;
    match (salt_method, needs_positive_charge) {
        ("remove", true) => (0, -count),
        ("remove", false) => (-count, 0),
        ("mean", true) => ((count + 1) / 2, -(count / 2)),
        ("mean", false) => (-(count / 2), (count + 1) / 2),
        (_, true) => (count, 0),
        (_, false) => (0, count),
    }
}

pub(super) fn ion_charge_sum(species: &[ResolvedIonSpecies], counts: &[isize]) -> f32 {
    species
        .iter()
        .zip(counts.iter())
        .map(|(species, count)| species.charge_e as f32 * *count as f32)
        .sum()
}

pub(super) fn solvent_neutralization_deltas(
    ions: &IonPolicy,
    cations: &[ResolvedIonSpecies],
    anions: &[ResolvedIonSpecies],
    current_charge: f32,
) -> (isize, isize, Option<String>, Option<i32>) {
    if current_charge.abs() <= 1.0e-5 {
        return (0, 0, None, None);
    }
    if current_charge < 0.0 {
        let charge = representative_charge(cations).unwrap_or(ions.cation_charge_e);
        let count = neutralizer_count(current_charge, charge.unsigned_abs() as f32).unwrap_or(0);
        let (cation_delta, anion_delta) = neutralization_deltas(&ions.salt_method, true, count);
        (
            cation_delta,
            anion_delta,
            cations.first().map(|species| species.name.clone()),
            Some(charge),
        )
    } else {
        let charge = representative_charge(anions).unwrap_or(ions.anion_charge_e);
        let count = neutralizer_count(current_charge, charge.unsigned_abs() as f32).unwrap_or(0);
        let (cation_delta, anion_delta) = neutralization_deltas(&ions.salt_method, false, count);
        (
            cation_delta,
            anion_delta,
            anions.first().map(|species| species.name.clone()),
            Some(charge),
        )
    }
}

pub(super) fn representative_charge(species: &[ResolvedIonSpecies]) -> Option<i32> {
    species.first().map(|species| species.charge_e)
}

pub(super) fn apply_delta_to_ions(
    counts: &mut [isize],
    species: &[ResolvedIonSpecies],
    delta: isize,
) {
    if delta == 0 || counts.is_empty() {
        return;
    }
    let mut remaining = delta.abs();
    while remaining > 0 {
        let idx = if delta > 0 {
            most_underrepresented_ion(counts, species)
        } else {
            most_overrepresented_ion(counts, species)
        };
        counts[idx] += delta.signum();
        remaining -= 1;
    }
}

fn most_underrepresented_ion(counts: &[isize], species: &[ResolvedIonSpecies]) -> usize {
    let total = counts
        .iter()
        .map(|count| (*count).max(0))
        .sum::<isize>()
        .max(1) as f32;
    let ratio_sum = species
        .iter()
        .map(|species| species.ratio)
        .sum::<f32>()
        .max(1.0);
    species
        .iter()
        .enumerate()
        .max_by(|(left_idx, left), (right_idx, right)| {
            let left_diff = left.ratio / ratio_sum - counts[*left_idx] as f32 / total;
            let right_diff = right.ratio / ratio_sum - counts[*right_idx] as f32 / total;
            left_diff
                .partial_cmp(&right_diff)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

fn most_overrepresented_ion(counts: &[isize], species: &[ResolvedIonSpecies]) -> usize {
    let total = counts
        .iter()
        .map(|count| (*count).max(0))
        .sum::<isize>()
        .max(1) as f32;
    let ratio_sum = species
        .iter()
        .map(|species| species.ratio)
        .sum::<f32>()
        .max(1.0);
    species
        .iter()
        .enumerate()
        .max_by(|(left_idx, left), (right_idx, right)| {
            let left_diff = counts[*left_idx] as f32 / total - left.ratio / ratio_sum;
            let right_diff = counts[*right_idx] as f32 / total - right.ratio / ratio_sum;
            left_diff
                .partial_cmp(&right_diff)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}
