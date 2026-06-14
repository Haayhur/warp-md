use std::collections::HashMap;

use crate::bonded_terms::{AngleTermGroup, BondTermGroup, DihedralTermGroup};
use crate::parameters::{
    angle_deg, dihedral_deg, AngleValueSeries, BondValueSeries, DihedralValueSeries,
};

pub(crate) fn empty_bond_group_values(groups: &[BondTermGroup]) -> Vec<Vec<f64>> {
    groups.iter().map(|_| Vec::new()).collect()
}

pub(crate) fn empty_angle_group_values(groups: &[AngleTermGroup]) -> Vec<Vec<f64>> {
    groups.iter().map(|_| Vec::new()).collect()
}

pub(crate) fn empty_dihedral_group_values(groups: &[DihedralTermGroup]) -> Vec<Vec<f64>> {
    groups.iter().map(|_| Vec::new()).collect()
}

pub(crate) fn accumulate_bond_group_values(
    values: &mut [Vec<f64>],
    groups: &[BondTermGroup],
    positions: &[[f32; 3]],
) {
    for (group_idx, group) in groups.iter().enumerate() {
        for &[i, j] in &group.members {
            if i < positions.len() && j < positions.len() {
                values[group_idx].push(distance(positions[i], positions[j]));
            }
        }
    }
}

pub(crate) fn accumulate_angle_group_values(
    values: &mut [Vec<f64>],
    groups: &[AngleTermGroup],
    positions: &[[f32; 3]],
) {
    for (group_idx, group) in groups.iter().enumerate() {
        for &[i, j, k] in &group.members {
            if i < positions.len() && j < positions.len() && k < positions.len() {
                values[group_idx].push(angle_deg(positions[i], positions[j], positions[k]));
            }
        }
    }
}

pub(crate) fn accumulate_dihedral_group_values(
    values: &mut [Vec<f64>],
    groups: &[DihedralTermGroup],
    positions: &[[f32; 3]],
) {
    for (group_idx, group) in groups.iter().enumerate() {
        for &[i, j, k, l] in &group.members {
            if i < positions.len()
                && j < positions.len()
                && k < positions.len()
                && l < positions.len()
            {
                values[group_idx].push(dihedral_deg(
                    positions[i],
                    positions[j],
                    positions[k],
                    positions[l],
                ));
            }
        }
    }
}

pub(crate) fn bond_group_series(
    groups: &[BondTermGroup],
    values: Vec<Vec<f64>>,
) -> Vec<BondValueSeries> {
    groups
        .iter()
        .zip(values)
        .filter_map(|(group, values)| {
            let first = group.members.first()?;
            Some(BondValueSeries {
                label: group.label.clone(),
                members: group.members.clone(),
                bead_i: first[0],
                bead_j: first[1],
                values,
            })
        })
        .collect()
}

pub(crate) fn angle_group_series(
    groups: &[AngleTermGroup],
    values: Vec<Vec<f64>>,
) -> Vec<AngleValueSeries> {
    groups
        .iter()
        .zip(values)
        .filter_map(|(group, values)| {
            let first = group.members.first()?;
            Some(AngleValueSeries {
                label: group.label.clone(),
                members: group.members.clone(),
                bead_i: first[0],
                bead_j: first[1],
                bead_k: first[2],
                values_deg: values,
            })
        })
        .collect()
}

pub(crate) fn dihedral_group_series(
    groups: &[DihedralTermGroup],
    values: Vec<Vec<f64>>,
) -> Vec<DihedralValueSeries> {
    groups
        .iter()
        .zip(values)
        .filter_map(|(group, values)| {
            let first = group.members.first()?;
            Some(DihedralValueSeries {
                label: group.label.clone(),
                members: group.members.clone(),
                bead_i: first[0],
                bead_j: first[1],
                bead_k: first[2],
                bead_l: first[3],
                values_deg: values,
            })
        })
        .collect()
}

pub(crate) fn single_bond_values(
    groups: &[BondTermGroup],
    values: Vec<Vec<f64>>,
) -> HashMap<(usize, usize), Vec<f64>> {
    groups
        .iter()
        .zip(values)
        .filter_map(|(group, values)| {
            (group.members.len() == 1).then(|| {
                let [i, j] = group.members[0];
                let key = if i <= j { (i, j) } else { (j, i) };
                (key, values)
            })
        })
        .collect()
}

pub(crate) fn single_angle_values(
    groups: &[AngleTermGroup],
    values: Vec<Vec<f64>>,
) -> HashMap<(usize, usize, usize), Vec<f64>> {
    groups
        .iter()
        .zip(values)
        .filter_map(|(group, values)| {
            (group.members.len() == 1).then(|| {
                let [i, j, k] = group.members[0];
                ((i, j, k), values)
            })
        })
        .collect()
}

pub(crate) fn single_dihedral_values(
    groups: &[DihedralTermGroup],
    values: Vec<Vec<f64>>,
) -> HashMap<(usize, usize, usize, usize), Vec<f64>> {
    groups
        .iter()
        .zip(values)
        .filter_map(|(group, values)| {
            (group.members.len() == 1).then(|| {
                let [i, j, k, l] = group.members[0];
                ((i, j, k, l), values)
            })
        })
        .collect()
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f64 {
    let dx = f64::from(a[0] - b[0]);
    let dy = f64::from(a[1] - b[1]);
    let dz = f64::from(a[2] - b[2]);
    (dx * dx + dy * dy + dz * dz).sqrt()
}
