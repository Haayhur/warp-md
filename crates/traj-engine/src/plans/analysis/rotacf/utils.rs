use std::collections::HashSet;

use traj_core::error::{TrajError, TrajResult};
use traj_core::system::System;

use crate::plans::analysis::grouping::GroupMap;
use crate::plans::analysis::msd::DtDecimation;

use super::OrientationSpec;

pub(super) fn lag_allowed(lag: usize, dec: Option<DtDecimation>) -> bool {
    if let Some(dec) = dec {
        if lag > dec.cut2 && (lag % dec.stride2) != 0 {
            return false;
        }
        if lag > dec.cut1 && (lag % dec.stride1) != 0 {
            return false;
        }
    }
    true
}

pub(super) fn finalize_rotacf(
    lags: &[usize],
    acc: &[f64],
    counts: &[u64],
    dt0: f64,
    cols: usize,
    p2_legendre: bool,
) -> (Vec<f32>, Vec<f32>) {
    let n_types = cols / 2 - 1;
    let mut time = Vec::with_capacity(lags.len() + 1);
    let mut data = vec![0.0f32; (lags.len() + 1) * cols];
    time.push(0.0);
    for t in 0..(n_types + 1) {
        data[t] = 1.0;
        data[(n_types + 1) + t] = 1.0;
    }
    for (idx, &lag) in lags.iter().enumerate() {
        time.push((dt0 * lag as f64) as f32);
        let count = counts.get(idx).copied().unwrap_or(0) as f64;
        if count == 0.0 {
            continue;
        }
        let base = idx * cols;
        let out_base = (idx + 1) * cols;
        for t in 0..(n_types + 1) {
            let p1 = acc[base + t] / count;
            let mut p2 = acc[base + (n_types + 1) + t] / count;
            if p2_legendre {
                p2 = 1.5 * p2 - 0.5;
            }
            data[out_base + t] = p1 as f32;
            data[out_base + (n_types + 1) + t] = p2 as f32;
        }
    }
    (time, data)
}

pub(super) fn resolve_orientation(
    spec: &OrientationSpec,
    groups: &GroupMap,
    _system: &System,
) -> TrajResult<Vec<[usize; 3]>> {
    let mut anchors = Vec::with_capacity(groups.n_groups());
    match spec {
        OrientationSpec::PlaneIndices(idxs) => {
            if idxs.iter().any(|&idx| idx == 0) {
                return Err(TrajError::Mismatch(
                    "orientation indices are 1-based and must be >= 1".into(),
                ));
            }
            for atoms in &groups.groups {
                if atoms.len() < idxs[0].max(idxs[1]).max(idxs[2]) {
                    return Err(TrajError::Mismatch("orientation index out of range".into()));
                }
                anchors.push([atoms[idxs[0] - 1], atoms[idxs[1] - 1], atoms[idxs[2] - 1]]);
            }
        }
        OrientationSpec::VectorIndices(idxs) => {
            if idxs.iter().any(|&idx| idx == 0) {
                return Err(TrajError::Mismatch(
                    "orientation indices are 1-based and must be >= 1".into(),
                ));
            }
            for atoms in &groups.groups {
                if atoms.len() < idxs[0].max(idxs[1]) {
                    return Err(TrajError::Mismatch("orientation index out of range".into()));
                }
                anchors.push([atoms[idxs[0] - 1], atoms[idxs[1] - 1], atoms[idxs[1] - 1]]);
            }
        }
        OrientationSpec::PlaneSelections(sels) => {
            let sets: Vec<HashSet<u32>> = sels
                .iter()
                .map(|sel| sel.indices.iter().copied().collect())
                .collect();
            for atoms in &groups.groups {
                let a = pick_atom(atoms, &sets[0])?;
                let b = pick_atom(atoms, &sets[1])?;
                let c = pick_atom(atoms, &sets[2])?;
                anchors.push([a, b, c]);
            }
        }
        OrientationSpec::VectorSelections(sels) => {
            let sets: Vec<HashSet<u32>> = sels
                .iter()
                .map(|sel| sel.indices.iter().copied().collect())
                .collect();
            for atoms in &groups.groups {
                let a = pick_atom(atoms, &sets[0])?;
                let b = pick_atom(atoms, &sets[1])?;
                anchors.push([a, b, b]);
            }
        }
    }
    Ok(anchors)
}

fn pick_atom(atoms: &[usize], set: &HashSet<u32>) -> TrajResult<usize> {
    for &atom in atoms {
        if set.contains(&(atom as u32)) {
            return Ok(atom);
        }
    }
    Err(TrajError::Mismatch(
        "orientation selection missing atom in group".into(),
    ))
}

pub(super) fn cross_unit(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    let x = a[1] * b[2] - a[2] * b[1];
    let y = a[2] * b[0] - a[0] * b[2];
    let z = a[0] * b[1] - a[1] * b[0];
    let norm = (x * x + y * y + z * z).sqrt();
    if norm == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [x / norm, y / norm, z / norm]
    }
}

pub(super) fn unit(v: [f64; 3]) -> [f64; 3] {
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if norm == 0.0 {
        [0.0, 0.0, 0.0]
    } else {
        [v[0] / norm, v[1] / norm, v[2] / norm]
    }
}
