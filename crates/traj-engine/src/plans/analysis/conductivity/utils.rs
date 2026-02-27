use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::Box3;

use crate::executor::PlanOutput;
use crate::plans::analysis::msd::DtDecimation;

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

pub(super) fn build_conductivity_output(
    lags: &[usize],
    acc: &[f64],
    counts: &[u64],
    dt0: f64,
    cols: usize,
    multi: f64,
) -> PlanOutput {
    let mut time = Vec::with_capacity(lags.len());
    let mut data = vec![0.0f32; lags.len() * cols];
    for (idx, &lag) in lags.iter().enumerate() {
        time.push((dt0 * lag as f64) as f32);
        let count = counts.get(idx).copied().unwrap_or(0) as f64;
        if count == 0.0 {
            continue;
        }
        let base = idx * cols;
        for c in 0..cols {
            data[base + c] = (acc[base + c] * multi / count) as f32;
        }
    }

    PlanOutput::TimeSeries {
        time,
        data,
        rows: lags.len(),
        cols,
    }
}

pub(super) fn box_lengths(box_: &Box3) -> Option<[f64; 3]> {
    match box_ {
        Box3::Orthorhombic { lx, ly, lz } => Some([*lx as f64, *ly as f64, *lz as f64]),
        Box3::Triclinic { .. } => None,
        Box3::None => None,
    }
}

pub(super) fn average_volume(sum: f64, count: usize) -> TrajResult<f64> {
    if count == 0 {
        return Err(TrajError::Mismatch(
            "conductivity requires orthorhombic box".into(),
        ));
    }
    let vol_nm3 = sum / (count as f64);
    Ok(vol_nm3 * 1.0e-27)
}
