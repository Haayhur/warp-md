use traj_core::error::{TrajError, TrajResult};

use crate::executor::PlanOutput;

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

pub(super) fn average_volume(sum: f64, count: usize) -> TrajResult<f64> {
    if count == 0 {
        return Err(TrajError::Mismatch(
            "conductivity requires orthorhombic box".into(),
        ));
    }
    let vol_nm3 = sum / (count as f64);
    Ok(vol_nm3 * 1.0e-27)
}
