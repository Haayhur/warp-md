use traj_core::error::TrajResult;

use super::utils::{lag_allowed, msd_cols};
use super::DtDecimation;

pub(super) fn msd_fft(
    series: &[f32],
    n_frames: usize,
    n_groups: usize,
    type_ids: &[usize],
    type_counts: &[usize],
    axis: Option<[f64; 3]>,
    dt_decimation: Option<DtDecimation>,
) -> TrajResult<(Vec<usize>, Vec<f64>, Vec<u64>)> {
    let n_types = type_counts.len();
    let cols = msd_cols(axis, n_types);
    let mut lags = Vec::with_capacity(n_frames - 1);
    for lag in 1..n_frames {
        if lag_allowed(lag, dt_decimation) {
            lags.push(lag);
        }
    }

    if lags.is_empty() {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    let mut acc = vec![0.0f64; lags.len() * cols];
    let mut counts = vec![0u64; lags.len()];
    for (idx, &lag) in lags.iter().enumerate() {
        counts[idx] = (n_frames - lag) as u64;
    }

    let n_groups_f = n_groups as f64;
    for g in 0..n_groups {
        let mut x = vec![0.0f32; n_frames];
        let mut y = vec![0.0f32; n_frames];
        let mut z = vec![0.0f32; n_frames];
        for t in 0..n_frames {
            let base = t * n_groups * 3 + g * 3;
            x[t] = series[base];
            y[t] = series[base + 1];
            z[t] = series[base + 2];
        }
        let (msd_x, msd_y, msd_z, msd_axis) = msd_fft_series(&x, &y, &z, axis)?;
        let type_id = type_ids[g];
        let type_count = type_counts[type_id] as f64;
        for (out_idx, &lag) in lags.iter().enumerate() {
            let idx = lag - 1;
            let base = out_idx * cols;
            let block = n_types + 1;
            let mut offset = base + type_id;
            acc[offset] += msd_x[idx] / type_count;
            acc[offset + block] += msd_y[idx] / type_count;
            acc[offset + 2 * block] += msd_z[idx] / type_count;
            if let Some(ref axis_vals) = msd_axis {
                acc[offset + 3 * block] += axis_vals[idx] / type_count;
                acc[offset + 4 * block] += (msd_x[idx] + msd_y[idx] + msd_z[idx]) / type_count;
            } else {
                acc[offset + 3 * block] += (msd_x[idx] + msd_y[idx] + msd_z[idx]) / type_count;
            }

            offset = base + n_types;
            acc[offset] += msd_x[idx] / n_groups_f;
            acc[offset + block] += msd_y[idx] / n_groups_f;
            acc[offset + 2 * block] += msd_z[idx] / n_groups_f;
            if let Some(ref axis_vals) = msd_axis {
                acc[offset + 3 * block] += axis_vals[idx] / n_groups_f;
                acc[offset + 4 * block] += (msd_x[idx] + msd_y[idx] + msd_z[idx]) / n_groups_f;
            } else {
                acc[offset + 3 * block] += (msd_x[idx] + msd_y[idx] + msd_z[idx]) / n_groups_f;
            }
        }
    }

    Ok((lags, acc, counts))
}

fn msd_fft_series(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    axis: Option<[f64; 3]>,
) -> TrajResult<(Vec<f64>, Vec<f64>, Vec<f64>, Option<Vec<f64>>)> {
    let msd_x = msd_from_series(x)?;
    let msd_y = msd_from_series(y)?;
    let msd_z = msd_from_series(z)?;
    let msd_axis = if let Some(a) = axis {
        let mut p = vec![0.0f32; x.len()];
        for i in 0..x.len() {
            p[i] = (x[i] as f64 * a[0] + y[i] as f64 * a[1] + z[i] as f64 * a[2]) as f32;
        }
        Some(msd_from_series(&p)?)
    } else {
        None
    };
    Ok((msd_x, msd_y, msd_z, msd_axis))
}

fn msd_from_series(series: &[f32]) -> TrajResult<Vec<f64>> {
    let n = series.len();
    if n < 2 {
        return Ok(Vec::new());
    }
    let mut r2 = vec![0.0f64; n + 1];
    for i in 0..n {
        r2[i + 1] = r2[i] + (series[i] as f64) * (series[i] as f64);
    }
    let ac = autocorr_real(series)?;
    let mut msd = vec![0.0f64; n - 1];
    for lag in 1..n {
        let count = (n - lag) as f64;
        let sum1 = r2[n - lag];
        let sum2 = r2[n] - r2[lag];
        let val = (sum1 + sum2 - 2.0 * ac[lag] as f64) / count;
        msd[lag - 1] = val;
    }
    Ok(msd)
}

fn autocorr_real(series: &[f32]) -> TrajResult<Vec<f32>> {
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;

    let n = series.len();
    let size = (n * 2).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(size);
    let ifft = planner.plan_fft_inverse(size);
    let mut buf = vec![
        Complex {
            re: 0.0f32,
            im: 0.0f32
        };
        size
    ];
    for i in 0..n {
        buf[i].re = series[i];
    }
    fft.process(&mut buf);
    for v in &mut buf {
        let re = v.re;
        let im = v.im;
        v.re = re * re + im * im;
        v.im = 0.0;
    }
    ifft.process(&mut buf);
    let scale = 1.0 / size as f32;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = buf[i].re * scale;
    }
    Ok(out)
}
