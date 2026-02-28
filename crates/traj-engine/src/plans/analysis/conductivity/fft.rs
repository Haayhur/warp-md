use traj_core::error::TrajResult;

use crate::plans::analysis::msd::DtDecimation;

use super::utils::lag_allowed;

pub(super) fn conductivity_fft(
    series: &[f32],
    n_frames: usize,
    n_types: usize,
    transference: bool,
    dt_decimation: Option<DtDecimation>,
) -> TrajResult<(Vec<usize>, Vec<f64>, Vec<u64>)> {
    let streams = if transference { n_types + 1 } else { 1 };
    let cols = if transference {
        n_types * n_types + 1
    } else {
        1
    };
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

    let mut comps = vec![vec![vec![0.0f32; n_frames]; 3]; streams];
    for t in 0..n_frames {
        for s in 0..streams {
            let base = (t * streams + s) * 3;
            comps[s][0][t] = series[base];
            comps[s][1][t] = series[base + 1];
            comps[s][2][t] = series[base + 2];
        }
    }

    let compute_pair = |i: usize, j: usize| -> TrajResult<Vec<f64>> {
        let mut dot_series = vec![0.0f64; n_frames];
        for t in 0..n_frames {
            dot_series[t] = comps[i][0][t] as f64 * comps[j][0][t] as f64
                + comps[i][1][t] as f64 * comps[j][1][t] as f64
                + comps[i][2][t] as f64 * comps[j][2][t] as f64;
        }
        let mut prefix = vec![0.0f64; n_frames + 1];
        for t in 0..n_frames {
            prefix[t + 1] = prefix[t] + dot_series[t];
        }
        let xcorr_x = xcorr_real(&comps[i][0], &comps[j][0])?;
        let xcorr_y = xcorr_real(&comps[i][1], &comps[j][1])?;
        let xcorr_z = xcorr_real(&comps[i][2], &comps[j][2])?;
        let (xcorr_rx, xcorr_ry, xcorr_rz) = if i == j {
            (xcorr_x.clone(), xcorr_y.clone(), xcorr_z.clone())
        } else {
            (
                xcorr_real(&comps[j][0], &comps[i][0])?,
                xcorr_real(&comps[j][1], &comps[i][1])?,
                xcorr_real(&comps[j][2], &comps[i][2])?,
            )
        };
        let mut sums = vec![0.0f64; lags.len()];
        for (idx, &lag) in lags.iter().enumerate() {
            let sum1 = prefix[n_frames - lag];
            let sum2 = prefix[n_frames] - prefix[lag];
            let cross = xcorr_x[lag] as f64 + xcorr_y[lag] as f64 + xcorr_z[lag] as f64;
            let cross_r = xcorr_rx[lag] as f64 + xcorr_ry[lag] as f64 + xcorr_rz[lag] as f64;
            sums[idx] = sum1 + sum2 - cross - cross_r;
        }
        Ok(sums)
    };

    if transference {
        for i in 0..n_types {
            for j in i..n_types {
                let sums = compute_pair(i, j)?;
                for (idx, sum_val) in sums.into_iter().enumerate() {
                    let base = idx * cols;
                    acc[base + j + i * n_types] += sum_val;
                    if i != j {
                        acc[base + i + j * n_types] += sum_val;
                    }
                }
            }
        }
        let total = n_types;
        let sums = compute_pair(total, total)?;
        for (idx, sum_val) in sums.into_iter().enumerate() {
            let base = idx * cols;
            acc[base + cols - 1] += sum_val;
        }
    } else {
        let sums = compute_pair(0, 0)?;
        for (idx, sum_val) in sums.into_iter().enumerate() {
            let base = idx * cols;
            acc[base] += sum_val;
        }
    }

    Ok((lags, acc, counts))
}

fn xcorr_real(a: &[f32], b: &[f32]) -> TrajResult<Vec<f32>> {
    use rustfft::num_complex::Complex;
    use rustfft::FftPlanner;

    let n = a.len();
    let size = (n * 2).next_power_of_two();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(size);
    let ifft = planner.plan_fft_inverse(size);
    let mut buf_a = vec![
        Complex {
            re: 0.0f32,
            im: 0.0f32
        };
        size
    ];
    let mut buf_b = vec![
        Complex {
            re: 0.0f32,
            im: 0.0f32
        };
        size
    ];
    for i in 0..n {
        buf_a[i].re = a[i];
        buf_b[i].re = b[i];
    }
    fft.process(&mut buf_a);
    fft.process(&mut buf_b);
    for i in 0..size {
        let ar = buf_a[i].re;
        let ai = buf_a[i].im;
        let br = buf_b[i].re;
        let bi = buf_b[i].im;
        buf_a[i].re = ar * br + ai * bi;
        buf_a[i].im = ar * bi - ai * br;
    }
    ifft.process(&mut buf_a);
    let scale = 1.0 / size as f32;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = buf_a[i].re * scale;
    }
    Ok(out)
}
