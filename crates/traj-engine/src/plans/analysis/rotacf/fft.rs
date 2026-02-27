use traj_core::error::TrajResult;

use super::utils::lag_allowed;
use super::DtDecimation;

pub(super) fn rotacf_fft(
    series: &[f32],
    n_frames: usize,
    n_groups: usize,
    type_ids: &[usize],
    type_counts: &[usize],
    dt_decimation: Option<DtDecimation>,
) -> TrajResult<(Vec<usize>, Vec<f64>, Vec<u64>)> {
    let n_types = type_counts.len();
    let cols = 2 * (n_types + 1);
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
        let corr_x = autocorr_real(&x)?;
        let corr_y = autocorr_real(&y)?;
        let corr_z = autocorr_real(&z)?;
        let corr_xx = autocorr_real(&x.iter().map(|v| v * v).collect::<Vec<_>>())?;
        let corr_yy = autocorr_real(&y.iter().map(|v| v * v).collect::<Vec<_>>())?;
        let corr_zz = autocorr_real(&z.iter().map(|v| v * v).collect::<Vec<_>>())?;
        let corr_xy = autocorr_real(
            &x.iter()
                .zip(y.iter())
                .map(|(a, b)| a * b)
                .collect::<Vec<_>>(),
        )?;
        let corr_xz = autocorr_real(
            &x.iter()
                .zip(z.iter())
                .map(|(a, b)| a * b)
                .collect::<Vec<_>>(),
        )?;
        let corr_yz = autocorr_real(
            &y.iter()
                .zip(z.iter())
                .map(|(a, b)| a * b)
                .collect::<Vec<_>>(),
        )?;

        let type_id = type_ids[g];
        let type_count = type_counts[type_id] as f64;
        for (out_idx, &lag) in lags.iter().enumerate() {
            let base = out_idx * cols;
            let dot = (corr_x[lag] + corr_y[lag] + corr_z[lag]) as f64;
            let dot2 = (corr_xx[lag]
                + corr_yy[lag]
                + corr_zz[lag]
                + 2.0 * (corr_xy[lag] + corr_xz[lag] + corr_yz[lag])) as f64;
            acc[base + type_id] += dot / type_count;
            acc[base + n_types] += dot / n_groups_f;
            acc[base + (n_types + 1) + type_id] += dot2 / type_count;
            acc[base + (n_types + 1) + n_types] += dot2 / n_groups_f;
        }
    }

    Ok((lags, acc, counts))
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
