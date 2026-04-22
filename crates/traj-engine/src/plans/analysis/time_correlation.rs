use crate::correlators::multi_tau::MultiTauBuffer;
use crate::correlators::ring::RingBuffer;
use crate::correlators::{DecimationMode, LagMode, LagSettings};
use crate::plans::analysis::msd::DtDecimation;

pub(crate) enum AutoLagMode {
    MultiTau,
    FftIfFits {
        frames_hint: Option<usize>,
        streams: usize,
        width: usize,
        scalar_bytes: usize,
    },
}

pub(crate) struct LagRuntimeBuffers {
    pub(crate) lags: Vec<usize>,
    pub(crate) acc: Vec<f64>,
    pub(crate) multi_tau: Option<MultiTauBuffer>,
    pub(crate) ring: Option<RingBuffer>,
}

pub(crate) fn resolve_lag_mode(lag: &LagSettings, auto_mode: AutoLagMode) -> LagMode {
    if lag.mode != LagMode::Auto {
        return lag.mode;
    }
    match auto_mode {
        AutoLagMode::MultiTau => LagMode::MultiTau,
        AutoLagMode::FftIfFits {
            frames_hint,
            streams,
            width,
            scalar_bytes,
        } => {
            if frames_hint
                .map(|n_frames| lag.fft_fits(n_frames, streams, width, scalar_bytes))
                .unwrap_or(false)
            {
                LagMode::Fft
            } else {
                LagMode::MultiTau
            }
        }
    }
}

pub(crate) fn build_lag_runtime(
    lag: &LagSettings,
    resolved_mode: LagMode,
    streams: usize,
    width: usize,
    acc_cols: usize,
    decimation_mode: Option<DecimationMode>,
) -> LagRuntimeBuffers {
    match resolved_mode {
        LagMode::MultiTau => {
            let buffer = match decimation_mode {
                Some(mode) => MultiTauBuffer::new_with_mode(
                    streams,
                    width,
                    lag.multi_tau_m,
                    lag.multi_tau_max_levels,
                    mode,
                ),
                None => {
                    MultiTauBuffer::new(streams, width, lag.multi_tau_m, lag.multi_tau_max_levels)
                }
            };
            let lags = buffer.out_lags().to_vec();
            let acc = vec![0.0f64; lags.len() * acc_cols];
            LagRuntimeBuffers {
                lags,
                acc,
                multi_tau: Some(buffer),
                ring: None,
            }
        }
        LagMode::Ring => {
            let max_lag = lag.ring_max_lag_capped(streams, width, 4);
            let buffer = RingBuffer::new(streams, width, max_lag);
            let lags: Vec<usize> = (1..=max_lag).collect();
            let acc = vec![0.0f64; lags.len() * acc_cols];
            LagRuntimeBuffers {
                lags,
                acc,
                multi_tau: None,
                ring: Some(buffer),
            }
        }
        LagMode::Fft | LagMode::Auto => LagRuntimeBuffers {
            lags: Vec::new(),
            acc: Vec::new(),
            multi_tau: None,
            ring: None,
        },
    }
}

pub(crate) fn fft_capacity(frames_hint: Option<usize>, streams: usize, width: usize) -> usize {
    frames_hint
        .unwrap_or(0)
        .saturating_mul(streams)
        .saturating_mul(width)
}

pub(crate) fn lag_allowed(lag: usize, dec: Option<DtDecimation>) -> bool {
    if let Some(dec) = dec {
        if lag > dec.cut2 {
            return (lag % dec.stride2) == 0;
        }
        if lag > dec.cut1 {
            return (lag % dec.stride1) == 0;
        }
    }
    true
}

pub(crate) fn update_time_axis(
    frame_index: usize,
    frame: usize,
    time_ps: Option<&[f32]>,
    dt0: &mut Option<f64>,
    uniform_time: &mut bool,
    last_time: &mut Option<f64>,
    eps: f64,
) {
    let time = time_ps
        .map(|times| times[frame] as f64)
        .unwrap_or(frame_index.saturating_sub(1) as f64);
    if let Some(last) = *last_time {
        let dt = time - last;
        if let Some(base_dt) = *dt0 {
            if (dt - base_dt).abs() > eps {
                *uniform_time = false;
            }
        } else if dt > 0.0 {
            *dt0 = Some(dt);
        }
    }
    *last_time = Some(time);
}

macro_rules! impl_lag_builder_methods {
    ($plan:ty) => {
        impl $plan {
            pub fn with_lag_mode(mut self, mode: crate::correlators::LagMode) -> Self {
                self.lag = self.lag.with_mode(mode);
                self
            }

            pub fn with_max_lag(mut self, max_lag: usize) -> Self {
                self.lag = self.lag.with_max_lag(max_lag);
                self
            }

            pub fn with_memory_budget_bytes(mut self, budget: usize) -> Self {
                self.lag = self.lag.with_memory_budget_bytes(budget);
                self
            }

            pub fn with_multi_tau_m(mut self, m: usize) -> Self {
                self.lag = self.lag.with_multi_tau_m(m);
                self
            }

            pub fn with_multi_tau_levels(mut self, levels: usize) -> Self {
                self.lag = self.lag.with_multi_tau_levels(levels);
                self
            }
        }
    };
}

pub(crate) use impl_lag_builder_methods;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lag_allowed_respects_two_stage_decimation() {
        let dec = DtDecimation {
            cut1: 2,
            stride1: 2,
            cut2: 6,
            stride2: 3,
        };
        assert!(lag_allowed(2, Some(dec)));
        assert!(!lag_allowed(3, Some(dec)));
        assert!(lag_allowed(4, Some(dec)));
        assert!(!lag_allowed(7, Some(dec)));
        assert!(lag_allowed(9, Some(dec)));
    }

    #[test]
    fn update_time_axis_tracks_uniform_spacing() {
        let times = [0.0f32, 2.0, 4.0];
        let mut dt0 = None;
        let mut uniform = true;
        let mut last_time = None;
        for frame in 0..times.len() {
            update_time_axis(
                frame + 1,
                frame,
                Some(&times),
                &mut dt0,
                &mut uniform,
                &mut last_time,
                1.0e-6,
            );
        }
        assert_eq!(dt0, Some(2.0));
        assert!(uniform);
        assert_eq!(last_time, Some(4.0));
    }

    #[test]
    fn update_time_axis_flags_nonuniform_spacing() {
        let times = [0.0f32, 2.0, 4.5];
        let mut dt0 = None;
        let mut uniform = true;
        let mut last_time = None;
        for frame in 0..times.len() {
            update_time_axis(
                frame + 1,
                frame,
                Some(&times),
                &mut dt0,
                &mut uniform,
                &mut last_time,
                1.0e-6,
            );
        }
        assert_eq!(dt0, Some(2.0));
        assert!(!uniform);
    }
}
