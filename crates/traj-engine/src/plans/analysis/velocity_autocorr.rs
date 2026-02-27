use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::correlators::multi_tau::MultiTauBuffer;
use crate::correlators::ring::RingBuffer;
use crate::correlators::{LagMode, LagSettings};
use crate::executor::{Device, Plan, PlanOutput};

#[cfg(feature = "cuda")]
use traj_gpu::GpuContext;

pub struct VelocityAutoCorrPlan {
    selection: Selection,
    lag: LagSettings,
    frames_hint: Option<usize>,
    resolved_mode: LagMode,
    n_sel: usize,
    sample: Vec<f32>,
    lags: Vec<usize>,
    acc: Vec<f64>,
    multi_tau: Option<MultiTauBuffer>,
    ring: Option<RingBuffer>,
    series: Vec<f32>,
    prev_coords: Vec<[f32; 3]>,
    prev_time: Option<f64>,
    has_prev: bool,
    time_scale: f64,
    normalize: bool,
    include_zero: bool,
    zero_sum: f64,
    zero_count: u64,
    #[cfg(feature = "cuda")]
    gpu: Option<GpuContext>,
}

impl VelocityAutoCorrPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            lag: LagSettings::default(),
            frames_hint: None,
            resolved_mode: LagMode::Auto,
            n_sel: 0,
            sample: Vec::new(),
            lags: Vec::new(),
            acc: Vec::new(),
            multi_tau: None,
            ring: None,
            series: Vec::new(),
            prev_coords: Vec::new(),
            prev_time: None,
            has_prev: false,
            time_scale: 1.0,
            normalize: false,
            include_zero: false,
            zero_sum: 0.0,
            zero_count: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_lag_mode(mut self, mode: LagMode) -> Self {
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

    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    pub fn with_include_zero(mut self, include_zero: bool) -> Self {
        self.include_zero = include_zero;
        self
    }
}

impl Plan for VelocityAutoCorrPlan {
    fn name(&self) -> &'static str {
        "velocityautocorr"
    }

    fn set_frames_hint(&mut self, n_frames: Option<usize>) {
        self.frames_hint = n_frames;
    }

    fn init(&mut self, _system: &System, device: &Device) -> TrajResult<()> {
        self.n_sel = self.selection.indices.len();
        self.sample.clear();
        self.sample.resize(self.n_sel * 3, 0.0);
        self.lags.clear();
        self.acc.clear();
        self.multi_tau = None;
        self.ring = None;
        self.series.clear();
        self.prev_coords.clear();
        self.prev_coords.resize(self.n_sel, [0.0; 3]);
        self.prev_time = None;
        self.has_prev = false;
        self.zero_sum = 0.0;
        self.zero_count = 0;

        if self.n_sel == 0 {
            self.resolved_mode = self.lag.mode;
            return Ok(());
        }

        let mut resolved = self.lag.mode;
        if resolved == LagMode::Auto {
            resolved = LagMode::MultiTau;
        }
        self.resolved_mode = resolved;

        match self.resolved_mode {
            LagMode::MultiTau => {
                let buffer = MultiTauBuffer::new(
                    self.n_sel,
                    3,
                    self.lag.multi_tau_m,
                    self.lag.multi_tau_max_levels,
                );
                self.lags = buffer.out_lags().to_vec();
                self.acc = vec![0.0f64; self.lags.len()];
                self.multi_tau = Some(buffer);
            }
            LagMode::Ring => {
                let max_lag = self.lag.ring_max_lag_capped(self.n_sel, 3, 4);
                let buffer = RingBuffer::new(self.n_sel, 3, max_lag);
                self.lags = (1..=max_lag).collect();
                self.acc = vec![0.0f64; self.lags.len()];
                self.ring = Some(buffer);
            }
            LagMode::Fft => {
                let capacity = self
                    .frames_hint
                    .unwrap_or(0)
                    .saturating_mul(self.n_sel)
                    .saturating_mul(3);
                self.series = Vec::with_capacity(capacity);
            }
            LagMode::Auto => {}
        }

        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = device {
                self.gpu = Some(ctx.clone());
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = device;

        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.n_sel == 0 {
            return Ok(());
        }
        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        if self.prev_coords.len() != sel.len() {
            self.prev_coords.resize(sel.len(), [0.0; 3]);
            self.has_prev = false;
            self.prev_time = None;
        }

        for frame in 0..chunk.n_frames {
            let time = chunk
                .time_ps
                .as_ref()
                .and_then(|times| times.get(frame))
                .map(|t| *t as f64);
            let mut dt = match (self.prev_time, time) {
                (Some(prev), Some(cur)) if cur > prev => (cur - prev) * self.time_scale,
                _ => 1.0 * self.time_scale,
            };
            if !dt.is_finite() || dt == 0.0 {
                dt = 1.0 * self.time_scale;
            }

            let had_prev = self.has_prev;
            if had_prev {
                for (i, &idx) in sel.iter().enumerate() {
                    let p = chunk.coords[frame * n_atoms + idx as usize];
                    let prev = self.prev_coords[i];
                    let base = i * 3;
                    self.sample[base] = (p[0] as f64 - prev[0] as f64) as f32 / dt as f32;
                    self.sample[base + 1] = (p[1] as f64 - prev[1] as f64) as f32 / dt as f32;
                    self.sample[base + 2] = (p[2] as f64 - prev[2] as f64) as f32 / dt as f32;
                    self.prev_coords[i] = [p[0], p[1], p[2]];
                }
            } else {
                for (i, &idx) in sel.iter().enumerate() {
                    let p = chunk.coords[frame * n_atoms + idx as usize];
                    self.prev_coords[i] = [p[0], p[1], p[2]];
                }
            }

            if let Some(t) = time {
                self.prev_time = Some(t);
            }
            self.has_prev = true;

            if !had_prev {
                continue;
            }

            let mut dot0 = 0.0f64;
            for val in self.sample.iter() {
                dot0 += (*val as f64) * (*val as f64);
            }
            self.zero_sum += dot0;
            self.zero_count += 1;

            if let Some(buffer) = self.multi_tau.as_mut() {
                let acc = &mut self.acc;
                buffer.update(&self.sample, |out_idx, current, old| {
                    let mut dot = 0.0f64;
                    for i in 0..current.len() {
                        dot += current[i] as f64 * old[i] as f64;
                    }
                    acc[out_idx] += dot;
                });
            }

            if let Some(buffer) = self.ring.as_mut() {
                let acc = &mut self.acc;
                buffer.update(&self.sample, |lag, current, old| {
                    if lag == 0 {
                        return;
                    }
                    let idx = lag - 1;
                    if idx >= acc.len() {
                        return;
                    }
                    let mut dot = 0.0f64;
                    for i in 0..current.len() {
                        dot += current[i] as f64 * old[i] as f64;
                    }
                    acc[idx] += dot;
                });
            }

            if self.resolved_mode == LagMode::Fft {
                self.series.extend_from_slice(&self.sample);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let n_sel = self.n_sel as f64;
        if n_sel == 0.0 {
            return Ok(PlanOutput::TimeSeries {
                time: Vec::new(),
                data: Vec::new(),
                rows: 0,
                cols: 1,
            });
        }
        let zero_value = if self.zero_count > 0 {
            self.zero_sum / self.zero_count as f64 / n_sel
        } else {
            0.0
        };
        let norm = if self.normalize && zero_value > 0.0 {
            zero_value
        } else {
            1.0
        };

        if self.resolved_mode == LagMode::Fft {
            let stride = self.n_sel * 3;
            let n_frames = if stride > 0 {
                self.series.len() / stride
            } else {
                0
            };
            if n_frames < 2 {
                return Ok(PlanOutput::TimeSeries {
                    time: Vec::new(),
                    data: Vec::new(),
                    rows: 0,
                    cols: 1,
                });
            }
            let ndframe = n_frames - 1;
            #[cfg(feature = "cuda")]
            if let Some(gpu) = &self.gpu {
                let (out, n_diff) = gpu.timecorr_series_lag(
                    &self.series,
                    &self.series,
                    n_frames,
                    self.n_sel,
                    ndframe,
                )?;
                let extra = if self.include_zero { 1 } else { 0 };
                let mut time = Vec::with_capacity(ndframe + extra);
                let mut data = Vec::with_capacity(ndframe + extra);
                if self.include_zero {
                    time.push(0.0);
                    data.push((zero_value / norm) as f32);
                }
                for lag in 1..=ndframe {
                    time.push(lag as f32);
                    let count = n_diff[lag] as f64;
                    let val = if count > 0.0 {
                        out[lag] as f64 / (count * n_sel)
                    } else {
                        0.0
                    };
                    data.push((val / norm) as f32);
                }
                return Ok(PlanOutput::TimeSeries {
                    time,
                    data,
                    rows: ndframe + extra,
                    cols: 1,
                });
            }
            #[cfg(not(feature = "cuda"))]
            let _ = ndframe;
            return Err(TrajError::Unsupported(
                "velocity autocorr fft requires cuda".into(),
            ));
        }

        let extra = if self.include_zero { 1 } else { 0 };
        let mut data = Vec::with_capacity(self.lags.len() + extra);
        if self.include_zero {
            data.push((zero_value / norm) as f32);
        }
        if let Some(buffer) = &self.multi_tau {
            let n_pairs = buffer.n_pairs();
            for (i, _lag) in self.lags.iter().enumerate() {
                let pairs = n_pairs[i] as f64;
                let val = if pairs > 0.0 {
                    self.acc[i] / (pairs * n_sel)
                } else {
                    0.0
                };
                data.push((val / norm) as f32);
            }
        } else if let Some(buffer) = &self.ring {
            let n_pairs = buffer.n_pairs();
            for (i, _lag) in self.lags.iter().enumerate() {
                let pairs = n_pairs[i + 1] as f64;
                let val = if pairs > 0.0 {
                    self.acc[i] / (pairs * n_sel)
                } else {
                    0.0
                };
                data.push((val / norm) as f32);
            }
        }
        let mut time = Vec::with_capacity(self.lags.len() + extra);
        if self.include_zero {
            time.push(0.0);
        }
        time.extend(self.lags.iter().map(|&lag| lag as f32));
        Ok(PlanOutput::TimeSeries {
            time,
            data,
            rows: self.lags.len() + extra,
            cols: 1,
        })
    }
}
