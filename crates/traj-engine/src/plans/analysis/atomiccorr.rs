use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::correlators::multi_tau::MultiTauBuffer;
use crate::correlators::ring::RingBuffer;
use crate::correlators::{LagMode, LagSettings};
use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::ReferenceMode;

#[cfg(feature = "cuda")]
use traj_gpu::GpuContext;

pub struct AtomicCorrPlan {
    selection: Selection,
    reference_mode: ReferenceMode,
    reference: Option<Vec<[f32; 4]>>,
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
    #[cfg(feature = "cuda")]
    gpu: Option<GpuContext>,
}

impl AtomicCorrPlan {
    pub fn new(selection: Selection, reference_mode: ReferenceMode) -> Self {
        Self {
            selection,
            reference_mode,
            reference: None,
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
}

impl Plan for AtomicCorrPlan {
    fn name(&self) -> &'static str {
        "atomiccorr"
    }

    fn set_frames_hint(&mut self, n_frames: Option<usize>) {
        self.frames_hint = n_frames;
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.reference = match self.reference_mode {
            ReferenceMode::Topology => {
                let positions0 = system
                    .positions0
                    .as_ref()
                    .ok_or_else(|| TrajError::Mismatch("no topology reference coords".into()))?;
                let mut reference = Vec::with_capacity(self.selection.indices.len());
                for &idx in self.selection.indices.iter() {
                    reference.push(positions0[idx as usize]);
                }
                Some(reference)
            }
            ReferenceMode::Frame0 => None,
        };
        self.n_sel = self.selection.indices.len();
        self.sample = vec![0.0f32; self.n_sel * 3];
        self.lags.clear();
        self.acc.clear();
        self.multi_tau = None;
        self.ring = None;
        self.series.clear();

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
            if let Device::Cuda(ctx) = _device {
                self.gpu = Some(ctx.clone());
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = _device;
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
        if self.reference.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0) {
            if chunk.n_frames == 0 {
                return Ok(());
            }
            let mut reference = Vec::with_capacity(self.selection.indices.len());
            for &idx in self.selection.indices.iter() {
                reference.push(chunk.coords[idx as usize]);
            }
            self.reference = Some(reference);
        }
        let reference = self
            .reference
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;

        for frame in 0..chunk.n_frames {
            for (i, &idx) in self.selection.indices.iter().enumerate() {
                let p = chunk.coords[frame * n_atoms + idx as usize];
                let r = reference[i];
                let base = i * 3;
                self.sample[base] = p[0] - r[0];
                self.sample[base + 1] = p[1] - r[1];
                self.sample[base + 2] = p[2] - r[2];
            }

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
                let mut time = Vec::with_capacity(ndframe);
                let mut data = Vec::with_capacity(ndframe);
                let n_sel = self.n_sel as f64;
                for lag in 1..=ndframe {
                    time.push(lag as f32);
                    let count = n_diff[lag] as f64;
                    let val = if count > 0.0 {
                        out[lag] as f64 / (count * n_sel)
                    } else {
                        0.0
                    };
                    data.push(val as f32);
                }
                return Ok(PlanOutput::TimeSeries {
                    time,
                    data,
                    rows: ndframe,
                    cols: 1,
                });
            }
            #[cfg(not(feature = "cuda"))]
            let _ = ndframe;
            return Err(TrajError::Unsupported(
                "atomiccorr fft requires cuda".into(),
            ));
        }

        let mut data = Vec::with_capacity(self.lags.len());
        let n_sel = self.n_sel as f64;
        if n_sel == 0.0 {
            return Ok(PlanOutput::TimeSeries {
                time: Vec::new(),
                data,
                rows: 0,
                cols: 1,
            });
        }
        if let Some(buffer) = &self.multi_tau {
            let n_pairs = buffer.n_pairs();
            for (i, &lag) in self.lags.iter().enumerate() {
                let pairs = n_pairs[i] as f64;
                let val = if pairs > 0.0 {
                    (self.acc[i] / (pairs * n_sel)) as f32
                } else {
                    0.0
                };
                let _ = lag;
                data.push(val);
            }
        } else if let Some(buffer) = &self.ring {
            let n_pairs = buffer.n_pairs();
            for (i, _lag) in self.lags.iter().enumerate() {
                let pairs = n_pairs[i + 1] as f64;
                let val = if pairs > 0.0 {
                    (self.acc[i] / (pairs * n_sel)) as f32
                } else {
                    0.0
                };
                data.push(val);
            }
        }
        let time = self.lags.iter().map(|&lag| lag as f32).collect();
        Ok(PlanOutput::TimeSeries {
            time,
            data,
            rows: self.lags.len(),
            cols: 1,
        })
    }
}
