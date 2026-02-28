use traj_core::centers::{center_of_coords, center_of_selection};
use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::correlators::multi_tau::MultiTauBuffer;
use crate::correlators::ring::RingBuffer;
use crate::correlators::{LagMode, LagSettings};
use crate::executor::{Device, Plan, PlanOutput};
#[cfg(feature = "cuda")]
use crate::plans::analysis::common::groups_to_csr;
use crate::plans::ReferenceMode;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuBufferF32, GpuContext, GpuGroups};

pub struct XcorrPlan {
    sel_a: Selection,
    sel_b: Selection,
    mass_weighted: bool,
    reference_mode: ReferenceMode,
    ref_a: Option<[f64; 3]>,
    ref_b: Option<[f64; 3]>,
    lag: LagSettings,
    frames_hint: Option<usize>,
    resolved_mode: LagMode,
    sample: Vec<f32>,
    lags: Vec<usize>,
    acc: Vec<f64>,
    multi_tau: Option<MultiTauBuffer>,
    ring: Option<RingBuffer>,
    series_a: Vec<f32>,
    series_b: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<XcorrGpuState>,
}

#[cfg(feature = "cuda")]
struct XcorrGpuState {
    ctx: GpuContext,
    groups: GpuGroups,
    masses: GpuBufferF32,
}

impl XcorrPlan {
    pub fn new(
        sel_a: Selection,
        sel_b: Selection,
        reference_mode: ReferenceMode,
        mass_weighted: bool,
    ) -> Self {
        Self {
            sel_a,
            sel_b,
            mass_weighted,
            reference_mode,
            ref_a: None,
            ref_b: None,
            lag: LagSettings::default(),
            frames_hint: None,
            resolved_mode: LagMode::Auto,
            sample: vec![0.0f32; 6],
            lags: Vec::new(),
            acc: Vec::new(),
            multi_tau: None,
            ring: None,
            series_a: Vec::new(),
            series_b: Vec::new(),
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

impl Plan for XcorrPlan {
    fn name(&self) -> &'static str {
        "xcorr"
    }

    fn set_frames_hint(&mut self, n_frames: Option<usize>) {
        self.frames_hint = n_frames;
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.ref_a = None;
        self.ref_b = None;
        self.sample.fill(0.0);
        self.lags.clear();
        self.acc.clear();
        self.multi_tau = None;
        self.ring = None;
        self.series_a.clear();
        self.series_b.clear();

        if matches!(self.reference_mode, ReferenceMode::Topology) {
            let positions0 = system
                .positions0
                .as_ref()
                .ok_or_else(|| TrajError::Mismatch("no topology reference coords".into()))?;
            let masses = &system.atoms.mass;
            self.ref_a = Some(center_of_coords(
                positions0,
                &self.sel_a.indices,
                masses,
                self.mass_weighted,
            ));
            self.ref_b = Some(center_of_coords(
                positions0,
                &self.sel_b.indices,
                masses,
                self.mass_weighted,
            ));
        }

        let mut resolved = self.lag.mode;
        if resolved == LagMode::Auto {
            resolved = LagMode::MultiTau;
        }
        self.resolved_mode = resolved;
        match self.resolved_mode {
            LagMode::MultiTau => {
                let buffer =
                    MultiTauBuffer::new(1, 6, self.lag.multi_tau_m, self.lag.multi_tau_max_levels);
                self.lags = buffer.out_lags().to_vec();
                self.acc = vec![0.0f64; self.lags.len()];
                self.multi_tau = Some(buffer);
            }
            LagMode::Ring => {
                let max_lag = self.lag.ring_max_lag_capped(1, 6, 4);
                let buffer = RingBuffer::new(1, 6, max_lag);
                self.lags = (1..=max_lag).collect();
                self.acc = vec![0.0f64; self.lags.len()];
                self.ring = Some(buffer);
            }
            LagMode::Fft => {
                let capacity = self.frames_hint.unwrap_or(0).saturating_mul(3);
                self.series_a = Vec::with_capacity(capacity);
                self.series_b = Vec::with_capacity(capacity);
            }
            LagMode::Auto => {}
        }

        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = device {
                let groups = vec![
                    self.sel_a
                        .indices
                        .iter()
                        .map(|v| *v as usize)
                        .collect::<Vec<_>>(),
                    self.sel_b
                        .indices
                        .iter()
                        .map(|v| *v as usize)
                        .collect::<Vec<_>>(),
                ];
                let (offsets, indices, max_len) = groups_to_csr(&groups);
                let groups = ctx.groups(&offsets, &indices, max_len)?;
                let masses: Vec<f32> = system.atoms.mass.iter().map(|m| *m).collect();
                let masses = ctx.upload_f32(&masses)?;
                self.gpu = Some(XcorrGpuState {
                    ctx: ctx.clone(),
                    groups,
                    masses,
                });
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = device;

        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.sel_a.indices.is_empty() || self.sel_b.indices.is_empty() {
            return Ok(());
        }
        let masses = &system.atoms.mass;
        let _ = chunk.n_atoms;
        if self.ref_a.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0) {
            if chunk.n_frames == 0 {
                return Ok(());
            }
            self.ref_a = Some(center_of_selection(
                chunk,
                0,
                &self.sel_a.indices,
                masses,
                self.mass_weighted,
            ));
            self.ref_b = Some(center_of_selection(
                chunk,
                0,
                &self.sel_b.indices,
                masses,
                self.mass_weighted,
            ));
        }
        let ref_a = self
            .ref_a
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;
        let ref_b = self
            .ref_b
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;

        if self.resolved_mode == LagMode::Fft {
            #[cfg(feature = "cuda")]
            if let Some(gpu) = &self.gpu {
                let coords = convert_coords(&chunk.coords);
                let n_atoms = if chunk.n_frames > 0 {
                    chunk.coords.len() / chunk.n_frames
                } else {
                    0
                };
                let com = gpu.ctx.group_com(
                    &coords,
                    n_atoms,
                    chunk.n_frames,
                    &gpu.groups,
                    &gpu.masses,
                    1.0,
                )?;
                for frame in 0..chunk.n_frames {
                    if self.ref_a.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0)
                    {
                        let a = com[frame * 2];
                        let b = com[frame * 2 + 1];
                        self.ref_a = Some([a.x as f64, a.y as f64, a.z as f64]);
                        self.ref_b = Some([b.x as f64, b.y as f64, b.z as f64]);
                    }
                    let ref_a = self
                        .ref_a
                        .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;
                    let ref_b = self
                        .ref_b
                        .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;
                    let a = com[frame * 2];
                    let b = com[frame * 2 + 1];
                    self.series_a.push(a.x as f32 - ref_a[0] as f32);
                    self.series_a.push(a.y as f32 - ref_a[1] as f32);
                    self.series_a.push(a.z as f32 - ref_a[2] as f32);
                    self.series_b.push(b.x as f32 - ref_b[0] as f32);
                    self.series_b.push(b.y as f32 - ref_b[1] as f32);
                    self.series_b.push(b.z as f32 - ref_b[2] as f32);
                }
                return Ok(());
            }
            for frame in 0..chunk.n_frames {
                if self.ref_a.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0) {
                    self.ref_a = Some(center_of_selection(
                        chunk,
                        frame,
                        &self.sel_a.indices,
                        masses,
                        self.mass_weighted,
                    ));
                    self.ref_b = Some(center_of_selection(
                        chunk,
                        frame,
                        &self.sel_b.indices,
                        masses,
                        self.mass_weighted,
                    ));
                }
                let ref_a = self
                    .ref_a
                    .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;
                let ref_b = self
                    .ref_b
                    .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;
                let com_a = center_of_selection(
                    chunk,
                    frame,
                    &self.sel_a.indices,
                    masses,
                    self.mass_weighted,
                );
                let com_b = center_of_selection(
                    chunk,
                    frame,
                    &self.sel_b.indices,
                    masses,
                    self.mass_weighted,
                );
                self.series_a.push((com_a[0] - ref_a[0]) as f32);
                self.series_a.push((com_a[1] - ref_a[1]) as f32);
                self.series_a.push((com_a[2] - ref_a[2]) as f32);
                self.series_b.push((com_b[0] - ref_b[0]) as f32);
                self.series_b.push((com_b[1] - ref_b[1]) as f32);
                self.series_b.push((com_b[2] - ref_b[2]) as f32);
            }
            return Ok(());
        }

        for frame in 0..chunk.n_frames {
            let com_a = center_of_selection(
                chunk,
                frame,
                &self.sel_a.indices,
                masses,
                self.mass_weighted,
            );
            let com_b = center_of_selection(
                chunk,
                frame,
                &self.sel_b.indices,
                masses,
                self.mass_weighted,
            );
            let da = [
                (com_a[0] - ref_a[0]) as f32,
                (com_a[1] - ref_a[1]) as f32,
                (com_a[2] - ref_a[2]) as f32,
            ];
            let db = [
                (com_b[0] - ref_b[0]) as f32,
                (com_b[1] - ref_b[1]) as f32,
                (com_b[2] - ref_b[2]) as f32,
            ];
            self.sample[0] = da[0];
            self.sample[1] = da[1];
            self.sample[2] = da[2];
            self.sample[3] = db[0];
            self.sample[4] = db[1];
            self.sample[5] = db[2];

            if let Some(buffer) = self.multi_tau.as_mut() {
                let acc = &mut self.acc;
                buffer.update(&self.sample, |out_idx, current, old| {
                    let mut dot = 0.0f64;
                    for i in 0..3 {
                        dot += current[i] as f64 * old[3 + i] as f64;
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
                    for i in 0..3 {
                        dot += current[i] as f64 * old[3 + i] as f64;
                    }
                    acc[idx] += dot;
                });
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.resolved_mode == LagMode::Fft {
            let n_frames = self.series_a.len() / 3;
            if n_frames < 2 {
                return Ok(PlanOutput::TimeSeries {
                    time: Vec::new(),
                    data: Vec::new(),
                    rows: 0,
                    cols: 0,
                });
            }
            let ndframe = n_frames - 1;
            #[cfg(feature = "cuda")]
            if let Some(gpu) = &self.gpu {
                let (out, n_diff) =
                    gpu.ctx
                        .xcorr_time_lag(&self.series_a, &self.series_b, n_frames, ndframe)?;
                let mut time = Vec::with_capacity(ndframe);
                let mut data = Vec::with_capacity(ndframe);
                for lag in 1..=ndframe {
                    time.push(lag as f32);
                    let count = n_diff[lag] as f64;
                    let val = if count > 0.0 {
                        (out[lag] as f64 / count) as f32
                    } else {
                        0.0
                    };
                    data.push(val);
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
            return Err(TrajError::Unsupported("xcorr fft requires cuda".into()));
        }

        let mut data = Vec::with_capacity(self.lags.len());
        if let Some(buffer) = &self.multi_tau {
            let n_pairs = buffer.n_pairs();
            for (i, _lag) in self.lags.iter().enumerate() {
                let pairs = n_pairs[i] as f64;
                let val = if pairs > 0.0 {
                    (self.acc[i] / pairs) as f32
                } else {
                    0.0
                };
                data.push(val);
            }
        } else if let Some(buffer) = &self.ring {
            let n_pairs = buffer.n_pairs();
            for (i, _lag) in self.lags.iter().enumerate() {
                let pairs = n_pairs[i + 1] as f64;
                let val = if pairs > 0.0 {
                    (self.acc[i] / pairs) as f32
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
