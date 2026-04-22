use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::system::System;

use crate::correlators::LagMode;
use crate::executor::{Device, Plan, PlanOutput};
#[cfg(feature = "cuda")]
use crate::plans::analysis::group_runtime::anchors_to_u32;
use crate::plans::analysis::grouping::GroupSpec;
use crate::plans::analysis::time_correlation::{
    build_lag_runtime, fft_capacity, lag_allowed, resolve_lag_mode, update_time_axis, AutoLagMode,
};

#[cfg(feature = "cuda")]
use traj_gpu::coords_as_float4;

use super::accumulate::accumulate_rotacf;
use super::fft::rotacf_fft;
use super::orientation_support::{cross_unit, finalize_rotacf, resolve_orientation, unit};
#[cfg(feature = "cuda")]
use super::RotAcfGpuState;
use super::{OrientationKind, RotAcfPlan};

impl Plan for RotAcfPlan {
    fn name(&self) -> &'static str {
        "rotacf"
    }

    fn set_frames_hint(&mut self, n_frames: Option<usize>) {
        self.frames_hint = n_frames;
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        let mut spec = GroupSpec::new(self.selection.clone(), self.group_by);
        if let Some(types) = &self.group_types {
            spec = spec.with_group_types(types.clone());
        }
        let groups = spec.build(system)?;
        let anchors = resolve_orientation(&self.orientation, &groups, system)?;
        self.n_groups = groups.n_groups();
        self.type_ids = groups.type_ids();
        self.type_counts = groups.type_counts();
        self.groups = Some(groups);
        self.anchors = Some(anchors);

        self.dt0 = None;
        self.uniform_time = true;
        self.last_time = None;
        self.frame_index = 0;
        self.samples_seen = 0;
        self.sample_f32 = vec![0.0f32; self.n_groups * 3];
        self.lags.clear();
        self.acc.clear();
        self.multi_tau = None;
        self.ring = None;
        self.series.clear();

        self.resolved_mode = resolve_lag_mode(
            &self.lag,
            AutoLagMode::FftIfFits {
                frames_hint: self.frames_hint,
                streams: self.n_groups,
                width: 3,
                scalar_bytes: 4,
            },
        );
        let cols = 2 * (self.type_counts.len() + 1);
        let runtime =
            build_lag_runtime(&self.lag, self.resolved_mode, self.n_groups, 3, cols, None);
        self.lags = runtime.lags;
        self.acc = runtime.acc;
        self.multi_tau = runtime.multi_tau;
        self.ring = runtime.ring;
        if self.resolved_mode == LagMode::Fft {
            self.series = Vec::with_capacity(fft_capacity(self.frames_hint, self.n_groups, 3));
        }

        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let anchors_u32 = anchors_to_u32(self.anchors.as_ref().unwrap());
                let anchors_gpu = ctx.anchors(&anchors_u32)?;
                self.gpu = Some(RotAcfGpuState {
                    ctx: ctx.clone(),
                    anchors: anchors_gpu,
                    kind: self.orientation_kind,
                });
            }
        }

        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.n_groups == 0 {
            return Ok(());
        }

        let n_frames = chunk.n_frames;
        if n_frames == 0 {
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        let mut used_gpu = false;
        #[cfg(not(feature = "cuda"))]
        let used_gpu = false;
        #[cfg(feature = "cuda")]
        let orient_gpu = {
            if let Some(gpu) = &self.gpu {
                let coords = coords_as_float4(&chunk.coords);
                let orient = match gpu.kind {
                    OrientationKind::Plane => gpu.ctx.orientation_plane(
                        coords,
                        chunk.coords.len() / chunk.n_frames,
                        n_frames,
                        &gpu.anchors,
                        self.length_scale as f32,
                    )?,
                    OrientationKind::Vector => gpu.ctx.orientation_vector(
                        coords,
                        chunk.coords.len() / chunk.n_frames,
                        n_frames,
                        &gpu.anchors,
                        self.length_scale as f32,
                    )?,
                };
                used_gpu = true;
                Some(orient)
            } else {
                None
            }
        };
        for frame in 0..n_frames {
            self.frame_index += 1;
            if let Some(dec) = self.frame_decimation {
                let idx1 = self.frame_index;
                if idx1 > dec.start && (idx1 % dec.stride) != 0 {
                    continue;
                }
            }

            update_time_axis(
                self.frame_index,
                frame,
                chunk.time_ps.as_deref(),
                &mut self.dt0,
                &mut self.uniform_time,
                &mut self.last_time,
                self.time_binning.eps_num.max(1.0e-6),
            );

            if used_gpu {
                #[cfg(feature = "cuda")]
                {
                    let orient = orient_gpu.as_ref().unwrap();
                    for g in 0..self.n_groups {
                        let v = orient[frame * self.n_groups + g];
                        let base = g * 3;
                        self.sample_f32[base] = v.x as f32;
                        self.sample_f32[base + 1] = v.y as f32;
                        self.sample_f32[base + 2] = v.z as f32;
                    }
                }
            } else {
                let frame_offset = frame * (chunk.coords.len() / n_frames);
                for g in 0..self.n_groups {
                    let [a, b, c] = self.anchors.as_ref().unwrap()[g];
                    let pa = chunk.coords[frame_offset + a];
                    let pb = chunk.coords[frame_offset + b];
                    let vec = match self.orientation_kind {
                        OrientationKind::Plane => {
                            let pc = chunk.coords[frame_offset + c];
                            let v1 = [
                                (pa[0] - pb[0]) as f64 * self.length_scale,
                                (pa[1] - pb[1]) as f64 * self.length_scale,
                                (pa[2] - pb[2]) as f64 * self.length_scale,
                            ];
                            let v2 = [
                                (pa[0] - pc[0]) as f64 * self.length_scale,
                                (pa[1] - pc[1]) as f64 * self.length_scale,
                                (pa[2] - pc[2]) as f64 * self.length_scale,
                            ];
                            cross_unit(v1, v2)
                        }
                        OrientationKind::Vector => {
                            let v = [
                                (pb[0] - pa[0]) as f64 * self.length_scale,
                                (pb[1] - pa[1]) as f64 * self.length_scale,
                                (pb[2] - pa[2]) as f64 * self.length_scale,
                            ];
                            unit(v)
                        }
                    };
                    let base = g * 3;
                    self.sample_f32[base] = vec[0] as f32;
                    self.sample_f32[base + 1] = vec[1] as f32;
                    self.sample_f32[base + 2] = vec[2] as f32;
                }
            }

            match self.resolved_mode {
                LagMode::MultiTau => {
                    if let Some(buffer) = &mut self.multi_tau {
                        let cols = 2 * (self.type_counts.len() + 1);
                        let type_ids = &self.type_ids;
                        let type_counts = &self.type_counts;
                        let n_groups = self.n_groups;
                        let dt_dec = self.dt_decimation;
                        let lags = &self.lags;
                        let acc = &mut self.acc;
                        buffer.update(&self.sample_f32, |lag_idx, cur, old| {
                            let lag = lags[lag_idx];
                            if !lag_allowed(lag, dt_dec) {
                                return;
                            }
                            accumulate_rotacf(
                                acc,
                                lag_idx,
                                cols,
                                n_groups,
                                type_ids,
                                type_counts,
                                cur,
                                old,
                            );
                        });
                    }
                }
                LagMode::Ring => {
                    if let Some(buffer) = &mut self.ring {
                        let cols = 2 * (self.type_counts.len() + 1);
                        let type_ids = &self.type_ids;
                        let type_counts = &self.type_counts;
                        let n_groups = self.n_groups;
                        let dt_dec = self.dt_decimation;
                        let acc = &mut self.acc;
                        buffer.update(&self.sample_f32, |lag, cur, old| {
                            if !lag_allowed(lag, dt_dec) {
                                return;
                            }
                            let lag_idx = lag - 1;
                            accumulate_rotacf(
                                acc,
                                lag_idx,
                                cols,
                                n_groups,
                                type_ids,
                                type_counts,
                                cur,
                                old,
                            );
                        });
                    }
                }
                LagMode::Fft => {
                    self.series.extend_from_slice(&self.sample_f32);
                }
                LagMode::Auto => {}
            }

            self.samples_seen += 1;
        }

        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let n_frames = if self.resolved_mode == LagMode::Fft {
            if self.n_groups == 0 {
                0
            } else {
                self.series.len() / (self.n_groups * 3)
            }
        } else {
            self.samples_seen
        };

        if n_frames < 2 || self.n_groups == 0 {
            return Ok(PlanOutput::TimeSeries {
                time: Vec::new(),
                data: Vec::new(),
                rows: 0,
                cols: 0,
            });
        }

        let dt0 = self.dt0.unwrap_or(1.0);
        if !self.uniform_time {
            return Err(TrajError::Mismatch(
                "rotacf requires uniform frame spacing for lag-time output".into(),
            ));
        }
        let n_types = self.type_counts.len();
        let cols = 2 * (n_types + 1);

        match self.resolved_mode {
            LagMode::Fft => {
                #[cfg(feature = "cuda")]
                if let Some(gpu) = &self.gpu {
                    let ndframe = n_frames.saturating_sub(1);
                    if ndframe > 0 {
                        let times_f32: Vec<f32> =
                            (0..n_frames).map(|i| (dt0 * i as f64) as f32).collect();
                        let type_ids_u32: Vec<u32> =
                            self.type_ids.iter().map(|t| *t as u32).collect();
                        let type_counts_u32: Vec<u32> =
                            self.type_counts.iter().map(|c| *c as u32).collect();
                        let time_binning = (
                            self.time_binning.eps_num as f32,
                            self.time_binning.eps_add as f32,
                        );
                        let (corr_gpu, corr_p2_gpu, n_diff_gpu) = gpu.ctx.rotacf_time_lag(
                            &self.series,
                            &times_f32,
                            &type_ids_u32,
                            &type_counts_u32,
                            self.n_groups,
                            n_types,
                            ndframe,
                            None,
                            self.dt_decimation
                                .map(|d| (d.cut1, d.stride1, d.cut2, d.stride2)),
                            time_binning,
                        )?;
                        if !corr_gpu.is_empty() && !corr_p2_gpu.is_empty() {
                            let mut lags = Vec::new();
                            for lag in 1..=ndframe {
                                if lag_allowed(lag, self.dt_decimation) {
                                    lags.push(lag);
                                }
                            }
                            let stride = n_types + 1;
                            let mut acc = vec![0.0f64; lags.len() * cols];
                            let mut counts = vec![0u64; lags.len()];
                            for (idx, &lag) in lags.iter().enumerate() {
                                counts[idx] = n_diff_gpu[lag] as u64;
                                let base = lag * stride;
                                let out_base = idx * cols;
                                for t in 0..stride {
                                    acc[out_base + t] = corr_gpu[base + t] as f64;
                                    acc[out_base + stride + t] = corr_p2_gpu[base + t] as f64;
                                }
                            }
                            let (time, data) =
                                finalize_rotacf(&lags, &acc, &counts, dt0, cols, self.p2_legendre);
                            return Ok(PlanOutput::TimeSeries {
                                time,
                                data,
                                rows: lags.len() + 1,
                                cols,
                            });
                        }
                    }
                }

                let (lags, acc, counts) = rotacf_fft(
                    &self.series,
                    n_frames,
                    self.n_groups,
                    &self.type_ids,
                    &self.type_counts,
                    self.dt_decimation,
                )?;
                let (time, data) =
                    finalize_rotacf(&lags, &acc, &counts, dt0, cols, self.p2_legendre);
                Ok(PlanOutput::TimeSeries {
                    time,
                    data,
                    rows: lags.len() + 1,
                    cols,
                })
            }
            LagMode::Ring => {
                let raw_counts: &[u64] = self.ring.as_ref().map(|r| r.n_pairs()).unwrap_or(&[]);
                let counts: Vec<u64> = self
                    .lags
                    .iter()
                    .map(|&lag| raw_counts.get(lag).copied().unwrap_or(0))
                    .collect();
                let (time, data) =
                    finalize_rotacf(&self.lags, &self.acc, &counts, dt0, cols, self.p2_legendre);
                Ok(PlanOutput::TimeSeries {
                    time,
                    data,
                    rows: self.lags.len() + 1,
                    cols,
                })
            }
            LagMode::MultiTau => {
                let counts: &[u64] = self.multi_tau.as_ref().map(|m| m.n_pairs()).unwrap_or(&[]);
                let (time, data) =
                    finalize_rotacf(&self.lags, &self.acc, &counts, dt0, cols, self.p2_legendre);
                Ok(PlanOutput::TimeSeries {
                    time,
                    data,
                    rows: self.lags.len() + 1,
                    cols,
                })
            }
            LagMode::Auto => Ok(PlanOutput::TimeSeries {
                time: Vec::new(),
                data: Vec::new(),
                rows: 0,
                cols: 0,
            }),
        }
    }
}
