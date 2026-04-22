use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::system::System;

use crate::correlators::LagMode;
use crate::executor::{Device, Plan, PlanOutput};
#[cfg(feature = "cuda")]
use crate::plans::analysis::group_runtime::groups_to_csr;
use crate::plans::analysis::group_runtime::{
    alloc_group_unwrap_buffers, compute_group_inv_mass, fill_frame_group_positions,
    seed_unwrapped_groups, unwrap_frame_groups,
};
use crate::plans::analysis::grouping::GroupSpec;
use crate::plans::analysis::time_correlation::{
    build_lag_runtime, fft_capacity, lag_allowed, resolve_lag_mode, update_time_axis, AutoLagMode,
};

#[cfg(feature = "cuda")]
use traj_gpu::coords_as_float4;

use super::accumulate::accumulate_msd;
use super::fft::msd_fft;
use super::time_lag::msd_cols;
#[cfg(feature = "cuda")]
use super::MsdGpuState;
use super::MsdPlan;

impl Plan for MsdPlan {
    fn name(&self) -> &'static str {
        "msd_multi"
    }

    fn set_frames_hint(&mut self, n_frames: Option<usize>) {
        self.frames_hint = n_frames;
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        self.io_selection.as_deref()
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        if matches!(
            self.group_by,
            crate::plans::analysis::grouping::GroupBy::Atom
        ) {
            Some(self.selection.indices.as_ref())
        } else {
            None
        }
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.n_atoms = system.n_atoms();
        let mut spec = GroupSpec::new(self.selection.clone(), self.group_by);
        if let Some(types) = &self.group_types {
            spec = spec.with_group_types(types.clone());
        }
        let groups = spec.build(system)?;
        self.n_groups = groups.n_groups();
        self.type_ids = groups.type_ids();
        self.type_counts = groups.type_counts();
        self.inv_type_counts = self
            .type_counts
            .iter()
            .map(|&count| if count > 0 { 1.0 / count as f64 } else { 0.0 })
            .collect();
        self.inv_n_groups = if self.n_groups > 0 {
            1.0 / self.n_groups as f64
        } else {
            0.0
        };
        self.groups = Some(groups);
        self.masses = system.atoms.mass.iter().map(|m| *m as f64).collect();
        self.group_inv_mass = self
            .groups
            .as_ref()
            .map(|group_map| compute_group_inv_mass(group_map, &self.masses))
            .unwrap_or_default();

        self.dt0 = None;
        self.uniform_time = true;
        self.last_time = None;
        self.frame_index = 0;
        self.samples_seen = 0;
        (
            self.last_wrapped,
            self.wrapped_curr,
            self.unwrap_prev,
            self.unwrap_curr,
        ) = alloc_group_unwrap_buffers(self.n_groups);
        self.sample_f32 = vec![0.0f32; self.n_groups * 3];
        self.lags.clear();
        self.acc.clear();
        self.multi_tau = None;
        self.ring = None;
        self.series.clear();
        self.io_selection = None;
        self.atom_io_fastpath = false;
        self.profile_transport = std::env::var_os("WARP_MD_TRANSPORT_PROFILE")
            .map(|v| v != "0")
            .unwrap_or(false);
        self.perf_chunks = 0;
        self.perf_frames = 0;
        self.perf_read_ns = 0;
        self.perf_unwrap_ns = 0;
        self.perf_accum_ns = 0;
        self.perf_finalize_ns = 0;

        self.resolved_mode = resolve_lag_mode(
            &self.lag,
            AutoLagMode::FftIfFits {
                frames_hint: self.frames_hint,
                streams: self.n_groups,
                width: 3,
                scalar_bytes: 4,
            },
        );
        let cols = msd_cols(self.axis, self.type_counts.len());
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
                let (offsets, indices, max_len) =
                    groups_to_csr(&self.groups.as_ref().unwrap().groups);
                let groups = ctx.groups(&offsets, &indices, max_len)?;
                let masses: Vec<f32> = system.atoms.mass.iter().map(|m| *m).collect();
                let masses = ctx.upload_f32(&masses)?;
                self.gpu = Some(MsdGpuState {
                    ctx: ctx.clone(),
                    groups,
                    masses,
                });
            }
        }
        let has_gpu = {
            #[cfg(feature = "cuda")]
            {
                self.gpu.is_some()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        let can_select_io = matches!(
            self.group_by,
            crate::plans::analysis::grouping::GroupBy::Atom
        ) && !has_gpu;
        if can_select_io {
            self.io_selection = Some(self.selection.indices.as_ref().clone());
            self.atom_io_fastpath = self.n_groups == self.selection.indices.len();
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
        if chunk.n_atoms != self.n_atoms && !self.atom_io_fastpath {
            return Err(TrajError::Mismatch(
                "msd plan expected full-atom chunk for non-atom grouping".into(),
            ));
        }

        if self.profile_transport {
            self.perf_chunks = self.perf_chunks.saturating_add(1);
        }

        #[cfg(feature = "cuda")]
        let mut used_gpu = false;
        #[cfg(not(feature = "cuda"))]
        let used_gpu = false;
        #[cfg(feature = "cuda")]
        let com_gpu = {
            if let Some(gpu) = &self.gpu {
                let coords = coords_as_float4(&chunk.coords);
                let com = gpu.ctx.group_com(
                    coords,
                    self.n_atoms,
                    n_frames,
                    &gpu.groups,
                    &gpu.masses,
                    self.length_scale as f32,
                )?;
                used_gpu = true;
                Some(com)
            } else {
                None
            }
        };
        let axis = self.axis;
        let cols = msd_cols(axis, self.type_counts.len());
        let type_ids = &self.type_ids;
        let inv_type_counts = &self.inv_type_counts;
        let n_groups = self.n_groups;
        let inv_n_groups = self.inv_n_groups;
        let dt_dec = self.dt_decimation;
        let lags = &self.lags;
        let groups_ref: Option<&[Vec<usize>]> = if self.atom_io_fastpath {
            None
        } else {
            Some(
                self.groups
                    .as_ref()
                    .ok_or_else(|| {
                        TrajError::Mismatch("msd grouping state not initialized".into())
                    })?
                    .groups
                    .as_slice(),
            )
        };

        for frame in 0..n_frames {
            let read_t0 = self.profile_transport.then(std::time::Instant::now);
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
                    let com = com_gpu.as_ref().unwrap();
                    for g in 0..self.n_groups {
                        let v = com[frame * self.n_groups + g];
                        self.wrapped_curr[g] = [v.x as f64, v.y as f64, v.z as f64];
                    }
                }
            } else {
                let frame_offset = frame * chunk.n_atoms;
                let groups = if self.atom_io_fastpath && chunk.n_atoms == self.n_groups {
                    None
                } else {
                    groups_ref
                };
                fill_frame_group_positions(
                    &chunk.coords,
                    frame_offset,
                    groups,
                    &self.masses,
                    &self.group_inv_mass,
                    self.length_scale,
                    &mut self.wrapped_curr,
                );
            }
            add_elapsed_ns(&mut self.perf_read_ns, read_t0);

            let unwrap_t0 = self.profile_transport.then(std::time::Instant::now);
            if self.samples_seen == 0 {
                seed_unwrapped_groups(
                    &self.wrapped_curr,
                    &mut self.last_wrapped,
                    &mut self.unwrap_prev,
                    &mut self.unwrap_curr,
                );
            } else {
                unwrap_frame_groups(
                    &self.wrapped_curr,
                    &mut self.last_wrapped,
                    &mut self.unwrap_prev,
                    &mut self.unwrap_curr,
                    &chunk.box_[frame],
                    self.length_scale,
                );
            }
            add_elapsed_ns(&mut self.perf_unwrap_ns, unwrap_t0);

            let accum_t0 = self.profile_transport.then(std::time::Instant::now);
            for g in 0..self.n_groups {
                let base = g * 3;
                self.sample_f32[base] = self.unwrap_curr[g][0] as f32;
                self.sample_f32[base + 1] = self.unwrap_curr[g][1] as f32;
                self.sample_f32[base + 2] = self.unwrap_curr[g][2] as f32;
            }

            match self.resolved_mode {
                LagMode::MultiTau => {
                    if let Some(buffer) = &mut self.multi_tau {
                        let acc = &mut self.acc;
                        buffer.update(&self.sample_f32, |lag_idx, cur, old| {
                            let lag = lags[lag_idx];
                            if !lag_allowed(lag, dt_dec) {
                                return;
                            }
                            accumulate_msd(
                                acc,
                                lag_idx,
                                cols,
                                axis,
                                n_groups,
                                type_ids,
                                inv_type_counts,
                                inv_n_groups,
                                cur,
                                old,
                            );
                        });
                    }
                }
                LagMode::Ring => {
                    if let Some(buffer) = &mut self.ring {
                        let acc = &mut self.acc;
                        buffer.update(&self.sample_f32, |lag, cur, old| {
                            if !lag_allowed(lag, dt_dec) {
                                return;
                            }
                            let lag_idx = lag - 1;
                            accumulate_msd(
                                acc,
                                lag_idx,
                                cols,
                                axis,
                                n_groups,
                                type_ids,
                                inv_type_counts,
                                inv_n_groups,
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
            add_elapsed_ns(&mut self.perf_accum_ns, accum_t0);

            self.samples_seen += 1;
            if self.profile_transport {
                self.perf_frames = self.perf_frames.saturating_add(1);
            }
        }

        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.profile_transport {
            eprintln!(
                "[transport-profile] plan=msd chunks={} frames={} read_ms={:.3} unwrap_ms={:.3} accum_ms={:.3} finalize_ms={:.3}",
                self.perf_chunks,
                self.perf_frames,
                self.perf_read_ns as f64 / 1_000_000.0,
                self.perf_unwrap_ns as f64 / 1_000_000.0,
                self.perf_accum_ns as f64 / 1_000_000.0,
                self.perf_finalize_ns as f64 / 1_000_000.0,
            );
        }
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
                "msd requires uniform frame spacing for lag-time output".into(),
            ));
        }
        let n_types = self.type_counts.len();
        let cols = msd_cols(self.axis, n_types);

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
                        let axis_f32 = self.axis.map(|a| [a[0] as f32, a[1] as f32, a[2] as f32]);
                        let time_binning = (
                            self.time_binning.eps_num as f32,
                            self.time_binning.eps_add as f32,
                        );
                        let (msd_gpu, n_diff_gpu) = gpu.ctx.msd_time_lag(
                            &self.series,
                            &times_f32,
                            &type_ids_u32,
                            &type_counts_u32,
                            self.n_groups,
                            n_types,
                            ndframe,
                            axis_f32,
                            None,
                            self.dt_decimation
                                .map(|d| (d.cut1, d.stride1, d.cut2, d.stride2)),
                            time_binning,
                        )?;
                        if !msd_gpu.is_empty() && !n_diff_gpu.is_empty() {
                            let mut lags = Vec::new();
                            for lag in 1..=ndframe {
                                if lag_allowed(lag, self.dt_decimation) {
                                    lags.push(lag);
                                }
                            }
                            let mut time = Vec::with_capacity(lags.len());
                            let mut data = vec![0.0f32; lags.len() * cols];
                            for (idx, &lag) in lags.iter().enumerate() {
                                time.push((dt0 * lag as f64) as f32);
                                let count = n_diff_gpu[lag] as f64;
                                if count == 0.0 {
                                    continue;
                                }
                                let base = lag * cols;
                                let out_base = idx * cols;
                                for c in 0..cols {
                                    data[out_base + c] = (msd_gpu[base + c] as f64 / count) as f32;
                                }
                            }
                            return Ok(PlanOutput::TimeSeries {
                                time,
                                data,
                                rows: lags.len(),
                                cols,
                            });
                        }
                    }
                }

                let (lags, acc, counts) = msd_fft(
                    &self.series,
                    n_frames,
                    self.n_groups,
                    &self.type_ids,
                    &self.type_counts,
                    self.axis,
                    self.dt_decimation,
                )?;
                let mut time = Vec::with_capacity(lags.len());
                let mut data = vec![0.0f32; lags.len() * cols];
                for (idx, &lag) in lags.iter().enumerate() {
                    time.push((dt0 * lag as f64) as f32);
                    if counts[idx] == 0 {
                        continue;
                    }
                    let base = idx * cols;
                    for c in 0..cols {
                        data[base + c] = acc[base + c] as f32;
                    }
                }
                Ok(PlanOutput::TimeSeries {
                    time,
                    data,
                    rows: lags.len(),
                    cols,
                })
            }
            LagMode::Ring => {
                let counts: &[u64] = self.ring.as_ref().map(|r| r.n_pairs()).unwrap_or(&[]);
                let mut time = Vec::with_capacity(self.lags.len());
                let mut data = vec![0.0f32; self.lags.len() * cols];
                for (idx, &lag) in self.lags.iter().enumerate() {
                    time.push((dt0 * lag as f64) as f32);
                    let count = counts.get(lag).copied().unwrap_or(0) as f64;
                    if count == 0.0 {
                        continue;
                    }
                    let base = idx * cols;
                    for c in 0..cols {
                        data[base + c] = (self.acc[base + c] / count) as f32;
                    }
                }
                Ok(PlanOutput::TimeSeries {
                    time,
                    data,
                    rows: self.lags.len(),
                    cols,
                })
            }
            LagMode::MultiTau => {
                let counts: &[u64] = self.multi_tau.as_ref().map(|m| m.n_pairs()).unwrap_or(&[]);
                let mut time = Vec::with_capacity(self.lags.len());
                let mut data = vec![0.0f32; self.lags.len() * cols];
                for (idx, &lag) in self.lags.iter().enumerate() {
                    time.push((dt0 * lag as f64) as f32);
                    let count = counts.get(idx).copied().unwrap_or(0) as f64;
                    if count == 0.0 {
                        continue;
                    }
                    let base = idx * cols;
                    for c in 0..cols {
                        data[base + c] = (self.acc[base + c] / count) as f32;
                    }
                }
                Ok(PlanOutput::TimeSeries {
                    time,
                    data,
                    rows: self.lags.len(),
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

#[inline]
fn add_elapsed_ns(dst: &mut u64, start: Option<std::time::Instant>) {
    if let Some(t0) = start {
        let ns = t0.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        *dst = dst.saturating_add(ns);
    }
}
