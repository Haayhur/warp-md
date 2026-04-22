use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::pbc_math::orthorhombic_lengths;
use traj_core::system::System;

use crate::correlators::{DecimationMode, LagMode};
use crate::executor::{Device, Plan, PlanOutput};
#[cfg(feature = "cuda")]
use crate::plans::analysis::group_runtime::groups_to_csr;
use crate::plans::analysis::group_runtime::{
    alloc_group_positions, compute_group_inv_mass, fill_frame_group_positions,
};
use crate::plans::analysis::grouping::GroupSpec;
use crate::plans::analysis::time_correlation::{
    build_lag_runtime, fft_capacity, resolve_lag_mode, update_time_axis, AutoLagMode,
};

#[cfg(feature = "cuda")]
use traj_gpu::convert_coords;

use super::accumulate::accumulate_ion_pair;
use super::cluster_math::{distance_vec, hash_cluster};
use super::IonPairCorrelationPlan;
#[cfg(feature = "cuda")]
use super::IonPairGpuState;

impl Plan for IonPairCorrelationPlan {
    fn name(&self) -> &'static str {
        "ion_pair_corr"
    }

    fn set_frames_hint(&mut self, n_frames: Option<usize>) {
        self.frames_hint = n_frames;
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
        self.groups = Some(groups);
        self.masses = system.atoms.mass.iter().map(|m| *m as f64).collect();
        self.group_inv_mass = self
            .groups
            .as_ref()
            .map(|group_map| compute_group_inv_mass(group_map, &self.masses))
            .unwrap_or_default();
        self.cat_indices = self
            .type_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, &t)| {
                if t == self.cation_type {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        self.ani_indices = self
            .type_ids
            .iter()
            .enumerate()
            .filter_map(|(idx, &t)| {
                if t == self.anion_type {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect();
        if self.cat_indices.is_empty() || self.ani_indices.is_empty() {
            return Err(TrajError::Mismatch(
                "ion pair correlation requires both cation and anion groups".into(),
            ));
        }

        self.dt0 = None;
        self.uniform_time = true;
        self.last_time = None;
        self.frame_index = 0;
        self.samples_seen = 0;
        self.wrapped_curr = alloc_group_positions(self.n_groups);
        self.sample_f32 = vec![0.0f32; self.n_groups * 3];
        self.pair_idx = vec![u32::MAX; self.n_groups];
        self.cluster_hash = vec![0u64; self.n_groups];
        self.fft_com.clear();
        self.fft_box.clear();
        self.lags.clear();
        self.acc.clear();
        self.multi_tau = None;
        self.ring = None;

        self.resolved_mode = resolve_lag_mode(&self.lag, AutoLagMode::MultiTau);
        let runtime = build_lag_runtime(
            &self.lag,
            self.resolved_mode,
            self.n_groups,
            3,
            6,
            Some(DecimationMode::Latest),
        );
        self.lags = runtime.lags;
        self.acc = runtime.acc;
        self.multi_tau = runtime.multi_tau;
        self.ring = runtime.ring;
        if self.resolved_mode == LagMode::Fft {
            self.fft_com = Vec::with_capacity(fft_capacity(self.frames_hint, self.n_groups, 3));
            self.fft_box = Vec::with_capacity(fft_capacity(self.frames_hint, 1, 3));
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
                self.gpu = Some(IonPairGpuState {
                    ctx: ctx.clone(),
                    groups,
                    masses,
                });
            }
            if self.resolved_mode == LagMode::Fft && self.gpu.is_none() {
                return Err(TrajError::Unsupported(
                    "ion_pair_corr fft requires cuda".into(),
                ));
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            if self.resolved_mode == LagMode::Fft {
                return Err(TrajError::Unsupported(
                    "ion_pair_corr fft requires cuda".into(),
                ));
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
        let com_gpu = {
            if let Some(gpu) = &self.gpu {
                let coords = convert_coords(&chunk.coords);
                let com = gpu.ctx.group_com(
                    &coords,
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
        for frame in 0..n_frames {
            self.frame_index += 1;

            update_time_axis(
                self.frame_index,
                frame,
                chunk.time_ps.as_deref(),
                &mut self.dt0,
                &mut self.uniform_time,
                &mut self.last_time,
                1.0e-6,
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
                let frame_offset = frame * self.n_atoms;
                fill_frame_group_positions(
                    &chunk.coords,
                    frame_offset,
                    Some(&self.groups.as_ref().unwrap().groups),
                    &self.masses,
                    &self.group_inv_mass,
                    self.length_scale,
                    &mut self.wrapped_curr,
                );
            }

            let box_l = orthorhombic_lengths(&chunk.box_[frame]).map(|[lx, ly, lz]| {
                [
                    lx * self.length_scale,
                    ly * self.length_scale,
                    lz * self.length_scale,
                ]
            });

            if self.resolved_mode == LagMode::Fft {
                for g in 0..self.n_groups {
                    let v = self.wrapped_curr[g];
                    self.fft_com.push(v[0] as f32);
                    self.fft_com.push(v[1] as f32);
                    self.fft_com.push(v[2] as f32);
                }
                if let Some(b) = box_l {
                    self.fft_box.push(b[0] as f32);
                    self.fft_box.push(b[1] as f32);
                    self.fft_box.push(b[2] as f32);
                } else {
                    self.fft_box.extend_from_slice(&[0.0, 0.0, 0.0]);
                }
                self.samples_seen += 1;
                continue;
            }

            self.pair_idx.fill(u32::MAX);
            self.cluster_hash.fill(0);

            for &cat in &self.cat_indices {
                let mut cluster = Vec::with_capacity(self.max_cluster);
                let mut best_dist = f64::MAX;
                let mut best_idx = u32::MAX;
                for &ani in &self.ani_indices {
                    let (dx, dy, dz) =
                        distance_vec(self.wrapped_curr[cat], self.wrapped_curr[ani], box_l);
                    let dr2 = dx * dx + dy * dy + dz * dz;
                    let dr = dr2.sqrt();
                    if dr < self.rclust_cat {
                        if cluster.len() < self.max_cluster {
                            cluster.push(ani as u32);
                        }
                        if dr < best_dist {
                            best_dist = dr;
                            best_idx = ani as u32;
                        }
                    }
                }
                if best_idx != u32::MAX {
                    self.pair_idx[cat] = best_idx;
                }
                self.cluster_hash[cat] = hash_cluster(&cluster);
            }

            for &ani in &self.ani_indices {
                let mut cluster = Vec::with_capacity(self.max_cluster);
                let mut best_dist = f64::MAX;
                let mut best_idx = u32::MAX;
                for &cat in &self.cat_indices {
                    let (dx, dy, dz) =
                        distance_vec(self.wrapped_curr[ani], self.wrapped_curr[cat], box_l);
                    let dr2 = dx * dx + dy * dy + dz * dz;
                    let dr = dr2.sqrt();
                    if dr < self.rclust_ani {
                        if cluster.len() < self.max_cluster {
                            cluster.push(cat as u32);
                        }
                        if dr < best_dist {
                            best_dist = dr;
                            best_idx = cat as u32;
                        }
                    }
                }
                if best_idx != u32::MAX {
                    self.pair_idx[ani] = best_idx;
                }
                self.cluster_hash[ani] = hash_cluster(&cluster);
            }

            for g in 0..self.n_groups {
                let base = g * 3;
                self.sample_f32[base] = f32::from_bits(self.pair_idx[g]);
                let hash = self.cluster_hash[g];
                self.sample_f32[base + 1] = f32::from_bits(hash as u32);
                self.sample_f32[base + 2] = f32::from_bits((hash >> 32) as u32);
            }

            match self.resolved_mode {
                LagMode::MultiTau => {
                    if let Some(buffer) = &mut self.multi_tau {
                        let acc = &mut self.acc;
                        let cat_indices = &self.cat_indices;
                        let ani_indices = &self.ani_indices;
                        buffer.update(&self.sample_f32, |lag_idx, cur, old| {
                            accumulate_ion_pair(acc, lag_idx, cat_indices, ani_indices, cur, old);
                        });
                    }
                }
                LagMode::Ring => {
                    if let Some(buffer) = &mut self.ring {
                        let acc = &mut self.acc;
                        let cat_indices = &self.cat_indices;
                        let ani_indices = &self.ani_indices;
                        buffer.update(&self.sample_f32, |lag, cur, old| {
                            let lag_idx = lag - 1;
                            accumulate_ion_pair(acc, lag_idx, cat_indices, ani_indices, cur, old);
                        });
                    }
                }
                LagMode::Fft | LagMode::Auto => {}
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
                self.fft_com.len() / (self.n_groups * 3)
            }
        } else {
            self.samples_seen
        };
        if n_frames == 0 || self.n_groups == 0 {
            return Ok(PlanOutput::TimeSeries {
                time: Vec::new(),
                data: Vec::new(),
                rows: 0,
                cols: 0,
            });
        }

        let dt0 = self.dt0.unwrap_or(1.0);

        if self.resolved_mode == LagMode::Fft {
            #[cfg(feature = "cuda")]
            if let Some(gpu) = &self.gpu {
                let cat_idx_u32: Vec<u32> = self.cat_indices.iter().map(|v| *v as u32).collect();
                let ani_idx_u32: Vec<u32> = self.ani_indices.iter().map(|v| *v as u32).collect();
                let (ip_cat, ip_ani, cp_cat, cp_ani) = gpu.ctx.ion_pair_correlations(
                    &self.fft_com,
                    &self.fft_box,
                    &cat_idx_u32,
                    &ani_idx_u32,
                    self.n_groups,
                    n_frames,
                    self.max_cluster,
                    self.rclust_cat as f32,
                    self.rclust_ani as f32,
                )?;
                let n_cat = self.cat_indices.len() as f64;
                let n_ani = self.ani_indices.len() as f64;
                let n_groups_f = self.n_groups as f64;
                let mut time = Vec::with_capacity(n_frames);
                let mut data = vec![0.0f32; n_frames * 6];
                time.push(0.0);
                for c in 0..6 {
                    data[c] = 1.0;
                }
                for lag in 1..n_frames {
                    let count = (n_frames - lag) as f64;
                    let base = lag * 6;
                    time.push((dt0 * lag as f64) as f32);
                    if count == 0.0 {
                        continue;
                    }
                    let ip_cat_val = (ip_cat[lag] as f64) / (n_cat * count);
                    let ip_ani_val = (ip_ani[lag] as f64) / (n_ani * count);
                    let cp_cat_val = (cp_cat[lag] as f64) / (n_cat * count);
                    let cp_ani_val = (cp_ani[lag] as f64) / (n_ani * count);
                    let ip_total = (ip_cat[lag] as f64 + ip_ani[lag] as f64) / (n_groups_f * count);
                    let cp_total = (cp_cat[lag] as f64 + cp_ani[lag] as f64) / (n_groups_f * count);
                    data[base] = ip_total as f32;
                    data[base + 1] = ip_cat_val as f32;
                    data[base + 2] = ip_ani_val as f32;
                    data[base + 3] = cp_total as f32;
                    data[base + 4] = cp_cat_val as f32;
                    data[base + 5] = cp_ani_val as f32;
                }
                return Ok(PlanOutput::TimeSeries {
                    time,
                    data,
                    rows: n_frames,
                    cols: 6,
                });
            }
            return Err(TrajError::Unsupported(
                "ion_pair_corr fft requires cuda".into(),
            ));
        }

        let counts = match self.resolved_mode {
            LagMode::Ring => self
                .ring
                .as_ref()
                .map(|r| r.n_pairs().to_vec())
                .unwrap_or_default(),
            LagMode::MultiTau => self
                .multi_tau
                .as_ref()
                .map(|m| m.n_pairs().to_vec())
                .unwrap_or_default(),
            _ => Vec::new(),
        };

        let n_cat = self.cat_indices.len() as f64;
        let n_ani = self.ani_indices.len() as f64;
        let n_groups_f = self.n_groups as f64;
        let mut lags_out = Vec::new();
        for (idx, &lag) in self.lags.iter().enumerate() {
            let count = if self.resolved_mode == LagMode::Ring {
                counts.get(lag).copied().unwrap_or(0)
            } else {
                counts.get(idx).copied().unwrap_or(0)
            };
            if count > 0 {
                lags_out.push((idx, lag, count));
            }
        }

        let mut time = Vec::with_capacity(lags_out.len() + 1);
        let mut data = vec![0.0f32; (lags_out.len() + 1) * 6];
        time.push(0.0);
        for c in 0..6 {
            data[c] = 1.0;
        }
        for (row_idx, (idx, lag, count_u64)) in lags_out.iter().enumerate() {
            time.push((dt0 * *lag as f64) as f32);
            let count = *count_u64 as f64;
            let acc_base = *idx * 6;
            let base = (row_idx + 1) * 6;
            let ip_cat = self.acc[acc_base + 1];
            let ip_ani = self.acc[acc_base + 2];
            let cp_cat = self.acc[acc_base + 4];
            let cp_ani = self.acc[acc_base + 5];
            let denom_cat = n_cat * count;
            let denom_ani = n_ani * count;
            let denom_tot = n_groups_f * count;
            let ip_cat_val = if denom_cat > 0.0 {
                ip_cat / denom_cat
            } else {
                0.0
            };
            let ip_ani_val = if denom_ani > 0.0 {
                ip_ani / denom_ani
            } else {
                0.0
            };
            let cp_cat_val = if denom_cat > 0.0 {
                cp_cat / denom_cat
            } else {
                0.0
            };
            let cp_ani_val = if denom_ani > 0.0 {
                cp_ani / denom_ani
            } else {
                0.0
            };
            let ip_total = if denom_tot > 0.0 {
                (ip_cat + ip_ani) / denom_tot
            } else {
                0.0
            };
            let cp_total = if denom_tot > 0.0 {
                (cp_cat + cp_ani) / denom_tot
            } else {
                0.0
            };
            data[base] = ip_total as f32;
            data[base + 1] = ip_cat_val as f32;
            data[base + 2] = ip_ani_val as f32;
            data[base + 3] = cp_total as f32;
            data[base + 4] = cp_cat_val as f32;
            data[base + 5] = cp_ani_val as f32;
        }

        Ok(PlanOutput::TimeSeries {
            time,
            data,
            rows: lags_out.len() + 1,
            cols: 6,
        })
    }
}
