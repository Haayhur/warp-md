use traj_core::error::{TrajError, TrajResult};
use traj_core::system::System;

use crate::correlators::LagMode;
use crate::executor::Device;
#[cfg(feature = "cuda")]
use crate::plans::analysis::group_runtime::groups_to_csr;
use crate::plans::analysis::group_runtime::{alloc_group_unwrap_buffers, compute_group_inv_mass};
use crate::plans::analysis::grouping::{GroupBy, GroupSpec};
use crate::plans::analysis::time_correlation::{
    build_lag_runtime, fft_capacity, resolve_lag_mode, AutoLagMode,
};

use super::super::ConductivityPlan;

#[cfg(feature = "cuda")]
use super::super::ConductivityGpuState;

impl ConductivityPlan {
    pub(in super::super) fn init_state(
        &mut self,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        self.n_atoms = system.n_atoms();
        if self.charges.len() != self.n_atoms {
            return Err(TrajError::Mismatch(
                "charges length does not match atom count".into(),
            ));
        }
        let mut spec = GroupSpec::new(self.selection.clone(), self.group_by);
        if let Some(types) = &self.group_types {
            spec = spec.with_group_types(types.clone());
        }
        let groups = spec.build(system)?;
        self.n_groups = groups.n_groups();
        self.type_ids = groups.type_ids();
        self.type_counts = groups.type_counts();
        self.groups = Some(groups);
        self.masses = system.atoms.mass.iter().map(|m| *m as f64).collect();
        self.group_charge = vec![0.0f64; self.n_groups];
        self.group_inv_mass = self
            .groups
            .as_ref()
            .map(|group_map| compute_group_inv_mass(group_map, &self.masses))
            .unwrap_or_default();
        self.charged_groups.clear();
        for (g_idx, atoms) in self.groups.as_ref().unwrap().groups.iter().enumerate() {
            let mut charge_sum = 0.0f64;
            for &atom_idx in atoms {
                charge_sum += self.charges[atom_idx];
            }
            self.group_charge[g_idx] = charge_sum;
            if charge_sum.abs() > f64::EPSILON {
                self.charged_groups.push(g_idx);
            }
        }

        self.dt0 = None;
        self.uniform_time = true;
        self.last_time = None;
        self.frame_index = 0;
        self.samples_seen = 0;
        self.vol_sum = 0.0;
        self.vol_count = 0;
        (
            self.last_wrapped,
            self.wrapped_curr,
            self.unwrap_prev,
            self.unwrap_curr,
        ) = alloc_group_unwrap_buffers(self.n_groups);
        self.sample_f32 = vec![0.0f32; self.n_groups * 3];
        self.sample_total_f32 = [0.0; 3];
        self.total_stream_fastpath = false;
        self.lags.clear();
        self.acc.clear();
        self.multi_tau = None;
        self.ring = None;
        self.series.clear();
        self.io_selection = None;
        self.atom_io_fastpath = false;
        self.type_disp_sums.clear();
        self.fft_stream_sums.clear();
        self.profile_transport = std::env::var_os("WARP_MD_TRANSPORT_PROFILE")
            .map(|v| v != "0")
            .unwrap_or(false);
        self.perf_chunks = 0;
        self.perf_frames = 0;
        self.perf_read_ns = 0;
        self.perf_unwrap_ns = 0;
        self.perf_accum_ns = 0;
        self.perf_finalize_ns = 0;

        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let (offsets, indices, max_len) =
                    groups_to_csr(&self.groups.as_ref().unwrap().groups);
                let groups = ctx.groups(&offsets, &indices, max_len)?;
                let masses: Vec<f32> = system.atoms.mass.iter().map(|m| *m).collect();
                let masses = ctx.upload_f32(&masses)?;
                self.gpu = Some(ConductivityGpuState {
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
        let can_select_io = matches!(self.group_by, GroupBy::Atom) && !has_gpu;
        if can_select_io {
            let selection = self.selection.indices.as_ref().clone();
            let atom_groups_match_selection = self
                .groups
                .as_ref()
                .map(|gmap| {
                    gmap.groups.len() == selection.len()
                        && gmap.groups.iter().zip(selection.iter().copied()).all(
                            |(group, src_idx)| group.len() == 1 && group[0] == src_idx as usize,
                        )
                })
                .unwrap_or(false);
            if atom_groups_match_selection {
                self.io_selection = Some(selection);
                self.atom_io_fastpath = true;
            }
        }
        let n_types = self.type_counts.len();
        let fft_streams = if self.transference { n_types + 1 } else { 1 };
        self.type_disp_sums.resize(n_types, [0.0; 3]);
        self.fft_stream_sums.resize(fft_streams, [0.0; 3]);
        self.resolved_mode = resolve_lag_mode(
            &self.lag,
            AutoLagMode::FftIfFits {
                frames_hint: self.frames_hint,
                streams: fft_streams,
                width: 3,
                scalar_bytes: 4,
            },
        );
        self.fft_store_groups = false;
        self.total_stream_fastpath =
            !self.transference && matches!(self.resolved_mode, LagMode::Ring | LagMode::MultiTau);

        let cols = if self.transference {
            n_types * n_types + 1
        } else {
            1
        };

        let n_streams = if self.total_stream_fastpath {
            1
        } else {
            self.n_groups
        };
        let runtime = build_lag_runtime(&self.lag, self.resolved_mode, n_streams, 3, cols, None);
        self.lags = runtime.lags;
        self.acc = runtime.acc;
        self.multi_tau = runtime.multi_tau;
        self.ring = runtime.ring;
        if self.resolved_mode == LagMode::Fft {
            self.fft_store_groups = has_gpu;
            let fft_series_streams = if has_gpu { self.n_groups } else { fft_streams };
            self.series = Vec::with_capacity(fft_capacity(self.frames_hint, fft_series_streams, 3));
        }

        Ok(())
    }
}
