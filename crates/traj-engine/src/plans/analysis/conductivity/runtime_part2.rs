#[cfg(feature = "cuda")]
use super::ConductivityGpuState;

impl ConductivityPlan {
    pub(super) fn init_state(&mut self, system: &System, device: &Device) -> TrajResult<()> {
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
        self.group_inv_mass.clear();
        self.group_inv_mass.reserve(self.n_groups);
        self.charged_groups.clear();
        for (g_idx, atoms) in self.groups.as_ref().unwrap().groups.iter().enumerate() {
            let mut charge_sum = 0.0f64;
            let mut mass_sum = 0.0f64;
            for &atom_idx in atoms {
                charge_sum += self.charges[atom_idx];
                mass_sum += self.masses[atom_idx];
            }
            self.group_charge[g_idx] = charge_sum;
            let inv_mass = if mass_sum > 0.0 { 1.0 / mass_sum } else { 0.0 };
            self.group_inv_mass.push(inv_mass);
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
        self.last_wrapped = vec![[0.0; 3]; self.n_groups];
        self.wrapped_curr = vec![[0.0; 3]; self.n_groups];
        self.unwrap_prev = vec![[0.0; 3]; self.n_groups];
        self.unwrap_curr = vec![[0.0; 3]; self.n_groups];
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
            if let Device::Cuda(ctx) = device {
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

        #[cfg(not(feature = "cuda"))]
        let _ = device;

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
        let mut resolved_mode = self.lag.mode;
        if resolved_mode == LagMode::Auto {
            let use_fft = if let Some(n_frames) = self.frames_hint {
                self.lag.fft_fits(n_frames, fft_streams, 3, 4)
            } else {
                false
            };
            resolved_mode = if use_fft {
                LagMode::Fft
            } else {
                LagMode::MultiTau
            };
        }
        self.resolved_mode = resolved_mode;
        self.fft_store_groups = false;
        self.total_stream_fastpath =
            !self.transference && matches!(self.resolved_mode, LagMode::Ring | LagMode::MultiTau);

        let cols = if self.transference {
            n_types * n_types + 1
        } else {
            1
        };

        match self.resolved_mode {
            LagMode::MultiTau => {
                let n_streams = if self.total_stream_fastpath {
                    1
                } else {
                    self.n_groups
                };
                let buffer = MultiTauBuffer::new(
                    n_streams,
                    3,
                    self.lag.multi_tau_m,
                    self.lag.multi_tau_max_levels,
                );
                self.lags = buffer.out_lags().to_vec();
                self.acc = vec![0.0f64; self.lags.len() * cols];
                self.multi_tau = Some(buffer);
            }
            LagMode::Ring => {
                let n_streams = if self.total_stream_fastpath {
                    1
                } else {
                    self.n_groups
                };
                let max_lag = self.lag.ring_max_lag_capped(n_streams, 3, 4);
                let buffer = RingBuffer::new(n_streams, 3, max_lag);
                self.lags = (1..=max_lag).collect();
                self.acc = vec![0.0f64; self.lags.len() * cols];
                self.ring = Some(buffer);
            }
            LagMode::Fft => {
                if has_gpu {
                    self.fft_store_groups = true;
                    let capacity = self
                        .frames_hint
                        .unwrap_or(0)
                        .saturating_mul(self.n_groups)
                        .saturating_mul(3);
                    self.series = Vec::with_capacity(capacity);
                } else {
                    let capacity = self
                        .frames_hint
                        .unwrap_or(0)
                        .saturating_mul(fft_streams)
                        .saturating_mul(3);
                    self.series = Vec::with_capacity(capacity);
                }
            }
            LagMode::Auto => {}
        }

        Ok(())
    }
}
