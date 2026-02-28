impl ConductivityPlan {
    pub(super) fn process_chunk_state(
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
                "conductivity plan expected full-atom chunk for non-atom grouping".into(),
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

        #[cfg(not(feature = "cuda"))]
        let _com_gpu = ();

        let groups_ref: Option<&[Vec<usize>]> = if self.atom_io_fastpath {
            None
        } else {
            Some(
                self.groups
                    .as_ref()
                    .ok_or_else(|| {
                        TrajError::Mismatch("conductivity grouping state not initialized".into())
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

            let time = if let Some(times) = &chunk.time_ps {
                times[frame] as f64
            } else {
                (self.frame_index - 1) as f64
            };
            if let Some(last) = self.last_time {
                let dt = time - last;
                if let Some(dt0) = self.dt0 {
                    if (dt - dt0).abs() > self.time_binning.eps_num.max(1.0e-6) {
                        self.uniform_time = false;
                    }
                } else if dt > 0.0 {
                    self.dt0 = Some(dt);
                }
            }
            self.last_time = Some(time);

            if let Some(b) = box_lengths(&chunk.box_[frame]) {
                let vol = b[0] * b[1] * b[2] * self.length_scale.powi(3);
                self.vol_sum += vol;
                self.vol_count += 1;
            }

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
                if self.atom_io_fastpath && chunk.n_atoms == self.n_groups {
                    for g in 0..self.n_groups {
                        let p = chunk.coords[frame_offset + g];
                        self.wrapped_curr[g][0] = (p[0] as f64) * self.length_scale;
                        self.wrapped_curr[g][1] = (p[1] as f64) * self.length_scale;
                        self.wrapped_curr[g][2] = (p[2] as f64) * self.length_scale;
                    }
                } else {
                    for (g_idx, atoms) in groups_ref.unwrap().iter().enumerate() {
                        let mut sum = [0.0f64; 3];
                        for &atom_idx in atoms {
                            let p = chunk.coords[frame_offset + atom_idx];
                            let m = self.masses[atom_idx];
                            sum[0] += (p[0] as f64) * m;
                            sum[1] += (p[1] as f64) * m;
                            sum[2] += (p[2] as f64) * m;
                        }
                        let scale = self.group_inv_mass.get(g_idx).copied().unwrap_or(0.0)
                            * self.length_scale;
                        self.wrapped_curr[g_idx][0] = sum[0] * scale;
                        self.wrapped_curr[g_idx][1] = sum[1] * scale;
                        self.wrapped_curr[g_idx][2] = sum[2] * scale;
                    }
                }
            }
            add_elapsed_ns(&mut self.perf_read_ns, read_t0);

            let unwrap_t0 = self.profile_transport.then(std::time::Instant::now);
            if self.samples_seen == 0 {
                self.last_wrapped.copy_from_slice(&self.wrapped_curr);
                self.unwrap_prev.copy_from_slice(&self.wrapped_curr);
                self.unwrap_curr.copy_from_slice(&self.wrapped_curr);
            } else {
                let box_l = box_lengths(&chunk.box_[frame]).map(|b| {
                    [
                        b[0] * self.length_scale,
                        b[1] * self.length_scale,
                        b[2] * self.length_scale,
                    ]
                });
                for g in 0..self.n_groups {
                    let curr = self.wrapped_curr[g];
                    let prev = self.last_wrapped[g];
                    let mut diff = [curr[0] - prev[0], curr[1] - prev[1], curr[2] - prev[2]];
                    if let Some(b) = box_l {
                        for k in 0..3 {
                            let l = b[k];
                            if l > 0.0 {
                                diff[k] -= (diff[k] / l).round() * l;
                            }
                        }
                    }
                    self.unwrap_curr[g][0] = self.unwrap_prev[g][0] + diff[0];
                    self.unwrap_curr[g][1] = self.unwrap_prev[g][1] + diff[1];
                    self.unwrap_curr[g][2] = self.unwrap_prev[g][2] + diff[2];
                }
                self.unwrap_prev.copy_from_slice(&self.unwrap_curr);
                self.last_wrapped.copy_from_slice(&self.wrapped_curr);
            }
            add_elapsed_ns(&mut self.perf_unwrap_ns, unwrap_t0);

            let accum_t0 = self.profile_transport.then(std::time::Instant::now);
            if self.total_stream_fastpath {
                let mut sx = 0.0f64;
                let mut sy = 0.0f64;
                let mut sz = 0.0f64;
                for &g in self.charged_groups.iter() {
                    let q = self.group_charge[g];
                    let u = self.unwrap_curr[g];
                    sx += u[0] * q;
                    sy += u[1] * q;
                    sz += u[2] * q;
                }
                self.sample_total_f32[0] = sx as f32;
                self.sample_total_f32[1] = sy as f32;
                self.sample_total_f32[2] = sz as f32;
            }

            if !self.total_stream_fastpath || matches!(self.resolved_mode, LagMode::Fft) {
                for g in 0..self.n_groups {
                    let base = g * 3;
                    self.sample_f32[base] = self.unwrap_curr[g][0] as f32;
                    self.sample_f32[base + 1] = self.unwrap_curr[g][1] as f32;
                    self.sample_f32[base + 2] = self.unwrap_curr[g][2] as f32;
                }
            }

            match self.resolved_mode {
                LagMode::MultiTau => {
                    if let Some(buffer) = &mut self.multi_tau {
                        let dt_dec = self.dt_decimation;
                        let lags = &self.lags;
                        let acc = &mut self.acc;
                        if self.total_stream_fastpath {
                            buffer.update(&self.sample_total_f32, |lag_idx, cur, old| {
                                let lag = lags[lag_idx];
                                if !lag_allowed(lag, dt_dec) {
                                    return;
                                }
                                accumulate_conductivity_streams_total(acc, lag_idx, cur, old);
                            });
                        } else if self.transference {
                            let type_ids = &self.type_ids;
                            let n_types = self.type_counts.len();
                            let type_sums = &mut self.type_disp_sums;
                            let group_charge = &self.group_charge;
                            let charged_groups = &self.charged_groups;
                            buffer.update(&self.sample_f32, |lag_idx, cur, old| {
                                let lag = lags[lag_idx];
                                if !lag_allowed(lag, dt_dec) {
                                    return;
                                }
                                accumulate_conductivity_transference(
                                    acc,
                                    lag_idx,
                                    n_types,
                                    type_ids,
                                    group_charge,
                                    charged_groups,
                                    type_sums,
                                    cur,
                                    old,
                                );
                            });
                        } else {
                            let group_charge = &self.group_charge;
                            let charged_groups = &self.charged_groups;
                            buffer.update(&self.sample_f32, |lag_idx, cur, old| {
                                let lag = lags[lag_idx];
                                if !lag_allowed(lag, dt_dec) {
                                    return;
                                }
                                accumulate_conductivity_total(
                                    acc,
                                    lag_idx,
                                    group_charge,
                                    charged_groups,
                                    cur,
                                    old,
                                );
                            });
                        }
                    }
                }
                LagMode::Ring => {
                    let dt_dec = self.dt_decimation;
                    let acc = &mut self.acc;
                    if let Some(buffer) = &mut self.ring {
                        if self.total_stream_fastpath {
                            buffer.update(&self.sample_total_f32, |lag, cur, old| {
                                if !lag_allowed(lag, dt_dec) {
                                    return;
                                }
                                let lag_idx = lag - 1;
                                accumulate_conductivity_streams_total(acc, lag_idx, cur, old);
                            });
                        } else if self.transference {
                            let type_ids = &self.type_ids;
                            let n_types = self.type_counts.len();
                            let type_sums = &mut self.type_disp_sums;
                            let group_charge = &self.group_charge;
                            let charged_groups = &self.charged_groups;
                            buffer.update(&self.sample_f32, |lag, cur, old| {
                                if !lag_allowed(lag, dt_dec) {
                                    return;
                                }
                                let lag_idx = lag - 1;
                                accumulate_conductivity_transference(
                                    acc,
                                    lag_idx,
                                    n_types,
                                    type_ids,
                                    group_charge,
                                    charged_groups,
                                    type_sums,
                                    cur,
                                    old,
                                );
                            });
                        } else {
                            let group_charge = &self.group_charge;
                            let charged_groups = &self.charged_groups;
                            buffer.update(&self.sample_f32, |lag, cur, old| {
                                if !lag_allowed(lag, dt_dec) {
                                    return;
                                }
                                let lag_idx = lag - 1;
                                accumulate_conductivity_total(
                                    acc,
                                    lag_idx,
                                    group_charge,
                                    charged_groups,
                                    cur,
                                    old,
                                );
                            });
                        }
                    }
                }
                LagMode::Fft => {
                    if self.fft_store_groups {
                        self.series.extend_from_slice(&self.sample_f32);
                    } else {
                        let n_types = self.type_counts.len();
                        let streams = if self.transference { n_types + 1 } else { 1 };
                        if self.fft_stream_sums.len() < streams {
                            self.fft_stream_sums.resize(streams, [0.0; 3]);
                        }
                        let sums = &mut self.fft_stream_sums[..streams];
                        for sum in sums.iter_mut() {
                            *sum = [0.0; 3];
                        }
                        if self.transference {
                            for &g in self.charged_groups.iter() {
                                let q = self.group_charge[g];
                                let base = g * 3;
                                let vec = [
                                    self.sample_f32[base] as f64 * q,
                                    self.sample_f32[base + 1] as f64 * q,
                                    self.sample_f32[base + 2] as f64 * q,
                                ];
                                let t = self.type_ids[g];
                                sums[t][0] += vec[0];
                                sums[t][1] += vec[1];
                                sums[t][2] += vec[2];
                                let total = n_types;
                                sums[total][0] += vec[0];
                                sums[total][1] += vec[1];
                                sums[total][2] += vec[2];
                            }
                        } else {
                            for &g in self.charged_groups.iter() {
                                let q = self.group_charge[g];
                                let base = g * 3;
                                sums[0][0] += self.sample_f32[base] as f64 * q;
                                sums[0][1] += self.sample_f32[base + 1] as f64 * q;
                                sums[0][2] += self.sample_f32[base + 2] as f64 * q;
                            }
                        }
                        for s in 0..streams {
                            self.series.push(sums[s][0] as f32);
                            self.series.push(sums[s][1] as f32);
                            self.series.push(sums[s][2] as f32);
                        }
                    }
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

    pub(super) fn finalize_state(&mut self) -> TrajResult<PlanOutput> {
        if self.profile_transport {
            eprintln!(
                "[transport-profile] plan=conductivity chunks={} frames={} read_ms={:.3} unwrap_ms={:.3} accum_ms={:.3} finalize_ms={:.3}",
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
                let n_types = self.type_counts.len();
                let streams = if self.transference { n_types + 1 } else { 1 };
                self.series.len() / (streams * 3)
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

        let volavg = average_volume(self.vol_sum, self.vol_count)?;
        let kb = 1.380648813e-23_f64;
        let qelec = 1.60217656535e-19_f64;
        let multi = qelec * qelec / (6.0 * kb * self.temperature * volavg);
        let dt0 = self.dt0.unwrap_or(1.0);
        if !self.uniform_time {
            return Err(TrajError::Mismatch(
                "conductivity requires uniform frame spacing for lag-time output".into(),
            ));
        }
        let n_types = self.type_counts.len();
        let cols = if self.transference {
            n_types * n_types + 1
        } else {
            1
        };

        match self.resolved_mode {
            LagMode::Fft => {
                if self.fft_store_groups {
                    #[cfg(feature = "cuda")]
                    if let Some(gpu) = &self.gpu {
                        let ndframe = n_frames.saturating_sub(1);
                        if ndframe > 0 {
                            let times_f32: Vec<f32> =
                                (0..n_frames).map(|i| (dt0 * i as f64) as f32).collect();
                            let group_charge_f32: Vec<f32> =
                                self.group_charge.iter().map(|q| *q as f32).collect();
                            let type_ids_u32: Vec<u32> =
                                self.type_ids.iter().map(|t| *t as u32).collect();
                            let mut type_charge_f32 = vec![0.0f32; n_types];
                            for (g_idx, &t) in self.type_ids.iter().enumerate() {
                                type_charge_f32[t] += self.group_charge[g_idx] as f32;
                            }
                            let time_binning = (
                                self.time_binning.eps_num as f32,
                                self.time_binning.eps_add as f32,
                            );
                            let (out_gpu, n_diff_gpu, cols_gpu) = gpu.ctx.conductivity_time_lag(
                                &self.series,
                                &times_f32,
                                &group_charge_f32,
                                &type_ids_u32,
                                &type_charge_f32,
                                self.n_groups,
                                n_types,
                                ndframe,
                                self.transference,
                                None,
                                self.dt_decimation
                                    .map(|d| (d.cut1, d.stride1, d.cut2, d.stride2)),
                                time_binning,
                            )?;
                            if cols_gpu == cols && !out_gpu.is_empty() {
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
                                        data[out_base + c] =
                                            (out_gpu[base + c] as f64 * multi / count) as f32;
                                    }
                                }
                                if self.transference {
                                    for i in 0..n_types {
                                        for j in 0..i {
                                            let idx = j + i * n_types;
                                            let idx2 = i + j * n_types;
                                            for row in 0..lags.len() {
                                                let base = row * cols;
                                                let val = 0.5 * data[base + idx];
                                                data[base + idx] = val;
                                                data[base + idx2] = val;
                                            }
                                        }
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
                }

                let (lags, acc, counts) = conductivity_fft(
                    &self.series,
                    n_frames,
                    n_types,
                    self.transference,
                    self.dt_decimation,
                )?;
                Ok(build_conductivity_output(
                    &lags, &acc, &counts, dt0, cols, multi,
                ))
            }
            LagMode::Ring => {
                let raw_counts: &[u64] = self.ring.as_ref().map(|r| r.n_pairs()).unwrap_or(&[]);
                let counts: Vec<u64> = self
                    .lags
                    .iter()
                    .map(|&lag| raw_counts.get(lag).copied().unwrap_or(0))
                    .collect();
                Ok(build_conductivity_output(
                    &self.lags, &self.acc, &counts, dt0, cols, multi,
                ))
            }
            LagMode::MultiTau => {
                let counts: &[u64] = self.multi_tau.as_ref().map(|m| m.n_pairs()).unwrap_or(&[]);
                Ok(build_conductivity_output(
                    &self.lags, &self.acc, &counts, dt0, cols, multi,
                ))
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
