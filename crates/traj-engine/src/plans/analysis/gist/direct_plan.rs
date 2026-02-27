impl Plan for GistDirectPlan {
    fn name(&self) -> &'static str {
        "gist_direct"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        if self.auto_grid {
            self.counts.clear();
            self.orient_counts.clear();
            self.energy_sw.clear();
            self.energy_ww.clear();
            self.dims = [0, 0, 0];
            self.origin = [0.0, 0.0, 0.0];
        } else {
            self.counts.fill(0);
            self.orient_counts.fill(0);
            self.energy_sw.fill(0.0);
            self.energy_ww.fill(0.0);
        }
        self.direct_sw_total = 0.0;
        self.direct_ww_total = 0.0;
        self.frame_direct_sw.clear();
        self.frame_direct_ww.clear();
        self.frame_pme_sw.clear();
        self.frame_pme_ww.clear();
        self.frame_offsets.clear();
        self.frame_cells.clear();
        self.frame_sw.clear();
        self.frame_ww.clear();
        if self.record_frame_energies {
            self.frame_offsets.push(0);
        }
        self.n_frames = 0;
        self.global_frame = 0;
        self.frame_filter_pos = 0;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let mut water_offsets_u32 = Vec::with_capacity(self.water_offsets.len());
                for &off in self.water_offsets.iter() {
                    if off > u32::MAX as usize {
                        return Err(TrajError::Mismatch(
                            "gist water offset exceeds u32 range".into(),
                        ));
                    }
                    water_offsets_u32.push(off as u32);
                }
                let water_atoms_u32 = self.water_atoms.clone();
                let solute_atoms_u32 = self.solute_indices.clone();
                let mut ex_i = Vec::with_capacity(self.exceptions.len());
                let mut ex_j = Vec::with_capacity(self.exceptions.len());
                let mut ex_qprod = Vec::with_capacity(self.exceptions.len());
                let mut ex_sigma = Vec::with_capacity(self.exceptions.len());
                let mut ex_epsilon = Vec::with_capacity(self.exceptions.len());
                let orient_valid: Vec<u32> = self
                    .orientation_valid
                    .iter()
                    .map(|&v| if v != 0 { 1u32 } else { 0u32 })
                    .collect();
                for (&key, pair) in self.exceptions.iter() {
                    let a = (key >> 32) as u32;
                    let b = (key & 0xFFFF_FFFF) as u32;
                    ex_i.push(a);
                    ex_j.push(b);
                    ex_qprod.push(pair.qprod as f32);
                    ex_sigma.push(pair.sigma as f32);
                    ex_epsilon.push(pair.epsilon as f32);
                }
                let gpu = GistDirectGpuState {
                    ctx: ctx.clone(),
                    oxygen_idx: ctx.upload_u32(&self.oxygen_indices)?,
                    h1_idx: ctx.upload_u32(&self.hydrogen1_indices)?,
                    h2_idx: ctx.upload_u32(&self.hydrogen2_indices)?,
                    orient_valid: ctx.upload_u32(&orient_valid)?,
                    counts: None,
                    orient_counts: None,
                    n_cells: 0,
                    water_offsets: ctx.upload_u32(&water_offsets_u32)?,
                    water_atoms: ctx.upload_u32(&water_atoms_u32)?,
                    solute_atoms: ctx.upload_u32(&solute_atoms_u32)?,
                    charges: ctx
                        .upload_f32(&self.charges.iter().map(|&v| v as f32).collect::<Vec<_>>())?,
                    sigmas: ctx
                        .upload_f32(&self.sigmas.iter().map(|&v| v as f32).collect::<Vec<_>>())?,
                    epsilons: ctx
                        .upload_f32(&self.epsilons.iter().map(|&v| v as f32).collect::<Vec<_>>())?,
                    ex_i: ctx.upload_u32(&ex_i)?,
                    ex_j: ctx.upload_u32(&ex_j)?,
                    ex_qprod: ctx.upload_f32(&ex_qprod)?,
                    ex_sigma: ctx.upload_f32(&ex_sigma)?,
                    ex_epsilon: ctx.upload_f32(&ex_epsilon)?,
                };
                self.gpu = Some(gpu);
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
        let n_atoms = chunk.n_atoms;
        if self.charges.len() < n_atoms
            || self.sigmas.len() < n_atoms
            || self.epsilons.len() < n_atoms
        {
            return Err(TrajError::Mismatch(
                "gist nonbonded parameter arrays shorter than trajectory atom count".into(),
            ));
        }
        let n_waters = self.oxygen_indices.len();
        let n_cells = self.dims[0] * self.dims[1] * self.dims[2];
        if self.counts.len() != n_cells
            || self.orient_counts.len() != n_cells * self.orientation_bins
            || self.energy_sw.len() != n_cells
            || self.energy_ww.len() != n_cells
        {
            return Err(TrajError::Mismatch(
                "gist direct buffer shape mismatch".into(),
            ));
        }

        #[cfg(feature = "cuda")]
        let coords_gpu: Option<GpuCoords> = if let Some(gpu) = &self.gpu {
            Some(gpu.ctx.upload_coords(&convert_coords(&chunk.coords))?)
        } else {
            None
        };

        for local_frame in 0..chunk.n_frames {
            let abs_frame = self.global_frame + local_frame;
            if !self.keep_frame(abs_frame) {
                continue;
            }
            self.ensure_grid_for_frame(chunk, local_frame)?;
            let pbc = self.pbc_geometry(chunk, local_frame)?;
            let center = if self.solute_indices.is_empty() {
                mean_center_all_atoms(chunk, local_frame, self.length_scale)?
            } else {
                mean_center_indices(chunk, local_frame, self.length_scale, &self.solute_indices)?
            };

            let mut water_voxels: Vec<Option<usize>> = vec![None; n_waters];
            let mut frame_sw_total = 0.0f64;
            let mut frame_ww_total = 0.0f64;
            let mut frame_pme_sw_total = 0.0f64;
            let mut frame_pme_ww_total = 0.0f64;
            let track_pme_totals = self.record_pme_frame_totals;
            let mut frame_sparse = if self.record_frame_energies {
                Some(HashMap::<usize, [f64; 2]>::new())
            } else {
                None
            };
            let used_gpu_counts = {
                #[cfg(feature = "cuda")]
                {
                    if let (Some(gpu), Some(coords_gpu), Device::Cuda(_)) =
                        (self.gpu.as_mut(), coords_gpu.as_ref(), _device)
                    {
                        let n_cells = self.dims[0] * self.dims[1] * self.dims[2];
                        ensure_gist_gpu_hist_buffers(
                            &gpu.ctx,
                            &mut gpu.counts,
                            &mut gpu.orient_counts,
                            &mut gpu.n_cells,
                            n_cells,
                            self.orientation_bins,
                        )?;
                        let (cells_dev, bins_dev) = gpu.ctx.gist_counts_orient_frame_device(
                            coords_gpu,
                            n_atoms,
                            local_frame,
                            &gpu.oxygen_idx,
                            &gpu.h1_idx,
                            &gpu.h2_idx,
                            &gpu.orient_valid,
                            n_waters,
                            [center[0] as f32, center[1] as f32, center[2] as f32],
                            [
                                self.origin[0] as f32,
                                self.origin[1] as f32,
                                self.origin[2] as f32,
                            ],
                            self.spacing as f32,
                            self.dims,
                            self.orientation_bins,
                            self.length_scale as f32,
                        )?;
                        let (Some(counts_dev), Some(orient_counts_dev)) =
                            (gpu.counts.as_mut(), gpu.orient_counts.as_mut())
                        else {
                            return Err(TrajError::Mismatch(
                                "gist cuda histogram buffers not initialized".into(),
                            ));
                        };
                        gpu.ctx.gist_accumulate_hist(
                            &cells_dev,
                            &bins_dev,
                            n_waters,
                            self.orientation_bins,
                            counts_dev,
                            orient_counts_dev,
                        )?;
                        let cells = gpu.ctx.download_u32(&cells_dev, n_waters)?;
                        for i in 0..n_waters {
                            let cell = cells[i];
                            if cell == u32::MAX {
                                continue;
                            }
                            let flat = cell as usize;
                            if flat >= self.counts.len() {
                                continue;
                            }
                            water_voxels[i] = Some(flat);
                        }
                        true
                    } else {
                        false
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    false
                }
            };
            if !used_gpu_counts {
                let frame_base = local_frame * n_atoms;
                for i in 0..n_waters {
                    let oxy_idx = self.oxygen_indices[i] as usize;
                    if oxy_idx >= n_atoms {
                        return Err(TrajError::Mismatch(
                            "gist oxygen index out of bounds".into(),
                        ));
                    }
                    let po = chunk.coords[frame_base + oxy_idx];
                    let ox = po[0] as f64 * self.length_scale;
                    let oy = po[1] as f64 * self.length_scale;
                    let oz = po[2] as f64 * self.length_scale;
                    let Some(flat) = voxel_flat([ox, oy, oz], self.origin, self.spacing, self.dims)
                    else {
                        continue;
                    };
                    water_voxels[i] = Some(flat);
                    self.counts[flat] = self.counts[flat].saturating_add(1);

                    if self.orientation_valid[i] == 0 {
                        continue;
                    }
                    let h1_idx = self.hydrogen1_indices[i] as usize;
                    let h2_idx = self.hydrogen2_indices[i] as usize;
                    if h1_idx >= n_atoms || h2_idx >= n_atoms {
                        continue;
                    }
                    let ph1 = chunk.coords[frame_base + h1_idx];
                    let ph2 = chunk.coords[frame_base + h2_idx];
                    let hmid = [
                        0.5 * ((ph1[0] as f64 + ph2[0] as f64) * self.length_scale),
                        0.5 * ((ph1[1] as f64 + ph2[1] as f64) * self.length_scale),
                        0.5 * ((ph1[2] as f64 + ph2[2] as f64) * self.length_scale),
                    ];
                    let hvec = [hmid[0] - ox, hmid[1] - oy, hmid[2] - oz];
                    let rvec = [ox - center[0], oy - center[1], oz - center[2]];
                    let Some(bin) = orientation_bin(hvec, rvec, self.orientation_bins) else {
                        continue;
                    };
                    let orient_flat = flat * self.orientation_bins + bin;
                    self.orient_counts[orient_flat] =
                        self.orient_counts[orient_flat].saturating_add(1);
                }
            }

            let mut water_sw = vec![0.0f64; n_waters];
            let mut water_ww = vec![0.0f64; n_waters];
            let used_gpu = {
                #[cfg(feature = "cuda")]
                {
                    if let (Some(gpu), Some(coords_gpu), Device::Cuda(_)) =
                        (self.gpu.as_ref(), coords_gpu.as_ref(), _device)
                    {
                        let mut pbc_mode_gpu = 0i32;
                        let mut box_lengths_f32 = [0.0f32, 0.0f32, 0.0f32];
                        let mut cell_f32 = [0.0f32; 9];
                        let mut inv_f32 = [0.0f32; 9];
                        match pbc {
                            GistPbc::None => {}
                            GistPbc::Orthorhombic { lx, ly, lz } => {
                                pbc_mode_gpu = 1;
                                box_lengths_f32 = [lx as f32, ly as f32, lz as f32];
                            }
                            GistPbc::Triclinic { cell, inv } => {
                                pbc_mode_gpu = 2;
                                for row in 0..3 {
                                    for col in 0..3 {
                                        let idx = row * 3 + col;
                                        cell_f32[idx] = cell[row][col] as f32;
                                        inv_f32[idx] = inv[row][col] as f32;
                                    }
                                }
                            }
                        }
                        let (sw_gpu, ww_gpu) = gpu.ctx.gist_direct_energy_frame(
                            coords_gpu,
                            n_atoms,
                            local_frame,
                            &gpu.water_offsets,
                            &gpu.water_atoms,
                            n_waters,
                            &gpu.solute_atoms,
                            self.solute_indices.len(),
                            &gpu.charges,
                            &gpu.sigmas,
                            &gpu.epsilons,
                            &gpu.ex_i,
                            &gpu.ex_j,
                            &gpu.ex_qprod,
                            &gpu.ex_sigma,
                            &gpu.ex_epsilon,
                            self.exceptions.len(),
                            pbc_mode_gpu,
                            box_lengths_f32,
                            cell_f32,
                            inv_f32,
                            self.cutoff as f32,
                            self.length_scale as f32,
                        )?;
                        for i in 0..n_waters {
                            water_sw[i] = sw_gpu[i] as f64;
                            water_ww[i] = ww_gpu[i] as f64;
                        }
                        true
                    } else {
                        false
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    false
                }
            };

            if !used_gpu {
                if !self.solute_indices.is_empty() {
                    for i in 0..n_waters {
                        if !track_pme_totals && water_voxels[i].is_none() {
                            continue;
                        }
                        water_sw[i] = self.group_energy(
                            chunk,
                            local_frame,
                            self.water_atom_slice(i),
                            &self.solute_indices,
                            pbc,
                        )?;
                    }
                }
                for i in 0..n_waters {
                    for j in (i + 1)..n_waters {
                        let vi = water_voxels[i];
                        let vj = water_voxels[j];
                        if !track_pme_totals && vi.is_none() && vj.is_none() {
                            continue;
                        }
                        let e_ij = self.group_energy(
                            chunk,
                            local_frame,
                            self.water_atom_slice(i),
                            self.water_atom_slice(j),
                            pbc,
                        )?;
                        if e_ij == 0.0 {
                            continue;
                        }
                        let half = 0.5 * e_ij;
                        water_ww[i] += half;
                        water_ww[j] += half;
                    }
                }
            }
            if track_pme_totals {
                frame_pme_sw_total = water_sw.iter().sum();
                frame_pme_ww_total = water_ww.iter().sum();
            }

            for i in 0..n_waters {
                let Some(cell) = water_voxels[i] else {
                    continue;
                };
                let e_sw = water_sw[i];
                if e_sw != 0.0 {
                    self.energy_sw[cell] += e_sw;
                    self.direct_sw_total += e_sw;
                    frame_sw_total += e_sw;
                    if let Some(sparse) = frame_sparse.as_mut() {
                        let entry = sparse.entry(cell).or_insert([0.0, 0.0]);
                        entry[0] += e_sw;
                    }
                }
                let e_ww = water_ww[i];
                if e_ww != 0.0 {
                    self.energy_ww[cell] += e_ww;
                    self.direct_ww_total += e_ww;
                    frame_ww_total += e_ww;
                    if let Some(sparse) = frame_sparse.as_mut() {
                        let entry = sparse.entry(cell).or_insert([0.0, 0.0]);
                        entry[1] += e_ww;
                    }
                }
            }

            if self.record_frame_energies {
                self.frame_direct_sw.push(frame_sw_total);
                self.frame_direct_ww.push(frame_ww_total);
                let mut entries: Vec<(usize, [f64; 2])> =
                    frame_sparse.unwrap_or_default().into_iter().collect();
                entries.sort_unstable_by_key(|(cell, _)| *cell);
                for (cell, [sw, ww]) in entries.into_iter() {
                    if cell > u32::MAX as usize {
                        return Err(TrajError::Mismatch(
                            "gist cell index exceeds u32 range".into(),
                        ));
                    }
                    self.frame_cells.push(cell as u32);
                    self.frame_sw.push(sw);
                    self.frame_ww.push(ww);
                }
                self.frame_offsets.push(self.frame_cells.len());
            }
            if track_pme_totals {
                self.frame_pme_sw.push(frame_pme_sw_total);
                self.frame_pme_ww.push(frame_pme_ww_total);
            }

            self.n_frames += 1;
        }
        self.global_frame += chunk.n_frames;
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        #[cfg(feature = "cuda")]
        {
            if let Some(gpu) = self.gpu.as_ref() {
                let n_cells = self.dims[0] * self.dims[1] * self.dims[2];
                if n_cells != 0 {
                    let orient_len =
                        n_cells.checked_mul(self.orientation_bins).ok_or_else(|| {
                            TrajError::Mismatch("gist orientation histogram size overflow".into())
                        })?;
                    if let (Some(counts_dev), Some(orient_counts_dev)) =
                        (gpu.counts.as_ref(), gpu.orient_counts.as_ref())
                    {
                        self.counts = gpu.ctx.download_u32(counts_dev, n_cells)?;
                        self.orient_counts = gpu.ctx.download_u32(orient_counts_dev, orient_len)?;
                    }
                }
            }
        }
        if self.record_frame_energies {
            if self.frame_offsets.len() != self.n_frames + 1 {
                return Err(TrajError::Mismatch(
                    "gist frame_offsets length must equal n_frames + 1".into(),
                ));
            }
            if self.frame_offsets.first().copied().unwrap_or(usize::MAX) != 0 {
                return Err(TrajError::Mismatch(
                    "gist frame_offsets must start at 0".into(),
                ));
            }
            if self.frame_offsets.windows(2).any(|pair| pair[1] < pair[0]) {
                return Err(TrajError::Mismatch(
                    "gist frame_offsets must be non-decreasing".into(),
                ));
            }
            if self.frame_direct_sw.len() != self.n_frames
                || self.frame_direct_ww.len() != self.n_frames
            {
                return Err(TrajError::Mismatch(
                    "gist frame direct totals length must equal n_frames".into(),
                ));
            }
            if self.frame_cells.len() != self.frame_sw.len()
                || self.frame_cells.len() != self.frame_ww.len()
            {
                return Err(TrajError::Mismatch(
                    "gist sparse frame vectors must have identical length".into(),
                ));
            }
            if self.frame_offsets.last().copied().unwrap_or(usize::MAX) != self.frame_cells.len() {
                return Err(TrajError::Mismatch(
                    "gist frame_offsets end must equal sparse vector length".into(),
                ));
            }
        } else if !self.frame_offsets.is_empty()
            || !self.frame_cells.is_empty()
            || !self.frame_sw.is_empty()
            || !self.frame_ww.is_empty()
            || !self.frame_direct_sw.is_empty()
            || !self.frame_direct_ww.is_empty()
        {
            return Err(TrajError::Mismatch(
                "gist frame vectors present but frame energy tracking disabled".into(),
            ));
        }
        if self.record_pme_frame_totals {
            if self.frame_pme_sw.len() != self.n_frames || self.frame_pme_ww.len() != self.n_frames
            {
                return Err(TrajError::Mismatch(
                    "gist native PME frame totals length must equal n_frames".into(),
                ));
            }
        } else if !self.frame_pme_sw.is_empty() || !self.frame_pme_ww.is_empty() {
            return Err(TrajError::Mismatch(
                "gist PME frame totals present but PME tracking disabled".into(),
            ));
        }
        finalize_counts_orientation(
            &self.counts,
            &self.orient_counts,
            self.dims,
            self.orientation_bins,
        )
    }
}
