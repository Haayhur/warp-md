use super::*;

impl GistDirectPlan {
    pub(super) fn process_frame(
        &mut self,
        chunk: &FrameChunk,
        _device: &Device,
        local_frame: usize,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        let n_waters = self.oxygen_indices.len();
        self.ensure_grid_for_frame(chunk, local_frame)?;
        let pbc = self.pbc_geometry(chunk, local_frame)?;
        let center = if self.solute_indices.is_empty() {
            mean_center_all_atoms(chunk, local_frame, self.length_scale)?
        } else {
            mean_center_indices(chunk, local_frame, self.length_scale, &self.solute_indices)?
        };

        #[cfg(feature = "cuda")]
        let coords_gpu: Option<GpuCoords> = if let Some(gpu) = &self.gpu {
            Some(gpu.ctx.upload_coords(&convert_coords(&chunk.coords))?)
        } else {
            None
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
                self.orient_counts[orient_flat] = self.orient_counts[orient_flat].saturating_add(1);
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
        Ok(())
    }
}
