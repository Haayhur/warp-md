use super::*;

impl GistDirectPlan {
    pub(super) fn reset_runtime(&mut self) {
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
    }

    pub(super) fn init_gpu_state(&mut self, _device: &Device) -> TrajResult<()> {
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
                self.gpu = Some(GistDirectGpuState {
                    ctx: ctx.clone(),
                    oxygen_idx: ctx.upload_u32(&self.oxygen_indices)?,
                    h1_idx: ctx.upload_u32(&self.hydrogen1_indices)?,
                    h2_idx: ctx.upload_u32(&self.hydrogen2_indices)?,
                    orient_valid: ctx.upload_u32(&orient_valid)?,
                    counts: None,
                    orient_counts: None,
                    n_cells: 0,
                    water_offsets: ctx.upload_u32(&water_offsets_u32)?,
                    water_atoms: ctx.upload_u32(&self.water_atoms)?,
                    solute_atoms: ctx.upload_u32(&self.solute_indices)?,
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
                });
            }
        }
        Ok(())
    }

    pub(super) fn validate_process_inputs(&self, n_atoms: usize) -> TrajResult<()> {
        if self.charges.len() < n_atoms
            || self.sigmas.len() < n_atoms
            || self.epsilons.len() < n_atoms
        {
            return Err(TrajError::Mismatch(
                "gist nonbonded parameter arrays shorter than trajectory atom count".into(),
            ));
        }
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
        Ok(())
    }

    pub(super) fn finalize_histograms(&mut self) -> TrajResult<()> {
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
        Ok(())
    }

    pub(super) fn validate_final_state(&self) -> TrajResult<()> {
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
        Ok(())
    }
}
