use super::*;

impl GistDirectPlan {
    pub(super) fn keep_frame(&mut self, abs_frame: usize) -> bool {
        keep_frame_internal(
            self.max_frames,
            self.n_frames,
            self.frame_filter.as_ref(),
            &mut self.frame_filter_pos,
            abs_frame,
        )
    }

    pub(super) fn ensure_grid_for_frame(
        &mut self,
        chunk: &FrameChunk,
        frame: usize,
    ) -> TrajResult<()> {
        if !self.auto_grid || !self.counts.is_empty() {
            return Ok(());
        }
        let n_atoms = chunk.n_atoms;
        if n_atoms == 0 {
            return Err(TrajError::Mismatch(
                "gist grid requires at least one atom".into(),
            ));
        }
        let base = frame * n_atoms;
        let use_all = self.solute_indices.is_empty();
        let mut min_xyz = [f64::INFINITY; 3];
        let mut max_xyz = [f64::NEG_INFINITY; 3];
        if use_all {
            for atom_idx in 0..n_atoms {
                let p = chunk.coords[base + atom_idx];
                let x = p[0] as f64 * self.length_scale;
                let y = p[1] as f64 * self.length_scale;
                let z = p[2] as f64 * self.length_scale;
                min_xyz[0] = min_xyz[0].min(x);
                min_xyz[1] = min_xyz[1].min(y);
                min_xyz[2] = min_xyz[2].min(z);
                max_xyz[0] = max_xyz[0].max(x);
                max_xyz[1] = max_xyz[1].max(y);
                max_xyz[2] = max_xyz[2].max(z);
            }
        } else {
            for &idx in self.solute_indices.iter() {
                let atom_idx = idx as usize;
                if atom_idx >= n_atoms {
                    return Err(TrajError::Mismatch(
                        "gist solute selection index out of bounds".into(),
                    ));
                }
                let p = chunk.coords[base + atom_idx];
                let x = p[0] as f64 * self.length_scale;
                let y = p[1] as f64 * self.length_scale;
                let z = p[2] as f64 * self.length_scale;
                min_xyz[0] = min_xyz[0].min(x);
                min_xyz[1] = min_xyz[1].min(y);
                min_xyz[2] = min_xyz[2].min(z);
                max_xyz[0] = max_xyz[0].max(x);
                max_xyz[1] = max_xyz[1].max(y);
                max_xyz[2] = max_xyz[2].max(z);
            }
        }
        self.origin = [
            min_xyz[0] - self.padding,
            min_xyz[1] - self.padding,
            min_xyz[2] - self.padding,
        ];
        self.dims = dims_from_bounds(min_xyz, max_xyz, self.padding, self.spacing);
        let n_cells = self.dims[0] * self.dims[1] * self.dims[2];
        self.counts = vec![0u32; n_cells];
        self.orient_counts = vec![0u32; n_cells * self.orientation_bins];
        self.energy_sw = vec![0.0f64; n_cells];
        self.energy_ww = vec![0.0f64; n_cells];
        Ok(())
    }

    pub(super) fn water_atom_slice(&self, water_idx: usize) -> &[u32] {
        let start = self.water_offsets[water_idx];
        let end = self.water_offsets[water_idx + 1];
        &self.water_atoms[start..end]
    }

    pub(super) fn pbc_geometry(&self, chunk: &FrameChunk, frame: usize) -> TrajResult<GistPbc> {
        if !self.periodic {
            return Ok(GistPbc::None);
        }
        match chunk.box_[frame] {
            Box3::Orthorhombic { lx, ly, lz } => Ok(GistPbc::Orthorhombic {
                lx: lx as f64 * self.length_scale,
                ly: ly as f64 * self.length_scale,
                lz: lz as f64 * self.length_scale,
            }),
            Box3::Triclinic { m } => {
                let mut scaled = [0.0f32; 9];
                for (i, value) in m.into_iter().enumerate() {
                    scaled[i] = value * self.length_scale as f32;
                }
                let (cell, inv) = pbc_math::cell_and_inv_from_box(Box3::Triclinic { m: scaled })?;
                Ok(GistPbc::Triclinic { cell, inv })
            }
            Box3::None => Err(TrajError::Mismatch(
                "gist direct with periodic=true requires box vectors".into(),
            )),
        }
    }
}
