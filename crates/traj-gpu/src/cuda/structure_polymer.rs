use super::*;

impl GpuContext {
    pub fn align_covariance(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
        reference: &GpuReference,
    ) -> TrajResult<AlignCovariance> {
        if n_frames == 0 {
            return Ok(AlignCovariance {
                cov: Vec::new(),
                sum_x: Vec::new(),
                sum_y: Vec::new(),
                sum_w: Vec::new(),
            });
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut sum_x = stream
            .alloc_zeros::<f32>(n_frames * 3)
            .map_err(map_driver_err)?;
        let mut sum_y = stream
            .alloc_zeros::<f32>(n_frames * 3)
            .map_err(map_driver_err)?;
        let mut sum_w = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;
        let mut cov = stream
            .alloc_zeros::<f32>(n_frames * 9)
            .map_err(map_driver_err)?;

        let n_sel = sel.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(sel.n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.align_centroid);
            builder.arg(&coords_dev);
            builder.arg(&reference.coords);
            builder.arg(&sel.sel);
            builder.arg(&sel.masses);
            builder.arg(&n_sel);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&mut sum_x);
            builder.arg(&mut sum_y);
            builder.arg(&mut sum_w);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.align_cov);
            builder.arg(&coords_dev);
            builder.arg(&reference.coords);
            builder.arg(&sel.sel);
            builder.arg(&sel.masses);
            builder.arg(&n_sel);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&sum_x);
            builder.arg(&sum_y);
            builder.arg(&sum_w);
            builder.arg(&mut cov);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut cov_host = vec![0.0f32; n_frames * 9];
        let mut sum_x_host = vec![0.0f32; n_frames * 3];
        let mut sum_y_host = vec![0.0f32; n_frames * 3];
        let mut sum_w_host = vec![0.0f32; n_frames];
        stream
            .memcpy_dtoh(&cov, &mut cov_host)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&sum_x, &mut sum_x_host)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&sum_y, &mut sum_y_host)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&sum_w, &mut sum_w_host)
            .map_err(map_driver_err)?;

        let mut cov_frames = Vec::with_capacity(n_frames);
        let mut sum_x_frames = Vec::with_capacity(n_frames);
        let mut sum_y_frames = Vec::with_capacity(n_frames);
        for frame in 0..n_frames {
            let cov_offset = frame * 9;
            let mut mat = [0.0f32; 9];
            mat.copy_from_slice(&cov_host[cov_offset..cov_offset + 9]);
            cov_frames.push(mat);
            let sum_offset = frame * 3;
            sum_x_frames.push([
                sum_x_host[sum_offset],
                sum_x_host[sum_offset + 1],
                sum_x_host[sum_offset + 2],
            ]);
            sum_y_frames.push([
                sum_y_host[sum_offset],
                sum_y_host[sum_offset + 1],
                sum_y_host[sum_offset + 2],
            ]);
        }

        Ok(AlignCovariance {
            cov: cov_frames,
            sum_x: sum_x_frames,
            sum_y: sum_y_frames,
            sum_w: sum_w_host,
        })
    }

    pub fn inertia_tensor(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
        centers: &[Float4],
    ) -> TrajResult<Vec<[f32; 6]>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if centers.len() != n_frames {
            return Err(TrajError::Mismatch("center buffer mismatch".into()));
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let centers_dev = stream.clone_htod(centers).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * 6)
            .map_err(map_driver_err)?;

        let n_sel = sel.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(sel.n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.inertia_tensor);
            builder.arg(&coords_dev);
            builder.arg(&sel.sel);
            builder.arg(&sel.masses);
            builder.arg(&n_sel);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&centers_dev);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames * 6];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        let mut frames = Vec::with_capacity(n_frames);
        for frame in 0..n_frames {
            let offset = frame * 6;
            let mut vals = [0.0f32; 6];
            vals.copy_from_slice(&host[offset..offset + 6]);
            frames.push(vals);
        }
        Ok(frames)
    }

    pub fn rmsd_per_res_accum(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        groups: &GpuGroups,
        reference: &GpuReference,
        rotations: &[f32],
        cx: &[f32],
        cy: &[f32],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || groups.n_groups == 0 || groups.max_len == 0 {
            return Ok(Vec::new());
        }
        if rotations.len() != n_frames * 9 || cx.len() != n_frames * 3 || cy.len() != n_frames * 3 {
            return Err(TrajError::Mismatch(
                "rotation/centroid buffer mismatch".into(),
            ));
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let rotations_dev = stream.clone_htod(rotations).map_err(map_driver_err)?;
        let cx_dev = stream.clone_htod(cx).map_err(map_driver_err)?;
        let cy_dev = stream.clone_htod(cy).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * groups.n_groups)
            .map_err(map_driver_err)?;

        let n_groups = groups.n_groups as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let max_len = groups.max_len as i32;

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (
                ceil_div(groups.max_len.max(1), block),
                n_frames as u32,
                n_groups as u32,
            ),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.rmsd_per_res_accum);
            builder.arg(&coords_dev);
            builder.arg(&reference.coords);
            builder.arg(&groups.offsets);
            builder.arg(&groups.indices);
            builder.arg(&n_groups);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&max_len);
            builder.arg(&rotations_dev);
            builder.arg(&cx_dev);
            builder.arg(&cy_dev);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames * groups.n_groups];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn rdf_accum(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel_a: &GpuSelection,
        sel_b: &GpuSelection,
        r_max: f32,
        bins: usize,
        pbc: bool,
        box_l: Option<&[f32]>,
        same_sel: bool,
        counts: &mut GpuCounts,
    ) -> TrajResult<()> {
        if n_frames == 0 || bins == 0 {
            return Ok(());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let bin_width = r_max / bins as f32;
        let pbc_flag = if pbc { 1i32 } else { 0i32 };
        let same_flag = if same_sel { 1i32 } else { 0i32 };
        let n_a = sel_a.n_sel as i32;
        let n_b = sel_b.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let box_dev = if pbc {
            let box_len = box_l.ok_or_else(|| TrajError::Mismatch("missing box lengths".into()))?;
            Some(stream.clone_htod(box_len).map_err(map_driver_err)?)
        } else {
            None
        };

        let total_pairs = sel_a.n_sel * sel_b.n_sel;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(total_pairs, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        let dummy_box = if !pbc {
            Some(stream.alloc_zeros::<f32>(1).map_err(map_driver_err)?)
        } else {
            None
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.rdf_hist);
            builder.arg(&coords_dev);
            builder.arg(&sel_a.sel);
            builder.arg(&sel_b.sel);
            builder.arg(&n_a);
            builder.arg(&n_b);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&r_max);
            builder.arg(&bin_width);
            builder.arg(&pbc_flag);
            if let Some(box_dev) = &box_dev {
                builder.arg(box_dev);
            } else if let Some(dummy) = &dummy_box {
                builder.arg(dummy);
            }
            builder.arg(&mut counts.inner);
            builder.arg(&same_flag);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        Ok(())
    }

    pub fn polymer_end_to_end(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        polymer: &GpuPolymer,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || polymer.n_chains == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * polymer.n_chains)
            .map_err(map_driver_err)?;
        let n_chains = polymer.n_chains as i32;
        let n_atoms = n_atoms as i32;
        let n_frames = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(polymer.n_chains, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.polymer_end_to_end);
            builder.arg(&coords_dev);
            builder.arg(&polymer.chain_offsets);
            builder.arg(&polymer.chain_indices);
            builder.arg(&n_chains);
            builder.arg(&n_atoms);
            builder.arg(&n_frames);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![0.0f32; n_frames as usize * polymer.n_chains];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn polymer_contour_length(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        polymer: &GpuPolymer,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || polymer.n_chains == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * polymer.n_chains)
            .map_err(map_driver_err)?;
        let n_chains = polymer.n_chains as i32;
        let n_atoms = n_atoms as i32;
        let n_frames = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(polymer.n_chains, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.polymer_contour_length);
            builder.arg(&coords_dev);
            builder.arg(&polymer.chain_offsets);
            builder.arg(&polymer.chain_indices);
            builder.arg(&n_chains);
            builder.arg(&n_atoms);
            builder.arg(&n_frames);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![0.0f32; n_frames as usize * polymer.n_chains];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn polymer_chain_rg(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        polymer: &GpuPolymer,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || polymer.n_chains == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * polymer.n_chains)
            .map_err(map_driver_err)?;
        let n_chains = polymer.n_chains as i32;
        let n_atoms = n_atoms as i32;
        let n_frames = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(polymer.n_chains, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.polymer_chain_rg);
            builder.arg(&coords_dev);
            builder.arg(&polymer.chain_offsets);
            builder.arg(&polymer.chain_indices);
            builder.arg(&n_chains);
            builder.arg(&n_atoms);
            builder.arg(&n_frames);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![0.0f32; n_frames as usize * polymer.n_chains];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn polymer_bond_hist(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        polymer: &GpuPolymer,
        r_max: f32,
        bins: usize,
        counts: &mut GpuCounts,
    ) -> TrajResult<()> {
        if n_frames == 0 || bins == 0 || polymer.n_bonds == 0 {
            return Ok(());
        }
        let bond_pairs = polymer
            .bond_pairs
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("missing bond pairs on GPU".into()))?;
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let bin_width = r_max / bins as f32;
        let n_bonds = polymer.n_bonds as i32;
        let n_atoms = n_atoms as i32;
        let n_frames = n_frames as i32;
        let bins_i = bins as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(polymer.n_bonds, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.polymer_bond_hist);
            builder.arg(&coords_dev);
            builder.arg(bond_pairs);
            builder.arg(&n_bonds);
            builder.arg(&n_atoms);
            builder.arg(&n_frames);
            builder.arg(&r_max);
            builder.arg(&bin_width);
            builder.arg(&bins_i);
            builder.arg(&mut counts.inner);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        Ok(())
    }

    pub fn polymer_angle_hist(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        polymer: &GpuPolymer,
        max_angle: f32,
        bins: usize,
        degrees: bool,
        counts: &mut GpuCounts,
    ) -> TrajResult<()> {
        if n_frames == 0 || bins == 0 || polymer.n_angles == 0 {
            return Ok(());
        }
        let angle_triplets = polymer
            .angle_triplets
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("missing angle triplets on GPU".into()))?;
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let bin_width = max_angle / bins as f32;
        let n_angles = polymer.n_angles as i32;
        let n_atoms = n_atoms as i32;
        let n_frames = n_frames as i32;
        let bins_i = bins as i32;
        let deg_flag = if degrees { 1i32 } else { 0i32 };
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(polymer.n_angles, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.polymer_angle_hist);
            builder.arg(&coords_dev);
            builder.arg(angle_triplets);
            builder.arg(&n_angles);
            builder.arg(&n_atoms);
            builder.arg(&n_frames);
            builder.arg(&max_angle);
            builder.arg(&bin_width);
            builder.arg(&bins_i);
            builder.arg(&deg_flag);
            builder.arg(&mut counts.inner);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        Ok(())
    }
}
