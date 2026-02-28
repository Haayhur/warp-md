use super::*;

impl GpuContext {
    pub fn group_com(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        groups: &GpuGroups,
        masses: &GpuBufferF32,
        scale: f32,
    ) -> TrajResult<Vec<Float4>> {
        if n_frames == 0 || groups.n_groups == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let total = n_frames * groups.n_groups;
        let mut sum_x = stream.alloc_zeros::<f32>(total).map_err(map_driver_err)?;
        let mut sum_y = stream.alloc_zeros::<f32>(total).map_err(map_driver_err)?;
        let mut sum_z = stream.alloc_zeros::<f32>(total).map_err(map_driver_err)?;
        let mut mass_sum = stream.alloc_zeros::<f32>(total).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<Float4>(total)
            .map_err(map_driver_err)?;

        let n_groups = groups.n_groups as i32;
        let n_atoms = n_atoms as i32;
        let n_frames = n_frames as i32;
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
            let mut builder = stream.launch_builder(&self.inner.kernels.group_com_accum);
            builder.arg(&coords_dev);
            builder.arg(&groups.offsets);
            builder.arg(&groups.indices);
            builder.arg(&masses.inner);
            builder.arg(&n_groups);
            builder.arg(&n_atoms);
            builder.arg(&n_frames);
            builder.arg(&max_len);
            builder.arg(&mut sum_x);
            builder.arg(&mut sum_y);
            builder.arg(&mut sum_z);
            builder.arg(&mut mass_sum);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let cfg_out = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(total, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.group_com_finalize);
            builder.arg(&sum_x);
            builder.arg(&sum_y);
            builder.arg(&sum_z);
            builder.arg(&mass_sum);
            builder.arg(&n_groups);
            builder.arg(&n_frames);
            builder.arg(&scale);
            builder.arg(&mut out);
            builder.launch(cfg_out).map_err(map_driver_err)?;
        }

        let mut host = vec![Float4::default(); total];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn angle_from_coms(
        &self,
        coms: &[Float4],
        n_frames: usize,
        boxes: &[Float4],
        degrees: bool,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if boxes.len() != n_frames {
            return Err(TrajError::Mismatch("box length buffer mismatch".into()));
        }
        let stream = &self.inner.stream;
        let coms_dev = stream.clone_htod(coms).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;
        let n_frames_i32 = n_frames as i32;
        let degrees_flag = if degrees { 1i32 } else { 0i32 };
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.angle_from_com);
            builder.arg(&coms_dev);
            builder.arg(&n_frames_i32);
            builder.arg(&boxes_dev);
            builder.arg(&degrees_flag);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![0.0f32; n_frames];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn dihedral_from_coms(
        &self,
        coms: &[Float4],
        n_frames: usize,
        boxes: &[Float4],
        degrees: bool,
        range360: bool,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if boxes.len() != n_frames {
            return Err(TrajError::Mismatch("box length buffer mismatch".into()));
        }
        let stream = &self.inner.stream;
        let coms_dev = stream.clone_htod(coms).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;
        let n_frames_i32 = n_frames as i32;
        let degrees_flag = if degrees { 1i32 } else { 0i32 };
        let range_flag = if range360 { 1i32 } else { 0i32 };
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.dihedral_from_com);
            builder.arg(&coms_dev);
            builder.arg(&n_frames_i32);
            builder.arg(&boxes_dev);
            builder.arg(&degrees_flag);
            builder.arg(&range_flag);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![0.0f32; n_frames];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn distance_from_coms(
        &self,
        coms: &[Float4],
        n_frames: usize,
        boxes: &[Float4],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if boxes.len() != n_frames {
            return Err(TrajError::Mismatch("box length buffer mismatch".into()));
        }
        let stream = &self.inner.stream;
        let coms_dev = stream.clone_htod(coms).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;
        let n_frames_i32 = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.distance_from_com);
            builder.arg(&coms_dev);
            builder.arg(&n_frames_i32);
            builder.arg(&boxes_dev);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![0.0f32; n_frames];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn distance_from_coms_min(
        &self,
        coms: &[Float4],
        n_frames: usize,
        boxes: &[Float4],
    ) -> TrajResult<f32> {
        if n_frames == 0 {
            return Ok(f32::INFINITY);
        }
        if boxes.len() != n_frames {
            return Err(TrajError::Mismatch("box length buffer mismatch".into()));
        }
        let stream = &self.inner.stream;
        let coms_dev = stream.clone_htod(coms).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let init = [f32::INFINITY];
        let mut out_min = stream.clone_htod(&init).map_err(map_driver_err)?;
        let n_frames_i32 = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.distance_from_com_min);
            builder.arg(&coms_dev);
            builder.arg(&n_frames_i32);
            builder.arg(&boxes_dev);
            builder.arg(&mut out_min);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = [0.0f32];
        stream
            .memcpy_dtoh(&out_min, &mut host)
            .map_err(map_driver_err)?;
        Ok(host[0])
    }

    pub fn group_dipole(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        groups: &GpuGroups,
        charges: &GpuBufferF32,
        scale: f32,
    ) -> TrajResult<Vec<Float4>> {
        if n_frames == 0 || groups.n_groups == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let total = n_frames * groups.n_groups;
        let mut sum_x = stream.alloc_zeros::<f32>(total).map_err(map_driver_err)?;
        let mut sum_y = stream.alloc_zeros::<f32>(total).map_err(map_driver_err)?;
        let mut sum_z = stream.alloc_zeros::<f32>(total).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<Float4>(total)
            .map_err(map_driver_err)?;

        let n_groups = groups.n_groups as i32;
        let n_atoms = n_atoms as i32;
        let n_frames = n_frames as i32;
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
            let mut builder = stream.launch_builder(&self.inner.kernels.group_dipole_accum);
            builder.arg(&coords_dev);
            builder.arg(&groups.offsets);
            builder.arg(&groups.indices);
            builder.arg(&charges.inner);
            builder.arg(&n_groups);
            builder.arg(&n_atoms);
            builder.arg(&n_frames);
            builder.arg(&max_len);
            builder.arg(&mut sum_x);
            builder.arg(&mut sum_y);
            builder.arg(&mut sum_z);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let cfg_out = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(total, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.group_dipole_finalize);
            builder.arg(&sum_x);
            builder.arg(&sum_y);
            builder.arg(&sum_z);
            builder.arg(&n_groups);
            builder.arg(&n_frames);
            builder.arg(&scale);
            builder.arg(&mut out);
            builder.launch(cfg_out).map_err(map_driver_err)?;
        }

        let mut host = vec![Float4::default(); total];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn group_ke(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        groups: &GpuGroups,
        masses: &GpuBufferF32,
        vel_scale: f32,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || groups.n_groups == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let total = n_frames * groups.n_groups;
        let mut sum_ke = stream.alloc_zeros::<f32>(total).map_err(map_driver_err)?;

        let n_groups = groups.n_groups as i32;
        let n_atoms = n_atoms as i32;
        let n_frames = n_frames as i32;
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
            let mut builder = stream.launch_builder(&self.inner.kernels.group_ke_accum);
            builder.arg(&coords_dev);
            builder.arg(&groups.offsets);
            builder.arg(&groups.indices);
            builder.arg(&masses.inner);
            builder.arg(&n_groups);
            builder.arg(&n_atoms);
            builder.arg(&n_frames);
            builder.arg(&max_len);
            builder.arg(&vel_scale);
            builder.arg(&mut sum_ke);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; total];
        stream
            .memcpy_dtoh(&sum_ke, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn orientation_plane(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        anchors: &GpuAnchors,
        scale: f32,
    ) -> TrajResult<Vec<Float4>> {
        if n_frames == 0 || anchors.n_groups == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let total = n_frames * anchors.n_groups;
        let mut out = stream
            .alloc_zeros::<Float4>(total)
            .map_err(map_driver_err)?;
        let n_groups = anchors.n_groups as i32;
        let n_atoms = n_atoms as i32;
        let n_frames = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(anchors.n_groups, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.orientation_plane);
            builder.arg(&coords_dev);
            builder.arg(&anchors.anchors);
            builder.arg(&n_groups);
            builder.arg(&n_atoms);
            builder.arg(&n_frames);
            builder.arg(&scale);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![Float4::default(); total];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn orientation_vector(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        anchors: &GpuAnchors,
        scale: f32,
    ) -> TrajResult<Vec<Float4>> {
        if n_frames == 0 || anchors.n_groups == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let total = n_frames * anchors.n_groups;
        let mut out = stream
            .alloc_zeros::<Float4>(total)
            .map_err(map_driver_err)?;
        let n_groups = anchors.n_groups as i32;
        let n_atoms = n_atoms as i32;
        let n_frames = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(anchors.n_groups, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.orientation_vector);
            builder.arg(&coords_dev);
            builder.arg(&anchors.anchors);
            builder.arg(&n_groups);
            builder.arg(&n_atoms);
            builder.arg(&n_frames);
            builder.arg(&scale);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![Float4::default(); total];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn orientation_vector_pbc(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        anchors: &GpuAnchors,
        boxes: &[Float4],
        scale: f32,
    ) -> TrajResult<Vec<Float4>> {
        if n_frames == 0 || anchors.n_groups == 0 {
            return Ok(Vec::new());
        }
        if boxes.len() != n_frames {
            return Err(TrajError::Mismatch("box length buffer mismatch".into()));
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let total = n_frames * anchors.n_groups;
        let mut out = stream
            .alloc_zeros::<Float4>(total)
            .map_err(map_driver_err)?;
        let n_groups = anchors.n_groups as i32;
        let n_atoms = n_atoms as i32;
        let n_frames = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(anchors.n_groups, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.orientation_vector_pbc);
            builder.arg(&coords_dev);
            builder.arg(&anchors.anchors);
            builder.arg(&n_groups);
            builder.arg(&n_atoms);
            builder.arg(&n_frames);
            builder.arg(&boxes_dev);
            builder.arg(&scale);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![Float4::default(); total];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn water_count_frame(
        &self,
        coords: &GpuCoords,
        n_atoms: usize,
        frame: usize,
        selection: &GpuSelection,
        center: [f32; 3],
        box_unit: [f32; 3],
        region: [f32; 3],
        dims: [i32; 3],
        scale: f32,
        counts: &mut GpuCountsU32,
    ) -> TrajResult<()> {
        let stream = &self.inner.stream;
        let n_sel = selection.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let frame = frame as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(selection.n_sel.max(1), block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.water_count);
            builder.arg(&coords.inner);
            builder.arg(&selection.sel);
            builder.arg(&n_sel);
            builder.arg(&n_atoms);
            builder.arg(&frame);
            builder.arg(&center[0]);
            builder.arg(&center[1]);
            builder.arg(&center[2]);
            builder.arg(&box_unit[0]);
            builder.arg(&box_unit[1]);
            builder.arg(&box_unit[2]);
            builder.arg(&region[0]);
            builder.arg(&region[1]);
            builder.arg(&region[2]);
            builder.arg(&dims[0]);
            builder.arg(&dims[1]);
            builder.arg(&dims[2]);
            builder.arg(&scale);
            builder.arg(&mut counts.inner);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gist_counts_orient_frame_device(
        &self,
        coords: &GpuCoords,
        n_atoms: usize,
        frame: usize,
        oxygen_idx: &GpuBufferU32,
        h1_idx: &GpuBufferU32,
        h2_idx: &GpuBufferU32,
        orient_valid: &GpuBufferU32,
        n_waters: usize,
        center: [f32; 3],
        origin: [f32; 3],
        spacing: f32,
        dims: [usize; 3],
        orientation_bins: usize,
        length_scale: f32,
    ) -> TrajResult<(GpuBufferU32, GpuBufferU32)> {
        if n_waters == 0 {
            let empty = self.upload_u32(&[])?;
            let empty_b = self.upload_u32(&[])?;
            return Ok((empty, empty_b));
        }
        let n_atoms_i32 = i32::try_from(n_atoms)
            .map_err(|_| TrajError::Mismatch("n_atoms exceeds i32 range".into()))?;
        let frame_i32 = i32::try_from(frame)
            .map_err(|_| TrajError::Mismatch("frame index exceeds i32 range".into()))?;
        let n_waters_i32 = i32::try_from(n_waters)
            .map_err(|_| TrajError::Mismatch("n_waters exceeds i32 range".into()))?;
        let dim_x_i32 = i32::try_from(dims[0])
            .map_err(|_| TrajError::Mismatch("gist dim_x exceeds i32 range".into()))?;
        let dim_y_i32 = i32::try_from(dims[1])
            .map_err(|_| TrajError::Mismatch("gist dim_y exceeds i32 range".into()))?;
        let dim_z_i32 = i32::try_from(dims[2])
            .map_err(|_| TrajError::Mismatch("gist dim_z exceeds i32 range".into()))?;
        let bins_i32 = i32::try_from(orientation_bins)
            .map_err(|_| TrajError::Mismatch("orientation_bins exceeds i32 range".into()))?;
        let stream = &self.inner.stream;
        let out_cell = GpuBufferU32 {
            inner: stream
                .alloc_zeros::<u32>(n_waters)
                .map_err(map_driver_err)?,
        };
        let out_bin = GpuBufferU32 {
            inner: stream
                .alloc_zeros::<u32>(n_waters)
                .map_err(map_driver_err)?,
        };
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_waters.max(1), block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.gist_counts_orient);
            builder.arg(&coords.inner);
            builder.arg(&n_atoms_i32);
            builder.arg(&frame_i32);
            builder.arg(&oxygen_idx.inner);
            builder.arg(&h1_idx.inner);
            builder.arg(&h2_idx.inner);
            builder.arg(&orient_valid.inner);
            builder.arg(&n_waters_i32);
            builder.arg(&center[0]);
            builder.arg(&center[1]);
            builder.arg(&center[2]);
            builder.arg(&origin[0]);
            builder.arg(&origin[1]);
            builder.arg(&origin[2]);
            builder.arg(&spacing);
            builder.arg(&dim_x_i32);
            builder.arg(&dim_y_i32);
            builder.arg(&dim_z_i32);
            builder.arg(&bins_i32);
            builder.arg(&length_scale);
            builder.arg(&out_cell.inner);
            builder.arg(&out_bin.inner);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        Ok((out_cell, out_bin))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gist_counts_orient_frame(
        &self,
        coords: &GpuCoords,
        n_atoms: usize,
        frame: usize,
        oxygen_idx: &GpuBufferU32,
        h1_idx: &GpuBufferU32,
        h2_idx: &GpuBufferU32,
        orient_valid: &GpuBufferU32,
        n_waters: usize,
        center: [f32; 3],
        origin: [f32; 3],
        spacing: f32,
        dims: [usize; 3],
        orientation_bins: usize,
        length_scale: f32,
    ) -> TrajResult<(Vec<u32>, Vec<u32>)> {
        let (cells_dev, bins_dev) = self.gist_counts_orient_frame_device(
            coords,
            n_atoms,
            frame,
            oxygen_idx,
            h1_idx,
            h2_idx,
            orient_valid,
            n_waters,
            center,
            origin,
            spacing,
            dims,
            orientation_bins,
            length_scale,
        )?;
        let host_cell = self.download_u32(&cells_dev, n_waters)?;
        let host_bin = self.download_u32(&bins_dev, n_waters)?;
        Ok((host_cell, host_bin))
    }

    pub fn gist_accumulate_hist(
        &self,
        cell_idx: &GpuBufferU32,
        bin_idx: &GpuBufferU32,
        n_waters: usize,
        orientation_bins: usize,
        counts: &mut GpuBufferU32,
        orient_counts: &mut GpuBufferU32,
    ) -> TrajResult<()> {
        if n_waters == 0 {
            return Ok(());
        }
        let n_waters_i32 = i32::try_from(n_waters)
            .map_err(|_| TrajError::Mismatch("n_waters exceeds i32 range".into()))?;
        let bins_i32 = i32::try_from(orientation_bins)
            .map_err(|_| TrajError::Mismatch("orientation_bins exceeds i32 range".into()))?;
        let stream = &self.inner.stream;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_waters.max(1), block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.gist_accumulate_hist);
            builder.arg(&cell_idx.inner);
            builder.arg(&bin_idx.inner);
            builder.arg(&n_waters_i32);
            builder.arg(&bins_i32);
            builder.arg(&mut counts.inner);
            builder.arg(&mut orient_counts.inner);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gist_direct_energy_frame(
        &self,
        coords: &GpuCoords,
        n_atoms: usize,
        frame: usize,
        water_offsets: &GpuBufferU32,
        water_atoms: &GpuBufferU32,
        n_waters: usize,
        solute_atoms: &GpuBufferU32,
        n_solute: usize,
        charges: &GpuBufferF32,
        sigmas: &GpuBufferF32,
        epsilons: &GpuBufferF32,
        ex_i: &GpuBufferU32,
        ex_j: &GpuBufferU32,
        ex_qprod: &GpuBufferF32,
        ex_sigma: &GpuBufferF32,
        ex_epsilon: &GpuBufferF32,
        n_exceptions: usize,
        pbc_mode: i32,
        box_lengths: [f32; 3],
        cell: [f32; 9],
        inv: [f32; 9],
        cutoff: f32,
        length_scale: f32,
    ) -> TrajResult<(Vec<f32>, Vec<f32>)> {
        if n_waters == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        let n_atoms_i32 = i32::try_from(n_atoms)
            .map_err(|_| TrajError::Mismatch("n_atoms exceeds i32 range".into()))?;
        let frame_i32 = i32::try_from(frame)
            .map_err(|_| TrajError::Mismatch("frame index exceeds i32 range".into()))?;
        let n_waters_i32 = i32::try_from(n_waters)
            .map_err(|_| TrajError::Mismatch("n_waters exceeds i32 range".into()))?;
        let n_solute_i32 = i32::try_from(n_solute)
            .map_err(|_| TrajError::Mismatch("n_solute exceeds i32 range".into()))?;
        let n_ex_i32 = i32::try_from(n_exceptions)
            .map_err(|_| TrajError::Mismatch("n_exceptions exceeds i32 range".into()))?;
        let stream = &self.inner.stream;
        let mut out_sw = stream
            .alloc_zeros::<f32>(n_waters)
            .map_err(map_driver_err)?;
        let mut out_ww = stream
            .alloc_zeros::<f32>(n_waters)
            .map_err(map_driver_err)?;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_waters.max(1), block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.gist_direct_energy);
            builder.arg(&coords.inner);
            builder.arg(&n_atoms_i32);
            builder.arg(&frame_i32);
            builder.arg(&water_offsets.inner);
            builder.arg(&water_atoms.inner);
            builder.arg(&n_waters_i32);
            builder.arg(&solute_atoms.inner);
            builder.arg(&n_solute_i32);
            builder.arg(&charges.inner);
            builder.arg(&sigmas.inner);
            builder.arg(&epsilons.inner);
            builder.arg(&ex_i.inner);
            builder.arg(&ex_j.inner);
            builder.arg(&ex_qprod.inner);
            builder.arg(&ex_sigma.inner);
            builder.arg(&ex_epsilon.inner);
            builder.arg(&n_ex_i32);
            builder.arg(&pbc_mode);
            builder.arg(&box_lengths[0]);
            builder.arg(&box_lengths[1]);
            builder.arg(&box_lengths[2]);
            builder.arg(&cell[0]);
            builder.arg(&cell[1]);
            builder.arg(&cell[2]);
            builder.arg(&cell[3]);
            builder.arg(&cell[4]);
            builder.arg(&cell[5]);
            builder.arg(&cell[6]);
            builder.arg(&cell[7]);
            builder.arg(&cell[8]);
            builder.arg(&inv[0]);
            builder.arg(&inv[1]);
            builder.arg(&inv[2]);
            builder.arg(&inv[3]);
            builder.arg(&inv[4]);
            builder.arg(&inv[5]);
            builder.arg(&inv[6]);
            builder.arg(&inv[7]);
            builder.arg(&inv[8]);
            builder.arg(&cutoff);
            builder.arg(&length_scale);
            builder.arg(&mut out_sw);
            builder.arg(&mut out_ww);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host_sw = vec![0.0f32; n_waters];
        let mut host_ww = vec![0.0f32; n_waters];
        stream
            .memcpy_dtoh(&out_sw, &mut host_sw)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&out_ww, &mut host_ww)
            .map_err(map_driver_err)?;
        Ok((host_sw, host_ww))
    }

    pub fn hbond_counts(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        donors: &GpuSelection,
        acceptors: &GpuSelection,
        dist2: f32,
    ) -> TrajResult<Vec<u32>> {
        if n_frames == 0 || donors.n_sel == 0 || acceptors.n_sel == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut counts = stream
            .alloc_zeros::<u32>(n_frames)
            .map_err(map_driver_err)?;
        let n_donors = donors.n_sel as i32;
        let n_acceptors = acceptors.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let total_pairs = donors.n_sel * acceptors.n_sel;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(total_pairs.max(1), block), n_frames_i32 as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.hbond_count);
            builder.arg(&coords_dev);
            builder.arg(&donors.sel);
            builder.arg(&acceptors.sel);
            builder.arg(&n_donors);
            builder.arg(&n_acceptors);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&dist2);
            builder.arg(&mut counts);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![0u32; n_frames];
        stream
            .memcpy_dtoh(&counts, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }
}
