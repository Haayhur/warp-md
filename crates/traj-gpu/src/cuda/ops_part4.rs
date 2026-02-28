use super::*;

impl GpuContext {
    pub fn shift_coords(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        shifts: &[Float4],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || n_atoms == 0 {
            return Ok(Vec::new());
        }
        if shifts.len() != n_frames {
            return Err(TrajError::Mismatch("shift buffer length mismatch".into()));
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let shifts_dev = stream.clone_htod(shifts).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * n_atoms * 3)
            .map_err(map_driver_err)?;

        let n_atoms_i32 = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_atoms, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.shift_coords);
            builder.arg(&coords_dev);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&shifts_dev);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![0.0f32; n_frames * n_atoms * 3];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn translate_coords(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        delta: [f32; 3],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || n_atoms == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * n_atoms * 3)
            .map_err(map_driver_err)?;

        let n_atoms_i32 = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_atoms, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.translate_coords);
            builder.arg(&coords_dev);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&delta[0]);
            builder.arg(&delta[1]);
            builder.arg(&delta[2]);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![0.0f32; n_frames * n_atoms * 3];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn scale_coords(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        scale: f32,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || n_atoms == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * n_atoms * 3)
            .map_err(map_driver_err)?;

        let n_atoms_i32 = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_atoms, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.scale_coords);
            builder.arg(&coords_dev);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&scale);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![0.0f32; n_frames * n_atoms * 3];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn transform_coords(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        rot: &[f32; 9],
        trans: &[f32; 3],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || n_atoms == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let rot_dev = stream.clone_htod(rot).map_err(map_driver_err)?;
        let trans_dev = stream.clone_htod(trans).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * n_atoms * 3)
            .map_err(map_driver_err)?;

        let n_atoms_i32 = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_atoms, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.transform_coords);
            builder.arg(&coords_dev);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&rot_dev);
            builder.arg(&trans_dev);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![0.0f32; n_frames * n_atoms * 3];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn randomize_ions(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        mask: &GpuBufferU32,
        rand_vals: &[f32],
        cell: &[Float4],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || n_atoms == 0 {
            return Ok(Vec::new());
        }
        if cell.len() != n_frames * 3 {
            return Err(TrajError::Mismatch("cell vector buffer mismatch".into()));
        }
        if rand_vals.len() != n_frames * n_atoms * 3 {
            return Err(TrajError::Mismatch("random buffer length mismatch".into()));
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let cell_dev = stream.clone_htod(cell).map_err(map_driver_err)?;
        let rand_dev = stream.clone_htod(rand_vals).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * n_atoms * 3)
            .map_err(map_driver_err)?;

        let n_atoms_i32 = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_atoms, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.randomize_ions_apply);
            builder.arg(&coords_dev);
            builder.arg(&mask.inner);
            builder.arg(&rand_dev);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&cell_dev);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames * n_atoms * 3];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn native_contacts_counts(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        pairs: &GpuPairs,
        boxes: &[Float4],
        cutoff: f32,
    ) -> TrajResult<Vec<u32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if boxes.len() != n_frames {
            return Err(TrajError::Mismatch("box length buffer mismatch".into()));
        }
        if pairs.n_pairs == 0 {
            return Ok(vec![0u32; n_frames]);
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let mut counts = stream
            .alloc_zeros::<u32>(n_frames)
            .map_err(map_driver_err)?;

        let n_pairs = pairs.n_pairs as i32;
        let n_atoms_i32 = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(pairs.n_pairs, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.native_contacts_count);
            builder.arg(&coords_dev);
            builder.arg(&pairs.pairs);
            builder.arg(&n_pairs);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&boxes_dev);
            builder.arg(&cutoff);
            builder.arg(&mut counts);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0u32; n_frames];
        stream
            .memcpy_dtoh(&counts, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn gather_selection(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        selection: &GpuSelection,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || selection.n_sel == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * selection.n_sel * 3)
            .map_err(map_driver_err)?;

        let n_sel = selection.n_sel as i32;
        let n_atoms_i32 = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(selection.n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.gather_selection);
            builder.arg(&coords_dev);
            builder.arg(&selection.sel);
            builder.arg(&n_sel);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames * selection.n_sel * 3];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn replicate_cell(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        selection: &GpuSelection,
        cell: &[Float4],
        repeats: [usize; 3],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || selection.n_sel == 0 {
            return Ok(Vec::new());
        }
        if cell.len() != n_frames * 3 {
            return Err(TrajError::Mismatch("cell vector buffer mismatch".into()));
        }
        let reps = repeats[0] * repeats[1] * repeats[2];
        if reps == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let cell_dev = stream.clone_htod(cell).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * selection.n_sel * reps * 3)
            .map_err(map_driver_err)?;

        let n_sel = selection.n_sel as i32;
        let n_atoms_i32 = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let rx = repeats[0] as i32;
        let ry = repeats[1] as i32;
        let rz = repeats[2] as i32;
        let total = selection.n_sel * reps;

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(total, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.replicate_cell);
            builder.arg(&coords_dev);
            builder.arg(&selection.sel);
            builder.arg(&n_sel);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&cell_dev);
            builder.arg(&rx);
            builder.arg(&ry);
            builder.arg(&rz);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames * selection.n_sel * reps * 3];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn rmsf_accum(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
    ) -> TrajResult<RmsfAccum> {
        if n_frames == 0 || sel.n_sel == 0 {
            return Ok(RmsfAccum {
                sum_x: Vec::new(),
                sum_y: Vec::new(),
                sum_z: Vec::new(),
                sum_sq: Vec::new(),
            });
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut sum_x = stream
            .alloc_zeros::<f32>(sel.n_sel)
            .map_err(map_driver_err)?;
        let mut sum_y = stream
            .alloc_zeros::<f32>(sel.n_sel)
            .map_err(map_driver_err)?;
        let mut sum_z = stream
            .alloc_zeros::<f32>(sel.n_sel)
            .map_err(map_driver_err)?;
        let mut sum_sq = stream
            .alloc_zeros::<f32>(sel.n_sel)
            .map_err(map_driver_err)?;

        let n_sel = sel.n_sel as i32;
        let n_atoms_i32 = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(sel.n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.rmsf_accum);
            builder.arg(&coords_dev);
            builder.arg(&sel.sel);
            builder.arg(&n_sel);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&mut sum_x);
            builder.arg(&mut sum_y);
            builder.arg(&mut sum_z);
            builder.arg(&mut sum_sq);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host_x = vec![0.0f32; sel.n_sel];
        let mut host_y = vec![0.0f32; sel.n_sel];
        let mut host_z = vec![0.0f32; sel.n_sel];
        let mut host_sq = vec![0.0f32; sel.n_sel];
        stream
            .memcpy_dtoh(&sum_x, &mut host_x)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&sum_y, &mut host_y)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&sum_z, &mut host_z)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&sum_sq, &mut host_sq)
            .map_err(map_driver_err)?;

        Ok(RmsfAccum {
            sum_x: host_x,
            sum_y: host_y,
            sum_z: host_z,
            sum_sq: host_sq,
        })
    }

    pub fn mean_structure_accum(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
    ) -> TrajResult<MeanStructureAccum> {
        if n_frames == 0 || sel.n_sel == 0 {
            return Ok(MeanStructureAccum {
                sum_x: Vec::new(),
                sum_y: Vec::new(),
                sum_z: Vec::new(),
            });
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut sum_x = stream
            .alloc_zeros::<f32>(sel.n_sel)
            .map_err(map_driver_err)?;
        let mut sum_y = stream
            .alloc_zeros::<f32>(sel.n_sel)
            .map_err(map_driver_err)?;
        let mut sum_z = stream
            .alloc_zeros::<f32>(sel.n_sel)
            .map_err(map_driver_err)?;

        let n_sel = sel.n_sel as i32;
        let n_atoms_i32 = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(sel.n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.mean_structure_accum);
            builder.arg(&coords_dev);
            builder.arg(&sel.sel);
            builder.arg(&n_sel);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&mut sum_x);
            builder.arg(&mut sum_y);
            builder.arg(&mut sum_z);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host_x = vec![0.0f32; sel.n_sel];
        let mut host_y = vec![0.0f32; sel.n_sel];
        let mut host_z = vec![0.0f32; sel.n_sel];
        stream
            .memcpy_dtoh(&sum_x, &mut host_x)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&sum_y, &mut host_y)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&sum_z, &mut host_z)
            .map_err(map_driver_err)?;

        Ok(MeanStructureAccum {
            sum_x: host_x,
            sum_y: host_y,
            sum_z: host_z,
        })
    }
}
