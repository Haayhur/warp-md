use super::*;

impl GpuContext {
    pub fn distance_to_reference(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
        reference: &GpuReference,
        boxes: &[Float4],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if boxes.len() != n_frames {
            return Err(TrajError::Mismatch("box length buffer mismatch".into()));
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * sel.n_sel)
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
            let mut builder = stream.launch_builder(&self.inner.kernels.distance_to_reference);
            builder.arg(&coords_dev);
            builder.arg(&reference.coords);
            builder.arg(&sel.sel);
            builder.arg(&n_sel);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&boxes_dev);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames * sel.n_sel];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn pairwise_distance(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel_a: &GpuSelection,
        sel_b: &GpuSelection,
        boxes: &[Float4],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if boxes.len() != n_frames {
            return Err(TrajError::Mismatch("box length buffer mismatch".into()));
        }
        let n_pairs = sel_a.n_sel * sel_b.n_sel;
        if n_pairs == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * n_pairs)
            .map_err(map_driver_err)?;

        let n_sel_a = sel_a.n_sel as i32;
        let n_sel_b = sel_b.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_pairs, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.pairwise_distance);
            builder.arg(&coords_dev);
            builder.arg(&sel_a.sel);
            builder.arg(&sel_b.sel);
            builder.arg(&n_sel_a);
            builder.arg(&n_sel_b);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&boxes_dev);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames * n_pairs];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn mindist_pairs(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel_a: &GpuSelection,
        sel_b: &GpuSelection,
        boxes: &[Float4],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if boxes.len() != n_frames {
            return Err(TrajError::Mismatch("box length buffer mismatch".into()));
        }
        if sel_a.n_sel == 0 || sel_b.n_sel == 0 {
            return Ok(vec![0.0f32; n_frames]);
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;

        let n_sel_a = sel_a.n_sel as i32;
        let n_sel_b = sel_b.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;

        let block = 256u32;
        let shared_mem = (block as usize) * std::mem::size_of::<f32>();
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (n_frames as u32, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.mindist_pairs);
            builder.arg(&coords_dev);
            builder.arg(&sel_a.sel);
            builder.arg(&sel_b.sel);
            builder.arg(&n_sel_a);
            builder.arg(&n_sel_b);
            builder.arg(&n_atoms);
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

    pub fn closest_atom(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
        point: [f32; 3],
        boxes: &[Float4],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if boxes.len() != n_frames {
            return Err(TrajError::Mismatch("box length buffer mismatch".into()));
        }
        if sel.n_sel == 0 {
            return Ok(vec![-1.0f32; n_frames]);
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;

        let n_sel = sel.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let point = Float4 {
            x: point[0],
            y: point[1],
            z: point[2],
            w: 0.0,
        };

        let block = 256u32;
        let shared =
            ((block as usize) * (std::mem::size_of::<f32>() + std::mem::size_of::<u32>())) as u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (n_frames as u32, 1, 1),
            shared_mem_bytes: shared,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.closest_atom_point);
            builder.arg(&coords_dev);
            builder.arg(&sel.sel);
            builder.arg(&n_sel);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&point);
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

    pub fn search_neighbors(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel_a: &GpuSelection,
        sel_b: &GpuSelection,
        boxes: &[Float4],
        cutoff: f32,
    ) -> TrajResult<Vec<u32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if boxes.len() != n_frames {
            return Err(TrajError::Mismatch("box length buffer mismatch".into()));
        }
        if sel_a.n_sel == 0 || sel_b.n_sel == 0 {
            return Ok(vec![0u32; n_frames]);
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let mut counts = stream
            .alloc_zeros::<u32>(n_frames)
            .map_err(map_driver_err)?;

        let n_sel_a = sel_a.n_sel as i32;
        let n_sel_b = sel_b.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(sel_b.n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.search_neighbors_count);
            builder.arg(&coords_dev);
            builder.arg(&sel_a.sel);
            builder.arg(&sel_b.sel);
            builder.arg(&n_sel_a);
            builder.arg(&n_sel_b);
            builder.arg(&n_atoms);
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

    pub fn hausdorff_pairs(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel_a: &GpuSelection,
        sel_b: &GpuSelection,
        boxes: &[Float4],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if boxes.len() != n_frames {
            return Err(TrajError::Mismatch("box length buffer mismatch".into()));
        }
        if sel_a.n_sel == 0 || sel_b.n_sel == 0 {
            return Ok(vec![0.0f32; n_frames]);
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let mut min_a = stream
            .alloc_zeros::<f32>(n_frames * sel_a.n_sel)
            .map_err(map_driver_err)?;
        let mut min_b = stream
            .alloc_zeros::<f32>(n_frames * sel_b.n_sel)
            .map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;

        let n_sel_a = sel_a.n_sel as i32;
        let n_sel_b = sel_b.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;

        let block = 256u32;
        let cfg_a = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(sel_a.n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.min_dist_a);
            builder.arg(&coords_dev);
            builder.arg(&sel_a.sel);
            builder.arg(&sel_b.sel);
            builder.arg(&n_sel_a);
            builder.arg(&n_sel_b);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&boxes_dev);
            builder.arg(&mut min_a);
            builder.launch(cfg_a).map_err(map_driver_err)?;
        }
        let cfg_b = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(sel_b.n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.min_dist_b);
            builder.arg(&coords_dev);
            builder.arg(&sel_a.sel);
            builder.arg(&sel_b.sel);
            builder.arg(&n_sel_a);
            builder.arg(&n_sel_b);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&boxes_dev);
            builder.arg(&mut min_b);
            builder.launch(cfg_b).map_err(map_driver_err)?;
        }

        let cfg_out = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (n_frames as u32, 1, 1),
            shared_mem_bytes: ((block as usize) * std::mem::size_of::<f32>()) as u32,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.hausdorff_reduce);
            builder.arg(&min_a);
            builder.arg(&min_b);
            builder.arg(&n_sel_a);
            builder.arg(&n_sel_b);
            builder.arg(&n_frames_i32);
            builder.arg(&mut out);
            builder.launch(cfg_out).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn hausdorff_pairs_triclinic(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel_a: &GpuSelection,
        sel_b: &GpuSelection,
        cell: &[Float4],
        inv: &[Float4],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if cell.len() != n_frames * 3 || inv.len() != n_frames * 3 {
            return Err(TrajError::Mismatch("cell matrix buffer mismatch".into()));
        }
        if sel_a.n_sel == 0 || sel_b.n_sel == 0 {
            return Ok(vec![0.0f32; n_frames]);
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let cell_dev = stream.clone_htod(cell).map_err(map_driver_err)?;
        let inv_dev = stream.clone_htod(inv).map_err(map_driver_err)?;
        let mut min_a = stream
            .alloc_zeros::<f32>(n_frames * sel_a.n_sel)
            .map_err(map_driver_err)?;
        let mut min_b = stream
            .alloc_zeros::<f32>(n_frames * sel_b.n_sel)
            .map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;

        let n_sel_a = sel_a.n_sel as i32;
        let n_sel_b = sel_b.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;

        let block = 256u32;
        let cfg_a = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(sel_a.n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.min_dist_a_triclinic);
            builder.arg(&coords_dev);
            builder.arg(&sel_a.sel);
            builder.arg(&sel_b.sel);
            builder.arg(&n_sel_a);
            builder.arg(&n_sel_b);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&cell_dev);
            builder.arg(&inv_dev);
            builder.arg(&mut min_a);
            builder.launch(cfg_a).map_err(map_driver_err)?;
        }
        let cfg_b = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(sel_b.n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.min_dist_b_triclinic);
            builder.arg(&coords_dev);
            builder.arg(&sel_a.sel);
            builder.arg(&sel_b.sel);
            builder.arg(&n_sel_a);
            builder.arg(&n_sel_b);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&cell_dev);
            builder.arg(&inv_dev);
            builder.arg(&mut min_b);
            builder.launch(cfg_b).map_err(map_driver_err)?;
        }

        let cfg_out = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (n_frames as u32, 1, 1),
            shared_mem_bytes: ((block as usize) * std::mem::size_of::<f32>()) as u32,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.hausdorff_reduce);
            builder.arg(&min_a);
            builder.arg(&min_b);
            builder.arg(&n_sel_a);
            builder.arg(&n_sel_b);
            builder.arg(&n_frames_i32);
            builder.arg(&mut out);
            builder.launch(cfg_out).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn min_distance_points(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
        points: &[Float4],
        cell: &[Float4],
        inv: &[Float4],
        image: bool,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if points.len() % n_frames != 0 {
            return Err(TrajError::Mismatch("points length buffer mismatch".into()));
        }
        if cell.len() != n_frames * 3 || inv.len() != n_frames * 3 {
            return Err(TrajError::Mismatch("cell matrix buffer mismatch".into()));
        }
        if sel.n_sel == 0 {
            return Ok(vec![0.0f32; points.len()]);
        }
        let n_points = points.len() / n_frames;
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let points_dev = stream.clone_htod(points).map_err(map_driver_err)?;
        let cell_dev = stream.clone_htod(cell).map_err(map_driver_err)?;
        let inv_dev = stream.clone_htod(inv).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(points.len())
            .map_err(map_driver_err)?;

        let n_sel = sel.n_sel as i32;
        let n_points_i32 = n_points as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let image_flag = if image { 1i32 } else { 0i32 };

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_points, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.min_dist_points);
            builder.arg(&coords_dev);
            builder.arg(&sel.sel);
            builder.arg(&n_sel);
            builder.arg(&points_dev);
            builder.arg(&n_points_i32);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&cell_dev);
            builder.arg(&inv_dev);
            builder.arg(&image_flag);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; points.len()];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }
}
