use super::*;

impl GpuContext {
    pub fn max_dist_points(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
        points: &[Float4],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if points.len() != n_frames {
            return Err(TrajError::Mismatch("points length buffer mismatch".into()));
        }
        if sel.n_sel == 0 {
            return Ok(vec![0.0f32; n_frames]);
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let points_dev = stream.clone_htod(points).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames)
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
            let mut builder = stream.launch_builder(&self.inner.kernels.max_dist_points);
            builder.arg(&coords_dev);
            builder.arg(&sel.sel);
            builder.arg(&n_sel);
            builder.arg(&points_dev);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn multipucker_histogram(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
        centers: &[Float4],
        bins: usize,
        range_max: f32,
        normalize: bool,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || bins == 0 {
            return Ok(Vec::new());
        }
        if centers.len() != n_frames {
            return Err(TrajError::Mismatch("centers length buffer mismatch".into()));
        }
        if sel.n_sel == 0 {
            return Ok(vec![0.0f32; n_frames * bins]);
        }
        if range_max <= 0.0 {
            return Err(TrajError::Parse(
                "multipucker histogram range_max must be > 0".into(),
            ));
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let centers_dev = stream.clone_htod(centers).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * bins)
            .map_err(map_driver_err)?;

        let n_sel = sel.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let bins_i32 = bins as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(sel.n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.multipucker_histogram);
            builder.arg(&coords_dev);
            builder.arg(&sel.sel);
            builder.arg(&n_sel);
            builder.arg(&centers_dev);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&bins_i32);
            builder.arg(&range_max);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        if normalize {
            let cfg_norm = LaunchConfig {
                block_dim: (1, 1, 1),
                grid_dim: (n_frames as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                let mut builder =
                    stream.launch_builder(&self.inner.kernels.multipucker_normalize_rows);
                builder.arg(&mut out);
                builder.arg(&n_frames_i32);
                builder.arg(&bins_i32);
                builder.launch(cfg_norm).map_err(map_driver_err)?;
            }
        }

        let mut host = vec![0.0f32; n_frames * bins];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn multipucker_distances(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
        centers: &[Float4],
    ) -> TrajResult<(Vec<f32>, Vec<f32>)> {
        if n_frames == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if centers.len() != n_frames {
            return Err(TrajError::Mismatch("centers length buffer mismatch".into()));
        }
        if sel.n_sel == 0 {
            return Ok((Vec::new(), vec![0.0f32; n_frames]));
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let centers_dev = stream.clone_htod(centers).map_err(map_driver_err)?;
        let mut dist = stream
            .alloc_zeros::<f32>(n_frames * sel.n_sel)
            .map_err(map_driver_err)?;
        let mut max_per_frame = stream
            .alloc_zeros::<f32>(n_frames)
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
            let mut builder = stream.launch_builder(&self.inner.kernels.multipucker_distances);
            builder.arg(&coords_dev);
            builder.arg(&sel.sel);
            builder.arg(&n_sel);
            builder.arg(&centers_dev);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&mut dist);
            builder.arg(&mut max_per_frame);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut dist_host = vec![0.0f32; n_frames * sel.n_sel];
        let mut max_host = vec![0.0f32; n_frames];
        stream
            .memcpy_dtoh(&dist, &mut dist_host)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&max_per_frame, &mut max_host)
            .map_err(map_driver_err)?;
        Ok((dist_host, max_host))
    }

    pub fn multipucker_histogram_from_distances(
        &self,
        distances: &[f32],
        n_sel: usize,
        n_frames: usize,
        bins: usize,
        range_max: f32,
        normalize: bool,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || bins == 0 {
            return Ok(Vec::new());
        }
        if n_sel == 0 {
            return Ok(vec![0.0f32; n_frames * bins]);
        }
        if distances.len() != n_frames * n_sel {
            return Err(TrajError::Mismatch(
                "distances length buffer mismatch".into(),
            ));
        }
        if range_max <= 0.0 {
            return Err(TrajError::Parse(
                "multipucker histogram range_max must be > 0".into(),
            ));
        }
        let stream = &self.inner.stream;
        let dist_dev = stream.clone_htod(distances).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * bins)
            .map_err(map_driver_err)?;
        let n_sel_i32 = n_sel as i32;
        let n_frames_i32 = n_frames as i32;
        let bins_i32 = bins as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder =
                stream.launch_builder(&self.inner.kernels.multipucker_histogram_from_distances);
            builder.arg(&dist_dev);
            builder.arg(&n_sel_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&bins_i32);
            builder.arg(&range_max);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        if normalize {
            let cfg_norm = LaunchConfig {
                block_dim: (1, 1, 1),
                grid_dim: (n_frames as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe {
                let mut builder =
                    stream.launch_builder(&self.inner.kernels.multipucker_normalize_rows);
                builder.arg(&mut out);
                builder.arg(&n_frames_i32);
                builder.arg(&bins_i32);
                builder.launch(cfg_norm).map_err(map_driver_err)?;
            }
        }
        let mut host = vec![0.0f32; n_frames * bins];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn atom_map_pairs(
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
        if sel_a.n_sel == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * sel_a.n_sel)
            .map_err(map_driver_err)?;

        let n_sel_a = sel_a.n_sel as i32;
        let n_sel_b = sel_b.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(sel_a.n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.atom_map_pairs);
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

        let mut host = vec![0.0f32; n_frames * sel_a.n_sel];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn closest_topk(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel_a: &GpuSelection,
        sel_b: &GpuSelection,
        boxes: &[Float4],
        k: usize,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || k == 0 {
            return Ok(Vec::new());
        }
        if boxes.len() != n_frames {
            return Err(TrajError::Mismatch("box length buffer mismatch".into()));
        }
        if sel_a.n_sel == 0 || sel_b.n_sel == 0 {
            return Ok(vec![-1.0f32; n_frames * k]);
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let mut min_dists = stream
            .alloc_zeros::<f32>(n_frames * sel_b.n_sel)
            .map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * k)
            .map_err(map_driver_err)?;

        let n_sel_a = sel_a.n_sel as i32;
        let n_sel_b = sel_b.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let k_i32 = k as i32;

        let block = 256u32;
        let cfg_min = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(sel_b.n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.closest_min_dist);
            builder.arg(&coords_dev);
            builder.arg(&sel_a.sel);
            builder.arg(&sel_b.sel);
            builder.arg(&n_sel_a);
            builder.arg(&n_sel_b);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&boxes_dev);
            builder.arg(&mut min_dists);
            builder.launch(cfg_min).map_err(map_driver_err)?;
        }

        let cfg_topk = LaunchConfig {
            block_dim: (1, 1, 1),
            grid_dim: (n_frames as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.closest_topk);
            builder.arg(&min_dists);
            builder.arg(&sel_b.sel);
            builder.arg(&n_sel_b);
            builder.arg(&n_frames_i32);
            builder.arg(&k_i32);
            builder.arg(&mut out);
            builder.launch(cfg_topk).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames * k];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn rotate_dihedral(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        mask: &GpuBufferU32,
        pivots: &[Float4],
        axes: &[Float4],
        angles: &[f32],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || n_atoms == 0 {
            return Ok(Vec::new());
        }
        if pivots.len() != n_frames || axes.len() != n_frames || angles.len() != n_frames {
            return Err(TrajError::Mismatch(
                "rotate_dihedral input length mismatch".into(),
            ));
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let pivots_dev = stream.clone_htod(pivots).map_err(map_driver_err)?;
        let axes_dev = stream.clone_htod(axes).map_err(map_driver_err)?;
        let angles_dev = stream.clone_htod(angles).map_err(map_driver_err)?;
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
            let mut builder = stream.launch_builder(&self.inner.kernels.rotate_dihedral_apply);
            builder.arg(&coords_dev);
            builder.arg(&mask.inner);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&pivots_dev);
            builder.arg(&axes_dev);
            builder.arg(&angles_dev);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames * n_atoms * 3];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn image_coords(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        cell: &[Float4],
        inv: &[Float4],
        mask: &GpuBufferU32,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || n_atoms == 0 {
            return Ok(Vec::new());
        }
        if cell.len() != n_frames * 3 || inv.len() != n_frames * 3 {
            return Err(TrajError::Mismatch("cell matrix buffer mismatch".into()));
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let cell_dev = stream.clone_htod(cell).map_err(map_driver_err)?;
        let inv_dev = stream.clone_htod(inv).map_err(map_driver_err)?;
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
            let mut builder = stream.launch_builder(&self.inner.kernels.image_coords);
            builder.arg(&coords_dev);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&cell_dev);
            builder.arg(&inv_dev);
            builder.arg(&mask.inner);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames * n_atoms * 3];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn chirality_volume(
        &self,
        coms: &[Float4],
        n_frames: usize,
        n_centers: usize,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 || n_centers == 0 {
            return Ok(Vec::new());
        }
        let expected = n_frames * n_centers * 4;
        if coms.len() != expected {
            return Err(TrajError::Mismatch("com buffer length mismatch".into()));
        }
        let stream = &self.inner.stream;
        let coms_dev = stream.clone_htod(coms).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * n_centers)
            .map_err(map_driver_err)?;

        let n_frames_i32 = n_frames as i32;
        let n_centers_i32 = n_centers as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_centers, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.chirality_volume);
            builder.arg(&coms_dev);
            builder.arg(&n_frames_i32);
            builder.arg(&n_centers_i32);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames * n_centers];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn torsion_diffusion_counts(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        torsions: &GpuBufferU32,
        n_torsions: usize,
    ) -> TrajResult<Vec<u32>> {
        if n_frames == 0 || n_torsions == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut counts = stream
            .alloc_zeros::<u32>(n_frames * 4)
            .map_err(map_driver_err)?;

        let n_atoms_i32 = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let n_torsions_i32 = n_torsions as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_torsions, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.torsion_diffusion_counts);
            builder.arg(&coords_dev);
            builder.arg(&torsions.inner);
            builder.arg(&n_torsions_i32);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&mut counts);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0u32; n_frames * 4];
        stream
            .memcpy_dtoh(&counts, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn bbox_area(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        selection: &GpuSelection,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if selection.n_sel == 0 {
            return Ok(vec![0.0f32; n_frames]);
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let init_min = vec![f32::INFINITY; n_frames];
        let init_max = vec![f32::NEG_INFINITY; n_frames];
        let min_x = stream.clone_htod(&init_min).map_err(map_driver_err)?;
        let min_y = stream.clone_htod(&init_min).map_err(map_driver_err)?;
        let min_z = stream.clone_htod(&init_min).map_err(map_driver_err)?;
        let max_x = stream.clone_htod(&init_max).map_err(map_driver_err)?;
        let max_y = stream.clone_htod(&init_max).map_err(map_driver_err)?;
        let max_z = stream.clone_htod(&init_max).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames)
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
            let mut builder = stream.launch_builder(&self.inner.kernels.bbox_minmax);
            builder.arg(&coords_dev);
            builder.arg(&selection.sel);
            builder.arg(&n_sel);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&min_x);
            builder.arg(&min_y);
            builder.arg(&min_z);
            builder.arg(&max_x);
            builder.arg(&max_y);
            builder.arg(&max_z);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let cfg_out = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.bbox_area);
            builder.arg(&min_x);
            builder.arg(&min_y);
            builder.arg(&min_z);
            builder.arg(&max_x);
            builder.arg(&max_y);
            builder.arg(&max_z);
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

    pub fn sasa_approx(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        selection: &GpuSelection,
        radii: &[f32],
        sphere_points: &[Float4],
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if selection.n_sel == 0 {
            return Ok(vec![0.0f32; n_frames]);
        }
        if radii.len() != selection.n_sel {
            return Err(TrajError::Mismatch("radii length buffer mismatch".into()));
        }
        if sphere_points.is_empty() {
            return Err(TrajError::Mismatch(
                "sphere_points must be non-empty".into(),
            ));
        }
        let total = n_frames
            .saturating_mul(selection.n_sel)
            .saturating_mul(sphere_points.len());
        if total == 0 {
            return Ok(vec![0.0f32; n_frames]);
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let radii_dev = stream.clone_htod(radii).map_err(map_driver_err)?;
        let sphere_dev = stream.clone_htod(sphere_points).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;

        let n_sel = selection.n_sel as i32;
        let n_points = sphere_points.len() as i32;
        let n_atoms_i32 = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(total, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.sasa_approx);
            builder.arg(&coords_dev);
            builder.arg(&selection.sel);
            builder.arg(&radii_dev);
            builder.arg(&sphere_dev);
            builder.arg(&n_sel);
            builder.arg(&n_points);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![0.0f32; n_frames];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn volume_orthorhombic(&self, boxes: &[Float4], n_frames: usize) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if boxes.len() != n_frames {
            return Err(TrajError::Mismatch("box length buffer mismatch".into()));
        }
        let stream = &self.inner.stream;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;

        let n_frames_i32 = n_frames as i32;
        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames as usize, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.volume_orthorhombic);
            builder.arg(&boxes_dev);
            builder.arg(&n_frames_i32);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        let mut host = vec![0.0f32; n_frames];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn volume_cell(&self, cell: &[Float4], n_frames: usize) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        if cell.len() != n_frames * 3 {
            return Err(TrajError::Mismatch("cell matrix buffer mismatch".into()));
        }
        let stream = &self.inner.stream;
        let cell_dev = stream.clone_htod(cell).map_err(map_driver_err)?;
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
            let mut builder = stream.launch_builder(&self.inner.kernels.volume_cell);
            builder.arg(&cell_dev);
            builder.arg(&n_frames_i32);
            builder.arg(&mut out);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }
}
