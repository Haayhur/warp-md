use super::*;

impl GpuContext {
    pub fn new(device: usize) -> TrajResult<Self> {
        let ctx = CudaContext::new(device).map_err(map_driver_err)?;
        let stream = ctx.default_stream();
        let ptx = compile_ptx(KERNELS_SRC).map_err(map_compile_err)?;
        let module = ctx.load_module(ptx).map_err(map_driver_err)?;
        let kernels = Kernels::load(&module)?;
        Ok(Self {
            inner: Arc::new(GpuContextInner {
                stream,
                module,
                kernels,
            }),
        })
    }

    pub fn selection(&self, indices: &[u32], masses: Option<&[f32]>) -> TrajResult<GpuSelection> {
        let sel = self
            .inner
            .stream
            .clone_htod(indices)
            .map_err(map_driver_err)?;
        let masses_host: Vec<f32> = match masses {
            Some(values) => values.to_vec(),
            None => vec![1.0; indices.len()],
        };
        let masses_dev = self
            .inner
            .stream
            .clone_htod(&masses_host)
            .map_err(map_driver_err)?;
        Ok(GpuSelection {
            sel,
            masses: masses_dev,
            n_sel: indices.len(),
        })
    }

    pub fn upload_f32(&self, data: &[f32]) -> TrajResult<GpuBufferF32> {
        let buf = self.inner.stream.clone_htod(data).map_err(map_driver_err)?;
        Ok(GpuBufferF32 { inner: buf })
    }

    pub fn upload_u32(&self, data: &[u32]) -> TrajResult<GpuBufferU32> {
        let buf = self.inner.stream.clone_htod(data).map_err(map_driver_err)?;
        Ok(GpuBufferU32 { inner: buf })
    }

    pub fn download_u32(&self, data: &GpuBufferU32, n: usize) -> TrajResult<Vec<u32>> {
        let mut out = vec![0u32; n];
        self.inner
            .stream
            .memcpy_dtoh(&data.inner, &mut out)
            .map_err(map_driver_err)?;
        Ok(out)
    }

    pub fn pairs(&self, pairs: &[u32]) -> TrajResult<GpuPairs> {
        if pairs.len() % 2 != 0 {
            return Err(TrajError::Mismatch("pair list length must be even".into()));
        }
        let n_pairs = pairs.len() / 2;
        let buf = self
            .inner
            .stream
            .clone_htod(pairs)
            .map_err(map_driver_err)?;
        Ok(GpuPairs {
            pairs: buf,
            n_pairs,
        })
    }

    pub fn upload_coords(&self, coords: &[Float4]) -> TrajResult<GpuCoords> {
        let buf = self
            .inner
            .stream
            .clone_htod(coords)
            .map_err(map_driver_err)?;
        Ok(GpuCoords { inner: buf })
    }

    pub fn upload_coords_f32x4(&self, coords: &[[f32; 4]]) -> TrajResult<GpuCoords> {
        self.upload_coords(coords_as_float4(coords))
    }

    pub fn groups(
        &self,
        offsets: &[u32],
        indices: &[u32],
        max_len: usize,
    ) -> TrajResult<GpuGroups> {
        if offsets.len() < 2 {
            return Err(TrajError::Mismatch(
                "group offsets must include at least one group".into(),
            ));
        }
        let n_groups = offsets.len() - 1;
        let stream = &self.inner.stream;
        let offsets_dev = stream.clone_htod(offsets).map_err(map_driver_err)?;
        let indices_dev = stream.clone_htod(indices).map_err(map_driver_err)?;
        Ok(GpuGroups {
            offsets: offsets_dev,
            indices: indices_dev,
            n_groups,
            max_len,
        })
    }

    pub fn anchors(&self, anchors: &[u32]) -> TrajResult<GpuAnchors> {
        if anchors.len() % 3 != 0 {
            return Err(TrajError::Mismatch(
                "anchors length must be a multiple of 3".into(),
            ));
        }
        let n_groups = anchors.len() / 3;
        let stream = &self.inner.stream;
        let anchors_dev = stream.clone_htod(anchors).map_err(map_driver_err)?;
        Ok(GpuAnchors {
            anchors: anchors_dev,
            n_groups,
        })
    }

    pub fn reference(&self, coords: &[Float4]) -> TrajResult<GpuReference> {
        let coords_dev = self
            .inner
            .stream
            .clone_htod(coords)
            .map_err(map_driver_err)?;
        Ok(GpuReference {
            coords: coords_dev,
            n_sel: coords.len(),
        })
    }

    pub fn polymer_data(
        &self,
        chain_offsets: &[u32],
        chain_indices: &[u32],
        bond_pairs: Option<&[u32]>,
        angle_triplets: Option<&[u32]>,
    ) -> TrajResult<GpuPolymer> {
        if chain_offsets.len() < 2 {
            return Err(TrajError::Mismatch(
                "polymer chain offsets must include at least one chain".into(),
            ));
        }
        let n_chains = chain_offsets.len() - 1;
        let last = *chain_offsets.last().unwrap() as usize;
        if last > chain_indices.len() {
            return Err(TrajError::Mismatch(
                "polymer chain offsets exceed index buffer".into(),
            ));
        }
        let stream = &self.inner.stream;
        let offsets_dev = stream.clone_htod(chain_offsets).map_err(map_driver_err)?;
        let indices_dev = stream.clone_htod(chain_indices).map_err(map_driver_err)?;
        let (bond_dev, n_bonds) = match bond_pairs {
            Some(pairs) => {
                if pairs.len() % 2 != 0 {
                    return Err(TrajError::Mismatch("bond_pairs length must be even".into()));
                }
                let n_bonds = pairs.len() / 2;
                let dev = stream.clone_htod(pairs).map_err(map_driver_err)?;
                (Some(dev), n_bonds)
            }
            None => (None, 0),
        };
        let (angle_dev, n_angles) = match angle_triplets {
            Some(trips) => {
                if trips.len() % 3 != 0 {
                    return Err(TrajError::Mismatch(
                        "angle_triplets length must be a multiple of 3".into(),
                    ));
                }
                let n_angles = trips.len() / 3;
                let dev = stream.clone_htod(trips).map_err(map_driver_err)?;
                (Some(dev), n_angles)
            }
            None => (None, 0),
        };
        Ok(GpuPolymer {
            chain_offsets: offsets_dev,
            chain_indices: indices_dev,
            n_chains,
            bond_pairs: bond_dev,
            n_bonds,
            angle_triplets: angle_dev,
            n_angles,
        })
    }

    pub fn rg(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
        mass_weighted: bool,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut sum_x = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;
        let mut sum_y = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;
        let mut sum_z = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;
        let mut mass_sum = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;
        let mut sum_sq = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;

        let n_sel = sel.n_sel as i32;
        let n_atoms = n_atoms as i32;
        let n_frames_i32 = n_frames as i32;
        let mass_flag = if mass_weighted { 1i32 } else { 0i32 };
        let masses_dev = &sel.masses;

        let block = 256u32;
        let grid_x = ceil_div(sel.n_sel, block);
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (grid_x, n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.rg_accum);
            builder.arg(&coords_dev);
            builder.arg(&sel.sel);
            builder.arg(masses_dev);
            builder.arg(&n_sel);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&mass_flag);
            builder.arg(&mut sum_x);
            builder.arg(&mut sum_y);
            builder.arg(&mut sum_z);
            builder.arg(&mut mass_sum);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.rg_sumsq);
            builder.arg(&coords_dev);
            builder.arg(&sel.sel);
            builder.arg(masses_dev);
            builder.arg(&n_sel);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&mass_flag);
            builder.arg(&sum_x);
            builder.arg(&sum_y);
            builder.arg(&sum_z);
            builder.arg(&mass_sum);
            builder.arg(&mut sum_sq);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let cfg_out = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames as usize, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.rg_finalize);
            builder.arg(&mass_sum);
            builder.arg(&sum_sq);
            builder.arg(&n_frames_i32);
            builder.arg(&mut out);
            builder.launch(cfg_out).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames as usize];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn rg_f32x4(
        &self,
        coords: &[[f32; 4]],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
        mass_weighted: bool,
    ) -> TrajResult<Vec<f32>> {
        self.rg(
            coords_as_float4(coords),
            n_atoms,
            n_frames,
            sel,
            mass_weighted,
        )
    }

    pub fn msd(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
        origin: &GpuReference,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut sum_sq = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;
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
            let mut builder = stream.launch_builder(&self.inner.kernels.msd_accum);
            builder.arg(&coords_dev);
            builder.arg(&origin.coords);
            builder.arg(&sel.sel);
            builder.arg(&n_sel);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&mut sum_sq);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let cfg_out = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames as usize, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.msd_finalize);
            builder.arg(&sum_sq);
            builder.arg(&n_frames_i32);
            builder.arg(&n_sel);
            builder.arg(&mut out);
            builder.launch(cfg_out).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames as usize];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn rmsd_covariance(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
        reference: &GpuReference,
    ) -> TrajResult<RmsdCovariance> {
        if n_frames == 0 {
            return Ok(RmsdCovariance {
                cov: Vec::new(),
                sum_x2: Vec::new(),
                sum_y2: Vec::new(),
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
        let mut cov = stream
            .alloc_zeros::<f32>(n_frames * 9)
            .map_err(map_driver_err)?;
        let mut sum_x2 = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;
        let mut sum_y2 = stream
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
            let mut builder = stream.launch_builder(&self.inner.kernels.rmsd_centroid);
            builder.arg(&coords_dev);
            builder.arg(&reference.coords);
            builder.arg(&sel.sel);
            builder.arg(&n_sel);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&mut sum_x);
            builder.arg(&mut sum_y);
            builder.launch(cfg).map_err(map_driver_err)?;
        }
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.rmsd_cov);
            builder.arg(&coords_dev);
            builder.arg(&reference.coords);
            builder.arg(&sel.sel);
            builder.arg(&n_sel);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&sum_x);
            builder.arg(&sum_y);
            builder.arg(&mut cov);
            builder.arg(&mut sum_x2);
            builder.arg(&mut sum_y2);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut cov_host = vec![0.0f32; n_frames as usize * 9];
        let mut sum_x2_host = vec![0.0f32; n_frames as usize];
        let mut sum_y2_host = vec![0.0f32; n_frames as usize];
        stream
            .memcpy_dtoh(&cov, &mut cov_host)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&sum_x2, &mut sum_x2_host)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&sum_y2, &mut sum_y2_host)
            .map_err(map_driver_err)?;

        let mut cov_frames = Vec::with_capacity(n_frames as usize);
        for frame in 0..(n_frames as usize) {
            let mut mat = [0.0f32; 9];
            let offset = frame * 9;
            mat.copy_from_slice(&cov_host[offset..offset + 9]);
            cov_frames.push(mat);
        }

        Ok(RmsdCovariance {
            cov: cov_frames,
            sum_x2: sum_x2_host,
            sum_y2: sum_y2_host,
        })
    }

    pub fn rmsd_raw(
        &self,
        coords: &[Float4],
        n_atoms: usize,
        n_frames: usize,
        sel: &GpuSelection,
        reference: &GpuReference,
    ) -> TrajResult<Vec<f32>> {
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let mut sum_sq = stream
            .alloc_zeros::<f32>(n_frames)
            .map_err(map_driver_err)?;
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
            let mut builder = stream.launch_builder(&self.inner.kernels.rmsd_raw_accum);
            builder.arg(&coords_dev);
            builder.arg(&reference.coords);
            builder.arg(&sel.sel);
            builder.arg(&n_sel);
            builder.arg(&n_atoms);
            builder.arg(&n_frames_i32);
            builder.arg(&mut sum_sq);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let cfg_out = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames as usize, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.rmsd_finalize);
            builder.arg(&sum_sq);
            builder.arg(&n_frames_i32);
            builder.arg(&n_sel);
            builder.arg(&mut out);
            builder.launch(cfg_out).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_frames as usize];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn distance_to_point(
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
        let stream = &self.inner.stream;
        let coords_dev = stream.clone_htod(coords).map_err(map_driver_err)?;
        let boxes_dev = stream.clone_htod(boxes).map_err(map_driver_err)?;
        let mut out = stream
            .alloc_zeros::<f32>(n_frames * sel.n_sel)
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
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(sel.n_sel, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.distance_to_point);
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

        let mut host = vec![0.0f32; n_frames * sel.n_sel];
        stream
            .memcpy_dtoh(&out, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }
}
