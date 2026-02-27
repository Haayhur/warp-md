use super::*;

impl GpuContext {
    pub fn pack_short_tol_penalty_grad_cells(
        &self,
        positions: &[Float4],
        radii: &[f32],
        short_radius: &[f32],
        fscale: &[f32],
        short_scale: &[f32],
        use_short: &[u8],
        mol_id: &[i32],
        movable: &[u8],
        cell_offsets: &[i32],
        cell_atoms: &[i32],
        dims: [i32; 3],
        box_min: [f32; 3],
        box_len: [f32; 3],
        cell_size: f32,
    ) -> TrajResult<(Vec<f32>, Vec<Float4>)> {
        let n_atoms = positions.len();
        if n_atoms == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        if radii.len() != n_atoms
            || short_radius.len() != n_atoms
            || fscale.len() != n_atoms
            || short_scale.len() != n_atoms
            || use_short.len() != n_atoms
            || mol_id.len() != n_atoms
            || movable.len() != n_atoms
        {
            return Err(TrajError::Mismatch(
                "pack_short_tol_penalty_grad_cells input lengths do not match".into(),
            ));
        }
        if cell_offsets.len() < 2 {
            return Err(TrajError::Mismatch(
                "pack_short_tol_penalty_grad_cells requires non-empty cell offsets".into(),
            ));
        }
        let stream = &self.inner.stream;
        let pos_dev = stream.clone_htod(positions).map_err(map_driver_err)?;
        let rad_dev = stream.clone_htod(radii).map_err(map_driver_err)?;
        let short_rad_dev = stream.clone_htod(short_radius).map_err(map_driver_err)?;
        let fscale_dev = stream.clone_htod(fscale).map_err(map_driver_err)?;
        let short_scale_dev = stream.clone_htod(short_scale).map_err(map_driver_err)?;
        let use_short_dev = stream.clone_htod(use_short).map_err(map_driver_err)?;
        let mol_dev = stream.clone_htod(mol_id).map_err(map_driver_err)?;
        let movable_dev = stream.clone_htod(movable).map_err(map_driver_err)?;
        let offsets_dev = stream.clone_htod(cell_offsets).map_err(map_driver_err)?;
        let atoms_dev = stream.clone_htod(cell_atoms).map_err(map_driver_err)?;
        let mut out_penalty_dev = stream.alloc_zeros::<f32>(n_atoms).map_err(map_driver_err)?;
        let mut out_grad_dev = stream
            .alloc_zeros::<Float4>(n_atoms)
            .map_err(map_driver_err)?;
        let n_atoms_i32 = n_atoms as i32;
        let box_min = Float4 {
            x: box_min[0],
            y: box_min[1],
            z: box_min[2],
            w: 0.0,
        };
        let box_len = Float4 {
            x: box_len[0],
            y: box_len[1],
            z: box_len[2],
            w: 0.0,
        };
        let inv_cell = if cell_size > 0.0 {
            1.0 / cell_size
        } else {
            0.0
        };
        let nx = dims[0];
        let ny = dims[1];
        let nz = dims[2];

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_atoms, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder =
                stream.launch_builder(&self.inner.kernels.pack_short_tol_penalty_grad_cells);
            builder.arg(&pos_dev);
            builder.arg(&rad_dev);
            builder.arg(&short_rad_dev);
            builder.arg(&fscale_dev);
            builder.arg(&short_scale_dev);
            builder.arg(&use_short_dev);
            builder.arg(&mol_dev);
            builder.arg(&movable_dev);
            builder.arg(&n_atoms_i32);
            builder.arg(&box_min);
            builder.arg(&box_len);
            builder.arg(&cell_size);
            builder.arg(&inv_cell);
            builder.arg(&nx);
            builder.arg(&ny);
            builder.arg(&nz);
            builder.arg(&offsets_dev);
            builder.arg(&atoms_dev);
            builder.arg(&mut out_penalty_dev);
            builder.arg(&mut out_grad_dev);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut penalty = vec![0.0f32; n_atoms];
        let mut grad = vec![Float4::default(); n_atoms];
        stream
            .memcpy_dtoh(&out_penalty_dev, &mut penalty)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&out_grad_dev, &mut grad)
            .map_err(map_driver_err)?;
        Ok((penalty, grad))
    }

    pub fn pack_constraint_penalty(
        &self,
        positions: &[Float4],
        types: &[u8],
        modes: &[u8],
        data0: &[Float4],
        data1: &[Float4],
        atom_offsets: &[i32],
        atom_constraints: &[i32],
    ) -> TrajResult<(Vec<f32>, Vec<Float4>, Vec<f32>, Vec<f32>)> {
        let n_atoms = positions.len();
        if n_atoms == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }
        if types.len() != modes.len() || types.len() != data0.len() || types.len() != data1.len() {
            return Err(TrajError::Mismatch(
                "pack_constraint_penalty constraint arrays length mismatch".into(),
            ));
        }
        if atom_offsets.len() != n_atoms + 1 {
            return Err(TrajError::Mismatch(
                "pack_constraint_penalty atom_offsets length mismatch".into(),
            ));
        }
        let stream = &self.inner.stream;
        let pos_dev = stream.clone_htod(positions).map_err(map_driver_err)?;
        let types_dev = stream.clone_htod(types).map_err(map_driver_err)?;
        let modes_dev = stream.clone_htod(modes).map_err(map_driver_err)?;
        let data0_dev = stream.clone_htod(data0).map_err(map_driver_err)?;
        let data1_dev = stream.clone_htod(data1).map_err(map_driver_err)?;
        let offsets_dev = stream.clone_htod(atom_offsets).map_err(map_driver_err)?;
        let indices_dev = stream
            .clone_htod(atom_constraints)
            .map_err(map_driver_err)?;
        let mut sum_dev = stream.alloc_zeros::<f32>(n_atoms).map_err(map_driver_err)?;
        let mut grad_dev = stream
            .alloc_zeros::<Float4>(n_atoms)
            .map_err(map_driver_err)?;
        let mut max_val_dev = stream.alloc_zeros::<f32>(n_atoms).map_err(map_driver_err)?;
        let mut max_viol_dev = stream.alloc_zeros::<f32>(n_atoms).map_err(map_driver_err)?;
        let n_atoms_i32 = n_atoms as i32;

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_atoms, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.pack_constraint_penalty);
            builder.arg(&pos_dev);
            builder.arg(&types_dev);
            builder.arg(&modes_dev);
            builder.arg(&data0_dev);
            builder.arg(&data1_dev);
            builder.arg(&offsets_dev);
            builder.arg(&indices_dev);
            builder.arg(&n_atoms_i32);
            builder.arg(&mut sum_dev);
            builder.arg(&mut grad_dev);
            builder.arg(&mut max_val_dev);
            builder.arg(&mut max_viol_dev);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut sum = vec![0.0f32; n_atoms];
        let mut grad = vec![Float4::default(); n_atoms];
        let mut max_val = vec![0.0f32; n_atoms];
        let mut max_viol = vec![0.0f32; n_atoms];
        stream
            .memcpy_dtoh(&sum_dev, &mut sum)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&grad_dev, &mut grad)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&max_val_dev, &mut max_val)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&max_viol_dev, &mut max_viol)
            .map_err(map_driver_err)?;
        Ok((sum, grad, max_val, max_viol))
    }

    pub fn pack_relax_accum(
        &self,
        positions: &[Float4],
        radii: &[f32],
        mol_id: &[i32],
        mol_movable: &[u8],
        cell_offsets: &[i32],
        cell_atoms: &[i32],
        dims: [i32; 3],
        box_min: [f32; 3],
        n_mols: usize,
        cell_size: f32,
    ) -> TrajResult<(Vec<Float4>, f32)> {
        let n_atoms = positions.len();
        if n_atoms == 0 || n_mols == 0 {
            return Ok((Vec::new(), 0.0));
        }
        if radii.len() != n_atoms || mol_id.len() != n_atoms {
            return Err(TrajError::Mismatch(
                "pack_relax_accum input lengths do not match".into(),
            ));
        }
        if mol_movable.len() != n_mols {
            return Err(TrajError::Mismatch(
                "pack_relax_accum mol_movable length mismatch".into(),
            ));
        }
        if cell_offsets.len() < 2 {
            return Err(TrajError::Mismatch(
                "pack_relax_accum requires non-empty cell offsets".into(),
            ));
        }
        let stream = &self.inner.stream;
        let pos_dev = stream.clone_htod(positions).map_err(map_driver_err)?;
        let rad_dev = stream.clone_htod(radii).map_err(map_driver_err)?;
        let mol_dev = stream.clone_htod(mol_id).map_err(map_driver_err)?;
        let movable_dev = stream.clone_htod(mol_movable).map_err(map_driver_err)?;
        let offsets_dev = stream.clone_htod(cell_offsets).map_err(map_driver_err)?;
        let atoms_dev = stream.clone_htod(cell_atoms).map_err(map_driver_err)?;
        let mut disp_dev = stream
            .alloc_zeros::<Float4>(n_mols)
            .map_err(map_driver_err)?;
        let mut max_dev = stream.alloc_zeros::<f32>(1).map_err(map_driver_err)?;

        let n_atoms_i32 = n_atoms as i32;
        let n_mols_i32 = n_mols as i32;
        let box_min = Float4 {
            x: box_min[0],
            y: box_min[1],
            z: box_min[2],
            w: 0.0,
        };
        let inv_cell = if cell_size > 0.0 {
            1.0 / cell_size
        } else {
            0.0
        };
        let nx = dims[0];
        let ny = dims[1];
        let nz = dims[2];

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_atoms, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.pack_relax_accum);
            builder.arg(&pos_dev);
            builder.arg(&rad_dev);
            builder.arg(&mol_dev);
            builder.arg(&movable_dev);
            builder.arg(&n_atoms_i32);
            builder.arg(&n_mols_i32);
            builder.arg(&box_min);
            builder.arg(&cell_size);
            builder.arg(&inv_cell);
            builder.arg(&nx);
            builder.arg(&ny);
            builder.arg(&nz);
            builder.arg(&offsets_dev);
            builder.arg(&atoms_dev);
            builder.arg(&mut disp_dev);
            builder.arg(&mut max_dev);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut disp = vec![Float4::default(); n_mols];
        let mut max_host = vec![0.0f32; 1];
        stream
            .memcpy_dtoh(&disp_dev, &mut disp)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&max_dev, &mut max_host)
            .map_err(map_driver_err)?;
        Ok((disp, max_host[0]))
    }
    pub fn pack_overlap_penalty_cells(
        &self,
        positions: &[Float4],
        radii: &[f32],
        fscale: &[f32],
        mol_id: &[i32],
        movable: &[u8],
        cell_offsets: &[i32],
        cell_atoms: &[i32],
        dims: [i32; 3],
        box_min: [f32; 3],
        box_len: [f32; 3],
        cell_size: f32,
    ) -> TrajResult<Vec<f32>> {
        if positions.len() != radii.len()
            || positions.len() != mol_id.len()
            || positions.len() != fscale.len()
            || positions.len() != movable.len()
        {
            return Err(TrajError::Mismatch(
                "pack_overlap_penalty_cells input lengths do not match".into(),
            ));
        }
        if cell_offsets.len() < 2 {
            return Err(TrajError::Mismatch(
                "pack_overlap_penalty_cells requires non-empty cell offsets".into(),
            ));
        }
        let n_atoms = positions.len();
        if n_atoms == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let pos_dev = stream.clone_htod(positions).map_err(map_driver_err)?;
        let rad_dev = stream.clone_htod(radii).map_err(map_driver_err)?;
        let fscale_dev = stream.clone_htod(fscale).map_err(map_driver_err)?;
        let mol_dev = stream.clone_htod(mol_id).map_err(map_driver_err)?;
        let movable_dev = stream.clone_htod(movable).map_err(map_driver_err)?;
        let offsets_dev = stream.clone_htod(cell_offsets).map_err(map_driver_err)?;
        let atoms_dev = stream.clone_htod(cell_atoms).map_err(map_driver_err)?;
        let mut out_dev = stream.alloc_zeros::<f32>(n_atoms).map_err(map_driver_err)?;
        let n_atoms_i32 = n_atoms as i32;
        let box_min = Float4 {
            x: box_min[0],
            y: box_min[1],
            z: box_min[2],
            w: 0.0,
        };
        let box_len = Float4 {
            x: box_len[0],
            y: box_len[1],
            z: box_len[2],
            w: 0.0,
        };
        let inv_cell = if cell_size > 0.0 {
            1.0 / cell_size
        } else {
            0.0
        };
        let nx = dims[0];
        let ny = dims[1];
        let nz = dims[2];

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_atoms, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.pack_overlap_penalty_cells);
            builder.arg(&pos_dev);
            builder.arg(&rad_dev);
            builder.arg(&fscale_dev);
            builder.arg(&mol_dev);
            builder.arg(&movable_dev);
            builder.arg(&n_atoms_i32);
            builder.arg(&box_min);
            builder.arg(&box_len);
            builder.arg(&cell_size);
            builder.arg(&inv_cell);
            builder.arg(&nx);
            builder.arg(&ny);
            builder.arg(&nz);
            builder.arg(&offsets_dev);
            builder.arg(&atoms_dev);
            builder.arg(&mut out_dev);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_atoms];
        stream
            .memcpy_dtoh(&out_dev, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn pack_overlap_max_cells_movable(
        &self,
        positions: &[Float4],
        radii: &[f32],
        mol_id: &[i32],
        movable: &[u8],
        cell_offsets: &[i32],
        cell_atoms: &[i32],
        dims: [i32; 3],
        box_min: [f32; 3],
        box_len: [f32; 3],
        cell_size: f32,
    ) -> TrajResult<Vec<f32>> {
        if positions.len() != radii.len() || positions.len() != mol_id.len() {
            return Err(TrajError::Mismatch(
                "pack_overlap_max_cells_movable input lengths do not match".into(),
            ));
        }
        if positions.len() != movable.len() {
            return Err(TrajError::Mismatch(
                "pack_overlap_max_cells_movable movable length does not match".into(),
            ));
        }
        if cell_offsets.len() < 2 {
            return Err(TrajError::Mismatch(
                "pack_overlap_max_cells_movable requires non-empty cell offsets".into(),
            ));
        }
        let n_atoms = positions.len();
        if n_atoms == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let pos_dev = stream.clone_htod(positions).map_err(map_driver_err)?;
        let rad_dev = stream.clone_htod(radii).map_err(map_driver_err)?;
        let mol_dev = stream.clone_htod(mol_id).map_err(map_driver_err)?;
        let movable_dev = stream.clone_htod(movable).map_err(map_driver_err)?;
        let offsets_dev = stream.clone_htod(cell_offsets).map_err(map_driver_err)?;
        let atoms_dev = stream.clone_htod(cell_atoms).map_err(map_driver_err)?;
        let mut out_dev = stream.alloc_zeros::<f32>(n_atoms).map_err(map_driver_err)?;
        let n_atoms_i32 = n_atoms as i32;
        let box_min = Float4 {
            x: box_min[0],
            y: box_min[1],
            z: box_min[2],
            w: 0.0,
        };
        let box_len = Float4 {
            x: box_len[0],
            y: box_len[1],
            z: box_len[2],
            w: 0.0,
        };
        let inv_cell = if cell_size > 0.0 {
            1.0 / cell_size
        } else {
            0.0
        };
        let nx = dims[0];
        let ny = dims[1];
        let nz = dims[2];

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_atoms, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder =
                stream.launch_builder(&self.inner.kernels.pack_overlap_max_cells_movable);
            builder.arg(&pos_dev);
            builder.arg(&rad_dev);
            builder.arg(&mol_dev);
            builder.arg(&movable_dev);
            builder.arg(&n_atoms_i32);
            builder.arg(&box_min);
            builder.arg(&box_len);
            builder.arg(&cell_size);
            builder.arg(&inv_cell);
            builder.arg(&nx);
            builder.arg(&ny);
            builder.arg(&nz);
            builder.arg(&offsets_dev);
            builder.arg(&atoms_dev);
            builder.arg(&mut out_dev);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_atoms];
        stream
            .memcpy_dtoh(&out_dev, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn pack_overlap_grad_cells(
        &self,
        positions: &[Float4],
        radii: &[f32],
        fscale: &[f32],
        mol_id: &[i32],
        movable: &[u8],
        cell_offsets: &[i32],
        cell_atoms: &[i32],
        dims: [i32; 3],
        box_min: [f32; 3],
        box_len: [f32; 3],
        cell_size: f32,
    ) -> TrajResult<Vec<Float4>> {
        if positions.len() != radii.len()
            || positions.len() != mol_id.len()
            || positions.len() != fscale.len()
            || positions.len() != movable.len()
        {
            return Err(TrajError::Mismatch(
                "pack_overlap_grad_cells input lengths do not match".into(),
            ));
        }
        if cell_offsets.len() < 2 {
            return Err(TrajError::Mismatch(
                "pack_overlap_grad_cells requires non-empty cell offsets".into(),
            ));
        }
        let n_atoms = positions.len();
        if n_atoms == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let pos_dev = stream.clone_htod(positions).map_err(map_driver_err)?;
        let rad_dev = stream.clone_htod(radii).map_err(map_driver_err)?;
        let fscale_dev = stream.clone_htod(fscale).map_err(map_driver_err)?;
        let mol_dev = stream.clone_htod(mol_id).map_err(map_driver_err)?;
        let movable_dev = stream.clone_htod(movable).map_err(map_driver_err)?;
        let offsets_dev = stream.clone_htod(cell_offsets).map_err(map_driver_err)?;
        let atoms_dev = stream.clone_htod(cell_atoms).map_err(map_driver_err)?;
        let mut out_dev = stream
            .alloc_zeros::<Float4>(n_atoms)
            .map_err(map_driver_err)?;
        let n_atoms_i32 = n_atoms as i32;
        let box_min = Float4 {
            x: box_min[0],
            y: box_min[1],
            z: box_min[2],
            w: 0.0,
        };
        let box_len = Float4 {
            x: box_len[0],
            y: box_len[1],
            z: box_len[2],
            w: 0.0,
        };
        let inv_cell = if cell_size > 0.0 {
            1.0 / cell_size
        } else {
            0.0
        };
        let nx = dims[0];
        let ny = dims[1];
        let nz = dims[2];

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_atoms, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.pack_overlap_grad_cells);
            builder.arg(&pos_dev);
            builder.arg(&rad_dev);
            builder.arg(&fscale_dev);
            builder.arg(&mol_dev);
            builder.arg(&movable_dev);
            builder.arg(&n_atoms_i32);
            builder.arg(&box_min);
            builder.arg(&box_len);
            builder.arg(&cell_size);
            builder.arg(&inv_cell);
            builder.arg(&nx);
            builder.arg(&ny);
            builder.arg(&nz);
            builder.arg(&offsets_dev);
            builder.arg(&atoms_dev);
            builder.arg(&mut out_dev);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![Float4::default(); n_atoms];
        stream
            .memcpy_dtoh(&out_dev, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn pack_overlap_max_cells(
        &self,
        positions: &[Float4],
        radii: &[f32],
        mol_id: &[i32],
        cell_offsets: &[i32],
        cell_atoms: &[i32],
        dims: [i32; 3],
        box_min: [f32; 3],
        box_len: [f32; 3],
        cell_size: f32,
    ) -> TrajResult<Vec<f32>> {
        if positions.len() != radii.len() || positions.len() != mol_id.len() {
            return Err(TrajError::Mismatch(
                "pack_overlap_max_cells input lengths do not match".into(),
            ));
        }
        if cell_offsets.len() < 2 {
            return Err(TrajError::Mismatch(
                "pack_overlap_max_cells requires non-empty cell offsets".into(),
            ));
        }
        let n_atoms = positions.len();
        if n_atoms == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let pos_dev = stream.clone_htod(positions).map_err(map_driver_err)?;
        let rad_dev = stream.clone_htod(radii).map_err(map_driver_err)?;
        let mol_dev = stream.clone_htod(mol_id).map_err(map_driver_err)?;
        let offsets_dev = stream.clone_htod(cell_offsets).map_err(map_driver_err)?;
        let atoms_dev = stream.clone_htod(cell_atoms).map_err(map_driver_err)?;
        let mut out_dev = stream.alloc_zeros::<f32>(n_atoms).map_err(map_driver_err)?;
        let n_atoms_i32 = n_atoms as i32;
        let box_min = Float4 {
            x: box_min[0],
            y: box_min[1],
            z: box_min[2],
            w: 0.0,
        };
        let box_len = Float4 {
            x: box_len[0],
            y: box_len[1],
            z: box_len[2],
            w: 0.0,
        };
        let inv_cell = if cell_size > 0.0 {
            1.0 / cell_size
        } else {
            0.0
        };
        let nx = dims[0];
        let ny = dims[1];
        let nz = dims[2];

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_atoms, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.pack_overlap_max_cells);
            builder.arg(&pos_dev);
            builder.arg(&rad_dev);
            builder.arg(&mol_dev);
            builder.arg(&n_atoms_i32);
            builder.arg(&box_min);
            builder.arg(&box_len);
            builder.arg(&cell_size);
            builder.arg(&inv_cell);
            builder.arg(&nx);
            builder.arg(&ny);
            builder.arg(&nz);
            builder.arg(&offsets_dev);
            builder.arg(&atoms_dev);
            builder.arg(&mut out_dev);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_atoms];
        stream
            .memcpy_dtoh(&out_dev, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn pack_overlap_max(
        &self,
        positions: &[Float4],
        radii: &[f32],
        mol_id: &[i32],
        box_l: Option<[f32; 3]>,
    ) -> TrajResult<Vec<f32>> {
        if positions.len() != radii.len() || positions.len() != mol_id.len() {
            return Err(TrajError::Mismatch(
                "pack_overlap_max input lengths do not match".into(),
            ));
        }
        let n_atoms = positions.len();
        if n_atoms == 0 {
            return Ok(Vec::new());
        }
        let stream = &self.inner.stream;
        let pos_dev = stream.clone_htod(positions).map_err(map_driver_err)?;
        let rad_dev = stream.clone_htod(radii).map_err(map_driver_err)?;
        let mol_dev = stream.clone_htod(mol_id).map_err(map_driver_err)?;
        let mut out_dev = stream.alloc_zeros::<f32>(n_atoms).map_err(map_driver_err)?;
        let n_atoms_i32 = n_atoms as i32;
        let box_v = match box_l {
            Some([x, y, z]) => Float4 { x, y, z, w: 0.0 },
            None => Float4 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 0.0,
            },
        };

        let block = 256u32;
        let cfg = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_atoms, block), 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.pack_overlap_max);
            builder.arg(&pos_dev);
            builder.arg(&rad_dev);
            builder.arg(&mol_dev);
            builder.arg(&n_atoms_i32);
            builder.arg(&box_v);
            builder.arg(&mut out_dev);
            builder.launch(cfg).map_err(map_driver_err)?;
        }

        let mut host = vec![0.0f32; n_atoms];
        stream
            .memcpy_dtoh(&out_dev, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn ion_pair_correlations(
        &self,
        com: &[f32],
        box_l: &[f32],
        cat_indices: &[u32],
        ani_indices: &[u32],
        n_groups: usize,
        n_frames: usize,
        max_cluster: usize,
        rclust_cat: f32,
        rclust_ani: f32,
    ) -> TrajResult<(Vec<u32>, Vec<u32>, Vec<u32>, Vec<u32>)> {
        let n_cat = cat_indices.len();
        let n_ani = ani_indices.len();
        if n_frames == 0 || n_cat == 0 || n_ani == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
        }
        let stream = &self.inner.stream;
        let com_dev = stream.clone_htod(com).map_err(map_driver_err)?;
        let box_dev = stream.clone_htod(box_l).map_err(map_driver_err)?;
        let cat_dev = stream.clone_htod(cat_indices).map_err(map_driver_err)?;
        let ani_dev = stream.clone_htod(ani_indices).map_err(map_driver_err)?;
        let mut idx_pair_cat = stream
            .alloc_zeros::<u32>(n_frames * n_cat)
            .map_err(map_driver_err)?;
        let mut idx_cluster_cat = stream
            .alloc_zeros::<u32>(n_frames * n_cat * max_cluster)
            .map_err(map_driver_err)?;
        let mut idx_pair_ani = stream
            .alloc_zeros::<u32>(n_frames * n_ani)
            .map_err(map_driver_err)?;
        let mut idx_cluster_ani = stream
            .alloc_zeros::<u32>(n_frames * n_ani * max_cluster)
            .map_err(map_driver_err)?;

        let n_cat_i32 = n_cat as i32;
        let n_ani_i32 = n_ani as i32;
        let n_groups_i32 = n_groups as i32;
        let n_frames_i32 = n_frames as i32;
        let max_cluster_i32 = max_cluster as i32;

        let block = 256u32;
        let cfg_cat = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_cat, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.ion_pair_frame_cat);
            builder.arg(&com_dev);
            builder.arg(&box_dev);
            builder.arg(&cat_dev);
            builder.arg(&ani_dev);
            builder.arg(&n_cat_i32);
            builder.arg(&n_ani_i32);
            builder.arg(&n_groups_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&max_cluster_i32);
            builder.arg(&rclust_cat);
            builder.arg(&mut idx_pair_cat);
            builder.arg(&mut idx_cluster_cat);
            builder.launch(cfg_cat).map_err(map_driver_err)?;
        }

        let cfg_ani = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_ani, block), n_frames as u32, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.ion_pair_frame_ani);
            builder.arg(&com_dev);
            builder.arg(&box_dev);
            builder.arg(&cat_dev);
            builder.arg(&ani_dev);
            builder.arg(&n_cat_i32);
            builder.arg(&n_ani_i32);
            builder.arg(&n_groups_i32);
            builder.arg(&n_frames_i32);
            builder.arg(&max_cluster_i32);
            builder.arg(&rclust_ani);
            builder.arg(&mut idx_pair_ani);
            builder.arg(&mut idx_cluster_ani);
            builder.launch(cfg_ani).map_err(map_driver_err)?;
        }

        let mut ip_cat_dev = stream
            .alloc_zeros::<u32>(n_frames)
            .map_err(map_driver_err)?;
        let mut cp_cat_dev = stream
            .alloc_zeros::<u32>(n_frames)
            .map_err(map_driver_err)?;
        let mut ip_ani_dev = stream
            .alloc_zeros::<u32>(n_frames)
            .map_err(map_driver_err)?;
        let mut cp_ani_dev = stream
            .alloc_zeros::<u32>(n_frames)
            .map_err(map_driver_err)?;

        let cfg_corr_cat = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames, block), n_frames as u32, n_cat as u32),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.ion_pair_corr_cat);
            builder.arg(&idx_pair_cat);
            builder.arg(&idx_cluster_cat);
            builder.arg(&n_frames_i32);
            builder.arg(&n_cat_i32);
            builder.arg(&max_cluster_i32);
            builder.arg(&mut ip_cat_dev);
            builder.arg(&mut cp_cat_dev);
            builder.launch(cfg_corr_cat).map_err(map_driver_err)?;
        }

        let cfg_corr_ani = LaunchConfig {
            block_dim: (block, 1, 1),
            grid_dim: (ceil_div(n_frames, block), n_frames as u32, n_ani as u32),
            shared_mem_bytes: 0,
        };
        unsafe {
            let mut builder = stream.launch_builder(&self.inner.kernels.ion_pair_corr_ani);
            builder.arg(&idx_pair_ani);
            builder.arg(&idx_cluster_ani);
            builder.arg(&n_frames_i32);
            builder.arg(&n_ani_i32);
            builder.arg(&max_cluster_i32);
            builder.arg(&mut ip_ani_dev);
            builder.arg(&mut cp_ani_dev);
            builder.launch(cfg_corr_ani).map_err(map_driver_err)?;
        }

        let mut ip_cat = vec![0u32; n_frames];
        let mut cp_cat = vec![0u32; n_frames];
        let mut ip_ani = vec![0u32; n_frames];
        let mut cp_ani = vec![0u32; n_frames];
        stream
            .memcpy_dtoh(&ip_cat_dev, &mut ip_cat)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&cp_cat_dev, &mut cp_cat)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&ip_ani_dev, &mut ip_ani)
            .map_err(map_driver_err)?;
        stream
            .memcpy_dtoh(&cp_ani_dev, &mut cp_ani)
            .map_err(map_driver_err)?;
        Ok((ip_cat, ip_ani, cp_cat, cp_ani))
    }

    pub fn read_counts(&self, counts: &GpuCounts) -> TrajResult<Vec<u64>> {
        let mut host = vec![0u64; counts.inner.len()];
        self.inner
            .stream
            .memcpy_dtoh(&counts.inner, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }

    pub fn alloc_counts(&self, bins: usize) -> TrajResult<GpuCounts> {
        let inner = self
            .inner
            .stream
            .alloc_zeros::<u64>(bins)
            .map_err(map_driver_err)?;
        Ok(GpuCounts { inner })
    }

    pub fn reset_counts(&self, counts: &mut GpuCounts) -> TrajResult<()> {
        self.inner
            .stream
            .memset_zeros(&mut counts.inner)
            .map_err(map_driver_err)
    }

    pub fn alloc_counts_u32(&self, len: usize) -> TrajResult<GpuCountsU32> {
        let inner = self
            .inner
            .stream
            .alloc_zeros::<u32>(len)
            .map_err(map_driver_err)?;
        Ok(GpuCountsU32 { inner })
    }

    pub fn reset_counts_u32(&self, counts: &mut GpuCountsU32) -> TrajResult<()> {
        self.inner
            .stream
            .memset_zeros(&mut counts.inner)
            .map_err(map_driver_err)
    }

    pub fn read_counts_u32(&self, counts: &GpuCountsU32) -> TrajResult<Vec<u32>> {
        let mut host = vec![0u32; counts.inner.len()];
        self.inner
            .stream
            .memcpy_dtoh(&counts.inner, &mut host)
            .map_err(map_driver_err)?;
        Ok(host)
    }
}
