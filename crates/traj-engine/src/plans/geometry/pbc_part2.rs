pub struct XtalSymmPlan {
    inner: ReplicateCellPlan,
    selection: Selection,
    symmetry_ops: Option<Vec<[f64; 12]>>,
    results: Vec<f32>,
}


impl XtalSymmPlan {
    pub fn new(selection: Selection, repeats: [usize; 3]) -> Self {
        Self {
            inner: ReplicateCellPlan::new(selection.clone(), repeats),
            selection,
            symmetry_ops: None,
            results: Vec::new(),
        }
    }

    pub fn with_symmetry_ops(mut self, symmetry_ops: Option<Vec<[f64; 12]>>) -> Self {
        self.symmetry_ops = symmetry_ops.filter(|ops| !ops.is_empty());
        self
    }
}


impl Plan for XtalSymmPlan {
    fn name(&self) -> &'static str {
        "xtalsymm"
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.results.clear();
        if self.symmetry_ops.is_some() {
            return Ok(());
        }
        self.inner.init(system, device)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        if let Some(ops) = &self.symmetry_ops {
            let n_atoms = chunk.n_atoms;
            let sel = &self.selection.indices;
            for frame in 0..chunk.n_frames {
                let frame_offset = frame * n_atoms;
                for op in ops.iter() {
                    for &idx in sel.iter() {
                        let p = chunk.coords[frame_offset + idx as usize];
                        let x = p[0] as f64;
                        let y = p[1] as f64;
                        let z = p[2] as f64;
                        let ox = op[0] * x + op[1] * y + op[2] * z + op[3];
                        let oy = op[4] * x + op[5] * y + op[6] * z + op[7];
                        let oz = op[8] * x + op[9] * y + op[10] * z + op[11];
                        self.results.push(ox as f32);
                        self.results.push(oy as f32);
                        self.results.push(oz as f32);
                    }
                }
            }
            return Ok(());
        }
        self.inner.process_chunk(chunk, system, device)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.symmetry_ops.is_some() {
            return Ok(PlanOutput::Series(std::mem::take(&mut self.results)));
        }
        self.inner.finalize()
    }
}



pub struct VolumePlan {
    results: Vec<f32>,
}


impl VolumePlan {

    pub fn new() -> Self {
        Self { results: Vec::new() }
    }
}


impl Plan for VolumePlan {
    fn name(&self) -> &'static str {
        "volume"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {

        #[cfg(feature = "cuda")]
        if let Device::Cuda(ctx) = _device {
            let all_orth = chunk
                .box_
                .iter()
                .all(|b| matches!(b, Box3::Orthorhombic { .. }));
            if all_orth {
                let boxes = chunk_boxes(chunk, PbcMode::Orthorhombic)?;
                let out = ctx.volume_orthorhombic(&boxes, chunk.n_frames)?;
                self.results.extend(out);
                return Ok(());
            }
            let (cell, _inv) = chunk_cell_mats(chunk)?;
            let out = ctx.volume_cell(&cell, chunk.n_frames)?;
            self.results.extend(out);
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            match chunk.box_[frame] {
                Box3::Orthorhombic { lx, ly, lz } => {
                    self.results.push((lx * ly * lz) as f32);
                }
                Box3::Triclinic { m } => {
                    let m0 = m[0] as f64;
                    let m1 = m[1] as f64;
                    let m2 = m[2] as f64;
                    let m3 = m[3] as f64;
                    let m4 = m[4] as f64;
                    let m5 = m[5] as f64;
                    let m6 = m[6] as f64;
                    let m7 = m[7] as f64;
                    let m8 = m[8] as f64;
                    let det = m0 * (m4 * m8 - m5 * m7)
                        - m1 * (m3 * m8 - m5 * m6)
                        + m2 * (m3 * m7 - m4 * m6);
                    self.results.push(det.abs() as f32);
                }
                Box3::None => {
                    return Err(TrajError::Mismatch(
                        "volume requires orthorhombic or triclinic box".into(),
                    ))
                }
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}
