pub struct DistanceToPointPlan {
    selection: Selection,
    point: [f64; 3],
    pbc: PbcMode,
    results: Vec<f32>,
    frames: usize,

    #[cfg(feature = "cuda")]
    gpu: Option<DistancePointGpuState>,
}



#[cfg(feature = "cuda")]
struct DistancePointGpuState {
    selection: GpuSelection,
}


impl DistanceToPointPlan {

    pub fn new(selection: Selection, point: [f64; 3], pbc: PbcMode) -> Self {
        Self {
            selection,
            point,
            pbc,
            results: Vec::new(),
            frames: 0,

            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}


impl Plan for DistanceToPointPlan {
    fn name(&self) -> &'static str {
        "distance_to_point"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;

        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let selection = ctx.selection(&self.selection.indices, None)?;
                self.gpu = Some(DistancePointGpuState { selection });
            }
        }
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {

        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let boxes = chunk_boxes(chunk, self.pbc)?;
            let out = ctx.distance_to_point(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.selection,
                [self.point[0] as f32, self.point[1] as f32, self.point[2] as f32],
                &boxes,
            )?;
            self.results.extend(out);
            self.frames += chunk.n_frames;
            return Ok(());
        }
        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let (lx, ly, lz) = if matches!(self.pbc, PbcMode::Orthorhombic) {
                box_lengths(chunk, frame)?
            } else {
                (0.0, 0.0, 0.0)
            };
            for &idx in self.selection.indices.iter() {
                let p = chunk.coords[frame * n_atoms + idx as usize];
                let mut dx = p[0] as f64 - self.point[0];
                let mut dy = p[1] as f64 - self.point[1];
                let mut dz = p[2] as f64 - self.point[2];
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                }
                self.results.push((dx * dx + dy * dy + dz * dz).sqrt() as f32);
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: self.selection.indices.len(),
        })
    }
}



pub struct DistanceToReferencePlan {
    selection: Selection,
    reference_mode: ReferenceMode,
    pbc: PbcMode,
    reference_sel: Option<Vec<[f32; 4]>>,
    results: Vec<f32>,
    frames: usize,

    #[cfg(feature = "cuda")]
    gpu: Option<DistanceRefGpuState>,
}



#[cfg(feature = "cuda")]
struct DistanceRefGpuState {
    selection: GpuSelection,
    reference: GpuReference,
}


impl DistanceToReferencePlan {

    pub fn new(selection: Selection, reference_mode: ReferenceMode, pbc: PbcMode) -> Self {
        Self {
            selection,
            reference_mode,
            pbc,
            reference_sel: None,
            results: Vec::new(),
            frames: 0,

            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}


impl Plan for DistanceToReferencePlan {
    fn name(&self) -> &'static str {
        "distance_to_reference"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;

        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
        }
        self.reference_sel = match self.reference_mode {
            ReferenceMode::Topology => {
                let positions0 = system
                    .positions0
                    .as_ref()
                    .ok_or_else(|| TrajError::Mismatch("no topology reference coords".into()))?;
                let mut reference = Vec::with_capacity(self.selection.indices.len());
                for &idx in self.selection.indices.iter() {
                    reference.push(positions0[idx as usize]);
                }
                Some(reference)
            }
            ReferenceMode::Frame0 => None,
        };

        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(reference)) = (_device, self.reference_sel.as_ref()) {
            let selection = ctx.selection(&self.selection.indices, None)?;
            let reference_gpu = ctx.reference(&convert_coords(reference))?;
            self.gpu = Some(DistanceRefGpuState {
                selection,
                reference: reference_gpu,
            });
        }
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.reference_sel.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0) {
            if chunk.n_frames == 0 {
                return Ok(());
            }
            let mut reference = Vec::with_capacity(self.selection.indices.len());
            for &idx in self.selection.indices.iter() {
                reference.push(chunk.coords[idx as usize]);
            }
            self.reference_sel = Some(reference);

            #[cfg(feature = "cuda")]
            if let Device::Cuda(ctx) = _device {
                let selection = ctx.selection(&self.selection.indices, None)?;
                let reference_gpu =
                    ctx.reference(&convert_coords(self.reference_sel.as_ref().unwrap()))?;
                self.gpu = Some(DistanceRefGpuState {
                    selection,
                    reference: reference_gpu,
                });
            }
        }

        let reference_sel = self
            .reference_sel
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;


        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let boxes = chunk_boxes(chunk, self.pbc)?;
            let out = ctx.distance_to_reference(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.selection,
                &gpu.reference,
                &boxes,
            )?;
            self.results.extend(out);
            self.frames += chunk.n_frames;
            return Ok(());
        }

        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let (lx, ly, lz) = if matches!(self.pbc, PbcMode::Orthorhombic) {
                box_lengths(chunk, frame)?
            } else {
                (0.0, 0.0, 0.0)
            };
            for (i, &idx) in self.selection.indices.iter().enumerate() {
                let p = chunk.coords[frame * n_atoms + idx as usize];
                let r = reference_sel[i];
                let mut dx = p[0] as f64 - r[0] as f64;
                let mut dy = p[1] as f64 - r[1] as f64;
                let mut dz = p[2] as f64 - r[2] as f64;
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                }
                self.results.push((dx * dx + dy * dy + dz * dz).sqrt() as f32);
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: self.selection.indices.len(),
        })
    }
}



pub struct AnglePlan {
    sel_a: Selection,
    sel_b: Selection,
    sel_c: Selection,
    mass_weighted: bool,
    pbc: PbcMode,
    degrees: bool,
    results: Vec<f32>,

    #[cfg(feature = "cuda")]
    gpu: Option<AngleGpuState>,
}



#[cfg(feature = "cuda")]
struct AngleGpuState {
    groups: GpuGroups,
    masses: GpuBufferF32,
}


impl AnglePlan {

    pub fn new(
        sel_a: Selection,
        sel_b: Selection,
        sel_c: Selection,
        mass_weighted: bool,
        pbc: PbcMode,
        degrees: bool,
    ) -> Self {
        Self {
            sel_a,
            sel_b,
            sel_c,
            mass_weighted,
            pbc,
            degrees,
            results: Vec::new(),

            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}


impl Plan for AnglePlan {
    fn name(&self) -> &'static str {
        "angle"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();

        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let mut indices = Vec::with_capacity(
                    self.sel_a.indices.len() + self.sel_b.indices.len() + self.sel_c.indices.len(),
                );
                indices.extend(self.sel_a.indices.iter().copied());
                indices.extend(self.sel_b.indices.iter().copied());
                indices.extend(self.sel_c.indices.iter().copied());
                let offsets = vec![
                    0u32,
                    self.sel_a.indices.len() as u32,
                    (self.sel_a.indices.len() + self.sel_b.indices.len()) as u32,
                    (self.sel_a.indices.len()
                        + self.sel_b.indices.len()
                        + self.sel_c.indices.len()) as u32,
                ];
                let max_len = self
                    .sel_a
                    .indices
                    .len()
                    .max(self.sel_b.indices.len())
                    .max(self.sel_c.indices.len());
                let groups = ctx.groups(&offsets, &indices, max_len)?;
                let masses = if self.mass_weighted {
                    _system.atoms.mass.clone()
                } else {
                    vec![1.0f32; _system.n_atoms()]
                };
                let masses = ctx.upload_f32(&masses)?;
                self.gpu = Some(AngleGpuState {
                    groups,
                    masses,
                });
            }
        }
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {

        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let coms = ctx.group_com(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.groups,
                &gpu.masses,
                1.0,
            )?;
            let boxes = chunk_boxes(chunk, self.pbc)?;
            let angles = ctx.angle_from_coms(&coms, chunk.n_frames, &boxes, self.degrees)?;
            self.results.extend(angles);
            return Ok(());
        }

        let masses = &system.atoms.mass;
        for frame in 0..chunk.n_frames {
            let a = center_of_selection(
                chunk,
                frame,
                &self.sel_a.indices,
                masses,
                self.mass_weighted,
            );
            let b = center_of_selection(
                chunk,
                frame,
                &self.sel_b.indices,
                masses,
                self.mass_weighted,
            );
            let c = center_of_selection(
                chunk,
                frame,
                &self.sel_c.indices,
                masses,
                self.mass_weighted,
            );
            let mut v1x = a[0] - b[0];
            let mut v1y = a[1] - b[1];
            let mut v1z = a[2] - b[2];
            let mut v2x = c[0] - b[0];
            let mut v2y = c[1] - b[1];
            let mut v2z = c[2] - b[2];
            if matches!(self.pbc, PbcMode::Orthorhombic) {
                let (lx, ly, lz) = box_lengths(chunk, frame)?;
                apply_pbc(&mut v1x, &mut v1y, &mut v1z, lx, ly, lz);
                apply_pbc(&mut v2x, &mut v2y, &mut v2z, lx, ly, lz);
            }
            self.results
                .push(angle_from_vectors([v1x, v1y, v1z], [v2x, v2y, v2z], self.degrees));
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}



pub struct DihedralPlan {
    sel_a: Selection,
    sel_b: Selection,
    sel_c: Selection,
    sel_d: Selection,
    mass_weighted: bool,
    pbc: PbcMode,
    degrees: bool,
    range360: bool,
    results: Vec<f32>,

    #[cfg(feature = "cuda")]
    gpu: Option<DihedralGpuState>,
}
