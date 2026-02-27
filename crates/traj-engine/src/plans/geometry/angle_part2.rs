#[cfg(feature = "cuda")]
struct DihedralGpuState {
    groups: GpuGroups,
    masses: GpuBufferF32,
}


impl DihedralPlan {

    pub fn new(
        sel_a: Selection,
        sel_b: Selection,
        sel_c: Selection,
        sel_d: Selection,
        mass_weighted: bool,
        pbc: PbcMode,
        degrees: bool,
        range360: bool,
    ) -> Self {
        Self {
            sel_a,
            sel_b,
            sel_c,
            sel_d,
            mass_weighted,
            pbc,
            degrees,
            range360,
            results: Vec::new(),

            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}


impl Plan for DihedralPlan {
    fn name(&self) -> &'static str {
        "dihedral"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let mut indices = Vec::with_capacity(
                    self.sel_a.indices.len()
                        + self.sel_b.indices.len()
                        + self.sel_c.indices.len()
                        + self.sel_d.indices.len(),
                );
                indices.extend(self.sel_a.indices.iter().copied());
                indices.extend(self.sel_b.indices.iter().copied());
                indices.extend(self.sel_c.indices.iter().copied());
                indices.extend(self.sel_d.indices.iter().copied());
                let offsets = vec![
                    0u32,
                    self.sel_a.indices.len() as u32,
                    (self.sel_a.indices.len() + self.sel_b.indices.len()) as u32,
                    (self.sel_a.indices.len()
                        + self.sel_b.indices.len()
                        + self.sel_c.indices.len()) as u32,
                    (self.sel_a.indices.len()
                        + self.sel_b.indices.len()
                        + self.sel_c.indices.len()
                        + self.sel_d.indices.len()) as u32,
                ];
                let max_len = self
                    .sel_a
                    .indices
                    .len()
                    .max(self.sel_b.indices.len())
                    .max(self.sel_c.indices.len())
                    .max(self.sel_d.indices.len());
                let groups = ctx.groups(&offsets, &indices, max_len)?;
                let masses = if self.mass_weighted {
                    _system.atoms.mass.clone()
                } else {
                    vec![1.0f32; _system.n_atoms()]
                };
                let masses = ctx.upload_f32(&masses)?;
                self.gpu = Some(DihedralGpuState {
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
            let angles = ctx.dihedral_from_coms(
                &coms,
                chunk.n_frames,
                &boxes,
                self.degrees,
                self.range360,
            )?;
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
            let d = center_of_selection(
                chunk,
                frame,
                &self.sel_d.indices,
                masses,
                self.mass_weighted,
            );
            let mut b0x = a[0] - b[0];
            let mut b0y = a[1] - b[1];
            let mut b0z = a[2] - b[2];
            let mut b1x = c[0] - b[0];
            let mut b1y = c[1] - b[1];
            let mut b1z = c[2] - b[2];
            let mut b2x = d[0] - c[0];
            let mut b2y = d[1] - c[1];
            let mut b2z = d[2] - c[2];
            if matches!(self.pbc, PbcMode::Orthorhombic) {
                let (lx, ly, lz) = box_lengths(chunk, frame)?;
                apply_pbc(&mut b0x, &mut b0y, &mut b0z, lx, ly, lz);
                apply_pbc(&mut b1x, &mut b1y, &mut b1z, lx, ly, lz);
                apply_pbc(&mut b2x, &mut b2y, &mut b2z, lx, ly, lz);
            }
            self.results.push(dihedral_from_vectors(
                [b0x, b0y, b0z],
                [b1x, b1y, b1z],
                [b2x, b2y, b2z],
                self.degrees,
                self.range360,
            ));
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}
