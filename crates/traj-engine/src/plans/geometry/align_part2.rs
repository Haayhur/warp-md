#[cfg(feature = "cuda")]
struct SuperposeGpuState {
    selection: GpuSelection,
    reference: GpuReference,
}


impl AlignPlan {

    pub fn new(
        selection: Selection,
        reference_mode: ReferenceMode,
        mass_weighted: bool,
        norotate: bool,
    ) -> Self {
        Self {
            selection,
            reference_mode,
            mass_weighted,
            norotate,
            reference_sel: None,
            results: Vec::new(),
            frames: 0,

            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}


impl SuperposePlan {

    pub fn new(
        selection: Selection,
        reference_mode: ReferenceMode,
        mass_weighted: bool,
        norotate: bool,
    ) -> Self {
        Self {
            selection,
            reference_mode,
            mass_weighted,
            norotate,
            reference_sel: None,
            results: Vec::new(),

            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}


impl Plan for AlignPlan {
    fn name(&self) -> &'static str {
        "align"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
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
        {
            self.gpu = None;
            if let (Device::Cuda(ctx), Some(reference)) = (_device, self.reference_sel.as_ref()) {
                let masses = if self.mass_weighted {
                    Some(
                        self.selection
                            .indices
                            .iter()
                            .map(|&idx| system.atoms.mass[idx as usize])
                            .collect::<Vec<f32>>(),
                    )
                } else {
                    None
                };
                let selection = ctx.selection(&self.selection.indices, masses.as_deref())?;
                let reference_gpu = ctx.reference(&convert_coords(reference))?;
                self.gpu = Some(AlignGpuState {
                    selection,
                    reference: reference_gpu,
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
                let masses = if self.mass_weighted {
                    Some(
                        self.selection
                            .indices
                            .iter()
                            .map(|&idx| system.atoms.mass[idx as usize])
                            .collect::<Vec<f32>>(),
                    )
                } else {
                    None
                };
                let selection = ctx.selection(&self.selection.indices, masses.as_deref())?;
                let reference_gpu = ctx.reference(&convert_coords(self.reference_sel.as_ref().unwrap()))?;
                self.gpu = Some(AlignGpuState {
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
            let cov = ctx.align_covariance(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.selection,
                &gpu.reference,
            )?;
            for frame in 0..chunk.n_frames {
                let wsum = cov.sum_w[frame] as f64;
                let mut cx = Vector3::zeros();
                let mut cy = Vector3::zeros();
                if wsum > 0.0 {
                    cx = Vector3::new(
                        cov.sum_x[frame][0] as f64 / wsum,
                        cov.sum_x[frame][1] as f64 / wsum,
                        cov.sum_x[frame][2] as f64 / wsum,
                    );
                    cy = Vector3::new(
                        cov.sum_y[frame][0] as f64 / wsum,
                        cov.sum_y[frame][1] as f64 / wsum,
                        cov.sum_y[frame][2] as f64 / wsum,
                    );
                }
                let r = if self.norotate {
                    Matrix3::identity()
                } else {
                    rotation_from_cov(&cov.cov[frame])
                };
                let t = cy - r * cx;
                self.results.extend_from_slice(&[
                    r[(0, 0)] as f32,
                    r[(0, 1)] as f32,
                    r[(0, 2)] as f32,
                    r[(1, 0)] as f32,
                    r[(1, 1)] as f32,
                    r[(1, 2)] as f32,
                    r[(2, 0)] as f32,
                    r[(2, 1)] as f32,
                    r[(2, 2)] as f32,
                    t[0] as f32,
                    t[1] as f32,
                    t[2] as f32,
                ]);
                self.frames += 1;
            }
            return Ok(());
        }

        let n_atoms = chunk.n_atoms;
        let masses = &system.atoms.mass;
        let weights = if self.mass_weighted {
            Some(
                self.selection
                    .indices
                    .iter()
                    .map(|&idx| masses[idx as usize])
                    .collect::<Vec<f32>>(),
            )
        } else {
            None
        };

        for frame in 0..chunk.n_frames {
            let mut frame_sel = Vec::with_capacity(self.selection.indices.len());
            for &idx in self.selection.indices.iter() {
                frame_sel.push(chunk.coords[frame * n_atoms + idx as usize]);
            }

            let (r, cx, cy) = if self.norotate {
                let cx = centroid_weighted(&frame_sel, weights.as_deref());
                let cy = centroid_weighted(reference_sel, weights.as_deref());
                (Matrix3::identity(), cx, cy)
            } else {
                kabsch_rotation_weighted(&frame_sel, reference_sel, weights.as_deref())
            };

            let t = cy - r * cx;
            self.results.extend_from_slice(&[
                r[(0, 0)] as f32,
                r[(0, 1)] as f32,
                r[(0, 2)] as f32,
                r[(1, 0)] as f32,
                r[(1, 1)] as f32,
                r[(1, 2)] as f32,
                r[(2, 0)] as f32,
                r[(2, 1)] as f32,
                r[(2, 2)] as f32,
                t[0] as f32,
                t[1] as f32,
                t[2] as f32,
            ]);
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: 12,
        })
    }
}


impl Plan for SuperposePlan {
    fn name(&self) -> &'static str {
        "superpose"
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        let _ = device;
        self.results.clear();
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
        {
            self.gpu = None;
            if let (Device::Cuda(ctx), Some(reference)) = (device, self.reference_sel.as_ref()) {
                let masses = if self.mass_weighted {
                    Some(
                        self.selection
                            .indices
                            .iter()
                            .map(|&idx| system.atoms.mass[idx as usize])
                            .collect::<Vec<f32>>(),
                    )
                } else {
                    None
                };
                let selection = ctx.selection(&self.selection.indices, masses.as_deref())?;
                let reference_gpu = ctx.reference(&convert_coords(reference))?;
                self.gpu = Some(SuperposeGpuState {
                    selection,
                    reference: reference_gpu,
                });
            }
        }
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        let _ = device;
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
            if let Device::Cuda(ctx) = device {
                let masses = if self.mass_weighted {
                    Some(
                        self.selection
                            .indices
                            .iter()
                            .map(|&idx| system.atoms.mass[idx as usize])
                            .collect::<Vec<f32>>(),
                    )
                } else {
                    None
                };
                let selection = ctx.selection(&self.selection.indices, masses.as_deref())?;
                let reference_gpu =
                    ctx.reference(&convert_coords(self.reference_sel.as_ref().unwrap()))?;
                self.gpu = Some(SuperposeGpuState {
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
        if let (Device::Cuda(ctx), Some(gpu)) = (device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let cov = ctx.align_covariance(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.selection,
                &gpu.reference,
            )?;
            for frame in 0..chunk.n_frames {
                let wsum = cov.sum_w[frame] as f64;
                let mut cx = Vector3::zeros();
                let mut cy = Vector3::zeros();
                if wsum > 0.0 {
                    cx = Vector3::new(
                        cov.sum_x[frame][0] as f64 / wsum,
                        cov.sum_x[frame][1] as f64 / wsum,
                        cov.sum_x[frame][2] as f64 / wsum,
                    );
                    cy = Vector3::new(
                        cov.sum_y[frame][0] as f64 / wsum,
                        cov.sum_y[frame][1] as f64 / wsum,
                        cov.sum_y[frame][2] as f64 / wsum,
                    );
                }
                let r = if self.norotate {
                    Matrix3::identity()
                } else {
                    rotation_from_cov(&cov.cov[frame])
                };
                let t = cy - r * cx;
                let base = frame * chunk.n_atoms;
                for atom in 0..chunk.n_atoms {
                    let p = chunk.coords[base + atom];
                    let x = p[0] as f64;
                    let y = p[1] as f64;
                    let z = p[2] as f64;
                    let rx = r[(0, 0)] * x + r[(0, 1)] * y + r[(0, 2)] * z + t[0];
                    let ry = r[(1, 0)] * x + r[(1, 1)] * y + r[(1, 2)] * z + t[1];
                    let rz = r[(2, 0)] * x + r[(2, 1)] * y + r[(2, 2)] * z + t[2];
                    self.results.push(rx as f32);
                    self.results.push(ry as f32);
                    self.results.push(rz as f32);
                }
            }
            return Ok(());
        }

        let n_atoms = chunk.n_atoms;
        let masses = &system.atoms.mass;
        let weights = if self.mass_weighted {
            Some(
                self.selection
                    .indices
                    .iter()
                    .map(|&idx| masses[idx as usize])
                    .collect::<Vec<f32>>(),
            )
        } else {
            None
        };
        for frame in 0..chunk.n_frames {
            let mut frame_sel = Vec::with_capacity(self.selection.indices.len());
            for &idx in self.selection.indices.iter() {
                frame_sel.push(chunk.coords[frame * n_atoms + idx as usize]);
            }
            let (r, cx, cy) = if self.norotate {
                let cx = centroid_weighted(&frame_sel, weights.as_deref());
                let cy = centroid_weighted(reference_sel, weights.as_deref());
                (Matrix3::identity(), cx, cy)
            } else {
                kabsch_rotation_weighted(&frame_sel, reference_sel, weights.as_deref())
            };
            let t = cy - r * cx;
            for atom in 0..n_atoms {
                let p = chunk.coords[frame * n_atoms + atom];
                let x = p[0] as f64;
                let y = p[1] as f64;
                let z = p[2] as f64;
                let rx = r[(0, 0)] * x + r[(0, 1)] * y + r[(0, 2)] * z + t[0];
                let ry = r[(1, 0)] * x + r[(1, 1)] * y + r[(1, 2)] * z + t[1];
                let rz = r[(2, 0)] * x + r[(2, 1)] * y + r[(2, 2)] * z + t[2];
                self.results.push(rx as f32);
                self.results.push(ry as f32);
                self.results.push(rz as f32);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}
