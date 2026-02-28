#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuBufferF32, GpuGroups, GpuReference, GpuSelection};



pub struct PrincipalAxesPlan {
    selection: Selection,
    mass_weighted: bool,
    results: Vec<f32>,
    frames: usize,

    #[cfg(feature = "cuda")]
    gpu: Option<PrincipalAxesGpuState>,
}



#[cfg(feature = "cuda")]
struct PrincipalAxesGpuState {
    groups: GpuGroups,
    masses: GpuBufferF32,
    selection: GpuSelection,
}


impl PrincipalAxesPlan {

    pub fn new(selection: Selection, mass_weighted: bool) -> Self {
        Self {
            selection,
            mass_weighted,
            results: Vec::new(),
            frames: 0,

            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}


impl Plan for PrincipalAxesPlan {
    fn name(&self) -> &'static str {
        "principal_axes"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;

        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let offsets = vec![0u32, self.selection.indices.len() as u32];
                let indices = self.selection.indices.as_ref().clone();
                let max_len = self.selection.indices.len();
                let groups = ctx.groups(&offsets, &indices, max_len)?;
                let masses = if self.mass_weighted {
                    _system.atoms.mass.clone()
                } else {
                    vec![1.0f32; _system.n_atoms()]
                };
                let masses = ctx.upload_f32(&masses)?;
                let sel_masses = if self.mass_weighted {
                    Some(
                        self.selection
                            .indices
                            .iter()
                            .map(|&idx| _system.atoms.mass[idx as usize])
                            .collect::<Vec<f32>>(),
                    )
                } else {
                    None
                };
                let selection = ctx.selection(&self.selection.indices, sel_masses.as_deref())?;
                self.gpu = Some(PrincipalAxesGpuState {
                    groups,
                    masses,
                    selection,
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
            let centers = ctx.group_com(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.groups,
                &gpu.masses,
                1.0,
            )?;
            let inertia = ctx.inertia_tensor(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.selection,
                &centers,
            )?;
            for frame in 0..chunk.n_frames {
                let vals = inertia[frame];
                let (axes, evals) = principal_axes_from_inertia(
                    vals[0] as f64,
                    vals[1] as f64,
                    vals[2] as f64,
                    vals[3] as f64,
                    vals[4] as f64,
                    vals[5] as f64,
                );
                self.results.extend_from_slice(&[
                    axes[(0, 0)] as f32,
                    axes[(0, 1)] as f32,
                    axes[(0, 2)] as f32,
                    axes[(1, 0)] as f32,
                    axes[(1, 1)] as f32,
                    axes[(1, 2)] as f32,
                    axes[(2, 0)] as f32,
                    axes[(2, 1)] as f32,
                    axes[(2, 2)] as f32,
                    evals[0],
                    evals[1],
                    evals[2],
                ]);
                self.frames += 1;
            }
            return Ok(());
        }
        let masses = &system.atoms.mass;
        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let center = center_of_selection(
                chunk,
                frame,
                &self.selection.indices,
                masses,
                self.mass_weighted,
            );
            let mut i_xx = 0.0f64;
            let mut i_yy = 0.0f64;
            let mut i_zz = 0.0f64;
            let mut i_xy = 0.0f64;
            let mut i_xz = 0.0f64;
            let mut i_yz = 0.0f64;
            for &idx in self.selection.indices.iter() {
                let atom_idx = idx as usize;
                let p = chunk.coords[frame * n_atoms + atom_idx];
                let x = p[0] as f64 - center[0];
                let y = p[1] as f64 - center[1];
                let z = p[2] as f64 - center[2];
                let w = if self.mass_weighted {
                    masses[atom_idx] as f64
                } else {
                    1.0
                };
                i_xx += w * (y * y + z * z);
                i_yy += w * (x * x + z * z);
                i_zz += w * (x * x + y * y);
                i_xy -= w * x * y;
                i_xz -= w * x * z;
                i_yz -= w * y * z;
            }

            let (axes, vals) =
                principal_axes_from_inertia(i_xx, i_yy, i_zz, i_xy, i_xz, i_yz);

            self.results.extend_from_slice(&[
                axes[(0, 0)] as f32,
                axes[(0, 1)] as f32,
                axes[(0, 2)] as f32,
                axes[(1, 0)] as f32,
                axes[(1, 1)] as f32,
                axes[(1, 2)] as f32,
                axes[(2, 0)] as f32,
                axes[(2, 1)] as f32,
                axes[(2, 2)] as f32,
                vals[0],
                vals[1],
                vals[2],
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



pub struct AlignPlan {
    selection: Selection,
    reference_mode: ReferenceMode,
    mass_weighted: bool,
    norotate: bool,
    reference_sel: Option<Vec<[f32; 4]>>,
    results: Vec<f32>,
    frames: usize,

    #[cfg(feature = "cuda")]
    gpu: Option<AlignGpuState>,
}



#[cfg(feature = "cuda")]
struct AlignGpuState {
    selection: GpuSelection,
    reference: GpuReference,
}



pub struct SuperposePlan {
    selection: Selection,
    reference_mode: ReferenceMode,
    mass_weighted: bool,
    norotate: bool,
    reference_sel: Option<Vec<[f32; 4]>>,
    results: Vec<f32>,

    #[cfg(feature = "cuda")]
    gpu: Option<SuperposeGpuState>,
}
