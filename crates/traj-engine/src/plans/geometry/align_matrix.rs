use nalgebra::Vector3;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use super::align::AlignPlan;
use super::utils::*;
use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::ReferenceMode;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuBufferF32, GpuGroups, GpuSelection};

pub struct RotationMatrixPlan {
    inner: AlignPlan,
}

impl RotationMatrixPlan {
    pub fn new(selection: Selection, reference_mode: ReferenceMode, mass_weighted: bool) -> Self {
        Self {
            inner: AlignPlan::new(selection, reference_mode, mass_weighted, false),
        }
    }
}

impl Plan for RotationMatrixPlan {
    fn name(&self) -> &'static str {
        "rotation_matrix"
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.inner.init(system, device)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner.process_chunk(chunk, system, device)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let out = self.inner.finalize()?;
        match out {
            PlanOutput::Matrix { data, rows, cols } => {
                if cols != 12 {
                    return Err(TrajError::Mismatch(
                        "rotation_matrix expects 12-column align output".into(),
                    ));
                }
                let mut rot = Vec::with_capacity(rows * 9);
                for row in 0..rows {
                    let base = row * cols;
                    rot.extend_from_slice(&data[base..base + 9]);
                }
                Ok(PlanOutput::Matrix {
                    data: rot,
                    rows,
                    cols: 9,
                })
            }
            _ => Err(TrajError::Mismatch(
                "rotation_matrix expects align output".into(),
            )),
        }
    }
}

pub struct AlignPrincipalAxisPlan {
    selection: Selection,
    mass_weighted: bool,
    results: Vec<f32>,
    frames: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<AlignPrincipalGpuState>,
}

#[cfg(feature = "cuda")]
struct AlignPrincipalGpuState {
    groups: GpuGroups,
    masses: GpuBufferF32,
    selection: GpuSelection,
}

impl AlignPrincipalAxisPlan {
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

impl Plan for AlignPrincipalAxisPlan {
    fn name(&self) -> &'static str {
        "align_principal_axis"
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
                self.gpu = Some(AlignPrincipalGpuState {
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
                let (axes, _) = principal_axes_from_inertia(
                    vals[0] as f64,
                    vals[1] as f64,
                    vals[2] as f64,
                    vals[3] as f64,
                    vals[4] as f64,
                    vals[5] as f64,
                );
                let r = axes.transpose();
                let c = Vector3::new(
                    centers[frame].x as f64,
                    centers[frame].y as f64,
                    centers[frame].z as f64,
                );
                let t = -r * c;
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

            let (axes, _) = principal_axes_from_inertia(i_xx, i_yy, i_zz, i_xy, i_xz, i_yz);

            let r = axes.transpose();
            let c = Vector3::new(center[0], center[1], center[2]);
            let t = -r * c;
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
