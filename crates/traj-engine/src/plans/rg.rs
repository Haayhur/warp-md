use traj_core::error::TrajResult;
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput, PlanRequirements};

#[cfg(feature = "cuda")]
use traj_gpu::GpuSelection;

pub struct RgPlan {
    selection: Selection,
    mass_weighted: bool,
    selected_masses: Vec<f32>,
    use_selected_input: bool,
    results: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<RgGpuState>,
}

pub struct RadgyrTensorPlan {
    selection: Selection,
    mass_weighted: bool,
    results: Vec<f32>,
    frames: usize,
}

#[cfg(feature = "cuda")]
struct RgGpuState {
    selection: GpuSelection,
}

impl RgPlan {
    pub fn new(selection: Selection, mass_weighted: bool) -> Self {
        Self {
            selection,
            mass_weighted,
            selected_masses: Vec::new(),
            use_selected_input: false,
            results: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl RadgyrTensorPlan {
    pub fn new(selection: Selection, mass_weighted: bool) -> Self {
        Self {
            selection,
            mass_weighted,
            results: Vec::new(),
            frames: 0,
        }
    }
}

impl Plan for RgPlan {
    fn name(&self) -> &'static str {
        "rg"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        let _ = device;
        self.results.clear();
        self.selected_masses.clear();
        self.use_selected_input = true;
        if self.mass_weighted {
            self.selected_masses.extend(
                self.selection
                    .indices
                    .iter()
                    .map(|idx| system.atoms.mass[*idx as usize]),
            );
        }
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = device {
                let masses = if self.mass_weighted {
                    Some(self.selected_masses.clone())
                } else {
                    None
                };
                let dense_selection: Vec<u32> = (0..self.selection.indices.len())
                    .map(|i| i as u32)
                    .collect();
                let selection = if self.use_selected_input {
                    ctx.selection(&dense_selection, masses.as_deref())?
                } else {
                    ctx.selection(&self.selection.indices, masses.as_deref())?
                };
                self.gpu = Some(RgGpuState { selection });
            }
        }
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(&self.selection.indices)
        } else {
            None
        }
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        let _ = device;
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (device, &self.gpu) {
            let results = ctx.rg_f32x4(
                &chunk.coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.selection,
                self.mass_weighted,
            )?;
            self.results.extend(results);
            return Ok(());
        }
        self.compute_rg_from_full_chunk(chunk, system);
        Ok(())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        _source_selection: &[u32],
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        let _ = (system, device);
        #[cfg(feature = "cuda")]
        if matches!(device, Device::Cuda(_)) {
            return self.process_chunk(chunk, system, device);
        }
        self.compute_rg_from_selected_chunk(chunk);
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

impl RgPlan {
    fn compute_rg_from_full_chunk(&mut self, chunk: &FrameChunk, system: &System) {
        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        for frame in 0..chunk.n_frames {
            let mut com = [0.0f32; 3];
            let mut mass_sum = 0.0f32;
            let frame_base = frame * n_atoms;
            for &idx in sel.iter() {
                let atom_idx = idx as usize;
                let mass = if self.mass_weighted {
                    system.atoms.mass[atom_idx]
                } else {
                    1.0
                };
                let pos = chunk.coords[frame_base + atom_idx];
                com[0] += pos[0] * mass;
                com[1] += pos[1] * mass;
                com[2] += pos[2] * mass;
                mass_sum += mass;
            }
            if mass_sum == 0.0 {
                self.results.push(0.0);
                continue;
            }
            com[0] /= mass_sum;
            com[1] /= mass_sum;
            com[2] /= mass_sum;
            let mut sum = 0.0f32;
            for &idx in sel.iter() {
                let atom_idx = idx as usize;
                let mass = if self.mass_weighted {
                    system.atoms.mass[atom_idx]
                } else {
                    1.0
                };
                let pos = chunk.coords[frame_base + atom_idx];
                let dx = pos[0] - com[0];
                let dy = pos[1] - com[1];
                let dz = pos[2] - com[2];
                sum += mass * (dx * dx + dy * dy + dz * dz);
            }
            self.results.push((sum / mass_sum).sqrt());
        }
    }

    fn compute_rg_from_selected_chunk(&mut self, chunk: &FrameChunk) {
        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let mut com = [0.0f32; 3];
            let mut mass_sum = 0.0f32;
            let frame_base = frame * n_atoms;
            let frame_coords = &chunk.coords[frame_base..frame_base + n_atoms];
            if self.mass_weighted {
                for (pos, &mass) in frame_coords.iter().zip(self.selected_masses.iter()) {
                    com[0] += pos[0] * mass;
                    com[1] += pos[1] * mass;
                    com[2] += pos[2] * mass;
                    mass_sum += mass;
                }
            } else {
                for pos in frame_coords {
                    com[0] += pos[0];
                    com[1] += pos[1];
                    com[2] += pos[2];
                }
                mass_sum = n_atoms as f32;
            }

            if mass_sum == 0.0 {
                self.results.push(0.0);
                continue;
            }
            com[0] /= mass_sum;
            com[1] /= mass_sum;
            com[2] /= mass_sum;

            let mut sum = 0.0f32;
            if self.mass_weighted {
                for (pos, &mass) in frame_coords.iter().zip(self.selected_masses.iter()) {
                    let dx = pos[0] - com[0];
                    let dy = pos[1] - com[1];
                    let dz = pos[2] - com[2];
                    sum += mass * (dx * dx + dy * dy + dz * dz);
                }
            } else {
                for pos in frame_coords {
                    let dx = pos[0] - com[0];
                    let dy = pos[1] - com[1];
                    let dz = pos[2] - com[2];
                    sum += dx * dx + dy * dy + dz * dz;
                }
            }
            self.results.push((sum / mass_sum).sqrt());
        }
    }
}

impl Plan for RadgyrTensorPlan {
    fn name(&self) -> &'static str {
        "radgyr_tensor"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        for frame in 0..chunk.n_frames {
            let mut com = [0.0f64; 3];
            let mut mass_sum = 0.0f64;
            for &idx in sel.iter() {
                let atom_idx = idx as usize;
                let mass = if self.mass_weighted {
                    system.atoms.mass[atom_idx] as f64
                } else {
                    1.0
                };
                let pos = chunk.coords[frame * n_atoms + atom_idx];
                com[0] += pos[0] as f64 * mass;
                com[1] += pos[1] as f64 * mass;
                com[2] += pos[2] as f64 * mass;
                mass_sum += mass;
            }
            if mass_sum == 0.0 {
                self.results.extend_from_slice(&[0.0; 7]);
                self.frames += 1;
                continue;
            }
            com[0] /= mass_sum;
            com[1] /= mass_sum;
            com[2] /= mass_sum;
            let mut g_xx = 0.0f64;
            let mut g_yy = 0.0f64;
            let mut g_zz = 0.0f64;
            let mut g_xy = 0.0f64;
            let mut g_xz = 0.0f64;
            let mut g_yz = 0.0f64;
            for &idx in sel.iter() {
                let atom_idx = idx as usize;
                let mass = if self.mass_weighted {
                    system.atoms.mass[atom_idx] as f64
                } else {
                    1.0
                };
                let pos = chunk.coords[frame * n_atoms + atom_idx];
                let dx = pos[0] as f64 - com[0];
                let dy = pos[1] as f64 - com[1];
                let dz = pos[2] as f64 - com[2];
                g_xx += mass * dx * dx;
                g_yy += mass * dy * dy;
                g_zz += mass * dz * dz;
                g_xy += mass * dx * dy;
                g_xz += mass * dx * dz;
                g_yz += mass * dy * dz;
            }
            g_xx /= mass_sum;
            g_yy /= mass_sum;
            g_zz /= mass_sum;
            g_xy /= mass_sum;
            g_xz /= mass_sum;
            g_yz /= mass_sum;
            let rg = (g_xx + g_yy + g_zz).sqrt() as f32;
            self.results.extend_from_slice(&[
                rg,
                g_xx as f32,
                g_yy as f32,
                g_zz as f32,
                g_xy as f32,
                g_xz as f32,
                g_yz as f32,
            ]);
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: 7,
        })
    }
}
