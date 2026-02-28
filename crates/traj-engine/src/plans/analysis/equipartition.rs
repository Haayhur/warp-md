use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};
#[cfg(feature = "cuda")]
use crate::plans::analysis::common::groups_to_csr;
use crate::plans::analysis::grouping::{GroupBy, GroupMap, GroupSpec};

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuBufferF32, GpuContext, GpuGroups};

pub struct EquipartitionPlan {
    selection: Selection,
    group_by: GroupBy,
    group_types: Option<Vec<usize>>,
    velocity_scale: f64,
    length_scale: f64,
    groups: Option<GroupMap>,
    masses: Vec<f64>,
    coords: Vec<[f32; 4]>,
    times: Vec<f64>,
    n_atoms: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<EquipartitionGpuState>,
}

#[cfg(feature = "cuda")]
struct EquipartitionGpuState {
    ctx: GpuContext,
    groups: GpuGroups,
    masses: GpuBufferF32,
}

impl EquipartitionPlan {
    pub fn new(selection: Selection, group_by: GroupBy) -> Self {
        Self {
            selection,
            group_by,
            group_types: None,
            velocity_scale: 1.0,
            length_scale: 1.0,
            groups: None,
            masses: Vec::new(),
            coords: Vec::new(),
            times: Vec::new(),
            n_atoms: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_group_types(mut self, types: Vec<usize>) -> Self {
        self.group_types = Some(types);
        self
    }

    pub fn with_velocity_scale(mut self, scale: f64) -> Self {
        self.velocity_scale = scale;
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for EquipartitionPlan {
    fn name(&self) -> &'static str {
        "equipartition"
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.coords.clear();
        self.times.clear();
        self.n_atoms = system.n_atoms();
        let mut spec = GroupSpec::new(self.selection.clone(), self.group_by);
        if let Some(types) = &self.group_types {
            spec = spec.with_group_types(types.clone());
        }
        self.groups = Some(spec.build(system)?);
        self.masses = system.atoms.mass.iter().map(|m| *m as f64).collect();
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = device {
                let (offsets, indices, max_len) =
                    groups_to_csr(&self.groups.as_ref().unwrap().groups);
                let groups = ctx.groups(&offsets, &indices, max_len)?;
                let masses: Vec<f32> = system.atoms.mass.iter().map(|m| *m).collect();
                let masses = ctx.upload_f32(&masses)?;
                self.gpu = Some(EquipartitionGpuState {
                    ctx: ctx.clone(),
                    groups,
                    masses,
                });
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = device;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        self.coords.extend_from_slice(&chunk.coords);
        let base = self.times.len();
        if let Some(times) = &chunk.time_ps {
            for &t in times {
                self.times.push(t as f64);
            }
        } else {
            for i in 0..chunk.n_frames {
                self.times.push((base + i) as f64);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let groups = self
            .groups
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("groups not initialized".into()))?;
        let n_frames = self.times.len();
        if n_frames == 0 {
            return Ok(PlanOutput::TimeSeries {
                time: Vec::new(),
                data: Vec::new(),
                rows: 0,
                cols: 0,
            });
        }
        let n_groups = groups.n_groups();
        let type_ids = groups.type_ids();
        let type_counts = groups.type_counts();
        let n_types = type_counts.len();
        let cols = n_types + 1;

        let mut data = vec![0.0f32; n_frames * cols];
        let mut time = Vec::with_capacity(n_frames);
        let kb = 1.380648813e-23_f64;
        let amu_to_kg = 1.66053906660e-27_f64;

        let mut ke_per_group = Vec::new();
        let use_gpu = {
            #[cfg(feature = "cuda")]
            {
                if let Some(gpu) = &self.gpu {
                    let coords = convert_coords(&self.coords);
                    let scale = (self.velocity_scale * self.length_scale) as f32;
                    let ke_gpu = gpu.ctx.group_ke(
                        &coords,
                        self.n_atoms,
                        n_frames,
                        &gpu.groups,
                        &gpu.masses,
                        scale,
                    )?;
                    ke_per_group = ke_gpu.iter().map(|v| *v as f64).collect();
                    true
                } else {
                    false
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        if !use_gpu {
            ke_per_group = vec![0.0f64; n_frames * n_groups];
            let scale = self.velocity_scale * self.length_scale;
            for frame in 0..n_frames {
                let frame_offset = frame * self.n_atoms;
                for (g_idx, atoms) in groups.groups.iter().enumerate() {
                    let mut ke = 0.0f64;
                    for &atom_idx in atoms {
                        let v = self.coords[frame_offset + atom_idx];
                        let vx = v[0] as f64 * scale;
                        let vy = v[1] as f64 * scale;
                        let vz = v[2] as f64 * scale;
                        let m = self.masses[atom_idx] * amu_to_kg;
                        ke += 0.5 * m * (vx * vx + vy * vy + vz * vz);
                    }
                    ke_per_group[frame * n_groups + g_idx] = ke;
                }
            }
        }

        for frame in 0..n_frames {
            let mut temp_sum = vec![0.0f64; n_types + 1];
            for (g_idx, atoms) in groups.groups.iter().enumerate() {
                let ke = ke_per_group[frame * n_groups + g_idx];
                let dof = (atoms.len() as f64) * 3.0;
                let temp = if dof > 0.0 {
                    2.0 * ke / (kb * dof)
                } else {
                    0.0
                };
                let t = type_ids[g_idx];
                temp_sum[t] += temp;
                temp_sum[n_types] += temp;
            }
            for t in 0..n_types {
                let denom = type_counts[t].max(1) as f64;
                temp_sum[t] /= denom;
            }
            temp_sum[n_types] /= n_groups.max(1) as f64;
            let row = frame * cols;
            for t in 0..cols {
                data[row + t] = temp_sum[t] as f32;
            }
            time.push(self.times[frame] as f32);
        }

        Ok(PlanOutput::TimeSeries {
            time,
            data,
            rows: n_frames,
            cols,
        })
    }
}
