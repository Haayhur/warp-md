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

pub struct DipoleAlignmentPlan {
    selection: Selection,
    group_by: GroupBy,
    group_types: Option<Vec<usize>>,
    charges: Vec<f64>,
    length_scale: f64,
    groups: Option<GroupMap>,
    coords: Vec<[f32; 4]>,
    times: Vec<f64>,
    n_atoms: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<DipoleGpuState>,
}

#[cfg(feature = "cuda")]
struct DipoleGpuState {
    ctx: GpuContext,
    groups: GpuGroups,
    charges: GpuBufferF32,
}

impl DipoleAlignmentPlan {
    pub fn new(selection: Selection, group_by: GroupBy, charges: Vec<f64>) -> Self {
        Self {
            selection,
            group_by,
            group_types: None,
            charges,
            length_scale: 1.0,
            groups: None,
            coords: Vec::new(),
            times: Vec::new(),
            n_atoms: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }

    pub fn with_group_types(mut self, types: Vec<usize>) -> Self {
        self.group_types = Some(types);
        self
    }
}

impl Plan for DipoleAlignmentPlan {
    fn name(&self) -> &'static str {
        "dipole_alignment"
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.coords.clear();
        self.times.clear();
        self.n_atoms = system.n_atoms();
        if self.charges.len() != self.n_atoms {
            return Err(TrajError::Mismatch(
                "charges length does not match atom count".into(),
            ));
        }
        let mut spec = GroupSpec::new(self.selection.clone(), self.group_by);
        if let Some(types) = &self.group_types {
            spec = spec.with_group_types(types.clone());
        }
        self.groups = Some(spec.build(system)?);
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = device {
                let (offsets, indices, max_len) =
                    groups_to_csr(&self.groups.as_ref().unwrap().groups);
                let groups = ctx.groups(&offsets, &indices, max_len)?;
                let charges: Vec<f32> = self.charges.iter().map(|c| *c as f32).collect();
                let charges = ctx.upload_f32(&charges)?;
                self.gpu = Some(DipoleGpuState {
                    ctx: ctx.clone(),
                    groups,
                    charges,
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
        let cols = 6 * (n_types + 1);
        let mut data = vec![0.0f32; n_frames * cols];
        let mut time = Vec::with_capacity(n_frames);

        let mut dipoles = Vec::new();
        let use_gpu = {
            #[cfg(feature = "cuda")]
            {
                if let Some(gpu) = &self.gpu {
                    let coords = convert_coords(&self.coords);
                    let dip_gpu = gpu.ctx.group_dipole(
                        &coords,
                        self.n_atoms,
                        n_frames,
                        &gpu.groups,
                        &gpu.charges,
                        self.length_scale as f32,
                    )?;
                    dipoles = dip_gpu
                        .iter()
                        .map(|v| [v.x as f64, v.y as f64, v.z as f64])
                        .collect();
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
            dipoles = vec![[0.0f64; 3]; n_frames * n_groups];
            for frame in 0..n_frames {
                let frame_offset = frame * self.n_atoms;
                for (g_idx, atoms) in groups.groups.iter().enumerate() {
                    let mut dip = [0.0f64; 3];
                    for &atom_idx in atoms {
                        let p = self.coords[frame_offset + atom_idx];
                        let q = self.charges[atom_idx];
                        dip[0] += (p[0] as f64) * q;
                        dip[1] += (p[1] as f64) * q;
                        dip[2] += (p[2] as f64) * q;
                    }
                    dip[0] *= self.length_scale;
                    dip[1] *= self.length_scale;
                    dip[2] *= self.length_scale;
                    dipoles[frame * n_groups + g_idx] = dip;
                }
            }
        }

        for frame in 0..n_frames {
            let mut cos_sum = vec![[0.0f64; 3]; n_types + 1];
            let mut cos2_sum = vec![[0.0f64; 3]; n_types + 1];
            for g_idx in 0..n_groups {
                let dip = dipoles[frame * n_groups + g_idx];
                let norm = (dip[0] * dip[0] + dip[1] * dip[1] + dip[2] * dip[2]).sqrt();
                let vec = if norm > 0.0 {
                    [dip[0] / norm, dip[1] / norm, dip[2] / norm]
                } else {
                    [0.0, 0.0, 0.0]
                };
                let t = type_ids[g_idx];
                cos_sum[t][0] += vec[0];
                cos_sum[t][1] += vec[1];
                cos_sum[t][2] += vec[2];
                cos2_sum[t][0] += vec[0] * vec[0];
                cos2_sum[t][1] += vec[1] * vec[1];
                cos2_sum[t][2] += vec[2] * vec[2];
                cos_sum[n_types][0] += vec[0];
                cos_sum[n_types][1] += vec[1];
                cos_sum[n_types][2] += vec[2];
                cos2_sum[n_types][0] += vec[0] * vec[0];
                cos2_sum[n_types][1] += vec[1] * vec[1];
                cos2_sum[n_types][2] += vec[2] * vec[2];
            }
            for t in 0..n_types {
                let inv = 1.0 / (type_counts[t] as f64);
                for k in 0..3 {
                    cos_sum[t][k] *= inv;
                    cos2_sum[t][k] *= inv;
                }
            }
            let inv_total = 1.0 / (n_groups as f64);
            for k in 0..3 {
                cos_sum[n_types][k] *= inv_total;
                cos2_sum[n_types][k] *= inv_total;
            }
            let row = frame * cols;
            for t in 0..(n_types + 1) {
                let base = row + t * 3;
                data[base] = cos_sum[t][0] as f32;
                data[base + 1] = cos_sum[t][1] as f32;
                data[base + 2] = cos_sum[t][2] as f32;
                let base2 = row + (n_types + 1) * 3 + t * 3;
                data[base2] = cos2_sum[t][0] as f32;
                data[base2 + 1] = cos2_sum[t][1] as f32;
                data[base2 + 2] = cos2_sum[t][2] as f32;
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
