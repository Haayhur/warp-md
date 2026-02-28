use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::analysis::common::compute_group_com;
#[cfg(feature = "cuda")]
use crate::plans::analysis::common::groups_to_csr;
use crate::plans::analysis::grouping::{GroupBy, GroupMap, GroupSpec};

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuBufferF32, GpuContext, GpuGroups};

pub struct DielectricPlan {
    selection: Selection,
    group_by: GroupBy,
    group_types: Option<Vec<usize>>,
    charges: Vec<f64>,
    length_scale: f64,
    groups: Option<GroupMap>,
    masses: Vec<f64>,
    coords: Vec<[f32; 4]>,
    boxes: Vec<Box3>,
    times: Vec<f64>,
    n_atoms: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<DielectricGpuState>,
}

#[cfg(feature = "cuda")]
struct DielectricGpuState {
    ctx: GpuContext,
    groups: GpuGroups,
    masses: GpuBufferF32,
    charges: GpuBufferF32,
}

impl DielectricPlan {
    pub fn new(selection: Selection, group_by: GroupBy, charges: Vec<f64>) -> Self {
        Self {
            selection,
            group_by,
            group_types: None,
            charges,
            length_scale: 1.0,
            groups: None,
            masses: Vec::new(),
            coords: Vec::new(),
            boxes: Vec::new(),
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

impl Plan for DielectricPlan {
    fn name(&self) -> &'static str {
        "dielectric"
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.coords.clear();
        self.boxes.clear();
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
        self.masses = system.atoms.mass.iter().map(|m| *m as f64).collect();
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = device {
                let (offsets, indices, max_len) =
                    groups_to_csr(&self.groups.as_ref().unwrap().groups);
                let groups = ctx.groups(&offsets, &indices, max_len)?;
                let masses: Vec<f32> = system.atoms.mass.iter().map(|m| *m).collect();
                let charges: Vec<f32> = self.charges.iter().map(|c| *c as f32).collect();
                let masses = ctx.upload_f32(&masses)?;
                let charges = ctx.upload_f32(&charges)?;
                self.gpu = Some(DielectricGpuState {
                    ctx: ctx.clone(),
                    groups,
                    masses,
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
        self.boxes.extend_from_slice(&chunk.box_);
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
            return Ok(PlanOutput::Dielectric(crate::executor::DielectricOutput {
                time: Vec::new(),
                rot_sq: Vec::new(),
                trans_sq: Vec::new(),
                rot_trans: Vec::new(),
                dielectric_rot: 0.0,
                dielectric_total: 0.0,
                mu_avg: 0.0,
            }));
        }
        let n_groups = groups.n_groups();
        let type_counts = groups.type_counts();
        let denom = if !type_counts.is_empty() {
            type_counts[0].max(1) as f64
        } else {
            n_groups.max(1) as f64
        };

        let mut com = Vec::new();
        let mut dipoles = Vec::new();
        let use_gpu = {
            #[cfg(feature = "cuda")]
            {
                if let Some(gpu) = &self.gpu {
                    let coords = convert_coords(&self.coords);
                    let com_gpu = gpu.ctx.group_com(
                        &coords,
                        self.n_atoms,
                        n_frames,
                        &gpu.groups,
                        &gpu.masses,
                        self.length_scale as f32,
                    )?;
                    com = vec![[0.0f64; 3]; com_gpu.len()];
                    for (idx, v) in com_gpu.iter().enumerate() {
                        com[idx][0] = v.x as f64;
                        com[idx][1] = v.y as f64;
                        com[idx][2] = v.z as f64;
                    }
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
            com = compute_group_com(
                &self.coords,
                self.n_atoms,
                groups,
                &self.masses,
                self.length_scale,
            );
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
        let mut rot_sq = Vec::with_capacity(n_frames);
        let mut trans_sq = Vec::with_capacity(n_frames);
        let mut rot_trans = Vec::with_capacity(n_frames);

        let mut murot_avg = 0.0f64;
        let mut murottrans_avg = 0.0f64;
        let mut muavg = 0.0f64;

        for frame in 0..n_frames {
            let mut murot = [0.0f64; 3];
            let mut mutrans = [0.0f64; 3];
            for (g_idx, atoms) in groups.groups.iter().enumerate() {
                let dip = dipoles[frame * n_groups + g_idx];
                let mut mol_charge = 0.0f64;
                for &atom_idx in atoms {
                    mol_charge += self.charges[atom_idx];
                }
                let com_idx = frame * n_groups + g_idx;
                let trans = [
                    com[com_idx][0] * mol_charge,
                    com[com_idx][1] * mol_charge,
                    com[com_idx][2] * mol_charge,
                ];
                let rot = [dip[0] - trans[0], dip[1] - trans[1], dip[2] - trans[2]];
                muavg += (rot[0] * rot[0] + rot[1] * rot[1] + rot[2] * rot[2]).sqrt();
                murot[0] += rot[0];
                murot[1] += rot[1];
                murot[2] += rot[2];
                mutrans[0] += trans[0];
                mutrans[1] += trans[1];
                mutrans[2] += trans[2];
            }
            let r2 = murot[0] * murot[0] + murot[1] * murot[1] + murot[2] * murot[2];
            let t2 = mutrans[0] * mutrans[0] + mutrans[1] * mutrans[1] + mutrans[2] * mutrans[2];
            let rt = murot[0] * mutrans[0] + murot[1] * mutrans[1] + murot[2] * mutrans[2];
            rot_sq.push(r2 as f32);
            trans_sq.push(t2 as f32);
            rot_trans.push(rt as f32);
            murot_avg += r2;
            murottrans_avg += rt;
        }

        let n_frames_f = n_frames as f64;
        murot_avg /= n_frames_f;
        murottrans_avg /= n_frames_f;
        muavg /= n_frames_f * denom;

        let debye = 3.33564e-30_f64;
        let eps0 = 8.85418781762e-12_f64;
        let kb = 1.380648813e-23_f64;
        let enm = 0.020819434_f64;
        let multiple = debye * debye / (enm * enm * eps0 * kb);

        let murot_scaled = murot_avg * multiple;
        let murottrans_scaled = murottrans_avg * multiple;
        let dielectric_rot = (murot_scaled + 1.0) as f32;
        let dielectric_total = (1.0 + murot_scaled + murottrans_scaled) as f32;
        let mu_avg = (muavg / enm) as f32;

        let time: Vec<f32> = self.times.iter().map(|t| *t as f32).collect();
        Ok(PlanOutput::Dielectric(crate::executor::DielectricOutput {
            time,
            rot_sq,
            trans_sq,
            rot_trans,
            dielectric_rot,
            dielectric_total,
            mu_avg,
        }))
    }
}
