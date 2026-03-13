use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::pbc_utils::{apply_pbc, apply_pbc_triclinic, cell_and_inv_from_box};
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
    temperature: f64,
    make_whole: bool,
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
            temperature: 300.0,
            make_whole: true,
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

    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn with_make_whole(mut self, make_whole: bool) -> Self {
        self.make_whole = make_whole;
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
        if !(self.temperature.is_finite() && self.temperature > 0.0) {
            return Err(TrajError::Mismatch(
                "dielectric requires a positive temperature".into(),
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
                if self.make_whole {
                    return Ok(());
                }
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

        let mut murot_mean = [0.0f64; 3];
        let mut murot_sq_mean = [0.0f64; 3];
        let mut mu_mean = [0.0f64; 3];
        let mut mu_sq_mean = [0.0f64; 3];
        let mut muavg = 0.0f64;
        let mut volume_sum = 0.0f64;
        let mut volume_count = 0usize;

        for frame in 0..n_frames {
            let mut murot = [0.0f64; 3];
            let mut mutrans = [0.0f64; 3];
            if frame >= self.boxes.len() {
                return Err(TrajError::Mismatch(
                    "dielectric requires box metadata for every frame".into(),
                ));
            }
            let box_ = self.boxes[frame];
            volume_sum += box_volume(box_, self.length_scale)?;
            volume_count += 1;
            for (g_idx, atoms) in groups.groups.iter().enumerate() {
                let dip = if self.make_whole {
                    group_dipole_whole(
                        &self.coords[frame * self.n_atoms..(frame + 1) * self.n_atoms],
                        atoms,
                        &self.charges,
                        box_,
                        self.length_scale,
                    )?
                } else {
                    dipoles[frame * n_groups + g_idx]
                };
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
            let mu = [
                murot[0] + mutrans[0],
                murot[1] + mutrans[1],
                murot[2] + mutrans[2],
            ];
            rot_sq.push(r2 as f32);
            trans_sq.push(t2 as f32);
            rot_trans.push(rt as f32);
            for k in 0..3 {
                murot_mean[k] += murot[k];
                murot_sq_mean[k] += murot[k] * murot[k];
                mu_mean[k] += mu[k];
                mu_sq_mean[k] += mu[k] * mu[k];
            }
        }

        let n_frames_f = n_frames as f64;
        for k in 0..3 {
            murot_mean[k] /= n_frames_f;
            murot_sq_mean[k] /= n_frames_f;
            mu_mean[k] /= n_frames_f;
            mu_sq_mean[k] /= n_frames_f;
        }
        muavg /= n_frames_f * denom;
        let avg_volume = volume_sum / volume_count.max(1) as f64;
        if !(avg_volume.is_finite() && avg_volume > 0.0) {
            return Err(TrajError::Mismatch(
                "dielectric requires a positive periodic box volume".into(),
            ));
        }

        let fluct_rot = (0..3)
            .map(|k| murot_sq_mean[k] - murot_mean[k] * murot_mean[k])
            .sum::<f64>();
        let fluct_total = (0..3)
            .map(|k| mu_sq_mean[k] - mu_mean[k] * mu_mean[k])
            .sum::<f64>();

        let elementary_charge = 1.602_176_634e-19_f64;
        let angstrom_to_m = 1.0e-10_f64;
        let angstrom3_to_m3 = 1.0e-30_f64;
        let eps0 = 8.854_187_812_8e-12_f64;
        let kb = 1.380_649e-23_f64;
        let dipole_scale = elementary_charge * angstrom_to_m;
        let fluct_scale = dipole_scale * dipole_scale;
        let denom_eps = 3.0 * eps0 * kb * self.temperature * (avg_volume * angstrom3_to_m3);
        let dielectric_rot = (1.0 + fluct_rot * fluct_scale / denom_eps) as f32;
        let dielectric_total = (1.0 + fluct_total * fluct_scale / denom_eps) as f32;
        let debye_per_ea = (elementary_charge * angstrom_to_m) / 3.33564e-30_f64;
        let mu_avg = (muavg * debye_per_ea) as f32;

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

fn box_volume(box_: Box3, length_scale: f64) -> TrajResult<f64> {
    let raw = match box_ {
        Box3::Orthorhombic { lx, ly, lz } => (lx as f64) * (ly as f64) * (lz as f64),
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
            (m0 * (m4 * m8 - m5 * m7) - m1 * (m3 * m8 - m5 * m6) + m2 * (m3 * m7 - m4 * m6)).abs()
        }
        Box3::None => {
            return Err(TrajError::Mismatch(
                "dielectric requires orthorhombic or triclinic box metadata".into(),
            ))
        }
    };
    Ok(raw * length_scale.powi(3))
}

fn group_dipole_whole(
    frame_coords: &[[f32; 4]],
    atoms: &[usize],
    charges: &[f64],
    box_: Box3,
    length_scale: f64,
) -> TrajResult<[f64; 3]> {
    if atoms.is_empty() {
        return Ok([0.0, 0.0, 0.0]);
    }
    let anchor = frame_coords[atoms[0]];
    let anchor_pos = [
        (anchor[0] as f64) * length_scale,
        (anchor[1] as f64) * length_scale,
        (anchor[2] as f64) * length_scale,
    ];
    let maybe_triclinic = match box_ {
        Box3::Triclinic { .. } => Some(cell_and_inv_from_box(box_)?),
        _ => None,
    };
    let mut dip = [0.0f64; 3];
    for &atom_idx in atoms {
        let pos = frame_coords[atom_idx];
        let mut dx = ((pos[0] - anchor[0]) as f64) * length_scale;
        let mut dy = ((pos[1] - anchor[1]) as f64) * length_scale;
        let mut dz = ((pos[2] - anchor[2]) as f64) * length_scale;
        match box_ {
            Box3::Orthorhombic { lx, ly, lz } => {
                apply_pbc(
                    &mut dx,
                    &mut dy,
                    &mut dz,
                    (lx as f64) * length_scale,
                    (ly as f64) * length_scale,
                    (lz as f64) * length_scale,
                );
            }
            Box3::Triclinic { .. } => {
                let (cell, inv) = maybe_triclinic
                    .as_ref()
                    .expect("triclinic box conversion must be available");
                apply_pbc_triclinic(&mut dx, &mut dy, &mut dz, cell, inv);
            }
            Box3::None => {}
        }
        let q = charges[atom_idx];
        dip[0] += q * (anchor_pos[0] + dx);
        dip[1] += q * (anchor_pos[1] + dy);
        dip[2] += q * (anchor_pos[2] + dz);
    }
    Ok(dip)
}
