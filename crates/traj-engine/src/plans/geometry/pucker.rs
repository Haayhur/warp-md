use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use super::utils::*;
use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::PbcMode;
use nalgebra::{Matrix3, SymmetricEigen};

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, Float4, GpuBufferF32, GpuBufferU32, GpuGroups, GpuSelection};

pub struct PuckerPlan {
    selection: Selection,
    metric: PuckerMetric,
    results: Vec<f32>,
    return_phase: bool,
    phases: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<PuckerGpuState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PuckerMetric {
    MaxRadius,
    Amplitude,
}

#[cfg(feature = "cuda")]
struct PuckerGpuState {
    sel: GpuSelection,
    groups: GpuGroups,
    masses: GpuBufferF32,
}

impl PuckerPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            metric: PuckerMetric::MaxRadius,
            results: Vec::new(),
            return_phase: false,
            phases: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_metric(mut self, metric: PuckerMetric) -> Self {
        self.metric = metric;
        self
    }

    pub fn with_return_phase(mut self, return_phase: bool) -> Self {
        self.return_phase = return_phase;
        self
    }

    pub fn return_phase(&self) -> bool {
        self.return_phase
    }
}

impl Plan for PuckerPlan {
    fn name(&self) -> &'static str {
        "pucker"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.phases.clear();
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                if !self.selection.indices.is_empty() {
                    let offsets = vec![0u32, self.selection.indices.len() as u32];
                    let indices = self.selection.indices.clone();
                    let groups = ctx.groups(&offsets, &indices, indices.len())?;
                    let masses = vec![1.0f32; _system.n_atoms()];
                    let masses = ctx.upload_f32(&masses)?;
                    let sel = ctx.selection(&self.selection.indices, None)?;
                    self.gpu = Some(PuckerGpuState {
                        sel,
                        groups,
                        masses,
                    });
                }
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
        if self.metric == PuckerMetric::MaxRadius && !self.return_phase {
            if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
                if self.selection.indices.is_empty() {
                    self.results
                        .extend(std::iter::repeat(0.0).take(chunk.n_frames));
                    return Ok(());
                }
                let coords = convert_coords(&chunk.coords);
                let coms = ctx.group_com(
                    &coords,
                    chunk.n_atoms,
                    chunk.n_frames,
                    &gpu.groups,
                    &gpu.masses,
                    1.0,
                )?;
                let maxes =
                    ctx.max_dist_points(&coords, chunk.n_atoms, chunk.n_frames, &gpu.sel, &coms)?;
                self.results.extend(maxes);
                return Ok(());
            }
        }

        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        for frame in 0..chunk.n_frames {
            if sel.is_empty() {
                self.results.push(0.0);
                if self.return_phase {
                    self.phases.push(0.0);
                }
                continue;
            }
            let mut sum = [0.0f64; 3];
            for &idx in sel.iter() {
                let p = chunk.coords[frame * n_atoms + idx as usize];
                sum[0] += p[0] as f64;
                sum[1] += p[1] as f64;
                sum[2] += p[2] as f64;
            }
            let n = sel.len() as f64;
            let center = [sum[0] / n, sum[1] / n, sum[2] / n];
            let need_cov = self.metric == PuckerMetric::Amplitude || self.return_phase;
            let mut max_r = 0.0f64;
            let mut xx = 0.0f64;
            let mut xy = 0.0f64;
            let mut xz = 0.0f64;
            let mut yy = 0.0f64;
            let mut yz = 0.0f64;
            let mut zz = 0.0f64;
            for &idx in sel.iter() {
                let p = chunk.coords[frame * n_atoms + idx as usize];
                let dx = p[0] as f64 - center[0];
                let dy = p[1] as f64 - center[1];
                let dz = p[2] as f64 - center[2];
                if self.metric == PuckerMetric::MaxRadius {
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    if r > max_r {
                        max_r = r;
                    }
                }
                if need_cov {
                    xx += dx * dx;
                    xy += dx * dy;
                    xz += dx * dz;
                    yy += dy * dy;
                    yz += dy * dz;
                    zz += dz * dz;
                }
            }

            let eig = if need_cov {
                let inv_n = if n > 0.0 { 1.0 / n } else { 0.0 };
                let cov = Matrix3::new(
                    xx * inv_n,
                    xy * inv_n,
                    xz * inv_n,
                    xy * inv_n,
                    yy * inv_n,
                    yz * inv_n,
                    xz * inv_n,
                    yz * inv_n,
                    zz * inv_n,
                );
                Some(SymmetricEigen::new(cov))
            } else {
                None
            };

            match self.metric {
                PuckerMetric::MaxRadius => {
                    self.results.push(max_r as f32);
                }
                PuckerMetric::Amplitude => {
                    let eig = eig.as_ref().expect("covariance eigen must exist");
                    let mut min_eval = eig.eigenvalues[0];
                    if eig.eigenvalues[1] < min_eval {
                        min_eval = eig.eigenvalues[1];
                    }
                    if eig.eigenvalues[2] < min_eval {
                        min_eval = eig.eigenvalues[2];
                    }
                    let amp = min_eval.max(0.0).sqrt();
                    self.results.push(amp as f32);
                }
            }

            if self.return_phase {
                let eig = eig.as_ref().expect("covariance eigen must exist");
                let mut min_idx = 0usize;
                if eig.eigenvalues[1] < eig.eigenvalues[min_idx] {
                    min_idx = 1;
                }
                if eig.eigenvalues[2] < eig.eigenvalues[min_idx] {
                    min_idx = 2;
                }
                let normal = [
                    eig.eigenvectors[(0, min_idx)],
                    eig.eigenvectors[(1, min_idx)],
                    eig.eigenvectors[(2, min_idx)],
                ];
                let phase = pucker_phase(chunk, frame, sel, center, normal);
                self.phases.push(phase);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.return_phase {
            if self.phases.len() != self.results.len() {
                return Err(TrajError::Invalid(
                    "pucker phase buffer length mismatch".into(),
                ));
            }
            let rows = self.results.len();
            let mut data = Vec::with_capacity(rows * 2);
            for (value, phase) in self.results.iter().zip(self.phases.iter()) {
                data.push(*value);
                data.push(*phase);
            }
            self.results.clear();
            self.phases.clear();
            return Ok(PlanOutput::Matrix {
                data,
                rows,
                cols: 2,
            });
        }
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

fn pucker_phase(
    chunk: &FrameChunk,
    frame: usize,
    sel: &[u32],
    center: [f64; 3],
    normal: [f64; 3],
) -> f32 {
    if sel.len() < 3 {
        return 0.0;
    }
    let n = sel.len() as f64;
    let frame_offset = frame * chunk.n_atoms;
    let mut qx = 0.0f64;
    let mut qy = 0.0f64;
    for (j, &idx) in sel.iter().enumerate() {
        let p = chunk.coords[frame_offset + idx as usize];
        let dx = p[0] as f64 - center[0];
        let dy = p[1] as f64 - center[1];
        let dz = p[2] as f64 - center[2];
        let z = dx * normal[0] + dy * normal[1] + dz * normal[2];
        let theta = 2.0 * std::f64::consts::PI * (j as f64) / n;
        qx += z * theta.cos();
        qy += z * theta.sin();
    }
    let mut phase = qy.atan2(qx).to_degrees();
    if phase < 0.0 {
        phase += 360.0;
    }
    phase as f32
}

pub struct RotateDihedralPlan {
    sel_a: Selection,
    sel_b: Selection,
    sel_c: Selection,
    sel_d: Selection,
    rotate_sel: Selection,
    angle: f64,
    mass_weighted: bool,
    degrees: bool,
    results: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<RotateDihedralGpuState>,
}

#[cfg(feature = "cuda")]
struct RotateDihedralGpuState {
    groups: GpuGroups,
    masses: GpuBufferF32,
    mask: GpuBufferU32,
}

impl RotateDihedralPlan {
    pub fn new(
        sel_a: Selection,
        sel_b: Selection,
        sel_c: Selection,
        sel_d: Selection,
        rotate_sel: Selection,
        angle: f64,
        mass_weighted: bool,
        degrees: bool,
    ) -> Self {
        Self {
            sel_a,
            sel_b,
            sel_c,
            sel_d,
            rotate_sel,
            angle,
            mass_weighted,
            degrees,
            results: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for RotateDihedralPlan {
    fn name(&self) -> &'static str {
        "rotate_dihedral"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let mut indices =
                    Vec::with_capacity(self.sel_b.indices.len() + self.sel_c.indices.len());
                indices.extend(self.sel_b.indices.iter().copied());
                indices.extend(self.sel_c.indices.iter().copied());
                let offsets = vec![
                    0u32,
                    self.sel_b.indices.len() as u32,
                    (self.sel_b.indices.len() + self.sel_c.indices.len()) as u32,
                ];
                let max_len = self.sel_b.indices.len().max(self.sel_c.indices.len());
                let groups = ctx.groups(&offsets, &indices, max_len)?;
                let masses = if self.mass_weighted {
                    _system.atoms.mass.clone()
                } else {
                    vec![1.0f32; _system.n_atoms()]
                };
                let masses = ctx.upload_f32(&masses)?;
                let mut mask = vec![0u32; _system.n_atoms()];
                for &idx in self.rotate_sel.indices.iter() {
                    if let Some(slot) = mask.get_mut(idx as usize) {
                        *slot = 1;
                    }
                }
                let mask = ctx.upload_u32(&mask)?;
                self.gpu = Some(RotateDihedralGpuState {
                    groups,
                    masses,
                    mask,
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
        let n_atoms = chunk.n_atoms;
        if n_atoms == 0 {
            return Ok(());
        }
        let _ = (&self.sel_a, &self.sel_d);
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
            let mut pivots = Vec::with_capacity(chunk.n_frames);
            let mut axes = Vec::with_capacity(chunk.n_frames);
            for frame in 0..chunk.n_frames {
                let base = frame * 2;
                let b = coms[base];
                let c = coms[base + 1];
                pivots.push(b);
                axes.push(Float4 {
                    x: c.x - b.x,
                    y: c.y - b.y,
                    z: c.z - b.z,
                    w: 0.0,
                });
            }
            let angle = if self.degrees {
                self.angle.to_radians()
            } else {
                self.angle
            };
            let angles = vec![angle as f32; chunk.n_frames];
            let out = ctx.rotate_dihedral(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.mask,
                &pivots,
                &axes,
                &angles,
            )?;
            self.results.extend(out);
            return Ok(());
        }
        let mut mask = vec![false; n_atoms];
        for &idx in self.rotate_sel.indices.iter() {
            if let Some(slot) = mask.get_mut(idx as usize) {
                *slot = true;
            }
        }
        let masses = &system.atoms.mass;
        let angle = if self.degrees {
            self.angle.to_radians()
        } else {
            self.angle
        };
        for frame in 0..chunk.n_frames {
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
            let axis = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
            for atom in 0..n_atoms {
                let p = chunk.coords[frame * n_atoms + atom];
                let out = if mask[atom] {
                    rotate_about_axis([p[0] as f64, p[1] as f64, p[2] as f64], b, axis, angle)
                } else {
                    [p[0] as f64, p[1] as f64, p[2] as f64]
                };
                self.results.push(out[0] as f32);
                self.results.push(out[1] as f32);
                self.results.push(out[2] as f32);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

pub struct SetDihedralPlan {
    sel_a: Selection,
    sel_b: Selection,
    sel_c: Selection,
    sel_d: Selection,
    rotate_sel: Selection,
    target: f64,
    mass_weighted: bool,
    pbc: PbcMode,
    degrees: bool,
    range360: bool,
    results: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<SetDihedralGpuState>,
}

#[cfg(feature = "cuda")]
struct SetDihedralGpuState {
    groups: GpuGroups,
    masses: GpuBufferF32,
    mask: GpuBufferU32,
}

impl SetDihedralPlan {
    pub fn new(
        sel_a: Selection,
        sel_b: Selection,
        sel_c: Selection,
        sel_d: Selection,
        rotate_sel: Selection,
        target: f64,
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
            rotate_sel,
            target,
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

impl Plan for SetDihedralPlan {
    fn name(&self) -> &'static str {
        "set_dihedral"
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
                    (self.sel_a.indices.len() + self.sel_b.indices.len() + self.sel_c.indices.len())
                        as u32,
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
                let mut mask = vec![0u32; _system.n_atoms()];
                for &idx in self.rotate_sel.indices.iter() {
                    if let Some(slot) = mask.get_mut(idx as usize) {
                        *slot = 1;
                    }
                }
                let mask = ctx.upload_u32(&mask)?;
                self.gpu = Some(SetDihedralGpuState {
                    groups,
                    masses,
                    mask,
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
        let n_atoms = chunk.n_atoms;
        if n_atoms == 0 {
            return Ok(());
        }
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
            let mut pivots = Vec::with_capacity(chunk.n_frames);
            let mut axes = Vec::with_capacity(chunk.n_frames);
            let mut angles = Vec::with_capacity(chunk.n_frames);
            for frame in 0..chunk.n_frames {
                let base = frame * 4;
                let a = coms[base];
                let b = coms[base + 1];
                let c = coms[base + 2];
                let d = coms[base + 3];
                let mut b0x = a.x as f64 - b.x as f64;
                let mut b0y = a.y as f64 - b.y as f64;
                let mut b0z = a.z as f64 - b.z as f64;
                let mut b1x = c.x as f64 - b.x as f64;
                let mut b1y = c.y as f64 - b.y as f64;
                let mut b1z = c.z as f64 - b.z as f64;
                let mut b2x = d.x as f64 - c.x as f64;
                let mut b2y = d.y as f64 - c.y as f64;
                let mut b2z = d.z as f64 - c.z as f64;
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    let (lx, ly, lz) = box_lengths(chunk, frame)?;
                    apply_pbc(&mut b0x, &mut b0y, &mut b0z, lx, ly, lz);
                    apply_pbc(&mut b1x, &mut b1y, &mut b1z, lx, ly, lz);
                    apply_pbc(&mut b2x, &mut b2y, &mut b2z, lx, ly, lz);
                }
                let current = dihedral_from_vectors(
                    [b0x, b0y, b0z],
                    [b1x, b1y, b1z],
                    [b2x, b2y, b2z],
                    self.degrees,
                    self.range360,
                ) as f64;
                let diff = angle_diff(self.target, current, self.degrees);
                let angle = if self.degrees {
                    diff.to_radians()
                } else {
                    diff
                };
                pivots.push(b);
                axes.push(Float4 {
                    x: c.x - b.x,
                    y: c.y - b.y,
                    z: c.z - b.z,
                    w: 0.0,
                });
                angles.push(angle as f32);
            }
            let out = ctx.rotate_dihedral(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.mask,
                &pivots,
                &axes,
                &angles,
            )?;
            self.results.extend(out);
            return Ok(());
        }
        let mut mask = vec![false; n_atoms];
        for &idx in self.rotate_sel.indices.iter() {
            if let Some(slot) = mask.get_mut(idx as usize) {
                *slot = true;
            }
        }
        let masses = &system.atoms.mass;
        let target = self.target;
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
            let current = dihedral_value(
                a,
                b,
                c,
                d,
                self.pbc,
                Some((chunk, frame)),
                self.degrees,
                self.range360,
            )?;
            let diff = angle_diff(target, current as f64, self.degrees);
            let angle = if self.degrees {
                diff.to_radians()
            } else {
                diff
            };
            let axis = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
            for atom in 0..n_atoms {
                let p = chunk.coords[frame * n_atoms + atom];
                let out = if mask[atom] {
                    rotate_about_axis([p[0] as f64, p[1] as f64, p[2] as f64], b, axis, angle)
                } else {
                    [p[0] as f64, p[1] as f64, p[2] as f64]
                };
                self.results.push(out[0] as f32);
                self.results.push(out[1] as f32);
                self.results.push(out[2] as f32);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}
