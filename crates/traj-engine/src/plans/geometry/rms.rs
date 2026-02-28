use std::collections::BTreeMap;

use nalgebra::{Matrix3, Vector3};

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use super::utils::*;
use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::ReferenceMode;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuGroups, GpuReference, GpuSelection};

pub struct RmsfPlan {
    selection: Selection,
    mean: Vec<[f64; 3]>,
    m2: Vec<[f64; 3]>,
    sum: Vec<[f64; 3]>,
    sum_sq: Vec<f64>,
    frames: usize,
    use_gpu: bool,
    use_selected_input: bool,
    #[cfg(feature = "cuda")]
    gpu: Option<RmsfGpuState>,
}

#[cfg(feature = "cuda")]
struct RmsfGpuState {
    selection: GpuSelection,
}

impl RmsfPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            mean: Vec::new(),
            m2: Vec::new(),
            sum: Vec::new(),
            sum_sq: Vec::new(),
            frames: 0,
            use_gpu: false,
            use_selected_input: true,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for RmsfPlan {
    fn name(&self) -> &'static str {
        "rmsf"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        let n = self.selection.indices.len();
        self.mean = vec![[0.0; 3]; n];
        self.m2 = vec![[0.0; 3]; n];
        self.sum = vec![[0.0; 3]; n];
        self.sum_sq = vec![0.0; n];
        self.frames = 0;
        self.use_gpu = false;
        self.use_selected_input = true;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let dense_selection: Vec<u32> = (0..self.selection.indices.len())
                    .map(|i| i as u32)
                    .collect();
                let selection = if self.use_selected_input {
                    ctx.selection(&dense_selection, None)?
                } else {
                    ctx.selection(&self.selection.indices, None)?
                };
                self.gpu = Some(RmsfGpuState { selection });
                self.use_gpu = true;
            }
        }
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(self.selection.indices.as_slice())
        } else {
            None
        }
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        Some(self.selection.indices.as_slice())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let accum = ctx.rmsf_accum(&coords, chunk.n_atoms, chunk.n_frames, &gpu.selection)?;
            for i in 0..self.selection.indices.len() {
                self.sum[i][0] += accum.sum_x[i] as f64;
                self.sum[i][1] += accum.sum_y[i] as f64;
                self.sum[i][2] += accum.sum_z[i] as f64;
                self.sum_sq[i] += accum.sum_sq[i] as f64;
            }
            self.frames += chunk.n_frames;
            return Ok(());
        }
        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            self.frames += 1;
            let n = self.frames as f64;
            for (i, &idx) in self.selection.indices.iter().enumerate() {
                let p = chunk.coords[frame * n_atoms + idx as usize];
                let x = p[0] as f64;
                let y = p[1] as f64;
                let z = p[2] as f64;
                let mean = &mut self.mean[i];
                let m2 = &mut self.m2[i];
                let dx = x - mean[0];
                let dy = y - mean[1];
                let dz = z - mean[2];
                mean[0] += dx / n;
                mean[1] += dy / n;
                mean[2] += dz / n;
                let dx2 = x - mean[0];
                let dy2 = y - mean[1];
                let dz2 = z - mean[2];
                m2[0] += dx * dx2;
                m2[1] += dy * dy2;
                m2[2] += dz * dz2;
            }
        }
        Ok(())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        _source_selection: &[u32],
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let accum = ctx.rmsf_accum(&coords, chunk.n_atoms, chunk.n_frames, &gpu.selection)?;
            for i in 0..self.selection.indices.len() {
                self.sum[i][0] += accum.sum_x[i] as f64;
                self.sum[i][1] += accum.sum_y[i] as f64;
                self.sum[i][2] += accum.sum_z[i] as f64;
                self.sum_sq[i] += accum.sum_sq[i] as f64;
            }
            self.frames += chunk.n_frames;
            return Ok(());
        }
        // Selected-read chunks are already ordered to match self.selection.
        for frame in 0..chunk.n_frames {
            self.frames += 1;
            let n = self.frames as f64;
            let frame_base = frame * chunk.n_atoms;
            for i in 0..chunk.n_atoms {
                let p = chunk.coords[frame_base + i];
                let x = p[0] as f64;
                let y = p[1] as f64;
                let z = p[2] as f64;
                let mean = &mut self.mean[i];
                let m2 = &mut self.m2[i];
                let dx = x - mean[0];
                let dy = y - mean[1];
                let dz = z - mean[2];
                mean[0] += dx / n;
                mean[1] += dy / n;
                mean[2] += dz / n;
                let dx2 = x - mean[0];
                let dy2 = y - mean[1];
                let dz2 = z - mean[2];
                m2[0] += dx * dx2;
                m2[1] += dy * dy2;
                m2[2] += dz * dz2;
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.frames == 0 {
            return Ok(PlanOutput::Series(vec![0.0; self.selection.indices.len()]));
        }
        let n = self.frames as f64;
        let mut out = Vec::with_capacity(self.selection.indices.len());
        if self.use_gpu {
            for i in 0..self.selection.indices.len() {
                let mx = self.sum[i][0] / n;
                let my = self.sum[i][1] / n;
                let mz = self.sum[i][2] / n;
                let mut var = self.sum_sq[i] / n - (mx * mx + my * my + mz * mz);
                if var < 0.0 {
                    var = 0.0;
                }
                out.push(var.sqrt() as f32);
            }
        } else {
            for i in 0..self.selection.indices.len() {
                let m2 = &self.m2[i];
                let var = (m2[0] + m2[1] + m2[2]) / n;
                out.push(var.sqrt() as f32);
            }
        }
        Ok(PlanOutput::Series(out))
    }
}

pub struct BfactorsPlan {
    inner: RmsfPlan,
}

impl BfactorsPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            inner: RmsfPlan::new(selection),
        }
    }
}

impl Plan for BfactorsPlan {
    fn name(&self) -> &'static str {
        "bfactors"
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
        let output = self.inner.finalize()?;
        match output {
            PlanOutput::Series(vals) => {
                let factor = 8.0 * std::f64::consts::PI * std::f64::consts::PI / 3.0;
                let mut out = Vec::with_capacity(vals.len());
                for v in vals {
                    let b = factor * (v as f64) * (v as f64);
                    out.push(b as f32);
                }
                Ok(PlanOutput::Series(out))
            }
            _ => Err(TrajError::Mismatch("bfactors expects rmsf output".into())),
        }
    }
}

pub struct AtomicFluctPlan {
    inner: RmsfPlan,
}

impl AtomicFluctPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            inner: RmsfPlan::new(selection),
        }
    }
}

impl Plan for AtomicFluctPlan {
    fn name(&self) -> &'static str {
        "atomicfluct"
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
        self.inner.finalize()
    }
}

pub struct RmsdPerResPlan {
    selection: Selection,
    align: bool,
    reference_mode: ReferenceMode,
    reference_sel: Option<Vec<[f32; 4]>>,
    reference_all: Option<Vec<[f32; 4]>>,
    results: Vec<f32>,
    frames: usize,
    resids: Vec<i32>,
    groups: Vec<Vec<usize>>,
    group_sizes: Vec<usize>,
    #[cfg(feature = "cuda")]
    gpu: Option<RmsdPerResGpuState>,
}

#[cfg(feature = "cuda")]
struct RmsdPerResGpuState {
    groups: GpuGroups,
    reference: Option<GpuReference>,
}

impl RmsdPerResPlan {
    pub fn new(selection: Selection, reference_mode: ReferenceMode, align: bool) -> Self {
        Self {
            selection,
            align,
            reference_mode,
            reference_sel: None,
            reference_all: None,
            results: Vec::new(),
            frames: 0,
            resids: Vec::new(),
            groups: Vec::new(),
            group_sizes: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn resids(&self) -> &[i32] {
        &self.resids
    }
}

impl Plan for RmsdPerResPlan {
    fn name(&self) -> &'static str {
        "rmsd_perres"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        self.groups.clear();
        self.resids.clear();
        self.group_sizes.clear();

        let mut map: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
        for &idx in self.selection.indices.iter() {
            let atom_idx = idx as usize;
            let resid = system.atoms.resid[atom_idx];
            map.entry(resid).or_default().push(atom_idx);
        }
        for (resid, atoms) in map {
            self.resids.push(resid);
            self.groups.push(atoms);
        }
        self.group_sizes = self.groups.iter().map(|g| g.len()).collect();

        match self.reference_mode {
            ReferenceMode::Topology => {
                let positions0 = system
                    .positions0
                    .as_ref()
                    .ok_or_else(|| TrajError::Mismatch("no topology reference coords".into()))?;
                self.reference_all = Some(positions0.clone());
                let mut reference = Vec::with_capacity(self.selection.indices.len());
                for &idx in self.selection.indices.iter() {
                    reference.push(positions0[idx as usize]);
                }
                self.reference_sel = Some(reference);
            }
            ReferenceMode::Frame0 => {
                self.reference_all = None;
                self.reference_sel = None;
            }
        }

        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let mut offsets = Vec::with_capacity(self.groups.len() + 1);
                offsets.push(0u32);
                let mut indices: Vec<u32> = Vec::new();
                let mut max_len = 0usize;
                for group in self.groups.iter() {
                    max_len = max_len.max(group.len());
                    for &atom_idx in group.iter() {
                        indices.push(atom_idx as u32);
                    }
                    let next = *offsets.last().unwrap() + group.len() as u32;
                    offsets.push(next);
                }
                let groups = ctx.groups(&offsets, &indices, max_len)?;
                let reference = match self.reference_all.as_ref() {
                    Some(coords) => Some(ctx.reference(&convert_coords(coords))?),
                    None => None,
                };
                self.gpu = Some(RmsdPerResGpuState { groups, reference });
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
        if self.reference_sel.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0) {
            if chunk.n_frames == 0 {
                return Ok(());
            }
            let mut reference_all = Vec::with_capacity(chunk.n_atoms);
            for idx in 0..chunk.n_atoms {
                reference_all.push(chunk.coords[idx]);
            }
            let mut reference_sel = Vec::with_capacity(self.selection.indices.len());
            for &idx in self.selection.indices.iter() {
                reference_sel.push(chunk.coords[idx as usize]);
            }
            self.reference_all = Some(reference_all);
            self.reference_sel = Some(reference_sel);
            #[cfg(feature = "cuda")]
            if let Device::Cuda(ctx) = _device {
                if let Some(gpu) = &mut self.gpu {
                    gpu.reference =
                        Some(ctx.reference(&convert_coords(self.reference_all.as_ref().unwrap()))?);
                }
            }
        }

        let reference_sel = self
            .reference_sel
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;
        let reference_all = self
            .reference_all
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;

        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            if let Some(reference_gpu) = gpu.reference.as_ref() {
                let coords = convert_coords(&chunk.coords);
                let mut rotations = Vec::with_capacity(chunk.n_frames * 9);
                let mut cx = Vec::with_capacity(chunk.n_frames * 3);
                let mut cy = Vec::with_capacity(chunk.n_frames * 3);
                if self.align {
                    let n_atoms = chunk.n_atoms;
                    for frame in 0..chunk.n_frames {
                        let mut frame_sel = Vec::with_capacity(self.selection.indices.len());
                        for &idx in self.selection.indices.iter() {
                            frame_sel.push(chunk.coords[frame * n_atoms + idx as usize]);
                        }
                        let (r, cx_v, cy_v) = kabsch_rotation(&frame_sel, reference_sel);
                        rotations.extend_from_slice(&[
                            r[(0, 0)] as f32,
                            r[(0, 1)] as f32,
                            r[(0, 2)] as f32,
                            r[(1, 0)] as f32,
                            r[(1, 1)] as f32,
                            r[(1, 2)] as f32,
                            r[(2, 0)] as f32,
                            r[(2, 1)] as f32,
                            r[(2, 2)] as f32,
                        ]);
                        cx.extend_from_slice(&[cx_v[0] as f32, cx_v[1] as f32, cx_v[2] as f32]);
                        cy.extend_from_slice(&[cy_v[0] as f32, cy_v[1] as f32, cy_v[2] as f32]);
                    }
                } else {
                    for _ in 0..chunk.n_frames {
                        rotations.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
                        cx.extend_from_slice(&[0.0, 0.0, 0.0]);
                        cy.extend_from_slice(&[0.0, 0.0, 0.0]);
                    }
                }
                let sums = ctx.rmsd_per_res_accum(
                    &coords,
                    chunk.n_atoms,
                    chunk.n_frames,
                    &gpu.groups,
                    reference_gpu,
                    &rotations,
                    &cx,
                    &cy,
                )?;
                for frame in 0..chunk.n_frames {
                    let base = frame * self.groups.len();
                    for (i, &size) in self.group_sizes.iter().enumerate() {
                        let n = size.max(1) as f32;
                        let sum = sums[base + i];
                        self.results.push((sum / n).sqrt());
                    }
                    self.frames += 1;
                }
                return Ok(());
            }
        }

        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let mut frame_sel = Vec::with_capacity(self.selection.indices.len());
            for &idx in self.selection.indices.iter() {
                frame_sel.push(chunk.coords[frame * n_atoms + idx as usize]);
            }

            let (r, cx, cy) = if self.align {
                kabsch_rotation(&frame_sel, reference_sel)
            } else {
                (Matrix3::identity(), Vector3::zeros(), Vector3::zeros())
            };

            for group in self.groups.iter() {
                let n = group.len().max(1) as f64;
                let mut sum = 0.0f64;
                for &atom_idx in group.iter() {
                    let p = chunk.coords[frame * n_atoms + atom_idx];
                    let pr = reference_all[atom_idx];
                    let pv = Vector3::new(p[0] as f64, p[1] as f64, p[2] as f64);
                    let rv = Vector3::new(pr[0] as f64, pr[1] as f64, pr[2] as f64);
                    let diff = if self.align {
                        let aligned = r * (pv - cx);
                        aligned - (rv - cy)
                    } else {
                        pv - rv
                    };
                    sum += diff.dot(&diff);
                }
                self.results.push((sum / n).sqrt() as f32);
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: self.groups.len(),
        })
    }
}
