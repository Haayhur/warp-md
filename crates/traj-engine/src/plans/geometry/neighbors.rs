use traj_core::error::TrajResult;
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use super::utils::*;
use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::PbcMode;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuSelection};

#[cfg(feature = "cuda")]
struct PairwiseGpuState {
    sel_a: GpuSelection,
    sel_b: GpuSelection,
}

#[cfg(feature = "cuda")]
struct DistancePointGpuState {
    selection: GpuSelection,
}

pub struct MindistPlan {
    sel_a: Selection,
    sel_b: Selection,
    pbc: PbcMode,
    results: Vec<f32>,
    frames: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<PairwiseGpuState>,
}

impl MindistPlan {
    pub fn new(sel_a: Selection, sel_b: Selection, pbc: PbcMode) -> Self {
        Self {
            sel_a,
            sel_b,
            pbc,
            results: Vec::new(),
            frames: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for MindistPlan {
    fn name(&self) -> &'static str {
        "mindist"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let sel_a = ctx.selection(&self.sel_a.indices, None)?;
                let sel_b = ctx.selection(&self.sel_b.indices, None)?;
                self.gpu = Some(PairwiseGpuState { sel_a, sel_b });
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
        let n_pairs = self.sel_a.indices.len() * self.sel_b.indices.len();
        if n_pairs == 0 {
            self.results
                .extend(std::iter::repeat(0.0).take(chunk.n_frames));
            self.frames += chunk.n_frames;
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let boxes = chunk_boxes(chunk, self.pbc)?;
            let out = ctx.mindist_pairs(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.sel_a,
                &gpu.sel_b,
                &boxes,
            )?;
            self.results.extend(out);
            self.frames += chunk.n_frames;
            return Ok(());
        }

        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let (lx, ly, lz) = if matches!(self.pbc, PbcMode::Orthorhombic) {
                box_lengths(chunk, frame)?
            } else {
                (0.0, 0.0, 0.0)
            };
            let mut min_val = f64::INFINITY;
            for &a in self.sel_a.indices.iter() {
                let a_idx = a as usize;
                let pa = chunk.coords[frame * n_atoms + a_idx];
                for &b in self.sel_b.indices.iter() {
                    let b_idx = b as usize;
                    let pb = chunk.coords[frame * n_atoms + b_idx];
                    let mut dx = (pb[0] - pa[0]) as f64;
                    let mut dy = (pb[1] - pa[1]) as f64;
                    let mut dz = (pb[2] - pa[2]) as f64;
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                    }
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < min_val {
                        min_val = dist;
                    }
                }
            }
            if !min_val.is_finite() {
                min_val = 0.0;
            }
            self.results.push(min_val as f32);
        }
        self.frames += chunk.n_frames;
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

pub struct HausdorffPlan {
    sel_a: Selection,
    sel_b: Selection,
    pbc: PbcMode,
    results: Vec<f32>,
    frames: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<PairwiseGpuState>,
}

impl HausdorffPlan {
    pub fn new(sel_a: Selection, sel_b: Selection, pbc: PbcMode) -> Self {
        Self {
            sel_a,
            sel_b,
            pbc,
            results: Vec::new(),
            frames: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for HausdorffPlan {
    fn name(&self) -> &'static str {
        "hausdorff"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let sel_a = ctx.selection(&self.sel_a.indices, None)?;
                let sel_b = ctx.selection(&self.sel_b.indices, None)?;
                self.gpu = Some(PairwiseGpuState { sel_a, sel_b });
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
        let n_a = self.sel_a.indices.len();
        let n_b = self.sel_b.indices.len();
        if n_a == 0 || n_b == 0 {
            self.results
                .extend(std::iter::repeat(0.0).take(chunk.n_frames));
            self.frames += chunk.n_frames;
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let out = if matches!(self.pbc, PbcMode::Orthorhombic) {
                let (cell, inv) = chunk_cell_mats(chunk)?;
                ctx.hausdorff_pairs_triclinic(
                    &coords,
                    chunk.n_atoms,
                    chunk.n_frames,
                    &gpu.sel_a,
                    &gpu.sel_b,
                    &cell,
                    &inv,
                )?
            } else {
                let boxes = chunk_boxes(chunk, self.pbc)?;
                ctx.hausdorff_pairs(
                    &coords,
                    chunk.n_atoms,
                    chunk.n_frames,
                    &gpu.sel_a,
                    &gpu.sel_b,
                    &boxes,
                )?
            };
            self.results.extend(out);
            self.frames += chunk.n_frames;
            return Ok(());
        }

        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let cell_inv = if matches!(self.pbc, PbcMode::Orthorhombic) {
                Some(cell_and_inv_from_box(chunk.box_[frame])?)
            } else {
                None
            };
            let mut max_min_a = 0.0f64;
            for &a in self.sel_a.indices.iter() {
                let pa = chunk.coords[frame * n_atoms + a as usize];
                let mut min_val = f64::INFINITY;
                for &b in self.sel_b.indices.iter() {
                    let pb = chunk.coords[frame * n_atoms + b as usize];
                    let mut dx = (pb[0] - pa[0]) as f64;
                    let mut dy = (pb[1] - pa[1]) as f64;
                    let mut dz = (pb[2] - pa[2]) as f64;
                    if let Some((cell, inv)) = &cell_inv {
                        apply_pbc_triclinic(&mut dx, &mut dy, &mut dz, cell, inv);
                    }
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < min_val {
                        min_val = dist;
                    }
                }
                if min_val > max_min_a {
                    max_min_a = min_val;
                }
            }
            let mut max_min_b = 0.0f64;
            for &b in self.sel_b.indices.iter() {
                let pb = chunk.coords[frame * n_atoms + b as usize];
                let mut min_val = f64::INFINITY;
                for &a in self.sel_a.indices.iter() {
                    let pa = chunk.coords[frame * n_atoms + a as usize];
                    let mut dx = (pb[0] - pa[0]) as f64;
                    let mut dy = (pb[1] - pa[1]) as f64;
                    let mut dz = (pb[2] - pa[2]) as f64;
                    if let Some((cell, inv)) = &cell_inv {
                        apply_pbc_triclinic(&mut dx, &mut dy, &mut dz, cell, inv);
                    }
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < min_val {
                        min_val = dist;
                    }
                }
                if min_val > max_min_b {
                    max_min_b = min_val;
                }
            }
            let h = max_min_a.max(max_min_b) as f32;
            self.results.push(h);
        }
        self.frames += chunk.n_frames;
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

pub struct ClosestAtomPlan {
    selection: Selection,
    point: [f64; 3],
    pbc: PbcMode,
    results: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<DistancePointGpuState>,
}

impl ClosestAtomPlan {
    pub fn new(selection: Selection, point: [f64; 3], pbc: PbcMode) -> Self {
        Self {
            selection,
            point,
            pbc,
            results: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for ClosestAtomPlan {
    fn name(&self) -> &'static str {
        "closest_atom"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let selection = ctx.selection(&self.selection.indices, None)?;
                self.gpu = Some(DistancePointGpuState { selection });
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
        if self.selection.indices.is_empty() {
            self.results
                .extend(std::iter::repeat(-1.0).take(chunk.n_frames));
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let boxes = chunk_boxes(chunk, self.pbc)?;
            let out = ctx.closest_atom(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.selection,
                [
                    self.point[0] as f32,
                    self.point[1] as f32,
                    self.point[2] as f32,
                ],
                &boxes,
            )?;
            self.results.extend(out);
            return Ok(());
        }

        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let (lx, ly, lz) = if matches!(self.pbc, PbcMode::Orthorhombic) {
                box_lengths(chunk, frame)?
            } else {
                (0.0, 0.0, 0.0)
            };
            let mut min_val = f64::INFINITY;
            let mut min_idx = 0usize;
            for (i, &idx) in self.selection.indices.iter().enumerate() {
                let p = chunk.coords[frame * n_atoms + idx as usize];
                let mut dx = p[0] as f64 - self.point[0];
                let mut dy = p[1] as f64 - self.point[1];
                let mut dz = p[2] as f64 - self.point[2];
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                }
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist < min_val {
                    min_val = dist;
                    min_idx = i;
                }
            }
            let atom_idx = self.selection.indices[min_idx] as f32;
            self.results.push(atom_idx);
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

pub struct SearchNeighborsPlan {
    target: Selection,
    probe: Selection,
    cutoff: f64,
    pbc: PbcMode,
    results: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<PairwiseGpuState>,
}

impl SearchNeighborsPlan {
    pub fn new(target: Selection, probe: Selection, cutoff: f64, pbc: PbcMode) -> Self {
        Self {
            target,
            probe,
            cutoff,
            pbc,
            results: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for SearchNeighborsPlan {
    fn name(&self) -> &'static str {
        "search_neighbors"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let sel_a = ctx.selection(&self.target.indices, None)?;
                let sel_b = ctx.selection(&self.probe.indices, None)?;
                self.gpu = Some(PairwiseGpuState { sel_a, sel_b });
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
        let n_a = self.target.indices.len();
        let n_b = self.probe.indices.len();
        if n_a == 0 || n_b == 0 {
            self.results
                .extend(std::iter::repeat(0.0).take(chunk.n_frames));
            return Ok(());
        }
        let cutoff = self.cutoff;
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let boxes = chunk_boxes(chunk, self.pbc)?;
            let out = ctx.search_neighbors(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.sel_a,
                &gpu.sel_b,
                &boxes,
                cutoff as f32,
            )?;
            self.results.extend(out.into_iter().map(|v| v as f32));
            return Ok(());
        }

        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let (lx, ly, lz) = if matches!(self.pbc, PbcMode::Orthorhombic) {
                box_lengths(chunk, frame)?
            } else {
                (0.0, 0.0, 0.0)
            };
            let mut count = 0usize;
            for &b in self.probe.indices.iter() {
                let pb = chunk.coords[frame * n_atoms + b as usize];
                let mut min_val = f64::INFINITY;
                for &a in self.target.indices.iter() {
                    let pa = chunk.coords[frame * n_atoms + a as usize];
                    let mut dx = (pb[0] - pa[0]) as f64;
                    let mut dy = (pb[1] - pa[1]) as f64;
                    let mut dz = (pb[2] - pa[2]) as f64;
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                    }
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < min_val {
                        min_val = dist;
                    }
                }
                if min_val <= cutoff {
                    count += 1;
                }
            }
            self.results.push(count as f32);
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}
