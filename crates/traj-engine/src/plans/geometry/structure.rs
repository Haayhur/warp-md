use traj_core::error::TrajResult;
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use super::utils::*;
use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::PbcMode;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuBufferF32, GpuGroups, GpuSelection};

#[cfg(feature = "cuda")]
struct PairwiseGpuState {
    sel_a: GpuSelection,
    sel_b: GpuSelection,
}

pub struct CheckStructurePlan {
    selection: Selection,
    results: Vec<f32>,
}

impl CheckStructurePlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            results: Vec::new(),
        }
    }
}

impl Plan for CheckStructurePlan {
    fn name(&self) -> &'static str {
        "check_structure"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        for frame in 0..chunk.n_frames {
            let mut ok = true;
            for &idx in sel.iter() {
                let p = chunk.coords[frame * n_atoms + idx as usize];
                if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
                    ok = false;
                    break;
                }
            }
            self.results.push(if ok { 1.0 } else { 0.0 });
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

pub struct CheckChiralityPlan {
    groups: Vec<(Selection, Selection, Selection, Selection)>,
    mass_weighted: bool,
    results: Vec<f32>,
    frames: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<ChiralityGpuState>,
}

#[cfg(feature = "cuda")]
struct ChiralityGpuState {
    groups: GpuGroups,
    masses: GpuBufferF32,
    n_centers: usize,
}

impl CheckChiralityPlan {
    pub fn new(
        groups: Vec<(Selection, Selection, Selection, Selection)>,
        mass_weighted: bool,
    ) -> Self {
        Self {
            groups,
            mass_weighted,
            results: Vec::new(),
            frames: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for CheckChiralityPlan {
    fn name(&self) -> &'static str {
        "check_chirality"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let mut indices = Vec::new();
                let mut offsets = Vec::with_capacity(self.groups.len() * 4 + 1);
                offsets.push(0u32);
                let mut max_len = 0usize;
                for (a, b, c, d) in self.groups.iter() {
                    for sel in [&a.indices, &b.indices, &c.indices, &d.indices] {
                        indices.extend(sel.iter().copied());
                        offsets.push(indices.len() as u32);
                        if sel.len() > max_len {
                            max_len = sel.len();
                        }
                    }
                }
                if offsets.len() > 1 {
                    let groups = ctx.groups(&offsets, &indices, max_len)?;
                    let masses = if self.mass_weighted {
                        _system.atoms.mass.clone()
                    } else {
                        vec![1.0f32; _system.n_atoms()]
                    };
                    let masses = ctx.upload_f32(&masses)?;
                    self.gpu = Some(ChiralityGpuState {
                        groups,
                        masses,
                        n_centers: self.groups.len(),
                    });
                }
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
            if gpu.n_centers == 0 {
                self.frames += chunk.n_frames;
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
            let vols = ctx.chirality_volume(&coms, chunk.n_frames, gpu.n_centers)?;
            self.results.extend(vols);
            self.frames += chunk.n_frames;
            return Ok(());
        }
        let masses = &system.atoms.mass;
        for frame in 0..chunk.n_frames {
            for (a, b, c, d) in self.groups.iter() {
                let pa = center_of_selection(chunk, frame, &a.indices, masses, self.mass_weighted);
                let pb = center_of_selection(chunk, frame, &b.indices, masses, self.mass_weighted);
                let pc = center_of_selection(chunk, frame, &c.indices, masses, self.mass_weighted);
                let pd = center_of_selection(chunk, frame, &d.indices, masses, self.mass_weighted);
                let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
                let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
                let ad = [pd[0] - pa[0], pd[1] - pa[1], pd[2] - pa[2]];
                let cross = [
                    ab[1] * ac[2] - ab[2] * ac[1],
                    ab[2] * ac[0] - ab[0] * ac[2],
                    ab[0] * ac[1] - ab[1] * ac[0],
                ];
                let vol = cross[0] * ad[0] + cross[1] * ad[1] + cross[2] * ad[2];
                self.results.push(vol as f32);
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

pub struct AtomMapPlan {
    sel_a: Selection,
    sel_b: Selection,
    pbc: PbcMode,
    results: Vec<f32>,
    frames: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<PairwiseGpuState>,
}

impl AtomMapPlan {
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

impl Plan for AtomMapPlan {
    fn name(&self) -> &'static str {
        "atom_map"
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
        if n_a == 0 {
            self.frames += chunk.n_frames;
            return Ok(());
        }
        if n_b == 0 {
            for _ in 0..chunk.n_frames {
                self.results.extend(std::iter::repeat(-1.0).take(n_a));
            }
            self.frames += chunk.n_frames;
            return Ok(());
        }

        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let boxes = chunk_boxes(chunk, self.pbc)?;
            let out = ctx.atom_map_pairs(
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
            for &a in self.sel_a.indices.iter() {
                let pa = chunk.coords[frame * n_atoms + a as usize];
                let mut min_val = f64::INFINITY;
                let mut min_idx = 0usize;
                for (b_i, &b) in self.sel_b.indices.iter().enumerate() {
                    let pb = chunk.coords[frame * n_atoms + b as usize];
                    let mut dx = (pb[0] - pa[0]) as f64;
                    let mut dy = (pb[1] - pa[1]) as f64;
                    let mut dz = (pb[2] - pa[2]) as f64;
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                    }
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist < min_val {
                        min_val = dist;
                        min_idx = b_i;
                    }
                }
                let mapped = self.sel_b.indices[min_idx] as f32;
                self.results.push(mapped);
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: self.sel_a.indices.len(),
        })
    }
}

pub struct StripPlan {
    selection: Selection,
    results: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<GpuSelection>,
}

impl StripPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            results: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for StripPlan {
    fn name(&self) -> &'static str {
        "strip"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let selection = ctx.selection(&self.selection.indices, None)?;
                self.gpu = Some(selection);
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
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let out = ctx.gather_selection(&coords, chunk.n_atoms, chunk.n_frames, gpu)?;
            self.results.extend(out);
            return Ok(());
        }
        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            for &idx in self.selection.indices.iter() {
                let p = chunk.coords[frame * n_atoms + idx as usize];
                self.results.push(p[0]);
                self.results.push(p[1]);
                self.results.push(p[2]);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}
