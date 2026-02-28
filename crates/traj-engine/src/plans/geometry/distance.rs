use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use super::utils::*;
use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::PbcMode;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuBufferF32, GpuGroups, GpuSelection};

pub struct DistancePlan {
    sel_a: Selection,
    sel_b: Selection,
    mass_weighted: bool,
    pbc: PbcMode,
    results: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<DistanceGpuState>,
}

#[cfg(feature = "cuda")]
struct DistanceGpuState {
    groups: GpuGroups,
    masses: GpuBufferF32,
}

pub struct CenterOfMassPlan {
    selection: Selection,
    results: Vec<f32>,
    frames: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<CenterGpuState>,
}

#[cfg(feature = "cuda")]
struct CenterGpuState {
    groups: GpuGroups,
    masses: GpuBufferF32,
}

impl CenterOfMassPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            results: Vec::new(),
            frames: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for CenterOfMassPlan {
    fn name(&self) -> &'static str {
        "center_of_mass"
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
                let masses = ctx.upload_f32(&_system.atoms.mass)?;
                self.gpu = Some(CenterGpuState { groups, masses });
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
            let coms = ctx.group_com(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.groups,
                &gpu.masses,
                1.0,
            )?;
            for frame in 0..chunk.n_frames {
                let com = coms[frame];
                self.results
                    .extend_from_slice(&[com.x as f32, com.y as f32, com.z as f32]);
                self.frames += 1;
            }
            return Ok(());
        }
        let masses = &system.atoms.mass;
        for frame in 0..chunk.n_frames {
            let center = center_of_selection(chunk, frame, &self.selection.indices, masses, true);
            self.results
                .extend_from_slice(&[center[0] as f32, center[1] as f32, center[2] as f32]);
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: 3,
        })
    }
}

pub struct CenterOfGeometryPlan {
    selection: Selection,
    results: Vec<f32>,
    frames: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<CenterGpuState>,
}

impl CenterOfGeometryPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            results: Vec::new(),
            frames: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for CenterOfGeometryPlan {
    fn name(&self) -> &'static str {
        "center_of_geometry"
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
                let masses = vec![1.0f32; _system.n_atoms()];
                let masses = ctx.upload_f32(&masses)?;
                self.gpu = Some(CenterGpuState { groups, masses });
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
            let coms = ctx.group_com(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.groups,
                &gpu.masses,
                1.0,
            )?;
            for frame in 0..chunk.n_frames {
                let com = coms[frame];
                self.results
                    .extend_from_slice(&[com.x as f32, com.y as f32, com.z as f32]);
                self.frames += 1;
            }
            return Ok(());
        }
        let masses = &system.atoms.mass;
        for frame in 0..chunk.n_frames {
            let center = center_of_selection(chunk, frame, &self.selection.indices, masses, false);
            self.results
                .extend_from_slice(&[center[0] as f32, center[1] as f32, center[2] as f32]);
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: 3,
        })
    }
}

impl DistancePlan {
    pub fn new(sel_a: Selection, sel_b: Selection, mass_weighted: bool, pbc: PbcMode) -> Self {
        Self {
            sel_a,
            sel_b,
            mass_weighted,
            pbc,
            results: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for DistancePlan {
    fn name(&self) -> &'static str {
        "distance"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let mut indices =
                    Vec::with_capacity(self.sel_a.indices.len() + self.sel_b.indices.len());
                indices.extend(self.sel_a.indices.iter().copied());
                indices.extend(self.sel_b.indices.iter().copied());
                let offsets = vec![
                    0u32,
                    self.sel_a.indices.len() as u32,
                    (self.sel_a.indices.len() + self.sel_b.indices.len()) as u32,
                ];
                let max_len = self.sel_a.indices.len().max(self.sel_b.indices.len());
                let groups = ctx.groups(&offsets, &indices, max_len)?;
                let masses = if self.mass_weighted {
                    _system.atoms.mass.clone()
                } else {
                    vec![1.0f32; _system.n_atoms()]
                };
                let masses = ctx.upload_f32(&masses)?;
                self.gpu = Some(DistanceGpuState { groups, masses });
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
            let coms = ctx.group_com(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.groups,
                &gpu.masses,
                1.0,
            )?;
            let boxes = chunk_boxes(chunk, self.pbc)?;
            let out = ctx.distance_from_coms(&coms, chunk.n_frames, &boxes)?;
            self.results.extend(out);
            return Ok(());
        }
        let masses = &system.atoms.mass;
        for frame in 0..chunk.n_frames {
            let com_a = center_of_selection(
                chunk,
                frame,
                &self.sel_a.indices,
                masses,
                self.mass_weighted,
            );
            let com_b = center_of_selection(
                chunk,
                frame,
                &self.sel_b.indices,
                masses,
                self.mass_weighted,
            );
            let mut dx = com_b[0] - com_a[0];
            let mut dy = com_b[1] - com_a[1];
            let mut dz = com_b[2] - com_a[2];
            if matches!(self.pbc, PbcMode::Orthorhombic) {
                let (lx, ly, lz) = box_lengths(chunk, frame)?;
                apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
            }
            let dist = (dx * dx + dy * dy + dz * dz).sqrt() as f32;
            self.results.push(dist);
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

pub struct LowestCurvePlan {
    sel_a: Selection,
    sel_b: Selection,
    mass_weighted: bool,
    pbc: PbcMode,
    min_val: f64,
    seen: bool,
    #[cfg(feature = "cuda")]
    gpu: Option<DistanceGpuState>,
}

impl LowestCurvePlan {
    pub fn new(sel_a: Selection, sel_b: Selection, mass_weighted: bool, pbc: PbcMode) -> Self {
        Self {
            sel_a,
            sel_b,
            mass_weighted,
            pbc,
            min_val: f64::INFINITY,
            seen: false,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for LowestCurvePlan {
    fn name(&self) -> &'static str {
        "lowestcurve"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.min_val = f64::INFINITY;
        self.seen = false;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let mut indices =
                    Vec::with_capacity(self.sel_a.indices.len() + self.sel_b.indices.len());
                indices.extend(self.sel_a.indices.iter().copied());
                indices.extend(self.sel_b.indices.iter().copied());
                let offsets = vec![
                    0u32,
                    self.sel_a.indices.len() as u32,
                    (self.sel_a.indices.len() + self.sel_b.indices.len()) as u32,
                ];
                let max_len = self.sel_a.indices.len().max(self.sel_b.indices.len());
                let groups = ctx.groups(&offsets, &indices, max_len)?;
                let masses = if self.mass_weighted {
                    _system.atoms.mass.clone()
                } else {
                    vec![1.0f32; _system.n_atoms()]
                };
                let masses = ctx.upload_f32(&masses)?;
                self.gpu = Some(DistanceGpuState { groups, masses });
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
            let coms = ctx.group_com(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.groups,
                &gpu.masses,
                1.0,
            )?;
            let boxes = chunk_boxes(chunk, self.pbc)?;
            let chunk_min = ctx.distance_from_coms_min(&coms, chunk.n_frames, &boxes)?;
            if chunk_min.is_finite() {
                let val = chunk_min as f64;
                if val < self.min_val {
                    self.min_val = val;
                }
                self.seen = true;
            }
            return Ok(());
        }
        let masses = &system.atoms.mass;
        for frame in 0..chunk.n_frames {
            let com_a = center_of_selection(
                chunk,
                frame,
                &self.sel_a.indices,
                masses,
                self.mass_weighted,
            );
            let com_b = center_of_selection(
                chunk,
                frame,
                &self.sel_b.indices,
                masses,
                self.mass_weighted,
            );
            let mut dx = com_b[0] - com_a[0];
            let mut dy = com_b[1] - com_a[1];
            let mut dz = com_b[2] - com_a[2];
            if matches!(self.pbc, PbcMode::Orthorhombic) {
                let (lx, ly, lz) = box_lengths(chunk, frame)?;
                apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
            }
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist < self.min_val {
                self.min_val = dist;
            }
            self.seen = true;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if !self.seen || !self.min_val.is_finite() {
            return Ok(PlanOutput::Series(vec![0.0]));
        }
        Ok(PlanOutput::Series(vec![self.min_val as f32]))
    }
}

pub struct PairwiseDistancePlan {
    sel_a: Selection,
    sel_b: Selection,
    pbc: PbcMode,
    use_selected_input: bool,
    io_selection: Vec<u32>,
    sel_a_local: Vec<usize>,
    sel_b_local: Vec<usize>,
    results: Vec<f32>,
    frames: usize,
    pairs: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<PairwiseGpuState>,
}

#[cfg(feature = "cuda")]
struct PairwiseGpuState {
    sel_a: GpuSelection,
    sel_b: GpuSelection,
}

impl PairwiseDistancePlan {
    pub fn new(sel_a: Selection, sel_b: Selection, pbc: PbcMode) -> Self {
        let pairs = sel_a.indices.len() * sel_b.indices.len();
        let (io_selection, sel_a_local, sel_b_local) =
            build_pairwise_io_layout(&sel_a.indices, &sel_b.indices);
        Self {
            sel_a,
            sel_b,
            pbc,
            use_selected_input: false,
            io_selection,
            sel_a_local,
            sel_b_local,
            results: Vec::new(),
            frames: 0,
            pairs,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

fn build_pairwise_io_layout(sel_a: &[u32], sel_b: &[u32]) -> (Vec<u32>, Vec<usize>, Vec<usize>) {
    let mut io = Vec::<u32>::with_capacity(sel_a.len() + sel_b.len());
    let mut index =
        std::collections::HashMap::<u32, usize>::with_capacity(sel_a.len() + sel_b.len());
    for &idx in sel_a.iter().chain(sel_b.iter()) {
        if let std::collections::hash_map::Entry::Vacant(entry) = index.entry(idx) {
            let pos = io.len();
            io.push(idx);
            entry.insert(pos);
        }
    }

    let mut sel_a_local = Vec::with_capacity(sel_a.len());
    for &idx in sel_a.iter() {
        let local = *index
            .get(&idx)
            .expect("pairwise_distance io layout missing sel_a index");
        sel_a_local.push(local);
    }

    let mut sel_b_local = Vec::with_capacity(sel_b.len());
    for &idx in sel_b.iter() {
        let local = *index
            .get(&idx)
            .expect("pairwise_distance io layout missing sel_b index");
        sel_b_local.push(local);
    }

    (io, sel_a_local, sel_b_local)
}

impl Plan for PairwiseDistancePlan {
    fn name(&self) -> &'static str {
        "pairwise_distance"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        self.use_selected_input = matches!(_device, Device::Cpu);
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                self.use_selected_input = false;
                let sel_a = ctx.selection(&self.sel_a.indices, None)?;
                let sel_b = ctx.selection(&self.sel_b.indices, None)?;
                self.gpu = Some(PairwiseGpuState { sel_a, sel_b });
            }
        }
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(self.io_selection.as_slice())
        } else {
            None
        }
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        Some(self.io_selection.as_slice())
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
            let boxes = chunk_boxes(chunk, self.pbc)?;
            let out = ctx.pairwise_distance(
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
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt() as f32;
                    self.results.push(dist);
                }
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if !self.use_selected_input {
            return Err(TrajError::Mismatch(
                "pairwise_distance selected chunk received while selected IO is disabled".into(),
            ));
        }
        if source_selection != self.io_selection.as_slice() {
            return Err(TrajError::Mismatch(
                "pairwise_distance selected chunk does not match expected IO selection".into(),
            ));
        }

        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let (lx, ly, lz) = if matches!(self.pbc, PbcMode::Orthorhombic) {
                box_lengths(chunk, frame)?
            } else {
                (0.0, 0.0, 0.0)
            };
            for &a_idx in self.sel_a_local.iter() {
                let pa = chunk.coords[frame * n_atoms + a_idx];
                for &b_idx in self.sel_b_local.iter() {
                    let pb = chunk.coords[frame * n_atoms + b_idx];
                    let mut dx = (pb[0] - pa[0]) as f64;
                    let mut dy = (pb[1] - pa[1]) as f64;
                    let mut dz = (pb[2] - pa[2]) as f64;
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                    }
                    self.results
                        .push((dx * dx + dy * dy + dz * dz).sqrt() as f32);
                }
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: self.pairs,
        })
    }
}
