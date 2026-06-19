use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use super::geometry_math::*;
use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::{PbcMode, ReferenceMode};

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuBufferF32, GpuGroups, GpuSelection};

pub struct DistancePlan {
    sel_a: Selection,
    sel_b: Selection,
    io_selection: Vec<u32>,
    sel_a_local: Vec<u32>,
    sel_b_local: Vec<u32>,
    selected_masses: Vec<f32>,
    use_selected_input: bool,
    mass_weighted: bool,
    pbc: PbcMode,
    results: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<DistanceGpuState>,
}

pub struct DistanceVectorPlan {
    sel_a: Selection,
    sel_b: Selection,
    io_selection: Vec<u32>,
    sel_a_local: Vec<u32>,
    sel_b_local: Vec<u32>,
    selected_masses: Vec<f32>,
    mass_weighted: bool,
    pbc: PbcMode,
    results: Vec<f32>,
    frames: usize,
}

pub struct MultiDistancePlan {
    pairs: Vec<(Selection, Selection)>,
    io_selection: Vec<u32>,
    local_pairs: Vec<(Vec<u32>, Vec<u32>)>,
    selected_masses: Vec<f32>,
    use_selected_input: bool,
    mass_weighted: bool,
    pbc: PbcMode,
    results: Vec<f32>,
    frames: usize,
}

pub struct DistanceCenterToPointPlan {
    selection: Selection,
    selected_masses: Vec<f32>,
    point: [f64; 3],
    mass_weighted: bool,
    pbc: PbcMode,
    results: Vec<f32>,
}

pub struct DistanceCenterToReferencePlan {
    selection: Selection,
    selected_masses: Vec<f32>,
    reference_mode: ReferenceMode,
    mass_weighted: bool,
    pbc: PbcMode,
    reference_center: Option<[f64; 3]>,
    results: Vec<f32>,
}

#[cfg(feature = "cuda")]
struct DistanceGpuState {
    groups: GpuGroups,
    masses: GpuBufferF32,
}

impl DistanceCenterToPointPlan {
    pub fn new(selection: Selection, point: [f64; 3], mass_weighted: bool, pbc: PbcMode) -> Self {
        Self {
            selection,
            selected_masses: Vec::new(),
            point,
            mass_weighted,
            pbc,
            results: Vec::new(),
        }
    }
}

impl DistanceCenterToReferencePlan {
    pub fn new(
        selection: Selection,
        reference_mode: ReferenceMode,
        mass_weighted: bool,
        pbc: PbcMode,
    ) -> Self {
        Self {
            selection,
            selected_masses: Vec::new(),
            reference_mode,
            mass_weighted,
            pbc,
            reference_center: None,
            results: Vec::new(),
        }
    }
}

fn center_of_positions(
    positions: &[[f32; 4]],
    indices: &[u32],
    masses: &[f32],
    mass_weighted: bool,
) -> [f64; 3] {
    let mut center = [0.0f64; 3];
    let mut weight_sum = 0.0f64;
    for &idx in indices.iter() {
        let atom_idx = idx as usize;
        if atom_idx >= positions.len() {
            continue;
        }
        let weight = if mass_weighted {
            masses.get(atom_idx).copied().unwrap_or(1.0) as f64
        } else {
            1.0
        };
        let pos = positions[atom_idx];
        center[0] += pos[0] as f64 * weight;
        center[1] += pos[1] as f64 * weight;
        center[2] += pos[2] as f64 * weight;
        weight_sum += weight;
    }
    if weight_sum > 0.0 {
        center[0] /= weight_sum;
        center[1] /= weight_sum;
        center[2] /= weight_sum;
    }
    center
}

fn selected_masses(system: &System, selection: &[u32]) -> Vec<f32> {
    selection
        .iter()
        .map(|idx| system.atoms.mass.get(*idx as usize).copied().unwrap_or(1.0))
        .collect()
}

fn center_of_selected_chunk(
    chunk: &FrameChunk,
    frame: usize,
    local_indices: &[u32],
    masses: &[f32],
    mass_weighted: bool,
) -> [f64; 3] {
    center_of_selection(chunk, frame, local_indices, masses, mass_weighted)
}

fn center_of_all_selected_chunk(
    chunk: &FrameChunk,
    frame: usize,
    masses: &[f32],
    mass_weighted: bool,
) -> [f64; 3] {
    let n_atoms = chunk.n_atoms;
    let mut sum = [0.0f64; 3];
    let mut mass_sum = 0.0f64;
    let frame_base = frame * n_atoms;
    for atom_idx in 0..n_atoms {
        let p = chunk.coords[frame_base + atom_idx];
        let m = if mass_weighted {
            masses.get(atom_idx).copied().unwrap_or(1.0) as f64
        } else {
            1.0
        };
        sum[0] += p[0] as f64 * m;
        sum[1] += p[1] as f64 * m;
        sum[2] += p[2] as f64 * m;
        mass_sum += m;
    }
    if mass_sum == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [sum[0] / mass_sum, sum[1] / mass_sum, sum[2] / mass_sum]
}

fn distance_between_centers(
    center: [f64; 3],
    reference: [f64; 3],
    chunk: &FrameChunk,
    frame: usize,
    pbc: PbcMode,
) -> TrajResult<f32> {
    let mut dx = center[0] - reference[0];
    let mut dy = center[1] - reference[1];
    let mut dz = center[2] - reference[2];
    if matches!(pbc, PbcMode::Orthorhombic) {
        let (lx, ly, lz) = box_lengths(chunk, frame)?;
        apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
    }
    Ok((dx * dx + dy * dy + dz * dz).sqrt() as f32)
}

fn build_multi_distance_io_layout(
    pairs: &[(Selection, Selection)],
) -> (Vec<u32>, Vec<(Vec<u32>, Vec<u32>)>) {
    let mut io = Vec::<u32>::new();
    let mut index =
        std::collections::HashMap::<u32, usize>::with_capacity(pairs.len().saturating_mul(2));
    for (sel_a, sel_b) in pairs.iter() {
        for &idx in sel_a.indices.iter().chain(sel_b.indices.iter()) {
            if let std::collections::hash_map::Entry::Vacant(entry) = index.entry(idx) {
                let local = io.len();
                io.push(idx);
                entry.insert(local);
            }
        }
    }

    let local_pairs = pairs
        .iter()
        .map(|(sel_a, sel_b)| {
            let local_a = sel_a
                .indices
                .iter()
                .map(|idx| {
                    *index
                        .get(idx)
                        .expect("multi_distance io layout missing sel_a index")
                        as u32
                })
                .collect();
            let local_b = sel_b
                .indices
                .iter()
                .map(|idx| {
                    *index
                        .get(idx)
                        .expect("multi_distance io layout missing sel_b index")
                        as u32
                })
                .collect();
            (local_a, local_b)
        })
        .collect();

    (io, local_pairs)
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
        let (io_selection, sel_a_local, sel_b_local) =
            build_pairwise_io_layout(&sel_a.indices, &sel_b.indices);
        let sel_a_local = sel_a_local.into_iter().map(|idx| idx as u32).collect();
        let sel_b_local = sel_b_local.into_iter().map(|idx| idx as u32).collect();
        Self {
            sel_a,
            sel_b,
            io_selection,
            sel_a_local,
            sel_b_local,
            selected_masses: Vec::new(),
            use_selected_input: true,
            mass_weighted,
            pbc,
            results: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl DistanceVectorPlan {
    pub fn new(sel_a: Selection, sel_b: Selection, mass_weighted: bool, pbc: PbcMode) -> Self {
        let (io_selection, sel_a_local, sel_b_local) =
            build_pairwise_io_layout(&sel_a.indices, &sel_b.indices);
        let sel_a_local = sel_a_local.into_iter().map(|idx| idx as u32).collect();
        let sel_b_local = sel_b_local.into_iter().map(|idx| idx as u32).collect();
        Self {
            sel_a,
            sel_b,
            io_selection,
            sel_a_local,
            sel_b_local,
            selected_masses: Vec::new(),
            mass_weighted,
            pbc,
            results: Vec::new(),
            frames: 0,
        }
    }
}

impl MultiDistancePlan {
    pub fn new(pairs: Vec<(Selection, Selection)>, mass_weighted: bool, pbc: PbcMode) -> Self {
        let (io_selection, local_pairs) = build_multi_distance_io_layout(&pairs);
        Self {
            pairs,
            io_selection,
            local_pairs,
            selected_masses: Vec::new(),
            use_selected_input: true,
            mass_weighted,
            pbc,
            results: Vec::new(),
            frames: 0,
        }
    }
}

impl Plan for DistancePlan {
    fn name(&self) -> &'static str {
        "distance"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.selected_masses = selected_masses(_system, &self.io_selection);
        self.use_selected_input = matches!(_device, Device::Cpu);
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                self.use_selected_input = false;
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

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if source_selection != self.io_selection.as_slice() {
            return Err(TrajError::Mismatch(
                "distance selected chunk does not match expected IO selection".into(),
            ));
        }
        for frame in 0..chunk.n_frames {
            let com_a = center_of_selected_chunk(
                chunk,
                frame,
                &self.sel_a_local,
                &self.selected_masses,
                self.mass_weighted,
            );
            let com_b = center_of_selected_chunk(
                chunk,
                frame,
                &self.sel_b_local,
                &self.selected_masses,
                self.mass_weighted,
            );
            let mut dx = com_b[0] - com_a[0];
            let mut dy = com_b[1] - com_a[1];
            let mut dz = com_b[2] - com_a[2];
            if matches!(self.pbc, PbcMode::Orthorhombic) {
                let (lx, ly, lz) = box_lengths(chunk, frame)?;
                apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
            }
            self.results
                .push((dx * dx + dy * dy + dz * dz).sqrt() as f32);
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

impl Plan for DistanceVectorPlan {
    fn name(&self) -> &'static str {
        "distance_vector"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        self.selected_masses = selected_masses(system, &self.io_selection);
        let n_atoms = system.n_atoms() as u32;
        for &idx in self.sel_a.indices.iter().chain(self.sel_b.indices.iter()) {
            if idx >= n_atoms {
                return Err(TrajError::Mismatch(
                    "distance_vector atom index out of range".into(),
                ));
            }
        }
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        Some(self.io_selection.as_slice())
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        Some(self.io_selection.as_slice())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
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
            self.results
                .extend_from_slice(&[dx as f32, dy as f32, dz as f32]);
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
        if source_selection != self.io_selection.as_slice() {
            return Err(TrajError::Mismatch(
                "distance_vector selected chunk does not match expected IO selection".into(),
            ));
        }
        for frame in 0..chunk.n_frames {
            let com_a = center_of_selected_chunk(
                chunk,
                frame,
                &self.sel_a_local,
                &self.selected_masses,
                self.mass_weighted,
            );
            let com_b = center_of_selected_chunk(
                chunk,
                frame,
                &self.sel_b_local,
                &self.selected_masses,
                self.mass_weighted,
            );
            let mut dx = com_b[0] - com_a[0];
            let mut dy = com_b[1] - com_a[1];
            let mut dz = com_b[2] - com_a[2];
            if matches!(self.pbc, PbcMode::Orthorhombic) {
                let (lx, ly, lz) = box_lengths(chunk, frame)?;
                apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
            }
            self.results
                .extend_from_slice(&[dx as f32, dy as f32, dz as f32]);
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

impl Plan for MultiDistancePlan {
    fn name(&self) -> &'static str {
        "multi_distance"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        self.selected_masses = selected_masses(system, &self.io_selection);
        self.use_selected_input = true;
        let n_atoms = system.n_atoms() as u32;
        for (sel_a, sel_b) in self.pairs.iter() {
            for &idx in sel_a.indices.iter().chain(sel_b.indices.iter()) {
                if idx >= n_atoms {
                    return Err(TrajError::Mismatch(
                        "multi_distance atom index out of range".into(),
                    ));
                }
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
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let masses = &system.atoms.mass;
        for frame in 0..chunk.n_frames {
            let (lx, ly, lz) = if matches!(self.pbc, PbcMode::Orthorhombic) {
                box_lengths(chunk, frame)?
            } else {
                (0.0, 0.0, 0.0)
            };
            for (sel_a, sel_b) in self.pairs.iter() {
                let com_a =
                    center_of_selection(chunk, frame, &sel_a.indices, masses, self.mass_weighted);
                let com_b =
                    center_of_selection(chunk, frame, &sel_b.indices, masses, self.mass_weighted);
                let mut dx = com_b[0] - com_a[0];
                let mut dy = com_b[1] - com_a[1];
                let mut dz = com_b[2] - com_a[2];
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                }
                self.results
                    .push((dx * dx + dy * dy + dz * dz).sqrt() as f32);
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
        if source_selection != self.io_selection.as_slice() {
            return Err(TrajError::Mismatch(
                "multi_distance selected chunk does not match expected IO selection".into(),
            ));
        }
        for frame in 0..chunk.n_frames {
            let (lx, ly, lz) = if matches!(self.pbc, PbcMode::Orthorhombic) {
                box_lengths(chunk, frame)?
            } else {
                (0.0, 0.0, 0.0)
            };
            for (sel_a, sel_b) in self.local_pairs.iter() {
                let com_a = center_of_selected_chunk(
                    chunk,
                    frame,
                    sel_a,
                    &self.selected_masses,
                    self.mass_weighted,
                );
                let com_b = center_of_selected_chunk(
                    chunk,
                    frame,
                    sel_b,
                    &self.selected_masses,
                    self.mass_weighted,
                );
                let mut dx = com_b[0] - com_a[0];
                let mut dy = com_b[1] - com_a[1];
                let mut dz = com_b[2] - com_a[2];
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                }
                self.results
                    .push((dx * dx + dy * dy + dz * dz).sqrt() as f32);
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: self.pairs.len(),
        })
    }
}

impl Plan for DistanceCenterToPointPlan {
    fn name(&self) -> &'static str {
        "distance_center_to_point"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.selected_masses = selected_masses(_system, &self.selection.indices);
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        Some(&self.selection.indices)
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        Some(&self.selection.indices)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let masses = &system.atoms.mass;
        for frame in 0..chunk.n_frames {
            let center = center_of_selection(
                chunk,
                frame,
                &self.selection.indices,
                masses,
                self.mass_weighted,
            );
            let dist = distance_between_centers(center, self.point, chunk, frame, self.pbc)?;
            self.results.push(dist);
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
        if source_selection != self.selection.indices.as_ref().as_slice() {
            return Err(TrajError::Mismatch(
                "distance_center_to_point selected chunk does not match expected selection".into(),
            ));
        }
        for frame in 0..chunk.n_frames {
            let center = center_of_all_selected_chunk(
                chunk,
                frame,
                &self.selected_masses,
                self.mass_weighted,
            );
            let dist = distance_between_centers(center, self.point, chunk, frame, self.pbc)?;
            self.results.push(dist);
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

impl Plan for DistanceCenterToReferencePlan {
    fn name(&self) -> &'static str {
        "distance_center_to_reference"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.selected_masses = selected_masses(system, &self.selection.indices);
        self.reference_center = match self.reference_mode {
            ReferenceMode::Topology => {
                let positions0 = system
                    .positions0
                    .as_ref()
                    .ok_or_else(|| TrajError::Mismatch("no topology reference coords".into()))?;
                Some(center_of_positions(
                    positions0,
                    &self.selection.indices,
                    &system.atoms.mass,
                    self.mass_weighted,
                ))
            }
            ReferenceMode::Frame0 => None,
        };
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        Some(&self.selection.indices)
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        Some(&self.selection.indices)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.reference_center.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0) {
            if chunk.n_frames == 0 {
                return Ok(());
            }
            self.reference_center = Some(center_of_selection(
                chunk,
                0,
                &self.selection.indices,
                &system.atoms.mass,
                self.mass_weighted,
            ));
        }
        let reference = self
            .reference_center
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;
        for frame in 0..chunk.n_frames {
            let center = center_of_selection(
                chunk,
                frame,
                &self.selection.indices,
                &system.atoms.mass,
                self.mass_weighted,
            );
            let dist = distance_between_centers(center, reference, chunk, frame, self.pbc)?;
            self.results.push(dist);
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
        if source_selection != self.selection.indices.as_ref().as_slice() {
            return Err(TrajError::Mismatch(
                "distance_center_to_reference selected chunk does not match expected selection"
                    .into(),
            ));
        }
        if self.reference_center.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0) {
            if chunk.n_frames == 0 {
                return Ok(());
            }
            self.reference_center = Some(center_of_all_selected_chunk(
                chunk,
                0,
                &self.selected_masses,
                self.mass_weighted,
            ));
        }
        let reference = self
            .reference_center
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;
        for frame in 0..chunk.n_frames {
            let center = center_of_all_selected_chunk(
                chunk,
                frame,
                &self.selected_masses,
                self.mass_weighted,
            );
            let dist = distance_between_centers(center, reference, chunk, frame, self.pbc)?;
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

pub struct PairListDistancePlan {
    pairs: Vec<(u32, u32)>,
    pbc: PbcMode,
    results: Vec<f32>,
    frames: usize,
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

impl PairListDistancePlan {
    pub fn new(pairs: Vec<(u32, u32)>, pbc: PbcMode) -> Self {
        Self {
            pairs,
            pbc,
            results: Vec::new(),
            frames: 0,
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

impl Plan for PairListDistancePlan {
    fn name(&self) -> &'static str {
        "pair_list_distance"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        let n_atoms = system.n_atoms() as u32;
        for &(left, right) in self.pairs.iter() {
            if left >= n_atoms || right >= n_atoms {
                return Err(TrajError::Mismatch(
                    "pair_list_distance atom index out of range".into(),
                ));
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
        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let (lx, ly, lz) = if matches!(self.pbc, PbcMode::Orthorhombic) {
                box_lengths(chunk, frame)?
            } else {
                (0.0, 0.0, 0.0)
            };
            let frame_base = frame * n_atoms;
            for &(left, right) in self.pairs.iter() {
                let pa = chunk.coords[frame_base + left as usize];
                let pb = chunk.coords[frame_base + right as usize];
                let mut dx = (pb[0] - pa[0]) as f64;
                let mut dy = (pb[1] - pa[1]) as f64;
                let mut dz = (pb[2] - pa[2]) as f64;
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                }
                self.results
                    .push((dx * dx + dy * dy + dz * dz).sqrt() as f32);
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: self.pairs.len(),
        })
    }
}
