use traj_core::error::TrajResult;
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use super::geometry_math::*;
use crate::executor::{Device, Plan, PlanOutput, PlanRequirements, TrajectoryOutput};
use crate::plans::PbcMode;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuSelection};

#[cfg(feature = "cuda")]
struct PairwiseGpuState {
    sel_a: GpuSelection,
    sel_b: GpuSelection,
}

pub struct ClosestPlan {
    target: Selection,
    probe: Selection,
    n_solvents: usize,
    pbc: PbcMode,
    results: Vec<f32>,
    frames: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<PairwiseGpuState>,
}

impl ClosestPlan {
    pub fn new(target: Selection, probe: Selection, n_solvents: usize, pbc: PbcMode) -> Self {
        Self {
            target,
            probe,
            n_solvents,
            pbc,
            results: Vec::new(),
            frames: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for ClosestPlan {
    fn name(&self) -> &'static str {
        "closest"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(matches!(self.pbc, PbcMode::Orthorhombic), false)
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
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
        if self.n_solvents == 0 {
            self.frames += chunk.n_frames;
            return Ok(());
        }
        let n_a = self.target.indices.len();
        let n_b = self.probe.indices.len();
        if n_a == 0 || n_b == 0 {
            self.results
                .extend(std::iter::repeat(-1.0).take(chunk.n_frames * self.n_solvents));
            self.frames += chunk.n_frames;
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let boxes = chunk_boxes(chunk, self.pbc)?;
            let out = ctx.closest_topk(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.sel_a,
                &gpu.sel_b,
                &boxes,
                self.n_solvents,
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
            let mut mins = Vec::with_capacity(n_b);
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
                mins.push((min_val, b));
            }
            mins.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            for k in 0..self.n_solvents {
                if k < mins.len() {
                    self.results.push(mins[k].1 as f32);
                } else {
                    self.results.push(-1.0);
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
            cols: self.n_solvents,
        })
    }
}

pub struct ClosestCoordsPlan {
    target: Selection,
    probe: Selection,
    n_solvents: usize,
    pbc: PbcMode,
    preferred_selection: Vec<u32>,
    local_target: Vec<u32>,
    local_probe: Vec<u32>,
    results: Vec<f32>,
    box_: Vec<Box3>,
    time: Vec<f32>,
    saw_time: bool,
    frames: usize,
}

impl ClosestCoordsPlan {
    pub fn new(target: Selection, probe: Selection, n_solvents: usize, pbc: PbcMode) -> Self {
        Self {
            target,
            probe,
            n_solvents,
            pbc,
            preferred_selection: Vec::new(),
            local_target: Vec::new(),
            local_probe: Vec::new(),
            results: Vec::new(),
            box_: Vec::new(),
            time: Vec::new(),
            saw_time: false,
            frames: 0,
        }
    }
}

impl Plan for ClosestCoordsPlan {
    fn name(&self) -> &'static str {
        "closest_coords"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(true, true)
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.preferred_selection.clear();
        for &idx in self.target.indices.iter().chain(self.probe.indices.iter()) {
            if !self.preferred_selection.contains(&idx) {
                self.preferred_selection.push(idx);
            }
        }
        self.local_target = local_indices(&self.target.indices, &self.preferred_selection);
        self.local_probe = local_indices(&self.probe.indices, &self.preferred_selection);
        self.results.clear();
        self.box_.clear();
        self.time.clear();
        self.saw_time = false;
        self.frames = 0;
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.preferred_selection.is_empty() {
            None
        } else {
            Some(&self.preferred_selection)
        }
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let target = self.target.indices.clone();
        let probe = self.probe.indices.clone();
        self.process_coords_chunk(chunk, &target, &probe)
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        _source_selection: &[u32],
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let target = self.local_target.clone();
        let probe = self.local_probe.clone();
        self.process_coords_chunk(chunk, &target, &probe)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Trajectory(TrajectoryOutput {
            coords: std::mem::take(&mut self.results),
            frames: self.frames,
            atoms: self.n_solvents,
            box_: std::mem::take(&mut self.box_),
            time: if self.saw_time {
                std::mem::take(&mut self.time)
            } else {
                Vec::new()
            },
        }))
    }
}

impl ClosestCoordsPlan {
    fn process_coords_chunk(
        &mut self,
        chunk: &FrameChunk,
        target_indices: &[u32],
        probe_indices: &[u32],
    ) -> TrajResult<()> {
        if self.n_solvents == 0 {
            self.record_chunk_metadata(chunk);
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
            let mut mins = Vec::with_capacity(probe_indices.len());
            for &probe in probe_indices.iter() {
                let pb = chunk.coords[frame * n_atoms + probe as usize];
                let mut min_val = f64::INFINITY;
                for &target in target_indices.iter() {
                    let pa = chunk.coords[frame * n_atoms + target as usize];
                    let mut dx = (pb[0] - pa[0]) as f64;
                    let mut dy = (pb[1] - pa[1]) as f64;
                    let mut dz = (pb[2] - pa[2]) as f64;
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                    }
                    let dist = dx * dx + dy * dy + dz * dz;
                    if dist < min_val {
                        min_val = dist;
                    }
                }
                mins.push((min_val, probe));
            }
            mins.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.1.cmp(&b.1))
            });
            for k in 0..self.n_solvents {
                if k < mins.len() && mins[k].0.is_finite() {
                    let p = chunk.coords[frame * n_atoms + mins[k].1 as usize];
                    self.results.extend_from_slice(&[p[0], p[1], p[2]]);
                } else {
                    self.results.extend_from_slice(&[0.0, 0.0, 0.0]);
                }
            }
            if let Some(box_) = chunk.box_.get(frame) {
                self.box_.push(*box_);
            }
            if let Some(times) = &chunk.time_ps {
                if let Some(value) = times.get(frame) {
                    self.time.push(*value);
                    self.saw_time = true;
                }
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn record_chunk_metadata(&mut self, chunk: &FrameChunk) {
        self.box_
            .extend(chunk.box_.iter().take(chunk.n_frames).copied());
        if let Some(times) = &chunk.time_ps {
            self.time.extend(times.iter().take(chunk.n_frames).copied());
            if !times.is_empty() {
                self.saw_time = true;
            }
        }
    }
}

fn local_indices(indices: &[u32], preferred_selection: &[u32]) -> Vec<u32> {
    indices
        .iter()
        .filter_map(|idx| {
            preferred_selection
                .iter()
                .position(|candidate| candidate == idx)
                .map(|local| local as u32)
        })
        .collect()
}
