use traj_core::error::TrajResult;
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use super::geometry_math::*;
use crate::executor::{Device, Plan, PlanOutput, PlanRequirements, TrajectoryOutput};
use crate::plans::PbcMode;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuSelection};

pub struct VectorPlan {
    sel_a: Selection,
    sel_b: Selection,
    mass_weighted: bool,
    pbc: PbcMode,
    preferred_selection: Vec<u32>,
    local_a: Vec<u32>,
    local_b: Vec<u32>,
    results: Vec<f32>,
    frames: usize,
}

#[derive(Clone)]
pub enum MultiVectorCommand {
    Between {
        sel_a: Selection,
        sel_b: Selection,
        mass_weighted: bool,
        pbc: PbcMode,
    },
    Center {
        selection: Selection,
        mass_weighted: bool,
    },
}

enum LocalMultiVectorCommand {
    Between {
        local_a: Vec<u32>,
        local_b: Vec<u32>,
        mass_weighted: bool,
        pbc: PbcMode,
    },
    Center {
        local: Vec<u32>,
        mass_weighted: bool,
    },
}

pub struct MultiVectorPlan {
    commands: Vec<MultiVectorCommand>,
    preferred_selection: Vec<u32>,
    local_commands: Vec<LocalMultiVectorCommand>,
    results: Vec<f32>,
    frames: usize,
    needs_box: bool,
}

impl VectorPlan {
    pub fn new(sel_a: Selection, sel_b: Selection, mass_weighted: bool, pbc: PbcMode) -> Self {
        Self {
            sel_a,
            sel_b,
            mass_weighted,
            pbc,
            preferred_selection: Vec::new(),
            local_a: Vec::new(),
            local_b: Vec::new(),
            results: Vec::new(),
            frames: 0,
        }
    }
}

impl MultiVectorPlan {
    pub fn new(commands: Vec<MultiVectorCommand>) -> Self {
        let needs_box = commands.iter().any(|command| match command {
            MultiVectorCommand::Between { pbc, .. } => matches!(pbc, PbcMode::Orthorhombic),
            MultiVectorCommand::Center { .. } => false,
        });
        Self {
            commands,
            preferred_selection: Vec::new(),
            local_commands: Vec::new(),
            results: Vec::new(),
            frames: 0,
            needs_box,
        }
    }

    fn add_preferred(&mut self, indices: &[u32]) {
        for &idx in indices {
            if !self.preferred_selection.contains(&idx) {
                self.preferred_selection.push(idx);
            }
        }
    }
}

impl Plan for VectorPlan {
    fn name(&self) -> &'static str {
        "vector"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(matches!(self.pbc, PbcMode::Orthorhombic), false)
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.preferred_selection.clear();
        for &idx in self.sel_a.indices.iter().chain(self.sel_b.indices.iter()) {
            if !self.preferred_selection.contains(&idx) {
                self.preferred_selection.push(idx);
            }
        }
        self.local_a = local_indices(&self.sel_a.indices, &self.preferred_selection);
        self.local_b = local_indices(&self.sel_b.indices, &self.preferred_selection);
        self.results.clear();
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
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let masses = &system.atoms.mass;
        for frame in 0..chunk.n_frames {
            let com_a = center_of_local_selection(
                chunk,
                frame,
                &self.local_a,
                source_selection,
                masses,
                self.mass_weighted,
            );
            let com_b = center_of_local_selection(
                chunk,
                frame,
                &self.local_b,
                source_selection,
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

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: 3,
        })
    }
}

impl Plan for MultiVectorPlan {
    fn name(&self) -> &'static str {
        "multi_vector"
    }

    fn set_frames_hint(&mut self, n_frames: Option<usize>) {
        if let Some(frames) = n_frames {
            if let Some(values) = frames
                .checked_mul(self.commands.len())
                .and_then(|n| n.checked_mul(3))
            {
                self.results.reserve(values);
            }
        }
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(self.needs_box, false)
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.preferred_selection.clear();
        let commands = self.commands.clone();
        for command in commands.iter() {
            match command {
                MultiVectorCommand::Between { sel_a, sel_b, .. } => {
                    self.add_preferred(&sel_a.indices);
                    self.add_preferred(&sel_b.indices);
                }
                MultiVectorCommand::Center { selection, .. } => {
                    self.add_preferred(&selection.indices);
                }
            }
        }
        self.local_commands = self
            .commands
            .iter()
            .map(|command| match command {
                MultiVectorCommand::Between {
                    sel_a,
                    sel_b,
                    mass_weighted,
                    pbc,
                } => LocalMultiVectorCommand::Between {
                    local_a: local_indices(&sel_a.indices, &self.preferred_selection),
                    local_b: local_indices(&sel_b.indices, &self.preferred_selection),
                    mass_weighted: *mass_weighted,
                    pbc: *pbc,
                },
                MultiVectorCommand::Center {
                    selection,
                    mass_weighted,
                } => LocalMultiVectorCommand::Center {
                    local: local_indices(&selection.indices, &self.preferred_selection),
                    mass_weighted: *mass_weighted,
                },
            })
            .collect();
        self.results.clear();
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
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let masses = &system.atoms.mass;
        for frame in 0..chunk.n_frames {
            for command in self.commands.iter() {
                match command {
                    MultiVectorCommand::Between {
                        sel_a,
                        sel_b,
                        mass_weighted,
                        pbc,
                    } => {
                        let com_a = center_of_selection(
                            chunk,
                            frame,
                            &sel_a.indices,
                            masses,
                            *mass_weighted,
                        );
                        let com_b = center_of_selection(
                            chunk,
                            frame,
                            &sel_b.indices,
                            masses,
                            *mass_weighted,
                        );
                        push_between_vector(&mut self.results, chunk, frame, com_a, com_b, *pbc)?;
                    }
                    MultiVectorCommand::Center {
                        selection,
                        mass_weighted,
                    } => {
                        let center = center_of_selection(
                            chunk,
                            frame,
                            &selection.indices,
                            masses,
                            *mass_weighted,
                        );
                        self.results.extend_from_slice(&[
                            center[0] as f32,
                            center[1] as f32,
                            center[2] as f32,
                        ]);
                    }
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
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let masses = &system.atoms.mass;
        for frame in 0..chunk.n_frames {
            for command in self.local_commands.iter() {
                match command {
                    LocalMultiVectorCommand::Between {
                        local_a,
                        local_b,
                        mass_weighted,
                        pbc,
                    } => {
                        let com_a = center_of_local_selection(
                            chunk,
                            frame,
                            local_a,
                            source_selection,
                            masses,
                            *mass_weighted,
                        );
                        let com_b = center_of_local_selection(
                            chunk,
                            frame,
                            local_b,
                            source_selection,
                            masses,
                            *mass_weighted,
                        );
                        push_between_vector(&mut self.results, chunk, frame, com_a, com_b, *pbc)?;
                    }
                    LocalMultiVectorCommand::Center {
                        local,
                        mass_weighted,
                    } => {
                        let center = center_of_local_selection(
                            chunk,
                            frame,
                            local,
                            source_selection,
                            masses,
                            *mass_weighted,
                        );
                        self.results.extend_from_slice(&[
                            center[0] as f32,
                            center[1] as f32,
                            center[2] as f32,
                        ]);
                    }
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
            cols: self.commands.len() * 3,
        })
    }
}

fn push_between_vector(
    results: &mut Vec<f32>,
    chunk: &FrameChunk,
    frame: usize,
    com_a: [f64; 3],
    com_b: [f64; 3],
    pbc: PbcMode,
) -> TrajResult<()> {
    let mut dx = com_b[0] - com_a[0];
    let mut dy = com_b[1] - com_a[1];
    let mut dz = com_b[2] - com_a[2];
    if matches!(pbc, PbcMode::Orthorhombic) {
        let (lx, ly, lz) = box_lengths(chunk, frame)?;
        apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
    }
    results.extend_from_slice(&[dx as f32, dy as f32, dz as f32]);
    Ok(())
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

fn center_of_local_selection(
    chunk: &FrameChunk,
    frame: usize,
    local_indices: &[u32],
    source_selection: &[u32],
    masses: &[f32],
    mass_weighted: bool,
) -> [f64; 3] {
    if local_indices.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let frame_base = frame * chunk.n_atoms;
    let mut sum = [0.0f64; 3];
    let mut weight_sum = 0.0f64;
    for &local_idx in local_indices {
        let local = local_idx as usize;
        let p = chunk.coords[frame_base + local];
        let weight = if mass_weighted {
            source_selection
                .get(local)
                .and_then(|idx| masses.get(*idx as usize))
                .copied()
                .unwrap_or(1.0)
                .max(0.0) as f64
        } else {
            1.0
        };
        sum[0] += p[0] as f64 * weight;
        sum[1] += p[1] as f64 * weight;
        sum[2] += p[2] as f64 * weight;
        weight_sum += weight;
    }
    if weight_sum <= 0.0 {
        let mut unweighted = [0.0f64; 3];
        for &local_idx in local_indices {
            let p = chunk.coords[frame_base + local_idx as usize];
            unweighted[0] += p[0] as f64;
            unweighted[1] += p[1] as f64;
            unweighted[2] += p[2] as f64;
        }
        let inv = 1.0 / local_indices.len() as f64;
        return [
            unweighted[0] * inv,
            unweighted[1] * inv,
            unweighted[2] * inv,
        ];
    }
    [
        sum[0] / weight_sum,
        sum[1] / weight_sum,
        sum[2] / weight_sum,
    ]
}

pub struct GetVelocityPlan {
    selection: Selection,
    results: Vec<f32>,
    frames: usize,
    prev_coords: Vec<[f32; 3]>,
    prev_time: Option<f64>,
    has_prev: bool,
    time_scale: f64,
}

pub struct RunningAveragePlan {
    selection: Selection,
    window: usize,
    sums: Vec<f64>,
    ring: Vec<Vec<f32>>,
    ring_pos: usize,
    ring_len: usize,
    results: Vec<f32>,
    frames: usize,
}

impl RunningAveragePlan {
    pub fn new(selection: Selection, window: usize) -> Self {
        Self {
            selection,
            window,
            sums: Vec::new(),
            ring: Vec::new(),
            ring_pos: 0,
            ring_len: 0,
            results: Vec::new(),
            frames: 0,
        }
    }
}

impl Plan for RunningAveragePlan {
    fn name(&self) -> &'static str {
        "runningavg"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        let len = self.selection.indices.len() * 3;
        self.sums.clear();
        self.sums.resize(len, 0.0);
        self.ring.clear();
        self.ring_pos = 0;
        self.ring_len = 0;
        self.results.clear();
        self.frames = 0;
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        Some(&self.selection.indices)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        let sel = self.selection.indices.clone();
        for frame in 0..chunk.n_frames {
            let frame_base = frame * n_atoms;
            self.push_frame(|local| chunk.coords[frame_base + sel[local] as usize]);
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
        for frame in 0..chunk.n_frames {
            let frame_base = frame * chunk.n_atoms;
            self.push_frame(|local| chunk.coords[frame_base + local]);
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: self.selection.indices.len() * 3,
        })
    }
}

impl RunningAveragePlan {
    fn push_frame<P>(&mut self, point: P)
    where
        P: Fn(usize) -> [f32; 4],
    {
        let n_selected = self.selection.indices.len();
        let mut current = vec![0.0f32; n_selected * 3];
        for local in 0..n_selected {
            let pos = point(local);
            let base = local * 3;
            current[base] = pos[0];
            current[base + 1] = pos[1];
            current[base + 2] = pos[2];
        }

        let denom = if self.window == 0 {
            for (sum, value) in self.sums.iter_mut().zip(current.iter()) {
                *sum += *value as f64;
            }
            self.frames + 1
        } else {
            if self.ring_len < self.window {
                for (sum, value) in self.sums.iter_mut().zip(current.iter()) {
                    *sum += *value as f64;
                }
                self.ring.push(current);
                self.ring_len += 1;
            } else {
                let old = &mut self.ring[self.ring_pos];
                for i in 0..self.sums.len() {
                    self.sums[i] -= old[i] as f64;
                    self.sums[i] += current[i] as f64;
                    old[i] = current[i];
                }
                self.ring_pos = (self.ring_pos + 1) % self.window;
            }
            self.ring_len
        };

        let denom = denom.max(1) as f64;
        for sum in self.sums.iter() {
            self.results.push((*sum / denom) as f32);
        }
        self.frames += 1;
    }
}

pub struct RunningAverageTrajectoryPlan {
    inner: RunningAveragePlan,
    box_: Vec<Box3>,
    time: Vec<f32>,
    saw_time: bool,
    frames: usize,
}

impl RunningAverageTrajectoryPlan {
    pub fn new(selection: Selection, window: usize) -> Self {
        Self {
            inner: RunningAveragePlan::new(selection, window),
            box_: Vec::new(),
            time: Vec::new(),
            saw_time: false,
            frames: 0,
        }
    }

    fn record_metadata(&mut self, chunk: &FrameChunk) {
        self.box_
            .extend(chunk.box_.iter().take(chunk.n_frames).copied());
        if let Some(time) = &chunk.time_ps {
            self.time.extend(time.iter().take(chunk.n_frames).copied());
            self.saw_time |= !time.is_empty();
        }
    }
}

impl Plan for RunningAverageTrajectoryPlan {
    fn name(&self) -> &'static str {
        "runningavg_trajectory"
    }

    fn set_frames_hint(&mut self, n_frames: Option<usize>) {
        if let Some(frames) = n_frames {
            self.box_.reserve(frames);
            self.time.reserve(frames);
            if let Some(values) = frames
                .checked_mul(self.inner.selection.indices.len())
                .and_then(|n| n.checked_mul(3))
            {
                self.inner.results.reserve(values);
            }
        }
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(true, true)
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.inner.init(system, device)?;
        self.box_.clear();
        self.time.clear();
        self.saw_time = false;
        self.frames = 0;
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        self.inner.preferred_selection()
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner.process_chunk(chunk, system, device)?;
        self.record_metadata(chunk);
        self.frames += chunk.n_frames;
        Ok(())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner
            .process_chunk_selected(chunk, source_selection, system, device)?;
        self.record_metadata(chunk);
        self.frames += chunk.n_frames;
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Trajectory(TrajectoryOutput {
            coords: std::mem::take(&mut self.inner.results),
            frames: self.frames,
            atoms: self.inner.selection.indices.len(),
            box_: std::mem::take(&mut self.box_),
            time: if self.saw_time {
                std::mem::take(&mut self.time)
            } else {
                Vec::new()
            },
        }))
    }
}

impl GetVelocityPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            results: Vec::new(),
            frames: 0,
            prev_coords: Vec::new(),
            prev_time: None,
            has_prev: false,
            time_scale: 1.0,
        }
    }
}

impl Plan for GetVelocityPlan {
    fn name(&self) -> &'static str {
        "get_velocity"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, true)
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        let len = self.selection.indices.len();
        self.results.clear();
        self.frames = 0;
        self.prev_coords.clear();
        self.prev_coords.resize(len, [0.0; 3]);
        self.prev_time = None;
        self.has_prev = false;
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        Some(&self.selection.indices)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.prev_coords.len() != self.selection.indices.len() {
            self.prev_coords
                .resize(self.selection.indices.len(), [0.0; 3]);
            self.has_prev = false;
            self.prev_time = None;
        }
        let sel = self.selection.indices.clone();
        for frame in 0..chunk.n_frames {
            let time = chunk
                .time_ps
                .as_ref()
                .and_then(|times| times.get(frame))
                .map(|t| *t as f64);
            let dt = self.frame_dt(time);
            let frame_base = frame * chunk.n_atoms;
            for (local, &idx) in sel.iter().enumerate() {
                let p = chunk.coords[frame_base + idx as usize];
                self.push_velocity(local, p, dt);
            }
            if let Some(t) = time {
                self.prev_time = Some(t);
            }
            self.has_prev = true;
            self.frames += 1;
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
        if self.prev_coords.len() != self.selection.indices.len() {
            self.prev_coords
                .resize(self.selection.indices.len(), [0.0; 3]);
            self.has_prev = false;
            self.prev_time = None;
        }
        for frame in 0..chunk.n_frames {
            let time = chunk
                .time_ps
                .as_ref()
                .and_then(|times| times.get(frame))
                .map(|t| *t as f64);
            let dt = self.frame_dt(time);
            let frame_base = frame * chunk.n_atoms;
            for local in 0..self.selection.indices.len() {
                let p = chunk.coords[frame_base + local];
                self.push_velocity(local, p, dt);
            }
            if let Some(t) = time {
                self.prev_time = Some(t);
            }
            self.has_prev = true;
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: self.selection.indices.len() * 3,
        })
    }
}

impl GetVelocityPlan {
    fn frame_dt(&self, time: Option<f64>) -> f64 {
        let mut dt = match (self.prev_time, time) {
            (Some(prev), Some(cur)) if cur > prev => (cur - prev) * self.time_scale,
            _ => 1.0 * self.time_scale,
        };
        if !dt.is_finite() || dt == 0.0 {
            dt = 1.0 * self.time_scale;
        }
        dt
    }

    fn push_velocity(&mut self, local: usize, p: [f32; 4], dt: f64) {
        if self.has_prev {
            let prev = self.prev_coords[local];
            let vx = (p[0] as f64 - prev[0] as f64) / dt;
            let vy = (p[1] as f64 - prev[1] as f64) / dt;
            let vz = (p[2] as f64 - prev[2] as f64) / dt;
            self.results
                .extend_from_slice(&[vx as f32, vy as f32, vz as f32]);
        } else {
            self.results.extend_from_slice(&[0.0, 0.0, 0.0]);
        }
        self.prev_coords[local] = [p[0], p[1], p[2]];
    }
}

pub struct SetVelocityPlan {
    selection: Selection,
    temperature: f64,
    seed: u64,
    results: Vec<f32>,
    frames: usize,
    state: u64,
}

impl SetVelocityPlan {
    pub fn new(selection: Selection, temperature: f64, seed: u64) -> Self {
        Self {
            selection,
            temperature,
            seed,
            results: Vec::new(),
            frames: 0,
            state: seed,
        }
    }
}

impl Plan for SetVelocityPlan {
    fn name(&self) -> &'static str {
        "set_velocity"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        self.state = self.seed;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let masses = &system.atoms.mass;
        let sel = &self.selection.indices;
        let base_temp = self.temperature.max(0.0);
        for _frame in 0..chunk.n_frames {
            for &idx in sel.iter() {
                let atom_idx = idx as usize;
                let mass = masses.get(atom_idx).copied().unwrap_or(0.0).max(0.0) as f64;
                let sigma = if mass > 0.0 {
                    (base_temp / mass).sqrt()
                } else {
                    0.0
                };
                let (g0, g1) = gaussian_pair(&mut self.state);
                let (g2, _) = gaussian_pair(&mut self.state);
                self.results.push((g0 * sigma) as f32);
                self.results.push((g1 * sigma) as f32);
                self.results.push((g2 * sigma) as f32);
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: self.selection.indices.len() * 3,
        })
    }
}

pub struct MeanStructurePlan {
    selection: Selection,
    sums: Vec<f64>,
    frames: usize,
    use_selected_input: bool,
    #[cfg(feature = "cuda")]
    gpu: Option<GpuSelection>,
}

impl MeanStructurePlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            sums: Vec::new(),
            frames: 0,
            use_selected_input: false,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for MeanStructurePlan {
    fn name(&self) -> &'static str {
        "mean_structure"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        let len = self.selection.indices.len() * 3;
        self.sums.clear();
        self.sums.resize(len, 0.0);
        self.frames = 0;
        self.use_selected_input = matches!(_device, Device::Cpu);
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                self.use_selected_input = false;
                let selection = ctx.selection(&self.selection.indices, None)?;
                self.gpu = Some(selection);
            }
        }
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(&self.selection.indices)
        } else {
            None
        }
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
            let accum = ctx.mean_structure_accum(&coords, chunk.n_atoms, chunk.n_frames, gpu)?;
            for i in 0..self.selection.indices.len() {
                let base = i * 3;
                self.sums[base] += accum.sum_x[i] as f64;
                self.sums[base + 1] += accum.sum_y[i] as f64;
                self.sums[base + 2] += accum.sum_z[i] as f64;
            }
            self.frames += chunk.n_frames;
            return Ok(());
        }
        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        for frame in 0..chunk.n_frames {
            let frame_base = frame * n_atoms;
            for (i, &idx) in sel.iter().enumerate() {
                let atom_idx = idx as usize;
                let p = chunk.coords[frame_base + atom_idx];
                let base = i * 3;
                self.sums[base] += p[0] as f64;
                self.sums[base + 1] += p[1] as f64;
                self.sums[base + 2] += p[2] as f64;
            }
        }
        self.frames += chunk.n_frames;
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
        if matches!(_device, Device::Cuda(_)) {
            return self.process_chunk(chunk, _system, _device);
        }

        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let frame_base = frame * n_atoms;
            for i in 0..n_atoms {
                let p = chunk.coords[frame_base + i];
                let base = i * 3;
                self.sums[base] += p[0] as f64;
                self.sums[base + 1] += p[1] as f64;
                self.sums[base + 2] += p[2] as f64;
            }
        }
        self.frames += chunk.n_frames;
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.frames == 0 {
            return Ok(PlanOutput::Series(Vec::new()));
        }
        let inv = 1.0 / self.frames as f64;
        let mut out = Vec::with_capacity(self.sums.len());
        for v in self.sums.iter() {
            out.push((v * inv) as f32);
        }
        Ok(PlanOutput::Series(out))
    }
}

pub struct AverageFramePlan {
    inner: MeanStructurePlan,
}

impl AverageFramePlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            inner: MeanStructurePlan::new(selection),
        }
    }
}

impl Plan for AverageFramePlan {
    fn name(&self) -> &'static str {
        "get_average_frame"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.inner.init(system, device)
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.inner.use_selected_input {
            Some(&self.inner.selection.indices)
        } else {
            None
        }
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner.process_chunk(chunk, system, device)
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner
            .process_chunk_selected(chunk, source_selection, system, device)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        self.inner.finalize()
    }
}

pub struct MakeStructurePlan {
    inner: MeanStructurePlan,
}

impl MakeStructurePlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            inner: MeanStructurePlan::new(selection),
        }
    }
}

impl Plan for MakeStructurePlan {
    fn name(&self) -> &'static str {
        "make_structure"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.inner.init(system, device)
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.inner.use_selected_input {
            Some(&self.inner.selection.indices)
        } else {
            None
        }
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner.process_chunk(chunk, system, device)
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner
            .process_chunk_selected(chunk, source_selection, system, device)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        self.inner.finalize()
    }
}
