use traj_core::error::TrajResult;
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use super::utils::*;
use crate::executor::{Device, Plan, PlanOutput, PlanRequirements};
use crate::plans::PbcMode;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuSelection};

pub struct VectorPlan {
    sel_a: Selection,
    sel_b: Selection,
    mass_weighted: bool,
    pbc: PbcMode,
    results: Vec<f32>,
    frames: usize,
}

impl VectorPlan {
    pub fn new(sel_a: Selection, sel_b: Selection, mass_weighted: bool, pbc: PbcMode) -> Self {
        Self {
            sel_a,
            sel_b,
            mass_weighted,
            pbc,
            results: Vec::new(),
            frames: 0,
        }
    }
}

impl Plan for VectorPlan {
    fn name(&self) -> &'static str {
        "vector"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        Ok(())
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

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: 3,
        })
    }
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

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        if self.prev_coords.len() != sel.len() {
            self.prev_coords.resize(sel.len(), [0.0; 3]);
            self.has_prev = false;
            self.prev_time = None;
        }
        for frame in 0..chunk.n_frames {
            let time = chunk
                .time_ps
                .as_ref()
                .and_then(|times| times.get(frame))
                .map(|t| *t as f64);
            let mut dt = match (self.prev_time, time) {
                (Some(prev), Some(cur)) if cur > prev => (cur - prev) * self.time_scale,
                _ => 1.0 * self.time_scale,
            };
            if !dt.is_finite() || dt == 0.0 {
                dt = 1.0 * self.time_scale;
            }
            for (i, &idx) in sel.iter().enumerate() {
                let p = chunk.coords[frame * n_atoms + idx as usize];
                if self.has_prev {
                    let prev = self.prev_coords[i];
                    let vx = (p[0] as f64 - prev[0] as f64) / dt;
                    let vy = (p[1] as f64 - prev[1] as f64) / dt;
                    let vz = (p[2] as f64 - prev[2] as f64) / dt;
                    self.results
                        .extend_from_slice(&[vx as f32, vy as f32, vz as f32]);
                } else {
                    self.results.extend_from_slice(&[0.0, 0.0, 0.0]);
                }
                self.prev_coords[i] = [p[0], p[1], p[2]];
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
