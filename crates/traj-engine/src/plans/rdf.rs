use std::sync::Arc;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput, RdfOutput};
use crate::plans::PbcMode;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuCounts, GpuSelection};

pub struct RdfPlan {
    sel_a: Selection,
    sel_b: Selection,
    bins: usize,
    r_max: f32,
    pbc: PbcMode,
    counts: Vec<u64>,
    frames: usize,
    volume_sum: f64,
    #[cfg(feature = "cuda")]
    gpu: Option<RdfGpuState>,
}

#[cfg(feature = "cuda")]
struct RdfGpuState {
    ctx: traj_gpu::GpuContext,
    sel_a: GpuSelection,
    sel_b: GpuSelection,
    counts: GpuCounts,
    same_sel: bool,
}

pub struct PairDistPlan {
    sel_a: Selection,
    sel_b: Selection,
    bins: usize,
    r_max: f32,
    pbc: PbcMode,
    counts: Vec<u64>,
    frames: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<PairDistGpuState>,
}

#[cfg(feature = "cuda")]
struct PairDistGpuState {
    ctx: traj_gpu::GpuContext,
    sel_a: GpuSelection,
    sel_b: GpuSelection,
    counts: GpuCounts,
    same_sel: bool,
}

impl RdfPlan {
    pub fn new(sel_a: Selection, sel_b: Selection, bins: usize, r_max: f32, pbc: PbcMode) -> Self {
        Self {
            sel_a,
            sel_b,
            bins,
            r_max,
            pbc,
            counts: vec![0; bins],
            frames: 0,
            volume_sum: 0.0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl PairDistPlan {
    pub fn new(sel_a: Selection, sel_b: Selection, bins: usize, r_max: f32, pbc: PbcMode) -> Self {
        Self {
            sel_a,
            sel_b,
            bins,
            r_max,
            pbc,
            counts: vec![0; bins],
            frames: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for RdfPlan {
    fn name(&self) -> &'static str {
        "rdf"
    }

    fn init(&mut self, _system: &System, device: &Device) -> TrajResult<()> {
        self.counts.fill(0);
        self.frames = 0;
        self.volume_sum = 0.0;
        let _ = device;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = device {
                let sel_a = ctx.selection(&self.sel_a.indices, None)?;
                let sel_b = ctx.selection(&self.sel_b.indices, None)?;
                let counts = ctx.alloc_counts(self.bins)?;
                let same_sel = Arc::ptr_eq(&self.sel_a.indices, &self.sel_b.indices);
                let mut gpu = RdfGpuState {
                    ctx: ctx.clone(),
                    sel_a,
                    sel_b,
                    counts,
                    same_sel,
                };
                ctx.reset_counts(&mut gpu.counts)?;
                self.gpu = Some(gpu);
            }
        }
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        let _ = device;
        let n_atoms = chunk.n_atoms;
        let bin_width = self.r_max / self.bins as f32;
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (device, &mut self.gpu) {
            let mut box_l = Vec::new();
            if matches!(self.pbc, PbcMode::Orthorhombic) {
                box_l.reserve(chunk.n_frames * 3);
            }
            for frame in 0..chunk.n_frames {
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    match chunk.box_[frame] {
                        Box3::Orthorhombic { lx, ly, lz } => {
                            box_l.push(lx);
                            box_l.push(ly);
                            box_l.push(lz);
                            self.volume_sum += (lx as f64) * (ly as f64) * (lz as f64);
                        }
                        _ => {
                            return Err(TrajError::Mismatch(
                                "RDF requires orthorhombic box for PBC".into(),
                            ))
                        }
                    }
                }
                self.frames += 1;
            }
            let coords = convert_coords(&chunk.coords);
            ctx.rdf_accum(
                &coords,
                n_atoms,
                chunk.n_frames,
                &gpu.sel_a,
                &gpu.sel_b,
                self.r_max,
                self.bins,
                matches!(self.pbc, PbcMode::Orthorhombic),
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    Some(&box_l)
                } else {
                    None
                },
                gpu.same_sel,
                &mut gpu.counts,
            )?;
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            let box_ = chunk.box_[frame];
            let (lx, ly, lz) = match self.pbc {
                PbcMode::None => (0.0, 0.0, 0.0),
                PbcMode::Orthorhombic => match box_ {
                    Box3::Orthorhombic { lx, ly, lz } => (lx, ly, lz),
                    _ => {
                        return Err(TrajError::Mismatch(
                            "RDF requires orthorhombic box for PBC".into(),
                        ))
                    }
                },
            };
            if matches!(self.pbc, PbcMode::Orthorhombic) {
                self.volume_sum += (lx as f64) * (ly as f64) * (lz as f64);
            }
            for &a in self.sel_a.indices.iter() {
                let a_idx = a as usize;
                let pos_a = chunk.coords[frame * n_atoms + a_idx];
                for &b in self.sel_b.indices.iter() {
                    let b_idx = b as usize;
                    if a_idx == b_idx && Arc::ptr_eq(&self.sel_a.indices, &self.sel_b.indices) {
                        continue;
                    }
                    let pos_b = chunk.coords[frame * n_atoms + b_idx];
                    let mut dx = pos_b[0] - pos_a[0];
                    let mut dy = pos_b[1] - pos_a[1];
                    let mut dz = pos_b[2] - pos_a[2];
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        dx -= (dx / lx).round() * lx;
                        dy -= (dy / ly).round() * ly;
                        dz -= (dz / lz).round() * lz;
                    }
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    if r < self.r_max {
                        let bin = (r / bin_width) as usize;
                        if bin < self.bins {
                            self.counts[bin] += 1;
                        }
                    }
                }
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        #[cfg(feature = "cuda")]
        if let Some(gpu) = &self.gpu {
            self.counts = gpu.ctx.read_counts(&gpu.counts)?;
        }
        let bin_width = self.r_max / self.bins as f32;
        let mut r = Vec::with_capacity(self.bins);
        let mut g_r = Vec::with_capacity(self.bins);
        let n_a = self.sel_a.indices.len() as f64;
        let n_b = self.sel_b.indices.len() as f64;
        let frames = self.frames.max(1) as f64;
        let volume = if matches!(self.pbc, PbcMode::Orthorhombic) {
            if self.volume_sum == 0.0 {
                1.0
            } else {
                self.volume_sum / frames
            }
        } else {
            1.0
        };
        let density = if volume > 0.0 { n_b / volume } else { 0.0 };
        for i in 0..self.bins {
            let r_inner = i as f32 * bin_width;
            let r_outer = (i + 1) as f32 * bin_width;
            let shell_vol =
                (4.0 / 3.0) * std::f32::consts::PI * (r_outer.powi(3) - r_inner.powi(3));
            let norm = if shell_vol > 0.0 {
                frames * n_a * density * shell_vol as f64
            } else {
                0.0
            };
            r.push(r_inner + 0.5 * bin_width);
            let count = self.counts[i] as f64;
            let g = if norm > 0.0 { count / norm } else { 0.0 };
            g_r.push(g as f32);
        }
        Ok(PlanOutput::Rdf(RdfOutput {
            r,
            g_r,
            counts: std::mem::take(&mut self.counts),
        }))
    }
}

impl Plan for PairDistPlan {
    fn name(&self) -> &'static str {
        "pairdist"
    }

    fn init(&mut self, _system: &System, device: &Device) -> TrajResult<()> {
        self.counts.fill(0);
        self.frames = 0;
        let _ = device;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = device {
                let sel_a = ctx.selection(&self.sel_a.indices, None)?;
                let sel_b = ctx.selection(&self.sel_b.indices, None)?;
                let counts = ctx.alloc_counts(self.bins)?;
                let same_sel = Arc::ptr_eq(&self.sel_a.indices, &self.sel_b.indices);
                let mut gpu = PairDistGpuState {
                    ctx: ctx.clone(),
                    sel_a,
                    sel_b,
                    counts,
                    same_sel,
                };
                ctx.reset_counts(&mut gpu.counts)?;
                self.gpu = Some(gpu);
            }
        }
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        let _ = device;
        let n_atoms = chunk.n_atoms;
        let bin_width = self.r_max / self.bins as f32;
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (device, &mut self.gpu) {
            let mut box_l = Vec::new();
            if matches!(self.pbc, PbcMode::Orthorhombic) {
                box_l.reserve(chunk.n_frames * 3);
            }
            for frame in 0..chunk.n_frames {
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    match chunk.box_[frame] {
                        Box3::Orthorhombic { lx, ly, lz } => {
                            box_l.push(lx);
                            box_l.push(ly);
                            box_l.push(lz);
                        }
                        _ => {
                            return Err(TrajError::Mismatch(
                                "pairdist requires orthorhombic box for PBC".into(),
                            ))
                        }
                    }
                }
                self.frames += 1;
            }
            let coords = convert_coords(&chunk.coords);
            ctx.rdf_accum(
                &coords,
                n_atoms,
                chunk.n_frames,
                &gpu.sel_a,
                &gpu.sel_b,
                self.r_max,
                self.bins,
                matches!(self.pbc, PbcMode::Orthorhombic),
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    Some(&box_l)
                } else {
                    None
                },
                gpu.same_sel,
                &mut gpu.counts,
            )?;
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            let (lx, ly, lz) = match self.pbc {
                PbcMode::None => (0.0, 0.0, 0.0),
                PbcMode::Orthorhombic => match chunk.box_[frame] {
                    Box3::Orthorhombic { lx, ly, lz } => (lx, ly, lz),
                    _ => {
                        return Err(TrajError::Mismatch(
                            "pairdist requires orthorhombic box for PBC".into(),
                        ))
                    }
                },
            };
            for &a in self.sel_a.indices.iter() {
                let a_idx = a as usize;
                let pos_a = chunk.coords[frame * n_atoms + a_idx];
                for &b in self.sel_b.indices.iter() {
                    let b_idx = b as usize;
                    if a_idx == b_idx && Arc::ptr_eq(&self.sel_a.indices, &self.sel_b.indices) {
                        continue;
                    }
                    let pos_b = chunk.coords[frame * n_atoms + b_idx];
                    let mut dx = pos_b[0] - pos_a[0];
                    let mut dy = pos_b[1] - pos_a[1];
                    let mut dz = pos_b[2] - pos_a[2];
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        dx -= (dx / lx).round() * lx;
                        dy -= (dy / ly).round() * ly;
                        dz -= (dz / lz).round() * lz;
                    }
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    if r < self.r_max {
                        let bin = (r / bin_width) as usize;
                        if bin < self.bins {
                            self.counts[bin] += 1;
                        }
                    }
                }
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        #[cfg(feature = "cuda")]
        if let Some(gpu) = &self.gpu {
            self.counts = gpu.ctx.read_counts(&gpu.counts)?;
        }
        let bin_width = self.r_max / self.bins as f32;
        let mut centers = Vec::with_capacity(self.bins);
        for i in 0..self.bins {
            centers.push(i as f32 * bin_width + 0.5 * bin_width);
        }
        Ok(PlanOutput::Histogram {
            centers,
            counts: std::mem::take(&mut self.counts),
        })
    }
}
