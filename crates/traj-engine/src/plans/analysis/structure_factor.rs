use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput, StructureFactorOutput};
use crate::plans::PbcMode;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuCounts, GpuSelection};

pub struct StructureFactorPlan {
    selection: Selection,
    bins: usize,
    r_max: f64,
    q_bins: usize,
    q_max: f64,
    pbc: PbcMode,
    length_scale: f64,
    counts: Vec<u64>,
    frames: usize,
    volume_sum: f64,
    #[cfg(feature = "cuda")]
    gpu: Option<StructureFactorGpuState>,
}

#[cfg(feature = "cuda")]
struct StructureFactorGpuState {
    ctx: traj_gpu::GpuContext,
    sel: GpuSelection,
    counts: GpuCounts,
}

impl StructureFactorPlan {
    pub fn new(
        selection: Selection,
        bins: usize,
        r_max: f64,
        q_bins: usize,
        q_max: f64,
        pbc: PbcMode,
    ) -> Self {
        Self {
            selection,
            bins,
            r_max,
            q_bins,
            q_max,
            pbc,
            length_scale: 1.0,
            counts: vec![0; bins],
            frames: 0,
            volume_sum: 0.0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for StructureFactorPlan {
    fn name(&self) -> &'static str {
        "structure_factor"
    }

    fn init(&mut self, _system: &System, device: &Device) -> TrajResult<()> {
        if self.bins == 0 || self.q_bins == 0 {
            return Err(TrajError::InvalidSelection(
                "structure factor requires bins and q_bins > 0".into(),
            ));
        }
        self.counts.fill(0);
        self.frames = 0;
        self.volume_sum = 0.0;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = device {
                let sel = ctx.selection(&self.selection.indices, None)?;
                let counts = ctx.alloc_counts(self.bins)?;
                self.gpu = Some(StructureFactorGpuState {
                    ctx: ctx.clone(),
                    sel,
                    counts,
                });
                ctx.reset_counts(&mut self.gpu.as_mut().unwrap().counts)?;
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = device;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        let bin_width = self.r_max / self.bins as f64;
        #[cfg(feature = "cuda")]
        if let Some(gpu) = &mut self.gpu {
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
                            self.volume_sum +=
                                (lx as f64) * (ly as f64) * (lz as f64) * self.length_scale.powi(3);
                        }
                        _ => {
                            return Err(TrajError::Mismatch(
                                "structure factor requires orthorhombic box for PBC".into(),
                            ))
                        }
                    }
                }
                self.frames += 1;
            }
            let coords = convert_coords(&chunk.coords);
            gpu.ctx.rdf_accum(
                &coords,
                n_atoms,
                chunk.n_frames,
                &gpu.sel,
                &gpu.sel,
                self.r_max as f32,
                self.bins,
                matches!(self.pbc, PbcMode::Orthorhombic),
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    Some(&box_l)
                } else {
                    None
                },
                true,
                &mut gpu.counts,
            )?;
            return Ok(());
        }

        for frame in 0..chunk.n_frames {
            let box_ = chunk.box_[frame];
            let (lx, ly, lz) = match self.pbc {
                PbcMode::None => (0.0, 0.0, 0.0),
                PbcMode::Orthorhombic => match box_ {
                    Box3::Orthorhombic { lx, ly, lz } => (lx as f64, ly as f64, lz as f64),
                    _ => {
                        return Err(TrajError::Mismatch(
                            "structure factor requires orthorhombic box for PBC".into(),
                        ))
                    }
                },
            };
            if matches!(self.pbc, PbcMode::Orthorhombic) {
                self.volume_sum += (lx * ly * lz) * self.length_scale.powi(3);
            }
            for &a in self.selection.indices.iter() {
                let a_idx = a as usize;
                let pos_a = chunk.coords[frame * n_atoms + a_idx];
                for &b in self.selection.indices.iter() {
                    let b_idx = b as usize;
                    if a_idx == b_idx {
                        continue;
                    }
                    let pos_b = chunk.coords[frame * n_atoms + b_idx];
                    let mut dx = (pos_b[0] - pos_a[0]) as f64 * self.length_scale;
                    let mut dy = (pos_b[1] - pos_a[1]) as f64 * self.length_scale;
                    let mut dz = (pos_b[2] - pos_a[2]) as f64 * self.length_scale;
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        let lx = lx * self.length_scale;
                        let ly = ly * self.length_scale;
                        let lz = lz * self.length_scale;
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
        let bin_width = self.r_max / self.bins as f64;
        let n = self.selection.indices.len() as f64;
        let frames = self.frames.max(1) as f64;
        let volume = if matches!(self.pbc, PbcMode::Orthorhombic) {
            if self.volume_sum == 0.0 {
                return Err(TrajError::Mismatch(
                    "structure factor requires box volume".into(),
                ));
            }
            self.volume_sum / frames
        } else {
            1.0
        };
        let rho = if volume > 0.0 { n / volume } else { 0.0 };
        let mut r = Vec::with_capacity(self.bins);
        let mut g_r = Vec::with_capacity(self.bins);
        for i in 0..self.bins {
            let r0 = (i as f64 + 0.5) * bin_width;
            let shell = 4.0 * std::f64::consts::PI * r0 * r0 * bin_width;
            let norm = if shell > 0.0 { rho * shell * n } else { 1.0 };
            let g = (self.counts[i] as f64) / (norm * frames);
            r.push(r0 as f32);
            g_r.push(g as f32);
        }

        let mut q = Vec::with_capacity(self.q_bins);
        let mut s_q = Vec::with_capacity(self.q_bins);
        let dq = if self.q_bins > 1 {
            self.q_max / (self.q_bins as f64 - 1.0)
        } else {
            0.0
        };
        for i in 0..self.q_bins {
            let qval = dq * i as f64;
            let mut sum = 0.0f64;
            for (idx, &gr) in g_r.iter().enumerate() {
                let r0 = (idx as f64 + 0.5) * bin_width;
                let gr = gr as f64;
                let diff = gr - 1.0;
                let weight = if qval == 0.0 {
                    1.0
                } else {
                    (qval * r0).sin() / (qval * r0)
                };
                sum += diff * weight * r0 * r0 * bin_width;
            }
            let sq = 1.0 + 4.0 * std::f64::consts::PI * rho * sum;
            q.push(qval as f32);
            s_q.push(sq as f32);
        }

        Ok(PlanOutput::StructureFactor(StructureFactorOutput {
            r,
            g_r,
            q,
            s_q,
        }))
    }
}
