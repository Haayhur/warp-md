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
