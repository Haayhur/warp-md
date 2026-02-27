use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::pbc_utils;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::PbcMode;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, Float4, GpuAnchors};

pub struct NmrIredPlan {
    pairs: Vec<[u32; 2]>,
    length_scale: f64,
    pbc: PbcMode,
    vectors: Vec<f32>,
    frames: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<NmrGpuState>,
}

#[cfg(feature = "cuda")]
struct NmrGpuState {
    anchors: GpuAnchors,
}

impl NmrIredPlan {
    pub fn new(pairs: Vec<[u32; 2]>, length_scale: f64, pbc: PbcMode) -> Self {
        Self {
            pairs,
            length_scale,
            pbc,
            vectors: Vec::new(),
            frames: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for NmrIredPlan {
    fn name(&self) -> &'static str {
        "nmr_ired"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.vectors.clear();
        self.frames = 0;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if matches!(self.pbc, PbcMode::None | PbcMode::Orthorhombic) {
                if let Device::Cuda(ctx) = _device {
                    if !self.pairs.is_empty() {
                        let mut anchors = Vec::with_capacity(self.pairs.len() * 3);
                        for [a, b] in self.pairs.iter() {
                            anchors.push(*a);
                            anchors.push(*b);
                            anchors.push(*a);
                        }
                        let anchors = ctx.anchors(&anchors)?;
                        self.gpu = Some(NmrGpuState { anchors });
                    }
                }
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
        let n_pairs = self.pairs.len();
        if n_pairs == 0 {
            self.frames += chunk.n_frames;
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let out = match self.pbc {
                PbcMode::None => ctx.orientation_vector(
                    &coords,
                    chunk.n_atoms,
                    chunk.n_frames,
                    &gpu.anchors,
                    self.length_scale as f32,
                )?,
                PbcMode::Orthorhombic => {
                    let mut boxes = Vec::with_capacity(chunk.n_frames);
                    for box_ in chunk.box_.iter().take(chunk.n_frames) {
                        let (lx, ly, lz) = pbc_utils::box_lengths(*box_)?;
                        boxes.push(Float4 {
                            x: lx as f32 * self.length_scale as f32,
                            y: ly as f32 * self.length_scale as f32,
                            z: lz as f32 * self.length_scale as f32,
                            w: 0.0,
                        });
                    }
                    ctx.orientation_vector_pbc(
                        &coords,
                        chunk.n_atoms,
                        chunk.n_frames,
                        &gpu.anchors,
                        &boxes,
                        self.length_scale as f32,
                    )?
                }
            };
            for v in out.into_iter() {
                self.vectors.push(v.x);
                self.vectors.push(v.y);
                self.vectors.push(v.z);
            }
            self.frames += chunk.n_frames;
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            let (lx, ly, lz) = if matches!(self.pbc, PbcMode::Orthorhombic) {
                pbc_utils::box_lengths(chunk.box_[frame])?
            } else {
                (0.0, 0.0, 0.0)
            };
            let frame_offset = frame * n_atoms;
            for pair in self.pairs.iter() {
                let a = pair[0] as usize;
                let b = pair[1] as usize;
                if a >= n_atoms || b >= n_atoms {
                    return Err(TrajError::Parse(
                        "nmr pair index out of bounds for frame chunk".into(),
                    ));
                }
                let pa = chunk.coords[frame_offset + a];
                let pb = chunk.coords[frame_offset + b];
                let mut dx = (pb[0] - pa[0]) as f64 * self.length_scale;
                let mut dy = (pb[1] - pa[1]) as f64 * self.length_scale;
                let mut dz = (pb[2] - pa[2]) as f64 * self.length_scale;
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    pbc_utils::apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                }
                let norm = (dx * dx + dy * dy + dz * dz).sqrt();
                if norm > 0.0 {
                    self.vectors.push((dx / norm) as f32);
                    self.vectors.push((dy / norm) as f32);
                    self.vectors.push((dz / norm) as f32);
                } else {
                    self.vectors.extend_from_slice(&[0.0, 0.0, 0.0]);
                }
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.vectors),
            rows: self.frames,
            cols: self.pairs.len() * 3,
        })
    }
}
