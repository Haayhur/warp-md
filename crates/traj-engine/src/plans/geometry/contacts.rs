use std::sync::Arc;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use super::neighbors::SearchNeighborsPlan;
use super::utils::*;
use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::{PbcMode, ReferenceMode};

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuPairs};

pub struct WatershellPlan {
    inner: SearchNeighborsPlan,
}

impl WatershellPlan {
    pub fn new(target: Selection, probe: Selection, cutoff: f64, pbc: PbcMode) -> Self {
        Self {
            inner: SearchNeighborsPlan::new(target, probe, cutoff, pbc),
        }
    }
}

impl Plan for WatershellPlan {
    fn name(&self) -> &'static str {
        "watershell"
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

pub struct NativeContactsPlan {
    sel_a: Selection,
    sel_b: Selection,
    cutoff: f64,
    pbc: PbcMode,
    reference_mode: ReferenceMode,
    reference_pairs: Vec<(u32, u32)>,
    results: Vec<f32>,
    reference_ready: bool,
    same_sel: bool,
    #[cfg(feature = "cuda")]
    gpu: Option<NativeContactsGpuState>,
}

#[cfg(feature = "cuda")]
struct NativeContactsGpuState {
    pairs: GpuPairs,
}

impl NativeContactsPlan {
    pub fn new(
        sel_a: Selection,
        sel_b: Selection,
        reference_mode: ReferenceMode,
        cutoff: f64,
        pbc: PbcMode,
    ) -> Self {
        let same_sel = Arc::ptr_eq(&sel_a.indices, &sel_b.indices);
        Self {
            sel_a,
            sel_b,
            cutoff,
            pbc,
            reference_mode,
            reference_pairs: Vec::new(),
            results: Vec::new(),
            reference_ready: false,
            same_sel,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    fn build_reference(
        &mut self,
        coords: &[[f32; 4]],
        n_atoms: usize,
        box_: Option<(f64, f64, f64)>,
    ) -> TrajResult<()> {
        self.reference_pairs.clear();
        let n_a = self.sel_a.indices.len();
        let n_b = self.sel_b.indices.len();
        if n_a == 0 || n_b == 0 {
            self.reference_ready = true;
            return Ok(());
        }
        let cutoff = self.cutoff;
        let (lx, ly, lz) = box_.unwrap_or((0.0, 0.0, 0.0));
        if self.same_sel {
            for i in 0..n_a {
                let a_idx = self.sel_a.indices[i] as usize;
                let pa = coords[a_idx];
                for j in (i + 1)..n_a {
                    let b_idx = self.sel_a.indices[j] as usize;
                    let pb = coords[b_idx];
                    let mut dx = (pb[0] - pa[0]) as f64;
                    let mut dy = (pb[1] - pa[1]) as f64;
                    let mut dz = (pb[2] - pa[2]) as f64;
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                    }
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist <= cutoff {
                        self.reference_pairs
                            .push((self.sel_a.indices[i], self.sel_a.indices[j]));
                    }
                }
            }
        } else {
            for &a in self.sel_a.indices.iter() {
                let a_idx = a as usize;
                let pa = coords[a_idx];
                for &b in self.sel_b.indices.iter() {
                    if a == b {
                        continue;
                    }
                    let b_idx = b as usize;
                    let pb = coords[b_idx];
                    let mut dx = (pb[0] - pa[0]) as f64;
                    let mut dy = (pb[1] - pa[1]) as f64;
                    let mut dz = (pb[2] - pa[2]) as f64;
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                    }
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist <= cutoff {
                        self.reference_pairs.push((a, b));
                    }
                }
            }
        }
        self.reference_ready = true;
        let _ = n_atoms;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn upload_pairs(&mut self, ctx: &traj_gpu::GpuContext) -> TrajResult<()> {
        if self.reference_pairs.is_empty() {
            self.gpu = None;
            return Ok(());
        }
        let mut pairs_flat = Vec::with_capacity(self.reference_pairs.len() * 2);
        for &(a, b) in self.reference_pairs.iter() {
            pairs_flat.push(a);
            pairs_flat.push(b);
        }
        let pairs = ctx.pairs(&pairs_flat)?;
        self.gpu = Some(NativeContactsGpuState { pairs });
        Ok(())
    }
}

impl Plan for NativeContactsPlan {
    fn name(&self) -> &'static str {
        "native_contacts"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.reference_pairs.clear();
        self.reference_ready = false;
        self.same_sel = Arc::ptr_eq(&self.sel_a.indices, &self.sel_b.indices);
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
        }
        if matches!(self.reference_mode, ReferenceMode::Topology) {
            if matches!(self.pbc, PbcMode::Orthorhombic) {
                return Err(TrajError::Mismatch(
                    "native_contacts with topology reference does not support PBC".into(),
                ));
            }
            let positions0 = system
                .positions0
                .as_ref()
                .ok_or_else(|| TrajError::Mismatch("no topology reference coords".into()))?;
            self.build_reference(positions0, system.n_atoms(), None)?;
            #[cfg(feature = "cuda")]
            if let Device::Cuda(ctx) = _device {
                self.upload_pairs(ctx)?;
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
        if !self.reference_ready && matches!(self.reference_mode, ReferenceMode::Frame0) {
            if chunk.n_frames == 0 {
                return Ok(());
            }
            let box_ = if matches!(self.pbc, PbcMode::Orthorhombic) {
                Some(box_lengths(chunk, 0)?)
            } else {
                None
            };
            let frame0 = &chunk.coords[0..n_atoms];
            self.build_reference(frame0, n_atoms, box_)?;
            #[cfg(feature = "cuda")]
            if let Device::Cuda(ctx) = _device {
                self.upload_pairs(ctx)?;
            }
        }
        let n_ref = self.reference_pairs.len();
        if n_ref == 0 {
            self.results
                .extend(std::iter::repeat(0.0).take(chunk.n_frames));
            return Ok(());
        }
        let cutoff = self.cutoff;
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let boxes = chunk_boxes(chunk, self.pbc)?;
            let counts = ctx.native_contacts_counts(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.pairs,
                &boxes,
                cutoff as f32,
            )?;
            let denom = n_ref as f32;
            for count in counts {
                let frac = if denom > 0.0 {
                    count as f32 / denom
                } else {
                    0.0
                };
                self.results.push(frac);
            }
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            let (lx, ly, lz) = if matches!(self.pbc, PbcMode::Orthorhombic) {
                box_lengths(chunk, frame)?
            } else {
                (0.0, 0.0, 0.0)
            };
            let mut count = 0usize;
            for &(a, b) in self.reference_pairs.iter() {
                let pa = chunk.coords[frame * n_atoms + a as usize];
                let pb = chunk.coords[frame * n_atoms + b as usize];
                let mut dx = (pb[0] - pa[0]) as f64;
                let mut dy = (pb[1] - pa[1]) as f64;
                let mut dz = (pb[2] - pa[2]) as f64;
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                }
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist <= cutoff {
                    count += 1;
                }
            }
            let frac = count as f64 / n_ref as f64;
            self.results.push(frac as f32);
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}
