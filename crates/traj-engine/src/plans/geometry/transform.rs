use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use super::utils::*;
use crate::executor::{Device, Plan, PlanOutput, PlanRequirements};

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, Float4, GpuBufferF32, GpuBufferU32, GpuGroups};

fn reserve_output_capacity(results: &mut Vec<f32>, n_frames_hint: Option<usize>, n_atoms: usize) {
    if let Some(frames) = n_frames_hint {
        if let Some(cap) = frames.checked_mul(n_atoms).and_then(|v| v.checked_mul(3)) {
            results.reserve(cap);
        }
    }
}

fn append_scaled_coords_uninit(
    results: &mut Vec<f32>,
    coords: &[[f32; 4]],
    scale: f32,
) -> TrajResult<()> {
    let needed = coords
        .len()
        .checked_mul(3)
        .ok_or_else(|| TrajError::Invalid("scale output size overflow".into()))?;
    if needed == 0 {
        return Ok(());
    }
    let start = results.len();
    results.reserve(needed);
    // Safety: we reserved enough capacity for `needed` elements and only write
    // initialized values into that reserved tail before advancing len.
    let mut dst = unsafe { results.as_mut_ptr().add(start) };
    for p in coords.iter() {
        // Safety: `dst` always points into the reserved tail and advances by 3
        // per source coordinate; total writes are exactly `needed`.
        unsafe {
            dst.write(p[0] * scale);
            dst = dst.add(1);
            dst.write(p[1] * scale);
            dst = dst.add(1);
            dst.write(p[2] * scale);
            dst = dst.add(1);
        }
    }
    // Safety: exactly `needed` initialized values were written into the tail.
    unsafe {
        results.set_len(start + needed);
    }
    Ok(())
}

pub struct CenterTrajectoryPlan {
    selection: Selection,
    center: [f64; 3],
    mode: CenterMode,
    mass_weighted: bool,
    results: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<CenterGpuState>,
}

#[cfg(feature = "cuda")]
struct CenterGpuState {
    groups: GpuGroups,
    masses: GpuBufferF32,
}

#[derive(Debug, Clone, Copy)]
pub enum CenterMode {
    Origin,
    Point,
    Box,
}

impl CenterTrajectoryPlan {
    pub fn new(
        selection: Selection,
        center: [f64; 3],
        mode: CenterMode,
        mass_weighted: bool,
    ) -> Self {
        Self {
            selection,
            center,
            mode,
            mass_weighted,
            results: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for CenterTrajectoryPlan {
    fn name(&self) -> &'static str {
        "center"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let mut indices = Vec::with_capacity(self.selection.indices.len());
                indices.extend(self.selection.indices.iter().copied());
                let offsets = vec![0u32, self.selection.indices.len() as u32];
                let max_len = self.selection.indices.len().max(1);
                let groups = ctx.groups(&offsets, &indices, max_len)?;
                let masses = if self.mass_weighted {
                    _system.atoms.mass.clone()
                } else {
                    vec![1.0f32; _system.n_atoms()]
                };
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
        let n_atoms = chunk.n_atoms;
        let masses = &system.atoms.mass;
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
            let mut shifts = Vec::with_capacity(chunk.n_frames);
            for frame in 0..chunk.n_frames {
                let center = match self.mode {
                    CenterMode::Origin => [0.0, 0.0, 0.0],
                    CenterMode::Point => self.center,
                    CenterMode::Box => {
                        let (lx, ly, lz) = box_lengths(chunk, frame)?;
                        [lx / 2.0, ly / 2.0, lz / 2.0]
                    }
                };
                let com = coms[frame];
                let shift = Float4 {
                    x: (center[0] - com.x as f64) as f32,
                    y: (center[1] - com.y as f64) as f32,
                    z: (center[2] - com.z as f64) as f32,
                    w: 0.0,
                };
                shifts.push(shift);
            }
            let out = ctx.shift_coords(&coords, chunk.n_atoms, chunk.n_frames, &shifts)?;
            self.results.extend(out);
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            let center = match self.mode {
                CenterMode::Origin => [0.0, 0.0, 0.0],
                CenterMode::Point => self.center,
                CenterMode::Box => {
                    let (lx, ly, lz) = box_lengths(chunk, frame)?;
                    [lx / 2.0, ly / 2.0, lz / 2.0]
                }
            };
            let com = center_of_selection(
                chunk,
                frame,
                &self.selection.indices,
                masses,
                self.mass_weighted,
            );
            let shift = [center[0] - com[0], center[1] - com[1], center[2] - com[2]];
            for atom in 0..n_atoms {
                let p = chunk.coords[frame * n_atoms + atom];
                self.results.push((p[0] as f64 + shift[0]) as f32);
                self.results.push((p[1] as f64 + shift[1]) as f32);
                self.results.push((p[2] as f64 + shift[2]) as f32);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

pub struct TranslatePlan {
    delta: [f32; 3],
    n_frames_hint: Option<usize>,
    results: Vec<f32>,
}

impl TranslatePlan {
    pub fn new(delta: [f64; 3]) -> Self {
        Self {
            delta: [delta[0] as f32, delta[1] as f32, delta[2] as f32],
            n_frames_hint: None,
            results: Vec::new(),
        }
    }
}

impl Plan for TranslatePlan {
    fn name(&self) -> &'static str {
        "translate"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn set_frames_hint(&mut self, n_frames: Option<usize>) {
        self.n_frames_hint = n_frames;
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        reserve_output_capacity(&mut self.results, self.n_frames_hint, _system.n_atoms());
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(ctx) = _device {
            let coords = convert_coords(&chunk.coords);
            let out = ctx.translate_coords(&coords, chunk.n_atoms, chunk.n_frames, self.delta)?;
            self.results.extend(out);
            return Ok(());
        }
        let dx = self.delta[0];
        let dy = self.delta[1];
        let dz = self.delta[2];
        let base = self.results.len();
        let needed = chunk.coords.len().saturating_mul(3);
        self.results.resize(base + needed, 0.0);
        let out = &mut self.results[base..];
        for (dst, p) in out.chunks_exact_mut(3).zip(chunk.coords.iter()) {
            dst[0] = p[0] + dx;
            dst[1] = p[1] + dy;
            dst[2] = p[2] + dz;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

pub struct TransformPlan {
    rotation: [f32; 9],
    translation: [f32; 3],
    n_frames_hint: Option<usize>,
    results: Vec<f32>,
}

impl TransformPlan {
    pub fn new(rotation: [f64; 9], translation: [f64; 3]) -> Self {
        let mut rot = [0.0f32; 9];
        for (dst, src) in rot.iter_mut().zip(rotation.iter()) {
            *dst = *src as f32;
        }
        Self {
            rotation: rot,
            translation: [
                translation[0] as f32,
                translation[1] as f32,
                translation[2] as f32,
            ],
            n_frames_hint: None,
            results: Vec::new(),
        }
    }
}

impl Plan for TransformPlan {
    fn name(&self) -> &'static str {
        "transform"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn set_frames_hint(&mut self, n_frames: Option<usize>) {
        self.n_frames_hint = n_frames;
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        reserve_output_capacity(&mut self.results, self.n_frames_hint, _system.n_atoms());
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(ctx) = _device {
            let coords = convert_coords(&chunk.coords);
            let out = ctx.transform_coords(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &self.rotation,
                &self.translation,
            )?;
            self.results.extend(out);
            return Ok(());
        }
        let r = &self.rotation;
        let t = &self.translation;
        let base = self.results.len();
        let needed = chunk.coords.len().saturating_mul(3);
        self.results.resize(base + needed, 0.0);
        let out = &mut self.results[base..];
        for (dst, p) in out.chunks_exact_mut(3).zip(chunk.coords.iter()) {
            let x = p[0];
            let y = p[1];
            let z = p[2];
            dst[0] = r[0] * x + r[1] * y + r[2] * z + t[0];
            dst[1] = r[3] * x + r[4] * y + r[5] * z + t[1];
            dst[2] = r[6] * x + r[7] * y + r[8] * z + t[2];
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

pub struct RotatePlan {
    inner: TransformPlan,
}

impl RotatePlan {
    pub fn new(rotation: [f64; 9]) -> Self {
        Self {
            inner: TransformPlan::new(rotation, [0.0, 0.0, 0.0]),
        }
    }
}

impl Plan for RotatePlan {
    fn name(&self) -> &'static str {
        "rotate"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn set_frames_hint(&mut self, n_frames: Option<usize>) {
        self.inner.set_frames_hint(n_frames);
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

pub struct ScalePlan {
    scale: f32,
    n_frames_hint: Option<usize>,
    results: Vec<f32>,
}

impl ScalePlan {
    pub fn new(scale: f64) -> Self {
        Self {
            scale: scale as f32,
            n_frames_hint: None,
            results: Vec::new(),
        }
    }
}

impl Plan for ScalePlan {
    fn name(&self) -> &'static str {
        "scale"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn set_frames_hint(&mut self, n_frames: Option<usize>) {
        self.n_frames_hint = n_frames;
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        reserve_output_capacity(&mut self.results, self.n_frames_hint, _system.n_atoms());
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        #[cfg(feature = "cuda")]
        if let Device::Cuda(ctx) = _device {
            let coords = convert_coords(&chunk.coords);
            let out = ctx.scale_coords(&coords, chunk.n_atoms, chunk.n_frames, self.scale)?;
            self.results.extend(out);
            return Ok(());
        }
        append_scaled_coords_uninit(&mut self.results, &chunk.coords, self.scale)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

pub struct ImagePlan {
    selection: Selection,
    mask: Vec<bool>,
    results: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<GpuBufferU32>,
}

impl ImagePlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            mask: Vec::new(),
            results: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl Plan for ImagePlan {
    fn name(&self) -> &'static str {
        "image"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.mask = vec![false; _system.n_atoms()];
        for &idx in self.selection.indices.iter() {
            if let Some(slot) = self.mask.get_mut(idx as usize) {
                *slot = true;
            }
        }
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let mut mask_u32 = vec![0u32; _system.n_atoms()];
                for (i, flag) in self.mask.iter().enumerate() {
                    if *flag {
                        mask_u32[i] = 1;
                    }
                }
                let mask = ctx.upload_u32(&mask_u32)?;
                self.gpu = Some(mask);
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
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(mask)) = (_device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let (cell, inv) = chunk_cell_mats(chunk)?;
            let out =
                ctx.image_coords(&coords, chunk.n_atoms, chunk.n_frames, &cell, &inv, mask)?;
            self.results.extend(out);
            return Ok(());
        }
        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let box_ = chunk.box_[frame];
            let (cell, inv) = match box_ {
                Box3::Orthorhombic { .. } => {
                    let (lx, ly, lz) = box_lengths(chunk, frame)?;
                    let mut out = vec![0.0f32; n_atoms * 3];
                    for atom in 0..n_atoms {
                        let p = chunk.coords[frame * n_atoms + atom];
                        if self.mask.get(atom).copied().unwrap_or(false) {
                            let mut x = p[0] as f64;
                            let mut y = p[1] as f64;
                            let mut z = p[2] as f64;
                            if lx > 0.0 {
                                x -= (x / lx).floor() * lx;
                            }
                            if ly > 0.0 {
                                y -= (y / ly).floor() * ly;
                            }
                            if lz > 0.0 {
                                z -= (z / lz).floor() * lz;
                            }
                            out[atom * 3] = x as f32;
                            out[atom * 3 + 1] = y as f32;
                            out[atom * 3 + 2] = z as f32;
                        } else {
                            out[atom * 3] = p[0];
                            out[atom * 3 + 1] = p[1];
                            out[atom * 3 + 2] = p[2];
                        }
                    }
                    self.results.extend(out);
                    continue;
                }
                Box3::Triclinic { .. } => cell_and_inv_from_box(box_)?,
                Box3::None => return Err(TrajError::Mismatch("image requires box vectors".into())),
            };
            for atom in 0..n_atoms {
                let p = chunk.coords[frame * n_atoms + atom];
                if !self.mask.get(atom).copied().unwrap_or(false) {
                    self.results.push(p[0]);
                    self.results.push(p[1]);
                    self.results.push(p[2]);
                    continue;
                }
                let x = p[0] as f64;
                let y = p[1] as f64;
                let z = p[2] as f64;
                let mut f0 = inv[0][0] * x + inv[1][0] * y + inv[2][0] * z;
                let mut f1 = inv[0][1] * x + inv[1][1] * y + inv[2][1] * z;
                let mut f2 = inv[0][2] * x + inv[1][2] * y + inv[2][2] * z;
                f0 -= f0.floor();
                f1 -= f1.floor();
                f2 -= f2.floor();
                let nx = f0 * cell[0][0] + f1 * cell[1][0] + f2 * cell[2][0];
                let ny = f0 * cell[0][1] + f1 * cell[1][1] + f2 * cell[2][1];
                let nz = f0 * cell[0][2] + f1 * cell[1][2] + f2 * cell[2][2];
                self.results.push(nx as f32);
                self.results.push(ny as f32);
                self.results.push(nz as f32);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

pub struct AutoImagePlan {
    inner: ImagePlan,
}

impl AutoImagePlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            inner: ImagePlan::new(selection),
        }
    }
}

impl Plan for AutoImagePlan {
    fn name(&self) -> &'static str {
        "autoimage"
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
