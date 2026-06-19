use crate::executor::{Device, Plan, PlanOutput, PlanRequirements};
use crate::plans::{PbcMode, ReferenceMode};
use nalgebra::{Matrix3, Vector3};
use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::pbc_math::{apply_pbc, box_lengths};
use traj_core::selection::Selection;
use traj_core::system::System;

use super::math::centroid;
#[cfg(feature = "cuda")]
use super::math::rmsd_from_cov;
#[cfg(feature = "cuda")]
use super::types::RmsdGpuState;
use super::types::{DistanceRmsdPlan, PairwiseMetric, PairwiseRmsdPlan, RmsdPlan, SymmRmsdPlan};

#[cfg(feature = "cuda")]
use traj_gpu::convert_coords;

impl RmsdPlan {
    pub fn new(selection: Selection, reference_mode: ReferenceMode, align: bool) -> Self {
        Self {
            selection,
            selection_usize: Vec::new(),
            dense_selection_usize: Vec::new(),
            use_selected_input: false,
            align,
            reference_mode,
            reference: None,
            results: Vec::new(),

            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl SymmRmsdPlan {
    pub fn new(selection: Selection, reference_mode: ReferenceMode, align: bool) -> Self {
        Self {
            inner: RmsdPlan::new(selection, reference_mode, align),
            symmetry_groups: Vec::new(),
            max_permutations: 4096,
        }
    }

    pub fn with_symmetry_groups(
        mut self,
        symmetry_groups: Vec<Vec<usize>>,
        max_permutations: usize,
    ) -> TrajResult<Self> {
        self.symmetry_groups = normalize_symmetry_groups(
            symmetry_groups,
            self.inner.selection.indices.len(),
            max_permutations,
        )?;
        self.max_permutations = max_permutations;
        Ok(self)
    }

    fn has_remap(&self) -> bool {
        !self.symmetry_groups.is_empty()
    }
}

impl PairwiseRmsdPlan {
    pub fn new(selection: Selection, metric: PairwiseMetric, pbc: PbcMode) -> Self {
        Self {
            selection,
            metric,
            pbc,
            use_selected_input: false,
            frames: Vec::new(),
            boxes: Vec::new(),
        }
    }
}

impl DistanceRmsdPlan {
    pub fn new(selection: Selection, reference_mode: ReferenceMode, pbc: PbcMode) -> Self {
        Self {
            selection,
            reference_mode,
            pbc,
            reference_dists: None,
            results: Vec::new(),
        }
    }
}

fn checked_factorial(n: usize) -> TrajResult<usize> {
    let mut out = 1usize;
    for value in 2..=n {
        out = out
            .checked_mul(value)
            .ok_or_else(|| TrajError::Parse("symmetry remap permutation count overflow".into()))?;
    }
    Ok(out)
}

fn normalize_symmetry_groups(
    symmetry_groups: Vec<Vec<usize>>,
    selection_len: usize,
    max_permutations: usize,
) -> TrajResult<Vec<Vec<usize>>> {
    if max_permutations == 0 {
        return Err(TrajError::Parse("max_permutations must be positive".into()));
    }

    let mut used = vec![false; selection_len];
    let mut total = 1usize;
    let mut out = Vec::new();
    for mut group in symmetry_groups {
        group.sort_unstable();
        group.dedup();
        if group.len() < 2 {
            continue;
        }
        for &position in group.iter() {
            if position >= selection_len {
                return Err(TrajError::Mismatch(format!(
                    "symmetry group position {position} out of selection range"
                )));
            }
            if used[position] {
                return Err(TrajError::Mismatch(format!(
                    "symmetry group position {position} appears in more than one group"
                )));
            }
            used[position] = true;
        }
        total = total
            .checked_mul(checked_factorial(group.len())?)
            .ok_or_else(|| TrajError::Parse("symmetry remap permutation count overflow".into()))?;
        if total > max_permutations {
            return Err(TrajError::Unsupported(format!(
                "symmetry remap requires {total} permutations, exceeds max_permutations={max_permutations}"
            )));
        }
        out.push(group);
    }
    Ok(out)
}

fn process_cpu_chunk_out(
    chunk: &FrameChunk,
    n_atoms: usize,
    reference: &[[f32; 4]],
    selection: &[usize],
    align: bool,
    out: &mut Vec<f32>,
) {
    for frame in 0..chunk.n_frames {
        let frame_coords = &chunk.coords[frame * n_atoms..(frame + 1) * n_atoms];
        let rmsd = if align {
            rmsd_aligned_selected(frame_coords, reference, selection)
        } else {
            rmsd_raw_selected(frame_coords, reference, selection)
        };
        out.push(rmsd);
    }
}

impl Plan for RmsdPlan {
    fn name(&self) -> &'static str {
        "rmsd"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.selection_usize.clear();
        self.dense_selection_usize.clear();
        self.use_selected_input = true;
        self.selection_usize
            .extend(self.selection.indices.iter().map(|&idx| idx as usize));
        self.dense_selection_usize
            .extend(0..self.selection_usize.len());
        let _ = device;

        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
        }
        match self.reference_mode {
            ReferenceMode::Topology => {
                let positions0 = system
                    .positions0
                    .as_ref()
                    .ok_or_else(|| TrajError::Mismatch("no topology reference coords".into()))?;
                let mut reference = Vec::with_capacity(self.selection_usize.len());
                for &idx in self.selection_usize.iter() {
                    reference.push(positions0[idx]);
                }
                self.reference = Some(reference);
            }
            ReferenceMode::Frame0 => {
                self.reference = None;
            }
        }

        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(reference)) = (device, self.reference.as_ref()) {
            let dense_selection: Vec<u32> = (0..self.selection.indices.len())
                .map(|i| i as u32)
                .collect();
            let selection = if self.use_selected_input {
                ctx.selection(&dense_selection, None)?
            } else {
                ctx.selection(&self.selection.indices, None)?
            };
            let reference_gpu = ctx.reference(&convert_coords(reference))?;
            self.gpu = Some(RmsdGpuState {
                selection,
                reference: reference_gpu,
            });
        }
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(self.selection.indices.as_slice())
        } else {
            None
        }
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        Some(self.selection.indices.as_slice())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        let _ = device;
        let n_atoms = chunk.n_atoms;
        if self.reference.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0) {
            if chunk.n_frames == 0 {
                return Ok(());
            }
            let mut reference = Vec::with_capacity(self.selection_usize.len());
            for &idx in self.selection_usize.iter() {
                reference.push(chunk.coords[idx]);
            }
            self.reference = Some(reference);

            #[cfg(feature = "cuda")]
            if let Device::Cuda(ctx) = device {
                let dense_selection: Vec<u32> = (0..self.selection.indices.len())
                    .map(|i| i as u32)
                    .collect();
                let selection = if self.use_selected_input {
                    ctx.selection(&dense_selection, None)?
                } else {
                    ctx.selection(&self.selection.indices, None)?
                };
                let reference_gpu =
                    ctx.reference(&convert_coords(self.reference.as_ref().unwrap()))?;
                self.gpu = Some(RmsdGpuState {
                    selection,
                    reference: reference_gpu,
                });
            }
        }

        let reference = self
            .reference
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;

        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            if self.align {
                let cov = ctx.rmsd_covariance(
                    &coords,
                    n_atoms,
                    chunk.n_frames,
                    &gpu.selection,
                    &gpu.reference,
                )?;
                let n_sel = gpu.selection.n_sel();
                for frame in 0..chunk.n_frames {
                    let rmsd = rmsd_from_cov(
                        &cov.cov[frame],
                        cov.sum_x2[frame] as f64,
                        cov.sum_y2[frame] as f64,
                        n_sel,
                    );
                    self.results.push(rmsd);
                }
            } else {
                let results = ctx.rmsd_raw(
                    &coords,
                    n_atoms,
                    chunk.n_frames,
                    &gpu.selection,
                    &gpu.reference,
                )?;
                self.results.extend(results);
            }
            return Ok(());
        }

        process_cpu_chunk_out(
            chunk,
            n_atoms,
            reference,
            &self.selection_usize,
            self.align,
            &mut self.results,
        );
        Ok(())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        _system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        let _ = device;
        if source_selection != self.selection.indices.as_slice() {
            return Err(TrajError::Mismatch(
                "rmsd selected chunk does not match expected IO selection".into(),
            ));
        }
        if self.reference.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0) {
            if chunk.n_frames == 0 {
                return Ok(());
            }
            let end = chunk.n_atoms.min(chunk.coords.len());
            self.reference = Some(chunk.coords[..end].to_vec());

            #[cfg(feature = "cuda")]
            if let Device::Cuda(ctx) = device {
                let dense_selection: Vec<u32> = (0..chunk.n_atoms).map(|i| i as u32).collect();
                let selection = ctx.selection(&dense_selection, None)?;
                let reference_gpu =
                    ctx.reference(&convert_coords(self.reference.as_ref().unwrap()))?;
                self.gpu = Some(RmsdGpuState {
                    selection,
                    reference: reference_gpu,
                });
            }
        }

        let reference = self
            .reference
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;

        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            if self.align {
                let cov = ctx.rmsd_covariance(
                    &coords,
                    chunk.n_atoms,
                    chunk.n_frames,
                    &gpu.selection,
                    &gpu.reference,
                )?;
                let n_sel = gpu.selection.n_sel();
                for frame in 0..chunk.n_frames {
                    let rmsd = rmsd_from_cov(
                        &cov.cov[frame],
                        cov.sum_x2[frame] as f64,
                        cov.sum_y2[frame] as f64,
                        n_sel,
                    );
                    self.results.push(rmsd);
                }
            } else {
                let results = ctx.rmsd_raw(
                    &coords,
                    chunk.n_atoms,
                    chunk.n_frames,
                    &gpu.selection,
                    &gpu.reference,
                )?;
                self.results.extend(results);
            }
            return Ok(());
        }

        process_cpu_chunk_out(
            chunk,
            chunk.n_atoms,
            reference,
            &self.dense_selection_usize,
            self.align,
            &mut self.results,
        );
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

impl Plan for SymmRmsdPlan {
    fn name(&self) -> &'static str {
        "symmrmsd"
    }

    fn requirements(&self) -> PlanRequirements {
        self.inner.requirements()
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.inner.init(system, device)
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        self.inner.preferred_selection()
    }

    fn preferred_selection_hint(&self, system: &System) -> Option<&[u32]> {
        self.inner.preferred_selection_hint(system)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        if !self.has_remap() {
            return self.inner.process_chunk(chunk, system, device);
        }
        ensure_symm_reference(&mut self.inner, chunk, false)?;
        let reference = self
            .inner
            .reference
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;
        process_symm_cpu_chunk(
            chunk,
            chunk.n_atoms,
            reference,
            &self.inner.selection_usize,
            &self.symmetry_groups,
            self.inner.align,
            &mut self.inner.results,
        );
        Ok(())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        if !self.has_remap() {
            return self
                .inner
                .process_chunk_selected(chunk, source_selection, system, device);
        }
        if source_selection != self.inner.selection.indices.as_slice() {
            return Err(TrajError::Mismatch(
                "symmrmsd selected chunk does not match expected IO selection".into(),
            ));
        }
        if chunk.n_atoms != self.inner.selection.indices.len() {
            return Err(TrajError::Mismatch(
                "symmrmsd selected chunk atom count does not match selection".into(),
            ));
        }
        ensure_symm_reference(&mut self.inner, chunk, true)?;
        let reference = self
            .inner
            .reference
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;
        process_symm_cpu_chunk(
            chunk,
            chunk.n_atoms,
            reference,
            &self.inner.dense_selection_usize,
            &self.symmetry_groups,
            self.inner.align,
            &mut self.inner.results,
        );
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        self.inner.finalize()
    }
}

impl Plan for PairwiseRmsdPlan {
    fn name(&self) -> &'static str {
        "pairwise_rmsd"
    }

    fn requirements(&self) -> PlanRequirements {
        let needs_box =
            matches!(self.metric, PairwiseMetric::Dme) && matches!(self.pbc, PbcMode::Orthorhombic);
        PlanRequirements::new(needs_box, false)
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.use_selected_input = matches!(_device, Device::Cpu);
        self.frames.clear();
        self.boxes.clear();
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(self.selection.indices.as_slice())
        } else {
            None
        }
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        Some(self.selection.indices.as_slice())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        for frame in 0..chunk.n_frames {
            let mut frame_sel = Vec::with_capacity(sel.len());
            for &idx in sel.iter() {
                frame_sel.push(chunk.coords[frame * n_atoms + idx as usize]);
            }
            self.frames.push(frame_sel);
            if matches!(self.metric, PairwiseMetric::Dme)
                && matches!(self.pbc, PbcMode::Orthorhombic)
            {
                self.boxes.push(Some(box_lengths(chunk.box_[frame])?));
            } else {
                self.boxes.push(None);
            }
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
                "pairwise_rmsd selected chunk received while selected IO is disabled".into(),
            ));
        }
        if source_selection != self.selection.indices.as_slice() {
            return Err(TrajError::Mismatch(
                "pairwise_rmsd selected chunk does not match expected IO selection".into(),
            ));
        }

        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let start = frame * n_atoms;
            let end = start + n_atoms;
            self.frames.push(chunk.coords[start..end].to_vec());
            if matches!(self.metric, PairwiseMetric::Dme)
                && matches!(self.pbc, PbcMode::Orthorhombic)
            {
                self.boxes.push(Some(box_lengths(chunk.box_[frame])?));
            } else {
                self.boxes.push(None);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let n_frames = self.frames.len();
        if n_frames == 0 {
            return Ok(PlanOutput::Matrix {
                data: Vec::new(),
                rows: 0,
                cols: 0,
            });
        }
        let mut data = vec![0.0f32; n_frames * n_frames];
        let n_sel = self.selection.indices.len();
        if n_sel < 2 {
            return Ok(PlanOutput::Matrix {
                data,
                rows: n_frames,
                cols: n_frames,
            });
        }

        if matches!(self.metric, PairwiseMetric::Dme) {
            let mut dists = Vec::with_capacity(n_frames);
            for (idx, frame) in self.frames.iter().enumerate() {
                let box_ = if matches!(self.pbc, PbcMode::Orthorhombic) {
                    self.boxes[idx].ok_or_else(|| {
                        TrajError::Mismatch("pairwise_rmsd requires orthorhombic box".into())
                    })?
                } else {
                    (0.0, 0.0, 0.0)
                };
                let dist = pair_distances_compact(
                    frame,
                    self.pbc,
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        Some(box_)
                    } else {
                        None
                    },
                );
                dists.push(dist);
            }
            let n_pairs = dists[0].len();
            for i in 0..n_frames {
                for j in (i + 1)..n_frames {
                    let mut sum = 0.0f64;
                    for k in 0..n_pairs {
                        let diff = dists[i][k] - dists[j][k];
                        sum += diff * diff;
                    }
                    let rmsd = if n_pairs == 0 {
                        0.0
                    } else {
                        (sum / n_pairs as f64).sqrt() as f32
                    };
                    data[i * n_frames + j] = rmsd;
                    data[j * n_frames + i] = rmsd;
                }
            }
            return Ok(PlanOutput::Matrix {
                data,
                rows: n_frames,
                cols: n_frames,
            });
        }

        for i in 0..n_frames {
            for j in (i + 1)..n_frames {
                let rmsd = match self.metric {
                    PairwiseMetric::Rms => rmsd_aligned(&self.frames[i], &self.frames[j]),
                    PairwiseMetric::Nofit => rmsd_raw(&self.frames[i], &self.frames[j]),
                    PairwiseMetric::Dme => 0.0,
                };
                data[i * n_frames + j] = rmsd;
                data[j * n_frames + i] = rmsd;
            }
        }

        Ok(PlanOutput::Matrix {
            data,
            rows: n_frames,
            cols: n_frames,
        })
    }
}
impl Plan for DistanceRmsdPlan {
    fn name(&self) -> &'static str {
        "distance_rmsd"
    }

    fn requirements(&self) -> PlanRequirements {
        let needs_box = matches!(self.pbc, PbcMode::Orthorhombic);
        PlanRequirements::new(needs_box, false)
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.reference_dists = None;
        match self.reference_mode {
            ReferenceMode::Topology => {
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    return Err(TrajError::Mismatch(
                        "distance_rmsd with topology reference does not support PBC".into(),
                    ));
                }
                let positions0 = system
                    .positions0
                    .as_ref()
                    .ok_or_else(|| TrajError::Mismatch("no topology reference coords".into()))?;
                let ref_dists =
                    pair_distances(positions0, &self.selection.indices, PbcMode::None, None);
                self.reference_dists = Some(ref_dists);
            }
            ReferenceMode::Frame0 => {}
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
        if self.reference_dists.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0) {
            if chunk.n_frames == 0 {
                return Ok(());
            }
            let box_ = if matches!(self.pbc, PbcMode::Orthorhombic) {
                Some(box_lengths(chunk.box_[0])?)
            } else {
                None
            };
            let coords = &chunk.coords[0..n_atoms];
            let ref_dists = pair_distances(coords, &self.selection.indices, self.pbc, box_);
            self.reference_dists = Some(ref_dists);
        }

        let reference = self
            .reference_dists
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;
        let n_pairs = reference.len();
        if n_pairs == 0 {
            self.results
                .extend(std::iter::repeat(0.0).take(chunk.n_frames));
            return Ok(());
        }

        for frame in 0..chunk.n_frames {
            let box_ = if matches!(self.pbc, PbcMode::Orthorhombic) {
                Some(box_lengths(chunk.box_[frame])?)
            } else {
                None
            };
            let mut sum = 0.0f64;
            let mut idx = 0usize;
            let sel = &self.selection.indices;
            for i in 0..sel.len() {
                let a_idx = sel[i] as usize;
                let pa = chunk.coords[frame * n_atoms + a_idx];
                for j in (i + 1)..sel.len() {
                    let b_idx = sel[j] as usize;
                    let pb = chunk.coords[frame * n_atoms + b_idx];
                    let mut dx = (pb[0] - pa[0]) as f64;
                    let mut dy = (pb[1] - pa[1]) as f64;
                    let mut dz = (pb[2] - pa[2]) as f64;
                    if let Some((lx, ly, lz)) = box_ {
                        apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                    }
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    let diff = dist - reference[idx];
                    sum += diff * diff;
                    idx += 1;
                }
            }
            let rmsd = (sum / n_pairs as f64).sqrt() as f32;
            self.results.push(rmsd);
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

fn ensure_symm_reference(
    inner: &mut RmsdPlan,
    chunk: &FrameChunk,
    selected_input: bool,
) -> TrajResult<()> {
    if inner.reference.is_some() || !matches!(inner.reference_mode, ReferenceMode::Frame0) {
        return Ok(());
    }
    if chunk.n_frames == 0 {
        return Ok(());
    }
    if selected_input {
        let end = chunk.n_atoms.min(chunk.coords.len());
        inner.reference = Some(chunk.coords[..end].to_vec());
        return Ok(());
    }

    let mut reference = Vec::with_capacity(inner.selection_usize.len());
    for &idx in inner.selection_usize.iter() {
        if idx >= chunk.n_atoms {
            return Err(TrajError::Mismatch(
                "symmrmsd reference selection index out of frame bounds".into(),
            ));
        }
        reference.push(chunk.coords[idx]);
    }
    inner.reference = Some(reference);
    Ok(())
}

fn process_symm_cpu_chunk(
    chunk: &FrameChunk,
    n_atoms: usize,
    reference: &[[f32; 4]],
    selection: &[usize],
    groups: &[Vec<usize>],
    align: bool,
    out: &mut Vec<f32>,
) {
    for frame in 0..chunk.n_frames {
        let frame_coords = &chunk.coords[frame * n_atoms..(frame + 1) * n_atoms];
        out.push(best_remapped_rmsd(
            frame_coords,
            reference,
            selection,
            groups,
            align,
        ));
    }
}

fn best_remapped_rmsd(
    frame: &[[f32; 4]],
    reference: &[[f32; 4]],
    selection: &[usize],
    groups: &[Vec<usize>],
    align: bool,
) -> f32 {
    let n = selection.len().min(reference.len());
    if n == 0 {
        return 0.0;
    }
    if groups.iter().any(|group| group.iter().any(|&pos| pos >= n)) {
        return 0.0;
    }
    let mut mapping: Vec<usize> = (0..n).collect();
    let mut best = f32::INFINITY;
    enumerate_symm_groups(
        0,
        groups,
        &mut mapping,
        frame,
        reference,
        selection,
        align,
        &mut best,
    );
    if best.is_finite() {
        best
    } else {
        0.0
    }
}

fn enumerate_symm_groups(
    group_index: usize,
    groups: &[Vec<usize>],
    mapping: &mut [usize],
    frame: &[[f32; 4]],
    reference: &[[f32; 4]],
    selection: &[usize],
    align: bool,
    best: &mut f32,
) {
    if group_index == groups.len() {
        let value = if align {
            rmsd_aligned_mapped(frame, reference, selection, mapping)
        } else {
            rmsd_raw_mapped(frame, reference, selection, mapping)
        };
        if value < *best {
            *best = value;
        }
        return;
    }

    let group = &groups[group_index];
    let mut values = group.clone();
    enumerate_group_values(
        0,
        group,
        &mut values,
        group_index,
        groups,
        mapping,
        frame,
        reference,
        selection,
        align,
        best,
    );
}

fn enumerate_group_values(
    value_index: usize,
    group: &[usize],
    values: &mut [usize],
    group_index: usize,
    groups: &[Vec<usize>],
    mapping: &mut [usize],
    frame: &[[f32; 4]],
    reference: &[[f32; 4]],
    selection: &[usize],
    align: bool,
    best: &mut f32,
) {
    if value_index == values.len() {
        for (slot, source_pos) in group.iter().zip(values.iter()) {
            mapping[*slot] = *source_pos;
        }
        enumerate_symm_groups(
            group_index + 1,
            groups,
            mapping,
            frame,
            reference,
            selection,
            align,
            best,
        );
        return;
    }

    for swap_index in value_index..values.len() {
        values.swap(value_index, swap_index);
        enumerate_group_values(
            value_index + 1,
            group,
            values,
            group_index,
            groups,
            mapping,
            frame,
            reference,
            selection,
            align,
            best,
        );
        values.swap(value_index, swap_index);
    }
}

fn rmsd_raw_mapped(
    frame: &[[f32; 4]],
    reference: &[[f32; 4]],
    selection: &[usize],
    mapping: &[usize],
) -> f32 {
    let n = selection.len().min(reference.len()).min(mapping.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for ref_pos in 0..n {
        let source_pos = mapping[ref_pos];
        if source_pos >= selection.len() {
            return 0.0;
        }
        let atom_idx = selection[source_pos];
        if atom_idx >= frame.len() {
            return 0.0;
        }
        let p = frame[atom_idx];
        let q = reference[ref_pos];
        let dx = p[0] as f64 - q[0] as f64;
        let dy = p[1] as f64 - q[1] as f64;
        let dz = p[2] as f64 - q[2] as f64;
        sum += dx * dx + dy * dy + dz * dz;
    }
    ((sum / n as f64).sqrt()) as f32
}

fn rmsd_aligned_mapped(
    frame: &[[f32; 4]],
    reference: &[[f32; 4]],
    selection: &[usize],
    mapping: &[usize],
) -> f32 {
    let n = selection.len().min(reference.len()).min(mapping.len());
    if n == 0 {
        return 0.0;
    }

    let mut cx = [0.0f64; 3];
    let mut cy = [0.0f64; 3];
    for ref_pos in 0..n {
        let source_pos = mapping[ref_pos];
        if source_pos >= selection.len() {
            return 0.0;
        }
        let atom_idx = selection[source_pos];
        if atom_idx >= frame.len() {
            return 0.0;
        }
        let p = frame[atom_idx];
        let q = reference[ref_pos];
        cx[0] += p[0] as f64;
        cx[1] += p[1] as f64;
        cx[2] += p[2] as f64;
        cy[0] += q[0] as f64;
        cy[1] += q[1] as f64;
        cy[2] += q[2] as f64;
    }
    let inv_n = 1.0 / n as f64;
    cx[0] *= inv_n;
    cx[1] *= inv_n;
    cx[2] *= inv_n;
    cy[0] *= inv_n;
    cy[1] *= inv_n;
    cy[2] *= inv_n;

    let mut h: Matrix3<f64> = Matrix3::zeros();
    for ref_pos in 0..n {
        let source_pos = mapping[ref_pos];
        let atom_idx = selection[source_pos];
        let p = frame[atom_idx];
        let q = reference[ref_pos];
        let px = p[0] as f64 - cx[0];
        let py = p[1] as f64 - cx[1];
        let pz = p[2] as f64 - cx[2];
        let qx = q[0] as f64 - cy[0];
        let qy = q[1] as f64 - cy[1];
        let qz = q[2] as f64 - cy[2];

        h[(0, 0)] += px * qx;
        h[(0, 1)] += px * qy;
        h[(0, 2)] += px * qz;
        h[(1, 0)] += py * qx;
        h[(1, 1)] += py * qy;
        h[(1, 2)] += py * qz;
        h[(2, 0)] += pz * qx;
        h[(2, 1)] += pz * qy;
        h[(2, 2)] += pz * qz;
    }

    let svd = h.svd(true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => (u, v_t),
        _ => return rmsd_raw_mapped(frame, reference, selection, mapping),
    };
    let mut r: Matrix3<f64> = v_t.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        let mut v_t_adj = v_t;
        v_t_adj.row_mut(2).neg_mut();
        r = v_t_adj.transpose() * u.transpose();
    }

    let mut sum = 0.0f64;
    for ref_pos in 0..n {
        let source_pos = mapping[ref_pos];
        let atom_idx = selection[source_pos];
        let p = frame[atom_idx];
        let q = reference[ref_pos];
        let px = p[0] as f64 - cx[0];
        let py = p[1] as f64 - cx[1];
        let pz = p[2] as f64 - cx[2];
        let qx = q[0] as f64 - cy[0];
        let qy = q[1] as f64 - cy[1];
        let qz = q[2] as f64 - cy[2];

        let ax = r[(0, 0)] * px + r[(0, 1)] * py + r[(0, 2)] * pz;
        let ay = r[(1, 0)] * px + r[(1, 1)] * py + r[(1, 2)] * pz;
        let az = r[(2, 0)] * px + r[(2, 1)] * py + r[(2, 2)] * pz;

        let dx = ax - qx;
        let dy = ay - qy;
        let dz = az - qz;
        sum += dx * dx + dy * dy + dz * dz;
    }
    ((sum * inv_n).sqrt()) as f32
}

fn rmsd_raw(frame: &[[f32; 4]], reference: &[[f32; 4]]) -> f32 {
    let mut sum = 0.0f64;
    let n = frame.len().min(reference.len());
    for i in 0..n {
        let dx = frame[i][0] as f64 - reference[i][0] as f64;
        let dy = frame[i][1] as f64 - reference[i][1] as f64;
        let dz = frame[i][2] as f64 - reference[i][2] as f64;
        sum += dx * dx + dy * dy + dz * dz;
    }
    ((sum / n as f64).sqrt()) as f32
}

fn rmsd_raw_selected(frame: &[[f32; 4]], reference: &[[f32; 4]], selection: &[usize]) -> f32 {
    let n = selection.len().min(reference.len());
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for i in 0..n {
        let idx = selection[i];
        if idx >= frame.len() {
            break;
        }
        let dx = frame[idx][0] as f64 - reference[i][0] as f64;
        let dy = frame[idx][1] as f64 - reference[i][1] as f64;
        let dz = frame[idx][2] as f64 - reference[i][2] as f64;
        sum += dx * dx + dy * dy + dz * dz;
    }
    ((sum / n as f64).sqrt()) as f32
}

fn rmsd_aligned(frame: &[[f32; 4]], reference: &[[f32; 4]]) -> f32 {
    let n = frame.len().min(reference.len());
    if n == 0 {
        return 0.0;
    }
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        x.push(Vector3::new(
            frame[i][0] as f64,
            frame[i][1] as f64,
            frame[i][2] as f64,
        ));
        y.push(Vector3::new(
            reference[i][0] as f64,
            reference[i][1] as f64,
            reference[i][2] as f64,
        ));
    }
    let cx = centroid(&x);
    let cy = centroid(&y);
    let mut h: Matrix3<f64> = Matrix3::zeros();
    for i in 0..n {
        let xr = x[i] - cx;
        let yr = y[i] - cy;
        h += xr * yr.transpose();
    }
    let svd = h.svd(true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => (u, v_t),
        _ => return rmsd_raw(frame, reference),
    };
    let mut r: Matrix3<f64> = v_t.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        let mut v_t_adj = v_t;
        v_t_adj.row_mut(2).neg_mut();
        r = v_t_adj.transpose() * u.transpose();
    }
    let mut sum = 0.0f64;
    for i in 0..n {
        let xr = x[i] - cx;
        let yr = y[i] - cy;
        let aligned = r * xr;
        let diff = aligned - yr;
        sum += diff.dot(&diff);
    }
    ((sum / n as f64).sqrt()) as f32
}

fn rmsd_aligned_selected(frame: &[[f32; 4]], reference: &[[f32; 4]], selection: &[usize]) -> f32 {
    let n = selection.len().min(reference.len());
    if n == 0 {
        return 0.0;
    }

    let mut cx = [0.0f64; 3];
    let mut cy = [0.0f64; 3];
    for i in 0..n {
        let idx = selection[i];
        if idx >= frame.len() {
            return 0.0;
        }
        let p = frame[idx];
        let q = reference[i];
        cx[0] += p[0] as f64;
        cx[1] += p[1] as f64;
        cx[2] += p[2] as f64;
        cy[0] += q[0] as f64;
        cy[1] += q[1] as f64;
        cy[2] += q[2] as f64;
    }
    let inv_n = 1.0 / n as f64;
    cx[0] *= inv_n;
    cx[1] *= inv_n;
    cx[2] *= inv_n;
    cy[0] *= inv_n;
    cy[1] *= inv_n;
    cy[2] *= inv_n;

    let mut h: Matrix3<f64> = Matrix3::zeros();
    for i in 0..n {
        let idx = selection[i];
        let p = frame[idx];
        let q = reference[i];
        let px = p[0] as f64 - cx[0];
        let py = p[1] as f64 - cx[1];
        let pz = p[2] as f64 - cx[2];
        let qx = q[0] as f64 - cy[0];
        let qy = q[1] as f64 - cy[1];
        let qz = q[2] as f64 - cy[2];

        h[(0, 0)] += px * qx;
        h[(0, 1)] += px * qy;
        h[(0, 2)] += px * qz;
        h[(1, 0)] += py * qx;
        h[(1, 1)] += py * qy;
        h[(1, 2)] += py * qz;
        h[(2, 0)] += pz * qx;
        h[(2, 1)] += pz * qy;
        h[(2, 2)] += pz * qz;
    }

    let svd = h.svd(true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => (u, v_t),
        _ => return rmsd_raw_selected(frame, reference, selection),
    };
    let mut r: Matrix3<f64> = v_t.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        let mut v_t_adj = v_t;
        v_t_adj.row_mut(2).neg_mut();
        r = v_t_adj.transpose() * u.transpose();
    }

    let mut sum = 0.0f64;
    for i in 0..n {
        let idx = selection[i];
        let p = frame[idx];
        let q = reference[i];
        let px = p[0] as f64 - cx[0];
        let py = p[1] as f64 - cx[1];
        let pz = p[2] as f64 - cx[2];
        let qx = q[0] as f64 - cy[0];
        let qy = q[1] as f64 - cy[1];
        let qz = q[2] as f64 - cy[2];

        let ax = r[(0, 0)] * px + r[(0, 1)] * py + r[(0, 2)] * pz;
        let ay = r[(1, 0)] * px + r[(1, 1)] * py + r[(1, 2)] * pz;
        let az = r[(2, 0)] * px + r[(2, 1)] * py + r[(2, 2)] * pz;

        let dx = ax - qx;
        let dy = ay - qy;
        let dz = az - qz;
        sum += dx * dx + dy * dy + dz * dz;
    }
    ((sum * inv_n).sqrt()) as f32
}

fn pair_distances(
    coords: &[[f32; 4]],
    sel: &[u32],
    pbc: PbcMode,
    box_: Option<(f64, f64, f64)>,
) -> Vec<f64> {
    let n_pairs = sel.len().saturating_sub(1) * sel.len() / 2;
    let mut out = Vec::with_capacity(n_pairs);
    for i in 0..sel.len() {
        let a_idx = sel[i] as usize;
        let pa = coords[a_idx];
        for j in (i + 1)..sel.len() {
            let b_idx = sel[j] as usize;
            let pb = coords[b_idx];
            let mut dx = (pb[0] - pa[0]) as f64;
            let mut dy = (pb[1] - pa[1]) as f64;
            let mut dz = (pb[2] - pa[2]) as f64;
            if matches!(pbc, PbcMode::Orthorhombic) {
                if let Some((lx, ly, lz)) = box_ {
                    apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                }
            }
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            out.push(dist);
        }
    }
    out
}

fn pair_distances_compact(
    coords: &[[f32; 4]],
    pbc: PbcMode,
    box_: Option<(f64, f64, f64)>,
) -> Vec<f64> {
    let n_pairs = coords.len().saturating_sub(1) * coords.len() / 2;
    let mut out = Vec::with_capacity(n_pairs);
    let (lx, ly, lz) = box_.unwrap_or((0.0, 0.0, 0.0));
    for i in 0..coords.len() {
        let pa = coords[i];
        for j in (i + 1)..coords.len() {
            let pb = coords[j];
            let mut dx = (pb[0] - pa[0]) as f64;
            let mut dy = (pb[1] - pa[1]) as f64;
            let mut dz = (pb[2] - pa[2]) as f64;
            if matches!(pbc, PbcMode::Orthorhombic) {
                apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
            }
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            out.push(dist);
        }
    }
    out
}
