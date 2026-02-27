use std::collections::VecDeque;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuBufferF32, GpuBufferU32, GpuContext, GpuGroups, GpuSelection};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorsionStat {
    pub trans: f32,
    pub cis: f32,
    pub g_plus: f32,
    pub g_minus: f32,
}

pub struct TorsionDiffusionPlan {
    selection: Selection,
    results: Vec<TorsionStat>,
    #[cfg(feature = "cuda")]
    gpu: Option<TorsionGpuState>,
}

pub struct ToroidalDiffusionPlan {
    selection: Selection,
    results: Vec<TorsionStat>,
    mass_weighted: bool,
    emit_transitions: bool,
    store_transition_states: bool,
    transition_lag: usize,
    transition_counts: [u64; 16],
    transition_states: Vec<u8>,
    transition_state_rows: usize,
    n_torsions: usize,
    state_window: VecDeque<Vec<u8>>,
    #[cfg(feature = "cuda")]
    gpu: Option<TorsionGpuState>,
}

pub struct MultiPuckerPlan {
    selection: Selection,
    results: Vec<f32>,
    frames: usize,
    bins: usize,
    mode: MultiPuckerMode,
    range_max: Option<f32>,
    normalize: bool,
    histogram_distances: Vec<f32>,
    histogram_auto_range_max: f64,
    #[cfg(feature = "cuda")]
    gpu: Option<MultiPuckerGpuState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiPuckerMode {
    Legacy,
    Histogram,
}

#[cfg(feature = "cuda")]
struct MultiPuckerGpuState {
    ctx: GpuContext,
    sel: GpuSelection,
    groups: GpuGroups,
    masses: GpuBufferF32,
}

#[cfg(feature = "cuda")]
struct TorsionGpuState {
    torsions: GpuBufferU32,
    n_torsions: usize,
}

impl TorsionDiffusionPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            results: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }
}

impl ToroidalDiffusionPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            results: Vec::new(),
            mass_weighted: false,
            emit_transitions: false,
            store_transition_states: false,
            transition_lag: 1,
            transition_counts: [0u64; 16],
            transition_states: Vec::new(),
            transition_state_rows: 0,
            n_torsions: 0,
            state_window: VecDeque::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_mass_weighted(mut self, mass_weighted: bool) -> Self {
        self.mass_weighted = mass_weighted;
        self
    }

    pub fn with_emit_transitions(mut self, emit_transitions: bool) -> Self {
        self.emit_transitions = emit_transitions;
        self
    }

    pub fn with_store_transition_states(mut self, store_transition_states: bool) -> Self {
        self.store_transition_states = store_transition_states;
        self
    }

    pub fn with_transition_lag(mut self, transition_lag: usize) -> Self {
        self.transition_lag = transition_lag.max(1);
        self
    }

    pub fn transition_counts_flat(&self) -> [u64; 16] {
        self.transition_counts
    }

    pub fn transition_matrix_flat(&self) -> [f32; 16] {
        let mut probs = [0.0f32; 16];
        for row in 0..4 {
            let row_base = row * 4;
            let row_sum: u64 = self.transition_counts[row_base..row_base + 4].iter().sum();
            if row_sum == 0 {
                continue;
            }
            let denom = row_sum as f32;
            for col in 0..4 {
                probs[row_base + col] = self.transition_counts[row_base + col] as f32 / denom;
            }
        }
        probs
    }

    pub fn transition_rate(&self) -> f32 {
        let total: u64 = self.transition_counts.iter().sum();
        if total == 0 {
            return 0.0;
        }
        let diag = self.transition_counts[0]
            + self.transition_counts[5]
            + self.transition_counts[10]
            + self.transition_counts[15];
        (total.saturating_sub(diag)) as f32 / total as f32
    }

    pub fn transition_states_flat(&self) -> &[u8] {
        &self.transition_states
    }

    pub fn transition_state_rows(&self) -> usize {
        self.transition_state_rows
    }

    pub fn transition_state_cols(&self) -> usize {
        self.n_torsions
    }

    fn push_transition_state(&mut self, state_row: &[u8]) {
        if !self.emit_transitions || self.n_torsions == 0 {
            return;
        }
        if self.state_window.len() == self.transition_lag {
            if let Some(prev) = self.state_window.pop_front() {
                for (src, dst) in prev.iter().zip(state_row.iter()) {
                    let src = *src as usize;
                    let dst = *dst as usize;
                    if src < 4 && dst < 4 {
                        self.transition_counts[src * 4 + dst] += 1;
                    }
                }
            }
        }
        self.state_window.push_back(state_row.to_vec());
    }
}

impl MultiPuckerPlan {
    pub fn new(selection: Selection, bins: usize) -> Self {
        Self {
            selection,
            results: Vec::new(),
            frames: 0,
            bins: bins.max(1),
            mode: MultiPuckerMode::Legacy,
            range_max: None,
            normalize: true,
            histogram_distances: Vec::new(),
            histogram_auto_range_max: 0.0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_mode(mut self, mode: MultiPuckerMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_range_max(mut self, range_max: Option<f32>) -> Self {
        self.range_max = range_max;
        self
    }

    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

impl Plan for TorsionDiffusionPlan {
    fn name(&self) -> &'static str {
        "tordiff"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let n_torsions = self.selection.indices.len() / 4;
                if n_torsions > 0 {
                    let indices = self.selection.indices[..n_torsions * 4].to_vec();
                    let torsions = ctx.upload_u32(&indices)?;
                    self.gpu = Some(TorsionGpuState {
                        torsions,
                        n_torsions,
                    });
                }
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
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            if gpu.n_torsions == 0 {
                self.results.extend(
                    std::iter::repeat(TorsionStat {
                        trans: 0.0,
                        cis: 0.0,
                        g_plus: 0.0,
                        g_minus: 0.0,
                    })
                    .take(chunk.n_frames),
                );
                return Ok(());
            }
            let coords = convert_coords(&chunk.coords);
            let counts = ctx.torsion_diffusion_counts(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.torsions,
                gpu.n_torsions,
            )?;
            let denom = gpu.n_torsions.max(1) as f32;
            for frame in 0..chunk.n_frames {
                let base = frame * 4;
                self.results.push(TorsionStat {
                    trans: counts[base] as f32 / denom,
                    cis: counts[base + 1] as f32 / denom,
                    g_plus: counts[base + 2] as f32 / denom,
                    g_minus: counts[base + 3] as f32 / denom,
                });
            }
            return Ok(());
        }
        let stats = torsion_stats(chunk, system, &self.selection)?;
        self.results.extend(stats);
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let mut data = Vec::with_capacity(self.results.len() * 4);
        for stat in self.results.drain(..) {
            data.push(stat.trans);
            data.push(stat.cis);
            data.push(stat.g_plus);
            data.push(stat.g_minus);
        }
        let rows = data.len() / 4;
        Ok(PlanOutput::Matrix {
            data,
            rows,
            cols: 4,
        })
    }
}

impl Plan for ToroidalDiffusionPlan {
    fn name(&self) -> &'static str {
        "toroidal_diffusion"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.transition_counts = [0u64; 16];
        self.transition_states.clear();
        self.transition_state_rows = 0;
        self.state_window.clear();
        self.n_torsions = self.selection.indices.len() / 4;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if !self.mass_weighted && !self.emit_transitions {
                if let Device::Cuda(ctx) = _device {
                    let n_torsions = self.selection.indices.len() / 4;
                    if n_torsions > 0 {
                        let indices = self.selection.indices[..n_torsions * 4].to_vec();
                        let torsions = ctx.upload_u32(&indices)?;
                        self.gpu = Some(TorsionGpuState {
                            torsions,
                            n_torsions,
                        });
                    }
                }
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
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            if gpu.n_torsions == 0 {
                self.results.extend(
                    std::iter::repeat(TorsionStat {
                        trans: 0.0,
                        cis: 0.0,
                        g_plus: 0.0,
                        g_minus: 0.0,
                    })
                    .take(chunk.n_frames),
                );
                return Ok(());
            }
            let coords = convert_coords(&chunk.coords);
            let counts = ctx.torsion_diffusion_counts(
                &coords,
                chunk.n_atoms,
                chunk.n_frames,
                &gpu.torsions,
                gpu.n_torsions,
            )?;
            let denom = gpu.n_torsions.max(1) as f32;
            for frame in 0..chunk.n_frames {
                let base = frame * 4;
                self.results.push(TorsionStat {
                    trans: counts[base] as f32 / denom,
                    cis: counts[base + 1] as f32 / denom,
                    g_plus: counts[base + 2] as f32 / denom,
                    g_minus: counts[base + 3] as f32 / denom,
                });
            }
            return Ok(());
        }
        let detailed = torsion_stats_detailed(
            chunk,
            system,
            &self.selection,
            self.mass_weighted,
            self.emit_transitions,
        )?;
        self.results.extend(detailed.stats.into_iter());
        if self.emit_transitions {
            for row in detailed.states.iter() {
                self.push_transition_state(row);
                if self.store_transition_states {
                    self.transition_states.extend_from_slice(row);
                    self.transition_state_rows += 1;
                }
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let mut data = Vec::with_capacity(self.results.len() * 4);
        for stat in self.results.drain(..) {
            data.push(stat.trans);
            data.push(stat.cis);
            data.push(stat.g_plus);
            data.push(stat.g_minus);
        }
        let rows = data.len() / 4;
        Ok(PlanOutput::Matrix {
            data,
            rows,
            cols: 4,
        })
    }
}

impl Plan for MultiPuckerPlan {
    fn name(&self) -> &'static str {
        "multipucker"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        self.histogram_distances.clear();
        self.histogram_auto_range_max = 0.0;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                if !self.selection.indices.is_empty() {
                    let offsets = vec![0u32, self.selection.indices.len() as u32];
                    let indices = self.selection.indices.clone();
                    let groups = ctx.groups(&offsets, &indices, indices.len())?;
                    let masses = vec![1.0f32; _system.n_atoms()];
                    let masses = ctx.upload_f32(&masses)?;
                    let sel = ctx.selection(&self.selection.indices, None)?;
                    self.gpu = Some(MultiPuckerGpuState {
                        ctx: ctx.clone(),
                        sel,
                        groups,
                        masses,
                    });
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
        let sel = &self.selection.indices;
        if sel.is_empty() {
            for _ in 0..chunk.n_frames {
                self.results
                    .extend(std::iter::repeat(0.0f32).take(self.bins));
            }
            self.frames += chunk.n_frames;
            return Ok(());
        }

        let hist_range = if matches!(self.mode, MultiPuckerMode::Histogram) {
            if let Some(range_max) = self.range_max {
                if range_max <= 0.0 {
                    return Err(TrajError::Parse(
                        "multipucker histogram range_max must be > 0".into(),
                    ));
                }
                Some(range_max as f64)
            } else {
                None
            }
        } else {
            None
        };

        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
            if matches!(self.mode, MultiPuckerMode::Legacy) {
                let coords = convert_coords(&chunk.coords);
                let coms = ctx.group_com(
                    &coords,
                    chunk.n_atoms,
                    chunk.n_frames,
                    &gpu.groups,
                    &gpu.masses,
                    1.0,
                )?;
                let maxes =
                    ctx.max_dist_points(&coords, chunk.n_atoms, chunk.n_frames, &gpu.sel, &coms)?;
                for max_r in maxes {
                    let bin = if max_r == 0.0 { 0 } else { self.bins - 1 };
                    for i in 0..self.bins {
                        self.results.push(if i == bin { 1.0 } else { 0.0 });
                    }
                    self.frames += 1;
                }
                return Ok(());
            }
            if matches!(self.mode, MultiPuckerMode::Histogram) {
                if let Some(range_max) = hist_range {
                    let coords = convert_coords(&chunk.coords);
                    let coms = gpu.ctx.group_com(
                        &coords,
                        chunk.n_atoms,
                        chunk.n_frames,
                        &gpu.groups,
                        &gpu.masses,
                        1.0,
                    )?;
                    let hist = gpu.ctx.multipucker_histogram(
                        &coords,
                        chunk.n_atoms,
                        chunk.n_frames,
                        &gpu.sel,
                        &coms,
                        self.bins,
                        range_max as f32,
                        self.normalize,
                    )?;
                    self.results.extend(hist);
                    self.frames += chunk.n_frames;
                    return Ok(());
                } else {
                    let coords = convert_coords(&chunk.coords);
                    let coms = gpu.ctx.group_com(
                        &coords,
                        chunk.n_atoms,
                        chunk.n_frames,
                        &gpu.groups,
                        &gpu.masses,
                        1.0,
                    )?;
                    let (distances, maxes) = gpu.ctx.multipucker_distances(
                        &coords,
                        chunk.n_atoms,
                        chunk.n_frames,
                        &gpu.sel,
                        &coms,
                    )?;
                    for max_r in maxes {
                        self.histogram_auto_range_max =
                            self.histogram_auto_range_max.max(max_r as f64);
                    }
                    self.histogram_distances.extend(distances);
                    self.frames += chunk.n_frames;
                    return Ok(());
                }
            }
        }
        let mut row = vec![0.0f32; self.bins];
        for frame in 0..chunk.n_frames {
            row.fill(0.0);
            let frame_offset = frame * n_atoms;
            let mut sum = [0.0f64; 3];
            for &idx in sel.iter() {
                let p = chunk.coords[frame_offset + idx as usize];
                sum[0] += p[0] as f64;
                sum[1] += p[1] as f64;
                sum[2] += p[2] as f64;
            }
            let inv_n = 1.0f64 / sel.len() as f64;
            let center = [sum[0] * inv_n, sum[1] * inv_n, sum[2] * inv_n];

            if matches!(self.mode, MultiPuckerMode::Legacy) {
                let mut max_r2 = 0.0f64;
                for &idx in sel.iter() {
                    let p = chunk.coords[frame_offset + idx as usize];
                    let dx = p[0] as f64 - center[0];
                    let dy = p[1] as f64 - center[1];
                    let dz = p[2] as f64 - center[2];
                    let r2 = dx * dx + dy * dy + dz * dz;
                    if r2 > max_r2 {
                        max_r2 = r2;
                    }
                }
                let bin = if max_r2 <= 0.0 { 0 } else { self.bins - 1 };
                row[bin] = 1.0;
                self.results.extend_from_slice(&row);
            } else {
                if let Some(range_max) = hist_range {
                    let scale = self.bins as f64 / range_max;
                    for &idx in sel.iter() {
                        let p = chunk.coords[frame_offset + idx as usize];
                        let dx = p[0] as f64 - center[0];
                        let dy = p[1] as f64 - center[1];
                        let dz = p[2] as f64 - center[2];
                        let r = (dx * dx + dy * dy + dz * dz).sqrt();
                        if r > range_max {
                            continue;
                        }
                        let mut bin = (r * scale).floor() as usize;
                        if bin >= self.bins {
                            bin = self.bins - 1;
                        }
                        row[bin] += 1.0;
                    }
                    if self.normalize {
                        let sum: f32 = row.iter().sum();
                        if sum > 0.0 {
                            for value in row.iter_mut() {
                                *value /= sum;
                            }
                        }
                    }
                    self.results.extend_from_slice(&row);
                } else {
                    for &idx in sel.iter() {
                        let p = chunk.coords[frame_offset + idx as usize];
                        let dx = p[0] as f64 - center[0];
                        let dy = p[1] as f64 - center[1];
                        let dz = p[2] as f64 - center[2];
                        let r = (dx * dx + dy * dy + dz * dz).sqrt();
                        self.histogram_auto_range_max = self.histogram_auto_range_max.max(r);
                        self.histogram_distances.push(r as f32);
                    }
                }
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if matches!(self.mode, MultiPuckerMode::Histogram)
            && self.range_max.is_none()
            && !self.selection.indices.is_empty()
        {
            let sel_len = self.selection.indices.len();
            let expected = self.frames.saturating_mul(sel_len);
            if self.histogram_distances.len() != expected {
                return Err(TrajError::Invalid(
                    "multipucker histogram internal distance buffer mismatch".into(),
                ));
            }
            let range_max = if self.histogram_auto_range_max > 0.0 {
                self.histogram_auto_range_max
            } else {
                1.0
            };
            #[cfg(feature = "cuda")]
            {
                if let Some(gpu) = &self.gpu {
                    let data = gpu.ctx.multipucker_histogram_from_distances(
                        &self.histogram_distances,
                        sel_len,
                        self.frames,
                        self.bins,
                        range_max as f32,
                        self.normalize,
                    )?;
                    self.results = data;
                } else {
                    let scale = self.bins as f64 / range_max;
                    let mut data = vec![0.0f32; self.frames * self.bins];
                    for frame in 0..self.frames {
                        let row_start = frame * self.bins;
                        let row_end = row_start + self.bins;
                        let row = &mut data[row_start..row_end];
                        let dist_start = frame * sel_len;
                        let dist_end = dist_start + sel_len;
                        for &radius in self.histogram_distances[dist_start..dist_end].iter() {
                            let r = radius as f64;
                            if r > range_max {
                                continue;
                            }
                            let mut bin = (r * scale).floor() as usize;
                            if bin >= self.bins {
                                bin = self.bins - 1;
                            }
                            row[bin] += 1.0;
                        }
                        if self.normalize {
                            let sum: f32 = row.iter().sum();
                            if sum > 0.0 {
                                for value in row.iter_mut() {
                                    *value /= sum;
                                }
                            }
                        }
                    }
                    self.results = data;
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let scale = self.bins as f64 / range_max;
                let mut data = vec![0.0f32; self.frames * self.bins];
                for frame in 0..self.frames {
                    let row_start = frame * self.bins;
                    let row_end = row_start + self.bins;
                    let row = &mut data[row_start..row_end];
                    let dist_start = frame * sel_len;
                    let dist_end = dist_start + sel_len;
                    for &radius in self.histogram_distances[dist_start..dist_end].iter() {
                        let r = radius as f64;
                        if r > range_max {
                            continue;
                        }
                        let mut bin = (r * scale).floor() as usize;
                        if bin >= self.bins {
                            bin = self.bins - 1;
                        }
                        row[bin] += 1.0;
                    }
                    if self.normalize {
                        let sum: f32 = row.iter().sum();
                        if sum > 0.0 {
                            for value in row.iter_mut() {
                                *value /= sum;
                            }
                        }
                    }
                }
                self.results = data;
            }
        }
        self.histogram_distances.clear();
        self.histogram_auto_range_max = 0.0;
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: self.bins,
        })
    }
}

fn torsion_stats(
    chunk: &FrameChunk,
    system: &System,
    selection: &Selection,
) -> TrajResult<Vec<TorsionStat>> {
    Ok(torsion_stats_detailed(chunk, system, selection, false, false)?.stats)
}

struct TorsionStatsDetailed {
    stats: Vec<TorsionStat>,
    states: Vec<Vec<u8>>,
}

fn classify_torsion_state(angle_deg: f64) -> u8 {
    if angle_deg > 150.0 || angle_deg < -150.0 {
        0
    } else if angle_deg.abs() < 30.0 {
        1
    } else if angle_deg > 0.0 {
        2
    } else {
        3
    }
}

fn torsion_stats_detailed(
    chunk: &FrameChunk,
    system: &System,
    selection: &Selection,
    mass_weighted: bool,
    collect_states: bool,
) -> TrajResult<TorsionStatsDetailed> {
    let n_atoms = chunk.n_atoms;
    let sel = &selection.indices;
    let n_torsions = sel.len() / 4;
    if n_torsions == 0 {
        let stats = vec![
            TorsionStat {
                trans: 0.0,
                cis: 0.0,
                g_plus: 0.0,
                g_minus: 0.0,
            };
            chunk.n_frames
        ];
        let states = if collect_states {
            vec![Vec::new(); chunk.n_frames]
        } else {
            Vec::new()
        };
        return Ok(TorsionStatsDetailed { stats, states });
    }

    let masses = &system.atoms.mass;
    let use_mass = mass_weighted && masses.len() >= n_atoms;
    let mut stats = Vec::with_capacity(chunk.n_frames);
    let mut states = if collect_states {
        Vec::with_capacity(chunk.n_frames)
    } else {
        Vec::new()
    };

    for frame in 0..chunk.n_frames {
        let frame_offset = frame * n_atoms;
        let mut counts = [0.0f64; 4];
        let mut total = 0.0f64;
        let mut frame_states = if collect_states {
            Vec::with_capacity(n_torsions)
        } else {
            Vec::new()
        };

        for group in sel.chunks_exact(4) {
            let a_idx = group[0] as usize;
            let b_idx = group[1] as usize;
            let c_idx = group[2] as usize;
            let d_idx = group[3] as usize;
            let a = chunk.coords[frame_offset + a_idx];
            let b = chunk.coords[frame_offset + b_idx];
            let c = chunk.coords[frame_offset + c_idx];
            let d = chunk.coords[frame_offset + d_idx];

            let angle = dihedral_value(
                [a[0] as f64, a[1] as f64, a[2] as f64],
                [b[0] as f64, b[1] as f64, b[2] as f64],
                [c[0] as f64, c[1] as f64, c[2] as f64],
                [d[0] as f64, d[1] as f64, d[2] as f64],
            )?;
            let state = classify_torsion_state(angle.to_degrees()) as usize;
            if collect_states {
                frame_states.push(state as u8);
            }
            let weight = if use_mass {
                (masses[a_idx] as f64
                    + masses[b_idx] as f64
                    + masses[c_idx] as f64
                    + masses[d_idx] as f64)
                    / 4.0
            } else {
                1.0
            };
            counts[state] += weight;
            total += weight;
        }

        let denom = if total > 0.0 { total as f32 } else { 1.0 };
        stats.push(TorsionStat {
            trans: counts[0] as f32 / denom,
            cis: counts[1] as f32 / denom,
            g_plus: counts[2] as f32 / denom,
            g_minus: counts[3] as f32 / denom,
        });
        if collect_states {
            states.push(frame_states);
        }
    }

    Ok(TorsionStatsDetailed { stats, states })
}

fn dihedral_value(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> TrajResult<f64> {
    let b0 = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let b1 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    let b2 = [d[0] - c[0], d[1] - c[1], d[2] - c[2]];
    let b1_norm = {
        let n = (b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2]).sqrt();
        if n == 0.0 {
            return Err(TrajError::Mismatch("dihedral axis has zero length".into()));
        }
        [b1[0] / n, b1[1] / n, b1[2] / n]
    };
    let dot_b0 = b0[0] * b1_norm[0] + b0[1] * b1_norm[1] + b0[2] * b1_norm[2];
    let dot_b2 = b2[0] * b1_norm[0] + b2[1] * b1_norm[1] + b2[2] * b1_norm[2];
    let v = [
        b0[0] - dot_b0 * b1_norm[0],
        b0[1] - dot_b0 * b1_norm[1],
        b0[2] - dot_b0 * b1_norm[2],
    ];
    let w = [
        b2[0] - dot_b2 * b1_norm[0],
        b2[1] - dot_b2 * b1_norm[1],
        b2[2] - dot_b2 * b1_norm[2],
    ];
    let x = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
    let m = [
        b1_norm[1] * v[2] - b1_norm[2] * v[1],
        b1_norm[2] * v[0] - b1_norm[0] * v[2],
        b1_norm[0] * v[1] - b1_norm[1] * v[0],
    ];
    let y = m[0] * w[0] + m[1] * w[1] + m[2] * w[2];
    Ok(y.atan2(x))
}
