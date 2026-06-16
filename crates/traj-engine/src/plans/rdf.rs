use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use traj_core::centers::center_of_selection;
use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, PairDistributionOutput, Plan, PlanOutput, RdfOutput};
use crate::plans::PbcMode;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuCounts, GpuSelection};

pub struct RdfPlan {
    sel_a: Selection,
    sel_b: Selection,
    io_selection: Vec<u32>,
    sel_a_local: Vec<usize>,
    sel_b_local: Vec<usize>,
    bins: usize,
    r_max: f32,
    pbc: PbcMode,
    counts: Vec<u64>,
    frames: usize,
    volume_sum: f64,
    center_a: bool,
    center_b: bool,
    byres_a: bool,
    byres_b: bool,
    bymol_a: bool,
    bymol_b: bool,
    no_intramol: bool,
    mass_weighted_centers: bool,
    density: f64,
    use_volume: bool,
    raw_output: bool,
    integral_output: bool,
    dimension: RdfDimension,
    common_count: usize,
    groups_a: Vec<RdfGroup>,
    groups_b: Vec<RdfGroup>,
    use_selected_input: bool,
    #[cfg(feature = "cuda")]
    gpu: Option<RdfGpuState>,
}

#[derive(Clone, Debug)]
struct RdfGroup {
    indices: Vec<u32>,
    residue_key: Option<(u32, i32, u32)>,
    molecule_key: Option<i32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RdfDimension {
    ThreeD,
    PlanarXY,
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
    io_selection: Vec<u32>,
    sel_a_local: Vec<usize>,
    sel_b_local: Vec<usize>,
    bins: usize,
    r_max: f32,
    pbc: PbcMode,
    counts: Vec<u64>,
    sum_counts: Vec<f64>,
    sumsq_counts: Vec<f64>,
    frames: usize,
    output_distribution: bool,
    unique_pairs: bool,
    compact_output: bool,
    use_selected_input: bool,
    #[cfg(feature = "cuda")]
    gpu: Option<PairDistGpuState>,
}

pub struct PairDistDynamicPlan {
    sel_a: Selection,
    sel_b: Selection,
    io_selection: Vec<u32>,
    sel_a_local: Vec<usize>,
    sel_b_local: Vec<usize>,
    delta: f32,
    pbc: PbcMode,
    counts: Vec<u64>,
    sum_counts: Vec<f64>,
    sumsq_counts: Vec<f64>,
    frames: usize,
    output_distribution: bool,
    unique_pairs: bool,
    compact_output: bool,
    use_selected_input: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairDistanceExtremaMode {
    Min,
    Max,
}

pub struct PairDistanceExtremaPlan {
    sel_a: Selection,
    sel_b: Selection,
    io_selection: Vec<u32>,
    sel_a_local: Vec<usize>,
    sel_b_local: Vec<usize>,
    mode: PairDistanceExtremaMode,
    pbc: PbcMode,
    unique_pairs: bool,
    use_selected_input: bool,
    results: Vec<f32>,
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
        let (io_selection, sel_a_local, sel_b_local) =
            build_pairdist_io_layout(&sel_a.indices, &sel_b.indices);
        Self {
            sel_a,
            sel_b,
            io_selection,
            sel_a_local,
            sel_b_local,
            bins,
            r_max,
            pbc,
            counts: vec![0; bins],
            frames: 0,
            volume_sum: 0.0,
            center_a: false,
            center_b: false,
            byres_a: false,
            byres_b: false,
            bymol_a: false,
            bymol_b: false,
            no_intramol: false,
            mass_weighted_centers: true,
            density: 0.033456,
            use_volume: false,
            raw_output: false,
            integral_output: false,
            dimension: RdfDimension::ThreeD,
            common_count: 0,
            groups_a: Vec::new(),
            groups_b: Vec::new(),
            use_selected_input: true,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_center_modes(
        mut self,
        center_a: bool,
        center_b: bool,
        mass_weighted: bool,
    ) -> Self {
        self.center_a = center_a;
        self.center_b = center_b;
        self.mass_weighted_centers = mass_weighted;
        self
    }

    pub fn with_radial_options(
        mut self,
        center_a: bool,
        center_b: bool,
        byres_a: bool,
        byres_b: bool,
        bymol_a: bool,
        bymol_b: bool,
        no_intramol: bool,
        mass_weighted: bool,
    ) -> Self {
        self.center_a = center_a;
        self.center_b = center_b;
        self.byres_a = byres_a;
        self.byres_b = byres_b;
        self.bymol_a = bymol_a;
        self.bymol_b = bymol_b;
        self.no_intramol = no_intramol;
        self.mass_weighted_centers = mass_weighted;
        self
    }

    pub fn with_output_options(
        mut self,
        density: f64,
        use_volume: bool,
        raw_output: bool,
        integral_output: bool,
    ) -> Self {
        self.density = density;
        self.use_volume = use_volume;
        self.raw_output = raw_output;
        self.integral_output = integral_output;
        self
    }

    pub fn with_dimension(mut self, dimension: RdfDimension) -> Self {
        self.dimension = dimension;
        self
    }

    fn grouped_mode(&self) -> bool {
        self.center_a
            || self.center_b
            || self.byres_a
            || self.byres_b
            || self.bymol_a
            || self.bymol_b
            || self.no_intramol
    }

    fn effective_count_a(&self) -> f64 {
        if self.grouped_mode() {
            self.groups_a.len() as f64
        } else {
            self.sel_a.indices.len() as f64
        }
    }

    fn effective_count_b(&self) -> f64 {
        if self.grouped_mode() {
            self.groups_b.len() as f64
        } else {
            self.sel_b.indices.len() as f64
        }
    }
}

fn residue_key(system: &System, atom_idx: usize) -> Option<(u32, i32, u32)> {
    if atom_idx >= system.atoms.chain_id.len()
        || atom_idx >= system.atoms.resid.len()
        || atom_idx >= system.atoms.resname_id.len()
    {
        return None;
    }
    Some((
        system.atoms.chain_id[atom_idx],
        system.atoms.resid[atom_idx],
        system.atoms.resname_id[atom_idx],
    ))
}

fn atom_groups(selection: &Selection, system: &System) -> Vec<RdfGroup> {
    selection
        .indices
        .iter()
        .map(|&idx| RdfGroup {
            indices: vec![idx],
            residue_key: residue_key(system, idx as usize),
            molecule_key: system.molecule_id_for_atom(idx as usize),
        })
        .collect()
}

fn residue_groups(selection: &Selection, system: &System) -> Vec<RdfGroup> {
    let mut grouped = BTreeMap::<(u32, i32, u32), Vec<u32>>::new();
    for &idx in selection.indices.iter() {
        if let Some(key) = residue_key(system, idx as usize) {
            grouped.entry(key).or_default().push(idx);
        }
    }
    grouped
        .into_iter()
        .map(|(key, indices)| RdfGroup {
            indices,
            residue_key: Some(key),
            molecule_key: None,
        })
        .collect()
}

fn molecule_groups(selection: &Selection, system: &System) -> Vec<RdfGroup> {
    let mut grouped = BTreeMap::<i32, Vec<u32>>::new();
    for &idx in selection.indices.iter() {
        if let Some(key) = system.molecule_id_for_atom(idx as usize) {
            grouped.entry(key).or_default().push(idx);
        }
    }
    grouped
        .into_iter()
        .map(|(key, indices)| RdfGroup {
            indices,
            residue_key: None,
            molecule_key: Some(key),
        })
        .collect()
}

fn rdf_groups(
    selection: &Selection,
    system: &System,
    center: bool,
    byres: bool,
    bymol: bool,
) -> Vec<RdfGroup> {
    if center {
        return vec![RdfGroup {
            indices: selection.indices.as_ref().clone(),
            residue_key: None,
            molecule_key: None,
        }];
    }
    if bymol {
        let groups = molecule_groups(selection, system);
        if !groups.is_empty() {
            return groups;
        }
    }
    if byres {
        let groups = residue_groups(selection, system);
        if !groups.is_empty() {
            return groups;
        }
    }
    atom_groups(selection, system)
}

fn group_position(
    chunk: &FrameChunk,
    frame: usize,
    group: &RdfGroup,
    masses: &[f32],
    mass_weighted: bool,
) -> [f32; 4] {
    if group.indices.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }
    if group.indices.len() == 1 {
        return chunk.coords[frame * chunk.n_atoms + group.indices[0] as usize];
    }
    let center = center_of_selection(chunk, frame, &group.indices, masses, mass_weighted);
    [center[0] as f32, center[1] as f32, center[2] as f32, 0.0]
}

fn same_intramolecular_group(left: &RdfGroup, right: &RdfGroup) -> bool {
    if let (Some(a), Some(b)) = (left.molecule_key, right.molecule_key) {
        return a == b;
    }
    matches!((left.residue_key, right.residue_key), (Some(a), Some(b)) if a == b)
}

fn common_atom_count(sel_a: &Selection, sel_b: &Selection) -> usize {
    let left: BTreeSet<u32> = sel_a.indices.iter().copied().collect();
    sel_b
        .indices
        .iter()
        .filter(|idx| left.contains(idx))
        .count()
}

fn common_group_count(groups_a: &[RdfGroup], groups_b: &[RdfGroup], no_intramol: bool) -> usize {
    let mut count = 0usize;
    for group_a in groups_a {
        for group_b in groups_b {
            if no_intramol {
                if same_intramolecular_group(group_a, group_b) {
                    count += 1;
                }
            } else if group_a.indices == group_b.indices {
                count += 1;
            }
        }
    }
    count
}

fn ensure_pairdist_stats_len(
    counts: &mut Vec<u64>,
    sum_counts: &mut Vec<f64>,
    sumsq_counts: &mut Vec<f64>,
    len: usize,
) {
    if counts.len() < len {
        counts.resize(len, 0);
    }
    if sum_counts.len() < len {
        sum_counts.resize(len, 0.0);
    }
    if sumsq_counts.len() < len {
        sumsq_counts.resize(len, 0.0);
    }
}

fn accumulate_pairdist_frame(
    counts: &mut Vec<u64>,
    sum_counts: &mut Vec<f64>,
    sumsq_counts: &mut Vec<f64>,
    frame_counts: &[u64],
) {
    ensure_pairdist_stats_len(counts, sum_counts, sumsq_counts, frame_counts.len());
    for (idx, &value) in frame_counts.iter().enumerate() {
        counts[idx] += value;
        let value = value as f64;
        sum_counts[idx] += value;
        sumsq_counts[idx] += value * value;
    }
}

fn pairdist_output_from_stats(
    bin_width: f32,
    frames: usize,
    counts: Vec<u64>,
    sum_counts: Vec<f64>,
    sumsq_counts: Vec<f64>,
    compact_output: bool,
) -> PairDistributionOutput {
    let mut centers = Vec::new();
    let mut probability = Vec::new();
    let mut std = Vec::new();
    let mut output_counts = Vec::new();
    let frames_f = frames.max(1) as f64;
    for idx in 0..sum_counts.len() {
        if compact_output && sum_counts[idx] <= 0.0 {
            continue;
        }
        let mean = sum_counts[idx] / frames_f;
        let variance = if frames > 1 {
            let raw = sumsq_counts[idx] - (sum_counts[idx] * sum_counts[idx] / frames_f);
            (raw / (frames_f - 1.0)).max(0.0)
        } else {
            0.0
        };
        centers.push(idx as f32 * bin_width + 0.5 * bin_width);
        probability.push((mean / bin_width as f64) as f32);
        std.push(variance.sqrt() as f32);
        output_counts.push(counts.get(idx).copied().unwrap_or(0));
    }
    PairDistributionOutput {
        centers,
        probability,
        std,
        counts: output_counts,
        frames,
    }
}

fn rdf_distance(dimension: RdfDimension, dx: f32, dy: f32, dz: f32) -> f32 {
    match dimension {
        RdfDimension::ThreeD => (dx * dx + dy * dy + dz * dz).sqrt(),
        RdfDimension::PlanarXY => (dx * dx + dy * dy).sqrt(),
    }
}

fn rdf_box_measure(dimension: RdfDimension, lx: f32, ly: f32, lz: f32) -> f64 {
    match dimension {
        RdfDimension::ThreeD => (lx as f64) * (ly as f64) * (lz as f64),
        RdfDimension::PlanarXY => (lx as f64) * (ly as f64),
    }
}

fn rdf_shell_measure(dimension: RdfDimension, r_inner: f32, r_outer: f32) -> f64 {
    match dimension {
        RdfDimension::ThreeD => {
            ((4.0 / 3.0) * std::f32::consts::PI * (r_outer.powi(3) - r_inner.powi(3))) as f64
        }
        RdfDimension::PlanarXY => {
            (std::f32::consts::PI * (r_outer.powi(2) - r_inner.powi(2))) as f64
        }
    }
}

fn build_pairdist_io_layout(sel_a: &[u32], sel_b: &[u32]) -> (Vec<u32>, Vec<usize>, Vec<usize>) {
    let mut io = Vec::<u32>::with_capacity(sel_a.len() + sel_b.len());
    let mut global_to_local = BTreeMap::<u32, usize>::new();
    for &idx in sel_a.iter().chain(sel_b.iter()) {
        if !global_to_local.contains_key(&idx) {
            let local = io.len();
            io.push(idx);
            global_to_local.insert(idx, local);
        }
    }

    let sel_a_local = sel_a
        .iter()
        .map(|idx| {
            *global_to_local
                .get(idx)
                .expect("pairdist io layout missing sel_a index")
        })
        .collect();
    let sel_b_local = sel_b
        .iter()
        .map(|idx| {
            *global_to_local
                .get(idx)
                .expect("pairdist io layout missing sel_b index")
        })
        .collect();
    (io, sel_a_local, sel_b_local)
}

fn validate_pairdist_selected_chunk(
    name: &str,
    source_selection: &[u32],
    expected: &[u32],
) -> TrajResult<()> {
    if source_selection != expected {
        return Err(TrajError::Mismatch(format!(
            "{name} selected chunk does not match expected IO selection"
        )));
    }
    Ok(())
}

impl PairDistPlan {
    pub fn new(sel_a: Selection, sel_b: Selection, bins: usize, r_max: f32, pbc: PbcMode) -> Self {
        let (io_selection, sel_a_local, sel_b_local) =
            build_pairdist_io_layout(&sel_a.indices, &sel_b.indices);
        Self {
            sel_a,
            sel_b,
            io_selection,
            sel_a_local,
            sel_b_local,
            bins,
            r_max,
            pbc,
            counts: vec![0; bins],
            sum_counts: vec![0.0; bins],
            sumsq_counts: vec![0.0; bins],
            frames: 0,
            output_distribution: false,
            unique_pairs: false,
            compact_output: false,
            use_selected_input: true,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_output_options(
        mut self,
        output_distribution: bool,
        unique_pairs: bool,
        compact_output: bool,
    ) -> Self {
        self.output_distribution = output_distribution;
        self.unique_pairs = unique_pairs;
        self.compact_output = compact_output;
        self
    }
}

impl PairDistanceExtremaPlan {
    pub fn new(
        sel_a: Selection,
        sel_b: Selection,
        mode: PairDistanceExtremaMode,
        pbc: PbcMode,
    ) -> Self {
        let (io_selection, sel_a_local, sel_b_local) =
            build_pairdist_io_layout(&sel_a.indices, &sel_b.indices);
        Self {
            sel_a,
            sel_b,
            io_selection,
            sel_a_local,
            sel_b_local,
            mode,
            pbc,
            unique_pairs: false,
            use_selected_input: true,
            results: Vec::new(),
        }
    }

    pub fn with_unique_pairs(mut self, unique_pairs: bool) -> Self {
        self.unique_pairs = unique_pairs;
        self
    }
}

impl PairDistDynamicPlan {
    pub fn new(sel_a: Selection, sel_b: Selection, delta: f32, pbc: PbcMode) -> Self {
        let (io_selection, sel_a_local, sel_b_local) =
            build_pairdist_io_layout(&sel_a.indices, &sel_b.indices);
        Self {
            sel_a,
            sel_b,
            io_selection,
            sel_a_local,
            sel_b_local,
            delta,
            pbc,
            counts: Vec::new(),
            sum_counts: Vec::new(),
            sumsq_counts: Vec::new(),
            frames: 0,
            output_distribution: false,
            unique_pairs: false,
            compact_output: false,
            use_selected_input: true,
        }
    }

    pub fn with_output_options(
        mut self,
        output_distribution: bool,
        unique_pairs: bool,
        compact_output: bool,
    ) -> Self {
        self.output_distribution = output_distribution;
        self.unique_pairs = unique_pairs;
        self.compact_output = compact_output;
        self
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
        self.common_count = 0;
        if !self.density.is_finite() || self.density <= 0.0 {
            return Err(TrajError::Parse(
                "rdf density must be finite and positive".into(),
            ));
        }
        if self.grouped_mode() {
            self.groups_a = rdf_groups(
                &self.sel_a,
                _system,
                self.center_a,
                self.byres_a,
                self.bymol_a,
            );
            self.groups_b = rdf_groups(
                &self.sel_b,
                _system,
                self.center_b,
                self.byres_b,
                self.bymol_b,
            );
            self.common_count =
                common_group_count(&self.groups_a, &self.groups_b, self.no_intramol);
        } else {
            self.groups_a.clear();
            self.groups_b.clear();
            self.common_count = common_atom_count(&self.sel_a, &self.sel_b);
        }
        self.use_selected_input = matches!(device, Device::Cpu) && !self.grouped_mode();
        let _ = device;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = device {
                self.use_selected_input = false;
                if self.grouped_mode() || matches!(self.dimension, RdfDimension::PlanarXY) {
                    return Ok(());
                }
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

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(self.io_selection.as_slice())
        } else {
            None
        }
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        if self.grouped_mode() {
            None
        } else {
            Some(self.io_selection.as_slice())
        }
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
        if !self.grouped_mode() && matches!(self.dimension, RdfDimension::ThreeD) {
            if let (Device::Cuda(ctx), Some(gpu)) = (device, &mut self.gpu) {
                let mut box_l = Vec::new();
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    box_l.reserve(chunk.n_frames * 3);
                }
                for frame in 0..chunk.n_frames {
                    if matches!(self.pbc, PbcMode::Orthorhombic) || self.use_volume {
                        match chunk.box_[frame] {
                            Box3::Orthorhombic { lx, ly, lz } => {
                                if matches!(self.pbc, PbcMode::Orthorhombic) {
                                    box_l.push(lx);
                                    box_l.push(ly);
                                    box_l.push(lz);
                                }
                                self.volume_sum += (lx as f64) * (ly as f64) * (lz as f64);
                            }
                            _ => {
                                return Err(TrajError::Mismatch(
                                    "RDF requires orthorhombic box for PBC or volume normalization"
                                        .into(),
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
        }
        for frame in 0..chunk.n_frames {
            let box_ = chunk.box_[frame];
            let (lx, ly, lz) = match (self.pbc, self.use_volume) {
                (PbcMode::None, false) => (0.0, 0.0, 0.0),
                (PbcMode::None, true) | (PbcMode::Orthorhombic, _) => match box_ {
                    Box3::Orthorhombic { lx, ly, lz } => (lx, ly, lz),
                    _ => {
                        return Err(TrajError::Mismatch(
                            "RDF requires orthorhombic box for PBC or volume normalization".into(),
                        ))
                    }
                },
            };
            if matches!(self.pbc, PbcMode::Orthorhombic) || self.use_volume {
                self.volume_sum += rdf_box_measure(self.dimension, lx, ly, lz);
            }
            if self.grouped_mode() {
                for group_a in self.groups_a.iter() {
                    let pos_a = group_position(
                        chunk,
                        frame,
                        group_a,
                        &_system.atoms.mass,
                        self.mass_weighted_centers,
                    );
                    for group_b in self.groups_b.iter() {
                        if group_a.indices == group_b.indices {
                            continue;
                        }
                        if self.no_intramol && same_intramolecular_group(group_a, group_b) {
                            continue;
                        }
                        let pos_b = group_position(
                            chunk,
                            frame,
                            group_b,
                            &_system.atoms.mass,
                            self.mass_weighted_centers,
                        );
                        let mut dx = pos_b[0] - pos_a[0];
                        let mut dy = pos_b[1] - pos_a[1];
                        let mut dz = pos_b[2] - pos_a[2];
                        if matches!(self.pbc, PbcMode::Orthorhombic) {
                            dx -= (dx / lx).round() * lx;
                            dy -= (dy / ly).round() * ly;
                            dz -= (dz / lz).round() * lz;
                        }
                        let r = rdf_distance(self.dimension, dx, dy, dz);
                        if r < self.r_max {
                            let bin = (r / bin_width) as usize;
                            if bin < self.bins {
                                self.counts[bin] += 1;
                            }
                        }
                    }
                }
                self.frames += 1;
                continue;
            }
            for &a in self.sel_a.indices.iter() {
                let a_idx = a as usize;
                let pos_a = chunk.coords[frame * n_atoms + a_idx];
                for &b in self.sel_b.indices.iter() {
                    let b_idx = b as usize;
                    if a_idx == b_idx {
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
                    let r = rdf_distance(self.dimension, dx, dy, dz);
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

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.grouped_mode() {
            return Err(TrajError::Mismatch(
                "rdf grouped mode cannot process selected atom chunks".into(),
            ));
        }
        validate_pairdist_selected_chunk("rdf", source_selection, &self.io_selection)?;
        let n_atoms = chunk.n_atoms;
        let bin_width = self.r_max / self.bins as f32;
        for frame in 0..chunk.n_frames {
            let box_ = chunk.box_[frame];
            let (lx, ly, lz) = match (self.pbc, self.use_volume) {
                (PbcMode::None, false) => (0.0, 0.0, 0.0),
                (PbcMode::None, true) | (PbcMode::Orthorhombic, _) => match box_ {
                    Box3::Orthorhombic { lx, ly, lz } => (lx, ly, lz),
                    _ => {
                        return Err(TrajError::Mismatch(
                            "RDF requires orthorhombic box for PBC or volume normalization".into(),
                        ))
                    }
                },
            };
            if matches!(self.pbc, PbcMode::Orthorhombic) || self.use_volume {
                self.volume_sum += rdf_box_measure(self.dimension, lx, ly, lz);
            }
            let frame_base = frame * n_atoms;
            for &a_idx in self.sel_a_local.iter() {
                let pos_a = chunk.coords[frame_base + a_idx];
                for &b_idx in self.sel_b_local.iter() {
                    if a_idx == b_idx {
                        continue;
                    }
                    let pos_b = chunk.coords[frame_base + b_idx];
                    let mut dx = pos_b[0] - pos_a[0];
                    let mut dy = pos_b[1] - pos_a[1];
                    let mut dz = pos_b[2] - pos_a[2];
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        dx -= (dx / lx).round() * lx;
                        dy -= (dy / ly).round() * ly;
                        dz -= (dz / lz).round() * lz;
                    }
                    let r = rdf_distance(self.dimension, dx, dy, dz);
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
        let mut integral = if self.integral_output {
            Vec::with_capacity(self.bins)
        } else {
            Vec::new()
        };
        let n_a = self.effective_count_a();
        let n_b = self.effective_count_b();
        let frames = self.frames.max(1) as f64;
        let common = self.common_count as f64;
        let total_pairs = (n_a * n_b - common).max(0.0);
        let density = if self.use_volume {
            if self.volume_sum <= 0.0 {
                return Err(TrajError::Mismatch(
                    "rdf box normalization requires frame box metadata".into(),
                ));
            }
            total_pairs / (self.volume_sum / frames)
        } else if n_a > 0.0 {
            self.density * total_pairs / n_a
        } else {
            0.0
        };
        let mut integral_sum = 0.0f64;
        for i in 0..self.bins {
            let r_inner = i as f32 * bin_width;
            let r_outer = (i + 1) as f32 * bin_width;
            let shell_measure = rdf_shell_measure(self.dimension, r_inner, r_outer);
            let expected = shell_measure * density;
            let norm = if expected > 0.0 {
                frames * expected
            } else {
                0.0
            };
            r.push(r_inner + 0.5 * bin_width);
            let count = self.counts[i] as f64;
            let normalized = if norm > 0.0 { count / norm } else { 0.0 };
            if self.integral_output {
                if n_b > 0.0 {
                    integral_sum += normalized * expected / n_b;
                }
                integral.push(integral_sum as f32);
            }
            let value = if self.raw_output { count } else { normalized };
            g_r.push(value as f32);
        }
        Ok(PlanOutput::Rdf(RdfOutput {
            r,
            g_r,
            counts: std::mem::take(&mut self.counts),
            integral,
        }))
    }
}

impl Plan for PairDistPlan {
    fn name(&self) -> &'static str {
        "pairdist"
    }

    fn init(&mut self, _system: &System, device: &Device) -> TrajResult<()> {
        self.counts.fill(0);
        self.sum_counts.fill(0.0);
        self.sumsq_counts.fill(0.0);
        self.frames = 0;
        self.use_selected_input = matches!(device, Device::Cpu);
        let _ = device;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if !self.output_distribution {
                if let Device::Cuda(ctx) = device {
                    self.use_selected_input = false;
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
        }
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(self.io_selection.as_slice())
        } else {
            None
        }
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        Some(self.io_selection.as_slice())
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
            let mut frame_counts = if self.output_distribution {
                vec![0_u64; self.bins]
            } else {
                Vec::new()
            };
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
            for (a_pos, &a) in self.sel_a.indices.iter().enumerate() {
                let a_idx = a as usize;
                let pos_a = chunk.coords[frame * n_atoms + a_idx];
                for (b_pos, &b) in self.sel_b.indices.iter().enumerate() {
                    let b_idx = b as usize;
                    if self.unique_pairs && b_pos <= a_pos {
                        continue;
                    }
                    if !self.unique_pairs
                        && a_idx == b_idx
                        && Arc::ptr_eq(&self.sel_a.indices, &self.sel_b.indices)
                    {
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
                            if self.output_distribution {
                                frame_counts[bin] += 1;
                            } else {
                                self.counts[bin] += 1;
                            }
                        }
                    }
                }
            }
            if self.output_distribution {
                accumulate_pairdist_frame(
                    &mut self.counts,
                    &mut self.sum_counts,
                    &mut self.sumsq_counts,
                    &frame_counts,
                );
            }
            self.frames += 1;
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
        validate_pairdist_selected_chunk("pairdist", source_selection, &self.io_selection)?;
        let n_atoms = chunk.n_atoms;
        let bin_width = self.r_max / self.bins as f32;
        let same_sel = Arc::ptr_eq(&self.sel_a.indices, &self.sel_b.indices);
        for frame in 0..chunk.n_frames {
            let mut frame_counts = if self.output_distribution {
                vec![0_u64; self.bins]
            } else {
                Vec::new()
            };
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
            let frame_base = frame * n_atoms;
            for (a_pos, &a_idx) in self.sel_a_local.iter().enumerate() {
                let pos_a = chunk.coords[frame_base + a_idx];
                for (b_pos, &b_idx) in self.sel_b_local.iter().enumerate() {
                    if self.unique_pairs && b_pos <= a_pos {
                        continue;
                    }
                    if !self.unique_pairs && same_sel && a_idx == b_idx {
                        continue;
                    }
                    let pos_b = chunk.coords[frame_base + b_idx];
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
                            if self.output_distribution {
                                frame_counts[bin] += 1;
                            } else {
                                self.counts[bin] += 1;
                            }
                        }
                    }
                }
            }
            if self.output_distribution {
                accumulate_pairdist_frame(
                    &mut self.counts,
                    &mut self.sum_counts,
                    &mut self.sumsq_counts,
                    &frame_counts,
                );
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
        if self.output_distribution {
            let counts = std::mem::take(&mut self.counts);
            let sum_counts = std::mem::take(&mut self.sum_counts);
            let sumsq_counts = std::mem::take(&mut self.sumsq_counts);
            return Ok(PlanOutput::PairDistribution(pairdist_output_from_stats(
                bin_width,
                self.frames,
                counts,
                sum_counts,
                sumsq_counts,
                self.compact_output,
            )));
        }
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

impl Plan for PairDistanceExtremaPlan {
    fn name(&self) -> &'static str {
        "pairdist_extrema"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.use_selected_input = matches!(_device, Device::Cpu);
        let n_atoms = system.n_atoms() as u32;
        for &idx in self.sel_a.indices.iter().chain(self.sel_b.indices.iter()) {
            if idx >= n_atoms {
                return Err(TrajError::Mismatch(
                    "pairdist atom index out of range".into(),
                ));
            }
        }
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(self.io_selection.as_slice())
        } else {
            None
        }
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        Some(self.io_selection.as_slice())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
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
            let mut best = match self.mode {
                PairDistanceExtremaMode::Min => f32::INFINITY,
                PairDistanceExtremaMode::Max => f32::NEG_INFINITY,
            };
            let mut seen = false;
            let frame_base = frame * n_atoms;
            for (a_pos, &a) in self.sel_a.indices.iter().enumerate() {
                let a_idx = a as usize;
                let pos_a = chunk.coords[frame_base + a_idx];
                for (b_pos, &b) in self.sel_b.indices.iter().enumerate() {
                    let b_idx = b as usize;
                    if self.unique_pairs && b_pos <= a_pos {
                        continue;
                    }
                    if !self.unique_pairs
                        && a_idx == b_idx
                        && Arc::ptr_eq(&self.sel_a.indices, &self.sel_b.indices)
                    {
                        continue;
                    }
                    let pos_b = chunk.coords[frame_base + b_idx];
                    let mut dx = pos_b[0] - pos_a[0];
                    let mut dy = pos_b[1] - pos_a[1];
                    let mut dz = pos_b[2] - pos_a[2];
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        dx -= (dx / lx).round() * lx;
                        dy -= (dy / ly).round() * ly;
                        dz -= (dz / lz).round() * lz;
                    }
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist.is_finite() {
                        best = match self.mode {
                            PairDistanceExtremaMode::Min => best.min(dist),
                            PairDistanceExtremaMode::Max => best.max(dist),
                        };
                        seen = true;
                    }
                }
            }
            self.results.push(if seen { best } else { f32::NAN });
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
        validate_pairdist_selected_chunk("pairdist_extrema", source_selection, &self.io_selection)?;
        let n_atoms = chunk.n_atoms;
        let same_sel = Arc::ptr_eq(&self.sel_a.indices, &self.sel_b.indices);
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
            let mut best = match self.mode {
                PairDistanceExtremaMode::Min => f32::INFINITY,
                PairDistanceExtremaMode::Max => f32::NEG_INFINITY,
            };
            let mut seen = false;
            let frame_base = frame * n_atoms;
            for (a_pos, &a_idx) in self.sel_a_local.iter().enumerate() {
                let pos_a = chunk.coords[frame_base + a_idx];
                for (b_pos, &b_idx) in self.sel_b_local.iter().enumerate() {
                    if self.unique_pairs && b_pos <= a_pos {
                        continue;
                    }
                    if !self.unique_pairs && same_sel && a_idx == b_idx {
                        continue;
                    }
                    let pos_b = chunk.coords[frame_base + b_idx];
                    let mut dx = pos_b[0] - pos_a[0];
                    let mut dy = pos_b[1] - pos_a[1];
                    let mut dz = pos_b[2] - pos_a[2];
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        dx -= (dx / lx).round() * lx;
                        dy -= (dy / ly).round() * ly;
                        dz -= (dz / lz).round() * lz;
                    }
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    if dist.is_finite() {
                        best = match self.mode {
                            PairDistanceExtremaMode::Min => best.min(dist),
                            PairDistanceExtremaMode::Max => best.max(dist),
                        };
                        seen = true;
                    }
                }
            }
            self.results.push(if seen { best } else { f32::NAN });
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

impl Plan for PairDistDynamicPlan {
    fn name(&self) -> &'static str {
        "pairdist_dynamic"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        if !self.delta.is_finite() || self.delta <= 0.0 {
            return Err(TrajError::Parse(
                "pairdist delta must be finite and positive".into(),
            ));
        }
        self.counts.clear();
        self.sum_counts.clear();
        self.sumsq_counts.clear();
        self.frames = 0;
        self.use_selected_input = matches!(_device, Device::Cpu);
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(self.io_selection.as_slice())
        } else {
            None
        }
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        Some(self.io_selection.as_slice())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let mut frame_counts = if self.output_distribution {
                vec![0_u64; self.counts.len()]
            } else {
                Vec::new()
            };
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
            for (a_pos, &a) in self.sel_a.indices.iter().enumerate() {
                let a_idx = a as usize;
                let pos_a = chunk.coords[frame * n_atoms + a_idx];
                for (b_pos, &b) in self.sel_b.indices.iter().enumerate() {
                    let b_idx = b as usize;
                    if self.unique_pairs && b_pos <= a_pos {
                        continue;
                    }
                    if !self.unique_pairs
                        && a_idx == b_idx
                        && Arc::ptr_eq(&self.sel_a.indices, &self.sel_b.indices)
                    {
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
                    if r.is_finite() {
                        let bin = (r / self.delta) as usize;
                        if self.output_distribution {
                            if bin >= frame_counts.len() {
                                frame_counts.resize(bin + 1, 0);
                            }
                            frame_counts[bin] += 1;
                        } else {
                            if bin >= self.counts.len() {
                                self.counts.resize(bin + 1, 0);
                            }
                            self.counts[bin] += 1;
                        }
                    }
                }
            }
            if self.output_distribution {
                accumulate_pairdist_frame(
                    &mut self.counts,
                    &mut self.sum_counts,
                    &mut self.sumsq_counts,
                    &frame_counts,
                );
            }
            self.frames += 1;
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
        validate_pairdist_selected_chunk("pairdist_dynamic", source_selection, &self.io_selection)?;
        let n_atoms = chunk.n_atoms;
        let same_sel = Arc::ptr_eq(&self.sel_a.indices, &self.sel_b.indices);
        for frame in 0..chunk.n_frames {
            let mut frame_counts = if self.output_distribution {
                vec![0_u64; self.counts.len()]
            } else {
                Vec::new()
            };
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
            let frame_base = frame * n_atoms;
            for (a_pos, &a_idx) in self.sel_a_local.iter().enumerate() {
                let pos_a = chunk.coords[frame_base + a_idx];
                for (b_pos, &b_idx) in self.sel_b_local.iter().enumerate() {
                    if self.unique_pairs && b_pos <= a_pos {
                        continue;
                    }
                    if !self.unique_pairs && same_sel && a_idx == b_idx {
                        continue;
                    }
                    let pos_b = chunk.coords[frame_base + b_idx];
                    let mut dx = pos_b[0] - pos_a[0];
                    let mut dy = pos_b[1] - pos_a[1];
                    let mut dz = pos_b[2] - pos_a[2];
                    if matches!(self.pbc, PbcMode::Orthorhombic) {
                        dx -= (dx / lx).round() * lx;
                        dy -= (dy / ly).round() * ly;
                        dz -= (dz / lz).round() * lz;
                    }
                    let r = (dx * dx + dy * dy + dz * dz).sqrt();
                    if r.is_finite() {
                        let bin = (r / self.delta) as usize;
                        if self.output_distribution {
                            if bin >= frame_counts.len() {
                                frame_counts.resize(bin + 1, 0);
                            }
                            frame_counts[bin] += 1;
                        } else {
                            if bin >= self.counts.len() {
                                self.counts.resize(bin + 1, 0);
                            }
                            self.counts[bin] += 1;
                        }
                    }
                }
            }
            if self.output_distribution {
                accumulate_pairdist_frame(
                    &mut self.counts,
                    &mut self.sum_counts,
                    &mut self.sumsq_counts,
                    &frame_counts,
                );
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.output_distribution {
            let counts = std::mem::take(&mut self.counts);
            let sum_counts = std::mem::take(&mut self.sum_counts);
            let sumsq_counts = std::mem::take(&mut self.sumsq_counts);
            return Ok(PlanOutput::PairDistribution(pairdist_output_from_stats(
                self.delta,
                self.frames,
                counts,
                sum_counts,
                sumsq_counts,
                self.compact_output,
            )));
        }
        let mut centers = Vec::with_capacity(self.counts.len());
        for i in 0..self.counts.len() {
            centers.push(i as f32 * self.delta + 0.5 * self.delta);
        }
        Ok(PlanOutput::Histogram {
            centers,
            counts: std::mem::take(&mut self.counts),
        })
    }
}
