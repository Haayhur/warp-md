use std::collections::{BTreeMap, HashSet};

use geo::{Area, LineString, Polygon};
use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, LipidFlipFlopOutput, LipidMatrixOutput, Plan, PlanOutput};

#[derive(Clone)]
struct ResidueGroup {
    resid: i32,
    atoms: Vec<usize>,
}

fn groups_for_selection(system: &System, selection: &Selection) -> TrajResult<Vec<ResidueGroup>> {
    if selection.indices.is_empty() {
        return Err(TrajError::Mismatch(
            "lipid analysis selection is empty".into(),
        ));
    }
    let mut map: BTreeMap<(u32, i32), Vec<usize>> = BTreeMap::new();
    for &idx in selection.indices.iter() {
        let atom = idx as usize;
        map.entry((system.atoms.chain_id[atom], system.atoms.resid[atom]))
            .or_default()
            .push(atom);
    }
    let mut groups = Vec::with_capacity(map.len());
    for ((_chain, resid), atoms) in map {
        groups.push(ResidueGroup { resid, atoms });
    }
    if groups.is_empty() {
        return Err(TrajError::Mismatch(
            "lipid analysis produced no residues".into(),
        ));
    }
    Ok(groups)
}

fn box_lengths(chunk: &FrameChunk, frame: usize) -> TrajResult<[f64; 3]> {
    match chunk.box_.get(frame).copied().unwrap_or(Box3::None) {
        Box3::Orthorhombic { lx, ly, lz } => Ok([lx as f64, ly as f64, lz as f64]),
        Box3::None => Err(TrajError::Mismatch(
            "lipid analysis requires orthorhombic box metadata".into(),
        )),
        Box3::Triclinic { .. } => Err(TrajError::Mismatch(
            "lipid analysis currently requires orthorhombic boxes".into(),
        )),
    }
}

fn wrap_value(mut v: f64, len: f64) -> f64 {
    if len <= 0.0 || !len.is_finite() {
        return v;
    }
    v %= len;
    if v < 0.0 {
        v += len;
    }
    v
}

fn atom_pos(chunk: &FrameChunk, frame: usize, atom: usize, scale: f64) -> [f64; 3] {
    let p = chunk.coords[frame * chunk.n_atoms + atom];
    [
        p[0] as f64 * scale,
        p[1] as f64 * scale,
        p[2] as f64 * scale,
    ]
}

fn group_mean(chunk: &FrameChunk, frame: usize, atoms: &[usize], scale: f64) -> [f64; 3] {
    let inv = 1.0 / atoms.len() as f64;
    let mut out = [0.0; 3];
    for &atom in atoms {
        let p = atom_pos(chunk, frame, atom, scale);
        out[0] += p[0];
        out[1] += p[1];
        out[2] += p[2];
    }
    [out[0] * inv, out[1] * inv, out[2] * inv]
}

fn resolve_bin(value: f64, length: f64, bins: usize) -> Option<usize> {
    if bins == 0 || length <= 0.0 || !value.is_finite() {
        return None;
    }
    let wrapped = wrap_value(value, length);
    let idx = ((wrapped / length) * bins as f64).floor() as usize;
    Some(idx.min(bins - 1))
}

fn midpoint_grid(
    chunk: &FrameChunk,
    frame: usize,
    groups: &[ResidueGroup],
    bins: usize,
    scale: f64,
    lengths: [f64; 3],
) -> Vec<f64> {
    let bins = bins.max(1);
    let mut sum = vec![0.0; bins * bins];
    let mut count = vec![0usize; bins * bins];
    for group in groups {
        for &atom in &group.atoms {
            let p = atom_pos(chunk, frame, atom, scale);
            if let (Some(ix), Some(iy)) = (
                resolve_bin(p[0], lengths[0] * scale, bins),
                resolve_bin(p[1], lengths[1] * scale, bins),
            ) {
                let idx = ix * bins + iy;
                sum[idx] += p[2];
                count[idx] += 1;
            }
        }
    }
    let global = if count.iter().sum::<usize>() > 0 {
        sum.iter().sum::<f64>() / count.iter().sum::<usize>() as f64
    } else {
        0.0
    };
    for idx in 0..sum.len() {
        sum[idx] = if count[idx] == 0 {
            global
        } else {
            sum[idx] / count[idx] as f64
        };
    }
    sum
}

fn matrix_output(
    kind: &str,
    values_frame_major: Vec<f32>,
    rows: usize,
    cols: usize,
    groups: &[ResidueGroup],
) -> PlanOutput {
    let mut values = vec![f32::NAN; rows * cols];
    if values_frame_major.len() == rows * cols {
        for frame in 0..cols {
            for row in 0..rows {
                values[row * cols + frame] = values_frame_major[frame * rows + row];
            }
        }
    }
    PlanOutput::LipidMatrix(LipidMatrixOutput {
        values,
        rows,
        cols,
        residue_ids: groups.iter().map(|g| g.resid).collect(),
        frames: (0..cols).collect(),
        kind: kind.to_string(),
    })
}

fn explicit_matrix_output(
    kind: &str,
    values: Vec<f32>,
    rows: usize,
    cols: usize,
    residue_ids: Vec<i32>,
) -> PlanOutput {
    PlanOutput::LipidMatrix(LipidMatrixOutput {
        values,
        rows,
        cols,
        residue_ids,
        frames: (0..cols).collect(),
        kind: kind.to_string(),
    })
}

fn min_image(mut delta: f64, length: f64) -> f64 {
    if length > 0.0 && length.is_finite() {
        delta -= length * (delta / length).round();
    }
    delta
}

fn leaflet_value(
    leaflets: &[i8],
    rows: usize,
    cols: usize,
    row: usize,
    frame: usize,
) -> TrajResult<i8> {
    if rows == 0 || cols == 0 || leaflets.len() != rows * cols {
        return Err(TrajError::Mismatch(
            "leaflets must have shape (n_residues, n_frames) or (n_residues, 1)".into(),
        ));
    }
    let col = if cols == 1 { 0 } else { frame };
    if col >= cols || row >= rows {
        return Err(TrajError::Mismatch(
            "leaflet data does not cover requested frame/residue".into(),
        ));
    }
    Ok(leaflets[row * cols + col])
}

fn pearson(a: &[f64], b: &[f64]) -> f32 {
    if a.len() != b.len() || a.len() < 2 {
        return f32::NAN;
    }
    let n = a.len() as f64;
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut da = 0.0;
    let mut db = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xa = x - mean_a;
        let yb = y - mean_b;
        num += xa * yb;
        da += xa * xa;
        db += yb * yb;
    }
    if da <= 0.0 || db <= 0.0 {
        f32::NAN
    } else {
        (num / (da.sqrt() * db.sqrt())) as f32
    }
}

fn smooth_grid(grid: &[f64], bins: usize, sigma_bins: f64) -> Vec<f64> {
    if sigma_bins <= 0.0 || bins == 0 {
        return grid.to_vec();
    }
    let radius = (sigma_bins * 3.0).ceil() as isize;
    let denom = 2.0 * sigma_bins * sigma_bins;
    let mut out = vec![0.0; grid.len()];
    for ix in 0..bins {
        for iy in 0..bins {
            let mut sum = 0.0;
            let mut weight_sum = 0.0;
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    let wx = dx as f64;
                    let wy = dy as f64;
                    let weight = (-(wx * wx + wy * wy) / denom).exp();
                    let sx = ((ix as isize + dx).rem_euclid(bins as isize)) as usize;
                    let sy = ((iy as isize + dy).rem_euclid(bins as isize)) as usize;
                    sum += grid[sx * bins + sy] * weight;
                    weight_sum += weight;
                }
            }
            out[ix * bins + iy] = if weight_sum > 0.0 {
                sum / weight_sum
            } else {
                0.0
            };
        }
    }
    out
}

fn groups_contact(
    chunk: &FrameChunk,
    frame: usize,
    a: &ResidueGroup,
    b: &ResidueGroup,
    scale: f64,
    lengths: [f64; 3],
    cutoff2: f64,
) -> bool {
    for &ai in &a.atoms {
        let pi = atom_pos(chunk, frame, ai, scale);
        for &bi in &b.atoms {
            let pj = atom_pos(chunk, frame, bi, scale);
            let dx = min_image(pi[0] - pj[0], lengths[0]);
            let dy = min_image(pi[1] - pj[1], lengths[1]);
            let dz = min_image(pi[2] - pj[2], lengths[2]);
            if dx * dx + dy * dy + dz * dz <= cutoff2 {
                return true;
            }
        }
    }
    false
}

fn connected_components(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut seen = vec![false; adj.len()];
    let mut out = Vec::new();
    for start in 0..adj.len() {
        if seen[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut component = Vec::new();
        seen[start] = true;
        while let Some(node) = stack.pop() {
            component.push(node);
            for &next in &adj[node] {
                if !seen[next] {
                    seen[next] = true;
                    stack.push(next);
                }
            }
        }
        out.push(component);
    }
    out
}

fn mean_component_z(
    chunk: &FrameChunk,
    frame: usize,
    groups: &[ResidueGroup],
    indices: &[usize],
    scale: f64,
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    indices
        .iter()
        .map(|&idx| group_mean(chunk, frame, &groups[idx].atoms, scale)[2])
        .sum::<f64>()
        / indices.len() as f64
}

fn mean_groups_position(
    chunk: &FrameChunk,
    frame: usize,
    groups: &[ResidueGroup],
    indices: &[usize],
    scale: f64,
) -> [f64; 3] {
    if indices.is_empty() {
        return [0.0; 3];
    }
    let mut out = [0.0; 3];
    for &idx in indices {
        let p = group_mean(chunk, frame, &groups[idx].atoms, scale);
        out[0] += p[0];
        out[1] += p[1];
        out[2] += p[2];
    }
    [
        out[0] / indices.len() as f64,
        out[1] / indices.len() as f64,
        out[2] / indices.len() as f64,
    ]
}

fn mean_component_radius(
    chunk: &FrameChunk,
    frame: usize,
    groups: &[ResidueGroup],
    indices: &[usize],
    scale: f64,
    center: [f64; 3],
    lengths: [f64; 3],
) -> f64 {
    if indices.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for &idx in indices {
        let p = group_mean(chunk, frame, &groups[idx].atoms, scale);
        let dx = min_image(p[0] - center[0], lengths[0]);
        let dy = min_image(p[1] - center[1], lengths[1]);
        let dz = min_image(p[2] - center[2], lengths[2]);
        sum += (dx * dx + dy * dy + dz * dz).sqrt();
    }
    sum / indices.len() as f64
}

fn min_group_component_distance2(
    chunk: &FrameChunk,
    frame: usize,
    group: &ResidueGroup,
    groups: &[ResidueGroup],
    component: &[usize],
    scale: f64,
    lengths: [f64; 3],
) -> f64 {
    let mut best = f64::INFINITY;
    for &idx in component {
        for &ai in &group.atoms {
            let pi = atom_pos(chunk, frame, ai, scale);
            for &bi in &groups[idx].atoms {
                let pj = atom_pos(chunk, frame, bi, scale);
                let dx = min_image(pi[0] - pj[0], lengths[0]);
                let dy = min_image(pi[1] - pj[1], lengths[1]);
                let dz = min_image(pi[2] - pj[2], lengths[2]);
                best = best.min(dx * dx + dy * dy + dz * dz);
            }
        }
    }
    best
}

pub struct LipidLeafletPlan {
    selection: Selection,
    midplane_selection: Option<Selection>,
    midplane_cutoff: f64,
    bins: usize,
    length_scale: f64,
    groups: Vec<ResidueGroup>,
    midplane_groups: Vec<ResidueGroup>,
    values: Vec<f32>,
    frames: usize,
}

impl LipidLeafletPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            midplane_selection: None,
            midplane_cutoff: 0.0,
            bins: 1,
            length_scale: 1.0,
            groups: Vec::new(),
            midplane_groups: Vec::new(),
            values: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_midplane(mut self, selection: Selection, cutoff: f64) -> Self {
        self.midplane_selection = Some(selection);
        self.midplane_cutoff = cutoff;
        self
    }

    pub fn with_bins(mut self, bins: usize) -> Self {
        self.bins = bins.max(1);
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for LipidLeafletPlan {
    fn name(&self) -> &'static str {
        "lipid_leaflets"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.length_scale <= 0.0 || !self.length_scale.is_finite() {
            return Err(TrajError::Parse(
                "lipid_leaflets length_scale must be > 0".into(),
            ));
        }
        self.groups = groups_for_selection(system, &self.selection)?;
        self.midplane_groups = if let Some(sel) = &self.midplane_selection {
            groups_for_selection(system, sel)?
        } else {
            Vec::new()
        };
        self.values.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        for frame in 0..chunk.n_frames {
            let lengths = box_lengths(chunk, frame)?;
            let grid = midpoint_grid(
                chunk,
                frame,
                &self.groups,
                self.bins,
                self.length_scale,
                lengths,
            );
            let x_len = lengths[0] * self.length_scale;
            let y_len = lengths[1] * self.length_scale;
            let mut frame_values = vec![0.0f32; self.groups.len()];
            for (idx, group) in self.groups.iter().enumerate() {
                let p = group_mean(chunk, frame, &group.atoms, self.length_scale);
                let ix = resolve_bin(p[0], x_len, self.bins).unwrap_or(0);
                let iy = resolve_bin(p[1], y_len, self.bins).unwrap_or(0);
                let midpoint = grid[ix * self.bins + iy];
                frame_values[idx] = if p[2] >= midpoint { 1.0 } else { -1.0 };
            }
            if self.midplane_cutoff > 0.0 && !self.midplane_groups.is_empty() {
                for mid_group in &self.midplane_groups {
                    let all_midplane = mid_group.atoms.iter().all(|&atom| {
                        let p = atom_pos(chunk, frame, atom, self.length_scale);
                        let ix = resolve_bin(p[0], x_len, self.bins).unwrap_or(0);
                        let iy = resolve_bin(p[1], y_len, self.bins).unwrap_or(0);
                        (p[2] - grid[ix * self.bins + iy]).abs() <= self.midplane_cutoff
                    });
                    if all_midplane {
                        if let Some(pos) =
                            self.groups.iter().position(|g| g.resid == mid_group.resid)
                        {
                            frame_values[pos] = 0.0;
                        }
                    }
                }
            }
            self.values.extend(frame_values);
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let rows = self.groups.len();
        let cols = self.frames;
        let groups = self.groups.clone();
        Ok(matrix_output(
            "leaflets",
            std::mem::take(&mut self.values),
            rows,
            cols,
            &groups,
        ))
    }
}

pub struct LipidCurvedLeafletPlan {
    selection: Selection,
    midplane_selection: Option<Selection>,
    cutoff: f64,
    midplane_cutoff: f64,
    length_scale: f64,
    groups: Vec<ResidueGroup>,
    midplane_resids: HashSet<i32>,
    values: Vec<f32>,
    frames: usize,
}

impl LipidCurvedLeafletPlan {
    pub fn new(selection: Selection, cutoff: f64) -> Self {
        Self {
            selection,
            midplane_selection: None,
            cutoff,
            midplane_cutoff: 0.0,
            length_scale: 1.0,
            groups: Vec::new(),
            midplane_resids: HashSet::new(),
            values: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_midplane(mut self, selection: Selection, cutoff: f64) -> Self {
        self.midplane_selection = Some(selection);
        self.midplane_cutoff = cutoff;
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for LipidCurvedLeafletPlan {
    fn name(&self) -> &'static str {
        "lipid_curved_leaflets"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.cutoff <= 0.0 || !self.cutoff.is_finite() {
            return Err(TrajError::Parse(
                "lipid_curved_leaflets cutoff must be > 0".into(),
            ));
        }
        if self.midplane_cutoff < 0.0 || !self.midplane_cutoff.is_finite() {
            return Err(TrajError::Parse(
                "lipid_curved_leaflets midplane_cutoff must be >= 0".into(),
            ));
        }
        self.groups = groups_for_selection(system, &self.selection)?;
        self.midplane_resids = if let Some(sel) = &self.midplane_selection {
            groups_for_selection(system, sel)?
                .into_iter()
                .map(|g| g.resid)
                .collect()
        } else {
            HashSet::new()
        };
        self.values.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let cutoff2 = self.cutoff * self.cutoff;
        let mid_cutoff2 = self.midplane_cutoff * self.midplane_cutoff;
        for frame in 0..chunk.n_frames {
            let lengths = box_lengths(chunk, frame)?;
            let scaled = [
                lengths[0] * self.length_scale,
                lengths[1] * self.length_scale,
                lengths[2] * self.length_scale,
            ];
            let center = [0.5 * scaled[0], 0.5 * scaled[1], 0.5 * scaled[2]];
            let static_rows: Vec<usize> = self
                .groups
                .iter()
                .enumerate()
                .filter_map(|(idx, group)| {
                    (!self.midplane_resids.contains(&group.resid)).then_some(idx)
                })
                .collect();
            if static_rows.is_empty() {
                return Err(TrajError::Mismatch(
                    "lipid_curved_leaflets static selection is empty".into(),
                ));
            }
            let mut adj = vec![Vec::<usize>::new(); static_rows.len()];
            for local_i in 0..static_rows.len() {
                for local_j in (local_i + 1)..static_rows.len() {
                    let gi = static_rows[local_i];
                    let gj = static_rows[local_j];
                    if groups_contact(
                        chunk,
                        frame,
                        &self.groups[gi],
                        &self.groups[gj],
                        self.length_scale,
                        scaled,
                        cutoff2,
                    ) {
                        adj[local_i].push(local_j);
                        adj[local_j].push(local_i);
                    }
                }
            }
            let mut components = connected_components(&adj);
            components.sort_by_key(|component| std::cmp::Reverse(component.len()));
            let mut frame_values = vec![0.0f32; self.groups.len()];
            if components.len() >= 2 {
                let first: Vec<usize> = components[0].iter().map(|&idx| static_rows[idx]).collect();
                let second: Vec<usize> =
                    components[1].iter().map(|&idx| static_rows[idx]).collect();
                let first_radius = mean_component_radius(
                    chunk,
                    frame,
                    &self.groups,
                    &first,
                    self.length_scale,
                    center,
                    scaled,
                );
                let second_radius = mean_component_radius(
                    chunk,
                    frame,
                    &self.groups,
                    &second,
                    self.length_scale,
                    center,
                    scaled,
                );
                let first_z =
                    mean_component_z(chunk, frame, &self.groups, &first, self.length_scale);
                let second_z =
                    mean_component_z(chunk, frame, &self.groups, &second, self.length_scale);
                let first_is_positive = if (first_radius - second_radius).abs() > 1.0e-6 {
                    first_radius > second_radius
                } else {
                    first_z >= second_z
                };
                let (positive, negative) = if first_is_positive {
                    (first, second)
                } else {
                    (second, first)
                };
                for &idx in &positive {
                    frame_values[idx] = 1.0;
                }
                for &idx in &negative {
                    frame_values[idx] = -1.0;
                }
                for component in components.iter().skip(2) {
                    for &local_idx in component {
                        let group_idx = static_rows[local_idx];
                        let dp = min_group_component_distance2(
                            chunk,
                            frame,
                            &self.groups[group_idx],
                            &self.groups,
                            &positive,
                            self.length_scale,
                            scaled,
                        );
                        let dn = min_group_component_distance2(
                            chunk,
                            frame,
                            &self.groups[group_idx],
                            &self.groups,
                            &negative,
                            self.length_scale,
                            scaled,
                        );
                        frame_values[group_idx] = if dp <= dn { 1.0 } else { -1.0 };
                    }
                }
                if self.midplane_cutoff > 0.0 {
                    for (idx, group) in self.groups.iter().enumerate() {
                        if !self.midplane_resids.contains(&group.resid) {
                            continue;
                        }
                        let in_positive = min_group_component_distance2(
                            chunk,
                            frame,
                            group,
                            &self.groups,
                            &positive,
                            self.length_scale,
                            scaled,
                        ) <= mid_cutoff2;
                        let in_negative = min_group_component_distance2(
                            chunk,
                            frame,
                            group,
                            &self.groups,
                            &negative,
                            self.length_scale,
                            scaled,
                        ) <= mid_cutoff2;
                        frame_values[idx] = match (in_positive, in_negative) {
                            (true, false) => 1.0,
                            (false, true) => -1.0,
                            _ => 0.0,
                        };
                    }
                }
            } else {
                let global_center = mean_groups_position(
                    chunk,
                    frame,
                    &self.groups,
                    &static_rows,
                    self.length_scale,
                );
                for &idx in &static_rows {
                    let p = group_mean(chunk, frame, &self.groups[idx].atoms, self.length_scale);
                    frame_values[idx] = if p[2] >= global_center[2] { 1.0 } else { -1.0 };
                }
            }
            self.values.extend(frame_values);
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let rows = self.groups.len();
        let cols = self.frames;
        let groups = self.groups.clone();
        Ok(matrix_output(
            "curved_leaflets",
            std::mem::take(&mut self.values),
            rows,
            cols,
            &groups,
        ))
    }
}

pub struct LipidZPositionPlan {
    membrane_selection: Selection,
    height_selection: Selection,
    bins: usize,
    length_scale: f64,
    membrane_groups: Vec<ResidueGroup>,
    height_groups: Vec<ResidueGroup>,
    values: Vec<f32>,
    frames: usize,
}

impl LipidZPositionPlan {
    pub fn new(membrane_selection: Selection, height_selection: Selection) -> Self {
        Self {
            membrane_selection,
            height_selection,
            bins: 1,
            length_scale: 1.0,
            membrane_groups: Vec::new(),
            height_groups: Vec::new(),
            values: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_bins(mut self, bins: usize) -> Self {
        self.bins = bins.max(1);
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for LipidZPositionPlan {
    fn name(&self) -> &'static str {
        "lipid_z_positions"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.membrane_groups = groups_for_selection(system, &self.membrane_selection)?;
        self.height_groups = groups_for_selection(system, &self.height_selection)?;
        self.values.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        for frame in 0..chunk.n_frames {
            let lengths = box_lengths(chunk, frame)?;
            let grid = midpoint_grid(
                chunk,
                frame,
                &self.membrane_groups,
                self.bins,
                self.length_scale,
                lengths,
            );
            let x_len = lengths[0] * self.length_scale;
            let y_len = lengths[1] * self.length_scale;
            for group in &self.height_groups {
                let p = group_mean(chunk, frame, &group.atoms, self.length_scale);
                let ix = resolve_bin(p[0], x_len, self.bins).unwrap_or(0);
                let iy = resolve_bin(p[1], y_len, self.bins).unwrap_or(0);
                self.values.push((p[2] - grid[ix * self.bins + iy]) as f32);
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let rows = self.height_groups.len();
        let cols = self.frames;
        let groups = self.height_groups.clone();
        Ok(matrix_output(
            "z_positions",
            std::mem::take(&mut self.values),
            rows,
            cols,
            &groups,
        ))
    }
}

pub struct LipidZThicknessPlan {
    selection: Selection,
    length_scale: f64,
    groups: Vec<ResidueGroup>,
    values: Vec<f32>,
    frames: usize,
}

impl LipidZThicknessPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            length_scale: 1.0,
            groups: Vec::new(),
            values: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for LipidZThicknessPlan {
    fn name(&self) -> &'static str {
        "lipid_z_thickness"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.groups = groups_for_selection(system, &self.selection)?;
        self.values.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        for frame in 0..chunk.n_frames {
            let lengths = box_lengths(chunk, frame)?;
            let z_len = lengths[2] * self.length_scale;
            for group in &self.groups {
                let mut min_z = f64::INFINITY;
                let mut max_z = f64::NEG_INFINITY;
                for &atom in &group.atoms {
                    let z = atom_pos(chunk, frame, atom, self.length_scale)[2];
                    min_z = min_z.min(z);
                    max_z = max_z.max(z);
                }
                let mut thickness = max_z - min_z;
                if thickness > z_len * 0.5 {
                    thickness = z_len - thickness;
                }
                self.values.push(thickness as f32);
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let rows = self.groups.len();
        let cols = self.frames;
        let groups = self.groups.clone();
        Ok(matrix_output(
            "z_thickness",
            std::mem::take(&mut self.values),
            rows,
            cols,
            &groups,
        ))
    }
}

pub struct LipidZAnglePlan {
    atom_a: Selection,
    atom_b: Selection,
    degrees: bool,
    length_scale: f64,
    groups: Vec<ResidueGroup>,
    pairs: Vec<(usize, usize)>,
    values: Vec<f32>,
    frames: usize,
}

impl LipidZAnglePlan {
    pub fn new(atom_a: Selection, atom_b: Selection) -> Self {
        Self {
            atom_a,
            atom_b,
            degrees: true,
            length_scale: 1.0,
            groups: Vec::new(),
            pairs: Vec::new(),
            values: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_degrees(mut self, degrees: bool) -> Self {
        self.degrees = degrees;
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for LipidZAnglePlan {
    fn name(&self) -> &'static str {
        "lipid_z_angles"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.atom_a.indices.len() != self.atom_b.indices.len() {
            return Err(TrajError::Mismatch(
                "lipid_z_angles atom selections must have equal length".into(),
            ));
        }
        self.groups = groups_for_selection(system, &self.atom_a)?;
        self.pairs = self
            .atom_a
            .indices
            .iter()
            .zip(self.atom_b.indices.iter())
            .map(|(&a, &b)| (a as usize, b as usize))
            .collect();
        self.values.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        for frame in 0..chunk.n_frames {
            let lengths = box_lengths(chunk, frame)?;
            for &(a, b) in &self.pairs {
                let pa = atom_pos(chunk, frame, a, self.length_scale);
                let pb = atom_pos(chunk, frame, b, self.length_scale);
                let mut v = [pa[0] - pb[0], pa[1] - pb[1], pa[2] - pb[2]];
                for dim in 0..2 {
                    let len = lengths[dim] * self.length_scale;
                    if v[dim] > len * 0.5 {
                        v[dim] -= len;
                    } else if v[dim] < -len * 0.5 {
                        v[dim] += len;
                    }
                }
                let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                let mut angle = if norm > 0.0 {
                    (v[2] / norm).clamp(-1.0, 1.0).acos()
                } else {
                    f64::NAN
                };
                if self.degrees {
                    angle = angle.to_degrees();
                }
                self.values.push(angle as f32);
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let rows = self.pairs.len();
        let cols = self.frames;
        let groups = if self.groups.len() == rows {
            self.groups.clone()
        } else {
            (0..rows)
                .map(|i| ResidueGroup {
                    resid: i as i32,
                    atoms: Vec::new(),
                })
                .collect()
        };
        Ok(matrix_output(
            "z_angles",
            std::mem::take(&mut self.values),
            rows,
            cols,
            &groups,
        ))
    }
}

pub struct LipidFlipFlopPlan {
    leaflets: Vec<i8>,
    rows: usize,
    cols: usize,
    residue_ids: Vec<i32>,
    frame_cutoff: usize,
}

impl LipidFlipFlopPlan {
    pub fn new(
        leaflets: Vec<i8>,
        rows: usize,
        cols: usize,
        residue_ids: Vec<i32>,
        frame_cutoff: usize,
    ) -> Self {
        Self {
            leaflets,
            rows,
            cols,
            residue_ids,
            frame_cutoff: frame_cutoff.max(1),
        }
    }
}

impl Plan for LipidFlipFlopPlan {
    fn name(&self) -> &'static str {
        "lipid_flip_flop"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        if self.rows * self.cols != self.leaflets.len() {
            return Err(TrajError::Mismatch(
                "leaflets matrix shape does not match data length".into(),
            ));
        }
        Ok(())
    }

    fn process_chunk(
        &mut self,
        _chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let mut events = Vec::new();
        let mut success = Vec::new();
        for row in 0..self.rows {
            let series = &self.leaflets[row * self.cols..(row + 1) * self.cols];
            if series.windows(2).all(|w| w[0] == w[1]) {
                continue;
            }
            for event in molecule_flip_flop(series, self.frame_cutoff) {
                events.extend([
                    self.residue_ids.get(row).copied().unwrap_or(row as i32),
                    event.start as i32,
                    event.end as i32,
                    event.leaflet as i32,
                ]);
                success.push(event.success);
            }
        }
        Ok(PlanOutput::LipidFlipFlop(LipidFlipFlopOutput {
            rows: success.len(),
            cols: 4,
            events,
            success,
            residue_ids: self.residue_ids.clone(),
        }))
    }
}

struct FlipEvent {
    start: usize,
    end: usize,
    leaflet: i8,
    success: String,
}

fn molecule_flip_flop(leaflets: &[i8], frame_cutoff: usize) -> Vec<FlipEvent> {
    if leaflets.is_empty() {
        return Vec::new();
    }
    let mut events = Vec::new();
    let mut current = leaflets[0];
    let mut opposing = -current;
    let mut event_start = None;
    let mut event_stop = None;
    let mut event_success = None;
    let mut n_left = 0usize;
    let mut n_opposing = 0usize;
    let mut n_returned = 0usize;
    for frame in 1..leaflets.len() {
        let leaflet = leaflets[frame];
        if event_start.is_none() {
            n_left = if leaflet == current { 0 } else { n_left + 1 };
            if n_left == frame_cutoff {
                event_start = Some(frame - frame_cutoff);
                n_left = 0;
            }
        }
        n_opposing = if leaflet != opposing {
            0
        } else {
            n_opposing + 1
        };
        if event_start.is_none() {
            continue;
        }
        n_returned = if leaflet != current {
            0
        } else {
            n_returned + 1
        };
        if n_returned == frame_cutoff {
            event_stop = Some(frame - (frame_cutoff - 1));
            event_success = Some(false);
        }
        if n_opposing == frame_cutoff {
            event_stop = Some(frame - (frame_cutoff - 1));
            event_success = Some(true);
        }
        if event_stop.is_none() {
            continue;
        }
        let succeeded = event_success.unwrap_or(false);
        if succeeded {
            (current, opposing) = (opposing, current);
        }
        events.push(FlipEvent {
            start: event_start.unwrap_or(0),
            end: event_stop.unwrap_or(frame),
            leaflet: current,
            success: if succeeded { "Success" } else { "Fail" }.to_string(),
        });
        event_start = None;
        event_stop = None;
        event_success = None;
        n_left = 0;
        n_opposing = 0;
        n_returned = 0;
    }
    if let Some(start) = event_start {
        events.push(FlipEvent {
            start,
            end: leaflets.len() - 1,
            leaflet: *leaflets.last().unwrap(),
            success: "Ongoing".to_string(),
        });
    }
    events
}

pub struct LipidAreaPlan {
    selection: Selection,
    leaflets: Vec<i8>,
    leaflet_rows: usize,
    leaflet_cols: usize,
    length_scale: f64,
    groups: Vec<ResidueGroup>,
    values: Vec<f32>,
    frames: usize,
}

impl LipidAreaPlan {
    pub fn new(selection: Selection, leaflets: Vec<i8>, rows: usize, cols: usize) -> Self {
        Self {
            selection,
            leaflets,
            leaflet_rows: rows,
            leaflet_cols: cols,
            length_scale: 1.0,
            groups: Vec::new(),
            values: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for LipidAreaPlan {
    fn name(&self) -> &'static str {
        "lipid_area"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.groups = groups_for_selection(system, &self.selection)?;
        if self.leaflet_rows != self.groups.len()
            || self.leaflet_cols == 0
            || self.leaflet_rows * self.leaflet_cols != self.leaflets.len()
        {
            return Err(TrajError::Mismatch(
                "lipid_area leaflets must have shape (n_residues, n_frames) or (n_residues, 1)"
                    .into(),
            ));
        }
        self.values.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        for frame in 0..chunk.n_frames {
            let global_frame = self.frames;
            let leaflet_frame = if self.leaflet_cols == 1 {
                0
            } else {
                global_frame
            };
            if leaflet_frame >= self.leaflet_cols {
                return Err(TrajError::Mismatch(
                    "lipid_area processed more frames than provided in leaflets".into(),
                ));
            }
            let lengths = box_lengths(chunk, frame)?;
            let lx = lengths[0] * self.length_scale;
            let ly = lengths[1] * self.length_scale;
            let mut frame_values = vec![0.0f32; self.groups.len()];
            let mut has_area = vec![false; self.groups.len()];
            for leaflet in [-1i8, 1i8] {
                let mut seed_to_group = Vec::new();
                let mut seeds = Vec::new();
                for (group_idx, group) in self.groups.iter().enumerate() {
                    if self.leaflets[group_idx * self.leaflet_cols + leaflet_frame] != leaflet {
                        continue;
                    }
                    for &atom in &group.atoms {
                        let p = atom_pos(chunk, frame, atom, self.length_scale);
                        seeds.push([wrap_value(p[0], lx), wrap_value(p[1], ly)]);
                        seed_to_group.push(group_idx);
                    }
                }
                if seeds.is_empty() {
                    continue;
                }
                let areas = periodic_voronoi_areas(&seeds, lx, ly);
                for (seed_idx, area) in areas.into_iter().enumerate() {
                    let group_idx = seed_to_group[seed_idx];
                    frame_values[group_idx] += area as f32;
                    has_area[group_idx] = true;
                }
            }
            for (value, has) in frame_values.iter_mut().zip(has_area.iter()) {
                if !has {
                    *value = f32::NAN;
                }
            }
            self.values.extend(frame_values);
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let rows = self.groups.len();
        let cols = self.frames;
        let groups = self.groups.clone();
        Ok(matrix_output(
            "area_per_lipid",
            std::mem::take(&mut self.values),
            rows,
            cols,
            &groups,
        ))
    }
}

fn periodic_voronoi_areas(seeds: &[[f64; 2]], lx: f64, ly: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(seeds.len());
    for (i, seed) in seeds.iter().enumerate() {
        let half_x = 0.5 * lx;
        let half_y = 0.5 * ly;
        let mut poly = vec![
            [seed[0] - half_x, seed[1] - half_y],
            [seed[0] + half_x, seed[1] - half_y],
            [seed[0] + half_x, seed[1] + half_y],
            [seed[0] - half_x, seed[1] + half_y],
        ];
        for (j, other) in seeds.iter().enumerate() {
            if i == j {
                continue;
            }
            for ox in -1..=1 {
                for oy in -1..=1 {
                    let image = [other[0] + ox as f64 * lx, other[1] + oy as f64 * ly];
                    if (image[0] - seed[0]).abs() < 1.0e-12 && (image[1] - seed[1]).abs() < 1.0e-12
                    {
                        continue;
                    }
                    poly = clip_to_bisector(&poly, *seed, image);
                    if poly.is_empty() {
                        break;
                    }
                }
                if poly.is_empty() {
                    break;
                }
            }
            if poly.is_empty() {
                break;
            }
        }
        out.push(polygon_area_geo(&poly));
    }
    out
}

fn clip_to_bisector(poly: &[[f64; 2]], site: [f64; 2], other: [f64; 2]) -> Vec<[f64; 2]> {
    if poly.is_empty() {
        return Vec::new();
    }
    let nx = other[0] - site[0];
    let ny = other[1] - site[1];
    let c =
        0.5 * (other[0] * other[0] + other[1] * other[1] - site[0] * site[0] - site[1] * site[1]);
    let inside = |p: [f64; 2]| p[0] * nx + p[1] * ny <= c + 1.0e-9;
    let intersect = |a: [f64; 2], b: [f64; 2]| {
        let da = a[0] * nx + a[1] * ny - c;
        let db = b[0] * nx + b[1] * ny - c;
        let t = if (da - db).abs() < 1.0e-12 {
            0.0
        } else {
            da / (da - db)
        };
        [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
    };
    let mut out = Vec::new();
    let mut prev = *poly.last().unwrap();
    let mut prev_inside = inside(prev);
    for &curr in poly {
        let curr_inside = inside(curr);
        if curr_inside {
            if !prev_inside {
                out.push(intersect(prev, curr));
            }
            out.push(curr);
        } else if prev_inside {
            out.push(intersect(prev, curr));
        }
        prev = curr;
        prev_inside = curr_inside;
    }
    out
}

fn polygon_area_geo(poly: &[[f64; 2]]) -> f64 {
    if poly.len() < 3 {
        return 0.0;
    }
    let mut coords: Vec<(f64, f64)> = poly.iter().map(|p| (p[0], p[1])).collect();
    coords.push((poly[0][0], poly[0][1]));
    let polygon = Polygon::new(LineString::from(coords), Vec::new());
    polygon.unsigned_area()
}

pub struct LipidSccPlan {
    selection: Selection,
    normals: Option<Vec<f32>>,
    normal_rows: usize,
    normal_cols: usize,
    length_scale: f64,
    groups: Vec<ResidueGroup>,
    values: Vec<f32>,
    frames: usize,
}

impl LipidSccPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            normals: None,
            normal_rows: 0,
            normal_cols: 0,
            length_scale: 1.0,
            groups: Vec::new(),
            values: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_normals(mut self, normals: Vec<f32>, rows: usize, cols: usize) -> Self {
        self.normals = Some(normals);
        self.normal_rows = rows;
        self.normal_cols = cols;
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for LipidSccPlan {
    fn name(&self) -> &'static str {
        "lipid_scc"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.groups = groups_for_selection(system, &self.selection)?;
        if let Some(normals) = &self.normals {
            if self.normal_rows != self.groups.len()
                || self.normal_cols == 0
                || normals.len() != self.normal_rows * self.normal_cols * 3
            {
                return Err(TrajError::Mismatch(
                    "lipid_scc normals must have shape (n_residues, n_frames, 3) or (n_residues, 1, 3)"
                        .into(),
                ));
            }
        }
        self.values.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        for frame in 0..chunk.n_frames {
            let global_frame = self.frames;
            let lengths = box_lengths(chunk, frame)?;
            let scaled = [
                lengths[0] * self.length_scale,
                lengths[1] * self.length_scale,
                lengths[2] * self.length_scale,
            ];
            let mut frame_values = vec![f32::NAN; self.groups.len()];
            for (row, group) in self.groups.iter().enumerate() {
                if group.atoms.len() < 2 {
                    continue;
                }
                let normal = if let Some(normals) = &self.normals {
                    let col = if self.normal_cols == 1 {
                        0
                    } else {
                        global_frame
                    };
                    if col >= self.normal_cols {
                        return Err(TrajError::Mismatch(
                            "lipid_scc processed more frames than provided in normals".into(),
                        ));
                    }
                    let idx = (row * self.normal_cols + col) * 3;
                    [
                        normals[idx] as f64,
                        normals[idx + 1] as f64,
                        normals[idx + 2] as f64,
                    ]
                } else {
                    [0.0, 0.0, 1.0]
                };
                let normal_norm =
                    (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
                if normal_norm <= 0.0 || !normal_norm.is_finite() {
                    continue;
                }
                let mut sum = 0.0;
                let mut count = 0usize;
                for pair in group.atoms.windows(2) {
                    let a = atom_pos(chunk, frame, pair[0], self.length_scale);
                    let b = atom_pos(chunk, frame, pair[1], self.length_scale);
                    let v = [
                        min_image(b[0] - a[0], scaled[0]),
                        min_image(b[1] - a[1], scaled[1]),
                        min_image(b[2] - a[2], scaled[2]),
                    ];
                    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    if norm <= 0.0 || !norm.is_finite() {
                        continue;
                    }
                    let cos = (v[0] * normal[0] + v[1] * normal[1] + v[2] * normal[2])
                        / (norm * normal_norm);
                    sum += 0.5 * (3.0 * cos * cos - 1.0);
                    count += 1;
                }
                if count > 0 {
                    frame_values[row] = (sum / count as f64) as f32;
                }
            }
            self.values.extend(frame_values);
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let rows = self.groups.len();
        let cols = self.frames;
        let groups = self.groups.clone();
        Ok(matrix_output(
            "scc",
            std::mem::take(&mut self.values),
            rows,
            cols,
            &groups,
        ))
    }
}

pub struct LipidNeighbourPlan {
    selection: Selection,
    cutoff: f64,
    length_scale: f64,
    groups: Vec<ResidueGroup>,
    values: Vec<f32>,
    frames: usize,
}

impl LipidNeighbourPlan {
    pub fn new(selection: Selection, cutoff: f64) -> Self {
        Self {
            selection,
            cutoff,
            length_scale: 1.0,
            groups: Vec::new(),
            values: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for LipidNeighbourPlan {
    fn name(&self) -> &'static str {
        "lipid_neighbours"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.cutoff <= 0.0 || !self.cutoff.is_finite() {
            return Err(TrajError::Parse(
                "lipid_neighbours cutoff must be > 0".into(),
            ));
        }
        self.groups = groups_for_selection(system, &self.selection)?;
        self.values.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let cutoff2 = self.cutoff * self.cutoff;
        for frame in 0..chunk.n_frames {
            let lengths = box_lengths(chunk, frame)?;
            let scaled = [
                lengths[0] * self.length_scale,
                lengths[1] * self.length_scale,
                lengths[2] * self.length_scale,
            ];
            let mut counts = vec![0.0f32; self.groups.len()];
            for i in 0..self.groups.len() {
                for j in (i + 1)..self.groups.len() {
                    let mut contact = false;
                    'pairs: for &ai in &self.groups[i].atoms {
                        let pi = atom_pos(chunk, frame, ai, self.length_scale);
                        for &aj in &self.groups[j].atoms {
                            let pj = atom_pos(chunk, frame, aj, self.length_scale);
                            let dx = min_image(pi[0] - pj[0], scaled[0]);
                            let dy = min_image(pi[1] - pj[1], scaled[1]);
                            let dz = min_image(pi[2] - pj[2], scaled[2]);
                            if dx * dx + dy * dy + dz * dz <= cutoff2 {
                                contact = true;
                                break 'pairs;
                            }
                        }
                    }
                    if contact {
                        counts[i] += 1.0;
                        counts[j] += 1.0;
                    }
                }
            }
            self.values.extend(counts);
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let rows = self.groups.len();
        let cols = self.frames;
        let groups = self.groups.clone();
        Ok(matrix_output(
            "neighbour_counts",
            std::mem::take(&mut self.values),
            rows,
            cols,
            &groups,
        ))
    }
}

pub struct LipidLargestClusterPlan {
    selection: Selection,
    cutoff: f64,
    length_scale: f64,
    groups: Vec<ResidueGroup>,
    values: Vec<f32>,
}

impl LipidLargestClusterPlan {
    pub fn new(selection: Selection, cutoff: f64) -> Self {
        Self {
            selection,
            cutoff,
            length_scale: 1.0,
            groups: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for LipidLargestClusterPlan {
    fn name(&self) -> &'static str {
        "lipid_largest_cluster"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.cutoff <= 0.0 || !self.cutoff.is_finite() {
            return Err(TrajError::Parse(
                "lipid_largest_cluster cutoff must be > 0".into(),
            ));
        }
        self.groups = groups_for_selection(system, &self.selection)?;
        self.values.clear();
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let cutoff2 = self.cutoff * self.cutoff;
        for frame in 0..chunk.n_frames {
            let lengths = box_lengths(chunk, frame)?;
            let scaled = [
                lengths[0] * self.length_scale,
                lengths[1] * self.length_scale,
                lengths[2] * self.length_scale,
            ];
            let mut adj = vec![Vec::<usize>::new(); self.groups.len()];
            for i in 0..self.groups.len() {
                for j in (i + 1)..self.groups.len() {
                    let mut contact = false;
                    'pairs: for &ai in &self.groups[i].atoms {
                        let pi = atom_pos(chunk, frame, ai, self.length_scale);
                        for &aj in &self.groups[j].atoms {
                            let pj = atom_pos(chunk, frame, aj, self.length_scale);
                            let dx = min_image(pi[0] - pj[0], scaled[0]);
                            let dy = min_image(pi[1] - pj[1], scaled[1]);
                            let dz = min_image(pi[2] - pj[2], scaled[2]);
                            if dx * dx + dy * dy + dz * dz <= cutoff2 {
                                contact = true;
                                break 'pairs;
                            }
                        }
                    }
                    if contact {
                        adj[i].push(j);
                        adj[j].push(i);
                    }
                }
            }
            let mut seen = vec![false; self.groups.len()];
            let mut largest = 0usize;
            for start in 0..self.groups.len() {
                if seen[start] {
                    continue;
                }
                let mut stack = vec![start];
                seen[start] = true;
                let mut size = 0usize;
                while let Some(node) = stack.pop() {
                    size += 1;
                    for &next in &adj[node] {
                        if !seen[next] {
                            seen[next] = true;
                            stack.push(next);
                        }
                    }
                }
                largest = largest.max(size);
            }
            self.values.push(largest as f32);
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let cols = self.values.len();
        Ok(explicit_matrix_output(
            "largest_cluster",
            std::mem::take(&mut self.values),
            1,
            cols,
            vec![0],
        ))
    }
}

pub struct LipidNeighbourMatrixPlan {
    selection: Selection,
    cutoff: f64,
    length_scale: f64,
    groups: Vec<ResidueGroup>,
    values: Vec<f32>,
    frames: usize,
}

impl LipidNeighbourMatrixPlan {
    pub fn new(selection: Selection, cutoff: f64) -> Self {
        Self {
            selection,
            cutoff,
            length_scale: 1.0,
            groups: Vec::new(),
            values: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for LipidNeighbourMatrixPlan {
    fn name(&self) -> &'static str {
        "lipid_neighbour_matrix"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.cutoff <= 0.0 || !self.cutoff.is_finite() {
            return Err(TrajError::Parse(
                "lipid_neighbour_matrix cutoff must be > 0".into(),
            ));
        }
        self.groups = groups_for_selection(system, &self.selection)?;
        self.values.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let cutoff2 = self.cutoff * self.cutoff;
        let n = self.groups.len();
        for frame in 0..chunk.n_frames {
            let lengths = box_lengths(chunk, frame)?;
            let scaled = [
                lengths[0] * self.length_scale,
                lengths[1] * self.length_scale,
                lengths[2] * self.length_scale,
            ];
            let mut matrix = vec![0.0f32; n * n];
            for i in 0..n {
                for j in (i + 1)..n {
                    if groups_contact(
                        chunk,
                        frame,
                        &self.groups[i],
                        &self.groups[j],
                        self.length_scale,
                        scaled,
                        cutoff2,
                    ) {
                        matrix[i * n + j] = 1.0;
                        matrix[j * n + i] = 1.0;
                    }
                }
            }
            self.values.extend(matrix);
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let n = self.groups.len();
        let cols = self.frames;
        let values = std::mem::take(&mut self.values);
        let mut row_major = vec![f32::NAN; n * n * cols];
        for frame in 0..cols {
            for row in 0..(n * n) {
                row_major[row * cols + frame] = values[frame * n * n + row];
            }
        }
        Ok(explicit_matrix_output(
            "neighbour_matrix",
            row_major,
            n * n,
            cols,
            self.groups.iter().map(|g| g.resid).collect(),
        ))
    }
}

pub struct LipidMembraneThicknessPlan {
    selection: Selection,
    leaflets: Vec<i8>,
    leaflet_rows: usize,
    leaflet_cols: usize,
    bins: usize,
    length_scale: f64,
    groups: Vec<ResidueGroup>,
    values: Vec<f32>,
    frames: usize,
}

impl LipidMembraneThicknessPlan {
    pub fn new(selection: Selection, leaflets: Vec<i8>, rows: usize, cols: usize) -> Self {
        Self {
            selection,
            leaflets,
            leaflet_rows: rows,
            leaflet_cols: cols,
            bins: 1,
            length_scale: 1.0,
            groups: Vec::new(),
            values: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_bins(mut self, bins: usize) -> Self {
        self.bins = bins.max(1);
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for LipidMembraneThicknessPlan {
    fn name(&self) -> &'static str {
        "lipid_membrane_thickness"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.groups = groups_for_selection(system, &self.selection)?;
        if self.leaflet_rows != self.groups.len()
            || self.leaflet_cols == 0
            || self.leaflet_rows * self.leaflet_cols != self.leaflets.len()
        {
            return Err(TrajError::Mismatch(
                "lipid_membrane_thickness leaflets must have shape (n_residues, n_frames) or (n_residues, 1)"
                    .into(),
            ));
        }
        self.values.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        for frame in 0..chunk.n_frames {
            let global_frame = self.frames;
            let lengths = box_lengths(chunk, frame)?;
            let x_len = lengths[0] * self.length_scale;
            let y_len = lengths[1] * self.length_scale;
            let bins = self.bins;
            let mut upper_sum = vec![0.0; bins * bins];
            let mut lower_sum = vec![0.0; bins * bins];
            let mut upper_count = vec![0usize; bins * bins];
            let mut lower_count = vec![0usize; bins * bins];
            for (idx, group) in self.groups.iter().enumerate() {
                let leaflet = leaflet_value(
                    &self.leaflets,
                    self.leaflet_rows,
                    self.leaflet_cols,
                    idx,
                    global_frame,
                )?;
                if leaflet == 0 {
                    continue;
                }
                for &atom in &group.atoms {
                    let p = atom_pos(chunk, frame, atom, self.length_scale);
                    let ix = resolve_bin(p[0], x_len, bins).unwrap_or(0);
                    let iy = resolve_bin(p[1], y_len, bins).unwrap_or(0);
                    let cell = ix * bins + iy;
                    if leaflet > 0 {
                        upper_sum[cell] += p[2];
                        upper_count[cell] += 1;
                    } else {
                        lower_sum[cell] += p[2];
                        lower_count[cell] += 1;
                    }
                }
            }
            let mut sum = 0.0;
            let mut count = 0usize;
            for cell in 0..bins * bins {
                if upper_count[cell] > 0 && lower_count[cell] > 0 {
                    sum += upper_sum[cell] / upper_count[cell] as f64
                        - lower_sum[cell] / lower_count[cell] as f64;
                    count += 1;
                }
            }
            self.values.push(if count > 0 {
                (sum / count as f64) as f32
            } else {
                f32::NAN
            });
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let cols = self.values.len();
        Ok(explicit_matrix_output(
            "membrane_thickness",
            std::mem::take(&mut self.values),
            1,
            cols,
            vec![0],
        ))
    }
}

pub struct LipidRegistrationPlan {
    upper_selection: Selection,
    lower_selection: Selection,
    leaflets: Vec<i8>,
    leaflet_rows: usize,
    leaflet_cols: usize,
    bins: usize,
    gaussian_sd: f64,
    length_scale: f64,
    upper_groups: Vec<ResidueGroup>,
    lower_groups: Vec<ResidueGroup>,
    values: Vec<f32>,
    frames: usize,
}

impl LipidRegistrationPlan {
    pub fn new(
        upper_selection: Selection,
        lower_selection: Selection,
        leaflets: Vec<i8>,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self {
            upper_selection,
            lower_selection,
            leaflets,
            leaflet_rows: rows,
            leaflet_cols: cols,
            bins: 1,
            gaussian_sd: 0.0,
            length_scale: 1.0,
            upper_groups: Vec::new(),
            lower_groups: Vec::new(),
            values: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_bins(mut self, bins: usize) -> Self {
        self.bins = bins.max(1);
        self
    }

    pub fn with_gaussian_sd(mut self, sd: f64) -> Self {
        self.gaussian_sd = sd.max(0.0);
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for LipidRegistrationPlan {
    fn name(&self) -> &'static str {
        "lipid_registration"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.upper_groups = groups_for_selection(system, &self.upper_selection)?;
        self.lower_groups = groups_for_selection(system, &self.lower_selection)?;
        if self.upper_groups.len() != self.lower_groups.len()
            || self.upper_groups.len() != self.leaflet_rows
            || self.leaflet_cols == 0
            || self.leaflet_rows * self.leaflet_cols != self.leaflets.len()
        {
            return Err(TrajError::Mismatch(
                "lipid_registration selections and leaflets must cover the same residues".into(),
            ));
        }
        for (upper, lower) in self.upper_groups.iter().zip(self.lower_groups.iter()) {
            if upper.resid != lower.resid {
                return Err(TrajError::Mismatch(
                    "lipid_registration upper/lower selections must have matching residue order"
                        .into(),
                ));
            }
        }
        self.values.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        for frame in 0..chunk.n_frames {
            let global_frame = self.frames;
            let lengths = box_lengths(chunk, frame)?;
            let x_len = lengths[0] * self.length_scale;
            let y_len = lengths[1] * self.length_scale;
            let bins = self.bins;
            let mut upper = vec![0.0; bins * bins];
            let mut lower = vec![0.0; bins * bins];
            for row in 0..self.upper_groups.len() {
                let leaflet = leaflet_value(
                    &self.leaflets,
                    self.leaflet_rows,
                    self.leaflet_cols,
                    row,
                    global_frame,
                )?;
                if leaflet > 0 {
                    let p = group_mean(
                        chunk,
                        frame,
                        &self.upper_groups[row].atoms,
                        self.length_scale,
                    );
                    let ix = resolve_bin(p[0], x_len, bins).unwrap_or(0);
                    let iy = resolve_bin(p[1], y_len, bins).unwrap_or(0);
                    upper[ix * bins + iy] += 1.0;
                } else if leaflet < 0 {
                    let p = group_mean(
                        chunk,
                        frame,
                        &self.lower_groups[row].atoms,
                        self.length_scale,
                    );
                    let ix = resolve_bin(p[0], x_len, bins).unwrap_or(0);
                    let iy = resolve_bin(p[1], y_len, bins).unwrap_or(0);
                    lower[ix * bins + iy] += 1.0;
                }
            }
            let bin_width = if bins > 0 { x_len / bins as f64 } else { 1.0 };
            let sigma_bins = if bin_width > 0.0 {
                self.gaussian_sd / bin_width
            } else {
                0.0
            };
            let upper = smooth_grid(&upper, bins, sigma_bins);
            let lower = smooth_grid(&lower, bins, sigma_bins);
            self.values.push(pearson(&upper, &lower));
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let cols = self.values.len();
        Ok(explicit_matrix_output(
            "registration",
            std::mem::take(&mut self.values),
            1,
            cols,
            vec![0],
        ))
    }
}

pub struct LipidMsdPlan {
    selection: Selection,
    com_removal_selection: Option<Selection>,
    length_scale: f64,
    groups: Vec<ResidueGroup>,
    com_group: Vec<usize>,
    positions: Vec<Vec<[f64; 2]>>,
}

impl LipidMsdPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            com_removal_selection: None,
            length_scale: 1.0,
            groups: Vec::new(),
            com_group: Vec::new(),
            positions: Vec::new(),
        }
    }

    pub fn with_com_removal(mut self, selection: Selection) -> Self {
        self.com_removal_selection = Some(selection);
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for LipidMsdPlan {
    fn name(&self) -> &'static str {
        "lipid_msd"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.groups = groups_for_selection(system, &self.selection)?;
        self.com_group = self
            .com_removal_selection
            .as_ref()
            .map(|sel| sel.indices.iter().map(|&idx| idx as usize).collect())
            .unwrap_or_default();
        self.positions = vec![Vec::new(); self.groups.len()];
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        for frame in 0..chunk.n_frames {
            let com_shift = if self.com_group.is_empty() {
                [0.0, 0.0]
            } else {
                let p = group_mean(chunk, frame, &self.com_group, self.length_scale);
                [p[0], p[1]]
            };
            for (idx, group) in self.groups.iter().enumerate() {
                let p = group_mean(chunk, frame, &group.atoms, self.length_scale);
                self.positions[idx].push([p[0] - com_shift[0], p[1] - com_shift[1]]);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let rows = self.groups.len();
        let cols = self.positions.first().map(|p| p.len()).unwrap_or(0);
        let mut values = vec![f32::NAN; rows * cols];
        for row in 0..rows {
            for lag in 0..cols {
                let count = cols - lag;
                if count == 0 {
                    continue;
                }
                let mut sum = 0.0;
                for start in 0..count {
                    let a = self.positions[row][start];
                    let b = self.positions[row][start + lag];
                    let dx = b[0] - a[0];
                    let dy = b[1] - a[1];
                    sum += dx * dx + dy * dy;
                }
                values[row * cols + lag] = (sum / count as f64) as f32;
            }
        }
        let residue_ids = self.groups.iter().map(|g| g.resid).collect();
        Ok(explicit_matrix_output(
            "lateral_msd",
            values,
            rows,
            cols,
            residue_ids,
        ))
    }
}
