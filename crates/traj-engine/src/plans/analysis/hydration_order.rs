use std::cmp::Ordering;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::pbc_math::box_diagonal_extents;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, HydOrderOutput, Plan, PlanOutput};

use super::binning::resolve_bins;
use super::solvent_orientation::{dot, minimum_image_between_atoms, normalize};

pub struct HydrationOrderPlan {
    selection: Selection,
    axis: usize,
    bin: f64,
    tblock: usize,
    sgang1: Option<f64>,
    sgang2: Option<f64>,
    length_scale: f64,
    dims: [usize; 3],
    origins: [f64; 3],
    bounds: [f64; 6],
    sg_grid_sum: Vec<f64>,
    sk_grid_sum: Vec<f64>,
    counts: Vec<u64>,
    sg_sum_total: f64,
    sk_sum_total: f64,
    sample_count: u64,
    n_frames: usize,
    used_box: bool,
    block_sg_mean: Vec<f64>,
    block_frames: usize,
    interface_lower: Vec<f32>,
    interface_upper: Vec<f32>,
    interface_blocks: usize,
    initialized: bool,
}

impl HydrationOrderPlan {
    pub fn new(selection: Selection, axis: usize, bin: f64) -> Self {
        Self {
            selection,
            axis,
            bin,
            tblock: 1,
            sgang1: None,
            sgang2: None,
            length_scale: 1.0,
            dims: [0, 0, 0],
            origins: [0.0, 0.0, 0.0],
            bounds: [0.0; 6],
            sg_grid_sum: Vec::new(),
            sk_grid_sum: Vec::new(),
            counts: Vec::new(),
            sg_sum_total: 0.0,
            sk_sum_total: 0.0,
            sample_count: 0,
            n_frames: 0,
            used_box: false,
            block_sg_mean: Vec::new(),
            block_frames: 0,
            interface_lower: Vec::new(),
            interface_upper: Vec::new(),
            interface_blocks: 0,
            initialized: false,
        }
    }

    pub fn with_tblock(mut self, tblock: usize) -> Self {
        self.tblock = tblock;
        self
    }

    pub fn with_interface_thresholds(mut self, sgang1: Option<f64>, sgang2: Option<f64>) -> Self {
        self.sgang1 = sgang1;
        self.sgang2 = sgang2;
        self
    }

    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    fn interface_enabled(&self) -> bool {
        self.sgang1.is_some() && self.sgang2.is_some()
    }

    fn plane_axes(&self) -> [usize; 2] {
        match self.axis {
            0 => [1, 2],
            1 => [0, 2],
            _ => [0, 1],
        }
    }

    fn initialize_from_frame(&mut self, chunk: &FrameChunk, frame: usize) -> TrajResult<()> {
        let base = frame * chunk.n_atoms;
        if let Some(lengths) = box_diagonal_extents(chunk.box_[frame]) {
            self.used_box = true;
            self.origins = [0.0, 0.0, 0.0];
            for axis in 0..3 {
                let extent = lengths[axis] * self.length_scale;
                self.dims[axis] = resolve_bins(None, self.bin, extent, "hydorder")?;
                self.bounds[2 * axis] = 0.0;
                self.bounds[2 * axis + 1] = self.dims[axis] as f64 * self.bin;
            }
        } else {
            self.used_box = false;
            let mut mins = [f64::INFINITY; 3];
            let mut maxs = [f64::NEG_INFINITY; 3];
            for &idx_u32 in self.selection.indices.iter() {
                let pos = chunk.coords[base + idx_u32 as usize];
                for axis in 0..3 {
                    let value = pos[axis] as f64 * self.length_scale;
                    mins[axis] = mins[axis].min(value);
                    maxs[axis] = maxs[axis].max(value);
                }
            }
            for axis in 0..3 {
                if !mins[axis].is_finite() || !maxs[axis].is_finite() {
                    return Err(TrajError::Mismatch(
                        "hydorder could not determine spatial bounds".into(),
                    ));
                }
                if maxs[axis] <= mins[axis] {
                    maxs[axis] = mins[axis] + self.bin;
                }
                let extent = maxs[axis] - mins[axis];
                self.dims[axis] = resolve_bins(None, self.bin, extent, "hydorder")?;
                self.origins[axis] = mins[axis];
                self.bounds[2 * axis] = mins[axis];
                self.bounds[2 * axis + 1] = mins[axis] + self.dims[axis] as f64 * self.bin;
            }
        }

        let grid_len = self.dims[0] * self.dims[1] * self.dims[2];
        self.sg_grid_sum = vec![0.0; grid_len];
        self.sk_grid_sum = vec![0.0; grid_len];
        self.counts = vec![0; grid_len];
        if self.interface_enabled() {
            self.block_sg_mean = vec![0.0; grid_len];
        } else {
            self.block_sg_mean.clear();
        }
        self.initialized = true;
        Ok(())
    }
}

impl Plan for HydrationOrderPlan {
    fn name(&self) -> &'static str {
        "hydorder"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.axis > 2 {
            return Err(TrajError::Parse("hydorder axis must be 0, 1, or 2".into()));
        }
        if !self.bin.is_finite() || self.bin <= 0.0 {
            return Err(TrajError::Parse(
                "hydorder bin must be finite and > 0".into(),
            ));
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(TrajError::Parse(
                "hydorder length_scale must be finite and > 0".into(),
            ));
        }
        if self.tblock == 0 {
            return Err(TrajError::Parse("hydorder tblock must be > 0".into()));
        }
        if self.selection.indices.len() < 5 {
            return Err(TrajError::Mismatch(
                "hydorder requires at least 5 selected atoms".into(),
            ));
        }
        match (self.sgang1, self.sgang2) {
            (Some(a), Some(b)) if a.is_finite() && b.is_finite() => {}
            (None, None) => {}
            _ => {
                return Err(TrajError::Parse(
                    "hydorder requires both sgang1 and sgang2 to enable interface extraction"
                        .into(),
                ));
            }
        }
        let n_atoms = system.n_atoms();
        for &idx_u32 in self.selection.indices.iter() {
            if idx_u32 as usize >= n_atoms {
                return Err(TrajError::Mismatch(
                    "hydorder selected atom index out of bounds".into(),
                ));
            }
        }

        self.dims = [0, 0, 0];
        self.origins = [0.0, 0.0, 0.0];
        self.bounds = [0.0; 6];
        self.sg_grid_sum.clear();
        self.sk_grid_sum.clear();
        self.counts.clear();
        self.sg_sum_total = 0.0;
        self.sk_sum_total = 0.0;
        self.sample_count = 0;
        self.n_frames = 0;
        self.used_box = false;
        self.block_sg_mean.clear();
        self.block_frames = 0;
        self.interface_lower.clear();
        self.interface_upper.clear();
        self.interface_blocks = 0;
        self.initialized = false;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.selection.indices.is_empty() {
            self.n_frames += chunk.n_frames;
            return Ok(());
        }

        for frame in 0..chunk.n_frames {
            if !self.initialized {
                self.initialize_from_frame(chunk, frame)?;
            }
            let base = frame * chunk.n_atoms;
            let frame_coords = &chunk.coords[base..base + chunk.n_atoms];
            let lengths = if self.used_box {
                box_diagonal_extents(chunk.box_[frame])
                    .ok_or_else(|| {
                        TrajError::Mismatch(
                        "hydorder requires box metadata on every frame once initialized from a box"
                            .into(),
                    )
                    })?
                    .map(|v| v * self.length_scale)
            } else {
                [
                    self.bounds[1] - self.bounds[0],
                    self.bounds[3] - self.bounds[2],
                    self.bounds[5] - self.bounds[4],
                ]
            };

            let mut frame_sg_sum = if self.interface_enabled() {
                vec![0.0; self.sg_grid_sum.len()]
            } else {
                Vec::new()
            };
            let mut frame_counts = if self.interface_enabled() {
                vec![0u64; self.counts.len()]
            } else {
                Vec::new()
            };

            for &atom_u32 in self.selection.indices.iter() {
                let atom_idx = atom_u32 as usize;
                let (sg, sk) = tetrahedral_order_for_atom(
                    frame_coords,
                    atom_idx,
                    &self.selection.indices,
                    chunk.box_[frame],
                    self.length_scale,
                )?;
                let atom = frame_coords[atom_idx];
                let cell = if self.used_box {
                    [
                        nearest_periodic_index(
                            atom[0] as f64 * self.length_scale,
                            lengths[0],
                            self.dims[0],
                        ),
                        nearest_periodic_index(
                            atom[1] as f64 * self.length_scale,
                            lengths[1],
                            self.dims[1],
                        ),
                        nearest_periodic_index(
                            atom[2] as f64 * self.length_scale,
                            lengths[2],
                            self.dims[2],
                        ),
                    ]
                } else {
                    [
                        nearest_bounded_index(
                            atom[0] as f64 * self.length_scale,
                            self.origins[0],
                            self.bin,
                            self.dims[0],
                        ),
                        nearest_bounded_index(
                            atom[1] as f64 * self.length_scale,
                            self.origins[1],
                            self.bin,
                            self.dims[1],
                        ),
                        nearest_bounded_index(
                            atom[2] as f64 * self.length_scale,
                            self.origins[2],
                            self.bin,
                            self.dims[2],
                        ),
                    ]
                };
                let flat = flatten3(cell[0], cell[1], cell[2], self.dims);
                self.sg_grid_sum[flat] += sg;
                self.sk_grid_sum[flat] += sk;
                self.counts[flat] = self.counts[flat].saturating_add(1);
                self.sg_sum_total += sg;
                self.sk_sum_total += sk;
                self.sample_count = self.sample_count.saturating_add(1);
                if self.interface_enabled() {
                    frame_sg_sum[flat] += sg;
                    frame_counts[flat] = frame_counts[flat].saturating_add(1);
                }
            }

            if self.interface_enabled() {
                for i in 0..self.block_sg_mean.len() {
                    if frame_counts[i] > 0 {
                        self.block_sg_mean[i] +=
                            frame_sg_sum[i] / frame_counts[i] as f64 / self.tblock as f64;
                    }
                }
                self.block_frames += 1;
                if self.block_frames == self.tblock {
                    let threshold = 0.5 * (self.sgang1.unwrap() + self.sgang2.unwrap());
                    let (lower, upper) = extract_interfaces(
                        &self.block_sg_mean,
                        self.dims,
                        self.axis,
                        threshold,
                        self.bin,
                        self.origins,
                    );
                    self.interface_lower.extend(lower);
                    self.interface_upper.extend(upper);
                    self.interface_blocks += 1;
                    self.block_sg_mean.fill(0.0);
                    self.block_frames = 0;
                }
            }

            self.n_frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let plane_axes = self.plane_axes();
        let interface_rows = self.dims[plane_axes[0]];
        let interface_cols = self.dims[plane_axes[1]];
        if self.n_frames == 0 || self.sample_count == 0 || self.sg_grid_sum.is_empty() {
            return Ok(PlanOutput::HydOrder(HydOrderOutput {
                sg_mean: 0.0,
                sk_mean: 0.0,
                dims: [0, 0, 0],
                sg_grid: Vec::new(),
                sk_grid: Vec::new(),
                counts: Vec::new(),
                bounds: [0.0; 6],
                bin_width: self.bin as f32,
                axis: self.axis,
                plane_axes,
                n_frames: self.n_frames,
                used_box: self.used_box,
                length_scale: self.length_scale as f32,
                interface_lower: Vec::new(),
                interface_upper: Vec::new(),
                interface_blocks: 0,
                interface_rows: 0,
                interface_cols: 0,
                interface_threshold: self
                    .sgang1
                    .zip(self.sgang2)
                    .map(|(a, b)| (0.5 * (a + b)) as f32),
                block_size: self.tblock,
            }));
        }

        let mut sg_grid = vec![0.0f32; self.sg_grid_sum.len()];
        let mut sk_grid = vec![0.0f32; self.sk_grid_sum.len()];
        for i in 0..self.sg_grid_sum.len() {
            let count = self.counts[i] as f64;
            if count <= 0.0 {
                continue;
            }
            sg_grid[i] = (self.sg_grid_sum[i] / count) as f32;
            sk_grid[i] = (self.sk_grid_sum[i] / count) as f32;
        }

        Ok(PlanOutput::HydOrder(HydOrderOutput {
            sg_mean: (self.sg_sum_total / self.sample_count as f64) as f32,
            sk_mean: (self.sk_sum_total / self.sample_count as f64) as f32,
            dims: self.dims,
            sg_grid,
            sk_grid,
            counts: std::mem::take(&mut self.counts),
            bounds: self.bounds.map(|v| v as f32),
            bin_width: self.bin as f32,
            axis: self.axis,
            plane_axes,
            n_frames: self.n_frames,
            used_box: self.used_box,
            length_scale: self.length_scale as f32,
            interface_lower: std::mem::take(&mut self.interface_lower),
            interface_upper: std::mem::take(&mut self.interface_upper),
            interface_blocks: self.interface_blocks,
            interface_rows,
            interface_cols,
            interface_threshold: self
                .sgang1
                .zip(self.sgang2)
                .map(|(a, b)| (0.5 * (a + b)) as f32),
            block_size: self.tblock,
        }))
    }
}

fn tetrahedral_order_for_atom(
    frame_coords: &[[f32; 4]],
    atom_idx: usize,
    selection: &[u32],
    box_: Box3,
    length_scale: f64,
) -> TrajResult<(f64, f64)> {
    let mut best_d2 = [f64::INFINITY; 4];
    let mut best_vecs = [[0.0f64; 3]; 4];
    for &other_u32 in selection.iter() {
        let other_idx = other_u32 as usize;
        if other_idx == atom_idx {
            continue;
        }
        let delta = minimum_image_between_atoms(
            frame_coords[other_idx],
            frame_coords[atom_idx],
            box_,
            length_scale,
        )?;
        let d2 = dot(delta, delta);
        insert_neighbor(d2, delta, &mut best_d2, &mut best_vecs);
    }
    if best_d2[3].is_infinite() {
        return Err(TrajError::Mismatch(
            "hydorder requires 4 neighbors for every selected atom".into(),
        ));
    }

    let mut rmean = 0.0;
    let mut norms = [0.0f64; 4];
    for i in 0..4 {
        norms[i] = best_d2[i].sqrt();
        rmean += norms[i];
    }
    rmean /= 4.0;

    let mut sg = 0.0;
    for i in 0..3 {
        for j in (i + 1)..4 {
            let ui = normalize(best_vecs[i]).unwrap_or([0.0, 0.0, 0.0]);
            let uj = normalize(best_vecs[j]).unwrap_or([0.0, 0.0, 0.0]);
            let value = dot(ui, uj) + (1.0 / 3.0);
            sg += value * value;
        }
    }
    sg = 3.0 * sg / 32.0;

    let mut sk = 0.0;
    if rmean > 0.0 {
        let denom = 12.0 * rmean * rmean;
        for norm in norms {
            let delta = rmean - norm;
            sk += (delta * delta) / denom;
        }
    }

    Ok((sg, sk))
}

fn insert_neighbor(
    d2: f64,
    delta: [f64; 3],
    best_d2: &mut [f64; 4],
    best_vecs: &mut [[f64; 3]; 4],
) {
    for slot in 0..4 {
        if d2 < best_d2[slot] {
            for shift in (slot + 1..4).rev() {
                best_d2[shift] = best_d2[shift - 1];
                best_vecs[shift] = best_vecs[shift - 1];
            }
            best_d2[slot] = d2;
            best_vecs[slot] = delta;
            break;
        }
    }
}

fn nearest_periodic_index(value: f64, length: f64, bins: usize) -> usize {
    if bins <= 1 || !length.is_finite() || length <= 0.0 {
        return 0;
    }
    let frac = value.rem_euclid(length) / length;
    let idx = (frac * bins as f64 - 0.5).round() as isize;
    idx.rem_euclid(bins as isize) as usize
}

fn nearest_bounded_index(value: f64, origin: f64, bin: f64, bins: usize) -> usize {
    if bins <= 1 {
        return 0;
    }
    let idx = ((value - origin) / bin - 0.5).round() as isize;
    idx.clamp(0, bins as isize - 1) as usize
}

fn flatten3(ix: usize, iy: usize, iz: usize, dims: [usize; 3]) -> usize {
    (ix * dims[1] + iy) * dims[2] + iz
}

fn extract_interfaces(
    block_sg: &[f64],
    dims: [usize; 3],
    axis: usize,
    threshold: f64,
    bin: f64,
    origins: [f64; 3],
) -> (Vec<f32>, Vec<f32>) {
    let plane_axes = match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    };
    let rows = dims[plane_axes[0]];
    let cols = dims[plane_axes[1]];
    let normal_bins = dims[axis];
    let split = (normal_bins / 2).max(1);
    let mut lower = Vec::with_capacity(rows * cols);
    let mut upper = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        for col in 0..cols {
            let mut line = Vec::with_capacity(normal_bins);
            for normal in 0..normal_bins {
                let mut coords = [0usize; 3];
                coords[axis] = normal;
                coords[plane_axes[0]] = row;
                coords[plane_axes[1]] = col;
                line.push(block_sg[flatten3(coords[0], coords[1], coords[2], dims)]);
            }
            let lower_bin = threshold_bin(&line, 0, split, threshold, false);
            let upper_bin = threshold_bin(&line, split, normal_bins, threshold, true);
            lower.push((origins[axis] + (lower_bin as f64 + 0.5) * bin) as f32);
            upper.push((origins[axis] + (upper_bin as f64 + 0.5) * bin) as f32);
        }
    }

    (lower, upper)
}

fn threshold_bin(
    line: &[f64],
    start: usize,
    end: usize,
    threshold: f64,
    descending: bool,
) -> usize {
    let mut pairs: Vec<(f64, usize)> = (start..end).map(|i| (line[i], i)).collect();
    if pairs.is_empty() {
        return start.saturating_sub(1);
    }
    if descending {
        pairs.sort_by(|a, b| cmp_f64_desc(a.0, b.0));
        let mut chosen = pairs[0].1;
        for (value, idx) in pairs {
            if value >= threshold {
                chosen = idx;
            } else {
                break;
            }
        }
        chosen
    } else {
        pairs.sort_by(|a, b| cmp_f64_asc(a.0, b.0));
        let mut chosen = pairs[0].1;
        for (value, idx) in pairs {
            if value <= threshold {
                chosen = idx;
            } else {
                break;
            }
        }
        chosen
    }
}

fn cmp_f64_asc(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

fn cmp_f64_desc(a: f64, b: f64) -> Ordering {
    b.partial_cmp(&a).unwrap_or(Ordering::Equal)
}

#[cfg(test)]
mod tests {
    use super::extract_interfaces;

    #[test]
    fn extract_interfaces_finds_threshold_bins_in_both_halves() {
        let grid = vec![0.10, 0.30, 0.80, 0.90];
        let (lower, upper) = extract_interfaces(&grid, [1, 1, 4], 2, 0.5, 1.0, [0.0, 0.0, 0.0]);
        assert_eq!(lower, vec![1.5]);
        assert_eq!(upper, vec![2.5]);
    }
}
