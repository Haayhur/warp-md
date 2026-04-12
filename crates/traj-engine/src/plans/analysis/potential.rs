use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput, PotentialOutput};

const EPS0: f64 = 8.854_187_812_8e-12;
const ELEMENTARY_CHARGE: f64 = 1.602_176_634e-19;

pub struct PotentialPlan {
    selection: Selection,
    center_selection: Option<Selection>,
    axis: usize,
    bin: f64,
    n_slices: Option<usize>,
    charges: Vec<f64>,
    length_scale: f64,
    correct: bool,
    symmetrize: bool,
    discard_start: usize,
    discard_end: usize,
    bins: usize,
    bounds: [f64; 2],
    density_sum: Vec<f64>,
    frames: usize,
    extent_sum: f64,
    cross_section_area: f64,
    used_box: bool,
    centered: bool,
    initialized: bool,
}

impl PotentialPlan {
    pub fn new(selection: Selection, axis: usize, bin: f64, charges: Vec<f64>) -> Self {
        Self {
            selection,
            center_selection: None,
            axis,
            bin,
            n_slices: None,
            charges,
            length_scale: 1.0,
            correct: false,
            symmetrize: false,
            discard_start: 0,
            discard_end: 0,
            bins: 0,
            bounds: [0.0, 0.0],
            density_sum: Vec::new(),
            frames: 0,
            extent_sum: 0.0,
            cross_section_area: 0.0,
            used_box: false,
            centered: false,
            initialized: false,
        }
    }

    pub fn with_center_selection(mut self, selection: Selection) -> Self {
        self.center_selection = Some(selection);
        self
    }

    pub fn with_n_slices(mut self, n_slices: Option<usize>) -> Self {
        self.n_slices = n_slices;
        self
    }

    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    pub fn with_correct(mut self, correct: bool) -> Self {
        self.correct = correct;
        self
    }

    pub fn with_symmetrize(mut self, symmetrize: bool) -> Self {
        self.symmetrize = symmetrize;
        self
    }

    pub fn with_integration_discard(mut self, discard_start: usize, discard_end: usize) -> Self {
        self.discard_start = discard_start;
        self.discard_end = discard_end;
        self
    }

    fn initialize_from_frame(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        frame: usize,
    ) -> TrajResult<()> {
        let extent = if let Some(lengths) = orthorhombic_lengths(chunk.box_[frame]) {
            self.used_box = true;
            self.cross_section_area = lengths[other_axis(self.axis, 0)]
                * self.length_scale
                * lengths[other_axis(self.axis, 1)]
                * self.length_scale;
            lengths[self.axis] * self.length_scale
        } else {
            self.used_box = false;
            self.initialize_fallback_bounds(chunk, system, frame)?
        };
        self.bins = resolve_bins(self.n_slices, self.bin, extent)?;
        if self.discard_start.saturating_add(self.discard_end) >= self.bins {
            return Err(TrajError::Parse(
                "potential discard_start + discard_end must be smaller than the number of slices"
                    .into(),
            ));
        }
        if self.used_box {
            self.bounds = if self.centered {
                [-0.5 * extent, 0.5 * extent]
            } else {
                [0.0, extent]
            };
        }
        self.density_sum = vec![0.0; self.bins];
        self.initialized = true;
        Ok(())
    }

    fn initialize_fallback_bounds(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        frame: usize,
    ) -> TrajResult<f64> {
        let base = frame * chunk.n_atoms;
        let center = self.center_coordinate(chunk, system, frame, 0.0);
        let ax0 = other_axis(self.axis, 0);
        let ax1 = other_axis(self.axis, 1);
        let mut min_axis = f64::INFINITY;
        let mut max_axis = f64::NEG_INFINITY;
        let mut max_abs_axis = 0.0f64;
        let mut min0 = f64::INFINITY;
        let mut max0 = f64::NEG_INFINITY;
        let mut min1 = f64::INFINITY;
        let mut max1 = f64::NEG_INFINITY;
        for &atom_u32 in self.selection.indices.iter() {
            let atom = &chunk.coords[base + atom_u32 as usize];
            let axis_value = atom[self.axis] as f64 * self.length_scale - center;
            min_axis = min_axis.min(axis_value);
            max_axis = max_axis.max(axis_value);
            max_abs_axis = max_abs_axis.max(axis_value.abs());
            let v0 = atom[ax0] as f64 * self.length_scale;
            let v1 = atom[ax1] as f64 * self.length_scale;
            min0 = min0.min(v0);
            max0 = max0.max(v0);
            min1 = min1.min(v1);
            max1 = max1.max(v1);
        }
        if !min0.is_finite()
            || !max0.is_finite()
            || !min1.is_finite()
            || !max1.is_finite()
            || (!min_axis.is_finite() && !self.centered)
        {
            return Err(TrajError::Mismatch(
                "potential could not determine profile bounds from the selected atoms".into(),
            ));
        }
        let extent0 = extent_or_bin(min0, max0, self.bin);
        let extent1 = extent_or_bin(min1, max1, self.bin);
        self.cross_section_area = extent0 * extent1;
        if self.centered {
            let extent = (2.0 * max_abs_axis).max(self.bin);
            self.bounds = [-0.5 * extent, 0.5 * extent];
            Ok(extent)
        } else {
            let upper = if max_axis <= min_axis {
                min_axis + self.bin
            } else {
                max_axis
            };
            self.bounds = [min_axis, upper];
            Ok(upper - min_axis)
        }
    }

    fn center_coordinate(
        &self,
        chunk: &FrameChunk,
        system: &System,
        frame: usize,
        extent: f64,
    ) -> f64 {
        let Some(selection) = &self.center_selection else {
            return 0.0;
        };
        if selection.indices.is_empty() {
            return 0.0;
        }
        let base = frame * chunk.n_atoms;
        let masses = &system.atoms.mass;
        let mut weighted_sum = 0.0f64;
        let mut mass_sum = 0.0f64;
        for &atom_u32 in selection.indices.iter() {
            let atom_idx = atom_u32 as usize;
            let mass = masses.get(atom_idx).copied().unwrap_or(0.0).max(0.0) as f64;
            if mass > 0.0 {
                let value = chunk.coords[base + atom_idx][self.axis] as f64 * self.length_scale;
                weighted_sum += value * mass;
                mass_sum += mass;
            }
        }
        let mut center = if mass_sum > 0.0 {
            weighted_sum / mass_sum
        } else {
            let mut sum = 0.0f64;
            for &atom_u32 in selection.indices.iter() {
                sum += chunk.coords[base + atom_u32 as usize][self.axis] as f64 * self.length_scale;
            }
            sum / selection.indices.len() as f64
        };
        if self.used_box && extent > 0.0 {
            center = center.rem_euclid(extent);
        }
        center
    }
}

impl Plan for PotentialPlan {
    fn name(&self) -> &'static str {
        "potential"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.axis > 2 {
            return Err(TrajError::Parse("potential axis must be 0, 1, or 2".into()));
        }
        if !self.bin.is_finite() || self.bin <= 0.0 {
            return Err(TrajError::Parse(
                "potential bin must be finite and > 0".into(),
            ));
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(TrajError::Parse(
                "potential length_scale must be finite and > 0".into(),
            ));
        }
        if self.charges.len() != system.n_atoms() {
            return Err(TrajError::Mismatch(
                "potential charges length must match system atom count".into(),
            ));
        }
        if self.charges.iter().any(|q| !q.is_finite()) {
            return Err(TrajError::Parse(
                "potential charges must contain only finite values".into(),
            ));
        }
        if let Some(n_slices) = self.n_slices {
            if n_slices == 0 {
                return Err(TrajError::Parse("potential n_slices must be > 0".into()));
            }
            if self.discard_start.saturating_add(self.discard_end) >= n_slices {
                return Err(TrajError::Parse(
                    "potential discard_start + discard_end must be smaller than n_slices".into(),
                ));
            }
        }
        if self.symmetrize && self.center_selection.is_none() {
            self.center_selection = Some(self.selection.clone());
        }
        self.centered = self.center_selection.is_some();
        self.bins = 0;
        self.bounds = [0.0, 0.0];
        self.density_sum.clear();
        self.frames = 0;
        self.extent_sum = 0.0;
        self.cross_section_area = 0.0;
        self.used_box = false;
        self.initialized = false;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.selection.indices.is_empty() {
            self.frames += chunk.n_frames;
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            if !self.initialized {
                self.initialize_from_frame(chunk, system, frame)?;
            }
            if self.bins == 0 {
                self.frames += 1;
                continue;
            }
            let extent = if self.used_box {
                let lengths = orthorhombic_lengths(chunk.box_[frame]).ok_or_else(|| {
                    TrajError::Mismatch(
                        "potential currently requires orthorhombic boxes when box metadata is present"
                            .into(),
                    )
                })?;
                lengths[self.axis] * self.length_scale
            } else {
                self.bounds[1] - self.bounds[0]
            };
            if !extent.is_finite() || extent <= 0.0 {
                return Err(TrajError::Mismatch(
                    "potential requires positive axis extent".into(),
                ));
            }
            let cross_section_area = if self.used_box {
                let lengths = orthorhombic_lengths(chunk.box_[frame]).ok_or_else(|| {
                    TrajError::Mismatch(
                        "potential currently requires orthorhombic boxes when box metadata is present"
                            .into(),
                    )
                })?;
                lengths[other_axis(self.axis, 0)]
                    * self.length_scale
                    * lengths[other_axis(self.axis, 1)]
                    * self.length_scale
            } else {
                self.cross_section_area
            };
            if !cross_section_area.is_finite() || cross_section_area <= 0.0 {
                return Err(TrajError::Mismatch(
                    "potential requires positive cross-sectional area".into(),
                ));
            }
            let slice_volume = cross_section_area * extent / self.bins as f64;
            if !slice_volume.is_finite() || slice_volume <= 0.0 {
                return Err(TrajError::Mismatch(
                    "potential requires positive slice volume".into(),
                ));
            }
            let center = self.center_coordinate(chunk, system, frame, extent);
            let base = frame * chunk.n_atoms;
            for &atom_u32 in self.selection.indices.iter() {
                let atom_idx = atom_u32 as usize;
                let charge = self.charges[atom_idx];
                let mut value = chunk.coords[base + atom_idx][self.axis] as f64 * self.length_scale;
                let bin = if self.used_box {
                    if self.centered {
                        value = wrap_centered(value - center, extent);
                        bounded_bin(value, -0.5 * extent, 0.5 * extent, self.bins)
                    } else {
                        periodic_bin(value, extent, self.bins)
                    }
                } else {
                    if self.centered {
                        value -= center;
                    }
                    bounded_bin(value, self.bounds[0], self.bounds[1], self.bins)
                };
                if let Some(bin_idx) = bin {
                    self.density_sum[bin_idx] += charge / slice_volume;
                }
            }
            self.extent_sum += extent;
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.frames == 0 || self.bins == 0 {
            return Ok(PlanOutput::Potential(PotentialOutput {
                coordinate: Vec::new(),
                charge_density: Vec::new(),
                field: Vec::new(),
                potential: Vec::new(),
                axis: self.axis,
                bounds: [0.0, 0.0],
                slice_width: 0.0,
                n_frames: self.frames,
                used_box: self.used_box,
                centered: self.centered,
                symmetrized: self.symmetrize,
                corrected: self.correct,
                length_scale: self.length_scale as f32,
                discard_start: self.discard_start,
                discard_end: self.discard_end,
            }));
        }

        let bounds = if self.used_box {
            let mean_extent = self.extent_sum / self.frames as f64;
            if self.centered {
                [-0.5 * mean_extent, 0.5 * mean_extent]
            } else {
                [0.0, mean_extent]
            }
        } else {
            self.bounds
        };
        let slice_width = (bounds[1] - bounds[0]) / self.bins as f64;
        let mut charge_density: Vec<f64> = self
            .density_sum
            .drain(..)
            .map(|value| value / self.frames as f64)
            .collect();
        if self.correct {
            subtract_mean_nonzero(&mut charge_density);
        }
        let mut field = integrate_profile(
            &charge_density,
            slice_width,
            self.discard_start,
            self.discard_end,
        );
        if self.correct {
            subtract_mean_masked(&charge_density, &mut field);
        }
        let mut potential =
            integrate_profile(&field, slice_width, self.discard_start, self.discard_end);
        let field_factor = ELEMENTARY_CHARGE * 1.0e9 / EPS0;
        let potential_factor = -ELEMENTARY_CHARGE * 1.0e9 / EPS0;
        for value in &mut field {
            *value *= field_factor;
        }
        for value in &mut potential {
            *value *= potential_factor;
        }
        if self.symmetrize {
            symmetrize_in_place(&mut charge_density);
            symmetrize_in_place(&mut field);
            symmetrize_in_place(&mut potential);
        }

        Ok(PlanOutput::Potential(PotentialOutput {
            coordinate: build_centers(bounds[0], bounds[1], self.bins),
            charge_density: charge_density
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            field: field.into_iter().map(|value| value as f32).collect(),
            potential: potential.into_iter().map(|value| value as f32).collect(),
            axis: self.axis,
            bounds: [bounds[0] as f32, bounds[1] as f32],
            slice_width: slice_width as f32,
            n_frames: self.frames,
            used_box: self.used_box,
            centered: self.centered,
            symmetrized: self.symmetrize,
            corrected: self.correct,
            length_scale: self.length_scale as f32,
            discard_start: self.discard_start,
            discard_end: self.discard_end,
        }))
    }
}

fn other_axis(axis: usize, which: usize) -> usize {
    match (axis, which) {
        (0, 0) => 1,
        (0, 1) => 2,
        (1, 0) => 0,
        (1, 1) => 2,
        (2, 0) => 0,
        (2, 1) => 1,
        _ => 0,
    }
}

fn orthorhombic_lengths(box_: Box3) -> Option<[f64; 3]> {
    match box_ {
        Box3::Orthorhombic { lx, ly, lz } => Some([lx as f64, ly as f64, lz as f64]),
        _ => None,
    }
}

fn resolve_bins(explicit: Option<usize>, bin: f64, extent: f64) -> TrajResult<usize> {
    if let Some(value) = explicit {
        return Ok(value.max(1));
    }
    if !extent.is_finite() || extent <= 0.0 {
        return Err(TrajError::Mismatch(
            "potential requires positive spatial extent".into(),
        ));
    }
    Ok(((extent / bin).round() as usize).max(1))
}

fn extent_or_bin(min: f64, max: f64, bin: f64) -> f64 {
    if !min.is_finite() || !max.is_finite() || max <= min {
        bin
    } else {
        max - min
    }
}

fn build_centers(min: f64, max: f64, bins: usize) -> Vec<f32> {
    if bins == 0 {
        return Vec::new();
    }
    let step = (max - min) / bins as f64;
    let mut out = Vec::with_capacity(bins);
    for idx in 0..bins {
        out.push((min + (idx as f64 + 0.5) * step) as f32);
    }
    out
}

fn periodic_bin(value: f64, extent: f64, bins: usize) -> Option<usize> {
    if !value.is_finite() || !extent.is_finite() || extent <= 0.0 || bins == 0 {
        return None;
    }
    let wrapped = value.rem_euclid(extent);
    let mut index = ((wrapped / extent) * bins as f64).floor() as usize;
    if index >= bins {
        index = bins - 1;
    }
    Some(index)
}

fn wrap_centered(value: f64, extent: f64) -> f64 {
    let shifted = (value + 0.5 * extent).rem_euclid(extent);
    shifted - 0.5 * extent
}

fn bounded_bin(value: f64, min: f64, max: f64, bins: usize) -> Option<usize> {
    if !value.is_finite() || !min.is_finite() || !max.is_finite() || max <= min || bins == 0 {
        return None;
    }
    if value < min || value > max {
        return None;
    }
    if value == max {
        return Some(bins - 1);
    }
    let frac = (value - min) / (max - min);
    let mut index = (frac * bins as f64).floor() as usize;
    if index >= bins {
        index = bins - 1;
    }
    Some(index)
}

fn subtract_mean_nonzero(values: &mut [f64]) {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for &value in values.iter() {
        if value.abs() >= f64::MIN_POSITIVE {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        return;
    }
    let mean = sum / count as f64;
    for value in values.iter_mut() {
        if value.abs() >= f64::MIN_POSITIVE {
            *value -= mean;
        }
    }
}

fn subtract_mean_masked(mask: &[f64], values: &mut [f64]) {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for (&mask_value, &value) in mask.iter().zip(values.iter()) {
        if mask_value.abs() >= f64::MIN_POSITIVE {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        return;
    }
    let mean = sum / count as f64;
    for (&mask_value, value) in mask.iter().zip(values.iter_mut()) {
        if mask_value.abs() >= f64::MIN_POSITIVE {
            *value -= mean;
        }
    }
}

fn integrate_profile(
    data: &[f64],
    slice_width: f64,
    discard_start: usize,
    discard_end: usize,
) -> Vec<f64> {
    let n = data.len();
    let mut out = vec![0.0f64; n];
    if n == 0 {
        return out;
    }
    let end = n.saturating_sub(discard_end);
    for slice in discard_start..end {
        let mut sum = 0.0f64;
        for i in discard_start..slice {
            sum += slice_width * (data[i] + 0.5 * (data[i + 1] - data[i]));
        }
        out[slice] = sum;
    }
    out
}

fn symmetrize_in_place(values: &mut [f64]) {
    let len = values.len();
    for i in 0..(len / 2) {
        let j = len - i - 1;
        let mean = 0.5 * (values[i] + values[j]);
        values[i] = mean;
        values[j] = mean;
    }
}
