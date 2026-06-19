use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::pbc_math::orthorhombic_lengths;
use traj_core::selection::Selection;
use traj_core::system::System;

use super::binning::{bounded_bin, build_centers, periodic_bin, resolve_bins};
use crate::executor::{DensityMapOutput, Device, Plan, PlanOutput};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DensityMapUnit {
    NumberDensity,
    AreaDensity,
    Count,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearDensityWeight {
    Number,
    Mass,
    Charge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearDensityNorm {
    Count,
    Density,
}

impl DensityMapUnit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NumberDensity => "nm-3",
            Self::AreaDensity => "nm-2",
            Self::Count => "count",
        }
    }
}

pub struct DensityMapPlan {
    selection: Selection,
    average_axis: usize,
    bin: f64,
    n1: Option<usize>,
    n2: Option<usize>,
    xmin: Option<f64>,
    xmax: Option<f64>,
    unit: DensityMapUnit,
    length_scale: f64,
    plane_axes: [usize; 2],
    plane_bounds: [f64; 4],
    average_span: f64,
    bins: [usize; 2],
    accum: Vec<f64>,
    frames: usize,
    plane_extent_sum: [f64; 2],
    used_box: bool,
    initialized: bool,
}

pub struct LinearDensityPlan {
    selection: Selection,
    axis: usize,
    bin: f64,
    range: Option<[f64; 2]>,
    cross_section_area: Option<f64>,
    weight: LinearDensityWeight,
    norm: LinearDensityNorm,
    charges: Vec<f64>,
    length_scale: f64,
    bounds: [f64; 2],
    bins: usize,
    accum_profile: Vec<f64>,
    accum_weight: Vec<f64>,
    axis_extent_sum: f64,
    frames: usize,
    used_box: bool,
    initialized: bool,
}

impl DensityMapPlan {
    pub fn new(selection: Selection, average_axis: usize, bin: f64, unit: DensityMapUnit) -> Self {
        Self {
            selection,
            average_axis,
            bin,
            n1: None,
            n2: None,
            xmin: None,
            xmax: None,
            unit,
            length_scale: 1.0,
            plane_axes: plane_axes_for_average(average_axis),
            plane_bounds: [0.0; 4],
            average_span: 0.0,
            bins: [0, 0],
            accum: Vec::new(),
            frames: 0,
            plane_extent_sum: [0.0; 2],
            used_box: false,
            initialized: false,
        }
    }

    pub fn with_bins(mut self, n1: Option<usize>, n2: Option<usize>) -> Self {
        self.n1 = n1;
        self.n2 = n2;
        self
    }

    pub fn with_average_window(mut self, xmin: Option<f64>, xmax: Option<f64>) -> Self {
        self.xmin = xmin;
        self.xmax = xmax;
        self
    }

    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }
}

impl LinearDensityPlan {
    pub fn new(
        selection: Selection,
        axis: usize,
        bin: f64,
        weight: LinearDensityWeight,
        norm: LinearDensityNorm,
    ) -> Self {
        Self {
            selection,
            axis,
            bin,
            range: None,
            cross_section_area: None,
            weight,
            norm,
            charges: Vec::new(),
            length_scale: 1.0,
            bounds: [0.0, 0.0],
            bins: 0,
            accum_profile: Vec::new(),
            accum_weight: Vec::new(),
            axis_extent_sum: 0.0,
            frames: 0,
            used_box: false,
            initialized: false,
        }
    }

    pub fn with_range(mut self, range: Option<[f64; 2]>) -> Self {
        self.range = range;
        self
    }

    pub fn with_cross_section_area(mut self, area: Option<f64>) -> Self {
        self.cross_section_area = area;
        self
    }

    pub fn with_charges(mut self, charges: Vec<f64>) -> Self {
        self.charges = charges;
        self
    }

    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }
}

impl Plan for DensityMapPlan {
    fn name(&self) -> &'static str {
        "densmap"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        if self.average_axis > 2 {
            return Err(TrajError::Parse(
                "densmap average axis must be 0, 1, or 2".into(),
            ));
        }
        if !self.bin.is_finite() || self.bin <= 0.0 {
            return Err(TrajError::Parse(
                "densmap bin must be finite and > 0".into(),
            ));
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(TrajError::Parse(
                "densmap length_scale must be finite and > 0".into(),
            ));
        }
        if let Some(n1) = self.n1 {
            if n1 == 0 {
                return Err(TrajError::Parse("densmap n1 must be > 0".into()));
            }
        }
        if let Some(n2) = self.n2 {
            if n2 == 0 {
                return Err(TrajError::Parse("densmap n2 must be > 0".into()));
            }
        }
        if let (Some(xmin), Some(xmax)) = (self.xmin, self.xmax) {
            if !xmin.is_finite() || !xmax.is_finite() || xmax < xmin {
                return Err(TrajError::Parse(
                    "densmap requires finite xmin/xmax with xmax >= xmin".into(),
                ));
            }
        }
        self.plane_axes = plane_axes_for_average(self.average_axis);
        self.plane_bounds = [0.0; 4];
        self.average_span = 0.0;
        self.bins = [0, 0];
        self.accum.clear();
        self.frames = 0;
        self.plane_extent_sum = [0.0; 2];
        self.used_box = false;
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
            self.frames += chunk.n_frames;
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            if !self.initialized {
                self.initialize_from_frame(chunk, frame)?;
            }
            if self.bins[0] == 0 || self.bins[1] == 0 {
                self.frames += 1;
                continue;
            }
            let (plane_extent, average_extent) = if self.used_box {
                let lengths = orthorhombic_lengths(&chunk.box_[frame]).ok_or_else(|| {
                    TrajError::Mismatch(
                        "densmap currently requires orthorhombic boxes when box metadata is present"
                            .into(),
                    )
                })?;
                let plane_extent = [
                    lengths[self.plane_axes[0]] * self.length_scale,
                    lengths[self.plane_axes[1]] * self.length_scale,
                ];
                let average_extent = lengths[self.average_axis] * self.length_scale;
                self.plane_extent_sum[0] += plane_extent[0];
                self.plane_extent_sum[1] += plane_extent[1];
                (plane_extent, average_extent)
            } else {
                (
                    [
                        self.plane_bounds[1] - self.plane_bounds[0],
                        self.plane_bounds[3] - self.plane_bounds[2],
                    ],
                    self.average_span,
                )
            };
            let weight = density_weight(self.unit, self.bins, plane_extent, average_extent)?;
            let base = frame * chunk.n_atoms;
            for &atom_u32 in self.selection.indices.iter() {
                let atom = &chunk.coords[base + atom_u32 as usize];
                let average_value = atom[self.average_axis] as f64 * self.length_scale;
                if let Some(xmin) = self.xmin {
                    if average_value < xmin {
                        continue;
                    }
                }
                if let Some(xmax) = self.xmax {
                    if average_value > xmax {
                        continue;
                    }
                }
                let idx1 = if self.used_box {
                    periodic_bin(
                        atom[self.plane_axes[0]] as f64 * self.length_scale,
                        plane_extent[0],
                        self.bins[0],
                    )
                } else {
                    bounded_bin(
                        atom[self.plane_axes[0]] as f64 * self.length_scale,
                        self.plane_bounds[0],
                        self.plane_bounds[1],
                        self.bins[0],
                    )
                };
                let idx2 = if self.used_box {
                    periodic_bin(
                        atom[self.plane_axes[1]] as f64 * self.length_scale,
                        plane_extent[1],
                        self.bins[1],
                    )
                } else {
                    bounded_bin(
                        atom[self.plane_axes[1]] as f64 * self.length_scale,
                        self.plane_bounds[2],
                        self.plane_bounds[3],
                        self.bins[1],
                    )
                };
                if let (Some(i), Some(j)) = (idx1, idx2) {
                    self.accum[i * self.bins[1] + j] += weight;
                }
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.frames == 0 || self.bins[0] == 0 || self.bins[1] == 0 {
            return Ok(PlanOutput::DensityMap(DensityMapOutput {
                axis1: Vec::new(),
                axis2: Vec::new(),
                matrix: Vec::new(),
                rows: 0,
                cols: 0,
                plane_axes: self.plane_axes,
                average_axis: self.average_axis,
                unit: self.unit.as_str().to_string(),
                n_frames: self.frames,
                bounds: [0.0; 4],
                bin_width: [0.0; 2],
                used_box: self.used_box,
                length_scale: self.length_scale as f32,
            }));
        }

        let (bounds, bin_width) = if self.used_box {
            let mean_extent = [
                self.plane_extent_sum[0] / self.frames as f64,
                self.plane_extent_sum[1] / self.frames as f64,
            ];
            (
                [0.0, mean_extent[0], 0.0, mean_extent[1]],
                [
                    (mean_extent[0] / self.bins[0] as f64) as f32,
                    (mean_extent[1] / self.bins[1] as f64) as f32,
                ],
            )
        } else {
            let extent1 = self.plane_bounds[1] - self.plane_bounds[0];
            let extent2 = self.plane_bounds[3] - self.plane_bounds[2];
            (
                self.plane_bounds,
                [
                    (extent1 / self.bins[0] as f64) as f32,
                    (extent2 / self.bins[1] as f64) as f32,
                ],
            )
        };
        let axis1 = build_centers(bounds[0], bounds[1], self.bins[0]);
        let axis2 = build_centers(bounds[2], bounds[3], self.bins[1]);
        let inv_frames = 1.0f32 / self.frames as f32;
        let mut matrix = Vec::with_capacity(self.accum.len());
        for value in self.accum.drain(..) {
            matrix.push(value as f32 * inv_frames);
        }
        Ok(PlanOutput::DensityMap(DensityMapOutput {
            axis1,
            axis2,
            matrix,
            rows: self.bins[0],
            cols: self.bins[1],
            plane_axes: self.plane_axes,
            average_axis: self.average_axis,
            unit: self.unit.as_str().to_string(),
            n_frames: self.frames,
            bounds: [
                bounds[0] as f32,
                bounds[1] as f32,
                bounds[2] as f32,
                bounds[3] as f32,
            ],
            bin_width,
            used_box: self.used_box,
            length_scale: self.length_scale as f32,
        }))
    }
}

impl Plan for LinearDensityPlan {
    fn name(&self) -> &'static str {
        "lineardensity"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.axis > 2 {
            return Err(TrajError::Parse(
                "lineardensity axis must be 0, 1, or 2".into(),
            ));
        }
        if !self.bin.is_finite() || self.bin <= 0.0 {
            return Err(TrajError::Parse(
                "lineardensity bin must be finite and > 0".into(),
            ));
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(TrajError::Parse(
                "lineardensity length_scale must be finite and > 0".into(),
            ));
        }
        if let Some([lo, hi]) = self.range {
            if !lo.is_finite() || !hi.is_finite() || hi <= lo {
                return Err(TrajError::Parse(
                    "lineardensity range must be finite with max > min".into(),
                ));
            }
        }
        if let Some(area) = self.cross_section_area {
            if !area.is_finite() || area <= 0.0 {
                return Err(TrajError::Parse(
                    "lineardensity cross_section_area must be finite and > 0".into(),
                ));
            }
        }
        match self.weight {
            LinearDensityWeight::Number => {}
            LinearDensityWeight::Mass => {
                if system.atoms.mass.len() != system.n_atoms() {
                    return Err(TrajError::Mismatch(
                        "lineardensity mass weighting requires per-atom masses".into(),
                    ));
                }
            }
            LinearDensityWeight::Charge => {
                if self.charges.len() != system.n_atoms() {
                    return Err(TrajError::Mismatch(
                        "lineardensity charge weighting requires one charge per atom".into(),
                    ));
                }
            }
        }
        let n_atoms = system.n_atoms();
        for &idx in self.selection.indices.iter() {
            if idx as usize >= n_atoms {
                return Err(TrajError::Mismatch(
                    "lineardensity atom index out of bounds".into(),
                ));
            }
        }
        self.bounds = [0.0, 0.0];
        self.bins = 0;
        self.accum_profile.clear();
        self.accum_weight.clear();
        self.axis_extent_sum = 0.0;
        self.frames = 0;
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
        for frame in 0..chunk.n_frames {
            if !self.initialized {
                self.initialize_from_frame(chunk, frame)?;
            }
            if self.bins == 0 {
                self.frames += 1;
                continue;
            }
            let lengths = orthorhombic_lengths(&chunk.box_[frame]);
            let axis_extent = if self.used_box {
                let lengths = lengths.ok_or_else(|| {
                    TrajError::Mismatch(
                        "lineardensity requires orthorhombic boxes after box initialization".into(),
                    )
                })?;
                lengths[self.axis] * self.length_scale
            } else {
                self.bounds[1] - self.bounds[0]
            };
            let area = self.cross_section_area.or_else(|| {
                lengths.map(|values| cross_section_area(values, self.axis, self.length_scale))
            });
            let density_scale = match self.norm {
                LinearDensityNorm::Count => 1.0,
                LinearDensityNorm::Density => {
                    let area = area.ok_or_else(|| {
                        TrajError::Mismatch(
                            "lineardensity density normalization requires box metadata or cross_section_area"
                                .into(),
                        )
                    })?;
                    1.0 / (area * self.bin)
                }
            };
            self.axis_extent_sum += axis_extent;
            let base = frame * chunk.n_atoms;
            for &atom_u32 in self.selection.indices.iter() {
                let atom = atom_u32 as usize;
                let coord = chunk.coords[base + atom][self.axis] as f64 * self.length_scale;
                let bin = if self.used_box {
                    periodic_bin(coord, axis_extent, self.bins)
                } else {
                    bounded_bin(coord, self.bounds[0], self.bounds[1], self.bins)
                };
                if let Some(idx) = bin {
                    let weight = atom_weight(self.weight, atom, system, &self.charges);
                    self.accum_weight[idx] += weight;
                    self.accum_profile[idx] += weight * density_scale;
                }
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.frames == 0 || self.bins == 0 {
            return Ok(PlanOutput::Matrix {
                data: Vec::new(),
                rows: 0,
                cols: 3,
            });
        }
        let bounds = if self.used_box {
            [0.0, self.axis_extent_sum / self.frames as f64]
        } else {
            self.bounds
        };
        let centers = build_centers(bounds[0], bounds[1], self.bins);
        let inv_frames = 1.0 / self.frames as f64;
        let mut data = Vec::with_capacity(self.bins * 3);
        for (idx, center) in centers.into_iter().enumerate() {
            data.push(center);
            data.push((self.accum_profile[idx] * inv_frames) as f32);
            data.push((self.accum_weight[idx] * inv_frames) as f32);
        }
        Ok(PlanOutput::Matrix {
            data,
            rows: self.bins,
            cols: 3,
        })
    }
}

impl DensityMapPlan {
    fn initialize_from_frame(&mut self, chunk: &FrameChunk, frame: usize) -> TrajResult<()> {
        self.plane_axes = plane_axes_for_average(self.average_axis);
        if self.selection.indices.is_empty() {
            self.initialized = true;
            return Ok(());
        }
        if let Some(lengths) = orthorhombic_lengths(&chunk.box_[frame]) {
            self.used_box = true;
            let extent1 = lengths[self.plane_axes[0]] * self.length_scale;
            let extent2 = lengths[self.plane_axes[1]] * self.length_scale;
            self.plane_bounds = [0.0, extent1, 0.0, extent2];
            self.average_span = lengths[self.average_axis] * self.length_scale;
            self.plane_extent_sum = [0.0, 0.0];
            self.bins = [
                resolve_bins(self.n1, self.bin, extent1, "densmap")?,
                resolve_bins(self.n2, self.bin, extent2, "densmap")?,
            ];
        } else {
            self.used_box = false;
            let base = frame * chunk.n_atoms;
            let mut min1 = f64::INFINITY;
            let mut max1 = f64::NEG_INFINITY;
            let mut min2 = f64::INFINITY;
            let mut max2 = f64::NEG_INFINITY;
            let mut min_avg = f64::INFINITY;
            let mut max_avg = f64::NEG_INFINITY;
            for &atom_u32 in self.selection.indices.iter() {
                let atom = &chunk.coords[base + atom_u32 as usize];
                let a1 = atom[self.plane_axes[0]] as f64 * self.length_scale;
                let a2 = atom[self.plane_axes[1]] as f64 * self.length_scale;
                let av = atom[self.average_axis] as f64 * self.length_scale;
                min1 = min1.min(a1);
                max1 = max1.max(a1);
                min2 = min2.min(a2);
                max2 = max2.max(a2);
                min_avg = min_avg.min(av);
                max_avg = max_avg.max(av);
            }
            if !min1.is_finite()
                || !max1.is_finite()
                || !min2.is_finite()
                || !max2.is_finite()
                || !min_avg.is_finite()
                || !max_avg.is_finite()
            {
                return Err(TrajError::Mismatch(
                    "densmap could not determine bounds from the selected atoms".into(),
                ));
            }
            if max1 <= min1 {
                max1 = min1 + self.bin;
            }
            if max2 <= min2 {
                max2 = min2 + self.bin;
            }
            if max_avg <= min_avg {
                max_avg = min_avg + self.bin;
            }
            self.plane_bounds = [min1, max1, min2, max2];
            self.average_span = max_avg - min_avg;
            self.bins = [
                resolve_bins(self.n1, self.bin, max1 - min1, "densmap")?,
                resolve_bins(self.n2, self.bin, max2 - min2, "densmap")?,
            ];
        }
        self.accum = vec![0.0; self.bins[0] * self.bins[1]];
        self.initialized = true;
        Ok(())
    }
}

impl LinearDensityPlan {
    fn initialize_from_frame(&mut self, chunk: &FrameChunk, frame: usize) -> TrajResult<()> {
        if self.selection.indices.is_empty() {
            self.initialized = true;
            return Ok(());
        }
        if let Some([lo, hi]) = self.range {
            self.bounds = [lo, hi];
            self.used_box = false;
            self.bins = resolve_bins(None, self.bin, hi - lo, "lineardensity")?;
        } else if let Some(lengths) = orthorhombic_lengths(&chunk.box_[frame]) {
            self.used_box = true;
            let extent = lengths[self.axis] * self.length_scale;
            self.bounds = [0.0, extent];
            self.bins = resolve_bins(None, self.bin, extent, "lineardensity")?;
        } else {
            self.used_box = false;
            let base = frame * chunk.n_atoms;
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &atom_u32 in self.selection.indices.iter() {
                let value =
                    chunk.coords[base + atom_u32 as usize][self.axis] as f64 * self.length_scale;
                lo = lo.min(value);
                hi = hi.max(value);
            }
            if !lo.is_finite() || !hi.is_finite() {
                return Err(TrajError::Mismatch(
                    "lineardensity could not determine bounds from selected atoms".into(),
                ));
            }
            if hi <= lo {
                hi = lo + self.bin;
            }
            self.bounds = [lo, hi];
            self.bins = resolve_bins(None, self.bin, hi - lo, "lineardensity")?;
        }
        self.accum_profile = vec![0.0; self.bins];
        self.accum_weight = vec![0.0; self.bins];
        self.initialized = true;
        Ok(())
    }
}

fn plane_axes_for_average(axis: usize) -> [usize; 2] {
    match axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    }
}

fn atom_weight(weight: LinearDensityWeight, atom: usize, system: &System, charges: &[f64]) -> f64 {
    match weight {
        LinearDensityWeight::Number => 1.0,
        LinearDensityWeight::Mass => system.atoms.mass[atom] as f64,
        LinearDensityWeight::Charge => charges[atom],
    }
}

fn cross_section_area(lengths: [f64; 3], axis: usize, length_scale: f64) -> f64 {
    let axes = plane_axes_for_average(axis);
    lengths[axes[0]] * lengths[axes[1]] * length_scale * length_scale
}

fn density_weight(
    unit: DensityMapUnit,
    bins: [usize; 2],
    plane_extent: [f64; 2],
    average_span: f64,
) -> TrajResult<f64> {
    let nxy = (bins[0] * bins[1]) as f64;
    let plane_area = plane_extent[0] * plane_extent[1];
    if !plane_area.is_finite() || plane_area <= 0.0 {
        return Err(TrajError::Mismatch(
            "densmap requires positive planar extent".into(),
        ));
    }
    match unit {
        DensityMapUnit::Count => Ok(1.0),
        DensityMapUnit::AreaDensity => Ok(nxy / plane_area),
        DensityMapUnit::NumberDensity => {
            if !average_span.is_finite() || average_span <= 0.0 {
                return Err(TrajError::Mismatch(
                    "densmap requires positive averaging-axis span".into(),
                ));
            }
            Ok(nxy / (plane_area * average_span))
        }
    }
}
