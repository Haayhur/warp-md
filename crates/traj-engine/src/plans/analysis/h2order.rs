use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::pbc_utils::{apply_pbc, apply_pbc_triclinic, cell_and_inv_from_box};
use traj_core::system::System;

use crate::executor::{Device, H2OrderOutput, Plan, PlanOutput};

const DEBYE_PER_E_NM: f64 = 1.602_177_33 / 3.336e-2;

pub struct H2OrderPlan {
    oxygen_indices: Vec<u32>,
    hydrogen1_indices: Vec<u32>,
    hydrogen2_indices: Vec<u32>,
    charges: Vec<f64>,
    axis: usize,
    bin: f64,
    n_slices: Option<usize>,
    length_scale: f64,
    bins: usize,
    bounds: [f64; 2],
    order_sum: Vec<f64>,
    dipole_sum: Vec<f64>,
    counts: Vec<u64>,
    frames: usize,
    extent_sum: f64,
    used_box: bool,
    initialized: bool,
}

impl H2OrderPlan {
    pub fn new(
        oxygen_indices: Vec<u32>,
        hydrogen1_indices: Vec<u32>,
        hydrogen2_indices: Vec<u32>,
        charges: Vec<f64>,
        axis: usize,
        bin: f64,
    ) -> Self {
        Self {
            oxygen_indices,
            hydrogen1_indices,
            hydrogen2_indices,
            charges,
            axis,
            bin,
            n_slices: None,
            length_scale: 1.0,
            bins: 0,
            bounds: [0.0, 0.0],
            order_sum: Vec::new(),
            dipole_sum: Vec::new(),
            counts: Vec::new(),
            frames: 0,
            extent_sum: 0.0,
            used_box: false,
            initialized: false,
        }
    }

    pub fn with_n_slices(mut self, n_slices: Option<usize>) -> Self {
        self.n_slices = n_slices;
        self
    }

    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    fn initialize_from_frame(&mut self, chunk: &FrameChunk, frame: usize) -> TrajResult<()> {
        let extent = if let Some(lengths) = box_lengths(chunk.box_[frame]) {
            self.used_box = true;
            self.bounds = [0.0, lengths[self.axis] * self.length_scale];
            lengths[self.axis] * self.length_scale
        } else {
            self.used_box = false;
            let base = frame * chunk.n_atoms;
            let mut min_axis = f64::INFINITY;
            let mut max_axis = f64::NEG_INFINITY;
            for &oxy_u32 in self.oxygen_indices.iter() {
                let oxy = chunk.coords[base + oxy_u32 as usize];
                let value = oxy[self.axis] as f64 * self.length_scale;
                min_axis = min_axis.min(value);
                max_axis = max_axis.max(value);
            }
            if !min_axis.is_finite() || !max_axis.is_finite() {
                return Err(TrajError::Mismatch(
                    "h2order could not determine bounds from oxygen positions".into(),
                ));
            }
            if max_axis <= min_axis {
                max_axis = min_axis + self.bin;
            }
            self.bounds = [min_axis, max_axis];
            max_axis - min_axis
        };
        self.bins = resolve_bins(self.n_slices, self.bin, extent)?;
        self.order_sum = vec![0.0; self.bins];
        self.dipole_sum = vec![0.0; self.bins * 3];
        self.counts = vec![0; self.bins];
        self.initialized = true;
        Ok(())
    }
}

impl Plan for H2OrderPlan {
    fn name(&self) -> &'static str {
        "h2order"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.axis > 2 {
            return Err(TrajError::Parse("h2order axis must be 0, 1, or 2".into()));
        }
        if !self.bin.is_finite() || self.bin <= 0.0 {
            return Err(TrajError::Parse(
                "h2order bin must be finite and > 0".into(),
            ));
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(TrajError::Parse(
                "h2order length_scale must be finite and > 0".into(),
            ));
        }
        let n_waters = self.oxygen_indices.len();
        if self.hydrogen1_indices.len() != n_waters || self.hydrogen2_indices.len() != n_waters {
            return Err(TrajError::Mismatch(
                "h2order oxygen/hydrogen index vectors must have identical length".into(),
            ));
        }
        if self.charges.len() != system.n_atoms() {
            return Err(TrajError::Mismatch(
                "h2order charges length must match system atom count".into(),
            ));
        }
        if self.charges.iter().any(|q| !q.is_finite()) {
            return Err(TrajError::Parse(
                "h2order charges must contain only finite values".into(),
            ));
        }
        let n_atoms = system.n_atoms();
        for &idx in self
            .oxygen_indices
            .iter()
            .chain(self.hydrogen1_indices.iter())
            .chain(self.hydrogen2_indices.iter())
        {
            if idx as usize >= n_atoms {
                return Err(TrajError::Mismatch(
                    "h2order water atom index out of bounds".into(),
                ));
            }
        }
        if let Some(n_slices) = self.n_slices {
            if n_slices == 0 {
                return Err(TrajError::Parse("h2order n_slices must be > 0".into()));
            }
        }
        self.bins = 0;
        self.bounds = [0.0, 0.0];
        self.order_sum.clear();
        self.dipole_sum.clear();
        self.counts.clear();
        self.frames = 0;
        self.extent_sum = 0.0;
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
        if self.oxygen_indices.is_empty() {
            self.frames += chunk.n_frames;
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            if !self.initialized {
                self.initialize_from_frame(chunk, frame)?;
            }
            let extent = if self.used_box {
                let lengths = box_lengths(chunk.box_[frame]).ok_or_else(|| {
                    TrajError::Mismatch(
                        "h2order currently requires orthorhombic or triclinic boxes when box metadata is present"
                            .into(),
                    )
                })?;
                lengths[self.axis] * self.length_scale
            } else {
                self.bounds[1] - self.bounds[0]
            };
            let base = frame * chunk.n_atoms;
            for i in 0..self.oxygen_indices.len() {
                let oxy_idx = self.oxygen_indices[i] as usize;
                let h1_idx = self.hydrogen1_indices[i] as usize;
                let h2_idx = self.hydrogen2_indices[i] as usize;
                let oxygen = chunk.coords[base + oxy_idx];
                let slice = if self.used_box {
                    periodic_bin(
                        oxygen[self.axis] as f64 * self.length_scale,
                        extent,
                        self.bins,
                    )
                } else {
                    bounded_bin(
                        oxygen[self.axis] as f64 * self.length_scale,
                        self.bounds[0],
                        self.bounds[1],
                        self.bins,
                    )
                };
                let Some(slice) = slice else {
                    continue;
                };
                let dipole = water_dipole(
                    &chunk.coords[base..base + chunk.n_atoms],
                    oxy_idx,
                    h1_idx,
                    h2_idx,
                    &self.charges,
                    chunk.box_[frame],
                    self.length_scale,
                )?;
                let norm =
                    (dipole[0] * dipole[0] + dipole[1] * dipole[1] + dipole[2] * dipole[2]).sqrt();
                let cos_theta = if norm > 0.0 {
                    dipole[self.axis] / norm
                } else {
                    0.0
                };
                self.order_sum[slice] += cos_theta;
                let dip_base = slice * 3;
                self.dipole_sum[dip_base] += dipole[0] * DEBYE_PER_E_NM;
                self.dipole_sum[dip_base + 1] += dipole[1] * DEBYE_PER_E_NM;
                self.dipole_sum[dip_base + 2] += dipole[2] * DEBYE_PER_E_NM;
                self.counts[slice] = self.counts[slice].saturating_add(1);
            }
            self.extent_sum += extent;
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.frames == 0 || self.bins == 0 {
            return Ok(PlanOutput::H2Order(H2OrderOutput {
                coordinate: Vec::new(),
                order: Vec::new(),
                dipole: Vec::new(),
                rows: 0,
                cols: 3,
                counts: Vec::new(),
                axis: self.axis,
                bounds: [0.0, 0.0],
                slice_width: 0.0,
                n_frames: self.frames,
                used_box: self.used_box,
                length_scale: self.length_scale as f32,
                dipole_unit: "debye".to_string(),
            }));
        }

        let bounds = if self.used_box {
            [0.0, self.extent_sum / self.frames as f64]
        } else {
            self.bounds
        };
        let slice_width = (bounds[1] - bounds[0]) / self.bins as f64;
        let coordinate = build_centers(bounds[0], bounds[1], self.bins);
        let mut order = vec![0.0f32; self.bins];
        let mut dipole = vec![0.0f32; self.bins * 3];
        for i in 0..self.bins {
            let count = self.counts[i] as f64;
            if count <= 0.0 {
                continue;
            }
            order[i] = (self.order_sum[i] / count) as f32;
            let base = i * 3;
            dipole[base] = (self.dipole_sum[base] / count) as f32;
            dipole[base + 1] = (self.dipole_sum[base + 1] / count) as f32;
            dipole[base + 2] = (self.dipole_sum[base + 2] / count) as f32;
        }

        Ok(PlanOutput::H2Order(H2OrderOutput {
            coordinate,
            order,
            dipole,
            rows: self.bins,
            cols: 3,
            counts: std::mem::take(&mut self.counts),
            axis: self.axis,
            bounds: [bounds[0] as f32, bounds[1] as f32],
            slice_width: slice_width as f32,
            n_frames: self.frames,
            used_box: self.used_box,
            length_scale: self.length_scale as f32,
            dipole_unit: "debye".to_string(),
        }))
    }
}

fn box_lengths(box_: Box3) -> Option<[f64; 3]> {
    match box_ {
        Box3::Orthorhombic { lx, ly, lz } => Some([lx as f64, ly as f64, lz as f64]),
        Box3::Triclinic { m } => Some([m[0] as f64, m[4] as f64, m[8] as f64]),
        Box3::None => None,
    }
}

fn resolve_bins(explicit: Option<usize>, bin: f64, extent: f64) -> TrajResult<usize> {
    if let Some(value) = explicit {
        return Ok(value.max(1));
    }
    if !extent.is_finite() || extent <= 0.0 {
        return Err(TrajError::Mismatch(
            "h2order requires positive spatial extent".into(),
        ));
    }
    Ok(((extent / bin).round() as usize).max(1))
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

fn water_dipole(
    frame_coords: &[[f32; 4]],
    oxygen_idx: usize,
    hydrogen1_idx: usize,
    hydrogen2_idx: usize,
    charges: &[f64],
    box_: Box3,
    length_scale: f64,
) -> TrajResult<[f64; 3]> {
    let anchor = frame_coords[oxygen_idx];
    let anchor_pos = [
        anchor[0] as f64 * length_scale,
        anchor[1] as f64 * length_scale,
        anchor[2] as f64 * length_scale,
    ];
    let maybe_triclinic = match box_ {
        Box3::Triclinic { .. } => Some(cell_and_inv_from_box(box_)?),
        _ => None,
    };
    let mut dip = [0.0f64; 3];
    for &atom_idx in &[oxygen_idx, hydrogen1_idx, hydrogen2_idx] {
        let pos = frame_coords[atom_idx];
        let mut dx = (pos[0] - anchor[0]) as f64 * length_scale;
        let mut dy = (pos[1] - anchor[1]) as f64 * length_scale;
        let mut dz = (pos[2] - anchor[2]) as f64 * length_scale;
        match box_ {
            Box3::Orthorhombic { lx, ly, lz } => {
                apply_pbc(
                    &mut dx,
                    &mut dy,
                    &mut dz,
                    lx as f64 * length_scale,
                    ly as f64 * length_scale,
                    lz as f64 * length_scale,
                );
            }
            Box3::Triclinic { .. } => {
                let (cell, inv) = maybe_triclinic
                    .as_ref()
                    .expect("triclinic box conversion must be available");
                apply_pbc_triclinic(&mut dx, &mut dy, &mut dz, cell, inv);
            }
            Box3::None => {}
        }
        let q = charges[atom_idx];
        dip[0] += q * (anchor_pos[0] + dx);
        dip[1] += q * (anchor_pos[1] + dy);
        dip[2] += q * (anchor_pos[2] + dz);
    }
    Ok(dip)
}
