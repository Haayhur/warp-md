use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::pbc_utils::{apply_pbc, apply_pbc_triclinic, cell_and_inv_from_box};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput, SOrientOutput};

pub struct SOrientPlan {
    solute_selection: Selection,
    atom1_indices: Vec<u32>,
    atom2_indices: Vec<u32>,
    atom3_indices: Vec<u32>,
    r_min: f64,
    r_max: f64,
    cbin: f64,
    rbin: f64,
    r_profile_max: Option<f64>,
    use_com: bool,
    use_vector23: bool,
    length_scale: f64,
    hist_theta1: Vec<u64>,
    hist_theta2: Vec<u64>,
    radial_sum_theta1: Vec<f64>,
    radial_sum_p2_theta2: Vec<f64>,
    radial_counts: Vec<u64>,
    radial_bins: usize,
    theta1_bins: usize,
    theta2_bins: usize,
    effective_r_profile_max: f64,
    window_count: u64,
    window_sum_theta1: f64,
    window_sum_p2_theta2: f64,
    n_frames: usize,
    n_reference_positions: usize,
    used_box: bool,
    initialized: bool,
}

impl SOrientPlan {
    pub fn new(
        solute_selection: Selection,
        atom1_indices: Vec<u32>,
        atom2_indices: Vec<u32>,
        atom3_indices: Vec<u32>,
        r_min: f64,
        r_max: f64,
        cbin: f64,
        rbin: f64,
    ) -> Self {
        Self {
            solute_selection,
            atom1_indices,
            atom2_indices,
            atom3_indices,
            r_min,
            r_max,
            cbin,
            rbin,
            r_profile_max: None,
            use_com: false,
            use_vector23: false,
            length_scale: 1.0,
            hist_theta1: Vec::new(),
            hist_theta2: Vec::new(),
            radial_sum_theta1: Vec::new(),
            radial_sum_p2_theta2: Vec::new(),
            radial_counts: Vec::new(),
            radial_bins: 0,
            theta1_bins: 0,
            theta2_bins: 0,
            effective_r_profile_max: 0.0,
            window_count: 0,
            window_sum_theta1: 0.0,
            window_sum_p2_theta2: 0.0,
            n_frames: 0,
            n_reference_positions: 0,
            used_box: false,
            initialized: false,
        }
    }

    pub fn with_r_profile_max(mut self, r_profile_max: Option<f64>) -> Self {
        self.r_profile_max = r_profile_max;
        self
    }

    pub fn with_use_com(mut self, use_com: bool) -> Self {
        self.use_com = use_com;
        self
    }

    pub fn with_use_vector23(mut self, use_vector23: bool) -> Self {
        self.use_vector23 = use_vector23;
        self
    }

    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    fn initialize_from_frame(&mut self, chunk: &FrameChunk, frame: usize) -> TrajResult<()> {
        self.used_box = !matches!(chunk.box_[frame], Box3::None);
        let requested_profile_max = if let Some(value) = self.r_profile_max {
            value
        } else if let Some(lengths) = box_lengths(chunk.box_[frame]) {
            0.99 * 0.5 * lengths.into_iter().fold(f64::INFINITY, f64::min) * self.length_scale
        } else {
            10.0 * self.r_max
        };
        if !requested_profile_max.is_finite() || requested_profile_max <= 0.0 {
            return Err(TrajError::Parse(
                "sorient requires positive radial profile extent".into(),
            ));
        }

        self.theta1_bins = ((2.0 / self.cbin).round() as usize).max(1);
        self.theta2_bins = ((1.0 / self.cbin).round() as usize).max(1);
        self.radial_bins = ((requested_profile_max / self.rbin).ceil() as usize).max(1);
        self.effective_r_profile_max = self.radial_bins as f64 * self.rbin;
        self.hist_theta1 = vec![0; self.theta1_bins];
        self.hist_theta2 = vec![0; self.theta2_bins];
        self.radial_sum_theta1 = vec![0.0; self.radial_bins];
        self.radial_sum_p2_theta2 = vec![0.0; self.radial_bins];
        self.radial_counts = vec![0; self.radial_bins];
        self.initialized = true;
        Ok(())
    }
}

impl Plan for SOrientPlan {
    fn name(&self) -> &'static str {
        "sorient"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.solute_selection.indices.is_empty() {
            return Err(TrajError::Mismatch(
                "sorient requires a non-empty solute selection".into(),
            ));
        }
        if self.atom1_indices.len() != self.atom2_indices.len()
            || self.atom1_indices.len() != self.atom3_indices.len()
        {
            return Err(TrajError::Mismatch(
                "sorient solvent triplet index vectors must have identical length".into(),
            ));
        }
        if !self.r_min.is_finite() || self.r_min < 0.0 {
            return Err(TrajError::Parse(
                "sorient r_min must be finite and >= 0".into(),
            ));
        }
        if !self.r_max.is_finite() || self.r_max <= self.r_min {
            return Err(TrajError::Parse(
                "sorient r_max must be finite and > r_min".into(),
            ));
        }
        if !self.cbin.is_finite() || self.cbin <= 0.0 {
            return Err(TrajError::Parse(
                "sorient cbin must be finite and > 0".into(),
            ));
        }
        if !self.rbin.is_finite() || self.rbin <= 0.0 {
            return Err(TrajError::Parse(
                "sorient rbin must be finite and > 0".into(),
            ));
        }
        if let Some(value) = self.r_profile_max {
            if !value.is_finite() || value <= 0.0 {
                return Err(TrajError::Parse(
                    "sorient r_profile_max must be finite and > 0".into(),
                ));
            }
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(TrajError::Parse(
                "sorient length_scale must be finite and > 0".into(),
            ));
        }
        if system.atoms.mass.len() != system.n_atoms() {
            return Err(TrajError::Mismatch(
                "sorient requires per-atom masses for COM reference mode".into(),
            ));
        }
        let n_atoms = system.n_atoms();
        for &idx in self
            .atom1_indices
            .iter()
            .chain(self.atom2_indices.iter())
            .chain(self.atom3_indices.iter())
            .chain(self.solute_selection.indices.iter())
        {
            if idx as usize >= n_atoms {
                return Err(TrajError::Mismatch(
                    "sorient atom index out of bounds".into(),
                ));
            }
        }
        self.hist_theta1.clear();
        self.hist_theta2.clear();
        self.radial_sum_theta1.clear();
        self.radial_sum_p2_theta2.clear();
        self.radial_counts.clear();
        self.radial_bins = 0;
        self.theta1_bins = 0;
        self.theta2_bins = 0;
        self.effective_r_profile_max = 0.0;
        self.window_count = 0;
        self.window_sum_theta1 = 0.0;
        self.window_sum_p2_theta2 = 0.0;
        self.n_frames = 0;
        self.n_reference_positions = if self.use_com {
            1
        } else {
            self.solute_selection.indices.len()
        };
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
            self.used_box = self.used_box || !matches!(chunk.box_[frame], Box3::None);
            let base = frame * chunk.n_atoms;
            let frame_coords = &chunk.coords[base..base + chunk.n_atoms];
            let reference_positions = reference_positions(
                frame_coords,
                &self.solute_selection.indices,
                &system.atoms.mass,
                chunk.box_[frame],
                self.length_scale,
                self.use_com,
            )?;

            for reference in reference_positions.iter() {
                for i in 0..self.atom1_indices.len() {
                    let atom1 = self.atom1_indices[i] as usize;
                    let atom2 = self.atom2_indices[i] as usize;
                    let atom3 = self.atom3_indices[i] as usize;
                    let radial = minimum_image_to_reference(
                        frame_coords[atom1],
                        *reference,
                        chunk.box_[frame],
                        self.length_scale,
                    )?;
                    let r2 = dot(radial, radial);
                    if r2 > self.effective_r_profile_max * self.effective_r_profile_max {
                        continue;
                    }
                    let r = r2.sqrt();
                    let Some(radial_unit) = normalize(radial) else {
                        continue;
                    };
                    let (cos_theta1, cos_theta2) = solvent_orientation(
                        frame_coords,
                        atom1,
                        atom2,
                        atom3,
                        radial_unit,
                        chunk.box_[frame],
                        self.length_scale,
                        self.use_vector23,
                    )?;
                    let p2_theta2 = 3.0 * cos_theta2 * cos_theta2 - 1.0;

                    let radial_bin = radial_bin(r, self.rbin, self.radial_bins);
                    self.radial_sum_theta1[radial_bin] += cos_theta1;
                    self.radial_sum_p2_theta2[radial_bin] += p2_theta2;
                    self.radial_counts[radial_bin] =
                        self.radial_counts[radial_bin].saturating_add(1);

                    if r >= self.r_min && r < self.r_max {
                        let theta1_bin =
                            unit_interval_bin((cos_theta1 + 1.0) * 0.5, self.theta1_bins);
                        let theta2_bin = unit_interval_bin(cos_theta2.abs(), self.theta2_bins);
                        self.hist_theta1[theta1_bin] =
                            self.hist_theta1[theta1_bin].saturating_add(1);
                        self.hist_theta2[theta2_bin] =
                            self.hist_theta2[theta2_bin].saturating_add(1);
                        self.window_count = self.window_count.saturating_add(1);
                        self.window_sum_theta1 += cos_theta1;
                        self.window_sum_p2_theta2 += p2_theta2;
                    }
                }
            }
            self.n_frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.n_frames == 0 || !self.initialized {
            return Ok(PlanOutput::SOrient(SOrientOutput {
                cos_theta1: Vec::new(),
                cos_theta1_distribution: Vec::new(),
                abs_cos_theta2: Vec::new(),
                abs_cos_theta2_distribution: Vec::new(),
                r: Vec::new(),
                mean_cos_theta1: Vec::new(),
                mean_p2_theta2: Vec::new(),
                cumulative_r: Vec::new(),
                cumulative_cos_theta1: Vec::new(),
                cumulative_p2_theta2: Vec::new(),
                count_density: Vec::new(),
                counts: Vec::new(),
                window_count: 0,
                average_shell_size: 0.0,
                window_mean_cos_theta1: 0.0,
                window_mean_p2_theta2: 0.0,
                r_window: [self.r_min as f32, self.r_max as f32],
                cbin: self.cbin as f32,
                rbin: self.rbin as f32,
                r_profile_max: 0.0,
                use_vector23: self.use_vector23,
                use_com: self.use_com,
                n_frames: self.n_frames,
                n_reference_positions: self.n_reference_positions,
                used_box: self.used_box,
                length_scale: self.length_scale as f32,
            }));
        }

        let theta1_width = 2.0 / self.theta1_bins as f64;
        let theta2_width = 1.0 / self.theta2_bins as f64;
        let mut theta1_axis = Vec::with_capacity(self.theta1_bins);
        let mut theta1_distribution = Vec::with_capacity(self.theta1_bins);
        for i in 0..self.theta1_bins {
            theta1_axis.push((-1.0 + (i as f64 + 0.5) * theta1_width) as f32);
            let value = if self.window_count == 0 {
                0.0
            } else {
                self.hist_theta1[i] as f64 / (self.window_count as f64 * theta1_width)
            };
            theta1_distribution.push(value as f32);
        }

        let mut theta2_axis = Vec::with_capacity(self.theta2_bins);
        let mut theta2_distribution = Vec::with_capacity(self.theta2_bins);
        for i in 0..self.theta2_bins {
            theta2_axis.push(((i as f64 + 0.5) * theta2_width) as f32);
            let value = if self.window_count == 0 {
                0.0
            } else {
                self.hist_theta2[i] as f64 / (self.window_count as f64 * theta2_width)
            };
            theta2_distribution.push(value as f32);
        }

        let mut r = Vec::with_capacity(self.radial_bins);
        let mut mean_cos_theta1 = Vec::with_capacity(self.radial_bins);
        let mut mean_p2_theta2 = Vec::with_capacity(self.radial_bins);
        let mut cumulative_r = Vec::with_capacity(self.radial_bins);
        let mut cumulative_cos_theta1 = Vec::with_capacity(self.radial_bins);
        let mut cumulative_p2_theta2 = Vec::with_capacity(self.radial_bins);
        let mut count_density = Vec::with_capacity(self.radial_bins);
        let mut running_cos = 0.0f64;
        let mut running_p2 = 0.0f64;
        let cumulative_norm = (self.n_reference_positions.max(1) * self.n_frames.max(1)) as f64;
        for i in 0..self.radial_bins {
            r.push(((i as f64 + 0.5) * self.rbin) as f32);
            cumulative_r.push(((i as f64 + 1.0) * self.rbin) as f32);
            let count = self.radial_counts[i] as f64;
            if count > 0.0 {
                mean_cos_theta1.push((self.radial_sum_theta1[i] / count) as f32);
                mean_p2_theta2.push((self.radial_sum_p2_theta2[i] / count) as f32);
            } else {
                mean_cos_theta1.push(0.0);
                mean_p2_theta2.push(0.0);
            }
            running_cos += self.radial_sum_theta1[i];
            running_p2 += self.radial_sum_p2_theta2[i];
            cumulative_cos_theta1.push((running_cos / cumulative_norm) as f32);
            cumulative_p2_theta2.push((running_p2 / cumulative_norm) as f32);
            count_density
                .push((self.radial_counts[i] as f64 / (self.rbin * self.n_frames as f64)) as f32);
        }

        let average_shell_size = self.window_count as f64
            / (self.n_reference_positions.max(1) * self.n_frames.max(1)) as f64;
        let window_mean_cos_theta1 = if self.window_count == 0 {
            0.0
        } else {
            self.window_sum_theta1 / self.window_count as f64
        };
        let window_mean_p2_theta2 = if self.window_count == 0 {
            0.0
        } else {
            self.window_sum_p2_theta2 / self.window_count as f64
        };

        Ok(PlanOutput::SOrient(SOrientOutput {
            cos_theta1: theta1_axis,
            cos_theta1_distribution: theta1_distribution,
            abs_cos_theta2: theta2_axis,
            abs_cos_theta2_distribution: theta2_distribution,
            r,
            mean_cos_theta1,
            mean_p2_theta2,
            cumulative_r,
            cumulative_cos_theta1,
            cumulative_p2_theta2,
            count_density,
            counts: std::mem::take(&mut self.radial_counts),
            window_count: self.window_count,
            average_shell_size: average_shell_size as f32,
            window_mean_cos_theta1: window_mean_cos_theta1 as f32,
            window_mean_p2_theta2: window_mean_p2_theta2 as f32,
            r_window: [self.r_min as f32, self.r_max as f32],
            cbin: self.cbin as f32,
            rbin: self.rbin as f32,
            r_profile_max: self.effective_r_profile_max as f32,
            use_vector23: self.use_vector23,
            use_com: self.use_com,
            n_frames: self.n_frames,
            n_reference_positions: self.n_reference_positions,
            used_box: self.used_box,
            length_scale: self.length_scale as f32,
        }))
    }
}

pub(crate) fn box_lengths(box_: Box3) -> Option<[f64; 3]> {
    match box_ {
        Box3::Orthorhombic { lx, ly, lz } => Some([lx as f64, ly as f64, lz as f64]),
        Box3::Triclinic { m } => Some([m[0] as f64, m[4] as f64, m[8] as f64]),
        Box3::None => None,
    }
}

fn radial_bin(r: f64, rbin: f64, bins: usize) -> usize {
    let mut index = (r / rbin).floor() as usize;
    if index >= bins {
        index = bins - 1;
    }
    index
}

fn unit_interval_bin(value: f64, bins: usize) -> usize {
    if bins <= 1 {
        return 0;
    }
    if value >= 1.0 {
        return bins - 1;
    }
    let clamped = value.max(0.0);
    let mut index = (clamped * bins as f64).floor() as usize;
    if index >= bins {
        index = bins - 1;
    }
    index
}

pub(crate) fn reference_positions(
    frame_coords: &[[f32; 4]],
    indices: &[u32],
    masses: &[f32],
    box_: Box3,
    length_scale: f64,
    use_com: bool,
) -> TrajResult<Vec<[f64; 3]>> {
    if !use_com {
        return Ok(indices
            .iter()
            .map(|&idx| {
                let pos = frame_coords[idx as usize];
                [
                    pos[0] as f64 * length_scale,
                    pos[1] as f64 * length_scale,
                    pos[2] as f64 * length_scale,
                ]
            })
            .collect());
    }

    let first = frame_coords[indices[0] as usize];
    let anchor = [
        first[0] as f64 * length_scale,
        first[1] as f64 * length_scale,
        first[2] as f64 * length_scale,
    ];
    let mut total_mass = 0.0f64;
    let mut com = [0.0f64; 3];
    for &idx_u32 in indices.iter() {
        let idx = idx_u32 as usize;
        let mass = masses[idx].max(0.0) as f64;
        let weight = if mass > 0.0 { mass } else { 1.0 };
        let absolute = if idx_u32 == indices[0] {
            anchor
        } else {
            let delta = minimum_image_between_atoms(frame_coords[idx], first, box_, length_scale)?;
            [
                anchor[0] + delta[0],
                anchor[1] + delta[1],
                anchor[2] + delta[2],
            ]
        };
        com[0] += weight * absolute[0];
        com[1] += weight * absolute[1];
        com[2] += weight * absolute[2];
        total_mass += weight;
    }
    if total_mass <= 0.0 {
        return Err(TrajError::Mismatch(
            "sorient could not determine reference COM mass".into(),
        ));
    }
    com[0] /= total_mass;
    com[1] /= total_mass;
    com[2] /= total_mass;
    Ok(vec![com])
}

fn solvent_orientation(
    frame_coords: &[[f32; 4]],
    atom1: usize,
    atom2: usize,
    atom3: usize,
    radial_unit: [f64; 3],
    box_: Box3,
    length_scale: f64,
    use_vector23: bool,
) -> TrajResult<(f64, f64)> {
    let bond12 =
        minimum_image_between_atoms(frame_coords[atom2], frame_coords[atom1], box_, length_scale)?;
    let bond13 =
        minimum_image_between_atoms(frame_coords[atom3], frame_coords[atom1], box_, length_scale)?;
    let Some(bisector) = normalize(add(bond12, bond13)) else {
        return Ok((0.0, 0.0));
    };
    let cos_theta1 = dot(radial_unit, bisector).clamp(-1.0, 1.0);
    let orient = if use_vector23 {
        let bond23 = minimum_image_between_atoms(
            frame_coords[atom3],
            frame_coords[atom2],
            box_,
            length_scale,
        )?;
        normalize(bond23)
    } else {
        normalize(cross(bisector, bond13))
    };
    let cos_theta2 = orient
        .map(|value| dot(radial_unit, value).clamp(-1.0, 1.0))
        .unwrap_or(0.0);
    Ok((cos_theta1, cos_theta2))
}

pub(crate) fn minimum_image_between_atoms(
    x1: [f32; 4],
    x2: [f32; 4],
    box_: Box3,
    length_scale: f64,
) -> TrajResult<[f64; 3]> {
    let mut dx = (x1[0] - x2[0]) as f64 * length_scale;
    let mut dy = (x1[1] - x2[1]) as f64 * length_scale;
    let mut dz = (x1[2] - x2[2]) as f64 * length_scale;
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
            let (mut cell, mut inv) = cell_and_inv_from_box(box_)?;
            for row in 0..3 {
                for col in 0..3 {
                    cell[row][col] *= length_scale;
                    inv[row][col] /= length_scale;
                }
            }
            apply_pbc_triclinic(&mut dx, &mut dy, &mut dz, &cell, &inv);
        }
        Box3::None => {}
    }
    Ok([dx, dy, dz])
}

pub(crate) fn minimum_image_to_reference(
    atom: [f32; 4],
    reference: [f64; 3],
    box_: Box3,
    length_scale: f64,
) -> TrajResult<[f64; 3]> {
    let mut dx = atom[0] as f64 * length_scale - reference[0];
    let mut dy = atom[1] as f64 * length_scale - reference[1];
    let mut dz = atom[2] as f64 * length_scale - reference[2];
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
            let (mut cell, mut inv) = cell_and_inv_from_box(box_)?;
            for row in 0..3 {
                for col in 0..3 {
                    cell[row][col] *= length_scale;
                    inv[row][col] /= length_scale;
                }
            }
            apply_pbc_triclinic(&mut dx, &mut dy, &mut dz, &cell, &inv);
        }
        Box3::None => {}
    }
    Ok([dx, dy, dz])
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn normalize(v: [f64; 3]) -> Option<[f64; 3]> {
    let norm2 = dot(v, v);
    if norm2 <= 0.0 {
        return None;
    }
    let inv = norm2.sqrt().recip();
    Some([v[0] * inv, v[1] * inv, v[2] * inv])
}
