use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::pbc_math::{box_diagonal_extents, minimum_image_vector};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput, SpolOutput};

use super::solvent_orientation::{
    dot, minimum_image_between_atoms, normalize, reference_positions,
};

const DEBYE_PER_E_NM: f64 = 1.602_177_33 / 3.336e-2;

pub struct SolventPolarizationPlan {
    solute_selection: Selection,
    molecule_atoms: Vec<u32>,
    molecule_offsets: Vec<usize>,
    direction_local_atoms: [usize; 3],
    charges: Vec<f64>,
    r_min: f64,
    r_max: f64,
    bin: f64,
    use_com: bool,
    reference_atom: usize,
    refdip: f64,
    r_hist_max: Option<f64>,
    length_scale: f64,
    counts: Vec<u64>,
    bins: usize,
    effective_r_hist_max: f64,
    window_count: u64,
    sum_dipole: f64,
    sum_dipole_sq: f64,
    sum_radial_dipole: f64,
    sum_radial_polarization: f64,
    n_frames: usize,
    used_box: bool,
    initialized: bool,
}

impl SolventPolarizationPlan {
    pub fn new(
        solute_selection: Selection,
        atom1_indices: Vec<u32>,
        atom2_indices: Vec<u32>,
        atom3_indices: Vec<u32>,
        charges: Vec<f64>,
        r_min: f64,
        r_max: f64,
        bin: f64,
    ) -> Self {
        let (molecule_atoms, molecule_offsets) =
            triplets_to_molecule_layout(&atom1_indices, &atom2_indices, &atom3_indices);
        Self {
            solute_selection,
            molecule_atoms,
            molecule_offsets,
            direction_local_atoms: [0, 1, 2],
            charges,
            r_min,
            r_max,
            bin,
            use_com: false,
            reference_atom: 0,
            refdip: 0.0,
            r_hist_max: None,
            length_scale: 1.0,
            counts: Vec::new(),
            bins: 0,
            effective_r_hist_max: 0.0,
            window_count: 0,
            sum_dipole: 0.0,
            sum_dipole_sq: 0.0,
            sum_radial_dipole: 0.0,
            sum_radial_polarization: 0.0,
            n_frames: 0,
            used_box: false,
            initialized: false,
        }
    }

    pub fn with_use_com(mut self, use_com: bool) -> Self {
        self.use_com = use_com;
        self
    }

    pub fn with_reference_atom(mut self, reference_atom: usize) -> Self {
        self.reference_atom = reference_atom;
        self
    }

    pub fn with_refdip(mut self, refdip: f64) -> Self {
        self.refdip = refdip;
        self
    }

    pub fn with_r_hist_max(mut self, r_hist_max: Option<f64>) -> Self {
        self.r_hist_max = r_hist_max;
        self
    }

    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    pub fn with_molecules(
        mut self,
        molecule_atoms: Vec<u32>,
        molecule_offsets: Vec<usize>,
        direction_local_atoms: [usize; 3],
    ) -> Self {
        self.molecule_atoms = molecule_atoms;
        self.molecule_offsets = molecule_offsets;
        self.direction_local_atoms = direction_local_atoms;
        self
    }

    fn initialize_from_frame(&mut self, chunk: &FrameChunk, frame: usize) -> TrajResult<()> {
        self.used_box = !matches!(chunk.box_[frame], Box3::None);
        let requested_hist_max = if let Some(value) = self.r_hist_max {
            value
        } else if let Some(lengths) = box_diagonal_extents(chunk.box_[frame]) {
            0.99 * 0.5 * lengths.into_iter().fold(f64::INFINITY, f64::min) * self.length_scale
        } else {
            10.0 * self.r_max
        };
        if !requested_hist_max.is_finite() || requested_hist_max <= 0.0 {
            return Err(TrajError::Parse(
                "spol requires positive histogram extent".into(),
            ));
        }
        self.bins = ((requested_hist_max / self.bin).ceil() as usize).max(1);
        self.effective_r_hist_max = self.bins as f64 * self.bin;
        self.counts = vec![0; self.bins];
        self.initialized = true;
        Ok(())
    }
}

impl Plan for SolventPolarizationPlan {
    fn name(&self) -> &'static str {
        "spol"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.solute_selection.indices.is_empty() {
            return Err(TrajError::Mismatch(
                "spol requires a non-empty solute selection".into(),
            ));
        }
        if self.charges.len() != system.n_atoms() {
            return Err(TrajError::Mismatch(
                "spol charges length must match system atom count".into(),
            ));
        }
        if self.charges.iter().any(|q| !q.is_finite()) {
            return Err(TrajError::Parse(
                "spol charges must contain only finite values".into(),
            ));
        }
        if !self.r_min.is_finite() || self.r_min < 0.0 {
            return Err(TrajError::Parse(
                "spol r_min must be finite and >= 0".into(),
            ));
        }
        if !self.r_max.is_finite() || self.r_max <= self.r_min {
            return Err(TrajError::Parse(
                "spol r_max must be finite and > r_min".into(),
            ));
        }
        if !self.bin.is_finite() || self.bin <= 0.0 {
            return Err(TrajError::Parse("spol bin must be finite and > 0".into()));
        }
        if !self.refdip.is_finite() {
            return Err(TrajError::Parse("spol refdip must be finite".into()));
        }
        if let Some(value) = self.r_hist_max {
            if !value.is_finite() || value <= 0.0 {
                return Err(TrajError::Parse(
                    "spol r_hist_max must be finite and > 0".into(),
                ));
            }
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(TrajError::Parse(
                "spol length_scale must be finite and > 0".into(),
            ));
        }
        if system.atoms.mass.len() != system.n_atoms() {
            return Err(TrajError::Mismatch(
                "spol requires per-atom masses for COM reference mode".into(),
            ));
        }
        let n_atoms = system.n_atoms();
        if self.molecule_offsets.is_empty() || self.molecule_offsets[0] != 0 {
            return Err(TrajError::Mismatch(
                "spol molecule_offsets must start at 0".into(),
            ));
        }
        if *self.molecule_offsets.last().unwrap_or(&0) != self.molecule_atoms.len() {
            return Err(TrajError::Mismatch(
                "spol molecule_offsets must end at molecule_atoms length".into(),
            ));
        }
        if self.molecule_offsets.windows(2).any(|w| w[1] < w[0]) {
            return Err(TrajError::Mismatch(
                "spol molecule_offsets must be non-decreasing".into(),
            ));
        }
        if self.molecule_offsets.len() < 2 {
            return Err(TrajError::Mismatch(
                "spol requires at least one solvent molecule".into(),
            ));
        }
        for window in self.molecule_offsets.windows(2) {
            let len = window[1] - window[0];
            if len == 0 {
                return Err(TrajError::Mismatch(
                    "spol molecule_offsets may not contain empty molecules".into(),
                ));
            }
            let max_direction = *self.direction_local_atoms.iter().max().unwrap_or(&0);
            if self.reference_atom >= len || max_direction >= len {
                return Err(TrajError::Mismatch(
                    "spol reference_atom/direction_atom exceeds molecule size".into(),
                ));
            }
        }
        for &idx in self
            .molecule_atoms
            .iter()
            .chain(self.solute_selection.indices.iter())
        {
            if idx as usize >= n_atoms {
                return Err(TrajError::Mismatch("spol atom index out of bounds".into()));
            }
        }
        self.counts.clear();
        self.bins = 0;
        self.effective_r_hist_max = 0.0;
        self.window_count = 0;
        self.sum_dipole = 0.0;
        self.sum_dipole_sq = 0.0;
        self.sum_radial_dipole = 0.0;
        self.sum_radial_polarization = 0.0;
        self.n_frames = 0;
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
            let references = reference_positions(
                frame_coords,
                &self.solute_selection.indices,
                &system.atoms.mass,
                chunk.box_[frame],
                self.length_scale,
                self.use_com,
            )?;

            for mol in 0..self.molecule_offsets.len() - 1 {
                let start = self.molecule_offsets[mol];
                let end = self.molecule_offsets[mol + 1];
                let atoms = &self.molecule_atoms[start..end];
                let absolute_positions = molecule_absolute_positions(
                    frame_coords,
                    atoms,
                    chunk.box_[frame],
                    self.length_scale,
                )?;
                let reference_atom = absolute_positions[self.reference_atom];
                let radial = closest_reference_vector(
                    reference_atom,
                    &references,
                    chunk.box_[frame],
                    self.length_scale,
                )?;
                let r = dot(radial, radial).sqrt();
                if r < self.effective_r_hist_max {
                    let bin = shell_bin(r, self.bin, self.bins);
                    self.counts[bin] = self.counts[bin].saturating_add(1);
                }
                if r < self.r_min || r >= self.r_max {
                    continue;
                }
                let Some(radial_unit) = normalize(radial) else {
                    continue;
                };
                let (dipole, direction) = molecule_dipole_direction(
                    atoms,
                    &absolute_positions,
                    &self.charges,
                    self.direction_local_atoms,
                )?;
                let dipole_magnitude = dot(dipole, dipole).sqrt();
                self.sum_dipole += dipole_magnitude;
                self.sum_dipole_sq += dipole_magnitude * dipole_magnitude;
                self.sum_radial_dipole += dot(radial_unit, dipole);
                self.sum_radial_polarization += dot(
                    radial_unit,
                    [
                        dipole[0] - self.refdip * direction[0],
                        dipole[1] - self.refdip * direction[1],
                        dipole[2] - self.refdip * direction[2],
                    ],
                );
                self.window_count = self.window_count.saturating_add(1);
            }
            self.n_frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.n_frames == 0 || !self.initialized {
            return Ok(PlanOutput::Spol(SpolOutput {
                r: Vec::new(),
                cumulative_count: Vec::new(),
                shell_count: Vec::new(),
                shell_count_per_frame: Vec::new(),
                average_shell_size: 0.0,
                average_dipole: 0.0,
                dipole_std: 0.0,
                average_radial_dipole: 0.0,
                average_radial_polarization: 0.0,
                window_count: 0,
                r_window: [self.r_min as f32, self.r_max as f32],
                bin_width: self.bin as f32,
                r_hist_max: 0.0,
                use_com: self.use_com,
                reference_atom: self.reference_atom,
                refdip: self.refdip as f32,
                n_frames: 0,
                used_box: self.used_box,
                length_scale: self.length_scale as f32,
                dipole_unit: "debye".to_string(),
            }));
        }

        let mut r = Vec::with_capacity(self.bins);
        let mut cumulative_count = Vec::with_capacity(self.bins);
        let mut shell_count_per_frame = Vec::with_capacity(self.bins);
        let mut running = 0u64;
        for i in 0..self.bins {
            running = running.saturating_add(self.counts[i]);
            r.push(((i as f64 + 1.0) * self.bin) as f32);
            cumulative_count.push(running as f32 / self.n_frames as f32);
            shell_count_per_frame.push(self.counts[i] as f32 / self.n_frames as f32);
        }
        let average_shell_size = self.window_count as f64 / self.n_frames as f64;
        let average_dipole = if self.window_count == 0 {
            0.0
        } else {
            self.sum_dipole / self.window_count as f64
        };
        let variance = if self.window_count == 0 {
            0.0
        } else {
            (self.sum_dipole_sq / self.window_count as f64) - average_dipole * average_dipole
        };
        let average_radial_dipole = if self.window_count == 0 {
            0.0
        } else {
            self.sum_radial_dipole / self.window_count as f64
        };
        let average_radial_polarization = if self.window_count == 0 {
            0.0
        } else {
            self.sum_radial_polarization / self.window_count as f64
        };

        Ok(PlanOutput::Spol(SpolOutput {
            r,
            cumulative_count,
            shell_count: std::mem::take(&mut self.counts),
            shell_count_per_frame,
            average_shell_size: average_shell_size as f32,
            average_dipole: average_dipole as f32,
            dipole_std: variance.max(0.0).sqrt() as f32,
            average_radial_dipole: average_radial_dipole as f32,
            average_radial_polarization: average_radial_polarization as f32,
            window_count: self.window_count,
            r_window: [self.r_min as f32, self.r_max as f32],
            bin_width: self.bin as f32,
            r_hist_max: self.effective_r_hist_max as f32,
            use_com: self.use_com,
            reference_atom: self.reference_atom,
            refdip: self.refdip as f32,
            n_frames: self.n_frames,
            used_box: self.used_box,
            length_scale: self.length_scale as f32,
            dipole_unit: "debye".to_string(),
        }))
    }
}

fn closest_reference_vector(
    atom: [f64; 3],
    references: &[[f64; 3]],
    box_: Box3,
    length_scale: f64,
) -> TrajResult<[f64; 3]> {
    let mut best = [0.0f64; 3];
    let mut best_r2 = f64::INFINITY;
    for reference in references.iter() {
        let trial = minimum_image_point_to_reference(atom, *reference, box_, length_scale)?;
        let r2 = dot(trial, trial);
        if r2 < best_r2 {
            best = trial;
            best_r2 = r2;
        }
    }
    Ok(best)
}

fn shell_bin(r: f64, bin_width: f64, bins: usize) -> usize {
    let mut index = (r / bin_width).floor() as usize;
    if index >= bins {
        index = bins - 1;
    }
    index
}

fn triplets_to_molecule_layout(
    atom1_indices: &[u32],
    atom2_indices: &[u32],
    atom3_indices: &[u32],
) -> (Vec<u32>, Vec<usize>) {
    let n = atom1_indices.len();
    let mut molecule_atoms = Vec::with_capacity(n * 3);
    let mut molecule_offsets = Vec::with_capacity(n + 1);
    molecule_offsets.push(0);
    for i in 0..n {
        molecule_atoms.push(atom1_indices[i]);
        molecule_atoms.push(atom2_indices[i]);
        molecule_atoms.push(atom3_indices[i]);
        molecule_offsets.push(molecule_atoms.len());
    }
    (molecule_atoms, molecule_offsets)
}

fn molecule_absolute_positions(
    frame_coords: &[[f32; 4]],
    atoms: &[u32],
    box_: Box3,
    length_scale: f64,
) -> TrajResult<Vec<[f64; 3]>> {
    let first_idx = atoms[0] as usize;
    let first = frame_coords[first_idx];
    let anchor = [
        first[0] as f64 * length_scale,
        first[1] as f64 * length_scale,
        first[2] as f64 * length_scale,
    ];
    let mut out = Vec::with_capacity(atoms.len());
    for &atom_u32 in atoms.iter() {
        let atom_idx = atom_u32 as usize;
        if atom_idx == first_idx {
            out.push(anchor);
            continue;
        }
        let delta = minimum_image_between_atoms(frame_coords[atom_idx], first, box_, length_scale)?;
        out.push([
            anchor[0] + delta[0],
            anchor[1] + delta[1],
            anchor[2] + delta[2],
        ]);
    }
    Ok(out)
}

fn molecule_dipole_direction(
    atoms: &[u32],
    absolute_positions: &[[f64; 3]],
    charges: &[f64],
    direction_local_atoms: [usize; 3],
) -> TrajResult<([f64; 3], [f64; 3])> {
    let q_avg = atoms.iter().map(|&idx| charges[idx as usize]).sum::<f64>() / atoms.len() as f64;
    let mut dipole = [0.0f64; 3];
    for (local_idx, &atom_u32) in atoms.iter().enumerate() {
        let q = charges[atom_u32 as usize] - q_avg;
        let pos = absolute_positions[local_idx];
        dipole[0] += q * pos[0] * DEBYE_PER_E_NM;
        dipole[1] += q * pos[1] * DEBYE_PER_E_NM;
        dipole[2] += q * pos[2] * DEBYE_PER_E_NM;
    }
    let p0 = absolute_positions[direction_local_atoms[0]];
    let p1 = absolute_positions[direction_local_atoms[1]];
    let p2 = absolute_positions[direction_local_atoms[2]];
    let direction = normalize([
        0.5 * (p1[0] + p2[0]) - p0[0],
        0.5 * (p1[1] + p2[1]) - p0[1],
        0.5 * (p1[2] + p2[2]) - p0[2],
    ])
    .unwrap_or([0.0, 0.0, 0.0]);
    Ok((dipole, direction))
}

fn minimum_image_point_to_reference(
    atom: [f64; 3],
    reference: [f64; 3],
    box_: Box3,
    length_scale: f64,
) -> TrajResult<[f64; 3]> {
    minimum_image_vector(reference, atom, box_, length_scale)
}
