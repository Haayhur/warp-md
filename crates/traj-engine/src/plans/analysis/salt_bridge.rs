use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::pbc_math::{minimum_image_delta, orthorhombic_lengths};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::analysis::grouping::{GroupBy, GroupSpec};

const CHARGE_TOL: f64 = 1.0e-5;
const CLASS_PLUS_PLUS: u8 = 0;
const CLASS_MIN_MIN: u8 = 1;
const CLASS_PLUS_MIN: u8 = 2;

pub struct SaltBridgePlan {
    selection: Selection,
    group_by: GroupBy,
    charges: Vec<f64>,
    length_scale: f64,
    truncate: Option<f64>,
    contact_cutoff: Option<f64>,
    groups: Vec<Vec<usize>>,
    group_labels: Vec<String>,
    group_charges: Vec<f64>,
    group_signs: Vec<i8>,
    pair_indices: Vec<[u32; 2]>,
    pair_labels: Vec<String>,
    pair_classes: Vec<u8>,
    time: Vec<f32>,
    distances: Vec<f32>,
    min_distance: Vec<f32>,
    contact_count: Vec<u64>,
    frames: usize,
}

impl SaltBridgePlan {
    pub fn new(selection: Selection, group_by: GroupBy, charges: Vec<f64>) -> Self {
        Self {
            selection,
            group_by,
            charges,
            length_scale: 1.0,
            truncate: None,
            contact_cutoff: None,
            groups: Vec::new(),
            group_labels: Vec::new(),
            group_charges: Vec::new(),
            group_signs: Vec::new(),
            pair_indices: Vec::new(),
            pair_labels: Vec::new(),
            pair_classes: Vec::new(),
            time: Vec::new(),
            distances: Vec::new(),
            min_distance: Vec::new(),
            contact_count: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }

    pub fn with_truncate(mut self, truncate: Option<f64>) -> Self {
        self.truncate = truncate;
        self
    }

    pub fn with_contact_cutoff(mut self, cutoff: Option<f64>) -> Self {
        self.contact_cutoff = cutoff;
        self
    }

    pub fn group_labels(&self) -> &[String] {
        &self.group_labels
    }

    pub fn group_charges(&self) -> &[f64] {
        &self.group_charges
    }

    pub fn pair_indices(&self) -> &[[u32; 2]] {
        &self.pair_indices
    }

    pub fn pair_labels(&self) -> &[String] {
        &self.pair_labels
    }

    pub fn pair_classes(&self) -> &[u8] {
        &self.pair_classes
    }

    pub fn min_distance(&self) -> &[f32] {
        &self.min_distance
    }

    pub fn contact_count(&self) -> &[u64] {
        &self.contact_count
    }

    pub fn contact_cutoff(&self) -> Option<f64> {
        self.contact_cutoff
    }

    pub fn truncate(&self) -> Option<f64> {
        self.truncate
    }

    fn init_groups(&mut self, system: &System) -> TrajResult<()> {
        if self.charges.len() != system.n_atoms() {
            return Err(TrajError::Mismatch(
                "charges length does not match atom count".into(),
            ));
        }
        if self.charges.iter().any(|value| !value.is_finite()) {
            return Err(TrajError::Parse(
                "saltbr charges must contain only finite values".into(),
            ));
        }
        if let Some(value) = self.truncate {
            if !value.is_finite() || value <= 0.0 {
                return Err(TrajError::Parse(
                    "saltbr truncate must be finite and > 0".into(),
                ));
            }
        }
        if let Some(value) = self.contact_cutoff {
            if !value.is_finite() || value <= 0.0 {
                return Err(TrajError::Parse(
                    "saltbr contact_cutoff must be finite and > 0".into(),
                ));
            }
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(TrajError::Parse(
                "saltbr length_scale must be finite and > 0".into(),
            ));
        }

        let groups = GroupSpec::new(self.selection.clone(), self.group_by).build(system)?;
        self.groups.clear();
        self.group_labels.clear();
        self.group_charges.clear();
        self.group_signs.clear();

        for atoms in groups.groups.iter() {
            let mut charge_sum = 0.0f64;
            for &atom_idx in atoms {
                charge_sum += self.charges[atom_idx];
            }
            if charge_sum.abs() <= CHARGE_TOL {
                continue;
            }
            self.groups.push(atoms.clone());
            self.group_labels
                .push(label_for_group(system, self.group_by, atoms));
            self.group_charges.push(charge_sum);
            self.group_signs.push(if charge_sum > 0.0 { 1 } else { -1 });
        }

        if self.groups.len() < 2 {
            return Err(TrajError::Mismatch(
                "saltbr requires at least two non-neutral charged groups".into(),
            ));
        }

        self.pair_indices.clear();
        self.pair_labels.clear();
        self.pair_classes.clear();
        for left in 0..self.groups.len() {
            for right in (left + 1)..self.groups.len() {
                self.pair_indices.push([left as u32, right as u32]);
                self.pair_labels.push(format!(
                    "{}:{}",
                    self.group_labels[left], self.group_labels[right]
                ));
                self.pair_classes.push(classify_pair(
                    self.group_signs[left],
                    self.group_signs[right],
                ));
            }
        }
        self.min_distance = vec![f32::INFINITY; self.pair_indices.len()];
        self.contact_count = vec![0u64; self.pair_indices.len()];
        Ok(())
    }
}

impl Plan for SaltBridgePlan {
    fn name(&self) -> &'static str {
        "saltbr"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.time.clear();
        self.distances.clear();
        self.frames = 0;
        self.init_groups(system)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.groups.is_empty() {
            return Ok(());
        }
        let n_groups = self.groups.len();
        let mut centers = vec![[0.0f64; 3]; n_groups];
        for frame in 0..chunk.n_frames {
            let time_ps = chunk
                .time_ps
                .as_ref()
                .and_then(|time| time.get(frame).copied())
                .unwrap_or(self.frames as f32);
            self.time.push(time_ps);
            for group_idx in 0..n_groups {
                centers[group_idx] = charge_center(
                    chunk,
                    frame,
                    &self.groups[group_idx],
                    &self.charges,
                    self.group_charges[group_idx],
                    self.length_scale,
                );
            }
            let box_l = orthorhombic_lengths(&chunk.box_[frame]).map(|[lx, ly, lz]| {
                [
                    lx * self.length_scale,
                    ly * self.length_scale,
                    lz * self.length_scale,
                ]
            });
            for (pair_idx, [left, right]) in self.pair_indices.iter().copied().enumerate() {
                let [dx, dy, dz] =
                    minimum_image_delta(centers[left as usize], centers[right as usize], box_l);
                let distance = (dx * dx + dy * dy + dz * dz).sqrt() as f32;
                self.distances.push(distance);
                if distance < self.min_distance[pair_idx] {
                    self.min_distance[pair_idx] = distance;
                }
                if let Some(cutoff) = self.contact_cutoff {
                    if (distance as f64) < cutoff {
                        self.contact_count[pair_idx] += 1;
                    }
                }
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let old_cols = self.pair_indices.len();
        let keep: Vec<usize> = if let Some(truncate) = self.truncate {
            self.min_distance
                .iter()
                .enumerate()
                .filter_map(|(idx, &distance)| {
                    if (distance as f64) < truncate {
                        Some(idx)
                    } else {
                        None
                    }
                })
                .collect()
        } else {
            (0..old_cols).collect()
        };

        let old_distances = std::mem::take(&mut self.distances);
        let mut filtered = Vec::with_capacity(self.frames.saturating_mul(keep.len()));
        if !keep.is_empty() && old_cols > 0 {
            for row in 0..self.frames {
                let row_offset = row * old_cols;
                for &pair_idx in keep.iter() {
                    filtered.push(old_distances[row_offset + pair_idx]);
                }
            }
        }

        self.pair_indices = keep.iter().map(|&idx| self.pair_indices[idx]).collect();
        self.pair_labels = keep
            .iter()
            .map(|&idx| self.pair_labels[idx].clone())
            .collect();
        self.pair_classes = keep.iter().map(|&idx| self.pair_classes[idx]).collect();
        self.min_distance = keep.iter().map(|&idx| self.min_distance[idx]).collect();
        self.contact_count = keep.iter().map(|&idx| self.contact_count[idx]).collect();

        Ok(PlanOutput::TimeSeries {
            time: std::mem::take(&mut self.time),
            data: filtered,
            rows: self.frames,
            cols: self.pair_indices.len(),
        })
    }
}

fn classify_pair(left: i8, right: i8) -> u8 {
    if left > 0 && right > 0 {
        CLASS_PLUS_PLUS
    } else if left < 0 && right < 0 {
        CLASS_MIN_MIN
    } else {
        CLASS_PLUS_MIN
    }
}

fn label_for_group(system: &System, group_by: GroupBy, atoms: &[usize]) -> String {
    let atom_idx = atoms[0];
    let resname = system
        .interner
        .resolve(system.atoms.resname_id[atom_idx])
        .unwrap_or("RES");
    let resid = system.atoms.resid[atom_idx];
    let chain_raw = system
        .interner
        .resolve(system.atoms.chain_id[atom_idx])
        .unwrap_or("");
    let chain = if chain_raw.is_empty() {
        system.atoms.chain_id[atom_idx].to_string()
    } else {
        chain_raw.to_string()
    };
    match group_by {
        GroupBy::Atom => format!("{resname}{resid}-{}", atom_idx + 1),
        GroupBy::Resid => format!("{resname}{resid}"),
        GroupBy::Chain => format!("chain:{chain}"),
        GroupBy::ResidChain => format!("{resname}{resid}@{chain}"),
    }
}

fn charge_center(
    chunk: &FrameChunk,
    frame: usize,
    atoms: &[usize],
    charges: &[f64],
    group_charge: f64,
    length_scale: f64,
) -> [f64; 3] {
    let base = frame * chunk.n_atoms;
    let mut sum = [0.0f64; 3];
    for &atom_idx in atoms {
        let q = charges[atom_idx];
        let p = chunk.coords[base + atom_idx];
        sum[0] += q * p[0] as f64;
        sum[1] += q * p[1] as f64;
        sum[2] += q * p[2] as f64;
    }
    let inv = 1.0 / group_charge;
    [
        sum[0] * inv * length_scale,
        sum[1] * inv * length_scale,
        sum[2] * inv * length_scale,
    ]
}
