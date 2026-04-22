use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, MdmatOutput, Plan, PlanOutput, PlanRequirements};
use crate::plans::geometry::geometry_math::{
    apply_pbc, apply_pbc_triclinic, cell_and_inv_from_box,
};

#[derive(Clone, Debug)]
struct ResidueGroup {
    label: String,
    global_atoms: Vec<usize>,
    selected_atoms: Vec<usize>,
    atom_slots: Vec<usize>,
}

#[derive(Clone, Debug)]
enum BoxTransform {
    None,
    Orthorhombic([f64; 3]),
    Triclinic {
        cell: [[f64; 3]; 3],
        inv: [[f64; 3]; 3],
    },
}

pub struct MdmatPlan {
    selection: Selection,
    truncate: f64,
    include_contacts: bool,
    include_frames: bool,
    length_scale: f64,
    residues: Vec<ResidueGroup>,
    io_selection: Vec<u32>,
    use_selected_input: bool,
    mean_matrix_sum: Vec<f64>,
    frame_matrices: Vec<f32>,
    time: Vec<f32>,
    contact_frame_counts: Vec<u32>,
    frames: usize,
    used_box: bool,
}

impl MdmatPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            truncate: 1.5,
            include_contacts: false,
            include_frames: false,
            length_scale: 1.0,
            residues: Vec::new(),
            io_selection: Vec::new(),
            use_selected_input: false,
            mean_matrix_sum: Vec::new(),
            frame_matrices: Vec::new(),
            time: Vec::new(),
            contact_frame_counts: Vec::new(),
            frames: 0,
            used_box: false,
        }
    }

    pub fn with_truncate(mut self, truncate: f64) -> Self {
        self.truncate = truncate;
        self
    }

    pub fn with_include_contacts(mut self, include_contacts: bool) -> Self {
        self.include_contacts = include_contacts;
        self
    }

    pub fn with_include_frames(mut self, include_frames: bool) -> Self {
        self.include_frames = include_frames;
        self
    }

    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    fn build_residue_groups(&mut self, system: &System) -> TrajResult<()> {
        if !self.truncate.is_finite() || self.truncate <= 0.0 {
            return Err(TrajError::Parse(
                "mdmat truncate must be finite and > 0".into(),
            ));
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(TrajError::Parse(
                "mdmat length_scale must be finite and > 0".into(),
            ));
        }

        self.residues.clear();
        self.io_selection = self.selection.indices.as_ref().clone();
        self.use_selected_input = !self.io_selection.is_empty();
        if self.io_selection.is_empty() {
            return Ok(());
        }

        let n_atoms = system.n_atoms();
        let atoms = &system.atoms;
        let mut slot_by_atom = vec![usize::MAX; n_atoms];
        for (slot, &idx) in self.io_selection.iter().enumerate() {
            let atom_idx = idx as usize;
            if atom_idx < n_atoms {
                slot_by_atom[atom_idx] = slot;
            }
        }

        let mut start = 0usize;
        while start < n_atoms {
            let chain = atoms.chain_id[start];
            let resid = atoms.resid[start];
            let resname_id = atoms.resname_id[start];
            let mut end = start + 1;
            while end < n_atoms
                && atoms.chain_id[end] == chain
                && atoms.resid[end] == resid
                && atoms.resname_id[end] == resname_id
            {
                end += 1;
            }

            let mut global_atoms = Vec::new();
            let mut selected_atoms = Vec::new();
            let mut atom_slots = Vec::new();
            for idx in start..end {
                let slot = slot_by_atom[idx];
                if slot != usize::MAX {
                    global_atoms.push(idx);
                    selected_atoms.push(slot);
                    atom_slots.push(slot);
                }
            }
            if !global_atoms.is_empty() {
                let resname = system.interner.resolve(resname_id).unwrap_or("RES");
                self.residues.push(ResidueGroup {
                    label: format!("{resname}:{resid}"),
                    global_atoms,
                    selected_atoms,
                    atom_slots,
                });
            }
            start = end;
        }
        Ok(())
    }

    fn process_chunk_impl(
        &mut self,
        chunk: &FrameChunk,
        use_selected_atoms: bool,
    ) -> TrajResult<()> {
        let n_res = self.residues.len();
        if n_res == 0 {
            return Ok(());
        }
        let n_slots = self.io_selection.len();
        let trunc2 = (self.truncate / self.length_scale).powi(2);

        for frame in 0..chunk.n_frames {
            let box_ = chunk.box_.get(frame).copied().unwrap_or(Box3::None);
            self.used_box |= !matches!(box_, Box3::None);
            let transform = box_transform(box_)?;
            let time = chunk
                .time_ps
                .as_ref()
                .and_then(|values| values.get(frame).copied())
                .unwrap_or(self.frames as f32);
            let matrix_start = self.frame_matrices.len();
            let mut matrix = if self.include_frames {
                self.frame_matrices
                    .resize(matrix_start + n_res * n_res, 0.0);
                Some(&mut self.frame_matrices[matrix_start..matrix_start + n_res * n_res])
            } else {
                None
            };
            if self.include_frames {
                self.time.push(time);
            }

            let mut contacts_this_frame = if self.include_contacts {
                Some(vec![0u8; n_res * n_slots])
            } else {
                None
            };

            for i in 0..n_res {
                let diag_idx = i * n_res + i;
                if let Some(frame_matrix) = matrix.as_deref_mut() {
                    frame_matrix[diag_idx] = 0.0;
                }
                self.mean_matrix_sum[diag_idx] += 0.0;
            }

            for res_i in 0..n_res {
                let atoms_i = if use_selected_atoms {
                    &self.residues[res_i].selected_atoms
                } else {
                    &self.residues[res_i].global_atoms
                };
                for res_j in (res_i + 1)..n_res {
                    let atoms_j = if use_selected_atoms {
                        &self.residues[res_j].selected_atoms
                    } else {
                        &self.residues[res_j].global_atoms
                    };
                    let mut min_r2 = f64::INFINITY;
                    for (local_i, &atom_i) in atoms_i.iter().enumerate() {
                        let coord_i = frame_coord(chunk, frame, atom_i);
                        let slot_i = self.residues[res_i].atom_slots[local_i];
                        for (local_j, &atom_j) in atoms_j.iter().enumerate() {
                            let coord_j = frame_coord(chunk, frame, atom_j);
                            let slot_j = self.residues[res_j].atom_slots[local_j];
                            let r2 = distance2(coord_i, coord_j, &transform);
                            if r2 < min_r2 {
                                min_r2 = r2;
                            }
                            if let Some(frame_contacts) = contacts_this_frame.as_mut() {
                                if r2 < trunc2 {
                                    frame_contacts[res_i * n_slots + slot_j] = 1;
                                    frame_contacts[res_j * n_slots + slot_i] = 1;
                                }
                            }
                        }
                    }
                    let dist = (min_r2.sqrt() * self.length_scale) as f32;
                    let ij = res_i * n_res + res_j;
                    let ji = res_j * n_res + res_i;
                    self.mean_matrix_sum[ij] += dist as f64;
                    self.mean_matrix_sum[ji] += dist as f64;
                    if let Some(frame_matrix) = matrix.as_deref_mut() {
                        frame_matrix[ij] = dist;
                        frame_matrix[ji] = dist;
                    }
                }
            }

            if let Some(frame_contacts) = contacts_this_frame.as_ref() {
                for idx in 0..frame_contacts.len() {
                    if frame_contacts[idx] != 0 {
                        self.contact_frame_counts[idx] += 1;
                    }
                }
            }
            self.frames += 1;
        }
        Ok(())
    }
}

impl Plan for MdmatPlan {
    fn name(&self) -> &'static str {
        "mdmat"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(true, false)
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.mean_matrix_sum.clear();
        self.frame_matrices.clear();
        self.time.clear();
        self.contact_frame_counts.clear();
        self.frames = 0;
        self.used_box = false;
        self.build_residue_groups(system)?;
        let n_res = self.residues.len();
        self.mean_matrix_sum.resize(n_res * n_res, 0.0);
        if self.include_contacts {
            self.contact_frame_counts
                .resize(n_res * self.io_selection.len(), 0);
        }
        Ok(())
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(&self.io_selection)
        } else {
            None
        }
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(&self.io_selection)
        } else {
            None
        }
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        self.process_chunk_impl(chunk, false)
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        _source_selection: &[u32],
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        self.process_chunk_impl(chunk, true)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let n_res = self.residues.len();
        let mut mean_matrix = vec![0.0f32; n_res * n_res];
        if self.frames > 0 {
            let inv_frames = 1.0 / self.frames as f64;
            for (dst, src) in mean_matrix.iter_mut().zip(self.mean_matrix_sum.iter()) {
                *dst = (src * inv_frames) as f32;
            }
        }

        let mut distinct_contact_atoms = Vec::new();
        let mut mean_contact_atoms = Vec::new();
        let mut contact_ratio = Vec::new();
        let mut residue_atom_counts = self
            .residues
            .iter()
            .map(|group| group.global_atoms.len() as u32)
            .collect::<Vec<_>>();
        let mut mean_contact_atoms_per_residue_atom = Vec::new();

        if self.include_contacts {
            let n_slots = self.io_selection.len();
            distinct_contact_atoms.reserve(n_res);
            mean_contact_atoms.reserve(n_res);
            contact_ratio.reserve(n_res);
            mean_contact_atoms_per_residue_atom.reserve(n_res);
            for res in 0..n_res {
                let row = &self.contact_frame_counts[res * n_slots..(res + 1) * n_slots];
                let total = row.iter().filter(|&&count| count != 0).count() as u32;
                let mean = if self.frames == 0 {
                    0.0
                } else {
                    row.iter().map(|&count| count as f64).sum::<f64>() / self.frames as f64
                };
                let ratio = if mean == 0.0 {
                    1.0
                } else {
                    total as f64 / mean
                };
                let natm = residue_atom_counts.get(res).copied().unwrap_or(0);
                distinct_contact_atoms.push(total);
                mean_contact_atoms.push(mean as f32);
                contact_ratio.push(ratio as f32);
                mean_contact_atoms_per_residue_atom.push(if natm == 0 {
                    0.0
                } else {
                    (mean / natm as f64) as f32
                });
            }
        } else {
            residue_atom_counts.clear();
        }

        Ok(PlanOutput::Mdmat(MdmatOutput {
            labels: self
                .residues
                .iter()
                .map(|group| group.label.clone())
                .collect(),
            mean_matrix,
            time: std::mem::take(&mut self.time),
            frame_matrices: std::mem::take(&mut self.frame_matrices),
            frames: self.frames,
            residues: n_res,
            truncate: self.truncate as f32,
            used_box: self.used_box,
            length_scale: self.length_scale as f32,
            distinct_contact_atoms,
            mean_contact_atoms,
            contact_ratio,
            residue_atom_counts,
            mean_contact_atoms_per_residue_atom,
        }))
    }
}

fn box_transform(box_: Box3) -> TrajResult<BoxTransform> {
    match box_ {
        Box3::None => Ok(BoxTransform::None),
        Box3::Orthorhombic { lx, ly, lz } => Ok(BoxTransform::Orthorhombic([
            lx as f64, ly as f64, lz as f64,
        ])),
        _ => {
            let (cell, inv) = cell_and_inv_from_box(box_)?;
            Ok(BoxTransform::Triclinic { cell, inv })
        }
    }
}

fn frame_coord(chunk: &FrameChunk, frame: usize, atom: usize) -> [f32; 4] {
    chunk.coords[frame * chunk.n_atoms + atom]
}

fn distance2(a: [f32; 4], b: [f32; 4], transform: &BoxTransform) -> f64 {
    let mut dx = b[0] as f64 - a[0] as f64;
    let mut dy = b[1] as f64 - a[1] as f64;
    let mut dz = b[2] as f64 - a[2] as f64;
    match transform {
        BoxTransform::None => {}
        BoxTransform::Orthorhombic(lengths) => {
            apply_pbc(
                &mut dx, &mut dy, &mut dz, lengths[0], lengths[1], lengths[2],
            );
        }
        BoxTransform::Triclinic { cell, inv } => {
            apply_pbc_triclinic(&mut dx, &mut dy, &mut dz, cell, inv);
        }
    }
    dx * dx + dy * dy + dz * dz
}
