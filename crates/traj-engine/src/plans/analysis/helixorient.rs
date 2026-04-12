use nalgebra::{Matrix3, Vector3};
use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::pbc_utils::{apply_pbc, apply_pbc_triclinic, cell_and_inv_from_box};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, HelixOrientOutput, Plan, PlanOutput};
use crate::plans::geometry::utils::kabsch_rotation;

#[derive(Clone, Debug)]
struct HelixResidue {
    ca_idx: usize,
    sidechain_idx: Option<usize>,
}

pub struct HelixOrientPlan {
    ca_selection: Selection,
    sidechain_selection: Option<Selection>,
    residues: Vec<HelixResidue>,
    labels: Vec<String>,
    io_selection: Vec<u32>,
    use_selected_input: bool,
    incremental: bool,
    length_scale: f64,
    time: Vec<f32>,
    axis: Vec<f32>,
    center: Vec<f32>,
    residue_vector: Vec<f32>,
    normal: Vec<f32>,
    rise: Vec<f32>,
    radius: Vec<f32>,
    twist: Vec<f32>,
    bending: Vec<f32>,
    tilt: Vec<f32>,
    rotation: Vec<f32>,
    theta1: Vec<f32>,
    theta2: Vec<f32>,
    theta3: Vec<f32>,
    frames: usize,
    used_box: bool,
    use_sidechain: bool,
    first_frame_basis: Option<FrameBasis>,
    last_frame_basis: Option<FrameBasis>,
}

#[derive(Clone, Debug)]
struct FrameBasis {
    axis: Vec<[f64; 3]>,
    residue_vector: Vec<[f64; 3]>,
    normal: Vec<[f64; 3]>,
}

#[derive(Clone, Debug)]
struct BoxTransform {
    orthorhombic: Option<[f64; 3]>,
    triclinic: Option<([[f64; 3]; 3], [[f64; 3]; 3])>,
}

impl HelixOrientPlan {
    pub fn new(ca_selection: Selection) -> Self {
        Self {
            ca_selection,
            sidechain_selection: None,
            residues: Vec::new(),
            labels: Vec::new(),
            io_selection: Vec::new(),
            use_selected_input: false,
            incremental: false,
            length_scale: 1.0,
            time: Vec::new(),
            axis: Vec::new(),
            center: Vec::new(),
            residue_vector: Vec::new(),
            normal: Vec::new(),
            rise: Vec::new(),
            radius: Vec::new(),
            twist: Vec::new(),
            bending: Vec::new(),
            tilt: Vec::new(),
            rotation: Vec::new(),
            theta1: Vec::new(),
            theta2: Vec::new(),
            theta3: Vec::new(),
            frames: 0,
            used_box: false,
            use_sidechain: false,
            first_frame_basis: None,
            last_frame_basis: None,
        }
    }

    pub fn with_sidechain_selection(mut self, selection: Selection) -> Self {
        self.sidechain_selection = Some(selection);
        self
    }

    pub fn with_incremental(mut self, incremental: bool) -> Self {
        self.incremental = incremental;
        self
    }

    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    fn build_residues(&mut self, system: &System) -> TrajResult<()> {
        self.residues.clear();
        self.labels.clear();
        self.io_selection.clear();
        self.use_sidechain = self.sidechain_selection.is_some();

        let ca_indices = self.ca_selection.indices.as_ref();
        if ca_indices.len() < 4 {
            return Err(TrajError::Mismatch(
                "helixorient requires at least 4 Calpha atoms".into(),
            ));
        }
        let sidechain_indices = self
            .sidechain_selection
            .as_ref()
            .map(|sel| sel.indices.as_ref());
        if let Some(sidechain_indices) = sidechain_indices {
            if sidechain_indices.len() != ca_indices.len() {
                return Err(TrajError::Mismatch(
                    "helixorient sidechain selection must match Calpha selection length".into(),
                ));
            }
        }

        let n_atoms = system.n_atoms();
        let atoms = &system.atoms;
        let mut last_resid: Option<i32> = None;
        let mut chain_id: Option<u32> = None;
        let mut seen_ca = std::collections::HashSet::new();
        let mut seen_sidechain = std::collections::HashSet::new();

        for (i, &ca_idx_u32) in ca_indices.iter().enumerate() {
            let ca_idx = ca_idx_u32 as usize;
            if ca_idx >= n_atoms {
                return Err(TrajError::Mismatch(
                    "helixorient Calpha atom index out of bounds".into(),
                ));
            }
            if !seen_ca.insert(ca_idx) {
                return Err(TrajError::Mismatch(
                    "helixorient Calpha selection must not contain duplicate atoms".into(),
                ));
            }
            let atom_name = system.interner.resolve(atoms.name_id[ca_idx]).unwrap_or("");
            if !atom_name.eq_ignore_ascii_case("CA") {
                return Err(TrajError::Mismatch(
                    "helixorient selection must contain only Calpha atoms".into(),
                ));
            }
            let current_chain = atoms.chain_id[ca_idx];
            let current_resid = atoms.resid[ca_idx];
            if let Some(expected_chain) = chain_id {
                if current_chain != expected_chain {
                    return Err(TrajError::Mismatch(
                        "helixorient requires Calpha atoms from a single continuous chain".into(),
                    ));
                }
            } else {
                chain_id = Some(current_chain);
            }
            if let Some(prev_resid) = last_resid {
                if current_resid != prev_resid + 1 {
                    return Err(TrajError::Mismatch(
                        "helixorient requires Calpha atoms from consecutive residues".into(),
                    ));
                }
            }
            last_resid = Some(current_resid);

            let sidechain_idx = if let Some(sidechain_indices) = sidechain_indices {
                let idx = sidechain_indices[i] as usize;
                if idx >= n_atoms {
                    return Err(TrajError::Mismatch(
                        "helixorient sidechain atom index out of bounds".into(),
                    ));
                }
                if !seen_sidechain.insert(idx) {
                    return Err(TrajError::Mismatch(
                        "helixorient sidechain selection must not contain duplicate atoms".into(),
                    ));
                }
                if atoms.chain_id[idx] != current_chain || atoms.resid[idx] != current_resid {
                    return Err(TrajError::Mismatch(
                        "helixorient sidechain atoms must match Calpha residue identities".into(),
                    ));
                }
                if idx == ca_idx {
                    return Err(TrajError::Mismatch(
                        "helixorient sidechain atoms must be distinct from Calpha atoms".into(),
                    ));
                }
                Some(idx)
            } else {
                None
            };

            let resname = system
                .interner
                .resolve(atoms.resname_id[ca_idx])
                .unwrap_or("RES");
            let label = format!("{resname}:{current_resid}");
            self.labels.push(label.clone());
            self.residues.push(HelixResidue {
                ca_idx,
                sidechain_idx,
            });
        }

        self.io_selection
            .extend(self.residues.iter().map(|res| res.ca_idx as u32));
        if self.use_sidechain {
            self.io_selection.extend(
                self.residues
                    .iter()
                    .filter_map(|res| res.sidechain_idx.map(|idx| idx as u32)),
            );
        }
        Ok(())
    }

    fn process_chunk_impl(
        &mut self,
        chunk: &FrameChunk,
        ca_indices: &[usize],
        sidechain_indices: Option<&[usize]>,
    ) -> TrajResult<()> {
        let n_res = self.residues.len();
        if n_res == 0 {
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            let box_ = chunk.box_.get(frame).copied().unwrap_or(Box3::None);
            self.used_box |= !matches!(box_, Box3::None);
            let frame_time = chunk
                .time_ps
                .as_ref()
                .and_then(|values| values.get(frame).copied())
                .unwrap_or(self.frames as f32);
            self.time.push(frame_time);

            let state = compute_frame_state(chunk, frame, ca_indices, sidechain_indices, box_)?;
            append_vectors(&mut self.axis, &state.residue_axis);
            append_vectors_scaled(&mut self.center, &state.residue_origin, self.length_scale);
            append_vectors(&mut self.residue_vector, &state.residue_vector);
            append_vectors(&mut self.normal, &state.axis3);
            append_scalars_scaled(&mut self.rise, &state.residue_rise, self.length_scale);
            append_scalars_scaled(&mut self.radius, &state.residue_radius, self.length_scale);
            append_scalars(&mut self.twist, &state.residue_twist);
            append_scalars(&mut self.bending, &state.residue_bending);

            let basis = FrameBasis {
                axis: state.residue_axis.clone(),
                residue_vector: state.residue_vector.clone(),
                normal: state.axis3.clone(),
            };
            let zero = vec![0.0f32; n_res];
            if self.frames == 0 {
                self.tilt.extend_from_slice(&zero);
                self.rotation.extend_from_slice(&zero);
                self.theta1.extend_from_slice(&zero);
                self.theta2.extend_from_slice(&zero);
                self.theta3.extend_from_slice(&zero);
                self.first_frame_basis = Some(basis.clone());
            } else {
                let reference = if self.incremental {
                    self.last_frame_basis.as_ref().ok_or_else(|| {
                        TrajError::Mismatch("helixorient missing previous frame basis".into())
                    })?
                } else {
                    self.first_frame_basis.as_ref().ok_or_else(|| {
                        TrajError::Mismatch("helixorient missing first frame basis".into())
                    })?
                };
                let euler = compute_relative_euler(reference, &basis);
                self.tilt
                    .extend(euler.iter().map(|values| values[0] as f32));
                self.rotation
                    .extend(euler.iter().map(|values| values[1] as f32));
                self.theta1
                    .extend(euler.iter().map(|values| values[2] as f32));
                self.theta2
                    .extend(euler.iter().map(|values| values[3] as f32));
                self.theta3
                    .extend(euler.iter().map(|values| values[4] as f32));
            }
            self.last_frame_basis = Some(basis);
            self.frames += 1;
        }
        Ok(())
    }
}

impl Plan for HelixOrientPlan {
    fn name(&self) -> &'static str {
        "helixorient"
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        Some(self.io_selection.as_slice())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(self.io_selection.as_slice())
        } else {
            None
        }
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(TrajError::Parse(
                "helixorient length_scale must be finite and > 0".into(),
            ));
        }
        self.build_residues(system)?;
        self.use_selected_input = matches!(device, Device::Cpu);
        self.time.clear();
        self.axis.clear();
        self.center.clear();
        self.residue_vector.clear();
        self.normal.clear();
        self.rise.clear();
        self.radius.clear();
        self.twist.clear();
        self.bending.clear();
        self.tilt.clear();
        self.rotation.clear();
        self.theta1.clear();
        self.theta2.clear();
        self.theta3.clear();
        self.frames = 0;
        self.used_box = false;
        self.first_frame_basis = None;
        self.last_frame_basis = None;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let ca_indices: Vec<usize> = self.residues.iter().map(|res| res.ca_idx).collect();
        let sidechain_indices: Option<Vec<usize>> = if self.use_sidechain {
            Some(
                self.residues
                    .iter()
                    .map(|res| res.sidechain_idx.expect("validated sidechain selection"))
                    .collect(),
            )
        } else {
            None
        };
        self.process_chunk_impl(chunk, &ca_indices, sidechain_indices.as_deref())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        if !self.use_selected_input {
            return self.process_chunk(chunk, system, device);
        }
        if source_selection != self.io_selection.as_slice() {
            return Err(TrajError::Mismatch(
                "helixorient selected chunk does not match expected IO selection".into(),
            ));
        }
        let n_res = self.residues.len();
        let ca_indices: Vec<usize> = (0..n_res).collect();
        let sidechain_indices: Option<Vec<usize>> = if self.use_sidechain {
            Some((n_res..2 * n_res).collect())
        } else {
            None
        };
        self.process_chunk_impl(chunk, &ca_indices, sidechain_indices.as_deref())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let residues = self.residues.len();
        Ok(PlanOutput::HelixOrient(HelixOrientOutput {
            labels: self.labels.clone(),
            time: std::mem::take(&mut self.time),
            axis: std::mem::take(&mut self.axis),
            center: std::mem::take(&mut self.center),
            residue_vector: std::mem::take(&mut self.residue_vector),
            normal: std::mem::take(&mut self.normal),
            rise: std::mem::take(&mut self.rise),
            radius: std::mem::take(&mut self.radius),
            twist: std::mem::take(&mut self.twist),
            bending: std::mem::take(&mut self.bending),
            tilt: std::mem::take(&mut self.tilt),
            rotation: std::mem::take(&mut self.rotation),
            theta1: std::mem::take(&mut self.theta1),
            theta2: std::mem::take(&mut self.theta2),
            theta3: std::mem::take(&mut self.theta3),
            frames: self.frames,
            residues,
            use_sidechain: self.use_sidechain,
            incremental: self.incremental,
            used_box: self.used_box,
            length_scale: self.length_scale as f32,
        }))
    }
}

#[derive(Clone, Debug)]
struct FrameState {
    residue_axis: Vec<[f64; 3]>,
    residue_origin: Vec<[f64; 3]>,
    residue_vector: Vec<[f64; 3]>,
    axis3: Vec<[f64; 3]>,
    residue_rise: Vec<f64>,
    residue_radius: Vec<f64>,
    residue_twist: Vec<f64>,
    residue_bending: Vec<f64>,
}

fn compute_frame_state(
    chunk: &FrameChunk,
    frame: usize,
    ca_indices: &[usize],
    sidechain_indices: Option<&[usize]>,
    box_: Box3,
) -> TrajResult<FrameState> {
    let n_res = ca_indices.len();
    let transform = box_transform(box_);
    let ca_positions = unwrapped_positions(chunk, frame, ca_indices, &transform);
    let sidechain_positions = sidechain_indices.map(|indices| {
        mapped_positions(chunk, frame, indices, ca_indices, &ca_positions, &transform)
    });

    let mut helixaxis = vec![[0.0; 3]; n_res - 3];
    let mut twist = vec![0.0; n_res - 3];
    let mut radius = vec![0.0; n_res - 3];
    let mut rise = vec![0.0; n_res - 3];
    let mut residue_origin = vec![[0.0; 3]; n_res];

    for i in 0..(n_res - 3) {
        let r12 = sub(ca_positions[i + 1], ca_positions[i]);
        let r23 = sub(ca_positions[i + 2], ca_positions[i + 1]);
        let r34 = sub(ca_positions[i + 3], ca_positions[i + 2]);
        let diff13 = sub(r12, r23);
        let diff24 = sub(r23, r34);
        let axis = cross(diff13, diff24);
        helixaxis[i] = normalize(axis);
        let cos_d = clamp(dot(diff13, diff24) / (norm(diff13) * norm(diff24)));
        twist[i] = cos_d.acos().to_degrees();
        let denom = 2.0 * (1.0 - cos_d);
        radius[i] = if denom.abs() < 1e-12 {
            0.0
        } else {
            (norm(diff13) * norm(diff24)).sqrt() / denom
        };
        rise[i] = dot(r23, helixaxis[i]).abs();
        let v1 = mul(diff13, safe_scale(radius[i], norm(diff13)));
        let v2 = mul(diff24, safe_scale(radius[i], norm(diff24)));
        residue_origin[i + 1] = sub(ca_positions[i + 1], v1);
        residue_origin[i + 2] = sub(ca_positions[i + 2], v2);
    }

    let mut residue_radius = vec![0.0; n_res];
    let mut residue_twist = vec![0.0; n_res];
    let mut residue_rise = vec![0.0; n_res];
    let mut residue_bending = vec![0.0; n_res];
    if n_res >= 2 {
        residue_radius[1] = radius[0];
        residue_twist[1] = twist[0];
        residue_rise[1] = rise[0];
    }
    for i in 2..(n_res - 2) {
        residue_radius[i] = 0.5 * (radius[i - 2] + radius[i - 1]);
        residue_twist[i] = 0.5 * (twist[i - 2] + twist[i - 1]);
        residue_rise[i] = 0.5 * (rise[i - 2] + rise[i - 1]);
        residue_bending[i] = clamp(dot(helixaxis[i - 2], helixaxis[i - 1]))
            .acos()
            .to_degrees();
    }
    residue_radius[n_res - 2] = radius[n_res - 4];
    residue_twist[n_res - 2] = twist[n_res - 4];
    residue_rise[n_res - 2] = rise[n_res - 4];
    residue_origin[0] = [0.0, 0.0, 0.0];
    residue_origin[n_res - 1] = [0.0, 0.0, 0.0];

    let mut residue_axis = vec![[0.0; 3]; n_res];
    residue_axis[0] = helixaxis[0];
    residue_axis[1] = helixaxis[0];
    for i in 2..(n_res - 2) {
        residue_axis[i] = mul(add(helixaxis[i - 2], helixaxis[i - 1]), 0.5);
    }
    residue_axis[n_res - 2] = helixaxis[n_res - 4];
    residue_axis[n_res - 1] = helixaxis[n_res - 4];
    for axis in residue_axis.iter_mut() {
        *axis = normalize(*axis);
    }

    let mut residue_vector = vec![[0.0; 3]; n_res];
    let mut axis3 = vec![[0.0; 3]; n_res];
    for i in 1..(n_res - 1) {
        let target = if let Some(sidechain_positions) = sidechain_positions.as_ref() {
            sidechain_positions[i]
        } else {
            ca_positions[i]
        };
        residue_vector[i] = normalize(sub(target, residue_origin[i]));
        axis3[i] = cross(residue_axis[i], residue_vector[i]);
    }

    Ok(FrameState {
        residue_axis,
        residue_origin,
        residue_vector,
        axis3,
        residue_rise,
        residue_radius,
        residue_twist,
        residue_bending,
    })
}

fn compute_relative_euler(reference: &FrameBasis, current: &FrameBasis) -> Vec<[f64; 5]> {
    let n_res = reference.axis.len().min(current.axis.len());
    let unitaxes_f32 = [
        [1.0f32, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ];
    let mut out = vec![[0.0; 5]; n_res];
    for i in 1..n_res.saturating_sub(1) {
        let ref_axes = [
            as_f32(reference.axis[i]),
            as_f32(reference.residue_vector[i]),
            as_f32(reference.normal[i]),
        ];
        let new_axes = [
            as_f32(current.axis[i]),
            as_f32(current.residue_vector[i]),
            as_f32(current.normal[i]),
        ];
        let (align_ref, _, _) = kabsch_rotation(&ref_axes, &unitaxes_f32);
        let rot_ref = apply_rotation_set(&align_ref, &ref_axes);
        let rot_new = apply_rotation_set(&align_ref, &new_axes);
        let (a, _, _) = kabsch_rotation(&rot_new, &rot_ref);
        let theta1 = a[(0, 2)].atan2(a[(0, 0)]).to_degrees();
        let theta2 = (-a[(0, 1)]).asin().to_degrees();
        let theta3 = a[(2, 1)].atan2(a[(1, 1)]).to_degrees();
        let tilt = (theta1 * theta1 + theta2 * theta2).sqrt();
        out[i] = [tilt, theta3, theta1, theta2, theta3];
    }
    out
}

fn apply_rotation_set(rotation: &Matrix3<f64>, points: &[[f32; 4]; 3]) -> [[f32; 4]; 3] {
    let mut out = [[0.0f32; 4]; 3];
    for (dst, src) in out.iter_mut().zip(points.iter()) {
        let v = Vector3::new(src[0] as f64, src[1] as f64, src[2] as f64);
        let rotated = rotation * v;
        *dst = [rotated[0] as f32, rotated[1] as f32, rotated[2] as f32, 0.0];
    }
    out
}

fn unwrapped_positions(
    chunk: &FrameChunk,
    frame: usize,
    indices: &[usize],
    transform: &BoxTransform,
) -> Vec<[f64; 3]> {
    let n_atoms = chunk.n_atoms;
    let base = frame * n_atoms;
    let mut positions = Vec::with_capacity(indices.len());
    for (i, &idx) in indices.iter().enumerate() {
        let raw = point(chunk, base + idx);
        if i == 0 {
            positions.push(raw);
            continue;
        }
        let prev_raw = point(chunk, base + indices[i - 1]);
        let prev_unwrapped = positions[i - 1];
        let mut delta = sub(raw, prev_raw);
        apply_minimum_image(&mut delta, transform);
        positions.push(add(prev_unwrapped, delta));
    }
    positions
}

fn mapped_positions(
    chunk: &FrameChunk,
    frame: usize,
    indices: &[usize],
    reference_indices: &[usize],
    reference_positions: &[[f64; 3]],
    transform: &BoxTransform,
) -> Vec<[f64; 3]> {
    let n_atoms = chunk.n_atoms;
    let base = frame * n_atoms;
    let mut positions = Vec::with_capacity(indices.len());
    for ((&idx, &reference_idx), &reference_pos) in indices
        .iter()
        .zip(reference_indices.iter())
        .zip(reference_positions.iter())
    {
        let raw = point(chunk, base + idx);
        let reference_raw = point(chunk, base + reference_idx);
        let mut delta = sub(raw, reference_raw);
        apply_minimum_image(&mut delta, transform);
        positions.push(add(reference_pos, delta));
    }
    positions
}

fn box_transform(box_: Box3) -> BoxTransform {
    match box_ {
        Box3::Orthorhombic { lx, ly, lz } => BoxTransform {
            orthorhombic: Some([lx as f64, ly as f64, lz as f64]),
            triclinic: None,
        },
        Box3::Triclinic { .. } => match cell_and_inv_from_box(box_) {
            Ok((cell, inv)) => BoxTransform {
                orthorhombic: None,
                triclinic: Some((cell, inv)),
            },
            Err(_) => BoxTransform {
                orthorhombic: None,
                triclinic: None,
            },
        },
        Box3::None => BoxTransform {
            orthorhombic: None,
            triclinic: None,
        },
    }
}

fn apply_minimum_image(delta: &mut [f64; 3], transform: &BoxTransform) {
    let [mut dx, mut dy, mut dz] = *delta;
    if let Some([lx, ly, lz]) = transform.orthorhombic {
        apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
    } else if let Some((cell, inv)) = transform.triclinic.as_ref() {
        apply_pbc_triclinic(&mut dx, &mut dy, &mut dz, cell, inv);
    }
    *delta = [dx, dy, dz];
}

fn append_vectors(out: &mut Vec<f32>, vectors: &[[f64; 3]]) {
    out.extend(
        vectors
            .iter()
            .flat_map(|value| value.iter().map(|&x| x as f32)),
    );
}

fn append_vectors_scaled(out: &mut Vec<f32>, vectors: &[[f64; 3]], scale: f64) {
    out.extend(
        vectors
            .iter()
            .flat_map(|value| value.iter().map(move |&x| (x * scale) as f32)),
    );
}

fn append_scalars(out: &mut Vec<f32>, values: &[f64]) {
    out.extend(values.iter().map(|&x| x as f32));
}

fn append_scalars_scaled(out: &mut Vec<f32>, values: &[f64], scale: f64) {
    out.extend(values.iter().map(|&x| (x * scale) as f32));
}

fn as_f32(value: [f64; 3]) -> [f32; 4] {
    [value[0] as f32, value[1] as f32, value[2] as f32, 0.0]
}

fn safe_scale(value: f64, norm: f64) -> f64 {
    if norm <= 1e-12 {
        0.0
    } else {
        value / norm
    }
}

fn clamp(value: f64) -> f64 {
    value.clamp(-1.0, 1.0)
}

fn point(chunk: &FrameChunk, idx: usize) -> [f64; 3] {
    let p = chunk.coords[idx];
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn mul(a: [f64; 3], scale: f64) -> [f64; 3] {
    [a[0] * scale, a[1] * scale, a[2] * scale]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn normalize(a: [f64; 3]) -> [f64; 3] {
    let len = norm(a);
    if len <= 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        mul(a, 1.0 / len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helixorient_relative_euler_reports_axis_rotation() {
        let reference = FrameBasis {
            axis: vec![[0.0, 0.0, 1.0]; 5],
            residue_vector: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            normal: vec![
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        };
        let angle = 20.0f64.to_radians();
        let current = FrameBasis {
            axis: vec![[0.0, 0.0, 1.0]; 5],
            residue_vector: vec![
                [0.0, 0.0, 0.0],
                [angle.cos(), angle.sin(), 0.0],
                [angle.cos(), angle.sin(), 0.0],
                [angle.cos(), angle.sin(), 0.0],
                [0.0, 0.0, 0.0],
            ],
            normal: vec![
                [0.0, 0.0, 0.0],
                [-angle.sin(), angle.cos(), 0.0],
                [-angle.sin(), angle.cos(), 0.0],
                [-angle.sin(), angle.cos(), 0.0],
                [0.0, 0.0, 0.0],
            ],
        };
        let euler = compute_relative_euler(&reference, &current);
        assert!(euler[2][0].abs() < 1e-3);
        assert!((euler[2][1].abs() - 20.0).abs() < 1e-2);
    }
}
