use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{BundleOutput, Device, Plan, PlanOutput};
use crate::plans::analysis::secondary_structure::{
    add, clamp, cross, dot, mul, norm, normalize, sub,
};
use crate::plans::geometry::utils::{apply_pbc, apply_pbc_triclinic, cell_and_inv_from_box};

#[derive(Clone, Debug)]
struct AxisGroups {
    top: Vec<usize>,
    bottom: Vec<usize>,
    kink: Option<Vec<usize>>,
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

pub struct BundlePlan {
    top_selection: Selection,
    bottom_selection: Selection,
    kink_selection: Option<Selection>,
    n_axes: usize,
    use_z_reference: bool,
    mass_weighted: bool,
    length_scale: f64,
    labels: Vec<String>,
    groups: Vec<AxisGroups>,
    io_selection: Vec<u32>,
    io_masses: Vec<f32>,
    use_selected_input: bool,
    time: Vec<f32>,
    reference_axis: Vec<f32>,
    top: Vec<f32>,
    bottom: Vec<f32>,
    mid: Vec<f32>,
    direction: Vec<f32>,
    length: Vec<f32>,
    distance: Vec<f32>,
    z_shift: Vec<f32>,
    tilt: Vec<f32>,
    radial_tilt: Vec<f32>,
    lateral_tilt: Vec<f32>,
    kink: Vec<f32>,
    kink_angle: Vec<f32>,
    kink_radial: Vec<f32>,
    kink_lateral: Vec<f32>,
    frames: usize,
    used_box: bool,
}

impl BundlePlan {
    pub fn new(top_selection: Selection, bottom_selection: Selection, n_axes: usize) -> Self {
        Self {
            top_selection,
            bottom_selection,
            kink_selection: None,
            n_axes,
            use_z_reference: false,
            mass_weighted: true,
            length_scale: 1.0,
            labels: Vec::new(),
            groups: Vec::new(),
            io_selection: Vec::new(),
            io_masses: Vec::new(),
            use_selected_input: false,
            time: Vec::new(),
            reference_axis: Vec::new(),
            top: Vec::new(),
            bottom: Vec::new(),
            mid: Vec::new(),
            direction: Vec::new(),
            length: Vec::new(),
            distance: Vec::new(),
            z_shift: Vec::new(),
            tilt: Vec::new(),
            radial_tilt: Vec::new(),
            lateral_tilt: Vec::new(),
            kink: Vec::new(),
            kink_angle: Vec::new(),
            kink_radial: Vec::new(),
            kink_lateral: Vec::new(),
            frames: 0,
            used_box: false,
        }
    }

    pub fn with_kink_selection(mut self, kink_selection: Selection) -> Self {
        self.kink_selection = Some(kink_selection);
        self
    }

    pub fn with_use_z_reference(mut self, use_z_reference: bool) -> Self {
        self.use_z_reference = use_z_reference;
        self
    }

    pub fn with_mass_weighted(mut self, mass_weighted: bool) -> Self {
        self.mass_weighted = mass_weighted;
        self
    }

    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    fn build_groups(&mut self, system: &System) -> TrajResult<()> {
        if self.n_axes == 0 {
            return Err(TrajError::Parse("bundle n_axes must be > 0".into()));
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(TrajError::Parse(
                "bundle length_scale must be finite and > 0".into(),
            ));
        }
        let top = self.top_selection.indices.as_ref();
        let bottom = self.bottom_selection.indices.as_ref();
        if top.is_empty() || bottom.is_empty() {
            return Err(TrajError::Mismatch(
                "bundle top and bottom selections must be non-empty".into(),
            ));
        }
        if top.len() % self.n_axes != 0 || bottom.len() % self.n_axes != 0 {
            return Err(TrajError::Mismatch(
                "bundle selection sizes must be divisible by n_axes".into(),
            ));
        }
        let kink = self.kink_selection.as_ref().map(|sel| sel.indices.as_ref());
        if let Some(kink) = kink {
            if kink.is_empty() || kink.len() % self.n_axes != 0 {
                return Err(TrajError::Mismatch(
                    "bundle kink selection size must be divisible by n_axes".into(),
                ));
            }
        }
        self.labels = (0..self.n_axes)
            .map(|i| format!("axis_{}", i + 1))
            .collect();
        self.groups.clear();
        self.io_selection.clear();
        self.io_masses.clear();

        let top_chunk = top.len() / self.n_axes;
        let bottom_chunk = bottom.len() / self.n_axes;
        let kink_chunk = kink.map(|values| values.len() / self.n_axes);
        for axis in 0..self.n_axes {
            let top_group: Vec<usize> = top[axis * top_chunk..(axis + 1) * top_chunk]
                .iter()
                .map(|&idx| idx as usize)
                .collect();
            let bottom_group: Vec<usize> = bottom[axis * bottom_chunk..(axis + 1) * bottom_chunk]
                .iter()
                .map(|&idx| idx as usize)
                .collect();
            let kink_group = match (kink, kink_chunk) {
                (Some(values), Some(chunk)) => Some(
                    values[axis * chunk..(axis + 1) * chunk]
                        .iter()
                        .map(|&idx| idx as usize)
                        .collect(),
                ),
                _ => None,
            };
            self.groups.push(AxisGroups {
                top: top_group,
                bottom: bottom_group,
                kink: kink_group,
            });
        }

        for &idx in top.iter() {
            self.io_selection.push(idx);
            self.io_masses.push(system.atoms.mass[idx as usize]);
        }
        for &idx in bottom.iter() {
            self.io_selection.push(idx);
            self.io_masses.push(system.atoms.mass[idx as usize]);
        }
        if let Some(kink) = kink {
            for &idx in kink.iter() {
                self.io_selection.push(idx);
                self.io_masses.push(system.atoms.mass[idx as usize]);
            }
        }
        Ok(())
    }

    fn process_chunk_impl(
        &mut self,
        chunk: &FrameChunk,
        groups: &[AxisGroups],
        masses: &[f32],
    ) -> TrajResult<()> {
        if groups.is_empty() {
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            let box_ = chunk.box_.get(frame).copied().unwrap_or(Box3::None);
            self.used_box |= !matches!(box_, Box3::None);
            let transform = box_transform(box_)?;
            let time = chunk
                .time_ps
                .as_ref()
                .and_then(|values| values.get(frame).copied())
                .unwrap_or(self.frames as f32);
            self.time.push(time);

            let mut tops = Vec::with_capacity(self.n_axes);
            let mut bottoms = Vec::with_capacity(self.n_axes);
            let mut kinks = if self.kink_selection.is_some() {
                Some(Vec::with_capacity(self.n_axes))
            } else {
                None
            };

            for group in groups.iter() {
                tops.push(group_center(
                    chunk,
                    frame,
                    &group.top,
                    masses,
                    self.mass_weighted,
                    &transform,
                ));
                bottoms.push(group_center(
                    chunk,
                    frame,
                    &group.bottom,
                    masses,
                    self.mass_weighted,
                    &transform,
                ));
                if let (Some(group_kink), Some(kinks_vec)) = (group.kink.as_ref(), kinks.as_mut()) {
                    kinks_vec.push(group_center(
                        chunk,
                        frame,
                        group_kink,
                        masses,
                        self.mass_weighted,
                        &transform,
                    ));
                }
            }

            let avg_top = average_point(&tops);
            let avg_bottom = average_point(&bottoms);
            let center = mul(add(avg_top, avg_bottom), 0.5);
            for point in tops.iter_mut() {
                *point = sub(*point, center);
            }
            for point in bottoms.iter_mut() {
                *point = sub(*point, center);
            }
            if let Some(kinks_vec) = kinks.as_mut() {
                for point in kinks_vec.iter_mut() {
                    *point = sub(*point, center);
                }
            }

            let mut ref_axis = normalize(sub(avg_top, center));
            let original_ref_axis = ref_axis;
            if !self.use_z_reference {
                rotate_bundle_points(&mut tops, &mut bottoms, kinks.as_mut(), &mut ref_axis);
            }
            self.reference_axis
                .extend_from_slice(&vector_to_f32(original_ref_axis));

            for axis in 0..self.n_axes {
                let top = tops[axis];
                let bottom = bottoms[axis];
                let mid = mul(add(top, bottom), 0.5);
                let dir_raw = sub(top, bottom);
                let len = norm(dir_raw);
                let dir = normalize(dir_raw);
                self.top
                    .extend_from_slice(&scaled_vector_to_f32(top, self.length_scale));
                self.bottom
                    .extend_from_slice(&scaled_vector_to_f32(bottom, self.length_scale));
                self.mid
                    .extend_from_slice(&scaled_vector_to_f32(mid, self.length_scale));
                self.direction.extend_from_slice(&vector_to_f32(dir));
                self.length.push((len * self.length_scale) as f32);
                self.distance.push((norm(mid) * self.length_scale) as f32);
                self.z_shift.push((mid[2] * self.length_scale) as f32);
                self.tilt.push(clamp(dir[2]).acos().to_degrees() as f32);
                self.radial_tilt
                    .push(signed_tilt(mid[0] * dir[0] + mid[1] * dir[1], dir[2]) as f32);
                self.lateral_tilt
                    .push(signed_tilt(mid[1] * dir[0] - mid[0] * dir[1], dir[2]) as f32);

                if let Some(kinks_vec) = kinks.as_ref() {
                    let kink = kinks_vec[axis];
                    self.kink
                        .extend_from_slice(&scaled_vector_to_f32(kink, self.length_scale));
                    let va = normalize(sub(top, kink));
                    let vb = normalize(sub(kink, bottom));
                    let vc = cross(va, vb);
                    let vr = normalize([mid[0], mid[1], 0.0]);
                    let vl = [vr[1], -vr[0], 0.0];
                    self.kink_angle
                        .push(clamp(dot(va, vb)).acos().to_degrees() as f32);
                    self.kink_radial
                        .push(clamp(dot(vc, vr)).asin().to_degrees() as f32);
                    self.kink_lateral
                        .push(clamp(dot(vc, vl)).asin().to_degrees() as f32);
                }
            }
            self.frames += 1;
        }
        Ok(())
    }
}

impl Plan for BundlePlan {
    fn name(&self) -> &'static str {
        "bundle"
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
        self.build_groups(system)?;
        self.use_selected_input = matches!(device, Device::Cpu);
        self.time.clear();
        self.reference_axis.clear();
        self.top.clear();
        self.bottom.clear();
        self.mid.clear();
        self.direction.clear();
        self.length.clear();
        self.distance.clear();
        self.z_shift.clear();
        self.tilt.clear();
        self.radial_tilt.clear();
        self.lateral_tilt.clear();
        self.kink.clear();
        self.kink_angle.clear();
        self.kink_radial.clear();
        self.kink_lateral.clear();
        self.frames = 0;
        self.used_box = false;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let groups = self.groups.clone();
        self.process_chunk_impl(chunk, &groups, &system.atoms.mass)
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
                "bundle selected chunk does not match expected IO selection".into(),
            ));
        }
        let top_len = self.top_selection.indices.len();
        let bottom_len = self.bottom_selection.indices.len();
        let top_chunk = top_len / self.n_axes;
        let bottom_chunk = bottom_len / self.n_axes;
        let kink_len = self
            .kink_selection
            .as_ref()
            .map(|sel| sel.indices.len())
            .unwrap_or(0);
        let kink_chunk = if kink_len > 0 {
            Some(kink_len / self.n_axes)
        } else {
            None
        };
        let mut groups = Vec::with_capacity(self.n_axes);
        for axis in 0..self.n_axes {
            let top = (axis * top_chunk..(axis + 1) * top_chunk).collect();
            let bottom_offset = top_len;
            let bottom = (bottom_offset + axis * bottom_chunk
                ..bottom_offset + (axis + 1) * bottom_chunk)
                .collect();
            let kink = kink_chunk.map(|chunk| {
                let offset = top_len + bottom_len;
                (offset + axis * chunk..offset + (axis + 1) * chunk).collect()
            });
            groups.push(AxisGroups { top, bottom, kink });
        }
        let masses = self.io_masses.clone();
        self.process_chunk_impl(chunk, &groups, &masses)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Bundle(BundleOutput {
            labels: self.labels.clone(),
            time: std::mem::take(&mut self.time),
            reference_axis: std::mem::take(&mut self.reference_axis),
            top: std::mem::take(&mut self.top),
            bottom: std::mem::take(&mut self.bottom),
            mid: std::mem::take(&mut self.mid),
            direction: std::mem::take(&mut self.direction),
            length: std::mem::take(&mut self.length),
            distance: std::mem::take(&mut self.distance),
            z_shift: std::mem::take(&mut self.z_shift),
            tilt: std::mem::take(&mut self.tilt),
            radial_tilt: std::mem::take(&mut self.radial_tilt),
            lateral_tilt: std::mem::take(&mut self.lateral_tilt),
            kink: std::mem::take(&mut self.kink),
            kink_angle: std::mem::take(&mut self.kink_angle),
            kink_radial: std::mem::take(&mut self.kink_radial),
            kink_lateral: std::mem::take(&mut self.kink_lateral),
            frames: self.frames,
            axes: self.n_axes,
            has_kink: self.kink_selection.is_some(),
            use_z_reference: self.use_z_reference,
            mass_weighted: self.mass_weighted,
            used_box: self.used_box,
            length_scale: self.length_scale as f32,
        }))
    }
}

fn point(chunk: &FrameChunk, frame: usize, idx: usize) -> [f64; 3] {
    let p = chunk.coords[frame * chunk.n_atoms + idx];
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

fn average_point(points: &[[f64; 3]]) -> [f64; 3] {
    if points.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let mut sum = [0.0; 3];
    for point in points.iter() {
        sum = add(sum, *point);
    }
    mul(sum, 1.0 / points.len() as f64)
}

fn box_transform(box_: Box3) -> TrajResult<BoxTransform> {
    Ok(match box_ {
        Box3::None => BoxTransform::None,
        Box3::Orthorhombic { lx, ly, lz } => {
            BoxTransform::Orthorhombic([lx as f64, ly as f64, lz as f64])
        }
        Box3::Triclinic { .. } => {
            let (cell, inv) = cell_and_inv_from_box(box_)?;
            BoxTransform::Triclinic { cell, inv }
        }
    })
}

fn minimum_image(delta: &mut [f64; 3], transform: &BoxTransform) {
    let (mut dx, mut dy, mut dz) = (delta[0], delta[1], delta[2]);
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
    delta[0] = dx;
    delta[1] = dy;
    delta[2] = dz;
}

fn group_center(
    chunk: &FrameChunk,
    frame: usize,
    indices: &[usize],
    masses: &[f32],
    mass_weighted: bool,
    transform: &BoxTransform,
) -> [f64; 3] {
    if indices.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let ref_point = point(chunk, frame, indices[0]);
    let mut sum = [0.0; 3];
    let mut weight_sum = 0.0;
    for &idx in indices.iter() {
        let mut pos = point(chunk, frame, idx);
        let mut delta = sub(pos, ref_point);
        minimum_image(&mut delta, transform);
        pos = add(ref_point, delta);
        let weight = if mass_weighted {
            masses.get(idx).copied().unwrap_or(1.0).max(0.0) as f64
        } else {
            1.0
        };
        sum = add(sum, mul(pos, weight));
        weight_sum += weight;
    }
    if weight_sum <= 0.0 {
        ref_point
    } else {
        mul(sum, 1.0 / weight_sum)
    }
}

fn rotate_collection(points: &mut [[f64; 3]], axis: &mut [f64; 3], c0: usize, c1: usize) {
    let ax = normalize(*axis);
    if norm(ax) <= 1e-12 {
        return;
    }
    for point in points.iter_mut() {
        let tmp = *point;
        point[c0] = ax[c1] * tmp[c0] - ax[c0] * tmp[c1];
        point[c1] = ax[c0] * tmp[c0] + ax[c1] * tmp[c1];
    }
    let tmp = *axis;
    axis[c0] = ax[c1] * tmp[c0] - ax[c0] * tmp[c1];
    axis[c1] = ax[c0] * tmp[c0] + ax[c1] * tmp[c1];
}

fn rotate_bundle_points(
    tops: &mut Vec<[f64; 3]>,
    bottoms: &mut Vec<[f64; 3]>,
    mut kinks: Option<&mut Vec<[f64; 3]>>,
    axis: &mut [f64; 3],
) {
    rotate_collection(tops.as_mut_slice(), axis, 1, 2);
    rotate_collection(bottoms.as_mut_slice(), axis, 1, 2);
    if let Some(kinks) = kinks.as_mut() {
        rotate_collection(kinks.as_mut_slice(), axis, 1, 2);
    }
    rotate_collection(tops.as_mut_slice(), axis, 0, 2);
    rotate_collection(bottoms.as_mut_slice(), axis, 0, 2);
    if let Some(kinks) = kinks {
        rotate_collection(kinks.as_mut_slice(), axis, 0, 2);
    }
}

fn vector_to_f32(value: [f64; 3]) -> [f32; 3] {
    [value[0] as f32, value[1] as f32, value[2] as f32]
}

fn scaled_vector_to_f32(value: [f64; 3], scale: f64) -> [f32; 3] {
    [
        (value[0] * scale) as f32,
        (value[1] * scale) as f32,
        (value[2] * scale) as f32,
    ]
}

fn signed_tilt(comp: f64, dir_z: f64) -> f64 {
    let denom = comp.hypot(dir_z);
    if denom <= 1e-12 {
        0.0
    } else {
        clamp(comp / denom).asin().to_degrees()
    }
}
