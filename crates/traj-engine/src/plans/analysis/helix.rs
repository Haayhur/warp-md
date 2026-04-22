use nalgebra::{Matrix3, Vector3};
use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, HelixOutput, Plan, PlanOutput};
use crate::plans::analysis::secondary_structure::{
    add, average_phi_psi_weights, build_backbone_model, clamp, compute_backbone_frame, cross,
    distance, dot, helix_flags, jcaha_delta, longest_true_run, mul, norm, normalize, sub,
    BackboneModel,
};
use crate::plans::geometry::geometry_math::kabsch_rotation;

const IDEAL_RADIUS: f64 = 2.3;
const IDEAL_RISE: f64 = 1.5;
const IDEAL_TWIST_DEG: f64 = 100.0;

pub struct HelixPlan {
    selection: Selection,
    fit: bool,
    check_each_frame: bool,
    residue_start: Option<i32>,
    residue_end: Option<i32>,
    length_scale: f64,
    model: BackboneModel,
    fixed_fragment: Option<(usize, usize)>,
    reference_fragment: Option<(usize, usize)>,
    time: Vec<f32>,
    fragment_start: Vec<i32>,
    fragment_end: Vec<i32>,
    radius: Vec<f32>,
    twist: Vec<f32>,
    rise: Vec<f32>,
    length: Vec<f32>,
    dipole: Vec<f32>,
    rmsd: Vec<f32>,
    ca_phi: Vec<f32>,
    phi: Vec<f32>,
    psi: Vec<f32>,
    hb3: Vec<f32>,
    hb4: Vec<f32>,
    hb5: Vec<f32>,
    ellipticity: Vec<f32>,
    fragment_mask: Vec<bool>,
    residue_rmsd: Vec<f32>,
    helicity_counts: Vec<usize>,
    jcaha_sum: Vec<f64>,
    jcaha_count: Vec<usize>,
    frames: usize,
    used_box: bool,
}

impl HelixPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            fit: true,
            check_each_frame: false,
            residue_start: None,
            residue_end: None,
            length_scale: 1.0,
            model: BackboneModel::default(),
            fixed_fragment: None,
            reference_fragment: None,
            time: Vec::new(),
            fragment_start: Vec::new(),
            fragment_end: Vec::new(),
            radius: Vec::new(),
            twist: Vec::new(),
            rise: Vec::new(),
            length: Vec::new(),
            dipole: Vec::new(),
            rmsd: Vec::new(),
            ca_phi: Vec::new(),
            phi: Vec::new(),
            psi: Vec::new(),
            hb3: Vec::new(),
            hb4: Vec::new(),
            hb5: Vec::new(),
            ellipticity: Vec::new(),
            fragment_mask: Vec::new(),
            residue_rmsd: Vec::new(),
            helicity_counts: Vec::new(),
            jcaha_sum: Vec::new(),
            jcaha_count: Vec::new(),
            frames: 0,
            used_box: false,
        }
    }

    pub fn with_fit(mut self, fit: bool) -> Self {
        self.fit = fit;
        self
    }

    pub fn with_check_each_frame(mut self, check_each_frame: bool) -> Self {
        self.check_each_frame = check_each_frame;
        self
    }

    pub fn with_residue_range(
        mut self,
        residue_start: Option<i32>,
        residue_end: Option<i32>,
    ) -> Self {
        self.residue_start = residue_start;
        self.residue_end = residue_end;
        self
    }

    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    pub fn labels(&self) -> &[String] {
        &self.model.labels
    }
}

impl Plan for HelixPlan {
    fn name(&self) -> &'static str {
        "helix"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(TrajError::Parse(
                "helix length_scale must be finite and > 0".into(),
            ));
        }
        self.model = build_backbone_model(system, &self.selection);
        self.fixed_fragment =
            resolve_fixed_fragment(&self.model, self.residue_start, self.residue_end)?;
        self.reference_fragment = self.fixed_fragment;
        self.time.clear();
        self.fragment_start.clear();
        self.fragment_end.clear();
        self.radius.clear();
        self.twist.clear();
        self.rise.clear();
        self.length.clear();
        self.dipole.clear();
        self.rmsd.clear();
        self.ca_phi.clear();
        self.phi.clear();
        self.psi.clear();
        self.hb3.clear();
        self.hb4.clear();
        self.hb5.clear();
        self.ellipticity.clear();
        self.fragment_mask.clear();
        self.residue_rmsd.clear();
        self.helicity_counts = vec![0; self.model.residues.len()];
        self.jcaha_sum = vec![0.0; self.model.residues.len()];
        self.jcaha_count = vec![0; self.model.residues.len()];
        self.frames = 0;
        self.used_box = false;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_res = self.model.residues.len();
        if n_res == 0 {
            return Ok(());
        }
        for frame_idx in 0..chunk.n_frames {
            self.used_box |= !matches!(
                chunk
                    .box_
                    .get(frame_idx)
                    .copied()
                    .unwrap_or(traj_core::frame::Box3::None),
                traj_core::frame::Box3::None
            );
            let frame_time = chunk
                .time_ps
                .as_ref()
                .and_then(|values| values.get(frame_idx).copied())
                .unwrap_or(self.frames as f32);
            self.time.push(frame_time);
            let frame = compute_backbone_frame(&self.model, chunk, frame_idx);
            for i in 0..n_res {
                if let (Some(phi), Some(psi)) = (frame.phi[i], frame.psi[i]) {
                    self.jcaha_sum[i] += jcaha_delta(phi, psi);
                    self.jcaha_count[i] += 1;
                }
            }

            let fragment = if let Some(fragment) = self.fixed_fragment {
                Some(fragment)
            } else if self.check_each_frame {
                longest_true_run(&self.model, &helix_flags(&self.model, &frame))
            } else if let Some(fragment) = self.reference_fragment {
                Some(fragment)
            } else {
                let fragment = longest_true_run(&self.model, &helix_flags(&self.model, &frame));
                self.reference_fragment = fragment;
                fragment
            };

            let mut mask_row = vec![false; n_res];
            let mut residue_rmsd_row = vec![f32::NAN; n_res];
            match fragment {
                Some((start, end)) => {
                    for i in start..=end {
                        mask_row[i] = true;
                        self.helicity_counts[i] += 1;
                    }
                    let metrics = compute_fragment_metrics(
                        &self.model,
                        &frame,
                        start,
                        end,
                        self.fit,
                        self.length_scale,
                    );
                    for (offset, value) in metrics.residue_rmsd.iter().enumerate() {
                        residue_rmsd_row[start + offset] = *value as f32;
                    }
                    self.fragment_start.push(self.model.residues[start].resid);
                    self.fragment_end.push(self.model.residues[end].resid);
                    self.radius.push(metrics.radius as f32);
                    self.twist.push(metrics.twist as f32);
                    self.rise.push(metrics.rise as f32);
                    self.length.push(metrics.length as f32);
                    self.dipole.push(metrics.dipole as f32);
                    self.rmsd.push(metrics.rmsd as f32);
                    self.ca_phi.push(metrics.ca_phi as f32);
                    self.phi.push(metrics.phi as f32);
                    self.psi.push(metrics.psi as f32);
                    self.hb3.push(metrics.hb3 as f32);
                    self.hb4.push(metrics.hb4 as f32);
                    self.hb5.push(metrics.hb5 as f32);
                    self.ellipticity.push(metrics.ellipticity as f32);
                }
                None => {
                    self.fragment_start.push(-1);
                    self.fragment_end.push(-1);
                    self.radius.push(f32::NAN);
                    self.twist.push(f32::NAN);
                    self.rise.push(f32::NAN);
                    self.length.push(f32::NAN);
                    self.dipole.push(f32::NAN);
                    self.rmsd.push(f32::NAN);
                    self.ca_phi.push(f32::NAN);
                    self.phi.push(f32::NAN);
                    self.psi.push(f32::NAN);
                    self.hb3.push(f32::NAN);
                    self.hb4.push(f32::NAN);
                    self.hb5.push(f32::NAN);
                    self.ellipticity.push(f32::NAN);
                }
            }
            self.fragment_mask.extend(mask_row);
            self.residue_rmsd.extend(residue_rmsd_row);
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let residues = self.model.residues.len();
        let frames = self.frames;
        let helicity_fraction = if frames == 0 {
            vec![0.0; residues]
        } else {
            self.helicity_counts
                .iter()
                .map(|&count| count as f32 / frames as f32)
                .collect()
        };
        let jca_ha = self
            .jcaha_sum
            .iter()
            .zip(self.jcaha_count.iter())
            .map(|(&sum, &count)| {
                if count == 0 {
                    f32::NAN
                } else {
                    (140.3 + sum / count as f64) as f32
                }
            })
            .collect();
        Ok(PlanOutput::Helix(HelixOutput {
            labels: self.model.labels.clone(),
            time: std::mem::take(&mut self.time),
            fragment_start: std::mem::take(&mut self.fragment_start),
            fragment_end: std::mem::take(&mut self.fragment_end),
            radius: std::mem::take(&mut self.radius),
            twist: std::mem::take(&mut self.twist),
            rise: std::mem::take(&mut self.rise),
            length: std::mem::take(&mut self.length),
            dipole: std::mem::take(&mut self.dipole),
            rmsd: std::mem::take(&mut self.rmsd),
            ca_phi: std::mem::take(&mut self.ca_phi),
            phi: std::mem::take(&mut self.phi),
            psi: std::mem::take(&mut self.psi),
            hb3: std::mem::take(&mut self.hb3),
            hb4: std::mem::take(&mut self.hb4),
            hb5: std::mem::take(&mut self.hb5),
            ellipticity: std::mem::take(&mut self.ellipticity),
            fragment_mask: std::mem::take(&mut self.fragment_mask),
            residue_rmsd: std::mem::take(&mut self.residue_rmsd),
            helicity_fraction,
            jca_ha,
            frames,
            residues,
            fit: self.fit,
            check_each_frame: self.check_each_frame,
            length_scale: self.length_scale as f32,
            used_box: self.used_box,
        }))
    }
}

#[derive(Clone, Debug)]
struct FragmentMetrics {
    radius: f64,
    twist: f64,
    rise: f64,
    length: f64,
    dipole: f64,
    rmsd: f64,
    ca_phi: f64,
    phi: f64,
    psi: f64,
    hb3: f64,
    hb4: f64,
    hb5: f64,
    ellipticity: f64,
    residue_rmsd: Vec<f64>,
}

fn resolve_fixed_fragment(
    model: &BackboneModel,
    residue_start: Option<i32>,
    residue_end: Option<i32>,
) -> TrajResult<Option<(usize, usize)>> {
    match (residue_start, residue_end) {
        (Some(start), Some(end)) => {
            if end < start {
                return Err(TrajError::Parse(
                    "helix residue_end must be >= residue_start".into(),
                ));
            }
            let start_idx = model
                .residues
                .iter()
                .position(|residue| residue.resid == start)
                .ok_or_else(|| {
                    TrajError::Mismatch(format!("helix residue_start {start} not found"))
                })?;
            let end_idx = model
                .residues
                .iter()
                .position(|residue| residue.resid == end)
                .ok_or_else(|| TrajError::Mismatch(format!("helix residue_end {end} not found")))?;
            if !contiguous_fragment(model, start_idx, end_idx) {
                return Err(TrajError::Mismatch(
                    "helix residue range must be a contiguous single-chain fragment".into(),
                ));
            }
            Ok(Some((start_idx, end_idx)))
        }
        (None, None) => Ok(None),
        _ => Err(TrajError::Parse(
            "helix residue_start and residue_end must be set together".into(),
        )),
    }
}

fn contiguous_fragment(model: &BackboneModel, start: usize, end: usize) -> bool {
    start <= end
        && (start == end || (start + 1..=end).all(|i| model.residues[i].prev_index == Some(i - 1)))
        && model.residues.get(start).map(|residue| residue.segment_id)
            == model.residues.get(end).map(|residue| residue.segment_id)
}

fn compute_fragment_metrics(
    model: &BackboneModel,
    frame: &crate::plans::analysis::secondary_structure::BackboneFrame,
    start: usize,
    end: usize,
    fit: bool,
    length_scale: f64,
) -> FragmentMetrics {
    let ca_positions: Vec<[f64; 3]> = (start..=end).filter_map(|i| frame.ca[i]).collect();
    if ca_positions.len() < 2 {
        return FragmentMetrics {
            radius: f64::NAN,
            twist: f64::NAN,
            rise: f64::NAN,
            length: f64::NAN,
            dipole: f64::NAN,
            rmsd: f64::NAN,
            ca_phi: f64::NAN,
            phi: f64::NAN,
            psi: f64::NAN,
            hb3: f64::NAN,
            hb4: f64::NAN,
            hb5: f64::NAN,
            ellipticity: f64::NAN,
            residue_rmsd: vec![f64::NAN; end - start + 1],
        };
    }

    let ideal = ideal_helix_reference(ca_positions.len());
    let fit_result = align_to_reference(&ca_positions, &ideal);
    let axis_aligned = align_to_axis(&ca_positions);
    let metric_positions = if fit {
        fit_result.aligned.clone()
    } else {
        axis_aligned
    };

    let radius = metric_positions
        .iter()
        .map(|point| (point[0] * point[0] + point[1] * point[1]).sqrt())
        .sum::<f64>()
        / metric_positions.len() as f64
        * length_scale;
    let twist = average_twist(&metric_positions);
    let rise = average_rise(&metric_positions) * length_scale;
    let length = distance(
        metric_positions.first().copied().unwrap_or([0.0, 0.0, 0.0]),
        metric_positions.last().copied().unwrap_or([0.0, 0.0, 0.0]),
    ) * length_scale;
    let dipole = backbone_dipole(frame, start, end) * length_scale;
    let rmsd = fit_result.rmsd * length_scale;
    let residue_rmsd = fit_result
        .residue_rmsd
        .iter()
        .map(|value| value * length_scale)
        .collect();
    let ca_phi = average_ca_dihedral(&metric_positions);
    let phi = average_values((start..=end).filter_map(|i| frame.phi[i]));
    let psi = average_values((start..=end).filter_map(|i| frame.psi[i]));
    let hb3 = average_values((start..=end).filter_map(|i| frame.d3[i])) * length_scale;
    let hb4 = average_values((start..=end).filter_map(|i| frame.d4[i])) * length_scale;
    let hb5 = average_values((start..=end).filter_map(|i| frame.d5[i])) * length_scale;
    let ellipticity = (start..=end)
        .filter_map(|i| match (frame.phi[i], frame.psi[i]) {
            (Some(phi), Some(psi)) => average_phi_psi_weights(phi, psi),
            _ => None,
        })
        .sum();

    let _ = model;
    FragmentMetrics {
        radius,
        twist,
        rise,
        length,
        dipole,
        rmsd,
        ca_phi,
        phi,
        psi,
        hb3,
        hb4,
        hb5,
        ellipticity,
        residue_rmsd,
    }
}

#[derive(Clone, Debug)]
struct FitResult {
    aligned: Vec<[f64; 3]>,
    residue_rmsd: Vec<f64>,
    rmsd: f64,
}

fn align_to_reference(points: &[[f64; 3]], reference: &[[f64; 3]]) -> FitResult {
    let frame: Vec<[f32; 4]> = points
        .iter()
        .map(|point| [point[0] as f32, point[1] as f32, point[2] as f32, 1.0])
        .collect();
    let reference_f32: Vec<[f32; 4]> = reference
        .iter()
        .map(|point| [point[0] as f32, point[1] as f32, point[2] as f32, 1.0])
        .collect();
    let (rotation, cx, cy) = kabsch_rotation(&frame, &reference_f32);
    let mut aligned = Vec::with_capacity(points.len());
    let mut residue_rmsd = Vec::with_capacity(points.len());
    let mut sumsq = 0.0;
    for (point, ref_point) in points.iter().zip(reference.iter()) {
        let vector = Vector3::new(point[0], point[1], point[2]) - cx;
        let aligned_vec = rotation * vector + cy;
        let aligned_point = [aligned_vec[0], aligned_vec[1], aligned_vec[2]];
        let diff = sub(aligned_point, *ref_point);
        let residue = norm(diff);
        sumsq += residue * residue;
        aligned.push(aligned_point);
        residue_rmsd.push(residue);
    }
    FitResult {
        aligned,
        residue_rmsd,
        rmsd: (sumsq / points.len() as f64).sqrt(),
    }
}

fn align_to_axis(points: &[[f64; 3]]) -> Vec<[f64; 3]> {
    let centroid = points.iter().fold([0.0; 3], |acc, point| add(acc, *point));
    let centroid = mul(centroid, 1.0 / points.len() as f64);
    let centered: Vec<[f64; 3]> = points.iter().map(|point| sub(*point, centroid)).collect();
    let axis = normalize(sub(
        centered.last().copied().unwrap_or([0.0, 0.0, 1.0]),
        centered.first().copied().unwrap_or([0.0, 0.0, 0.0]),
    ));
    let rotation = rotation_between(axis, [0.0, 0.0, 1.0]);
    let mut aligned: Vec<[f64; 3]> = centered
        .iter()
        .map(|point| apply_rotation(&rotation, *point))
        .collect();
    if let Some(first) = aligned.first().copied() {
        let angle = first[1].atan2(first[0]);
        let z_rotation = rotation_z(-angle);
        for point in aligned.iter_mut() {
            *point = apply_rotation(&z_rotation, *point);
        }
    }
    aligned
}

fn ideal_helix_reference(n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            let angle = (i as f64 * IDEAL_TWIST_DEG).to_radians();
            [
                IDEAL_RADIUS * angle.cos(),
                IDEAL_RADIUS * angle.sin(),
                i as f64 * IDEAL_RISE,
            ]
        })
        .collect()
}

fn average_twist(points: &[[f64; 3]]) -> f64 {
    if points.len() < 2 {
        return f64::NAN;
    }
    let mut total = 0.0;
    for window in points.windows(2) {
        let phi0 = window[0][1].atan2(window[0][0]).to_degrees();
        let phi1 = window[1][1].atan2(window[1][0]).to_degrees();
        let mut delta = phi1 - phi0;
        if delta < -90.0 {
            delta += 360.0;
        } else if delta > 270.0 {
            delta -= 360.0;
        }
        total += delta;
    }
    total / (points.len() - 1) as f64
}

fn average_rise(points: &[[f64; 3]]) -> f64 {
    if points.len() < 2 {
        return f64::NAN;
    }
    points
        .windows(2)
        .map(|window| window[1][2] - window[0][2])
        .sum::<f64>()
        / (points.len() - 1) as f64
}

fn average_ca_dihedral(points: &[[f64; 3]]) -> f64 {
    if points.len() < 4 {
        return f64::NAN;
    }
    let mut values = Vec::new();
    for window in points.windows(4) {
        if let Some(value) = ca_dihedral(window[0], window[1], window[2], window[3]) {
            values.push(value);
        }
    }
    average_values(values.into_iter())
}

fn ca_dihedral(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> Option<f64> {
    let b0 = sub(a, b);
    let b1 = sub(c, b);
    let b2 = sub(d, c);
    let b1n = normalize(b1);
    if norm(b1n) <= 1e-12 {
        return None;
    }
    let v = sub(b0, mul(b1n, dot(b0, b1n)));
    let w = sub(b2, mul(b1n, dot(b2, b1n)));
    let vnorm = norm(v);
    let wnorm = norm(w);
    if vnorm <= 1e-12 || wnorm <= 1e-12 {
        return None;
    }
    let vn = mul(v, 1.0 / vnorm);
    let wn = mul(w, 1.0 / wnorm);
    Some(dot(cross(b1n, vn), wn).atan2(dot(vn, wn)).to_degrees())
}

fn average_values(values: impl Iterator<Item = f64>) -> f64 {
    let collected: Vec<f64> = values.filter(|value| value.is_finite()).collect();
    if collected.is_empty() {
        f64::NAN
    } else {
        collected.iter().sum::<f64>() / collected.len() as f64
    }
}

fn backbone_dipole(
    frame: &crate::plans::analysis::secondary_structure::BackboneFrame,
    start: usize,
    end: usize,
) -> f64 {
    let mut dipole = [0.0; 3];
    for i in start..=end {
        if let Some(c_pos) = frame.c[i] {
            dipole = add(dipole, mul(c_pos, 0.42));
        }
        if let Some(o_pos) = frame.o[i] {
            dipole = add(dipole, mul(o_pos, -0.42));
        }
        if let Some(n_pos) = frame.n[i] {
            dipole = add(dipole, mul(n_pos, -0.20));
        }
        if let Some(h_pos) = frame.h[i] {
            dipole = add(dipole, mul(h_pos, 0.20));
        }
    }
    norm(dipole)
}

fn rotation_between(from: [f64; 3], to: [f64; 3]) -> Matrix3<f64> {
    let from = normalize(from);
    let to = normalize(to);
    let axis = cross(from, to);
    let axis_norm = norm(axis);
    let cos_theta = clamp(dot(from, to));
    if axis_norm <= 1e-12 {
        if cos_theta > 0.0 {
            return Matrix3::identity();
        }
        let fallback = if from[0].abs() < 0.9 {
            normalize(cross(from, [1.0, 0.0, 0.0]))
        } else {
            normalize(cross(from, [0.0, 1.0, 0.0]))
        };
        return rodrigues(fallback, std::f64::consts::PI);
    }
    rodrigues(mul(axis, 1.0 / axis_norm), cos_theta.acos())
}

fn rodrigues(axis: [f64; 3], angle: f64) -> Matrix3<f64> {
    let [x, y, z] = axis;
    let c = angle.cos();
    let s = angle.sin();
    let one_c = 1.0 - c;
    Matrix3::new(
        c + x * x * one_c,
        x * y * one_c - z * s,
        x * z * one_c + y * s,
        y * x * one_c + z * s,
        c + y * y * one_c,
        y * z * one_c - x * s,
        z * x * one_c - y * s,
        z * y * one_c + x * s,
        c + z * z * one_c,
    )
}

fn rotation_z(angle: f64) -> Matrix3<f64> {
    Matrix3::new(
        angle.cos(),
        -angle.sin(),
        0.0,
        angle.sin(),
        angle.cos(),
        0.0,
        0.0,
        0.0,
        1.0,
    )
}

fn apply_rotation(rotation: &Matrix3<f64>, point: [f64; 3]) -> [f64; 3] {
    let vec = rotation * Vector3::new(point[0], point[1], point[2]);
    [vec[0], vec[1], vec[2]]
}
