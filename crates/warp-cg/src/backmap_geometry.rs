use crate::backmap::{BackmapError, BeadSite, ResidueTemplate};
use nalgebra::{Matrix3, Vector3};

const EPSILON: f64 = 1.0e-12;

pub(super) fn normalized_atom_pair(
    left: (usize, usize),
    right: (usize, usize),
) -> ((usize, usize), (usize, usize)) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

pub(super) fn signed_volume_sign(
    center: [f64; 3],
    neighbor_a: [f64; 3],
    neighbor_b: [f64; 3],
    neighbor_c: [f64; 3],
) -> i8 {
    let a = vec3(neighbor_a) - vec3(center);
    let b = vec3(neighbor_b) - vec3(center);
    let c = vec3(neighbor_c) - vec3(center);
    let volume = a.cross(&b).dot(&c);
    if volume > EPSILON {
        1
    } else if volume < -EPSILON {
        -1
    } else {
        0
    }
}

pub(super) fn contracted_reference_coords(template: &ResidueTemplate, fudge: f64) -> Vec<[f64; 3]> {
    let site_centers = template
        .bead_sites
        .iter()
        .map(|site| weighted_center(&template.reference_coords, site))
        .collect::<Vec<_>>();
    let mut owner = vec![None; template.reference_coords.len()];
    for (site_idx, site) in template.bead_sites.iter().enumerate() {
        for &atom_idx in &site.atom_indices {
            owner[atom_idx] = Some(site_idx);
        }
    }
    template
        .reference_coords
        .iter()
        .enumerate()
        .map(|(atom_idx, &coord)| {
            let point = vec3(coord);
            let site_idx = owner[atom_idx].unwrap_or_else(|| {
                site_centers
                    .iter()
                    .enumerate()
                    .min_by(|(_, left), (_, right)| {
                        (point - **left)
                            .norm_squared()
                            .total_cmp(&(point - **right).norm_squared())
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            });
            let center = site_centers[site_idx];
            let contracted = center + fudge * (point - center);
            [contracted.x, contracted.y, contracted.z]
        })
        .collect()
}

pub(super) fn weighted_center(coords: &[[f64; 3]], site: &BeadSite) -> Vector3<f64> {
    let mut center = Vector3::zeros();
    let mut weight_sum = 0.0;
    for (local_idx, &atom_idx) in site.atom_indices.iter().enumerate() {
        let weight = site
            .weights
            .as_ref()
            .map(|weights| weights[local_idx])
            .unwrap_or(1.0);
        center += vec3(coords[atom_idx]) * weight;
        weight_sum += weight;
    }
    center / weight_sum
}

pub(super) fn kabsch_rotation(
    sources: &[Vector3<f64>],
    targets: &[Vector3<f64>],
    template_idx: usize,
) -> Result<Matrix3<f64>, BackmapError> {
    if sources.len() != targets.len() || sources.is_empty() {
        return Err(BackmapError::AlignmentFailure(template_idx));
    }
    if sources.len() == 1 {
        return Ok(Matrix3::identity());
    }
    let source_center = mean(sources);
    let target_center = mean(targets);
    let mut covariance = Matrix3::zeros();
    for (source, target) in sources.iter().zip(targets) {
        covariance += (source - source_center) * (target - target_center).transpose();
    }
    if covariance.norm_squared() <= EPSILON {
        return Ok(Matrix3::identity());
    }
    let svd = covariance.svd(true, true);
    let (Some(u), Some(mut v_t)) = (svd.u, svd.v_t) else {
        return Err(BackmapError::AlignmentFailure(template_idx));
    };
    let mut rotation = v_t.transpose() * u.transpose();
    if rotation.determinant() < 0.0 {
        v_t.row_mut(2).neg_mut();
        rotation = v_t.transpose() * u.transpose();
    }
    if rotation.iter().all(|value| value.is_finite()) {
        Ok(rotation)
    } else {
        Err(BackmapError::AlignmentFailure(template_idx))
    }
}

pub(super) fn mean(points: &[Vector3<f64>]) -> Vector3<f64> {
    points.iter().copied().sum::<Vector3<f64>>() / points.len() as f64
}

pub(super) fn vec3(coord: [f64; 3]) -> Vector3<f64> {
    Vector3::new(coord[0], coord[1], coord[2])
}

pub(super) fn finite_coord(coord: [f64; 3]) -> bool {
    coord.iter().all(|value| value.is_finite())
}
