use super::*;
use crate::build_box::{cross, dot, pbc_axes, unit_cell_to_vectors};

pub(super) fn squared_distance3(left: [f32; 3], right: [f32; 3]) -> f32 {
    (left[0] - right[0]).powi(2) + (left[1] - right[1]).powi(2) + (left[2] - right[2]).powi(2)
}

pub(super) fn squared_distance3_for_system(
    left: [f32; 3],
    right: [f32; 3],
    system: &BuildSystem,
) -> f32 {
    if let Some(vectors) = distance_box_vectors(system) {
        return squared_distance3_pbc_vectors(left, right, vectors, &system.pbc);
    }
    squared_distance3_pbc(left, right, system.box_size_angstrom, &system.pbc)
}

pub(super) fn distance_box_vectors(system: &BuildSystem) -> Option<[[f32; 3]; 3]> {
    if let Some(vectors) = system.box_vectors_angstrom {
        return Some(vectors);
    }
    if let Some(unit_cell) = system.unit_cell_angstrom {
        return unit_cell_to_vectors(unit_cell).ok();
    }
    None
}

pub(super) fn placement_box_size_angstrom(system: &BuildSystem) -> [f32; 3] {
    distance_box_vectors(system)
        .map(box_vector_extents_angstrom)
        .unwrap_or(system.box_size_angstrom)
}

pub(super) fn box_vector_extents_angstrom(vectors: [[f32; 3]; 3]) -> [f32; 3] {
    let corners = [
        [0.0, 0.0, 0.0],
        vectors[0],
        vectors[1],
        vectors[2],
        add3(vectors[0], vectors[1]),
        add3(vectors[0], vectors[2]),
        add3(vectors[1], vectors[2]),
        add3(add3(vectors[0], vectors[1]), vectors[2]),
    ];
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for corner in corners {
        for axis in 0..3 {
            min[axis] = min[axis].min(corner[axis]);
            max[axis] = max[axis].max(corner[axis]);
        }
    }
    [max[0] - min[0], max[1] - min[1], max[2] - min[2]]
}

pub(super) fn add3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] + right[0], left[1] + right[1], left[2] + right[2]]
}

pub(super) fn sub3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

pub(super) fn scale3(vector: [f32; 3], scale: f32) -> [f32; 3] {
    [vector[0] * scale, vector[1] * scale, vector[2] * scale]
}

pub(super) fn cell_origin(center: [f32; 3], vectors: [[f32; 3]; 3]) -> [f32; 3] {
    sub3(
        center,
        scale3(add3(add3(vectors[0], vectors[1]), vectors[2]), 0.5),
    )
}

pub(super) fn vector_lengths(vectors: [[f32; 3]; 3]) -> [f32; 3] {
    [
        dot(vectors[0], vectors[0]).sqrt(),
        dot(vectors[1], vectors[1]).sqrt(),
        dot(vectors[2], vectors[2]).sqrt(),
    ]
}

pub(super) fn vector_face_heights(vectors: [[f32; 3]; 3]) -> Option<[f32; 3]> {
    let volume = dot(vectors[0], cross(vectors[1], vectors[2])).abs();
    if volume <= 1.0e-8 {
        return None;
    }
    let face_areas = [
        dot(cross(vectors[1], vectors[2]), cross(vectors[1], vectors[2])).sqrt(),
        dot(cross(vectors[2], vectors[0]), cross(vectors[2], vectors[0])).sqrt(),
        dot(cross(vectors[0], vectors[1]), cross(vectors[0], vectors[1])).sqrt(),
    ];
    if face_areas.iter().any(|area| *area <= 1.0e-8) {
        return None;
    }
    Some([
        volume / face_areas[0],
        volume / face_areas[1],
        volume / face_areas[2],
    ])
}

pub(super) fn fractional_margin_for_radius(
    vectors: [[f32; 3]; 3],
    radius_angstrom: f32,
) -> Option<[f32; 3]> {
    let heights = vector_face_heights(vectors)?;
    Some([
        (radius_angstrom / heights[0]).max(0.0),
        (radius_angstrom / heights[1]).max(0.0),
        (radius_angstrom / heights[2]).max(0.0),
    ])
}

pub(super) fn point_inside_vector_cell(
    point: [f32; 3],
    center: [f32; 3],
    vectors: [[f32; 3]; 3],
    radius_angstrom: f32,
) -> bool {
    let Some(fractional) = fractional_delta(point, cell_origin(center, vectors), vectors) else {
        return false;
    };
    let Some(margins) = fractional_margin_for_radius(vectors, radius_angstrom) else {
        return false;
    };
    (0..3).all(|axis| {
        margins[axis] <= 0.5
            && fractional[axis] >= margins[axis]
            && fractional[axis] <= 1.0 - margins[axis]
    })
}

pub(super) fn random_cell_center(
    center: [f32; 3],
    vectors: [[f32; 3]; 3],
    radius_angstrom: f32,
    state: &mut u64,
) -> [f32; 3] {
    let margins = fractional_margin_for_radius(vectors, radius_angstrom).unwrap_or([0.5; 3]);
    let mut fractional = [0.5; 3];
    for axis in 0..3 {
        if margins[axis] < 0.5 {
            let span = 1.0 - 2.0 * margins[axis];
            fractional[axis] = margins[axis] + span * seeded_unit_f32(state);
        }
    }
    add3(
        cell_origin(center, vectors),
        cartesian_from_fractional_delta(fractional, vectors),
    )
}

pub(super) fn cell_center_candidates(
    center: [f32; 3],
    vectors: [[f32; 3]; 3],
    radius_angstrom: f32,
    spacing: f32,
    phase_offsets: &[[f32; 3]],
) -> Vec<[f32; 3]> {
    let Some(margins) = fractional_margin_for_radius(vectors, radius_angstrom) else {
        return Vec::new();
    };
    if margins.iter().any(|margin| *margin > 0.5) {
        return Vec::new();
    }
    let lengths = vector_lengths(vectors);
    let counts = lengths.map(|length| (length / spacing).floor().max(1.0) as usize);
    let origin = cell_origin(center, vectors);
    let mut candidates =
        Vec::with_capacity(counts[0] * counts[1] * counts[2] * phase_offsets.len().max(1));
    for phase in phase_offsets {
        for iz in 0..counts[2] {
            let fz = fractional_grid_coordinate(iz, counts[2], phase[2], margins[2]);
            if fz.is_none() {
                continue;
            }
            for iy in 0..counts[1] {
                let fy = fractional_grid_coordinate(iy, counts[1], phase[1], margins[1]);
                if fy.is_none() {
                    continue;
                }
                for ix in 0..counts[0] {
                    let fx = fractional_grid_coordinate(ix, counts[0], phase[0], margins[0]);
                    let (Some(fx), Some(fy), Some(fz)) = (fx, fy, fz) else {
                        continue;
                    };
                    candidates.push(add3(
                        origin,
                        cartesian_from_fractional_delta([fx, fy, fz], vectors),
                    ));
                }
            }
        }
    }
    candidates
}

pub(super) fn fractional_grid_coordinate(
    index: usize,
    count: usize,
    phase: f32,
    margin: f32,
) -> Option<f32> {
    if margin > 0.5 || count == 0 {
        return None;
    }
    let span = 1.0 - 2.0 * margin;
    let shifted = (index as f32 + 0.5 + phase) / count as f32;
    if shifted > 1.0 {
        return None;
    }
    Some(margin + span * shifted)
}

pub(super) fn squared_distance3_pbc(
    left: [f32; 3],
    right: [f32; 3],
    box_size_angstrom: [f32; 3],
    pbc: &str,
) -> f32 {
    let axes = pbc_axes(pbc);
    (0..3)
        .map(|axis| {
            let delta = minimum_image_delta(
                left[axis] - right[axis],
                box_size_angstrom[axis],
                axes[axis],
            );
            delta * delta
        })
        .sum()
}

pub(super) fn squared_distance3_pbc_vectors(
    left: [f32; 3],
    right: [f32; 3],
    vectors: [[f32; 3]; 3],
    pbc: &str,
) -> f32 {
    let axes = pbc_axes(pbc);
    if !axes.iter().any(|enabled| *enabled) {
        return squared_distance3(left, right);
    }
    let Some(mut fractional) = fractional_delta(left, right, vectors) else {
        return squared_distance3(left, right);
    };
    for axis in 0..3 {
        if axes[axis] {
            fractional[axis] -= fractional[axis].round();
        }
    }
    let delta = cartesian_from_fractional_delta(fractional, vectors);
    dot(delta, delta)
}

pub(super) fn fractional_delta(
    left: [f32; 3],
    right: [f32; 3],
    vectors: [[f32; 3]; 3],
) -> Option<[f32; 3]> {
    let delta = [left[0] - right[0], left[1] - right[1], left[2] - right[2]];
    let volume = dot(vectors[0], cross(vectors[1], vectors[2]));
    if volume.abs() <= 1.0e-8 {
        return None;
    }
    Some([
        dot(delta, cross(vectors[1], vectors[2])) / volume,
        dot(delta, cross(vectors[2], vectors[0])) / volume,
        dot(delta, cross(vectors[0], vectors[1])) / volume,
    ])
}

pub(super) fn cartesian_from_fractional_delta(
    fractional: [f32; 3],
    vectors: [[f32; 3]; 3],
) -> [f32; 3] {
    [
        fractional[0] * vectors[0][0]
            + fractional[1] * vectors[1][0]
            + fractional[2] * vectors[2][0],
        fractional[0] * vectors[0][1]
            + fractional[1] * vectors[1][1]
            + fractional[2] * vectors[2][1],
        fractional[0] * vectors[0][2]
            + fractional[1] * vectors[1][2]
            + fractional[2] * vectors[2][2],
    ]
}

pub(super) fn minimum_image_delta(delta: f32, box_size_angstrom: f32, periodic: bool) -> f32 {
    if periodic && box_size_angstrom > 0.0 && box_size_angstrom.is_finite() {
        delta - box_size_angstrom * (delta / box_size_angstrom).round()
    } else {
        delta
    }
}
