use super::*;

pub(super) fn inverse_rotated_xy(
    point: [f32; 2],
    center: [f32; 2],
    rotate_degrees: f32,
) -> [f32; 2] {
    let theta = -rotate_degrees.to_radians();
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let x = point[0] - center[0];
    let y = point[1] - center[1];
    [x * cos_t - y * sin_t, x * sin_t + y * cos_t]
}

pub(super) fn forward_rotated_xy(
    local: [f32; 2],
    center: [f32; 2],
    rotate_degrees: f32,
) -> [f32; 2] {
    let theta = rotate_degrees.to_radians();
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    [
        center[0] + local[0] * cos_t - local[1] * sin_t,
        center[1] + local[0] * sin_t + local[1] * cos_t,
    ]
}

pub(super) fn rectangle_boundary_distance(local: [f32; 2], half_size: [f32; 2]) -> f32 {
    let dx = (local[0].abs() - half_size[0]).abs();
    let dy = (local[1].abs() - half_size[1]).abs();
    dx.min(dy)
}

pub(super) fn polygon_boundary_distance(point: [f32; 2], polygon: &[[f32; 2]]) -> f32 {
    if polygon.len() < 2 {
        return 0.0;
    }
    (0..polygon.len())
        .map(|idx| {
            let projected =
                nearest_point_on_segment(point, polygon[idx], polygon[(idx + 1) % polygon.len()]);
            squared_distance2(point, projected).sqrt()
        })
        .fold(f32::INFINITY, f32::min)
}

pub(super) fn polygon_area_with_inset(points: &[[f32; 2]], inset_angstrom: f32) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    let Some((xmin, xmax, ymin, ymax)) = polygon_bounds(points) else {
        return 0.0;
    };
    let spacing = 0.5f32;
    let nx = ((xmax - xmin) / spacing).ceil() as usize;
    let ny = ((ymax - ymin) / spacing).ceil() as usize;
    let mut covered = 0usize;
    for ix in 0..nx {
        let x = xmin + (ix as f32 + 0.5) * spacing;
        for iy in 0..ny {
            let y = ymin + (iy as f32 + 0.5) * spacing;
            let point = [x, y];
            if point_in_polygon(point, points)
                && polygon_boundary_distance(point, points) >= inset_angstrom
            {
                covered += 1;
            }
        }
    }
    covered as f32 * spacing.powi(2)
}

pub(super) fn conservative_grid_region_union_error_bound(
    regions: &[&LeafletRegion],
    clipped_domain: (f32, f32, f32, f32),
    spacing: f32,
) -> f32 {
    let (xmin, xmax, ymin, ymax) = clipped_domain;
    let domain_perimeter = 2.0 * ((xmax - xmin).max(0.0) + (ymax - ymin).max(0.0));
    let shape_perimeter = regions
        .iter()
        .map(|region| region_perimeter(region))
        .sum::<f32>();
    let boundary_count = regions.len() as f32 + 1.0;
    let band_radius = spacing * std::f32::consts::SQRT_2 * 0.5;
    2.0 * band_radius * (domain_perimeter + shape_perimeter)
        + boundary_count * std::f32::consts::PI * band_radius.powi(2)
}

pub(super) fn region_perimeter(region: &LeafletRegion) -> f32 {
    match &region.geometry {
        RegionGeometry::Circle {
            radius_angstrom, ..
        } => std::f32::consts::TAU * radius_angstrom.max(0.0),
        RegionGeometry::Ellipse {
            radius_angstrom, ..
        } => ellipse_perimeter_ramanujan(radius_angstrom[0].max(0.0), radius_angstrom[1].max(0.0)),
        RegionGeometry::Rectangle { size_angstrom, .. } => {
            2.0 * (size_angstrom[0].max(0.0) + size_angstrom[1].max(0.0))
        }
        RegionGeometry::Polygon { .. } => polygon_perimeter(&transformed_polygon_points(region)),
    }
}

pub(super) fn ellipse_perimeter_ramanujan(radius_x: f32, radius_y: f32) -> f32 {
    if radius_x <= 0.0 || radius_y <= 0.0 {
        return 0.0;
    }
    let h = ((radius_x - radius_y) / (radius_x + radius_y)).powi(2);
    std::f32::consts::PI * (radius_x + radius_y) * (1.0 + 3.0 * h / (10.0 + (4.0 - 3.0 * h).sqrt()))
}

pub(super) fn exact_axis_aligned_rectangle_polygon_inset_area(
    points: &[[f32; 2]],
    inset_angstrom: f32,
) -> Option<f32> {
    if points.len() != 4 || inset_angstrom < 0.0 {
        return None;
    }
    let (xmin, xmax, ymin, ymax) = polygon_bounds(points)?;
    let mut unmatched_corners = vec![[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]];
    for point in points {
        let corner_idx = unmatched_corners.iter().position(|corner| {
            (corner[0] - point[0]).abs() < 1.0e-5 && (corner[1] - point[1]).abs() < 1.0e-5
        })?;
        unmatched_corners.remove(corner_idx);
    }
    let width = (xmax - xmin - 2.0 * inset_angstrom).max(0.0);
    let height = (ymax - ymin - 2.0 * inset_angstrom).max(0.0);
    Some(width * height)
}

pub(super) fn exact_convex_polygon_inset_area(
    points: &[[f32; 2]],
    inset_angstrom: f32,
) -> Option<f32> {
    if points.len() < 3 || inset_angstrom < 0.0 || !polygon_is_convex(points) {
        return None;
    }
    if inset_angstrom <= 1.0e-6 {
        return Some(polygon_area(points));
    }
    let mut inset = points.to_vec();
    ensure_ccw_polygon(&mut inset);
    let edges = inset.clone();
    for idx in 0..edges.len() {
        let a = edges[idx];
        let b = edges[(idx + 1) % edges.len()];
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let len = (dx * dx + dy * dy).sqrt();
        if len <= 1.0e-8 {
            return None;
        }
        let inward = [-dy / len, dx / len];
        let shifted_a = [
            a[0] + inward[0] * inset_angstrom,
            a[1] + inward[1] * inset_angstrom,
        ];
        let shifted_b = [
            b[0] + inward[0] * inset_angstrom,
            b[1] + inward[1] * inset_angstrom,
        ];
        inset = convex_polygon_clip_half_plane(&inset, shifted_a, shifted_b);
        if inset.len() < 3 || polygon_area(&inset) <= 1.0e-6 {
            return Some(0.0);
        }
    }
    Some(polygon_area(&inset))
}

pub(super) fn convex_polygon_clip_half_plane(
    subject: &[[f32; 2]],
    edge_start: [f32; 2],
    edge_end: [f32; 2],
) -> Vec<[f32; 2]> {
    if subject.len() < 3 {
        return Vec::new();
    }
    let mut output = Vec::new();
    let mut previous = *subject.last().unwrap();
    let mut previous_inside = point_left_of_edge(previous, edge_start, edge_end);
    for &current in subject {
        let current_inside = point_left_of_edge(current, edge_start, edge_end);
        if current_inside {
            if !previous_inside {
                output.push(line_segment_intersection(
                    previous, current, edge_start, edge_end,
                ));
            }
            output.push(current);
        } else if previous_inside {
            output.push(line_segment_intersection(
                previous, current, edge_start, edge_end,
            ));
        }
        previous = current;
        previous_inside = current_inside;
    }
    output
}

pub(super) fn polygon_bounds(points: &[[f32; 2]]) -> Option<(f32, f32, f32, f32)> {
    let mut iter = points.iter();
    let first = iter.next()?;
    let mut xmin = first[0];
    let mut xmax = first[0];
    let mut ymin = first[1];
    let mut ymax = first[1];
    for point in iter {
        xmin = xmin.min(point[0]);
        xmax = xmax.max(point[0]);
        ymin = ymin.min(point[1]);
        ymax = ymax.max(point[1]);
    }
    Some((xmin, xmax, ymin, ymax))
}

pub(super) fn transformed_polygon_points(region: &LeafletRegion) -> Vec<[f32; 2]> {
    let RegionGeometry::Polygon {
        points_angstrom,
        scale_xy,
        rotate_degrees,
    } = &region.geometry
    else {
        return Vec::new();
    };
    let Some(center) = polygon_bounds_center(points_angstrom) else {
        return points_angstrom.clone();
    };
    let scale = scale_xy.unwrap_or([1.0, 1.0]);
    let theta = rotate_degrees.to_radians();
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    points_angstrom
        .iter()
        .map(|point| {
            let x = (point[0] - center[0]) * scale[0];
            let y = (point[1] - center[1]) * scale[1];
            [
                center[0] + x * cos_t - y * sin_t,
                center[1] + x * sin_t + y * cos_t,
            ]
        })
        .collect()
}

pub(super) fn polygon_bounds_center(points: &[[f32; 2]]) -> Option<[f32; 2]> {
    let mut iter = points.iter();
    let first = iter.next()?;
    let mut xmin = first[0];
    let mut xmax = first[0];
    let mut ymin = first[1];
    let mut ymax = first[1];
    for point in iter {
        xmin = xmin.min(point[0]);
        xmax = xmax.max(point[0]);
        ymin = ymin.min(point[1]);
        ymax = ymax.max(point[1]);
    }
    Some([(xmin + xmax) * 0.5, (ymin + ymax) * 0.5])
}

pub(super) fn inverse_rotated_point(
    point: [f32; 2],
    center: [f32; 2],
    rotate_degrees: f32,
) -> [f32; 2] {
    let theta = -rotate_degrees.to_radians();
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let dx = point[0] - center[0];
    let dy = point[1] - center[1];
    [dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t]
}

pub(super) fn point_in_polygon(point: [f32; 2], polygon: &[[f32; 2]]) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let pi = polygon[i];
        let pj = polygon[j];
        if (pi[1] > point[1]) != (pj[1] > point[1])
            && point[0] < (pj[0] - pi[0]) * (point[1] - pi[1]) / (pj[1] - pi[1]) + pi[0]
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

pub(super) fn project_point_to_polygon(point: [f32; 2], polygon: &[[f32; 2]]) -> [f32; 2] {
    if polygon.len() < 3 || point_in_polygon(point, polygon) {
        return point;
    }
    let centroid = polygon_centroid(polygon);
    let mut best = point;
    let mut best_distance = f32::INFINITY;
    for idx in 0..polygon.len() {
        let projected =
            nearest_point_on_segment(point, polygon[idx], polygon[(idx + 1) % polygon.len()]);
        let distance = (projected[0] - point[0]).powi(2) + (projected[1] - point[1]).powi(2);
        if distance < best_distance {
            best_distance = distance;
            best = projected;
        }
    }
    [
        best[0] + (centroid[0] - best[0]) * 1.0e-3,
        best[1] + (centroid[1] - best[1]) * 1.0e-3,
    ]
}

pub(super) fn project_point_to_polygon_with_margin(
    point: [f32; 2],
    polygon: &[[f32; 2]],
    margin_angstrom: f32,
) -> [f32; 2] {
    let margin = margin_angstrom.max(0.0);
    if polygon.len() < 3 {
        return point;
    }
    if margin <= 1.0e-6 && point_in_polygon(point, polygon) {
        return point;
    }
    let inside_point = if point_in_polygon(point, polygon) {
        point
    } else {
        project_point_to_polygon(point, polygon)
    };
    if margin <= 1.0e-6 || polygon_boundary_distance(inside_point, polygon) >= margin {
        return inside_point;
    }
    let nearest = nearest_point_on_polygon_boundary(inside_point, polygon);
    let mut direction = [inside_point[0] - nearest[0], inside_point[1] - nearest[1]];
    let mut norm = (direction[0].powi(2) + direction[1].powi(2)).sqrt();
    if norm <= 1.0e-6 {
        let centroid = polygon_centroid(polygon);
        direction = [centroid[0] - nearest[0], centroid[1] - nearest[1]];
        norm = (direction[0].powi(2) + direction[1].powi(2)).sqrt();
    }
    if norm <= 1.0e-6 {
        return inside_point;
    }
    let candidate = [
        nearest[0] + direction[0] / norm * margin,
        nearest[1] + direction[1] / norm * margin,
    ];
    if point_in_polygon(candidate, polygon)
        && polygon_boundary_distance(candidate, polygon) >= margin - 1.0e-4
    {
        candidate
    } else {
        polygon_centroid(polygon)
    }
}

pub(super) fn nearest_point_on_polygon_boundary(point: [f32; 2], polygon: &[[f32; 2]]) -> [f32; 2] {
    let mut best = point;
    let mut best_distance = f32::INFINITY;
    for idx in 0..polygon.len() {
        let projected =
            nearest_point_on_segment(point, polygon[idx], polygon[(idx + 1) % polygon.len()]);
        let distance = squared_distance2(point, projected);
        if distance < best_distance {
            best_distance = distance;
            best = projected;
        }
    }
    best
}

pub(super) fn nearest_point_on_segment(point: [f32; 2], a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let len_sq = ab[0] * ab[0] + ab[1] * ab[1];
    if len_sq <= f32::EPSILON {
        return a;
    }
    let t = (((point[0] - a[0]) * ab[0] + (point[1] - a[1]) * ab[1]) / len_sq).clamp(0.0, 1.0);
    [a[0] + t * ab[0], a[1] + t * ab[1]]
}

pub(super) fn polygon_centroid(points: &[[f32; 2]]) -> [f32; 2] {
    if points.is_empty() {
        return [0.0, 0.0];
    }
    let sum = points.iter().fold([0.0f32, 0.0f32], |acc, point| {
        [acc[0] + point[0], acc[1] + point[1]]
    });
    [sum[0] / points.len() as f32, sum[1] / points.len() as f32]
}
