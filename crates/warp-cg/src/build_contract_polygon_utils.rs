use super::*;

pub(super) fn segments_share_endpoint(
    a0: [f32; 2],
    a1: [f32; 2],
    b0: [f32; 2],
    b1: [f32; 2],
) -> bool {
    points_close2(a0, b0) || points_close2(a0, b1) || points_close2(a1, b0) || points_close2(a1, b1)
}

pub(super) fn segments_intersect(a0: [f32; 2], a1: [f32; 2], b0: [f32; 2], b1: [f32; 2]) -> bool {
    fn orient(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    }
    let o1 = orient(a0, a1, b0);
    let o2 = orient(a0, a1, b1);
    let o3 = orient(b0, b1, a0);
    let o4 = orient(b0, b1, a1);
    o1 * o2 < -1.0e-6 && o3 * o4 < -1.0e-6
}

pub(super) fn point_in_polygon_or_boundary(point: [f32; 2], polygon: &[[f32; 2]]) -> bool {
    point_in_polygon(point, polygon)
        || (0..polygon.len())
            .any(|idx| point_on_segment(point, polygon[idx], polygon[(idx + 1) % polygon.len()]))
}

pub(super) fn point_on_segment(point: [f32; 2], a: [f32; 2], b: [f32; 2]) -> bool {
    let projected = nearest_point_on_segment(point, a, b);
    squared_distance2(point, projected) <= 1.0e-6
}

pub(super) fn points_close2(left: [f32; 2], right: [f32; 2]) -> bool {
    squared_distance2(left, right) <= 1.0e-10
}

pub(super) fn polygon_area(points: &[[f32; 2]]) -> f32 {
    signed_polygon_area(points).abs()
}

pub(super) fn signed_polygon_area(points: &[[f32; 2]]) -> f32 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut doubled = 0.0f32;
    for idx in 0..points.len() {
        let next = (idx + 1) % points.len();
        doubled += points[idx][0] * points[next][1] - points[next][0] * points[idx][1];
    }
    doubled * 0.5
}

pub(super) fn polygon_perimeter(points: &[[f32; 2]]) -> f32 {
    if points.len() < 2 {
        return 0.0;
    }
    (0..points.len())
        .map(|idx| distance2(points[idx], points[(idx + 1) % points.len()]))
        .sum()
}

pub(super) fn distance2(left: [f32; 2], right: [f32; 2]) -> f32 {
    squared_distance2(left, right).sqrt()
}

pub(super) fn squared_distance2(left: [f32; 2], right: [f32; 2]) -> f32 {
    let dx = left[0] - right[0];
    let dy = left[1] - right[1];
    dx * dx + dy * dy
}

pub(super) fn axis_aligned_rectangle_polygon_bounds(
    points: &[[f32; 2]],
) -> Option<(f32, f32, f32, f32)> {
    if points.len() != 4 {
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
    Some((xmin, xmax, ymin, ymax))
}

pub(super) fn rectangle_area(bounds: (f32, f32, f32, f32)) -> f32 {
    (bounds.1 - bounds.0).max(0.0) * (bounds.3 - bounds.2).max(0.0)
}

pub(super) fn rounded_rectangle_dilation_area(bounds: (f32, f32, f32, f32), radius: f32) -> f32 {
    if radius <= 0.0 {
        return rectangle_area(bounds);
    }
    rectangle_area(bounds)
        + rectangle_perimeter(bounds) * radius
        + std::f32::consts::PI * radius.powi(2)
}

pub(super) fn rectangle_perimeter(bounds: (f32, f32, f32, f32)) -> f32 {
    2.0 * ((bounds.1 - bounds.0).max(0.0) + (bounds.3 - bounds.2).max(0.0))
}

pub(super) fn rectangle_bounds_strictly_contains(
    outer: (f32, f32, f32, f32),
    inner: (f32, f32, f32, f32),
) -> bool {
    inner.0 > outer.0 + 1.0e-5
        && inner.1 < outer.1 - 1.0e-5
        && inner.2 > outer.2 + 1.0e-5
        && inner.3 < outer.3 - 1.0e-5
}

pub(super) fn rectangle_containment_gap(
    outer: (f32, f32, f32, f32),
    inner: (f32, f32, f32, f32),
) -> f32 {
    (inner.0 - outer.0)
        .min(outer.1 - inner.1)
        .min(inner.2 - outer.2)
        .min(outer.3 - inner.3)
}

pub(super) fn rectangle_bounds_distance(
    left: (f32, f32, f32, f32),
    right: (f32, f32, f32, f32),
) -> f32 {
    let dx = if left.1 < right.0 {
        right.0 - left.1
    } else if right.1 < left.0 {
        left.0 - right.1
    } else {
        0.0
    };
    let dy = if left.3 < right.2 {
        right.2 - left.3
    } else if right.3 < left.2 {
        left.2 - right.3
    } else {
        0.0
    };
    (dx * dx + dy * dy).sqrt()
}

pub(super) fn nested_polygon_forest_contains_point(
    rings: &[Vec<[f32; 2]>],
    inset_angstrom: f32,
    point: [f32; 2],
    margin_angstrom: f32,
) -> bool {
    let depths = nested_polygon_ring_depths(rings);
    for (ring, depth) in rings.iter().zip(depths.iter()) {
        if point_in_polygon_or_boundary(point, ring) && !point_in_polygon(point, ring) {
            return depth % 2 == 0 && inset_angstrom <= 1.0e-6 && margin_angstrom <= 1.0e-6;
        }
    }
    let containing_count = rings
        .iter()
        .filter(|ring| point_in_polygon(point, ring))
        .count();
    if containing_count % 2 == 0 {
        return false;
    }
    let effective_margin = if inset_angstrom > 0.0 {
        inset_angstrom + margin_angstrom.max(0.0)
    } else {
        margin_angstrom.max(0.0)
    };
    effective_margin <= 1.0e-6
        || rings
            .iter()
            .all(|ring| polygon_boundary_distance(point, ring) >= effective_margin - 1.0e-5)
}

pub(super) fn project_point_to_nested_polygon_forest(
    point: [f32; 2],
    rings: &[Vec<[f32; 2]>],
    inset_angstrom: f32,
    radius: f32,
) -> [f32; 2] {
    let margin = if inset_angstrom > 0.0 {
        inset_angstrom + radius
    } else {
        radius
    };
    if nested_polygon_forest_contains_point(rings, inset_angstrom, point, radius) {
        return point;
    }
    let depths = nested_polygon_ring_depths(rings);
    let mut candidates = Vec::new();
    for (ring, depth) in rings.iter().zip(depths.iter()) {
        if depth % 2 != 0 {
            continue;
        }
        let mut candidate = project_point_to_polygon_with_margin(point, ring, margin);
        for (hole, hole_depth) in rings.iter().zip(depths.iter()) {
            if hole_depth % 2 == 1 && polygon_hole_rejects_point(hole, margin, candidate) {
                if let Some(projected) =
                    project_point_outside_polygon_hole(candidate, ring, hole, margin)
                {
                    candidate = projected;
                }
            }
        }
        if nested_polygon_forest_contains_point(rings, inset_angstrom, candidate, radius) {
            candidates.push(candidate);
        }
    }
    candidates
        .into_iter()
        .min_by(|left, right| {
            squared_distance2(point, *left)
                .partial_cmp(&squared_distance2(point, *right))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or_else(|| {
            rings
                .iter()
                .zip(depths.iter())
                .filter(|(_, depth)| **depth % 2 == 0)
                .map(|(ring, _)| nearest_point_on_polygon_boundary(point, ring))
                .min_by(|left, right| {
                    squared_distance2(point, *left)
                        .partial_cmp(&squared_distance2(point, *right))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(point)
        })
}

pub(super) fn nested_polygon_ring_depths(rings: &[Vec<[f32; 2]>]) -> Vec<usize> {
    rings
        .iter()
        .enumerate()
        .map(|(idx, ring)| {
            ring.first()
                .map(|point| {
                    rings
                        .iter()
                        .enumerate()
                        .filter(|(other_idx, other)| {
                            *other_idx != idx && point_in_polygon(*point, other)
                        })
                        .count()
                })
                .unwrap_or(0)
        })
        .collect()
}

pub(super) fn convex_polygon_inset_polygon(
    points: &[[f32; 2]],
    inset_angstrom: f32,
) -> Option<Vec<[f32; 2]>> {
    if points.len() < 3 || inset_angstrom < 0.0 || !polygon_is_convex(points) {
        return None;
    }
    let mut inset = points.to_vec();
    ensure_ccw_polygon(&mut inset);
    if inset_angstrom <= 1.0e-6 {
        return Some(inset);
    }
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
            return Some(Vec::new());
        }
    }
    Some(inset)
}

pub(super) fn exact_simple_polygon_inset_area_before_topology_event(
    points: &[[f32; 2]],
    inset_angstrom: f32,
) -> Option<f32> {
    if points.len() < 3 || inset_angstrom < 0.0 || polygon_has_self_intersections(points) {
        return None;
    }
    if inset_angstrom <= 1.0e-6 {
        return Some(polygon_area(points));
    }
    let clearance = polygon_non_adjacent_edge_clearance(points)?;
    if clearance <= 2.0 * inset_angstrom + 1.0e-5 {
        return None;
    }
    let mut oriented = points.to_vec();
    ensure_ccw_polygon(&mut oriented);
    let corner_sum = polygon_inset_corner_area_coefficient_sum(&oriented)?;
    let area = polygon_area(&oriented) - polygon_perimeter(&oriented) * inset_angstrom
        + corner_sum * inset_angstrom.powi(2);
    (area >= -1.0e-4).then_some(area.max(0.0))
}

pub(super) fn polygon_non_adjacent_edge_clearance(points: &[[f32; 2]]) -> Option<f32> {
    if points.len() < 3 {
        return None;
    }
    let mut clearance = f32::INFINITY;
    for left_idx in 0..points.len() {
        let left_start = points[left_idx];
        let left_end = points[(left_idx + 1) % points.len()];
        if squared_distance2(left_start, left_end) <= 1.0e-10 {
            return None;
        }
        for right_idx in (left_idx + 1)..points.len() {
            if polygon_edges_are_adjacent(left_idx, right_idx, points.len()) {
                continue;
            }
            let right_start = points[right_idx];
            let right_end = points[(right_idx + 1) % points.len()];
            if squared_distance2(right_start, right_end) <= 1.0e-10 {
                return None;
            }
            clearance = clearance.min(segment_segment_distance(
                left_start,
                left_end,
                right_start,
                right_end,
            ));
        }
    }
    clearance.is_finite().then_some(clearance)
}

pub(super) fn polygon_edges_are_adjacent(
    left_idx: usize,
    right_idx: usize,
    edge_count: usize,
) -> bool {
    left_idx == right_idx
        || left_idx + 1 == right_idx
        || right_idx + 1 == left_idx
        || (left_idx == 0 && right_idx + 1 == edge_count)
        || (right_idx == 0 && left_idx + 1 == edge_count)
}

pub(super) fn segment_segment_distance(
    left_start: [f32; 2],
    left_end: [f32; 2],
    right_start: [f32; 2],
    right_end: [f32; 2],
) -> f32 {
    if segments_intersect(left_start, left_end, right_start, right_end) {
        return 0.0;
    }
    point_segment_distance(left_start, right_start, right_end)
        .min(point_segment_distance(left_end, right_start, right_end))
        .min(point_segment_distance(right_start, left_start, left_end))
        .min(point_segment_distance(right_end, left_start, left_end))
}

pub(super) fn polygon_inset_corner_area_coefficient_sum(points: &[[f32; 2]]) -> Option<f32> {
    let mut sum = 0.0f32;
    for idx in 0..points.len() {
        let previous = points[(idx + points.len() - 1) % points.len()];
        let current = points[idx];
        let next = points[(idx + 1) % points.len()];
        let incoming = [current[0] - previous[0], current[1] - previous[1]];
        let outgoing = [next[0] - current[0], next[1] - current[1]];
        let incoming_len = (incoming[0].powi(2) + incoming[1].powi(2)).sqrt();
        let outgoing_len = (outgoing[0].powi(2) + outgoing[1].powi(2)).sqrt();
        if incoming_len <= 1.0e-8 || outgoing_len <= 1.0e-8 {
            return None;
        }
        let cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0];
        let dot = incoming[0] * outgoing[0] + incoming[1] * outgoing[1];
        let turn = cross.atan2(dot);
        let interior = if turn > 0.0 {
            std::f32::consts::PI - turn
        } else {
            std::f32::consts::PI - turn
        };
        if interior <= std::f32::consts::PI {
            let tan_half = (0.5 * interior).tan();
            if tan_half.abs() <= 1.0e-8 {
                return None;
            }
            sum += 1.0 / tan_half;
        } else {
            sum -= 0.5 * (interior - std::f32::consts::PI);
        }
    }
    Some(sum)
}

pub(super) fn polygon_distance(left: &[[f32; 2]], right: &[[f32; 2]]) -> f32 {
    if left.len() < 2 || right.len() < 2 {
        return f32::INFINITY;
    }
    if left
        .iter()
        .any(|point| point_in_polygon_or_boundary(*point, right))
        || right
            .iter()
            .any(|point| point_in_polygon_or_boundary(*point, left))
    {
        return 0.0;
    }
    let mut distance = f32::INFINITY;
    for left_idx in 0..left.len() {
        let a = left[left_idx];
        let b = left[(left_idx + 1) % left.len()];
        for right_idx in 0..right.len() {
            let c = right[right_idx];
            let d = right[(right_idx + 1) % right.len()];
            if segments_intersect(a, b, c, d) {
                return 0.0;
            }
            distance = distance.min(point_segment_distance(a, c, d));
            distance = distance.min(point_segment_distance(b, c, d));
            distance = distance.min(point_segment_distance(c, a, b));
            distance = distance.min(point_segment_distance(d, a, b));
        }
    }
    distance
}

pub(super) fn polygon_boundary_gap(left: &[[f32; 2]], right: &[[f32; 2]]) -> f32 {
    if left.len() < 2 || right.len() < 2 {
        return f32::INFINITY;
    }
    let mut distance = f32::INFINITY;
    for left_idx in 0..left.len() {
        let a = left[left_idx];
        let b = left[(left_idx + 1) % left.len()];
        for right_idx in 0..right.len() {
            let c = right[right_idx];
            let d = right[(right_idx + 1) % right.len()];
            distance = distance.min(segment_segment_distance(a, b, c, d));
        }
    }
    distance
}

pub(super) fn point_segment_distance(point: [f32; 2], start: [f32; 2], end: [f32; 2]) -> f32 {
    let nearest = nearest_point_on_segment(point, start, end);
    ((point[0] - nearest[0]).powi(2) + (point[1] - nearest[1]).powi(2)).sqrt()
}

pub(super) fn polygon_boundary_contains_point(
    points: &[[f32; 2]],
    inset_angstrom: f32,
    point: [f32; 2],
    margin_angstrom: f32,
) -> bool {
    let effective_margin = if inset_angstrom > 0.0 {
        inset_angstrom + margin_angstrom.max(0.0)
    } else {
        0.0
    };
    let inside = if effective_margin <= 1.0e-6 {
        point_in_polygon_or_boundary(point, points)
    } else {
        point_in_polygon(point, points)
    };
    inside && polygon_boundary_distance(point, points) >= effective_margin - 1.0e-5
}

pub(super) fn polygon_hole_rejects_point(
    hole: &[[f32; 2]],
    margin_angstrom: f32,
    point: [f32; 2],
) -> bool {
    if hole.len() < 3 {
        return false;
    }
    point_in_polygon_or_boundary(point, hole)
        || (margin_angstrom > 1.0e-6
            && polygon_boundary_distance(point, hole) < margin_angstrom - 1.0e-5)
}

pub(super) fn project_point_outside_polygon_hole(
    point: [f32; 2],
    outer: &[[f32; 2]],
    hole: &[[f32; 2]],
    margin_angstrom: f32,
) -> Option<[f32; 2]> {
    if hole.len() < 3 {
        return Some(point);
    }
    let nearest = nearest_point_on_polygon_boundary(point, hole);
    let centroid = polygon_centroid(hole);
    let mut direction = [nearest[0] - centroid[0], nearest[1] - centroid[1]];
    let mut norm = (direction[0].powi(2) + direction[1].powi(2)).sqrt();
    if norm <= 1.0e-6 {
        direction = [point[0] - centroid[0], point[1] - centroid[1]];
        norm = (direction[0].powi(2) + direction[1].powi(2)).sqrt();
    }
    if norm <= 1.0e-6 {
        return None;
    }
    let offset = margin_angstrom.max(0.0) + 1.0e-2;
    let candidate = [
        nearest[0] + direction[0] / norm * offset,
        nearest[1] + direction[1] / norm * offset,
    ];
    (point_in_polygon_or_boundary(candidate, outer)
        && !polygon_hole_rejects_point(hole, margin_angstrom, candidate))
    .then_some(candidate)
}
