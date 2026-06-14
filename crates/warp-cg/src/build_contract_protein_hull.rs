use super::*;

pub(super) fn buffered_convex_hull_area(points: &[[f32; 2]], buffer_angstrom: f32) -> f32 {
    let mut unique = points.to_vec();
    unique.sort_by(|left, right| {
        left[0]
            .partial_cmp(&right[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left[1]
                    .partial_cmp(&right[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    unique.dedup_by(|left, right| {
        (left[0] - right[0]).abs() < 1.0e-5 && (left[1] - right[1]).abs() < 1.0e-5
    });

    match unique.len() {
        0 => 0.0,
        1 => std::f32::consts::PI * buffer_angstrom.powi(2),
        2 => {
            let length = distance2(unique[0], unique[1]);
            2.0 * buffer_angstrom * length + std::f32::consts::PI * buffer_angstrom.powi(2)
        }
        _ => {
            let hull = convex_hull(unique);
            polygon_area(&hull)
                + polygon_perimeter(&hull) * buffer_angstrom
                + std::f32::consts::PI * buffer_angstrom.powi(2)
        }
    }
}

pub(super) fn convex_hull_area(points: &[[f32; 2]]) -> f32 {
    let mut unique = points.to_vec();
    unique.sort_by(|left, right| {
        left[0]
            .partial_cmp(&right[0])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left[1]
                    .partial_cmp(&right[1])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    unique.dedup_by(|left, right| {
        (left[0] - right[0]).abs() < 1.0e-5 && (left[1] - right[1]).abs() < 1.0e-5
    });
    if unique.len() < 3 {
        return 0.0;
    }
    polygon_area(&convex_hull(unique))
}

pub(super) fn convex_hull(points: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    fn cross(origin: [f32; 2], left: [f32; 2], right: [f32; 2]) -> f32 {
        (left[0] - origin[0]) * (right[1] - origin[1])
            - (left[1] - origin[1]) * (right[0] - origin[0])
    }

    let mut lower = Vec::new();
    for point in &points {
        while lower.len() >= 2
            && cross(lower[lower.len() - 2], lower[lower.len() - 1], *point) <= 0.0
        {
            lower.pop();
        }
        lower.push(*point);
    }

    let mut upper = Vec::new();
    for point in points.iter().rev() {
        while upper.len() >= 2
            && cross(upper[upper.len() - 2], upper[upper.len() - 1], *point) <= 0.0
        {
            upper.pop();
        }
        upper.push(*point);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

pub(super) fn concave_hull(points: Vec<[f32; 2]>) -> Vec<[f32; 2]> {
    if points.len() < 4 {
        return convex_hull(points);
    }
    if let Some(ordered) = ordered_concave_boundary(&points) {
        return ordered;
    }
    let max_k = points.len().saturating_sub(1).max(3);
    for k in 3..=max_k {
        if let Some(hull) = concave_hull_with_k(&points, k) {
            if hull.len() >= 3
                && points
                    .iter()
                    .all(|point| point_in_polygon_or_boundary(*point, &hull))
            {
                return hull;
            }
        }
    }
    convex_hull(points)
}

pub(super) fn alpha_shape(points: &[[f32; 2]], alpha_radius: f32) -> Option<Vec<[f32; 2]>> {
    let loop_candidate = alpha_shape_components(points, alpha_radius).and_then(|loops| {
        loops.into_iter().max_by(|left, right| {
            left.len().cmp(&right.len()).then_with(|| {
                polygon_area(left)
                    .partial_cmp(&polygon_area(right))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        })
    });
    let walk_candidate = alpha_walk_boundary(points, alpha_radius * 2.0);
    match (loop_candidate, walk_candidate) {
        (Some(loop_polygon), Some(walk_polygon))
            if polygon_area(&walk_polygon) < polygon_area(&loop_polygon) =>
        {
            Some(walk_polygon)
        }
        (Some(loop_polygon), _) => Some(loop_polygon),
        (None, walk_polygon) => walk_polygon,
    }
}

pub(super) fn alpha_shape_components(
    points: &[[f32; 2]],
    alpha_radius: f32,
) -> Option<Vec<Vec<[f32; 2]>>> {
    if points.len() < 4 || alpha_radius <= 0.0 || !alpha_radius.is_finite() {
        return None;
    }
    let mut edges = BTreeSet::new();
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            if alpha_edge_is_exposed(points, i, j, alpha_radius) {
                edges.insert((i, j));
            }
        }
    }
    let mut loops = alpha_boundary_loops(points, &edges);
    loops.sort_by(|left, right| {
        polygon_area(right)
            .partial_cmp(&polygon_area(left))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| right.len().cmp(&left.len()))
    });
    (!loops.is_empty()).then_some(loops)
}

pub(super) fn default_alpha_radius(points: &[[f32; 2]]) -> f32 {
    let mut nearest = points
        .iter()
        .enumerate()
        .filter_map(|(idx, point)| {
            points
                .iter()
                .enumerate()
                .filter(|(jdx, _)| *jdx != idx)
                .map(|(_, other)| distance2(*point, *other))
                .filter(|distance| *distance > 1.0e-5)
                .min_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal))
        })
        .collect::<Vec<_>>();
    if nearest.is_empty() {
        return 1.0;
    }
    nearest.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    (nearest[nearest.len() / 2] * 1.75).max(1.0)
}

pub(super) fn alpha_edge_is_exposed(
    points: &[[f32; 2]],
    i: usize,
    j: usize,
    alpha_radius: f32,
) -> bool {
    let a = points[i];
    let b = points[j];
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let distance_sq = dx * dx + dy * dy;
    if distance_sq <= 1.0e-10 {
        return false;
    }
    let distance = distance_sq.sqrt();
    if distance > alpha_radius * 2.0 + 1.0e-5 {
        return false;
    }
    let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
    let half = distance * 0.5;
    let height = (alpha_radius * alpha_radius - half * half).max(0.0).sqrt();
    let perp = [-dy / distance, dx / distance];
    let centers = [
        [mid[0] + perp[0] * height, mid[1] + perp[1] * height],
        [mid[0] - perp[0] * height, mid[1] - perp[1] * height],
    ];
    centers.iter().any(|center| {
        points.iter().enumerate().all(|(idx, point)| {
            idx == i
                || idx == j
                || squared_distance2(*center, *point)
                    >= (alpha_radius * alpha_radius - 1.0e-4).max(0.0)
        })
    })
}

pub(super) fn alpha_boundary_loops(
    points: &[[f32; 2]],
    edges: &BTreeSet<(usize, usize)>,
) -> Vec<Vec<[f32; 2]>> {
    if edges.len() < 3 {
        return Vec::new();
    }
    let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
    for (left, right) in edges {
        adjacency.entry(*left).or_default().push(*right);
        adjacency.entry(*right).or_default().push(*left);
    }
    for neighbors in adjacency.values_mut() {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let mut visited = BTreeSet::new();
    let mut loops: Vec<Vec<[f32; 2]>> = Vec::new();
    for (left, right) in edges {
        for (start, next) in [(*left, *right), (*right, *left)] {
            if visited.contains(&(start, next)) {
                continue;
            }
            if let Some(loop_indices) =
                trace_alpha_loop(points, &adjacency, &mut visited, start, next)
            {
                let mut polygon = loop_indices
                    .iter()
                    .map(|idx| points[*idx])
                    .collect::<Vec<_>>();
                if signed_polygon_area(&polygon) < 0.0 {
                    polygon.reverse();
                }
                let unique_vertices = loop_indices.iter().collect::<BTreeSet<_>>().len();
                if polygon.len() >= 3
                    && unique_vertices == loop_indices.len()
                    && polygon_area(&polygon) > 1.0e-5
                    && !polygon_has_self_intersections(&polygon)
                    && !loops
                        .iter()
                        .any(|existing| polygons_have_same_vertices(existing, &polygon))
                {
                    loops.push(polygon);
                }
            }
        }
    }
    loops
}

pub(super) fn polygons_have_same_vertices(left: &[[f32; 2]], right: &[[f32; 2]]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().all(|point| {
        right
            .iter()
            .any(|other| squared_distance2(*point, *other) <= 1.0e-8)
    })
}

pub(super) fn trace_alpha_loop(
    points: &[[f32; 2]],
    adjacency: &HashMap<usize, Vec<usize>>,
    visited: &mut BTreeSet<(usize, usize)>,
    start: usize,
    next: usize,
) -> Option<Vec<usize>> {
    let mut out = vec![start];
    let mut previous = start;
    let mut current = next;
    let max_steps = points.len().saturating_mul(4).max(8);
    for _ in 0..max_steps {
        visited.insert((previous, current));
        if current == start {
            return (out.len() >= 3).then_some(out);
        }
        out.push(current);
        let neighbors = adjacency.get(&current)?;
        let previous_angle = segment_angle(points[previous], points[current]);
        let mut candidates = neighbors
            .iter()
            .copied()
            .filter(|candidate| {
                (*candidate != previous || neighbors.len() == 1)
                    && (*candidate != start || out.len() >= 3)
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            clockwise_turn(previous_angle, points[current], points[*left])
                .partial_cmp(&clockwise_turn(
                    previous_angle,
                    points[current],
                    points[*right],
                ))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
        let candidate = candidates
            .into_iter()
            .find(|candidate| !visited.contains(&(current, *candidate)))?;
        previous = current;
        current = candidate;
    }
    None
}

pub(super) fn alpha_walk_boundary(
    points: &[[f32; 2]],
    max_edge_angstrom: f32,
) -> Option<Vec<[f32; 2]>> {
    let start = points
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            left[1]
                .partial_cmp(&right[1])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left[0]
                        .partial_cmp(&right[0])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        })?
        .0;
    let mut hull = vec![points[start]];
    let mut used = vec![false; points.len()];
    used[start] = true;
    let mut current = start;
    let mut previous_angle = 0.0f32;
    let max_edge_sq = max_edge_angstrom * max_edge_angstrom;

    loop {
        let mut candidates = (0..points.len())
            .filter(|idx| {
                *idx != current
                    && (!used[*idx] || (*idx == start && hull.len() >= 3))
                    && squared_distance2(points[current], points[*idx]) <= max_edge_sq + 1.0e-5
            })
            .map(|idx| {
                (
                    idx,
                    clockwise_turn(previous_angle, points[current], points[idx]),
                    squared_distance2(points[current], points[idx]),
                )
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .1
                .partial_cmp(&left.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left.2
                        .partial_cmp(&right.2)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| left.0.cmp(&right.0))
        });

        let mut accepted = None;
        for (idx, _, _) in candidates {
            if idx == start && hull.len() < 3 {
                continue;
            }
            if hull_edge_intersects_existing(&hull, points[idx], idx == start) {
                continue;
            }
            accepted = Some(idx);
            break;
        }
        let next = accepted?;
        if next == start {
            break;
        }
        previous_angle = segment_angle(points[current], points[next]);
        current = next;
        used[current] = true;
        hull.push(points[current]);
        if hull.len() > points.len() {
            return None;
        }
    }

    (hull.len() >= 3 && polygon_area(&hull) > 1.0e-5).then_some(hull)
}

pub(super) fn ordered_concave_boundary(points: &[[f32; 2]]) -> Option<Vec<[f32; 2]>> {
    let mut out = Vec::new();
    for point in points {
        if !out.iter().any(|existing| points_close2(*existing, *point)) {
            out.push(*point);
        }
    }
    if out.len() < 3 || polygon_area(&out) <= 1.0e-6 || polygon_has_self_intersections(&out) {
        return None;
    }
    Some(out)
}

pub(super) fn polygon_has_self_intersections(points: &[[f32; 2]]) -> bool {
    if points.len() < 4 {
        return false;
    }
    for idx in 0..points.len() {
        let a0 = points[idx];
        let a1 = points[(idx + 1) % points.len()];
        for jdx in (idx + 1)..points.len() {
            if idx == jdx || (idx + 1) % points.len() == jdx || idx == (jdx + 1) % points.len() {
                continue;
            }
            let b0 = points[jdx];
            let b1 = points[(jdx + 1) % points.len()];
            if segments_intersect(a0, a1, b0, b1) {
                return true;
            }
        }
    }
    false
}

pub(super) fn concave_hull_with_k(points: &[[f32; 2]], k: usize) -> Option<Vec<[f32; 2]>> {
    let start = points
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            left[1]
                .partial_cmp(&right[1])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left[0]
                        .partial_cmp(&right[0])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        })?
        .0;
    let mut hull = vec![points[start]];
    let mut used = vec![false; points.len()];
    used[start] = true;
    let mut current = start;
    let mut previous_angle = 0.0f32;

    loop {
        let mut neighbors = (0..points.len())
            .filter(|idx| !used[*idx] || (*idx == start && hull.len() >= 3))
            .map(|idx| {
                (
                    idx,
                    squared_distance2(points[current], points[idx]),
                    clockwise_turn(previous_angle, points[current], points[idx]),
                )
            })
            .collect::<Vec<_>>();
        neighbors.sort_by(|left, right| {
            left.1
                .partial_cmp(&right.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left.0
                        .partial_cmp(&right.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        neighbors.truncate(k.min(neighbors.len()));
        neighbors.sort_by(|left, right| {
            right
                .2
                .partial_cmp(&left.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left.1
                        .partial_cmp(&right.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        let mut accepted = None;
        for (idx, _, _) in neighbors {
            if idx == current {
                continue;
            }
            let candidate = points[idx];
            if idx == start && hull.len() < 3 {
                continue;
            }
            if hull_edge_intersects_existing(&hull, candidate, idx == start) {
                continue;
            }
            accepted = Some(idx);
            break;
        }
        let next = accepted?;
        if next == start {
            break;
        }
        previous_angle = segment_angle(points[current], points[next]);
        current = next;
        used[current] = true;
        hull.push(points[current]);
        if hull.len() > points.len() {
            return None;
        }
    }
    Some(hull)
}

pub(super) fn clockwise_turn(previous_angle: f32, current: [f32; 2], candidate: [f32; 2]) -> f32 {
    let angle = segment_angle(current, candidate);
    (previous_angle - angle).rem_euclid(std::f32::consts::TAU)
}

pub(super) fn segment_angle(from: [f32; 2], to: [f32; 2]) -> f32 {
    (to[1] - from[1]).atan2(to[0] - from[0])
}

pub(super) fn hull_edge_intersects_existing(
    hull: &[[f32; 2]],
    candidate: [f32; 2],
    closing: bool,
) -> bool {
    if hull.len() < 2 {
        return false;
    }
    let start = hull[hull.len() - 1];
    let end = candidate;
    let limit = if closing {
        hull.len().saturating_sub(2)
    } else {
        hull.len().saturating_sub(1)
    };
    for idx in 0..limit {
        let a = hull[idx];
        let b = hull[idx + 1];
        if segments_share_endpoint(start, end, a, b) {
            continue;
        }
        if segments_intersect(start, end, a, b) {
            return true;
        }
    }
    false
}
