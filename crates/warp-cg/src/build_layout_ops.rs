use std::collections::{HashMap, HashSet};

use super::{
    CircularExclusion, LayoutBasis2D, LayoutBounds, LayoutPeriodicity, LayoutPoint,
    OptimizerTraceSample, PlacementMetrics,
};

pub(super) fn neighbor_pairs_with_periodicity_basis(
    points: &[LayoutPoint],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> Vec<(usize, usize)> {
    if points.len() < 2 {
        return Vec::new();
    }
    if basis.is_some() && periodicity.x && periodicity.y {
        return all_neighbor_pairs(points.len());
    }
    let largest_radius = points
        .iter()
        .map(|point| point.radius)
        .fold(0.0f32, f32::max);
    let bin_size = (largest_radius * 4.0).max(1.0);
    let x_len = bounds.xmax - bounds.xmin;
    let y_len = bounds.ymax - bounds.ymin;
    let x_bins = if periodicity.x && x_len > 0.0 {
        ((x_len / bin_size).ceil() as i32).max(1)
    } else {
        0
    };
    let y_bins = if periodicity.y && y_len > 0.0 {
        ((y_len / bin_size).ceil() as i32).max(1)
    } else {
        0
    };
    let mut bins: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (idx, point) in points.iter().enumerate() {
        bins.entry(bin_index_periodic(
            point.x,
            point.y,
            bounds,
            bin_size,
            periodicity,
            x_bins,
            y_bins,
        ))
        .or_default()
        .push(idx);
    }

    let mut pairs = Vec::new();
    let mut seen = HashSet::new();
    for (&(bx, by), indices) in &bins {
        for dx in -1..=1 {
            for dy in -1..=1 {
                let nbx = neighbor_bin_index(bx + dx, x_bins, periodicity.x);
                let nby = neighbor_bin_index(by + dy, y_bins, periodicity.y);
                let Some(other) = bins.get(&(nbx, nby)) else {
                    continue;
                };
                for &i in indices {
                    for &j in other {
                        if i < j && seen.insert((i, j)) {
                            pairs.push((i, j));
                        }
                    }
                }
            }
        }
    }
    pairs
}

pub(super) fn all_neighbor_pairs(len: usize) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(len.saturating_mul(len.saturating_sub(1)) / 2);
    for i in 0..len {
        for j in (i + 1)..len {
            pairs.push((i, j));
        }
    }
    pairs
}

pub(super) fn bin_index_periodic(
    x: f32,
    y: f32,
    bounds: LayoutBounds,
    bin_size: f32,
    periodicity: LayoutPeriodicity,
    x_bins: i32,
    y_bins: i32,
) -> (i32, i32) {
    let bx = if periodicity.x && x_bins > 0 {
        ((wrap_coordinate(x, bounds.xmin, bounds.xmax) - bounds.xmin) / bin_size).floor() as i32
    } else {
        (x / bin_size).floor() as i32
    };
    let by = if periodicity.y && y_bins > 0 {
        ((wrap_coordinate(y, bounds.ymin, bounds.ymax) - bounds.ymin) / bin_size).floor() as i32
    } else {
        (y / bin_size).floor() as i32
    };
    (
        if periodicity.x {
            bx.rem_euclid(x_bins.max(1))
        } else {
            bx
        },
        if periodicity.y {
            by.rem_euclid(y_bins.max(1))
        } else {
            by
        },
    )
}

pub(super) fn neighbor_bin_index(index: i32, bin_count: i32, periodic: bool) -> i32 {
    if periodic && bin_count > 0 {
        index.rem_euclid(bin_count)
    } else {
        index
    }
}

pub(super) fn apply_edge_push(
    point: &mut LayoutPoint,
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    multiplier: f32,
    bounce_counter: &mut f32,
) -> f32 {
    let mut push_x = 0.0f32;
    let mut push_y = 0.0f32;
    let min_x = bounds.xmin + point.radius;
    let max_x = bounds.xmax - point.radius;
    let min_y = bounds.ymin + point.radius;
    let max_y = bounds.ymax - point.radius;
    if !periodicity.x && point.x < min_x {
        push_x = (min_x - point.x) * multiplier;
        *bounce_counter += 1.0;
    } else if !periodicity.x && point.x > max_x {
        push_x = (max_x - point.x) * multiplier;
        *bounce_counter += 1.0;
    }
    if !periodicity.y && point.y < min_y {
        push_y = (min_y - point.y) * multiplier;
        *bounce_counter += 1.0;
    } else if !periodicity.y && point.y > max_y {
        push_y = (max_y - point.y) * multiplier;
        *bounce_counter += 1.0;
    }
    if *bounce_counter > 1.0 {
        push_x *= *bounce_counter;
        push_y *= *bounce_counter;
    }
    point.x += push_x;
    point.y += push_y;
    (push_x * push_x + push_y * push_y).sqrt()
}

pub(super) fn apply_exclusion_push(
    point: &mut LayoutPoint,
    exclusions: &[CircularExclusion],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
    multiplier: f32,
    bounce_counter: &mut f32,
) -> f32 {
    let mut max_push = 0.0f32;
    for exclusion in exclusions {
        let [dx, dy] = layout_delta_from_xy(
            [point.x, point.y],
            [exclusion.x, exclusion.y],
            bounds,
            periodicity,
            basis,
        );
        let dist = (dx * dx + dy * dy).sqrt();
        let min_dist = exclusion.radius + point.radius;
        if dist >= min_dist {
            continue;
        }
        let (ux, uy) = if dist > f32::EPSILON {
            (dx / dist, dy / dist)
        } else {
            deterministic_unit_vector(point.x.to_bits() as usize, point.y.to_bits() as usize)
        };
        let push = if dist > f32::EPSILON {
            (min_dist - dist) * multiplier
        } else {
            min_dist
        };
        *bounce_counter += 1.0;
        let push = push * (*bounce_counter).max(1.0);
        let px = ux * push;
        let py = uy * push;
        point.x += px;
        point.y += py;
        max_push = max_push.max((px * px + py * py).sqrt());
    }
    max_push
}

pub(super) fn placement_metrics(
    points: &[LayoutPoint],
    bounds: LayoutBounds,
    exclusions: &[CircularExclusion],
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
    relaxation_enabled: bool,
    relaxation_steps: usize,
    max_push_angstrom: f32,
    optimizer_neighbor_cutoff_angstrom: Option<f32>,
    neighbor_search_rebuild_count: usize,
    initial_points: &[(f32, f32)],
    optimizer_trace: Vec<OptimizerTraceSample>,
) -> PlacementMetrics {
    let (max_total_displacement_angstrom, mean_total_displacement_angstrom) =
        displacement_metrics(points, initial_points, bounds, periodicity, basis);
    PlacementMetrics {
        relaxation_enabled,
        relaxation_steps,
        max_push_angstrom,
        trajectory_frame_count: optimizer_trace.len(),
        optimizer_neighbor_cutoff_angstrom,
        neighbor_search_rebuild_count,
        max_total_displacement_angstrom,
        mean_total_displacement_angstrom,
        optimizer_trace,
        min_pair_clearance_angstrom: min_pair_clearance(points, bounds, periodicity, basis),
        max_edge_violation_angstrom: max_edge_violation(points, bounds, periodicity, basis),
        max_exclusion_violation_angstrom: max_exclusion_violation(
            points,
            exclusions,
            bounds,
            periodicity,
            basis,
        ),
    }
}

pub(super) fn optimizer_trace_sample(
    points: &[LayoutPoint],
    bounds: LayoutBounds,
    exclusions: &[CircularExclusion],
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
    step: usize,
    step_multiplier: f32,
    max_push_angstrom: f32,
    initial_points: &[(f32, f32)],
) -> OptimizerTraceSample {
    let (max_total_displacement_angstrom, mean_total_displacement_angstrom) =
        displacement_metrics(points, initial_points, bounds, periodicity, basis);
    OptimizerTraceSample {
        step,
        step_multiplier,
        max_push_angstrom,
        max_total_displacement_angstrom,
        mean_total_displacement_angstrom,
        min_pair_clearance_angstrom: min_pair_clearance(points, bounds, periodicity, basis),
        max_edge_violation_angstrom: max_edge_violation(points, bounds, periodicity, basis),
        max_exclusion_violation_angstrom: max_exclusion_violation(
            points,
            exclusions,
            bounds,
            periodicity,
            basis,
        ),
    }
}

pub(super) fn optimizer_neighbor_cutoff(points: &[LayoutPoint]) -> Option<f32> {
    points
        .iter()
        .map(|point| point.radius)
        .reduce(f32::max)
        .map(|largest_radius| (largest_radius * 4.0).max(1.0))
}

pub(super) fn displacement_metrics(
    points: &[LayoutPoint],
    initial_points: &[(f32, f32)],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> (f32, f32) {
    if points.is_empty() || initial_points.len() != points.len() {
        return (0.0, 0.0);
    }
    let mut max_displacement = 0.0f32;
    let mut sum_displacement = 0.0f32;
    for (point, (initial_x, initial_y)) in points.iter().zip(initial_points.iter()) {
        let [dx, dy] = layout_delta_from_xy(
            [point.x, point.y],
            [*initial_x, *initial_y],
            bounds,
            periodicity,
            basis,
        );
        let displacement = (dx * dx + dy * dy).sqrt();
        max_displacement = max_displacement.max(displacement);
        sum_displacement += displacement;
    }
    (max_displacement, sum_displacement / points.len() as f32)
}

pub(super) fn max_displacement_from_reference(
    points: &[LayoutPoint],
    reference_points: &[(f32, f32)],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> f32 {
    if reference_points.len() != points.len() {
        return 0.0;
    }
    points
        .iter()
        .zip(reference_points.iter())
        .map(|(point, (reference_x, reference_y))| {
            let [dx, dy] = layout_delta_from_xy(
                [point.x, point.y],
                [*reference_x, *reference_y],
                bounds,
                periodicity,
                basis,
            );
            (dx * dx + dy * dy).sqrt()
        })
        .fold(0.0, f32::max)
}

pub(super) fn min_pair_clearance(
    points: &[LayoutPoint],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> Option<f32> {
    if points.len() < 2 {
        return None;
    }
    let mut min_clearance = f32::INFINITY;
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let [dx, dy] = layout_delta(points[j], points[i], bounds, periodicity, basis);
            let dist = (dx * dx + dy * dy).sqrt();
            min_clearance = min_clearance.min(dist - points[i].radius - points[j].radius);
        }
    }
    Some(min_clearance)
}

pub(super) fn max_edge_violation(
    points: &[LayoutPoint],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> f32 {
    if basis.is_some() && periodicity.x && periodicity.y {
        return 0.0;
    }
    points.iter().fold(0.0, |max_violation, point| {
        let violations = [
            if periodicity.x {
                0.0
            } else {
                bounds.xmin + point.radius - point.x
            },
            if periodicity.x {
                0.0
            } else {
                point.x - (bounds.xmax - point.radius)
            },
            if periodicity.y {
                0.0
            } else {
                bounds.ymin + point.radius - point.y
            },
            if periodicity.y {
                0.0
            } else {
                point.y - (bounds.ymax - point.radius)
            },
        ];
        max_violation.max(
            violations
                .into_iter()
                .filter(|value| *value > 0.0)
                .fold(0.0, f32::max),
        )
    })
}

pub(super) fn minimum_image_delta(delta: f32, length: f32, periodic: bool) -> f32 {
    if periodic && length > 0.0 && length.is_finite() {
        delta - length * (delta / length).round()
    } else {
        delta
    }
}

pub(super) fn wrap_point_into_periodic_bounds(
    point: &mut LayoutPoint,
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) {
    if let Some(basis) = basis.filter(|_| periodicity.x && periodicity.y) {
        let fractional = basis.fractional([point.x, point.y]);
        let wrapped =
            basis.cartesian([fractional[0].rem_euclid(1.0), fractional[1].rem_euclid(1.0)]);
        point.x = wrapped[0];
        point.y = wrapped[1];
        return;
    }
    if periodicity.x {
        point.x = wrap_coordinate(point.x, bounds.xmin, bounds.xmax);
    }
    if periodicity.y {
        point.y = wrap_coordinate(point.y, bounds.ymin, bounds.ymax);
    }
}

pub(super) fn wrap_coordinate(value: f32, min: f32, max: f32) -> f32 {
    let length = max - min;
    if length <= 0.0 || !length.is_finite() {
        return value;
    }
    min + (value - min).rem_euclid(length)
}

pub(super) fn max_exclusion_violation(
    points: &[LayoutPoint],
    exclusions: &[CircularExclusion],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> f32 {
    points.iter().fold(0.0, |max_violation, point| {
        max_violation.max(exclusions.iter().fold(0.0, |inner_max, exclusion| {
            let [dx, dy] = layout_delta_from_xy(
                [point.x, point.y],
                [exclusion.x, exclusion.y],
                bounds,
                periodicity,
                basis,
            );
            let dist = (dx * dx + dy * dy).sqrt();
            inner_max.max(exclusion.radius + point.radius - dist)
        }))
    })
}

impl LayoutBasis2D {
    pub(crate) fn new(origin: [f32; 2], a: [f32; 2], b: [f32; 2]) -> Option<Self> {
        let det = a[0] * b[1] - a[1] * b[0];
        if det.abs() <= 1.0e-8
            || !origin
                .iter()
                .chain(a.iter())
                .chain(b.iter())
                .all(|v| v.is_finite())
        {
            return None;
        }
        Some(Self { origin, a, b })
    }

    pub(crate) fn fractional(self, point: [f32; 2]) -> [f32; 2] {
        let dx = point[0] - self.origin[0];
        let dy = point[1] - self.origin[1];
        let det = self.a[0] * self.b[1] - self.a[1] * self.b[0];
        [
            (dx * self.b[1] - dy * self.b[0]) / det,
            (self.a[0] * dy - self.a[1] * dx) / det,
        ]
    }

    pub(crate) fn cartesian(self, fractional: [f32; 2]) -> [f32; 2] {
        [
            self.origin[0] + fractional[0] * self.a[0] + fractional[1] * self.b[0],
            self.origin[1] + fractional[0] * self.a[1] + fractional[1] * self.b[1],
        ]
    }

    pub(crate) fn minimum_image_delta(self, left: [f32; 2], right: [f32; 2]) -> [f32; 2] {
        let mut left_f = self.fractional(left);
        let right_f = self.fractional(right);
        left_f[0] -= right_f[0];
        left_f[1] -= right_f[1];
        left_f[0] -= left_f[0].round();
        left_f[1] -= left_f[1].round();
        [
            left_f[0] * self.a[0] + left_f[1] * self.b[0],
            left_f[0] * self.a[1] + left_f[1] * self.b[1],
        ]
    }
}

pub(super) fn layout_delta(
    left: LayoutPoint,
    right: LayoutPoint,
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> [f32; 2] {
    layout_delta_from_xy(
        [left.x, left.y],
        [right.x, right.y],
        bounds,
        periodicity,
        basis,
    )
}

pub(super) fn layout_delta_from_xy(
    left: [f32; 2],
    right: [f32; 2],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> [f32; 2] {
    if let Some(basis) = basis.filter(|_| periodicity.x && periodicity.y) {
        return basis.minimum_image_delta(left, right);
    }
    [
        minimum_image_delta(left[0] - right[0], bounds.xmax - bounds.xmin, periodicity.x),
        minimum_image_delta(left[1] - right[1], bounds.ymax - bounds.ymin, periodicity.y),
    ]
}

pub(super) fn deterministic_unit_vector(i: usize, j: usize) -> (f32, f32) {
    let angle = ((i * 31 + j * 17 + 1) as f32) * 0.618_034;
    (angle.cos(), angle.sin())
}
