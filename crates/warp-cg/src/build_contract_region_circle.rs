use crate::build_layout::LayoutBounds;

use super::{polygon_area, squared_distance2, LeafletRegion, RegionGeometry};

#[derive(Clone, Copy, Debug)]
pub(super) struct CircleRegion {
    pub(super) center: [f32; 2],
    pub(super) radius: f32,
}

pub(super) fn exact_circle_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() > 64 {
        return None;
    }
    let mut circles = Vec::with_capacity(regions.len());
    let mut all_unclipped = true;
    for region in regions {
        let RegionGeometry::Circle {
            center_angstrom,
            radius_angstrom,
        } = &region.geometry
        else {
            return None;
        };
        if *radius_angstrom <= 0.0 {
            continue;
        }
        if center_angstrom[0] - radius_angstrom < bounds.xmin
            || center_angstrom[0] + radius_angstrom > bounds.xmax
            || center_angstrom[1] - radius_angstrom < bounds.ymin
            || center_angstrom[1] + radius_angstrom > bounds.ymax
        {
            all_unclipped = false;
        }
        circles.push(CircleRegion {
            center: *center_angstrom,
            radius: *radius_angstrom,
        });
    }
    if circles.is_empty() {
        return Some(0.0);
    }
    if all_unclipped {
        return Some(exact_unclipped_circle_union_area(&circles));
    }
    Some(exact_clipped_circle_union_area(&circles, bounds))
}

pub(super) fn circle_rectangle_intersection_area(
    circle: CircleRegion,
    bounds: LayoutBounds,
) -> f32 {
    let xmin = (bounds.xmin - circle.center[0]).max(-circle.radius);
    let xmax = (bounds.xmax - circle.center[0]).min(circle.radius);
    let ymin = bounds.ymin - circle.center[1];
    let ymax = bounds.ymax - circle.center[1];
    if xmin >= xmax || ymin >= circle.radius || ymax <= -circle.radius {
        return 0.0;
    }

    let mut xs = vec![xmin, xmax];
    add_circle_rectangle_breakpoints(&mut xs, circle.radius, ymin, xmin, xmax);
    add_circle_rectangle_breakpoints(&mut xs, circle.radius, ymax, xmin, xmax);
    xs.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    xs.dedup_by(|left, right| (*left - *right).abs() < 1.0e-6);

    let mut area = 0.0f64;
    for pair in xs.windows(2) {
        let x0 = pair[0] as f64;
        let x1 = pair[1] as f64;
        if x1 <= x0 {
            continue;
        }
        let mid = ((x0 + x1) * 0.5) as f32;
        let chord = (circle.radius * circle.radius - mid * mid).max(0.0).sqrt();
        let top = ymax.min(chord);
        let bottom = ymin.max(-chord);
        if top <= bottom {
            continue;
        }
        area += integrate_circle_clipped_height(
            circle.radius as f64,
            x0,
            x1,
            ymin as f64,
            ymax as f64,
            top,
            bottom,
            chord,
        );
    }
    area.max(0.0) as f32
}

pub(super) fn circle_polygon_intersection_area(circle: CircleRegion, polygon: &[[f32; 2]]) -> f32 {
    if circle.radius <= 0.0 || polygon.len() < 3 {
        return 0.0;
    }
    let radius = circle.radius as f64;
    let center = [circle.center[0] as f64, circle.center[1] as f64];
    let mut signed_area = 0.0f64;
    for idx in 0..polygon.len() {
        let next = (idx + 1) % polygon.len();
        let a = [
            polygon[idx][0] as f64 - center[0],
            polygon[idx][1] as f64 - center[1],
        ];
        let b = [
            polygon[next][0] as f64 - center[0],
            polygon[next][1] as f64 - center[1],
        ];
        signed_area += circle_segment_polygon_contribution(a, b, radius);
    }
    signed_area
        .abs()
        .min(std::f64::consts::PI * radius * radius)
        .min(polygon_area(polygon) as f64) as f32
}

fn circle_segment_polygon_contribution(a: [f64; 2], b: [f64; 2], radius: f64) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let aa = d[0] * d[0] + d[1] * d[1];
    if aa <= 1.0e-20 {
        return 0.0;
    }
    let bb = 2.0 * (a[0] * d[0] + a[1] * d[1]);
    let cc = a[0] * a[0] + a[1] * a[1] - radius * radius;
    let disc = bb * bb - 4.0 * aa * cc;
    let mut splits = vec![0.0, 1.0];
    if disc > 1.0e-12 {
        let root = disc.sqrt();
        for t in [(-bb - root) / (2.0 * aa), (-bb + root) / (2.0 * aa)] {
            if t > 1.0e-10 && t < 1.0 - 1.0e-10 {
                splits.push(t);
            }
        }
    }
    splits.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    splits.dedup_by(|left, right| (*left - *right).abs() < 1.0e-10);

    let radius2 = radius * radius;
    let mut area = 0.0f64;
    for pair in splits.windows(2) {
        let p = [a[0] + d[0] * pair[0], a[1] + d[1] * pair[0]];
        let q = [a[0] + d[0] * pair[1], a[1] + d[1] * pair[1]];
        let mid = [(p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5];
        let cross = p[0] * q[1] - p[1] * q[0];
        if mid[0] * mid[0] + mid[1] * mid[1] <= radius2 + 1.0e-9 {
            area += 0.5 * cross;
        } else {
            let dot = p[0] * q[0] + p[1] * q[1];
            area += 0.5 * radius2 * cross.atan2(dot);
        }
    }
    area
}

#[derive(Clone, Copy, Debug)]
enum CircleIntervalBoundary {
    Constant(f32),
    UpperArc(CircleRegion),
    LowerArc(CircleRegion),
}

#[derive(Clone, Copy, Debug)]
struct CircleYInterval {
    bottom: CircleIntervalBoundary,
    top: CircleIntervalBoundary,
}

pub(super) fn exact_clipped_circle_union_area(
    circles: &[CircleRegion],
    bounds: LayoutBounds,
) -> f32 {
    let mut xs = vec![bounds.xmin, bounds.xmax];
    for circle in circles {
        add_if_inside(
            &mut xs,
            circle.center[0] - circle.radius,
            bounds.xmin,
            bounds.xmax,
        );
        add_if_inside(
            &mut xs,
            circle.center[0] + circle.radius,
            bounds.xmin,
            bounds.xmax,
        );
        add_circle_edge_x_breakpoints(&mut xs, *circle, bounds.ymin, bounds.xmin, bounds.xmax);
        add_circle_edge_x_breakpoints(&mut xs, *circle, bounds.ymax, bounds.xmin, bounds.xmax);
    }
    for left_idx in 0..circles.len() {
        for right_idx in (left_idx + 1)..circles.len() {
            add_circle_intersection_x_breakpoints(
                &mut xs,
                circles[left_idx],
                circles[right_idx],
                bounds.xmin,
                bounds.xmax,
            );
        }
    }
    xs.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    xs.dedup_by(|left, right| (*left - *right).abs() < 1.0e-5);

    let mut area = 0.0f64;
    for pair in xs.windows(2) {
        let x0 = pair[0];
        let x1 = pair[1];
        if x1 <= x0 {
            continue;
        }
        let mid = (x0 + x1) * 0.5;
        let mut intervals = clipped_circle_y_intervals_at_x(circles, bounds, mid);
        if intervals.is_empty() {
            continue;
        }
        intervals.sort_by(|left, right| {
            circle_boundary_value(left.bottom, mid)
                .partial_cmp(&circle_boundary_value(right.bottom, mid))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut merged = Vec::<CircleYInterval>::new();
        for interval in intervals {
            if let Some(last) = merged.last_mut() {
                if circle_boundary_value(interval.bottom, mid)
                    <= circle_boundary_value(last.top, mid) + 1.0e-5
                {
                    if circle_boundary_value(interval.top, mid)
                        > circle_boundary_value(last.top, mid)
                    {
                        last.top = interval.top;
                    }
                    continue;
                }
            }
            merged.push(interval);
        }
        for interval in merged {
            area += circle_boundary_integral(interval.top, x0, x1)
                - circle_boundary_integral(interval.bottom, x0, x1);
        }
    }
    area.max(0.0) as f32
}

pub(super) fn add_if_inside(values: &mut Vec<f32>, value: f32, min: f32, max: f32) {
    if value > min + 1.0e-6 && value < max - 1.0e-6 {
        values.push(value);
    }
}

fn add_circle_edge_x_breakpoints(
    xs: &mut Vec<f32>,
    circle: CircleRegion,
    y_edge: f32,
    xmin: f32,
    xmax: f32,
) {
    let local_y = y_edge - circle.center[1];
    if local_y.abs() >= circle.radius {
        return;
    }
    let dx = (circle.radius * circle.radius - local_y * local_y).sqrt();
    for x in [circle.center[0] - dx, circle.center[0] + dx] {
        add_if_inside(xs, x, xmin, xmax);
    }
}

fn add_circle_intersection_x_breakpoints(
    xs: &mut Vec<f32>,
    left: CircleRegion,
    right: CircleRegion,
    xmin: f32,
    xmax: f32,
) {
    let dx = right.center[0] - left.center[0];
    let dy = right.center[1] - left.center[1];
    let distance_sq = dx * dx + dy * dy;
    if distance_sq <= 1.0e-12 {
        return;
    }
    let distance = distance_sq.sqrt();
    if distance >= left.radius + right.radius - 1.0e-6
        || distance <= (left.radius - right.radius).abs() + 1.0e-6
    {
        return;
    }
    let along = (left.radius.powi(2) - right.radius.powi(2) + distance_sq) / (2.0 * distance);
    let height_sq = left.radius.powi(2) - along.powi(2);
    if height_sq <= 1.0e-12 {
        return;
    }
    let height = height_sq.sqrt();
    let ux = dx / distance;
    let uy = dy / distance;
    let base_x = left.center[0] + along * ux;
    for x in [base_x - height * uy, base_x + height * uy] {
        add_if_inside(xs, x, xmin, xmax);
    }
}

fn clipped_circle_y_intervals_at_x(
    circles: &[CircleRegion],
    bounds: LayoutBounds,
    x: f32,
) -> Vec<CircleYInterval> {
    let mut intervals = Vec::new();
    for circle in circles {
        let dx = x - circle.center[0];
        if dx.abs() >= circle.radius {
            continue;
        }
        let chord = (circle.radius * circle.radius - dx * dx).sqrt();
        let lower = circle.center[1] - chord;
        let upper = circle.center[1] + chord;
        if upper <= bounds.ymin || lower >= bounds.ymax {
            continue;
        }
        let bottom = if lower < bounds.ymin {
            CircleIntervalBoundary::Constant(bounds.ymin)
        } else {
            CircleIntervalBoundary::LowerArc(*circle)
        };
        let top = if upper > bounds.ymax {
            CircleIntervalBoundary::Constant(bounds.ymax)
        } else {
            CircleIntervalBoundary::UpperArc(*circle)
        };
        if circle_boundary_value(top, x) > circle_boundary_value(bottom, x) {
            intervals.push(CircleYInterval { bottom, top });
        }
    }
    intervals
}

fn circle_boundary_value(boundary: CircleIntervalBoundary, x: f32) -> f32 {
    match boundary {
        CircleIntervalBoundary::Constant(y) => y,
        CircleIntervalBoundary::UpperArc(circle) => {
            circle.center[1]
                + (circle.radius.powi(2) - (x - circle.center[0]).powi(2))
                    .max(0.0)
                    .sqrt()
        }
        CircleIntervalBoundary::LowerArc(circle) => {
            circle.center[1]
                - (circle.radius.powi(2) - (x - circle.center[0]).powi(2))
                    .max(0.0)
                    .sqrt()
        }
    }
}

fn circle_boundary_integral(boundary: CircleIntervalBoundary, x0: f32, x1: f32) -> f64 {
    match boundary {
        CircleIntervalBoundary::Constant(y) => y as f64 * (x1 - x0) as f64,
        CircleIntervalBoundary::UpperArc(circle) => {
            circle.center[1] as f64 * (x1 - x0) as f64
                + circle_upper_arc_integral(circle.radius as f64, (x1 - circle.center[0]) as f64)
                - circle_upper_arc_integral(circle.radius as f64, (x0 - circle.center[0]) as f64)
        }
        CircleIntervalBoundary::LowerArc(circle) => {
            circle.center[1] as f64 * (x1 - x0) as f64
                - circle_upper_arc_integral(circle.radius as f64, (x1 - circle.center[0]) as f64)
                + circle_upper_arc_integral(circle.radius as f64, (x0 - circle.center[0]) as f64)
        }
    }
}

fn add_circle_rectangle_breakpoints(
    xs: &mut Vec<f32>,
    radius: f32,
    y_edge: f32,
    xmin: f32,
    xmax: f32,
) {
    if y_edge.abs() >= radius {
        return;
    }
    let dx = (radius * radius - y_edge * y_edge).sqrt();
    for x in [-dx, dx] {
        if x > xmin + 1.0e-6 && x < xmax - 1.0e-6 {
            xs.push(x);
        }
    }
}

fn integrate_circle_clipped_height(
    radius: f64,
    x0: f64,
    x1: f64,
    ymin: f64,
    ymax: f64,
    top_at_mid: f32,
    bottom_at_mid: f32,
    chord_at_mid: f32,
) -> f64 {
    let chord_integral =
        circle_upper_arc_integral(radius, x1) - circle_upper_arc_integral(radius, x0);
    let width = x1 - x0;
    let top = if (top_at_mid - chord_at_mid).abs() < 1.0e-5 {
        chord_integral
    } else {
        ymax * width
    };
    let bottom = if (bottom_at_mid + chord_at_mid).abs() < 1.0e-5 {
        -chord_integral
    } else {
        ymin * width
    };
    top - bottom
}

pub(super) fn circle_upper_arc_integral(radius: f64, x: f64) -> f64 {
    let clamped_x = x.clamp(-radius, radius);
    0.5 * (clamped_x * (radius * radius - clamped_x * clamped_x).max(0.0).sqrt()
        + radius * radius * (clamped_x / radius).asin())
}

pub(super) fn exact_unclipped_circle_union_area(circles: &[CircleRegion]) -> f32 {
    let mut area = 0.0f32;
    for (idx, circle) in circles.iter().enumerate() {
        let mut angles = vec![0.0, std::f32::consts::TAU];
        let mut covered = false;
        for (other_idx, other) in circles.iter().enumerate() {
            if idx == other_idx {
                continue;
            }
            let dx = other.center[0] - circle.center[0];
            let dy = other.center[1] - circle.center[1];
            let distance = (dx * dx + dy * dy).sqrt();
            if distance + circle.radius <= other.radius + 1.0e-5 {
                covered = true;
                break;
            }
            if distance >= circle.radius + other.radius - 1.0e-5
                || distance <= (circle.radius - other.radius).abs() + 1.0e-5
            {
                continue;
            }
            let base = dy.atan2(dx);
            let cosine = ((circle.radius.powi(2) + distance.powi(2) - other.radius.powi(2))
                / (2.0 * circle.radius * distance))
                .clamp(-1.0, 1.0);
            let delta = cosine.acos();
            angles.push((base - delta).rem_euclid(std::f32::consts::TAU));
            angles.push((base + delta).rem_euclid(std::f32::consts::TAU));
        }
        if covered {
            continue;
        }
        angles.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
        angles.dedup_by(|left, right| (*left - *right).abs() < 1.0e-6);
        for pair in angles.windows(2) {
            area += visible_circle_arc_area(*circle, pair[0], pair[1], circles, idx);
        }
        area += visible_circle_arc_area(
            *circle,
            *angles.last().unwrap(),
            angles[0] + std::f32::consts::TAU,
            circles,
            idx,
        );
    }
    area.abs()
}

fn visible_circle_arc_area(
    circle: CircleRegion,
    start: f32,
    end: f32,
    circles: &[CircleRegion],
    circle_idx: usize,
) -> f32 {
    if end <= start {
        return 0.0;
    }
    let mid = (start + end) * 0.5;
    let point = [
        circle.center[0] + circle.radius * mid.cos(),
        circle.center[1] + circle.radius * mid.sin(),
    ];
    if circles.iter().enumerate().any(|(idx, other)| {
        idx != circle_idx && squared_distance2(point, other.center) < other.radius.powi(2) - 1.0e-5
    }) {
        return 0.0;
    }
    circle_arc_area_contribution(circle, start, end)
}

fn circle_arc_area_contribution(circle: CircleRegion, start: f32, end: f32) -> f32 {
    0.5 * (circle.radius * circle.center[0] * (end.sin() - start.sin())
        - circle.radius * circle.center[1] * (end.cos() - start.cos())
        + circle.radius.powi(2) * (end - start))
}
