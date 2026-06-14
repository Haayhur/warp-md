use crate::build_layout::LayoutBounds;

use super::build_contract_region_circle::{add_if_inside, circle_upper_arc_integral, CircleRegion};
use super::build_contract_region_ellipse::{
    add_rotated_ellipse_boundary_roots, add_rotated_ellipse_horizontal_breakpoints,
    ellipse_upper_arc_integral, max_rotated_boundary_at_x, min_rotated_boundary_at_x,
    rotated_ellipse_boundary_integral, rotated_ellipse_boundary_value, RotatedEllipseBoundary,
    RotatedEllipseVerticalParams,
};

pub(super) fn axis_aligned_ellipse_pair_intersection_area_clipped(
    left_center: [f32; 2],
    left_radius: [f32; 2],
    right_center: [f32; 2],
    right_radius: [f32; 2],
    bounds: LayoutBounds,
) -> f32 {
    if left_radius[0] <= 0.0
        || left_radius[1] <= 0.0
        || right_radius[0] <= 0.0
        || right_radius[1] <= 0.0
    {
        return 0.0;
    }
    let xmin = (left_center[0] - left_radius[0])
        .max(right_center[0] - right_radius[0])
        .max(bounds.xmin);
    let xmax = (left_center[0] + left_radius[0])
        .min(right_center[0] + right_radius[0])
        .min(bounds.xmax);
    if xmin >= xmax {
        return 0.0;
    }

    let left_upper = CircleEllipseBoundary::EllipseUpper {
        center: left_center,
        radius: left_radius,
    };
    let left_lower = CircleEllipseBoundary::EllipseLower {
        center: left_center,
        radius: left_radius,
    };
    let right_upper = CircleEllipseBoundary::EllipseUpper {
        center: right_center,
        radius: right_radius,
    };
    let right_lower = CircleEllipseBoundary::EllipseLower {
        center: right_center,
        radius: right_radius,
    };
    let clip_top = CircleEllipseBoundary::Constant(bounds.ymax);
    let clip_bottom = CircleEllipseBoundary::Constant(bounds.ymin);
    let mut xs = vec![xmin, xmax];
    for (left, right) in [
        (left_upper, right_upper),
        (left_upper, right_lower),
        (left_lower, right_upper),
        (left_lower, right_lower),
        (left_upper, clip_top),
        (right_upper, clip_top),
        (left_lower, clip_bottom),
        (right_lower, clip_bottom),
    ] {
        add_circle_ellipse_boundary_roots(&mut xs, left, right, xmin, xmax);
    }
    xs.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    xs.dedup_by(|left, right| (*left - *right).abs() < 1.0e-5);

    let mut area = 0.0f64;
    for pair in xs.windows(2) {
        let x0 = pair[0];
        let x1 = pair[1];
        if x1 <= x0 + 1.0e-6 {
            continue;
        }
        let mid = (x0 + x1) * 0.5;
        let top = min_boundary_at_x([left_upper, right_upper, clip_top], mid);
        let bottom = max_boundary_at_x([left_lower, right_lower, clip_bottom], mid);
        if circle_ellipse_boundary_value(top, mid) > circle_ellipse_boundary_value(bottom, mid) {
            area += circle_ellipse_boundary_integral(top, x0, x1)
                - circle_ellipse_boundary_integral(bottom, x0, x1);
        }
    }
    area.max(0.0) as f32
}

pub(super) fn rotated_ellipse_pair_intersection_area_clipped(
    left_center: [f32; 2],
    left_radius: [f32; 2],
    left_rotate_degrees: f32,
    right_center: [f32; 2],
    right_radius: [f32; 2],
    right_rotate_degrees: f32,
    bounds: LayoutBounds,
) -> f32 {
    let left = RotatedEllipseVerticalParams::new(left_center, left_radius, left_rotate_degrees);
    let right = RotatedEllipseVerticalParams::new(right_center, right_radius, right_rotate_degrees);
    if left.projected_half_x <= 0.0
        || left.projected_half_height <= 0.0
        || right.projected_half_x <= 0.0
        || right.projected_half_height <= 0.0
    {
        return 0.0;
    }
    let xmin = bounds
        .xmin
        .max(left.center[0] - left.projected_half_x)
        .max(right.center[0] - right.projected_half_x);
    let xmax = bounds
        .xmax
        .min(left.center[0] + left.projected_half_x)
        .min(right.center[0] + right.projected_half_x);
    if xmin >= xmax {
        return 0.0;
    }

    let left_upper = RotatedEllipseBoundary::Upper(left);
    let left_lower = RotatedEllipseBoundary::Lower(left);
    let right_upper = RotatedEllipseBoundary::Upper(right);
    let right_lower = RotatedEllipseBoundary::Lower(right);
    let clip_top = RotatedEllipseBoundary::Constant(bounds.ymax);
    let clip_bottom = RotatedEllipseBoundary::Constant(bounds.ymin);
    let boundaries = [
        left_upper,
        left_lower,
        right_upper,
        right_lower,
        clip_top,
        clip_bottom,
    ];
    let mut xs = vec![xmin, xmax];
    add_rotated_ellipse_horizontal_breakpoints(&mut xs, left, bounds.ymin, xmin, xmax);
    add_rotated_ellipse_horizontal_breakpoints(&mut xs, left, bounds.ymax, xmin, xmax);
    add_rotated_ellipse_horizontal_breakpoints(&mut xs, right, bounds.ymin, xmin, xmax);
    add_rotated_ellipse_horizontal_breakpoints(&mut xs, right, bounds.ymax, xmin, xmax);
    for left_idx in 0..boundaries.len() {
        for right_idx in (left_idx + 1)..boundaries.len() {
            add_rotated_ellipse_boundary_roots(
                &mut xs,
                boundaries[left_idx],
                boundaries[right_idx],
                xmin,
                xmax,
            );
        }
    }
    xs.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    xs.dedup_by(|left, right| (*left - *right).abs() < 1.0e-5);

    let mut area = 0.0f64;
    for pair in xs.windows(2) {
        let x0 = pair[0];
        let x1 = pair[1];
        if x1 <= x0 + 1.0e-6 {
            continue;
        }
        let mid = (x0 + x1) * 0.5;
        let top = min_rotated_boundary_at_x([left_upper, right_upper, clip_top], mid);
        let bottom = max_rotated_boundary_at_x([left_lower, right_lower, clip_bottom], mid);
        if rotated_ellipse_boundary_value(top, mid) > rotated_ellipse_boundary_value(bottom, mid) {
            area += rotated_ellipse_boundary_integral(top, x0, x1)
                - rotated_ellipse_boundary_integral(bottom, x0, x1);
        }
    }
    area.max(0.0) as f32
}

pub(super) fn circle_axis_aligned_ellipse_intersection_area(
    circle: CircleRegion,
    ellipse_center: [f32; 2],
    ellipse_radius: [f32; 2],
) -> f32 {
    let bounds = LayoutBounds {
        xmin: (circle.center[0] - circle.radius).max(ellipse_center[0] - ellipse_radius[0]),
        xmax: (circle.center[0] + circle.radius).min(ellipse_center[0] + ellipse_radius[0]),
        ymin: (circle.center[1] - circle.radius).min(ellipse_center[1] - ellipse_radius[1]),
        ymax: (circle.center[1] + circle.radius).max(ellipse_center[1] + ellipse_radius[1]),
    };
    circle_axis_aligned_ellipse_intersection_area_clipped(
        circle,
        ellipse_center,
        ellipse_radius,
        bounds,
    )
}

pub(super) fn circle_axis_aligned_ellipse_intersection_area_clipped(
    circle: CircleRegion,
    ellipse_center: [f32; 2],
    ellipse_radius: [f32; 2],
    bounds: LayoutBounds,
) -> f32 {
    let a = ellipse_radius[0];
    let b = ellipse_radius[1];
    let r = circle.radius;
    if r <= 0.0 || a <= 0.0 || b <= 0.0 {
        return 0.0;
    }
    let xmin = (circle.center[0] - r)
        .max(ellipse_center[0] - a)
        .max(bounds.xmin);
    let xmax = (circle.center[0] + r)
        .min(ellipse_center[0] + a)
        .min(bounds.xmax);
    if xmin >= xmax {
        return 0.0;
    }

    let circle_upper = CircleEllipseBoundary::CircleUpper(circle);
    let circle_lower = CircleEllipseBoundary::CircleLower(circle);
    let ellipse_upper = CircleEllipseBoundary::EllipseUpper {
        center: ellipse_center,
        radius: ellipse_radius,
    };
    let ellipse_lower = CircleEllipseBoundary::EllipseLower {
        center: ellipse_center,
        radius: ellipse_radius,
    };
    let clip_top = CircleEllipseBoundary::Constant(bounds.ymax);
    let clip_bottom = CircleEllipseBoundary::Constant(bounds.ymin);
    let mut xs = vec![xmin, xmax];
    for (left, right) in [
        (circle_upper, ellipse_upper),
        (circle_upper, ellipse_lower),
        (circle_lower, ellipse_upper),
        (circle_lower, ellipse_lower),
        (circle_upper, clip_top),
        (ellipse_upper, clip_top),
        (circle_lower, clip_bottom),
        (ellipse_lower, clip_bottom),
    ] {
        add_circle_ellipse_boundary_roots(&mut xs, left, right, xmin, xmax);
    }
    xs.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    xs.dedup_by(|left, right| (*left - *right).abs() < 1.0e-5);

    let mut area = 0.0f64;
    for pair in xs.windows(2) {
        let x0 = pair[0];
        let x1 = pair[1];
        if x1 <= x0 + 1.0e-6 {
            continue;
        }
        let mid = (x0 + x1) * 0.5;
        let top = min_boundary_at_x([circle_upper, ellipse_upper, clip_top], mid);
        let bottom = max_boundary_at_x([circle_lower, ellipse_lower, clip_bottom], mid);
        if circle_ellipse_boundary_value(top, mid) > circle_ellipse_boundary_value(bottom, mid) {
            area += circle_ellipse_boundary_integral(top, x0, x1)
                - circle_ellipse_boundary_integral(bottom, x0, x1);
        }
    }
    area.max(0.0) as f32
}

fn min_boundary_at_x<const N: usize>(
    boundaries: [CircleEllipseBoundary; N],
    x: f32,
) -> CircleEllipseBoundary {
    boundaries
        .into_iter()
        .min_by(|left, right| {
            circle_ellipse_boundary_value(*left, x)
                .partial_cmp(&circle_ellipse_boundary_value(*right, x))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
}

fn max_boundary_at_x<const N: usize>(
    boundaries: [CircleEllipseBoundary; N],
    x: f32,
) -> CircleEllipseBoundary {
    boundaries
        .into_iter()
        .max_by(|left, right| {
            circle_ellipse_boundary_value(*left, x)
                .partial_cmp(&circle_ellipse_boundary_value(*right, x))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
}

#[derive(Clone, Copy, Debug)]
enum CircleEllipseBoundary {
    Constant(f32),
    CircleUpper(CircleRegion),
    CircleLower(CircleRegion),
    EllipseUpper { center: [f32; 2], radius: [f32; 2] },
    EllipseLower { center: [f32; 2], radius: [f32; 2] },
}

fn add_circle_ellipse_boundary_roots(
    xs: &mut Vec<f32>,
    left: CircleEllipseBoundary,
    right: CircleEllipseBoundary,
    xmin: f32,
    xmax: f32,
) {
    const ROOT_SCAN_STEPS: usize = 256;
    let mut prev_x = xmin;
    let mut prev_y =
        circle_ellipse_boundary_value(left, prev_x) - circle_ellipse_boundary_value(right, prev_x);
    for step in 1..=ROOT_SCAN_STEPS {
        let x = xmin + (xmax - xmin) * step as f32 / ROOT_SCAN_STEPS as f32;
        let y = circle_ellipse_boundary_value(left, x) - circle_ellipse_boundary_value(right, x);
        if y.abs() <= 1.0e-6 {
            add_if_inside(xs, x, xmin, xmax);
        } else if prev_y.abs() <= 1.0e-6 {
            add_if_inside(xs, prev_x, xmin, xmax);
        } else if y.signum() != prev_y.signum() {
            let root = bisect_circle_ellipse_boundary_root(left, right, prev_x, x);
            add_if_inside(xs, root, xmin, xmax);
        }
        prev_x = x;
        prev_y = y;
    }
}

fn bisect_circle_ellipse_boundary_root(
    left: CircleEllipseBoundary,
    right: CircleEllipseBoundary,
    mut lo: f32,
    mut hi: f32,
) -> f32 {
    let mut f_lo =
        circle_ellipse_boundary_value(left, lo) - circle_ellipse_boundary_value(right, lo);
    for _ in 0..48 {
        let mid = (lo + hi) * 0.5;
        let f_mid =
            circle_ellipse_boundary_value(left, mid) - circle_ellipse_boundary_value(right, mid);
        if f_mid.abs() <= 1.0e-7 || (hi - lo).abs() <= 1.0e-6 {
            return mid;
        }
        if f_mid.signum() == f_lo.signum() {
            lo = mid;
            f_lo = f_mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) * 0.5
}

fn circle_ellipse_boundary_value(boundary: CircleEllipseBoundary, x: f32) -> f32 {
    match boundary {
        CircleEllipseBoundary::Constant(y) => y,
        CircleEllipseBoundary::CircleUpper(circle) => {
            circle.center[1]
                + (circle.radius.powi(2) - (x - circle.center[0]).powi(2))
                    .max(0.0)
                    .sqrt()
        }
        CircleEllipseBoundary::CircleLower(circle) => {
            circle.center[1]
                - (circle.radius.powi(2) - (x - circle.center[0]).powi(2))
                    .max(0.0)
                    .sqrt()
        }
        CircleEllipseBoundary::EllipseUpper { center, radius } => {
            center[1]
                + radius[1]
                    * (1.0 - ((x - center[0]) / radius[0]).powi(2))
                        .max(0.0)
                        .sqrt()
        }
        CircleEllipseBoundary::EllipseLower { center, radius } => {
            center[1]
                - radius[1]
                    * (1.0 - ((x - center[0]) / radius[0]).powi(2))
                        .max(0.0)
                        .sqrt()
        }
    }
}

fn circle_ellipse_boundary_integral(boundary: CircleEllipseBoundary, x0: f32, x1: f32) -> f64 {
    match boundary {
        CircleEllipseBoundary::Constant(y) => y as f64 * (x1 - x0) as f64,
        CircleEllipseBoundary::CircleUpper(circle) => {
            circle.center[1] as f64 * (x1 - x0) as f64
                + circle_upper_arc_integral(circle.radius as f64, (x1 - circle.center[0]) as f64)
                - circle_upper_arc_integral(circle.radius as f64, (x0 - circle.center[0]) as f64)
        }
        CircleEllipseBoundary::CircleLower(circle) => {
            circle.center[1] as f64 * (x1 - x0) as f64
                - circle_upper_arc_integral(circle.radius as f64, (x1 - circle.center[0]) as f64)
                + circle_upper_arc_integral(circle.radius as f64, (x0 - circle.center[0]) as f64)
        }
        CircleEllipseBoundary::EllipseUpper { center, radius } => {
            center[1] as f64 * (x1 - x0) as f64
                + ellipse_upper_arc_integral(
                    radius[0] as f64,
                    radius[1] as f64,
                    (x1 - center[0]) as f64,
                )
                - ellipse_upper_arc_integral(
                    radius[0] as f64,
                    radius[1] as f64,
                    (x0 - center[0]) as f64,
                )
        }
        CircleEllipseBoundary::EllipseLower { center, radius } => {
            center[1] as f64 * (x1 - x0) as f64
                - ellipse_upper_arc_integral(
                    radius[0] as f64,
                    radius[1] as f64,
                    (x1 - center[0]) as f64,
                )
                + ellipse_upper_arc_integral(
                    radius[0] as f64,
                    radius[1] as f64,
                    (x0 - center[0]) as f64,
                )
        }
    }
}
