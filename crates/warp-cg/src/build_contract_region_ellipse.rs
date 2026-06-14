use crate::build_layout::LayoutBounds;

use super::build_contract_region_circle::{
    add_if_inside, exact_unclipped_circle_union_area, CircleRegion,
};
use super::{
    axis_aligned_bounds_overlap, clipped_bounds, inverse_rotated_xy, region_bounds, LeafletRegion,
    RegionGeometry,
};

pub(super) fn exact_disjoint_ellipse_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    let mut ellipses = Vec::with_capacity(regions.len());
    let mut all_unclipped = true;
    for region in regions {
        let RegionGeometry::Ellipse {
            center_angstrom,
            radius_angstrom,
            rotate_degrees,
        } = &region.geometry
        else {
            return None;
        };
        let ellipse_bounds = region_bounds(region)?;
        if ellipse_bounds.0 < bounds.xmin
            || ellipse_bounds.1 > bounds.xmax
            || ellipse_bounds.2 < bounds.ymin
            || ellipse_bounds.3 > bounds.ymax
        {
            all_unclipped = false;
        }
        ellipses.push((
            *center_angstrom,
            *radius_angstrom,
            *rotate_degrees,
            ellipse_bounds,
        ));
    }
    for left_idx in 0..ellipses.len() {
        for right_idx in (left_idx + 1)..ellipses.len() {
            if axis_aligned_bounds_overlap(ellipses[left_idx].3, ellipses[right_idx].3) {
                return None;
            }
        }
    }
    if all_unclipped {
        return Some(
            ellipses
                .iter()
                .map(|(_, radius, _, _)| std::f32::consts::PI * radius[0] * radius[1])
                .sum(),
        );
    }
    Some(
        ellipses
            .iter()
            .map(|(center, radius, rotate_degrees, _)| {
                ellipse_rectangle_intersection_area(*center, *radius, *rotate_degrees, bounds)
            })
            .sum(),
    )
}

pub(super) fn exact_similar_oriented_ellipse_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() > 64 {
        return None;
    }
    let mut scaled_circles = Vec::with_capacity(regions.len());
    let mut common_aspect: Option<f32> = None;
    let mut common_angle = None;
    for region in regions {
        let RegionGeometry::Ellipse {
            center_angstrom,
            radius_angstrom,
            rotate_degrees,
        } = &region.geometry
        else {
            return None;
        };
        if radius_angstrom[0] <= 0.0 || radius_angstrom[1] <= 0.0 {
            continue;
        }
        let aspect = radius_angstrom[0] / radius_angstrom[1];
        let ellipse_bounds = region_bounds(region)?;
        if clipped_bounds(ellipse_bounds, bounds)? != ellipse_bounds {
            return None;
        }
        let angle = rotate_degrees.rem_euclid(180.0);
        if let Some(existing) = common_angle {
            if angular_difference_degrees(angle, existing) > 1.0e-5 {
                return None;
            }
        } else {
            common_angle = Some(angle);
        }
        if let Some(existing) = common_aspect {
            if (aspect - existing).abs() > 1.0e-6 {
                return None;
            }
        } else {
            common_aspect = Some(aspect);
        }
        let local_center = inverse_rotated_xy(*center_angstrom, [0.0, 0.0], angle);
        scaled_circles.push(CircleRegion {
            center: [local_center[0] / aspect, local_center[1]],
            radius: radius_angstrom[1],
        });
    }
    if scaled_circles.is_empty() {
        return Some(0.0);
    }
    Some(exact_unclipped_circle_union_area(&scaled_circles) * common_aspect?)
}

fn angular_difference_degrees(left: f32, right: f32) -> f32 {
    let delta = (left - right).rem_euclid(180.0).abs();
    delta.min(180.0 - delta)
}

pub(super) fn axis_aligned_ellipse_rectangle_intersection_area(
    center: [f32; 2],
    radius: [f32; 2],
    bounds: LayoutBounds,
) -> f32 {
    ellipse_rectangle_intersection_area(center, radius, 0.0, bounds)
}

pub(super) fn ellipse_rectangle_intersection_area(
    center: [f32; 2],
    radius: [f32; 2],
    rotate_degrees: f32,
    bounds: LayoutBounds,
) -> f32 {
    let a = radius[0];
    let b = radius[1];
    if a <= 0.0 || b <= 0.0 {
        return 0.0;
    }
    let angle = rotate_degrees.rem_euclid(180.0);
    if angle <= 1.0e-6 {
        return axis_aligned_ellipse_rectangle_intersection_area_inner(center, radius, bounds);
    }
    rotated_ellipse_rectangle_intersection_area(center, radius, rotate_degrees, bounds)
}

fn axis_aligned_ellipse_rectangle_intersection_area_inner(
    center: [f32; 2],
    radius: [f32; 2],
    bounds: LayoutBounds,
) -> f32 {
    let a = radius[0];
    let b = radius[1];
    let xmin = (bounds.xmin - center[0]).max(-a);
    let xmax = (bounds.xmax - center[0]).min(a);
    let ymin = bounds.ymin - center[1];
    let ymax = bounds.ymax - center[1];
    if xmin >= xmax || ymin >= b || ymax <= -b {
        return 0.0;
    }

    let mut xs = vec![xmin, xmax];
    add_ellipse_rectangle_breakpoints(&mut xs, a, b, ymin, xmin, xmax);
    add_ellipse_rectangle_breakpoints(&mut xs, a, b, ymax, xmin, xmax);
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
        let chord = b * (1.0 - (mid / a).powi(2)).max(0.0).sqrt();
        let top = ymax.min(chord);
        let bottom = ymin.max(-chord);
        if top <= bottom {
            continue;
        }
        area += integrate_ellipse_clipped_height(
            a as f64,
            b as f64,
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

fn rotated_ellipse_rectangle_intersection_area(
    center: [f32; 2],
    radius: [f32; 2],
    rotate_degrees: f32,
    bounds: LayoutBounds,
) -> f32 {
    let params = RotatedEllipseVerticalParams::new(center, radius, rotate_degrees);
    if params.projected_half_x <= 0.0 || params.projected_half_height <= 0.0 {
        return 0.0;
    }
    let xmin = bounds.xmin.max(center[0] - params.projected_half_x);
    let xmax = bounds.xmax.min(center[0] + params.projected_half_x);
    if xmin >= xmax
        || bounds.ymin >= center[1] + params.projected_half_y
        || bounds.ymax <= center[1] - params.projected_half_y
    {
        return 0.0;
    }

    let upper = RotatedEllipseBoundary::Upper(params);
    let lower = RotatedEllipseBoundary::Lower(params);
    let clip_top = RotatedEllipseBoundary::Constant(bounds.ymax);
    let clip_bottom = RotatedEllipseBoundary::Constant(bounds.ymin);
    let mut xs = vec![xmin, xmax];
    add_rotated_ellipse_horizontal_breakpoints(&mut xs, params, bounds.ymin, xmin, xmax);
    add_rotated_ellipse_horizontal_breakpoints(&mut xs, params, bounds.ymax, xmin, xmax);
    xs.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    xs.dedup_by(|left, right| (*left - *right).abs() < 1.0e-6);

    let mut area = 0.0f64;
    for pair in xs.windows(2) {
        let x0 = pair[0];
        let x1 = pair[1];
        if x1 <= x0 + 1.0e-6 {
            continue;
        }
        let mid = (x0 + x1) * 0.5;
        let top = if rotated_ellipse_boundary_value(upper, mid)
            <= rotated_ellipse_boundary_value(clip_top, mid)
        {
            upper
        } else {
            clip_top
        };
        let bottom = if rotated_ellipse_boundary_value(lower, mid)
            >= rotated_ellipse_boundary_value(clip_bottom, mid)
        {
            lower
        } else {
            clip_bottom
        };
        if rotated_ellipse_boundary_value(top, mid) > rotated_ellipse_boundary_value(bottom, mid) {
            area += rotated_ellipse_boundary_integral(top, x0, x1)
                - rotated_ellipse_boundary_integral(bottom, x0, x1);
        }
    }
    area.max(0.0) as f32
}

#[derive(Clone, Copy, Debug)]
pub(super) struct RotatedEllipseVerticalParams {
    pub(super) center: [f32; 2],
    pub(super) projected_half_x: f32,
    projected_half_y: f32,
    pub(super) projected_half_height: f32,
    centerline_slope: f32,
}

impl RotatedEllipseVerticalParams {
    pub(super) fn new(center: [f32; 2], radius: [f32; 2], rotate_degrees: f32) -> Self {
        let theta = rotate_degrees.to_radians();
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let a = radius[0];
        let b = radius[1];
        let projected_half_x = ((a * cos_t).powi(2) + (b * sin_t).powi(2)).sqrt();
        let projected_half_y = ((a * sin_t).powi(2) + (b * cos_t).powi(2)).sqrt();
        let projected_half_height = a * b / projected_half_x;
        let inv_a2 = 1.0 / a.powi(2);
        let inv_b2 = 1.0 / b.powi(2);
        let quad_y = sin_t.powi(2) * inv_a2 + cos_t.powi(2) * inv_b2;
        let centerline_slope = cos_t * sin_t * (inv_b2 - inv_a2) / quad_y;
        Self {
            center,
            projected_half_x,
            projected_half_y,
            projected_half_height,
            centerline_slope,
        }
    }

    fn centerline_at_x(self, x: f32) -> f32 {
        self.center[1] + self.centerline_slope * (x - self.center[0])
    }

    fn chord_at_x(self, x: f32) -> f32 {
        self.projected_half_height
            * (1.0 - ((x - self.center[0]) / self.projected_half_x).powi(2))
                .max(0.0)
                .sqrt()
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum RotatedEllipseBoundary {
    Constant(f32),
    Upper(RotatedEllipseVerticalParams),
    Lower(RotatedEllipseVerticalParams),
}

pub(super) fn rotated_ellipse_boundary_value(boundary: RotatedEllipseBoundary, x: f32) -> f32 {
    match boundary {
        RotatedEllipseBoundary::Constant(y) => y,
        RotatedEllipseBoundary::Upper(params) => params.centerline_at_x(x) + params.chord_at_x(x),
        RotatedEllipseBoundary::Lower(params) => params.centerline_at_x(x) - params.chord_at_x(x),
    }
}

pub(super) fn rotated_ellipse_boundary_integral(
    boundary: RotatedEllipseBoundary,
    x0: f32,
    x1: f32,
) -> f64 {
    match boundary {
        RotatedEllipseBoundary::Constant(y) => y as f64 * (x1 - x0) as f64,
        RotatedEllipseBoundary::Upper(params) => {
            rotated_ellipse_centerline_integral(params, x0, x1)
                + ellipse_upper_arc_integral(
                    params.projected_half_x as f64,
                    params.projected_half_height as f64,
                    (x1 - params.center[0]) as f64,
                )
                - ellipse_upper_arc_integral(
                    params.projected_half_x as f64,
                    params.projected_half_height as f64,
                    (x0 - params.center[0]) as f64,
                )
        }
        RotatedEllipseBoundary::Lower(params) => {
            rotated_ellipse_centerline_integral(params, x0, x1)
                - ellipse_upper_arc_integral(
                    params.projected_half_x as f64,
                    params.projected_half_height as f64,
                    (x1 - params.center[0]) as f64,
                )
                + ellipse_upper_arc_integral(
                    params.projected_half_x as f64,
                    params.projected_half_height as f64,
                    (x0 - params.center[0]) as f64,
                )
        }
    }
}

pub(super) fn min_rotated_boundary_at_x<const N: usize>(
    boundaries: [RotatedEllipseBoundary; N],
    x: f32,
) -> RotatedEllipseBoundary {
    boundaries
        .into_iter()
        .min_by(|left, right| {
            rotated_ellipse_boundary_value(*left, x)
                .partial_cmp(&rotated_ellipse_boundary_value(*right, x))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
}

pub(super) fn max_rotated_boundary_at_x<const N: usize>(
    boundaries: [RotatedEllipseBoundary; N],
    x: f32,
) -> RotatedEllipseBoundary {
    boundaries
        .into_iter()
        .max_by(|left, right| {
            rotated_ellipse_boundary_value(*left, x)
                .partial_cmp(&rotated_ellipse_boundary_value(*right, x))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap()
}

pub(super) fn add_rotated_ellipse_boundary_roots(
    xs: &mut Vec<f32>,
    left: RotatedEllipseBoundary,
    right: RotatedEllipseBoundary,
    xmin: f32,
    xmax: f32,
) {
    const ROOT_SCAN_STEPS: usize = 384;
    let mut prev_x = xmin;
    let mut prev_y = rotated_ellipse_boundary_value(left, prev_x)
        - rotated_ellipse_boundary_value(right, prev_x);
    for step in 1..=ROOT_SCAN_STEPS {
        let x = xmin + (xmax - xmin) * step as f32 / ROOT_SCAN_STEPS as f32;
        let y = rotated_ellipse_boundary_value(left, x) - rotated_ellipse_boundary_value(right, x);
        if y.abs() <= 1.0e-6 {
            add_if_inside(xs, x, xmin, xmax);
        } else if prev_y.abs() <= 1.0e-6 {
            add_if_inside(xs, prev_x, xmin, xmax);
        } else if y.signum() != prev_y.signum() {
            let root = bisect_rotated_ellipse_boundary_root(left, right, prev_x, x);
            add_if_inside(xs, root, xmin, xmax);
        }
        prev_x = x;
        prev_y = y;
    }
}

fn bisect_rotated_ellipse_boundary_root(
    left: RotatedEllipseBoundary,
    right: RotatedEllipseBoundary,
    mut lo: f32,
    mut hi: f32,
) -> f32 {
    let mut f_lo =
        rotated_ellipse_boundary_value(left, lo) - rotated_ellipse_boundary_value(right, lo);
    for _ in 0..48 {
        let mid = (lo + hi) * 0.5;
        let f_mid =
            rotated_ellipse_boundary_value(left, mid) - rotated_ellipse_boundary_value(right, mid);
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

fn rotated_ellipse_centerline_integral(
    params: RotatedEllipseVerticalParams,
    x0: f32,
    x1: f32,
) -> f64 {
    let cx = params.center[0] as f64;
    params.center[1] as f64 * (x1 - x0) as f64
        + params.centerline_slope as f64
            * (0.5 * ((x1 as f64 - cx).powi(2) - (x0 as f64 - cx).powi(2)))
}

pub(super) fn add_rotated_ellipse_horizontal_breakpoints(
    xs: &mut Vec<f32>,
    params: RotatedEllipseVerticalParams,
    y: f32,
    xmin: f32,
    xmax: f32,
) {
    let dy = y - params.center[1];
    let m = params.centerline_slope;
    let h = params.projected_half_height;
    let hx = params.projected_half_x;
    let a = m * m + h * h / hx.powi(2);
    let b = -2.0 * dy * m;
    let c = dy * dy - h * h;
    let discriminant = b * b - 4.0 * a * c;
    if discriminant < -1.0e-6 {
        return;
    }
    let sqrt_disc = discriminant.max(0.0).sqrt();
    for root in [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)] {
        let x = params.center[0] + root;
        if x > xmin + 1.0e-6 && x < xmax - 1.0e-6 {
            xs.push(x);
        }
    }
}

fn add_ellipse_rectangle_breakpoints(
    xs: &mut Vec<f32>,
    radius_x: f32,
    radius_y: f32,
    y_edge: f32,
    xmin: f32,
    xmax: f32,
) {
    if y_edge.abs() >= radius_y {
        return;
    }
    let dx = radius_x * (1.0 - (y_edge / radius_y).powi(2)).sqrt();
    for x in [-dx, dx] {
        if x > xmin + 1.0e-6 && x < xmax - 1.0e-6 {
            xs.push(x);
        }
    }
}

fn integrate_ellipse_clipped_height(
    radius_x: f64,
    radius_y: f64,
    x0: f64,
    x1: f64,
    ymin: f64,
    ymax: f64,
    top_at_mid: f32,
    bottom_at_mid: f32,
    chord_at_mid: f32,
) -> f64 {
    let chord_integral = ellipse_upper_arc_integral(radius_x, radius_y, x1)
        - ellipse_upper_arc_integral(radius_x, radius_y, x0);
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

pub(super) fn ellipse_upper_arc_integral(radius_x: f64, radius_y: f64, x: f64) -> f64 {
    let clamped_x = x.clamp(-radius_x, radius_x);
    let u = clamped_x / radius_x;
    0.5 * radius_x * radius_y * (u * (1.0 - u * u).max(0.0).sqrt() + u.asin())
}
