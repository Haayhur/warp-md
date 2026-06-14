use crate::build_layout::LayoutBounds;

use super::build_contract_region_circle::{circle_rectangle_intersection_area, CircleRegion};
use super::build_contract_region_ellipse::ellipse_rectangle_intersection_area;
use super::build_contract_region_polygon::{convex_polygon_for_region, polygon_within_bounds};
use super::{
    polygon_area, signed_polygon_area, transformed_polygon_points, LeafletRegion, RegionGeometry,
};

pub(super) fn clipped_bounds(
    region_bounds: (f32, f32, f32, f32),
    bounds: LayoutBounds,
) -> Option<(f32, f32, f32, f32)> {
    let xmin = region_bounds.0.max(bounds.xmin);
    let xmax = region_bounds.1.min(bounds.xmax);
    let ymin = region_bounds.2.max(bounds.ymin);
    let ymax = region_bounds.3.min(bounds.ymax);
    (xmin < xmax && ymin < ymax).then_some((xmin, xmax, ymin, ymax))
}

pub(super) fn exact_single_region_area(
    region: &LeafletRegion,
    bounds: LayoutBounds,
) -> Option<f32> {
    match &region.geometry {
        RegionGeometry::Circle {
            center_angstrom,
            radius_angstrom,
        } => {
            if *radius_angstrom <= 0.0 {
                Some(0.0)
            } else {
                Some(circle_rectangle_intersection_area(
                    CircleRegion {
                        center: *center_angstrom,
                        radius: *radius_angstrom,
                    },
                    bounds,
                ))
            }
        }
        RegionGeometry::Ellipse {
            center_angstrom,
            radius_angstrom,
            rotate_degrees,
        } => {
            let region_bounds = region_bounds(region)?;
            if clipped_bounds(region_bounds, bounds)? == region_bounds {
                return Some(std::f32::consts::PI * radius_angstrom[0] * radius_angstrom[1]);
            }
            Some(ellipse_rectangle_intersection_area(
                *center_angstrom,
                *radius_angstrom,
                *rotate_degrees,
                bounds,
            ))
        }
        RegionGeometry::Rectangle { .. } => {
            let polygon = convex_polygon_for_region(region)?;
            let clipped = convex_polygon_intersection(&polygon, &layout_bounds_polygon(bounds));
            Some(polygon_area(&clipped))
        }
        RegionGeometry::Polygon { .. } => {
            if let Some(polygon) = convex_polygon_for_region(region) {
                let clipped = convex_polygon_intersection(&polygon, &layout_bounds_polygon(bounds));
                return Some(polygon_area(&clipped));
            }
            let mut points = transformed_polygon_points(region);
            if points.len() < 3 || !polygon_within_bounds(&points, bounds) {
                return None;
            }
            ensure_ccw_polygon(&mut points);
            Some(polygon_area(&points))
        }
    }
}

pub(super) fn layout_bounds_polygon(bounds: LayoutBounds) -> Vec<[f32; 2]> {
    vec![
        [bounds.xmin, bounds.ymin],
        [bounds.xmax, bounds.ymin],
        [bounds.xmax, bounds.ymax],
        [bounds.xmin, bounds.ymax],
    ]
}

pub(super) fn convex_polygon_intersection(
    subject: &[[f32; 2]],
    clip: &[[f32; 2]],
) -> Vec<[f32; 2]> {
    if subject.len() < 3 || clip.len() < 3 {
        return Vec::new();
    }
    let mut output = subject.to_vec();
    for idx in 0..clip.len() {
        let a = clip[idx];
        let b = clip[(idx + 1) % clip.len()];
        let input = output;
        output = Vec::new();
        if input.is_empty() {
            break;
        }
        let mut previous = *input.last().unwrap();
        for current in input {
            let current_inside = point_left_of_edge(current, a, b);
            let previous_inside = point_left_of_edge(previous, a, b);
            if current_inside {
                if !previous_inside {
                    output.push(line_segment_intersection(previous, current, a, b));
                }
                output.push(current);
            } else if previous_inside {
                output.push(line_segment_intersection(previous, current, a, b));
            }
            previous = current;
        }
    }
    output
}

pub(super) fn point_left_of_edge(
    point: [f32; 2],
    edge_start: [f32; 2],
    edge_end: [f32; 2],
) -> bool {
    ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1])
        - (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0]))
        >= -1.0e-5
}

pub(super) fn line_segment_intersection(
    p0: [f32; 2],
    p1: [f32; 2],
    q0: [f32; 2],
    q1: [f32; 2],
) -> [f32; 2] {
    let r = [p1[0] - p0[0], p1[1] - p0[1]];
    let s = [q1[0] - q0[0], q1[1] - q0[1]];
    let denom = r[0] * s[1] - r[1] * s[0];
    if denom.abs() <= 1.0e-8 {
        return p1;
    }
    let qp = [q0[0] - p0[0], q0[1] - p0[1]];
    let t = (qp[0] * s[1] - qp[1] * s[0]) / denom;
    [p0[0] + t * r[0], p0[1] + t * r[1]]
}

pub(super) fn polygon_is_convex(points: &[[f32; 2]]) -> bool {
    if points.len() < 3 {
        return false;
    }
    let mut sign = 0.0f32;
    for idx in 0..points.len() {
        let a = points[idx];
        let b = points[(idx + 1) % points.len()];
        let c = points[(idx + 2) % points.len()];
        let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if cross.abs() <= 1.0e-5 {
            continue;
        }
        if sign == 0.0 {
            sign = cross.signum();
        } else if sign * cross < 0.0 {
            return false;
        }
    }
    true
}

pub(super) fn ensure_ccw_polygon(points: &mut [[f32; 2]]) {
    if signed_polygon_area(points) < 0.0 {
        points.reverse();
    }
}

pub(super) fn axis_aligned_bounds_overlap(
    left: (f32, f32, f32, f32),
    right: (f32, f32, f32, f32),
) -> bool {
    left.0 < right.1 && left.1 > right.0 && left.2 < right.3 && left.3 > right.2
}

pub(super) fn clipped_axis_aligned_rectangle_bounds(
    region: &LeafletRegion,
    bounds: LayoutBounds,
) -> Option<(f32, f32, f32, f32)> {
    let (xmin, xmax, ymin, ymax) = axis_aligned_rectangle_bounds(region)?;
    let xmin = xmin.max(bounds.xmin);
    let xmax = xmax.min(bounds.xmax);
    let ymin = ymin.max(bounds.ymin);
    let ymax = ymax.min(bounds.ymax);
    (xmin < xmax && ymin < ymax).then_some((xmin, xmax, ymin, ymax))
}

pub(super) fn axis_aligned_rectangle_bounds(
    region: &LeafletRegion,
) -> Option<(f32, f32, f32, f32)> {
    let RegionGeometry::Rectangle {
        center_angstrom,
        size_angstrom,
        rotate_degrees,
    } = &region.geometry
    else {
        return None;
    };
    if rotate_degrees.rem_euclid(180.0).abs() > 1.0e-6 {
        return None;
    }
    Some((
        center_angstrom[0] - size_angstrom[0] * 0.5,
        center_angstrom[0] + size_angstrom[0] * 0.5,
        center_angstrom[1] - size_angstrom[1] * 0.5,
        center_angstrom[1] + size_angstrom[1] * 0.5,
    ))
}

pub(super) fn region_union_bounds(regions: &[&LeafletRegion]) -> Option<(f32, f32, f32, f32)> {
    let mut bounds: Option<(f32, f32, f32, f32)> = None;
    for region in regions {
        let Some(region_bounds) = region_bounds(region) else {
            continue;
        };
        bounds = Some(match bounds {
            Some((xmin, xmax, ymin, ymax)) => (
                xmin.min(region_bounds.0),
                xmax.max(region_bounds.1),
                ymin.min(region_bounds.2),
                ymax.max(region_bounds.3),
            ),
            None => region_bounds,
        });
    }
    bounds
}

pub(super) fn region_bounds(region: &LeafletRegion) -> Option<(f32, f32, f32, f32)> {
    match &region.geometry {
        RegionGeometry::Circle {
            center_angstrom,
            radius_angstrom,
        } => Some((
            center_angstrom[0] - radius_angstrom,
            center_angstrom[0] + radius_angstrom,
            center_angstrom[1] - radius_angstrom,
            center_angstrom[1] + radius_angstrom,
        )),
        RegionGeometry::Ellipse {
            center_angstrom,
            radius_angstrom,
            rotate_degrees,
        } => {
            let theta = rotate_degrees.to_radians();
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let half_x = ((radius_angstrom[0] * cos_t).powi(2)
                + (radius_angstrom[1] * sin_t).powi(2))
            .sqrt();
            let half_y = ((radius_angstrom[0] * sin_t).powi(2)
                + (radius_angstrom[1] * cos_t).powi(2))
            .sqrt();
            Some((
                center_angstrom[0] - half_x,
                center_angstrom[0] + half_x,
                center_angstrom[1] - half_y,
                center_angstrom[1] + half_y,
            ))
        }
        RegionGeometry::Rectangle {
            center_angstrom,
            size_angstrom,
            rotate_degrees,
        } => {
            let theta = rotate_degrees.to_radians();
            let cos_t = theta.cos().abs();
            let sin_t = theta.sin().abs();
            let half_x = size_angstrom[0] * 0.5 * cos_t + size_angstrom[1] * 0.5 * sin_t;
            let half_y = size_angstrom[0] * 0.5 * sin_t + size_angstrom[1] * 0.5 * cos_t;
            Some((
                center_angstrom[0] - half_x,
                center_angstrom[0] + half_x,
                center_angstrom[1] - half_y,
                center_angstrom[1] + half_y,
            ))
        }
        RegionGeometry::Polygon { .. } => {
            let points = transformed_polygon_points(region);
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
    }
}
