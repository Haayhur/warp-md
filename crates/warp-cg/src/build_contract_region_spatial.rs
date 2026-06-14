use super::*;

pub(super) fn region_contains_point(region: &LeafletRegion, point: [f32; 2]) -> bool {
    match &region.geometry {
        RegionGeometry::Circle {
            center_angstrom,
            radius_angstrom,
        } => {
            let dx = point[0] - center_angstrom[0];
            let dy = point[1] - center_angstrom[1];
            dx * dx + dy * dy <= radius_angstrom.powi(2)
        }
        RegionGeometry::Ellipse {
            center_angstrom,
            radius_angstrom,
            rotate_degrees,
        } => {
            let [x, y] = inverse_rotated_point(point, *center_angstrom, *rotate_degrees);
            (x / radius_angstrom[0]).powi(2) + (y / radius_angstrom[1]).powi(2) <= 1.0
        }
        RegionGeometry::Rectangle {
            center_angstrom,
            size_angstrom,
            rotate_degrees,
        } => {
            let [x, y] = inverse_rotated_point(point, *center_angstrom, *rotate_degrees);
            x.abs() <= size_angstrom[0] * 0.5 && y.abs() <= size_angstrom[1] * 0.5
        }
        RegionGeometry::Polygon { .. } => {
            point_in_polygon(point, &transformed_polygon_points(region))
        }
    }
}

pub(super) fn protein_boundary_margin_violation_distance(
    boundary: &ProteinBoundaryGeometry,
    point: [f32; 2],
    margin_angstrom: f32,
) -> f32 {
    match boundary {
        ProteinBoundaryGeometry::Circle(circle) => {
            let dx = point[0] - circle.center_angstrom[0];
            let dy = point[1] - circle.center_angstrom[1];
            let distance = (dx * dx + dy * dy).sqrt();
            (distance - (circle.radius_angstrom - margin_angstrom).max(0.0)).max(0.0)
        }
        ProteinBoundaryGeometry::Polygon {
            points,
            inset_angstrom,
        } => polygon_boundary_margin_violation_distance(
            points,
            *inset_angstrom,
            point,
            margin_angstrom,
        ),
        ProteinBoundaryGeometry::MultiPolygon {
            polygons,
            inset_angstrom,
        } => {
            if polygons.iter().any(|polygon| {
                polygon_boundary_contains_point(polygon, *inset_angstrom, point, margin_angstrom)
            }) {
                return 0.0;
            }
            polygons
                .iter()
                .map(|polygon| {
                    polygon_boundary_margin_violation_distance(
                        polygon,
                        *inset_angstrom,
                        point,
                        margin_angstrom,
                    )
                })
                .fold(f32::INFINITY, f32::min)
        }
        ProteinBoundaryGeometry::NestedPolygons {
            outer,
            holes,
            inset_angstrom,
        } => {
            if boundary.contains_point_with_margin(point, margin_angstrom) {
                return 0.0;
            }
            if !polygon_boundary_contains_point(outer, *inset_angstrom, point, margin_angstrom) {
                return polygon_boundary_margin_violation_distance(
                    outer,
                    *inset_angstrom,
                    point,
                    margin_angstrom,
                );
            }
            let hole_margin = *inset_angstrom + margin_angstrom.max(0.0);
            holes
                .iter()
                .filter(|hole| polygon_hole_rejects_point(hole, hole_margin, point))
                .map(|hole| {
                    if point_in_polygon_or_boundary(point, hole) {
                        polygon_boundary_distance(point, hole)
                    } else {
                        (hole_margin - polygon_boundary_distance(point, hole)).max(0.0)
                    }
                })
                .fold(0.0, f32::max)
        }
        ProteinBoundaryGeometry::NestedPolygonForest {
            rings,
            inset_angstrom,
        } => {
            if boundary.contains_point_with_margin(point, margin_angstrom) {
                return 0.0;
            }
            let effective_margin = if *inset_angstrom > 0.0 {
                *inset_angstrom + margin_angstrom.max(0.0)
            } else {
                margin_angstrom.max(0.0)
            };
            if nested_polygon_forest_contains_point(rings, *inset_angstrom, point, 0.0)
                && effective_margin > 1.0e-6
            {
                let nearest = rings
                    .iter()
                    .map(|ring| polygon_boundary_distance(point, ring))
                    .fold(f32::INFINITY, f32::min);
                return (effective_margin - nearest).max(0.0);
            }
            rings
                .iter()
                .map(|ring| polygon_boundary_distance(point, ring))
                .fold(f32::INFINITY, f32::min)
        }
    }
}

pub(super) fn polygon_boundary_margin_violation_distance(
    points: &[[f32; 2]],
    inset_angstrom: f32,
    point: [f32; 2],
    margin_angstrom: f32,
) -> f32 {
    if !point_in_polygon(point, points) {
        return polygon_boundary_distance(point, points);
    }
    if inset_angstrom <= 0.0 {
        0.0
    } else {
        (inset_angstrom + margin_angstrom - polygon_boundary_distance(point, points)).max(0.0)
    }
}

pub(super) fn region_boundary_distance(region: &LeafletRegion, point: [f32; 2]) -> f32 {
    match &region.geometry {
        RegionGeometry::Circle {
            center_angstrom,
            radius_angstrom,
        } => {
            let dx = point[0] - center_angstrom[0];
            let dy = point[1] - center_angstrom[1];
            ((dx * dx + dy * dy).sqrt() - *radius_angstrom).abs()
        }
        RegionGeometry::Ellipse {
            center_angstrom,
            radius_angstrom,
            rotate_degrees,
        } => {
            let local = inverse_rotated_xy(point, *center_angstrom, *rotate_degrees);
            let nx = local[0] / radius_angstrom[0];
            let ny = local[1] / radius_angstrom[1];
            let normalized = (nx * nx + ny * ny).sqrt();
            (normalized - 1.0).abs() * radius_angstrom[0].min(radius_angstrom[1])
        }
        RegionGeometry::Rectangle {
            center_angstrom,
            size_angstrom,
            rotate_degrees,
        } => {
            let local = inverse_rotated_xy(point, *center_angstrom, *rotate_degrees);
            rectangle_boundary_distance(local, [size_angstrom[0] * 0.5, size_angstrom[1] * 0.5])
        }
        RegionGeometry::Polygon { .. } => {
            polygon_boundary_distance(point, &transformed_polygon_points(region))
        }
    }
}

pub(super) fn region_boundary_distance_periodic_basis(
    region: &LeafletRegion,
    point: [f32; 2],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> f32 {
    periodic_region_point_images_basis(region, point, bounds, periodicity, basis)
        .into_iter()
        .map(|image| region_boundary_distance(region, image))
        .fold(f32::INFINITY, f32::min)
}

pub(super) fn region_contains_point_periodic_basis(
    region: &LeafletRegion,
    point: [f32; 2],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> bool {
    periodic_region_point_images_basis(region, point, bounds, periodicity, basis)
        .into_iter()
        .any(|image| region_contains_point(region, image))
}

pub(super) fn closest_periodic_region_image_basis(
    region: &LeafletRegion,
    point: [f32; 2],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> [f32; 2] {
    periodic_region_point_images_basis(region, point, bounds, periodicity, basis)
        .into_iter()
        .min_by(|left, right| {
            region_boundary_distance(region, *left)
                .partial_cmp(&region_boundary_distance(region, *right))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(point)
}

fn periodic_region_point_images_basis(
    region: &LeafletRegion,
    point: [f32; 2],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> Vec<[f32; 2]> {
    if let Some(basis) = basis.filter(|_| periodicity.x && periodicity.y) {
        if let Some((xmin, xmax, ymin, ymax)) = region_bounds(region) {
            let fractional = basis.fractional(point);
            let corners = [
                basis.fractional([xmin, ymin]),
                basis.fractional([xmax, ymin]),
                basis.fractional([xmax, ymax]),
                basis.fractional([xmin, ymax]),
            ];
            let min_fx = corners
                .iter()
                .map(|corner| corner[0])
                .fold(f32::INFINITY, f32::min);
            let max_fx = corners
                .iter()
                .map(|corner| corner[0])
                .fold(f32::NEG_INFINITY, f32::max);
            let min_fy = corners
                .iter()
                .map(|corner| corner[1])
                .fold(f32::INFINITY, f32::min);
            let max_fy = corners
                .iter()
                .map(|corner| corner[1])
                .fold(f32::NEG_INFINITY, f32::max);
            let x_offsets = offset_range_for_interval(min_fx, max_fx, fractional[0]);
            let y_offsets = offset_range_for_interval(min_fy, max_fy, fractional[1]);
            let mut images = Vec::with_capacity(x_offsets.len() * y_offsets.len());
            for x_offset in x_offsets {
                for y_offset in &y_offsets {
                    images.push(basis.cartesian([
                        fractional[0] + x_offset as f32,
                        fractional[1] + *y_offset as f32,
                    ]));
                }
            }
            if !images.is_empty() {
                return images;
            }
        }
    } else if let Some((xmin, xmax, ymin, ymax)) = region_bounds(region) {
        let x_len = bounds.xmax - bounds.xmin;
        let y_len = bounds.ymax - bounds.ymin;
        let x_offsets = if periodicity.x && x_len > 0.0 && x_len.is_finite() {
            offset_range_for_interval(xmin / x_len, xmax / x_len, point[0] / x_len)
        } else {
            vec![0]
        };
        let y_offsets = if periodicity.y && y_len > 0.0 && y_len.is_finite() {
            offset_range_for_interval(ymin / y_len, ymax / y_len, point[1] / y_len)
        } else {
            vec![0]
        };
        let mut images = Vec::with_capacity(x_offsets.len() * y_offsets.len());
        for x_offset in x_offsets {
            for y_offset in &y_offsets {
                images.push([
                    point[0] + x_offset as f32 * x_len,
                    point[1] + *y_offset as f32 * y_len,
                ]);
            }
        }
        if !images.is_empty() {
            return images;
        }
    }
    periodic_point_images_basis(point, bounds, periodicity, basis)
}

fn offset_range_for_interval(min_value: f32, max_value: f32, point_value: f32) -> Vec<i32> {
    if !min_value.is_finite() || !max_value.is_finite() || !point_value.is_finite() {
        return vec![0];
    }
    let start = (min_value - point_value).floor() as i32 - 1;
    let end = (max_value - point_value).ceil() as i32 + 1;
    (start.min(end)..=start.max(end)).collect()
}

pub(super) fn periodic_point_images_basis(
    point: [f32; 2],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> Vec<[f32; 2]> {
    const OFFSETS: [f32; 17] = [
        -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ];
    if let Some(basis) = basis.filter(|_| periodicity.x && periodicity.y) {
        let fractional = basis.fractional(point);
        let mut images = Vec::with_capacity(OFFSETS.len() * OFFSETS.len());
        for x_offset in OFFSETS {
            for y_offset in OFFSETS {
                images.push(basis.cartesian([fractional[0] + x_offset, fractional[1] + y_offset]));
            }
        }
        return images;
    }
    let x_len = bounds.xmax - bounds.xmin;
    let y_len = bounds.ymax - bounds.ymin;
    let x_offsets: &[f32] = if periodicity.x && x_len > 0.0 && x_len.is_finite() {
        &OFFSETS
    } else {
        &[0.0]
    };
    let y_offsets: &[f32] = if periodicity.y && y_len > 0.0 && y_len.is_finite() {
        &OFFSETS
    } else {
        &[0.0]
    };
    let mut images = Vec::with_capacity(x_offsets.len() * y_offsets.len());
    for x_offset in x_offsets {
        for y_offset in y_offsets {
            images.push([point[0] + x_offset * x_len, point[1] + y_offset * y_len]);
        }
    }
    images
}

pub(super) fn wrap_coordinate_for_layout_axis(
    value: f32,
    min: f32,
    max: f32,
    periodic: bool,
) -> f32 {
    if !periodic || max <= min || !value.is_finite() {
        return value;
    }
    (value - min).rem_euclid(max - min) + min
}

pub(super) fn wrap_point_for_layout_basis(
    point: [f32; 2],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> [f32; 2] {
    if let Some(basis) = basis.filter(|_| periodicity.x && periodicity.y) {
        let fractional = basis.fractional(point);
        return basis.cartesian([fractional[0].rem_euclid(1.0), fractional[1].rem_euclid(1.0)]);
    }
    [
        wrap_coordinate_for_layout_axis(point[0], bounds.xmin, bounds.xmax, periodicity.x),
        wrap_coordinate_for_layout_axis(point[1], bounds.ymin, bounds.ymax, periodicity.y),
    ]
}
