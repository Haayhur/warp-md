use super::*;

pub(super) fn exact_circle_simple_polygon_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() != 2 {
        return None;
    }
    let (circle_region, polygon_region) = match (&regions[0].geometry, &regions[1].geometry) {
        (RegionGeometry::Circle { .. }, RegionGeometry::Polygon { .. }) => (regions[0], regions[1]),
        (RegionGeometry::Polygon { .. }, RegionGeometry::Circle { .. }) => (regions[1], regions[0]),
        _ => return None,
    };
    let RegionGeometry::Circle {
        center_angstrom,
        radius_angstrom,
    } = &circle_region.geometry
    else {
        return None;
    };
    if *radius_angstrom <= 0.0 {
        return exact_single_region_area(polygon_region, bounds);
    }
    let polygon = simple_polygon_for_region_clipped_to_bounds(polygon_region, bounds)?;
    let circle = CircleRegion {
        center: *center_angstrom,
        radius: *radius_angstrom,
    };
    let circle_area = circle_rectangle_intersection_area(circle, bounds);
    let polygon_area = polygon_area(&polygon);
    let overlap_area = circle_polygon_intersection_area(circle, &polygon);
    Some((circle_area + polygon_area - overlap_area).max(0.0))
}

pub(super) fn exact_circle_simple_polygons_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 3 || regions.len() > 17 {
        return None;
    }
    let mut circle = None;
    let mut polygons: Vec<(&LeafletRegion, Vec<[f32; 2]>, (f32, f32, f32, f32))> =
        Vec::with_capacity(regions.len().saturating_sub(1));
    for region in regions {
        match &region.geometry {
            RegionGeometry::Circle {
                center_angstrom,
                radius_angstrom,
            } => {
                if circle.is_some() || *radius_angstrom <= 0.0 {
                    return None;
                }
                circle = Some(CircleRegion {
                    center: *center_angstrom,
                    radius: *radius_angstrom,
                });
            }
            RegionGeometry::Polygon { .. } => {
                let polygon = simple_polygon_for_region_clipped_to_bounds(region, bounds)?;
                let polygon_bounds = polygon_bounds(&polygon)?;
                for (existing_region, _, existing_bounds) in &polygons {
                    if !regions_are_exactly_disjoint(
                        existing_region,
                        *existing_bounds,
                        region,
                        polygon_bounds,
                        bounds,
                    )? {
                        return None;
                    }
                }
                polygons.push((*region, polygon, polygon_bounds));
            }
            _ => return None,
        }
    }
    let circle = circle?;
    if polygons.len() < 2 {
        return None;
    }
    let circle_area = circle_rectangle_intersection_area(circle, bounds);
    let mut total_polygon_area = 0.0f32;
    let mut overlap_area = 0.0f32;
    for (_, polygon, _) in &polygons {
        total_polygon_area += polygon_area(polygon);
        overlap_area += circle_polygon_intersection_area(circle, polygon);
    }
    Some((circle_area + total_polygon_area - overlap_area).max(0.0))
}

pub(super) fn exact_ellipse_convex_polygon_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() != 2 {
        return None;
    }
    let (ellipse_region, polygon_region) = match (&regions[0].geometry, &regions[1].geometry) {
        (RegionGeometry::Ellipse { .. }, RegionGeometry::Polygon { .. }) => {
            (regions[0], regions[1])
        }
        (RegionGeometry::Polygon { .. }, RegionGeometry::Ellipse { .. }) => {
            (regions[1], regions[0])
        }
        _ => return None,
    };
    let RegionGeometry::Ellipse {
        center_angstrom,
        radius_angstrom,
        rotate_degrees,
    } = &ellipse_region.geometry
    else {
        return None;
    };
    if radius_angstrom[0] <= 0.0 || radius_angstrom[1] <= 0.0 {
        return exact_single_region_area(polygon_region, bounds);
    }
    let polygon = convex_polygon_for_region(polygon_region)?;
    let clipped_polygon = convex_polygon_intersection(&polygon, &layout_bounds_polygon(bounds));
    let ellipse_area = ellipse_rectangle_intersection_area(
        *center_angstrom,
        *radius_angstrom,
        *rotate_degrees,
        bounds,
    );
    if clipped_polygon.len() < 3 {
        return Some(ellipse_area);
    }
    let transformed_polygon = clipped_polygon
        .iter()
        .map(|point| {
            let local = inverse_rotated_xy(*point, *center_angstrom, *rotate_degrees);
            [local[0] / radius_angstrom[0], local[1] / radius_angstrom[1]]
        })
        .collect::<Vec<_>>();
    let unit_circle = CircleRegion {
        center: [0.0, 0.0],
        radius: 1.0,
    };
    let scale = radius_angstrom[0] * radius_angstrom[1];
    let polygon_area = polygon_area(&clipped_polygon);
    let overlap_area = circle_polygon_intersection_area(unit_circle, &transformed_polygon) * scale;
    Some((ellipse_area + polygon_area - overlap_area).max(0.0))
}

pub(super) fn exact_ellipse_convex_polygons_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 3 || regions.len() > 13 {
        return None;
    }
    let mut ellipse = None;
    let mut polygons = Vec::with_capacity(regions.len().saturating_sub(1));
    let bounds_polygon = layout_bounds_polygon(bounds);
    for region in regions {
        match &region.geometry {
            RegionGeometry::Ellipse {
                center_angstrom,
                radius_angstrom,
                rotate_degrees,
            } => {
                if ellipse.is_some() || radius_angstrom[0] <= 0.0 || radius_angstrom[1] <= 0.0 {
                    return None;
                }
                ellipse = Some((*center_angstrom, *radius_angstrom, *rotate_degrees));
            }
            RegionGeometry::Rectangle { .. } | RegionGeometry::Polygon { .. } => {
                let polygon = convex_polygon_for_region(region)?;
                let clipped = convex_polygon_intersection(&polygon, &bounds_polygon);
                if clipped.len() >= 3 && polygon_area(&clipped) > 1.0e-5 {
                    polygons.push(clipped);
                }
            }
            _ => return None,
        }
    }
    let (center, radius, rotate_degrees) = ellipse?;
    if polygons.is_empty() {
        return Some(ellipse_rectangle_intersection_area(
            center,
            radius,
            rotate_degrees,
            bounds,
        ));
    }
    let ellipse_area = ellipse_rectangle_intersection_area(center, radius, rotate_degrees, bounds);
    let polygon_area = exact_convex_polygon_union_area_from_polygons(&polygons)?;
    let transformed_polygons = polygons
        .iter()
        .map(|polygon| {
            polygon
                .iter()
                .map(|point| {
                    let local = inverse_rotated_xy(*point, center, rotate_degrees);
                    [local[0] / radius[0], local[1] / radius[1]]
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let overlap_area = circle_convex_polygon_union_intersection_area(
        CircleRegion {
            center: [0.0, 0.0],
            radius: 1.0,
        },
        &transformed_polygons,
    )? * radius[0]
        * radius[1];
    Some((ellipse_area + polygon_area - overlap_area).max(0.0))
}

pub(super) fn exact_ellipse_simple_polygon_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() != 2 {
        return None;
    }
    let (ellipse_region, polygon_region) = match (&regions[0].geometry, &regions[1].geometry) {
        (RegionGeometry::Ellipse { .. }, RegionGeometry::Polygon { .. }) => {
            (regions[0], regions[1])
        }
        (RegionGeometry::Polygon { .. }, RegionGeometry::Ellipse { .. }) => {
            (regions[1], regions[0])
        }
        _ => return None,
    };
    let RegionGeometry::Ellipse {
        center_angstrom,
        radius_angstrom,
        rotate_degrees,
    } = &ellipse_region.geometry
    else {
        return None;
    };
    if radius_angstrom[0] <= 0.0 || radius_angstrom[1] <= 0.0 {
        return exact_single_region_area(polygon_region, bounds);
    }
    let polygon = simple_polygon_for_region_clipped_to_bounds(polygon_region, bounds)?;
    let transformed_polygon = polygon
        .iter()
        .map(|point| {
            let local = inverse_rotated_xy(*point, *center_angstrom, *rotate_degrees);
            [local[0] / radius_angstrom[0], local[1] / radius_angstrom[1]]
        })
        .collect::<Vec<_>>();
    let unit_circle = CircleRegion {
        center: [0.0, 0.0],
        radius: 1.0,
    };
    let scale = radius_angstrom[0] * radius_angstrom[1];
    let ellipse_area = ellipse_rectangle_intersection_area(
        *center_angstrom,
        *radius_angstrom,
        *rotate_degrees,
        bounds,
    );
    let polygon_area = polygon_area(&polygon);
    let overlap_area = circle_polygon_intersection_area(unit_circle, &transformed_polygon) * scale;
    Some((ellipse_area + polygon_area - overlap_area).max(0.0))
}

pub(super) fn exact_ellipse_simple_polygons_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 3 || regions.len() > 17 {
        return None;
    }
    let mut ellipse = None;
    let mut polygons: Vec<(&LeafletRegion, Vec<[f32; 2]>, (f32, f32, f32, f32))> =
        Vec::with_capacity(regions.len().saturating_sub(1));
    for region in regions {
        match &region.geometry {
            RegionGeometry::Ellipse {
                center_angstrom,
                radius_angstrom,
                rotate_degrees,
            } => {
                if ellipse.is_some() || radius_angstrom[0] <= 0.0 || radius_angstrom[1] <= 0.0 {
                    return None;
                }
                ellipse = Some((*center_angstrom, *radius_angstrom, *rotate_degrees));
            }
            RegionGeometry::Polygon { .. } => {
                let polygon = simple_polygon_for_region_clipped_to_bounds(region, bounds)?;
                let polygon_bounds = polygon_bounds(&polygon)?;
                for (existing_region, _, existing_bounds) in &polygons {
                    if !regions_are_exactly_disjoint(
                        existing_region,
                        *existing_bounds,
                        region,
                        polygon_bounds,
                        bounds,
                    )? {
                        return None;
                    }
                }
                polygons.push((*region, polygon, polygon_bounds));
            }
            _ => return None,
        }
    }
    let (center, radius, rotate_degrees) = ellipse?;
    if polygons.len() < 2 {
        return None;
    }
    let unit_circle = CircleRegion {
        center: [0.0, 0.0],
        radius: 1.0,
    };
    let scale = radius[0] * radius[1];
    let ellipse_area = ellipse_rectangle_intersection_area(center, radius, rotate_degrees, bounds);
    let mut total_polygon_area = 0.0f32;
    let mut overlap_area = 0.0f32;
    for (_, polygon, _) in &polygons {
        total_polygon_area += polygon_area(polygon);
        let transformed_polygon = polygon
            .iter()
            .map(|point| {
                let local = inverse_rotated_xy(*point, center, rotate_degrees);
                [local[0] / radius[0], local[1] / radius[1]]
            })
            .collect::<Vec<_>>();
        overlap_area += circle_polygon_intersection_area(unit_circle, &transformed_polygon) * scale;
    }
    Some((ellipse_area + total_polygon_area - overlap_area).max(0.0))
}

pub(super) fn simple_polygon_for_region_clipped_to_bounds(
    region: &LeafletRegion,
    bounds: LayoutBounds,
) -> Option<Vec<[f32; 2]>> {
    let RegionGeometry::Polygon { .. } = &region.geometry else {
        return None;
    };
    let points = transformed_polygon_points(region);
    if points.len() < 3 || polygon_has_self_intersections(&points) {
        return None;
    }
    let mut clipped = if polygon_within_bounds(&points, bounds) {
        points
    } else {
        convex_polygon_intersection(&points, &layout_bounds_polygon(bounds))
    };
    if clipped.len() < 3
        || polygon_area(&clipped) <= 1.0e-5
        || polygon_has_self_intersections(&clipped)
        || !polygon_within_bounds(&clipped, bounds)
    {
        return None;
    }
    ensure_ccw_polygon(&mut clipped);
    Some(clipped)
}
