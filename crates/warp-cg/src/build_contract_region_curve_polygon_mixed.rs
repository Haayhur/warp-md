use super::*;

pub(super) fn exact_circle_convex_polygons_disjoint_ellipses_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 4 || regions.len() > 17 {
        return None;
    }
    let mut circle_region = None;
    let mut polygons = Vec::with_capacity(regions.len().saturating_sub(2));
    let mut ellipses: Vec<(&LeafletRegion, f32, (f32, f32, f32, f32))> =
        Vec::with_capacity(regions.len().saturating_sub(1));

    for region in regions {
        match &region.geometry {
            RegionGeometry::Circle {
                radius_angstrom, ..
            } => {
                if circle_region.is_some() || *radius_angstrom <= 0.0 {
                    return None;
                }
                circle_region = Some(*region);
            }
            RegionGeometry::Rectangle { .. } | RegionGeometry::Polygon { .. } => {
                convex_polygon_for_region(region)?;
                polygons.push(*region);
            }
            RegionGeometry::Ellipse {
                radius_angstrom, ..
            } => {
                if radius_angstrom[0] <= 0.0 || radius_angstrom[1] <= 0.0 {
                    return None;
                }
                let Some(ellipse_bounds) =
                    region_bounds(region).and_then(|bounds_raw| clipped_bounds(bounds_raw, bounds))
                else {
                    continue;
                };
                let ellipse_area = exact_single_region_area(region, bounds)?;
                if ellipse_area <= 1.0e-6 {
                    continue;
                }
                for (existing, _, existing_bounds) in &ellipses {
                    if !regions_are_exactly_disjoint(
                        existing,
                        *existing_bounds,
                        region,
                        ellipse_bounds,
                        bounds,
                    )? {
                        return None;
                    }
                }
                ellipses.push((*region, ellipse_area, ellipse_bounds));
            }
        }
    }

    let circle_region = circle_region?;
    if polygons.len() < 2 || ellipses.is_empty() {
        return None;
    }
    require_regions_disjoint_from_polygons(&ellipses, &polygons, bounds)?;

    let mut circle_polygons = Vec::with_capacity(polygons.len() + 1);
    circle_polygons.push(circle_region);
    circle_polygons.extend(polygons);
    let base_area = exact_circle_convex_polygons_region_union_area(&circle_polygons, bounds)?;
    let circle_area = exact_single_region_area(circle_region, bounds)?;
    let mut total = base_area;
    for (ellipse_region, ellipse_area, _) in &ellipses {
        let pair = [circle_region, *ellipse_region];
        let pair_union = exact_circle_oriented_ellipse_region_union_area(&pair, bounds)
            .or_else(|| exact_clipped_circle_rotated_ellipse_region_union_area(&pair, bounds))?;
        let circle_ellipse_overlap = (circle_area + *ellipse_area - pair_union).max(0.0);
        total += *ellipse_area - circle_ellipse_overlap;
    }
    Some(total.max(0.0))
}

pub(super) fn exact_ellipse_convex_polygons_disjoint_circles_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 4 || regions.len() > 17 {
        return None;
    }
    let mut ellipse_region = None;
    let mut polygons = Vec::with_capacity(regions.len().saturating_sub(2));
    let mut circles: Vec<(&LeafletRegion, f32, (f32, f32, f32, f32))> =
        Vec::with_capacity(regions.len().saturating_sub(1));

    for region in regions {
        match &region.geometry {
            RegionGeometry::Ellipse {
                radius_angstrom, ..
            } => {
                if ellipse_region.is_some()
                    || radius_angstrom[0] <= 0.0
                    || radius_angstrom[1] <= 0.0
                {
                    return None;
                }
                ellipse_region = Some(*region);
            }
            RegionGeometry::Rectangle { .. } | RegionGeometry::Polygon { .. } => {
                convex_polygon_for_region(region)?;
                polygons.push(*region);
            }
            RegionGeometry::Circle {
                radius_angstrom, ..
            } => {
                if *radius_angstrom <= 0.0 {
                    return None;
                }
                let Some(circle_bounds) =
                    region_bounds(region).and_then(|bounds_raw| clipped_bounds(bounds_raw, bounds))
                else {
                    continue;
                };
                let circle_area = exact_single_region_area(region, bounds)?;
                if circle_area <= 1.0e-6 {
                    continue;
                }
                for (existing, _, existing_bounds) in &circles {
                    if !regions_are_exactly_disjoint(
                        existing,
                        *existing_bounds,
                        region,
                        circle_bounds,
                        bounds,
                    )? {
                        return None;
                    }
                }
                circles.push((*region, circle_area, circle_bounds));
            }
        }
    }

    let ellipse_region = ellipse_region?;
    if polygons.len() < 2 || circles.is_empty() {
        return None;
    }
    require_regions_disjoint_from_polygons(&circles, &polygons, bounds)?;

    let mut ellipse_polygons = Vec::with_capacity(polygons.len() + 1);
    ellipse_polygons.push(ellipse_region);
    ellipse_polygons.extend(polygons);
    let base_area = exact_ellipse_convex_polygons_region_union_area(&ellipse_polygons, bounds)?;
    let ellipse_area = exact_single_region_area(ellipse_region, bounds)?;
    let mut total = base_area;
    for (circle_region, circle_area, _) in &circles {
        let pair = [*circle_region, ellipse_region];
        let pair_union = exact_circle_oriented_ellipse_region_union_area(&pair, bounds)
            .or_else(|| exact_clipped_circle_rotated_ellipse_region_union_area(&pair, bounds))?;
        let ellipse_circle_overlap = (ellipse_area + *circle_area - pair_union).max(0.0);
        total += *circle_area - ellipse_circle_overlap;
    }
    Some(total.max(0.0))
}

pub(super) fn exact_circle_convex_polygons_disjoint_mixed_shapes_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    exact_curve_convex_polygons_disjoint_mixed_shapes_region_union_area(regions, bounds, "circle")
}

pub(super) fn exact_ellipse_convex_polygons_disjoint_mixed_shapes_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    exact_curve_convex_polygons_disjoint_mixed_shapes_region_union_area(regions, bounds, "ellipse")
}

fn exact_curve_convex_polygons_disjoint_mixed_shapes_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
    curve_kind: &str,
) -> Option<f32> {
    if regions.len() < 5 || regions.len() > 17 {
        return None;
    }
    let mut curve_region = None;
    let mut polygons = Vec::with_capacity(regions.len().saturating_sub(3));
    let mut secondaries: Vec<(&LeafletRegion, f32, (f32, f32, f32, f32))> =
        Vec::with_capacity(regions.len().saturating_sub(1));
    let mut secondary_kinds = std::collections::BTreeSet::new();

    for region in regions {
        match (&region.geometry, curve_kind) {
            (
                RegionGeometry::Circle {
                    radius_angstrom, ..
                },
                "circle",
            ) => {
                if curve_region.is_some() || *radius_angstrom <= 0.0 {
                    return None;
                }
                curve_region = Some(*region);
            }
            (
                RegionGeometry::Ellipse {
                    radius_angstrom, ..
                },
                "ellipse",
            ) => {
                if curve_region.is_some() || radius_angstrom[0] <= 0.0 || radius_angstrom[1] <= 0.0
                {
                    return None;
                }
                curve_region = Some(*region);
            }
            (RegionGeometry::Polygon { .. }, _) => {
                if convex_polygon_for_region(region).is_some() {
                    polygons.push(*region);
                    continue;
                }
                return None;
            }
            (
                RegionGeometry::Circle { .. }
                | RegionGeometry::Ellipse { .. }
                | RegionGeometry::Rectangle { .. },
                _,
            ) => {
                let Some(secondary_bounds) =
                    region_bounds(region).and_then(|bounds_raw| clipped_bounds(bounds_raw, bounds))
                else {
                    continue;
                };
                let secondary_area = exact_single_region_area(region, bounds)?;
                if secondary_area <= 1.0e-6 {
                    continue;
                }
                for (existing, _, existing_bounds) in &secondaries {
                    if !regions_are_exactly_disjoint(
                        existing,
                        *existing_bounds,
                        region,
                        secondary_bounds,
                        bounds,
                    )? {
                        return None;
                    }
                }
                secondary_kinds.insert(region_geometry_kind(region));
                secondaries.push((*region, secondary_area, secondary_bounds));
            }
        }
    }

    let curve_region = curve_region?;
    if polygons.len() < 2 || secondaries.len() < 2 || secondary_kinds.len() < 2 {
        return None;
    }
    require_regions_disjoint_from_polygons(&secondaries, &polygons, bounds)?;

    let mut curve_polygons = Vec::with_capacity(polygons.len() + 1);
    curve_polygons.push(curve_region);
    curve_polygons.extend(polygons);
    let base_area = match curve_kind {
        "circle" => exact_circle_convex_polygons_region_union_area(&curve_polygons, bounds)?,
        "ellipse" => exact_ellipse_convex_polygons_region_union_area(&curve_polygons, bounds)?,
        _ => return None,
    };
    let curve_area = exact_single_region_area(curve_region, bounds)?;
    let mut total = base_area;
    for (secondary_region, secondary_area, _) in &secondaries {
        let pair_union =
            exact_pair_region_union_area_without_disjoint(curve_region, secondary_region, bounds)?;
        let curve_secondary_overlap = (curve_area + *secondary_area - pair_union).max(0.0);
        total += *secondary_area - curve_secondary_overlap;
    }
    Some(total.max(0.0))
}

fn require_regions_disjoint_from_polygons(
    regions: &[(&LeafletRegion, f32, (f32, f32, f32, f32))],
    polygons: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<()> {
    for (region, _, candidate_bounds) in regions {
        for polygon_region in polygons {
            let polygon_bounds = region_bounds(polygon_region)
                .and_then(|bounds_raw| clipped_bounds(bounds_raw, bounds))?;
            if !regions_are_exactly_disjoint(
                region,
                *candidate_bounds,
                polygon_region,
                polygon_bounds,
                bounds,
            )? {
                return None;
            }
        }
    }
    Some(())
}
