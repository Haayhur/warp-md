use super::*;

pub(super) fn exact_circle_disjoint_ellipses_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 3 || regions.len() > 17 {
        return None;
    }
    let mut circle_region = None;
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
            _ => return None,
        }
    }
    let circle_region = circle_region?;
    if ellipses.len() < 2 {
        return None;
    }
    let circle_area = exact_single_region_area(circle_region, bounds)?;
    let mut ellipse_area_sum = 0.0f32;
    let mut circle_ellipse_overlap_sum = 0.0f32;
    for (ellipse_region, ellipse_area, _) in &ellipses {
        let pair = [circle_region, *ellipse_region];
        let pair_union = exact_circle_oriented_ellipse_region_union_area(&pair, bounds)
            .or_else(|| exact_clipped_circle_rotated_ellipse_region_union_area(&pair, bounds))?;
        ellipse_area_sum += *ellipse_area;
        circle_ellipse_overlap_sum += (circle_area + *ellipse_area - pair_union).max(0.0);
    }
    Some((circle_area + ellipse_area_sum - circle_ellipse_overlap_sum).max(0.0))
}

pub(super) fn exact_ellipse_disjoint_circles_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 3 || regions.len() > 17 {
        return None;
    }
    let mut ellipse_region = None;
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
            _ => return None,
        }
    }
    let ellipse_region = ellipse_region?;
    if circles.len() < 2 {
        return None;
    }
    let ellipse_area = exact_single_region_area(ellipse_region, bounds)?;
    let mut circle_area_sum = 0.0f32;
    let mut ellipse_circle_overlap_sum = 0.0f32;
    for (circle_region, circle_area, _) in &circles {
        let pair = [*circle_region, ellipse_region];
        let pair_union = exact_circle_oriented_ellipse_region_union_area(&pair, bounds)
            .or_else(|| exact_clipped_circle_rotated_ellipse_region_union_area(&pair, bounds))?;
        circle_area_sum += *circle_area;
        ellipse_circle_overlap_sum += (ellipse_area + *circle_area - pair_union).max(0.0);
    }
    Some((ellipse_area + circle_area_sum - ellipse_circle_overlap_sum).max(0.0))
}

pub(super) fn exact_circle_disjoint_mixed_shapes_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 3 || regions.len() > 17 {
        return None;
    }
    let mut circle_region = None;
    let mut secondaries: Vec<(&LeafletRegion, f32, (f32, f32, f32, f32))> =
        Vec::with_capacity(regions.len().saturating_sub(1));
    let mut secondary_kinds = std::collections::BTreeSet::new();
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
            RegionGeometry::Ellipse { .. }
            | RegionGeometry::Rectangle { .. }
            | RegionGeometry::Polygon { .. } => {
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
    let circle_region = circle_region?;
    if secondaries.len() < 2 || secondary_kinds.len() < 2 {
        return None;
    }
    let circle_area = exact_single_region_area(circle_region, bounds)?;
    let mut secondary_area_sum = 0.0f32;
    let mut circle_secondary_overlap_sum = 0.0f32;
    for (secondary_region, secondary_area, _) in &secondaries {
        let pair_union =
            exact_pair_region_union_area_without_disjoint(circle_region, secondary_region, bounds)?;
        secondary_area_sum += *secondary_area;
        circle_secondary_overlap_sum += (circle_area + *secondary_area - pair_union).max(0.0);
    }
    Some((circle_area + secondary_area_sum - circle_secondary_overlap_sum).max(0.0))
}

pub(super) fn exact_ellipse_disjoint_mixed_shapes_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 3 || regions.len() > 17 {
        return None;
    }
    let mut ellipse_region = None;
    let mut secondaries: Vec<(&LeafletRegion, f32, (f32, f32, f32, f32))> =
        Vec::with_capacity(regions.len().saturating_sub(1));
    let mut secondary_kinds = std::collections::BTreeSet::new();
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
            RegionGeometry::Circle { .. }
            | RegionGeometry::Rectangle { .. }
            | RegionGeometry::Polygon { .. } => {
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
    let ellipse_region = ellipse_region?;
    if secondaries.len() < 2 || secondary_kinds.len() < 2 {
        return None;
    }
    let ellipse_area = exact_single_region_area(ellipse_region, bounds)?;
    let mut secondary_area_sum = 0.0f32;
    let mut ellipse_secondary_overlap_sum = 0.0f32;
    for (secondary_region, secondary_area, _) in &secondaries {
        let pair_union = exact_pair_region_union_area_without_disjoint(
            ellipse_region,
            secondary_region,
            bounds,
        )?;
        secondary_area_sum += *secondary_area;
        ellipse_secondary_overlap_sum += (ellipse_area + *secondary_area - pair_union).max(0.0);
    }
    Some((ellipse_area + secondary_area_sum - ellipse_secondary_overlap_sum).max(0.0))
}

pub(super) fn exact_rectangle_disjoint_mixed_shapes_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 3 || regions.len() > 17 {
        return None;
    }
    let mut rectangle_region = None;
    let mut secondaries: Vec<(&LeafletRegion, f32, (f32, f32, f32, f32))> =
        Vec::with_capacity(regions.len().saturating_sub(1));
    let mut secondary_kinds = std::collections::BTreeSet::new();
    for region in regions {
        match &region.geometry {
            RegionGeometry::Rectangle { size_angstrom, .. } => {
                if rectangle_region.is_some() || size_angstrom[0] <= 0.0 || size_angstrom[1] <= 0.0
                {
                    return None;
                }
                rectangle_region = Some(*region);
            }
            RegionGeometry::Circle { .. }
            | RegionGeometry::Ellipse { .. }
            | RegionGeometry::Polygon { .. } => {
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
    let rectangle_region = rectangle_region?;
    if secondaries.len() < 2 || secondary_kinds.len() < 2 {
        return None;
    }
    let rectangle_area = exact_single_region_area(rectangle_region, bounds)?;
    let mut secondary_area_sum = 0.0f32;
    let mut rectangle_secondary_overlap_sum = 0.0f32;
    for (secondary_region, secondary_area, _) in &secondaries {
        let pair_union = exact_pair_region_union_area_without_disjoint(
            rectangle_region,
            secondary_region,
            bounds,
        )?;
        secondary_area_sum += *secondary_area;
        rectangle_secondary_overlap_sum += (rectangle_area + *secondary_area - pair_union).max(0.0);
    }
    Some((rectangle_area + secondary_area_sum - rectangle_secondary_overlap_sum).max(0.0))
}

pub(super) fn exact_convex_polygon_disjoint_mixed_shapes_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 3 || regions.len() > 17 {
        return None;
    }
    let mut polygon_region = None;
    let mut secondaries: Vec<(&LeafletRegion, f32, (f32, f32, f32, f32))> =
        Vec::with_capacity(regions.len().saturating_sub(1));
    let mut secondary_kinds = std::collections::BTreeSet::new();
    for region in regions {
        match &region.geometry {
            RegionGeometry::Polygon { .. } => {
                if polygon_region.is_some() || convex_polygon_for_region(region).is_none() {
                    return None;
                }
                polygon_region = Some(*region);
            }
            RegionGeometry::Circle { .. }
            | RegionGeometry::Ellipse { .. }
            | RegionGeometry::Rectangle { .. } => {
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
    let polygon_region = polygon_region?;
    if secondaries.len() < 2 || secondary_kinds.len() < 2 {
        return None;
    }
    let polygon_area = exact_single_region_area(polygon_region, bounds)?;
    let mut secondary_area_sum = 0.0f32;
    let mut polygon_secondary_overlap_sum = 0.0f32;
    for (secondary_region, secondary_area, _) in &secondaries {
        let pair_union = exact_pair_region_union_area_without_disjoint(
            polygon_region,
            secondary_region,
            bounds,
        )?;
        secondary_area_sum += *secondary_area;
        polygon_secondary_overlap_sum += (polygon_area + *secondary_area - pair_union).max(0.0);
    }
    Some((polygon_area + secondary_area_sum - polygon_secondary_overlap_sum).max(0.0))
}

pub(super) fn region_geometry_kind(region: &LeafletRegion) -> &'static str {
    match &region.geometry {
        RegionGeometry::Circle { .. } => "circle",
        RegionGeometry::Ellipse { .. } => "ellipse",
        RegionGeometry::Rectangle { .. } => "rectangle",
        RegionGeometry::Polygon { .. } => "polygon",
    }
}
