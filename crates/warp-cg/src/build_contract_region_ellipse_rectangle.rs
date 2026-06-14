use super::*;

pub(super) fn exact_axis_aligned_ellipse_axis_aligned_rectangle_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() != 2 {
        return None;
    }
    let (ellipse_region, rectangle_region) = match (&regions[0].geometry, &regions[1].geometry) {
        (RegionGeometry::Ellipse { .. }, RegionGeometry::Rectangle { .. }) => {
            (regions[0], regions[1])
        }
        (RegionGeometry::Rectangle { .. }, RegionGeometry::Ellipse { .. }) => {
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
    if rotate_degrees.rem_euclid(180.0).abs() > 1.0e-6 {
        return None;
    }
    if radius_angstrom[0] <= 0.0 || radius_angstrom[1] <= 0.0 {
        return exact_single_region_area(rectangle_region, bounds);
    }

    let rectangle_bounds = clipped_axis_aligned_rectangle_bounds(rectangle_region, bounds)?;
    let rectangle_layout = LayoutBounds {
        xmin: rectangle_bounds.0,
        xmax: rectangle_bounds.1,
        ymin: rectangle_bounds.2,
        ymax: rectangle_bounds.3,
    };
    let ellipse_area = axis_aligned_ellipse_rectangle_intersection_area(
        *center_angstrom,
        *radius_angstrom,
        bounds,
    );
    let rectangle_area =
        (rectangle_bounds.1 - rectangle_bounds.0) * (rectangle_bounds.3 - rectangle_bounds.2);
    let overlap_area = axis_aligned_ellipse_rectangle_intersection_area(
        *center_angstrom,
        *radius_angstrom,
        rectangle_layout,
    );
    Some(ellipse_area + rectangle_area - overlap_area)
}

pub(super) fn exact_rotated_ellipse_axis_aligned_rectangle_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() != 2 {
        return None;
    }
    let (ellipse_region, rectangle_region) = match (&regions[0].geometry, &regions[1].geometry) {
        (RegionGeometry::Ellipse { .. }, RegionGeometry::Rectangle { .. }) => {
            (regions[0], regions[1])
        }
        (RegionGeometry::Rectangle { .. }, RegionGeometry::Ellipse { .. }) => {
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
    if rotate_degrees.rem_euclid(180.0).abs() <= 1.0e-6 {
        return None;
    }
    if radius_angstrom[0] <= 0.0 || radius_angstrom[1] <= 0.0 {
        return exact_single_region_area(rectangle_region, bounds);
    }

    let rectangle_bounds = clipped_axis_aligned_rectangle_bounds(rectangle_region, bounds)?;
    let rectangle_layout = LayoutBounds {
        xmin: rectangle_bounds.0,
        xmax: rectangle_bounds.1,
        ymin: rectangle_bounds.2,
        ymax: rectangle_bounds.3,
    };
    let ellipse_area = ellipse_rectangle_intersection_area(
        *center_angstrom,
        *radius_angstrom,
        *rotate_degrees,
        bounds,
    );
    let rectangle_area =
        (rectangle_bounds.1 - rectangle_bounds.0) * (rectangle_bounds.3 - rectangle_bounds.2);
    let overlap_area = ellipse_rectangle_intersection_area(
        *center_angstrom,
        *radius_angstrom,
        *rotate_degrees,
        rectangle_layout,
    );
    Some(ellipse_area + rectangle_area - overlap_area)
}

pub(super) fn exact_ellipse_axis_aligned_rectangles_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 3 || regions.len() > 17 {
        return None;
    }
    let mut ellipse = None;
    let mut rectangles = Vec::with_capacity(regions.len().saturating_sub(1));
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
            RegionGeometry::Rectangle { .. } => {
                rectangles.push(clipped_axis_aligned_rectangle_bounds(region, bounds)?);
            }
            _ => return None,
        }
    }
    let (center, radius, rotate_degrees) = ellipse?;
    if rectangles.is_empty() {
        return None;
    }
    let ellipse_area = ellipse_rectangle_intersection_area(center, radius, rotate_degrees, bounds);
    let rectangle_area = exact_axis_aligned_rectangle_bounds_union_area(&rectangles);
    let overlap_area = ellipse_axis_aligned_rectangle_bounds_union_intersection_area(
        center,
        radius,
        rotate_degrees,
        &rectangles,
        bounds,
    )?;
    Some((ellipse_area + rectangle_area - overlap_area).max(0.0))
}

pub(super) fn ellipse_axis_aligned_rectangle_bounds_union_intersection_area(
    center: [f32; 2],
    radius: [f32; 2],
    rotate_degrees: f32,
    rectangles: &[(f32, f32, f32, f32)],
    bounds: LayoutBounds,
) -> Option<f32> {
    if rectangles.len() > 16 {
        return None;
    }
    let total_masks = 1usize.checked_shl(rectangles.len() as u32)?;
    let mut area = 0.0f32;
    'subset: for mask in 1usize..total_masks {
        let mut intersection: Option<(f32, f32, f32, f32)> = None;
        let mut bits = 0usize;
        for (idx, rectangle) in rectangles.iter().enumerate() {
            if mask & (1usize << idx) == 0 {
                continue;
            }
            bits += 1;
            intersection = match intersection {
                Some(current) => axis_aligned_bounds_intersection(current, *rectangle),
                None => Some(*rectangle),
            };
            if intersection.is_none() {
                continue 'subset;
            }
        }
        let Some((xmin, xmax, ymin, ymax)) = intersection else {
            continue;
        };
        let clipped = clipped_bounds((xmin, xmax, ymin, ymax), bounds)?;
        let rect_bounds = LayoutBounds {
            xmin: clipped.0,
            xmax: clipped.1,
            ymin: clipped.2,
            ymax: clipped.3,
        };
        let term = ellipse_rectangle_intersection_area(center, radius, rotate_degrees, rect_bounds);
        if bits % 2 == 1 {
            area += term;
        } else {
            area -= term;
        }
    }
    Some(area.max(0.0))
}

pub(super) fn exact_disjoint_ellipses_axis_aligned_rectangles_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 4 || regions.len() > 32 {
        return None;
    }
    let mut ellipses: Vec<(
        &LeafletRegion,
        [f32; 2],
        [f32; 2],
        f32,
        (f32, f32, f32, f32),
    )> = Vec::with_capacity(regions.len());
    let mut rectangles = Vec::with_capacity(regions.len());
    for region in regions {
        match &region.geometry {
            RegionGeometry::Ellipse {
                center_angstrom,
                radius_angstrom,
                rotate_degrees,
            } => {
                if radius_angstrom[0] <= 0.0 || radius_angstrom[1] <= 0.0 {
                    return None;
                }
                let region_bounds = region_bounds(region)
                    .and_then(|candidate_bounds| clipped_bounds(candidate_bounds, bounds))?;
                for (existing_region, _, _, _, existing_bounds) in &ellipses {
                    if !regions_are_exactly_disjoint(
                        existing_region,
                        *existing_bounds,
                        region,
                        region_bounds,
                        bounds,
                    )? {
                        return None;
                    }
                }
                ellipses.push((
                    *region,
                    *center_angstrom,
                    *radius_angstrom,
                    *rotate_degrees,
                    region_bounds,
                ));
            }
            RegionGeometry::Rectangle { .. } => {
                rectangles.push(clipped_axis_aligned_rectangle_bounds(region, bounds)?);
            }
            _ => return None,
        }
    }
    if ellipses.len() < 2 || rectangles.len() < 2 {
        return None;
    }

    let mut area = exact_axis_aligned_rectangle_bounds_union_area(&rectangles);
    for (_, center, radius, rotate_degrees, _) in ellipses {
        area += ellipse_rectangle_intersection_area(center, radius, rotate_degrees, bounds);
        area -= ellipse_axis_aligned_rectangle_bounds_union_intersection_area(
            center,
            radius,
            rotate_degrees,
            &rectangles,
            bounds,
        )?;
    }
    Some(area.max(0.0))
}

pub(super) fn exact_oriented_ellipse_rotated_rectangle_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() != 2 {
        return None;
    }
    let (ellipse_region, rectangle_region) = match (&regions[0].geometry, &regions[1].geometry) {
        (RegionGeometry::Ellipse { .. }, RegionGeometry::Rectangle { .. }) => {
            (regions[0], regions[1])
        }
        (RegionGeometry::Rectangle { .. }, RegionGeometry::Ellipse { .. }) => {
            (regions[1], regions[0])
        }
        _ => return None,
    };
    let RegionGeometry::Ellipse {
        center_angstrom: ellipse_center,
        radius_angstrom,
        rotate_degrees: ellipse_rotate_degrees,
    } = &ellipse_region.geometry
    else {
        return None;
    };
    let RegionGeometry::Rectangle {
        center_angstrom: rectangle_center,
        size_angstrom,
        rotate_degrees: rectangle_rotate_degrees,
    } = &rectangle_region.geometry
    else {
        return None;
    };
    if rectangle_rotate_degrees.rem_euclid(180.0).abs() <= 1.0e-6 {
        return None;
    }
    if radius_angstrom[0] <= 0.0 || radius_angstrom[1] <= 0.0 {
        return exact_single_region_area(rectangle_region, bounds);
    }
    let ellipse_bounds = region_bounds(ellipse_region)?;
    let rectangle_points = convex_polygon_for_region(rectangle_region)?;
    if clipped_bounds(ellipse_bounds, bounds)? != ellipse_bounds
        || !polygon_within_bounds(&rectangle_points, bounds)
    {
        return None;
    }

    let local_ellipse_center = inverse_rotated_xy(
        *ellipse_center,
        *rectangle_center,
        *rectangle_rotate_degrees,
    );
    let local_rectangle = LayoutBounds {
        xmin: -size_angstrom[0] * 0.5,
        xmax: size_angstrom[0] * 0.5,
        ymin: -size_angstrom[1] * 0.5,
        ymax: size_angstrom[1] * 0.5,
    };
    let ellipse_area = std::f32::consts::PI * radius_angstrom[0] * radius_angstrom[1];
    let rectangle_area = size_angstrom[0] * size_angstrom[1];
    let overlap_area = ellipse_rectangle_intersection_area(
        local_ellipse_center,
        *radius_angstrom,
        *ellipse_rotate_degrees - *rectangle_rotate_degrees,
        local_rectangle,
    );
    Some(ellipse_area + rectangle_area - overlap_area)
}

pub(super) fn exact_circle_oriented_ellipse_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() != 2 {
        return None;
    }
    let (circle_region, ellipse_region) = match (&regions[0].geometry, &regions[1].geometry) {
        (RegionGeometry::Circle { .. }, RegionGeometry::Ellipse { .. }) => (regions[0], regions[1]),
        (RegionGeometry::Ellipse { .. }, RegionGeometry::Circle { .. }) => (regions[1], regions[0]),
        _ => return None,
    };
    let RegionGeometry::Circle {
        center_angstrom: circle_center,
        radius_angstrom,
    } = &circle_region.geometry
    else {
        return None;
    };
    let RegionGeometry::Ellipse {
        center_angstrom: ellipse_center,
        radius_angstrom: ellipse_radius,
        rotate_degrees,
    } = &ellipse_region.geometry
    else {
        return None;
    };
    if *radius_angstrom <= 0.0 || ellipse_radius[0] <= 0.0 || ellipse_radius[1] <= 0.0 {
        return None;
    }
    let angle = rotate_degrees.rem_euclid(180.0);
    let circle = CircleRegion {
        center: *circle_center,
        radius: *radius_angstrom,
    };
    let circle_area = circle_rectangle_intersection_area(circle, bounds);
    let ellipse_area =
        axis_aligned_ellipse_rectangle_intersection_area(*ellipse_center, *ellipse_radius, bounds);
    let overlap_area = if angle <= 1.0e-6 {
        circle_axis_aligned_ellipse_intersection_area_clipped(
            circle,
            *ellipse_center,
            *ellipse_radius,
            bounds,
        )
    } else {
        let circle_bounds = region_bounds(circle_region)?;
        let ellipse_bounds = region_bounds(ellipse_region)?;
        if clipped_bounds(circle_bounds, bounds)? != circle_bounds
            || clipped_bounds(ellipse_bounds, bounds)? != ellipse_bounds
        {
            return None;
        }
        let circle_area_unclipped = std::f32::consts::PI * radius_angstrom.powi(2);
        let ellipse_area_unclipped = std::f32::consts::PI * ellipse_radius[0] * ellipse_radius[1];
        if (circle_area - circle_area_unclipped).abs() > 1.0e-3
            || (ellipse_area - ellipse_area_unclipped).abs() > 1.0e-3
        {
            return None;
        }
        let local_circle_center =
            inverse_rotated_xy(*circle_center, *ellipse_center, *rotate_degrees);
        circle_axis_aligned_ellipse_intersection_area(
            CircleRegion {
                center: local_circle_center,
                radius: *radius_angstrom,
            },
            [0.0, 0.0],
            *ellipse_radius,
        )
    };
    Some(circle_area + ellipse_area - overlap_area)
}

pub(super) fn exact_clipped_circle_rotated_ellipse_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() != 2 {
        return None;
    }
    let (circle_region, ellipse_region) = match (&regions[0].geometry, &regions[1].geometry) {
        (RegionGeometry::Circle { .. }, RegionGeometry::Ellipse { .. }) => (regions[0], regions[1]),
        (RegionGeometry::Ellipse { .. }, RegionGeometry::Circle { .. }) => (regions[1], regions[0]),
        _ => return None,
    };
    let RegionGeometry::Circle {
        center_angstrom: circle_center,
        radius_angstrom,
    } = &circle_region.geometry
    else {
        return None;
    };
    let RegionGeometry::Ellipse {
        center_angstrom: ellipse_center,
        radius_angstrom: ellipse_radius,
        rotate_degrees,
    } = &ellipse_region.geometry
    else {
        return None;
    };
    if *radius_angstrom <= 0.0 || ellipse_radius[0] <= 0.0 || ellipse_radius[1] <= 0.0 {
        return None;
    }
    if rotate_degrees.rem_euclid(180.0) <= 1.0e-6 {
        return None;
    }
    let circle = CircleRegion {
        center: *circle_center,
        radius: *radius_angstrom,
    };
    let circle_area = circle_rectangle_intersection_area(circle, bounds);
    let ellipse_area = ellipse_rectangle_intersection_area(
        *ellipse_center,
        *ellipse_radius,
        *rotate_degrees,
        bounds,
    );
    let overlap_area = rotated_ellipse_pair_intersection_area_clipped(
        *circle_center,
        [*radius_angstrom, *radius_angstrom],
        0.0,
        *ellipse_center,
        *ellipse_radius,
        *rotate_degrees,
        bounds,
    );
    Some((circle_area + ellipse_area - overlap_area).max(0.0))
}

pub(super) fn exact_axis_aligned_ellipse_pair_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() != 2 {
        return None;
    }
    let mut ellipses = Vec::with_capacity(2);
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
            return None;
        }
        if rotate_degrees.rem_euclid(180.0) > 1.0e-6 {
            return None;
        }
        ellipses.push((*center_angstrom, *radius_angstrom));
    }
    let left_area =
        axis_aligned_ellipse_rectangle_intersection_area(ellipses[0].0, ellipses[0].1, bounds);
    let right_area =
        axis_aligned_ellipse_rectangle_intersection_area(ellipses[1].0, ellipses[1].1, bounds);
    let overlap_area = axis_aligned_ellipse_pair_intersection_area_clipped(
        ellipses[0].0,
        ellipses[0].1,
        ellipses[1].0,
        ellipses[1].1,
        bounds,
    );
    Some((left_area + right_area - overlap_area).max(0.0))
}

pub(super) fn exact_rotated_ellipse_pair_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() != 2 {
        return None;
    }
    let mut ellipses = Vec::with_capacity(2);
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
            return None;
        }
        if rotate_degrees.rem_euclid(180.0) <= 1.0e-6 {
            return None;
        }
        ellipses.push((*center_angstrom, *radius_angstrom, *rotate_degrees));
    }
    let left_area =
        ellipse_rectangle_intersection_area(ellipses[0].0, ellipses[0].1, ellipses[0].2, bounds);
    let right_area =
        ellipse_rectangle_intersection_area(ellipses[1].0, ellipses[1].1, ellipses[1].2, bounds);
    let overlap_area = rotated_ellipse_pair_intersection_area_clipped(
        ellipses[0].0,
        ellipses[0].1,
        ellipses[0].2,
        ellipses[1].0,
        ellipses[1].1,
        ellipses[1].2,
        bounds,
    );
    Some((left_area + right_area - overlap_area).max(0.0))
}
