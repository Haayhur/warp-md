use super::*;

pub(super) fn exact_circle_axis_aligned_rectangle_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() != 2 {
        return None;
    }
    let (circle_region, rectangle_region) = match (&regions[0].geometry, &regions[1].geometry) {
        (RegionGeometry::Circle { .. }, RegionGeometry::Rectangle { .. }) => {
            (regions[0], regions[1])
        }
        (RegionGeometry::Rectangle { .. }, RegionGeometry::Circle { .. }) => {
            (regions[1], regions[0])
        }
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
        return exact_single_region_area(rectangle_region, bounds);
    }
    let rectangle_bounds = clipped_axis_aligned_rectangle_bounds(rectangle_region, bounds)?;
    let rectangle_layout = LayoutBounds {
        xmin: rectangle_bounds.0,
        xmax: rectangle_bounds.1,
        ymin: rectangle_bounds.2,
        ymax: rectangle_bounds.3,
    };
    let circle = CircleRegion {
        center: *center_angstrom,
        radius: *radius_angstrom,
    };
    let circle_area = circle_rectangle_intersection_area(circle, bounds);
    let rectangle_area =
        (rectangle_bounds.1 - rectangle_bounds.0) * (rectangle_bounds.3 - rectangle_bounds.2);
    let overlap_area = circle_rectangle_intersection_area(circle, rectangle_layout);
    Some(circle_area + rectangle_area - overlap_area)
}

pub(super) fn exact_circles_axis_aligned_rectangle_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 3 || regions.len() > 65 {
        return None;
    }
    let mut circles = Vec::with_capacity(regions.len().saturating_sub(1));
    let mut rectangle = None;
    for region in regions {
        match &region.geometry {
            RegionGeometry::Circle {
                center_angstrom,
                radius_angstrom,
            } => {
                if *radius_angstrom > 0.0 {
                    circles.push(CircleRegion {
                        center: *center_angstrom,
                        radius: *radius_angstrom,
                    });
                }
            }
            RegionGeometry::Rectangle { .. } => {
                if rectangle.is_some() || axis_aligned_rectangle_bounds(region).is_none() {
                    return None;
                }
                rectangle = clipped_axis_aligned_rectangle_bounds(region, bounds);
            }
            _ => return None,
        }
    }
    if circles.len() < 2 {
        return None;
    }
    let circle_area = exact_clipped_circle_union_area(&circles, bounds);
    let Some((xmin, xmax, ymin, ymax)) = rectangle else {
        return Some(circle_area);
    };
    let rectangle_area = (xmax - xmin) * (ymax - ymin);
    let rectangle_bounds = LayoutBounds {
        xmin,
        xmax,
        ymin,
        ymax,
    };
    let overlap_area = exact_clipped_circle_union_area(&circles, rectangle_bounds);
    Some((circle_area + rectangle_area - overlap_area).max(0.0))
}

pub(super) fn exact_circle_axis_aligned_rectangles_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 3 || regions.len() > 17 {
        return None;
    }
    let mut circle = None;
    let mut rectangles = Vec::with_capacity(regions.len().saturating_sub(1));
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
            RegionGeometry::Rectangle { .. } => {
                rectangles.push(clipped_axis_aligned_rectangle_bounds(region, bounds)?);
            }
            _ => return None,
        }
    }
    let circle = circle?;
    if rectangles.is_empty() {
        return None;
    }
    let circle_area = circle_rectangle_intersection_area(circle, bounds);
    let rectangle_area = exact_axis_aligned_rectangle_bounds_union_area(&rectangles);
    let overlap_area =
        circle_axis_aligned_rectangle_bounds_union_intersection_area(circle, &rectangles, bounds)?;
    Some((circle_area + rectangle_area - overlap_area).max(0.0))
}

pub(super) fn exact_circles_axis_aligned_rectangles_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 4 || regions.len() > 24 {
        return None;
    }
    let mut circles = Vec::with_capacity(regions.len());
    let mut rectangles = Vec::with_capacity(regions.len());
    for region in regions {
        match &region.geometry {
            RegionGeometry::Circle {
                center_angstrom,
                radius_angstrom,
            } => {
                if *radius_angstrom > 0.0 {
                    circles.push(CircleRegion {
                        center: *center_angstrom,
                        radius: *radius_angstrom,
                    });
                }
            }
            RegionGeometry::Rectangle { .. } => {
                rectangles.push(clipped_axis_aligned_rectangle_bounds(region, bounds)?);
            }
            _ => return None,
        }
    }
    if circles.len() < 2 || rectangles.len() < 2 {
        return None;
    }
    let circle_area = exact_clipped_circle_union_area(&circles, bounds);
    let rectangle_area = exact_axis_aligned_rectangle_bounds_union_area(&rectangles);
    let overlap_area = clipped_circle_union_axis_aligned_rectangle_bounds_union_intersection_area(
        &circles,
        &rectangles,
        bounds,
    )?;
    Some((circle_area + rectangle_area - overlap_area).max(0.0))
}

pub(super) fn circle_axis_aligned_rectangle_bounds_union_intersection_area(
    circle: CircleRegion,
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
        let term = circle_rectangle_intersection_area(circle, rect_bounds);
        if bits % 2 == 1 {
            area += term;
        } else {
            area -= term;
        }
    }
    Some(area.max(0.0))
}

pub(super) fn clipped_circle_union_axis_aligned_rectangle_bounds_union_intersection_area(
    circles: &[CircleRegion],
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
        let term = exact_clipped_circle_union_area(circles, rect_bounds);
        if bits % 2 == 1 {
            area += term;
        } else {
            area -= term;
        }
    }
    Some(area.max(0.0))
}

pub(super) fn axis_aligned_bounds_intersection(
    left: (f32, f32, f32, f32),
    right: (f32, f32, f32, f32),
) -> Option<(f32, f32, f32, f32)> {
    let xmin = left.0.max(right.0);
    let xmax = left.1.min(right.1);
    let ymin = left.2.max(right.2);
    let ymax = left.3.min(right.3);
    (xmax > xmin && ymax > ymin).then_some((xmin, xmax, ymin, ymax))
}

pub(super) fn exact_circle_rotated_rectangle_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() != 2 {
        return None;
    }
    let (circle_region, rectangle_region) = match (&regions[0].geometry, &regions[1].geometry) {
        (RegionGeometry::Circle { .. }, RegionGeometry::Rectangle { .. }) => {
            (regions[0], regions[1])
        }
        (RegionGeometry::Rectangle { .. }, RegionGeometry::Circle { .. }) => {
            (regions[1], regions[0])
        }
        _ => return None,
    };
    let RegionGeometry::Circle {
        center_angstrom: circle_center,
        radius_angstrom,
    } = &circle_region.geometry
    else {
        return None;
    };
    let RegionGeometry::Rectangle {
        center_angstrom: rectangle_center,
        size_angstrom,
        rotate_degrees,
    } = &rectangle_region.geometry
    else {
        return None;
    };
    if rotate_degrees.rem_euclid(180.0).abs() <= 1.0e-6 {
        return None;
    }
    if *radius_angstrom <= 0.0 {
        return exact_single_region_area(rectangle_region, bounds);
    }
    let circle_bounds = region_bounds(circle_region)?;
    let rectangle_points = convex_polygon_for_region(rectangle_region)?;
    if clipped_bounds(circle_bounds, bounds)? != circle_bounds
        || !polygon_within_bounds(&rectangle_points, bounds)
    {
        return None;
    }

    let local_circle_center =
        inverse_rotated_xy(*circle_center, *rectangle_center, *rotate_degrees);
    let local_rectangle = LayoutBounds {
        xmin: -size_angstrom[0] * 0.5,
        xmax: size_angstrom[0] * 0.5,
        ymin: -size_angstrom[1] * 0.5,
        ymax: size_angstrom[1] * 0.5,
    };
    let circle = CircleRegion {
        center: local_circle_center,
        radius: *radius_angstrom,
    };
    let circle_area = std::f32::consts::PI * radius_angstrom.powi(2);
    let rectangle_area = size_angstrom[0] * size_angstrom[1];
    let overlap_area = circle_rectangle_intersection_area(circle, local_rectangle);
    Some(circle_area + rectangle_area - overlap_area)
}

pub(super) fn exact_circle_convex_polygon_region_union_area(
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
    let polygon = convex_polygon_for_region(polygon_region)?;
    let clipped_polygon = convex_polygon_intersection(&polygon, &layout_bounds_polygon(bounds));
    let circle = CircleRegion {
        center: *center_angstrom,
        radius: *radius_angstrom,
    };
    let circle_area = circle_rectangle_intersection_area(circle, bounds);
    if clipped_polygon.len() < 3 {
        return Some(circle_area);
    }
    let polygon_area = polygon_area(&clipped_polygon);
    let overlap_area = circle_polygon_intersection_area(circle, &clipped_polygon);
    Some((circle_area + polygon_area - overlap_area).max(0.0))
}

pub(super) fn exact_circle_convex_polygons_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 3 || regions.len() > 13 {
        return None;
    }
    let mut circle = None;
    let mut polygons = Vec::with_capacity(regions.len().saturating_sub(1));
    let bounds_polygon = layout_bounds_polygon(bounds);
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
    let circle = circle?;
    if polygons.is_empty() {
        return Some(circle_rectangle_intersection_area(circle, bounds));
    }
    let circle_area = circle_rectangle_intersection_area(circle, bounds);
    let polygon_area = exact_convex_polygon_union_area_from_polygons(&polygons)?;
    let overlap_area = circle_convex_polygon_union_intersection_area(circle, &polygons)?;
    Some((circle_area + polygon_area - overlap_area).max(0.0))
}

pub(super) fn exact_disjoint_circles_convex_polygons_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 4 || regions.len() > 24 {
        return None;
    }
    let mut circles: Vec<(CircleRegion, &LeafletRegion, (f32, f32, f32, f32))> =
        Vec::with_capacity(regions.len());
    let mut polygons = Vec::with_capacity(regions.len());
    let bounds_polygon = layout_bounds_polygon(bounds);
    for region in regions {
        match &region.geometry {
            RegionGeometry::Circle {
                center_angstrom,
                radius_angstrom,
            } => {
                if *radius_angstrom <= 0.0 {
                    return None;
                }
                let region_bounds = region_bounds(region)
                    .and_then(|candidate_bounds| clipped_bounds(candidate_bounds, bounds))?;
                for (_, existing_region, existing_bounds) in &circles {
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
                circles.push((
                    CircleRegion {
                        center: *center_angstrom,
                        radius: *radius_angstrom,
                    },
                    *region,
                    region_bounds,
                ));
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
    if circles.len() < 2 || polygons.len() < 2 {
        return None;
    }

    let circle_area = circles
        .iter()
        .map(|(circle, _, _)| circle_rectangle_intersection_area(*circle, bounds))
        .sum::<f32>();
    let polygon_area = exact_convex_polygon_union_area_from_polygons(&polygons)?;
    let mut overlap_area = 0.0f32;
    for (circle, _, _) in circles {
        overlap_area += circle_convex_polygon_union_intersection_area(circle, &polygons)?;
    }
    Some((circle_area + polygon_area - overlap_area).max(0.0))
}

pub(super) fn exact_convex_polygon_union_area_from_polygons(
    polygons: &[Vec<[f32; 2]>],
) -> Option<f32> {
    if polygons.len() > 12 {
        return None;
    }
    let total_masks = 1usize.checked_shl(polygons.len() as u32)?;
    let mut area = 0.0f32;
    for mask in 1usize..total_masks {
        let mut intersection: Option<Vec<[f32; 2]>> = None;
        let mut bits = 0usize;
        for (idx, polygon) in polygons.iter().enumerate() {
            if mask & (1usize << idx) == 0 {
                continue;
            }
            bits += 1;
            intersection = Some(match intersection {
                Some(current) => convex_polygon_intersection(&current, polygon),
                None => polygon.clone(),
            });
            if intersection.as_ref().is_none_or(|points| points.len() < 3) {
                break;
            }
        }
        let Some(points) = intersection else {
            continue;
        };
        if points.len() < 3 {
            continue;
        }
        let signed = polygon_area(&points);
        if bits % 2 == 1 {
            area += signed;
        } else {
            area -= signed;
        }
    }
    Some(area.max(0.0))
}

pub(super) fn circle_convex_polygon_union_intersection_area(
    circle: CircleRegion,
    polygons: &[Vec<[f32; 2]>],
) -> Option<f32> {
    if polygons.len() > 12 {
        return None;
    }
    let total_masks = 1usize.checked_shl(polygons.len() as u32)?;
    let mut area = 0.0f32;
    for mask in 1usize..total_masks {
        let mut intersection: Option<Vec<[f32; 2]>> = None;
        let mut bits = 0usize;
        for (idx, polygon) in polygons.iter().enumerate() {
            if mask & (1usize << idx) == 0 {
                continue;
            }
            bits += 1;
            intersection = Some(match intersection {
                Some(current) => convex_polygon_intersection(&current, polygon),
                None => polygon.clone(),
            });
            if intersection.as_ref().is_none_or(|points| points.len() < 3) {
                break;
            }
        }
        let Some(points) = intersection else {
            continue;
        };
        if points.len() < 3 {
            continue;
        }
        let term = circle_polygon_intersection_area(circle, &points);
        if bits % 2 == 1 {
            area += term;
        } else {
            area -= term;
        }
    }
    Some(area.max(0.0))
}
