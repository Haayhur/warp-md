use super::*;

pub(super) fn exact_component_mixed_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 3 {
        return None;
    }
    let mut bounded = Vec::with_capacity(regions.len());
    for region in regions {
        let region_bounds =
            region_bounds(region).and_then(|region_bounds| clipped_bounds(region_bounds, bounds));
        if let Some(region_bounds) = region_bounds {
            bounded.push((*region, region_bounds));
        }
    }
    if bounded.len() < 3 {
        return None;
    }

    let mut visited = vec![false; bounded.len()];
    let mut components: Vec<Vec<&LeafletRegion>> = Vec::new();
    for start in 0..bounded.len() {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        visited[start] = true;
        let mut component = Vec::new();
        while let Some(idx) = stack.pop() {
            component.push(bounded[idx].0);
            for next in 0..bounded.len() {
                if visited[next] {
                    continue;
                }
                let pair_is_disjoint = regions_are_exactly_disjoint(
                    bounded[idx].0,
                    bounded[idx].1,
                    bounded[next].0,
                    bounded[next].1,
                    bounds,
                )
                .unwrap_or(false);
                if !pair_is_disjoint {
                    visited[next] = true;
                    stack.push(next);
                }
            }
        }
        components.push(component);
    }
    if components.len() <= 1 {
        return None;
    }

    let mut total = 0.0f32;
    for component in components {
        total += exact_region_union_area_without_grid(&component, bounds)?;
    }
    Some(total)
}

fn exact_region_union_area_without_grid(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.is_empty() {
        return Some(0.0);
    }
    if regions.len() == 1 {
        return exact_single_region_area(regions[0], bounds);
    }
    if regions
        .iter()
        .all(|region| axis_aligned_rectangle_bounds(region).is_some())
    {
        return Some(exact_axis_aligned_rectangle_union_area(regions, bounds));
    }
    exact_convex_polygon_region_union_area(regions, bounds)
        .or_else(|| exact_disjoint_simple_polygon_region_union_area(regions, bounds))
        .or_else(|| exact_simple_polygon_region_union_area(regions, bounds))
        .or_else(|| exact_rectangle_simple_polygon_region_union_area(regions, bounds))
        .or_else(|| exact_circle_region_union_area(regions, bounds))
        .or_else(|| exact_similar_oriented_ellipse_region_union_area(regions, bounds))
        .or_else(|| exact_disjoint_ellipse_region_union_area(regions, bounds))
        .or_else(|| exact_axis_aligned_ellipse_pair_region_union_area(regions, bounds))
        .or_else(|| exact_rotated_ellipse_pair_region_union_area(regions, bounds))
        .or_else(|| exact_circle_axis_aligned_rectangle_region_union_area(regions, bounds))
        .or_else(|| exact_circles_axis_aligned_rectangle_region_union_area(regions, bounds))
        .or_else(|| exact_circle_axis_aligned_rectangles_region_union_area(regions, bounds))
        .or_else(|| exact_circles_axis_aligned_rectangles_region_union_area(regions, bounds))
        .or_else(|| exact_circle_rotated_rectangle_region_union_area(regions, bounds))
        .or_else(|| exact_circle_convex_polygon_region_union_area(regions, bounds))
        .or_else(|| exact_circle_convex_polygons_region_union_area(regions, bounds))
        .or_else(|| exact_disjoint_circles_convex_polygons_region_union_area(regions, bounds))
        .or_else(|| exact_circle_simple_polygons_region_union_area(regions, bounds))
        .or_else(|| exact_circle_simple_polygon_region_union_area(regions, bounds))
        .or_else(|| exact_ellipse_convex_polygon_region_union_area(regions, bounds))
        .or_else(|| exact_ellipse_simple_polygons_region_union_area(regions, bounds))
        .or_else(|| exact_ellipse_simple_polygon_region_union_area(regions, bounds))
        .or_else(|| {
            exact_axis_aligned_ellipse_axis_aligned_rectangle_region_union_area(regions, bounds)
        })
        .or_else(|| exact_rotated_ellipse_axis_aligned_rectangle_region_union_area(regions, bounds))
        .or_else(|| exact_ellipse_axis_aligned_rectangles_region_union_area(regions, bounds))
        .or_else(|| {
            exact_disjoint_ellipses_axis_aligned_rectangles_region_union_area(regions, bounds)
        })
        .or_else(|| exact_ellipse_convex_polygons_region_union_area(regions, bounds))
        .or_else(|| exact_oriented_ellipse_rotated_rectangle_region_union_area(regions, bounds))
        .or_else(|| exact_circle_oriented_ellipse_region_union_area(regions, bounds))
        .or_else(|| exact_clipped_circle_rotated_ellipse_region_union_area(regions, bounds))
        .or_else(|| exact_circle_disjoint_ellipses_region_union_area(regions, bounds))
        .or_else(|| exact_ellipse_disjoint_circles_region_union_area(regions, bounds))
        .or_else(|| {
            exact_circle_convex_polygons_disjoint_mixed_shapes_region_union_area(regions, bounds)
        })
        .or_else(|| {
            exact_ellipse_convex_polygons_disjoint_mixed_shapes_region_union_area(regions, bounds)
        })
        .or_else(|| {
            exact_circle_convex_polygons_disjoint_ellipses_region_union_area(regions, bounds)
        })
        .or_else(|| {
            exact_ellipse_convex_polygons_disjoint_circles_region_union_area(regions, bounds)
        })
        .or_else(|| exact_circle_disjoint_mixed_shapes_region_union_area(regions, bounds))
        .or_else(|| exact_ellipse_disjoint_mixed_shapes_region_union_area(regions, bounds))
        .or_else(|| exact_rectangle_disjoint_mixed_shapes_region_union_area(regions, bounds))
        .or_else(|| exact_convex_polygon_disjoint_mixed_shapes_region_union_area(regions, bounds))
        .or_else(|| exact_disjoint_mixed_region_union_area(regions, bounds))
}

pub(super) fn exact_disjoint_mixed_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    let mut exact_regions = Vec::with_capacity(regions.len());
    for region in regions {
        let Some(region_bounds) =
            region_bounds(region).and_then(|region_bounds| clipped_bounds(region_bounds, bounds))
        else {
            continue;
        };
        for (existing_region, existing_bounds, _) in &exact_regions {
            if !regions_are_exactly_disjoint(
                *existing_region,
                *existing_bounds,
                region,
                region_bounds,
                bounds,
            )? {
                return None;
            }
        }
        let area = exact_single_region_area(region, bounds)?;
        exact_regions.push((*region, region_bounds, area));
    }
    if exact_regions.is_empty() {
        return Some(0.0);
    }
    Some(exact_regions.iter().map(|(_, _, area)| *area).sum())
}

pub(super) fn regions_are_exactly_disjoint(
    left: &LeafletRegion,
    left_bounds: (f32, f32, f32, f32),
    right: &LeafletRegion,
    right_bounds: (f32, f32, f32, f32),
    bounds: LayoutBounds,
) -> Option<bool> {
    if !axis_aligned_bounds_overlap(left_bounds, right_bounds) {
        return Some(true);
    }
    let left_area = exact_single_region_area(left, bounds)?;
    let right_area = exact_single_region_area(right, bounds)?;
    let union_area = exact_pair_region_union_area_without_disjoint(left, right, bounds)?;
    let overlap = (left_area + right_area - union_area).max(0.0);
    Some(overlap <= (left_area + right_area).max(1.0) * 1.0e-5)
}

pub(super) fn exact_pair_region_union_area_without_disjoint(
    left: &LeafletRegion,
    right: &LeafletRegion,
    bounds: LayoutBounds,
) -> Option<f32> {
    let regions = [left, right];
    exact_convex_polygon_region_union_area(&regions, bounds)
        .or_else(|| exact_simple_polygon_region_union_area(&regions, bounds))
        .or_else(|| exact_rectangle_simple_polygon_region_union_area(&regions, bounds))
        .or_else(|| exact_circle_region_union_area(&regions, bounds))
        .or_else(|| exact_similar_oriented_ellipse_region_union_area(&regions, bounds))
        .or_else(|| exact_disjoint_ellipse_region_union_area(&regions, bounds))
        .or_else(|| exact_axis_aligned_ellipse_pair_region_union_area(&regions, bounds))
        .or_else(|| exact_rotated_ellipse_pair_region_union_area(&regions, bounds))
        .or_else(|| exact_circle_axis_aligned_rectangle_region_union_area(&regions, bounds))
        .or_else(|| exact_circles_axis_aligned_rectangle_region_union_area(&regions, bounds))
        .or_else(|| exact_circle_rotated_rectangle_region_union_area(&regions, bounds))
        .or_else(|| exact_circle_convex_polygon_region_union_area(&regions, bounds))
        .or_else(|| exact_circle_simple_polygon_region_union_area(&regions, bounds))
        .or_else(|| exact_ellipse_convex_polygon_region_union_area(&regions, bounds))
        .or_else(|| exact_ellipse_simple_polygon_region_union_area(&regions, bounds))
        .or_else(|| {
            exact_axis_aligned_ellipse_axis_aligned_rectangle_region_union_area(&regions, bounds)
        })
        .or_else(|| {
            exact_rotated_ellipse_axis_aligned_rectangle_region_union_area(&regions, bounds)
        })
        .or_else(|| exact_oriented_ellipse_rotated_rectangle_region_union_area(&regions, bounds))
        .or_else(|| exact_circle_oriented_ellipse_region_union_area(&regions, bounds))
        .or_else(|| exact_clipped_circle_rotated_ellipse_region_union_area(&regions, bounds))
        .or_else(|| exact_circle_disjoint_ellipses_region_union_area(&regions, bounds))
        .or_else(|| exact_ellipse_disjoint_circles_region_union_area(&regions, bounds))
        .or_else(|| exact_circle_disjoint_mixed_shapes_region_union_area(&regions, bounds))
}
