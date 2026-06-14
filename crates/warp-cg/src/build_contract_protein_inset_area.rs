use super::*;

pub(super) fn exact_convex_multipolygon_inset_union_area(
    polygons: &[Vec<[f32; 2]>],
    inset_angstrom: f32,
) -> Option<f32> {
    if polygons.len() > 24 || inset_angstrom < 0.0 {
        return None;
    }
    let mut inset_polygons = Vec::with_capacity(polygons.len());
    for polygon in polygons {
        if polygon.len() < 3 || !polygon_is_convex(polygon) {
            return None;
        }
        let inset = convex_polygon_inset_polygon(polygon, inset_angstrom)?;
        if inset.len() >= 3 && polygon_area(&inset) > 1.0e-6 {
            inset_polygons.push(inset);
        }
    }
    if inset_polygons.is_empty() {
        return Some(0.0);
    }
    let bounds = polygon_collection_layout_bounds(&inset_polygons)?;
    exact_simple_polygon_union_area_from_polygons(&inset_polygons, bounds)
}

pub(super) fn multipolygon_components_overlap(polygons: &[Vec<[f32; 2]>]) -> bool {
    for left_idx in 0..polygons.len() {
        for right_idx in (left_idx + 1)..polygons.len() {
            if polygon_distance(&polygons[left_idx], &polygons[right_idx]) <= 1.0e-5 {
                return true;
            }
        }
    }
    false
}

pub(super) fn multipolygon_area_with_inset_union(
    polygons: &[Vec<[f32; 2]>],
    inset_angstrom: f32,
) -> f32 {
    let Some(bounds) = polygon_collection_layout_bounds(polygons) else {
        return 0.0;
    };
    let spacing = 0.5f32;
    let nx = ((bounds.xmax - bounds.xmin) / spacing).ceil() as usize;
    let ny = ((bounds.ymax - bounds.ymin) / spacing).ceil() as usize;
    let mut covered = 0usize;
    for ix in 0..nx {
        let x = bounds.xmin + (ix as f32 + 0.5) * spacing;
        for iy in 0..ny {
            let y = bounds.ymin + (iy as f32 + 0.5) * spacing;
            let point = [x, y];
            if polygons.iter().any(|polygon| {
                point_in_polygon(point, polygon)
                    && polygon_boundary_distance(point, polygon) >= inset_angstrom
            }) {
                covered += 1;
            }
        }
    }
    covered as f32 * spacing.powi(2)
}

pub(super) fn polygon_collection_layout_bounds(polygons: &[Vec<[f32; 2]>]) -> Option<LayoutBounds> {
    let mut xmin = f32::INFINITY;
    let mut xmax = f32::NEG_INFINITY;
    let mut ymin = f32::INFINITY;
    let mut ymax = f32::NEG_INFINITY;
    for polygon in polygons {
        let bounds = polygon_bounds(polygon)?;
        xmin = xmin.min(bounds.0);
        xmax = xmax.max(bounds.1);
        ymin = ymin.min(bounds.2);
        ymax = ymax.max(bounds.3);
    }
    (xmin.is_finite() && xmin < xmax && ymin < ymax).then_some(LayoutBounds {
        xmin,
        xmax,
        ymin,
        ymax,
    })
}

pub(super) fn polygon_boundary_area_estimate(
    points: &[[f32; 2]],
    inset_angstrom: f32,
) -> (f32, bool) {
    if inset_angstrom <= 0.0 {
        (polygon_area(points), true)
    } else if let Some(area) =
        exact_axis_aligned_rectangle_polygon_inset_area(points, inset_angstrom)
    {
        (area, true)
    } else if let Some(area) = exact_convex_polygon_inset_area(points, inset_angstrom) {
        (area, true)
    } else if let Some(area) =
        exact_simple_polygon_inset_area_before_topology_event(points, inset_angstrom)
    {
        (area, true)
    } else {
        (polygon_area_with_inset(points, inset_angstrom), false)
    }
}

pub(super) fn exact_axis_aligned_rectangle_nested_inset_area(
    outer: &[[f32; 2]],
    holes: &[Vec<[f32; 2]>],
    inset_angstrom: f32,
) -> Option<f32> {
    let outer_area = exact_axis_aligned_rectangle_polygon_inset_area(outer, inset_angstrom)?;
    if holes.is_empty() || outer_area <= 0.0 {
        return Some(outer_area);
    }
    let (outer_xmin, outer_xmax, outer_ymin, outer_ymax) = polygon_bounds(outer)?;
    let clipped_outer = (
        outer_xmin + inset_angstrom,
        outer_xmax - inset_angstrom,
        outer_ymin + inset_angstrom,
        outer_ymax - inset_angstrom,
    );
    if clipped_outer.0 >= clipped_outer.1 || clipped_outer.2 >= clipped_outer.3 {
        return Some(0.0);
    }
    let mut hole_bounds = Vec::with_capacity(holes.len());
    for hole in holes {
        let bounds = axis_aligned_rectangle_polygon_bounds(hole)?;
        if rectangle_containment_gap(clipped_outer, bounds) < inset_angstrom - 1.0e-5 {
            return None;
        }
        for existing in &hole_bounds {
            if rectangle_bounds_distance(*existing, bounds) < 2.0 * inset_angstrom - 1.0e-5 {
                return None;
            }
        }
        hole_bounds.push(bounds);
    }
    let expanded_hole_area = hole_bounds
        .iter()
        .map(|bounds| rounded_rectangle_dilation_area(*bounds, inset_angstrom))
        .sum::<f32>();
    Some((outer_area - expanded_hole_area).max(0.0))
}

pub(super) fn exact_convex_nested_inset_area(
    outer: &[[f32; 2]],
    holes: &[Vec<[f32; 2]>],
    inset_angstrom: f32,
) -> Option<f32> {
    if outer.len() < 3
        || inset_angstrom < 0.0
        || !polygon_is_convex(outer)
        || holes
            .iter()
            .any(|hole| hole.len() < 3 || !polygon_is_convex(hole))
    {
        return None;
    }
    let mut inner_outer = outer.to_vec();
    ensure_ccw_polygon(&mut inner_outer);
    if inset_angstrom > 1.0e-6 {
        inner_outer = convex_polygon_inset_polygon(&inner_outer, inset_angstrom)?;
    }
    if inner_outer.len() < 3 {
        return Some(0.0);
    }
    let outer_area = polygon_area(&inner_outer);
    if holes.is_empty() || outer_area <= 1.0e-6 {
        return Some(outer_area);
    }
    let mut excluded_area = 0.0f32;
    for (idx, hole) in holes.iter().enumerate() {
        let mut hole_points = hole.clone();
        ensure_ccw_polygon(&mut hole_points);
        if !hole_points
            .iter()
            .all(|point| point_in_polygon(*point, &inner_outer))
        {
            return None;
        }
        if inset_angstrom > 1.0e-6
            && !hole_points
                .iter()
                .all(|point| polygon_boundary_distance(*point, &inner_outer) >= inset_angstrom)
        {
            return None;
        }
        for other in holes.iter().skip(idx + 1) {
            if polygon_distance(&hole_points, other) < 2.0 * inset_angstrom - 1.0e-5 {
                return None;
            }
        }
        excluded_area += polygon_area(&hole_points)
            + polygon_perimeter(&hole_points) * inset_angstrom
            + std::f32::consts::PI * inset_angstrom.powi(2);
    }
    Some((outer_area - excluded_area).max(0.0))
}

pub(super) fn exact_simple_nested_inset_area_before_topology_event(
    outer: &[[f32; 2]],
    holes: &[Vec<[f32; 2]>],
    inset_angstrom: f32,
) -> Option<f32> {
    if outer.len() < 3
        || inset_angstrom < 0.0
        || polygon_has_self_intersections(outer)
        || holes
            .iter()
            .any(|hole| hole.len() < 3 || polygon_has_self_intersections(hole))
    {
        return None;
    }
    let outer_area = exact_simple_polygon_inset_area_before_topology_event(outer, inset_angstrom)?;
    if holes.is_empty() || outer_area <= 1.0e-6 {
        return Some(outer_area);
    }

    let mut oriented_outer = outer.to_vec();
    ensure_ccw_polygon(&mut oriented_outer);
    let mut oriented_holes: Vec<Vec<[f32; 2]>> = Vec::with_capacity(holes.len());
    for hole in holes {
        let mut hole_points = hole.clone();
        ensure_ccw_polygon(&mut hole_points);
        if !hole_points
            .iter()
            .all(|point| point_in_polygon(*point, &oriented_outer))
        {
            return None;
        }
        if polygon_boundary_gap(&oriented_outer, &hole_points) < 2.0 * inset_angstrom - 1.0e-5 {
            return None;
        }
        if inset_angstrom > 1.0e-6 {
            let clearance = polygon_non_adjacent_edge_clearance(&hole_points)?;
            if clearance <= 2.0 * inset_angstrom + 1.0e-5 {
                return None;
            }
        }
        for existing in &oriented_holes {
            if polygon_distance(existing, &hole_points) < 2.0 * inset_angstrom - 1.0e-5 {
                return None;
            }
        }
        oriented_holes.push(hole_points);
    }

    let excluded_area = oriented_holes
        .iter()
        .map(|hole| {
            polygon_area(hole)
                + polygon_perimeter(hole) * inset_angstrom
                + std::f32::consts::PI * inset_angstrom.powi(2)
        })
        .sum::<f32>();
    Some((outer_area - excluded_area).max(0.0))
}

pub(super) fn nested_polygon_forest_area_estimate(
    rings: &[Vec<[f32; 2]>],
    inset_angstrom: f32,
) -> (f32, bool) {
    let depths = nested_polygon_ring_depths(rings);
    if inset_angstrom <= 1.0e-6 {
        let mut area = 0.0f32;
        for (ring, depth) in rings.iter().zip(depths.iter()) {
            if depth % 2 == 0 {
                area += polygon_area(ring);
            } else {
                area -= polygon_area(ring);
            }
        }
        return (area.max(0.0), true);
    }
    if let Some(area) =
        exact_axis_aligned_rectangle_nested_forest_inset_area(rings, &depths, inset_angstrom)
    {
        return (area, true);
    }
    if let Some(area) = exact_convex_nested_forest_inset_area(rings, &depths, inset_angstrom) {
        return (area, true);
    }
    if let Some(area) =
        exact_simple_nested_forest_inset_area_before_topology_event(rings, &depths, inset_angstrom)
    {
        return (area, true);
    }

    let mut area = 0.0f32;
    let mut exact = true;
    for (ring, depth) in rings.iter().zip(depths.iter()) {
        let (ring_area, ring_exact) = if depth % 2 == 0 {
            polygon_boundary_area_estimate(ring, inset_angstrom)
        } else {
            (polygon_area(ring), false)
        };
        if depth % 2 == 0 {
            area += ring_area;
        } else {
            area -= ring_area;
        }
        exact &= ring_exact && *depth == 0;
    }
    (area.max(0.0), exact)
}

pub(super) fn exact_axis_aligned_rectangle_nested_forest_inset_area(
    rings: &[Vec<[f32; 2]>],
    depths: &[usize],
    inset_angstrom: f32,
) -> Option<f32> {
    if rings.is_empty() || rings.len() != depths.len() || inset_angstrom < 0.0 {
        return None;
    }
    let rectangles = rings
        .iter()
        .map(|ring| axis_aligned_rectangle_polygon_bounds(ring))
        .collect::<Option<Vec<_>>>()?;
    for (child_idx, child) in rectangles.iter().enumerate() {
        let child_depth = depths[child_idx];
        if child_depth == 0 {
            continue;
        }
        let parent_idx = rectangles
            .iter()
            .enumerate()
            .filter(|(idx, parent)| {
                *idx != child_idx
                    && depths[*idx] + 1 == child_depth
                    && rectangle_bounds_strictly_contains(**parent, *child)
            })
            .min_by(|(_, left), (_, right)| {
                rectangle_area(**left)
                    .partial_cmp(&rectangle_area(**right))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)?;
        let parent = rectangles[parent_idx];
        if rectangle_containment_gap(parent, *child) < 2.0 * inset_angstrom - 1.0e-5 {
            return None;
        }
    }
    for left_idx in 0..rectangles.len() {
        if depths[left_idx] % 2 == 0 {
            continue;
        }
        for right_idx in (left_idx + 1)..rectangles.len() {
            if depths[right_idx] == depths[left_idx]
                && rectangle_bounds_distance(rectangles[left_idx], rectangles[right_idx])
                    < 2.0 * inset_angstrom - 1.0e-5
            {
                return None;
            }
        }
    }

    let mut area = 0.0f32;
    for (rectangle, depth) in rectangles.iter().zip(depths.iter()) {
        let offset = if depth % 2 == 0 {
            (
                rectangle.0 + inset_angstrom,
                rectangle.1 - inset_angstrom,
                rectangle.2 + inset_angstrom,
                rectangle.3 - inset_angstrom,
            )
        } else {
            (
                rectangle.0 - inset_angstrom,
                rectangle.1 + inset_angstrom,
                rectangle.2 - inset_angstrom,
                rectangle.3 + inset_angstrom,
            )
        };
        let offset_area = if depth % 2 == 0 {
            rectangle_area(offset)
        } else {
            rounded_rectangle_dilation_area(*rectangle, inset_angstrom)
        };
        if depth % 2 == 0 {
            area += offset_area;
        } else {
            area -= offset_area;
        }
    }
    Some(area.max(0.0))
}

pub(super) fn exact_convex_nested_forest_inset_area(
    rings: &[Vec<[f32; 2]>],
    depths: &[usize],
    inset_angstrom: f32,
) -> Option<f32> {
    if rings.is_empty()
        || rings.len() != depths.len()
        || inset_angstrom < 0.0
        || rings
            .iter()
            .any(|ring| ring.len() < 3 || !polygon_is_convex(ring))
    {
        return None;
    }
    let mut oriented = rings.to_vec();
    for ring in &mut oriented {
        ensure_ccw_polygon(ring);
    }
    for (child_idx, child) in oriented.iter().enumerate() {
        let child_depth = depths[child_idx];
        if child_depth == 0 {
            continue;
        }
        let parent_idx = oriented
            .iter()
            .enumerate()
            .filter(|(idx, parent)| {
                *idx != child_idx
                    && depths[*idx] + 1 == child_depth
                    && child.iter().all(|point| point_in_polygon(*point, parent))
            })
            .min_by(|(_, left), (_, right)| {
                polygon_area(left)
                    .partial_cmp(&polygon_area(right))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)?;
        let gap = polygon_boundary_gap(&oriented[parent_idx], child);
        if gap < 2.0 * inset_angstrom - 1.0e-5 {
            return None;
        }
    }
    for left_idx in 0..oriented.len() {
        if depths[left_idx] % 2 == 0 {
            continue;
        }
        for right_idx in (left_idx + 1)..oriented.len() {
            if depths[right_idx] == depths[left_idx]
                && polygon_boundary_gap(&oriented[left_idx], &oriented[right_idx])
                    < 2.0 * inset_angstrom - 1.0e-5
            {
                return None;
            }
        }
    }

    let mut area = 0.0f32;
    for (ring, depth) in oriented.iter().zip(depths.iter()) {
        let offset_area = if depth % 2 == 0 {
            let inset = convex_polygon_inset_polygon(ring, inset_angstrom)?;
            if inset.len() < 3 {
                0.0
            } else {
                polygon_area(&inset)
            }
        } else {
            polygon_area(ring)
                + polygon_perimeter(ring) * inset_angstrom
                + std::f32::consts::PI * inset_angstrom.powi(2)
        };
        if depth % 2 == 0 {
            area += offset_area;
        } else {
            area -= offset_area;
        }
    }
    Some(area.max(0.0))
}

pub(super) fn exact_simple_nested_forest_inset_area_before_topology_event(
    rings: &[Vec<[f32; 2]>],
    depths: &[usize],
    inset_angstrom: f32,
) -> Option<f32> {
    if rings.is_empty()
        || rings.len() != depths.len()
        || inset_angstrom < 0.0
        || rings
            .iter()
            .any(|ring| ring.len() < 3 || polygon_has_self_intersections(ring))
    {
        return None;
    }
    let mut oriented = rings.to_vec();
    for ring in &mut oriented {
        ensure_ccw_polygon(ring);
        if inset_angstrom > 1.0e-6 {
            let clearance = polygon_non_adjacent_edge_clearance(ring)?;
            if clearance <= 2.0 * inset_angstrom + 1.0e-5 {
                return None;
            }
        }
    }

    for (child_idx, child) in oriented.iter().enumerate() {
        let child_depth = depths[child_idx];
        if child_depth == 0 {
            continue;
        }
        let parent_idx = oriented
            .iter()
            .enumerate()
            .filter(|(idx, parent)| {
                *idx != child_idx
                    && depths[*idx] + 1 == child_depth
                    && child.iter().all(|point| point_in_polygon(*point, parent))
            })
            .min_by(|(_, left), (_, right)| {
                polygon_area(left)
                    .partial_cmp(&polygon_area(right))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(idx, _)| idx)?;
        if polygon_boundary_gap(&oriented[parent_idx], child) < 2.0 * inset_angstrom - 1.0e-5 {
            return None;
        }
    }
    for left_idx in 0..oriented.len() {
        for right_idx in (left_idx + 1)..oriented.len() {
            if depths[left_idx] == depths[right_idx]
                && polygon_distance(&oriented[left_idx], &oriented[right_idx])
                    < 2.0 * inset_angstrom - 1.0e-5
            {
                return None;
            }
        }
    }

    let mut area = 0.0f32;
    for (ring, depth) in oriented.iter().zip(depths.iter()) {
        let offset_area = if depth % 2 == 0 {
            exact_simple_polygon_inset_area_before_topology_event(ring, inset_angstrom)?
        } else {
            polygon_area(ring)
                + polygon_perimeter(ring) * inset_angstrom
                + std::f32::consts::PI * inset_angstrom.powi(2)
        };
        if depth % 2 == 0 {
            area += offset_area;
        } else {
            area -= offset_area;
        }
    }
    Some(area.max(0.0))
}
