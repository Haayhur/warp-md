use super::*;

pub(super) fn grid_region_union_area(
    regions: &[&LeafletRegion],
    xmin: f32,
    xmax: f32,
    ymin: f32,
    ymax: f32,
    spacing: f32,
) -> f32 {
    let nx = ((xmax - xmin) / spacing).ceil() as usize;
    let ny = ((ymax - ymin) / spacing).ceil() as usize;
    let mut covered = 0usize;
    for ix in 0..nx {
        let x = xmin + (ix as f32 + 0.5) * spacing;
        if x > xmax {
            continue;
        }
        for iy in 0..ny {
            let y = ymin + (iy as f32 + 0.5) * spacing;
            if y > ymax {
                continue;
            }
            if regions
                .iter()
                .any(|region| region_contains_point(region, [x, y]))
            {
                covered += 1;
            }
        }
    }
    covered as f32 * spacing * spacing
}

pub(super) fn exact_axis_aligned_rectangle_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> f32 {
    let rectangles = regions
        .iter()
        .filter_map(|region| clipped_axis_aligned_rectangle_bounds(region, bounds))
        .collect::<Vec<_>>();
    exact_axis_aligned_rectangle_bounds_union_area(&rectangles)
}

pub(super) fn exact_axis_aligned_rectangle_bounds_union_area(
    rectangles: &[(f32, f32, f32, f32)],
) -> f32 {
    if rectangles.is_empty() {
        return 0.0;
    }
    let mut xs = rectangles
        .iter()
        .flat_map(|(xmin, xmax, _, _)| [*xmin, *xmax])
        .collect::<Vec<_>>();
    xs.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    xs.dedup_by(|left, right| (*left - *right).abs() < 1.0e-6);
    let mut area = 0.0f32;
    for pair in xs.windows(2) {
        let x0 = pair[0];
        let x1 = pair[1];
        if x1 <= x0 {
            continue;
        }
        let x_mid = (x0 + x1) * 0.5;
        let mut y_intervals = rectangles
            .iter()
            .filter_map(|(xmin, xmax, ymin, ymax)| {
                (*xmin <= x_mid && x_mid <= *xmax).then_some((*ymin, *ymax))
            })
            .collect::<Vec<_>>();
        if y_intervals.is_empty() {
            continue;
        }
        y_intervals.sort_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left.1
                        .partial_cmp(&right.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        let mut covered_y = 0.0f32;
        let mut current = y_intervals[0];
        for interval in y_intervals.into_iter().skip(1) {
            if interval.0 <= current.1 {
                current.1 = current.1.max(interval.1);
            } else {
                covered_y += current.1 - current.0;
                current = interval;
            }
        }
        covered_y += current.1 - current.0;
        area += (x1 - x0) * covered_y;
    }
    area
}

pub(super) fn exact_convex_polygon_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() > 12 {
        return None;
    }
    let bounds_polygon = layout_bounds_polygon(bounds);
    let mut polygons = Vec::with_capacity(regions.len());
    for region in regions {
        let polygon = convex_polygon_for_region(region)?;
        let clipped = convex_polygon_intersection(&polygon, &bounds_polygon);
        if clipped.len() >= 3 && polygon_area(&clipped) > 1.0e-5 {
            polygons.push(clipped);
        }
    }
    if polygons.is_empty() {
        return Some(0.0);
    }
    let mut area = 0.0f32;
    let total_masks = 1usize.checked_shl(polygons.len() as u32)?;
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

pub(super) fn convex_polygon_for_region(region: &LeafletRegion) -> Option<Vec<[f32; 2]>> {
    match &region.geometry {
        RegionGeometry::Rectangle {
            center_angstrom,
            size_angstrom,
            rotate_degrees,
        } => {
            let half_x = size_angstrom[0] * 0.5;
            let half_y = size_angstrom[1] * 0.5;
            let theta = rotate_degrees.to_radians();
            let cos_t = theta.cos();
            let sin_t = theta.sin();
            let mut points = [
                [-half_x, -half_y],
                [half_x, -half_y],
                [half_x, half_y],
                [-half_x, half_y],
            ]
            .into_iter()
            .map(|point| {
                [
                    center_angstrom[0] + point[0] * cos_t - point[1] * sin_t,
                    center_angstrom[1] + point[0] * sin_t + point[1] * cos_t,
                ]
            })
            .collect::<Vec<_>>();
            ensure_ccw_polygon(&mut points);
            Some(points)
        }
        RegionGeometry::Polygon { .. } => {
            let mut points = transformed_polygon_points(region);
            if points.len() < 3 || !polygon_is_convex(&points) {
                return None;
            }
            ensure_ccw_polygon(&mut points);
            Some(points)
        }
        RegionGeometry::Circle { .. } | RegionGeometry::Ellipse { .. } => None,
    }
}

pub(super) fn exact_disjoint_simple_polygon_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    let mut polygons = Vec::with_capacity(regions.len());
    for region in regions {
        let RegionGeometry::Polygon { .. } = &region.geometry else {
            return None;
        };
        let mut points = transformed_polygon_points(region);
        if points.len() < 3 || !polygon_within_bounds(&points, bounds) {
            return None;
        }
        ensure_ccw_polygon(&mut points);
        let polygon_bounds = polygon_bounds(&points)?;
        if polygons.iter().any(|(_, existing_bounds)| {
            axis_aligned_bounds_overlap(*existing_bounds, polygon_bounds)
        }) {
            return None;
        }
        polygons.push((points, polygon_bounds));
    }
    Some(
        polygons
            .iter()
            .map(|(points, _)| polygon_area(points))
            .sum::<f32>(),
    )
}

pub(super) fn exact_simple_polygon_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() > 24 {
        return None;
    }
    let mut polygons = Vec::with_capacity(regions.len());
    for region in regions {
        let RegionGeometry::Polygon { .. } = &region.geometry else {
            return None;
        };
        let mut points = transformed_polygon_points(region);
        if points.len() < 3 || polygon_has_self_intersections(&points) {
            return None;
        }
        ensure_ccw_polygon(&mut points);
        let clipped_bounds = clipped_bounds(polygon_bounds(&points)?, bounds)?;
        polygons.push((points, clipped_bounds));
    }
    exact_simple_polygon_union_area_from_prepared(polygons, bounds)
}

pub(super) fn exact_rectangle_simple_polygon_region_union_area(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> Option<f32> {
    if regions.len() < 2 || regions.len() > 24 {
        return None;
    }
    let mut has_rectangle = false;
    let mut has_polygon = false;
    let mut polygons = Vec::with_capacity(regions.len());
    for region in regions {
        let mut points = match &region.geometry {
            RegionGeometry::Rectangle { .. } => {
                has_rectangle = true;
                convex_polygon_for_region(region)?
            }
            RegionGeometry::Polygon { .. } => {
                has_polygon = true;
                transformed_polygon_points(region)
            }
            _ => return None,
        };
        if points.len() < 3 || polygon_has_self_intersections(&points) {
            return None;
        }
        ensure_ccw_polygon(&mut points);
        let clipped_bounds = clipped_bounds(polygon_bounds(&points)?, bounds)?;
        polygons.push((points, clipped_bounds));
    }
    if !has_rectangle || !has_polygon {
        return None;
    }
    exact_simple_polygon_union_area_from_prepared(polygons, bounds)
}

pub(super) fn exact_simple_polygon_union_area_from_polygons(
    raw_polygons: &[Vec<[f32; 2]>],
    bounds: LayoutBounds,
) -> Option<f32> {
    if raw_polygons.len() > 24 {
        return None;
    }
    let mut polygons = Vec::with_capacity(raw_polygons.len());
    for raw_polygon in raw_polygons {
        if raw_polygon.len() < 3 || polygon_has_self_intersections(raw_polygon) {
            return None;
        }
        let mut points = raw_polygon.clone();
        ensure_ccw_polygon(&mut points);
        let clipped_bounds = clipped_bounds(polygon_bounds(&points)?, bounds)?;
        polygons.push((points, clipped_bounds));
    }
    exact_simple_polygon_union_area_from_prepared(polygons, bounds)
}

pub(super) fn exact_simple_polygon_union_area_from_prepared(
    polygons: Vec<(Vec<[f32; 2]>, (f32, f32, f32, f32))>,
    bounds: LayoutBounds,
) -> Option<f32> {
    if polygons.is_empty() {
        return Some(0.0);
    }

    let mut xs = vec![bounds.xmin, bounds.xmax];
    for (polygon, clipped_bounds) in &polygons {
        xs.push(clipped_bounds.0);
        xs.push(clipped_bounds.1);
        for point in polygon {
            if point[0] > bounds.xmin + 1.0e-6 && point[0] < bounds.xmax - 1.0e-6 {
                xs.push(point[0]);
            }
        }
    }
    for left_idx in 0..polygons.len() {
        for right_idx in (left_idx + 1)..polygons.len() {
            add_polygon_edge_intersection_xs(
                &mut xs,
                &polygons[left_idx].0,
                &polygons[right_idx].0,
                bounds,
            );
        }
    }
    xs.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    xs.dedup_by(|left, right| (*left - *right).abs() < 1.0e-5);

    let mut area = 0.0f64;
    for pair in xs.windows(2) {
        let x0 = pair[0].max(bounds.xmin);
        let x1 = pair[1].min(bounds.xmax);
        if x1 <= x0 + 1.0e-6 {
            continue;
        }
        let width = x1 - x0;
        let eps = (width * 1.0e-5).clamp(1.0e-6, 1.0e-3);
        let h0 = simple_polygon_union_height_at_x(&polygons, x0 + eps, bounds);
        let h1 = simple_polygon_union_height_at_x(&polygons, x1 - eps, bounds);
        area += width as f64 * (h0 as f64 + h1 as f64) * 0.5;
    }
    Some(area.max(0.0) as f32)
}

pub(super) fn add_polygon_edge_intersection_xs(
    xs: &mut Vec<f32>,
    left: &[[f32; 2]],
    right: &[[f32; 2]],
    bounds: LayoutBounds,
) {
    for left_idx in 0..left.len() {
        let a0 = left[left_idx];
        let a1 = left[(left_idx + 1) % left.len()];
        for right_idx in 0..right.len() {
            let b0 = right[right_idx];
            let b1 = right[(right_idx + 1) % right.len()];
            if let Some(point) = proper_segment_intersection_point(a0, a1, b0, b1) {
                if point[0] > bounds.xmin + 1.0e-6
                    && point[0] < bounds.xmax - 1.0e-6
                    && point[1] >= bounds.ymin - 1.0e-6
                    && point[1] <= bounds.ymax + 1.0e-6
                {
                    xs.push(point[0]);
                }
            }
        }
    }
}

pub(super) fn proper_segment_intersection_point(
    a0: [f32; 2],
    a1: [f32; 2],
    b0: [f32; 2],
    b1: [f32; 2],
) -> Option<[f32; 2]> {
    if !segments_intersect(a0, a1, b0, b1) {
        return None;
    }
    let r = [a1[0] - a0[0], a1[1] - a0[1]];
    let s = [b1[0] - b0[0], b1[1] - b0[1]];
    let denom = r[0] * s[1] - r[1] * s[0];
    if denom.abs() <= 1.0e-8 {
        return None;
    }
    let ba = [b0[0] - a0[0], b0[1] - a0[1]];
    let t = (ba[0] * s[1] - ba[1] * s[0]) / denom;
    Some([a0[0] + t * r[0], a0[1] + t * r[1]])
}

pub(super) fn simple_polygon_union_height_at_x(
    polygons: &[(Vec<[f32; 2]>, (f32, f32, f32, f32))],
    x: f32,
    bounds: LayoutBounds,
) -> f32 {
    if x <= bounds.xmin || x >= bounds.xmax {
        return 0.0;
    }
    let mut intervals = Vec::new();
    for (polygon, clipped_bounds) in polygons {
        if x < clipped_bounds.0 - 1.0e-6 || x > clipped_bounds.1 + 1.0e-6 {
            continue;
        }
        intervals.extend(simple_polygon_y_intervals_at_x(polygon, x, bounds));
    }
    merged_interval_length(intervals)
}

pub(super) fn simple_polygon_y_intervals_at_x(
    polygon: &[[f32; 2]],
    x: f32,
    bounds: LayoutBounds,
) -> Vec<(f32, f32)> {
    let mut ys = Vec::new();
    for idx in 0..polygon.len() {
        let a = polygon[idx];
        let b = polygon[(idx + 1) % polygon.len()];
        if (a[0] - b[0]).abs() <= 1.0e-8 {
            continue;
        }
        let xmin = a[0].min(b[0]);
        let xmax = a[0].max(b[0]);
        if x <= xmin || x > xmax {
            continue;
        }
        let t = (x - a[0]) / (b[0] - a[0]);
        ys.push(a[1] + t * (b[1] - a[1]));
    }
    ys.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    ys.dedup_by(|left, right| (*left - *right).abs() < 1.0e-5);

    let mut intervals = Vec::new();
    for pair in ys.chunks_exact(2) {
        let ymin = pair[0].max(bounds.ymin);
        let ymax = pair[1].min(bounds.ymax);
        if ymax > ymin + 1.0e-6 {
            intervals.push((ymin, ymax));
        }
    }
    intervals
}

pub(super) fn merged_interval_length(mut intervals: Vec<(f32, f32)>) -> f32 {
    if intervals.is_empty() {
        return 0.0;
    }
    intervals.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });
    let mut current = intervals[0];
    let mut length = 0.0f32;
    for interval in intervals.into_iter().skip(1) {
        if interval.0 <= current.1 + 1.0e-6 {
            current.1 = current.1.max(interval.1);
        } else {
            length += current.1 - current.0;
            current = interval;
        }
    }
    length + current.1 - current.0
}

pub(super) fn polygon_within_bounds(points: &[[f32; 2]], bounds: LayoutBounds) -> bool {
    points.iter().all(|point| {
        point[0] >= bounds.xmin - 1.0e-6
            && point[0] <= bounds.xmax + 1.0e-6
            && point[1] >= bounds.ymin - 1.0e-6
            && point[1] <= bounds.ymax + 1.0e-6
    })
}
