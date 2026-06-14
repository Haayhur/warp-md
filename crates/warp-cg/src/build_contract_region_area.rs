use super::*;

#[derive(Clone, Debug)]
pub(super) struct RegionAreaEstimate {
    pub(super) area_angstrom2: f32,
    pub(super) method: String,
    pub(super) is_exact: bool,
    pub(super) reported_error_bound_angstrom2: Option<f32>,
}

impl RegionAreaEstimate {
    pub(super) fn zero(method: &str) -> Self {
        Self {
            area_angstrom2: 0.0,
            method: method.to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: None,
        }
    }
}

pub(super) fn region_area_estimate(region: &LeafletRegion) -> RegionAreaEstimate {
    match &region.geometry {
        RegionGeometry::Circle {
            radius_angstrom, ..
        } => RegionAreaEstimate {
            area_angstrom2: region_buffer_unit_area() * radius_angstrom.powi(2),
            method: "analytic_circle_buffered_64_segment".to_string(),
            is_exact: false,
            reported_error_bound_angstrom2: None,
        },
        RegionGeometry::Ellipse {
            radius_angstrom, ..
        } => RegionAreaEstimate {
            area_angstrom2: region_buffer_unit_area() * radius_angstrom[0] * radius_angstrom[1],
            method: "analytic_ellipse_buffered_64_segment".to_string(),
            is_exact: false,
            reported_error_bound_angstrom2: None,
        },
        RegionGeometry::Rectangle { size_angstrom, .. } => RegionAreaEstimate {
            area_angstrom2: size_angstrom[0] * size_angstrom[1],
            method: "analytic_rectangle".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: None,
        },
        RegionGeometry::Polygon { .. } => RegionAreaEstimate {
            area_angstrom2: polygon_area(&transformed_polygon_points(region)),
            method: "analytic_polygon_shoelace".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: None,
        },
    }
}

fn region_buffer_unit_area() -> f32 {
    0.5 * REGION_BUFFER_SEGMENTS * (2.0 * std::f32::consts::PI / REGION_BUFFER_SEGMENTS).sin()
}

pub(super) fn region_union_area_angstrom2(
    regions: &[&LeafletRegion],
    bounds: LayoutBounds,
) -> RegionAreaEstimate {
    if regions.is_empty() {
        return RegionAreaEstimate::zero("empty_region_union");
    }
    let Some((mut xmin, mut xmax, mut ymin, mut ymax)) = region_union_bounds(regions) else {
        return RegionAreaEstimate::zero("empty_region_union_bounds");
    };
    xmin = xmin.max(bounds.xmin);
    xmax = xmax.min(bounds.xmax);
    ymin = ymin.max(bounds.ymin);
    ymax = ymax.min(bounds.ymax);
    if xmin >= xmax || ymin >= ymax {
        return RegionAreaEstimate::zero("empty_clipped_region_union");
    }

    if regions
        .iter()
        .all(|region| axis_aligned_rectangle_bounds(region).is_some())
    {
        return RegionAreaEstimate {
            area_angstrom2: exact_axis_aligned_rectangle_union_area(regions, bounds),
            method: "exact_axis_aligned_rectangle_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_convex_polygon_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_convex_polygon_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_disjoint_simple_polygon_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_disjoint_simple_polygon_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_simple_polygon_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_simple_polygon_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_rectangle_simple_polygon_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_rectangle_simple_polygon_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_circle_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circle_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_similar_oriented_ellipse_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_similar_oriented_ellipse_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_disjoint_ellipse_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_disjoint_ellipse_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_axis_aligned_ellipse_pair_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_axis_aligned_ellipse_pair_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_rotated_ellipse_pair_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_rotated_ellipse_pair_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_circle_axis_aligned_rectangle_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circle_axis_aligned_rectangle_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_circles_axis_aligned_rectangle_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circles_axis_aligned_rectangle_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_circle_axis_aligned_rectangles_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circle_axis_aligned_rectangles_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_circles_axis_aligned_rectangles_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circles_axis_aligned_rectangles_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_circle_rotated_rectangle_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circle_rotated_rectangle_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_circle_convex_polygon_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circle_convex_polygon_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_circle_convex_polygons_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circle_convex_polygons_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_disjoint_circles_convex_polygons_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_disjoint_circles_convex_polygons_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_circle_simple_polygons_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circle_simple_polygons_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_circle_simple_polygon_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circle_simple_polygon_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_ellipse_convex_polygon_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_ellipse_convex_polygon_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_ellipse_simple_polygons_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_ellipse_simple_polygons_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_ellipse_simple_polygon_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_ellipse_simple_polygon_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) =
        exact_axis_aligned_ellipse_axis_aligned_rectangle_region_union_area(regions, bounds)
    {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_axis_aligned_ellipse_axis_aligned_rectangle_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) =
        exact_rotated_ellipse_axis_aligned_rectangle_region_union_area(regions, bounds)
    {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_rotated_ellipse_axis_aligned_rectangle_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_ellipse_axis_aligned_rectangles_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_ellipse_axis_aligned_rectangles_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) =
        exact_disjoint_ellipses_axis_aligned_rectangles_region_union_area(regions, bounds)
    {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_disjoint_ellipses_axis_aligned_rectangles_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_ellipse_convex_polygons_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_ellipse_convex_polygons_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_oriented_ellipse_rotated_rectangle_region_union_area(regions, bounds)
    {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_oriented_ellipse_rotated_rectangle_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_circle_oriented_ellipse_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circle_oriented_ellipse_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_clipped_circle_rotated_ellipse_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_clipped_circle_rotated_ellipse_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_circle_disjoint_ellipses_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circle_disjoint_ellipses_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_ellipse_disjoint_circles_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_ellipse_disjoint_circles_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) =
        exact_circle_convex_polygons_disjoint_mixed_shapes_region_union_area(regions, bounds)
    {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circle_convex_polygons_disjoint_mixed_shapes_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) =
        exact_ellipse_convex_polygons_disjoint_mixed_shapes_region_union_area(regions, bounds)
    {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_ellipse_convex_polygons_disjoint_mixed_shapes_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) =
        exact_circle_convex_polygons_disjoint_ellipses_region_union_area(regions, bounds)
    {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circle_convex_polygons_disjoint_ellipses_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) =
        exact_ellipse_convex_polygons_disjoint_circles_region_union_area(regions, bounds)
    {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_ellipse_convex_polygons_disjoint_circles_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_disjoint_mixed_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_disjoint_mixed_region_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_component_mixed_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_component_mixed_region_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_circle_disjoint_mixed_shapes_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_circle_disjoint_mixed_shapes_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_ellipse_disjoint_mixed_shapes_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_ellipse_disjoint_mixed_shapes_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = exact_rectangle_disjoint_mixed_shapes_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_rectangle_disjoint_mixed_shapes_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) =
        exact_convex_polygon_disjoint_mixed_shapes_region_union_area(regions, bounds)
    {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "exact_convex_polygon_disjoint_mixed_shapes_union".to_string(),
            is_exact: true,
            reported_error_bound_angstrom2: Some(0.0),
        };
    }
    if let Some(area) = geo_polygonized_region_union_area(regions, bounds) {
        return RegionAreaEstimate {
            area_angstrom2: area,
            method: "geo_polygonized_boolean_region_union".to_string(),
            is_exact: false,
            reported_error_bound_angstrom2: Some(
                conservative_grid_region_union_error_bound(regions, (xmin, xmax, ymin, ymax), 0.05)
                    .min(area.max(1.0) * 0.02),
            ),
        };
    }

    let coarse = grid_region_union_area(
        regions,
        xmin,
        xmax,
        ymin,
        ymax,
        REGION_UNION_COARSE_GRID_SPACING_ANGSTROM,
    );
    let fine = grid_region_union_area(
        regions,
        xmin,
        xmax,
        ymin,
        ymax,
        REGION_UNION_FINE_GRID_SPACING_ANGSTROM,
    );
    RegionAreaEstimate {
        area_angstrom2: fine,
        method: format!(
            "adaptive_region_union_grid_fine_{:.2}A",
            REGION_UNION_FINE_GRID_SPACING_ANGSTROM
        ),
        is_exact: false,
        reported_error_bound_angstrom2: Some(
            conservative_grid_region_union_error_bound(
                regions,
                (xmin, xmax, ymin, ymax),
                REGION_UNION_FINE_GRID_SPACING_ANGSTROM,
            )
            .max((fine - coarse).abs() + REGION_UNION_FINE_GRID_SPACING_ANGSTROM.powi(2)),
        ),
    }
}
