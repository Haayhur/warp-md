use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct ProteinBoundaryCircle {
    pub(super) center_angstrom: [f32; 2],
    pub(super) radius_angstrom: f32,
}

#[derive(Clone, Debug)]
pub(super) enum ProteinBoundaryGeometry {
    Circle(ProteinBoundaryCircle),
    Polygon {
        points: Vec<[f32; 2]>,
        inset_angstrom: f32,
    },
    MultiPolygon {
        polygons: Vec<Vec<[f32; 2]>>,
        inset_angstrom: f32,
    },
    NestedPolygons {
        outer: Vec<[f32; 2]>,
        holes: Vec<Vec<[f32; 2]>>,
        inset_angstrom: f32,
    },
    NestedPolygonForest {
        rings: Vec<Vec<[f32; 2]>>,
        inset_angstrom: f32,
    },
}

impl ProteinBoundaryGeometry {
    pub(super) fn area_estimate(&self) -> (f32, bool) {
        match self {
            Self::Circle(circle) => (std::f32::consts::PI * circle.radius_angstrom.powi(2), true),
            Self::Polygon {
                points,
                inset_angstrom,
            } => polygon_boundary_area_estimate(points, *inset_angstrom),
            Self::MultiPolygon {
                polygons,
                inset_angstrom,
            } => {
                if *inset_angstrom <= 1.0e-6 {
                    if let Some(bounds) = polygon_collection_layout_bounds(polygons) {
                        if let Some(area) = geo_simple_polygon_union_area(polygons, bounds) {
                            return (area, true);
                        } else if let Some(area) =
                            exact_simple_polygon_union_area_from_polygons(polygons, bounds)
                        {
                            return (area, true);
                        }
                    }
                } else if let Some(area) =
                    exact_convex_multipolygon_inset_union_area(polygons, *inset_angstrom)
                {
                    return (area, true);
                } else if polygons.iter().any(|polygon| !polygon_is_convex(polygon)) {
                    let depths = vec![0usize; polygons.len()];
                    if let Some(area) =
                        geo_nested_polygon_forest_inset_area(polygons, &depths, *inset_angstrom)
                    {
                        return (area, false);
                    }
                } else if multipolygon_components_overlap(polygons) {
                    return (
                        multipolygon_area_with_inset_union(polygons, *inset_angstrom),
                        false,
                    );
                }
                let mut total = 0.0;
                let mut exact = true;
                for polygon in polygons {
                    let (area, is_exact) = polygon_boundary_area_estimate(polygon, *inset_angstrom);
                    total += area;
                    exact &= is_exact;
                }
                (total, exact)
            }
            Self::NestedPolygons {
                outer,
                holes,
                inset_angstrom,
            } => {
                if *inset_angstrom <= 1.0e-6 && multipolygon_components_overlap(holes) {
                    if let Some(area) = geo_nested_polygon_inset_area(outer, holes, *inset_angstrom)
                    {
                        return (area, true);
                    }
                }
                if *inset_angstrom <= 1.0e-6 {
                    let hole_area = holes.iter().map(|hole| polygon_area(hole)).sum::<f32>();
                    ((polygon_area(outer) - hole_area).max(0.0), true)
                } else if let Some(area) =
                    exact_axis_aligned_rectangle_nested_inset_area(outer, holes, *inset_angstrom)
                {
                    (area, true)
                } else if let Some(area) =
                    exact_convex_nested_inset_area(outer, holes, *inset_angstrom)
                {
                    (area, true)
                } else if let Some(area) = exact_simple_nested_inset_area_before_topology_event(
                    outer,
                    holes,
                    *inset_angstrom,
                ) {
                    (area, true)
                } else {
                    if let Some(area) = geo_nested_polygon_inset_area(outer, holes, *inset_angstrom)
                    {
                        (area, false)
                    } else {
                        let (outer_area, outer_exact) =
                            polygon_boundary_area_estimate(outer, *inset_angstrom);
                        let hole_area = holes.iter().map(|hole| polygon_area(hole)).sum::<f32>();
                        (
                            (outer_area - hole_area).max(0.0),
                            outer_exact && holes.is_empty(),
                        )
                    }
                }
            }
            Self::NestedPolygonForest {
                rings,
                inset_angstrom,
            } => nested_polygon_forest_area_estimate(rings, *inset_angstrom),
        }
    }

    pub(super) fn contains_point(&self, point: [f32; 2]) -> bool {
        self.contains_point_with_margin(point, 0.0)
    }

    pub(super) fn contains_point_with_margin(&self, point: [f32; 2], margin_angstrom: f32) -> bool {
        match self {
            Self::Circle(circle) => {
                let dx = point[0] - circle.center_angstrom[0];
                let dy = point[1] - circle.center_angstrom[1];
                let radius = (circle.radius_angstrom - margin_angstrom.max(0.0)).max(0.0);
                dx * dx + dy * dy <= radius.powi(2)
            }
            Self::Polygon {
                points,
                inset_angstrom,
            } => polygon_boundary_contains_point(points, *inset_angstrom, point, margin_angstrom),
            Self::MultiPolygon {
                polygons,
                inset_angstrom,
            } => polygons.iter().any(|polygon| {
                polygon_boundary_contains_point(polygon, *inset_angstrom, point, margin_angstrom)
            }),
            Self::NestedPolygons {
                outer,
                holes,
                inset_angstrom,
            } => {
                polygon_boundary_contains_point(outer, *inset_angstrom, point, margin_angstrom)
                    && holes.iter().all(|hole| {
                        !polygon_hole_rejects_point(
                            hole,
                            *inset_angstrom + margin_angstrom.max(0.0),
                            point,
                        )
                    })
            }
            Self::NestedPolygonForest {
                rings,
                inset_angstrom,
            } => {
                nested_polygon_forest_contains_point(rings, *inset_angstrom, point, margin_angstrom)
            }
        }
    }

    pub(super) fn bounds(&self) -> (f32, f32, f32, f32) {
        match self {
            Self::Circle(circle) => (
                circle.center_angstrom[0] - circle.radius_angstrom,
                circle.center_angstrom[0] + circle.radius_angstrom,
                circle.center_angstrom[1] - circle.radius_angstrom,
                circle.center_angstrom[1] + circle.radius_angstrom,
            ),
            Self::Polygon { points, .. } => {
                let mut xmin = f32::INFINITY;
                let mut xmax = f32::NEG_INFINITY;
                let mut ymin = f32::INFINITY;
                let mut ymax = f32::NEG_INFINITY;
                for point in points {
                    xmin = xmin.min(point[0]);
                    xmax = xmax.max(point[0]);
                    ymin = ymin.min(point[1]);
                    ymax = ymax.max(point[1]);
                }
                (xmin, xmax, ymin, ymax)
            }
            Self::MultiPolygon { polygons, .. } => {
                let mut xmin = f32::INFINITY;
                let mut xmax = f32::NEG_INFINITY;
                let mut ymin = f32::INFINITY;
                let mut ymax = f32::NEG_INFINITY;
                for polygon in polygons {
                    for point in polygon {
                        xmin = xmin.min(point[0]);
                        xmax = xmax.max(point[0]);
                        ymin = ymin.min(point[1]);
                        ymax = ymax.max(point[1]);
                    }
                }
                (xmin, xmax, ymin, ymax)
            }
            Self::NestedPolygons { outer, .. } => {
                polygon_bounds(outer).unwrap_or((0.0, 0.0, 0.0, 0.0))
            }
            Self::NestedPolygonForest { rings, .. } => {
                let mut xmin = f32::INFINITY;
                let mut xmax = f32::NEG_INFINITY;
                let mut ymin = f32::INFINITY;
                let mut ymax = f32::NEG_INFINITY;
                for ring in rings {
                    for point in ring {
                        xmin = xmin.min(point[0]);
                        xmax = xmax.max(point[0]);
                        ymin = ymin.min(point[1]);
                        ymax = ymax.max(point[1]);
                    }
                }
                if xmin.is_finite() {
                    (xmin, xmax, ymin, ymax)
                } else {
                    (0.0, 0.0, 0.0, 0.0)
                }
            }
        }
    }

    pub(super) fn project_point(&self, point: [f32; 2], radius: f32) -> [f32; 2] {
        match self {
            Self::Circle(circle) => {
                let dx = point[0] - circle.center_angstrom[0];
                let dy = point[1] - circle.center_angstrom[1];
                let dist = (dx * dx + dy * dy).sqrt();
                let max_dist = (circle.radius_angstrom - radius).max(0.0);
                if dist > max_dist && dist > 1.0e-6 {
                    let scale = max_dist / dist;
                    [
                        circle.center_angstrom[0] + dx * scale,
                        circle.center_angstrom[1] + dy * scale,
                    ]
                } else {
                    point
                }
            }
            Self::Polygon {
                points,
                inset_angstrom,
            } => {
                let margin = if *inset_angstrom > 0.0 {
                    *inset_angstrom + radius
                } else {
                    0.0
                };
                project_point_to_polygon_with_margin(point, points, margin)
            }
            Self::MultiPolygon {
                polygons,
                inset_angstrom,
            } => {
                let margin = if *inset_angstrom > 0.0 {
                    *inset_angstrom + radius
                } else {
                    0.0
                };
                polygons
                    .iter()
                    .map(|polygon| {
                        let projected =
                            project_point_to_polygon_with_margin(point, polygon, margin);
                        (squared_distance2(point, projected), projected)
                    })
                    .min_by(|left, right| {
                        left.0
                            .partial_cmp(&right.0)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(_, projected)| projected)
                    .unwrap_or(point)
            }
            Self::NestedPolygons {
                outer,
                holes,
                inset_angstrom,
            } => {
                let margin = if *inset_angstrom > 0.0 {
                    *inset_angstrom + radius
                } else {
                    radius
                };
                let inside_outer = project_point_to_polygon_with_margin(point, outer, margin);
                if holes
                    .iter()
                    .all(|hole| !polygon_hole_rejects_point(hole, margin, inside_outer))
                {
                    return inside_outer;
                }
                holes
                    .iter()
                    .filter(|hole| polygon_hole_rejects_point(hole, margin, inside_outer))
                    .filter_map(|hole| {
                        project_point_outside_polygon_hole(inside_outer, outer, hole, margin)
                    })
                    .min_by(|left, right| {
                        squared_distance2(point, *left)
                            .partial_cmp(&squared_distance2(point, *right))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(inside_outer)
            }
            Self::NestedPolygonForest {
                rings,
                inset_angstrom,
            } => project_point_to_nested_polygon_forest(point, rings, *inset_angstrom, radius),
        }
    }
}
