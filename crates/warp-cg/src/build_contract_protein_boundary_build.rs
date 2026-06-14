use super::*;

pub(super) fn protein_boundary_geometry(
    membrane: &MembraneRequest,
    proteins: &[InsertedComponent],
) -> Result<Option<ProteinBoundaryGeometry>> {
    let Some(boundary) = &membrane.protein_boundary else {
        return Ok(None);
    };
    if let (Some(center), Some(radius)) = (boundary.center_angstrom, boundary.radius_angstrom) {
        return Ok(Some(ProteinBoundaryGeometry::Circle(
            ProteinBoundaryCircle {
                center_angstrom: center,
                radius_angstrom: (radius - boundary.buffer_angstrom).max(0.0),
            },
        )));
    }

    let points = protein_boundary_points(membrane, proteins, boundary)?;
    if points.is_empty() {
        return Err(anyhow!(
            "membrane {} protein_boundary matched proteins without coordinates",
            membrane.name
        ));
    }
    if matches!(
        boundary.geometry.as_str(),
        "convex_hull" | "concave_hull" | "alpha_shape"
    ) && boundary.radius_angstrom.is_none()
    {
        let mut unique = points.clone();
        unique.sort_by(|left, right| {
            left[0]
                .partial_cmp(&right[0])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left[1]
                        .partial_cmp(&right[1])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        unique.dedup_by(|left, right| {
            (left[0] - right[0]).abs() < 1.0e-5 && (left[1] - right[1]).abs() < 1.0e-5
        });
        if unique.len() >= 3 {
            let polygons = match boundary.geometry.as_str() {
                "alpha_shape" => {
                    let alpha_radius = boundary
                        .alpha_radius_angstrom
                        .unwrap_or_else(|| default_alpha_radius(&unique));
                    match alpha_shape_components(&unique, alpha_radius) {
                        Some(components)
                            if components.len() > 1
                                && polygon_components_are_disconnected(&components) =>
                        {
                            components
                        }
                        Some(components) if components.len() > 1 => {
                            if components.len() == 2 {
                                if let Some((outer, holes)) =
                                    nested_polygon_from_components(&components)
                                {
                                    return Ok(Some(ProteinBoundaryGeometry::NestedPolygons {
                                        outer,
                                        holes,
                                        inset_angstrom: boundary.buffer_angstrom,
                                    }));
                                }
                            }
                            if let Some(rings) = nested_polygon_forest_from_components(&components)
                            {
                                return Ok(Some(ProteinBoundaryGeometry::NestedPolygonForest {
                                    rings,
                                    inset_angstrom: boundary.buffer_angstrom,
                                }));
                            }
                            vec![alpha_shape(&unique, alpha_radius)
                                .unwrap_or_else(|| concave_hull(unique))]
                        }
                        Some(mut components) => vec![alpha_shape(&unique, alpha_radius)
                            .unwrap_or_else(|| {
                                components.pop().unwrap_or_else(|| concave_hull(unique))
                            })],
                        None => vec![concave_hull(unique)],
                    }
                }
                "concave_hull" => {
                    vec![ordered_concave_boundary(&points).unwrap_or_else(|| concave_hull(unique))]
                }
                _ => vec![convex_hull(unique)],
            };
            if polygons.len() > 1 {
                return Ok(Some(ProteinBoundaryGeometry::MultiPolygon {
                    polygons,
                    inset_angstrom: boundary.buffer_angstrom,
                }));
            }
            return Ok(Some(ProteinBoundaryGeometry::Polygon {
                points: polygons.into_iter().next().unwrap_or_default(),
                inset_angstrom: boundary.buffer_angstrom,
            }));
        }
    }
    let center = boundary
        .center_angstrom
        .unwrap_or_else(|| xy_center_of_points(&points));
    let radius = boundary
        .radius_angstrom
        .unwrap_or_else(|| protein_boundary_radius_from_points(&points, center, boundary));
    Ok(Some(ProteinBoundaryGeometry::Circle(
        ProteinBoundaryCircle {
            center_angstrom: center,
            radius_angstrom: (radius - boundary.buffer_angstrom).max(0.0),
        },
    )))
}

pub(super) fn polygon_components_are_disconnected(polygons: &[Vec<[f32; 2]>]) -> bool {
    for left_idx in 0..polygons.len() {
        for right_idx in (left_idx + 1)..polygons.len() {
            if polygon_bounds_overlap(&polygons[left_idx], &polygons[right_idx], 1.0e-5) {
                return false;
            }
        }
    }
    true
}

pub(super) fn nested_polygon_from_components(
    polygons: &[Vec<[f32; 2]>],
) -> Option<(Vec<[f32; 2]>, Vec<Vec<[f32; 2]>>)> {
    let (outer_idx, outer) = polygons
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| {
            polygon_area(left)
                .partial_cmp(&polygon_area(right))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;
    if outer.len() < 3 {
        return None;
    }
    let mut holes = Vec::new();
    for (idx, polygon) in polygons.iter().enumerate() {
        if idx == outer_idx || polygon.len() < 3 {
            continue;
        }
        if polygon.iter().all(|point| point_in_polygon(*point, outer))
            && polygon_area(polygon) < polygon_area(outer)
        {
            holes.push(polygon.clone());
        } else {
            return None;
        }
    }
    (!holes.is_empty()).then(|| (outer.clone(), holes))
}

pub(super) fn nested_polygon_forest_from_components(
    polygons: &[Vec<[f32; 2]>],
) -> Option<Vec<Vec<[f32; 2]>>> {
    if polygons.len() < 2 {
        return None;
    }
    let mut rings = Vec::with_capacity(polygons.len());
    for polygon in polygons {
        if polygon.len() < 3
            || polygon_area(polygon) <= 1.0e-5
            || polygon_has_self_intersections(polygon)
        {
            return None;
        }
        let mut ring = polygon.clone();
        ensure_ccw_polygon(&mut ring);
        rings.push(ring);
    }
    for idx in 0..rings.len() {
        for jdx in (idx + 1)..rings.len() {
            let left_in_right = rings[idx]
                .iter()
                .all(|point| point_in_polygon(*point, &rings[jdx]));
            let right_in_left = rings[jdx]
                .iter()
                .all(|point| point_in_polygon(*point, &rings[idx]));
            if left_in_right || right_in_left {
                continue;
            }
            if polygon_bounds_overlap(&rings[idx], &rings[jdx], 1.0e-5)
                && polygon_distance(&rings[idx], &rings[jdx]) <= 1.0e-5
            {
                return None;
            }
        }
    }
    rings.sort_by(|left, right| {
        polygon_area(right)
            .partial_cmp(&polygon_area(left))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Some(rings)
}

pub(super) fn polygon_bounds_overlap(
    left: &[[f32; 2]],
    right: &[[f32; 2]],
    tolerance: f32,
) -> bool {
    let Some(left_bounds) = polygon_bounds(left) else {
        return false;
    };
    let Some(right_bounds) = polygon_bounds(right) else {
        return false;
    };
    left_bounds.0 <= right_bounds.1 + tolerance
        && left_bounds.1 + tolerance >= right_bounds.0
        && left_bounds.2 <= right_bounds.3 + tolerance
        && left_bounds.3 + tolerance >= right_bounds.2
}

pub(super) fn protein_boundary_points(
    membrane: &MembraneRequest,
    proteins: &[InsertedComponent],
    boundary: &ProteinBoundaryRequest,
) -> Result<Vec<[f32; 2]>> {
    let matching = proteins
        .iter()
        .filter(|protein| {
            boundary
                .protein
                .as_ref()
                .is_none_or(|name| protein.name == *name)
        })
        .collect::<Vec<_>>();
    if matching.is_empty() {
        return Err(anyhow!(
            "membrane {} protein_boundary did not match any inserted protein",
            membrane.name
        ));
    }

    let mut points = Vec::new();
    for protein in matching {
        let Some(path) = &protein.coordinates else {
            continue;
        };
        let molecule = read_molecule(
            Path::new(path),
            protein.format.as_deref(),
            true,
            false,
            None,
        )?;
        if molecule.atoms.is_empty() {
            continue;
        }
        let source_center = match protein.placement.center_method.as_str() {
            "cog" => molecule_center_of_geometry(&molecule.atoms),
            "none" => [0.0, 0.0, 0.0],
            method => {
                return Err(anyhow!(
                    "protein {} placement center_method must be cog or none, got {method}",
                    protein.name
                ))
            }
        };
        for atom in &molecule.atoms {
            let [x, y, _z] = transform_inserted_position(
                [atom.position.x, atom.position.y, atom.position.z],
                source_center,
                protein.placement.rotate_degrees_xyz,
                protein.placement.center_angstrom,
            );
            points.push([x, y]);
        }
    }
    Ok(points)
}

pub(super) fn protein_boundary_radius_from_points(
    points: &[[f32; 2]],
    center: [f32; 2],
    boundary: &ProteinBoundaryRequest,
) -> f32 {
    match boundary.radius_strategy.as_str() {
        "radial_quantile" => {
            radial_quantile_radius(points, center, boundary.radius_quantile.unwrap_or(0.5))
        }
        _ => xy_radius_of_points(points, center),
    }
}

pub(super) fn radial_quantile_radius(points: &[[f32; 2]], center: [f32; 2], quantile: f32) -> f32 {
    if points.is_empty() {
        return 0.0;
    }
    let mut radii = points
        .iter()
        .map(|point| {
            let dx = point[0] - center[0];
            let dy = point[1] - center[1];
            (dx * dx + dy * dy).sqrt()
        })
        .collect::<Vec<_>>();
    radii.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((radii.len() - 1) as f32 * quantile).round() as usize;
    radii[idx.min(radii.len() - 1)]
}

pub(super) fn membrane_boundary_matches_protein(
    membrane: &MembraneRequest,
    protein: &InsertedComponent,
) -> bool {
    membrane.protein_boundary.as_ref().is_some_and(|boundary| {
        boundary
            .protein
            .as_ref()
            .is_none_or(|name| protein.name == *name)
    })
}

pub(super) trait IfEmptyThen {
    fn if_empty_then(self, fallback: &str) -> String;
}

impl IfEmptyThen for String {
    fn if_empty_then(self, fallback: &str) -> String {
        if self.is_empty() {
            fallback.to_string()
        } else {
            self
        }
    }
}

pub(super) fn leaflet_geometry_diagnostics(
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
    points: &[LayoutPoint],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> Result<LeafletGeometryDiagnostics> {
    const TOLERANCE_ANGSTROM: f32 = 1.0e-3;
    let mut constraints = Vec::new();

    if let Some(boundary) = protein_boundary_geometry(membrane, proteins)? {
        let mut diagnostic =
            new_geometry_constraint_diagnostic("protein_boundary", "protein_boundary", "inside");
        for point in points {
            if !boundary.contains_point_with_margin([point.x, point.y], point.radius) {
                let violation = protein_boundary_margin_violation_distance(
                    &boundary,
                    [point.x, point.y],
                    point.radius,
                );
                record_geometry_violation(&mut diagnostic, violation, TOLERANCE_ANGSTROM);
            }
        }
        constraints.push(diagnostic);
    }

    let patch_regions = leaflet
        .regions
        .iter()
        .filter(|region| region.role == "patch")
        .collect::<Vec<_>>();
    if !patch_regions.is_empty() {
        let mut diagnostic =
            new_geometry_constraint_diagnostic("patch_union", "leaflet_region_union", "patch");
        for point in points {
            let xy = [point.x, point.y];
            if !patch_regions.iter().any(|region| {
                region_contains_point_periodic_basis(region, xy, bounds, periodicity, basis)
            }) {
                let violation = patch_regions
                    .iter()
                    .map(|region| {
                        region_boundary_distance_periodic_basis(
                            region,
                            xy,
                            bounds,
                            periodicity,
                            basis,
                        )
                    })
                    .fold(f32::INFINITY, f32::min);
                record_geometry_violation(&mut diagnostic, violation, TOLERANCE_ANGSTROM);
            }
        }
        constraints.push(diagnostic);
    }

    for region in leaflet
        .regions
        .iter()
        .filter(|region| region.role == "hole")
    {
        let mut diagnostic = new_geometry_constraint_diagnostic(
            region.name.as_deref().unwrap_or("hole"),
            region_shape_name(region),
            "hole",
        );
        for point in points {
            let xy = [point.x, point.y];
            if region_contains_point_periodic_basis(region, xy, bounds, periodicity, basis) {
                let violation =
                    region_boundary_distance_periodic_basis(region, xy, bounds, periodicity, basis);
                record_geometry_violation(&mut diagnostic, violation, TOLERANCE_ANGSTROM);
            }
        }
        constraints.push(diagnostic);
    }

    let violation_count = constraints
        .iter()
        .map(|diagnostic| diagnostic.violation_count)
        .sum();
    let max_violation_angstrom = constraints
        .iter()
        .map(|diagnostic| diagnostic.max_violation_angstrom)
        .fold(0.0f32, f32::max);
    Ok(LeafletGeometryDiagnostics {
        tolerance_angstrom: TOLERANCE_ANGSTROM,
        constraint_count: constraints.len(),
        violation_count,
        max_violation_angstrom,
        constraints,
    })
}

fn new_geometry_constraint_diagnostic(
    name: &str,
    kind: &str,
    role: &str,
) -> LeafletGeometryConstraintDiagnostic {
    LeafletGeometryConstraintDiagnostic {
        name: name.to_string(),
        kind: kind.to_string(),
        role: role.to_string(),
        violation_count: 0,
        max_violation_angstrom: 0.0,
    }
}

fn record_geometry_violation(
    diagnostic: &mut LeafletGeometryConstraintDiagnostic,
    violation_angstrom: f32,
    tolerance_angstrom: f32,
) {
    if violation_angstrom > tolerance_angstrom {
        diagnostic.violation_count += 1;
        diagnostic.max_violation_angstrom =
            diagnostic.max_violation_angstrom.max(violation_angstrom);
    }
}

fn region_shape_name(region: &LeafletRegion) -> &'static str {
    match &region.geometry {
        RegionGeometry::Circle { .. } => "circle",
        RegionGeometry::Ellipse { .. } => "ellipse",
        RegionGeometry::Rectangle { .. } => "rectangle",
        RegionGeometry::Polygon { .. } => "polygon",
    }
}
