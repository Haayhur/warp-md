use super::*;

pub(super) fn membrane_has_spatial_constraints(
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
) -> bool {
    membrane.protein_boundary.is_some() || leaflet_has_spatial_regions(leaflet)
}

pub(super) fn leaflet_has_spatial_regions(leaflet: &LeafletRequest) -> bool {
    leaflet
        .regions
        .iter()
        .any(|region| matches!(region.role.as_str(), "hole" | "patch"))
}

pub(super) fn membrane_allows_layout_point(
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
    point: &LayoutPoint,
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> Result<bool> {
    if !leaflet_allows_point_periodic_basis(leaflet, [point.x, point.y], bounds, periodicity, basis)
    {
        return Ok(false);
    }
    let Some(boundary) = protein_boundary_geometry(membrane, proteins)? else {
        return Ok(true);
    };
    Ok(boundary.contains_point_with_margin([point.x, point.y], point.radius))
}

pub(super) fn membrane_allows_layout_point_with_boundary(
    leaflet: &LeafletRequest,
    boundary: Option<&ProteinBoundaryGeometry>,
    point: &LayoutPoint,
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> bool {
    if !leaflet_allows_point_periodic_basis(leaflet, [point.x, point.y], bounds, periodicity, basis)
    {
        return false;
    }
    boundary.is_none_or(|boundary| {
        boundary.contains_point_with_margin([point.x, point.y], point.radius)
    })
}

pub(super) fn confine_points_to_protein_boundary(
    points: &mut [LayoutPoint],
    membrane: &MembraneRequest,
    proteins: &[InsertedComponent],
) -> Result<()> {
    let Some(boundary) = protein_boundary_geometry(membrane, proteins)? else {
        return Ok(());
    };
    for point in points {
        if !boundary.contains_point_with_margin([point.x, point.y], point.radius) {
            let projected = boundary.project_point([point.x, point.y], point.radius);
            point.x = projected[0];
            point.y = projected[1];
        }
    }
    Ok(())
}

pub(super) fn confine_points_to_allowed_leaflet_regions(
    points: &mut [LayoutPoint],
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> Result<()> {
    if !membrane_has_spatial_constraints(membrane, leaflet) {
        return Ok(());
    }
    for point in points {
        if point_inside_layout_domain(point, bounds, periodicity, basis)
            && membrane_allows_layout_point(
                membrane,
                leaflet,
                proteins,
                point,
                bounds,
                periodicity,
                basis,
            )?
        {
            continue;
        }
        let Some(projected) = nearest_allowed_leaflet_point(
            *point,
            membrane,
            leaflet,
            proteins,
            bounds,
            periodicity,
            basis,
        )?
        else {
            return Err(anyhow!(
                "leaflet {} could not keep relaxed lipid point inside requested regions",
                leaflet.name
            ));
        };
        point.x = projected.x;
        point.y = projected.y;
    }
    Ok(())
}

pub(super) fn nearest_allowed_leaflet_point(
    point: LayoutPoint,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> Result<Option<LayoutPoint>> {
    if let Some(projected) = analytic_allowed_leaflet_projection(
        point,
        membrane,
        leaflet,
        proteins,
        bounds,
        periodicity,
        basis,
    )? {
        return Ok(Some(projected));
    }
    let spacing = 0.5f32.max(point.radius * 0.25);
    let max_radius = ((bounds.xmax - bounds.xmin).powi(2) + (bounds.ymax - bounds.ymin).powi(2))
        .sqrt()
        + point.radius;
    let rings = (max_radius / spacing).ceil() as i32;
    let mut best: Option<(f32, LayoutPoint)> = None;
    for ring in 0..=rings {
        for ix in -ring..=ring {
            for iy in -ring..=ring {
                if ring > 0 && ix.abs() != ring && iy.abs() != ring {
                    continue;
                }
                let candidate = LayoutPoint {
                    x: point.x + ix as f32 * spacing,
                    y: point.y + iy as f32 * spacing,
                    radius: point.radius,
                };
                if !point_inside_layout_domain(&candidate, bounds, periodicity, basis)
                    || !membrane_allows_layout_point(
                        membrane,
                        leaflet,
                        proteins,
                        &candidate,
                        bounds,
                        periodicity,
                        basis,
                    )?
                {
                    continue;
                }
                let distance = (candidate.x - point.x).powi(2) + (candidate.y - point.y).powi(2);
                match best {
                    Some((best_distance, _)) if distance >= best_distance => {}
                    _ => best = Some((distance, candidate)),
                }
            }
        }
        if best.is_some() {
            break;
        }
    }
    Ok(best.map(|(_, point)| point))
}

pub(super) fn analytic_allowed_leaflet_projection(
    point: LayoutPoint,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> Result<Option<LayoutPoint>> {
    let boundary = protein_boundary_geometry(membrane, proteins)?;
    Ok(analytic_allowed_leaflet_projection_with_boundary(
        point,
        leaflet,
        bounds,
        periodicity,
        basis,
        boundary.as_ref(),
    ))
}

pub(super) fn analytic_allowed_leaflet_projection_with_boundary(
    point: LayoutPoint,
    leaflet: &LeafletRequest,
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
    boundary: Option<&ProteinBoundaryGeometry>,
) -> Option<LayoutPoint> {
    let xy = [point.x, point.y];
    if let Some(boundary) = boundary {
        if !boundary.contains_point_with_margin(xy, point.radius) {
            let projected = boundary.project_point(xy, point.radius);
            let candidate = LayoutPoint {
                x: projected[0],
                y: projected[1],
                radius: point.radius,
            };
            if point_inside_layout_domain(&candidate, bounds, periodicity, basis)
                && membrane_allows_layout_point_with_boundary(
                    leaflet,
                    Some(boundary),
                    &candidate,
                    bounds,
                    periodicity,
                    basis,
                )
            {
                return Some(candidate);
            }
        }
    }

    let patch_regions = leaflet
        .regions
        .iter()
        .filter(|region| region.role == "patch")
        .collect::<Vec<_>>();
    if !patch_regions.is_empty()
        && !patch_regions.iter().any(|region| {
            region_contains_point_periodic_basis(region, xy, bounds, periodicity, basis)
        })
    {
        let Some(region) = patch_regions.iter().min_by(|left, right| {
            region_boundary_distance_periodic_basis(left, xy, bounds, periodicity, basis)
                .partial_cmp(&region_boundary_distance_periodic_basis(
                    right,
                    xy,
                    bounds,
                    periodicity,
                    basis,
                ))
                .unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            return None;
        };
        if let Some(candidate) =
            project_to_region_legal_side_periodic(point, region, true, bounds, periodicity, basis)
        {
            if point_inside_layout_domain(&candidate, bounds, periodicity, basis)
                && membrane_allows_layout_point_with_boundary(
                    leaflet,
                    boundary,
                    &candidate,
                    bounds,
                    periodicity,
                    basis,
                )
            {
                return Some(candidate);
            }
        }
    }

    for region in leaflet.regions.iter().filter(|region| {
        region.role == "hole"
            && region_contains_point_periodic_basis(region, xy, bounds, periodicity, basis)
    }) {
        if let Some(candidate) =
            project_to_region_legal_side_periodic(point, region, false, bounds, periodicity, basis)
        {
            if point_inside_layout_domain(&candidate, bounds, periodicity, basis)
                && membrane_allows_layout_point_with_boundary(
                    leaflet,
                    boundary,
                    &candidate,
                    bounds,
                    periodicity,
                    basis,
                )
            {
                return Some(candidate);
            }
        }
    }
    None
}

fn project_to_region_legal_side(
    point: LayoutPoint,
    region: &LeafletRegion,
    inside_is_legal: bool,
) -> Option<LayoutPoint> {
    let boundary = region_boundary_point(region, [point.x, point.y])?;
    let center = region_center(region)?;
    let direction = if inside_is_legal {
        normalize2([center[0] - boundary[0], center[1] - boundary[1]])
    } else {
        normalize2([boundary[0] - center[0], boundary[1] - center[1]])
    }
    .unwrap_or([1.0, 0.0]);
    let margin = point.radius + 1.0e-3;
    Some(LayoutPoint {
        x: boundary[0] + direction[0] * margin,
        y: boundary[1] + direction[1] * margin,
        radius: point.radius,
    })
}

fn project_to_region_legal_side_periodic(
    point: LayoutPoint,
    region: &LeafletRegion,
    inside_is_legal: bool,
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> Option<LayoutPoint> {
    let image_point =
        closest_periodic_region_image_basis(region, [point.x, point.y], bounds, periodicity, basis);
    let projected = project_to_region_legal_side(
        LayoutPoint {
            x: image_point[0],
            y: image_point[1],
            radius: point.radius,
        },
        region,
        inside_is_legal,
    )?;
    let wrapped =
        wrap_point_for_layout_basis([projected.x, projected.y], bounds, periodicity, basis);
    Some(LayoutPoint {
        x: wrapped[0],
        y: wrapped[1],
        radius: projected.radius,
    })
}

fn region_boundary_point(region: &LeafletRegion, point: [f32; 2]) -> Option<[f32; 2]> {
    match &region.geometry {
        RegionGeometry::Circle {
            center_angstrom,
            radius_angstrom,
        } => {
            let direction =
                normalize2([point[0] - center_angstrom[0], point[1] - center_angstrom[1]])
                    .unwrap_or([1.0, 0.0]);
            Some([
                center_angstrom[0] + direction[0] * radius_angstrom,
                center_angstrom[1] + direction[1] * radius_angstrom,
            ])
        }
        RegionGeometry::Ellipse {
            center_angstrom,
            radius_angstrom,
            rotate_degrees,
        } => {
            let local = inverse_rotated_xy(point, *center_angstrom, *rotate_degrees);
            let denom = ((local[0] / radius_angstrom[0]).powi(2)
                + (local[1] / radius_angstrom[1]).powi(2))
            .sqrt();
            let boundary_local = if denom > f32::EPSILON {
                [local[0] / denom, local[1] / denom]
            } else {
                [radius_angstrom[0], 0.0]
            };
            Some(forward_rotated_xy(
                boundary_local,
                *center_angstrom,
                *rotate_degrees,
            ))
        }
        RegionGeometry::Rectangle {
            center_angstrom,
            size_angstrom,
            rotate_degrees,
        } => {
            let local = inverse_rotated_xy(point, *center_angstrom, *rotate_degrees);
            let half = [size_angstrom[0] * 0.5, size_angstrom[1] * 0.5];
            let inside = local[0].abs() <= half[0] && local[1].abs() <= half[1];
            let boundary_local = if inside {
                let dx = half[0] - local[0].abs();
                let dy = half[1] - local[1].abs();
                if dx <= dy {
                    [local[0].signum() * half[0], local[1]]
                } else {
                    [local[0], local[1].signum() * half[1]]
                }
            } else {
                [
                    local[0].clamp(-half[0], half[0]),
                    local[1].clamp(-half[1], half[1]),
                ]
            };
            Some(forward_rotated_xy(
                boundary_local,
                *center_angstrom,
                *rotate_degrees,
            ))
        }
        RegionGeometry::Polygon { .. } => Some(nearest_point_on_polygon_boundary(
            point,
            &transformed_polygon_points(region),
        )),
    }
}

fn region_center(region: &LeafletRegion) -> Option<[f32; 2]> {
    match &region.geometry {
        RegionGeometry::Circle {
            center_angstrom, ..
        }
        | RegionGeometry::Ellipse {
            center_angstrom, ..
        }
        | RegionGeometry::Rectangle {
            center_angstrom, ..
        } => Some(*center_angstrom),
        RegionGeometry::Polygon { .. } => {
            polygon_bounds_center(&transformed_polygon_points(region))
        }
    }
}

fn normalize2(vector: [f32; 2]) -> Option<[f32; 2]> {
    let length = (vector[0] * vector[0] + vector[1] * vector[1]).sqrt();
    (length > f32::EPSILON).then_some([vector[0] / length, vector[1] / length])
}

pub(super) fn point_inside_bounds_with_radius_periodic(
    point: &LayoutPoint,
    bounds: LayoutBounds,
    _periodicity: LayoutPeriodicity,
) -> bool {
    point.x >= bounds.xmin + point.radius
        && point.x <= bounds.xmax - point.radius
        && point.y >= bounds.ymin + point.radius
        && point.y <= bounds.ymax - point.radius
}

pub(super) fn point_inside_layout_domain(
    point: &LayoutPoint,
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> bool {
    if let Some(basis) = basis.filter(|_| periodicity.x && periodicity.y) {
        let Some(margins) = basis_fractional_margins(basis, point.radius) else {
            return false;
        };
        if margins[0] > 0.5 || margins[1] > 0.5 {
            return false;
        }
        let fractional = basis.fractional([point.x, point.y]);
        return fractional[0] >= margins[0]
            && fractional[0] <= 1.0 - margins[0]
            && fractional[1] >= margins[1]
            && fractional[1] <= 1.0 - margins[1];
    }
    point_inside_bounds_with_radius_periodic(point, bounds, periodicity)
}

pub(super) fn leaflet_allows_point(leaflet: &LeafletRequest, point: [f32; 2]) -> bool {
    let patch_regions = leaflet
        .regions
        .iter()
        .filter(|region| region.role == "patch")
        .collect::<Vec<_>>();
    if !patch_regions.is_empty()
        && !patch_regions
            .iter()
            .any(|region| region_contains_point(region, point))
    {
        return false;
    }
    !leaflet
        .regions
        .iter()
        .filter(|region| region.role == "hole")
        .any(|region| region_contains_point(region, point))
}

#[cfg(test)]
pub(super) fn leaflet_allows_point_periodic(
    leaflet: &LeafletRequest,
    point: [f32; 2],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
) -> bool {
    leaflet_allows_point_periodic_basis(leaflet, point, bounds, periodicity, None)
}

pub(super) fn leaflet_allows_point_periodic_basis(
    leaflet: &LeafletRequest,
    point: [f32; 2],
    bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
) -> bool {
    let patch_regions = leaflet
        .regions
        .iter()
        .filter(|region| region.role == "patch")
        .collect::<Vec<_>>();
    if !patch_regions.is_empty()
        && !patch_regions.iter().any(|region| {
            region_contains_point_periodic_basis(region, point, bounds, periodicity, basis)
        })
    {
        return false;
    }
    !leaflet
        .regions
        .iter()
        .filter(|region| region.role == "hole")
        .any(|region| {
            region_contains_point_periodic_basis(region, point, bounds, periodicity, basis)
        })
}
