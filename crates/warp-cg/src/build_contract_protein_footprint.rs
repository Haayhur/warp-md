use super::*;

pub(super) fn protein_footprint_area_for_leaflet(
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
) -> Result<f32> {
    let mut total = 0.0f32;
    for protein in proteins {
        if membrane_boundary_matches_protein(membrane, protein) {
            continue;
        }
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
        let target = protein.placement.center_angstrom;
        let mut points = molecule
            .atoms
            .iter()
            .filter_map(|atom| {
                let [x, y, z] = transform_inserted_position(
                    [atom.position.x, atom.position.y, atom.position.z],
                    source_center,
                    protein.placement.rotate_degrees_xyz,
                    target,
                );
                protein_bead_overlaps_leaflet_z(z, membrane.center_z_angstrom, &leaflet.side)
                    .then_some([x, y])
            })
            .collect::<Vec<_>>();
        let footprint_area = if leaflet_has_spatial_regions(leaflet) {
            points.retain(|point| leaflet_allows_point(leaflet, *point));
            convex_hull_area(&points)
        } else {
            buffered_convex_hull_area(&points, DEFAULT_PROTEIN_FOOTPRINT_AREA_BUFFER_ANGSTROM)
        };
        total += protein.count as f32 * footprint_area;
    }
    Ok(total)
}

pub(super) fn protein_bead_overlaps_leaflet_z(z: f32, membrane_center_z: f32, side: &str) -> bool {
    let height = DEFAULT_PROTEIN_FOOTPRINT_HEIGHT_ANGSTROM;
    let buffer = DEFAULT_PROTEIN_FOOTPRINT_Z_BUFFER_ANGSTROM;
    match side {
        "upper" => membrane_center_z - buffer < z && z < membrane_center_z + height + buffer,
        "lower" => membrane_center_z + buffer > z && z > membrane_center_z - height - buffer,
        _ => false,
    }
}

pub(super) fn protein_component_exclusions(
    proteins: &[InsertedComponent],
) -> Result<Vec<ExclusionZone>> {
    let mut exclusions = Vec::new();
    for (idx, protein) in proteins.iter().enumerate() {
        let Some(footprint) = &protein.footprint else {
            continue;
        };
        let exclusion = if let Some(center) = footprint.center_angstrom {
            let radius = footprint.radius_angstrom.or_else(|| {
                protein.coordinates.as_ref().and_then(|path| {
                    molecule_xy_radius(path, protein.format.as_deref(), center).ok()
                })
            });
            let Some(radius) = radius else {
                return Err(anyhow!(
                    "proteins[{idx}] footprint radius could not be resolved"
                ));
            };
            ExclusionZone {
                name: Some(format!("protein:{}:footprint", protein.name)),
                center_angstrom: center,
                radius_angstrom: radius + footprint.buffer_angstrom,
            }
        } else if let Some(path) = &protein.coordinates {
            let (center, radius) = molecule_xy_center_radius(path, protein.format.as_deref())?;
            ExclusionZone {
                name: Some(format!("protein:{}:footprint", protein.name)),
                center_angstrom: center,
                radius_angstrom: footprint.radius_angstrom.unwrap_or(radius)
                    + footprint.buffer_angstrom,
            }
        } else {
            continue;
        };
        exclusions.push(exclusion);
    }
    Ok(exclusions)
}

pub(super) fn molecule_xy_center_radius(
    path: &str,
    format: Option<&str>,
) -> Result<([f32; 2], f32)> {
    let molecule = read_molecule(Path::new(path), format, true, false, None)?;
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for atom in &molecule.atoms {
        min_x = min_x.min(atom.position.x);
        max_x = max_x.max(atom.position.x);
        min_y = min_y.min(atom.position.y);
        max_y = max_y.max(atom.position.y);
    }
    let center = [(min_x + max_x) * 0.5, (min_y + max_y) * 0.5];
    Ok((
        center,
        molecule_xy_radius_for_atoms(&molecule.atoms, center),
    ))
}

pub(super) fn molecule_xy_radius(
    path: &str,
    format: Option<&str>,
    center: [f32; 2],
) -> Result<f32> {
    let molecule = read_molecule(Path::new(path), format, true, false, None)?;
    Ok(molecule_xy_radius_for_atoms(&molecule.atoms, center))
}

pub(super) fn molecule_xy_radius_for_atoms(atoms: &[AtomRecord], center: [f32; 2]) -> f32 {
    atoms.iter().fold(0.0, |radius, atom| {
        let dx = atom.position.x - center[0];
        let dy = atom.position.y - center[1];
        radius.max((dx * dx + dy * dy).sqrt())
    })
}

pub(super) fn boundary_protein_exclusions_for_leaflet(
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
) -> Result<Vec<ExclusionZone>> {
    let Some(boundary) = &membrane.protein_boundary else {
        return Ok(Vec::new());
    };
    if boundary.bead_exclusion_radius_angstrom == 0.0 {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for protein in proteins
        .iter()
        .filter(|protein| membrane_boundary_matches_protein(membrane, protein))
    {
        for [x, y, _z] in transformed_component_points_for_leaflet(protein, membrane, leaflet)? {
            out.push(ExclusionZone {
                name: Some(format!("protein-boundary:{}:bead", protein.name)),
                center_angstrom: [x, y],
                radius_angstrom: boundary.bead_exclusion_radius_angstrom,
            });
        }
    }
    Ok(out)
}

pub(super) fn boundary_protein_exclusion_area_for_leaflet(
    bounds: LayoutBounds,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
) -> Result<f32> {
    let Some(boundary) = protein_boundary_geometry(membrane, proteins)? else {
        return Ok(0.0);
    };
    let exclusions = boundary_protein_exclusions_for_leaflet(membrane, leaflet, proteins)?;
    if exclusions.is_empty() {
        return Ok(0.0);
    }
    let radius = exclusions[0].radius_angstrom;
    let spacing = 1.0f32;
    let boundary_bounds = boundary.bounds();
    let xmin = bounds.xmin.max(boundary_bounds.0);
    let xmax = bounds.xmax.min(boundary_bounds.1);
    let ymin = bounds.ymin.max(boundary_bounds.2);
    let ymax = bounds.ymax.min(boundary_bounds.3);
    if xmin >= xmax || ymin >= ymax {
        return Ok(0.0);
    }
    let mut bins: HashMap<(i32, i32), Vec<[f32; 2]>> = HashMap::new();
    let bin_size = (radius * 2.0).max(1.0);
    for exclusion in &exclusions {
        let key = (
            (exclusion.center_angstrom[0] / bin_size).floor() as i32,
            (exclusion.center_angstrom[1] / bin_size).floor() as i32,
        );
        bins.entry(key).or_default().push(exclusion.center_angstrom);
    }
    let nx = ((xmax - xmin) / spacing).ceil() as usize;
    let ny = ((ymax - ymin) / spacing).ceil() as usize;
    let radius_sq = radius * radius;
    let mut covered = 0usize;
    for ix in 0..nx {
        let x = xmin + (ix as f32 + 0.5) * spacing;
        for iy in 0..ny {
            let y = ymin + (iy as f32 + 0.5) * spacing;
            if !boundary.contains_point([x, y]) {
                continue;
            }
            let key = ((x / bin_size).floor() as i32, (y / bin_size).floor() as i32);
            let mut inside = false;
            'neighbors: for dx in -1..=1 {
                for dy in -1..=1 {
                    let Some(points) = bins.get(&(key.0 + dx, key.1 + dy)) else {
                        continue;
                    };
                    if points.iter().any(|point| {
                        let px = x - point[0];
                        let py = y - point[1];
                        px * px + py * py <= radius_sq
                    }) {
                        inside = true;
                        break 'neighbors;
                    }
                }
            }
            if inside {
                covered += 1;
            }
        }
    }
    Ok(covered as f32 * spacing * spacing)
}

pub(super) fn transformed_component_points_for_leaflet(
    component: &InsertedComponent,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
) -> Result<Vec<[f32; 3]>> {
    let Some(path) = &component.coordinates else {
        return Ok(Vec::new());
    };
    let molecule = read_molecule(
        Path::new(path),
        component.format.as_deref(),
        true,
        false,
        None,
    )?;
    if molecule.atoms.is_empty() {
        return Ok(Vec::new());
    }
    let source_center = match component.placement.center_method.as_str() {
        "cog" => molecule_center_of_geometry(&molecule.atoms),
        "none" => [0.0, 0.0, 0.0],
        method => {
            return Err(anyhow!(
                "protein {} placement center_method must be cog or none, got {method}",
                component.name
            ))
        }
    };
    Ok(molecule
        .atoms
        .iter()
        .filter_map(|atom| {
            let [x, y, z] = transform_inserted_position(
                [atom.position.x, atom.position.y, atom.position.z],
                source_center,
                component.placement.rotate_degrees_xyz,
                component.placement.center_angstrom,
            );
            protein_bead_overlaps_leaflet_z(z, membrane.center_z_angstrom, &leaflet.side)
                .then_some([x, y, z])
        })
        .collect())
}

pub(super) fn xy_center_of_points(points: &[[f32; 2]]) -> [f32; 2] {
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for point in points {
        min_x = min_x.min(point[0]);
        max_x = max_x.max(point[0]);
        min_y = min_y.min(point[1]);
        max_y = max_y.max(point[1]);
    }
    [(min_x + max_x) * 0.5, (min_y + max_y) * 0.5]
}

pub(super) fn xy_radius_of_points(points: &[[f32; 2]], center: [f32; 2]) -> f32 {
    points.iter().fold(0.0, |radius, point| {
        let dx = point[0] - center[0];
        let dy = point[1] - center[1];
        radius.max((dx * dx + dy * dy).sqrt())
    })
}
