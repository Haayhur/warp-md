use super::*;

pub(super) fn combined_leaflet_exclusions(
    leaflet: &LeafletRequest,
    protein_exclusions: &[ExclusionZone],
) -> Vec<ExclusionZone> {
    let mut exclusions = leaflet.exclusions.clone();
    exclusions.extend(leaflet.regions.iter().filter_map(region_circular_exclusion));
    exclusions.extend_from_slice(protein_exclusions);
    exclusions
}

fn region_circular_exclusion(region: &LeafletRegion) -> Option<ExclusionZone> {
    if region.role != "hole" {
        return None;
    }
    let RegionGeometry::Circle {
        center_angstrom,
        radius_angstrom,
    } = &region.geometry
    else {
        return None;
    };
    Some(ExclusionZone {
        name: region.name.clone(),
        center_angstrom: *center_angstrom,
        radius_angstrom: *radius_angstrom,
    })
}

pub(super) fn normalized_lipid_beads(
    lipid: &LipidComponent,
    template: Option<&crate::build_lipids::LipidTemplate>,
) -> Result<Vec<BuildBeadTemplate>> {
    if !lipid.beads.is_empty() {
        return Ok(lipid.beads.clone());
    }
    if let Some(template) = template {
        return Ok(template
            .beads
            .iter()
            .map(|bead| BuildBeadTemplate {
                name: bead.name.clone(),
                offset_angstrom: bead.offset_angstrom,
                charge_e: bead.charge_e,
            })
            .collect());
    }
    Ok(vec![BuildBeadTemplate {
        name: bead_name_from_lipid(&lipid.lipid),
        offset_angstrom: [0.0, 0.0, 0.0],
        charge_e: lipid.charge_e.unwrap_or(0.0),
    }])
}

pub(super) fn resolve_lipid_charge(
    lipid: &LipidComponent,
    template_net_charge: f32,
    has_template: bool,
    mut beads: Vec<BuildBeadTemplate>,
) -> Result<(f32, String, Vec<BuildBeadTemplate>)> {
    if beads.is_empty() {
        return Err(anyhow!(
            "lipid {} resolved to zero beads; provide at least one bead",
            lipid.lipid
        ));
    }

    let bead_charge_sum =
        sum_bead_charges(&beads.iter().map(|bead| bead.charge_e).collect::<Vec<f32>>());

    if let Some(total_charge) = lipid.charge_e {
        if !total_charge.is_finite() {
            return Err(anyhow!("lipid {} charge_e must be finite", lipid.lipid));
        }
        if lipid.beads.is_empty() {
            let distributed = spread_total_charge(total_charge, beads.len()).ok_or_else(|| {
                anyhow!(
                    "lipid {} charge_e could not be spread over beads",
                    lipid.lipid
                )
            })?;
            for (bead, charge) in beads.iter_mut().zip(distributed) {
                bead.charge_e = charge;
            }
        } else if !charges_match(
            total_charge,
            &beads.iter().map(|bead| bead.charge_e).collect::<Vec<f32>>(),
            1.0e-5,
        ) {
            return Err(anyhow!(
                "lipid {} charge_e ({total_charge}) does not match explicit bead charge sum ({bead_charge_sum})",
                lipid.lipid
            ));
        }
        return Ok((total_charge, "request_lipid.charge_e".to_string(), beads));
    }

    if !lipid.beads.is_empty() {
        return Ok((
            bead_charge_sum,
            "request_lipid.beads.charge_e".to_string(),
            beads,
        ));
    }

    if has_template {
        return Ok((
            template_net_charge,
            "lipid_template.net_charge_e".to_string(),
            beads,
        ));
    }

    Ok((
        bead_charge_sum,
        "single_bead_fallback.charge_e".to_string(),
        beads,
    ))
}

pub(super) fn default_lipid_radius(apl_angstrom2: f32) -> f32 {
    (apl_angstrom2.sqrt() * 0.5 * 0.95).max(1.0)
}

fn bead_name_from_lipid(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .take(5)
        .collect::<String>()
        .to_ascii_uppercase()
        .if_empty_then("B")
}

pub(super) fn layout_leaflet_beads(
    system: &BuildSystem,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
    exclusions: &[ExclusionZone],
    lipids: &[ResolvedLipid],
    next_residue_id: &mut i32,
) -> Result<(
    Vec<EmittedBead>,
    PlacementMetrics,
    LeafletGeometryDiagnostics,
)> {
    let total: usize = lipids.iter().map(|lipid| lipid.count).sum();
    if total == 0 {
        return Ok((
            Vec::new(),
            PlacementMetrics::default(),
            LeafletGeometryDiagnostics::default(),
        ));
    }
    let apl = leaflet.apl_angstrom2.unwrap_or(60.0);
    if apl <= 0.0 {
        return Err(anyhow!(
            "leaflet {} apl_angstrom2 must be > 0",
            leaflet.name
        ));
    }
    let side_sign = match leaflet.side.as_str() {
        "upper" => 1.0,
        "lower" => -1.0,
        _ => {
            return Err(anyhow!(
                "leaflet {} side must be upper or lower",
                leaflet.name
            ))
        }
    };
    let lipid_sequence = leaflet_lipid_sequence(system, membrane, leaflet, lipids);
    let radii = lipid_sequence
        .iter()
        .map(|&lipid_idx| lipids[lipid_idx].radius_angstrom)
        .collect::<Vec<_>>();
    let pbc = pbc_axes(&system.pbc);
    let periodicity = LayoutPeriodicity {
        x: pbc[0],
        y: pbc[1],
    };
    let bounds = membrane_layout_bounds(system, membrane);
    let basis = membrane_layout_basis(system, membrane, bounds, periodicity);
    let mut grid = initial_leaflet_grid(
        system,
        &radii,
        apl,
        bounds,
        basis,
        periodicity,
        membrane,
        leaflet,
        proteins,
    )?;
    let exclusions = layout_exclusions(exclusions);
    let occupation_modifier = leaflet_occupation_modifier(bounds, &radii);
    let relaxation_config = RelaxationConfig {
        enabled: system.placement.relaxation,
        max_steps: system.placement.max_steps,
        push_tolerance: system.placement.push_tolerance_angstrom,
        lipid_push_multiplier: system.placement.lipid_push_multiplier,
        edge_push_multiplier: system.placement.edge_push_multiplier,
        occupation_modifier,
    };
    let boundary = protein_boundary_geometry(membrane, proteins)?;
    let metrics = if membrane_has_spatial_constraints(membrane, leaflet) {
        relax_leaflet_points_with_projector_basis(
            &mut grid,
            bounds,
            &exclusions,
            relaxation_config,
            periodicity,
            basis,
            |point| {
                if point_inside_layout_domain(&point, bounds, periodicity, basis)
                    && membrane_allows_layout_point_with_boundary(
                        leaflet,
                        boundary.as_ref(),
                        &point,
                        bounds,
                        periodicity,
                        basis,
                    )
                {
                    return None;
                }
                analytic_allowed_leaflet_projection_with_boundary(
                    point,
                    leaflet,
                    bounds,
                    periodicity,
                    basis,
                    boundary.as_ref(),
                )
            },
        )
    } else if let Some(basis) = basis {
        relax_leaflet_points_with_projector_basis(
            &mut grid,
            bounds,
            &exclusions,
            relaxation_config,
            periodicity,
            Some(basis),
            |_| None,
        )
    } else {
        relax_leaflet_points_periodic(
            &mut grid,
            bounds,
            &exclusions,
            relaxation_config,
            periodicity,
        )
    };
    confine_points_to_protein_boundary(&mut grid, membrane, proteins)?;
    confine_points_to_allowed_leaflet_regions(
        &mut grid,
        membrane,
        leaflet,
        proteins,
        bounds,
        periodicity,
        basis,
    )?;
    let geometry = leaflet_geometry_diagnostics(
        membrane,
        leaflet,
        proteins,
        &grid,
        bounds,
        periodicity,
        basis,
    )?;
    let z = membrane.center_z_angstrom + side_sign * 15.0;
    let mut out = Vec::new();

    for (&lipid_idx, &point) in lipid_sequence.iter().zip(grid.iter()) {
        let lipid = &lipids[lipid_idx];
        let residue_id = *next_residue_id;
        *next_residue_id += 1;
        for bead in &lipid.beads {
            out.push(EmittedBead {
                residue_id,
                residue_name: lipid.name.clone(),
                atom_name: bead.name.clone(),
                charge_e: bead.charge_e,
                position_angstrom: [
                    point.x + bead.offset_angstrom[0],
                    point.y + bead.offset_angstrom[1],
                    z + side_sign * bead.offset_angstrom[2],
                ],
                excluded_volume_factor: 1.0,
            });
        }
    }
    Ok((out, metrics, geometry))
}
