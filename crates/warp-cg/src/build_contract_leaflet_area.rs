use super::build_contract_region_area::{
    region_area_estimate, region_union_area_angstrom2, RegionAreaEstimate,
};
use super::*;

pub(super) fn resolve_leaflet_lipid_counts(
    system: &BuildSystem,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
) -> Result<(Vec<ResolvedLipid>, LeafletAreaSummary)> {
    let explicit_total: usize = leaflet
        .composition
        .iter()
        .filter_map(|lipid| lipid.count)
        .sum();
    let mut area_summary = LeafletAreaSummary {
        available_area_angstrom2: None,
        method: "explicit_counts".to_string(),
        is_exact: true,
        reported_error_bound_angstrom2: None,
    };
    let target_total = if explicit_total > 0 {
        explicit_total
    } else {
        let apl = leaflet.apl_angstrom2.unwrap_or(60.0);
        if apl <= 0.0 {
            return Err(anyhow!(
                "leaflet {} apl_angstrom2 must be > 0",
                leaflet.name
            ));
        }
        let area_estimate = available_leaflet_area_angstrom2(system, membrane, leaflet, proteins)?;
        area_summary = area_estimate.summary;
        let area = area_estimate.area_angstrom2;
        (area / apl + 0.5).floor() as usize
    }
    .max(leaflet.composition.len());

    let mut resolved = Vec::with_capacity(leaflet.composition.len());
    let fraction_sum: f32 = leaflet
        .composition
        .iter()
        .map(|lipid| lipid.fraction.unwrap_or(0.0))
        .sum();
    let counts = resolve_component_counts(&leaflet.composition, target_total, fraction_sum);

    for (idx, lipid) in leaflet.composition.iter().enumerate() {
        let count = counts[idx];
        let template = lookup_lipid_template(&lipid.lipid, &system.force_field);
        let template_net_charge = template
            .as_ref()
            .map(|template| template.net_charge_e)
            .unwrap_or(0.0);
        let template_radius = template.as_ref().map(|template| template.radius_angstrom);
        let template_source = template
            .as_ref()
            .map(|template| template.source.to_string())
            .unwrap_or_else(|| "single_bead_fallback".to_string());
        let beads = normalized_lipid_beads(lipid, template.as_ref())?;
        let (charge_e, charge_source, beads) =
            resolve_lipid_charge(lipid, template_net_charge, template.is_some(), beads)?;
        resolved.push(ResolvedLipid {
            name: lipid.lipid.clone(),
            count,
            charge_e,
            radius_angstrom: lipid
                .radius_angstrom
                .or(template_radius)
                .unwrap_or_else(|| default_lipid_radius(leaflet.apl_angstrom2.unwrap_or(60.0))),
            beads,
            template_source,
            charge_source,
        });
    }

    Ok((resolved, area_summary))
}

#[derive(Clone, Debug)]
struct LeafletAreaEstimate {
    area_angstrom2: f32,
    summary: LeafletAreaSummary,
}

fn available_leaflet_area_angstrom2(
    system: &BuildSystem,
    membrane: &MembraneRequest,
    leaflet: &LeafletRequest,
    proteins: &[InsertedComponent],
) -> Result<LeafletAreaEstimate> {
    let bounds = membrane_layout_bounds(system, membrane);
    let base_area = (bounds.xmax - bounds.xmin) * (bounds.ymax - bounds.ymin);
    let boundary_area_estimate =
        protein_boundary_geometry(membrane, proteins)?.map(|boundary| boundary.area_estimate());
    let patch_regions = leaflet
        .regions
        .iter()
        .filter(|region| region.role == "patch")
        .collect::<Vec<_>>();
    let patch_estimate = if patch_regions.is_empty() {
        RegionAreaEstimate::zero("base_layout_area")
    } else if patch_regions.len() > 1 {
        region_union_area_angstrom2(&patch_regions, bounds)
    } else {
        region_area_estimate(patch_regions[0])
    };
    let patch_area = patch_estimate.area_angstrom2;
    let mut starting_area = if patch_area > 0.0 {
        patch_area.min(base_area)
    } else {
        base_area
    };
    if let Some((boundary_area, _)) = boundary_area_estimate {
        starting_area = starting_area.min(boundary_area);
    }
    let legacy_circular_holes = leaflet
        .exclusions
        .iter()
        .map(|zone| std::f32::consts::PI * zone.radius_angstrom.powi(2))
        .sum::<f32>();
    let region_holes = leaflet
        .regions
        .iter()
        .filter(|region| region.role == "hole")
        .collect::<Vec<_>>();
    let region_hole_estimate = if region_holes.is_empty() {
        RegionAreaEstimate::zero("no_region_holes")
    } else if region_holes.len() > 1 {
        region_union_area_angstrom2(&region_holes, bounds)
    } else {
        region_area_estimate(region_holes[0])
    };
    let protein_area = protein_footprint_area_for_leaflet(membrane, leaflet, proteins)?;
    let boundary_bead_area =
        boundary_protein_exclusion_area_for_leaflet(bounds, membrane, leaflet, proteins)?;
    let available_area = (starting_area
        - legacy_circular_holes
        - region_hole_estimate.area_angstrom2
        - protein_area
        - boundary_bead_area)
        .max(0.0);
    let method = if !patch_regions.is_empty() && !region_holes.is_empty() {
        format!(
            "patch:{};hole:{}",
            patch_estimate.method, region_hole_estimate.method
        )
    } else if !patch_regions.is_empty() {
        format!("patch:{}", patch_estimate.method)
    } else if !region_holes.is_empty() {
        format!("hole:{}", region_hole_estimate.method)
    } else if boundary_area_estimate.is_some() {
        "protein_boundary_area".to_string()
    } else {
        "base_layout_area".to_string()
    };
    let error_bound = [
        patch_estimate.reported_error_bound_angstrom2,
        region_hole_estimate.reported_error_bound_angstrom2,
    ]
    .into_iter()
    .flatten()
    .reduce(|left, right| left + right);
    let area_is_exact = patch_estimate.is_exact
        && region_hole_estimate.is_exact
        && boundary_area_estimate.is_none_or(|(_, is_exact)| is_exact)
        && error_bound.unwrap_or(0.0) == 0.0;
    Ok(LeafletAreaEstimate {
        area_angstrom2: available_area,
        summary: LeafletAreaSummary {
            available_area_angstrom2: Some(available_area),
            method,
            is_exact: area_is_exact,
            reported_error_bound_angstrom2: error_bound,
        },
    })
}
