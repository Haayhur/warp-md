use super::build_contract_validation_fields::{
    validate_generated_solvent_name, validate_inserted_component_beads,
    validate_inserted_component_placement, validate_inserted_component_topology_fields,
    validate_inserted_orientation_mode, validate_leaflet_region, validate_outputs,
    validate_solvent_component,
};
use super::*;

pub(super) fn validate_request(request: BuildRequest) -> Result<BuildRequest> {
    if request.schema_version != BUILD_SCHEMA_VERSION {
        return Err(anyhow!(
            "schema_version must be {BUILD_SCHEMA_VERSION}, got {}",
            request.schema_version
        ));
    }
    if request.mode != "membrane" {
        return Err(anyhow!("build mode must be membrane"));
    }
    if request
        .system
        .box_size_angstrom
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(anyhow!(
            "system.box_size_angstrom values must be finite and > 0"
        ));
    }
    validate_box_contract(&request.system)?;
    if !matches!(
        request.system.placement.mode.as_str(),
        "deterministic" | "seeded"
    ) {
        return Err(anyhow!(
            "system.placement.mode must be deterministic or seeded"
        ));
    }
    if !matches!(
        request.system.placement.candidate_source.as_str(),
        "grid" | "random"
    ) {
        return Err(anyhow!(
            "system.placement.candidate_source must be grid or random"
        ));
    }
    if request.system.placement.mode == "seeded" && request.system.placement.random_seed.is_none() {
        return Err(anyhow!(
            "system.placement.random_seed is required when system.placement.mode is seeded"
        ));
    }
    if request.system.placement.candidate_source == "random"
        && request.system.placement.mode != "seeded"
    {
        return Err(anyhow!(
            "system.placement.candidate_source random requires system.placement.mode seeded"
        ));
    }
    if request.system.placement.max_steps == 0 && request.system.placement.relaxation {
        return Err(anyhow!(
            "system.placement.max_steps must be > 0 when relaxation is enabled"
        ));
    }
    if request.system.placement.push_tolerance_angstrom <= 0.0 {
        return Err(anyhow!(
            "system.placement.push_tolerance_angstrom must be > 0"
        ));
    }
    if request.system.placement.lipid_push_multiplier < 0.0 {
        return Err(anyhow!(
            "system.placement.lipid_push_multiplier must be >= 0"
        ));
    }
    if request.system.placement.edge_push_multiplier < 0.0 {
        return Err(anyhow!(
            "system.placement.edge_push_multiplier must be >= 0"
        ));
    }
    if request.membranes.is_empty()
        && request.proteins.is_empty()
        && request.solutes.is_empty()
        && !request.environment.solvent.enabled
    {
        return Err(anyhow!(
            "build request must contain membranes, inserted components, or enabled solvent"
        ));
    }
    validate_outputs(&request.outputs)?;
    for (component_idx, component) in request.proteins.iter().enumerate() {
        if component.count == 0 {
            return Err(anyhow!("proteins[{component_idx}].count must be > 0"));
        }
        if component
            .coordinates
            .as_ref()
            .is_some_and(|path| path.trim().is_empty())
        {
            return Err(anyhow!(
                "proteins[{component_idx}].coordinates must not be empty"
            ));
        }
        if component
            .format
            .as_ref()
            .is_some_and(|format| format.trim().is_empty())
        {
            return Err(anyhow!(
                "proteins[{component_idx}].format must not be empty"
            ));
        }
        if component
            .definition
            .as_ref()
            .is_some_and(|path| path.trim().is_empty())
        {
            return Err(anyhow!(
                "proteins[{component_idx}].definition must not be empty"
            ));
        }
        validate_inserted_component_topology_fields("proteins", component_idx, component)?;
        validate_inserted_component_beads("proteins", component_idx, component)?;
        validate_inserted_component_placement("proteins", component_idx, component)?;
        validate_inserted_orientation_mode(
            "proteins",
            component_idx,
            component,
            &request.system.placement,
        )?;
        if let Some(footprint) = &component.footprint {
            if footprint.buffer_angstrom < 0.0 || !footprint.buffer_angstrom.is_finite() {
                return Err(anyhow!(
                    "proteins[{component_idx}].footprint.buffer_angstrom must be finite and >= 0"
                ));
            }
            if footprint
                .radius_angstrom
                .is_some_and(|radius| radius <= 0.0 || !radius.is_finite())
            {
                return Err(anyhow!(
                    "proteins[{component_idx}].footprint.radius_angstrom must be finite and > 0"
                ));
            }
            if footprint
                .center_angstrom
                .is_some_and(|center| center.iter().any(|value| !value.is_finite()))
            {
                return Err(anyhow!(
                    "proteins[{component_idx}].footprint.center_angstrom values must be finite"
                ));
            }
            if footprint.center_angstrom.is_some()
                && footprint.radius_angstrom.is_none()
                && component.coordinates.is_none()
            {
                return Err(anyhow!(
                    "proteins[{component_idx}].footprint.radius_angstrom is required when center_angstrom is provided without coordinates"
                ));
            }
        }
    }
    for (component_idx, component) in request.solutes.iter().enumerate() {
        if component.count == 0 {
            return Err(anyhow!("solutes[{component_idx}].count must be > 0"));
        }
        if component
            .coordinates
            .as_ref()
            .is_some_and(|path| path.trim().is_empty())
        {
            return Err(anyhow!(
                "solutes[{component_idx}].coordinates must not be empty"
            ));
        }
        if component
            .format
            .as_ref()
            .is_some_and(|format| format.trim().is_empty())
        {
            return Err(anyhow!("solutes[{component_idx}].format must not be empty"));
        }
        if component
            .definition
            .as_ref()
            .is_some_and(|path| path.trim().is_empty())
        {
            return Err(anyhow!(
                "solutes[{component_idx}].definition must not be empty"
            ));
        }
        validate_inserted_component_topology_fields("solutes", component_idx, component)?;
        validate_inserted_component_beads("solutes", component_idx, component)?;
        validate_inserted_component_placement("solutes", component_idx, component)?;
        validate_inserted_orientation_mode(
            "solutes",
            component_idx,
            component,
            &request.system.placement,
        )?;
    }
    for (memb_idx, membrane) in request.membranes.iter().enumerate() {
        if membrane.leaflets.is_empty() {
            return Err(anyhow!("membranes[{memb_idx}].leaflets cannot be empty"));
        }
        if membrane
            .center_xy_angstrom
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(anyhow!(
                "membranes[{memb_idx}].center_xy_angstrom values must be finite"
            ));
        }
        if membrane.size_xy_angstrom.is_some_and(|values| {
            values
                .iter()
                .any(|value| *value <= 0.0 || !value.is_finite())
        }) {
            return Err(anyhow!(
                "membranes[{memb_idx}].size_xy_angstrom values must be finite and > 0"
            ));
        }
        if membrane.solvent_exclusion_half_thickness_angstrom <= 0.0
            || !membrane
                .solvent_exclusion_half_thickness_angstrom
                .is_finite()
        {
            return Err(anyhow!(
                "membranes[{memb_idx}].solvent_exclusion_half_thickness_angstrom must be finite and > 0"
            ));
        }
        if let Some(boundary) = &membrane.protein_boundary {
            if boundary.mode != "inside" {
                return Err(anyhow!(
                    "membranes[{memb_idx}].protein_boundary.mode must be inside"
                ));
            }
            if !matches!(
                boundary.geometry.as_str(),
                "circle" | "convex_hull" | "concave_hull" | "alpha_shape"
            ) {
                return Err(anyhow!(
                    "membranes[{memb_idx}].protein_boundary.geometry must be circle, convex_hull, concave_hull, or alpha_shape"
                ));
            }
            if !matches!(
                boundary.radius_strategy.as_str(),
                "outer" | "radial_quantile"
            ) {
                return Err(anyhow!(
                    "membranes[{memb_idx}].protein_boundary.radius_strategy must be outer or radial_quantile"
                ));
            }
            if boundary
                .radius_quantile
                .is_some_and(|quantile| !(0.0..=1.0).contains(&quantile) || !quantile.is_finite())
            {
                return Err(anyhow!(
                    "membranes[{memb_idx}].protein_boundary.radius_quantile must be finite and between 0 and 1"
                ));
            }
            if boundary
                .protein
                .as_ref()
                .is_some_and(|name| name.trim().is_empty())
            {
                return Err(anyhow!(
                    "membranes[{memb_idx}].protein_boundary.protein must not be empty"
                ));
            }
            if boundary
                .center_angstrom
                .is_some_and(|center| center.iter().any(|value| !value.is_finite()))
            {
                return Err(anyhow!(
                    "membranes[{memb_idx}].protein_boundary.center_angstrom values must be finite"
                ));
            }
            if boundary
                .radius_angstrom
                .is_some_and(|radius| radius <= 0.0 || !radius.is_finite())
            {
                return Err(anyhow!(
                    "membranes[{memb_idx}].protein_boundary.radius_angstrom must be finite and > 0"
                ));
            }
            if boundary
                .alpha_radius_angstrom
                .is_some_and(|radius| radius <= 0.0 || !radius.is_finite())
            {
                return Err(anyhow!(
                    "membranes[{memb_idx}].protein_boundary.alpha_radius_angstrom must be finite and > 0"
                ));
            }
            if boundary.buffer_angstrom < 0.0 || !boundary.buffer_angstrom.is_finite() {
                return Err(anyhow!(
                    "membranes[{memb_idx}].protein_boundary.buffer_angstrom must be finite and >= 0"
                ));
            }
            if boundary.bead_exclusion_radius_angstrom < 0.0
                || !boundary.bead_exclusion_radius_angstrom.is_finite()
            {
                return Err(anyhow!(
                    "membranes[{memb_idx}].protein_boundary.bead_exclusion_radius_angstrom must be finite and >= 0"
                ));
            }
        }
        for (leaf_idx, leaflet) in membrane.leaflets.iter().enumerate() {
            if !matches!(leaflet.side.as_str(), "upper" | "lower") {
                return Err(anyhow!(
                    "membranes[{memb_idx}].leaflets[{leaf_idx}].side must be upper or lower"
                ));
            }
            if leaflet.composition.is_empty() {
                return Err(anyhow!(
                    "membranes[{memb_idx}].leaflets[{leaf_idx}].composition cannot be empty"
                ));
            }
            for (exclusion_idx, exclusion) in leaflet.exclusions.iter().enumerate() {
                if exclusion.radius_angstrom <= 0.0 || !exclusion.radius_angstrom.is_finite() {
                    return Err(anyhow!(
                        "membranes[{memb_idx}].leaflets[{leaf_idx}].exclusions[{exclusion_idx}].radius_angstrom must be finite and > 0"
                    ));
                }
                if exclusion
                    .center_angstrom
                    .iter()
                    .any(|value| !value.is_finite())
                {
                    return Err(anyhow!(
                        "membranes[{memb_idx}].leaflets[{leaf_idx}].exclusions[{exclusion_idx}].center_angstrom values must be finite"
                    ));
                }
            }
            for (region_idx, region) in leaflet.regions.iter().enumerate() {
                validate_leaflet_region(memb_idx, leaf_idx, region_idx, region)?;
            }
            for (lipid_idx, lipid) in leaflet.composition.iter().enumerate() {
                if lipid.count == Some(0) {
                    return Err(anyhow!(
                        "membranes[{memb_idx}].leaflets[{leaf_idx}].composition[{lipid_idx}].count must be > 0"
                    ));
                }
                if lipid.fraction.is_some_and(|fraction| fraction < 0.0) {
                    return Err(anyhow!(
                        "membranes[{memb_idx}].leaflets[{leaf_idx}].composition[{lipid_idx}].fraction must be >= 0"
                    ));
                }
                if lipid
                    .radius_angstrom
                    .is_some_and(|radius| radius <= 0.0 || !radius.is_finite())
                {
                    return Err(anyhow!(
                        "membranes[{memb_idx}].leaflets[{leaf_idx}].composition[{lipid_idx}].radius_angstrom must be finite and > 0"
                    ));
                }
                if lipid.beads.iter().any(|bead| bead.name.trim().is_empty()) {
                    return Err(anyhow!(
                        "membranes[{memb_idx}].leaflets[{leaf_idx}].composition[{lipid_idx}].beads names cannot be empty"
                    ));
                }
            }
        }
    }
    if request.environment.ions.cation_charge_e <= 0 {
        return Err(anyhow!("environment.ions.cation_charge_e must be positive"));
    }
    if request.environment.ions.anion_charge_e >= 0 {
        return Err(anyhow!("environment.ions.anion_charge_e must be negative"));
    }
    for (idx, ion) in request.environment.ions.cations.iter().enumerate() {
        if ion.name.trim().is_empty() {
            return Err(anyhow!(
                "environment.ions.cations[{idx}].name must not be empty"
            ));
        }
        if ion.ratio <= 0.0 || !ion.ratio.is_finite() {
            return Err(anyhow!(
                "environment.ions.cations[{idx}].ratio must be finite and > 0"
            ));
        }
        let resolved_charge = lookup_ion_library(&ion.name)
            .filter(|_| ion.charge_e == 0)
            .map(|entry| entry.charge_e)
            .unwrap_or(ion.charge_e);
        if resolved_charge <= 0 {
            return Err(anyhow!(
                "environment.ions.cations[{idx}].charge_e must be positive"
            ));
        }
    }
    for (idx, ion) in request.environment.ions.anions.iter().enumerate() {
        if ion.name.trim().is_empty() {
            return Err(anyhow!(
                "environment.ions.anions[{idx}].name must not be empty"
            ));
        }
        if ion.ratio <= 0.0 || !ion.ratio.is_finite() {
            return Err(anyhow!(
                "environment.ions.anions[{idx}].ratio must be finite and > 0"
            ));
        }
        let resolved_charge = lookup_ion_library(&ion.name)
            .filter(|_| ion.charge_e == 0)
            .map(|entry| entry.charge_e)
            .unwrap_or(ion.charge_e);
        if resolved_charge >= 0 {
            return Err(anyhow!(
                "environment.ions.anions[{idx}].charge_e must be negative"
            ));
        }
    }
    if !matches!(
        request.environment.ions.salt_method.as_str(),
        "add" | "remove" | "mean"
    ) {
        return Err(anyhow!(
            "environment.ions.salt_method must be add, remove, or mean"
        ));
    }
    if request.environment.ions.salt_molarity_mol_l < 0.0
        || !request.environment.ions.salt_molarity_mol_l.is_finite()
    {
        return Err(anyhow!(
            "environment.ions.salt_molarity_mol_l must be finite and >= 0"
        ));
    }
    let solvent = &request.environment.solvent;
    if solvent.enabled {
        if solvent.name.trim().is_empty() {
            return Err(anyhow!("environment.solvent.name must not be empty"));
        }
        validate_generated_solvent_name("environment.solvent.name", &solvent.name)?;
        if solvent.molarity_mol_l < 0.0 || !solvent.molarity_mol_l.is_finite() {
            return Err(anyhow!(
                "environment.solvent.molarity_mol_l must be finite and >= 0"
            ));
        }
        if solvent.mapping_ratio <= 0.0 || !solvent.mapping_ratio.is_finite() {
            return Err(anyhow!(
                "environment.solvent.mapping_ratio must be finite and > 0"
            ));
        }
        if solvent.molar_mass_g_mol <= 0.0 || !solvent.molar_mass_g_mol.is_finite() {
            return Err(anyhow!(
                "environment.solvent.molar_mass_g_mol must be finite and > 0"
            ));
        }
        if solvent.density_kg_m3 <= 0.0 || !solvent.density_kg_m3.is_finite() {
            return Err(anyhow!(
                "environment.solvent.density_kg_m3 must be finite and > 0"
            ));
        }
        if solvent.box_size_angstrom.is_some_and(|values| {
            values
                .iter()
                .any(|value| *value <= 0.0 || !value.is_finite())
        }) {
            return Err(anyhow!(
                "environment.solvent.box_size_angstrom values must be finite and > 0"
            ));
        }
        if solvent
            .center_angstrom
            .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
        {
            return Err(anyhow!(
                "environment.solvent.center_angstrom values must be finite"
            ));
        }
        if solvent.excluded_bead_radius_angstrom <= 0.0
            || !solvent.excluded_bead_radius_angstrom.is_finite()
        {
            return Err(anyhow!(
                "environment.solvent.excluded_bead_radius_angstrom must be finite and > 0"
            ));
        }
        if solvent.grid_spacing_angstrom <= 0.0 || !solvent.grid_spacing_angstrom.is_finite() {
            return Err(anyhow!(
                "environment.solvent.grid_spacing_angstrom must be finite and > 0"
            ));
        }
        if solvent.exclusion_buffer_angstrom < 0.0 || !solvent.exclusion_buffer_angstrom.is_finite()
        {
            return Err(anyhow!(
                "environment.solvent.exclusion_buffer_angstrom must be finite and >= 0"
            ));
        }
        if solvent
            .solvent_per_lipid
            .is_some_and(|value| value <= 0.0 || !value.is_finite())
        {
            return Err(anyhow!(
                "environment.solvent.solvent_per_lipid must be finite and > 0"
            ));
        }
        if !(0.0..=1.0).contains(&solvent.solvent_per_lipid_cutoff)
            || !solvent.solvent_per_lipid_cutoff.is_finite()
        {
            return Err(anyhow!(
                "environment.solvent.solvent_per_lipid_cutoff must be finite and between 0 and 1"
            ));
        }
        for (idx, species) in solvent.species.iter().enumerate() {
            validate_solvent_component(&format!("environment.solvent.species[{idx}]"), species)?;
        }
        for (zone_idx, zone) in solvent.zones.iter().enumerate() {
            let path = format!("environment.solvent.zones[{zone_idx}]");
            if zone
                .name
                .as_ref()
                .is_some_and(|name| name.trim().is_empty())
            {
                return Err(anyhow!("{path}.name must not be empty"));
            }
            if zone.box_size_angstrom.is_some_and(|values| {
                values
                    .iter()
                    .any(|value| *value <= 0.0 || !value.is_finite())
            }) {
                return Err(anyhow!(
                    "{path}.box_size_angstrom values must be finite and > 0"
                ));
            }
            if zone
                .center_angstrom
                .is_some_and(|values| values.iter().any(|value| !value.is_finite()))
            {
                return Err(anyhow!("{path}.center_angstrom values must be finite"));
            }
            if zone
                .molarity_mol_l
                .is_some_and(|value| value < 0.0 || !value.is_finite())
            {
                return Err(anyhow!("{path}.molarity_mol_l must be finite and >= 0"));
            }
            if zone
                .salt_molarity_mol_l
                .is_some_and(|value| value < 0.0 || !value.is_finite())
            {
                return Err(anyhow!(
                    "{path}.salt_molarity_mol_l must be finite and >= 0"
                ));
            }
            if zone
                .solvent_per_lipid
                .is_some_and(|value| value <= 0.0 || !value.is_finite())
            {
                return Err(anyhow!("{path}.solvent_per_lipid must be finite and > 0"));
            }
            if zone
                .solvent_per_lipid_cutoff
                .is_some_and(|value| !(0.0..=1.0).contains(&value) || !value.is_finite())
            {
                return Err(anyhow!(
                    "{path}.solvent_per_lipid_cutoff must be finite and between 0 and 1"
                ));
            }
            for (species_idx, species) in zone.species.iter().enumerate() {
                validate_solvent_component(&format!("{path}.species[{species_idx}]"), species)?;
            }
        }
    }
    Ok(request)
}
