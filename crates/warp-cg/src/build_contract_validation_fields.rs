use super::*;

pub(super) fn validate_outputs(outputs: &BuildOutputs) -> Result<()> {
    for (name, path) in [
        ("outputs.coordinates", outputs.coordinates.as_ref()),
        ("outputs.gro", outputs.gro.as_ref()),
        ("outputs.pdb", outputs.pdb.as_ref()),
        ("outputs.cif", outputs.cif.as_ref()),
        ("outputs.topology", outputs.topology.as_ref()),
        ("outputs.log", outputs.log.as_ref()),
        ("outputs.snapshot", outputs.snapshot.as_ref()),
        ("outputs.manifest", Some(&outputs.manifest)),
    ] {
        if path.is_some_and(|path| path.trim().is_empty()) {
            return Err(anyhow!("{name} must not be empty"));
        }
    }
    if outputs.backup_existing && !outputs.overwrite {
        return Err(anyhow!(
            "outputs.backup_existing requires outputs.overwrite to be true"
        ));
    }
    Ok(())
}

pub(super) fn validate_solvent_component(path: &str, species: &SolventComponent) -> Result<()> {
    if species.name.trim().is_empty() {
        return Err(anyhow!("{path}.name must not be empty"));
    }
    validate_generated_solvent_name(&format!("{path}.name"), &species.name)?;
    if species.ratio <= 0.0 || !species.ratio.is_finite() {
        return Err(anyhow!("{path}.ratio must be finite and > 0"));
    }
    if species.mapping_ratio <= 0.0 || !species.mapping_ratio.is_finite() {
        return Err(anyhow!("{path}.mapping_ratio must be finite and > 0"));
    }
    if species.molar_mass_g_mol <= 0.0 || !species.molar_mass_g_mol.is_finite() {
        return Err(anyhow!("{path}.molar_mass_g_mol must be finite and > 0"));
    }
    if species.density_kg_m3 <= 0.0 || !species.density_kg_m3.is_finite() {
        return Err(anyhow!("{path}.density_kg_m3 must be finite and > 0"));
    }
    if !species.charge_e.is_finite() {
        return Err(anyhow!("{path}.charge_e must be finite"));
    }
    Ok(())
}

pub(super) fn validate_generated_solvent_name(path: &str, name: &str) -> Result<()> {
    if let Some((family, _)) = split_tailcode_solvent_name(name) {
        if is_tailcode_solvent_family(family) && lookup_tailcode_solvent_library(name).is_none() {
            return Err(anyhow!(
                "{path} has an invalid generated solvent tail code; expected hydrocarbon:<tailcode>, fattyacid:<tailcode>, monoglyceride:<tailcode>, diglyceride:<tailcode>,<tailcode>, triglyceride:<tailcode>,<tailcode>,<tailcode>, bmp2:<tailcode>,<tailcode>, bmp3:<tailcode>,<tailcode>, cardiolipin:<tailcode>,<tailcode>,<tailcode>,<tailcode>, or sphingolipid:<head>,<tailcode>,<tailcode> with C/c/D/T/t/F codes"
            ));
        }
    }
    Ok(())
}

pub(super) fn split_tailcode_solvent_name(name: &str) -> Option<(&str, &str)> {
    let trimmed = name.trim();
    trimmed
        .split_once(':')
        .or_else(|| trimmed.split_once('/'))
        .or_else(|| trimmed.split_once('_'))
        .or_else(|| trimmed.split_once('-'))
}

fn is_tailcode_solvent_family(family: &str) -> bool {
    matches!(
        family.trim().to_ascii_lowercase().as_str(),
        "hc" | "hydrocarbon"
            | "fa"
            | "fattyacid"
            | "fatty-acid"
            | "mg"
            | "monoglyceride"
            | "mono-glyceride"
            | "dg"
            | "diglyceride"
            | "di-glyceride"
            | "tg"
            | "triglyceride"
            | "tri-glyceride"
            | "bmp2"
            | "bmp3"
            | "cl"
            | "cardiolipin"
            | "cardio-lipin"
            | "sm"
            | "sphingolipid"
            | "sphingo-lipid"
    )
}

pub(super) fn validate_leaflet_region(
    memb_idx: usize,
    leaf_idx: usize,
    region_idx: usize,
    region: &LeafletRegion,
) -> Result<()> {
    if !matches!(region.role.as_str(), "hole" | "patch") {
        return Err(anyhow!(
            "membranes[{memb_idx}].leaflets[{leaf_idx}].regions[{region_idx}].role must be hole or patch"
        ));
    }
    let path = format!("membranes[{memb_idx}].leaflets[{leaf_idx}].regions[{region_idx}]");
    match &region.geometry {
        RegionGeometry::Circle {
            center_angstrom,
            radius_angstrom,
        } => {
            validate_region_center(&path, center_angstrom)?;
            if *radius_angstrom <= 0.0 || !radius_angstrom.is_finite() {
                return Err(anyhow!("{path}.radius_angstrom must be finite and > 0"));
            }
        }
        RegionGeometry::Ellipse {
            center_angstrom,
            radius_angstrom,
            rotate_degrees,
        } => {
            validate_region_center(&path, center_angstrom)?;
            if radius_angstrom
                .iter()
                .any(|value| *value <= 0.0 || !value.is_finite())
            {
                return Err(anyhow!(
                    "{path}.radius_angstrom values must be finite and > 0"
                ));
            }
            if !rotate_degrees.is_finite() {
                return Err(anyhow!("{path}.rotate_degrees must be finite"));
            }
        }
        RegionGeometry::Rectangle {
            center_angstrom,
            size_angstrom,
            rotate_degrees,
        } => {
            validate_region_center(&path, center_angstrom)?;
            if size_angstrom
                .iter()
                .any(|value| *value <= 0.0 || !value.is_finite())
            {
                return Err(anyhow!(
                    "{path}.size_angstrom values must be finite and > 0"
                ));
            }
            if !rotate_degrees.is_finite() {
                return Err(anyhow!("{path}.rotate_degrees must be finite"));
            }
        }
        RegionGeometry::Polygon {
            points_angstrom,
            scale_xy,
            rotate_degrees,
        } => {
            if points_angstrom.len() < 3 {
                return Err(anyhow!(
                    "{path}.points_angstrom must contain at least three points"
                ));
            }
            if points_angstrom
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
            {
                return Err(anyhow!("{path}.points_angstrom values must be finite"));
            }
            if scale_xy.is_some_and(|values| {
                values
                    .iter()
                    .any(|value| *value <= 0.0 || !value.is_finite())
            }) {
                return Err(anyhow!("{path}.scale_xy values must be finite and > 0"));
            }
            if !rotate_degrees.is_finite() {
                return Err(anyhow!("{path}.rotate_degrees must be finite"));
            }
        }
    }
    Ok(())
}

fn validate_region_center(path: &str, center: &[f32; 2]) -> Result<()> {
    if center.iter().any(|value| !value.is_finite()) {
        return Err(anyhow!("{path}.center_angstrom values must be finite"));
    }
    Ok(())
}

pub(super) fn validate_inserted_component_placement(
    collection: &str,
    component_idx: usize,
    component: &InsertedComponent,
) -> Result<()> {
    if component
        .placement
        .center_angstrom
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(anyhow!(
            "{collection}[{component_idx}].placement.center_angstrom values must be finite"
        ));
    }
    if !matches!(component.placement.center_method.as_str(), "cog" | "none") {
        return Err(anyhow!(
            "{collection}[{component_idx}].placement.center_method must be cog or none"
        ));
    }
    if !matches!(
        component.placement.orientation.as_str(),
        "fixed" | "seeded_random"
    ) {
        return Err(anyhow!(
            "{collection}[{component_idx}].placement.orientation must be fixed or seeded_random"
        ));
    }
    if component
        .placement
        .rotate_degrees_xyz
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(anyhow!(
            "{collection}[{component_idx}].placement.rotate_degrees_xyz values must be finite"
        ));
    }
    Ok(())
}

pub(super) fn validate_inserted_orientation_mode(
    collection: &str,
    component_idx: usize,
    component: &InsertedComponent,
    placement: &PlacementOptions,
) -> Result<()> {
    if component.placement.orientation == "seeded_random" && placement.mode != "seeded" {
        return Err(anyhow!(
            "{collection}[{component_idx}].placement.orientation seeded_random requires system.placement.mode seeded"
        ));
    }
    Ok(())
}

pub(super) fn validate_inserted_component_beads(
    collection: &str,
    component_idx: usize,
    component: &InsertedComponent,
) -> Result<()> {
    for (bead_idx, bead) in component.beads.iter().enumerate() {
        if bead.name.trim().is_empty() {
            return Err(anyhow!(
                "{collection}[{component_idx}].beads[{bead_idx}].name must not be empty"
            ));
        }
        if bead.offset_angstrom.iter().any(|value| !value.is_finite()) {
            return Err(anyhow!(
                "{collection}[{component_idx}].beads[{bead_idx}].offset_angstrom values must be finite"
            ));
        }
        if !bead.charge_e.is_finite() {
            return Err(anyhow!(
                "{collection}[{component_idx}].beads[{bead_idx}].charge_e must be finite"
            ));
        }
    }
    Ok(())
}

pub(super) fn validate_inserted_component_topology_fields(
    collection: &str,
    component_idx: usize,
    component: &InsertedComponent,
) -> Result<()> {
    if component
        .charge_topology
        .as_ref()
        .is_some_and(|path| path.trim().is_empty())
    {
        return Err(anyhow!(
            "{collection}[{component_idx}].charge_topology must not be empty"
        ));
    }
    if component
        .charge_topologies
        .iter()
        .any(|path| path.trim().is_empty())
    {
        return Err(anyhow!(
            "{collection}[{component_idx}].charge_topologies entries must not be empty"
        ));
    }
    if component
        .charge_molecule_type
        .as_ref()
        .is_some_and(|name| name.trim().is_empty())
    {
        return Err(anyhow!(
            "{collection}[{component_idx}].charge_molecule_type must not be empty"
        ));
    }
    if component
        .molecule_types
        .iter()
        .any(|name| name.trim().is_empty())
    {
        return Err(anyhow!(
            "{collection}[{component_idx}].molecule_types entries must not be empty"
        ));
    }
    if !component.charge_topologies.is_empty()
        && component.charge_topologies.len() != inserted_molecule_types(component).len()
    {
        return Err(anyhow!(
            "{collection}[{component_idx}].charge_topologies length must match resolved molecule_types length"
        ));
    }
    Ok(())
}
