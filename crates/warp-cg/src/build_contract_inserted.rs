use super::build_contract_inserted_library::{
    inserted_solvent_library_beads, inserted_solvent_library_charge_estimate,
};
use super::build_contract_inserted_placement::{
    accumulate_inserted_flood_summary, inserted_copy_centers,
};
use super::*;

pub(super) fn append_inserted_charge(
    component: &InsertedComponent,
    component_charges: &mut Vec<ComponentChargeSummary>,
    shared_components: &mut Vec<ComponentCharge>,
    charge_sources: &mut Vec<String>,
) -> Result<()> {
    let resolved = resolve_inserted_charge(component)?;
    let source = resolved
        .source
        .clone()
        .or_else(|| component.charge_source.clone())
        .unwrap_or_else(|| "inserted_component.net_charge_e".to_string());
    component_charges.push(ComponentChargeSummary {
        name: component.name.clone(),
        count: component.count,
        per_instance_net_charge_e: resolved.net_charge_e,
        per_instance_bead_charge_sum_e: None,
        charge_balance_delta_e: None,
        total_charge_e: resolved
            .net_charge_e
            .map(|charge| charge * component.count as f32),
        source: source.clone(),
    });
    shared_components.push(ComponentCharge {
        name: component.name.clone(),
        count: component.count,
        per_instance_net_charge_e: resolved.net_charge_e,
    });
    if resolved.net_charge_e.is_some() {
        charge_sources.push(source);
    }
    Ok(())
}

pub(super) fn append_inserted_counts(
    component: &InsertedComponent,
    inserted_counts: &mut BTreeMap<String, usize>,
) {
    for molecule_type in inserted_molecule_types(component) {
        *inserted_counts.entry(molecule_type).or_insert(0) += component.count;
    }
}

pub(super) fn inserted_molecule_types(component: &InsertedComponent) -> Vec<String> {
    if !component.molecule_types.is_empty() {
        return component.molecule_types.clone();
    }
    if let Some(molecule_type) = &component.charge_molecule_type {
        return vec![molecule_type.clone()];
    }
    vec![component.name.clone()]
}

fn resolve_inserted_charge(
    component: &InsertedComponent,
) -> Result<warp_common::charge::NetChargeEstimate> {
    if component.charge_topology.is_some() || !component.charge_topologies.is_empty() {
        let molecule_types = inserted_molecule_types(component);
        let topology_paths = inserted_charge_topology_paths(component, molecule_types.len())?;
        let mut net_charge = 0.0f32;
        let mut has_charge = false;
        let mut sources = Vec::new();
        for (molecule_type, topology) in molecule_types.iter().zip(topology_paths.iter()) {
            let estimate = warp_common::charge::compute_gromacs_molecule_net_charge(
                Path::new(topology),
                molecule_type,
            )
            .map_err(|err| anyhow!(err))?;
            if let Some(charge) = estimate.net_charge_e {
                net_charge += charge;
                has_charge = true;
            }
            if let Some(source) = estimate.source {
                sources.push(source);
            }
        }
        let estimate = warp_common::charge::NetChargeEstimate {
            net_charge_e: has_charge.then_some(net_charge),
            source: Some(sources.join("+")),
        };
        if let Some(explicit) = component.net_charge_e {
            let derived = estimate.net_charge_e.unwrap_or(0.0);
            if (derived - explicit).abs() > 1.0e-4 {
                return Err(anyhow!(
                    "inserted component {} net_charge_e ({explicit}) does not match charge_topology-derived charge ({derived})",
                    component.name
                ));
            }
        }
        return Ok(estimate);
    }

    if let Some(definition_path) = &component.definition {
        let definition = load_molecule_definition(definition_path)?;
        let beads = molecule_definition_beads(&definition)?;
        let derived = sum_bead_charges(&beads.iter().map(|bead| bead.charge_e).collect::<Vec<_>>());
        if let Some(definition_charge) = definition.net_charge_e {
            if (derived - definition_charge).abs() > 1.0e-4 {
                return Err(anyhow!(
                    "inserted component {} definition net_charge_e ({definition_charge}) does not match bead charge sum ({derived})",
                    component.name
                ));
            }
        }
        if let Some(explicit) = component.net_charge_e {
            if (derived - explicit).abs() > 1.0e-4 {
                return Err(anyhow!(
                    "inserted component {} net_charge_e ({explicit}) does not match definition bead charge sum ({derived})",
                    component.name
                ));
            }
        }
        return Ok(warp_common::charge::NetChargeEstimate {
            net_charge_e: Some(derived),
            source: Some(format!("molecule_definition:{definition_path}")),
        });
    }

    if !component.beads.is_empty() {
        let derived = sum_bead_charges(
            &component
                .beads
                .iter()
                .map(|bead| bead.charge_e)
                .collect::<Vec<_>>(),
        );
        if let Some(explicit) = component.net_charge_e {
            if (derived - explicit).abs() > 1.0e-4 {
                return Err(anyhow!(
                    "inserted component {} net_charge_e ({explicit}) does not match explicit bead charge sum ({derived})",
                    component.name
                ));
            }
        }
        return Ok(warp_common::charge::NetChargeEstimate {
            net_charge_e: Some(derived),
            source: Some("inserted_component.beads.charge_e".to_string()),
        });
    }

    if let Some(template) = lookup_solute_template(&component.name) {
        let derived = template.net_charge_e();
        if let Some(explicit) = component.net_charge_e {
            if (derived - explicit).abs() > 1.0e-4 {
                return Err(anyhow!(
                    "inserted component {} net_charge_e ({explicit}) does not match solute-template-derived charge ({derived})",
                    component.name
                ));
            }
        }
        return Ok(warp_common::charge::NetChargeEstimate {
            net_charge_e: Some(derived),
            source: Some(format!("solute_template:{}", template.source)),
        });
    }
    if let Some(library) = lookup_solvent_library(&component.name) {
        return inserted_solvent_library_charge_estimate(component, &library);
    }

    Ok(warp_common::charge::NetChargeEstimate {
        net_charge_e: component.net_charge_e,
        source: component.charge_source.clone(),
    })
}

pub(super) fn load_molecule_definition(path: &str) -> Result<MoleculeDefinition> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| anyhow!("failed to read molecule definition {path}: {err}"))?;
    let definition: MoleculeDefinition = serde_json::from_str(&text)
        .map_err(|err| anyhow!("failed to parse molecule definition {path}: {err}"))?;
    if definition.schema_version != MOLECULE_DEFINITION_SCHEMA_VERSION {
        return Err(anyhow!(
            "molecule definition {path} schema_version must be {MOLECULE_DEFINITION_SCHEMA_VERSION}"
        ));
    }
    let beads = molecule_definition_beads(&definition)?;
    validate_molecule_definition_beads(path, &beads)?;
    validate_molecule_definition_bonds(path, &definition.bonds, beads.len())?;
    validate_molecule_definition_angles(path, &definition.angles, beads.len())?;
    validate_molecule_definition_dihedrals(path, &definition.dihedrals, beads.len())?;
    Ok(definition)
}

pub(super) fn molecule_definition_beads(
    definition: &MoleculeDefinition,
) -> Result<Vec<BuildBeadTemplate>> {
    if !definition.beads.is_empty() && !definition.residues.is_empty() {
        return Err(anyhow!(
            "molecule definition cannot contain both top-level beads and residues"
        ));
    }
    if !definition.beads.is_empty() {
        return Ok(definition.beads.clone());
    }
    let beads = definition
        .residues
        .iter()
        .flat_map(|residue| residue.beads.iter().cloned())
        .collect::<Vec<_>>();
    if beads.is_empty() {
        return Err(anyhow!(
            "molecule definition must contain at least one bead"
        ));
    }
    Ok(beads)
}

fn validate_molecule_definition_beads(path: &str, beads: &[BuildBeadTemplate]) -> Result<()> {
    for (bead_idx, bead) in beads.iter().enumerate() {
        if bead.name.trim().is_empty() {
            return Err(anyhow!(
                "molecule definition {path} beads[{bead_idx}].name must not be empty"
            ));
        }
        if bead.offset_angstrom.iter().any(|value| !value.is_finite()) {
            return Err(anyhow!(
                "molecule definition {path} beads[{bead_idx}].offset_angstrom values must be finite"
            ));
        }
        if !bead.charge_e.is_finite() {
            return Err(anyhow!(
                "molecule definition {path} beads[{bead_idx}].charge_e must be finite"
            ));
        }
    }
    Ok(())
}

fn validate_molecule_definition_bonds(
    path: &str,
    bonds: &[MoleculeDefinitionBond],
    bead_count: usize,
) -> Result<()> {
    for (bond_idx, bond) in bonds.iter().enumerate() {
        let [left, right] = bond.bead_indices;
        if left == right {
            return Err(anyhow!(
                "molecule definition {path} bonds[{bond_idx}].bead_indices must reference two distinct beads"
            ));
        }
        if left >= bead_count || right >= bead_count {
            return Err(anyhow!(
                "molecule definition {path} bonds[{bond_idx}].bead_indices out of range for {bead_count} beads"
            ));
        }
        if bond
            .length_nm
            .is_some_and(|length| !length.is_finite() || length <= 0.0)
        {
            return Err(anyhow!(
                "molecule definition {path} bonds[{bond_idx}].length_nm must be finite and > 0"
            ));
        }
        if bond
            .force_kj_mol_nm2
            .is_some_and(|force| !force.is_finite() || force < 0.0)
        {
            return Err(anyhow!(
                "molecule definition {path} bonds[{bond_idx}].force_kj_mol_nm2 must be finite and >= 0"
            ));
        }
    }
    Ok(())
}

fn validate_molecule_definition_angles(
    path: &str,
    angles: &[MoleculeDefinitionAngle],
    bead_count: usize,
) -> Result<()> {
    for (angle_idx, angle) in angles.iter().enumerate() {
        let [left, center, right] = angle.bead_indices;
        if left == center || center == right || left == right {
            return Err(anyhow!(
                "molecule definition {path} angles[{angle_idx}].bead_indices must reference three distinct beads"
            ));
        }
        if left >= bead_count || center >= bead_count || right >= bead_count {
            return Err(anyhow!(
                "molecule definition {path} angles[{angle_idx}].bead_indices out of range for {bead_count} beads"
            ));
        }
        if angle
            .angle_degrees
            .is_some_and(|value| !value.is_finite() || value <= 0.0 || value >= 180.0)
        {
            return Err(anyhow!(
                "molecule definition {path} angles[{angle_idx}].angle_degrees must be finite and between 0 and 180"
            ));
        }
        if angle
            .force_kj_mol_rad2
            .is_some_and(|force| !force.is_finite() || force < 0.0)
        {
            return Err(anyhow!(
                "molecule definition {path} angles[{angle_idx}].force_kj_mol_rad2 must be finite and >= 0"
            ));
        }
    }
    Ok(())
}

fn validate_molecule_definition_dihedrals(
    path: &str,
    dihedrals: &[MoleculeDefinitionDihedral],
    bead_count: usize,
) -> Result<()> {
    for (dihedral_idx, dihedral) in dihedrals.iter().enumerate() {
        let [first, second, third, fourth] = dihedral.bead_indices;
        if first == second
            || first == third
            || first == fourth
            || second == third
            || second == fourth
            || third == fourth
        {
            return Err(anyhow!(
                "molecule definition {path} dihedrals[{dihedral_idx}].bead_indices must reference four distinct beads"
            ));
        }
        if first >= bead_count
            || second >= bead_count
            || third >= bead_count
            || fourth >= bead_count
        {
            return Err(anyhow!(
                "molecule definition {path} dihedrals[{dihedral_idx}].bead_indices out of range for {bead_count} beads"
            ));
        }
        if dihedral
            .phase_degrees
            .is_some_and(|value| !value.is_finite())
        {
            return Err(anyhow!(
                "molecule definition {path} dihedrals[{dihedral_idx}].phase_degrees must be finite"
            ));
        }
        if dihedral
            .force_kj_mol
            .is_some_and(|force| !force.is_finite() || force < 0.0)
        {
            return Err(anyhow!(
                "molecule definition {path} dihedrals[{dihedral_idx}].force_kj_mol must be finite and >= 0"
            ));
        }
        if dihedral
            .multiplicity
            .is_some_and(|multiplicity| multiplicity == 0)
        {
            return Err(anyhow!(
                "molecule definition {path} dihedrals[{dihedral_idx}].multiplicity must be > 0"
            ));
        }
    }
    Ok(())
}

pub(super) fn inserted_charge_topology_paths(
    component: &InsertedComponent,
    count: usize,
) -> Result<Vec<String>> {
    if !component.charge_topologies.is_empty() {
        if component.charge_topologies.len() != count {
            return Err(anyhow!(
                "inserted component {} charge_topologies length must match molecule_types length",
                component.name
            ));
        }
        return Ok(component.charge_topologies.clone());
    }
    let Some(path) = &component.charge_topology else {
        return Ok(Vec::new());
    };
    Ok(vec![path.clone(); count])
}

pub(super) fn emit_inserted_component(
    component: &InsertedComponent,
    kind: InsertedKind,
    system: &BuildSystem,
    occupied: &[EmittedBead],
    flood_summary: &mut InsertedFloodPlacementSummary,
    next_residue_id: &mut i32,
) -> Result<Vec<EmittedBead>> {
    let Some(path) = &component.coordinates else {
        if let Some(definition_path) = &component.definition {
            let definition = load_molecule_definition(definition_path)?;
            let beads = molecule_definition_beads(&definition)?;
            let residue_name = definition.name.as_deref().unwrap_or(&component.name);
            return emit_inserted_template_beads(
                component,
                residue_name,
                beads
                    .iter()
                    .map(|bead| (bead.name.clone(), bead.charge_e, bead.offset_angstrom))
                    .collect(),
                kind,
                system,
                occupied,
                flood_summary,
                next_residue_id,
            );
        }
        if !component.beads.is_empty() {
            return emit_inserted_template_beads(
                component,
                &component.name,
                component
                    .beads
                    .iter()
                    .map(|bead| (bead.name.clone(), bead.charge_e, bead.offset_angstrom))
                    .collect(),
                kind,
                system,
                occupied,
                flood_summary,
                next_residue_id,
            );
        }
        if matches!(kind, InsertedKind::Solute) {
            if let Some(template) = lookup_solute_template(&component.name) {
                return emit_inserted_template_beads(
                    component,
                    template.name,
                    template
                        .beads
                        .iter()
                        .map(|bead| (bead.name.to_string(), bead.charge_e, bead.offset_angstrom))
                        .collect(),
                    kind,
                    system,
                    occupied,
                    flood_summary,
                    next_residue_id,
                );
            }
            if let Some(library) = lookup_solvent_library(&component.name) {
                return emit_inserted_template_beads(
                    component,
                    &library.name,
                    inserted_solvent_library_beads(&library),
                    kind,
                    system,
                    occupied,
                    flood_summary,
                    next_residue_id,
                );
            }
        }
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
        return Err(anyhow!(
            "inserted component {} coordinate file contains no atoms",
            component.name
        ));
    }

    let source_center = match component.placement.center_method.as_str() {
        "cog" => molecule_center_of_geometry(&molecule.atoms),
        "none" => [0.0, 0.0, 0.0],
        method => {
            return Err(anyhow!(
                "inserted component {} placement center_method must be cog or none, got {method}",
                component.name
            ))
        }
    };
    let residue_templates = molecule_residue_templates(&molecule.atoms);
    let mut out = Vec::with_capacity(molecule.atoms.len() * component.count);
    let transformed_atoms = molecule
        .atoms
        .iter()
        .map(|atom| {
            let offset = transform_inserted_position(
                [atom.position.x, atom.position.y, atom.position.z],
                source_center,
                component.placement.rotate_degrees_xyz,
                [0.0, 0.0, 0.0],
            );
            (atom, offset)
        })
        .collect::<Vec<_>>();
    let offsets = transformed_atoms
        .iter()
        .map(|(_, offset)| *offset)
        .collect::<Vec<_>>();
    let center_plan = inserted_copy_centers(
        component,
        system,
        occupied,
        &offsets,
        kind.excluded_volume_factor(),
    )?;
    accumulate_inserted_flood_summary(flood_summary, &center_plan);

    for (copy_idx, target) in center_plan.centers.into_iter().enumerate() {
        let mut residue_ids = BTreeMap::new();
        for key in &residue_templates {
            residue_ids.insert(key.clone(), *next_residue_id);
            *next_residue_id += 1;
        }
        let copy_rotation = seeded_inserted_orientation_degrees(system, component, copy_idx);
        for (atom, offset) in &transformed_atoms {
            let key = (atom.resid, atom.resname.clone());
            let residue_id = residue_ids.get(&key).copied().ok_or_else(|| {
                anyhow!(
                    "missing residue mapping for inserted component {}",
                    component.name
                )
            })?;
            let oriented = rotate_xyz_reference_order(*offset, copy_rotation);
            out.push(EmittedBead {
                residue_id,
                residue_name: atom.resname.clone(),
                atom_name: atom.name.clone(),
                charge_e: atom.charge,
                position_angstrom: [
                    target[0] + oriented[0],
                    target[1] + oriented[1],
                    target[2] + oriented[2],
                ],
                excluded_volume_factor: kind.excluded_volume_factor(),
            });
        }
    }
    Ok(out)
}

fn emit_inserted_template_beads(
    component: &InsertedComponent,
    residue_name: &str,
    beads: Vec<(String, f32, [f32; 3])>,
    kind: InsertedKind,
    system: &BuildSystem,
    occupied: &[EmittedBead],
    flood_summary: &mut InsertedFloodPlacementSummary,
    next_residue_id: &mut i32,
) -> Result<Vec<EmittedBead>> {
    let mut out = Vec::with_capacity(beads.len() * component.count);
    let base_beads = beads
        .into_iter()
        .map(|(name, charge, offset)| {
            (
                name,
                charge,
                rotate_xyz_reference_order(offset, component.placement.rotate_degrees_xyz),
            )
        })
        .collect::<Vec<_>>();
    let offsets = base_beads
        .iter()
        .map(|(_, _, offset)| *offset)
        .collect::<Vec<_>>();
    let center_plan = inserted_copy_centers(
        component,
        system,
        occupied,
        &offsets,
        kind.excluded_volume_factor(),
    )?;
    accumulate_inserted_flood_summary(flood_summary, &center_plan);
    for (copy_idx, target) in center_plan.centers.into_iter().enumerate() {
        let residue_id = *next_residue_id;
        *next_residue_id += 1;
        let copy_rotation = seeded_inserted_orientation_degrees(system, component, copy_idx);
        for (name, charge, rotated) in &base_beads {
            let oriented = rotate_xyz_reference_order(*rotated, copy_rotation);
            out.push(EmittedBead {
                residue_id,
                residue_name: residue_name.to_string(),
                atom_name: name.clone(),
                charge_e: *charge,
                position_angstrom: [
                    target[0] + oriented[0],
                    target[1] + oriented[1],
                    target[2] + oriented[2],
                ],
                excluded_volume_factor: kind.excluded_volume_factor(),
            });
        }
    }
    Ok(out)
}
