use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use warp_structure::io::read_molecule;
use warp_structure::AtomRecord;

use crate::mapping::MappingResult;

use super::agent_bonded_classing::resolve_bonded_classing;
use super::agent_source_bonded_terms::template_bonded_term_set;
use super::agent_source_mapping::{
    append_bead_count_mismatch_warnings, append_chemistry_hint_warnings, apply_source_selection,
    atom_group_is_connected, atom_name_bonds_for_group, bead_center, load_mapping_template,
    residue_role, resolve_bonds, source_atom_element, source_atom_name,
    source_connections_from_mapping, source_mapping_provenance, source_mapping_template_ref,
    source_residues, template_policy,
};
use super::{
    CgRequest, SourceBeadClassContext, SourceBeadRecord, SourceHandoff, SourceMappingResult,
    SourceResidue,
};

fn template_beads_for_role<'a>(template: &'a Value, role: &str) -> Result<&'a Vec<Value>> {
    template
        .pointer(&format!("/residue_role_templates/{role}/beads"))
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("mapping template is missing residue_role_templates.{role}.beads"))
}

fn template_atom_indices_for_residue(
    residue: &SourceResidue,
    atoms: &[AtomRecord],
    atom_names: &[Value],
) -> Result<Vec<usize>> {
    let names = atom_names
        .iter()
        .map(|name| {
            name.as_str()
                .map(str::to_string)
                .ok_or_else(|| anyhow!("mapping template atom_names entries must be strings"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut indices = Vec::new();
    for wanted in &names {
        let matches = residue
            .atom_indices
            .iter()
            .copied()
            .filter(|idx| source_atom_name(&atoms[*idx]) == *wanted)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [idx] => indices.push(*idx),
            [] => {
                return Err(anyhow!(
                    "mapping template missing atom {wanted} in residue {} {}",
                    residue.resname,
                    residue.resid
                ))
            }
            _ => {
                return Err(anyhow!(
                    "mapping template atom {wanted} matched multiple atoms in residue {} {}",
                    residue.resname,
                    residue.resid
                ))
            }
        }
    }
    Ok(indices)
}

fn value_string_list(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn value_bond_list(value: Option<&Value>) -> Vec<[String; 2]> {
    let mut bonds = value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let pair = item.as_array()?;
                    let mut names = [
                        pair.first()?.as_str()?.to_string(),
                        pair.get(1)?.as_str()?.to_string(),
                    ];
                    names.sort();
                    Some(names)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    bonds.sort();
    bonds.dedup();
    bonds
}

fn validate_template_bead_match(
    residue: &SourceResidue,
    bead_name: &str,
    group: &[usize],
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
    bead_template: &Value,
) -> Result<()> {
    let expected_elements = value_string_list(bead_template.get("elements"));
    if !expected_elements.is_empty() {
        let actual_elements = group
            .iter()
            .map(|idx| source_atom_element(&atoms[*idx]))
            .collect::<Vec<_>>();
        if actual_elements != expected_elements {
            return Err(anyhow!(
                "mapping template bead {bead_name} element mismatch in residue {} {}: expected {:?}, got {:?}",
                residue.resname,
                residue.resid,
                expected_elements,
                actual_elements
            ));
        }
    }
    let expected_local_bonds = value_bond_list(bead_template.get("local_bonds"));
    if !expected_local_bonds.is_empty() {
        let actual_local_bonds = atom_name_bonds_for_group(group, atoms, bonds);
        if actual_local_bonds != expected_local_bonds {
            return Err(anyhow!(
                "mapping template bead {bead_name} local bond mismatch in residue {} {}",
                residue.resname,
                residue.resid
            ));
        }
    }
    if bead_template
        .get("connected")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        && !atom_group_is_connected(group, bonds)
    {
        return Err(anyhow!(
            "mapping template bead {bead_name} is disconnected in residue {} {}",
            residue.resname,
            residue.resid
        ));
    }
    Ok(())
}

pub(super) fn build_template_source_mapping(
    request: &CgRequest,
    handoff: &SourceHandoff,
) -> Result<SourceMappingResult> {
    let template_path = source_mapping_template_ref(request)
        .ok_or_else(|| anyhow!("mapping.mode=template requires mapping.template"))?;
    let policy = template_policy(request);
    let template = load_mapping_template(template_path)?;
    let mut molecule = read_molecule(
        Path::new(&handoff.coordinates),
        handoff.coordinate_format.as_deref(),
        false,
        true,
        handoff.topology.as_deref().map(Path::new),
    )
    .map_err(|err| anyhow!("failed to read source coordinates: {err}"))?;
    let mut warnings = Vec::new();
    if let Some(source) = request.source.as_ref() {
        molecule = apply_source_selection(request, source, handoff, molecule, &mut warnings)?;
    }
    let bond_source = resolve_bonds(request, &mut molecule, &mut warnings)?;
    append_chemistry_hint_warnings(request, &molecule, &mut warnings)?;
    let residues = source_residues(&molecule.atoms);
    if residues.is_empty() {
        return Err(anyhow!("source coordinates contain no residues"));
    }

    let mut bead_names = Vec::new();
    let mut atom_groups = Vec::new();
    let mut beads = Vec::new();
    let mut residue_to_beads = Vec::new();
    let mut residue_to_bead_indices = Vec::new();
    let mut bead_contexts = Vec::new();
    for (residue_idx, residue) in residues.iter().enumerate() {
        let role = residue_role(residue_idx, residues.len());
        let bead_templates = template_beads_for_role(&template, role)?;
        let mut residue_bead_indices = Vec::new();
        for bead_template in bead_templates {
            let bead_name = bead_template
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("mapping template bead requires name"))?
                .to_string();
            let bead_type = bead_template
                .get("bead_type")
                .and_then(Value::as_str)
                .unwrap_or(&bead_name)
                .to_string();
            let atom_names = bead_template
                .get("atom_names")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("mapping template bead {bead_name} requires atom_names"))?;
            let group = template_atom_indices_for_residue(residue, &molecule.atoms, atom_names)?;
            if policy == "strict_graph" {
                validate_template_bead_match(
                    residue,
                    &bead_name,
                    &group,
                    &molecule.atoms,
                    &molecule.bonds,
                    bead_template,
                )?;
            }
            let global_bead_idx = bead_names.len();
            let coord = bead_center(&group, &molecule.atoms);
            bead_names.push(bead_name.clone());
            atom_groups.push(group.clone());
            residue_bead_indices.push(global_bead_idx);
            bead_contexts.push(SourceBeadClassContext {
                residue_index: residue_idx,
                role: role.to_string(),
                template_bead_name: bead_name.clone(),
            });
            beads.push(SourceBeadRecord {
                index: global_bead_idx,
                name: bead_name.clone(),
                bead_type,
                features: bead_template
                    .get("features")
                    .and_then(Value::as_array)
                    .map(|features| {
                        features
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default(),
                formal_charge: bead_template
                    .get("formal_charge")
                    .and_then(Value::as_i64)
                    .unwrap_or(0) as i32,
                resid: residue.resid,
                resname: residue.resname.clone(),
                chain: residue.chain,
                atom_names: group
                    .iter()
                    .map(|idx| source_atom_name(&molecule.atoms[*idx]))
                    .collect(),
                atom_indices: group,
                coord,
            });
        }
        residue_to_bead_indices.push(residue_bead_indices.clone());
        residue_to_beads.push(json!({
            "residue_index": residue_idx,
            "role": role,
            "resid": residue.resid,
            "resname": residue.resname,
            "chain": residue.chain.to_string(),
            "beads": residue_bead_indices
        }));
    }

    let mut atom_to_bead = BTreeMap::<usize, usize>::new();
    for (bead_idx, group) in atom_groups.iter().enumerate() {
        for atom_idx in group {
            atom_to_bead.insert(*atom_idx, bead_idx);
        }
    }
    let geometry_connections =
        source_connections_from_mapping(&molecule.bonds, &atom_to_bead, &residue_to_bead_indices);
    let template_connections = template_authority_connections(&residue_to_bead_indices);
    let connection_source = if policy == "assignment_only" {
        "template_role_order"
    } else {
        "source_geometry"
    };
    let effective_connections = if policy == "assignment_only" {
        template_connections
    } else {
        geometry_connections.clone()
    };
    let auto_bonded_terms =
        template_bonded_term_set(bead_names.len(), &effective_connections, &bead_contexts);
    let bonded_classing_request = request
        .mapping
        .as_ref()
        .and_then(|mapping| mapping.bonded_classing.as_ref());
    let bonded_classing =
        resolve_bonded_classing(bonded_classing_request, auto_bonded_terms, bead_names.len())?;
    let bonded_terms = bonded_classing.terms;
    let mut bonded_class_summary = bonded_classing.summary;
    if let Some(summary) = bonded_class_summary.as_object_mut() {
        summary.insert(
            "connection_source".to_string(),
            Value::String(connection_source.to_string()),
        );
    }
    let legacy_bonded_class_summary = json!({
        "enabled": true,
        "class_source": bonded_class_summary["class_source"].clone(),
        "connection_source": connection_source,
        "raw_instance_counts": bonded_class_summary["raw_instance_counts"].clone(),
        "class_counts": bonded_class_summary["class_counts"].clone(),
        "unclassified_counts": bonded_class_summary["unclassified_counts"].clone(),
        "mode": bonded_class_summary["mode"].clone()
    });
    if policy == "assignment_only" {
        warnings.push(json!({
            "code": "warp_cg.template_assignment_only",
            "severity": "info",
            "message": "template replay used atom-name assignment only; local graph, local bond, and connected-bead checks were skipped; bonded classes were derived from template bead role/order, not target source geometry",
            "template": template_path
        }));
    }
    append_bead_count_mismatch_warnings(
        request,
        &residues,
        &residue_to_bead_indices,
        true,
        &mut warnings,
    )?;
    let mut templates = template.clone();
    if let Some(object) = templates.as_object_mut() {
        object.insert(
            "template_match_report".to_string(),
            json!({
                "status": "ok",
                "template": template_path,
                "template_policy": policy,
                "residue_count": residues.len(),
                "matched_residues": residues.len(),
                "unmapped_atoms": molecule.atoms.len().saturating_sub(atom_to_bead.len()),
                "missing_atoms": [],
                "extra_atoms": []
            }),
        );
    }
    let provenance = source_mapping_provenance(
        request,
        handoff,
        &residues,
        &molecule.atoms,
        residue_to_beads,
        &atom_to_bead,
        "template",
    );
    let warning_count = warnings.len();

    Ok(SourceMappingResult {
        mapping: MappingResult {
            bead_names,
            atom_groups,
            connections: effective_connections,
            bead_features: beads.iter().map(|bead| bead.features.clone()).collect(),
            bead_formal_charges: beads.iter().map(|bead| bead.formal_charge).collect(),
        },
        bonded_terms: Some(bonded_terms),
        beads,
        residue_count: residues.len(),
        aa_atom_count: molecule.atoms.len(),
        templates,
        provenance,
        warnings,
        mapping_summary: json!({
            "bond_source": "template",
            "aromaticity_source": "template",
            "template_policy": policy,
            "polymer_enabled": true,
            "residue_count": residues.len(),
            "bond_count": geometry_connections.len(),
            "source_bond_source": bond_source,
            "template_connection_source": connection_source,
            "bonded_classing_summary": bonded_class_summary,
            "bonded_parameter_classing": legacy_bonded_class_summary,
            "warning_count": warning_count,
            "chemistry_hint_count": request.chemistry_hints.len(),
            "residue_bead_counts": residue_to_bead_indices.iter().enumerate().map(|(idx, beads)| {
                let residue = &residues[idx];
                json!({
                    "residue_index": idx,
                    "resid": residue.resid,
                    "resname": residue.resname,
                    "chain": residue.chain.to_string(),
                    "role": residue_role(idx, residues.len()),
                    "bead_count": beads.len()
                })
            }).collect::<Vec<_>>()
        }),
    })
}

fn template_authority_connections(residue_to_bead_indices: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let mut connections = Vec::new();
    for residue_beads in residue_to_bead_indices {
        for pair in residue_beads.windows(2) {
            connections.push((pair[0].min(pair[1]), pair[0].max(pair[1])));
        }
    }
    let residue_first_last = residue_to_bead_indices
        .iter()
        .filter_map(|beads| Some((*beads.first()?, *beads.last()?)))
        .collect::<Vec<_>>();
    for pair in residue_first_last.windows(2) {
        connections.push((pair[0].1.min(pair[1].0), pair[0].1.max(pair[1].0)));
    }
    connections.sort_unstable();
    connections.dedup();
    connections
}
