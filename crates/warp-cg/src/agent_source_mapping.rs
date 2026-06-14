use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use warp_structure::io::read_molecule;
use warp_structure::AtomRecord;

use crate::mapping::{map_molecule_with_options, MappingOptions, MappingResult};
use crate::molecule::Molecule;

use super::agent_source_ndx::build_ndx_source_mapping;
use super::agent_source_template::build_template_source_mapping;
use super::{CgRequest, SourceBeadRecord, SourceHandoff, SourceMappingResult, SourceResidue};

pub(super) fn source_residues(atoms: &[AtomRecord]) -> Vec<SourceResidue> {
    let mut residues = Vec::<SourceResidue>::new();
    let mut lookup = BTreeMap::<(char, i32, String), usize>::new();
    for (atom_idx, atom) in atoms.iter().enumerate() {
        let chain = if atom.chain == ' ' { 'A' } else { atom.chain };
        let resname = if atom.resname.trim().is_empty() {
            "MOL".to_string()
        } else {
            atom.resname.trim().to_string()
        };
        let key = (chain, atom.resid, resname.clone());
        let residue_idx = if let Some(idx) = lookup.get(&key).copied() {
            idx
        } else {
            let idx = residues.len();
            lookup.insert(key, idx);
            residues.push(SourceResidue {
                resid: atom.resid,
                resname,
                chain,
                atom_indices: Vec::new(),
            });
            idx
        };
        residues[residue_idx].atom_indices.push(atom_idx);
    }
    residues
}

fn source_bead_name(
    residue_idx: usize,
    bead_idx: usize,
    bead_count: usize,
    terminal_aware: bool,
) -> String {
    let prefix = if terminal_aware {
        if residue_idx == 0 {
            "H"
        } else {
            "M"
        }
    } else {
        "B"
    };
    if bead_count == 1 {
        prefix.to_string()
    } else {
        format!("{prefix}{}", bead_idx + 1)
    }
}

fn source_atom_groups_for_residue(
    residue: &SourceResidue,
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
    target_bead_size: usize,
) -> Vec<(String, Vec<String>, i32, Vec<usize>)> {
    let residue_atoms = residue.atom_indices.iter().copied().collect::<Vec<_>>();
    let local_by_global = residue_atoms
        .iter()
        .enumerate()
        .map(|(local, global)| (*global, local))
        .collect::<BTreeMap<_, _>>();
    let elements = residue_atoms
        .iter()
        .map(|idx| source_atom_element(&atoms[*idx]))
        .collect::<Vec<_>>();
    let positions = residue_atoms
        .iter()
        .map(|idx| {
            let pos = atoms[*idx].position;
            [pos.x, pos.y, pos.z]
        })
        .collect::<Vec<_>>();
    let local_bonds = bonds
        .iter()
        .filter_map(|&(a, b)| Some((*local_by_global.get(&a)?, *local_by_global.get(&b)?)))
        .collect::<Vec<_>>();
    let residue_molecule =
        Molecule::from_elements_bonds_and_positions(&elements, &local_bonds, Some(&positions));
    let mapping = map_molecule_with_options(
        &residue_molecule,
        &MappingOptions {
            target_bead_size: target_bead_size.max(1),
        },
    );
    mapping
        .bead_names
        .into_iter()
        .zip(mapping.bead_features)
        .zip(mapping.bead_formal_charges)
        .zip(mapping.atom_groups)
        .map(|(((bead_type, features), formal_charge), local_group)| {
            (
                bead_type,
                features,
                formal_charge,
                local_group
                    .into_iter()
                    .filter_map(|local| residue_atoms.get(local).copied())
                    .collect::<Vec<_>>(),
            )
        })
        .filter(|(_, _, _, group)| !group.is_empty())
        .collect()
}

pub(super) fn bead_center(atom_indices: &[usize], atoms: &[AtomRecord]) -> [f32; 3] {
    let mut center = [0.0f32; 3];
    let count = atom_indices.len().max(1) as f32;
    for idx in atom_indices {
        let pos = atoms[*idx].position;
        center[0] += pos.x;
        center[1] += pos.y;
        center[2] += pos.z;
    }
    [center[0] / count, center[1] / count, center[2] / count]
}

fn source_mapping_mode(request: &CgRequest) -> &str {
    request
        .mapping
        .as_ref()
        .map(|mapping| mapping.mode.as_str())
        .unwrap_or("auto")
}

pub(super) fn source_mapping_template_ref(request: &CgRequest) -> Option<&str> {
    request
        .mapping
        .as_ref()
        .and_then(|mapping| mapping.template.as_deref())
}

fn is_template_source_mapping(request: &CgRequest) -> bool {
    source_mapping_mode(request) == "template"
}

fn is_ndx_source_mapping(request: &CgRequest) -> bool {
    source_mapping_mode(request) == "ndx"
}

pub(super) fn load_mapping_template(path: &str) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| anyhow!("failed to read mapping template {path}: {err}"))?;
    let value = serde_json::from_str::<Value>(&text)
        .map_err(|err| anyhow!("failed to parse mapping template {path}: {err}"))?;
    let schema = value
        .get("schema_version")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if schema != "warp-cg.mapping_template.v1" {
        return Err(anyhow!(
            "mapping template {path} must use schema_version warp-cg.mapping_template.v1"
        ));
    }
    Ok(value)
}

pub(super) fn residue_role(residue_idx: usize, residue_count: usize) -> &'static str {
    if residue_idx == 0 {
        "head"
    } else if residue_idx + 1 == residue_count {
        "tail"
    } else {
        "middle"
    }
}

pub(super) fn source_atom_name(atom: &AtomRecord) -> String {
    atom.name.trim().to_string()
}

pub(super) fn source_atom_element(atom: &AtomRecord) -> String {
    let element = atom.element.trim();
    if !element.is_empty() {
        return element.to_string();
    }
    atom.name
        .chars()
        .find(|ch| ch.is_ascii_alphabetic())
        .map(|ch| ch.to_ascii_uppercase().to_string())
        .unwrap_or_else(|| "X".to_string())
}

fn residue_template_from_beads(
    role: &str,
    residue: &SourceResidue,
    residue_beads: &[usize],
    beads: &[SourceBeadRecord],
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
) -> Value {
    json!({
        "role": role,
        "resname": residue.resname,
        "beads": residue_beads.iter().filter_map(|bead_idx| beads.get(*bead_idx)).map(|bead| {
            json!({
                "name": bead.name,
                "bead_type": bead.bead_type,
                "features": bead.features,
                "formal_charge": bead.formal_charge,
                "atom_names": bead.atom_names,
                "elements": bead.atom_indices.iter().map(|idx| source_atom_element(&atoms[*idx])).collect::<Vec<_>>(),
                "local_bonds": atom_name_bonds_for_group(&bead.atom_indices, atoms, bonds),
                "connected": atom_group_is_connected(&bead.atom_indices, bonds)
            })
        }).collect::<Vec<_>>()
    })
}

pub(super) fn atom_name_bonds_for_group(
    atom_indices: &[usize],
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
) -> Vec<[String; 2]> {
    let group = atom_indices.iter().copied().collect::<BTreeSet<_>>();
    let mut pairs = bonds
        .iter()
        .filter_map(|&(a, b)| {
            if group.contains(&a) && group.contains(&b) {
                let mut names = [source_atom_name(&atoms[a]), source_atom_name(&atoms[b])];
                names.sort();
                Some(names)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    pairs.sort();
    pairs.dedup();
    pairs
}

pub(super) fn atom_group_is_connected(atom_indices: &[usize], bonds: &[(usize, usize)]) -> bool {
    if atom_indices.len() <= 1 {
        return true;
    }
    let group = atom_indices.iter().copied().collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    let mut queue = VecDeque::from([atom_indices[0]]);
    while let Some(current) = queue.pop_front() {
        if !seen.insert(current) {
            continue;
        }
        for &(a, b) in bonds {
            let neighbor = if a == current {
                b
            } else if b == current {
                a
            } else {
                continue;
            };
            if group.contains(&neighbor) && !seen.contains(&neighbor) {
                queue.push_back(neighbor);
            }
        }
    }
    seen.len() == atom_indices.len()
}

fn build_generated_mapping_template(
    request: &CgRequest,
    residues: &[SourceResidue],
    residue_to_bead_indices: &[Vec<usize>],
    beads: &[SourceBeadRecord],
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
    terminal_aware: bool,
) -> Value {
    let head = residues.first().map(|residue| {
        residue_template_from_beads(
            "head",
            residue,
            residue_to_bead_indices
                .first()
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            beads,
            atoms,
            bonds,
        )
    });
    let middle_idx = if residues.len() > 2 { 1 } else { 0 };
    let middle = residues.get(middle_idx).map(|residue| {
        residue_template_from_beads(
            "middle",
            residue,
            residue_to_bead_indices
                .get(middle_idx)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            beads,
            atoms,
            bonds,
        )
    });
    let tail = residues.last().map(|residue| {
        residue_template_from_beads(
            "tail",
            residue,
            residue_to_bead_indices
                .last()
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            beads,
            atoms,
            bonds,
        )
    });
    json!({
        "schema_version": "warp-cg.mapping_template.v1",
        "name": request.mapping.as_ref().and_then(|mapping| mapping.repeat_unit_hint.clone()).unwrap_or_else(|| request.name.clone()),
        "generated_by": "warp-cg.auto",
        "mapping_granularity": "residue_graph_partition",
        "strategy": request.mapping.as_ref().and_then(|mapping| mapping.strategy.clone()).unwrap_or_else(|| "polymer_residue_graph".to_string()),
        "target_bead_size": request.mapping.as_ref().and_then(|mapping| mapping.target_bead_size).unwrap_or(4),
        "preserve_functional_groups": request.mapping.as_ref().and_then(|mapping| mapping.preserve_functional_groups).unwrap_or(true),
        "terminal_aware": terminal_aware,
        "repeat_unit_hint": request.mapping.as_ref().and_then(|mapping| mapping.repeat_unit_hint.clone()),
        "residue_role_templates": {
            "head": head,
            "middle": middle,
            "tail": tail
        },
        "validation": {
            "require_connected_beads": true,
            "match_by": ["atom_name", "element", "local_graph"]
        }
    })
}

pub(super) fn source_mapping_provenance(
    request: &CgRequest,
    handoff: &SourceHandoff,
    residues: &[SourceResidue],
    atoms: &[AtomRecord],
    residue_to_beads: Vec<Value>,
    atom_to_bead: &BTreeMap<usize, usize>,
    mode: &str,
) -> Value {
    let source = request.source.as_ref();
    let terminal_aware = request
        .mapping
        .as_ref()
        .and_then(|mapping| mapping.terminal_aware)
        .unwrap_or(true);
    let mut residue_name_counts = BTreeMap::<String, usize>::new();
    for residue in residues {
        *residue_name_counts
            .entry(residue.resname.clone())
            .or_default() += 1;
    }
    let selected_atom_indices = atom_to_bead.keys().copied().collect::<Vec<_>>();
    json!({
        "mapping_mode": mode,
        "source_coordinates": handoff.coordinates.clone(),
        "source_topology": handoff.topology.clone(),
        "source_trajectory": handoff.trajectory.clone(),
        "selection": {
            "target_selection": source.and_then(|source| source.target_selection.clone()),
            "policy": if source.and_then(|source| source.target_selection.as_ref()).is_some() {
                "source.target_selection declared; provenance records atoms mapped by the resolved source coordinates"
            } else {
                "default_all_source_coordinate_atoms_and_residues"
            },
            "default_scope": "all atoms and residues in resolved source coordinates",
            "selected_atom_count": atom_to_bead.len(),
            "selected_residue_count": residues.len(),
            "selected_atom_indices": selected_atom_indices
        },
        "residue_interpretation": {
            "terminal_aware": terminal_aware,
            "repeat_unit_hint": request.mapping.as_ref().and_then(|mapping| mapping.repeat_unit_hint.clone()),
            "repeat_unit_interpretation": "one source residue is treated as one polymer repeat/terminal unit for source-driven polymer mapping",
            "residue_count": residues.len(),
            "residue_name_counts": residue_name_counts,
            "residues": residues.iter().enumerate().map(|(idx, residue)| {
                json!({
                    "residue_index": idx,
                    "role": residue_role(idx, residues.len()),
                    "resid": residue.resid,
                    "resname": residue.resname,
                    "chain": residue.chain.to_string(),
                    "atom_count": residue.atom_indices.len(),
                    "atom_indices": residue.atom_indices,
                    "atom_names": residue.atom_indices.iter().map(|atom_idx| {
                        source_atom_name(&atoms[*atom_idx])
                    }).collect::<Vec<_>>()
                })
            }).collect::<Vec<_>>()
        },
        "residue_to_bead_map": residue_to_beads,
        "aa_atom_to_cg_bead": atom_to_bead.iter().map(|(atom_idx, bead_idx)| {
            json!({"aa_atom_index": atom_idx, "cg_bead_index": bead_idx})
        }).collect::<Vec<_>>()
    })
}

pub(super) fn source_connections_from_mapping(
    molecule_bonds: &[(usize, usize)],
    atom_to_bead: &BTreeMap<usize, usize>,
    residue_to_bead_indices: &[Vec<usize>],
) -> Vec<(usize, usize)> {
    let mut connections = molecule_bonds
        .iter()
        .filter_map(|(a, b)| {
            let bead_a = atom_to_bead.get(a).copied()?;
            let bead_b = atom_to_bead.get(b).copied()?;
            (bead_a != bead_b).then_some((bead_a.min(bead_b), bead_a.max(bead_b)))
        })
        .collect::<Vec<_>>();
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

pub(super) fn build_source_mapping(
    request: &CgRequest,
    handoff: &SourceHandoff,
) -> Result<SourceMappingResult> {
    if is_template_source_mapping(request) {
        return build_template_source_mapping(request, handoff);
    }
    let molecule = read_molecule(
        Path::new(&handoff.coordinates),
        handoff.coordinate_format.as_deref(),
        false,
        true,
        handoff.topology.as_deref().map(Path::new),
    )
    .map_err(|err| anyhow!("failed to read source coordinates: {err}"))?;
    let residues = source_residues(&molecule.atoms);
    if residues.is_empty() {
        return Err(anyhow!("source coordinates contain no residues"));
    }
    if is_ndx_source_mapping(request) {
        return build_ndx_source_mapping(request, handoff, &molecule, &residues);
    }
    let terminal_aware = request
        .mapping
        .as_ref()
        .and_then(|mapping| mapping.terminal_aware)
        .unwrap_or(true);
    let target_bead_size = request
        .mapping
        .as_ref()
        .and_then(|mapping| mapping.target_bead_size)
        .unwrap_or(4);
    let mut bead_names = Vec::new();
    let mut atom_groups = Vec::new();
    let mut beads = Vec::new();
    let mut residue_to_beads = Vec::new();
    let mut residue_to_bead_indices = Vec::new();
    for (residue_idx, residue) in residues.iter().enumerate() {
        let groups = source_atom_groups_for_residue(
            residue,
            &molecule.atoms,
            &molecule.bonds,
            target_bead_size,
        );
        let mut residue_beads = Vec::new();
        for (local_bead_idx, (mapped_bead_type, features, formal_charge, group)) in
            groups.iter().enumerate()
        {
            let global_bead_idx = bead_names.len();
            let is_tail = terminal_aware && residue_idx + 1 == residues.len();
            let mut bead_name =
                source_bead_name(residue_idx, local_bead_idx, groups.len(), terminal_aware);
            if !terminal_aware {
                bead_name = mapped_bead_type.clone();
            }
            if is_tail {
                bead_name = if groups.len() == 1 {
                    "T".to_string()
                } else {
                    format!("T{}", local_bead_idx + 1)
                };
            }
            let coord = bead_center(group, &molecule.atoms);
            bead_names.push(bead_name.clone());
            atom_groups.push(group.clone());
            residue_beads.push(global_bead_idx);
            beads.push(SourceBeadRecord {
                index: global_bead_idx,
                name: bead_name.clone(),
                bead_type: mapped_bead_type.clone(),
                features: features.clone(),
                formal_charge: *formal_charge,
                resid: residue.resid,
                resname: residue.resname.clone(),
                chain: residue.chain,
                atom_indices: group.clone(),
                atom_names: group
                    .iter()
                    .map(|idx| source_atom_name(&molecule.atoms[*idx]))
                    .collect(),
                coord,
            });
        }
        residue_to_bead_indices.push(residue_beads.clone());
        residue_to_beads.push(json!({
            "residue_index": residue_idx,
            "role": residue_role(residue_idx, residues.len()),
            "resid": residue.resid,
            "resname": residue.resname,
            "chain": residue.chain.to_string(),
            "beads": residue_beads
        }));
    }

    let mut atom_to_bead = BTreeMap::<usize, usize>::new();
    for (bead_idx, group) in atom_groups.iter().enumerate() {
        for atom_idx in group {
            atom_to_bead.insert(*atom_idx, bead_idx);
        }
    }
    let connections =
        source_connections_from_mapping(&molecule.bonds, &atom_to_bead, &residue_to_bead_indices);

    let templates = build_generated_mapping_template(
        request,
        &residues,
        &residue_to_bead_indices,
        &beads,
        &molecule.atoms,
        &molecule.bonds,
        terminal_aware,
    );
    let provenance = source_mapping_provenance(
        request,
        handoff,
        &residues,
        &molecule.atoms,
        residue_to_beads,
        &atom_to_bead,
        "auto",
    );

    Ok(SourceMappingResult {
        mapping: MappingResult {
            bead_names,
            atom_groups,
            connections,
            bead_features: beads.iter().map(|bead| bead.features.clone()).collect(),
            bead_formal_charges: beads.iter().map(|bead| bead.formal_charge).collect(),
        },
        beads,
        residue_count: residues.len(),
        aa_atom_count: molecule.atoms.len(),
        templates,
        provenance,
    })
}
