use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use warp_structure::io::{read_molecule, read_system_auto, MoleculeData};
use warp_structure::model::BoxVectors;
use warp_structure::AtomRecord;

use crate::backmap::{
    BackmapPlan, BeadSite, BondConstraint, ChiralConstraint, Link, ResidueTemplate,
};
use crate::mapping::{map_molecule_with_options, MappingOptions, MappingResult};
use crate::molecule::Molecule;

use super::agent_source_ndx::build_ndx_source_mapping;
use super::agent_source_template::build_template_source_mapping;
use super::agent_source_validation::source_selection;
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

pub(super) fn source_backmap_plan(
    atoms: &[AtomRecord],
    residues: &[SourceResidue],
    atom_groups: &[Vec<usize>],
    bonds: &[(usize, usize)],
    mass_weighted: bool,
) -> Result<BackmapPlan> {
    let mut atom_residue = BTreeMap::<usize, usize>::new();
    for (residue_idx, residue) in residues.iter().enumerate() {
        for &atom_idx in &residue.atom_indices {
            atom_residue.insert(atom_idx, residue_idx);
        }
    }
    let mut component_adjacency = vec![BTreeSet::new(); residues.len()];
    let mut mapped_atoms = BTreeSet::new();
    for (bead_idx, group) in atom_groups.iter().enumerate() {
        let Some(&first_residue) = group.first().and_then(|idx| atom_residue.get(idx)) else {
            return Err(anyhow!("backmap bead {bead_idx} has no source atoms"));
        };
        for atom_idx in group {
            if !mapped_atoms.insert(*atom_idx) {
                return Err(anyhow!(
                    "backmap mapping is non-invertible: source atom {atom_idx} belongs to multiple beads"
                ));
            }
            let Some(&residue_idx) = atom_residue.get(atom_idx) else {
                return Err(anyhow!(
                    "backmap bead {bead_idx} references missing source atom {atom_idx}"
                ));
            };
            if residue_idx != first_residue {
                component_adjacency[first_residue].insert(residue_idx);
                component_adjacency[residue_idx].insert(first_residue);
            }
        }
    }
    let mut component_ids = vec![usize::MAX; residues.len()];
    let mut component_residues = Vec::<Vec<usize>>::new();
    for root in 0..residues.len() {
        if component_ids[root] != usize::MAX {
            continue;
        }
        let component_idx = component_residues.len();
        let mut members = Vec::new();
        let mut queue = VecDeque::from([root]);
        component_ids[root] = component_idx;
        while let Some(residue_idx) = queue.pop_front() {
            members.push(residue_idx);
            for &neighbor in &component_adjacency[residue_idx] {
                if component_ids[neighbor] == usize::MAX {
                    component_ids[neighbor] = component_idx;
                    queue.push_back(neighbor);
                }
            }
        }
        members.sort_unstable();
        component_residues.push(members);
    }
    let component_atoms = component_residues
        .iter()
        .map(|members| {
            members
                .iter()
                .flat_map(|&residue_idx| residues[residue_idx].atom_indices.iter().copied())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut atom_location = BTreeMap::<usize, (usize, usize)>::new();
    for (component_idx, component) in component_atoms.iter().enumerate() {
        for (local_idx, &atom_idx) in component.iter().enumerate() {
            atom_location.insert(atom_idx, (component_idx, local_idx));
        }
    }
    let mut sites_by_component = vec![Vec::new(); component_atoms.len()];
    for (bead_idx, group) in atom_groups.iter().enumerate() {
        let Some(&(component_idx, _)) = group.first().and_then(|idx| atom_location.get(idx)) else {
            return Err(anyhow!("backmap bead {bead_idx} has no source atoms"));
        };
        let local_indices = group
            .iter()
            .map(|atom_idx| {
                atom_location
                    .get(atom_idx)
                    .filter(|(group_component, _)| *group_component == component_idx)
                    .map(|(_, local_idx)| *local_idx)
                    .ok_or_else(|| anyhow!("backmap bead {bead_idx} has inconsistent atoms"))
            })
            .collect::<Result<Vec<_>>>()?;
        sites_by_component[component_idx].push(BeadSite {
            target_bead_index: bead_idx,
            atom_indices: local_indices,
            weights: mass_weighted.then(|| {
                group
                    .iter()
                    .map(|&atom_idx| element_mass(&atoms[atom_idx].element))
                    .collect()
            }),
        });
    }
    let templates = component_atoms
        .iter()
        .enumerate()
        .map(|(component_idx, component)| {
            let internal_bonds = bonds
                .iter()
                .filter_map(|&(left, right)| {
                    let &(left_component, left_local) = atom_location.get(&left)?;
                    let &(right_component, right_local) = atom_location.get(&right)?;
                    if left_component != component_idx || right_component != component_idx {
                        return None;
                    }
                    let left_position = atoms[left].position;
                    let right_position = atoms[right].position;
                    let dx = f64::from(left_position.x - right_position.x);
                    let dy = f64::from(left_position.y - right_position.y);
                    let dz = f64::from(left_position.z - right_position.z);
                    Some(BondConstraint {
                        atom_i: left_local,
                        atom_j: right_local,
                        target_distance: (dx * dx + dy * dy + dz * dz).sqrt(),
                    })
                })
                .collect::<Vec<_>>();
            let mut neighbors = vec![Vec::new(); component.len()];
            for bond in &internal_bonds {
                neighbors[bond.atom_i].push(bond.atom_j);
                neighbors[bond.atom_j].push(bond.atom_i);
            }
            let reference_coords = component
                .iter()
                .map(|&atom_idx| {
                    let position = atoms[atom_idx].position;
                    [
                        f64::from(position.x),
                        f64::from(position.y),
                        f64::from(position.z),
                    ]
                })
                .collect::<Vec<_>>();
            let chirality = neighbors
                .iter()
                .enumerate()
                .filter_map(|(center, adjacent)| {
                    if adjacent.len() < 3 {
                        return None;
                    }
                    let sign = local_signed_volume_sign(
                        reference_coords[center],
                        reference_coords[adjacent[0]],
                        reference_coords[adjacent[1]],
                        reference_coords[adjacent[2]],
                    );
                    (sign != 0).then_some(ChiralConstraint {
                        center,
                        neighbor_a: adjacent[0],
                        neighbor_b: adjacent[1],
                        neighbor_c: adjacent[2],
                        reference_sign: sign,
                    })
                })
                .collect();
            ResidueTemplate {
                name: component_residues[component_idx]
                    .iter()
                    .map(|&residue_idx| {
                        let residue = &residues[residue_idx];
                        format!("{}:{}:{}", residue.chain, residue.resid, residue.resname)
                    })
                    .collect::<Vec<_>>()
                    .join("+"),
                atom_names: component
                    .iter()
                    .map(|&atom_idx| source_atom_name(&atoms[atom_idx]))
                    .collect(),
                elements: component
                    .iter()
                    .map(|&atom_idx| atoms[atom_idx].element.clone())
                    .collect(),
                residue_names: component
                    .iter()
                    .map(|&atom_idx| atoms[atom_idx].resname.clone())
                    .collect(),
                residue_ids: component
                    .iter()
                    .map(|&atom_idx| atoms[atom_idx].resid)
                    .collect(),
                chains: component
                    .iter()
                    .map(|&atom_idx| atoms[atom_idx].chain.to_string())
                    .collect(),
                source_atom_indices: component.clone(),
                reference_coords,
                bead_sites: sites_by_component[component_idx].clone(),
                bonds: internal_bonds,
                chirality,
            }
        })
        .collect::<Vec<_>>();
    let mut links = Vec::new();
    for &(left, right) in bonds {
        let (Some(&(left_residue, left_local)), Some(&(right_residue, right_local))) =
            (atom_location.get(&left), atom_location.get(&right))
        else {
            continue;
        };
        if left_residue != right_residue {
            links.push(Link {
                from_template: left_residue,
                from_atom: left_local,
                to_template: right_residue,
                to_atom: right_local,
                target_distance: Some({
                    let left_position = atoms[left].position;
                    let right_position = atoms[right].position;
                    let dx = f64::from(left_position.x - right_position.x);
                    let dy = f64::from(left_position.y - right_position.y);
                    let dz = f64::from(left_position.z - right_position.z);
                    (dx * dx + dy * dy + dz * dz).sqrt()
                }),
            });
        }
    }
    links.sort_by_key(|link| {
        (
            link.from_template,
            link.to_template,
            link.from_atom,
            link.to_atom,
        )
    });
    links.dedup();
    let plan = BackmapPlan {
        templates,
        links,
        fudge_factor: 1.0,
    };
    let cg_coords = atom_groups
        .iter()
        .map(|group| {
            let mut total = 0.0;
            let sum = group.iter().fold([0.0; 3], |mut sum, &atom_idx| {
                let position = atoms[atom_idx].position;
                let weight = if mass_weighted {
                    element_mass(&atoms[atom_idx].element)
                } else {
                    1.0
                };
                sum[0] += f64::from(position.x) * weight;
                sum[1] += f64::from(position.y) * weight;
                sum[2] += f64::from(position.z) * weight;
                total += weight;
                sum
            });
            [sum[0] / total, sum[1] / total, sum[2] / total]
        })
        .collect::<Vec<_>>();
    plan.validate(&cg_coords)
        .map_err(|err| anyhow!("generated backmap plan is invalid: {err}"))?;
    Ok(plan)
}

fn element_mass(element: &str) -> f64 {
    match element.trim().to_ascii_uppercase().as_str() {
        "H" => 1.008,
        "C" => 12.011,
        "N" => 14.007,
        "O" => 15.999,
        "F" => 18.998,
        "P" => 30.974,
        "S" => 32.06,
        "CL" => 35.45,
        "BR" => 79.904,
        "I" => 126.904,
        _ => 1.0,
    }
}

fn local_signed_volume_sign(
    center: [f64; 3],
    neighbor_a: [f64; 3],
    neighbor_b: [f64; 3],
    neighbor_c: [f64; 3],
) -> i8 {
    let a = [
        neighbor_a[0] - center[0],
        neighbor_a[1] - center[1],
        neighbor_a[2] - center[2],
    ];
    let b = [
        neighbor_b[0] - center[0],
        neighbor_b[1] - center[1],
        neighbor_b[2] - center[2],
    ];
    let c = [
        neighbor_c[0] - center[0],
        neighbor_c[1] - center[1],
        neighbor_c[2] - center[2],
    ];
    let cross = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    let volume = cross[0] * c[0] + cross[1] * c[1] + cross[2] * c[2];
    if volume > 1.0e-12 {
        1
    } else if volume < -1.0e-12 {
        -1
    } else {
        0
    }
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

pub(super) fn source_mapping_mass_weighted(request: &CgRequest) -> bool {
    request
        .trajectory_source
        .as_ref()
        .and_then(|source| source.mass_weighted)
        .unwrap_or(false)
}

pub(super) fn mapped_bead_center(
    atom_indices: &[usize],
    atoms: &[AtomRecord],
    mass_weighted: bool,
) -> [f32; 3] {
    if !mass_weighted {
        return bead_center(atom_indices, atoms);
    }
    let mut center = [0.0f64; 3];
    let mut total = 0.0;
    for &atom_idx in atom_indices {
        let weight = element_mass(&atoms[atom_idx].element);
        let position = atoms[atom_idx].position;
        center[0] += f64::from(position.x) * weight;
        center[1] += f64::from(position.y) * weight;
        center[2] += f64::from(position.z) * weight;
        total += weight;
    }
    [
        (center[0] / total) as f32,
        (center[1] / total) as f32,
        (center[2] / total) as f32,
    ]
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

pub(super) fn source_polymer_enabled(request: &CgRequest) -> bool {
    request
        .polymer
        .as_ref()
        .and_then(|polymer| polymer.enabled)
        .unwrap_or(false)
        || request
            .mapping
            .as_ref()
            .and_then(|mapping| mapping.strategy.as_deref())
            .is_some_and(|strategy| strategy == "polymer_residue_graph")
}

pub(super) fn source_terminal_aware(request: &CgRequest) -> bool {
    if !source_polymer_enabled(request) {
        return false;
    }
    request
        .polymer
        .as_ref()
        .and_then(|polymer| polymer.terminal_aware)
        .or_else(|| {
            request
                .mapping
                .as_ref()
                .and_then(|mapping| mapping.terminal_aware)
        })
        .unwrap_or(true)
}

pub(super) fn residue_role_for_policy(
    residue_idx: usize,
    residue_count: usize,
    polymer_enabled: bool,
) -> &'static str {
    if !polymer_enabled {
        return "standalone";
    }
    if residue_idx == 0 {
        "head"
    } else if residue_idx + 1 == residue_count {
        "tail"
    } else {
        "middle"
    }
}

pub(super) fn residue_role(residue_idx: usize, residue_count: usize) -> &'static str {
    residue_role_for_policy(residue_idx, residue_count, true)
}

pub(super) fn template_policy(request: &CgRequest) -> &str {
    request
        .mapping
        .as_ref()
        .and_then(|mapping| mapping.template_policy.as_deref())
        .unwrap_or("strict_graph")
}

pub(super) fn append_bead_count_mismatch_warnings(
    request: &CgRequest,
    residues: &[SourceResidue],
    residue_to_bead_indices: &[Vec<usize>],
    polymer_enabled: bool,
    warnings: &mut Vec<Value>,
) -> Result<()> {
    let Some(mapping) = request.mapping.as_ref() else {
        return Ok(());
    };
    if mapping.expected_beads_per_role.is_empty() {
        return Ok(());
    }
    let on_mismatch = mapping.on_bead_count_mismatch.as_deref().unwrap_or("error");
    for (idx, residue) in residues.iter().enumerate() {
        let role = residue_role_for_policy(idx, residues.len(), polymer_enabled);
        let Some(expected) = mapping.expected_beads_per_role.get(role) else {
            continue;
        };
        let actual = residue_to_bead_indices.get(idx).map(Vec::len).unwrap_or(0);
        if actual == *expected {
            continue;
        }
        let warning = json!({
            "code": "warp_cg.bead_count_mismatch",
            "severity": on_mismatch,
            "message": "mapped bead count does not match mapping.expected_beads_per_role",
            "residue_index": idx,
            "resid": residue.resid,
            "resname": residue.resname,
            "chain": residue.chain.to_string(),
            "role": role,
            "expected_bead_count": expected,
            "actual_bead_count": actual
        });
        if on_mismatch == "error" {
            return Err(anyhow!("{warning}"));
        }
        warnings.push(warning);
    }
    Ok(())
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
    polymer_enabled: bool,
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
        "polymer_enabled": polymer_enabled,
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
    let polymer_enabled = source_polymer_enabled(request);
    let terminal_aware = source_terminal_aware(request);
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
            "selection": source.and_then(|source| source.selection.clone()),
            "policy": if source.and_then(source_selection).is_some() {
                "source.selection/source.target_selection declared; provenance records atoms mapped by the resolved source coordinates"
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
            "polymer_enabled": polymer_enabled,
            "role_mode": request.polymer.as_ref().and_then(|polymer| polymer.role_mode.clone()).unwrap_or_else(|| "infer".to_string()),
            "end_group_policy": request.polymer.as_ref().and_then(|polymer| polymer.end_group_policy.clone()).unwrap_or_else(|| "preserve".to_string()),
            "repeat_unit_hint": request.mapping.as_ref().and_then(|mapping| mapping.repeat_unit_hint.clone()),
            "repeat_unit_interpretation": if polymer_enabled { "one source residue is treated as one polymer repeat/terminal unit for source-driven polymer mapping" } else { "structure input is treated as standalone residue/molecule mapping" },
            "residue_count": residues.len(),
            "residue_name_counts": residue_name_counts,
            "residues": residues.iter().enumerate().map(|(idx, residue)| {
                json!({
                    "residue_index": idx,
                    "role": residue_role_for_policy(idx, residues.len(), polymer_enabled),
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
        "chemistry_hints": request.chemistry_hints,
        "chemistry_policy": request.chemistry_policy,
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
    let mut molecule = read_molecule(
        Path::new(&handoff.coordinates),
        handoff.coordinate_format.as_deref(),
        false,
        true,
        handoff.topology.as_deref().map(Path::new),
    )
    .map_err(|err| anyhow!("failed to read source coordinates: {err}"))?;
    let source_box_vectors = molecule
        .box_vectors
        .or_else(|| read_source_box_vectors(handoff).ok().flatten());
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
    if is_ndx_source_mapping(request) {
        return build_ndx_source_mapping(request, handoff, &molecule, &residues);
    }
    let polymer_enabled = source_polymer_enabled(request);
    let terminal_aware = source_terminal_aware(request);
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
            let coord = mapped_bead_center(
                group,
                &molecule.atoms,
                source_mapping_mass_weighted(request),
            );
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
                    "role": residue_role_for_policy(residue_idx, residues.len(), polymer_enabled),
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
        polymer_enabled,
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
    append_bead_count_mismatch_warnings(
        request,
        &residues,
        &residue_to_bead_indices,
        polymer_enabled,
        &mut warnings,
    )?;
    let mapping_summary = source_mapping_summary(
        request,
        &residues,
        &residue_to_bead_indices,
        &molecule.bonds,
        &bond_source,
        &warnings,
    );

    Ok(SourceMappingResult {
        backmap_plan: Some(source_backmap_plan(
            &molecule.atoms,
            &residues,
            &atom_groups,
            &molecule.bonds,
            source_mapping_mass_weighted(request),
        )?),
        mapping: MappingResult {
            bead_names,
            atom_groups,
            connections,
            bead_features: beads.iter().map(|bead| bead.features.clone()).collect(),
            bead_formal_charges: beads.iter().map(|bead| bead.formal_charge).collect(),
        },
        bonded_terms: None,
        beads,
        residue_count: residues.len(),
        aa_atom_count: molecule.atoms.len(),
        templates,
        provenance,
        warnings,
        mapping_summary,
        box_vectors: source_box_vectors,
    })
}

pub(super) fn read_source_box_vectors(handoff: &SourceHandoff) -> Result<Option<BoxVectors>> {
    let format = handoff
        .coordinate_format
        .as_deref()
        .map(str::to_ascii_lowercase)
        .or_else(|| {
            Path::new(&handoff.coordinates)
                .extension()
                .and_then(|ext| ext.to_str())
                .map(str::to_ascii_lowercase)
        })
        .unwrap_or_default();
    match format.as_str() {
        "pdb" | "ent" | "brk" => read_pdb_cryst1_box_vectors(&handoff.coordinates),
        _ => Ok(None),
    }
}

fn read_pdb_cryst1_box_vectors(path: &str) -> Result<Option<BoxVectors>> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| anyhow!("failed to read source coordinates for CRYST1 box: {err}"))?;
    let Some(line) = text.lines().find(|line| line.starts_with("CRYST1")) else {
        return Ok(None);
    };
    let a = parse_pdb_float_field(line, 6, 15, "CRYST1 a")?;
    let b = parse_pdb_float_field(line, 15, 24, "CRYST1 b")?;
    let c = parse_pdb_float_field(line, 24, 33, "CRYST1 c")?;
    let alpha = parse_pdb_float_field(line, 33, 40, "CRYST1 alpha")?;
    let beta = parse_pdb_float_field(line, 40, 47, "CRYST1 beta")?;
    let gamma = parse_pdb_float_field(line, 47, 54, "CRYST1 gamma")?;
    Ok(Some(box_vectors_from_lengths_angles(
        a, b, c, alpha, beta, gamma,
    )))
}

fn parse_pdb_float_field(line: &str, start: usize, end: usize, label: &str) -> Result<f32> {
    line.get(start..end)
        .unwrap_or("")
        .trim()
        .parse::<f32>()
        .map_err(|_| anyhow!("invalid {label} field in source PDB CRYST1 record"))
}

fn box_vectors_from_lengths_angles(
    a: f32,
    b: f32,
    c: f32,
    alpha_deg: f32,
    beta_deg: f32,
    gamma_deg: f32,
) -> BoxVectors {
    let alpha = alpha_deg.to_radians();
    let beta = beta_deg.to_radians();
    let gamma = gamma_deg.to_radians();
    let ax = a;
    let bx = b * gamma.cos();
    let by = b * gamma.sin();
    let cx = c * beta.cos();
    let cy = if by.abs() > f32::EPSILON {
        c * (alpha.cos() - beta.cos() * gamma.cos()) / gamma.sin()
    } else {
        0.0
    };
    let cz2 = (c * c - cx * cx - cy * cy).max(0.0);
    [[ax, 0.0, 0.0], [bx, by, 0.0], [cx, cy, cz2.sqrt()]]
}

pub(super) fn apply_source_selection(
    request: &CgRequest,
    source: &super::CgSource,
    handoff: &SourceHandoff,
    molecule: MoleculeData,
    warnings: &mut Vec<Value>,
) -> Result<MoleculeData> {
    let Some(selection_expr) = source_selection(source) else {
        return Ok(molecule);
    };
    let mut system = read_system_auto(
        Path::new(&handoff.coordinates),
        handoff.coordinate_format.as_deref(),
    )
    .map_err(|err| anyhow!("failed to read source coordinates for selection: {err}"))?;
    let selection = system
        .select(selection_expr)
        .map_err(|err| anyhow!("source selection '{selection_expr}' failed: {err}"))?;
    if selection.indices.is_empty() {
        return Err(anyhow!(
            "source selection '{selection_expr}' selected no atoms"
        ));
    }
    let selected = selection
        .indices
        .iter()
        .map(|idx| *idx as usize)
        .collect::<BTreeSet<_>>();
    if selected.len() == molecule.atoms.len() {
        return Ok(molecule);
    }
    warnings.push(json!({
        "code": "warp_cg.source_selection_applied",
        "severity": "info",
        "message": "source selection was applied before CG mapping",
        "selection": selection_expr,
        "selected_atoms": selected.len(),
        "source_atoms": molecule.atoms.len()
    }));
    let old_to_new = selected
        .iter()
        .enumerate()
        .map(|(new_idx, old_idx)| (*old_idx, new_idx))
        .collect::<BTreeMap<_, _>>();
    let atoms = selected
        .iter()
        .filter_map(|idx| molecule.atoms.get(*idx).cloned())
        .collect::<Vec<_>>();
    let bonds = molecule
        .bonds
        .into_iter()
        .filter_map(|(a, b)| Some((*old_to_new.get(&a)?, *old_to_new.get(&b)?)))
        .collect::<Vec<_>>();
    let ter_after = molecule
        .ter_after
        .into_iter()
        .filter_map(|idx| old_to_new.get(&idx).copied())
        .collect::<Vec<_>>();
    let _ = request;
    Ok(MoleculeData {
        atoms,
        bonds,
        box_vectors: molecule.box_vectors,
        ter_after,
    })
}

pub(super) fn resolve_bonds(
    request: &CgRequest,
    molecule: &mut MoleculeData,
    warnings: &mut Vec<Value>,
) -> Result<String> {
    if !molecule.bonds.is_empty() {
        return Ok(
            if request
                .source
                .as_ref()
                .and_then(|source| source.topology.as_ref())
                .is_some()
            {
                "explicit_topology".to_string()
            } else {
                "coordinates_connectivity".to_string()
            },
        );
    }
    let structure_default = request
        .source
        .as_ref()
        .is_some_and(|source| source.kind == "structure");
    let infer_bonds = request
        .bonding
        .as_ref()
        .and_then(|bonding| bonding.infer_bonds)
        .unwrap_or(structure_default);
    if !infer_bonds {
        return Err(anyhow!(
            "source structure has no explicit bonds; set bonding.infer_bonds=true to infer bonds from coordinates"
        ));
    }
    molecule.bonds = infer_coordinate_bonds(&molecule.atoms);
    let on_ambiguous = request
        .bonding
        .as_ref()
        .and_then(|bonding| bonding.on_ambiguous.as_deref())
        .unwrap_or("warn");
    let unknown_elements = molecule
        .atoms
        .iter()
        .filter(|atom| source_atom_element(atom) == "X")
        .count();
    if molecule.bonds.is_empty() || unknown_elements > 0 {
        let warning = json!({
            "code": "warp_cg.bond_inference_ambiguous",
            "severity": on_ambiguous,
            "message": "bond inference from coordinates is ambiguous",
            "inferred_bonds": molecule.bonds.len(),
            "unknown_element_atoms": unknown_elements
        });
        if on_ambiguous == "error" {
            return Err(anyhow!("{warning}"));
        }
        warnings.push(warning);
    } else {
        warnings.push(json!({
            "code": "warp_cg.bonds_inferred_from_coordinates",
            "severity": "warning",
            "message": "source had no explicit bonds; bonds were inferred from interatomic distances",
            "inferred_bonds": molecule.bonds.len()
        }));
    }
    Ok("inferred_distance".to_string())
}

fn infer_coordinate_bonds(atoms: &[AtomRecord]) -> Vec<(usize, usize)> {
    let mut bonds = Vec::new();
    for i in 0..atoms.len() {
        let ri = covalent_radius_angstrom(&source_atom_element(&atoms[i]));
        if ri <= 0.0 {
            continue;
        }
        for j in (i + 1)..atoms.len() {
            let rj = covalent_radius_angstrom(&source_atom_element(&atoms[j]));
            if rj <= 0.0 {
                continue;
            }
            let cutoff = (ri + rj + 0.45).max(0.4);
            let pi = atoms[i].position;
            let pj = atoms[j].position;
            let dx = pi.x - pj.x;
            let dy = pi.y - pj.y;
            let dz = pi.z - pj.z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist > 0.1 && dist <= cutoff {
                bonds.push((i, j));
            }
        }
    }
    bonds
}

pub(super) fn append_chemistry_hint_warnings(
    request: &CgRequest,
    molecule: &MoleculeData,
    warnings: &mut Vec<Value>,
) -> Result<()> {
    if request.chemistry_hints.is_empty() {
        return Ok(());
    }
    let hint_mode = request
        .chemistry_policy
        .as_ref()
        .and_then(|policy| policy.hint_mode.as_deref())
        .unwrap_or("validate");
    warnings.push(json!({
        "code": "warp_cg.chemistry_hints_recorded",
        "severity": "info",
        "message": "chemistry hints were accepted and recorded in provenance; auto mapping uses source geometry unless template or ndx mapping is selected",
        "hint_count": request.chemistry_hints.len(),
        "hint_mode": hint_mode
    }));
    for hint in &request.chemistry_hints {
        if hint.kind == "smiles" {
            validate_smiles_hint_against_geometry(request, hint, molecule, warnings)?;
        }
    }
    Ok(())
}

fn validate_smiles_hint_against_geometry(
    request: &CgRequest,
    hint: &super::ChemistryHintRequest,
    molecule: &MoleculeData,
    warnings: &mut Vec<Value>,
) -> Result<()> {
    let smiles = hint.value.as_deref().unwrap_or_default();
    let hint_molecule = match Molecule::from_smiles(smiles) {
        Ok(molecule) => molecule,
        Err(err) => {
            let warning = json!({
                "code": "warp_cg.chemistry_hint_invalid_smiles",
                "severity": chemistry_conflict_severity(request),
                "message": "SMILES chemistry hint could not be parsed",
                "scope": hint.scope,
                "error": err.to_string()
            });
            return handle_chemistry_warning(request, warning, warnings);
        }
    };
    let hinted_aromatic_rings = hint_molecule.aromatic_six_ring_count();
    if hinted_aromatic_rings == 0 {
        return Ok(());
    }
    let elements = molecule
        .atoms
        .iter()
        .map(source_atom_element)
        .collect::<Vec<_>>();
    let positions = molecule
        .atoms
        .iter()
        .map(|atom| [atom.position.x, atom.position.y, atom.position.z])
        .collect::<Vec<_>>();
    let geometry_molecule =
        Molecule::from_elements_bonds_and_positions(&elements, &molecule.bonds, Some(&positions));
    let geometry_aromatic_rings = geometry_molecule.aromatic_six_ring_count();
    if geometry_aromatic_rings < hinted_aromatic_rings {
        let warning = json!({
            "code": "warp_cg.chemistry_hint_geometry_conflict",
            "severity": chemistry_conflict_severity(request),
            "message": "SMILES hint contains more aromatic six-rings than source geometry supports; check bond inference, protonation/capping, or whether the input structure is minimized",
            "hint_kind": hint.kind,
            "hint_scope": hint.scope,
            "hint_aromatic_six_ring_count": hinted_aromatic_rings,
            "geometry_aromatic_six_ring_count": geometry_aromatic_rings,
            "hint_mode": request.chemistry_policy.as_ref().and_then(|policy| policy.hint_mode.as_deref()).unwrap_or("validate")
        });
        return handle_chemistry_warning(request, warning, warnings);
    }
    warnings.push(json!({
        "code": "warp_cg.chemistry_hint_validated",
        "severity": "info",
        "message": "SMILES chemistry hint aromaticity is consistent with source geometry",
        "hint_kind": hint.kind,
        "hint_scope": hint.scope,
        "hint_aromatic_six_ring_count": hinted_aromatic_rings,
        "geometry_aromatic_six_ring_count": geometry_aromatic_rings
    }));
    Ok(())
}

fn chemistry_conflict_severity(request: &CgRequest) -> &str {
    request
        .chemistry_policy
        .as_ref()
        .and_then(|policy| policy.on_conflict.as_deref())
        .unwrap_or("warn")
}

fn handle_chemistry_warning(
    request: &CgRequest,
    warning: Value,
    warnings: &mut Vec<Value>,
) -> Result<()> {
    if chemistry_conflict_severity(request) == "error" {
        return Err(anyhow!("{warning}"));
    }
    warnings.push(warning);
    Ok(())
}

fn covalent_radius_angstrom(element: &str) -> f32 {
    match element.trim().to_ascii_uppercase().as_str() {
        "H" => 0.31,
        "C" => 0.76,
        "N" => 0.71,
        "O" => 0.66,
        "F" => 0.57,
        "P" => 1.07,
        "S" => 1.05,
        "CL" => 1.02,
        "BR" => 1.20,
        "I" => 1.39,
        _ => 0.0,
    }
}

fn source_mapping_summary(
    request: &CgRequest,
    residues: &[SourceResidue],
    residue_to_bead_indices: &[Vec<usize>],
    bonds: &[(usize, usize)],
    bond_source: &str,
    warnings: &[Value],
) -> Value {
    let polymer_enabled = source_polymer_enabled(request);
    json!({
        "bond_source": bond_source,
        "aromaticity_source": if bond_source == "inferred_distance" { "geometry" } else { "explicit_or_geometry" },
        "polymer_enabled": polymer_enabled,
        "terminal_aware": source_terminal_aware(request),
        "residue_count": residues.len(),
        "bond_count": bonds.len(),
        "warning_count": warnings.len(),
        "chemistry_hint_count": request.chemistry_hints.len(),
        "residue_bead_counts": residues.iter().enumerate().map(|(idx, residue)| {
            json!({
                "residue_index": idx,
                "resid": residue.resid,
                "resname": residue.resname,
                "chain": residue.chain.to_string(),
                "role": residue_role_for_policy(idx, residues.len(), polymer_enabled),
                "bead_count": residue_to_bead_indices.get(idx).map(Vec::len).unwrap_or(0)
            })
        }).collect::<Vec<_>>()
    })
}
