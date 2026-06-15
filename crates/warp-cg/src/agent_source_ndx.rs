use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use serde_json::json;
use warp_structure::io::MoleculeData;

use crate::gromacs_ndx::read_gromacs_ndx_mapping;
use crate::mapping::MappingResult;

use super::agent_source_mapping::{
    bead_center, residue_role_for_policy, source_atom_name, source_mapping_provenance,
    source_polymer_enabled,
};
use super::{CgRequest, SourceBeadRecord, SourceHandoff, SourceMappingResult, SourceResidue};

pub(super) fn build_ndx_source_mapping(
    request: &CgRequest,
    handoff: &SourceHandoff,
    molecule: &MoleculeData,
    residues: &[SourceResidue],
) -> Result<SourceMappingResult> {
    let ndx_path = request
        .mapping
        .as_ref()
        .and_then(|mapping| mapping.ndx.as_deref())
        .ok_or_else(|| anyhow!("mapping.mode=ndx requires mapping.ndx"))?;
    let ndx_mapping = read_gromacs_ndx_mapping(ndx_path)?;
    let atom_count = molecule.atoms.len();
    for (bead_idx, group) in ndx_mapping.atom_indices.iter().enumerate() {
        for &atom_idx in group {
            if atom_idx >= atom_count {
                return Err(anyhow!(
                    "mapping.ndx bead '{}' references atom index {} but source has {} atoms",
                    ndx_mapping.bead_names[bead_idx],
                    atom_idx + 1,
                    atom_count
                ));
            }
        }
    }

    let residue_by_atom = residues
        .iter()
        .enumerate()
        .flat_map(|(residue_idx, residue)| {
            residue
                .atom_indices
                .iter()
                .map(move |atom_idx| (*atom_idx, residue_idx))
        })
        .collect::<BTreeMap<_, _>>();
    let mut beads = Vec::new();
    let mut residue_to_bead_indices = vec![Vec::<usize>::new(); residues.len()];
    for (bead_idx, group) in ndx_mapping.atom_indices.iter().enumerate() {
        let first_atom_idx = *group.first().ok_or_else(|| {
            anyhow!(
                "mapping.ndx bead '{}' contains no atoms",
                ndx_mapping.bead_names[bead_idx]
            )
        })?;
        let residue_idx = residue_by_atom.get(&first_atom_idx).copied().unwrap_or(0);
        let residue = &residues[residue_idx];
        residue_to_bead_indices[residue_idx].push(bead_idx);
        beads.push(SourceBeadRecord {
            index: bead_idx,
            name: ndx_mapping.bead_names[bead_idx].clone(),
            bead_type: ndx_mapping.bead_names[bead_idx].clone(),
            features: vec!["gromacs_ndx_mapping".to_string()],
            formal_charge: 0,
            resid: residue.resid,
            resname: residue.resname.clone(),
            chain: residue.chain,
            atom_indices: group.clone(),
            atom_names: group
                .iter()
                .map(|idx| source_atom_name(&molecule.atoms[*idx]))
                .collect(),
            coord: bead_center(group, &molecule.atoms),
        });
    }

    let residue_to_beads = residue_to_bead_indices
        .iter()
        .enumerate()
        .filter(|(_, beads)| !beads.is_empty())
        .map(|(residue_idx, beads)| {
            let residue = &residues[residue_idx];
            json!({
                "residue_index": residue_idx,
                "role": residue_role_for_policy(
                    residue_idx,
                    residues.len(),
                    source_polymer_enabled(request)
                ),
                "resid": residue.resid,
                "resname": residue.resname,
                "chain": residue.chain.to_string(),
                "beads": beads
            })
        })
        .collect::<Vec<_>>();
    let atom_to_bead = ndx_mapping
        .atom_indices
        .iter()
        .enumerate()
        .flat_map(|(bead_idx, group)| group.iter().map(move |atom_idx| (*atom_idx, bead_idx)))
        .collect::<BTreeMap<_, _>>();
    let connections =
        source_connections_from_atom_groups(&molecule.bonds, &ndx_mapping.atom_indices, atom_count);
    let templates = json!({
        "schema_version": "warp-cg.ndx_mapping.v1",
        "generated_by": "warp-cg.ndx",
        "source": ndx_path,
        "beads": beads.iter().map(|bead| {
            json!({
                "name": bead.name,
                "atom_indices": bead.atom_indices,
                "atom_names": bead.atom_names,
                "resid": bead.resid,
                "resname": bead.resname,
                "chain": bead.chain.to_string()
            })
        }).collect::<Vec<_>>()
    });
    let provenance = source_mapping_provenance(
        request,
        handoff,
        residues,
        &molecule.atoms,
        residue_to_beads,
        &atom_to_bead,
        "ndx",
    );

    Ok(SourceMappingResult {
        mapping: MappingResult {
            bead_names: ndx_mapping.bead_names,
            atom_groups: ndx_mapping.atom_indices,
            connections,
            bead_features: beads.iter().map(|bead| bead.features.clone()).collect(),
            bead_formal_charges: beads.iter().map(|bead| bead.formal_charge).collect(),
        },
        bonded_terms: None,
        beads,
        residue_count: residues.len(),
        aa_atom_count: atom_count,
        templates,
        provenance,
        warnings: Vec::new(),
        mapping_summary: json!({
            "bond_source": "explicit_topology_or_coordinates_connectivity",
            "aromaticity_source": "not_applicable_ndx_mapping",
            "polymer_enabled": source_polymer_enabled(request),
            "residue_count": residues.len(),
            "bond_count": molecule.bonds.len(),
            "warning_count": 0,
            "chemistry_hint_count": request.chemistry_hints.len(),
            "residue_bead_counts": residue_to_bead_indices.iter().enumerate().map(|(idx, beads)| {
                let residue = &residues[idx];
                json!({
                    "residue_index": idx,
                    "resid": residue.resid,
                    "resname": residue.resname,
                    "chain": residue.chain.to_string(),
                    "role": residue_role_for_policy(idx, residues.len(), source_polymer_enabled(request)),
                    "bead_count": beads.len()
                })
            }).collect::<Vec<_>>()
        }),
    })
}

fn source_connections_from_atom_groups(
    molecule_bonds: &[(usize, usize)],
    atom_groups: &[Vec<usize>],
    atom_count: usize,
) -> Vec<(usize, usize)> {
    let mut atom_to_beads = vec![Vec::<usize>::new(); atom_count];
    for (bead_idx, group) in atom_groups.iter().enumerate() {
        for &atom_idx in group {
            if let Some(beads) = atom_to_beads.get_mut(atom_idx) {
                beads.push(bead_idx);
            }
        }
    }
    let mut connections = Vec::new();
    for &(atom_a, atom_b) in molecule_bonds {
        let Some(beads_a) = atom_to_beads.get(atom_a) else {
            continue;
        };
        let Some(beads_b) = atom_to_beads.get(atom_b) else {
            continue;
        };
        for &bead_a in beads_a {
            for &bead_b in beads_b {
                if bead_a != bead_b {
                    connections.push((bead_a.min(bead_b), bead_a.max(bead_b)));
                }
            }
        }
    }
    connections.sort_unstable();
    connections.dedup();
    connections
}
