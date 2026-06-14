use anyhow::{anyhow, Result};
use std::path::Path;
use traj_core::{minimum_image_vector, Box3};
use warp_structure::io::{read_molecule, read_prmtop_topology};

use crate::bonded_terms::BondedTermSet;

use super::NativeTrajectoryOptions;

pub(super) fn bonded_whole_connections(terms: &BondedTermSet) -> Vec<(usize, usize)> {
    let mut connections = terms.bonds_as_connections();
    connections.extend(
        terms
            .constraints
            .iter()
            .flat_map(|group| group.members.iter())
            .map(|member| normalized_pair(member[0], member[1])),
    );
    connections.extend(terms.virtual_sites.iter().flat_map(|site| {
        site.defining_beads
            .iter()
            .map(|&defining_bead| normalized_pair(site.site, defining_bead))
    }));
    connections.sort_unstable();
    connections.dedup();
    connections
}

pub(super) fn resolve_source_whole_connections(
    options: &NativeTrajectoryOptions,
    source_atom_count: usize,
) -> Result<Vec<(usize, usize)>> {
    if options.topology.is_none() {
        return Ok(Vec::new());
    }
    if let Some(connections) = topology_bond_connections(options, source_atom_count)? {
        return Ok(connections);
    }
    let indices = options
        .atom_indices
        .clone()
        .unwrap_or_else(|| (0..source_atom_count).collect());
    validate_source_indices(&indices, source_atom_count)?;
    Ok(indices
        .windows(2)
        .map(|window| normalized_pair(window[0], window[1]))
        .collect())
}

fn topology_bond_connections(
    options: &NativeTrajectoryOptions,
    source_atom_count: usize,
) -> Result<Option<Vec<(usize, usize)>>> {
    let Some(topology) = options.topology.as_deref() else {
        return Ok(None);
    };
    let path = Path::new(topology);
    let format = topology_format(path, options.topology_format.as_deref());
    let bonds = match format.as_str() {
        "prmtop" | "top" => {
            read_prmtop_topology(path)
                .map_err(|err| anyhow!("failed to read prmtop topology bonds: {err}"))?
                .bonds
        }
        "pdb" | "brk" | "ent" | "pdbqt" | "pqr" | "mol2" | "tinker" | "txyz" | "gro" | "g96"
        | "gromos96" | "lammps" | "lammps-data" | "lmp" | "crd" => {
            read_molecule(path, Some(&format), false, true, None)
                .map_err(|err| anyhow!("failed to read topology bonds: {err}"))?
                .bonds
        }
        _ => Vec::new(),
    };
    if bonds.is_empty() {
        return Ok(None);
    }
    let mut connections = Vec::with_capacity(bonds.len());
    for (left, right) in bonds {
        if left >= source_atom_count || right >= source_atom_count {
            return Err(anyhow!(
                "topology bond {}-{} exceeds trajectory atom count {}",
                left,
                right,
                source_atom_count
            ));
        }
        connections.push(normalized_pair(left, right));
    }
    connections.sort_unstable();
    connections.dedup();
    Ok(Some(connections))
}

fn topology_format(path: &Path, requested: Option<&str>) -> String {
    requested
        .filter(|format| !format.trim().is_empty())
        .map(|format| format.to_ascii_lowercase())
        .or_else(|| {
            path.extension()
                .and_then(|value| value.to_str())
                .map(|value| value.to_ascii_lowercase())
        })
        .unwrap_or_default()
}

pub(super) fn make_source_whole_by_bonded_connectivity(
    coords: &[[f32; 4]],
    connections: &[(usize, usize)],
    box_: Box3,
) -> Result<Vec<[f32; 4]>> {
    let xyz = coords
        .iter()
        .map(|coord| [coord[0], coord[1], coord[2]])
        .collect::<Vec<_>>();
    let whole = make_whole_by_bonded_connectivity(&xyz, connections, box_)?;
    Ok(coords
        .iter()
        .zip(whole)
        .map(|(source, repaired)| [repaired[0], repaired[1], repaired[2], source[3]])
        .collect())
}

pub(super) fn make_whole_by_bonded_connectivity(
    coords: &[[f32; 3]],
    connections: &[(usize, usize)],
    box_: Box3,
) -> Result<Vec<[f32; 3]>> {
    if coords.is_empty() || connections.is_empty() || matches!(box_, Box3::None) {
        return Ok(coords.to_vec());
    }
    let adjacency = adjacency_list(coords.len(), connections)?;
    let mut repaired = coords.to_vec();
    let mut seen = vec![false; coords.len()];
    for root in 0..coords.len() {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        let mut stack = vec![root];
        while let Some(current) = stack.pop() {
            for &next in &adjacency[current] {
                if seen[next] {
                    continue;
                }
                let delta = minimum_image_vector(
                    repaired[current].map(f64::from),
                    coords[next].map(f64::from),
                    box_,
                    1.0,
                )
                .map_err(|err| anyhow!("failed to make trajectory whole: {err}"))?;
                repaired[next] = [
                    repaired[current][0] + delta[0] as f32,
                    repaired[current][1] + delta[1] as f32,
                    repaired[current][2] + delta[2] as f32,
                ];
                seen[next] = true;
                stack.push(next);
            }
        }
    }
    Ok(repaired)
}

fn adjacency_list(n_coords: usize, connections: &[(usize, usize)]) -> Result<Vec<Vec<usize>>> {
    let mut adjacency = vec![Vec::new(); n_coords];
    for &(left, right) in connections {
        if left >= n_coords || right >= n_coords {
            return Err(anyhow!(
                "whole-trajectory connection {left}-{right} exceeds coordinate count {n_coords}"
            ));
        }
        adjacency[left].push(right);
        adjacency[right].push(left);
    }
    Ok(adjacency)
}

fn validate_source_indices(indices: &[usize], source_atom_count: usize) -> Result<()> {
    for &idx in indices {
        if idx >= source_atom_count {
            return Err(anyhow!(
                "trajectory_source.atom_indices contains index {idx} but trajectory has {source_atom_count} atoms"
            ));
        }
    }
    Ok(())
}

fn normalized_pair(left: usize, right: usize) -> (usize, usize) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}
