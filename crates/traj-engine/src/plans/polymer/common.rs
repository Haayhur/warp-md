use std::collections::BTreeMap;

use traj_core::error::{TrajError, TrajResult};
use traj_core::selection::Selection;
use traj_core::system::System;

pub struct PolymerChains {
    pub chains: Vec<Vec<usize>>, // per chain atom indices
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub offsets: Vec<u32>, // length n_chains + 1
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub indices: Vec<u32>, // flattened atom indices
    pub bond_pairs: Vec<u32>,    // flattened pairs (len = 2 * n_bonds)
    pub angle_triplets: Vec<u32>, // flattened triplets (len = 3 * n_angles)
}

impl PolymerChains {
    pub fn from_selection(system: &System, selection: &Selection) -> TrajResult<Self> {
        if selection.indices.is_empty() {
            return Err(TrajError::Mismatch(
                "empty selection for polymer analysis".into(),
            ));
        }
        let mut by_chain: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for &idx in selection.indices.iter() {
            let atom_idx = idx as usize;
            let chain_id = system.atoms.chain_id[atom_idx];
            by_chain.entry(chain_id).or_default().push(atom_idx);
        }
        let mut chains = Vec::with_capacity(by_chain.len());
        for (_chain, mut atoms) in by_chain {
            atoms.sort_by(|a, b| {
                let ra = system.atoms.resid[*a];
                let rb = system.atoms.resid[*b];
                ra.cmp(&rb).then(a.cmp(b))
            });
            chains.push(atoms);
        }
        if chains.is_empty() {
            return Err(TrajError::Mismatch("no chains in selection".into()));
        }
        let mut offsets = Vec::with_capacity(chains.len() + 1);
        let mut indices = Vec::new();
        let mut bond_pairs = Vec::new();
        let mut angle_triplets = Vec::new();
        offsets.push(0);
        for chain in &chains {
            for &idx in chain {
                indices.push(idx as u32);
            }
            if chain.len() >= 2 {
                for pair in chain.windows(2) {
                    bond_pairs.push(pair[0] as u32);
                    bond_pairs.push(pair[1] as u32);
                }
            }
            if chain.len() >= 3 {
                for trip in chain.windows(3) {
                    angle_triplets.push(trip[0] as u32);
                    angle_triplets.push(trip[1] as u32);
                    angle_triplets.push(trip[2] as u32);
                }
            }
            offsets.push(indices.len() as u32);
        }
        Ok(Self {
            chains,
            offsets,
            indices,
            bond_pairs,
            angle_triplets,
        })
    }

    pub fn ensure_equal_length(&self) -> TrajResult<usize> {
        let mut len = None;
        for chain in &self.chains {
            if chain.len() < 2 {
                return Err(TrajError::Mismatch(
                    "polymer chain must have at least 2 atoms".into(),
                ));
            }
            match len {
                None => len = Some(chain.len()),
                Some(l) if l == chain.len() => {}
                Some(_) => {
                    return Err(TrajError::Mismatch(
                        "all polymer chains must have same length".into(),
                    ))
                }
            }
        }
        Ok(len.unwrap_or(0))
    }

    pub fn n_chains(&self) -> usize {
        self.chains.len()
    }
}

pub fn histogram_centers(max_value: f32, bins: usize) -> Vec<f32> {
    let mut centers = Vec::with_capacity(bins);
    let width = max_value / bins as f32;
    for i in 0..bins {
        centers.push((i as f32 + 0.5) * width);
    }
    centers
}
