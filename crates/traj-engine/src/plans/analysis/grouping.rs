use std::collections::BTreeMap;

use traj_core::error::{TrajError, TrajResult};
use traj_core::selection::Selection;
use traj_core::system::System;

#[derive(Debug, Clone, Copy)]
pub enum GroupBy {
    Atom,
    Resid,
    Chain,
    ResidChain,
}

impl GroupBy {
    pub fn parse(value: &str) -> TrajResult<Self> {
        match value {
            "atom" => Ok(GroupBy::Atom),
            "resid" => Ok(GroupBy::Resid),
            "chain" => Ok(GroupBy::Chain),
            "resid_chain" | "chain_resid" => Ok(GroupBy::ResidChain),
            _ => Err(TrajError::Parse(format!(
                "invalid group_by '{value}' (use atom, resid, chain, resid_chain)"
            ))),
        }
    }
}

#[derive(Clone)]
pub struct GroupSpec {
    pub selection: Selection,
    pub group_by: GroupBy,
    pub group_types: Option<Vec<usize>>,
}

impl GroupSpec {
    pub fn new(selection: Selection, group_by: GroupBy) -> Self {
        Self {
            selection,
            group_by,
            group_types: None,
        }
    }

    pub fn with_group_types(mut self, types: Vec<usize>) -> Self {
        self.group_types = Some(types);
        self
    }

    pub fn build(&self, system: &System) -> TrajResult<GroupMap> {
        if self.selection.indices.is_empty() {
            return Err(TrajError::Mismatch("empty selection for grouping".into()));
        }
        let mut groups = Vec::new();
        match self.group_by {
            GroupBy::Atom => {
                for &idx in self.selection.indices.iter() {
                    groups.push(vec![idx as usize]);
                }
            }
            GroupBy::Chain => {
                let mut map: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
                for &idx in self.selection.indices.iter() {
                    let atom_idx = idx as usize;
                    let chain = system.atoms.chain_id[atom_idx];
                    map.entry(chain).or_default().push(atom_idx);
                }
                for (_chain, atoms) in map {
                    groups.push(atoms);
                }
            }
            GroupBy::Resid => {
                let mut map: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
                for &idx in self.selection.indices.iter() {
                    let atom_idx = idx as usize;
                    let resid = system.atoms.resid[atom_idx];
                    map.entry(resid).or_default().push(atom_idx);
                }
                for (_resid, atoms) in map {
                    groups.push(atoms);
                }
            }
            GroupBy::ResidChain => {
                let mut map: BTreeMap<(u32, i32), Vec<usize>> = BTreeMap::new();
                for &idx in self.selection.indices.iter() {
                    let atom_idx = idx as usize;
                    let chain = system.atoms.chain_id[atom_idx];
                    let resid = system.atoms.resid[atom_idx];
                    map.entry((chain, resid)).or_default().push(atom_idx);
                }
                for (_key, atoms) in map {
                    groups.push(atoms);
                }
            }
        }
        if groups.is_empty() {
            return Err(TrajError::Mismatch("no groups produced".into()));
        }
        let group_types = if let Some(types) = &self.group_types {
            if types.len() != groups.len() {
                return Err(TrajError::Mismatch(
                    "group_types length does not match group count".into(),
                ));
            }
            Some(types.clone())
        } else {
            None
        };
        Ok(GroupMap {
            groups,
            group_types,
        })
    }
}

#[derive(Clone)]
pub struct GroupMap {
    pub groups: Vec<Vec<usize>>,
    pub group_types: Option<Vec<usize>>,
}

impl GroupMap {
    pub fn n_groups(&self) -> usize {
        self.groups.len()
    }

    pub fn type_counts(&self) -> Vec<usize> {
        match &self.group_types {
            Some(types) => {
                let max = types.iter().copied().max().unwrap_or(0);
                let mut counts = vec![0usize; max + 1];
                for &t in types {
                    counts[t] += 1;
                }
                counts
            }
            None => vec![self.groups.len()],
        }
    }

    pub fn type_ids(&self) -> Vec<usize> {
        match &self.group_types {
            Some(types) => types.clone(),
            None => vec![0usize; self.groups.len()],
        }
    }
}
