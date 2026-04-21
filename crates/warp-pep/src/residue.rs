//! Atom and Residue data structures for peptide building.

use crate::coord::Vec3;
use crate::non_standard::NonStdResidue;
pub use crate::types::{parse_amber_name, AmberVariant, ResName};

/// A single atom with name, element, and 3D position.
#[derive(Debug, Clone)]
pub struct Atom {
    pub name: String,
    pub element: String,
    pub coord: Vec3,
    pub serial: u32,
    pub bfactor: f64,
    pub occupancy: f64,
}

impl Atom {
    pub fn new(name: &str, element: &str, coord: Vec3) -> Self {
        Self {
            name: name.to_string(),
            element: element.to_string(),
            coord,
            serial: 0,
            bfactor: 0.0,
            occupancy: 1.0,
        }
    }
}

/// A residue: named, numbered, with atoms.
#[derive(Debug, Clone)]
pub struct Residue {
    pub name: ResName,
    pub seq_id: i32,
    pub atoms: Vec<Atom>,
    /// Amber force field naming variant (e.g. CYX, HID, HIE).
    pub variant: Option<AmberVariant>,
    /// Non-standard residue identity (e.g. MSE, PCA).
    pub non_std: Option<NonStdResidue>,
}

impl Residue {
    pub fn new(name: ResName, seq_id: i32) -> Self {
        Self {
            name,
            seq_id,
            atoms: Vec::new(),
            variant: None,
            non_std: None,
        }
    }

    pub fn add_atom(&mut self, atom: Atom) {
        self.atoms.push(atom);
    }

    /// Find atom by name, returning its coordinate.
    pub fn atom_coord(&self, name: &str) -> Option<Vec3> {
        self.atoms.iter().find(|a| a.name == name).map(|a| a.coord)
    }

    /// Get mutable reference to an atom by name.
    pub fn atom_mut(&mut self, name: &str) -> Option<&mut Atom> {
        self.atoms.iter_mut().find(|a| a.name == name)
    }

    /// Amber-convention residue name. Uses non_std > variant > canonical priority.
    pub fn amber_name(&self) -> &str {
        if let Some(nsr) = self.non_std {
            return nsr.as_str();
        }
        match self.variant {
            Some(v) => v.as_str(),
            None => self.name.as_str(),
        }
    }
}

/// A chain of residues (single chain for now).
#[derive(Debug, Clone)]
pub struct Chain {
    pub id: char,
    pub residues: Vec<Residue>,
}

impl Chain {
    pub fn new(id: char) -> Self {
        Self {
            id,
            residues: Vec::new(),
        }
    }

    pub fn last_residue(&self) -> Option<&Residue> {
        self.residues.last()
    }

    pub fn last_residue_mut(&mut self) -> Option<&mut Residue> {
        self.residues.last_mut()
    }

    /// Insert a residue at `index` (0-based). Panics if index > len.
    pub fn insert_residue(&mut self, index: usize, residue: Residue) {
        self.residues.insert(index, residue);
    }

    /// Remove and return the residue at `index` (0-based). Panics if out of bounds.
    pub fn delete_residue(&mut self, index: usize) -> Residue {
        self.residues.remove(index)
    }
}

/// Disulfide bond between two residues.
#[derive(Debug, Clone)]
pub struct SSBond {
    pub chain1: char,
    pub resid1: i32,
    pub chain2: char,
    pub resid2: i32,
}

/// Top-level structure holding chains.
#[derive(Debug, Clone)]
pub struct Structure {
    pub chains: Vec<Chain>,
    /// Detected disulfide bonds.
    pub ssbonds: Vec<SSBond>,
}

impl Structure {
    pub fn new() -> Self {
        Self {
            chains: vec![Chain::new('A')],
            ssbonds: Vec::new(),
        }
    }

    /// Create a structure with no chains (for multi-chain building).
    pub fn new_empty() -> Self {
        Self {
            chains: Vec::new(),
            ssbonds: Vec::new(),
        }
    }

    pub fn chain_a(&self) -> &Chain {
        &self.chains[0]
    }

    pub fn chain_a_mut(&mut self) -> &mut Chain {
        &mut self.chains[0]
    }

    /// Get a chain by its ID character.
    pub fn chain_by_id(&self, id: char) -> Option<&Chain> {
        self.chains.iter().find(|c| c.id == id)
    }

    /// Add a new chain. Panics if chain ID already exists.
    pub fn add_chain(&mut self, chain: Chain) {
        assert!(
            !self.chains.iter().any(|c| c.id == chain.id),
            "duplicate chain ID '{}'",
            chain.id
        );
        self.chains.push(chain);
    }

    /// Total number of residues across all chains.
    pub fn total_residues(&self) -> usize {
        self.chains.iter().map(|c| c.residues.len()).sum()
    }

    /// One-letter amino acid sequence across all chains (no separator).
    /// Cap residues (ACE/NME) are excluded.
    pub fn sequence(&self) -> String {
        self.chains
            .iter()
            .flat_map(|c| {
                c.residues
                    .iter()
                    .filter(|r| !r.name.is_cap())
                    .map(|r| r.name.to_one_letter())
            })
            .collect()
    }

    /// Per-chain sequences as `Vec<(chain_id, sequence)>`.
    /// Cap residues (ACE/NME) are excluded.
    pub fn sequences_by_chain(&self) -> Vec<(char, String)> {
        self.chains
            .iter()
            .map(|c| {
                let seq: String = c
                    .residues
                    .iter()
                    .filter(|r| !r.name.is_cap())
                    .map(|r| r.name.to_one_letter())
                    .collect();
                (c.id, seq)
            })
            .collect()
    }

    /// Renumber all residues globally starting from `start` (1-based default).
    pub fn renumber(&mut self, start: i32) {
        let mut n = start;
        for chain in &mut self.chains {
            for res in &mut chain.residues {
                res.seq_id = n;
                n += 1;
            }
        }
    }

    /// Renumber residues within each chain independently, each starting from `start`.
    pub fn renumber_per_chain(&mut self, start: i32) {
        for chain in &mut self.chains {
            let mut n = start;
            for res in &mut chain.residues {
                res.seq_id = n;
                n += 1;
            }
        }
    }
}
