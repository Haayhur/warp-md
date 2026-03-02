//! JSON-based structured input for peptide building.
//!
//! Single-chain schema (all fields optional except `residues`):
//! ```json
//! {
//!   "residues": ["ALA", "CYX", "HID", "GLU"],
//!   "preset":   "alpha-helix",
//!   "oxt":      true,
//!   "detect_ss": true,
//!   "mutations": ["A1G", "H3W"],
//!   "output":   "out.pdb"
//! }
//! ```
//!
//! Multi-chain schema (use `chains` instead of `residues`):
//! ```json
//! {
//!   "chains": [
//!     { "id": "A", "residues": ["ALA", "CYS", "GLU"], "preset": "alpha-helix" },
//!     { "id": "B", "residues": ["GLY", "VAL", "TRP"], "preset": "beta-sheet" }
//!   ],
//!   "oxt":      true,
//!   "detect_ss": true,
//!   "output":   "out.pdb"
//! }
//! ```
//!
//! Presets: "extended", "alpha-helix", "beta-sheet", "polyproline".
//! If `preset` is set, `phi`/`psi` are ignored.

use serde::Deserialize;
use std::path::Path;

use crate::builder::{self, parse_three_letter_sequence, ChainSpec, RamaPreset, ResSpec};
use crate::disulfide;
use crate::mutation;
use crate::residue::Structure;

/// Per-chain definition for multi-chain JSON specs.
#[derive(Debug, Deserialize)]
pub struct ChainDef {
    /// Chain ID character (e.g. "A", "B").
    pub id: String,
    /// Three-letter residue names.
    pub residues: Vec<String>,
    /// Optional Ramachandran preset for this chain.
    #[serde(default)]
    pub preset: Option<String>,
}

/// Top-level JSON build specification.
#[derive(Debug, Deserialize)]
pub struct BuildSpec {
    /// Three-letter residue names (single-chain mode).
    /// Mutually exclusive with `chains`.
    #[serde(default)]
    pub residues: Vec<String>,

    /// Multi-chain definitions. If present, `residues`/`phi`/`psi`/`omega` are ignored.
    #[serde(default)]
    pub chains: Vec<ChainDef>,

    /// Ramachandran preset: "extended", "alpha-helix", "beta-sheet", "polyproline".
    /// If set, phi/psi fields are ignored.
    #[serde(default)]
    pub preset: Option<String>,

    /// Phi angles (length = residues.len() - 1). Omit for extended conformation.
    #[serde(default)]
    pub phi: Option<Vec<f64>>,

    /// Psi angles (length = residues.len() - 1). Omit for extended conformation.
    #[serde(default)]
    pub psi: Option<Vec<f64>>,

    /// Omega angles (length = residues.len() - 1). Omit for default (180°).
    #[serde(default)]
    pub omega: Option<Vec<f64>>,

    /// Add terminal OXT oxygen.
    #[serde(default)]
    pub oxt: bool,

    /// Detect disulfide bonds and mark CYS→CYX.
    #[serde(default)]
    pub detect_ss: bool,

    /// Post-build mutations (e.g. ["A5G", "L10W"]).
    #[serde(default)]
    pub mutations: Vec<String>,

    /// Output file path. If omitted, caller decides (e.g. stdout).
    #[serde(default)]
    pub output: Option<String>,

    /// Output format (pdb, pdbx, xyz, gro, mol2, crd, lammps). Auto-detected if omitted.
    #[serde(default)]
    pub format: Option<String>,
}

impl BuildSpec {
    /// Load from a JSON file.
    pub fn from_file(path: &str) -> Result<Self, String> {
        let p = Path::new(path);
        let text =
            std::fs::read_to_string(p).map_err(|e| format!("failed to read '{}': {}", path, e))?;
        serde_json::from_str(&text).map_err(|e| format!("invalid JSON in '{}': {}", path, e))
    }

    /// Parse the residues list into ResSpecs.
    pub fn to_specs(&self) -> Result<Vec<ResSpec>, String> {
        let joined = self.residues.join("-");
        parse_three_letter_sequence(&joined)
    }

    /// Execute the full build pipeline: construct → mutate → detect_ss → oxt.
    pub fn execute(&self) -> Result<Structure, String> {
        // Multi-chain mode
        if !self.chains.is_empty() {
            return self.execute_multi_chain();
        }

        if self.residues.is_empty() {
            return Err("must provide 'residues' or 'chains'".into());
        }

        let specs = self.to_specs()?;

        // Preset takes priority over explicit angles.
        let mut struc = if let Some(ref preset_str) = self.preset {
            let preset = RamaPreset::from_str(preset_str)
                .ok_or_else(|| format!("unknown preset '{}'", preset_str))?;
            builder::make_preset_structure_from_specs(&specs, preset)?
        } else {
            match (&self.phi, &self.psi) {
                (Some(phi), Some(psi)) => {
                    builder::make_structure_from_specs(&specs, phi, psi, self.omega.as_deref())?
                }
                (None, None) => builder::make_extended_structure_from_specs(&specs)?,
                _ => return Err("must provide both phi and psi or neither".into()),
            }
        };

        for spec_str in &self.mutations {
            let (from, pos, to) = mutation::parse_mutation_spec(spec_str)?;
            mutation::mutate_residue_checked(&mut struc, Some(from), pos, to)?;
        }

        if self.oxt {
            builder::add_terminal_oxt(&mut struc);
        }

        if self.detect_ss {
            disulfide::detect_disulfides(&mut struc);
        }

        Ok(struc)
    }

    fn execute_multi_chain(&self) -> Result<Structure, String> {
        let mut chain_specs = Vec::new();
        for cdef in &self.chains {
            if cdef.id.len() != 1 {
                return Err(format!("chain id must be one character, got '{}'", cdef.id));
            }
            let id = cdef.id.chars().next().unwrap();
            let joined = cdef.residues.join("-");
            let residues = parse_three_letter_sequence(&joined)?;
            let preset = match &cdef.preset {
                Some(p) => Some(
                    RamaPreset::from_str(p)
                        .ok_or_else(|| format!("unknown preset '{}' for chain '{}'", p, id))?,
                ),
                None => None,
            };
            chain_specs.push(ChainSpec {
                id,
                residues,
                preset,
            });
        }
        let mut struc = builder::make_multi_chain_structure(&chain_specs)?;

        for spec_str in &self.mutations {
            let (from, pos, to) = mutation::parse_mutation_spec(spec_str)?;
            mutation::mutate_residue_checked(&mut struc, Some(from), pos, to)?;
        }

        if self.oxt {
            builder::add_terminal_oxt(&mut struc);
        }

        if self.detect_ss {
            disulfide::detect_disulfides(&mut struc);
        }

        Ok(struc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::residue::ResName;

    #[test]
    fn test_build_spec_execute_extended() {
        let spec = BuildSpec {
            residues: vec!["ALA".into(), "CYX".into(), "HID".into()],
            chains: vec![],
            preset: None,
            phi: None,
            psi: None,
            omega: None,
            oxt: true,
            detect_ss: false,
            mutations: vec![],
            output: None,
            format: None,
        };
        let struc = spec.execute().unwrap();
        assert_eq!(struc.chain_a().residues.len(), 3);
        assert_eq!(struc.chain_a().residues[1].amber_name(), "CYX");
        assert_eq!(struc.chain_a().residues[2].amber_name(), "HID");
        // OXT should be on last residue
        assert!(struc.chain_a().residues[2].atom_coord("OXT").is_some());
    }

    #[test]
    fn test_build_spec_with_mutation() {
        let spec = BuildSpec {
            residues: vec!["ALA".into(), "ALA".into(), "ALA".into()],
            chains: vec![],
            preset: None,
            phi: None,
            psi: None,
            omega: None,
            oxt: false,
            detect_ss: false,
            mutations: vec!["A2G".into()],
            output: None,
            format: None,
        };
        let struc = spec.execute().unwrap();
        assert_eq!(struc.chain_a().residues[1].name, ResName::GLY);
    }

    #[test]
    fn test_build_spec_deserialize() {
        let json = r#"{
            "residues": ["ALA", "CYX", "HIE"],
            "oxt": true,
            "detect_ss": true
        }"#;
        let spec: BuildSpec = serde_json::from_str(json).unwrap();
        assert_eq!(spec.residues.len(), 3);
        assert!(spec.oxt);
        assert!(spec.detect_ss);
        assert!(spec.phi.is_none());
    }

    #[test]
    fn test_build_spec_preset_alpha_helix() {
        let json = r#"{
            "residues": ["ALA", "ALA", "ALA", "ALA"],
            "preset": "alpha-helix",
            "oxt": true
        }"#;
        let spec: BuildSpec = serde_json::from_str(json).unwrap();
        let struc = spec.execute().unwrap();
        assert_eq!(struc.chain_a().residues.len(), 4);
        assert!(struc.chain_a().residues[3].atom_coord("OXT").is_some());
    }

    #[test]
    fn test_build_spec_multi_chain() {
        let json = r#"{
            "chains": [
                { "id": "A", "residues": ["ALA", "GLY"], "preset": "alpha-helix" },
                { "id": "B", "residues": ["VAL", "TRP", "SER"] }
            ],
            "oxt": true
        }"#;
        let spec: BuildSpec = serde_json::from_str(json).unwrap();
        let struc = spec.execute().unwrap();
        assert_eq!(struc.chains.len(), 2);
        assert_eq!(struc.chain_by_id('A').unwrap().residues.len(), 2);
        assert_eq!(struc.chain_by_id('B').unwrap().residues.len(), 3);
    }
}
