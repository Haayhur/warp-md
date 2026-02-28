//! Non-standard amino acid residues: MSE (selenomethionine), PCA (pyroglutamic acid).
//!
//! These are common non-standard residues in PDB structures.
//! MSE is methionine with sulfur replaced by selenium.
//! PCA is a cyclized glutamic acid found at some N-termini.

use crate::geometry::{Geo, SideChainAtom, geometry};
use crate::residue::ResName;

/// Non-standard residue identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonStdResidue {
    /// Selenomethionine (Se replaces SD in MET).
    MSE,
    /// Pyroglutamic acid (cyclized N-terminal GLU).
    PCA,
}

impl NonStdResidue {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MSE => "MSE",
            Self::PCA => "PCA",
        }
    }

    /// Parse a non-standard residue name.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "MSE" => Some(Self::MSE),
            "PCA" => Some(Self::PCA),
            _ => None,
        }
    }

    /// Canonical standard amino acid this maps to.
    pub fn canonical(self) -> ResName {
        match self {
            Self::MSE => ResName::MET,
            Self::PCA => ResName::GLU,
        }
    }
}

/// Return geometry for MSE (selenomethionine).
/// Same as MET but SD → SE with Se–C bond length 1.95 Å.
pub fn mse_geometry() -> Geo {
    let mut geo = geometry(ResName::MET);
    // Replace SD with SE
    for sc in &mut geo.side_chain {
        if sc.name == "SD" {
            *sc = SideChainAtom::new("SE", "SE", sc.parents, 1.95, sc.angle, sc.dihedral);
        }
    }
    // Update CE parent reference (if it references SD)
    for sc in &mut geo.side_chain {
        if sc.name == "CE" {
            let (a, b, c) = sc.parents;
            if c == "SD" {
                sc.parents = (a, b, "SE");
            }
        }
    }
    geo
}

/// Return geometry for PCA (pyroglutamic acid).
/// Approximated as GLU with a CD–N cyclization (5-membered ring).
/// Uses GLU geometry as base; the ring closure is approximate.
pub fn pca_geometry() -> Geo {
    let mut geo = geometry(ResName::GLU);
    // PCA lacks OE2 (ring closure replaces it)
    geo.side_chain.retain(|sc| sc.name != "OE2");
    geo
}

/// Check if a residue name string is a known non-standard residue.
pub fn is_non_standard(name: &str) -> bool {
    NonStdResidue::from_str(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_has_selenium() {
        let geo = mse_geometry();
        assert!(geo.side_chain.iter().any(|sc| sc.name == "SE" && sc.element == "SE"));
        assert!(!geo.side_chain.iter().any(|sc| sc.name == "SD"));
    }

    #[test]
    fn test_mse_ce_parent() {
        let geo = mse_geometry();
        let ce = geo.side_chain.iter().find(|sc| sc.name == "CE").unwrap();
        // CE should reference SE, not SD
        assert_eq!(ce.parents.2, "SE");
    }

    #[test]
    fn test_pca_no_oe2() {
        let geo = pca_geometry();
        assert!(!geo.side_chain.iter().any(|sc| sc.name == "OE2"));
        // Should still have OE1
        assert!(geo.side_chain.iter().any(|sc| sc.name == "OE1"));
    }

    #[test]
    fn test_non_std_parse() {
        assert_eq!(NonStdResidue::from_str("MSE"), Some(NonStdResidue::MSE));
        assert_eq!(NonStdResidue::from_str("PCA"), Some(NonStdResidue::PCA));
        assert_eq!(NonStdResidue::from_str("ALA"), None);
    }

    #[test]
    fn test_canonical_mapping() {
        assert_eq!(NonStdResidue::MSE.canonical(), ResName::MET);
        assert_eq!(NonStdResidue::PCA.canonical(), ResName::GLU);
    }

    #[test]
    fn test_is_non_standard() {
        assert!(is_non_standard("MSE"));
        assert!(is_non_standard("mse"));
        assert!(is_non_standard("PCA"));
        assert!(!is_non_standard("ALA"));
    }
}
