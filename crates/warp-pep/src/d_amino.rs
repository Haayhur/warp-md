//! D-amino acid support: mirror L-amino acid geometry at Cα.
//!
//! D-amino acids have inverted chirality at the Cα carbon. This is achieved by
//! negating the dihedral angles in the side-chain geometry, producing the mirror
//! image of the L-form.

use crate::geometry::{geometry, Geo};
use crate::residue::ResName;

/// Return geometry for the D-form of an amino acid.
///
/// D-amino acids differ from L-amino acids by having inverted chirality
/// at Cα: the side chain dihedrals are negated.
pub fn d_geometry(aa: ResName) -> Geo {
    let mut geo = geometry(aa);
    // Negate all side-chain dihedral angles to invert chirality
    for sc in &mut geo.side_chain {
        sc.dihedral = -sc.dihedral;
    }
    // Also negate the carbonyl dihedral
    geo.n_ca_c_o_diangle = -geo.n_ca_c_o_diangle;
    geo
}

/// Check if a one-letter code is lowercase (convention for D-amino acids).
/// e.g., 'a' = D-Ala, 'A' = L-Ala.
pub fn is_d_amino_acid(c: char) -> bool {
    c.is_ascii_lowercase() && ResName::from_one_letter(c.to_ascii_uppercase()).is_some()
}

/// Parse a possibly-D amino acid character. Returns (ResName, is_d).
pub fn parse_d_amino_acid(c: char) -> Option<(ResName, bool)> {
    if c.is_ascii_uppercase() {
        ResName::from_one_letter(c).map(|r| (r, false))
    } else if c.is_ascii_lowercase() {
        ResName::from_one_letter(c.to_ascii_uppercase()).map(|r| (r, true))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d_geometry_inverts_dihedrals() {
        let l_geo = geometry(ResName::ALA);
        let d_geo = d_geometry(ResName::ALA);
        // ALA has one side chain atom (CB)
        assert_eq!(l_geo.side_chain.len(), d_geo.side_chain.len());
        for (l, d) in l_geo.side_chain.iter().zip(d_geo.side_chain.iter()) {
            assert!(
                (l.dihedral + d.dihedral).abs() < 1e-10,
                "dihedrals should be negated: {} vs {}",
                l.dihedral,
                d.dihedral
            );
        }
    }

    #[test]
    fn test_d_geometry_carbonyl() {
        let l_geo = geometry(ResName::ALA);
        let d_geo = d_geometry(ResName::ALA);
        assert!((l_geo.n_ca_c_o_diangle + d_geo.n_ca_c_o_diangle).abs() < 1e-10);
    }

    #[test]
    fn test_is_d_amino_acid() {
        assert!(is_d_amino_acid('a'));
        assert!(is_d_amino_acid('g'));
        assert!(!is_d_amino_acid('A'));
        assert!(!is_d_amino_acid('1'));
        assert!(!is_d_amino_acid('x')); // X is not a standard AA
    }

    #[test]
    fn test_parse_d_amino_acid() {
        let (name, is_d) = parse_d_amino_acid('a').unwrap();
        assert_eq!(name, ResName::ALA);
        assert!(is_d);

        let (name, is_d) = parse_d_amino_acid('A').unwrap();
        assert_eq!(name, ResName::ALA);
        assert!(!is_d);
    }

    #[test]
    fn test_d_geometry_trp_all_inverted() {
        let l = geometry(ResName::TRP);
        let d = d_geometry(ResName::TRP);
        for (la, da) in l.side_chain.iter().zip(d.side_chain.iter()) {
            assert!(
                (la.dihedral + da.dihedral).abs() < 1e-10,
                "{}: {} vs {}",
                la.name,
                la.dihedral,
                da.dihedral
            );
        }
    }
}
