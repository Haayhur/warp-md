//! Backbone hydrogen placement.
//!
//! Adds the amide hydrogen (H) atom to backbone nitrogens. The H atom is placed
//! trans to the Cα across the peptide bond, at N–H distance 1.02 Å.
//! Proline residues lack this hydrogen (imino nitrogen).
//! The first residue gets two H atoms (H1, H2) on the amino group NH₂.

use crate::coord::{calculate_coordinates, Vec3};
use crate::residue::{Atom, ResName, Structure};

/// Standard N–H bond length in Å.
const NH_BOND: f64 = 1.02;

/// Add backbone amide H atoms to all residues (except proline).
///
/// For residue i (i > 0, not PRO): H is placed using C(i-1), CA(i-1), N(i) reference,
/// trans to the C–N–CA angle.
pub fn add_backbone_hydrogens(struc: &mut Structure) {
    for chain in &mut struc.chains {
        let n = chain.residues.len();
        if n == 0 {
            continue;
        }

        // First residue: add H1, H2 on the amino group
        {
            let first = &chain.residues[0];
            if first.name != ResName::PRO {
                if let (Some(n_coord), Some(ca_coord), Some(c_coord)) = (
                    first.atom_coord("N"),
                    first.atom_coord("CA"),
                    first.atom_coord("C"),
                ) {
                    let h1 =
                        calculate_coordinates(c_coord, ca_coord, n_coord, NH_BOND, 109.5, 60.0);
                    let h2 =
                        calculate_coordinates(c_coord, ca_coord, n_coord, NH_BOND, 109.5, -60.0);
                    chain.residues[0].atoms.push(Atom::new("H1", "H", h1));
                    chain.residues[0].atoms.push(Atom::new("H2", "H", h2));
                }
            }
        }

        // Collect previous-residue coords before mutating
        let mut prev_data: Vec<Option<(Vec3, Vec3)>> = Vec::with_capacity(n);
        prev_data.push(None); // no previous for first residue
        for i in 1..n {
            let prev = &chain.residues[i - 1];
            let data = match (prev.atom_coord("CA"), prev.atom_coord("C")) {
                (Some(ca), Some(c)) => Some((ca, c)),
                _ => None,
            };
            prev_data.push(data);
        }

        for i in 1..n {
            if chain.residues[i].name == ResName::PRO {
                continue; // Proline has no amide H
            }
            if let Some((ca_prev, c_prev)) = prev_data[i] {
                if let Some(n_coord) = chain.residues[i].atom_coord("N") {
                    let h = calculate_coordinates(ca_prev, c_prev, n_coord, NH_BOND, 119.0, 180.0);
                    chain.residues[i].atoms.push(Atom::new("H", "H", h));
                }
            }
        }
    }
}

/// Add Hα hydrogen on CA for all residues.
/// GLY gets HA2 and HA3; all others get HA.
pub fn add_ha_hydrogens(struc: &mut Structure) {
    let ca_h_bond = 1.09; // C–H bond length
    for chain in &mut struc.chains {
        for res in &mut chain.residues {
            if let (Some(n), Some(ca), Some(c)) = (
                res.atom_coord("N"),
                res.atom_coord("CA"),
                res.atom_coord("C"),
            ) {
                if res.name == ResName::GLY {
                    let ha2 = calculate_coordinates(n, c, ca, ca_h_bond, 109.5, 120.0);
                    let ha3 = calculate_coordinates(n, c, ca, ca_h_bond, 109.5, -120.0);
                    res.atoms.push(Atom::new("HA2", "H", ha2));
                    res.atoms.push(Atom::new("HA3", "H", ha3));
                } else {
                    let ha = calculate_coordinates(n, c, ca, ca_h_bond, 109.5, 120.0);
                    res.atoms.push(Atom::new("HA", "H", ha));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::make_extended_structure;

    #[test]
    fn test_backbone_h_added() {
        let mut struc = make_extended_structure("AAA").unwrap();
        add_backbone_hydrogens(&mut struc);
        let chain = struc.chain_a();
        // First residue: H1, H2
        assert!(chain.residues[0].atom_coord("H1").is_some());
        assert!(chain.residues[0].atom_coord("H2").is_some());
        // Second, third: H
        assert!(chain.residues[1].atom_coord("H").is_some());
        assert!(chain.residues[2].atom_coord("H").is_some());
    }

    #[test]
    fn test_proline_no_h() {
        let mut struc = make_extended_structure("APA").unwrap();
        add_backbone_hydrogens(&mut struc);
        // Proline (residue 2) should NOT have an amide H
        assert!(struc.chain_a().residues[1].atom_coord("H").is_none());
        // But residue 3 should
        assert!(struc.chain_a().residues[2].atom_coord("H").is_some());
    }

    #[test]
    fn test_nh_bond_length() {
        let mut struc = make_extended_structure("AA").unwrap();
        add_backbone_hydrogens(&mut struc);
        let res = &struc.chain_a().residues[1];
        let n = res.atom_coord("N").unwrap();
        let h = res.atom_coord("H").unwrap();
        let dist = n.sub(h).length();
        assert!((dist - 1.02).abs() < 0.05, "N-H bond {dist} not ~1.02");
    }

    #[test]
    fn test_ha_added() {
        let mut struc = make_extended_structure("AG").unwrap();
        add_ha_hydrogens(&mut struc);
        // ALA gets HA
        assert!(struc.chain_a().residues[0].atom_coord("HA").is_some());
        // GLY gets HA2 and HA3
        assert!(struc.chain_a().residues[1].atom_coord("HA2").is_some());
        assert!(struc.chain_a().residues[1].atom_coord("HA3").is_some());
    }

    #[test]
    fn test_ha_bond_length() {
        let mut struc = make_extended_structure("A").unwrap();
        add_ha_hydrogens(&mut struc);
        let ca = struc.chain_a().residues[0].atom_coord("CA").unwrap();
        let ha = struc.chain_a().residues[0].atom_coord("HA").unwrap();
        let dist = ca.sub(ha).length();
        assert!((dist - 1.09).abs() < 0.05, "CA-HA bond {dist} not ~1.09");
    }
}
