//! Terminal capping groups: ACE (acetyl, N-terminal) and NME (N-methylamide, C-terminal).

use crate::coord::calculate_coordinates;
use crate::residue::{Atom, Residue, ResName, Structure};

/// Pseudo-ResName tag for caps — ACE and NME have first-class ResName variants.

/// Add an ACE cap (acetyl group: CH3-CO-) to the N-terminus of each chain.
///
/// ACE consists of atoms: CH3, C, O placed before the first residue.
/// The peptide bond N of residue 1 connects to ACE's C.
pub fn add_ace_cap(struc: &mut Structure) {
    for chain in &mut struc.chains {
        if chain.residues.is_empty() {
            continue;
        }
        let first_res = &chain.residues[0];
        let n = match first_res.atom_coord("N") {
            Some(c) => c,
            None => continue,
        };
        let ca = match first_res.atom_coord("CA") {
            Some(c) => c,
            None => continue,
        };
        let c_first = match first_res.atom_coord("C") {
            Some(c) => c,
            None => continue,
        };

        // Place ACE C at peptide bond distance from N
        let ace_c = calculate_coordinates(c_first, ca, n, 1.33, 121.7, 180.0);
        // Place ACE O
        let ace_o = calculate_coordinates(ca, n, ace_c, 1.23, 120.5, 0.0);
        // Place ACE CH3
        let ace_ch3 = calculate_coordinates(ca, n, ace_c, 1.52, 116.6, 180.0);

        let seq_id = chain.residues[0].seq_id - 1;
        let mut cap = Residue::new(ResName::ACE, seq_id);
        cap.atoms.push(Atom::new("CH3", "C", ace_ch3));
        cap.atoms.push(Atom::new("C", "C", ace_c));
        cap.atoms.push(Atom::new("O", "O", ace_o));

        chain.insert_residue(0, cap);
    }
}

/// Add an NME cap (N-methylamide: -NH-CH3) to the C-terminus of each chain.
///
/// NME consists of atoms: N, CH3 placed after the last residue.
pub fn add_nme_cap(struc: &mut Structure) {
    for chain in &mut struc.chains {
        if chain.residues.is_empty() {
            continue;
        }
        let last = chain.residues.len() - 1;
        let (ca, c_last, o_last) = {
            let last_res = &chain.residues[last];
            let ca = match last_res.atom_coord("CA") {
                Some(c) => c,
                None => continue,
            };
            let c_last = match last_res.atom_coord("C") {
                Some(c) => c,
                None => continue,
            };
            let o_last = match last_res.atom_coord("O") {
                Some(c) => c,
                None => continue,
            };
            (ca, c_last, o_last)
        };

        // NME forms the new C-terminus; old OXT must be removed from the
        // previous terminal residue to avoid invalid mixed terminal chemistry.
        chain.residues[last].atoms.retain(|a| a.name != "OXT");

        // Place NME N at peptide bond distance from last C
        let nme_n = calculate_coordinates(o_last, ca, c_last, 1.33, 116.6, 180.0);
        // Place NME CH3
        let nme_ch3 = calculate_coordinates(ca, c_last, nme_n, 1.46, 121.7, 180.0);

        let seq_id = chain.residues[last].seq_id + 1;
        let mut cap = Residue::new(ResName::NME, seq_id);
        cap.atoms.push(Atom::new("N", "N", nme_n));
        cap.atoms.push(Atom::new("CH3", "C", nme_ch3));

        chain.residues.push(cap);
    }
}

/// Convenience: add both ACE and NME caps.
pub fn add_caps(struc: &mut Structure) {
    add_ace_cap(struc);
    add_nme_cap(struc);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::make_extended_structure;

    #[test]
    fn test_ace_cap_atoms() {
        let mut struc = make_extended_structure("AA").unwrap();
        let before = struc.chain_a().residues.len();
        add_ace_cap(&mut struc);
        assert_eq!(struc.chain_a().residues.len(), before + 1);
        let ace = &struc.chain_a().residues[0];
        assert_eq!(ace.atoms.len(), 3); // CH3, C, O
        assert!(ace.atom_coord("CH3").is_some());
        assert!(ace.atom_coord("C").is_some());
        assert!(ace.atom_coord("O").is_some());
    }

    #[test]
    fn test_nme_cap_atoms() {
        let mut struc = make_extended_structure("AA").unwrap();
        let before = struc.chain_a().residues.len();
        add_nme_cap(&mut struc);
        assert_eq!(struc.chain_a().residues.len(), before + 1);
        let nme = struc.chain_a().residues.last().unwrap();
        assert_eq!(nme.atoms.len(), 2); // N, CH3
        assert!(nme.atom_coord("N").is_some());
        assert!(nme.atom_coord("CH3").is_some());
    }

    #[test]
    fn test_both_caps() {
        let mut struc = make_extended_structure("AA").unwrap();
        add_caps(&mut struc);
        assert_eq!(struc.chain_a().residues.len(), 4); // ACE + A + A + NME
    }

    #[test]
    fn test_ace_bond_distance() {
        let mut struc = make_extended_structure("A").unwrap();
        add_ace_cap(&mut struc);
        let ace_c = struc.chain_a().residues[0].atom_coord("C").unwrap();
        let res_n = struc.chain_a().residues[1].atom_coord("N").unwrap();
        let dist = ace_c.sub(res_n).length();
        assert!((dist - 1.33).abs() < 0.1, "ACE C–N bond {dist} not ~1.33");
    }

    #[test]
    fn test_nme_bond_distance() {
        let mut struc = make_extended_structure("A").unwrap();
        add_nme_cap(&mut struc);
        let last_c = struc.chain_a().residues[0].atom_coord("C").unwrap();
        let nme_n = struc.chain_a().residues[1].atom_coord("N").unwrap();
        let dist = last_c.sub(nme_n).length();
        assert!((dist - 1.33).abs() < 0.15, "C–NME N bond {dist} not ~1.33");
    }

    #[test]
    fn test_nme_cap_removes_existing_oxt() {
        let mut struc = make_extended_structure("AA").unwrap();
        crate::builder::add_terminal_oxt(&mut struc);
        assert!(struc.chain_a().residues[1].atom_coord("OXT").is_some());

        add_nme_cap(&mut struc);

        let chain = struc.chain_a();
        let former_terminal = &chain.residues[chain.residues.len() - 2];
        assert!(former_terminal.atom_coord("OXT").is_none());
        assert_eq!(chain.residues.last().unwrap().name, ResName::NME);
    }
}
