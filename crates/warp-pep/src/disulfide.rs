//! Disulfide bond detection and CYS→CYX annotation (Amber convention).

use crate::residue::{AmberVariant, ResName, SSBond, Structure};

/// S–S bond cutoff in Ångström (typical S–S ≈ 2.04 Å).
const SS_CUTOFF: f64 = 2.5;

/// Detect disulfide bonds between CYS residues by SG–SG distance.
///
/// Marks bonded cysteines as CYX and records SSBond entries on the structure.
/// Returns the number of disulfide bonds found.
pub fn detect_disulfides(struc: &mut Structure) -> usize {
    // Clear prior SS metadata so repeated calls are idempotent.
    struc.ssbonds.clear();
    for chain in &mut struc.chains {
        for res in &mut chain.residues {
            if res.name == ResName::CYS && res.variant == Some(AmberVariant::CYX) {
                res.variant = None;
            }
        }
    }

    // Collect (chain_idx, res_idx, SG coord) for every CYS with an SG atom.
    let mut cys_sg = Vec::new();
    for (ci, chain) in struc.chains.iter().enumerate() {
        for (ri, res) in chain.residues.iter().enumerate() {
            if res.name == ResName::CYS {
                if let Some(sg) = res.atom_coord("SG") {
                    cys_sg.push((ci, ri, sg));
                }
            }
        }
    }

    let mut pairs = Vec::new();
    let mut used = vec![false; cys_sg.len()];

    for i in 0..cys_sg.len() {
        if used[i] {
            continue;
        }
        for j in (i + 1)..cys_sg.len() {
            if used[j] {
                continue;
            }
            let dist = cys_sg[i].2.sub(cys_sg[j].2).length();
            if dist < SS_CUTOFF {
                pairs.push((i, j));
                used[i] = true;
                used[j] = true;
                break; // each CYS participates in at most one SS bond
            }
        }
    }

    let count = pairs.len();
    for (i, j) in pairs {
        let (ci1, ri1, _) = cys_sg[i];
        let (ci2, ri2, _) = cys_sg[j];

        struc.chains[ci1].residues[ri1].variant = Some(AmberVariant::CYX);
        struc.chains[ci2].residues[ri2].variant = Some(AmberVariant::CYX);

        struc.ssbonds.push(SSBond {
            chain1: struc.chains[ci1].id,
            resid1: struc.chains[ci1].residues[ri1].seq_id,
            chain2: struc.chains[ci2].id,
            resid2: struc.chains[ci2].residues[ri2].seq_id,
        });
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::make_extended_structure;
    use crate::coord::Vec3;

    #[test]
    fn test_no_disulfides_in_polyala() {
        let mut struc = make_extended_structure("AAAA").unwrap();
        assert_eq!(detect_disulfides(&mut struc), 0);
        assert!(struc.ssbonds.is_empty());
    }

    #[test]
    fn test_cys_too_far_apart() {
        // Extended chain — CYS SG atoms far apart
        let mut struc = make_extended_structure("CAAAC").unwrap();
        assert_eq!(detect_disulfides(&mut struc), 0);
        // Should remain CYS (no variant)
        for res in &struc.chain_a().residues {
            if res.name == ResName::CYS {
                assert!(res.variant.is_none());
            }
        }
    }

    #[test]
    fn test_forced_close_sg_detected() {
        // Manually build two CYS with SG atoms within cutoff
        let mut struc = make_extended_structure("CC").unwrap();
        // Move second CYS SG close to first CYS SG
        let sg1 = struc.chain_a().residues[0]
            .atom_coord("SG")
            .unwrap();
        let close = sg1.add(Vec3::new(2.0, 0.0, 0.0)); // 2.0 Å apart

        let sg2 = struc.chain_a_mut().residues[1]
            .atom_mut("SG")
            .unwrap();
        sg2.coord = close;

        assert_eq!(detect_disulfides(&mut struc), 1);
        assert_eq!(struc.ssbonds.len(), 1);
        assert_eq!(
            struc.chain_a().residues[0].variant,
            Some(AmberVariant::CYX)
        );
        assert_eq!(
            struc.chain_a().residues[1].variant,
            Some(AmberVariant::CYX)
        );
    }
}
