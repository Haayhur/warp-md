//! Structure analysis: phi/psi dihedrals, RMSD, radius of gyration.

use crate::coord::{calc_dihedral, Vec3};
use crate::residue::{Chain, Structure};

/// Backbone dihedral angles for a single residue.
#[derive(Debug, Clone, Copy)]
pub struct PhiPsi {
    /// φ (phi): C(i-1)–N(i)–CA(i)–C(i) dihedral. `None` for first residue.
    pub phi: Option<f64>,
    /// ψ (psi): N(i)–CA(i)–C(i)–N(i+1) dihedral. `None` for last residue.
    pub psi: Option<f64>,
    /// ω (omega): CA(i-1)–C(i-1)–N(i)–CA(i). `None` for first residue.
    pub omega: Option<f64>,
}

/// Measure phi/psi/omega for every residue in a single chain.
pub fn measure_phi_psi(chain: &Chain) -> Vec<PhiPsi> {
    let n = chain.residues.len();
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let res = &chain.residues[i];

        let phi = if i > 0 {
            let prev = &chain.residues[i - 1];
            match (
                prev.atom_coord("C"),
                res.atom_coord("N"),
                res.atom_coord("CA"),
                res.atom_coord("C"),
            ) {
                (Some(c_prev), Some(n_i), Some(ca_i), Some(c_i)) => {
                    Some(calc_dihedral(c_prev, n_i, ca_i, c_i))
                }
                _ => None,
            }
        } else {
            None
        };

        let psi = if i + 1 < n {
            let next = &chain.residues[i + 1];
            match (
                res.atom_coord("N"),
                res.atom_coord("CA"),
                res.atom_coord("C"),
                next.atom_coord("N"),
            ) {
                (Some(n_i), Some(ca_i), Some(c_i), Some(n_next)) => {
                    Some(calc_dihedral(n_i, ca_i, c_i, n_next))
                }
                _ => None,
            }
        } else {
            None
        };

        let omega = if i > 0 {
            let prev = &chain.residues[i - 1];
            match (
                prev.atom_coord("CA"),
                prev.atom_coord("C"),
                res.atom_coord("N"),
                res.atom_coord("CA"),
            ) {
                (Some(ca_prev), Some(c_prev), Some(n_i), Some(ca_i)) => {
                    Some(calc_dihedral(ca_prev, c_prev, n_i, ca_i))
                }
                _ => None,
            }
        } else {
            None
        };

        result.push(PhiPsi { phi, psi, omega });
    }
    result
}

/// Measure phi/psi for all chains, returned as `Vec<(chain_id, Vec<PhiPsi>)>`.
pub fn measure_all_phi_psi(struc: &Structure) -> Vec<(char, Vec<PhiPsi>)> {
    struc
        .chains
        .iter()
        .map(|c| (c.id, measure_phi_psi(c)))
        .collect()
}

/// RMSD between two coordinate sets of equal length.
/// Returns `None` if lengths differ or are zero.
pub fn rmsd_coords(a: &[Vec3], b: &[Vec3]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(va, vb)| {
            let d = va.sub(*vb);
            d.dot(d)
        })
        .sum();
    Some((sum / a.len() as f64).sqrt())
}

/// RMSD of all atoms between two structures (must have identical atom counts per residue/chain).
pub fn rmsd_all_atoms(a: &Structure, b: &Structure) -> Option<f64> {
    let ca: Vec<Vec3> = a
        .chains
        .iter()
        .flat_map(|c| {
            c.residues
                .iter()
                .flat_map(|r| r.atoms.iter().map(|at| at.coord))
        })
        .collect();
    let cb: Vec<Vec3> = b
        .chains
        .iter()
        .flat_map(|c| {
            c.residues
                .iter()
                .flat_map(|r| r.atoms.iter().map(|at| at.coord))
        })
        .collect();
    rmsd_coords(&ca, &cb)
}

/// RMSD over Cα atoms only between two structures.
pub fn rmsd_ca(a: &Structure, b: &Structure) -> Option<f64> {
    let ca: Vec<Vec3> = a
        .chains
        .iter()
        .flat_map(|c| c.residues.iter().filter_map(|r| r.atom_coord("CA")))
        .collect();
    let cb: Vec<Vec3> = b
        .chains
        .iter()
        .flat_map(|c| c.residues.iter().filter_map(|r| r.atom_coord("CA")))
        .collect();
    rmsd_coords(&ca, &cb)
}

/// Radius of gyration over all atoms.
pub fn radius_of_gyration(struc: &Structure) -> f64 {
    let coords: Vec<Vec3> = struc
        .chains
        .iter()
        .flat_map(|c| {
            c.residues
                .iter()
                .flat_map(|r| r.atoms.iter().map(|a| a.coord))
        })
        .collect();
    if coords.is_empty() {
        return 0.0;
    }
    let n = coords.len() as f64;
    let com = coords
        .iter()
        .fold(Vec3::zero(), |acc, c| acc.add(*c))
        .scale(1.0 / n);
    let sum_sq: f64 = coords.iter().map(|c| c.sub(com).dot(c.sub(com))).sum();
    (sum_sq / n).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{make_extended_structure, make_preset_structure, RamaPreset};

    #[test]
    fn test_phi_psi_extended() {
        let struc = make_extended_structure("AAAA").unwrap();
        let chain = struc.chain_a();
        let angles = measure_phi_psi(chain);
        assert_eq!(angles.len(), 4);
        // First residue: no phi
        assert!(angles[0].phi.is_none());
        assert!(angles[0].psi.is_some());
        // Last residue: no psi
        assert!(angles[3].phi.is_some());
        assert!(angles[3].psi.is_none());
    }

    #[test]
    fn test_phi_psi_alpha_helix() {
        let struc = make_preset_structure("AAAA", RamaPreset::AlphaHelix).unwrap();
        let angles = measure_phi_psi(struc.chain_a());
        // Interior residues should have |phi| near 57°
        // (sign depends on dihedral convention; calc_dihedral returns [0,360))
        for pp in &angles[1..3] {
            if let Some(phi) = pp.phi {
                let normed = if phi > 180.0 { phi - 360.0 } else { phi };
                assert!(
                    normed.abs() > 40.0 && normed.abs() < 75.0,
                    "phi={phi} (normed={normed}) not near ±57°"
                );
            }
        }
    }

    #[test]
    fn test_rmsd_self_zero() {
        let struc = make_extended_structure("AAA").unwrap();
        let r = rmsd_all_atoms(&struc, &struc).unwrap();
        assert!(r < 1e-10, "self-RMSD should be 0, got {r}");
    }

    #[test]
    fn test_rmsd_ca_self_zero() {
        let struc = make_extended_structure("AAAA").unwrap();
        let r = rmsd_ca(&struc, &struc).unwrap();
        assert!(r < 1e-10);
    }

    #[test]
    fn test_rmsd_different_structures() {
        let a = make_preset_structure("AAA", RamaPreset::AlphaHelix).unwrap();
        let b = make_preset_structure("AAA", RamaPreset::BetaSheet).unwrap();
        let r = rmsd_ca(&a, &b).unwrap();
        assert!(
            r > 0.1,
            "different conformations should have non-zero RMSD, got {r}"
        );
    }

    #[test]
    fn test_rmsd_mismatched_lengths() {
        let a = make_extended_structure("AA").unwrap();
        let b = make_extended_structure("AAA").unwrap();
        assert!(rmsd_all_atoms(&a, &b).is_none());
    }

    #[test]
    fn test_radius_of_gyration() {
        let struc = make_extended_structure("AAAA").unwrap();
        let rg = radius_of_gyration(&struc);
        assert!(rg > 0.0, "Rg should be positive");
    }

    #[test]
    fn test_omega_near_180() {
        let struc = make_extended_structure("AAA").unwrap();
        let angles = measure_phi_psi(struc.chain_a());
        // Omega for residues 1,2 should be near 180°
        for pp in &angles[1..] {
            if let Some(omega) = pp.omega {
                assert!(
                    (omega - 180.0).abs() < 10.0
                        || (omega + 180.0).abs() < 10.0
                        || (omega - 360.0).abs() < 10.0,
                    "omega={omega} not near 180°"
                );
            }
        }
    }
}
