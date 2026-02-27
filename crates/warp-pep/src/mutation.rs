//! Point-mutation support: swap a residue's side chain while keeping backbone.

use crate::coord::calculate_coordinates;
use crate::geometry;
use crate::residue::{Atom, ResName, Residue, Structure};

const BACKBONE_ATOMS: &[&str] = &["N", "CA", "C", "O", "OXT"];

/// Mutate residue at 1-based `position` (global across all chains, excluding
/// terminal caps ACE/NME) to `target` amino acid. Keeps backbone atoms
/// (N, CA, C, O) and rebuilds side chain from the target geometry.
pub fn mutate_residue(
    struc: &mut Structure,
    position: usize,
    target: ResName,
) -> Result<(), String> {
    mutate_residue_checked(struc, None, position, target)
}

/// Mutate with optional source-residue verification.
/// If `expected_from` is Some, the residue at `position` must match it.
pub fn mutate_residue_checked(
    struc: &mut Structure,
    expected_from: Option<ResName>,
    position: usize,
    target: ResName,
) -> Result<(), String> {
    let mut count = 0usize;
    for chain in &mut struc.chains {
        for res in &mut chain.residues {
            if res.name.is_cap() {
                continue;
            }
            count += 1;
            if count == position {
                if let Some(expected) = expected_from {
                    if res.name != expected {
                        return Err(format!(
                            "mutation at position {}: expected {} but found {}",
                            position,
                            expected.as_str(),
                            res.name.as_str(),
                        ));
                    }
                }
                rebuild_sidechain(res, target)?;
                return Ok(());
            }
        }
    }
    Err(format!(
        "position {} out of range (1..={})",
        position, count
    ))
}

/// Replace a residue's side chain with `target` geometry, keeping backbone.
fn rebuild_sidechain(res: &mut Residue, target: ResName) -> Result<(), String> {
    let bb: Vec<(String, Atom)> = res
        .atoms
        .iter()
        .filter(|a| BACKBONE_ATOMS.contains(&a.name.as_str()))
        .map(|a| (a.name.clone(), a.clone()))
        .collect();

    let geo = geometry::geometry(target);
    let seq_id = res.seq_id;
    let mut new_res = Residue::new(target, seq_id);

    for name in BACKBONE_ATOMS {
        if let Some((_, atom)) = bb.iter().find(|(n, _)| n == name) {
            new_res.add_atom(atom.clone());
        }
    }

    for sc in &geo.side_chain {
        let a = new_res
            .atom_coord(sc.parents.0)
            .ok_or_else(|| format!("missing parent atom '{}' during mutation", sc.parents.0))?;
        let b = new_res
            .atom_coord(sc.parents.1)
            .ok_or_else(|| format!("missing parent atom '{}' during mutation", sc.parents.1))?;
        let c = new_res
            .atom_coord(sc.parents.2)
            .ok_or_else(|| format!("missing parent atom '{}' during mutation", sc.parents.2))?;
        let pos = calculate_coordinates(a, b, c, sc.length, sc.angle, sc.dihedral);
        new_res.add_atom(Atom::new(sc.name, sc.element, pos));
    }

    *res = new_res;
    Ok(())
}

/// Parse a mutation spec like "A5G" (Ala at position 5 → Gly).
/// Returns (original_aa, position, target_aa).
pub fn parse_mutation_spec(spec: &str) -> Result<(ResName, usize, ResName), String> {
    let spec = spec.trim();
    if spec.len() < 3 {
        return Err(format!("mutation spec too short: '{spec}'"));
    }
    let chars: Vec<char> = spec.chars().collect();
    let from_aa = ResName::from_one_letter(chars[0])
        .ok_or_else(|| format!("unknown source amino acid '{}'", chars[0]))?;
    let to_aa = ResName::from_one_letter(*chars.last().unwrap())
        .ok_or_else(|| format!("unknown target amino acid '{}'", chars.last().unwrap()))?;
    let pos_str: String = chars[1..chars.len() - 1].iter().collect();
    let pos: usize = pos_str
        .parse()
        .map_err(|_| format!("invalid position in mutation spec: '{pos_str}'"))?;
    Ok((from_aa, pos, to_aa))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::make_extended_structure;

    #[test]
    fn test_mutate_ala_to_gly() {
        let mut struc = make_extended_structure("AAA").unwrap();
        mutate_residue(&mut struc, 2, ResName::GLY).unwrap();
        let r = &struc.chain_a().residues[1];
        assert_eq!(r.name, ResName::GLY);
        // GLY has no side chain: N, CA, C, O
        assert_eq!(r.atoms.len(), 4);
    }

    #[test]
    fn test_mutate_gly_to_trp() {
        let mut struc = make_extended_structure("GGG").unwrap();
        mutate_residue(&mut struc, 2, ResName::TRP).unwrap();
        let r = &struc.chain_a().residues[1];
        assert_eq!(r.name, ResName::TRP);
        // TRP: N,CA,C,O + 10 side chain = 14
        assert_eq!(r.atoms.len(), 14);
    }

    #[test]
    fn test_parse_mutation_spec() {
        let (from, pos, to) = parse_mutation_spec("A5G").unwrap();
        assert_eq!(from, ResName::ALA);
        assert_eq!(pos, 5);
        assert_eq!(to, ResName::GLY);
    }

    #[test]
    fn test_parse_mutation_spec_multidigit() {
        let (from, pos, to) = parse_mutation_spec("W123A").unwrap();
        assert_eq!(from, ResName::TRP);
        assert_eq!(pos, 123);
        assert_eq!(to, ResName::ALA);
    }

    #[test]
    fn test_mutate_out_of_range() {
        let mut struc = make_extended_structure("AA").unwrap();
        assert!(mutate_residue(&mut struc, 5, ResName::GLY).is_err());
    }

    #[test]
    fn test_mutate_chain_b() {
        use crate::builder::{ChainSpec, make_multi_chain_structure, parse_three_letter_sequence};
        let chains = vec![
            ChainSpec {
                id: 'A',
                residues: parse_three_letter_sequence("ALA-ALA").unwrap(),
                preset: None,
            },
            ChainSpec {
                id: 'B',
                residues: parse_three_letter_sequence("ALA-ALA").unwrap(),
                preset: None,
            },
        ];
        let mut struc = make_multi_chain_structure(&chains).unwrap();
        // Position 3 = first residue of chain B (global numbering: 1,2 in A; 3,4 in B)
        mutate_residue(&mut struc, 3, ResName::TRP).unwrap();
        assert_eq!(struc.chain_by_id('B').unwrap().residues[0].name, ResName::TRP);
        // Chain A untouched
        assert_eq!(struc.chain_a().residues[0].name, ResName::ALA);
    }

    #[test]
    fn test_mutate_positions_skip_terminal_caps() {
        let mut struc = make_extended_structure("AAA").unwrap();
        crate::caps::add_caps(&mut struc);
        // Position 1 should map to the first amino acid, not ACE.
        mutate_residue(&mut struc, 1, ResName::GLY).unwrap();
        let chain = struc.chain_a();
        assert_eq!(chain.residues[0].name, ResName::ACE);
        assert_eq!(chain.residues[1].name, ResName::GLY);
        assert_eq!(chain.residues.last().unwrap().name, ResName::NME);
    }
}
