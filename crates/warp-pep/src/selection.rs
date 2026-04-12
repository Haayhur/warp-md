//! Atom selection language for filtering structures.
//!
//! Shared parser/AST lives in `traj-core`; this crate keeps only the
//! peptide-structure evaluator.

use crate::coord::Vec3;
use crate::residue::Structure;
use traj_core::{
    is_backbone_atom, is_protein_resname, is_sidechain_heavy_atom, parse_selection_expression,
    SelectionExpr, SelectionPredicate,
};

/// Reference to a single atom within a structure.
#[derive(Debug, Clone)]
pub struct AtomRef {
    pub chain_id: char,
    pub resname: String,
    pub resid: i32,
    pub atom_name: String,
    pub element: String,
    pub coord: Vec3,
    pub chain_idx: usize,
    pub res_idx: usize,
    pub atom_idx: usize,
}

/// Apply a selection expression to a structure, returning matching atom refs.
pub fn select(struc: &Structure, expr: &str) -> Result<Vec<AtomRef>, String> {
    let selection = parse_selection_expression(expr).map_err(|err| err.to_string())?;
    let mut result = Vec::new();
    for (chain_idx, chain) in struc.chains.iter().enumerate() {
        for (res_idx, residue) in chain.residues.iter().enumerate() {
            let resname = residue.amber_name().to_string();
            for (atom_idx, atom) in residue.atoms.iter().enumerate() {
                let atom_ref = AtomRef {
                    chain_id: chain.id,
                    resname: resname.clone(),
                    resid: residue.seq_id,
                    atom_name: atom.name.clone(),
                    element: atom.element.clone(),
                    coord: atom.coord,
                    chain_idx,
                    res_idx,
                    atom_idx,
                };
                if matches_expr(&selection, &atom_ref) {
                    result.push(atom_ref);
                }
            }
        }
    }
    Ok(result)
}

/// Extract just coordinates from a selection.
pub fn select_coords(struc: &Structure, expr: &str) -> Result<Vec<Vec3>, String> {
    Ok(select(struc, expr)?
        .into_iter()
        .map(|atom| atom.coord)
        .collect())
}

fn matches_expr(expr: &SelectionExpr, atom: &AtomRef) -> bool {
    match expr {
        SelectionExpr::All => true,
        SelectionExpr::Predicate(predicate) => matches_predicate(predicate, atom),
        SelectionExpr::Not(inner) => !matches_expr(inner, atom),
        SelectionExpr::And(left, right) => matches_expr(left, atom) && matches_expr(right, atom),
        SelectionExpr::Or(left, right) => matches_expr(left, atom) || matches_expr(right, atom),
    }
}

fn matches_predicate(predicate: &SelectionPredicate, atom: &AtomRef) -> bool {
    match predicate {
        SelectionPredicate::Name(names) => names
            .iter()
            .any(|name| name.eq_ignore_ascii_case(&atom.atom_name)),
        SelectionPredicate::Resname(names) => names
            .iter()
            .any(|resname| resname.eq_ignore_ascii_case(&atom.resname)),
        SelectionPredicate::Resid(ranges) => ranges.iter().any(|range| range.contains(atom.resid)),
        SelectionPredicate::Chain(chains) => {
            let chain = atom.chain_id.to_string();
            chains
                .iter()
                .any(|value| value.eq_ignore_ascii_case(&chain))
        }
        SelectionPredicate::Element(elements) => elements
            .iter()
            .any(|element| element.eq_ignore_ascii_case(&atom.element)),
        SelectionPredicate::Protein => is_protein_resname(&atom.resname),
        SelectionPredicate::Backbone => is_backbone_atom(&atom.atom_name),
        SelectionPredicate::SideChain => is_sidechain_heavy_atom(&atom.atom_name, &atom.element),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{
        make_extended_structure, make_multi_chain_structure, parse_three_letter_sequence, ChainSpec,
    };

    #[test]
    fn test_select_all() {
        let struc = make_extended_structure("AA").unwrap();
        let atoms = select(&struc, "all").unwrap();
        let total: usize = struc
            .chains
            .iter()
            .map(|chain| {
                chain
                    .residues
                    .iter()
                    .map(|residue| residue.atoms.len())
                    .sum::<usize>()
            })
            .sum();
        assert_eq!(atoms.len(), total);
    }

    #[test]
    fn test_select_name_ca() {
        let struc = make_extended_structure("AAA").unwrap();
        let atoms = select(&struc, "name CA").unwrap();
        assert_eq!(atoms.len(), 3);
        assert!(atoms.iter().all(|atom| atom.atom_name == "CA"));
    }

    #[test]
    fn test_select_backbone() {
        let struc = make_extended_structure("AG").unwrap();
        let atoms = select(&struc, "backbone").unwrap();
        assert_eq!(atoms.len(), 8);
    }

    #[test]
    fn test_select_resid_range() {
        let struc = make_extended_structure("AAAA").unwrap();
        let atoms = select(&struc, "resid 2-3 and name CA").unwrap();
        assert_eq!(atoms.len(), 2);
    }

    #[test]
    fn test_select_chain() {
        let chains = vec![
            ChainSpec {
                id: 'A',
                residues: parse_three_letter_sequence("ALA-ALA").unwrap(),
                preset: None,
            },
            ChainSpec {
                id: 'B',
                residues: parse_three_letter_sequence("GLY-GLY").unwrap(),
                preset: None,
            },
        ];
        let struc = make_multi_chain_structure(&chains).unwrap();
        let atoms = select(&struc, "chain B and name CA").unwrap();
        assert_eq!(atoms.len(), 2);
        assert!(atoms.iter().all(|atom| atom.chain_id == 'B'));
    }

    #[test]
    fn test_select_not() {
        let struc = make_extended_structure("AG").unwrap();
        let atoms = select(&struc, "not resname GLY").unwrap();
        assert!(atoms.iter().all(|atom| atom.resname != "GLY"));
    }

    #[test]
    fn test_select_or() {
        let struc = make_extended_structure("ACDE").unwrap();
        let atoms = select(&struc, "resname ALA or resname CYS").unwrap();
        assert!(atoms
            .iter()
            .all(|atom| atom.resname == "ALA" || atom.resname == "CYS"));
    }

    #[test]
    fn test_select_parens() {
        let struc = make_extended_structure("ACDE").unwrap();
        let atoms = select(&struc, "(resname ALA or resname CYS) and name CA").unwrap();
        assert_eq!(atoms.len(), 2);
    }

    #[test]
    fn test_select_protein() {
        let struc = make_extended_structure("AA").unwrap();
        let atoms = select(&struc, "protein").unwrap();
        let total: usize = struc
            .chains
            .iter()
            .map(|chain| {
                chain
                    .residues
                    .iter()
                    .map(|residue| residue.atoms.len())
                    .sum::<usize>()
            })
            .sum();
        assert_eq!(atoms.len(), total);
    }

    #[test]
    fn test_select_sidechain() {
        let struc = make_extended_structure("A").unwrap();
        let atoms = select(&struc, "sidechain").unwrap();
        assert_eq!(atoms.len(), 1);
        assert_eq!(atoms[0].atom_name, "CB");
    }

    #[test]
    fn test_select_empty_string() {
        let struc = make_extended_structure("A").unwrap();
        let atoms = select(&struc, "").unwrap();
        assert!(!atoms.is_empty());
    }

    #[test]
    fn test_select_error() {
        let struc = make_extended_structure("A").unwrap();
        assert!(select(&struc, "bogus").is_err());
    }

    #[test]
    fn test_select_coords() {
        let struc = make_extended_structure("AA").unwrap();
        let coords = select_coords(&struc, "name CA").unwrap();
        assert_eq!(coords.len(), 2);
    }

    #[test]
    fn test_select_multi_name() {
        let struc = make_extended_structure("A").unwrap();
        let atoms = select(&struc, "name CA CB").unwrap();
        assert_eq!(atoms.len(), 2);
    }

    #[test]
    fn test_select_element() {
        let struc = make_extended_structure("C").unwrap();
        let atoms = select(&struc, "element S").unwrap();
        assert_eq!(atoms.len(), 1);
        assert_eq!(atoms[0].atom_name, "SG");
    }
}
