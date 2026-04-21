use std::sync::Arc;

use crate::error::TrajResult;
use crate::selection_expression::{
    is_backbone_atom, is_protein_resname, is_sidechain_heavy_atom, parse_selection_expression,
    SelectionExpr, SelectionPredicate,
};

#[derive(Debug, Clone)]
pub struct Selection {
    pub expr: String,
    pub indices: Arc<Vec<u32>>,
}

pub trait SelectionContext {
    fn n_atoms(&self) -> usize;
    fn atom_name_ids(&self) -> &[u32];
    fn residue_name_ids(&self) -> &[u32];
    fn residue_numbers(&self) -> &[i32];
    fn chain_ids(&self) -> &[u32];
    fn element_ids(&self) -> &[u32];
    fn intern_upper(&mut self, value: &str) -> u32;
    fn resolve(&self, id: u32) -> Option<&str>;
}

pub fn compile_selection(expr: &str, system: &mut dyn SelectionContext) -> TrajResult<Selection> {
    let ast = parse_selection_expression(expr)?;
    let mask = eval(&ast, system);
    let indices = mask
        .into_iter()
        .enumerate()
        .filter_map(|(idx, keep)| keep.then_some(idx as u32))
        .collect();
    Ok(Selection {
        expr: expr.to_string(),
        indices: Arc::new(indices),
    })
}

fn eval(expr: &SelectionExpr, system: &mut dyn SelectionContext) -> Vec<bool> {
    match expr {
        SelectionExpr::All => vec![true; system.n_atoms()],
        SelectionExpr::Predicate(predicate) => eval_predicate(predicate, system),
        SelectionExpr::And(left, right) => {
            let left_mask = eval(left, system);
            let right_mask = eval(right, system);
            left_mask
                .iter()
                .zip(right_mask.iter())
                .map(|(left, right)| *left && *right)
                .collect()
        }
        SelectionExpr::Or(left, right) => {
            let left_mask = eval(left, system);
            let right_mask = eval(right, system);
            left_mask
                .iter()
                .zip(right_mask.iter())
                .map(|(left, right)| *left || *right)
                .collect()
        }
        SelectionExpr::Not(inner) => eval(inner, system).into_iter().map(|keep| !keep).collect(),
    }
}

fn eval_predicate(predicate: &SelectionPredicate, system: &mut dyn SelectionContext) -> Vec<bool> {
    let n_atoms = system.n_atoms();
    let mut mask = vec![false; n_atoms];
    match predicate {
        SelectionPredicate::Name(names) => {
            let ids: Vec<u32> = names.iter().map(|name| system.intern_upper(name)).collect();
            for (idx, atom_name_id) in system.atom_name_ids().iter().enumerate() {
                if ids.contains(atom_name_id) {
                    mask[idx] = true;
                }
            }
        }
        SelectionPredicate::Resname(names) => {
            let ids: Vec<u32> = names
                .iter()
                .map(|resname| system.intern_upper(resname))
                .collect();
            for (idx, resname_id) in system.residue_name_ids().iter().enumerate() {
                if ids.contains(resname_id) {
                    mask[idx] = true;
                }
            }
        }
        SelectionPredicate::Resid(ranges) => {
            for (idx, resid) in system.residue_numbers().iter().enumerate() {
                if ranges.iter().any(|range| range.contains(*resid)) {
                    mask[idx] = true;
                }
            }
        }
        SelectionPredicate::Chain(chains) => {
            let ids: Vec<u32> = chains
                .iter()
                .map(|chain| system.intern_upper(chain))
                .collect();
            for (idx, chain_id) in system.chain_ids().iter().enumerate() {
                if ids.contains(chain_id) {
                    mask[idx] = true;
                }
            }
        }
        SelectionPredicate::Element(elements) => {
            let ids: Vec<u32> = elements
                .iter()
                .map(|element| system.intern_upper(element))
                .collect();
            for (idx, element_id) in system.element_ids().iter().enumerate() {
                if ids.contains(element_id) {
                    mask[idx] = true;
                }
            }
        }
        SelectionPredicate::Protein => {
            for idx in 0..n_atoms {
                let resname = system.resolve(system.residue_name_ids()[idx]).unwrap_or("");
                if is_protein_resname(resname) {
                    mask[idx] = true;
                }
            }
        }
        SelectionPredicate::Backbone => {
            for idx in 0..n_atoms {
                let resname = system.resolve(system.residue_name_ids()[idx]).unwrap_or("");
                let atom_name = system.resolve(system.atom_name_ids()[idx]).unwrap_or("");
                if is_protein_resname(resname) && is_backbone_atom(atom_name) {
                    mask[idx] = true;
                }
            }
        }
        SelectionPredicate::SideChain => {
            for idx in 0..n_atoms {
                let atom_name = system.resolve(system.atom_name_ids()[idx]).unwrap_or("");
                let element = system.resolve(system.element_ids()[idx]).unwrap_or("");
                if is_sidechain_heavy_atom(atom_name, element) {
                    mask[idx] = true;
                }
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interner::StringInterner;
    use crate::system::{AtomTable, System};

    fn build_system() -> System {
        let mut interner = StringInterner::new();
        let name_n = interner.intern_upper("N");
        let name_ca = interner.intern_upper("CA");
        let name_cb = interner.intern_upper("CB");
        let name_o = interner.intern_upper("O");
        let res_ala = interner.intern_upper("ALA");
        let res_hoh = interner.intern_upper("HOH");
        let chain_a = interner.intern_upper("A");
        let chain_b = interner.intern_upper("B");
        let element_n = interner.intern_upper("N");
        let element_c = interner.intern_upper("C");
        let element_o = interner.intern_upper("O");
        let atoms = AtomTable {
            name_id: vec![name_n, name_ca, name_cb, name_o],
            resname_id: vec![res_ala, res_ala, res_ala, res_hoh],
            resid: vec![1, 1, 1, 2],
            chain_id: vec![chain_a, chain_a, chain_a, chain_b],
            element_id: vec![element_n, element_c, element_c, element_o],
            mass: vec![1.0, 1.0, 1.0, 1.0],
        };
        System::with_atoms(atoms, interner, None)
    }

    #[test]
    fn selection_name() {
        let mut system = build_system();
        let sel = compile_selection("name CA", &mut system).unwrap();
        assert_eq!(sel.indices.as_slice(), &[1]);
    }

    #[test]
    fn selection_resid_range() {
        let mut system = build_system();
        let sel = compile_selection("resid 1:2", &mut system).unwrap();
        assert_eq!(sel.indices.len(), 4);
    }

    #[test]
    fn selection_protein_backbone() {
        let mut system = build_system();
        let sel = compile_selection("protein and backbone", &mut system).unwrap();
        assert_eq!(sel.indices.as_slice(), &[0, 1]);
    }

    #[test]
    fn selection_chain() {
        let mut system = build_system();
        let sel = compile_selection("chain A", &mut system).unwrap();
        assert_eq!(sel.indices.as_slice(), &[0, 1, 2]);
    }

    #[test]
    fn selection_multi_name() {
        let mut system = build_system();
        let sel = compile_selection("name CA CB", &mut system).unwrap();
        assert_eq!(sel.indices.as_slice(), &[1, 2]);
    }

    #[test]
    fn selection_element() {
        let mut system = build_system();
        let sel = compile_selection("element O", &mut system).unwrap();
        assert_eq!(sel.indices.as_slice(), &[3]);
    }

    #[test]
    fn selection_sidechain() {
        let mut system = build_system();
        let sel = compile_selection("sidechain", &mut system).unwrap();
        assert_eq!(sel.indices.as_slice(), &[2]);
    }
}
