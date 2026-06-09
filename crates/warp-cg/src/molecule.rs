use anyhow::{anyhow, Result};
use petgraph::graph::{NodeIndex, UnGraph};
use sci_form::graph::{BondOrder, Molecule as SciMol};
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct Atom {
    pub symbol: String,
    pub atomic_number: u8,
    pub formal_charge: i8,
    pub num_h: u32,
    pub is_in_ring: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BondType {
    Single,
    Double,
    Triple,
    Aromatic,
}

pub type MolGraph = UnGraph<Atom, BondType>;

pub struct Molecule {
    pub graph: MolGraph,
}

impl Molecule {
    pub fn from_smiles(smiles: &str) -> Result<Self> {
        let scimol =
            SciMol::from_smiles(smiles).map_err(|e| anyhow!("SMILES parse error: {}", e))?;

        let mut graph = MolGraph::new_undirected();
        let mut node_map = Vec::new();

        for node in scimol.graph.node_indices() {
            let sciatom = &scimol.graph[node];
            let atom = Atom {
                symbol: atomic_number_to_symbol(sciatom.element).to_string(),
                atomic_number: sciatom.element,
                formal_charge: sciatom.formal_charge,
                num_h: sciatom.explicit_h as u32,
                is_in_ring: false,
            };
            node_map.push(graph.add_node(atom));
        }

        for edge in scimol.graph.edge_indices() {
            let (u, v) = scimol.graph.edge_endpoints(edge).unwrap();
            let scibond = &scimol.graph[edge];
            let bt = match scibond.order {
                BondOrder::Single => BondType::Single,
                BondOrder::Double => BondType::Double,
                BondOrder::Triple => BondType::Triple,
                BondOrder::Aromatic => BondType::Aromatic,
                BondOrder::Unknown => BondType::Single,
            };
            graph.add_edge(node_map[u.index()], node_map[v.index()], bt);
        }

        let mut mol = Molecule { graph };
        mol.perceive_rings();
        Ok(mol)
    }

    pub fn from_elements_and_bonds(elements: &[String], bonds: &[(usize, usize)]) -> Self {
        let mut graph = MolGraph::new_undirected();
        let node_map = elements
            .iter()
            .map(|element| {
                graph.add_node(Atom {
                    symbol: element.clone(),
                    atomic_number: symbol_to_atomic_number(element),
                    formal_charge: 0,
                    num_h: 0,
                    is_in_ring: false,
                })
            })
            .collect::<Vec<_>>();
        for &(a, b) in bonds {
            if a < node_map.len() && b < node_map.len() {
                graph.add_edge(node_map[a], node_map[b], BondType::Single);
            }
        }
        let mut mol = Molecule { graph };
        mol.perceive_rings();
        mol
    }

    fn perceive_rings(&mut self) {
        let mut in_ring = HashSet::new();
        for idx in self.graph.node_indices() {
            if self.is_atom_in_ring(idx) {
                in_ring.insert(idx);
            }
        }
        for idx in in_ring {
            self.graph[idx].is_in_ring = true;
        }
    }

    fn is_atom_in_ring(&self, start_node: NodeIndex) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![(start_node, None)];

        while let Some((curr, parent)) = stack.pop() {
            if curr == start_node && parent.is_some() {
                return true;
            }
            if visited.contains(&curr) {
                continue;
            }
            visited.insert(curr);

            for neighbor in self.graph.neighbors(curr) {
                if Some(neighbor) != parent {
                    stack.push((neighbor, Some(curr)));
                }
            }
        }
        false
    }
}

pub(crate) fn atomic_number_to_symbol(z: u8) -> &'static str {
    match z {
        1 => "H",
        6 => "C",
        7 => "N",
        8 => "O",
        9 => "F",
        15 => "P",
        16 => "S",
        17 => "Cl",
        35 => "Br",
        53 => "I",
        _ => "X",
    }
}

pub(crate) fn symbol_to_atomic_number(symbol: &str) -> u8 {
    match symbol.trim().to_ascii_uppercase().as_str() {
        "H" => 1,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        "P" => 15,
        "S" => 16,
        "CL" => 17,
        "BR" => 35,
        "I" => 53,
        _ => 0,
    }
}
