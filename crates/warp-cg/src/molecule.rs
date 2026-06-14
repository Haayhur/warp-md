use anyhow::{anyhow, Result};
use petgraph::graph::{NodeIndex, UnGraph};
use sci_form::graph::{BondOrder, Molecule as SciMol};
use std::collections::{BTreeSet, HashSet};

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
        Self::from_elements_bonds_and_positions(elements, bonds, None)
    }

    pub fn from_elements_bonds_and_positions(
        elements: &[String],
        bonds: &[(usize, usize)],
        positions: Option<&[[f32; 3]]>,
    ) -> Self {
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
        mol.perceive_aromatic_bonds(positions);
        mol
    }

    fn perceive_rings(&mut self) {
        for idx in self
            .simple_cycles_with_size(5, 6)
            .into_iter()
            .flatten()
            .collect::<HashSet<_>>()
        {
            self.graph[idx].is_in_ring = true;
        }
    }

    fn simple_cycles_with_size(&self, min_size: usize, max_size: usize) -> Vec<Vec<NodeIndex>> {
        let mut cycles = Vec::new();
        let mut seen = BTreeSet::<Vec<usize>>::new();
        for start in self.graph.node_indices() {
            let mut path = vec![start];
            let mut visited = HashSet::from([start]);
            self.collect_cycles(
                start,
                start,
                min_size,
                max_size,
                &mut path,
                &mut visited,
                &mut seen,
                &mut cycles,
            );
        }
        cycles
    }

    fn perceive_aromatic_bonds(&mut self, positions: Option<&[[f32; 3]]>) {
        let Some(positions) = positions else {
            return;
        };
        if positions.len() != self.graph.node_count() {
            return;
        }
        for cycle in self.simple_cycles_with_size(6, 6) {
            if self.is_source_aromatic_six_ring(&cycle, positions) {
                for i in 0..cycle.len() {
                    for j in (i + 1)..cycle.len() {
                        if let Some(edge) = self.graph.find_edge(cycle[i], cycle[j]) {
                            self.graph[edge] = BondType::Aromatic;
                        }
                    }
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_cycles(
        &self,
        start: NodeIndex,
        current: NodeIndex,
        min_size: usize,
        max_size: usize,
        path: &mut Vec<NodeIndex>,
        visited: &mut HashSet<NodeIndex>,
        seen: &mut BTreeSet<Vec<usize>>,
        cycles: &mut Vec<Vec<NodeIndex>>,
    ) {
        if path.len() > max_size {
            return;
        }
        for neighbor in self.graph.neighbors(current) {
            if neighbor == start {
                if path.len() >= min_size {
                    let mut key = path.iter().map(|idx| idx.index()).collect::<Vec<_>>();
                    key.sort_unstable();
                    if seen.insert(key) {
                        cycles.push(path.clone());
                    }
                }
                continue;
            }
            if path.len() == max_size
                || neighbor.index() < start.index()
                || visited.contains(&neighbor)
                || self.graph[neighbor].atomic_number <= 1
            {
                continue;
            }
            visited.insert(neighbor);
            path.push(neighbor);
            self.collect_cycles(
                start, neighbor, min_size, max_size, path, visited, seen, cycles,
            );
            path.pop();
            visited.remove(&neighbor);
        }
    }

    fn is_source_aromatic_six_ring(&self, component: &[NodeIndex], positions: &[[f32; 3]]) -> bool {
        if component.len() != 6 {
            return false;
        }
        if !component
            .iter()
            .all(|idx| matches!(self.graph[*idx].atomic_number, 6 | 7))
        {
            return false;
        }
        let mut ring_bond_lengths = Vec::new();
        for i in 0..component.len() {
            for j in (i + 1)..component.len() {
                if self.graph.find_edge(component[i], component[j]).is_some() {
                    ring_bond_lengths.push(distance(
                        positions[component[i].index()],
                        positions[component[j].index()],
                    ));
                }
            }
        }
        if ring_bond_lengths.len() != 6 {
            return false;
        }
        let mean_bond =
            ring_bond_lengths.iter().copied().sum::<f32>() / ring_bond_lengths.len() as f32;
        let aromatic_scale_bonds =
            (1.20..=1.48).contains(&mean_bond) || (0.120..=0.148).contains(&mean_bond);
        if !aromatic_scale_bonds {
            return false;
        }
        is_planar(component, positions, mean_bond)
    }
}

fn distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn is_planar(component: &[NodeIndex], positions: &[[f32; 3]], scale: f32) -> bool {
    for a in 0..component.len() {
        for b in (a + 1)..component.len() {
            for c in (b + 1)..component.len() {
                let p0 = positions[component[a].index()];
                let p1 = positions[component[b].index()];
                let p2 = positions[component[c].index()];
                let normal = cross(sub(p1, p0), sub(p2, p0));
                let norm = dot(normal, normal).sqrt();
                if norm <= f32::EPSILON {
                    continue;
                }
                let tolerance = (scale.abs() * 0.12).max(0.03);
                return component.iter().all(|idx| {
                    let p = positions[idx.index()];
                    (dot(normal, sub(p, p0)) / norm).abs() <= tolerance
                });
            }
        }
    }
    false
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
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

#[cfg(test)]
mod tests {
    use super::{BondType, Molecule};
    use crate::mapping::map_molecule;

    fn six_carbon_ring() -> (Vec<String>, Vec<(usize, usize)>, Vec<[f32; 3]>) {
        (
            vec!["C".to_string(); 6],
            vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
            vec![
                [1.400, 0.000, 0.000],
                [0.700, 1.212, 0.000],
                [-0.700, 1.212, 0.000],
                [-1.400, 0.000, 0.000],
                [-0.700, -1.212, 0.000],
                [0.700, -1.212, 0.000],
            ],
        )
    }

    #[test]
    fn source_positions_perceive_aromatic_six_ring() {
        let (elements, bonds, positions) = six_carbon_ring();
        let mol = Molecule::from_elements_bonds_and_positions(&elements, &bonds, Some(&positions));

        assert_eq!(
            mol.graph
                .edge_weights()
                .filter(|bond| **bond == BondType::Aromatic)
                .count(),
            6
        );
        let mapping = map_molecule(&mol);
        assert_eq!(mapping.atom_groups.len(), 3);
        assert!(mapping
            .bead_features
            .iter()
            .all(|features| features == &["aromatic_ring".to_string()]));
    }

    #[test]
    fn source_without_positions_keeps_six_ring_non_aromatic() {
        let (elements, bonds, _) = six_carbon_ring();
        let mol = Molecule::from_elements_and_bonds(&elements, &bonds);

        assert_eq!(
            mol.graph
                .edge_weights()
                .filter(|bond| **bond == BondType::Aromatic)
                .count(),
            0
        );
        let mapping = map_molecule(&mol);
        assert_eq!(mapping.atom_groups.len(), 2);
        assert!(mapping
            .bead_features
            .iter()
            .all(|features| features == &["aliphatic_ring".to_string()]));
    }

    #[test]
    fn source_long_bond_six_ring_stays_non_aromatic() {
        let (elements, bonds, positions) = six_carbon_ring();
        let positions = positions
            .into_iter()
            .map(|[x, y, z]| [x * 1.1, y * 1.1, z])
            .collect::<Vec<_>>();
        let mol = Molecule::from_elements_bonds_and_positions(&elements, &bonds, Some(&positions));

        assert_eq!(
            mol.graph
                .edge_weights()
                .filter(|bond| **bond == BondType::Aromatic)
                .count(),
            0
        );
        let mapping = map_molecule(&mol);
        assert_eq!(mapping.atom_groups.len(), 2);
        assert!(mapping
            .bead_features
            .iter()
            .all(|features| features == &["aliphatic_ring".to_string()]));
    }
}
