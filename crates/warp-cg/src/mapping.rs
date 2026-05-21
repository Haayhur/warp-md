use crate::molecule::{BondType, Molecule};
use petgraph::graph::NodeIndex;
use std::collections::{HashSet, VecDeque};

pub struct MappingResult {
    pub bead_names: Vec<String>,
    pub atom_groups: Vec<Vec<usize>>,
    pub connections: Vec<(usize, usize)>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SectionType {
    Benzene,
    NonBenzene6Ring,
    FiveRing,
    NonRing,
}

pub struct Section {
    pub section_type: SectionType,
    pub atom_indices: Vec<NodeIndex>,
}

pub fn map_molecule(mol: &Molecule) -> MappingResult {
    let sections = find_sections(mol);
    let mut bead_names = Vec::new();
    let mut atom_groups = Vec::new();
    let mut mapped_atoms: HashSet<NodeIndex> = HashSet::new();

    for section in &sections {
        let heavy_atoms: Vec<_> = section
            .atom_indices
            .iter()
            .filter(|&&idx| mol.graph[idx].atomic_number > 1)
            .cloned()
            .collect();

        if heavy_atoms.is_empty() {
            continue;
        }

        match section.section_type {
            SectionType::Benzene => {
                for i in 0..(heavy_atoms.len() / 2) {
                    let group = vec![heavy_atoms[i * 2].index(), heavy_atoms[i * 2 + 1].index()];
                    bead_names.push("TC5".to_string());
                    atom_groups.push(group);
                }
                for &idx in &heavy_atoms {
                    mapped_atoms.insert(idx);
                }
            }
            SectionType::NonBenzene6Ring => {
                for i in 0..(heavy_atoms.len() / 3) {
                    let group = vec![
                        heavy_atoms[i * 3].index(),
                        heavy_atoms[i * 3 + 1].index(),
                        heavy_atoms[i * 3 + 2].index(),
                    ];
                    bead_names.push("SC3".to_string());
                    atom_groups.push(group);
                }
                for &idx in &heavy_atoms {
                    mapped_atoms.insert(idx);
                }
            }
            _ => {
                let mut unmapped_heavy: Vec<_> = heavy_atoms
                    .into_iter()
                    .filter(|idx| !mapped_atoms.contains(idx))
                    .collect();

                while !unmapped_heavy.is_empty() {
                    let take = if unmapped_heavy.len() <= 3 {
                        unmapped_heavy.len()
                    } else {
                        4
                    };
                    let group: Vec<_> = unmapped_heavy
                        .drain(0..take)
                        .map(|idx| idx.index())
                        .collect();

                    let bead_type = if group
                        .iter()
                        .any(|&i| mol.graph[NodeIndex::new(i)].atomic_number == 8)
                    {
                        if group.len() <= 3 {
                            "SP1".to_string()
                        } else {
                            "P1".to_string()
                        }
                    } else if group.len() <= 2 {
                        "SC2".to_string()
                    } else {
                        "C1".to_string()
                    };

                    bead_names.push(bead_type);
                    atom_groups.push(group);
                }
            }
        }
    }

    // Determine connections between beads
    let mut connections = HashSet::new();
    for i in 0..atom_groups.len() {
        for j in (i + 1)..atom_groups.len() {
            let mut connected = false;
            for &atom_i in &atom_groups[i] {
                for &atom_j in &atom_groups[j] {
                    if mol
                        .graph
                        .contains_edge(NodeIndex::new(atom_i), NodeIndex::new(atom_j))
                    {
                        connected = true;
                        break;
                    }
                }
                if connected {
                    break;
                }
            }
            if connected {
                connections.insert((i, j));
            }
        }
    }

    let mut connections: Vec<_> = connections.into_iter().collect();
    connections.sort_unstable();

    MappingResult {
        bead_names,
        atom_groups,
        connections,
    }
}

fn find_sections(mol: &Molecule) -> Vec<Section> {
    let mut sections = Vec::new();
    let mut visited = HashSet::new();

    // 1. Find Ring components
    for idx in mol.graph.node_indices() {
        if mol.graph[idx].is_in_ring && !visited.contains(&idx) && mol.graph[idx].atomic_number > 1
        {
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(idx);
            visited.insert(idx);

            while let Some(curr) = queue.pop_front() {
                component.push(curr);
                for neighbor in mol.graph.neighbors(curr) {
                    if mol.graph[neighbor].is_in_ring
                        && !visited.contains(&neighbor)
                        && mol.graph[neighbor].atomic_number > 1
                    {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }

            if component.len() == 6 {
                let is_aromatic = component.iter().any(|&c| {
                    mol.graph
                        .edges(c)
                        .any(|e| *e.weight() == BondType::Aromatic)
                });
                sections.push(Section {
                    section_type: if is_aromatic {
                        SectionType::Benzene
                    } else {
                        SectionType::NonBenzene6Ring
                    },
                    atom_indices: component,
                });
            } else if component.len() == 5 {
                sections.push(Section {
                    section_type: SectionType::FiveRing,
                    atom_indices: component,
                });
            } else {
                sections.push(Section {
                    section_type: SectionType::NonRing,
                    atom_indices: component,
                });
            }
        }
    }

    // 2. Find remaining chain components
    for idx in mol.graph.node_indices() {
        if !visited.contains(&idx) && mol.graph[idx].atomic_number > 1 {
            let mut component = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(idx);
            visited.insert(idx);

            while let Some(curr) = queue.pop_front() {
                component.push(curr);
                for neighbor in mol.graph.neighbors(curr) {
                    if !visited.contains(&neighbor) && mol.graph[neighbor].atomic_number > 1 {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }

            sections.push(Section {
                section_type: SectionType::NonRing,
                atom_indices: component,
            });
        }
    }

    sections
}
