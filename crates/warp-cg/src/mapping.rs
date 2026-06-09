use crate::molecule::{BondType, Molecule};
use petgraph::graph::NodeIndex;
use std::collections::{BTreeSet, HashSet, VecDeque};

pub struct MappingResult {
    pub bead_names: Vec<String>,
    pub atom_groups: Vec<Vec<usize>>,
    pub connections: Vec<(usize, usize)>,
    pub bead_features: Vec<Vec<String>>,
    pub bead_formal_charges: Vec<i32>,
}

#[derive(Clone, Copy, Debug)]
pub struct MappingOptions {
    pub target_bead_size: usize,
}

impl Default for MappingOptions {
    fn default() -> Self {
        Self {
            target_bead_size: 4,
        }
    }
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
    map_molecule_with_options(mol, &MappingOptions::default())
}

pub fn map_molecule_with_options(mol: &Molecule, options: &MappingOptions) -> MappingResult {
    let sections = find_sections(mol);
    let mut bead_names = Vec::new();
    let mut atom_groups = Vec::new();
    let mut bead_features = Vec::new();
    let mut bead_formal_charges = Vec::new();
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
                    bead_features.push(vec!["aromatic_ring".to_string()]);
                    bead_formal_charges.push(group_formal_charge(mol, &group));
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
                    bead_features.push(vec!["aliphatic_ring".to_string()]);
                    bead_formal_charges.push(group_formal_charge(mol, &group));
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
                for protected_group in protected_functional_groups(mol, &unmapped_heavy) {
                    if protected_group
                        .iter()
                        .all(|idx| unmapped_heavy.contains(idx))
                    {
                        let group = protected_group
                            .iter()
                            .map(|idx| idx.index())
                            .collect::<Vec<_>>();
                        unmapped_heavy.retain(|idx| !protected_group.contains(idx));
                        let bead_type = bead_type_for_group(mol, &group);
                        bead_names.push(bead_type);
                        bead_features.push(group_features(mol, &group));
                        bead_formal_charges.push(group_formal_charge(mol, &group));
                        atom_groups.push(group);
                    }
                }

                while !unmapped_heavy.is_empty() {
                    let take = if unmapped_heavy.len() <= 3 {
                        unmapped_heavy.len()
                    } else {
                        options.target_bead_size.max(1)
                    };
                    let group: Vec<_> = unmapped_heavy
                        .drain(0..take)
                        .map(|idx| idx.index())
                        .collect();

                    let bead_type = bead_type_for_group(mol, &group);

                    bead_names.push(bead_type);
                    bead_features.push(group_features(mol, &group));
                    bead_formal_charges.push(group_formal_charge(mol, &group));
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
        bead_features,
        bead_formal_charges,
    }
}

fn bead_type_for_group(mol: &Molecule, group: &[usize]) -> String {
    if group
        .iter()
        .any(|&i| mol.graph[NodeIndex::new(i)].atomic_number == 8)
    {
        if group.len() <= 3 {
            "SP1".to_string()
        } else {
            "P1".to_string()
        }
    } else if group
        .iter()
        .any(|&i| mol.graph[NodeIndex::new(i)].atomic_number == 16)
    {
        "SQ1".to_string()
    } else if group
        .iter()
        .any(|&i| mol.graph[NodeIndex::new(i)].atomic_number == 7)
    {
        "SN1".to_string()
    } else if group.len() <= 2 {
        "SC2".to_string()
    } else {
        "C1".to_string()
    }
}

fn protected_functional_groups(mol: &Molecule, heavy_atoms: &[NodeIndex]) -> Vec<Vec<NodeIndex>> {
    let heavy_set = heavy_atoms.iter().copied().collect::<HashSet<_>>();
    let mut groups = Vec::new();
    let mut used = HashSet::new();
    for &idx in heavy_atoms {
        if used.contains(&idx) {
            continue;
        }
        let atom = &mol.graph[idx];
        let oxygen_neighbors = mol
            .graph
            .neighbors(idx)
            .filter(|neighbor| heavy_set.contains(neighbor))
            .filter(|neighbor| mol.graph[*neighbor].atomic_number == 8)
            .collect::<Vec<_>>();
        let nitrogen_neighbors = mol
            .graph
            .neighbors(idx)
            .filter(|neighbor| heavy_set.contains(neighbor))
            .filter(|neighbor| mol.graph[*neighbor].atomic_number == 7)
            .collect::<Vec<_>>();
        let group = if atom.atomic_number == 6 && oxygen_neighbors.len() >= 2 {
            let mut group = vec![idx];
            group.extend(oxygen_neighbors.iter().copied().take(2));
            Some(group)
        } else if atom.atomic_number == 6
            && !oxygen_neighbors.is_empty()
            && !nitrogen_neighbors.is_empty()
        {
            Some(vec![idx, oxygen_neighbors[0], nitrogen_neighbors[0]])
        } else if matches!(atom.atomic_number, 15 | 16) && oxygen_neighbors.len() >= 2 {
            let mut group = vec![idx];
            group.extend(oxygen_neighbors.iter().copied().take(2));
            Some(group)
        } else {
            None
        };
        if let Some(mut group) = group {
            group.sort_by_key(|idx| idx.index());
            if group.iter().all(|idx| !used.contains(idx)) {
                for idx in &group {
                    used.insert(*idx);
                }
                groups.push(group);
            }
        }
    }
    groups.sort_by_key(|group| group.first().map(|idx| idx.index()).unwrap_or(usize::MAX));
    groups
}

fn group_formal_charge(mol: &Molecule, group: &[usize]) -> i32 {
    group
        .iter()
        .map(|&idx| mol.graph[NodeIndex::new(idx)].formal_charge as i32)
        .sum()
}

fn group_features(mol: &Molecule, group: &[usize]) -> Vec<String> {
    let group_set = group.iter().copied().collect::<HashSet<_>>();
    let mut features = BTreeSet::new();
    let mut has_oxygen = false;
    let mut has_nitrogen = false;
    let mut has_sulfur = false;
    let mut has_phosphorus = false;
    let mut has_halogen = false;
    let mut has_ring = false;

    for &idx in group {
        let atom_idx = NodeIndex::new(idx);
        let atom = &mol.graph[atom_idx];
        has_oxygen |= atom.atomic_number == 8;
        has_nitrogen |= atom.atomic_number == 7;
        has_sulfur |= atom.atomic_number == 16;
        has_phosphorus |= atom.atomic_number == 15;
        has_halogen |= matches!(atom.atomic_number, 9 | 17 | 35 | 53);
        has_ring |= atom.is_in_ring;
        if atom.formal_charge != 0 {
            features.insert("charged".to_string());
        }
        if atom.atomic_number == 6 {
            let oxygen_neighbors = mol
                .graph
                .neighbors(atom_idx)
                .filter(|neighbor| group_set.contains(&neighbor.index()))
                .filter(|neighbor| mol.graph[*neighbor].atomic_number == 8)
                .count();
            if oxygen_neighbors >= 2 {
                features.insert("carboxylate_or_carboxylic_acid".to_string());
            }
            if oxygen_neighbors >= 1
                && mol.graph.neighbors(atom_idx).any(|neighbor| {
                    group_set.contains(&neighbor.index()) && mol.graph[neighbor].atomic_number == 7
                })
            {
                features.insert("amide".to_string());
            }
        }
        if atom.atomic_number == 8 {
            let carbon_neighbors = mol
                .graph
                .neighbors(atom_idx)
                .filter(|neighbor| group_set.contains(&neighbor.index()))
                .filter(|neighbor| mol.graph[*neighbor].atomic_number == 6)
                .count();
            if carbon_neighbors >= 2 {
                features.insert("ether".to_string());
            }
        }
        if atom.atomic_number == 16 {
            let oxygen_neighbors = mol
                .graph
                .neighbors(atom_idx)
                .filter(|neighbor| group_set.contains(&neighbor.index()))
                .filter(|neighbor| mol.graph[*neighbor].atomic_number == 8)
                .count();
            if oxygen_neighbors >= 2 {
                features.insert("sulfone_or_sulfonate".to_string());
            }
        }
        if atom.atomic_number == 15 {
            let oxygen_neighbors = mol
                .graph
                .neighbors(atom_idx)
                .filter(|neighbor| group_set.contains(&neighbor.index()))
                .filter(|neighbor| mol.graph[*neighbor].atomic_number == 8)
                .count();
            if oxygen_neighbors >= 2 {
                features.insert("phosphate".to_string());
            }
        }
    }

    if has_ring {
        features.insert("ring".to_string());
    }
    if has_oxygen || has_nitrogen || has_sulfur || has_phosphorus {
        features.insert("polar".to_string());
    }
    if has_halogen {
        features.insert("halogenated".to_string());
    }
    if features.is_empty() {
        features.insert("hydrocarbon".to_string());
    }
    features.into_iter().collect()
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
