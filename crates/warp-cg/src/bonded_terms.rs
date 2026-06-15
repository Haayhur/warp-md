use serde::{Deserialize, Serialize};
use warp_common::charge::{parse_gromacs_molecule_topology, GromacsMoleculeTopology};

use crate::parameters::bonded_term_definitions;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BondedTermSet {
    pub constraints: Vec<BondTermGroup>,
    pub bonds: Vec<BondTermGroup>,
    pub angles: Vec<AngleTermGroup>,
    pub dihedrals: Vec<DihedralTermGroup>,
    #[serde(default)]
    pub virtual_sites: Vec<VirtualSiteTerm>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondTermGroup {
    pub label: Option<String>,
    pub members: Vec<[usize; 2]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngleTermGroup {
    pub label: Option<String>,
    pub members: Vec<[usize; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DihedralTermGroup {
    pub label: Option<String>,
    pub members: Vec<[usize; 4]>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VirtualSiteTerm {
    pub site: usize,
    pub kind: String,
    pub function: usize,
    pub defining_beads: Vec<usize>,
    pub parameters: Vec<f64>,
}

impl BondedTermSet {
    pub fn from_connections(n_beads: usize, connections: &[(usize, usize)]) -> Self {
        let bonds = connections
            .iter()
            .map(|&(i, j)| BondTermGroup {
                label: Some(format!("bond_{i}_{j}")),
                members: vec![[i, j]],
            })
            .collect();
        let (angles, dihedrals) = bonded_term_definitions(n_beads, connections);
        Self {
            constraints: Vec::new(),
            bonds,
            angles: angles
                .into_iter()
                .map(|(i, j, k)| AngleTermGroup {
                    label: Some(format!("angle_{i}_{j}_{k}")),
                    members: vec![[i, j, k]],
                })
                .collect(),
            dihedrals: dihedrals
                .into_iter()
                .map(|(i, j, k, l)| DihedralTermGroup {
                    label: Some(format!("dihedral_{i}_{j}_{k}_{l}")),
                    members: vec![[i, j, k, l]],
                })
                .collect(),
            virtual_sites: Vec::new(),
        }
    }

    pub fn bonds_as_connections(&self) -> Vec<(usize, usize)> {
        self.bonds
            .iter()
            .flat_map(|group| group.members.iter())
            .map(|member| {
                let i = member[0];
                let j = member[1];
                if i <= j {
                    (i, j)
                } else {
                    (j, i)
                }
            })
            .collect()
    }

    pub fn from_gromacs_molecule_topology(topology: &GromacsMoleculeTopology) -> Self {
        Self {
            constraints: topology
                .constraints
                .iter()
                .map(|constraint| BondTermGroup {
                    label: Some(format!(
                        "constraint_{}_{}_f{}",
                        constraint.atom_i, constraint.atom_j, constraint.function
                    )),
                    members: vec![[
                        constraint.atom_i.saturating_sub(1),
                        constraint.atom_j.saturating_sub(1),
                    ]],
                })
                .collect(),
            bonds: topology
                .bonds
                .iter()
                .map(|bond| BondTermGroup {
                    label: Some(format!(
                        "bond_{}_{}_f{}",
                        bond.atom_i, bond.atom_j, bond.function
                    )),
                    members: vec![[bond.atom_i.saturating_sub(1), bond.atom_j.saturating_sub(1)]],
                })
                .collect(),
            angles: topology
                .angles
                .iter()
                .map(|angle| AngleTermGroup {
                    label: Some(format!(
                        "angle_{}_{}_{}_f{}",
                        angle.atom_i, angle.atom_j, angle.atom_k, angle.function
                    )),
                    members: vec![[
                        angle.atom_i.saturating_sub(1),
                        angle.atom_j.saturating_sub(1),
                        angle.atom_k.saturating_sub(1),
                    ]],
                })
                .collect(),
            dihedrals: topology
                .dihedrals
                .iter()
                .map(|dihedral| DihedralTermGroup {
                    label: Some(format!(
                        "dihedral_{}_{}_{}_{}_f{}",
                        dihedral.atom_i,
                        dihedral.atom_j,
                        dihedral.atom_k,
                        dihedral.atom_l,
                        dihedral.function
                    )),
                    members: vec![[
                        dihedral.atom_i.saturating_sub(1),
                        dihedral.atom_j.saturating_sub(1),
                        dihedral.atom_k.saturating_sub(1),
                        dihedral.atom_l.saturating_sub(1),
                    ]],
                })
                .collect(),
            virtual_sites: topology
                .virtual_sites
                .iter()
                .map(|site| VirtualSiteTerm {
                    site: site.site.saturating_sub(1),
                    kind: site.kind.clone(),
                    function: site.function,
                    defining_beads: site
                        .defining_atoms
                        .iter()
                        .map(|atom| atom.saturating_sub(1))
                        .collect(),
                    parameters: site
                        .parameters
                        .iter()
                        .map(|value| f64::from(*value))
                        .collect(),
                })
                .collect(),
        }
    }

    pub fn from_gromacs_topology_str(topology: &str, molecule_type: &str) -> Result<Self, String> {
        let parsed = parse_gromacs_molecule_topology(topology, molecule_type)?;
        parse_grouped_bonded_terms(topology, molecule_type).map(|terms| {
            let fallback = Self::from_gromacs_molecule_topology(&parsed);
            if terms.is_empty() {
                fallback
            } else {
                BondedTermSet {
                    virtual_sites: fallback.virtual_sites,
                    ..terms
                }
            }
        })
    }

    fn is_empty(&self) -> bool {
        self.constraints.is_empty()
            && self.bonds.is_empty()
            && self.angles.is_empty()
            && self.dihedrals.is_empty()
            && self.virtual_sites.is_empty()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GromacsSection {
    MoleculeType,
    Constraints,
    Bonds,
    Angles,
    Dihedrals,
    Other,
}

fn parse_grouped_bonded_terms(
    topology: &str,
    molecule_type: &str,
) -> Result<BondedTermSet, String> {
    let mut terms = BondedTermSet::default();
    let mut section = GromacsSection::Other;
    let mut in_molecule = false;
    let mut previous_boundary = false;
    let mut pending_group_label: Option<String> = None;

    for raw_line in topology.lines() {
        let (code, comment) = split_gromacs_comment(raw_line);
        let line = code.trim();
        let comment = comment.map(str::trim).filter(|item| !item.is_empty());

        if line.is_empty() {
            if let Some(label) = comment {
                pending_group_label = Some(label.to_string());
            }
            previous_boundary = true;
            continue;
        }

        if let Some(next_section) = parse_section_header(line) {
            section = next_section;
            if section == GromacsSection::MoleculeType {
                in_molecule = false;
            }
            previous_boundary = true;
            pending_group_label = None;
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.is_empty() {
            previous_boundary = true;
            continue;
        }

        if section == GromacsSection::MoleculeType {
            in_molecule = fields[0] == molecule_type;
            previous_boundary = false;
            pending_group_label = None;
            continue;
        }
        if !in_molecule {
            previous_boundary = false;
            continue;
        }

        match section {
            GromacsSection::Constraints => push_bond_group_member(
                &mut terms.constraints,
                fields.as_slice(),
                previous_boundary,
                pending_group_label.take(),
                "constraint_group",
            )?,
            GromacsSection::Bonds => push_bond_group_member(
                &mut terms.bonds,
                fields.as_slice(),
                previous_boundary,
                pending_group_label.take(),
                "bond_group",
            )?,
            GromacsSection::Angles => push_angle_group_member(
                &mut terms.angles,
                fields.as_slice(),
                previous_boundary,
                pending_group_label.take(),
            )?,
            GromacsSection::Dihedrals => push_dihedral_group_member(
                &mut terms.dihedrals,
                fields.as_slice(),
                previous_boundary,
                pending_group_label.take(),
            )?,
            _ => {
                pending_group_label = None;
            }
        }
        previous_boundary = false;
    }

    Ok(terms)
}

fn split_gromacs_comment(line: &str) -> (&str, Option<&str>) {
    line.split_once(';')
        .map(|(code, comment)| (code, Some(comment)))
        .unwrap_or((line, None))
}

fn parse_section_header(line: &str) -> Option<GromacsSection> {
    let trimmed = line.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return None;
    }
    let name = trimmed
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim()
        .to_ascii_lowercase();
    Some(match name.as_str() {
        "moleculetype" => GromacsSection::MoleculeType,
        "constraints" => GromacsSection::Constraints,
        "bonds" => GromacsSection::Bonds,
        "angles" => GromacsSection::Angles,
        "dihedrals" => GromacsSection::Dihedrals,
        _ => GromacsSection::Other,
    })
}

fn push_bond_group_member(
    groups: &mut Vec<BondTermGroup>,
    fields: &[&str],
    force_new_group: bool,
    pending_label: Option<String>,
    default_prefix: &str,
) -> Result<(), String> {
    let member = [
        parse_one_based_index(fields.first(), default_prefix)?,
        parse_one_based_index(fields.get(1), default_prefix)?,
    ];
    if groups.is_empty() || force_new_group {
        let next = groups.len() + 1;
        groups.push(BondTermGroup {
            label: pending_label.or_else(|| Some(format!("{default_prefix}_{next}"))),
            members: Vec::new(),
        });
    }
    groups
        .last_mut()
        .ok_or_else(|| format!("missing {default_prefix} group"))?
        .members
        .push(member);
    Ok(())
}

fn push_angle_group_member(
    groups: &mut Vec<AngleTermGroup>,
    fields: &[&str],
    force_new_group: bool,
    pending_label: Option<String>,
) -> Result<(), String> {
    let member = [
        parse_one_based_index(fields.first(), "angle_group")?,
        parse_one_based_index(fields.get(1), "angle_group")?,
        parse_one_based_index(fields.get(2), "angle_group")?,
    ];
    if groups.is_empty() || force_new_group {
        let next = groups.len() + 1;
        groups.push(AngleTermGroup {
            label: pending_label.or_else(|| Some(format!("angle_group_{next}"))),
            members: Vec::new(),
        });
    }
    groups
        .last_mut()
        .ok_or_else(|| "missing angle group".to_string())?
        .members
        .push(member);
    Ok(())
}

fn push_dihedral_group_member(
    groups: &mut Vec<DihedralTermGroup>,
    fields: &[&str],
    force_new_group: bool,
    pending_label: Option<String>,
) -> Result<(), String> {
    let member = [
        parse_one_based_index(fields.first(), "dihedral_group")?,
        parse_one_based_index(fields.get(1), "dihedral_group")?,
        parse_one_based_index(fields.get(2), "dihedral_group")?,
        parse_one_based_index(fields.get(3), "dihedral_group")?,
    ];
    if groups.is_empty() || force_new_group {
        let next = groups.len() + 1;
        groups.push(DihedralTermGroup {
            label: pending_label.or_else(|| Some(format!("dihedral_group_{next}"))),
            members: Vec::new(),
        });
    }
    groups
        .last_mut()
        .ok_or_else(|| "missing dihedral group".to_string())?
        .members
        .push(member);
    Ok(())
}

fn parse_one_based_index(field: Option<&&str>, context: &str) -> Result<usize, String> {
    let raw = field.ok_or_else(|| format!("{context} is missing bead index"))?;
    let index = raw
        .parse::<usize>()
        .map_err(|_| format!("{context} has invalid bead index '{raw}'"))?;
    index
        .checked_sub(1)
        .ok_or_else(|| format!("{context} bead indices must start from 1"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gromacs_topology_converts_to_zero_based_bonded_terms() {
        let topology = r#"
[ moleculetype ]
  MOL 1

[ atoms ]
  1 P1 1 MOL A 1 0
  2 P2 1 MOL B 2 0
  3 P3 1 MOL C 3 0
  4 P4 1 MOL D 4 0

[ constraints ]
  1 2 1 0.47

[ bonds ]
  2 3 1 0.48 1000

[ angles ]
  1 2 3 2 150.0 100.0

[ dihedrals ]
  1 2 3 4 1 180.0 5.0 2

[ virtual_sites2 ]
  4 1 2 1 0.5
"#;

        let terms = BondedTermSet::from_gromacs_topology_str(topology, "MOL").unwrap();

        assert_eq!(terms.constraints[0].members, vec![[0, 1]]);
        assert_eq!(terms.bonds[0].members, vec![[1, 2]]);
        assert_eq!(terms.angles[0].members, vec![[0, 1, 2]]);
        assert_eq!(terms.dihedrals[0].members, vec![[0, 1, 2, 3]]);
        assert_eq!(
            terms.virtual_sites,
            vec![VirtualSiteTerm {
                site: 3,
                kind: "2".to_string(),
                function: 1,
                defining_beads: vec![0, 1],
                parameters: vec![0.5],
            }]
        );
    }

    #[test]
    fn gromacs_topology_preserves_grouped_bonded_comment_grouping() {
        let topology = r#"
[ moleculetype ]
  MOL 1

[ atoms ]
  1 P1 1 MOL A 1 0
  2 P2 1 MOL B 2 0
  3 P3 1 MOL C 3 0
  4 P4 1 MOL D 4 0
  5 P5 1 MOL E 5 0

[ bonds ]
; bond group 1
  1 2 1 0.48 1000
  3 4 1 0.48 1000

; bond group 2
  4 5 1 0.50 1000

[ angles ]
; angle group 1
  1 2 3 2 150.0 100.0
  3 4 5 2 150.0 100.0

[ virtual_sitesn ]
  5 1 1 2 3
"#;

        let terms = BondedTermSet::from_gromacs_topology_str(topology, "MOL").unwrap();

        assert_eq!(terms.bonds.len(), 2);
        assert_eq!(terms.bonds[0].label.as_deref(), Some("bond group 1"));
        assert_eq!(terms.bonds[0].members, vec![[0, 1], [2, 3]]);
        assert_eq!(terms.bonds[1].members, vec![[3, 4]]);
        assert_eq!(terms.angles.len(), 1);
        assert_eq!(terms.angles[0].members, vec![[0, 1, 2], [2, 3, 4]]);
        assert_eq!(terms.virtual_sites.len(), 1);
        assert_eq!(terms.virtual_sites[0].site, 4);
        assert_eq!(terms.virtual_sites[0].defining_beads, vec![0, 1, 2]);
    }
}
