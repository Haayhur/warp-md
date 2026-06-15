use std::collections::BTreeMap;

use crate::bonded_terms::{AngleTermGroup, BondTermGroup, BondedTermSet, DihedralTermGroup};
use crate::parameters::bonded_term_definitions;

use super::SourceBeadClassContext;

pub(super) fn template_bonded_term_set(
    bead_count: usize,
    connections: &[(usize, usize)],
    bead_contexts: &[SourceBeadClassContext],
) -> BondedTermSet {
    let mut bond_groups = BTreeMap::<String, Vec<[usize; 2]>>::new();
    for &(i, j) in connections {
        if i >= bead_count || j >= bead_count {
            continue;
        }
        let tuple = canonical_pair(i, j, bead_contexts);
        let label = term_label("bond", &tuple, bead_contexts);
        bond_groups.entry(label).or_default().push([i, j]);
    }

    let (angles, dihedrals) = bonded_term_definitions(bead_count, connections);
    let mut angle_groups = BTreeMap::<String, Vec<[usize; 3]>>::new();
    for (i, j, k) in angles {
        let tuple = canonical_sequence(&[i, j, k], bead_contexts);
        let label = term_label("angle", &tuple, bead_contexts);
        angle_groups.entry(label).or_default().push([i, j, k]);
    }

    let mut dihedral_groups = BTreeMap::<String, Vec<[usize; 4]>>::new();
    for (i, j, k, l) in dihedrals {
        let tuple = canonical_sequence(&[i, j, k, l], bead_contexts);
        let label = term_label("dihedral", &tuple, bead_contexts);
        dihedral_groups.entry(label).or_default().push([i, j, k, l]);
    }

    BondedTermSet {
        constraints: Vec::new(),
        bonds: bond_groups
            .into_iter()
            .map(|(label, members)| BondTermGroup {
                label: Some(label),
                members,
            })
            .collect(),
        angles: angle_groups
            .into_iter()
            .map(|(label, members)| AngleTermGroup {
                label: Some(label),
                members,
            })
            .collect(),
        dihedrals: dihedral_groups
            .into_iter()
            .map(|(label, members)| DihedralTermGroup {
                label: Some(label),
                members,
            })
            .collect(),
        virtual_sites: Vec::new(),
    }
}

fn canonical_pair(i: usize, j: usize, bead_contexts: &[SourceBeadClassContext]) -> Vec<usize> {
    let forward = vec![i, j];
    let reverse = vec![j, i];
    if descriptor_key(&reverse, bead_contexts) < descriptor_key(&forward, bead_contexts) {
        reverse
    } else {
        forward
    }
}

fn canonical_sequence(beads: &[usize], bead_contexts: &[SourceBeadClassContext]) -> Vec<usize> {
    let forward = beads.to_vec();
    let reverse = beads.iter().rev().copied().collect::<Vec<_>>();
    if descriptor_key(&reverse, bead_contexts) < descriptor_key(&forward, bead_contexts) {
        reverse
    } else {
        forward
    }
}

fn descriptor_key(beads: &[usize], bead_contexts: &[SourceBeadClassContext]) -> Vec<String> {
    beads
        .iter()
        .map(|idx| {
            bead_contexts
                .get(*idx)
                .map(|context| {
                    format!(
                        "{}:{}:{}",
                        context.role, context.residue_index, context.template_bead_name
                    )
                })
                .unwrap_or_else(|| format!("unknown:{idx}"))
        })
        .collect()
}

fn term_label(
    kind: &str,
    canonical_beads: &[usize],
    bead_contexts: &[SourceBeadClassContext],
) -> String {
    let role = role_signature(canonical_beads, bead_contexts);
    let anchor = canonical_beads
        .iter()
        .filter_map(|idx| bead_contexts.get(*idx).map(|context| context.residue_index))
        .min()
        .unwrap_or(0);
    let descriptors = canonical_beads
        .iter()
        .map(|idx| bead_descriptor(*idx, anchor, bead_contexts))
        .collect::<Vec<_>>()
        .join("__");
    format!("{kind}.{role}.{descriptors}")
}

fn role_signature(beads: &[usize], bead_contexts: &[SourceBeadClassContext]) -> String {
    let mut roles = Vec::<String>::new();
    for idx in beads {
        let role = bead_contexts
            .get(*idx)
            .map(|context| context.role.as_str())
            .unwrap_or("unknown");
        if !roles.iter().any(|item| item == role) {
            roles.push(role.to_string());
        }
    }
    roles.join("_")
}

fn bead_descriptor(
    bead_idx: usize,
    anchor_residue: usize,
    bead_contexts: &[SourceBeadClassContext],
) -> String {
    let Some(context) = bead_contexts.get(bead_idx) else {
        return format!("U0_B{bead_idx}");
    };
    let role_prefix = match context.role.as_str() {
        "head" => "H".to_string(),
        "middle" => {
            let offset = context.residue_index as isize - anchor_residue as isize;
            format!("M{offset}")
        }
        "tail" => "T".to_string(),
        "standalone" => "S".to_string(),
        other => parameter_safe_label(other),
    };
    format!(
        "{}_{}",
        role_prefix,
        parameter_safe_label(&context.template_bead_name)
    )
}

fn parameter_safe_label(label: &str) -> String {
    let mut out = String::with_capacity(label.len());
    let mut last_was_sep = false;
    for ch in label.chars() {
        let safe = ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-';
        if safe {
            out.push(ch);
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "bead".to_string()
    } else {
        trimmed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(residue_index: usize, role: &str, name: &str) -> SourceBeadClassContext {
        SourceBeadClassContext {
            residue_index,
            role: role.to_string(),
            template_bead_name: name.to_string(),
        }
    }

    #[test]
    fn template_bonded_terms_collapse_repeated_middle_members() {
        let contexts = vec![
            ctx(0, "head", "AR1"),
            ctx(0, "head", "SO2"),
            ctx(1, "middle", "AR1"),
            ctx(1, "middle", "SO2"),
            ctx(2, "middle", "AR1"),
            ctx(2, "middle", "SO2"),
            ctx(3, "tail", "AR1"),
            ctx(3, "tail", "SO2"),
        ];
        let connections = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)];

        let terms = template_bonded_term_set(contexts.len(), &connections, &contexts);
        let middle_bond = terms
            .bonds
            .iter()
            .find(|group| group.label.as_deref() == Some("bond.middle.M0_AR1__M0_SO2"))
            .expect("middle class");
        let middle_link = terms
            .bonds
            .iter()
            .find(|group| group.label.as_deref() == Some("bond.middle.M0_SO2__M1_AR1"))
            .expect("middle link class");

        assert_eq!(middle_bond.members, vec![[2, 3], [4, 5]]);
        assert_eq!(middle_link.members, vec![[3, 4]]);
        assert!(terms
            .bonds
            .iter()
            .any(|group| group.label.as_deref() == Some("bond.head_middle.H_SO2__M1_AR1")));
    }
}
