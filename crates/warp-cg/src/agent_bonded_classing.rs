use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use crate::bonded_terms::{AngleTermGroup, BondTermGroup, BondedTermSet, DihedralTermGroup};

use super::BondedClassingRequest;

#[derive(Clone, Debug)]
pub(super) struct BondedClassingResult {
    pub terms: BondedTermSet,
    pub summary: Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TermKind {
    Bond,
    Angle,
    Dihedral,
}

impl TermKind {
    fn name(self) -> &'static str {
        match self {
            Self::Bond => "bonds",
            Self::Angle => "angles",
            Self::Dihedral => "dihedrals",
        }
    }

    fn prefix(self) -> &'static str {
        match self {
            Self::Bond => "bond.",
            Self::Angle => "angle.",
            Self::Dihedral => "dihedral.",
        }
    }
}

#[derive(Default)]
struct AssignedMembers {
    bonds: BTreeSet<Vec<usize>>,
    angles: BTreeSet<Vec<usize>>,
    dihedrals: BTreeSet<Vec<usize>>,
}

impl AssignedMembers {
    fn get_mut(&mut self, kind: TermKind) -> &mut BTreeSet<Vec<usize>> {
        match kind {
            TermKind::Bond => &mut self.bonds,
            TermKind::Angle => &mut self.angles,
            TermKind::Dihedral => &mut self.dihedrals,
        }
    }

    fn get(&self, kind: TermKind) -> &BTreeSet<Vec<usize>> {
        match kind {
            TermKind::Bond => &self.bonds,
            TermKind::Angle => &self.angles,
            TermKind::Dihedral => &self.dihedrals,
        }
    }
}

pub(super) fn resolve_bonded_classing(
    request: Option<&BondedClassingRequest>,
    auto_terms: BondedTermSet,
    bead_count: usize,
) -> Result<BondedClassingResult> {
    let mode = request.map(|item| item.mode.as_str()).unwrap_or("auto");
    let result = match mode {
        "auto" => auto_terms.clone(),
        "explicit" => explicit_terms(request.expect("explicit request"), &auto_terms, bead_count)?,
        "patch" => patched_terms(
            request.expect("patch request"),
            auto_terms.clone(),
            bead_count,
        )?,
        other => {
            return Err(anyhow!(
                "mapping.bonded_classing.mode must be auto, explicit, or patch, got '{other}'"
            ))
        }
    };
    let summary = classing_summary(
        mode,
        request,
        &auto_terms,
        &result,
        unclassified_counts(&auto_terms, &result),
    );
    Ok(BondedClassingResult {
        terms: result,
        summary,
    })
}

fn explicit_terms(
    request: &BondedClassingRequest,
    auto_terms: &BondedTermSet,
    bead_count: usize,
) -> Result<BondedTermSet> {
    let allow_duplicate = duplicate_policy(request)? == "allow";
    let mut assigned = AssignedMembers::default();
    let mut result = BondedTermSet::default();

    for class in &request.bonds {
        result.bonds.push(BondTermGroup {
            label: Some(validate_label(
                &class.label,
                "mapping.bonded_classing.bonds.label",
            )?),
            members: validate_explicit_members(
                TermKind::Bond,
                &class.members,
                auto_terms,
                bead_count,
                &mut assigned,
                allow_duplicate,
            )?,
        });
    }
    for class in &request.angles {
        result.angles.push(AngleTermGroup {
            label: Some(validate_label(
                &class.label,
                "mapping.bonded_classing.angles.label",
            )?),
            members: validate_explicit_members(
                TermKind::Angle,
                &class.members,
                auto_terms,
                bead_count,
                &mut assigned,
                allow_duplicate,
            )?,
        });
    }
    for class in &request.dihedrals {
        result.dihedrals.push(DihedralTermGroup {
            label: Some(validate_label(
                &class.label,
                "mapping.bonded_classing.dihedrals.label",
            )?),
            members: validate_explicit_members(
                TermKind::Dihedral,
                &class.members,
                auto_terms,
                bead_count,
                &mut assigned,
                allow_duplicate,
            )?,
        });
    }

    append_unclassified(
        &mut result,
        auto_terms,
        &assigned,
        unclassified_policy(request)?,
    )?;
    Ok(result)
}

fn validate_explicit_members<const N: usize>(
    kind: TermKind,
    members: &[[usize; N]],
    auto_terms: &BondedTermSet,
    bead_count: usize,
    assigned: &mut AssignedMembers,
    allow_duplicate: bool,
) -> Result<Vec<[usize; N]>> {
    if members.is_empty() {
        return Err(anyhow!(
            "mapping.bonded_classing.{} class has no members",
            kind.name()
        ));
    }
    let known = known_members(auto_terms, kind);
    let mut out = Vec::with_capacity(members.len());
    for member in members {
        validate_member_range(member, bead_count, kind)?;
        let key = canonical_key(member);
        if !known.contains(&key) {
            return Err(anyhow!(
                "mapping.bonded_classing.{} member {:?} is not present in the generated CG bonded graph",
                kind.name(),
                member
            ));
        }
        if !allow_duplicate && !assigned.get_mut(kind).insert(key) {
            return Err(anyhow!(
                "mapping.bonded_classing.{} member {:?} is assigned to more than one class",
                kind.name(),
                member
            ));
        }
        out.push(*member);
    }
    Ok(out)
}

fn append_unclassified(
    result: &mut BondedTermSet,
    auto_terms: &BondedTermSet,
    assigned: &AssignedMembers,
    policy: &str,
) -> Result<()> {
    append_unclassified_bonds(result, auto_terms, assigned, policy)?;
    append_unclassified_angles(result, auto_terms, assigned, policy)?;
    append_unclassified_dihedrals(result, auto_terms, assigned, policy)?;
    Ok(())
}

fn append_unclassified_bonds(
    result: &mut BondedTermSet,
    auto_terms: &BondedTermSet,
    assigned: &AssignedMembers,
    policy: &str,
) -> Result<()> {
    for group in &auto_terms.bonds {
        let missing = group
            .members
            .iter()
            .copied()
            .filter(|member| {
                !assigned
                    .get(TermKind::Bond)
                    .contains(&canonical_key(member))
            })
            .collect::<Vec<_>>();
        handle_unclassified_bonds(result, group.label.as_deref(), missing, policy)?;
    }
    Ok(())
}

fn append_unclassified_angles(
    result: &mut BondedTermSet,
    auto_terms: &BondedTermSet,
    assigned: &AssignedMembers,
    policy: &str,
) -> Result<()> {
    for group in &auto_terms.angles {
        let missing = group
            .members
            .iter()
            .copied()
            .filter(|member| {
                !assigned
                    .get(TermKind::Angle)
                    .contains(&canonical_key(member))
            })
            .collect::<Vec<_>>();
        handle_unclassified_angles(result, group.label.as_deref(), missing, policy)?;
    }
    Ok(())
}

fn append_unclassified_dihedrals(
    result: &mut BondedTermSet,
    auto_terms: &BondedTermSet,
    assigned: &AssignedMembers,
    policy: &str,
) -> Result<()> {
    for group in &auto_terms.dihedrals {
        let missing = group
            .members
            .iter()
            .copied()
            .filter(|member| {
                !assigned
                    .get(TermKind::Dihedral)
                    .contains(&canonical_key(member))
            })
            .collect::<Vec<_>>();
        handle_unclassified_dihedrals(result, group.label.as_deref(), missing, policy)?;
    }
    Ok(())
}

fn handle_unclassified_bonds(
    result: &mut BondedTermSet,
    label: Option<&str>,
    missing: Vec<[usize; 2]>,
    policy: &str,
) -> Result<()> {
    if missing.is_empty() || policy == "drop" {
        return Ok(());
    }
    match policy {
        "auto" => result.bonds.push(BondTermGroup {
            label: label.map(str::to_string),
            members: missing,
        }),
        "singleton" => {
            for member in missing {
                result.bonds.push(BondTermGroup {
                    label: Some(format!("bond.singleton.{}_{}", member[0], member[1])),
                    members: vec![member],
                });
            }
        }
        "error" => {
            return Err(anyhow!(
            "mapping.bonded_classing.on_unclassified=error but generated bonds remain unclassified"
        ))
        }
        other => return Err(invalid_unclassified_policy(other)),
    }
    Ok(())
}

fn handle_unclassified_angles(
    result: &mut BondedTermSet,
    label: Option<&str>,
    missing: Vec<[usize; 3]>,
    policy: &str,
) -> Result<()> {
    if missing.is_empty() || policy == "drop" {
        return Ok(());
    }
    match policy {
        "auto" => result.angles.push(AngleTermGroup {
            label: label.map(str::to_string),
            members: missing,
        }),
        "singleton" => {
            for member in missing {
                result.angles.push(AngleTermGroup {
                    label: Some(format!(
                        "angle.singleton.{}_{}_{}",
                        member[0], member[1], member[2]
                    )),
                    members: vec![member],
                });
            }
        }
        "error" => {
            return Err(anyhow!(
                "mapping.bonded_classing.on_unclassified=error but generated angles remain unclassified"
            ))
        }
        other => return Err(invalid_unclassified_policy(other)),
    }
    Ok(())
}

fn handle_unclassified_dihedrals(
    result: &mut BondedTermSet,
    label: Option<&str>,
    missing: Vec<[usize; 4]>,
    policy: &str,
) -> Result<()> {
    if missing.is_empty() || policy == "drop" {
        return Ok(());
    }
    match policy {
        "auto" => result.dihedrals.push(DihedralTermGroup {
            label: label.map(str::to_string),
            members: missing,
        }),
        "singleton" => {
            for member in missing {
                result.dihedrals.push(DihedralTermGroup {
                    label: Some(format!(
                        "dihedral.singleton.{}_{}_{}_{}",
                        member[0], member[1], member[2], member[3]
                    )),
                    members: vec![member],
                });
            }
        }
        "error" => {
            return Err(anyhow!(
                "mapping.bonded_classing.on_unclassified=error but generated dihedrals remain unclassified"
            ))
        }
        other => return Err(invalid_unclassified_policy(other)),
    }
    Ok(())
}

fn patched_terms(
    request: &BondedClassingRequest,
    mut terms: BondedTermSet,
    bead_count: usize,
) -> Result<BondedTermSet> {
    let base = request.base.as_deref().unwrap_or("auto");
    if base != "auto" {
        return Err(anyhow!("mapping.bonded_classing.patch base must be auto"));
    }
    validate_term_set_members(&terms, bead_count)?;
    for merge in &request.merge {
        let label = validate_label(&merge.label, "mapping.bonded_classing.merge.label")?;
        if merge.from.is_empty() {
            return Err(anyhow!(
                "mapping.bonded_classing.merge.from must not be empty"
            ));
        }
        let kind = infer_label_kind(&merge.from[0])?;
        if merge
            .from
            .iter()
            .any(|label| infer_label_kind(label).ok() != Some(kind))
        {
            return Err(anyhow!(
                "mapping.bonded_classing.merge.from must reference one term kind"
            ));
        }
        merge_labels(&mut terms, kind, &label, &merge.from)?;
    }
    for split in &request.split {
        let kind = infer_label_kind(&split.from)?;
        split_label(&mut terms, kind, split, bead_count)?;
    }
    for rename in &request.rename {
        let label = validate_label(&rename.to, "mapping.bonded_classing.rename.to")?;
        let kind = infer_label_kind(&rename.from)?;
        rename_label(&mut terms, kind, &rename.from, &label)?;
    }
    validate_no_duplicate_members(&terms)?;
    Ok(terms)
}

fn merge_labels(
    terms: &mut BondedTermSet,
    kind: TermKind,
    label: &str,
    from: &[String],
) -> Result<()> {
    match kind {
        TermKind::Bond => {
            let mut members = Vec::new();
            let mut found = BTreeSet::new();
            terms.bonds.retain(|group| {
                if group
                    .label
                    .as_deref()
                    .is_some_and(|item| from.iter().any(|wanted| wanted == item))
                {
                    if let Some(label) = group.label.as_deref() {
                        found.insert(label.to_string());
                    }
                    members.extend(group.members.iter().copied());
                    false
                } else {
                    true
                }
            });
            ensure_all_labels_found(kind, from, &found)?;
            terms.bonds.push(BondTermGroup {
                label: Some(label.to_string()),
                members,
            });
        }
        TermKind::Angle => {
            let mut members = Vec::new();
            let mut found = BTreeSet::new();
            terms.angles.retain(|group| {
                if group
                    .label
                    .as_deref()
                    .is_some_and(|item| from.iter().any(|wanted| wanted == item))
                {
                    if let Some(label) = group.label.as_deref() {
                        found.insert(label.to_string());
                    }
                    members.extend(group.members.iter().copied());
                    false
                } else {
                    true
                }
            });
            ensure_all_labels_found(kind, from, &found)?;
            terms.angles.push(AngleTermGroup {
                label: Some(label.to_string()),
                members,
            });
        }
        TermKind::Dihedral => {
            let mut members = Vec::new();
            let mut found = BTreeSet::new();
            terms.dihedrals.retain(|group| {
                if group
                    .label
                    .as_deref()
                    .is_some_and(|item| from.iter().any(|wanted| wanted == item))
                {
                    if let Some(label) = group.label.as_deref() {
                        found.insert(label.to_string());
                    }
                    members.extend(group.members.iter().copied());
                    false
                } else {
                    true
                }
            });
            ensure_all_labels_found(kind, from, &found)?;
            terms.dihedrals.push(DihedralTermGroup {
                label: Some(label.to_string()),
                members,
            });
        }
    }
    Ok(())
}

fn ensure_all_labels_found(
    kind: TermKind,
    labels: &[String],
    found: &BTreeSet<String>,
) -> Result<()> {
    let missing = labels
        .iter()
        .filter(|label| !found.contains(label.as_str()))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(anyhow!(
            "mapping.bonded_classing.patch requested {} class label does not exist: {}",
            kind.name(),
            missing
                .into_iter()
                .map(|label| label.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    Ok(())
}

fn split_label(
    terms: &mut BondedTermSet,
    kind: TermKind,
    split: &super::BondedClassSplit,
    bead_count: usize,
) -> Result<()> {
    match kind {
        TermKind::Bond => {
            let idx = find_group_index(&terms.bonds, &split.from, kind)?;
            let original = terms.bonds.remove(idx);
            let original_keys = original
                .members
                .iter()
                .map(canonical_key)
                .collect::<BTreeSet<_>>();
            let mut split_keys = BTreeSet::new();
            for target in &split.into {
                let label = validate_label(&target.label, "mapping.bonded_classing.split.label")?;
                let members =
                    split_members::<2>(target, kind, bead_count, &original_keys, &mut split_keys)?;
                terms.bonds.push(BondTermGroup {
                    label: Some(label),
                    members,
                });
            }
            let remaining = original
                .members
                .into_iter()
                .filter(|member| !split_keys.contains(&canonical_key(member)))
                .collect::<Vec<_>>();
            if !remaining.is_empty() {
                terms.bonds.push(BondTermGroup {
                    label: original.label,
                    members: remaining,
                });
            }
        }
        TermKind::Angle => {
            let idx = find_group_index(&terms.angles, &split.from, kind)?;
            let original = terms.angles.remove(idx);
            let original_keys = original
                .members
                .iter()
                .map(canonical_key)
                .collect::<BTreeSet<_>>();
            let mut split_keys = BTreeSet::new();
            for target in &split.into {
                let label = validate_label(&target.label, "mapping.bonded_classing.split.label")?;
                let members =
                    split_members::<3>(target, kind, bead_count, &original_keys, &mut split_keys)?;
                terms.angles.push(AngleTermGroup {
                    label: Some(label),
                    members,
                });
            }
            let remaining = original
                .members
                .into_iter()
                .filter(|member| !split_keys.contains(&canonical_key(member)))
                .collect::<Vec<_>>();
            if !remaining.is_empty() {
                terms.angles.push(AngleTermGroup {
                    label: original.label,
                    members: remaining,
                });
            }
        }
        TermKind::Dihedral => {
            let idx = find_group_index(&terms.dihedrals, &split.from, kind)?;
            let original = terms.dihedrals.remove(idx);
            let original_keys = original
                .members
                .iter()
                .map(canonical_key)
                .collect::<BTreeSet<_>>();
            let mut split_keys = BTreeSet::new();
            for target in &split.into {
                let label = validate_label(&target.label, "mapping.bonded_classing.split.label")?;
                let members =
                    split_members::<4>(target, kind, bead_count, &original_keys, &mut split_keys)?;
                terms.dihedrals.push(DihedralTermGroup {
                    label: Some(label),
                    members,
                });
            }
            let remaining = original
                .members
                .into_iter()
                .filter(|member| !split_keys.contains(&canonical_key(member)))
                .collect::<Vec<_>>();
            if !remaining.is_empty() {
                terms.dihedrals.push(DihedralTermGroup {
                    label: original.label,
                    members: remaining,
                });
            }
        }
    }
    Ok(())
}

fn split_members<const N: usize>(
    target: &super::BondedClassSplitTarget,
    kind: TermKind,
    bead_count: usize,
    original_keys: &BTreeSet<Vec<usize>>,
    split_keys: &mut BTreeSet<Vec<usize>>,
) -> Result<Vec<[usize; N]>> {
    if target.members.is_empty() {
        return Err(anyhow!(
            "mapping.bonded_classing.split target class has no members"
        ));
    }
    let mut out = Vec::new();
    for member in &target.members {
        let tuple: [usize; N] = member.as_slice().try_into().map_err(|_| {
            anyhow!(
                "mapping.bonded_classing.split member {:?} has wrong tuple length for {}",
                member,
                kind.name()
            )
        })?;
        validate_member_range(&tuple, bead_count, kind)?;
        let key = canonical_key(&tuple);
        if !original_keys.contains(&key) {
            return Err(anyhow!(
                "mapping.bonded_classing.split member {:?} is not in source class",
                member
            ));
        }
        if !split_keys.insert(key) {
            return Err(anyhow!(
                "mapping.bonded_classing.split member {:?} appears in more than one split target",
                member
            ));
        }
        out.push(tuple);
    }
    Ok(out)
}

fn rename_label(terms: &mut BondedTermSet, kind: TermKind, from: &str, to: &str) -> Result<()> {
    let mut found = false;
    match kind {
        TermKind::Bond => {
            for group in &mut terms.bonds {
                if group.label.as_deref() == Some(from) {
                    group.label = Some(to.to_string());
                    found = true;
                }
            }
        }
        TermKind::Angle => {
            for group in &mut terms.angles {
                if group.label.as_deref() == Some(from) {
                    group.label = Some(to.to_string());
                    found = true;
                }
            }
        }
        TermKind::Dihedral => {
            for group in &mut terms.dihedrals {
                if group.label.as_deref() == Some(from) {
                    group.label = Some(to.to_string());
                    found = true;
                }
            }
        }
    }
    if !found {
        return Err(anyhow!(
            "mapping.bonded_classing.patch requested {} class label does not exist: {from}",
            kind.name()
        ));
    }
    Ok(())
}

trait LabeledGroup {
    fn label(&self) -> Option<&str>;
}

impl LabeledGroup for BondTermGroup {
    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

impl LabeledGroup for AngleTermGroup {
    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

impl LabeledGroup for DihedralTermGroup {
    fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }
}

fn find_group_index<T: LabeledGroup>(groups: &[T], label: &str, kind: TermKind) -> Result<usize> {
    groups
        .iter()
        .position(|group| group.label() == Some(label))
        .ok_or_else(|| {
            anyhow!(
                "mapping.bonded_classing.patch requested {} class label does not exist: {label}",
                kind.name()
            )
        })
}

fn validate_term_set_members(terms: &BondedTermSet, bead_count: usize) -> Result<()> {
    for group in &terms.bonds {
        if group.members.is_empty() {
            return Err(anyhow!("generated bonded class has no bond members"));
        }
        for member in &group.members {
            validate_member_range(member, bead_count, TermKind::Bond)?;
        }
    }
    for group in &terms.angles {
        if group.members.is_empty() {
            return Err(anyhow!("generated bonded class has no angle members"));
        }
        for member in &group.members {
            validate_member_range(member, bead_count, TermKind::Angle)?;
        }
    }
    for group in &terms.dihedrals {
        if group.members.is_empty() {
            return Err(anyhow!("generated bonded class has no dihedral members"));
        }
        for member in &group.members {
            validate_member_range(member, bead_count, TermKind::Dihedral)?;
        }
    }
    Ok(())
}

fn validate_no_duplicate_members(terms: &BondedTermSet) -> Result<()> {
    let mut assigned = AssignedMembers::default();
    for group in &terms.bonds {
        for member in &group.members {
            let key = canonical_key(member);
            if !assigned.get_mut(TermKind::Bond).insert(key) {
                return Err(anyhow!(
                    "mapping.bonded_classing produced duplicate bond member"
                ));
            }
        }
    }
    for group in &terms.angles {
        for member in &group.members {
            let key = canonical_key(member);
            if !assigned.get_mut(TermKind::Angle).insert(key) {
                return Err(anyhow!(
                    "mapping.bonded_classing produced duplicate angle member"
                ));
            }
        }
    }
    for group in &terms.dihedrals {
        for member in &group.members {
            let key = canonical_key(member);
            if !assigned.get_mut(TermKind::Dihedral).insert(key) {
                return Err(anyhow!(
                    "mapping.bonded_classing produced duplicate dihedral member"
                ));
            }
        }
    }
    Ok(())
}

fn validate_member_range<const N: usize>(
    member: &[usize; N],
    bead_count: usize,
    kind: TermKind,
) -> Result<()> {
    for index in member {
        if *index >= bead_count {
            return Err(anyhow!(
                "mapping.bonded_classing.{} member {:?} has bead index {index} outside 0..{}",
                kind.name(),
                member,
                bead_count
            ));
        }
    }
    Ok(())
}

fn validate_label(label: &str, field: &str) -> Result<String> {
    let label = label.trim();
    if label.is_empty() {
        return Err(anyhow!("{field} must not be empty"));
    }
    Ok(label.to_string())
}

fn duplicate_policy(request: &BondedClassingRequest) -> Result<&str> {
    match request.on_duplicate_member.as_deref().unwrap_or("error") {
        "error" => Ok("error"),
        "allow" => Ok("allow"),
        other => Err(anyhow!(
            "mapping.bonded_classing.on_duplicate_member must be error or allow, got '{other}'"
        )),
    }
}

fn unclassified_policy(request: &BondedClassingRequest) -> Result<&str> {
    match request.on_unclassified.as_deref().unwrap_or("auto") {
        "auto" => Ok("auto"),
        "singleton" => Ok("singleton"),
        "drop" => Ok("drop"),
        "error" => Ok("error"),
        other => Err(invalid_unclassified_policy(other)),
    }
}

fn invalid_unclassified_policy(policy: &str) -> anyhow::Error {
    anyhow!(
        "mapping.bonded_classing.on_unclassified must be auto, singleton, drop, or error, got '{policy}'"
    )
}

fn known_members(terms: &BondedTermSet, kind: TermKind) -> BTreeSet<Vec<usize>> {
    match kind {
        TermKind::Bond => terms
            .bonds
            .iter()
            .flat_map(|group| group.members.iter().map(canonical_key))
            .collect(),
        TermKind::Angle => terms
            .angles
            .iter()
            .flat_map(|group| group.members.iter().map(canonical_key))
            .collect(),
        TermKind::Dihedral => terms
            .dihedrals
            .iter()
            .flat_map(|group| group.members.iter().map(canonical_key))
            .collect(),
    }
}

fn infer_label_kind(label: &str) -> Result<TermKind> {
    if label.starts_with(TermKind::Bond.prefix()) {
        Ok(TermKind::Bond)
    } else if label.starts_with(TermKind::Angle.prefix()) {
        Ok(TermKind::Angle)
    } else if label.starts_with(TermKind::Dihedral.prefix()) {
        Ok(TermKind::Dihedral)
    } else {
        Err(anyhow!(
            "mapping.bonded_classing.patch class label must start with bond., angle., or dihedral.: {label}"
        ))
    }
}

fn canonical_key<const N: usize>(member: &[usize; N]) -> Vec<usize> {
    let forward = member.to_vec();
    let reverse = member.iter().rev().copied().collect::<Vec<_>>();
    if reverse < forward {
        reverse
    } else {
        forward
    }
}

fn raw_counts(terms: &BondedTermSet) -> Value {
    json!({
        "bonds": terms.bonds.iter().map(|group| group.members.len()).sum::<usize>(),
        "angles": terms.angles.iter().map(|group| group.members.len()).sum::<usize>(),
        "dihedrals": terms.dihedrals.iter().map(|group| group.members.len()).sum::<usize>()
    })
}

fn class_counts(terms: &BondedTermSet) -> Value {
    json!({
        "bonds": terms.bonds.len(),
        "angles": terms.angles.len(),
        "dihedrals": terms.dihedrals.len()
    })
}

fn unclassified_counts(
    auto_terms: &BondedTermSet,
    result: &BondedTermSet,
) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    counts.insert(
        "bonds",
        missing_count(
            known_members(auto_terms, TermKind::Bond),
            known_members(result, TermKind::Bond),
        ),
    );
    counts.insert(
        "angles",
        missing_count(
            known_members(auto_terms, TermKind::Angle),
            known_members(result, TermKind::Angle),
        ),
    );
    counts.insert(
        "dihedrals",
        missing_count(
            known_members(auto_terms, TermKind::Dihedral),
            known_members(result, TermKind::Dihedral),
        ),
    );
    counts
}

fn missing_count(auto: BTreeSet<Vec<usize>>, result: BTreeSet<Vec<usize>>) -> usize {
    auto.difference(&result).count()
}

fn classing_summary(
    mode: &str,
    request: Option<&BondedClassingRequest>,
    auto_terms: &BondedTermSet,
    result: &BondedTermSet,
    unclassified: BTreeMap<&'static str, usize>,
) -> Value {
    json!({
        "enabled": true,
        "mode": mode,
        "base": request.and_then(|item| item.base.as_deref()).unwrap_or("auto"),
        "source": request.and_then(|item| item.source.as_deref()).unwrap_or("template_role_order"),
        "class_source": match mode {
            "auto" => "template",
            "explicit" => "explicit",
            "patch" => "patch",
            _ => mode,
        },
        "raw_instance_counts": raw_counts(auto_terms),
        "class_counts": class_counts(result),
        "unclassified_counts": {
            "bonds": *unclassified.get("bonds").unwrap_or(&0),
            "angles": *unclassified.get("angles").unwrap_or(&0),
            "dihedrals": *unclassified.get("dihedrals").unwrap_or(&0)
        },
        "warnings": []
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auto_terms() -> BondedTermSet {
        BondedTermSet {
            bonds: vec![
                BondTermGroup {
                    label: Some("bond.middle.M0_M1__M0_M2".to_string()),
                    members: vec![[0, 1], [4, 5]],
                },
                BondTermGroup {
                    label: Some("bond.middle.M0_M2__M1_M1".to_string()),
                    members: vec![[1, 2], [5, 6]],
                },
            ],
            angles: vec![AngleTermGroup {
                label: Some("angle.middle.M0_M1__M0_M2__M1_M1".to_string()),
                members: vec![[0, 1, 2], [4, 5, 6]],
            }],
            dihedrals: vec![DihedralTermGroup {
                label: Some("dihedral.middle.M0_M1__M0_M2__M1_M1__M1_M2".to_string()),
                members: vec![[0, 1, 2, 3], [4, 5, 6, 7]],
            }],
            ..BondedTermSet::default()
        }
    }

    #[test]
    fn explicit_classing_overrides_auto_and_autofills_missing_terms() {
        let request = BondedClassingRequest {
            mode: "explicit".to_string(),
            bonds: vec![super::super::ExplicitBondClass {
                label: "aryl_ring_edge".to_string(),
                members: vec![[0, 1], [4, 5]],
            }],
            on_unclassified: Some("auto".to_string()),
            ..empty_request()
        };

        let resolved = resolve_bonded_classing(Some(&request), auto_terms(), 8).unwrap();

        assert!(resolved
            .terms
            .bonds
            .iter()
            .any(|group| group.label.as_deref() == Some("aryl_ring_edge")));
        assert!(
            resolved.summary["unclassified_counts"]["bonds"]
                .as_u64()
                .unwrap()
                == 0
        );
    }

    #[test]
    fn explicit_classing_rejects_duplicate_member_by_default() {
        let request = BondedClassingRequest {
            mode: "explicit".to_string(),
            bonds: vec![
                super::super::ExplicitBondClass {
                    label: "one".to_string(),
                    members: vec![[0, 1]],
                },
                super::super::ExplicitBondClass {
                    label: "two".to_string(),
                    members: vec![[1, 0]],
                },
            ],
            ..empty_request()
        };

        let err = resolve_bonded_classing(Some(&request), auto_terms(), 8).unwrap_err();
        assert!(err.to_string().contains("assigned to more than one class"));
    }

    #[test]
    fn patch_classing_can_merge_rename_and_split_auto_classes() {
        let request = BondedClassingRequest {
            mode: "patch".to_string(),
            base: Some("auto".to_string()),
            merge: vec![super::super::BondedClassMerge {
                label: "bond.aryl_ring_edge".to_string(),
                from: vec![
                    "bond.middle.M0_M1__M0_M2".to_string(),
                    "bond.middle.M0_M2__M1_M1".to_string(),
                ],
            }],
            rename: vec![super::super::BondedClassRename {
                from: "angle.middle.M0_M1__M0_M2__M1_M1".to_string(),
                to: "angle.aryl_internal".to_string(),
            }],
            split: vec![super::super::BondedClassSplit {
                from: "dihedral.middle.M0_M1__M0_M2__M1_M1__M1_M2".to_string(),
                into: vec![super::super::BondedClassSplitTarget {
                    label: "dihedral.terminal_variant".to_string(),
                    members: vec![vec![4, 5, 6, 7]],
                }],
            }],
            ..empty_request()
        };

        let resolved = resolve_bonded_classing(Some(&request), auto_terms(), 8).unwrap();

        assert!(resolved
            .terms
            .bonds
            .iter()
            .any(
                |group| group.label.as_deref() == Some("bond.aryl_ring_edge")
                    && group.members.len() == 4
            ));
        assert!(resolved
            .terms
            .angles
            .iter()
            .any(|group| group.label.as_deref() == Some("angle.aryl_internal")));
        assert!(resolved
            .terms
            .dihedrals
            .iter()
            .any(|group| group.label.as_deref() == Some("dihedral.terminal_variant")));
    }

    #[test]
    fn patch_merge_rejects_any_missing_source_label() {
        let request = BondedClassingRequest {
            mode: "patch".to_string(),
            merge: vec![super::super::BondedClassMerge {
                label: "bond.merged".to_string(),
                from: vec![
                    "bond.middle.M0_M1__M0_M2".to_string(),
                    "bond.middle.missing".to_string(),
                ],
            }],
            ..empty_request()
        };

        let err = resolve_bonded_classing(Some(&request), auto_terms(), 8).unwrap_err();

        assert!(err.to_string().contains("does not exist"));
    }

    fn empty_request() -> BondedClassingRequest {
        BondedClassingRequest {
            mode: "auto".to_string(),
            source: None,
            base: None,
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
            on_unclassified: None,
            on_duplicate_member: None,
            merge: Vec::new(),
            rename: Vec::new(),
            split: Vec::new(),
        }
    }
}
