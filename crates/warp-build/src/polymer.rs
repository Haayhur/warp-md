use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};

use rand::{rngs::StdRng, Rng, SeedableRng};
use roxmltree::{Document, Node};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use traj_core::{
    center_of_geometry,
    elements::{mass_for_element, normalize_element},
    normalize_vec3 as normalize, rotate_about_axis_vec3 as rotate_about_axis,
    rotate_from_to_vec3 as rotate_from_to, SpatialHash, Vec3,
};
use warp_pack::{PackError, PackResult};
use warp_structure::io::{
    read_molecule, read_prmtop_atom_charges, read_prmtop_topology, read_prmtop_total_charge,
    write_minimal_prmtop, write_output, AmberTopology, MoleculeData,
};
use warp_structure::{AtomRecord, OutputSpec, PackOutput};

pub const CHARGE_MANIFEST_VERSION: &str = "warp-build.charge-manifest.v1";
pub const LEGACY_CHARGE_MANIFEST_VERSION: &str = "warp-pack.charge-manifest.v1";
pub const SOURCE_BUNDLE_VERSION: &str = "polymer-param-source.bundle.v1";
pub const LEGACY_POLYMER_PARAM_VERSION: &str = "warp-pack.polymer-param.v1";

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChargeAtom {
    pub index: usize,
    pub charge_e: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChargeManifest {
    #[serde(
        default = "default_charge_manifest_version",
        alias = "version",
        rename = "schema_version"
    )]
    #[schemars(default = "default_charge_manifest_version", rename = "schema_version")]
    pub schema_version: String,
    #[serde(default)]
    pub solute_path: Option<String>,
    #[serde(default)]
    pub topology_ref: Option<String>,
    #[serde(default)]
    pub source_topology_ref: Option<String>,
    #[serde(default)]
    pub target_topology_ref: Option<String>,
    #[serde(default)]
    pub forcefield_ref: Option<String>,
    #[serde(default)]
    pub charge_derivation: Option<String>,
    #[serde(default)]
    pub net_charge_e: Option<f32>,
    #[serde(default)]
    pub atom_count: Option<usize>,
    #[serde(default)]
    pub partial_charges: Option<serde_json::Value>,
    #[serde(default)]
    pub atom_charges: Option<Vec<ChargeAtom>>,
    #[serde(default)]
    pub head_charge_e: Option<f32>,
    #[serde(default)]
    pub repeat_charge_e: Option<f32>,
    #[serde(default)]
    pub tail_charge_e: Option<f32>,
}

fn default_charge_manifest_version() -> String {
    CHARGE_MANIFEST_VERSION.to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PolymerParamBundle {
    pub version: String,
    pub training_structure: String,
    #[serde(default)]
    pub topology_ref: Option<String>,
    #[serde(default)]
    pub charge_manifest: Option<String>,
}

#[derive(Clone, Debug)]
pub struct PolymerSourceResolved {
    pub training_structure_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct PolymerBuiltArtifact {
    pub path: PathBuf,
    pub step_length_angstrom: f32,
    pub sequence_labels: Vec<String>,
    pub template_sequence_resnames: Vec<String>,
    pub residue_resnames: Vec<String>,
    pub output: PackOutput,
    pub qc_context: BuildQcContext,
    pub qc_report: BuildQcReport,
    pub solver_report: Option<BuildSolverReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct TrainingSourceMetrics {
    pub min_nonbonded_distance_angstrom: Option<f32>,
    pub max_local_bond_ratio: Option<f32>,
    pub impossible_valence_count: usize,
    pub ambiguous_assignment_count: usize,
    pub junction_consistency_ok: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct TrainingSourceAssessment {
    pub quality: String,
    pub parameter_source: String,
    #[serde(default)]
    pub reasons: Vec<String>,
    pub metrics: TrainingSourceMetrics,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrainingSourceQuality {
    Trusted,
    Risky,
    Unreliable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ParameterSourceChoice {
    SyntheticPdb,
    SourceTopologyRef,
    ForcefieldRef,
    Rejected,
}

#[derive(Clone, Debug)]
struct TrainingAssessmentThresholds {
    min_nonbonded_distance: Option<f32>,
    max_local_bond_ratio: Option<f32>,
    max_local_bond_delta: Option<f32>,
    impossible_valence_count: usize,
    ambiguous_assignment_count: usize,
    junction_consistency_ok: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BondQcViolation {
    pub edge_id: String,
    pub parent_resid: usize,
    pub child_resid: usize,
    pub parent_atom: String,
    pub child_atom: String,
    pub measured_distance_angstrom: f32,
    pub expected_distance_angstrom: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ClashQcViolation {
    pub atom_a: usize,
    pub atom_b: usize,
    pub distance_angstrom: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BuildQcReport {
    pub inter_residue_bond_count: usize,
    pub terminal_connectivity_consistent: bool,
    pub sequence_token_template_consistent: bool,
    pub min_nonbonded_distance_angstrom: Option<f32>,
    pub severe_nonbonded_clash_count: usize,
    pub severe_bond_violations: Vec<BondQcViolation>,
    pub severe_clash_examples: Vec<ClashQcViolation>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BuildSolverReport {
    pub enabled: bool,
    pub mode: String,
    pub seed: u64,
    pub passes_requested: usize,
    pub passes_executed: usize,
    pub rotatable_edge_count: usize,
    pub candidate_evaluations: usize,
    pub accepted_moves: usize,
    pub termination_reason: String,
    pub best_score: f32,
    #[serde(default)]
    pub hard_fail_reason: Option<String>,
}

#[derive(Clone, Debug)]
pub struct BuildBondExpectation {
    pub edge_id: String,
    pub parent_resid: usize,
    pub child_resid: usize,
    pub parent_atom: String,
    pub child_atom: String,
    pub parent_idx: usize,
    pub child_idx: usize,
    pub expected_distance_angstrom: f32,
}

#[derive(Clone, Debug)]
pub struct BuildQcContext {
    pub inter_residue_bond_count: usize,
    pub terminal_connectivity_consistent: bool,
    pub sequence_token_template_consistent: bool,
    pub bond_expectations: Vec<BuildBondExpectation>,
}

#[derive(Clone, Debug)]
pub struct NetChargeEstimate {
    pub net_charge_e: Option<f32>,
    pub source: Option<String>,
}

#[derive(Clone, Debug)]
pub struct TokenJunctionSpec {
    pub head_attach_atom: Option<String>,
    pub head_leaving_atoms: Vec<String>,
    pub tail_attach_atom: Option<String>,
    pub tail_leaving_atoms: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct GraphNodeSpec {
    pub sequence_label: String,
    pub template_resname: String,
    pub applied_resname: String,
}

#[derive(Clone, Debug)]
pub struct GraphEdgeSpec {
    pub edge_id: String,
    pub parent: usize,
    pub child: usize,
    pub parent_attach_atom: String,
    pub parent_leaving_atoms: Vec<String>,
    pub child_attach_atom: String,
    pub child_leaving_atoms: Vec<String>,
    pub bond_order: u8,
    pub branch_spread: String,
    pub torsion_mode: String,
    pub torsion_deg: Option<f32>,
    pub torsion_window_deg: Option<[f32; 2]>,
}

#[derive(Clone, Debug)]
struct ResidueTemplate {
    resname: String,
    atoms: Vec<AtomRecord>,
    centroid: Vec3,
    local_bonds: Vec<(usize, usize)>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct BuildResidueSpec {
    sequence_label: String,
    template_resname: String,
    applied_resname: String,
}

fn centroid_of_atoms(atoms: &[AtomRecord]) -> Vec3 {
    let positions: Vec<Vec3> = atoms.iter().map(|atom| atom.position).collect();
    center_of_geometry(&positions)
}

#[allow(dead_code)]
fn normalize_label(label: Option<&str>, fallback: &str) -> String {
    let raw = label.unwrap_or("default").trim();
    if raw.is_empty()
        || raw.eq_ignore_ascii_case("default")
        || raw.eq_ignore_ascii_case("training")
        || raw.eq_ignore_ascii_case("source_default")
    {
        return fallback.to_string();
    }
    raw.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .take(3)
        .collect::<String>()
        .to_ascii_uppercase()
}

fn tacticity_phase(tacticity_mode: &str, idx: usize, rng: &mut StdRng) -> PackResult<f32> {
    match tacticity_mode {
        "default" | "training" | "isotactic" => Ok(0.0),
        "syndiotactic" => Ok(if idx % 2 == 0 {
            0.0
        } else {
            std::f32::consts::PI
        }),
        "atactic" => Ok(rng.gen_range(0.0..(2.0 * std::f32::consts::PI))),
        other => Err(PackError::Invalid(format!(
            "unsupported polymer tacticity mode '{}'",
            other
        ))),
    }
}

fn group_residues(molecule: MoleculeData) -> PackResult<Vec<ResidueTemplate>> {
    let mut grouped = Vec::new();
    let atoms = molecule.atoms;
    let bonds = molecule.bonds;
    let mut residue_index = BTreeMap::<(char, i32, String), usize>::new();
    let mut global_to_local = vec![None; atoms.len()];

    for (global_idx, atom) in atoms.into_iter().enumerate() {
        let chain = if atom.chain == ' ' { 'A' } else { atom.chain };
        let resname = if atom.resname.trim().is_empty() {
            "MOL".to_string()
        } else {
            atom.resname.clone()
        };
        let key = (chain, atom.resid, resname.clone());
        let group_idx = if let Some(idx) = residue_index.get(&key).copied() {
            idx
        } else {
            let idx = grouped.len();
            residue_index.insert(key, idx);
            grouped.push(ResidueTemplate {
                resname: resname.clone(),
                atoms: Vec::new(),
                centroid: Vec3::new(0.0, 0.0, 0.0),
                local_bonds: Vec::new(),
            });
            idx
        };
        let local_idx = grouped[group_idx].atoms.len();
        grouped[group_idx].atoms.push(AtomRecord {
            record_kind: atom.record_kind,
            name: atom.name,
            element: atom.element,
            resname,
            resid: atom.resid,
            chain,
            segid: atom.segid,
            charge: 0.0,
            position: Vec3::new(atom.position.x, atom.position.y, atom.position.z),
            mol_id: 1,
            pdb_metadata: atom.pdb_metadata,
        });
        global_to_local[global_idx] = Some((group_idx, local_idx));
    }

    if grouped.len() < 3 {
        return Err(PackError::Invalid(
            "training oligomer must contain at least 3 residues".into(),
        ));
    }

    for template in &mut grouped {
        template.centroid = centroid_of_atoms(&template.atoms);
    }

    if bonds.is_empty() {
        for template in &mut grouped {
            template.local_bonds = infer_local_bonds(&template.atoms);
        }
    } else {
        for &(a, b) in &bonds {
            let Some(Some((group_a, local_a))) = global_to_local.get(a) else {
                continue;
            };
            let Some(Some((group_b, local_b))) = global_to_local.get(b) else {
                continue;
            };
            if group_a != group_b {
                continue;
            }
            let (i, j) = if local_a <= local_b {
                (*local_a, *local_b)
            } else {
                (*local_b, *local_a)
            };
            if i != j {
                grouped[*group_a].local_bonds.push((i, j));
            }
        }
        for template in &mut grouped {
            template.local_bonds.sort_unstable();
            template.local_bonds.dedup();
            if template.local_bonds.is_empty() {
                template.local_bonds = infer_local_bonds(&template.atoms);
            }
        }
    }

    Ok(grouped)
}

fn infer_local_bonds(atoms: &[AtomRecord]) -> Vec<(usize, usize)> {
    let mut bonds = Vec::new();
    for i in 0..atoms.len() {
        for j in (i + 1)..atoms.len() {
            let max_distance = (covalent_radius_angstrom(&atoms[i].element)
                + covalent_radius_angstrom(&atoms[j].element)
                + 0.45)
                .clamp(0.9, 2.2);
            let distance = atoms[i].position.sub(atoms[j].position).norm();
            if distance >= 0.35 && distance <= max_distance {
                bonds.push((i, j));
            }
        }
    }
    bonds
}

fn infer_global_bonds(atoms: &[AtomRecord]) -> Vec<(usize, usize)> {
    let mut bonds = Vec::new();
    for i in 0..atoms.len() {
        for j in (i + 1)..atoms.len() {
            let max_distance = (covalent_radius_angstrom(&atoms[i].element)
                + covalent_radius_angstrom(&atoms[j].element)
                + 0.45)
                .clamp(0.9, 2.2);
            let distance = atoms[i].position.sub(atoms[j].position).norm();
            if distance >= 0.35 && distance <= max_distance {
                bonds.push((i, j));
            }
        }
    }
    bonds
}

fn dedup_bonds(mut bonds: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    for bond in &mut bonds {
        if bond.0 > bond.1 {
            *bond = (bond.1, bond.0);
        }
    }
    bonds.sort_unstable();
    bonds.dedup();
    bonds
}

fn heavy_atom(element: &str) -> bool {
    normalize_element(element)
        .map(|value| value != "H")
        .unwrap_or(true)
}

fn element_max_valence(element: &str) -> usize {
    match normalize_element(element).as_deref() {
        Some("H") => 1,
        Some("B") => 4,
        Some("C") => 4,
        Some("N") => 4,
        Some("O") => 2,
        Some("F") => 1,
        Some("P") => 6,
        Some("S") => 6,
        Some("Cl") => 1,
        Some("Br") => 1,
        Some("I") => 1,
        Some("Si") => 4,
        _ => 6,
    }
}

fn atom_assignment_ambiguous(
    atoms: &[AtomRecord],
    adjacency: &[Vec<usize>],
    atom_idx: usize,
) -> bool {
    let Some(atom) = atoms.get(atom_idx) else {
        return false;
    };
    if !heavy_atom(&atom.element) {
        return false;
    }
    let element = normalize_element(&atom.element).unwrap_or_else(|| atom.element.clone());
    let degree = adjacency
        .get(atom_idx)
        .map(|items| items.len())
        .unwrap_or(0);
    if degree == 0 {
        return false;
    }
    let mean_angle = average_neighbor_angle(atoms, adjacency, atom_idx).unwrap_or(109.5);
    match element.as_str() {
        "C" | "N" => {
            (degree == 2 && (112.0..160.0).contains(&mean_angle))
                || (degree == 3 && (111.0..119.0).contains(&mean_angle))
        }
        "O" | "S" => {
            degree == 1
                && adjacency
                    .get(atom_idx)
                    .into_iter()
                    .flatten()
                    .any(|neighbor| {
                        let distance = atoms[atom_idx]
                            .position
                            .sub(atoms[*neighbor].position)
                            .norm();
                        distance > 1.28 && distance < 1.42
                    })
        }
        _ => false,
    }
}

fn collect_training_thresholds(
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
    graph_node_specs: &[GraphNodeSpec],
    token_junctions: &BTreeMap<String, TokenJunctionSpec>,
    templates: &[ResidueTemplate],
) -> TrainingAssessmentThresholds {
    let adjacency = rebuild_bond_adjacency(atoms.len(), bonds);
    let typings = (0..atoms.len())
        .map(|atom_idx| infer_uff_like_typing(atoms, &adjacency, atom_idx))
        .collect::<Vec<_>>();
    let bonded = bonds
        .iter()
        .copied()
        .map(|(a, b)| ordered_pair(a, b))
        .collect::<BTreeSet<_>>();
    let mut one_three = BTreeSet::new();
    for (center, neighbors) in adjacency.iter().enumerate() {
        for left in neighbors {
            for right in neighbors {
                if left >= right {
                    continue;
                }
                one_three.insert(ordered_pair(*left, *right));
            }
        }
        for neighbor in neighbors {
            for outer in adjacency.get(*neighbor).into_iter().flatten() {
                if *outer == center {
                    continue;
                }
                one_three.insert(ordered_pair(center, *outer));
            }
        }
    }

    let mut min_nonbonded_distance: Option<f32> = None;
    for left in 0..atoms.len() {
        for right in (left + 1)..atoms.len() {
            let pair = ordered_pair(left, right);
            if bonded.contains(&pair) || one_three.contains(&pair) {
                continue;
            }
            let distance = atoms[left].position.sub(atoms[right].position).norm();
            min_nonbonded_distance = Some(match min_nonbonded_distance {
                Some(current) => current.min(distance),
                None => distance,
            });
        }
    }

    let mut max_local_bond_ratio: Option<f32> = None;
    let mut max_local_bond_delta: Option<f32> = None;
    for &(left, right) in bonds {
        let measured = atoms[left].position.sub(atoms[right].position).norm();
        let left_typing = typings
            .get(left)
            .map(|item| item.params)
            .unwrap_or_else(|| uff_params_for_label("C_3"));
        let right_typing = typings
            .get(right)
            .map(|item| item.params)
            .unwrap_or_else(|| uff_params_for_label("C_3"));
        let order = guess_bond_order(atoms, &adjacency, &typings, left, right);
        let ideal = uff_bond_rest_length(order, left_typing, right_typing).max(0.8);
        let ratio = measured / ideal.max(1.0e-3);
        let delta = (measured - ideal).max(0.0);
        max_local_bond_ratio = Some(match max_local_bond_ratio {
            Some(current) => current.max(ratio),
            None => ratio,
        });
        max_local_bond_delta = Some(match max_local_bond_delta {
            Some(current) => current.max(delta),
            None => delta,
        });
    }

    let mut impossible_valence_count = 0usize;
    let mut ambiguous_assignment_count = 0usize;
    for atom_idx in 0..atoms.len() {
        let Some(atom) = atoms.get(atom_idx) else {
            continue;
        };
        if !heavy_atom(&atom.element) {
            continue;
        }
        let degree = adjacency
            .get(atom_idx)
            .map(|items| items.len())
            .unwrap_or(0);
        if degree > element_max_valence(&atom.element) {
            impossible_valence_count += 1;
            continue;
        }
        if atom_assignment_ambiguous(atoms, &adjacency, atom_idx) {
            ambiguous_assignment_count += 1;
        }
    }

    let mut critical_atoms = BTreeMap::<String, BTreeSet<String>>::new();
    for spec in graph_node_specs {
        if let Some(junction) = token_junctions.get(&spec.sequence_label) {
            let entry = critical_atoms
                .entry(spec.template_resname.clone())
                .or_default();
            if let Some(atom) = junction.head_attach_atom.as_ref() {
                entry.insert(atom.trim().to_string());
            }
            if let Some(atom) = junction.tail_attach_atom.as_ref() {
                entry.insert(atom.trim().to_string());
            }
            entry.extend(
                junction
                    .head_leaving_atoms
                    .iter()
                    .map(|item| item.trim().to_string()),
            );
            entry.extend(
                junction
                    .tail_leaving_atoms
                    .iter()
                    .map(|item| item.trim().to_string()),
            );
        }
    }
    let mut junction_consistency_ok = true;
    for template in templates {
        let Some(required) = critical_atoms.get(&template.resname) else {
            continue;
        };
        let local_adjacency = rebuild_bond_adjacency(template.atoms.len(), &template.local_bonds);
        let mut critical_indices = BTreeSet::new();
        for atom_name in required {
            let Some(atom_idx) = template
                .atoms
                .iter()
                .position(|atom| atom.name.trim() == atom_name.trim())
            else {
                junction_consistency_ok = false;
                continue;
            };
            critical_indices.insert(atom_idx);
            critical_indices.extend(local_adjacency.get(atom_idx).into_iter().flatten().copied());
        }
        for atom_idx in critical_indices {
            let degree = local_adjacency
                .get(atom_idx)
                .map(|items| items.len())
                .unwrap_or(0);
            let atom = &template.atoms[atom_idx];
            if degree > element_max_valence(&atom.element) {
                junction_consistency_ok = false;
                break;
            }
            if atom_assignment_ambiguous(&template.atoms, &local_adjacency, atom_idx) {
                junction_consistency_ok = false;
                break;
            }
        }
        if !junction_consistency_ok {
            break;
        }
    }

    TrainingAssessmentThresholds {
        min_nonbonded_distance,
        max_local_bond_ratio,
        max_local_bond_delta,
        impossible_valence_count,
        ambiguous_assignment_count,
        junction_consistency_ok,
    }
}

fn training_source_quality(thresholds: &TrainingAssessmentThresholds) -> TrainingSourceQuality {
    let severe_overlap = thresholds
        .min_nonbonded_distance
        .map(|value| value < 0.60)
        .unwrap_or(false);
    let risky_overlap = thresholds
        .min_nonbonded_distance
        .map(|value| value < 0.90)
        .unwrap_or(false);
    let severe_bond = thresholds
        .max_local_bond_ratio
        .map(|value| value > 1.30)
        .unwrap_or(false)
        || thresholds
            .max_local_bond_delta
            .map(|value| value > 0.25)
            .unwrap_or(false);
    let risky_bond = thresholds
        .max_local_bond_ratio
        .map(|value| value > 1.15)
        .unwrap_or(false)
        || thresholds
            .max_local_bond_delta
            .map(|value| value > 0.10)
            .unwrap_or(false);
    if thresholds.impossible_valence_count > 0 || severe_overlap || severe_bond {
        TrainingSourceQuality::Unreliable
    } else if !thresholds.junction_consistency_ok
        || risky_overlap
        || risky_bond
        || thresholds.ambiguous_assignment_count > 0
    {
        TrainingSourceQuality::Risky
    } else {
        TrainingSourceQuality::Trusted
    }
}

fn training_source_quality_label(quality: TrainingSourceQuality) -> String {
    match quality {
        TrainingSourceQuality::Trusted => "trusted".into(),
        TrainingSourceQuality::Risky => "risky".into(),
        TrainingSourceQuality::Unreliable => "unreliable".into(),
    }
}

fn parameter_source_label(choice: ParameterSourceChoice) -> String {
    match choice {
        ParameterSourceChoice::SyntheticPdb => "synthetic_pdb".into(),
        ParameterSourceChoice::SourceTopologyRef => "source_topology_ref".into(),
        ParameterSourceChoice::ForcefieldRef => "forcefield_ref".into(),
        ParameterSourceChoice::Rejected => "rejected".into(),
    }
}

fn load_training_templates(path: &Path) -> PackResult<Vec<ResidueTemplate>> {
    let molecule = read_molecule(path, None, false, true, None)?;
    group_residues(molecule)
}

fn residue_charge_sums(templates: &[ResidueTemplate], atom_charges: &[f32]) -> Vec<f32> {
    let mut sums = Vec::with_capacity(templates.len());
    let mut cursor = 0usize;
    for residue in templates {
        let mut total = 0.0f32;
        for _ in &residue.atoms {
            if let Some(charge) = atom_charges.get(cursor) {
                total += *charge;
            }
            cursor += 1;
        }
        sums.push(total);
    }
    sums
}

fn template_charge_map(
    templates: &[ResidueTemplate],
    atom_charges: &[f32],
) -> BTreeMap<String, f32> {
    let sums = residue_charge_sums(templates, atom_charges);
    let mut totals = BTreeMap::new();
    let mut counts = BTreeMap::new();
    for (template, charge) in templates.iter().zip(sums) {
        *totals.entry(template.resname.clone()).or_insert(0.0) += charge;
        *counts.entry(template.resname.clone()).or_insert(0usize) += 1;
    }
    totals
        .into_iter()
        .map(|(resname, total)| {
            let count = counts.get(&resname).copied().unwrap_or(1).max(1) as f32;
            (resname, total / count)
        })
        .collect()
}

fn sum_sequence_template_charges(
    template_charge_map: &BTreeMap<String, f32>,
    template_sequence_resnames: &[String],
) -> Option<f32> {
    let mut total = 0.0f32;
    for resname in template_sequence_resnames {
        total += *template_charge_map.get(resname)?;
    }
    Some(total)
}

#[allow(dead_code)]
fn resolve_template_sequence(
    sequence_labels: &[String],
    template_resname_by_token: &BTreeMap<String, String>,
    head_label: Option<&str>,
    tail_label: Option<&str>,
) -> PackResult<Vec<BuildResidueSpec>> {
    let resolve_terminus_token =
        |policy: Option<&str>, fallback_token: &str| -> PackResult<String> {
            let raw = policy.unwrap_or("default").trim();
            if raw.is_empty()
                || raw.eq_ignore_ascii_case("default")
                || raw.eq_ignore_ascii_case("training")
                || raw.eq_ignore_ascii_case("source_default")
            {
                return Ok(fallback_token.to_string());
            }
            if template_resname_by_token.contains_key(raw) {
                return Ok(raw.to_string());
            }
            Ok(fallback_token.to_string())
        };

    let mut resolved_tokens = sequence_labels.to_vec();
    if let Some(first) = resolved_tokens.first_mut() {
        *first = resolve_terminus_token(head_label, first)?;
    }
    if let Some(last) = resolved_tokens.last_mut() {
        *last = resolve_terminus_token(tail_label, last)?;
    }

    let mut specs = resolved_tokens
        .iter()
        .map(|token| {
            let template_resname =
                template_resname_by_token
                    .get(token)
                    .cloned()
                    .ok_or_else(|| {
                        PackError::Invalid(format!(
                            "missing template_resname mapping for sequence token '{token}'"
                        ))
                    })?;
            Ok(BuildResidueSpec {
                sequence_label: token.clone(),
                applied_resname: normalize_label(Some(token), &template_resname),
                template_resname,
            })
        })
        .collect::<PackResult<Vec<_>>>()?;

    if let Some(first) = specs.first_mut() {
        first.applied_resname = normalize_label(head_label, &first.applied_resname);
    }
    if let Some(last) = specs.last_mut() {
        last.applied_resname = normalize_label(tail_label, &last.applied_resname);
    }

    Ok(specs)
}

fn template_atom_index_by_name(template: &ResidueTemplate) -> BTreeMap<String, usize> {
    template
        .atoms
        .iter()
        .enumerate()
        .map(|(idx, atom)| (atom.name.trim().to_string(), idx))
        .collect()
}

fn template_atom_by_name<'a>(
    template: &'a ResidueTemplate,
    atom_name: &str,
) -> PackResult<&'a AtomRecord> {
    template
        .atoms
        .iter()
        .find(|atom| atom.name.trim() == atom_name.trim())
        .ok_or_else(|| {
            let available = template
                .atoms
                .iter()
                .map(|atom| atom.name.trim())
                .filter(|name| !name.is_empty())
                .collect::<Vec<_>>()
                .join(", ");
            PackError::Invalid(format!(
                "attach atom '{}' missing from template '{}' (available atoms: [{}])",
                atom_name, template.resname, available
            ))
        })
}

fn covalent_radius_angstrom(element: &str) -> f32 {
    match element.trim().to_ascii_uppercase().as_str() {
        "H" => 0.31,
        "B" => 0.85,
        "C" => 0.76,
        "N" => 0.71,
        "O" => 0.66,
        "F" => 0.57,
        "P" => 1.07,
        "S" => 1.05,
        "CL" => 1.02,
        "BR" => 1.20,
        "I" => 1.39,
        "SI" => 1.11,
        _ => 0.80,
    }
}

#[derive(Clone, Copy, Debug)]
struct UffAtomParams {
    label: &'static str,
    r1: f32,
    theta0_rad: f32,
    x1: f32,
    d1: f32,
    z1: f32,
    v1: f32,
    u1: f32,
    xi: f32,
}

#[derive(Clone, Debug)]
struct SyntheticAtomTyping {
    params: UffAtomParams,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AtomGeometryClass {
    Sp1,
    Sp2,
    Sp3,
}

#[derive(Clone, Debug)]
struct SyntheticBondSpec {
    bond: (usize, usize),
    rest_length: f32,
    force_constant: f32,
}

#[derive(Clone, Debug)]
struct SyntheticDihedralParam {
    force_constant: f32,
    periodicity: f32,
    phase_rad: f32,
}

fn deg_to_rad(value: f32) -> f32 {
    value * std::f32::consts::PI / 180.0
}

fn atomic_number_for_element(element: &str) -> i32 {
    match normalize_element(element).as_deref().unwrap_or(element) {
        "H" => 1,
        "B" => 5,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        "P" => 15,
        "S" => 16,
        "Cl" => 17,
        "Si" => 14,
        "Br" => 35,
        "I" => 53,
        _ => 0,
    }
}

fn element_mass_amu(element: &str) -> f32 {
    let normalized = normalize_element(element).unwrap_or_else(|| element.trim().to_string());
    let mass = mass_for_element(&normalized);
    if mass > 0.0 {
        mass
    } else {
        12.0
    }
}

fn average_neighbor_angle(
    atoms: &[AtomRecord],
    adjacency: &[Vec<usize>],
    atom_idx: usize,
) -> Option<f32> {
    let neighbors = adjacency.get(atom_idx)?;
    if neighbors.len() < 2 {
        return None;
    }
    let center = atoms.get(atom_idx)?.position;
    let mut total = 0.0f32;
    let mut count = 0usize;
    for left_idx in 0..neighbors.len() {
        for right_idx in (left_idx + 1)..neighbors.len() {
            let left = atoms.get(neighbors[left_idx])?.position.sub(center);
            let right = atoms.get(neighbors[right_idx])?.position.sub(center);
            let left_norm = left.norm();
            let right_norm = right.norm();
            if left_norm <= 1.0e-6 || right_norm <= 1.0e-6 {
                continue;
            }
            let cos_theta = (left.dot(right) / (left_norm * right_norm)).clamp(-1.0, 1.0);
            total += cos_theta.acos() * 180.0 / std::f32::consts::PI;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(total / count as f32)
    }
}

fn infer_geometry_class(
    atoms: &[AtomRecord],
    adjacency: &[Vec<usize>],
    atom_idx: usize,
) -> AtomGeometryClass {
    let degree = adjacency
        .get(atom_idx)
        .map(|items| items.len())
        .unwrap_or(0);
    let mean_angle = average_neighbor_angle(atoms, adjacency, atom_idx).unwrap_or(109.5);
    if degree <= 2 && mean_angle >= 155.0 {
        AtomGeometryClass::Sp1
    } else if mean_angle >= 116.0 || degree <= 2 {
        AtomGeometryClass::Sp2
    } else {
        AtomGeometryClass::Sp3
    }
}

fn uff_params_for_label(label: &'static str) -> UffAtomParams {
    match label {
        "H_" => UffAtomParams {
            label,
            r1: 0.354,
            theta0_rad: deg_to_rad(180.0),
            x1: 2.886,
            d1: 0.044,
            z1: 0.712,
            v1: 0.0,
            u1: 0.0,
            xi: 4.528,
        },
        "C_1" => UffAtomParams {
            label,
            r1: 0.706,
            theta0_rad: deg_to_rad(180.0),
            x1: 3.851,
            d1: 0.105,
            z1: 1.912,
            v1: 0.0,
            u1: 2.0,
            xi: 5.343,
        },
        "C_2" => UffAtomParams {
            label,
            r1: 0.732,
            theta0_rad: deg_to_rad(120.0),
            x1: 3.851,
            d1: 0.105,
            z1: 1.912,
            v1: 0.0,
            u1: 2.0,
            xi: 5.343,
        },
        "C_3" => UffAtomParams {
            label,
            r1: 0.757,
            theta0_rad: deg_to_rad(109.47),
            x1: 3.851,
            d1: 0.105,
            z1: 1.912,
            v1: 2.119,
            u1: 2.0,
            xi: 5.343,
        },
        "N_1" => UffAtomParams {
            label,
            r1: 0.656,
            theta0_rad: deg_to_rad(180.0),
            x1: 3.66,
            d1: 0.069,
            z1: 2.544,
            v1: 0.0,
            u1: 2.0,
            xi: 6.899,
        },
        "N_2" => UffAtomParams {
            label,
            r1: 0.685,
            theta0_rad: deg_to_rad(111.2),
            x1: 3.66,
            d1: 0.069,
            z1: 2.544,
            v1: 0.0,
            u1: 2.0,
            xi: 6.899,
        },
        "N_3" => UffAtomParams {
            label,
            r1: 0.7,
            theta0_rad: deg_to_rad(106.7),
            x1: 3.66,
            d1: 0.069,
            z1: 2.544,
            v1: 0.45,
            u1: 2.0,
            xi: 6.899,
        },
        "O_1" => UffAtomParams {
            label,
            r1: 0.639,
            theta0_rad: deg_to_rad(180.0),
            x1: 3.5,
            d1: 0.06,
            z1: 2.3,
            v1: 0.0,
            u1: 2.0,
            xi: 8.741,
        },
        "O_2" => UffAtomParams {
            label,
            r1: 0.634,
            theta0_rad: deg_to_rad(120.0),
            x1: 3.5,
            d1: 0.06,
            z1: 2.3,
            v1: 0.0,
            u1: 2.0,
            xi: 8.741,
        },
        "O_3" => UffAtomParams {
            label,
            r1: 0.658,
            theta0_rad: deg_to_rad(104.51),
            x1: 3.5,
            d1: 0.06,
            z1: 2.3,
            v1: 0.018,
            u1: 2.0,
            xi: 8.741,
        },
        "F_" => UffAtomParams {
            label,
            r1: 0.668,
            theta0_rad: deg_to_rad(180.0),
            x1: 3.364,
            d1: 0.05,
            z1: 1.735,
            v1: 0.0,
            u1: 2.0,
            xi: 10.874,
        },
        "Si3" => UffAtomParams {
            label,
            r1: 1.117,
            theta0_rad: deg_to_rad(109.47),
            x1: 4.295,
            d1: 0.402,
            z1: 2.323,
            v1: 1.225,
            u1: 1.25,
            xi: 4.168,
        },
        "P_3+5" => UffAtomParams {
            label,
            r1: 1.056,
            theta0_rad: deg_to_rad(109.47),
            x1: 4.147,
            d1: 0.305,
            z1: 2.863,
            v1: 2.4,
            u1: 1.25,
            xi: 5.463,
        },
        "S_2" => UffAtomParams {
            label,
            r1: 0.854,
            theta0_rad: deg_to_rad(120.0),
            x1: 4.035,
            d1: 0.274,
            z1: 2.703,
            v1: 0.0,
            u1: 1.25,
            xi: 6.928,
        },
        "S_3+6" => UffAtomParams {
            label,
            r1: 1.027,
            theta0_rad: deg_to_rad(109.47),
            x1: 4.035,
            d1: 0.274,
            z1: 2.703,
            v1: 0.484,
            u1: 1.25,
            xi: 6.928,
        },
        "Cl" => UffAtomParams {
            label,
            r1: 1.044,
            theta0_rad: deg_to_rad(180.0),
            x1: 3.947,
            d1: 0.227,
            z1: 2.348,
            v1: 0.0,
            u1: 1.25,
            xi: 8.564,
        },
        "Br" => UffAtomParams {
            label,
            r1: 1.192,
            theta0_rad: deg_to_rad(180.0),
            x1: 4.189,
            d1: 0.251,
            z1: 2.519,
            v1: 0.0,
            u1: 0.7,
            xi: 7.79,
        },
        "I_" => UffAtomParams {
            label,
            r1: 1.382,
            theta0_rad: deg_to_rad(180.0),
            x1: 4.5,
            d1: 0.339,
            z1: 2.65,
            v1: 0.0,
            u1: 0.2,
            xi: 6.822,
        },
        _ => uff_params_for_label("C_3"),
    }
}

fn infer_uff_like_typing(
    atoms: &[AtomRecord],
    adjacency: &[Vec<usize>],
    atom_idx: usize,
) -> SyntheticAtomTyping {
    let atom = atoms.get(atom_idx);
    let normalized = atom
        .and_then(|item| normalize_element(&item.element))
        .unwrap_or_else(|| "C".to_string());
    let geometry = infer_geometry_class(atoms, adjacency, atom_idx);
    let label = match normalized.as_str() {
        "H" => "H_",
        "C" => {
            let shortest_neighbor = adjacency
                .get(atom_idx)
                .into_iter()
                .flatten()
                .filter_map(|neighbor| {
                    Some(
                        atoms
                            .get(atom_idx)?
                            .position
                            .sub(atoms.get(*neighbor)?.position)
                            .norm(),
                    )
                })
                .fold(f32::INFINITY, f32::min);
            match geometry {
                AtomGeometryClass::Sp1 if shortest_neighbor <= 1.24 => "C_1",
                AtomGeometryClass::Sp1 => "C_3",
                AtomGeometryClass::Sp2 if shortest_neighbor <= 1.47 => "C_2",
                AtomGeometryClass::Sp2 => "C_3",
                AtomGeometryClass::Sp3 => "C_3",
            }
        }
        "N" => match geometry {
            AtomGeometryClass::Sp1 => "N_1",
            AtomGeometryClass::Sp2 => "N_2",
            AtomGeometryClass::Sp3 => "N_3",
        },
        "O" => {
            let degree = adjacency
                .get(atom_idx)
                .map(|items| items.len())
                .unwrap_or(0);
            if degree <= 1 {
                "O_2"
            } else {
                "O_3"
            }
        }
        "F" => "F_",
        "Si" => "Si3",
        "P" => "P_3+5",
        "S" => {
            let degree = adjacency
                .get(atom_idx)
                .map(|items| items.len())
                .unwrap_or(0);
            if degree <= 2 && geometry != AtomGeometryClass::Sp3 {
                "S_2"
            } else {
                "S_3+6"
            }
        }
        "Cl" => "Cl",
        "Br" => "Br",
        "I" => "I_",
        _ => "C_3",
    };
    SyntheticAtomTyping {
        params: uff_params_for_label(label),
    }
}

fn template_bond_length_map(
    templates: &[ResidueTemplate],
) -> BTreeMap<(String, String, String), f32> {
    let mut totals = BTreeMap::<(String, String, String), (f32, usize)>::new();
    for template in templates {
        for &(left, right) in &template.local_bonds {
            let Some(atom_left) = template.atoms.get(left) else {
                continue;
            };
            let Some(atom_right) = template.atoms.get(right) else {
                continue;
            };
            let distance = atom_left.position.sub(atom_right.position).norm();
            let name_left = atom_left.name.trim().to_string();
            let name_right = atom_right.name.trim().to_string();
            let (a, b) = if name_left <= name_right {
                (name_left, name_right)
            } else {
                (name_right, name_left)
            };
            let entry = totals
                .entry((template.resname.clone(), a, b))
                .or_insert((0.0, 0));
            entry.0 += distance;
            entry.1 += 1;
        }
    }
    totals
        .into_iter()
        .map(|(key, (total, count))| (key, total / count.max(1) as f32))
        .collect()
}

fn template_angle_key(
    resname: &str,
    a: &str,
    b: &str,
    c: &str,
) -> (String, String, String, String) {
    let left = (a.trim().to_string(), c.trim().to_string());
    let right = (c.trim().to_string(), a.trim().to_string());
    let (outer_a, outer_c) = if left <= right { left } else { right };
    (resname.to_string(), outer_a, b.trim().to_string(), outer_c)
}

fn template_angle_map(
    templates: &[ResidueTemplate],
) -> BTreeMap<(String, String, String, String), f32> {
    let mut totals = BTreeMap::<(String, String, String, String), (f32, usize)>::new();
    for template in templates {
        let adjacency = rebuild_bond_adjacency(template.atoms.len(), &template.local_bonds);
        for angle in rebuild_angles(&adjacency) {
            let Some(atom_a) = template.atoms.get(angle[0]) else {
                continue;
            };
            let Some(atom_b) = template.atoms.get(angle[1]) else {
                continue;
            };
            let Some(atom_c) = template.atoms.get(angle[2]) else {
                continue;
            };
            let ab = atom_a.position.sub(atom_b.position);
            let cb = atom_c.position.sub(atom_b.position);
            let ab_norm = ab.norm();
            let cb_norm = cb.norm();
            if ab_norm <= 1.0e-6 || cb_norm <= 1.0e-6 {
                continue;
            }
            let theta = (ab.dot(cb) / (ab_norm * cb_norm)).clamp(-1.0, 1.0).acos();
            let entry = totals
                .entry(template_angle_key(
                    &template.resname,
                    &atom_a.name,
                    &atom_b.name,
                    &atom_c.name,
                ))
                .or_insert((0.0, 0));
            entry.0 += theta;
            entry.1 += 1;
        }
    }
    totals
        .into_iter()
        .map(|(key, (total, count))| (key, total / count.max(1) as f32))
        .collect()
}

fn template_atom_charge_map(
    templates: &[ResidueTemplate],
    atom_charges: &[f32],
) -> BTreeMap<(String, String), f32> {
    let mut totals = BTreeMap::<(String, String), (f32, usize)>::new();
    let mut cursor = 0usize;
    for template in templates {
        for atom in &template.atoms {
            let Some(charge) = atom_charges.get(cursor) else {
                break;
            };
            let entry = totals
                .entry((template.resname.clone(), atom.name.trim().to_string()))
                .or_insert((0.0, 0));
            entry.0 += *charge;
            entry.1 += 1;
            cursor += 1;
        }
    }
    totals
        .into_iter()
        .map(|(key, (total, count))| (key, total / count.max(1) as f32))
        .collect()
}

fn build_template_charge_map_from_manifest(
    training_structure_path: &Path,
    manifest_path: &Path,
) -> PackResult<BTreeMap<(String, String), f32>> {
    let manifest = load_charge_manifest(manifest_path)?;
    let atom_charges = manifest
        .atom_charges
        .unwrap_or_default()
        .into_iter()
        .map(|item| item.charge_e)
        .collect::<Vec<_>>();
    if atom_charges.is_empty() {
        return Ok(BTreeMap::new());
    }
    let templates = load_training_templates(training_structure_path)?;
    Ok(template_atom_charge_map(&templates, &atom_charges))
}

#[derive(Clone, Debug)]
struct FfxmlAtomTypeDef {
    class_name: String,
    element: String,
    mass: f32,
}

#[derive(Clone, Debug)]
struct FfxmlResidueAtomDef {
    name: String,
    type_name: String,
    charge_e: Option<f32>,
}

#[derive(Clone, Debug)]
struct FfxmlResidueDef {
    atoms: BTreeMap<String, FfxmlResidueAtomDef>,
}

#[derive(Clone, Debug)]
enum FfxmlAtomSelector {
    Any,
    Type(String),
    Class(String),
}

#[derive(Clone, Debug)]
struct FfxmlBondParam {
    left: FfxmlAtomSelector,
    right: FfxmlAtomSelector,
    length_angstrom: f32,
    force_constant: f32,
}

#[derive(Clone, Debug)]
struct FfxmlAngleParam {
    left: FfxmlAtomSelector,
    center: FfxmlAtomSelector,
    right: FfxmlAtomSelector,
    theta0_rad: f32,
    force_constant: f32,
}

#[derive(Clone, Debug)]
struct FfxmlTorsionTerm {
    periodicity: f32,
    phase_rad: f32,
    force_constant: f32,
}

#[derive(Clone, Debug)]
struct FfxmlTorsionParam {
    atoms: [FfxmlAtomSelector; 4],
    terms: Vec<FfxmlTorsionTerm>,
}

#[derive(Clone, Debug)]
struct FfxmlNonbondedParam {
    charge_e: Option<f32>,
    sigma_angstrom: f32,
    epsilon_kcal: f32,
}

#[derive(Clone, Debug)]
struct FfxmlForcefield {
    atom_types: BTreeMap<String, FfxmlAtomTypeDef>,
    residues: BTreeMap<String, FfxmlResidueDef>,
    bond_params: Vec<FfxmlBondParam>,
    angle_params: Vec<FfxmlAngleParam>,
    proper_torsions: Vec<FfxmlTorsionParam>,
    improper_torsions: Vec<FfxmlTorsionParam>,
    nonbonded_by_type: BTreeMap<String, FfxmlNonbondedParam>,
    nonbonded_by_class: BTreeMap<String, FfxmlNonbondedParam>,
    nonbonded_charge_from_residue: bool,
    scee_scale: f32,
    scnb_scale: f32,
}

#[derive(Clone, Debug)]
struct FfxmlAssignedAtom {
    type_name: String,
    class_name: String,
    element: String,
    mass: f32,
    charge_e: f32,
    sigma_angstrom: f32,
    epsilon_kcal: f32,
}

#[derive(Clone, Debug)]
pub struct FfxmlTopologySummary {
    pub net_charge_e: f32,
}

fn xml_attr<'input>(node: Node<'input, 'input>, name: &str) -> PackResult<&'input str> {
    node.attribute(name).ok_or_else(|| {
        PackError::Invalid(format!(
            "ffxml element '{}' is missing required attribute '{}'",
            node.tag_name().name(),
            name
        ))
    })
}

fn xml_opt_attr<'input>(node: Node<'input, 'input>, name: &str) -> Option<&'input str> {
    node.attribute(name)
}

fn xml_attr_f32<'input>(node: Node<'input, 'input>, name: &str) -> PackResult<f32> {
    let value = xml_attr(node, name)?;
    value.parse::<f32>().map_err(|_| {
        PackError::Invalid(format!(
            "ffxml attribute '{}.{}' must be numeric",
            node.tag_name().name(),
            name
        ))
    })
}

fn parse_ffxml_selector<'a, 'input>(node: Node<'a, 'input>, idx: usize) -> FfxmlAtomSelector {
    let type_attr = format!("type{idx}");
    if let Some(value) = node.attribute(type_attr.as_str()) {
        return if value.trim().is_empty() || value.trim() == "*" {
            FfxmlAtomSelector::Any
        } else {
            FfxmlAtomSelector::Type(value.trim().to_string())
        };
    }
    let class_attr = format!("class{idx}");
    if let Some(value) = node.attribute(class_attr.as_str()) {
        return if value.trim().is_empty() || value.trim() == "*" {
            FfxmlAtomSelector::Any
        } else {
            FfxmlAtomSelector::Class(value.trim().to_string())
        };
    }
    FfxmlAtomSelector::Any
}

fn ffxml_selector_matches(selector: &FfxmlAtomSelector, atom: &FfxmlAssignedAtom) -> bool {
    match selector {
        FfxmlAtomSelector::Any => true,
        FfxmlAtomSelector::Type(value) => atom.type_name == *value,
        FfxmlAtomSelector::Class(value) => atom.class_name == *value,
    }
}

fn ffxml_selector_specificity(selector: &FfxmlAtomSelector) -> usize {
    match selector {
        FfxmlAtomSelector::Any => 0,
        FfxmlAtomSelector::Class(_) => 1,
        FfxmlAtomSelector::Type(_) => 2,
    }
}

fn parse_ffxml_forcefield(ffxml_path: &Path) -> PackResult<FfxmlForcefield> {
    let text = std::fs::read_to_string(ffxml_path)?;
    let doc = Document::parse(&text)
        .map_err(|err| PackError::Invalid(format!("failed to parse ffxml: {err}")))?;
    let root = doc.root_element();
    if root.tag_name().name() != "ForceField" {
        return Err(PackError::Invalid(
            "ffxml root element must be <ForceField>".into(),
        ));
    }

    let mut atom_types = BTreeMap::new();
    let mut residues = BTreeMap::new();
    let mut bond_params = Vec::new();
    let mut angle_params = Vec::new();
    let mut proper_torsions = Vec::new();
    let mut improper_torsions = Vec::new();
    let mut nonbonded_by_type = BTreeMap::new();
    let mut nonbonded_by_class = BTreeMap::new();
    let mut nonbonded_charge_from_residue = false;
    let mut scee_scale = 1.2f32;
    let mut scnb_scale = 2.0f32;

    for section in root.children().filter(|node| node.is_element()) {
        match section.tag_name().name() {
            "Info" => {}
            "AtomTypes" => {
                for node in section.children().filter(|node| node.is_element()) {
                    if node.tag_name().name() != "Type" {
                        return Err(PackError::Invalid(format!(
                            "unsupported ffxml AtomTypes entry '{}'",
                            node.tag_name().name()
                        )));
                    }
                    let name = xml_attr(node, "name")?.trim().to_string();
                    atom_types.insert(
                        name.clone(),
                        FfxmlAtomTypeDef {
                            class_name: xml_opt_attr(node, "class")
                                .unwrap_or("")
                                .trim()
                                .to_string(),
                            element: xml_attr(node, "element")?.trim().to_string(),
                            mass: xml_attr_f32(node, "mass")?,
                        },
                    );
                }
            }
            "Residues" => {
                for residue in section.children().filter(|node| node.is_element()) {
                    if residue.tag_name().name() != "Residue" {
                        return Err(PackError::Invalid(format!(
                            "unsupported ffxml Residues entry '{}'",
                            residue.tag_name().name()
                        )));
                    }
                    let mut atoms = BTreeMap::new();
                    for node in residue.children().filter(|node| node.is_element()) {
                        match node.tag_name().name() {
                            "Atom" => {
                                let name = xml_attr(node, "name")?.trim().to_string();
                                atoms.insert(
                                    name.clone(),
                                    FfxmlResidueAtomDef {
                                        name,
                                        type_name: xml_attr(node, "type")?.trim().to_string(),
                                        charge_e: xml_opt_attr(node, "charge")
                                            .map(|value| value.parse::<f32>())
                                            .transpose()
                                            .map_err(|_| {
                                                PackError::Invalid(
                                                    "ffxml residue atom charge must be numeric"
                                                        .into(),
                                                )
                                            })?,
                                    },
                                );
                            }
                            "Bond" | "ExternalBond" | "AllowPatch" => {}
                            other => {
                                return Err(PackError::Invalid(format!(
                                    "unsupported ffxml Residue entry '{}'",
                                    other
                                )))
                            }
                        }
                    }
                    let name = xml_attr(residue, "name")?.trim().to_string();
                    residues.insert(name.clone(), FfxmlResidueDef { atoms });
                }
            }
            "Patches" => {}
            "HarmonicBondForce" => {
                for node in section.children().filter(|node| node.is_element()) {
                    if node.tag_name().name() != "Bond" {
                        return Err(PackError::Invalid(format!(
                            "unsupported ffxml HarmonicBondForce entry '{}'",
                            node.tag_name().name()
                        )));
                    }
                    bond_params.push(FfxmlBondParam {
                        left: parse_ffxml_selector(node, 1),
                        right: parse_ffxml_selector(node, 2),
                        length_angstrom: xml_attr_f32(node, "length")? * 10.0,
                        force_constant: xml_attr_f32(node, "k")? * 0.239_005_74 / 100.0,
                    });
                }
            }
            "HarmonicAngleForce" => {
                for node in section.children().filter(|node| node.is_element()) {
                    if node.tag_name().name() != "Angle" {
                        return Err(PackError::Invalid(format!(
                            "unsupported ffxml HarmonicAngleForce entry '{}'",
                            node.tag_name().name()
                        )));
                    }
                    angle_params.push(FfxmlAngleParam {
                        left: parse_ffxml_selector(node, 1),
                        center: parse_ffxml_selector(node, 2),
                        right: parse_ffxml_selector(node, 3),
                        theta0_rad: xml_attr_f32(node, "angle")?,
                        force_constant: xml_attr_f32(node, "k")? * 0.239_005_74,
                    });
                }
            }
            "PeriodicTorsionForce" => {
                for node in section.children().filter(|node| node.is_element()) {
                    let target = match node.tag_name().name() {
                        "Proper" => &mut proper_torsions,
                        "Improper" => &mut improper_torsions,
                        other => {
                            return Err(PackError::Invalid(format!(
                                "unsupported ffxml PeriodicTorsionForce entry '{}'",
                                other
                            )))
                        }
                    };
                    let mut terms = Vec::new();
                    for idx in 1..=6 {
                        let periodicity_key = format!("periodicity{idx}");
                        let Some(periodicity) = node.attribute(periodicity_key.as_str()) else {
                            continue;
                        };
                        let phase_key = format!("phase{idx}");
                        let force_key = format!("k{idx}");
                        terms.push(FfxmlTorsionTerm {
                            periodicity: periodicity.parse::<f32>().map_err(|_| {
                                PackError::Invalid(
                                    "ffxml torsion periodicity must be numeric".into(),
                                )
                            })?,
                            phase_rad: xml_attr(node, phase_key.as_str())?.parse::<f32>().map_err(
                                |_| {
                                    PackError::Invalid("ffxml torsion phase must be numeric".into())
                                },
                            )?,
                            force_constant: xml_attr(node, force_key.as_str())?
                                .parse::<f32>()
                                .map_err(|_| {
                                    PackError::Invalid("ffxml torsion k must be numeric".into())
                                })?
                                * 0.239_005_74,
                        });
                    }
                    if terms.is_empty() {
                        return Err(PackError::Invalid(
                            "ffxml torsion entry must provide at least one periodicity/phase/k term"
                                .into(),
                        ));
                    }
                    target.push(FfxmlTorsionParam {
                        atoms: [
                            parse_ffxml_selector(node, 1),
                            parse_ffxml_selector(node, 2),
                            parse_ffxml_selector(node, 3),
                            parse_ffxml_selector(node, 4),
                        ],
                        terms,
                    });
                }
            }
            "NonbondedForce" => {
                if let Some(value) = xml_opt_attr(section, "coulomb14scale") {
                    let parsed = value.parse::<f32>().map_err(|_| {
                        PackError::Invalid(
                            "ffxml NonbondedForce coulomb14scale must be numeric".into(),
                        )
                    })?;
                    if parsed > 0.0 {
                        scee_scale = 1.0 / parsed;
                    }
                }
                if let Some(value) = xml_opt_attr(section, "lj14scale") {
                    let parsed = value.parse::<f32>().map_err(|_| {
                        PackError::Invalid("ffxml NonbondedForce lj14scale must be numeric".into())
                    })?;
                    if parsed > 0.0 {
                        scnb_scale = 1.0 / parsed;
                    }
                }
                for node in section.children().filter(|node| node.is_element()) {
                    match node.tag_name().name() {
                        "UseAttributeFromResidue" => {
                            let attr_name = xml_attr(node, "name")?;
                            if attr_name == "charge" {
                                nonbonded_charge_from_residue = true;
                            } else {
                                return Err(PackError::Invalid(format!(
                                    "unsupported ffxml UseAttributeFromResidue '{}'",
                                    attr_name
                                )));
                            }
                            continue;
                        }
                        "Atom" => {}
                        other => {
                            return Err(PackError::Invalid(format!(
                                "unsupported ffxml NonbondedForce entry '{}'",
                                other
                            )));
                        }
                    }
                    let param = FfxmlNonbondedParam {
                        charge_e: xml_opt_attr(node, "charge")
                            .map(|value| {
                                value.parse::<f32>().map_err(|_| {
                                    PackError::Invalid(
                                        "ffxml NonbondedForce atom charge must be numeric".into(),
                                    )
                                })
                            })
                            .transpose()?,
                        sigma_angstrom: xml_attr_f32(node, "sigma")? * 10.0,
                        epsilon_kcal: xml_attr_f32(node, "epsilon")? * 0.239_005_74,
                    };
                    if let Some(value) = xml_opt_attr(node, "type") {
                        nonbonded_by_type.insert(value.trim().to_string(), param.clone());
                    }
                    if let Some(value) = xml_opt_attr(node, "class") {
                        nonbonded_by_class.insert(value.trim().to_string(), param.clone());
                    }
                }
            }
            other => {
                return Err(PackError::Invalid(format!(
                    "unsupported ffxml section '{}'",
                    other
                )))
            }
        }
    }

    Ok(FfxmlForcefield {
        atom_types,
        residues,
        bond_params,
        angle_params,
        proper_torsions,
        improper_torsions,
        nonbonded_by_type,
        nonbonded_by_class,
        nonbonded_charge_from_residue,
        scee_scale,
        scnb_scale,
    })
}

fn ffxml_assign_training_atom(
    forcefield: &FfxmlForcefield,
    residue_name: &str,
    atom_name: &str,
    override_charge_e: Option<f32>,
) -> PackResult<FfxmlAssignedAtom> {
    let residue = forcefield.residues.get(residue_name).ok_or_else(|| {
        PackError::Invalid(format!(
            "ffxml is missing residue template '{}'",
            residue_name
        ))
    })?;
    let atom = residue.atoms.get(atom_name.trim()).ok_or_else(|| {
        PackError::Invalid(format!(
            "ffxml residue '{}' is missing atom '{}'",
            residue_name, atom_name
        ))
    })?;
    let atom_type = forcefield.atom_types.get(&atom.type_name).ok_or_else(|| {
        PackError::Invalid(format!(
            "ffxml atom type '{}' referenced by '{}:{}' is missing",
            atom.type_name, residue_name, atom.name
        ))
    })?;
    let nonbonded = forcefield
        .nonbonded_by_type
        .get(&atom.type_name)
        .or_else(|| {
            if atom_type.class_name.is_empty() {
                None
            } else {
                forcefield.nonbonded_by_class.get(&atom_type.class_name)
            }
        })
        .ok_or_else(|| {
            PackError::Invalid(format!(
                "ffxml nonbonded parameters are missing for atom type '{}'",
                atom.type_name
            ))
        })?;
    Ok(FfxmlAssignedAtom {
        type_name: atom.type_name.clone(),
        class_name: atom_type.class_name.clone(),
        element: atom_type.element.clone(),
        mass: atom_type.mass,
        charge_e: override_charge_e
            .or(atom.charge_e)
            .or(nonbonded.charge_e)
            .ok_or_else(|| {
                if forcefield.nonbonded_charge_from_residue {
                    PackError::Invalid(format!(
                        "ffxml residue '{}' atom '{}' is missing charge data",
                        residue_name, atom.name
                    ))
                } else {
                    PackError::Invalid(format!(
                        "ffxml nonbonded parameters are missing charge for atom type '{}'",
                        atom.type_name
                    ))
                }
            })?,
        sigma_angstrom: nonbonded.sigma_angstrom,
        epsilon_kcal: nonbonded.epsilon_kcal,
    })
}

fn ffxml_match_bond_param<'a>(
    params: &'a [FfxmlBondParam],
    left: &FfxmlAssignedAtom,
    right: &FfxmlAssignedAtom,
) -> Option<&'a FfxmlBondParam> {
    params
        .iter()
        .filter(|param| {
            (ffxml_selector_matches(&param.left, left)
                && ffxml_selector_matches(&param.right, right))
                || (ffxml_selector_matches(&param.left, right)
                    && ffxml_selector_matches(&param.right, left))
        })
        .max_by_key(|param| {
            ffxml_selector_specificity(&param.left) + ffxml_selector_specificity(&param.right)
        })
}

fn ffxml_match_angle_param<'a>(
    params: &'a [FfxmlAngleParam],
    left: &FfxmlAssignedAtom,
    center: &FfxmlAssignedAtom,
    right: &FfxmlAssignedAtom,
) -> Option<&'a FfxmlAngleParam> {
    params
        .iter()
        .filter(|param| {
            (ffxml_selector_matches(&param.left, left)
                && ffxml_selector_matches(&param.center, center)
                && ffxml_selector_matches(&param.right, right))
                || (ffxml_selector_matches(&param.left, right)
                    && ffxml_selector_matches(&param.center, center)
                    && ffxml_selector_matches(&param.right, left))
        })
        .max_by_key(|param| {
            ffxml_selector_specificity(&param.left)
                + ffxml_selector_specificity(&param.center)
                + ffxml_selector_specificity(&param.right)
        })
}

fn ffxml_match_torsion_param<'a>(
    params: &'a [FfxmlTorsionParam],
    atoms: [&FfxmlAssignedAtom; 4],
    improper: bool,
) -> Option<&'a FfxmlTorsionParam> {
    params
        .iter()
        .filter(|param| {
            let forward = param
                .atoms
                .iter()
                .zip(atoms.iter())
                .all(|(selector, atom)| ffxml_selector_matches(selector, atom));
            if forward {
                return true;
            }
            if improper {
                let reverse = [&atoms[0], &atoms[1], &atoms[3], &atoms[2]];
                return param
                    .atoms
                    .iter()
                    .zip(reverse.iter())
                    .all(|(selector, atom)| ffxml_selector_matches(selector, atom));
            }
            let reverse = [&atoms[3], &atoms[2], &atoms[1], &atoms[0]];
            param
                .atoms
                .iter()
                .zip(reverse.iter())
                .all(|(selector, atom)| ffxml_selector_matches(selector, atom))
        })
        .max_by_key(|param| {
            param
                .atoms
                .iter()
                .map(ffxml_selector_specificity)
                .sum::<usize>()
        })
}

fn build_nonbonded_tables_from_sigma_epsilon(
    params: &[(f32, f32)],
) -> (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut nonbonded_index = Vec::with_capacity(params.len() * params.len());
    let mut acoef = Vec::with_capacity(params.len() * params.len());
    let mut bcoef = Vec::with_capacity(params.len() * params.len());
    for &(sigma_left, epsilon_left) in params {
        for &(sigma_right, epsilon_right) in params {
            nonbonded_index.push(acoef.len() + 1);
            let sigma = 0.5 * (sigma_left + sigma_right);
            let epsilon = (epsilon_left * epsilon_right).sqrt().max(1.0e-6);
            let rmin = (sigma * 2.0_f32.powf(1.0 / 6.0)).max(1.0e-3);
            let b = 2.0 * epsilon * rmin.powi(6);
            let a = epsilon * rmin.powi(12);
            acoef.push(a);
            bcoef.push(b);
        }
    }
    (nonbonded_index, acoef.clone(), bcoef.clone(), acoef, bcoef)
}

fn validate_ffxml_forcefield_support(
    training_structure_path: &Path,
    source_charge_manifest_path: Option<&Path>,
    ffxml_path: &Path,
    graph_node_specs: &[GraphNodeSpec],
) -> PackResult<()> {
    let forcefield = parse_ffxml_forcefield(ffxml_path)?;
    let templates = load_training_templates(training_structure_path)?;
    let template_charge_map = if let Some(path) = source_charge_manifest_path {
        build_template_charge_map_from_manifest(training_structure_path, path)?
    } else {
        BTreeMap::new()
    };
    let required_templates = graph_node_specs
        .iter()
        .map(|spec| spec.template_resname.clone())
        .collect::<BTreeSet<_>>();
    for template in templates
        .iter()
        .filter(|template| required_templates.contains(&template.resname))
    {
        for atom in &template.atoms {
            ffxml_assign_training_atom(
                &forcefield,
                &template.resname,
                atom.name.trim(),
                template_charge_map
                    .get(&(template.resname.clone(), atom.name.trim().to_string()))
                    .copied(),
            )?;
        }
    }
    Ok(())
}

pub fn assess_training_source(
    training_structure_path: &Path,
    source_charge_manifest_path: Option<&Path>,
    source_topology_path: Option<&Path>,
    source_forcefield_path: Option<&Path>,
    graph_node_specs: &[GraphNodeSpec],
    token_junctions: &BTreeMap<String, TokenJunctionSpec>,
) -> PackResult<TrainingSourceAssessment> {
    let molecule = read_molecule(training_structure_path, None, false, true, None)?;
    let templates = group_residues(molecule.clone())?;
    let bonds = dedup_bonds(if molecule.bonds.is_empty() {
        infer_global_bonds(&molecule.atoms)
    } else {
        molecule.bonds.clone()
    });
    let thresholds = collect_training_thresholds(
        &molecule.atoms,
        &bonds,
        graph_node_specs,
        token_junctions,
        &templates,
    );
    let quality = training_source_quality(&thresholds);
    let mut reasons = Vec::new();
    if thresholds.impossible_valence_count > 0 {
        reasons.push("impossible_valence_detected".into());
    }
    if !thresholds.junction_consistency_ok {
        reasons.push("junction_geometry_inconsistent".into());
    }
    if thresholds
        .min_nonbonded_distance
        .map(|value| value < 0.60)
        .unwrap_or(false)
    {
        reasons.push("severe_nonbonded_overlap".into());
    } else if thresholds
        .min_nonbonded_distance
        .map(|value| value < 0.90)
        .unwrap_or(false)
    {
        reasons.push("tight_nonbonded_contacts".into());
    }
    if thresholds
        .max_local_bond_ratio
        .map(|value| value > 1.30)
        .unwrap_or(false)
        || thresholds
            .max_local_bond_delta
            .map(|value| value > 0.25)
            .unwrap_or(false)
    {
        reasons.push("severely_stretched_local_bonds".into());
    } else if thresholds
        .max_local_bond_ratio
        .map(|value| value > 1.15)
        .unwrap_or(false)
        || thresholds
            .max_local_bond_delta
            .map(|value| value > 0.10)
            .unwrap_or(false)
    {
        reasons.push("stretched_local_bonds".into());
    }
    if thresholds.ambiguous_assignment_count > 0 {
        reasons.push("heuristic_assignment_required".into());
    }

    let mut parameter_source = match quality {
        TrainingSourceQuality::Trusted | TrainingSourceQuality::Risky => {
            ParameterSourceChoice::SyntheticPdb
        }
        TrainingSourceQuality::Unreliable => ParameterSourceChoice::Rejected,
    };
    if quality == TrainingSourceQuality::Unreliable {
        let topology_available = source_topology_path
            .filter(|path| path.exists())
            .and_then(|path| path.extension().and_then(|value| value.to_str()))
            .map(|ext| ext.eq_ignore_ascii_case("prmtop"))
            .unwrap_or(false);
        if topology_available {
            parameter_source = ParameterSourceChoice::SourceTopologyRef;
            reasons.push("switched_to_source_topology_ref".into());
        } else if let Some(ffxml_path) = source_forcefield_path.filter(|path| path.exists()) {
            match validate_ffxml_forcefield_support(
                training_structure_path,
                source_charge_manifest_path,
                ffxml_path,
                graph_node_specs,
            ) {
                Ok(()) => {
                    parameter_source = ParameterSourceChoice::ForcefieldRef;
                    reasons.push("switched_to_forcefield_ref".into());
                }
                Err(err) => {
                    reasons.push("ffxml_unsupported_feature".into());
                    reasons.push(format!("ffxml_validation_failed:{err}"));
                }
            }
        } else {
            reasons.push("no_strong_parameter_source_available".into());
        }
    }

    Ok(TrainingSourceAssessment {
        quality: training_source_quality_label(quality),
        parameter_source: parameter_source_label(parameter_source),
        reasons,
        metrics: TrainingSourceMetrics {
            min_nonbonded_distance_angstrom: thresholds.min_nonbonded_distance,
            max_local_bond_ratio: thresholds.max_local_bond_ratio,
            impossible_valence_count: thresholds.impossible_valence_count,
            ambiguous_assignment_count: thresholds.ambiguous_assignment_count,
            junction_consistency_ok: thresholds.junction_consistency_ok,
        },
    })
}

fn group6_atomic_number(atomic_number: i32) -> bool {
    matches!(atomic_number, 8 | 16 | 34 | 52 | 84)
}

fn guess_bond_order(
    atoms: &[AtomRecord],
    adjacency: &[Vec<usize>],
    typings: &[SyntheticAtomTyping],
    left: usize,
    right: usize,
) -> f32 {
    let Some(atom_left) = atoms.get(left) else {
        return 1.0;
    };
    let Some(atom_right) = atoms.get(right) else {
        return 1.0;
    };
    if atom_left.resid != atom_right.resid {
        return 1.0;
    }
    let element_left =
        normalize_element(&atom_left.element).unwrap_or_else(|| atom_left.element.clone());
    let element_right =
        normalize_element(&atom_right.element).unwrap_or_else(|| atom_right.element.clone());
    if matches!(element_left.as_str(), "H" | "F" | "Cl" | "Br" | "I")
        || matches!(element_right.as_str(), "H" | "F" | "Cl" | "Br" | "I")
    {
        return 1.0;
    }
    let distance = atom_left.position.sub(atom_right.position).norm();
    let left_geom = infer_geometry_class(atoms, adjacency, left);
    let right_geom = infer_geometry_class(atoms, adjacency, right);
    if matches!(
        (element_left.as_str(), element_right.as_str()),
        ("C", "C") | ("C", "N") | ("N", "C") | ("N", "N")
    ) && distance <= 1.24
    {
        return 3.0;
    }
    if matches!(
        (element_left.as_str(), element_right.as_str()),
        ("C", "O") | ("O", "C") | ("C", "N") | ("N", "C") | ("N", "O") | ("O", "N")
    ) && distance <= 1.34
    {
        return 2.0;
    }
    if matches!(
        (element_left.as_str(), element_right.as_str()),
        ("S", "O") | ("O", "S") | ("P", "O") | ("O", "P")
    ) && distance <= 1.58
    {
        return 2.0;
    }
    if left_geom == AtomGeometryClass::Sp2
        && right_geom == AtomGeometryClass::Sp2
        && distance <= 1.47
        && typings
            .get(left)
            .map(|item| item.params.label.starts_with('C') || item.params.label.starts_with('N'))
            .unwrap_or(false)
        && typings
            .get(right)
            .map(|item| item.params.label.starts_with('C') || item.params.label.starts_with('N'))
            .unwrap_or(false)
    {
        return 1.5;
    }
    if left_geom != AtomGeometryClass::Sp3
        && right_geom != AtomGeometryClass::Sp3
        && distance
            <= covalent_radius_angstrom(&element_left) + covalent_radius_angstrom(&element_right)
                - 0.08
    {
        return 2.0;
    }
    1.0
}

fn uff_bond_rest_length(order: f32, left: UffAtomParams, right: UffAtomParams) -> f32 {
    let r_bo = -0.1332 * (left.r1 + right.r1) * order.max(1.0e-3).ln();
    let sqrt_xi_left = left.xi.sqrt();
    let sqrt_xi_right = right.xi.sqrt();
    let r_en = left.r1 * right.r1 * (sqrt_xi_left - sqrt_xi_right).powi(2)
        / ((left.xi * left.r1) + (right.xi * right.r1)).max(1.0e-6);
    left.r1 + right.r1 + r_bo - r_en
}

fn uff_bond_force_constant(rest_length: f32, left: UffAtomParams, right: UffAtomParams) -> f32 {
    2.0 * 332.06 * left.z1 * right.z1 / rest_length.max(1.0e-4).powi(3)
}

fn approx_bond_length_from_training(
    bond_map: &BTreeMap<(String, String, String), f32>,
    template_resname: &str,
    atom_left: &str,
    atom_right: &str,
) -> Option<f32> {
    let left = atom_left.trim().to_string();
    let right = atom_right.trim().to_string();
    let (a, b) = if left <= right {
        (left, right)
    } else {
        (right, left)
    };
    bond_map.get(&(template_resname.to_string(), a, b)).copied()
}

fn uff_angle_force_constant(
    theta0_rad: f32,
    bond_order_left: f32,
    bond_order_right: f32,
    left: UffAtomParams,
    center: UffAtomParams,
    right: UffAtomParams,
) -> f32 {
    let r12 = uff_bond_rest_length(bond_order_left, left, center);
    let r23 = uff_bond_rest_length(bond_order_right, center, right);
    let cos_theta0 = theta0_rad.cos();
    let r13 = (r12 * r12 + r23 * r23 - 2.0 * r12 * r23 * cos_theta0)
        .max(1.0e-6)
        .sqrt();
    let beta = 2.0 * 332.06 / (r12 * r23).max(1.0e-6);
    let pre_factor = beta * left.z1 * right.z1 / r13.powi(5).max(1.0e-6);
    let r_term = r12 * r23;
    let inner = 3.0 * r_term * (1.0 - cos_theta0 * cos_theta0) - r13 * r13 * cos_theta0;
    (pre_factor * r_term * inner).max(1.0)
}

fn synthetic_improper_force_constant(
    center_atomic_number: i32,
    carbon_bound_to_oxygen: bool,
) -> f32 {
    match center_atomic_number {
        6 | 7 | 8 => {
            if carbon_bound_to_oxygen && center_atomic_number == 6 {
                50.0 / 3.0
            } else {
                6.0 / 3.0
            }
        }
        15 | 33 | 51 | 83 => 22.0 / 3.0,
        _ => 2.0,
    }
}

fn synthetic_torsion_params(
    atoms: &[AtomRecord],
    adjacency: &[Vec<usize>],
    typings: &[SyntheticAtomTyping],
    bond_orders: &BTreeMap<(usize, usize), f32>,
    dihedral: [usize; 4],
) -> SyntheticDihedralParam {
    let central_order = bond_orders
        .get(&ordered_pair(dihedral[1], dihedral[2]))
        .copied()
        .unwrap_or(1.0);
    let type_b = typings
        .get(dihedral[1])
        .map(|item| item.params)
        .unwrap_or_else(|| uff_params_for_label("C_3"));
    let type_c = typings
        .get(dihedral[2])
        .map(|item| item.params)
        .unwrap_or_else(|| uff_params_for_label("C_3"));
    let geom_b = infer_geometry_class(atoms, adjacency, dihedral[1]);
    let geom_c = infer_geometry_class(atoms, adjacency, dihedral[2]);
    if geom_b == AtomGeometryClass::Sp3 && geom_c == AtomGeometryClass::Sp3 {
        let atomic_b = atomic_number_for_element(&atoms[dihedral[1]].element);
        let atomic_c = atomic_number_for_element(&atoms[dihedral[2]].element);
        if (central_order - 1.0).abs() <= 1.0e-3
            && group6_atomic_number(atomic_b)
            && group6_atomic_number(atomic_c)
        {
            let v2: f32 = if atomic_b == 8 { 2.0 } else { 6.8 };
            let v3: f32 = if atomic_c == 8 { 2.0 } else { 6.8 };
            return SyntheticDihedralParam {
                force_constant: (v2 * v3).sqrt(),
                periodicity: 2.0,
                phase_rad: 0.0,
            };
        }
        return SyntheticDihedralParam {
            force_constant: (type_b.v1 * type_c.v1).sqrt().max(0.5),
            periodicity: 3.0,
            phase_rad: 0.0,
        };
    }
    if geom_b == AtomGeometryClass::Sp2 && geom_c == AtomGeometryClass::Sp2 {
        let force_constant =
            5.0 * (type_b.u1 * type_c.u1).sqrt() * (1.0 + 4.18 * central_order.max(1.0e-3).ln());
        return SyntheticDihedralParam {
            force_constant: force_constant.max(1.0),
            periodicity: 2.0,
            phase_rad: std::f32::consts::PI,
        };
    }
    let atomic_b = atomic_number_for_element(&atoms[dihedral[1]].element);
    let atomic_c = atomic_number_for_element(&atoms[dihedral[2]].element);
    let end_sp2 = infer_geometry_class(atoms, adjacency, dihedral[0]) == AtomGeometryClass::Sp2
        || infer_geometry_class(atoms, adjacency, dihedral[3]) == AtomGeometryClass::Sp2;
    if (central_order - 1.0).abs() <= 1.0e-3
        && ((geom_b == AtomGeometryClass::Sp3
            && group6_atomic_number(atomic_b)
            && geom_c == AtomGeometryClass::Sp2
            && !group6_atomic_number(atomic_c))
            || (geom_c == AtomGeometryClass::Sp3
                && group6_atomic_number(atomic_c)
                && geom_b == AtomGeometryClass::Sp2
                && !group6_atomic_number(atomic_b)))
    {
        let force_constant =
            5.0 * (type_b.u1 * type_c.u1).sqrt() * (1.0 + 4.18 * central_order.max(1.0e-3).ln());
        return SyntheticDihedralParam {
            force_constant: force_constant.max(1.0),
            periodicity: 2.0,
            phase_rad: 0.0,
        };
    }
    if (central_order - 1.0).abs() <= 1.0e-3 && end_sp2 {
        return SyntheticDihedralParam {
            force_constant: 2.0,
            periodicity: 3.0,
            phase_rad: std::f32::consts::PI,
        };
    }
    SyntheticDihedralParam {
        force_constant: 1.0,
        periodicity: 6.0,
        phase_rad: std::f32::consts::PI,
    }
}

fn build_nonbonded_tables(
    unique_types: &[UffAtomParams],
) -> (Vec<usize>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut nonbonded_index = Vec::with_capacity(unique_types.len() * unique_types.len());
    let mut acoef = Vec::with_capacity(unique_types.len() * unique_types.len());
    let mut bcoef = Vec::with_capacity(unique_types.len() * unique_types.len());
    for left in unique_types {
        for right in unique_types {
            nonbonded_index.push(acoef.len() + 1);
            let rmin = (left.x1 * right.x1).sqrt().max(1.0e-3);
            let epsilon = (left.d1 * right.d1).sqrt().max(1.0e-4);
            let b = 2.0 * epsilon * rmin.powi(6);
            let a = epsilon * rmin.powi(12);
            acoef.push(a);
            bcoef.push(b);
        }
    }
    (nonbonded_index, acoef.clone(), bcoef.clone(), acoef, bcoef)
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn guessed_attach_bond_distance(
    parent_template: &ResidueTemplate,
    child_template: &ResidueTemplate,
    parent_attach_atom: &str,
    child_attach_atom: &str,
    bond_order: u8,
) -> PackResult<f32> {
    let parent_atom = template_atom_by_name(parent_template, parent_attach_atom)?;
    let child_atom = template_atom_by_name(child_template, child_attach_atom)?;
    let mut guess = covalent_radius_angstrom(&parent_atom.element)
        + covalent_radius_angstrom(&child_atom.element);
    if bond_order > 1 {
        guess -= 0.10 * (bond_order.saturating_sub(1) as f32);
    }
    Ok(guess.clamp(0.9, 2.2))
}

fn observed_attach_distance(
    templates: &[ResidueTemplate],
    parent_resname: &str,
    child_resname: &str,
    parent_attach_atom: &str,
    child_attach_atom: &str,
) -> Option<f32> {
    templates.windows(2).find_map(|window| {
        let [parent, child] = window else {
            return None;
        };
        if parent.resname != parent_resname || child.resname != child_resname {
            return None;
        }
        let parent_atom = parent
            .atoms
            .iter()
            .find(|atom| atom.name.trim() == parent_attach_atom.trim())?;
        let child_atom = child
            .atoms
            .iter()
            .find(|atom| atom.name.trim() == child_attach_atom.trim())?;
        Some(child_atom.position.sub(parent_atom.position).norm())
    })
}

fn orthonormal_basis(axis: Vec3) -> (Vec3, Vec3) {
    let axis = normalize(axis);
    let tangent = if axis.x.abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u = normalize(axis.cross(tangent));
    let v = normalize(axis.cross(u));
    (u, v)
}

fn aligned_zigzag_direction(axis: Vec3, step_idx: usize, torsion_angle: f32) -> Vec3 {
    let axis = normalize(axis);
    let (u, v) = orthonormal_basis(axis);
    let phase = if step_idx % 2 == 0 {
        torsion_angle
    } else {
        std::f32::consts::PI + torsion_angle
    };
    normalize(
        axis.scale(0.816_496_6)
            .add(u.scale(0.577_350_26 * phase.cos()))
            .add(v.scale(0.577_350_26 * phase.sin())),
    )
}

fn preferred_branch_direction(
    parent_dir: Vec3,
    sibling_idx: usize,
    sibling_count: usize,
    depth: usize,
    branch_spread: &str,
    torsion_angle: f32,
) -> Vec3 {
    if sibling_count <= 1 {
        return rotate_about_axis(normalize(parent_dir), parent_dir, torsion_angle);
    }
    let axis = normalize(parent_dir);
    let (u, v) = orthonormal_basis(axis);
    let phase = (2.0 * std::f32::consts::PI * sibling_idx as f32) / sibling_count as f32
        + (depth as f32 * 0.37);
    let spread = match branch_spread {
        "staggered" => {
            if depth == 0 {
                0.92
            } else {
                0.78
            }
        }
        _ => {
            if depth == 0 {
                0.80
            } else {
                0.68
            }
        }
    };
    let base = normalize(
        axis.scale(0.55)
            .add(u.scale(spread * phase.cos()))
            .add(v.scale(spread * phase.sin())),
    );
    rotate_about_axis(base, axis, torsion_angle)
}

fn template_collision_radius(template: &ResidueTemplate) -> f32 {
    template
        .atoms
        .iter()
        .filter(|atom| heavy_atom(&atom.element))
        .map(|atom| atom.position.sub(template.centroid).norm())
        .fold(0.0f32, f32::max)
        .max(0.65)
        .min(4.0)
}

fn envelope_cell_size(radii: &[f32], clearance: f32) -> f32 {
    let max_radius = radii.iter().copied().fold(0.65f32, f32::max);
    (2.0 * max_radius + clearance).max(1.0)
}

struct ResidueEnvelopeIndex {
    hash: SpatialHash,
}

impl ResidueEnvelopeIndex {
    fn new(radii: &[f32], clearance: f32, expected_items: usize) -> Self {
        Self {
            hash: SpatialHash::with_capacity(
                envelope_cell_size(radii, clearance),
                expected_items.saturating_mul(2).max(1),
            ),
        }
    }

    fn insert(&mut self, idx: usize, position: Vec3) {
        self.hash.insert(idx, position);
    }

    fn remove(&mut self, idx: usize, position: Vec3) {
        self.hash.remove(idx, position);
    }

    fn clearance_against_options(
        &self,
        candidate: Vec3,
        existing: &[Option<Vec3>],
        parent: Option<usize>,
        candidate_radius: f32,
        existing_radii: &[f32],
        clearance: f32,
    ) -> f32 {
        let mut best = f32::INFINITY;
        self.hash.for_each_neighbor(candidate, |idx| {
            let Some(prev) = existing.get(idx).and_then(|item| *item) else {
                return;
            };
            if Some(idx) == parent {
                return;
            }
            let required =
                candidate_radius + existing_radii.get(idx).copied().unwrap_or(0.65) + clearance;
            best = best.min(candidate.sub(prev).norm() - required);
        });
        if best.is_finite() {
            best
        } else {
            f32::INFINITY
        }
    }

    fn linear_clearance(
        &self,
        candidate: Vec3,
        centroids: &[Vec3],
        candidate_idx: usize,
        candidate_radius: f32,
        radii: &[f32],
        clearance: f32,
    ) -> f32 {
        let mut best = f32::INFINITY;
        self.hash.for_each_neighbor(candidate, |idx| {
            if idx + 2 >= candidate_idx {
                return;
            }
            let Some(prev) = centroids.get(idx).copied() else {
                return;
            };
            let near_chain_scale = if idx + 3 >= candidate_idx { 0.86 } else { 1.0 };
            let required = (candidate_radius + radii.get(idx).copied().unwrap_or(0.65) + clearance)
                * near_chain_scale;
            best = best.min(candidate.sub(prev).norm() - required);
        });
        if best.is_finite() {
            best
        } else {
            f32::INFINITY
        }
    }
}

fn apply_centroid_repulsion_with_hash(positions: &[Vec3], deltas: &mut [Vec3], min_distance: f32) {
    let mut index = SpatialHash::with_capacity(min_distance.max(1.0e-6), positions.len() * 2);
    for (idx, position) in positions.iter().copied().enumerate() {
        index.insert(idx, position);
    }
    for left in 0..positions.len() {
        index.for_each_neighbor(positions[left], |right| {
            if right <= left {
                return;
            }
            let diff = positions[right].sub(positions[left]);
            let dist = diff.norm().max(1.0e-4);
            if dist >= min_distance {
                return;
            }
            let dir = diff.scale(1.0 / dist);
            let push = dir.scale(0.08 * (min_distance - dist));
            deltas[left] = deltas[left].sub(push);
            deltas[right] = deltas[right].add(push);
        });
    }
}

fn perturb_direction(rng: &mut StdRng, preferred: Vec3, width: f32) -> Vec3 {
    let (u, v) = orthonormal_basis(preferred);
    normalize(
        preferred
            .scale(1.0)
            .add(u.scale(rng.gen_range(-width..width)))
            .add(v.scale(rng.gen_range(-width..width))),
    )
}

fn resolved_torsion_angle(edge: &GraphEdgeSpec, rng: &mut StdRng) -> f32 {
    match edge.torsion_mode.as_str() {
        "cis" => 0.0,
        "gauche_plus" => 60.0f32.to_radians(),
        "gauche_minus" => -60.0f32.to_radians(),
        "fixed_deg" => edge.torsion_deg.unwrap_or(180.0).to_radians(),
        "sample_window" => {
            let [lo, hi] = edge.torsion_window_deg.unwrap_or([-30.0, 30.0]);
            rng.gen_range(lo.min(hi)..=lo.max(hi)).to_radians()
        }
        _ => edge.torsion_deg.unwrap_or(180.0).to_radians(),
    }
}

fn graph_tree(
    node_count: usize,
    edge_specs: &[GraphEdgeSpec],
    root_idx: usize,
) -> PackResult<(Vec<Option<usize>>, Vec<Option<usize>>, Vec<usize>)> {
    if node_count == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let mut adjacency = vec![Vec::<(usize, usize)>::new(); node_count];
    for (edge_idx, edge) in edge_specs.iter().enumerate() {
        if edge.parent >= node_count || edge.child >= node_count {
            return Err(PackError::Invalid(
                "polymer graph edge references invalid node".into(),
            ));
        }
        adjacency[edge.parent].push((edge.child, edge_idx));
        adjacency[edge.child].push((edge.parent, edge_idx));
    }
    let mut parent_of = vec![None; node_count];
    let mut parent_edge = vec![None; node_count];
    let mut visit_order = Vec::with_capacity(node_count);
    let mut queue = VecDeque::from([root_idx]);
    let mut seen = BTreeSet::from([root_idx]);
    while let Some(node_idx) = queue.pop_front() {
        visit_order.push(node_idx);
        for (neighbor, edge_idx) in adjacency[node_idx].clone() {
            if seen.insert(neighbor) {
                parent_of[neighbor] = Some(node_idx);
                parent_edge[neighbor] = Some(edge_idx);
                queue.push_back(neighbor);
            }
        }
    }
    if seen.len() != node_count {
        return Err(PackError::Invalid("polymer graph must be connected".into()));
    }
    Ok((parent_of, parent_edge, visit_order))
}

fn is_descendant(parent_of: &[Option<usize>], candidate: usize, ancestor: usize) -> bool {
    if candidate == ancestor {
        return true;
    }
    let mut current = parent_of[candidate];
    while let Some(node_idx) = current {
        if node_idx == ancestor {
            return true;
        }
        current = parent_of[node_idx];
    }
    false
}

#[derive(Clone, Debug)]
struct RotatableEdge {
    parent_attach_idx: usize,
    child_attach_idx: usize,
    movable_atom_indices: Vec<usize>,
    torsion_mode: String,
    torsion_deg: Option<f32>,
    torsion_window_deg: Option<[f32; 2]>,
}

fn solver_score(report: &BuildQcReport, output: &PackOutput) -> f32 {
    let severe_bond_penalty = report.severe_bond_violations.len() as f32 * 1_000_000.0;
    let severe_clash_penalty = report.severe_nonbonded_clash_count as f32 * 100_000.0;
    let bond_distance_penalty = report
        .severe_bond_violations
        .iter()
        .map(|item| (item.measured_distance_angstrom - item.expected_distance_angstrom).abs())
        .sum::<f32>()
        * 1_000.0;
    let clash_distance_penalty = report
        .min_nonbonded_distance_angstrom
        .map(|value| {
            if value < 1.2 {
                (1.2 - value) * 500.0
            } else {
                0.0
            }
        })
        .unwrap_or(0.0);
    let compactness = output
        .atoms
        .iter()
        .map(|atom| atom.position.norm())
        .sum::<f32>()
        / output.atoms.len().max(1) as f32;
    severe_bond_penalty
        + severe_clash_penalty
        + bond_distance_penalty
        + clash_distance_penalty
        + compactness * 0.01
}

fn candidate_rotation_degrees(edge: &RotatableEdge) -> Vec<f32> {
    let mut values = match edge.torsion_mode.as_str() {
        "trans" => vec![0.0, 180.0, 120.0, -120.0, 60.0, -60.0],
        "cis" => vec![0.0, 30.0, -30.0, 60.0, -60.0, 180.0],
        "gauche_plus" => vec![0.0, 60.0, 90.0, 120.0, -60.0],
        "gauche_minus" => vec![0.0, -60.0, -90.0, -120.0, 60.0],
        "fixed_deg" => {
            let base = edge.torsion_deg.unwrap_or(0.0);
            vec![0.0, base, base + 30.0, base - 30.0]
        }
        "sample_window" => {
            let [start, end] = edge.torsion_window_deg.unwrap_or([-180.0, 180.0]);
            let steps = 7usize;
            let mut samples = vec![0.0];
            if steps <= 1 || (end - start).abs() <= 1.0e-6 {
                samples.push(start);
            } else {
                for idx in 0..steps {
                    let fraction = idx as f32 / (steps - 1) as f32;
                    samples.push(start + (end - start) * fraction);
                }
            }
            samples
        }
        _ => vec![0.0, 180.0, 60.0, -60.0, 120.0, -120.0],
    };
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup_by(|left, right| (*left - *right).abs() <= 1.0e-4);
    values
}

fn rotate_subtree_atoms(
    atoms: &mut [AtomRecord],
    edge: &RotatableEdge,
    delta_deg: f32,
) -> PackResult<()> {
    if delta_deg.abs() <= 1.0e-6 {
        return Ok(());
    }
    let pivot = atoms
        .get(edge.parent_attach_idx)
        .map(|atom| atom.position)
        .ok_or_else(|| PackError::Invalid("solver parent attach atom index out of range".into()))?;
    let child = atoms
        .get(edge.child_attach_idx)
        .map(|atom| atom.position)
        .ok_or_else(|| PackError::Invalid("solver child attach atom index out of range".into()))?;
    let axis = child.sub(pivot);
    if axis.norm() <= 1.0e-6 {
        return Ok(());
    }
    let theta = delta_deg.to_radians();
    for atom_idx in &edge.movable_atom_indices {
        let Some(atom) = atoms.get_mut(*atom_idx) else {
            continue;
        };
        let relative = atom.position.sub(pivot);
        atom.position = pivot.add(rotate_about_axis(relative, axis, theta));
    }
    Ok(())
}

fn solve_rotatable_edges(
    output: &mut PackOutput,
    qc_context: &BuildQcContext,
    rotatable_edges: &[RotatableEdge],
    seed: u64,
) -> PackResult<BuildSolverReport> {
    let passes_requested = 3usize;
    let mut passes_executed = 0usize;
    let mut candidate_evaluations = 0usize;
    let mut accepted_moves = 0usize;
    let mut best_report = recompute_build_qc_report(output, qc_context);
    let mut best_score = solver_score(&best_report, output);
    let mut termination_reason = "pass_budget_exhausted".to_string();
    if best_report.severe_bond_violations.is_empty()
        && best_report.severe_nonbonded_clash_count == 0
    {
        return Ok(BuildSolverReport {
            enabled: true,
            mode: "torsion_solve".into(),
            seed,
            passes_requested,
            passes_executed: 0,
            rotatable_edge_count: rotatable_edges.len(),
            candidate_evaluations: 0,
            accepted_moves: 0,
            termination_reason: "qc_already_passed".into(),
            best_score,
            hard_fail_reason: None,
        });
    }
    for pass_idx in 0..passes_requested {
        let mut improved_any = false;
        for edge in rotatable_edges {
            let baseline_atoms = output.atoms.clone();
            let mut edge_best_atoms = baseline_atoms.clone();
            let mut edge_best_report = best_report.clone();
            let mut edge_best_score = best_score;
            for delta_deg in candidate_rotation_degrees(edge) {
                let mut candidate_atoms = baseline_atoms.clone();
                rotate_subtree_atoms(&mut candidate_atoms, edge, delta_deg)?;
                let candidate_output = PackOutput {
                    atoms: candidate_atoms,
                    bonds: output.bonds.clone(),
                    box_size: output.box_size,
                    ter_after: output.ter_after.clone(),
                    box_vectors: output.box_vectors,
                };
                let report = recompute_build_qc_report(&candidate_output, qc_context);
                let score = solver_score(&report, &candidate_output);
                candidate_evaluations += 1;
                if score + 1.0e-4 < edge_best_score {
                    edge_best_score = score;
                    edge_best_report = report;
                    edge_best_atoms = candidate_output.atoms;
                }
            }
            if edge_best_score + 1.0e-4 < best_score {
                output.atoms = edge_best_atoms;
                best_score = edge_best_score;
                best_report = edge_best_report;
                accepted_moves += 1;
                improved_any = true;
            }
        }
        passes_executed = pass_idx + 1;
        if best_report.severe_bond_violations.is_empty()
            && best_report.severe_nonbonded_clash_count == 0
        {
            termination_reason = "qc_passed".into();
            break;
        }
        if !improved_any {
            termination_reason = "no_improvement".into();
            break;
        }
    }
    Ok(BuildSolverReport {
        enabled: true,
        mode: "torsion_solve".into(),
        seed,
        passes_requested,
        passes_executed,
        rotatable_edge_count: rotatable_edges.len(),
        candidate_evaluations,
        accepted_moves,
        termination_reason,
        best_score,
        hard_fail_reason: None,
    })
}

fn graph_edge_ideal_distance(
    edge: &GraphEdgeSpec,
    templates: &[ResidueTemplate],
    node_specs: &[GraphNodeSpec],
    template_lookup: &BTreeMap<String, &ResidueTemplate>,
    repeat: &ResidueTemplate,
) -> PackResult<f32> {
    let parent_template = template_lookup
        .get(&node_specs[edge.parent].template_resname)
        .copied()
        .unwrap_or(repeat);
    let child_template = template_lookup
        .get(&node_specs[edge.child].template_resname)
        .copied()
        .unwrap_or(repeat);
    let guessed = guessed_attach_bond_distance(
        parent_template,
        child_template,
        &edge.parent_attach_atom,
        &edge.child_attach_atom,
        edge.bond_order,
    )?;
    if let Some(observed) = observed_attach_distance(
        templates,
        &parent_template.resname,
        &child_template.resname,
        &edge.parent_attach_atom,
        &edge.child_attach_atom,
    ) {
        if observed >= guessed * 0.75 && observed <= guessed * 1.25 {
            return Ok(observed.max(0.5));
        }
    }
    Ok(guessed.max(0.5))
}

fn severe_bond_threshold(expected_distance: f32) -> f32 {
    (expected_distance * 1.8).max(expected_distance + 2.0)
}

const BUILD_QC_NEIGHBOR_CUTOFF_ANGSTROM: f32 = 4.0;

fn build_qc_excluded_pairs(bonds: &[(usize, usize)]) -> BTreeSet<(usize, usize)> {
    let mut adjacency = BTreeMap::<usize, Vec<usize>>::new();
    let mut excluded = BTreeSet::new();
    for &(a, b) in bonds {
        let pair = if a <= b { (a, b) } else { (b, a) };
        excluded.insert(pair);
        adjacency.entry(a).or_default().push(b);
        adjacency.entry(b).or_default().push(a);
    }
    for neighbors in adjacency.values() {
        for left_idx in 0..neighbors.len() {
            for right_idx in (left_idx + 1)..neighbors.len() {
                let left = neighbors[left_idx];
                let right = neighbors[right_idx];
                let pair = if left <= right {
                    (left, right)
                } else {
                    (right, left)
                };
                excluded.insert(pair);
            }
        }
    }
    excluded
}

fn build_qc_report(
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
    context: &BuildQcContext,
) -> BuildQcReport {
    let severe_bond_violations = context
        .bond_expectations
        .iter()
        .filter_map(|expectation| {
            let parent = atoms.get(expectation.parent_idx)?;
            let child = atoms.get(expectation.child_idx)?;
            let measured = child.position.sub(parent.position).norm();
            (measured > severe_bond_threshold(expectation.expected_distance_angstrom)).then(|| {
                BondQcViolation {
                    edge_id: expectation.edge_id.clone(),
                    parent_resid: expectation.parent_resid,
                    child_resid: expectation.child_resid,
                    parent_atom: expectation.parent_atom.clone(),
                    child_atom: expectation.child_atom.clone(),
                    measured_distance_angstrom: measured,
                    expected_distance_angstrom: expectation.expected_distance_angstrom,
                }
            })
        })
        .collect::<Vec<_>>();
    let excluded_pairs = build_qc_excluded_pairs(bonds);
    let mut min_nonbonded: Option<f32> = None;
    let mut severe_clash_examples = Vec::new();
    let mut severe_clash_count = 0usize;
    let possible_nonbonded_count = atoms.len().saturating_mul(atoms.len().saturating_sub(1)) / 2;
    let has_nonbonded_pairs = possible_nonbonded_count > excluded_pairs.len();
    let mut spatial =
        SpatialHash::with_capacity(BUILD_QC_NEIGHBOR_CUTOFF_ANGSTROM, atoms.len() * 2);
    for (idx, atom) in atoms.iter().enumerate() {
        spatial.insert(idx, atom.position);
    }
    for left in 0..atoms.len() {
        spatial.for_each_neighbor(atoms[left].position, |right| {
            if right <= left {
                return;
            }
            if excluded_pairs.contains(&(left, right)) {
                return;
            }
            let distance = atoms[right].position.sub(atoms[left].position).norm();
            min_nonbonded = Some(match min_nonbonded {
                Some(current) => current.min(distance),
                None => distance,
            });
            if distance < 0.8 {
                severe_clash_count += 1;
                if severe_clash_examples.len() < 8 {
                    severe_clash_examples.push(ClashQcViolation {
                        atom_a: left + 1,
                        atom_b: right + 1,
                        distance_angstrom: distance,
                    });
                }
            }
        });
    }
    if min_nonbonded.is_none() && has_nonbonded_pairs {
        min_nonbonded = Some(BUILD_QC_NEIGHBOR_CUTOFF_ANGSTROM);
    }

    BuildQcReport {
        inter_residue_bond_count: context.inter_residue_bond_count,
        terminal_connectivity_consistent: context.terminal_connectivity_consistent,
        sequence_token_template_consistent: context.sequence_token_template_consistent,
        min_nonbonded_distance_angstrom: min_nonbonded,
        severe_nonbonded_clash_count: severe_clash_count,
        severe_bond_violations,
        severe_clash_examples,
    }
}

pub fn recompute_build_qc_report(output: &PackOutput, context: &BuildQcContext) -> BuildQcReport {
    build_qc_report(&output.atoms, &output.bonds, context)
}

pub fn ensure_build_qc_passes(report: &BuildQcReport) -> PackResult<()> {
    if !report.severe_bond_violations.is_empty() {
        let first = &report.severe_bond_violations[0];
        return Err(PackError::Invalid(format!(
            "build QC failed: inter-residue bond '{}' measured {:.3} A; expected {:.3} A",
            first.edge_id, first.measured_distance_angstrom, first.expected_distance_angstrom
        )));
    }
    if report.severe_nonbonded_clash_count > 0 {
        let first = &report.severe_clash_examples[0];
        return Err(PackError::Invalid(format!(
            "build QC failed: severe nonbonded clash between atoms {} and {} at {:.3} A",
            first.atom_a, first.atom_b, first.distance_angstrom
        )));
    }
    Ok(())
}

fn layout_graph_centroids(
    node_count: usize,
    edge_specs: &[GraphEdgeSpec],
    root_idx: usize,
    step_length: f32,
    conformation_mode: &str,
    source_axis: Vec3,
    seed: u64,
    collision_radii: &[f32],
) -> PackResult<Vec<Vec3>> {
    if node_count == 0 {
        return Err(PackError::Invalid(
            "polymer graph requires at least one node".into(),
        ));
    }
    if root_idx >= node_count {
        return Err(PackError::Invalid(
            "polymer graph root references invalid node".into(),
        ));
    }
    let mut adjacency = vec![Vec::<usize>::new(); node_count];
    let mut edge_lookup = BTreeMap::<(usize, usize), usize>::new();
    for (edge_idx, edge) in edge_specs.iter().enumerate() {
        if edge.parent >= node_count || edge.child >= node_count || edge.parent == edge.child {
            return Err(PackError::Invalid(
                "polymer graph edge references invalid node".into(),
            ));
        }
        edge_lookup.insert((edge.parent, edge.child), edge_idx);
        edge_lookup.insert((edge.child, edge.parent), edge_idx);
        adjacency[edge.parent].push(edge.child);
        adjacency[edge.child].push(edge.parent);
    }
    for items in &mut adjacency {
        items.sort_unstable();
        items.dedup();
    }
    let mut parent_of = vec![None; node_count];
    let mut depth_of = vec![0usize; node_count];
    let mut visit_order = Vec::with_capacity(node_count);
    let mut queue = VecDeque::from([root_idx]);
    let mut seen = BTreeSet::from([root_idx]);
    while let Some(node_idx) = queue.pop_front() {
        visit_order.push(node_idx);
        for neighbor in adjacency[node_idx].clone() {
            if seen.insert(neighbor) {
                parent_of[neighbor] = Some(node_idx);
                depth_of[neighbor] = depth_of[node_idx] + 1;
                queue.push_back(neighbor);
            }
        }
    }
    if seen.len() != node_count {
        return Err(PackError::Invalid("polymer graph must be connected".into()));
    }
    let mut children_by_parent = vec![Vec::<usize>::new(); node_count];
    for (node_idx, parent) in parent_of.iter().enumerate() {
        if let Some(parent_idx) = parent {
            children_by_parent[*parent_idx].push(node_idx);
        }
    }
    let mut positions = vec![None; node_count];
    let mut incoming_dirs = vec![normalize(source_axis); node_count];
    positions[root_idx] = Some(Vec3::new(0.0, 0.0, 0.0));
    let mut envelope_index = ResidueEnvelopeIndex::new(collision_radii, 0.30, node_count);
    envelope_index.insert(root_idx, Vec3::new(0.0, 0.0, 0.0));
    let mut rng = StdRng::seed_from_u64(seed ^ 0xB4A7_CE11);
    for node_idx in visit_order {
        let depth = depth_of[node_idx];
        let parent_pos = positions[node_idx].unwrap_or(Vec3::new(0.0, 0.0, 0.0));
        let parent_dir = incoming_dirs[node_idx];
        let children = children_by_parent[node_idx].clone();
        for (sibling_idx, child_idx) in children.iter().enumerate() {
            if positions[*child_idx].is_some() {
                continue;
            }
            let edge = edge_lookup
                .get(&(node_idx, *child_idx))
                .and_then(|idx| edge_specs.get(*idx))
                .ok_or_else(|| {
                    PackError::Invalid("polymer graph layout edge lookup failed".into())
                })?;
            let torsion_angle = resolved_torsion_angle(edge, &mut rng);
            let preferred = if conformation_mode == "extended" && children.len() == 1 {
                aligned_zigzag_direction(source_axis, depth, torsion_angle)
            } else {
                preferred_branch_direction(
                    parent_dir,
                    sibling_idx,
                    children.len(),
                    depth,
                    &edge.branch_spread,
                    torsion_angle,
                )
            };
            let child_radius = collision_radii.get(*child_idx).copied().unwrap_or(0.65);
            let mut chosen = None;
            let mut best_candidate = None;
            let mut best_clearance = f32::NEG_INFINITY;
            let attempts = if conformation_mode == "random_walk" {
                2048
            } else {
                256
            };
            for attempt in 0..attempts {
                let width = if conformation_mode == "random_walk" {
                    if attempt < 512 {
                        0.45
                    } else if attempt < 1536 {
                        0.85
                    } else {
                        1.35
                    }
                } else if attempt == 0 {
                    0.0
                } else if attempt < 96 {
                    0.35
                } else {
                    0.70
                };
                let dir = if width <= 1.0e-6 {
                    preferred
                } else {
                    perturb_direction(&mut rng, preferred, width)
                };
                let candidate = parent_pos.add(dir.scale(step_length));
                let clearance = envelope_index.clearance_against_options(
                    candidate,
                    &positions,
                    Some(node_idx),
                    child_radius,
                    collision_radii,
                    0.30,
                );
                if clearance > best_clearance {
                    best_clearance = clearance;
                    best_candidate = Some((dir, candidate));
                }
                if clearance >= 0.0 {
                    chosen = Some((dir, candidate));
                    break;
                }
            }
            let (chosen_dir, chosen_pos) = chosen
                .or_else(|| best_candidate.filter(|_| best_clearance >= -0.10))
                .ok_or_else(|| {
                    PackError::Invalid(format!(
                        "polymer graph self-avoiding placement failed for node {}",
                        child_idx
                    ))
                })?;
            incoming_dirs[*child_idx] = chosen_dir;
            positions[*child_idx] = Some(chosen_pos);
            envelope_index.insert(*child_idx, chosen_pos);
        }
    }
    let mut positions = positions
        .into_iter()
        .map(|item| {
            item.ok_or_else(|| {
                PackError::Invalid("polymer graph layout left unplaced nodes".into())
            })
        })
        .collect::<PackResult<Vec<_>>>()?;
    let min_distance = (step_length * 0.68).max(1.4);
    for _ in 0..96 {
        let mut deltas = vec![Vec3::new(0.0, 0.0, 0.0); node_count];
        for edge in edge_specs {
            let diff = positions[edge.child].sub(positions[edge.parent]);
            let dist = diff.norm().max(1.0e-4);
            let dir = diff.scale(1.0 / dist);
            let stretch = dist - step_length;
            let correction = dir.scale(0.16 * stretch);
            deltas[edge.parent] = deltas[edge.parent].add(correction);
            deltas[edge.child] = deltas[edge.child].sub(correction);
        }
        apply_centroid_repulsion_with_hash(&positions, &mut deltas, min_distance);
        deltas[root_idx] = Vec3::new(0.0, 0.0, 0.0);
        for idx in 0..node_count {
            if idx == root_idx {
                continue;
            }
            positions[idx] = positions[idx].add(deltas[idx]);
        }
        let root_shift = positions[root_idx];
        for pos in &mut positions {
            *pos = pos.sub(root_shift);
        }
    }
    Ok(positions)
}

fn atom_name_set(names: &[String]) -> BTreeSet<String> {
    names.iter().map(|name| name.trim().to_string()).collect()
}

pub fn resolve_polymer_param_source(
    artifact: &str,
    explicit_charge_manifest: Option<&str>,
    explicit_topology_ref: Option<&str>,
) -> PackResult<PolymerSourceResolved> {
    let artifact_path = PathBuf::from(artifact);

    if artifact_path.is_file() {
        let ext = artifact_path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext == "json" {
            let payload = std::fs::read_to_string(&artifact_path)?;
            let bundle: PolymerParamBundle =
                serde_json::from_str(&payload).map_err(|err| PackError::Parse(err.to_string()))?;
            if bundle.version != SOURCE_BUNDLE_VERSION
                && bundle.version != LEGACY_POLYMER_PARAM_VERSION
            {
                return Err(PackError::Invalid(format!(
                    "unsupported polymer param bundle version '{}'",
                    bundle.version
                )));
            }
            let base = artifact_path.parent().unwrap_or_else(|| Path::new("."));
            return Ok(PolymerSourceResolved {
                training_structure_path: base.join(bundle.training_structure),
            });
        }
        if matches!(ext.as_str(), "pdb" | "cif" | "mmcif" | "pdbx") {
            return Ok(PolymerSourceResolved {
                training_structure_path: artifact_path.clone(),
            });
        }
        return Err(PackError::Invalid(
            "polymer param artifact must be a .json bundle or a .pdb/.cif/.mmcif/.pdbx training structure".into(),
        ));
    }

    if artifact_path.is_dir() {
        let candidates = [
            artifact_path.join("polymer_param.json"),
            artifact_path.join("polymer_source.json"),
            artifact_path.join("training_oligomer.pdb"),
            artifact_path.join("training_oligomer.cif"),
            artifact_path.join("training_oligomer.mmcif"),
            artifact_path.join("training_oligomer.pdbx"),
            artifact_path.join("oligomer.pdb"),
            artifact_path.join("oligomer.cif"),
            artifact_path.join("oligomer.mmcif"),
            artifact_path.join("oligomer.pdbx"),
            artifact_path.join("source.pdb"),
            artifact_path.join("source.cif"),
            artifact_path.join("source.mmcif"),
            artifact_path.join("source.pdbx"),
        ];
        for candidate in candidates {
            if candidate.exists() {
                return resolve_polymer_param_source(
                    candidate.to_string_lossy().as_ref(),
                    explicit_charge_manifest,
                    explicit_topology_ref,
                );
            }
        }
        return Err(PackError::Invalid(format!(
            "could not resolve polymer param artifact from directory '{}'",
            artifact
        )));
    }

    Err(PackError::Invalid(format!(
        "polymer param artifact does not exist: {}",
        artifact
    )))
}

pub fn load_charge_manifest(path: &Path) -> PackResult<ChargeManifest> {
    let payload = std::fs::read_to_string(path)?;
    let manifest: ChargeManifest =
        serde_json::from_str(&payload).map_err(|err| PackError::Parse(err.to_string()))?;
    if manifest.schema_version != CHARGE_MANIFEST_VERSION
        && manifest.schema_version != LEGACY_CHARGE_MANIFEST_VERSION
    {
        return Err(PackError::Invalid(format!(
            "unsupported charge manifest version '{}'",
            manifest.schema_version
        )));
    }
    Ok(manifest)
}

#[allow(dead_code)]
pub fn charge_manifest_field_kinds(manifest: &ChargeManifest) -> Vec<String> {
    let mut kinds = Vec::new();
    if manifest.net_charge_e.is_some() {
        kinds.push("net_charge_e".to_string());
    }
    if manifest.atom_charges.is_some() {
        kinds.push("atom_charges".to_string());
    }
    if manifest.head_charge_e.is_some()
        || manifest.repeat_charge_e.is_some()
        || manifest.tail_charge_e.is_some()
    {
        kinds.push("repeat_scalars".to_string());
    }
    kinds
}

#[allow(dead_code)]
pub fn compute_solute_net_charge(manifest: &ChargeManifest) -> NetChargeEstimate {
    NetChargeEstimate {
        net_charge_e: manifest.net_charge_e,
        source: manifest
            .net_charge_e
            .map(|_| "charge_manifest.net_charge_e".to_string()),
    }
}

fn compute_homopolymer_net_charge(
    manifest: &ChargeManifest,
    templates: &[ResidueTemplate],
    target_n_repeat: usize,
    source_nmer: usize,
) -> NetChargeEstimate {
    if let Some(atom_charges) = &manifest.atom_charges {
        let mut residue_charges = vec![0.0f32; templates.len()];
        let mut running = 1usize;
        let mut offsets = Vec::with_capacity(templates.len());
        for residue in templates {
            let start = running;
            running += residue.atoms.len();
            offsets.push((start, running - 1));
        }
        for charge in atom_charges {
            for (idx, (start, end)) in offsets.iter().enumerate() {
                if charge.index >= *start && charge.index <= *end {
                    residue_charges[idx] += charge.charge_e;
                    break;
                }
            }
        }
        let head = residue_charges[0];
        let repeat = residue_charges[templates.len() / 2];
        let tail = residue_charges[templates.len() - 1];
        let total = if target_n_repeat == 1 {
            repeat
        } else if target_n_repeat == 2 {
            head + tail
        } else {
            head + tail + repeat * (target_n_repeat.saturating_sub(2) as f32)
        };
        return NetChargeEstimate {
            net_charge_e: Some(total),
            source: Some("charge_manifest.atom_charges.scaled_from_training_oligomer".into()),
        };
    }

    if let (Some(head), Some(repeat), Some(tail)) = (
        manifest.head_charge_e,
        manifest.repeat_charge_e,
        manifest.tail_charge_e,
    ) {
        let total = if target_n_repeat == 1 {
            repeat
        } else if target_n_repeat == 2 {
            head + tail
        } else {
            head + tail + repeat * (target_n_repeat.saturating_sub(2) as f32)
        };
        return NetChargeEstimate {
            net_charge_e: Some(total),
            source: Some("charge_manifest.repeat_scalars".into()),
        };
    }

    if source_nmer == target_n_repeat {
        return NetChargeEstimate {
            net_charge_e: manifest.net_charge_e,
            source: manifest
                .net_charge_e
                .map(|_| "charge_manifest.net_charge_e".to_string()),
        };
    }

    NetChargeEstimate {
        net_charge_e: None,
        source: None,
    }
}

fn compute_sequence_polymer_net_charge(
    manifest: &ChargeManifest,
    templates: &[ResidueTemplate],
    template_sequence_resnames: &[String],
    source_nmer: usize,
) -> NetChargeEstimate {
    if let Some(atom_charges) = &manifest.atom_charges {
        let charge_map = template_charge_map(
            templates,
            &atom_charges
                .iter()
                .map(|item| item.charge_e)
                .collect::<Vec<_>>(),
        );
        if let Some(total) = sum_sequence_template_charges(&charge_map, template_sequence_resnames)
        {
            return NetChargeEstimate {
                net_charge_e: Some(total),
                source: Some("charge_manifest.atom_charges.sequence_template_map".into()),
            };
        }
    }

    if let (Some(head_charge), Some(repeat_charge), Some(tail_charge)) = (
        manifest.head_charge_e,
        manifest.repeat_charge_e,
        manifest.tail_charge_e,
    ) {
        let head_template = templates.first().map(|template| template.resname.as_str());
        let repeat_template = templates
            .get(templates.len() / 2)
            .map(|template| template.resname.as_str());
        let tail_template = templates.last().map(|template| template.resname.as_str());
        let mut total = 0.0f32;
        let mut recognized = true;
        for (idx, resname) in template_sequence_resnames.iter().enumerate() {
            let charge = if idx == 0 && Some(resname.as_str()) == head_template {
                head_charge
            } else if idx + 1 == template_sequence_resnames.len()
                && Some(resname.as_str()) == tail_template
            {
                tail_charge
            } else if Some(resname.as_str()) == repeat_template {
                repeat_charge
            } else {
                recognized = false;
                break;
            };
            total += charge;
        }
        if recognized {
            return NetChargeEstimate {
                net_charge_e: Some(total),
                source: Some("charge_manifest.repeat_scalars.sequence_roles".into()),
            };
        }
    }

    let unique_templates = template_sequence_resnames
        .iter()
        .collect::<std::collections::BTreeSet<_>>();
    if unique_templates.len() == 1 {
        return compute_homopolymer_net_charge(
            manifest,
            templates,
            template_sequence_resnames.len(),
            source_nmer,
        );
    }

    if source_nmer == template_sequence_resnames.len() {
        return NetChargeEstimate {
            net_charge_e: manifest.net_charge_e,
            source: manifest
                .net_charge_e
                .map(|_| "charge_manifest.net_charge_e".to_string()),
        };
    }

    NetChargeEstimate {
        net_charge_e: None,
        source: None,
    }
}

#[allow(dead_code)]
pub fn compute_polymer_net_charge_from_source(
    manifest: &ChargeManifest,
    training_structure_path: &Path,
    target_n_repeat: usize,
    source_nmer: usize,
) -> PackResult<NetChargeEstimate> {
    let templates = load_training_templates(training_structure_path)?;
    Ok(compute_homopolymer_net_charge(
        manifest,
        &templates,
        target_n_repeat,
        source_nmer,
    ))
}

pub fn compute_sequence_polymer_net_charge_from_source(
    manifest: &ChargeManifest,
    training_structure_path: &Path,
    template_sequence_resnames: &[String],
    source_nmer: usize,
) -> PackResult<NetChargeEstimate> {
    let templates = load_training_templates(training_structure_path)?;
    Ok(compute_sequence_polymer_net_charge(
        manifest,
        &templates,
        template_sequence_resnames,
        source_nmer,
    ))
}

#[allow(dead_code)]
pub fn compute_solute_net_charge_from_prmtop(path: &Path) -> PackResult<NetChargeEstimate> {
    Ok(NetChargeEstimate {
        net_charge_e: Some(read_prmtop_total_charge(path)?),
        source: Some("prmtop.total_charge".into()),
    })
}

pub fn compute_sequence_polymer_net_charge_from_prmtop(
    training_structure_path: &Path,
    template_sequence_resnames: &[String],
    prmtop_path: &Path,
    source_nmer: usize,
) -> PackResult<NetChargeEstimate> {
    let templates = load_training_templates(training_structure_path)?;
    let atom_charges = read_prmtop_atom_charges(prmtop_path)?;
    let charge_map = template_charge_map(&templates, &atom_charges);
    if let Some(total) = sum_sequence_template_charges(&charge_map, template_sequence_resnames) {
        return Ok(NetChargeEstimate {
            net_charge_e: Some(total),
            source: Some("prmtop.atom_charges.sequence_template_map".into()),
        });
    }
    if source_nmer == template_sequence_resnames.len() {
        return Ok(NetChargeEstimate {
            net_charge_e: Some(atom_charges.iter().sum()),
            source: Some("prmtop.total_charge".into()),
        });
    }
    Ok(NetChargeEstimate {
        net_charge_e: None,
        source: None,
    })
}

fn residue_blocks(topology: &AmberTopology) -> Vec<AmberTopology> {
    let mut blocks = Vec::new();
    for (idx, start) in topology.residue_pointers.iter().enumerate() {
        let start_idx = start.saturating_sub(1);
        let end_idx = topology
            .residue_pointers
            .get(idx + 1)
            .copied()
            .unwrap_or(topology.atom_names.len() + 1)
            .saturating_sub(1);
        blocks.push(AmberTopology {
            atom_names: topology.atom_names[start_idx..end_idx].to_vec(),
            residue_labels: vec![topology
                .residue_labels
                .get(idx)
                .cloned()
                .unwrap_or_else(|| "MOL".to_string())],
            residue_pointers: vec![1],
            atomic_numbers: topology.atomic_numbers[start_idx..end_idx].to_vec(),
            masses: (start_idx..end_idx)
                .map(|atom_idx| topology.masses.get(atom_idx).copied().unwrap_or(12.0))
                .collect(),
            charges: (start_idx..end_idx)
                .map(|atom_idx| topology.charges.get(atom_idx).copied().unwrap_or(0.0))
                .collect(),
            atom_type_indices: (start_idx..end_idx)
                .map(|atom_idx| {
                    topology
                        .atom_type_indices
                        .get(atom_idx)
                        .copied()
                        .unwrap_or(atom_idx + 1)
                })
                .collect(),
            amber_atom_types: (start_idx..end_idx)
                .map(|atom_idx| {
                    topology
                        .amber_atom_types
                        .get(atom_idx)
                        .cloned()
                        .unwrap_or_else(|| "X".into())
                })
                .collect(),
            radii: (start_idx..end_idx)
                .map(|atom_idx| topology.radii.get(atom_idx).copied().unwrap_or(1.5))
                .collect(),
            screen: (start_idx..end_idx)
                .map(|atom_idx| topology.screen.get(atom_idx).copied().unwrap_or(0.8))
                .collect(),
            bonds: topology
                .bonds
                .iter()
                .enumerate()
                .filter_map(|(bond_idx, &(a, b))| {
                    if a >= start_idx && a < end_idx && b >= start_idx && b < end_idx {
                        Some(((a - start_idx, b - start_idx), bond_idx))
                    } else {
                        None
                    }
                })
                .map(|(bond, _)| bond)
                .collect(),
            bond_type_indices: topology
                .bonds
                .iter()
                .enumerate()
                .filter_map(|(bond_idx, &(a, b))| {
                    if a >= start_idx && a < end_idx && b >= start_idx && b < end_idx {
                        Some(
                            topology
                                .bond_type_indices
                                .get(bond_idx)
                                .copied()
                                .unwrap_or(1),
                        )
                    } else {
                        None
                    }
                })
                .collect(),
            bond_force_constants: topology.bond_force_constants.clone(),
            bond_equil_values: topology.bond_equil_values.clone(),
            angles: Vec::new(),
            angle_type_indices: Vec::new(),
            angle_force_constants: topology.angle_force_constants.clone(),
            angle_equil_values: topology.angle_equil_values.clone(),
            dihedrals: Vec::new(),
            dihedral_type_indices: Vec::new(),
            dihedral_force_constants: topology.dihedral_force_constants.clone(),
            dihedral_periodicities: topology.dihedral_periodicities.clone(),
            dihedral_phases: topology.dihedral_phases.clone(),
            scee_scale_factors: topology.scee_scale_factors.clone(),
            scnb_scale_factors: topology.scnb_scale_factors.clone(),
            solty: topology.solty.clone(),
            impropers: Vec::new(),
            improper_type_indices: Vec::new(),
            excluded_atoms: Vec::new(),
            nonbonded_parm_index: topology.nonbonded_parm_index.clone(),
            lennard_jones_acoef: topology.lennard_jones_acoef.clone(),
            lennard_jones_bcoef: topology.lennard_jones_bcoef.clone(),
            lennard_jones_14_acoef: topology.lennard_jones_14_acoef.clone(),
            lennard_jones_14_bcoef: topology.lennard_jones_14_bcoef.clone(),
            hbond_acoef: topology.hbond_acoef.clone(),
            hbond_bcoef: topology.hbond_bcoef.clone(),
            hbcut: topology.hbcut.clone(),
            tree_chain_classification: (start_idx..end_idx)
                .map(|atom_idx| {
                    topology
                        .tree_chain_classification
                        .get(atom_idx)
                        .cloned()
                        .unwrap_or_else(|| "M".into())
                })
                .collect(),
            join_array: (start_idx..end_idx)
                .map(|atom_idx| topology.join_array.get(atom_idx).copied().unwrap_or(0))
                .collect(),
            irotat: (start_idx..end_idx)
                .map(|atom_idx| topology.irotat.get(atom_idx).copied().unwrap_or(0))
                .collect(),
            solvent_pointers: topology.solvent_pointers.clone(),
            atoms_per_molecule: topology.atoms_per_molecule.clone(),
            box_dimensions: topology.box_dimensions.clone(),
            radius_set: topology.radius_set.clone(),
            ipol: topology.ipol,
        });
    }
    blocks
}

#[derive(Default)]
struct RebuiltTermAssignments {
    bonds: Vec<(usize, usize)>,
    bond_type_indices: Vec<usize>,
    angles: Vec<[usize; 3]>,
    angle_type_indices: Vec<usize>,
    dihedrals: Vec<[usize; 4]>,
    dihedral_type_indices: Vec<usize>,
    impropers: Vec<[usize; 4]>,
    improper_type_indices: Vec<usize>,
}

fn rebuild_bond_adjacency(atom_count: usize, bonds: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); atom_count];
    for &(a, b) in bonds {
        if let Some(items) = adjacency.get_mut(a) {
            items.push(b);
        }
        if let Some(items) = adjacency.get_mut(b) {
            items.push(a);
        }
    }
    for items in &mut adjacency {
        items.sort_unstable();
        items.dedup();
    }
    adjacency
}

fn rebuild_angles(adjacency: &[Vec<usize>]) -> Vec<[usize; 3]> {
    let mut angles = Vec::new();
    for (center, neighbors) in adjacency.iter().enumerate() {
        for left_idx in 0..neighbors.len() {
            for right_idx in (left_idx + 1)..neighbors.len() {
                angles.push([neighbors[left_idx], center, neighbors[right_idx]]);
            }
        }
    }
    angles
}

fn rebuild_dihedrals(adjacency: &[Vec<usize>], bonds: &[(usize, usize)]) -> Vec<[usize; 4]> {
    let mut dihedrals = Vec::new();
    for &(b, c) in bonds {
        for &a in adjacency.get(b).unwrap_or(&Vec::new()) {
            if a == c {
                continue;
            }
            for &d in adjacency.get(c).unwrap_or(&Vec::new()) {
                if d == b || d == a {
                    continue;
                }
                dihedrals.push([a, b, c, d]);
            }
        }
    }
    dihedrals.sort_unstable();
    dihedrals.dedup();
    dihedrals
}

fn rebuild_impropers(adjacency: &[Vec<usize>]) -> Vec<[usize; 4]> {
    let mut impropers = Vec::new();
    for (center, neighbors) in adjacency.iter().enumerate() {
        if neighbors.len() < 3 {
            continue;
        }
        for left in 0..neighbors.len() {
            for mid in (left + 1)..neighbors.len() {
                for right in (mid + 1)..neighbors.len() {
                    impropers.push([neighbors[left], center, neighbors[mid], neighbors[right]]);
                }
            }
        }
    }
    impropers.sort_unstable();
    impropers.dedup();
    impropers
}

fn topology_atom_signature(
    atom_names: &[String],
    amber_atom_types: &[String],
    atom_idx: usize,
) -> String {
    let name = atom_names
        .get(atom_idx)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .unwrap_or("X");
    let atom_type = amber_atom_types
        .get(atom_idx)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .unwrap_or(name);
    format!("{name}:{atom_type}")
}

fn canonical_pair_signature(left: &str, right: &str) -> (String, String) {
    if left <= right {
        (left.to_string(), right.to_string())
    } else {
        (right.to_string(), left.to_string())
    }
}

fn canonical_angle_signature(left: &str, center: &str, right: &str) -> [String; 3] {
    let forward = [left.to_string(), center.to_string(), right.to_string()];
    let reverse = [right.to_string(), center.to_string(), left.to_string()];
    if forward <= reverse {
        forward
    } else {
        reverse
    }
}

fn canonical_dihedral_signature(a: &str, b: &str, c: &str, d: &str) -> [String; 4] {
    let forward = [a.to_string(), b.to_string(), c.to_string(), d.to_string()];
    let reverse = [d.to_string(), c.to_string(), b.to_string(), a.to_string()];
    if forward <= reverse {
        forward
    } else {
        reverse
    }
}

fn canonical_improper_signature(a: &str, center: &str, c: &str, d: &str) -> (String, [String; 3]) {
    let mut outers = [a.to_string(), c.to_string(), d.to_string()];
    outers.sort();
    (center.to_string(), outers)
}

fn topology_atom_type_signature(amber_atom_types: &[String], atom_idx: usize) -> String {
    amber_atom_types
        .get(atom_idx)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .unwrap_or("X")
        .to_string()
}

fn insert_signature_mapping<K: Ord>(map: &mut BTreeMap<K, usize>, key: K, type_idx: usize) {
    map.entry(key).or_insert(type_idx);
}

fn insert_fallback_signature_mapping<K: Ord>(
    map: &mut BTreeMap<K, Option<usize>>,
    key: K,
    type_idx: usize,
) {
    use std::collections::btree_map::Entry;

    match map.entry(key) {
        Entry::Vacant(entry) => {
            entry.insert(Some(type_idx));
        }
        Entry::Occupied(mut entry) => {
            if *entry.get() != Some(type_idx) {
                entry.insert(None);
            }
        }
    }
}

fn missing_transferred_term_error(
    kind: &str,
    term: &[usize],
    atom_names: &[String],
    amber_atom_types: &[String],
) -> PackError {
    let labels = term
        .iter()
        .map(|&atom_idx| topology_atom_signature(atom_names, amber_atom_types, atom_idx))
        .collect::<Vec<_>>()
        .join(" - ");
    PackError::Invalid(format!(
        "source topology does not provide {kind} typing for transferred term {labels}"
    ))
}

fn rebuild_transferred_term_assignments(
    source: &AmberTopology,
    atom_names: &[String],
    amber_atom_types: &[String],
    bonds: &[(usize, usize)],
) -> PackResult<RebuiltTermAssignments> {
    let mut assignments = RebuiltTermAssignments {
        bonds: bonds.to_vec(),
        ..RebuiltTermAssignments::default()
    };
    let adjacency = rebuild_bond_adjacency(atom_names.len(), bonds);

    let mut bond_map = BTreeMap::new();
    let mut bond_type_map = BTreeMap::new();
    for (bond_idx, &(a, b)) in source.bonds.iter().enumerate() {
        let exact_key = canonical_pair_signature(
            &topology_atom_signature(&source.atom_names, &source.amber_atom_types, a),
            &topology_atom_signature(&source.atom_names, &source.amber_atom_types, b),
        );
        let type_key = canonical_pair_signature(
            &topology_atom_type_signature(&source.amber_atom_types, a),
            &topology_atom_type_signature(&source.amber_atom_types, b),
        );
        let type_idx = source.bond_type_indices.get(bond_idx).copied().unwrap_or(1);
        insert_signature_mapping(&mut bond_map, exact_key, type_idx);
        insert_fallback_signature_mapping(&mut bond_type_map, type_key, type_idx);
    }
    assignments.bond_type_indices = bonds
        .iter()
        .map(|&(a, b)| {
            let exact_key = canonical_pair_signature(
                &topology_atom_signature(atom_names, amber_atom_types, a),
                &topology_atom_signature(atom_names, amber_atom_types, b),
            );
            let type_key = canonical_pair_signature(
                &topology_atom_type_signature(amber_atom_types, a),
                &topology_atom_type_signature(amber_atom_types, b),
            );
            bond_map
                .get(&exact_key)
                .copied()
                .or_else(|| bond_type_map.get(&type_key).and_then(|value| *value))
                .ok_or_else(|| {
                    missing_transferred_term_error("bond", &[a, b], atom_names, amber_atom_types)
                })
        })
        .collect::<PackResult<Vec<_>>>()?;

    if !source.angles.is_empty() {
        let mut angle_map = BTreeMap::new();
        let mut angle_type_map = BTreeMap::new();
        for (angle_idx, term) in source.angles.iter().enumerate() {
            let exact_key = canonical_angle_signature(
                &topology_atom_signature(&source.atom_names, &source.amber_atom_types, term[0]),
                &topology_atom_signature(&source.atom_names, &source.amber_atom_types, term[1]),
                &topology_atom_signature(&source.atom_names, &source.amber_atom_types, term[2]),
            );
            let type_key = canonical_angle_signature(
                &topology_atom_type_signature(&source.amber_atom_types, term[0]),
                &topology_atom_type_signature(&source.amber_atom_types, term[1]),
                &topology_atom_type_signature(&source.amber_atom_types, term[2]),
            );
            let type_idx = source
                .angle_type_indices
                .get(angle_idx)
                .copied()
                .unwrap_or(1);
            insert_signature_mapping(&mut angle_map, exact_key, type_idx);
            insert_fallback_signature_mapping(&mut angle_type_map, type_key, type_idx);
        }
        for term in rebuild_angles(&adjacency) {
            let exact_key = canonical_angle_signature(
                &topology_atom_signature(atom_names, amber_atom_types, term[0]),
                &topology_atom_signature(atom_names, amber_atom_types, term[1]),
                &topology_atom_signature(atom_names, amber_atom_types, term[2]),
            );
            let type_key = canonical_angle_signature(
                &topology_atom_type_signature(amber_atom_types, term[0]),
                &topology_atom_type_signature(amber_atom_types, term[1]),
                &topology_atom_type_signature(amber_atom_types, term[2]),
            );
            let type_idx = angle_map
                .get(&exact_key)
                .copied()
                .or_else(|| angle_type_map.get(&type_key).and_then(|value| *value))
                .ok_or_else(|| {
                    missing_transferred_term_error("angle", &term, atom_names, amber_atom_types)
                })?;
            assignments.angles.push(term);
            assignments.angle_type_indices.push(type_idx);
        }
    }

    if !source.dihedrals.is_empty() {
        let mut dihedral_map: BTreeMap<[String; 4], Vec<usize>> = BTreeMap::new();
        let mut dihedral_type_map: BTreeMap<[String; 4], Vec<usize>> = BTreeMap::new();
        for (dihedral_idx, term) in source.dihedrals.iter().enumerate() {
            let exact_key = canonical_dihedral_signature(
                &topology_atom_signature(&source.atom_names, &source.amber_atom_types, term[0]),
                &topology_atom_signature(&source.atom_names, &source.amber_atom_types, term[1]),
                &topology_atom_signature(&source.atom_names, &source.amber_atom_types, term[2]),
                &topology_atom_signature(&source.atom_names, &source.amber_atom_types, term[3]),
            );
            let type_key = canonical_dihedral_signature(
                &topology_atom_type_signature(&source.amber_atom_types, term[0]),
                &topology_atom_type_signature(&source.amber_atom_types, term[1]),
                &topology_atom_type_signature(&source.amber_atom_types, term[2]),
                &topology_atom_type_signature(&source.amber_atom_types, term[3]),
            );
            let type_idx = source
                .dihedral_type_indices
                .get(dihedral_idx)
                .copied()
                .unwrap_or(1);
            dihedral_map.entry(exact_key).or_default().push(type_idx);
            let fallback = dihedral_type_map.entry(type_key).or_default();
            if !fallback.contains(&type_idx) {
                fallback.push(type_idx);
            }
        }
        for term in rebuild_dihedrals(&adjacency, bonds) {
            let exact_key = canonical_dihedral_signature(
                &topology_atom_signature(atom_names, amber_atom_types, term[0]),
                &topology_atom_signature(atom_names, amber_atom_types, term[1]),
                &topology_atom_signature(atom_names, amber_atom_types, term[2]),
                &topology_atom_signature(atom_names, amber_atom_types, term[3]),
            );
            let type_key = canonical_dihedral_signature(
                &topology_atom_type_signature(amber_atom_types, term[0]),
                &topology_atom_type_signature(amber_atom_types, term[1]),
                &topology_atom_type_signature(amber_atom_types, term[2]),
                &topology_atom_type_signature(amber_atom_types, term[3]),
            );
            let type_indices = dihedral_map
                .get(&exact_key)
                .or_else(|| dihedral_type_map.get(&type_key))
                .ok_or_else(|| {
                    missing_transferred_term_error("dihedral", &term, atom_names, amber_atom_types)
                })?;
            for &type_idx in type_indices {
                assignments.dihedrals.push(term);
                assignments.dihedral_type_indices.push(type_idx);
            }
        }
    }

    if !source.impropers.is_empty() {
        let mut improper_map: BTreeMap<(String, [String; 3]), Vec<usize>> = BTreeMap::new();
        let mut improper_type_map: BTreeMap<(String, [String; 3]), Vec<usize>> = BTreeMap::new();
        for (improper_idx, term) in source.impropers.iter().enumerate() {
            let exact_key = canonical_improper_signature(
                &topology_atom_signature(&source.atom_names, &source.amber_atom_types, term[0]),
                &topology_atom_signature(&source.atom_names, &source.amber_atom_types, term[1]),
                &topology_atom_signature(&source.atom_names, &source.amber_atom_types, term[2]),
                &topology_atom_signature(&source.atom_names, &source.amber_atom_types, term[3]),
            );
            let type_key = canonical_improper_signature(
                &topology_atom_type_signature(&source.amber_atom_types, term[0]),
                &topology_atom_type_signature(&source.amber_atom_types, term[1]),
                &topology_atom_type_signature(&source.amber_atom_types, term[2]),
                &topology_atom_type_signature(&source.amber_atom_types, term[3]),
            );
            let type_idx = source
                .improper_type_indices
                .get(improper_idx)
                .copied()
                .unwrap_or(1);
            improper_map.entry(exact_key).or_default().push(type_idx);
            let fallback = improper_type_map.entry(type_key).or_default();
            if !fallback.contains(&type_idx) {
                fallback.push(type_idx);
            }
        }
        for term in rebuild_impropers(&adjacency) {
            let exact_key = canonical_improper_signature(
                &topology_atom_signature(atom_names, amber_atom_types, term[0]),
                &topology_atom_signature(atom_names, amber_atom_types, term[1]),
                &topology_atom_signature(atom_names, amber_atom_types, term[2]),
                &topology_atom_signature(atom_names, amber_atom_types, term[3]),
            );
            let type_key = canonical_improper_signature(
                &topology_atom_type_signature(amber_atom_types, term[0]),
                &topology_atom_type_signature(amber_atom_types, term[1]),
                &topology_atom_type_signature(amber_atom_types, term[2]),
                &topology_atom_type_signature(amber_atom_types, term[3]),
            );
            let type_indices = improper_map
                .get(&exact_key)
                .or_else(|| improper_type_map.get(&type_key))
                .ok_or_else(|| {
                    missing_transferred_term_error("improper", &term, atom_names, amber_atom_types)
                })?;
            for &type_idx in type_indices {
                assignments.impropers.push(term);
                assignments.improper_type_indices.push(type_idx);
            }
        }
    }

    Ok(assignments)
}

pub fn write_polymer_prmtop_from_source(
    built: &PolymerBuiltArtifact,
    source_topology_path: &Path,
    out_path: &str,
) -> PackResult<()> {
    let source = read_prmtop_topology(source_topology_path)?;
    let source_blocks = residue_blocks(&source);
    if source_blocks.is_empty() {
        return Err(PackError::Invalid(
            "source prmtop does not contain any residue templates".into(),
        ));
    }
    if built.template_sequence_resnames.len() != built.residue_resnames.len() {
        return Err(PackError::Invalid(
            "built polymer residue metadata is inconsistent".into(),
        ));
    }

    let head_block = &source_blocks[0];
    let tail_block = &source_blocks[source_blocks.len() - 1];
    let repeat_block = &source_blocks[source_blocks.len() / 2];
    let source_by_label = source_blocks
        .iter()
        .filter_map(|block| {
            block
                .residue_labels
                .first()
                .map(|label| (label.clone(), block.clone()))
        })
        .collect::<BTreeMap<_, _>>();

    let mut atom_names = Vec::new();
    let mut atomic_numbers = Vec::new();
    let mut masses = Vec::new();
    let mut charges = Vec::new();
    let mut atom_type_indices = Vec::new();
    let mut amber_atom_types = Vec::new();
    let mut radii = Vec::new();
    let mut screen = Vec::new();
    let mut residue_labels = Vec::new();
    let mut residue_pointers = Vec::new();
    let mut atom_cursor = 1usize;
    let mut built_atoms_by_residue = vec![Vec::new(); built.residue_resnames.len()];
    for atom in &built.output.atoms {
        let resid = atom.resid.max(1) as usize;
        if let Some(items) = built_atoms_by_residue.get_mut(resid - 1) {
            items.push(atom.name.trim().to_string());
        }
    }

    for (idx, (template_resname, applied_resname)) in built
        .template_sequence_resnames
        .iter()
        .zip(built.residue_resnames.iter())
        .enumerate()
    {
        let block = source_by_label
            .get(template_resname)
            .or_else(|| if idx == 0 { Some(head_block) } else { None })
            .or_else(|| {
                if idx + 1 == built.residue_resnames.len() {
                    Some(tail_block)
                } else {
                    None
                }
            })
            .unwrap_or(repeat_block);
        let keep_names = atom_name_set(&built_atoms_by_residue[idx]);
        residue_pointers.push(atom_cursor);
        residue_labels.push(applied_resname.clone());
        for atom_idx in 0..block.atom_names.len() {
            if !keep_names.contains(block.atom_names[atom_idx].trim()) {
                continue;
            }
            atom_names.push(block.atom_names[atom_idx].clone());
            atomic_numbers.push(block.atomic_numbers[atom_idx]);
            charges.push(block.charges[atom_idx]);
            masses.push(block.masses.get(atom_idx).copied().unwrap_or(12.0));
            atom_type_indices.push(
                block
                    .atom_type_indices
                    .get(atom_idx)
                    .copied()
                    .unwrap_or(atom_type_indices.len() + 1),
            );
            amber_atom_types.push(
                block
                    .amber_atom_types
                    .get(atom_idx)
                    .cloned()
                    .unwrap_or_else(|| block.atom_names[atom_idx].clone()),
            );
            radii.push(block.radii.get(atom_idx).copied().unwrap_or(1.5));
            screen.push(block.screen.get(atom_idx).copied().unwrap_or(0.8));
            atom_cursor += 1;
        }
    }

    if atom_names.len() != built.output.atoms.len() {
        return Err(PackError::Invalid(format!(
            "source topology transfer produced {} atoms but built polymer has {} atoms",
            atom_names.len(),
            built.output.atoms.len()
        )));
    }
    let rebuilt_terms = rebuild_transferred_term_assignments(
        &source,
        &atom_names,
        &amber_atom_types,
        &built.output.bonds,
    )?;

    Ok(write_minimal_prmtop(
        out_path,
        &AmberTopology {
            atom_names,
            residue_labels,
            residue_pointers,
            atomic_numbers,
            masses,
            charges,
            atom_type_indices,
            amber_atom_types,
            radii,
            screen,
            bonds: rebuilt_terms.bonds,
            bond_type_indices: rebuilt_terms.bond_type_indices,
            bond_force_constants: source.bond_force_constants.clone(),
            bond_equil_values: source.bond_equil_values.clone(),
            angles: rebuilt_terms.angles,
            angle_type_indices: rebuilt_terms.angle_type_indices,
            angle_force_constants: source.angle_force_constants.clone(),
            angle_equil_values: source.angle_equil_values.clone(),
            dihedrals: rebuilt_terms.dihedrals,
            dihedral_type_indices: rebuilt_terms.dihedral_type_indices,
            dihedral_force_constants: source.dihedral_force_constants.clone(),
            dihedral_periodicities: source.dihedral_periodicities.clone(),
            dihedral_phases: source.dihedral_phases.clone(),
            scee_scale_factors: source.scee_scale_factors.clone(),
            scnb_scale_factors: source.scnb_scale_factors.clone(),
            solty: source.solty.clone(),
            impropers: rebuilt_terms.impropers,
            improper_type_indices: rebuilt_terms.improper_type_indices,
            excluded_atoms: Vec::new(),
            nonbonded_parm_index: source.nonbonded_parm_index.clone(),
            lennard_jones_acoef: source.lennard_jones_acoef.clone(),
            lennard_jones_bcoef: source.lennard_jones_bcoef.clone(),
            lennard_jones_14_acoef: source.lennard_jones_14_acoef.clone(),
            lennard_jones_14_bcoef: source.lennard_jones_14_bcoef.clone(),
            hbond_acoef: source.hbond_acoef.clone(),
            hbond_bcoef: source.hbond_bcoef.clone(),
            hbcut: source.hbcut.clone(),
            tree_chain_classification: vec!["M".into(); built.output.atoms.len()],
            join_array: vec![0; built.output.atoms.len()],
            irotat: vec![0; built.output.atoms.len()],
            solvent_pointers: source.solvent_pointers.clone(),
            atoms_per_molecule: source.atoms_per_molecule.clone(),
            box_dimensions: source.box_dimensions.clone(),
            radius_set: source.radius_set.clone(),
            ipol: source.ipol,
        },
    )?)
}

pub fn write_polymer_prmtop_from_ffxml(
    built: &PolymerBuiltArtifact,
    training_structure_path: &Path,
    source_charge_manifest_path: Option<&Path>,
    ffxml_path: &Path,
    out_path: &str,
) -> PackResult<FfxmlTopologySummary> {
    let forcefield = parse_ffxml_forcefield(ffxml_path)?;
    let template_charge_map = if let Some(path) = source_charge_manifest_path {
        build_template_charge_map_from_manifest(training_structure_path, path)?
    } else {
        BTreeMap::new()
    };
    if built.template_sequence_resnames.len() != built.residue_resnames.len() {
        return Err(PackError::Invalid(
            "built polymer residue metadata is inconsistent".into(),
        ));
    }

    let atom_count = built.output.atoms.len();
    let bonds = built.output.bonds.clone();
    let adjacency = rebuild_bond_adjacency(atom_count, &bonds);

    let mut residue_pointers = Vec::new();
    let residue_labels = built.residue_resnames.clone();
    let mut next_resid = 1usize;
    for (idx, atom) in built.output.atoms.iter().enumerate() {
        let resid = atom.resid.max(1) as usize;
        while next_resid <= resid {
            residue_pointers.push(idx + 1);
            next_resid += 1;
        }
    }
    while residue_pointers.len() < residue_labels.len() {
        residue_pointers.push(atom_count + 1);
    }

    let assigned_atoms = built
        .output
        .atoms
        .iter()
        .map(|atom| {
            let resid = atom.resid.max(1) as usize;
            let template_resname = built
                .template_sequence_resnames
                .get(resid.saturating_sub(1))
                .cloned()
                .unwrap_or_else(|| atom.resname.clone());
            ffxml_assign_training_atom(
                &forcefield,
                &template_resname,
                atom.name.trim(),
                template_charge_map
                    .get(&(template_resname.clone(), atom.name.trim().to_string()))
                    .copied(),
            )
        })
        .collect::<PackResult<Vec<_>>>()?;

    let atom_names = built
        .output
        .atoms
        .iter()
        .map(|atom| atom.name.clone())
        .collect::<Vec<_>>();
    let atomic_numbers = assigned_atoms
        .iter()
        .map(|atom| atomic_number_for_element(&atom.element))
        .collect::<Vec<_>>();
    let masses = assigned_atoms
        .iter()
        .map(|atom| atom.mass)
        .collect::<Vec<_>>();
    let charges = assigned_atoms
        .iter()
        .map(|atom| atom.charge_e)
        .collect::<Vec<_>>();
    let net_charge_e = charges.iter().sum::<f32>();

    let mut unique_type_indices = BTreeMap::<String, usize>::new();
    let mut unique_nonbonded = Vec::<(f32, f32)>::new();
    let atom_type_indices = assigned_atoms
        .iter()
        .map(|atom| {
            if let Some(idx) = unique_type_indices.get(&atom.type_name) {
                *idx
            } else {
                let next = unique_nonbonded.len() + 1;
                unique_type_indices.insert(atom.type_name.clone(), next);
                unique_nonbonded.push((atom.sigma_angstrom, atom.epsilon_kcal));
                next
            }
        })
        .collect::<Vec<_>>();
    let amber_atom_types = assigned_atoms
        .iter()
        .map(|atom| atom.type_name.clone())
        .collect::<Vec<_>>();
    let radii = assigned_atoms
        .iter()
        .map(|atom| (atom.sigma_angstrom * 0.5 * 2.0_f32.powf(1.0 / 6.0)).max(0.5))
        .collect::<Vec<_>>();
    let screen = vec![0.8; atom_count];

    let mut bond_force_constants = Vec::with_capacity(bonds.len());
    let mut bond_equil_values = Vec::with_capacity(bonds.len());
    let mut bond_type_indices = Vec::with_capacity(bonds.len());
    for &(left, right) in &bonds {
        let param = ffxml_match_bond_param(
            &forcefield.bond_params,
            &assigned_atoms[left],
            &assigned_atoms[right],
        )
        .ok_or_else(|| {
            PackError::Invalid(format!(
                "ffxml is missing bond parameters for '{}:{}' and '{}:{}'",
                built.output.atoms[left].resname,
                built.output.atoms[left].name.trim(),
                built.output.atoms[right].resname,
                built.output.atoms[right].name.trim(),
            ))
        })?;
        bond_type_indices.push(bond_force_constants.len() + 1);
        bond_force_constants.push(param.force_constant);
        bond_equil_values.push(param.length_angstrom);
    }

    let angles = rebuild_angles(&adjacency);
    let mut angle_force_constants = Vec::with_capacity(angles.len());
    let mut angle_equil_values = Vec::with_capacity(angles.len());
    let mut angle_type_indices = Vec::with_capacity(angles.len());
    for angle in &angles {
        let param = ffxml_match_angle_param(
            &forcefield.angle_params,
            &assigned_atoms[angle[0]],
            &assigned_atoms[angle[1]],
            &assigned_atoms[angle[2]],
        )
        .ok_or_else(|| {
            PackError::Invalid(format!(
                "ffxml is missing angle parameters for '{}'-'{}'-'{}'",
                built.output.atoms[angle[0]].name.trim(),
                built.output.atoms[angle[1]].name.trim(),
                built.output.atoms[angle[2]].name.trim(),
            ))
        })?;
        angle_type_indices.push(angle_force_constants.len() + 1);
        angle_force_constants.push(param.force_constant);
        angle_equil_values.push(param.theta0_rad);
    }

    let dihedral_terms = rebuild_dihedrals(&adjacency, &bonds);
    let mut dihedrals = Vec::new();
    let mut dihedral_type_indices = Vec::new();
    let mut dihedral_force_constants = Vec::new();
    let mut dihedral_periodicities = Vec::new();
    let mut dihedral_phases = Vec::new();
    let mut scee_scale_factors = Vec::new();
    let mut scnb_scale_factors = Vec::new();
    for dihedral in &dihedral_terms {
        let param = ffxml_match_torsion_param(
            &forcefield.proper_torsions,
            [
                &assigned_atoms[dihedral[0]],
                &assigned_atoms[dihedral[1]],
                &assigned_atoms[dihedral[2]],
                &assigned_atoms[dihedral[3]],
            ],
            false,
        )
        .ok_or_else(|| {
            PackError::Invalid(format!(
                "ffxml is missing torsion parameters for '{}'-'{}'-'{}'-'{}'",
                built.output.atoms[dihedral[0]].name.trim(),
                built.output.atoms[dihedral[1]].name.trim(),
                built.output.atoms[dihedral[2]].name.trim(),
                built.output.atoms[dihedral[3]].name.trim(),
            ))
        })?;
        for term in &param.terms {
            dihedrals.push(*dihedral);
            dihedral_type_indices.push(dihedral_force_constants.len() + 1);
            dihedral_force_constants.push(term.force_constant);
            dihedral_periodicities.push(term.periodicity);
            dihedral_phases.push(term.phase_rad);
            scee_scale_factors.push(forcefield.scee_scale);
            scnb_scale_factors.push(forcefield.scnb_scale);
        }
    }

    let improper_terms = rebuild_impropers(&adjacency);
    let mut impropers = Vec::new();
    let mut improper_type_indices = Vec::new();
    for improper in &improper_terms {
        let param = ffxml_match_torsion_param(
            &forcefield.improper_torsions,
            [
                &assigned_atoms[improper[0]],
                &assigned_atoms[improper[1]],
                &assigned_atoms[improper[2]],
                &assigned_atoms[improper[3]],
            ],
            true,
        )
        .ok_or_else(|| {
            PackError::Invalid(format!(
                "ffxml is missing improper parameters for '{}'-'{}'-'{}'-'{}'",
                built.output.atoms[improper[0]].name.trim(),
                built.output.atoms[improper[1]].name.trim(),
                built.output.atoms[improper[2]].name.trim(),
                built.output.atoms[improper[3]].name.trim(),
            ))
        })?;
        for term in &param.terms {
            impropers.push(*improper);
            improper_type_indices.push(dihedral_force_constants.len() + 1);
            dihedral_force_constants.push(term.force_constant);
            dihedral_periodicities.push(term.periodicity);
            dihedral_phases.push(term.phase_rad);
            scee_scale_factors.push(forcefield.scee_scale);
            scnb_scale_factors.push(forcefield.scnb_scale);
        }
    }

    let (
        nonbonded_parm_index,
        lennard_jones_acoef,
        lennard_jones_bcoef,
        lennard_jones_14_acoef,
        lennard_jones_14_bcoef,
    ) = build_nonbonded_tables_from_sigma_epsilon(&unique_nonbonded);

    write_minimal_prmtop(
        out_path,
        &AmberTopology {
            atom_names,
            residue_labels,
            residue_pointers,
            atomic_numbers,
            masses,
            charges,
            atom_type_indices,
            amber_atom_types,
            radii,
            screen,
            bonds,
            bond_type_indices,
            bond_force_constants,
            bond_equil_values,
            angles,
            angle_type_indices,
            angle_force_constants,
            angle_equil_values,
            dihedrals,
            dihedral_type_indices,
            dihedral_force_constants,
            dihedral_periodicities,
            dihedral_phases,
            scee_scale_factors,
            scnb_scale_factors,
            solty: vec![0.0; unique_nonbonded.len().max(1)],
            impropers,
            improper_type_indices,
            excluded_atoms: Vec::new(),
            nonbonded_parm_index,
            lennard_jones_acoef,
            lennard_jones_bcoef,
            lennard_jones_14_acoef,
            lennard_jones_14_bcoef,
            hbond_acoef: Vec::new(),
            hbond_bcoef: Vec::new(),
            hbcut: Vec::new(),
            tree_chain_classification: vec!["M".into(); atom_count],
            join_array: vec![0; atom_count],
            irotat: vec![0; atom_count],
            solvent_pointers: Vec::new(),
            atoms_per_molecule: vec![atom_count],
            box_dimensions: Vec::new(),
            radius_set: Some("forcefield-ref radii".into()),
            ipol: 0,
        },
    )?;

    Ok(FfxmlTopologySummary { net_charge_e })
}

pub fn build_polymer_synthetic_uff_topology(
    built: &PolymerBuiltArtifact,
    training_structure_path: &Path,
    source_charge_manifest_path: Option<&Path>,
) -> PackResult<AmberTopology> {
    let templates = load_training_templates(training_structure_path)?;
    if built.template_sequence_resnames.len() != built.residue_resnames.len() {
        return Err(PackError::Invalid(
            "built polymer residue metadata is inconsistent".into(),
        ));
    }
    let atom_count = built.output.atoms.len();
    let bonds = built.output.bonds.clone();
    let adjacency = rebuild_bond_adjacency(atom_count, &bonds);
    let typings = (0..atom_count)
        .map(|atom_idx| infer_uff_like_typing(&built.output.atoms, &adjacency, atom_idx))
        .collect::<Vec<_>>();
    let bond_length_map = template_bond_length_map(&templates);
    let angle_map = template_angle_map(&templates);
    let template_charge_map = if let Some(path) = source_charge_manifest_path {
        build_template_charge_map_from_manifest(training_structure_path, path)?
    } else {
        BTreeMap::new()
    };

    let mut residue_pointers = Vec::new();
    let residue_labels = built.residue_resnames.clone();
    let mut next_resid = 1usize;
    for (idx, atom) in built.output.atoms.iter().enumerate() {
        let resid = atom.resid.max(1) as usize;
        while next_resid <= resid {
            residue_pointers.push(idx + 1);
            next_resid += 1;
        }
    }
    while residue_pointers.len() < residue_labels.len() {
        residue_pointers.push(atom_count + 1);
    }

    let atom_names = built
        .output
        .atoms
        .iter()
        .map(|atom| atom.name.clone())
        .collect::<Vec<_>>();
    let atomic_numbers = built
        .output
        .atoms
        .iter()
        .map(|atom| atomic_number_for_element(&atom.element))
        .collect::<Vec<_>>();
    let masses = built
        .output
        .atoms
        .iter()
        .map(|atom| element_mass_amu(&atom.element))
        .collect::<Vec<_>>();

    let mut unique_type_indices = BTreeMap::<&'static str, usize>::new();
    let mut unique_types = Vec::<UffAtomParams>::new();
    let atom_type_indices = typings
        .iter()
        .map(|typing| {
            if let Some(idx) = unique_type_indices.get(typing.params.label) {
                *idx
            } else {
                let next = unique_types.len() + 1;
                unique_type_indices.insert(typing.params.label, next);
                unique_types.push(typing.params);
                next
            }
        })
        .collect::<Vec<_>>();
    let amber_atom_types = typings
        .iter()
        .map(|typing| typing.params.label.to_string())
        .collect::<Vec<_>>();
    let radii = typings
        .iter()
        .map(|typing| (typing.params.x1 * 0.5).max(0.5))
        .collect::<Vec<_>>();
    let screen = vec![0.8; atom_count];
    let charges = built
        .output
        .atoms
        .iter()
        .map(|atom| {
            let resid = atom.resid.max(1) as usize;
            let template_resname = built
                .template_sequence_resnames
                .get(resid.saturating_sub(1))
                .cloned()
                .unwrap_or_else(|| atom.resname.clone());
            template_charge_map
                .get(&(template_resname, atom.name.trim().to_string()))
                .copied()
                .unwrap_or(0.0)
        })
        .collect::<Vec<_>>();

    let mut bond_specs = Vec::with_capacity(bonds.len());
    let mut bond_orders = BTreeMap::<(usize, usize), f32>::new();
    for &(left, right) in &bonds {
        let order = guess_bond_order(&built.output.atoms, &adjacency, &typings, left, right);
        let left_typing = typings
            .get(left)
            .map(|item| item.params)
            .unwrap_or_else(|| uff_params_for_label("C_3"));
        let right_typing = typings
            .get(right)
            .map(|item| item.params)
            .unwrap_or_else(|| uff_params_for_label("C_3"));
        let uff_rest = uff_bond_rest_length(order, left_typing, right_typing).clamp(0.8, 2.4);
        let observed_rest = if built.output.atoms[left].resid == built.output.atoms[right].resid {
            let resid = built.output.atoms[left].resid.max(1) as usize;
            let template_resname = built
                .template_sequence_resnames
                .get(resid.saturating_sub(1))
                .map(String::as_str)
                .unwrap_or(built.output.atoms[left].resname.as_str());
            approx_bond_length_from_training(
                &bond_length_map,
                template_resname,
                &built.output.atoms[left].name,
                &built.output.atoms[right].name,
            )
        } else {
            let left_resid = built.output.atoms[left].resid.max(1) as usize;
            let right_resid = built.output.atoms[right].resid.max(1) as usize;
            let left_template = built
                .template_sequence_resnames
                .get(left_resid.saturating_sub(1))
                .map(String::as_str)
                .unwrap_or(built.output.atoms[left].resname.as_str());
            let right_template = built
                .template_sequence_resnames
                .get(right_resid.saturating_sub(1))
                .map(String::as_str)
                .unwrap_or(built.output.atoms[right].resname.as_str());
            observed_attach_distance(
                &templates,
                left_template,
                right_template,
                &built.output.atoms[left].name,
                &built.output.atoms[right].name,
            )
        };
        let rest_length = observed_rest
            .filter(|value| (*value - uff_rest).abs() <= 0.35)
            .unwrap_or(uff_rest);
        let force_constant =
            uff_bond_force_constant(rest_length, left_typing, right_typing).clamp(120.0, 1200.0);
        bond_orders.insert(ordered_pair(left, right), order);
        bond_specs.push(SyntheticBondSpec {
            bond: (left, right),
            rest_length,
            force_constant,
        });
    }

    let angles = rebuild_angles(&adjacency);
    let angle_type_indices = (1..=angles.len()).collect::<Vec<_>>();
    let mut angle_force_constants = Vec::with_capacity(angles.len());
    let mut angle_equil_values = Vec::with_capacity(angles.len());
    for angle in &angles {
        let left = angle[0];
        let center = angle[1];
        let right = angle[2];
        let center_typing = typings
            .get(center)
            .map(|item| item.params)
            .unwrap_or_else(|| uff_params_for_label("C_3"));
        let left_typing = typings
            .get(left)
            .map(|item| item.params)
            .unwrap_or_else(|| uff_params_for_label("C_3"));
        let right_typing = typings
            .get(right)
            .map(|item| item.params)
            .unwrap_or_else(|| uff_params_for_label("C_3"));
        let bond_order_left = bond_orders
            .get(&ordered_pair(left, center))
            .copied()
            .unwrap_or(1.0);
        let bond_order_right = bond_orders
            .get(&ordered_pair(center, right))
            .copied()
            .unwrap_or(1.0);
        let theta0 = if built.output.atoms[left].resid == built.output.atoms[center].resid
            && built.output.atoms[center].resid == built.output.atoms[right].resid
        {
            let resid = built.output.atoms[center].resid.max(1) as usize;
            let template_resname = built
                .template_sequence_resnames
                .get(resid.saturating_sub(1))
                .map(String::as_str)
                .unwrap_or(built.output.atoms[center].resname.as_str());
            angle_map
                .get(&template_angle_key(
                    template_resname,
                    &built.output.atoms[left].name,
                    &built.output.atoms[center].name,
                    &built.output.atoms[right].name,
                ))
                .copied()
                .unwrap_or(center_typing.theta0_rad)
        } else {
            center_typing.theta0_rad
        };
        let force_constant = uff_angle_force_constant(
            theta0,
            bond_order_left,
            bond_order_right,
            left_typing,
            center_typing,
            right_typing,
        )
        .clamp(10.0, 500.0);
        angle_equil_values.push(theta0);
        angle_force_constants.push(force_constant);
    }

    let dihedrals = rebuild_dihedrals(&adjacency, &bonds);
    let mut dihedral_type_indices = Vec::with_capacity(dihedrals.len());
    let mut dihedral_force_constants = Vec::new();
    let mut dihedral_periodicities = Vec::new();
    let mut dihedral_phases = Vec::new();
    let mut scee_scale_factors = Vec::new();
    let mut scnb_scale_factors = Vec::new();
    for dihedral in &dihedrals {
        let params = synthetic_torsion_params(
            &built.output.atoms,
            &adjacency,
            &typings,
            &bond_orders,
            *dihedral,
        );
        dihedral_type_indices.push(dihedral_force_constants.len() + 1);
        dihedral_force_constants.push(params.force_constant);
        dihedral_periodicities.push(params.periodicity);
        dihedral_phases.push(params.phase_rad);
        scee_scale_factors.push(1.2);
        scnb_scale_factors.push(2.0);
    }

    let impropers = rebuild_impropers(&adjacency);
    let mut improper_type_indices = Vec::with_capacity(impropers.len());
    for improper in &impropers {
        let center = improper[1];
        let center_atomic_number = atomic_numbers.get(center).copied().unwrap_or_default();
        let carbon_bound_to_oxygen = center_atomic_number == 6
            && adjacency.get(center).into_iter().flatten().any(|neighbor| {
                atomic_numbers.get(*neighbor).copied().unwrap_or_default() == 8
                    && bond_orders
                        .get(&ordered_pair(center, *neighbor))
                        .copied()
                        .unwrap_or(1.0)
                        >= 1.8
            });
        improper_type_indices.push(dihedral_force_constants.len() + 1);
        dihedral_force_constants.push(synthetic_improper_force_constant(
            center_atomic_number,
            carbon_bound_to_oxygen,
        ));
        dihedral_periodicities.push(2.0);
        dihedral_phases.push(std::f32::consts::PI);
        scee_scale_factors.push(1.2);
        scnb_scale_factors.push(2.0);
    }

    let (
        nonbonded_parm_index,
        lennard_jones_acoef,
        lennard_jones_bcoef,
        lennard_jones_14_acoef,
        lennard_jones_14_bcoef,
    ) = build_nonbonded_tables(&unique_types);

    Ok(AmberTopology {
        atom_names,
        residue_labels,
        residue_pointers,
        atomic_numbers,
        masses,
        charges,
        atom_type_indices,
        amber_atom_types,
        radii,
        screen,
        bonds: bond_specs.iter().map(|spec| spec.bond).collect(),
        bond_type_indices: (1..=bond_specs.len()).collect(),
        bond_force_constants: bond_specs.iter().map(|spec| spec.force_constant).collect(),
        bond_equil_values: bond_specs.iter().map(|spec| spec.rest_length).collect(),
        angles,
        angle_type_indices,
        angle_force_constants,
        angle_equil_values,
        dihedrals,
        dihedral_type_indices,
        dihedral_force_constants,
        dihedral_periodicities,
        dihedral_phases,
        scee_scale_factors,
        scnb_scale_factors,
        solty: vec![0.0; unique_types.len().max(1)],
        impropers,
        improper_type_indices,
        excluded_atoms: Vec::new(),
        nonbonded_parm_index,
        lennard_jones_acoef,
        lennard_jones_bcoef,
        lennard_jones_14_acoef,
        lennard_jones_14_bcoef,
        hbond_acoef: Vec::new(),
        hbond_bcoef: Vec::new(),
        hbcut: Vec::new(),
        tree_chain_classification: vec!["M".into(); atom_count],
        join_array: vec![0; atom_count],
        irotat: vec![0; atom_count],
        solvent_pointers: Vec::new(),
        atoms_per_molecule: vec![atom_count],
        box_dimensions: Vec::new(),
        radius_set: Some("synthetic UFF-like radii".into()),
        ipol: 0,
    })
}

pub fn write_polymer_prmtop_synthetic_uff_like(
    built: &PolymerBuiltArtifact,
    training_structure_path: &Path,
    source_charge_manifest_path: Option<&Path>,
    out_path: &str,
) -> PackResult<()> {
    let topology = build_polymer_synthetic_uff_topology(
        built,
        training_structure_path,
        source_charge_manifest_path,
    )?;
    Ok(write_minimal_prmtop(out_path, &topology)?)
}

pub(crate) fn stage_polymer_output_path(final_coordinates_path: &str) -> PathBuf {
    let final_path = PathBuf::from(final_coordinates_path);
    let parent = final_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = final_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("system");
    parent.join(format!("{stem}_built_solute.pdb"))
}

#[allow(dead_code)]
fn build_centroids(
    n_repeat: usize,
    step_length: f32,
    conformation_mode: &str,
    source_axis: Vec3,
    seed: u64,
    collision_radii: &[f32],
) -> PackResult<Vec<Vec3>> {
    let axis = normalize(source_axis);
    if conformation_mode == "extended" {
        let mut centroids = Vec::with_capacity(n_repeat);
        let mut current = Vec3::new(0.0, 0.0, 0.0);
        for idx in 0..n_repeat {
            if idx > 0 {
                current = current.add(
                    aligned_zigzag_direction(axis, idx.saturating_sub(1), 0.0).scale(step_length),
                );
            }
            centroids.push(current);
        }
        return Ok(centroids);
    }
    if conformation_mode != "random_walk" {
        return Err(PackError::Invalid(format!(
            "unsupported polymer conformation mode '{}'",
            conformation_mode
        )));
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = vec![Vec3::new(0.0, 0.0, 0.0)];
    let mut directions = vec![axis];
    let persistence = 0.72f32;
    let max_bend_cos = (-0.35f32).max(-1.0);
    let clearance = 0.30f32;
    let mut envelope_index = ResidueEnvelopeIndex::new(collision_radii, clearance, n_repeat);
    envelope_index.insert(0, Vec3::new(0.0, 0.0, 0.0));
    let mut backtracks = 0usize;

    while centroids.len() < n_repeat {
        let candidate_idx = centroids.len();
        let candidate_radius = collision_radii.get(candidate_idx).copied().unwrap_or(0.65);
        let prev_dir = *directions.last().unwrap_or(&axis);
        let mut best_candidate = None;
        let mut best_clearance = f32::NEG_INFINITY;
        for attempt in 0..4096 {
            let random_dir = normalize(Vec3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ));
            let persistence = if attempt < 1024 {
                persistence
            } else if attempt < 3072 {
                0.45
            } else {
                0.18
            };
            let dir = normalize(
                prev_dir
                    .scale(persistence)
                    .add(random_dir.scale(1.0 - persistence)),
            );
            if centroids.len() >= 2 && dir.dot(prev_dir) < max_bend_cos {
                continue;
            }
            let candidate = centroids
                .last()
                .copied()
                .unwrap_or(Vec3::new(0.0, 0.0, 0.0))
                .add(dir.scale(step_length));
            let candidate_clearance = envelope_index.linear_clearance(
                candidate,
                &centroids,
                candidate_idx,
                candidate_radius,
                collision_radii,
                clearance,
            );
            if candidate_clearance > best_clearance {
                best_clearance = candidate_clearance;
                best_candidate = Some((candidate, dir));
            }
            if candidate_clearance >= 0.0 {
                break;
            }
        }
        if let Some((candidate, dir)) = best_candidate.filter(|_| best_clearance >= -0.10) {
            centroids.push(candidate);
            directions.push(dir);
            envelope_index.insert(candidate_idx, candidate);
            backtracks = 0;
            continue;
        }
        if centroids.len() > 1 && backtracks < n_repeat.saturating_mul(4).max(16) {
            if let Some(removed) = centroids.pop() {
                envelope_index.remove(centroids.len(), removed);
            }
            directions.pop();
            backtracks += 1;
            continue;
        }
        return Err(PackError::Invalid(
            "random_walk self-avoiding placement exhausted candidate directions".into(),
        ));
    }

    Ok(centroids)
}

#[allow(dead_code)]
pub fn build_linear_homopolymer(
    artifact: &str,
    explicit_charge_manifest: Option<&str>,
    explicit_topology_ref: Option<&str>,
    source_nmer: usize,
    target_n_repeat: usize,
    token_junctions: Option<&BTreeMap<String, TokenJunctionSpec>>,
    conformation_mode: &str,
    tacticity_mode: &str,
    head_label: Option<&str>,
    repeat_label: Option<&str>,
    tail_label: Option<&str>,
    build_seed: u64,
    final_coordinates_path: &str,
) -> PackResult<PolymerBuiltArtifact> {
    let repeat_token = repeat_label
        .map(str::trim)
        .filter(|label| !label.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            PackError::Invalid(
                "linear_homopolymer requires an explicit repeat token; no default token exists"
                    .into(),
            )
        })?;
    let resolved =
        resolve_polymer_param_source(artifact, explicit_charge_manifest, explicit_topology_ref)?;
    let templates = load_training_templates(&resolved.training_structure_path)?;
    let repeat_template_resname = templates
        .get(templates.len() / 2)
        .map(|template| template.resname.clone())
        .ok_or_else(|| {
            PackError::Invalid("training structure does not contain any residues".into())
        })?;
    let sequence = vec![repeat_token.clone(); target_n_repeat];
    let mut template_resname_by_token = BTreeMap::new();
    template_resname_by_token.insert(repeat_token.clone(), repeat_template_resname);
    build_linear_sequence_polymer(
        artifact,
        explicit_charge_manifest,
        explicit_topology_ref,
        source_nmer,
        &sequence,
        &template_resname_by_token,
        token_junctions,
        conformation_mode,
        tacticity_mode,
        head_label,
        tail_label,
        build_seed,
        final_coordinates_path,
    )
}

#[allow(dead_code)]
pub fn build_linear_sequence_polymer(
    artifact: &str,
    explicit_charge_manifest: Option<&str>,
    explicit_topology_ref: Option<&str>,
    source_nmer: usize,
    sequence_labels: &[String],
    template_resname_by_token: &BTreeMap<String, String>,
    token_junctions: Option<&BTreeMap<String, TokenJunctionSpec>>,
    conformation_mode: &str,
    tacticity_mode: &str,
    head_label: Option<&str>,
    tail_label: Option<&str>,
    build_seed: u64,
    final_coordinates_path: &str,
) -> PackResult<PolymerBuiltArtifact> {
    if source_nmer < 3 {
        return Err(PackError::Invalid(
            "polymer param_source.nmer must be >= 3".into(),
        ));
    }
    if sequence_labels.is_empty() {
        return Err(PackError::Invalid(
            "polymer target sequence must contain at least one residue".into(),
        ));
    }

    let resolved =
        resolve_polymer_param_source(artifact, explicit_charge_manifest, explicit_topology_ref)?;
    let templates = load_training_templates(&resolved.training_structure_path)?;
    if templates.len() < source_nmer {
        return Err(PackError::Invalid(format!(
            "training structure has {} residues but param_source.nmer={source_nmer}",
            templates.len()
        )));
    }

    let head = &templates[0];
    let repeat = &templates[templates.len() / 2];
    let tail = &templates[templates.len() - 1];
    let template_lookup = templates
        .iter()
        .map(|template| (template.resname.clone(), template))
        .collect::<BTreeMap<_, _>>();
    let source_axis = normalize(
        repeat
            .centroid
            .sub(head.centroid)
            .add(tail.centroid.sub(repeat.centroid))
            .scale(0.5),
    );
    let step_length = repeat
        .centroid
        .sub(head.centroid)
        .norm()
        .max(tail.centroid.sub(repeat.centroid).norm())
        .max(1.5);
    let sequence_specs = resolve_template_sequence(
        sequence_labels,
        template_resname_by_token,
        head_label,
        tail_label,
    )?;
    let collision_radii = sequence_specs
        .iter()
        .map(|spec| {
            template_lookup
                .get(&spec.template_resname)
                .copied()
                .unwrap_or(repeat)
        })
        .map(template_collision_radius)
        .collect::<Vec<_>>();
    let centroids = build_centroids(
        sequence_specs.len(),
        step_length,
        conformation_mode,
        source_axis,
        build_seed,
        &collision_radii,
    )?;

    let mut atoms = Vec::new();
    let mut bonds = Vec::new();
    let mut ter_after = Vec::new();
    let mut serial_index = 0usize;
    let mut tacticity_rng = StdRng::seed_from_u64(build_seed ^ 0x5eed_5eed);
    let mut previous_tail_attach = None;

    for (idx, spec) in sequence_specs.iter().enumerate() {
        let template = template_lookup
            .get(&spec.template_resname)
            .copied()
            .unwrap_or(repeat);
        let junction_spec = token_junctions.and_then(|specs| specs.get(&spec.sequence_label));
        let mut removed_names = BTreeSet::new();
        if idx > 0 {
            if let Some(junction) = junction_spec {
                removed_names.extend(atom_name_set(&junction.head_leaving_atoms));
            }
        }
        if idx + 1 < sequence_specs.len() {
            if let Some(junction) = junction_spec {
                removed_names.extend(atom_name_set(&junction.tail_leaving_atoms));
            }
        }
        let atom_indices = template_atom_index_by_name(template);
        let direction = if sequence_specs.len() == 1 {
            source_axis
        } else if idx == 0 {
            normalize(centroids[1].sub(centroids[0]))
        } else if idx == sequence_specs.len() - 1 {
            normalize(centroids[idx].sub(centroids[idx - 1]))
        } else {
            normalize(centroids[idx + 1].sub(centroids[idx - 1]))
        };
        let phase = tacticity_phase(tacticity_mode, idx, &mut tacticity_rng)?;
        let residue_start = atoms.len();
        let mut local_to_global = BTreeMap::new();

        for (local_idx, atom) in template.atoms.iter().enumerate() {
            if removed_names.contains(atom.name.trim()) {
                continue;
            }
            let local = atom.position.sub(template.centroid);
            let rotated = rotate_from_to(local, source_axis, direction);
            let rotated = rotate_about_axis(rotated, direction, phase);
            let mut built = atom.clone();
            built.position = centroids[idx].add(rotated);
            built.resid = (idx + 1) as i32;
            built.chain = 'A';
            built.resname = spec.template_resname.clone();
            built.mol_id = 1;
            atoms.push(built);
            local_to_global.insert(local_idx, residue_start + local_to_global.len());
            serial_index += 1;
        }
        for &(a, b) in &template.local_bonds {
            let Some(&i) = local_to_global.get(&a) else {
                continue;
            };
            let Some(&j) = local_to_global.get(&b) else {
                continue;
            };
            let (i, j) = if i <= j { (i, j) } else { (j, i) };
            if i != j {
                bonds.push((i, j));
            }
        }
        let head_attach = if idx > 0 {
            junction_spec
                .and_then(|junction| junction.head_attach_atom.as_ref())
                .map(|name| {
                    let local_idx = atom_indices.get(name.trim()).copied().ok_or_else(|| {
                        PackError::Invalid(format!(
                            "head junction attach atom '{}' missing from template '{}'",
                            name, spec.template_resname
                        ))
                    })?;
                    local_to_global.get(&local_idx).copied().ok_or_else(|| {
                        PackError::Invalid(format!(
                            "head junction attach atom '{}' was removed from template '{}'",
                            name, spec.template_resname
                        ))
                    })
                })
                .transpose()?
        } else {
            None
        };
        let tail_attach = if idx + 1 < sequence_specs.len() {
            junction_spec
                .and_then(|junction| junction.tail_attach_atom.as_ref())
                .map(|name| {
                    let local_idx = atom_indices.get(name.trim()).copied().ok_or_else(|| {
                        PackError::Invalid(format!(
                            "tail junction attach atom '{}' missing from template '{}'",
                            name, spec.template_resname
                        ))
                    })?;
                    local_to_global.get(&local_idx).copied().ok_or_else(|| {
                        PackError::Invalid(format!(
                            "tail junction attach atom '{}' was removed from template '{}'",
                            name, spec.template_resname
                        ))
                    })
                })
                .transpose()?
        } else {
            None
        };
        if let (Some(prev), Some(curr)) = (previous_tail_attach, head_attach) {
            let (i, j) = if prev <= curr {
                (prev, curr)
            } else {
                (curr, prev)
            };
            if i != j {
                bonds.push((i, j));
            }
        }
        previous_tail_attach = tail_attach;

        if let Some(last) = serial_index.checked_sub(1) {
            ter_after.push(last);
        }
    }
    bonds.sort_unstable();
    bonds.dedup();

    let output = PackOutput {
        atoms,
        bonds,
        box_size: [0.0, 0.0, 0.0],
        ter_after,
        box_vectors: None,
    };
    let qc_context = BuildQcContext {
        inter_residue_bond_count: sequence_specs.len().saturating_sub(1),
        terminal_connectivity_consistent: true,
        sequence_token_template_consistent: true,
        bond_expectations: Vec::new(),
    };
    let qc_report = recompute_build_qc_report(&output, &qc_context);
    let path = stage_polymer_output_path(final_coordinates_path);
    let path_text = path.to_string_lossy().to_string();
    if let Some(parent) = Path::new(&path_text).parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_output(
        &output,
        &OutputSpec {
            path: path_text,
            format: "pdb".to_string(),
            scale: Some(1.0),
        },
        false,
        0.0,
        !output.bonds.is_empty(),
        false,
    )?;

    Ok(PolymerBuiltArtifact {
        path,
        step_length_angstrom: step_length,
        sequence_labels: sequence_specs
            .iter()
            .map(|item| item.sequence_label.clone())
            .collect(),
        template_sequence_resnames: sequence_specs
            .iter()
            .map(|item| item.template_resname.clone())
            .collect(),
        residue_resnames: sequence_specs
            .iter()
            .map(|item| item.applied_resname.clone())
            .collect(),
        output,
        qc_context,
        qc_report,
        solver_report: None,
    })
}

pub fn build_polymer_graph(
    artifact: &str,
    explicit_charge_manifest: Option<&str>,
    explicit_topology_ref: Option<&str>,
    source_nmer: usize,
    node_specs: &[GraphNodeSpec],
    edge_specs: &[GraphEdgeSpec],
    root_idx: usize,
    conformation_mode: &str,
    tacticity_mode: &str,
    build_seed: u64,
    strict_qc: bool,
    final_coordinates_path: &str,
) -> PackResult<PolymerBuiltArtifact> {
    if source_nmer < 3 {
        return Err(PackError::Invalid(
            "polymer param_source.nmer must be >= 3".into(),
        ));
    }
    if node_specs.is_empty() {
        return Err(PackError::Invalid(
            "polymer graph must contain at least one residue".into(),
        ));
    }
    if conformation_mode != "extended" && conformation_mode != "random_walk" {
        return Err(PackError::Invalid(format!(
            "unsupported polymer conformation mode '{}'",
            conformation_mode
        )));
    }

    let resolved =
        resolve_polymer_param_source(artifact, explicit_charge_manifest, explicit_topology_ref)?;
    let templates = load_training_templates(&resolved.training_structure_path)?;
    if templates.len() < source_nmer {
        return Err(PackError::Invalid(format!(
            "training structure has {} residues but param_source.nmer={source_nmer}",
            templates.len()
        )));
    }
    let head = &templates[0];
    let repeat = &templates[templates.len() / 2];
    let tail = &templates[templates.len() - 1];
    let template_lookup = templates
        .iter()
        .map(|template| (template.resname.clone(), template))
        .collect::<BTreeMap<_, _>>();
    let source_axis = normalize(
        repeat
            .centroid
            .sub(head.centroid)
            .add(tail.centroid.sub(repeat.centroid))
            .scale(0.5),
    );
    let step_length = repeat
        .centroid
        .sub(head.centroid)
        .norm()
        .max(tail.centroid.sub(repeat.centroid).norm())
        .max(1.5);
    let (parent_of, parent_edge, visit_order) = graph_tree(node_specs.len(), edge_specs, root_idx)?;
    let collision_radii = node_specs
        .iter()
        .map(|spec| {
            template_lookup
                .get(&spec.template_resname)
                .copied()
                .unwrap_or(repeat)
        })
        .map(template_collision_radius)
        .collect::<Vec<_>>();
    let centroids = layout_graph_centroids(
        node_specs.len(),
        edge_specs,
        root_idx,
        step_length,
        conformation_mode,
        source_axis,
        build_seed,
        &collision_radii,
    )?;
    let mut node_neighbors = vec![Vec::<usize>::new(); node_specs.len()];
    let mut parent_edges = vec![Vec::<usize>::new(); node_specs.len()];
    let mut child_edges = vec![Vec::<usize>::new(); node_specs.len()];
    for (edge_idx, edge) in edge_specs.iter().enumerate() {
        node_neighbors[edge.parent].push(edge.child);
        node_neighbors[edge.child].push(edge.parent);
        parent_edges[edge.parent].push(edge_idx);
        child_edges[edge.child].push(edge_idx);
    }

    let mut atoms = Vec::new();
    let mut bonds = Vec::new();
    let mut ter_after = Vec::new();
    let mut tacticity_rng = StdRng::seed_from_u64(build_seed ^ 0x5eed_5eed);
    let mut edge_parent_attach = vec![None; edge_specs.len()];
    let mut edge_child_attach = vec![None; edge_specs.len()];
    let edge_ideal_distances = edge_specs
        .iter()
        .map(|edge| {
            graph_edge_ideal_distance(edge, &templates, node_specs, &template_lookup, repeat)
        })
        .collect::<PackResult<Vec<_>>>()?;
    let mut node_atom_indices = vec![Vec::new(); node_specs.len()];
    let mut rotatable_edges = Vec::new();

    for (idx, spec) in node_specs.iter().enumerate() {
        let template = template_lookup
            .get(&spec.template_resname)
            .copied()
            .unwrap_or(repeat);
        let atom_indices = template_atom_index_by_name(template);
        let mut removed_names = BTreeSet::new();
        for edge_idx in &parent_edges[idx] {
            removed_names.extend(atom_name_set(&edge_specs[*edge_idx].parent_leaving_atoms));
        }
        for edge_idx in &child_edges[idx] {
            removed_names.extend(atom_name_set(&edge_specs[*edge_idx].child_leaving_atoms));
        }
        let mut direction = Vec3::new(0.0, 0.0, 0.0);
        for neighbor in &node_neighbors[idx] {
            direction = direction.add(centroids[*neighbor].sub(centroids[idx]));
        }
        if direction.norm() <= 1.0e-6 {
            direction = source_axis;
        }
        let phase = tacticity_phase(tacticity_mode, idx, &mut tacticity_rng)?;
        let residue_start = atoms.len();
        let mut local_to_global = BTreeMap::new();
        for (local_idx, atom) in template.atoms.iter().enumerate() {
            if removed_names.contains(atom.name.trim()) {
                continue;
            }
            let local = atom.position.sub(template.centroid);
            let rotated = rotate_from_to(local, source_axis, direction);
            let rotated = rotate_about_axis(rotated, direction, phase);
            let mut built = atom.clone();
            built.position = centroids[idx].add(rotated);
            built.resid = (idx + 1) as i32;
            built.chain = 'A';
            built.resname = spec.template_resname.clone();
            built.mol_id = 1;
            atoms.push(built);
            let global_idx = residue_start + local_to_global.len();
            local_to_global.insert(local_idx, global_idx);
            node_atom_indices[idx].push(global_idx);
        }
        for &(a, b) in &template.local_bonds {
            let Some(&i) = local_to_global.get(&a) else {
                continue;
            };
            let Some(&j) = local_to_global.get(&b) else {
                continue;
            };
            let (i, j) = if i <= j { (i, j) } else { (j, i) };
            if i != j {
                bonds.push((i, j));
            }
        }
        for edge_idx in &parent_edges[idx] {
            let edge = &edge_specs[*edge_idx];
            let local_idx = atom_indices
                .get(edge.parent_attach_atom.trim())
                .copied()
                .ok_or_else(|| {
                    PackError::Invalid(format!(
                        "parent attach atom '{}' missing from template '{}'",
                        edge.parent_attach_atom, spec.template_resname
                    ))
                })?;
            edge_parent_attach[*edge_idx] =
                Some(local_to_global.get(&local_idx).copied().ok_or_else(|| {
                    PackError::Invalid(format!(
                        "parent attach atom '{}' was removed from template '{}'",
                        edge.parent_attach_atom, spec.template_resname
                    ))
                })?);
        }
        for edge_idx in &child_edges[idx] {
            let edge = &edge_specs[*edge_idx];
            let local_idx = atom_indices
                .get(edge.child_attach_atom.trim())
                .copied()
                .ok_or_else(|| {
                    PackError::Invalid(format!(
                        "child attach atom '{}' missing from template '{}'",
                        edge.child_attach_atom, spec.template_resname
                    ))
                })?;
            edge_child_attach[*edge_idx] =
                Some(local_to_global.get(&local_idx).copied().ok_or_else(|| {
                    PackError::Invalid(format!(
                        "child attach atom '{}' was removed from template '{}'",
                        edge.child_attach_atom, spec.template_resname
                    ))
                })?);
        }
        if let Some(last) = atoms.len().checked_sub(1) {
            ter_after.push(last);
        }
    }

    for &node_idx in visit_order.iter().skip(1) {
        let Some(edge_idx) = parent_edge[node_idx] else {
            continue;
        };
        let Some(parent_idx) = parent_of[node_idx] else {
            continue;
        };
        let Some(parent_attach_idx) = edge_parent_attach[edge_idx] else {
            continue;
        };
        let Some(child_attach_idx) = edge_child_attach[edge_idx] else {
            continue;
        };
        let current = atoms[child_attach_idx]
            .position
            .sub(atoms[parent_attach_idx].position);
        let fallback = centroids[node_idx].sub(centroids[parent_idx]);
        let direction = if current.norm() > 1.0e-6 {
            normalize(current)
        } else {
            normalize(fallback)
        };
        let desired = atoms[parent_attach_idx]
            .position
            .add(direction.scale(edge_ideal_distances[edge_idx]));
        let delta = desired.sub(atoms[child_attach_idx].position);
        if delta.norm() <= 1.0e-6 {
            continue;
        }
        for descendant_idx in 0..node_specs.len() {
            if !is_descendant(&parent_of, descendant_idx, node_idx) {
                continue;
            }
            for atom_idx in &node_atom_indices[descendant_idx] {
                atoms[*atom_idx].position = atoms[*atom_idx].position.add(delta);
            }
        }
    }

    for (edge_idx, _) in edge_specs.iter().enumerate() {
        let Some(parent_idx) = edge_parent_attach[edge_idx] else {
            continue;
        };
        let Some(child_idx) = edge_child_attach[edge_idx] else {
            continue;
        };
        let (i, j) = if parent_idx <= child_idx {
            (parent_idx, child_idx)
        } else {
            (child_idx, parent_idx)
        };
        if i != j {
            bonds.push((i, j));
        }
    }
    bonds.sort_unstable();
    bonds.dedup();

    for &node_idx in visit_order.iter().skip(1) {
        let Some(edge_idx) = parent_edge[node_idx] else {
            continue;
        };
        let Some(parent_attach_idx) = edge_parent_attach[edge_idx] else {
            continue;
        };
        let Some(child_attach_idx) = edge_child_attach[edge_idx] else {
            continue;
        };
        let movable_atom_indices = (0..node_specs.len())
            .filter(|descendant_idx| is_descendant(&parent_of, *descendant_idx, node_idx))
            .flat_map(|descendant_idx| node_atom_indices[descendant_idx].clone())
            .collect::<Vec<_>>();
        if movable_atom_indices.is_empty() {
            continue;
        }
        let edge = &edge_specs[edge_idx];
        rotatable_edges.push(RotatableEdge {
            parent_attach_idx,
            child_attach_idx,
            movable_atom_indices,
            torsion_mode: edge.torsion_mode.clone(),
            torsion_deg: edge.torsion_deg,
            torsion_window_deg: edge.torsion_window_deg,
        });
    }

    let mut output = PackOutput {
        atoms,
        bonds,
        box_size: [0.0, 0.0, 0.0],
        ter_after,
        box_vectors: None,
    };
    let qc_context = BuildQcContext {
        inter_residue_bond_count: edge_specs.len(),
        terminal_connectivity_consistent: !edge_specs.is_empty() || node_specs.len() <= 1,
        sequence_token_template_consistent: node_specs
            .iter()
            .all(|node| !node.template_resname.is_empty()),
        bond_expectations: edge_specs
            .iter()
            .enumerate()
            .filter_map(|(edge_idx, edge)| {
                let parent_idx = edge_parent_attach.get(edge_idx).and_then(|item| *item)?;
                let child_idx = edge_child_attach.get(edge_idx).and_then(|item| *item)?;
                let parent_atom = output.atoms.get(parent_idx)?.name.clone();
                let child_atom = output.atoms.get(child_idx)?.name.clone();
                Some(BuildBondExpectation {
                    edge_id: edge.edge_id.clone(),
                    parent_resid: edge.parent + 1,
                    child_resid: edge.child + 1,
                    parent_atom,
                    child_atom,
                    parent_idx,
                    child_idx,
                    expected_distance_angstrom: edge_ideal_distances
                        .get(edge_idx)
                        .copied()
                        .unwrap_or(1.5),
                })
            })
            .collect(),
    };
    let solver_report = if conformation_mode == "random_walk" {
        Some(solve_rotatable_edges(
            &mut output,
            &qc_context,
            &rotatable_edges,
            build_seed,
        )?)
    } else {
        None
    };
    let path = stage_polymer_output_path(final_coordinates_path);
    let path_text = path.to_string_lossy().to_string();
    if let Some(parent) = Path::new(&path_text).parent() {
        std::fs::create_dir_all(parent)?;
    }
    write_output(
        &output,
        &OutputSpec {
            path: path_text,
            format: "pdb".to_string(),
            scale: Some(1.0),
        },
        false,
        0.0,
        !output.bonds.is_empty(),
        false,
    )?;
    let qc_report = recompute_build_qc_report(&output, &qc_context);
    if strict_qc && conformation_mode != "random_walk" {
        ensure_build_qc_passes(&qc_report)?;
    }

    Ok(PolymerBuiltArtifact {
        path,
        step_length_angstrom: step_length,
        sequence_labels: node_specs
            .iter()
            .map(|item| item.sequence_label.clone())
            .collect(),
        template_sequence_resnames: node_specs
            .iter()
            .map(|item| item.template_resname.clone())
            .collect(),
        residue_resnames: node_specs
            .iter()
            .map(|item| item.applied_resname.clone())
            .collect(),
        output,
        qc_context,
        qc_report,
        solver_report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(label: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!(
            "warp_build_polymer_test_{}_{}_{}",
            label,
            std::process::id(),
            nanos
        ));
        path
    }

    #[test]
    fn polymer_prmtop_transfer_rebuilds_term_type_indices() {
        let source = AmberTopology {
            atom_names: vec![
                "H1".into(),
                "R1".into(),
                "R1".into(),
                "R1".into(),
                "R1".into(),
                "T1".into(),
            ],
            residue_labels: vec![
                "HDA".into(),
                "RPT".into(),
                "RPT".into(),
                "RPT".into(),
                "RPT".into(),
                "TLA".into(),
            ],
            residue_pointers: vec![1, 2, 3, 4, 5, 6],
            atomic_numbers: vec![6; 6],
            masses: vec![12.0; 6],
            charges: vec![0.5, 1.0, 1.0, 1.0, 1.0, 0.5],
            atom_type_indices: vec![1, 2, 2, 2, 2, 3],
            amber_atom_types: vec![
                "H".into(),
                "R".into(),
                "R".into(),
                "R".into(),
                "R".into(),
                "T".into(),
            ],
            radii: vec![1.5; 6],
            screen: vec![0.8; 6],
            bonds: vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)],
            bond_type_indices: vec![1, 2, 2, 2, 3],
            bond_force_constants: vec![100.0, 200.0, 300.0],
            bond_equil_values: vec![1.0, 1.1, 1.2],
            angles: vec![[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]],
            angle_type_indices: vec![1, 2, 2, 3],
            angle_force_constants: vec![10.0, 20.0, 30.0],
            angle_equil_values: vec![100.0, 110.0, 120.0],
            dihedrals: vec![[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]],
            dihedral_type_indices: vec![1, 2, 3],
            dihedral_force_constants: vec![1.0, 2.0, 3.0],
            dihedral_periodicities: vec![1.0, 2.0, 3.0],
            dihedral_phases: vec![0.0, 0.5, 1.0],
            scee_scale_factors: vec![1.2, 1.2, 1.2],
            scnb_scale_factors: vec![2.0, 2.0, 2.0],
            solty: vec![0.0, 0.0, 0.0],
            impropers: Vec::new(),
            improper_type_indices: Vec::new(),
            excluded_atoms: Vec::new(),
            nonbonded_parm_index: vec![1, 2, 2, 2, 2, 3, 2, 2, 2],
            lennard_jones_acoef: vec![1.0, 2.0, 3.0],
            lennard_jones_bcoef: vec![1.0, 2.0, 3.0],
            lennard_jones_14_acoef: vec![1.0, 2.0, 3.0],
            lennard_jones_14_bcoef: vec![1.0, 2.0, 3.0],
            hbond_acoef: vec![0.0],
            hbond_bcoef: vec![0.0],
            hbcut: vec![0.0],
            tree_chain_classification: vec!["M".into(); 6],
            join_array: vec![0; 6],
            irotat: vec![0; 6],
            solvent_pointers: Vec::new(),
            atoms_per_molecule: Vec::new(),
            box_dimensions: Vec::new(),
            radius_set: Some("test".into()),
            ipol: 0,
        };

        let rebuilt = rebuild_transferred_term_assignments(
            &source,
            &[
                "H1".into(),
                "R1".into(),
                "R1".into(),
                "R1".into(),
                "R1".into(),
                "R1".into(),
                "T1".into(),
            ],
            &[
                "H".into(),
                "R".into(),
                "R".into(),
                "R".into(),
                "R".into(),
                "R".into(),
                "T".into(),
            ],
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
        )
        .expect("rebuild transferred terms");

        assert_eq!(rebuilt.bond_type_indices, vec![1, 2, 2, 2, 2, 3]);
        assert_eq!(rebuilt.angle_type_indices, vec![1, 2, 2, 2, 3]);
        assert_eq!(rebuilt.dihedral_type_indices, vec![1, 2, 2, 3]);
    }

    #[test]
    fn resolve_polymer_param_source_accepts_mmcif_training_structures() {
        let dir = temp_path("mmcif_source");
        fs::create_dir_all(&dir).expect("create temp dir");
        let training = dir.join("training_oligomer.cif");
        fs::write(
            &training,
            "data_warp_build\n\
loop_\n\
_atom_site.group_PDB\n\
_atom_site.id\n\
_atom_site.type_symbol\n\
_atom_site.label_atom_id\n\
_atom_site.label_comp_id\n\
_atom_site.label_asym_id\n\
_atom_site.label_seq_id\n\
_atom_site.Cartn_x\n\
_atom_site.Cartn_y\n\
_atom_site.Cartn_z\n\
ATOM 1 C C1 HDA A 1 0.000 0.000 0.000\n\
ATOM 2 H H1 HDA A 1 1.090 0.000 0.000\n\
ATOM 3 C C2 RPT A 2 3.000 0.000 0.000\n\
ATOM 4 H H2 RPT A 2 4.090 0.000 0.000\n\
ATOM 5 C C3 TLA A 3 6.000 0.000 0.000\n\
ATOM 6 H H3 TLA A 3 7.090 0.000 0.000\n",
        )
        .expect("write mmcif");

        let resolved = resolve_polymer_param_source(dir.to_string_lossy().as_ref(), None, None)
            .expect("resolve mmcif source");
        let templates =
            load_training_templates(&resolved.training_structure_path).expect("load templates");

        assert_eq!(resolved.training_structure_path, training);
        assert_eq!(templates.len(), 3);
        assert_eq!(templates[0].local_bonds, vec![(0, 1)]);
        assert_eq!(templates[1].local_bonds, vec![(0, 1)]);
        assert_eq!(templates[2].local_bonds, vec![(0, 1)]);

        let _ = fs::remove_file(&training);
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn training_templates_merge_noncontiguous_residue_atoms() {
        let dir = temp_path("noncontiguous_residue_source");
        fs::create_dir_all(&dir).expect("create temp dir");
        let training = dir.join("training_oligomer.pdb");
        fs::write(
            &training,
            "ATOM      1  C1  HDA A   1       0.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  C2  RPT A   2       3.000   0.000   0.000  1.00  0.00           C\n\
ATOM      3  C3  TLA A   3       6.000   0.000   0.000  1.00  0.00           C\n\
ATOM      4  H1  HDA A   1       1.090   0.000   0.000  1.00  0.00           H\n\
ATOM      5  H2  RPT A   2       4.090   0.000   0.000  1.00  0.00           H\n\
ATOM      6  H3  TLA A   3       7.090   0.000   0.000  1.00  0.00           H\n\
CONECT    1    4\n\
CONECT    2    5\n\
CONECT    3    6\n\
END\n",
        )
        .expect("write pdb");

        let templates = load_training_templates(&training).expect("load templates");

        assert_eq!(templates.len(), 3);
        assert_eq!(
            templates[0]
                .atoms
                .iter()
                .map(|atom| atom.name.trim())
                .collect::<Vec<_>>(),
            vec!["C1", "H1"]
        );
        assert_eq!(templates[0].local_bonds, vec![(0, 1)]);
        assert_eq!(templates[1].local_bonds, vec![(0, 1)]);
        assert_eq!(templates[2].local_bonds, vec![(0, 1)]);

        let _ = fs::remove_file(&training);
        let _ = fs::remove_dir(&dir);
    }

    #[test]
    fn polymer_prmtop_transfer_falls_back_to_atom_type_signatures() {
        let source = AmberTopology {
            atom_names: vec!["C1".into(), "C2".into(), "C3".into(), "C4".into()],
            residue_labels: vec!["HDA".into(), "RPT".into(), "RPT".into(), "TLA".into()],
            residue_pointers: vec![1, 2, 3, 4],
            atomic_numbers: vec![6; 4],
            masses: vec![12.0; 4],
            charges: vec![0.0; 4],
            atom_type_indices: vec![1; 4],
            amber_atom_types: vec!["CT".into(), "CT".into(), "CT".into(), "CT".into()],
            radii: vec![1.5; 4],
            screen: vec![0.8; 4],
            bonds: vec![(0, 1), (1, 2), (2, 3)],
            bond_type_indices: vec![1, 1, 1],
            bond_force_constants: vec![100.0],
            bond_equil_values: vec![1.0],
            angles: vec![[0, 1, 2], [1, 2, 3]],
            angle_type_indices: vec![1, 1],
            angle_force_constants: vec![10.0],
            angle_equil_values: vec![100.0],
            dihedrals: vec![[0, 1, 2, 3]],
            dihedral_type_indices: vec![1],
            dihedral_force_constants: vec![1.0],
            dihedral_periodicities: vec![1.0],
            dihedral_phases: vec![0.0],
            scee_scale_factors: vec![1.2],
            scnb_scale_factors: vec![2.0],
            solty: vec![0.0],
            impropers: Vec::new(),
            improper_type_indices: Vec::new(),
            excluded_atoms: Vec::new(),
            nonbonded_parm_index: vec![1],
            lennard_jones_acoef: vec![1.0],
            lennard_jones_bcoef: vec![1.0],
            lennard_jones_14_acoef: vec![1.0],
            lennard_jones_14_bcoef: vec![1.0],
            hbond_acoef: vec![0.0],
            hbond_bcoef: vec![0.0],
            hbcut: vec![0.0],
            tree_chain_classification: vec!["M".into(); 4],
            join_array: vec![0; 4],
            irotat: vec![0; 4],
            solvent_pointers: Vec::new(),
            atoms_per_molecule: Vec::new(),
            box_dimensions: Vec::new(),
            radius_set: Some("test".into()),
            ipol: 0,
        };

        let rebuilt = rebuild_transferred_term_assignments(
            &source,
            &[
                "C1".into(),
                "C2".into(),
                "C2".into(),
                "C3".into(),
                "C4".into(),
            ],
            &[
                "CT".into(),
                "CT".into(),
                "CT".into(),
                "CT".into(),
                "CT".into(),
            ],
            &[(0, 1), (1, 2), (2, 3), (3, 4)],
        )
        .expect("rebuild transferred terms with type fallback");

        assert_eq!(rebuilt.bond_type_indices, vec![1, 1, 1, 1]);
        assert_eq!(rebuilt.angle_type_indices, vec![1, 1, 1]);
        assert_eq!(rebuilt.dihedral_type_indices, vec![1, 1]);
    }
}
