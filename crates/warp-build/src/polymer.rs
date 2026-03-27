use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use rand::{rngs::StdRng, Rng, SeedableRng};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use traj_core::{parse_pdb_reader, PdbParseOptions, PdbParseResult, PdbRecordKind, Vec3};
use warp_pack::io::{
    read_prmtop_atom_charges, read_prmtop_topology, read_prmtop_total_charge, write_minimal_prmtop,
    write_output, AmberTopology,
};
use warp_pack::pack::AtomRecordKind;
use warp_pack::{AtomRecord, OutputSpec, PackError, PackOutput, PackResult};

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
    pub artifact_path: PathBuf,
    pub training_structure_path: PathBuf,
    pub topology_ref: Option<String>,
    pub charge_manifest_path: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct PolymerBuiltArtifact {
    pub path: PathBuf,
    pub source_training_structure_path: PathBuf,
    pub source_nmer: usize,
    pub target_n_repeat: usize,
    pub conformation_mode: String,
    pub tacticity_mode: String,
    pub step_length_angstrom: f32,
    pub residue_count: usize,
    pub template_resnames: BTreeMap<String, String>,
    pub applied_resnames: BTreeMap<String, String>,
    pub sequence_label: Option<String>,
    pub sequence_labels: Vec<String>,
    pub template_sequence_resnames: Vec<String>,
    pub residue_resnames: Vec<String>,
    pub output: PackOutput,
    pub qc_context: BuildQcContext,
    pub qc_report: BuildQcReport,
    pub solver_report: Option<BuildSolverReport>,
    pub source_charge_manifest_path: Option<PathBuf>,
    pub topology_ref: Option<String>,
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
    pub parent_port: String,
    pub child_port: String,
    pub parent_attach_atom: String,
    pub parent_leaving_atoms: Vec<String>,
    pub child_attach_atom: String,
    pub child_leaving_atoms: Vec<String>,
    pub bond_order: u8,
    pub layout_mode: String,
    pub branch_spread: String,
    pub torsion_mode: String,
    pub torsion_deg: Option<f32>,
    pub torsion_window_deg: Option<[f32; 2]>,
    pub ring_mode: Option<String>,
}

#[derive(Clone, Debug)]
struct ResidueTemplate {
    resname: String,
    atoms: Vec<AtomRecord>,
    centroid: Vec3,
    local_bonds: Vec<(usize, usize)>,
}

#[derive(Clone, Debug)]
struct BuildResidueSpec {
    sequence_label: String,
    template_resname: String,
    applied_resname: String,
}

fn normalize(v: Vec3) -> Vec3 {
    let norm = v.norm();
    if norm <= 1.0e-12 {
        return Vec3::new(1.0, 0.0, 0.0);
    }
    v.scale(1.0 / norm)
}

fn centroid_of_atoms(atoms: &[AtomRecord]) -> Vec3 {
    if atoms.is_empty() {
        return Vec3::new(0.0, 0.0, 0.0);
    }
    let mut center = Vec3::new(0.0, 0.0, 0.0);
    for atom in atoms {
        center = center.add(atom.position);
    }
    center.scale(1.0 / atoms.len() as f32)
}

fn rotate_from_to(v: Vec3, source_axis: Vec3, target_axis: Vec3) -> Vec3 {
    let source = normalize(source_axis);
    let target = normalize(target_axis);
    let dot = source.dot(target).clamp(-1.0, 1.0);
    if dot > 1.0 - 1.0e-6 {
        return v;
    }
    if dot < -1.0 + 1.0e-6 {
        let basis = if source.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let axis = normalize(source.cross(basis));
        return axis.scale(2.0 * axis.dot(v)).sub(v);
    }
    let axis = normalize(source.cross(target));
    let theta = dot.acos();
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    v.scale(cos_t)
        .add(axis.cross(v).scale(sin_t))
        .add(axis.scale(axis.dot(v) * (1.0 - cos_t)))
}

fn rotate_about_axis(v: Vec3, axis: Vec3, theta: f32) -> Vec3 {
    if theta.abs() <= 1.0e-8 {
        return v;
    }
    let axis = normalize(axis);
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    v.scale(cos_t)
        .add(axis.cross(v).scale(sin_t))
        .add(axis.scale(axis.dot(v) * (1.0 - cos_t)))
}

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

fn group_residues(parsed: PdbParseResult) -> PackResult<Vec<ResidueTemplate>> {
    let mut grouped = Vec::new();
    let mut current_key: Option<(char, i32)> = None;
    let mut current_atoms = Vec::new();
    let mut current_resname = String::new();
    let mut residue_starts = Vec::new();
    let mut residue_ends = Vec::new();
    let atoms = parsed.atoms;
    let bonds = parsed.bonds;

    for (global_idx, atom) in atoms.into_iter().enumerate() {
        let key = (atom.chain, atom.resid);
        if current_key.is_none() {
            current_key = Some(key);
            residue_starts.push(global_idx);
        }
        if Some(key) != current_key {
            let centroid = centroid_of_atoms(&current_atoms);
            residue_ends.push(global_idx);
            grouped.push(ResidueTemplate {
                resname: current_resname.clone(),
                atoms: current_atoms,
                centroid,
                local_bonds: Vec::new(),
            });
            current_atoms = Vec::new();
            current_key = Some(key);
            residue_starts.push(global_idx);
        }
        current_resname = if atom.resname.trim().is_empty() {
            "MOL".to_string()
        } else {
            atom.resname.clone()
        };
        current_atoms.push(AtomRecord {
            record_kind: match atom.record_kind {
                PdbRecordKind::Atom => AtomRecordKind::Atom,
                PdbRecordKind::HetAtom => AtomRecordKind::HetAtom,
            },
            name: atom.name,
            element: atom.element,
            resname: current_resname.clone(),
            resid: atom.resid,
            chain: if atom.chain == ' ' { 'A' } else { atom.chain },
            segid: atom.segid,
            charge: 0.0,
            position: Vec3::new(atom.position[0], atom.position[1], atom.position[2]),
            mol_id: 1,
        });
    }

    if !current_atoms.is_empty() {
        let centroid = centroid_of_atoms(&current_atoms);
        residue_ends.push(atoms_len_from_grouped(&grouped) + current_atoms.len());
        grouped.push(ResidueTemplate {
            resname: current_resname,
            atoms: current_atoms,
            centroid,
            local_bonds: Vec::new(),
        });
    }

    if grouped.len() < 3 {
        return Err(PackError::Invalid(
            "training oligomer must contain at least 3 residues".into(),
        ));
    }

    for (residue_idx, template) in grouped.iter_mut().enumerate() {
        let start = residue_starts[residue_idx];
        let end = residue_ends[residue_idx];
        let mut local_bonds = Vec::new();
        for &(a, b) in &bonds {
            if a >= start && a < end && b >= start && b < end {
                let i = a - start;
                let j = b - start;
                let (i, j) = if i <= j { (i, j) } else { (j, i) };
                if i != j {
                    local_bonds.push((i, j));
                }
            }
        }
        local_bonds.sort_unstable();
        local_bonds.dedup();
        template.local_bonds = local_bonds;
    }

    Ok(grouped)
}

fn atoms_len_from_grouped(grouped: &[ResidueTemplate]) -> usize {
    grouped.iter().map(|template| template.atoms.len()).sum()
}

fn load_training_templates(path: &Path) -> PackResult<Vec<ResidueTemplate>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let parsed = parse_pdb_reader(
        reader,
        &PdbParseOptions {
            include_conect: true,
            non_standard_conect: true,
            include_ter: true,
            strict: false,
            only_first_model: true,
        },
    )
    .map_err(|err| PackError::Parse(err.to_string()))?;
    group_residues(parsed)
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
            PackError::Invalid(format!(
                "attach atom '{}' missing from template '{}'",
                atom_name, template.resname
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

fn placement_accepts(
    candidate: Vec3,
    existing: &[Option<Vec3>],
    parent: Option<usize>,
    step: f32,
) -> bool {
    let min_distance = (step * 0.72).max(1.6);
    for (idx, prev) in existing.iter().enumerate() {
        let Some(prev) = prev else {
            continue;
        };
        if Some(idx) == parent {
            continue;
        }
        if candidate.sub(*prev).norm() < min_distance {
            return false;
        }
    }
    true
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

fn attachment_local_position(template: &ResidueTemplate, atom_name: &str) -> PackResult<Vec3> {
    let atom = template
        .atoms
        .iter()
        .find(|atom| atom.name.trim() == atom_name.trim())
        .ok_or_else(|| {
            PackError::Invalid(format!(
                "attach atom '{}' missing from template '{}'",
                atom_name, template.resname
            ))
        })?;
    Ok(atom.position.sub(template.centroid))
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
    let bonded_pairs = bonds
        .iter()
        .map(|&(a, b)| if a <= b { (a, b) } else { (b, a) })
        .collect::<BTreeSet<_>>();
    let mut min_nonbonded: Option<f32> = None;
    let mut severe_clash_examples = Vec::new();
    for left in 0..atoms.len() {
        for right in (left + 1)..atoms.len() {
            if bonded_pairs.contains(&(left, right)) {
                continue;
            }
            let distance = atoms[right].position.sub(atoms[left].position).norm();
            min_nonbonded = Some(match min_nonbonded {
                Some(current) => current.min(distance),
                None => distance,
            });
            if distance < 0.8 && severe_clash_examples.len() < 8 {
                severe_clash_examples.push(ClashQcViolation {
                    atom_a: left + 1,
                    atom_b: right + 1,
                    distance_angstrom: distance,
                });
            }
        }
    }

    BuildQcReport {
        inter_residue_bond_count: context.inter_residue_bond_count,
        terminal_connectivity_consistent: context.terminal_connectivity_consistent,
        sequence_token_template_consistent: context.sequence_token_template_consistent,
        min_nonbonded_distance_angstrom: min_nonbonded,
        severe_nonbonded_clash_count: severe_clash_examples.len(),
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
            let preferred = preferred_branch_direction(
                parent_dir,
                sibling_idx,
                children.len(),
                depth,
                &edge.branch_spread,
                resolved_torsion_angle(edge, &mut rng),
            );
            let (chosen_dir, chosen_pos) = if conformation_mode == "random_walk" {
                let mut placed = None;
                for _ in 0..512 {
                    let dir = perturb_direction(&mut rng, preferred, 0.45);
                    let candidate = parent_pos.add(dir.scale(step_length));
                    if placement_accepts(candidate, &positions, Some(node_idx), step_length) {
                        placed = Some((dir, candidate));
                        break;
                    }
                }
                placed.unwrap_or((preferred, parent_pos.add(preferred.scale(step_length))))
            } else {
                (preferred, parent_pos.add(preferred.scale(step_length)))
            };
            incoming_dirs[*child_idx] = chosen_dir;
            positions[*child_idx] = Some(chosen_pos);
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
        for left in 0..node_count {
            for right in (left + 1)..node_count {
                let diff = positions[right].sub(positions[left]);
                let dist = diff.norm().max(1.0e-4);
                if dist >= min_distance {
                    continue;
                }
                let dir = diff.scale(1.0 / dist);
                let push = dir.scale(0.08 * (min_distance - dist));
                deltas[left] = deltas[left].sub(push);
                deltas[right] = deltas[right].add(push);
            }
        }
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
                artifact_path: artifact_path.clone(),
                training_structure_path: base.join(bundle.training_structure),
                topology_ref: explicit_topology_ref
                    .map(ToOwned::to_owned)
                    .or(bundle.topology_ref),
                charge_manifest_path: explicit_charge_manifest
                    .map(PathBuf::from)
                    .or_else(|| bundle.charge_manifest.map(|value| base.join(value))),
            });
        }
        if ext == "pdb" {
            return Ok(PolymerSourceResolved {
                artifact_path: artifact_path.clone(),
                training_structure_path: artifact_path.clone(),
                topology_ref: explicit_topology_ref.map(ToOwned::to_owned),
                charge_manifest_path: explicit_charge_manifest.map(PathBuf::from),
            });
        }
        return Err(PackError::Invalid(
            "polymer param artifact must be a .json bundle or .pdb training structure".into(),
        ));
    }

    if artifact_path.is_dir() {
        let candidates = [
            artifact_path.join("polymer_param.json"),
            artifact_path.join("polymer_source.json"),
            artifact_path.join("training_oligomer.pdb"),
            artifact_path.join("oligomer.pdb"),
            artifact_path.join("source.pdb"),
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
    )
}

fn stage_polymer_output_path(final_coordinates_path: &str) -> PathBuf {
    let final_path = PathBuf::from(final_coordinates_path);
    let parent = final_path.parent().unwrap_or_else(|| Path::new("."));
    let stem = final_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("system");
    parent.join(format!("{stem}_built_solute.pdb"))
}

fn build_centroids(
    n_repeat: usize,
    step_length: f32,
    conformation_mode: &str,
    source_axis: Vec3,
    seed: u64,
) -> PackResult<Vec<Vec3>> {
    let axis = normalize(source_axis);
    if conformation_mode == "extended" {
        let mut centroids = Vec::with_capacity(n_repeat);
        for idx in 0..n_repeat {
            centroids.push(axis.scale(step_length * idx as f32));
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
    let min_nonbonded = (step_length * 0.70).max(1.8);
    let persistence = 0.72f32;
    let max_bend_cos = (-0.35f32).max(-1.0);

    while centroids.len() < n_repeat {
        let mut placed = false;
        let prev_dir = *directions.last().unwrap_or(&axis);
        for _ in 0..768 {
            let random_dir = normalize(Vec3::new(
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
                rng.gen_range(-1.0..1.0),
            ));
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
            let mut clashes = false;
            for (prev_idx, prev) in centroids
                .iter()
                .enumerate()
                .take(centroids.len().saturating_sub(2))
            {
                let distance = candidate.sub(*prev).norm();
                let required = if prev_idx + 3 >= centroids.len() {
                    min_nonbonded * 0.92
                } else {
                    min_nonbonded
                };
                if distance < required {
                    clashes = true;
                    break;
                }
            }
            if clashes {
                continue;
            }
            centroids.push(candidate);
            directions.push(dir);
            placed = true;
            break;
        }
        if !placed {
            let fallback = normalize(prev_dir.add(axis.scale(0.35)));
            let dir = if fallback.norm() <= 1.0e-6 {
                prev_dir
            } else {
                fallback
            };
            let candidate = centroids
                .last()
                .copied()
                .unwrap_or(Vec3::new(0.0, 0.0, 0.0))
                .add(dir.scale(step_length));
            centroids.push(candidate);
            directions.push(dir);
        }
    }

    Ok(centroids)
}

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
    let centroids = build_centroids(
        sequence_specs.len(),
        step_length,
        conformation_mode,
        source_axis,
        build_seed,
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
            built.resname = spec.applied_resname.clone();
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

    let mut template_resnames = BTreeMap::new();
    template_resnames.insert(
        "head".into(),
        sequence_specs
            .first()
            .map(|item| item.template_resname.clone())
            .unwrap_or_else(|| head.resname.clone()),
    );
    template_resnames.insert(
        "repeat".into(),
        sequence_specs
            .get(sequence_specs.len().saturating_sub(1).min(1))
            .map(|item| item.template_resname.clone())
            .unwrap_or_else(|| repeat.resname.clone()),
    );
    template_resnames.insert(
        "tail".into(),
        sequence_specs
            .last()
            .map(|item| item.template_resname.clone())
            .unwrap_or_else(|| tail.resname.clone()),
    );
    let mut applied_resnames = BTreeMap::new();
    applied_resnames.insert(
        "head".into(),
        sequence_specs
            .first()
            .map(|item| item.applied_resname.clone())
            .unwrap_or_else(|| head.resname.clone()),
    );
    applied_resnames.insert(
        "repeat".into(),
        sequence_specs
            .get(sequence_specs.len().saturating_sub(1).min(1))
            .map(|item| item.applied_resname.clone())
            .unwrap_or_else(|| repeat.resname.clone()),
    );
    applied_resnames.insert(
        "tail".into(),
        sequence_specs
            .last()
            .map(|item| item.applied_resname.clone())
            .unwrap_or_else(|| tail.resname.clone()),
    );

    Ok(PolymerBuiltArtifact {
        path,
        source_training_structure_path: resolved.training_structure_path,
        source_nmer,
        target_n_repeat: sequence_specs.len(),
        conformation_mode: conformation_mode.to_string(),
        tacticity_mode: tacticity_mode.to_string(),
        step_length_angstrom: step_length,
        residue_count: sequence_specs.len(),
        template_resnames,
        applied_resnames,
        sequence_label: if sequence_labels
            .windows(2)
            .all(|window| window[0] == window[1])
        {
            sequence_labels.first().cloned()
        } else {
            None
        },
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
        source_charge_manifest_path: resolved.charge_manifest_path,
        topology_ref: resolved.topology_ref,
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
    let centroids = layout_graph_centroids(
        node_specs.len(),
        edge_specs,
        root_idx,
        step_length,
        conformation_mode,
        source_axis,
        build_seed,
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
            built.resname = spec.applied_resname.clone();
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
    let qc_report = recompute_build_qc_report(&output, &qc_context);
    if conformation_mode != "random_walk" {
        ensure_build_qc_passes(&qc_report)?;
    }
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

    let mut template_resnames = BTreeMap::new();
    template_resnames.insert(
        "head".into(),
        node_specs
            .first()
            .map(|item| item.template_resname.clone())
            .unwrap_or_else(|| head.resname.clone()),
    );
    template_resnames.insert(
        "repeat".into(),
        node_specs
            .get(node_specs.len().saturating_sub(1).min(1))
            .map(|item| item.template_resname.clone())
            .unwrap_or_else(|| repeat.resname.clone()),
    );
    template_resnames.insert(
        "tail".into(),
        node_specs
            .last()
            .map(|item| item.template_resname.clone())
            .unwrap_or_else(|| tail.resname.clone()),
    );
    let mut applied_resnames = BTreeMap::new();
    applied_resnames.insert(
        "head".into(),
        node_specs
            .first()
            .map(|item| item.applied_resname.clone())
            .unwrap_or_else(|| head.resname.clone()),
    );
    applied_resnames.insert(
        "repeat".into(),
        node_specs
            .get(node_specs.len().saturating_sub(1).min(1))
            .map(|item| item.applied_resname.clone())
            .unwrap_or_else(|| repeat.resname.clone()),
    );
    applied_resnames.insert(
        "tail".into(),
        node_specs
            .last()
            .map(|item| item.applied_resname.clone())
            .unwrap_or_else(|| tail.resname.clone()),
    );

    Ok(PolymerBuiltArtifact {
        path,
        source_training_structure_path: resolved.training_structure_path,
        source_nmer,
        target_n_repeat: node_specs.len(),
        conformation_mode: conformation_mode.to_string(),
        tacticity_mode: tacticity_mode.to_string(),
        step_length_angstrom: step_length,
        residue_count: node_specs.len(),
        template_resnames,
        applied_resnames,
        sequence_label: if node_specs
            .windows(2)
            .all(|window| window[0].sequence_label == window[1].sequence_label)
        {
            node_specs.first().map(|item| item.sequence_label.clone())
        } else {
            None
        },
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
        source_charge_manifest_path: resolved.charge_manifest_path,
        topology_ref: resolved.topology_ref,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
