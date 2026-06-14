use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

pub const CHARGE_MANIFEST_VERSION: &str = "warp-build.charge-manifest.v1";
pub const LEGACY_CHARGE_MANIFEST_VERSION: &str = "warp-pack.charge-manifest.v1";

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

#[derive(Clone, Debug, PartialEq)]
pub struct NetChargeEstimate {
    pub net_charge_e: Option<f32>,
    pub source: Option<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComponentCharge {
    pub name: String,
    pub count: usize,
    pub per_instance_net_charge_e: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GromacsMoleculeTopology {
    pub molecule_type: String,
    pub atoms: Vec<GromacsAtomTopology>,
    pub constraints: Vec<GromacsConstraintTopology>,
    pub bonds: Vec<GromacsBondTopology>,
    pub angles: Vec<GromacsAngleTopology>,
    pub dihedrals: Vec<GromacsDihedralTopology>,
    pub virtual_sites: Vec<GromacsVirtualSiteTopology>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GromacsAtomTopology {
    pub nr: usize,
    pub atom_type: String,
    pub residue_nr: usize,
    pub residue_name: String,
    pub atom_name: String,
    pub charge_group: usize,
    pub charge_e: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GromacsBondTopology {
    pub atom_i: usize,
    pub atom_j: usize,
    pub function: usize,
    pub length_nm: Option<f32>,
    pub force_kj_mol_nm2: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GromacsConstraintTopology {
    pub atom_i: usize,
    pub atom_j: usize,
    pub function: usize,
    pub length_nm: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GromacsAngleTopology {
    pub atom_i: usize,
    pub atom_j: usize,
    pub atom_k: usize,
    pub function: usize,
    pub angle_degrees: Option<f32>,
    pub force_kj_mol_rad2: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GromacsDihedralTopology {
    pub atom_i: usize,
    pub atom_j: usize,
    pub atom_k: usize,
    pub atom_l: usize,
    pub function: usize,
    pub phase_degrees: Option<f32>,
    pub force_kj_mol: Option<f32>,
    pub multiplicity: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GromacsVirtualSiteTopology {
    pub site: usize,
    pub kind: String,
    pub function: usize,
    pub defining_atoms: Vec<usize>,
    pub parameters: Vec<f32>,
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

pub fn sum_atom_charges(charges: &[ChargeAtom]) -> f32 {
    charges.iter().map(|atom| atom.charge_e).sum()
}

pub fn sum_bead_charges(charges: &[f32]) -> f32 {
    charges.iter().sum()
}

pub fn spread_total_charge(total_charge_e: f32, bead_count: usize) -> Option<Vec<f32>> {
    if bead_count == 0 || !total_charge_e.is_finite() {
        return None;
    }
    Some(vec![total_charge_e / bead_count as f32; bead_count])
}

pub fn charges_match(expected_total_e: f32, charges: &[f32], tolerance_e: f32) -> bool {
    if tolerance_e < 0.0 || !expected_total_e.is_finite() {
        return false;
    }
    charges.iter().all(|charge| charge.is_finite())
        && (sum_bead_charges(charges) - expected_total_e).abs() <= tolerance_e
}

pub fn sum_component_charges(components: &[ComponentCharge]) -> Option<f32> {
    components.iter().try_fold(0.0f32, |total, component| {
        component
            .per_instance_net_charge_e
            .map(|charge| total + charge * component.count as f32)
    })
}

pub fn neutralizer_count(net_charge_e: f32, ion_valence_abs: f32) -> Option<usize> {
    if ion_valence_abs <= 0.0 || !ion_valence_abs.is_finite() || !net_charge_e.is_finite() {
        return None;
    }
    Some((net_charge_e.abs() / ion_valence_abs).ceil() as usize)
}

pub fn compute_gromacs_molecule_net_charge(
    topology_path: &Path,
    molecule_type: &str,
) -> Result<NetChargeEstimate, String> {
    let mut visited = BTreeSet::new();
    let payload = read_gromacs_topology_with_local_includes(topology_path, &mut visited)?;
    let charge = sum_gromacs_molecule_atom_charges(&payload, molecule_type)?;
    Ok(NetChargeEstimate {
        net_charge_e: Some(charge),
        source: Some(format!(
            "gromacs_topology:{}:{}",
            topology_path.display(),
            molecule_type
        )),
    })
}

pub fn read_gromacs_molecule_topology(
    topology_path: &Path,
    molecule_type: &str,
) -> Result<GromacsMoleculeTopology, String> {
    let mut visited = BTreeSet::new();
    let payload = read_gromacs_topology_with_local_includes(topology_path, &mut visited)?;
    parse_gromacs_molecule_topology(&payload, molecule_type)
}

fn read_gromacs_topology_with_local_includes(
    topology_path: &Path,
    visited: &mut BTreeSet<PathBuf>,
) -> Result<String, String> {
    let canonical = topology_path.canonicalize().map_err(|err| {
        format!(
            "failed to resolve topology '{}': {err}",
            topology_path.display()
        )
    })?;
    if !visited.insert(canonical.clone()) {
        return Err(format!(
            "topology include cycle detected at '{}'",
            topology_path.display()
        ));
    }
    let payload = std::fs::read_to_string(&canonical)
        .map_err(|err| format!("failed to read topology '{}': {err}", canonical.display()))?;
    let base_dir = canonical.parent().unwrap_or_else(|| Path::new("."));
    let mut expanded = String::new();
    for raw_line in payload.lines() {
        if let Some(include_path) = parse_gromacs_include_path(raw_line) {
            let resolved = base_dir.join(include_path);
            expanded.push_str(&read_gromacs_topology_with_local_includes(
                &resolved, visited,
            )?);
            expanded.push('\n');
        } else {
            expanded.push_str(raw_line);
            expanded.push('\n');
        }
    }
    visited.remove(&canonical);
    Ok(expanded)
}

fn parse_gromacs_include_path(line: &str) -> Option<&str> {
    let line = line.split(';').next().unwrap_or("").trim();
    let rest = line.strip_prefix("#include")?.trim();
    if rest.len() < 2 {
        return None;
    }
    if rest.starts_with('"') {
        let end = rest[1..].find('"')?;
        return Some(&rest[1..1 + end]);
    }
    if rest.starts_with('<') {
        return None;
    }
    None
}

pub fn sum_gromacs_molecule_atom_charges(
    topology: &str,
    molecule_type: &str,
) -> Result<f32, String> {
    let mut current_section = String::new();
    let mut current_molecule: Option<String> = None;
    let mut saw_target_molecule = false;
    let mut saw_target_atoms = false;
    let mut charge = 0.0f32;

    for (line_idx, raw_line) in topology.lines().enumerate() {
        let line = raw_line.split(';').next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            current_section = line
                .trim_matches(|ch| ch == '[' || ch == ']')
                .trim()
                .to_ascii_lowercase();
            continue;
        }

        match current_section.as_str() {
            "moleculetype" => {
                let Some(name) = line.split_whitespace().next() else {
                    continue;
                };
                current_molecule = Some(name.to_string());
                if name == molecule_type {
                    saw_target_molecule = true;
                }
            }
            "atoms" if current_molecule.as_deref() == Some(molecule_type) => {
                let fields = line.split_whitespace().collect::<Vec<_>>();
                if fields.len() < 7 {
                    return Err(format!(
                        "topology atoms line {} for molecule '{}' has fewer than 7 fields",
                        line_idx + 1,
                        molecule_type
                    ));
                }
                let atom_charge = fields[6].parse::<f32>().map_err(|err| {
                    format!(
                        "topology atoms line {} for molecule '{}' has invalid charge '{}': {err}",
                        line_idx + 1,
                        molecule_type,
                        fields[6]
                    )
                })?;
                charge += atom_charge;
                saw_target_atoms = true;
            }
            _ => {}
        }
    }

    if !saw_target_molecule {
        return Err(format!(
            "molecule type '{}' was not found in topology",
            molecule_type
        ));
    }
    if !saw_target_atoms {
        return Err(format!(
            "molecule type '{}' has no [ atoms ] charges in topology",
            molecule_type
        ));
    }
    Ok(charge)
}

pub fn parse_gromacs_molecule_topology(
    topology: &str,
    molecule_type: &str,
) -> Result<GromacsMoleculeTopology, String> {
    let mut current_section = String::new();
    let mut current_molecule: Option<String> = None;
    let mut saw_target_molecule = false;
    let mut atoms = Vec::new();
    let mut constraints = Vec::new();
    let mut bonds = Vec::new();
    let mut angles = Vec::new();
    let mut dihedrals = Vec::new();
    let mut virtual_sites = Vec::new();

    for (line_idx, raw_line) in topology.lines().enumerate() {
        let line = raw_line.split(';').next().unwrap_or("").trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            current_section = line
                .trim_matches(|ch| ch == '[' || ch == ']')
                .trim()
                .to_ascii_lowercase();
            continue;
        }

        match current_section.as_str() {
            "moleculetype" => {
                let Some(name) = line.split_whitespace().next() else {
                    continue;
                };
                current_molecule = Some(name.to_string());
                if name == molecule_type {
                    saw_target_molecule = true;
                }
            }
            "atoms" if current_molecule.as_deref() == Some(molecule_type) => {
                atoms.push(parse_gromacs_atom_topology_line(
                    line,
                    line_idx + 1,
                    molecule_type,
                )?);
            }
            "constraints" if current_molecule.as_deref() == Some(molecule_type) => {
                constraints.push(parse_gromacs_constraint_topology_line(
                    line,
                    line_idx + 1,
                    molecule_type,
                )?);
            }
            "bonds" if current_molecule.as_deref() == Some(molecule_type) => {
                bonds.push(parse_gromacs_bond_topology_line(
                    line,
                    line_idx + 1,
                    molecule_type,
                )?);
            }
            "angles" if current_molecule.as_deref() == Some(molecule_type) => {
                angles.push(parse_gromacs_angle_topology_line(
                    line,
                    line_idx + 1,
                    molecule_type,
                )?);
            }
            "dihedrals" if current_molecule.as_deref() == Some(molecule_type) => {
                dihedrals.push(parse_gromacs_dihedral_topology_line(
                    line,
                    line_idx + 1,
                    molecule_type,
                )?);
            }
            "virtual_sites2" if current_molecule.as_deref() == Some(molecule_type) => {
                virtual_sites.push(parse_gromacs_virtual_site_topology_line(
                    line,
                    line_idx + 1,
                    molecule_type,
                    "2",
                    2,
                    3,
                )?);
            }
            "virtual_sites3" if current_molecule.as_deref() == Some(molecule_type) => {
                virtual_sites.push(parse_gromacs_virtual_site_topology_line(
                    line,
                    line_idx + 1,
                    molecule_type,
                    "3",
                    3,
                    4,
                )?);
            }
            "virtual_sites4" if current_molecule.as_deref() == Some(molecule_type) => {
                virtual_sites.push(parse_gromacs_virtual_site_topology_line(
                    line,
                    line_idx + 1,
                    molecule_type,
                    "4",
                    4,
                    5,
                )?);
            }
            "virtual_sitesn" if current_molecule.as_deref() == Some(molecule_type) => {
                virtual_sites.push(parse_gromacs_virtual_sitesn_topology_line(
                    line,
                    line_idx + 1,
                    molecule_type,
                )?);
            }
            _ => {}
        }
    }

    if !saw_target_molecule {
        return Err(format!(
            "molecule type '{}' was not found in topology",
            molecule_type
        ));
    }
    if atoms.is_empty() {
        return Err(format!(
            "molecule type '{}' has no [ atoms ] topology",
            molecule_type
        ));
    }
    Ok(GromacsMoleculeTopology {
        molecule_type: molecule_type.to_string(),
        atoms,
        constraints,
        bonds,
        angles,
        dihedrals,
        virtual_sites,
    })
}

fn parse_gromacs_atom_topology_line(
    line: &str,
    line_number: usize,
    molecule_type: &str,
) -> Result<GromacsAtomTopology, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 7 {
        return Err(format!(
            "topology atoms line {line_number} for molecule '{molecule_type}' has fewer than 7 fields"
        ));
    }
    Ok(GromacsAtomTopology {
        nr: parse_gromacs_field(fields[0], "atom nr", line_number, molecule_type)?,
        atom_type: fields[1].to_string(),
        residue_nr: parse_gromacs_field(fields[2], "residue nr", line_number, molecule_type)?,
        residue_name: fields[3].to_string(),
        atom_name: fields[4].to_string(),
        charge_group: parse_gromacs_field(fields[5], "charge group", line_number, molecule_type)?,
        charge_e: parse_gromacs_field(fields[6], "charge", line_number, molecule_type)?,
    })
}

fn parse_gromacs_bond_topology_line(
    line: &str,
    line_number: usize,
    molecule_type: &str,
) -> Result<GromacsBondTopology, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 2 {
        return Err(format!(
            "topology bonds line {line_number} for molecule '{molecule_type}' has fewer than 2 fields"
        ));
    }
    Ok(GromacsBondTopology {
        atom_i: parse_gromacs_field(fields[0], "bond atom i", line_number, molecule_type)?,
        atom_j: parse_gromacs_field(fields[1], "bond atom j", line_number, molecule_type)?,
        function: fields
            .get(2)
            .map(|value| parse_gromacs_field(value, "bond function", line_number, molecule_type))
            .transpose()?
            .unwrap_or(1),
        length_nm: fields
            .get(3)
            .map(|value| parse_gromacs_field(value, "bond length", line_number, molecule_type))
            .transpose()?,
        force_kj_mol_nm2: fields
            .get(4)
            .map(|value| parse_gromacs_field(value, "bond force", line_number, molecule_type))
            .transpose()?,
    })
}

fn parse_gromacs_constraint_topology_line(
    line: &str,
    line_number: usize,
    molecule_type: &str,
) -> Result<GromacsConstraintTopology, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 2 {
        return Err(format!(
            "topology constraints line {line_number} for molecule '{molecule_type}' has fewer than 2 fields"
        ));
    }
    Ok(GromacsConstraintTopology {
        atom_i: parse_gromacs_field(fields[0], "constraint atom i", line_number, molecule_type)?,
        atom_j: parse_gromacs_field(fields[1], "constraint atom j", line_number, molecule_type)?,
        function: fields
            .get(2)
            .map(|value| {
                parse_gromacs_field(value, "constraint function", line_number, molecule_type)
            })
            .transpose()?
            .unwrap_or(1),
        length_nm: fields
            .get(3)
            .map(|value| {
                parse_gromacs_field(value, "constraint length", line_number, molecule_type)
            })
            .transpose()?,
    })
}

fn parse_gromacs_angle_topology_line(
    line: &str,
    line_number: usize,
    molecule_type: &str,
) -> Result<GromacsAngleTopology, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 3 {
        return Err(format!(
            "topology angles line {line_number} for molecule '{molecule_type}' has fewer than 3 fields"
        ));
    }
    Ok(GromacsAngleTopology {
        atom_i: parse_gromacs_field(fields[0], "angle atom i", line_number, molecule_type)?,
        atom_j: parse_gromacs_field(fields[1], "angle atom j", line_number, molecule_type)?,
        atom_k: parse_gromacs_field(fields[2], "angle atom k", line_number, molecule_type)?,
        function: fields
            .get(3)
            .map(|value| parse_gromacs_field(value, "angle function", line_number, molecule_type))
            .transpose()?
            .unwrap_or(2),
        angle_degrees: fields
            .get(4)
            .map(|value| parse_gromacs_field(value, "angle value", line_number, molecule_type))
            .transpose()?,
        force_kj_mol_rad2: fields
            .get(5)
            .map(|value| parse_gromacs_field(value, "angle force", line_number, molecule_type))
            .transpose()?,
    })
}

fn parse_gromacs_dihedral_topology_line(
    line: &str,
    line_number: usize,
    molecule_type: &str,
) -> Result<GromacsDihedralTopology, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 4 {
        return Err(format!(
            "topology dihedrals line {line_number} for molecule '{molecule_type}' has fewer than 4 fields"
        ));
    }
    Ok(GromacsDihedralTopology {
        atom_i: parse_gromacs_field(fields[0], "dihedral atom i", line_number, molecule_type)?,
        atom_j: parse_gromacs_field(fields[1], "dihedral atom j", line_number, molecule_type)?,
        atom_k: parse_gromacs_field(fields[2], "dihedral atom k", line_number, molecule_type)?,
        atom_l: parse_gromacs_field(fields[3], "dihedral atom l", line_number, molecule_type)?,
        function: fields
            .get(4)
            .map(|value| {
                parse_gromacs_field(value, "dihedral function", line_number, molecule_type)
            })
            .transpose()?
            .unwrap_or(1),
        phase_degrees: fields
            .get(5)
            .map(|value| parse_gromacs_field(value, "dihedral phase", line_number, molecule_type))
            .transpose()?,
        force_kj_mol: fields
            .get(6)
            .map(|value| parse_gromacs_field(value, "dihedral force", line_number, molecule_type))
            .transpose()?,
        multiplicity: fields
            .get(7)
            .map(|value| {
                parse_gromacs_field(value, "dihedral multiplicity", line_number, molecule_type)
            })
            .transpose()?,
    })
}

fn parse_gromacs_virtual_site_topology_line(
    line: &str,
    line_number: usize,
    molecule_type: &str,
    kind: &str,
    defining_count: usize,
    function_index: usize,
) -> Result<GromacsVirtualSiteTopology, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() <= function_index {
        return Err(format!(
            "topology virtual_sites{kind} line {line_number} for molecule '{molecule_type}' has too few fields"
        ));
    }
    let site = parse_gromacs_field(fields[0], "virtual site atom", line_number, molecule_type)?;
    let defining_atoms = fields[1..=defining_count]
        .iter()
        .map(|value| {
            parse_gromacs_field(
                value,
                "virtual site defining atom",
                line_number,
                molecule_type,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let function = parse_gromacs_field(
        fields[function_index],
        "virtual site function",
        line_number,
        molecule_type,
    )?;
    let parameters = fields[function_index + 1..]
        .iter()
        .map(|value| {
            parse_gromacs_field(value, "virtual site parameter", line_number, molecule_type)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GromacsVirtualSiteTopology {
        site,
        kind: kind.to_string(),
        function,
        defining_atoms,
        parameters,
    })
}

fn parse_gromacs_virtual_sitesn_topology_line(
    line: &str,
    line_number: usize,
    molecule_type: &str,
) -> Result<GromacsVirtualSiteTopology, String> {
    let fields = line.split_whitespace().collect::<Vec<_>>();
    if fields.len() < 3 {
        return Err(format!(
            "topology virtual_sitesn line {line_number} for molecule '{molecule_type}' has fewer than 3 fields"
        ));
    }
    let site = parse_gromacs_field(fields[0], "virtual site atom", line_number, molecule_type)?;
    let function = parse_gromacs_field(
        fields[1],
        "virtual site function",
        line_number,
        molecule_type,
    )?;
    let (defining_atoms, parameters) = if function == 3 {
        let tail = &fields[2..];
        if tail.len() % 2 != 0 {
            return Err(format!(
                "topology virtual_sitesn line {line_number} for molecule '{molecule_type}' has unmatched atom/weight pairs"
            ));
        }
        let mut atoms = Vec::new();
        let mut weights = Vec::new();
        for pair in tail.chunks(2) {
            atoms.push(parse_gromacs_field(
                pair[0],
                "virtual site defining atom",
                line_number,
                molecule_type,
            )?);
            weights.push(parse_gromacs_field(
                pair[1],
                "virtual site weight",
                line_number,
                molecule_type,
            )?);
        }
        (atoms, weights)
    } else {
        let atoms = fields[2..]
            .iter()
            .map(|value| {
                parse_gromacs_field(
                    value,
                    "virtual site defining atom",
                    line_number,
                    molecule_type,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        (atoms, Vec::new())
    };
    Ok(GromacsVirtualSiteTopology {
        site,
        kind: "n".to_string(),
        function,
        defining_atoms,
        parameters,
    })
}

fn parse_gromacs_field<T: std::str::FromStr>(
    value: &str,
    field: &str,
    line_number: usize,
    molecule_type: &str,
) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    value.parse::<T>().map_err(|err| {
        format!(
            "topology line {line_number} for molecule '{molecule_type}' has invalid {field} '{value}': {err}"
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn field_kinds_report_available_charge_sources() {
        let manifest = ChargeManifest {
            schema_version: CHARGE_MANIFEST_VERSION.to_string(),
            solute_path: None,
            topology_ref: None,
            source_topology_ref: None,
            target_topology_ref: None,
            forcefield_ref: None,
            charge_derivation: None,
            net_charge_e: Some(-1.0),
            atom_count: Some(2),
            partial_charges: None,
            atom_charges: Some(vec![ChargeAtom {
                index: 1,
                charge_e: -1.0,
            }]),
            head_charge_e: None,
            repeat_charge_e: Some(0.0),
            tail_charge_e: None,
        };

        assert_eq!(
            charge_manifest_field_kinds(&manifest),
            vec!["net_charge_e", "atom_charges", "repeat_scalars"]
        );
    }

    #[test]
    fn component_charge_sum_requires_every_component_charge() {
        let complete = vec![
            ComponentCharge {
                name: "lipid".into(),
                count: 10,
                per_instance_net_charge_e: Some(-1.0),
            },
            ComponentCharge {
                name: "protein".into(),
                count: 1,
                per_instance_net_charge_e: Some(3.0),
            },
        ];
        assert_eq!(sum_component_charges(&complete), Some(-7.0));

        let missing = vec![ComponentCharge {
            name: "unknown".into(),
            count: 1,
            per_instance_net_charge_e: None,
        }];
        assert_eq!(sum_component_charges(&missing), None);
    }

    #[test]
    fn neutralizer_count_ceilings_by_abs_valence() {
        assert_eq!(neutralizer_count(-3.1, 2.0), Some(2));
        assert_eq!(neutralizer_count(3.0, 1.0), Some(3));
        assert_eq!(neutralizer_count(1.0, 0.0), None);
    }

    #[test]
    fn total_charge_spreads_across_beads() {
        let charges = spread_total_charge(-1.0, 4).unwrap();
        assert_eq!(charges, vec![-0.25; 4]);
        assert!(charges_match(-1.0, &charges, 1.0e-6));
        assert_eq!(spread_total_charge(1.0, 0), None);
    }

    #[test]
    fn gromacs_topology_charge_sums_selected_molecule_atoms() {
        let topology = r#"
[ moleculetype ]
; name nrexcl
  AAA 1

[ atoms ]
; nr type resnr residue atom cgnr charge mass
  1 P1 1 AAA A 1 -0.5 72
  2 P1 1 AAA B 2  0.25 72

[ moleculetype ]
  BBB 1

[ atoms ]
  1 P1 1 BBB A 1 1.0
  2 P1 1 BBB B 2 1.0
"#;
        assert_eq!(
            sum_gromacs_molecule_atom_charges(topology, "AAA").unwrap(),
            -0.25
        );
        assert_eq!(
            sum_gromacs_molecule_atom_charges(topology, "BBB").unwrap(),
            2.0
        );
    }

    #[test]
    fn gromacs_topology_reads_atoms_bonds_angles_and_dihedrals_for_selected_molecule() {
        let topology = r#"
[ moleculetype ]
  SOL 1

[ atoms ]
  1 P1 1 SOL A 1 -0.5
  2 P2 1 SOL B 2  0.25
  3 P3 1 SOL C 3  0.25
  4 P4 1 SOL D 4  0.00

[ constraints ]
  1 2 1 0.47

[ bonds ]
  1 2 1 0.47 1250

[ angles ]
  1 2 3 2 150.0 100.0

[ dihedrals ]
  1 2 3 4 1 180.0 5.0 2

[ virtual_sites2 ]
  4 1 2 1 0.5
"#;

        let molecule = parse_gromacs_molecule_topology(topology, "SOL").unwrap();

        assert_eq!(molecule.molecule_type, "SOL");
        assert_eq!(molecule.atoms.len(), 4);
        assert_eq!(molecule.atoms[0].atom_type, "P1");
        assert_eq!(molecule.atoms[1].atom_name, "B");
        assert_eq!(molecule.atoms[1].charge_e, 0.25);
        assert_eq!(
            molecule.constraints,
            vec![GromacsConstraintTopology {
                atom_i: 1,
                atom_j: 2,
                function: 1,
                length_nm: Some(0.47),
            }]
        );
        assert_eq!(
            molecule.bonds,
            vec![GromacsBondTopology {
                atom_i: 1,
                atom_j: 2,
                function: 1,
                length_nm: Some(0.47),
                force_kj_mol_nm2: Some(1250.0),
            }]
        );
        assert_eq!(
            molecule.angles,
            vec![GromacsAngleTopology {
                atom_i: 1,
                atom_j: 2,
                atom_k: 3,
                function: 2,
                angle_degrees: Some(150.0),
                force_kj_mol_rad2: Some(100.0),
            }]
        );
        assert_eq!(
            molecule.dihedrals,
            vec![GromacsDihedralTopology {
                atom_i: 1,
                atom_j: 2,
                atom_k: 3,
                atom_l: 4,
                function: 1,
                phase_degrees: Some(180.0),
                force_kj_mol: Some(5.0),
                multiplicity: Some(2),
            }]
        );
        assert_eq!(
            molecule.virtual_sites,
            vec![GromacsVirtualSiteTopology {
                site: 4,
                kind: "2".to_string(),
                function: 1,
                defining_atoms: vec![1, 2],
                parameters: vec![0.5],
            }]
        );
    }

    #[test]
    fn gromacs_topology_skips_preprocessor_lines_inside_sections() {
        let topology = r#"
[ moleculetype ]
  SOL 1

[ atoms ]
  1 P1 1 SOL A 1 -0.5
  2 P2 1 SOL B 2  0.5

[ bonds ]
#ifdef FLEXIBLE
  1 2 1 0.47 1250
#endif
"#;

        let molecule = parse_gromacs_molecule_topology(topology, "SOL").unwrap();

        assert_eq!(molecule.bonds.len(), 1);
        assert_eq!(molecule.bonds[0].atom_i, 1);
        assert_eq!(molecule.bonds[0].atom_j, 2);
        assert_eq!(
            sum_gromacs_molecule_atom_charges(topology, "SOL").unwrap(),
            0.0
        );
    }

    #[test]
    fn gromacs_topology_reads_virtual_sites_sections() {
        let topology = r#"
[ moleculetype ]
  SOL 1

[ atoms ]
  1 P1 1 SOL A 1 0
  2 P2 1 SOL B 2 0
  3 P3 1 SOL C 3 0
  4 P4 1 SOL D 4 0
  5 vP1 1 SOL V1 5 0
  6 vP2 1 SOL V2 6 0
  7 vP3 1 SOL V3 7 0
  8 vP4 1 SOL V4 8 0

[ virtual_sites2 ]
  5 1 2 1 0.25

[ virtual_sites3 ]
  6 1 2 3 4 0.1 0.2 0.3

[ virtual_sites4 ]
  7 1 2 3 4 2 0.1 0.2 0.3

[ virtual_sitesn ]
  8 3 1 0.5 2 0.25 3 0.25
"#;

        let molecule = parse_gromacs_molecule_topology(topology, "SOL").unwrap();

        assert_eq!(molecule.virtual_sites.len(), 4);
        assert_eq!(
            molecule.virtual_sites[0],
            GromacsVirtualSiteTopology {
                site: 5,
                kind: "2".to_string(),
                function: 1,
                defining_atoms: vec![1, 2],
                parameters: vec![0.25],
            }
        );
        assert_eq!(molecule.virtual_sites[1].kind, "3");
        assert_eq!(molecule.virtual_sites[1].defining_atoms, vec![1, 2, 3]);
        assert_eq!(molecule.virtual_sites[1].parameters, vec![0.1, 0.2, 0.3]);
        assert_eq!(molecule.virtual_sites[2].kind, "4");
        assert_eq!(molecule.virtual_sites[2].defining_atoms, vec![1, 2, 3, 4]);
        assert_eq!(molecule.virtual_sites[3].kind, "n");
        assert_eq!(molecule.virtual_sites[3].function, 3);
        assert_eq!(molecule.virtual_sites[3].defining_atoms, vec![1, 2, 3]);
        assert_eq!(molecule.virtual_sites[3].parameters, vec![0.5, 0.25, 0.25]);
    }

    #[test]
    fn gromacs_topology_charge_follows_local_includes() {
        let temp = temp_test_dir("gromacs_topology_charge_follows_local_includes");
        let main = temp.join("topol.top");
        let included = temp.join("solute.itp");
        std::fs::write(
            &included,
            r#"
[ moleculetype ]
  SOL 1

[ atoms ]
  1 P1 1 SOL A 1 -0.5
  2 P1 1 SOL B 2 -0.5
"#,
        )
        .unwrap();
        std::fs::write(
            &main,
            r#"
#include "solute.itp"

[ system ]
included charge topology
"#,
        )
        .unwrap();

        let estimate = compute_gromacs_molecule_net_charge(&main, "SOL").unwrap();

        assert_eq!(estimate.net_charge_e, Some(-1.0));
        std::fs::remove_dir_all(temp).unwrap();
    }

    #[test]
    fn gromacs_topology_include_cycles_are_rejected() {
        let temp = temp_test_dir("gromacs_topology_include_cycles_are_rejected");
        let main = temp.join("topol.top");
        let included = temp.join("loop.itp");
        std::fs::write(&main, "#include \"loop.itp\"\n").unwrap();
        std::fs::write(&included, "#include \"topol.top\"\n").unwrap();

        let error = compute_gromacs_molecule_net_charge(&main, "SOL").unwrap_err();

        assert!(error.contains("topology include cycle detected"));
        std::fs::remove_dir_all(temp).unwrap();
    }

    fn temp_test_dir(name: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "warp_common_charge_{name}_{}_{}",
            std::process::id(),
            nanos
        ));
        std::fs::create_dir_all(&path).unwrap();
        path
    }
}
