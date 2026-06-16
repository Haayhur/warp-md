use serde_json::{json, Value};

use crate::bonded_terms::BondedTermSet;
use crate::mapping::MappingResult;
use crate::optimize::OptimizationReport;
use crate::parameters::{AngleStats, BondStats, DihedralStats};
use crate::reference::{
    ReferenceBinConfig, ReferenceDistributionTarget, ReferenceTargetSet, ReferenceTermKind,
};

use super::{mapping_mode, CgBead, CgRequest, SourceMappingResult, AGENT_SCHEMA_VERSION};

pub(super) fn mapping_json(request: &CgRequest, mapping: &MappingResult) -> Value {
    json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "kind": "martini_mapping",
        "name": request.name,
        "smiles": request.smiles,
        "repeat_smiles": request.repeat_smiles,
        "source": request.source,
        "mapping_mode": mapping_mode(request),
        "bead_count": mapping.bead_names.len(),
        "beads": beads(mapping),
        "connections": mapping.connections.iter().map(|&(i, j)| [i, j]).collect::<Vec<_>>()
    })
}

pub(super) fn source_mapping_json(
    request: &CgRequest,
    source_mapping: &SourceMappingResult,
) -> Value {
    json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "kind": "martini_source_residue_mapping",
        "name": request.name,
        "source": request.source,
        "mapping_mode": mapping_mode(request),
        "repeat_smiles": request.repeat_smiles,
        "aa_atom_count": source_mapping.aa_atom_count,
        "mapped_residue_count": source_mapping.residue_count,
        "bead_count": source_mapping.mapping.bead_names.len(),
        "beads": source_mapping.beads.iter().map(|bead| {
            json!({
                "index": bead.index,
                "name": bead.name,
                "bead_type": bead.bead_type,
                "features": bead.features,
                "formal_charge": bead.formal_charge,
                "resid": bead.resid,
                "resname": bead.resname,
                "chain": bead.chain.to_string(),
                "atom_indices": bead.atom_indices,
                "atom_names": bead.atom_names,
                "coord": bead.coord
            })
        }).collect::<Vec<_>>(),
        "connections": source_mapping
            .mapping
            .connections
            .iter()
            .map(|&(i, j)| [i, j])
            .collect::<Vec<_>>(),
        "repeat_unit_bead_template": source_mapping.templates.pointer("/residue_role_templates/middle"),
        "head_middle_tail_bead_templates": source_mapping.templates.get("residue_role_templates"),
        "generated_mapping_template": source_mapping.templates,
        "residue_to_bead_map": source_mapping
            .provenance
            .get("residue_to_bead_map")
            .cloned()
            .unwrap_or_else(|| json!([])),
        "provenance": source_mapping.provenance
    })
}

pub(super) fn source_bonded_parameter_map_json(
    request: &CgRequest,
    source_mapping: &SourceMappingResult,
    bond_stats: &[BondStats],
    angle_stats: &[AngleStats],
    dihedral_stats: &[DihedralStats],
    reference_targets: Option<&ReferenceTargetSet>,
    tuning: Option<&OptimizationReport>,
) -> Value {
    let source_term_targets = if reference_targets.is_none() {
        source_mapping
            .bonded_terms
            .as_ref()
            .map(|terms| reference_targets_from_source_terms(source_mapping, terms))
    } else {
        None
    };
    let effective_targets = reference_targets.or(source_term_targets.as_ref());
    let mut value = bonded_parameter_map_json(
        request,
        &source_mapping.mapping,
        bond_stats,
        angle_stats,
        dihedral_stats,
        effective_targets,
        tuning,
    );
    if let Some(object) = value.as_object_mut() {
        if reference_targets.is_none() && source_term_targets.is_some() {
            if let Some(classing) = object.get_mut("bonded_parameter_classing") {
                if let Some(classing) = classing.as_object_mut() {
                    classing.insert(
                        "class_source".to_string(),
                        Value::String("source_mapping.bonded_terms".to_string()),
                    );
                    classing.insert(
                        "reference_distribution_available".to_string(),
                        Value::Bool(false),
                    );
                }
            }
        }
        object.insert(
            "source_mapping_templates".to_string(),
            source_mapping.templates.clone(),
        );
        object.insert(
            "mapping_provenance".to_string(),
            source_mapping.provenance.clone(),
        );
    }
    value
}

pub(super) fn bonded_parameter_map_json(
    request: &CgRequest,
    mapping: &MappingResult,
    bond_stats: &[BondStats],
    angle_stats: &[AngleStats],
    dihedral_stats: &[DihedralStats],
    reference_targets: Option<&ReferenceTargetSet>,
    tuning: Option<&OptimizationReport>,
) -> Value {
    if let Some(targets) = reference_targets {
        return grouped_bonded_parameter_map_json(request, mapping, targets, tuning);
    }
    json!({
        "schema_version": "warp-cg.bonded-parameter-map.v1",
        "name": request.name,
        "smiles": request.smiles,
        "repeat_smiles": request.repeat_smiles,
        "source": request.source,
        "mapping_mode": mapping_mode(request),
        "itp": format!("{}_martini.itp", request.name),
        "units": {
            "bond_length": "nm",
            "bond_force": "kJ mol^-1 nm^-2",
            "angle": "degree",
            "angle_force": "kJ mol^-1 rad^-2",
            "dihedral_phase": "degree",
            "dihedral_force": "kJ mol^-1"
        },
        "bonds": mapping.connections.iter().map(|&(i, j)| {
            let a = i.min(j);
            let b = i.max(j);
            let (length_nm, force) = bonded_pair_parameters(i, j, bond_stats, tuning);
            json!({
                "itp_section": "bonds",
                "beads_zero_based": [i, j],
                "itp_atoms_one_based": [i + 1, j + 1],
                "parameter_names": {
                    "length_angstrom": format!("bond_{a}_{b}_length_angstrom"),
                    "force": format!("bond_{a}_{b}_force")
                },
                "source_stat": bond_stats.iter().find(|stat| stat.bead_i == a && stat.bead_j == b),
                "itp_values": {
                    "funct": 1,
                    "length_nm": length_nm,
                    "force": force
                }
            })
        }).collect::<Vec<_>>(),
        "angles": angle_stats.iter().map(|stat| {
            let (angle_deg, force) = bonded_angle_parameters(stat, tuning);
            json!({
                "itp_section": "angles",
                "beads_zero_based": [stat.bead_i, stat.bead_j, stat.bead_k],
                "itp_atoms_one_based": [stat.bead_i + 1, stat.bead_j + 1, stat.bead_k + 1],
                "parameter_names": {
                    "angle_deg": format!("angle_{}_{}_{}_angle_deg", stat.bead_i, stat.bead_j, stat.bead_k),
                    "force": format!("angle_{}_{}_{}_force", stat.bead_i, stat.bead_j, stat.bead_k)
                },
                "source_stat": stat,
                "itp_values": {
                    "funct": 2,
                    "angle_deg": angle_deg,
                    "force": force
                }
            })
        }).collect::<Vec<_>>(),
        "dihedrals": dihedral_stats.iter().map(|stat| {
            let (phase_deg, force) = bonded_dihedral_parameters(stat, tuning);
            json!({
                "itp_section": "dihedrals",
                "beads_zero_based": [stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l],
                "itp_atoms_one_based": [stat.bead_i + 1, stat.bead_j + 1, stat.bead_k + 1, stat.bead_l + 1],
                "parameter_names": {
                    "phase_deg": format!(
                        "dihedral_{}_{}_{}_{}_phase_deg",
                        stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
                    ),
                    "force": format!(
                        "dihedral_{}_{}_{}_{}_force",
                        stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
                    )
                },
                "source_stat": stat,
                "itp_values": {
                    "funct": 1,
                    "phase_deg": phase_deg,
                    "force": force,
                    "multiplicity": 1
                }
            })
        }).collect::<Vec<_>>()
    })
}

pub(super) fn beads(mapping: &MappingResult) -> Vec<CgBead> {
    mapping
        .bead_names
        .iter()
        .zip(mapping.atom_groups.iter())
        .enumerate()
        .map(|(index, (name, atom_indices))| CgBead {
            index,
            name: name.clone(),
            atom_indices: atom_indices.clone(),
            features: mapping
                .bead_features
                .get(index)
                .cloned()
                .unwrap_or_default(),
            formal_charge: mapping.bead_formal_charges.get(index).copied().unwrap_or(0),
        })
        .collect()
}

pub(super) fn render_martini_itp(
    name: &str,
    mapping: &MappingResult,
    bond_stats: &[BondStats],
    angle_stats: &[AngleStats],
    dihedral_stats: &[DihedralStats],
    reference_targets: Option<&ReferenceTargetSet>,
    tuning: Option<&OptimizationReport>,
) -> String {
    let molecule = topology_name(name);
    let mut out = String::new();
    out.push_str("; Generated by warp-cg\n");
    out.push_str("[ moleculetype ]\n");
    out.push_str("; name  nrexcl\n");
    out.push_str(&format!("{molecule:<16} 1\n\n"));
    out.push_str("[ atoms ]\n");
    out.push_str("; nr  type  resnr  residue  atom  cgnr  charge\n");
    for (idx, bead) in mapping.bead_names.iter().enumerate() {
        let charge = mapping.bead_formal_charges.get(idx).copied().unwrap_or(0) as f64;
        out.push_str(&format!(
            "{:>5} {:<6} {:>5} {:<8} {:<6} {:>5} {:>8.3}\n",
            idx + 1,
            bead,
            1,
            molecule,
            format!("B{}", idx + 1),
            idx + 1,
            charge
        ));
    }
    if let Some(targets) = reference_targets {
        render_grouped_bonded_sections(&mut out, targets, tuning);
    } else if !mapping.connections.is_empty() {
        out.push_str("\n[ bonds ]\n");
        out.push_str("; i  j  funct  length(nm)  force\n");
        for &(i, j) in &mapping.connections {
            let (length_nm, force) = bonded_pair_parameters(i, j, bond_stats, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>10.5} {:>10.3}\n",
                i + 1,
                j + 1,
                1,
                length_nm,
                force
            ));
        }
    }
    if reference_targets.is_none() && !angle_stats.is_empty() {
        out.push_str("\n[ angles ]\n");
        out.push_str("; i  j  k  funct  angle(deg)  force\n");
        for stat in angle_stats {
            let (angle, force) = bonded_angle_parameters(stat, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>5} {:>10.4} {:>10.3}\n",
                stat.bead_i + 1,
                stat.bead_j + 1,
                stat.bead_k + 1,
                2,
                angle,
                force
            ));
        }
    }
    if reference_targets.is_none() && !dihedral_stats.is_empty() {
        out.push_str("\n[ dihedrals ]\n");
        out.push_str("; i  j  k  l  funct  angle(deg)  force  multiplicity\n");
        for stat in dihedral_stats {
            let (phase, force) = bonded_dihedral_parameters(stat, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>5} {:>5} {:>10.4} {:>10.3} {:>5}\n",
                stat.bead_i + 1,
                stat.bead_j + 1,
                stat.bead_k + 1,
                stat.bead_l + 1,
                1,
                phase,
                force,
                1
            ));
        }
    }
    out
}

pub(super) fn render_source_martini_itp(
    name: &str,
    source_mapping: &SourceMappingResult,
    bond_stats: &[BondStats],
    angle_stats: &[AngleStats],
    dihedral_stats: &[DihedralStats],
    reference_targets: Option<&ReferenceTargetSet>,
    tuning: Option<&OptimizationReport>,
) -> String {
    let molecule = topology_name(name);
    let mut out = String::new();
    out.push_str("; Generated by warp-cg\n");
    out.push_str("[ moleculetype ]\n");
    out.push_str("; name  nrexcl\n");
    out.push_str(&format!("{molecule:<16} 1\n\n"));
    out.push_str("[ atoms ]\n");
    out.push_str("; nr  type  resnr  residue  atom  cgnr  charge\n");
    for bead in &source_mapping.beads {
        out.push_str(&format!(
            "{:>5} {:<6} {:>5} {:<8} {:<6} {:>5} {:>8.3}\n",
            bead.index + 1,
            bead.bead_type,
            bead.resid.max(1),
            bead.resname.chars().take(8).collect::<String>(),
            pdb_atom_name(&bead.name, bead.index),
            bead.index + 1,
            bead.formal_charge as f64
        ));
    }
    let source_term_targets = if reference_targets.is_none() {
        source_mapping
            .bonded_terms
            .as_ref()
            .map(|terms| reference_targets_from_source_terms(source_mapping, terms))
    } else {
        None
    };
    let effective_targets = reference_targets.or(source_term_targets.as_ref());
    if let Some(targets) = effective_targets {
        render_grouped_bonded_sections(&mut out, targets, tuning);
    } else if !source_mapping.mapping.connections.is_empty() {
        out.push_str("\n[ bonds ]\n");
        out.push_str("; i  j  funct  length(nm)  force\n");
        for &(i, j) in &source_mapping.mapping.connections {
            let (length_nm, force) = bonded_pair_parameters(i, j, bond_stats, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>10.5} {:>10.3}\n",
                i + 1,
                j + 1,
                1,
                length_nm,
                force
            ));
        }
    }
    if effective_targets.is_none() && !angle_stats.is_empty() {
        out.push_str("\n[ angles ]\n");
        out.push_str("; i  j  k  funct  angle(deg)  force\n");
        for stat in angle_stats {
            let (angle, force) = bonded_angle_parameters(stat, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>5} {:>10.4} {:>10.3}\n",
                stat.bead_i + 1,
                stat.bead_j + 1,
                stat.bead_k + 1,
                2,
                angle,
                force
            ));
        }
    }
    if effective_targets.is_none() && !dihedral_stats.is_empty() {
        out.push_str("\n[ dihedrals ]\n");
        out.push_str("; i  j  k  l  funct  angle(deg)  force  multiplicity\n");
        for stat in dihedral_stats {
            let (phase, force) = bonded_dihedral_parameters(stat, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>5} {:>5} {:>10.4} {:>10.3} {:>5}\n",
                stat.bead_i + 1,
                stat.bead_j + 1,
                stat.bead_k + 1,
                stat.bead_l + 1,
                1,
                phase,
                force,
                1
            ));
        }
    }
    out
}

fn reference_targets_from_source_terms(
    source_mapping: &SourceMappingResult,
    terms: &BondedTermSet,
) -> ReferenceTargetSet {
    let bin_config = ReferenceBinConfig::default();
    ReferenceTargetSet {
        version: 1,
        bin_config: bin_config.clone(),
        constraints: terms
            .constraints
            .iter()
            .map(|group| {
                let members = group
                    .members
                    .iter()
                    .map(|member| vec![member[0], member[1]])
                    .collect::<Vec<_>>();
                source_geometry_target(
                    ReferenceTermKind::Constraint,
                    group.label.clone(),
                    members,
                    source_mapping,
                    &bin_config,
                )
            })
            .collect(),
        bonds: terms
            .bonds
            .iter()
            .map(|group| {
                let members = group
                    .members
                    .iter()
                    .map(|member| vec![member[0], member[1]])
                    .collect::<Vec<_>>();
                source_geometry_target(
                    ReferenceTermKind::Bond,
                    group.label.clone(),
                    members,
                    source_mapping,
                    &bin_config,
                )
            })
            .collect(),
        angles: terms
            .angles
            .iter()
            .map(|group| {
                let members = group
                    .members
                    .iter()
                    .map(|member| vec![member[0], member[1], member[2]])
                    .collect::<Vec<_>>();
                source_geometry_target(
                    ReferenceTermKind::Angle,
                    group.label.clone(),
                    members,
                    source_mapping,
                    &bin_config,
                )
            })
            .collect(),
        dihedrals: terms
            .dihedrals
            .iter()
            .map(|group| {
                let members = group
                    .members
                    .iter()
                    .map(|member| vec![member[0], member[1], member[2], member[3]])
                    .collect::<Vec<_>>();
                source_geometry_target(
                    ReferenceTermKind::Dihedral,
                    group.label.clone(),
                    members,
                    source_mapping,
                    &bin_config,
                )
            })
            .collect(),
    }
}

fn source_geometry_target(
    kind: ReferenceTermKind,
    label: Option<String>,
    members: Vec<Vec<usize>>,
    source_mapping: &SourceMappingResult,
    bin_config: &ReferenceBinConfig,
) -> ReferenceDistributionTarget {
    let values = members
        .iter()
        .filter_map(|member| source_member_value(kind, member, source_mapping))
        .collect::<Vec<_>>();
    let (units, periodic, min, max, width) = match kind {
        ReferenceTermKind::Constraint | ReferenceTermKind::Bond => (
            "nm",
            false,
            0.0,
            bin_config.bonded_max_range_nm,
            bin_config.bond_bin_width_nm,
        ),
        ReferenceTermKind::Angle => ("deg", false, 0.0, 180.0, bin_config.angle_bin_width_deg),
        ReferenceTermKind::Dihedral => (
            "deg",
            true,
            -180.0,
            180.0,
            bin_config.dihedral_bin_width_deg,
        ),
    };
    let beads = members.first().cloned().unwrap_or_default();
    ReferenceDistributionTarget::from_samples(
        kind, label, beads, members, &values, units, periodic, min, max, width,
    )
}

fn source_member_value(
    kind: ReferenceTermKind,
    member: &[usize],
    source_mapping: &SourceMappingResult,
) -> Option<f64> {
    match kind {
        ReferenceTermKind::Constraint | ReferenceTermKind::Bond => {
            let a = source_bead_coord(source_mapping, *member.first()?)?;
            let b = source_bead_coord(source_mapping, *member.get(1)?)?;
            Some(distance(a, b) / 10.0)
        }
        ReferenceTermKind::Angle => {
            let a = source_bead_coord(source_mapping, *member.first()?)?;
            let b = source_bead_coord(source_mapping, *member.get(1)?)?;
            let c = source_bead_coord(source_mapping, *member.get(2)?)?;
            Some(angle_deg(a, b, c))
        }
        ReferenceTermKind::Dihedral => {
            let a = source_bead_coord(source_mapping, *member.first()?)?;
            let b = source_bead_coord(source_mapping, *member.get(1)?)?;
            let c = source_bead_coord(source_mapping, *member.get(2)?)?;
            let d = source_bead_coord(source_mapping, *member.get(3)?)?;
            Some(dihedral_deg(a, b, c, d))
        }
    }
}

fn source_bead_coord(source_mapping: &SourceMappingResult, idx: usize) -> Option<[f64; 3]> {
    let coord = source_mapping.beads.get(idx)?.coord;
    Some([coord[0] as f64, coord[1] as f64, coord[2] as f64])
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn angle_deg(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let ba = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    let denom = norm(ba) * norm(bc);
    if denom <= f64::EPSILON {
        return 180.0;
    }
    let cosine = (dot(ba, bc) / denom).clamp(-1.0, 1.0);
    cosine.acos().to_degrees()
}

fn dihedral_deg(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> f64 {
    let b0 = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    let b1 = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
    let b2 = [d[0] - c[0], d[1] - c[1], d[2] - c[2]];
    let b1_norm = norm(b1);
    if b1_norm <= f64::EPSILON {
        return 180.0;
    }
    let b1_unit = [b1[0] / b1_norm, b1[1] / b1_norm, b1[2] / b1_norm];
    let v = subtract(b0, scale(b1_unit, dot(b0, b1_unit)));
    let w = subtract(b2, scale(b1_unit, dot(b2, b1_unit)));
    let x = dot(v, w);
    let y = dot(cross(b1_unit, v), w);
    y.atan2(x).to_degrees()
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn scale(a: [f64; 3], value: f64) -> [f64; 3] {
    [a[0] * value, a[1] * value, a[2] * value]
}

fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn grouped_bonded_parameter_map_json(
    request: &CgRequest,
    mapping: &MappingResult,
    targets: &ReferenceTargetSet,
    tuning: Option<&OptimizationReport>,
) -> Value {
    json!({
        "schema_version": "warp-cg.bonded-parameter-map.v1",
        "name": request.name,
        "smiles": request.smiles,
        "repeat_smiles": request.repeat_smiles,
        "source": request.source,
        "mapping_mode": mapping_mode(request),
        "itp": format!("{}_martini.itp", request.name),
        "bonded_parameter_classing": {
            "enabled": true,
            "class_source": "reference_targets",
            "raw_instance_counts": {
                "constraints": target_member_count(&targets.constraints),
                "bonds": target_member_count(&targets.bonds),
                "angles": target_member_count(&targets.angles),
                "dihedrals": target_member_count(&targets.dihedrals)
            },
            "class_counts": {
                "constraints": targets.constraints.len(),
                "bonds": targets.bonds.len(),
                "angles": targets.angles.len(),
                "dihedrals": targets.dihedrals.len()
            }
        },
        "units": {
            "bond_length": "nm",
            "bond_force": "kJ mol^-1 nm^-2",
            "angle": "degree",
            "angle_force": "kJ mol^-1 rad^-2",
            "dihedral_phase": "degree",
            "dihedral_force": "kJ mol^-1"
        },
        "constraints": targets.constraints.iter().map(|target| grouped_bond_map_entry(target, tuning, true)).collect::<Vec<_>>(),
        "bonds": targets.bonds.iter().map(|target| grouped_bond_map_entry(target, tuning, false)).collect::<Vec<_>>(),
        "angles": targets.angles.iter().map(|target| grouped_angle_map_entry(target, tuning)).collect::<Vec<_>>(),
        "dihedrals": targets.dihedrals.iter().map(|target| grouped_dihedral_map_entry(target, tuning)).collect::<Vec<_>>(),
        "bead_count": mapping.bead_names.len()
    })
}

fn render_grouped_bonded_sections(
    out: &mut String,
    targets: &ReferenceTargetSet,
    tuning: Option<&OptimizationReport>,
) {
    if !targets.constraints.is_empty() {
        out.push_str("\n[ constraints ]\n");
        out.push_str("; i  j  funct  length(nm)\n");
        for target in &targets.constraints {
            push_class_comment(out, target);
            let (length, _) = grouped_target_parameters(target, tuning);
            let length_nm = grouped_length_nm(target, length);
            for member in &target.members {
                if member.len() >= 2 {
                    out.push_str(&format!(
                        "{:>5} {:>5} {:>5} {:>10.5}\n",
                        member[0] + 1,
                        member[1] + 1,
                        1,
                        length_nm
                    ));
                }
            }
        }
    }
    if !targets.bonds.is_empty() {
        out.push_str("\n[ bonds ]\n");
        out.push_str("; i  j  funct  length(nm)  force\n");
        for target in &targets.bonds {
            push_class_comment(out, target);
            let (length, force) = grouped_target_parameters(target, tuning);
            let length_nm = grouped_length_nm(target, length);
            for member in &target.members {
                if member.len() >= 2 {
                    out.push_str(&format!(
                        "{:>5} {:>5} {:>5} {:>10.5} {:>10.3}\n",
                        member[0] + 1,
                        member[1] + 1,
                        1,
                        length_nm,
                        force.unwrap_or_else(|| grouped_target_force(target))
                    ));
                }
            }
        }
    }
    if !targets.angles.is_empty() {
        out.push_str("\n[ angles ]\n");
        out.push_str("; i  j  k  funct  angle(deg)  force\n");
        for target in &targets.angles {
            push_class_comment(out, target);
            let (angle, force) = grouped_target_parameters(target, tuning);
            for member in &target.members {
                if member.len() >= 3 {
                    out.push_str(&format!(
                        "{:>5} {:>5} {:>5} {:>5} {:>10.4} {:>10.3}\n",
                        member[0] + 1,
                        member[1] + 1,
                        member[2] + 1,
                        2,
                        angle,
                        force.unwrap_or_else(|| grouped_target_force(target))
                    ));
                }
            }
        }
    }
    if !targets.dihedrals.is_empty() {
        out.push_str("\n[ dihedrals ]\n");
        out.push_str("; i  j  k  l  funct  angle(deg)  force  multiplicity\n");
        for target in &targets.dihedrals {
            push_class_comment(out, target);
            let (phase, force) = grouped_target_parameters(target, tuning);
            for member in &target.members {
                if member.len() >= 4 {
                    out.push_str(&format!(
                        "{:>5} {:>5} {:>5} {:>5} {:>5} {:>10.4} {:>10.3} {:>5}\n",
                        member[0] + 1,
                        member[1] + 1,
                        member[2] + 1,
                        member[3] + 1,
                        1,
                        phase,
                        force.unwrap_or_else(|| grouped_target_force(target)),
                        1
                    ));
                }
            }
        }
    }
}

fn grouped_bond_map_entry(
    target: &ReferenceDistributionTarget,
    tuning: Option<&OptimizationReport>,
    constraint: bool,
) -> Value {
    let (length, force) = grouped_target_parameters(target, tuning);
    json!({
        "itp_section": if constraint { "constraints" } else { "bonds" },
        "class_label": target.label.clone(),
        "members_zero_based": target.members.clone(),
        "parameter_names": grouped_bond_parameter_names(target),
        "source_target": target,
        "itp_values": {
            "funct": 1,
            "length_nm": grouped_length_nm(target, length),
            "force": force.unwrap_or_else(|| grouped_target_force(target))
        }
    })
}

fn grouped_angle_map_entry(
    target: &ReferenceDistributionTarget,
    tuning: Option<&OptimizationReport>,
) -> Value {
    let (angle, force) = grouped_target_parameters(target, tuning);
    json!({
        "itp_section": "angles",
        "class_label": target.label.clone(),
        "members_zero_based": target.members.clone(),
        "parameter_names": {
            "angle_deg": grouped_value_name(target),
            "force": grouped_force_name(target)
        },
        "source_target": target,
        "itp_values": {
            "funct": 2,
            "angle_deg": angle,
            "force": force.unwrap_or_else(|| grouped_target_force(target))
        }
    })
}

fn grouped_dihedral_map_entry(
    target: &ReferenceDistributionTarget,
    tuning: Option<&OptimizationReport>,
) -> Value {
    let (phase, force) = grouped_target_parameters(target, tuning);
    json!({
        "itp_section": "dihedrals",
        "class_label": target.label.clone(),
        "members_zero_based": target.members.clone(),
        "parameter_names": {
            "phase_deg": grouped_value_name(target),
            "force": grouped_force_name(target)
        },
        "source_target": target,
        "itp_values": {
            "funct": 1,
            "phase_deg": phase,
            "force": force.unwrap_or_else(|| grouped_target_force(target)),
            "multiplicity": 1
        }
    })
}

fn push_class_comment(out: &mut String, target: &ReferenceDistributionTarget) {
    if let Some(label) = target.label.as_deref().filter(|label| !label.is_empty()) {
        out.push_str(&format!("; class: {label}\n"));
    }
}

fn target_member_count(targets: &[ReferenceDistributionTarget]) -> usize {
    targets.iter().map(|target| target.members.len()).sum()
}

fn grouped_target_parameters(
    target: &ReferenceDistributionTarget,
    tuning: Option<&OptimizationReport>,
) -> (f64, Option<f64>) {
    let mut value = target.mean;
    let mut force = Some(grouped_target_force(target));
    if let Some(tuning) = tuning {
        let value_name = grouped_value_name(target);
        let force_name = grouped_force_name(target);
        for (name, candidate) in &tuning.best_parameters {
            if name == &value_name {
                value = *candidate;
            } else if Some(name.as_str()) == force_name.as_deref() {
                force = Some(*candidate);
            }
        }
    }
    (value, force)
}

fn grouped_bond_parameter_names(target: &ReferenceDistributionTarget) -> Value {
    let mut names = serde_json::Map::new();
    names.insert(
        grouped_length_parameter_key(target).to_string(),
        Value::String(grouped_value_name(target)),
    );
    if let Some(force_name) = grouped_force_name(target) {
        names.insert("force".to_string(), Value::String(force_name));
    }
    Value::Object(names)
}

fn grouped_length_nm(target: &ReferenceDistributionTarget, value: f64) -> f64 {
    length_value_to_nm(value, &target.units)
}

fn length_value_to_nm(value: f64, units: &str) -> f64 {
    match units.trim().to_ascii_lowercase().as_str() {
        "nm" | "nanometer" | "nanometers" => value,
        _ => value / 10.0,
    }
}

fn grouped_target_force(target: &ReferenceDistributionTarget) -> f64 {
    match target.kind {
        ReferenceTermKind::Constraint => 0.0,
        ReferenceTermKind::Bond => (1.0 / target.std.max(0.02).powi(2)).clamp(1.0, 5000.0),
        ReferenceTermKind::Angle => {
            (1.0 / target.std.max(1.0).powi(2) * 10_000.0).clamp(1.0, 500.0)
        }
        ReferenceTermKind::Dihedral => {
            (1.0 / target.std.max(1.0).powi(2) * 1_000.0).clamp(0.1, 100.0)
        }
    }
}

fn grouped_value_name(target: &ReferenceDistributionTarget) -> String {
    if let Some(label) = target.label.as_deref().filter(|label| !label.is_empty()) {
        let label = parameter_safe_label(label);
        return match target.kind {
            ReferenceTermKind::Constraint | ReferenceTermKind::Bond => {
                format!("{label}_{}", grouped_length_parameter_key(target))
            }
            ReferenceTermKind::Angle => format!("{label}_angle_deg"),
            ReferenceTermKind::Dihedral => format!("{label}_phase_deg"),
        };
    }
    let beads = target
        .members
        .first()
        .cloned()
        .unwrap_or_else(|| target.beads.clone());
    match target.kind {
        ReferenceTermKind::Constraint => {
            format!(
                "constraint_{}_{}_{}",
                beads[0],
                beads[1],
                grouped_length_parameter_key(target)
            )
        }
        ReferenceTermKind::Bond => {
            format!(
                "bond_{}_{}_{}",
                beads[0],
                beads[1],
                grouped_length_parameter_key(target)
            )
        }
        ReferenceTermKind::Angle => {
            format!("angle_{}_{}_{}_angle_deg", beads[0], beads[1], beads[2])
        }
        ReferenceTermKind::Dihedral => format!(
            "dihedral_{}_{}_{}_{}_phase_deg",
            beads[0], beads[1], beads[2], beads[3]
        ),
    }
}

fn grouped_length_parameter_key(target: &ReferenceDistributionTarget) -> &'static str {
    match target.units.trim().to_ascii_lowercase().as_str() {
        "nm" | "nanometer" | "nanometers" => "length_nm",
        _ => "length_angstrom",
    }
}

fn grouped_force_name(target: &ReferenceDistributionTarget) -> Option<String> {
    if target.kind == ReferenceTermKind::Constraint {
        return None;
    }
    if let Some(label) = target.label.as_deref().filter(|label| !label.is_empty()) {
        let label = parameter_safe_label(label);
        return Some(format!("{label}_force"));
    }
    let beads = target
        .members
        .first()
        .cloned()
        .unwrap_or_else(|| target.beads.clone());
    Some(match target.kind {
        ReferenceTermKind::Constraint => return None,
        ReferenceTermKind::Bond => format!("bond_{}_{}_force", beads[0], beads[1]),
        ReferenceTermKind::Angle => format!("angle_{}_{}_{}_force", beads[0], beads[1], beads[2]),
        ReferenceTermKind::Dihedral => format!(
            "dihedral_{}_{}_{}_{}_force",
            beads[0], beads[1], beads[2], beads[3]
        ),
    })
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
        "bonded_class".to_string()
    } else {
        trimmed
    }
}

pub(super) fn render_cg_pdb(
    name: &str,
    mapping: &MappingResult,
    coords: Option<&[[f32; 3]]>,
) -> String {
    let residue = topology_name(name);
    let residue = residue.chars().take(3).collect::<String>();
    let mut out = String::new();
    out.push_str("REMARK Generated by warp-cg\n");
    if coords.is_none() {
        out.push_str("REMARK Coordinates are deterministic scaffold positions; prefer mapped trajectory/GRO when available.\n");
    }
    for (idx, bead) in mapping.bead_names.iter().enumerate() {
        let coord = coords
            .and_then(|values| values.get(idx))
            .copied()
            .unwrap_or([idx as f32 * 4.7, 0.0, 0.0]);
        out.push_str(&format!(
            "ATOM  {:>5} {:<4} {:<3} A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00  0.00          {:>2}\n",
            idx + 1,
            pdb_atom_name(bead, idx),
            residue,
            1,
            coord[0],
            coord[1],
            coord[2],
            pdb_element(bead)
        ));
    }
    for &(i, j) in &mapping.connections {
        out.push_str(&format!("CONECT{:>5}{:>5}\n", i + 1, j + 1));
    }
    out.push_str("END\n");
    out
}

pub(super) fn render_source_cg_pdb(
    source_mapping: &SourceMappingResult,
    coords: Option<&[[f32; 3]]>,
) -> String {
    let mut out = String::new();
    out.push_str("REMARK Generated by warp-cg\n");
    if coords.is_none() {
        out.push_str("REMARK Coordinates are residue bead centers from source coordinates.\n");
    }
    for bead in &source_mapping.beads {
        let coord = coords
            .and_then(|values| values.get(bead.index))
            .copied()
            .unwrap_or(bead.coord);
        out.push_str(&format!(
            "ATOM  {:>5} {:<4} {:<3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00  0.00          {:>2}\n",
            bead.index + 1,
            pdb_atom_name(&bead.name, bead.index),
            bead.resname.chars().take(3).collect::<String>(),
            bead.chain,
            bead.resid,
            coord[0],
            coord[1],
            coord[2],
            pdb_element(&bead.name)
        ));
    }
    for &(i, j) in &source_mapping.mapping.connections {
        out.push_str(&format!("CONECT{:>5}{:>5}\n", i + 1, j + 1));
    }
    out.push_str("END\n");
    out
}

pub(super) fn pdb_atom_name(bead: &str, idx: usize) -> String {
    let mut name: String = bead
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .take(3)
        .collect();
    if name.is_empty() {
        name.push('B');
    }
    format!("{name}{}", (idx + 1) % 10)
}

pub(super) fn pdb_element(bead: &str) -> &'static str {
    if bead.starts_with('P') {
        "P"
    } else if bead.starts_with('N') {
        "N"
    } else {
        "C"
    }
}

pub(super) fn angle_force(stat: &AngleStats) -> f64 {
    (1.0 / stat.std_deg.max(1.0).powi(2) * 10_000.0).clamp(1.0, 500.0)
}

pub(super) fn dihedral_force(stat: &DihedralStats) -> f64 {
    (1.0 / stat.std_deg.max(1.0).powi(2) * 1_000.0).clamp(0.1, 100.0)
}

pub(super) fn bonded_angle_parameters(
    stat: &AngleStats,
    tuning: Option<&OptimizationReport>,
) -> (f64, f64) {
    let mut angle = stat.mean_deg;
    let mut force = angle_force(stat);
    if let Some(tuning) = tuning {
        let angle_name = format!(
            "angle_{}_{}_{}_angle_deg",
            stat.bead_i, stat.bead_j, stat.bead_k
        );
        let force_name = format!(
            "angle_{}_{}_{}_force",
            stat.bead_i, stat.bead_j, stat.bead_k
        );
        for (name, value) in &tuning.best_parameters {
            if name == &angle_name {
                angle = *value;
            } else if name == &force_name {
                force = *value;
            }
        }
    }
    (angle, force)
}

pub(super) fn bonded_dihedral_parameters(
    stat: &DihedralStats,
    tuning: Option<&OptimizationReport>,
) -> (f64, f64) {
    let mut phase = stat.mean_deg;
    let mut force = dihedral_force(stat);
    if let Some(tuning) = tuning {
        let phase_name = format!(
            "dihedral_{}_{}_{}_{}_phase_deg",
            stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
        );
        let force_name = format!(
            "dihedral_{}_{}_{}_{}_force",
            stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
        );
        for (name, value) in &tuning.best_parameters {
            if name == &phase_name {
                phase = *value;
            } else if name == &force_name {
                force = *value;
            }
        }
    }
    (phase, force)
}

pub(super) fn bonded_pair_parameters(
    i: usize,
    j: usize,
    bond_stats: &[BondStats],
    tuning: Option<&OptimizationReport>,
) -> (f64, f64) {
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    let stat = bond_stats
        .iter()
        .find(|stat| stat.bead_i == a && stat.bead_j == b);
    let mut length_angstrom = stat.map(|stat| stat.mean).unwrap_or(4.7);
    let mut force = stat
        .map(|stat| (1.0 / stat.std.max(0.02).powi(2)).clamp(1.0, 5000.0))
        .unwrap_or(1250.0);

    if let Some(tuning) = tuning {
        for (name, value) in &tuning.best_parameters {
            if name == &format!("bond_{a}_{b}_length_angstrom") {
                length_angstrom = *value;
            } else if name == &format!("bond_{a}_{b}_force") {
                force = *value;
            }
        }
    }
    (length_angstrom / 10.0, force)
}

pub(super) fn topology_name(name: &str) -> String {
    let mut out: String = name
        .chars()
        .filter_map(|ch| {
            if ch.is_ascii_alphanumeric() {
                Some(ch.to_ascii_uppercase())
            } else if ch == '_' || ch == '-' {
                Some('_')
            } else {
                None
            }
        })
        .take(12)
        .collect();
    if out.is_empty() {
        out.push_str("MOL");
    }
    out
}

pub(super) fn render_martini_top(
    name: &str,
    itp_file: &str,
    forcefield_includes: &[String],
) -> String {
    let molecule = topology_name(name);
    let mut out = String::new();
    out.push_str("; Generated by warp-cg\n");
    if forcefield_includes.is_empty() {
        out.push_str("; Include the Martini force-field file used by your simulation engine before this molecule include.\n");
    } else {
        out.push_str("; Martini force-field includes resolved by warp-cg\n");
        for include in forcefield_includes {
            out.push_str(&format!("#include \"{include}\"\n"));
        }
    }
    out.push_str(&format!("#include \"{itp_file}\"\n\n"));
    out.push_str("[ system ]\n");
    out.push_str(&format!("{molecule} coarse-grained system\n\n"));
    out.push_str("[ molecules ]\n");
    out.push_str("; molecule  count\n");
    out.push_str(&format!("{molecule:<16} 1\n"));
    out
}
