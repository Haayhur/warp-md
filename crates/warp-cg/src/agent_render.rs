use serde_json::{json, Value};

use crate::mapping::MappingResult;
use crate::optimize::OptimizationReport;
use crate::parameters::{AngleStats, BondStats, DihedralStats};

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
    tuning: Option<&OptimizationReport>,
) -> Value {
    let mut value = bonded_parameter_map_json(
        request,
        &source_mapping.mapping,
        bond_stats,
        angle_stats,
        dihedral_stats,
        tuning,
    );
    if let Some(object) = value.as_object_mut() {
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
        out.push_str(&format!(
            "{:>5} {:<6} {:>5} {:<8} {:<6} {:>5} {:>8.3}\n",
            idx + 1,
            bead,
            1,
            molecule,
            format!("B{}", idx + 1),
            idx + 1,
            0.0
        ));
    }
    if !mapping.connections.is_empty() {
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
    if !angle_stats.is_empty() {
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
    if !dihedral_stats.is_empty() {
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
            bead.name,
            bead.resid.max(1),
            bead.resname.chars().take(8).collect::<String>(),
            pdb_atom_name(&bead.name, bead.index),
            bead.index + 1,
            0.0
        ));
    }
    if !source_mapping.mapping.connections.is_empty() {
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
    if !angle_stats.is_empty() {
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
    if !dihedral_stats.is_empty() {
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

pub(super) fn render_martini_top(name: &str, itp_file: &str) -> String {
    let molecule = topology_name(name);
    let mut out = String::new();
    out.push_str("; Generated by warp-cg\n");
    out.push_str("; Include the Martini force-field file used by your simulation engine before this molecule include.\n");
    out.push_str(&format!("#include \"{itp_file}\"\n\n"));
    out.push_str("[ system ]\n");
    out.push_str(&format!("{molecule} coarse-grained system\n\n"));
    out.push_str("[ molecules ]\n");
    out.push_str("; molecule  count\n");
    out.push_str(&format!("{molecule:<16} 1\n"));
    out
}
