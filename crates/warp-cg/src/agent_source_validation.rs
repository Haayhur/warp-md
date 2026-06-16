use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use warp_structure::io::{read_molecule, read_prmtop_topology, read_system_auto};

use crate::forcefield::forcefield_path_exists;

use super::{active_tuning_request, input_mode, mapping_mode, CgRequest, CgSource, SourceHandoff};

pub(super) fn validation_report(request: &CgRequest) -> Value {
    let mut checks = Vec::new();
    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let mut aa_atom_count = None;

    if let Some(source) = &request.source {
        validate_source_files(
            source,
            &mut checks,
            &mut warnings,
            &mut errors,
            &mut aa_atom_count,
        );
    }
    if let Some(ndx) = request
        .mapping
        .as_ref()
        .and_then(|mapping| mapping.ndx.as_ref())
    {
        validate_path_exists("mapping.ndx", ndx, &mut checks, &mut errors);
    }
    if let Some(source) = &request.trajectory_source {
        validate_path_exists(
            "trajectory_source.path",
            &source.path,
            &mut checks,
            &mut errors,
        );
        if let Some(topology) = &source.topology {
            validate_path_exists(
                "trajectory_source.topology",
                topology,
                &mut checks,
                &mut errors,
            );
        }
    }
    if let Some(source) = request
        .reference_source
        .as_ref()
        .and_then(|reference| reference.bonded_terms.as_ref())
    {
        validate_path_exists(
            "reference_source.bonded_terms.path",
            &source.path,
            &mut checks,
            &mut errors,
        );
    }
    if let Some(source) = request
        .reference_source
        .as_ref()
        .and_then(|reference| reference.precomputed.as_ref())
    {
        validate_path_exists(
            "reference_source.precomputed.target_set",
            &source.target_set,
            &mut checks,
            &mut errors,
        );
    }
    if let Some(forcefield) = &request.forcefield {
        let exists = forcefield_path_exists(forcefield);
        checks.push(json!({
            "name": "forcefield_available",
            "field": "forcefield",
            "kind": forcefield.kind,
            "source": forcefield.source,
            "path": forcefield.path,
            "status": if exists { "ok" } else { "error" }
        }));
        if !exists {
            errors.push(json!({
                "code": "warp_cg.forcefield_missing",
                "field": "forcefield",
                "kind": forcefield.kind,
                "source": forcefield.source,
                "path": forcefield.path,
                "message": "forcefield files are not available"
            }));
        }
    }
    validate_xtb_available(request, &mut checks, &mut warnings);
    validate_optimization_cost(request, &mut checks);

    let has_bonded_stats_source = request.trajectory_source.is_some()
        || request
            .source
            .as_ref()
            .and_then(|source| source.trajectory.as_ref())
            .is_some()
        || request
            .reference_source
            .as_ref()
            .is_some_and(|source| source.kind == "xtb" || source.kind == "precomputed");
    checks.push(json!({
        "name": "bonded_stats_preconditions",
        "status": if has_bonded_stats_source { "ok" } else { "not_available" },
        "message": "bonded stats require trajectory_source, source.trajectory, xTB reference trajectory, or precomputed reference targets"
    }));

    json!({
        "summary": {
            "input_mode": input_mode(request),
            "mapping_mode": mapping_mode(request),
            "aa_atom_count": aa_atom_count,
            "optimization_source": active_tuning_request(request).filter(|tuning| tuning.enabled).map(|tuning| tuning.source.clone()),
            "optimized_terms": active_tuning_request(request).and_then(|tuning| tuning.target_terms.clone()).unwrap_or_else(default_target_terms)
        },
        "checks": checks,
        "warnings": warnings,
        "errors": errors
    })
}

fn validate_source_files(
    source: &CgSource,
    checks: &mut Vec<Value>,
    warnings: &mut Vec<Value>,
    errors: &mut Vec<Value>,
    aa_atom_count: &mut Option<usize>,
) {
    let manifest = source
        .path
        .as_ref()
        .and_then(|path| read_manifest_json(path, checks, errors));
    let coordinates = source.coordinates.clone().or_else(|| {
        manifest
            .as_ref()
            .and_then(|value| manifest_artifact_path(value, "coordinates"))
    });
    let topology = source.topology.clone().or_else(|| {
        manifest
            .as_ref()
            .and_then(|value| manifest_artifact_path(value, "topology"))
    });
    let charge_manifest = source.charge_manifest.clone().or_else(|| {
        manifest
            .as_ref()
            .and_then(|value| manifest_artifact_path(value, "charge_manifest"))
    });

    if let Some(path) = &source.path {
        validate_path_exists("source.path", path, checks, errors);
    }
    if let Some(path) = &coordinates {
        validate_path_exists("source.coordinates", path, checks, errors);
    }
    if let Some(path) = &topology {
        validate_path_exists("source.topology", path, checks, errors);
    }
    if let Some(path) = &charge_manifest {
        validate_path_exists("source.charge_manifest", path, checks, errors);
    }
    if let Some(path) = &source.trajectory {
        validate_path_exists("source.trajectory", path, checks, errors);
    }

    let coordinate_atoms = coordinates
        .as_ref()
        .and_then(|path| coordinate_atom_count(path, source.format.as_deref(), checks, warnings));
    let topology_atoms = topology.as_ref().and_then(|path| {
        topology_atom_count(path, source.topology_format.as_deref(), checks, warnings)
    });
    if aa_atom_count.is_none() {
        *aa_atom_count = coordinate_atoms.or(topology_atoms);
    }
    if let (Some(coord_atoms), Some(top_atoms)) = (coordinate_atoms, topology_atoms) {
        let status = if coord_atoms == top_atoms {
            "ok"
        } else {
            "error"
        };
        checks.push(json!({
            "name": "coordinate_topology_atom_count_match",
            "status": status,
            "coordinate_atoms": coord_atoms,
            "topology_atoms": top_atoms
        }));
        if coord_atoms != top_atoms {
            errors.push(json!({
                "code": "warp_cg.atom_count_mismatch",
                "message": "source coordinate and topology atom counts differ",
                "coordinate_atoms": coord_atoms,
                "topology_atoms": top_atoms
            }));
        }
    }

    validate_target_selection(
        source,
        topology.as_deref().or(coordinates.as_deref()),
        coordinate_atoms,
        checks,
        warnings,
        errors,
    );
    checks.push(json!({
        "name": "template_match_preconditions",
        "status": if source_mapping_mode_for_source(source).is_some() {
            "available"
        } else {
            "not_available"
        },
        "message": "template mode can replay warp-cg.mapping_template.v1 files against source residue atom names; auto mode emits a reusable generated template"
    }));
}

fn source_mapping_mode_for_source(source: &CgSource) -> Option<&'static str> {
    matches!(
        source.kind.as_str(),
        "structure"
            | "polymer_build_manifest"
            | "polymer_pack_manifest"
            | "coordinates_topology"
            | "coordinates_topology_charge_manifest"
            | "source_manifest"
    )
    .then_some("source")
}

fn validate_target_selection(
    source: &CgSource,
    topology: Option<&str>,
    coordinate_atoms: Option<usize>,
    checks: &mut Vec<Value>,
    warnings: &mut Vec<Value>,
    errors: &mut Vec<Value>,
) {
    let Some(selection_expr) = source_selection(source) else {
        checks.push(json!({
            "name": "target_selection_declared",
            "status": "not_declared"
        }));
        return;
    };
    checks.push(json!({
        "name": "target_selection_declared",
        "status": "declared",
        "target_selection": selection_expr
    }));
    let Some(topology) = topology else {
        warnings.push(json!({
            "code": "warp_cg.selection_validation_unavailable",
            "message": "source.target_selection was declared but no readable topology path was available for selection evaluation"
        }));
        checks.push(json!({
            "name": "target_selection_selects_atoms",
            "status": "not_available",
            "target_selection": selection_expr
        }));
        return;
    };
    match read_system_auto(Path::new(topology), source.topology_format.as_deref()) {
        Ok(mut system) => match system.select(selection_expr) {
            Ok(selection) => {
                let selected_atoms = selection.indices.len();
                let status = if selected_atoms > 0 { "ok" } else { "error" };
                checks.push(json!({
                    "name": "target_selection_selects_atoms",
                    "status": status,
                    "target_selection": selection_expr,
                    "selected_atoms": selected_atoms
                }));
                if selected_atoms == 0 {
                    errors.push(json!({
                        "code": "warp_cg.selection_empty",
                        "message": "source.target_selection selected no atoms",
                        "target_selection": selection_expr
                    }));
                }
                if let Some(coord_atoms) = coordinate_atoms {
                    checks.push(json!({
                        "name": "selection_topology_atom_count_context",
                        "status": if system.n_atoms() == coord_atoms { "ok" } else { "warning" },
                        "topology_atoms": system.n_atoms(),
                        "coordinate_atoms": coord_atoms
                    }));
                }
            }
            Err(err) => {
                errors.push(json!({
                    "code": "warp_cg.selection_invalid",
                    "message": err.to_string(),
                    "target_selection": selection_expr
                }));
                checks.push(json!({
                    "name": "target_selection_selects_atoms",
                    "status": "error",
                    "target_selection": selection_expr
                }));
            }
        },
        Err(err) => {
            warnings.push(json!({
                "code": "warp_cg.selection_validation_unavailable",
                "message": format!("failed to read topology for source.target_selection: {err}")
            }));
            checks.push(json!({
                "name": "target_selection_selects_atoms",
                "status": "not_available",
                "target_selection": selection_expr
            }));
        }
    }
}

fn read_manifest_json(
    path: &str,
    checks: &mut Vec<Value>,
    errors: &mut Vec<Value>,
) -> Option<Value> {
    let manifest_path = Path::new(path);
    if !manifest_path.exists() {
        return None;
    }
    match std::fs::read_to_string(manifest_path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
    {
        Some(value) => {
            checks.push(json!({
                "name": "source_manifest_parse",
                "status": "ok",
                "path": path
            }));
            Some(value)
        }
        None => {
            errors.push(json!({
                "code": "warp_cg.source_manifest_parse_failed",
                "path": path,
                "message": "source.path exists but is not readable JSON"
            }));
            None
        }
    }
}

fn manifest_artifact_path(manifest: &Value, key: &str) -> Option<String> {
    manifest
        .pointer(&format!("/artifacts/{key}"))
        .and_then(manifest_path_value)
        .or_else(|| {
            manifest
                .pointer(&format!("/artifact_paths/{key}"))
                .and_then(manifest_path_value)
        })
        .or_else(|| {
            manifest
                .pointer(&format!("/md_ready_handoff/{key}"))
                .and_then(manifest_path_value)
        })
}

fn manifest_path_value(value: &Value) -> Option<String> {
    value.as_str().map(str::to_string).or_else(|| {
        value
            .get("path")
            .and_then(Value::as_str)
            .map(str::to_string)
    })
}

fn resolve_relative_manifest_path(manifest_path: Option<&str>, path: &str) -> String {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        return candidate.to_string_lossy().to_string();
    }
    manifest_path
        .and_then(|path| Path::new(path).parent())
        .map(|parent| parent.join(candidate).to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

pub(super) fn resolve_source_handoff(source: &CgSource) -> Result<SourceHandoff> {
    let manifest = source.path.as_ref().and_then(|path| {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
    });
    let manifest_path = source.path.as_deref();
    let coordinates = source
        .coordinates
        .clone()
        .or_else(|| {
            manifest
                .as_ref()
                .and_then(|value| manifest_artifact_path(value, "coordinates"))
        })
        .ok_or_else(|| anyhow!("source coordinates are required for source-driven CG execution"))?;
    let topology = source.topology.clone().or_else(|| {
        manifest
            .as_ref()
            .and_then(|value| manifest_artifact_path(value, "topology"))
    });
    let trajectory = source.trajectory.clone().or_else(|| {
        manifest
            .as_ref()
            .and_then(|value| manifest_artifact_path(value, "trajectory"))
    });
    Ok(SourceHandoff {
        coordinates: resolve_relative_manifest_path(manifest_path, &coordinates),
        topology: topology.map(|path| resolve_relative_manifest_path(manifest_path, &path)),
        trajectory: trajectory.map(|path| resolve_relative_manifest_path(manifest_path, &path)),
        coordinate_format: source.format.clone(),
        topology_format: source.topology_format.clone(),
    })
}

pub(super) fn source_selection(source: &CgSource) -> Option<&str> {
    source
        .selection
        .as_deref()
        .or(source.target_selection.as_deref())
}

fn validate_path_exists(field: &str, path: &str, checks: &mut Vec<Value>, errors: &mut Vec<Value>) {
    let exists = Path::new(path).exists();
    checks.push(json!({
        "name": "file_exists",
        "field": field,
        "path": path,
        "status": if exists { "ok" } else { "error" }
    }));
    if !exists {
        errors.push(json!({
            "code": "warp_cg.source_missing",
            "field": field,
            "path": path,
            "message": format!("{field} does not exist")
        }));
    }
}

fn coordinate_atom_count(
    path: &str,
    format: Option<&str>,
    checks: &mut Vec<Value>,
    warnings: &mut Vec<Value>,
) -> Option<usize> {
    match read_molecule(Path::new(path), format, false, true, None) {
        Ok(molecule) => {
            checks.push(json!({
                "name": "coordinate_atom_count",
                "status": "ok",
                "path": path,
                "atom_count": molecule.atoms.len()
            }));
            Some(molecule.atoms.len())
        }
        Err(err) => {
            warnings.push(json!({
                "code": "warp_cg.coordinate_atom_count_unavailable",
                "path": path,
                "message": err.to_string()
            }));
            None
        }
    }
}

fn topology_atom_count(
    path: &str,
    format: Option<&str>,
    checks: &mut Vec<Value>,
    warnings: &mut Vec<Value>,
) -> Option<usize> {
    let is_prmtop = Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("prmtop"));
    let result = if is_prmtop {
        read_prmtop_topology(Path::new(path)).map(|topology| topology.atom_names.len())
    } else {
        read_molecule(Path::new(path), format, false, true, None)
            .map(|molecule| molecule.atoms.len())
    };
    match result {
        Ok(atom_count) => {
            checks.push(json!({
                "name": "topology_atom_count",
                "status": "ok",
                "path": path,
                "atom_count": atom_count
            }));
            Some(atom_count)
        }
        Err(err) => {
            warnings.push(json!({
                "code": "warp_cg.topology_atom_count_unavailable",
                "path": path,
                "message": err.to_string()
            }));
            None
        }
    }
}

fn validate_xtb_available(request: &CgRequest, checks: &mut Vec<Value>, warnings: &mut Vec<Value>) {
    let needs_xtb = request
        .reference_source
        .as_ref()
        .is_some_and(|source| source.kind == "xtb")
        || active_tuning_request(request)
            .is_some_and(|tuning| tuning.enabled && tuning.source == "xtb");
    if !needs_xtb {
        return;
    }
    let found = which::which("xtb").ok();
    checks.push(json!({
        "name": "xtb_executable",
        "status": if found.is_some() { "ok" } else { "missing" },
        "path": found.as_ref().map(|path| path.to_string_lossy().to_string())
    }));
    if found.is_none() {
        warnings.push(json!({
            "code": "warp_cg.xtb_missing",
            "message": "xTB was requested but no xtb executable was found on PATH"
        }));
    }
}

fn validate_optimization_cost(request: &CgRequest, checks: &mut Vec<Value>) {
    if let Some(tuning) = active_tuning_request(request).filter(|tuning| tuning.enabled) {
        let max_evaluations = tuning.max_evaluations.unwrap_or(32);
        checks.push(json!({
            "name": "optimization_runtime_estimate",
            "status": "estimated",
            "source": tuning.source,
            "method": tuning.method,
            "max_evaluations": max_evaluations,
            "cost_class": if tuning.source == "xtb" { "high" } else { "medium" },
            "message": if tuning.source == "xtb" {
                "xTB optimization/MD plus parameter tuning can be minutes to hours depending on system size"
            } else {
                "AA trajectory bonded-stat tuning cost scales with trajectory frames and bead count"
            }
        }));
    }
}

pub(super) fn default_target_terms() -> Vec<String> {
    vec!["bonds".into(), "angles".into(), "dihedrals".into()]
}
