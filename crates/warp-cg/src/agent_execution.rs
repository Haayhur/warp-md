use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{anyhow, Context, Result};

use crate::bonded_terms::BondedTermSet;
use crate::mapping::{map_molecule_with_options, MappingOptions};
use crate::molecule::Molecule;
use crate::optimize::OptimizationReport;
use crate::parameters::{AngleStats, BondStats, BondedStats, DihedralStats};
use crate::reference::{
    PrecomputedStatsReferenceProvider, ReferenceData, ReferenceMetricSource, ReferenceProvider,
    ReferenceRequest, ReferenceTransformConfig, TrajectoryReferenceProvider, XtbReferenceProvider,
};
use crate::trajectory::BeadMapping;

use super::agent_artifacts::{write_bonded_stats, write_topology_top};
use super::agent_reference_only::{is_reference_only_request, run_reference_only_request};
use super::agent_reference_result::reference_result_from_data;
use super::agent_render::{
    beads, bonded_parameter_map_json, mapping_json, render_cg_pdb, render_martini_itp,
    render_source_cg_pdb, render_source_martini_itp, source_bonded_parameter_map_json,
    source_mapping_json,
};
use super::agent_runtime::{
    native_options, normalized_trajectory_source, resolve_output_path, run_optimization,
    source_native_options, xtb_run_config,
};
use super::agent_source_mapping::build_source_mapping;
use super::agent_source_validation::{default_target_terms, resolve_source_handoff};
use super::{
    active_tuning_request, artifact_paths, input_mode, mapping_mode, CgArtifact, CgRequest,
    CgResult, CgSummary, SourceMappingResult, AGENT_SCHEMA_VERSION,
};

pub(super) fn run_request(request: &CgRequest, started: Instant) -> Result<CgResult> {
    if request.source.is_some() {
        return run_source_request(request, started);
    }
    if is_reference_only_request(request) {
        return run_reference_only_request(request, started);
    }
    let molecule_identity = request
        .smiles
        .as_ref()
        .or(request.repeat_smiles.as_ref())
        .ok_or_else(|| {
            anyhow!(
                "warp-cg execution requires smiles, repeat_smiles, or an executable source handoff"
            )
        })?;
    let mol = Molecule::from_smiles(molecule_identity)?;
    let mapping = map_molecule_with_options(&mol, &small_molecule_mapping_options(request));
    let out_dir = PathBuf::from(&request.output.out_dir);
    std::fs::create_dir_all(&out_dir)?;

    let mut artifacts = Vec::new();
    if request.output.write_mapping_json {
        let mapping_path = out_dir.join(format!("{}_martini_mapping.json", request.name));
        let mapping_value = mapping_json(request, &mapping);
        std::fs::write(&mapping_path, serde_json::to_vec_pretty(&mapping_value)?)?;
        artifacts.push(CgArtifact {
            path: mapping_path.to_string_lossy().to_string(),
            kind: "martini_mapping_json".to_string(),
        });
    }

    let mut bond_stats: Vec<BondStats> = Vec::new();
    let mut angle_stats: Vec<AngleStats> = Vec::new();
    let mut dihedral_stats: Vec<DihedralStats> = Vec::new();
    let mut first_cg_coords: Option<Vec<[f32; 3]>> = None;
    let mut reference_targets = None;
    let mut reference_result = None;
    let mut reference_metrics = BTreeMap::new();
    let mut reference_frames_read = None;
    let bead_mapping = BeadMapping {
        bead_names: mapping.bead_names.clone(),
        atom_indices: mapping.atom_groups.clone(),
    };
    if let Some(reference_data) = load_small_molecule_reference(
        request,
        molecule_identity,
        &out_dir,
        &bead_mapping,
        &mapping.connections,
    )? {
        append_reference_artifacts(&mut artifacts, &reference_data);
        reference_result = Some(reference_result_from_data(&reference_data));
        bond_stats = reference_data.bonded_stats.bonds;
        angle_stats = reference_data.bonded_stats.angles;
        dihedral_stats = reference_data.bonded_stats.dihedrals;
        reference_metrics = reference_data.metrics.clone();
        reference_frames_read = nonzero_frames_read(reference_data.metadata.frames_read);
        first_cg_coords = reference_data.first_cg_coords;
        reference_targets = reference_data.target_set;
    }

    let active_tuning = active_tuning_request(request);
    let optimization_result = active_tuning
        .filter(|tuning| tuning.enabled)
        .map(|tuning| {
            let bonded_stats = BondedStats {
                bonds: bond_stats.clone(),
                angles: angle_stats.clone(),
                dihedrals: dihedral_stats.clone(),
            };
            run_optimization(
                tuning,
                &bonded_stats,
                reference_targets.as_ref(),
                reference_frames_read,
                Some(&reference_metrics),
                &out_dir,
                &request.name,
                request.forcefield.as_ref(),
                &mut artifacts,
            )
        })
        .transpose()?;

    write_small_molecule_artifacts(
        request,
        &mapping,
        &out_dir,
        &mut artifacts,
        &bond_stats,
        &angle_stats,
        &dihedral_stats,
        first_cg_coords.as_deref(),
        reference_targets.as_ref(),
        optimization_result
            .as_ref()
            .and_then(|tuning| tuning.report.as_ref()),
    )?;

    Ok(CgResult {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        status: "ok".to_string(),
        exit_code: 0,
        name: request.name.clone(),
        summary: CgSummary {
            input_mode: input_mode(request).to_string(),
            mapping_mode: mapping_mode(request).to_string(),
            aa_atom_count: Some(mol.graph.node_count()),
            cg_bead_count: mapping.bead_names.len(),
            mapped_residue_count: None,
            optimized_terms: optimization_result
                .as_ref()
                .map(|_| {
                    active_tuning
                        .and_then(|tuning| tuning.target_terms.clone())
                        .unwrap_or_else(default_target_terms)
                })
                .unwrap_or_default(),
            optimization_source: active_tuning
                .filter(|tuning| tuning.enabled)
                .map(|tuning| tuning.source.clone()),
        },
        bead_count: mapping.bead_names.len(),
        beads: beads(&mapping),
        connections: mapping.connections.iter().map(|&(i, j)| [i, j]).collect(),
        warnings: Vec::new(),
        mapping_summary: None,
        artifact_paths: artifact_paths(&artifacts),
        artifacts,
        reference: reference_result,
        optimization: optimization_result,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

fn small_molecule_mapping_options(request: &CgRequest) -> MappingOptions {
    MappingOptions {
        target_bead_size: request
            .mapping
            .as_ref()
            .and_then(|mapping| mapping.target_bead_size)
            .unwrap_or_else(|| MappingOptions::default().target_bead_size),
    }
}

fn write_small_molecule_artifacts(
    request: &CgRequest,
    mapping: &crate::mapping::MappingResult,
    out_dir: &Path,
    artifacts: &mut Vec<CgArtifact>,
    bond_stats: &[BondStats],
    angle_stats: &[AngleStats],
    dihedral_stats: &[DihedralStats],
    first_cg_coords: Option<&[[f32; 3]]>,
    reference_targets: Option<&crate::reference::ReferenceTargetSet>,
    optimization_report: Option<&OptimizationReport>,
) -> Result<()> {
    if request.output.write_cg_pdb {
        let pdb_name = request
            .output
            .cg_pdb
            .clone()
            .unwrap_or_else(|| format!("{}_cg.pdb", request.name));
        let pdb_path = resolve_output_path(out_dir, &pdb_name);
        let pdb = render_cg_pdb(&request.name, mapping, first_cg_coords);
        std::fs::write(&pdb_path, pdb)?;
        artifacts.push(CgArtifact {
            path: pdb_path.to_string_lossy().to_string(),
            kind: "coarse_grained_pdb".to_string(),
        });
    }
    write_bonded_stats(
        &request.name,
        out_dir,
        artifacts,
        bond_stats,
        angle_stats,
        dihedral_stats,
    )?;
    if request.output.write_topology_itp {
        let itp_path = out_dir.join(format!("{}_martini.itp", request.name));
        let itp = render_martini_itp(
            &request.name,
            mapping,
            bond_stats,
            angle_stats,
            dihedral_stats,
            reference_targets,
            optimization_report,
        );
        std::fs::write(&itp_path, itp)?;
        artifacts.push(CgArtifact {
            path: itp_path.to_string_lossy().to_string(),
            kind: "martini_topology_itp".to_string(),
        });
        if request.output.write_bonded_parameter_map {
            let map_path = out_dir.join(format!("{}_bonded_parameter_map.json", request.name));
            let parameter_map = bonded_parameter_map_json(
                request,
                mapping,
                bond_stats,
                angle_stats,
                dihedral_stats,
                reference_targets,
                optimization_report,
            );
            std::fs::write(&map_path, serde_json::to_vec_pretty(&parameter_map)?)?;
            artifacts.push(CgArtifact {
                path: map_path.to_string_lossy().to_string(),
                kind: "bonded_parameter_map_json".to_string(),
            });
        }
    }
    write_topology_top(request, out_dir, artifacts)
}

fn load_small_molecule_reference(
    request: &CgRequest,
    molecule_identity: &str,
    out_dir: &Path,
    bead_mapping: &BeadMapping,
    connections: &[(usize, usize)],
) -> Result<Option<ReferenceData>> {
    let mapped_name = request
        .output
        .mapped_trajectory
        .clone()
        .unwrap_or_else(|| format!("{}_cg.xtc", request.name));
    let reference_kind = request
        .reference_source
        .as_ref()
        .map(|source| source.kind.as_str())
        .unwrap_or("external");
    let term_set = resolve_reference_term_set(request)?;
    let metric_sources = resolve_reference_metric_sources(request);
    let transform = resolve_reference_transform(request);
    let request_context = ReferenceRequest {
        name: &request.name,
        out_dir,
        mapped_trajectory_name: Some(&mapped_name),
        mapping: bead_mapping,
        connections,
        term_set: term_set.as_ref(),
        metric_sources: &metric_sources,
        transform: transform.as_ref(),
    };

    if reference_kind == "precomputed" {
        return load_precomputed_reference(request, &request_context).map(Some);
    }

    if reference_kind == "xtb" {
        let xtb_out_dir = request
            .reference_source
            .as_ref()
            .and_then(|source| source.xtb.as_ref())
            .and_then(|xtb| xtb.work_dir.as_ref())
            .map(PathBuf::from);
        let xtb_config = request
            .reference_source
            .as_ref()
            .and_then(|source| source.xtb.as_ref())
            .map(xtb_run_config)
            .unwrap_or_default();
        let mut provider = XtbReferenceProvider::new(
            molecule_identity,
            xtb_config,
            xtb_out_dir,
            native_options(request, None),
        );
        return provider.load_reference(&request_context).map(Some);
    }

    let Some(source) = request.trajectory_source.as_ref() else {
        return Ok(None);
    };
    let normalized_source = normalized_trajectory_source(request, &source.path);
    let mut provider = TrajectoryReferenceProvider::new(
        "external_trajectory",
        &source.path,
        native_options(request, normalized_source.as_ref()),
    );
    provider.load_reference(&request_context).map(Some)
}

fn load_precomputed_reference(
    request: &CgRequest,
    request_context: &ReferenceRequest<'_>,
) -> Result<ReferenceData> {
    let precomputed = request
        .reference_source
        .as_ref()
        .and_then(|source| source.precomputed.as_ref())
        .ok_or_else(|| {
            anyhow!("reference_source.kind=precomputed requires reference_source.precomputed")
        })?;
    let source_kind = precomputed
        .source_kind
        .as_deref()
        .unwrap_or("precomputed_reference");
    let mut provider = PrecomputedStatsReferenceProvider::new(source_kind, BondedStats::default())
        .with_target_set_path(&precomputed.target_set);
    provider.load_reference(request_context)
}

fn resolve_reference_term_set(request: &CgRequest) -> Result<Option<BondedTermSet>> {
    let Some(source) = request
        .reference_source
        .as_ref()
        .and_then(|reference| reference.bonded_terms.as_ref())
    else {
        return Ok(None);
    };
    if source.kind != "gromacs_topology" && source.kind != "gromacs_itp" {
        return Err(anyhow!(
            "reference_source.bonded_terms.kind must be gromacs_topology or gromacs_itp"
        ));
    }
    let topology = std::fs::read_to_string(&source.path).with_context(|| {
        format!(
            "failed to read reference_source.bonded_terms.path '{}'",
            source.path
        )
    })?;
    BondedTermSet::from_gromacs_topology_str(&topology, &source.molecule_type)
        .map_err(|err| anyhow!("{err}"))
        .map(Some)
        .with_context(|| {
            format!(
                "failed to parse bonded terms for molecule_type '{}' from '{}'",
                source.molecule_type, source.path
            )
        })
}

fn resolve_reference_metric_sources(request: &CgRequest) -> Vec<ReferenceMetricSource> {
    request
        .reference_source
        .as_ref()
        .map(|reference| {
            reference
                .metrics
                .iter()
                .map(|source| ReferenceMetricSource {
                    path: PathBuf::from(&source.path),
                    kind: source.kind.clone(),
                    namespace: source.namespace.clone(),
                    artifact_kind: source.artifact_kind.clone(),
                })
                .collect()
        })
        .unwrap_or_default()
}

fn resolve_reference_transform(request: &CgRequest) -> Option<ReferenceTransformConfig> {
    request
        .reference_source
        .as_ref()
        .and_then(|reference| reference.transform.as_ref())
        .map(|transform| ReferenceTransformConfig {
            bond_scaling: transform.bond_scaling,
            min_bond_length_nm: transform.min_bond_length_nm,
            specific_bond_lengths_nm: transform.specific_bond_lengths_nm.clone(),
            rg_offset_nm: transform.rg_offset_nm,
        })
}

fn append_reference_artifacts(artifacts: &mut Vec<CgArtifact>, reference: &ReferenceData) {
    for artifact in &reference.artifacts {
        artifacts.push(CgArtifact {
            path: artifact.path.to_string_lossy().to_string(),
            kind: artifact.kind.clone(),
        });
    }
}

fn nonzero_frames_read(frames_read: usize) -> Option<usize> {
    (frames_read > 0).then_some(frames_read)
}

fn run_source_request(request: &CgRequest, started: Instant) -> Result<CgResult> {
    let source = request
        .source
        .as_ref()
        .ok_or_else(|| anyhow!("source is required"))?;
    let handoff = resolve_source_handoff(source)?;
    let source_mapping = build_source_mapping(request, &handoff)?;
    let out_dir = PathBuf::from(&request.output.out_dir);
    std::fs::create_dir_all(&out_dir)?;

    let mut artifacts = Vec::new();
    if request.output.write_mapping_json {
        let mapping_path = out_dir.join(format!("{}_martini_mapping.json", request.name));
        std::fs::write(
            &mapping_path,
            serde_json::to_vec_pretty(&source_mapping_json(request, &source_mapping))?,
        )?;
        artifacts.push(CgArtifact {
            path: mapping_path.to_string_lossy().to_string(),
            kind: "martini_mapping_json".to_string(),
        });
        let provenance_path = out_dir.join(format!("{}_aa_to_cg_provenance.json", request.name));
        std::fs::write(
            &provenance_path,
            serde_json::to_vec_pretty(&source_mapping.provenance)?,
        )?;
        artifacts.push(CgArtifact {
            path: provenance_path.to_string_lossy().to_string(),
            kind: "aa_to_cg_mapping_provenance".to_string(),
        });
        let template_path = out_dir.join(format!("{}_mapping_template.json", request.name));
        std::fs::write(
            &template_path,
            serde_json::to_vec_pretty(&source_mapping.templates)?,
        )?;
        artifacts.push(CgArtifact {
            path: template_path.to_string_lossy().to_string(),
            kind: "mapping_template_json".to_string(),
        });
    }

    let mut bond_stats: Vec<BondStats> = Vec::new();
    let mut angle_stats: Vec<AngleStats> = Vec::new();
    let mut dihedral_stats: Vec<DihedralStats> = Vec::new();
    let mut first_cg_coords = Some(
        source_mapping
            .beads
            .iter()
            .map(|bead| bead.coord)
            .collect::<Vec<_>>(),
    );
    let mut reference_targets = None;
    let mut reference_result = None;
    let mut reference_metrics = BTreeMap::new();
    let mut reference_frames_read = None;

    if request
        .reference_source
        .as_ref()
        .is_some_and(|source| source.kind == "precomputed")
    {
        let bead_mapping = BeadMapping {
            bead_names: source_mapping.mapping.bead_names.clone(),
            atom_indices: source_mapping.mapping.atom_groups.clone(),
        };
        let metric_sources = resolve_reference_metric_sources(request);
        let reference = load_precomputed_reference(
            request,
            &ReferenceRequest {
                name: &request.name,
                out_dir: &out_dir,
                mapped_trajectory_name: None,
                mapping: &bead_mapping,
                connections: &source_mapping.mapping.connections,
                term_set: None,
                metric_sources: &metric_sources,
                transform: None,
            },
        )?;
        append_reference_artifacts(&mut artifacts, &reference);
        reference_result = Some(reference_result_from_data(&reference));
        bond_stats = reference.bonded_stats.bonds;
        angle_stats = reference.bonded_stats.angles;
        dihedral_stats = reference.bonded_stats.dihedrals;
        reference_metrics = reference.metrics.clone();
        reference_frames_read = nonzero_frames_read(reference.metadata.frames_read);
        reference_targets = reference.target_set;
    }

    if reference_targets.is_none() {
        if let Some(input_traj) = handoff.trajectory.clone() {
            let mapped_name = request
                .output
                .mapped_trajectory
                .clone()
                .unwrap_or_else(|| format!("{}_cg.xtc", request.name));
            let mut provider = TrajectoryReferenceProvider::new(
                "source_trajectory",
                input_traj,
                source_native_options(source, &handoff),
            );
            let bead_mapping = BeadMapping {
                bead_names: source_mapping.mapping.bead_names.clone(),
                atom_indices: source_mapping.mapping.atom_groups.clone(),
            };
            let term_set = resolve_reference_term_set(request)?
                .or_else(|| source_mapping.bonded_terms.clone());
            let metric_sources = resolve_reference_metric_sources(request);
            let transform = resolve_reference_transform(request);
            let reference = provider.load_reference(&ReferenceRequest {
                name: &request.name,
                out_dir: &out_dir,
                mapped_trajectory_name: Some(&mapped_name),
                mapping: &bead_mapping,
                connections: &source_mapping.mapping.connections,
                term_set: term_set.as_ref(),
                metric_sources: &metric_sources,
                transform: transform.as_ref(),
            })?;
            append_reference_artifacts(&mut artifacts, &reference);
            reference_result = Some(reference_result_from_data(&reference));
            bond_stats = reference.bonded_stats.bonds;
            angle_stats = reference.bonded_stats.angles;
            dihedral_stats = reference.bonded_stats.dihedrals;
            reference_metrics = reference.metrics.clone();
            reference_frames_read = nonzero_frames_read(reference.metadata.frames_read);
            reference_targets = reference.target_set;
            first_cg_coords = reference.first_cg_coords.or(first_cg_coords);
        }
    }

    let active_tuning = active_tuning_request(request);
    let optimization_result = active_tuning
        .filter(|tuning| tuning.enabled)
        .map(|tuning| {
            let bonded_stats = BondedStats {
                bonds: bond_stats.clone(),
                angles: angle_stats.clone(),
                dihedrals: dihedral_stats.clone(),
            };
            run_optimization(
                tuning,
                &bonded_stats,
                reference_targets.as_ref(),
                reference_frames_read,
                Some(&reference_metrics),
                &out_dir,
                &request.name,
                request.forcefield.as_ref(),
                &mut artifacts,
            )
        })
        .transpose()?;

    write_source_artifacts(
        request,
        &source_mapping,
        &out_dir,
        &mut artifacts,
        &bond_stats,
        &angle_stats,
        &dihedral_stats,
        first_cg_coords.as_deref(),
        reference_targets.as_ref(),
        optimization_result
            .as_ref()
            .and_then(|tuning| tuning.report.as_ref()),
    )?;

    Ok(CgResult {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        status: "ok".to_string(),
        exit_code: 0,
        name: request.name.clone(),
        summary: CgSummary {
            input_mode: input_mode(request).to_string(),
            mapping_mode: mapping_mode(request).to_string(),
            aa_atom_count: Some(source_mapping.aa_atom_count),
            cg_bead_count: source_mapping.mapping.bead_names.len(),
            mapped_residue_count: Some(source_mapping.residue_count),
            optimized_terms: optimization_result
                .as_ref()
                .map(|_| {
                    active_tuning
                        .and_then(|tuning| tuning.target_terms.clone())
                        .unwrap_or_else(default_target_terms)
                })
                .unwrap_or_default(),
            optimization_source: active_tuning
                .filter(|tuning| tuning.enabled)
                .map(|tuning| tuning.source.clone()),
        },
        bead_count: source_mapping.mapping.bead_names.len(),
        beads: beads(&source_mapping.mapping),
        connections: source_mapping
            .mapping
            .connections
            .iter()
            .map(|&(i, j)| [i, j])
            .collect(),
        warnings: source_mapping.warnings,
        mapping_summary: Some(source_mapping.mapping_summary),
        artifact_paths: artifact_paths(&artifacts),
        artifacts,
        reference: reference_result,
        optimization: optimization_result,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

fn write_source_artifacts(
    request: &CgRequest,
    source_mapping: &SourceMappingResult,
    out_dir: &Path,
    artifacts: &mut Vec<CgArtifact>,
    bond_stats: &[BondStats],
    angle_stats: &[AngleStats],
    dihedral_stats: &[DihedralStats],
    first_cg_coords: Option<&[[f32; 3]]>,
    reference_targets: Option<&crate::reference::ReferenceTargetSet>,
    optimization_report: Option<&OptimizationReport>,
) -> Result<()> {
    if request.output.write_cg_pdb {
        let pdb_name = request
            .output
            .cg_pdb
            .clone()
            .unwrap_or_else(|| format!("{}_cg.pdb", request.name));
        let pdb_path = resolve_output_path(out_dir, &pdb_name);
        let pdb = render_source_cg_pdb(source_mapping, first_cg_coords);
        std::fs::write(&pdb_path, pdb)?;
        artifacts.push(CgArtifact {
            path: pdb_path.to_string_lossy().to_string(),
            kind: "coarse_grained_pdb".to_string(),
        });
    }
    write_bonded_stats(
        &request.name,
        out_dir,
        artifacts,
        bond_stats,
        angle_stats,
        dihedral_stats,
    )?;
    if request.output.write_topology_itp {
        let itp_path = out_dir.join(format!("{}_martini.itp", request.name));
        let itp = render_source_martini_itp(
            &request.name,
            source_mapping,
            bond_stats,
            angle_stats,
            dihedral_stats,
            reference_targets,
            optimization_report,
        );
        std::fs::write(&itp_path, itp)?;
        artifacts.push(CgArtifact {
            path: itp_path.to_string_lossy().to_string(),
            kind: "martini_topology_itp".to_string(),
        });
        if request.output.write_bonded_parameter_map {
            let map_path = out_dir.join(format!("{}_bonded_parameter_map.json", request.name));
            let parameter_map = source_bonded_parameter_map_json(
                request,
                source_mapping,
                bond_stats,
                angle_stats,
                dihedral_stats,
                reference_targets,
                optimization_report,
            );
            std::fs::write(&map_path, serde_json::to_vec_pretty(&parameter_map)?)?;
            artifacts.push(CgArtifact {
                path: map_path.to_string_lossy().to_string(),
                kind: "bonded_parameter_map_json".to_string(),
            });
        }
    }
    write_topology_top(request, out_dir, artifacts)
}
