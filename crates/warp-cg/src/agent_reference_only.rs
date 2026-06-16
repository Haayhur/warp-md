use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};

use crate::bonded_terms::BondedTermSet;
use crate::gromacs_ndx::read_gromacs_ndx_mapping;
use crate::mapping::MappingResult;
use crate::parameters::BondedStats;
use crate::reference::{
    ReferenceMetricSource, ReferenceProvider, ReferenceRequest, ReferenceTransformConfig,
    TrajectoryReferenceProvider,
};

use super::agent_artifacts::write_bonded_stats;
use super::agent_reference_result::reference_result_from_data;
use super::agent_render::{beads, mapping_json};
use super::agent_runtime::{native_options, normalized_trajectory_source, run_optimization};
use super::agent_source_validation::default_target_terms;
use super::{
    active_tuning_request, artifact_paths, input_mode, mapping_mode, CgArtifact, CgRequest,
    CgResult, CgSummary, AGENT_SCHEMA_VERSION,
};

pub(super) fn run_reference_only_request(
    request: &CgRequest,
    started: Instant,
) -> Result<CgResult> {
    let trajectory = request
        .trajectory_source
        .as_ref()
        .ok_or_else(|| anyhow!("trajectory_source is required for reference-only NDX mapping"))?;
    let mapping_path = request
        .mapping
        .as_ref()
        .and_then(|mapping| mapping.ndx.as_ref())
        .ok_or_else(|| anyhow!("mapping.mode=ndx requires mapping.ndx"))?;
    let bead_mapping = read_gromacs_ndx_mapping(mapping_path)?;
    let term_set = resolve_reference_term_set(request)?;
    let connections = term_set.bonds_as_connections();
    let mapping = MappingResult {
        bead_names: bead_mapping.bead_names.clone(),
        atom_groups: bead_mapping.atom_indices.clone(),
        connections: connections.clone(),
        bead_features: vec![vec!["gromacs_ndx_mapping".to_string()]; bead_mapping.bead_names.len()],
        bead_formal_charges: vec![0; bead_mapping.bead_names.len()],
    };

    let out_dir = PathBuf::from(&request.output.out_dir);
    std::fs::create_dir_all(&out_dir)?;
    let mut artifacts = Vec::new();
    if request.output.write_mapping_json {
        let mapping_path = out_dir.join(format!("{}_martini_mapping.json", request.name));
        std::fs::write(
            &mapping_path,
            serde_json::to_vec_pretty(&mapping_json(request, &mapping))?,
        )?;
        artifacts.push(CgArtifact {
            path: mapping_path.to_string_lossy().to_string(),
            kind: "martini_mapping_json".to_string(),
        });
    }

    let mapped_name = request
        .output
        .mapped_trajectory
        .clone()
        .unwrap_or_else(|| format!("{}_cg.xtc", request.name));
    let normalized_source = normalized_trajectory_source(request, &trajectory.path);
    let metric_sources = resolve_reference_metric_sources(request);
    let transform = resolve_reference_transform(request);
    let mut provider = TrajectoryReferenceProvider::new(
        "aa_trajectory_ndx",
        &trajectory.path,
        native_options(request, normalized_source.as_ref()),
    );
    let reference = provider.load_reference(&ReferenceRequest {
        name: &request.name,
        out_dir: &out_dir,
        mapped_trajectory_name: Some(&mapped_name),
        mapping: &bead_mapping,
        connections: &connections,
        term_set: Some(&term_set),
        metric_sources: &metric_sources,
        transform: transform.as_ref(),
    })?;
    let reference_result = reference_result_from_data(&reference);
    for artifact in &reference.artifacts {
        artifacts.push(CgArtifact {
            path: artifact.path.to_string_lossy().to_string(),
            kind: artifact.kind.clone(),
        });
    }
    let bonded_stats = BondedStats {
        bonds: reference.bonded_stats.bonds.clone(),
        angles: reference.bonded_stats.angles.clone(),
        dihedrals: reference.bonded_stats.dihedrals.clone(),
    };

    let active_tuning = active_tuning_request(request);
    let optimization_result = active_tuning
        .filter(|tuning| tuning.enabled)
        .map(|tuning| {
            run_optimization(
                tuning,
                &bonded_stats,
                reference.target_set.as_ref(),
                (reference.metadata.frames_read > 0).then_some(reference.metadata.frames_read),
                Some(&reference.metrics),
                &out_dir,
                &request.name,
                request.forcefield.as_ref(),
                &mut artifacts,
            )
        })
        .transpose()?;
    write_bonded_stats(
        &request.name,
        &out_dir,
        &mut artifacts,
        &bonded_stats.bonds,
        &bonded_stats.angles,
        &bonded_stats.dihedrals,
    )?;

    Ok(CgResult {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        status: "ok".to_string(),
        exit_code: 0,
        name: request.name.clone(),
        summary: CgSummary {
            input_mode: input_mode(request).to_string(),
            mapping_mode: mapping_mode(request).to_string(),
            aa_atom_count: None,
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
        reference: Some(reference_result),
        optimization: optimization_result,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

pub(super) fn is_reference_only_request(request: &CgRequest) -> bool {
    request.smiles.is_none()
        && request.repeat_smiles.is_none()
        && request.source.is_none()
        && request.trajectory_source.is_some()
        && request
            .mapping
            .as_ref()
            .is_some_and(|mapping| mapping.mode == "ndx")
}

fn resolve_reference_term_set(request: &CgRequest) -> Result<BondedTermSet> {
    let source = request
        .reference_source
        .as_ref()
        .and_then(|reference| reference.bonded_terms.as_ref())
        .ok_or_else(|| {
            anyhow!(
                "reference-only NDX mapping requires reference_source.bonded_terms with Gromacs ITP/TOP terms"
            )
        })?;
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
