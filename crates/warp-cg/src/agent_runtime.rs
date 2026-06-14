use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};

use crate::bonded_terms::BondedTermSet;
use crate::optimize::{
    optimize_bonded_terms, optimize_reference_targets,
    optimize_reference_targets_with_named_evaluator, CandidateTrajectoryExtractionConfig,
    JsonFileEvaluatorCommand, JsonFileEvaluatorConfig, JsonFileObjectiveEvaluator,
    OptimizationConfig,
};
use crate::parameters::BondedStats;
use crate::reference::ReferenceTargetSet;
use crate::trajectory::{BeadMapping, NativeTrajectoryOptions};
use crate::xtb::XtbRunConfig;

use super::{
    CandidateTrajectoryExtractionRequest, CgArtifact, CgRequest, CgSource,
    JsonFileEvaluatorCommandRequest, ParameterTuningRequest, ParameterTuningResult, SourceHandoff,
    TrajectorySource, XtbRequest,
};

pub(super) fn run_optimization(
    tuning: &ParameterTuningRequest,
    bonded_stats: &BondedStats,
    reference_targets: Option<&ReferenceTargetSet>,
    out_dir: &Path,
    name: &str,
    artifacts: &mut Vec<CgArtifact>,
) -> Result<ParameterTuningResult> {
    let config = OptimizationConfig {
        method: tuning.method.clone(),
        objective: tuning.objective.clone(),
        max_evaluations: tuning.max_evaluations.unwrap_or(32),
        seed: tuning.seed.unwrap_or(42),
        swarm_size: tuning.swarm_size,
        pso: tuning.pso.as_ref().map(Into::into),
        bo: tuning.bo.as_ref().map(Into::into),
    };
    let selected_terms = tuning.target_terms.clone().unwrap_or_default();
    let filtered_stats = filter_bonded_stats(bonded_stats, &selected_terms);
    let filtered_targets = reference_targets.map(|targets| targets.filter_terms(&selected_terms));
    let report = if let Some(evaluator) = tuning.evaluator.as_ref() {
        let Some(targets) = filtered_targets.as_ref() else {
            return Err(anyhow!(
                "optimization.evaluator requires reference targets from reference_source"
            ));
        };
        let mut evaluator = json_file_evaluator(evaluator, targets, out_dir)?;
        optimize_reference_targets_with_named_evaluator(targets, &mut evaluator, &config, &[])
    } else if let Some(targets) = filtered_targets.as_ref() {
        optimize_reference_targets(targets, &config)
    } else {
        optimize_bonded_terms(&filtered_stats, &config)
    };
    let report_path = out_dir.join(format!("{}_tuning_report.json", name));
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    artifacts.push(CgArtifact {
        path: report_path.to_string_lossy().to_string(),
        kind: "bonded_optimization_report".to_string(),
    });

    Ok(ParameterTuningResult {
        status: report.status.clone(),
        method: tuning.method.clone(),
        source: tuning.source.clone(),
        objective: tuning.objective.clone(),
        message: report.message.clone(),
        report: Some(report),
    })
}

fn json_file_evaluator(
    evaluator: &super::ObjectiveEvaluatorRequest,
    reference_targets: &ReferenceTargetSet,
    out_dir: &Path,
) -> Result<JsonFileObjectiveEvaluator> {
    let json_file = evaluator
        .json_file
        .as_ref()
        .ok_or_else(|| anyhow!("optimization.evaluator.json_file is required"))?;
    Ok(JsonFileObjectiveEvaluator::new(JsonFileEvaluatorConfig {
        work_dir: resolve_output_path(out_dir, &json_file.work_dir),
        request_filename: json_file
            .request_filename
            .clone()
            .unwrap_or_else(|| "candidate.json".to_string()),
        result_filename: json_file
            .result_filename
            .clone()
            .unwrap_or_else(|| "result.json".to_string()),
        command: json_file.command.as_ref().map(json_file_command),
        reference_targets: Some(reference_targets.clone()),
        candidate_extraction: json_file
            .candidate_extraction
            .as_ref()
            .map(candidate_trajectory_extraction_config)
            .transpose()?,
    }))
}

fn json_file_command(command: &JsonFileEvaluatorCommandRequest) -> JsonFileEvaluatorCommand {
    JsonFileEvaluatorCommand {
        program: command.program.clone(),
        args: command.args.clone(),
    }
}

fn candidate_trajectory_extraction_config(
    request: &CandidateTrajectoryExtractionRequest,
) -> Result<CandidateTrajectoryExtractionConfig> {
    Ok(CandidateTrajectoryExtractionConfig {
        mapping: BeadMapping {
            bead_names: request.mapping.bead_names.clone(),
            atom_indices: request.mapping.atom_indices.clone(),
        },
        connections: request.connections.iter().map(|[i, j]| (*i, *j)).collect(),
        term_set: request
            .bonded_terms
            .as_ref()
            .map(candidate_bonded_term_set)
            .transpose()?,
        options: NativeTrajectoryOptions {
            topology: request.topology.clone(),
            topology_format: request.topology_format.clone(),
            format: request.format.clone(),
            start: request.start,
            stop: request.stop,
            stride: request.stride,
            length_scale: request.length_scale,
            target_selection: request.target_selection.clone(),
            atom_indices: request.atom_indices.clone(),
            mass_weighted: request.mass_weighted.unwrap_or(false),
            make_whole: request.make_whole.unwrap_or(false),
            chunk_frames: request.chunk_frames,
        },
        transform: None,
        mapped_trajectory_name: request.mapped_trajectory_name.clone(),
    })
}

fn candidate_bonded_term_set(source: &super::BondedTermSource) -> Result<BondedTermSet> {
    let topology = std::fs::read_to_string(&source.path).map_err(|err| {
        anyhow!(
            "failed to read evaluator candidate_extraction.bonded_terms.path '{}': {err}",
            source.path
        )
    })?;
    BondedTermSet::from_gromacs_topology_str(&topology, &source.molecule_type)
        .map_err(|err| anyhow!("{err}"))
        .map_err(|err| {
            anyhow!(
                "failed to parse evaluator candidate_extraction.bonded_terms for molecule_type '{}' from '{}': {err}",
                source.molecule_type,
                source.path
            )
        })
}

fn filter_bonded_stats(stats: &BondedStats, terms: &[String]) -> BondedStats {
    if terms.is_empty() {
        return stats.clone();
    }
    BondedStats {
        bonds: if has_term(terms, "bonds") {
            stats.bonds.clone()
        } else {
            Vec::new()
        },
        angles: if has_term(terms, "angles") {
            stats.angles.clone()
        } else {
            Vec::new()
        },
        dihedrals: if has_term(terms, "dihedrals") {
            stats.dihedrals.clone()
        } else {
            Vec::new()
        },
    }
}

fn has_term(terms: &[String], target: &str) -> bool {
    terms.iter().any(|term| term == target)
}

pub(super) fn resolve_output_path(out_dir: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        out_dir.join(candidate)
    }
}

pub(super) fn normalized_trajectory_source(
    request: &CgRequest,
    path: &str,
) -> Option<TrajectorySource> {
    request.trajectory_source.clone().or_else(|| {
        Some(TrajectorySource {
            path: path.to_string(),
            topology: request.topology.clone(),
            format: None,
            topology_format: None,
            kind: "external".to_string(),
            stride: None,
            start: None,
            stop: None,
            length_scale: None,
            target_selection: None,
            environment_selection: None,
            atom_indices: None,
            mass_weighted: None,
            make_whole: None,
        })
    })
}

pub(super) fn native_options(
    request: &CgRequest,
    source: Option<&TrajectorySource>,
) -> NativeTrajectoryOptions {
    NativeTrajectoryOptions {
        topology: source
            .and_then(|source| source.topology.clone())
            .or_else(|| request.topology.clone()),
        topology_format: source.and_then(|source| source.topology_format.clone()),
        format: source.and_then(|source| source.format.clone()),
        start: source.and_then(|source| source.start),
        stop: source.and_then(|source| source.stop),
        stride: source.and_then(|source| source.stride),
        length_scale: source.and_then(|source| source.length_scale),
        target_selection: source.and_then(|source| source.target_selection.clone()),
        atom_indices: source.and_then(|source| source.atom_indices.clone()),
        mass_weighted: source
            .and_then(|source| source.mass_weighted)
            .unwrap_or(false),
        make_whole: source.and_then(|source| source.make_whole).unwrap_or(false),
        chunk_frames: None,
    }
}

pub(super) fn source_native_options(
    source: &CgSource,
    handoff: &SourceHandoff,
) -> NativeTrajectoryOptions {
    NativeTrajectoryOptions {
        topology: handoff
            .topology
            .clone()
            .or_else(|| Some(handoff.coordinates.clone())),
        topology_format: handoff.topology_format.clone(),
        format: None,
        start: None,
        stop: None,
        stride: None,
        length_scale: None,
        target_selection: source
            .target_selection
            .as_ref()
            .filter(|selection| selection.as_str() != "polymer")
            .cloned(),
        atom_indices: None,
        mass_weighted: false,
        make_whole: false,
        chunk_frames: None,
    }
}

pub(super) fn xtb_run_config(request: &XtbRequest) -> XtbRunConfig {
    let default = XtbRunConfig::default();
    XtbRunConfig {
        temperature_k: request.temperature_k.unwrap_or(default.temperature_k),
        time_ps: request.time_ps.unwrap_or(default.time_ps),
        timestep_fs: request.timestep_fs.unwrap_or(default.timestep_fs),
        dump_fs: request.dump_fs.unwrap_or(default.dump_fs),
        gfn: request.gfn.clone().unwrap_or(default.gfn),
        seed: request.seed.unwrap_or(default.seed),
    }
}

pub(super) fn validate_positive(value: Option<f64>, field: &str) -> Result<()> {
    if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err(anyhow!("{field} must be finite and greater than zero"));
    }
    Ok(())
}
