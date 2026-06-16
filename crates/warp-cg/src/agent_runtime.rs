use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use serde_json::{json, Value};

use crate::bonded_terms::BondedTermSet;
use crate::forcefield::materialize_request_forcefield;
use crate::optimize::{
    direct_statistics_report, direct_statistics_report_from_targets, optimize_bonded_terms,
    optimize_reference_targets, optimize_reference_targets_with_named_evaluator,
    CandidateTrajectoryExtractionConfig, JsonFileEvaluatorCommand, JsonFileEvaluatorConfig,
    JsonFileObjectiveEvaluator, OptimizationConfig, OptimizationReport,
    StructuralMetricScoringConfig,
};
use crate::parameters::BondedStats;
use crate::reference::ReferenceTargetSet;
use crate::trajectory::{BeadMapping, NativeSasaOptions, NativeTrajectoryOptions};
use crate::xtb::XtbRunConfig;

use super::{
    CandidateTrajectoryExtractionRequest, CgArtifact, CgRequest, CgSource, ForcefieldRequest,
    JsonFileEvaluatorCommandRequest, ParameterTuningRequest, ParameterTuningResult,
    SimulationRunnerRequest, SourceHandoff, TrajectorySource, XtbRequest,
};

pub(super) fn run_optimization(
    tuning: &ParameterTuningRequest,
    bonded_stats: &BondedStats,
    reference_targets: Option<&ReferenceTargetSet>,
    reference_frames_read: Option<usize>,
    reference_metrics: Option<&BTreeMap<String, f64>>,
    out_dir: &Path,
    name: &str,
    forcefield: Option<&ForcefieldRequest>,
    artifacts: &mut Vec<CgArtifact>,
) -> Result<ParameterTuningResult> {
    let method = normalized_tuning_method(&tuning.method);
    let config = OptimizationConfig {
        method: method.clone(),
        objective: tuning.objective.clone(),
        max_evaluations: tuning.max_evaluations.unwrap_or(32),
        seed: tuning.seed.unwrap_or(42),
        swarm_size: tuning.swarm_size,
        pso: tuning.pso.as_ref().map(Into::into),
        bo: tuning.bo.as_ref().map(Into::into),
        initial_parameters: tuning.initial_parameters.clone(),
    };
    let selected_terms = tuning.target_terms.clone().unwrap_or_default();
    let filtered_stats = filter_bonded_stats(bonded_stats, &selected_terms);
    let filtered_targets = reference_targets.map(|targets| targets.filter_terms(&selected_terms));
    let sample_warning = insufficient_sample_warning(
        tuning,
        &filtered_stats,
        filtered_targets.as_ref(),
        reference_frames_read,
    )?;
    let metric_scoring = metric_scoring_config(tuning);
    let mut report = run_selected_fitting_mode(
        tuning,
        &config,
        &filtered_stats,
        filtered_targets.as_ref(),
        reference_metrics,
        &metric_scoring,
        out_dir,
        forcefield,
    )?;
    if report.status == "error" {
        return Err(anyhow!("{}", report.message));
    }
    if tuning.fitting_mode.as_deref() == Some("simulation_fit") {
        report.message = if report.message.is_empty() {
            "Optimized bonded parameters with simulation-backed candidate trajectory scoring."
                .to_string()
        } else {
            format!(
                "{} Simulation-backed scoring required a runner/evaluator candidate_trajectory for every completed evaluation.",
                report.message
            )
        };
    }
    if let Some(warning) = sample_warning {
        if report.message.is_empty() {
            report.message = warning;
        } else {
            report.message = format!("{} {}", report.message, warning);
        }
    }
    let report_path = out_dir.join(format!("{}_tuning_report.json", name));
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    artifacts.push(CgArtifact {
        path: report_path.to_string_lossy().to_string(),
        kind: "bonded_optimization_report".to_string(),
    });

    Ok(ParameterTuningResult {
        status: report.status.clone(),
        method,
        source: tuning.source.clone(),
        objective: tuning.objective.clone(),
        message: report.message.clone(),
        report: Some(report),
    })
}

fn normalized_tuning_method(method: &str) -> String {
    match method {
        "bo" => "bayesian_optimization".to_string(),
        _ => method.to_string(),
    }
}

fn run_selected_fitting_mode(
    tuning: &ParameterTuningRequest,
    config: &OptimizationConfig,
    filtered_stats: &BondedStats,
    filtered_targets: Option<&ReferenceTargetSet>,
    reference_metrics: Option<&BTreeMap<String, f64>>,
    metric_scoring: &StructuralMetricScoringConfig,
    out_dir: &Path,
    forcefield: Option<&ForcefieldRequest>,
) -> Result<OptimizationReport> {
    match tuning.fitting_mode.as_deref().unwrap_or("auto") {
        "direct_statistics" => {
            if let Some(targets) = filtered_targets {
                Ok(direct_statistics_report_from_targets(targets, config))
            } else {
                Ok(direct_statistics_report(filtered_stats, config))
            }
        }
        "distribution_fit" => {
            let Some(targets) = filtered_targets else {
                return Err(anyhow!(
                    "optimization.fitting_mode=distribution_fit requires reference targets from reference_source"
                ));
            };
            Ok(optimize_reference_targets(targets, config))
        }
        "external_evaluator" => {
            let Some(targets) = filtered_targets else {
                return Err(anyhow!(
                    "optimization.fitting_mode=external_evaluator requires reference targets from reference_source"
                ));
            };
            let mut evaluator = if let Some(runner) = tuning.runner.as_ref() {
                runner_json_file_evaluator(
                    runner,
                    targets,
                    reference_metrics,
                    metric_scoring,
                    false,
                    out_dir,
                    forcefield,
                )?
            } else if let Some(evaluator) = tuning.evaluator.as_ref() {
                json_file_evaluator(
                    evaluator,
                    targets,
                    reference_metrics,
                    metric_scoring,
                    false,
                    out_dir,
                )?
            } else {
                return Err(anyhow!(
                    "optimization.fitting_mode=external_evaluator requires optimization.evaluator or optimization.runner"
                ));
            };
            Ok(optimize_reference_targets_with_named_evaluator(
                targets,
                &mut evaluator,
                config,
                &[],
            ))
        }
        "simulation_fit" => {
            let Some(targets) = filtered_targets else {
                return Err(anyhow!(
                    "optimization.fitting_mode=simulation_fit requires reference targets from reference_source"
                ));
            };
            let mut evaluator = if let Some(runner) = tuning.runner.as_ref() {
                runner_json_file_evaluator(
                    runner,
                    targets,
                    reference_metrics,
                    metric_scoring,
                    true,
                    out_dir,
                    forcefield,
                )?
            } else if let Some(evaluator) = tuning.evaluator.as_ref() {
                json_file_evaluator(
                    evaluator,
                    targets,
                    reference_metrics,
                    metric_scoring,
                    true,
                    out_dir,
                )?
            } else {
                return Err(anyhow!(
                    "optimization.fitting_mode=simulation_fit requires optimization.evaluator or optimization.runner"
                ));
            };
            Ok(optimize_reference_targets_with_named_evaluator(
                targets,
                &mut evaluator,
                config,
                &[],
            ))
        }
        _ => {
            if let Some(runner) = tuning.runner.as_ref() {
                let Some(targets) = filtered_targets else {
                    return Err(anyhow!(
                        "optimization.runner requires reference targets from reference_source"
                    ));
                };
                let simulation_fit = runner_has_candidate_extraction(runner);
                let mut evaluator = runner_json_file_evaluator(
                    runner,
                    targets,
                    reference_metrics,
                    metric_scoring,
                    simulation_fit,
                    out_dir,
                    forcefield,
                )?;
                Ok(optimize_reference_targets_with_named_evaluator(
                    targets,
                    &mut evaluator,
                    config,
                    &[],
                ))
            } else if let Some(evaluator) = tuning.evaluator.as_ref() {
                let Some(targets) = filtered_targets else {
                    return Err(anyhow!(
                        "optimization.evaluator requires reference targets from reference_source"
                    ));
                };
                let simulation_fit = evaluator_has_candidate_extraction(evaluator);
                let mut evaluator = json_file_evaluator(
                    evaluator,
                    targets,
                    reference_metrics,
                    metric_scoring,
                    simulation_fit,
                    out_dir,
                )?;
                Ok(optimize_reference_targets_with_named_evaluator(
                    targets,
                    &mut evaluator,
                    config,
                    &[],
                ))
            } else if let Some(targets) = filtered_targets {
                Ok(optimize_reference_targets(targets, config))
            } else {
                Ok(optimize_bonded_terms(filtered_stats, config))
            }
        }
    }
}

fn evaluator_has_candidate_extraction(evaluator: &super::ObjectiveEvaluatorRequest) -> bool {
    evaluator
        .json_file
        .as_ref()
        .and_then(|json_file| json_file.candidate_extraction.as_ref())
        .is_some()
}

fn runner_has_candidate_extraction(runner: &SimulationRunnerRequest) -> bool {
    runner.candidate_extraction.is_some()
}

fn insufficient_sample_warning(
    tuning: &ParameterTuningRequest,
    stats: &BondedStats,
    targets: Option<&ReferenceTargetSet>,
    reference_frames_read: Option<usize>,
) -> Result<Option<String>> {
    if tuning.allow_single_frame.unwrap_or(false) {
        return Ok(None);
    }
    if reference_frames_read.is_some_and(|frames| frames < 2) {
        let message = "warning: bonded fitting reference has fewer than 2 frames; repeated bonded members are not a temporal distribution. Set optimization.allow_single_frame=true for smoke runs or provide a multi-frame reference for production fitting.".to_string();
        if tuning.on_insufficient_samples.as_deref() == Some("error") {
            return Err(anyhow!("{message}"));
        }
        return Ok(Some(message));
    }
    let min_samples = tuning.min_samples_per_term.unwrap_or(2);
    let insufficient_count = if let Some(targets) = targets {
        targets
            .constraints
            .iter()
            .chain(targets.bonds.iter())
            .chain(targets.angles.iter())
            .chain(targets.dihedrals.iter())
            .filter(|target| target.samples < min_samples)
            .count()
    } else {
        stats
            .bonds
            .iter()
            .map(|stat| stat.samples)
            .chain(stats.angles.iter().map(|stat| stat.samples))
            .chain(stats.dihedrals.iter().map(|stat| stat.samples))
            .filter(|samples| *samples < min_samples)
            .count()
    };
    if insufficient_count == 0 {
        return Ok(None);
    }
    let message = format!(
        "warning: {insufficient_count} bonded fitting terms have fewer than {min_samples} samples; set optimization.allow_single_frame=true for smoke runs or provide a multi-frame reference for production fitting."
    );
    if tuning.on_insufficient_samples.as_deref().unwrap_or("warn") == "error" {
        return Err(anyhow!("{message}"));
    }
    Ok(Some(message))
}

fn json_file_evaluator(
    evaluator: &super::ObjectiveEvaluatorRequest,
    reference_targets: &ReferenceTargetSet,
    reference_metrics: Option<&BTreeMap<String, f64>>,
    metric_scoring: &StructuralMetricScoringConfig,
    simulation_fit: bool,
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
        reference_metrics: reference_metrics.cloned().unwrap_or_default(),
        metric_scoring: metric_scoring.clone(),
        force_reference_scoring: simulation_fit,
        require_candidate_trajectory: simulation_fit,
        candidate_extraction: json_file
            .candidate_extraction
            .as_ref()
            .map(|request| candidate_trajectory_extraction_config(request, Some(reference_targets)))
            .transpose()?,
    }))
}

fn runner_json_file_evaluator(
    runner: &SimulationRunnerRequest,
    reference_targets: &ReferenceTargetSet,
    reference_metrics: Option<&BTreeMap<String, f64>>,
    metric_scoring: &StructuralMetricScoringConfig,
    simulation_fit: bool,
    out_dir: &Path,
    forcefield: Option<&ForcefieldRequest>,
) -> Result<JsonFileObjectiveEvaluator> {
    let work_dir = runner
        .work_dir
        .clone()
        .unwrap_or_else(|| "martini_openmm_evaluations".to_string());
    let evaluator_work_dir = resolve_output_path(out_dir, &work_dir);
    std::fs::create_dir_all(&evaluator_work_dir)?;
    let spec_path = evaluator_work_dir.join("martini_openmm_runner_spec.json");
    std::fs::write(
        &spec_path,
        serde_json::to_vec_pretty(&runner_spec_json(
            runner,
            out_dir,
            forcefield,
            simulation_fit,
        )?)?,
    )?;
    Ok(JsonFileObjectiveEvaluator::new(JsonFileEvaluatorConfig {
        work_dir: evaluator_work_dir,
        request_filename: "candidate.json".to_string(),
        result_filename: "result.json".to_string(),
        command: Some(JsonFileEvaluatorCommand {
            program: runner
                .python
                .clone()
                .unwrap_or_else(|| "python".to_string()),
            args: vec![
                "-m".to_string(),
                "warp_md.cg_martini_openmm_evaluator".to_string(),
                "--spec".to_string(),
                spec_path.to_string_lossy().to_string(),
            ],
        }),
        reference_targets: Some(reference_targets.clone()),
        reference_metrics: reference_metrics.cloned().unwrap_or_default(),
        metric_scoring: metric_scoring.clone(),
        force_reference_scoring: simulation_fit,
        require_candidate_trajectory: simulation_fit,
        candidate_extraction: runner
            .candidate_extraction
            .as_ref()
            .map(|request| candidate_trajectory_extraction_config(request, Some(reference_targets)))
            .transpose()?,
    }))
}

fn runner_spec_json(
    runner: &SimulationRunnerRequest,
    out_dir: &Path,
    forcefield: Option<&ForcefieldRequest>,
    simulation_fit: bool,
) -> Result<Value> {
    let mut value = serde_json::to_value(runner)?;
    let Value::Object(map) = &mut value else {
        return Err(anyhow!("failed to serialize optimization.runner"));
    };
    map.insert(
        "schema_version".to_string(),
        json!("warp-cg.martini-openmm-runner.v1"),
    );
    map.insert(
        "base_dir".to_string(),
        json!(std::env::current_dir()?.to_string_lossy().to_string()),
    );
    map.insert(
        "out_dir".to_string(),
        json!(out_dir.to_string_lossy().to_string()),
    );
    map.insert("simulation_fit".to_string(), json!(simulation_fit));
    map.insert(
        "require_candidate_trajectory".to_string(),
        json!(simulation_fit),
    );
    map.insert(
        "require_parameter_replacements".to_string(),
        json!(simulation_fit),
    );
    if let Some(forcefield) = forcefield {
        let materialized = materialize_request_forcefield(forcefield, out_dir)?;
        map.insert(
            "forcefield_directory".to_string(),
            json!(materialized.root.to_string_lossy().to_string()),
        );
        map.insert(
            "forcefield_includes".to_string(),
            json!(materialized.include_paths),
        );
    }
    Ok(value)
}

fn metric_scoring_config(tuning: &ParameterTuningRequest) -> StructuralMetricScoringConfig {
    let mut config = StructuralMetricScoringConfig::default();
    if let Some(request) = &tuning.metric_scoring {
        if let Some(value) = request.rg_weight {
            config.rg_weight = value;
        }
        if let Some(value) = request.sasa_weight {
            config.sasa_weight = value;
        }
        if let Some(value) = request.missing_metric_penalty {
            config.missing_metric_penalty = value;
        }
        config.require_rg = request.require_rg.unwrap_or(false);
        config.require_sasa = request.require_sasa.unwrap_or(false);
    }
    config
}

fn json_file_command(command: &JsonFileEvaluatorCommandRequest) -> JsonFileEvaluatorCommand {
    JsonFileEvaluatorCommand {
        program: command.program.clone(),
        args: command.args.clone(),
    }
}

fn candidate_trajectory_extraction_config(
    request: &CandidateTrajectoryExtractionRequest,
    reference_targets: Option<&ReferenceTargetSet>,
) -> Result<CandidateTrajectoryExtractionConfig> {
    let term_set = request
        .bonded_terms
        .as_ref()
        .map(candidate_bonded_term_set)
        .transpose()?
        .or_else(|| reference_targets.map(reference_targets_to_bonded_terms));
    Ok(CandidateTrajectoryExtractionConfig {
        mapping: BeadMapping {
            bead_names: request.mapping.bead_names.clone(),
            atom_indices: request.mapping.atom_indices.clone(),
        },
        connections: request.connections.iter().map(|[i, j]| (*i, *j)).collect(),
        term_set,
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
            sasa: native_sasa_options(request.sasa.as_ref()),
        },
        transform: None,
        mapped_trajectory_name: request.mapped_trajectory_name.clone(),
    })
}

fn reference_targets_to_bonded_terms(targets: &ReferenceTargetSet) -> BondedTermSet {
    BondedTermSet {
        constraints: targets
            .constraints
            .iter()
            .map(|target| crate::bonded_terms::BondTermGroup {
                label: target.label.clone(),
                members: target
                    .members
                    .iter()
                    .filter_map(|member| Some([*member.first()?, *member.get(1)?]))
                    .collect(),
            })
            .collect(),
        bonds: targets
            .bonds
            .iter()
            .map(|target| crate::bonded_terms::BondTermGroup {
                label: target.label.clone(),
                members: target
                    .members
                    .iter()
                    .filter_map(|member| Some([*member.first()?, *member.get(1)?]))
                    .collect(),
            })
            .collect(),
        angles: targets
            .angles
            .iter()
            .map(|target| crate::bonded_terms::AngleTermGroup {
                label: target.label.clone(),
                members: target
                    .members
                    .iter()
                    .filter_map(|member| Some([*member.first()?, *member.get(1)?, *member.get(2)?]))
                    .collect(),
            })
            .collect(),
        dihedrals: targets
            .dihedrals
            .iter()
            .map(|target| crate::bonded_terms::DihedralTermGroup {
                label: target.label.clone(),
                members: target
                    .members
                    .iter()
                    .filter_map(|member| {
                        Some([
                            *member.first()?,
                            *member.get(1)?,
                            *member.get(2)?,
                            *member.get(3)?,
                        ])
                    })
                    .collect(),
            })
            .collect(),
        virtual_sites: Vec::new(),
    }
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
            sasa: None,
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
        sasa: native_sasa_options(source.and_then(|source| source.sasa.as_ref())),
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
        sasa: NativeSasaOptions::default(),
    }
}

fn native_sasa_options(request: Option<&super::SasaRequest>) -> NativeSasaOptions {
    let mut options = NativeSasaOptions::default();
    if let Some(request) = request {
        if let Some(value) = request.probe_radius_nm {
            options.probe_radius_nm = value;
        }
        if let Some(value) = request.n_sphere_points {
            options.n_sphere_points = value;
        }
        if let Some(value) = &request.radii_nm {
            options.radii_nm = Some(value.clone());
        }
        if let Some(value) = request.fallback_radius_nm {
            options.fallback_radius_nm = value;
        }
    }
    options
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
