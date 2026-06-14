use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::bonded_terms::BondedTermSet;
use crate::reference::{
    ReferenceDistributionTarget, ReferenceTargetSet, ReferenceTransformConfig,
    TargetExtractionRequest, TargetExtractor, TrajectoryTargetExtractor,
};
use crate::trajectory::{BeadMapping, NativeTrajectoryOptions};

use super::objective::reference_score_evaluation;
use super::{
    EvaluationStatus, NamedObjectiveEvaluator, ObjectiveEvaluation, OptimizationCandidate,
};

pub const JSON_OBJECTIVE_REQUEST_SCHEMA: &str = "warp-cg.objective-request.v1";
pub const JSON_OBJECTIVE_RESULT_SCHEMA: &str = "warp-cg.objective-result.v1";

#[derive(Debug, Clone)]
pub struct JsonFileEvaluatorConfig {
    pub work_dir: PathBuf,
    pub request_filename: String,
    pub result_filename: String,
    pub command: Option<JsonFileEvaluatorCommand>,
    pub reference_targets: Option<ReferenceTargetSet>,
    pub candidate_extraction: Option<CandidateTrajectoryExtractionConfig>,
}

impl JsonFileEvaluatorConfig {
    pub fn new(work_dir: impl Into<PathBuf>) -> Self {
        Self {
            work_dir: work_dir.into(),
            request_filename: "candidate.json".to_string(),
            result_filename: "result.json".to_string(),
            command: None,
            reference_targets: None,
            candidate_extraction: None,
        }
    }

    pub fn with_reference_targets(mut self, reference_targets: ReferenceTargetSet) -> Self {
        self.reference_targets = Some(reference_targets);
        self
    }
}

#[derive(Debug, Clone)]
pub struct CandidateTrajectoryExtractionConfig {
    pub mapping: BeadMapping,
    pub connections: Vec<(usize, usize)>,
    pub term_set: Option<BondedTermSet>,
    pub options: NativeTrajectoryOptions,
    pub transform: Option<ReferenceTransformConfig>,
    pub mapped_trajectory_name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct JsonFileEvaluatorCommand {
    pub program: String,
    pub args: Vec<String>,
}

pub struct JsonFileObjectiveEvaluator {
    config: JsonFileEvaluatorConfig,
    next_id: usize,
}

impl JsonFileObjectiveEvaluator {
    pub fn new(config: JsonFileEvaluatorConfig) -> Self {
        Self { config, next_id: 0 }
    }

    fn evaluate_with_files(&mut self, candidate: &OptimizationCandidate) -> ObjectiveEvaluation {
        let id = self.next_id;
        self.next_id += 1;
        let eval_dir = self.config.work_dir.join(format!("evaluation_{id:06}"));
        if let Err(err) = std::fs::create_dir_all(&eval_dir) {
            return failed_extraction(format!(
                "failed to create evaluator directory '{}': {err}",
                eval_dir.display()
            ));
        }

        let request_path = eval_dir.join(&self.config.request_filename);
        let result_path = eval_dir.join(&self.config.result_filename);
        if let Err(err) = write_request(
            id,
            candidate,
            self.config.reference_targets.as_ref(),
            &request_path,
        ) {
            return failed_extraction(err);
        }
        if let Some(command) = &self.config.command {
            if let Err(err) = run_command(command, &eval_dir, &request_path, &result_path) {
                return ObjectiveEvaluation::failed(EvaluationStatus::FailedSimulation {
                    reason: err,
                });
            }
        }
        read_result(
            &result_path,
            self.config.reference_targets.as_ref(),
            self.config.candidate_extraction.as_ref(),
            &eval_dir,
        )
    }
}

impl NamedObjectiveEvaluator for JsonFileObjectiveEvaluator {
    fn evaluate_candidate(&mut self, candidate: &OptimizationCandidate) -> ObjectiveEvaluation {
        self.evaluate_with_files(candidate)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct JsonObjectiveRequest<'a> {
    pub schema_version: &'static str,
    pub id: usize,
    pub candidate: &'a OptimizationCandidate,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_targets: Option<&'a ReferenceTargetSet>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonObjectiveResult {
    #[serde(default)]
    pub schema_version: Option<String>,
    #[serde(default)]
    pub objective: Option<f64>,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub status: Option<JsonObjectiveStatus>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default)]
    pub candidate_targets: Option<ReferenceTargetSet>,
    #[serde(default)]
    pub candidate_trajectory: Option<JsonCandidateTrajectoryResult>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JsonCandidateTrajectoryResult {
    pub path: String,
    #[serde(default)]
    pub mapped_trajectory_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum JsonObjectiveStatus {
    Text(String),
    Object(EvaluationStatus),
}

fn write_request(
    id: usize,
    candidate: &OptimizationCandidate,
    reference_targets: Option<&ReferenceTargetSet>,
    path: &Path,
) -> Result<(), String> {
    let request = JsonObjectiveRequest {
        schema_version: JSON_OBJECTIVE_REQUEST_SCHEMA,
        id,
        candidate,
        reference_targets,
    };
    let bytes = serde_json::to_vec_pretty(&request)
        .map_err(|err| format!("failed to serialize objective request: {err}"))?;
    std::fs::write(path, bytes).map_err(|err| {
        format!(
            "failed to write objective request '{}': {err}",
            path.display()
        )
    })
}

fn run_command(
    command: &JsonFileEvaluatorCommand,
    eval_dir: &Path,
    request_path: &Path,
    result_path: &Path,
) -> Result<(), String> {
    let output = Command::new(&command.program)
        .args(&command.args)
        .current_dir(eval_dir)
        .env("WARP_CG_CANDIDATE_JSON", request_path)
        .env("WARP_CG_RESULT_JSON", result_path)
        .output()
        .map_err(|err| {
            format!(
                "failed to run evaluator command '{}': {err}",
                command.program
            )
        })?;
    if output.status.success() {
        return Ok(());
    }
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        format!("exit status {}", output.status)
    };
    Err(format!(
        "evaluator command '{}' failed: {detail}",
        command.program
    ))
}

fn read_result(
    path: &Path,
    reference_targets: Option<&ReferenceTargetSet>,
    candidate_extraction: Option<&CandidateTrajectoryExtractionConfig>,
    eval_dir: &Path,
) -> ObjectiveEvaluation {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) => {
            return failed_extraction(format!(
                "failed to read objective result '{}': {err}",
                path.display()
            ));
        }
    };
    let result = match serde_json::from_slice::<JsonObjectiveResult>(&bytes) {
        Ok(result) => result,
        Err(err) => {
            return failed_extraction(format!(
                "failed to parse objective result '{}': {err}",
                path.display()
            ));
        }
    };
    if let Some(schema) = result.schema_version.as_deref() {
        if schema != JSON_OBJECTIVE_RESULT_SCHEMA {
            return failed_extraction(format!(
                "objective result '{}' has schema_version '{schema}', expected '{JSON_OBJECTIVE_RESULT_SCHEMA}'",
                path.display()
            ));
        }
    }
    let status = result
        .status
        .clone()
        .map(|status| status.into_status(result.reason.clone()))
        .unwrap_or(EvaluationStatus::Completed);
    if !status.is_completed() {
        return ObjectiveEvaluation {
            objective: result.objective.filter(|value| value.is_finite()),
            metrics: finite_metrics(result.metrics),
            status,
        };
    }
    if let Some(objective) = result.objective.filter(|value| value.is_finite()) {
        let mut evaluation = ObjectiveEvaluation::completed(objective);
        evaluation
            .metrics
            .extend(finite_metrics(result.metrics.clone()));
        if let Some(reference) = reference_targets {
            let candidate =
                match candidate_targets_for_result(&result, path, candidate_extraction, eval_dir) {
                    Ok(Some(candidate)) => candidate,
                    Ok(None) => {
                        evaluation
                            .metrics
                            .insert("objective".to_string(), objective);
                        return evaluation;
                    }
                    Err(reason) => {
                        return failed_extraction(format!(
                            "objective result '{}' returned incompatible candidate data: {reason}",
                            path.display()
                        ));
                    }
                };
            if let Err(reason) = validate_candidate_targets(reference, &candidate.targets) {
                return failed_extraction(format!(
                    "objective result '{}' returned incompatible candidate_targets: {reason}",
                    path.display()
                ));
            }
            merge_candidate_extraction_metrics(&mut evaluation, &candidate.metrics);
            let score = reference.compare_swarm_cg(&candidate.targets);
            merge_reference_score_metrics(&mut evaluation, reference_score_evaluation(score));
        }
        evaluation
            .metrics
            .insert("objective".to_string(), objective);
        return evaluation;
    }
    let Some(reference_targets) = reference_targets else {
        return failed_extraction(format!(
            "objective result '{}' returned candidate data but evaluator has no reference targets",
            path.display()
        ));
    };
    let candidate_targets = match candidate_targets_for_result(
        &result,
        path,
        candidate_extraction,
        eval_dir,
    ) {
        Ok(Some(candidate_targets)) => candidate_targets,
        Ok(None) => {
            return failed_extraction(format!(
                    "objective result '{}' completed without a finite objective, candidate_targets, or candidate_trajectory",
                    path.display()
                ));
        }
        Err(reason) => {
            return failed_extraction(format!(
                "objective result '{}' returned incompatible candidate data: {reason}",
                path.display()
            ));
        }
    };
    if let Err(reason) = validate_candidate_targets(reference_targets, &candidate_targets.targets) {
        return failed_extraction(format!(
            "objective result '{}' returned incompatible candidate_targets: {reason}",
            path.display()
        ));
    }
    let score = reference_targets.compare_swarm_cg(&candidate_targets.targets);
    let mut evaluation = reference_score_evaluation(score);
    evaluation
        .metrics
        .extend(finite_metrics(result.metrics.clone()));
    merge_candidate_extraction_metrics(&mut evaluation, &candidate_targets.metrics);
    evaluation
}

struct CandidateTargetsForScore {
    targets: ReferenceTargetSet,
    metrics: BTreeMap<String, f64>,
}

fn candidate_targets_for_result(
    result: &JsonObjectiveResult,
    result_path: &Path,
    candidate_extraction: Option<&CandidateTrajectoryExtractionConfig>,
    eval_dir: &Path,
) -> Result<Option<CandidateTargetsForScore>, String> {
    if let Some(candidate_targets) = result.candidate_targets.as_ref() {
        return Ok(Some(CandidateTargetsForScore {
            targets: candidate_targets.clone(),
            metrics: BTreeMap::new(),
        }));
    }
    let Some(candidate_trajectory) = result.candidate_trajectory.as_ref() else {
        return Ok(None);
    };
    let Some(extraction) = candidate_extraction else {
        return Err(
            "candidate_trajectory requires evaluator candidate_extraction config".to_string(),
        );
    };
    let trajectory_path = resolve_result_path(eval_dir, &candidate_trajectory.path);
    let mapped_name = candidate_trajectory
        .mapped_trajectory_name
        .as_deref()
        .or(extraction.mapped_trajectory_name.as_deref());
    let mut extractor = TrajectoryTargetExtractor::new(trajectory_path, extraction.options.clone());
    let target_extraction = extractor
        .extract_targets(&TargetExtractionRequest {
            name: "candidate",
            out_dir: eval_dir,
            mapped_trajectory_name: mapped_name,
            mapping: &extraction.mapping,
            connections: &extraction.connections,
            term_set: extraction.term_set.as_ref(),
            transform: extraction.transform.as_ref(),
        })
        .map_err(|err| {
            format!(
                "failed to extract candidate_trajectory from result '{}': {err}",
                result_path.display()
            )
        })?;
    Ok(Some(CandidateTargetsForScore {
        targets: target_extraction.target_set,
        metrics: target_extraction.metrics,
    }))
}

fn resolve_result_path(eval_dir: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        eval_dir.join(candidate)
    }
}

fn validate_candidate_targets(
    reference: &ReferenceTargetSet,
    candidate: &ReferenceTargetSet,
) -> Result<(), String> {
    if reference.version != candidate.version {
        return Err(format!(
            "version mismatch: reference {}, candidate {}",
            reference.version, candidate.version
        ));
    }
    validate_target_group(
        "constraints",
        &reference.constraints,
        &candidate.constraints,
    )?;
    validate_target_group("bonds", &reference.bonds, &candidate.bonds)?;
    validate_target_group("angles", &reference.angles, &candidate.angles)?;
    validate_target_group("dihedrals", &reference.dihedrals, &candidate.dihedrals)?;
    Ok(())
}

fn validate_target_group(
    category: &str,
    reference: &[ReferenceDistributionTarget],
    candidate: &[ReferenceDistributionTarget],
) -> Result<(), String> {
    if reference.len() != candidate.len() {
        return Err(format!(
            "candidate_targets.{category} length mismatch: reference {}, candidate {}",
            reference.len(),
            candidate.len()
        ));
    }
    for (idx, target) in reference.iter().enumerate() {
        let Some(candidate_target) = candidate
            .iter()
            .find(|item| item.kind == target.kind && item.members == target.members)
        else {
            return Err(format!(
                "candidate_targets.{category}[{idx}] missing term kind {:?} members {:?}",
                target.kind, target.members
            ));
        };
        validate_target_shape(category, idx, target, candidate_target)?;
    }
    Ok(())
}

fn validate_target_shape(
    category: &str,
    idx: usize,
    reference: &ReferenceDistributionTarget,
    candidate: &ReferenceDistributionTarget,
) -> Result<(), String> {
    let field = format!("candidate_targets.{category}[{idx}]");
    if reference.beads != candidate.beads {
        return Err(format!(
            "{field}.beads mismatch: reference {:?}, candidate {:?}",
            reference.beads, candidate.beads
        ));
    }
    if reference.units != candidate.units {
        return Err(format!(
            "{field}.units mismatch: reference '{}', candidate '{}'",
            reference.units, candidate.units
        ));
    }
    if reference.periodic != candidate.periodic {
        return Err(format!(
            "{field}.periodic mismatch: reference {}, candidate {}",
            reference.periodic, candidate.periodic
        ));
    }
    if reference.bin_edges.len() != candidate.bin_edges.len() {
        return Err(format!(
            "{field}.bin_edges length mismatch: reference {}, candidate {}",
            reference.bin_edges.len(),
            candidate.bin_edges.len()
        ));
    }
    if reference.probabilities.len() != candidate.probabilities.len() {
        return Err(format!(
            "{field}.probabilities length mismatch: reference {}, candidate {}",
            reference.probabilities.len(),
            candidate.probabilities.len()
        ));
    }
    if candidate.probabilities.is_empty() {
        return Err(format!("{field}.probabilities must not be empty"));
    }
    for (edge_idx, (reference_edge, candidate_edge)) in reference
        .bin_edges
        .iter()
        .zip(candidate.bin_edges.iter())
        .enumerate()
    {
        if !candidate_edge.is_finite() || (reference_edge - candidate_edge).abs() > 1.0e-12 {
            return Err(format!(
                "{field}.bin_edges[{edge_idx}] mismatch: reference {reference_edge}, candidate {candidate_edge}"
            ));
        }
    }
    for (prob_idx, probability) in candidate.probabilities.iter().enumerate() {
        if !probability.is_finite() || *probability < 0.0 {
            return Err(format!(
                "{field}.probabilities[{prob_idx}] must be finite and non-negative"
            ));
        }
    }
    if !candidate.mean.is_finite()
        || !candidate.std.is_finite()
        || !candidate.domain.iter().all(|value| value.is_finite())
    {
        return Err(format!(
            "{field} mean, std, and domain values must be finite"
        ));
    }
    Ok(())
}

fn merge_reference_score_metrics(
    evaluation: &mut ObjectiveEvaluation,
    reference_evaluation: ObjectiveEvaluation,
) {
    for (key, value) in reference_evaluation.metrics {
        evaluation
            .metrics
            .insert(format!("reference_target.{key}"), value);
    }
    if let Some(value) = reference_evaluation.objective {
        evaluation
            .metrics
            .insert("reference_target.objective".to_string(), value);
    }
}

fn merge_candidate_extraction_metrics(
    evaluation: &mut ObjectiveEvaluation,
    metrics: &BTreeMap<String, f64>,
) {
    for (key, value) in metrics {
        if value.is_finite() {
            evaluation
                .metrics
                .insert(format!("candidate_trajectory.{key}"), *value);
        }
    }
}

impl JsonObjectiveStatus {
    fn into_status(self, reason: Option<String>) -> EvaluationStatus {
        match self {
            Self::Object(status) => status,
            Self::Text(status) => text_status(&status, reason),
        }
    }
}

fn text_status(status: &str, reason: Option<String>) -> EvaluationStatus {
    let reason = reason.unwrap_or_else(|| status.to_string());
    match status {
        "completed" => EvaluationStatus::Completed,
        "failed_simulation" => EvaluationStatus::FailedSimulation { reason },
        "failed_extraction" => EvaluationStatus::FailedExtraction { reason },
        "timed_out" => EvaluationStatus::TimedOut,
        "invalid_parameters" => EvaluationStatus::InvalidParameters { reason },
        other => EvaluationStatus::FailedExtraction {
            reason: format!("unknown objective result status '{other}'"),
        },
    }
}

fn finite_metrics(metrics: BTreeMap<String, f64>) -> BTreeMap<String, f64> {
    metrics
        .into_iter()
        .filter(|(_, value)| value.is_finite())
        .collect()
}

fn failed_extraction(reason: String) -> ObjectiveEvaluation {
    ObjectiveEvaluation::failed(EvaluationStatus::FailedExtraction { reason })
}
