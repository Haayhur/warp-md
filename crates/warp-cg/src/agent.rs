use crate::optimize::OptimizationReport;
use anyhow::{anyhow, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::time::Instant;
#[path = "agent_artifacts.rs"]
mod agent_artifacts;
#[path = "agent_bo.rs"]
mod agent_bo;
#[path = "agent_contract.rs"]
mod agent_contract;
#[path = "agent_defaults.rs"]
mod agent_defaults;
#[path = "agent_execution.rs"]
mod agent_execution;
#[path = "agent_pso.rs"]
mod agent_pso;
#[path = "agent_reference_only.rs"]
mod agent_reference_only;
#[path = "agent_reference_result.rs"]
mod agent_reference_result;
#[path = "agent_render.rs"]
mod agent_render;
#[path = "agent_runtime.rs"]
mod agent_runtime;
#[path = "agent_source_bonded_terms.rs"]
mod agent_source_bonded_terms;
#[path = "agent_source_mapping.rs"]
mod agent_source_mapping;
#[path = "agent_source_ndx.rs"]
mod agent_source_ndx;
#[path = "agent_source_template.rs"]
mod agent_source_template;
#[path = "agent_source_types.rs"]
mod agent_source_types;
#[path = "agent_source_validation.rs"]
mod agent_source_validation;
#[path = "agent_validation.rs"]
mod agent_validation;
#[cfg(test)]
use crate::mapping::map_molecule;
#[cfg(test)]
use crate::mapping::MappingResult;
#[cfg(test)]
use crate::molecule::Molecule;
#[cfg(test)]
use crate::parameters::{AngleStats, DihedralStats};
pub use agent_contract::{capabilities, example_request, example_requests, schema_json};
use agent_defaults::*;
use agent_execution::run_request;
#[cfg(test)]
use agent_render::{render_martini_itp, render_martini_top};
use agent_runtime::validate_positive;
use agent_source_types::{
    SourceBeadClassContext, SourceBeadRecord, SourceHandoff, SourceMappingResult, SourceResidue,
};
use agent_source_validation::validation_report;
use agent_validation::validate_request;
#[cfg(test)]
use std::path::Path;
pub const AGENT_SCHEMA_VERSION: &str = "warp-cg.agent.v1";

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CgRequest {
    #[serde(default = "default_schema_version")]
    #[schemars(default = "default_schema_version")]
    pub schema_version: String,
    pub name: String,
    #[serde(default)]
    pub smiles: Option<String>,
    #[serde(default)]
    pub repeat_smiles: Option<String>,
    #[serde(default)]
    pub source: Option<CgSource>,
    #[serde(default)]
    pub bonding: Option<BondingPolicyRequest>,
    #[serde(default)]
    pub chemistry_hints: Vec<ChemistryHintRequest>,
    #[serde(default)]
    pub chemistry_policy: Option<ChemistryPolicyRequest>,
    #[serde(default)]
    pub polymer: Option<PolymerPolicyRequest>,
    #[serde(default)]
    pub mapping: Option<CgMappingRequest>,
    #[serde(default)]
    pub topology: Option<String>,
    #[serde(default)]
    pub trajectory_source: Option<TrajectorySource>,
    #[serde(default)]
    pub reference_source: Option<ReferenceSource>,
    #[serde(default)]
    pub forcefield: Option<ForcefieldRequest>,
    #[serde(default)]
    pub optimization: Option<ParameterTuningRequest>,
    #[serde(default = "default_output")]
    #[schemars(default = "default_output")]
    pub output: CgOutputRequest,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CgSource {
    pub kind: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub coordinates: Option<String>,
    #[serde(default)]
    pub topology: Option<String>,
    #[serde(default)]
    pub charge_manifest: Option<String>,
    #[serde(default)]
    pub trajectory: Option<String>,
    #[serde(default)]
    pub target_selection: Option<String>,
    #[serde(default)]
    pub selection: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub topology_format: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BondingPolicyRequest {
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub infer_bonds: Option<bool>,
    #[serde(default)]
    pub on_ambiguous: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChemistryHintRequest {
    pub kind: String,
    pub scope: String,
    #[serde(default)]
    pub value: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ChemistryPolicyRequest {
    #[serde(default)]
    pub hint_mode: Option<String>,
    #[serde(default)]
    pub on_conflict: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PolymerPolicyRequest {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub role_mode: Option<String>,
    #[serde(default)]
    pub terminal_aware: Option<bool>,
    #[serde(default)]
    pub end_group_policy: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CgMappingRequest {
    #[serde(default = "default_mapping_mode")]
    #[schemars(default = "default_mapping_mode")]
    pub mode: String,
    #[serde(default)]
    pub strategy: Option<String>,
    #[serde(default)]
    pub target_bead_size: Option<usize>,
    #[serde(default)]
    pub preserve_functional_groups: Option<bool>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub template_policy: Option<String>,
    #[serde(default)]
    pub expected_beads_per_role: BTreeMap<String, usize>,
    #[serde(default)]
    pub on_bead_count_mismatch: Option<String>,
    #[serde(default)]
    pub ndx: Option<String>,
    #[serde(default)]
    pub repeat_unit_hint: Option<String>,
    #[serde(default)]
    pub terminal_aware: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TrajectorySource {
    /// External trajectory path accepted by warp-md loaders, e.g. xtc, dcd, trr, pdb, xyz.
    pub path: String,
    #[serde(default)]
    pub topology: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub topology_format: Option<String>,
    #[serde(default = "default_external_trajectory_kind")]
    #[schemars(default = "default_external_trajectory_kind")]
    pub kind: String,
    #[serde(default)]
    pub stride: Option<usize>,
    #[serde(default)]
    pub start: Option<usize>,
    #[serde(default)]
    pub stop: Option<usize>,
    #[serde(default)]
    pub length_scale: Option<f32>,
    #[serde(default)]
    pub target_selection: Option<String>,
    #[serde(default)]
    pub environment_selection: Option<String>,
    #[serde(default)]
    pub atom_indices: Option<Vec<usize>>,
    #[serde(default)]
    pub mass_weighted: Option<bool>,
    #[serde(default)]
    pub make_whole: Option<bool>,
    #[serde(default)]
    pub sasa: Option<SasaRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReferenceSource {
    #[serde(default = "default_reference_kind")]
    #[schemars(default = "default_reference_kind")]
    pub kind: String,
    #[serde(default)]
    pub xtb: Option<XtbRequest>,
    #[serde(default)]
    pub precomputed: Option<PrecomputedReferenceRequest>,
    #[serde(default)]
    pub bonded_terms: Option<BondedTermSource>,
    #[serde(default)]
    pub metrics: Vec<ReferenceMetricSourceRequest>,
    #[serde(default)]
    pub transform: Option<ReferenceTransformRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PrecomputedReferenceRequest {
    #[serde(default)]
    pub source_kind: Option<String>,
    pub target_set: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ForcefieldRequest {
    #[serde(default = "default_forcefield_kind")]
    #[schemars(default = "default_forcefield_kind")]
    pub kind: String,
    #[serde(default = "default_forcefield_source")]
    #[schemars(default = "default_forcefield_source")]
    pub source: String,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub dest: Option<String>,
    #[serde(default)]
    pub include_files: Vec<String>,
    #[serde(default = "default_forcefield_materialize")]
    #[schemars(default = "default_forcefield_materialize")]
    pub materialize: String,
    #[serde(default)]
    pub overwrite: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BondedTermSource {
    #[serde(default = "default_bonded_term_source_kind")]
    #[schemars(default = "default_bonded_term_source_kind")]
    pub kind: String,
    pub path: String,
    pub molecule_type: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReferenceMetricSourceRequest {
    #[serde(default = "default_reference_metric_kind")]
    #[schemars(default = "default_reference_metric_kind")]
    pub kind: String,
    pub path: String,
    #[serde(default)]
    pub namespace: Option<String>,
    #[serde(default)]
    pub artifact_kind: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReferenceTransformRequest {
    #[serde(default)]
    pub bond_scaling: Option<f64>,
    #[serde(default)]
    pub min_bond_length_nm: Option<f64>,
    #[serde(default)]
    pub specific_bond_lengths_nm: BTreeMap<String, f64>,
    #[serde(default)]
    pub rg_offset_nm: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SasaRequest {
    #[serde(default)]
    pub probe_radius_nm: Option<f64>,
    #[serde(default)]
    pub n_sphere_points: Option<usize>,
    #[serde(default)]
    pub radii_nm: Option<Vec<f64>>,
    #[serde(default)]
    pub fallback_radius_nm: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct XtbRequest {
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub temperature_k: Option<f64>,
    #[serde(default)]
    pub time_ps: Option<f64>,
    #[serde(default)]
    pub timestep_fs: Option<f64>,
    #[serde(default)]
    pub dump_fs: Option<f64>,
    #[serde(default)]
    pub gfn: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub work_dir: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ParameterTuningRequest {
    #[serde(default = "default_tuning_enabled")]
    #[schemars(default = "default_tuning_enabled")]
    pub enabled: bool,
    #[serde(default = "default_tuning_source")]
    #[schemars(default = "default_tuning_source")]
    pub source: String,
    #[serde(default = "default_tuning_method")]
    #[schemars(default = "default_tuning_method")]
    pub method: String,
    #[serde(default)]
    pub fitting_mode: Option<String>,
    #[serde(default)]
    pub allow_single_frame: Option<bool>,
    #[serde(default)]
    pub min_samples_per_term: Option<usize>,
    #[serde(default)]
    pub on_insufficient_samples: Option<String>,
    #[serde(default)]
    pub max_evaluations: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub initial_parameters: BTreeMap<String, f64>,
    #[serde(default)]
    pub swarm_size: Option<usize>,
    #[serde(default)]
    pub pso: Option<PsoTuningRequest>,
    #[serde(default)]
    pub bo: Option<BoTuningRequest>,
    #[serde(default = "default_tuning_objective")]
    #[schemars(default = "default_tuning_objective")]
    pub objective: String,
    #[serde(default)]
    pub target_terms: Option<Vec<String>>,
    #[serde(default)]
    pub xtb: Option<XtbRequest>,
    #[serde(default)]
    pub metric_scoring: Option<MetricScoringRequest>,
    #[serde(default)]
    pub evaluator: Option<ObjectiveEvaluatorRequest>,
    #[serde(default)]
    pub runner: Option<SimulationRunnerRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MetricScoringRequest {
    #[serde(default)]
    pub rg_weight: Option<f64>,
    #[serde(default)]
    pub sasa_weight: Option<f64>,
    #[serde(default)]
    pub missing_metric_penalty: Option<f64>,
    #[serde(default)]
    pub require_rg: Option<bool>,
    #[serde(default)]
    pub require_sasa: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ObjectiveEvaluatorRequest {
    #[serde(default = "default_objective_evaluator_kind")]
    #[schemars(default = "default_objective_evaluator_kind")]
    pub kind: String,
    #[serde(default)]
    pub json_file: Option<JsonFileEvaluatorRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonFileEvaluatorRequest {
    pub work_dir: String,
    #[serde(default)]
    pub request_filename: Option<String>,
    #[serde(default)]
    pub result_filename: Option<String>,
    #[serde(default)]
    pub command: Option<JsonFileEvaluatorCommandRequest>,
    #[serde(default)]
    pub candidate_extraction: Option<CandidateTrajectoryExtractionRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JsonFileEvaluatorCommandRequest {
    pub program: String,
    #[serde(default)]
    pub args: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SimulationRunnerRequest {
    #[serde(default = "default_simulation_runner_kind")]
    #[schemars(default = "default_simulation_runner_kind")]
    pub kind: String,
    #[serde(default)]
    pub work_dir: Option<String>,
    #[serde(default)]
    pub python: Option<String>,
    pub gro: String,
    pub top: String,
    #[serde(default)]
    pub template_dir: Option<String>,
    #[serde(default)]
    pub replacements: Vec<RunnerTemplateReplacementRequest>,
    #[serde(default)]
    pub protocol: Option<MartiniOpenMmProtocolRequest>,
    #[serde(default)]
    pub candidate_extraction: Option<CandidateTrajectoryExtractionRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RunnerTemplateReplacementRequest {
    pub path: String,
    pub parameter: String,
    #[serde(default)]
    pub placeholder: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MartiniOpenMmProtocolRequest {
    #[serde(default)]
    pub prefix: Option<String>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub pressure: Option<f64>,
    #[serde(default)]
    pub friction: Option<f64>,
    #[serde(default)]
    pub eq_timestep_fs: Option<f64>,
    #[serde(default)]
    pub prod_timestep_fs: Option<f64>,
    #[serde(default)]
    pub cutoff_nm: Option<f64>,
    #[serde(default)]
    pub eq_ns: Option<f64>,
    #[serde(default)]
    pub prod_ns: Option<f64>,
    #[serde(default)]
    pub production_ensemble: Option<String>,
    #[serde(default)]
    pub platform: Option<String>,
    #[serde(default)]
    pub precision: Option<String>,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub cpu_threads: Option<usize>,
    #[serde(default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub minimize_iterations: Option<usize>,
    #[serde(default)]
    pub barostat_frequency: Option<usize>,
    #[serde(default)]
    pub report_interval_steps: Option<usize>,
    #[serde(default)]
    pub trajectory_interval_steps: Option<usize>,
    #[serde(default)]
    pub checkpoint_interval_steps: Option<usize>,
    #[serde(default)]
    pub energy_interval_steps: Option<usize>,
    #[serde(default)]
    pub status_interval_steps: Option<usize>,
    #[serde(default)]
    pub trajectory_format: Option<String>,
    #[serde(default)]
    pub energy_log: Option<bool>,
    #[serde(default)]
    pub epsilon_r: Option<f64>,
    #[serde(default)]
    pub defines_file: Option<String>,
    #[serde(default)]
    pub defines: Vec<String>,
    #[serde(default)]
    pub dry_run: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CandidateTrajectoryExtractionRequest {
    pub mapping: CandidateTrajectoryMappingRequest,
    #[serde(default)]
    pub connections: Vec<[usize; 2]>,
    #[serde(default)]
    pub bonded_terms: Option<BondedTermSource>,
    #[serde(default)]
    pub mapped_trajectory_name: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub topology: Option<String>,
    #[serde(default)]
    pub topology_format: Option<String>,
    #[serde(default)]
    pub start: Option<usize>,
    #[serde(default)]
    pub stop: Option<usize>,
    #[serde(default)]
    pub stride: Option<usize>,
    #[serde(default)]
    pub length_scale: Option<f32>,
    #[serde(default)]
    pub target_selection: Option<String>,
    #[serde(default)]
    pub atom_indices: Option<Vec<usize>>,
    #[serde(default)]
    pub mass_weighted: Option<bool>,
    #[serde(default)]
    pub make_whole: Option<bool>,
    #[serde(default)]
    pub chunk_frames: Option<usize>,
    #[serde(default)]
    pub sasa: Option<SasaRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CandidateTrajectoryMappingRequest {
    pub bead_names: Vec<String>,
    pub atom_indices: Vec<Vec<usize>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PsoTuningRequest {
    #[serde(default)]
    pub fuzzy_self_tuning: Option<bool>,
    #[serde(default)]
    pub fuzzy_adapt_inertia: Option<bool>,
    #[serde(default)]
    pub fuzzy_adapt_cognitive: Option<bool>,
    #[serde(default)]
    pub fuzzy_adapt_social: Option<bool>,
    #[serde(default)]
    pub fuzzy_adapt_min_velocity: Option<bool>,
    #[serde(default)]
    pub fuzzy_adapt_max_velocity: Option<bool>,
    #[serde(default)]
    pub reboot_stalled_particles: Option<bool>,
    #[serde(default)]
    pub reboot_after_local_stall_iterations: Option<usize>,
    #[serde(default)]
    pub restart_strategy: Option<String>,
    #[serde(default)]
    pub linear_population_decrease: Option<bool>,
    #[serde(default)]
    pub max_iterations_without_global_best: Option<usize>,
    #[serde(default)]
    pub checkpoint_path: Option<String>,
    #[serde(default)]
    pub checkpoint_interval_evaluations: Option<usize>,
    #[serde(default)]
    pub resume_from_checkpoint: Option<bool>,
    #[serde(default)]
    pub discrete_probability_dilation: Option<bool>,
    #[serde(default)]
    pub discrete_probability_dilation_alpha: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoTuningRequest {
    #[serde(default)]
    pub algorithm: Option<String>,
    #[serde(default)]
    pub acquisition: Option<String>,
    #[serde(default)]
    pub n_startup_trials: Option<usize>,
    #[serde(default)]
    pub n_candidates: Option<usize>,
    #[serde(default)]
    pub noise_variance: Option<f64>,
    #[serde(default)]
    pub training_set_policy: Option<BoTrainingSetPolicyRequest>,
    #[serde(default)]
    pub failure_handling: Option<String>,
    #[serde(default)]
    pub failure_penalty: Option<f64>,
    #[serde(default)]
    pub checkpoint_path: Option<String>,
    #[serde(default)]
    pub checkpoint_interval_evaluations: Option<usize>,
    #[serde(default)]
    pub resume_from_checkpoint: Option<bool>,
    #[serde(default)]
    pub evaluator_signature: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BoTrainingSetPolicyRequest {
    pub max_points: usize,
    pub keep_best: usize,
    pub keep_recent: usize,
    pub keep_diverse: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CgOutputRequest {
    #[serde(default = "default_out_dir")]
    #[schemars(default = "default_out_dir")]
    pub out_dir: String,
    #[serde(default)]
    pub mapped_trajectory: Option<String>,
    #[serde(default = "default_write_mapping")]
    #[schemars(default = "default_write_mapping")]
    pub write_mapping_json: bool,
    #[serde(default = "default_write_topology_itp")]
    #[schemars(default = "default_write_topology_itp")]
    pub write_topology_itp: bool,
    #[serde(default = "default_write_topology_top")]
    #[schemars(default = "default_write_topology_top")]
    pub write_topology_top: bool,
    #[serde(default = "default_write_cg_pdb")]
    #[schemars(default = "default_write_cg_pdb")]
    pub write_cg_pdb: bool,
    #[serde(default)]
    pub cg_pdb: Option<String>,
    #[serde(default = "default_write_bonded_parameter_map")]
    #[schemars(default = "default_write_bonded_parameter_map")]
    pub write_bonded_parameter_map: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CgBead {
    pub index: usize,
    pub name: String,
    pub atom_indices: Vec<usize>,
    pub features: Vec<String>,
    pub formal_charge: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CgArtifact {
    pub path: String,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CgResult {
    pub schema_version: String,
    pub status: String,
    pub exit_code: i32,
    pub name: String,
    pub summary: CgSummary,
    pub bead_count: usize,
    pub beads: Vec<CgBead>,
    pub connections: Vec<[usize; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapping_summary: Option<Value>,
    pub artifacts: Vec<CgArtifact>,
    pub artifact_paths: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<CgReferenceResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimization: Option<ParameterTuningResult>,
    pub elapsed_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CgReferenceResult {
    pub source_kind: String,
    pub target_set_available: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mapped_trajectory: Option<String>,
    pub metrics: BTreeMap<String, f64>,
    pub metadata: CgReferenceMetadata,
    pub artifacts: Vec<CgArtifact>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CgReferenceMetadata {
    pub frames_read: usize,
    pub frames_written: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,
    pub mapped_by: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CgSummary {
    pub input_mode: String,
    pub mapping_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aa_atom_count: Option<usize>,
    pub cg_bead_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mapped_residue_count: Option<usize>,
    pub optimized_terms: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimization_source: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ParameterTuningResult {
    pub status: String,
    pub method: String,
    pub source: String,
    pub objective: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report: Option<OptimizationReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CgEvent {
    pub schema_version: String,
    pub event: String,
    pub message: String,
    #[serde(default)]
    pub name: Option<String>,
}

pub fn validate_request_json(text: &str) -> (i32, Value) {
    match parse_request(text).and_then(validate_request) {
        Ok(request) => {
            let validation = validation_report(&request);
            let valid = validation["errors"]
                .as_array()
                .map_or(true, |errors| errors.is_empty());
            let code = if valid { 0 } else { 2 };
            (
                code,
                json!({
                    "schema_version": AGENT_SCHEMA_VERSION,
                    "valid": valid,
                    "status": if valid { "ok" } else { "error" },
                    "name": request.name,
                    "summary": validation["summary"].clone(),
                    "checks": validation["checks"].clone(),
                    "warnings": validation["warnings"].clone(),
                    "errors": validation["errors"].clone(),
                }),
            )
        }
        Err(err) => (
            2,
            json!({
                "schema_version": AGENT_SCHEMA_VERSION,
                "valid": false,
                "status": "error",
                "error": {
                    "code": "warp_cg.invalid_request",
                    "message": err.to_string()
                }
            }),
        ),
    }
}

pub fn run_request_json(text: &str, stream_ndjson: bool) -> (i32, Value) {
    let started = Instant::now();
    let request = match parse_request(text).and_then(validate_request) {
        Ok(request) => request,
        Err(err) => {
            return (
                2,
                json!({
                    "schema_version": AGENT_SCHEMA_VERSION,
                    "status": "error",
                    "exit_code": 2,
                    "error": {
                        "code": "warp_cg.invalid_request",
                        "message": err.to_string()
                    }
                }),
            );
        }
    };

    emit_event(
        stream_ndjson,
        "operation_started",
        "mapping molecule",
        Some(&request.name),
    );
    match run_request(&request, started) {
        Ok(result) => {
            emit_event(
                stream_ndjson,
                "operation_complete",
                "mapping complete",
                Some(&request.name),
            );
            (
                0,
                serde_json::to_value(result).expect("serialize warp-cg result"),
            )
        }
        Err(err) => {
            emit_event(
                stream_ndjson,
                "error",
                &err.to_string(),
                Some(&request.name),
            );
            (
                1,
                json!({
                    "schema_version": AGENT_SCHEMA_VERSION,
                    "status": "error",
                    "exit_code": 1,
                    "name": request.name,
                    "error": {
                        "code": "warp_cg.run_failed",
                        "message": err.to_string()
                    },
                    "elapsed_ms": started.elapsed().as_millis()
                }),
            )
        }
    }
}

fn parse_request(text: &str) -> Result<CgRequest> {
    serde_json::from_str(text).map_err(|err| anyhow!("invalid JSON request: {err}"))
}

pub(super) fn input_mode(request: &CgRequest) -> &str {
    if let Some(source) = &request.source {
        source.kind.as_str()
    } else if request.trajectory_source.is_some()
        && request
            .mapping
            .as_ref()
            .is_some_and(|mapping| mapping.mode == "ndx")
    {
        "trajectory_ndx_reference"
    } else if request.repeat_smiles.is_some() {
        "repeat_smiles"
    } else if request.smiles.is_some() {
        "smiles"
    } else {
        "unknown"
    }
}

pub(super) fn mapping_mode(request: &CgRequest) -> &str {
    request
        .mapping
        .as_ref()
        .map(|mapping| mapping.mode.as_str())
        .unwrap_or("small_molecule_auto")
}

pub(super) fn artifact_paths(artifacts: &[CgArtifact]) -> BTreeMap<String, String> {
    artifacts
        .iter()
        .map(|artifact| (artifact.kind.clone(), artifact.path.clone()))
        .collect()
}

pub(super) fn active_tuning_request(request: &CgRequest) -> Option<&ParameterTuningRequest> {
    request.optimization.as_ref()
}

fn emit_event(enabled: bool, event: &str, message: &str, name: Option<&str>) {
    if !enabled {
        return;
    }
    let payload = CgEvent {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        event: event.to_string(),
        message: message.to_string(),
        name: name.map(str::to_string),
    };
    eprintln!(
        "{}",
        serde_json::to_string(&payload).expect("serialize warp-cg event")
    );
}

#[cfg(test)]
#[path = "agent_tests.rs"]
mod tests;
