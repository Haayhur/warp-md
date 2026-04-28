use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use traj_core::{
    elements::{mass_for_element, normalize_element},
    SpatialHash,
};
use warp_common::resolve_relative_path;
use warp_pack::agent::shared_contract::{self, to_error, to_warning};
use warp_pack::{PackError, PackResult};
use warp_structure::io::{
    read_molecule, read_prmtop_topology, write_amber_inpcrd, write_minimal_prmtop, write_output,
    AmberTopology,
};
use warp_structure::{center_of_geometry, AtomRecord, OutputSpec, PackOutput, Vec3};

use crate::minimize::minimize_synthetic_topology;
use crate::polymer::{
    assess_training_source, build_polymer_graph, build_polymer_synthetic_uff_topology,
    compute_sequence_polymer_net_charge_from_prmtop,
    compute_sequence_polymer_net_charge_from_source, ensure_build_qc_passes, load_charge_manifest,
    recompute_build_qc_report, write_polymer_prmtop_from_ffxml, write_polymer_prmtop_from_source,
    write_polymer_prmtop_synthetic_uff_like, BuildQcReport, GraphEdgeSpec, GraphNodeSpec,
    TokenJunctionSpec, TrainingSourceAssessment, CHARGE_MANIFEST_VERSION,
};
use warp_topology_graph::{
    AlignmentPath as TopologyGraphAlignmentPath, Angle as TopologyGraphAngle,
    AppliedCap as TopologyGraphAppliedCap, Atom as TopologyGraphAtom,
    BranchPoint as TopologyGraphBranchPoint, BuildPlan as TopologyGraphBuildPlan, CapBinding,
    ConformerEdge as TopologyGraphConformerEdge,
    ConnectionDefinition as TopologyGraphConnectionDefinition,
    CycleRecord as TopologyGraphCycleRecord, Exclusion as TopologyGraphExclusion,
    InterResidueBond as TopologyGraphInterResidueBond, MotifInstance as TopologyGraphMotifInstance,
    MotifPortBinding as TopologyGraphMotifPortBinding,
    NonbondedTyping as TopologyGraphNonbondedTyping, OpenPort as TopologyGraphOpenPort,
    Pair as TopologyGraphPair, PortPolicy as TopologyGraphPortPolicy,
    RelaxMetadata as TopologyGraphRelaxMetadata, Residue as TopologyGraphResidue,
    ResiduePort as TopologyGraphResiduePort, TerminiRequest as TopologyGraphTerminiRequest,
    TopologyGraph, Torsion as TopologyGraphTorsion, TOPOLOGY_GRAPH_VERSION,
};

pub const SOURCE_BUNDLE_SCHEMA_VERSION: &str = "polymer-param-source.bundle.v1";
pub const BUILD_SCHEMA_VERSION: &str = "warp-build.agent.v1";
pub const BUILD_MANIFEST_VERSION: &str = "warp-build.manifest.v1";
const ENSEMBLE_MANIFEST_VERSION: &str = "warp-build.ensemble-manifest.v1";

const SUPPORTED_TARGET_MODES: &[&str] = &[
    "linear_homopolymer",
    "linear_sequence_polymer",
    "block_copolymer",
    "random_copolymer",
    "star_polymer",
    "branched_polymer",
    "polymer_graph",
];
const OVERLAP_REPORT_METRIC: &str = "vdw_overlap_pairs_excluding_1_2_and_1_3";
const DECLARED_PLAN_MODES: &[&str] = &[
    "linear_homopolymer",
    "linear_sequence_polymer",
    "block_copolymer",
    "random_copolymer",
    "star_polymer",
    "branched_polymer",
    "polymer_graph",
];
const EXECUTABLE_PLAN_MODES: &[&str] = &[
    "linear_homopolymer",
    "linear_sequence_polymer",
    "block_copolymer",
    "random_copolymer",
    "star_polymer",
    "branched_polymer",
    "polymer_graph",
];
const SUPPORTED_CONFORMATION_MODES: &[&str] = &["extended", "random_walk", "aligned", "ensemble"];
const SUPPORTED_TACTICITY_MODES: &[&str] = &[
    "inherit",
    "training",
    "isotactic",
    "syndiotactic",
    "atactic",
];
const SUPPORTED_TERMINI_POLICIES: &[&str] = &["default", "source_default"];
const EXAMPLE_BUNDLE_ID: &str = "example_polymer_bundle_v1";
const EXAMPLE_BUNDLE_PATH: &str = "source.bundle.json";
const EXAMPLE_SOURCE_COORDINATES: &str = "training.pdb";
const EXAMPLE_SOURCE_TOPOLOGY: &str = "training.prmtop";
const EXAMPLE_SOURCE_CHARGE_MANIFEST: &str = "training_charge.json";
const EXAMPLE_FORCEFIELD_REF: &str = "training.ffxml";

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct TrainingContext {
    pub mode: String,
    #[serde(default)]
    pub training_oligomer_n: usize,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct UnitEntry {
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub junctions: BTreeMap<String, String>,
    #[serde(default)]
    pub template_resname: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct MotifNodeEntry {
    pub id: String,
    pub token: String,
    #[serde(default)]
    pub applied_resname: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct MotifEdgeEntry {
    pub from: String,
    pub to: String,
    pub from_junction: String,
    pub to_junction: String,
    #[serde(default = "default_bond_order")]
    #[schemars(default = "default_bond_order")]
    pub bond_order: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ExposedPortSpec {
    pub node_id: String,
    pub junction: String,
    #[serde(default)]
    pub port_class: Option<String>,
    #[serde(default)]
    pub default_cap: Option<CapBinding>,
    #[serde(default)]
    pub allowed_caps: Vec<CapBinding>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct MotifEntry {
    #[serde(default)]
    pub display_name: Option<String>,
    #[serde(default)]
    pub root_node_id: Option<String>,
    #[serde(default)]
    pub nodes: Vec<MotifNodeEntry>,
    #[serde(default)]
    pub edges: Vec<MotifEdgeEntry>,
    #[serde(default)]
    pub exposed_ports: BTreeMap<String, ExposedPortSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct JunctionSelector {
    pub scope: String,
    pub selector: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct JunctionEntry {
    pub attach_atom: JunctionSelector,
    #[serde(default)]
    pub leaving_atoms: Vec<JunctionSelector>,
    pub bond_order: u8,
    #[serde(default)]
    pub anchor_atoms: Vec<JunctionSelector>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SequenceTokenSupport {
    #[serde(default)]
    pub tokens: Vec<String>,
    #[serde(default)]
    pub allowed_adjacencies: Vec<[String; 2]>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SourceCapabilities {
    #[serde(default)]
    pub supported_target_modes: Vec<String>,
    #[serde(default)]
    pub supported_conformation_modes: Vec<String>,
    #[serde(default)]
    pub supported_tacticity_modes: Vec<String>,
    #[serde(default)]
    pub supported_termini_policies: Vec<String>,
    #[serde(default)]
    pub sequence_token_support: Option<SequenceTokenSupport>,
    #[serde(default)]
    pub charge_transfer_supported: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SourceArtifacts {
    pub source_coordinates: String,
    #[serde(default)]
    pub source_topology_ref: Option<String>,
    #[serde(default)]
    pub forcefield_ref: Option<String>,
    #[serde(default)]
    pub source_charge_manifest: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SourceBundle {
    pub schema_version: String,
    pub bundle_id: String,
    pub training_context: TrainingContext,
    #[serde(default)]
    pub provenance: Value,
    #[serde(default)]
    pub unit_library: BTreeMap<String, UnitEntry>,
    #[serde(default)]
    pub motif_library: BTreeMap<String, MotifEntry>,
    #[serde(default)]
    pub junction_library: BTreeMap<String, JunctionEntry>,
    pub capabilities: SourceCapabilities,
    pub artifacts: SourceArtifacts,
    #[serde(default)]
    pub charge_model: Value,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SourceRef {
    pub bundle_id: String,
    pub bundle_path: String,
    #[serde(default)]
    pub bundle_digest: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct TerminiPolicy {
    #[serde(default = "default_policy")]
    #[schemars(default = "default_policy")]
    pub head: String,
    #[serde(default = "default_policy")]
    #[schemars(default = "default_policy")]
    pub tail: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct StereoSpec {
    #[serde(default = "default_stereo")]
    #[schemars(default = "default_stereo")]
    pub mode: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BlockSpec {
    pub token: String,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BranchNode {
    pub token: String,
    #[serde(default)]
    pub children: Vec<BranchChild>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BranchChild {
    pub parent_junction: String,
    pub child_junction: String,
    #[serde(default)]
    pub sequence: Vec<String>,
    pub repeat_count: usize,
    #[serde(default)]
    pub child: Option<Box<BranchNode>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct GraphTargetNode {
    pub id: String,
    pub token: String,
    #[serde(default)]
    pub applied_resname: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct GraphTargetEdge {
    #[serde(default)]
    pub id: Option<String>,
    pub from: String,
    pub to: String,
    pub from_junction: String,
    pub to_junction: String,
    #[serde(default = "default_bond_order")]
    #[schemars(default = "default_bond_order")]
    pub bond_order: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct EdgeConformerOverride {
    pub edge_id: String,
    #[serde(default = "default_torsion_mode")]
    #[schemars(default = "default_torsion_mode")]
    pub torsion_mode: String,
    #[serde(default)]
    pub torsion_deg: Option<f32>,
    #[serde(default)]
    pub torsion_window_deg: Option<[f32; 2]>,
    #[serde(default)]
    pub ring_mode: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ConformerPolicy {
    #[serde(default = "default_layout_mode")]
    #[schemars(default = "default_layout_mode")]
    pub layout_mode: String,
    #[serde(default = "default_torsion_mode")]
    #[schemars(default = "default_torsion_mode")]
    pub default_torsion: String,
    #[serde(default)]
    pub default_torsion_deg: Option<f32>,
    #[serde(default)]
    pub torsion_window_deg: Option<[f32; 2]>,
    #[serde(default = "default_branch_spread")]
    #[schemars(default = "default_branch_spread")]
    pub branch_spread: String,
    #[serde(default = "default_ring_mode")]
    #[schemars(default = "default_ring_mode")]
    pub ring_mode: String,
    #[serde(default)]
    pub edge_overrides: Vec<EdgeConformerOverride>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PortCapOverride {
    pub node_id: String,
    pub port: String,
    pub cap: CapBinding,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BuildTarget {
    pub mode: String,
    #[serde(default)]
    pub repeat_unit: Option<String>,
    #[serde(default)]
    pub n_repeat: Option<usize>,
    #[serde(default)]
    pub sequence: Option<Vec<String>>,
    #[serde(default)]
    pub repeat_count: Option<usize>,
    #[serde(default)]
    pub blocks: Option<Vec<BlockSpec>>,
    #[serde(default)]
    pub composition: Option<BTreeMap<String, usize>>,
    #[serde(default)]
    pub total_units: Option<usize>,
    #[serde(default)]
    pub arm_count: Option<usize>,
    #[serde(default)]
    pub arm_length: Option<usize>,
    #[serde(default)]
    pub core_token: Option<String>,
    #[serde(default)]
    pub core_junctions: Option<Vec<String>>,
    #[serde(default)]
    pub arm_sequence: Option<Vec<String>>,
    #[serde(default)]
    pub arm_repeat_count: Option<usize>,
    #[serde(default)]
    pub branch_tree: Option<BranchNode>,
    #[serde(default)]
    pub graph_root: Option<String>,
    #[serde(default)]
    pub graph_nodes: Option<Vec<GraphTargetNode>>,
    #[serde(default)]
    pub graph_edges: Option<Vec<GraphTargetEdge>>,
    #[serde(default)]
    pub termini: TerminiPolicy,
    #[serde(default)]
    pub stereochemistry: StereoSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct Realization {
    pub conformation_mode: String,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub alignment_axis: Option<String>,
    #[serde(default)]
    pub ensemble_size: Option<usize>,
    #[serde(default)]
    pub relax: Option<RelaxSpec>,
    #[serde(default = "default_qc_policy")]
    #[schemars(default = "default_qc_policy")]
    pub qc_policy: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct RelaxSpec {
    #[serde(default = "default_relax_mode")]
    #[schemars(default = "default_relax_mode")]
    pub mode: String,
    #[serde(default)]
    pub steps: Option<usize>,
    #[serde(default)]
    pub step_scale: Option<f32>,
    #[serde(default)]
    pub clash_scale: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ValidationSpec {
    #[serde(default = "default_validation_depth")]
    #[schemars(default = "default_validation_depth")]
    pub depth: String,
    #[serde(default = "default_validation_cache_mode")]
    #[schemars(default = "default_validation_cache_mode")]
    pub cache_mode: String,
    #[serde(default)]
    pub cache_dir: Option<String>,
}

impl Default for ValidationSpec {
    fn default() -> Self {
        Self {
            depth: default_validation_depth(),
            cache_mode: default_validation_cache_mode(),
            cache_dir: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ArtifactRequest {
    pub coordinates: String,
    #[serde(default)]
    pub raw_coordinates: Option<String>,
    pub build_manifest: String,
    pub charge_manifest: String,
    #[serde(default)]
    pub inpcrd: Option<String>,
    #[serde(default)]
    pub topology: Option<String>,
    #[serde(default)]
    pub topology_graph: Option<String>,
    #[serde(default)]
    pub ensemble_manifest: Option<String>,
    #[serde(default)]
    pub forcefield_ref: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BuildRequest {
    #[serde(default = "default_build_schema_version")]
    #[schemars(default = "default_build_schema_version")]
    pub schema_version: String,
    pub request_id: String,
    pub source_ref: SourceRef,
    pub target: BuildTarget,
    pub realization: Realization,
    #[serde(default)]
    pub conformer_policy: Option<ConformerPolicy>,
    #[serde(default)]
    pub port_cap_overrides: Vec<PortCapOverride>,
    #[serde(default)]
    pub validation: ValidationSpec,
    pub artifacts: ArtifactRequest,
}

pub type ErrorDetail = shared_contract::ErrorDetail;

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct SuccessEnvelope {
    pub schema_version: String,
    pub status: String,
    pub request_id: String,
    pub artifacts: ArtifactRequest,
    pub summary: RunSummary,
    pub warnings: Vec<ErrorDetail>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct ErrorEnvelope {
    pub schema_version: String,
    pub status: String,
    pub request_id: String,
    pub errors: Vec<ErrorDetail>,
    pub warnings: Vec<ErrorDetail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_diagnostics: Option<FailureDiagnostics>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct BuildManifestSchema {
    pub schema_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    pub request_id: String,
    pub normalized_request: NormalizedBuildRequest,
    pub resolved_inputs: ResolvedInputsSummary,
    pub source_bundle: BuildManifestSourceBundle,
    pub target: BuildTarget,
    pub realization: BuildManifestRealization,
    pub artifacts: BuildManifestArtifacts,
    pub artifact_digests: BuildArtifactDigests,
    pub md_ready_handoff: MdReadyHandoff,
    pub summary: BuildManifestSummary,
    pub provenance: BuildManifestProvenance,
    pub warnings: Vec<ErrorDetail>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct ValidateEnvelope {
    pub schema_version: String,
    pub status: String,
    pub valid: bool,
    pub normalized_request: NormalizedBuildRequest,
    pub resolved_inputs: ResolvedInputsSummary,
    pub preflight_cache: PreflightCacheSummary,
    pub preflight: PreflightSummary,
    pub warnings: Vec<ErrorDetail>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct PreflightCacheSummary {
    pub cache_key: String,
    pub request_digest: String,
    pub input_digest: String,
    pub cache_mode: String,
    pub reusable: bool,
    pub state: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact_paths: Option<ArtifactRequest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, JsonSchema)]
pub struct BuildPhaseTimingsMs {
    compile_plan: u64,
    resolve_junctions: u64,
    prepare_graph_specs: u64,
    build_graph: u64,
    solver_cleanup: u64,
    user_relax: u64,
    artifact_write: u64,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct NormalizedSourceRef {
    pub bundle_id: String,
    pub bundle_path: String,
    pub bundle_digest: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct NormalizedRealization {
    pub conformation_mode: String,
    pub seed: u64,
    pub seed_policy: String,
    pub alignment_axis: Option<String>,
    pub ensemble_size: Option<usize>,
    pub relax: Option<RelaxSpec>,
    pub qc_policy: String,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct ResolvedArtifactSummary {
    pub coordinates: String,
    pub raw_coordinates: Option<String>,
    pub build_manifest: String,
    pub charge_manifest: String,
    pub inpcrd: String,
    pub topology: Option<String>,
    pub topology_graph: String,
    pub ensemble_manifest: Option<String>,
    pub forcefield_ref: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct NormalizedBuildRequest {
    pub schema_version: String,
    pub request_id: String,
    pub source_ref: NormalizedSourceRef,
    pub target: BuildTarget,
    pub realization: NormalizedRealization,
    pub conformer_policy: Option<ConformerPolicy>,
    pub port_cap_overrides: Vec<PortCapOverride>,
    pub artifacts: ResolvedArtifactSummary,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct ValidationDepthSummary {
    pub requested_depth: String,
    pub default_depth: String,
    pub cache_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_dir: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct ResolvedTerminiPolicy {
    pub head: String,
    pub tail: String,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct ResolvedSourceArtifacts {
    pub coordinates: String,
    pub charge_manifest: Option<String>,
    pub topology: Option<String>,
    pub forcefield_ref: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct TopologyTransferRequestedOutputs {
    pub inpcrd: bool,
    pub topology: bool,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct TopologyTransferRequirements {
    pub topology: String,
    pub inpcrd: String,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct TopologyTransferSummary {
    pub supported: bool,
    pub requested_outputs: TopologyTransferRequestedOutputs,
    pub requirements: TopologyTransferRequirements,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct TargetNormalizationSummary {
    pub applied: String,
    pub reason: String,
    pub head_token: String,
    pub tail_token: String,
    pub sequence: Vec<String>,
    pub requested_mode: String,
    pub requested_repeat_unit: Option<String>,
    pub requested_n_repeat: Option<usize>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct ResolvedInputsSummary {
    pub source_bundle_id: String,
    pub source_bundle_path: String,
    pub source_bundle_digest: Option<String>,
    pub target_mode: String,
    pub realization_mode: String,
    pub validation: ValidationDepthSummary,
    pub resolved_termini_policy: ResolvedTerminiPolicy,
    pub resolved_seed: u64,
    pub seed_policy: String,
    pub target_normalization: Option<TargetNormalizationSummary>,
    pub resolved_artifacts: ResolvedArtifactSummary,
    pub resolved_source_artifacts: ResolvedSourceArtifacts,
    pub topology_transfer: TopologyTransferSummary,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct AppliedJunctionSummary {
    pub head_attach_atom: Option<String>,
    pub head_leaving_atoms: Vec<String>,
    pub tail_attach_atom: Option<String>,
    pub tail_leaving_atoms: Vec<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct FailureDiagnostics {
    pub phase: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qc: Option<crate::polymer::BuildQcReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applied_junctions: Option<BTreeMap<String, AppliedJunctionSummary>>,
    pub overlap_status: OverlapStatusSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub debug_coordinates: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_coordinates: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct PreflightSummary {
    pub executed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_conformation_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_residue_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applied_junctions: Option<BTreeMap<String, AppliedJunctionSummary>>,
    pub timings_ms: BuildPhaseTimingsMs,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub qc: Option<crate::polymer::BuildQcReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solver: Option<crate::polymer::BuildSolverReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solver_cleanup: Option<RelaxReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relax: Option<RelaxReport>,
    pub overlap_status: OverlapStatusSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter_source_decision: Option<crate::polymer::TrainingSourceAssessment>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct RunSummary {
    pub build_mode: String,
    pub n_repeat: usize,
    pub atom_count: usize,
    pub total_repeat_units: usize,
    pub total_residues: usize,
    pub conformation_mode: String,
    pub seed: u64,
    pub ensemble_size: usize,
    pub topology_graph_version: String,
    pub qc: crate::polymer::BuildQcReport,
    pub solver: Option<crate::polymer::BuildSolverReport>,
    pub timings_ms: BuildPhaseTimingsMs,
    pub solver_cleanup: Option<RelaxReport>,
    pub relax: Option<RelaxReport>,
    pub overlap_status: OverlapStatusSummary,
    pub acceptance_state: String,
    pub handoff_level: String,
    pub limitations: Vec<String>,
    pub parameter_source_decision: crate::polymer::TrainingSourceAssessment,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct BuildManifestSourceBundle {
    pub bundle_id: String,
    pub bundle_path: String,
    pub bundle_digest: Option<String>,
    pub training_context: TrainingContext,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct BuildManifestRealization {
    pub conformation_mode: String,
    pub seed: u64,
    pub seed_policy: String,
    pub relax: Option<RelaxSpec>,
    pub qc_policy: String,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct BuildManifestArtifacts {
    pub coordinates: String,
    pub raw_coordinates: Option<String>,
    pub charge_manifest: String,
    pub inpcrd: String,
    pub topology: Option<String>,
    pub topology_graph: String,
    pub ensemble_manifest: Option<String>,
    pub forcefield_ref: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct BuildArtifactDigests {
    pub coordinates: Option<String>,
    pub raw_coordinates: Option<String>,
    pub charge_manifest: Option<String>,
    pub inpcrd: Option<String>,
    pub topology: Option<String>,
    pub topology_graph: Option<String>,
    pub ensemble_manifest: Option<String>,
    pub forcefield_ref: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct HandoffCoordinates {
    pub path: String,
    pub format: String,
    pub strict_columns: bool,
    pub write_conect: bool,
    pub has_cryst1: bool,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct HandoffChainInstanceMapping {
    pub copy_index: usize,
    pub source_chain_indices: Vec<usize>,
    pub packed_chain_indices: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct MdReadyHandoff {
    pub version: String,
    pub coordinates: HandoffCoordinates,
    pub topology_graph: String,
    pub topology: Option<String>,
    pub charge_manifest: String,
    pub acceptance_state: String,
    pub handoff_level: String,
    pub limitations: Vec<String>,
    pub source_structure_path: String,
    pub source_topology_path: Option<String>,
    pub source_charge_manifest_path: Option<String>,
    pub sequence_tokens: Vec<String>,
    pub template_sequence_resnames: Vec<String>,
    pub applied_residue_resnames: Vec<String>,
    pub copy_count: usize,
    pub chain_instance_mapping: Vec<HandoffChainInstanceMapping>,
    pub forcefield_ref: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct AppliedTerminusSummary {
    pub requested_policy: String,
    pub resolved_token: Option<String>,
    pub template_resname: Option<String>,
    pub applied_resname: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct AppliedTerminiSummary {
    pub head: AppliedTerminusSummary,
    pub tail: AppliedTerminusSummary,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct AppliedCapSummary {
    pub node_id: String,
    pub request_node_id: String,
    pub port_name: String,
    pub cap: CapBinding,
    pub application_source: String,
    pub cap_node_id: String,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct BuildManifestSummary {
    pub atom_count: usize,
    pub total_repeat_units: usize,
    pub total_residues: usize,
    pub net_charge_e: Option<f32>,
    pub resolved_sequence: Vec<String>,
    pub template_sequence_resnames: Vec<String>,
    pub applied_residue_resnames: Vec<String>,
    pub request_root_node_id: String,
    pub expanded_root_node_id: String,
    pub graph_has_cycle: bool,
    pub applied_termini: AppliedTerminiSummary,
    pub applied_caps: Vec<AppliedCapSummary>,
    pub bond_count: usize,
    pub realization_mode: String,
    pub ensemble_size: usize,
    pub relax: Option<RelaxReport>,
    pub solver_cleanup: Option<RelaxReport>,
    pub solver: Option<crate::polymer::BuildSolverReport>,
    pub applied_junctions: BTreeMap<String, AppliedJunctionSummary>,
    pub timings_ms: BuildPhaseTimingsMs,
    pub qc: crate::polymer::BuildQcReport,
    pub overlap_status: OverlapStatusSummary,
    pub acceptance_state: String,
    pub handoff_level: String,
    pub limitations: Vec<String>,
    pub parameter_source_decision: crate::polymer::TrainingSourceAssessment,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct OverlapStatusSummary {
    pub status: String,
    pub may_report_no_overlaps: bool,
    pub metric: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report_source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overlap_pairs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_overlap_angstrom: Option<f32>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct BuildMetadataSummary {
    pub git_commit: Option<String>,
    pub build_timestamp: Option<String>,
    pub target_triple: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct BuildManifestProvenance {
    pub schema_version: String,
    pub builder_version: String,
    pub binary_version: String,
    pub algorithm_version: String,
    pub topology_transfer_mode: String,
    pub source_bundle_path: String,
    pub source_bundle_digest: Option<String>,
    pub target_normalization: Option<TargetNormalizationSummary>,
    pub parameter_source_decision: crate::polymer::TrainingSourceAssessment,
    pub build_metadata: BuildMetadataSummary,
}

#[derive(Clone, Debug)]
struct PreparedBuildExecution {
    compiled_plan: CompiledBuildPlan,
    compiled_sequence: Vec<String>,
    token_junctions: BTreeMap<String, TokenJunctionSpec>,
    base_mode: String,
    graph_node_specs: Vec<GraphNodeSpec>,
    graph_edge_specs: Vec<GraphEdgeSpec>,
    graph_root_idx: usize,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
#[serde(tag = "event")]
pub enum RunEvent {
    #[serde(rename = "run_started")]
    RunStarted { request_id: String },
    #[serde(rename = "source_loaded")]
    SourceLoaded {
        request_id: String,
        training_oligomer_n: usize,
        bundle_id: String,
    },
    #[serde(rename = "phase_started")]
    PhaseStarted { request_id: String, phase: String },
    #[serde(rename = "chain_growth_started")]
    ChainGrowthStarted {
        request_id: String,
        target_repeats: usize,
        conformation_mode: String,
        seed: u64,
    },
    #[serde(rename = "chain_growth_progress")]
    ChainGrowthProgress {
        request_id: String,
        completed_repeats: usize,
        target_repeats: usize,
        progress_pct: f32,
    },
    #[serde(rename = "chain_growth_completed")]
    ChainGrowthCompleted {
        request_id: String,
        target_repeats: usize,
        elapsed_ms: u64,
    },
    #[serde(rename = "manifest_written")]
    ManifestWritten { request_id: String, path: String },
    #[serde(rename = "phase_completed")]
    PhaseCompleted { request_id: String, phase: String },
    #[serde(rename = "run_completed")]
    RunCompleted {
        request_id: String,
        artifacts: ArtifactRequest,
    },
    #[serde(rename = "run_failed")]
    RunFailed {
        request_id: String,
        error: ErrorDetail,
        #[serde(skip_serializing_if = "Option::is_none")]
        diagnostics: Option<FailureDiagnostics>,
    },
}

fn default_build_schema_version() -> String {
    BUILD_SCHEMA_VERSION.to_string()
}

#[derive(Serialize, JsonSchema)]
#[serde(untagged)]
#[allow(dead_code)]
enum ResultEnvelopeSchema {
    Success(SuccessEnvelope),
    Error(ErrorEnvelope),
}

fn default_policy() -> String {
    "default".to_string()
}

fn default_stereo() -> String {
    "inherit".to_string()
}

fn default_validation_depth() -> String {
    "deep".to_string()
}

fn default_validation_cache_mode() -> String {
    "off".to_string()
}

fn default_qc_policy() -> String {
    "strict".to_string()
}

fn default_bond_order() -> u8 {
    1
}

fn default_layout_mode() -> String {
    "auto".to_string()
}

fn default_torsion_mode() -> String {
    "trans".to_string()
}

fn default_branch_spread() -> String {
    "even".to_string()
}

fn default_ring_mode() -> String {
    "auto".to_string()
}

fn default_relax_mode() -> String {
    "graph_spring".to_string()
}

fn canonical_relax_mode(mode: &str) -> Option<&'static str> {
    match mode {
        "graph_spring" => Some("graph_spring"),
        "targeted_steric" | "targeted_stearic" => Some("targeted_steric"),
        _ => None,
    }
}

fn internal_solver_relax_spec() -> RelaxSpec {
    RelaxSpec {
        mode: default_relax_mode(),
        steps: Some(64),
        step_scale: Some(0.25),
        clash_scale: Some(0.9),
    }
}

fn ensure_parent(path: &str) -> PackResult<()> {
    let parent = Path::new(path).parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    Ok(())
}

fn sha256_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("sha256:{:x}", hasher.finalize())
}

fn sha256_json_value(value: &Value) -> String {
    let bytes = serde_json::to_vec(value).unwrap_or_else(|_| value.to_string().into_bytes());
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("sha256:{:x}", hasher.finalize())
}

fn canonicalize_artifact_summary(value: &mut Value) {
    let Some(artifacts) = value.as_object_mut() else {
        return;
    };
    for key in [
        "coordinates",
        "raw_coordinates",
        "build_manifest",
        "charge_manifest",
        "inpcrd",
        "topology",
        "topology_graph",
        "ensemble_manifest",
        "forcefield_ref",
    ] {
        if artifacts.contains_key(key) && !artifacts[key].is_null() {
            artifacts.insert(key.to_string(), Value::String(format!("<artifact:{key}>")));
        }
    }
}

fn preflight_cache_payload(
    normalized_request: &NormalizedBuildRequest,
    resolved_inputs: &ResolvedInputsSummary,
    include_artifact_paths: bool,
) -> Value {
    let mut normalized = serde_json::to_value(normalized_request).unwrap_or_else(|_| json!({}));
    let mut resolved = serde_json::to_value(resolved_inputs).unwrap_or_else(|_| json!({}));
    if !include_artifact_paths {
        if let Some(artifacts) = normalized.get_mut("artifacts") {
            canonicalize_artifact_summary(artifacts);
        }
        if let Some(artifacts) = resolved.get_mut("resolved_artifacts") {
            canonicalize_artifact_summary(artifacts);
        }
        if let Some(validation) = resolved
            .get_mut("validation")
            .and_then(Value::as_object_mut)
        {
            validation.insert("cache_mode".into(), Value::String("<cache_mode>".into()));
            validation.insert("cache_dir".into(), Value::String("<cache_dir>".into()));
        }
    }
    json!({
        "contract": "warp-build-preflight-cache.v1",
        "schema_version": BUILD_SCHEMA_VERSION,
        "normalized_request": normalized,
        "resolved_inputs": resolved,
    })
}

fn cache_key_from_digest(input_digest: &str) -> String {
    let bare_digest = input_digest.strip_prefix("sha256:").unwrap_or(input_digest);
    format!("warp-build-preflight-v1:{bare_digest}")
}

fn cache_key_slug(cache_key: &str) -> String {
    cache_key
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn preflight_cache_record_path(cache_dir: &str, cache_key: &str) -> String {
    Path::new(cache_dir)
        .join(cache_key_slug(cache_key))
        .join("record.json")
        .to_string_lossy()
        .to_string()
}

fn preflight_cache_artifact_paths(
    cache_dir: &str,
    cache_key: &str,
    resolved_artifacts: &ResolvedArtifactSummary,
) -> ArtifactRequest {
    let root = Path::new(cache_dir)
        .join(cache_key_slug(cache_key))
        .join("artifacts");
    ArtifactRequest {
        coordinates: root.join("coordinates.pdb").to_string_lossy().to_string(),
        raw_coordinates: resolved_artifacts.raw_coordinates.as_ref().map(|_| {
            root.join("raw_coordinates.pdb")
                .to_string_lossy()
                .to_string()
        }),
        build_manifest: root
            .join("build_manifest.json")
            .to_string_lossy()
            .to_string(),
        charge_manifest: root
            .join("charge_manifest.json")
            .to_string_lossy()
            .to_string(),
        inpcrd: Some(
            root.join("coordinates.inpcrd")
                .to_string_lossy()
                .to_string(),
        ),
        topology: resolved_artifacts
            .topology
            .as_ref()
            .map(|_| root.join("topology.prmtop").to_string_lossy().to_string()),
        topology_graph: Some(
            root.join("topology_graph.json")
                .to_string_lossy()
                .to_string(),
        ),
        ensemble_manifest: resolved_artifacts.ensemble_manifest.as_ref().map(|_| {
            root.join("ensemble_manifest.json")
                .to_string_lossy()
                .to_string()
        }),
        forcefield_ref: resolved_artifacts
            .forcefield_ref
            .as_ref()
            .map(|_| root.join("forcefield.ffxml").to_string_lossy().to_string()),
    }
}

fn preflight_cache_summary(
    normalized_request: &NormalizedBuildRequest,
    resolved_inputs: &ResolvedInputsSummary,
    cache_mode: &str,
    cache_dir: Option<&str>,
    reusable: bool,
    state: &str,
    reason: Option<&str>,
) -> PreflightCacheSummary {
    let request_digest = sha256_json_value(&preflight_cache_payload(
        normalized_request,
        resolved_inputs,
        true,
    ));
    let input_digest = sha256_json_value(&preflight_cache_payload(
        normalized_request,
        resolved_inputs,
        false,
    ));
    let cache_key = cache_key_from_digest(&input_digest);
    let record_path = cache_dir.map(|dir| preflight_cache_record_path(dir, &cache_key));
    let artifact_paths = cache_dir.map(|dir| {
        preflight_cache_artifact_paths(dir, &cache_key, &resolved_inputs.resolved_artifacts)
    });
    PreflightCacheSummary {
        cache_key,
        request_digest,
        input_digest,
        cache_mode: cache_mode.to_string(),
        reusable,
        state: state.to_string(),
        record_path,
        artifact_paths,
        reason: reason.map(str::to_string),
    }
}

fn sha256_file(path: &Path) -> PackResult<String> {
    let bytes = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn parse_json<T: for<'de> Deserialize<'de>>(text: &str) -> Result<T, ErrorDetail> {
    let mut deserializer = serde_json::Deserializer::from_str(text);
    serde_path_to_error::deserialize::<_, T>(&mut deserializer).map_err(|err| {
        let path = err.path().to_string();
        to_error(
            "E_CONFIG_VALIDATION",
            if path.is_empty() { None } else { Some(path) },
            err.inner().to_string(),
        )
    })
}

fn load_bundle(path: &Path) -> Result<SourceBundle, ErrorDetail> {
    let text = fs::read_to_string(path)
        .map_err(|err| to_error("E_SOURCE_LOAD", None, format!("bundle read error: {err}")))?;
    let bundle: SourceBundle = parse_json(&text)?;
    if bundle.schema_version != SOURCE_BUNDLE_SCHEMA_VERSION {
        return Err(to_error(
            "E_SOURCE_SCHEMA",
            Some("schema_version".into()),
            format!(
                "unsupported source bundle schema '{}'; expected {}",
                bundle.schema_version, SOURCE_BUNDLE_SCHEMA_VERSION
            ),
        ));
    }
    Ok(bundle)
}

fn source_template_atoms_from_coordinates(
    path: &Path,
) -> Result<BTreeMap<String, BTreeSet<String>>, ErrorDetail> {
    let molecule = read_molecule(path, None, false, true, None).map_err(|err| {
        to_error(
            "E_SOURCE_ARTIFACT",
            Some("artifacts.source_coordinates".into()),
            format!("source coordinates could not be read: {err}"),
        )
    })?;
    let mut templates = BTreeMap::<String, BTreeSet<String>>::new();
    for atom in molecule.atoms {
        let resname = atom.resname.trim();
        if resname.is_empty() {
            continue;
        }
        templates
            .entry(resname.to_string())
            .or_default()
            .insert(atom.name.trim().to_string());
    }
    Ok(templates)
}

fn source_template_atoms_from_prmtop(
    path: &Path,
) -> Result<BTreeMap<String, BTreeSet<String>>, ErrorDetail> {
    let topology = read_prmtop_topology(path).map_err(|err| {
        to_error(
            "E_SOURCE_ARTIFACT",
            Some("artifacts.source_topology_ref".into()),
            format!("source topology could not be read: {err}"),
        )
    })?;
    let mut templates = BTreeMap::<String, BTreeSet<String>>::new();
    for (residue_idx, label) in topology.residue_labels.iter().enumerate() {
        let start = topology
            .residue_pointers
            .get(residue_idx)
            .copied()
            .unwrap_or(1)
            .saturating_sub(1);
        let end = topology
            .residue_pointers
            .get(residue_idx + 1)
            .copied()
            .unwrap_or(topology.atom_names.len() + 1)
            .saturating_sub(1)
            .min(topology.atom_names.len());
        let atoms = templates.entry(label.trim().to_string()).or_default();
        for atom_name in topology.atom_names[start..end].iter() {
            atoms.insert(atom_name.trim().to_string());
        }
    }
    Ok(templates)
}

fn forcefield_ref_placeholder_reason(path: &Path) -> Option<String> {
    let text = fs::read_to_string(path).ok()?;
    let compact = text.split_whitespace().collect::<String>();
    if compact.eq_ignore_ascii_case("<ForceField/>")
        || compact.eq_ignore_ascii_case("<ForceField></ForceField>")
    {
        return Some("forcefield_ref is an empty <ForceField/> placeholder".into());
    }
    let has_atom_types = text.contains("<AtomTypes") && text.contains("<Type ");
    let has_residues = text.contains("<Residues") && text.contains("<Residue ");
    let has_nonbonded = text.contains("<NonbondedForce") && text.contains("<Atom ");
    if !(has_atom_types && has_residues && has_nonbonded) {
        return Some(
            "forcefield_ref does not contain AtomTypes, Residues, and NonbondedForce parameters"
                .into(),
        );
    }
    None
}

fn source_forcefield_transferable(path: Option<&String>) -> bool {
    path.map(|value| {
        let path = Path::new(value);
        path.exists() && forcefield_ref_placeholder_reason(path).is_none()
    })
    .unwrap_or(false)
}

fn validate_selector_atom_against_template(
    errors: &mut Vec<ErrorDetail>,
    template_atoms: &BTreeMap<String, BTreeSet<String>>,
    topology_atoms: Option<&BTreeMap<String, BTreeSet<String>>>,
    template_resname: &str,
    atom_name: &str,
    selector_kind: &str,
    path: String,
) {
    if !template_atoms
        .get(template_resname)
        .map(|atoms| atoms.contains(atom_name))
        .unwrap_or(false)
    {
        errors.push(to_error(
            "E_SOURCE_SCHEMA",
            Some(path.clone()),
            format!(
                "{selector_kind} atom '{}' missing from template '{}' in source coordinates",
                atom_name, template_resname
            ),
        ));
    }
    if let Some(topology_atoms) = topology_atoms {
        if !topology_atoms
            .get(template_resname)
            .map(|atoms| atoms.contains(atom_name))
            .unwrap_or(false)
        {
            errors.push(to_error(
                "E_SOURCE_SCHEMA",
                Some(path),
                format!(
                    "{selector_kind} atom '{}' missing from template '{}' in source topology",
                    atom_name, template_resname
                ),
            ));
        }
    }
}

fn validate_bundle_template_selectors(
    bundle: &SourceBundle,
    template_atoms: &BTreeMap<String, BTreeSet<String>>,
    topology_atoms: Option<&BTreeMap<String, BTreeSet<String>>>,
) -> Vec<ErrorDetail> {
    let mut errors = Vec::new();
    for (token, unit) in &bundle.unit_library {
        let template_resname = unit.template_resname.as_deref().unwrap_or("").trim();
        if template_resname.is_empty() {
            errors.push(to_error(
                "E_SOURCE_SCHEMA",
                Some(format!("unit_library.{token}.template_resname")),
                "sequence-capable unit definitions must declare template_resname",
            ));
            continue;
        }
        if !template_atoms.contains_key(template_resname) {
            errors.push(to_error(
                "E_SOURCE_SCHEMA",
                Some(format!("unit_library.{token}.template_resname")),
                format!(
                    "template_resname '{}' missing from source coordinates",
                    template_resname
                ),
            ));
        }
        if let Some(topology_atoms) = topology_atoms {
            if !topology_atoms.contains_key(template_resname) {
                errors.push(to_error(
                    "E_SOURCE_SCHEMA",
                    Some(format!("unit_library.{token}.template_resname")),
                    format!(
                        "template_resname '{}' missing from source topology",
                        template_resname
                    ),
                ));
            }
        }
        for (port, junction_name) in &unit.junctions {
            let Some(junction) = bundle.junction_library.get(junction_name) else {
                errors.push(to_error(
                    "E_SOURCE_SCHEMA",
                    Some(format!("unit_library.{token}.junctions.{port}")),
                    format!("junction '{}' missing from junction_library", junction_name),
                ));
                continue;
            };
            match selector_atom_name(&junction.attach_atom) {
                Ok(atom_name) => validate_selector_atom_against_template(
                    &mut errors,
                    template_atoms,
                    topology_atoms,
                    template_resname,
                    atom_name.trim(),
                    "attach",
                    format!("junction_library.{junction_name}.attach_atom.selector"),
                ),
                Err(err) => errors.push(to_error(
                    "E_SOURCE_SCHEMA",
                    Some(format!(
                        "junction_library.{junction_name}.attach_atom.selector"
                    )),
                    err.to_string(),
                )),
            }
            for (idx, selector) in junction.leaving_atoms.iter().enumerate() {
                match selector_atom_name(selector) {
                    Ok(atom_name) => validate_selector_atom_against_template(
                        &mut errors,
                        template_atoms,
                        topology_atoms,
                        template_resname,
                        atom_name.trim(),
                        "leaving",
                        format!("junction_library.{junction_name}.leaving_atoms[{idx}].selector"),
                    ),
                    Err(err) => errors.push(to_error(
                        "E_SOURCE_SCHEMA",
                        Some(format!(
                            "junction_library.{junction_name}.leaving_atoms[{idx}].selector"
                        )),
                        err.to_string(),
                    )),
                }
            }
            for (idx, selector) in junction.anchor_atoms.iter().enumerate() {
                match selector_atom_name(selector) {
                    Ok(atom_name) => validate_selector_atom_against_template(
                        &mut errors,
                        template_atoms,
                        topology_atoms,
                        template_resname,
                        atom_name.trim(),
                        "anchor",
                        format!("junction_library.{junction_name}.anchor_atoms[{idx}].selector"),
                    ),
                    Err(err) => errors.push(to_error(
                        "E_SOURCE_SCHEMA",
                        Some(format!(
                            "junction_library.{junction_name}.anchor_atoms[{idx}].selector"
                        )),
                        err.to_string(),
                    )),
                }
            }
        }
    }
    errors
}

fn inspect_source_artifacts(
    bundle_path: &Path,
    bundle: &SourceBundle,
) -> (Vec<ErrorDetail>, Vec<ErrorDetail>) {
    let mut errors = Vec::new();
    let mut warnings = Vec::new();
    let source_coordinates =
        resolve_relative_path(bundle_path, &bundle.artifacts.source_coordinates);
    let coordinate_atoms =
        match source_template_atoms_from_coordinates(Path::new(&source_coordinates)) {
            Ok(value) => value,
            Err(err) => {
                errors.push(err);
                BTreeMap::new()
            }
        };
    let topology_atoms = bundle
        .artifacts
        .source_topology_ref
        .as_ref()
        .map(|path| resolve_relative_path(bundle_path, path))
        .map(|path| {
            let path_ref = Path::new(&path);
            if !path_ref.exists() {
                Err(to_error(
                    "E_SOURCE_ARTIFACT",
                    Some("artifacts.source_topology_ref".into()),
                    format!("source topology '{}' does not exist", path),
                ))
            } else {
                source_template_atoms_from_prmtop(path_ref)
            }
        })
        .transpose();
    let topology_atoms = match topology_atoms {
        Ok(value) => value,
        Err(err) => {
            errors.push(err);
            None
        }
    };
    if let Some(path) = bundle.artifacts.source_charge_manifest.as_ref() {
        let resolved = resolve_relative_path(bundle_path, path);
        if !Path::new(&resolved).exists() {
            errors.push(to_error(
                "E_SOURCE_ARTIFACT",
                Some("artifacts.source_charge_manifest".into()),
                format!("source charge manifest '{}' does not exist", resolved),
            ));
        } else if let Err(err) = load_charge_manifest(Path::new(&resolved)) {
            errors.push(to_error(
                "E_SOURCE_ARTIFACT",
                Some("artifacts.source_charge_manifest".into()),
                format!("source charge manifest could not be read: {err}"),
            ));
        }
    }
    if let Some(path) = bundle.artifacts.forcefield_ref.as_ref() {
        let resolved = resolve_relative_path(bundle_path, path);
        let path_ref = Path::new(&resolved);
        if !path_ref.exists() {
            errors.push(to_error(
                "E_SOURCE_ARTIFACT",
                Some("artifacts.forcefield_ref".into()),
                format!("forcefield_ref '{}' does not exist", resolved),
            ));
        } else if let Some(reason) = forcefield_ref_placeholder_reason(path_ref) {
            warnings.push(to_warning(
                "W_PLACEHOLDER_FORCEFIELD",
                Some("artifacts.forcefield_ref".into()),
                format!("{reason}; it will not be copied as an MD-ready force-field artifact"),
            ));
        }
    }
    if !coordinate_atoms.is_empty() {
        errors.extend(validate_bundle_template_selectors(
            bundle,
            &coordinate_atoms,
            topology_atoms.as_ref(),
        ));
    }
    (errors, warnings)
}

fn default_inpcrd_path(coordinates_path: &str) -> String {
    let path = Path::new(coordinates_path);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("polymer");
    parent
        .join(format!("{stem}.inpcrd"))
        .to_string_lossy()
        .to_string()
}

fn default_raw_coordinates_path(coordinates_path: &str) -> String {
    let path = Path::new(coordinates_path);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("polymer");
    parent
        .join(format!("{stem}.raw.pdb"))
        .to_string_lossy()
        .to_string()
}

fn default_topology_path(coordinates_path: &str) -> String {
    let path = Path::new(coordinates_path);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("polymer");
    parent
        .join(format!("{stem}.prmtop"))
        .to_string_lossy()
        .to_string()
}

fn default_topology_graph_path(coordinates_path: &str) -> String {
    let path = Path::new(coordinates_path);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("polymer");
    parent
        .join(format!("{stem}.topology.json"))
        .to_string_lossy()
        .to_string()
}

fn default_forcefield_ref_path(coordinates_path: &str) -> String {
    let path = Path::new(coordinates_path);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("polymer");
    parent
        .join(format!("{stem}.ffxml"))
        .to_string_lossy()
        .to_string()
}

fn default_ensemble_manifest_path(coordinates_path: &str) -> String {
    let path = Path::new(coordinates_path);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("polymer");
    parent
        .join(format!("{stem}.ensemble.json"))
        .to_string_lossy()
        .to_string()
}

#[derive(Clone, Debug)]
struct ResolvedArtifacts {
    coordinates: String,
    raw_coordinates: Option<String>,
    build_manifest: String,
    charge_manifest: String,
    inpcrd: String,
    topology: Option<String>,
    topology_graph: String,
    ensemble_manifest: Option<String>,
    forcefield_ref: Option<String>,
}

#[derive(Clone, Debug)]
struct ResolvedBuildRequest {
    bundle_digest: Option<String>,
    seed: u64,
    seed_policy: String,
    warnings: Vec<ErrorDetail>,
    normalization: Option<TargetNormalizationSummary>,
    source_coordinates: String,
    source_charge_manifest: Option<String>,
    source_topology_ref: Option<String>,
    source_forcefield_ref: Option<String>,
    artifacts: ResolvedArtifacts,
    normalized_request: NormalizedBuildRequest,
    resolved_inputs: ResolvedInputsSummary,
}

#[derive(Clone, Debug)]
struct NormalizeRequestResult {
    request: BuildRequest,
    warnings: Vec<ErrorDetail>,
    normalization: Option<TargetNormalizationSummary>,
}

fn deterministic_seed(req: &BuildRequest) -> u64 {
    let mut value = serde_json::to_value(req).unwrap_or_else(|_| json!({}));
    if let Some(object) = value.as_object_mut() {
        object.remove("artifacts");
        object.remove("validation");
    }
    let text = serde_json::to_string(&value).unwrap_or_default();
    let digest = sha256_text(&text);
    u64::from_str_radix(
        digest
            .trim_start_matches("sha256:")
            .get(..16)
            .unwrap_or("1"),
        16,
    )
    .unwrap_or(1_234_567)
}

fn requires_explicit_seed(mode: &str) -> bool {
    matches!(mode, "random_walk" | "ensemble")
}

fn resolve_seed(req: &BuildRequest) -> Result<(u64, String), ErrorDetail> {
    match req.realization.seed {
        Some(seed) => Ok((seed, "explicit".to_string())),
        None if requires_explicit_seed(&req.realization.conformation_mode) => Err(to_error(
            "E_MISSING_SEED",
            Some("/realization/seed".into()),
            format!(
                "{} realization requires explicit seed",
                req.realization.conformation_mode
            ),
        )),
        None => Ok((deterministic_seed(req), "deterministic_default".to_string())),
    }
}

fn collect_warnings(req: &BuildRequest) -> Vec<ErrorDetail> {
    let mut warnings = Vec::new();
    for (path, policy) in [
        ("target.termini.head", req.target.termini.head.as_str()),
        ("target.termini.tail", req.target.termini.tail.as_str()),
    ] {
        if policy == "default" {
            warnings.push(to_warning(
                "W_DEFAULT_TERMINI_RESOLVED",
                Some(path.to_string()),
                "default resolved to source_default",
            ));
        }
    }
    warnings
}

fn infer_terminal_sequence_for_homopolymer(
    req: &BuildRequest,
    bundle: &SourceBundle,
) -> Result<Option<(Vec<String>, String, String)>, ErrorDetail> {
    if req.target.mode != "linear_homopolymer" {
        return Ok(None);
    }
    if !is_default_termini_policy(&req.target.termini.head)
        || !is_default_termini_policy(&req.target.termini.tail)
    {
        return Ok(None);
    }
    let repeat_token = req
        .target
        .repeat_unit
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_string();
    let n_repeat = req.target.n_repeat.unwrap_or(0);
    if repeat_token.is_empty() || n_repeat == 0 {
        return Ok(None);
    }
    let Some(token_support) = bundle.capabilities.sequence_token_support.as_ref() else {
        return Ok(None);
    };
    if !bundle
        .capabilities
        .supported_target_modes
        .iter()
        .any(|mode| mode == "linear_sequence_polymer")
    {
        return Ok(None);
    }
    let adjacency_pairs = token_support
        .allowed_adjacencies
        .iter()
        .filter(|pair| pair[0] != pair[1])
        .collect::<Vec<_>>();
    if adjacency_pairs.is_empty() {
        return Ok(None);
    }
    let head_candidates = adjacency_pairs
        .iter()
        .filter(|pair| pair[1] == repeat_token)
        .map(|pair| pair[0].clone())
        .filter(|token| {
            !adjacency_pairs
                .iter()
                .any(|pair| pair[0] == repeat_token && pair[1] == *token)
        })
        .filter(|token| token != &repeat_token)
        .collect::<BTreeSet<_>>();
    let tail_candidates = adjacency_pairs
        .iter()
        .filter(|pair| pair[0] == repeat_token)
        .map(|pair| pair[1].clone())
        .filter(|token| {
            !adjacency_pairs
                .iter()
                .any(|pair| pair[1] == repeat_token && pair[0] == *token)
        })
        .filter(|token| token != &repeat_token)
        .collect::<BTreeSet<_>>();
    if head_candidates.is_empty() && tail_candidates.is_empty() {
        return Ok(None);
    }
    if head_candidates.len() != 1 || tail_candidates.len() != 1 {
        let mut suggestion = Vec::new();
        if let Some(head) = head_candidates.iter().next() {
            suggestion.push(head.clone());
        }
        suggestion.extend(std::iter::repeat_n(repeat_token.clone(), n_repeat));
        if let Some(tail) = tail_candidates.iter().next() {
            suggestion.push(tail.clone());
        }
        return Err(to_error(
            "E_TERMINAL_SEQUENCE_REQUIRED",
            Some("/target".into()),
            format!(
                "terminal-aware source bundle requires explicit sequence mode for repeat token '{}'; use linear_sequence_polymer with sequence like {:?}",
                repeat_token,
                suggestion
            ),
        ));
    }
    let head = head_candidates.iter().next().cloned().unwrap_or_default();
    let tail = tail_candidates.iter().next().cloned().unwrap_or_default();
    let mut sequence = Vec::with_capacity(n_repeat + 2);
    sequence.push(head.clone());
    sequence.extend(std::iter::repeat_n(repeat_token, n_repeat));
    sequence.push(tail.clone());
    Ok(Some((sequence, head, tail)))
}

fn normalize_request_for_execution(
    req: &BuildRequest,
    bundle: &SourceBundle,
) -> Result<NormalizeRequestResult, Vec<ErrorDetail>> {
    let mut normalized = req.clone();
    let mut warnings = Vec::new();
    let mut normalization = None;
    match infer_terminal_sequence_for_homopolymer(req, bundle) {
        Ok(Some((sequence, head, tail))) => {
            normalized.target.mode = "linear_sequence_polymer".into();
            normalized.target.sequence = Some(sequence.clone());
            normalized.target.repeat_count = Some(1);
            normalized.target.repeat_unit = None;
            normalized.target.n_repeat = None;
            warnings.push(to_warning(
                "W_TERMINAL_SEQUENCE_AUTOPROMOTED",
                Some("/target".into()),
                format!(
                    "terminal-aware source bundle auto-promoted linear_homopolymer to linear_sequence_polymer with explicit termini sequence [{}, ..., {}]",
                    head, tail
                ),
            ));
            normalization = Some(TargetNormalizationSummary {
                applied: "linear_homopolymer_to_linear_sequence_polymer".into(),
                reason: "terminal_aware_source_bundle".into(),
                head_token: head,
                tail_token: tail,
                sequence,
                requested_mode: req.target.mode.clone(),
                requested_repeat_unit: req.target.repeat_unit.clone(),
                requested_n_repeat: req.target.n_repeat,
            });
        }
        Ok(None) => {}
        Err(error) => return Err(vec![error]),
    }
    Ok(NormalizeRequestResult {
        request: normalized,
        warnings,
        normalization,
    })
}

fn resolve_request_state(
    req: &BuildRequest,
    bundle: &SourceBundle,
    preflight_warnings: Vec<ErrorDetail>,
    normalization: Option<TargetNormalizationSummary>,
) -> Result<ResolvedBuildRequest, Vec<ErrorDetail>> {
    let errors = validate_request(req, bundle);
    if !errors.is_empty() {
        return Err(errors);
    }
    let (seed, seed_policy) = resolve_seed(req).map_err(|err| vec![err])?;
    let source_coordinates = resolve_relative_path(
        Path::new(&req.source_ref.bundle_path),
        &bundle.artifacts.source_coordinates,
    );
    let source_charge_manifest = bundle
        .artifacts
        .source_charge_manifest
        .as_ref()
        .map(|path| resolve_relative_path(Path::new(&req.source_ref.bundle_path), path));
    let source_topology_ref = bundle
        .artifacts
        .source_topology_ref
        .as_ref()
        .map(|path| resolve_relative_path(Path::new(&req.source_ref.bundle_path), path));
    let source_forcefield_ref = bundle
        .artifacts
        .forcefield_ref
        .as_ref()
        .map(|path| resolve_relative_path(Path::new(&req.source_ref.bundle_path), path));
    let forcefield_transferable = source_forcefield_transferable(source_forcefield_ref.as_ref());
    let artifacts = ResolvedArtifacts {
        coordinates: req.artifacts.coordinates.clone(),
        raw_coordinates: req.realization.relax.as_ref().map(|_| {
            req.artifacts
                .raw_coordinates
                .clone()
                .unwrap_or_else(|| default_raw_coordinates_path(&req.artifacts.coordinates))
        }),
        build_manifest: req.artifacts.build_manifest.clone(),
        charge_manifest: req.artifacts.charge_manifest.clone(),
        inpcrd: req
            .artifacts
            .inpcrd
            .clone()
            .unwrap_or_else(|| default_inpcrd_path(&req.artifacts.coordinates)),
        topology: Some(
            req.artifacts
                .topology
                .clone()
                .unwrap_or_else(|| default_topology_path(&req.artifacts.coordinates)),
        ),
        topology_graph: req
            .artifacts
            .topology_graph
            .clone()
            .unwrap_or_else(|| default_topology_graph_path(&req.artifacts.coordinates)),
        ensemble_manifest: if req.realization.conformation_mode == "ensemble" {
            Some(
                req.artifacts
                    .ensemble_manifest
                    .clone()
                    .unwrap_or_else(|| default_ensemble_manifest_path(&req.artifacts.coordinates)),
            )
        } else {
            req.artifacts.ensemble_manifest.clone()
        },
        forcefield_ref: if forcefield_transferable {
            Some(
                req.artifacts
                    .forcefield_ref
                    .clone()
                    .unwrap_or_else(|| default_forcefield_ref_path(&req.artifacts.coordinates)),
            )
        } else {
            None
        },
    };
    let mut warnings = preflight_warnings;
    warnings.extend(collect_warnings(req));
    if let Some(path) = source_forcefield_ref.as_ref() {
        if let Some(reason) = forcefield_ref_placeholder_reason(Path::new(path)) {
            warnings.push(to_warning(
                "W_PLACEHOLDER_FORCEFIELD",
                Some("/source_ref/bundle_path".into()),
                format!("{reason}; forcefield_ref will not be emitted as a handoff artifact"),
            ));
        }
    }
    let bundle_digest = sha256_file(Path::new(&req.source_ref.bundle_path)).ok();
    let normalized_artifacts = ResolvedArtifactSummary {
        coordinates: artifacts.coordinates.clone(),
        raw_coordinates: artifacts.raw_coordinates.clone(),
        build_manifest: artifacts.build_manifest.clone(),
        charge_manifest: artifacts.charge_manifest.clone(),
        inpcrd: artifacts.inpcrd.clone(),
        topology: artifacts.topology.clone(),
        topology_graph: artifacts.topology_graph.clone(),
        ensemble_manifest: artifacts.ensemble_manifest.clone(),
        forcefield_ref: artifacts.forcefield_ref.clone(),
    };
    let normalized_request = NormalizedBuildRequest {
        schema_version: BUILD_SCHEMA_VERSION.into(),
        request_id: req.request_id.clone(),
        source_ref: NormalizedSourceRef {
            bundle_id: req.source_ref.bundle_id.clone(),
            bundle_path: req.source_ref.bundle_path.clone(),
            bundle_digest: req
                .source_ref
                .bundle_digest
                .clone()
                .or_else(|| bundle_digest.clone()),
        },
        target: req.target.clone(),
        realization: NormalizedRealization {
            conformation_mode: req.realization.conformation_mode.clone(),
            seed,
            seed_policy: seed_policy.clone(),
            alignment_axis: req.realization.alignment_axis.clone(),
            ensemble_size: req.realization.ensemble_size,
            relax: req.realization.relax.clone(),
            qc_policy: req.realization.qc_policy.clone(),
        },
        conformer_policy: req.conformer_policy.clone(),
        port_cap_overrides: req.port_cap_overrides.clone(),
        artifacts: normalized_artifacts.clone(),
    };
    let resolved_inputs = ResolvedInputsSummary {
        source_bundle_id: bundle.bundle_id.clone(),
        source_bundle_path: req.source_ref.bundle_path.clone(),
        source_bundle_digest: bundle_digest.clone(),
        target_mode: req.target.mode.clone(),
        realization_mode: req.realization.conformation_mode.clone(),
        validation: ValidationDepthSummary {
            requested_depth: req.validation.depth.clone(),
            default_depth: default_validation_depth(),
            cache_mode: req.validation.cache_mode.clone(),
            cache_dir: req.validation.cache_dir.clone(),
        },
        resolved_termini_policy: ResolvedTerminiPolicy {
            head: if req.target.termini.head == "default" {
                "source_default".into()
            } else {
                req.target.termini.head.clone()
            },
            tail: if req.target.termini.tail == "default" {
                "source_default".into()
            } else {
                req.target.termini.tail.clone()
            },
        },
        resolved_seed: seed,
        seed_policy: seed_policy.clone(),
        target_normalization: normalization.clone(),
        resolved_artifacts: normalized_artifacts,
        resolved_source_artifacts: ResolvedSourceArtifacts {
            coordinates: source_coordinates.clone(),
            charge_manifest: source_charge_manifest.clone(),
            topology: source_topology_ref.clone(),
            forcefield_ref: source_forcefield_ref.clone(),
        },
        topology_transfer: TopologyTransferSummary {
            supported: topology_transfer_supported(bundle),
            requested_outputs: TopologyTransferRequestedOutputs {
                inpcrd: req.artifacts.inpcrd.is_some(),
                topology: req.artifacts.topology.is_some(),
            },
            requirements: TopologyTransferRequirements {
                topology: "artifacts.topology auto-derives a .prmtop output; transferable source .prmtop and validated ffxml fallback take precedence when the training source is unreliable, otherwise warp-build emits a synthetic UFF-like minimizer topology".into(),
                inpcrd: "artifacts.inpcrd is coordinate-only and does not transfer bonded terms by itself".into(),
            },
        },
    };
    Ok(ResolvedBuildRequest {
        bundle_digest,
        seed,
        seed_policy,
        warnings,
        normalization,
        source_coordinates,
        source_charge_manifest,
        source_topology_ref,
        source_forcefield_ref,
        artifacts,
        normalized_request,
        resolved_inputs,
    })
}

fn validation_depth(req: &BuildRequest) -> Result<&str, ErrorDetail> {
    match req.validation.depth.as_str() {
        "shallow" => Ok("shallow"),
        "deep" => Ok("deep"),
        other => Err(to_error(
            "E_INVALID_REQUEST",
            Some("/validation/depth".into()),
            format!(
                "unsupported validation depth '{}'; expected 'shallow' or 'deep'",
                other
            ),
        )),
    }
}

fn validation_cache_mode(req: &BuildRequest) -> Result<&str, ErrorDetail> {
    match req.validation.cache_mode.as_str() {
        "off" => Ok("off"),
        "record" => Ok("record"),
        "prefer" => Ok("prefer"),
        "require" => Ok("require"),
        other => Err(to_error(
            "E_INVALID_REQUEST",
            Some("/validation/cache_mode".into()),
            format!(
                "unsupported validation cache_mode '{}'; expected 'off', 'record', 'prefer', or 'require'",
                other
            ),
        )),
    }
}

fn qc_policy(req: &BuildRequest) -> Result<&str, ErrorDetail> {
    match req.realization.qc_policy.as_str() {
        "strict" => Ok("strict"),
        "salvage" => Ok("salvage"),
        other => Err(to_error(
            "E_INVALID_REQUEST",
            Some("/realization/qc_policy".into()),
            format!(
                "unsupported qc_policy '{}'; expected 'strict' or 'salvage'",
                other
            ),
        )),
    }
}

fn elapsed_ms(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

fn requested_n_repeat(
    normalization: Option<&TargetNormalizationSummary>,
    fallback: usize,
) -> usize {
    normalization
        .and_then(|value| value.requested_n_repeat)
        .unwrap_or(fallback)
}

fn build_metadata_summary() -> BuildMetadataSummary {
    BuildMetadataSummary {
        git_commit: option_env!("VERGEN_GIT_SHA").map(str::to_string),
        build_timestamp: option_env!("VERGEN_BUILD_TIMESTAMP").map(str::to_string),
        target_triple: option_env!("TARGET").map(str::to_string),
    }
}

fn applied_junctions_value(
    token_junctions: &BTreeMap<String, TokenJunctionSpec>,
) -> BTreeMap<String, AppliedJunctionSummary> {
    token_junctions
        .iter()
        .map(|(token, junction)| {
            (
                token.clone(),
                AppliedJunctionSummary {
                    head_attach_atom: junction.head_attach_atom.clone(),
                    head_leaving_atoms: junction.head_leaving_atoms.clone(),
                    tail_attach_atom: junction.tail_attach_atom.clone(),
                    tail_leaving_atoms: junction.tail_leaving_atoms.clone(),
                },
            )
        })
        .collect::<BTreeMap<_, _>>()
}

fn failure_diagnostics(
    phase: &str,
    qc: Option<&crate::polymer::BuildQcReport>,
    token_junctions: Option<&BTreeMap<String, TokenJunctionSpec>>,
    overlap_status: OverlapStatusSummary,
    debug_coordinates: Option<String>,
    raw_coordinates: Option<String>,
) -> FailureDiagnostics {
    FailureDiagnostics {
        phase: phase.into(),
        qc: qc.cloned(),
        applied_junctions: token_junctions.map(applied_junctions_value),
        overlap_status,
        debug_coordinates,
        raw_coordinates,
    }
}

fn acceptance_state(salvaged: bool) -> String {
    if salvaged {
        "salvaged".into()
    } else {
        "accepted".into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TopologyArtifactMode {
    None,
    SourceTransfer,
    ForcefieldRef,
    SyntheticUffLike,
}

fn handoff_level_and_limitations(
    topology_mode: TopologyArtifactMode,
    net_charge: Option<f32>,
    salvaged: bool,
    parameter_source: &TrainingSourceAssessment,
    conformation_mode: &str,
    target_mode: &str,
    qc_report: &BuildQcReport,
) -> (String, Vec<String>) {
    let mut limitations = Vec::new();
    match topology_mode {
        TopologyArtifactMode::None => limitations.push("topology_unavailable".into()),
        TopologyArtifactMode::ForcefieldRef => {}
        TopologyArtifactMode::SyntheticUffLike => limitations.push("synthetic_topology".into()),
        TopologyArtifactMode::SourceTransfer => {}
    }
    match parameter_source.quality.as_str() {
        "risky" => limitations.push("training_source_risky".into()),
        "unreliable" => limitations.push("training_source_unreliable".into()),
        _ => {}
    }
    if net_charge.is_none() {
        limitations.push("charge_unavailable".into());
    }
    if salvaged {
        limitations.push("salvaged_qc_failure".into());
    }
    let synthetic_geometry_clean = qc_report.severe_bond_violations.is_empty()
        && qc_report.severe_nonbonded_clash_count == 0
        && qc_report
            .min_nonbonded_distance_angstrom
            .map(|value| value >= 1.20)
            .unwrap_or(true);
    let synthetic_production_conformation = synthetic_geometry_clean
        && matches!(
            conformation_mode,
            "aligned" | "extended" | "random_walk" | "ensemble"
        )
        && matches!(
            target_mode,
            "linear_homopolymer"
                | "linear_sequence_polymer"
                | "block_copolymer"
                | "random_copolymer"
                | "star_polymer"
                | "branched_polymer"
                | "polymer_graph"
        );
    if matches!(topology_mode, TopologyArtifactMode::SyntheticUffLike)
        && !synthetic_production_conformation
    {
        limitations.push("conformer_not_production_minimized".into());
    }
    let handoff_level = if salvaged {
        "graph_bonded_only"
    } else {
        match topology_mode {
            TopologyArtifactMode::SourceTransfer if net_charge.is_some() => "md_ready",
            TopologyArtifactMode::ForcefieldRef if net_charge.is_some() => "forcefield_backed",
            TopologyArtifactMode::SyntheticUffLike if synthetic_production_conformation => {
                "minimizable_synthetic"
            }
            _ => "graph_bonded_only",
        }
    };
    (handoff_level.into(), limitations)
}

fn salvage_warning(detail: &ErrorDetail) -> ErrorDetail {
    to_warning(
        "W_SALVAGED_BUILD",
        Some("/realization/qc_policy".into()),
        format!(
            "QC failed under salvage mode; non-final outputs were written for recovery/minimization: {}",
            detail.message
        ),
    )
}

fn source_fallback_qc_warning(parameter_source: &str, detail: &ErrorDetail) -> ErrorDetail {
    to_warning(
        "W_SOURCE_FALLBACK_QC",
        Some("/source_ref/bundle_path".into()),
        format!(
            "training source fallback '{}' kept the build moving despite QC failure; downstream minimization is expected: {}",
            parameter_source, detail.message
        ),
    )
}

fn copy_forcefield_artifact(source_path: &str, target_path: &str) -> Result<(), ErrorDetail> {
    ensure_parent(target_path).map_err(|err| {
        to_error(
            "E_OUTPUT_WRITE",
            Some("/artifacts/forcefield_ref".into()),
            err.to_string(),
        )
    })?;
    fs::copy(source_path, target_path).map_err(|err| {
        to_error(
            "E_OUTPUT_WRITE",
            Some("/artifacts/forcefield_ref".into()),
            err.to_string(),
        )
    })?;
    Ok(())
}

fn topology_graph_mass(element: &str) -> f32 {
    let normalized = normalize_element(element).unwrap_or_else(|| element.trim().to_string());
    let mass = mass_for_element(&normalized);
    if mass > 0.0 {
        mass
    } else {
        12.0
    }
}

fn topology_graph_atom_type(element: &str) -> String {
    normalize_element(element).unwrap_or_else(|| element.trim().to_string())
}

fn topology_graph_atom_type_index(element: &str) -> i32 {
    match topology_graph_atom_type(element).as_str() {
        "H" => 1,
        "B" => 5,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        "Si" => 14,
        "P" => 15,
        "S" => 16,
        "Cl" => 17,
        "Br" => 35,
        "I" => 53,
        _ => 0,
    }
}

fn prepare_build_execution(
    req: &BuildRequest,
    bundle: &SourceBundle,
    seed: u64,
) -> Result<(PreparedBuildExecution, BuildPhaseTimingsMs), ErrorDetail> {
    let mut timings = BuildPhaseTimingsMs::default();

    let compile_started = Instant::now();
    let compiled_plan = match compile_build_plan(req, bundle, seed)? {
        Some(plan) => plan,
        None => {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some("/target".into()),
                "target expands to an empty sequence",
            ))
        }
    };
    timings.compile_plan = elapsed_ms(compile_started);

    let compiled_sequence = compiled_plan
        .nodes
        .iter()
        .map(|node| node.token.clone())
        .collect::<Vec<_>>();

    let junction_started = Instant::now();
    let token_junctions = token_junction_specs(bundle, &compiled_sequence)
        .map_err(|err| to_error("E_SOURCE_SCHEMA", None, err.to_string()))?;
    timings.resolve_junctions = elapsed_ms(junction_started);

    let graph_prep_started = Instant::now();
    let base_mode = base_conformation_mode(&req.realization.conformation_mode)?.to_string();
    let graph_node_specs = compiled_plan
        .nodes
        .iter()
        .map(|node| GraphNodeSpec {
            sequence_label: node.token.clone(),
            template_resname: node.template_resname.clone(),
            applied_resname: node.applied_resname.clone(),
        })
        .collect::<Vec<_>>();
    let graph_edge_specs = compiled_plan
        .edges
        .iter()
        .map(|edge| GraphEdgeSpec {
            edge_id: edge.edge_id.clone(),
            parent: edge.parent,
            child: edge.child,
            parent_attach_atom: edge.parent_attach_atom.clone(),
            parent_leaving_atoms: edge.parent_leaving_atoms.clone(),
            child_attach_atom: edge.child_attach_atom.clone(),
            child_leaving_atoms: edge.child_leaving_atoms.clone(),
            bond_order: edge.bond_order,
            branch_spread: edge.branch_spread.clone(),
            torsion_mode: edge.torsion_mode.clone(),
            torsion_deg: edge.torsion_deg,
            torsion_window_deg: edge.torsion_window_deg,
        })
        .collect::<Vec<_>>();
    let graph_root_idx = compiled_plan
        .nodes
        .iter()
        .position(|node| node.node_id == compiled_plan.expanded_root_node_id)
        .unwrap_or(0);
    timings.prepare_graph_specs = elapsed_ms(graph_prep_started);

    Ok((
        PreparedBuildExecution {
            compiled_plan,
            compiled_sequence,
            token_junctions,
            base_mode,
            graph_node_specs,
            graph_edge_specs,
            graph_root_idx,
        },
        timings,
    ))
}

fn assess_parameter_source(
    prepared: &PreparedBuildExecution,
    resolved: &ResolvedBuildRequest,
) -> Result<TrainingSourceAssessment, Vec<ErrorDetail>> {
    let assessment = assess_training_source(
        Path::new(&resolved.source_coordinates),
        resolved.source_charge_manifest.as_deref().map(Path::new),
        resolved.source_topology_ref.as_deref().map(Path::new),
        resolved.source_forcefield_ref.as_deref().map(Path::new),
        &prepared.graph_node_specs,
        &prepared.token_junctions,
    )
    .map_err(|err| {
        vec![to_error(
            "E_SOURCE_GEOMETRY",
            Some("/source_ref/bundle_path".into()),
            err.to_string(),
        )]
    })?;
    if assessment.parameter_source == "rejected" {
        let message = if assessment.reasons.is_empty() {
            "training source is unreliable and no trusted fallback parameter source is available"
                .to_string()
        } else {
            format!(
                "training source is unreliable: {}",
                assessment.reasons.join(", ")
            )
        };
        return Err(vec![to_error(
            "E_SOURCE_UNRELIABLE",
            Some("/source_ref/bundle_path".into()),
            message,
        )]);
    }
    Ok(assessment)
}

fn preflight_temp_dir(request_id: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "warp_build_preflight_{}_{}_{}",
        request_id.replace(|ch: char| !ch.is_ascii_alphanumeric(), "_"),
        std::process::id(),
        nanos
    ))
}

fn shallow_preflight_summary() -> PreflightSummary {
    PreflightSummary {
        executed: false,
        mode: Some("shallow".into()),
        reason: Some(
            "geometry/QC preflight skipped because validation.depth=shallow; default validation is deep"
                .into(),
        ),
        base_conformation_mode: None,
        target_residue_count: None,
        applied_junctions: None,
        timings_ms: BuildPhaseTimingsMs::default(),
        qc: None,
        solver: None,
        solver_cleanup: None,
        relax: None,
        overlap_status: overlap_status_summary(None, None, None),
        parameter_source_decision: None,
    }
}

fn qc_failure_warnings(built_path: &Path, raw_coordinates: Option<&str>) -> Vec<ErrorDetail> {
    let mut warnings = vec![to_warning(
        "W_DEBUG_ARTIFACT",
        Some("/artifacts/coordinates".into()),
        format!(
            "staged debug coordinates retained at '{}' for recovery/debugging; final build artifacts were not written",
            built_path.display()
        ),
    )];
    if let Some(path) = raw_coordinates {
        warnings.push(to_warning(
            "W_DEBUG_ARTIFACT",
            Some("/artifacts/raw_coordinates".into()),
            format!(
                "raw debug coordinates retained at '{}' for recovery/debugging",
                path
            ),
        ));
    }
    warnings
}

fn preflight_build(
    req: &BuildRequest,
    bundle: &SourceBundle,
    resolved: &ResolvedBuildRequest,
) -> Result<PreflightSummary, Vec<ErrorDetail>> {
    let (prepared, mut timings_ms) =
        prepare_build_execution(req, bundle, resolved.seed).map_err(|err| vec![err])?;
    let parameter_source_decision = assess_parameter_source(&prepared, resolved)?;
    if parameter_source_decision.parameter_source != "synthetic_pdb" {
        return Ok(PreflightSummary {
            executed: false,
            mode: Some("deep".into()),
            reason: Some(format!(
                "geometry/QC preflight skipped because training source was marked '{}' and fallback parameter source '{}' was selected",
                parameter_source_decision.quality, parameter_source_decision.parameter_source
            )),
            base_conformation_mode: Some(prepared.base_mode),
            target_residue_count: Some(prepared.compiled_plan.nodes.len()),
            applied_junctions: Some(applied_junctions_value(&prepared.token_junctions)),
            timings_ms,
            qc: None,
            solver: None,
            solver_cleanup: None,
            relax: None,
            overlap_status: overlap_status_summary(None, None, None),
            parameter_source_decision: Some(parameter_source_decision),
        });
    }
    let temp_dir = preflight_temp_dir(&req.request_id);
    fs::create_dir_all(&temp_dir)
        .map_err(|err| vec![to_error("E_OUTPUT_WRITE", None, err.to_string())])?;
    let final_coords = temp_dir.join("preflight_coordinates.pdb");
    let raw_coords = temp_dir.join("preflight_raw_coordinates.pdb");
    let final_coords_text = final_coords.to_string_lossy().to_string();
    let raw_coords_text = raw_coords.to_string_lossy().to_string();
    let build_started = Instant::now();
    let build_result = build_polymer_graph(
        &resolved.source_coordinates,
        resolved.source_charge_manifest.as_deref(),
        resolved.source_topology_ref.as_deref(),
        bundle.training_context.training_oligomer_n,
        &prepared.graph_node_specs,
        &prepared.graph_edge_specs,
        prepared.graph_root_idx,
        &prepared.base_mode,
        match req.target.stereochemistry.mode.as_str() {
            "inherit" => "training",
            other => other,
        },
        resolved.seed,
        true,
        &final_coords_text,
    );
    timings_ms.build_graph = elapsed_ms(build_started);
    let mut built = match build_result {
        Ok(built) => built,
        Err(err) => {
            let _ = fs::remove_dir_all(&temp_dir);
            return Err(vec![to_error(
                "E_SOURCE_GEOMETRY",
                Some("/source_ref/bundle_path".into()),
                format!("preflight build failed: {err}"),
            )]);
        }
    };
    let defer_qc_until_synthetic_cleanup =
        parameter_source_decision.parameter_source == "synthetic_pdb";
    let cleanup_initial_positions = output_positions(&built.output);
    let (mut internal_cleanup_report, cleanup_elapsed_ms) =
        run_internal_cleanup_pipeline(&mut built, &prepared, &req.realization.conformation_mode);
    timings_ms.solver_cleanup = cleanup_elapsed_ms;
    if internal_cleanup_report.is_some() && !defer_qc_until_synthetic_cleanup {
        if let Err(err) = ensure_build_qc_passes(&built.qc_report) {
            let _ = fs::remove_dir_all(&temp_dir);
            return Err(vec![to_error(
                "E_SOURCE_GEOMETRY",
                Some("/source_ref/bundle_path".into()),
                format!("preflight QC failed: {err}"),
            )]);
        }
    }
    let relax_report = req.realization.relax.as_ref().map(|relax| {
        let relax_started = Instant::now();
        let report = relax_built_output(
            &mut built.output,
            &prepared.compiled_plan,
            built.step_length_angstrom,
            relax,
            Some(raw_coords_text.clone()),
        );
        timings_ms.user_relax = elapsed_ms(relax_started);
        report
    });
    if relax_report.is_some() {
        built.qc_report = recompute_build_qc_report(&built.output, &built.qc_context);
        if !defer_qc_until_synthetic_cleanup {
            if let Err(err) = ensure_build_qc_passes(&built.qc_report) {
                let _ = fs::remove_dir_all(&temp_dir);
                return Err(vec![to_error(
                    "E_SOURCE_GEOMETRY",
                    Some("/source_ref/bundle_path".into()),
                    format!("preflight QC failed after relax: {err}"),
                )]);
            }
        }
    }
    if parameter_source_decision.parameter_source == "synthetic_pdb" {
        let (synthetic_report, synthetic_elapsed_ms, _synthetic_warning) =
            run_synthetic_topology_cleanup_stage(
                &mut built,
                &resolved.source_coordinates,
                resolved.source_charge_manifest.as_deref(),
            );
        timings_ms.solver_cleanup = timings_ms
            .solver_cleanup
            .saturating_add(synthetic_elapsed_ms);
        if let Some(followup) = synthetic_report {
            let final_positions = output_positions(&built.output);
            internal_cleanup_report = Some(if let Some(primary) = internal_cleanup_report.take() {
                merge_relax_report_with_followup_stage(
                    primary,
                    followup,
                    &cleanup_initial_positions,
                    &final_positions,
                )
            } else {
                followup
            });
        }
        if let Err(err) = ensure_build_qc_passes(&built.qc_report) {
            let _ = fs::remove_dir_all(&temp_dir);
            return Err(vec![to_error(
                "E_SOURCE_GEOMETRY",
                Some("/source_ref/bundle_path".into()),
                format!("preflight QC failed after synthetic topology cleanup: {err}"),
            )]);
        }
    }
    let overlap_status = overlap_status_summary(
        Some(&built.output),
        internal_cleanup_report.as_ref(),
        relax_report.as_ref(),
    );
    let _ = fs::remove_dir_all(&temp_dir);
    Ok(PreflightSummary {
        executed: true,
        mode: None,
        reason: None,
        base_conformation_mode: Some(prepared.base_mode),
        target_residue_count: Some(prepared.compiled_plan.nodes.len()),
        applied_junctions: Some(applied_junctions_value(&prepared.token_junctions)),
        timings_ms,
        qc: Some(built.qc_report),
        solver: built.solver_report,
        solver_cleanup: internal_cleanup_report,
        relax: relax_report,
        overlap_status,
        parameter_source_decision: Some(parameter_source_decision),
    })
}

fn resolved_artifact_request(resolved: &ResolvedBuildRequest) -> ArtifactRequest {
    ArtifactRequest {
        coordinates: resolved.artifacts.coordinates.clone(),
        raw_coordinates: resolved.artifacts.raw_coordinates.clone(),
        build_manifest: resolved.artifacts.build_manifest.clone(),
        charge_manifest: resolved.artifacts.charge_manifest.clone(),
        inpcrd: Some(resolved.artifacts.inpcrd.clone()),
        topology: resolved.artifacts.topology.clone(),
        topology_graph: Some(resolved.artifacts.topology_graph.clone()),
        ensemble_manifest: resolved.artifacts.ensemble_manifest.clone(),
        forcefield_ref: resolved.artifacts.forcefield_ref.clone(),
    }
}

fn artifact_path_map(artifacts: &ArtifactRequest) -> BTreeMap<String, String> {
    let mut paths = BTreeMap::new();
    paths.insert("coordinates".into(), artifacts.coordinates.clone());
    if let Some(path) = artifacts.raw_coordinates.clone() {
        paths.insert("raw_coordinates".into(), path);
    }
    paths.insert("build_manifest".into(), artifacts.build_manifest.clone());
    paths.insert("charge_manifest".into(), artifacts.charge_manifest.clone());
    if let Some(path) = artifacts.inpcrd.clone() {
        paths.insert("inpcrd".into(), path);
    }
    if let Some(path) = artifacts.topology.clone() {
        paths.insert("topology".into(), path);
    }
    if let Some(path) = artifacts.topology_graph.clone() {
        paths.insert("topology_graph".into(), path);
    }
    if let Some(path) = artifacts.ensemble_manifest.clone() {
        paths.insert("ensemble_manifest".into(), path);
    }
    if let Some(path) = artifacts.forcefield_ref.clone() {
        paths.insert("forcefield_ref".into(), path);
    }
    paths
}

fn artifact_digests_value(artifacts: &ArtifactRequest) -> Value {
    let digest = |path: Option<&str>| path.and_then(|path| sha256_file(Path::new(path)).ok());
    json!({
        "coordinates": digest(Some(&artifacts.coordinates)),
        "raw_coordinates": digest(artifacts.raw_coordinates.as_deref()),
        "charge_manifest": digest(Some(&artifacts.charge_manifest)),
        "inpcrd": digest(artifacts.inpcrd.as_deref()),
        "topology": digest(artifacts.topology.as_deref()),
        "topology_graph": digest(artifacts.topology_graph.as_deref()),
        "ensemble_manifest": digest(artifacts.ensemble_manifest.as_deref()),
        "forcefield_ref": digest(artifacts.forcefield_ref.as_deref()),
    })
}

fn replace_json_path_strings(value: &mut Value, replacements: &BTreeMap<String, String>) {
    match value {
        Value::String(text) => {
            if let Some(replacement) = replacements.get(text) {
                *text = replacement.clone();
            }
        }
        Value::Array(items) => {
            for item in items {
                replace_json_path_strings(item, replacements);
            }
        }
        Value::Object(map) => {
            for item in map.values_mut() {
                replace_json_path_strings(item, replacements);
            }
        }
        _ => {}
    }
}

fn copy_cached_file(src: &str, dst: &str, path: &str) -> Result<(), ErrorDetail> {
    if !Path::new(src).exists() {
        return Err(to_error(
            "E_PREFLIGHT_CACHE",
            Some(path.into()),
            format!("cached artifact '{}' is missing", src),
        ));
    }
    ensure_parent(dst).map_err(|err| {
        to_error(
            "E_OUTPUT_WRITE",
            Some(path.into()),
            format!("failed to create output parent for '{}': {err}", dst),
        )
    })?;
    fs::copy(src, dst).map(|_| ()).map_err(|err| {
        to_error(
            "E_OUTPUT_WRITE",
            Some(path.into()),
            format!(
                "failed to copy cached artifact '{}' to '{}': {err}",
                src, dst
            ),
        )
    })
}

fn write_rewritten_cached_json(
    src: &str,
    dst: &str,
    path: &str,
    replacements: &BTreeMap<String, String>,
    mutate: impl FnOnce(&mut Value),
) -> Result<(), ErrorDetail> {
    let text = fs::read_to_string(src).map_err(|err| {
        to_error(
            "E_PREFLIGHT_CACHE",
            Some(path.into()),
            format!("failed to read cached JSON artifact '{}': {err}", src),
        )
    })?;
    let mut value: Value = serde_json::from_str(&text).map_err(|err| {
        to_error(
            "E_PREFLIGHT_CACHE",
            Some(path.into()),
            format!("cached JSON artifact '{}' is invalid: {err}", src),
        )
    })?;
    replace_json_path_strings(&mut value, replacements);
    mutate(&mut value);
    ensure_parent(dst).map_err(|err| {
        to_error(
            "E_OUTPUT_WRITE",
            Some(path.into()),
            format!("failed to create output parent for '{}': {err}", dst),
        )
    })?;
    fs::write(
        dst,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&value).unwrap_or_else(|_| "{}".into())
        ),
    )
    .map_err(|err| {
        to_error(
            "E_OUTPUT_WRITE",
            Some(path.into()),
            format!("failed to write cached JSON artifact '{}': {err}", dst),
        )
    })
}

fn verify_cached_artifact_digests(
    artifacts: &ArtifactRequest,
    digests: &Value,
) -> Result<(), ErrorDetail> {
    for (key, path) in artifact_path_map(artifacts) {
        let Some(expected) = digests.get(&key).and_then(Value::as_str) else {
            continue;
        };
        let actual = sha256_file(Path::new(&path)).map_err(|err| {
            to_error(
                "E_PREFLIGHT_CACHE",
                Some(format!("artifact_digests.{key}")),
                format!("failed to hash cached artifact '{}': {err}", path),
            )
        })?;
        if actual != expected {
            return Err(to_error(
                "E_PREFLIGHT_CACHE",
                Some(format!("artifact_digests.{key}")),
                format!("cached artifact digest mismatch for '{}'", key),
            ));
        }
    }
    Ok(())
}

fn materialize_cached_artifacts(
    cached: &ArtifactRequest,
    final_artifacts: &ArtifactRequest,
) -> Result<(), ErrorDetail> {
    let cached_paths = artifact_path_map(cached);
    let final_paths = artifact_path_map(final_artifacts);
    let replacements = cached_paths
        .iter()
        .filter_map(|(key, cached)| {
            final_paths
                .get(key)
                .map(|final_path| (cached.clone(), final_path.clone()))
        })
        .collect::<BTreeMap<_, _>>();

    for key in [
        "coordinates",
        "raw_coordinates",
        "inpcrd",
        "topology",
        "forcefield_ref",
    ] {
        if let (Some(src), Some(dst)) = (cached_paths.get(key), final_paths.get(key)) {
            copy_cached_file(src, dst, &format!("/artifacts/{key}"))?;
        }
    }
    for key in ["topology_graph", "ensemble_manifest"] {
        if let (Some(src), Some(dst)) = (cached_paths.get(key), final_paths.get(key)) {
            write_rewritten_cached_json(
                src,
                dst,
                &format!("/artifacts/{key}"),
                &replacements,
                |_| {},
            )?;
        }
    }
    if let (Some(src), Some(dst)) = (
        cached_paths.get("charge_manifest"),
        final_paths.get("charge_manifest"),
    ) {
        write_rewritten_cached_json(
            src,
            dst,
            "/artifacts/charge_manifest",
            &replacements,
            |_| {},
        )?;
    }
    if let (Some(src), Some(dst)) = (
        cached_paths.get("build_manifest"),
        final_paths.get("build_manifest"),
    ) {
        let digests = artifact_digests_value(final_artifacts);
        write_rewritten_cached_json(
            src,
            dst,
            "/artifacts/build_manifest",
            &replacements,
            |value| {
                if let Some(object) = value.as_object_mut() {
                    object.insert("artifact_digests".into(), digests);
                }
            },
        )?;
    }
    Ok(())
}

fn materialize_preflight_artifact_cache(
    req: &BuildRequest,
    resolved: &ResolvedBuildRequest,
    preflight: &PreflightSummary,
    mut cache: PreflightCacheSummary,
) -> Result<PreflightCacheSummary, Vec<ErrorDetail>> {
    let Some(cache_artifacts) = cache.artifact_paths.clone() else {
        return Ok(cache);
    };
    let Some(record_path) = cache.record_path.clone() else {
        return Ok(cache);
    };
    let mut cached_req = req.clone();
    cached_req.validation.cache_mode = "off".into();
    cached_req.validation.cache_dir = None;
    cached_req.artifacts = cache_artifacts;
    let request_text = serde_json::to_string(&cached_req).map_err(|err| {
        vec![to_error(
            "E_PREFLIGHT_CACHE",
            Some("/validation/cache_dir".into()),
            format!("failed to serialize cache build request: {err}"),
        )]
    })?;
    let (code, run_result) = run_request_json(&request_text, false);
    if code != 0 {
        return Err(vec![to_error(
            "E_PREFLIGHT_CACHE",
            Some("/validation/cache_dir".into()),
            format!("failed to materialize preflight artifact cache: {run_result}"),
        )]);
    }
    let actual_artifacts: ArtifactRequest = serde_json::from_value(
        run_result
            .get("artifacts")
            .cloned()
            .unwrap_or_else(|| json!({})),
    )
    .map_err(|err| {
        vec![to_error(
            "E_PREFLIGHT_CACHE",
            Some("/validation/cache_dir".into()),
            format!("cached run did not return a valid artifact contract: {err}"),
        )]
    })?;
    let record = json!({
        "schema_version": "warp-build.preflight-cache.v1",
        "cache_key": cache.cache_key,
        "request_digest": cache.request_digest,
        "input_digest": cache.input_digest,
        "created_unix_seconds": SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|value| value.as_secs())
            .unwrap_or(0),
        "normalized_request": resolved.normalized_request,
        "resolved_inputs": resolved.resolved_inputs,
        "preflight": preflight,
        "artifacts": actual_artifacts,
        "artifact_digests": artifact_digests_value(&actual_artifacts),
        "run_result": run_result,
    });
    if let Some(parent) = Path::new(&record_path).parent() {
        fs::create_dir_all(parent).map_err(|err| {
            vec![to_error(
                "E_OUTPUT_WRITE",
                Some("/validation/cache_dir".into()),
                err.to_string(),
            )]
        })?;
    }
    fs::write(
        &record_path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&record).unwrap_or_else(|_| "{}".into())
        ),
    )
    .map_err(|err| {
        vec![to_error(
            "E_OUTPUT_WRITE",
            Some("/validation/cache_dir".into()),
            err.to_string(),
        )]
    })?;
    cache.state = "written".into();
    cache.artifact_paths = Some(actual_artifacts);
    Ok(cache)
}

fn try_reuse_preflight_artifact_cache(
    req: &BuildRequest,
    resolved: &ResolvedBuildRequest,
) -> Result<Option<(i32, Value)>, ErrorDetail> {
    let cache_mode = validation_cache_mode(req)?;
    if !matches!(cache_mode, "prefer" | "require") {
        return Ok(None);
    }
    let cache = preflight_cache_summary(
        &resolved.normalized_request,
        &resolved.resolved_inputs,
        cache_mode,
        req.validation.cache_dir.as_deref(),
        true,
        "lookup",
        None,
    );
    let Some(record_path) = cache.record_path.as_ref() else {
        return Ok(None);
    };
    if !Path::new(record_path).exists() {
        return Ok(None);
    }
    let record_text = fs::read_to_string(record_path).map_err(|err| {
        to_error(
            "E_PREFLIGHT_CACHE",
            Some("/validation/cache_dir".into()),
            format!(
                "failed to read preflight cache record '{}': {err}",
                record_path
            ),
        )
    })?;
    let record: Value = serde_json::from_str(&record_text).map_err(|err| {
        to_error(
            "E_PREFLIGHT_CACHE",
            Some("/validation/cache_dir".into()),
            format!("invalid preflight cache record '{}': {err}", record_path),
        )
    })?;
    if record.get("schema_version").and_then(Value::as_str) != Some("warp-build.preflight-cache.v1")
        || record.get("cache_key").and_then(Value::as_str) != Some(cache.cache_key.as_str())
        || record.get("input_digest").and_then(Value::as_str) != Some(cache.input_digest.as_str())
    {
        return Ok(None);
    }
    let cached_artifacts: ArtifactRequest = serde_json::from_value(
        record
            .get("artifacts")
            .cloned()
            .unwrap_or_else(|| json!({})),
    )
    .map_err(|err| {
        to_error(
            "E_PREFLIGHT_CACHE",
            Some("/validation/cache_dir".into()),
            format!("preflight cache record has invalid artifacts: {err}"),
        )
    })?;
    verify_cached_artifact_digests(
        &cached_artifacts,
        record.get("artifact_digests").unwrap_or(&Value::Null),
    )?;
    let final_artifacts = resolved_artifact_request(resolved);
    materialize_cached_artifacts(&cached_artifacts, &final_artifacts)?;

    let mut result = record.get("run_result").cloned().ok_or_else(|| {
        to_error(
            "E_PREFLIGHT_CACHE",
            Some("/validation/cache_dir".into()),
            "preflight cache record is missing run_result",
        )
    })?;
    let cached_paths = artifact_path_map(&cached_artifacts);
    let final_paths = artifact_path_map(&final_artifacts);
    let replacements = cached_paths
        .iter()
        .filter_map(|(key, cached)| {
            final_paths
                .get(key)
                .map(|final_path| (cached.clone(), final_path.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    replace_json_path_strings(&mut result, &replacements);
    if let Some(object) = result.as_object_mut() {
        object.insert(
            "artifacts".into(),
            serde_json::to_value(&final_artifacts).unwrap_or_else(|_| json!({})),
        );
        let warning = serde_json::to_value(to_warning(
            "W_PREFLIGHT_CACHE_HIT",
            Some("/validation/cache_dir".into()),
            format!("reused preflight artifact cache '{}'", record_path),
        ))
        .unwrap_or_else(|_| json!({}));
        if let Some(warnings) = object
            .entry("warnings")
            .or_insert_with(|| json!([]))
            .as_array_mut()
        {
            warnings.push(warning);
        }
    }
    let code = if result.get("status").and_then(Value::as_str) == Some("salvaged") {
        3
    } else {
        0
    };
    Ok(Some((code, result)))
}

fn topology_transfer_supported(bundle: &SourceBundle) -> bool {
    bundle
        .artifacts
        .source_topology_ref
        .as_deref()
        .map(|path| {
            Path::new(path)
                .extension()
                .and_then(|value| value.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("prmtop"))
                .unwrap_or(false)
        })
        .unwrap_or(false)
}

fn random_copolymer_sequence(
    composition: &BTreeMap<String, usize>,
    seed: u64,
) -> Result<Vec<String>, ErrorDetail> {
    let mut sequence = Vec::new();
    for (token, count) in composition {
        if *count == 0 {
            continue;
        }
        sequence.extend(std::iter::repeat_n(token.clone(), *count));
    }
    if sequence.is_empty() {
        return Err(to_error(
            "E_INVALID_TARGET",
            Some("/target/composition".into()),
            "random_copolymer requires a non-empty composition",
        ));
    }
    let mut state = seed.max(1);
    for idx in (1..sequence.len()).rev() {
        state = state.wrapping_mul(6364136223846793005u64).wrapping_add(1);
        let swap_idx = (state as usize) % (idx + 1);
        sequence.swap(idx, swap_idx);
    }
    Ok(sequence)
}

fn expand_sequence(target: &BuildTarget, seed: u64) -> Result<Vec<String>, ErrorDetail> {
    match target.mode.as_str() {
        "linear_homopolymer" => {
            let token = target
                .repeat_unit
                .as_deref()
                .unwrap_or("")
                .trim()
                .to_string();
            let n_repeat = target.n_repeat.unwrap_or(0);
            if token.is_empty() || n_repeat == 0 {
                return Ok(Vec::new());
            }
            Ok(vec![token; n_repeat])
        }
        "linear_sequence_polymer" => {
            let sequence = target.sequence.clone().unwrap_or_default();
            if sequence.is_empty() {
                return Ok(Vec::new());
            }
            let repeat_count = target.repeat_count.unwrap_or(1);
            if repeat_count == 0 {
                return Err(to_error(
                    "E_INVALID_TARGET",
                    Some("/target/repeat_count".into()),
                    "repeat_count must be >= 1",
                ));
            }
            let mut expanded = Vec::with_capacity(sequence.len() * repeat_count);
            for _ in 0..repeat_count {
                expanded.extend(sequence.iter().cloned());
            }
            Ok(expanded)
        }
        "block_copolymer" => {
            let blocks = target.blocks.as_ref().ok_or_else(|| {
                to_error(
                    "E_INVALID_TARGET",
                    Some("/target/blocks".into()),
                    "block_copolymer requires blocks",
                )
            })?;
            if blocks.is_empty() {
                return Err(to_error(
                    "E_INVALID_TARGET",
                    Some("/target/blocks".into()),
                    "block_copolymer requires at least one block",
                ));
            }
            let mut sequence = Vec::new();
            for (idx, block) in blocks.iter().enumerate() {
                if block.token.trim().is_empty() || block.count == 0 {
                    return Err(to_error(
                        "E_INVALID_TARGET",
                        Some(format!("/target/blocks/{idx}")),
                        "each block requires token and count >= 1",
                    ));
                }
                sequence.extend(std::iter::repeat_n(block.token.clone(), block.count));
            }
            Ok(sequence)
        }
        "random_copolymer" => {
            let composition = target.composition.as_ref().ok_or_else(|| {
                to_error(
                    "E_INVALID_TARGET",
                    Some("/target/composition".into()),
                    "random_copolymer requires composition",
                )
            })?;
            if let Some(total_units) = target.total_units {
                let total = composition.values().sum::<usize>();
                if total != total_units {
                    return Err(to_error(
                        "E_INVALID_TARGET",
                        Some("/target/total_units".into()),
                        format!(
                            "target.total_units ({total_units}) must equal composition total ({total})"
                        ),
                    ));
                }
            }
            random_copolymer_sequence(composition, seed)
        }
        other => Err(to_error(
            "E_UNSUPPORTED_TARGET",
            Some("/target/mode".into()),
            format!(
                "target mode '{}' unsupported; supported: {}",
                other,
                SUPPORTED_TARGET_MODES.join(", ")
            ),
        )),
    }
}

fn resolve_terminus_token(
    bundle: &SourceBundle,
    policy: &str,
    fallback_token: &str,
    path: &str,
) -> Result<String, ErrorDetail> {
    let raw = policy.trim();
    if raw.is_empty()
        || raw.eq_ignore_ascii_case("default")
        || raw.eq_ignore_ascii_case("training")
        || raw.eq_ignore_ascii_case("source_default")
    {
        return Ok(fallback_token.to_string());
    }
    if bundle.unit_library.contains_key(raw) || bundle.motif_library.contains_key(raw) {
        if bundle.motif_library.contains_key(raw)
            && !motif_supports_linear_ports(bundle, raw, path)?
        {
            return Err(to_error(
                "E_UNSUPPORTED_TERMINI",
                Some(path.into()),
                format!(
                    "motif token '{}' must expose head and tail ports for linear termini use",
                    raw
                ),
            ));
        }
        return Ok(raw.to_string());
    }
    Err(to_error(
        "E_UNSUPPORTED_TERMINI",
        Some(path.into()),
        format!(
            "unsupported termini policy '{}' ; expected default/source_default or a token from unit_library/motif_library",
            raw
        ),
    ))
}

fn is_default_termini_policy(policy: &str) -> bool {
    let raw = policy.trim();
    raw.is_empty()
        || raw.eq_ignore_ascii_case("default")
        || raw.eq_ignore_ascii_case("training")
        || raw.eq_ignore_ascii_case("source_default")
}

fn resolve_cap_token(
    bundle: &SourceBundle,
    policy: &str,
    path: &str,
) -> Result<Option<String>, ErrorDetail> {
    let raw = policy.trim();
    if is_default_termini_policy(raw) {
        return Ok(None);
    }
    if bundle.unit_library.contains_key(raw) {
        return Ok(Some(raw.to_string()));
    }
    Err(to_error(
        "E_UNSUPPORTED_TERMINI",
        Some(path.into()),
        format!(
            "unsupported termini policy '{}' ; expected default/source_default or a token from unit_library",
            raw
        ),
    ))
}

fn resolve_cap_binding(
    bundle: &SourceBundle,
    cap: &CapBinding,
    path: &str,
) -> Result<String, ErrorDetail> {
    if let Some(unit) = bundle.unit_library.get(&cap.token) {
        if let Some(junction) = cap.junction.as_deref() {
            if unit.junctions.contains_key(junction) {
                return Ok(junction.to_string());
            }
            return Err(to_error(
                "E_SOURCE_SCHEMA",
                Some(path.into()),
                format!(
                    "cap token '{}' does not expose requested junction '{}'",
                    cap.token, junction
                ),
            ));
        }
        if unit.junctions.len() == 1 {
            return Ok(unit.junctions.keys().next().cloned().unwrap_or_default());
        }
        if unit.junctions.contains_key("head") {
            return Ok("head".into());
        }
        if unit.junctions.contains_key("tail") {
            return Ok("tail".into());
        }
        return Err(to_error(
            "E_SOURCE_SCHEMA",
            Some(path.into()),
            format!(
                "cap token '{}' is ambiguous; provide cap.junction",
                cap.token
            ),
        ));
    }
    if let Some(motif) = bundle.motif_library.get(&cap.token) {
        if let Some(port) = cap.port.as_deref() {
            if motif.exposed_ports.contains_key(port) {
                return Ok(port.to_string());
            }
            return Err(to_error(
                "E_SOURCE_SCHEMA",
                Some(path.into()),
                format!(
                    "cap motif '{}' does not expose requested port '{}'",
                    cap.token, port
                ),
            ));
        }
        if motif.exposed_ports.len() == 1 {
            return Ok(motif
                .exposed_ports
                .keys()
                .next()
                .cloned()
                .unwrap_or_default());
        }
        return Err(to_error(
            "E_SOURCE_SCHEMA",
            Some(path.into()),
            format!("cap motif '{}' is ambiguous; provide cap.port", cap.token),
        ));
    }
    Err(to_error(
        "E_UNKNOWN_TOKEN",
        Some(path.into()),
        format!(
            "cap token '{}' missing from source token libraries",
            cap.token
        ),
    ))
}

fn cap_binding_allowed(cap: &CapBinding, allowed: &[CapBinding]) -> bool {
    allowed.is_empty() || allowed.iter().any(|item| item == cap)
}

fn compile_sequence_tokens(
    req: &BuildRequest,
    bundle: &SourceBundle,
    expanded_sequence: &[String],
) -> Result<Vec<String>, ErrorDetail> {
    let mut compiled = expanded_sequence.to_vec();
    if let Some(first) = compiled.first_mut() {
        *first = resolve_terminus_token(
            bundle,
            &req.target.termini.head,
            first,
            "target.termini.head",
        )?;
    }
    if let Some(last) = compiled.last_mut() {
        *last = resolve_terminus_token(
            bundle,
            &req.target.termini.tail,
            last,
            "target.termini.tail",
        )?;
    }
    Ok(compiled)
}

#[derive(Clone, Debug)]
struct CompiledBuildNode {
    node_id: String,
    request_node_id: String,
    token: String,
    token_kind: String,
    source_token: String,
    motif_instance_id: Option<String>,
    motif_token: Option<String>,
    template_resname: String,
    applied_resname: String,
    branch_depth: usize,
    branch_path: String,
}

#[derive(Clone, Debug)]
struct CompiledBuildEdge {
    edge_id: String,
    parent: usize,
    child: usize,
    parent_port: String,
    child_port: String,
    parent_junction: String,
    child_junction: String,
    parent_attach_atom: String,
    parent_leaving_atoms: Vec<String>,
    child_attach_atom: String,
    child_leaving_atoms: Vec<String>,
    bond_order: u8,
    layout_mode: String,
    branch_spread: String,
    torsion_mode: String,
    torsion_deg: Option<f32>,
    torsion_window_deg: Option<[f32; 2]>,
    ring_mode: Option<String>,
}

#[derive(Clone, Debug)]
struct CompiledBuildPlan {
    nodes: Vec<CompiledBuildNode>,
    edges: Vec<CompiledBuildEdge>,
    applied_caps: Vec<CompiledAppliedCap>,
    request_root_node_id: String,
    expanded_root_node_id: String,
    root_token: String,
    arm_count: usize,
    max_branch_depth: usize,
    graph_has_cycle: bool,
}

#[derive(Clone, Debug)]
struct ResolvedTokenPort {
    node_idx: usize,
    node_id: String,
    binding: String,
}

#[derive(Clone, Debug)]
struct CompiledAppliedCap {
    node_id: String,
    request_node_id: String,
    resid: usize,
    port_name: String,
    junction: String,
    cap: CapBinding,
    application_source: String,
    cap_node_id: String,
    cap_resid: usize,
}

#[derive(Clone, Debug)]
struct TokenExpansion {
    root_node_idx: usize,
    request_node_id: String,
    source_token: String,
    exposed_ports: BTreeMap<String, ResolvedTokenPort>,
}

fn bundle_has_token(bundle: &SourceBundle, token: &str) -> bool {
    bundle.unit_library.contains_key(token) || bundle.motif_library.contains_key(token)
}

fn motif_supports_linear_ports(
    bundle: &SourceBundle,
    token: &str,
    path: &str,
) -> Result<bool, ErrorDetail> {
    let motif = bundle.motif_library.get(token).ok_or_else(|| {
        to_error(
            "E_UNKNOWN_TOKEN",
            Some(path.into()),
            format!(
                "motif token '{}' not present in source motif_library",
                token
            ),
        )
    })?;
    Ok(motif.exposed_ports.contains_key("head") && motif.exposed_ports.contains_key("tail"))
}

fn canonical_edge_id(
    left: &str,
    left_port: &str,
    right: &str,
    right_port: &str,
    ordinal: usize,
) -> String {
    format!("{left}:{left_port}->{right}:{right_port}#{ordinal}")
}

fn effective_policy<'a>(req: &'a BuildRequest) -> Option<&'a ConformerPolicy> {
    req.conformer_policy.as_ref()
}

fn append_unit_token(
    bundle: &SourceBundle,
    token: &str,
    request_node_id: &str,
    node_id_prefix: &str,
    branch_path: &str,
    applied_resname: Option<&str>,
    nodes: &mut Vec<CompiledBuildNode>,
) -> Result<TokenExpansion, ErrorDetail> {
    let node_idx = nodes.len();
    nodes.push(compiled_node(
        node_id_prefix,
        request_node_id,
        token,
        "unit",
        token,
        None,
        None,
        bundle,
        0,
        branch_path.to_string(),
        applied_resname,
    )?);
    let unit = bundle.unit_library.get(token).ok_or_else(|| {
        to_error(
            "E_UNKNOWN_TOKEN",
            None,
            format!(
                "sequence token '{}' not present in source unit_library",
                token
            ),
        )
    })?;
    let exposed_ports = unit
        .junctions
        .keys()
        .map(|port| {
            (
                port.clone(),
                ResolvedTokenPort {
                    node_idx,
                    node_id: node_id_prefix.to_string(),
                    binding: port.clone(),
                },
            )
        })
        .collect();
    Ok(TokenExpansion {
        root_node_idx: node_idx,
        request_node_id: request_node_id.to_string(),
        source_token: token.to_string(),
        exposed_ports,
    })
}

fn append_motif_token(
    bundle: &SourceBundle,
    token: &str,
    request_node_id: &str,
    node_id_prefix: &str,
    branch_path: &str,
    nodes: &mut Vec<CompiledBuildNode>,
    edges: &mut Vec<CompiledBuildEdge>,
    conformer_policy: Option<&ConformerPolicy>,
) -> Result<TokenExpansion, ErrorDetail> {
    let motif = bundle.motif_library.get(token).ok_or_else(|| {
        to_error(
            "E_UNKNOWN_TOKEN",
            None,
            format!(
                "motif token '{}' not present in source motif_library",
                token
            ),
        )
    })?;
    if motif.nodes.is_empty() {
        return Err(to_error(
            "E_SOURCE_SCHEMA",
            Some(format!("motif_library.{token}.nodes")),
            "motif must declare at least one node",
        ));
    }
    let motif_instance_id = format!("motif::{request_node_id}");
    let mut local_to_global = BTreeMap::new();
    let mut global_by_local = BTreeMap::new();
    for local in &motif.nodes {
        let global_id = format!("{node_id_prefix}::{}", local.id);
        let node_idx = nodes.len();
        nodes.push(compiled_node(
            &global_id,
            request_node_id,
            &local.token,
            "motif",
            token,
            Some(motif_instance_id.clone()),
            Some(token.to_string()),
            bundle,
            0,
            format!("{branch_path}::{}", local.id),
            local.applied_resname.as_deref(),
        )?);
        local_to_global.insert(local.id.clone(), node_idx);
        global_by_local.insert(local.id.clone(), global_id);
    }
    let mut edge_ordinal = edges.len() + 1;
    for (edge_idx, edge) in motif.edges.iter().enumerate() {
        let from_idx = local_to_global.get(&edge.from).copied().ok_or_else(|| {
            to_error(
                "E_SOURCE_SCHEMA",
                Some(format!("motif_library.{token}.edges[{edge_idx}].from")),
                format!("motif edge references unknown node '{}'", edge.from),
            )
        })?;
        let to_idx = local_to_global.get(&edge.to).copied().ok_or_else(|| {
            to_error(
                "E_SOURCE_SCHEMA",
                Some(format!("motif_library.{token}.edges[{edge_idx}].to")),
                format!("motif edge references unknown node '{}'", edge.to),
            )
        })?;
        let from_token = nodes[from_idx].token.clone();
        let to_token = nodes[to_idx].token.clone();
        let mut compiled = compile_connection(
            bundle,
            canonical_edge_id(
                &global_by_local[&edge.from],
                &edge.from_junction,
                &global_by_local[&edge.to],
                &edge.to_junction,
                edge_ordinal,
            ),
            from_idx,
            &global_by_local[&edge.from],
            &from_token,
            &edge.from_junction,
            to_idx,
            &global_by_local[&edge.to],
            &to_token,
            &edge.to_junction,
            conformer_policy,
            None,
            &format!("motif_library.{token}.edges[{edge_idx}]"),
        )?;
        compiled.bond_order = edge.bond_order.max(compiled.bond_order);
        edges.push(compiled);
        edge_ordinal += 1;
    }
    let mut exposed_ports = BTreeMap::new();
    for (port_name, port) in &motif.exposed_ports {
        let node_idx = local_to_global.get(&port.node_id).copied().ok_or_else(|| {
            to_error(
                "E_SOURCE_SCHEMA",
                Some(format!(
                    "motif_library.{token}.exposed_ports.{port_name}.node_id"
                )),
                format!(
                    "motif exposed port references unknown node '{}'",
                    port.node_id
                ),
            )
        })?;
        exposed_ports.insert(
            port_name.clone(),
            ResolvedTokenPort {
                node_idx,
                node_id: global_by_local[&port.node_id].clone(),
                binding: port.junction.clone(),
            },
        );
    }
    let root_local_id = motif
        .root_node_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| motif.nodes[0].id.as_str());
    let root_node_idx = local_to_global.get(root_local_id).copied().ok_or_else(|| {
        to_error(
            "E_SOURCE_SCHEMA",
            Some(format!("motif_library.{token}.root_node_id")),
            format!(
                "motif root_node_id '{}' not present in motif nodes",
                root_local_id
            ),
        )
    })?;
    Ok(TokenExpansion {
        root_node_idx,
        request_node_id: request_node_id.to_string(),
        source_token: token.to_string(),
        exposed_ports,
    })
}

fn append_token_expansion(
    bundle: &SourceBundle,
    token: &str,
    request_node_id: &str,
    node_id_prefix: &str,
    branch_path: &str,
    applied_resname: Option<&str>,
    nodes: &mut Vec<CompiledBuildNode>,
    edges: &mut Vec<CompiledBuildEdge>,
    conformer_policy: Option<&ConformerPolicy>,
) -> Result<TokenExpansion, ErrorDetail> {
    if bundle.unit_library.contains_key(token) {
        append_unit_token(
            bundle,
            token,
            request_node_id,
            node_id_prefix,
            branch_path,
            applied_resname,
            nodes,
        )
    } else if bundle.motif_library.contains_key(token) {
        append_motif_token(
            bundle,
            token,
            request_node_id,
            node_id_prefix,
            branch_path,
            nodes,
            edges,
            conformer_policy,
        )
    } else {
        Err(to_error(
            "E_UNKNOWN_TOKEN",
            None,
            format!(
                "sequence token '{}' not present in source token libraries",
                token
            ),
        ))
    }
}

fn connect_token_expansions(
    bundle: &SourceBundle,
    nodes: &[CompiledBuildNode],
    edges: &mut Vec<CompiledBuildEdge>,
    left: &TokenExpansion,
    left_port: &str,
    right: &TokenExpansion,
    right_port: &str,
    edge_id: String,
    conformer_policy: Option<&ConformerPolicy>,
    edge_override: Option<&EdgeConformerOverride>,
    path: &str,
) -> Result<(), ErrorDetail> {
    let left_resolved = left.exposed_ports.get(left_port).ok_or_else(|| {
        to_error(
            "E_SOURCE_SCHEMA",
            Some(path.into()),
            format!(
                "token '{}' does not expose required port '{}'",
                left.source_token, left_port
            ),
        )
    })?;
    let right_resolved = right.exposed_ports.get(right_port).ok_or_else(|| {
        to_error(
            "E_SOURCE_SCHEMA",
            Some(path.into()),
            format!(
                "token '{}' does not expose required port '{}'",
                right.source_token, right_port
            ),
        )
    })?;
    let left_token = nodes[left_resolved.node_idx].token.clone();
    let right_token = nodes[right_resolved.node_idx].token.clone();
    edges.push(compile_connection(
        bundle,
        edge_id,
        left_resolved.node_idx,
        &left_resolved.node_id,
        &left_token,
        &left_resolved.binding,
        right_resolved.node_idx,
        &right_resolved.node_id,
        &right_token,
        &right_resolved.binding,
        conformer_policy,
        edge_override,
        path,
    )?);
    Ok(())
}

fn finalize_plan_metrics(mut plan: CompiledBuildPlan) -> Result<CompiledBuildPlan, ErrorDetail> {
    let root_idx = plan
        .nodes
        .iter()
        .position(|node| node.node_id == plan.expanded_root_node_id)
        .ok_or_else(|| {
            to_error(
                "E_RUNTIME_BUILD",
                None,
                format!(
                    "compiled root '{}' not present in expanded node set",
                    plan.expanded_root_node_id
                ),
            )
        })?;
    if plan.nodes.is_empty() {
        return Ok(plan);
    }
    let mut adjacency = vec![Vec::<usize>::new(); plan.nodes.len()];
    for edge in &plan.edges {
        adjacency[edge.parent].push(edge.child);
        adjacency[edge.child].push(edge.parent);
    }
    let mut queue = std::collections::VecDeque::from([root_idx]);
    let mut seen = BTreeSet::from([root_idx]);
    let mut parent = vec![None; plan.nodes.len()];
    let mut depth = vec![usize::MAX; plan.nodes.len()];
    depth[root_idx] = 0;
    while let Some(node_idx) = queue.pop_front() {
        for neighbor in adjacency[node_idx].clone() {
            if seen.insert(neighbor) {
                parent[neighbor] = Some(node_idx);
                depth[neighbor] = depth[node_idx] + 1;
                queue.push_back(neighbor);
            }
        }
    }
    if seen.len() != plan.nodes.len() {
        return Err(to_error(
            "E_INVALID_TARGET",
            Some("/target/graph_edges".into()),
            "compiled build graph must be connected",
        ));
    }
    let mut branch_paths = vec![String::new(); plan.nodes.len()];
    for (idx, node) in plan.nodes.iter_mut().enumerate() {
        node.branch_depth = depth[idx];
        branch_paths[idx] = if idx == root_idx {
            node.node_id.clone()
        } else if let Some(parent_idx) = parent[idx] {
            format!("{}>{}", branch_paths[parent_idx], node.node_id)
        } else {
            node.node_id.clone()
        };
        node.branch_path = branch_paths[idx].clone();
    }
    plan.arm_count = adjacency[root_idx].len();
    plan.max_branch_depth = depth.into_iter().max().unwrap_or(0);
    plan.graph_has_cycle = plan.edges.len() + 1 > plan.nodes.len();
    Ok(plan)
}

fn preferred_cap_binding(
    bundle: &SourceBundle,
    token: &str,
    preferred: &str,
    fallback: &str,
    path: &str,
) -> Result<String, ErrorDetail> {
    let unit = bundle.unit_library.get(token).ok_or_else(|| {
        to_error(
            "E_UNKNOWN_TOKEN",
            Some(path.into()),
            format!(
                "sequence token '{}' not present in source unit_library",
                token
            ),
        )
    })?;
    if unit.junctions.contains_key(preferred) {
        Ok(preferred.to_string())
    } else if unit.junctions.contains_key(fallback) {
        Ok(fallback.to_string())
    } else {
        Err(to_error(
            "E_SOURCE_SCHEMA",
            Some(path.into()),
            format!(
                "cap token '{}' must expose '{}' or '{}' junction binding",
                token, preferred, fallback
            ),
        ))
    }
}

fn apply_branched_caps(
    target: &BuildTarget,
    bundle: &SourceBundle,
    conformer_policy: Option<&ConformerPolicy>,
    mut plan: CompiledBuildPlan,
) -> Result<CompiledBuildPlan, ErrorDetail> {
    let head_cap = resolve_cap_token(bundle, &target.termini.head, "target.termini.head")?;
    let tail_cap = resolve_cap_token(bundle, &target.termini.tail, "target.termini.tail")?;
    if head_cap.is_none() && tail_cap.is_none() {
        return Ok(plan);
    }
    let mut used_junctions = vec![BTreeSet::<String>::new(); plan.nodes.len()];
    for edge in &plan.edges {
        if let Some(items) = used_junctions.get_mut(edge.parent) {
            items.insert(edge.parent_port.clone());
        }
        if let Some(items) = used_junctions.get_mut(edge.child) {
            items.insert(edge.child_port.clone());
        }
    }
    let base_count = plan.nodes.len();
    for node_idx in 0..base_count {
        let node = plan.nodes[node_idx].clone();
        let used = used_junctions.get(node_idx).cloned().unwrap_or_default();
        if let Some(cap_token) = head_cap.as_ref().filter(|_| !used.contains("head")) {
            let cap_idx = plan.nodes.len();
            plan.nodes.push(compiled_node(
                &format!("{}.head_cap", node.branch_path),
                &node.request_node_id,
                cap_token,
                "unit",
                cap_token,
                None,
                None,
                bundle,
                node.branch_depth + 1,
                format!("{}.head_cap", node.branch_path),
                None,
            )?);
            plan.edges.push(compile_connection(
                bundle,
                canonical_edge_id(
                    &node.node_id,
                    "head",
                    &plan.nodes[cap_idx].node_id,
                    "tail",
                    plan.edges.len() + 1,
                ),
                node_idx,
                &node.node_id,
                &node.token,
                "head",
                cap_idx,
                &plan.nodes[cap_idx].node_id,
                cap_token,
                &preferred_cap_binding(bundle, cap_token, "tail", "head", "target.termini.head")?,
                conformer_policy,
                None,
                "target.termini.head",
            )?);
        }
        if let Some(cap_token) = tail_cap.as_ref().filter(|_| !used.contains("tail")) {
            let cap_idx = plan.nodes.len();
            plan.nodes.push(compiled_node(
                &format!("{}.tail_cap", node.branch_path),
                &node.request_node_id,
                cap_token,
                "unit",
                cap_token,
                None,
                None,
                bundle,
                node.branch_depth + 1,
                format!("{}.tail_cap", node.branch_path),
                None,
            )?);
            plan.edges.push(compile_connection(
                bundle,
                canonical_edge_id(
                    &node.node_id,
                    "tail",
                    &plan.nodes[cap_idx].node_id,
                    "head",
                    plan.edges.len() + 1,
                ),
                node_idx,
                &node.node_id,
                &node.token,
                "tail",
                cap_idx,
                &plan.nodes[cap_idx].node_id,
                cap_token,
                &preferred_cap_binding(bundle, cap_token, "head", "tail", "target.termini.tail")?,
                conformer_policy,
                None,
                "target.termini.tail",
            )?);
        }
    }
    Ok(plan)
}

fn compiled_node(
    node_id: &str,
    request_node_id: &str,
    token: &str,
    token_kind: &str,
    source_token: &str,
    motif_instance_id: Option<String>,
    motif_token: Option<String>,
    bundle: &SourceBundle,
    branch_depth: usize,
    branch_path: String,
    applied_resname: Option<&str>,
) -> Result<CompiledBuildNode, ErrorDetail> {
    let unit = bundle.unit_library.get(token).ok_or_else(|| {
        to_error(
            "E_UNKNOWN_TOKEN",
            None,
            format!(
                "sequence token '{}' not present in source unit_library",
                token
            ),
        )
    })?;
    let template_resname = unit
        .template_resname
        .clone()
        .unwrap_or_else(|| token.to_string());
    Ok(CompiledBuildNode {
        node_id: node_id.to_string(),
        request_node_id: request_node_id.to_string(),
        token: token.to_string(),
        token_kind: token_kind.to_string(),
        source_token: source_token.to_string(),
        motif_instance_id,
        motif_token,
        applied_resname: applied_resname
            .map(|value| {
                value
                    .chars()
                    .filter(|ch| ch.is_ascii_alphanumeric())
                    .take(3)
                    .collect::<String>()
                    .to_ascii_uppercase()
            })
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| {
                token
                    .chars()
                    .filter(|ch| ch.is_ascii_alphanumeric())
                    .take(3)
                    .collect::<String>()
                    .to_ascii_uppercase()
            }),
        template_resname,
        branch_depth,
        branch_path,
    })
}

fn resolve_named_junction<'a>(
    bundle: &'a SourceBundle,
    token: &str,
    binding_name: &str,
    path: &str,
) -> Result<(String, &'a JunctionEntry), ErrorDetail> {
    let unit = bundle.unit_library.get(token).ok_or_else(|| {
        to_error(
            "E_UNKNOWN_TOKEN",
            Some(path.into()),
            format!(
                "sequence token '{}' not present in source unit_library",
                token
            ),
        )
    })?;
    let junction_name = unit.junctions.get(binding_name).ok_or_else(|| {
        to_error(
            "E_SOURCE_SCHEMA",
            Some(path.into()),
            format!(
                "unit '{}' missing required junction binding '{}'",
                token, binding_name
            ),
        )
    })?;
    let junction = bundle.junction_library.get(junction_name).ok_or_else(|| {
        to_error(
            "E_SOURCE_SCHEMA",
            Some(path.into()),
            format!("junction '{}' missing from junction_library", junction_name),
        )
    })?;
    Ok((junction_name.clone(), junction))
}

fn compile_connection(
    bundle: &SourceBundle,
    edge_id: String,
    parent_idx: usize,
    _parent_node_id: &str,
    parent_token: &str,
    parent_binding: &str,
    child_idx: usize,
    _child_node_id: &str,
    child_token: &str,
    child_binding: &str,
    conformer_policy: Option<&ConformerPolicy>,
    edge_override: Option<&EdgeConformerOverride>,
    path: &str,
) -> Result<CompiledBuildEdge, ErrorDetail> {
    let (parent_junction, parent_entry) =
        resolve_named_junction(bundle, parent_token, parent_binding, path)?;
    let (child_junction, child_entry) =
        resolve_named_junction(bundle, child_token, child_binding, path)?;
    let parent_attach_atom = selector_atom_name(&parent_entry.attach_atom)
        .map_err(|err| to_error("E_SOURCE_SCHEMA", Some(path.into()), err.to_string()))?;
    let child_attach_atom = selector_atom_name(&child_entry.attach_atom)
        .map_err(|err| to_error("E_SOURCE_SCHEMA", Some(path.into()), err.to_string()))?;
    let parent_leaving_atoms = parent_entry
        .leaving_atoms
        .iter()
        .map(selector_atom_name)
        .collect::<PackResult<Vec<_>>>()
        .map_err(|err| to_error("E_SOURCE_SCHEMA", Some(path.into()), err.to_string()))?;
    let child_leaving_atoms = child_entry
        .leaving_atoms
        .iter()
        .map(selector_atom_name)
        .collect::<PackResult<Vec<_>>>()
        .map_err(|err| to_error("E_SOURCE_SCHEMA", Some(path.into()), err.to_string()))?;
    Ok(CompiledBuildEdge {
        edge_id,
        parent: parent_idx,
        child: child_idx,
        parent_port: parent_binding.to_string(),
        child_port: child_binding.to_string(),
        parent_junction,
        child_junction,
        parent_attach_atom,
        parent_leaving_atoms,
        child_attach_atom,
        child_leaving_atoms,
        bond_order: parent_entry.bond_order.max(child_entry.bond_order),
        layout_mode: conformer_policy
            .map(|policy| policy.layout_mode.clone())
            .unwrap_or_else(default_layout_mode),
        branch_spread: conformer_policy
            .map(|policy| policy.branch_spread.clone())
            .unwrap_or_else(default_branch_spread),
        torsion_mode: edge_override
            .map(|item| item.torsion_mode.clone())
            .unwrap_or_else(|| {
                conformer_policy
                    .map(|policy| policy.default_torsion.clone())
                    .unwrap_or_else(default_torsion_mode)
            }),
        torsion_deg: edge_override
            .and_then(|item| item.torsion_deg)
            .or_else(|| conformer_policy.and_then(|policy| policy.default_torsion_deg)),
        torsion_window_deg: edge_override
            .and_then(|item| item.torsion_window_deg)
            .or_else(|| conformer_policy.and_then(|policy| policy.torsion_window_deg)),
        ring_mode: edge_override
            .and_then(|item| item.ring_mode.clone())
            .or_else(|| conformer_policy.map(|policy| policy.ring_mode.clone())),
    })
}

fn compile_linear_plan(
    req: &BuildRequest,
    bundle: &SourceBundle,
    compiled_sequence: &[String],
) -> Result<CompiledBuildPlan, ErrorDetail> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut previous: Option<TokenExpansion> = None;
    let mut first_expansion: Option<TokenExpansion> = None;
    let policy = effective_policy(req);
    for (idx, token) in compiled_sequence.iter().enumerate() {
        let request_node_id = format!("linear.{}", idx + 1);
        let expansion = append_token_expansion(
            bundle,
            token,
            &request_node_id,
            &request_node_id,
            &request_node_id,
            None,
            &mut nodes,
            &mut edges,
            policy,
        )?;
        if let Some(prev) = previous.as_ref() {
            connect_token_expansions(
                bundle,
                &nodes,
                &mut edges,
                prev,
                "tail",
                &expansion,
                "head",
                canonical_edge_id(&prev.request_node_id, "tail", &request_node_id, "head", idx),
                policy,
                None,
                &format!("/target/sequence/{idx}"),
            )?;
        }
        if first_expansion.is_none() {
            first_expansion = Some(expansion.clone());
        }
        previous = Some(expansion);
    }
    let first = first_expansion.as_ref().ok_or_else(|| {
        to_error(
            "E_INVALID_TARGET",
            Some("/target/sequence".into()),
            "target expands to an empty sequence",
        )
    })?;
    let root_request_node_id = "linear.1".to_string();
    let root_expanded_node_id = nodes[first.root_node_idx].node_id.clone();
    finalize_plan_metrics(CompiledBuildPlan {
        nodes,
        edges,
        applied_caps: Vec::new(),
        request_root_node_id: root_request_node_id,
        expanded_root_node_id: root_expanded_node_id,
        root_token: compiled_sequence.first().cloned().unwrap_or_default(),
        arm_count: 0,
        max_branch_depth: 0,
        graph_has_cycle: false,
    })
}

fn compile_star_plan(
    target: &BuildTarget,
    bundle: &SourceBundle,
    conformer_policy: Option<&ConformerPolicy>,
) -> Result<CompiledBuildPlan, ErrorDetail> {
    let core_token = target
        .core_token
        .as_deref()
        .unwrap_or("")
        .trim()
        .to_string();
    if core_token.is_empty() {
        return Err(to_error(
            "E_INVALID_TARGET",
            Some("/target/core_token".into()),
            "star_polymer requires core_token",
        ));
    }
    let core_junctions = target.core_junctions.clone().unwrap_or_default();
    if core_junctions.is_empty() {
        return Err(to_error(
            "E_INVALID_TARGET",
            Some("/target/core_junctions".into()),
            "star_polymer requires core_junctions",
        ));
    }
    let mut unique = BTreeSet::new();
    for junction in &core_junctions {
        if !unique.insert(junction.clone()) {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some("/target/core_junctions".into()),
                format!("duplicate core junction '{}'", junction),
            ));
        }
    }
    let arm_sequence = target.arm_sequence.clone().unwrap_or_default();
    if arm_sequence.is_empty() {
        return Err(to_error(
            "E_INVALID_TARGET",
            Some("/target/arm_sequence".into()),
            "star_polymer requires arm_sequence",
        ));
    }
    let arm_repeat_count = target.arm_repeat_count.unwrap_or(0);
    if arm_repeat_count == 0 {
        return Err(to_error(
            "E_INVALID_TARGET",
            Some("/target/arm_repeat_count".into()),
            "arm_repeat_count must be >= 1",
        ));
    }
    let mut expanded_arm = Vec::with_capacity(arm_sequence.len() * arm_repeat_count);
    for _ in 0..arm_repeat_count {
        expanded_arm.extend(arm_sequence.iter().cloned());
    }
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let core = append_token_expansion(
        bundle,
        &core_token,
        "root",
        "root",
        "root",
        None,
        &mut nodes,
        &mut edges,
        conformer_policy,
    )?;
    for (arm_idx, junction_name) in core_junctions.iter().enumerate() {
        let mut previous = core.clone();
        for (seq_idx, token) in expanded_arm.iter().enumerate() {
            let request_node_id = format!("arm{}.{}", arm_idx + 1, seq_idx + 1);
            let expansion = append_token_expansion(
                bundle,
                token,
                &request_node_id,
                &request_node_id,
                &request_node_id,
                None,
                &mut nodes,
                &mut edges,
                conformer_policy,
            )?;
            let edge_ordinal = edges.len() + 1;
            connect_token_expansions(
                bundle,
                &nodes,
                &mut edges,
                &previous,
                if seq_idx == 0 { junction_name } else { "tail" },
                &expansion,
                "head",
                canonical_edge_id(
                    &previous.request_node_id,
                    if seq_idx == 0 { junction_name } else { "tail" },
                    &request_node_id,
                    "head",
                    edge_ordinal,
                ),
                conformer_policy,
                None,
                &format!("/target/core_junctions/{arm_idx}"),
            )?;
            previous = expansion;
        }
    }
    let expanded_root_node_id = nodes[core.root_node_idx].node_id.clone();
    finalize_plan_metrics(apply_branched_caps(
        target,
        bundle,
        conformer_policy,
        CompiledBuildPlan {
            nodes,
            edges,
            applied_caps: Vec::new(),
            request_root_node_id: "root".into(),
            expanded_root_node_id,
            root_token: core_token,
            arm_count: 0,
            max_branch_depth: 0,
            graph_has_cycle: false,
        },
    )?)
}

fn compile_branch_children(
    bundle: &SourceBundle,
    parent: &TokenExpansion,
    children: &[BranchChild],
    path_prefix: &str,
    nodes: &mut Vec<CompiledBuildNode>,
    edges: &mut Vec<CompiledBuildEdge>,
    conformer_policy: Option<&ConformerPolicy>,
) -> Result<(), ErrorDetail> {
    let mut used_parent_junctions = BTreeSet::new();
    for (child_idx, child) in children.iter().enumerate() {
        let path = format!("{path_prefix}.children[{child_idx}]");
        if child.parent_junction.trim().is_empty() {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some(format!("{path}.parent_junction")),
                "parent_junction is required",
            ));
        }
        if child.child_junction.trim().is_empty() {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some(format!("{path}.child_junction")),
                "child_junction is required",
            ));
        }
        if !used_parent_junctions.insert(child.parent_junction.clone()) {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some(format!("{path}.parent_junction")),
                format!(
                    "junction '{}' reused on the same node",
                    child.parent_junction
                ),
            ));
        }
        if child.sequence.is_empty() {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some(format!("{path}.sequence")),
                "branch child sequence must be non-empty",
            ));
        }
        if child.repeat_count == 0 {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some(format!("{path}.repeat_count")),
                "repeat_count must be >= 1",
            ));
        }
        let mut expanded = Vec::with_capacity(child.sequence.len() * child.repeat_count);
        for _ in 0..child.repeat_count {
            expanded.extend(child.sequence.iter().cloned());
        }
        let mut previous = parent.clone();
        for (seq_idx, token) in expanded.iter().enumerate() {
            let request_node_id = format!("{path}.{}", seq_idx + 1);
            let expansion = append_token_expansion(
                bundle,
                token,
                &request_node_id,
                &request_node_id,
                &request_node_id,
                None,
                nodes,
                edges,
                conformer_policy,
            )?;
            let edge_ordinal = edges.len() + 1;
            connect_token_expansions(
                bundle,
                nodes,
                edges,
                &previous,
                if seq_idx == 0 {
                    child.parent_junction.as_str()
                } else {
                    "tail"
                },
                &expansion,
                if seq_idx == 0 {
                    child.child_junction.as_str()
                } else {
                    "head"
                },
                canonical_edge_id(
                    &previous.request_node_id,
                    if seq_idx == 0 {
                        child.parent_junction.as_str()
                    } else {
                        "tail"
                    },
                    &request_node_id,
                    if seq_idx == 0 {
                        child.child_junction.as_str()
                    } else {
                        "head"
                    },
                    edge_ordinal,
                ),
                conformer_policy,
                None,
                &path,
            )?;
            previous = expansion;
        }
        if let Some(nested) = child.child.as_deref() {
            let nested_request_node_id = format!("{path}.branch");
            let nested_root = append_token_expansion(
                bundle,
                &nested.token,
                &nested_request_node_id,
                &nested_request_node_id,
                &nested_request_node_id,
                None,
                nodes,
                edges,
                conformer_policy,
            )?;
            let edge_ordinal = edges.len() + 1;
            connect_token_expansions(
                bundle,
                nodes,
                edges,
                &previous,
                "tail",
                &nested_root,
                "head",
                canonical_edge_id(
                    &previous.request_node_id,
                    "tail",
                    &nested_request_node_id,
                    "head",
                    edge_ordinal,
                ),
                conformer_policy,
                None,
                &nested_request_node_id,
            )?;
            compile_branch_children(
                bundle,
                &nested_root,
                &nested.children,
                &nested_request_node_id,
                nodes,
                edges,
                conformer_policy,
            )?;
        }
    }
    Ok(())
}

fn compile_branched_plan(
    target: &BuildTarget,
    bundle: &SourceBundle,
    conformer_policy: Option<&ConformerPolicy>,
) -> Result<CompiledBuildPlan, ErrorDetail> {
    let root = target.branch_tree.as_ref().ok_or_else(|| {
        to_error(
            "E_INVALID_TARGET",
            Some("/target/branch_tree".into()),
            "branched_polymer requires branch_tree",
        )
    })?;
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let root_expansion = append_token_expansion(
        bundle,
        &root.token,
        "root",
        "root",
        "root",
        None,
        &mut nodes,
        &mut edges,
        conformer_policy,
    )?;
    compile_branch_children(
        bundle,
        &root_expansion,
        &root.children,
        "target.branch_tree",
        &mut nodes,
        &mut edges,
        conformer_policy,
    )?;
    let expanded_root_node_id = nodes[root_expansion.root_node_idx].node_id.clone();
    finalize_plan_metrics(apply_branched_caps(
        target,
        bundle,
        conformer_policy,
        CompiledBuildPlan {
            nodes,
            edges,
            applied_caps: Vec::new(),
            request_root_node_id: "root".into(),
            expanded_root_node_id,
            root_token: root.token.clone(),
            arm_count: 0,
            max_branch_depth: 0,
            graph_has_cycle: false,
        },
    )?)
}

fn compile_graph_plan(
    req: &BuildRequest,
    bundle: &SourceBundle,
) -> Result<CompiledBuildPlan, ErrorDetail> {
    let target = &req.target;
    let graph_nodes = target.graph_nodes.as_ref().ok_or_else(|| {
        to_error(
            "E_INVALID_TARGET",
            Some("/target/graph_nodes".into()),
            "polymer_graph requires graph_nodes",
        )
    })?;
    if graph_nodes.is_empty() {
        return Err(to_error(
            "E_INVALID_TARGET",
            Some("/target/graph_nodes".into()),
            "graph_nodes must be non-empty",
        ));
    }
    let root_id = target
        .graph_root
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| graph_nodes[0].id.as_str())
        .to_string();
    let mut top_level_index_by_id = BTreeMap::new();
    for (idx, node) in graph_nodes.iter().enumerate() {
        if node.id.trim().is_empty() {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some(format!("/target/graph_nodes/{idx}/id")),
                "graph node id must be non-empty",
            ));
        }
        if top_level_index_by_id.insert(node.id.clone(), idx).is_some() {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some(format!("/target/graph_nodes/{idx}/id")),
                format!("duplicate graph node id '{}'", node.id),
            ));
        }
    }
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut expansions = BTreeMap::new();
    let policy = effective_policy(req);
    for node in graph_nodes {
        let expansion = append_token_expansion(
            bundle,
            &node.token,
            &node.id,
            &format!("graph.{}", node.id),
            &format!("graph.{}", node.id),
            node.applied_resname.as_deref(),
            &mut nodes,
            &mut edges,
            policy,
        )?;
        expansions.insert(node.id.clone(), expansion);
    }
    let graph_edges = target.graph_edges.clone().unwrap_or_default();
    let mut used_ports = BTreeMap::<String, BTreeSet<String>>::new();
    let mut seen_edges = BTreeSet::new();
    for (edge_idx, edge) in graph_edges.iter().enumerate() {
        if edge.from == edge.to {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some(format!("/target/graph_edges/{edge_idx}")),
                "graph edge may not connect a node to itself",
            ));
        }
        let left = expansions.get(&edge.from).ok_or_else(|| {
            to_error(
                "E_INVALID_TARGET",
                Some(format!("/target/graph_edges/{edge_idx}/from")),
                format!("graph edge references unknown node '{}'", edge.from),
            )
        })?;
        let right = expansions.get(&edge.to).ok_or_else(|| {
            to_error(
                "E_INVALID_TARGET",
                Some(format!("/target/graph_edges/{edge_idx}/to")),
                format!("graph edge references unknown node '{}'", edge.to),
            )
        })?;
        if !used_ports
            .entry(edge.from.clone())
            .or_default()
            .insert(edge.from_junction.clone())
        {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some(format!("/target/graph_edges/{edge_idx}/from_junction")),
                format!(
                    "junction '{}' reused on graph node '{}'",
                    edge.from_junction, edge.from
                ),
            ));
        }
        if !used_ports
            .entry(edge.to.clone())
            .or_default()
            .insert(edge.to_junction.clone())
        {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some(format!("/target/graph_edges/{edge_idx}/to_junction")),
                format!(
                    "junction '{}' reused on graph node '{}'",
                    edge.to_junction, edge.to
                ),
            ));
        }
        let dedupe = if edge.from <= edge.to {
            (
                edge.from.clone(),
                edge.to.clone(),
                edge.from_junction.clone(),
                edge.to_junction.clone(),
            )
        } else {
            (
                edge.to.clone(),
                edge.from.clone(),
                edge.to_junction.clone(),
                edge.from_junction.clone(),
            )
        };
        if !seen_edges.insert(dedupe) {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some(format!("/target/graph_edges/{edge_idx}")),
                "duplicate graph edge",
            ));
        }
        let resolved_edge_id = edge.id.clone().unwrap_or_else(|| {
            canonical_edge_id(
                &edge.from,
                &edge.from_junction,
                &edge.to,
                &edge.to_junction,
                edge_idx + 1,
            )
        });
        let override_ref = req.conformer_policy.as_ref().and_then(|policy| {
            policy
                .edge_overrides
                .iter()
                .find(|item| item.edge_id == resolved_edge_id)
        });
        connect_token_expansions(
            bundle,
            &nodes,
            &mut edges,
            left,
            &edge.from_junction,
            right,
            &edge.to_junction,
            resolved_edge_id.clone(),
            policy,
            override_ref,
            &format!("/target/graph_edges/{edge_idx}"),
        )?;
        if let Some(last) = edges.last_mut() {
            last.bond_order = edge.bond_order.max(last.bond_order);
        }
    }
    let root_expansion = expansions.get(&root_id).ok_or_else(|| {
        to_error(
            "E_INVALID_TARGET",
            Some("/target/graph_root".into()),
            format!("graph_root '{}' not present in graph_nodes", root_id),
        )
    })?;
    let expanded_root_node_id = nodes[root_expansion.root_node_idx].node_id.clone();
    finalize_plan_metrics(CompiledBuildPlan {
        nodes,
        edges,
        applied_caps: Vec::new(),
        request_root_node_id: root_id.clone(),
        expanded_root_node_id,
        root_token: graph_nodes[top_level_index_by_id[&root_id]].token.clone(),
        arm_count: 0,
        max_branch_depth: 0,
        graph_has_cycle: false,
    })
}

fn compile_build_plan(
    req: &BuildRequest,
    bundle: &SourceBundle,
    seed: u64,
) -> Result<Option<CompiledBuildPlan>, ErrorDetail> {
    let plan = match req.target.mode.as_str() {
        "star_polymer" => Ok(Some(compile_star_plan(
            &req.target,
            bundle,
            effective_policy(req),
        )?)),
        "branched_polymer" => Ok(Some(compile_branched_plan(
            &req.target,
            bundle,
            effective_policy(req),
        )?)),
        "polymer_graph" => Ok(Some(compile_graph_plan(req, bundle)?)),
        _ => {
            let expanded_sequence = expand_sequence(&req.target, seed)?;
            if expanded_sequence.is_empty() {
                return Ok(None);
            }
            let compiled_sequence = compile_sequence_tokens(req, bundle, &expanded_sequence)?;
            Ok(Some(compile_linear_plan(req, bundle, &compiled_sequence)?))
        }
    }?;
    let Some(plan) = plan else {
        return Ok(None);
    };
    Ok(Some(finalize_plan_metrics(apply_motif_port_caps(
        req,
        bundle,
        effective_policy(req),
        plan,
    )?)?))
}

fn apply_motif_port_caps(
    req: &BuildRequest,
    bundle: &SourceBundle,
    conformer_policy: Option<&ConformerPolicy>,
    mut plan: CompiledBuildPlan,
) -> Result<CompiledBuildPlan, ErrorDetail> {
    if plan.nodes.is_empty() {
        return Ok(plan);
    }
    let mut used_ports = vec![BTreeSet::<String>::new(); plan.nodes.len()];
    for edge in &plan.edges {
        if let Some(items) = used_ports.get_mut(edge.parent) {
            items.insert(edge.parent_port.clone());
        }
        if let Some(items) = used_ports.get_mut(edge.child) {
            items.insert(edge.child_port.clone());
        }
    }
    let mut overrides = BTreeMap::<(String, String), (CapBinding, String)>::new();
    for (idx, item) in req.port_cap_overrides.iter().enumerate() {
        let key = (item.node_id.clone(), item.port.clone());
        if overrides.contains_key(&key) {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some(format!("port_cap_overrides[{idx}]")),
                format!(
                    "duplicate port cap override for node '{}' port '{}'",
                    item.node_id, item.port
                ),
            ));
        }
        overrides.insert(
            key,
            (item.cap.clone(), format!("port_cap_overrides[{idx}].cap")),
        );
    }

    let existing_instances = plan
        .nodes
        .iter()
        .filter_map(|node| {
            node.motif_token
                .as_ref()
                .map(|motif_token| (node.request_node_id.clone(), motif_token.clone()))
        })
        .collect::<BTreeSet<_>>();
    for (request_node_id, motif_token) in existing_instances {
        let Some(motif) = bundle.motif_library.get(&motif_token) else {
            continue;
        };
        let local_node_map = plan
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| {
                node.request_node_id == request_node_id
                    && node.motif_token.as_deref() == Some(motif_token.as_str())
            })
            .map(|(idx, node)| {
                (
                    node.node_id
                        .split("::")
                        .last()
                        .unwrap_or(node.node_id.as_str())
                        .to_string(),
                    (idx, node.node_id.clone()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for (port_name, port_spec) in &motif.exposed_ports {
            let Some((node_idx, compiled_node_id)) =
                local_node_map.get(&port_spec.node_id).cloned()
            else {
                return Err(to_error(
                    "E_RUNTIME_BUILD",
                    None,
                    format!(
                        "compiled motif port '{}' for node '{}' missing from expanded graph",
                        port_name, port_spec.node_id
                    ),
                ));
            };
            if used_ports[node_idx].contains(&port_spec.junction) {
                if overrides.contains_key(&(request_node_id.clone(), port_name.clone())) {
                    return Err(to_error(
                        "E_INVALID_TARGET",
                        Some(format!(
                            "port_cap_overrides.{}.{}",
                            request_node_id, port_name
                        )),
                        format!(
                            "port '{}' on node '{}' is already bound by the graph",
                            port_name, request_node_id
                        ),
                    ));
                }
                continue;
            }
            let (cap_binding, source_path, application_source) = if let Some((binding, path)) =
                overrides.get(&(request_node_id.clone(), port_name.clone()))
            {
                (
                    binding.clone(),
                    path.clone(),
                    "request_override".to_string(),
                )
            } else if let Some(binding) = port_spec.default_cap.clone() {
                (
                    binding,
                    format!("motif_library.{motif_token}.exposed_ports.{port_name}.default_cap"),
                    "bundle_default".to_string(),
                )
            } else {
                continue;
            };
            if !cap_binding_allowed(&cap_binding, &port_spec.allowed_caps) {
                return Err(to_error(
                    "E_INVALID_TARGET",
                    Some(source_path.clone()),
                    format!(
                        "cap '{}' not allowed on motif '{}' port '{}'",
                        cap_binding.token, motif_token, port_name
                    ),
                ));
            }
            let cap_port = resolve_cap_binding(bundle, &cap_binding, &source_path)?;
            let cap_request_node_id = format!("{request_node_id}::{port_name}::cap");
            let cap_expansion = append_token_expansion(
                bundle,
                &cap_binding.token,
                &cap_request_node_id,
                &cap_request_node_id,
                &format!("{}.cap.{}", request_node_id, port_name),
                None,
                &mut plan.nodes,
                &mut plan.edges,
                conformer_policy,
            )?;
            let mut exposed_ports = BTreeMap::new();
            exposed_ports.insert(
                port_name.clone(),
                ResolvedTokenPort {
                    node_idx,
                    node_id: compiled_node_id.clone(),
                    binding: port_spec.junction.clone(),
                },
            );
            let source_expansion = TokenExpansion {
                root_node_idx: node_idx,
                request_node_id: request_node_id.clone(),
                source_token: motif_token.clone(),
                exposed_ports,
            };
            let cap_edge_id = canonical_edge_id(
                &compiled_node_id,
                port_name,
                &plan.nodes[cap_expansion.root_node_idx].node_id,
                &cap_port,
                plan.edges.len() + 1,
            );
            connect_token_expansions(
                bundle,
                &plan.nodes,
                &mut plan.edges,
                &source_expansion,
                port_name,
                &cap_expansion,
                &cap_port,
                cap_edge_id,
                conformer_policy,
                None,
                &source_path,
            )?;
            used_ports[node_idx].insert(port_spec.junction.clone());
            plan.applied_caps.push(CompiledAppliedCap {
                node_id: compiled_node_id,
                request_node_id: request_node_id.clone(),
                resid: node_idx + 1,
                port_name: port_name.clone(),
                junction: port_spec.junction.clone(),
                cap: cap_binding.clone(),
                application_source,
                cap_node_id: plan.nodes[cap_expansion.root_node_idx].node_id.clone(),
                cap_resid: cap_expansion.root_node_idx + 1,
            });
        }
    }

    Ok(plan)
}

fn selector_atom_name(selector: &JunctionSelector) -> PackResult<String> {
    if selector.scope.trim() != "unit" {
        return Err(PackError::Invalid(format!(
            "unsupported junction selector scope '{}'; only 'unit' is executable",
            selector.scope
        )));
    }
    let raw = selector.selector.trim();
    let lower = raw.to_ascii_lowercase();
    if let Some(name) = lower.strip_prefix("name ") {
        let offset = raw.len() - name.len();
        return Ok(raw[offset..].trim().to_string());
    }
    Err(PackError::Invalid(format!(
        "unsupported junction selector '{}'; only 'name <atom>' is executable",
        selector.selector
    )))
}

fn token_junction_specs(
    bundle: &SourceBundle,
    sequence: &[String],
) -> PackResult<BTreeMap<String, TokenJunctionSpec>> {
    let mut specs = BTreeMap::new();
    for token in sequence {
        if specs.contains_key(token) {
            continue;
        }
        let unit = bundle.unit_library.get(token).ok_or_else(|| {
            PackError::Invalid(format!(
                "sequence token '{}' missing from unit_library",
                token
            ))
        })?;
        let head = unit
            .junctions
            .get("head")
            .map(|name| {
                bundle.junction_library.get(name).ok_or_else(|| {
                    PackError::Invalid(format!("junction '{}' missing from junction_library", name))
                })
            })
            .transpose()?;
        let tail = unit
            .junctions
            .get("tail")
            .map(|name| {
                bundle.junction_library.get(name).ok_or_else(|| {
                    PackError::Invalid(format!("junction '{}' missing from junction_library", name))
                })
            })
            .transpose()?;
        specs.insert(
            token.clone(),
            TokenJunctionSpec {
                head_attach_atom: head
                    .map(|junction| selector_atom_name(&junction.attach_atom))
                    .transpose()?,
                head_leaving_atoms: head
                    .map(|junction| {
                        junction
                            .leaving_atoms
                            .iter()
                            .map(selector_atom_name)
                            .collect::<PackResult<Vec<_>>>()
                    })
                    .transpose()?
                    .unwrap_or_default(),
                tail_attach_atom: tail
                    .map(|junction| selector_atom_name(&junction.attach_atom))
                    .transpose()?,
                tail_leaving_atoms: tail
                    .map(|junction| {
                        junction
                            .leaving_atoms
                            .iter()
                            .map(selector_atom_name)
                            .collect::<PackResult<Vec<_>>>()
                    })
                    .transpose()?
                    .unwrap_or_default(),
            },
        );
    }
    Ok(specs)
}

fn base_conformation_mode(mode: &str) -> Result<&str, ErrorDetail> {
    match mode {
        "extended" => Ok("extended"),
        "random_walk" => Ok("random_walk"),
        "aligned" => Ok("extended"),
        "ensemble" => Ok("random_walk"),
        other => Err(to_error(
            "E_UNSUPPORTED_REALIZATION",
            Some("/realization/conformation_mode".into()),
            format!("unsupported conformation mode '{}'", other),
        )),
    }
}

fn validate_conformer_policy(
    req: &BuildRequest,
    compiled_plan: Option<&CompiledBuildPlan>,
) -> Vec<ErrorDetail> {
    let Some(policy) = req.conformer_policy.as_ref() else {
        return Vec::new();
    };
    let mut errors = Vec::new();
    let allowed_layout = ["auto", "tree_radial", "cycle_planar", "mixed"];
    let allowed_torsion = [
        "trans",
        "cis",
        "gauche_plus",
        "gauche_minus",
        "fixed_deg",
        "sample_window",
    ];
    let allowed_branch_spread = ["even", "staggered"];
    let allowed_ring = ["auto", "planar", "puckered"];
    if !allowed_layout.contains(&policy.layout_mode.as_str()) {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("/conformer_policy/layout_mode".into()),
            format!("unsupported layout_mode '{}'", policy.layout_mode),
        ));
    }
    if !allowed_torsion.contains(&policy.default_torsion.as_str()) {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("/conformer_policy/default_torsion".into()),
            format!("unsupported default_torsion '{}'", policy.default_torsion),
        ));
    }
    if !allowed_branch_spread.contains(&policy.branch_spread.as_str()) {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("/conformer_policy/branch_spread".into()),
            format!("unsupported branch_spread '{}'", policy.branch_spread),
        ));
    }
    if !allowed_ring.contains(&policy.ring_mode.as_str()) {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("/conformer_policy/ring_mode".into()),
            format!("unsupported ring_mode '{}'", policy.ring_mode),
        ));
    }
    if policy.default_torsion == "fixed_deg" && policy.default_torsion_deg.is_none() {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("/conformer_policy/default_torsion_deg".into()),
            "fixed_deg default_torsion requires default_torsion_deg",
        ));
    }
    if policy.default_torsion == "sample_window" && policy.torsion_window_deg.is_none() {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("/conformer_policy/torsion_window_deg".into()),
            "sample_window default_torsion requires torsion_window_deg",
        ));
    }
    if !policy.edge_overrides.is_empty() && req.target.mode != "polymer_graph" {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("/conformer_policy/edge_overrides".into()),
            "edge_overrides are only supported for polymer_graph requests",
        ));
    }
    let known_edge_ids = compiled_plan
        .map(|plan| {
            plan.edges
                .iter()
                .map(|edge| edge.edge_id.clone())
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    for (idx, edge) in policy.edge_overrides.iter().enumerate() {
        if !allowed_torsion.contains(&edge.torsion_mode.as_str()) {
            errors.push(to_error(
                "E_INVALID_TARGET",
                Some(format!(
                    "conformer_policy.edge_overrides[{idx}].torsion_mode"
                )),
                format!("unsupported torsion_mode '{}'", edge.torsion_mode),
            ));
        }
        if edge.torsion_mode == "fixed_deg" && edge.torsion_deg.is_none() {
            errors.push(to_error(
                "E_INVALID_TARGET",
                Some(format!(
                    "conformer_policy.edge_overrides[{idx}].torsion_deg"
                )),
                "fixed_deg edge override requires torsion_deg",
            ));
        }
        if edge.torsion_mode == "sample_window" && edge.torsion_window_deg.is_none() {
            errors.push(to_error(
                "E_INVALID_TARGET",
                Some(format!(
                    "conformer_policy.edge_overrides[{idx}].torsion_window_deg"
                )),
                "sample_window edge override requires torsion_window_deg",
            ));
        }
        if !known_edge_ids.is_empty() && !known_edge_ids.contains(&edge.edge_id) {
            errors.push(to_error(
                "E_INVALID_TARGET",
                Some(format!("/conformer_policy/edge_overrides/{idx}/edge_id")),
                format!("edge_id '{}' not present in compiled graph", edge.edge_id),
            ));
        }
    }
    errors
}

fn validate_relax_spec(req: &BuildRequest) -> Vec<ErrorDetail> {
    let mut errors = Vec::new();
    let Some(relax) = req.realization.relax.as_ref() else {
        if req.artifacts.raw_coordinates.is_some() {
            errors.push(to_error(
                "E_INVALID_TARGET",
                Some("/artifacts/raw_coordinates".into()),
                "raw_coordinates is only valid when realization.relax is set",
            ));
        }
        return errors;
    };
    if canonical_relax_mode(&relax.mode).is_none() {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("/realization/relax/mode".into()),
            format!("unsupported relax mode '{}'", relax.mode),
        ));
    }
    if relax.steps.unwrap_or(64) == 0 {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("/realization/relax/steps".into()),
            "relax.steps must be >= 1",
        ));
    }
    if relax.step_scale.unwrap_or(0.25) <= 0.0 {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("/realization/relax/step_scale".into()),
            "relax.step_scale must be > 0",
        ));
    }
    if relax.clash_scale.unwrap_or(0.9) <= 0.0 {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("/realization/relax/clash_scale".into()),
            "relax.clash_scale must be > 0",
        ));
    }
    errors
}

fn validate_port_cap_overrides(req: &BuildRequest, bundle: &SourceBundle) -> Vec<ErrorDetail> {
    let mut errors = Vec::new();
    if req.port_cap_overrides.is_empty() {
        return errors;
    }
    if req.target.mode != "polymer_graph" {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("port_cap_overrides".into()),
            "port_cap_overrides are only supported for polymer_graph requests",
        ));
        return errors;
    }
    let graph_nodes = req.target.graph_nodes.as_ref().cloned().unwrap_or_default();
    let graph_edges = req.target.graph_edges.as_ref().cloned().unwrap_or_default();
    let node_by_id = graph_nodes
        .iter()
        .map(|node| (node.id.clone(), node))
        .collect::<BTreeMap<_, _>>();
    let mut claimed_ports = BTreeSet::<(String, String)>::new();
    for edge in &graph_edges {
        claimed_ports.insert((edge.from.clone(), edge.from_junction.clone()));
        claimed_ports.insert((edge.to.clone(), edge.to_junction.clone()));
    }
    let mut seen = BTreeSet::new();
    for (idx, item) in req.port_cap_overrides.iter().enumerate() {
        let key = (item.node_id.clone(), item.port.clone());
        if !seen.insert(key.clone()) {
            errors.push(to_error(
                "E_INVALID_TARGET",
                Some(format!("port_cap_overrides[{idx}]")),
                format!(
                    "duplicate port cap override for node '{}' port '{}'",
                    item.node_id, item.port
                ),
            ));
            continue;
        }
        let Some(node) = node_by_id.get(&item.node_id) else {
            errors.push(to_error(
                "E_INVALID_TARGET",
                Some(format!("port_cap_overrides[{idx}].node_id")),
                format!(
                    "override node '{}' not present in graph_nodes",
                    item.node_id
                ),
            ));
            continue;
        };
        let Some(motif) = bundle.motif_library.get(&node.token) else {
            errors.push(to_error(
                "E_INVALID_TARGET",
                Some(format!("port_cap_overrides[{idx}].node_id")),
                format!(
                    "graph node '{}' does not reference a motif token",
                    item.node_id
                ),
            ));
            continue;
        };
        let Some(port_spec) = motif.exposed_ports.get(&item.port) else {
            errors.push(to_error(
                "E_INVALID_TARGET",
                Some(format!("port_cap_overrides[{idx}].port")),
                format!(
                    "motif token '{}' does not expose port '{}'",
                    node.token, item.port
                ),
            ));
            continue;
        };
        if claimed_ports.contains(&(item.node_id.clone(), item.port.clone()))
            || claimed_ports.contains(&(item.node_id.clone(), port_spec.junction.clone()))
        {
            errors.push(to_error(
                "E_INVALID_TARGET",
                Some(format!("port_cap_overrides[{idx}]")),
                format!(
                    "port '{}' on node '{}' is already consumed by graph_edges",
                    item.port, item.node_id
                ),
            ));
        }
        if let Err(err) =
            resolve_cap_binding(bundle, &item.cap, &format!("port_cap_overrides[{idx}].cap"))
        {
            errors.push(err);
        }
        if !cap_binding_allowed(&item.cap, &port_spec.allowed_caps) {
            errors.push(to_error(
                "E_INVALID_TARGET",
                Some(format!("port_cap_overrides[{idx}].cap")),
                format!(
                    "cap '{}' not allowed on node '{}' port '{}'",
                    item.cap.token, item.node_id, item.port
                ),
            ));
        }
    }
    errors
}

fn validate_motif_library(bundle: &SourceBundle) -> Vec<ErrorDetail> {
    let mut errors = Vec::new();
    for token in bundle.motif_library.keys() {
        if bundle.unit_library.contains_key(token) {
            errors.push(to_error(
                "E_SOURCE_SCHEMA",
                Some(format!("motif_library.{token}")),
                format!(
                    "token '{}' collides across unit_library and motif_library",
                    token
                ),
            ));
        }
    }
    for (token, motif) in &bundle.motif_library {
        if motif.nodes.is_empty() {
            errors.push(to_error(
                "E_SOURCE_SCHEMA",
                Some(format!("motif_library.{token}.nodes")),
                "motif must contain at least one node",
            ));
            continue;
        }
        let mut index_by_id = BTreeMap::new();
        for (idx, node) in motif.nodes.iter().enumerate() {
            if node.id.trim().is_empty() {
                errors.push(to_error(
                    "E_SOURCE_SCHEMA",
                    Some(format!("motif_library.{token}.nodes[{idx}].id")),
                    "motif node id must be non-empty",
                ));
            }
            if index_by_id.insert(node.id.clone(), idx).is_some() {
                errors.push(to_error(
                    "E_SOURCE_SCHEMA",
                    Some(format!("motif_library.{token}.nodes[{idx}].id")),
                    format!("duplicate motif node id '{}'", node.id),
                ));
            }
            if bundle.motif_library.contains_key(&node.token) {
                errors.push(to_error(
                    "E_SOURCE_SCHEMA",
                    Some(format!("motif_library.{token}.nodes[{idx}].token")),
                    "motif nodes may reference unit_library tokens only",
                ));
            }
            if !bundle.unit_library.contains_key(&node.token) {
                errors.push(to_error(
                    "E_SOURCE_SCHEMA",
                    Some(format!("motif_library.{token}.nodes[{idx}].token")),
                    format!(
                        "motif node token '{}' missing from unit_library",
                        node.token
                    ),
                ));
            }
        }
        let root_node_id = motif
            .root_node_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| motif.nodes[0].id.as_str());
        if !index_by_id.contains_key(root_node_id) {
            errors.push(to_error(
                "E_SOURCE_SCHEMA",
                Some(format!("motif_library.{token}.root_node_id")),
                format!(
                    "motif root_node_id '{}' missing from motif nodes",
                    root_node_id
                ),
            ));
        }
        let mut adjacency = vec![Vec::<usize>::new(); motif.nodes.len()];
        for (edge_idx, edge) in motif.edges.iter().enumerate() {
            let Some(from_idx) = index_by_id.get(&edge.from).copied() else {
                errors.push(to_error(
                    "E_SOURCE_SCHEMA",
                    Some(format!("motif_library.{token}.edges[{edge_idx}].from")),
                    format!("motif edge references unknown node '{}'", edge.from),
                ));
                continue;
            };
            let Some(to_idx) = index_by_id.get(&edge.to).copied() else {
                errors.push(to_error(
                    "E_SOURCE_SCHEMA",
                    Some(format!("motif_library.{token}.edges[{edge_idx}].to")),
                    format!("motif edge references unknown node '{}'", edge.to),
                ));
                continue;
            };
            adjacency[from_idx].push(to_idx);
            adjacency[to_idx].push(from_idx);
        }
        let mut seen = BTreeSet::new();
        let mut queue = std::collections::VecDeque::from([index_by_id[root_node_id]]);
        seen.insert(index_by_id[root_node_id]);
        while let Some(node_idx) = queue.pop_front() {
            for neighbor in adjacency[node_idx].clone() {
                if seen.insert(neighbor) {
                    queue.push_back(neighbor);
                }
            }
        }
        if seen.len() != motif.nodes.len() {
            errors.push(to_error(
                "E_SOURCE_SCHEMA",
                Some(format!("motif_library.{token}.edges")),
                "motif graph must be connected",
            ));
        }
        for (port_name, port) in &motif.exposed_ports {
            let Some(node_idx) = index_by_id.get(&port.node_id).copied() else {
                errors.push(to_error(
                    "E_SOURCE_SCHEMA",
                    Some(format!(
                        "motif_library.{token}.exposed_ports.{port_name}.node_id"
                    )),
                    format!(
                        "motif exposed port references unknown node '{}'",
                        port.node_id
                    ),
                ));
                continue;
            };
            let unit_token = &motif.nodes[node_idx].token;
            if let Some(unit) = bundle.unit_library.get(unit_token) {
                if !unit.junctions.contains_key(&port.junction) {
                    errors.push(to_error(
                        "E_SOURCE_SCHEMA",
                        Some(format!("motif_library.{token}.exposed_ports.{port_name}.junction")),
                        format!(
                            "motif exposed port '{}' references unknown junction binding '{}' on token '{}'",
                            port_name, port.junction, unit_token
                        ),
                    ));
                }
            }
            if let Some(default_cap) = &port.default_cap {
                if let Err(err) = resolve_cap_binding(
                    bundle,
                    default_cap,
                    &format!("motif_library.{token}.exposed_ports.{port_name}.default_cap"),
                ) {
                    errors.push(err);
                }
                if !cap_binding_allowed(default_cap, &port.allowed_caps) {
                    errors.push(to_error(
                        "E_SOURCE_SCHEMA",
                        Some(format!("motif_library.{token}.exposed_ports.{port_name}.default_cap")),
                        "default_cap must also be present in allowed_caps when allowed_caps is non-empty",
                    ));
                }
            }
            for (cap_idx, cap) in port.allowed_caps.iter().enumerate() {
                if let Err(err) = resolve_cap_binding(
                    bundle,
                    cap,
                    &format!(
                        "motif_library.{token}.exposed_ports.{port_name}.allowed_caps[{cap_idx}]"
                    ),
                ) {
                    errors.push(err);
                }
            }
        }
    }
    errors
}

fn validate_request(req: &BuildRequest, bundle: &SourceBundle) -> Vec<ErrorDetail> {
    let mut errors = Vec::new();
    errors.extend(validate_motif_library(bundle));
    errors.extend(validate_relax_spec(req));
    errors.extend(validate_port_cap_overrides(req, bundle));
    if let Err(err) = validation_depth(req) {
        errors.push(err);
    }
    if let Err(err) = validation_cache_mode(req) {
        errors.push(err);
    }
    if !matches!(validation_cache_mode(req), Ok("off")) && req.validation.cache_dir.is_none() {
        errors.push(to_error(
            "E_INVALID_REQUEST",
            Some("/validation/cache_dir".into()),
            "validation.cache_dir is required when validation.cache_mode is not 'off'",
        ));
    }
    if let Err(err) = qc_policy(req) {
        errors.push(err);
    }
    if req.schema_version != BUILD_SCHEMA_VERSION {
        errors.push(to_error(
            "E_CONFIG_VERSION",
            Some("schema_version".into()),
            format!(
                "unsupported schema '{}'; expected {}",
                req.schema_version, BUILD_SCHEMA_VERSION
            ),
        ));
    }
    if req.source_ref.bundle_id != bundle.bundle_id {
        errors.push(to_error(
            "E_SOURCE_ID",
            Some("/source_ref/bundle_id".into()),
            "source_ref.bundle_id does not match bundle_id",
        ));
    }
    if let Some(expected) = req.source_ref.bundle_digest.as_ref() {
        match sha256_file(Path::new(&req.source_ref.bundle_path)) {
            Ok(actual) if &actual != expected => errors.push(to_error(
                "E_SOURCE_DIGEST",
                Some("/source_ref/bundle_digest".into()),
                format!("bundle digest mismatch: expected {expected}, got {actual}"),
            )),
            Err(err) => errors.push(to_error(
                "E_SOURCE_LOAD",
                Some("/source_ref/bundle_path".into()),
                err.to_string(),
            )),
            _ => {}
        }
    }
    if !SUPPORTED_TARGET_MODES.contains(&req.target.mode.as_str()) {
        errors.push(to_error(
            "E_UNSUPPORTED_TARGET",
            Some("/target/mode".into()),
            format!(
                "target mode '{}' unsupported; supported: {}",
                req.target.mode,
                SUPPORTED_TARGET_MODES.join(", ")
            ),
        ));
    }
    if !bundle
        .capabilities
        .supported_target_modes
        .iter()
        .any(|mode| mode == &req.target.mode)
    {
        errors.push(to_error(
            "E_UNSUPPORTED_TARGET",
            Some("/target/mode".into()),
            "source bundle does not advertise requested target mode",
        ));
    }
    match req.target.mode.as_str() {
        "linear_homopolymer" => {
            if req
                .target
                .repeat_unit
                .as_deref()
                .unwrap_or("")
                .trim()
                .is_empty()
            {
                errors.push(to_error(
                    "E_INVALID_TARGET",
                    Some("/target/repeat_unit".into()),
                    "linear_homopolymer requires repeat_unit",
                ));
            }
            if req.target.n_repeat.unwrap_or(0) == 0 {
                errors.push(to_error(
                    "E_INVALID_TARGET",
                    Some("/target/n_repeat".into()),
                    "n_repeat must be >= 1",
                ));
            }
        }
        "linear_sequence_polymer" => {
            if req
                .target
                .sequence
                .as_ref()
                .map(|items| items.is_empty())
                .unwrap_or(true)
            {
                errors.push(to_error(
                    "E_INVALID_TARGET",
                    Some("/target/sequence".into()),
                    "linear_sequence_polymer requires sequence",
                ));
            }
            if bundle.capabilities.sequence_token_support.is_none() {
                errors.push(to_error(
                    "E_UNSUPPORTED_TARGET",
                    Some("/target/sequence".into()),
                    "source bundle does not advertise sequence token support",
                ));
            }
        }
        "block_copolymer" => {
            if req
                .target
                .blocks
                .as_ref()
                .map(|items| items.is_empty())
                .unwrap_or(true)
            {
                errors.push(to_error(
                    "E_INVALID_TARGET",
                    Some("/target/blocks".into()),
                    "block_copolymer requires blocks",
                ));
            }
        }
        "random_copolymer" => {
            if req
                .target
                .composition
                .as_ref()
                .map(|items| items.is_empty())
                .unwrap_or(true)
            {
                errors.push(to_error(
                    "E_INVALID_TARGET",
                    Some("/target/composition".into()),
                    "random_copolymer requires composition",
                ));
            }
        }
        "star_polymer" => {
            if req
                .target
                .core_token
                .as_deref()
                .unwrap_or("")
                .trim()
                .is_empty()
            {
                errors.push(to_error(
                    "E_INVALID_TARGET",
                    Some("/target/core_token".into()),
                    "star_polymer requires core_token",
                ));
            }
            if req
                .target
                .core_junctions
                .as_ref()
                .map(|items| items.is_empty())
                .unwrap_or(true)
            {
                errors.push(to_error(
                    "E_INVALID_TARGET",
                    Some("/target/core_junctions".into()),
                    "star_polymer requires core_junctions",
                ));
            }
            if req
                .target
                .arm_sequence
                .as_ref()
                .map(|items| items.is_empty())
                .unwrap_or(true)
            {
                errors.push(to_error(
                    "E_INVALID_TARGET",
                    Some("/target/arm_sequence".into()),
                    "star_polymer requires arm_sequence",
                ));
            }
            if req.target.arm_repeat_count.unwrap_or(0) == 0 {
                errors.push(to_error(
                    "E_INVALID_TARGET",
                    Some("/target/arm_repeat_count".into()),
                    "arm_repeat_count must be >= 1",
                ));
            }
        }
        "branched_polymer" => {
            if req.target.branch_tree.is_none() {
                errors.push(to_error(
                    "E_INVALID_TARGET",
                    Some("/target/branch_tree".into()),
                    "branched_polymer requires branch_tree",
                ));
            }
        }
        "polymer_graph" => {
            if req
                .target
                .graph_nodes
                .as_ref()
                .map(|items| items.is_empty())
                .unwrap_or(true)
            {
                errors.push(to_error(
                    "E_INVALID_TARGET",
                    Some("/target/graph_nodes".into()),
                    "polymer_graph requires graph_nodes",
                ));
            }
        }
        _ => {}
    }
    let seed = match resolve_seed(req) {
        Ok((seed, _)) => seed,
        Err(error) => {
            errors.push(error);
            1_234_567
        }
    };
    let expanded_sequence = match expand_sequence(&req.target, seed) {
        Ok(sequence) => sequence,
        Err(error) => {
            if req.target.mode != "star_polymer"
                && req.target.mode != "branched_polymer"
                && req.target.mode != "polymer_graph"
            {
                errors.push(error);
            }
            Vec::new()
        }
    };
    let compiled_plan = match compile_build_plan(req, bundle, seed) {
        Ok(plan) => plan,
        Err(error) => {
            errors.push(error);
            None
        }
    };
    errors.extend(validate_conformer_policy(req, compiled_plan.as_ref()));
    let compiled_sequence = compiled_plan
        .as_ref()
        .map(|plan| {
            plan.nodes
                .iter()
                .map(|node| node.token.clone())
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| expanded_sequence.clone());
    if compiled_sequence.is_empty() {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("target".into()),
            "target expands to an empty sequence",
        ));
    }
    for (idx, token) in compiled_sequence.iter().enumerate() {
        if !bundle.unit_library.contains_key(token) {
            errors.push(to_error(
                "E_UNKNOWN_TOKEN",
                Some(format!("/target/sequence/{idx}")),
                format!(
                    "sequence token '{}' not present in source unit_library",
                    token
                ),
            ));
        }
    }
    if let Some(token_support) = bundle.capabilities.sequence_token_support.as_ref() {
        if !token_support.tokens.is_empty() {
            for (idx, token) in compiled_sequence.iter().enumerate() {
                if !token_support.tokens.iter().any(|item| item == token) {
                    errors.push(to_error(
                        "E_UNKNOWN_TOKEN",
                        Some(format!("/target/sequence/{idx}")),
                        format!(
                            "sequence token '{}' not present in sequence_token_support",
                            token
                        ),
                    ));
                }
            }
        }
        if !token_support.allowed_adjacencies.is_empty() {
            if let Some(plan) = compiled_plan.as_ref().filter(|plan| !plan.edges.is_empty()) {
                for edge in &plan.edges {
                    let pair = [
                        plan.nodes[edge.parent].token.clone(),
                        plan.nodes[edge.child].token.clone(),
                    ];
                    let reverse = [pair[1].clone(), pair[0].clone()];
                    if !token_support
                        .allowed_adjacencies
                        .iter()
                        .any(|item| item == &pair || item == &reverse)
                    {
                        errors.push(to_error(
                            "E_INVALID_TARGET",
                            Some("/target/sequence".into()),
                            format!("unsupported token adjacency '{}-{}'", pair[0], pair[1]),
                        ));
                    }
                }
            } else {
                for window in compiled_sequence.windows(2) {
                    let pair = [window[0].clone(), window[1].clone()];
                    if !token_support
                        .allowed_adjacencies
                        .iter()
                        .any(|item| item == &pair)
                    {
                        errors.push(to_error(
                            "E_INVALID_TARGET",
                            Some("/target/sequence".into()),
                            format!("unsupported token adjacency '{}-{}'", window[0], window[1]),
                        ));
                    }
                }
            }
        }
    }
    for (token, unit) in &bundle.unit_library {
        if unit
            .template_resname
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
        {
            errors.push(to_error(
                "E_SOURCE_SCHEMA",
                Some(format!("unit_library.{token}.template_resname")),
                "sequence-capable unit definitions must declare template_resname",
            ));
        }
        for junction_name in unit.junctions.values() {
            if !bundle.junction_library.contains_key(junction_name) {
                errors.push(to_error(
                    "E_SOURCE_SCHEMA",
                    Some(format!("unit_library.{token}.junctions")),
                    format!("junction '{}' missing from junction_library", junction_name),
                ));
            }
        }
    }
    for (idx, token) in req
        .target
        .sequence
        .clone()
        .unwrap_or_default()
        .iter()
        .enumerate()
    {
        if !bundle_has_token(bundle, token) {
            errors.push(to_error(
                "E_UNKNOWN_TOKEN",
                Some(format!("/target/sequence/{idx}")),
                format!(
                    "sequence token '{}' not present in source token libraries",
                    token
                ),
            ));
        }
    }
    if let Some(graph_nodes) = req.target.graph_nodes.as_ref() {
        for (idx, node) in graph_nodes.iter().enumerate() {
            if !bundle_has_token(bundle, &node.token) {
                errors.push(to_error(
                    "E_UNKNOWN_TOKEN",
                    Some(format!("/target/graph_nodes/{idx}/token")),
                    format!(
                        "graph token '{}' not present in source token libraries",
                        node.token
                    ),
                ));
            }
        }
    }
    let requested_base_mode = match base_conformation_mode(&req.realization.conformation_mode) {
        Ok(mode) => mode,
        Err(err) => {
            errors.push(err);
            "extended"
        }
    };
    if !bundle
        .capabilities
        .supported_conformation_modes
        .iter()
        .any(|mode| mode == requested_base_mode)
    {
        errors.push(to_error(
            "E_UNSUPPORTED_REALIZATION",
            Some("/realization/conformation_mode".into()),
            "source bundle does not advertise requested conformation mode",
        ));
    }
    if req.realization.conformation_mode == "aligned"
        && req
            .realization
            .alignment_axis
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
    {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("/realization/alignment_axis".into()),
            "aligned realization requires alignment_axis",
        ));
    }
    if req.realization.conformation_mode == "ensemble"
        && req.realization.ensemble_size.unwrap_or(0) == 0
    {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("/realization/ensemble_size".into()),
            "ensemble realization requires ensemble_size >= 1",
        ));
    }
    if !bundle
        .capabilities
        .supported_tacticity_modes
        .iter()
        .any(|mode| mode == &req.target.stereochemistry.mode)
    {
        errors.push(to_error(
            "E_UNSUPPORTED_STEREOCHEMISTRY",
            Some("/target/stereochemistry/mode".into()),
            "source bundle does not advertise requested stereochemistry mode",
        ));
    }
    if !SUPPORTED_TACTICITY_MODES.contains(&req.target.stereochemistry.mode.as_str()) {
        errors.push(to_error(
            "E_UNSUPPORTED_STEREOCHEMISTRY",
            Some("/target/stereochemistry/mode".into()),
            format!(
                "unsupported stereochemistry mode '{}'",
                req.target.stereochemistry.mode
            ),
        ));
    }
    if req.target.mode == "star_polymer"
        || req.target.mode == "branched_polymer"
        || req.target.mode == "polymer_graph"
    {
        for (path, policy) in [
            ("target.termini.head", req.target.termini.head.as_str()),
            ("target.termini.tail", req.target.termini.tail.as_str()),
        ] {
            if let Err(error) = resolve_cap_token(bundle, policy, path) {
                errors.push(error);
            }
        }
    } else {
        for (path, policy, fallback_token) in [
            (
                "target.termini.head",
                req.target.termini.head.as_str(),
                compiled_sequence
                    .first()
                    .or_else(|| expanded_sequence.first()),
            ),
            (
                "target.termini.tail",
                req.target.termini.tail.as_str(),
                compiled_sequence
                    .last()
                    .or_else(|| expanded_sequence.last()),
            ),
        ] {
            let supported_policy = bundle
                .capabilities
                .supported_termini_policies
                .iter()
                .any(|item| item == policy)
                || SUPPORTED_TERMINI_POLICIES.contains(&policy)
                || bundle.unit_library.contains_key(policy);
            if !supported_policy {
                errors.push(to_error(
                    "E_UNSUPPORTED_TERMINI",
                    Some(path.into()),
                    format!("unsupported termini policy '{}'", policy),
                ));
            } else if let Some(fallback_token) = fallback_token {
                if let Err(error) = resolve_terminus_token(bundle, policy, fallback_token, path) {
                    errors.push(error);
                }
            }
        }
    }
    if req.artifacts.coordinates == req.artifacts.build_manifest
        || req
            .artifacts
            .raw_coordinates
            .as_ref()
            .map(|path| {
                path == &req.artifacts.coordinates
                    || path == &req.artifacts.build_manifest
                    || path == &req.artifacts.charge_manifest
            })
            .unwrap_or(false)
        || req.artifacts.coordinates == req.artifacts.charge_manifest
        || req.artifacts.build_manifest == req.artifacts.charge_manifest
        || req
            .artifacts
            .inpcrd
            .as_ref()
            .map(|path| {
                path == &req.artifacts.coordinates
                    || req
                        .artifacts
                        .raw_coordinates
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
                    || path == &req.artifacts.build_manifest
                    || path == &req.artifacts.charge_manifest
            })
            .unwrap_or(false)
        || req
            .artifacts
            .topology
            .as_ref()
            .map(|path| {
                path == &req.artifacts.coordinates
                    || req
                        .artifacts
                        .raw_coordinates
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
                    || path == &req.artifacts.build_manifest
                    || path == &req.artifacts.charge_manifest
                    || req
                        .artifacts
                        .inpcrd
                        .as_ref()
                        .map(|inpcrd| inpcrd == path)
                        .unwrap_or(false)
            })
            .unwrap_or(false)
        || req
            .artifacts
            .topology_graph
            .as_ref()
            .map(|path| {
                path == &req.artifacts.coordinates
                    || req
                        .artifacts
                        .raw_coordinates
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
                    || path == &req.artifacts.build_manifest
                    || path == &req.artifacts.charge_manifest
                    || req
                        .artifacts
                        .inpcrd
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
                    || req
                        .artifacts
                        .topology
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
            })
            .unwrap_or(false)
        || req
            .artifacts
            .ensemble_manifest
            .as_ref()
            .map(|path| {
                path == &req.artifacts.coordinates
                    || req
                        .artifacts
                        .raw_coordinates
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
                    || path == &req.artifacts.build_manifest
                    || path == &req.artifacts.charge_manifest
                    || req
                        .artifacts
                        .inpcrd
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
                    || req
                        .artifacts
                        .topology
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
                    || req
                        .artifacts
                        .topology_graph
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
                    || req
                        .artifacts
                        .forcefield_ref
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
            })
            .unwrap_or(false)
        || req
            .artifacts
            .forcefield_ref
            .as_ref()
            .map(|path| {
                path == &req.artifacts.coordinates
                    || req
                        .artifacts
                        .raw_coordinates
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
                    || path == &req.artifacts.build_manifest
                    || path == &req.artifacts.charge_manifest
                    || req
                        .artifacts
                        .inpcrd
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
                    || req
                        .artifacts
                        .topology
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
                    || req
                        .artifacts
                        .topology_graph
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
                    || req
                        .artifacts
                        .ensemble_manifest
                        .as_ref()
                        .map(|item| item == path)
                        .unwrap_or(false)
            })
            .unwrap_or(false)
    {
        errors.push(to_error(
            "E_OUTPUT_COLLISION",
            Some("artifacts".into()),
            "output artifact paths must differ",
        ));
    }
    if req.artifacts.forcefield_ref.is_some() && bundle.artifacts.forcefield_ref.is_none() {
        errors.push(to_error(
            "E_UNSUPPORTED_TARGET",
            Some("/artifacts/forcefield_ref".into()),
            "artifacts.forcefield_ref requires bundle.artifacts.forcefield_ref to point to a transferable .ffxml",
        ));
    }
    if req.artifacts.forcefield_ref.is_some() {
        if let Some(path) = bundle.artifacts.forcefield_ref.as_ref() {
            let resolved = resolve_relative_path(Path::new(&req.source_ref.bundle_path), path);
            if let Some(reason) = forcefield_ref_placeholder_reason(Path::new(&resolved)) {
                errors.push(to_error(
                    "E_UNSUPPORTED_TARGET",
                    Some("/artifacts/forcefield_ref".into()),
                    format!("artifacts.forcefield_ref requires a non-placeholder .ffxml: {reason}"),
                ));
            }
        }
    }
    errors
}

fn emit(event: &RunEvent, enabled: bool) {
    if enabled {
        eprintln!(
            "{}",
            serde_json::to_string(event).unwrap_or_else(|_| "{}".into())
        );
    }
}

fn derived_member_path(coordinates_path: &str, member_idx: usize) -> String {
    let path = Path::new(coordinates_path);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("polymer");
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("pdb");
    parent
        .join(format!("{stem}.member_{:03}.{ext}", member_idx + 1))
        .to_string_lossy()
        .to_string()
}

fn rotate_output_along_axis(output: &mut PackOutput, axis_name: &str) {
    if output.atoms.is_empty() {
        return;
    }
    let target = match axis_name.trim().to_ascii_lowercase().as_str() {
        "x" | "+x" => [1.0f32, 0.0, 0.0],
        "-x" => [-1.0, 0.0, 0.0],
        "y" | "+y" => [0.0, 1.0, 0.0],
        "-y" => [0.0, -1.0, 0.0],
        "z" | "+z" => [0.0, 0.0, 1.0],
        "-z" => [0.0, 0.0, -1.0],
        _ => [0.0, 0.0, 1.0],
    };
    let mut center = output.atoms[0].position.scale(0.0);
    for atom in &output.atoms {
        center = center.add(atom.position);
    }
    center = center.scale(1.0 / output.atoms.len() as f32);
    let first = output
        .atoms
        .first()
        .map(|atom| atom.position)
        .unwrap_or(center);
    let last = output
        .atoms
        .last()
        .map(|atom| atom.position)
        .unwrap_or(center);
    let mut from = last.sub(first);
    if from.norm() <= 1.0e-6 {
        from = output.atoms[0]
            .position
            .scale(0.0)
            .add(Vec3::new(1.0, 0.0, 0.0));
    }
    let target = Vec3::new(target[0], target[1], target[2]);
    let dot = (from.dot(target) / (from.norm() * target.norm()).max(1.0e-6)).clamp(-1.0, 1.0);
    let axis = from.cross(target);
    let theta = dot.acos();
    let axis = if axis.norm() <= 1.0e-6 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        axis.scale(1.0 / axis.norm())
    };
    for atom in &mut output.atoms {
        let rel = atom.position.sub(center);
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        let rotated = rel
            .scale(cos_t)
            .add(axis.cross(rel).scale(sin_t))
            .add(axis.scale(axis.dot(rel) * (1.0 - cos_t)));
        atom.position = center.add(rotated);
    }
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct RelaxReport {
    mode: String,
    steps_requested: usize,
    steps_executed: usize,
    initial_max_clash: f32,
    final_max_clash: f32,
    rms_displacement: f32,
    initial_overlap_pairs: usize,
    final_overlap_pairs: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    movable_atom_count: Option<usize>,
    max_atom_displacement_angstrom: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    fallback_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fallback_steps_requested: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fallback_steps_executed: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pre_fallback_max_clash: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pre_fallback_overlap_pairs: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    initial_energy_kcal_mol: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_energy_kcal_mol: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    initial_max_force: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_max_force: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    accepted_line_search_steps: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rejected_line_search_steps: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    termination_reason: Option<String>,
    raw_coordinates: Option<String>,
}

fn overlap_status_summary(
    output: Option<&PackOutput>,
    solver_cleanup: Option<&RelaxReport>,
    relax: Option<&RelaxReport>,
) -> OverlapStatusSummary {
    let (report_source, report) = if let Some(report) = relax {
        (Some("relax"), Some(report))
    } else if let Some(report) = solver_cleanup {
        (Some("solver_cleanup"), Some(report))
    } else {
        (None, None)
    };
    if let Some(report) = report {
        let clear = report.final_overlap_pairs == 0;
        return OverlapStatusSummary {
            status: if clear { "clear" } else { "remaining" }.into(),
            may_report_no_overlaps: clear,
            metric: OVERLAP_REPORT_METRIC.into(),
            report_source: report_source.map(str::to_string),
            overlap_pairs: Some(report.final_overlap_pairs),
            max_overlap_angstrom: Some(report.final_max_clash),
        };
    }
    let Some(output) = output else {
        return OverlapStatusSummary {
            status: "not_evaluated".into(),
            may_report_no_overlaps: false,
            metric: OVERLAP_REPORT_METRIC.into(),
            report_source: None,
            overlap_pairs: None,
            max_overlap_angstrom: None,
        };
    };
    let positions = output
        .atoms
        .iter()
        .map(|atom| atom.position)
        .collect::<Vec<_>>();
    let topology_context = build_relax_topology_context(output.atoms.len(), &output.bonds);
    let stats = collect_overlap_stats(&output.atoms, &positions, 0.9, &topology_context, false);
    let clear = stats.pair_count == 0;
    OverlapStatusSummary {
        status: if clear { "clear" } else { "remaining" }.into(),
        may_report_no_overlaps: clear,
        metric: OVERLAP_REPORT_METRIC.into(),
        report_source: Some("final_structure".into()),
        overlap_pairs: Some(stats.pair_count),
        max_overlap_angstrom: Some(stats.max_overlap),
    }
}

#[derive(Clone, Debug)]
struct OverlapPair {
    left: usize,
    right: usize,
    distance: f32,
    overlap: f32,
}

#[derive(Clone, Debug, Default)]
struct OverlapStats {
    pair_count: usize,
    max_overlap: f32,
    pairs: Vec<OverlapPair>,
}

#[derive(Clone, Debug)]
struct RelaxTopologyContext {
    adjacency: Vec<Vec<usize>>,
    excluded_pairs: BTreeSet<(usize, usize)>,
}

fn atom_vdw_radius(element: &str) -> f32 {
    match element.trim().to_ascii_uppercase().as_str() {
        "H" => 1.20,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.80,
        "P" => 1.80,
        _ => 1.60,
    }
}

fn max_scaled_vdw_cutoff(clash_scale: f32) -> f32 {
    (2.0 * atom_vdw_radius("S") * clash_scale).max(1.0)
}

fn atom_position_index(positions: &[Vec3], cell_size: f32) -> SpatialHash {
    let mut index = SpatialHash::with_capacity(cell_size, positions.len() * 2);
    for (idx, position) in positions.iter().copied().enumerate() {
        index.insert(idx, position);
    }
    index
}

fn max_atom_clash(output: &PackOutput, clash_scale: f32) -> f32 {
    let bonded = output
        .bonds
        .iter()
        .map(|(a, b)| if a <= b { (*a, *b) } else { (*b, *a) })
        .collect::<BTreeSet<_>>();
    let mut max_clash = 0.0f32;
    let positions = output
        .atoms
        .iter()
        .map(|atom| atom.position)
        .collect::<Vec<_>>();
    let index = atom_position_index(&positions, max_scaled_vdw_cutoff(clash_scale));
    for left in 0..output.atoms.len() {
        index.for_each_neighbor(output.atoms[left].position, |right| {
            if right <= left {
                return;
            }
            if bonded.contains(&(left, right)) {
                return;
            }
            let atom_left = &output.atoms[left];
            let atom_right = &output.atoms[right];
            if atom_left.resid == atom_right.resid {
                return;
            }
            let cutoff = (atom_vdw_radius(&atom_left.element)
                + atom_vdw_radius(&atom_right.element))
                * clash_scale;
            let dist = atom_left.position.sub(atom_right.position).norm();
            max_clash = max_clash.max(cutoff - dist);
        });
    }
    max_clash.max(0.0)
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn fallback_pair_direction(left: usize, right: usize) -> Vec3 {
    let mut seed = (left as u64).wrapping_mul(6364136223846793005);
    seed = seed
        .wrapping_add(right as u64)
        .wrapping_add(1442695040888963407);
    let x = ((seed & 0xFF) as f32 / 255.0) * 2.0 - 1.0;
    let y = (((seed >> 8) & 0xFF) as f32 / 255.0) * 2.0 - 1.0;
    let z = (((seed >> 16) & 0xFF) as f32 / 255.0) * 2.0 - 1.0;
    let direction = Vec3::new(x, y, z);
    let norm = direction.norm();
    if norm > 1.0e-6 {
        direction.scale(1.0 / norm)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    }
}

fn build_relax_topology_context(
    atom_count: usize,
    bonds: &[(usize, usize)],
) -> RelaxTopologyContext {
    let adjacency = bond_adjacency(atom_count, bonds);
    let mut excluded_pairs = bonds
        .iter()
        .map(|&(left, right)| ordered_pair(left, right))
        .collect::<BTreeSet<_>>();
    for neighbors in &adjacency {
        for left_idx in 0..neighbors.len() {
            for right_idx in (left_idx + 1)..neighbors.len() {
                excluded_pairs.insert(ordered_pair(neighbors[left_idx], neighbors[right_idx]));
            }
        }
    }
    RelaxTopologyContext {
        adjacency,
        excluded_pairs,
    }
}

fn collect_overlap_stats(
    atoms: &[AtomRecord],
    positions: &[Vec3],
    clash_scale: f32,
    context: &RelaxTopologyContext,
    collect_pairs: bool,
) -> OverlapStats {
    if atoms.len() != positions.len() {
        return OverlapStats::default();
    }
    let mut stats = OverlapStats::default();
    let index = atom_position_index(positions, max_scaled_vdw_cutoff(clash_scale));
    for left in 0..atoms.len() {
        index.for_each_neighbor(positions[left], |right| {
            if right <= left {
                return;
            }
            if context.excluded_pairs.contains(&(left, right)) {
                return;
            }
            let cutoff = (atom_vdw_radius(&atoms[left].element)
                + atom_vdw_radius(&atoms[right].element))
                * clash_scale;
            let distance = positions[left].sub(positions[right]).norm();
            let overlap = (cutoff - distance).max(0.0);
            if overlap <= 1.0e-4 {
                return;
            }
            stats.pair_count += 1;
            stats.max_overlap = stats.max_overlap.max(overlap);
            if collect_pairs {
                stats.pairs.push(OverlapPair {
                    left,
                    right,
                    distance,
                    overlap,
                });
            }
        });
    }
    stats
}

fn movable_overlap_shell_depths(
    atom_count: usize,
    adjacency: &[Vec<usize>],
    seed_atoms: &BTreeSet<usize>,
    max_depth: usize,
) -> Vec<Option<usize>> {
    let mut depths = vec![None; atom_count];
    let mut queue = VecDeque::new();
    for &atom_idx in seed_atoms {
        if atom_idx >= atom_count {
            continue;
        }
        depths[atom_idx] = Some(0);
        queue.push_back(atom_idx);
    }
    while let Some(atom_idx) = queue.pop_front() {
        let depth = depths[atom_idx].unwrap_or(0);
        if depth >= max_depth {
            continue;
        }
        for &neighbor in adjacency.get(atom_idx).into_iter().flatten() {
            let next_depth = depth + 1;
            if depths[neighbor].map_or(true, |current| next_depth < current) {
                depths[neighbor] = Some(next_depth);
                queue.push_back(neighbor);
            }
        }
    }
    depths
}

fn max_atom_displacement(initial_positions: &[Vec3], positions: &[Vec3]) -> f32 {
    initial_positions
        .iter()
        .zip(positions.iter())
        .map(|(initial, current)| current.sub(*initial).norm())
        .fold(0.0, f32::max)
}

fn rms_atom_displacement(initial_positions: &[Vec3], positions: &[Vec3]) -> f32 {
    (positions
        .iter()
        .zip(initial_positions.iter())
        .map(|(position, initial)| position.sub(*initial).norm().powi(2))
        .sum::<f32>()
        / positions.len().max(1) as f32)
        .sqrt()
}

fn targeted_steric_fallback_spec(relax: &RelaxSpec) -> RelaxSpec {
    RelaxSpec {
        mode: "targeted_steric".into(),
        steps: Some(relax.steps.unwrap_or(64).clamp(24, 128)),
        step_scale: Some(relax.step_scale.unwrap_or(0.25).clamp(0.2, 0.5)),
        clash_scale: Some(relax.clash_scale.unwrap_or(0.9)),
    }
}

fn merge_relax_report_with_followup_stage(
    primary: RelaxReport,
    followup: RelaxReport,
    initial_positions: &[Vec3],
    final_positions: &[Vec3],
) -> RelaxReport {
    RelaxReport {
        mode: primary.mode,
        steps_requested: primary
            .steps_requested
            .saturating_add(followup.steps_requested),
        steps_executed: primary
            .steps_executed
            .saturating_add(followup.steps_executed),
        initial_max_clash: primary.initial_max_clash,
        final_max_clash: followup.final_max_clash,
        rms_displacement: rms_atom_displacement(initial_positions, final_positions),
        initial_overlap_pairs: primary.initial_overlap_pairs,
        final_overlap_pairs: followup.final_overlap_pairs,
        movable_atom_count: followup.movable_atom_count.or(primary.movable_atom_count),
        max_atom_displacement_angstrom: max_atom_displacement(initial_positions, final_positions),
        fallback_mode: Some(followup.mode),
        fallback_steps_requested: Some(followup.steps_requested),
        fallback_steps_executed: Some(followup.steps_executed),
        pre_fallback_max_clash: Some(primary.final_max_clash),
        pre_fallback_overlap_pairs: Some(primary.final_overlap_pairs),
        initial_energy_kcal_mol: primary.initial_energy_kcal_mol,
        final_energy_kcal_mol: followup
            .final_energy_kcal_mol
            .or(primary.final_energy_kcal_mol),
        initial_max_force: primary.initial_max_force,
        final_max_force: followup.final_max_force.or(primary.final_max_force),
        accepted_line_search_steps: followup
            .accepted_line_search_steps
            .or(primary.accepted_line_search_steps),
        rejected_line_search_steps: followup
            .rejected_line_search_steps
            .or(primary.rejected_line_search_steps),
        termination_reason: followup.termination_reason.or(primary.termination_reason),
        raw_coordinates: primary.raw_coordinates.or(followup.raw_coordinates),
    }
}

fn output_positions(output: &PackOutput) -> Vec<Vec3> {
    output.atoms.iter().map(|atom| atom.position).collect()
}

fn relax_synthetic_topology_output(
    output: &mut PackOutput,
    topology: &AmberTopology,
    _qc_context: &crate::polymer::BuildQcContext,
    raw_coordinates: Option<String>,
) -> RelaxReport {
    let steps_requested = if output.atoms.len() <= 512 { 720 } else { 420 };
    let initial_positions = output
        .atoms
        .iter()
        .map(|atom| atom.position)
        .collect::<Vec<_>>();
    let topology_context = build_relax_topology_context(output.atoms.len(), &output.bonds);
    let initial_overlap_stats = collect_overlap_stats(
        &output.atoms,
        &initial_positions,
        0.9,
        &topology_context,
        false,
    );
    if output.atoms.len() < 4 {
        return RelaxReport {
            mode: "synthetic_bonded".into(),
            steps_requested,
            steps_executed: 0,
            initial_max_clash: initial_overlap_stats.max_overlap,
            final_max_clash: initial_overlap_stats.max_overlap,
            rms_displacement: 0.0,
            initial_overlap_pairs: initial_overlap_stats.pair_count,
            final_overlap_pairs: initial_overlap_stats.pair_count,
            movable_atom_count: None,
            max_atom_displacement_angstrom: 0.0,
            fallback_mode: None,
            fallback_steps_requested: None,
            fallback_steps_executed: None,
            pre_fallback_max_clash: None,
            pre_fallback_overlap_pairs: None,
            initial_energy_kcal_mol: None,
            final_energy_kcal_mol: None,
            initial_max_force: None,
            final_max_force: None,
            accepted_line_search_steps: None,
            rejected_line_search_steps: None,
            termination_reason: Some("not_needed".into()),
            raw_coordinates,
        };
    }
    let telemetry = minimize_synthetic_topology(output, topology, steps_requested, 0.45);
    let final_positions = output
        .atoms
        .iter()
        .map(|atom| atom.position)
        .collect::<Vec<_>>();
    let final_overlap_stats = collect_overlap_stats(
        &output.atoms,
        &final_positions,
        0.9,
        &topology_context,
        false,
    );
    RelaxReport {
        mode: "synthetic_bonded".into(),
        steps_requested,
        steps_executed: telemetry.steps_executed,
        initial_max_clash: initial_overlap_stats.max_overlap,
        final_max_clash: final_overlap_stats.max_overlap,
        rms_displacement: rms_atom_displacement(&initial_positions, &final_positions),
        initial_overlap_pairs: initial_overlap_stats.pair_count,
        final_overlap_pairs: final_overlap_stats.pair_count,
        movable_atom_count: Some(telemetry.moved_atom_count),
        max_atom_displacement_angstrom: max_atom_displacement(&initial_positions, &final_positions),
        fallback_mode: None,
        fallback_steps_requested: None,
        fallback_steps_executed: None,
        pre_fallback_max_clash: None,
        pre_fallback_overlap_pairs: None,
        initial_energy_kcal_mol: Some(telemetry.initial_energy),
        final_energy_kcal_mol: Some(telemetry.final_energy),
        initial_max_force: Some(telemetry.initial_max_force),
        final_max_force: Some(telemetry.final_max_force),
        accepted_line_search_steps: Some(telemetry.accepted_steps),
        rejected_line_search_steps: Some(telemetry.rejected_steps),
        termination_reason: Some(telemetry.termination_reason),
        raw_coordinates,
    }
}

#[derive(Clone, Debug)]
struct GraphRelaxEdgeTarget {
    parent_group: usize,
    child_group: usize,
    parent_idx: usize,
    child_idx: usize,
    target: f32,
}

fn restore_graph_edge_targets(
    output: &mut PackOutput,
    residue_atoms: &[Vec<usize>],
    edge_targets: &[GraphRelaxEdgeTarget],
    root_group: usize,
    passes: usize,
) {
    for _ in 0..passes {
        for target in edge_targets {
            let current = output.atoms[target.child_idx]
                .position
                .sub(output.atoms[target.parent_idx].position);
            let distance = current.norm().max(1.0e-6);
            let stretch = distance - target.target;
            if stretch.abs() <= 1.0e-3 {
                continue;
            }
            let correction = current.scale(stretch / distance);
            let parent_locked = target.parent_group == root_group;
            let child_locked = target.child_group == root_group;
            match (parent_locked, child_locked) {
                (true, true) => {}
                (true, false) => {
                    if let Some(atom_indices) = residue_atoms.get(target.child_group) {
                        for atom_idx in atom_indices {
                            output.atoms[*atom_idx].position =
                                output.atoms[*atom_idx].position.sub(correction);
                        }
                    }
                }
                (false, true) => {
                    if let Some(atom_indices) = residue_atoms.get(target.parent_group) {
                        for atom_idx in atom_indices {
                            output.atoms[*atom_idx].position =
                                output.atoms[*atom_idx].position.add(correction);
                        }
                    }
                }
                (false, false) => {
                    let half = correction.scale(0.5);
                    if let Some(atom_indices) = residue_atoms.get(target.parent_group) {
                        for atom_idx in atom_indices {
                            output.atoms[*atom_idx].position =
                                output.atoms[*atom_idx].position.add(half);
                        }
                    }
                    if let Some(atom_indices) = residue_atoms.get(target.child_group) {
                        for atom_idx in atom_indices {
                            output.atoms[*atom_idx].position =
                                output.atoms[*atom_idx].position.sub(half);
                        }
                    }
                }
            }
        }
    }
}

fn relax_graph_spring_output(
    output: &mut PackOutput,
    plan: &CompiledBuildPlan,
    step_length: f32,
    relax: &RelaxSpec,
    raw_coordinates: Option<String>,
) -> RelaxReport {
    let steps_requested = relax.steps.unwrap_or(64);
    let step_scale = relax.step_scale.unwrap_or(0.25);
    let clash_scale = relax.clash_scale.unwrap_or(0.9);
    let initial_positions = output
        .atoms
        .iter()
        .map(|atom| atom.position)
        .collect::<Vec<_>>();
    let topology_context = build_relax_topology_context(output.atoms.len(), &output.bonds);
    let initial_overlap_stats = collect_overlap_stats(
        &output.atoms,
        &initial_positions,
        clash_scale,
        &topology_context,
        false,
    );
    let initial_max_clash = initial_overlap_stats.max_overlap;
    if output.atoms.is_empty() || steps_requested == 0 {
        return RelaxReport {
            mode: canonical_relax_mode(&relax.mode)
                .unwrap_or("graph_spring")
                .into(),
            steps_requested,
            steps_executed: 0,
            initial_max_clash,
            final_max_clash: initial_max_clash,
            rms_displacement: 0.0,
            initial_overlap_pairs: initial_overlap_stats.pair_count,
            final_overlap_pairs: initial_overlap_stats.pair_count,
            movable_atom_count: None,
            max_atom_displacement_angstrom: 0.0,
            fallback_mode: None,
            fallback_steps_requested: None,
            fallback_steps_executed: None,
            pre_fallback_max_clash: None,
            pre_fallback_overlap_pairs: None,
            initial_energy_kcal_mol: None,
            final_energy_kcal_mol: None,
            initial_max_force: None,
            final_max_force: None,
            accepted_line_search_steps: None,
            rejected_line_search_steps: None,
            termination_reason: None,
            raw_coordinates,
        };
    }
    let residue_count = output
        .atoms
        .iter()
        .map(|atom| atom.resid.max(1) as usize)
        .max()
        .unwrap_or(0);
    let mut residue_atoms = vec![Vec::<usize>::new(); residue_count];
    for (idx, atom) in output.atoms.iter().enumerate() {
        let resid = atom.resid.max(1) as usize;
        if let Some(group) = residue_atoms.get_mut(resid.saturating_sub(1)) {
            group.push(idx);
        }
    }
    let edge_targets = plan
        .edges
        .iter()
        .filter_map(|edge| {
            let parent_resid = edge.parent + 1;
            let child_resid = edge.child + 1;
            let parent_group = parent_resid.saturating_sub(1);
            let child_group = child_resid.saturating_sub(1);
            let parent_idx = residue_atoms
                .get(parent_group)?
                .iter()
                .copied()
                .find(|idx| output.atoms[*idx].name.trim() == edge.parent_attach_atom.trim())?;
            let child_idx = residue_atoms
                .get(child_group)?
                .iter()
                .copied()
                .find(|idx| output.atoms[*idx].name.trim() == edge.child_attach_atom.trim())?;
            let target = output.atoms[child_idx]
                .position
                .sub(output.atoms[parent_idx].position)
                .norm()
                .max(0.9)
                .min(step_length.max(1.5));
            Some(GraphRelaxEdgeTarget {
                parent_group,
                child_group,
                parent_idx,
                child_idx,
                target,
            })
        })
        .collect::<Vec<_>>();
    let restore_edge_targets = !plan.graph_has_cycle;
    let root_resid = plan
        .nodes
        .iter()
        .position(|node| node.node_id == plan.expanded_root_node_id)
        .map(|idx| idx + 1)
        .unwrap_or(1);
    let initial_center = center_of_geometry(&initial_positions);
    let mut steps_executed = 0usize;
    for step_idx in 0..steps_requested {
        let centers = residue_atoms
            .iter()
            .map(|atom_indices| {
                let points = atom_indices
                    .iter()
                    .map(|idx| output.atoms[*idx].position)
                    .collect::<Vec<_>>();
                if points.is_empty() {
                    Vec3::new(0.0, 0.0, 0.0)
                } else {
                    center_of_geometry(&points)
                }
            })
            .collect::<Vec<_>>();
        let mut deltas = vec![Vec3::new(0.0, 0.0, 0.0); residue_atoms.len()];
        for target in &edge_targets {
            if target.parent_group >= centers.len() || target.child_group >= centers.len() {
                continue;
            }
            let diff = output.atoms[target.child_idx]
                .position
                .sub(output.atoms[target.parent_idx].position);
            let dist = diff.norm().max(1.0e-4);
            let dir = diff.scale(1.0 / dist);
            let stretch = dist - target.target;
            let correction = dir.scale(0.18 * stretch);
            deltas[target.parent_group] = deltas[target.parent_group].add(correction);
            deltas[target.child_group] = deltas[target.child_group].sub(correction);
        }
        let positions = output
            .atoms
            .iter()
            .map(|atom| atom.position)
            .collect::<Vec<_>>();
        let atom_index = atom_position_index(&positions, max_scaled_vdw_cutoff(clash_scale));
        for left in 0..output.atoms.len() {
            atom_index.for_each_neighbor(output.atoms[left].position, |right| {
                if right <= left {
                    return;
                }
                let atom_left = &output.atoms[left];
                let atom_right = &output.atoms[right];
                if atom_left.resid == atom_right.resid {
                    return;
                }
                let left_group = atom_left.resid.max(1) as usize - 1;
                let right_group = atom_right.resid.max(1) as usize - 1;
                let cutoff = (atom_vdw_radius(&atom_left.element)
                    + atom_vdw_radius(&atom_right.element))
                    * clash_scale;
                let diff = atom_right.position.sub(atom_left.position);
                let dist = diff.norm().max(1.0e-4);
                if dist >= cutoff {
                    return;
                }
                let dir = diff.scale(1.0 / dist);
                let push = dir.scale(0.12 * (cutoff - dist));
                deltas[left_group] = deltas[left_group].sub(push);
                deltas[right_group] = deltas[right_group].add(push);
            });
        }
        for (resid_idx, atom_indices) in residue_atoms.iter().enumerate() {
            if resid_idx + 1 == root_resid {
                continue;
            }
            let delta = deltas[resid_idx].scale(step_scale / atom_indices.len().max(1) as f32);
            for atom_idx in atom_indices {
                output.atoms[*atom_idx].position = output.atoms[*atom_idx].position.add(delta);
            }
        }
        let current_center = center_of_geometry(
            &output
                .atoms
                .iter()
                .map(|atom| atom.position)
                .collect::<Vec<_>>(),
        );
        let shift = current_center.sub(initial_center);
        for atom in &mut output.atoms {
            atom.position = atom.position.sub(shift);
        }
        if restore_edge_targets {
            restore_graph_edge_targets(
                output,
                &residue_atoms,
                &edge_targets,
                root_resid.saturating_sub(1),
                4,
            );
        }
        steps_executed = step_idx + 1;
        if max_atom_clash(output, clash_scale) <= 1.0e-2 {
            break;
        }
    }
    if restore_edge_targets {
        restore_graph_edge_targets(
            output,
            &residue_atoms,
            &edge_targets,
            root_resid.saturating_sub(1),
            24,
        );
    }
    let final_positions = output
        .atoms
        .iter()
        .map(|atom| atom.position)
        .collect::<Vec<_>>();
    let final_overlap_stats = collect_overlap_stats(
        &output.atoms,
        &final_positions,
        clash_scale,
        &topology_context,
        false,
    );
    let rms_displacement = rms_atom_displacement(&initial_positions, &final_positions);
    let final_max_clash = final_overlap_stats.max_overlap;
    RelaxReport {
        mode: canonical_relax_mode(&relax.mode)
            .unwrap_or("graph_spring")
            .into(),
        steps_requested,
        steps_executed,
        initial_max_clash,
        final_max_clash,
        rms_displacement,
        initial_overlap_pairs: initial_overlap_stats.pair_count,
        final_overlap_pairs: final_overlap_stats.pair_count,
        movable_atom_count: None,
        max_atom_displacement_angstrom: max_atom_displacement(&initial_positions, &final_positions),
        fallback_mode: None,
        fallback_steps_requested: None,
        fallback_steps_executed: None,
        pre_fallback_max_clash: None,
        pre_fallback_overlap_pairs: None,
        initial_energy_kcal_mol: None,
        final_energy_kcal_mol: None,
        initial_max_force: None,
        final_max_force: None,
        accepted_line_search_steps: None,
        rejected_line_search_steps: None,
        termination_reason: None,
        raw_coordinates,
    }
}

fn targeted_steric_position_weight(depth: usize) -> f32 {
    match depth {
        0 => 0.04,
        1 => 0.14,
        _ => 0.28,
    }
}

fn targeted_steric_objective(
    stats: &OverlapStats,
    positions: &[Vec3],
    anchor_positions: &[Vec3],
    movable_depths: &[Option<usize>],
    movable_atoms: &[usize],
    bond_targets: &[(usize, usize, f32)],
    bond_weight: f32,
    steric_weight: f32,
    anchor_weight_scale: f32,
) -> f32 {
    let steric_energy = stats
        .pairs
        .iter()
        .map(|pair| 0.5 * steric_weight * pair.overlap * pair.overlap)
        .sum::<f32>();
    let bond_energy = bond_targets
        .iter()
        .filter(|&&(left, right, _)| {
            movable_depths.get(left).and_then(|depth| *depth).is_some()
                || movable_depths.get(right).and_then(|depth| *depth).is_some()
        })
        .map(|&(left, right, target)| {
            let stretch = positions[left].sub(positions[right]).norm() - target;
            0.5 * bond_weight * stretch * stretch
        })
        .sum::<f32>();
    let anchor_energy = movable_atoms
        .iter()
        .map(|&atom_idx| {
            let depth = movable_depths[atom_idx].unwrap_or(2);
            let displacement = positions[atom_idx].sub(anchor_positions[atom_idx]).norm();
            0.5 * targeted_steric_position_weight(depth)
                * anchor_weight_scale
                * displacement
                * displacement
        })
        .sum::<f32>();
    steric_energy + bond_energy + anchor_energy
}

fn restore_targeted_steric_bonds(
    positions: &mut [Vec3],
    bond_targets: &[(usize, usize, f32)],
    movable_depths: &[Option<usize>],
    max_step: f32,
) {
    for _ in 0..2 {
        for &(left, right, target) in bond_targets {
            let left_movable = movable_depths[left].is_some();
            let right_movable = movable_depths[right].is_some();
            if !left_movable && !right_movable {
                continue;
            }
            let delta = positions[right].sub(positions[left]);
            let distance = delta.norm().max(1.0e-6);
            let stretch = distance - target;
            if stretch.abs() <= 1.0e-4 {
                continue;
            }
            let direction = delta.scale(1.0 / distance);
            let correction = (0.5 * stretch).clamp(-max_step, max_step);
            match (left_movable, right_movable) {
                (true, true) => {
                    let shift = direction.scale(0.5 * correction);
                    positions[left] = positions[left].add(shift);
                    positions[right] = positions[right].sub(shift);
                }
                (true, false) => {
                    positions[left] = positions[left].add(direction.scale(correction));
                }
                (false, true) => {
                    positions[right] = positions[right].sub(direction.scale(correction));
                }
                (false, false) => {}
            }
        }
    }
}

fn apply_targeted_steric_overlap_push(
    atoms: &[AtomRecord],
    positions: &mut [Vec3],
    clash_scale: f32,
    context: &RelaxTopologyContext,
    movable_depths: &[Option<usize>],
    movable_atoms: &[usize],
    bond_targets: &[(usize, usize, f32)],
    max_step: f32,
) -> OverlapStats {
    let residual_stats = collect_overlap_stats(atoms, positions, clash_scale, context, true);
    if residual_stats.pair_count == 0 {
        return residual_stats;
    }
    let mut deltas = vec![Vec3::new(0.0, 0.0, 0.0); positions.len()];
    for pair in &residual_stats.pairs {
        let left_movable = movable_depths[pair.left].is_some();
        let right_movable = movable_depths[pair.right].is_some();
        if !left_movable && !right_movable {
            continue;
        }
        let delta = positions[pair.left].sub(positions[pair.right]);
        let direction = if pair.distance > 1.0e-6 {
            delta.scale(1.0 / pair.distance)
        } else {
            fallback_pair_direction(pair.left, pair.right)
        };
        let push = (pair.overlap * 0.55 + 1.0e-3).min(max_step);
        match (left_movable, right_movable) {
            (true, true) => {
                let shift = direction.scale(0.5 * push);
                deltas[pair.left] = deltas[pair.left].add(shift);
                deltas[pair.right] = deltas[pair.right].sub(shift);
            }
            (true, false) => {
                deltas[pair.left] = deltas[pair.left].add(direction.scale(push));
            }
            (false, true) => {
                deltas[pair.right] = deltas[pair.right].sub(direction.scale(push));
            }
            (false, false) => {}
        }
    }
    for &atom_idx in movable_atoms {
        let delta = deltas[atom_idx];
        let norm = delta.norm();
        let applied = if norm > max_step && norm > 1.0e-8 {
            delta.scale(max_step / norm)
        } else {
            delta
        };
        positions[atom_idx] = positions[atom_idx].add(applied);
    }
    restore_targeted_steric_bonds(positions, bond_targets, movable_depths, max_step * 0.5);
    collect_overlap_stats(atoms, positions, clash_scale, context, true)
}

fn run_targeted_steric_stage(
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
    positions: &mut [Vec3],
    clash_scale: f32,
    step_scale: f32,
    steps_requested: usize,
    context: &RelaxTopologyContext,
    shell_depth: usize,
) -> (OverlapStats, usize, usize) {
    let mut current_stats = collect_overlap_stats(atoms, positions, clash_scale, context, true);
    if current_stats.pair_count == 0 {
        return (current_stats, 0, 0);
    }

    let seed_atoms = current_stats
        .pairs
        .iter()
        .flat_map(|pair| [pair.left, pair.right])
        .collect::<BTreeSet<_>>();
    let movable_depths =
        movable_overlap_shell_depths(atoms.len(), &context.adjacency, &seed_atoms, shell_depth);
    let movable_atoms = movable_depths
        .iter()
        .enumerate()
        .filter_map(|(atom_idx, depth)| depth.map(|_| atom_idx))
        .collect::<Vec<_>>();
    if movable_atoms.is_empty() {
        return (current_stats, 0, 0);
    }

    let anchor_positions = positions.to_vec();
    let bond_targets = bonds
        .iter()
        .map(|&(left, right)| {
            (
                left,
                right,
                anchor_positions[right]
                    .sub(anchor_positions[left])
                    .norm()
                    .max(1.0e-3),
            )
        })
        .collect::<Vec<_>>();
    let shell_boost = 1.0 + 0.2 * shell_depth.saturating_sub(2) as f32;
    let anchor_weight_scale = match shell_depth {
        0..=2 => 1.0,
        3 => 0.7,
        _ => 0.5,
    };
    let base_alpha = 0.18 * step_scale.max(0.05) * shell_boost;
    let base_max_step =
        0.14 * step_scale.max(0.05) * (1.0 + 0.3 * shell_depth.saturating_sub(2) as f32);
    let bond_weight = 4.0f32;
    let steric_weight = 3.0f32 * shell_boost;
    let mut current_objective = targeted_steric_objective(
        &current_stats,
        positions,
        &anchor_positions,
        &movable_depths,
        &movable_atoms,
        &bond_targets,
        bond_weight,
        steric_weight,
        anchor_weight_scale,
    );
    let mut steps_executed = 0usize;

    for step_idx in 0..steps_requested {
        if current_stats.pair_count == 0 {
            break;
        }
        let mut gradient = vec![Vec3::new(0.0, 0.0, 0.0); positions.len()];

        for pair in &current_stats.pairs {
            let delta = positions[pair.left].sub(positions[pair.right]);
            let direction = if pair.distance > 1.0e-6 {
                delta.scale(1.0 / pair.distance)
            } else {
                fallback_pair_direction(pair.left, pair.right)
            };
            let overlap_grad = direction.scale(-pair.overlap * steric_weight);
            if movable_depths[pair.left].is_some() {
                gradient[pair.left] = gradient[pair.left].add(overlap_grad);
            }
            if movable_depths[pair.right].is_some() {
                gradient[pair.right] = gradient[pair.right].sub(overlap_grad);
            }
        }

        for &(left, right, target) in &bond_targets {
            if movable_depths[left].is_none() && movable_depths[right].is_none() {
                continue;
            }
            let delta = positions[left].sub(positions[right]);
            let distance = delta.norm().max(1.0e-6);
            let direction = delta.scale(1.0 / distance);
            let stretch = distance - target;
            let bond_grad = direction.scale(bond_weight * stretch);
            if movable_depths[left].is_some() {
                gradient[left] = gradient[left].add(bond_grad);
            }
            if movable_depths[right].is_some() {
                gradient[right] = gradient[right].sub(bond_grad);
            }
        }

        for &atom_idx in &movable_atoms {
            let depth = movable_depths[atom_idx].unwrap_or(shell_depth);
            let weight = targeted_steric_position_weight(depth) * anchor_weight_scale;
            gradient[atom_idx] = gradient[atom_idx].add(
                positions[atom_idx]
                    .sub(anchor_positions[atom_idx])
                    .scale(weight),
            );
        }

        let mut accepted = None;
        let mut trial_scale = 1.0f32;
        for _ in 0..8 {
            let alpha = base_alpha * trial_scale;
            let max_step = base_max_step * trial_scale;
            let mut candidate_positions = positions.to_vec();
            for &atom_idx in &movable_atoms {
                let mut delta = gradient[atom_idx].scale(-alpha);
                let norm = delta.norm();
                if norm > max_step && norm > 1.0e-8 {
                    delta = delta.scale(max_step / norm);
                }
                candidate_positions[atom_idx] = candidate_positions[atom_idx].add(delta);
            }
            let candidate_stats =
                collect_overlap_stats(atoms, &candidate_positions, clash_scale, context, true);
            let candidate_objective = targeted_steric_objective(
                &candidate_stats,
                &candidate_positions,
                &anchor_positions,
                &movable_depths,
                &movable_atoms,
                &bond_targets,
                bond_weight,
                steric_weight,
                anchor_weight_scale,
            );
            if candidate_stats.pair_count == 0 || candidate_objective + 1.0e-5 < current_objective {
                accepted = Some((candidate_positions, candidate_stats, candidate_objective));
                break;
            }
            trial_scale *= 0.5;
        }

        let Some((candidate_positions, candidate_stats, candidate_objective)) = accepted else {
            break;
        };
        positions.copy_from_slice(&candidate_positions);
        current_stats = candidate_stats;
        current_objective = candidate_objective;
        steps_executed = step_idx + 1;
    }

    let residual_passes = (steps_requested / 3).clamp(8, 32);
    let residual_max_step = base_max_step * (1.25 + 0.15 * shell_depth.saturating_sub(2) as f32);
    for _ in 0..residual_passes {
        if current_stats.pair_count == 0 {
            break;
        }
        let mut candidate_positions = positions.to_vec();
        let next_stats = apply_targeted_steric_overlap_push(
            atoms,
            &mut candidate_positions,
            clash_scale,
            context,
            &movable_depths,
            &movable_atoms,
            &bond_targets,
            residual_max_step,
        );
        let next_objective = targeted_steric_objective(
            &next_stats,
            &candidate_positions,
            &anchor_positions,
            &movable_depths,
            &movable_atoms,
            &bond_targets,
            bond_weight,
            steric_weight,
            anchor_weight_scale,
        );
        if next_stats.pair_count == 0 || next_objective + 1.0e-5 < current_objective {
            positions.copy_from_slice(&candidate_positions);
            current_stats = next_stats;
            current_objective = next_objective;
            steps_executed = steps_executed.saturating_add(1);
            continue;
        }
        break;
    }

    (current_stats, steps_executed, movable_atoms.len())
}

fn clear_targeted_steric_residual_overlaps(
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
    positions: &mut [Vec3],
    clash_scale: f32,
    context: &RelaxTopologyContext,
) -> (OverlapStats, usize, usize) {
    let mut current_stats = collect_overlap_stats(atoms, positions, clash_scale, context, true);
    if current_stats.pair_count == 0 {
        return (current_stats, 0, 0);
    }

    let seed_atoms = current_stats
        .pairs
        .iter()
        .flat_map(|pair| [pair.left, pair.right])
        .collect::<BTreeSet<_>>();
    let movable_depths =
        movable_overlap_shell_depths(atoms.len(), &context.adjacency, &seed_atoms, 4);
    let movable_atoms = movable_depths
        .iter()
        .enumerate()
        .filter_map(|(atom_idx, depth)| depth.map(|_| atom_idx))
        .collect::<Vec<_>>();
    if movable_atoms.is_empty() {
        return (current_stats, 0, 0);
    }

    let bond_targets = bonds
        .iter()
        .map(|&(left, right)| {
            (
                left,
                right,
                positions[right].sub(positions[left]).norm().max(1.0e-3),
            )
        })
        .collect::<Vec<_>>();
    let mut steps_executed = 0usize;
    let mut no_progress = 0usize;

    for _ in 0..16 {
        if current_stats.pair_count == 0 {
            break;
        }
        let max_step = (current_stats.max_overlap + 0.01).clamp(0.01, 0.08);
        let mut candidate_positions = positions.to_vec();
        let mut deltas = vec![Vec3::new(0.0, 0.0, 0.0); candidate_positions.len()];
        for pair in &current_stats.pairs {
            let left_movable = movable_depths[pair.left].is_some();
            let right_movable = movable_depths[pair.right].is_some();
            if !left_movable && !right_movable {
                continue;
            }
            let delta = candidate_positions[pair.left].sub(candidate_positions[pair.right]);
            let direction = if pair.distance > 1.0e-6 {
                delta.scale(1.0 / pair.distance)
            } else {
                fallback_pair_direction(pair.left, pair.right)
            };
            let push = (pair.overlap * 0.75 + 0.0025).min(max_step);
            match (left_movable, right_movable) {
                (true, true) => {
                    let shift = direction.scale(0.5 * push);
                    deltas[pair.left] = deltas[pair.left].add(shift);
                    deltas[pair.right] = deltas[pair.right].sub(shift);
                }
                (true, false) => {
                    deltas[pair.left] = deltas[pair.left].add(direction.scale(push));
                }
                (false, true) => {
                    deltas[pair.right] = deltas[pair.right].sub(direction.scale(push));
                }
                (false, false) => {}
            }
        }
        for &atom_idx in &movable_atoms {
            let delta = deltas[atom_idx];
            let norm = delta.norm();
            let applied = if norm > max_step && norm > 1.0e-8 {
                delta.scale(max_step / norm)
            } else {
                delta
            };
            candidate_positions[atom_idx] = candidate_positions[atom_idx].add(applied);
        }
        restore_targeted_steric_bonds(
            &mut candidate_positions,
            &bond_targets,
            &movable_depths,
            max_step * 0.35,
        );
        let candidate_stats =
            collect_overlap_stats(atoms, &candidate_positions, clash_scale, context, true);
        let improved = candidate_stats.pair_count < current_stats.pair_count
            || (candidate_stats.pair_count == current_stats.pair_count
                && candidate_stats.max_overlap + 1.0e-5 < current_stats.max_overlap);
        if improved || candidate_stats.pair_count == 0 {
            positions.copy_from_slice(&candidate_positions);
            current_stats = candidate_stats;
            steps_executed = steps_executed.saturating_add(1);
            no_progress = 0;
            continue;
        }
        no_progress += 1;
        if no_progress >= 2 {
            break;
        }
    }

    (current_stats, steps_executed, movable_atoms.len())
}

fn relax_targeted_steric_output(
    output: &mut PackOutput,
    relax: &RelaxSpec,
    raw_coordinates: Option<String>,
) -> RelaxReport {
    let steps_requested = relax.steps.unwrap_or(64);
    let step_scale = relax.step_scale.unwrap_or(0.25);
    let clash_scale = relax.clash_scale.unwrap_or(0.9);
    let initial_positions = output
        .atoms
        .iter()
        .map(|atom| atom.position)
        .collect::<Vec<_>>();
    let topology_context = build_relax_topology_context(output.atoms.len(), &output.bonds);
    let initial_overlap_stats = collect_overlap_stats(
        &output.atoms,
        &initial_positions,
        clash_scale,
        &topology_context,
        true,
    );
    let initial_max_clash = initial_overlap_stats.max_overlap;
    if output.atoms.is_empty() || steps_requested == 0 || initial_overlap_stats.pair_count == 0 {
        return RelaxReport {
            mode: "targeted_steric".into(),
            steps_requested,
            steps_executed: 0,
            initial_max_clash,
            final_max_clash: initial_max_clash,
            rms_displacement: 0.0,
            initial_overlap_pairs: initial_overlap_stats.pair_count,
            final_overlap_pairs: initial_overlap_stats.pair_count,
            movable_atom_count: Some(0),
            max_atom_displacement_angstrom: 0.0,
            fallback_mode: None,
            fallback_steps_requested: None,
            fallback_steps_executed: None,
            pre_fallback_max_clash: None,
            pre_fallback_overlap_pairs: None,
            initial_energy_kcal_mol: None,
            final_energy_kcal_mol: None,
            initial_max_force: None,
            final_max_force: None,
            accepted_line_search_steps: None,
            rejected_line_search_steps: None,
            termination_reason: None,
            raw_coordinates,
        };
    }

    let mut positions = initial_positions.clone();
    let mut steps_executed = 0usize;
    let mut current_stats = initial_overlap_stats.clone();
    let mut max_movable_atom_count = 0usize;
    for (shell_depth, stage_steps) in [
        (2usize, steps_requested),
        (3usize, (steps_requested / 2).max(16)),
        (4usize, (steps_requested / 2).max(16)),
    ] {
        if current_stats.pair_count == 0 {
            break;
        }
        let (stage_stats, stage_steps_executed, movable_atom_count) = run_targeted_steric_stage(
            &output.atoms,
            &output.bonds,
            &mut positions,
            clash_scale,
            step_scale,
            stage_steps,
            &topology_context,
            shell_depth,
        );
        current_stats = stage_stats;
        steps_executed = steps_executed.saturating_add(stage_steps_executed);
        max_movable_atom_count = max_movable_atom_count.max(movable_atom_count);
    }
    if current_stats.pair_count > 0 && current_stats.max_overlap <= 0.02 {
        let (_clearance_stats, clearance_steps, movable_atom_count) =
            clear_targeted_steric_residual_overlaps(
                &output.atoms,
                &output.bonds,
                &mut positions,
                clash_scale,
                &topology_context,
            );
        steps_executed = steps_executed.saturating_add(clearance_steps);
        max_movable_atom_count = max_movable_atom_count.max(movable_atom_count);
    }

    for (atom, position) in output.atoms.iter_mut().zip(positions.iter()) {
        atom.position = *position;
    }
    let final_overlap_stats = collect_overlap_stats(
        &output.atoms,
        &positions,
        clash_scale,
        &topology_context,
        false,
    );
    let rms_displacement = rms_atom_displacement(&initial_positions, &positions);
    RelaxReport {
        mode: "targeted_steric".into(),
        steps_requested,
        steps_executed,
        initial_max_clash,
        final_max_clash: final_overlap_stats.max_overlap,
        rms_displacement,
        initial_overlap_pairs: initial_overlap_stats.pair_count,
        final_overlap_pairs: final_overlap_stats.pair_count,
        movable_atom_count: Some(max_movable_atom_count),
        max_atom_displacement_angstrom: max_atom_displacement(&initial_positions, &positions),
        fallback_mode: None,
        fallback_steps_requested: None,
        fallback_steps_executed: None,
        pre_fallback_max_clash: None,
        pre_fallback_overlap_pairs: None,
        initial_energy_kcal_mol: None,
        final_energy_kcal_mol: None,
        initial_max_force: None,
        final_max_force: None,
        accepted_line_search_steps: None,
        rejected_line_search_steps: None,
        termination_reason: None,
        raw_coordinates,
    }
}

fn relax_built_output(
    output: &mut PackOutput,
    plan: &CompiledBuildPlan,
    step_length: f32,
    relax: &RelaxSpec,
    raw_coordinates: Option<String>,
) -> RelaxReport {
    match canonical_relax_mode(&relax.mode).unwrap_or("graph_spring") {
        "graph_spring" => {
            let stage_initial_positions = output
                .atoms
                .iter()
                .map(|atom| atom.position)
                .collect::<Vec<_>>();
            let primary =
                relax_graph_spring_output(output, plan, step_length, relax, raw_coordinates);
            if primary.final_overlap_pairs == 0 {
                return primary;
            }
            let fallback = relax_targeted_steric_output(
                output,
                &targeted_steric_fallback_spec(relax),
                primary.raw_coordinates.clone(),
            );
            let final_positions = output
                .atoms
                .iter()
                .map(|atom| atom.position)
                .collect::<Vec<_>>();
            merge_relax_report_with_followup_stage(
                primary,
                fallback,
                &stage_initial_positions,
                &final_positions,
            )
        }
        "targeted_steric" => relax_targeted_steric_output(output, relax, raw_coordinates),
        _ => relax_graph_spring_output(output, plan, step_length, relax, raw_coordinates),
    }
}

fn run_internal_cleanup_pipeline(
    built: &mut crate::polymer::PolymerBuiltArtifact,
    prepared: &PreparedBuildExecution,
    conformation_mode: &str,
) -> (Option<RelaxReport>, u64) {
    let mut cleanup_elapsed_ms = 0u64;
    let cleanup_report = if matches!(conformation_mode, "random_walk" | "ensemble") {
        let cleanup_started = Instant::now();
        let report = relax_built_output(
            &mut built.output,
            &prepared.compiled_plan,
            built.step_length_angstrom,
            &internal_solver_relax_spec(),
            None,
        );
        cleanup_elapsed_ms = cleanup_elapsed_ms.saturating_add(elapsed_ms(cleanup_started));
        Some(report)
    } else {
        None
    };
    if cleanup_report.is_some() {
        built.qc_report = recompute_build_qc_report(&built.output, &built.qc_context);
    }
    (cleanup_report, cleanup_elapsed_ms)
}

fn run_synthetic_topology_cleanup_stage(
    built: &mut crate::polymer::PolymerBuiltArtifact,
    source_coordinates: &str,
    source_charge_manifest: Option<&str>,
) -> (Option<RelaxReport>, u64, Option<String>) {
    match build_polymer_synthetic_uff_topology(
        built,
        Path::new(source_coordinates),
        source_charge_manifest.map(Path::new),
    ) {
        Ok(topology) => {
            let stage_initial_positions = output_positions(&built.output);
            let synthetic_started = Instant::now();
            let mut report = relax_synthetic_topology_output(
                &mut built.output,
                &topology,
                &built.qc_context,
                None,
            );
            let mut qc_after_synthetic =
                recompute_build_qc_report(&built.output, &built.qc_context);
            if report.final_overlap_pairs > 0 || qc_after_synthetic.severe_nonbonded_clash_count > 0
            {
                let fallback = relax_targeted_steric_output(
                    &mut built.output,
                    &RelaxSpec {
                        mode: "targeted_steric".into(),
                        steps: Some(96),
                        step_scale: Some(0.35),
                        clash_scale: Some(0.9),
                    },
                    None,
                );
                let final_positions = output_positions(&built.output);
                report = merge_relax_report_with_followup_stage(
                    report,
                    fallback,
                    &stage_initial_positions,
                    &final_positions,
                );
                qc_after_synthetic = recompute_build_qc_report(&built.output, &built.qc_context);
            }
            let elapsed = elapsed_ms(synthetic_started);
            built.qc_report = qc_after_synthetic;
            (Some(report), elapsed, None)
        }
        Err(err) => (
            None,
            0,
            Some(format!(
                "synthetic bonded cleanup skipped because synthetic topology could not be built: {err}"
            )),
        ),
    }
}

fn bond_adjacency(atom_count: usize, bonds: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); atom_count];
    for &(a, b) in bonds {
        if let Some(items) = adjacency.get_mut(a) {
            items.push(b);
        }
        if let Some(items) = adjacency.get_mut(b) {
            items.push(a);
        }
    }
    for items in &mut adjacency {
        items.sort_unstable();
        items.dedup();
    }
    adjacency
}

fn graph_angles(adjacency: &[Vec<usize>]) -> Vec<[usize; 3]> {
    let mut angles = BTreeSet::new();
    for (center, neighbors) in adjacency.iter().enumerate() {
        for left_idx in 0..neighbors.len() {
            for right_idx in (left_idx + 1)..neighbors.len() {
                let left = neighbors[left_idx];
                let right = neighbors[right_idx];
                let triple = if left <= right {
                    [left + 1, center + 1, right + 1]
                } else {
                    [right + 1, center + 1, left + 1]
                };
                angles.insert(triple);
            }
        }
    }
    angles.into_iter().collect()
}

fn graph_dihedrals(adjacency: &[Vec<usize>], bonds: &[(usize, usize)]) -> Vec<[usize; 4]> {
    let mut dihedrals = BTreeSet::new();
    for &(b, c) in bonds {
        let left_neighbors = adjacency.get(b).cloned().unwrap_or_default();
        let right_neighbors = adjacency.get(c).cloned().unwrap_or_default();
        for a in left_neighbors {
            if a == c {
                continue;
            }
            for d in &right_neighbors {
                if *d == b || *d == a {
                    continue;
                }
                let forward = [a + 1, b + 1, c + 1, *d + 1];
                let reverse = [*d + 1, c + 1, b + 1, a + 1];
                dihedrals.insert(if forward <= reverse { forward } else { reverse });
            }
        }
    }
    dihedrals.into_iter().collect()
}

fn graph_impropers(adjacency: &[Vec<usize>]) -> Vec<[usize; 4]> {
    let mut impropers = BTreeSet::new();
    for (center, neighbors) in adjacency.iter().enumerate() {
        if neighbors.len() < 3 {
            continue;
        }
        for left in 0..neighbors.len() {
            for mid in (left + 1)..neighbors.len() {
                for right in (mid + 1)..neighbors.len() {
                    let mut shell = [
                        neighbors[left] + 1,
                        neighbors[mid] + 1,
                        neighbors[right] + 1,
                    ];
                    shell.sort_unstable();
                    impropers.insert([shell[0], center + 1, shell[1], shell[2]]);
                }
            }
        }
    }
    impropers.into_iter().collect()
}

fn graph_exclusions(adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut exclusions = Vec::with_capacity(adjacency.len());
    for atom_idx in 0..adjacency.len() {
        let mut excluded = BTreeSet::new();
        for bonded in adjacency.get(atom_idx).cloned().unwrap_or_default() {
            excluded.insert(bonded + 1);
            for angle_partner in adjacency.get(bonded).cloned().unwrap_or_default() {
                if angle_partner != atom_idx {
                    excluded.insert(angle_partner + 1);
                }
                for dihedral_partner in adjacency.get(angle_partner).cloned().unwrap_or_default() {
                    if dihedral_partner != bonded && dihedral_partner != atom_idx {
                        excluded.insert(dihedral_partner + 1);
                    }
                }
            }
        }
        exclusions.push(excluded.into_iter().collect());
    }
    exclusions
}

fn compiled_cycle_basis(plan: Option<&CompiledBuildPlan>) -> Vec<TopologyGraphCycleRecord> {
    let Some(plan) = plan else {
        return Vec::new();
    };
    if plan.edges.len() + 1 <= plan.nodes.len() {
        return Vec::new();
    }
    let root_idx = plan
        .nodes
        .iter()
        .position(|node| node.node_id == plan.expanded_root_node_id)
        .unwrap_or(0);
    let mut adjacency = vec![Vec::<usize>::new(); plan.nodes.len()];
    let mut edge_id_by_pair = BTreeMap::new();
    for edge in &plan.edges {
        adjacency[edge.parent].push(edge.child);
        adjacency[edge.child].push(edge.parent);
        edge_id_by_pair.insert((edge.parent, edge.child), edge.edge_id.clone());
        edge_id_by_pair.insert((edge.child, edge.parent), edge.edge_id.clone());
    }
    let mut parent = vec![None; plan.nodes.len()];
    let mut seen = BTreeSet::from([root_idx]);
    let mut queue = std::collections::VecDeque::from([root_idx]);
    while let Some(node_idx) = queue.pop_front() {
        for neighbor in adjacency[node_idx].clone() {
            if seen.insert(neighbor) {
                parent[neighbor] = Some(node_idx);
                queue.push_back(neighbor);
            }
        }
    }
    let mut tree_pairs = BTreeSet::new();
    for (idx, parent_idx) in parent.iter().enumerate() {
        if let Some(parent_idx) = parent_idx {
            tree_pairs.insert((*parent_idx.min(&idx), (*parent_idx).max(idx)));
        }
    }
    let mut cycles = Vec::new();
    for (cycle_idx, edge) in plan.edges.iter().enumerate() {
        let pair = (edge.parent.min(edge.child), edge.parent.max(edge.child));
        if tree_pairs.contains(&pair) {
            continue;
        }
        let mut left_chain = Vec::new();
        let mut cursor = Some(edge.parent);
        while let Some(idx) = cursor {
            left_chain.push(idx);
            cursor = parent[idx];
        }
        let mut right_chain = Vec::new();
        let mut cursor = Some(edge.child);
        while let Some(idx) = cursor {
            right_chain.push(idx);
            cursor = parent[idx];
        }
        let lca = left_chain
            .iter()
            .find(|idx| right_chain.contains(idx))
            .copied()
            .unwrap_or(root_idx);
        let mut path = Vec::new();
        let mut cursor = edge.parent;
        while cursor != lca {
            path.push(cursor);
            cursor = parent[cursor].unwrap_or(lca);
        }
        path.push(lca);
        let mut tail = Vec::new();
        let mut cursor = edge.child;
        while cursor != lca {
            tail.push(cursor);
            cursor = parent[cursor].unwrap_or(lca);
        }
        tail.reverse();
        path.extend(tail);
        let mut edge_ids = Vec::new();
        for window in path.windows(2) {
            if let Some(edge_id) = edge_id_by_pair.get(&(window[0], window[1])) {
                edge_ids.push(edge_id.clone());
            }
        }
        edge_ids.push(edge.edge_id.clone());
        cycles.push(TopologyGraphCycleRecord {
            cycle_id: format!("cycle.{}", cycle_idx + 1),
            node_ids: path
                .iter()
                .map(|idx| plan.nodes[*idx].node_id.clone())
                .collect(),
            residue_ids: path.iter().map(|idx| idx + 1).collect(),
            edge_ids,
        });
    }
    cycles
}

fn motif_instances_from_plan(
    plan: Option<&CompiledBuildPlan>,
    source_bundle: &SourceBundle,
) -> Vec<TopologyGraphMotifInstance> {
    let Some(plan) = plan else {
        return Vec::new();
    };
    let mut grouped = BTreeMap::<String, Vec<(usize, &CompiledBuildNode)>>::new();
    for (idx, node) in plan.nodes.iter().enumerate() {
        if let Some(instance_id) = node.motif_instance_id.as_ref() {
            grouped
                .entry(instance_id.clone())
                .or_default()
                .push((idx, node));
        }
    }
    grouped
        .into_iter()
        .filter_map(|(instance_id, items)| {
            let motif_token = items
                .first()
                .and_then(|(_, node)| node.motif_token.clone())?;
            let request_node_id = items
                .first()
                .map(|(_, node)| node.request_node_id.clone())
                .unwrap_or_default();
            let motif = source_bundle.motif_library.get(&motif_token)?;
            let node_map = items
                .iter()
                .map(|(idx, node)| {
                    let local = node
                        .node_id
                        .split("::")
                        .last()
                        .unwrap_or(node.node_id.as_str())
                        .to_string();
                    (local, (*idx, node.node_id.clone()))
                })
                .collect::<BTreeMap<_, _>>();
            Some(TopologyGraphMotifInstance {
                motif_instance_id: instance_id,
                motif_token: motif_token.clone(),
                request_node_id,
                expanded_node_ids: items.iter().map(|(_, node)| node.node_id.clone()).collect(),
                expanded_resids: items.iter().map(|(idx, _)| idx + 1).collect(),
                exposed_ports: motif
                    .exposed_ports
                    .iter()
                    .filter_map(|(port_name, port)| {
                        node_map.get(&port.node_id).map(|(_, global_id)| {
                            TopologyGraphMotifPortBinding {
                                port_name: port_name.clone(),
                                node_id: global_id.clone(),
                                junction: port.junction.clone(),
                            }
                        })
                    })
                    .collect(),
            })
        })
        .collect()
}

fn port_policies_from_plan(
    plan: Option<&CompiledBuildPlan>,
    source_bundle: &SourceBundle,
) -> Vec<TopologyGraphPortPolicy> {
    let Some(plan) = plan else {
        return Vec::new();
    };
    let instances = plan
        .nodes
        .iter()
        .filter_map(|node| {
            node.motif_token
                .as_ref()
                .map(|motif_token| (node.request_node_id.clone(), motif_token.clone()))
        })
        .collect::<BTreeSet<_>>();
    let mut policies = Vec::new();
    for (request_node_id, motif_token) in instances {
        let Some(motif) = source_bundle.motif_library.get(&motif_token) else {
            continue;
        };
        let local_node_map = plan
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| {
                node.request_node_id == request_node_id
                    && node.motif_token.as_deref() == Some(motif_token.as_str())
            })
            .map(|(idx, node)| {
                (
                    node.node_id
                        .split("::")
                        .last()
                        .unwrap_or(node.node_id.as_str())
                        .to_string(),
                    (idx, node.node_id.clone()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for (port_name, port_spec) in &motif.exposed_ports {
            let Some((node_idx, compiled_node_id)) =
                local_node_map.get(&port_spec.node_id).cloned()
            else {
                continue;
            };
            policies.push(TopologyGraphPortPolicy {
                node_id: compiled_node_id,
                request_node_id: Some(request_node_id.clone()),
                resid: node_idx + 1,
                port_name: port_name.clone(),
                junction: port_spec.junction.clone(),
                port_class: port_spec.port_class.clone(),
                default_cap: port_spec.default_cap.clone(),
                allowed_caps: port_spec.allowed_caps.clone(),
            });
        }
    }
    policies
}

fn applied_caps_from_plan(plan: Option<&CompiledBuildPlan>) -> Vec<TopologyGraphAppliedCap> {
    plan.map(|plan| {
        plan.applied_caps
            .iter()
            .map(|cap| TopologyGraphAppliedCap {
                node_id: cap.node_id.clone(),
                request_node_id: Some(cap.request_node_id.clone()),
                resid: cap.resid,
                port_name: cap.port_name.clone(),
                junction: cap.junction.clone(),
                cap: cap.cap.clone(),
                application_source: cap.application_source.clone(),
                cap_node_id: cap.cap_node_id.clone(),
                cap_resid: cap.cap_resid,
            })
            .collect()
    })
    .unwrap_or_default()
}

fn open_ports_from_plan(
    plan: Option<&CompiledBuildPlan>,
    source_bundle: &SourceBundle,
) -> Vec<TopologyGraphOpenPort> {
    let Some(plan) = plan else {
        return Vec::new();
    };
    let mut used = BTreeMap::<usize, BTreeSet<String>>::new();
    for edge in &plan.edges {
        used.entry(edge.parent)
            .or_default()
            .insert(edge.parent_port.clone());
        used.entry(edge.child)
            .or_default()
            .insert(edge.child_port.clone());
    }
    let applied = plan
        .applied_caps
        .iter()
        .map(|cap| (cap.node_id.clone(), cap.port_name.clone()))
        .collect::<BTreeSet<_>>();
    port_policies_from_plan(Some(plan), source_bundle)
        .into_iter()
        .filter(|policy| {
            !used
                .get(&(policy.resid.saturating_sub(1)))
                .cloned()
                .unwrap_or_default()
                .contains(&policy.junction)
                && !applied.contains(&(policy.node_id.clone(), policy.port_name.clone()))
        })
        .map(|policy| TopologyGraphOpenPort {
            node_id: policy.node_id,
            request_node_id: policy.request_node_id,
            resid: policy.resid,
            port_name: policy.port_name,
            junction: policy.junction,
            port_class: policy.port_class,
        })
        .collect()
}

fn alignment_paths_from_plan(plan: Option<&CompiledBuildPlan>) -> Vec<TopologyGraphAlignmentPath> {
    let Some(plan) = plan else {
        return Vec::new();
    };
    if plan.nodes.len() < 2 {
        return Vec::new();
    }
    let mut adjacency = vec![Vec::<usize>::new(); plan.nodes.len()];
    for edge in &plan.edges {
        adjacency[edge.parent].push(edge.child);
        adjacency[edge.child].push(edge.parent);
    }
    let bfs = |start: usize| -> (usize, Vec<Option<usize>>, Vec<usize>) {
        let mut parent = vec![None; plan.nodes.len()];
        let mut depth = vec![usize::MAX; plan.nodes.len()];
        let mut queue = std::collections::VecDeque::from([start]);
        depth[start] = 0;
        while let Some(node_idx) = queue.pop_front() {
            for neighbor in adjacency[node_idx].clone() {
                if depth[neighbor] == usize::MAX {
                    depth[neighbor] = depth[node_idx] + 1;
                    parent[neighbor] = Some(node_idx);
                    queue.push_back(neighbor);
                }
            }
        }
        let farthest = depth
            .iter()
            .enumerate()
            .filter(|(_, value)| **value != usize::MAX)
            .max_by_key(|(_, value)| **value)
            .map(|(idx, _)| idx)
            .unwrap_or(start);
        (farthest, parent, depth)
    };
    let root_idx = plan
        .nodes
        .iter()
        .position(|node| node.node_id == plan.expanded_root_node_id)
        .unwrap_or(0);
    let (start, _, _) = bfs(root_idx);
    let (end, parent, _) = bfs(start);
    let mut order = vec![end];
    let mut cursor = end;
    while let Some(prev) = parent[cursor] {
        order.push(prev);
        cursor = prev;
    }
    order.reverse();
    if order.len() < 2 {
        return Vec::new();
    }
    vec![TopologyGraphAlignmentPath {
        kind: if plan.graph_has_cycle {
            "longest_residue_path".into()
        } else {
            "terminal_path".into()
        },
        residue_ids: order.iter().map(|idx| idx + 1).collect(),
        node_ids: order
            .iter()
            .map(|idx| plan.nodes[*idx].node_id.clone())
            .collect(),
    }]
}

fn topology_graph_value(
    request_id: &str,
    built: &crate::polymer::PolymerBuiltArtifact,
    sequence: &[String],
    source_bundle: &SourceBundle,
    target_mode: &str,
    realization_mode: &str,
    token_junctions: &BTreeMap<String, TokenJunctionSpec>,
    requested_termini: &TerminiPolicy,
    compiled_plan: Option<&CompiledBuildPlan>,
    relax_metadata: Option<TopologyGraphRelaxMetadata>,
) -> Value {
    let adjacency = bond_adjacency(built.output.atoms.len(), &built.output.bonds);
    let angles = graph_angles(&adjacency);
    let dihedrals = graph_dihedrals(&adjacency, &built.output.bonds);
    let impropers = graph_impropers(&adjacency);
    let exclusions = graph_exclusions(&adjacency);
    let mut residue_edges = BTreeSet::new();
    let mut residue_atom_links = Vec::new();
    for &(a, b) in &built.output.bonds {
        let Some(atom_a) = built.output.atoms.get(a) else {
            continue;
        };
        let Some(atom_b) = built.output.atoms.get(b) else {
            continue;
        };
        if atom_a.resid != atom_b.resid {
            let resid_a = atom_a.resid.min(atom_b.resid);
            let resid_b = atom_a.resid.max(atom_b.resid);
            residue_edges.insert((resid_a, resid_b));
            residue_atom_links.push(TopologyGraphInterResidueBond {
                a: a + 1,
                b: b + 1,
                resid_a: atom_a.resid,
                resid_b: atom_b.resid,
            });
        }
    }
    let branch_points = adjacency
        .iter()
        .enumerate()
        .filter(|(_, items)| items.len() > 2)
        .map(|(idx, items)| TopologyGraphBranchPoint {
            atom_index: idx + 1,
            degree: items.len(),
            resid: built.output.atoms.get(idx).map(|atom| atom.resid),
            atom_name: built.output.atoms.get(idx).map(|atom| atom.name.clone()),
        })
        .collect::<Vec<_>>();
    let residues = built
        .residue_resnames
        .iter()
        .enumerate()
        .map(|(idx, resname)| {
            let atom_indices = built
                .output
                .atoms
                .iter()
                .enumerate()
                .filter_map(|(atom_idx, atom)| {
                    if atom.resid == (idx + 1) as i32 {
                        Some(atom_idx + 1)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            TopologyGraphResidue {
                resid: idx + 1,
                resname: resname.clone(),
                node_id: compiled_plan
                    .and_then(|plan| plan.nodes.get(idx).map(|node| node.node_id.clone())),
                request_node_id: compiled_plan
                    .and_then(|plan| plan.nodes.get(idx).map(|node| node.request_node_id.clone())),
                sequence_token: sequence.get(idx).cloned(),
                token_kind: compiled_plan.and_then(|plan| {
                    plan.nodes
                        .get(idx)
                        .map(|node| warp_topology_graph::TokenKind::from(node.token_kind.clone()))
                }),
                source_token: compiled_plan
                    .and_then(|plan| plan.nodes.get(idx).map(|node| node.source_token.clone())),
                motif_instance_id: compiled_plan.and_then(|plan| {
                    plan.nodes
                        .get(idx)
                        .and_then(|node| node.motif_instance_id.clone())
                }),
                motif_token: compiled_plan.and_then(|plan| {
                    plan.nodes
                        .get(idx)
                        .and_then(|node| node.motif_token.clone())
                }),
                branch_depth: compiled_plan
                    .and_then(|plan| plan.nodes.get(idx).map(|node| node.branch_depth)),
                branch_path: compiled_plan
                    .and_then(|plan| plan.nodes.get(idx).map(|node| node.branch_path.clone())),
                atom_indices,
                ports: sequence
                    .get(idx)
                    .and_then(|token| token_junctions.get(token))
                    .map(|junction| {
                        vec![
                            TopologyGraphResiduePort {
                                name: "head".into(),
                                attach_atom: junction.head_attach_atom.clone(),
                                leaving_atoms: junction.head_leaving_atoms.clone(),
                            },
                            TopologyGraphResiduePort {
                                name: "tail".into(),
                                attach_atom: junction.tail_attach_atom.clone(),
                                leaving_atoms: junction.tail_leaving_atoms.clone(),
                            },
                        ]
                    })
                    .unwrap_or_default(),
            }
        })
        .collect::<Vec<_>>();
    let connection_definitions = compiled_plan
        .map(|plan| {
            plan.edges
                .iter()
                .map(|edge| TopologyGraphConnectionDefinition {
                    edge_id: Some(edge.edge_id.clone()),
                    parent_resid: edge.parent + 1,
                    child_resid: edge.child + 1,
                    parent_port: edge.parent_port.clone(),
                    child_port: edge.child_port.clone(),
                    parent_junction: edge.parent_junction.clone(),
                    child_junction: edge.child_junction.clone(),
                    parent_attach_atom: edge.parent_attach_atom.clone(),
                    child_attach_atom: edge.child_attach_atom.clone(),
                    parent_leaving_atoms: edge.parent_leaving_atoms.clone(),
                    child_leaving_atoms: edge.child_leaving_atoms.clone(),
                    bond_order: edge.bond_order,
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    serde_json::to_value(TopologyGraph {
        schema_version: TOPOLOGY_GRAPH_VERSION.into(),
        request_id: request_id.into(),
        bundle_id: source_bundle.bundle_id.clone(),
        build_plan: TopologyGraphBuildPlan {
            target_mode: target_mode.into(),
            realization_mode: realization_mode.into(),
            resolved_sequence: sequence.to_vec(),
            request_root_node_id: compiled_plan.map(|plan| plan.request_root_node_id.clone()),
            expanded_root_node_id: compiled_plan.map(|plan| plan.expanded_root_node_id.clone()),
            root_token: compiled_plan.map(|plan| plan.root_token.clone()),
            arm_count: compiled_plan.map(|plan| plan.arm_count),
            max_branch_depth: compiled_plan.map(|plan| plan.max_branch_depth),
            graph_has_cycle: compiled_plan.map(|plan| plan.graph_has_cycle),
            requested_termini: TopologyGraphTerminiRequest {
                head: requested_termini.head.clone(),
                tail: requested_termini.tail.clone(),
            },
        },
        atoms: built
            .output
            .atoms
            .iter()
            .enumerate()
            .map(|(idx, atom)| {
                let atom_type = topology_graph_atom_type(&atom.element);
                TopologyGraphAtom {
                    index: idx + 1,
                    name: atom.name.clone(),
                    element: atom.element.clone(),
                    resid: atom.resid,
                    resname: atom.resname.clone(),
                    charge_e: atom.charge,
                    mass: topology_graph_mass(&atom.element),
                    atom_type_index: topology_graph_atom_type_index(&atom.element),
                    amber_atom_type: atom_type.clone(),
                    lj_class: atom_type,
                    position: [atom.position.x, atom.position.y, atom.position.z],
                    neighbors: adjacency
                        .get(idx)
                        .cloned()
                        .unwrap_or_default()
                        .into_iter()
                        .map(|value| value + 1)
                        .collect(),
                }
            })
            .collect(),
        bonds: built
            .output
            .bonds
            .iter()
            .map(|(a, b)| TopologyGraphPair { a: a + 1, b: b + 1 })
            .collect(),
        angles: angles
            .iter()
            .map(|items| TopologyGraphAngle {
                a: items[0],
                b: items[1],
                c: items[2],
            })
            .collect(),
        dihedrals: dihedrals
            .iter()
            .map(|items| TopologyGraphTorsion {
                a: items[0],
                b: items[1],
                c: items[2],
                d: items[3],
            })
            .collect(),
        impropers: impropers
            .iter()
            .map(|items| TopologyGraphTorsion {
                a: items[0],
                b: items[1],
                c: items[2],
                d: items[3],
            })
            .collect(),
        exclusions: exclusions
            .iter()
            .enumerate()
            .map(|(idx, items)| TopologyGraphExclusion {
                atom_index: idx + 1,
                excluded_atoms: items.clone(),
            })
            .collect(),
        branch_points,
        residue_connections: residue_edges
            .iter()
            .map(|(a, b)| TopologyGraphPair {
                a: *a as usize,
                b: *b as usize,
            })
            .collect(),
        inter_residue_bonds: residue_atom_links,
        connection_definitions,
        nonbonded_typing: TopologyGraphNonbondedTyping {
            atom_type_indices: built
                .output
                .atoms
                .iter()
                .map(|atom| topology_graph_atom_type_index(&atom.element))
                .collect(),
            amber_atom_types: built
                .output
                .atoms
                .iter()
                .map(|atom| topology_graph_atom_type(&atom.element))
                .collect(),
            lj_classes: built
                .output
                .atoms
                .iter()
                .map(|atom| topology_graph_atom_type(&atom.element))
                .collect(),
        },
        residues,
        sequence: sequence.to_vec(),
        template_sequence_resnames: built.template_sequence_resnames.clone(),
        applied_residue_resnames: built.residue_resnames.clone(),
        motif_instances: motif_instances_from_plan(compiled_plan, source_bundle),
        cycle_basis: compiled_cycle_basis(compiled_plan),
        open_ports: open_ports_from_plan(compiled_plan, source_bundle),
        port_policies: port_policies_from_plan(compiled_plan, source_bundle),
        applied_caps: applied_caps_from_plan(compiled_plan),
        conformer_edges: compiled_plan
            .map(|plan| {
                plan.edges
                    .iter()
                    .map(|edge| TopologyGraphConformerEdge {
                        edge_id: edge.edge_id.clone(),
                        layout_mode: edge.layout_mode.clone().into(),
                        torsion_mode: edge.torsion_mode.clone().into(),
                        torsion_deg: edge.torsion_deg,
                        torsion_window_deg: edge.torsion_window_deg,
                        ring_mode: edge.ring_mode.clone().map(|value| value.into()),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default(),
        alignment_paths: alignment_paths_from_plan(compiled_plan),
        relax_metadata,
        metadata: json!({
            "motif_library_size": source_bundle.motif_library.len(),
            "token_junction_count": token_junctions.len(),
        }),
    })
    .unwrap_or_else(|_| json!({}))
}

pub fn schema_json(kind: &str) -> PackResult<String> {
    let value = match kind {
        "source" | "source_bundle" => serde_json::to_value(&schema_for!(SourceBundle))
            .map_err(|err| PackError::Parse(err.to_string()))?,
        "request" => serde_json::to_value(&schema_for!(BuildRequest))
            .map_err(|err| PackError::Parse(err.to_string()))?,
        "result" => serde_json::to_value(&schema_for!(ResultEnvelopeSchema))
            .map_err(|err| PackError::Parse(err.to_string()))?,
        "event" => serde_json::to_value(&schema_for!(RunEvent))
            .map_err(|err| PackError::Parse(err.to_string()))?,
        "build_manifest" => serde_json::to_value(&schema_for!(BuildManifestSchema))
            .map_err(|err| PackError::Parse(err.to_string()))?,
        "charge_manifest" => serde_json::to_value(&schema_for!(crate::polymer::ChargeManifest))
            .map_err(|err| PackError::Parse(err.to_string()))?,
        "topology_graph" => serde_json::to_value(&schema_for!(TopologyGraph))
            .map_err(|err| PackError::Parse(err.to_string()))?,
        _ => {
            return Err(PackError::Invalid(
                "schema target must be source_bundle, request, result, event, build_manifest, charge_manifest, or topology_graph".into(),
            ))
        }
    };
    serde_json::to_string_pretty(&value).map_err(|err| PackError::Parse(err.to_string()))
}

fn example_training_oligomer_pdb() -> &'static str {
    "ATOM      1  C1  HDA A   1       0.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  C2  RPT A   2       3.000   0.000   0.000  1.00  0.00           C\n\
ATOM      3  C3  TLA A   3       6.000   0.000   0.000  1.00  0.00           C\n\
END\n"
}

fn example_training_prmtop() -> AmberTopology {
    AmberTopology {
        atom_names: vec!["C1".into(), "C2".into(), "C3".into()],
        residue_labels: vec!["HDA".into(), "RPT".into(), "TLA".into()],
        residue_pointers: vec![1, 2, 3],
        atomic_numbers: vec![6, 6, 6],
        masses: vec![12.01, 12.01, 12.01],
        charges: vec![0.5, 1.0, 0.5],
        atom_type_indices: vec![1, 1, 1],
        amber_atom_types: vec!["CT".into(), "CT".into(), "CT".into()],
        radii: vec![1.7, 1.7, 1.7],
        screen: vec![0.8, 0.8, 0.8],
        bonds: vec![(0, 1), (1, 2)],
        bond_type_indices: vec![1, 1],
        bond_force_constants: vec![310.0],
        bond_equil_values: vec![1.53],
        angles: vec![[0, 1, 2]],
        angle_type_indices: vec![1],
        angle_force_constants: vec![55.0],
        angle_equil_values: vec![109.5],
        dihedrals: Vec::new(),
        dihedral_type_indices: Vec::new(),
        dihedral_force_constants: Vec::new(),
        dihedral_periodicities: Vec::new(),
        dihedral_phases: Vec::new(),
        scee_scale_factors: Vec::new(),
        scnb_scale_factors: Vec::new(),
        solty: vec![0.0],
        impropers: Vec::new(),
        improper_type_indices: Vec::new(),
        excluded_atoms: vec![vec![2], vec![1, 3], vec![2]],
        nonbonded_parm_index: vec![1],
        lennard_jones_acoef: vec![1.0],
        lennard_jones_bcoef: vec![0.5],
        lennard_jones_14_acoef: vec![0.8],
        lennard_jones_14_bcoef: vec![0.4],
        hbond_acoef: vec![0.0],
        hbond_bcoef: vec![0.0],
        hbcut: vec![0.0],
        tree_chain_classification: vec!["M".into(), "M".into(), "E".into()],
        join_array: vec![0, 0, 0],
        irotat: vec![0, 0, 0],
        solvent_pointers: Vec::new(),
        atoms_per_molecule: Vec::new(),
        box_dimensions: Vec::new(),
        radius_set: Some("modified Bondi radii".into()),
        ipol: 0,
    }
}

fn example_charge_manifest() -> Value {
    json!({
        "version": CHARGE_MANIFEST_VERSION,
        "head_charge_e": 0.5,
        "repeat_charge_e": 1.0,
        "tail_charge_e": 0.5,
    })
}

pub fn example_bundle() -> Value {
    json!({
        "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        "bundle_id": EXAMPLE_BUNDLE_ID,
        "training_context": {
            "mode": "oligomer_training",
            "training_oligomer_n": 3,
            "notes": "RESP/GAFF2 surrogate training. attach_atom keeps the atom that forms the new inter-unit bond, leaving_atoms are deleted before bonding, and anchor_atoms are local orientation hints near the junction."
        },
        "provenance": {},
        "unit_library": {
            "H": {
                "display_name": "Example head cap",
                "junctions": {"head": "example_head_cap", "tail": "example_head_cap"},
                "template_resname": "HDA"
            },
            "A": {
                "display_name": "Example repeat unit",
                "junctions": {"head": "example_head", "tail": "example_tail"},
                "template_resname": "RPT"
            },
            "B": {
                "display_name": "Example alternate repeat unit",
                "junctions": {"head": "example_head", "tail": "example_tail"},
                "template_resname": "RPT"
            },
            "T": {
                "display_name": "Example tail cap",
                "junctions": {"head": "example_tail_cap", "tail": "example_tail_cap"},
                "template_resname": "TLA"
            }
        },
        "motif_library": {
            "M2": {
                "display_name": "Example dimer motif",
                "root_node_id": "m1",
                "nodes": [
                    {"id": "m1", "token": "A"},
                    {"id": "m2", "token": "B"}
                ],
                "edges": [
                    {"from": "m1", "to": "m2", "from_junction": "tail", "to_junction": "head", "bond_order": 1}
                ],
                "exposed_ports": {
                    "head": {"node_id": "m1", "junction": "head"},
                    "tail": {"node_id": "m2", "junction": "tail"}
                }
            }
        },
        "junction_library": {
            "example_head_cap": {
                "attach_atom": {"scope": "unit", "selector": "name C1"},
                "leaving_atoms": [],
                "bond_order": 1,
                "anchor_atoms": [{"scope": "unit", "selector": "name C1"}],
                "notes": "Head-cap junction. attach_atom is retained and bonded to the next unit. anchor_atoms mark nearby atoms that preserve local junction orientation."
            },
            "example_head": {
                "attach_atom": {"scope": "unit", "selector": "name C2"},
                "leaving_atoms": [],
                "bond_order": 1,
                "anchor_atoms": [{"scope": "unit", "selector": "name C2"}],
                "notes": "Repeat-unit head junction. Use leaving_atoms for atoms removed before the new bond is created."
            },
            "example_tail": {
                "attach_atom": {"scope": "unit", "selector": "name C2"},
                "leaving_atoms": [],
                "bond_order": 1,
                "anchor_atoms": [{"scope": "unit", "selector": "name C2"}],
                "notes": "Repeat-unit tail junction. attach_atom names the retained bonding atom; anchor_atoms help orient the residue around that junction."
            },
            "example_tail_cap": {
                "attach_atom": {"scope": "unit", "selector": "name C3"},
                "leaving_atoms": [],
                "bond_order": 1,
                "anchor_atoms": [{"scope": "unit", "selector": "name C3"}],
                "notes": "Tail-cap junction. Prefer leaving_atoms, not attach_atom changes, when custom chemistry needs atoms deleted before bonding."
            }
        },
        "capabilities": {
            "supported_target_modes": ["linear_homopolymer", "linear_sequence_polymer", "block_copolymer", "random_copolymer", "star_polymer", "branched_polymer", "polymer_graph"],
            "supported_conformation_modes": ["extended", "random_walk"],
            "supported_tacticity_modes": ["inherit", "isotactic", "syndiotactic", "atactic"],
            "supported_termini_policies": ["default", "source_default"],
            "sequence_token_support": {
                "tokens": ["H", "A", "B", "T", "M2"],
                "allowed_adjacencies": [["H", "A"], ["H", "B"], ["A", "A"], ["A", "B"], ["B", "A"], ["B", "B"], ["A", "T"], ["B", "T"], ["H", "T"]]
            },
            "charge_transfer_supported": true
        },
        "artifacts": {
            "source_coordinates": EXAMPLE_SOURCE_COORDINATES,
            "source_topology_ref": EXAMPLE_SOURCE_TOPOLOGY,
            "forcefield_ref": EXAMPLE_FORCEFIELD_REF,
            "source_charge_manifest": EXAMPLE_SOURCE_CHARGE_MANIFEST
        },
        "charge_model": {}
    })
}

pub fn write_example_bundle(path: &str) -> Result<Value, String> {
    let bundle = example_bundle();
    let bundle_path = Path::new(path);
    if let Some(parent) = bundle_path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    fs::write(
        bundle_path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&bundle).map_err(|err| err.to_string())?
        ),
    )
    .map_err(|err| err.to_string())?;
    let outdir = bundle_path.parent().unwrap_or_else(|| Path::new("."));
    fs::write(
        outdir.join(EXAMPLE_SOURCE_COORDINATES),
        example_training_oligomer_pdb(),
    )
    .map_err(|err| err.to_string())?;
    write_minimal_prmtop(
        outdir
            .join(EXAMPLE_SOURCE_TOPOLOGY)
            .to_string_lossy()
            .as_ref(),
        &example_training_prmtop(),
    )
    .map_err(|err| err.to_string())?;
    fs::write(
        outdir.join(EXAMPLE_SOURCE_CHARGE_MANIFEST),
        format!(
            "{}\n",
            serde_json::to_string_pretty(&example_charge_manifest())
                .map_err(|err| err.to_string())?
        ),
    )
    .map_err(|err| err.to_string())?;
    fs::write(outdir.join(EXAMPLE_FORCEFIELD_REF), "<ForceField/>\n")
        .map_err(|err| err.to_string())?;
    Ok(bundle)
}

pub fn example_request_for_bundle(mode: &str, bundle_path: &str) -> Value {
    let bundle_id = load_bundle(Path::new(bundle_path))
        .map(|bundle| bundle.bundle_id)
        .unwrap_or_else(|_| EXAMPLE_BUNDLE_ID.into());
    let output_prefix = example_artifact_prefix(&bundle_id);
    let request_id = if bundle_id == EXAMPLE_BUNDLE_ID {
        "polymer-build-50mer-001".to_string()
    } else {
        format!("{output_prefix}-build-001")
    };
    let (target, realization) = match mode {
        "sequence" => (
            json!({
                "mode": "linear_sequence_polymer",
                "sequence": ["A", "B", "A"],
                "repeat_count": 4,
                "termini": {"head": "H", "tail": "T"},
                "stereochemistry": {"mode": "inherit"}
            }),
            json!({
                "conformation_mode": "random_walk",
                "seed": 12345
            }),
        ),
        "block" | "block_copolymer" => (
            json!({
                "mode": "block_copolymer",
                "blocks": [
                    {"token": "A", "count": 12},
                    {"token": "B", "count": 8}
                ],
                "termini": {"head": "default", "tail": "default"},
                "stereochemistry": {"mode": "inherit"}
            }),
            json!({
                "conformation_mode": "random_walk",
                "seed": 12345
            }),
        ),
        "random" | "random_copolymer" => (
            json!({
                "mode": "random_copolymer",
                "composition": {"A": 30, "B": 20},
                "total_units": 50,
                "termini": {"head": "default", "tail": "default"},
                "stereochemistry": {"mode": "atactic"}
            }),
            json!({
                "conformation_mode": "random_walk",
                "seed": 12345
            }),
        ),
        "random_walk" => (
            json!({
                "mode": "linear_homopolymer",
                "repeat_unit": "A",
                "n_repeat": 50,
                "termini": {"head": "default", "tail": "default"},
                "stereochemistry": {"mode": "syndiotactic"}
            }),
            json!({
                "conformation_mode": "random_walk",
                "seed": 12345
            }),
        ),
        "aligned" => (
            json!({
                "mode": "linear_homopolymer",
                "repeat_unit": "A",
                "n_repeat": 50,
                "termini": {"head": "default", "tail": "default"},
                "stereochemistry": {"mode": "syndiotactic"}
            }),
            json!({
                "conformation_mode": "aligned",
                "alignment_axis": "z",
                "seed": 12345
            }),
        ),
        "ensemble" => (
            json!({
                "mode": "linear_sequence_polymer",
                "sequence": ["A", "B"],
                "repeat_count": 10,
                "termini": {"head": "default", "tail": "default"},
                "stereochemistry": {"mode": "inherit"}
            }),
            json!({
                "conformation_mode": "ensemble",
                "ensemble_size": 4,
                "seed": 12345
            }),
        ),
        "extended" => (
            json!({
                "mode": "linear_homopolymer",
                "repeat_unit": "A",
                "n_repeat": 50,
                "termini": {"head": "default", "tail": "default"},
                "stereochemistry": {"mode": "syndiotactic"}
            }),
            json!({
                "conformation_mode": "extended",
                "seed": 12345
            }),
        ),
        "star" | "star_polymer" => (
            json!({
                "mode": "star_polymer",
                "core_token": "A",
                "core_junctions": ["head", "tail"],
                "arm_sequence": ["B", "A"],
                "arm_repeat_count": 3,
                "termini": {"head": "H", "tail": "T"},
                "stereochemistry": {"mode": "inherit"}
            }),
            json!({
                "conformation_mode": "aligned",
                "alignment_axis": "z",
                "seed": 12345
            }),
        ),
        "branched" | "branched_polymer" => (
            json!({
                "mode": "branched_polymer",
                "branch_tree": {
                    "token": "A",
                    "children": [
                        {
                            "parent_junction": "head",
                            "child_junction": "head",
                            "sequence": ["B", "A"],
                            "repeat_count": 2,
                            "child": {"token": "B", "children": []}
                        },
                        {
                            "parent_junction": "tail",
                            "child_junction": "head",
                            "sequence": ["A"],
                            "repeat_count": 3
                        }
                    ]
                },
                "termini": {"head": "H", "tail": "T"},
                "stereochemistry": {"mode": "inherit"}
            }),
            json!({
                "conformation_mode": "aligned",
                "alignment_axis": "z",
                "seed": 12345
            }),
        ),
        "graph" | "polymer_graph" => (
            json!({
                "mode": "polymer_graph",
                "graph_root": "n1",
                "graph_nodes": [
                    {"id": "n1", "token": "A"},
                    {"id": "n2", "token": "B"},
                    {"id": "n3", "token": "A"}
                ],
                "graph_edges": [
                    {"id": "e1", "from": "n1", "to": "n2", "from_junction": "tail", "to_junction": "head", "bond_order": 1},
                    {"id": "e2", "from": "n2", "to": "n3", "from_junction": "tail", "to_junction": "head", "bond_order": 1},
                ],
                "termini": {"head": "default", "tail": "default"},
                "stereochemistry": {"mode": "inherit"}
            }),
            json!({
                "conformation_mode": "random_walk",
                "seed": 12345
            }),
        ),
        _ => (
            json!({
                "mode": "linear_homopolymer",
                "repeat_unit": "A",
                "n_repeat": 50,
                "termini": {"head": "default", "tail": "default"},
                "stereochemistry": {"mode": "syndiotactic"}
            }),
            json!({
                "conformation_mode": "aligned",
                "alignment_axis": "z",
                "seed": 12345
            }),
        ),
    };
    let ensemble_manifest = if mode == "ensemble" {
        Some(format!("{output_prefix}.ensemble.json"))
    } else {
        None
    };
    json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "request_id": request_id,
        "source_ref": {
            "bundle_id": bundle_id,
            "bundle_path": bundle_path,
        },
        "target": target,
        "realization": {
            "conformation_mode": realization["conformation_mode"].clone(),
            "alignment_axis": realization.get("alignment_axis").cloned().unwrap_or(Value::Null),
            "seed": realization["seed"].clone(),
            "ensemble_size": realization.get("ensemble_size").cloned().unwrap_or(Value::Null),
            "qc_policy": "strict",
            "relax": {
                "mode": "graph_spring",
                "steps": 64,
                "step_scale": 0.25,
                "clash_scale": 0.9
            }
        },
        "conformer_policy": if mode == "graph" || mode == "polymer_graph" {
            Some(json!({
                "layout_mode": "mixed",
                "default_torsion": "trans",
                "branch_spread": "staggered",
                "ring_mode": "planar",
                "edge_overrides": [
                    {"edge_id": "e2", "torsion_mode": "fixed_deg", "torsion_deg": 60.0}
                ]
            }))
        } else {
            None
        },
        "port_cap_overrides": Vec::<Value>::new(),
        "validation": {
            "depth": "deep"
        },
        "artifacts": {
            "coordinates": format!("{output_prefix}.pdb"),
            "raw_coordinates": format!("{output_prefix}.raw.pdb"),
            "build_manifest": format!("{output_prefix}.build.json"),
            "charge_manifest": format!("{output_prefix}.charge.json"),
            "inpcrd": format!("{output_prefix}.inpcrd"),
            "topology": format!("{output_prefix}.prmtop"),
            "topology_graph": format!("{output_prefix}.topology.json"),
            "ensemble_manifest": ensemble_manifest
        }
    })
}

fn example_artifact_prefix(bundle_id: &str) -> String {
    if bundle_id == EXAMPLE_BUNDLE_ID {
        return "polymer_50mer".into();
    }
    let mut stem = sanitize_artifact_stem(bundle_id);
    for suffix in [
        "_param_bundle_v1",
        "_fixture_bundle_v1",
        "_bundle_v1",
        "_bundle",
    ] {
        if let Some(stripped) = stem.strip_suffix(suffix) {
            stem = stripped.to_string();
            break;
        }
    }
    if stem.is_empty() {
        stem = "polymer".into();
    }
    format!("{stem}_50mer")
}

fn sanitize_artifact_stem(value: &str) -> String {
    let mut stem = String::new();
    let mut last_was_separator = false;
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            stem.push(ch.to_ascii_lowercase());
            last_was_separator = false;
        } else if !last_was_separator && !stem.is_empty() {
            stem.push('_');
            last_was_separator = true;
        }
    }
    while stem.ends_with('_') {
        stem.pop();
    }
    stem
}

pub fn example_request(mode: &str) -> Value {
    example_request_for_bundle(mode, EXAMPLE_BUNDLE_PATH)
}

pub fn capabilities() -> Value {
    json!({
        "schema_versions": {
            "source_bundle": SOURCE_BUNDLE_SCHEMA_VERSION,
            "build_request": BUILD_SCHEMA_VERSION,
            "build_manifest": BUILD_MANIFEST_VERSION,
            "charge_manifest": CHARGE_MANIFEST_VERSION,
            "topology_graph": TOPOLOGY_GRAPH_VERSION,
        },
        "builder_version": env!("CARGO_PKG_VERSION"),
        "declared_target_modes": DECLARED_PLAN_MODES,
        "executable_target_modes": EXECUTABLE_PLAN_MODES,
        "supported_target_modes": SUPPORTED_TARGET_MODES,
        "supported_realization_modes": SUPPORTED_CONFORMATION_MODES,
        "base_conformation_modes": ["extended", "random_walk"],
        "realization_mode_map": {
            "extended": "extended",
            "random_walk": "random_walk",
            "aligned": "extended",
            "ensemble": "random_walk",
        },
        "supported_tacticity_modes": SUPPORTED_TACTICITY_MODES,
        "supported_termini_policies": SUPPORTED_TERMINI_POLICIES,
        "supported_validation_depths": ["shallow", "deep"],
        "default_validation_depth": "deep",
        "supported_validation_cache_modes": ["off", "record", "prefer", "require"],
        "default_validation_cache_mode": "off",
        "supported_qc_policies": ["strict", "salvage"],
        "artifact_outputs": [
            "coordinates",
            "raw_coordinates",
            "build_manifest",
            "charge_manifest",
            "inpcrd",
            "topology",
            "topology_graph",
            "ensemble_manifest",
            "forcefield_ref"
        ],
        "topology_outputs": ["inpcrd", "prmtop"],
        "artifact_contract": {
            "coordinates": "strict mode writes accepted coordinates after QC passes; salvage mode may emit non-final coordinates with acceptance_state=salvaged",
            "raw_coordinates": "optional pre-relax snapshot when realization.relax is enabled",
            "built_solute_debug": "*_built_solute.pdb may be staged during execution; treat it as a recovery/debug artifact, not the final contract output",
            "topology_transfer_requirement": "artifacts.topology auto-derives a .prmtop output; transferable source .prmtop and validated ffxml fallback take precedence when the training source is unreliable, otherwise warp-build emits a synthetic UFF-like minimizer topology",
            "forcefield_transfer_requirement": "artifacts.forcefield_ref requires bundle.artifacts.forcefield_ref to reference a transferable, non-placeholder .ffxml",
        },
        "junction_selector_semantics": {
            "attach_atom": "retained atom on the unit that forms the new inter-unit bond",
            "leaving_atoms": "atoms removed from the unit before the new bond is created",
            "anchor_atoms": "orientation hints near the junction; they do not create bonds by themselves",
        },
        "parameter_outputs": ["forcefield_ref"],
        "agent_contract": {
            "machine_readable_errors": true,
            "deterministic_seeded_output": true,
            "preflight_artifact_cache": true,
            "streaming_events": true,
            "preferred_handoff": ["build_manifest", "charge_manifest", "topology", "topology_graph"]
        },
        "schema_targets": ["source_bundle", "request", "result", "event", "build_manifest", "charge_manifest", "topology_graph"],
        "supports_named_termini_tokens": true,
        "supports_motif_tokens": true,
        "supports_conformer_policy": true,
        "supports_port_cap_overrides": true,
        "supports_relax_modes": ["graph_spring", "targeted_steric"],
        "topology_graph_features": [
            "atom_neighbors",
            "bonds",
            "angles",
            "dihedrals",
            "impropers",
            "exclusions",
            "nonbonded_typing",
            "residue_connections",
            "branch_points",
            "connection_definitions",
            "graph_node_ids",
            "graph_cycle_metadata",
            "motif_instances",
            "open_ports",
            "port_policies",
            "applied_caps",
            "conformer_edges",
            "alignment_paths",
            "relax_metadata"
        ],
    })
}

pub fn inspect_source_json(path: &str) -> (i32, Value) {
    match load_bundle(Path::new(path)) {
        Ok(bundle) => {
            let (artifact_errors, artifact_warnings) =
                inspect_source_artifacts(Path::new(path), &bundle);
            let status = if artifact_errors.is_empty() {
                "ok"
            } else {
                "error"
            };
            let code = if artifact_errors.is_empty() { 0 } else { 2 };
            let topology_supported = topology_transfer_supported(&bundle);
            let unit_tokens = bundle.unit_library.keys().cloned().collect::<Vec<_>>();
            let motif_tokens = bundle.motif_library.keys().cloned().collect::<Vec<_>>();
            (
                code,
                json!({
                    "status": status,
                    "bundle_id": bundle.bundle_id,
                    "schema_version": bundle.schema_version,
                    "training_context": bundle.training_context,
                    "supported_target_modes": bundle.capabilities.supported_target_modes,
                    "supported_conformation_modes": bundle.capabilities.supported_conformation_modes,
                    "supported_tacticity_modes": bundle.capabilities.supported_tacticity_modes,
                    "supported_termini_policies": bundle.capabilities.supported_termini_policies,
                    "sequence_token_support": bundle.capabilities.sequence_token_support,
                    "charge_transfer_supported": bundle.capabilities.charge_transfer_supported,
                    "topology_transfer_supported": topology_supported,
                    "artifact_validation": {
                        "valid": artifact_errors.is_empty(),
                        "checked": ["source_coordinates", "source_topology_ref", "source_charge_manifest", "forcefield_ref", "junction_selectors"],
                    },
                    "artifact_contract": {
                        "coordinates": "strict mode writes accepted coordinates after QC passes; salvage mode may emit non-final coordinates with acceptance_state=salvaged",
                        "raw_coordinates": "optional pre-relax snapshot when realization.relax is enabled",
                        "built_solute_debug": "*_built_solute.pdb may be staged during execution; treat it as a recovery/debug artifact, not the final contract output",
                        "topology_transfer_requirement": "artifacts.topology auto-derives a .prmtop output; transferable source .prmtop and validated ffxml fallback take precedence when the training source is unreliable, otherwise warp-build emits a synthetic UFF-like minimizer topology",
                        "forcefield_transfer_requirement": "artifacts.forcefield_ref requires bundle.artifacts.forcefield_ref to reference a transferable, non-placeholder .ffxml",
                    },
                    "junction_selector_semantics": {
                        "attach_atom": "retained atom on the unit that forms the new inter-unit bond",
                        "leaving_atoms": "atoms removed from the unit before the new bond is created",
                        "anchor_atoms": "orientation hints near the junction; they do not create bonds by themselves",
                    },
                    "unit_tokens": unit_tokens,
                    "motif_tokens": motif_tokens,
                    "artifacts": bundle.artifacts,
                    "errors": artifact_errors,
                    "warnings": artifact_warnings,
                }),
            )
        }
        Err(err) => (2, json!({"status": "error", "errors": [err]})),
    }
}

fn error_envelope(
    request_id: &str,
    errors: Vec<ErrorDetail>,
    warnings: Vec<ErrorDetail>,
    failure_diagnostics: Option<FailureDiagnostics>,
) -> ErrorEnvelope {
    ErrorEnvelope {
        schema_version: BUILD_SCHEMA_VERSION.into(),
        status: "error".into(),
        request_id: request_id.into(),
        errors,
        warnings,
        failure_diagnostics,
    }
}

fn run_failed_event(
    request_id: &str,
    error: ErrorDetail,
    diagnostics: Option<FailureDiagnostics>,
) -> RunEvent {
    RunEvent::RunFailed {
        request_id: request_id.into(),
        error,
        diagnostics,
    }
}

pub fn validate_request_json(text: &str) -> (i32, Value) {
    let req: BuildRequest = match parse_json(text) {
        Ok(req) => req,
        Err(err) => {
            return (
                2,
                json!({
                    "schema_version": BUILD_SCHEMA_VERSION,
                    "status": "error",
                    "valid": false,
                    "errors": [err],
                    "warnings": [],
                }),
            )
        }
    };
    let bundle = match load_bundle(Path::new(&req.source_ref.bundle_path)) {
        Ok(bundle) => bundle,
        Err(err) => {
            return (
                2,
                json!({
                    "schema_version": BUILD_SCHEMA_VERSION,
                    "status": "error",
                    "valid": false,
                    "errors": [err],
                    "warnings": [],
                }),
            )
        }
    };
    let normalized = match normalize_request_for_execution(&req, &bundle) {
        Ok(result) => result,
        Err(errors) => {
            return (
                2,
                json!({
                    "schema_version": BUILD_SCHEMA_VERSION,
                    "status": "error",
                    "valid": false,
                    "errors": errors,
                    "warnings": [],
                }),
            )
        }
    };
    match resolve_request_state(
        &normalized.request,
        &bundle,
        normalized.warnings,
        normalized.normalization,
    ) {
        Ok(resolved) => match validation_depth(&normalized.request) {
            Ok("deep") => match preflight_build(&normalized.request, &bundle, &resolved) {
                Ok(preflight) => {
                    let cache_mode = validation_cache_mode(&normalized.request).unwrap_or("off");
                    let mut preflight_cache = preflight_cache_summary(
                        &resolved.normalized_request,
                        &resolved.resolved_inputs,
                        cache_mode,
                        normalized.request.validation.cache_dir.as_deref(),
                        true,
                        if cache_mode == "off" {
                            "disabled"
                        } else {
                            "ready"
                        },
                        None,
                    );
                    if cache_mode != "off" {
                        match materialize_preflight_artifact_cache(
                            &normalized.request,
                            &resolved,
                            &preflight,
                            preflight_cache,
                        ) {
                            Ok(cache) => preflight_cache = cache,
                            Err(errors) => {
                                return (
                                    2,
                                    json!({
                                        "schema_version": BUILD_SCHEMA_VERSION,
                                        "status": "error",
                                        "valid": false,
                                        "errors": errors,
                                        "warnings": resolved.warnings,
                                    }),
                                );
                            }
                        }
                    }
                    (
                        0,
                        json!(ValidateEnvelope {
                            schema_version: BUILD_SCHEMA_VERSION.into(),
                            status: "ok".into(),
                            valid: true,
                            normalized_request: resolved.normalized_request,
                            resolved_inputs: resolved.resolved_inputs,
                            preflight_cache,
                            preflight,
                            warnings: resolved.warnings,
                        }),
                    )
                }
                Err(errors) => (
                    2,
                    json!({
                        "schema_version": BUILD_SCHEMA_VERSION,
                        "status": "error",
                        "valid": false,
                        "errors": errors,
                        "warnings": resolved.warnings,
                    }),
                ),
            },
            Ok("shallow") => {
                let cache_mode = validation_cache_mode(&normalized.request).unwrap_or("off");
                let preflight_cache = preflight_cache_summary(
                    &resolved.normalized_request,
                    &resolved.resolved_inputs,
                    cache_mode,
                    normalized.request.validation.cache_dir.as_deref(),
                    false,
                    if cache_mode == "off" {
                        "disabled"
                    } else {
                        "not_recorded"
                    },
                    Some("shallow_validation"),
                );
                (
                    0,
                    json!(ValidateEnvelope {
                        schema_version: BUILD_SCHEMA_VERSION.into(),
                        status: "ok".into(),
                        valid: true,
                        normalized_request: resolved.normalized_request,
                        resolved_inputs: resolved.resolved_inputs,
                        preflight_cache,
                        preflight: shallow_preflight_summary(),
                        warnings: resolved.warnings,
                    }),
                )
            }
            Ok(_) => unreachable!(),
            Err(err) => (
                2,
                json!({
                    "schema_version": BUILD_SCHEMA_VERSION,
                    "status": "error",
                    "valid": false,
                    "errors": [err],
                    "warnings": resolved.warnings,
                }),
            ),
        },
        Err(errors) => (
            2,
            json!({
                "schema_version": BUILD_SCHEMA_VERSION,
                "status": "error",
                "valid": false,
                "errors": errors,
                "warnings": [],
            }),
        ),
    }
}

pub fn run_request_json(text: &str, stream_ndjson: bool) -> (i32, Value) {
    let t0 = Instant::now();
    let raw_req: BuildRequest = match parse_json(text) {
        Ok(req) => req,
        Err(err) => {
            return (
                2,
                json!(error_envelope("warp-build", vec![err], vec![], None)),
            )
        }
    };
    emit(
        &RunEvent::RunStarted {
            request_id: raw_req.request_id.clone(),
        },
        stream_ndjson,
    );
    let bundle = match load_bundle(Path::new(&raw_req.source_ref.bundle_path)) {
        Ok(bundle) => bundle,
        Err(err) => {
            emit(
                &run_failed_event(&raw_req.request_id, err.clone(), None),
                stream_ndjson,
            );
            return (
                2,
                json!(error_envelope(&raw_req.request_id, vec![err], vec![], None)),
            );
        }
    };
    let normalized = match normalize_request_for_execution(&raw_req, &bundle) {
        Ok(result) => result,
        Err(errors) => {
            emit(
                &run_failed_event(&raw_req.request_id, errors[0].clone(), None),
                stream_ndjson,
            );
            return (
                2,
                json!(error_envelope(&raw_req.request_id, errors, vec![], None)),
            );
        }
    };
    let req = normalized.request;
    let resolved =
        match resolve_request_state(&req, &bundle, normalized.warnings, normalized.normalization) {
            Ok(resolved) => resolved,
            Err(errors) => {
                emit(
                    &run_failed_event(&req.request_id, errors[0].clone(), None),
                    stream_ndjson,
                );
                return (
                    2,
                    json!(error_envelope(&req.request_id, errors, vec![], None)),
                );
            }
        };

    emit(
        &RunEvent::SourceLoaded {
            request_id: req.request_id.clone(),
            training_oligomer_n: bundle.training_context.training_oligomer_n,
            bundle_id: bundle.bundle_id.clone(),
        },
        stream_ndjson,
    );
    let mut cache_warnings = Vec::new();
    match try_reuse_preflight_artifact_cache(&req, &resolved) {
        Ok(Some((code, value))) => {
            if let Some(path) = value
                .get("artifacts")
                .and_then(|artifacts| artifacts.get("build_manifest"))
                .and_then(Value::as_str)
            {
                emit(
                    &RunEvent::ManifestWritten {
                        request_id: req.request_id.clone(),
                        path: path.to_string(),
                    },
                    stream_ndjson,
                );
            }
            if let Ok(artifacts) = serde_json::from_value::<ArtifactRequest>(
                value.get("artifacts").cloned().unwrap_or_else(|| json!({})),
            ) {
                emit(
                    &RunEvent::RunCompleted {
                        request_id: req.request_id.clone(),
                        artifacts,
                    },
                    stream_ndjson,
                );
            }
            return (code, value);
        }
        Ok(None) => {
            if matches!(validation_cache_mode(&req), Ok("require")) {
                let cache = preflight_cache_summary(
                    &resolved.normalized_request,
                    &resolved.resolved_inputs,
                    "require",
                    req.validation.cache_dir.as_deref(),
                    true,
                    "lookup",
                    None,
                );
                let detail = to_error(
                    "E_PREFLIGHT_CACHE",
                    Some("/validation/cache_dir".into()),
                    format!(
                        "required preflight artifact cache was not found or did not match this request; expected record_path={:?} input_digest={}",
                        cache.record_path, cache.input_digest
                    ),
                );
                emit(
                    &run_failed_event(&req.request_id, detail.clone(), None),
                    stream_ndjson,
                );
                return (
                    4,
                    json!(error_envelope(&req.request_id, vec![detail], vec![], None)),
                );
            }
        }
        Err(err) => {
            if matches!(validation_cache_mode(&req), Ok("require")) {
                emit(
                    &run_failed_event(&req.request_id, err.clone(), None),
                    stream_ndjson,
                );
                return (
                    4,
                    json!(error_envelope(&req.request_id, vec![err], vec![], None)),
                );
            }
            cache_warnings.push(to_warning(
                "W_PREFLIGHT_CACHE_MISS",
                Some("/validation/cache_dir".into()),
                err.message,
            ));
        }
    }
    emit(
        &RunEvent::PhaseStarted {
            request_id: req.request_id.clone(),
            phase: "chain_growth".into(),
        },
        stream_ndjson,
    );

    let seed = resolved.seed;
    let (prepared, mut timings_ms) = match prepare_build_execution(&req, &bundle, seed) {
        Ok(prepared) => prepared,
        Err(err) => {
            return (
                2,
                json!(error_envelope(&req.request_id, vec![err], vec![], None)),
            );
        }
    };
    let parameter_source_decision = match assess_parameter_source(&prepared, &resolved) {
        Ok(value) => value,
        Err(errors) => {
            emit(
                &run_failed_event(&req.request_id, errors[0].clone(), None),
                stream_ndjson,
            );
            return (
                2,
                json!(error_envelope(&req.request_id, errors, vec![], None)),
            );
        }
    };
    let skip_strict_qc = parameter_source_decision.parameter_source != "synthetic_pdb";
    let salvage_qc = matches!(qc_policy(&req), Ok("salvage"));
    let n_repeat = prepared.compiled_sequence.len();
    emit(
        &RunEvent::ChainGrowthStarted {
            request_id: req.request_id.clone(),
            target_repeats: n_repeat,
            conformation_mode: req.realization.conformation_mode.clone(),
            seed,
        },
        stream_ndjson,
    );

    let tacticity = match req.target.stereochemistry.mode.as_str() {
        "inherit" => "training",
        other => other,
    };
    let source_coordinates = resolved.source_coordinates.clone();
    let source_charge_manifest = resolved.source_charge_manifest.clone();
    let source_topology_ref = resolved.source_topology_ref.clone();
    let source_forcefield_ref = resolved.source_forcefield_ref.clone();
    let inpcrd_path = resolved.artifacts.inpcrd.clone();
    let topology_graph_path = resolved.artifacts.topology_graph.clone();
    let raw_coordinates_path = resolved.artifacts.raw_coordinates.clone();
    let ensemble_manifest_path = resolved.artifacts.ensemble_manifest.clone();
    let topology_path = resolved.artifacts.topology.clone();
    let forcefield_output_path = resolved.artifacts.forcefield_ref.clone();
    let mut runtime_warnings = resolved.warnings.clone();
    runtime_warnings.extend(cache_warnings);
    let mut any_salvaged = false;
    let ensemble_size = if req.realization.conformation_mode == "ensemble" {
        req.realization.ensemble_size.unwrap_or(4)
    } else {
        1
    };
    let mut built_members = Vec::with_capacity(ensemble_size);
    for member_idx in 0..ensemble_size {
        let member_seed = seed.wrapping_add((member_idx as u64).wrapping_mul(7_919));
        let member_target_path = if ensemble_size == 1 {
            resolved.artifacts.coordinates.clone()
        } else {
            derived_member_path(&resolved.artifacts.coordinates, member_idx)
        };
        let member_raw_path = raw_coordinates_path.as_ref().map(|path| {
            if ensemble_size == 1 {
                path.clone()
            } else {
                derived_member_path(path, member_idx)
            }
        });
        let build_started = Instant::now();
        let build_result = build_polymer_graph(
            &source_coordinates,
            source_charge_manifest.as_deref(),
            source_topology_ref.as_deref(),
            bundle.training_context.training_oligomer_n,
            &prepared.graph_node_specs,
            &prepared.graph_edge_specs,
            prepared.graph_root_idx,
            &prepared.base_mode,
            tacticity,
            member_seed,
            !skip_strict_qc,
            &member_target_path,
        );
        timings_ms.build_graph = timings_ms
            .build_graph
            .saturating_add(elapsed_ms(build_started));
        let mut built = match build_result {
            Ok(built) => built,
            Err(err) => {
                let detail = to_error("E_RUNTIME_BUILD", None, err.to_string());
                let diagnostics = failure_diagnostics(
                    "build_graph",
                    None,
                    Some(&prepared.token_junctions),
                    overlap_status_summary(None, None, None),
                    None,
                    member_raw_path.clone(),
                );
                emit(
                    &run_failed_event(&req.request_id, detail.clone(), Some(diagnostics.clone())),
                    stream_ndjson,
                );
                return (
                    4,
                    json!(error_envelope(
                        &req.request_id,
                        vec![detail],
                        vec![],
                        Some(diagnostics),
                    )),
                );
            }
        };
        let cleanup_initial_positions = output_positions(&built.output);
        let defer_qc_until_synthetic_cleanup =
            parameter_source_decision.parameter_source == "synthetic_pdb";
        let (mut internal_cleanup_report, cleanup_elapsed_ms) = run_internal_cleanup_pipeline(
            &mut built,
            &prepared,
            &req.realization.conformation_mode,
        );
        timings_ms.solver_cleanup = timings_ms.solver_cleanup.saturating_add(cleanup_elapsed_ms);
        if internal_cleanup_report.is_some() && !defer_qc_until_synthetic_cleanup {
            if let Err(err) = ensure_build_qc_passes(&built.qc_report) {
                if let Some(report) = built.solver_report.as_mut() {
                    report.termination_reason = "qc_failed".into();
                    report.hard_fail_reason = Some(err.to_string());
                }
                let detail = to_error("E_RUNTIME_BUILD", None, err.to_string());
                let diagnostics = failure_diagnostics(
                    "solver_cleanup",
                    Some(&built.qc_report),
                    Some(&prepared.token_junctions),
                    overlap_status_summary(
                        Some(&built.output),
                        internal_cleanup_report.as_ref(),
                        None,
                    ),
                    Some(built.path.to_string_lossy().to_string()),
                    member_raw_path.clone(),
                );
                if req.realization.relax.is_none() {
                    if skip_strict_qc {
                        runtime_warnings.push(source_fallback_qc_warning(
                            &parameter_source_decision.parameter_source,
                            &detail,
                        ));
                    } else if salvage_qc {
                        any_salvaged = true;
                        runtime_warnings.push(salvage_warning(&detail));
                    } else {
                        emit(
                            &run_failed_event(
                                &req.request_id,
                                detail.clone(),
                                Some(diagnostics.clone()),
                            ),
                            stream_ndjson,
                        );
                        return (
                            4,
                            json!(error_envelope(
                                &req.request_id,
                                vec![detail],
                                qc_failure_warnings(&built.path, member_raw_path.as_deref()),
                                Some(diagnostics),
                            )),
                        );
                    }
                }
            }
        }
        let relax_report = if let Some(relax) = req.realization.relax.as_ref() {
            if let Some(raw_path) = member_raw_path.as_ref() {
                ensure_parent(raw_path).ok();
                if let Err(err) = fs::copy(&built.path, raw_path) {
                    let detail = to_error("E_OUTPUT_WRITE", None, err.to_string());
                    return (
                        4,
                        json!(error_envelope(&req.request_id, vec![detail], vec![], None)),
                    );
                }
            }
            let relax_started = Instant::now();
            let report = relax_built_output(
                &mut built.output,
                &prepared.compiled_plan,
                built.step_length_angstrom,
                relax,
                member_raw_path.clone(),
            );
            timings_ms.user_relax = timings_ms
                .user_relax
                .saturating_add(elapsed_ms(relax_started));
            Some(report)
        } else {
            None
        };
        if relax_report.is_some() && !defer_qc_until_synthetic_cleanup {
            built.qc_report = recompute_build_qc_report(&built.output, &built.qc_context);
            if let Err(err) = ensure_build_qc_passes(&built.qc_report) {
                let detail = to_error("E_RUNTIME_BUILD", None, err.to_string());
                let diagnostics = failure_diagnostics(
                    "user_relax",
                    Some(&built.qc_report),
                    Some(&prepared.token_junctions),
                    overlap_status_summary(
                        Some(&built.output),
                        internal_cleanup_report.as_ref(),
                        relax_report.as_ref(),
                    ),
                    Some(built.path.to_string_lossy().to_string()),
                    member_raw_path.clone(),
                );
                if skip_strict_qc {
                    runtime_warnings.push(source_fallback_qc_warning(
                        &parameter_source_decision.parameter_source,
                        &detail,
                    ));
                } else if salvage_qc {
                    any_salvaged = true;
                    runtime_warnings.push(salvage_warning(&detail));
                } else {
                    emit(
                        &run_failed_event(
                            &req.request_id,
                            detail.clone(),
                            Some(diagnostics.clone()),
                        ),
                        stream_ndjson,
                    );
                    return (
                        4,
                        json!(error_envelope(
                            &req.request_id,
                            vec![detail],
                            qc_failure_warnings(&built.path, member_raw_path.as_deref()),
                            Some(diagnostics),
                        )),
                    );
                }
            }
        } else if relax_report.is_some() {
            built.qc_report = recompute_build_qc_report(&built.output, &built.qc_context);
        }
        if req.realization.conformation_mode == "aligned" {
            rotate_output_along_axis(
                &mut built.output,
                req.realization.alignment_axis.as_deref().unwrap_or("z"),
            );
        }
        if parameter_source_decision.parameter_source == "synthetic_pdb" {
            let (synthetic_report, synthetic_elapsed_ms, synthetic_warning) =
                run_synthetic_topology_cleanup_stage(
                    &mut built,
                    &source_coordinates,
                    source_charge_manifest.as_deref(),
                );
            timings_ms.solver_cleanup = timings_ms
                .solver_cleanup
                .saturating_add(synthetic_elapsed_ms);
            if let Some(message) = synthetic_warning {
                runtime_warnings.push(to_warning("W_SYNTHETIC_CLEANUP", None, message));
            }
            if let Some(followup) = synthetic_report {
                let final_positions = output_positions(&built.output);
                internal_cleanup_report =
                    Some(if let Some(primary) = internal_cleanup_report.take() {
                        merge_relax_report_with_followup_stage(
                            primary,
                            followup,
                            &cleanup_initial_positions,
                            &final_positions,
                        )
                    } else {
                        followup
                    });
            }
            if let Err(err) = ensure_build_qc_passes(&built.qc_report) {
                if let Some(report) = built.solver_report.as_mut() {
                    report.termination_reason = "qc_failed".into();
                    report.hard_fail_reason = Some(err.to_string());
                }
                let detail = to_error("E_RUNTIME_BUILD", None, err.to_string());
                let diagnostics = failure_diagnostics(
                    "synthetic_topology_cleanup",
                    Some(&built.qc_report),
                    Some(&prepared.token_junctions),
                    overlap_status_summary(
                        Some(&built.output),
                        internal_cleanup_report.as_ref(),
                        relax_report.as_ref(),
                    ),
                    Some(built.path.to_string_lossy().to_string()),
                    member_raw_path.clone(),
                );
                if skip_strict_qc {
                    runtime_warnings.push(source_fallback_qc_warning(
                        &parameter_source_decision.parameter_source,
                        &detail,
                    ));
                } else if salvage_qc {
                    any_salvaged = true;
                    runtime_warnings.push(salvage_warning(&detail));
                } else {
                    emit(
                        &run_failed_event(
                            &req.request_id,
                            detail.clone(),
                            Some(diagnostics.clone()),
                        ),
                        stream_ndjson,
                    );
                    return (
                        4,
                        json!(error_envelope(
                            &req.request_id,
                            vec![detail],
                            qc_failure_warnings(&built.path, member_raw_path.as_deref()),
                            Some(diagnostics),
                        )),
                    );
                }
            }
        }
        for path in [
            built.path.to_string_lossy().to_string(),
            member_target_path.clone(),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        {
            let write_started = Instant::now();
            ensure_parent(&path).ok();
            if let Err(err) = write_output(
                &built.output,
                &OutputSpec {
                    path,
                    format: "pdb".to_string(),
                    scale: Some(1.0),
                },
                false,
                0.0,
                !built.output.bonds.is_empty(),
                false,
            ) {
                let detail = to_error("E_OUTPUT_WRITE", None, err.to_string());
                return (
                    4,
                    json!(error_envelope(&req.request_id, vec![detail], vec![], None)),
                );
            }
            timings_ms.artifact_write = timings_ms
                .artifact_write
                .saturating_add(elapsed_ms(write_started));
        }
        built_members.push((
            member_target_path,
            built,
            member_seed,
            internal_cleanup_report,
            relax_report,
        ));
    }
    let built = built_members[0].1.clone();
    let primary_coordinates = built_members[0].0.clone();
    let primary_internal_cleanup_report = built_members[0].3.clone();
    let primary_relax_report = built_members[0].4.clone();
    ensure_parent(&resolved.artifacts.coordinates).ok();
    if primary_coordinates != resolved.artifacts.coordinates {
        let write_started = Instant::now();
        if let Err(err) = fs::copy(&primary_coordinates, &resolved.artifacts.coordinates) {
            return (
                4,
                json!(error_envelope(
                    &req.request_id,
                    vec![to_error("E_OUTPUT_WRITE", None, err.to_string())],
                    vec![],
                    None,
                )),
            );
        }
        timings_ms.artifact_write = timings_ms
            .artifact_write
            .saturating_add(elapsed_ms(write_started));
    }
    ensure_parent(&topology_graph_path).ok();
    let topology_graph = topology_graph_value(
        &req.request_id,
        &built,
        &prepared.compiled_sequence,
        &bundle,
        &req.target.mode,
        &req.realization.conformation_mode,
        &prepared.token_junctions,
        &req.target.termini,
        Some(&prepared.compiled_plan),
        primary_relax_report
            .as_ref()
            .or(primary_internal_cleanup_report.as_ref())
            .as_ref()
            .map(|report| TopologyGraphRelaxMetadata {
                mode: report.mode.clone().into(),
                steps_requested: report.steps_requested,
                steps_executed: report.steps_executed,
                initial_max_clash: report.initial_max_clash,
                final_max_clash: report.final_max_clash,
                initial_overlap_pairs: report.initial_overlap_pairs,
                final_overlap_pairs: report.final_overlap_pairs,
                overlap_metric: OVERLAP_REPORT_METRIC.into(),
                rms_displacement: report.rms_displacement,
                raw_coordinates: report.raw_coordinates.clone(),
            }),
    );
    let write_started = Instant::now();
    if let Err(err) = fs::write(
        &topology_graph_path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&topology_graph).unwrap_or_else(|_| "{}".into())
        ),
    ) {
        return (
            4,
            json!(error_envelope(
                &req.request_id,
                vec![to_error(
                    "E_OUTPUT_WRITE",
                    Some("/artifacts/topology_graph".into()),
                    err.to_string()
                )],
                vec![],
                None,
            )),
        );
    }
    timings_ms.artifact_write = timings_ms
        .artifact_write
        .saturating_add(elapsed_ms(write_started));
    if ensemble_size > 1 {
        let ensemble_manifest_path = ensemble_manifest_path
            .clone()
            .unwrap_or_else(|| default_ensemble_manifest_path(&resolved.artifacts.coordinates));
        ensure_parent(&ensemble_manifest_path).ok();
        let ensemble_manifest = json!({
            "schema_version": ENSEMBLE_MANIFEST_VERSION,
            "request_id": req.request_id,
            "member_count": ensemble_size,
            "member_paths": built_members
                .iter()
                .enumerate()
                .map(|(idx, (path, _, seed, _, _))| json!({"member_index": idx + 1, "coordinates": path, "seed": seed}))
                .collect::<Vec<_>>(),
            "shared_artifacts": {
                "topology_graph": topology_graph_path,
                "topology": topology_path,
                "charge_manifest": resolved.artifacts.charge_manifest,
            },
        });
        let write_started = Instant::now();
        if let Err(err) = fs::write(
            &ensemble_manifest_path,
            format!(
                "{}\n",
                serde_json::to_string_pretty(&ensemble_manifest).unwrap_or_else(|_| "{}".into())
            ),
        ) {
            return (
                4,
                json!(error_envelope(
                    &req.request_id,
                    vec![to_error(
                        "E_OUTPUT_WRITE",
                        Some("/artifacts/ensemble_manifest".into()),
                        err.to_string()
                    )],
                    vec![],
                    None,
                )),
            );
        }
        timings_ms.artifact_write = timings_ms
            .artifact_write
            .saturating_add(elapsed_ms(write_started));
    }
    ensure_parent(&inpcrd_path).ok();
    let write_started = Instant::now();
    if let Err(err) = write_amber_inpcrd(&built.output, &inpcrd_path, 1.0) {
        return (
            4,
            json!(error_envelope(
                &req.request_id,
                vec![to_error("E_OUTPUT_WRITE", None, err.to_string())],
                vec![],
                None,
            )),
        );
    }
    timings_ms.artifact_write = timings_ms
        .artifact_write
        .saturating_add(elapsed_ms(write_started));
    let mut topology_mode = TopologyArtifactMode::None;
    let mut ffxml_topology_charge = None;
    if let Some(topology_path) = topology_path.as_ref() {
        ensure_parent(topology_path).ok();
        let write_started = Instant::now();
        let topology_result = match parameter_source_decision.parameter_source.as_str() {
            "source_topology_ref" => {
                let Some(source_topology_path) = source_topology_ref.as_ref() else {
                    return (
                        4,
                        json!(error_envelope(
                            &req.request_id,
                            vec![to_error(
                                "E_OUTPUT_WRITE",
                                Some("/artifacts/topology".into()),
                                "source topology fallback was selected but no source_topology_ref is resolved",
                            )],
                            vec![],
                            None,
                        )),
                    );
                };
                topology_mode = TopologyArtifactMode::SourceTransfer;
                write_polymer_prmtop_from_source(
                    &built,
                    Path::new(source_topology_path),
                    topology_path,
                )
                .map(|_| None)
            }
            "forcefield_ref" => {
                let Some(source_forcefield_path) = source_forcefield_ref.as_ref() else {
                    return (
                        4,
                        json!(error_envelope(
                            &req.request_id,
                            vec![to_error(
                                "E_OUTPUT_WRITE",
                                Some("/artifacts/topology".into()),
                                "forcefield fallback was selected but no forcefield_ref is resolved",
                            )],
                            vec![],
                            None,
                        )),
                    );
                };
                topology_mode = TopologyArtifactMode::ForcefieldRef;
                write_polymer_prmtop_from_ffxml(
                    &built,
                    Path::new(&source_coordinates),
                    source_charge_manifest.as_ref().map(Path::new),
                    Path::new(source_forcefield_path),
                    topology_path,
                )
                .map(|summary| Some(summary.net_charge_e))
            }
            _ => {
                topology_mode = TopologyArtifactMode::SyntheticUffLike;
                write_polymer_prmtop_synthetic_uff_like(
                    &built,
                    Path::new(&source_coordinates),
                    source_charge_manifest.as_ref().map(Path::new),
                    topology_path,
                )
                .map(|_| None)
            }
        };
        match topology_result {
            Ok(net_charge) => {
                ffxml_topology_charge = net_charge;
            }
            Err(err) => {
                return (
                    4,
                    json!(error_envelope(
                        &req.request_id,
                        vec![to_error(
                            "E_OUTPUT_WRITE",
                            Some("/artifacts/topology".into()),
                            err.to_string(),
                        )],
                        vec![],
                        None,
                    )),
                );
            }
        }
        timings_ms.artifact_write = timings_ms
            .artifact_write
            .saturating_add(elapsed_ms(write_started));
    }
    if let (Some(source_forcefield_ref), Some(forcefield_output_path)) = (
        source_forcefield_ref.as_ref(),
        forcefield_output_path.as_ref(),
    ) {
        let write_started = Instant::now();
        if let Err(err) = copy_forcefield_artifact(source_forcefield_ref, forcefield_output_path) {
            return (
                4,
                json!(error_envelope(&req.request_id, vec![err], vec![], None)),
            );
        }
        timings_ms.artifact_write = timings_ms
            .artifact_write
            .saturating_add(elapsed_ms(write_started));
    }
    emit(
        &RunEvent::ChainGrowthProgress {
            request_id: req.request_id.clone(),
            completed_repeats: n_repeat,
            target_repeats: n_repeat,
            progress_pct: 100.0,
        },
        stream_ndjson,
    );
    emit(
        &RunEvent::ChainGrowthCompleted {
            request_id: req.request_id.clone(),
            target_repeats: n_repeat,
            elapsed_ms: t0.elapsed().as_millis() as u64,
        },
        stream_ndjson,
    );

    let (net_charge, charge_derivation) = if let Some(path) = source_charge_manifest.as_ref() {
        let manifest = match load_charge_manifest(Path::new(path)) {
            Ok(m) => m,
            Err(_) => {
                return (
                    4,
                    json!(error_envelope(
                        &req.request_id,
                        vec![to_error(
                            "E_CHARGE_HANDOFF",
                            None,
                            "failed to load source charge manifest"
                        )],
                        vec![],
                        None,
                    )),
                )
            }
        };
        let estimate = compute_sequence_polymer_net_charge_from_source(
            &manifest,
            Path::new(&source_coordinates),
            &built.template_sequence_resnames,
            bundle.training_context.training_oligomer_n,
        );
        match estimate {
            Ok(value) => (
                value.net_charge_e,
                value
                    .source
                    .unwrap_or_else(|| "source_bundle_transfer".to_string()),
            ),
            Err(err) => {
                return (
                    4,
                    json!(error_envelope(
                        &req.request_id,
                        vec![to_error("E_CHARGE_HANDOFF", None, err.to_string())],
                        vec![],
                        None,
                    )),
                )
            }
        }
    } else if let Some(topology_ref) = bundle.artifacts.source_topology_ref.as_ref() {
        let topology_path =
            resolve_relative_path(Path::new(&req.source_ref.bundle_path), topology_ref);
        match compute_sequence_polymer_net_charge_from_prmtop(
            Path::new(&source_coordinates),
            &built.template_sequence_resnames,
            Path::new(&topology_path),
            bundle.training_context.training_oligomer_n,
        ) {
            Ok(value) => (
                value.net_charge_e,
                value
                    .source
                    .unwrap_or_else(|| "prmtop.total_charge".to_string()),
            ),
            Err(err) => {
                return (
                    4,
                    json!(error_envelope(
                        &req.request_id,
                        vec![to_error("E_CHARGE_HANDOFF", None, err.to_string())],
                        vec![],
                        None,
                    )),
                )
            }
        }
    } else if let Some(net_charge) = ffxml_topology_charge {
        (Some(net_charge), "forcefield_ref.atom_charges".to_string())
    } else {
        (None, "unavailable".to_string())
    };
    let acceptance_state = acceptance_state(any_salvaged);
    let (handoff_level, limitations) = handoff_level_and_limitations(
        topology_mode,
        net_charge,
        any_salvaged,
        &parameter_source_decision,
        &req.realization.conformation_mode,
        &req.target.mode,
        &built.qc_report,
    );

    ensure_parent(&resolved.artifacts.charge_manifest).ok();
    let charge_manifest = json!({
        "schema_version": CHARGE_MANIFEST_VERSION,
        "solute_path": resolved.artifacts.coordinates,
        "source_topology_ref": bundle.artifacts.source_topology_ref,
        "target_topology_ref": topology_path,
        "forcefield_ref": forcefield_output_path,
        "charge_derivation": charge_derivation,
        "net_charge_e": net_charge,
        "atom_count": built.output.atoms.len(),
    });
    if let Err(err) = fs::write(
        &resolved.artifacts.charge_manifest,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&charge_manifest).unwrap_or_else(|_| "{}".into())
        ),
    ) {
        return (
            4,
            json!(error_envelope(
                &req.request_id,
                vec![to_error("E_OUTPUT_WRITE", None, err.to_string())],
                vec![],
                None,
            )),
        );
    }

    ensure_parent(&resolved.artifacts.build_manifest).ok();
    let applied_head_terminus = AppliedTerminusSummary {
        requested_policy: req.target.termini.head.clone(),
        resolved_token: built.sequence_labels.first().cloned(),
        template_resname: built.template_sequence_resnames.first().cloned(),
        applied_resname: built.residue_resnames.first().cloned(),
    };
    let applied_tail_terminus = AppliedTerminusSummary {
        requested_policy: req.target.termini.tail.clone(),
        resolved_token: built.sequence_labels.last().cloned(),
        template_resname: built.template_sequence_resnames.last().cloned(),
        applied_resname: built.residue_resnames.last().cloned(),
    };
    let artifact_digests = BuildArtifactDigests {
        coordinates: sha256_file(Path::new(&resolved.artifacts.coordinates)).ok(),
        raw_coordinates: raw_coordinates_path
            .as_ref()
            .and_then(|path| sha256_file(Path::new(path)).ok()),
        charge_manifest: sha256_file(Path::new(&resolved.artifacts.charge_manifest)).ok(),
        inpcrd: sha256_file(Path::new(&inpcrd_path)).ok(),
        topology: topology_path
            .as_ref()
            .and_then(|path| sha256_file(Path::new(path)).ok()),
        topology_graph: sha256_file(Path::new(&topology_graph_path)).ok(),
        ensemble_manifest: ensemble_manifest_path
            .as_ref()
            .and_then(|path| sha256_file(Path::new(path)).ok()),
        forcefield_ref: forcefield_output_path
            .as_ref()
            .and_then(|path| sha256_file(Path::new(path)).ok()),
    };
    let md_ready_handoff = MdReadyHandoff {
        version: "warp-md.polymer-md-handoff.v1".into(),
        coordinates: HandoffCoordinates {
            path: resolved.artifacts.coordinates.clone(),
            format: "pdb-strict".into(),
            strict_columns: true,
            write_conect: true,
            has_cryst1: false,
        },
        topology_graph: topology_graph_path.clone(),
        topology: topology_path.clone(),
        charge_manifest: resolved.artifacts.charge_manifest.clone(),
        acceptance_state: acceptance_state.clone(),
        handoff_level: handoff_level.clone(),
        limitations: limitations.clone(),
        source_structure_path: source_coordinates.clone(),
        source_topology_path: source_topology_ref.clone(),
        source_charge_manifest_path: source_charge_manifest.clone(),
        sequence_tokens: built.sequence_labels.clone(),
        template_sequence_resnames: built.template_sequence_resnames.clone(),
        applied_residue_resnames: built.residue_resnames.clone(),
        copy_count: 1,
        chain_instance_mapping: vec![HandoffChainInstanceMapping {
            copy_index: 1,
            source_chain_indices: vec![1],
            packed_chain_indices: vec![1],
        }],
        forcefield_ref: forcefield_output_path.clone(),
    };
    let manifest = BuildManifestSchema {
        schema_version: BUILD_MANIFEST_VERSION.into(),
        version: None,
        request_id: req.request_id.clone(),
        normalized_request: resolved.normalized_request.clone(),
        resolved_inputs: resolved.resolved_inputs.clone(),
        source_bundle: BuildManifestSourceBundle {
            bundle_id: bundle.bundle_id.clone(),
            bundle_path: req.source_ref.bundle_path.clone(),
            bundle_digest: resolved.bundle_digest.clone(),
            training_context: bundle.training_context.clone(),
        },
        target: req.target.clone(),
        realization: BuildManifestRealization {
            conformation_mode: req.realization.conformation_mode.clone(),
            seed,
            seed_policy: resolved.seed_policy.clone(),
            relax: req.realization.relax.clone(),
            qc_policy: req.realization.qc_policy.clone(),
        },
        artifacts: BuildManifestArtifacts {
            coordinates: resolved.artifacts.coordinates.clone(),
            raw_coordinates: raw_coordinates_path.clone(),
            charge_manifest: resolved.artifacts.charge_manifest.clone(),
            inpcrd: inpcrd_path.clone(),
            topology: topology_path.clone(),
            topology_graph: topology_graph_path.clone(),
            ensemble_manifest: ensemble_manifest_path.clone(),
            forcefield_ref: forcefield_output_path.clone(),
        },
        artifact_digests,
        md_ready_handoff,
        summary: BuildManifestSummary {
            atom_count: built.output.atoms.len(),
            total_repeat_units: requested_n_repeat(resolved.normalization.as_ref(), n_repeat),
            total_residues: n_repeat,
            net_charge_e: net_charge,
            resolved_sequence: built.sequence_labels.clone(),
            template_sequence_resnames: built.template_sequence_resnames.clone(),
            applied_residue_resnames: built.residue_resnames.clone(),
            request_root_node_id: prepared.compiled_plan.request_root_node_id.clone(),
            expanded_root_node_id: prepared.compiled_plan.expanded_root_node_id.clone(),
            graph_has_cycle: prepared.compiled_plan.graph_has_cycle,
            applied_termini: AppliedTerminiSummary {
                head: applied_head_terminus,
                tail: applied_tail_terminus,
            },
            applied_caps: prepared
                .compiled_plan
                .applied_caps
                .iter()
                .map(|cap| AppliedCapSummary {
                    node_id: cap.node_id.clone(),
                    request_node_id: cap.request_node_id.clone(),
                    port_name: cap.port_name.clone(),
                    cap: cap.cap.clone(),
                    application_source: cap.application_source.clone(),
                    cap_node_id: cap.cap_node_id.clone(),
                })
                .collect::<Vec<_>>(),
            bond_count: built.output.bonds.len(),
            realization_mode: req.realization.conformation_mode.clone(),
            ensemble_size,
            relax: primary_relax_report.clone(),
            solver_cleanup: primary_internal_cleanup_report.clone(),
            solver: built.solver_report.clone(),
            applied_junctions: applied_junctions_value(&prepared.token_junctions),
            timings_ms: timings_ms.clone(),
            qc: built.qc_report.clone(),
            overlap_status: overlap_status_summary(
                Some(&built.output),
                primary_internal_cleanup_report.as_ref(),
                primary_relax_report.as_ref(),
            ),
            acceptance_state: acceptance_state.clone(),
            handoff_level: handoff_level.clone(),
            limitations: limitations.clone(),
            parameter_source_decision: parameter_source_decision.clone(),
        },
        provenance: BuildManifestProvenance {
            schema_version: BUILD_MANIFEST_VERSION.into(),
            builder_version: env!("CARGO_PKG_VERSION").into(),
            binary_version: env!("CARGO_PKG_VERSION").into(),
            algorithm_version: format!("{}.v1", req.realization.conformation_mode),
            topology_transfer_mode: match topology_mode {
                TopologyArtifactMode::SourceTransfer => "residue_filtered_with_bonds".into(),
                TopologyArtifactMode::ForcefieldRef => "forcefield_ref".into(),
                TopologyArtifactMode::SyntheticUffLike => "synthetic_uff_like".into(),
                TopologyArtifactMode::None => "none".into(),
            },
            source_bundle_path: req.source_ref.bundle_path.clone(),
            source_bundle_digest: resolved.bundle_digest.clone(),
            target_normalization: resolved.normalization.clone(),
            parameter_source_decision: parameter_source_decision.clone(),
            build_metadata: build_metadata_summary(),
        },
        warnings: runtime_warnings.clone(),
    };
    let write_started = Instant::now();
    if let Err(err) = fs::write(
        &resolved.artifacts.build_manifest,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&manifest).unwrap_or_else(|_| "{}".into())
        ),
    ) {
        return (
            4,
            json!(error_envelope(
                &req.request_id,
                vec![to_error("E_OUTPUT_WRITE", None, err.to_string())],
                vec![],
                None,
            )),
        );
    }
    timings_ms.artifact_write = timings_ms
        .artifact_write
        .saturating_add(elapsed_ms(write_started));

    emit(
        &RunEvent::ManifestWritten {
            request_id: req.request_id.clone(),
            path: resolved.artifacts.build_manifest.clone(),
        },
        stream_ndjson,
    );
    emit(
        &RunEvent::PhaseCompleted {
            request_id: req.request_id.clone(),
            phase: "chain_growth".into(),
        },
        stream_ndjson,
    );
    emit(
        &RunEvent::RunCompleted {
            request_id: req.request_id.clone(),
            artifacts: ArtifactRequest {
                coordinates: resolved.artifacts.coordinates.clone(),
                raw_coordinates: raw_coordinates_path.clone(),
                build_manifest: resolved.artifacts.build_manifest.clone(),
                charge_manifest: resolved.artifacts.charge_manifest.clone(),
                inpcrd: Some(inpcrd_path.clone()),
                topology: topology_path.clone(),
                topology_graph: Some(topology_graph_path.clone()),
                ensemble_manifest: ensemble_manifest_path.clone(),
                forcefield_ref: forcefield_output_path.clone(),
            },
        },
        stream_ndjson,
    );

    let result_status = if acceptance_state == "salvaged" {
        "salvaged"
    } else {
        "ok"
    };
    let result_code = if acceptance_state == "salvaged" { 3 } else { 0 };

    (
        result_code,
        json!(SuccessEnvelope {
            schema_version: BUILD_SCHEMA_VERSION.into(),
            status: result_status.into(),
            request_id: req.request_id,
            artifacts: ArtifactRequest {
                coordinates: resolved.artifacts.coordinates,
                raw_coordinates: raw_coordinates_path,
                build_manifest: resolved.artifacts.build_manifest,
                charge_manifest: resolved.artifacts.charge_manifest,
                inpcrd: Some(inpcrd_path),
                topology: topology_path,
                topology_graph: Some(topology_graph_path),
                ensemble_manifest: ensemble_manifest_path,
                forcefield_ref: forcefield_output_path,
            },
            summary: RunSummary {
                build_mode: req.target.mode.clone(),
                n_repeat: requested_n_repeat(resolved.normalization.as_ref(), n_repeat),
                atom_count: built.output.atoms.len(),
                total_repeat_units: requested_n_repeat(resolved.normalization.as_ref(), n_repeat),
                total_residues: n_repeat,
                conformation_mode: req.realization.conformation_mode.clone(),
                seed,
                ensemble_size,
                topology_graph_version: TOPOLOGY_GRAPH_VERSION.into(),
                qc: built.qc_report.clone(),
                solver: built.solver_report.clone(),
                timings_ms,
                solver_cleanup: primary_internal_cleanup_report.clone(),
                relax: primary_relax_report.clone(),
                overlap_status: overlap_status_summary(
                    Some(&built.output),
                    primary_internal_cleanup_report.as_ref(),
                    primary_relax_report.as_ref(),
                ),
                acceptance_state,
                handoff_level,
                limitations,
                parameter_source_decision,
            },
            warnings: runtime_warnings,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_structure::{AtomRecord, AtomRecordKind, PackOutput, Vec3};

    fn test_atom(name: &str, element: &str, resid: i32, position: Vec3) -> AtomRecord {
        AtomRecord {
            record_kind: AtomRecordKind::Atom,
            name: name.into(),
            element: element.into(),
            resname: "TST".into(),
            resid,
            chain: 'A',
            segid: String::new(),
            charge: 0.0,
            position,
            mol_id: 1,
            pdb_metadata: None,
        }
    }

    fn test_plan() -> CompiledBuildPlan {
        CompiledBuildPlan {
            nodes: vec![CompiledBuildNode {
                node_id: "n1".into(),
                request_node_id: "n1".into(),
                token: "A".into(),
                token_kind: "unit".into(),
                source_token: "A".into(),
                motif_instance_id: None,
                motif_token: None,
                template_resname: "TST".into(),
                applied_resname: "TST".into(),
                branch_depth: 0,
                branch_path: "0".into(),
            }],
            edges: Vec::new(),
            applied_caps: Vec::new(),
            request_root_node_id: "n1".into(),
            expanded_root_node_id: "n1".into(),
            root_token: "A".into(),
            arm_count: 1,
            max_branch_depth: 0,
            graph_has_cycle: false,
        }
    }

    fn test_synthetic_topology() -> AmberTopology {
        AmberTopology {
            atom_names: vec!["C1".into(), "C2".into(), "C3".into(), "C4".into()],
            residue_labels: vec!["TST".into()],
            residue_pointers: vec![1],
            atomic_numbers: vec![6, 6, 6, 6],
            masses: vec![12.011; 4],
            charges: vec![0.0; 4],
            atom_type_indices: vec![1; 4],
            amber_atom_types: vec!["CT".into(); 4],
            radii: vec![1.7; 4],
            screen: vec![0.72; 4],
            bonds: vec![(0, 1), (1, 2), (2, 3)],
            bond_type_indices: vec![1, 1, 1],
            bond_force_constants: vec![320.0],
            bond_equil_values: vec![1.54],
            angles: vec![[0, 1, 2], [1, 2, 3]],
            angle_type_indices: vec![1, 1],
            angle_force_constants: vec![70.0],
            angle_equil_values: vec![1.9106332],
            dihedrals: vec![[0, 1, 2, 3]],
            dihedral_type_indices: vec![1],
            dihedral_force_constants: vec![1.0],
            dihedral_periodicities: vec![3.0],
            dihedral_phases: vec![0.0],
            scee_scale_factors: vec![1.2],
            scnb_scale_factors: vec![2.0],
            solty: vec![0.0],
            impropers: Vec::new(),
            improper_type_indices: Vec::new(),
            excluded_atoms: vec![Vec::new(); 4],
            nonbonded_parm_index: vec![1],
            lennard_jones_acoef: vec![1.0],
            lennard_jones_bcoef: vec![1.0],
            lennard_jones_14_acoef: vec![1.0],
            lennard_jones_14_bcoef: vec![1.0],
            hbond_acoef: vec![0.0],
            hbond_bcoef: vec![0.0],
            hbcut: vec![0.0],
            tree_chain_classification: vec!["M".into(); 4],
            join_array: vec![0; 4],
            irotat: vec![0; 4],
            solvent_pointers: Vec::new(),
            atoms_per_molecule: vec![4],
            box_dimensions: Vec::new(),
            radius_set: Some("modified Bondi radii".into()),
            ipol: 0,
        }
    }

    #[test]
    fn targeted_steric_relax_resolves_local_overlap_shell() {
        let mut output = PackOutput {
            atoms: vec![
                test_atom("C1", "C", 1, Vec3::new(0.0, 0.0, 0.0)),
                test_atom("H1", "H", 1, Vec3::new(0.0, 1.0, 0.0)),
                test_atom("C2", "C", 2, Vec3::new(1.54, 0.0, 0.0)),
                test_atom("H2", "H", 2, Vec3::new(1.54, 1.0, 0.0)),
            ],
            bonds: vec![(0, 1), (0, 2), (2, 3)],
            box_size: [0.0, 0.0, 0.0],
            ter_after: vec![1, 3],
            box_vectors: None,
        };
        let report = relax_built_output(
            &mut output,
            &test_plan(),
            1.5,
            &RelaxSpec {
                mode: "targeted_steric".into(),
                steps: Some(128),
                step_scale: Some(0.5),
                clash_scale: Some(0.9),
            },
            None,
        );
        assert_eq!(report.mode, "targeted_steric");
        assert!(report.initial_overlap_pairs > 0, "{report:#?}");
        assert_eq!(report.final_overlap_pairs, 0, "{report:#?}");
        assert!(report.max_atom_displacement_angstrom > 0.0);
        assert_eq!(report.movable_atom_count, Some(4));
    }

    #[test]
    fn graph_spring_relax_falls_back_to_targeted_steric_for_residual_overlap() {
        let output = PackOutput {
            atoms: vec![
                test_atom("C1", "C", 1, Vec3::new(0.0, 0.0, 0.0)),
                test_atom("H1", "H", 1, Vec3::new(0.0, 1.0, 0.0)),
                test_atom("C2", "C", 2, Vec3::new(1.54, 0.0, 0.0)),
                test_atom("H2", "H", 2, Vec3::new(1.54, 1.0, 0.0)),
            ],
            bonds: vec![(0, 1), (0, 2), (2, 3)],
            box_size: [0.0, 0.0, 0.0],
            ter_after: vec![1, 3],
            box_vectors: None,
        };
        let relax = RelaxSpec {
            mode: "graph_spring".into(),
            steps: Some(8),
            step_scale: Some(0.0),
            clash_scale: Some(0.9),
        };

        let mut graph_only_output = output.clone();
        let graph_only =
            relax_graph_spring_output(&mut graph_only_output, &test_plan(), 1.5, &relax, None);
        assert!(graph_only.final_overlap_pairs > 0, "{graph_only:#?}");

        let mut fallback_output = output;
        let report = relax_built_output(&mut fallback_output, &test_plan(), 1.5, &relax, None);
        assert_eq!(report.mode, "graph_spring");
        assert_eq!(report.fallback_mode.as_deref(), Some("targeted_steric"));
        assert_eq!(
            report.pre_fallback_overlap_pairs,
            Some(graph_only.final_overlap_pairs)
        );
        assert_eq!(report.final_overlap_pairs, 0, "{report:#?}");
        assert!(
            report.fallback_steps_executed.unwrap_or(0) > 0,
            "{report:#?}"
        );
    }

    #[test]
    fn synthetic_bonded_cleanup_reduces_bond_error_and_clash() {
        let mut output = PackOutput {
            atoms: vec![
                test_atom("C1", "C", 1, Vec3::new(0.0, 0.0, 0.0)),
                test_atom("C2", "C", 1, Vec3::new(1.54, 0.0, 0.0)),
                test_atom("C3", "C", 2, Vec3::new(3.08, 0.0, 0.0)),
                test_atom("C4", "C", 2, Vec3::new(0.2, 0.2, 0.0)),
            ],
            bonds: vec![(0, 1), (1, 2), (2, 3)],
            box_size: [0.0, 0.0, 0.0],
            ter_after: vec![3],
            box_vectors: None,
        };
        let qc_context = crate::polymer::BuildQcContext {
            inter_residue_bond_count: 1,
            terminal_connectivity_consistent: true,
            sequence_token_template_consistent: true,
            bond_expectations: vec![crate::polymer::BuildBondExpectation {
                edge_id: "edge_2_3".into(),
                parent_resid: 2,
                child_resid: 3,
                parent_atom: "C3".into(),
                child_atom: "C4".into(),
                parent_idx: 2,
                child_idx: 3,
                expected_distance_angstrom: 1.54,
            }],
        };
        let topology = test_synthetic_topology();
        let initial_report = recompute_build_qc_report(&output, &qc_context);
        let initial_bond_error = (output.atoms[3]
            .position
            .sub(output.atoms[2].position)
            .norm()
            - 1.54)
            .abs();
        assert!(initial_report.severe_nonbonded_clash_count > 0);

        let report = relax_synthetic_topology_output(&mut output, &topology, &qc_context, None);
        let final_report = recompute_build_qc_report(&output, &qc_context);
        let final_bond_error = (output.atoms[3]
            .position
            .sub(output.atoms[2].position)
            .norm()
            - 1.54)
            .abs();

        assert_eq!(report.mode, "synthetic_bonded");
        assert!(report.steps_executed > 0, "{report:#?}");
        assert!(report.max_atom_displacement_angstrom > 0.0, "{report:#?}");
        assert!(
            report.final_energy_kcal_mol.unwrap_or(f32::INFINITY)
                < report.initial_energy_kcal_mol.unwrap_or(f32::INFINITY),
            "{report:#?}"
        );
        assert!(
            report.final_max_force.unwrap_or(f32::INFINITY)
                < report.initial_max_force.unwrap_or(f32::INFINITY),
            "{report:#?}"
        );
        assert!(
            report.accepted_line_search_steps.unwrap_or(0) > 0,
            "{report:#?}"
        );
        assert!(report.termination_reason.is_some(), "{report:#?}");
        assert!(final_bond_error < initial_bond_error, "{final_report:#?}");
        assert!(
            final_report.severe_nonbonded_clash_count < initial_report.severe_nonbonded_clash_count,
            "{final_report:#?}"
        );
        assert!(
            final_report.min_nonbonded_distance_angstrom.unwrap_or(0.0)
                > initial_report
                    .min_nonbonded_distance_angstrom
                    .unwrap_or(0.0)
        );
    }

    #[test]
    fn synthetic_bonded_cleanup_runs_when_qc_is_green() {
        let mut output = PackOutput {
            atoms: vec![
                test_atom("C1", "C", 1, Vec3::new(0.0, 0.0, 0.0)),
                test_atom("C2", "C", 1, Vec3::new(2.2, 0.0, 0.0)),
                test_atom("C3", "C", 2, Vec3::new(4.4, 0.0, 0.0)),
                test_atom("C4", "C", 2, Vec3::new(6.6, 0.0, 0.0)),
            ],
            bonds: vec![(0, 1), (1, 2), (2, 3)],
            box_size: [0.0, 0.0, 0.0],
            ter_after: vec![3],
            box_vectors: None,
        };
        let qc_context = crate::polymer::BuildQcContext {
            inter_residue_bond_count: 1,
            terminal_connectivity_consistent: true,
            sequence_token_template_consistent: true,
            bond_expectations: Vec::new(),
        };
        let topology = test_synthetic_topology();
        let initial_report = recompute_build_qc_report(&output, &qc_context);
        let initial_bond_error = (output.atoms[1]
            .position
            .sub(output.atoms[0].position)
            .norm()
            - 1.54)
            .abs();
        assert_eq!(initial_report.severe_nonbonded_clash_count, 0);
        assert!(initial_report.severe_bond_violations.is_empty());

        let report = relax_synthetic_topology_output(&mut output, &topology, &qc_context, None);
        let final_bond_error = (output.atoms[1]
            .position
            .sub(output.atoms[0].position)
            .norm()
            - 1.54)
            .abs();

        assert_eq!(report.mode, "synthetic_bonded");
        assert!(report.steps_executed > 0, "{report:#?}");
        assert!(final_bond_error < initial_bond_error, "{report:#?}");
    }
}
