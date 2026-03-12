use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;
use std::time::Instant;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use warp_pack::io::{write_amber_inpcrd, write_output};
use warp_pack::{geom::center_of_geometry, OutputSpec, PackError, PackResult};

use crate::polymer::{
    build_polymer_graph, compute_sequence_polymer_net_charge_from_prmtop,
    compute_sequence_polymer_net_charge_from_source, load_charge_manifest,
    write_polymer_prmtop_from_source, GraphEdgeSpec, GraphNodeSpec, TokenJunctionSpec,
    CHARGE_MANIFEST_VERSION,
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
pub const BUILD_SCHEMA_VERSION: &str = "polymer-build.agent.v1";
pub const BUILD_MANIFEST_VERSION: &str = "polymer-build.manifest.v1";
const ENSEMBLE_MANIFEST_VERSION: &str = "polymer-build.ensemble-manifest.v1";

const SUPPORTED_TARGET_MODES: &[&str] = &[
    "linear_homopolymer",
    "linear_sequence_polymer",
    "block_copolymer",
    "random_copolymer",
    "star_polymer",
    "branched_polymer",
    "polymer_graph",
];
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
    pub artifacts: ArtifactRequest,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct ErrorDetail {
    pub code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub message: String,
    pub severity: String,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct SuccessEnvelope {
    pub schema_version: String,
    pub status: String,
    pub request_id: String,
    pub artifacts: ArtifactRequest,
    pub summary: Value,
    pub warnings: Vec<ErrorDetail>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct ErrorEnvelope {
    pub schema_version: String,
    pub status: String,
    pub request_id: String,
    pub errors: Vec<ErrorDetail>,
    pub warnings: Vec<ErrorDetail>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct BuildManifestSchema {
    pub schema_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    pub request_id: String,
    pub normalized_request: Value,
    pub source_bundle: Value,
    pub target: Value,
    pub realization: Value,
    pub artifacts: Value,
    pub artifact_digests: Value,
    pub summary: Value,
    pub provenance: Value,
    pub warnings: Vec<ErrorDetail>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct ValidateEnvelope {
    pub schema_version: String,
    pub status: String,
    pub valid: bool,
    pub normalized_request: Value,
    pub resolved_inputs: Value,
    pub warnings: Vec<ErrorDetail>,
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

fn error_severity() -> String {
    "error".to_string()
}

fn warning_severity() -> String {
    "warning".to_string()
}

fn json_pointer_token(token: &str) -> String {
    token.replace('~', "~0").replace('/', "~1")
}

fn json_pointer(path: &str) -> Option<String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('/') {
        return Some(trimmed.to_string());
    }
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = trimmed.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '.' => {
                if !current.is_empty() {
                    segments.push(std::mem::take(&mut current));
                }
            }
            '[' => {
                if !current.is_empty() {
                    segments.push(std::mem::take(&mut current));
                }
                let mut index = String::new();
                while let Some(next) = chars.next() {
                    if next == ']' {
                        break;
                    }
                    index.push(next);
                }
                if !index.is_empty() {
                    segments.push(index);
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    if segments.is_empty() {
        None
    } else {
        Some(format!(
            "/{}",
            segments
                .iter()
                .map(|segment| json_pointer_token(segment))
                .collect::<Vec<_>>()
                .join("/")
        ))
    }
}

fn to_error(
    code: &str,
    path: impl Into<Option<String>>,
    message: impl Into<String>,
) -> ErrorDetail {
    ErrorDetail {
        code: code.to_string(),
        path: path.into().and_then(|value| json_pointer(&value)),
        message: message.into(),
        severity: error_severity(),
    }
}

fn to_warning(
    code: &str,
    path: impl Into<Option<String>>,
    message: impl Into<String>,
) -> ErrorDetail {
    ErrorDetail {
        code: code.to_string(),
        path: path.into().and_then(|value| json_pointer(&value)),
        message: message.into(),
        severity: warning_severity(),
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

fn sha256_file(path: &Path) -> PackResult<String> {
    let bytes = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn resolve_relative(base_path: &Path, value: &str) -> String {
    let path = Path::new(value);
    if path.is_absolute() {
        value.to_string()
    } else {
        base_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
            .to_string_lossy()
            .to_string()
    }
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
}

#[derive(Clone, Debug)]
struct ResolvedBuildRequest {
    bundle_digest: Option<String>,
    seed: u64,
    seed_policy: String,
    warnings: Vec<ErrorDetail>,
    source_coordinates: String,
    source_charge_manifest: Option<String>,
    source_topology_ref: Option<String>,
    artifacts: ResolvedArtifacts,
    normalized_request: Value,
    resolved_inputs: Value,
}

fn deterministic_seed(req: &BuildRequest) -> u64 {
    let text = serde_json::to_string(req).unwrap_or_default();
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
            Some("realization.seed".into()),
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

fn resolve_request_state(
    req: &BuildRequest,
    bundle: &SourceBundle,
) -> Result<ResolvedBuildRequest, Vec<ErrorDetail>> {
    let errors = validate_request(req, bundle);
    if !errors.is_empty() {
        return Err(errors);
    }
    let (seed, seed_policy) = resolve_seed(req).map_err(|err| vec![err])?;
    let source_coordinates = resolve_relative(
        Path::new(&req.source_ref.bundle_path),
        &bundle.artifacts.source_coordinates,
    );
    let source_charge_manifest = bundle
        .artifacts
        .source_charge_manifest
        .as_ref()
        .map(|path| resolve_relative(Path::new(&req.source_ref.bundle_path), path));
    let source_topology_ref = bundle
        .artifacts
        .source_topology_ref
        .as_ref()
        .map(|path| resolve_relative(Path::new(&req.source_ref.bundle_path), path));
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
        topology: if topology_transfer_supported(bundle) {
            Some(
                req.artifacts
                    .topology
                    .clone()
                    .unwrap_or_else(|| default_topology_path(&req.artifacts.coordinates)),
            )
        } else {
            req.artifacts.topology.clone()
        },
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
    };
    let warnings = collect_warnings(req);
    let bundle_digest = sha256_file(Path::new(&req.source_ref.bundle_path)).ok();
    let normalized_request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "request_id": req.request_id,
        "source_ref": {
            "bundle_id": req.source_ref.bundle_id,
            "bundle_path": req.source_ref.bundle_path,
            "bundle_digest": req.source_ref.bundle_digest.clone().or_else(|| bundle_digest.clone()),
        },
        "target": req.target,
        "realization": {
            "conformation_mode": req.realization.conformation_mode,
            "seed": seed,
            "seed_policy": seed_policy,
            "alignment_axis": req.realization.alignment_axis,
            "ensemble_size": req.realization.ensemble_size,
            "relax": req.realization.relax,
        },
        "conformer_policy": req.conformer_policy,
        "port_cap_overrides": req.port_cap_overrides,
        "artifacts": {
            "coordinates": artifacts.coordinates,
            "raw_coordinates": artifacts.raw_coordinates,
            "build_manifest": artifacts.build_manifest,
            "charge_manifest": artifacts.charge_manifest,
            "inpcrd": artifacts.inpcrd,
            "topology": artifacts.topology,
            "topology_graph": artifacts.topology_graph,
            "ensemble_manifest": artifacts.ensemble_manifest,
        },
    });
    let resolved_inputs = json!({
        "source_bundle_id": bundle.bundle_id,
        "source_bundle_path": req.source_ref.bundle_path,
        "source_bundle_digest": bundle_digest,
        "target_mode": req.target.mode,
        "realization_mode": req.realization.conformation_mode,
        "resolved_termini_policy": {
            "head": if req.target.termini.head == "default" { "source_default" } else { req.target.termini.head.as_str() },
            "tail": if req.target.termini.tail == "default" { "source_default" } else { req.target.termini.tail.as_str() },
        },
        "resolved_seed": seed,
        "seed_policy": seed_policy,
        "resolved_artifacts": {
            "coordinates": artifacts.coordinates,
            "raw_coordinates": artifacts.raw_coordinates,
            "build_manifest": artifacts.build_manifest,
            "charge_manifest": artifacts.charge_manifest,
            "inpcrd": artifacts.inpcrd,
            "topology": artifacts.topology,
            "topology_graph": artifacts.topology_graph,
            "ensemble_manifest": artifacts.ensemble_manifest,
        },
        "resolved_source_artifacts": {
            "coordinates": source_coordinates,
            "charge_manifest": source_charge_manifest,
            "topology": source_topology_ref,
            "forcefield_ref": bundle.artifacts.forcefield_ref,
        },
    });
    Ok(ResolvedBuildRequest {
        bundle_digest,
        seed,
        seed_policy,
        warnings,
        source_coordinates,
        source_charge_manifest,
        source_topology_ref,
        artifacts,
        normalized_request,
        resolved_inputs,
    })
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
            Some("target.composition".into()),
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
                    Some("target.repeat_count".into()),
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
                    Some("target.blocks".into()),
                    "block_copolymer requires blocks",
                )
            })?;
            if blocks.is_empty() {
                return Err(to_error(
                    "E_INVALID_TARGET",
                    Some("target.blocks".into()),
                    "block_copolymer requires at least one block",
                ));
            }
            let mut sequence = Vec::new();
            for (idx, block) in blocks.iter().enumerate() {
                if block.token.trim().is_empty() || block.count == 0 {
                    return Err(to_error(
                        "E_INVALID_TARGET",
                        Some(format!("target.blocks[{idx}]")),
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
                    Some("target.composition".into()),
                    "random_copolymer requires composition",
                )
            })?;
            if let Some(total_units) = target.total_units {
                let total = composition.values().sum::<usize>();
                if total != total_units {
                    return Err(to_error(
                        "E_INVALID_TARGET",
                        Some("target.total_units".into()),
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
            Some("target.mode".into()),
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
    parent_node_id: String,
    child_node_id: String,
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
    request_node_id: String,
    port_name: String,
    port_class: Option<String>,
    default_cap: Option<CapBinding>,
    allowed_caps: Vec<CapBinding>,
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
    motif_instance_id: Option<String>,
    motif_token: Option<String>,
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

fn token_template_resname(bundle: &SourceBundle, token: &str) -> Result<String, ErrorDetail> {
    if let Some(unit) = bundle.unit_library.get(token) {
        return Ok(unit
            .template_resname
            .clone()
            .unwrap_or_else(|| token.to_string()));
    }
    Err(to_error(
        "E_UNKNOWN_TOKEN",
        None,
        format!(
            "sequence token '{}' not present in source unit_library",
            token
        ),
    ))
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

fn apply_conformer_override(
    edge: &mut CompiledBuildEdge,
    conformer_policy: Option<&ConformerPolicy>,
    edge_override: Option<&EdgeConformerOverride>,
) {
    edge.layout_mode = conformer_policy
        .map(|policy| policy.layout_mode.clone())
        .unwrap_or_else(default_layout_mode);
    edge.torsion_mode = edge_override
        .map(|item| item.torsion_mode.clone())
        .unwrap_or_else(|| {
            conformer_policy
                .map(|policy| policy.default_torsion.clone())
                .unwrap_or_else(default_torsion_mode)
        });
    edge.torsion_deg = edge_override
        .and_then(|item| item.torsion_deg)
        .or_else(|| conformer_policy.and_then(|policy| policy.default_torsion_deg));
    edge.torsion_window_deg = edge_override
        .and_then(|item| item.torsion_window_deg)
        .or_else(|| conformer_policy.and_then(|policy| policy.torsion_window_deg));
    edge.ring_mode = edge_override
        .and_then(|item| item.ring_mode.clone())
        .or_else(|| conformer_policy.map(|policy| policy.ring_mode.clone()));
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
                    request_node_id: request_node_id.to_string(),
                    port_name: port.clone(),
                    port_class: None,
                    default_cap: None,
                    allowed_caps: Vec::new(),
                },
            )
        })
        .collect();
    Ok(TokenExpansion {
        root_node_idx: node_idx,
        request_node_id: request_node_id.to_string(),
        source_token: token.to_string(),
        motif_instance_id: None,
        motif_token: None,
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
                request_node_id: request_node_id.to_string(),
                port_name: port_name.clone(),
                port_class: port.port_class.clone(),
                default_cap: port.default_cap.clone(),
                allowed_caps: port.allowed_caps.clone(),
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
        motif_instance_id: Some(motif_instance_id),
        motif_token: Some(token.to_string()),
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
            Some("target.graph_edges".into()),
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
    parent_node_id: &str,
    parent_token: &str,
    parent_binding: &str,
    child_idx: usize,
    child_node_id: &str,
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
        parent_node_id: parent_node_id.to_string(),
        child_node_id: child_node_id.to_string(),
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
                &format!("target.sequence[{idx}]"),
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
            Some("target.sequence".into()),
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
            Some("target.core_token".into()),
            "star_polymer requires core_token",
        ));
    }
    let core_junctions = target.core_junctions.clone().unwrap_or_default();
    if core_junctions.is_empty() {
        return Err(to_error(
            "E_INVALID_TARGET",
            Some("target.core_junctions".into()),
            "star_polymer requires core_junctions",
        ));
    }
    let mut unique = BTreeSet::new();
    for junction in &core_junctions {
        if !unique.insert(junction.clone()) {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some("target.core_junctions".into()),
                format!("duplicate core junction '{}'", junction),
            ));
        }
    }
    let arm_sequence = target.arm_sequence.clone().unwrap_or_default();
    if arm_sequence.is_empty() {
        return Err(to_error(
            "E_INVALID_TARGET",
            Some("target.arm_sequence".into()),
            "star_polymer requires arm_sequence",
        ));
    }
    let arm_repeat_count = target.arm_repeat_count.unwrap_or(0);
    if arm_repeat_count == 0 {
        return Err(to_error(
            "E_INVALID_TARGET",
            Some("target.arm_repeat_count".into()),
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
                &format!("target.core_junctions[{arm_idx}]"),
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
            Some("target.branch_tree".into()),
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
            Some("target.graph_nodes".into()),
            "polymer_graph requires graph_nodes",
        )
    })?;
    if graph_nodes.is_empty() {
        return Err(to_error(
            "E_INVALID_TARGET",
            Some("target.graph_nodes".into()),
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
                Some(format!("target.graph_nodes[{idx}].id")),
                "graph node id must be non-empty",
            ));
        }
        if top_level_index_by_id.insert(node.id.clone(), idx).is_some() {
            return Err(to_error(
                "E_INVALID_TARGET",
                Some(format!("target.graph_nodes[{idx}].id")),
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
                Some(format!("target.graph_edges[{edge_idx}]")),
                "graph edge may not connect a node to itself",
            ));
        }
        let left = expansions.get(&edge.from).ok_or_else(|| {
            to_error(
                "E_INVALID_TARGET",
                Some(format!("target.graph_edges[{edge_idx}].from")),
                format!("graph edge references unknown node '{}'", edge.from),
            )
        })?;
        let right = expansions.get(&edge.to).ok_or_else(|| {
            to_error(
                "E_INVALID_TARGET",
                Some(format!("target.graph_edges[{edge_idx}].to")),
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
                Some(format!("target.graph_edges[{edge_idx}].from_junction")),
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
                Some(format!("target.graph_edges[{edge_idx}].to_junction")),
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
                Some(format!("target.graph_edges[{edge_idx}]")),
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
            &format!("target.graph_edges[{edge_idx}]"),
        )?;
        if let Some(last) = edges.last_mut() {
            last.bond_order = edge.bond_order.max(last.bond_order);
        }
    }
    let root_expansion = expansions.get(&root_id).ok_or_else(|| {
        to_error(
            "E_INVALID_TARGET",
            Some("target.graph_root".into()),
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
                    request_node_id: request_node_id.clone(),
                    port_name: port_name.clone(),
                    port_class: port_spec.port_class.clone(),
                    default_cap: port_spec.default_cap.clone(),
                    allowed_caps: port_spec.allowed_caps.clone(),
                },
            );
            let source_expansion = TokenExpansion {
                root_node_idx: node_idx,
                request_node_id: request_node_id.clone(),
                source_token: motif_token.clone(),
                motif_instance_id: Some(format!("motif::{request_node_id}")),
                motif_token: Some(motif_token.clone()),
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

fn template_resname_by_token(bundle: &SourceBundle) -> BTreeMap<String, String> {
    bundle
        .unit_library
        .iter()
        .map(|(token, entry)| {
            (
                token.clone(),
                entry
                    .template_resname
                    .clone()
                    .unwrap_or_else(|| token.clone()),
            )
        })
        .collect()
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
        let head_name = unit.junctions.get("head").ok_or_else(|| {
            PackError::Invalid(format!(
                "unit '{}' missing required 'head' junction binding",
                token
            ))
        })?;
        let tail_name = unit.junctions.get("tail").ok_or_else(|| {
            PackError::Invalid(format!(
                "unit '{}' missing required 'tail' junction binding",
                token
            ))
        })?;
        let head = bundle.junction_library.get(head_name).ok_or_else(|| {
            PackError::Invalid(format!(
                "junction '{}' missing from junction_library",
                head_name
            ))
        })?;
        let tail = bundle.junction_library.get(tail_name).ok_or_else(|| {
            PackError::Invalid(format!(
                "junction '{}' missing from junction_library",
                tail_name
            ))
        })?;
        specs.insert(
            token.clone(),
            TokenJunctionSpec {
                head_attach_atom: Some(selector_atom_name(&head.attach_atom)?),
                head_leaving_atoms: head
                    .leaving_atoms
                    .iter()
                    .map(selector_atom_name)
                    .collect::<PackResult<Vec<_>>>()?,
                tail_attach_atom: Some(selector_atom_name(&tail.attach_atom)?),
                tail_leaving_atoms: tail
                    .leaving_atoms
                    .iter()
                    .map(selector_atom_name)
                    .collect::<PackResult<Vec<_>>>()?,
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
            Some("realization.conformation_mode".into()),
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
            Some("conformer_policy.layout_mode".into()),
            format!("unsupported layout_mode '{}'", policy.layout_mode),
        ));
    }
    if !allowed_torsion.contains(&policy.default_torsion.as_str()) {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("conformer_policy.default_torsion".into()),
            format!("unsupported default_torsion '{}'", policy.default_torsion),
        ));
    }
    if !allowed_branch_spread.contains(&policy.branch_spread.as_str()) {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("conformer_policy.branch_spread".into()),
            format!("unsupported branch_spread '{}'", policy.branch_spread),
        ));
    }
    if !allowed_ring.contains(&policy.ring_mode.as_str()) {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("conformer_policy.ring_mode".into()),
            format!("unsupported ring_mode '{}'", policy.ring_mode),
        ));
    }
    if policy.default_torsion == "fixed_deg" && policy.default_torsion_deg.is_none() {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("conformer_policy.default_torsion_deg".into()),
            "fixed_deg default_torsion requires default_torsion_deg",
        ));
    }
    if policy.default_torsion == "sample_window" && policy.torsion_window_deg.is_none() {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("conformer_policy.torsion_window_deg".into()),
            "sample_window default_torsion requires torsion_window_deg",
        ));
    }
    if !policy.edge_overrides.is_empty() && req.target.mode != "polymer_graph" {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("conformer_policy.edge_overrides".into()),
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
                Some(format!("conformer_policy.edge_overrides[{idx}].edge_id")),
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
                Some("artifacts.raw_coordinates".into()),
                "raw_coordinates is only valid when realization.relax is set",
            ));
        }
        return errors;
    };
    if relax.mode != "graph_spring" {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("realization.relax.mode".into()),
            format!("unsupported relax mode '{}'", relax.mode),
        ));
    }
    if relax.steps.unwrap_or(64) == 0 {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("realization.relax.steps".into()),
            "relax.steps must be >= 1",
        ));
    }
    if relax.step_scale.unwrap_or(0.25) <= 0.0 {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("realization.relax.step_scale".into()),
            "relax.step_scale must be > 0",
        ));
    }
    if relax.clash_scale.unwrap_or(0.9) <= 0.0 {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("realization.relax.clash_scale".into()),
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
            Some("source_ref.bundle_id".into()),
            "source_ref.bundle_id does not match bundle_id",
        ));
    }
    if let Some(expected) = req.source_ref.bundle_digest.as_ref() {
        match sha256_file(Path::new(&req.source_ref.bundle_path)) {
            Ok(actual) if &actual != expected => errors.push(to_error(
                "E_SOURCE_DIGEST",
                Some("source_ref.bundle_digest".into()),
                format!("bundle digest mismatch: expected {expected}, got {actual}"),
            )),
            Err(err) => errors.push(to_error(
                "E_SOURCE_LOAD",
                Some("source_ref.bundle_path".into()),
                err.to_string(),
            )),
            _ => {}
        }
    }
    if !SUPPORTED_TARGET_MODES.contains(&req.target.mode.as_str()) {
        errors.push(to_error(
            "E_UNSUPPORTED_TARGET",
            Some("target.mode".into()),
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
            Some("target.mode".into()),
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
                    Some("target.repeat_unit".into()),
                    "linear_homopolymer requires repeat_unit",
                ));
            }
            if req.target.n_repeat.unwrap_or(0) == 0 {
                errors.push(to_error(
                    "E_INVALID_TARGET",
                    Some("target.n_repeat".into()),
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
                    Some("target.sequence".into()),
                    "linear_sequence_polymer requires sequence",
                ));
            }
            if bundle.capabilities.sequence_token_support.is_none() {
                errors.push(to_error(
                    "E_UNSUPPORTED_TARGET",
                    Some("target.sequence".into()),
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
                    Some("target.blocks".into()),
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
                    Some("target.composition".into()),
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
                    Some("target.core_token".into()),
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
                    Some("target.core_junctions".into()),
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
                    Some("target.arm_sequence".into()),
                    "star_polymer requires arm_sequence",
                ));
            }
            if req.target.arm_repeat_count.unwrap_or(0) == 0 {
                errors.push(to_error(
                    "E_INVALID_TARGET",
                    Some("target.arm_repeat_count".into()),
                    "arm_repeat_count must be >= 1",
                ));
            }
        }
        "branched_polymer" => {
            if req.target.branch_tree.is_none() {
                errors.push(to_error(
                    "E_INVALID_TARGET",
                    Some("target.branch_tree".into()),
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
                    Some("target.graph_nodes".into()),
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
                Some(format!("target.sequence[{idx}]")),
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
                        Some(format!("target.sequence[{idx}]")),
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
                            Some("target.sequence".into()),
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
                            Some("target.sequence".into()),
                            format!("unsupported token adjacency '{}-{}'", window[0], window[1]),
                        ));
                    }
                }
            }
        }
    }
    for token in &compiled_sequence {
        if let Some(unit) = bundle.unit_library.get(token) {
            for direction in ["head", "tail"] {
                let Some(junction_name) = unit.junctions.get(direction) else {
                    errors.push(to_error(
                        "E_SOURCE_SCHEMA",
                        Some(format!("unit_library.{token}.junctions.{direction}")),
                        format!(
                            "sequence token '{}' must declare '{}' junction",
                            token, direction
                        ),
                    ));
                    continue;
                };
                let Some(junction) = bundle.junction_library.get(junction_name) else {
                    continue;
                };
                if let Err(err) = selector_atom_name(&junction.attach_atom) {
                    errors.push(to_error(
                        "E_SOURCE_SCHEMA",
                        Some(format!(
                            "junction_library.{junction_name}.attach_atom.selector"
                        )),
                        err.to_string(),
                    ));
                }
                for (idx, selector) in junction.leaving_atoms.iter().enumerate() {
                    if let Err(err) = selector_atom_name(selector) {
                        errors.push(to_error(
                            "E_SOURCE_SCHEMA",
                            Some(format!(
                                "junction_library.{junction_name}.leaving_atoms[{idx}].selector"
                            )),
                            err.to_string(),
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
                Some(format!("target.sequence[{idx}]")),
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
                    Some(format!("target.graph_nodes[{idx}].token")),
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
            Some("realization.conformation_mode".into()),
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
            Some("realization.alignment_axis".into()),
            "aligned realization requires alignment_axis",
        ));
    }
    if req.realization.conformation_mode == "ensemble"
        && req.realization.ensemble_size.unwrap_or(0) == 0
    {
        errors.push(to_error(
            "E_INVALID_TARGET",
            Some("realization.ensemble_size".into()),
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
            Some("target.stereochemistry.mode".into()),
            "source bundle does not advertise requested stereochemistry mode",
        ));
    }
    if !SUPPORTED_TACTICITY_MODES.contains(&req.target.stereochemistry.mode.as_str()) {
        errors.push(to_error(
            "E_UNSUPPORTED_STEREOCHEMISTRY",
            Some("target.stereochemistry.mode".into()),
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
            })
            .unwrap_or(false)
    {
        errors.push(to_error(
            "E_OUTPUT_COLLISION",
            Some("artifacts".into()),
            "output artifact paths must differ",
        ));
    }
    if req.artifacts.topology.is_some() && !topology_transfer_supported(&bundle) {
        errors.push(to_error(
            "E_UNSUPPORTED_TARGET",
            Some("artifacts.topology".into()),
            "source bundle does not provide a transferable source prmtop",
        ));
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

fn rotate_output_along_axis(output: &mut warp_pack::pack::PackOutput, axis_name: &str) {
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
            .add(warp_pack::geom::Vec3::new(1.0, 0.0, 0.0));
    }
    let target = warp_pack::geom::Vec3::new(target[0], target[1], target[2]);
    let dot = (from.dot(target) / (from.norm() * target.norm()).max(1.0e-6)).clamp(-1.0, 1.0);
    let axis = from.cross(target);
    let theta = dot.acos();
    let axis = if axis.norm() <= 1.0e-6 {
        warp_pack::geom::Vec3::new(0.0, 0.0, 1.0)
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

#[derive(Clone, Debug)]
struct RelaxReport {
    mode: String,
    steps_requested: usize,
    steps_executed: usize,
    initial_max_clash: f32,
    final_max_clash: f32,
    rms_displacement: f32,
    raw_coordinates: Option<String>,
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

fn max_atom_clash(output: &warp_pack::pack::PackOutput, clash_scale: f32) -> f32 {
    let bonded = output
        .bonds
        .iter()
        .map(|(a, b)| if a <= b { (*a, *b) } else { (*b, *a) })
        .collect::<BTreeSet<_>>();
    let mut max_clash = 0.0f32;
    for left in 0..output.atoms.len() {
        for right in (left + 1)..output.atoms.len() {
            if bonded.contains(&(left, right)) {
                continue;
            }
            let atom_left = &output.atoms[left];
            let atom_right = &output.atoms[right];
            if atom_left.resid == atom_right.resid {
                continue;
            }
            let cutoff = (atom_vdw_radius(&atom_left.element)
                + atom_vdw_radius(&atom_right.element))
                * clash_scale;
            let dist = atom_left.position.sub(atom_right.position).norm();
            max_clash = max_clash.max(cutoff - dist);
        }
    }
    max_clash.max(0.0)
}

fn relax_built_output(
    output: &mut warp_pack::pack::PackOutput,
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
    let initial_max_clash = max_atom_clash(output, clash_scale);
    if output.atoms.is_empty() || steps_requested == 0 {
        return RelaxReport {
            mode: relax.mode.clone(),
            steps_requested,
            steps_executed: 0,
            initial_max_clash,
            final_max_clash: initial_max_clash,
            rms_displacement: 0.0,
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
                    warp_pack::geom::Vec3::new(0.0, 0.0, 0.0)
                } else {
                    center_of_geometry(&points)
                }
            })
            .collect::<Vec<_>>();
        let mut deltas = vec![warp_pack::geom::Vec3::new(0.0, 0.0, 0.0); residue_atoms.len()];
        for edge in &plan.edges {
            if edge.parent >= centers.len() || edge.child >= centers.len() {
                continue;
            }
            let diff = centers[edge.child].sub(centers[edge.parent]);
            let dist = diff.norm().max(1.0e-4);
            let dir = diff.scale(1.0 / dist);
            let stretch = dist - step_length.max(1.5);
            let correction = dir.scale(0.18 * stretch);
            deltas[edge.parent] = deltas[edge.parent].add(correction);
            deltas[edge.child] = deltas[edge.child].sub(correction);
        }
        for left in 0..output.atoms.len() {
            for right in (left + 1)..output.atoms.len() {
                let atom_left = &output.atoms[left];
                let atom_right = &output.atoms[right];
                if atom_left.resid == atom_right.resid {
                    continue;
                }
                let left_group = atom_left.resid.max(1) as usize - 1;
                let right_group = atom_right.resid.max(1) as usize - 1;
                let cutoff = (atom_vdw_radius(&atom_left.element)
                    + atom_vdw_radius(&atom_right.element))
                    * clash_scale;
                let diff = atom_right.position.sub(atom_left.position);
                let dist = diff.norm().max(1.0e-4);
                if dist >= cutoff {
                    continue;
                }
                let dir = diff.scale(1.0 / dist);
                let push = dir.scale(0.12 * (cutoff - dist));
                deltas[left_group] = deltas[left_group].sub(push);
                deltas[right_group] = deltas[right_group].add(push);
            }
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
        steps_executed = step_idx + 1;
        if max_atom_clash(output, clash_scale) <= 1.0e-2 {
            break;
        }
    }
    let final_max_clash = max_atom_clash(output, clash_scale);
    let rms_displacement = (output
        .atoms
        .iter()
        .zip(initial_positions.iter())
        .map(|(atom, initial)| atom.position.sub(*initial).norm().powi(2))
        .sum::<f32>()
        / output.atoms.len().max(1) as f32)
        .sqrt();
    RelaxReport {
        mode: relax.mode.clone(),
        steps_requested,
        steps_executed,
        initial_max_clash,
        final_max_clash,
        rms_displacement,
        raw_coordinates,
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
                token_kind: compiled_plan
                    .and_then(|plan| plan.nodes.get(idx).map(|node| node.token_kind.clone())),
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
            .map(|(idx, atom)| TopologyGraphAtom {
                index: idx + 1,
                name: atom.name.clone(),
                element: atom.element.clone(),
                resid: atom.resid,
                resname: atom.resname.clone(),
                charge_e: atom.charge,
                mass: 12.0,
                atom_type_index: atom.element.bytes().next().unwrap_or(b'X') as i32,
                amber_atom_type: atom.element.clone(),
                lj_class: atom.element.clone(),
                position: [atom.position.x, atom.position.y, atom.position.z],
                neighbors: adjacency
                    .get(idx)
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|value| value + 1)
                    .collect(),
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
                .map(|atom| atom.element.bytes().next().unwrap_or(b'X') as i32)
                .collect(),
            amber_atom_types: built
                .output
                .atoms
                .iter()
                .map(|atom| atom.element.clone())
                .collect(),
            lj_classes: built
                .output
                .atoms
                .iter()
                .map(|atom| atom.element.clone())
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
                        layout_mode: edge.layout_mode.clone(),
                        torsion_mode: edge.torsion_mode.clone(),
                        torsion_deg: edge.torsion_deg,
                        torsion_window_deg: edge.torsion_window_deg,
                        ring_mode: edge.ring_mode.clone(),
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

pub fn example_bundle() -> Value {
    json!({
        "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        "bundle_id": "pmma_param_bundle_v1",
        "training_context": {
            "mode": "oligomer_training",
            "training_oligomer_n": 3,
            "notes": "RESP/GAFF2 surrogate training"
        },
        "provenance": {},
        "unit_library": {
            "H": {
                "display_name": "PMMA head cap",
                "junctions": {"head": "pmma_head_cap", "tail": "pmma_head_cap"},
                "template_resname": "HDA"
            },
            "A": {
                "display_name": "PMMA repeat unit",
                "junctions": {"head": "pmma_head", "tail": "pmma_tail"},
                "template_resname": "RPT"
            },
            "B": {
                "display_name": "PMMA alternate repeat unit",
                "junctions": {"head": "pmma_head", "tail": "pmma_tail"},
                "template_resname": "RPT"
            },
            "T": {
                "display_name": "PMMA tail cap",
                "junctions": {"head": "pmma_tail_cap", "tail": "pmma_tail_cap"},
                "template_resname": "TLA"
            }
        },
        "motif_library": {
            "M2": {
                "display_name": "PMMA dimer motif",
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
            "pmma_head_cap": {
                "attach_atom": {"scope": "unit", "selector": "name C1"},
                "leaving_atoms": [],
                "bond_order": 1,
                "anchor_atoms": [{"scope": "unit", "selector": "name C1"}]
            },
            "pmma_head": {
                "attach_atom": {"scope": "unit", "selector": "name C1"},
                "leaving_atoms": [],
                "bond_order": 1,
                "anchor_atoms": [{"scope": "unit", "selector": "name C1"}]
            },
            "pmma_tail": {
                "attach_atom": {"scope": "unit", "selector": "name C2"},
                "leaving_atoms": [],
                "bond_order": 1,
                "anchor_atoms": [{"scope": "unit", "selector": "name C2"}]
            },
            "pmma_tail_cap": {
                "attach_atom": {"scope": "unit", "selector": "name C3"},
                "leaving_atoms": [],
                "bond_order": 1,
                "anchor_atoms": [{"scope": "unit", "selector": "name C3"}]
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
            "source_coordinates": "pmma_trimer.pdb",
            "source_topology_ref": "pmma_trimer.prmtop",
            "forcefield_ref": "pmma_polymer.ffxml",
            "source_charge_manifest": "pmma_trimer_charge.json"
        },
        "charge_model": {}
    })
}

pub fn example_request(mode: &str) -> Value {
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
                    {"id": "n2", "token": "M2"},
                    {"id": "n3", "token": "A"}
                ],
                "graph_edges": [
                    {"id": "e1", "from": "n1", "to": "n2", "from_junction": "head", "to_junction": "head", "bond_order": 1},
                    {"id": "e2", "from": "n2", "to": "n3", "from_junction": "tail", "to_junction": "head", "bond_order": 1},
                    {"id": "e3", "from": "n3", "to": "n1", "from_junction": "tail", "to_junction": "tail", "bond_order": 1}
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
                "conformation_mode": "random_walk",
                "seed": 12345
            }),
        ),
    };
    let ensemble_manifest = if mode == "ensemble" {
        Some("pmma_50mer.ensemble.json")
    } else {
        None
    };
    json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "request_id": "pmma-build-50mer-001",
        "source_ref": {
            "bundle_id": "pmma_param_bundle_v1",
            "bundle_path": "pmma_param_bundle.json",
            "bundle_digest": "sha256:..."
        },
        "target": target,
        "realization": {
            "conformation_mode": realization["conformation_mode"].clone(),
            "alignment_axis": realization.get("alignment_axis").cloned().unwrap_or(Value::Null),
            "seed": realization["seed"].clone(),
            "ensemble_size": realization.get("ensemble_size").cloned().unwrap_or(Value::Null),
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
        "port_cap_overrides": if mode == "graph" || mode == "polymer_graph" {
            vec![json!({
                "node_id": "n2",
                "port": "tail",
                "cap": {"token": "T", "junction": "head"}
            })]
        } else {
            Vec::<Value>::new()
        },
        "artifacts": {
            "coordinates": "pmma_50mer.pdb",
            "raw_coordinates": "pmma_50mer.raw.pdb",
            "build_manifest": "pmma_50mer.build.json",
            "charge_manifest": "pmma_50mer.charge.json",
            "inpcrd": "pmma_50mer.inpcrd",
            "topology": "pmma_50mer.prmtop",
            "topology_graph": "pmma_50mer.topology.json",
            "ensemble_manifest": ensemble_manifest
        }
    })
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
        "artifact_outputs": [
            "coordinates",
            "raw_coordinates",
            "build_manifest",
            "charge_manifest",
            "inpcrd",
            "topology",
            "topology_graph",
            "ensemble_manifest"
        ],
        "topology_outputs": ["inpcrd", "prmtop"],
        "parameter_outputs": ["forcefield_ref"],
        "agent_contract": {
            "machine_readable_errors": true,
            "deterministic_seeded_output": true,
            "streaming_events": true,
            "preferred_handoff": ["build_manifest", "charge_manifest", "topology", "topology_graph"]
        },
        "schema_targets": ["source_bundle", "request", "result", "event", "build_manifest", "charge_manifest", "topology_graph"],
        "supports_named_termini_tokens": true,
        "supports_motif_tokens": true,
        "supports_conformer_policy": true,
        "supports_port_cap_overrides": true,
        "supports_relax_modes": ["graph_spring"],
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
        Ok(bundle) => (
            0,
            json!({
                "status": "ok",
                "bundle_id": bundle.bundle_id,
                "schema_version": bundle.schema_version,
                "training_context": bundle.training_context,
                "supported_target_modes": bundle.capabilities.supported_target_modes,
                "supported_conformation_modes": bundle.capabilities.supported_conformation_modes,
                "supported_tacticity_modes": bundle.capabilities.supported_tacticity_modes,
                "supported_termini_policies": bundle.capabilities.supported_termini_policies,
                "sequence_token_support": bundle.capabilities.sequence_token_support,
                "charge_transfer_supported": bundle.capabilities.charge_transfer_supported,
                "topology_transfer_supported": topology_transfer_supported(&bundle),
                "unit_tokens": bundle.unit_library.keys().cloned().collect::<Vec<_>>(),
                "motif_tokens": bundle.motif_library.keys().cloned().collect::<Vec<_>>(),
                "artifacts": bundle.artifacts,
            }),
        ),
        Err(err) => (2, json!({"status": "error", "errors": [err]})),
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
    match resolve_request_state(&req, &bundle) {
        Ok(resolved) => (
            0,
            json!(ValidateEnvelope {
                schema_version: BUILD_SCHEMA_VERSION.into(),
                status: "ok".into(),
                valid: true,
                normalized_request: resolved.normalized_request,
                resolved_inputs: resolved.resolved_inputs,
                warnings: resolved.warnings,
            }),
        ),
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
    let req: BuildRequest = match parse_json(text) {
        Ok(req) => req,
        Err(err) => {
            return (
                2,
                json!(ErrorEnvelope {
                    schema_version: BUILD_SCHEMA_VERSION.into(),
                    status: "error".into(),
                    request_id: "polymer-build".into(),
                    errors: vec![err],
                    warnings: vec![],
                }),
            )
        }
    };
    emit(
        &RunEvent::RunStarted {
            request_id: req.request_id.clone(),
        },
        stream_ndjson,
    );
    let bundle = match load_bundle(Path::new(&req.source_ref.bundle_path)) {
        Ok(bundle) => bundle,
        Err(err) => {
            emit(
                &RunEvent::RunFailed {
                    request_id: req.request_id.clone(),
                    error: err.clone(),
                },
                stream_ndjson,
            );
            return (
                2,
                json!(ErrorEnvelope {
                    schema_version: BUILD_SCHEMA_VERSION.into(),
                    status: "error".into(),
                    request_id: req.request_id,
                    errors: vec![err],
                    warnings: vec![],
                }),
            );
        }
    };
    let resolved = match resolve_request_state(&req, &bundle) {
        Ok(resolved) => resolved,
        Err(errors) => {
            emit(
                &RunEvent::RunFailed {
                    request_id: req.request_id.clone(),
                    error: errors[0].clone(),
                },
                stream_ndjson,
            );
            return (
                2,
                json!(ErrorEnvelope {
                    schema_version: BUILD_SCHEMA_VERSION.into(),
                    status: "error".into(),
                    request_id: req.request_id,
                    errors,
                    warnings: vec![],
                }),
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
    emit(
        &RunEvent::PhaseStarted {
            request_id: req.request_id.clone(),
            phase: "chain_growth".into(),
        },
        stream_ndjson,
    );

    let seed = resolved.seed;
    let _expanded_sequence = match expand_sequence(&req.target, seed) {
        Ok(sequence) => sequence,
        Err(err) => {
            if req.target.mode != "star_polymer"
                && req.target.mode != "branched_polymer"
                && req.target.mode != "polymer_graph"
            {
                return (
                    2,
                    json!(ErrorEnvelope {
                        schema_version: BUILD_SCHEMA_VERSION.into(),
                        status: "error".into(),
                        request_id: req.request_id,
                        errors: vec![err],
                        warnings: vec![],
                    }),
                );
            }
            Vec::new()
        }
    };
    let compiled_plan = match compile_build_plan(&req, &bundle, seed) {
        Ok(Some(plan)) => plan,
        Ok(None) => {
            return (
                2,
                json!(ErrorEnvelope {
                    schema_version: BUILD_SCHEMA_VERSION.into(),
                    status: "error".into(),
                    request_id: req.request_id,
                    errors: vec![to_error(
                        "E_INVALID_TARGET",
                        Some("target".into()),
                        "target expands to an empty sequence"
                    )],
                    warnings: vec![],
                }),
            )
        }
        Err(err) => {
            return (
                2,
                json!(ErrorEnvelope {
                    schema_version: BUILD_SCHEMA_VERSION.into(),
                    status: "error".into(),
                    request_id: req.request_id,
                    errors: vec![err],
                    warnings: vec![],
                }),
            );
        }
    };
    let compiled_sequence = compiled_plan
        .nodes
        .iter()
        .map(|node| node.token.clone())
        .collect::<Vec<_>>();
    let n_repeat = compiled_sequence.len();
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
    let inpcrd_path = resolved.artifacts.inpcrd.clone();
    let topology_graph_path = resolved.artifacts.topology_graph.clone();
    let raw_coordinates_path = resolved.artifacts.raw_coordinates.clone();
    let ensemble_manifest_path = resolved.artifacts.ensemble_manifest.clone();
    let topology_path = resolved.artifacts.topology.clone();
    let token_junctions = match token_junction_specs(&bundle, &compiled_sequence) {
        Ok(specs) => specs,
        Err(err) => {
            let detail = to_error("E_SOURCE_SCHEMA", None, err.to_string());
            emit(
                &RunEvent::RunFailed {
                    request_id: req.request_id.clone(),
                    error: detail.clone(),
                },
                stream_ndjson,
            );
            return (
                2,
                json!(ErrorEnvelope {
                    schema_version: BUILD_SCHEMA_VERSION.into(),
                    status: "error".into(),
                    request_id: req.request_id,
                    errors: vec![detail],
                    warnings: vec![],
                }),
            );
        }
    };
    let base_mode = match base_conformation_mode(&req.realization.conformation_mode) {
        Ok(mode) => mode,
        Err(err) => {
            return (
                2,
                json!(ErrorEnvelope {
                    schema_version: BUILD_SCHEMA_VERSION.into(),
                    status: "error".into(),
                    request_id: req.request_id,
                    errors: vec![err],
                    warnings: vec![],
                }),
            );
        }
    };
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
                parent_port: edge.parent_port.clone(),
                child_port: edge.child_port.clone(),
                parent_attach_atom: edge.parent_attach_atom.clone(),
                parent_leaving_atoms: edge.parent_leaving_atoms.clone(),
                child_attach_atom: edge.child_attach_atom.clone(),
                child_leaving_atoms: edge.child_leaving_atoms.clone(),
                bond_order: edge.bond_order,
                layout_mode: edge.layout_mode.clone(),
                branch_spread: edge.branch_spread.clone(),
                torsion_mode: edge.torsion_mode.clone(),
                torsion_deg: edge.torsion_deg,
                torsion_window_deg: edge.torsion_window_deg,
                ring_mode: edge.ring_mode.clone(),
            })
            .collect::<Vec<_>>();
        let graph_root_idx = compiled_plan
            .nodes
            .iter()
            .position(|node| node.node_id == compiled_plan.expanded_root_node_id)
            .unwrap_or(0);
        let build_result = build_polymer_graph(
            &source_coordinates,
            source_charge_manifest.as_deref(),
            source_topology_ref.as_deref(),
            bundle.training_context.training_oligomer_n,
            &graph_node_specs,
            &graph_edge_specs,
            graph_root_idx,
            base_mode,
            tacticity,
            member_seed,
            &member_target_path,
        );
        let mut built = match build_result {
            Ok(built) => built,
            Err(err) => {
                let detail = to_error("E_RUNTIME_BUILD", None, err.to_string());
                emit(
                    &RunEvent::RunFailed {
                        request_id: req.request_id.clone(),
                        error: detail.clone(),
                    },
                    stream_ndjson,
                );
                return (
                    4,
                    json!(ErrorEnvelope {
                        schema_version: BUILD_SCHEMA_VERSION.into(),
                        status: "error".into(),
                        request_id: req.request_id,
                        errors: vec![detail],
                        warnings: vec![],
                    }),
                );
            }
        };
        let relax_report = if let Some(relax) = req.realization.relax.as_ref() {
            if let Some(raw_path) = member_raw_path.as_ref() {
                ensure_parent(raw_path).ok();
                if let Err(err) = fs::copy(&built.path, raw_path) {
                    let detail = to_error("E_OUTPUT_WRITE", None, err.to_string());
                    return (
                        4,
                        json!(ErrorEnvelope {
                            schema_version: BUILD_SCHEMA_VERSION.into(),
                            status: "error".into(),
                            request_id: req.request_id,
                            errors: vec![detail],
                            warnings: vec![],
                        }),
                    );
                }
            }
            Some(relax_built_output(
                &mut built.output,
                &compiled_plan,
                built.step_length_angstrom,
                relax,
                member_raw_path.clone(),
            ))
        } else {
            None
        };
        if req.realization.conformation_mode == "aligned" {
            rotate_output_along_axis(
                &mut built.output,
                req.realization.alignment_axis.as_deref().unwrap_or("z"),
            );
        }
        for path in [
            built.path.to_string_lossy().to_string(),
            member_target_path.clone(),
        ]
        .into_iter()
        .collect::<BTreeSet<_>>()
        {
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
                    json!(ErrorEnvelope {
                        schema_version: BUILD_SCHEMA_VERSION.into(),
                        status: "error".into(),
                        request_id: req.request_id,
                        errors: vec![detail],
                        warnings: vec![],
                    }),
                );
            }
        }
        built_members.push((member_target_path, built, member_seed, relax_report));
    }
    let built = built_members[0].1.clone();
    let primary_coordinates = built_members[0].0.clone();
    let primary_relax_report = built_members[0].3.clone();
    ensure_parent(&resolved.artifacts.coordinates).ok();
    if primary_coordinates != resolved.artifacts.coordinates {
        if let Err(err) = fs::copy(&primary_coordinates, &resolved.artifacts.coordinates) {
            return (
                4,
                json!(ErrorEnvelope {
                    schema_version: BUILD_SCHEMA_VERSION.into(),
                    status: "error".into(),
                    request_id: req.request_id,
                    errors: vec![to_error("E_OUTPUT_WRITE", None, err.to_string())],
                    warnings: vec![],
                }),
            );
        }
    }
    ensure_parent(&topology_graph_path).ok();
    let topology_graph = topology_graph_value(
        &req.request_id,
        &built,
        &compiled_sequence,
        &bundle,
        &req.target.mode,
        &req.realization.conformation_mode,
        &token_junctions,
        &req.target.termini,
        Some(&compiled_plan),
        primary_relax_report
            .as_ref()
            .map(|report| TopologyGraphRelaxMetadata {
                mode: report.mode.clone(),
                steps_requested: report.steps_requested,
                steps_executed: report.steps_executed,
                initial_max_clash: report.initial_max_clash,
                final_max_clash: report.final_max_clash,
                rms_displacement: report.rms_displacement,
                raw_coordinates: report.raw_coordinates.clone(),
            }),
    );
    if let Err(err) = fs::write(
        &topology_graph_path,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&topology_graph).unwrap_or_else(|_| "{}".into())
        ),
    ) {
        return (
            4,
            json!(ErrorEnvelope {
                schema_version: BUILD_SCHEMA_VERSION.into(),
                status: "error".into(),
                request_id: req.request_id,
                errors: vec![to_error(
                    "E_OUTPUT_WRITE",
                    Some("artifacts.topology_graph".into()),
                    err.to_string()
                )],
                warnings: vec![],
            }),
        );
    }
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
                .map(|(idx, (path, _, seed, _))| json!({"member_index": idx + 1, "coordinates": path, "seed": seed}))
                .collect::<Vec<_>>(),
            "shared_artifacts": {
                "topology_graph": topology_graph_path,
                "topology": topology_path,
                "charge_manifest": resolved.artifacts.charge_manifest,
            },
        });
        if let Err(err) = fs::write(
            &ensemble_manifest_path,
            format!(
                "{}\n",
                serde_json::to_string_pretty(&ensemble_manifest).unwrap_or_else(|_| "{}".into())
            ),
        ) {
            return (
                4,
                json!(ErrorEnvelope {
                    schema_version: BUILD_SCHEMA_VERSION.into(),
                    status: "error".into(),
                    request_id: req.request_id,
                    errors: vec![to_error(
                        "E_OUTPUT_WRITE",
                        Some("artifacts.ensemble_manifest".into()),
                        err.to_string()
                    )],
                    warnings: vec![],
                }),
            );
        }
    }
    ensure_parent(&inpcrd_path).ok();
    if let Err(err) = write_amber_inpcrd(&built.output, &inpcrd_path, 1.0) {
        return (
            4,
            json!(ErrorEnvelope {
                schema_version: BUILD_SCHEMA_VERSION.into(),
                status: "error".into(),
                request_id: req.request_id,
                errors: vec![to_error("E_OUTPUT_WRITE", None, err.to_string())],
                warnings: vec![],
            }),
        );
    }
    if let (Some(source_topology_ref), Some(topology_path)) = (
        bundle.artifacts.source_topology_ref.as_ref(),
        topology_path.as_ref(),
    ) {
        let source_topology_path =
            resolve_relative(Path::new(&req.source_ref.bundle_path), source_topology_ref);
        ensure_parent(topology_path).ok();
        if let Err(err) = write_polymer_prmtop_from_source(
            &built,
            Path::new(&source_topology_path),
            topology_path,
        ) {
            return (
                4,
                json!(ErrorEnvelope {
                    schema_version: BUILD_SCHEMA_VERSION.into(),
                    status: "error".into(),
                    request_id: req.request_id,
                    errors: vec![to_error(
                        "E_OUTPUT_WRITE",
                        Some("artifacts.topology".into()),
                        err.to_string(),
                    )],
                    warnings: vec![],
                }),
            );
        }
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
                    json!(ErrorEnvelope {
                        schema_version: BUILD_SCHEMA_VERSION.into(),
                        status: "error".into(),
                        request_id: req.request_id,
                        errors: vec![to_error(
                            "E_CHARGE_HANDOFF",
                            None,
                            "failed to load source charge manifest"
                        )],
                        warnings: vec![],
                    }),
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
                    json!(ErrorEnvelope {
                        schema_version: BUILD_SCHEMA_VERSION.into(),
                        status: "error".into(),
                        request_id: req.request_id,
                        errors: vec![to_error("E_CHARGE_HANDOFF", None, err.to_string())],
                        warnings: vec![],
                    }),
                )
            }
        }
    } else if let Some(topology_ref) = bundle.artifacts.source_topology_ref.as_ref() {
        let topology_path = resolve_relative(Path::new(&req.source_ref.bundle_path), topology_ref);
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
                    json!(ErrorEnvelope {
                        schema_version: BUILD_SCHEMA_VERSION.into(),
                        status: "error".into(),
                        request_id: req.request_id,
                        errors: vec![to_error("E_CHARGE_HANDOFF", None, err.to_string())],
                        warnings: vec![],
                    }),
                )
            }
        }
    } else {
        (None, "unavailable".to_string())
    };

    ensure_parent(&resolved.artifacts.charge_manifest).ok();
    let charge_manifest = json!({
        "schema_version": CHARGE_MANIFEST_VERSION,
        "solute_path": resolved.artifacts.coordinates,
        "source_topology_ref": bundle.artifacts.source_topology_ref,
        "target_topology_ref": topology_path,
        "forcefield_ref": bundle.artifacts.forcefield_ref,
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
            json!(ErrorEnvelope {
                schema_version: BUILD_SCHEMA_VERSION.into(),
                status: "error".into(),
                request_id: req.request_id,
                errors: vec![to_error("E_OUTPUT_WRITE", None, err.to_string())],
                warnings: vec![],
            }),
        );
    }

    ensure_parent(&resolved.artifacts.build_manifest).ok();
    let applied_head_terminus = json!({
        "requested_policy": req.target.termini.head.clone(),
        "resolved_token": built.sequence_labels.first().cloned(),
        "template_resname": built.template_sequence_resnames.first().cloned(),
        "applied_resname": built.residue_resnames.first().cloned(),
    });
    let applied_tail_terminus = json!({
        "requested_policy": req.target.termini.tail.clone(),
        "resolved_token": built.sequence_labels.last().cloned(),
        "template_resname": built.template_sequence_resnames.last().cloned(),
        "applied_resname": built.residue_resnames.last().cloned(),
    });
    let artifact_digests = json!({
        "coordinates": sha256_file(Path::new(&resolved.artifacts.coordinates)).ok(),
        "raw_coordinates": raw_coordinates_path
            .as_ref()
            .and_then(|path| sha256_file(Path::new(path)).ok()),
        "charge_manifest": sha256_file(Path::new(&resolved.artifacts.charge_manifest)).ok(),
        "inpcrd": sha256_file(Path::new(&inpcrd_path)).ok(),
        "topology": topology_path
            .as_ref()
            .and_then(|path| sha256_file(Path::new(path)).ok()),
        "topology_graph": sha256_file(Path::new(&topology_graph_path)).ok(),
        "ensemble_manifest": ensemble_manifest_path
            .as_ref()
            .and_then(|path| sha256_file(Path::new(path)).ok()),
    });
    let manifest = json!({
        "schema_version": BUILD_MANIFEST_VERSION,
        "request_id": req.request_id,
        "normalized_request": resolved.normalized_request,
        "resolved_inputs": resolved.resolved_inputs,
        "source_bundle": {
            "bundle_id": bundle.bundle_id,
            "bundle_path": req.source_ref.bundle_path,
            "bundle_digest": resolved.bundle_digest,
            "training_context": bundle.training_context,
        },
        "target": req.target,
        "realization": {
            "conformation_mode": req.realization.conformation_mode,
            "seed": seed,
            "seed_policy": resolved.seed_policy,
            "relax": req.realization.relax,
        },
        "artifacts": {
            "coordinates": resolved.artifacts.coordinates,
            "raw_coordinates": raw_coordinates_path,
            "charge_manifest": resolved.artifacts.charge_manifest,
            "inpcrd": inpcrd_path,
            "topology": topology_path,
            "topology_graph": topology_graph_path,
            "ensemble_manifest": ensemble_manifest_path,
            "forcefield_ref": bundle.artifacts.forcefield_ref,
        },
        "artifact_digests": artifact_digests,
        "summary": {
            "atom_count": built.output.atoms.len(),
            "total_repeat_units": n_repeat,
            "net_charge_e": net_charge,
            "resolved_sequence": built.sequence_labels,
            "request_root_node_id": compiled_plan.request_root_node_id,
            "expanded_root_node_id": compiled_plan.expanded_root_node_id,
            "graph_has_cycle": compiled_plan.graph_has_cycle,
            "applied_termini": {
                "head": applied_head_terminus,
                "tail": applied_tail_terminus,
            },
            "applied_caps": compiled_plan
                .applied_caps
                .iter()
                .map(|cap| json!({
                    "node_id": cap.node_id,
                    "request_node_id": cap.request_node_id,
                    "port_name": cap.port_name,
                    "cap": cap.cap,
                    "application_source": cap.application_source,
                    "cap_node_id": cap.cap_node_id,
                }))
                .collect::<Vec<_>>(),
            "bond_count": built.output.bonds.len(),
            "realization_mode": req.realization.conformation_mode,
            "ensemble_size": ensemble_size,
            "relax": primary_relax_report.as_ref().map(|report| json!({
                "mode": report.mode,
                "steps_requested": report.steps_requested,
                "steps_executed": report.steps_executed,
                "initial_max_clash": report.initial_max_clash,
                "final_max_clash": report.final_max_clash,
                "rms_displacement": report.rms_displacement,
                "raw_coordinates": report.raw_coordinates,
            })),
            "applied_junctions": token_junctions
                .iter()
                .map(|(token, junction)| {
                    (
                        token.clone(),
                        json!({
                            "head_attach_atom": junction.head_attach_atom,
                            "head_leaving_atoms": junction.head_leaving_atoms,
                            "tail_attach_atom": junction.tail_attach_atom,
                            "tail_leaving_atoms": junction.tail_leaving_atoms,
                        }),
                    )
                })
                .collect::<BTreeMap<_, _>>(),
        },
        "provenance": {
            "schema_version": BUILD_MANIFEST_VERSION,
            "builder_version": env!("CARGO_PKG_VERSION"),
            "binary_version": env!("CARGO_PKG_VERSION"),
            "algorithm_version": format!("{}.v1", req.realization.conformation_mode),
            "topology_transfer_mode": if topology_path.is_some() {
                "residue_filtered_with_bonds"
            } else {
                "none"
            },
            "source_bundle_path": req.source_ref.bundle_path,
            "source_bundle_digest": resolved.bundle_digest,
            "build_metadata": {
                "git_commit": option_env!("VERGEN_GIT_SHA"),
                "build_timestamp": option_env!("VERGEN_BUILD_TIMESTAMP"),
                "target_triple": option_env!("TARGET"),
            },
        },
        "warnings": resolved.warnings,
    });
    if let Err(err) = fs::write(
        &resolved.artifacts.build_manifest,
        format!(
            "{}\n",
            serde_json::to_string_pretty(&manifest).unwrap_or_else(|_| "{}".into())
        ),
    ) {
        return (
            4,
            json!(ErrorEnvelope {
                schema_version: BUILD_SCHEMA_VERSION.into(),
                status: "error".into(),
                request_id: req.request_id,
                errors: vec![to_error("E_OUTPUT_WRITE", None, err.to_string())],
                warnings: vec![],
            }),
        );
    }

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
            },
        },
        stream_ndjson,
    );

    (
        0,
        json!(SuccessEnvelope {
            schema_version: BUILD_SCHEMA_VERSION.into(),
            status: "ok".into(),
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
            },
            summary: json!({
                "build_mode": req.target.mode,
                "n_repeat": n_repeat,
                "atom_count": built.output.atoms.len(),
                "total_repeat_units": n_repeat,
                "conformation_mode": req.realization.conformation_mode,
                "seed": seed,
                "ensemble_size": ensemble_size,
                "topology_graph_version": TOPOLOGY_GRAPH_VERSION,
                "relax": primary_relax_report.as_ref().map(|report| json!({
                    "mode": report.mode,
                    "steps_requested": report.steps_requested,
                    "steps_executed": report.steps_executed,
                    "initial_max_clash": report.initial_max_clash,
                    "final_max_clash": report.final_max_clash,
                    "rms_displacement": report.rms_displacement,
                    "raw_coordinates": report.raw_coordinates,
                })),
            }),
            warnings: resolved.warnings,
        }),
    )
}
