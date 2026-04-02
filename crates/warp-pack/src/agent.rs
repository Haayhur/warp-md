use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

use schemars::{schema_for, JsonSchema};
use serde::de::IntoDeserializer;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use traj_core::principal_axes_from_inertia;
use warp_topology_graph::TopologyGraph;

use crate::charge::{
    charge_manifest_field_kinds, compute_solute_net_charge, compute_solute_net_charge_from_prmtop,
    load_charge_manifest,
};
use crate::config::{BoxSpec, OutputSpec, PackConfig, StructureSpec};
use crate::constraints::{ConstraintMode, ConstraintSpec, ShapeSpec};
use crate::error::{PackError, PackResult};
use crate::geom::{center_of_geometry, Quaternion, Vec3};
use crate::io::{read_molecule, write_output, MoleculeData};
use crate::pack::run;

pub const AGENT_SCHEMA_VERSION: &str = "warp-pack.agent.v1";
const WARP_BUILD_MANIFEST_VERSION: &str = "warp-build.manifest.v1";

const SUPPORTED_BUILD_MODES: &[&str] = &[
    "solute_solvate",
    "polymer_build_handoff",
    "components_amorphous_bulk",
    "components_backbone_aligned_bulk",
];
const SUPPORTED_COMPONENT_KINDS: &[&str] = &[
    "polymer_chain",
    "protein",
    "complex",
    "small_molecule",
    "assembly",
];
const SUPPORTED_SOLVENT_MODELS: &[&str] = &["spce", "tip3p", "tip4pew", "tip5p"];
const SUPPORTED_MORPHOLOGY_MODES: &[&str] = &[
    "single_chain_solution",
    "amorphous_bulk",
    "backbone_aligned_bulk",
];
const SUPPORTED_OUTPUT_FORMATS: &[&str] = &[
    "pdb",
    "pdb-strict",
    "pdbx",
    "cif",
    "mmcif",
    "gro",
    "lammps",
    "lammps-data",
    "lmp",
    "mol2",
    "crd",
    "inpcrd",
    "rst",
    "rst7",
    "xyz",
];

const AVOGADRO: f64 = 6.022_140_76e23;
const ANGSTROM3_TO_LITER: f64 = 1.0e-27;
const ANGSTROM3_TO_CM3: f64 = 1.0e-24;
const AMU_TO_GRAM: f64 = 1.660_539_066_60e-24;
const WATER_MOLARITY: f64 = 55.5;
const WATER_MASS_AMU: f64 = 18.015_28;
const DEFAULT_PACKING_FRACTION: f64 = 0.80;
const ION_REGISTRY_ENV: &str = "WARP_MD_ION_REGISTRY";
const SALT_REGISTRY_ENV: &str = "WARP_MD_SALT_REGISTRY";

#[derive(Clone, Debug, Deserialize)]
struct IonRegistryEntry {
    species: String,
    #[serde(default)]
    aliases: Vec<String>,
    template: String,
    formula_symbol: String,
    charge_e: i32,
    mass_amu: f64,
    #[serde(default)]
    topology_kind: Option<String>,
    #[serde(default)]
    atom_count: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
struct IonRegistryFile {
    ions: Vec<IonRegistryEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct SaltRegistryEntry {
    name: String,
    #[serde(default)]
    aliases: Vec<String>,
    formula: String,
    species: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Deserialize)]
struct SaltRegistryFile {
    salts: Vec<SaltRegistryEntry>,
}

#[derive(Clone, Debug)]
struct IonSpeciesInfo {
    species: String,
    template: String,
    formula_symbol: String,
    charge_e: i32,
    mass_amu: f64,
}

#[derive(Clone, Debug)]
struct IonRegistry {
    supported_species: Vec<String>,
    by_lookup: BTreeMap<String, IonSpeciesInfo>,
    by_symbol: BTreeMap<String, String>,
    ambiguous_symbols: BTreeSet<String>,
}

#[derive(Clone, Debug)]
struct NormalizedSaltSpec {
    name: Option<String>,
    formula: Option<String>,
    species: BTreeMap<String, usize>,
    molar: Option<f32>,
}

#[derive(Clone, Debug)]
struct SaltRegistry {
    supported_names: Vec<String>,
    by_lookup: BTreeMap<String, NormalizedSaltSpec>,
    by_formula: BTreeMap<String, NormalizedSaltSpec>,
}

#[derive(Clone, Debug)]
struct ChemistryCatalog {
    ions: IonRegistry,
    salts: SaltRegistry,
}

static ION_REGISTRY: OnceLock<Result<IonRegistry, String>> = OnceLock::new();
static SALT_REGISTRY: OnceLock<Result<SaltRegistry, String>> = OnceLock::new();

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ArtifactRef {
    pub path: String,
    #[serde(default)]
    pub topology: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub connectivity_hint: Option<String>,
    #[serde(default)]
    pub parameter_source: Option<String>,
    #[serde(default)]
    pub charge_manifest: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PolymerBuildArtifactRef {
    pub build_manifest: String,
    #[serde(default)]
    pub coordinates: Option<String>,
    #[serde(default)]
    pub charge_manifest: Option<String>,
    #[serde(default)]
    pub topology: Option<String>,
    #[serde(default)]
    pub topology_graph: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ComponentSource {
    Artifact {
        artifact: ArtifactRef,
    },
    PolymerBuild {
        polymer_build: PolymerBuildArtifactRef,
    },
}

fn default_component_count() -> usize {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ComponentRequest {
    pub name: String,
    #[serde(default = "default_component_count")]
    #[schemars(default = "default_component_count")]
    pub count: usize,
    pub source: ComponentSource,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BoxRequest {
    pub mode: String,
    #[serde(default)]
    pub padding_angstrom: Option<f32>,
    #[serde(default = "default_cubic")]
    #[schemars(default = "default_cubic")]
    pub shape: String,
    #[serde(default)]
    pub size_angstrom: Option<Value>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct SolventRequest {
    #[serde(default = "default_none")]
    #[schemars(default = "default_none")]
    pub mode: String,
    #[serde(default)]
    pub model: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct NeutralizeConfig {
    #[serde(default = "default_true_bool")]
    #[schemars(default = "default_true_bool")]
    pub enabled: bool,
    #[serde(default, rename = "with")]
    #[schemars(rename = "with")]
    pub with_ion: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum NeutralizeRequest {
    Bool(bool),
    Config(NeutralizeConfig),
}

impl Default for NeutralizeRequest {
    fn default() -> Self {
        Self::Bool(false)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SaltRequest {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub formula: Option<String>,
    #[serde(default)]
    pub species: Option<BTreeMap<String, usize>>,
    #[serde(default)]
    pub molar: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CustomIonSpeciesRequest {
    pub species: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub template: String,
    pub formula_symbol: String,
    pub charge_e: i32,
    pub mass_amu: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CustomSaltRequest {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub formula: Option<String>,
    pub species: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct CustomChemistryCatalogRequest {
    #[serde(default)]
    pub ions: Vec<CustomIonSpeciesRequest>,
    #[serde(default)]
    pub salts: Vec<CustomSaltRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct IonRequest {
    #[serde(default)]
    pub neutralize: NeutralizeRequest,
    #[serde(default)]
    pub salt: Option<SaltRequest>,
    #[serde(default)]
    pub catalog: CustomChemistryCatalogRequest,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub salt_molar: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub cation: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(skip)]
    pub anion: Option<String>,
}

impl Default for IonRequest {
    fn default() -> Self {
        Self {
            neutralize: NeutralizeRequest::default(),
            salt: None,
            catalog: CustomChemistryCatalogRequest::default(),
            salt_molar: None,
            cation: None,
            anion: None,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct MorphologyRequest {
    #[serde(default = "default_single_chain_solution")]
    #[schemars(default = "default_single_chain_solution")]
    pub mode: String,
    #[serde(default)]
    pub alignment_axis: Option<String>,
    #[serde(default)]
    pub target_density_g_cm3: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct EnvironmentRequest {
    #[serde(rename = "box")]
    pub box_spec: BoxRequest,
    #[serde(default)]
    pub solvent: SolventRequest,
    #[serde(default)]
    pub ions: IonRequest,
    #[serde(default)]
    pub morphology: MorphologyRequest,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct OutputRequest {
    pub coordinates: String,
    pub manifest: String,
    #[serde(default)]
    pub md_package: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default = "default_write_conect")]
    #[schemars(default = "default_write_conect")]
    pub write_conect: bool,
    #[serde(default = "default_preserve_topology_graph")]
    #[schemars(default = "default_preserve_topology_graph")]
    pub preserve_topology_graph: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BuildRequest {
    #[serde(
        default = "default_schema_version",
        alias = "version",
        rename = "schema_version"
    )]
    #[schemars(default = "default_schema_version", rename = "schema_version")]
    pub schema_version: String,
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub components: Option<Vec<ComponentRequest>>,
    #[serde(default)]
    pub solute: Option<ArtifactRef>,
    #[serde(default)]
    pub polymer_build: Option<PolymerBuildArtifactRef>,
    pub environment: EnvironmentRequest,
    pub outputs: OutputRequest,
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
pub struct ArtifactEnvelope {
    pub coordinates: String,
    pub manifest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub md_package: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct RunSummary {
    pub component_count: usize,
    pub total_atoms: usize,
    pub water_count: usize,
    pub ion_counts: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct RunSuccessEnvelope {
    pub schema_version: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    pub output_dir: String,
    pub artifacts: ArtifactEnvelope,
    pub summary: RunSummary,
    pub manifest_path: String,
    pub warnings: Vec<ErrorDetail>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct RunErrorEnvelope {
    pub schema_version: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    pub exit_code: i32,
    pub error: ErrorDetail,
    pub errors: Vec<ErrorDetail>,
    pub warnings: Vec<ErrorDetail>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct ValidateSuccessEnvelope {
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
    RunStarted {
        schema_version: String,
        run_id: String,
        elapsed_ms: u64,
    },
    #[serde(rename = "phase_started")]
    PhaseStarted {
        schema_version: String,
        run_id: String,
        phase: String,
        elapsed_ms: u64,
    },
    #[serde(rename = "phase_progress")]
    PhaseProgress {
        schema_version: String,
        run_id: String,
        phase: String,
        progress_pct: f32,
        elapsed_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        eta_ms: Option<u64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        artifact: Option<String>,
    },
    #[serde(rename = "phase_completed")]
    PhaseCompleted {
        schema_version: String,
        run_id: String,
        phase: String,
        elapsed_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        artifact: Option<String>,
    },
    #[serde(rename = "run_completed")]
    RunCompleted {
        schema_version: String,
        run_id: String,
        elapsed_ms: u64,
        final_envelope: RunSuccessEnvelope,
    },
    #[serde(rename = "run_failed")]
    RunFailed {
        schema_version: String,
        run_id: String,
        elapsed_ms: u64,
        final_envelope: RunErrorEnvelope,
    },
}

#[derive(Clone, Debug)]
struct SoluteContext {
    atoms: usize,
    chain_count: usize,
    residue_counts: BTreeMap<String, usize>,
    positions: Vec<[f32; 3]>,
    total_mass_amu: f64,
}

#[derive(Clone, Debug)]
struct TranslationMetadata {
    solute_context: SoluteContext,
    resolved_box_size: [f32; 3],
    water_count: usize,
    ion_counts: BTreeMap<String, usize>,
    salt_pair_count: usize,
    component_inventory: Vec<Value>,
    component_charge_resolution: Vec<Value>,
    source_solute_artifact: Value,
    built_solute_artifact: Option<Value>,
    polymer_build_handoff: Option<Value>,
    polymer_artifact: Option<Value>,
    polymer_controls: Option<Value>,
    charge_manifest_path: Option<String>,
    charge_manifest_paths: Vec<String>,
    charge_source_kinds: Vec<String>,
    net_charge_before_neutralization: Option<f32>,
    neutralization_policy: String,
    warnings: Vec<ErrorDetail>,
    engine_decisions: Value,
}

#[derive(Clone, Debug)]
struct ResolvedComponent {
    name: String,
    count: usize,
    source_kind: String,
    kind: String,
    coordinates_path: String,
    topology: Option<String>,
    topology_graph_path: Option<String>,
    topology_graph: Option<TopologyGraph>,
    charge_manifest_path: Option<String>,
    build_manifest_path: Option<String>,
    forcefield_ref: Option<String>,
    connectivity_hint: Option<String>,
    parameter_source: Option<String>,
    source_detail: Value,
    built_artifact: Option<Value>,
    polymer_build_handoff: Option<Value>,
    polymer_artifact: Option<Value>,
    polymer_controls: Option<Value>,
    context: SoluteContext,
    per_instance_net_charge: Option<f32>,
    charge_source: Option<String>,
    charge_source_kinds: Vec<String>,
    fixed_rotation_euler: Option<[f32; 3]>,
}

#[derive(Serialize, JsonSchema)]
#[serde(untagged)]
#[allow(dead_code)]
enum ResultEnvelopeSchema {
    Success(RunSuccessEnvelope),
    Error(RunErrorEnvelope),
}

fn default_cubic() -> String {
    "cubic".to_string()
}

fn default_none() -> String {
    "none".to_string()
}

fn default_true_bool() -> bool {
    true
}

fn default_write_conect() -> bool {
    true
}

fn default_preserve_topology_graph() -> bool {
    true
}

fn default_na() -> String {
    "Na+".to_string()
}

fn default_cl() -> String {
    "Cl-".to_string()
}

fn default_single_chain_solution() -> String {
    "single_chain_solution".to_string()
}

fn default_schema_version() -> String {
    AGENT_SCHEMA_VERSION.to_string()
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

fn error_detail(
    code: impl Into<String>,
    path: impl Into<Option<String>>,
    message: impl Into<String>,
) -> ErrorDetail {
    ErrorDetail {
        code: code.into(),
        path: path.into().and_then(|value| json_pointer(&value)),
        message: message.into(),
        severity: error_severity(),
    }
}

fn warning_detail(
    code: impl Into<String>,
    path: impl Into<Option<String>>,
    message: impl Into<String>,
) -> ErrorDetail {
    ErrorDetail {
        code: code.into(),
        path: path.into().and_then(|value| json_pointer(&value)),
        message: message.into(),
        severity: warning_severity(),
    }
}

fn component_warning_path(req: &BuildRequest, index: usize, suffix: &str) -> Option<String> {
    let trimmed = suffix.trim_matches('/');
    if req.components.is_some() {
        if trimmed.is_empty() {
            Some(format!("components[{index}]"))
        } else {
            Some(format!("components[{index}].{trimmed}"))
        }
    } else if req.polymer_build.is_some() {
        if trimmed.is_empty() {
            Some("polymer_build".into())
        } else {
            Some(format!("polymer_build.{trimmed}"))
        }
    } else if req.solute.is_some() {
        if trimmed.is_empty() {
            Some("solute".into())
        } else {
            Some(format!("solute.{trimmed}"))
        }
    } else {
        None
    }
}

fn collect_warnings(req: &BuildRequest, components: &[ResolvedComponent]) -> Vec<ErrorDetail> {
    let mut warnings = Vec::new();
    for (index, component) in components.iter().enumerate() {
        if component.charge_manifest_path.is_none()
            && component
                .charge_source_kinds
                .iter()
                .any(|kind| kind == "prmtop.total_charge")
        {
            warnings.push(warning_detail(
                "W_CHARGE_SOURCE_PRMTOP_FALLBACK",
                component_warning_path(req, index, "source"),
                format!(
                    "component '{}' is using prmtop.total_charge fallback; charge_manifest would provide a stronger charge handoff",
                    component.name
                ),
            ));
        }
        if component.source_kind == "polymer_build" && component.topology_graph.is_none() {
            warnings.push(warning_detail(
                "W_TOPOLOGY_GRAPH_MISSING",
                component_warning_path(req, index, "topology_graph"),
                format!(
                    "component '{}' has no topology_graph handoff; warp-pack will fall back to coordinate-derived packing hints",
                    component.name
                ),
            ));
        }
    }
    warnings
}

fn sha256_file(path: &Path) -> PackResult<String> {
    let bytes = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn canonical_json_string(value: &Value) -> PackResult<String> {
    serde_json::to_string(value).map_err(|err| PackError::Parse(err.to_string()))
}

fn hash_value(value: &Value) -> PackResult<String> {
    let text = canonical_json_string(value)?;
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    Ok(format!("{:x}", hasher.finalize()))
}

#[derive(Clone, Debug)]
struct LoadedPolymerBuildHandoff {
    manifest_path: String,
    coordinates_path: String,
    charge_manifest_path: Option<String>,
    topology: Option<String>,
    topology_graph_path: Option<String>,
    topology_graph: Option<TopologyGraph>,
    forcefield_ref: Option<String>,
    manifest: Value,
}

fn resolve_relative(base_path: &str, value: &str) -> String {
    let path = Path::new(value);
    if path.is_absolute() {
        value.to_string()
    } else {
        Path::new(base_path)
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
            .to_string_lossy()
            .to_string()
    }
}

fn supported_topology_graph_version(version: &str) -> bool {
    matches!(version, "warp-build.topology-graph.v5")
}

fn manifest_string_path(manifest: &Value, pointers: &[&str]) -> Option<String> {
    pointers.iter().find_map(|pointer| {
        manifest
            .pointer(pointer)
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    })
}

fn load_polymer_build_handoff(
    handoff: &PolymerBuildArtifactRef,
) -> Result<LoadedPolymerBuildHandoff, ErrorDetail> {
    if handoff.build_manifest.trim().is_empty() {
        return Err(error_detail(
            "E_CONFIG_VALIDATION",
            Some("/polymer_build/build_manifest".into()),
            "polymer_build.build_manifest cannot be empty",
        ));
    }

    let text = fs::read_to_string(&handoff.build_manifest).map_err(|err| {
        error_detail(
            "E_CONFIG_VALIDATION",
            Some("/polymer_build/build_manifest".into()),
            format!("failed to read build manifest: {err}"),
        )
    })?;
    let manifest: Value = serde_json::from_str(&text).map_err(|err| {
        error_detail(
            "E_CONFIG_VALIDATION",
            Some("/polymer_build/build_manifest".into()),
            format!("failed to parse build manifest: {err}"),
        )
    })?;
    let version = manifest
        .get("schema_version")
        .or_else(|| manifest.get("version"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    if version != WARP_BUILD_MANIFEST_VERSION {
        return Err(error_detail(
            "E_CONFIG_VALIDATION",
            Some("/polymer_build/build_manifest".into()),
            format!(
                "unsupported polymer build manifest version '{version}'; expected {WARP_BUILD_MANIFEST_VERSION}"
            ),
        ));
    }

    let coordinates = handoff.coordinates.clone().or_else(|| {
        manifest
            .pointer("/artifacts/coordinates")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    });
    let coordinates = coordinates.ok_or_else(|| {
        error_detail(
            "E_CONFIG_VALIDATION",
            Some("/polymer_build/coordinates".into()),
            "polymer_build handoff requires coordinates or artifacts.coordinates in build manifest",
        )
    })?;
    let charge_manifest_path = handoff.charge_manifest.clone().or_else(|| {
        manifest_string_path(
            &manifest,
            &[
                "/artifacts/charge_manifest",
                "/md_ready_handoff/charge_manifest",
            ],
        )
    });
    let topology = handoff.topology.clone().or_else(|| {
        manifest_string_path(
            &manifest,
            &["/artifacts/topology", "/md_ready_handoff/topology"],
        )
    });
    let topology_graph_path = handoff.topology_graph.clone().or_else(|| {
        manifest_string_path(
            &manifest,
            &[
                "/artifacts/topology_graph",
                "/md_ready_handoff/topology_graph",
            ],
        )
    });
    let forcefield_ref = manifest_string_path(
        &manifest,
        &[
            "/artifacts/forcefield_ref",
            "/md_ready_handoff/forcefield_ref",
        ],
    );
    let topology_graph_path = topology_graph_path
        .as_deref()
        .map(|path| resolve_relative(&handoff.build_manifest, path));
    let topology_graph = if let Some(path) = topology_graph_path.as_deref() {
        let text = fs::read_to_string(path).map_err(|err| {
            error_detail(
                "E_CONFIG_VALIDATION",
                Some("/polymer_build/topology_graph".into()),
                format!("failed to read topology_graph: {err}"),
            )
        })?;
        let graph: TopologyGraph = serde_json::from_str(&text).map_err(|err| {
            error_detail(
                "E_CONFIG_VALIDATION",
                Some("/polymer_build/topology_graph".into()),
                format!("failed to parse topology_graph: {err}"),
            )
        })?;
        let graph_version = graph.schema_version.as_str();
        if !supported_topology_graph_version(graph_version) {
            return Err(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/polymer_build/topology_graph".into()),
                format!("unsupported topology_graph version '{}'", graph_version),
            ));
        }
        if let Some(request_id) = manifest.get("request_id").and_then(Value::as_str) {
            if graph.request_id != request_id {
                return Err(error_detail(
                    "E_CONFIG_VALIDATION",
                    Some("/polymer_build/topology_graph".into()),
                    "topology_graph request_id does not match build manifest",
                ));
            }
        }
        if let Some(bundle_id) = manifest
            .pointer("/source_bundle/bundle_id")
            .and_then(Value::as_str)
        {
            if graph.bundle_id != bundle_id {
                return Err(error_detail(
                    "E_CONFIG_VALIDATION",
                    Some("/polymer_build/topology_graph".into()),
                    "topology_graph bundle_id does not match build manifest",
                ));
            }
        }
        Some(graph)
    } else {
        None
    };

    Ok(LoadedPolymerBuildHandoff {
        manifest_path: handoff.build_manifest.clone(),
        coordinates_path: resolve_relative(&handoff.build_manifest, &coordinates),
        charge_manifest_path: charge_manifest_path
            .as_deref()
            .map(|path| resolve_relative(&handoff.build_manifest, path)),
        topology: topology
            .as_deref()
            .map(|path| resolve_relative(&handoff.build_manifest, path)),
        topology_graph_path,
        topology_graph,
        forcefield_ref: forcefield_ref
            .as_deref()
            .map(|path| resolve_relative(&handoff.build_manifest, path)),
        manifest,
    })
}

fn ensure_parent(path: &str) -> PackResult<()> {
    let parent = Path::new(path).parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    Ok(())
}

fn common_output_dir(coords: &str, manifest: &str) -> String {
    let coords_parent = Path::new(coords).parent().unwrap_or_else(|| Path::new("."));
    let manifest_parent = Path::new(manifest)
        .parent()
        .unwrap_or_else(|| Path::new("."));
    if coords_parent == manifest_parent {
        coords_parent.to_string_lossy().to_string()
    } else {
        manifest_parent.to_string_lossy().to_string()
    }
}

fn infer_output_format(path: &str) -> String {
    Path::new(path)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("pdb")
        .to_ascii_lowercase()
}

fn resolved_output_format(outputs: &OutputRequest) -> String {
    outputs
        .format
        .clone()
        .unwrap_or_else(|| infer_output_format(&outputs.coordinates))
}

fn default_md_package_path(manifest_path: &str) -> String {
    let path = Path::new(manifest_path);
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("system_manifest");
    parent
        .join(format!("{stem}.md-ready.json"))
        .to_string_lossy()
        .to_string()
}

fn resolved_md_package_path(outputs: &OutputRequest) -> String {
    outputs
        .md_package
        .clone()
        .unwrap_or_else(|| default_md_package_path(&outputs.manifest))
}

fn pack_data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../python/warp_md/pack/data")
}

fn water_template_path(model: &str) -> String {
    pack_data_dir()
        .join(format!("{model}.pdb"))
        .to_string_lossy()
        .to_string()
}

fn default_ion_registry_path() -> PathBuf {
    pack_data_dir().join("ions.json")
}

fn default_salt_registry_path() -> PathBuf {
    pack_data_dir().join("salts.json")
}

fn overlay_ion_registry_path() -> Option<PathBuf> {
    std::env::var_os(ION_REGISTRY_ENV).map(PathBuf::from)
}

fn overlay_salt_registry_path() -> Option<PathBuf> {
    std::env::var_os(SALT_REGISTRY_ENV).map(PathBuf::from)
}

fn normalize_ion_species_key(species: &str) -> String {
    species.trim().to_ascii_lowercase()
}

fn normalize_salt_name_key(name: &str) -> String {
    name.trim().to_ascii_lowercase()
}

fn normalize_salt_formula_key(formula: &str) -> String {
    formula
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>()
        .to_ascii_lowercase()
}

fn parse_ion_registry_file(path: &Path) -> PackResult<Vec<IonRegistryEntry>> {
    let text = fs::read_to_string(path).map_err(|err| {
        PackError::Invalid(format!("failed to read ion registry '{}': {err}", path.display()))
    })?;
    let mut file: IonRegistryFile = serde_json::from_str(&text).map_err(|err| {
        PackError::Invalid(format!(
            "failed to parse ion registry '{}': {err}",
            path.display()
        ))
    })?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    for entry in &mut file.ions {
        let template_path = Path::new(&entry.template);
        if template_path.is_relative() {
            entry.template = base.join(template_path).to_string_lossy().to_string();
        }
    }
    Ok(file.ions)
}

fn parse_salt_registry_file(path: &Path) -> PackResult<Vec<SaltRegistryEntry>> {
    let text = fs::read_to_string(path).map_err(|err| {
        PackError::Invalid(format!("failed to read salt registry '{}': {err}", path.display()))
    })?;
    let file: SaltRegistryFile = serde_json::from_str(&text).map_err(|err| {
        PackError::Invalid(format!(
            "failed to parse salt registry '{}': {err}",
            path.display()
        ))
    })?;
    Ok(file.salts)
}

fn build_ion_registry(entries: Vec<IonRegistryEntry>) -> PackResult<IonRegistry> {
    let mut canonical_entries = BTreeMap::new();
    for entry in entries {
        let canonical_key = normalize_ion_species_key(&entry.species);
        if canonical_key.is_empty() {
            return Err(PackError::Invalid(
                "ion registry species names cannot be empty".into(),
            ));
        }
        canonical_entries.insert(canonical_key, entry);
    }

    let mut supported_species = Vec::new();
    let mut by_lookup: BTreeMap<String, IonSpeciesInfo> = BTreeMap::new();
    let mut by_symbol: BTreeMap<String, String> = BTreeMap::new();
    let mut ambiguous_symbols = BTreeSet::new();

    for entry in canonical_entries.into_values() {
        if entry.template.trim().is_empty() {
            return Err(PackError::Invalid(format!(
                "ion registry entry '{}' is missing template",
                entry.species
            )));
        }
        if !Path::new(&entry.template).exists() {
            return Err(PackError::Invalid(format!(
                "ion registry entry '{}' template does not exist: {}",
                entry.species, entry.template
            )));
        }
        if entry.formula_symbol.trim().is_empty() {
            return Err(PackError::Invalid(format!(
                "ion registry entry '{}' is missing formula_symbol",
                entry.species
            )));
        }
        if entry.charge_e == 0 {
            return Err(PackError::Invalid(format!(
                "ion registry entry '{}' must define non-zero charge_e",
                entry.species
            )));
        }
        if entry.mass_amu <= 0.0 {
            return Err(PackError::Invalid(format!(
                "ion registry entry '{}' must define positive mass_amu",
                entry.species
            )));
        }
        if let Some(kind) = entry.topology_kind.as_deref() {
            if !matches!(kind, "single_atom" | "polyatomic") {
                return Err(PackError::Invalid(format!(
                    "ion registry entry '{}' has unsupported topology_kind '{}'",
                    entry.species, kind
                )));
            }
        }
        if let Some(atom_count) = entry.atom_count {
            if atom_count == 0 {
                return Err(PackError::Invalid(format!(
                    "ion registry entry '{}' atom_count must be > 0",
                    entry.species
                )));
            }
        }
        let info = IonSpeciesInfo {
            species: entry.species.clone(),
            template: entry.template.clone(),
            formula_symbol: entry.formula_symbol.clone(),
            charge_e: entry.charge_e,
            mass_amu: entry.mass_amu,
        };
        supported_species.push(info.species.clone());
        let mut lookups = entry.aliases;
        lookups.push(info.species.clone());
        for alias in lookups {
            let key = normalize_ion_species_key(&alias);
            if key.is_empty() {
                return Err(PackError::Invalid(format!(
                    "ion registry entry '{}' has an empty alias",
                    info.species
                )));
            }
            match by_lookup.get(&key) {
                Some(existing) if existing.species != info.species => {
                    return Err(PackError::Invalid(format!(
                        "ion registry alias '{}' conflicts between '{}' and '{}'",
                        alias, existing.species, info.species
                    )));
                }
                Some(_) => {}
                None => {
                    by_lookup.insert(key, info.clone());
                }
            }
        }
        if !ambiguous_symbols.contains(&info.formula_symbol) {
            if let Some(existing_species) =
                by_symbol.insert(info.formula_symbol.clone(), info.species.clone())
            {
                if existing_species != info.species {
                    by_symbol.remove(&info.formula_symbol);
                    ambiguous_symbols.insert(info.formula_symbol.clone());
                }
            }
        }
    }

    Ok(IonRegistry {
        supported_species,
        by_lookup,
        by_symbol,
        ambiguous_symbols,
    })
}

fn base_ion_registry_entries() -> PackResult<Vec<IonRegistryEntry>> {
    let mut entries = parse_ion_registry_file(&default_ion_registry_path())?;
    if let Some(path) = overlay_ion_registry_path() {
        entries.extend(parse_ion_registry_file(&path)?);
    }
    Ok(entries)
}

fn load_ion_registry() -> PackResult<IonRegistry> {
    build_ion_registry(base_ion_registry_entries()?)
}

fn ion_registry() -> PackResult<&'static IonRegistry> {
    match ION_REGISTRY.get_or_init(|| load_ion_registry().map_err(|err| err.to_string())) {
        Ok(registry) => Ok(registry),
        Err(message) => Err(PackError::Invalid(message.clone())),
    }
}

fn supported_ion_species() -> PackResult<Vec<String>> {
    Ok(ion_registry()?.supported_species.clone())
}

fn build_salt_registry(entries: Vec<SaltRegistryEntry>, ions: &IonRegistry) -> PackResult<SaltRegistry> {
    let mut canonical_entries = BTreeMap::new();
    for entry in entries {
        let canonical_key = normalize_salt_name_key(&entry.name);
        if canonical_key.is_empty() {
            return Err(PackError::Invalid(
                "salt registry names cannot be empty".into(),
            ));
        }
        canonical_entries.insert(canonical_key, entry);
    }

    let mut supported_names = Vec::new();
    let mut by_lookup: BTreeMap<String, NormalizedSaltSpec> = BTreeMap::new();
    let mut by_formula: BTreeMap<String, NormalizedSaltSpec> = BTreeMap::new();

    for entry in canonical_entries.into_values() {
        if entry.formula.trim().is_empty() {
            return Err(PackError::Invalid(format!(
                "salt registry entry '{}' is missing formula",
                entry.name
            )));
        }
        validate_salt_species_map_with(ions, &entry.species)?;
        let spec = NormalizedSaltSpec {
            name: Some(entry.name.clone()),
            formula: Some(entry.formula.clone()),
            species: entry.species.clone(),
            molar: None,
        };
        supported_names.push(entry.name.clone());
        let mut lookups = entry.aliases;
        lookups.push(entry.name.clone());
        for alias in lookups {
            let key = normalize_salt_name_key(&alias);
            if key.is_empty() {
                return Err(PackError::Invalid(format!(
                    "salt registry entry '{}' has an empty alias",
                    entry.name
                )));
            }
            match by_lookup.get(&key) {
                Some(existing) if existing.name != spec.name => {
                    return Err(PackError::Invalid(format!(
                        "salt registry alias '{}' conflicts between '{}' and '{}'",
                        alias,
                        existing.name.clone().unwrap_or_default(),
                        entry.name
                    )));
                }
                Some(_) => {}
                None => {
                    by_lookup.insert(key, spec.clone());
                }
            }
        }
        by_formula.insert(normalize_salt_formula_key(&entry.formula), spec);
    }

    Ok(SaltRegistry {
        supported_names,
        by_lookup,
        by_formula,
    })
}

fn base_salt_registry_entries() -> PackResult<Vec<SaltRegistryEntry>> {
    let mut entries = parse_salt_registry_file(&default_salt_registry_path())?;
    if let Some(path) = overlay_salt_registry_path() {
        entries.extend(parse_salt_registry_file(&path)?);
    }
    Ok(entries)
}

fn load_salt_registry() -> PackResult<SaltRegistry> {
    build_salt_registry(base_salt_registry_entries()?, ion_registry()?)
}

fn salt_registry() -> PackResult<&'static SaltRegistry> {
    match SALT_REGISTRY.get_or_init(|| load_salt_registry().map_err(|err| err.to_string())) {
        Ok(registry) => Ok(registry),
        Err(message) => Err(PackError::Invalid(message.clone())),
    }
}

fn supported_salt_names() -> PackResult<Vec<String>> {
    Ok(salt_registry()?.supported_names.clone())
}

fn chemistry_catalog_for_request(ions: &IonRequest) -> PackResult<ChemistryCatalog> {
    let mut ion_entries = base_ion_registry_entries()?;
    ion_entries.extend(ions.catalog.ions.iter().cloned().map(|entry| IonRegistryEntry {
        species: entry.species,
        aliases: entry.aliases,
        template: entry.template,
        formula_symbol: entry.formula_symbol,
        charge_e: entry.charge_e,
        mass_amu: entry.mass_amu,
        topology_kind: None,
        atom_count: None,
    }));
    let ion_registry = build_ion_registry(ion_entries)?;

    let mut salt_entries = base_salt_registry_entries()?;
    for entry in ions.catalog.salts.iter().cloned() {
        let formula = match entry.formula {
            Some(formula) => formula,
            None => canonical_salt_formula_with(&ion_registry, &entry.species)?,
        };
        salt_entries.push(SaltRegistryEntry {
            name: entry.name,
            aliases: entry.aliases,
            formula,
            species: entry.species,
        });
    }
    let salt_registry = build_salt_registry(salt_entries, &ion_registry)?;

    Ok(ChemistryCatalog {
        ions: ion_registry,
        salts: salt_registry,
    })
}

fn named_salt_spec_with(salts: &SaltRegistry, name: &str) -> PackResult<NormalizedSaltSpec> {
    salts
        .by_lookup
        .get(&normalize_salt_name_key(name))
        .cloned()
        .ok_or_else(|| PackError::Invalid(format!("unsupported salt.name '{}'", name)))
}

fn named_salt_spec(name: &str) -> PackResult<NormalizedSaltSpec> {
    named_salt_spec_with(salt_registry()?, name)
}

fn catalog_salt_spec_for_formula_with(
    salts: &SaltRegistry,
    formula: &str,
) -> Option<NormalizedSaltSpec> {
    salts
        .by_formula
        .get(&normalize_salt_formula_key(formula))
        .cloned()
}

fn catalog_salt_spec_for_formula(formula: &str) -> PackResult<Option<NormalizedSaltSpec>> {
    Ok(catalog_salt_spec_for_formula_with(salt_registry()?, formula))
}

fn ion_species_info_with(ions: &IonRegistry, species: &str) -> Option<IonSpeciesInfo> {
    ions.by_lookup
        .get(&normalize_ion_species_key(species))
        .cloned()
}

fn ion_species_info(species: &str) -> Option<IonSpeciesInfo> {
    let registry = ion_registry().ok()?;
    ion_species_info_with(registry, species)
}

fn ion_species_for_formula_symbol_with(ions: &IonRegistry, symbol: &str) -> PackResult<String> {
    if ions.ambiguous_symbols.contains(symbol) {
        return Err(PackError::Invalid(format!(
            "salt.formula symbol '{}' is ambiguous; use salt.species instead",
            symbol
        )));
    }
    ions
        .by_symbol
        .get(symbol)
        .cloned()
        .ok_or_else(|| {
            PackError::Invalid(format!("unsupported salt.formula symbol '{}'", symbol))
        })
}

fn ion_species_for_formula_symbol(symbol: &str) -> PackResult<String> {
    ion_species_for_formula_symbol_with(ion_registry()?, symbol)
}

fn validate_salt_species_map_with(ions: &IonRegistry, species: &BTreeMap<String, usize>) -> PackResult<()> {
    if species.is_empty() {
        return Err(PackError::Invalid(
            "salt.species must include at least one ion species".into(),
        ));
    }
    let mut net_charge = 0i32;
    let mut has_positive = false;
    let mut has_negative = false;
    for (name, count) in species {
        if *count == 0 {
            return Err(PackError::Invalid(
                "salt.species counts must be > 0".into(),
            ));
        }
        let info = ion_species_info_with(ions, name)
            .ok_or_else(|| PackError::Invalid(format!("unsupported ion species '{name}'")))?;
        net_charge += info.charge_e * *count as i32;
        has_positive |= info.charge_e > 0;
        has_negative |= info.charge_e < 0;
    }
    if !has_positive || !has_negative {
        return Err(PackError::Invalid(
            "salt must include at least one cation and one anion".into(),
        ));
    }
    if net_charge != 0 {
        return Err(PackError::Invalid(
            "salt formula/species must be charge neutral".into(),
        ));
    }
    Ok(())
}

fn validate_salt_species_map(species: &BTreeMap<String, usize>) -> PackResult<()> {
    validate_salt_species_map_with(ion_registry()?, species)
}

fn parse_salt_formula_with(
    ions: &IonRegistry,
    salts: &SaltRegistry,
    formula: &str,
) -> PackResult<BTreeMap<String, usize>> {
    if let Some(spec) = catalog_salt_spec_for_formula_with(salts, formula) {
        return Ok(spec.species);
    }
    let compact = formula
        .trim()
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .collect::<String>();
    if compact.is_empty() {
        return Err(PackError::Invalid("salt.formula cannot be empty".into()));
    }
    let mut complex_symbols = ions
        .by_symbol
        .keys()
        .filter(|symbol| !looks_like_simple_formula_symbol(symbol))
        .cloned()
        .collect::<Vec<_>>();
    complex_symbols.sort_by(|left, right| right.len().cmp(&left.len()));
    if let Some(symbol) = complex_symbols.into_iter().find(|symbol| compact.contains(symbol)) {
        return Err(PackError::Invalid(format!(
            "salt.formula '{}' references polyatomic symbol '{}'; use salt.name or salt.species instead",
            formula, symbol
        )));
    }
    let chars = compact.chars().collect::<Vec<_>>();
    let mut idx = 0usize;
    let mut species = BTreeMap::new();
    while idx < chars.len() {
        let first = chars[idx];
        if !first.is_ascii_uppercase() {
            return Err(PackError::Invalid(format!(
                "invalid salt.formula '{}'",
                formula
            )));
        }
        let mut symbol = String::from(first);
        idx += 1;
        while idx < chars.len() && chars[idx].is_ascii_lowercase() {
            symbol.push(chars[idx]);
            idx += 1;
        }
        let mut digits = String::new();
        while idx < chars.len() && chars[idx].is_ascii_digit() {
            digits.push(chars[idx]);
            idx += 1;
        }
        let count = if digits.is_empty() {
            1usize
        } else {
            digits.parse::<usize>().map_err(|_| {
                PackError::Invalid(format!("invalid count in salt.formula '{}'", formula))
            })?
        };
        let species_name = ion_species_for_formula_symbol_with(ions, &symbol).map_err(|err| {
            PackError::Invalid(format!("{err} in '{}'", formula))
        })?;
        *species.entry(species_name.to_string()).or_insert(0) += count;
    }
    validate_salt_species_map_with(ions, &species)?;
    Ok(species)
}

fn parse_salt_formula(formula: &str) -> PackResult<BTreeMap<String, usize>> {
    parse_salt_formula_with(ion_registry()?, salt_registry()?, formula)
}

fn canonical_salt_formula_with(ions: &IonRegistry, species: &BTreeMap<String, usize>) -> PackResult<String> {
    validate_salt_species_map_with(ions, species)?;
    let mut positive = Vec::new();
    let mut negative = Vec::new();
    for (name, count) in species {
        let info = ion_species_info_with(ions, name)
            .ok_or_else(|| PackError::Invalid(format!("unsupported ion species '{name}'")))?;
        let token = if *count == 1 {
            info.formula_symbol.to_string()
        } else {
            format!("{}{}", info.formula_symbol, count)
        };
        if info.charge_e > 0 {
            positive.push(token);
        } else {
            negative.push(token);
        }
    }
    positive.sort();
    negative.sort();
    Ok(positive.into_iter().chain(negative).collect::<String>())
}

fn canonical_salt_formula(species: &BTreeMap<String, usize>) -> PackResult<String> {
    canonical_salt_formula_with(ion_registry()?, species)
}

fn looks_like_simple_formula_symbol(symbol: &str) -> bool {
    let chars = symbol.chars().collect::<Vec<_>>();
    match chars.as_slice() {
        [head] => head.is_ascii_uppercase(),
        [head, tail] => head.is_ascii_uppercase() && tail.is_ascii_lowercase(),
        _ => false,
    }
}

fn salt_stoichiometry_with(ions: &IonRegistry, cation: &str, anion: &str) -> PackResult<BTreeMap<String, usize>> {
    let cation_info = ion_species_info_with(ions, cation)
        .ok_or_else(|| PackError::Invalid(format!("unsupported cation '{cation}'")))?;
    let anion_info = ion_species_info_with(ions, anion)
        .ok_or_else(|| PackError::Invalid(format!("unsupported anion '{anion}'")))?;
    if cation_info.charge_e <= 0 {
        return Err(PackError::Invalid(format!(
            "cation '{cation}' must have positive charge"
        )));
    }
    if anion_info.charge_e >= 0 {
        return Err(PackError::Invalid(format!(
            "anion '{anion}' must have negative charge"
        )));
    }
    let cation_valence = cation_info.charge_e.unsigned_abs() as usize;
    let anion_valence = anion_info.charge_e.unsigned_abs() as usize;
    let gcd = gcd_usize(cation_valence, anion_valence);
    let lcm = (cation_valence / gcd) * anion_valence;
    let mut counts = BTreeMap::new();
    counts.insert(cation.to_string(), lcm / cation_valence);
    counts.insert(anion.to_string(), lcm / anion_valence);
    Ok(counts)
}

fn salt_stoichiometry(cation: &str, anion: &str) -> PackResult<BTreeMap<String, usize>> {
    salt_stoichiometry_with(ion_registry()?, cation, anion)
}

impl NeutralizeRequest {
    fn enabled(&self) -> bool {
        match self {
            Self::Bool(value) => *value,
            Self::Config(config) => config.enabled,
        }
    }

    fn preferred_ion(&self) -> Option<&str> {
        match self {
            Self::Bool(_) => None,
            Self::Config(config) => config.with_ion.as_deref(),
        }
    }
}

impl IonRequest {
    fn legacy_cation(&self) -> String {
        self.cation.clone().unwrap_or_else(default_na)
    }

    fn legacy_anion(&self) -> String {
        self.anion.clone().unwrap_or_else(default_cl)
    }

    fn uses_legacy_salt_fields(&self) -> bool {
        self.salt_molar.is_some() || self.cation.is_some() || self.anion.is_some()
    }

    fn normalized_salt_with(&self, chemistry: &ChemistryCatalog) -> PackResult<Option<NormalizedSaltSpec>> {
        if let Some(salt) = &self.salt {
            if self.uses_legacy_salt_fields() {
                return Err(PackError::Invalid(
                    "use either ions.salt or legacy ions.salt_molar/cation/anion, not both"
                        .into(),
                ));
            }
            let populated = [
                salt.name.as_ref().map(|_| "name"),
                salt.formula.as_ref().map(|_| "formula"),
                salt.species.as_ref().map(|_| "species"),
            ]
            .into_iter()
            .flatten()
            .count();
            if populated > 1 {
                return Err(PackError::Invalid(
                    "ions.salt accepts exactly one of name, formula, or species".into(),
                ));
            }
            let mut normalized = match (&salt.name, &salt.formula, &salt.species) {
                (None, None, None) => {
                    return Err(PackError::Invalid(
                        "ions.salt requires name, formula, or species".into(),
                    ))
                }
                (Some(name), None, None) => named_salt_spec_with(&chemistry.salts, name)?,
                (None, Some(formula), None) => {
                    if let Some(spec) = catalog_salt_spec_for_formula_with(&chemistry.salts, formula) {
                        spec
                    } else {
                        NormalizedSaltSpec {
                            name: None,
                            formula: Some(formula.clone()),
                            species: parse_salt_formula_with(&chemistry.ions, &chemistry.salts, formula)?,
                            molar: None,
                        }
                    }
                }
                (None, None, Some(species)) => {
                    validate_salt_species_map_with(&chemistry.ions, species)?;
                    NormalizedSaltSpec {
                        name: None,
                        formula: Some(canonical_salt_formula_with(&chemistry.ions, species)?),
                        species: species.clone(),
                        molar: None,
                    }
                }
                _ => unreachable!(),
            };
            normalized.molar = salt.molar;
            if normalized.formula.is_none() {
                normalized.formula =
                    Some(canonical_salt_formula_with(&chemistry.ions, &normalized.species)?);
            }
            Ok(Some(normalized))
        } else if self.salt_molar.is_some() {
            let cation = self.legacy_cation();
            let anion = self.legacy_anion();
            Ok(Some(NormalizedSaltSpec {
                name: None,
                formula: Some(canonical_salt_formula_with(
                    &chemistry.ions,
                    &salt_stoichiometry_with(&chemistry.ions, &cation, &anion)?,
                )?),
                species: salt_stoichiometry_with(&chemistry.ions, &cation, &anion)?,
                molar: self.salt_molar,
            }))
        } else {
            Ok(None)
        }
    }

    fn normalized_salt(&self) -> PackResult<Option<NormalizedSaltSpec>> {
        self.normalized_salt_with(&ChemistryCatalog {
            ions: ion_registry()?.clone(),
            salts: salt_registry()?.clone(),
        })
    }

    fn counterion_for_charge_with(
        &self,
        chemistry: &ChemistryCatalog,
        need_positive: bool,
    ) -> PackResult<String> {
        if let Some(preferred) = self.neutralize.preferred_ion() {
            let info = ion_species_info_with(&chemistry.ions, preferred).ok_or_else(|| {
                PackError::Invalid(format!(
                    "unsupported neutralize.with ion species '{}'",
                    preferred
                ))
            })?;
            if need_positive && info.charge_e <= 0 {
                return Err(PackError::Invalid(
                    "neutralize.with must be a cation for negatively charged systems".into(),
                ));
            }
            if !need_positive && info.charge_e >= 0 {
                return Err(PackError::Invalid(
                    "neutralize.with must be an anion for positively charged systems".into(),
                ));
            }
            return Ok(preferred.to_string());
        }
        if let Some(salt) = self.normalized_salt_with(chemistry)? {
            if let Some((species, _)) = salt.species.iter().find(|(species, _)| {
                ion_species_info_with(&chemistry.ions, species)
                    .map(|info| {
                        (need_positive && info.charge_e > 0)
                            || (!need_positive && info.charge_e < 0)
                    })
                    .unwrap_or(false)
            }) {
                return Ok(species.clone());
            }
        }
        Ok(if need_positive {
            self.legacy_cation()
        } else {
            self.legacy_anion()
        })
    }

    fn counterion_for_charge(&self, need_positive: bool) -> PackResult<String> {
        self.counterion_for_charge_with(
            &ChemistryCatalog {
                ions: ion_registry()?.clone(),
                salts: salt_registry()?.clone(),
            },
            need_positive,
        )
    }
}

fn gcd_usize(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a.max(1)
}

fn ion_template_path_with(ions: &IonRegistry, species: &str) -> String {
    let template = ion_species_info_with(ions, species)
        .map(|info| info.template)
        .unwrap_or_else(|| {
            pack_data_dir()
                .join(format!("{}.pdb", normalize_ion_species_key(species)))
                .to_string_lossy()
                .to_string()
        });
    template
}

fn ion_template_path(species: &str) -> String {
    match ion_registry() {
        Ok(registry) => ion_template_path_with(registry, species),
        Err(_) => pack_data_dir()
            .join(format!("{}.pdb", normalize_ion_species_key(species)))
            .to_string_lossy()
            .to_string(),
    }
}

fn parse_request_text(text: &str) -> Result<BuildRequest, ErrorDetail> {
    let payload: Value = serde_json::from_str(text)
        .map_err(|err| error_detail("E_CONFIG_VALIDATION", None, err.to_string()))?;
    if payload.get("polymer").is_some() {
        return Err(error_detail(
            "E_UNSUPPORTED_FEATURE",
            Some("polymer".into()),
            "inline polymer build was removed from warp-pack; build polymers with warp-build and pass polymer_build or components",
        ));
    }
    let deserializer = payload.into_deserializer();
    serde_path_to_error::deserialize::<_, BuildRequest>(deserializer).map_err(|err| {
        let path = err.path().to_string();
        error_detail(
            if path == "version" || path == "schema_version" {
                "E_CONFIG_VERSION"
            } else {
                "E_CONFIG_VALIDATION"
            },
            if path.is_empty() { None } else { Some(path) },
            err.inner().to_string(),
        )
    })
}

fn validate_request(req: &BuildRequest) -> Vec<ErrorDetail> {
    let mut errors = Vec::new();
    let chemistry = match chemistry_catalog_for_request(&req.environment.ions) {
        Ok(catalog) => Some(catalog),
        Err(err) => {
            errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/environment/ions/catalog".into()),
                err.to_string(),
            ));
            None
        }
    };

    if req.schema_version != AGENT_SCHEMA_VERSION {
        errors.push(error_detail(
            "E_CONFIG_VERSION",
            Some("schema_version".into()),
            format!(
                "unsupported schema_version '{}'; expected {AGENT_SCHEMA_VERSION}",
                req.schema_version
            ),
        ));
    }

    let input_count = usize::from(req.components.is_some())
        + usize::from(req.solute.is_some())
        + usize::from(req.polymer_build.is_some());
    if input_count == 0 {
        errors.push(error_detail(
            "E_CONFIG_VALIDATION",
            Some("components".into()),
            "provide one of components, solute, or polymer_build",
        ));
    } else if input_count > 1 {
        errors.push(error_detail(
            "E_CONFIG_VALIDATION",
            Some("components".into()),
            "provide exactly one of components, solute, or polymer_build",
        ));
    }

    if req.outputs.coordinates.trim().is_empty() {
        errors.push(error_detail(
            "E_CONFIG_VALIDATION",
            Some("/outputs/coordinates".into()),
            "outputs.coordinates cannot be empty",
        ));
    }
    if req.outputs.manifest.trim().is_empty() {
        errors.push(error_detail(
            "E_CONFIG_VALIDATION",
            Some("/outputs/manifest".into()),
            "outputs.manifest cannot be empty",
        ));
    }
    if req.outputs.coordinates == req.outputs.manifest {
        errors.push(error_detail(
            "E_CONFIG_VALIDATION",
            Some("/outputs/manifest".into()),
            "outputs.coordinates and outputs.manifest must differ",
        ));
    }
    let resolved_format = resolved_output_format(&req.outputs);
    if !SUPPORTED_OUTPUT_FORMATS.contains(&resolved_format.as_str()) {
        errors.push(error_detail(
            "E_CONFIG_VALIDATION",
            Some("/outputs/format".into()),
            format!("unsupported output format '{resolved_format}'"),
        ));
    }
    let md_package = resolved_md_package_path(&req.outputs);
    if md_package == req.outputs.coordinates || md_package == req.outputs.manifest {
        errors.push(error_detail(
            "E_CONFIG_VALIDATION",
            Some("/outputs/md_package".into()),
            "outputs.md_package must differ from outputs.coordinates and outputs.manifest",
        ));
    }

    match req.environment.box_spec.mode.as_str() {
        "padding" => {
            if req.environment.box_spec.padding_angstrom.unwrap_or(0.0) <= 0.0 {
                errors.push(error_detail(
                    "E_CONFIG_VALIDATION",
                    Some("/environment/box/padding_angstrom".into()),
                    "padding mode requires positive padding_angstrom",
                ));
            }
        }
        "fixed_size" => {
            if req.environment.box_spec.size_angstrom.is_none()
                && req.environment.morphology.target_density_g_cm3.is_none()
            {
                errors.push(error_detail(
                    "E_CONFIG_VALIDATION",
                    Some("/environment/box/size_angstrom".into()),
                    "fixed_size mode requires size_angstrom or target_density_g_cm3",
                ));
            }
        }
        other => errors.push(error_detail(
            "E_CONFIG_VALIDATION",
            Some("/environment/box/mode".into()),
            format!("unsupported box mode '{other}'"),
        )),
    }

    match req.environment.box_spec.shape.as_str() {
        "cubic" | "orthorhombic" => {}
        other => errors.push(error_detail(
            "E_CONFIG_VALIDATION",
            Some("/environment/box/shape".into()),
            format!("unsupported box shape '{other}'"),
        )),
    }

    match req.environment.solvent.mode.as_str() {
        "none" => {}
        "explicit" => match req.environment.solvent.model.as_deref() {
            Some(model) if SUPPORTED_SOLVENT_MODELS.contains(&model) => {}
            Some(model) => errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/environment/solvent/model".into()),
                format!("unsupported solvent model '{model}'"),
            )),
            None => errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/environment/solvent/model".into()),
                "explicit solvent mode requires solvent.model",
            )),
        },
        other => errors.push(error_detail(
            "E_CONFIG_VALIDATION",
            Some("/environment/solvent/mode".into()),
            format!("unsupported solvent mode '{other}'"),
        )),
    }

    if let Some(cation) = req.environment.ions.cation.as_deref() {
        let cation_info = chemistry
            .as_ref()
            .and_then(|catalog| ion_species_info_with(&catalog.ions, cation));
        if cation_info.is_none() {
            errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/environment/ions/cation".into()),
                format!("unsupported cation '{}'", cation),
            ));
        } else if cation_info.map(|info| info.charge_e <= 0).unwrap_or(false) {
            errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/environment/ions/cation".into()),
                "cation must have positive charge",
            ));
        }
    }
    if let Some(anion) = req.environment.ions.anion.as_deref() {
        let anion_info = chemistry
            .as_ref()
            .and_then(|catalog| ion_species_info_with(&catalog.ions, anion));
        if anion_info.is_none() {
            errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/environment/ions/anion".into()),
                format!("unsupported anion '{}'", anion),
            ));
        } else if anion_info.map(|info| info.charge_e >= 0).unwrap_or(false) {
            errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/environment/ions/anion".into()),
                "anion must have negative charge",
            ));
        }
    }
    if req.environment.ions.cation.is_some()
        && req.environment.ions.anion.is_some()
        && req.environment.ions.cation == req.environment.ions.anion
    {
        errors.push(error_detail(
            "E_CONFIG_VALIDATION",
            Some("/environment/ions/anion".into()),
            "cation and anion must differ",
        ));
    }
    if req.environment.ions.salt_molar.unwrap_or(0.0) < 0.0 {
        errors.push(error_detail(
            "E_CONFIG_VALIDATION",
            Some("/environment/ions/salt_molar".into()),
            "salt_molar must be >= 0",
        ));
    }
    if let Some(salt) = req.environment.ions.salt.as_ref() {
        if salt.molar.unwrap_or(0.0) < 0.0 {
            errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/environment/ions/salt/molar".into()),
                "salt.molar must be >= 0",
            ));
        }
    }
    if let Some(with_ion) = req.environment.ions.neutralize.preferred_ion() {
        if chemistry
            .as_ref()
            .and_then(|catalog| ion_species_info_with(&catalog.ions, with_ion))
            .is_none()
        {
            errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/environment/ions/neutralize/with".into()),
                format!("unsupported neutralize.with ion species '{}'", with_ion),
            ));
        }
    }
    if let Some(catalog) = chemistry.as_ref() {
        if let Err(err) = req.environment.ions.normalized_salt_with(catalog) {
            errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/environment/ions/salt".into()),
                err.to_string(),
            ));
        }
    }

    if !SUPPORTED_MORPHOLOGY_MODES.contains(&req.environment.morphology.mode.as_str()) {
        errors.push(error_detail(
            "E_UNSUPPORTED_FEATURE",
            Some("/environment/morphology/mode".into()),
            format!(
                "morphology mode '{}' is not executable; supported: {}",
                req.environment.morphology.mode,
                SUPPORTED_MORPHOLOGY_MODES.join(", ")
            ),
        ));
    }
    if let Some(target_density) = req.environment.morphology.target_density_g_cm3 {
        if target_density <= 0.0 {
            errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/environment/morphology/target_density_g_cm3".into()),
                "target_density_g_cm3 must be > 0",
            ));
        }
        if req.environment.morphology.mode == "single_chain_solution" {
            errors.push(error_detail(
                "E_UNSUPPORTED_FEATURE",
                Some("/environment/morphology/target_density_g_cm3".into()),
                "target_density_g_cm3 is only executable for bulk morphologies",
            ));
        }
        if req.environment.solvent.mode != "none"
            && req.environment.box_spec.size_angstrom.is_none()
        {
            errors.push(error_detail(
                "E_UNSUPPORTED_FEATURE",
                Some("/environment/solvent/mode".into()),
                "target_density_g_cm3 with explicit solvent currently requires fixed box size_angstrom",
            ));
        }
    }
    match req.environment.morphology.mode.as_str() {
        "single_chain_solution" => {}
        "amorphous_bulk" | "backbone_aligned_bulk" => {
            if req.environment.box_spec.mode != "fixed_size" {
                errors.push(error_detail(
                    "E_CONFIG_VALIDATION",
                    Some("/environment/box/mode".into()),
                    format!(
                        "{} requires fixed_size box mode",
                        req.environment.morphology.mode
                    ),
                ));
            }
            if req.environment.box_spec.size_angstrom.is_none()
                && req.environment.morphology.target_density_g_cm3.is_none()
            {
                errors.push(error_detail(
                    "E_CONFIG_VALIDATION",
                    Some("/environment/box/size_angstrom".into()),
                    format!(
                        "{} requires size_angstrom or target_density_g_cm3",
                        req.environment.morphology.mode
                    ),
                ));
            }
        }
        _ => {}
    }
    if req.environment.morphology.mode == "backbone_aligned_bulk"
        && req
            .environment
            .morphology
            .alignment_axis
            .as_deref()
            .is_none()
    {
        errors.push(error_detail(
            "E_CONFIG_VALIDATION",
            Some("/environment/morphology/alignment_axis".into()),
            "backbone_aligned_bulk requires alignment_axis",
        ));
    }
    if let Some(axis) = req.environment.morphology.alignment_axis.as_deref() {
        if let Err(error) = parse_alignment_axis(axis) {
            errors.push(error);
        }
    }

    let mut total_component_instances = 0usize;

    if let Some(components) = &req.components {
        if components.is_empty() {
            errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("components".into()),
                "components cannot be empty",
            ));
        }
        for (idx, component) in components.iter().enumerate() {
            total_component_instances += component.count;
            if component.name.trim().is_empty() {
                errors.push(error_detail(
                    "E_CONFIG_VALIDATION",
                    Some(format!("components[{idx}].name")),
                    "component name cannot be empty",
                ));
            }
            if component.count == 0 {
                errors.push(error_detail(
                    "E_CONFIG_VALIDATION",
                    Some(format!("components[{idx}].count")),
                    "component count must be >= 1",
                ));
            }
            match &component.source {
                ComponentSource::Artifact { artifact } => {
                    if artifact.path.trim().is_empty() {
                        errors.push(error_detail(
                            "E_CONFIG_VALIDATION",
                            Some(format!("components[{idx}].source.artifact.path")),
                            "artifact path cannot be empty",
                        ));
                    }
                    if let Some(kind) = &artifact.kind {
                        if !SUPPORTED_COMPONENT_KINDS.contains(&kind.as_str()) {
                            errors.push(error_detail(
                                "E_CONFIG_VALIDATION",
                                Some(format!("components[{idx}].source.artifact.kind")),
                                format!("unsupported component kind '{kind}'"),
                            ));
                        }
                    }
                    if req.environment.ions.neutralize.enabled()
                        && artifact.charge_manifest.is_none()
                        && artifact.topology.is_none()
                    {
                        errors.push(error_detail(
                            "E_CONFIG_VALIDATION",
                            Some(format!("components[{idx}].source.artifact.charge_manifest")),
                            "neutralize requires component charge_manifest or topology",
                        ));
                    }
                    if req.environment.morphology.mode == "backbone_aligned_bulk"
                        && artifact.kind.as_deref() != Some("polymer_chain")
                    {
                        errors.push(error_detail(
                            "E_CONFIG_VALIDATION",
                            Some(format!("components[{idx}].source.artifact.kind")),
                            "backbone_aligned_bulk requires polymer_chain components",
                        ));
                    }
                }
                ComponentSource::PolymerBuild { polymer_build } => {
                    match load_polymer_build_handoff(polymer_build) {
                        Ok(loaded) => {
                            if req.environment.ions.neutralize.enabled()
                                && loaded.charge_manifest_path.is_none()
                                && loaded.topology.is_none()
                            {
                                errors.push(error_detail(
                                    "E_CONFIG_VALIDATION",
                                    Some(format!(
                                        "components[{idx}].source.polymer_build.charge_manifest"
                                    )),
                                    "neutralize requires polymer_build charge_manifest or topology",
                                ));
                            }
                        }
                        Err(error) => {
                            errors.push(error_detail(
                                error.code,
                                error.path.map(|path| {
                                    path.replace(
                                        "/polymer_build",
                                        &format!("/components/{idx}/source/polymer_build"),
                                    )
                                }),
                                error.message,
                            ));
                        }
                    }
                }
            }
        }
    }

    if let Some(solute) = &req.solute {
        total_component_instances += 1;
        if solute.path.trim().is_empty() {
            errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/solute/path".into()),
                "solute.path cannot be empty",
            ));
        }
        if let Some(kind) = &solute.kind {
            if !SUPPORTED_COMPONENT_KINDS.contains(&kind.as_str()) {
                errors.push(error_detail(
                    "E_CONFIG_VALIDATION",
                    Some("/solute/kind".into()),
                    format!("unsupported solute kind '{kind}'"),
                ));
            }
        }
        if req.environment.ions.neutralize.enabled()
            && solute.charge_manifest.is_none()
            && solute.topology.is_none()
        {
            errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/solute/charge_manifest".into()),
                "neutralize requires solute.charge_manifest or solute.topology",
            ));
        }
        if req.environment.morphology.mode == "backbone_aligned_bulk"
            && solute.kind.as_deref() != Some("polymer_chain")
        {
            errors.push(error_detail(
                "E_CONFIG_VALIDATION",
                Some("/solute/kind".into()),
                "backbone_aligned_bulk requires polymer_chain inputs",
            ));
        }
    }

    if let Some(polymer_build) = &req.polymer_build {
        total_component_instances += 1;
        match load_polymer_build_handoff(polymer_build) {
            Ok(loaded) => {
                if req.environment.ions.neutralize.enabled()
                    && loaded.charge_manifest_path.is_none()
                    && loaded.topology.is_none()
                {
                    errors.push(error_detail(
                        "E_CONFIG_VALIDATION",
                        Some("/polymer_build/charge_manifest".into()),
                        "neutralize requires polymer_build charge_manifest or topology",
                    ));
                }
            }
            Err(error) => errors.push(error),
        }
    }

    if req.environment.morphology.mode == "single_chain_solution" && total_component_instances != 1
    {
        errors.push(error_detail(
            "E_CONFIG_VALIDATION",
            Some("components".into()),
            "single_chain_solution requires exactly one total component instance",
        ));
    }

    errors
}

fn load_agent_request(text: &str) -> Result<BuildRequest, RunErrorEnvelope> {
    match parse_request_text(text) {
        Ok(req) => {
            let errors = validate_request(&req);
            if errors.is_empty() {
                Ok(req)
            } else {
                let first = errors[0].clone();
                Err(RunErrorEnvelope {
                    schema_version: AGENT_SCHEMA_VERSION.into(),
                    status: "error".into(),
                    run_id: req.run_id.clone(),
                    exit_code: 2,
                    error: first,
                    errors,
                    warnings: Vec::new(),
                })
            }
        }
        Err(error) => Err(RunErrorEnvelope {
            schema_version: AGENT_SCHEMA_VERSION.into(),
            status: "error".into(),
            run_id: None,
            exit_code: 2,
            error: error.clone(),
            errors: vec![error],
            warnings: Vec::new(),
        }),
    }
}

fn value_to_triplet(value: &Value) -> PackResult<[f32; 3]> {
    match value {
        Value::Number(number) => {
            let side = number
                .as_f64()
                .ok_or_else(|| PackError::Invalid("size_angstrom must be numeric".into()))?
                as f32;
            Ok([side, side, side])
        }
        Value::Array(values) if values.len() == 3 => Ok([
            values[0]
                .as_f64()
                .ok_or_else(|| PackError::Invalid("size_angstrom[0] must be numeric".into()))?
                as f32,
            values[1]
                .as_f64()
                .ok_or_else(|| PackError::Invalid("size_angstrom[1] must be numeric".into()))?
                as f32,
            values[2]
                .as_f64()
                .ok_or_else(|| PackError::Invalid("size_angstrom[2] must be numeric".into()))?
                as f32,
        ]),
        _ => Err(PackError::Invalid(
            "environment.box.size_angstrom must be a number or 3-vector".into(),
        )),
    }
}

fn load_solute_context(
    path: &str,
    topology: Option<&str>,
) -> PackResult<(MoleculeData, SoluteContext)> {
    let ext = Path::new(path)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mol = read_molecule(
        Path::new(path),
        Some(&ext),
        false,
        true,
        topology.map(Path::new),
    )?;

    let mut chains = BTreeSet::new();
    let mut residue_counts = BTreeMap::new();
    let mut residue_seen = BTreeSet::new();
    let mut positions = Vec::with_capacity(mol.atoms.len());
    let mut total_mass_amu = 0.0f64;
    for atom in &mol.atoms {
        chains.insert(atom.chain);
        positions.push(atom.position.to_array());
        total_mass_amu += atomic_mass_amu(&atom.element);
        if residue_seen.insert((atom.chain, atom.resid, atom.resname.clone())) {
            *residue_counts.entry(atom.resname.clone()).or_insert(0) += 1;
        }
    }

    Ok((
        mol.clone(),
        SoluteContext {
            atoms: mol.atoms.len(),
            chain_count: chains.len().max(1),
            residue_counts,
            positions,
            total_mass_amu,
        },
    ))
}

fn bounds_from_positions(positions: &[[f32; 3]]) -> PackResult<([f32; 3], [f32; 3])> {
    if positions.is_empty() {
        return Err(PackError::Invalid("solute has no atoms".into()));
    }

    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for pos in positions {
        for axis in 0..3 {
            min[axis] = min[axis].min(pos[axis]);
            max[axis] = max[axis].max(pos[axis]);
        }
    }
    Ok((min, max))
}

fn bounding_box_volume(positions: &[[f32; 3]]) -> PackResult<f64> {
    let (min, max) = bounds_from_positions(positions)?;
    Ok(((max[0] - min[0]).max(0.0) as f64)
        * ((max[1] - min[1]).max(0.0) as f64)
        * ((max[2] - min[2]).max(0.0) as f64))
}

fn atomic_mass_amu(element: &str) -> f64 {
    match element.trim().to_ascii_uppercase().as_str() {
        "H" => 1.008,
        "C" => 12.011,
        "N" => 14.007,
        "O" => 15.999,
        "F" => 18.998,
        "NA" => 22.990,
        "MG" => 24.305,
        "P" => 30.974,
        "S" => 32.06,
        "CL" => 35.45,
        "K" => 39.098,
        "CA" => 40.078,
        "ZN" => 65.38,
        other if other.is_empty() => 12.011,
        _ => 12.011,
    }
}

fn density_target_box_size(
    box_req: &BoxRequest,
    context: &SoluteContext,
    target_density_g_cm3: f32,
) -> PackResult<[f32; 3]> {
    if target_density_g_cm3 <= 0.0 {
        return Err(PackError::Invalid(
            "target_density_g_cm3 must be > 0".into(),
        ));
    }
    let aspect = if let Some(size) = box_req.size_angstrom.as_ref() {
        value_to_triplet(size)?
    } else {
        let (min, max) = bounds_from_positions(&context.positions)?;
        [
            (max[0] - min[0]).max(1.0),
            (max[1] - min[1]).max(1.0),
            (max[2] - min[2]).max(1.0),
        ]
    };
    let mut ratios = aspect.map(|value| value.max(1.0e-3) as f64);
    if box_req.shape == "cubic" {
        ratios = [1.0, 1.0, 1.0];
    }
    let target_volume_ang3 =
        (context.total_mass_amu * 1.660_539_066_60f64) / target_density_g_cm3 as f64;
    let ratio_product = ratios[0] * ratios[1] * ratios[2];
    let scale = (target_volume_ang3 / ratio_product.max(1.0e-12)).cbrt();
    Ok([
        (ratios[0] * scale) as f32,
        (ratios[1] * scale) as f32,
        (ratios[2] * scale) as f32,
    ])
}

fn merge_solute_contexts(components: &[ResolvedComponent]) -> PackResult<SoluteContext> {
    let mut positions = Vec::new();
    let mut residue_counts = BTreeMap::new();
    let mut atoms = 0usize;
    let mut chain_count = 0usize;
    let mut total_mass_amu = 0.0f64;
    for component in components {
        atoms += component.context.atoms * component.count;
        chain_count += component.context.chain_count * component.count;
        total_mass_amu += component.context.total_mass_amu * component.count as f64;
        for (resname, count) in &component.context.residue_counts {
            *residue_counts.entry(resname.clone()).or_insert(0) += count * component.count;
        }
        if components.len() == 1 && component.count == 1 {
            positions.extend(component.context.positions.iter().copied());
        }
    }
    if positions.is_empty() {
        positions.push([0.0, 0.0, 0.0]);
    }
    Ok(SoluteContext {
        atoms,
        chain_count: chain_count.max(1),
        residue_counts,
        positions,
        total_mass_amu,
    })
}

fn total_component_instances(components: &[ResolvedComponent]) -> usize {
    components.iter().map(|component| component.count).sum()
}

fn parse_alignment_axis(value: &str) -> Result<Vec3, ErrorDetail> {
    match value.trim().to_ascii_lowercase().as_str() {
        "x" | "+x" => Ok(Vec3::new(1.0, 0.0, 0.0)),
        "-x" => Ok(Vec3::new(-1.0, 0.0, 0.0)),
        "y" | "+y" => Ok(Vec3::new(0.0, 1.0, 0.0)),
        "-y" => Ok(Vec3::new(0.0, -1.0, 0.0)),
        "z" | "+z" => Ok(Vec3::new(0.0, 0.0, 1.0)),
        "-z" => Ok(Vec3::new(0.0, 0.0, -1.0)),
        _ => Err(error_detail(
            "E_CONFIG_VALIDATION",
            Some("/environment/morphology/alignment_axis".into()),
            "alignment_axis must be one of x, y, z, -x, -y, -z",
        )),
    }
}

fn normalize_vec3(value: Vec3) -> Vec3 {
    let norm = value.norm();
    if norm > 1.0e-6 {
        value.scale(1.0 / norm)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    }
}

fn principal_axis_for_context(context: &SoluteContext) -> Vec3 {
    let points = context
        .positions
        .iter()
        .map(|position| Vec3::from_array(*position))
        .collect::<Vec<_>>();
    let center = center_of_geometry(&points);
    let mut i_xx = 0.0f64;
    let mut i_yy = 0.0f64;
    let mut i_zz = 0.0f64;
    let mut i_xy = 0.0f64;
    let mut i_xz = 0.0f64;
    let mut i_yz = 0.0f64;
    for point in &points {
        let rel = point.sub(center);
        let x = rel.x as f64;
        let y = rel.y as f64;
        let z = rel.z as f64;
        i_xx += y * y + z * z;
        i_yy += x * x + z * z;
        i_zz += x * x + y * y;
        i_xy -= x * y;
        i_xz -= x * z;
        i_yz -= y * z;
    }
    let (axes, _vals) = principal_axes_from_inertia(i_xx, i_yy, i_zz, i_xy, i_xz, i_yz);
    let axis = Vec3::new(
        axes[(0, 0)] as f32,
        axes[(1, 0)] as f32,
        axes[(2, 0)] as f32,
    );
    normalize_vec3(axis)
}

fn graph_alignment_axis(graph: &TopologyGraph) -> Option<Vec3> {
    let path = graph.alignment_paths.first()?;
    let first = path.residue_ids.first().copied()?;
    let last = path.residue_ids.last().copied()?;
    if first == last {
        return None;
    }
    let residue_center = |resid: usize| -> Option<Vec3> {
        let points = graph
            .atoms
            .iter()
            .filter(|atom| atom.resid == resid as i32)
            .map(|atom| Vec3::new(atom.position[0], atom.position[1], atom.position[2]))
            .collect::<Vec<_>>();
        if points.is_empty() {
            None
        } else {
            Some(center_of_geometry(&points))
        }
    };
    let start = residue_center(first)?;
    let end = residue_center(last)?;
    Some(normalize_vec3(end.sub(start)))
}

fn component_principal_axis(component: &ResolvedComponent) -> Vec3 {
    component
        .topology_graph
        .as_ref()
        .and_then(graph_alignment_axis)
        .unwrap_or_else(|| principal_axis_for_context(&component.context))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphArchitecture {
    Linear,
    Branched,
    Cyclic,
    BranchedCyclic,
}

#[derive(Clone, Debug)]
struct GraphPackingHelper {
    constraints: Vec<ConstraintSpec>,
    policy: Value,
}

#[derive(Clone, Debug, Default)]
struct GraphPackingCohort {
    typed_component_count: usize,
    typed_instance_count: usize,
    linear_count: usize,
    branched_count: usize,
    cyclic_count: usize,
    branched_cyclic_count: usize,
}

impl GraphPackingCohort {
    fn distinct_architecture_count(&self) -> usize {
        [
            self.linear_count,
            self.branched_count,
            self.cyclic_count,
            self.branched_cyclic_count,
        ]
        .into_iter()
        .filter(|count| *count > 0)
        .count()
    }

    fn mixed_architecture(&self) -> bool {
        self.distinct_architecture_count() > 1
    }
}

fn graph_architecture(graph: &TopologyGraph) -> GraphArchitecture {
    match (
        !graph.branch_points.is_empty(),
        !graph.cycle_basis.is_empty(),
    ) {
        (false, false) => GraphArchitecture::Linear,
        (true, false) => GraphArchitecture::Branched,
        (false, true) => GraphArchitecture::Cyclic,
        (true, true) => GraphArchitecture::BranchedCyclic,
    }
}

fn graph_architecture_label(graph: &TopologyGraph) -> &'static str {
    match graph_architecture(graph) {
        GraphArchitecture::Linear => "linear",
        GraphArchitecture::Branched => "branched",
        GraphArchitecture::Cyclic => "cyclic",
        GraphArchitecture::BranchedCyclic => "branched_cyclic",
    }
}

fn graph_packing_cohort(components: &[ResolvedComponent]) -> GraphPackingCohort {
    let mut cohort = GraphPackingCohort::default();
    for component in components {
        let Some(graph) = component.topology_graph.as_ref() else {
            continue;
        };
        cohort.typed_component_count += 1;
        cohort.typed_instance_count += component.count;
        match graph_architecture(graph) {
            GraphArchitecture::Linear => cohort.linear_count += component.count,
            GraphArchitecture::Branched => cohort.branched_count += component.count,
            GraphArchitecture::Cyclic => cohort.cyclic_count += component.count,
            GraphArchitecture::BranchedCyclic => cohort.branched_cyclic_count += component.count,
        }
    }
    cohort
}

fn graph_alignment_source(graph: &TopologyGraph) -> &'static str {
    if graph_alignment_axis(graph).is_some() {
        "topology_graph.alignment_path"
    } else {
        "principal_axis_fallback"
    }
}

fn dominant_axis_index(axis: Vec3) -> usize {
    let values = [axis.x.abs(), axis.y.abs(), axis.z.abs()];
    if values[0] >= values[1] && values[0] >= values[2] {
        0
    } else if values[1] >= values[2] {
        1
    } else {
        2
    }
}

fn clamp_region_bounds(
    mut min: [f32; 3],
    mut max: [f32; 3],
    box_size: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    for axis in 0..3 {
        let lower = min[axis].clamp(0.0, box_size[axis]);
        let upper = max[axis].clamp(0.0, box_size[axis]);
        let span = (upper - lower).max((box_size[axis] * 0.12).min(8.0).max(2.0));
        min[axis] = lower.min(box_size[axis] - span);
        max[axis] = (min[axis] + span).min(box_size[axis]);
        if max[axis] <= min[axis] {
            min[axis] = 0.0;
            max[axis] = box_size[axis];
        }
    }
    (min, max)
}

fn inside_box_constraint(min: [f32; 3], max: [f32; 3]) -> ConstraintSpec {
    ConstraintSpec {
        mode: ConstraintMode::Inside,
        shape: ShapeSpec::Box { min, max },
    }
}

fn slab_region_for_instance(
    box_size: [f32; 3],
    axis: usize,
    instance_idx: usize,
    instance_total: usize,
    outer_margin_frac: f32,
) -> ([f32; 3], [f32; 3]) {
    let margin = [
        box_size[0] * outer_margin_frac,
        box_size[1] * outer_margin_frac,
        box_size[2] * outer_margin_frac,
    ];
    let mut min = margin;
    let mut max = [
        (box_size[0] - margin[0]).max(margin[0] + 2.0),
        (box_size[1] - margin[1]).max(margin[1] + 2.0),
        (box_size[2] - margin[2]).max(margin[2] + 2.0),
    ];
    let usable = (max[axis] - min[axis]).max(2.0);
    let slots = instance_total.max(1) as f32;
    let slab = usable / slots;
    min[axis] += slab * instance_idx as f32;
    max[axis] = if instance_idx + 1 >= instance_total.max(1) {
        max[axis]
    } else {
        min[axis] + slab
    };
    clamp_region_bounds(min, max, box_size)
}

fn central_grid_region_for_instance(
    box_size: [f32; 3],
    instance_idx: usize,
    instance_total: usize,
    outer_margin_frac: f32,
) -> ([f32; 3], [f32; 3]) {
    let margin = [
        box_size[0] * outer_margin_frac,
        box_size[1] * outer_margin_frac,
        box_size[2] * outer_margin_frac,
    ];
    let core_min = margin;
    let core_max = [
        (box_size[0] - margin[0]).max(margin[0] + 2.0),
        (box_size[1] - margin[1]).max(margin[1] + 2.0),
        (box_size[2] - margin[2]).max(margin[2] + 2.0),
    ];
    let cells_per_axis = ((instance_total.max(1) as f32).cbrt().ceil() as usize).max(1);
    let ix = instance_idx % cells_per_axis;
    let iy = (instance_idx / cells_per_axis) % cells_per_axis;
    let iz = (instance_idx / (cells_per_axis * cells_per_axis)) % cells_per_axis;
    let mut min = [0.0; 3];
    let mut max = [0.0; 3];
    for (axis, slot) in [ix, iy, iz].into_iter().enumerate() {
        let span = (core_max[axis] - core_min[axis]).max(2.0);
        let cell = span / cells_per_axis as f32;
        min[axis] = core_min[axis] + cell * slot as f32;
        max[axis] = if slot + 1 >= cells_per_axis {
            core_max[axis]
        } else {
            min[axis] + cell
        };
    }
    clamp_region_bounds(min, max, box_size)
}

fn outer_shell_region_for_instance(
    box_size: [f32; 3],
    axis: usize,
    instance_idx: usize,
    instance_total: usize,
    shell_frac: f32,
    outer_margin_frac: f32,
) -> ([f32; 3], [f32; 3]) {
    let margin = [
        box_size[0] * outer_margin_frac,
        box_size[1] * outer_margin_frac,
        box_size[2] * outer_margin_frac,
    ];
    let mut min = margin;
    let mut max = [
        (box_size[0] - margin[0]).max(margin[0] + 2.0),
        (box_size[1] - margin[1]).max(margin[1] + 2.0),
        (box_size[2] - margin[2]).max(margin[2] + 2.0),
    ];
    let shell = ((max[axis] - min[axis]).max(2.0) * shell_frac).max(2.0);
    if instance_idx % 2 == 1 {
        min[axis] = max[axis] - shell;
    } else {
        max[axis] = min[axis] + shell;
    }

    let spread_axis = (axis + 1) % 3;
    let spread_slots = instance_total.max(1).div_ceil(2).max(1);
    let spread_slot = (instance_idx / 2).min(spread_slots - 1);
    let usable = (max[spread_axis] - min[spread_axis]).max(2.0);
    let cell = usable / spread_slots as f32;
    min[spread_axis] += cell * spread_slot as f32;
    max[spread_axis] = if spread_slot + 1 >= spread_slots {
        max[spread_axis]
    } else {
        min[spread_axis] + cell
    };
    clamp_region_bounds(min, max, box_size)
}

fn graph_port_classes(graph: &TopologyGraph) -> Vec<String> {
    let mut classes = BTreeSet::new();
    for port in &graph.open_ports {
        if let Some(class) = port
            .port_class
            .as_ref()
            .filter(|value| !value.trim().is_empty())
        {
            classes.insert(class.clone());
        }
    }
    for policy in &graph.port_policies {
        if let Some(class) = policy
            .port_class
            .as_ref()
            .filter(|value| !value.trim().is_empty())
        {
            classes.insert(class.clone());
        }
    }
    classes.into_iter().collect()
}

fn stable_axis_from_labels(labels: &[String], fallback: usize) -> usize {
    if labels.is_empty() {
        return fallback;
    }
    let folded = labels
        .iter()
        .flat_map(|label| label.bytes())
        .fold(0usize, |acc, byte| {
            acc.wrapping_mul(33).wrapping_add(byte as usize)
        });
    folded % 3
}

fn graph_packing_helper(
    component: &ResolvedComponent,
    cohort: &GraphPackingCohort,
    morphology_mode: &str,
    box_size: [f32; 3],
    alignment_axis: Option<Vec3>,
    instance_idx: usize,
    instance_total: usize,
) -> Option<GraphPackingHelper> {
    let graph = component.topology_graph.as_ref()?;
    let architecture = graph_architecture(graph);
    let architecture_label = graph_architecture_label(graph);
    let open_port_count = graph.open_ports.len();
    let port_classes = graph_port_classes(graph);
    let alignment_vec = alignment_axis.unwrap_or_else(|| component_principal_axis(component));
    let dominant_axis = dominant_axis_index(alignment_vec);
    let port_class_axis = stable_axis_from_labels(&port_classes, dominant_axis);
    let mixed_architecture = cohort.mixed_architecture();
    let applied_cap_count = graph.applied_caps.len();
    let cap_state = if open_port_count > 0 {
        "open_ports_present"
    } else if applied_cap_count > 0 {
        "capped_ports_only"
    } else {
        "neutral"
    };
    let (region_min, region_max, region_policy, open_port_policy, alignment_source) =
        match morphology_mode {
            "amorphous_bulk" if open_port_count > 0 => match architecture {
                GraphArchitecture::Linear => {
                    let (min, max) = outer_shell_region_for_instance(
                        box_size,
                        port_class_axis,
                        instance_idx,
                        instance_total,
                        0.24,
                        0.06,
                    );
                    (
                        min,
                        max,
                        if port_classes.is_empty() {
                            "open_port_shell_faces"
                        } else {
                            "port_class_shell_faces"
                        },
                        if port_classes.is_empty() {
                            "free_end_distribution"
                        } else {
                            "port_class_edge_exposure"
                        },
                        graph_alignment_source(graph),
                    )
                }
                GraphArchitecture::Branched => {
                    let (min, max) = slab_region_for_instance(
                        box_size,
                        port_class_axis,
                        instance_idx,
                        instance_total,
                        0.12,
                    );
                    (
                        min,
                        max,
                        if port_classes.is_empty() {
                            "open_port_access_bands"
                        } else {
                            "port_class_access_bands"
                        },
                        if port_classes.is_empty() {
                            "free_port_preserved"
                        } else {
                            "port_class_access_preserved"
                        },
                        graph_alignment_source(graph),
                    )
                }
                GraphArchitecture::Cyclic => {
                    let (min, max) = slab_region_for_instance(
                        box_size,
                        port_class_axis,
                        instance_idx,
                        instance_total,
                        0.18,
                    );
                    (
                        min,
                        max,
                        if port_classes.is_empty() {
                            "open_port_ring_bands"
                        } else {
                            "port_class_ring_bands"
                        },
                        if port_classes.is_empty() {
                            "free_port_preserved"
                        } else {
                            "port_class_access_preserved"
                        },
                        graph_alignment_source(graph),
                    )
                }
                GraphArchitecture::BranchedCyclic => {
                    let (min, max) = slab_region_for_instance(
                        box_size,
                        port_class_axis,
                        instance_idx,
                        instance_total,
                        0.16,
                    );
                    (
                        min,
                        max,
                        if port_classes.is_empty() {
                            "open_port_branched_ring_bands"
                        } else {
                            "port_class_branched_ring_bands"
                        },
                        if port_classes.is_empty() {
                            "free_port_preserved"
                        } else {
                            "port_class_access_preserved"
                        },
                        graph_alignment_source(graph),
                    )
                }
            },
            "amorphous_bulk" if mixed_architecture => match architecture {
                GraphArchitecture::Linear => {
                    let (min, max) = outer_shell_region_for_instance(
                        box_size,
                        dominant_axis,
                        instance_idx,
                        instance_total,
                        0.20,
                        0.08,
                    );
                    (
                        min,
                        max,
                        "mixed_outer_shell_faces",
                        "none",
                        graph_alignment_source(graph),
                    )
                }
                GraphArchitecture::Branched => {
                    let (min, max) = central_grid_region_for_instance(
                        box_size,
                        instance_idx,
                        instance_total,
                        0.18,
                    );
                    (
                        min,
                        max,
                        "mixed_central_core_grid",
                        "none",
                        graph_alignment_source(graph),
                    )
                }
                GraphArchitecture::Cyclic => {
                    let (min, max) = central_grid_region_for_instance(
                        box_size,
                        instance_idx,
                        instance_total,
                        0.24,
                    );
                    (
                        min,
                        max,
                        "mixed_ring_core_grid",
                        "none",
                        graph_alignment_source(graph),
                    )
                }
                GraphArchitecture::BranchedCyclic => {
                    let (min, max) = central_grid_region_for_instance(
                        box_size,
                        instance_idx,
                        instance_total,
                        0.22,
                    );
                    (
                        min,
                        max,
                        "mixed_branched_ring_core_grid",
                        "none",
                        graph_alignment_source(graph),
                    )
                }
            },
            "amorphous_bulk" => match architecture {
                GraphArchitecture::Linear => {
                    let (min, max) = slab_region_for_instance(
                        box_size,
                        dominant_axis,
                        instance_idx,
                        instance_total,
                        0.08,
                    );
                    (
                        min,
                        max,
                        "dominant_axis_slabs",
                        "none",
                        graph_alignment_source(graph),
                    )
                }
                GraphArchitecture::Branched => {
                    let (min, max) = central_grid_region_for_instance(
                        box_size,
                        instance_idx,
                        instance_total,
                        0.16,
                    );
                    (
                        min,
                        max,
                        "central_core_grid",
                        "none",
                        graph_alignment_source(graph),
                    )
                }
                GraphArchitecture::Cyclic => {
                    let (min, max) = central_grid_region_for_instance(
                        box_size,
                        instance_idx,
                        instance_total,
                        0.22,
                    );
                    (
                        min,
                        max,
                        "ring_core_grid",
                        "none",
                        graph_alignment_source(graph),
                    )
                }
                GraphArchitecture::BranchedCyclic => {
                    let (min, max) = central_grid_region_for_instance(
                        box_size,
                        instance_idx,
                        instance_total,
                        0.20,
                    );
                    (
                        min,
                        max,
                        "branched_ring_core_grid",
                        "none",
                        graph_alignment_source(graph),
                    )
                }
            },
            "backbone_aligned_bulk" => {
                let axis = dominant_axis_index(alignment_axis.unwrap_or(Vec3::new(0.0, 0.0, 1.0)));
                let (min, max) = if open_port_count > 0 {
                    outer_shell_region_for_instance(
                        box_size,
                        axis,
                        instance_idx,
                        instance_total,
                        0.20,
                        0.06,
                    )
                } else {
                    slab_region_for_instance(box_size, axis, instance_idx, instance_total, 0.06)
                };
                (
                    min,
                    max,
                    if open_port_count > 0 {
                        "aligned_outer_shell_faces"
                    } else {
                        "alignment_axis_slabs"
                    },
                    if open_port_count > 0 {
                        if port_classes.is_empty() {
                            "alignment_path_exposed"
                        } else {
                            "alignment_path.port_class_exposed"
                        }
                    } else {
                        "none"
                    },
                    graph_alignment_source(graph),
                )
            }
            _ => return None,
        };
    Some(GraphPackingHelper {
        constraints: vec![inside_box_constraint(region_min, region_max)],
        policy: json!({
            "component": component.name,
            "instance_index": instance_idx + 1,
            "instance_total": instance_total,
            "morphology_mode": morphology_mode,
            "architecture": architecture_label,
            "cohort_policy": if mixed_architecture {
                "mixed_architecture_partitioned"
            } else {
                "single_architecture"
            },
            "cohort_summary": {
                "typed_component_count": cohort.typed_component_count,
                "typed_instance_count": cohort.typed_instance_count,
                "distinct_architecture_count": cohort.distinct_architecture_count(),
                "linear_instances": cohort.linear_count,
                "branched_instances": cohort.branched_count,
                "cyclic_instances": cohort.cyclic_count,
                "branched_cyclic_instances": cohort.branched_cyclic_count,
            },
            "alignment_source": alignment_source,
            "region_policy": region_policy,
            "open_port_policy": open_port_policy,
            "port_class_policy": if port_classes.is_empty() {
                "none"
            } else {
                "class_axis_partitioned"
            },
            "port_classes": port_classes,
            "cap_state": cap_state,
            "applied_cap_count": applied_cap_count,
            "constraint_shape": "inside_box",
            "region_box_angstrom": {"min": region_min, "max": region_max},
            "graph_summary": {
                "build_mode": graph.build_plan.target_mode,
                "cycle_count": graph.cycle_basis.len(),
                "branch_point_count": graph.branch_points.len(),
                "open_port_count": open_port_count,
                "motif_count": graph.motif_instances.len(),
            }
        }),
    })
}

fn quaternion_from_unit_vectors(from: Vec3, to: Vec3) -> Quaternion {
    let from = normalize_vec3(from);
    let to = normalize_vec3(to);
    let dot = from.dot(to).clamp(-1.0, 1.0);
    if dot > 1.0 - 1.0e-6 {
        return Quaternion::identity();
    }
    if dot < -1.0 + 1.0e-6 {
        let axis = if from.x.abs() < 0.9 {
            normalize_vec3(from.cross(Vec3::new(1.0, 0.0, 0.0)))
        } else {
            normalize_vec3(from.cross(Vec3::new(0.0, 1.0, 0.0)))
        };
        return Quaternion {
            x: axis.x,
            y: axis.y,
            z: axis.z,
            w: 0.0,
        };
    }

    let cross = from.cross(to);
    let s = (2.0 * (1.0 + dot)).sqrt();
    let inv = 1.0 / s;
    let quat = Quaternion {
        x: cross.x * inv,
        y: cross.y * inv,
        z: cross.z * inv,
        w: 0.5 * s,
    };
    let norm = (quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w).sqrt();
    Quaternion {
        x: quat.x / norm,
        y: quat.y / norm,
        z: quat.z / norm,
        w: quat.w / norm,
    }
}

fn quaternion_mul(lhs: Quaternion, rhs: Quaternion) -> Quaternion {
    let quat = Quaternion {
        w: lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z,
        x: lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
        y: lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
        z: lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
    };
    let norm = (quat.x * quat.x + quat.y * quat.y + quat.z * quat.z + quat.w * quat.w).sqrt();
    Quaternion {
        x: quat.x / norm,
        y: quat.y / norm,
        z: quat.z / norm,
        w: quat.w / norm,
    }
}

fn quaternion_from_axis_angle(axis: Vec3, theta: f32) -> Quaternion {
    let axis = normalize_vec3(axis);
    let half = theta * 0.5;
    let sin_half = half.sin();
    Quaternion {
        x: axis.x * sin_half,
        y: axis.y * sin_half,
        z: axis.z * sin_half,
        w: half.cos(),
    }
}

fn aligned_packmol_euler(component: &ResolvedComponent, axis: Vec3) -> [f32; 3] {
    let rotation = quaternion_from_unit_vectors(component_principal_axis(component), axis);
    let (beta, gamma, teta) = rotation.to_packmol_euler();
    [beta, gamma, teta]
}

fn aligned_packmol_euler_with_azimuth(
    component: &ResolvedComponent,
    axis: Vec3,
    azimuth: f32,
) -> [f32; 3] {
    let base = quaternion_from_unit_vectors(component_principal_axis(component), axis);
    let twist = quaternion_from_axis_angle(axis, azimuth);
    let rotation = quaternion_mul(twist, base);
    let (beta, gamma, teta) = rotation.to_packmol_euler();
    [beta, gamma, teta]
}

fn box_from_positions(
    positions: &[[f32; 3]],
    box_req: &BoxRequest,
) -> PackResult<([f32; 3], Value)> {
    if positions.is_empty() {
        return Err(PackError::Invalid("solute has no atoms".into()));
    }

    let (min, max) = bounds_from_positions(positions)?;
    let extents = [
        (max[0] - min[0]).max(1.0e-3),
        (max[1] - min[1]).max(1.0e-3),
        (max[2] - min[2]).max(1.0e-3),
    ];

    let mut resolved =
        match box_req.mode.as_str() {
            "padding" => {
                let padding = box_req.padding_angstrom.unwrap_or(0.0);
                [
                    extents[0] + 2.0 * padding,
                    extents[1] + 2.0 * padding,
                    extents[2] + 2.0 * padding,
                ]
            }
            "fixed_size" => value_to_triplet(box_req.size_angstrom.as_ref().ok_or_else(|| {
                PackError::Invalid("fixed_size mode requires size_angstrom".into())
            })?)?,
            other => {
                return Err(PackError::Invalid(format!(
                    "unsupported box mode '{other}'"
                )))
            }
        };

    if box_req.shape == "cubic" {
        let side = resolved[0].max(resolved[1]).max(resolved[2]);
        resolved = [side, side, side];
    }

    Ok((
        resolved,
        json!({
            "solute_bounds_min": min,
            "solute_bounds_max": max,
            "solute_extent_angstrom": extents,
            "box_policy": box_req.mode,
            "box_shape": box_req.shape,
            "resolved_box_size_angstrom": resolved,
        }),
    ))
}

fn estimate_solvent_count(box_size: [f32; 3], occupied_volume: f64) -> usize {
    let box_volume = box_size[0] as f64 * box_size[1] as f64 * box_size[2] as f64;
    let free_volume = (box_volume - occupied_volume).max(0.0);
    let estimate =
        WATER_MOLARITY * AVOGADRO * ANGSTROM3_TO_LITER * free_volume * DEFAULT_PACKING_FRACTION;
    estimate.round().max(0.0) as usize
}

fn total_ion_mass_amu_with(
    ions: &IonRegistry,
    ion_counts: &BTreeMap<String, usize>,
) -> PackResult<f64> {
    ion_counts.iter().try_fold(0.0, |total, (species, count)| {
        let info = ion_species_info_with(ions, species).ok_or_else(|| {
            PackError::Invalid(format!("unsupported ion species '{species}'"))
        })?;
        Ok(total + info.mass_amu * *count as f64)
    })
}

fn total_ion_mass_amu(ion_counts: &BTreeMap<String, usize>) -> PackResult<f64> {
    total_ion_mass_amu_with(ion_registry()?, ion_counts)
}

fn estimate_solvent_count_for_density(
    box_size: [f32; 3],
    target_density_g_cm3: f32,
    dry_mass_amu: f64,
    ions: &IonRegistry,
    ion_counts: &BTreeMap<String, usize>,
) -> PackResult<usize> {
    let target_mass_amu = target_density_g_cm3 as f64
        * box_size[0] as f64
        * box_size[1] as f64
        * box_size[2] as f64
        * ANGSTROM3_TO_CM3
        / AMU_TO_GRAM;
    let ion_mass_amu = total_ion_mass_amu_with(ions, ion_counts)?;
    let solvent_mass_amu = (target_mass_amu - dry_mass_amu - ion_mass_amu).max(0.0);
    Ok((solvent_mass_amu / WATER_MASS_AMU).round().max(0.0) as usize)
}

fn estimate_salt_formula_units(box_size: [f32; 3], salt_molar: Option<f32>) -> usize {
    let Some(molar) = salt_molar else {
        return 0;
    };
    let volume_l =
        box_size[0] as f64 * box_size[1] as f64 * box_size[2] as f64 * ANGSTROM3_TO_LITER;
    (molar as f64 * volume_l * AVOGADRO).round().max(0.0) as usize
}

fn emit_event(event: &RunEvent, enabled: bool) {
    if enabled {
        eprintln!(
            "{}",
            serde_json::to_string(event).unwrap_or_else(|_| "{}".to_string())
        );
    }
}

fn elapsed_ms(start: &Instant) -> u64 {
    start.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

fn eta_ms(start: &Instant, progress_pct: f32) -> Option<u64> {
    if !(0.0..100.0).contains(&progress_pct) {
        return Some(0).filter(|_| progress_pct >= 100.0);
    }
    let elapsed = elapsed_ms(start) as f64;
    Some((elapsed * ((100.0 - progress_pct as f64) / progress_pct as f64)).round() as u64)
}

fn component_count(metadata: &TranslationMetadata) -> usize {
    metadata.component_inventory.len()
        + usize::from(metadata.water_count > 0)
        + metadata
            .ion_counts
            .values()
            .filter(|count| **count > 0)
            .count()
}

fn is_prmtop_path(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.eq_ignore_ascii_case("prmtop"))
        .unwrap_or(false)
}

fn resolve_artifact_component(
    name: String,
    count: usize,
    artifact: &ArtifactRef,
) -> PackResult<ResolvedComponent> {
    let (_mol, context) = load_solute_context(&artifact.path, artifact.topology.as_deref())?;
    let mut charge_source_kinds = Vec::new();
    let mut per_instance_net_charge = None;
    let mut charge_source = None;
    if let Some(path) = artifact.charge_manifest.as_deref() {
        let manifest = load_charge_manifest(Path::new(path))?;
        charge_source_kinds = charge_manifest_field_kinds(&manifest);
        let estimate = compute_solute_net_charge(&manifest);
        per_instance_net_charge = estimate.net_charge_e;
        charge_source = estimate.source;
    } else if let Some(path) = artifact
        .topology
        .as_deref()
        .filter(|path| is_prmtop_path(path))
    {
        let estimate = compute_solute_net_charge_from_prmtop(Path::new(path))?;
        charge_source_kinds.push("prmtop.total_charge".into());
        per_instance_net_charge = estimate.net_charge_e;
        charge_source = estimate.source;
    }

    Ok(ResolvedComponent {
        name,
        count,
        source_kind: "artifact".into(),
        kind: artifact.kind.clone().unwrap_or_else(|| "assembly".into()),
        coordinates_path: artifact.path.clone(),
        topology: artifact.topology.clone(),
        topology_graph_path: None,
        topology_graph: None,
        charge_manifest_path: artifact.charge_manifest.clone(),
        build_manifest_path: None,
        forcefield_ref: None,
        connectivity_hint: artifact.connectivity_hint.clone(),
        parameter_source: artifact.parameter_source.clone(),
        source_detail: serde_json::to_value(artifact)
            .map_err(|err| PackError::Parse(err.to_string()))?,
        built_artifact: None,
        polymer_build_handoff: None,
        polymer_artifact: None,
        polymer_controls: None,
        context,
        per_instance_net_charge,
        charge_source,
        charge_source_kinds,
        fixed_rotation_euler: None,
    })
}

fn resolve_polymer_build_component(
    name: String,
    count: usize,
    handoff: &PolymerBuildArtifactRef,
) -> PackResult<ResolvedComponent> {
    let loaded =
        load_polymer_build_handoff(handoff).map_err(|err| PackError::Invalid(err.message))?;
    let build_target = loaded
        .manifest
        .pointer("/normalized_request/target")
        .cloned()
        .unwrap_or(Value::Null);
    let build_realization = loaded
        .manifest
        .get("realization")
        .cloned()
        .or_else(|| {
            loaded
                .manifest
                .pointer("/normalized_request/realization")
                .cloned()
        })
        .unwrap_or(Value::Null);
    let build_summary = loaded
        .manifest
        .get("summary")
        .cloned()
        .unwrap_or(Value::Null);
    let source_bundle = loaded
        .manifest
        .get("source_bundle")
        .cloned()
        .unwrap_or(Value::Null);
    let (mol, context) = load_solute_context(&loaded.coordinates_path, loaded.topology.as_deref())?;
    let source_structure_path = loaded
        .manifest
        .pointer("/resolved_inputs/resolved_source_artifacts/coordinates")
        .cloned()
        .unwrap_or(Value::Null);
    let source_topology_path = loaded
        .manifest
        .pointer("/resolved_inputs/resolved_source_artifacts/topology")
        .cloned()
        .unwrap_or(Value::Null);
    let source_sequence_tokens = build_summary
        .get("resolved_sequence")
        .cloned()
        .unwrap_or(Value::Null);
    let template_sequence_resnames = build_summary
        .get("template_sequence_resnames")
        .cloned()
        .unwrap_or(Value::Null);
    let applied_residue_resnames = build_summary
        .get("applied_residue_resnames")
        .cloned()
        .unwrap_or(Value::Null);
    let chain_instance_mapping = (0..count)
        .map(|copy_idx| {
            let packed_start = copy_idx * context.chain_count + 1;
            let packed_end = packed_start + context.chain_count.saturating_sub(1);
            json!({
                "copy_index": copy_idx + 1,
                "source_chain_indices": (1..=context.chain_count).collect::<Vec<_>>(),
                "packed_chain_indices": (packed_start..=packed_end).collect::<Vec<_>>(),
            })
        })
        .collect::<Vec<_>>();
    if let Some(graph) = loaded.topology_graph.as_ref() {
        let residue_count = mol
            .atoms
            .iter()
            .map(|atom| (atom.chain, atom.resid, atom.resname.clone()))
            .collect::<BTreeSet<_>>()
            .len();
        if graph.atoms.len() != mol.atoms.len() {
            return Err(PackError::Invalid(
                "topology_graph atom count does not match coordinates".into(),
            ));
        }
        if graph.residues.len() != residue_count {
            return Err(PackError::Invalid(
                "topology_graph residue count does not match coordinates".into(),
            ));
        }
    }

    let mut charge_source_kinds = Vec::new();
    let mut per_instance_net_charge = None;
    let mut charge_source = None;
    if let Some(path) = loaded.charge_manifest_path.as_deref() {
        let manifest = load_charge_manifest(Path::new(path))?;
        charge_source_kinds = charge_manifest_field_kinds(&manifest);
        let estimate = compute_solute_net_charge(&manifest);
        per_instance_net_charge = estimate.net_charge_e;
        charge_source = estimate.source;
    } else if let Some(path) = loaded
        .topology
        .as_deref()
        .filter(|path| is_prmtop_path(path))
    {
        let estimate = compute_solute_net_charge_from_prmtop(Path::new(path))?;
        charge_source_kinds.push("prmtop.total_charge".into());
        per_instance_net_charge = estimate.net_charge_e;
        charge_source = estimate.source;
    }

    Ok(ResolvedComponent {
        name,
        count,
        source_kind: "polymer_build".into(),
        kind: "polymer_chain".into(),
        coordinates_path: loaded.coordinates_path.clone(),
        topology: loaded.topology.clone(),
        topology_graph_path: loaded.topology_graph_path.clone(),
        topology_graph: loaded.topology_graph.clone(),
        charge_manifest_path: loaded.charge_manifest_path.clone(),
        build_manifest_path: Some(loaded.manifest_path.clone()),
        forcefield_ref: loaded.forcefield_ref.clone(),
        connectivity_hint: Some("polymer_build_handoff".into()),
        parameter_source: Some("warp-build.agent.v1".into()),
        source_detail: json!({
            "kind": "polymer_chain",
            "path": loaded.coordinates_path,
            "topology": loaded.topology,
            "topology_graph": loaded.topology_graph_path,
            "charge_manifest": loaded.charge_manifest_path,
            "build_manifest": loaded.manifest_path,
            "connectivity_hint": "polymer_build_handoff",
            "parameter_source": "warp-build.agent.v1",
            "forcefield_ref": loaded.forcefield_ref,
            "source_structure_path": source_structure_path,
            "source_topology_path": source_topology_path,
            "source_sequence_tokens": source_sequence_tokens,
            "template_sequence_resnames": template_sequence_resnames,
            "applied_residue_resnames": applied_residue_resnames,
        }),
        built_artifact: Some(json!({
            "path": loaded.coordinates_path.clone(),
            "build_manifest": loaded.manifest_path.clone(),
            "build_mode": build_summary.get("build_mode").cloned().unwrap_or(Value::Null),
            "target_n_repeat": build_summary.get("total_repeat_units").cloned().unwrap_or(Value::Null),
            "conformation_mode": build_realization.get("conformation_mode").cloned().unwrap_or(Value::Null),
            "seed": build_realization.get("seed").cloned().unwrap_or(Value::Null),
            "source_bundle": source_bundle.clone(),
            "forcefield_ref": loaded.forcefield_ref.clone(),
            "source_sequence_tokens": source_sequence_tokens.clone(),
            "template_sequence_resnames": template_sequence_resnames.clone(),
            "applied_residue_resnames": applied_residue_resnames.clone(),
        })),
        polymer_build_handoff: Some(json!({
            "build_manifest": loaded.manifest_path.clone(),
            "manifest_version": loaded
                .manifest
                .get("schema_version")
                .cloned()
                .or_else(|| loaded.manifest.get("version").cloned())
                .unwrap_or(Value::Null),
            "coordinates": loaded.coordinates_path.clone(),
            "charge_manifest": loaded.charge_manifest_path.clone(),
            "topology": loaded.topology.clone(),
            "topology_graph": loaded.topology_graph_path.clone(),
            "forcefield_ref": loaded.forcefield_ref.clone(),
            "source_bundle": source_bundle.clone(),
            "summary": build_summary.clone(),
            "source_structure_path": source_structure_path,
            "source_topology_path": source_topology_path,
            "source_sequence_tokens": source_sequence_tokens,
            "template_sequence_resnames": template_sequence_resnames,
            "applied_residue_resnames": applied_residue_resnames,
            "copy_count": count,
            "source_chain_count_per_copy": context.chain_count,
            "chain_instance_mapping": chain_instance_mapping,
        })),
        polymer_artifact: Some(json!({
            "build_manifest": loaded.manifest_path.clone(),
            "source_ref": loaded.manifest.pointer("/normalized_request/source_ref").cloned().unwrap_or(Value::Null),
            "target": build_target.clone(),
            "realization": build_realization.clone(),
        })),
        polymer_controls: Some(json!({
            "handoff_source": WARP_BUILD_MANIFEST_VERSION,
            "target": build_target,
            "realization": build_realization,
        })),
        context,
        per_instance_net_charge,
        charge_source,
        charge_source_kinds,
        fixed_rotation_euler: None,
    })
}

fn resolve_components(req: &BuildRequest) -> PackResult<Vec<ResolvedComponent>> {
    if let Some(components) = &req.components {
        let mut resolved = Vec::with_capacity(components.len());
        for component in components {
            resolved.push(match &component.source {
                ComponentSource::Artifact { artifact } => {
                    resolve_artifact_component(component.name.clone(), component.count, artifact)?
                }
                ComponentSource::PolymerBuild { polymer_build } => resolve_polymer_build_component(
                    component.name.clone(),
                    component.count,
                    polymer_build,
                )?,
            });
        }
        return Ok(resolved);
    }
    if let Some(solute) = &req.solute {
        return Ok(vec![resolve_artifact_component(
            "solute".into(),
            1,
            solute,
        )?]);
    }
    if let Some(polymer_build) = &req.polymer_build {
        return Ok(vec![resolve_polymer_build_component(
            "polymer".into(),
            1,
            polymer_build,
        )?]);
    }
    Err(PackError::Invalid(
        "provide one of components, solute, or polymer_build".into(),
    ))
}

fn build_pack_config(
    req: &BuildRequest,
    chemistry: &ChemistryCatalog,
) -> PackResult<(PackConfig, TranslationMetadata)> {
    let mut components = resolve_components(req)?;

    let alignment_axis_vec = if req.environment.morphology.mode == "backbone_aligned_bulk" {
        Some(
            parse_alignment_axis(
                req.environment
                    .morphology
                    .alignment_axis
                    .as_deref()
                    .unwrap_or("z"),
            )
            .map_err(|err| PackError::Invalid(err.message))?,
        )
    } else {
        None
    };
    if let Some(axis) = alignment_axis_vec {
        for component in &mut components {
            component.fixed_rotation_euler = Some(aligned_packmol_euler(component, axis));
        }
    }

    let aggregated_context = merge_solute_contexts(&components)?;
    let explicit_box_size = req
        .environment
        .box_spec
        .size_angstrom
        .as_ref()
        .map(value_to_triplet)
        .transpose()?;
    let density_sets_solvent_fill = req.environment.solvent.mode == "explicit"
        && req.environment.morphology.target_density_g_cm3.is_some()
        && explicit_box_size.is_some();
    let (box_size, box_decisions) = if req.environment.morphology.mode == "single_chain_solution" {
        box_from_positions(&components[0].context.positions, &req.environment.box_spec)?
    } else if density_sets_solvent_fill {
        let mut resolved = explicit_box_size.expect("checked explicit box size");
        if req.environment.box_spec.shape == "cubic" {
            let side = resolved[0].max(resolved[1]).max(resolved[2]);
            resolved = [side, side, side];
        }
        (
            resolved,
            json!({
                "box_policy": req.environment.box_spec.mode,
                "box_shape": req.environment.box_spec.shape,
                "resolved_box_size_angstrom": resolved,
                "target_density_g_cm3": req.environment.morphology.target_density_g_cm3,
                "density_basis": "fixed_box_with_explicit_solvent_fill",
            }),
        )
    } else if let Some(target_density) = req.environment.morphology.target_density_g_cm3 {
        let resolved = density_target_box_size(
            &req.environment.box_spec,
            &aggregated_context,
            target_density,
        )?;
        (
            resolved,
            json!({
                "box_policy": "target_density_g_cm3",
                "box_shape": req.environment.box_spec.shape,
                "resolved_box_size_angstrom": resolved,
                "target_density_g_cm3": target_density,
                "density_basis": "components_only",
            }),
        )
    } else {
        let mut resolved =
            value_to_triplet(req.environment.box_spec.size_angstrom.as_ref().ok_or_else(
                || PackError::Invalid("fixed_size mode requires size_angstrom".into()),
            )?)?;
        if req.environment.box_spec.shape == "cubic" {
            let side = resolved[0].max(resolved[1]).max(resolved[2]);
            resolved = [side, side, side];
        }
        (
            resolved,
            json!({
                "box_policy": req.environment.box_spec.mode,
                "box_shape": req.environment.box_spec.shape,
                "resolved_box_size_angstrom": resolved,
            }),
        )
    };

    let box_center = [box_size[0] * 0.5, box_size[1] * 0.5, box_size[2] * 0.5];
    let alignment_axis = alignment_axis_vec.map(Vec3::to_array);
    let graph_cohort = graph_packing_cohort(&components);
    let mut component_packing_hints = Vec::new();

    let mut structures = Vec::new();
    for (idx, component) in components.iter().enumerate() {
        let base_structure = StructureSpec {
            path: component.coordinates_path.clone(),
            count: component.count,
            name: Some(component.name.clone()),
            topology: component.topology.clone(),
            restart_from: None,
            restart_to: None,
            fixed_eulers: None,
            chain: None,
            changechains: false,
            segid: None,
            connect: true,
            format: None,
            rotate: true,
            fixed: false,
            positions: None,
            translate: None,
            center: true,
            min_distance: None,
            resnumbers: None,
            maxmove: None,
            nloop: None,
            nloop0: None,
            constraints: Vec::new(),
            radius: None,
            fscale: None,
            short_radius: None,
            short_radius_scale: None,
            atom_overrides: Vec::new(),
            atom_constraints: Vec::new(),
            rot_bounds: None,
        };
        match req.environment.morphology.mode.as_str() {
            "single_chain_solution" => {
                let mut structure = base_structure.clone();
                structure.count = 1;
                structure.name = Some(component.name.clone());
                structure.rotate = false;
                structure.fixed = true;
                structure.positions = Some(vec![box_center]);
                if idx == 0 {
                    structure.name = Some("solute".into());
                }
                structures.push(structure);
            }
            "amorphous_bulk" => {
                if component.topology_graph.is_some() {
                    for instance_idx in 0..component.count {
                        let mut structure = base_structure.clone();
                        structure.count = 1;
                        structure.name = Some(format!("{}_{}", component.name, instance_idx + 1));
                        if let Some(helper) = graph_packing_helper(
                            component,
                            &graph_cohort,
                            "amorphous_bulk",
                            box_size,
                            None,
                            instance_idx,
                            component.count,
                        ) {
                            structure.constraints = helper.constraints;
                            component_packing_hints.push(helper.policy);
                        }
                        structures.push(structure);
                    }
                } else {
                    structures.push(base_structure);
                }
            }
            "backbone_aligned_bulk" => {
                let axis = alignment_axis_vec
                    .ok_or_else(|| PackError::Invalid("missing alignment axis".into()))?;
                for instance_idx in 0..component.count {
                    let mut structure = base_structure.clone();
                    structure.count = 1;
                    structure.name = Some(format!("{}_{}", component.name, instance_idx + 1));
                    let azimuth = 2.0
                        * std::f32::consts::PI
                        * (instance_idx as f32 / component.count.max(1) as f32);
                    let euler = aligned_packmol_euler_with_azimuth(component, axis, azimuth);
                    structure.rot_bounds = Some([
                        [euler[0], euler[0]],
                        [euler[1], euler[1]],
                        [euler[2], euler[2]],
                    ]);
                    if let Some(helper) = graph_packing_helper(
                        component,
                        &graph_cohort,
                        "backbone_aligned_bulk",
                        box_size,
                        Some(axis),
                        instance_idx,
                        component.count,
                    ) {
                        structure.constraints = helper.constraints;
                        component_packing_hints.push(helper.policy);
                    }
                    structures.push(structure);
                }
            }
            other => {
                return Err(PackError::Invalid(format!(
                    "unsupported morphology mode '{other}'"
                )))
            }
        }
    }

    let occupied_volume = components.iter().try_fold(0.0f64, |total, component| {
        Ok::<_, PackError>(
            total + bounding_box_volume(&component.context.positions)? * component.count as f64,
        )
    })?;

    let mut charge_manifest_paths = Vec::new();
    let mut charge_source_kind_set = BTreeSet::new();
    let mut component_charge_resolution = Vec::new();
    let mut net_charge_before_neutralization = None;
    let mut neutralization_policy = "not_requested".to_string();
    for component in &components {
        if let Some(path) = &component.charge_manifest_path {
            charge_manifest_paths.push(path.clone());
        }
        for kind in &component.charge_source_kinds {
            charge_source_kind_set.insert(kind.clone());
        }
        component_charge_resolution.push(json!({
            "name": component.name,
            "count": component.count,
            "source_kind": component.source_kind,
            "charge_manifest": component.charge_manifest_path,
            "topology": component.topology,
            "per_instance_net_charge_e": component.per_instance_net_charge,
            "total_component_charge_e": component.per_instance_net_charge.map(|value| value * component.count as f32),
            "charge_source": component.charge_source,
            "charge_source_kinds": component.charge_source_kinds,
        }));
    }
    if req.environment.ions.neutralize.enabled() {
        let mut total_charge = 0.0f32;
        for component in &components {
            let net_charge = component.per_instance_net_charge.ok_or_else(|| {
                PackError::Invalid(format!(
                    "neutralize requires charge data for component '{}'",
                    component.name
                ))
            })?;
            total_charge += net_charge * component.count as f32;
        }
        net_charge_before_neutralization = Some(total_charge);
        neutralization_policy =
            if components.len() == 1 && total_component_instances(&components) == 1 {
                components[0]
                    .charge_source
                    .clone()
                    .unwrap_or_else(|| "component_sum".to_string())
            } else {
                "component_sum".to_string()
            };
    }

    let normalized_salt = req.environment.ions.normalized_salt_with(chemistry)?;
    let salt_stoich = normalized_salt
        .as_ref()
        .map(|salt| salt.species.clone())
        .unwrap_or_default();
    let salt_pair_count = estimate_salt_formula_units(
        box_size,
        normalized_salt.as_ref().and_then(|salt| salt.molar),
    );
    let mut ion_counts = BTreeMap::new();
    for (species, per_formula_unit) in &salt_stoich {
        ion_counts.insert(species.clone(), per_formula_unit * salt_pair_count);
    }

    if req.environment.ions.neutralize.enabled() {
        if let Some(net_charge) = net_charge_before_neutralization {
            if net_charge < 0.0 {
                let counterion = req
                    .environment
                    .ions
                    .counterion_for_charge_with(chemistry, true)?;
                let cation_valence = ion_species_info_with(&chemistry.ions, &counterion)
                    .map(|info| info.charge_e.unsigned_abs() as f32)
                    .unwrap_or(1.0);
                let neutralizers = (net_charge.abs() / cation_valence).ceil() as usize;
                *ion_counts
                    .entry(counterion)
                    .or_insert(0) += neutralizers;
            } else if net_charge > 0.0 {
                let counterion = req
                    .environment
                    .ions
                    .counterion_for_charge_with(chemistry, false)?;
                let anion_valence = ion_species_info_with(&chemistry.ions, &counterion)
                    .map(|info| info.charge_e.unsigned_abs() as f32)
                    .unwrap_or(1.0);
                let neutralizers = (net_charge.abs() / anion_valence).ceil() as usize;
                *ion_counts
                    .entry(counterion)
                    .or_insert(0) += neutralizers;
            }
        }
    }

    let mut water_count = 0usize;
    if req.environment.solvent.mode == "explicit" {
        let model = req.environment.solvent.model.as_deref().ok_or_else(|| {
            PackError::Invalid("explicit solvent mode requires solvent.model".into())
        })?;
        water_count = if let Some(target_density) = req.environment.morphology.target_density_g_cm3
        {
            estimate_solvent_count_for_density(
                box_size,
                target_density,
                aggregated_context.total_mass_amu,
                &chemistry.ions,
                &ion_counts,
            )?
        } else {
            estimate_solvent_count(box_size, occupied_volume)
        };
        if water_count > 0 {
            structures.push(StructureSpec {
                path: water_template_path(model),
                count: water_count,
                name: Some("solvent".into()),
                topology: None,
                restart_from: None,
                restart_to: None,
                fixed_eulers: None,
                chain: None,
                changechains: false,
                segid: None,
                connect: true,
                format: None,
                rotate: true,
                fixed: false,
                positions: None,
                translate: None,
                center: true,
                min_distance: None,
                resnumbers: None,
                maxmove: None,
                nloop: None,
                nloop0: None,
                constraints: Vec::new(),
                radius: None,
                fscale: None,
                short_radius: None,
                short_radius_scale: None,
                atom_overrides: Vec::new(),
                atom_constraints: Vec::new(),
                rot_bounds: None,
            });
        }
    }

    for (species, count) in &ion_counts {
        if *count == 0 {
            continue;
        }
        structures.push(StructureSpec {
            path: ion_template_path_with(&chemistry.ions, species),
            count: *count,
            name: Some(species.clone()),
            topology: None,
            restart_from: None,
            restart_to: None,
            fixed_eulers: None,
            chain: None,
            changechains: false,
            segid: None,
            connect: true,
            format: None,
            rotate: true,
            fixed: false,
            positions: None,
            translate: None,
            center: true,
            min_distance: None,
            resnumbers: None,
            maxmove: None,
            nloop: None,
            nloop0: None,
            constraints: Vec::new(),
            radius: None,
            fscale: None,
            short_radius: None,
            short_radius_scale: None,
            atom_overrides: Vec::new(),
            atom_constraints: Vec::new(),
            rot_bounds: None,
        });
    }

    let cfg = PackConfig {
        box_: BoxSpec {
            size: box_size,
            shape: "orthorhombic".into(),
        },
        structures,
        seed: Some(1_234_567),
        max_attempts: None,
        min_distance: None,
        filetype: None,
        add_box_sides: true,
        add_box_sides_fix: None,
        add_amber_ter: false,
        amber_ter_preserve: false,
        hexadecimal_indices: false,
        ignore_conect: false,
        non_standard_conect: true,
        pbc: true,
        pbc_min: None,
        pbc_max: None,
        maxit: None,
        nloop: None,
        nloop0: None,
        avoid_overlap: true,
        packall: false,
        check: false,
        sidemax: None,
        discale: None,
        precision: None,
        chkgrad: false,
        iprint1: None,
        iprint2: None,
        gencan_maxit: None,
        gencan_step: None,
        use_short_tol: false,
        short_tol_dist: None,
        short_tol_scale: None,
        movefrac: None,
        movebadrandom: false,
        disable_movebad: false,
        maxmove: None,
        randominitialpoint: false,
        fbins: None,
        writeout: None,
        writebad: false,
        restart_from: None,
        restart_to: None,
        relax_steps: None,
        relax_step: None,
        write_crd: None,
        output: Some(OutputSpec {
            path: req.outputs.coordinates.clone(),
            format: resolved_output_format(&req.outputs),
            scale: Some(1.0),
        }),
    };

    let primary = components
        .first()
        .ok_or_else(|| PackError::Invalid("no components resolved".into()))?;
    let component_inventory = components
        .iter()
        .map(|component| {
            json!({
                "name": component.name,
                "count": component.count,
                "source_kind": component.source_kind,
                "kind": component.kind,
                "path": component.coordinates_path,
                "topology": component.topology,
                "topology_graph": component.topology_graph_path,
                "topology_graph_summary": component.topology_graph.as_ref().map(|graph| json!({
                    "schema_version": graph.schema_version,
                    "build_mode": graph.build_plan.target_mode,
                    "architecture": graph_architecture_label(graph),
                    "motif_count": graph.motif_instances.len(),
                    "cycle_count": graph.cycle_basis.len(),
                    "branch_point_count": graph.branch_points.len(),
                    "open_port_count": graph.open_ports.len(),
                    "applied_cap_count": graph.applied_caps.len(),
                    "port_class_count": graph_port_classes(graph).len(),
                })),
                "charge_manifest": component.charge_manifest_path,
                "build_manifest": component.build_manifest_path,
                "forcefield_ref": component.forcefield_ref,
                "connectivity_hint": component.connectivity_hint,
                "parameter_source": component.parameter_source,
                "aligned_euler": component.fixed_rotation_euler,
                "polymer_md_handoff": component.polymer_build_handoff,
            })
        })
        .collect::<Vec<_>>();
    let charge_source_kinds = charge_source_kind_set.into_iter().collect::<Vec<_>>();
    let charge_manifest_path = charge_manifest_paths.first().cloned();
    let warnings = collect_warnings(req, &components);

    Ok((
        cfg,
        TranslationMetadata {
            solute_context: aggregated_context,
            resolved_box_size: box_size,
            water_count,
            ion_counts,
            salt_pair_count,
            component_inventory: component_inventory.clone(),
            component_charge_resolution: component_charge_resolution.clone(),
            source_solute_artifact: primary.source_detail.clone(),
            built_solute_artifact: primary.built_artifact.clone(),
            polymer_build_handoff: primary.polymer_build_handoff.clone(),
            polymer_artifact: primary.polymer_artifact.clone(),
            polymer_controls: primary.polymer_controls.clone(),
            charge_manifest_path: charge_manifest_path.clone(),
            charge_manifest_paths: charge_manifest_paths.clone(),
            charge_source_kinds: charge_source_kinds.clone(),
            net_charge_before_neutralization,
            neutralization_policy: neutralization_policy.clone(),
            warnings,
            engine_decisions: json!({
                "box_resolution": box_decisions,
                "morphology_policy": {
                    "mode": req.environment.morphology.mode,
                    "alignment_axis": alignment_axis,
                    "component_instances": total_component_instances(&components),
                    "typed_graph_helper_applied": !component_packing_hints.is_empty(),
                    "graph_architecture_cohort": {
                        "typed_component_count": graph_cohort.typed_component_count,
                        "typed_instance_count": graph_cohort.typed_instance_count,
                        "distinct_architecture_count": graph_cohort.distinct_architecture_count(),
                        "linear_instances": graph_cohort.linear_count,
                        "branched_instances": graph_cohort.branched_count,
                        "cyclic_instances": graph_cohort.cyclic_count,
                        "branched_cyclic_instances": graph_cohort.branched_cyclic_count,
                    },
                    "component_packing_hints": component_packing_hints,
                    "orientation_policy": match req.environment.morphology.mode.as_str() {
                        "single_chain_solution" => "single_component_centered",
                        "amorphous_bulk" => "random_orientation",
                        "backbone_aligned_bulk" => "principal_axis_aligned.azimuth_staggered",
                        _ => "unknown",
                    },
                },
                "solvent_policy": req.environment.solvent,
                "ion_policy": {
                    "neutralize": req.environment.ions.neutralize.enabled(),
                    "neutralize_with": req.environment.ions.neutralize.preferred_ion(),
                    "legacy_pair": if req.environment.ions.cation.is_some() || req.environment.ions.anion.is_some() {
                        Some(json!({
                            "cation": req.environment.ions.legacy_cation(),
                            "anion": req.environment.ions.legacy_anion(),
                        }))
                    } else {
                        None
                    },
                    "salt": normalized_salt.as_ref().map(|salt| json!({
                        "name": salt.name,
                        "formula": salt.formula,
                        "species": salt.species,
                        "molar": salt.molar,
                    })),
                    "salt_stoichiometry": salt_stoich,
                    "salt_pair_count": salt_pair_count,
                    "net_charge_before_neutralization": net_charge_before_neutralization,
                    "neutralization_policy": neutralization_policy,
                },
                "output_format": resolved_output_format(&req.outputs),
                "write_conect": req.outputs.write_conect,
                "preserve_topology_graph": req.outputs.preserve_topology_graph,
                "md_package": resolved_md_package_path(&req.outputs),
                "charge_sources": {
                    "manifest_paths": charge_manifest_paths,
                    "source_kinds": charge_source_kinds,
                    "component_resolution": component_charge_resolution,
                },
                "components": component_inventory,
                "polymer_build_handoff": primary.polymer_build_handoff.clone(),
                "polymer_controls": primary.polymer_controls.clone(),
            }),
        },
    ))
}

fn build_manifest(
    req: &BuildRequest,
    chemistry: &ChemistryCatalog,
    cfg: &PackConfig,
    metadata: &TranslationMetadata,
    total_atoms: usize,
    coordinates_path: &str,
    coordinates_format: &str,
    md_package_path: &str,
    md_package_digest: Option<String>,
) -> PackResult<Value> {
    let normalized_salt = req.environment.ions.normalized_salt_with(chemistry)?;
    let request_value =
        serde_json::to_value(req).map_err(|err| PackError::Parse(err.to_string()))?;
    let cfg_value = serde_json::to_value(cfg).map_err(|err| PackError::Parse(err.to_string()))?;
    let component_input_digests = metadata
        .component_inventory
        .iter()
        .map(|item| {
            json!({
                "name": item.get("name").cloned().unwrap_or(Value::Null),
                "coordinates": item.get("path").and_then(Value::as_str).and_then(|path| sha256_file(Path::new(path)).ok()),
                "topology": item.get("topology").and_then(Value::as_str).and_then(|path| sha256_file(Path::new(path)).ok()),
                "topology_graph": item.get("topology_graph").and_then(Value::as_str).and_then(|path| sha256_file(Path::new(path)).ok()),
                "charge_manifest": item.get("charge_manifest").and_then(Value::as_str).and_then(|path| sha256_file(Path::new(path)).ok()),
                "build_manifest": item.get("build_manifest").and_then(Value::as_str).and_then(|path| sha256_file(Path::new(path)).ok()),
            })
        })
        .collect::<Vec<_>>();
    let build_metadata = json!({
        "git_commit": option_env!("VERGEN_GIT_SHA"),
        "build_timestamp": option_env!("VERGEN_BUILD_TIMESTAMP"),
        "target_triple": option_env!("TARGET"),
    });
    let provenance = json!({
        "cli_version": env!("CARGO_PKG_VERSION"),
        "binary_version": env!("CARGO_PKG_VERSION"),
        "schema_version": AGENT_SCHEMA_VERSION,
        "request_hash": hash_value(&request_value)?,
        "engine_config_hash": hash_value(&cfg_value)?,
        "build_metadata": build_metadata,
    });
    let artifact_digests = json!({
        "coordinates": sha256_file(Path::new(coordinates_path)).ok(),
        "md_package": md_package_digest,
        "component_inputs": component_input_digests,
    });
    let engine_decisions = json!({
        "normalized_pack_config_hash": hash_value(&cfg_value)?,
        "request_charge_sources": metadata.charge_manifest_paths,
        "details": metadata.engine_decisions,
    });

    let mut inventory = metadata.component_inventory.clone();
    let mut primary = metadata.source_solute_artifact.clone();
    if let Some(obj) = primary.as_object_mut() {
        obj.insert("path".into(), Value::String(cfg.structures[0].path.clone()));
        obj.insert(
            "atom_count".into(),
            Value::Number(serde_json::Number::from(metadata.solute_context.atoms)),
        );
        if let Some(built) = metadata.built_solute_artifact.clone() {
            obj.insert("built_solute_artifact".into(), built);
        }
    }

    if metadata.water_count > 0 {
        inventory.push(json!({
            "name": "solvent",
            "kind": "water",
            "model": req.environment.solvent.model,
            "molecule_count": metadata.water_count,
        }));
    }
    for (species, count) in &metadata.ion_counts {
        if *count > 0 {
            inventory.push(json!({
                "name": species,
                "kind": "ion",
                "molecule_count": count,
            }));
        }
    }

    Ok(json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "run_id": req.run_id,
        "source_solute_artifact": metadata.source_solute_artifact,
        "built_solute_artifact": metadata.built_solute_artifact,
        "polymer_build_handoff": metadata.polymer_build_handoff,
        "md_ready_polymer_handoff": metadata.polymer_build_handoff,
        "polymer_artifact": metadata.polymer_artifact,
        "polymer_controls": metadata.polymer_controls,
        "charge_manifest_path": metadata.charge_manifest_path,
        "charge_manifest_paths": metadata.charge_manifest_paths,
        "charge_source_kinds": metadata.charge_source_kinds,
        "component_charge_resolution": metadata.component_charge_resolution,
        "final_box_size_angstrom": metadata.resolved_box_size,
        "final_box_vectors_angstrom": [
            [metadata.resolved_box_size[0], 0.0, 0.0],
            [0.0, metadata.resolved_box_size[1], 0.0],
            [0.0, 0.0, metadata.resolved_box_size[2]],
        ],
        "final_box_dimensions_angstrom": {
            "lx": metadata.resolved_box_size[0],
            "ly": metadata.resolved_box_size[1],
            "lz": metadata.resolved_box_size[2],
        },
        "component_inventory": inventory,
        "warnings": metadata.warnings,
        "polymer_chain_count": metadata.solute_context.chain_count,
        "residue_counts": metadata.solute_context.residue_counts,
        "water_count": metadata.water_count,
        "ion_counts": metadata.ion_counts,
        "net_charge_before_neutralization": metadata.net_charge_before_neutralization,
        "neutralization_policy_applied": metadata.neutralization_policy,
        "target_salt_name": normalized_salt.as_ref().and_then(|salt| salt.name.clone()),
        "target_salt_concentration_molar": normalized_salt.as_ref().and_then(|salt| salt.molar),
        "target_salt_formula": normalized_salt.as_ref().and_then(|salt| salt.formula.clone()),
        "achieved_salt_count": metadata.salt_pair_count,
        "achieved_salt_counts_by_species": metadata.ion_counts,
        "morphology_mode": req.environment.morphology.mode,
        "morphology": {
            "mode": req.environment.morphology.mode,
            "alignment_axis": req.environment.morphology.alignment_axis,
            "target_density_g_cm3": req.environment.morphology.target_density_g_cm3,
        },
        "output_metadata": {
            "coordinates": {
                "path": coordinates_path,
                "format": coordinates_format,
                "write_conect": req.outputs.write_conect,
            },
            "manifest": {
                "path": req.outputs.manifest,
                "format": "json",
            },
            "md_package": {
                "path": md_package_path,
                "format": "json",
                "preserve_topology_graph": req.outputs.preserve_topology_graph,
            }
        },
        "artifact_digests": artifact_digests,
        "engine_decisions": engine_decisions,
        "outputs": {
            "coordinates": coordinates_path,
            "manifest": req.outputs.manifest,
            "md_package": md_package_path,
            "format": coordinates_format,
            "write_conect": req.outputs.write_conect,
            "preserve_topology_graph": req.outputs.preserve_topology_graph,
        },
        "provenance": provenance,
        "summary": {
            "component_count": component_count(metadata),
            "total_atoms": total_atoms,
            "water_count": metadata.water_count,
            "ion_counts": metadata.ion_counts,
        },
    }))
}

fn build_md_ready_package(
    req: &BuildRequest,
    metadata: &TranslationMetadata,
    coordinates_path: &str,
    coordinates_format: &str,
    md_package_path: &str,
) -> Value {
    let preserved_topology_graphs = if req.outputs.preserve_topology_graph {
        metadata
            .component_inventory
            .iter()
            .filter_map(|item| item.get("topology_graph").cloned())
            .filter(|value| !value.is_null())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    json!({
        "schema_version": "warp-md.md-ready-package.v1",
        "run_id": req.run_id,
        "coordinates": {
            "path": coordinates_path,
            "format": coordinates_format,
            "write_conect": req.outputs.write_conect,
        },
        "pack_manifest": {
            "path": req.outputs.manifest,
        },
        "md_package_path": md_package_path,
        "preserve_topology_graph": req.outputs.preserve_topology_graph,
        "topology_graphs": preserved_topology_graphs,
        "polymer_build_handoff": if req.outputs.preserve_topology_graph {
            metadata.polymer_build_handoff.clone()
        } else {
            None
        },
        "component_inventory": metadata.component_inventory,
        "md_engine_helpers": {
            "openmm": {
                "coordinates": coordinates_path,
                "pack_manifest": req.outputs.manifest,
                "polymer_build_handoff": if req.outputs.preserve_topology_graph {
                    metadata.polymer_build_handoff.clone()
                } else {
                    None
                },
            }
        }
    })
}

fn validate_warnings(req: &BuildRequest) -> PackResult<Vec<ErrorDetail>> {
    let components = resolve_components(req)?;
    Ok(collect_warnings(req, &components))
}

fn resolved_inputs(req: &BuildRequest) -> PackResult<Value> {
    let components = resolve_components(req)?;
    let chemistry = chemistry_catalog_for_request(&req.environment.ions)?;
    let normalized_salt = req.environment.ions.normalized_salt_with(&chemistry)?;
    let total_instances = total_component_instances(&components);
    let graph_versions = components
        .iter()
        .filter_map(|component| {
            component
                .topology_graph
                .as_ref()
                .map(|graph| graph.schema_version.clone())
        })
        .collect::<BTreeSet<_>>();
    let build_manifest_versions = components
        .iter()
        .filter_map(|component| {
            component
                .polymer_build_handoff
                .as_ref()
                .and_then(|value| value.get("manifest_version"))
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .collect::<BTreeSet<_>>();
    let charge_source_kinds = components
        .iter()
        .flat_map(|component| component.charge_source_kinds.iter().cloned())
        .collect::<BTreeSet<_>>();
    let missing_neutralization = components
        .iter()
        .filter(|component| component.per_instance_net_charge.is_none())
        .map(|component| component.name.clone())
        .collect::<Vec<_>>();
    Ok(json!({
        "morphology_mode": req.environment.morphology.mode,
        "component_inventory_count": components.len(),
        "component_instance_count": total_instances,
        "polymer_build_handoff_present": components.iter().any(|component| component.source_kind == "polymer_build"),
        "polymer_build_manifest_version": if build_manifest_versions.len() == 1 {
            build_manifest_versions.iter().next().cloned().map(Value::String).unwrap_or(Value::Null)
        } else {
            Value::Array(build_manifest_versions.into_iter().map(Value::String).collect())
        },
        "topology_graph_present": components.iter().any(|component| component.topology_graph.is_some()),
        "topology_graph_version": if graph_versions.len() == 1 {
            graph_versions.iter().next().cloned().map(Value::String).unwrap_or(Value::Null)
        } else {
            Value::Array(graph_versions.into_iter().map(Value::String).collect())
        },
        "output_format": resolved_output_format(&req.outputs),
        "md_package": resolved_md_package_path(&req.outputs),
        "write_conect": req.outputs.write_conect,
        "preserve_topology_graph": req.outputs.preserve_topology_graph,
        "charge_source_kind": if charge_source_kinds.len() == 1 {
            charge_source_kinds.iter().next().cloned().map(Value::String).unwrap_or(Value::Null)
        } else if charge_source_kinds.is_empty() {
            Value::Null
        } else {
            Value::String("mixed".into())
        },
        "charge_source_kinds": charge_source_kinds.into_iter().collect::<Vec<_>>(),
        "neutralization_preconditions": {
            "requested": req.environment.ions.neutralize.enabled(),
            "satisfied": !req.environment.ions.neutralize.enabled() || missing_neutralization.is_empty(),
            "missing_charge_components": missing_neutralization,
        },
        "salt": normalized_salt.as_ref().map(|salt| json!({
            "name": salt.name,
            "formula": salt.formula,
            "species": salt.species,
            "molar": salt.molar,
        })),
    }))
}

fn error_to_envelope(run_id: Option<String>, err: PackError) -> RunErrorEnvelope {
    let (exit_code, code) = match err {
        PackError::Invalid(_) | PackError::Parse(_) => (2, "E_CONFIG_VALIDATION"),
        PackError::Io(_) => (4, "E_OUTPUT_WRITE"),
        PackError::Placement(_) => (4, "E_RUNTIME_EXEC"),
    };
    let message = err.to_string();
    let detail = error_detail(code, None, message);
    RunErrorEnvelope {
        schema_version: AGENT_SCHEMA_VERSION.into(),
        status: "error".into(),
        run_id,
        exit_code,
        error: detail.clone(),
        errors: vec![detail],
        warnings: Vec::new(),
    }
}

pub fn schema_json(kind: &str) -> PackResult<String> {
    let value = match kind {
        "request" => serde_json::to_value(&schema_for!(BuildRequest))
            .map_err(|err| PackError::Parse(err.to_string()))?,
        "result" => serde_json::to_value(&schema_for!(ResultEnvelopeSchema))
            .map_err(|err| PackError::Parse(err.to_string()))?,
        "event" => serde_json::to_value(&schema_for!(RunEvent))
            .map_err(|err| PackError::Parse(err.to_string()))?,
        _ => {
            return Err(PackError::Invalid(
                "schema target must be request, result, or event".into(),
            ))
        }
    };
    serde_json::to_string_pretty(&value).map_err(|err| PackError::Parse(err.to_string()))
}

pub fn example_request(mode: &str) -> PackResult<Value> {
    match mode {
        "solute_solvate" => Ok(json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "run_id": "solute-solvate-001",
            "solute": {
                "path": "polymer_50mer.pdb",
                "topology": "polymer_50mer.prmtop",
                "kind": "polymer_chain",
                "charge_manifest": "polymer_50mer_charge_manifest.json",
            },
            "environment": {
                "box": {"mode": "padding", "padding_angstrom": 12.0, "shape": "cubic"},
                "solvent": {"mode": "explicit", "model": "tip3p"},
                "ions": {
                    "neutralize": {"enabled": true},
                    "salt": {"name": "nacl", "molar": 0.15}
                },
                "morphology": {"mode": "single_chain_solution"},
            },
            "outputs": {
                "coordinates": "outputs/system.pdb",
                "manifest": "outputs/system_manifest.json",
                "md_package": "outputs/system_manifest.md-ready.json",
                "format": "pdb-strict",
                "write_conect": true,
                "preserve_topology_graph": true
            },
        })),
        "polymer_build_handoff" => Ok(json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "run_id": "warp-build-handoff-001",
            "polymer_build": {
                "build_manifest": "outputs/pmma_50mer.build.json",
                "topology_graph": "outputs/pmma_50mer.topology.json"
            },
            "environment": {
                "box": {"mode": "padding", "padding_angstrom": 12.0, "shape": "cubic"},
                "solvent": {"mode": "explicit", "model": "tip3p"},
                "ions": {
                    "neutralize": {"enabled": true},
                    "salt": {"name": "nacl", "molar": 0.15}
                },
                "morphology": {"mode": "single_chain_solution"},
            },
            "outputs": {
                "coordinates": "outputs/polymer_50mer_solvated.pdb",
                "manifest": "outputs/polymer_50mer_solvated_manifest.json",
                "md_package": "outputs/polymer_50mer_solvated_manifest.md-ready.json",
                "format": "pdb-strict",
                "write_conect": true,
                "preserve_topology_graph": true
            },
        })),
        "components_amorphous_bulk" => Ok(json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "run_id": "components-bulk-001",
            "components": [
                {
                    "name": "chain_a",
                    "count": 4,
                    "source": {
                        "kind": "polymer_build",
                        "polymer_build": {
                            "build_manifest": "outputs/pmma_50mer.build.json"
                        }
                    }
                },
                {
                    "name": "dopant",
                    "count": 8,
                    "source": {
                        "kind": "artifact",
                        "artifact": {
                            "path": "dopant.pdb",
                            "kind": "small_molecule",
                            "topology": "dopant.prmtop"
                        }
                    }
                }
            ],
            "environment": {
                "box": {"mode": "fixed_size", "size_angstrom": [80.0, 80.0, 80.0], "shape": "orthorhombic"},
                "solvent": {"mode": "none"},
                "ions": {
                    "neutralize": {"enabled": true},
                    "salt": {"name": "nacl", "molar": 0.0}
                },
                "morphology": {"mode": "amorphous_bulk"},
            },
            "outputs": {
                "coordinates": "outputs/components_bulk.pdb",
                "manifest": "outputs/components_bulk_manifest.json",
                "md_package": "outputs/components_bulk_manifest.md-ready.json",
                "format": "pdb-strict",
                "write_conect": true,
                "preserve_topology_graph": true
            },
        })),
        "components_backbone_aligned_bulk" => Ok(json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "run_id": "components-aligned-001",
            "components": [
                {
                    "name": "chain_a",
                    "count": 4,
                    "source": {
                        "kind": "polymer_build",
                        "polymer_build": {
                            "build_manifest": "outputs/pmma_50mer.build.json"
                        }
                    }
                }
            ],
            "environment": {
                "box": {"mode": "fixed_size", "size_angstrom": [80.0, 80.0, 120.0], "shape": "orthorhombic"},
                "solvent": {"mode": "none"},
                "ions": {
                    "neutralize": {"enabled": true},
                    "salt": {"name": "nacl", "molar": 0.0}
                },
                "morphology": {"mode": "backbone_aligned_bulk", "alignment_axis": "z"},
            },
            "outputs": {
                "coordinates": "outputs/components_aligned_bulk.pdb",
                "manifest": "outputs/components_aligned_bulk_manifest.json",
                "md_package": "outputs/components_aligned_bulk_manifest.md-ready.json",
                "format": "pdb-strict",
                "write_conect": true,
                "preserve_topology_graph": true
            },
        })),
        _ => Err(PackError::Invalid(
            "example mode must be solute_solvate, polymer_build_handoff, components_amorphous_bulk, or components_backbone_aligned_bulk".into(),
        )),
    }
}

pub fn capabilities() -> Value {
    let supported_ions = supported_ion_species().unwrap_or_default();
    let supported_salts = supported_salt_names().unwrap_or_default();
    json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "supported_build_modes": SUPPORTED_BUILD_MODES,
        "supported_component_kinds": SUPPORTED_COMPONENT_KINDS,
        "supported_solvent_models": SUPPORTED_SOLVENT_MODELS,
        "supported_ion_species": supported_ions,
        "supported_salt_names": supported_salts,
        "supported_custom_chemistry_inputs": ["catalog.ions", "catalog.salts"],
        "supported_salt_inputs": ["salt.name", "salt.formula", "salt.species", "legacy_pair"],
        "supported_morphology_modes": SUPPORTED_MORPHOLOGY_MODES,
        "supported_charge_sources": ["charge_manifest", "prmtop"],
        "supported_output_formats": SUPPORTED_OUTPUT_FORMATS,
        "supported_output_controls": ["format", "write_conect", "preserve_topology_graph", "md_package"],
        "supported_ion_controls": [
            "neutralize",
            "neutralize.with",
            "salt.name",
            "salt.formula",
            "salt.species",
            "salt.molar",
            "catalog.ions",
            "catalog.salts"
        ],
        "supported_polymer_build_handoff_artifacts": ["build_manifest", "coordinates", "charge_manifest", "topology", "topology_graph"],
        "schema_targets": ["request", "result", "event"],
        "polymer_build_supported": true,
        "polymer_build_handoff_supported": true,
        "preferred_solute_input": "components",
        "supported_solute_inputs": ["components", "solute", "polymer_build"],
        "streaming_supported": true,
        "manifest_supported": true,
    })
}

pub fn validate_request_json(text: &str) -> (i32, Value) {
    match load_agent_request(text) {
        Ok(req) => {
            let normalized_request = serde_json::to_value(&req)
                .unwrap_or_else(|_| json!({"schema_version": AGENT_SCHEMA_VERSION}));
            match resolved_inputs(&req) {
                Ok(resolved_inputs) => (
                    0,
                    json!(ValidateSuccessEnvelope {
                        schema_version: AGENT_SCHEMA_VERSION.into(),
                        status: "ok".into(),
                        valid: true,
                        normalized_request,
                        resolved_inputs,
                        warnings: validate_warnings(&req).unwrap_or_default(),
                    }),
                ),
                Err(err) => {
                    let envelope = error_to_envelope(req.run_id.clone(), err);
                    (
                        envelope.exit_code,
                        json!({
                            "schema_version": AGENT_SCHEMA_VERSION,
                            "status": "error",
                            "valid": false,
                            "errors": envelope.errors,
                            "warnings": envelope.warnings,
                        }),
                    )
                }
            }
        }
        Err(err) => (
            err.exit_code,
            json!({
                "schema_version": AGENT_SCHEMA_VERSION,
                "status": "error",
                "valid": false,
                "errors": err.errors,
                "warnings": err.warnings,
            }),
        ),
    }
}

pub fn run_request_json(text: &str, stream_ndjson: bool) -> (i32, Value) {
    let started = Instant::now();
    let req = match load_agent_request(text) {
        Ok(req) => req,
        Err(err) => {
            emit_event(
                &RunEvent::RunFailed {
                    schema_version: AGENT_SCHEMA_VERSION.into(),
                    run_id: "warp-pack-run".into(),
                    elapsed_ms: elapsed_ms(&started),
                    final_envelope: err.clone(),
                },
                stream_ndjson,
            );
            return (
                err.exit_code,
                serde_json::to_value(err).unwrap_or_else(|_| json!({"status":"error"})),
            );
        }
    };

    let request_value = match serde_json::to_value(&req) {
        Ok(value) => value,
        Err(err) => {
            let envelope = error_to_envelope(
                req.run_id.clone(),
                PackError::Parse(format!("request serialization failed: {err}")),
            );
            return (
                envelope.exit_code,
                serde_json::to_value(envelope).unwrap_or_else(|_| json!({"status":"error"})),
            );
        }
    };
    let request_hash = hash_value(&request_value).unwrap_or_else(|_| "warp-pack-run".into());
    let run_id = req
        .run_id
        .clone()
        .unwrap_or_else(|| request_hash.chars().take(12).collect());

    emit_event(
        &RunEvent::RunStarted {
            schema_version: AGENT_SCHEMA_VERSION.into(),
            run_id: run_id.clone(),
            elapsed_ms: 0,
        },
        stream_ndjson,
    );

    let result = (|| -> PackResult<RunSuccessEnvelope> {
        let chemistry = chemistry_catalog_for_request(&req.environment.ions)?;
        emit_event(
            &RunEvent::PhaseStarted {
                schema_version: AGENT_SCHEMA_VERSION.into(),
                run_id: run_id.clone(),
                phase: "world_build".into(),
                elapsed_ms: elapsed_ms(&started),
            },
            stream_ndjson,
        );
        emit_event(
            &RunEvent::PhaseProgress {
                schema_version: AGENT_SCHEMA_VERSION.into(),
                run_id: run_id.clone(),
                phase: "world_build".into(),
                progress_pct: 10.0,
                elapsed_ms: elapsed_ms(&started),
                eta_ms: eta_ms(&started, 10.0),
                artifact: None,
            },
            stream_ndjson,
        );
        let (cfg, metadata) = build_pack_config(&req, &chemistry)?;
        emit_event(
            &RunEvent::PhaseCompleted {
                schema_version: AGENT_SCHEMA_VERSION.into(),
                run_id: run_id.clone(),
                phase: "world_build".into(),
                elapsed_ms: elapsed_ms(&started),
                artifact: metadata
                    .built_solute_artifact
                    .as_ref()
                    .and_then(|value| value.get("path"))
                    .and_then(|value| value.as_str())
                    .map(ToOwned::to_owned),
            },
            stream_ndjson,
        );

        if req.environment.solvent.mode == "explicit" {
            emit_event(
                &RunEvent::PhaseStarted {
                    schema_version: AGENT_SCHEMA_VERSION.into(),
                    run_id: run_id.clone(),
                    phase: "solvation".into(),
                    elapsed_ms: elapsed_ms(&started),
                },
                stream_ndjson,
            );
            emit_event(
                &RunEvent::PhaseProgress {
                    schema_version: AGENT_SCHEMA_VERSION.into(),
                    run_id: run_id.clone(),
                    phase: "solvation".into(),
                    progress_pct: 25.0,
                    elapsed_ms: elapsed_ms(&started),
                    eta_ms: eta_ms(&started, 25.0),
                    artifact: None,
                },
                stream_ndjson,
            );
            emit_event(
                &RunEvent::PhaseCompleted {
                    schema_version: AGENT_SCHEMA_VERSION.into(),
                    run_id: run_id.clone(),
                    phase: "solvation".into(),
                    elapsed_ms: elapsed_ms(&started),
                    artifact: None,
                },
                stream_ndjson,
            );
        }

        if req.environment.ions.normalized_salt_with(&chemistry)?.is_some()
            || req.environment.ions.neutralize.enabled()
        {
            emit_event(
                &RunEvent::PhaseStarted {
                    schema_version: AGENT_SCHEMA_VERSION.into(),
                    run_id: run_id.clone(),
                    phase: "ionization".into(),
                    elapsed_ms: elapsed_ms(&started),
                },
                stream_ndjson,
            );
            emit_event(
                &RunEvent::PhaseProgress {
                    schema_version: AGENT_SCHEMA_VERSION.into(),
                    run_id: run_id.clone(),
                    phase: "ionization".into(),
                    progress_pct: 40.0,
                    elapsed_ms: elapsed_ms(&started),
                    eta_ms: eta_ms(&started, 40.0),
                    artifact: None,
                },
                stream_ndjson,
            );
            emit_event(
                &RunEvent::PhaseCompleted {
                    schema_version: AGENT_SCHEMA_VERSION.into(),
                    run_id: run_id.clone(),
                    phase: "ionization".into(),
                    elapsed_ms: elapsed_ms(&started),
                    artifact: None,
                },
                stream_ndjson,
            );
        }

        emit_event(
            &RunEvent::PhaseStarted {
                schema_version: AGENT_SCHEMA_VERSION.into(),
                run_id: run_id.clone(),
                phase: "packing".into(),
                elapsed_ms: elapsed_ms(&started),
            },
            stream_ndjson,
        );
        emit_event(
            &RunEvent::PhaseProgress {
                schema_version: AGENT_SCHEMA_VERSION.into(),
                run_id: run_id.clone(),
                phase: "packing".into(),
                progress_pct: 55.0,
                elapsed_ms: elapsed_ms(&started),
                eta_ms: eta_ms(&started, 55.0),
                artifact: None,
            },
            stream_ndjson,
        );

        let packed = run(&cfg)?;
        ensure_parent(&req.outputs.coordinates)?;
        let written_output = write_output(
            &packed,
            cfg.output
                .as_ref()
                .ok_or_else(|| PackError::Invalid("missing output spec".into()))?,
            cfg.add_box_sides,
            cfg.add_box_sides_fix.unwrap_or(0.0),
            req.outputs.write_conect,
            false,
        )?;
        let coordinates_path = written_output.path.clone();
        let coordinates_format = written_output.format.clone();

        emit_event(
            &RunEvent::PhaseProgress {
                schema_version: AGENT_SCHEMA_VERSION.into(),
                run_id: run_id.clone(),
                phase: "packing".into(),
                progress_pct: 85.0,
                elapsed_ms: elapsed_ms(&started),
                eta_ms: eta_ms(&started, 85.0),
                artifact: Some(coordinates_path.clone()),
            },
            stream_ndjson,
        );
        emit_event(
            &RunEvent::PhaseCompleted {
                schema_version: AGENT_SCHEMA_VERSION.into(),
                run_id: run_id.clone(),
                phase: "packing".into(),
                elapsed_ms: elapsed_ms(&started),
                artifact: Some(coordinates_path.clone()),
            },
            stream_ndjson,
        );

        emit_event(
            &RunEvent::PhaseStarted {
                schema_version: AGENT_SCHEMA_VERSION.into(),
                run_id: run_id.clone(),
                phase: "manifest".into(),
                elapsed_ms: elapsed_ms(&started),
            },
            stream_ndjson,
        );
        emit_event(
            &RunEvent::PhaseProgress {
                schema_version: AGENT_SCHEMA_VERSION.into(),
                run_id: run_id.clone(),
                phase: "manifest".into(),
                progress_pct: 95.0,
                elapsed_ms: elapsed_ms(&started),
                eta_ms: eta_ms(&started, 95.0),
                artifact: Some(req.outputs.manifest.clone()),
            },
            stream_ndjson,
        );

        let md_package_path = resolved_md_package_path(&req.outputs);
        let md_package = build_md_ready_package(
            &req,
            &metadata,
            &coordinates_path,
            &coordinates_format,
            &md_package_path,
        );
        ensure_parent(&md_package_path)?;
        fs::write(
            &md_package_path,
            format!(
                "{}\n",
                serde_json::to_string_pretty(&md_package)
                    .map_err(|err| PackError::Parse(err.to_string()))?
            ),
        )?;
        let md_package_digest = sha256_file(Path::new(&md_package_path)).ok();
        let manifest = build_manifest(
            &req,
            &chemistry,
            &cfg,
            &metadata,
            packed.atoms.len(),
            &coordinates_path,
            &coordinates_format,
            &md_package_path,
            md_package_digest,
        )?;
        ensure_parent(&req.outputs.manifest)?;
        fs::write(
            &req.outputs.manifest,
            format!(
                "{}\n",
                serde_json::to_string_pretty(&manifest)
                    .map_err(|err| PackError::Parse(err.to_string()))?
            ),
        )?;

        emit_event(
            &RunEvent::PhaseCompleted {
                schema_version: AGENT_SCHEMA_VERSION.into(),
                run_id: run_id.clone(),
                phase: "manifest".into(),
                elapsed_ms: elapsed_ms(&started),
                artifact: Some(req.outputs.manifest.clone()),
            },
            stream_ndjson,
        );

        Ok(RunSuccessEnvelope {
            schema_version: AGENT_SCHEMA_VERSION.into(),
            status: "ok".into(),
            run_id: Some(run_id.clone()),
            output_dir: common_output_dir(&coordinates_path, &req.outputs.manifest),
            artifacts: ArtifactEnvelope {
                coordinates: coordinates_path,
                manifest: req.outputs.manifest.clone(),
                md_package: Some(md_package_path.clone()),
            },
            summary: RunSummary {
                component_count: component_count(&metadata),
                total_atoms: packed.atoms.len(),
                water_count: metadata.water_count,
                ion_counts: metadata.ion_counts,
            },
            manifest_path: req.outputs.manifest.clone(),
            warnings: metadata.warnings.clone(),
        })
    })();

    match result {
        Ok(success) => {
            emit_event(
                &RunEvent::RunCompleted {
                    schema_version: AGENT_SCHEMA_VERSION.into(),
                    run_id: run_id.clone(),
                    elapsed_ms: elapsed_ms(&started),
                    final_envelope: success.clone(),
                },
                stream_ndjson,
            );
            (
                0,
                serde_json::to_value(success).unwrap_or_else(|_| json!({"status":"ok"})),
            )
        }
        Err(err) => {
            let envelope = error_to_envelope(Some(run_id.clone()), err);
            emit_event(
                &RunEvent::RunFailed {
                    schema_version: AGENT_SCHEMA_VERSION.into(),
                    run_id,
                    elapsed_ms: elapsed_ms(&started),
                    final_envelope: envelope.clone(),
                },
                stream_ndjson,
            );
            (
                envelope.exit_code,
                serde_json::to_value(envelope).unwrap_or_else(|_| json!({"status":"error"})),
            )
        }
    }
}
