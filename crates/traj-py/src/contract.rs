use super::*;

use crate::io::load_system_auto;
use schemars::schema_for;
use serde::ser::SerializeStruct;
use std::fmt::Write as _;

const WARP_MD_AGENT_SCHEMA_VERSION: &str = "warp-md.agent.v1";
const WARP_MD_RUN_REQUEST_TOP_LEVEL_FIELDS: &[&str] = &[
    "version",
    "run_id",
    "system",
    "topology",
    "trajectory",
    "traj",
    "inputs",
    "device",
    "stream",
    "chunk_frames",
    "output_dir",
    "checkpoint",
    "fail_fast",
    "analyses",
];

#[derive(Clone, Debug, serde::Serialize)]
struct WarpMdFieldSpec {
    #[serde(rename = "type")]
    field_type: String,
    semantic_type: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    default: Option<serde_json::Value>,
    #[serde(default)]
    minimum: Option<f64>,
    #[serde(default)]
    maximum: Option<f64>,
    #[serde(default)]
    unit: Option<String>,
    #[serde(default)]
    choices: Option<Vec<String>>,
}

#[derive(Clone, Debug)]
struct WarpMdArtifactSpec {
    kind: String,
    format: String,
    fields: Vec<String>,
    description: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct PlotAxisSpec {
    field: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    units: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct PlotRecommendation {
    plot_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    x: Option<PlotAxisSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    y: Option<PlotAxisSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    z: Option<PlotAxisSpec>,
    title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    artifact: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    shape: Option<Vec<u64>>,
    #[serde(flatten)]
    extra: std::collections::BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct ArtifactCompanionSpec {
    format: String,
    role: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    fields: Vec<String>,
}

impl serde::Serialize for WarpMdArtifactSpec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let plot_recommendations = warp_md_default_plot_recommendations(self);
        let companions = warp_md_default_companion_specs(self);
        let mut field_count = 4;
        if self.fields.is_empty() {
            field_count -= 1;
        }
        if self.description.is_none() {
            field_count -= 1;
        }
        if !plot_recommendations.is_empty() {
            field_count += 1;
        }
        if !companions.is_empty() {
            field_count += 1;
        }
        let mut state = serializer.serialize_struct("WarpMdArtifactSpec", field_count)?;
        state.serialize_field("kind", &self.kind)?;
        state.serialize_field("format", &self.format)?;
        if !self.fields.is_empty() {
            state.serialize_field("fields", &self.fields)?;
        }
        if let Some(description) = &self.description {
            state.serialize_field("description", description)?;
        }
        if !plot_recommendations.is_empty() {
            state.serialize_field("plot_recommendations", &plot_recommendations)?;
        }
        if !companions.is_empty() {
            state.serialize_field("companions", &companions)?;
        }
        state.end()
    }
}

fn warp_md_default_plot_recommendations(artifact: &WarpMdArtifactSpec) -> Vec<PlotRecommendation> {
    let fields = &artifact.fields;
    match artifact.kind.as_str() {
        "timeseries" | "histogram" | "profile" if fields.len() >= 2 => fields[1..]
            .iter()
            .filter(|field| field.as_str() != "...")
            .map(|field| PlotRecommendation {
                plot_type: "line".into(),
                x: Some(warp_md_plot_axis(&fields[0], None)),
                y: Some(warp_md_plot_axis(field, None)),
                z: None,
                title: artifact
                    .description
                    .clone()
                    .unwrap_or_else(|| warp_md_title_from_field(field)),
                artifact: None,
                shape: None,
                extra: std::collections::BTreeMap::new(),
            })
            .collect(),
        "table" if fields.len() >= 2 => vec![PlotRecommendation {
            plot_type: "bar".into(),
            x: Some(warp_md_plot_axis(&fields[0], None)),
            y: Some(warp_md_plot_axis(&fields[1], None)),
            z: None,
            title: artifact
                .description
                .clone()
                .unwrap_or_else(|| warp_md_title_from_field(&fields[1])),
            artifact: None,
            shape: None,
            extra: std::collections::BTreeMap::new(),
        }],
        "grid" if !fields.is_empty() => vec![PlotRecommendation {
            plot_type: "volume_grid".into(),
            x: None,
            y: None,
            z: Some(warp_md_plot_axis(&fields[0], None)),
            title: artifact
                .description
                .clone()
                .unwrap_or_else(|| warp_md_title_from_field(&fields[0])),
            artifact: None,
            shape: None,
            extra: std::collections::BTreeMap::new(),
        }],
        "artifact" if fields.len() >= 2 => vec![PlotRecommendation {
            plot_type: "line".into(),
            x: Some(warp_md_plot_axis(&fields[0], None)),
            y: Some(warp_md_plot_axis(&fields[1], None)),
            z: None,
            title: artifact
                .description
                .clone()
                .unwrap_or_else(|| warp_md_title_from_field(&fields[1])),
            artifact: None,
            shape: None,
            extra: std::collections::BTreeMap::new(),
        }],
        _ => Vec::new(),
    }
}

fn warp_md_default_companion_specs(artifact: &WarpMdArtifactSpec) -> Vec<ArtifactCompanionSpec> {
    if artifact.format.as_str() != "npz" {
        return Vec::new();
    }
    vec![
        ArtifactCompanionSpec {
            format: "json".into(),
            role: "npz_companion_manifest".into(),
            fields: artifact.fields.clone(),
        },
        ArtifactCompanionSpec {
            format: "csv".into(),
            role: "array_table".into(),
            fields: artifact.fields.clone(),
        },
    ]
}

fn warp_md_plot_axis(field: &str, source: Option<&str>) -> PlotAxisSpec {
    PlotAxisSpec {
        field: field.into(),
        units: warp_md_field_units(field),
        source: source.map(str::to_string),
    }
}

fn warp_md_field_units(field: &str) -> Option<String> {
    let unit = if field.ends_with("_ps") {
        "ps"
    } else if field.ends_with("_nm2") || field.ends_with("_nm_2") {
        "nm^2"
    } else if field.ends_with("_nm") || field == "position" || field == "distance_nm" {
        "nm"
    } else if field.ends_with("_a3") {
        "angstrom^3"
    } else if field.ends_with("_hz") {
        "Hz"
    } else if field.ends_with("_kJ_per_mol") {
        "kJ/mol"
    } else if field.ends_with("_S_per_cm") {
        "S/cm"
    } else if field.ends_with("_g_cm3") {
        "g/cm^3"
    } else if matches!(
        field,
        "probability" | "gr" | "acf" | "correlation" | "q_value" | "structure_factor" | "cos_theta"
    ) {
        "dimensionless"
    } else {
        return None;
    };
    Some(unit.into())
}

fn warp_md_title_from_field(field: &str) -> String {
    field
        .replace('_', " ")
        .split_whitespace()
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[derive(Clone, Debug, serde::Serialize)]
struct InputRequirements {
    required: Vec<String>,
    optional: Vec<String>,
    requires_box: bool,
    requires_velocities: bool,
    requires_charges: bool,
    requires_selections: bool,
    supports_no_trajectory: bool,
    selection_fields: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize)]
struct AnalysisBundle {
    name: &'static str,
    description: &'static str,
    analyses: &'static [&'static str],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    external_tables: Option<&'static [&'static str]>,
}

const WARP_MD_ANALYSIS_BUNDLES: &[AnalysisBundle] = &[
    AnalysisBundle {
        name: "standard_md_report",
        description: "General MD report: structure, transport, density, and external state/energy series.",
        analyses: &["rg", "rmsd", "rdf", "msd", "density"],
        external_tables: Some(&["energy_table", "state_table"]),
    },
    AnalysisBundle {
        name: "protein_md_report",
        description: "Protein trajectory report: fold stability, secondary structure, hydrogen bonds, contacts.",
        analyses: &["rg", "rmsd", "dssp", "hbond", "native_contacts"],
        external_tables: None,
    },
    AnalysisBundle {
        name: "solvent_ion_report",
        description: "Solvent and ion report: RDF, diffusion, electrostatics, and hydration structure.",
        analyses: &["rdf", "msd", "conductivity", "dielectric", "water_count", "watershell"],
        external_tables: None,
    },
    AnalysisBundle {
        name: "polymer_report",
        description: "Polymer report: size, chain geometry, persistence, and free volume.",
        analyses: &[
            "rg",
            "chain_rg",
            "end_to_end",
            "contour_length",
            "persistence_length",
            "bondi_ffv",
        ],
        external_tables: None,
    },
];

const WARP_MD_BOX_REQUIRED_ANALYSES: &[&str] = &[
    "bondi_ffv",
    "conductivity",
    "density",
    "dielectric",
    "free_volume",
    "gist",
    "rdf",
    "structure_factor",
    "volmap",
    "water_count",
    "watershell",
];

const WARP_MD_VELOCITY_REQUIRED_ANALYSES: &[&str] = &["equipartition"];

const WARP_MD_ERROR_CODES: &[&str] = &[
    "E_CONFIG_LOAD",
    "E_CONFIG_VALIDATION",
    "E_CONFIG_VERSION",
    "E_CONFIG_MISSING_FIELD",
    "E_ANALYSIS_UNKNOWN",
    "E_ANALYSIS_SPEC",
    "E_SELECTION_EMPTY",
    "E_SELECTION_INVALID",
    "E_SYSTEM_LOAD",
    "E_TRAJECTORY_LOAD",
    "E_TRAJECTORY_EOF",
    "E_RUNTIME_EXEC",
    "E_OUTPUT_DIR",
    "E_OUTPUT_WRITE",
    "E_DEVICE_UNAVAILABLE",
    "E_INPUT_MISSING",
    "E_UNSUPPORTED_FORMAT",
    "E_TOPOLOGY_TRAJECTORY_MISMATCH",
    "E_TOPOLOGY_ATOM_MISSING",
    "E_NO_FRAMES",
    "E_EXTERNAL_TABLE_LOAD",
    "E_EXTERNAL_TABLE_COLUMN",
    "E_PLOT_RENDER",
    "E_BUNDLE_PARTIAL",
    "E_INTERNAL",
];

#[derive(Clone, Debug)]
struct WarpMdAnalysisContract {
    name: String,
    aliases: Vec<String>,
    description: String,
    required_fields: Vec<String>,
    optional_fields: Vec<String>,
    fields: std::collections::BTreeMap<String, WarpMdFieldSpec>,
    outputs: Vec<WarpMdArtifactSpec>,
    tags: Vec<String>,
    examples: Vec<serde_json::Value>,
}

impl serde::Serialize for WarpMdAnalysisContract {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let input_requirements = warp_md_input_requirements(self);
        let mut state = serializer.serialize_struct("WarpMdAnalysisContract", 10)?;
        state.serialize_field("name", &self.name)?;
        state.serialize_field("aliases", &self.aliases)?;
        state.serialize_field("description", &self.description)?;
        state.serialize_field("required_fields", &self.required_fields)?;
        state.serialize_field("optional_fields", &self.optional_fields)?;
        state.serialize_field("fields", &self.fields)?;
        state.serialize_field("outputs", &self.outputs)?;
        state.serialize_field("input_requirements", &input_requirements)?;
        state.serialize_field("tags", &self.tags)?;
        state.serialize_field("examples", &self.examples)?;
        state.end()
    }
}

fn warp_md_input_requirements(contract: &WarpMdAnalysisContract) -> InputRequirements {
    let selection_fields: Vec<String> = contract
        .fields
        .iter()
        .filter_map(|(name, spec)| match spec.semantic_type.as_str() {
            "selection" | "mask" => Some(name.clone()),
            _ => None,
        })
        .collect();
    let requires_box = WARP_MD_BOX_REQUIRED_ANALYSES.contains(&contract.name.as_str());
    let requires_velocities = WARP_MD_VELOCITY_REQUIRED_ANALYSES.contains(&contract.name.as_str());
    let requires_charges = contract
        .required_fields
        .iter()
        .any(|field| field == "charges");
    InputRequirements {
        required: vec!["topology".into(), "trajectory".into()],
        optional: Vec::new(),
        requires_box,
        requires_velocities,
        requires_charges,
        requires_selections: !selection_fields.is_empty(),
        supports_no_trajectory: false,
        selection_fields,
    }
}

#[derive(Clone, Debug, serde::Serialize)]
struct WarpMdContractCatalog {
    schema_version: String,
    #[serde(default)]
    analysis_shared_fields: Vec<String>,
    #[serde(default)]
    cli_to_analysis: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    analyses: Vec<WarpMdAnalysisContract>,
}

static WARP_MD_AGENT_CONTRACT_CATALOG: std::sync::OnceLock<WarpMdContractCatalog> =
    std::sync::OnceLock::new();
static WARP_MD_AGENT_NAME_LOOKUP: std::sync::OnceLock<std::collections::BTreeMap<String, String>> =
    std::sync::OnceLock::new();

mod catalog;

use self::catalog::warp_md_agent_contract_catalog_native;

fn warp_md_agent_contract_catalog_ref() -> &'static WarpMdContractCatalog {
    WARP_MD_AGENT_CONTRACT_CATALOG.get_or_init(warp_md_agent_contract_catalog_native)
}

fn warp_md_agent_name_lookup() -> &'static std::collections::BTreeMap<String, String> {
    WARP_MD_AGENT_NAME_LOOKUP.get_or_init(|| {
        let catalog = warp_md_agent_contract_catalog_ref();
        let mut lookup = std::collections::BTreeMap::new();
        for contract in &catalog.analyses {
            for alias in contract
                .aliases
                .iter()
                .cloned()
                .chain(std::iter::once(contract.name.clone()))
                .chain(std::iter::once(contract.name.replace('_', "-")))
            {
                lookup.insert(alias.clone(), contract.name.clone());
                lookup.insert(alias.replace('-', "_"), contract.name.clone());
            }
        }
        for (alias, canonical) in &catalog.cli_to_analysis {
            lookup.insert(alias.clone(), canonical.clone());
            lookup.insert(alias.replace('-', "_"), canonical.clone());
        }
        lookup
    })
}

fn warp_md_resolve_analysis_name(name: &str) -> Option<String> {
    warp_md_agent_name_lookup().get(name.trim()).cloned()
}

fn warp_md_contract_for_name(name: &str) -> Option<WarpMdAnalysisContract> {
    let canonical = warp_md_resolve_analysis_name(name)?;
    warp_md_agent_contract_catalog_ref()
        .analyses
        .iter()
        .find(|contract| contract.name == canonical)
        .cloned()
}

fn default_run_request_version() -> String {
    WARP_MD_AGENT_SCHEMA_VERSION.to_string()
}

fn default_stream_mode() -> StreamMode {
    StreamMode::None
}

fn default_output_dir() -> String {
    ".".to_string()
}

fn default_device_auto() -> String {
    "auto".to_string()
}

fn default_true_bool() -> bool {
    true
}

fn default_checkpoint_interval_frames() -> std::num::NonZeroU64 {
    std::num::NonZeroU64::new(1000).expect("non-zero checkpoint interval")
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
enum IoSpec {
    Path(String),
    Spec(std::collections::BTreeMap<String, serde_json::Value>),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum StreamMode {
    None,
    Ndjson,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct CheckpointConfig {
    #[serde(default)]
    enabled: bool,
    #[serde(default = "default_checkpoint_interval_frames")]
    #[schemars(default = "default_checkpoint_interval_frames")]
    interval_frames: std::num::NonZeroU64,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct AnalysisRequest {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    out: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    device: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chunk_frames: Option<std::num::NonZeroU64>,
    #[serde(flatten)]
    extra: std::collections::BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct RunRequest {
    #[serde(default = "default_run_request_version")]
    #[schemars(default = "default_run_request_version")]
    version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    system: Option<IoSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    topology: Option<IoSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    trajectory: Option<IoSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    traj: Option<IoSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    inputs: Option<std::collections::BTreeMap<String, IoSpec>>,
    #[serde(default = "default_device_auto")]
    #[schemars(default = "default_device_auto")]
    device: String,
    #[serde(default = "default_stream_mode")]
    #[schemars(default = "default_stream_mode")]
    stream: StreamMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    chunk_frames: Option<std::num::NonZeroU64>,
    #[serde(default = "default_output_dir")]
    #[schemars(default = "default_output_dir")]
    output_dir: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    checkpoint: Option<CheckpointConfig>,
    #[serde(default = "default_true_bool")]
    #[schemars(default = "default_true_bool")]
    fail_fast: bool,
    analyses: Vec<AnalysisRequest>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct ArtifactMetadata {
    path: String,
    format: String,
    bytes: u64,
    sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fields: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    units: Option<std::collections::BTreeMap<String, String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    preview_stats: Option<std::collections::BTreeMap<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    plot_recommendations: Option<Vec<PlotRecommendation>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    companions: Option<Vec<ArtifactCompanionMetadata>>,
    #[serde(flatten)]
    extra: std::collections::BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct ArtifactCompanionMetadata {
    path: String,
    format: String,
    role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_key: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    columns: Vec<String>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct PlotArtifact {
    path: String,
    format: String,
    #[serde(default = "default_plot_role")]
    #[schemars(default = "default_plot_role")]
    role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    plot_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    source_artifact: Option<String>,
    #[serde(flatten)]
    extra: std::collections::BTreeMap<String, serde_json::Value>,
}

fn default_plot_role() -> String {
    "plot".into()
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct PlotSkippedItem {
    reason: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    analysis: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    plot_type: Option<String>,
    #[serde(flatten)]
    extra: std::collections::BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum PlotManifestStatus {
    Ok,
    Error,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct PlotManifest {
    #[serde(default = "default_result_schema_version")]
    #[schemars(default = "default_result_schema_version")]
    schema_version: String,
    #[serde(default = "default_plot_manifest_status")]
    #[schemars(default = "default_plot_manifest_status")]
    status: PlotManifestStatus,
    plot_count: u64,
    #[serde(default)]
    artifacts: Vec<PlotArtifact>,
    #[serde(default)]
    skipped: Vec<PlotSkippedItem>,
    #[serde(flatten)]
    extra: std::collections::BTreeMap<String, serde_json::Value>,
}

fn default_result_schema_version() -> String {
    WARP_MD_AGENT_SCHEMA_VERSION.to_string()
}

fn default_plot_manifest_status() -> PlotManifestStatus {
    PlotManifestStatus::Ok
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum RunResultStatus {
    Ok,
    DryRun,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct RunResultEntry {
    analysis: String,
    out: String,
    status: RunResultStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    artifact: Option<ArtifactMetadata>,
    #[serde(flatten)]
    extra: std::collections::BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
enum ErrorCode {
    EConfigLoad,
    EConfigValidation,
    EConfigVersion,
    EConfigMissingField,
    EAnalysisUnknown,
    EAnalysisSpec,
    ESelectionEmpty,
    ESelectionInvalid,
    ESystemLoad,
    ETrajectoryLoad,
    ETrajectoryEof,
    ERuntimeExec,
    EOutputDir,
    EOutputWrite,
    EDeviceUnavailable,
    EAtlasFetch,
    EInputMissing,
    EUnsupportedFormat,
    ETopologyTrajectoryMismatch,
    ETopologyAtomMissing,
    ENoFrames,
    EExternalTableLoad,
    EExternalTableColumn,
    EPlotRender,
    EBundlePartial,
    EInternal,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct RunErrorPayload {
    code: ErrorCode,
    message: String,
    #[serde(default)]
    context: std::collections::BTreeMap<String, serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    details: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    traceback: Option<String>,
    #[serde(flatten)]
    extra: std::collections::BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum SuccessStatus {
    Ok,
    DryRun,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum ErrorStatus {
    Error,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct RunSuccessEnvelope {
    schema_version: String,
    status: SuccessStatus,
    exit_code: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    output_dir: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    system: Option<std::collections::BTreeMap<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    trajectory: Option<std::collections::BTreeMap<String, serde_json::Value>>,
    analysis_count: u64,
    started_at: String,
    finished_at: String,
    elapsed_ms: u64,
    #[serde(default)]
    warnings: Vec<String>,
    #[serde(default)]
    results: Vec<RunResultEntry>,
    #[serde(flatten)]
    extra: std::collections::BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct RunErrorEnvelope {
    schema_version: String,
    status: ErrorStatus,
    exit_code: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    output_dir: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    system: Option<std::collections::BTreeMap<String, serde_json::Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    trajectory: Option<std::collections::BTreeMap<String, serde_json::Value>>,
    analysis_count: u64,
    started_at: String,
    finished_at: String,
    elapsed_ms: u64,
    #[serde(default)]
    warnings: Vec<String>,
    #[serde(default)]
    results: Vec<RunResultEntry>,
    error: RunErrorPayload,
    #[serde(flatten)]
    extra: std::collections::BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
enum RunEnvelope {
    Success(RunSuccessEnvelope),
    Error(RunErrorEnvelope),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum RunStartedEventKind {
    RunStarted,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum AnalysisStartedEventKind {
    AnalysisStarted,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum CheckpointEventKind {
    Checkpoint,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum AnalysisCompletedEventKind {
    AnalysisCompleted,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum AnalysisFailedEventKind {
    AnalysisFailed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum RunCompletedEventKind {
    RunCompleted,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum RunFailedEventKind {
    RunFailed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct RunStartedEvent {
    event: RunStartedEventKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    run_id: Option<String>,
    config_path: String,
    dry_run: bool,
    analysis_count: u64,
    completed: u64,
    total: u64,
    progress_pct: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eta_ms: Option<u64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct AnalysisStartedEvent {
    event: AnalysisStartedEventKind,
    index: u64,
    analysis: String,
    out: String,
    completed: u64,
    total: u64,
    progress_pct: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eta_ms: Option<u64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct CheckpointEvent {
    event: CheckpointEventKind,
    analysis_index: u64,
    analysis_name: String,
    frames_processed: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frames_total: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    progress_pct: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eta_ms: Option<u64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct AnalysisCompletedEvent {
    event: AnalysisCompletedEventKind,
    index: u64,
    analysis: String,
    status: RunResultStatus,
    out: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    timing_ms: Option<u64>,
    completed: u64,
    total: u64,
    progress_pct: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eta_ms: Option<u64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct AnalysisFailedEvent {
    event: AnalysisFailedEventKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    index: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    analysis: Option<String>,
    error: RunErrorPayload,
    completed: u64,
    total: u64,
    progress_pct: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    eta_ms: Option<u64>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct RunCompletedEvent {
    event: RunCompletedEventKind,
    final_envelope: RunSuccessEnvelope,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct RunFailedEvent {
    event: RunFailedEventKind,
    final_envelope: RunErrorEnvelope,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
#[serde(untagged)]
enum RunEventSchema {
    RunStarted(RunStartedEvent),
    AnalysisStarted(AnalysisStartedEvent),
    Checkpoint(CheckpointEvent),
    AnalysisCompleted(AnalysisCompletedEvent),
    AnalysisFailed(AnalysisFailedEvent),
    RunCompleted(RunCompletedEvent),
    RunFailed(RunFailedEvent),
}

fn warp_md_schema_value<T: schemars::JsonSchema>() -> Result<serde_json::Value, String> {
    serde_json::to_value(&schema_for!(T)).map_err(|e| e.to_string())
}

fn warp_md_schema_rewrite(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(definitions) = map.remove("definitions") {
                map.insert("$defs".into(), definitions);
            }
            for (key, child) in map.iter_mut() {
                if key == "$ref" {
                    if let Some(reference) = child.as_str() {
                        if let Some(suffix) = reference.strip_prefix("#/definitions/") {
                            *child = serde_json::Value::String(format!("#/$defs/{suffix}"));
                        }
                    }
                    continue;
                }
                warp_md_schema_rewrite(child);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                warp_md_schema_rewrite(item);
            }
        }
        _ => {}
    }
}

fn warp_md_request_schema_value() -> Result<serde_json::Value, String> {
    let mut value = warp_md_schema_value::<RunRequest>()?;
    warp_md_schema_rewrite(&mut value);

    let analysis_names: Vec<serde_json::Value> = warp_md_agent_contract_catalog_ref()
        .analyses
        .iter()
        .map(|contract| serde_json::Value::String(contract.name.clone()))
        .collect();

    if let Some(analysis_name_schema) = value
        .get_mut("$defs")
        .and_then(serde_json::Value::as_object_mut)
        .and_then(|defs| defs.get_mut("AnalysisRequest"))
        .and_then(serde_json::Value::as_object_mut)
        .and_then(|schema| schema.get_mut("properties"))
        .and_then(serde_json::Value::as_object_mut)
        .and_then(|props| props.get_mut("name"))
        .and_then(serde_json::Value::as_object_mut)
    {
        analysis_name_schema.insert("enum".into(), serde_json::Value::Array(analysis_names));
    }

    if let Some(run_request_schema) = value.as_object_mut() {
        if let Some(properties) = run_request_schema
            .get_mut("properties")
            .and_then(serde_json::Value::as_object_mut)
        {
            if let Some(analyses_schema) = properties
                .get_mut("analyses")
                .and_then(serde_json::Value::as_object_mut)
            {
                analyses_schema.insert("minItems".into(), serde_json::json!(1));
            }
        }
    }

    Ok(value)
}

fn warp_md_agent_schema_value(kind: &str) -> Result<serde_json::Value, String> {
    let target = kind.trim().to_ascii_lowercase();
    if target == "request" {
        return warp_md_request_schema_value();
    }
    let mut value = match target.as_str() {
        "result" => warp_md_schema_value::<RunEnvelope>()?,
        "event" => warp_md_schema_value::<RunEventSchema>()?,
        "plot-manifest" | "plot_manifest" => warp_md_schema_value::<PlotManifest>()?,
        _ => {
            return Err(format!(
                "schema target must be 'request', 'result', 'event', or 'plot-manifest', got '{target}'"
            ))
        }
    };
    warp_md_schema_rewrite(&mut value);
    Ok(value)
}

fn warp_md_catalog_hash() -> String {
    let serialized = serde_json::to_vec(warp_md_agent_contract_catalog_ref())
        .expect("serialize warp-md analysis contract catalog");
    let digest = <sha2::Sha256 as sha2::Digest>::digest(&serialized);
    let hex = format!("{digest:x}");
    hex[..16].to_string()
}

fn warp_md_template_placeholder(
    field_name: &str,
    field_spec: &WarpMdFieldSpec,
) -> serde_json::Value {
    match field_spec.semantic_type.as_str() {
        "selection" | "mask" => serde_json::Value::String(format!("<{field_name}_expression>")),
        "charges" => serde_json::Value::String("by_resname".into()),
        _ => match field_spec.field_type.as_str() {
            "array" => serde_json::json!([]),
            "integer" => field_spec
                .default
                .clone()
                .unwrap_or_else(|| serde_json::json!(0)),
            "float" => field_spec
                .default
                .clone()
                .unwrap_or_else(|| serde_json::json!(0.0)),
            "boolean" => field_spec
                .default
                .clone()
                .unwrap_or_else(|| serde_json::json!(false)),
            _ => serde_json::Value::String(format!("<{field_name}>")),
        },
    }
}

fn warp_md_normalize_request_value(
    mut payload: serde_json::Value,
    strip_unknown: bool,
) -> serde_json::Value {
    let Some(root) = payload.as_object_mut() else {
        return payload;
    };

    if root.contains_key("topology") && !root.contains_key("system") {
        if let Some(value) = root.remove("topology") {
            root.insert("system".into(), value);
        }
    } else {
        root.remove("topology");
    }

    if root.contains_key("traj") && !root.contains_key("trajectory") {
        if let Some(value) = root.remove("traj") {
            root.insert("trajectory".into(), value);
        }
    } else {
        root.remove("traj");
    }

    if strip_unknown {
        let allowed: std::collections::BTreeSet<&str> = WARP_MD_RUN_REQUEST_TOP_LEVEL_FIELDS
            .iter()
            .copied()
            .collect();
        root.retain(|key, _| allowed.contains(key.as_str()));
    }

    let shared_fields: std::collections::BTreeSet<&str> = warp_md_agent_contract_catalog_ref()
        .analysis_shared_fields
        .iter()
        .map(String::as_str)
        .collect();

    if let Some(analyses) = root
        .get_mut("analyses")
        .and_then(serde_json::Value::as_array_mut)
    {
        for analysis in analyses {
            let Some(item) = analysis.as_object_mut() else {
                continue;
            };
            let Some(name_value) = item.get("name").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let Some(contract) = warp_md_contract_for_name(name_value) else {
                continue;
            };
            item.insert(
                "name".into(),
                serde_json::Value::String(contract.name.clone()),
            );

            if strip_unknown {
                let allowed_fields: std::collections::BTreeSet<&str> = contract
                    .fields
                    .keys()
                    .map(String::as_str)
                    .chain(shared_fields.iter().copied())
                    .chain(std::iter::once("name"))
                    .collect();
                item.retain(|key, _| allowed_fields.contains(key.as_str()));
            }

            for (field_name, field_spec) in &contract.fields {
                if !item.contains_key(field_name) {
                    if let Some(default) = &field_spec.default {
                        item.insert(field_name.clone(), default.clone());
                    }
                }
            }
        }
    }

    payload
}

fn warp_md_validation_error(
    code: &str,
    path: &str,
    message: impl Into<String>,
) -> serde_json::Value {
    serde_json::json!({
        "code": code,
        "path": path,
        "message": message.into(),
        "context": {},
    })
}

fn warp_md_validation_result_with_key(
    errors: Vec<serde_json::Value>,
    warnings: Vec<String>,
    normalized_key: &str,
    normalized_payload: Option<serde_json::Value>,
) -> serde_json::Value {
    let mut result = serde_json::json!({
        "schema_version": WARP_MD_AGENT_SCHEMA_VERSION,
        "status": if errors.is_empty() { "ok" } else { "error" },
        "valid": errors.is_empty(),
        "errors": errors,
        "warnings": warnings,
    });
    result
        .as_object_mut()
        .expect("validation result payload is always an object")
        .insert(
            normalized_key.into(),
            normalized_payload.unwrap_or(serde_json::Value::Null),
        );
    result
}

fn warp_md_validation_result(
    errors: Vec<serde_json::Value>,
    warnings: Vec<String>,
    normalized_request: Option<serde_json::Value>,
) -> serde_json::Value {
    warp_md_validation_result_with_key(errors, warnings, "normalized_request", normalized_request)
}

fn warp_md_invalid_validation_result(
    normalized_key: &str,
    path: &str,
    message: impl Into<String>,
) -> serde_json::Value {
    warp_md_validation_result_with_key(
        vec![warp_md_validation_error(
            "E_SCHEMA_VALIDATION",
            path,
            message,
        )],
        Vec::new(),
        normalized_key,
        None,
    )
}

fn warp_md_parse_payload<T>(payload: &serde_json::Value) -> Result<T, serde_json::Value>
where
    T: serde::de::DeserializeOwned,
{
    let payload_json = serde_json::to_string(payload)
        .map_err(|err| warp_md_validation_error("E_SCHEMA_VALIDATION", "root", err.to_string()))?;
    let mut deserializer = serde_json::Deserializer::from_str(&payload_json);
    match serde_path_to_error::deserialize::<_, T>(&mut deserializer) {
        Ok(value) => Ok(value),
        Err(err) => {
            let path = err.path().to_string();
            let path = if path.is_empty() {
                "root"
            } else {
                path.as_str()
            };
            Err(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                path,
                err.inner().to_string(),
            ))
        }
    }
}

fn warp_md_validate_serializable_payload<T>(
    payload: &serde_json::Value,
) -> Result<serde_json::Value, serde_json::Value>
where
    T: serde::de::DeserializeOwned + serde::Serialize,
{
    let typed = warp_md_parse_payload::<T>(payload)?;
    serde_json::to_value(typed)
        .map_err(|err| warp_md_validation_error("E_SCHEMA_VALIDATION", "root", err.to_string()))
}

fn warp_md_coerce_io_spec(
    value: &serde_json::Value,
    label: &str,
) -> Result<serde_json::Value, String> {
    match value {
        serde_json::Value::String(path) => {
            let trimmed = path.trim();
            if trimmed.is_empty() {
                Err(format!("{label} path cannot be empty"))
            } else {
                Ok(serde_json::json!({"path": trimmed}))
            }
        }
        serde_json::Value::Object(map) => {
            if map.is_empty() {
                Err(format!("{label} spec cannot be empty"))
            } else {
                Ok(value.clone())
            }
        }
        _ => Err(format!("{label} must be a path string or object")),
    }
}

fn warp_md_request_system_path(
    root: &serde_json::Map<String, serde_json::Value>,
) -> Option<String> {
    ["system", "topology"]
        .iter()
        .find_map(|key| match root.get(*key) {
            Some(serde_json::Value::String(path)) => {
                let trimmed = path.trim();
                (!trimmed.is_empty()).then(|| trimmed.to_string())
            }
            Some(serde_json::Value::Object(spec)) => spec
                .get("path")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|path| !path.is_empty())
                .map(str::to_string),
            _ => None,
        })
}

fn warp_md_append_selection_validation(
    errors: &mut Vec<serde_json::Value>,
    warnings: &mut Vec<String>,
    path: &str,
    field_type: &str,
    field_value: &serde_json::Value,
    system_path: Option<&str>,
) {
    let Some(expr) = field_value.as_str() else {
        return;
    };

    let lint = warp_md_lint_selection_value(expr, field_type, system_path);
    if lint["valid"].as_bool() != Some(true) {
        let message = lint["error"].as_str().unwrap_or("Selection syntax error");
        errors.push(warp_md_validation_error(
            "E_SELECTION_INVALID",
            path,
            message,
        ));
        return;
    }

    if let Some(lint_warnings) = lint["warnings"].as_array() {
        warnings.extend(
            lint_warnings
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(|warning| format!("{path}: {warning}")),
        );
    }
}

fn warp_md_validate_request_value(
    payload: serde_json::Value,
    strict: bool,
    check_selections: bool,
) -> serde_json::Value {
    let mut errors: Vec<serde_json::Value> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    let Some(root) = payload.as_object() else {
        return warp_md_validation_result(
            vec![warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                "root",
                "request payload must be an object",
            )],
            warnings,
            None,
        );
    };

    if let Err(error) = warp_md_parse_payload::<RunRequest>(&payload) {
        return warp_md_validation_result(vec![error], warnings, None);
    }

    let version = match root.get("version") {
        None => serde_json::Value::String(WARP_MD_AGENT_SCHEMA_VERSION.into()),
        Some(value) if value.as_str() == Some(WARP_MD_AGENT_SCHEMA_VERSION) => value.clone(),
        Some(value) => {
            let version = value
                .as_str()
                .map(str::to_string)
                .unwrap_or_else(|| WARP_MD_AGENT_SCHEMA_VERSION.to_string());
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                "version",
                format!(
                    "unsupported run config version: {version}; expected {WARP_MD_AGENT_SCHEMA_VERSION}"
                ),
            ));
            serde_json::Value::String(version)
        }
    };

    let system_spec = match (root.get("system"), root.get("topology")) {
        (Some(_), Some(_)) => {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                "root",
                "specify only one of `system` or `topology`",
            ));
            None
        }
        (None, None) => {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                "root",
                "one of `system` or `topology` is required",
            ));
            None
        }
        (Some(value), None) => match warp_md_coerce_io_spec(value, "system") {
            Ok(spec) => Some(spec),
            Err(message) => {
                errors.push(warp_md_validation_error(
                    "E_SCHEMA_VALIDATION",
                    "system",
                    message,
                ));
                None
            }
        },
        (None, Some(value)) => match warp_md_coerce_io_spec(value, "topology") {
            Ok(spec) => Some(spec),
            Err(message) => {
                errors.push(warp_md_validation_error(
                    "E_SCHEMA_VALIDATION",
                    "topology",
                    message,
                ));
                None
            }
        },
    };

    let trajectory_spec = match (root.get("trajectory"), root.get("traj")) {
        (Some(_), Some(_)) => {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                "root",
                "specify only one of `trajectory` or `traj`",
            ));
            None
        }
        (None, None) => {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                "root",
                "one of `trajectory` or `traj` is required",
            ));
            None
        }
        (Some(value), None) => match warp_md_coerce_io_spec(value, "trajectory") {
            Ok(spec) => Some(spec),
            Err(message) => {
                errors.push(warp_md_validation_error(
                    "E_SCHEMA_VALIDATION",
                    "trajectory",
                    message,
                ));
                None
            }
        },
        (None, Some(value)) => match warp_md_coerce_io_spec(value, "traj") {
            Ok(spec) => Some(spec),
            Err(message) => {
                errors.push(warp_md_validation_error(
                    "E_SCHEMA_VALIDATION",
                    "traj",
                    message,
                ));
                None
            }
        },
    };

    let mut normalized_root = root.clone();
    normalized_root.insert("version".into(), version);
    if let Some(spec) = system_spec {
        normalized_root.insert("system".into(), spec);
        normalized_root.remove("topology");
    }
    if let Some(spec) = trajectory_spec {
        normalized_root.insert("trajectory".into(), spec);
        normalized_root.remove("traj");
    }

    let analyses_value = root.get("analyses");
    let Some(analyses) = analyses_value.and_then(serde_json::Value::as_array) else {
        errors.push(warp_md_validation_error(
            "E_SCHEMA_VALIDATION",
            "analyses",
            "analyses must be a non-empty array",
        ));
        return warp_md_validation_result(errors, warnings, None);
    };
    if analyses.is_empty() {
        errors.push(warp_md_validation_error(
            "E_SCHEMA_VALIDATION",
            "analyses",
            "analyses must contain at least one item",
        ));
    }

    let shared_fields: std::collections::BTreeSet<&str> = warp_md_agent_contract_catalog_ref()
        .analysis_shared_fields
        .iter()
        .map(String::as_str)
        .collect();
    let mut normalized_analyses = Vec::with_capacity(analyses.len());

    for (idx, analysis) in analyses.iter().enumerate() {
        let path_prefix = format!("analyses[{idx}]");
        let Some(item) = analysis.as_object() else {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                &path_prefix,
                "analysis entry must be an object",
            ));
            normalized_analyses.push(analysis.clone());
            continue;
        };

        let mut normalized_analysis = item.clone();
        let Some(name_value) = item.get("name") else {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                &format!("{path_prefix}.name"),
                "analysis name is required",
            ));
            normalized_analyses.push(serde_json::Value::Object(normalized_analysis));
            continue;
        };
        let Some(name_str) = name_value.as_str() else {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                &format!("{path_prefix}.name"),
                "analysis name must be a string",
            ));
            normalized_analyses.push(serde_json::Value::Object(normalized_analysis));
            continue;
        };
        let trimmed_name = name_str.trim();
        if trimmed_name.is_empty() {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                &format!("{path_prefix}.name"),
                "analysis name cannot be empty",
            ));
            normalized_analyses.push(serde_json::Value::Object(normalized_analysis));
            continue;
        }

        let Some(contract) = warp_md_contract_for_name(trimmed_name) else {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                &format!("{path_prefix}.name"),
                format!("unknown analysis: {trimmed_name}"),
            ));
            normalized_analyses.push(serde_json::Value::Object(normalized_analysis));
            continue;
        };
        normalized_analysis.insert(
            "name".into(),
            serde_json::Value::String(contract.name.clone()),
        );

        for field_name in &contract.required_fields {
            match item.get(field_name) {
                None | Some(serde_json::Value::Null) => errors.push(warp_md_validation_error(
                    "E_REQUIRED_FIELD",
                    &format!("{path_prefix}.{field_name}"),
                    format!("{} requires field '{field_name}'", contract.name),
                )),
                Some(serde_json::Value::String(value)) if value.trim().is_empty() => {
                    errors.push(warp_md_validation_error(
                        "E_REQUIRED_FIELD",
                        &format!("{path_prefix}.{field_name}"),
                        format!("{} requires field '{field_name}'", contract.name),
                    ))
                }
                _ => {}
            }
        }

        for (field_name, field_value) in item {
            if field_name == "name" {
                continue;
            }
            if shared_fields.contains(field_name.as_str()) {
                continue;
            }
            let Some(field_spec) = contract.fields.get(field_name) else {
                if strict {
                    errors.push(warp_md_validation_error(
                        "E_UNKNOWN_FIELD",
                        &format!("{path_prefix}.{field_name}"),
                        format!("Unknown field for {}: {field_name}", contract.name),
                    ));
                }
                continue;
            };

            let type_ok = match field_spec.field_type.as_str() {
                "boolean" => field_value.is_boolean(),
                "integer" => field_value.as_i64().is_some() || field_value.as_u64().is_some(),
                "float" => field_value.is_number(),
                "array" => field_value.is_array(),
                _ => true,
            };
            if !type_ok {
                errors.push(warp_md_validation_error(
                    "E_FIELD_TYPE",
                    &format!("{path_prefix}.{field_name}"),
                    format!(
                        "Expected {}, got {}",
                        field_spec.field_type,
                        warp_md_json_type_name(field_value)
                    ),
                ));
                continue;
            }

            if let Some(number) = field_value.as_f64() {
                if let Some(minimum) = field_spec.minimum {
                    if number < minimum {
                        errors.push(warp_md_validation_error(
                            "E_VALUE_RANGE",
                            &format!("{path_prefix}.{field_name}"),
                            format!("Value {number} below minimum {minimum}"),
                        ));
                    }
                }
                if let Some(maximum) = field_spec.maximum {
                    if number > maximum {
                        errors.push(warp_md_validation_error(
                            "E_VALUE_RANGE",
                            &format!("{path_prefix}.{field_name}"),
                            format!("Value {number} above maximum {maximum}"),
                        ));
                    }
                }
            }

            if let Some(choices) = &field_spec.choices {
                if !choices
                    .iter()
                    .any(|choice| Some(choice.as_str()) == field_value.as_str())
                {
                    errors.push(warp_md_validation_error(
                        "E_INVALID_CHOICE",
                        &format!("{path_prefix}.{field_name}"),
                        format!("Invalid choice: {field_value}. Must be one of {choices:?}"),
                    ));
                }
            }
        }

        normalized_analyses.push(serde_json::Value::Object(normalized_analysis));
    }

    normalized_root.insert(
        "analyses".into(),
        serde_json::Value::Array(normalized_analyses),
    );
    if check_selections {
        let system_path = warp_md_request_system_path(&normalized_root);
        if let Some(analyses) = normalized_root
            .get("analyses")
            .and_then(serde_json::Value::as_array)
        {
            for (idx, analysis) in analyses.iter().enumerate() {
                let Some(item) = analysis.as_object() else {
                    continue;
                };
                let Some(analysis_name) = item.get("name").and_then(serde_json::Value::as_str)
                else {
                    continue;
                };
                let Some(contract) = warp_md_contract_for_name(analysis_name) else {
                    continue;
                };
                for (field_name, field_spec) in &contract.fields {
                    let field_type = match field_spec.semantic_type.as_str() {
                        "selection" | "mask" => field_spec.semantic_type.as_str(),
                        _ => continue,
                    };
                    let Some(field_value) = item.get(field_name) else {
                        continue;
                    };
                    warp_md_append_selection_validation(
                        &mut errors,
                        &mut warnings,
                        &format!("analyses[{idx}].{field_name}"),
                        field_type,
                        field_value,
                        system_path.as_deref(),
                    );
                }
            }
        }
    }
    let normalized_request = if errors.is_empty() {
        Some(warp_md_normalize_request_value(
            serde_json::Value::Object(normalized_root),
            false,
        ))
    } else {
        None
    };

    warp_md_validation_result(errors, warnings, normalized_request)
}

fn warp_md_json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(number)
            if number.as_i64().is_some() || number.as_u64().is_some() =>
        {
            "int"
        }
        serde_json::Value::Number(_) => "float",
        serde_json::Value::String(_) => "str",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn warp_md_lint_result(
    valid: bool,
    expression: &str,
    field_type: &str,
    matched_atoms: Option<usize>,
    total_atoms: Option<usize>,
    error: Option<String>,
    warnings: Vec<String>,
) -> serde_json::Value {
    serde_json::json!({
        "valid": valid,
        "expression": expression,
        "field_type": field_type,
        "matched_atoms": matched_atoms,
        "total_atoms": total_atoms,
        "error": error,
        "warnings": warnings,
    })
}

fn warp_md_load_system_from_path(path: &str) -> Result<System, String> {
    load_system_auto(path, None).map_err(|err| err.to_string())
}

fn warp_md_lint_selection_value(
    expr: &str,
    field_type: &str,
    system_path: Option<&str>,
) -> serde_json::Value {
    if expr.trim().is_empty() {
        return warp_md_lint_result(
            false,
            expr,
            field_type,
            None,
            None,
            Some("Selection expression cannot be empty".into()),
            vec![],
        );
    }

    let expr_stripped = expr.trim();
    if expr_stripped.matches('\'').count() % 2 != 0 {
        return warp_md_lint_result(
            false,
            expr,
            field_type,
            None,
            None,
            Some("Unbalanced single quotes in selection".into()),
            vec![],
        );
    }
    if expr_stripped.matches('"').count() % 2 != 0 {
        return warp_md_lint_result(
            false,
            expr,
            field_type,
            None,
            None,
            Some("Unbalanced double quotes in selection".into()),
            vec![],
        );
    }

    let mut paren_depth = 0i64;
    for (idx, ch) in expr_stripped.chars().enumerate() {
        if ch == '(' {
            paren_depth += 1;
        } else if ch == ')' {
            paren_depth -= 1;
        }
        if paren_depth < 0 {
            return warp_md_lint_result(
                false,
                expr,
                field_type,
                None,
                None,
                Some(format!("Unbalanced parentheses at position {idx}")),
                vec![],
            );
        }
    }
    if paren_depth != 0 {
        return warp_md_lint_result(
            false,
            expr,
            field_type,
            None,
            None,
            Some("Unbalanced parentheses in selection".into()),
            vec![],
        );
    }

    let mut matched_atoms = None;
    let mut total_atoms = None;
    let mut warnings = Vec::new();

    if let Some(path) = system_path {
        match warp_md_load_system_from_path(path) {
            Ok(mut system) => {
                total_atoms = Some(system.n_atoms());
                match system.select(expr_stripped) {
                    Ok(selection) => {
                        matched_atoms = Some(selection.indices.len());
                        if matched_atoms == Some(0) {
                            warnings.push("Selection matched zero atoms".into());
                        }
                    }
                    Err(err) => {
                        return warp_md_lint_result(
                            false,
                            expr,
                            field_type,
                            None,
                            total_atoms,
                            Some(format!("Selection syntax error: {err}")),
                            vec![],
                        );
                    }
                }
            }
            Err(err) => warnings.push(format!("Could not load topology for atom count: {err}")),
        }
    }

    warp_md_lint_result(
        true,
        expr,
        field_type,
        matched_atoms,
        total_atoms,
        None,
        warnings,
    )
}

const WARP_MD_GOAL_PHRASES: &[&str] = &[
    "radius of gyration",
    "secondary structure",
    "mean square displacement",
    "diffusion coefficient",
    "pair distribution",
    "radial distribution",
    "hydrogen bond",
    "hydrogen bonds",
    "free volume",
    "fractional free volume",
    "water shell",
    "solvation shell",
    "count water",
    "count waters",
    "water count",
    "molecular docking",
    "binding pose",
    "ligand contacts",
];

fn warp_md_goal_phrase_candidates(phrase: &str) -> &'static [(&'static str, f64)] {
    match phrase {
        "radius of gyration" => &[("rg", 12.0)],
        "secondary structure" => &[("dssp", 12.0)],
        "mean square displacement" => &[("msd", 12.0)],
        "diffusion coefficient" => &[("diffusion", 11.0), ("msd", 8.0)],
        "pair distribution" | "radial distribution" => &[("rdf", 11.0)],
        "hydrogen bond" | "hydrogen bonds" => &[("hbond", 12.0)],
        "free volume" => &[("free_volume", 12.0), ("bondi_ffv", 9.0)],
        "fractional free volume" => &[("bondi_ffv", 13.0)],
        "water shell" | "solvation shell" => &[("watershell", 12.0)],
        "count water" | "count waters" | "water count" => &[("water_count", 12.0)],
        "molecular docking" | "binding pose" | "ligand contacts" => &[("docking", 12.0)],
        _ => &[],
    }
}

fn warp_md_is_generic_goal_word(word: &str) -> bool {
    matches!(
        word,
        "analysis"
            | "analyze"
            | "around"
            | "calculate"
            | "compute"
            | "count"
            | "determine"
            | "find"
            | "function"
            | "interface"
            | "measure"
            | "over"
            | "series"
            | "show"
            | "time"
            | "track"
            | "want"
    )
}

fn warp_md_is_generic_tag(tag: &str) -> bool {
    matches!(
        tag,
        "dynamic" | "polymer" | "protein" | "solvent" | "spatial" | "structural" | "transport"
    )
}

fn warp_md_is_docking_goal(goal_lower: &str, words: &std::collections::BTreeSet<String>) -> bool {
    ["binding", "dock", "docking", "ligand", "pose", "receptor"]
        .iter()
        .any(|trigger| goal_lower.contains(trigger) || words.contains(*trigger))
}

fn warp_md_expand_goal_words(text: &str) -> std::collections::BTreeSet<String> {
    let mut words = std::collections::BTreeSet::new();
    for word in text
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|word| !word.is_empty())
    {
        words.insert(word.to_string());
        if word.len() > 4 && word.ends_with("ies") {
            words.insert(format!("{}y", &word[..word.len() - 3]));
        } else if word.len() > 4 && word.ends_with('s') {
            words.insert(word[..word.len() - 1].to_string());
        }
    }
    words
}

fn warp_md_tokenize_contract_text(text: &str) -> std::collections::BTreeSet<String> {
    warp_md_expand_goal_words(&text.to_ascii_lowercase())
}

fn warp_md_goal_keywords(word: &str) -> &'static [&'static str] {
    match word {
        "radius" => &["rg"],
        "gyration" => &["rg"],
        "size" => &["rg"],
        "compactness" => &["rg"],
        "rmsd" => &["rmsd"],
        "alignment" => &["rmsd", "dipole_alignment"],
        "structure" => &["rmsd", "dssp", "native_contacts"],
        "secondary" => &["dssp"],
        "backbone" => &["dssp", "rmsd"],
        "dihedral" => &["jcoupling", "tordiff"],
        "torsion" => &["tordiff", "jcoupling"],
        "motion" => &["msd", "diffusion"],
        "displacement" => &["msd"],
        "diffusion" => &["diffusion", "msd"],
        "transport" => &["diffusion", "msd", "conductivity"],
        "charge" => &["dipole_alignment", "conductivity"],
        "dipole" => &["dipole_alignment"],
        "polarization" => &["dipole_alignment"],
        "conductivity" => &["conductivity"],
        "permittivity" => &["dielectric"],
        "dielectric" => &["dielectric"],
        "electrostatic" => &["dielectric", "dipole_alignment", "conductivity"],
        "distribution" => &["rdf", "bond_length_distribution", "bond_angle_distribution"],
        "pair" => &["rdf"],
        "radial" => &["rdf"],
        "correlation" => &["rdf", "ion_pair_correlation", "rotacf"],
        "neighbor" => &["rdf", "native_contacts"],
        "contact" => &["native_contacts", "hbond"],
        "hydrogen" => &["hbond"],
        "water" => &["water_count", "watershell", "rdf"],
        "waters" => &["water_count", "watershell", "rdf"],
        "shell" => &["watershell"],
        "solvation" => &["watershell", "free_volume", "gist"],
        "hydration" => &["watershell"],
        "density" => &["density", "volmap"],
        "profile" => &["density", "rdf"],
        "spatial" => &["water_count", "free_volume", "volmap"],
        "map" => &["volmap", "free_volume", "gist"],
        "grid" => &["volmap", "free_volume", "gist"],
        "free" => &["free_volume", "bondi_ffv"],
        "volume" => &["free_volume", "bondi_ffv", "volmap"],
        "void" => &["free_volume", "bondi_ffv"],
        "ffv" => &["bondi_ffv"],
        "bondi" => &["bondi_ffv"],
        "energy" => &["equipartition", "gist"],
        "kinetic" => &["equipartition"],
        "temperature" => &["conductivity", "dielectric", "equipartition"],
        "docking" => &["docking"],
        "dock" => &["docking"],
        "binding" => &["docking"],
        "pose" => &["docking"],
        "ligand" => &["docking"],
        "receptor" => &["docking"],
        "cluster" => &["ion_pair_correlation"],
        "protein" => &["dssp", "rmsd", "rg", "native_contacts"],
        "polymer" => &[
            "free_volume",
            "bondi_ffv",
            "rg",
            "chain_rg",
            "contour_length",
            "end_to_end",
            "persistence_length",
        ],
        _ => &[],
    }
}

fn warp_md_suggest_analyses_value(
    goal: &str,
    provided_fields: &[String],
    top_n: usize,
) -> serde_json::Value {
    let provided: std::collections::BTreeSet<&str> =
        provided_fields.iter().map(String::as_str).collect();
    let goal_lower = goal.to_ascii_lowercase();
    let words = warp_md_expand_goal_words(&goal_lower);

    let mut scored: Vec<(String, f64, String)> = Vec::new();

    for contract in &warp_md_agent_contract_catalog_ref().analyses {
        if contract.name == "docking" && !warp_md_is_docking_goal(&goal_lower, &words) {
            continue;
        }

        let mut score = 0.0;
        let mut reasons: Vec<String> = Vec::new();

        for phrase in WARP_MD_GOAL_PHRASES {
            if goal_lower.contains(phrase) {
                for (candidate, weight) in warp_md_goal_phrase_candidates(phrase) {
                    if *candidate == contract.name
                        || contract.aliases.iter().any(|alias| alias == candidate)
                    {
                        score += *weight;
                        reasons.push(format!("phrase match: {phrase}"));
                    }
                }
            }
        }

        if goal_lower.contains(&contract.name) {
            score += 10.0;
            reasons.push(format!("name match: {}", contract.name));
        }
        for alias in &contract.aliases {
            if goal_lower.contains(alias) || goal_lower.contains(&alias.replace('-', "_")) {
                score += 8.0;
                reasons.push(format!("alias match: {alias}"));
            }
        }
        for tag in &contract.tags {
            if goal_lower.contains(tag) {
                score += if warp_md_is_generic_tag(tag) {
                    1.0
                } else {
                    3.0
                };
                reasons.push(format!("tag match: {tag}"));
            }
        }

        let desc_terms = warp_md_tokenize_contract_text(&contract.description);
        let mut matched_desc_terms = 0usize;
        for word in &words {
            if word.len() > 3 && !warp_md_is_generic_goal_word(word) && desc_terms.contains(word) {
                score += 1.5;
                matched_desc_terms += 1;
            }
        }
        if matched_desc_terms > 0 {
            reasons.push("description match".into());
        }

        for word in &words {
            for candidate in warp_md_goal_keywords(word) {
                if *candidate == contract.name
                    || contract.aliases.iter().any(|alias| alias == candidate)
                {
                    score += 4.0;
                    reasons.push(format!("keyword match: {word}"));
                }
            }
        }

        for field_name in contract.fields.keys() {
            if goal_lower.contains(field_name) {
                score += 2.0;
                reasons.push(format!("field match: {field_name}"));
            }
        }

        for output in &contract.outputs {
            if goal_lower.contains(&output.kind) {
                score += 2.0;
                reasons.push(format!("output kind: {}", output.kind));
            }
        }

        if score > 0.0 {
            let mut dedup = Vec::new();
            for reason in reasons {
                if !dedup.contains(&reason) {
                    dedup.push(reason);
                }
            }
            scored.push((contract.name.clone(), score, dedup.join(", ")));
        }
    }

    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let candidates: Vec<serde_json::Value> = scored
        .into_iter()
        .take(top_n)
        .filter_map(|(name, score, reason)| {
            let contract = warp_md_contract_for_name(&name)?;
            let missing_fields: Vec<String> = contract
                .required_fields
                .iter()
                .filter(|field| !provided.contains(field.as_str()))
                .cloned()
                .collect();
            Some(serde_json::json!({
                "name": name,
                "reason": reason,
                "missing_fields": missing_fields,
                "score": score,
            }))
        })
        .collect();

    serde_json::json!({
        "goal": goal,
        "total_analyses": warp_md_agent_contract_catalog_ref().analyses.len(),
        "candidates": candidates,
    })
}

pub fn agent_contract_catalog_value() -> Result<serde_json::Value, String> {
    serde_json::to_value(warp_md_agent_contract_catalog_ref()).map_err(|err| err.to_string())
}

pub fn agent_plan_schema_value(name: &str) -> Result<serde_json::Value, String> {
    let contract =
        warp_md_contract_for_name(name).ok_or_else(|| format!("unknown plan: {name}"))?;
    serde_json::to_value(contract).map_err(|err| err.to_string())
}

pub fn agent_capabilities_value() -> serde_json::Value {
    let catalog = warp_md_agent_contract_catalog_ref();
    let available_plans: Vec<String> = catalog
        .analyses
        .iter()
        .map(|contract| contract.name.clone())
        .collect();
    serde_json::json!({
        "schema_version": WARP_MD_AGENT_SCHEMA_VERSION,
        "available_plans": available_plans,
        "plan_catalog_hash": warp_md_catalog_hash(),
        "analysis_bundles": WARP_MD_ANALYSIS_BUNDLES,
        "error_codes": WARP_MD_ERROR_CODES,
        "supports_streaming": true,
        "supports_selection_linting": true,
    })
}

pub fn agent_generate_template_value(
    analysis_name: &str,
    fill_defaults: bool,
) -> Result<serde_json::Value, String> {
    let contract = warp_md_contract_for_name(analysis_name)
        .ok_or_else(|| format!("unknown analysis: {analysis_name}"))?;
    let mut analysis = serde_json::Map::new();
    analysis.insert(
        "name".into(),
        serde_json::Value::String(contract.name.clone()),
    );

    for field_name in &contract.required_fields {
        if let Some(field_spec) = contract.fields.get(field_name) {
            analysis.insert(
                field_name.clone(),
                warp_md_template_placeholder(field_name, field_spec),
            );
        }
    }

    if fill_defaults {
        for field_name in &contract.optional_fields {
            if let Some(field_spec) = contract.fields.get(field_name) {
                if let Some(default) = &field_spec.default {
                    analysis.insert(field_name.clone(), default.clone());
                }
            }
        }
    }

    Ok(serde_json::json!({
        "version": WARP_MD_AGENT_SCHEMA_VERSION,
        "system": {"path": "<topology-path>"},
        "trajectory": {"path": "<trajectory-path>"},
        "analyses": [serde_json::Value::Object(analysis)],
    }))
}

pub fn agent_normalize_request_json(
    json: &str,
    strip_unknown: bool,
) -> Result<serde_json::Value, String> {
    let payload: serde_json::Value = serde_json::from_str(json).map_err(|err| err.to_string())?;
    Ok(warp_md_normalize_request_value(payload, strip_unknown))
}

pub fn agent_validate_request_json(
    json: &str,
    strict: bool,
    check_selections: bool,
) -> Result<serde_json::Value, String> {
    let payload: serde_json::Value = serde_json::from_str(json).map_err(|err| err.to_string())?;
    Ok(warp_md_validate_request_value(
        payload,
        strict,
        check_selections,
    ))
}

pub fn agent_schema_value(kind: &str) -> Result<serde_json::Value, String> {
    warp_md_agent_schema_value(kind)
}

pub fn agent_validate_result_json(json: &str) -> Result<serde_json::Value, String> {
    let payload: serde_json::Value = serde_json::from_str(json).map_err(|err| err.to_string())?;
    Ok(warp_md_validate_result_value(payload))
}

pub fn agent_validate_event_json(json: &str) -> Result<serde_json::Value, String> {
    let payload: serde_json::Value = serde_json::from_str(json).map_err(|err| err.to_string())?;
    Ok(warp_md_validate_event_value(payload))
}

pub fn agent_lint_selection_value(
    expr: &str,
    field_type: &str,
    system_path: Option<&str>,
) -> serde_json::Value {
    warp_md_lint_selection_value(expr, field_type, system_path)
}

pub fn agent_suggest_analyses_value(
    goal: &str,
    provided_fields: &[String],
    top_n: usize,
) -> serde_json::Value {
    warp_md_suggest_analyses_value(goal, provided_fields, top_n)
}

fn warp_md_validate_result_value(payload: serde_json::Value) -> serde_json::Value {
    let Some(root) = payload.as_object() else {
        return warp_md_invalid_validation_result(
            "normalized_result",
            "root",
            "run result payload must be an object",
        );
    };

    let normalized = match root.get("status") {
        None => {
            return warp_md_invalid_validation_result(
                "normalized_result",
                "status",
                "run result payload missing required `status` field",
            );
        }
        Some(serde_json::Value::String(status)) => match status.as_str() {
            "ok" | "dry_run" => {
                warp_md_validate_serializable_payload::<RunSuccessEnvelope>(&payload)
            }
            "error" => warp_md_validate_serializable_payload::<RunErrorEnvelope>(&payload),
            _ => {
                return warp_md_invalid_validation_result(
                    "normalized_result",
                    "status",
                    format!(
                        "unsupported run result status: {status}; expected one of ok, dry_run, error"
                    ),
                );
            }
        },
        Some(_) => {
            return warp_md_invalid_validation_result(
                "normalized_result",
                "status",
                "run result `status` must be a string",
            );
        }
    };

    match normalized {
        Ok(value) => warp_md_validation_result_with_key(
            Vec::new(),
            Vec::new(),
            "normalized_result",
            Some(value),
        ),
        Err(error) => {
            warp_md_validation_result_with_key(vec![error], Vec::new(), "normalized_result", None)
        }
    }
}

fn warp_md_validate_event_value(payload: serde_json::Value) -> serde_json::Value {
    let Some(root) = payload.as_object() else {
        return warp_md_invalid_validation_result(
            "normalized_event",
            "root",
            "run event payload must be an object",
        );
    };

    let normalized = match root.get("event") {
        None => {
            return warp_md_invalid_validation_result(
                "normalized_event",
                "event",
                "run event payload missing required `event` field",
            );
        }
        Some(serde_json::Value::String(event)) => match event.as_str() {
            "run_started" => warp_md_validate_serializable_payload::<RunStartedEvent>(&payload),
            "analysis_started" => {
                warp_md_validate_serializable_payload::<AnalysisStartedEvent>(&payload)
            }
            "checkpoint" => warp_md_validate_serializable_payload::<CheckpointEvent>(&payload),
            "analysis_completed" => {
                warp_md_validate_serializable_payload::<AnalysisCompletedEvent>(&payload)
            }
            "analysis_failed" => {
                warp_md_validate_serializable_payload::<AnalysisFailedEvent>(&payload)
            }
            "run_completed" => warp_md_validate_serializable_payload::<RunCompletedEvent>(&payload),
            "run_failed" => warp_md_validate_serializable_payload::<RunFailedEvent>(&payload),
            _ => {
                return warp_md_invalid_validation_result(
                    "normalized_event",
                    "event",
                    format!(
                        "unsupported run event: {event}; expected one of run_started, analysis_started, checkpoint, analysis_completed, analysis_failed, run_completed, run_failed"
                    ),
                );
            }
        },
        Some(_) => {
            return warp_md_invalid_validation_result(
                "normalized_event",
                "event",
                "run event `event` must be a string",
            );
        }
    };

    match normalized {
        Ok(value) => warp_md_validation_result_with_key(
            Vec::new(),
            Vec::new(),
            "normalized_event",
            Some(value),
        ),
        Err(error) => {
            warp_md_validation_result_with_key(vec![error], Vec::new(), "normalized_event", None)
        }
    }
}

#[derive(Clone, Debug)]
struct PlotSeries {
    x: Vec<f64>,
    y: Vec<f64>,
}

fn warp_md_render_plots_value(
    payload: serde_json::Value,
    out_dir: Option<&str>,
) -> Result<serde_json::Value, String> {
    let root = payload
        .as_object()
        .ok_or_else(|| "result envelope must be a JSON object".to_string())?;
    let base_dir = std::path::PathBuf::from(out_dir.unwrap_or("plots"));
    std::fs::create_dir_all(&base_dir)
        .map_err(|err| format!("failed to create plot output directory: {err}"))?;

    let mut artifacts = Vec::new();
    let mut skipped = Vec::new();
    let Some(results) = root.get("results").and_then(serde_json::Value::as_array) else {
        return Ok(serde_json::json!({
            "status": "ok",
            "plot_count": 0,
            "artifacts": artifacts,
            "skipped": [{"reason": "result envelope has no results array"}],
        }));
    };

    for (result_index, result) in results.iter().enumerate() {
        let analysis = result
            .get("analysis")
            .and_then(serde_json::Value::as_str)
            .unwrap_or("analysis");
        let Some(artifact) = result
            .get("artifact")
            .and_then(serde_json::Value::as_object)
        else {
            continue;
        };
        let source_artifact = artifact
            .get("path")
            .and_then(serde_json::Value::as_str)
            .or_else(|| result.get("out").and_then(serde_json::Value::as_str))
            .unwrap_or("");
        let Some(recommendations) = artifact
            .get("plot_recommendations")
            .and_then(serde_json::Value::as_array)
        else {
            skipped.push(serde_json::json!({
                "analysis": analysis,
                "reason": "artifact has no plot_recommendations",
            }));
            continue;
        };
        for (plot_index, rec) in recommendations.iter().enumerate() {
            let plot_type = rec
                .get("plot_type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("line");
            if plot_type != "line" && plot_type != "bar" {
                skipped.push(serde_json::json!({
                    "analysis": analysis,
                    "plot_type": plot_type,
                    "reason": "plot_type is not supported by the SVG renderer",
                }));
                continue;
            }
            let title = rec
                .get("title")
                .and_then(serde_json::Value::as_str)
                .unwrap_or(analysis);
            match warp_md_series_from_recommendation(artifact, rec) {
                Ok(series) if !series.y.is_empty() => {
                    let filename = format!(
                        "{}_{}_{}.svg",
                        warp_md_slug(analysis),
                        result_index,
                        plot_index
                    );
                    let path = base_dir.join(filename);
                    let svg = warp_md_svg_plot(title, plot_type, &series);
                    std::fs::write(&path, svg)
                        .map_err(|err| format!("failed to write plot SVG: {err}"))?;
                    artifacts.push(serde_json::json!({
                        "path": path.to_string_lossy(),
                        "format": "svg",
                        "role": "plot",
                        "plot_type": plot_type,
                        "title": title,
                        "source_artifact": source_artifact,
                    }));
                }
                Ok(_) => skipped.push(serde_json::json!({
                    "analysis": analysis,
                    "plot_type": plot_type,
                    "reason": "plot series is empty",
                })),
                Err(reason) => skipped.push(serde_json::json!({
                    "analysis": analysis,
                    "plot_type": plot_type,
                    "reason": reason,
                })),
            }
        }
    }

    Ok(serde_json::json!({
        "status": "ok",
        "plot_count": artifacts.len(),
        "artifacts": artifacts,
        "skipped": skipped,
    }))
}

fn warp_md_series_from_recommendation(
    artifact: &serde_json::Map<String, serde_json::Value>,
    rec: &serde_json::Value,
) -> Result<PlotSeries, String> {
    let y_field = rec
        .get("y")
        .and_then(|axis| axis.get("field"))
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "plot recommendation has no y.field".to_string())?;
    let mut y = warp_md_read_companion_series(artifact, y_field)
        .ok_or_else(|| format!("no CSV companion found for y field `{y_field}`"))?;
    let x_axis = rec.get("x");
    let x_field = x_axis
        .and_then(|axis| axis.get("field"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("index");
    let x_source = x_axis
        .and_then(|axis| axis.get("source"))
        .and_then(serde_json::Value::as_str);
    let mut x = if x_source == Some("implicit_index") || x_field == "index" {
        (0..y.len()).map(|value| value as f64).collect()
    } else {
        warp_md_read_companion_series(artifact, x_field)
            .unwrap_or_else(|| (0..y.len()).map(|value| value as f64).collect())
    };
    if x.len() != y.len() {
        let n = x.len().min(y.len());
        x.truncate(n);
        y.truncate(n);
    }
    Ok(PlotSeries { x, y })
}

fn warp_md_read_companion_series(
    artifact: &serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Option<Vec<f64>> {
    let companions = artifact.get("companions")?.as_array()?;
    for companion in companions {
        let item = companion.as_object()?;
        if item.get("format").and_then(serde_json::Value::as_str) != Some("csv") {
            continue;
        }
        let source_key = item.get("source_key").and_then(serde_json::Value::as_str);
        let columns = item.get("columns").and_then(serde_json::Value::as_array);
        let matched_column_index = columns.and_then(|values| {
            values
                .iter()
                .position(|value| value.as_str().is_some_and(|name| name == field))
        });
        let selected_column =
            matched_column_index.or_else(|| (source_key == Some(field)).then_some(0));
        let path = item.get("path").and_then(serde_json::Value::as_str)?;
        let Some(column_index) = selected_column else {
            continue;
        };
        if let Ok(values) =
            warp_md_read_csv_numeric_column(std::path::Path::new(path), column_index)
        {
            return Some(values);
        }
    }
    None
}

fn warp_md_read_csv_numeric_column(
    path: &std::path::Path,
    column_index: usize,
) -> Result<Vec<f64>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("failed to read CSV companion {}: {err}", path.display()))?;
    let mut values = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let Some(cell) = trimmed.split(',').nth(column_index) else {
            continue;
        };
        match cell.trim().parse::<f64>() {
            Ok(value) => values.push(value),
            Err(_) => continue,
        }
    }
    Ok(values)
}

fn warp_md_svg_plot(title: &str, plot_type: &str, series: &PlotSeries) -> String {
    let width = 800.0;
    let height = 480.0;
    let left = 64.0;
    let right = 24.0;
    let top = 48.0;
    let bottom = 56.0;
    let plot_w = width - left - right;
    let plot_h = height - top - bottom;
    let (x_min, x_max) = warp_md_extent(&series.x);
    let (y_min, y_max) = warp_md_extent(&series.y);
    let sx = |x: f64| left + warp_md_normalize(x, x_min, x_max) * plot_w;
    let sy = |y: f64| top + (1.0 - warp_md_normalize(y, y_min, y_max)) * plot_h;
    let mut svg = String::new();
    let _ = write!(
        svg,
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0}" height="{height:.0}" viewBox="0 0 {width:.0} {height:.0}">"#
    );
    svg.push_str(r##"<rect width="100%" height="100%" fill="#ffffff"/>"##);
    let _ = write!(
        svg,
        r##"<text x="{left:.1}" y="28" font-family="sans-serif" font-size="18" fill="#111111">{}</text>"##,
        warp_md_xml_escape(title)
    );
    let _ = write!(
        svg,
        r##"<line x1="{left:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="#222222" stroke-width="1"/>"##,
        top + plot_h,
        left + plot_w,
        top + plot_h
    );
    let _ = write!(
        svg,
        r##"<line x1="{left:.1}" y1="{top:.1}" x2="{left:.1}" y2="{:.1}" stroke="#222222" stroke-width="1"/>"##,
        top + plot_h
    );
    for i in 0..=4 {
        let frac = i as f64 / 4.0;
        let x = left + frac * plot_w;
        let y = top + plot_h - frac * plot_h;
        let xv = x_min + frac * (x_max - x_min);
        let yv = y_min + frac * (y_max - y_min);
        let _ = write!(
            svg,
            r##"<line x1="{x:.1}" y1="{top:.1}" x2="{x:.1}" y2="{:.1}" stroke="#dddddd" stroke-width="1"/>"##,
            top + plot_h
        );
        let _ = write!(
            svg,
            r##"<line x1="{left:.1}" y1="{y:.1}" x2="{:.1}" y2="{y:.1}" stroke="#eeeeee" stroke-width="1"/>"##,
            left + plot_w
        );
        let _ = write!(
            svg,
            r##"<text x="{x:.1}" y="{:.1}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#444444">{xv:.3}</text>"##,
            top + plot_h + 18.0
        );
        let _ = write!(
            svg,
            r##"<text x="{:.1}" y="{:.1}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#444444">{yv:.3}</text>"##,
            left - 8.0,
            y + 4.0
        );
    }
    if plot_type == "bar" {
        let count = series.y.len().max(1) as f64;
        let bar_w = (plot_w / count).max(1.0) * 0.8;
        let baseline = sy(0.0_f64.max(y_min).min(y_max));
        for (index, yv) in series.y.iter().copied().enumerate() {
            let cx = left + (index as f64 + 0.5) * plot_w / count;
            let y = sy(yv);
            let h = (baseline - y).abs().max(1.0);
            let y0 = y.min(baseline);
            let _ = write!(
                svg,
                r##"<rect x="{:.1}" y="{y0:.1}" width="{bar_w:.1}" height="{h:.1}" fill="#2f6f9f"/>"##,
                cx - bar_w / 2.0
            );
        }
    } else if !series.x.is_empty() {
        let points = series
            .x
            .iter()
            .copied()
            .zip(series.y.iter().copied())
            .map(|(x, y)| format!("{:.2},{:.2}", sx(x), sy(y)))
            .collect::<Vec<_>>()
            .join(" ");
        let _ = write!(
            svg,
            r##"<polyline points="{points}" fill="none" stroke="#2f6f9f" stroke-width="2"/>"##
        );
    }
    svg.push_str("</svg>\n");
    svg
}

fn warp_md_extent(values: &[f64]) -> (f64, f64) {
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for value in values.iter().copied().filter(|value| value.is_finite()) {
        min_v = min_v.min(value);
        max_v = max_v.max(value);
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        return (0.0, 1.0);
    }
    if (max_v - min_v).abs() < f64::EPSILON {
        return (min_v - 0.5, max_v + 0.5);
    }
    (min_v, max_v)
}

fn warp_md_normalize(value: f64, min_v: f64, max_v: f64) -> f64 {
    if (max_v - min_v).abs() < f64::EPSILON {
        0.5
    } else {
        ((value - min_v) / (max_v - min_v)).clamp(0.0, 1.0)
    }
}

fn warp_md_xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn warp_md_slug(value: &str) -> String {
    let slug: String = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = slug.trim_matches('_');
    if trimmed.is_empty() {
        "plot".into()
    } else {
        trimmed.into()
    }
}

#[pyfunction]
fn warp_md_agent_contract_catalog<'py>(py: Python<'py>) -> PyResult<PyObject> {
    let value = serde_json::to_value(warp_md_agent_contract_catalog_ref())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_value_to_py(py, &value)
}

#[pyfunction]
fn warp_md_agent_plan_schema<'py>(py: Python<'py>, name: &str) -> PyResult<PyObject> {
    let value = agent_plan_schema_value(name).map_err(PyRuntimeError::new_err)?;
    json_value_to_py(py, &value)
}

#[pyfunction]
fn warp_md_agent_capabilities<'py>(py: Python<'py>) -> PyResult<PyObject> {
    let value = agent_capabilities_value();
    json_value_to_py(py, &value)
}

#[pyfunction]
#[pyo3(signature = (analysis_name, fill_defaults=false))]
fn warp_md_agent_generate_template<'py>(
    py: Python<'py>,
    analysis_name: &str,
    fill_defaults: bool,
) -> PyResult<PyObject> {
    let value = agent_generate_template_value(analysis_name, fill_defaults)
        .map_err(PyRuntimeError::new_err)?;
    json_value_to_py(py, &value)
}

#[pyfunction]
#[pyo3(signature = (json, strip_unknown=false))]
fn warp_md_agent_normalize_request<'py>(
    py: Python<'py>,
    json: &str,
    strip_unknown: bool,
) -> PyResult<PyObject> {
    let payload: serde_json::Value =
        serde_json::from_str(json).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let normalized = warp_md_normalize_request_value(payload, strip_unknown);
    json_value_to_py(py, &normalized)
}

#[pyfunction]
#[pyo3(signature = (json, strict=false, check_selections=false))]
fn warp_md_agent_validate_request<'py>(
    py: Python<'py>,
    json: &str,
    strict: bool,
    check_selections: bool,
) -> PyResult<PyObject> {
    let payload: serde_json::Value =
        serde_json::from_str(json).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = warp_md_validate_request_value(payload, strict, check_selections);
    json_value_to_py(py, &result)
}

#[pyfunction]
#[pyo3(signature = (kind="request"))]
fn warp_md_agent_schema<'py>(py: Python<'py>, kind: &str) -> PyResult<PyObject> {
    let value =
        warp_md_agent_schema_value(kind).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_value_to_py(py, &value)
}

#[pyfunction]
fn warp_md_agent_validate_result<'py>(py: Python<'py>, json: &str) -> PyResult<PyObject> {
    let payload: serde_json::Value =
        serde_json::from_str(json).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = warp_md_validate_result_value(payload);
    json_value_to_py(py, &result)
}

#[pyfunction]
fn warp_md_agent_validate_event<'py>(py: Python<'py>, json: &str) -> PyResult<PyObject> {
    let payload: serde_json::Value =
        serde_json::from_str(json).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = warp_md_validate_event_value(payload);
    json_value_to_py(py, &result)
}

#[pyfunction]
#[pyo3(signature = (json, out_dir=None))]
fn warp_md_agent_render_plots<'py>(
    py: Python<'py>,
    json: &str,
    out_dir: Option<&str>,
) -> PyResult<PyObject> {
    let payload: serde_json::Value =
        serde_json::from_str(json).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = warp_md_render_plots_value(payload, out_dir)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_value_to_py(py, &result)
}

#[pyfunction]
#[pyo3(signature = (expr, field_type="selection", system_path=None))]
fn warp_md_agent_lint_selection<'py>(
    py: Python<'py>,
    expr: &str,
    field_type: &str,
    system_path: Option<&str>,
) -> PyResult<PyObject> {
    let result = warp_md_lint_selection_value(expr, field_type, system_path);
    json_value_to_py(py, &result)
}

#[pyfunction]
#[pyo3(signature = (goal, provided_fields=None, top_n=5))]
fn warp_md_agent_suggest_analyses<'py>(
    py: Python<'py>,
    goal: &str,
    provided_fields: Option<Vec<String>>,
    top_n: usize,
) -> PyResult<PyObject> {
    let result = warp_md_suggest_analyses_value(goal, &provided_fields.unwrap_or_default(), top_n);
    json_value_to_py(py, &result)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(warp_md_agent_contract_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_plan_schema, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_generate_template, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_normalize_request, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_validate_request, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_schema, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_validate_result, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_validate_event, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_render_plots, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_lint_selection, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_suggest_analyses, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_alias_to_canonical_analysis_name() {
        assert_eq!(
            warp_md_resolve_analysis_name("dipole-alignment").as_deref(),
            Some("dipole_alignment")
        );
        assert_eq!(
            warp_md_resolve_analysis_name("free-volume-grid").as_deref(),
            Some("free_volume")
        );
    }

    #[test]
    fn normalizes_request_aliases_and_defaults() {
        let payload = serde_json::json!({
            "topology": "top.pdb",
            "traj": "traj.xtc",
            "analyses": [
                {
                    "name": "rdf",
                    "sel_a": "resname SOL",
                    "sel_b": "resname SOL",
                    "r_max": 1.0,
                    "mystery": true
                }
            ]
        });

        let normalized = warp_md_normalize_request_value(payload, true);
        assert_eq!(normalized["system"], "top.pdb");
        assert_eq!(normalized["trajectory"], "traj.xtc");
        assert_eq!(normalized["analyses"][0]["name"], "rdf");
        assert_eq!(normalized["analyses"][0]["bins"], 200);
        assert!(normalized["analyses"][0].get("mystery").is_none());
    }

    #[test]
    fn catalog_embeds_expected_analysis() {
        let contract = warp_md_contract_for_name("bondi-ffv").expect("bondi_ffv contract");
        assert_eq!(contract.name, "bondi_ffv");
        assert!(contract.fields.contains_key("bondi_scale"));
        let value = serde_json::to_value(&contract).expect("contract json");
        assert_eq!(
            value["input_requirements"]["required"],
            serde_json::json!(["topology", "trajectory"])
        );
        assert_eq!(value["input_requirements"]["requires_box"], true);
    }

    #[test]
    fn validates_request_and_reports_range_errors() {
        let payload = serde_json::json!({
            "system": "top.pdb",
            "trajectory": "traj.xtc",
            "analyses": [
                {
                    "name": "rdf",
                    "sel_a": "resname SOL",
                    "sel_b": "resname SOL",
                    "bins": -1,
                    "r_max": 1.0,
                }
            ]
        });
        let result = warp_md_validate_request_value(payload, false, false);
        assert_eq!(result["valid"], false);
        let errors = result["errors"].as_array().expect("errors");
        assert!(errors.iter().any(|error| error["code"] == "E_VALUE_RANGE"));
    }

    #[test]
    fn request_schema_exposes_defs_and_analysis_enum() {
        let schema = warp_md_agent_schema_value("request").expect("request schema");
        assert!(schema.get("$defs").is_some());
        assert_eq!(schema["title"], "RunRequest");
        assert!(schema["properties"].get("inputs").is_some());
        let analysis_name = &schema["$defs"]["AnalysisRequest"]["properties"]["name"]["enum"];
        let values = analysis_name.as_array().expect("analysis enum");
        assert!(values.iter().any(|value| value == "rg"));
        assert!(values.iter().any(|value| value == "docking"));
        assert_eq!(
            schema["$defs"]["StreamMode"]["enum"],
            serde_json::json!(["none", "ndjson"])
        );
    }

    #[test]
    fn plot_manifest_schema_is_available() {
        let schema = warp_md_agent_schema_value("plot-manifest").expect("plot manifest schema");
        assert_eq!(schema["title"], "PlotManifest");
        assert!(schema["properties"].get("artifacts").is_some());
    }

    #[test]
    fn capabilities_expose_analysis_bundles() {
        let value = serde_json::json!({
            "analysis_bundles": WARP_MD_ANALYSIS_BUNDLES,
            "error_codes": WARP_MD_ERROR_CODES,
        });
        let bundles = value["analysis_bundles"].as_array().expect("bundles");
        assert!(bundles
            .iter()
            .any(|item| item["name"] == "standard_md_report"));
        assert!(bundles.iter().any(|item| item["name"] == "polymer_report"));
        assert!(value["error_codes"]
            .as_array()
            .expect("error codes")
            .iter()
            .any(|item| item == "E_EXTERNAL_TABLE_COLUMN"));
    }

    #[test]
    fn request_validator_rejects_invalid_stream_mode() {
        let payload = serde_json::json!({
            "system": "top.pdb",
            "trajectory": "traj.xtc",
            "stream": "stdout",
            "analyses": [{"name": "rg", "selection": "protein"}],
        });
        let result = warp_md_validate_request_value(payload, false, false);
        assert_eq!(result["valid"], false);
        let errors = result["errors"].as_array().expect("errors");
        assert!(errors.iter().any(|error| error["path"] == "stream"));
    }

    #[test]
    fn request_validator_rejects_zero_checkpoint_interval() {
        let payload = serde_json::json!({
            "system": "top.pdb",
            "trajectory": "traj.xtc",
            "checkpoint": {"enabled": true, "interval_frames": 0},
            "analyses": [{"name": "rg", "selection": "protein"}],
        });
        let result = warp_md_validate_request_value(payload, false, false);
        assert_eq!(result["valid"], false);
        let errors = result["errors"].as_array().expect("errors");
        assert!(errors
            .iter()
            .any(|error| error["path"] == "checkpoint.interval_frames"));
    }

    #[test]
    fn request_validator_reports_invalid_selection_when_enabled() {
        let payload = serde_json::json!({
            "system": "top.pdb",
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg", "selection": "("}],
        });
        let result = warp_md_validate_request_value(payload, false, true);
        assert_eq!(result["valid"], false);
        let errors = result["errors"].as_array().expect("errors");
        assert!(errors.iter().any(|error| {
            error["code"] == "E_SELECTION_INVALID" && error["path"] == "analyses[0].selection"
        }));
    }

    #[test]
    fn request_validator_warns_when_topology_cannot_be_loaded() {
        pyo3::prepare_freethreaded_python();
        let payload = serde_json::json!({
            "system": {"path": "missing-topology.pdb"},
            "trajectory": "traj.xtc",
            "analyses": [{"name": "rg", "selection": "protein"}],
        });
        let result = warp_md_validate_request_value(payload, false, true);
        assert_eq!(result["valid"], true);
        let warnings = result["warnings"].as_array().expect("warnings");
        assert!(warnings.iter().any(|warning| {
            warning.as_str().is_some_and(|message| {
                message
                    .starts_with("analyses[0].selection: Could not load topology for atom count:")
            })
        }));
    }

    #[test]
    fn result_validator_preserves_contract_description() {
        let payload = serde_json::json!({
            "schema_version": WARP_MD_AGENT_SCHEMA_VERSION,
            "status": "ok",
            "exit_code": 0,
            "output_dir": ".",
            "system": {"path": "top.pdb"},
            "trajectory": {"path": "traj.xtc"},
            "analysis_count": 1,
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:01Z",
            "elapsed_ms": 1000,
            "warnings": [],
            "results": [
                {
                    "analysis": "rg",
                    "out": "rg.npz",
                    "status": "ok",
                    "kind": "array",
                    "artifact": {
                        "path": "rg.npz",
                        "format": "npz",
                        "bytes": 64,
                        "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "kind": "timeseries",
                        "fields": ["time_ps", "rg_nm"],
                        "description": "Time series of radius of gyration values"
                    }
                }
            ]
        });
        let value = warp_md_validate_result_value(payload);
        assert_eq!(value["valid"], true);
        assert_eq!(
            value["normalized_result"]["results"][0]["artifact"]["description"],
            "Time series of radius of gyration values"
        );
    }

    #[test]
    fn result_validator_reports_structured_status_errors() {
        let payload = serde_json::json!({
            "schema_version": WARP_MD_AGENT_SCHEMA_VERSION,
            "status": "finished",
            "exit_code": 0,
            "analysis_count": 0,
            "started_at": "2026-01-01T00:00:00Z",
            "finished_at": "2026-01-01T00:00:01Z",
            "elapsed_ms": 1000,
            "warnings": [],
            "results": []
        });
        let result = warp_md_validate_result_value(payload);
        assert_eq!(result["valid"], false);
        let errors = result["errors"].as_array().expect("errors");
        assert!(errors.iter().any(|error| error["path"] == "status"));
    }

    #[test]
    fn plot_renderer_reads_requested_csv_companion_column() {
        let dir = std::env::temp_dir().join(format!(
            "warp_md_plot_contract_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp plot dir");
        let csv_path = dir.join("values.csv");
        std::fs::write(&csv_path, "values_0,values_1\n1,10\n2,20\n3,30\n").expect("csv companion");
        let artifact = serde_json::json!({
            "companions": [
                {
                    "path": csv_path.to_string_lossy(),
                    "format": "csv",
                    "role": "array_table",
                    "source_key": "values",
                    "columns": ["values_0", "values_1"]
                }
            ]
        });
        let rec = serde_json::json!({
            "plot_type": "line",
            "x": {"field": "index", "source": "implicit_index"},
            "y": {"field": "values_1"},
            "title": "Selected column"
        });
        let series =
            warp_md_series_from_recommendation(artifact.as_object().expect("artifact"), &rec)
                .expect("plot series");

        assert_eq!(series.x, vec![0.0, 1.0, 2.0]);
        assert_eq!(series.y, vec![10.0, 20.0, 30.0]);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn event_validator_accepts_run_completed_payload() {
        let payload = serde_json::json!({
            "event": "run_completed",
            "final_envelope": {
                "schema_version": WARP_MD_AGENT_SCHEMA_VERSION,
                "status": "ok",
                "exit_code": 0,
                "output_dir": ".",
                "system": {"path": "top.pdb"},
                "trajectory": {"path": "traj.xtc"},
                "analysis_count": 1,
                "started_at": "2026-01-01T00:00:00Z",
                "finished_at": "2026-01-01T00:00:01Z",
                "elapsed_ms": 1000,
                "warnings": [],
                "results": []
            }
        });
        let value = warp_md_validate_event_value(payload);
        assert_eq!(value["valid"], true);
        assert_eq!(value["normalized_event"]["event"], "run_completed");
    }

    #[test]
    fn event_validator_reports_structured_event_errors() {
        let payload = serde_json::json!({"event": "run_done"});
        let result = warp_md_validate_event_value(payload);
        assert_eq!(result["valid"], false);
        let errors = result["errors"].as_array().expect("errors");
        assert!(errors.iter().any(|error| error["path"] == "event"));
    }

    #[test]
    fn suggests_expected_analysis_for_radius_goal() {
        let result = warp_md_suggest_analyses_value("radius of gyration", &[], 5);
        assert_eq!(result["candidates"][0]["name"], "rg");
    }

    #[test]
    fn suggests_water_count_without_docking_false_positive() {
        let result = warp_md_suggest_analyses_value("count waters around a protein", &[], 5);
        assert_eq!(result["candidates"][0]["name"], "water_count");
    }

    #[test]
    fn suggests_hbond_for_hydrogen_bond_goal() {
        let result =
            warp_md_suggest_analyses_value("hydrogen bonds in protein water interface", &[], 5);
        assert_eq!(result["candidates"][0]["name"], "hbond");
    }

    #[test]
    fn suggests_free_volume_for_generic_free_volume_goal() {
        let result = warp_md_suggest_analyses_value("measure free volume in polymer", &[], 5);
        assert_eq!(result["candidates"][0]["name"], "free_volume");
    }
}
