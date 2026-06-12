use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::adapters;
use crate::engines;
use crate::parsers;

pub const QM_SCHEMA_VERSION: &str = "warp-qm.agent.v1";
pub const JOB_MANIFEST_VERSION: &str = "warp-qm.job-manifest.v1";
pub const CHARGE_MANIFEST_VERSION: &str = "warp-qm.charge-manifest.v1";
pub const POLYMER_CHARGE_MANIFEST_VERSION: &str = "warp-qm.polymer-charge-manifest.v1";
pub const ESP_MANIFEST_VERSION: &str = "warp-qm.esp-manifest.v1";
pub const CUBE_MANIFEST_VERSION: &str = "warp-qm.cube-manifest.v1";

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QmRequest {
    #[serde(default = "default_schema_version")]
    #[schemars(default = "default_schema_version")]
    pub schema_version: String,
    #[serde(default)]
    pub request_id: Option<String>,
    pub engine: QmEngineSpec,
    pub molecule: QmMoleculeSpec,
    pub task: QmTaskSpec,
    #[serde(default)]
    pub runtime: QmRuntimeSpec,
    #[serde(default)]
    pub output: QmOutputSpec,
    #[serde(default)]
    pub validation: QmValidationSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QmEngineSpec {
    pub name: String,
    #[serde(default = "default_engine_mode")]
    #[schemars(default = "default_engine_mode")]
    pub mode: String,
    #[serde(default)]
    pub executable: Option<String>,
    #[serde(default = "default_version_policy")]
    #[schemars(default = "default_version_policy")]
    pub version_policy: String,
    #[serde(default)]
    pub settings: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QmMoleculeSpec {
    pub source: MoleculeSource,
    pub charge: i32,
    pub multiplicity: u32,
    #[serde(default = "default_units")]
    #[schemars(default = "default_units")]
    pub units: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MoleculeSource {
    #[serde(default = "default_source_kind")]
    #[schemars(default = "default_source_kind")]
    pub kind: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub topology: Option<String>,
    #[serde(default)]
    pub trajectory: Option<String>,
    #[serde(default)]
    pub selection: Option<String>,
    #[serde(default)]
    pub environment_selection: Option<String>,
    #[serde(default)]
    pub frames: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QmTaskSpec {
    pub kind: String,
    pub method: String,
    #[serde(default)]
    pub basis: Option<String>,
    #[serde(default)]
    pub charge_model: Option<String>,
    #[serde(default)]
    pub properties: Vec<String>,
    #[serde(default)]
    pub convergence: QmConvergenceSpec,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QmConvergenceSpec {
    #[serde(default)]
    pub energy_tol: Option<f64>,
    #[serde(default)]
    pub gradient_tol: Option<f64>,
    #[serde(default)]
    pub max_iterations: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QmRuntimeSpec {
    #[serde(default = "default_work_dir")]
    #[schemars(default = "default_work_dir")]
    pub work_dir: String,
    #[serde(default)]
    pub threads: Option<u32>,
    #[serde(default)]
    pub memory_mb: Option<u64>,
    #[serde(default)]
    pub scratch_dir: Option<String>,
    #[serde(default = "default_keep_raw")]
    #[schemars(default = "default_keep_raw")]
    pub keep_raw: bool,
}

impl Default for QmRuntimeSpec {
    fn default() -> Self {
        Self {
            work_dir: default_work_dir(),
            threads: None,
            memory_mb: None,
            scratch_dir: None,
            keep_raw: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QmOutputSpec {
    #[serde(default = "default_out_dir")]
    #[schemars(default = "default_out_dir")]
    pub out_dir: String,
    #[serde(default = "default_true")]
    #[schemars(default = "default_true")]
    pub write_json: bool,
    #[serde(default)]
    pub write_npz: bool,
    #[serde(default)]
    pub write_xyz: bool,
}

impl Default for QmOutputSpec {
    fn default() -> Self {
        Self {
            out_dir: default_out_dir(),
            write_json: true,
            write_npz: false,
            write_xyz: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct QmValidationSpec {
    #[serde(default = "default_validation_depth")]
    #[schemars(default = "default_validation_depth")]
    pub depth: String,
}

impl Default for QmValidationSpec {
    fn default() -> Self {
        Self {
            depth: default_validation_depth(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct QmArtifact {
    pub path: String,
    pub format: String,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct QmResultEnvelope {
    pub schema_version: String,
    pub status: String,
    pub exit_code: i32,
    pub request_id: Option<String>,
    pub engine: QmEngineSummary,
    pub task: QmTaskSummary,
    pub summary: QmResultSummary,
    pub properties: BTreeMap<String, Value>,
    pub artifacts: Vec<QmArtifact>,
    pub warnings: Vec<String>,
    pub elapsed_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct QmEngineSummary {
    pub name: String,
    pub version: Option<String>,
    pub command: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct QmTaskSummary {
    pub kind: String,
    pub method: String,
    pub basis: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct QmResultSummary {
    pub energy_hartree: Option<f64>,
    pub converged: Option<bool>,
    pub n_atoms: Option<usize>,
    pub wall_time_ms: Option<u128>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "event")]
pub enum QmRunEvent {
    #[serde(rename = "started")]
    Started {
        schema_version: String,
        request_id: Option<String>,
    },
    #[serde(rename = "engine_command")]
    EngineCommand {
        schema_version: String,
        request_id: Option<String>,
        engine: String,
        command: Vec<String>,
    },
    #[serde(rename = "stdout")]
    Stdout {
        schema_version: String,
        request_id: Option<String>,
        line: String,
    },
    #[serde(rename = "stderr")]
    Stderr {
        schema_version: String,
        request_id: Option<String>,
        line: String,
    },
    #[serde(rename = "artifact")]
    Artifact {
        schema_version: String,
        request_id: Option<String>,
        artifact: QmArtifact,
    },
    #[serde(rename = "warning")]
    Warning {
        schema_version: String,
        request_id: Option<String>,
        message: String,
    },
    #[serde(rename = "completed")]
    Completed {
        schema_version: String,
        request_id: Option<String>,
        result: QmResultEnvelope,
    },
    #[serde(rename = "failed")]
    Failed {
        schema_version: String,
        request_id: Option<String>,
        result: QmResultEnvelope,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct JobManifest {
    pub schema_version: String,
    pub request_id: Option<String>,
    pub engine: QmEngineSpec,
    pub task: QmTaskSpec,
    pub molecule: QmMoleculeSpec,
    pub artifacts: Vec<QmArtifact>,
    pub provenance: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChargeManifest {
    pub schema_version: String,
    pub model: String,
    pub charge_unit: String,
    pub total_charge_e: Option<f64>,
    pub atom_charges_e: Vec<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub atom_labels: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection: Option<ChargeProjectionManifest>,
    pub provenance: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChargeProjectionManifest {
    pub policy: String,
    #[serde(default)]
    pub projected_charges_e: Vec<f64>,
    #[serde(default)]
    pub deployable_sets: Vec<ChargeDeployableSet>,
    #[serde(default)]
    pub redistribution: Vec<ChargeRedistributionRule>,
    pub provenance: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChargeDeployableSet {
    pub name: String,
    pub atom_indices: Vec<usize>,
    pub charges_e: Vec<f64>,
    #[serde(default)]
    pub role: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChargeRedistributionRule {
    pub source_atom: usize,
    pub target_atoms: Vec<usize>,
    pub weights: Vec<f64>,
    pub source_charge_e: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PolymerChargeManifest {
    pub schema_version: String,
    pub source_charge_manifest: String,
    pub source_model: String,
    pub repeat_set: String,
    pub repeat_count: usize,
    pub terminal_policy: String,
    pub charge_unit: String,
    pub total_charge_e: f64,
    pub repeat_charges_e: Vec<f64>,
    pub atom_charges_e: Vec<f64>,
    pub provenance: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct EspManifest {
    pub schema_version: String,
    pub grid_path: Option<String>,
    pub grid_format: Option<String>,
    pub potential_unit: String,
    pub coordinate_unit: String,
    pub point_count: Option<usize>,
    pub provenance: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CubeManifest {
    pub schema_version: String,
    pub property: String,
    pub grid_path: String,
    pub grid_format: String,
    pub coordinate_unit: String,
    pub value_unit: Option<String>,
    pub point_count: Option<usize>,
    pub provenance: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct EngineCapabilities {
    pub schema_version: String,
    pub engines: Vec<engines::EngineProbe>,
    pub supported_tasks: Vec<String>,
}

fn default_schema_version() -> String {
    QM_SCHEMA_VERSION.into()
}

fn default_engine_mode() -> String {
    "cli".into()
}

fn default_version_policy() -> String {
    "record".into()
}

fn default_units() -> String {
    "angstrom".into()
}

fn default_source_kind() -> String {
    "file".into()
}

fn default_work_dir() -> String {
    "results/qm/work".into()
}

fn default_out_dir() -> String {
    "results/qm".into()
}

fn default_keep_raw() -> bool {
    true
}

fn default_true() -> bool {
    true
}

fn default_validation_depth() -> String {
    "shallow".into()
}

pub fn schema_json(kind: &str) -> Result<String, String> {
    let value = match kind {
        "request" => serde_json::to_value(schema_for!(QmRequest)),
        "result" => serde_json::to_value(schema_for!(QmResultEnvelope)),
        "event" => serde_json::to_value(schema_for!(QmRunEvent)),
        "job_manifest" => serde_json::to_value(schema_for!(JobManifest)),
        "molecule" => serde_json::to_value(schema_for!(QmMoleculeSpec)),
        "charge_manifest" => serde_json::to_value(schema_for!(ChargeManifest)),
        "polymer_charge_manifest" => serde_json::to_value(schema_for!(PolymerChargeManifest)),
        "esp_manifest" => serde_json::to_value(schema_for!(EspManifest)),
        "cube_manifest" => serde_json::to_value(schema_for!(CubeManifest)),
        "engine_capabilities" => serde_json::to_value(schema_for!(EngineCapabilities)),
        _ => return Err("schema target must be request, result, event, job_manifest, molecule, charge_manifest, polymer_charge_manifest, esp_manifest, cube_manifest, or engine_capabilities".into()),
    }
    .map_err(|err| err.to_string())?;
    serde_json::to_string_pretty(&value).map_err(|err| err.to_string())
}

pub fn project_polymer_charges_json(
    charge_manifest_path: &str,
    repeat_count: usize,
    repeat_set: &str,
    terminal_policy: &str,
) -> (i32, Value) {
    if repeat_count == 0 {
        return (
            2,
            json!({
                "schema_version": POLYMER_CHARGE_MANIFEST_VERSION,
                "status": "error",
                "errors": [{"code": "E_REPEAT_COUNT", "message": "repeat_count must be >= 1"}],
            }),
        );
    }
    let text = match fs::read_to_string(charge_manifest_path) {
        Ok(text) => text,
        Err(err) => {
            return (
                2,
                json!({
                    "schema_version": POLYMER_CHARGE_MANIFEST_VERSION,
                    "status": "error",
                    "errors": [{"code": "E_CHARGE_MANIFEST_READ", "message": err.to_string()}],
                }),
            )
        }
    };
    let manifest: ChargeManifest = match serde_json::from_str(&text) {
        Ok(manifest) => manifest,
        Err(err) => {
            return (
                2,
                json!({
                    "schema_version": POLYMER_CHARGE_MANIFEST_VERSION,
                    "status": "error",
                    "errors": [{"code": "E_CHARGE_MANIFEST_PARSE", "message": err.to_string()}],
                }),
            )
        }
    };
    let deployable = match manifest.projection.as_ref().and_then(|projection| {
        projection
            .deployable_sets
            .iter()
            .find(|set| set.name == repeat_set)
    }) {
        Some(set) => set,
        None => {
            return (
                2,
                json!({
                    "schema_version": POLYMER_CHARGE_MANIFEST_VERSION,
                    "status": "error",
                    "errors": [{"code": "E_REPEAT_SET", "message": format!("repeat set '{repeat_set}' was not found in charge_manifest.projection.deployable_sets")}],
                }),
            )
        }
    };
    let mut atom_charges = Vec::with_capacity(deployable.charges_e.len() * repeat_count);
    for _ in 0..repeat_count {
        atom_charges.extend(deployable.charges_e.iter().copied());
    }
    let total_charge_e: f64 = atom_charges.iter().sum();
    let output = PolymerChargeManifest {
        schema_version: POLYMER_CHARGE_MANIFEST_VERSION.into(),
        source_charge_manifest: charge_manifest_path.into(),
        source_model: manifest.model,
        repeat_set: repeat_set.into(),
        repeat_count,
        terminal_policy: terminal_policy.into(),
        charge_unit: manifest.charge_unit,
        total_charge_e,
        repeat_charges_e: deployable.charges_e.clone(),
        atom_charges_e: atom_charges,
        provenance: BTreeMap::from([
            ("tool".into(), json!("warp-qm")),
            (
                "policy".into(),
                json!("RadonPy-style capped-monomer RESP map tiled over polymer repeats"),
            ),
            (
                "source_projection_policy".into(),
                json!(manifest
                    .projection
                    .as_ref()
                    .map(|projection| &projection.policy)),
            ),
        ]),
    };
    (0, json!(output))
}

pub fn example_request(engine: &str, task: &str) -> Value {
    json!({
        "schema_version": QM_SCHEMA_VERSION,
        "request_id": format!("example-{engine}-{task}"),
        "engine": {
            "name": engine,
            "mode": if engine == "psi4" { "python" } else { "cli" },
            "version_policy": "record",
            "settings": {}
        },
        "molecule": {
            "source": {"kind": "file", "path": "molecule.xyz", "format": "xyz"},
            "charge": 0,
            "multiplicity": 1,
            "units": "angstrom"
        },
        "task": {
            "kind": task,
            "method": if engine == "xtb" { "gfn2" } else { "b3lyp" },
            "basis": if engine == "xtb" { Value::Null } else { json!("def2-svp") },
            "properties": ["energy", "gradient", "dipole", "charges"]
        },
        "runtime": {
            "work_dir": format!("results/qm/{engine}_{task}"),
            "threads": 4,
            "memory_mb": 8000,
            "keep_raw": true
        },
        "output": {
            "out_dir": format!("results/qm/{engine}_{task}"),
            "write_json": true,
            "write_npz": true,
            "write_xyz": true
        }
    })
}

pub fn capabilities() -> Value {
    json!(EngineCapabilities {
        schema_version: QM_SCHEMA_VERSION.into(),
        engines: engines::probe_all(),
        supported_tasks: supported_tasks(),
    })
}

pub fn validate_request_json(text: &str) -> (i32, Value) {
    let request = match parse_request(text) {
        Ok(request) => request,
        Err(err) => return validation_error(vec![err]),
    };
    let errors = validate_request(&request);
    if errors.is_empty() {
        (
            0,
            json!({
                "schema_version": QM_SCHEMA_VERSION,
                "status": "ok",
                "valid": true,
                "normalized_request": request,
                "warnings": []
            }),
        )
    } else {
        validation_error(errors)
    }
}

pub fn inspect_output_json(path: &str, engine: &str) -> (i32, Value) {
    match parsers::inspect_output(Path::new(path), engine) {
        Ok(report) => (0, json!(report)),
        Err(message) => (
            2,
            json!({
                "schema_version": QM_SCHEMA_VERSION,
                "status": "error",
                "valid": false,
                "errors": [{"code": "E_INSPECT_OUTPUT", "message": message}],
                "warnings": []
            }),
        ),
    }
}

pub fn run_request_json(text: &str, stream_ndjson: bool) -> (i32, Value) {
    let t0 = Instant::now();
    let request = match parse_request(text) {
        Ok(request) => request,
        Err(err) => {
            let result = error_result(None, "unknown", None, vec![err], t0.elapsed().as_millis());
            if stream_ndjson {
                emit(&QmRunEvent::Failed {
                    schema_version: QM_SCHEMA_VERSION.into(),
                    request_id: None,
                    result: result.clone(),
                });
            }
            return (2, json!(result));
        }
    };
    let request_id = request.request_id.clone();
    if stream_ndjson {
        emit(&QmRunEvent::Started {
            schema_version: QM_SCHEMA_VERSION.into(),
            request_id: request_id.clone(),
        });
    }
    let errors = validate_request(&request);
    if !errors.is_empty() {
        let result = error_result(
            request_id.clone(),
            &request.engine.name,
            Some(&request.task),
            errors,
            t0.elapsed().as_millis(),
        );
        if stream_ndjson {
            emit(&QmRunEvent::Failed {
                schema_version: QM_SCHEMA_VERSION.into(),
                request_id,
                result: result.clone(),
            });
        }
        return (2, json!(result));
    }

    let adapter = match request.engine.name.as_str() {
        "orca" => adapters::orca::run(&request),
        "multiwfn" => adapters::multiwfn::run(&request),
        "psi4" => adapters::psi4::run(&request),
        "gaussian" => {
            adapters::AdapterRun::error("Gaussian execution adapter is not implemented yet", 4)
        }
        "xtb" => adapters::AdapterRun::error("xTB execution adapter is not implemented yet", 4),
        _ => adapters::AdapterRun::error("unknown engine", 2),
    };
    if let Some(command) = adapter.command.as_ref() {
        if stream_ndjson {
            emit(&QmRunEvent::EngineCommand {
                schema_version: QM_SCHEMA_VERSION.into(),
                request_id: request_id.clone(),
                engine: request.engine.name.clone(),
                command: command.clone(),
            });
        }
    }
    for artifact in &adapter.artifacts {
        if stream_ndjson {
            emit(&QmRunEvent::Artifact {
                schema_version: QM_SCHEMA_VERSION.into(),
                request_id: request_id.clone(),
                artifact: artifact.clone(),
            });
        }
    }
    for warning in &adapter.warnings {
        if stream_ndjson {
            emit(&QmRunEvent::Warning {
                schema_version: QM_SCHEMA_VERSION.into(),
                request_id: request_id.clone(),
                message: warning.clone(),
            });
        }
    }
    let result = QmResultEnvelope {
        schema_version: QM_SCHEMA_VERSION.into(),
        status: adapter.status.clone(),
        exit_code: adapter.exit_code,
        request_id: request_id.clone(),
        engine: QmEngineSummary {
            name: request.engine.name.clone(),
            version: None,
            command: adapter.command,
        },
        task: QmTaskSummary {
            kind: request.task.kind.clone(),
            method: request.task.method.clone(),
            basis: request.task.basis.clone(),
        },
        summary: QmResultSummary {
            energy_hartree: adapter.summary.energy_hartree,
            converged: adapter.summary.converged,
            n_atoms: adapter.summary.n_atoms,
            wall_time_ms: Some(t0.elapsed().as_millis()),
        },
        properties: adapter.properties.into_iter().collect(),
        artifacts: adapter.artifacts,
        warnings: adapter.warnings,
        elapsed_ms: t0.elapsed().as_millis(),
    };
    if stream_ndjson {
        if result.status == "ok" {
            emit(&QmRunEvent::Completed {
                schema_version: QM_SCHEMA_VERSION.into(),
                request_id,
                result: result.clone(),
            });
        } else {
            emit(&QmRunEvent::Failed {
                schema_version: QM_SCHEMA_VERSION.into(),
                request_id,
                result: result.clone(),
            });
        }
    }
    (result.exit_code, json!(result))
}

fn parse_request(text: &str) -> Result<QmRequest, Value> {
    let mut deserializer = serde_json::Deserializer::from_str(text);
    serde_path_to_error::deserialize(&mut deserializer).map_err(|err| {
        json!({
            "code": "E_CONFIG_PARSE",
            "path": err.path().to_string(),
            "message": err.to_string()
        })
    })
}

fn validate_request(request: &QmRequest) -> Vec<Value> {
    let mut errors = Vec::new();
    if request.schema_version != QM_SCHEMA_VERSION {
        errors.push(error(
            "E_CONFIG_VERSION",
            "schema_version",
            format!(
                "unsupported schema '{}'; expected {}",
                request.schema_version, QM_SCHEMA_VERSION
            ),
        ));
    }
    if !["psi4", "orca", "gaussian", "xtb", "multiwfn"].contains(&request.engine.name.as_str()) {
        errors.push(error(
            "E_ENGINE_UNKNOWN",
            "engine.name",
            "engine must be psi4, orca, gaussian, xtb, or multiwfn",
        ));
    }
    if !supported_tasks().contains(&request.task.kind) {
        errors.push(error(
            "E_TASK_UNKNOWN",
            "task.kind",
            "unsupported QM task kind",
        ));
    }
    if request.molecule.multiplicity == 0 {
        errors.push(error(
            "E_MOLECULE_MULTIPLICITY",
            "molecule.multiplicity",
            "multiplicity must be >= 1",
        ));
    }
    if request.molecule.source.kind == "file"
        && request
            .molecule
            .source
            .path
            .as_deref()
            .unwrap_or("")
            .trim()
            .is_empty()
    {
        errors.push(error(
            "E_MOLECULE_SOURCE",
            "molecule.source.path",
            "file molecule source requires path",
        ));
    }
    if request.molecule.source.kind == "trajectory_frames"
        && request.molecule.source.frames.is_empty()
    {
        errors.push(error(
            "E_MOLECULE_FRAMES",
            "molecule.source.frames",
            "trajectory_frames source requires at least one frame",
        ));
    }
    if request.validation.depth != "shallow" && request.validation.depth != "deep" {
        errors.push(error(
            "E_VALIDATION_DEPTH",
            "validation.depth",
            "validation depth must be shallow or deep",
        ));
    }
    if request.validation.depth == "deep" {
        if let Some(path) = request.molecule.source.path.as_deref() {
            if !Path::new(path).exists() {
                errors.push(error(
                    "E_MOLECULE_SOURCE_MISSING",
                    "molecule.source.path",
                    "molecule source path does not exist",
                ));
            }
        }
    }
    errors
}

fn validation_error(errors: Vec<Value>) -> (i32, Value) {
    (
        2,
        json!({
            "schema_version": QM_SCHEMA_VERSION,
            "status": "error",
            "valid": false,
            "errors": errors,
            "warnings": []
        }),
    )
}

fn error(code: &str, path: &str, message: impl Into<String>) -> Value {
    json!({"code": code, "path": path, "message": message.into()})
}

fn error_result(
    request_id: Option<String>,
    engine_name: &str,
    task: Option<&QmTaskSpec>,
    errors: Vec<Value>,
    elapsed_ms: u128,
) -> QmResultEnvelope {
    QmResultEnvelope {
        schema_version: QM_SCHEMA_VERSION.into(),
        status: "error".into(),
        exit_code: 2,
        request_id,
        engine: QmEngineSummary {
            name: engine_name.into(),
            version: None,
            command: None,
        },
        task: QmTaskSummary {
            kind: task.map(|task| task.kind.clone()).unwrap_or_default(),
            method: task.map(|task| task.method.clone()).unwrap_or_default(),
            basis: task.and_then(|task| task.basis.clone()),
        },
        summary: QmResultSummary {
            wall_time_ms: Some(elapsed_ms),
            ..QmResultSummary::default()
        },
        properties: BTreeMap::from([("errors".into(), json!(errors))]),
        artifacts: Vec::new(),
        warnings: Vec::new(),
        elapsed_ms,
    }
}

fn supported_tasks() -> Vec<String> {
    [
        "single_point",
        "generic_run",
        "optimize",
        "frequency",
        "charges",
        "esp",
        "resp_fit",
        "resp_prepare",
        "resp_postprocess",
        "orca_molden_export",
        "population",
        "orbital_cube",
        "electron_density_cube",
        "elf_cube",
        "lol_cube",
        "laplacian_cube",
        "nmr_shielding",
        "binding_energy",
        "solvation_energy",
        "proton_affinity",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn emit(event: &QmRunEvent) {
    println!(
        "{}",
        serde_json::to_string(event).unwrap_or_else(|_| "{}".into())
    );
}

#[allow(dead_code)]
fn read_text(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|err| err.to_string())
}
