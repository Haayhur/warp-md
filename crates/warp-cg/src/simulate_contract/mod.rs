use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

mod status;

pub use status::{status_json, SimulationStatus};

pub const SIMULATE_SCHEMA_VERSION: &str = "warp-cg.simulate.v1";

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SimulateRequest {
    #[serde(default = "default_schema_version")]
    #[schemars(default = "default_schema_version")]
    pub schema_version: String,
    pub run_id: Option<String>,
    pub engine: String,
    pub system: SimulationSystem,
    pub protocol: SimulationProtocol,
    #[serde(default)]
    pub execution: SimulationExecution,
    #[serde(default)]
    pub outputs: SimulationOutputs,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SimulationSystem {
    pub coordinates: String,
    pub topology: String,
    #[serde(default)]
    pub index: Option<String>,
    #[serde(default)]
    pub parameters: Vec<String>,
    #[serde(default)]
    pub build_manifest: Option<String>,
    #[serde(default)]
    pub fitting_report: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SimulationProtocol {
    pub stages: Vec<SimulationStage>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SimulationStage {
    pub name: String,
    #[serde(rename = "type")]
    pub stage_type: String,
    #[serde(default)]
    pub ensemble: Option<String>,
    #[serde(default)]
    pub files: BTreeMap<String, String>,
    #[serde(default)]
    pub parameters: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SimulationExecution {
    #[serde(default = "default_execution_mode")]
    #[schemars(default = "default_execution_mode")]
    pub mode: String,
    #[serde(default)]
    pub work_dir: Option<String>,
    #[serde(default)]
    pub resources: BTreeMap<String, Value>,
}

impl Default for SimulationExecution {
    fn default() -> Self {
        Self {
            mode: default_execution_mode(),
            work_dir: None,
            resources: BTreeMap::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SimulationOutputs {
    #[serde(default = "default_manifest")]
    #[schemars(default = "default_manifest")]
    pub manifest: String,
    #[serde(default = "default_plan")]
    #[schemars(default = "default_plan")]
    pub plan: String,
}

impl Default for SimulationOutputs {
    fn default() -> Self {
        Self {
            manifest: default_manifest(),
            plan: default_plan(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SimulationPlan {
    pub schema_version: String,
    pub status: String,
    pub run_id: Option<String>,
    pub engine: String,
    pub execution_mode: String,
    pub work_dir: Option<String>,
    pub required_inputs: Vec<SimulationArtifact>,
    pub commands: Vec<SimulationCommand>,
    pub expected_outputs: Vec<SimulationArtifact>,
    pub checkpoints: Vec<SimulationArtifact>,
    pub environment: SimulationEnvironment,
    pub warnings: Vec<SimulationIssue>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SimulationArtifact {
    pub role: String,
    pub path: String,
    pub required: bool,
    pub exists: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SimulationCommand {
    pub stage: String,
    pub label: String,
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<String>,
    pub produces: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SimulationEnvironment {
    pub engine: String,
    pub required_programs: Vec<String>,
    pub gpu: String,
    pub cpu: String,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SimulationIssue {
    pub code: String,
    pub path: String,
    pub message: String,
    pub severity: String,
}

fn default_schema_version() -> String {
    SIMULATE_SCHEMA_VERSION.to_string()
}

fn default_execution_mode() -> String {
    "external".to_string()
}

fn default_manifest() -> String {
    "simulation_manifest.json".to_string()
}

fn default_plan() -> String {
    "simulation_plan.json".to_string()
}

pub fn schema_json(kind: &str) -> Result<String> {
    let schema = match kind {
        "request" => serde_json::to_value(schema_for!(SimulateRequest))?,
        "plan" | "result" => serde_json::to_value(schema_for!(SimulationPlan))?,
        "status" => serde_json::to_value(schema_for!(SimulationStatus))?,
        "manifest" => serde_json::to_value(schema_for!(SimulationPlan))?,
        other => return Err(anyhow!("unknown warp-cg simulate schema kind: {other}")),
    };
    Ok(serde_json::to_string_pretty(&schema)?)
}

pub fn capabilities() -> Value {
    json!({
        "schema_version": SIMULATE_SCHEMA_VERSION,
        "tool": "warp-cg simulate",
        "scope": "planning_manifest_status_only",
        "does_not_own": ["simulation execution", "scheduler submission", "scientific protocol defaults"],
        "commands": ["schema", "example", "validate", "plan", "status", "capabilities"],
        "engines": ["gromacs", "openmm"],
        "execution_modes": ["external", "local", "slurm"],
        "outputs": [
            "required input artifact list",
            "engine command or runner plan",
            "expected output artifact list",
            "checkpoint/restart artifact list",
            "environment requirements",
            "static validation warnings",
            "run directory status inspection"
        ],
        "integration": {
            "uses_existing_warp_cg_build_artifacts": true,
            "uses_existing_warp_cg_fitting_artifacts": true,
            "analysis_should_use_warp_md_capabilities": true
        }
    })
}

pub fn example_request(engine: &str) -> Result<Value> {
    match engine {
        "gromacs" => Ok(json!({
            "schema_version": SIMULATE_SCHEMA_VERSION,
            "run_id": "cg-gromacs-001",
            "engine": "gromacs",
            "system": {
                "coordinates": "outputs/membrane.gro",
                "topology": "outputs/topol.top",
                "index": "outputs/index.ndx",
                "parameters": ["martini_v3.0.0.itp"],
                "build_manifest": "outputs/membrane_manifest.json",
                "fitting_report": "outputs/tuning_report.json"
            },
            "protocol": {
                "stages": [
                    {
                        "name": "minimize",
                        "type": "energy_minimization",
                        "files": {"mdp": "protocol/minimize.mdp"},
                        "parameters": {"integrator": "steep"}
                    },
                    {
                        "name": "nvt",
                        "type": "md",
                        "ensemble": "nvt",
                        "files": {"mdp": "protocol/nvt.mdp"},
                        "parameters": {"integrator": "md", "dt_ps": 0.02, "nsteps": 50000}
                    }
                ]
            },
            "execution": {
                "mode": "external",
                "work_dir": "runs/cg-gromacs-001",
                "resources": {"gpu": true, "mpi_ranks": 1, "omp_threads": 8}
            }
        })),
        "openmm" => Ok(json!({
            "schema_version": SIMULATE_SCHEMA_VERSION,
            "run_id": "cg-openmm-001",
            "engine": "openmm",
            "system": {
                "coordinates": "outputs/membrane.pdb",
                "topology": "outputs/topol.top",
                "parameters": ["martini_v3_openmm.xml"],
                "build_manifest": "outputs/membrane_manifest.json"
            },
            "protocol": {
                "stages": [
                    {
                        "name": "minimize",
                        "type": "energy_minimization",
                        "files": {"runner": "protocol/run_openmm.py"},
                        "parameters": {"max_iterations": 1000}
                    },
                    {
                        "name": "production",
                        "type": "md",
                        "ensemble": "npt",
                        "files": {"runner": "protocol/run_openmm.py"},
                        "parameters": {"timestep_ps": 0.02, "steps": 500000, "platform": "CUDA"}
                    }
                ]
            },
            "execution": {
                "mode": "external",
                "work_dir": "runs/cg-openmm-001",
                "resources": {"gpu": true}
            }
        })),
        other => Err(anyhow!("unsupported simulate example engine: {other}")),
    }
}

pub fn validate_request_json(text: &str) -> (i32, Value) {
    match parse_request(text).and_then(validate_request) {
        Ok(request) => (
            0,
            json!({
                "schema_version": SIMULATE_SCHEMA_VERSION,
                "status": "ok",
                "valid": true,
                "normalized_request": request,
                "errors": [],
                "warnings": []
            }),
        ),
        Err(err) => (
            2,
            json!({
                "schema_version": SIMULATE_SCHEMA_VERSION,
                "status": "error",
                "valid": false,
                "errors": [{
                    "code": "E_SIMULATE_REQUEST",
                    "path": "",
                    "message": err.to_string(),
                    "severity": "error"
                }],
                "warnings": []
            }),
        ),
    }
}

pub fn plan_request_json(text: &str, engine_override: Option<&str>) -> (i32, Value) {
    match parse_request(text)
        .and_then(|mut request| {
            if let Some(engine) = engine_override {
                request.engine = engine.to_string();
            }
            validate_request(request)
        })
        .map(plan_request)
    {
        Ok(plan) => (0, serde_json::to_value(plan).unwrap_or_else(error_value)),
        Err(err) => (
            2,
            json!({
                "schema_version": SIMULATE_SCHEMA_VERSION,
                "status": "error",
                "error": {"code": "E_SIMULATE_REQUEST", "path": "", "message": err.to_string(), "severity": "error"}
            }),
        ),
    }
}

fn error_value(err: serde_json::Error) -> Value {
    json!({"schema_version": SIMULATE_SCHEMA_VERSION, "status": "error", "error": err.to_string()})
}

fn parse_request(text: &str) -> Result<SimulateRequest> {
    Ok(serde_json::from_str(text)?)
}

fn validate_request(request: SimulateRequest) -> Result<SimulateRequest> {
    if request.schema_version != SIMULATE_SCHEMA_VERSION {
        return Err(anyhow!(
            "schema_version must be {SIMULATE_SCHEMA_VERSION}, got {}",
            request.schema_version
        ));
    }
    if !matches!(request.engine.as_str(), "gromacs" | "openmm") {
        return Err(anyhow!("engine must be gromacs or openmm"));
    }
    require_nonempty("system.coordinates", &request.system.coordinates)?;
    require_nonempty("system.topology", &request.system.topology)?;
    if request.protocol.stages.is_empty() {
        return Err(anyhow!("protocol.stages must contain at least one stage"));
    }
    for (idx, stage) in request.protocol.stages.iter().enumerate() {
        let path = format!("protocol.stages[{idx}]");
        require_nonempty(&format!("{path}.name"), &stage.name)?;
        require_nonempty(&format!("{path}.type"), &stage.stage_type)?;
        if stage.parameters.is_empty() {
            return Err(anyhow!(
                "{path}.parameters must be explicit; warp-cg simulate does not provide scientific protocol defaults"
            ));
        }
        match request.engine.as_str() {
            "gromacs" if !stage.files.contains_key("mdp") => {
                return Err(anyhow!("{path}.files.mdp is required for gromacs plans"));
            }
            "openmm" if !stage.files.contains_key("runner") => {
                return Err(anyhow!("{path}.files.runner is required for openmm plans"));
            }
            _ => {}
        }
    }
    if !matches!(
        request.execution.mode.as_str(),
        "external" | "local" | "slurm"
    ) {
        return Err(anyhow!("execution.mode must be external, local, or slurm"));
    }
    Ok(request)
}

fn require_nonempty(path: &str, value: &str) -> Result<()> {
    if value.trim().is_empty() {
        Err(anyhow!("{path} must not be empty"))
    } else {
        Ok(())
    }
}

fn plan_request(request: SimulateRequest) -> SimulationPlan {
    let mut required_inputs = vec![
        artifact("coordinates", &request.system.coordinates, true),
        artifact("topology", &request.system.topology, true),
    ];
    if let Some(index) = &request.system.index {
        required_inputs.push(artifact("index", index, false));
    }
    for parameter in &request.system.parameters {
        required_inputs.push(artifact("parameter", parameter, true));
    }
    if let Some(path) = &request.system.build_manifest {
        required_inputs.push(artifact("build_manifest", path, false));
    }
    if let Some(path) = &request.system.fitting_report {
        required_inputs.push(artifact("fitting_report", path, false));
    }

    let mut commands = Vec::new();
    let mut expected_outputs = Vec::new();
    let mut checkpoints = Vec::new();
    match request.engine.as_str() {
        "gromacs" => plan_gromacs(
            &request,
            &mut commands,
            &mut expected_outputs,
            &mut checkpoints,
        ),
        "openmm" => plan_openmm(
            &request,
            &mut commands,
            &mut expected_outputs,
            &mut checkpoints,
        ),
        _ => {}
    }

    SimulationPlan {
        schema_version: SIMULATE_SCHEMA_VERSION.to_string(),
        status: "ok".to_string(),
        run_id: request.run_id.clone(),
        engine: request.engine.clone(),
        execution_mode: request.execution.mode.clone(),
        work_dir: request.execution.work_dir.clone(),
        required_inputs,
        commands,
        expected_outputs,
        checkpoints,
        environment: environment_for(&request),
        warnings: vec![SimulationIssue {
            code: "W_EXECUTION_NOT_OWNED".to_string(),
            path: "execution".to_string(),
            message: "warp-cg simulate emits a plan only; APS or another runner owns execution, scheduling, monitoring, and retries".to_string(),
            severity: "warning".to_string(),
        }],
    }
}

fn plan_gromacs(
    request: &SimulateRequest,
    commands: &mut Vec<SimulationCommand>,
    expected_outputs: &mut Vec<SimulationArtifact>,
    checkpoints: &mut Vec<SimulationArtifact>,
) {
    let mut previous_coordinates = request.system.coordinates.clone();
    for stage in &request.protocol.stages {
        let mdp = stage.files.get("mdp").cloned().unwrap_or_default();
        let tpr = format!("{}.tpr", stage.name);
        let deffnm = stage.name.clone();
        let mut grompp_args = vec![
            "grompp".to_string(),
            "-f".to_string(),
            mdp,
            "-c".to_string(),
            previous_coordinates.clone(),
            "-p".to_string(),
            request.system.topology.clone(),
            "-o".to_string(),
            tpr.clone(),
        ];
        if let Some(index) = &request.system.index {
            grompp_args.extend(["-n".to_string(), index.clone()]);
        }
        commands.push(command(
            stage,
            "prepare_tpr",
            "gmx",
            grompp_args,
            vec![tpr.clone()],
        ));
        commands.push(command(
            stage,
            "run_stage",
            "gmx",
            vec!["mdrun".to_string(), "-deffnm".to_string(), deffnm.clone()],
            vec![
                format!("{deffnm}.gro"),
                format!("{deffnm}.xtc"),
                format!("{deffnm}.edr"),
                format!("{deffnm}.log"),
                format!("{deffnm}.cpt"),
            ],
        ));
        previous_coordinates = format!("{deffnm}.gro");
        expected_outputs.extend([
            artifact("stage_tpr", &tpr, true),
            artifact("trajectory", &format!("{deffnm}.xtc"), true),
            artifact("energy", &format!("{deffnm}.edr"), true),
            artifact("log", &format!("{deffnm}.log"), true),
            artifact("final_structure", &format!("{deffnm}.gro"), true),
        ]);
        checkpoints.push(artifact("checkpoint", &format!("{deffnm}.cpt"), false));
    }
}

fn plan_openmm(
    request: &SimulateRequest,
    commands: &mut Vec<SimulationCommand>,
    expected_outputs: &mut Vec<SimulationArtifact>,
    checkpoints: &mut Vec<SimulationArtifact>,
) {
    for stage in &request.protocol.stages {
        let runner = stage.files.get("runner").cloned().unwrap_or_default();
        let stage_json = format!("{}_stage_request.json", stage.name);
        commands.push(command(
            stage,
            "run_stage",
            "python",
            vec![
                runner,
                "--stage".to_string(),
                stage.name.clone(),
                "--request".to_string(),
                stage_json,
            ],
            vec![
                format!("{}.dcd", stage.name),
                format!("{}.log", stage.name),
                format!("{}.chk", stage.name),
                format!("{}_final.pdb", stage.name),
            ],
        ));
        expected_outputs.extend([
            artifact("trajectory", &format!("{}.dcd", stage.name), true),
            artifact("log", &format!("{}.log", stage.name), true),
            artifact(
                "final_structure",
                &format!("{}_final.pdb", stage.name),
                true,
            ),
        ]);
        checkpoints.push(artifact(
            "checkpoint",
            &format!("{}.chk", stage.name),
            false,
        ));
    }
}

fn command(
    stage: &SimulationStage,
    label: &str,
    program: &str,
    args: Vec<String>,
    produces: Vec<String>,
) -> SimulationCommand {
    SimulationCommand {
        stage: stage.name.clone(),
        label: label.to_string(),
        program: program.to_string(),
        args,
        cwd: None,
        produces,
    }
}

fn artifact(role: &str, path: &str, required: bool) -> SimulationArtifact {
    SimulationArtifact {
        role: role.to_string(),
        path: path.to_string(),
        required,
        exists: Some(Path::new(path).exists()),
    }
}

fn environment_for(request: &SimulateRequest) -> SimulationEnvironment {
    match request.engine.as_str() {
        "gromacs" => SimulationEnvironment {
            engine: "gromacs".to_string(),
            required_programs: vec!["gmx".to_string()],
            gpu: "requested by execution.resources.gpu; not enforced by warp-cg".to_string(),
            cpu: "requested by execution.resources; not enforced by warp-cg".to_string(),
            notes: vec!["Gromacs version/module selection is owned by the runner".to_string()],
        },
        "openmm" => SimulationEnvironment {
            engine: "openmm".to_string(),
            required_programs: vec!["python".to_string()],
            gpu: "OpenMM platform is provided in stage parameters or runner config".to_string(),
            cpu: "runner-owned".to_string(),
            notes: vec![
                "OpenMM execution is script/API driven; warp-cg emits a runner command plan"
                    .to_string(),
            ],
        },
        _ => SimulationEnvironment {
            engine: request.engine.clone(),
            required_programs: Vec::new(),
            gpu: "unknown".to_string(),
            cpu: "unknown".to_string(),
            notes: Vec::new(),
        },
    }
}
