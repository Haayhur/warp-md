use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::mapping::{map_molecule, MappingResult};
use crate::molecule::Molecule;
use crate::optimize::{optimize_bonded_terms, OptimizationConfig, OptimizationReport};
use crate::parameters::{calculate_bonded_stats, AngleStats, BondStats, BondedStats, DihedralStats};
use crate::trajectory::{
    map_native_trajectory, map_trajectory_first_frame, BeadMapping, NativeTrajectoryOptions,
};
use crate::xtb::{run_xtb_pipeline_with_config, XtbRunConfig};

pub const AGENT_SCHEMA_VERSION: &str = "warp-cg.agent.v1";

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CgRequest {
    #[serde(default = "default_schema_version")]
    #[schemars(default = "default_schema_version")]
    pub schema_version: String,
    pub name: String,
    pub smiles: String,
    /// Legacy shorthand for an external trajectory path. Prefer trajectory_source.
    #[serde(default)]
    pub trajectory: Option<String>,
    #[serde(default)]
    pub topology: Option<String>,
    #[serde(default)]
    pub trajectory_source: Option<TrajectorySource>,
    #[serde(default)]
    pub reference_source: Option<ReferenceSource>,
    #[serde(default)]
    pub parameter_tuning: Option<ParameterTuningRequest>,
    #[serde(default = "default_output")]
    #[schemars(default = "default_output")]
    pub output: CgOutputRequest,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct TrajectorySource {
    /// External trajectory path accepted by chemfiles/warp-md loaders, e.g. xtc, dcd, trr, pdb.
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
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ReferenceSource {
    #[serde(default = "default_reference_kind")]
    #[schemars(default = "default_reference_kind")]
    pub kind: String,
    #[serde(default)]
    pub xtb: Option<XtbRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct XtbRequest {
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
    pub max_evaluations: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub swarm_size: Option<usize>,
    #[serde(default = "default_tuning_objective")]
    #[schemars(default = "default_tuning_objective")]
    pub objective: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
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
    pub bead_count: usize,
    pub beads: Vec<CgBead>,
    pub connections: Vec<[usize; 2]>,
    pub artifacts: Vec<CgArtifact>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameter_tuning: Option<ParameterTuningResult>,
    pub elapsed_ms: u128,
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

fn default_schema_version() -> String {
    AGENT_SCHEMA_VERSION.to_string()
}

fn default_out_dir() -> String {
    ".".to_string()
}

fn default_write_mapping() -> bool {
    true
}

fn default_write_topology_itp() -> bool {
    true
}

fn default_write_topology_top() -> bool {
    false
}

fn default_write_cg_pdb() -> bool {
    true
}

fn default_write_bonded_parameter_map() -> bool {
    true
}

fn default_external_trajectory_kind() -> String {
    "external".to_string()
}

fn default_reference_kind() -> String {
    "external".to_string()
}

fn default_tuning_enabled() -> bool {
    false
}

fn default_tuning_source() -> String {
    "external_trajectory".to_string()
}

fn default_tuning_method() -> String {
    "bayesian_optimization".to_string()
}

fn default_tuning_objective() -> String {
    "bonded_parameter_parity".to_string()
}

fn default_output() -> CgOutputRequest {
    CgOutputRequest {
        out_dir: default_out_dir(),
        mapped_trajectory: None,
        write_mapping_json: true,
        write_topology_itp: true,
        write_topology_top: true,
        write_cg_pdb: true,
        cg_pdb: None,
        write_bonded_parameter_map: true,
    }
}

pub fn schema_json(kind: &str) -> Result<String> {
    let value = match kind {
        "request" => serde_json::to_value(schema_for!(CgRequest))?,
        "result" => serde_json::to_value(schema_for!(CgResult))?,
        "event" => serde_json::to_value(schema_for!(CgEvent))?,
        other => return Err(anyhow!("unknown warp-cg schema kind: {other}")),
    };
    Ok(serde_json::to_string_pretty(&value)?)
}

pub fn example_request() -> Value {
    json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "trajectory_source": {
            "path": "traj.xtc",
            "topology": "topology.pdb",
            "kind": "external",
            "target_selection": "resname BENZ",
            "stride": 1
        },
        "output": {
            "out_dir": "results/cg/benzene",
            "mapped_trajectory": "benzene_cg.xtc",
            "write_mapping_json": true,
            "write_topology_itp": true,
            "write_topology_top": true,
            "write_cg_pdb": true,
            "write_bonded_parameter_map": true
        },
        "parameter_tuning": {
            "enabled": false,
            "source": "external_trajectory",
            "method": "bayesian_optimization",
            "objective": "bonded_parameter_parity"
        }
    })
}

pub fn capabilities() -> Value {
    json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "tool": "warp-cg",
        "contracts": {
            "request": AGENT_SCHEMA_VERSION,
            "result": AGENT_SCHEMA_VERSION,
            "event": AGENT_SCHEMA_VERSION
        },
        "inputs": {
            "smiles": true,
            "trajectory_mapping": {
                "legacy_field": "trajectory",
                "preferred_field": "trajectory_source",
                "formats": ["dcd", "xtc", "gro", "g96", "cpt", "h5md", "tng", "trr", "pdb", "pdbqt"],
                "target_description": "topology plus warp-md selection expressions or explicit atom indices",
                "environment_selection": "accepted for future solvent/environment metadata; mapping currently uses target_selection or atom_indices"
            },
            "topology": "used to validate solvated external trajectories and resolve target_selection",
            "xtb": "reference_source.kind=xtb can initiate xTB optimization/MD"
        },
        "parameter_tuning": {
            "status": "implemented",
            "methods": ["bayesian_optimization", "pso"],
            "sources": ["external_trajectory", "xtb"],
            "objective": "bonded_parameter_parity",
            "terms": ["bonds", "angles", "dihedrals"],
            "empty_stats_behavior": "returns skipped report when no bonded reference statistics are available"
        },
        "outputs": [
            "martini_bead_mapping_json",
            "coarse_grained_pdb",
            "optional_coarse_grained_trajectory_xtc_dcd_gro_g96_cpt_trr_h5md",
            "bond_stats_json",
            "bonded_stats_json",
            "bonded_parameter_map_json",
            "bonded_parameter_tuning_report",
            "martini_topology_itp",
            "martini_topology_top"
        ],
        "force_field": "Martini 3 coarse-grained bead assignment"
    })
}

pub fn validate_request_json(text: &str) -> (i32, Value) {
    match parse_request(text).and_then(validate_request) {
        Ok(request) => (
            0,
            json!({
                "schema_version": AGENT_SCHEMA_VERSION,
                "valid": true,
                "status": "ok",
                "name": request.name,
            }),
        ),
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

fn validate_request(request: CgRequest) -> Result<CgRequest> {
    if request.schema_version != AGENT_SCHEMA_VERSION {
        return Err(anyhow!(
            "schema_version must be {AGENT_SCHEMA_VERSION}, got {}",
            request.schema_version
        ));
    }
    if request.name.trim().is_empty() {
        return Err(anyhow!("name is required"));
    }
    if request.smiles.trim().is_empty() {
        return Err(anyhow!("smiles is required"));
    }
    if request
        .topology
        .as_ref()
        .is_some_and(|topology| topology.trim().is_empty())
    {
        return Err(anyhow!("topology must not be empty"));
    }
    if request.output.out_dir.trim().is_empty() {
        return Err(anyhow!("output.out_dir must not be empty"));
    }
    if request
        .output
        .mapped_trajectory
        .as_ref()
        .is_some_and(|path| path.trim().is_empty())
    {
        return Err(anyhow!("output.mapped_trajectory must not be empty"));
    }
    if request
        .output
        .cg_pdb
        .as_ref()
        .is_some_and(|path| path.trim().is_empty())
    {
        return Err(anyhow!("output.cg_pdb must not be empty"));
    }
    if request.output.write_topology_top && !request.output.write_topology_itp {
        return Err(anyhow!(
            "output.write_topology_top requires output.write_topology_itp because the .top includes the generated .itp"
        ));
    }
    if request.trajectory.is_none() && request.output.mapped_trajectory.is_some() {
        let has_xtb_reference = request
            .reference_source
            .as_ref()
            .is_some_and(|source| source.kind == "xtb");
        if request.trajectory_source.is_none() && !has_xtb_reference {
            return Err(anyhow!(
                "output.mapped_trajectory requires trajectory, trajectory_source, or reference_source.kind=xtb"
            ));
        }
    }
    if request.trajectory.is_some() && request.trajectory_source.is_some() {
        return Err(anyhow!(
            "use either trajectory or trajectory_source, not both"
        ));
    }
    if let Some(source) = &request.trajectory_source {
        if source.path.trim().is_empty() {
            return Err(anyhow!("trajectory_source.path is required"));
        }
        if source
            .topology
            .as_ref()
            .is_some_and(|topology| topology.trim().is_empty())
        {
            return Err(anyhow!("trajectory_source.topology must not be empty"));
        }
        if source.kind != "external" {
            return Err(anyhow!("trajectory_source.kind must be external"));
        }
        if source.stride == Some(0) {
            return Err(anyhow!(
                "trajectory_source.stride must be greater than zero"
            ));
        }
        if source.stop.is_some() && source.start.is_some() && source.stop <= source.start {
            return Err(anyhow!("trajectory_source.stop must be greater than start"));
        }
        if source
            .atom_indices
            .as_ref()
            .is_some_and(|indices| indices.is_empty())
        {
            return Err(anyhow!("trajectory_source.atom_indices must not be empty"));
        }
        if source.target_selection.is_some() && source.atom_indices.is_some() {
            return Err(anyhow!(
                "use either trajectory_source.target_selection or atom_indices, not both"
            ));
        }
        if source
            .target_selection
            .as_ref()
            .is_some_and(|selection| selection.trim().is_empty())
        {
            return Err(anyhow!(
                "trajectory_source.target_selection must not be empty"
            ));
        }
        if source
            .environment_selection
            .as_ref()
            .is_some_and(|selection| selection.trim().is_empty())
        {
            return Err(anyhow!(
                "trajectory_source.environment_selection must not be empty"
            ));
        }
        if source.mass_weighted == Some(true)
            && source.topology.is_none()
            && request.topology.is_none()
        {
            return Err(anyhow!(
                "trajectory_source.mass_weighted requires trajectory_source.topology or top-level topology"
            ));
        }
    }
    if let Some(reference) = &request.reference_source {
        if reference.kind != "external" && reference.kind != "xtb" {
            return Err(anyhow!("reference_source.kind must be external or xtb"));
        }
        if let Some(xtb) = &reference.xtb {
            validate_positive(xtb.temperature_k, "reference_source.xtb.temperature_k")?;
            validate_positive(xtb.time_ps, "reference_source.xtb.time_ps")?;
            validate_positive(xtb.timestep_fs, "reference_source.xtb.timestep_fs")?;
            validate_positive(xtb.dump_fs, "reference_source.xtb.dump_fs")?;
            if xtb.gfn.as_ref().is_some_and(|gfn| gfn.trim().is_empty()) {
                return Err(anyhow!("reference_source.xtb.gfn must not be empty"));
            }
        }
    }
    if let Some(tuning) = &request.parameter_tuning {
        if tuning.method != "bayesian_optimization" && tuning.method != "pso" {
            return Err(anyhow!(
                "parameter_tuning.method must be bayesian_optimization or pso"
            ));
        }
        if tuning.max_evaluations == Some(0) {
            return Err(anyhow!(
                "parameter_tuning.max_evaluations must be greater than zero"
            ));
        }
        if tuning.swarm_size == Some(0) {
            return Err(anyhow!(
                "parameter_tuning.swarm_size must be greater than zero"
            ));
        }
        if tuning.source != "external_trajectory" && tuning.source != "xtb" {
            return Err(anyhow!(
                "parameter_tuning.source must be external_trajectory or xtb"
            ));
        }
        if tuning.enabled
            && tuning.source == "external_trajectory"
            && request.trajectory.is_none()
            && request.trajectory_source.is_none()
        {
            return Err(anyhow!(
                "external_trajectory parameter tuning requires trajectory or trajectory_source"
            ));
        }
        if tuning.enabled
            && tuning.source == "xtb"
            && !request
                .reference_source
                .as_ref()
                .is_some_and(|source| source.kind == "xtb")
        {
            return Err(anyhow!(
                "xtb parameter tuning requires reference_source.kind=xtb"
            ));
        }
    }
    Ok(request)
}

fn run_request(request: &CgRequest, started: Instant) -> Result<CgResult> {
    let mol = Molecule::from_smiles(&request.smiles)?;
    let mapping = map_molecule(&mol);
    let out_dir = PathBuf::from(&request.output.out_dir);
    std::fs::create_dir_all(&out_dir)?;

    let mut artifacts = Vec::new();
    if request.output.write_mapping_json {
        let mapping_path = out_dir.join(format!("{}_martini_mapping.json", request.name));
        let mapping_value = mapping_json(request, &mapping);
        std::fs::write(&mapping_path, serde_json::to_vec_pretty(&mapping_value)?)?;
        artifacts.push(CgArtifact {
            path: mapping_path.to_string_lossy().to_string(),
            kind: "martini_mapping_json".to_string(),
        });
    }

    let reference_kind = request
        .reference_source
        .as_ref()
        .map(|source| source.kind.as_str())
        .unwrap_or("external");
    let mut xtb_reference_path: Option<PathBuf> = None;
    if reference_kind == "xtb" {
        let xtb_out_dir = request
            .reference_source
            .as_ref()
            .and_then(|source| source.xtb.as_ref())
            .and_then(|xtb| xtb.work_dir.as_ref())
            .map(PathBuf::from)
            .unwrap_or_else(|| out_dir.clone());
        let xtb_config = request
            .reference_source
            .as_ref()
            .and_then(|source| source.xtb.as_ref())
            .map(xtb_run_config)
            .unwrap_or_default();
        let xtb_res = run_xtb_pipeline_with_config(
            &request.name,
            &request.smiles,
            &xtb_out_dir,
            &xtb_config,
        )?;
        artifacts.push(CgArtifact {
            path: xtb_res.opt_xyz.to_string_lossy().to_string(),
            kind: "xtb_optimized_xyz".to_string(),
        });
        if let Some(trj) = xtb_res.trajectory_trj.as_ref() {
            artifacts.push(CgArtifact {
                path: trj.to_string_lossy().to_string(),
                kind: "xtb_reference_trajectory".to_string(),
            });
            xtb_reference_path = Some(trj.clone());
        } else {
            xtb_reference_path = Some(xtb_res.opt_xyz);
        }
    }

    let trajectory_path = xtb_reference_path
        .as_ref()
        .map(|path| path.to_string_lossy().to_string())
        .or_else(|| {
            request
                .trajectory_source
                .as_ref()
                .map(|source| source.path.clone())
                .or_else(|| request.trajectory.clone())
        });

    let mut bond_stats: Vec<BondStats> = Vec::new();
    let mut angle_stats: Vec<AngleStats> = Vec::new();
    let mut dihedral_stats: Vec<DihedralStats> = Vec::new();
    let mut first_cg_coords: Option<Vec<[f32; 3]>> = None;

    if let Some(input_traj) = trajectory_path {
        let mapped_name = request
            .output
            .mapped_trajectory
            .clone()
            .unwrap_or_else(|| format!("{}_cg.xtc", request.name));
        let output_path = resolve_output_path(&out_dir, &mapped_name);
        let bead_mapping = BeadMapping {
            bead_names: mapping.bead_names.clone(),
            atom_indices: mapping.atom_groups.clone(),
        };
        let source = normalized_trajectory_source(request, &input_traj);
        if reference_kind == "external" || request.trajectory_source.is_some() {
            let report = map_native_trajectory(
                Path::new(&input_traj),
                Some(&output_path),
                &bead_mapping,
                &mapping.connections,
                &native_options(request, source.as_ref()),
            )?;
            bond_stats = report.bond_stats;
            angle_stats = report.angle_stats;
            dihedral_stats = report.dihedral_stats;
            first_cg_coords = report.first_cg_coords;
        } else {
            first_cg_coords =
                map_trajectory_first_frame(&input_traj, &output_path.to_string_lossy(), &bead_mapping)?;
            let bonded_stats = calculate_bonded_stats(
                &output_path.to_string_lossy(),
                mapping.bead_names.len(),
                &mapping.connections,
            )?;
            bond_stats = bonded_stats.bonds;
            angle_stats = bonded_stats.angles;
            dihedral_stats = bonded_stats.dihedrals;
        }
        artifacts.push(CgArtifact {
            path: output_path.to_string_lossy().to_string(),
            kind: "coarse_grained_trajectory".to_string(),
        });
    }

    if request.output.write_cg_pdb {
        let pdb_name = request
            .output
            .cg_pdb
            .clone()
            .unwrap_or_else(|| format!("{}_cg.pdb", request.name));
        let pdb_path = resolve_output_path(&out_dir, &pdb_name);
        let pdb = render_cg_pdb(&request.name, &mapping, first_cg_coords.as_deref());
        std::fs::write(&pdb_path, pdb)?;
        artifacts.push(CgArtifact {
            path: pdb_path.to_string_lossy().to_string(),
            kind: "coarse_grained_pdb".to_string(),
        });
    }

    if !bond_stats.is_empty() {
        let stats_path = out_dir.join(format!("{}_bond_stats.json", request.name));
        std::fs::write(&stats_path, serde_json::to_vec_pretty(&bond_stats)?)?;
        artifacts.push(CgArtifact {
            path: stats_path.to_string_lossy().to_string(),
            kind: "bond_stats_json".to_string(),
        });
    }
    if !angle_stats.is_empty() || !dihedral_stats.is_empty() {
        let stats_path = out_dir.join(format!("{}_bonded_stats.json", request.name));
        std::fs::write(
            &stats_path,
            serde_json::to_vec_pretty(&json!({
                "bonds": bond_stats,
                "angles": angle_stats,
                "dihedrals": dihedral_stats,
            }))?,
        )?;
        artifacts.push(CgArtifact {
            path: stats_path.to_string_lossy().to_string(),
            kind: "bonded_stats_json".to_string(),
        });
    }

    let parameter_tuning = request
        .parameter_tuning
        .as_ref()
        .filter(|tuning| tuning.enabled)
        .map(|tuning| {
            let bonded_stats = BondedStats {
                bonds: bond_stats.clone(),
                angles: angle_stats.clone(),
                dihedrals: dihedral_stats.clone(),
            };
            run_parameter_tuning(tuning, &bonded_stats, &out_dir, &request.name, &mut artifacts)
        })
        .transpose()?;

    if request.output.write_topology_itp {
        let itp_path = out_dir.join(format!("{}_martini.itp", request.name));
        let itp = render_martini_itp(
            &request.name,
            &mapping,
            &bond_stats,
            &angle_stats,
            &dihedral_stats,
            parameter_tuning
                .as_ref()
                .and_then(|tuning| tuning.report.as_ref()),
        );
        std::fs::write(&itp_path, itp)?;
        artifacts.push(CgArtifact {
            path: itp_path.to_string_lossy().to_string(),
            kind: "martini_topology_itp".to_string(),
        });
        if request.output.write_bonded_parameter_map {
            let map_path = out_dir.join(format!("{}_bonded_parameter_map.json", request.name));
            let parameter_map = bonded_parameter_map_json(
                request,
                &mapping,
                &bond_stats,
                &angle_stats,
                &dihedral_stats,
                parameter_tuning
                    .as_ref()
                    .and_then(|tuning| tuning.report.as_ref()),
            );
            std::fs::write(&map_path, serde_json::to_vec_pretty(&parameter_map)?)?;
            artifacts.push(CgArtifact {
                path: map_path.to_string_lossy().to_string(),
                kind: "bonded_parameter_map_json".to_string(),
            });
        }
    }
    if request.output.write_topology_top {
        let top_path = out_dir.join(format!("{}_martini.top", request.name));
        let itp_file = format!("{}_martini.itp", request.name);
        let top = render_martini_top(&request.name, &itp_file);
        std::fs::write(&top_path, top)?;
        artifacts.push(CgArtifact {
            path: top_path.to_string_lossy().to_string(),
            kind: "martini_topology_top".to_string(),
        });
    }

    Ok(CgResult {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        status: "ok".to_string(),
        exit_code: 0,
        name: request.name.clone(),
        bead_count: mapping.bead_names.len(),
        beads: beads(&mapping),
        connections: mapping.connections.iter().map(|&(i, j)| [i, j]).collect(),
        artifacts,
        parameter_tuning,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

fn run_parameter_tuning(
    tuning: &ParameterTuningRequest,
    bonded_stats: &BondedStats,
    out_dir: &Path,
    name: &str,
    artifacts: &mut Vec<CgArtifact>,
) -> Result<ParameterTuningResult> {
    let report = optimize_bonded_terms(
        bonded_stats,
        &OptimizationConfig {
            method: tuning.method.clone(),
            objective: tuning.objective.clone(),
            max_evaluations: tuning.max_evaluations.unwrap_or(32),
            seed: tuning.seed.unwrap_or(42),
            swarm_size: tuning.swarm_size,
        },
    );
    let report_path = out_dir.join(format!("{}_tuning_report.json", name));
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    artifacts.push(CgArtifact {
        path: report_path.to_string_lossy().to_string(),
        kind: "bonded_parameter_tuning_report".to_string(),
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

fn mapping_json(request: &CgRequest, mapping: &MappingResult) -> Value {
    json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "kind": "martini_mapping",
        "name": request.name,
        "smiles": request.smiles,
        "bead_count": mapping.bead_names.len(),
        "beads": beads(mapping),
        "connections": mapping.connections.iter().map(|&(i, j)| [i, j]).collect::<Vec<_>>()
    })
}

fn bonded_parameter_map_json(
    request: &CgRequest,
    mapping: &MappingResult,
    bond_stats: &[BondStats],
    angle_stats: &[AngleStats],
    dihedral_stats: &[DihedralStats],
    tuning: Option<&OptimizationReport>,
) -> Value {
    json!({
        "schema_version": "warp-cg.bonded-parameter-map.v1",
        "name": request.name,
        "smiles": request.smiles,
        "itp": format!("{}_martini.itp", request.name),
        "units": {
            "bond_length": "nm",
            "bond_force": "kJ mol^-1 nm^-2",
            "angle": "degree",
            "angle_force": "kJ mol^-1 rad^-2",
            "dihedral_phase": "degree",
            "dihedral_force": "kJ mol^-1"
        },
        "bonds": mapping.connections.iter().map(|&(i, j)| {
            let a = i.min(j);
            let b = i.max(j);
            let (length_nm, force) = bonded_pair_parameters(i, j, bond_stats, tuning);
            json!({
                "itp_section": "bonds",
                "beads_zero_based": [i, j],
                "itp_atoms_one_based": [i + 1, j + 1],
                "parameter_names": {
                    "length_angstrom": format!("bond_{a}_{b}_length_angstrom"),
                    "force": format!("bond_{a}_{b}_force")
                },
                "source_stat": bond_stats.iter().find(|stat| stat.bead_i == a && stat.bead_j == b),
                "itp_values": {
                    "funct": 1,
                    "length_nm": length_nm,
                    "force": force
                }
            })
        }).collect::<Vec<_>>(),
        "angles": angle_stats.iter().map(|stat| {
            let (angle_deg, force) = bonded_angle_parameters(stat, tuning);
            json!({
                "itp_section": "angles",
                "beads_zero_based": [stat.bead_i, stat.bead_j, stat.bead_k],
                "itp_atoms_one_based": [stat.bead_i + 1, stat.bead_j + 1, stat.bead_k + 1],
                "parameter_names": {
                    "angle_deg": format!("angle_{}_{}_{}_angle_deg", stat.bead_i, stat.bead_j, stat.bead_k),
                    "force": format!("angle_{}_{}_{}_force", stat.bead_i, stat.bead_j, stat.bead_k)
                },
                "source_stat": stat,
                "itp_values": {
                    "funct": 2,
                    "angle_deg": angle_deg,
                    "force": force
                }
            })
        }).collect::<Vec<_>>(),
        "dihedrals": dihedral_stats.iter().map(|stat| {
            let (phase_deg, force) = bonded_dihedral_parameters(stat, tuning);
            json!({
                "itp_section": "dihedrals",
                "beads_zero_based": [stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l],
                "itp_atoms_one_based": [stat.bead_i + 1, stat.bead_j + 1, stat.bead_k + 1, stat.bead_l + 1],
                "parameter_names": {
                    "phase_deg": format!(
                        "dihedral_{}_{}_{}_{}_phase_deg",
                        stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
                    ),
                    "force": format!(
                        "dihedral_{}_{}_{}_{}_force",
                        stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
                    )
                },
                "source_stat": stat,
                "itp_values": {
                    "funct": 1,
                    "phase_deg": phase_deg,
                    "force": force,
                    "multiplicity": 1
                }
            })
        }).collect::<Vec<_>>()
    })
}

fn beads(mapping: &MappingResult) -> Vec<CgBead> {
    mapping
        .bead_names
        .iter()
        .zip(mapping.atom_groups.iter())
        .enumerate()
        .map(|(index, (name, atom_indices))| CgBead {
            index,
            name: name.clone(),
            atom_indices: atom_indices.clone(),
        })
        .collect()
}

fn render_martini_itp(
    name: &str,
    mapping: &MappingResult,
    bond_stats: &[BondStats],
    angle_stats: &[AngleStats],
    dihedral_stats: &[DihedralStats],
    tuning: Option<&OptimizationReport>,
) -> String {
    let molecule = topology_name(name);
    let mut out = String::new();
    out.push_str("; Generated by warp-cg\n");
    out.push_str("[ moleculetype ]\n");
    out.push_str("; name  nrexcl\n");
    out.push_str(&format!("{molecule:<16} 1\n\n"));
    out.push_str("[ atoms ]\n");
    out.push_str("; nr  type  resnr  residue  atom  cgnr  charge\n");
    for (idx, bead) in mapping.bead_names.iter().enumerate() {
        out.push_str(&format!(
            "{:>5} {:<6} {:>5} {:<8} {:<6} {:>5} {:>8.3}\n",
            idx + 1,
            bead,
            1,
            molecule,
            format!("B{}", idx + 1),
            idx + 1,
            0.0
        ));
    }
    if !mapping.connections.is_empty() {
        out.push_str("\n[ bonds ]\n");
        out.push_str("; i  j  funct  length(nm)  force\n");
        for &(i, j) in &mapping.connections {
            let (length_nm, force) = bonded_pair_parameters(i, j, bond_stats, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>10.5} {:>10.3}\n",
                i + 1,
                j + 1,
                1,
                length_nm,
                force
            ));
        }
    }
    if !angle_stats.is_empty() {
        out.push_str("\n[ angles ]\n");
        out.push_str("; i  j  k  funct  angle(deg)  force\n");
        for stat in angle_stats {
            let (angle, force) = bonded_angle_parameters(stat, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>5} {:>10.4} {:>10.3}\n",
                stat.bead_i + 1,
                stat.bead_j + 1,
                stat.bead_k + 1,
                2,
                angle,
                force
            ));
        }
    }
    if !dihedral_stats.is_empty() {
        out.push_str("\n[ dihedrals ]\n");
        out.push_str("; i  j  k  l  funct  angle(deg)  force  multiplicity\n");
        for stat in dihedral_stats {
            let (phase, force) = bonded_dihedral_parameters(stat, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>5} {:>5} {:>10.4} {:>10.3} {:>5}\n",
                stat.bead_i + 1,
                stat.bead_j + 1,
                stat.bead_k + 1,
                stat.bead_l + 1,
                1,
                phase,
                force,
                1
            ));
        }
    }
    out
}

fn render_cg_pdb(name: &str, mapping: &MappingResult, coords: Option<&[[f32; 3]]>) -> String {
    let residue = topology_name(name);
    let residue = residue.chars().take(3).collect::<String>();
    let mut out = String::new();
    out.push_str("REMARK Generated by warp-cg\n");
    if coords.is_none() {
        out.push_str("REMARK Coordinates are deterministic scaffold positions; prefer mapped trajectory/GRO when available.\n");
    }
    for (idx, bead) in mapping.bead_names.iter().enumerate() {
        let coord = coords
            .and_then(|values| values.get(idx))
            .copied()
            .unwrap_or([idx as f32 * 4.7, 0.0, 0.0]);
        out.push_str(&format!(
            "ATOM  {:>5} {:<4} {:<3} A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00  0.00          {:>2}\n",
            idx + 1,
            pdb_atom_name(bead, idx),
            residue,
            1,
            coord[0],
            coord[1],
            coord[2],
            pdb_element(bead)
        ));
    }
    for &(i, j) in &mapping.connections {
        out.push_str(&format!("CONECT{:>5}{:>5}\n", i + 1, j + 1));
    }
    out.push_str("END\n");
    out
}

fn pdb_atom_name(bead: &str, idx: usize) -> String {
    let mut name: String = bead
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .take(3)
        .collect();
    if name.is_empty() {
        name.push('B');
    }
    format!("{name}{}", (idx + 1) % 10)
}

fn pdb_element(bead: &str) -> &'static str {
    if bead.starts_with('P') {
        "P"
    } else if bead.starts_with('N') {
        "N"
    } else {
        "C"
    }
}

fn angle_force(stat: &AngleStats) -> f64 {
    (1.0 / stat.std_deg.max(1.0).powi(2) * 10_000.0).clamp(1.0, 500.0)
}

fn dihedral_force(stat: &DihedralStats) -> f64 {
    (1.0 / stat.std_deg.max(1.0).powi(2) * 1_000.0).clamp(0.1, 100.0)
}

fn bonded_angle_parameters(
    stat: &AngleStats,
    tuning: Option<&OptimizationReport>,
) -> (f64, f64) {
    let mut angle = stat.mean_deg;
    let mut force = angle_force(stat);
    if let Some(tuning) = tuning {
        let angle_name = format!(
            "angle_{}_{}_{}_angle_deg",
            stat.bead_i, stat.bead_j, stat.bead_k
        );
        let force_name = format!(
            "angle_{}_{}_{}_force",
            stat.bead_i, stat.bead_j, stat.bead_k
        );
        for (name, value) in &tuning.best_parameters {
            if name == &angle_name {
                angle = *value;
            } else if name == &force_name {
                force = *value;
            }
        }
    }
    (angle, force)
}

fn bonded_dihedral_parameters(
    stat: &DihedralStats,
    tuning: Option<&OptimizationReport>,
) -> (f64, f64) {
    let mut phase = stat.mean_deg;
    let mut force = dihedral_force(stat);
    if let Some(tuning) = tuning {
        let phase_name = format!(
            "dihedral_{}_{}_{}_{}_phase_deg",
            stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
        );
        let force_name = format!(
            "dihedral_{}_{}_{}_{}_force",
            stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
        );
        for (name, value) in &tuning.best_parameters {
            if name == &phase_name {
                phase = *value;
            } else if name == &force_name {
                force = *value;
            }
        }
    }
    (phase, force)
}

fn bonded_pair_parameters(
    i: usize,
    j: usize,
    bond_stats: &[BondStats],
    tuning: Option<&OptimizationReport>,
) -> (f64, f64) {
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    let stat = bond_stats
        .iter()
        .find(|stat| stat.bead_i == a && stat.bead_j == b);
    let mut length_angstrom = stat.map(|stat| stat.mean).unwrap_or(4.7);
    let mut force = stat
        .map(|stat| (1.0 / stat.std.max(0.02).powi(2)).clamp(1.0, 5000.0))
        .unwrap_or(1250.0);

    if let Some(tuning) = tuning {
        for (name, value) in &tuning.best_parameters {
            if name == &format!("bond_{a}_{b}_length_angstrom") {
                length_angstrom = *value;
            } else if name == &format!("bond_{a}_{b}_force") {
                force = *value;
            }
        }
    }
    (length_angstrom / 10.0, force)
}

fn topology_name(name: &str) -> String {
    let mut out: String = name
        .chars()
        .filter_map(|ch| {
            if ch.is_ascii_alphanumeric() {
                Some(ch.to_ascii_uppercase())
            } else if ch == '_' || ch == '-' {
                Some('_')
            } else {
                None
            }
        })
        .take(12)
        .collect();
    if out.is_empty() {
        out.push_str("MOL");
    }
    out
}

fn render_martini_top(name: &str, itp_file: &str) -> String {
    let molecule = topology_name(name);
    let mut out = String::new();
    out.push_str("; Generated by warp-cg\n");
    out.push_str("; Include the Martini force-field file used by your simulation engine before this molecule include.\n");
    out.push_str(&format!("#include \"{itp_file}\"\n\n"));
    out.push_str("[ system ]\n");
    out.push_str(&format!("{molecule} coarse-grained system\n\n"));
    out.push_str("[ molecules ]\n");
    out.push_str("; molecule  count\n");
    out.push_str(&format!("{molecule:<16} 1\n"));
    out
}

fn resolve_output_path(out_dir: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        out_dir.join(candidate)
    }
}

fn normalized_trajectory_source(request: &CgRequest, path: &str) -> Option<TrajectorySource> {
    request.trajectory_source.clone().or_else(|| {
        request.trajectory.as_ref().map(|_| TrajectorySource {
            path: path.to_string(),
            topology: request.topology.clone(),
            format: None,
            topology_format: None,
            kind: default_external_trajectory_kind(),
            stride: None,
            start: None,
            stop: None,
            length_scale: None,
            target_selection: None,
            environment_selection: None,
            atom_indices: None,
            mass_weighted: None,
        })
    })
}

fn native_options(
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
        chunk_frames: None,
    }
}

fn xtb_run_config(request: &XtbRequest) -> XtbRunConfig {
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

fn validate_positive(value: Option<f64>, field: &str) -> Result<()> {
    if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err(anyhow!("{field} must be finite and greater than zero"));
    }
    Ok(())
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
mod tests {
    use super::*;

    #[test]
    fn example_request_validates() {
        let text = serde_json::to_string(&example_request()).unwrap();
        let (exit_code, value) = validate_request_json(&text);
        assert_eq!(exit_code, 0);
        assert_eq!(value["valid"], true);
    }

    #[test]
    fn benzene_mapping_has_three_beads() {
        let request = CgRequest {
            schema_version: AGENT_SCHEMA_VERSION.to_string(),
            name: "benzene".to_string(),
            smiles: "c1ccccc1".to_string(),
            trajectory: None,
            topology: None,
            trajectory_source: None,
            reference_source: None,
            parameter_tuning: None,
            output: CgOutputRequest {
                out_dir: tempfile::tempdir()
                    .unwrap()
                    .path()
                    .to_string_lossy()
                    .to_string(),
                mapped_trajectory: None,
                write_mapping_json: false,
                write_topology_itp: false,
                write_topology_top: false,
                write_cg_pdb: false,
                cg_pdb: None,
                write_bonded_parameter_map: false,
            },
        };
        let result = run_request(&request, Instant::now()).unwrap();
        assert_eq!(result.bead_count, 3);
        assert!(result.beads.iter().all(|bead| bead.name == "TC5"));
    }

    #[test]
    fn downstream_setup_artifacts_include_pdb_itp_top_and_parameter_map() {
        let tmp = tempfile::tempdir().unwrap();
        let request = CgRequest {
            schema_version: AGENT_SCHEMA_VERSION.to_string(),
            name: "benzene".to_string(),
            smiles: "c1ccccc1".to_string(),
            trajectory: None,
            topology: None,
            trajectory_source: None,
            reference_source: None,
            parameter_tuning: None,
            output: CgOutputRequest {
                out_dir: tmp.path().to_string_lossy().to_string(),
                mapped_trajectory: None,
                write_mapping_json: true,
                write_topology_itp: true,
                write_topology_top: true,
                write_cg_pdb: true,
                cg_pdb: None,
                write_bonded_parameter_map: true,
            },
        };

        let result = run_request(&request, Instant::now()).unwrap();
        let artifact_kinds: Vec<&str> = result
            .artifacts
            .iter()
            .map(|artifact| artifact.kind.as_str())
            .collect();

        assert!(artifact_kinds.contains(&"coarse_grained_pdb"));
        assert!(artifact_kinds.contains(&"martini_topology_itp"));
        assert!(artifact_kinds.contains(&"martini_topology_top"));
        assert!(artifact_kinds.contains(&"bonded_parameter_map_json"));
        assert!(tmp.path().join("benzene_cg.pdb").exists());
        assert!(tmp.path().join("benzene_martini.itp").exists());
        assert!(tmp.path().join("benzene_martini.top").exists());
        assert!(tmp.path().join("benzene_bonded_parameter_map.json").exists());
    }

    #[test]
    fn external_trajectory_tuning_requires_trajectory() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "parameter_tuning": {
                "enabled": true,
                "source": "external_trajectory",
                "method": "bayesian_optimization"
            }
        });
        let (exit_code, value) = validate_request_json(&request.to_string());
        assert_eq!(exit_code, 2);
        assert_eq!(value["valid"], false);
    }

    #[test]
    fn topology_top_requires_itp_output() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "output": {
                "write_topology_itp": false,
                "write_topology_top": true
            }
        });
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("write_topology_top requires"));
    }

    #[test]
    fn disabling_itp_without_top_field_remains_valid() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "output": {
                "write_topology_itp": false
            }
        });
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 0);
        assert_eq!(value["valid"], true);
    }

    #[test]
    fn xtb_gfn_must_not_be_empty() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "ethanol",
            "smiles": "CCO",
            "reference_source": {
                "kind": "xtb",
                "xtb": {
                    "gfn": "  "
                }
            }
        });
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("reference_source.xtb.gfn"));
    }

    #[test]
    fn xtb_tuning_requires_xtb_reference_source() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "ethanol",
            "smiles": "CCO",
            "parameter_tuning": {
                "enabled": true,
                "source": "xtb",
                "method": "pso"
            }
        });
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("xtb parameter tuning requires"));
    }

    #[test]
    fn target_selection_and_atom_indices_are_mutually_exclusive() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "trajectory_source": {
                "path": "traj.xtc",
                "topology": "topology.pdb",
                "target_selection": "resname BENZ",
                "atom_indices": [0, 1, 2, 3, 4, 5]
            }
        });
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("target_selection or atom_indices"));
    }

    #[test]
    fn tuning_counts_must_be_positive() {
        let base = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "trajectory": "traj.xtc"
        });
        for tuning in [
            json!({
                "enabled": true,
                "source": "external_trajectory",
                "method": "bayesian_optimization",
                "max_evaluations": 0
            }),
            json!({
                "enabled": true,
                "source": "external_trajectory",
                "method": "pso",
                "swarm_size": 0
            }),
        ] {
            let mut request = base.clone();
            request["parameter_tuning"] = tuning;
            let (exit_code, value) = validate_request_json(&request.to_string());

            assert_eq!(exit_code, 2);
            assert_eq!(value["valid"], false);
        }
    }

    #[test]
    fn trajectory_source_kind_must_be_external() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "trajectory_source": {
                "kind": "xtb",
                "path": "traj.xtc"
            }
        });
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("trajectory_source.kind"));
    }

    #[test]
    fn trajectory_source_string_fields_must_not_be_empty() {
        for field in ["topology", "target_selection", "environment_selection"] {
            let mut request = json!({
                "schema_version": AGENT_SCHEMA_VERSION,
                "name": "benzene",
                "smiles": "c1ccccc1",
                "trajectory_source": {
                    "kind": "external",
                    "path": "traj.xtc"
                }
            });
            request["trajectory_source"][field] = json!("  ");
            let (exit_code, value) = validate_request_json(&request.to_string());

            assert_eq!(exit_code, 2);
            assert!(value["error"]["message"]
                .as_str()
                .unwrap()
                .contains(field));
        }
    }

    #[test]
    fn top_level_topology_must_not_be_empty() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "topology": "  "
        });
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("topology must not be empty"));
    }

    #[test]
    fn output_paths_must_not_be_empty() {
        for (field, value) in [
            ("out_dir", json!("  ")),
            ("mapped_trajectory", json!("  ")),
        ] {
            let mut request = json!({
                "schema_version": AGENT_SCHEMA_VERSION,
                "name": "benzene",
                "smiles": "c1ccccc1",
                "trajectory": "traj.xtc",
                "output": {
                    "out_dir": "."
                }
            });
            request["output"][field] = value;
            let (exit_code, result) = validate_request_json(&request.to_string());

            assert_eq!(exit_code, 2);
            assert!(result["error"]["message"]
                .as_str()
                .unwrap()
                .contains(field));
        }
    }

    #[test]
    fn martini_itp_contains_atoms_and_bonds() {
        let mol = Molecule::from_smiles("c1ccccc1").unwrap();
        let mapping = map_molecule(&mol);
        let itp = render_martini_itp("benzene", &mapping, &[], &[], &[], None);

        assert!(itp.contains("[ moleculetype ]"));
        assert!(itp.contains("[ atoms ]"));
        assert!(itp.contains("[ bonds ]"));
        assert!(itp.contains("BENZENE"));
    }

    #[test]
    fn martini_itp_contains_angle_and_dihedral_sections() {
        let mapping = MappingResult {
            bead_names: vec![
                "C1".to_string(),
                "C1".to_string(),
                "C1".to_string(),
                "C1".to_string(),
            ],
            atom_groups: vec![vec![0], vec![1], vec![2], vec![3]],
            connections: vec![(0, 1), (1, 2), (2, 3)],
        };
        let itp = render_martini_itp(
            "chain",
            &mapping,
            &[],
            &[AngleStats {
                bead_i: 0,
                bead_j: 1,
                bead_k: 2,
                mean_deg: 120.0,
                std_deg: 5.0,
                samples: 4,
            }],
            &[DihedralStats {
                bead_i: 0,
                bead_j: 1,
                bead_k: 2,
                bead_l: 3,
                mean_deg: 180.0,
                std_deg: 10.0,
                samples: 4,
            }],
            None,
        );

        assert!(itp.contains("[ angles ]"));
        assert!(itp.contains("[ dihedrals ]"));
    }

    #[test]
    fn martini_top_includes_generated_itp_and_molecule_count() {
        let top = render_martini_top("benzene", "benzene_martini.itp");

        assert!(top.contains("#include \"benzene_martini.itp\""));
        assert!(top.contains("[ system ]"));
        assert!(top.contains("[ molecules ]"));
        assert!(top.contains("BENZENE"));
    }
}
