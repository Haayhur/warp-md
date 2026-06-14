use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::{SimulationArtifact, SimulationCommand, SimulationIssue, SIMULATE_SCHEMA_VERSION};

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SimulationStatus {
    pub schema_version: String,
    pub status: String,
    pub run_dir: String,
    pub detected_files: Vec<SimulationArtifact>,
    pub completed_stages: Vec<String>,
    pub failed_stages: Vec<String>,
    pub last_checkpoint: Option<String>,
    pub next_command: Option<SimulationCommand>,
    pub restart_capable: bool,
    pub warnings: Vec<SimulationIssue>,
}

pub fn status_json(run_dir: impl AsRef<Path>) -> (i32, Value) {
    match inspect_status(run_dir.as_ref()) {
        Ok(status) => {
            let exit_code = if status.status == "ok" { 0 } else { 2 };
            (
                exit_code,
                serde_json::to_value(status).unwrap_or_else(error_value),
            )
        }
        Err(err) => (
            2,
            json!({
                "schema_version": SIMULATE_SCHEMA_VERSION,
                "status": "error",
                "error": {"code": "E_SIMULATE_STATUS", "path": "", "message": err.to_string(), "severity": "error"}
            }),
        ),
    }
}

fn inspect_status(run_dir: &Path) -> Result<SimulationStatus> {
    if !run_dir.exists() {
        return Err(anyhow!("run_dir does not exist: {}", run_dir.display()));
    }
    if !run_dir.is_dir() {
        return Err(anyhow!("run_dir is not a directory: {}", run_dir.display()));
    }
    let mut files = Vec::new();
    collect_files(run_dir, run_dir, &mut files)?;
    files.sort();

    let mut detected_files = Vec::new();
    let mut checkpoints = Vec::new();
    let mut completed_stages = Vec::new();
    let mut failed_stages = Vec::new();
    for rel in files {
        let path = rel.to_string_lossy().replace('\\', "/");
        let role = role_for_path(&path);
        if role == "checkpoint" {
            checkpoints.push(path.clone());
        }
        if role == "log" {
            let full = run_dir.join(&path);
            if let Ok(text) = fs::read_to_string(&full) {
                let stem = Path::new(&path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                if text.contains("Finished mdrun") || text.contains("completed") {
                    completed_stages.push(stem.clone());
                }
                if text.contains("Fatal error")
                    || text.contains("Traceback")
                    || text.contains("ERROR")
                {
                    failed_stages.push(stem);
                }
            }
        }
        detected_files.push(SimulationArtifact {
            role,
            path,
            required: false,
            exists: Some(true),
        });
    }
    let last_checkpoint = checkpoints.last().cloned();
    Ok(SimulationStatus {
        schema_version: SIMULATE_SCHEMA_VERSION.to_string(),
        status: if failed_stages.is_empty() {
            "ok"
        } else {
            "error"
        }
        .to_string(),
        run_dir: run_dir.to_string_lossy().to_string(),
        detected_files,
        completed_stages,
        failed_stages,
        last_checkpoint: last_checkpoint.clone(),
        next_command: None,
        restart_capable: last_checkpoint.is_some(),
        warnings: Vec::new(),
    })
}

fn collect_files(root: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_files(root, &path, out)?;
        } else if let Ok(rel) = path.strip_prefix(root) {
            out.push(rel.to_path_buf());
        }
    }
    Ok(())
}

fn role_for_path(path: &str) -> String {
    match Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
    {
        "gro" | "pdb" | "cif" => "structure",
        "xtc" | "trr" | "dcd" => "trajectory",
        "cpt" | "chk" => "checkpoint",
        "edr" => "energy",
        "log" => "log",
        "tpr" => "run_input",
        "top" | "itp" | "xml" => "topology_or_parameter",
        "ndx" => "index",
        "json" => "manifest_or_metadata",
        _ => "artifact",
    }
    .to_string()
}

fn error_value(err: serde_json::Error) -> Value {
    json!({"schema_version": SIMULATE_SCHEMA_VERSION, "status": "error", "error": err.to_string()})
}
