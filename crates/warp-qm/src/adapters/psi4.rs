use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};

use crate::adapters::{AdapterRun, AdapterSummary};
use crate::contract::{
    JobManifest, QmArtifact, QmRequest, JOB_MANIFEST_VERSION, QM_SCHEMA_VERSION,
};
use crate::engines::find_executable;
use crate::parsers;

pub fn run(request: &QmRequest) -> AdapterRun {
    if request.task.kind != "generic_run" {
        return AdapterRun::error(
            "Psi4 adapter currently supports only task.kind generic_run with agent-supplied input",
            4,
        );
    }
    run_generic(request)
}

fn run_generic(request: &QmRequest) -> AdapterRun {
    let work_dir = PathBuf::from(&request.runtime.work_dir);
    if let Err(err) = fs::create_dir_all(&work_dir) {
        return AdapterRun::error(format!("failed to create Psi4 work dir: {err}"), 2);
    }

    let (input_path, output_path) = match prepare_generic_input(request, &work_dir) {
        Ok(value) => value,
        Err(err) => return AdapterRun::error(err, 2),
    };
    let mut artifacts = vec![artifact(&input_path, "text", "engine_input")];
    if let Ok(manifest_artifact) = write_generic_job_manifest(request, &work_dir, &artifacts) {
        artifacts.push(manifest_artifact);
    }

    let executable = match resolve_psi4_executable(request) {
        Some(path) => path,
        None => {
            return AdapterRun {
                status: "error".into(),
                exit_code: 4,
                command: None,
                artifacts,
                warnings: vec![
                    "Psi4 executable not found; set engine.executable, WARP_QM_PSI4, PSI4_BINARY, or PATH"
                        .into(),
                ],
                properties: serde_json::Map::new(),
                summary: AdapterSummary::default(),
            }
        }
    };

    let input_name = match input_path.file_name() {
        Some(name) => name.to_string_lossy().into_owned(),
        None => return AdapterRun::error("generic Psi4 input path has no file name", 2),
    };
    let output_name = match output_path.file_name() {
        Some(name) => name.to_string_lossy().into_owned(),
        None => return AdapterRun::error("generic Psi4 output path has no file name", 2),
    };
    let command_vec = vec![executable.clone(), input_name.clone(), output_name.clone()];
    let output = Command::new(&executable)
        .arg(&input_name)
        .arg(&output_name)
        .current_dir(&work_dir)
        .output();
    let output = match output {
        Ok(output) => output,
        Err(err) => {
            return AdapterRun {
                status: "error".into(),
                exit_code: 4,
                command: Some(command_vec),
                artifacts,
                warnings: vec![format!("failed to execute Psi4: {err}")],
                properties: serde_json::Map::new(),
                summary: AdapterSummary::default(),
            }
        }
    };

    if !output.stdout.is_empty() {
        let stdout_path = work_dir.join("psi4.stdout");
        let _ = fs::write(&stdout_path, &output.stdout);
        artifacts.push(artifact(&stdout_path, "text", "engine_stdout"));
        if !output_path.exists() {
            let _ = fs::write(&output_path, &output.stdout);
        }
    }
    let stderr_path = work_dir.join("psi4.err");
    let _ = fs::write(&stderr_path, &output.stderr);
    if stderr_path
        .metadata()
        .map(|metadata| metadata.len() > 0)
        .unwrap_or(false)
    {
        artifacts.push(artifact(&stderr_path, "text", "engine_stderr"));
    }
    if output_path.exists() {
        artifacts.push(artifact(&output_path, "text", "engine_output"));
    }
    collect_expected_outputs(request, &work_dir, &mut artifacts);
    let missing_outputs = missing_expected_outputs(request, &work_dir);
    let mut warnings = Vec::new();
    if !missing_outputs.is_empty() {
        warnings.push(format!(
            "Psi4 completed but expected output artifacts are missing: {}",
            missing_outputs.join(", ")
        ));
    }

    let inspection = parsers::inspect_output(&output_path, "psi4").ok();
    let mut properties = serde_json::Map::new();
    if let Some(report) = inspection.as_ref() {
        properties.insert("inspection".into(), json!(report));
    }
    let converged = inspection
        .as_ref()
        .map(|report| report.fatal_errors.is_empty() && report.convergence_status.is_some());
    let success =
        output.status.success() && missing_outputs.is_empty() && converged.unwrap_or(false);

    AdapterRun {
        status: if success { "ok" } else { "error" }.into(),
        exit_code: if success { 0 } else { 4 },
        command: Some(command_vec),
        artifacts,
        warnings,
        properties,
        summary: AdapterSummary {
            energy_hartree: inspection.and_then(|report| report.final_energy_hartree),
            converged,
            n_atoms: None,
        },
    }
}

fn prepare_generic_input(
    request: &QmRequest,
    work_dir: &Path,
) -> Result<(PathBuf, PathBuf), String> {
    let basename = request
        .engine
        .settings
        .get("basename")
        .and_then(Value::as_str)
        .unwrap_or("job");
    let input_name = request
        .engine
        .settings
        .get("input_name")
        .and_then(Value::as_str)
        .and_then(|name| Path::new(name).file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| format!("{basename}.dat"));
    let input_path = work_dir.join(&input_name);
    if let Some(text) = request
        .engine
        .settings
        .get("input_text")
        .and_then(Value::as_str)
    {
        fs::write(&input_path, text)
            .map_err(|err| format!("failed to write generic Psi4 input: {err}"))?;
    } else if let Some(path) = request
        .engine
        .settings
        .get("input_file")
        .and_then(Value::as_str)
    {
        let source = Path::new(path);
        if !source.exists() {
            return Err(format!("generic Psi4 input file not found: {path}"));
        }
        if source != input_path {
            fs::copy(source, &input_path)
                .map_err(|err| format!("failed to copy generic Psi4 input: {err}"))?;
        }
    } else {
        return Err(
            "generic_run requires engine.settings.input_file or engine.settings.input_text".into(),
        );
    }
    let output_name = request
        .engine
        .settings
        .get("output_name")
        .and_then(Value::as_str)
        .and_then(|name| Path::new(name).file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| {
            input_path
                .file_stem()
                .map(|stem| format!("{}.out", stem.to_string_lossy()))
                .unwrap_or_else(|| format!("{basename}.out"))
        });
    Ok((input_path, work_dir.join(output_name)))
}

fn resolve_psi4_executable(request: &QmRequest) -> Option<String> {
    request
        .engine
        .executable
        .clone()
        .or_else(|| std::env::var("WARP_QM_PSI4").ok())
        .or_else(|| std::env::var("PSI4_BINARY").ok())
        .or_else(|| find_executable("psi4"))
}

fn write_generic_job_manifest(
    request: &QmRequest,
    work_dir: &Path,
    artifacts: &[QmArtifact],
) -> Result<QmArtifact, String> {
    let manifest = JobManifest {
        schema_version: JOB_MANIFEST_VERSION.into(),
        request_id: request.request_id.clone(),
        engine: request.engine.clone(),
        task: request.task.clone(),
        molecule: request.molecule.clone(),
        artifacts: artifacts.to_vec(),
        provenance: BTreeMap::from([
            ("tool".into(), json!("warp-qm")),
            ("schema_version".into(), json!(QM_SCHEMA_VERSION)),
            ("input_mode".into(), json!("generic_engine_input")),
            (
                "psi4_executable".into(),
                json!(resolve_psi4_executable(request)),
            ),
        ]),
    };
    let path = work_dir.join("job_manifest.json");
    fs::write(
        &path,
        serde_json::to_string_pretty(&manifest).map_err(|err| err.to_string())?,
    )
    .map_err(|err| err.to_string())?;
    Ok(artifact(&path, "json", "job_manifest"))
}

fn collect_expected_outputs(request: &QmRequest, work_dir: &Path, artifacts: &mut Vec<QmArtifact>) {
    for output in configured_expected_outputs(request) {
        let path = work_dir.join(&output);
        if path.exists() {
            artifacts.push(artifact(&path, output_format(&output), "engine_artifact"));
        }
    }
}

fn missing_expected_outputs(request: &QmRequest, work_dir: &Path) -> Vec<String> {
    configured_expected_outputs(request)
        .into_iter()
        .filter(|output| !work_dir.join(output).exists())
        .collect()
}

fn configured_expected_outputs(request: &QmRequest) -> Vec<String> {
    request
        .engine
        .settings
        .get("expected_outputs")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn output_format(path: &str) -> &'static str {
    if path.ends_with(".json") {
        "json"
    } else if path.ends_with(".cube") || path.ends_with(".cub") {
        "cube"
    } else if path.ends_with(".npy") || path.ends_with(".npz") {
        "numpy"
    } else {
        "text"
    }
}

fn artifact(path: &Path, format: &str, kind: &str) -> QmArtifact {
    QmArtifact {
        path: path.to_string_lossy().into_owned(),
        format: format.into(),
        kind: kind.into(),
    }
}
